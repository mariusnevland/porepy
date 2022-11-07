from __future__ import annotations

import porepy as pp
import numpy as np
import sympy as sym

from dataclasses import dataclass
from typing import Union
from time import time


@dataclass
class StoreSolution:
    """Data class to store variables of interest."""

    def __init__(self, model: "BiotWithRollers"):
        """Data class constructor.

        Args:
            model : BioWithRoller model class.

        """
        sd = model.mdg.subdomains()[0]
        data = model.mdg.subdomain_data(sd)
        p_var = model.scalar_variable
        u_var = model.displacement_variable
        t = model.time_manager.time
        exact = ExactSolution(model)

        # Time variables
        self.time = t

        # Spatial variables
        self.xc = sd.cell_centers[0]
        self.yc = sd.cell_centers[1]

        # Pressure variables
        self.numerical_pressure = data[pp.STATE][p_var]
        self.exact_pressure = model.eval_scalar(sd, exact.p, t, is_cc=True)

        # Displacement variables
        self.numerical_displacement = data[pp.STATE][u_var]
        self.exact_displacement = model.eval_vector(sd, exact.u, t, is_cc=True)

@dataclass
class ExactSolution:
    """Data class containing the exact solutions to the manufactured Biot's problem.

    The exact solutions are takem from https://epubs.siam.org/doi/pdf/10.1137/15M1014280
    and are given by:

        u(x, y, t) = t * [x * (a - x) * sin(2 * pi * y), sin(2 * pi * x) * sin(2 * pi * y)],
        p(x, y, t) = t * x * (a - x) * sin(2 * pi * y),

    where `u` and `p` are respectively the displacement and fluid pressure, and `a` is the
    width of the domain.

    Analytical sources can therefore be obtained by computing the relevant expressions. This
    class computes such sources automatically for a given exact pair (u, p) and a set of
    physical parameters.

    Notation:

        Variables:
            u : displacement
            p : pressure
            t : time
            q : darcy flux
            sigma_tot : total stress
            sigma_eff : effective stress
            eps : strain
            f_flow : flow source term
            f_mech : mechanics source term

        Operators:
            div_x : divergence of x
            grad_x : gradient of x
            dt_x : partial derivative of x w.r.t. time
            trans_x : transpose of x

    """

    def __init__(self, model: "BiotWithRollers"):
        """Data class constructor"""

        # Declare symbolic variables
        x, y, t = sym.symbols("x y t")

        # Retrieve parameters from the model
        lx, _ = model.params["domain_size"]  # [m]
        lmbda_s = model.params["lambda_lame"]  # [Pa]
        mu_s = model.params["mu_lame"]  # [Pa]
        alpha_biot = model.params["alpha_biot"]  # [-]
        S_phi = model.params["storativity"]  # [Pa^-1]
        mu_f = model.params["viscosity"]  # [Pa * s]
        k = model.params["permeability"]  # [m^2]

        # Pressure
        self.p = t * x * (lx - x) * sym.sin(2 * sym.pi * y)

        # Displacement
        self.u = [
            t * x * (lx - x) * sym.sin(2 * sym.pi * y),
            t * sym.sin(2 * sym.pi * x) * sym.sin(2 * sym.pi * y),
        ]

        # Pressure gradient
        self.grad_p = [sym.diff(self.p, x), sym.diff(self.p, y)]

        # Darcy flux
        self.q = [-(k / mu_f) * self.grad_p[0], -(k / mu_f) * self.grad_p[1]]

        # Divergence of Darcy flux
        self.div_q = sym.diff(self.q[0], x) + sym.diff(self.q[1], y)

        # Divergence of displacement
        self.div_u = sym.diff(self.u[0], x) + sym.diff(self.u[1], y)

        # Time derivative of pressure
        self.dt_p = sym.diff(self.p, t)

        # Time derivative of divergence of the displacement
        self.dt_div_u = sym.diff(self.div_u, t)

        # Flow source
        self.f_flow = S_phi * self.dt_p + alpha_biot * self.dt_div_u + self.div_q

        # Gradient of the displacement
        self.grad_u = [
            [sym.diff(self.u[0], x), sym.diff(self.u[0], y)],
            [sym.diff(self.u[1], x), sym.diff(self.u[1], y)],
        ]

        # Transpose of the gradient of the displacement
        self.trans_grad_u = [
            [self.grad_u[0][0], self.grad_u[1][0]],
            [self.grad_u[0][1], self.grad_u[1][1]],
        ]

        # Strain
        self.eps = [
            [
                0.5 * (self.grad_u[0][0] + self.trans_grad_u[0][0]),
                0.5 * (self.grad_u[0][1] + self.trans_grad_u[0][1]),
            ],
            [
                0.5 * (self.grad_u[1][0] + self.trans_grad_u[1][0]),
                0.5 * (self.grad_u[1][1] + self.trans_grad_u[1][1]),
            ],
        ]

        # Effective stress
        self.sigma_eff = [
            [
                lmbda_s * (self.eps[0][0] + self.eps[1][1]) + 2 * mu_s * self.eps[0][0],
                2 * mu_s * self.eps[0][1],
            ],
            [
                2 * mu_s * self.eps[1][0],
                lmbda_s * (self.eps[0][0] + self.eps[1][1]) + 2 * mu_s * self.eps[1][1],
            ],
        ]

        # Total stress
        self.sigma_tot = [
            [
                self.sigma_eff[0][0] - alpha_biot * self.p,
                self.sigma_eff[0][1],
            ],
            [self.sigma_eff[1][0], self.sigma_eff[1][1] - alpha_biot * self.p],
        ]

        # Divergence of the total stress
        self.div_sigma_tot = [
            sym.diff(self.sigma_tot[0][0], x) + sym.diff(self.sigma_tot[1][0], y),
            sym.diff(self.sigma_tot[0][1], x) + sym.diff(self.sigma_tot[1][1], y),
        ]

        # Mechanics source term
        self.f_mech = self.div_sigma_tot


class BiotWithRollers(pp.ContactMechanicsBiot):
    """Class for setting up a manufactured solution for the Biot without fractures"""

    def __init__(self, params: dict):
        """
        Constructor for the ManufacturedBiot class.

        Args:
            params: Model parameters.

        """
        super().__init__(params)

        # ad sanity check
        if not self.params["use_ad"]:
            raise ValueError("Model only valid when ad is used.")

        # Create a solution list to store variables
        self.solutions: list[StoreSolution] = []

    def create_grid(self) -> None:
        """Create Cartesian structured mixed-dimensional grid."""
        if self.params["mesh_type"] == "cartesian":
            nx, ny = self.params["num_cells"]
            lx, ly = self.params["domain_size"]
            phys_dims = np.array([lx, ly])
            n_cells = np.array([nx, ny])
            self.box = pp.geometry.bounding_box.from_points(
                np.array([[0, 0], phys_dims]).T
            )
            sd = pp.CartGrid(n_cells, phys_dims)
            # Perturb nodes
            np.random.seed(42)
            perturbation_factor = self.params["perturbation_factor"]
            perturbation = np.random.rand(sd.num_nodes) * perturbation_factor
            sd.nodes[0] += perturbation
            sd.nodes[1] += perturbation
            sd.compute_geometry()
            self.mdg = pp.meshing.subdomains_to_mdg([[sd]])
        elif self.params["mesh_type"] == "triangular":
            pass
        else:
            raise ValueError("Unsupported mesh type.")

    def _bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        super()._bc_type_mechanics(sd)

        # Get boundary sides, retrieve data dict, and bc object
        _, east, west, north, south, *_ = self._domain_boundary_sides(sd)
        data = self.mdg.subdomain_data(sd)
        bc = data[pp.PARAMETERS][self.mechanics_parameter_key]["bc"]

        # East side: Roller
        bc.is_neu[1, east] = True
        bc.is_dir[1, east] = False

        # West side: Roller
        bc.is_neu[1, west] = True
        bc.is_dir[1, west] = False

        # South side: Roller
        bc.is_neu[0, south] = True
        bc.is_dir[0, south] = False

        return bc

    # def _bc_values_mechanics(self, sd: pp.Grid) -> np.ndarray:
    #     """Set boundary condition values for the mechanics subproblem.
    #
    #     Args:
    #         sd: Subdomain grid.
    #
    #     Returns:
    #         Vectorial boundary condition values of shape (sd.dim * sd.num_faces, ).
    #
    #     """
    #     super()._bc_values_mechanics(sd)
    #
    #     # # Retrieve boundary sides
    #     # _, _, _, north, *_ = self._domain_boundary_sides(sd)
    #     #
    #     # # All zeros except vertical component of the north side
    #     # vertical_load = self.params["vertical_load"]
    #     # bc_values = np.array([np.zeros(sd.num_faces), np.zeros(sd.num_faces)])
    #     # bc_values[1, north] = -vertical_load * sd.face_areas[north]
    #     # bc_values = bc_values.ravel("F")
    #
    #     #return bc_values


    def before_newton_loop(self) -> None:
        """Method to be called before entering the newton loop"""
        super().before_newton_loop()

        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        time = self.time_manager.time
        dt = self.time_manager.dt

        # Retrieve exact solution
        exact = ExactSolution(self)

        # Update flow source term
        f_flow_sym = exact.f_flow
        f_flow_cc = self.eval_scalar(sd, f_flow_sym, time, is_cc=True)
        integrated_f_flow = f_flow_cc * sd.cell_volumes * dt
        # we have to explicitly multiply by dt to be in agreement with the discretization
        data[pp.PARAMETERS][self.scalar_parameter_key]["source"] = integrated_f_flow

        # Update mechanics source term
        f_mech_sym = exact.f_mech
        f_mech_cc = np.asarray(self.eval_vector(sd, f_mech_sym, time, is_cc=True)).ravel("F")
        integrated_f_mech = f_mech_cc * sd.cell_volumes.repeat(sd.dim)
        data[pp.PARAMETERS][self.mechanics_parameter_key]["source"] = integrated_f_mech

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Method to be called after the Netwon solver has converged."""
        super().after_newton_convergence(solution, errors, iteration_counter)

        # Store solutions
        schedule = self.time_manager.schedule
        if any([np.isclose(self.time_manager.time, t_sch) for t_sch in schedule]):
            self.solutions.append(StoreSolution(self))

    # -----> Helper methods
    @staticmethod
    def eval_scalar(
        sd: pp.Grid,
        sym_exp: object,
        time: Union[int, float],
        is_cc: bool
    ) -> np.ndarray:
        """
        Evaluate a symbolic scalar expression at the cell centers for a given time.

        Args:
            sd: PorePy grid.
            sym_exp: Symbolic expression dependent on x, y, and t.
            time: Time in seconds.
            is_cc: True for a cell-center quantity. False for a face-center quantity.

        Returns:
            Evaluated expression either at the cell centers or at the face centers.

        """

        if is_cc:
            s = sd.cell_centers
        else:
            s = sd.face_centers

        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Lambdify expression
        fun = sym.lambdify((x, y, t), sym_exp, "numpy")

        # Evaluate at the cell centers
        eval_exp = fun(s[0], s[1], time)

        return eval_exp

    @staticmethod
    def eval_vector(
        sd: pp.Grid, sym_exp: list[object], time: Union[int, float], is_cc: bool
    ) -> list[np.ndarray]:
        """
        Evaluate a symbolic scalar expression at the cell centers for a given time.

        Args:
            sd: PorePy grid.
            sym_exp: Symbolic expression dependent on x, y, and t.
            time: Time in seconds.
            is_cc: True for a cell-center quantity. False for a face-center quantity.

        Returns:
            Evaluated expression either at the cell centers or at the face centers.

        """
        if is_cc:
            s = sd.cell_centers
        else:
            s = sd.face_centers

        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Lambdify expression
        fun_x = sym.lambdify((x, y, t), sym_exp[0], "numpy")
        fun_y = sym.lambdify((x, y, t), sym_exp[1], "numpy")

        # Evaluate at the cell centers
        eval_exp_x = fun_x(s[0], s[1], time)
        eval_exp_y = fun_y(s[0], s[1], time)
        eval_exp = [eval_exp_x, eval_exp_y]

        return eval_exp

    @staticmethod
    def eval_tensor(
        sd: pp.Grid, sym_exp: list[list[object]], time: Union[int, float], is_cc: bool
    ) -> list[list[np.ndarray]]:
        """
        Evaluate a symbolic tensor expression at the cell centers for a given time.

        Args:
            sd: PorePy grid.
            sym_exp: Symbolic expression dependent on x, y, and t.
            time: Time in seconds.
            is_cc: True for a cell-center quantity. False for a face-center quantity.


        Returns:
            Evaluated expression either at the cell centers or at the face centers.


        """

        if is_cc:
            s = sd.cell_centers
        else:
            s = sd.face_centers

        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Lambdify expression
        fun_xx = sym.lambdify((x, y, t), sym_exp[0][0], "numpy")
        fun_xy = sym.lambdify((x, y, t), sym_exp[0][1], "numpy")
        fun_yx = sym.lambdify((x, y, t), sym_exp[1][0], "numpy")
        fun_yy = sym.lambdify((x, y, t), sym_exp[1][1], "numpy")

        # Evaluate at the cell centers
        eval_exp_xx = fun_xx(s[0], s[1], time)
        eval_exp_xy = fun_xy(s[0], s[1], time)
        eval_exp_yx = fun_yx(s[0], s[1], time)
        eval_exp_yy = fun_yy(s[0], s[1], time)
        eval_exp = [[eval_exp_xx, eval_exp_xy], [eval_exp_yx, eval_exp_yy]]

        return eval_exp

    def exact_traction(self):
        sd = self.mdg.subdomains()[0]
        normals = sd.face_normals
        nx = normals[0]
        ny = normals[1]

        exact = ExactSolution(self)
        sigma_tot = exact.sigma_tot
        sigma_fc = self.eval_tensor(sd, sigma_tot, self.time_manager.time, False)

        T_x = sigma_fc[0][0] * nx + sigma_fc[0][1] * ny
        T_y = sigma_fc[1][0] * nx + sigma_fc[1][1] * ny

        T = np.asarray([T_x, T_y]).ravel("F")

        return T

#%% Runner
params = {
    "use_ad": True,
    "num_cells": (20, 20),
    "domain_size": (1, 1),  # [m]
    "lambda_lame": 1,  # [Pa]
    "mu_lame": 1,  # [Pa]
    "storativity": 1,  # [Pa^-1]
    "alpha_biot": 1,  # [-]
    "viscosity": 1,  # [Pa * s]
    "permeability": 1,  # [m^2]
    "mesh_type": "cartesian",
    "time_manager": pp.TimeManager([0, 1], 1, constant_dt=True),
    "perturbation_factor": 1e-2,
}
# Run model
tic = time()
print("Simulation started...")
model = BiotWithRollers(params=params)
pp.run_time_dependent_model(model, params)
print(f"Simulation finished in {round(time() - tic)} sec.")

# Plot
sd = model.mdg.subdomains()[0]
data = model.mdg.subdomain_data(sd)
p_var = model.scalar_variable
u_var = model.displacement_variable
p_ex = model.solutions[0].exact_pressure
p_num = model.solutions[0].numerical_pressure

pp.plot_grid(sd, p_ex, plot_2d=True, title="Exact pressure")
pp.plot_grid(sd, p_num, plot_2d=True, title="Numerical pressure")

# %% Experimental
exact = ExactSolution(model)
model.eval_tensor(sd, exact.sigma_tot, model.time_manager.time, False)

