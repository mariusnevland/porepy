"""
This module contains an implementation of Mandel's problem of poroelasticity. The problem is
discretized using MPFA/MPSA-FV in space and backward Euler in time.

For futher details on Mandel's problem, see [1], [2], [3], [4]. For the implementational
details, see [5].

Note:

    References:

    - [1] Mandel, J.: Consolidation des sols (étude mathématique). Geotechnique.
      3(7), 287–299 (1953).

    - [2] Cheng, A.H.-D., Detournay, E.: A direct boundary element method for plane strain
      poroelasticity. Int. J. Numer. Anal. Methods Geomech. 12(5), 551–572 (1988).

    - [3] Abousleiman, Y., Cheng, A. D., Cui, L., Detournay, E., & Roegiers, J. C. (1996).
      Mandel's problem revisited. Geotechnique, 46(2), 187-195.

    - [4] Mikelić, A., Wang, B., & Wheeler, M. F. (2014). Numerical convergence study of
      iterative coupling for coupled flow and geomechanics. Computational Geosciences,
      18(3), 325-341.

    - [5] Keilegavlen, E., Berge, R., Fumagalli, A. et al. PorePy: an open-source software for
      simulation of multiphysics processes in fractured porous media. Comput Geosci 25,
      243–265 (2021).

"""
from __future__ import annotations


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import porepy as pp
import os
import scipy.optimize as opt

from typing import Literal, Union
from dataclasses import dataclass


@dataclass
class StoreSolution:
    """Data class to store variables of interest for Mandel's problem"""

    def __init__(self, app: "Mandel"):
        """Data class constructor.

        Args:
            app : Mandel's model application.

        """
        sd = app.mdg.subdomains()[0]
        data = app.mdg.subdomain_data(sd)
        p_var = app.scalar_variable
        u_var = app.displacement_variable
        t = app.time_manager.time

        # Time variables
        self.time = t

        # Flow variables
        self.numerical_pressure = data[pp.STATE][p_var]
        self.exact_pressure = None
        self.numerical_flux = None
        self.exact_flux = None

        # Mechanics variables
        self.numerical_displacement = data[pp.STATE][u_var]
        self.exact_displacement = None
        self.numerical_traction = None
        self.exact_traction = None

        # Error variables
        self.pressure_error = app.l2_relative_error(
            sd=sd,
            true_array=self.exact_pressure,
            approx_array=self.numerical_pressure,
            is_scalar=True,
            is_cc=True
        )
        self.flux_error = app.l2_relative_error(
            sd=sd,
            true_array=self.exact_flux,
            approx_array=self.numerical_flux,
            is_scalar=True,
            is_cc=False,
        )
        self.displacement_error = app.l2_relative_error(
            sd=sd,
            true_array=self.exact_displacement,
            approx_array=self.numerical_displacement,
            is_scalar=False,
            is_cc=True,
        )
        self.traction_error = app.l2_relative_error(
            sd=sd,
            true_array=self.exact_traction,
            approx_array=self.numerical_traction,
            is_scalar=False,
            is_cc=False,
        )


class Mandel(pp.ContactMechanicsBiot):
    """Parent class for Mandel's problem.

    Examples:

        # Instantiate time step control object
        time_manager = pp.TimeSteppingControl(
            schedule=[
                0,
                10,
                50,
                100,
                1000,
                5000,
                8000,
                10000,
                20000,
                30000,
                50000,
            ],  # [s]
            dt_init=10,  # [s]
            constant_dt=True
        )
        # Create model's parameter dictionary
        model_params = {
            "use_ad": True,  # Only "use_ad: True" is supported for this model
            "mu_lame": 2.475e9,  # [Pa]
            "lambda_lame": 1.650e9,  # [Pa]
            "permeability": 9.869e-14,  # [m^2]
            "alpha_biot": 1.0,  # [-]
            "viscosity": 1e-3,  # [Pa.s]
            "storativity": 6.0606e-11,  # [1/Pa]
            "applied_load": 6e8,  # [N/m]
            "height": 10.0,  # [m]
            "width": 100.0,  # [m]
            "mesh_size": 2.0,  # [m]
            "time_manager": time_manager,
            "plot_results": True,
        }
        # Run model
        model = Mandel(model_params)
        pp.run_time_dependent_model(model, model_params)

    """

    def __init__(self, params: dict):
        """Constructor of the Mandel class.

        Args:
            params: Dictionary containing mandatory and optional model parameters.

            Default physical parameters were adapted from
            https://link.springer.com/article/10.1007/s10596-013-9393-8.

            Optional parameters are:

            - 'alpha_biot' : Biot constant (int or float). Default is 1.0.
            - 'domain_size' : Size of the domain (tuple of int or float). The first element
              of the tuple is the width and the second the height. Default is (100.0, 10.0).
            - 'lambda_lame' : Lamé parameter in `Pa` (int or float). Default is 1.65e9.
            - 'mesh_size' : Mesh size in `m` (int or float). Used only when
              `mesh_size = "triangular"`. Default is 2.0.
            - 'mesh_type' : Type of mesh (str). Either "cartesian" or "triangular". The
              first is a perturbed Cartesian grid and the second an unstructured triangular
              mesh. Default is "cartesian".
            - 'mu_lame' : Lamé parameter in `m` (int or float). Default is 1.475E9.
            - 'number_of_roots' : Number of roots to approximate the exact solutions (int).
              Default is 200.
            - 'num_cells' : Number of cells in horizontal and vertical directions
              (tuple of int). This parameter is used only when `mesh_type = "cartesian"`.
              Default is (50, 5).
            - 'permeability' : Permeability in `m^2` (int or float). Default is 9.86e-14.
            - 'pertubation_factor' : Perturbation factor (int or float). Used for
              perturbing the physical nodes of the mesh. This is necessary to avoid
              singular matrices with MPSA and the use of rollers on Cartesian grids.
              Default is 1e-6.
            - 'plot_results' : Whether to plot the results (bool). The resulting plot is
              saved inside the `out` folder. Default is False.
            - 'storativity' : Storativity in `Pa^-1` (int or float). Default is 6.0606e-11.
            - 'time_manager' : Time manager object (pp.TimeManager). Default is
              pp.TimeManager([0, 20, 100, 400, 1200, 3000], 10, constant_dt=True).
            - 'use_ad' : Whether to use ad (bool). Must be set to True. Otherwise,
              an error will be raised. Default is True.
            - 'vertical_load' : Applied vertical load in `N * m^-1` (int or float).
              Default is 6e8.
            - 'viscosity' : Fluid viscosity in `Pa * s` (int or float). Default is 1e-3.

        """
        def set_default_params(keyword: str, value: object) -> None:
            """
            Set default parameters if a keyword is absent in the `params` dictionary.

            Args:
                keyword: Parameter keyword, e.g., "alpha_biot".
                value: Value of `keyword`, e.g., 1.0.

            """
            if keyword not in params.keys():
                params[keyword] = value

        # Default parameters
        default_tm = pp.TimeManager([0, 20, 100, 400, 1200, 3000], 10, constant_dt=True)
        default_params: list[tuple] = [
            ("alpha_biot", 1.0),  # [-]
            ("domain_size", (10.0, 100.0)),  # [m]
            ("height", 1.0),  # [m]
            ("lambda_lame", 1.65e9),  # [Pa]
            ("mesh_type", "cartesian"),
            ("mesh_size", 2.0),  # [m]
            ("mu_lame", 1.475e9),  # [Pa]
            ("number_of_roots", 200),
            ("num_cells", (50, 5)),
            ("permeability", 9.86e-14),  # [m^2]
            ("perturbation_factor", 1e-6),
            ("plot_results", False),
            ("specific_weight", 9.943e3),  # [Pa * m^-1]
            ("time_manager", default_tm),  # all time-related variables must be in [s]
            ("use_ad", True),  # only `use_ad = True` is supported
            ("vertical_load", 6e8),  # [N * m^-1]
            ("viscosity", 1e-3),  # [Pa * s]
        ]

        # Set default values
        for key, val in default_params:
            set_default_params(key, val)
        super().__init__(params)

        # ad sanity check
        if not self.params["use_ad"]:
            raise ValueError("Model only valid when ad is used.")

        # Create a solution list to store variables
        self.solutions: list[StoreSolution] = []

    def create_grid(self) -> None:
        """Create a two-dimensional Cartesian grid."""
        if self.params["mesh_type"] == "cartesian":
            nx, ny = self.params["num_cells"]
            lx, ly = self.params["domain_size"]
            phys_dims = np.array([lx, ly])
            n_cells = np.array([nx, ny])
            self.box = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
            sd: pp.Grid = pp.CartGrid(n_cells, phys_dims)
            sd.compute_geometry()
            np.random.seed(43)  # this seed is fixed but completely arbitrary
            # Perturb nodes to avoid singular matrices with rollers and MPSA.
            perturbation_factor = self.params["perturbation_factor"]
            perturbation = np.random.rand(sd.num_nodes) * perturbation_factor
            sd.nodes[0] += perturbation
            sd.nodes[1] += perturbation
            sd.compute_geometry()
            self.mdg = pp.meshing.subdomains_to_mdg([[sd]])
        elif self.params["mesh_type"] == "triangular":
            lx, ly = self.params["domain_size"]
            mesh_size = self.params["mesh_size"]
            self.box = {"xmin": 0.0, "xmax": lx, "ymin": 0.0, "ymax": ly}
            network_2d = pp.FractureNetwork2d(None, None, self.box)
            mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}
            self.mdg = network_2d.mesh(mesh_args)
        else:
            raise NotImplementedError("Mesh type not supported.")

    def _initial_condition(self) -> None:
        """Set up initial conditions.

        Note:
            Initial conditions are given by Eqs. (41) - (43) from 10.1007/s10596-013-9393-8.

        """
        super()._initial_condition()
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)

        # Set initial pressure
        data[pp.STATE][self.scalar_variable] = self.exact_pressure(0)
        data[pp.STATE][pp.ITERATE][self.scalar_variable] = self.exact_pressure(0)

        # Set initial displacement
        data[pp.STATE][self.displacement_variable] = self.exact_displacement(0)
        data[pp.STATE][pp.ITERATE][
            self.displacement_variable
        ] = self.exact_displacement(0)

        # Store initial pressure and displacement distributions in the `sol` dictionary
        self.sol[0]["p_num"] = self.exact_pressure(0)
        self.sol[0]["p_ex"] = self.exact_pressure(0)
        self.sol[0]["u_num"] = self.exact_displacement(0)
        self.sol[0]["u_ex"] = self.exact_displacement(0)
        self.sol[0]["Q_num"] = self.exact_flux(0)
        self.sol[0]["Q_ex"] = self.exact_flux(0)
        self.sol[0]["T_num"] = self.exact_traction(0)
        self.sol[0]["T_ex"] = self.exact_traction(0)
        self.sol[0]["U_num"] = 0
        self.sol[0]["U_ex"] = 0
        self.sol[0]["error_pressure"] = np.nan
        self.sol[0]["error_displacement"] = np.nan
        self.sol[0]["error_flux"] = np.nan
        self.sol[0]["error_traction"] = np.nan

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Define boundary condition types for the flow subproblem.

        Args:
            sd: Subdomain grid.

        Returns:
            bc: Scalar boundary condition representation.

        """

        # Define boundary regions
        tol = self.params["perturbation_factor"] * 10
        all_bc, east, *_ = self._domain_boundary_sides(sd, tol=tol)
        east_bc = np.isin(all_bc, np.where(east)).nonzero()

        # All sides Neumann, except the East side which is Dirichlet
        bc_type = np.asarray(all_bc.size * ["neu"])
        bc_type[east_bc] = "dir"

        bc = pp.BoundaryCondition(sd, faces=all_bc, cond=bc_type)

        return bc

    def _bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define boundary condition types for the mechanics subproblem

        Args:
            sd: Subdomain grid.

        Returns:
            bc: Vectorial boundary condition representation.

        """

        # Inherit bc from parent class. This sets all bc faces as Dirichlet.
        super()._bc_type_mechanics(sd=sd)

        # Get boundary sides, retrieve data dict, and bc object
        tol = self.params["perturbation_factor"] * 10
        _, east, west, north, south, *_ = self._domain_boundary_sides(sd, tol=tol)
        data = self.mdg.subdomain_data(sd)
        bc = data[pp.PARAMETERS][self.mechanics_parameter_key]["bc"]

        # East side: Stress-free
        bc.is_neu[:, east] = True
        bc.is_dir[:, east] = False

        # West side: Roller
        bc.is_neu[1, west] = True
        bc.is_dir[1, west] = False

        # North side: Roller
        bc.is_neu[0, north] = True
        bc.is_dir[0, north] = False

        # South side: Roller
        bc.is_neu[0, south] = True
        bc.is_dir[0, south] = False

        return bc

    def _bc_values_mechanics(self, sd: pp.Grid) -> np.ndarray:
        """Set boundary condition values for the mechanics subproblem.

        Args:
            sd: Subdomain grid.

        Returns:
            bc_values (sd.dim * sd.num_faces): Containing the boundary condition values.

        """

        # Retrieve boundary sides
        tol = self.params["perturbation_factor"] * 10
        _, _, _, north, *_ = self._domain_boundary_sides(sd, tol=tol)

        # All zeros except vertical component of the north side
        sd = self.mdg.subdomains()[0]

        # Retrieve physical data
        F = self.params["applied_load"]
        nu_u = self.undrained_poisson_coefficient()
        mu_s = self.params["mu_lame"]

        # Retrieve geometrical data
        a, b = self.params["domain_size"]

        u0y = (-F * b * (1 - nu_u)) / (2 * mu_s * a)
        bc_values = np.array([np.zeros(sd.num_faces), np.zeros(sd.num_faces)])
        bc_values[1, north] = u0y
        bc_values = bc_values.ravel("F")

        return bc_values

    def before_newton_loop(self) -> None:
        """Update time for time-stepping technique and bc values."""
        super().before_newton_loop()

        # Update value of boundary conditions
        self.update_north_bc_values(self.time_manager.time)

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        super().after_newton_convergence(solution, errors, iteration_counter)

        # Store solutions and errors
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time
        a, _ = self.params["domain_size"]
        vertical_load = self.params["applied_load"]

        if t in self.time_manager.schedule:
            # Displacement solutions
            self.sol[t]["u_num"] = data[pp.STATE][self.displacement_variable]
            self.sol[t]["u_ex"] = self.exact_displacement(t)
            # Pressure solutions
            self.sol[t]["p_num"] = data[pp.STATE][self.scalar_variable]
            self.sol[t]["p_ex"] = self.exact_pressure(t)
            # Flux solutions
            Q_num = self._fluid_flux([sd]).evaluate(self.dof_manager).val
            self.sol[t]["Q_num"] = Q_num
            self.sol[t]["Q_ex"] = self.exact_flux(t)
            # Traction solutions
            self.reconstruct_stress()
            self.sol[t]["T_num"] = data[pp.STATE]["stress"]
            self.sol[t]["T_ex"] = self.exact_traction(t)
            # Discrete L2-relative errors
            self.sol[t]["error_pressure"] = self.l2_relative_error(
                sd=sd,
                true_array=self.sol[t]["p_ex"] * a / vertical_load,
                approx_array=self.sol[t]["p_num"] * a / vertical_load,
                is_cc=True,
                is_scalar=True,
            )
            self.sol[t]["error_displacement"] = self.l2_relative_error(
                sd=sd,
                true_array=self.sol[t]["u_ex"],
                approx_array=self.sol[t]["u_num"],
                is_cc=True,
                is_scalar=False,
            )
            self.sol[t]["error_flux"] = self.l2_relative_error(
                sd=sd,
                true_array=self.sol[t]["Q_ex"],
                approx_array=self.sol[t]["Q_num"],
                is_cc=False,
                is_scalar=True,
            )
            self.sol[t]["error_traction"] = self.l2_relative_error(
                sd=sd,
                true_array=self.sol[t]["T_ex"],
                approx_array=self.sol[t]["T_num"],
                is_cc=False,
                is_scalar=False,
            )

    def after_simulation(self) -> None:
        """Postprocess and plot results"""
        self.postprocess_results()
        if self.params.get("plot_results", False):
            self.plot_results()

    # Physical parameters
    def _permeability(self, sd: pp.Grid) -> np.ndarray:
        """Override permeability value [m^2]

        Args:
            sd: Subdomain grid.

        Returns:
            Permeability.

        """
        return self.params["permeability"] * np.ones(sd.num_cells)

    def _stiffness_tensor(self, sd: pp.Grid) -> pp.FourthOrderTensor:
        """Override stifness tensor.

        Args:
            sd: Subdomain grid.

        Returns:
            Fourth order tensorial representation of the stiffness tensor.

        """
        lam = (self.params["lambda_lame"] * np.ones(sd.num_cells)) / self.scalar_scale
        mu = (self.params["mu_lame"] * np.ones(sd.num_cells)) / self.scalar_scale
        return pp.FourthOrderTensor(mu, lam)

    def _viscosity(self, sd: pp.Grid) -> np.ndarray:
        """Override fluid viscosity values [Pa.s]

        Args:
            sd: Subdomain grid.

        Returns:
            Viscosity
        """
        return self.params["viscosity"] * np.ones(sd.num_cells)

    def _storativity(self, sd: pp.Grid) -> np.ndarray:
        """Override storativity value of the porous medium [1/Pa]

        Args:
            sd: Subdomain grid.

        Retunrs:
            Storativity
        """
        return self.params["storativity"] * np.ones(sd.num_cells)

    def _biot_alpha(self, sd: pp.Grid) -> np.ndarray:
        """Override value of Biot-Willis coefficient.

        Args:
            sd: Subdomain grid.

        Returns:
            Biot's coefficient.
        """
        return self.params["alpha_biot"] * np.ones(sd.num_cells)

    # Other physical parameters used specifically for Mandel's problem
    def bulk_modulus(self) -> float:
        """Set bulk modulus [Pa].

        Returns:
            K_s: Bulk modulus.

        """
        mu_s = self.params["mu_lame"]
        lambda_s = self.params["lambda_lame"]
        K_s = (2 / 3) * mu_s + lambda_s

        return K_s

    def young_modulus(self) -> float:
        """Set Young modulus [Pa]

        Returns:
            E_s: Young modulus.

        """
        mu_s = self.params["mu_lame"]
        K_s = self.bulk_modulus()
        E_s = mu_s * ((9 * K_s) / (3 * K_s + mu_s))

        return E_s

    def poisson_coefficient(self) -> float:
        """Set Poisson coefficient [-]

        Returns:
            nu_s: Poisson coefficient.

        """
        mu_s = self.params["mu_lame"]
        K_s = self.bulk_modulus()
        nu_s = (3 * K_s - 2 * mu_s) / (2 * (3 * K_s + mu_s))

        return nu_s

    def undrained_bulk_modulus(self) -> float:
        """Set undrained bulk modulus [Pa]

        Returns:
            K_u: Undrained bulk modulus.

        """
        alpha_biot = self.params["alpha_biot"]
        K_s = self.bulk_modulus()
        S_m = self.params["storativity"]
        K_u = K_s + (alpha_biot**2) / S_m

        return K_u

    def skempton_coefficient(self) -> float:
        """Set Skempton's coefficient [-]

        Returns:
            B: Skempton's coefficent.

        """
        alpha_biot = self.params["alpha_biot"]
        K_u = self.undrained_bulk_modulus()
        S_m = self.params["storativity"]
        B = alpha_biot / (S_m * K_u)

        return B

    def undrained_poisson_coefficient(self) -> float:
        """Set Poisson coefficient under undrained conditions [-]

        Returns:
            nu_u: Undrained Poisson coefficient.

        """
        nu_s = self.poisson_coefficient()
        B = self.skempton_coefficient()
        nu_u = (3 * nu_s + B * (1 - 2 * nu_s)) / (3 - B * (1 - 2 * nu_s))

        return nu_u

    def fluid_diffusivity(self) -> float:
        """Set fluid diffusivity [m^2/s]

        Returns:
            c_f: Fluid diffusivity.

        """
        k_s = self.params["permeability"]
        B = self.skempton_coefficient()
        mu_s = self.params["mu_lame"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        mu_f = self.params["viscosity"]
        c_f = (2 * k_s * (B**2) * mu_s * (1 - nu_s) * (1 + nu_u) ** 2) / (
            9 * mu_f * (1 - nu_u) * (nu_u - nu_s)
        )
        return c_f

    # Analytical solutions
    def approximate_roots(self) -> np.ndarray:
        """
        Approximate roots to f(x) = 0, where f(x) = tan(x) - ((1-nu)/(nu_u-nu)) x

        Note that we have to solve the above equation numerically to get all positive
        solutions to the equation. Later, we will use them to compute the infinite series
        associated with the exact solutions. Experience has shown that 200 roots are enough
        to achieve accurate results.

        Implementation note:
            We find the roots using the bisection method. Thanks to Manuel Borregales who
            helped with the implementation of this part of the code.I have no idea what was
            the rationale behind the parameter tuning of the `bisect` method, but it seems
            to give good results.

        Returns:
            a_n: approximated roots of f(x) = 0.

        """

        # Retrieve physical data
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()

        # Define algebraic function
        def f(x):
            y = np.tan(x) - ((1 - nu_s) / (nu_u - nu_s)) * x
            return y

        n_series = self.params.get("number_of_roots", 200)
        a_n = np.zeros(n_series)  # initializing roots array
        x0 = 0  # initial point
        for i in range(n_series):
            a_n[i] = opt.bisect(
                f,  # function
                x0 + np.pi / 4,  # left point
                x0 + np.pi / 2 - 10000000 * 2.2204e-16,  # right point
                xtol=1e-30,  # absolute tolerance
                rtol=1e-14,  # relative tolerance
            )
            x0 += np.pi  # apply a phase change of pi to get the next root

        return a_n

    def exact_pressure(self, t: Union[float, int]) -> np.ndarray:
        """
        Exact pressure solution for a given time `t`.

        Args:
            t: Time in seconds.

        Returns:
            p (sd.num_cells, ): Exact pressure solution.

        """

        sd = self.mdg.subdomains()[0]

        # Retrieve data
        F = self.params["applied_load"]
        B = self.skempton_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.params["domain_size"]
        xc = sd.cell_centers[0]

        # -----> Compute exact fluid pressure

        if t == 0.0:  # initial condition has its own expression
            p = ((F * B * (1 + nu_u)) / (3 * a)) * np.ones(sd.num_cells)
        else:
            # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]
            # Exact p
            c0 = (2 * F * B * (1 + nu_u)) / (3 * a)
            p_sum_0 = np.sum(
                ((np.sin(aa_n)) / (aa_n - (np.sin(aa_n) * np.cos(aa_n))))
                * (np.cos((aa_n * xc) / a) - np.cos(aa_n))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            p = c0 * p_sum_0

        return p

    def exact_displacement(self, t: Union[float, int]) -> np.ndarray:
        """
        Exact displacement for a given time `t`.

        Args:
            t: Time in seconds.

        Returns:
            u (sd.dim * sd.num_cells, ): Exact displacement.

        """

        sd = self.mdg.subdomains()[0]

        # Retrieve physical data
        F = self.params["applied_load"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        mu_s = self.params["mu_lame"]
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.params["domain_size"]
        xc = sd.cell_centers[0]
        yc = sd.cell_centers[1]

        # -----> Compute exact displacement

        if t == 0.0:  # initial condition has its own expression
            u0x = ((F * nu_u) / (2 * mu_s * a)) * xc
            u0y = ((-F * (1 - nu_u)) / (2 * mu_s * a)) * yc
            u = np.array((u0x, u0y)).ravel("F")
        else:
            # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]
            # Exact ux
            cx0 = (F * nu_s) / (2 * mu_s * a)
            cx1 = -((F * nu_u) / (mu_s * a))
            cx2 = F / mu_s
            ux_sum1 = np.sum(
                (np.sin(aa_n) * np.cos(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            ux_sum2 = np.sum(
                (np.cos(aa_n) / (aa_n - (np.sin(aa_n) * np.cos(aa_n))))
                * np.sin((aa_n * xc) / a)
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            ux = (cx0 + cx1 * ux_sum1) * xc + cx2 * ux_sum2
            # Exact uy
            cy0 = (-F * (1 - nu_s)) / (2 * mu_s * a)
            cy1 = F * (1 - nu_u) / (mu_s * a)
            uy_sum1 = np.sum(
                ((np.sin(aa_n) * np.cos(aa_n)) / (aa_n - np.sin(aa_n) * np.cos(aa_n)))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            uy = (cy0 + cy1 * uy_sum1) * yc
            # Exact u
            u = np.array((ux, uy)).ravel("F")

        return u

    def exact_traction(self, t: Union[float, int]) -> np.ndarray:
        """Exact traction force for a given time `t`.

        Args:
            t: Time in seconds.

        Returns:
            T (sd.dim * sd.num_faces, ): Traction force on each face of the mesh.

        Technical note:
            Recall that the traction force (T) is defined as the dot product between the
            Biot stress tensor (sigma) and the face normal (n). Let T = [T_x; T_y],
            sigma = [sigma_xx, sigma_xy; sigma_xy, sigma_yy], and n = [n_x; n_y]. Then,
            there holds:
                            T_x = sigma_xx * n_x + sigma_xy * n_y
                            T_y = sigma_xy * n_x + simga_yy * n_y

            The exact Biot stress tensor for Mandel's problem has sigma_xy = sigma_xx = 0.
            Therefore, only T_y is expected to be non-zero.

        """

        sd = self.mdg.subdomains()[0]

        # Retrieve physical data
        F = self.params["applied_load"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.params["domain_size"]
        xf = sd.face_centers[0]
        nx = sd.face_normals[0]
        ny = sd.face_normals[1]

        # -----> Compute exact traction force

        sxx = np.zeros(sd.num_faces)  # sxx component of the stress is zero
        sxy = np.zeros(sd.num_faces)  # sxy (and syx) components of the stress are zero

        if t == 0:  # traction force at t = 0 has a different expression
            # Exact initial syy
            syy = -F / a * np.ones(sd.num_faces)
            # Components of the initial traction force
            Tx = sxx * nx + sxy * ny  # zero traction force in the x-direction
            Ty = sxy * nx + syy * ny  # non-zero traction force in the y-direction
            # Initial traction force
            T = np.array([Tx, Ty]).ravel("F")
        else:
            # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]
            # Exact syy
            c0 = -F / a
            c1 = (-2 * F * (nu_u - nu_s)) / (a * (1 - nu_s))
            syy_sum1 = np.sum(
                (np.sin(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.cos(aa_n * xf / a)
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            c2 = 2 * F / a
            syy_sum2 = np.sum(
                (np.sin(aa_n) * np.cos(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            syy = c0 + c1 * syy_sum1 + c2 * syy_sum2
            # Components of the initial traction force
            Tx = sxx * nx + sxy * ny  # zero traction force in the x-direction
            Ty = sxy * nx + syy * ny  # non-zero traction force in the y-direction
            # Traction force
            T = np.array([Tx, Ty]).ravel("F")

        return T

    def exact_flux(self, t: Union[float, int]) -> np.ndarray:
        """Exact Darcy flux at a given time `t`.

        Args:
            t: Time in seconds.

        Returns:
            Q (sd.num_faces, ): Darcy flux on each face of the grid.

        Technical note:
            Recall that the Darcy flux Q is the dot product between the specific discharge
            q = [q_x; q_y] and the normal vector n = [n_x; n_y], i.e.,

                                Q = q_x * n_x + q_y * n_y

            In Mandel's problem, q_y = 0.

        """
        sd = self.mdg.subdomains()[0]

        # Retrieve physical data
        F = self.params["applied_load"]
        B = self.skempton_coefficient()
        k = self.params["permeability"]
        mu_f = self.params["viscosity"]
        nu_u = self.undrained_poisson_coefficient()
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.params["domain_size"]
        xf = sd.face_centers[0]
        nx = sd.face_normals[0]
        ny = sd.face_normals[1]

        # -----> Compute exact Darcy flux

        if t == 0:  # initial Darcy flux is zero
            Q = np.zeros(sd.num_faces)
        else:
            # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]
            # x-component of specific discharge is non-zeros
            c0 = (2 * F * B * k * (1 + nu_u)) / (3 * mu_f * a**2)
            qx_sum0 = np.sum(
                (aa_n * np.sin(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.sin(aa_n * xf / a)
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            qx = c0 * qx_sum0
            # y-component of specific discharge is zero
            qy = np.zeros(sd.num_faces)
            # Compute exact Darcy flux
            Q = qx * nx + qy * ny

        return Q

    def exact_degree_of_consolidation(self, t: Union[float, int]) -> float:
        """Exact degree of consolidation for a given time `t`.

        Args:
              t: Time in seconds.

        Returns:
              U: Exact degree of consolidation for a given time `t`.

        Implementation note:
            The degrees of consolidation in the horizontal and vertical directions are
            identical.

        """

        # Retrieve physical data
        nu_u = self.undrained_poisson_coefficient()
        nu_s = self.poisson_coefficient()
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.params["domain_size"]

        # Retrieve approximated roots
        a_n = self.approximate_roots()

        # Compute degree of consolidation
        c0 = (4 * (1 - nu_u)) / (1 - 2 * nu_s)
        U_sum0 = np.sum(
            ((np.cos(a_n) * np.sin(a_n)) / (a_n - np.sin(a_n) * np.cos(a_n)))
            * np.exp((-(a_n**2) * c_f * t) / (a**2))
        )
        U = 1.0 - c0 * U_sum0

        return U

    def update_north_bc_values(self, t: Union[float, int]) -> None:
        """
        Updates boundary condition value at the north boundary of the domain.

        Args:
            t: Time in seconds.

        Note:
            The key `bc_values` from data[pp.PARAMETERS][self.mechanics_parameter_key]
            will be updated accordingly.

        """
        sd = self.mdg.subdomains()[0]

        # Retrieve physical data
        F = self.params["applied_load"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        mu_s = self.params["mu_lame"]
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a, b = self.params["domain_size"]
        yf = sd.face_centers[1]
        b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
        y_max = b_faces[yf[b_faces] > 0.9999 * b]

        # Auxiliary constant terms
        aa_n = self.approximate_roots()[:, np.newaxis]

        cy0 = (-F * (1 - nu_s)) / (2 * mu_s * a)
        cy1 = F * (1 - nu_u) / (mu_s * a)

        # Compute exact north boundary condition for the given time `t`
        uy_sum = np.sum(
            ((np.sin(aa_n) * np.cos(aa_n)) / (aa_n - np.sin(aa_n) * np.cos(aa_n)))
            * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
            axis=0,
        )

        north_bc = (cy0 + cy1 * uy_sum) * yf[y_max]

        # Update values
        data = self.mdg.subdomain_data(sd)
        kw_m = self.mechanics_parameter_key
        data[pp.PARAMETERS][kw_m]["bc_values"][1::2][y_max] = north_bc

    # Helper methods
    def horizontal_cut(self, array: np.ndarray) -> np.ndarray:
        """Perform a horizontal cut at the bottom of the domain.

        Note:
            This is done by obtaining the closest cell-centers to the line (0, 0) ; (a, 0).
            This functionality is similar to the Plot Over Line tool from ParaView.

        """
        sd = self.mdg.subdomains()[0]
        a, _ = self.params["domain_size"]
        half_max_diam = np.max(sd.cell_diameters()) / 2
        xc = np.arange(0, a, half_max_diam)
        closest_cells = sd.closest_cell(np.array([xc, np.zeros_like(xc)]))
        _, idx = np.unique(closest_cells, return_index=True)
        x_points = closest_cells[np.sort(idx)]
        cut_array = array[x_points]

        return cut_array

    @staticmethod
    def l2_relative_error(
        sd: pp.Grid,
        true_array: np.ndarray,
        approx_array: np.ndarray,
        is_cc: bool,
        is_scalar: bool,
    ) -> float:
        """Compute the error measured in the discrete (relative) L2-norm.

        The employed norms correspond respectively to equations (75) and (76) for the
        displacement and pressure from https://epubs.siam.org/doi/pdf/10.1137/15M1014280.

        Args:
            sd: PorePy grid.
            true_array: Exact array, e.g.: pressure, displacement, flux, or traction.
            approx_array: Approximated array, e.g.: pressure, displacement, flux, or traction.
            is_cc: True for cell-centered quanitities (e.g., pressure and displacement)
                and False for face-centered quantities (e.g., flux and traction).
            is_scalar: True for scalar quantities (e.g., pressure or flux) and False for
                vector quantities (displacement and traction).

        Returns:
            l2_error: discrete L2-error of the quantity of interest.

        """

        if is_cc:
            if is_scalar:
                meas = sd.cell_volumes
            else:
                meas = sd.cell_volumes.repeat(sd.dim)
        else:
            if is_scalar:
                meas = sd.face_areas
            else:
                meas = sd.face_areas.repeat(sd.dim)

        numerator = np.sqrt(np.sum(meas * np.abs(true_array - approx_array) ** 2))
        denominator = np.sqrt(np.sum(meas * np.abs(true_array) ** 2))
        l2_error = numerator / denominator

        return l2_error

    # Postprocessing methods
    def postprocess_results(self) -> None:
        """Postprocessing of results for plotting.

        Note:
            This method will create the following new fields, for all scheduled times,
            in the `self.sol` dictionary: `dimless_x`, `dimlesss_y`, `dimless_p_ex`,
            `dimless_p_num`, `dimless_ux_num`, `dimless_ux_ex`, `dimless_uy_num`,
            and `dimless_uy_ex`.

        """
        sd = self.mdg.subdomains()[0]

        F = self.params["applied_load"]

        a, b = self.params["domain_size"]
        xc = sd.cell_centers[0]
        yc = sd.cell_centers[1]

        for t in self.time_manager.schedule:

            # Retrieve numerical and exact pressures
            p_num = self.sol[t]["p_num"]
            p_ex = self.exact_pressure(t)

            # Retrieve numerical and exact displacements
            u_num = self.sol[t]["u_num"]
            ux_num = u_num[:: sd.dim]
            uy_num = u_num[1 :: sd.dim]
            u_ex = self.exact_displacement(t)
            ux_ex = u_ex[:: sd.dim]
            uy_ex = u_ex[1 :: sd.dim]

            # Store relevant quantities
            self.sol[t]["dimless_xc"] = self.horizontal_cut(xc / a)
            self.sol[t]["dimless_yc"] = self.horizontal_cut(yc / b)

            self.sol[t]["dimless_p_num"] = self.horizontal_cut(p_num * a / F)
            self.sol[t]["dimless_p_ex"] = self.horizontal_cut(p_ex * a / F)

            self.sol[t]["dimless_ux_num"] = self.horizontal_cut(ux_num / a)
            self.sol[t]["dimless_ux_ex"] = self.horizontal_cut(ux_ex / a)
            self.sol[t]["dimless_uy_num"] = self.horizontal_cut(uy_num / b)
            self.sol[t]["dimless_uy_ex"] = self.horizontal_cut(uy_ex / b)

    # Plotting methods
    def plot_results(self):
        """Plot dimensionless pressure, horizontal, and vertical displacements"""

        folder = "out/"
        fnamep = "pressure"
        fnameux = "horizontal_displacement"
        extension = ".pdf"
        cmap = mcolors.ListedColormap(
            plt.cm.tab20.colors[: len(self.time_manager.schedule)]
        )

        # -----> Pressure plot
        fig, ax = plt.subplots(figsize=(9, 8))
        for idx, t in enumerate(self.time_manager.schedule):
            ax.plot(
                self.sol[t]["dimless_xc"],
                self.sol[t]["dimless_p_ex"],
                color=cmap.colors[idx],
            )
            ax.plot(
                self.sol[t]["dimless_xc"],
                self.sol[t]["dimless_p_num"],
                color=cmap.colors[idx],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=cmap.colors[idx],
                linewidth=0,
                marker="s",
                markersize=12,
                label=rf"$t=${t}",
            )
        ax.set_xlabel(r"$x/a$", fontsize=15)
        ax.set_ylabel(r"$p/p_0$", fontsize=15)
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.set_title("Normalized pressure profiles", fontsize=16)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + fnamep + extension, bbox_inches="tight")
        plt.gcf().clear()

        # -----> Horizontal displacement plot
        fig, ax = plt.subplots(figsize=(9, 8))
        for idx, t in enumerate(self.time_manager.schedule):
            ax.plot(
                self.sol[t]["dimless_xc"],
                self.sol[t]["dimless_ux_ex"],
                color=cmap.colors[idx],
            )
            ax.plot(
                self.sol[t]["dimless_xc"],
                self.sol[t]["dimless_ux_num"],
                color=cmap.colors[idx],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=cmap.colors[idx],
                linewidth=0,
                marker="s",
                markersize=12,
                label=rf"$t=${t}",
            )
        ax.set_xlabel(r"$x/a$", fontsize=15)
        ax.set_ylabel(r"$u_x/a$", fontsize=15)
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.set_title("Normalized horizontal displacement profiles", fontsize=16)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + fnameux + extension, bbox_inches="tight")
        plt.gcf().clear()

        # -----> Errors as a function of time
        # error_p = np.asarray([self.sol[t]["error_pressure"] for t in self.tsc.schedule])
        # error_u = np.asarray([self.sol[t]["error_displacement"] for t in self.tsc.schedule])
        # error_Q = np.asarray([self.sol[t]["error_flux"] for t in self.tsc.schedule])
        # error_T = np.asarray([self.sol[t]["error_traction"] for t in self.tsc.schedule])
        # times = np.asarray(self.tsc.schedule)
        #
        # fig, ax = plt.subplots(figsize=(9, 8))
        # ax.loglog(times, error_p, "o-", label="Pressure error")
        # ax.loglog(times, error_u, "o-", label="Displacement error")
        # ax.loglog(times, error_Q, "o-", label="Flux error")
        # ax.loglog(times, error_T, "o-", label="Traction error")
        # ax.legend(fontsize=13)
        # ax.set_xlabel("Time [s]", fontsize=15)
        # ax.set_ylabel("Discrete L2-error [-]", fontsize=15)
        # ax.set_title("L2-errors as a function of time", fontsize=16)
        # ax.grid()
        # plt.show()

    def plot_field(
        self,
        t: Union[float, int],
        field: Literal["p_num", "ux_num", "uy_num", "p_ex", "ux_ex", "uy_ex"],
    ) -> None:
        """Plot pressure field for a given time `t`."""

        sd = self.mdg.subdomains()[0]

        ratio = np.round(self.params["width"] / self.params["height"])
        figsize = (int(5 * ratio), 5)

        if field == "p_num":
            array = self.sol[t]["p_num"]
        elif field == "ux_num":
            array = self.sol[t]["u_num"][:: sd.dim]
        elif field == "uy_num":
            array = self.sol[t]["u_num"][1 :: sd.dim]
        elif field == "p_ex":
            array = self.sol[t]["p_ex"]
        elif field == "ux_ex":
            array = self.sol[t]["u_ex"][:: sd.dim]
        else:
            array = self.sol[t]["u_ex"][1 :: sd.dim]

        pp.plot_grid(sd, array, figsize=figsize, plot_2d=True, title=field)


#%% Runner

# Time manager object
time_manager = pp.TimeManager(
    schedule=[
        0,
        10,
        50,
        100,
        500,
    ],  # [s]
    dt_init=1,  # [s]
    constant_dt=True
)
# Create application's parameter dictionary
app_params = {
    "use_ad": True,
    "mu_lame": 2.475e9,  # [Pa]
    "lambda_lame": 1.650e9,  # [Pa]
    "permeability": 9.869e-14,  # [m^2]
    "alpha_biot": 1.0,  # [-]
    "viscosity": 1e-3,  # [Pa.s]
    "storativity": 6.0606e-11,  # [1/Pa]
    "applied_load": 6e8,  # [N/m]
    "domain_size": (10.0, 10.0),  # [m]
    "num_cells": (20, 20),
    "time_manager": time_manager,
    "plot_results": True,
    "perturbation_factor": 1E-7,
    "mesh_type": "cartesian",
    "mesh_size": 2.0,  # [m]
}
# Run model
app = Mandel(app_params)
pp.run_time_dependent_model(app, app_params)

#%% Plotting
sd = app.mdg.subdomains()[0]
data = app.mdg.subdomain_data(sd)
p_num = data[pp.STATE][app.scalar_variable]
p_ex = app.exact_pressure(app.time_manager.time)
pp.plot_grid(sd, p_num, plot_2d=True, title="Numerical pressure")
pp.plot_grid(sd, p_ex, plot_2d=True, title="Exact pressure")