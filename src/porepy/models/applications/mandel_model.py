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
class MandelSolution:
    """Data class to store variables of interest for Mandel's problem"""

    def __init__(self, setup: "Mandel"):
        """Data class constructor.

        Args:
            setup : Mandel's model application.

        """
        sd = setup.mdg.subdomains()[0]
        data = setup.mdg.subdomain_data(sd)
        p_var = setup.scalar_variable
        u_var = setup.displacement_variable
        xc = sd.cell_centers[0]
        yc = sd.cell_centers[1]
        xf = sd.face_centers[0]
        t = setup.time_manager.time

        # Time variables
        self.time = t

        # Pressure variables
        self.num_pressure = data[pp.STATE][p_var]
        self.num_nondim_pressure = setup.nondim_p(self.num_pressure)

        self.ex_pressure = setup.exact_pressure(xc, t)
        self.ex_nondim_pressure = setup.nondim_p(self.ex_pressure)

        # Flux variables
        self.num_flux = setup.numerical_flux()
        self.ex_flux = setup.velocity_to_flux(setup.exact_velocity(xf, t))

        # Displacement variables
        self.num_displacement = data[pp.STATE][u_var]
        self.num_displacement_x = self.num_displacement[:: sd.dim]
        self.num_displacement_y = self.num_displacement[1 :: sd.dim]
        self.num_nondim_displacement_x = setup.nondim_x(self.num_displacement_x)
        self.num_nondim_displacement_y = setup.nondim_y(self.num_displacement_y)

        self.ex_displacement = setup.ravel(setup.exact_displacement(xc, yc, t))
        self.ex_displacement_x = self.ex_displacement[:: sd.dim]
        self.ex_displacement_y = self.ex_displacement[1 :: sd.dim]
        self.ex_nondim_displacement_x = setup.nondim_x(self.ex_displacement_x)
        self.ex_nondim_displacement_y = setup.nondim_y(self.ex_displacement_y)

        # Traction variables
        self.num_traction = setup.numerical_traction()
        self.ex_traction = setup.ravel(setup.stress_to_traction(setup.exact_stress(xf, t)))

        # Error variables
        self.pressure_error = setup.l2_relative_error(
            sd=sd,
            true_array=self.ex_pressure,
            approx_array=self.num_pressure,
            is_scalar=True,
            is_cc=True,
        )
        self.flux_error = setup.l2_relative_error(
            sd=sd,
            true_array=self.ex_flux,
            approx_array=self.num_flux,
            is_scalar=True,
            is_cc=False,
        )
        self.displacement_error = setup.l2_relative_error(
            sd=sd,
            true_array=self.ex_displacement,
            approx_array=self.num_displacement,
            is_scalar=False,
            is_cc=True,
        )
        self.traction_error = setup.l2_relative_error(
            sd=sd,
            true_array=self.ex_traction,
            approx_array=self.num_traction,
            is_scalar=False,
            is_cc=False,
        )


class Mandel(pp.ContactMechanicsBiot):
    """Parent class for Mandel's problem.

    Examples:

        .. code:: python

            # Import modules
            import porepy as pp
            from time import time

            # Run setup
            tic = time()
            setup = Mandel({plot_results: True})
            print("Simulation started...")
            pp.run_time_dependent_model(setup, setup.params)
            toc = time()
            print(f"Simulation finished in {round(toc - tic)} seconds."

    """

    def __init__(self, params: dict):
        """Constructor of the Mandel class.

        Parameters:
            params: Dictionary containing mandatory and optional model parameters.

                Default physical parameters were adapted from
                https://link.springer.com/article/10.1007/s10596-013-9393-8.

                Optional parameters are:

                - 'alpha_biot' : Biot constant (int or float). Default is 1.0.
                - 'domain_size' : Size of the domain (tuple of int or float). The first
                  element of the tuple is the width and the second the height. Default is
                  (100.0, 10.0).
                - 'lambda_lame' : Lamé parameter in `Pa` (int or float). Default is 1.65e9.
                - 'mesh_size' : Mesh size in `m` (int or float). Used only when
                  `mesh_size = "triangular"`. Default is 2.0.
                - 'mesh_type' : Type of mesh (str). Either "cartesian" or "triangular". The
                  first is a perturbed Cartesian grid and the second an unstructured
                  triangular mesh. Default is "cartesian".
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
        self.solutions: list[MandelSolution] = []

    def create_grid(self) -> None:
        """Create a two-dimensional Cartesian grid."""
        if self.params["mesh_type"] == "cartesian":
            nx, ny = self.params["num_cells"]
            lx, ly = self.params["domain_size"]
            phys_dims = np.array([lx, ly])
            n_cells = np.array([nx, ny])
            self.box = pp.geometry.bounding_box.from_points(
                np.array([[0, 0], phys_dims]).T
            )
            sd: pp.Grid = pp.CartGrid(n_cells, phys_dims)
            sd.compute_geometry()

            # Perturb nodes to avoid singular matrices with rollers and MPSA.
            np.random.seed(42)  # this seed is fixed but completely arbitrary
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
        xc = sd.cell_centers[0]
        yc = sd.cell_centers[1]
        data = self.mdg.subdomain_data(sd)

        # Set initial pressure
        data[pp.STATE][self.scalar_variable] = self.exact_pressure(xc, 0)
        data[pp.STATE][pp.ITERATE][self.scalar_variable] = self.exact_pressure(xc, 0)

        # Set initial displacement
        data[pp.STATE][self.displacement_variable] = self.ravel(self.exact_displacement(xc,
                                                                                      yc, 0))
        data[pp.STATE][pp.ITERATE][
            self.displacement_variable
        ] = self.ravel(self.exact_displacement(xc, yc, 0))

        # Store initial solution
        self.solutions.append(MandelSolution(self))

        # Store initial pressure and displacement distributions in the `sol` dictionary
        # self.sol[0]["p_num"] = self.exact_pressure(0)
        # self.sol[0]["p_ex"] = self.exact_pressure(0)
        # self.sol[0]["u_num"] = self.exact_displacement(0)
        # self.sol[0]["u_ex"] = self.exact_displacement(0)
        # self.sol[0]["Q_num"] = self.exact_flux(0)
        # self.sol[0]["Q_ex"] = self.exact_flux(0)
        # self.sol[0]["T_num"] = self.exact_traction(0)
        # self.sol[0]["T_ex"] = self.exact_traction(0)
        # self.sol[0]["U_num"] = 0
        # self.sol[0]["U_ex"] = 0
        # self.sol[0]["error_pressure"] = np.nan
        # self.sol[0]["error_displacement"] = np.nan
        # self.sol[0]["error_flux"] = np.nan
        # self.sol[0]["error_traction"] = np.nan

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Define boundary condition types for the flow subproblem.

        Args:
            sd: Subdomain grid.

        Returns:
            bc: Scalar boundary condition representation.

        """
        # Define boundary regions
        tol = self.params["perturbation_factor"]
        sides = self._domain_boundary_sides(sd, tol)
        east_bc = np.isin(sides.all_bf, np.where(sides.east)).nonzero()

        # All sides Neumann, except the East side which is Dirichlet
        bc_type = np.asarray(sides.all_bf.size * ["neu"])
        bc_type[east_bc] = "dir"

        bc = pp.BoundaryCondition(sd, faces=sides.all_bf, cond=list(bc_type))

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
        tol = self.params["perturbation_factor"]
        sides = self._domain_boundary_sides(sd, tol)
        data = self.mdg.subdomain_data(sd)
        bc = data[pp.PARAMETERS][self.mechanics_parameter_key]["bc"]

        # East side: Stress-free
        bc.is_neu[:, sides.east] = True
        bc.is_dir[:, sides.east] = False

        # West side: Roller
        bc.is_neu[1, sides.west] = True
        bc.is_dir[1, sides.west] = False

        # North side: Roller
        bc.is_neu[0, sides.north] = True
        bc.is_dir[0, sides.north] = False

        # South side: Roller
        bc.is_neu[0, sides.south] = True
        bc.is_dir[0, sides.south] = False

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

        # Store solutions
        schedule = self.time_manager.schedule
        if any([np.isclose(self.time_manager.time, t_sch) for t_sch in schedule]):
            self.solutions.append(MandelSolution(self))

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        if self.params["plot_results"]:
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

    # -----> Exact, numerical, and non-dimensional expressions
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

    def exact_pressure(self, x: np.ndarray, t: Union[float, int]) -> np.ndarray:
        """
        Exact pressure solution for a given time `t`.

        Args:
            x: Points in the horizontal axis.
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

        # Compute exact pressure
        if t == 0.0:  # initial condition has its own expression
            p = ((F * B * (1 + nu_u)) / (3 * a)) * np.ones(sd.num_cells)
        else:
            # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]
            # Exact p
            c0 = (2 * F * B * (1 + nu_u)) / (3 * a)
            p_sum_0 = np.sum(
                ((np.sin(aa_n)) / (aa_n - (np.sin(aa_n) * np.cos(aa_n))))
                * (np.cos((aa_n * x) / a) - np.cos(aa_n))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            p = c0 * p_sum_0

        return p

    def exact_displacement(self,
            x: np.ndarray,
            y: np.ndarray,
            t: Union[float, str]
        ) -> list[np.ndarray, np.ndarray]:
        """
        Exact horizontal and vertical displacement for a given time ``t``.

        Args:
            x: Points in the horizontal axis in meters.
            y: Points in the vertical axis in meters.
            t: Time in seconds.

        Returns:
            Exact displacement in the horizontal and vertical directions.

        """
        # Retrieve physical data
        F = self.params["applied_load"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        mu_s = self.params["mu_lame"]
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.params["domain_size"]

        # Determine displacements
        if t == 0.0:  # initial condition has its own expression

            ux = ((F * nu_u) / (2 * mu_s * a)) * x
            uy = ((-F * (1 - nu_u)) / (2 * mu_s * a)) * y
            u = [ux, uy]

        else:

            # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]

            # Exact horizontal displacement
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
                * np.sin((aa_n * x) / a)
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            ux = (cx0 + cx1 * ux_sum1) * x + cx2 * ux_sum2

            # Exact vertical displacement
            cy0 = (-F * (1 - nu_s)) / (2 * mu_s * a)
            cy1 = F * (1 - nu_u) / (mu_s * a)
            uy_sum1 = np.sum(
                ((np.sin(aa_n) * np.cos(aa_n)) / (aa_n - np.sin(aa_n) * np.cos(aa_n)))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            uy = (cy0 + cy1 * uy_sum1) * y

            # Exact u
            u = [ux, uy]

        return u

    def exact_velocity(
            self,
            x: np.ndarray,
            t: Union[float, int]
    ) -> list[np.ndarray, np.ndarray]:
        """Exact Darcy's velocity (specific discharge) in `m * s^{-1}`.

        For Mandel's problem, only the horizontal component in non-zero.

        Args:
            x: Points in the horizontal axis in meters.
            t: Time in seconds.

        Returns:
            List of exact velocities for the given time ``t``. Each item of the list has a
                size of sd.num_faces.

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

        # Compute specific discharge
        if t == 0:
            q = [np.zeros(sd.num_faces), np.zeros(sd.num_faces)]
        else:
            # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]

            c0 = (2 * F * B * k * (1 + nu_u)) / (3 * mu_f * a**2)
            qx_sum0 = np.sum(
                (aa_n * np.sin(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.sin(aa_n * x / a)
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            qx = c0 * qx_sum0

            q = [qx, np.zeros(sd.num_faces)]

        return q

    def exact_stress(self,
                     x: np.ndarray,
                     t: Union[float, int],
        ) -> list[list[np.ndarray, np.ndarray], list[np.ndarray, np.ndarray]]:
        """Exact stress tensor in `Pa`.

        In Mandel's problem, only the `yy` component of the stress tensor is non-zero.

        Parameters:
            x: Points in the horizontal axis in meters.
            t: Time in seconds.

        Returns:
            List of lists of arrays, representing the components of the exact symmetric
                stress tensor. Each item of the inner lists has size sd.num_faces.

        """

        sd = self.mdg.subdomains()[0]

        # Retrieve physical data
        F = self.params["applied_load"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a, _ = self.params["domain_size"]

        # Compute exact stress tensor

        sxx = np.zeros(sd.num_faces)  # sxx component of the stress is zero
        sxy = np.zeros(sd.num_faces)  # sxy components of the stress is zero
        syx = np.zeros(sd.num_faces)  # syx components of the stress is zero

        if t == 0:  # traction force at t = 0 has a different expression

            # Exact initial syy
            syy = -F / a * np.ones(sd.num_faces)

            # Exact stress
            stress = [[sxx, sxy], [syx, syy]]

        else:

           # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]

            # Exact syy
            c0 = -F / a
            c1 = (-2 * F * (nu_u - nu_s)) / (a * (1 - nu_s))
            syy_sum1 = np.sum(
                (np.sin(aa_n))
                / (aa_n - np.sin(aa_n) * np.cos(aa_n))
                * np.cos(aa_n * x / a)
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

            # Exact stress
            stress = [[sxx, sxy], [syx, syy]]

        return stress

    def numerical_flux(self) -> np.ndarray:
        """Compute numerical flux.

        Returns:
            Numerical Darcy fluxes at the face centers. Shape is (sd.num_faces, ).

        """
        sd = self.mdg.subdomains()[0]
        xc = sd.cell_centers[0]
        t = self.time_manager.time
        if t == 0:
            return self.velocity_to_flux(self.exact_velocity(xc, t))
        else:
            flux_ad = self._fluid_flux([sd])
            return flux_ad.evaluate(self.dof_manager).val

    def numerical_traction(self) -> np.ndarray:
        """Compute numerical traction.

        Returns:
            Numerical tractions at the face centers. Shape is (sd.dim * sd.num_faces, ).

        """
        sd = self.mdg.subdomains()[0]
        xc = sd.cell_centers[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        if t == 0:
            return self.ravel(self.stress_to_traction(self.exact_stress(xc, t)))
        else:
            self.reconstruct_stress()
            return data[pp.STATE]["stress"]

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

    def nondim_t(self, t: Union[float, int]) -> float:
        """Nondimensionalize time.

        Args:
            t: Time in seconds.

        Returns:
            Dimensionless time for the given time ``t``.

        """
        a, _ = self.params["domain_size"]
        c_f = self.fluid_diffusivity()

        return (t * c_f) / (a**2)

    def nondim_x(self, x: np.ndarray) -> np.ndarray:
        """Nondimensionalize length in the horizontal axis.

        Args:
            x: horizontal length in meters.

        Returns:
            Dimensionless horizontal length.

        """
        a, _ = self.params["domain_size"]
        return x / a

    def nondim_y(self, y: np.ndarray) -> np.ndarray:
        """Nondimensionalize length in the vertical axis.

        Args:
            y: vertical length in meters.

        Returns:
            Dimensionless vertical length.

        """
        _, b = self.params["domain_size"]
        return y / b

    def nondim_p(self, p: np.ndarray) -> np.ndarray:
        """Nondimensionalize pressure.

        Args:
            p: Pressure in Pascals.

        Returns:
            Nondimensional pressure.

        """
        a, _ = self.params["domain_size"]  # [m]
        F = self.params["vertical_load"]  # [N * m^{-1}]
        return p / (F * a)

    # -----> Helper methods

    def xcut(self, array: np.ndarray) -> np.ndarray:
        """Perform a horizontal cut at the bottom of the domain.

        For triangular grids, this is done by obtaining the closest cell-centers to the line
        constructed from the points (0, 0) and (a, 0). For perturbed Cartesian grids,
        we simply select the cells corresponding to the bottom row.

        Parameters:
            array: Array to be cut. Shape is (sd.num_cells, ).

        Returns:
            Horizontally cut array from (0, dx) to (a, dx), where dx = cell diameter / 2.

        """
        if self.params["mesh_type"] == "cartesian":
            nx, ny = self.params["num_cells"]
            cut_array = array[: ny : nx * ny]
        else:
            sd = self.mdg.subdomains()[0]
            a, _ = self.params["domain_size"]
            half_max_diam = np.max(sd.cell_diameters()) / 2
            xc = np.arange(0, a, half_max_diam)
            closest_cells = sd.closest_cell(np.array([xc, np.zeros_like(xc)]))
            _, idx = np.unique(closest_cells, return_index=True)
            x_points = closest_cells[np.sort(idx)]
            cut_array = array[x_points]

        return cut_array

    def velocity_to_flux(
            self,
            velocity: list[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Convert a velocity field into (integrated normal) fluxes.

        The usual application is to compute (integrated normal) Darcy fluxes form specific
        discharge.

        Parameters:
            velocity: list of arrays in `m * s^{-1}`. Expected shape of each item on the list
                is (sd.num_faces, ).

        Returns:
            Integrated normal fluxes `m^3 * s^{-1}` on the face centers of the grid.

        """
        sd = self.mdg.subdomains()[0]

        # Sanity check on input parameter
        assert velocity[0].size == sd.num_faces
        assert velocity[1].size == sd.num_faces

        flux = velocity[0] * sd.face_normals[0] + velocity[1] * sd.face_normals[1]

        return flux

    def stress_to_traction(
            self,
            stress: list[list[np.ndarray, np.ndarray], list[np.ndarray, np.ndarray]],

    ) -> list[np.ndarray, np.ndarray]:
        """Convert a stress field into (integrated normal) traction forces.

        Parameters:
            stress: list of lists of arrays in `Pa`. Expected shape for each item of the
                list is (sd.num_faces, ).

        Returns:
            List of integrated traction forces `N` on the face centers of the grids.

        """
        sd = self.mdg.subdomains()[0]

        # Sanity check on input parameter
        assert stress[0][0].size == sd.num_faces
        assert stress[0][1].size == sd.num_faces
        assert stress[1][0].size == sd.num_faces
        assert stress[1][1].size == sd.num_faces

        traction_x = stress[0][0] * sd.face_normals[0] + stress[0][1] * sd.face_normals[1]
        traction_y = stress[1][0] * sd.face_normals[0] + stress[1][1] * sd.face_normals[1]

        return [traction_x, traction_y]

    @staticmethod
    def ravel(vector: list[np.ndarray, np.ndarray]) -> np.ndarray:
        """Convert a vector quantity into PorePy format.

        Parameters:
            vector: A vector quantity represented as a list with two items.

        Returns:
            Raveled version of the vector.
        """
        return np.array(vector).ravel("F")

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



    # # Postprocessing methods
    # def postprocess_results(self) -> None:
    #     """Postprocessing of results for plotting.
    #
    #     Note:
    #         This method will create the following new fields, for all scheduled times,
    #         in the `self.sol` dictionary: `dimless_x`, `dimlesss_y`, `dimless_p_ex`,
    #         `dimless_p_num`, `dimless_ux_num`, `dimless_ux_ex`, `dimless_uy_num`,
    #         and `dimless_uy_ex`.
    #
    #     """
    #     sd = self.mdg.subdomains()[0]
    #
    #     F = self.params["applied_load"]
    #
    #     a, b = self.params["domain_size"]
    #     xc = sd.cell_centers[0]
    #     yc = sd.cell_centers[1]
    #
    #     for t in self.time_manager.schedule:
    #
    #         # Retrieve numerical and exact pressures
    #         p_num = self.sol[t]["p_num"]
    #         p_ex = self.exact_pressure(t)
    #
    #         # Retrieve numerical and exact displacements
    #         u_num = self.sol[t]["u_num"]
    #         ux_num = u_num[:: sd.dim]
    #         uy_num = u_num[1 :: sd.dim]
    #         u_ex = self.exact_displacement(t)
    #         ux_ex = u_ex[:: sd.dim]
    #         uy_ex = u_ex[1 :: sd.dim]
    #
    #         # Store relevant quantities
    #         self.sol[t]["dimless_xc"] = self.horizontal_cut(xc / a)
    #         self.sol[t]["dimless_yc"] = self.horizontal_cut(yc / b)
    #
    #         self.sol[t]["dimless_p_num"] = self.horizontal_cut(p_num * a / F)
    #         self.sol[t]["dimless_p_ex"] = self.horizontal_cut(p_ex * a / F)
    #
    #         self.sol[t]["dimless_ux_num"] = self.horizontal_cut(ux_num / a)
    #         self.sol[t]["dimless_ux_ex"] = self.horizontal_cut(ux_ex / a)
    #         self.sol[t]["dimless_uy_num"] = self.horizontal_cut(uy_num / b)
    #         self.sol[t]["dimless_uy_ex"] = self.horizontal_cut(uy_ex / b)

    # Plotting
    def plot_results(self):
        """Plot dimensionless pressure, horizontal, and vertical displacements"""

        folder = "out/"
        fnamep = "pressure"
        fnameux = "horizontal_displacement"
        fnameuy = "vertical_displacement"
        extension = ".pdf"
        cmap = mcolors.ListedColormap(
            plt.cm.tab20.colors[: len(self.time_manager.schedule)]
        )

        # Pressure plot
        self._pressure_plot(
            folder=folder, file_name=fnamep, file_extension=extension, color_map=cmap
        )

    def _pressure_plot(
        self,
        folder: str,
        file_name: str,
        file_extension: str,
        color_map: mcolors.ListedColormap,
    ) -> None:
        """Plot nondimensional pressure profiles.

        Args:
            folder: name of the folder to store the results e.g., "out/".
            file_name: name of the file e.g., "pressure_profiles".
            file_extension: extension of the file e.g., ".pdf".
            color_map: listed color map object.

        """
        fig, ax = plt.subplots(figsize=(9, 8))

        sd = self.mdg.subdomains()[0]
        xc = sd.cell_centers[0]

        a, _ = self.params["domain_size"]
        x_ex = np.linspace(0, a, 400)

        t = self.time_manager.time

        for idx, sol in enumerate(self.solutions):
            ax.plot(
                self.nondim_p(self.exact_pressure(x=x_ex, t=sol.time)),
                self.nondim_x(x=x_ex),
                color=color_map.colors[idx],
            )
            ax.plot(
                self.xcut(sol.num_pressure),
                self.xcut(self.nondim_x(xc)),
                color=color_map.colors[idx],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=color_map.colors[idx],
                linewidth=0,
                marker="s",
                markersize=12,
                label=rf"$t=${t}",
            )
        ax.set_xlabel(r"$\tilde{x} = x/a$", fontsize=15)
        ax.set_ylabel(r"$\tilde{p} = p/(F a)$", fontsize=15)
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.set_title("Normalized pressure profiles", fontsize=16)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + file_name + file_extension, bbox_inches="tight")
        plt.gcf().clear()

    def _displacement_plots(
        self,
        folder: str,
        file_names: tuple[str, str],
        file_extension: str,
        color_map: mcolors.ListedColormap,
    ) -> None:
        """Plot nondimensional pressure profiles.

        Args:
            folder: name of the folder to store the results e.g., "out/".
            file_names: name of the files e.g., ("ux", "uy").
            file_extension: extension of the file e.g., ".pdf".
            color_map: listed color map object.
        """

        fname_ux, fname_uy = file_names

        # Horizontal displacement plot
        fig, ax = plt.subplots(figsize=(9, 8))
        for idx, t in enumerate(self.time_manager.schedule):
            ax.plot(
                self.sol[t]["dimless_xc"],
                self.sol[t]["dimless_ux_ex"],
                color=color_map.colors[idx],
            )
            ax.plot(
                self.sol[t]["dimless_xc"],
                self.sol[t]["dimless_ux_num"],
                color=color_map.colors[idx],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=color_map.colors[idx],
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
        plt.savefig(folder + fname_ux + file_extension, bbox_inches="tight")
        plt.gcf().clear()

    # Vertical displacement plot

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
    constant_dt=True,
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
    "perturbation_factor": 1e-7,
    "mesh_type": "cartesian",
    "mesh_size": 2.0,  # [m]
}
# Run model
setup = Mandel(app_params)
pp.run_time_dependent_model(setup, app_params)

#%% Plotting
sd = setup.mdg.subdomains()[0]
data = setup.mdg.subdomain_data(sd)
xc = sd.cell_centers[0]
yc = sd.cell_centers[1]
t = setup.time_manager.time
p_num = data[pp.STATE][setup.scalar_variable]
p_ex = setup.exact_pressure(xc, t)
pp.plot_grid(sd, p_num, plot_2d=True, title="Numerical pressure")
pp.plot_grid(sd, p_ex, plot_2d=True, title="Exact pressure")
