"""
Implementation of Terzaghi's consolidation problem.

Even though the model is strictly speaking a one-dimensional problem, the implemented model
uses a two-dimensional unstructured grid with roller boundary conditions for the mechanical
subproblem and no-flux boundary conditions for the flow subproblem in order to emulate the
one-dimensional process.

"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import porepy as pp
import os

from typing import Union


class Terzaghi(pp.ContactMechanicsBiot):
    """Parent class for Terzaghi's consolidation problem model."""

    def __init__(self, params: dict):
        """
        Constructor of the Terzaghi class.

        Args:
            params: Model parameters.

        Mandatory model parameters:
            height (float): Height of the domain.
            vertical_load (float): Applied vertical load.
            time_manager (pp.TimeManager): Time stepping control object.

        Optional model parameters:
            mesh_size (float, Default is 0.1): Mesh size.
            upper_limit (int, Default is 1000): Upper limit of summation for computing exact
                solutions.

        """
        super().__init__(params)

        # Create a solution dictionary to store pressure and displacement solutions
        self.sol = {counter: {} for counter in range(len(self.time_manager.schedule))}
        self._ee: int = 0  # exporter counter

    def create_grid(self) -> None:
        """Create two-dimensional unstructured mixed-dimensional grid."""
        h = self.params["height"]
        mesh_size = self.params["mesh_size"]
        self.box = {"xmin": 0.0, "xmax": h, "ymin": 0.0, "ymax": h}
        network_2d = pp.FractureNetwork2d(None, None, self.box)
        mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}
        self.mdg = network_2d.mesh(mesh_args)

    def _initial_condition(self) -> None:
        """Override initial condition for the flow subproblem."""
        super()._initial_condition()
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        F = self.params["applied_load"]
        initial_p = F * np.ones(sd.num_cells)
        data[pp.STATE][self.scalar_variable] = initial_p
        data[pp.STATE][pp.ITERATE][self.scalar_variable] = initial_p

        # Store initial pressure and displacement solutions
        self.sol[self._ee]["time"] = self.time_manager.time
        self.sol[self._ee]["dimless_t"] = self.dimensionless_time(self.time_manager.time)
        self.sol[self._ee]["p_num"] = initial_p
        self.sol[self._ee]["p_ex"] = initial_p
        self.sol[self._ee]["u_num"] = np.zeros(sd.dim * sd.num_cells)
        self.sol[self._ee]["u_ex"] = np.zeros(sd.dim * sd.num_cells)

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Define boundary condition types for the flow subproblem.

        Args:
            sd: Subdomain grid.

        Returns:
            bc: Scalar boundary condition representation.

        """

        # Define boundary regions
        all_bc, _, _, north, *_ = self._domain_boundary_sides(sd)
        north_bc = np.isin(all_bc, np.where(north)).nonzero()

        # All sides Neumann, except the North which is Dirichlet
        bc_type = np.asarray(all_bc.size * ["neu"])
        bc_type[north_bc] = "dir"

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
        _, east, west, north, south, *_ = self._domain_boundary_sides(sd)
        data = self.mdg.subdomain_data(sd)
        bc = data[pp.PARAMETERS][self.mechanics_parameter_key]["bc"]

        # East side: Roller
        bc.is_neu[1, east] = True
        bc.is_dir[1, east] = False

        # West side: Roller
        bc.is_neu[1, west] = True
        bc.is_dir[1, west] = False

        # North side: Neumann
        bc.is_neu[:, north] = True
        bc.is_dir[:, north] = False

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
        _, _, _, north, *_ = self._domain_boundary_sides(sd)

        # All zeros except vertical component of the north side
        vertical_load = self.params["applied_load"]
        bc_values = np.array([np.zeros(sd.num_faces), np.zeros(sd.num_faces)])
        bc_values[1, north] = -vertical_load * sd.face_areas[north]
        bc_values = bc_values.ravel("F")

        return bc_values

    def after_newton_convergence(
        self,
        solution: np.ndarray,
        errors: float,
        iteration_counter: int,
    ) -> None:
        super().after_newton_convergence(solution, errors, iteration_counter)

        # Store solutions
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        schedule = self.time_manager.schedule
        if any([np.isclose(self.time_manager.time, t_sch) for t_sch in schedule]):
            self._ee += 1  # increase exporter counter
            self.sol[self._ee]["time"] = self.time_manager.time
            self.sol[self._ee]["dimless_t"] = self.dimensionless_time(self.time_manager.time)
            self.sol[self._ee]["u_num"] = data[pp.STATE][self.displacement_variable]
            self.sol[self._ee]["p_num"] = data[pp.STATE][self.scalar_variable]

    def after_simulation(self) -> None:
        """Postprocess and plot results"""
        self.postprocess_results()
        if self.params.get("plot_results", False):
            self.plot_results()

    # Physical parameters
    def _permeability(self, sd: pp.Grid) -> np.ndarray:
        """Override value of intrinsic permeability [m^2].

        Args:
            sd: Subdomain grid.

        Returns:
            permeability (sd.num_cells): containing the permeability values at each cell.

        """

        permeability = self.params["permeability"] * np.ones(sd.num_cells)

        return permeability

    def _storativity(self, sd: pp.Grid) -> np.ndarray:
        """Override value of storativity [Pa^{-1}].

        Args:
            sd: Subdomain grid.

        Returns:
            storativity (sd.num_cells): containing the storativity values at each cell.

        """

        storativity = np.zeros(sd.num_cells)

        return storativity

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

    def _biot_alpha(self, sd: pp.Grid) -> np.ndarray:
        """Override value of Biot-Willis coefficient.

        Args:
            sd: Subdomain grid.

        Returns:
            Biot's coefficient.
        """
        return self.params["alpha_biot"] * np.ones(sd.num_cells)

    # Other physical parameters used specifically for Terzaghi's problem
    def confined_compressibility(self) -> float:
        """Confined compressibility [1/Pa].

        Returns:
            m_v: confined compressibility.

        """
        mu_s = self.params["mu_lame"]
        lambda_s = self.params["lambda_lame"]
        m_v = 1 / (2 * mu_s + lambda_s)

        return m_v

    def consolidation_coefficient(self) -> float:
        """Consolidation coefficient [m^2 * s^-1}]

        Returns:
            c_v: coefficient of consolidation.

        """

        k = self.params["permeability"]  # [m^2]
        rho_f = self.params["fluid_density"]  # [kg/m^3]
        mu_f = self.params["viscosity"]  # [Pa.s]
        g = pp.GRAVITY_ACCELERATION  # [m/s^2]
        gamma_f = rho_f * g  # volumetric weight  [1/Pa]
        K = (k * gamma_f) / mu_f  # hydraulic conductivity [m/s]

        S_eps = 0  # storativity [1/Pa]
        alpha_biot = self.params["alpha_biot"]  # [-]
        m_v = self.confined_compressibility()  # [1/Pa]

        c_v = K / (gamma_f * (S_eps + alpha_biot**2 * m_v))

        return c_v

    #  Analytical solution methods
    def dimensionless_time(self, t: Union[float, int]) -> float:
        """
        Compute exact dimensionless time.

        Args:
            t: Time in seconds.

        Returns:
            Dimensionless time for the given time `t`.

        """

        h = self.params["height"]
        c_v = self.consolidation_coefficient()

        return (t * c_v) / (h**2)

    def exact_pressure(self, t: Union[float, int]) -> np.ndarray:
        """
        Compute exact pressure.

        Args:
            t: Time in seconds.

        Returns:
            Exact pressure for the given time `t`.

        """

        sd = self.mdg.subdomains()[0]
        yc = sd.cell_centers[1]
        h = self.params["height"]
        F = self.params["applied_load"]
        dimless_t = self.dimensionless_time(t)

        n = self.params.get("upper_summation_limit", 1000)

        sum_series = np.zeros_like(yc)
        for i in range(1, n + 1):
            sum_series += (
                (((-1) ** (i - 1)) / (2 * i - 1))
                * np.cos((2 * i - 1) * (np.pi / 2) * (yc / h))
                * np.exp((-((2 * i - 1) ** 2)) * (np.pi**2 / 4) * dimless_t)
            )
        p = (4 / np.pi) * F * sum_series

        return p

    # -----------> Helper methods
    def vertical_cut(self, array: np.ndarray) -> np.ndarray:
        """Perform a vertical cut in the middle of the domain.

        Note:
            This is done by obtaining the closest vertical cell-centers to the line
            (h/2, 0) (h/2, h). This functionality is similar to the Plot Over Line
            tool from ParaView.

        """
        sd = self.mdg.subdomains()[0]
        h = self.params["height"]
        half_max_diam = np.max(sd.cell_diameters()) / 2
        yc = np.arange(0, h, half_max_diam)
        closest_cells = sd.closest_cell(np.array([h / 2 * np.ones_like(yc), yc]))
        _, idx = np.unique(closest_cells, return_index=True)
        y_points = closest_cells[np.sort(idx)]
        cut_array = array[y_points]

        return cut_array

    def displacement_trace(
        self, displacement: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """Project the displacement vector onto the faces.

        Args:
            displacement (sd.dim * sd.num_cells, ): displacement solution.
            pressure (sd.num_cells, ): pressure solution.

        Returns:
            trace_u (sd.dim * sd.num_faces, ): trace of the displacement.

        """

        # Rename arguments
        u = displacement
        p = pressure

        # Discretization matrices
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        bound_u_cell = data[pp.DISCRETIZATION_MATRICES][self.mechanics_parameter_key][
            "bound_displacement_cell"
        ]
        bound_u_face = data[pp.DISCRETIZATION_MATRICES][self.mechanics_parameter_key][
            "bound_displacement_face"
        ]
        bound_u_pressure = data[pp.DISCRETIZATION_MATRICES][
            self.mechanics_parameter_key
        ]["bound_displacement_pressure"]

        # Mechanical boundary values
        bc_vals = data[pp.PARAMETERS][self.mechanics_parameter_key]["bc_values"]

        # Compute trace of the displacement
        trace_u = bound_u_cell * u + bound_u_face * bc_vals + bound_u_pressure * p

        return trace_u

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
        h = self.params["height"]
        yc = sd.cell_centers[1]

        for key in self.sol:

            # Retrieve numerical and exact pressures
            t = self.sol[key]["time"]
            p_num = self.sol[key]["p_num"]
            p_ex = self.exact_pressure(t)

            # Store relevant quantities
            self.sol[key]["dimless_yc"] = self.vertical_cut(yc / h)

            self.sol[key]["dimless_p_num"] = self.vertical_cut(p_num / F)
            self.sol[key]["dimless_p_ex"] = self.vertical_cut(p_ex / F)

    def plot_results(self):
        """Plot dimensionless pressure"""

        folder = "out/"
        fnamep = "pressure"
        extension = ".pdf"
        cmap = mcolors.ListedColormap(plt.cm.tab20.colors[: len(self.time_manager.schedule)])

        # -----> Pressure plot
        fig, ax = plt.subplots(figsize=(9, 8))
        for key in self.sol:
            ax.plot(
                self.sol[key]["dimless_p_ex"],
                self.sol[key]["dimless_yc"],
                color=cmap.colors[key],
            )
            ax.plot(
                self.sol[key]["dimless_p_num"],
                self.sol[key]["dimless_yc"],
                color=cmap.colors[key],
                linewidth=0,
                marker=".",
                markersize=8,
            )
            ax.plot(
                [],
                [],
                color=cmap.colors[key],
                linewidth=0,
                marker="s",
                markersize=12,
                label=rf"$\tau=${np.round(self.sol[key]['dimless_t'], 6)}",
            )
        ax.set_xlabel(r"$\tilde{p} = p/p_0$", fontsize=15)
        ax.set_ylabel(r"$\tilde{y} = y/h$", fontsize=15)
        ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), fontsize=13)
        ax.set_title("Normalized pressure profiles", fontsize=16)
        ax.grid()
        plt.subplots_adjust(right=0.7)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + fnamep + extension, bbox_inches="tight")
        plt.gcf().clear()

        # # -----> Errors as a function of time
        # error_p = np.asarray([self.sol[t]["error_pressure"] for t in self.tsc.schedule])
        # error_u = np.asarray(
        #     [self.sol[t]["error_displacement"] for t in self.tsc.schedule])
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
