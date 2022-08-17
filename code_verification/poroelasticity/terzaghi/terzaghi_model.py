"""
Implementation of Terzaghi's consolidation problem.

Even though the model is strictly speaking a one-dimensional problem, the implemented model
uses a two-dimensional unstructured grid with roller boundary conditions for the mechanical
subproblem and no-flux boundary conditions for the flow subproblem in order to emulate the
one-dimensional process.

"""

import numpy as np
import porepy as pp


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
            time_stepping_object (pp.TimeSteppingControl): Time stepping control object.

        Optional model parameters:
            mesh_size (float, Default is 0.1): Mesh size.
            upper_limit (int, Default is 1000): Upper limit of summation for computing exact
                solutions.

        """
        super().__init__(params)

        self.tsc = self.params["time_stepping_object"]
        self.time = self.tsc.time_init
        self.end_time = self.tsc.time_final
        self.time_step = self.tsc.dt

        # Create a solution dictionary to store pressure and displacement solutions
        self.sol = {t: {} for t in self.tsc.schedule}

    def create_grid(self) -> None:
        """Create two-dimensional unstructured mixed-dimensional grid."""
        height = self.params["height"]
        mesh_size = self.params.get("mesh_size", 0.1)
        self.box = {"xmin": 0.0, "xmax": height, "ymin": 0.0, "ymax": height}
        network_2d = pp.FractureNetwork2d(None, None, self.box)
        mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}
        self.mdg = network_2d.mesh(mesh_args)

    def _initial_condition(self) -> None:
        """Override initial condition for the flow subproblem."""
        super()._initial_condition()
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        vertical_load = self.params["vertical_load"]
        initial_p = vertical_load * np.ones(sd.num_cells)
        data[pp.STATE][self.scalar_variable] = initial_p
        data[pp.STATE][pp.ITERATE][self.scalar_variable] = initial_p

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
        vertical_load = self.params["vertical_load"]
        bc_values = np.array([np.zeros(sd.num_faces), np.zeros(sd.num_faces)])
        bc_values[1, north] = -vertical_load * sd.face_areas[north]
        bc_values = bc_values.ravel("F")

        return bc_values

    def _permeability(self, sd: pp.Grid) -> np.ndarray:
        """Overried value of intrinsic permeability [m^2].

        Args:
            sd: Subdomain grid.

        Returns:
            permeability (sd.num_cells): containing the permeability values at each cell.

        """

        permeability = 0.001 * np.ones(sd.num_cells)

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

    def before_newton_loop(self):
        super().before_newton_loop()

        # Update time for the time-stepping control routine
        self.tsc.time += self.time_step

    def after_newton_convergence(
        self,
        solution: np.ndarray,
        errors: float,
        iteration_counter: int,
    ) -> None:
        super().after_newton_convergence(solution, errors, iteration_counter)

        # Adjust time step
        self.time_step = self.tsc.next_time_step(1, recompute_solution=False)
        self._ad.time_step._value = self.time_step

        # Store solutions
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        if self.time in self.tsc.schedule[1:]:
            self.sol[self.time]["u_num"] = data[pp.STATE][self.displacement_variable]
            self.sol[self.time]["p_num"] = data[pp.STATE][self.scalar_variable]
