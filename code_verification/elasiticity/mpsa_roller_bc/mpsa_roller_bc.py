"""
Implementation of a simple two-dimensional elastic problem with roller boundary conditions.
"""

import numpy as np
import porepy as pp


class SimpleElastic(pp.ContactMechanics):
    """Parent class implementing MPSA on a two-dimensional grid"""

    def __init__(self, params: dict):
        super().__init__(params)

    def create_grid(self) -> None:
        """Create mixed-dimensional grid"""
        if model.params["is_cartesian"]:
            phys_dims = np.array([1, 1])
            n_cells = np.array([10, 10])
            self.box = pp.geometry.bounding_box.from_points(
                np.array([[0, 0], phys_dims]).T
            )
            sd: pp.Grid = pp.CartGrid(n_cells, phys_dims)
            sd.compute_geometry()
            self.mdg = pp.meshing.subdomains_to_mdg([[sd]])
        else:
            mesh_size = 0.1
            self.box = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
            network_2d = pp.FractureNetwork2d(None, None, self.box)
            mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}
            self.mdg = network_2d.mesh(mesh_args)

    def _bc_type(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Set boundary condition type.

        Args:
            sd: Subdomain grid.

        Returns:
            bc: Vectorial boundary condition representation.

        """

        # It is easier to set all boundaries as Dirichlet and then modify accordingly
        all_bf = sd.get_boundary_faces()
        bc = pp.BoundaryConditionVectorial(sd, all_bf, "dir")
        _, east, west, north, south, *_ = self._domain_boundary_sides(sd)

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

    def _bc_values(self, sd: pp.Grid) -> np.ndarray:
        """Set boundary condition values.

        Args:
            sd: Subdomain grid.

        Returns:
            bc_values (sd.dim * sd.num_faces): Containing the boundary condition values.

        """

        # Retrieve boundary sides
        _, _, _, north, *_ = self._domain_boundary_sides(sd)

        # All zeros except vertical component of the north side
        vertical_load = 1  # value of the vertically applied force on the north boundary
        bc_values = np.array([np.zeros(sd.num_faces), np.zeros(sd.num_faces)])
        bc_values[1, north] = -vertical_load * sd.face_areas[north]
        bc_values = bc_values.ravel("F")

        return bc_values


#%% Runscript

# Define model parameters
model_params = {
    "use_ad": True,  # this we don't usually touch
    "is_cartesian": False,  # use False for simplicial unstructured grid
}

# Create model
model = SimpleElastic(model_params)

# Run model
pp.run_stationary_model(model, model_params)

# Plot solution
sd = model.mdg.subdomains()[0]
data = model.mdg.subdomain_data(sd)
u = data[pp.STATE][model.displacement_variable]
u_mag = (u[::2] ** 2 + u[1::2] ** 2) ** 0.5
pp.plot_grid(sd, u_mag, plot_2d=True, title=r"$|\vec{u}(x,y)|$")
