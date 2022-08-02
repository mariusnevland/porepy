import numpy as np
import porepy as pp

from terzaghi_exact import ExactTerzaghi
from typing import Union


#%% Inherit model ContactMechanicsBiot class
class Terzaghi(pp.ContactMechanicsBiot):
    """Parent class for Terzaghi's consolidation problem model.

    """

    def __init__(self, params: dict):
        super().__init__(params)

    def create_grid(self) -> None:
        """Create a Cartesian grid with 10 horizontal cells and 40 vertical cells."""
        phys_dims = np.array([1, 1])
        n_cells = np.array([10, 30])
        self.box = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
        sd: pp.Grid = pp.CartGrid(n_cells, phys_dims)
        sd.compute_geometry()
        self.mdg = pp.meshing.subdomains_to_mdg([[sd]])

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Define boundary condition types for the flow subproblem.

        Args:
            sd: Subdomain grid.

        Returns:
            Scalar boundary condition representation.

        """

        # Define boundary regions
        all_bc, _, _, north, *_ = self._domain_boundary_sides(sd)
        north_bc = np.isin(all_bc, np.where(north)).nonzero()

        # All sides Neumann, except the North which is Dirichlet
        bc_type = np.asarray(all_bc.size * ["neu"])
        bc_type[north_bc] = "dir"

        return pp.BoundaryCondition(sd, faces=all_bc, cond=bc_type)

    def _bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define boundary condition types for the mechanics subproblem

        Args:
            sd: Subdomain grid.

        Returns:
            Vectorial boundary condition representation.

        """
        # Inherit bc from parent class. This sets all bc faces as Dirichlet.
        super()._bc_type_mechanics(sd=sd)

        # Define boundary regions, retrieve data dict, and bc object.
        _, east, west, north, south, *_ = self._domain_boundary_sides(sd)
        data = self.mdg.subdomain_data(sd)
        bc = data[pp.PARAMETERS][self.mechanics_parameter_key]["bc"]

        # East side: Roller
        bc.is_neu[0, east] = False
        bc.is_dir[0, east] = True
        bc.is_neu[1, east] = True
        bc.is_dir[1, east] = False

        # West side: Roller
        bc.is_neu[0, west] = False
        bc.is_dir[0, west] = True
        bc.is_neu[1, west] = True
        bc.is_dir[1, west] = False

        # North side: Neumann
        bc.is_neu[:, north] = True
        bc.is_dir[:, north] = False

        # South side: Roller
        bc.is_neu[0, south] = True
        bc.is_dir[0, south] = False
        bc.is_neu[1, south] = False
        bc.is_dir[1, south] = True

        return bc

    # def _bc_values_scalar(self, sd: pp.Grid) -> np.ndarray:
    #     """Set boundary condition values for the flow subproblem."""
    #     _, _, _, north, *_ = self._domain_boundary_sides(sd)
    #     bc_values = np.zeros(sd.num_faces)
    #     bc_values[north] = self.params["vertical_load"]
    #     return bc_values

    def _bc_values_mechanics(self, sd: pp.Grid) -> np.ndarray:
        """Set boundary condition values for the mechanics subproblem."""
        _, _, _, north, *_ = self._domain_boundary_sides(sd)
        bc_values = np.array([np.zeros(sd.num_faces), np.zeros(sd.num_faces)])
        bc_values[1, north] = - self.params["vertical_load"] * sd.face_areas[north]
        return bc_values.ravel("F")

    def _permeability(self, sd: pp.Grid) -> np.ndarray:
        """Set intrinsic permeability for the flow subproblem"""
        return 0.001 * np.ones(sd.num_cells)

    def _storativity(self, sd: pp.Grid) -> np.ndarray:
        """Zero storativity in Terzaghi's model"""
        return np.ones(sd.num_cells)

    def before_newton_loop(self) -> None:
        """Modify default time step"""
        if 0 < self.time <= 0.2:
            self.time_step = 0.05
        elif 0.2 < self.time <= 0.5:
            self.time_step = 0.1
        elif 0.5 < self.time <= 2.25:
            self.time_step = 0.25
        elif 2.25 < self.time <= 8.0:
            self.time_step = 0.5
        else:
            self.time_step = 1.0


#%% Main script
model_params = {
    "use_ad": True,
    "time_step": 0.05,
    "end_time": 0.05,
    "consolidation_coefficient": 1.0,
    "vertical_load": 1.0
    }
model = Terzaghi(model_params)
model.prepare_simulation()
# pp.run_time_dependent_model(model, model_params)

#%%
sd = model.mdg.subdomains()[0]
data = model.mdg.subdomain_data(sd)
all_bc, east, west, north, south, _, _ = model._domain_boundary_sides(sd)
bc_values = data[pp.PARAMETERS][model.mechanics_parameter_key]["bc_values"]
bc_values_x = bc_values[::2]
bc_values_y = bc_values[1::2]
bc = data[pp.PARAMETERS][model.mechanics_parameter_key]["bc"]


# #%% Plot results
# sd = model.mdg.subdomains()[0]
# data = model.mdg.subdomain_data(sd)
# p = data[pp.STATE]["p"]
# pp.plot_grid(sd, p, plot_2d=True)
#
# #%%
#
#
# #%% Exact solution
# ex = ExactTerzaghi(terzaghi_model=model)



