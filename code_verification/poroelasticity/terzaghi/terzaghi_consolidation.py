import numpy as np
import porepy as pp

from terzaghi_exact import ExactTerzaghi


#%% Inherit model ContactMechanicsBiot class
class Terzaghi(pp.ContactMechanicsBiot):
    """Parent class for Terzaghi's consolidation problem model.

    """

    def __init__(self, params: dict):
        super().__init__(params)

    def create_grid(self) -> None:
        """Create a Cartesian grid with 10 horizontal cells and 40 vertical cells."""
        phys_dims = np.array([1, 1])
        n_cells = np.array([10, 40])
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
        all_bc, east, west, south, north, _, _ = self._domain_boundary_sides(sd)
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
        # Define boundary regions
        all_bc, east, west, south, north, _, _ = self._domain_boundary_sides(sd)

        # Create a BoundaryConditionVectorial object setting all bc faces as Dirichlet type
        pp.BoundaryConditionVectorial(sd, faces=all_bc, cond="dir")
        data = self.mdg.subdomain_data(sd)
        bc = data[pp.PARAMETERS][self.mechanics_parameter_key]["bc"]

        # North side: Neumann
        bc.is_neu[:, north] = True
        bc.is_dir[:, north] = False

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

    def _bc_values_scalar(self, sd: pp.Grid) -> np.ndarray:
        """Set boundary condition values for the flow subproblem."""
        _, _, _, _, north, _, _ = self._domain_boundary_sides(sd)
        bc_values = np.zeros(sd.num_faces)
        bc_values[np.where(north)[0]] = self.params["vertical_load"]
        return bc_values

    def _bc_values_mechanics(self, sd: pp.Grid) -> np.ndarray:
        """Set boundary condition values for the mechanics subproblem."""
        _, _, _, _, north, _, _ = self._domain_boundary_sides(sd)
        bc_values = np.zeros(sd.dim * sd.num_faces)
        bc_values[sd.dim * np.where(north)[0] + 1] = - (
                self.params["vertical_load"] * sd.face_areas[np.where(north)[0]]
        )
        return bc_values

    def _permeability(self, sd: pp.Grid) -> np.ndarray:
        """Set intrinsic permeability for the flow subproblem"""
        return 0.001 * np.ones(sd.num_cells)

    def _specific_volume(self, sd: pp.Grid) -> np.ndarray:



#%% Main script
model_params = {}
model = Terzaghi(model_params)
model.params["consolidation_coefficient"] = 1
model.params["vertical_load"] = 1
model.prepare_simulation()

sd = model.mdg.subdomains()[0]
data = model.mdg.subdomain_data(sd)
bc = data[pp.PARAMETERS][model.mechanics_parameter_key]["bc"]

#%% Exact solution
ex = ExactTerzaghi(terzaghi_model=model)



