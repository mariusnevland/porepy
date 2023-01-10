import porepy as pp
import numpy as np

class ContactProblem(pp.ContactMechanics):
    """A ContactMechanics model with modified grid and
    changed things."""

    def create_grid(self) -> None:
        mesh_args = self.params.get("mesh_args", {"mesh_size_frac": 5})
        endp = np.array([.2, .5])
        xendp = np.array([.2, .8])
        yendp= np.array([.2, 1])
        self.mdg, self.box = pp.md_grids_2d.single_vertical(mesh_args, endp,simplex=True)
        # self.mdg, self.box = pp.md_grids_2d.two_intersecting(mesh_args, xendp,yendp,
        # simplex=True)
        # self.mdg, self.box = pp.md_grids_2d.seven_fractures_one_L_intersection(
        # mesh_args)
        pp.contact_conditions.set_projections(self.mdg)

    # Attempt mixed Dirichlet and Neumann conditions.
    def _bc_type(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        all_bf, east, west, north, south, _, _ = self._domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, all_bf, "dir")
        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the
        # fracture faces.
        frac_face = sd.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        # Neumann conditions on east and north faces.
        bc.is_dir[:, east + north] = False
        bc.is_neu[:, east + north] = True
        return bc

    def _bc_values(self, sd: pp.Grid) -> np.ndarray:
        # Values for all Nd components, face-wise
        _, east, west, north, south, _, _ = self._domain_boundary_sides(sd)
        values = np.zeros((self.nd, sd.num_faces))
        # Tractions on north and east faces.
        values[0, east] = 1
        values[1, north] = -1
        # Reshape according to PorePy convention
        values = values.ravel("F")
        return values

    def _friction_coefficient(self, sd: pp.Grid) -> np.ndarray:
        return np.ones(sd.num_cells)


params = {"use_ad": True}
#params = {}
model = ContactProblem(params)
pp.run_stationary_model(model, params)
pp.plot_grid(model.mdg, vector_value=model.displacement_variable)
inds = model.dof_manager.dof_var(var=[model.displacement_variable])
vals = model.dof_manager.assemble_variable(variables=[model.displacement_variable])
displacement = vals[inds]
print(displacement)