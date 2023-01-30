import porepy as pp
import numpy as np


class ModifiedGeometry(pp.ModelGeometry):

    # Note: Method which creates the grid does not have to be modified; it
    # automatically creates the mesh with the fracture network and domain bounds
    # ascribed in set_fracture_network.

    def set_fracture_network(self) -> None:
        # Two intersecting fractures.
        points = np.array([[0.4, 0.5], [1.6, 0.5], [0.4, 0.3], [1.6, 0.7]]).T
        edges = np.array([[0, 1], [2, 3]]).T
        domain = {"xmin": 0, "ymin": 0, "xmax": 2, "ymax": 1}
        self.fracture_network = pp.FractureNetwork2d(points, edges, domain)

    def mesh_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"mesh_size_frac": 0.2}
        return mesh_args


class ModifiedBoundaryConditions:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        all_bf, east, west, north, south, _, _ = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, all_bf, "dir")
        frac_face = sd.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        # Neumann conditions on east and north faces.
        bc.is_dir[:, east + north] = False
        bc.is_neu[:, east + north] = True
        # Not sure if there is a more elegant way than to set these Neumann
        # conditions manually.
        return bc

    def bc_values_mechanics(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        values = []
        for sd in subdomains:
            all_bf, east, west, north, south, _, _ = self.domain_boundary_sides(sd)
            val_loc = np.zeros((self.nd, sd.num_faces))
            # See section on scaling for explanation of the conversion.
            val_loc[0, east] = -0.1
            val_loc[1, north] = -0.1
            values.append(val_loc)
        values = np.array(values)
        values = values.ravel("F")
        return pp.wrap_as_ad_array(values, name="bc_vals_mechanics")
        

class ModifiedMomentumBalance(ModifiedGeometry,
                              ModifiedBoundaryConditions,
                              pp.momentum_balance.MomentumBalance):
    pass


params = {} # Possibly add something
model = ModifiedMomentumBalance(params)
pp.run_stationary_model(model, params)
pp.plot_grid(model.mdg, vector_value=model.displacement_variable)
# inds = model.dof_manager.dof_var(var=[model.displacement_variable])
# vals = model.dof_manager.assemble_variable(variables=[model.displacement_variable])
# displacement = vals[inds]
# print(displacement)
