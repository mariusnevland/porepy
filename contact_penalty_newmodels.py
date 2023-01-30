import porepy as pp
import numpy as np


class ModifiedGeometry:

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
        mesh_args: dict[str, float] = {"mesh_size_frac": 2}
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


class Regularized:
    """Mixin that regularizes the normal and/or tangential complementary equations."""

    def contact_mechanics_numerical_constant(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Scalar:
        """Penalty parameter to be used in the normal equation below"""
        val = self.solid.convert_units(1e5, "-")
        return pp.ad.Scalar(val, name="c_num")

    def normal_fracture_deformation_equation(self, subdomains: list[pp.Grid]):
        nd_vec_to_normal = self.normal_component(subdomains)
        # The normal component of the contact traction and the displacement jump
        t_n: pp.ad.Operator = nd_vec_to_normal * self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal * self.displacement_jump(subdomains)

        # Maximum function
        num_cells: int = sum([sd.num_cells for sd in subdomains])
        max_function = pp.ad.Function(pp.ad.maximum, "max_function")
        zeros_frac = pp.ad.Array(np.zeros(num_cells), "zeros_frac")

        # The complimentarity condition
        equation: pp.ad.Operator = t_n + max_function(
            (-1) * 0
            - self.contact_mechanics_numerical_constant(subdomains)
            * (u_n - self.gap(subdomains)),
            zeros_frac,
        )
        equation.set_name("normal_fracture_deformation_equation")
        return equation
        

class ModifiedMomentumBalance(ModifiedGeometry,
                              ModifiedBoundaryConditions,
                              pp.momentum_balance.MomentumBalance):
    pass


params = {} # Possibly add something
model = ModifiedMomentumBalance(params)
pp.run_stationary_model(model, params)
displacement = model.equation_system.get_variable_values(["u"])
# pp.plot_grid(model.mdg, vector_value=model.displacement_variable)
# print(model.equation_system.get_variable_values(["u"]))


class RegularizedMomentumBalance(ModifiedGeometry,
                                 ModifiedBoundaryConditions,
                                 Regularized,
                                 pp.momentum_balance.MomentumBalance):
    pass


model_reg = RegularizedMomentumBalance(params)
pp.run_stationary_model(model_reg, params)
displacement2 = model_reg.equation_system.get_variable_values(["u"])

print(np.linalg.norm(displacement2-displacement))


