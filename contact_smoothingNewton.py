import porepy as pp
import numpy as np
from functools import partial

# Here I will make a first attempt at replacing the normal and tangential complementary
# functions by smooth functions.


class ModifiedGeometry:
    # Note: Method which creates the grid does not have to be modified; it
    # automatically creates the mesh with the fracture network and domain bounds
    # ascribed in set_fracture_network.

    def set_fracture_network(self) -> None:
        # # Three intersecting fractures.
        points = np.array([[0.4, 0.5], [1.6, 0.5],
                           [0.4, 0.3], [1.6, 0.7],
                           [0.7, 0.3], [0.7, 0.7]
                           ]).T
        edges = np.array([[0, 1], [2, 3], [4, 5]]).T
        # points = np.array([[0.4, 0.5], [1.2, 0.5], [0.7, 0.3], [0.7, 0.7]]).T
        # edges = np.array([[0, 1], [2, 3]]).T
        # points = np.array([[0.4, 0.5], [1.6, 0.5],
        #                    [0.4, 0.3], [1.6, 0.7]
        #                    ]).T
        # edges = np.array([[0, 1], [2, 3]]).T
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
        return bc

    def bc_values_mechanics(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        values = []
        for sd in subdomains:
            all_bf, east, west, north, south, _, _ = self.domain_boundary_sides(sd)
            val_loc = np.zeros((self.nd, sd.num_faces))
            # See section on scaling for explanation of the conversion.
            val_loc[0, east] = -0.001
            val_loc[1, north] = -0.001
            values.append(val_loc)
        values = np.array(values)
        values = values.ravel("F")
        return pp.wrap_as_ad_array(values, name="bc_vals_mechanics")


class Regularized:
    """Mixin that regularizes the normal and/or tangential complementary equations."""

    def contact_mechanics_numerical_constant(
            self, subdomains: list[pp.Grid]
    ) -> pp.ad.Scalar:
        """Penalty parameter to be used in the normal and tangential equations below"""
        val = self.solid.convert_units(1e3, "-")
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

        # Smoothed complimentary function
        b = (-1) * t_n - self.contact_mechanics_numerical_constant(subdomains) \
            * (u_n - self.gap(subdomains))
        # mu = pp.ad.Scalar(0.1)
        mu_n = pp.ad.Array(0.01 * np.ones(num_cells), "smoothing_parameter_normal")

        equation: pp.ad.Operator = t_n + (0.5 * b + 0.5 * ((b ** 2) + (mu_n ** 2))**0.5)

        equation.set_name("normal_fracture_deformation_equation")
        return equation

    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:

        # Basis vector combinations
        num_cells = sum([sd.num_cells for sd in subdomains])
        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        tangential_basis = self.basis(
            subdomains, dim=self.nd - 1  # type: ignore[call-arg]
        )

        scalar_to_tangential = sum([e_i for e_i in tangential_basis])

        t_t: pp.ad.Operator = nd_vec_to_tangential * self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential * self.displacement_jump(subdomains)
        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Vectors needed to express the governing equations
        ones_frac = pp.ad.Array(np.ones(num_cells * (self.nd - 1)))
        zeros_frac = pp.ad.Array(np.zeros(num_cells))

        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)

        c_num = sum([e_i * c_num_as_scalar * e_i.T for e_i in tangential_basis])

        mu_scalar = 0.01

        tangential_sum = t_t + c_num * u_t_increment

        tangential_sum_mu = np.append(tangential_sum, mu_scalar)

        mu_t = pp.ad.Array(mu_scalar * np.ones(num_cells),
                           "smoothing_parameter_tangential")

        # norm_tangential_sum = f_norm(tangential_sum)
        # At the moment, I do not regularize the norm function, because I don't know
        # how.
        reg_norm_tangential_sum = f_norm(tangential_sum)
        reg_norm_tangential_sum.set_name("norm_tangential")
        # norm_tangential_sum.set_name("norm_tangential")

        b_p = self.friction_bound(subdomains)
        b_p.set_name("bp")

        equation: pp.ad.Operator = (0.5 * (b_p + reg_norm_tangential_sum) +
                                    0.5 * ((b_p - reg_norm_tangential_sum)**2 +
                                    mu_t**2)**0.5) * (-1) * t_t + tangential_sum * (
            0.5 * b_p + 0.5 * (b_p**2 + mu_t**2)**0.5
        )

        equation.set_name("tangential_fracture_deformation_equation")
        return equation

    # def tangential_fracture_deformation_equation(
    #     self,
    #     subdomains: list[pp.Grid],
    # ) -> pp.ad.Operator:
    #
    #     # Basis vector combinations
    #     num_cells = sum([sd.num_cells for sd in subdomains])
    #     # Mapping from a full vector to the tangential component
    #     nd_vec_to_tangential = self.tangential_component(subdomains)
    #
    #     tangential_basis = self.basis(
    #         subdomains, dim=self.nd - 1  # type: ignore[call-arg]
    #     )
    #
    #     scalar_to_tangential = sum([e_i for e_i in tangential_basis])
    #
    #     t_t: pp.ad.Operator = nd_vec_to_tangential * self.contact_traction(subdomains)
    #     u_t: pp.ad.Operator = nd_vec_to_tangential * self.displacement_jump(subdomains)
    #     # The time increment of the tangential displacement jump
    #     u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)
    #
    #     # Vectors needed to express the governing equations
    #     ones_frac = pp.ad.Array(np.ones(num_cells * (self.nd - 1)))
    #     zeros_frac = pp.ad.Array(np.zeros(num_cells))
    #
    #     # Functions EK: Should we try to agree on a name convention for ad functions?
    #     # EK: Yes. Suggestions?
    #     f_max = pp.ad.Function(pp.ad.maximum, "max_function")
    #     f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
    #
    #     tol = 1e-5  # FIXME: Revisit this tolerance!
    #
    #     f_characteristic = pp.ad.Function(
    #         partial(pp.ad.functions.characteristic_function, tol),
    #         "characteristic_function_for_zero_normal_traction",
    #     )
    #
    #     c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)
    #
    #     c_num = sum([e_i * c_num_as_scalar * e_i.T for e_i in tangential_basis])
    #
    #     # Version for the standard equation.
    #     # tangential_sum = t_t + c_num * u_t_increment
    #
    #     # # Version for the regularized equation.
    #     tangential_sum = c_num * u_t_increment
    #
    #     norm_tangential_sum = f_norm(tangential_sum)
    #     norm_tangential_sum.set_name("norm_tangential")
    #
    #     b_p = f_max(self.friction_bound(subdomains), zeros_frac)
    #     b_p.set_name("bp")
    #
    #     # Remove parentheses to make the equation more readable if possible
    #     bp_tang = (scalar_to_tangential * b_p) * tangential_sum
    #
    #     maxbp_abs = scalar_to_tangential * f_max(b_p, norm_tangential_sum)
    #     characteristic: pp.ad.Operator = scalar_to_tangential * f_characteristic(b_p)
    #     characteristic.set_name("characteristic_function_of_b_p")
    #
    #     equation: pp.ad.Operator = (ones_frac - characteristic) * (
    #         bp_tang - maxbp_abs * t_t
    #     ) + characteristic * t_t
    #     equation.set_name("tangential_fracture_deformation_equation")
    #     return equation


class ModifiedMomentumBalance(ModifiedGeometry,
                              ModifiedBoundaryConditions,
                              pp.momentum_balance.MomentumBalance):
    pass


# params = {} # Possibly add something
params = {"max_iterations": 30,
          "nl_convergence_tol": 1e-10,
          "nl_divergence_tol": 1e5,
          }
model = ModifiedMomentumBalance(params)
pp.run_stationary_model(model, params)
displacement = model.equation_system.get_variable_values(["u"])
# pp.plot_grid(model.mdg, vector_value=model.displacement_variable)
# print(model.equation_system.get_variable_values(["t"]))


class RegularizedMomentumBalance(ModifiedGeometry,
                                 ModifiedBoundaryConditions,
                                 Regularized,
                                 pp.momentum_balance.MomentumBalance):
    pass


model_reg = RegularizedMomentumBalance(params)
pp.run_stationary_model(model_reg, params)
# print(model_reg.equation_system.get_variable_values(["u"]))
displacement2 = model_reg.equation_system.get_variable_values(["u"])
# pp.plot_grid(model_reg.mdg, vector_value=model_reg.displacement_variable)

print(np.linalg.norm(displacement2-displacement))
