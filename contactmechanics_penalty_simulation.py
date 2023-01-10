import porepy as pp
import numpy as np
import inspect


class ContactProblem(pp.ContactMechanics):
    """A ContactMechanics model with modified grid and non-homogeneous boundary data."""

    def create_grid(self) -> None:

        mesh_args = self.params.get("mesh_args", {"mesh_size_frac": 1})
        endp = np.array([.2, .5])
        xendp = np.array([.2, .8])
        yendp= np.array([.2, 1])
        self.mdg, self.box = pp.md_grids_2d.single_vertical(mesh_args, endp,simplex=True)
        pp.contact_conditions.set_projections(self.mdg)

    def _bc_values(self, sd: pp.Grid) -> np.ndarray:
        # Values for all Nd components, face-wise
        values = np.zeros((self.nd, sd.num_faces))
        # This puts displacement of .5 at the top boundary, in the y-direction.
        values[1,16]=-1
        values[1,18]=-1
        # Reshape according to PorePy convention
        values = values.ravel("F")
        return values


params = {"use_ad": True}
model = ContactProblem(params)
pp.run_stationary_model(model, params)
#pp.plot_grid(model.mdg, vector_value=model.displacement_variable)
inds = model.dof_manager.dof_var(var=[model.displacement_variable])
vals = model.dof_manager.assemble_variable(variables=[model.displacement_variable])
displacement = vals[inds]


class RegularizedContact(ContactProblem):
    """Solve the contact mechanics problem with a regularized normal complementary
    equation."""
    def _numerical_constants(self, sd: pp.Grid) -> tuple[np.ndarray, np.ndarray]:
        penalty_parameter = 1e4
        c_num_n = penalty_parameter * np.ones(sd.num_cells)
        # Expand to tangential Nd-1 vector
        tangential_vals = np.ones(sd.num_cells)
        c_num_t = np.kron(tangential_vals, np.ones(sd.dim))
        return c_num_n, c_num_t

    def _contact_mechanics_normal_equation(
        self,
        fracture_subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:

        numerical_c_n = pp.ad.ParameterMatrix(
            self.mechanics_parameter_key,
            array_keyword="c_num_normal",
            subdomains=fracture_subdomains,
        )
        T_n: pp.ad.Operator = self._ad.normal_component_frac * self._ad.contact_traction

        MaxAd = pp.ad.Function(pp.ad.maximum, "max_function")
        zeros_frac = pp.ad.Array(np.zeros(self._num_frac_cells))
        u_n: pp.ad.Operator = self._ad.normal_component_frac * self._displacement_jump(
            fracture_subdomains
        )
        equation: pp.ad.Operator = T_n + MaxAd(
            (-1) * 0 - numerical_c_n * (u_n - self._gap(fracture_subdomains)),
            zeros_frac,
        )
        return equation


params = {"use_ad": True}
model2 = RegularizedContact(params)
pp.run_stationary_model(model2,params)
inds = model2.dof_manager.dof_var(var=[model2.displacement_variable])
vals = model2.dof_manager.assemble_variable(variables=[model2.displacement_variable])
displacement2 = vals[inds]

print(np.linalg.norm(displacement2-displacement))
