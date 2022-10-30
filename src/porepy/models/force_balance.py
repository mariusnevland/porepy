"""
Class types:
    Generic VectorBalanceEquation
    Specific ForceBalanceEquations defines subdomain and interface equations through the
        terms entering. Force balance between opposing fracture interfaces is imposed.
    TODO: Specific ConstitutiveEquations and
    TODO: specific SolutionStrategy

Notes:
    - The class ForceBalanceEquations is a mixin class, and should be inherited by a class
        that defines the variables and discretization.

    - Refactoring needed for constitutive equations. Modularisation and moving to the library.

"""

from __future__ import annotations

import logging
from functools import partial
from typing import Dict, Optional, Union

import constit_library
import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy import ModelGeometry

logger = logging.getLogger(__name__)


class VectorBalanceEquation:
    """Generic class for scalar balance equations on the form

    d_t(accumulation) + div(flux) - source = 0

    All terms need to be specified in order to define an equation.
    """

    def balance_equation(
        self, subdomains: list[pp.Grid], accumulation, stress, source
    ) -> pp.ad.Operator:
        """Balance equation for a vector variable.

        Args:
            subdomains: List of subdomains where the balance equation is defined.
            accumulation: Operator for the accumulation term.
            stress: Operator for the stress term.
            source: Operator for the source term.

        Returns:
            Operator for the balance equation.
        """

        dt = self.time_increment_method
        div = pp.ad.Divergence(subdomains, nd=self.nd)
        return dt(accumulation) + div * stress - source

    def volume_integral(
        self,
        integrand: pp.ad.Operator,
        grids: Union[list[pp.Grid], list[pp.MortarGrid]],
    ) -> pp.ad.Operator:
        """Numerical volume integral over subdomain or interface cells.

        Includes cell volumes and specific volume.

        Args:
            integrand: Operator for the integrand.
            grids: List of subdomain or interface grids over which the integral is to be
                computed.

        Returns:
            Operator for the volume integral.

        """
        geometry = pp.ad.Geometry(grids, nd=self.nd)
        # First factor expands from scalar to vector.
        vol = geometry.scalar_to_nd_cell * (
            geometry.cell_volumes * self.specific_volume(grids)
        )
        return vol * integrand


class ForceBalanceEquations(VectorBalanceEquation):
    """Class for force balance equations and fracture deformation equations."""

    def set_equations(self):
        """Set equations for the subdomains and interfaces.

        The following equations are set:
            - Force balance in the matrix.
            - Force balance between fracture interfaces.
            - Deformation constraints for fractures, split into normal and tangential part.
        See individual equation methods for details.
        """
        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        fracture_subdomains = self.mdg.subdomains(dim=self.nd - 1)
        interfaces = self.mdg.interfaces(dim=self.nd - 1)
        matrix_eq = self.matrix_force_balance_equation(matrix_subdomains)
        # We split the fracture deformation equations into two parts, for the normal and
        # tangential components for convenience.
        fracture_eq_normal = self.normal_fracture_deformation_equation(
            fracture_subdomains
        )
        fracture_eq_tangential = self.tangential_fracture_deformation_equation(
            fracture_subdomains
        )
        intf_eq = self.interface_force_balance_equation(interfaces)
        self.system_manager.set_equation(
            matrix_eq, (matrix_subdomains, "cells", self.nd)
        )
        self.system_manager.set_equation(
            fracture_eq_normal, (fracture_subdomains, "cells", 1)
        )
        self.system_manager.set_equation(
            fracture_eq_tangential, (fracture_subdomains, "cells", self.nd - 1)
        )
        self.system_manager.set_equation(intf_eq, (interfaces, "cells", self.nd))

    def matrix_force_balance_equation(self, subdomains: list[pp.Grid]):
        """Force balance equation in the matrix.

        Inertial term is not included.

        Args:
            subdomains: List of subdomains where the force balance is defined. Only known usage
                is for the matrix domain(s).

        Returns:
            Operator for the force balance equation in the matrix.

        """
        accumulation = 0
        stress = self.stress(subdomains)
        body_force = self.body_force(subdomains)
        return self.balance_equation(subdomains, accumulation, stress, body_force)

    def interface_force_balance_equation(
        self,
        interfaces: list[pp.MortarGrid],
    ) -> pp.ad.Operator:
        """Force balance equation at matrix-fracture interfaces.

        Args:
            interfaces: Fracture-matrix interfaces.

        Returns:
            Operator representing the force balance equation.

        """
        # Check that the interface is a fracture-matrix interface.
        for interface in interfaces:
            if interface.dim != self.nd - 1:
                raise ValueError("Interface must be a fracture-matrix interface.")
        subdomains = self.subdomains_from_interfaces(interfaces)
        # Split into matrix and fractures. Sort on dimension to allow for multiple matrix
        # domains. Otherwise, we could have picked the first element.
        matrix_subdomains = [sd for sd in subdomains if sd.dim == self.nd]
        fracture_subdomains = [sd for sd in subdomains if sd.dim == self.nd - 1]

        # Geometry related
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )
        geometry = pp.ad.Geometry(interfaces, nd=self.nd)
        volume_int = geometry.scalar_to_nd_cell * geometry.cell_volumes
        # Contact traction from primary grid and mortar displacements (via primary grid)
        contact_from_primary_mortar = (
            mortar_projection.primary_to_mortar_int
            * self.subdomain_projections(self.nd).face_prolongation(matrix_subdomains)
            * self._internal_boundary_normal_to_outwards(interfaces)
            * self.stress(matrix_subdomains)
        )
        # Traction from the actual contact force.
        contact_from_secondary = (
            volume_int
            * mortar_projection.sign_of_mortar_sides
            * mortar_projection.secondary_to_mortar_int
            * self.subdomain_projections(self.nd).cell_prolongation(fracture_subdomains)
            * self.local_coordinates(subdomains).transpose()
            * self.traction(subdomains)  # Using traction instead of force. Compensating
            # by including volume integral (on interfaces, as I think is proper).
        )
        force_balance_eq: pp.ad.Operator = (
            contact_from_primary_mortar + contact_from_secondary
        )
        return force_balance_eq

    def normal_fracture_deformation_equation(self, subdomains: list[pp.Grid]):
        """Equation for the normal component of the fracture deformation.

        This constraint equation enforces non-penetration of opposing fracture interfaces.


        Args:
            subdomains: List of subdomains where the normal deformation equation is defined.

        Returns:
            Operator for the normal deformation equation.

        """
        # Variables
        nd_vec_to_normal = self.normal_component(subdomains)
        t_n: pp.ad.Operator = nd_vec_to_normal * self.traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal * self.displacement_jump(subdomains)

        # Maximum function
        num_cells: int = sum([sd.num_cells for sd in subdomains])
        max_function = pp.ad.Function(pp.ad.maximum, "max_function")
        zeros_frac = pp.ad.Array(np.zeros(num_cells), "zeros_frac")

        equation: pp.ad.Operator = t_n + max_function(
            (-1) * t_n
            - self.numerical_constant(subdomains) * (u_n - self.gap(subdomains)),
            zeros_frac,
        )
        return equation

    def contact_mechanics_tangential_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """
        Contact mechanics equation for the tangential constraints.

        The function reads
            C_t = max(b_p, ||T_t+c_t u_t||) T_t - max(0, b_p) (T_t+c_t u_t)
        with u being displacement jump increments, t denoting tangential
        component and b_p the friction bound.

        For b_p = 0, the equation C_t = 0 does not in itself imply T_t = 0,
        which is what the contact conditions require. The case is handled
        through the use of a characteristic function.

        Parameters
        ----------
        fracture_subdomains : List[pp.Grid]
            List of fracture subdomains.

        Returns
        -------
        complementary_eq : pp.ad.Operator
            Contact mechanics equation for the tangential constraints.

        """
        # Basis vector combinations
        geometry = pp.ad.Geometry(subdomains, nd=self.nd)
        nd_vec_to_tangential = self.tangential_component(subdomains)
        nd_vec_to_normal = self.normal_component(subdomains)
        scalar_to_tangential = sum([geometry.e_i(i) for i in range(self.nd - 1)])
        # Variables
        t_t: pp.ad.Operator = nd_vec_to_tangential * self.traction(subdomains)
        t_n: pp.ad.Operator = nd_vec_to_normal * self.traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential * self.displacement_jump(subdomains)
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        ones_frac = pp.ad.Array(np.ones(self._num_frac_cells * (self.nd - 1)))
        zeros_frac = pp.ad.Array(np.zeros(self._num_frac_cells))

        # Functions
        # EK: Should we try to agree on a name convention for ad functions?
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        tol = 1e-5  # Revisit this tolerance!
        f_characteristic = pp.ad.Function(
            partial(pp.ad.functions.characteristic_function, tol),
            "characteristic_function_for_zero_normal_traction",
        )

        # Parameters
        c_num = scalar_to_tangential * self.numerical_constant(subdomains)
        # Combine the above into expressions that enter the equation

        tangential_sum = t_t + c_num * u_t_increment

        norm_tangential_sum = f_norm(tangential_sum)
        norm_tangential_sum.set_name("norm_tangential")

        b_p = f_max(self.friction_bound(subdomains), zeros_frac)
        b_p.set_name("bp")

        # Remove parentheses to make the equation more readable if possible
        bp_tang = (scalar_to_tangential * b_p) * tangential_sum

        maxbp_abs = scalar_to_tangential * f_max(b_p, norm_tangential_sum)
        characteristic: pp.ad.Operator = scalar_to_tangential * f_characteristic(b_p)
        characteristic.set_name("characteristic_function_of_b_p")

        # Compose the equation itself.
        # The last term handles the case bound=0, in which case t_t = 0 cannot
        # be deduced from the standard version of the complementary function
        # (i.e. without the characteristic function). Filter out the other terms
        # in this case to improve convergence
        complementary_eq: pp.ad.Operator = (ones_frac - characteristic) * (
            bp_tang - maxbp_abs * t_t
        ) + characteristic * t_t
        return complementary_eq

    def body_force(self, subdomains: list[pp.Grid]):
        """Body force.
        FIXME: See FluidMassBalanceEquations.fluid_source.
        Args:
            subdomains: List of subdomains where the body force is defined.

        Returns:
            Operator for the body force.

        """
        num_cells = sum([sd.num_cells for sd in subdomains])
        vals = np.zeros(num_cells * self.nd)
        source = pp.ad.Array(vals, "body_force")
        return source

    def _internal_boundary_normal_to_outwards(
        self, interfaces: list[pp.Grid]
    ) -> pp.ad.Matrix:
        """Flip sign if normal vector points inwards.

        Args:
            interfaces: List of interfaces.

        Returns:
            Matrix with flipped signs if normal vector points inwards.

        EK: This seems a bit messy to me. Let's discuss.
        """
        if hasattr(self, "_internal_boundary_vector_to_outwards_operator"):
            return self._internal_boundary_vector_to_outwards_operator
        mat = None
        for intf in interfaces:
            # Extracting matrix for each interface should in theory allow for multiple
            # matrix subdomains, but this is not tested.
            matrix_subdomain = self.mdg.interface_to_subdomain(intf)[0]
            faces_on_fracture_surface = intf.primary_to_mortar_int().tocsr().indices
            switcher_int = pp.grid_utils.switch_sign_if_inwards_normal(
                matrix_subdomain, self.nd, faces_on_fracture_surface
            )
            if mat is None:
                mat = switcher_int
            else:
                mat += switcher_int

        outwards_mat = pp.ad.Matrix(mat)
        self._internal_boundary_vector_to_outwards_operator = outwards_mat
        return outwards_mat


class ConstitutiveEquationsForceBalance(constit_library.LinearElasticRock):
    """Class for constitutive equations for force balance equations."""

    def stress(self, subdomains: list[pp.Grid]):
        """Stress operator.

        Args:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Operator for the stress.

        """
        # Method from constitutive library's LinearElasticRock.
        return self.mechanical_stress(subdomains)

    def bc_vals_mechanics(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """
        Not sure where this one should reside.

        Args:
            subdomains:

        Returns:

        """
        num_faces = sum([sd.num_faces for sd in subdomains])
        return constit_library.ad_wrapper(
            0, True, num_faces * self.nd, "bc_vals_mechanics"
        )


class VariablesForceBalance:
    """
    Variables for mixed-dimensional force balance and fracture deformation:
        Displacement in matrix and on fracture-matrix interfaces.
        Fracture contact traction.

    .. note::
        Implementation postponed till Veljko's more convenient SystemManager is available.

    """

    def set_variables(self):
        """Set variables for the subdomains and interfaces.

        The following variables are set:
            - Displacement in the matrix.
            - Displacement on fracture-matrix interfaces.
            - Fracture contact traction.
        See individual variable methods for details.
        """
        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        fracture_subdomains = self.mdg.subdomains(dim=self.nd - 1)
        interfaces = self.mdg.interfaces(dim=self.nd - 1)
        u_matrix = self.matrix_displacement(matrix_subdomains)
        u_intf = self.interface_displacement(interfaces)
        contact_var = self.fracture_contact_traction(fracture_subdomains)
        self.system_manager.set_variable(
            u_matrix, (matrix_subdomains, "cells", self.nd)
        )
        self.system_manager.set_variable(u_intf, (interfaces, "cells", self.nd))
        self.system_manager.set_variable(contact_var, (interfaces, "cells", self.nd))

    def displacement(self, grids: Union[list[pp.Grid], list[pp.MortarGrid]]):
        """Displacement in the matrix.

        Args:
            grids: List of subdomains or interface grids where the displacement is
            defined. Only known usage is for the matrix subdomain(s) and matrix-fracture
            interfaces.

        Returns:
            Variable for the displacement.

        """
        displacement = self._eq_manager.merge_variables(
            [(sd, "displacement") for sd in grids]
        )
        return displacement

    def contact_traction(self, subdomains: list[pp.Grid]):
        """Fracture contact traction.

        Args:
            subdomains: List of subdomains where the contact traction is defined. Only known usage
                is for the fracture subdomains.

        Returns:
            Variable for fracture contact traction.

        """
        contact_traction = self._eq_manager.merge_variables(
            [(sd, "traction") for sd in subdomains]
        )

        return contact_traction

    def displacement_jump(self, subdomains):
        """Displacement jump on fracture-matrix interfaces.

        Args:
            subdomains: List of subdomains where the displacement jump is defined. Only known
            usage is for fractures.

        Returns:
            Operator for the displacement jump.

        Raises:
             AssertionError: If the subdomains are not fractures, i.e. have dimension
                nd - 1.
        """
        assert [sd.dim == self.nd - 1 for sd in subdomains]
        interfaces = self.subdomains_to_interfaces(subdomains)
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )
        rotated_jumps: pp.ad.Operator = (
            self.local_coordinates(subdomains)
            * self.subdomain_projections(dim=self.nd).cell_restriction(subdomains)
            * mortar_projection.mortar_to_secondary_avg
            * mortar_projection.sign_of_mortar_sides
            * self.displacement(interfaces)
        )
        rotated_jumps.set_name("Rotated_displacement_jump")
        return rotated_jumps


class SolutionStrategyForceBalance(pp.models.abstract_model.AbstractModel):
    """This is whatever is left of pp.ContactMechanics.

    At some point, this will be refined to be a more sophisticated (modularised) solution
    strategy class.
    More refactoring may be beneficial.

    """

    def __init__(self, params: Optional[Dict] = None) -> None:
        super().__init__(params)
        # Variables
        self.mechanics_discretization_parameter_key: str = "mechanics"
        self.exporter: pp.Exporter

    def prepare_simulation(self) -> None:
        self.set_geometry()
        # Exporter initialization must be done after grid creation.
        self.exporter = pp.Exporter(
            self.mdg,
            self.params["file_name"],
            folder_name=self.params["folder_name"],
            export_constants_separately=self.params.get(
                "export_constants_separately", False
            ),
        )

        self._assign_variables()
        self._create_dof_and_eq_manager()
        self._initial_condition()
        # New: Set material components. Could be moved to init:
        self.set_materials()
        # New: renamed from _set_parameters
        self.set_discretization_parameters()

        self.set_equations()

        self._export()
        self._discretize()
        self._initialize_linear_solver()

    def initial_condition(self) -> None:
        """Set initial guess for the variables.

        The displacement is set to zero in the Nd-domain, and at the fracture interfaces
        The displacement jump is thereby also zero.

        The contact pressure is set to zero in the tangential direction,
        and -1 (that is, in contact) in the normal direction.

        """
        # Zero for displacement and initial bc values for Biot
        super()._initial_condition()

        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == self.nd - 1:
                # Contact as initial guess. Ensure traction is consistent with
                # zero jump, which follows from the default zeros set for all
                # variables, specifically interface displacement, by super method.
                vals = np.zeros((self.nd, sd.num_cells))
                vals[-1] = -1
                vals = vals.ravel("F")
                data[pp.STATE].update({self.contact_traction_variable: vals})

                data[pp.STATE][pp.ITERATE].update(
                    {self.contact_traction_variable: vals.copy()}
                )

    def set_discretization_parameters(self) -> None:
        """Set discretization parameters for the simulation."""

        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == self.nd:
                pp.initialize_data(
                    sd,
                    data,
                    self.mechanics_parameter_key,
                    {
                        "bc": self.bc_type_mechanics(sd),
                        "fourth_order_tensor": self.stiffness_tensor(sd),
                    },
                )

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define type of boundary conditions: Dirichlet on all global boundaries,
        Dirichlet also on fracture faces.


        Args:
            sd: Subdomain grid.

        Returns:
            bc: Boundary condition representation.

        FIXME: Move to different class?
        """
        all_bf = sd.get_boundary_faces()
        bc = pp.BoundaryConditionVectorial(sd, all_bf, "dir")
        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the
        # fracture faces.
        frac_face = sd.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        return bc

    def numerical_constant(self, subdomains: list[pp.Grid]) -> pp.ad.Matrix:
        """Numerical constant for the contact problem.

        The numerical constant is a cell-wise scalar, but we return a matrix to allow for
        automatic differentiation and left multiplication.

        Not sure about method location, but it is a property of the contact problem,
        and more solution strategy than material property or constitutive law.

        Args:
            subdomains: List of subdomains. Only the first is used.

        Returns:
            c_num: Numerical constant, as a matrix.

        """
        vals = self.rock.convert_and_expand(1, "-", subdomains)
        c_num = constit_library.ad_wrapper(vals, False, name="c_num")
        return c_num

    def before_newton_loop(self) -> None:
        """Actions to be performed before the Newton loop."""
        pass
        # self.convergence_status = False
        # self._nonlinear_iteration = 0

    def after_newton_iteration(self, solution_vector: np.ndarray) -> None:
        """
        Extract parts of the solution for current iterate.

        The iterate solutions in d[pp.STATE][pp.ITERATE] are updated for the
        mortar displacements and contact traction are updated.
        Method is a tailored copy from assembler.distribute_variable.

        Parameters:
            solution_vector (np.array): solution vector for the current iterate.

        """
        self._nonlinear_iteration += 1
        self.dof_manager.distribute_variable(
            values=solution_vector, additive=self._use_ad, to_iterate=True
        )

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        if self._use_ad:
            # Fetch iterate solution, which was updated in after_newton_iteration
            solution = self.dof_manager.assemble_variable(from_iterate=True)
            # Distribute to pp.STATE
            self.dof_manager.distribute_variable(values=solution, additive=False)
        else:
            self.assembler.distribute_variable(solution)
        self.convergence_status = True
        self._export()

    def after_simulation(self) -> None:
        """Called after a time-dependent problem"""
        pass

    def _discretize(self) -> None:
        """Discretize all terms"""
        tic = time.time()
        logger.info("Discretize")
        if self._use_ad:
            self._eq_manager.discretize(self.mdg)
        else:
            self.assembler.discretize()
        logger.info("Done. Elapsed time {}".format(time.time() - tic))

    def _is_nonlinear_problem(self) -> bool:
        """
        If there is no fracture, the problem is usually linear.
        Overwrite this function if e.g. parameter nonlinearities are included.
        """
        return self.mdg.dim_min() < self.nd


class MdForceBalanceCombined(
    ModelGeometry,
    ForceBalanceEquations,
    ConstitutiveEquationsForceBalance,
    VariablesForceBalance,
    SolutionStrategyForceBalance,
):
    """Demonstration of how to combine in a class which can be used with
    pp.run_stationary_problem (once cleanup has been done).
    """

    pass