"""
Class types:
    - MomentumBalanceEquations defines subdomain and interface equations through the
    terms entering. Momentum balance between opposing fracture interfaces is imposed.

Notes:
    - The class MomentumBalanceEquations is a mixin class, and should be inherited by a
      class that defines the variables and discretization.

    - Refactoring needed for constitutive equations. Modularisation and moving to the
      library.

"""

from __future__ import annotations

import logging
from functools import partial
from typing import Optional

import numpy as np

import porepy as pp
from porepy import ModelGeometry

from . import constitutive_laws

logger = logging.getLogger(__name__)


class MomentumBalanceEquations(pp.BalanceEquation):
    """Class for momentum balance equations and fracture deformation equations."""

    def set_equations(self):
        """Set equations for the subdomains and interfaces.

        The following equations are set:
            - Momentum balance in the matrix.
            - Force balance between fracture interfaces.
            - Deformation constraints for fractures, split into normal and tangential
              part.
        See individual equation methods for details.
        """
        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        fracture_subdomains = self.mdg.subdomains(dim=self.nd - 1)
        interfaces = self.mdg.interfaces(dim=self.nd - 1)
        matrix_eq = self.momentum_balance_equation(matrix_subdomains)
        # We split the fracture deformation equations into two parts, for the normal and
        # tangential components for convenience.
        fracture_eq_normal = self.normal_fracture_deformation_equation(
            fracture_subdomains
        )
        fracture_eq_tangential = self.tangential_fracture_deformation_equation(
            fracture_subdomains
        )
        intf_eq = self.interface_force_balance_equation(interfaces)
        self.equation_system.set_equation(
            matrix_eq, matrix_subdomains, {"cells": self.nd}
        )
        self.equation_system.set_equation(
            fracture_eq_normal, fracture_subdomains, {"cells": 1}
        )
        self.equation_system.set_equation(
            fracture_eq_tangential, fracture_subdomains, {"cells": self.nd - 1}
        )
        self.equation_system.set_equation(intf_eq, interfaces, {"cells": self.nd})

    def momentum_balance_equation(self, subdomains: list[pp.Grid]):
        """Momentum balance equation in the matrix.

        Inertial term is not included.

        Parameters:
            subdomains: List of subdomains where the force balance is defined. Only
            known usage
                is for the matrix domain(s).

        Returns:
            Operator for the force balance equation in the matrix.

        """
        accumulation = self.inertia(subdomains)
        stress = self.stress(subdomains)
        body_force = self.body_force(subdomains)
        equation = self.balance_equation(
            subdomains, accumulation, stress, body_force, dim=self.nd
        )
        equation.set_name("momentum_balance_equation")
        return equation

    def inertia(self, subdomains: list[pp.Grid]):
        """Inertial term [m^2/s].

        Parameters:
            subdomains: List of subdomains where the inertial term is defined.

        Returns:
            Operator for the inertial term.

        """
        return pp.ad.Scalar(0)

    def interface_force_balance_equation(
        self,
        interfaces: list[pp.MortarGrid],
    ) -> pp.ad.Operator:
        """Momentum balance equation at matrix-fracture interfaces.

        Parameters:
            interfaces: Fracture-matrix interfaces.

        Returns:
            Operator representing the force balance equation.

        """
        # Check that the interface is a fracture-matrix interface.
        for interface in interfaces:
            if interface.dim != self.nd - 1:
                raise ValueError("Interface must be a fracture-matrix interface.")
        subdomains = self.interfaces_to_subdomains(interfaces)
        # Split into matrix and fractures. Sort on dimension to allow for multiple
        # matrix domains. Otherwise, we could have picked the first element.
        matrix_subdomains = [sd for sd in subdomains if sd.dim == self.nd]
        fracture_subdomains = [sd for sd in subdomains if sd.dim == self.nd - 1]

        # Geometry related
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )

        # Contact traction from primary grid and mortar displacements (via primary grid)
        contact_from_primary_mortar = (
            mortar_projection.primary_to_mortar_int
            * self.subdomain_projections(self.nd).face_prolongation(matrix_subdomains)
            * self.internal_boundary_normal_to_outwards(interfaces)
            * self.stress(matrix_subdomains)
        )
        # Traction from the actual contact force.
        contact_from_secondary = (
            mortar_projection.sign_of_mortar_sides
            * mortar_projection.secondary_to_mortar_int
            * self.subdomain_projections(self.nd).cell_prolongation(fracture_subdomains)
            * self.local_coordinates(fracture_subdomains).transpose()
            * self.contact_traction(fracture_subdomains)
        )

        force_balance_eq: pp.ad.Operator = (
            contact_from_primary_mortar
            + self.volume_integral(contact_from_secondary, interfaces, dim=self.nd)
        )
        force_balance_eq.set_name("interface_force_balance_equation")
        return force_balance_eq

    def normal_fracture_deformation_equation(self, subdomains: list[pp.Grid]):
        """Equation for the normal component of the fracture deformation.

        This constraint equation enforces non-penetration of opposing fracture
        interfaces.


        Parameters:
            subdomains: List of subdomains where the normal deformation equation is
            defined.

        Returns:
            Operator for the normal deformation equation.

        """
        # Variables
        nd_vec_to_normal = self.normal_component(subdomains)
        t_n: pp.ad.Operator = nd_vec_to_normal * self.contact_traction(subdomains)
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
        equation.set_name("normal_fracture_deformation_equation")
        return equation

    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """
        Contact mechanics equation for the tangential constraints.

        The function reads
            C_t = max(b_p, ||T_t+c_t u_t||) T_t - max(0, b_p) (T_t+c_t u_t)
        with u being displacement jump increments, t denoting tangential component and
        b_p the friction bound.

        For b_p = 0, the equation C_t = 0 does not in itself imply T_t = 0, which is
        what the contact conditions require. The case is handled through the use of a
        characteristic function.

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
        num_cells = sum([sd.num_cells for sd in subdomains])
        nd_vec_to_tangential = self.tangential_component(subdomains)
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)
        scalar_to_tangential = sum([e_i for e_i in tangential_basis])
        # Variables
        t_t: pp.ad.Operator = nd_vec_to_tangential * self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential * self.displacement_jump(subdomains)
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        ones_frac = pp.ad.Array(np.ones(num_cells * (self.nd - 1)))
        zeros_frac = pp.ad.Array(np.zeros(num_cells))

        # Functions EK: Should we try to agree on a name convention for ad functions?
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        tol = 1e-5  # FIXME: Revisit this tolerance!
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

        # Compose the equation itself. The last term handles the case bound=0, in which
        # case t_t = 0 cannot be deduced from the standard version of the complementary
        # function (i.e. without the characteristic function). Filter out the other
        # terms in this case to improve convergence
        equation: pp.ad.Operator = (ones_frac - characteristic) * (
            bp_tang - maxbp_abs * t_t
        ) + characteristic * t_t
        equation.set_name("tangential_fracture_deformation_equation")
        return equation

    def body_force(self, subdomains: list[pp.Grid]):
        """Body force.

        FIXME: See FluidMassBalanceEquations.fluid_source. Parameters:
            subdomains: List of subdomains where the body force is defined.

        Returns:
            Operator for the body force.

        """
        num_cells = sum([sd.num_cells for sd in subdomains])
        vals = np.zeros(num_cells * self.nd)
        source = pp.ad.Array(vals, "body_force")
        return source


class ConstitutiveLawsMomentumBalance(
    constitutive_laws.LinearElasticSolid,
    constitutive_laws.FracturedSolid,
    constitutive_laws.FrictionBound,
):
    """Class for constitutive equations for momentum balance equations."""

    def stress(self, subdomains: list[pp.Grid]):
        """Stress operator.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Operator for the stress.

        """
        # Method from constitutive library's LinearElasticRock.
        return self.mechanical_stress(subdomains)


class VariablesMomentumBalance:
    """
    Variables for mixed-dimensional deformation:
        Displacement in matrix and on fracture-matrix interfaces. Fracture contact
        traction.

    .. note::
        Implementation postponed till Veljko's more convenient SystemManager is available.

    """

    def create_variables(self):
        """Set variables for the subdomains and interfaces.

        The following variables are set:
            - Displacement in the matrix.
            - Displacement on fracture-matrix interfaces.
            - Fracture contact traction.
        See individual variable methods for details.
        """

        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.displacement_variable,
            subdomains=self.mdg.subdomains(dim=self.nd),
        )
        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.interface_displacement_variable,
            interfaces=self.mdg.interfaces(dim=self.nd - 1),
        )
        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.contact_traction_variable,
            subdomains=self.mdg.subdomains(dim=self.nd - 1),
        )

    def displacement(self, subdomains: list[pp.Grid]):
        """Displacement in the matrix.

        Parameters:
            grids: List of subdomains or interface grids where the displacement is
            defined. Only known usage is for the matrix subdomain(s) and matrix-fracture
            interfaces.

        Returns:
            Variable for the displacement.

        """
        assert all([sd.dim == self.nd for sd in subdomains])

        return self.equation_system.md_variable(self.displacement_variable, subdomains)

    def interface_displacement(self, interfaces: list[pp.MortarGrid]):
        """Displacement on fracture-matrix interfaces.

        Parameters:
            interfaces: List of interface grids where the displacement is defined.

        Returns:
            Variable for the displacement.

        """
        assert all([intf.dim == self.nd - 1 for intf in interfaces])

        return self.equation_system.md_variable(
            self.interface_displacement_variable, interfaces
        )

    def contact_traction(self, subdomains: list[pp.Grid]):
        """Fracture contact traction.

        Parameters:
            subdomains: List of subdomains where the contact traction is defined. Only
            known usage is for the fracture subdomains.

        Returns:
            Variable for fracture contact traction.

        """
        # Check that the subdomains are fractures
        for sd in subdomains:
            assert sd.dim == self.nd - 1, "Contact traction only defined on fractures"
        return self.equation_system.md_variable(
            self.contact_traction_variable, subdomains
        )

    def displacement_jump(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Displacement jump on fracture-matrix interfaces.

        Parameters:
            subdomains: List of subdomains where the displacement jump is defined. Only
            known usage is for fractures.

        Returns:
            Operator for the displacement jump.

        Raises:
             AssertionError: If the subdomains are not fractures, i.e. have dimension
                nd - 1.
        """
        assert all([sd.dim == self.nd - 1 for sd in subdomains])
        interfaces = self.subdomains_to_interfaces(subdomains)
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )
        rotated_jumps: pp.ad.Operator = (
            self.local_coordinates(subdomains)
            * mortar_projection.mortar_to_secondary_avg
            * mortar_projection.sign_of_mortar_sides
            * self.interface_displacement(interfaces)
        )
        rotated_jumps.set_name("Rotated_displacement_jump")
        return rotated_jumps


class SolutionStrategyMomentumBalance(pp.SolutionStrategy):
    """This is whatever is left of pp.ContactMechanics.

    At some point, this will be refined to be a more sophisticated (modularised)
    solution strategy class. More refactoring may be beneficial.

    """

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        # Variables
        self.displacement_variable: str = "u"
        """Name of the displacement variable."""
        self.interface_displacement_variable: str = "u_interface"
        """Name of the displacement variable on fracture-matrix interfaces."""
        self.contact_traction_variable: str = "t"
        """Name of the contact traction variable."""

        # Discretization
        self.stress_keyword: str = "mechanics"
        """Keyword for stress term.

        Used to access discretization parameters and store discretization matrices.

        """

    def initial_condition(self) -> None:
        """Set initial guess for the variables.

        The displacement is set to zero in the Nd-domain, and at the fracture interfaces
        The displacement jump is thereby also zero.

        The contact pressure is set to zero in the tangential direction, and -1 (that
        is, in contact) in the normal direction.

        """
        # Zero for displacement and initial bc values for Biot
        super().initial_condition()
        # Contact as initial guess. Ensure traction is consistent with zero jump, which
        # follows from the default zeros set for all variables, specifically interface
        # displacement, by super method.
        num_frac_cells = sum(
            sd.num_cells for sd in self.mdg.subdomains(dim=self.nd - 1)
        )
        traction_vals = np.zeros((self.nd, num_frac_cells))
        traction_vals[-1] = -1
        self.equation_system.set_variable_values(
            traction_vals.ravel("F"),
            self.contact_traction_variable,
            to_state=True,
            to_iterate=True,
        )

    def set_discretization_parameters(self) -> None:
        """Set discretization parameters for the simulation."""

        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == self.nd:
                pp.initialize_data(
                    sd,
                    data,
                    self.stress_keyword,
                    {
                        "bc": self.bc_type_mechanics(sd),
                        "fourth_order_tensor": self.stiffness_tensor(sd),
                    },
                )

    def numerical_constant(self, subdomains: list[pp.Grid]) -> pp.ad.Matrix:
        """Numerical constant for the contact problem.

        The numerical constant is a cell-wise scalar, but we return a matrix to allow
        for automatic differentiation and left multiplication.

        Not sure about method location, but it is a property of the contact problem, and
        more solution strategy than material property or constitutive law.

        Parameters:
            subdomains: List of subdomains. Only the first is used.

        Returns:
            c_num: Numerical constant, as a matrix.

        """
        # Conversion unnecessary for dimensionless parameters, but included as good
        # practice.
        val = self.solid.convert_units(1, "-")
        return pp.ad.Scalar(val, name="c_num")

    def _is_nonlinear_problem(self) -> bool:
        """
        If there is no fracture, the problem is usually linear. Overwrite this function
        if e.g. parameter nonlinearities are included.
        """
        return self.mdg.dim_min() < self.nd


class BoundaryConditionsMomentumBalance:
    """Boundary conditions for the momentum balance."""

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define type of boundary conditions.


        Parameters:
            sd: Subdomain grid.

        Returns:
            bc: Boundary condition representation. Dirichlet on all global boundaries,
            Dirichlet also on fracture faces.

        """
        all_bf = sd.get_boundary_faces()
        bc = pp.BoundaryConditionVectorial(sd, all_bf, "dir")
        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the fracture
        # faces.
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_mechanics(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Boundary values for the momentum balance.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            bc_values: Array of boundary condition values, zero by default. If combined
            with transient problems in e.g. Biot, this should be a
            :class:`pp.ad.TimeDependentArray` (or a variable following BoundaryGrid
            extension).

        """
        num_faces = sum([sd.num_faces for sd in subdomains])
        return constitutive_laws.ad_wrapper(
            0, True, num_faces * self.nd, "bc_vals_mechanics"
        )


class MdMomentumBalanceCombined(
    ModelGeometry,
    MomentumBalanceEquations,
    ConstitutiveLawsMomentumBalance,
    VariablesMomentumBalance,
    SolutionStrategyMomentumBalance,
):
    """Demonstration of how to combine in a class which can be used with
    pp.run_stationary_problem (once cleanup has been done).
    """

    pass