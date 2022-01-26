"""Implementation of a model class for (eventually) non-isothermal reactive
multiphase transport. WIP.
"""
from typing import Dict, List, Tuple, Union, Optional
import warnings
from dataclasses import dataclass

import porepy as pp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

# Shorthand typing
interface_type = Tuple[pp.Grid, pp.Grid]
grid_like_type = Union[pp.Grid, interface_type]


@dataclass
class AdVariables:
    pressure: pp.ad.MergedVariable = None
    component: List[pp.ad.MergedVariable] = None
    component_phase: np.ndarray = None
    saturation: List[pp.ad.MergedVariable] = None
    phase_mole_fraction: pp.ad.MergedVariable = None


class MultiphaseReactive(pp.models.abstract_model.AbstractModel):
    def __init__(self, params: Dict) -> None:
        warnings.warn(
            """The multiphase reactive transport model is under development and
                      is prone to change without warning.
                      """
        )
        super().__init__(params)

        self._ad = AdVariables()

        # Experimental data structure, it is not clear how we will end up representing
        # the components. This should entre somewhere else eventually.
        self.components = ["H2O", "CO2"]
        self.num_components = len(self.components)

        # By convention, the numeration of phases is set so that liquid phases
        # come before solid phases. Not sure what we need these names for
        self.phase_names = ["liquid", "vapor"]
        self.num_fluid_phases = 2
        # Don't know if we will ever have more than one solid phase, but let's
        # include it as a separate number.
        self.num_solid_phases = 0

        self.num_phases = self.num_fluid_phases + self.num_solid_phases

        self.num_reactions = 0

        ## Representation of variables
        # String representation of pressure variable
        self.pressure_variable: str = "p"
        # Stem of variable name for overall fractions of components. The full name
        # for component with index i will be f"{self.component_variable}_{i}"
        self.component_variable: str = "z"

        self.component_phase_variable: str = "x"

        self.saturation_variable: str = "S"
        self.phase_mole_fraction_variable: str = "nu"

        # Introduce temperature here just as a reminder - will likely become enthalpy
        self.temperature_variable: str = "T"

        ## Parameter keywords
        self.flow_parameter_key: str = "flow"
        self.upwind_parameter_key: str = "upwind"
        self.mass_parameter_key: str = "mass"

    def create_grid(self) -> None:
        """Create the grid bucket.


        The method assigns the following attributes to self:
            gb (pp.GridBucket): The produced grid bucket.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
        """
        phys_dims = [10, 1]
        n_cells = [10, 1]
        self.box: Dict = pp.geometry.bounding_box.from_points(
            np.array([[0, 0], phys_dims])
        )
        g: pp.Grid = pp.CartGrid(n_cells, phys_dims)
        g.compute_geometry()
        self.gb: pp.GridBucket = pp.meshing._assemble_in_bucket([[g]])

    def prepare_simulation(self) -> None:
        """Method called prior to the start of time stepping, or prior to entering the
        non-linear solver for stationary problems.

        The intended use is to define parameters, geometry and grid, discretize linear
        and time-independent terms, and generally prepare for the simulation.

        """
        self.create_grid()
        # Exporter initialization must be done after grid creation.
        self.exporter = pp.Exporter(
            self.gb, self.params["file_name"], folder_name=self.params["folder_name"]
        )
        self._set_parameters()

        # Assign variables. This will also set up Dof and EqutaionManager, and
        # define Ad versions of the variables
        self._assign_variables()

        # Set equations
        self._assign_equations()

        # Assign initial conditions. If the problem is non-linear and the system should
        # be initialized in equilibrium, iterations will be needed here.
        self._initial_condition()
        self._discretize()

        self._export()

    def before_newton_loop(self) -> None:
        """Method to be called before entering the non-linear solver, thus at the start
        of a new time step.

        """
        self.convergence_status = False
        self._nonlinear_iteration = 0

    def before_newton_iteration(self) -> None:
        """Method to be called at the start of every non-linear iteration.

        Solve the non-linear problem formed by the secondary equations. Put this in a
        separate function, since this is surely something that should be streamlined
        to the characteristics of each problem.

        """
        self._solve_secondary_equations()

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
            values=solution_vector, additive=True, to_iterate=True
        )

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the solution, visualization, etc.

        Parameters:
            np.array: The new solution state, as computed by the non-linear solver.

        """
        solution = self.dof_manager.assemble_variable(from_iterate=True)

        self.assembler.distribute_variable(solution)
        self.convergence_status = True
        self._export()

    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        if self._is_nonlinear_problem():
            raise ValueError("Newton iterations did not converge")
        else:
            raise ValueError("Tried solving singular matrix for the linear problem.")

    def after_simulation(self) -> None:
        """Run at the end of simulation. Can be used for cleaup etc."""
        pass

    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:
        """Assemble the linearized system, described by the current state of the model,
        solve and return the new solution vector.

        Parameters:
            tol (double): Target tolerance for the linear solver. May be used for
                inexact approaches.

        Returns:
            np.array: Solution vector.

        """
        """Use a direct solver for the linear system."""

        eq_manager = self._eq_manager

        # Inverter for the Schur complement system. We have an inverter for block
        # diagnoal systems ready but I never got around to actually using it (this
        # is not difficult to do).
        inverter = lambda A: sps.csr_matrix(np.linalg.inv(A.A))

        primary_equations = [
            self.equation_manager[name] for name in self._primary_equation_names
        ]
        primary_variables = self._primary_variables()

        # This forms the Jacobian matrix and residual for the primary variables,
        # where the secondary variables are first discretized according to their
        # current state, and the eliminated using a Schur complement technique.
        A_red, b_red = eq_manager.assemble_schur_complement_system(
            primary_equations=primary_equations,
            primary_variables=primary_variables,
            inverter=inverter,
        )

        # Direct solver for the global linear system. Again, this is simple but efficient
        # for sufficiently small problems.
        x = sps.linalg.spsolve(A_red, b_red)

        # Prolongation from the primary to the full set of variables
        prolongation = self._prolongation_matrix(primary_variables)

        x_full = prolongation * x

        return x_full

    def _prolongation_matrix(self, variables) -> sps.spmatrix:
        # Construct mappings from subsects of variables to the full set.
        nrows = self.dof_manager.num_dofs()
        rows = np.unique(
            np.hstack(
                # The use of private variables here indicates that something is wrong
                # with the data structures. Todo..
                [
                    self.dof_manager.grid_and_variable_to_dofs(s._g, s._name)
                    for s in variables
                ]
            )
        )
        ncols = rows.size
        cols = np.arange(ncols)
        data = np.ones(ncols)

        return sps.coo_matrix((data, (rows, cols)), shape=(nrows, ncols)).tocsr()

    def _solve_secondary_equations(self):
        """Solve the cell-wise non-linear equilibrium problem of secondary variables
        and equations.

        This is a simplest possible approach, using Newton's method, with a primitive
        implementation to boot. This should be improved at some point, however, approaches
        tailored to the specific system at hand, and/or outsourcing to dedicated libraries
        are probably preferrable.
        """
        # Equation manager for the secondary equations
        sec_man = self._secondary_equation_manager

        # The non-linear system will be solved with Newton's method. However, to get the
        # assembly etc. correctly, the updates during iterations should be communicated
        # to the data storage in the GridBucket. With the PorePy data model, this is
        # most conveniently done by the DofManager's method to distribute variables.
        # This method again can be tailored to target specific grids and variables,
        # but for simplicity, we create a prolongation matrix to the full set of equations
        # and use the resulting vector.
        prolongation = self._prolongation_matrix()

        max_iters = 100
        i = 0
        while i < max_iters:
            A, b = sec_man.assemble()
            if np.linalg.norm(b) < 1e-10:
                break
            x = spla.spsolve(A, b)
            full_x = prolongation * x
            self.dof_manager.distribute_variable(full_x, additive=True, to_iterate=True)

            i += 1
        if i == max_iters:
            raise ValueError("Newton for local systems failed to converge.")

    def _discretize(self) -> None:
        """Discretize all terms"""
        self._eq_manager.discretize(self.gb)

    def _is_nonlinear_problem(self) -> bool:
        """Specifies whether the Model problem is nonlinear."""
        return True

    #### Set problem parameters

    def _initial_condition(self):
        """Set initial conditions: Homogeneous for all variables except the pressure.
        Also feed a zero Darcy flux to the upwind discretization.
        """
        # This will set homogeneous conditions for all variables
        super()._initial_conditions()

        for g, d in self.gb:
            d[pp.STATE][self.pressure_variable][:] = 1
            d[pp.STATE][pp.ITERATE][self.pressure_variable][:] = 1

            # Careful here: Use the same variable to store the Darcy flux for all phases.
            # This is okay in this case, but do not do so for other variables.
            darcy = np.zeros(g.num_faces)
            for j in range(self.num_fluid_phases):
                d[pp.PARAMETERS][f"{self.upwind_parameter_key}_{j}"][
                    "darcy_flux"
                ] = darcy

    def _set_parameters(self):
        """Set default parameters needed in the simulations.

        Many of the functions here may change in the future, partly to allow for more
        general descriptions of fluid and rock properties. Also, overriding some of
        the constitutive laws may lead to different parameters being needed.

        """
        for g, d in self.gb:
            # Parameters for single-phase Darcy. This is copied from the incompressible
            # flow model.
            bc = self._bc_type_flow(g)
            bc_values = self._bc_values_flow(g)

            source_values = self._source(g)

            specific_volume = self._specific_volume(g)

            kappa = self._permeability(g) / self._viscosity(g)
            diffusivity = pp.SecondOrderTensor(
                kappa * specific_volume * np.ones(g.num_cells)
            )

            # No gravity
            gravity = np.zeros((self.gb.dim_max(), g.num_cells))

            pp.initialize_data(
                g,
                d,
                self.flow_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "source": source_values,
                    "second_order_tensor": diffusivity,
                    "vector_source": gravity.ravel("F"),
                    "ambient_dimension": self.gb.dim_max(),
                },
            )

            ## Parameters for the transport problems. These are less mature.

            # NOTE: Seen from the upstream discretization, the Darcy velocity is a
            # parameter, although it is a derived quantity from the flow discretization
            # point of view. We will set a value for this in the initialization, to
            # increase the chances the user remembers to set compatible flux and
            # pressure.

            # Mass weight parameter. Same for all phases
            mass_weight = self._porosity(g) * self._specific_volume(g)
            d[pp.PARAMETERS][self.mass_parameter_key] = {"mass_weight": mass_weight}

            for j in range(self.num_fluid_phases):
                bc = self._bc_type_transport(g, j)
                bc_values = self._bc_values_transport(g, j)
                d[pp.PARAMETERS][f"{self.upwind_parameter_key}_{j}"] = {
                    "bc": bc,
                    "bc_values": bc_values,
                }

        # Assign diffusivity in the normal direction of the fractures.
        for e, data_edge in self.gb.edges():
            raise NotImplementedError("Only single grid for now")

    def _bc_type_flow(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Neumann conditions on all external boundaries."""
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "neu")

    def _bc_type_transport(self, g: pp.Grid, j: int) -> pp.BoundaryCondition:
        """Set type of boundary condition for transport of phase j"""
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "neu")

    def _bc_values_flow(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous boundary values.

        Units:
            Dirichlet conditions: Pa = kg / m^1 / s^2
            Neumann conditions: m^3 / s
        """
        return np.zeros(g.num_faces)

    def _bc_values_transport(self, g: pp.Grid, j: int) -> np.ndarray:
        """Homogeneous boundary values.

        Units:
            Dirichlet conditions: Pa = kg / m^1 / s^2
            Neumann conditions: m^3 / s
        """
        return np.zeros(g.num_faces)

    def _source(self, g: pp.Grid) -> np.ndarray:
        """Zero source term.

        Units: m^3 / s
        """
        return np.zeros(g.num_cells)

    def _permeability(self, g: pp.Grid) -> np.ndarray:
        """Unitary permeability.

        Units: m^2
        """
        return np.ones(g.num_cells)

    def _porosity(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous porosity"""
        if g.dim < self.gb.dim_max():
            # Unit porosity in fractures. Scaling with aperture (dimension reduction)
            # should be handled by including a specific volume.
            scaling = 1
        else:
            scaling = 0.2

        return np.zeros(g.num_cells) * scaling

    def _viscosity(self, g: pp.Grid) -> np.ndarray:
        """Unitary viscosity.

        Units: kg / m / s = Pa s
        """
        return np.ones(g.num_cells)

    def _aperture(self, g: pp.Grid) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of dimension 1 and 0.
        See also specific_volume.
        """
        aperture = np.ones(g.num_cells)
        if g.dim < self.gb.dim_max():
            aperture *= 0.1
        return aperture

    def _specific_volume(self, g: pp.Grid) -> np.ndarray:
        """
        The specific volume of a cell accounts for the dimension reduction and has
        dimensions [m^(Nd - d)].
        Typically equals 1 in Nd, the aperture in codimension 1 and the square/cube
        of aperture in dimension 1 and 0.
        """
        a = self._aperture(g)
        return np.power(a, self._nd_grid().dim - g.dim)

    #### Methods related to variables

    def _assign_variables(self) -> None:
        """Define variables used to describe the system.

        These will include both primary and secondary variables, however, to be
        compatible to the terminology in PorePy, they will all be denoted as primary
        in certain settings.

        """
        # This function works in three steps:
        # 1) All variables are defined, with the right set of DOFS.
        # 2) Set Dof- and EquationManagers.
        # 3) Define AD representations of all the variables. This is done by
        #    calling a separate function.

        for g, d in self.gb:
            # Naming scheme for component and phase indices:
            # Component index is always i, phase index always j.
            primary_variables = {self.pressure_variable: {"cells": 1}}

            # Total molar fraction of each component
            primary_variables.update(
                {
                    f"{self.component_variable}_{i}": {"cells": 1}
                    for i in range(self.num_components)
                }
            )

            # Phase mole fractions
            primary_variables.update(
                {
                    f"{self.phase_mole_fraction_variable}_{i}": {"cells": 1}
                    for i in range(self.num_phases)
                }
            )
            # Saturations
            primary_variables.update(
                {
                    f"{self.saturation_variable}_{i}": {"cells": 1}
                    for i in range(self.num_phases)
                }
            )

            # Component phase molar fractions
            # Note systematic naming convention: i is always component, j is phase.
            for j in range(self.num_phases):
                for i in range(self.num_components):
                    primary_variables.update(
                        {f"{self.component_phase_variable}_{i}_{j}": {"cells": 1}}
                    )
            # The wording is a bit confusing here, these will not be taken as
            d[pp.PRIMARY_VARIABLES] = primary_variables

        for e, d in self.gb.edges():
            raise NotImplementedError("Have only considered non-fractured domains.")

        # All variables defined, we can set up Dof and Equation managers
        self.dof_manager = pp.DofManager(self.gb)
        self.equation_manager = pp.ad.EquationManager(self.gb, self.dof_manager)

        # The manager set, we can finally do the Ad formulation of variables
        self._assign_ad_variables()

    def _assign_ad_variables(self) -> None:
        """Make lists of AD-variables, indexed by component and/or phase number.
        The idea is to enable easy access to the Ad variables without having to
        construct these from the equation manager every time we need them.
        """
        eqm = self.equation_manager

        grid_list = self._grid_list()

        self._ad.pressure: pp.ad.MergedVariable = eqm.merge_variables(
            [(g, self.pressure_variable) for g in grid_list]
        )

        self._ad.component: List[pp.ad.Variable] = []

        for i in range(self.num_components):
            name = f"{self.component_variable}_{i}"
            var = eqm.merge_variables([(g, name) for g in grid_list])
            self._ad.component.append(var)

        # Represent component phases as an numpy array instead of a list, so that we
        # can access items by array[i, j], rather the more cumbersome array[i][j]
        self._ad.component_phase: np.ndarray = np.empty(
            (self.num_components, self.num_phases), dtype=object
        )
        for i in range(self.num_components):
            # Make inner list
            for j in range(self.num_phases):
                name = f"{self.component_phase_variable}_{i}_{j}"
                var = eqm.merge_variables([(g, name) for g in grid_list])
                self._ad.component_phase[i, j] = var

        self._ad.saturation = []
        self._ad.phase_mole_fraction = []
        for j in range(self.num_fluid_phases):
            # Define saturation variables for each phase
            sat_var = eqm.merge_variables(
                [(g, f"{self.saturation_variable}_{j}") for g in grid_list]
            )
            self._ad.saturation.append(sat_var)

            # Define Molar fraction variables, one for each phase
            mf_var = eqm.merge_variables(
                [(g, f"{self.phase_mole_fraction_variable}_{j}") for g in grid_list]
            )
            self._ad.phase_mole_fraction.append(mf_var)

    def _primary_variables(self) -> List[pp.ad.MergedVariable]:
        """Get a list of the primary variables of the system on AD form.

        This will be the pressure and n-1 of the total molar fractions.

        """
        # The primary variables are the pressure and all but one of the total
        # molar fractions.
        # Represent primary variables by their AD format, since this is what is needed
        # to interact with the EquationManager.
        primary_variables = [self._ad.pressure] + self._ad.component[:-1]
        return primary_variables

    def _secondary_variables(self) -> List[pp.ad.MergedVariable]:
        """Get a list of secondary variables of the system on AD form.

        This will the final total molar fraction, phase molar fraction, component
        mole fractions, and saturations.
        """
        # The secondary variables are the final molar fraction, saturations, phase
        # mole fractions and component phases.
        secondary_variables = (
            [self._ad.component[-1]]
            + self._ad.saturation
            + self._ad.phase_mole_fraction
            + self._ad.component_phase
        )
        return secondary_variables

    def _grid_list(self) -> List[pp.Grid]:
        # Helper method to get list of grids. Not sure if we need this, but it
        # looks cleaner than creating this list in several other functions
        return [g for g, _ in self.gb]

    def _edge_list(self) -> List[interface_type]:
        return [e for e, _ in self.gb.edges()]

    #### Set governing equations, in AD form

    def _assign_equations(self) -> None:
        """Method to set all equations."""

        self._set_transport_equations()

        # Equilibrium equations
        self._phase_equilibrium_equations()
        self._chemical_equilibrium_equations()

        # Equations for pure bookkeeping, relations between variables etc.
        self._overall_molar_fraction_sum_equations()
        self._component_phase_sum_equations()
        self._phase_mole_fraction_sum_equation()

        # Now that all equations are set, we define sets of primary and secondary
        # equations, and similar with variables. These will be used to represent
        # the systems to be solved globally (transport equations) and locally
        # (equilibrium equations).
        eq_manager = self._eq_manager

        # What to do in the case of a single component is not clear to EK at the time
        # of writing. Question is, do we still eliminate (the only) one transport equation?
        # Maybe the answer is a trivial yes, but raise an error at this stage, just to
        # be sure.
        assert len(self._component_ad) > 1

        # FIXME: Mortar variables are needed here
        assert len(self._edge_list()) == 0

        # Create a separate EquationManager for the secondary variables and equations.
        # This set of secondary equations will still contain the primary variables,
        # but their derivatives will not be computed in the construction of the
        # Jacobian matrix (strictly speaking, derivatives will be computed, then dropped).
        # Thus, the secondary manager can be used to solve the local (to cells) systems
        # describing equilibrium.

        # Get the secondary variables of the system.
        secondary_variables = self._secondary_variables()

        # Ad hoc approach to get the names of the secondary equations. This is not beautiful.
        secondary_equation_names = [
            name
            for name in list(eq_manager.equations.keys())
            if name[:12] != "Mass_balance"
        ]

        self._secondary_equation_manager = eq_manager.subsystem_equation_manager(
            secondary_equation_names, secondary_variables
        )

        # Also store the name of the primary variables, we will need this to construct
        # the global linear system later on.
        # FIXME: Should we also store secondary equation names, for symmetry reasons?
        self._primary_equation_names = list(
            set(eq_manager.equations.keys()).difference(secondary_equation_names)
        )

    #### Methods to set transport equation

    def _set_transport_equations(self) -> None:
        """Set transport equations"""
        grid_list = self._grid_list()

        darcy = self._single_phase_darcy()

        rp = [self._rel_perm(j) for j in range(self.num_fluid_phases)]

        component_flux = [0 for i in range(self.num_components)]

        for j in range(self.num_fluid_phases):
            rp = self._rel_perm(j)

            upwind = pp.ad.UpwindAd(f"{self.upwind_parameter_key}_{j}", grid_list)

            rho_j = self._density(j)

            darcy_j = (upwind.upwind * rp) * darcy

            for i in range(self.num_components):
                component_flux[i] += darcy_j * (
                    upwind.upwind * (rho_j * self._ad.component_phase[i, j])
                )
        mass = pp.ad.MassMatrixAd(self.mass_parameter_key, grid_list)

        dt = 1

        rho_tot = self._density()
        rho_tot_prev_time = self._density(prev_time=True)

        div = pp.ad.Divergence(grid_list, dim=1, name="Divergence")

        component_mass_balance: List[pp.ad.Operator()] = []

        g = self.gb.grids_of_dimension(self.gb.dim_max())[0]

        for i in range(self.num_components):
            # Boundary conditions
            bc = pp.ad.ParameterArray(  # Not sure about this part - should there also be a phase-wise boundary condition?
                param_keyword=upwind.keyword, array_keyword="bc_values", grids=[g]
            )
            # The advective flux is the sum of the internal (computed in flux_{i} above)
            # and the boundary condition
            # FIXME: We need to account for both Neumann and Dirichlet boundary conditions,
            # and likely do some filtering.
            adv_flux = component_flux[i] + upwind.bound_transport_neu * bc

            z_i = self._ad.component[i]
            # accumulation term
            accum = (
                mass.mass
                * (z_i / rho_tot - z_i.previous_timestep() / rho_tot_prev_time)
                / dt
            )

            # Append to set of conservation equations
            component_mass_balance.append(accum + div * adv_flux)

        for i, eq in enumerate(component_mass_balance):
            self.equation_manager.equations[f"Mass_balance_component{i}"] = eq

    def _single_phase_darcy(self) -> pp.ad.Operator:
        """Discretize single-phase Darcy's law using Mpfa.

        Override method, e.g., to use Tpfa instead.

        Returns
        -------
        darcy : TYPE
            DESCRIPTION.

        """
        grid_list = self._grid_list()
        mpfa = pp.ad.MpfaAd(self.flow_parameter_key, grid_list)

        bc = pp.ad.ParameterArray(self.flow_parameter_key, "bc_values", grids=grid_list)

        darcy = mpfa.flux * self._ad.pressure + mpfa.bound_flux * bc
        return darcy

    def _upstream(self, phase_ind: int) -> pp.ad.Operator:
        # Not sure we need this one, but it may be convenient if we want to override this
        # (say, for countercurrent flow).
        grid_list = self._grid_list()

        upwind = pp.ad.UpwindAd(f"{self.upwind_parameter_key}_{phase_ind}", grid_list)

        rp = self._rel_perm(phase_ind)

        return upwind.upwind * rp

    #### Equations for phase and chemical equilibrium

    def _chemical_equilibrium_equations(self) -> None:
        if self.num_reactions > 0:
            raise NotImplementedError("Have not yet implemented reaction terms")

    def _phase_equilibrium_equations(self) -> None:
        """Define equations for phase equilibrium and assign to the EquationManager.

        For the moment, no standard 'simplest' model is implemented - this may change
        in the future.
        """
        raise NotImplementedError("This must be implemented in subclasses")

    #### Equations for bookkeeping (relation between various variables)

    def _overall_molar_fraction_sum_equations(self) -> None:
        """
        Set equation z_i = \sum_j x_ij * v_j
        """
        eq_manager = self.equation_manager

        for i in range(self.num_components):
            phase_sum_i = sum(
                [
                    self._ad.component_phase[i, j] * self._ad.phase_mole_fraction[i, j]
                    for j in range(self.num_phases)
                ]
            )

            eq = self._ad.component[i] - phase_sum_i
            eq_manager.equations.append(eq, name="Overall_comp_phase_comp")

    def _component_phase_sum_equations(self) -> None:
        """Force the component phases to sum to unity for all components.

        \sum_i x_i,0 = \sum_i x_ij, j=1,..
        """
        eq_manager = self.equation_manager

        def _comp_sum(j: int) -> pp.ad.Operator:
            return sum(
                [self._ad.component_phase[i, j] for i in range(self.num_components)]
            )

        sum_0 = _comp_sum(0)

        for j in range(1, self.num_phases):
            sum_j = _comp_sum(j)
            eq_manager.equations.append(sum_0 - sum_j, name=f"Comp_phase_sum_{j}")

    def _phase_mole_fraction_sum_equation(self) -> None:
        """Force mole fractions to sum to unity

        sum_j v_j = 1

        """
        eq = sum([self._ad.phase_mole_fraction[j] for j in range(self.num_phases)])
        self.equation_manager.equations.append(eq, "Phase_mole_fraction_sum")

    #### Constitutive laws

    def _kinetic_reaction_rate(self, i: int) -> float:
        """Get the kinetic rate for a given reaction."""
        raise NotImplementedError("This is not covered")

    def _rel_perm(self, j: int) -> pp.ad.Operator:
        """Get the relative permeability for a given phase.

        The implemented function is a quadratic Brooks-Corey function. Override to
        use a different function.

        IMPLEMENTATION NOTE: The data structure for relative permeabilities may change
        substantially in the future. Specifically, hysteretic effects and heterogeneities
        may require separate classes for flow functions.

        Parameters:
            j (int): Index of the phase.

        Returns:
            pp.ad.Operator: Relative permeability of the given phase.

        """
        sat = self._ad.saturation[j]
        # Brooks-Corey function
        return pp.ad.Function(lambda x: x ** 2, "Rel. perm. liquid")(sat)

    def _density(
        self, j: Optional[int] = None, prev_time: bool = False
    ) -> pp.ad.Operator:
        """Get the density of a specified phase, or a saturation-weighted sum over
        all phases.

        Optionally, the density can be evaluated at the previous time step.

        The implemented function is that of a slightly compressible fluid. Override
        to use different functions, including functions calculated from external
        packages.

        FIXME: Should this be a public function?

        Parameters:
            j (int, optional): Index of the target phase. If not provided, a saturation
                weighted mean density will be returned.
            prev_time (bool, optional): If True, the density is evaluated at the previous
             time step. Defaults to False.

        Returns:
            pp.ad.Operator: Ad representation of the density.

        """
        if j is None:
            average = sum(
                [
                    self._density(j, prev_time) * self._ad.saturation[j]
                    for j in range(self.num_fluid_phases)
                ]
            )
            return average

        # Set some semi-random values for densities here. These could be set in the
        # set_parameter method (will give more flexibility), or the density could be
        # provided as a separate function / class (perhaps better suited to accomodate
        # external libraries)
        base_pressure = 1
        base_density = [1000, 800]
        compressibility = [1e-6, 1e-5]

        var = self._ad.pressure.previous_timestep() if prev_time else self._ad.pressure
        return pp.ad.Function(
            lambda p: base_density[j] * (1 + compressibility[j] * (p - base_pressure)),
            f"Density_phase_{j}",
        )(var)
