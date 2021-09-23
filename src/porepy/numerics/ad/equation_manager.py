"""
* Resue assembly when relevant (if no operator that maps to a specific block has been changed)
* Concatenate equations with the same sequence of operators
  - Should use the same discretization object
  - divergence operators on different grids considered the same
* Concatenated variables will share ad derivatives. However, it should be possible to combine
  subsets of variables with other variables (outside the set) to assemble different terms
*
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

from . import operators
from .discretizations import _MergedOperator
from .forward_mode import initAdArrays

__all__ = ["Expression", "EquationManager"]

grid_like_type = Union[pp.Grid, Tuple[pp.Grid, pp.Grid]]


class Expression:
    """Ad representation of an expression which can be evaluated (translated to
    numerical values).

    Conceptually, an Equation is an Operator tree that has been equated to zero.

    The equation has a fixed set of variables, identified from the operator tree.

    The residual and Jacobian matrix of an Equation can be evaluated via the function
    to_ad().

    Attributes:
        operator (Operator): Top operator in the operator tree.
        dof_manager (pp.DofManager): Degree of freedom manager associated with the
            mixed-dimensional GridBucket with which this equation is associated. Used
            to map between local (to the equation) and global variables.
        name (str): Name identifier of this variable.

    """

    def __init__(
        self,
        operator: operators.Operator,
        dof_manager: pp.DofManager,
        name: str = "",
        grid_order: Optional[Sequence[Union[pp.Grid, Tuple[pp.Grid, pp.Grid]]]] = None,
    ):
        """Define an Equation.

        Parameters:
            operator (pp.ad.Operator): Top-level operator of the Operator tree that will
                be equated to zero.
            dof_manager (pp.DofManager): Degree of freedom manager associated with the
                mixed-dimensional GridBucket with which this equation is associated.
            name (str): Name of the Eqution.

        """
        # The only non-trival operation in __init__ is the identification of variables.
        # Besides, some bookkeeping is necessary.

        # Black sometimes formats long equations with parantheses in a way that is
        # interpreted as a tuple by Python. Sigh.
        if (
            isinstance(operator, tuple)
            and len(operator) == 1
            and isinstance(operator[0], operators.Operator)
        ):
            operator = operator[0]

        self._operator = operator
        self._dof_manager = dof_manager

        self.name = name
        self.grid_order = grid_order

        # Identify all variables in the Operator tree. This will include real variables,
        # and representation of previous time steps and iterations.
        (
            variable_dofs,
            variable_ids,
            is_prev_time,
            is_prev_iter,
        ) = self._identify_variables(dof_manager)

        # Split variable dof indices and ids into groups of current variables (those
        # of the current iteration step), and those from the previous time steps and
        # iterations.
        current_indices = []
        current_ids = []
        prev_indices = []
        prev_ids = []
        prev_iter_indices = []
        prev_iter_ids = []
        for ind, var_id, is_prev, is_prev_it in zip(
            variable_dofs, variable_ids, is_prev_time, is_prev_iter
        ):
            if is_prev:
                prev_indices.append(ind)
                prev_ids.append(var_id)
            elif is_prev_it:
                prev_iter_indices.append(ind)
                prev_iter_ids.append(var_id)
            else:
                current_indices.append(ind)
                current_ids.append(var_id)

        # Save information.
        self._variable_dofs = current_indices
        self._variable_ids = current_ids
        self._prev_time_dofs = prev_indices
        self._prev_time_ids = prev_ids
        self._prev_iter_dofs = prev_iter_indices
        self._prev_iter_ids = prev_iter_ids

        self.discretizations: Dict[
            _MergedOperator, grid_like_type
        ] = self._identify_discretizations()

    def __repr__(self) -> str:
        return f"Equation named {self.name}"

    def _find_subtree_variables(
        self, op: operators.Operator
    ) -> List[operators.Variable]:
        """Method to recursively look for Variables (or MergedVariables) in an
        operator tree.
        """
        # The variables should be located at leaves in the tree. Traverse the tree
        # recursively, look for variables, and then gather the results.

        if isinstance(op, operators.Variable) or isinstance(op, pp.ad.Variable):
            # We are at the bottom of the a branch of the tree, return the operator
            return [op]
        else:
            # We need to look deeper in the tree.
            # Look for variables among the children
            sub_variables = [
                self._find_subtree_variables(child) for child in op.tree.children
            ]
            # Some work is needed to parse the information
            var_list: List[operators.Variable] = []
            for var in sub_variables:
                if isinstance(var, operators.Variable) or isinstance(
                    var, pp.ad.Variable
                ):
                    # Effectively, this node is one step from the leaf
                    var_list.append(var)
                elif isinstance(var, list):
                    # We are further up in the tree.
                    for sub_var in var:
                        if isinstance(sub_var, operators.Variable) or isinstance(
                            sub_var, pp.ad.Variable
                        ):
                            var_list.append(sub_var)
            return var_list

    def _identify_variables(self, dof_manager, var: Optional[list] = None):
        # NOTES TO SELF:
        # state: state vector for all unknowns. Should be possible to pick this
        # from pp.STATE or pp.ITERATE

        # 1. Get all variables present in this equation.
        # The variable finder is implemented in a special function, aimed at recursion
        # through the operator tree.
        # Uniquify by making this a set, and then sort on variable id
        variables = sorted(
            list(set(self._find_subtree_variables(self._operator))),
            key=lambda var: var.id,
        )

        # 2. Get a mapping between variables (*not* only MergedVariables) and their
        # indices according to the DofManager. This is needed to access the state of
        # a variable when parsing the equation to Ad format.

        # For each variable, get the global index
        inds = []
        variable_ids = []
        prev_time = []
        prev_iter = []
        for variable in variables:
            # Indices (in DofManager sense) of this variable. Will be built gradually
            # for MergedVariables, in one go for plain Variables.
            ind_var = []
            prev_time.append(variable.prev_time)
            prev_iter.append(variable.prev_iter)

            if isinstance(variable, (pp.ad.MergedVariable, operators.MergedVariable)):
                # Loop over all subvariables for the merged variable
                for i, sub_var in enumerate(variable.sub_vars):
                    # Store dofs
                    ind_var.append(dof_manager.dof_ind(sub_var.g, sub_var._name))
                    if i == 0:
                        # Store id of variable, but only for the first one; we will
                        # concatenate the arrays in ind_var into one array
                        variable_ids.append(variable.id)
            else:
                # This is a variable that lives on a single grid
                ind_var.append(dof_manager.dof_ind(variable.g, variable._name))
                variable_ids.append(variable.id)

            # Gather all indices for this variable
            inds.append(np.hstack([i for i in ind_var]))

        return inds, variable_ids, prev_time, prev_iter

    def _identify_subtree_discretizations(
        self, op: operators.Operator, discr: List
    ) -> List:
        """Recursive search in the tree of this operator to identify all discretizations
        represented in the operator.
        """
        if len(op.tree.children) > 0:
            # Go further in recursion
            for child in op.tree.children:
                discr += self._identify_subtree_discretizations(child, [])

        if isinstance(op, _MergedOperator):
            # We have reached the bottom; this is a disrcetization (example: mpfa.flux)
            discr.append(op)

        return discr

    def _identify_discretizations(self) -> Dict[_MergedOperator, grid_like_type]:
        """Perform a recursive search to find all discretizations present in the
        operator tree. Uniquify the list to avoid double computations.

        """
        all_discr = self._identify_subtree_discretizations(self._operator, [])
        return _uniquify_discretization_list(all_discr)

    def discretize(self, gb: pp.GridBucket) -> None:
        """Perform discretization operation on all discretizations identified in
        the tree of this operator, using data from gb.

        IMPLEMENTATION NOTE: The discretizations are identified at initialization of
        this operator - would it be better to identify them just before discretization?

        """
        _discretize_from_list(self.discretizations, gb)

    def to_ad(
        self,
        gb: pp.GridBucket,
        state: Optional[np.ndarray] = None,
    ):
        """Evaluate the residual and Jacobian matrix for a given state.

        Parameters:
            gb (pp.GridBucket): GridBucket used to represent the problem. Will be used
                to parse the operators that combine to form this Equation..
            state (np.ndarray, optional): State vector for which the residual and its
                derivatives should be formed. If not provided, the state will be pulled from
                the previous iterate (if this exists), or alternatively from the state
                at the previous time step.

        Returns:
            An Ad-array representation of the residual and Jacbobian.

        """
        # Parsing in two stages: First make an Ad-representation of the variable state
        # (this must be done jointly for all variables of the Equation to get all
        # derivatives represented). Then parse the equation by traversing its
        # tree-representation, and parse and combine individual operators.

        # Initialize variables
        prev_vals = np.zeros(self._dof_manager.num_dofs())

        populate_state = state is None
        if populate_state:
            state = np.zeros(self._dof_manager.num_dofs())

        assert state is not None
        for (g, var) in self._dof_manager.block_dof:
            ind = self._dof_manager.dof_ind(g, var)
            if isinstance(g, tuple):
                prev_vals[ind] = gb.edge_props(g, pp.STATE)[var]
            else:
                prev_vals[ind] = gb.node_props(g, pp.STATE)[var]

            if populate_state:
                if isinstance(g, tuple):
                    try:
                        state[ind] = gb.edge_props(g, pp.STATE)[pp.ITERATE][var]
                    except KeyError:
                        prev_vals[ind] = gb.edge_props(g, pp.STATE)[var]
                else:
                    try:
                        state[ind] = gb.node_props(g, pp.STATE)[pp.ITERATE][var]
                    except KeyError:
                        state[ind] = gb.node_props(g, pp.STATE)[var]

        # Initialize Ad variables with the current iterates

        # The size of the Jacobian matrix will always be set according to the
        # variables found by the DofManager in the GridBucket.
        #
        # NOTE: This implies that to derive a subsystem from the Jacobian
        # matrix of this Expression will require restricting the columns of
        # this matrix.

        # First generate an Ad array (ready for forward Ad) for the full set.
        ad_vars = initAdArrays([state])[0]

        # Next, the Ad array must be split into variables of the right size
        # (splitting impacts values and number of rows in the Jacobian, but
        # the Jacobian columns must stay the same to preserve all cross couplings
        # in the derivatives).

        # Dictionary which mapps from Ad variable ids to Ad_array.
        self._ad: Dict[int, pp.ad.Ad_array] = {}

        # Loop over all variables, restrict to an Ad array corresponding to
        # this variable.
        for (var_id, dof) in zip(self._variable_ids, self._variable_dofs):
            ncol = state.size
            nrow = np.unique(dof).size
            # Restriction matrix from full state (in Forward Ad) to the specific
            # variable.
            R = sps.coo_matrix(
                (np.ones(nrow), (np.arange(nrow), dof)), shape=(nrow, ncol)
            ).tocsr()
            self._ad[var_id] = R * ad_vars

        # Also make mappings from the previous iteration.
        # This is simpler, since it is only a matter of getting the residual vector
        # correctly (not Jacobian matrix).

        prev_iter_vals_list = [state[ind] for ind in self._prev_iter_dofs]
        self._prev_iter_vals = {
            var_id: val
            for (var_id, val) in zip(self._prev_iter_ids, prev_iter_vals_list)
        }

        # Also make mappings from the previous time step.
        prev_vals_list = [prev_vals[ind] for ind in self._prev_time_dofs]
        self._prev_vals = {
            var_id: val for (var_id, val) in zip(self._prev_time_ids, prev_vals_list)
        }

        # Parse operators. This is left to a separate function to facilitate the
        # necessary recursion for complex operators.
        eq = self._parse_operator(self._operator, gb)

        return eq

    def _parse_operator(self, op: operators.Operator, gb):
        """TODO: Currently, there is no prioritization between the operations; for
        some reason, things just work. We may need to make an ordering in which the
        operations should be carried out. It seems that the strategy of putting on
        hold until all children are processed works, but there likely are cases where
        this is not the case.
        """

        # The parsing strategy depends on the operator at hand:
        # 1) If the operator is a Variable, it will be represented according to its
        #    state.
        # 2) If the operator is a leaf in the tree-representation of the equation,
        #    parsing is left to the operator itself.
        # 3) If the operator is formed by combining other operators lower in the tree,
        #    parsing is handled by first evaluating the children (leads to recursion)
        #    and then perform the operation on the result.

        # Check for case 1 or 2
        if isinstance(op, pp.ad.Variable) or isinstance(op, operators.Variable):
            # Case 1: Variable

            # How to access the array of (Ad representation of) states depends on wether
            # this is a single or combined variable; see self.__init__, definition of
            # self._variable_ids.
            # TODO no differecen between merged or no merged variables!?
            if isinstance(op, pp.ad.MergedVariable) or isinstance(
                op, operators.MergedVariable
            ):
                if op.prev_time:
                    return self._prev_vals[op.id]
                elif op.prev_iter:
                    return self._prev_iter_vals[op.id]
                else:
                    return self._ad[op.id]
            else:
                if op.prev_time:
                    return self._prev_vals[op.id]
                elif op.prev_iter or not (
                    op.id in self._ad
                ):  # TODO make it more explicit that op corresponds to a non_ad_variable?
                    # e.g. by op.id in non_ad_variable_ids.
                    return self._prev_iter_vals[op.id]
                else:
                    return self._ad[op.id]
        elif op.is_leaf():
            # Case 2
            return op.parse(gb)

        # This is not an atomic operator. First parse its children, then combine them
        tree = op.tree
        results = [self._parse_operator(child, gb) for child in tree.children]

        def get_shape(mat):
            # Get shape of a matrix
            if isinstance(mat, (pp.ad.Ad_array, pp.ad.forward_mode.Ad_array)):
                return mat.jac.shape
            else:
                return mat.shape

        def error_message(operation):
            # Helper function to format error message
            msg_0 = self._parse_readable(tree.children[0])
            msg_1 = self._parse_readable(tree.children[1])

            nl = "\n"
            msg = (
                f"Ad parsing: Error when {operation}\n"
                + "  "
                + msg_0
                + nl
                + "with"
                + nl
                + "  "
                + msg_1
                + nl
            )

            msg += (
                f"Matrix sizes are {get_shape(results[0])} and "
                f"{get_shape(results[1])}"
            )
            return msg

        # Combine the results
        if tree.op == operators.Operation.add:
            # To add we need two objects
            assert len(results) == 2

            # Convert any vectors that mascarade as a nx1 (1xn) scipy matrix
            self._ravel_scipy_matrix(results)

            if isinstance(results[0], np.ndarray):
                # With the implementation of Ad arrays, addition does not
                # commute for combinations with numpy arrays. Switch the order
                # of results, and everything works.
                results = results[::-1]
            try:
                return results[0] + results[1]
            except ValueError:
                msg = error_message("adding")
                raise ValueError(msg)

        elif tree.op == operators.Operation.sub:
            # To subtract we need two objects
            assert len(results) == 2

            # Convert any vectors that mascarade as a nx1 (1xn) scipy matrix
            self._ravel_scipy_matrix(results)

            factor = 1

            if isinstance(results[0], np.ndarray):
                # With the implementation of Ad arrays, subtraction does not
                # commute for combinations with numpy arrays. Switch the order
                # of results, and everything works.
                results = results[::-1]
                factor = -1

            try:
                return factor * (results[0] - results[1])
            except ValueError:
                msg = error_message("subtracting")
                raise ValueError(msg)

        elif tree.op == operators.Operation.mul:
            # To multiply we need two objects
            assert len(results) == 2

            if isinstance(results[0], np.ndarray) and isinstance(
                results[1], (pp.ad.Ad_array, pp.ad.forward_mode.Ad_array)
            ):
                # In the implementation of multiplication between an Ad_array and a
                # numpy array (in the forward mode Ad), a * b and b * a do not
                # commute. Flip the order of the results to get the expected behavior.
                results = results[::-1]
            try:
                return results[0] * results[1]
            except ValueError:
                if isinstance(
                    results[0], (pp.ad.Ad_array, pp.ad.forward_mode.Ad_array)
                ) and isinstance(results[1], np.ndarray):
                    # Special error message here, since the information provided by
                    # the standard method looks like a contradiction.
                    # Move this to a helper method if similar cases arise for other
                    # operations.
                    msg_0 = self._parse_readable(tree.children[0])
                    msg_1 = self._parse_readable(tree.children[1])
                    nl = "\n"
                    msg = (
                        "Error when right multiplying \n"
                        + f"  {msg_0}"
                        + nl
                        + "with"
                        + nl
                        + f"  numpy array {msg_1}"
                        + nl
                        + f"Size of arrays: {results[0].val.size} and {results[1].size}"
                        + nl
                        + "Did you forget some parantheses?"
                    )

                else:
                    msg = error_message("multiplying")
                raise ValueError(msg)

        elif tree.op == operators.Operation.evaluate:
            # This is a function, which should have at least one argument
            assert len(results) > 1
            return results[0].func(*results[1:])

        elif tree.op == operators.Operation.apply:
            assert len(results) > 1
            return results[0].apply(*results[1:])

        elif tree.op == operators.Operation.div:
            return results[0] / results[1]

        else:
            raise ValueError("Should not happen")

    def _parse_readable(self, op: operators.Operator) -> str:
        # Make a human-readable error message related to a parsing error.
        # NOTE: The exact formatting should be considered work in progress,
        # in particular when it comes to function evaluation.

        # There are three cases to consider: Either the operator is a leaf,
        # it is a composite operator with a name, or it is a general composite
        # operator.
        if op.is_leaf():
            # Leafs are represented by their strings.
            return str(op)
        elif op._name is not None:
            # Composite operators that have been given a name (possibly
            # with a goal of simple identification of an error)
            return op._name

        # General operator. Split into its parts by recursion.
        tree = op.tree

        child_str = [self._parse_readable(child) for child in tree.children]

        is_func = False

        if tree.op == operators.Operation.add:
            operator_str = "+"
        elif tree.op == operators.Operation.sub:
            operator_str = "-"
        elif tree.op == operators.Operation.mul:
            operator_str = "*"
        elif tree.op == operators.Operation.div:
            operator_str = "/"
        elif tree.op in [operators.Operation.evaluate, operators.Operation.apply]:
            # TODO: This has not really been tested.
            is_func = True
        else:
            # TODO: This corresponds to unknown (to EK) cases.
            print("Have not implemented string parsing of this operator")

        if is_func:
            # TODO: Not sure what to write here
            msg = f"{child_str[0]} evaluated on ("
            for child in range(1, len(child_str)):
                msg += f"{child_str[child]}, "

            msg += ")"
            return msg
        else:
            # TODO: Should we try to give parantheses here?
            return f"{child_str[0]} {operator_str} {child_str[1]}"

    def _ravel_scipy_matrix(self, results):
        # In some cases, parsing may leave what is essentially an array, but with the
        # format of a scipy matrix. This must be converted to a numpy array before
        # moving on.
        # Note: It is not clear that this conversion is meaningful in all cases, so be
        # cautious with adding this extra parsing to more operations.
        for i, res in enumerate(results):
            if isinstance(res, sps.spmatrix):
                assert res.shape[0] == 1 or res.shape[1] == 1
                results[i] = res.toarray().ravel()


class EquationManager:
    """Representation of a set of equations specified on Ad form.

    The equations are tied to a specific GridBucket, with variables fixed in a
    corresponding DofManager. Both these are set on initialization, and should
    not be modified later.

    Central methods are:
        discretize(): Discretize all operators identified in the set equations.
        assemble_matrix_rhs(): Provide a Jacobian matrix and residual for the
            current state in the GridBucket.

    TODO: Add functionality to derive subset of equations, fit for splitting
    algorithms.

    Attributes:
        gb (pp.GridBucket): Mixed-dimensional grid on which this EquationManager
            operates.
        dof_manager (pp.DofManager): Degree of freedom manager used for this
            EquationManager.
        equations (List of Expressions): Equations assigned to this EquationManager.
            can be expanded by direct addition to the list.
        variables (Dict): Mapping from grids or grid tuples (interfaces) to Ad
            variables. These are set at initialization from the GridBucket, and should
            not be changed later.

    """

    def __init__(
        self,
        gb: pp.GridBucket,
        dof_manager: pp.DofManager,
        equations: Optional[List[Expression]] = None,
    ) -> None:
        """Initialize the EquationManager.

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid for this EquationManager.
            dof_manager (pp.DofManager): Degree of freedom manager.
            equations (List, Optional): List of equations. Defaults to empty list.

        """
        self.gb = gb

        # Inform mypy about variables, and then set them by a dedicated method.
        self.variables: Dict[grid_like_type, Dict[str, "pp.ad.Variable"]]
        self._set_variables(gb)

        if equations is None:
            self.equations: List[Expression] = []
        else:
            self.equations = equations

        self.dof_manager: pp.DofManager = dof_manager

    def _set_variables(self, gb):
        # Define variables as specified in the GridBucket
        variables = {}
        for g, d in gb:
            variables[g] = {}
            for var, info in d[pp.PRIMARY_VARIABLES].items():
                variables[g][var] = operators.Variable(var, info, g)

        for e, d in gb.edges():
            variables[e] = {}
            num_cells = d["mortar_grid"].num_cells
            for var, info in d[pp.PRIMARY_VARIABLES].items():
                variables[e][var] = operators.Variable(var, info, e, num_cells)

        self.variables = variables
        # Define discretizations

    def merge_variables(
        self, grid_var: Sequence[Tuple[grid_like_type, str]]
    ) -> "pp.ad.MergedVariable":
        """Concatenate a variable defined over several grids or interface between grids,
        that is, a mortar grid.

        The merged variable can be used to define mathematical operations on multiple
        grids simultaneously (provided it is combined with other operators defined on
        the same grids).

        NOTE: Merged variables are assigned unique ids (see documentation of
        Variable and MergedVariable), thus two MergedVariables will have different
        ids even if they represent the same combination of grids and variables.
        This does not impact the parsing of the variables into numerical values.

        Returns:
            pp.ad.MergedVariable: Joint representation of the variable on the specified
                grids.

        """
        return pp.ad.MergedVariable([self.variables[g][v] for g, v in grid_var])

    def variable(self, grid_like: grid_like_type, variable: str) -> "pp.ad.Variable":
        """Get a variable for a specified grid (or interface between grids, that is
        a mortar grid.

        Subsequent calls of this method with the same grid and variable will return
        references to the same variable.

        Returns:
            pp.ad.Variable: Ad representation of a variable.

        """
        return self.variables[grid_like][variable]

    def variable_state(
        self, grid_var: List[Tuple[pp.Grid, str]], state: np.ndarray
    ) -> List[np.ndarray]:
        # This should likely be placed somewhere else
        values: List[np.ndarray] = []
        for item in grid_var:
            ind: np.ndarray = self.dof_manager.dof_ind(*item)
            values.append(state[ind])

        return values

    def assemble_matrix_rhs(
        self,
        state: Optional[np.ndarray] = None,
    ) -> Tuple[sps.spmatrix, np.ndarray]:
        """Assemble residual vector and Jacobian matrix with respect to the current
        state represented in self.gb.

        As an experimental feature, subset of variables and equations can also be
        assembled. This functionality may be moved somewhere else in the future.

        Parameters:
            state (np.ndarray, optional): State vector to assemble from. If not provided,
                the default behavior of pp.ad.Expression.to_ad() will be followed.

        Returns:
            sps.spmatrix: Jacobian matrix corresponding to the current variable state,
                as found in self.gb.
            np.ndarray: Residual vector corresponding to the current variable state,
                as found in self.gb.

        """
        mat: List[sps.spmatrix] = []
        b: List[np.ndarray] = []

        for eq in self.equations:
            ad = eq.to_ad(self.gb, state)

            # EK: Comment out this part for now; we may need something like this
            # when we get around to implementing subsystems.
            # The columns of the Jacobian has the size of the local variables.
            # Map these to the global ones
            # local_dofs = eq.local_dofs(true_ad_variables=variables)
            # if variables is not None:
            #    local_dofs = self.dof_manager.transform_dofs(local_dofs, var=names)

            # num_local_dofs = local_dofs.size
            # projection = sps.coo_matrix(
            #    (np.ones(num_local_dofs), (np.arange(num_local_dofs), local_dofs)),
            #    shape=(num_local_dofs, num_global_dofs),
            # )
            # mat.append(ad.jac * projection)
            mat.append(ad.jac)
            # Concatenate the residuals
            # Multiply by -1 to move to the rhs
            b.append(-ad.val)

        A = sps.bmat([[m] for m in mat]).tocsr()
        rhs = np.hstack([vec for vec in b])
        return A, rhs

    def discretize(self, gb: pp.GridBucket) -> None:
        """Loop over all discretizations in self.equations, find all discretizations
        and discretize.

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid from which parameters etc. will
                be taken.

        """
        # Somehow loop over all equations, discretize identified objects
        # (but should also be able to do rediscretization based on
        # dependency graph etc).

        # List of discretizations, build up by iterations over all equations
        discr: List = []
        for eqn in self.equations:
            # This will expand the list discr with new discretizations.
            # The list may contain duplicates.
            discr = eqn._identify_subtree_discretizations(eqn._operator, discr)

        # Uniquify to save computational time, then discretize.
        unique_discr = _uniquify_discretization_list(discr)
        _discretize_from_list(unique_discr, gb)

    def __repr__(self) -> str:
        s = (
            "Equation manager for mixed-dimensional grid with "
            f"{self.gb.num_graph_nodes()} grids and {self.gb.num_graph_edges()}"
            " interfaces.\n"
        )

        var = []
        for g, _ in self.gb:
            for v in self.variables[g]:
                var.append(v)

        unique_vars = list(set(var))
        s += "Variables present on at least one grid or interface:\n\t"
        s += ", ".join(unique_vars) + "\n"

        if self.equations is not None:
            eq_names = [eq.name for eq in self.equations]
            s += f"In total {len(self.equations)} equations, with names: \n\t"
            s += ", ".join(eq_names)

        return s


def _uniquify_discretization_list(all_discr):
    """From a list of Ad discretizations (in an Operator), define a unique list
    of discretization-keyword combinations.

    The intention is to avoid that what is essentially the same discretization
    operation is executed twice. For instance, if the list all_discr contains
    elements

        Mpfa(key1).flux, Mpfa(key2).flux and Mpfa(key1).bound_flux,

    where key1 and key2 are different parameter keywords, the function will
    register Mpfa(key1) and Mpfa(key2) (since these use data specified by different
    parameter keywords) but ignore the second instance Mpfa(key1), since this
    discretization is already registered.

    """
    discr_type = Union["pp.Discretization", "pp.AbstractInterfaceLaw"]
    unique_discr_grids: Dict[discr_type, List[grid_like_type]] = {}

    # Mapping from discretization classes to the discretization.
    # We needed this for some reason..
    cls_obj_map = {}
    # List of all combinations of discretizations and parameter keywords covered.
    cls_key_covered = []

    for discr in all_discr:
        # Get the class of the underlying dicsretization, so MpfaAd will return Mpfa.
        cls = discr.discr.__class__
        # Parameter keyword for this discretization
        param_keyword = discr.keyword

        # This discretization-keyword combination
        key = (cls, param_keyword)

        if key in cls_key_covered:
            # If this has been encountered before, we add grids not earlier associated
            # with this discretization to the existing list.
            # of grids.
            # Map from discretization class to Ad discretization
            d = cls_obj_map[cls]
            for g in discr.grids:
                if g not in unique_discr_grids[d]:
                    unique_discr_grids[d].append(g)
        else:
            # Take note we have now encountered this discretization and parameter keyword.
            cls_obj_map[cls] = discr.discr
            cls_key_covered.append(key)

            # Add new discretization with associated list of grids.
            # Need a copy here to avoid assigning additional grids to this
            # discretization (if not copy, this may happen if
            # the key-discr combination is encountered a second time and the
            # code enters the if part of this if-else).
            unique_discr_grids[discr.discr] = discr.grids.copy()

    return unique_discr_grids


def _discretize_from_list(
    discretizations: Dict,
    gb: pp.GridBucket,
) -> None:
    """For a list of (ideally uniquified) discretizations, perform the actual
    discretization.
    """
    for discr in discretizations:
        # discr is a discretization (on node or interface in the GridBucket sense)

        # Loop over all grids (or GridBucket edges), do discretization.
        for g in discretizations[discr]:
            if isinstance(g, tuple):
                data = gb.edge_props(g)
                g_primary, g_secondary = g
                d_primary = gb.node_props(g_primary)
                d_secondary = gb.node_props(g_secondary)
                discr.discretize(g_primary, g_secondary, d_primary, d_secondary, data)
            else:
                data = gb.node_props(g)
                try:
                    discr.discretize(g, data)
                except NotImplementedError:
                    # This will likely be GradP and other Biot discretizations
                    pass