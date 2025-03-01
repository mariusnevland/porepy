"""
Discretization of the source term of an equation for FV methods.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.discretization import Discretization


class ScalarSource(Discretization):
    """
    Discretization of the integrated source term
    int q * dx
    over each grid cell for scalar equations.

    All this function does is returning a zero lhs and
    rhs = param.get_source.keyword.
    """

    def __init__(self, keyword: str) -> None:
        """Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Parameters:
            keyword (str): Identifier of all information used for this
                discretization.
        """
        self.keyword = keyword

    def _key(self) -> str:
        """Get the keyword of this object, on a format friendly to access relevant
        fields in the data dictionary

        Returns:
            String, on the form self.keyword + '_'.

        """
        return self.keyword + "_"

    def ndof(self, sd: pp.Grid) -> int:
        """Return the number of degrees of freedom associated to the method.

        Parameter:
            sd: grid, or a subclass.

        Returns:
            int: the number of degrees of freedom.

        """
        return sd.num_cells

    def assemble_matrix_rhs(
        self, sd: pp.Grid, data: dict
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Return the (null) matrix and right-hand side for a discretization of the
        integrated source term. Also discretize the necessary operators if the data
        dictionary does not contain a source term.

        Parameters:
            sd: grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Returns:
            lhs (sparse dia, self.ndof x self.ndof): Null lhs.
            sources (array, self.ndof): Right-hand side vector.

        The names of data in the input dictionary (data) are:
        param (Parameter Class) with the source field set for self.keyword. The assigned
            source values are assumed to be integrated over the cell volumes.
        """
        return self.assemble_matrix(sd, data), self.assemble_rhs(sd, data)

    def assemble_matrix(self, sd: pp.Grid, data: dict):
        """Return the (null) matrix and for a discretization of the integrated source
        term. Also discretize the necessary operators if the data dictionary does not
        contain a source term.

        Parameters:
            sd (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix (self.ndof x self.ndof): Null system matrix of this
                discretization.
        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        return matrix_dictionary["source"]

    def assemble_rhs(self, sd: pp.Grid, data: dict):
        """Return the rhs for a discretization of the integrated source term. Also
        discretize the necessary operators if the data dictionary does not contain a
        source term.

        Parameters:
            sd (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.array (self.ndof): Right hand side vector representing the
                source.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]

        sources = parameter_dictionary["source"]
        assert sources.size == self.ndof(
            sd
        ), "There should be one source value for each cell"
        return matrix_dictionary["bound_source"] * sources

    def discretize(self, sd: pp.Grid, data: dict):
        """Discretize an integrated source term.

        Parameters:
            sd: grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Stores:
            lhs (sparse dia, self.ndof x self.ndof): Null lhs, stored as
                self._key() + "source".
            sources (array, self.ndof): Right-hand side vector, stored as
                self._key() + "bound_source".

        The names of data in the input dictionary (data) are:
        param (Parameter Class) with the source field set for self.keyword. The assigned
            source values are assumed to be integrated over the cell volumes.
        """
        lhs = sps.csc_matrix((self.ndof(sd), self.ndof(sd)))
        rhs = sps.diags(np.ones(self.ndof(sd))).tocsc()
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        matrix_dictionary["source"] = lhs
        matrix_dictionary["bound_source"] = rhs
