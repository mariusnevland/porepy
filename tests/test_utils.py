""" Utility functions for the tests.

Access: from test import test_utils.
"""

import os

import numpy as np
import scipy.sparse as sps

import porepy as pp


def permute_matrix_vector(A, rhs, block_dof, full_dof, grids, variables):
    """Permute the matrix and rhs from assembler order to a specified order.

    Args:
        A: global solution matrix as returned by Assembler.assemble_matrix_rhs.
        rhs: global rhs vector as returned by Assembler.assemble_matrix_rhs.
        block_dof: Map coupling a (grid, variable) pair to an block index of A, as
            returned by Assembler.assemble_matrix_rhs.
        full_dof: Number of DOFs for each pair in block_dof, as returned by
            Assembler.assemble_matrix_rhs.

    Returns:
        sps.bmat(A.size): Permuted matrix.
        np.ndarray(b.size): Permuted rhs vector.
    """
    sz = len(block_dof)
    mat = np.empty((sz, sz), dtype=object)
    b = np.empty(sz, dtype=object)
    dof = np.empty(sz, dtype=object)
    # Initialize dof vector
    dof[0] = np.arange(full_dof[0])
    for i in range(1, sz):
        dof[i] = dof[i - 1][-1] + 1 + np.arange(full_dof[i])

    for row in range(sz):
        # Assembler index 0
        i = block_dof[(grids[row], variables[row])]
        b[row] = rhs[dof[i]]
        for col in range(sz):
            # Assembler index 1
            j = block_dof[(grids[col], variables[col])]
            # Put the A block indexed by i and j in mat of running indexes row and col
            mat[row, col] = A[dof[i]][:, dof[j]]

    return sps.bmat(mat, format="csr"), np.concatenate(tuple(b))


def setup_flow_assembler(mdg, method, data_key=None, coupler=None):
    """Setup a standard assembler for the flow problem for a given grid bucket.

    The assembler will be set up with primary variable name 'pressure' on the
    GridBucket nodes, and mortar_flux for the mortar variables.

    Parameters:
        mdg: GridBucket.
        method (EllipticDiscretization).
        data_key (str, optional): Keyword used to identify data dictionary for
            node and edge discretization.
        Coupler (EllipticInterfaceLaw): Defaults to RobinCoulping.

    Returns:
        Assembler, ready to discretize and assemble problem.

    """

    if data_key is None:
        data_key = "flow"
    if coupler is None:
        coupler = pp.RobinCoupling(data_key, method)

    if isinstance(method, pp.MVEM) or isinstance(method, pp.RT0):
        mixed_form = True
    else:
        mixed_form = False

    for _, data in mdg.subdomains(return_data=True):
        if mixed_form:
            data[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1, "faces": 1}}
        else:
            data[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1}}
        data[pp.DISCRETIZATION] = {"pressure": {"diffusive": method}}
    for intf, data in mdg.interfaces(return_data=True):
        g2, g1 = mdg.interface_to_subdomain_pair(intf)
        data[pp.PRIMARY_VARIABLES] = {"mortar_flux": {"cells": 1}}
        data[pp.COUPLING_DISCRETIZATION] = {
            "lambda": {
                g1: ("pressure", "diffusive"),
                g2: ("pressure", "diffusive"),
                intf: ("mortar_flux", coupler),
            }
        }
        data[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    assembler = pp.Assembler(mdg)
    return assembler


def solve_and_distribute_pressure(mdg, assembler):
    """Given an assembler, assemble and solve the pressure equation, and distribute
    the result.

    Parameters:
        GridBucket: Of problem to be solved
        assembler (Assembler):
    """
    assembler.discretize()
    A, b = assembler.assemble_matrix_rhs()
    p = np.linalg.solve(A.A, b)
    assembler.distribute_variable(p)


def compare_arrays(a, b, tol=1e-4, sort=True):
    """Compare two arrays and check that they are equal up to a column permutation.

    Typical usage is to compare coordinate arrays.

    Parameters:
        a, b (np.array): Arrays to be compared. W
        tol (double, optional): Tolerance used in comparison.
        sort (boolean, defaults to True): Sort arrays columnwise before comparing

    Returns:
        True if there is a permutation ind so that all(a[:, ind] == b).
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)

    if not np.all(a.shape == b.shape):
        return False

    if sort:
        a = np.sort(a, axis=0)
        b = np.sort(b, axis=0)

    for i in range(a.shape[1]):
        dist = np.sum((b - a[:, i].reshape((-1, 1))) ** 2, axis=0)
        if dist.min() > tol:
            return False
    for i in range(b.shape[1]):
        dist = np.sum((a - b[:, i].reshape((-1, 1))) ** 2, axis=0)
        if dist.min() > tol:
            return False
    return True


def delete_file(file_name):
    """Delete a file if it exist. Cleanup after tests."""
    if os.path.exists(file_name):
        os.remove(file_name)


def compare_grids(g1, g2):
    """Compare two grids. They are considered equal if the topology and geometry is the
    same.
    """
    if g1.dim != g2.dim:
        return False

    if (g1.num_cells, g1.num_faces, g1.num_nodes) != (
        g2.num_cells,
        g2.num_faces,
        g2.num_nodes,
    ):
        return False

    dfn = g1.face_nodes - g2.face_nodes
    if dfn.data.size > 0 and np.max(np.abs(dfn.data)) > 0.1:
        return False

    dcf = g1.cell_faces - g2.cell_faces
    if dcf.data.size > 0 and np.max(np.abs(dcf.data)) > 0.1:
        return False

    if g1.dim > 0:
        coord = g1.nodes - g2.nodes
    else:
        coord = g1.cell_centers - g2.cell_centers
    dist = np.sum(coord**2, axis=0)
    if dist.max() > 1e-16:
        return False

    # No need to test other geometric quastities; these are processed from those already
    # checked, thus the grids are identical.
    return True


def compare_mortar_grids(mg1, mg2):
    if mg1.dim != mg2.dim:
        return False

    if mg1.num_cells != mg2.num_cells:
        return False

    for key, g1 in mg1.side_grids.items():
        if key not in mg2.side_grids:
            return False
        g2 = mg2.side_grids[key]
        if not compare_grids(g1, g2):
            return False

    return True


def compare_md_grids(mdg1, mdg2):
    for dim in range(4):
        subdomains_1 = mdg1.subdomains(dim=dim)
        subdomains_2 = mdg2.subdomains(dim=dim)
        # Two mdgs are considered equal only if the grids are returned in the same
        # order. This may be overly restrictive, but it will have to do.
        if len(subdomains_1) != len(subdomains_2):
            return False
        for sd1, sd2 in zip(subdomains_1, subdomains_2):
            if not compare_grids(sd1, sd2):
                return False

    # Not sure how to do testing on Mortar grids.
