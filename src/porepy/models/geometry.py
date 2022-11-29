"""Geometry definition for simulation setup.

"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

logger = logging.getLogger(__name__)


class ModelGeometry:
    """This class provides geometry related methods and information for a simulation
    model."""

    # Define attributes to be assigned later
    fracture_network: Union[pp.FractureNetwork2d, pp.FractureNetwork3d]
    """Representation of fracture network including intersections."""
    well_network: pp.WellNetwork3d
    """Well network."""
    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""
    box: dict
    """Box-shaped domain. FIXME: change to "domain"? """
    nd: int
    """Ambient dimension."""

    def set_geometry(self) -> None:
        """Define geometry and create a mixed-dimensional grid."""
        # Create fracture network and mixed-dimensional grid
        self.set_fracture_network()
        self.set_md_grid()
        self.nd: int = self.mdg.dim_max()
        # If fractures are present, it is advised to call
        pp.contact_conditions.set_projections(self.mdg)

    def set_fracture_network(self) -> None:
        """Assign fracture network class."""
        self.fracture_network = pp.FractureNetwork2d()

    def mesh_arguments(self) -> dict:
        """Mesh arguments for md-grid creation.

        Returns:
            mesh_args: Dictionary of meshing arguments compatible with
            FractureNetwork.mesh()
                method.

        """
        mesh_args = dict()
        return mesh_args

    def set_md_grid(self) -> None:
        """Create the mixed-dimensional grid.

        A unit square grid with no fractures is assigned by default if
        self.fracture_network contains no fractures. Otherwise, the network's mesh
        method is used.

        The method assigns the following attributes to self:
            mdg (pp.MixedDimensionalGrid): The produced grid bucket. box (dict): The
            bounding box of the domain, defined through minimum and
                maximum values in each dimension.
        """

        if self.fracture_network.num_frac() == 0:
            # Mono-dimensional grid by default
            phys_dims = np.array([1, 1])
            n_cells = np.array([1, 1])
            self.box = pp.geometry.bounding_box.from_points(
                np.array([[0, 0], phys_dims]).T
            )
            g: pp.Grid = pp.CartGrid(n_cells, phys_dims)
            g.compute_geometry()
            self.mdg = pp.meshing.subdomains_to_mdg([[g]])
        else:
            self.mdg = self.fracture_network.mesh(self.mesh_arguments())
            self.box = self.fracture_network.domain

    def subdomains_to_interfaces(
        self, subdomains: list[pp.Grid], codims: Optional[list] = None
    ) -> list[pp.MortarGrid]:
        """Interfaces neighbouring any of the subdomains.

        Args:
            subdomains: Subdomains for which to find interfaces.
            codims: Codimension of interfaces to return. Defaults to [1], i.e.
                only interfaces between one dimension apart.

        Returns:
            list[pp.MortarGrid]: Unique, sorted list of interfaces.
        """
        if codims is None:
            codims = [1]
        interfaces = list()
        for sd in subdomains:
            for intf in self.mdg.subdomain_to_interfaces(sd):
                if intf not in interfaces and intf.codim in codims:
                    interfaces.append(intf)
        return self.mdg.sort_interfaces(interfaces)

    def interfaces_to_subdomains(
        self, interfaces: list[pp.MortarGrid]
    ) -> list[pp.Grid]:
        """Subdomain neighbours of interfaces.

        Parameters:
            interfaces: List of interfaces for which to find subdomains.

        Returns:
            Unique sorted list of all subdomains neighbouring any of the interfaces.

        """
        subdomains = list()
        for interface in interfaces:
            for sd in self.mdg.interface_to_subdomain_pair(interface):
                if sd not in subdomains:
                    subdomains.append(sd)
        return self.mdg.sort_subdomains(subdomains)

    def wrap_grid_attribute(self, grids: list[pp.GridLike], attr: str) -> pp.ad.Matrix:
        """Wrap a grid attribute as an ad matrix.

        Parameters:
            grids: List of grids on which the property is defined. prop: Grid attribute
                to wrap. The attribute should be a ndarray and will be flattened if it
                is not already a vector.

        Returns:
            ad_matrix: The property wrapped as an ad matrix.

        TODO: Test the method (and other methods in this class).

        """
        if len(grids) > 0:
            mat = sps.diags(np.hstack([getattr(g, attr).ravel("F") for g in grids]))
        else:
            mat = sps.csr_matrix((0, 0))
        ad_matrix = pp.ad.Matrix(mat)
        return ad_matrix

    def basis(self, grids: list[pp.GridLike], dim: int = None) -> np.ndarray:
        """Return a cell-wise basis for all subdomains.

        Parameters:
            grids: List of grids on which the basis is defined. dim: Dimension of the
            base. Defaults to self.nd.

        Returns:
            Array (dim) of pp.ad.Matrix, each of which represents a basis function.

        """
        if dim is None:
            dim = self.nd

        assert dim <= self.nd, "Basis functions of higher dimension than the md grid"
        # Collect the basis functions for each dimension
        basis = []
        for i in range(dim):
            basis.append(self.e_i(grids, i, dim))
        # Stack the basis functions horizontally
        return np.hstack(basis)

    def e_i(self, grids: list[pp.GridLike], i: int, dim: int = None) -> np.ndarray:
        """Return a cell-wise basis function.

        Parameters:
            grids: List of grids on which the basis vector is defined. dim (int):
            Dimension of the functions. i (int): Index of the basis function. Note:
            Counts from 0.

        Returns:
            pp.ad.Matrix: Ad representation of a matrix with the basis functions as
                columns.

        """
        if dim is None:
            dim = self.nd
        assert dim <= self.nd, "Basis functions of higher dimension than the md grid"
        assert i < dim, "Basis function index out of range"
        # Collect the basis functions for each dimension
        e_i = np.zeros(dim).reshape(-1, 1)
        e_i[i] = 1
        # expand to cell-wise column vectors.
        num_cells = sum([g.num_cells for g in grids])
        mat = sps.kron(sps.eye(num_cells), e_i)
        return pp.ad.Matrix(mat)

    def local_coordinates(self, subdomains: list[pp.Grid]) -> pp.ad.Matrix:
        """Ad wrapper around tangential_normal_projections for fractures.

        TODO: Extend to all subdomains.

        Parameters:
            subdomains: List of subdomains for which to compute the local coordinates.

        Returns:
            Local coordinates as a pp.ad.Matrix.

        """
        # For now, assert all subdomains are fractures, i.e. dim == nd - 1
        assert all([sd.dim == self.nd - 1 for sd in subdomains])
        if len(subdomains) > 0:
            local_coord_proj_list = [
                self.mdg.subdomain_data(sd)[
                    "tangential_normal_projection"
                ].project_tangential_normal(sd.num_cells)
                for sd in subdomains
            ]
            local_coord_proj = sps.block_diag(local_coord_proj_list)
        else:
            local_coord_proj = sps.csr_matrix((0, 0))
        return pp.ad.Matrix(local_coord_proj)

    def subdomain_projections(self, dim: int):
        """Return the projection operators for all subdomains in md-grid.

        The projection operators restrict or prolong a dim-dimensional quantity from the
        full set of subdomains to any subset. Projection operators are constructed once
        and then stored. If you need to use projection operators based on a different
        set of subdomains, please construct them yourself. Alternatively, compose a
        projection from subset A to subset B as
            P_A_to_B = P_full_to_B * P_A_to_full.

        Parameters:
            dim: Dimension of the quantities to be projected.

        Returns:
            proj: Projection operator.
        """
        name = f"_subdomain_proj_{dim}"
        if hasattr(self, name):
            proj = getattr(self, name)
        else:
            proj = pp.ad.SubdomainProjections(self.mdg.subdomains(), dim)
            setattr(self, name, proj)
        return proj

    def domain_boundary_sides(
        self, g: pp.Grid
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Obtain indices of the faces of a grid that lie on each side of the domain
        boundaries. It is assumed the domain is box shaped.

        TODO: Update this from develop before merging.
        """
        tol = 1e-10
        box = self.box
        east = g.face_centers[0] > box["xmax"] - tol
        west = g.face_centers[0] < box["xmin"] + tol
        if self.nd == 1:
            north = np.zeros(g.num_faces, dtype=bool)
            south = north.copy()
        else:
            north = g.face_centers[1] > box["ymax"] - tol
            south = g.face_centers[1] < box["ymin"] + tol
        if self.nd < 3:
            top = np.zeros(g.num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = g.face_centers[2] > box["zmax"] - tol
            bottom = g.face_centers[2] < box["zmin"] + tol
        all_bf = g.get_boundary_faces()
        return all_bf, east, west, north, south, top, bottom

    # Local basis related methods
    def tangential_component(self, grids: list[pp.Grid]) -> pp.ad.Operator:
        """Compute the tangential component of a vector field.

        Parameters:
            grids: List of grids on which the vector field is defined.

        Returns:
            tangential: Operator extracting tangential component of the vector field and
            expressing it in tangential basis.
        """
        # We first need an inner product (or dot product), i.e. extract the tangential
        # component of the cell-wise vector v to be transformed. Then we want to express
        # it in the tangential basis. The two operations are combined in a single
        # operator composed right to left: v will be hit by first e_i.T (row vector) and
        # secondly t_i (column vector).
        op = sum(
            [
                self.e_i(grids, i, self.nd - 1) * self.e_i(grids, i, self.nd).T
                for i in range(self.nd - 1)
            ]
        )
        op.set_name("tangential_component")
        return op

    def normal_component(self, grids: list[pp.Grid]) -> pp.ad.Operator:
        """Compute the normal component of a vector field.

        Parameters:
            grids: List of grids on which the vector field is defined.

        Returns:
            normal: Operator extracting normal component of the vector field and
            expressing it in normal basis.
        """
        e_n = self.e_i(grids, self.nd - 1, self.nd)
        e_n.set_name("normal_component")
        return e_n.T

    def internal_boundary_normal_to_outwards(
        self, interfaces: list[pp.Grid]
    ) -> pp.ad.Matrix:
        """Flip sign if normal vector points inwards.

        Args:
            interfaces: List of interfaces.

        Returns:
            Matrix with flipped signs if normal vector points inwards.

        This seems a bit messy to me. Let's discuss, EK.
        """
        if hasattr(self, "_internal_boundary_vector_to_outwards_operator"):
            return self._internal_boundary_vector_to_outwards_operator
        if len(interfaces) == 0:
            mat = sps.csr_matrix((0, 0))
        else:
            mat = None
            for intf in interfaces:
                # Extracting matrix for each interface should in theory allow for multiple
                # matrix subdomains, but this is not tested.
                matrix_subdomain = self.mdg.interface_to_subdomain_pair(intf)[0]
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