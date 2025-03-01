"""The module contains functions for distance computations."""
from __future__ import annotations

from typing import Union

import numpy as np
from scipy.spatial import distance as scidist

import porepy as pp


def segment_set(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute distance and closest points between sets of line segments.

    Parameters:
        start: ``shape=(nd, num_segments)``

            Start points of segments.
        end: ``shape=(nd, num_segments)``

            End points of segments.

    Returns:
        A tuple of 2 elements.

        :obj:`~numpy.ndarray`: ``shape=(num_segments, num_segments)``

            Distances between segments.
        :obj:`~numpy.ndarray`: ``shape=(num_segments, num_segments, nd)``

            For segment ``i`` and ``j``, element ``[i, j]`` gives the point on ``i``
            closest to segment ``j``.

    """

    if start.size < 4:
        start = start.reshape((-1, 1))
    if end.size < 4:
        end = end.reshape((-1, 1))

    nd: int = start.shape[0]
    ns: int = start.shape[1]

    d = np.zeros((ns, ns))
    cp = np.zeros((ns, ns, nd))

    for i in range(ns):
        cp[i, i, :] = start[:, i] + 0.5 * (end[:, i] - start[:, i])
        dl: np.ndarray
        cpi: np.ndarray
        cpj: np.ndarray
        dl, cpi, cpj = segment_segment_set(
            start[:, i], end[:, i], start[:, i + 1 :], end[:, i + 1]
        )
        dl[i, i + 1 :] = dl
        dl[i + 1 :, i] = dl
        cp[i, i + 1 :] = cpi
        cp[i + 1 :, i] = cpj

    return d, cp


def segment_segment_set(
    start: np.ndarray, end: np.ndarray, start_set: np.ndarray, end_set: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute distance and closest points between a segment and a set of segments.

    Note:
        Algorithm can be found at http://geomalgorithms.com/code.html
        (see file "C07_Line_Line_Distance.cpp",
        function ``dist3D_Segment_to_Segment()``).

    Parameters:
        start:  ``shape=(nd, 1)``

            Start point of the main segment.
        end: ``shape=(nd, 1)``

            End point of the main segment.
        start_set: ``shape=(nd, num_segments)``

            Start points for the segment set.
        end_set: ``shape=(nd, num_segments)``

            End points for the segment set.

    Returns:
        A tuple of 3 elements.

        :obj:`~numpy.ndarray`: ``shape=(num_segments,)``

            The distance from the main segment to each of the segments in the set.
        :obj:`~numpy.ndarray`: ``shape=(nd, num_segments)``

            For each segment in the segment set, the point closest on the main segment.
        :obj:`~numpy.ndarray`: ``shape=(nd, num_segments)``

            For each segment in the segment set, the point closest on the secondary
            segment.

    """
    start = start.reshape((-1, 1))
    end = end.reshape((-1, 1))

    if start_set.size < 4:
        start_set = start_set.reshape((-1, 1))
        end_set = end_set.reshape((-1, 1))

    # For the rest of the algorithm, see the webpage referred to above for details.
    d1: np.ndarray = end - start
    d2: np.ndarray = end_set - start_set
    d_starts: np.ndarray = start - start_set

    def dot(v1, v2):
        return np.sum(v1 * v2, axis=0)

    dot_1_1 = dot(d1, d1)
    dot_1_2 = dot(d1, d2)
    dot_2_2 = dot(d2, d2)
    dot_1_starts = dot(d1, d_starts)
    dot_2_starts = dot(d2, d_starts)
    discr = dot_1_1 * dot_2_2 - dot_1_2**2

    # Variable used to fine almost parallel lines. Sensitivity to this value has not
    # been tested.
    SMALL_TOLERANCE: float = 1e-8 * np.minimum(dot_1_1, np.min(dot_2_2))

    sc = discr.copy()
    sN = discr.copy()
    sD = discr.copy()
    tc = discr.copy()
    tN = discr.copy()
    tD = discr.copy()

    parallel: np.ndarray = discr < SMALL_TOLERANCE
    not_parallel: np.ndarray = np.logical_not(parallel)

    sN[parallel] = 0
    sD[parallel] = 1
    tN[parallel] = dot_2_starts[parallel]
    tD[parallel] = dot_2_2[parallel]

    sN[not_parallel] = (
        dot_1_2[not_parallel] * dot_2_starts[not_parallel]
        - dot_2_2[not_parallel] * dot_1_starts[not_parallel]
    )
    tN[not_parallel] = (
        dot_1_1 * dot_2_starts[not_parallel]
        - dot_1_2[not_parallel] * dot_1_starts[not_parallel]
    )

    # sc < 0 => the s=0 edge is visible
    s0_visible = np.logical_and(not_parallel, sN < 0)
    # sc > 1  => the s=1 edge is visible
    s1_visible = np.logical_and(not_parallel, sN > sD)
    sN[s0_visible] = 0.0
    tN[s0_visible] = dot_2_starts[s0_visible]
    tD[s0_visible] = dot_2_2[s0_visible]

    sN[s1_visible] = sD[s1_visible]
    tN[s1_visible] = dot_1_2[s1_visible] + dot_2_starts[s1_visible]
    tD[s1_visible] = dot_2_2[s1_visible]

    t0_visible = tN < 0

    pos_dot_1_start = np.logical_and(t0_visible, dot_1_starts > 0)
    dot_1_start_g_dot_1_1 = np.logical_and(t0_visible, -dot_1_starts > dot_1_1)
    other = np.logical_and(
        t0_visible,
        np.logical_and(
            np.logical_not(pos_dot_1_start), np.logical_not(dot_1_start_g_dot_1_1)
        ),
    )

    tN[t0_visible] = 0
    sN[pos_dot_1_start] = 0
    sN[dot_1_start_g_dot_1_1] = sD[dot_1_start_g_dot_1_1]
    sN[other] = -dot_1_starts[other]
    sD[other] = dot_1_1

    t1_visible = tN > tD
    case_1 = np.logical_and(t1_visible, (-dot_1_starts + dot_1_2) < 0)
    case_2 = np.logical_and(t1_visible, (-dot_1_starts + dot_1_2) > dot_1_1)
    case_3 = np.logical_and(
        t1_visible, np.logical_and(np.logical_not(case_1), np.logical_not(case_2))
    )

    tN[t1_visible] = tD[t1_visible]
    sN[case_1] = 0
    sN[case_2] = sD[case_2]
    sN[case_3] = -dot_1_starts[case_3] + dot_1_2[case_3]
    sD[case_3] = dot_1_1

    sc = sN / sD
    sc[sN < SMALL_TOLERANCE] = 0

    tc = tN / tD
    tc[tN < SMALL_TOLERANCE] = 0

    # get the difference of the two closest points
    dist = d_starts + sc * d1 - tc * d2

    cp1 = start + d1 * sc
    cp2 = start_set + d2 * tc

    return np.sqrt(np.sum(np.power(dist, 2), axis=0)), cp1, cp2


def points_segments(
    p: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute distances between points and line segments.

    The function also returns the closest points on the segments.

    Parameters:
        p: ``shape=(nd, num_points)``

            Individual points.
        start: ``shape=(nd, num_segments)``

            Start points of segments.
        end: ``shape=(nd, num_segments)``

            End point of segments.

    Returns:
        A tuple of 2 elements.

        :obj:`~numpy.ndarray`: ``shape=(num_points, num_segments)``

            Distances.
        :obj:`~numpy.ndarray`: ``shape=(num_points, num_segments, nd)``

            Points on the segments closest to the individual points.

    """
    if start.size < 4:
        start = start.reshape((-1, 1))
        end = end.reshape((-1, 1))
    if p.size < 4:
        p = p.reshape((-1, 1))

    # Number of points - the case of an empty array needs special handling.
    if p.size > 0:
        num_p = p.shape[1]
    else:
        num_p = 0

    num_l = start.shape[1]
    d = np.zeros((num_p, num_l))

    line = end - start
    lengths = np.sqrt(np.sum(line * line, axis=0))

    nd = p.shape[0]
    # Closest points
    cp = np.zeros((num_p, num_l, nd))

    # We need to compare all points to all segments.
    # The implementation uses an outer for-loop, followed by an inner (almost)
    # vectorized part. Time consumption depends on the number of iterations in the outer
    # loop, so make a choice depending on whether there are more segments or points. For
    # borderline cases, more elaborate choices may be better, but the current form
    # should do for extreme cases.
    if num_p < num_l:
        for pi in range(num_p):
            # Project the vectors from start to point onto the line, and compute
            # relative length
            v = p[:, pi].reshape((-1, 1)) - start
            proj = np.sum(v * line, axis=0) / lengths**2

            # Projections with length less than zero have the closest point at start
            less = np.ma.less_equal(proj, 0)
            d[pi, less] = point_pointset(p[:, pi], start[:, less])
            cp[pi, less, :] = np.swapaxes(start[:, less], 1, 0)
            # Similarly, above one signifies closest to end
            above = np.ma.greater_equal(proj, 1)
            d[pi, above] = point_pointset(p[:, pi], end[:, above])
            cp[pi, above, :] = np.swapaxes(end[:, above], 1, 0)

            # Points inbetween
            between = np.logical_not(np.logical_or(less, above).data)
            proj_p = start[:, between] + proj[between] * line[:, between]
            d[pi, between] = point_pointset(p[:, pi], proj_p)
            cp[pi, between, :] = np.swapaxes(proj_p, 1, 0)
    else:
        for ei in range(num_l):
            # Project the vectors from start to point onto the line, and compute the
            # relative length.
            v = p - start[:, ei].reshape((-1, 1))
            proj = np.sum(v * line[:, ei].reshape((-1, 1)), axis=0) / lengths[ei] ** 2

            # Projections with length less than zero have the closest point at start.
            less = np.ma.less_equal(proj, 0)
            d[less, ei] = point_pointset(start[:, ei], p[:, less])
            cp[less, ei, :] = start[:, ei]
            # Similarly, above one signifies closest to end
            above = np.ma.greater_equal(proj, 1)
            d[above, ei] = point_pointset(end[:, ei], p[:, above])
            cp[above, ei, :] = end[:, ei]

            # Points inbetween
            between = np.logical_not(np.logical_or(less, above).data)
            proj_p = start[:, ei].reshape((-1, 1)) + proj[between] * line[
                :, ei
            ].reshape((-1, 1))
            for proj_i, bi in enumerate(np.where(between)[0]):
                d[bi, ei] = np.min(point_pointset(proj_p[:, proj_i], p[:, bi]))
                cp[bi, ei, :] = proj_p[:, proj_i]

    return d, cp


def point_pointset(
    p: np.ndarray, pset: np.ndarray, exponent: Union[float, int] = 2
) -> np.ndarray:
    """Compute distance between a point and a set of points.

    Parameters:
        p: ``shape=(nd, 1)``

            Point from which distances will be computed.
        pset: ``shape=(nd, num_points)``

            Point cloud to which we compute distances.
        exponent: ``default=2``

            Exponent of the norm used.

    Returns:
        Array of distances ``(shape=(num_points,))``.

    """

    # If p is 1D, do a reshape to facilitate broadcasting, but on a copy
    if p.ndim == 1:
        pt = p.reshape((-1, 1))
    else:
        pt = p

    # If the point cloud is a single point, it should still be a ndx1 array.
    if pset.size == 0:
        # In case of empty sets, return an empty zero
        return np.zeros(0)
    elif pset.size < 4:
        pset_copy = pset.reshape((-1, 1))
    else:
        # Call it a copy, even though it isn't
        pset_copy = pset

    return np.power(
        np.sum(np.power(np.abs(pt - pset_copy), exponent), axis=0), 1 / exponent
    )


def pointset(p: np.ndarray, max_diag: bool = False) -> np.ndarray:
    """Compute mutual distance between all points in a point set.

    Parameters:
        p: ``shape=(nd, num_points)``

            A set of points.
        max_diag: ``default=False``

            If True, the diagonal value corresponding to each point is set to twice the
            maximum of the distances for that point, rather than 0.

    Returns:
        Array of distances between points ``(shape=(num_points, num_points))``.

    """
    if p.ndim == 1 or p.ndim == 2 and p.shape[-1] == 1:
        return np.zeros((1, 1))

    # Use scipy spatial function to compute distances between points
    d = scidist.cdist(p.T, p.T, "euclidean")

    if max_diag:
        # Get the maximum per row (could have used column, the distance matrix is
        # symmetric), and add it to the diagonal
        row_max = np.max(d, axis=1)
        d += 2 * np.diag(row_max)

    return d


def points_polygon(
    p: np.ndarray, poly: np.ndarray, tol: float = 1e-5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute distance from points to a polygon. Also find closest point on the
    polygon.

    Parameters:
        p: ``shape=(nd, num_points)``

            Points for which we will compute distances.
        poly: ``shape=(nd, num_vertexes)``

            Vertexes of polygon. Edges are formed by subsequent points.
        tol: ``default=1e-5``

            Tolerance to be used.

    Returns:
        A tuple of 3 elements.

        :obj:`~numpy.ndarray`: ``shape=(num_points,)``

            Distance from points to polygon.
        :obj:`~numpy.ndarray`: ``shape=(nd, num_points,)``

            For each point, the closest point on the polygon.
        :obj:`~numpy.ndarray`: ``shape=(num_points)``

            ``True`` if the point is found in the interior,
            ``False`` if on a bounding segment.

    """
    # The computation is split into two, the closest point can either be in the
    # interior, or at the boundary of the point.

    if p.size < 4:
        p.reshape((-1, 1))

    num_p = p.shape[1]
    num_vert = poly.shape[1]
    nd = p.shape[0]

    # First translate the points so that the first plane is located at the origin
    center = np.mean(poly, axis=1).reshape((-1, 1))
    # Compute copies of polygon and point in new coordinate system
    orig_poly = poly
    poly = poly - center
    orig_p = p
    p = p - center

    # Obtain the rotation matrix that projects p1 to the xy-plane
    rot_p = pp.map_geometry.project_plane_matrix(poly)
    irot = rot_p.transpose()
    poly_rot = rot_p.dot(poly)

    # Sanity check: The points should lay on a plane
    assert np.all(np.abs(poly_rot[2]) < tol)

    poly_xy = poly_rot[:2]

    # Rotate the point set, using the same coordinate system.
    p = rot_p.dot(p)

    in_poly = pp.geometry_property_checks.point_in_polygon(poly_xy, p)

    # Distances
    d = np.zeros(num_p)
    # Closest points
    cp = np.zeros((nd, num_p))

    # For points that are located inside the extrusion of the polygon, the distance is
    # simply the z-coordinate
    d[in_poly] = np.abs(p[2, in_poly])

    # The corresponding closest points are found by inverse rotation of the closest
    # point on the polygon
    cp_inpoly = p[:, in_poly]
    if cp_inpoly.size == 3:
        cp_inpoly = cp_inpoly.reshape((-1, 1))
    cp_inpoly[2, :] = 0
    cp[:, in_poly] = center + irot.dot(cp_inpoly)

    if np.all(in_poly):
        return d, cp, in_poly

    # Next, points that are outside the extruded polygons. These will have their closest
    # point among one of the edges
    start = orig_poly
    end = orig_poly[:, (1 + np.arange(num_vert)) % num_vert]

    outside_poly = np.where(np.logical_not(in_poly))[0]

    d_outside, p_outside = points_segments(orig_p[:, outside_poly], start, end)

    for i, pi in enumerate(outside_poly):
        mi = np.argmin(d_outside[i])
        d[pi] = d_outside[i, mi]
        cp[:, pi] = p_outside[i, mi, :]

    return d, cp, in_poly


def segments_polygon(
    start: np.ndarray, end: np.ndarray, poly: np.ndarray, tol: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the distance from line segments to a polygon.

    Parameters:
        start: ``shape=(nd, num_segments)``

            One endpoint of segments.
        end: ``shape=(nd, num_segments)``

            Other endpoint of segments.
        poly: ``shape=(nd, n_vert)``

            Vertexes of polygon.
        tol: ``default=1e-5``

            Tolerance to be used.

    Returns:
        A tuple of 2 elements.

        :obj:`~numpy.ndarray`: ``(shape=(num_segments,), dtype=double)``

            Distance from segment to polygon.
        :obj:`~numpy.ndarray`: ``shape=(nd, num_segments)``

            Closest point.

    """
    if start.size < 4:
        start = start.reshape((-1, 1))
    if end.size < 4:
        end = end.reshape((-1, 1))

    num_p = start.shape[1]
    nd = start.shape[0]

    d = np.zeros(num_p)
    cp = np.zeros((nd, num_p))

    # First translate the points so that the first plane is located at the origin
    center = np.mean(poly, axis=1).reshape((-1, 1))
    # Compute copies of polygon and point in new coordinate system
    orig_poly = poly
    orig_start = start
    orig_end = end

    poly = poly - center
    start = start - center
    end = end - center

    # Obtain the rotation matrix that projects p1 to the xy-plane
    rot_p = pp.map_geometry.project_plane_matrix(poly)
    irot = rot_p.transpose()
    poly_rot = rot_p.dot(poly)

    # Sanity check: The points should lay on a plane
    assert np.all(np.abs(poly_rot[2]) < tol)

    poly_xy = poly_rot[:2]

    # Rotate the point set, using the same coordinate system.
    start = rot_p.dot(start)
    end = rot_p.dot(end)

    dz = end[2] - start[2]
    non_zero_incline = np.abs(dz) > tol

    # Parametrization along line of intersection point
    t = 0 * dz

    # Intersection point for segments with non-zero incline
    t[non_zero_incline] = -start[2, non_zero_incline] / dz[non_zero_incline]
    # Segments with z=0 along the segment
    zero_along_segment = np.logical_and(
        non_zero_incline, np.logical_and(t >= 0, t <= 1).astype(bool)
    )

    x0 = start + (end - start) * t
    # Check if zero point is inside the polygon.
    inside = pp.geometry_property_checks.point_in_polygon(poly_xy, x0[:2])
    crosses = np.logical_and(inside, zero_along_segment)

    # For points with zero incline, the z-coordinate should be zero for the
    # point to be inside.
    segment_in_plane = np.logical_and(
        np.abs(start[2]) < tol, np.logical_not(non_zero_incline)
    )
    # Check if either start or endpoint is inside the polygon. This leaves the option of
    # the segment crossing the polygon within the plane, but this will be handled by the
    # crossing of segments below.
    endpoint_in_polygon = np.logical_or(
        pp.geometry_property_checks.point_in_polygon(poly_xy, start[:2]),
        pp.geometry_property_checks.point_in_polygon(poly_xy, end[:2]),
    )

    segment_in_polygon = np.logical_and(segment_in_plane, endpoint_in_polygon)

    intersects = np.logical_or(crosses, segment_in_polygon)

    x0[2, intersects] = 0
    cp[:, intersects] = center + irot.dot(x0[:, intersects])

    # Check if we're done, or if we should consider proximity to polygon segments
    if np.all(intersects):
        # The distance is known to be zero, so no need to set it
        return d, cp

    not_found = np.where(np.logical_not(intersects))[0]

    # If we reach this, the minimum is not zero for all segments. The point with minimum
    # distance is then either 1) one endpoint of the segments (point-polygon), 2) found
    # as a segment-segment minimum (segment and boundary of polygon), or 3) anywhere
    # along the segment parallel with polygon.
    poly = orig_poly
    start = orig_start
    end = orig_end

    # Distance from endpoints to polygons
    d_start_poly, cp_s_p, _ = points_polygon(start, poly)
    d_end_poly, cp_e_p, _ = points_polygon(end, poly)

    # Loop over all segments that did not cross the polygon. The minimum is found either
    # by the endpoints, or as between two segments.
    for si in not_found:
        # For starters, assume the closest point is on the start of the segment
        md = d_start_poly[si]
        cp_l = cp_s_p[:, si]

        # Update with the end coordinate if relevant
        if d_end_poly[si] < md:
            md = d_end_poly[si]
            cp_l = cp_e_p[:, si]

        poly_start = poly
        poly_end = np.roll(poly, -1, axis=1)

        ds, cp_s, _ = segment_segment_set(
            start[:, si], end[:, si], poly_start, poly_end
        )

        min_seg = np.argmin(ds)
        if np.any(ds[min_seg] < md):
            md = ds[min_seg]
            cp_l = cp_s[:, min_seg]

        # By now, we have found the minimum
        d[si] = md
        cp[:, si] = cp_l.reshape((1, -1))

    return d, cp


def segment_overlap_segment_set(
    start: np.ndarray,
    end: np.ndarray,
    start_set: np.ndarray,
    end_set: np.ndarray,
    return_indices: bool = False,
    tol: float = 1e-5,
) -> Union[bool, tuple[bool, np.ndarray]]:
    """Detects whether a given segment overlaps the segments in a set.

    The function currently works only for 2D geometries: ``nd == 2``.

    Parameters:
        start: ``shape=(nd,)``

            Start point of a segment.
        end: ``shape=(nd,)``

            End point of a segment.
        start_set: ``shape=(nd, num_segments)``

            Start points of the set of segments.
        end_set: ``shape=(nd, num_segments)``

            End points of the set of segments.
        return_indices: ``default=False``

            Whether the function should also return indices of overlappings.
        tol: ``default=1e-5``

            Tolerance to be used.

    Returns:
        If ``return_indices==True``, we return a tuple of 2 elements.

        :obj:`bool`:
            ``True`` if the segment overlaps any of the segments in the set, ``False``
            otherwise.
        :obj:`~numpy.ndarray`: ``shape=(num_overlapping,)``

            Indices of overlappings.

        Otherwise, the 0th element of the tuple is returned.

    """

    # reshape the input points
    start_set = np.atleast_2d(start_set)[:2].reshape((2, -1))
    end_set = np.atleast_2d(end_set)[:2].reshape((2, -1))

    start = start.reshape(-1)[:2]
    end = end.reshape(-1)[:2]
    norm = np.linalg.norm(start - end)

    # compute the 2d-cross product useful for the collinearity
    cross = np.empty((2, start_set.shape[1]))
    for idx, (i, j) in enumerate(zip(start_set.T, end_set.T)):
        cross[0, idx] = np.cross(j - i, end - start) / norm
        cross[1, idx] = np.cross(start - i, j - i) / norm

    # determine if both conditions are satisfied
    test = np.abs(cross) < tol
    overlap = np.logical_and(test[0, :], test[1, :])

    # return if the segment overlap any of the others and which one
    if return_indices:
        # Note: We need the bool-conversion here since np.any returns np.bool_.
        # See <https://github.com/numpy/numpy/issues/18876>
        return bool(np.any(overlap)), np.where(overlap)[0]

    # return if the segment overlap any of the others
    return bool(np.any(overlap))
