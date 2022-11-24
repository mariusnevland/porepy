"""
This module contains an implementation of Sneddon's problem.

-> Insert explanation...
"""

from __future__ import annotations


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import porepy as pp
import os
import scipy.optimize as opt
import sympy as sym
import scipy as sps

from typing import Optional, Union
from dataclasses import dataclass

def get_bc_values_michele(
        sd_rock: pp.Grid,
        G: float,
        poi: float,
        p0: float,
        a: float,
        n: int,
        domain_size: tuple[float, float],
        theta: float
):

    def compute_eta(pointset_centers, point):
        """
        Compute the distance of bem segments centers to the
        fracture centre.

        Parameter
        ---------
        pointset_centers: array containing centers of bem segments
        point: fracture centre, middle point of the square domain

        """
        return pp.geometry.distances.point_pointset(pointset_centers, point)

    def get_bem_centers(a, h, n, theta, center):
        """
        Compute coordinates of the centers of the bem segments

        Parameter
        ---------
        a: half fracture length
        h: bem segment length
        n: number of bem segments
        theta: orientation of the fracture
        center: center of the fracture
        """
        bem_centers = np.zeros((3, n))
        x_0 = center[0] - (a - 0.5 * h) * np.sin(theta)
        y_0 = center[1] - (a - 0.5 * h) * np.cos(theta)
        for i in range(0, n):
            bem_centers[0, i] = x_0 + i * h * np.sin(theta)
            bem_centers[1, i] = y_0 + i * h * np.cos(theta)

        return bem_centers

    def analytical_displacements(a, eta, p0, mu, nu):
        """
        Compute Sneddon's analytical solution for the pressurized crack
        problem in question.

        Parameter
        ---------
        a: half fracture length
        eta: distance from fracture centre
        p0: pressure
        mu: shear modulus
        nu: poisson ratio
        """
        cons = (1 - nu) / mu * p0 * a * 2
        return cons * np.sqrt(1 - np.power(eta / a, 2))

    def transform(xc, x, alpha):
        """
        Coordinate transofrmation for the BEM method

        Parameter
        ---------
        xc: coordinates of BEM segment centre
        x: coordinates of boundary faces
        alpha: fracture orientation
        """
        x_bar = np.zeros_like(x)
        x_bar[0, :] = (x[0, :] - xc[0]) * np.cos(alpha) + (x[1, :] - xc[1]) * np.sin(
            alpha)
        x_bar[1, :] = - (x[0, :] - xc[0]) * np.sin(alpha) + (x[1, :] - xc[1]) * np.cos(
            alpha)
        return x_bar

    def get_bc_val(g, bound_faces, xf, h, poi, alpha, du):
        """
        Compute analytical displacement using the BEM method for the pressurized crack
        problem in question.

        Parameter
        ---------
        g: grid bucket
        bound_faces: boundary faces
        xf: coordinates of boundary faces
        h: bem segment length
        poi: Poisson ratio
        alpha: fracture orientation
        du: Sneddon's analytical relative normal displacement
        """
        f2 = np.zeros(bound_faces.size)
        f3 = np.zeros(bound_faces.size)
        f4 = np.zeros(bound_faces.size)
        f5 = np.zeros(bound_faces.size)

        u = np.zeros((g.dim, g.num_faces))

        m = 1 / (4 * np.pi * (1 - poi))

        f2[:] = m * (np.log(np.sqrt((xf[0, :] - h) ** 2 + xf[1] ** 2))
                     - np.log(np.sqrt((xf[0, :] + h) ** 2 + xf[1] ** 2)))

        f3[:] = - m * (np.arctan2(xf[1, :], (xf[0, :] - h))
                       - np.arctan2(xf[1, :], (xf[0, :] + h)))

        f4[:] = m * (xf[1, :] / ((xf[0, :] - h) ** 2 + xf[1, :] ** 2)
                     - xf[1, :] / ((xf[0, :] + h) ** 2 + xf[1, :] ** 2))

        f5[:] = m * ((xf[0, :] - h) / ((xf[0, :] - h) ** 2 + xf[1, :] ** 2)
                     - (xf[0, :] + h) / ((xf[0, :] + h) ** 2 + xf[1, :] ** 2))

        u[0, bound_faces] = du * (-(1 - 2 * poi) * np.cos(alpha) * f2[:]
                                  - 2 * (1 - poi) * np.sin(alpha) * f3[:]
                                  - xf[1, :] * (np.cos(alpha) * f4[:] + np.sin(
                    alpha) * f5[:]))
        u[1, bound_faces] = du * (-(1 - 2 * poi) * np.sin(alpha) * f2[:]
                                  + 2 * (1 - poi) * np.cos(alpha) * f3[:]
                                  - xf[1, :] * (np.sin(alpha) * f4[:] - np.cos(
                    alpha) * f5[:]))

        return u

    def assign_bem(g, h, bound_faces, theta, bem_centers, u_a, poi):

        """
        Compute analytical displacement using the BEM method for the pressurized crack
        problem in question.

        Parameter
        ---------
        g: grid bucket
        h: bem segment length
        bound_faces: boundary faces
        theta: fracture orientation
        bem_centers: bem segments centers
        u_a: Sneddon's analytical relative normal displacement
        poi: Poisson ratio
        """

        bc_val = np.zeros((g.dim, g.num_faces))

        alpha = np.pi / 2 - theta

        bound_face_centers = g.face_centers[:, bound_faces]

        for i in range(0, u_a.size):
            new_bound_face_centers = transform(bem_centers[:, i],
                                               bound_face_centers, alpha)

            u_bound = get_bc_val(g, bound_faces, new_bound_face_centers,
                                 h, poi, alpha, u_a[i])

            bc_val += u_bound

        return bc_val

    # Define boundary regions
    bound_faces = sd_rock.get_all_boundary_faces()
    box_faces = sd_rock.get_boundary_faces()
    length, height = domain_size

    h = 2 * a / n
    center = np.array([length / 2, height / 2, 0])
    bem_centers = get_bem_centers(a, h, n, theta, center)
    eta = compute_eta(bem_centers, center)
    u_a = analytical_displacements(a, eta, p0, G, poi)
    u_bc = assign_bem(sd_rock, h / 2, box_faces, theta, bem_centers, u_a, poi)

    return u_bc.ravel("F")


class BEM:
    """Parent class for BEM solution"""

    def __init__(self, params: dict) -> None:
        """Constructor of the BEM class.

        Parameters:
            params: SneddonSetup parameters.

        """
        self.params = params

    def get_bem_length(self, num_bem_segments: int | None = None) -> float:
        """Compute the length of each BEM segment.

        We assume that the crack is uniformly partitioned.

        Parameters:
            num_bem_segments: Number of BEM segments. If not specified, we use the
                number of elements given in ``self.params["num_bem_segments"]``.

        Returns:
            Length of the BEM segment.

        """
        if num_bem_segments is None:
            n = self.params["num_bem_segments"]
        else:
            n = num_bem_segments

        return self.params["crack_length"] / n

    def get_bem_centers(self, num_bem_segments: int | None = None) -> np.ndarray:
        """Compute the centers of the BEM segments.

        Parameters:
            num_bem_segments: Number of BEM segments. If not specified, we use the
                number of elements given in ``self.params["num_bem_segments"]``.

        Returns:
            Global coordinates of the BEM centers. Shape is (3, num_segments).

        """
        if num_bem_segments is None:
            n = self.params["num_bem_segments"]
        else:
            n = num_bem_segments

        cc = self.params["crack_center"]  # [m]
        b = self.params["crack_length"]  # [m]
        dl = self.get_bem_length(n)  # [m]
        beta = self.params["crack_angle"]  # [radians]

        x0 = cc[0] - 0.5 * (b - dl) * np.cos(beta)
        y0 = cc[1] - 0.5 * (b - dl) * np.sin(beta)

        xc = x0 + np.arange(n) * dl * np.cos(beta)
        yc = y0 + np.arange(n) * dl * np.sin(beta)
        zc = np.zeros(n)

        return np.array([xc, yc, zc])

    def transform_coordinates(
        self,
        bem_center: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        """Transform coordinates of a set of points relative to a BEM center.

        Parameters:
            bem_center: global coordinates of the bem segment center. Shape is (3, ).
            points: global coordinates of a set of points. Shape is (3, num_points).

        Returns:
            Transformed coordinates for the given set of ``points`` relative to the
                center of the BEM element. Shape is (3, num_points).

        """
        beta = self.params["crack_angle"]  # [radians]

        x_bar = np.zeros_like(points)
        x_bar[0] = (points[0] - bem_center[0]) * np.cos(beta) + (
            points[1] - bem_center[1]
        ) * np.sin(beta)
        x_bar[1] = -(points[0] - bem_center[0]) * np.sin(beta) + (
            points[1] - bem_center[1]
        ) * np.cos(beta)

        return x_bar

    def bem_contribution_to_displacement(
        self,
        normal_relative_displacement: float,
        points: np.ndarray,
        num_bem_segments: int | None = None,
    ) -> np.ndarray:
        """Displacements away from the crack approximated via BEM.

        Note that the following solutions are valid for the case when the BEM segment
        undergoes a *constant* relative displacement in the normal direction only.
        That is, the solution assumes that zero tangential relative displacement.

        Parameters:
            normal_relative_displacement : constant normal relative displacement
                (in meters) that the BEM segment undergoes.
            points : points in local coordinates relative to the crack center
                at which the displacement will be approximated.
            num_bem_segments : Number of BEM segments. If not specified, we use the
                number of elements given in ``self.params["num_bem_segments"]``.

        Returns:
            Approximate displacement at the given `points`.

        Notes:
            The expressions are given for `u_x` and `u_y` for an arbitrarily oriented
              BEM segment of length `2a` (see Eq. 5.5.4 from [1]).

            Note that for the pressurized crack problem, the displacement discontinuity
              in the tangential direction is zero.

            The expressions can therefore be written as:

            u_x = D_ybar * [
                - (1 - 2*nu) * cos(beta) * F2_bar
                - 2 * (1 - nu) * sin(beta) * F3_bar
                - y_bar * (cos(beta) * F4_bar + sin(beta) * F5_bar)
            ]

            u_y = D_ybar * [
                - (1 - 2 * nu) * sin(beta) * F2_bar
                + 2 * (1 - nu) * cos(beta) * F3_bar
                - y_bar * (sin(beta) * F4_bar - cos(beta) * F5_bar)
            ]

            Here, D_ybar is the exact relative normal displacement given by Sneddon's
              analytical solution, y_bar is the local vertical coordinate, beta is the
              angle (in radians) measured with respect to the horizontal axis,
              and F2_bar, F3_bar, F4_bar, and F5_bar are derivatives of the
              `f(x,y)` function (see Section 5.5 of [1]).

        """
        if num_bem_segments is None:
            n = self.params["num_bem_segments"]
        else:
            n = num_bem_segments

        D_ybar = normal_relative_displacement  # [m]
        beta = self.params["crack_angle"]  # [radians]
        nu = self.params["poisson_coefficient"]  # [-]
        dl = self.get_bem_length(num_bem_segments=n)  # [m]
        a = dl / 2  # half-length of the bem segment
        xbar = points[0]
        ybar = points[1]

        # Determine constant term that multiplies the expressions
        c0: float = 1 / (4 * np.pi * (1 - nu))

        # Derivatives of f(xbar, ybar)

        # F2(xbar, ybar) = f_{xbar}
        F2_bar = c0 * (
            np.log(np.sqrt((xbar - a) ** 2 + ybar**2))
            - np.log(np.sqrt((xbar + a) ** 2 + ybar**2))
        )

        # F3(x_bar, y_bar) = f_{y_bar}
        F3_bar = -c0 * (np.arctan(ybar / (xbar - a)) - np.arctan(ybar / (xbar + a)))

        # F4(x_bar, y_bar) = f_{x_bar, y_bar}
        F4_bar = c0 * (
            ybar / ((xbar - a) ** 2 + ybar**2) - ybar / ((xbar + a) ** 2 + ybar**2)
        )

        # F5(x_bar, y_bar) = f_{x_bar, x_bar} = - f_{y_bar, y_bar}
        F5_bar = c0 * (
            (xbar - a) / ((xbar - a) ** 2 + ybar**2)
            - (xbar + a) / ((xbar + a) ** 2 + ybar**2)
        )

        # Displacement in the global x-coordinate
        u_x = D_ybar * (
            -(1 - 2 * nu) * np.cos(beta) * F2_bar
            - 2 * (1 - nu) * np.sin(beta) * F3_bar
            - ybar * (np.cos(beta) * F4_bar + np.sin(beta) * F5_bar)
        )

        u_y = D_ybar * (
            -(1 - 2 * nu) * np.sin(beta) * F2_bar
            + 2 * (1 - nu) * np.cos(beta) * F3_bar
            - ybar * (np.sin(beta) * F4_bar - np.cos(beta) * F5_bar)
        )

        u = np.array([u_x, u_y]).ravel("F")

        return u

    def distance_from_crack_center(self, point_set: np.ndarray) -> np.ndarray:
        """Compute distance from a set of points to the fracture center.

        Args:
            point_set: coordinates of the set of points. Shape is (3, num_points).

        Returns:
            Distance from the set of point to the fracture center. Shape is
                (num_points, ).

        """
        crack_center = self.params["crack_center"]

        return pp.distances.point_pointset(crack_center, point_set)

    def far_field_displacement(
        self,
        points: np.ndarray,
        num_bem_segments: int | None = None,
    ):

        # Get number of bem segments used to discretize the crack
        if num_bem_segments is None:
            n = self.params["num_bem_segments"]
        else:
            n = num_bem_segments

        # Get bem centers in global coordinates
        bem_centers = self.get_bem_centers(n)

        # Compute distance from bem centers to the crack center
        eta = self.distance_from_crack_center(bem_centers)

        # Get exact relative normal displacement for each bem segment
        u_n = self.exact_relative_normal_displacement(eta)

        # BEM loop
        num_points = points.shape[1]
        u = np.zeros(2 * num_points)
        for bem in range(n):

            # Transform coordinates relative to each bem center
            xbar = self.transform_coordinates(bem_centers[:, bem], points)

            # Get bem contribution to the displacement at the given set of points
            u_bem = self.bem_contribution_to_displacement(
                normal_relative_displacement=u_n[bem], points=xbar, num_bem_segments=n
            )

            # Add contribution
            u += u_bem

        return u

    def exact_relative_normal_displacement(self, eta: np.ndarray) -> np.ndarray:
        """Compute exact relative normal displacement.

        Sneddon, I. (1952). Fourier transforms. Bull. Amer. Math. Soc, 58, 512-513.

        Parameters:
            eta: array containing the distances from a set of points to the crack
               center. Shape is (num_points, ). Typically, the set of points will
               correspond to the global coordinates of the bem centers.

        Returns:
            Exact relative normal displacement for each ``eta``.

        """
        nu_s = self.params["poisson_coefficient"]  # [-]
        mu_s = self.params["mu_lame"]  # [Pa]
        p0 = self.params["crack_pressure"]  # [Pa]
        frac_length = self.params["crack_length"]  # [m]
        a = frac_length / 2

        c0 = (1 - nu_s) / mu_s * p0 * frac_length
        u_n = c0 * np.sqrt(1 - (eta / a) ** 2)

        return u_n


    def bem_relative_normal_displacement(
        self, num_bem_segments: int | None = None
    ) -> np.ndarray:
        """Numerical approximation of the relative normal displacements using BEM.

        This procedure follows Section 5.3 from Crouch, S.L., Starfield, A.: Boundary
        Element Methods in Solid Mechanics: With Applications in Rock Mechanics and
        Geological Engineering. Allen & Unwin, London (1982).

        Parameters:
            num_bem_segments: Number of BEM segments used to generate the approximate
                solution. If not specified, "self.bem_num" is employed.

        Returns:
            Approximate relative normal displacement for the pressurized crack
            problem using the Boundary Element Method. Shape is (num_bem_segments, ).

        """
        if num_bem_segments is None:
            n = self.params["num_bem_segments"]
        else:
            n = num_bem_segments

        p = self.params["crack_pressure"]  # [Pa]
        mu_s = self.params["mu_lame"]  # [Pa]
        nu_s = self.params["poisson_coefficient"]  # [-]
        dl = self.get_bem_length(n)  # [m]
        c = self.get_bem_centers(n)

        # Transform coordinates
        # c_bar = self.coordinate_transform(c)

        # Compute matrix of influence coefficients
        a0 = -mu_s / (np.pi * (1 - nu_s))
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = a0 * ((dl / 2) / ((c[0][i] - c[0][j]) ** 2 - (dl / 2) ** 2))

        # Vector of constants
        b = p * np.ones(n)

        # Solve linear system
        x = sps.linalg.solve(A, b)

        return x


class SneddonSetup(pp.ContactMechanics):
    """Parent class for Sneddon's problem."""

    def __init__(self, params):
        """Constructor for Sneddon's class."""

        def set_default_params(keyword: str, value: object) -> None:
            """
            Set default parameters if a keyword is absent in the `params` dictionary.

            Parameters:
                keyword: Parameter keyword, e.g., "mesh_size".
                value: Value of `keyword`, e.g., 1.0.

            """
            if keyword not in params.keys():
                params[keyword] = value

        # Define default parameters
        default_params: list[tuple] = [
            ("domain_size", (50.0, 50.0)),  # [m]
            ("crack_angle", 0 * np.pi),  # [radians]
            ("crack_length", 20.0),  # [m]
            ("crack_pressure", 1e-4),  # [GPa]
            ("poisson_coefficient", 0.25),  # [-]
            ("mesh_size", 2.0),  # [m]
            ("mu_lame", 1.0),  # [GPa]
            ("num_bem_segments", 1000),
            ("plot_results", False),
            ("use_ad", True),  # only `use_ad = True` is supported
        ]

        # Set default values
        for key, val in default_params:
            set_default_params(key, val)
        super().__init__(params)

        # ad sanity check
        if not self.params["use_ad"]:
            raise ValueError("Model only valid when ad is used.")

        # Store other useful parameters
        lx, ly = self.params["domain_size"]
        self.params["crack_center"] = np.array([lx / 2, ly / 2, 0.0])

        # TODO: Store lambda lame parameter

        # Create a BEM dictionary to store BEM-related quantities
        self.bem = BEM(self.params)

    def create_grid(self) -> None:
        """Create mixed-dimensional grid.

        The fracture will be placed at the center of the domain at an angle `theta`
        measured with respect to the horizontal axis.

        """
        # Retrieve data
        lx, ly = self.params["domain_size"]  # [m]
        h = self.params["mesh_size"]  # [m]
        beta = self.params["crack_angle"]  # [radians]
        crack_length = self.params["crack_length"]  # [m]
        a = crack_length / 2

        # Create bounding box
        self.box = {"xmin": 0.0, "xmax": lx, "ymin": 0.0, "ymax": ly}

        # Create fracture network
        x_0 = (lx / 2) - a * np.cos(beta)  # initial tip coo in x
        x_1 = (lx / 2) + a * np.cos(beta)  # final tip coo in x
        y_0 = (ly / 2) - a * np.sin(beta)  # initial tip coo in y
        y_1 = (ly / 2) + a * np.sin(beta)  # final tip coo in y
        frac_pts = np.array([[x_0, y_0], [x_1, y_1]]).T
        frac_edges = np.array([[0, 1]]).T
        network_2d = pp.FractureNetwork2d(frac_pts, frac_edges, self.box)

        # Create mixed-dimensional grid
        mesh_args = {"mesh_size_bound": h, "mesh_size_frac": h}
        self.mdg = network_2d.mesh(mesh_args)

        # Set projections
        pp.contact_conditions.set_projections(self.mdg)

    def after_simulation(self) -> None:
        """Method to be called once the simulation has finished."""
        if self.params["plot_results"]:
            self.plot_results()

    # -----> Methods relate to BEM

    def get_boundary_conditions(self) -> np.ndarray:

        sd_rock = self.mdg.subdomains()[0]
        sides = self._domain_boundary_sides(sd_rock)
        bc_faces = sides.all_bf
        bc_coo = sd_rock.face_centers[:, bc_faces]

        u_bc = self.bem.far_field_displacement(bc_coo)
        bc_vals = np.zeros(sd_rock.dim * sd_rock.num_faces)
        bc_vals[::sd_rock.dim][bc_faces] = u_bc[::sd_rock.dim]
        bc_vals[1::sd_rock.dim][bc_faces] = u_bc[1::sd_rock.dim]

        return bc_vals

    # -----> Methods related to the analytical solution
    def exact_normal_displacement_jump(self, eta: np.ndarray) -> np.ndarray:
        """Compute exact relative normal displacement jump.

        Sneddon, I. (1952). Fourier transforms. Bull. Amer. Math. Soc, 58, 512-513.

        Parameters:
            eta: array containing the distances from a set of points `p` to the
                crack center. Shape is (num_points, ).

        Returns:
            Exact normal relative displacement jump for each ``eta``.

        """
        nu_s = self.params["poisson_coefficient"]  # [-]
        mu_s = self.params["mu_lame"]  # [Pa]
        p0 = self.params["crack_pressure"]  # [Pa]
        frac_length = self.params["crack_length"]  # [m]
        half_length = frac_length / 2

        c0 = (1 - nu_s) / mu_s * p0 * frac_length
        u_jump = c0 * (1 - (eta / half_length) ** 2) ** 0.5

        return u_jump

    # ------> Helper methods
    def distance_from_crack_center(self, point_set: np.ndarray) -> np.ndarray:
        """Compute distance from a set of points to the fracture center.

        Args:
            point_set: coordinates of the set of points. Shape is (3, num_points).

        Returns:
            Distance from the set of point to the fracture center. Shape is
                (num_points, ).

        """
        length, height = self.params["domain_size"]  # [m]

        frac_center = np.array([[length / 2], [height / 2], [0.0]])

        return pp.distances.point_pointset(frac_center, point_set)

    # -----> Plotting methods
    def plot_results(self) -> None:
        """Plot results."""

        # Relative displacement
        self._plot_relative_displacement()

    def _plot_relative_displacement(self):
        """Plot relative displacement as a function of distance from crack center"""

        # Generate exact points
        # TODO: Do we need to rotate the coordinates?
        sd_frac = self.mdg.subdomains()[1]
        crack_length = self.params["crack_length"]
        x_min = np.min(sd_frac.nodes[0])
        x_max = np.max(sd_frac.nodes[0])
        y_min = np.min(sd_frac.nodes[1])
        y_max = np.max(sd_frac.nodes[1])
        x_ex = np.linspace(x_min, x_max, 100)
        y_ex = np.linspace(y_min, y_max, 100)
        z_ex = np.zeros(100)
        points_ex = np.array([x_ex, y_ex, z_ex])
        eta_ex = self.distance_from_crack_center(points_ex)
        u_jump_ex = self.exact_normal_displacement_jump(eta_ex)

        fig, ax = plt.subplots(figsize=(9, 8))

        # Plot exact solution
        ax.plot(
            x_ex / crack_length,
            u_jump_ex / crack_length,
            linewidth=3,
            color="blue",
            alpha=0.4,
        )

        # Plot BEM solution
        bem_elements = 20
        bem_centers = self.bem.get_bem_centers(bem_elements)
        bem_sol = self.bem.bem_relative_normal_displacement(bem_elements)
        ax.plot(
            bem_centers[0] / crack_length,
            bem_sol / crack_length,
            marker="o",
            markersize=6,
            linewidth=0,
            color="orange",
        )
        plt.step(
            bem_centers[0] / crack_length,
            bem_sol / crack_length,
            where="mid",
            color="orange",
        )
        plt.step(
            np.array([2.0, bem_centers[0][0] / crack_length]),
            np.array([0, bem_sol[0] / crack_length]),
            where="pre",
            color="orange",
        )
        plt.step(
            np.array([bem_centers[0][-1] / crack_length, 3.0]),
            np.array([bem_sol[-1] / crack_length, 0]),
            where="post",
            color="orange",
            alpha=0.7,
        )

        # Label plot
        plt.plot(
            [],
            [],
            linewidth=4,
            color="blue",
            alpha=0.4,
            label="Exact, Sneddon (1951).",
        )
        plt.plot(
            [],
            [],
            linewidth=4,
            color="orange",
            marker="o",
            markersize=8,
            label=f"BEM ({bem_sol.size} elements), Crouch & Starfield (1983).",
        )

        # Set labels and legends
        ax.set_xlabel(r"Non-dimensional horizontal distance, " r"$x~/~b$", fontsize=13)
        ax.set_ylabel(
            r"Relative normal displacement, " r"$\hat{u}_n(x)~/~b$", fontsize=13
        )
        ax.legend(fontsize=13)

        if not os.path.exists("out"):
            os.makedirs("out")
        plt.savefig("out" + "sneddon" + ".pdf", bbox_inches="tight")
        plt.gcf().clear()


#%% Runner
params = {
    "frac_angle": 0,
    "plot_results": False,
    "num_bem_segments": 1000,
    "domain_size": (50, 50),
    "mesh_size": 5.0,
}
setup = SneddonSetup(params=params)
pp.run_stationary_model(setup, params)

mdg = setup.mdg
sd_rock = mdg.subdomains()[0]
sd_frac = mdg.subdomains()[1]

u_bem = setup.bem.far_field_displacement(sd_rock.cell_centers)

#pp.plot_grid(sd_rock, u_bem[::2], plot_2d=True, title="u_bem_x", linewidth=0)
#pp.plot_grid(sd_rock, u_bem[1::2], plot_2d=True, title="u_bem_y", linewidth=0)

#%%
bc_vals = setup.get_boundary_conditions()
sides = setup._domain_boundary_sides(sd_rock)
bc_faces = sides.all_bf
bc_vals_x = bc_vals[::2]
bc_vals_y = bc_vals[1::2]

bc_jv_x = bc_vals_x[bc_faces]
bc_jv_y = bc_vals_y[bc_faces]

# %% Michele's solution
import math
bc_michele = get_bc_values_michele(
    sd_rock = sd_rock,
    G = setup.params["mu_lame"],
    poi = setup.params["poisson_coefficient"],
    p0 = setup.params["crack_pressure"],
    a = setup.params["crack_length"] / 2,
    n = setup.params["num_bem_segments"],
    domain_size = setup.params["domain_size"],
    theta = math.radians(90-setup.params["crack_angle"])
)

bc_ms_x = bc_michele[0::2][bc_faces]
bc_ms_y = bc_michele[1::2][bc_faces]