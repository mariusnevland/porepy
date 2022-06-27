import numpy as np
import porepy as pp
import quadpy as qp
import scipy.sparse as sps
import sympy as sym


class ExactSolution:
    """Class containing the exact solutions to the manufactured Biot problem"""

    def __init__(self):

        # Symbolic variables
        x, y, t = sym.symbols("x y t")
        self.x = x
        self.y = y
        self.t = t

        # Physical parameters
        self.lambda_s = 1.0
        self.mu_s = 1.0
        self.alpha = 1.0
        self.stor = 1.0
        self.mu_f = 1.0
        self.k = [[1.0, 0], [0, 1.0]]

    def pressure(self, output_type="sym"):
        """Exact pressure"""

        # Sanity check
        if output_type not in ["sym", "fun"]:
            raise ValueError("Expected 'sym' or 'fun'.")

        p_sym = self.t ** 2 * sym.sin(2 * sym.pi * self.x) * sym.cos(2 * sym.pi * self.y)
        p_fun = sym.lambdify((self.x, self.y, self.t), p_sym, "numpy")

        if output_type == "sym":
            return p_sym
        else:
            return p_fun

    def displacement(self, output_type="sym"):
        """Exact displacement"""

        # Sanity check
        if output_type not in ["sym", "fun"]:
            raise ValueError("Expected 'sym' or 'fun'.")

        ux_sym = self.t ** 2 * self.x * (1 - self.x) * sym.sin(2 * sym.pi * self.y)
        uy_sym = self.t ** 2 * sym.sin(2 * sym.pi * self.x) * sym.sin(2 * sym.pi * self.y)
        u_sym = [ux_sym, uy_sym]

        u_fun = [sym.lambdify((self.x, self.y, self.t), u, "numpy") for u in u_sym]

        if output_type == "sym":
            return u_sym
        else:
            return u_fun

    def pressure_gradient(self, output_type="sym"):
        """Exact pressure gradient"""

        # Sanity check
        if output_type not in ["sym", "fun"]:
            raise ValueError("Expected 'sym' or 'fun'.")

        gradpx_sym = sym.diff(self.pressure(), self.x)
        gradpy_sym = sym.diff(self.pressure(), self.y)
        gradp_sym = [gradpx_sym, gradpy_sym]

        gradp_fun = [
            sym.lambdify((self.x, self.y, self.t), gradp, "numpy") for gradp in gradp_sym
        ]

        if output_type == "sym":
            return gradp_sym
        else:
            return gradp_fun

    def darcy_flux(self, output_type="sym"):
        """Exact Darcy flux"""

        # Sanity check
        if output_type not in ["sym", "fun"]:
            raise ValueError("Expected 'sym' or 'fun'.")

        qx_sym = (
                - self.k[0][0] * self.mu_f ** (-1) * self.pressure_gradient()[0]
                - self.k[0][1] * self.mu_f ** (-1) * self.pressure_gradient()[1]
        )
        qy_sym = (
                - self.k[1][0] * self.mu_f ** (-1) * self.pressure_gradient()[0]
                - self.k[1][1] * self.mu_f ** (-1) * self.pressure_gradient()[1]
        )
        q_sym = [qx_sym, qy_sym]

        q_fun = [sym.lambdify((self.x, self.y, self.t), q, "numpy") for q in q_sym]

        if output_type == "sym":
            return q_sym
        else:
            return q_fun

    def divergence_darcy_flux(self, output_type="sym"):
        """Exact divergence of the Darcy flux"""

        # Sanity check
        if output_type not in ["sym", "fun"]:
            raise ValueError("Expected 'sym' or 'fun'.")

        divq_sym = (
                sym.diff(self.darcy_flux()[0], self.x)
                + sym.diff(self.darcy_flux()[1], self.y)
        )

        divq_fun = sym.lambdify((self.x, self.y, self.t), divq_sym, "numpy")

        if output_type == "sym":
            return divq_sym
        else:
            return divq_fun

    def divergence_displacement(self, output_type="sym"):
        """Exact divergence of displacement"""

        # Sanity check
        if output_type not in ["sym", "fun"]:
            raise ValueError("Expected 'sym' or 'fun'.")

        divu_sym = (
                sym.diff(self.displacement()[0], self.x)
                + sym.diff(self.displacement()[1], self.y)
        )

        divu_fun = sym.lambdify((self.x, self.y, self.t), divu_sym, "numpy")

        if output_type == "sym":
            return divu_sym
        else:
            return divu_fun

    def time_derivative_pressure(self, output_type="sym"):
        """Exact partial derivative of the pressure w.r.t time"""

        # Sanity check
        if output_type not in ["sym", "fun"]:
            raise ValueError("Expected 'sym' or 'fun'.")

        dpdt_sym = sym.diff(self.pressure(), self.t)
        dpdt_fun = sym.lambdify((self.x, self.y, self.t), dpdt_sym, "numpy")

        if output_type == "sym":
            return dpdt_sym
        else:
            return dpdt_fun

    def time_derivative_divergence_displacement(self, output_type="sym"):
        """Exact partial derivative of the divergence of the displacement w.r.t. time"""

        # Sanity check
        if output_type not in ["sym", "fun"]:
            raise ValueError("Expected 'sym' or 'fun'.")

        ddivudt_sym = sym.diff(self.divergence_displacement(), self.t)
        ddivudt_fun = sym.lambdify((self.x, self.y, self.t), ddivudt_sym, "numpy")

        if output_type == "sym":
            return ddivudt_sym
        else:
            return ddivudt_fun

    def scalar_source(self, output_type="sym"):
        """Exact scalar source"""

        # Sanity check
        if output_type not in ["sym", "fun"]:
            raise ValueError("Expected 'sym' or 'fun'.")

        # Compute/collect different terms
        t1 = self.stor * self.time_derivative_pressure()
        t2 = self.alpha * self.time_derivative_divergence_displacement()
        t3 = self.divergence_darcy_flux()
        ff_sym = t1 + t2 + t3

        ff_fun = sym.lambdify((self.x, self.y, self.t), ff_sym, "numpy")

        if output_type == "sym":
            return ff_sym
        else:
            return ff_fun

    def gradient_displacement(self, output_type="sym"):
        """Exact gradient of the displacement"""

        # Sanity check
        if output_type not in ["sym", "fun"]:
            raise ValueError("Expected 'sym' or 'fun'.")

        graduxx_sym = sym.diff(self.displacement()[0], self.x)
        graduxy_sym = sym.diff(self.displacement()[0], self.y)
        graduyx_sym = sym.diff(self.displacement()[1], self.x)
        graduyy_sym = sym.diff(self.displacement()[1], self.y)

        gradu_sym = [[graduxx_sym, graduxy_sym], [graduyx_sym, graduyy_sym]]

        gradu_fun = [
            [
                sym.lambdify((self.x, self.y, self.t), graduxx_sym, "numpy"),
                sym.lambdify((self.x, self.y, self.t), graduxy_sym, "numpy"),
            ],
            [
                sym.lambdify((self.x, self.y, self.t), graduyx_sym, "numpy"),
                sym.lambdify((self.x, self.y, self.t), graduyy_sym, "numpy"),
            ]
        ]

        if output_type == "sym":
            return gradu_sym
        else:
            return gradu_fun

    def transpose_gradient_displacement(self, output_type="sym"):
        """Exact transpose of the gradient of the displacement"""

        # Sanity check
        if output_type not in ["sym", "fun"]:
            raise ValueError("Expected 'sym' or 'fun'.")

        gradutxx_sym = self.gradient_displacement()[0][0]
        gradutxy_sym = self.gradient_displacement()[1][0]
        gradutyx_sym = self.gradient_displacement()[0][1]
        gradutyy_sym = self.gradient_displacement()[1][1]

        gradut_sym = [[gradutxx_sym, gradutxy_sym], [gradutyx_sym, gradutyy_sym]]

        gradut_fun = [
            [
                sym.lambdify((self.x, self.y, self.t), gradutxx_sym, "numpy"),
                sym.lambdify((self.x, self.y, self.t), gradutxy_sym, "numpy"),
            ],
            [
                sym.lambdify((self.x, self.y, self.t), gradutyx_sym, "numpy"),
                sym.lambdify((self.x, self.y, self.t), gradutyy_sym, "numpy"),
            ]
        ]

        if output_type == "sym":
            return gradut_sym
        else:
            return gradut_fun

    def biot_stress(self, output_type="sym"):
        """Exact Biot stress"""

        # Sanity check
        if output_type not in ["sym", "fun"]:
            raise ValueError("Expected 'sym' or 'fun'.")

        # Compute first term: mu_s * (grad(u) + transpose(grad(u)))
        t1_xx = self.mu_s * (
                self.gradient_displacement()[0][0]
                + self.transpose_gradient_displacement()[0][0]
        )
        t1_xy = self.mu_s * (
                self.gradient_displacement()[0][1]
                + self.transpose_gradient_displacement()[0][1]
        )
        t1_yx = self.mu_s * (
                self.gradient_displacement()[1][0]
                + self.transpose_gradient_displacement()[1][0]
        )
        t1_yy = self.mu_s * (
                self.gradient_displacement()[1][1]
                + self.transpose_gradient_displacement()[1][1]
        )

        # Compute second term: lambda_s * div(u) * eye(2)
        t2_xx = self.lambda_s * self.divergence_displacement()
        t2_xy = 0
        t2_yx = 0
        t2_yy = self.lambda_s * self.divergence_displacement()

        # Compute third term: alpha * p * eye(2)
        t3_xx = self.alpha * self.pressure()
        t3_xy = 0
        t3_yx = 0
        t3_yy = self.alpha * self.pressure()

        # Compute Biot stress: mu_s * (grad(u) + transpose(grad(u)))
        # + lambda_s * div(u) * eye(2) - alpha * p * eye(2)
        sigma_xx = t1_xx + t2_xx - t3_xx
        sigma_xy = t1_xy + t2_xy - t3_xy
        sigma_yx = t1_yx + t2_yx - t3_yx
        sigma_yy = t1_yy + t2_yy - t3_yy

        sigma_sym = [[sigma_xx, sigma_xy], [sigma_yx, sigma_yy]]

        sigma_fun = [
            [
                sym.lambdify((self.x, self.y, self.t), sigma_xx, "numpy"),
                sym.lambdify((self.x, self.y, self.t), sigma_xy, "numpy"),
            ],
            [
                sym.lambdify((self.x, self.y, self.t), sigma_yx, "numpy"),
                sym.lambdify((self.x, self.y, self.t), sigma_yy, "numpy"),
            ]
        ]

        if output_type == "sym":
            return sigma_sym
        else:
            return sigma_fun

    def divergence_biot_stress(self, output_type="sym"):
        """Exact divergence of the Biot stress"""

        # Sanity check
        if output_type not in ["sym", "fun"]:
            raise ValueError("Expected 'sym' or 'fun'.")

        div_sigmax = (
                sym.diff(self.biot_stress()[0][0], self.x)
                + sym.diff(self.biot_stress()[1][0], self.y)
        )
        div_sigmay = (
                sym.diff(self.biot_stress()[0][1], self.x)
                + sym.diff(self.biot_stress()[1][1], self.y)
        )
        divsigma_sym = [div_sigmax, div_sigmay]

        divsigma_fun = [
            sym.lambdify((self.x, self.y, self.t), div_sigmax, "numpy"),
            sym.lambdify((self.x, self.y, self.t), div_sigmay, "numpy"),
        ]

        if output_type == "sym":
            return divsigma_sym
        else:
            return divsigma_fun

    def vector_source(self, output_type="sym"):
        """Exact source term (body force) for the mechanical subproblem"""

        # Sanity check
        if output_type not in ["sym", "fun"]:
            raise ValueError("Expected 'sym' or 'fun'.")

        fs_sym = self.divergence_biot_stress()
        fs_fun = self.divergence_biot_stress("fun")

        if output_type == "sym":
            return fs_sym
        else:
            return fs_fun

    def integrated_scalar_source(self, g: pp.Grid, time: float) -> np.ndarray:
        """Computes the integrated scalar sources over the grid

        Parameters
        ----------
        g : pp.Grid
            PorePy grid.
        time : float
            Current time.

        Returns
        -------
        integrated_source : np.ndarray(g.num_cells)
            Numerically integrated sources.
        """

        # Sanity check
        if g.dim != 2:
            raise ValueError("Expected two-dimensional grid.")

        # Retrieve QuadPy elements and declare numerical integration parameters
        elements = self.get_quadpy_elements(g)
        method = qp.t2.get_good_scheme(4)

        # Retrieve scalar source term
        f = self.scalar_source("fun")

        # Declare integrand
        def integrand(x):
            return f(x[0], x[1], time)

        # Compute integration
        integrated_source = method.integrate(integrand, elements)

        return integrated_source

    def integrated_vector_source(self, g: pp.Grid, time: float) -> np.ndarray:
        """Computes the integrated vector sources over the grid

        Parameters
        ----------
        g : pp.Grid
            PorePy grid.
        time : float
            Current time.

        Returns
        -------
        integrated_source : np.ndarray(g.num_cells * g.dim)
            Numerically integrated sources.
        """

        # Sanity check
        if g.dim != 2:
            raise ValueError("Expected two-dimensional grid.")

        # Retrieve QuadPy elements and declare numerical integration parameters
        elements = self.get_quadpy_elements(g)
        method = qp.t2.get_good_scheme(4)

        # Retrieve components of vector source term
        fx = self.vector_source("fun")[0]
        fy = self.vector_source("fun")[1]

        # Declare integrands
        def integrandx(x):
            return fx(x[0], x[1], time)

        def integrandy(x):
            return fy(x[0], x[1], time)

        # Compute integration
        integrated_fx = method.integrate(integrandx, elements)
        integrated_fy = method.integrate(integrandy, elements)
        integrated_source = np.zeros(g.dim * g.num_cells)
        integrated_source[::2] = integrated_fx
        integrated_source[1::2] = integrated_fy

        return integrated_source

    # Utility methods
    @staticmethod
    def get_quadpy_elements(g: pp.Grid) -> np.ndarray:
        """
        Assembles the elements of a given PorePy grid in QuadPy format.

        See e.g.: https://pypi.org/project/quadpy/.

        Parameters
        ----------
        g : pp.Grid
            PorePy grid.

        Returns
        --------
        elements : np.ndarray(dtype=int64)
            Elements in QuadPy format.
        """

        # Getting node coordinates for each cell
        nc = g.num_cells
        cell_nodes_map, _, _ = sps.find(g.cell_nodes())
        nodes_cell = cell_nodes_map.reshape(np.array([nc, g.dim + 1]))
        nodes_coor_cell = g.nodes[:, nodes_cell]

        # Stacking node coordinates
        cnc_stckd = np.empty([nc, (g.dim + 1) * g.dim])
        col = 0
        for vertex in range(g.dim + 1):
            for dim in range(g.dim):
                cnc_stckd[:, col] = nodes_coor_cell[dim][:, vertex]
                col += 1
        element_coord = np.reshape(cnc_stckd, np.array([nc, g.dim + 1, g.dim]))

        # Reshaping to please quadpy format i.e, (corners, num_elements, coords)
        elements = np.stack(element_coord, axis=-2)

        return elements
