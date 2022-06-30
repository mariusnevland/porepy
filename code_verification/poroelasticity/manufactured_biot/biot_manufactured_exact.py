import numpy as np
import porepy as pp
import quadpy as qp
import scipy.sparse as sps
import sympy as sym

from typing import Literal, List


class ExactSolution:
    """Class containing the exact solutions to the manufactured Biot problem"""

    def __init__(self):

        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Physical parameters
        self.lambda_s = 1.0
        self.mu_s = 1.0
        self.alpha = 1.0
        self.stor = 1.0
        self.mu_f = 1.0
        self.k = [[1.0, 0], [0, 1.0]]

        # Exact solutions

        # Pressure (user-defined)
        self.pressure = t * x * (1 - x) * sym.sin(2 * sym.pi * y)

        # Displacement (user-defined)
        ux = t * x * (1 - x) * sym.sin(2 * sym.pi * y)
        uy = t * sym.sin(2 * sym.pi * x) * sym.sin(2 * sym.pi * y)
        self.displacement = [ux, uy]

        # The rest of the attributes should not be modified

        # Pressure gradient
        gradpx = sym.diff(self.pressure, x)
        gradpy = sym.diff(self.pressure, y)
        self._gradp = [gradpx, gradpy]

        # Darcy flux
        qx = (
                - self.k[0][0] * self.mu_f ** (-1) * self._gradp[0]
                - self.k[0][1] * self.mu_f ** (-1) * self._gradp[1]
        )
        qy = (
                - self.k[1][0] * self.mu_f ** (-1) * self._gradp[0]
                - self.k[1][1] * self.mu_f ** (-1) * self._gradp[1]
        )
        self._q = [qx, qy]

        # Divergence of Darcy flux
        self._divq = sym.diff(self._q[0], x) + sym.diff(self._q[1], y)

        # Divergece of displacement
        self._divu = sym.diff(self.displacement[0], x) + sym.diff(self.displacement[1], y)

        # Time derivative of pressure
        self._dp_dt = sym.diff(self.pressure, t)

        # Time derivative of divergence of the displacement
        self._ddivu_dt = sym.diff(self._divu, t)

        # Flow source
        ff_t1 = self.stor * self._dp_dt
        ff_t2 = self.alpha * self._ddivu_dt
        ff_t3 = self._divq
        self.source_flow = ff_t1 + ff_t2 + ff_t3

        # Gradient of the displacement
        uxx = sym.diff(self.displacement[0], x)
        uxy = sym.diff(self.displacement[0], y)
        uyx = sym.diff(self.displacement[1], x)
        uyy = sym.diff(self.displacement[1], y)
        self._gradu = [[uxx, uxy], [uyx, uyy]]

        # Transpose of the gradient of the displacement
        self._gradut = [[uxx, uyx], [uxy, uyy]]

        # Strain
        eps_xx = 0.5 * (self._gradu[0][0] + self._gradut[0][0])
        eps_xy = 0.5 * (self._gradu[0][1] + self._gradut[0][1])
        eps_yx = 0.5 * (self._gradu[1][0] + self._gradut[1][0])
        eps_yy = 0.5 * (self._gradu[1][1] + self._gradut[1][1])
        self._eps = [[eps_xx, eps_xy], [eps_yx, eps_yy]]

        # Effective stress
        trace_eps = self._eps[0][0] + self._eps[1][1]
        sigmaeff_xx = self.lambda_s * trace_eps + 2 * self.mu_s * self._eps[0][0]
        sigmaeff_xy = 2 * self.mu_s * self._eps[0][1]
        sigmaeff_yx = 2 * self.mu_s * self._eps[1][0]
        sigmaeff_yy = self.lambda_s * trace_eps + 2 * self.mu_s * self._eps[1][1]
        self._sigmaeff = [[sigmaeff_xx, sigmaeff_xy], [sigmaeff_yx, sigmaeff_yy]]

        # Total stress
        sigma_xx = self._sigmaeff[0][0] - self.alpha * self.pressure
        sigma_xy = self._sigmaeff[0][1]
        sigma_yx = self._sigmaeff[1][0]
        sigma_yy = self._sigmaeff[1][1] - self.alpha * self.pressure
        self._sigma = [[sigma_xx, sigma_xy], [sigma_yx, sigma_yy]]

        # Divergence of total stress tensor = mechanical source term (body forces)
        fsx = sym.diff(self._sigma[0][0], x) + sym.diff(self._sigma[1][0], y)
        fsy = sym.diff(self._sigma[0][1], x) + sym.diff(self._sigma[1][1], y)
        self.source_mechanics = [fsx, fsy]

    def integrated_source_flow(self, g: pp.Grid, time: float) -> np.ndarray:
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
        x, y, t = sym.symbols("x y t")
        f = sym.lambdify((x, y, t), self.source_flow, "numpy")

        # Declare integrand
        def integrand(coo):
            return f(coo[0], coo[1], time)

        # Compute integration
        integrated_source = method.integrate(integrand, elements)

        return integrated_source

    def integrated_source_mechanics(self, g: pp.Grid, time: float) -> np.ndarray:
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

        # Retrieve scalar source term
        x, y, t = sym.symbols("x y t")
        fx = sym.lambdify((x, y, t), self.source_mechanics[0], "numpy")
        fy = sym.lambdify((x, y, t), self.source_mechanics[1], "numpy")

        # Declare integrands
        def integrandx(coo):
            return fx(coo[0], coo[1], time)

        def integrandy(coo):
            return fy(coo[0], coo[1], time)

        # Compute integration
        integrated_fx = method.integrate(integrandx, elements)
        integrated_fy = method.integrate(integrandy, elements)
        integrated_source = np.empty(g.dim * g.num_cells)
        integrated_source[::2] = integrated_fx
        integrated_source[1::2] = integrated_fy

        return integrated_source

    # Utility methods
    @staticmethod
    def eval_scalar(g: pp.Grid, sym_scalar, time: float) -> np.ndarray:
        """Evaluate scalar expresion at the cell centers of a grid for a given time"""

        cc = g.cell_centers
        x, y, t = sym.symbols("x y t")
        fun = sym.lambdify((x, y, t), sym_scalar, "numpy")
        eval_fun = fun(cc[0], cc[1], time)

        return eval_fun

    @staticmethod
    def eval_vector(
            g: pp.Grid,
            sym_vector,
            time: float
    ) -> List[np.ndarray]:
        """Evaluate vector expresion at the cell centers of a grid for a given time"""

        cc = g.cell_centers
        x, y, t = sym.symbols("x y t")
        fun_x = sym.lambdify((x, y, t), sym_vector[0], "numpy")
        fun_y = sym.lambdify((x, y, t), sym_vector[1], "numpy")
        eval_fun = [fun_x(cc[0], cc[1], time), fun_y(cc[0], cc[1], time)]

        return eval_fun

    @staticmethod
    def eval_tensor(
            g: pp.Grid,
            sym_tensor,
            time: float
    ) -> List[List[np.ndarray]]:
        """Evaluate tensor expresion at the cell centers of a grid for a given time"""

        # Sanity check
        if g.dim != 2:
            raise ValueError("Expected two-dimensional grid.")

        cc = g.cell_centers
        x, y, t = sym.symbols("x y t")
        fun_xx = sym.lambdify((x, y, t), sym_tensor[0][0], "numpy")
        fun_xy = sym.lambdify((x, y, t), sym_tensor[0][1], "numpy")
        fun_yx = sym.lambdify((x, y, t), sym_tensor[1][0], "numpy")
        fun_yy = sym.lambdify((x, y, t), sym_tensor[1][1], "numpy")
        eval_fun = [
            [fun_xx(cc[0], cc[1], time), fun_xy(cc[0], cc[1], time)],
            [fun_yx(cc[0], cc[1], time), fun_yy(cc[0], cc[1], time)],
        ]
        return eval_fun

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
