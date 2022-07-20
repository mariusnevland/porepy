import numpy as np
import porepy as pp
import quadpy as qp
import scipy.sparse as sps
import sympy as sym

from typing import List, Literal, Union

Scalar = Union[int, float]
SymMul = sym.core.mul.Mul
SymAdd = sym.core.add.Add


class ExactBiotManufactured:
    """Class containing the exact solutions to the manufactured Biot's problem.

    The exact solutions are taken from https://epubs.siam.org/doi/pdf/10.1137/15M1014280,
    and for a single time step in an unit square domain are given by:
        u(x1, x2) = [x1 * (1 - x) * sin(2 * pi * x2), sin(2 * pi * x1) * sin(2 * pi * x2)],
        p(x1, x2) = x1 * (1 - x) * sin(2 * pi * x2),
    where u and p are respectively the displacement and fluid pressure.

    Analytical sources can therefore be obtained by computing the relevant expressions. This
    class computes such sources automatically for a given exact pair (u, p) and a set of
    physical parameters that have to be given as attributes.

    Attributes:
        lambda_s: First Lamé parameter.
        mu_s: Second Lamé parameter.
        alpha: Biot's coupling coefficient.
        stor: Storativity.
        mu_f: Dynamic fluid viscosity.
        k: Intrinsic permeability.
        pressure: Exact symbolic pressure object.
        displacement: Exact symbolic displacement object.
        source_flow: Exact symbolic flow source object.
        source_mechanics: Exact symbolic mechanics object.

    """

    def __init__(self):
        """Constructor for the class.

        Note:
            Derived quantities are stored as private attributes.

        """

        # Declare symbolic variables
        x, y, t = sym.symbols("x y t")

        # Physical parameters (user-defined)
        self.lambda_s: Scalar = 1  # Lame parameter
        self.mu_s: Scalar = 1  # Lame parameter
        self.alpha: Scalar = 1  # Biot's coupling coefficient
        self.stor: Scalar = 1  # Storativity (the one multiplying dp/dt)
        self.mu_f: Scalar = 1  # Dynamic viscosity of the fluid
        self.k: List[List[Scalar]] = [[1, 0], [0, 1]]  # Permeability tensor

        # Exact solutions

        # Pressure (user-defined)
        self.pressure: SymMul = t * x * (1 - x) * sym.sin(2 * sym.pi * y)

        # Displacement (user-defined)
        self.displacement: List[SymMul] = [
            t * x * (1 - x) * sym.sin(2 * sym.pi * y),
            t * sym.sin(2 * sym.pi * x) * sym.sin(2 * sym.pi * y),
        ]

        # The rest of the attributes should not be modified.

        # Pressure gradient
        self._pressure_gradient = [
            sym.diff(self.pressure, x),
            sym.diff(self.pressure, y),
        ]

        # Darcy flux
        self._darcy_flux = [
            (
                -(self.k[0][0] / self.mu_f) * self._pressure_gradient[0]
                - (self.k[0][1] / self.mu_f) * self._pressure_gradient[1]
            ),
            (
                -(self.k[1][0] / self.mu_f) * self._pressure_gradient[0]
                - (self.k[1][1] / self.mu_f) * self._pressure_gradient[1]
            ),
        ]

        # Divergence of Darcy flux
        self._divergence_darcy_flux = sym.diff(self._darcy_flux[0], x) + sym.diff(
            self._darcy_flux[1], y
        )

        # Divergence of displacement
        self._divergence_displacement = sym.diff(self.displacement[0], x) + sym.diff(
            self.displacement[1], y
        )

        # Time derivative of pressure
        self._time_derivative_pressure = sym.diff(self.pressure, t)

        # Time derivative of divergence of the displacement
        self._time_derivate_displacement = sym.diff(self._divergence_displacement, t)

        # Flow source
        self.source_flow: SymAdd = (
            self.stor * self._time_derivative_pressure
            + self.alpha * self._time_derivate_displacement
            + self._divergence_darcy_flux
        )

        # Gradient of the displacement
        self._gradient_displacement = [
            [sym.diff(self.displacement[0], x), sym.diff(self.displacement[0], y)],
            [sym.diff(self.displacement[1], x), sym.diff(self.displacement[1], y)],
        ]

        # Transpose of the gradient of the displacement
        self._gradient_displacement_transpose = [
            [self._gradient_displacement[0][0], self._gradient_displacement[1][0]],
            [self._gradient_displacement[0][1], self._gradient_displacement[1][1]],
        ]

        # Strain
        eps_xx = 0.5 * (
            self._gradient_displacement[0][0]
            + self._gradient_displacement_transpose[0][0]
        )
        eps_xy = 0.5 * (
            self._gradient_displacement[0][1]
            + self._gradient_displacement_transpose[0][1]
        )
        eps_yx = 0.5 * (
            self._gradient_displacement[1][0]
            + self._gradient_displacement_transpose[1][0]
        )
        eps_yy = 0.5 * (
            self._gradient_displacement[1][1]
            + self._gradient_displacement_transpose[1][1]
        )
        self._strain = [[eps_xx, eps_xy], [eps_yx, eps_yy]]

        # Effective stress
        self._trace_strain = self._strain[0][0] + self._strain[1][1]
        sigmaeff_xx = (
            self.lambda_s * self._trace_strain + 2 * self.mu_s * self._strain[0][0]
        )
        sigmaeff_xy = 2 * self.mu_s * self._strain[0][1]
        sigmaeff_yx = 2 * self.mu_s * self._strain[1][0]
        sigmaeff_yy = (
            self.lambda_s * self._trace_strain + 2 * self.mu_s * self._strain[1][1]
        )
        self._effective_stress = [
            [sigmaeff_xx, sigmaeff_xy],
            [sigmaeff_yx, sigmaeff_yy],
        ]

        # Total stress
        sigma_xx = self._effective_stress[0][0] - self.alpha * self.pressure
        sigma_xy = self._effective_stress[0][1]
        sigma_yx = self._effective_stress[1][0]
        sigma_yy = self._effective_stress[1][1] - self.alpha * self.pressure
        self._total_stress = [[sigma_xx, sigma_xy], [sigma_yx, sigma_yy]]

        # Divergence of total stress = mechanics source term
        fs_x = sym.diff(self._total_stress[0][0], x) + sym.diff(
            self._total_stress[1][0], y
        )
        fs_y = sym.diff(self._total_stress[0][1], x) + sym.diff(
            self._total_stress[1][1], y
        )
        self.source_mechanics: List[SymAdd] = [fs_x, fs_y]

    def integrated_source_flow(self, g: pp.Grid, time: float) -> np.ndarray:
        """Computes the integrated scalar sources over the grid.

        Args:
            g: PorePy grid.
            time: Time at which the exact source must be evaluated.

        Returns:
            integrated_source: Numerically integrated sources. Shape is (g.num_cells, ).

        Raises:
            ValuError: If the grid dimension is different from 2.

        """

        # Sanity check
        if g.dim != 2:
            raise ValueError("Expected two-dimensional grid.")

        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Retrieve QuadPy elements and declare numerical integration parameters
        elements = self.get_quadpy_elements(g)
        method = qp.t2.get_good_scheme(4)

        # Retrieve scalar source term
        f = self.source_flow

        # Lambdify expression
        f_fun = sym.lambdify((x, y, t), f, "numpy")

        # Declare integrand
        def integrand(coo):
            return f_fun(coo[0], coo[1], time)

        # Compute integration
        integrated_source = method.integrate(integrand, elements)

        return integrated_source

    def integrated_source_mechanics(self, g: pp.Grid, time: float) -> np.ndarray:
        """Computes the integrated vector sources over the grid.

        Args:
            g: PorePy grid.
            time: Time at which the exact source must be evaluated.

        Returns:
            integrated_source: Numerically integrated sources. Shape is (g.num_cells *
            g.dim, ).

        Raises:
            ValuError: If the grid dimension is different from 2.

        """

        # Sanity check
        if g.dim != 2:
            raise ValueError("Expected two-dimensional grid.")

        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Retrieve QuadPy elements and declare numerical integration parameters
        elements = self.get_quadpy_elements(g)
        method = qp.t2.get_good_scheme(4)

        # Retrieve components of vector source term
        f = self.source_mechanics

        # Lambdify expression
        f_fun_x = sym.lambdify((x, y, t), f[0], "numpy")
        f_fun_y = sym.lambdify((x, y, t), f[1], "numpy")

        # Declare integrands
        def integrandx(coo):
            return f_fun_x(coo[0], coo[1], time)

        def integrandy(coo):
            return f_fun_y(coo[0], coo[1], time)

        # Compute integration
        integrated_fx = method.integrate(integrandx, elements)
        integrated_fy = method.integrate(integrandy, elements)
        integrated_source = np.zeros(g.dim * g.num_cells)
        integrated_source[::2] = integrated_fx
        integrated_source[1::2] = integrated_fy

        return integrated_source

    # -------------> Error related methods
    @staticmethod
    def l2_relative_error(
        g: pp.Grid,
        true_val: np.ndarray,
        approx_val: np.ndarray,
        is_cc: bool,
        is_scalar: bool,
    ) -> float:
        """Compute the error measured in the discrete (relative) L2-norm.

        The employed norms correspond respectively to equations (75) and (76) for the
        displacement and pressure from https://epubs.siam.org/doi/pdf/10.1137/15M1014280.

        Args:
            g: PorePy grid.
            true_val: Exact array, e.g.: pressure, displacement, flux, or traction.
            approx_val: Approximated array, e.g.: pressure, displacement, flux, or traction.
            is_cc: True for cell-centered quanitities (e.g., pressure and displacement)
                and False for face-centered quantities (e.g., flux and traction).
            is_scalar: True for scalar quantities (e.g., pressure or flux) and False for
                vector quantities (displacement and traction).

        Returns:
            l2_error: discrete L2-error of the quantity of interest.

        """

        if is_cc:
            if is_scalar:
                meas = g.cell_volumes
            else:
                meas = g.cell_volumes.repeat(g.dim)
        else:
            if is_scalar:
                meas = g.cell_faces
            else:
                meas = g.cell_faces.repat(g.dim)

        numerator = np.sqrt(np.sum(meas * np.abs(true_val - approx_val) ** 2))
        denominator = np.sqrt(np.sum(meas * np.abs(true_val) ** 2))
        l2_error = numerator / denominator

        return l2_error

    # --------------> Utility methods
    @staticmethod
    def eval_scalar(
        g: pp.Grid, sym_expression: Union[SymAdd, SymMul], time: float
    ) -> np.ndarray:
        """
        Evaluate a symbolic scalar expression at the cell centers for a given time.

        Args:
            g: PorePy grid (simplicial for the moment).
            sym_expression: Symbolic expression dependent on x, y, and t.
            time: Time at which the symbolic expression should be evaluated.

        Returns:
            eval_exp: Evaluated expression at the cell centers. Shape is (g.num_cells, ).

        """

        cc = g.cell_centers

        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Lambdify expression
        fun = sym.lambdify((x, y, t), sym_expression, "numpy")

        # Evaluate at the cell centers
        eval_exp = fun(cc[0], cc[1], time)

        return eval_exp

    @staticmethod
    def eval_vector(
        g: pp.Grid, sym_expression: List[Union[SymAdd, SymMul]], time: float
    ) -> List[np.ndarray]:
        """
        Evaluate a symbolic scalar expression at the cell centers for a given time.

        Args:
            g: PorePy grid (simplicial for the moment).
            sym_expression: Symbolic expression dependent on x, y, and t.
            time: Time at which the symbolic expression should be evaluated.

        Returns:
            eval_exp: Evaluated expression at the cell centers. The output is a list of g.dim
                numpy arrays, each of shape (g.num.cells, ).

        """

        cc = g.cell_centers

        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Lambdify expression
        fun_x = sym.lambdify((x, y, t), sym_expression[0], "numpy")
        fun_y = sym.lambdify((x, y, t), sym_expression[1], "numpy")

        # Evaluate at the cell centers
        eval_exp_x = fun_x(cc[0], cc[1], time)
        eval_exp_y = fun_y(cc[0], cc[1], time)
        eval_exp = [eval_exp_x, eval_exp_y]

        return eval_exp

    @staticmethod
    def eval_tensor(
        g: pp.Grid, sym_expression: List[List[Union[SymAdd, SymMul]]], time: float
    ) -> List[List[np.ndarray]]:
        """
        Evaluate a symbolic tensor expression at the cell centers for a given time.

        Args:
            g: PorePy grid (simplicial for the moment).
            sym_expression: Symbolic expression dependent on x, y, and t.
            time: Time at which the symbolic expression should be evaluated.

        Returns:
            eval_exp: Evaluated expression at the cell centers. The output is a list of g.dim
                lists, each inner list contains g.dim numpy arrays, each of shape
                (g.num.cells, ).

        """

        cc = g.cell_centers

        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Lambdify expression
        fun_xx = sym.lambdify((x, y, t), sym_expression[0][0], "numpy")
        fun_xy = sym.lambdify((x, y, t), sym_expression[0][1], "numpy")
        fun_yx = sym.lambdify((x, y, t), sym_expression[1][0], "numpy")
        fun_yy = sym.lambdify((x, y, t), sym_expression[1][1], "numpy")

        # Evaluate at the cell centers
        eval_exp_xx = fun_xx(cc[0], cc[1], time)
        eval_exp_xy = fun_xy(cc[0], cc[1], time)
        eval_exp_yx = fun_yx(cc[0], cc[1], time)
        eval_exp_yy = fun_yy(cc[0], cc[1], time)
        eval_exp = [[eval_exp_xx, eval_exp_xy], [eval_exp_yx, eval_exp_yy]]

        return eval_exp

    @staticmethod
    def get_quadpy_elements(g: pp.Grid) -> np.ndarray:
        """
        Assemble elements of a grid in QuadPy format to prepare for numerical integration.

        For more details, see: https://pypi.org/project/quadpy/.

        Args:
            g: PorePy grid (simplicial for the moment).

        Returns:
            elements: Grid elements shaped in QuadPy format. Shape is (corners, cells, dim).
                For triangles, this is (3, g.num_cells, 2).

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
        element_coord = np.reshape(cnc_stckd, (nc, g.dim + 1, g.dim))

        # Reshaping to please QuadPy format i.e, (corners, num_elements, coords)
        elements = np.stack(element_coord, axis=-2)

        return elements
