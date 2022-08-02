import numpy as np
import porepy as pp
import sympy as sym

from typing import List, Union

Scalar = Union[int, float]
SymMul = sym.core.mul.Mul
SymAdd = sym.core.add.Add


class ExactBiotManufactured:
    """Class containing the exact solutions to the manufactured Biot's problem.

    The exact solutions are modified from https://epubs.siam.org/doi/pdf/10.1137/15M1014280
    and are given by:

        u(x, y, t) = t * [x * (1 - x) * sin(2 * pi * y), sin(2 * pi * x) * sin(2 * pi * y)],
        p(x, y, t) = t * x * (1 - x) * sin(2 * pi * y),

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