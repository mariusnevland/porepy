import numpy as np
import porepy as pp

from typing import Union

Scalar = Union[int, float]


class TerzaghiUtilities:
    """Class containing the exact solutions to Terzaghi's consolidation problem.

    Attributes:
        model (Terzaghi): Terzaghi model class.
        upper_limit (int): Upper limit of summation in computation of exact solutions.
        dimless_y (np.ndarray): Dimensionless vertical coordinates obtained by cutting the
            domain trough the line (height/2, 0) (height/2, height).

    """

    def __init__(self, terzaghi_model):
        """Constructor for the class.

        Args:
            terzaghi_model: Terzaghi model class with mandatory parameters properly passed.

        """

        self.model = terzaghi_model
        self.upper_limit: int = self.model.params.get("upper_limit", 1000)

        yc = self.model.mdg.subdomains()[0].cell_centers[1]
        self.dimless_y = self.vert_cut(yc) / self.model.params["height"]

    # ---------> Physical parameters
    def confined_compressibility(self) -> np.ndarray:
        """Confined compressibility [1/Pa].

        Returns:
            confined_compressibility (sd.num_cells, ): confined compressibility.

        """

        sd = self.model.mdg.subdomains()[0]
        stifness_tensor = self.model._stiffness_tensor(sd)
        confined_compressibility = 1 / (2 * stifness_tensor.mu + stifness_tensor.lmbda)

        return confined_compressibility

    def consolidation_coefficient(self) -> np.ndarray:
        """Consolidation coefficient [-].

        Returns:
            consolidation_coefficient (sd.num_cells, ): coefficient of consolidation.

        """

        sd = self.model.mdg.subdomains()[0]

        permeability = self.model._permeability(sd)
        volumetric_weight = np.ones(sd.num_cells)
        viscosity = self.model._viscosity(sd)
        hydraulic_conductivity = (permeability * volumetric_weight) / viscosity
        storativity = self.model._storativity(sd)
        alpha_biot = self.model._biot_alpha(sd)
        confined_compressibility = self.confined_compressibility()
        consolidation_coefficient = hydraulic_conductivity / (
            volumetric_weight
            * (storativity + alpha_biot**2 * confined_compressibility)
        )

        return consolidation_coefficient

    # ----------> Analytical solutions and dimensionless quantities
    def dimless_t(self, t: Scalar) -> Scalar:
        """
        Compute exact dimensionless time.

        Args:
            t: Time in seconds.

        Returns:
            Dimensionless time for the given time `t`.

        """

        h = self.model.box["ymax"]
        c_f = np.mean(self.consolidation_coefficient())

        return (t * c_f) / (h**2)

    def dimless_p_ex(self, t: Scalar) -> np.ndarray:
        """
        Compute exact dimensionless pressure.

        Args:
            t: Time in seconds.

        Returns:
            Dimensionless pressure profile for the given time `t`.

        """

        sd = self.model.mdg.subdomains()[0]
        cc = sd.cell_centers[1]
        h = self.model.box["ymax"]
        vert_load = self.model.params["vertical_load"]
        dimless_t = self.dimless_t(t)

        sum_series = np.zeros_like(cc)
        for i in range(1, self.upper_limit + 1):
            sum_series += (
                (((-1) ** (i - 1)) / (2 * i - 1))
                * np.cos((2 * i - 1) * (np.pi / 2) * (cc / h))
                * np.exp((-((2 * i - 1) ** 2)) * (np.pi**2 / 4) * dimless_t)
            )
        dimless_p = (4 / np.pi) * vert_load * sum_series

        return self.vert_cut(dimless_p)

    def consol_deg_ex(self, t: Scalar) -> Scalar:
        """Compute the degree of consolidation for a given time.

        Args:
            t: Time in seconds.

        Returns:
            deg_cons: Degree of consolidation for the given time `t`.

        """

        dimless_t = self.dimless_t(t)
        sum_series = 0
        for i in range(1, self.upper_limit + 1):
            sum_series += (
                1
                / ((2 * i - 1) ** 2)
                * np.exp(-((2 * i - 1) ** 2) * (np.pi**2 / 4) * dimless_t)
            )
        deg_cons = 1 - (8 / (np.pi**2)) * sum_series

        return deg_cons

    def dimless_p_num(self, pressure: np.ndarray) -> np.ndarray:
        """Dimensionalize pressure solution.

        Args:
            pressure (sd.num_cells, ): pressure solution.

        Returns:
            dimless_p (sd.num_cells, ): dimensionless pressure solution.

        """

        p_0 = self.model.params["vertical_load"]
        dimless_pressure = pressure / p_0

        return self.vert_cut(dimless_pressure)

    def consol_deg_num(self, displacement: np.ndarray, pressure: np.ndarray) -> float:
        """Numerical consolidation coefficient.

        Args:
            displacement (sd.dim * sd.num_cells, ): displacement solution.
            pressure (sd.num_cells, ): pressure solution.

        Returns:
            consol_deg: consolidation coefficient.

        """

        sd = self.model.mdg.subdomains()[0]
        h = self.model.params["height"]
        m_v = np.mean(self.confined_compressibility())
        vert_load = self.model.params["vertical_load"]
        trace_u = self.displacement_trace(displacement, pressure)

        u_inf = m_v * h * vert_load
        u_0 = 0
        u = np.max(np.abs(trace_u[1 :: sd.dim]))

        consol_deg = (u - u_0) / (u_inf - u_0)

        return consol_deg

    # -----------> Helper methods
    def vert_cut(self, array: np.ndarray) -> np.ndarray:
        """Perform a vertical cut in the middle of the domain.

        Note:
            This is done by obtaining the closest vertical cell-centers to the line
            (h/2, 0) (h/2, h). This functionality is similar to the Plot Over Line
            tool from ParaView.

        """
        sd = self.model.mdg.subdomains()[0]
        h = self.model.box["ymax"]
        half_max_diam = np.max(sd.cell_diameters()) / 2
        yc = np.arange(0, h, half_max_diam)
        closest_cells = sd.closest_cell(np.array([h / 2 * np.ones_like(yc), yc]))
        _, idx = np.unique(closest_cells, return_index=True)
        y_points = closest_cells[np.sort(idx)]
        cut_array = array[y_points]

        return cut_array

    def displacement_trace(
        self, displacement: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """Project the displacement vector onto the faces.

        Args:
            displacement (sd.dim * sd.num_cells, ): displacement solution.
            pressure (sd.num_cells, ): pressure solution.

        Returns:
            trace_u (sd.dim * sd.num_faces, ): trace of the displacement.

        """

        # Rename arguments
        u = displacement
        p = pressure

        # Discretization matrices
        sd = self.model.mdg.subdomains()[0]
        data = self.model.mdg.subdomain_data(sd)
        bound_u_cell = data[pp.DISCRETIZATION_MATRICES][
            self.model.mechanics_parameter_key
        ]["bound_displacement_cell"]
        bound_u_face = data[pp.DISCRETIZATION_MATRICES][
            self.model.mechanics_parameter_key
        ]["bound_displacement_face"]
        bound_u_pressure = data[pp.DISCRETIZATION_MATRICES][
            self.model.mechanics_parameter_key
        ]["bound_displacement_pressure"]

        # Mechanical boundary values
        bc_vals = data[pp.PARAMETERS][self.model.mechanics_parameter_key]["bc_values"]

        # Compute trace of the displacement
        trace_u = bound_u_cell * u + bound_u_face * bc_vals + bound_u_pressure * p

        return trace_u
