import numpy as np
import porepy as pp

from typing import Union

Scalar = Union[int, float]


class TerzaghiUtilities:
    """Class containing the exact solutions to Terzaghi's consolidation problem.

    Attributes:
        cell_centers (np.ndarray): Coordinates of the cell centers in the vertical
            direction. The shape is (num_vertical_cells, ).
        roof_series (int): Roof value for computing the summation series of the
            analytical expressions.
        height (Scalar): Heigh of the domain.
        consol_coeff (Scalar): Coefficient of consolidation.
        vert_load (Scalar): Vertical load imposed on the top boundary of the domain.

    """

    def __init__(self, terzaghi_model):
        """Constructor for the class.

        Args:
            terzaghi_model: Terzaghi model class. It is assumed that a Cartesian grid was
                already created and computed. Morever, it is assumend that the coefficient of
                consolidation is stored in `terzaghi_model.params["consolidation_coefficient"]`
                and also the vertical load in `terzaghi_model.params["vertical_load"]`.

        """

        self.model = terzaghi_model
        self.roof_series: int = 1000

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

    # ----------> Analytical solutions
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

    def dimless_p(self, t: Scalar) -> np.ndarray:
        """
        Compute exact dimensionless pressure.

        Args:
            t: Time in seconds.

        Returns:
            dimless_p: Dimensionless pressure profile for the given time `t`.

        """

        sd = self.model.mdg.subdomains()[0]
        cc = sd.cell_centers[1]
        h = self.model.box["ymax"]
        vert_load = self.model.params["vertical_load"]
        dimless_t = self.dimless_t(t)

        sum_series = np.zeros_like(cc)
        for i in range(1, self.roof_series + 1):
            sum_series += (
                (((-1) ** (i - 1)) / (2 * i - 1))
                * np.cos((2 * i - 1) * (np.pi / 2) * (cc / h))
                * np.exp(
                    (-((2 * i - 1) ** 2)) * (np.pi**2 / 4) * dimless_t
                )
            )
        dimless_p = (4 / np.pi) * vert_load * sum_series

        return dimless_p

    def consol_deg(self, t: Scalar) -> Scalar:
        """Compute the degree of consolidation for a given time.

        Args:
            t: Time in seconds.

        Returns:
            deg_cons: Degree of consolidation for the given time `t`.

        """

        dimless_t = self.dimless_t(t)
        sum_series = 0
        for i in range(1, self.roof_series + 1):
            sum_series += (
                1
                / ((2 * i - 1) ** 2)
                * np.exp(-((2 * i - 1) ** 2) * (np.pi**2 / 4) * dimless_t)
            )
        deg_cons = 1 - (8 / (np.pi**2)) * sum_series

        return deg_cons

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
        closest_cells = sd.closest_cell(np.array([h/2 * np.ones_like(yc), yc]))
        _, idx = np.unique(closest_cells, return_index=True)
        y_points = closest_cells[np.sort(idx)]
        cut_array = array[y_points]

        return cut_array


