import numpy as np
import porepy as pp

from typing import Union

Scalar = Union[int, float]


class ExactTerzaghi:
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

        # Retrieve mixed-dimensional grid from the model class
        mdg: pp.MixedDimensionalGrid = terzaghi_model.mdg

        # Get Cartesian grid
        sd: pp.Grid = mdg.subdomains()[0]

        # Retrieve cell centers in the vertical direction
        self.cell_centers: np.ndarray = sd.cell_centers[1][:: sd.cart_dims[0]]

        # Define the roof value for stopping the sumation series of the analytical solution
        self.roof_series: int = 1000

        # Retrieve the maximum vertical length, a.k.a the height of the column
        self.height: Scalar = terzaghi_model.box["ymax"]

        # Get coefficient of consolidation from the model parameters
        self.consol_coeff: Scalar = terzaghi_model.params["consolidation_coefficient"]

        # Get vertical load from the model parameters
        self.vert_load: Scalar = terzaghi_model.params["vertical_load"]

    def dimensionless_times(self, t: Scalar) -> Scalar:
        """
        Compute exact dimensionless time.

        Args:
            t: Time in seconds.

        Returns:
            Dimensionless time for the given time `t`.

        """

        return (t * self.consol_coeff) / (self.height ** 2)

    def dimensionless_pressure(self, t: Scalar) -> np.ndarray:
        """
        Compute exact dimensionless pressure.

        Args:
            t: Time in seconds.

        Returns:
            dimless_p: Dimensionless pressure profile for the given time `t`.

        """

        sum_series = np.zeros_like(self.cell_centers)
        for i in range(1, self.roof_series + 1):
            sum_series += (
                (((-1) ** (i - 1)) / (2 * i - 1))
                * np.cos((2 * i - 1) * (np.pi / 2) * (self.cell_centers / self.height))
                * np.exp(
                    (-((2 * i - 1) ** 2))
                    * (np.pi ** 2 / 4)
                    * (self.consol_coeff * t)
                    / (self.height ** 2)
                )
            )
        dimless_p = (4 / np.pi) * self.F * sum_series

        return dimless_p

    def degree_of_consolidation(self, t: Scalar) -> Scalar:
        """Compute the degree of consolidation for a given time.

        Args:
            t: Time in seconds.

        Returns:
            deg_cons: Degree of consolidation for the given time `t`.

        """

        sum_series = 0
        for i in range(1, self.roof_series + 1):
            sum_series += (
                1
                / ((2 * i - 1) ** 2)
                * np.exp(-((2 * i - 1) ** 2) * (np.pi ** 2 / 4) * t)
            )
        deg_cons = 1 - (8 / (np.pi ** 2)) * sum_series

        return deg_cons
