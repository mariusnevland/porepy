"""Implementation of Mandel's problem."""

import porepy as pp
import numpy as np

from mandel_utilities import MandelUtilities

class Mandel(pp.ContactMechanicsBiot):
    """Parent class for Mandel's problem."""

    def __init__(self, params: dict):
        """Constructor of Mandel class.

        Args:
            params: Model parameters.

        Mandatory model parameters:
            mu_lame (float): First Lamé parameter [Pa]
            lambda_lame (float): Seconda Lamé parameter [Pa]
            permeability (float): Intrinsic permeability [m^2]
            alpha_biot (float): Biot-Willis coefficient [-]
            viscosity (float): Fluid dynamic viscosity [Pa s]
            storativity (float): Storativity or specific storage [1/Pa]
            applied_load (float): Vertically applied load [N/m]

        Optional model parameters:

        """
        super().__init__(params)

        self.tsc = self.params["time_stepping_object"]
        self.time = self.tsc.time_init
        self.end_time = self.tsc.time_final
        self.time_step = self.tsc.dt

        # Create a solution dictionary to store pressure and displacement solutions
        self.sol = {t: {} for t in self.tsc.schedule}

    def create_grid(self) -> None:
        """Create two-dimensional unstructured mixed-dimensional grid."""
        height = self.params["height"]
        mesh_size = self.params.get("mesh_size", 0.1)
        self.box = {"xmin": 0.0, "xmax": height, "ymin": 0.0, "ymax": height}
        network_2d = pp.FractureNetwork2d(None, None, self.box)
        mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}
        self.mdg = network_2d.mesh(mesh_args)


#%% Running the model
model_params = {
    "mu_lame": 2.475E9,
    "lambda_lame": 1.65E9,
    "permeability": 9.8692E-14,
    "alpha_biot": 1.0,
    "viscosity": 1E-3,
    "storativity": 6.0606E-11,
    "applied_load": 6.08E8,
}