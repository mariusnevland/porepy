"""
Runner script for Terzaghi's problem.
"""

import porepy as pp
import numpy as np
from terzaghi_model import Terzaghi
from time import time

# Create time manager object
time_manager = pp.TimeManager(
    schedule=np.array([0., 0.05, 0.1, 0.5, 2., 5., 10., 20., 40., 75., 110., 150., 250.]),
    dt_init=0.05,
    constant_dt=True
)

# Model parameters
params = {
    "use_ad": True,
    "height": 10.0,  # [m]
    "mesh_size": 0.5,  # [m]
    "applied_load": 6e8,  # [N/m]
    "mu_lame": 2.475e9,  # [Pa],
    "lambda_lame": 1.65e9,  # [Pa],
    "alpha_biot": 1.0,  # [-]
    "permeability": 9.86e-14,  # [m^2],
    "viscosity": 1e-3,  # [Pa.s],
    "fluid_density": 1e3,  # [kg/m^3],
    "upper_limit_summation": 1000,
    "time_manager": time_manager,
    "plot_results": True,
}

# Run model
tic = time()
print("Simulation started...")
model = Terzaghi(params)
pp.run_time_dependent_model(model, params)
print(f"Simulation finished in {round(time() - tic)} sec.")
