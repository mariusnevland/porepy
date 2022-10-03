"""
Runner script for Terzaghi's problem.
"""

import porepy as pp
import numpy as np
from terzaghi_model import Terzaghi
from time import time

# Create tsc object
tsc = pp.TimeSteppingControl(
    schedule=np.array([0.0, 0.05, 0.1, 0.5, 2.0, 5.0, 10.0]),
    dt_init=0.05,
)

# Model parameters
model_params = {
    "use_ad": True,
    "height": 10.0,  # [m]
    "mesh_size": 0.5,  # [m]
    "applied_load": 6E8,  # [N/m]
    "mu_lame": 2.475E9,  # [Pa],
    "lambda_lame": 1.65E9,  # [Pa],
    "alpha_biot": 1.0,  # [-]
    "permeability": 9.86E-14,  # [m^2],
    "viscosity": 1E-3,  # [Pa.s],
    "fluid_density": 1E3,  # [kg/m^3],
    "upper_limit_summation": 1000,
    "time_step_object": tsc,
    "plot_results": True
}

# Run model
tic = time()
print("Simulation started...")
model = Terzaghi(model_params)
pp.run_time_dependent_model(model, model_params)
print(f"Simulation finished in {round(time() - tic)} sec.")
