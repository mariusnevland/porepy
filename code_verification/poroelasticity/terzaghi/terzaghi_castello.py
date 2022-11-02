"""
Runner script using parameters from Castello et. al. 2015
"""

import porepy as pp
import numpy as np
from terzaghi_model import Terzaghi
from time import time


# We want to export solutions for 4 dimensionless times
# The idea is to use a constant dimensionless time step of dtau = 1E-04
# For this purpose, we first need the characterisitc time, which is height ** 2
# For L = 1, the characteristic time is 1.

# # Create time manager object
# time_manager = pp.TimeManager(
#     schedule=np.array([0., 0.05, 0.1, 0.5, 2., 5., 10., 20., 40., 75., 110., 150., 250.]),
#     dt_init=0.05,
#     constant_dt=True
# )

# Model parameters
params = {
    "use_ad": True,
    "height": 1.0,  # [m]
    "mesh_size": 0.05,  # [m]
    "applied_load": 6e8,  # [N/m]
    "mu_lame": 2.475e9,  # [Pa],
    "lambda_lame": 1.65e9,  # [Pa],
    "alpha_biot": 1.0,  # [-]
    "permeability": 9.86e-14,  # [m^2],
    "viscosity": 1e-3,  # [Pa.s],
    "fluid_density": 1e3,  # [kg/m^3],
    "upper_limit_summation": 1000,
    "plot_results": True,
}

# Prepare model
model = Terzaghi(params)
model.prepare_simulation()
c_f = model.consolidation_coefficient()
dimless_times = np.array([0, 1E-3, 1E-2, 1E-1, 1])
times = (dimless_times * model.params["height"] ** 2) / c_f
dimless_dt = 1E-4
dt = (dimless_dt * model.params["height"] ** 2) / c_f

new_params = params.copy()
new_params["time_manager"] = pp.TimeManager(schedule=times, dt_init=dt, constant_dt=True)
# Run model
tic = time()
print("Simulation started...")
model = Terzaghi(new_params)
pp.run_time_dependent_model(model, new_params)
print(f"Simulation finished in {round(time() - tic)} sec.")



