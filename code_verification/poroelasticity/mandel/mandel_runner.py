"""
Runner script for Mandel's problem.
"""

import porepy as pp
from mandel_model import Mandel
from time import time

# Create tsc object
time_manager = pp.TimeManager(
    schedule=[
        0,
        10,
        50,
        100,
        1000,
        5000,
        8000,
        10000,
        20000,
        30000,
        50000,
    ],  # [s]
    dt_init=10,  # [s]
    constant_dt=True,
)

# Create model's parameter dictionary
params = {
    "use_ad": True,  # Only "use_ad: True" is supported for this model
    "mu_lame": 2.475e9,  # [Pa]
    "lambda_lame": 1.650e9,  # [Pa]
    "permeability": 9.869e-14,  # [m^2]
    "alpha_biot": 1.0,  # [-]
    "viscosity": 1e-3,  # [Pa.s]
    "storativity": 6.0606e-11,  # [1/Pa]
    "applied_load": 6e8,  # [N/m]
    "height": 10.0,  # [m]
    "width": 100.0,  # [m]
    "mesh_size": 2.0,  # [m]
    "time_manager": time_manager,
    "number_of_roots": 200,
    "plot_results": True,
}

# Run model
tic = time()
print("Simulation started...")
model = Mandel(params)
pp.run_time_dependent_model(model, params)
print(f"Simulation finished in {round(time() - tic)} sec.")
