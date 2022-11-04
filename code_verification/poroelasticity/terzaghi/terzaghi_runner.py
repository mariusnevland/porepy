"""
Runner script for Terzaghi's consolidation problem.
"""

import porepy as pp
from terzaghi_model import Terzaghi
from time import time

# Time manager
time_manager = pp.TimeManager([0, 0.01, 0.1, 0.5, 1, 2], 0.001, constant_dt=True)

# Model parameters
params = {
        'alpha_biot': 1.0,  # [-]
        'height': 1.0,  # [m]
        'lambda_lame': 1.65E9,  # [Pa]
        'mu_lame': 1.475E9,  # [Pa]
        'num_cells': 20,
        'permeability': 9.86E-14,  # [m^2]
        'perturbation_factor': 1E-6,
        'plot_results': True,
        'specific_weight': 9.943E3,  # [Pa * m^-1]
        'time_manager': time_manager,
        'upper_limit_summation': 1000,
        'use_ad': True,
        'vertical_load': 6E8,  # [N * m^-1]
        'viscosity': 1E-3,  # [Pa * s]
    }

# Run model
tic = time()
print("Simulation started...")
model = Terzaghi(params)
pp.run_time_dependent_model(model, params)
print(f"Simulation finished in {round(time() - tic)} sec.")
