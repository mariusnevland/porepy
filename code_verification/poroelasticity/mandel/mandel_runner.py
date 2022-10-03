import porepy as pp
from mandel_model import Mandel
from time import time

# Create tsc object
tsc = pp.TimeSteppingControl(
    schedule=[
        0.0,
        10.0,
        50.0,
        100.0,
        1000.0,
        5000.0,
        8000.0,
        10000.0,
        20000.0,
        30000.0,
        50000.0,
    ],  # [s]
    dt_init=10.0,  # [s]
    constant_dt=True
)

# Create model's parameter dictionary
model_params = {
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
    "time_step_object": tsc,
    "number_of_roots": 200,
    "plot_results": True,
}

# Run model
tic = time()
print("Simulation started...")
model = Mandel(model_params)
pp.run_time_dependent_model(model, model_params)
print(f"Simulation finished in {round(time() - tic)} sec.")
