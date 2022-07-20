#%% Import modules
from time import time
import porepy as pp
import numpy as np
import scipy.sparse as sps
import sympy as sym

from biot_manufactured_exact import ExactSolution

from typing import Dict, List, Optional


#%% Inherit from ContactMechanicsBiot and override relevant methods
class ManufacturedBiot(pp.ContactMechanicsBiot):
    """Manufactured Biot class"""

    def __init__(
        self,
        exact_sol,
        params: Optional[Dict] = None,
    ) -> None:
        super().__init__(params)
        self.exact_sol = exact_sol

    def create_grid(self) -> None:
        """Create the grid bucket"""
        self.box = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
        network_2d = pp.FractureNetwork2d(None, None, self.box)
        mesh_args = self.params.get("mesh_args", {"mesh_size_frac": 0.1})
        self.gb: pp.GridBucket = network_2d.mesh(mesh_args)

    def before_newton_loop(self) -> None:
        """Here we retrieve the time dependent source terms"""
        for g, d in self.gb:
            exact_scalar_source = self.exact_sol.integrated_source_flow(g, self.time)
            exact_vector_source = self.exact_sol.integrated_source_mechanics(
                g, self.time
            )
            # Note that we have to add the scaling of the time step explicitly
            # This might change when the Assembler class is discontinued, see issue #675
            d[pp.PARAMETERS][self.scalar_parameter_key]["source"] = (
                self.time_step * exact_scalar_source
            )
            d[pp.PARAMETERS][self.mechanics_parameter_key][
                "source"
            ] = exact_vector_source


#%% Retrieve exact solution

# Retrieve exact solution object
ex = ExactSolution()

# Define mesh sizes
mesh_sizes = np.array([0.1, 0.05, 0.025, 0.0125])

# Create dictionary to store errors
errors = {
    "mesh_sizes": mesh_sizes,
    "pressure": np.empty(shape=mesh_sizes.shape),
    "displacement": np.empty(shape=mesh_sizes.shape)
}

# Converge loop
for idx, mesh_size in enumerate(mesh_sizes):

    # Define simulation parameters
    params = {
        "time": 0.0,
        "time_step": 1.0,
        "end_time": 1.0,
        "use_ad": True,
        "mesh_args": {"mesh_size_frac": mesh_size}
    }

    # Construct model
    model = ManufacturedBiot(ex, params)

    # Run model
    tic = time()
    pp.run_time_dependent_model(model, params)
    print(f"Solving for mesh size {mesh_size}.")
    print(f"Simulation finished in {time() - tic} seconds.")

    # Retrieve approximated and exact values of primary variables
    gb = model.gb
    g = model.gb.grids_of_dimension(2)[0]
    d = model.gb.node_props(g)
    p_approx = d[pp.STATE][model.scalar_variable]
    u_approx = d[pp.STATE][model.displacement_variable]
    p_exact = ex.eval_scalar(g, ex.pressure, model.end_time)
    u_exact = np.asarray(ex.eval_vector(g, ex.displacement, model.end_time)).ravel("F")

    # Measure error
    error_p = ex.l2_relative_error(g, p_exact, p_approx, is_cc=True, is_scalar=True)
    error_u = ex.l2_relative_error(g, u_exact, u_approx, is_cc=True, is_scalar=False)

    # Print summary
    print(f"Pressure error: {error_p}")
    print(f"Displacement error: {error_u}")

    # Dump error into dictionary
    errors["pressure"][idx] = error_p
    errors["displacement"][idx] = error_u

# Determine rates of convergence
errors["reduction_p"] = errors["pressure"][:-1] / errors["pressure"][1:]
errors["reduction_u"] = errors["displacement"][:-1] / errors["displacement"][1:]
errors["order_p"] = np.log2(errors["reduction_p"])
errors["order_u"] = np.log2(errors["reduction_u"])

#%% Plot
# cc = g.cell_centers
# ff = ex.eval_scalar(g, ex.source_flow, model.end_time)
# fs = ex.eval_vector(g, ex.source_mechanics, model.end_time)
# int_ff = ex.integrated_source_flow(g, model.end_time)
# int_fs = ex.integrated_source_mechanics(g, model.end_time)
#
# #%% Plot sources
# plot_source = False
# if plot_source:
#     pp.plot_grid(g, ff * g.cell_volumes, plot_2d=True, title="Scalar source")
#     pp.plot_grid(g, fs[0] * g.cell_volumes, plot_2d=True, title="Vector source (x)")
#     pp.plot_grid(g, fs[1] * g.cell_volumes, plot_2d=True, title="Vector source (y)")
#     # pp.plot_grid(g, int_ff, plot_2d=True, title="Integrated scalar source QP")
#     # pp.plot_grid(g, int_fs[::2], plot_2d=True, title="Integrated vector source (x) QP")
#     # pp.plot_grid(g, int_fs[1::2], plot_2d=True, title="Integrated vector source (y) QP")
#
# # %% Plot pressure
# plot_pressure = True
# if plot_pressure:
#     pp.plot_grid(g, d[pp.STATE]["p"], plot_2d=True, title="Approximated pressure")
#     pp.plot_grid(
#         g,
#         ex.eval_scalar(g, ex.pressure, model.end_time),
#         plot_2d=True,
#         title="Exact pressure",
#     )
#
# # %% Plot horizontal displacement
# plot_displacement = True
# if plot_displacement:
#     pp.plot_grid(
#         g,
#         d[pp.STATE]["u"][::2],
#         plot_2d=True,
#         title="Approximated horizontal displacement",
#     )
#     pp.plot_grid(
#         g,
#         ex.eval_vector(g, ex.displacement, model.end_time)[0],
#         plot_2d=True,
#         title="Exact horizontal displacement",
#     )
#
# # %% Plot vertical displacement
# if plot_displacement:
#     pp.plot_grid(
#         g,
#         d[pp.STATE]["u"][1::2],
#         plot_2d=True,
#         title="Approximated vertical displacement",
#     )
#     pp.plot_grid(
#         g,
#         ex.eval_vector(g, ex.displacement, model.end_time)[1],
#         plot_2d=True,
#         title="Exact vertical displacement",
#     )
