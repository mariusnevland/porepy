#%% Import modules
from time import time
import porepy as pp
import numpy as np
from biot_manufactured_exact import ExactBiotManufactured


#%% Inherit from ContactMechanicsBiot and override relevant methods
class ManufacturedBiot(pp.ContactMechanicsBiot):
    """Class for setting up a manufactured solution for the Biot problem without fractures.

    Attributes:
        exact_sol (ExactBiotManufactured): Exact solution object.

    """
    def __init__(self, exact_sol: ExactBiotManufactured, params: dict):
        """
        Constructor for the ManufacturedBiot class.

        Args:
            exact_sol: Exact solution object, containing the exact source terms.
            params: Model parameters.

        """
        super().__init__(params)
        self.exact_sol = exact_sol

    def create_grid(self) -> None:
        """Create Cartesian structured mixed-dimensional grid."""
        ref_lvl = self.params.get("refinement_level")
        phys_dims = [1.0, 1.0]
        n_cells = [2 * 2 ** ref_lvl, 2 * 2 ** ref_lvl]
        self.box = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
        g: pp.Grid = pp.CartGrid(n_cells, phys_dims)
        g.compute_geometry()
        self.gb: pp.GridBucket = pp.meshing._assemble_in_bucket([[g]])

    def before_newton_loop(self) -> None:
        """Assign time-dependent source terms."""
        for g, d in self.gb:
            # Retrieve exact sources
            source_flow_cc = self.exact_sol.eval_scalar(g, ex.source_flow, model.end_time)
            source_mech_cc = np.asarray(
                self.exact_sol.eval_vector(g, ex.source_mechanics, model.end_time)
            ).ravel("F")

            # Integrated sources (technically, we're applying the midpoint rule here).
            int_source_flow_cc = source_flow_cc * g.cell_volumes
            int_source_mech_cc = source_mech_cc * g.cell_volumes.repeat(g.dim)

            # Note that we have to add the scaling of the time step explicitly
            # This might change when the Assembler class is discontinued, see issue #675
            d[pp.PARAMETERS][self.scalar_parameter_key]["source"] = (
                self.time_step * int_source_flow_cc
            )
            d[pp.PARAMETERS][self.mechanics_parameter_key]["source"] = int_source_mech_cc


#%% Retrieve exact solution object
ex = ExactBiotManufactured()

# Define mesh sizes
refinement_levels = np.array([1, 2, 3, 4, 5, 6])

# Create dictionary to store errors
errors = {
    "refinement_levels": refinement_levels,
    "pressure": np.empty(shape=refinement_levels.shape),
    "displacement": np.empty(shape=refinement_levels.shape)
}

# Converge loop
for idx, refinement_level in enumerate(refinement_levels):

    # Define simulation parameters
    params = {
        "time": 0.0,
        "time_step": 1.0,
        "end_time": 1.0,
        "use_ad": True,
        "refinement_level": refinement_level
    }

    # Construct model
    model = ManufacturedBiot(ex, params)

    # Run model
    tic = time()
    pp.run_time_dependent_model(model, params)
    print(f"Refinement level {refinement_level}. Number of cells {model.gb.num_cells()}.")
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
    print()

    # Dump error into dictionary
    errors["pressure"][idx] = error_p
    errors["displacement"][idx] = error_u

# Determine rates of convergence
errors["reduction_p"] = errors["pressure"][:-1] / errors["pressure"][1:]
errors["reduction_u"] = errors["displacement"][:-1] / errors["displacement"][1:]
errors["order_p"] = np.log2(errors["reduction_p"])
errors["order_u"] = np.log2(errors["reduction_u"])


#%% Export solution
exporter = pp.Exporter(model.gb, file_name="manu_biot", folder_name="out")
exporter.write_vtu(["p", "u"])

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
