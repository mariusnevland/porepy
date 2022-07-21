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
        Constructor for the ManufacturedBiot class.x

        Args:
            exact_sol: Exact solution object, containing the exact source terms.
            params: Model parameters.

        """
        super().__init__(params)
        self.exact_sol = exact_sol

    def create_grid(self) -> None:
        """Create Cartesian structured mixed-dimensional grid."""
        ref_lvl = self.params.get("refinement_level")
        phys_dims = np.array([1, 1])
        n_cells = np.array([2 * 2 ** ref_lvl, 2 * 2 ** ref_lvl])
        self.box = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
        sd: pp.Grid = pp.CartGrid(n_cells, phys_dims)
        sd.compute_geometry()
        self.mdg = pp.meshing.subdomains_to_mdg([[sd]])

    def _source_scalar(self, sd: pp.Grid) -> np.ndarray:
        """Integrated flow sources."""
        for sd, data in self.mdg.subdomains(return_data=True):
            # Retrieve exact source term
            source_flow = self.exact_sol.eval_scalar(sd, ex.source_flow, model.end_time)
            # Integrated source term
            integrated_source_flow = source_flow * sd.cell_volumes
            # Note that we have to scale the integrated sources with the time step explicitly.
            # This might change when the Assembler class is discontinued, see issue #675
            integrated_source_flow *= self.time_step
            return integrated_source_flow

    def _body_force(self, sd: pp.Grid) -> np.ndarray:
        """Integrated mechanical sources."""
        for sd, data in self.mdg.subdomains(return_data=True):
            # Retrieve exact source term
            body_force = np.asarray(
                self.exact_sol.eval_vector(sd, ex.source_mechanics, model.end_time)
            ).ravel("F")
            # Integrated source term
            integrated_body_force = body_force * sd.cell_volumes.repeat(sd.dim)
            return integrated_body_force

#%% Retrieve exact solution object
ex = ExactBiotManufactured()

# Define mesh sizes
refinement_levels = np.array([1, 2, 3, 4, 5, 6, 7])

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
    print(f"Refinement level {refinement_level}")
    print(f"Number of cells {model.mdg.num_subdomain_cells()}.")
    print(f"Simulation finished in {time() - tic} seconds.")

    # Retrieve approximated and exact values of primary variables
    mdg = model.mdg
    sd = model.mdg.subdomains(dim=2)[0]
    data = mdg.subdomain_data(sd)
    p_approx = data[pp.STATE][model.scalar_variable]
    u_approx = data[pp.STATE][model.displacement_variable]
    p_exact = ex.eval_scalar(sd, ex.pressure, model.end_time)
    u_exact = np.asarray(ex.eval_vector(sd, ex.displacement, model.end_time)).ravel("F")
    data[pp.STATE]["p_exact"] = p_exact
    data[pp.STATE]["u_exact"] = u_exact

    # Measure error
    error_p = ex.l2_relative_error(sd, p_exact, p_approx, is_cc=True, is_scalar=True)
    error_u = ex.l2_relative_error(sd, u_exact, u_approx, is_cc=True, is_scalar=False)

    # Print summary
    print(f"Pressure error: {error_p}")
    print(f"Displacement error: {error_u}")
    print()

    # Dump error into dictionary
    errors["pressure"][idx] = error_p
    errors["displacement"][idx] = error_u

    # Save source terms in data dictionaries
    source_flow = ex.eval_scalar(sd, ex.source_flow, model.end_time)
    data[pp.STATE]["source_flow"] = source_flow

    body_force = np.asarray(
        ex.eval_vector(sd, ex.source_mechanics, model.end_time)
    ).ravel("F")
    data[pp.STATE]["source_mechanics"] = body_force

# Determine rates of convergence
errors["reduction_p"] = errors["pressure"][:-1] / errors["pressure"][1:]
errors["reduction_u"] = errors["displacement"][:-1] / errors["displacement"][1:]
errors["order_p"] = np.log2(errors["reduction_p"])
errors["order_u"] = np.log2(errors["reduction_u"])


#%% Export solution
exporter = pp.Exporter(mdg, file_name="manu_biot", folder_name="out")
exporter.write_vtu(["p", "u", "source_flow", "source_mechanics"])

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
