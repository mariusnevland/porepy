#%% Import modules
from time import time
import porepy as pp
import numpy as np
import scipy.sparse as sps
import sympy as sym

from biot_manufactured_exact import ExactSolution

from typing import Dict, Optional


#%% Inherit class
class ManufacturedBiot(pp.ContactMechanicsBiot):
    """Manufactured Biot class"""
    def __init__(self, exact_sol, params: Optional[Dict] = None, ) -> None:
        super().__init__(params)
        self.exact_sol = exact_sol

    def create_grid(self) -> None:
        """Create the grid bucket"""
        self.box = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
        network_2d = pp.FractureNetwork2d(None, None, self.box)
        mesh_args = self.params.get("mesh_args", {"mesh_size_frac": 0.025})
        self.gb: pp.GridBucket = network_2d.mesh(mesh_args)

    def before_newton_loop(self) -> None:
        """Here we retrieve the time dependent source terms"""
        t = self.time
        dt = self.time_step
        next_t = t + dt
        for g, d in self.gb:
            exact_scalar_source = self.exact_sol.integrated_source_flow(g, next_t)
            exact_vector_source = self.exact_sol.integrated_source_mechanics(g, next_t)
            d[pp.PARAMETERS][self.scalar_parameter_key]["source"] = exact_scalar_source
            d[pp.PARAMETERS][self.mechanics_parameter_key]["source"] = exact_vector_source


#%%

ex = ExactSolution()
params = {"time": 0, "time_step": 1.0, "end_time": 1.0, "use_ad": True}
model = ManufacturedBiot(ex, params)
model.create_grid()
gb = model.gb

tic = time()
pp.run_time_dependent_model(model, params)
print(f"Simulation finished in {time() - tic}")

g = model.gb.grids_of_dimension(2)[0]
d = model.gb.node_props(model.gb.grids_of_dimension(2)[0])


#%% Plot
cc = g.cell_centers
ff = ex.eval_scalar(g, ex.source_flow, 1)
fs = ex.eval_vector(g, ex.source_mechanics, 1)
int_ff = ex.integrated_source_flow(g, 1)
int_fs = ex.integrated_source_mechanics(g, 1)

pp.plot_grid(g, ff * g.cell_volumes, plot_2d=True, title="Scalar source")
pp.plot_grid(g, fs[0] * g.cell_volumes, plot_2d=True, title="Vector source (x)")
pp.plot_grid(g, fs[1] * g.cell_volumes, plot_2d=True, title="Vector source (y)")
pp.plot_grid(g, int_ff, plot_2d=True, title="Integrated scalar source QP")
pp.plot_grid(g, int_fs[::2], plot_2d=True, title="Integrated vector source (x) QP")
pp.plot_grid(g, int_fs[1::2], plot_2d=True, title="Integrated vector source (y) QP")

# Plot pressure
pp.plot_grid(g, d[pp.STATE]["p"], plot_2d=True, title="Approximated pressure")
pp.plot_grid(g, ex.eval_scalar(g, ex.pressure, 1), plot_2d=True, title="Exact pressure")

# Plot horizontal displacement
pp.plot_grid(
    g,
    d[pp.STATE]["u"][::2],
    plot_2d=True,
    title="Approximated horizontal displacement"
)
pp.plot_grid(
    g,
    ex.eval_vector(g, ex.displacement, 1)[0],
    plot_2d=True,
    title="Exact horizontal displacement"
)

# Plot vertical displacement
pp.plot_grid(
    g,
    d[pp.STATE]["u"][1::2],
    plot_2d=True,
    title="Approximated vertical displacement"
)
pp.plot_grid(
    g,
    ex.eval_vector(g, ex.displacement, 1)[1],
    plot_2d=True,
    title="Exact vertical displacement"
)






