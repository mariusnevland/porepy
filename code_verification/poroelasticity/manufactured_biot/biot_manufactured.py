#%% Import modules
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
        mesh_args = self.params.get("mesh_args", {"mesh_size_frac": 0.05})
        self.gb: pp.GridBucket = network_2d.mesh(mesh_args)

    def _source_scalar(self, g: pp.Grid) -> np.ndarray:
        """Override scalar source"""
        time = params.get("time")
        return self.exact_sol.integrated_scalar_source(g, time)

    def _vector_source(self, g: pp.Grid) -> np.ndarray:
        """Override vector source"""
        time = params.get("time")
        return self.exact_sol.integrated_vector_source(g, time)


#%%
ex = ExactSolution()
params = {"time": 0.1}
model = ManufacturedBiot(ex, params)
model.time_step = 0.1
model.create_grid()
gb = model.gb

pp.run_time_dependent_model(model, params)

g = model.gb.grids_of_dimension(2)[0]
d = model.gb.node_props(model.gb.grids_of_dimension(2)[0])


#%% Plot
cc = g.cell_centers
tt = 0.1
ff = ex.scalar_source("fun")(cc[0], cc[1], tt)
fsx = ex.vector_source("fun")[0](cc[0], cc[1], tt)
fsy = ex.vector_source("fun")[1](cc[0], cc[1], tt)
int_ff = ex.integrated_scalar_source(g, tt)
int_fs = ex.integrated_vector_source(g, tt)

#pp.plot_grid(g, ff, plot_2d=True, title="Scalar source")
#pp.plot_grid(g, fsx, plot_2d=True, title="Vector source (x)")
#pp.plot_grid(g, fsy, plot_2d=True, title="Vector source (y)")
pp.plot_grid(g, int_ff, plot_2d=True, title="Integrated scalar source QP")
pp.plot_grid(g, int_fs[::2], plot_2d=True, title="Integrated vector source (x) QP")
pp.plot_grid(g, int_fs[1::2], plot_2d=True, title="Integrated vector source (y) QP")