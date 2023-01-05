import porepy as pp
import numpy as np
#import inspect

# default_model = pp.IncompressibleFlow()

class ChangedGrid(pp.IncompressibleFlow):
    """An Incompressible flow class with non-default mixed-dimensional grid.
    """

    def create_grid(self) -> None:
        """Create the grid bucket.

        A unit square grid with one vertical fracture extending from y=0.2 to
        y=0.5 is assigned.

        The method assigns the following attributes to self:
            mdg (pp.MixedDimensionalGrid): The produced mixed-dimensional grid.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
        """
        # Use default mesh size if none are provided in the parameters passed to the class
        # on initialization
        mesh_args = self.params.get("mesh_args", {"mesh_size_frac": .1})
        endp = np.array([.2, .5])
        #xendp = np.array([.2, .8])
        #yendp = np.array([.2, 1])
        self.mdg, self.box = pp.md_grids_2d.single_vertical(mesh_args, endp, simplex=True)
        #self.mdg, self.box = pp.md_grids_2d.two_intersecting(mesh_args, xendp,
        # yendp, simplex=True)


params = {}
model_changed_grid = ChangedGrid(params)
#pp.run_stationary_model(model_changed_grid, params)
#pp.plot_grid(model_changed_grid.mdg, model_changed_grid.variable, figsize=[10,7])


class ChangedPermeabilityAndSource(ChangedGrid):
    """An IncompressibleFlow model with modified grid and
    changed permeability."""

    def _permeability(self, sd):
        """Unitary permeability.
        Units: m^2
        """
        #print(sd.num_cells)
        return np.ones(sd.num_cells)
    #
    # def _source(self, sd):
    #     if sd.dim == self.mdg.dim_max():
    #         val = np.zeros(sd.num_cells)
    #     else:
    #         val = np.ones(sd.num_cells)
    #     return val

    def _bc_values(self, sd) -> np.ndarray:
        """Homogeneous boundary values.
        Units:
            Dirichlet conditions: Pa = kg / m^1 / s^2
            Neumann conditions: m^3 / s
        """
        bd = np.zeros(sd.num_faces)
        #Assign non-zero pressure at the top boundary. Need to find the indices of
        # the top faces and make sure we are in the highest-dimensional subdomain.
        if sd.dim == self.mdg.dim_max():
            bd[21] = 1
            bd[19] = 1
        print(bd)
        return bd


# We also prescribe a smaller mesh size:
params.update({"mesh_args": {"mesh_size_frac": 3}})
model_source = ChangedPermeabilityAndSource(params)
pp.run_stationary_model(model_source, params)

#inds = model_source.dof_manager.dof_var(var=[model_source.variable])
#vals = model_source.dof_manager.assemble_variable(variables = [model_source.variable])
#pressure = vals[inds]
#print(np.shape(pressure))
pp.plot_grid(model_source.mdg, model_source.variable,figsize=[
    10,7])




