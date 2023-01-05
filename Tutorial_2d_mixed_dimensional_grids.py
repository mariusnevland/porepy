import numpy as np
import porepy as pp

"""HOW TO SET UP A SIMULATION IN POREPY"""

"""Step 1. Make a new class which contains all of the information about the problem 
to be
solved. This class should inherit from one of the model classes in PorePy, and add
or modify certain methods in order to specify our specific problem setup. Usually
the least that must be done is to specify a grid. You could also specify boundary
conditions, add source terms etc."""

"""Example:"""


class ChangedGrid(pp.ContactMechanics):
    """A contact mechanics class with non-default mixed-dimensional grid.
    """

    def create_grid(self) -> None:
        """Create the grid bucket.

        A unit square grid with two intersecting fractures is assigned.

        The method assigns the following attributes to self:
            mdg (pp.MixedDimensionalGrid): The produced mixed-dimensional grid.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
        """
        mesh_args = self.params.get("mesh_args", {"mesh_size_frac": .05})
        xendp = np.array([.2, .8])
        yendp= np.array([.2, 1])
        self.mdg, self.box = pp.md_grids_2d.two_intersecting(mesh_args, xendp,yendp, simplex=True)
        pp.contact_conditions.set_projections(self.mdg)


"""The problem above will have a completely homogeneous solution though, 
due to default parameters (notably Dirichlet conditions of value 0). A simple way to 
make the solution a little bit more interesting is to add gravity:"""

"""In order to add boundary conditions you should know how the cells/faces/nodes are 
arranged. It seems like the cells and faces are multiple tuples/lists, first listing 
the cells/faces of the higher-dimensional domain, and then of the lower-dimensional 
domains (mortar + fracture?)"""


class ContactWithGravity(ChangedGrid):
    def _body_force(self, sd: pp.Grid) -> np.ndarray:
        """Body force parameter.

        If the source term represents gravity in the y (2d) or z (3d) direction,
        use:
            vals = np.zeros((self,nd, sd.num_cells))
            vals[-1] = density * pp.GRAVITY_ACCELERATION * sd.cell_volumes
            return vals.ravel("F")

        Parameters
        ----------
        sd : pp.Grid
            Subdomain, usually the matrix.

        Returns
        -------
        np.ndarray
            Integrated source values, shape self.nd * g.num_cells.

        """
        vals = np.zeros((self.nd, sd.num_cells))
        vals[-1] = 1 * pp.GRAVITY_ACCELERATION * sd.cell_volumes
        return vals.ravel("F")


"""Step 2: Initialize an "instance" of your new model class, and call one of the 
functions from run_models, which takes as input your (instance of the) model, 
and some parameters related to the solution procedure (Newton loop). Default 
parameters are used if you input an empty dictionary. The function from 
run_models then performs the simulation, i.e., the problem is solved numerically.
    
For contact mechanics without fluid flow, you use run_stationary_model.

The results can be plotted by using the function 
plot_grid(grid, cell_value: Optional, vector_value: Optional, info: Optional, 
**kwargs:Optional)

Note that scalar fields are put as the cell_value argument, while vector fields are 
put as the vector_value argument. Don't worry about the last two arguments for now.

Example:"""

params = {}
model = ContactWithGravity(params)
pp.run_stationary_model(model, params)
pp.plot_grid(model.mdg, vector_value=model.displacement_variable)
