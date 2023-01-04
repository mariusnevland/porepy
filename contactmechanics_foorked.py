import porepy as pp
import numpy as np
import inspect

class ChangedGrid(pp.ContactMechanics):
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
        #phys_dims = np.array([1, 1])
        #n_cells = np.array([3, 3])
        #self.box = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
        #g: pp.Grid = pp.CartGrid(n_cells, phys_dims)
        #g.compute_geometry()
        #self.mdg = pp.meshing.subdomains_to_mdg([[g]])
        # mesh_args = self.params.get("mesh_args", {"mesh_size_frac": .1})
        mesh_args={"mesh_size_frac": .1}
        xendp = np.array([.2, .8])
        yendp= np.array([.2, 1])
        self.mdg, self.box = pp.md_grids_2d.two_intersecting(mesh_args, xendp,yendp, simplex=True)
        # self.mdg, self.box = pp.md_grids_2d.seven_fractures_one_L_intersection(
        # mesh_args)
        pp.contact_conditions.set_projections(self.mdg)

# params={}
# model_test=ChangedGrid(params)
# pp.run_stationary_model(model_test, params)
# pp.plot_grid(model_test.mdg)
#print(model_test.create_grid)
#print(model_test)

class ChangedPermeabilityAndSource(ChangedGrid):
    """A ContactMechanics model with modified grid and
    changed permeability."""


    def _bc_values(self, sd: pp.Grid) -> np.ndarray:
        """Set homogeneous conditions on all boundary faces."""
        # Values for all Nd components, face-wise
        values = np.zeros((self.nd, sd.num_faces))
        # Reshape according to PorePy convention
        values = values.ravel("F")
        values[0:9]=1
        return values

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

    def _friction_coefficient(self, sd: pp.Grid) -> np.ndarray:
        """Friction coefficient parameter.

        The friction coefficient is uniform and equal to 1.
        Parameters
        ----------
        sd : pp.Grid
            Fracture subdomain grid.

        Returns
        -------
        np.ndarray
            Cell-wise friction coefficient.

        """
        return np.ones(sd.num_cells)


# params={"max_iterations": 1,
        #"nl_convergence_tol": 1e-10,
        #"nl_divergence_tol": 1e5,
        #}
params = {}
model = ChangedPermeabilityAndSource()
pp.run_stationary_model(model, params)
pp.plot_grid(model.mdg, None, model.displacement_variable)
#inds = model.dof_manager.dof_var(var=[model.displacement_variable])
#vals = model.dof_manager.assemble_variable(variables = [model.displacement_variable])
#displacement = vals[inds]
#print(displacement)
#print(model.convergence_status)
#print(model.mortar_displacement_variable)
#pp.plot_grid(model.mdg, displacement, figsize=[10, 7])