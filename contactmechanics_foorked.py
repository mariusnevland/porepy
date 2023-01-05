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
        mesh_args = self.params.get("mesh_args", {"mesh_size_frac": .5})
        xendp = np.array([.2, .8])
        yendp= np.array([.2, 1])
        self.mdg, self.box = pp.md_grids_2d.two_intersecting(mesh_args, xendp,yendp, simplex=True)
        # self.mdg, self.box = pp.md_grids_2d.seven_fractures_one_L_intersection(
        # mesh_args)
        pp.contact_conditions.set_projections(self.mdg)

params={}
model_test=ChangedGrid(params)
pp.run_stationary_model(model_test, params)
pp.plot_grid(model_test.mdg, info="f", alpha=0.75)

class ChangedPermeabilityAndSource(ChangedGrid):
    """A ContactMechanics model with modified grid and
    changed things."""


    def _bc_values(self, sd: pp.Grid) -> np.ndarray:
        """Set homogeneous conditions on all boundary faces."""
        # Values for all Nd components, face-wise
        values = np.zeros((self.nd, sd.num_faces))
        # Reshape according to PorePy convention
        values = values.ravel("F")
        #values[0:2]=1
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


params={"max_iterations": 10,
        "nl_convergence_tol": 1e-10,
        "nl_divergence_tol": 1e5,
        }
#params = {}
model = ChangedPermeabilityAndSource(params)
pp.run_stationary_model(model, params)
pp.plot_grid(model.mdg, vector_value=model.displacement_variable, info="c")
#inds = model.dof_manager.dof_var(var=[model.displacement_variable])
#vals = model.dof_manager.assemble_variable(variables=[model.displacement_variable])
#mortar_displacement = vals[inds]
#print(mortar_displacement)