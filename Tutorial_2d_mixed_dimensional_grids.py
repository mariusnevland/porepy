import numpy as np
import porepy as pp

"""HOW TO SET UP A SIMULATION IN POREPY"""

"""1. Make a new class which contains all of the information about the problem to be
solved. This class should inherit from one of the model classes in PorePy, and add
or modify certain methods in order to specify our specific problem setup. Usually
the least that must be done is to specify a grid. You could also specify boundary
conditions, add source terms etc."""

"""Example:"""


def create_grid(self) -> None:
    """Create a 2d mixed-dimensional grid. Put this method inside a class inheriting
    from one of the model classes.

    The method assigns the following attributes to self:
    mdg (pp.MixedDimensionalGrid): The produced mixed-dimensional grid.
    box (dict): The bounding box of the domain, defined through minimum and
        maximum values in each dimension.

    """

    """Create a grid with two intersecting fractures, using the two_intersecting function
    from porepy.grids.standard_grids.md_grids_2d

    two_intersecting(mesh_args,xendp,yendp,simplex)
    
    Args:
        mesh_args:  For triangular grids: Dictionary containing at least "mesh_size_frac". If
                        the optional values of "mesh_size_bound" and "mesh_size_min" are
                        not provided, these are set by utils.set_mesh_sizes.
                    For cartesian grids: List containing number of cells in x and y
                        direction.
        x_endpoints (list): containing the x coordinates of the two endpoints of the
            horizontal fracture. If not provided, the endpoints will be set to [0, 1].
        y_endpoints (list): Contains the y coordinates of the two endpoints of the
            vertical fracture. If not provided, the endpoints will be set to [0, 1].
        simplex (bool): Whether to use triangular or Cartesian 2d grid.

    Returns:
        Mixed-dimensional grid and domain.
    """

    # Why use the more complicated line for mesh_args below?
    # mesh_args = self.params.get("mesh_args", {"mesh_size_frac": .1})
    mesh_args = {"mesh_size_frac": .1}
    xendp = np.array([.2, .8])
    yendp = np.array([.2, 1])
    self.mdg, self.box = pp.md_grids_2d.two_intersecting(mesh_args, xendp, yendp,
                                                         simplex=True)
    pp.contact_conditions.set_projections(self.mdg)

