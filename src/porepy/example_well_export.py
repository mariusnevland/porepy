import numpy as np

import models.run_models
import viz.plot_grid
import fracs.fracture_network_3d as fracture_network_3d
import fracs.wells_3d as wells_3d
import viz.exporter as exporter
from geometry.bounding_box import from_points
from fracs.meshing import subdomains_to_mdg
from models.incompressible_flow_model import IncompressibleFlow
from grids.structured import CartGrid
from grids.simplex import TetrahedralGrid

FOLDER_NAME: str = '/home/peter/PhD/test_folders/'


class PolylineWellGrid(IncompressibleFlow):

    def create_grid(self) -> None:
        # Create mdg without the well
        cell_dims = np.array([5, 5, 5])
        phys_dims = np.array([10, 10, 10])
        g = CartGrid(cell_dims, phys_dims)
        g = TetrahedralGrid(g.nodes)
        g.compute_geometry()
        self.mdg = subdomains_to_mdg([[g]])
        self.box = from_points(np.array([[0, 0, 0], [10, 10, 10]]).T)
        # Create the well and the well network
        well = wells_3d.Well(np.array([[5, 4, 6, 5], [5, 5, 5, 5], [0, 3, 6, 10]]))
        domain = {'xmin': 0, 'xmax': phys_dims[0], 'ymin': 0, 'ymax': phys_dims[1], 'zmin': 0, 'zmax': phys_dims[2]}
        mesh_args = {'mesh_size': 0.5}
        well_net = wells_3d.WellNetwork3d([well], domain, parameters=mesh_args)
        # Combine the mdg and the well network
        well_net.mesh(self.mdg)
        wells_3d.compute_well_rock_matrix_intersections(self.mdg)
        print(self.mdg)

class PolylineWellFractureGrid(IncompressibleFlow):

    def create_grid(self) -> None:
        phys_dims = np.array([10, 10, 10])
        domain = {'xmin': 0, 'xmax': 10, 'ymin': 0, 'ymax': 10, 'zmin': 0, 'zmax': 10}
        network = fracture_network_3d.FractureNetwork3d([], domain=domain)
        mesh_args = {'mesh_size_frac': 0.5, 'mesh_size_min': 0.5, 'mesh_size_bound': 1.0}
        # Generate the mixed-dimensional mesh and the bounding box.
        self.mdg = network.mesh(mesh_args)
        self.box = from_points(np.array([[0, 0, 0], [10, 10, 10]]).T)
        # Well defintion.
        well = wells_3d.Well(np.array([[5, 4, 6, 5], [5, 5, 5, 5], [0, 3, 6, 10]]))
        domain = {'xmin': 0, 'xmax': phys_dims[0], 'ymin': 0, 'ymax': phys_dims[1], 'zmin': 0, 'zmax': phys_dims[2]}
        mesh_args = {'mesh_size': 0.5}
        well_net = wells_3d.WellNetwork3d([well], domain, parameters=mesh_args)
        # Combine the fracture network and the well network
        wells_3d.compute_well_fracture_intersections(well_net, network)
        well_net.mesh(self.mdg)
        print(self.mdg)
        # wells_3d.compute_well_rock_matrix_intersections(self.mdg)


# model = PolylineWellGrid(params={'folder_name': FOLDER_NAME + '/exporter_test/', 'file_name': 'polyline_well'})
# model.create_grid()
# model.exporter = exporter.Exporter(
#     model.mdg,
#     model.params['file_name'],
#     folder_name=model.params['folder_name']
#     )
# model.exporter.write_vtu([])

# model = PolylineWellFractureGrid(params={'folder_name': FOLDER_NAME + '/exporter_test/', 'file_name': 'polyline_well_fracture'})
# model.create_grid()
# model.exporter = exporter.Exporter(
#     model.mdg,
#     model.params['file_name'],
#     folder_name=model.params['folder_name']
#     )
# model.exporter.write_vtu([])

class ModelEquation(PolylineWellGrid):

    def _source(self, sd) -> np.ndarray:
        if sd.dim == self.mdg.dim_max():
            val = np.zeros(sd.num_cells)
        else:
            val = np.ones(sd.num_cells)
        return val


model = ModelEquation()
params = {}

models.run_models.run_stationary_model(model, params)
viz.plot_grid(model.mdg, model.variable, figsize=[10,7])