"""Module contains tests of exporter functions on a mesh including a polyline well.

So far the test functions check only that the methods run without any errors.
"""
import unittest
import warnings

import numpy as np

import src.porepy.fracs.wells_3d as wells_3d
import src.porepy.viz.exporter as exporter
from src.porepy.geometry.bounding_box import from_points
from src.porepy.fracs.meshing import subdomains_to_mdg
from src.porepy.models.incompressible_flow_model import IncompressibleFlow
from src.porepy.grids.structured import CartGrid
from src.porepy.grids.simplex import TetrahedralGrid

FOLDER_NAME: str = '/home/peter/PhD/test_folders/'


class PolylineWellGrid(IncompressibleFlow):

    def create_grid(self) -> None:
        # Create mdg without the well
        cell_dims = np.array([5, 5, 5])
        phys_dims = np.array([10, 10, 10])
        g = CartGrid(cell_dims, phys_dims)
        g = TetrahedralGrid(g.nodes)
        g.compute_geometry()
        self.box = from_points(np.array([[0, 0, 0], [10, 10, 10]]).T)
        self.mdg = subdomains_to_mdg([[g]])
        # Create the well and the well network
        well = wells_3d.Well(np.array([[5, 4, 6, 5], [5, 5, 5, 5], [0, 3, 6, 10]]))
        domain = {'xmin': 0, 'xmax': phys_dims[0], 'ymin': 0, 'ymax': phys_dims[1], 'zmin': 0, 'zmax': phys_dims[2]}
        mesh_args = {'mesh_size': 0.5}
        well_net = wells_3d.WellNetwork3d([well], domain, parameters=mesh_args)
        # Combine the mdg and the well network
        well_net.mesh(self.mdg)
        wells_3d.compute_well_rock_matrix_intersections(self.mdg)


class ExporterWellTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model = PolylineWellGrid(params={'folder_name': FOLDER_NAME + '/exporter_test/', 'file_name': 'polyline_well'})
        self.model.create_grid()

    def test_exporter_init(self) -> None:
        self.model.exporter = exporter.Exporter(
            self.model.mdg,
            self.model.params['file_name'],
            folder_name=self.model.params['folder_name']
            )

    def test_write_vtu(self) -> None:
        self.model.exporter = exporter.Exporter(
            self.model.mdg,
            self.model.params['file_name'],
            folder_name=self.model.params['folder_name']
            )
        self.model.exporter.write_vtu([])

if __name__ == '__main__':
    unittest.main()
