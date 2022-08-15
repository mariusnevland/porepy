import numpy as np
import porepy as pp
import matplotlib.pyplot as plt

from terzaghi_exact import ExactTerzaghi
from typing import Union, Optional
from time import time

Scalar = Union[int, float]


#%% Inherit model ContactMechanicsBiot class
class Terzaghi(pp.ContactMechanicsBiot):
    """Parent class for Terzaghi's consolidation problem model."""

    def __init__(self, params: dict):
        super().__init__(params)

        # Create a dictionary to store relevant data
        self.sol = {t_idx: {} for t_idx in self.params["time_index"]}

    def create_grid(self) -> None:
        """Create mixed dimensional grid"""
        if self.params["is_cartesian"]:
            phys_dims = np.array([1, 1])
            n_cells = np.array([10, 10])
            self.box = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
            sd: pp.Grid = pp.CartGrid(n_cells, phys_dims)
            sd.compute_geometry()
            self.mdg = pp.meshing.subdomains_to_mdg([[sd]])
        else:
            mesh_size = model.params.get("mesh_size", 0.1)
            self.box = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
            network_2d = pp.FractureNetwork2d(None, None, self.box)
            mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}
            self.mdg = network_2d.mesh(mesh_args)

    def _initial_condition(self) -> None:
        """Override initial condition for the flow subproblem"""
        super()._initial_condition()
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        initial_p = self.params["vertical_load"] * np.ones(sd.num_cells)
        data[pp.STATE][self.scalar_variable] = initial_p
        data[pp.STATE][pp.ITERATE][self.scalar_variable] = initial_p

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Define boundary condition types for the flow subproblem.

        Args:
            sd: Subdomain grid.

        Returns:
            Scalar boundary condition representation.

        """

        # Define boundary regions
        all_bc, _, _, north, *_ = self._domain_boundary_sides(sd)
        north_bc = np.isin(all_bc, np.where(north)).nonzero()

        # All sides Neumann, except the North which is Dirichlet
        bc_type = np.asarray(all_bc.size * ["neu"])
        bc_type[north_bc] = "dir"

        return pp.BoundaryCondition(sd, faces=all_bc, cond=bc_type)

    def _bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define boundary condition types for the mechanics subproblem

        Args:
            sd: Subdomain grid.

        Returns:
            Vectorial boundary condition representation.

        """
        # Inherit bc from parent class. This sets all bc faces as Dirichlet.
        super()._bc_type_mechanics(sd=sd)

        # Define boundary regions, retrieve data dict, and bc object
        _, east, west, north, south, *_ = self._domain_boundary_sides(sd)
        data = self.mdg.subdomain_data(sd)
        bc = data[pp.PARAMETERS][self.mechanics_parameter_key]["bc"]

        # East side: Roller
        bc.is_neu[1, east] = True
        bc.is_dir[1, east] = False

        # West side: Roller
        bc.is_neu[1, west] = True
        bc.is_dir[1, west] = False

        # North side: Neumann
        bc.is_neu[:, north] = True
        bc.is_dir[:, north] = False

        # South side: Roller
        bc.is_neu[0, south] = True
        bc.is_dir[0, south] = False

        return bc

    def _bc_values_mechanics(self, sd: pp.Grid) -> np.ndarray:
        """Set boundary condition values for the mechanics subproblem."""
        _, _, _, north, *_ = self._domain_boundary_sides(sd)
        bc_values = np.array([np.zeros(sd.num_faces), np.zeros(sd.num_faces)])
        bc_values[1, north] = -self.params["vertical_load"] * sd.face_areas[north]
        return bc_values.ravel("F")

    def _permeability(self, sd: pp.Grid) -> np.ndarray:
        """Set intrinsic permeability for the flow subproblem"""
        return 0.001 * np.ones(sd.num_cells)

    def _storativity(self, sd: pp.Grid) -> np.ndarray:
        """Zero storativity in Terzaghi's model"""
        return np.zeros(sd.num_cells)

    def _confined_compressibility(self, sd: pp.Grid) -> np.ndarray:
        """Confined compressibility.

        Units: 1 / Pa
        """
        stifness_tensor = self._stiffness_tensor(sd)
        return 1 / (2 * stifness_tensor.mu + stifness_tensor.lmbda)

    def _consolidation_coefficient(self, sd: pp.Grid) -> np.ndarray:
        """Consolidation coefficient.

        Units: Dimensionless.
        """
        permeability = self._permeability(sd)
        volumetric_weight = np.ones(sd.num_cells)
        viscosity = self._viscosity(sd)
        hydraulic_conductivity = (permeability * volumetric_weight) / viscosity
        storativity = self._storativity(sd)
        alpha_biot = self._biot_alpha(sd)
        confined_compressibility = self._confined_compressibility(sd)
        consolidation_coefficient = hydraulic_conductivity / (
            volumetric_weight
            * (storativity + alpha_biot**2 * confined_compressibility)
        )
        return consolidation_coefficient

    def _set_scalar_parameters(self) -> None:
        """Add confined compressibility and consolidation coefficient to parameters"""
        super()._set_scalar_parameters()
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        kw = self.scalar_parameter_key
        data[pp.PARAMETERS][kw][
            "confined_compressibility"
        ] = self._confined_compressibility(sd)
        data[pp.PARAMETERS][kw][
            "consolidation_coefficient"
        ] = self._consolidation_coefficient(sd)

    def dimensionless_time(self, sd: pp.Grid) -> float:
        """Computes dimensionless time."""
        data = self.mdg.subdomain_data(sd)
        kw = self.scalar_parameter_key
        consolidation_coefficient = np.mean(
            data[pp.PARAMETERS][kw]["consolidation_coefficient"]
        )
        height = self.box["ymax"]
        return (self.time * consolidation_coefficient) / (height**2)

    def dimensionless_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Computes dimensionless pressure."""
        data = self.mdg.subdomain_data(sd)
        pressure = data[pp.STATE][self.scalar_variable]
        initial_pressure = self.params["vertical_load"]
        return pressure / initial_pressure

    def after_newton_convergence(
        self,
        solution: np.ndarray,
        errors: float,
        iteration_counter: int,
    ) -> None:
        super().after_newton_convergence(solution, errors, iteration_counter)

        # Adjust time step
        self.time_step *= 1.2
        if self.time_step + self.time > self.end_time:
            self.time_step = self.end_time - self.time
        self._ad.time_step._value = self.time_step

        if self.time_index in self.params["time_index"]:
            # Store data
            sd = self.mdg.subdomains()[0]
            model.sol[self.time_index]["p"] = self.dimensionless_pressure(sd)
            model.sol[self.time_index]["t"] = self.time
            model.sol[self.time_index]["p_ex"] = self.exact_dimless_pressure(self.time)

    # ----------> Analytical expressions
    def exact_dimless_pressure(self, t: Scalar) -> np.ndarray:
        """
        Compute exact dimensionless pressure.

        Args:
            t: Time in seconds.

        Returns:
            dimless_p: Dimensionless pressure profile for the given time `t`.

        """
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        kw = self.scalar_parameter_key

        cc = sd.cell_centers[1]
        n = self.params["roof_series"]
        h = self.box["ymax"]
        vert_load = self.params["vertical_load"]
        c_v = np.mean(data[pp.PARAMETERS][kw]["consolidation_coefficient"])

        sum_series = np.zeros_like(cc)
        for i in range(1, n + 1):
            sum_series += (
                (((-1) ** (i - 1)) / (2 * i - 1))
                * np.cos((2 * i - 1) * (np.pi / 2) * (cc / h))
                * np.exp(
                    (-((2 * i - 1) ** 2)) * (np.pi**2 / 4) * (c_v * t) / (h**2)
                )
            )
        dimless_p = (4 / np.pi) * vert_load * sum_series

        return dimless_p


#%% Main script
model_params = {
    "use_ad": True,
    "is_cartesian": False,
    "mesh_size": 0.025,
    "time_step": 0.05,
    "end_time": 600,
    "schedule": [0.1, 0.25, 0.5, 1.0, 2.0, 10.0, 80.0, 150.0, 300.0, 600.0],
    "time_index": [2, 4, 7, 10, 15, 25, 30, 43],
    "vertical_load": 1.0,
    "file_name": "terzaghi",
    "folder_name": "out",
    "roof_series": 1000,
}
tic = time()
model = Terzaghi(model_params)
pp.run_time_dependent_model(model, model_params)
print(f"Simulation finished in {time() - tic} seconds.")

#%%
sd = model.mdg.subdomains()[0]
data = model.mdg.subdomain_data(sd)
yc = sd.cell_centers[1]
p = model.sol[2]["p"]
half_max_diam = np.max(sd.cell_diameters()) / 2
yc_eval = np.arange(0, 1.0, half_max_diam)
closest_cells = sd.closest_cell(np.array([0.5 * np.ones_like(yc_eval), yc_eval]))
_, idx = np.unique(closest_cells, return_index=True)
yc_plot = closest_cells[np.sort(idx)]

# Preparing for ploting
y_dimless = yc[yc_plot] / 1.0


#%%
fig = plt.figure(figsize=(8, 10))
for idx in model.params["time_index"]:
    plt.plot(model.sol[idx]["p"][yc_plot], y_dimless)
plt.show()


