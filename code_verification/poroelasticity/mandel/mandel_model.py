"""Implementation of Mandel's problem."""

import porepy as pp
import numpy as np
import scipy.optimize as opt

from time import time
from typing import Union


class Mandel(pp.ContactMechanicsBiot):
    """Parent class for Mandel's problem."""

    def __init__(self, params: dict):
        """Constructor of Mandel class.

        Args:
            params: Model parameters.

        Mandatory model parameters:
            mu_lame (float): First Lamé parameter [Pa]
            lambda_lame (float): Seconda Lamé parameter [Pa]
            permeability (float): Intrinsic permeability [m^2]
            alpha_biot (float): Biot-Willis coefficient [-]
            viscosity (float): Fluid dynamic viscosity [Pa s]
            storativity (float): Storativity or specific storage [1/Pa]
            applied_load (float): Vertically applied load [N/m]

        Optional model parameters:

        """
        super().__init__(params)

        self.tsc = self.params["time_stepping_object"]
        self.time = self.tsc.time_init
        self.end_time = self.tsc.time_final
        self.time_step = self.tsc.dt

        # Create a solution dictionary to store pressure and displacement solutions
        self.sol = {t: {} for t in self.tsc.schedule}

    def create_grid(self) -> None:
        """Create two-dimensional unstructured mixed-dimensional grid."""
        height = self.params["height"]
        width = self.params["width"]
        mesh_size = self.params["mesh_size"]
        self.box = {"xmin": 0.0, "xmax": width, "ymin": 0.0, "ymax": height}
        network_2d = pp.FractureNetwork2d(None, None, self.box)
        mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}
        self.mdg = network_2d.mesh(mesh_args)

    def _initial_condition(self) -> None:
        """Set up initial conditions.

        Note:
            Initial conditions are given by Eqs. (41) - (43) from 10.1007/s10596-013-9393-8

        """
        super()._initial_condition()
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)

        # Retrieve physical data
        F = self.params["applied_load"]
        B = self.skempton_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        mu_s = self.params["mu_lame"]

        # Retrieve geometrical data
        a = self.params["width"]
        b = self.params["height"]

        # Set initial pressure
        p0 = (F * B * (1 + nu_u) / (3 * a)) * np.ones(sd.num_cells)
        data[pp.STATE][self.scalar_variable] = p0
        data[pp.STATE][pp.ITERATE][self.scalar_variable] = p0

        # Set initial displacement
        u0x = ((F * nu_u) / (2 * mu_s)) * np.ones(sd.num_cells)
        u0y = ((-F * b * (1 - nu_u)) / (2 * mu_s * a)) * np.ones(sd.num_cells)
        u0 = np.array((u0x, u0y)).ravel("F")
        data[pp.STATE][self.displacement_variable] = u0
        data[pp.STATE][pp.ITERATE][self.displacement_variable] = u0

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Define boundary condition types for the flow subproblem.

        Args:
            sd: Subdomain grid.

        Returns:
            bc: Scalar boundary condition representation.

        """

        # Define boundary regions
        all_bc, east, *_ = self._domain_boundary_sides(sd)
        east_bc = np.isin(all_bc, np.where(east)).nonzero()

        # All sides Neumann, except the East side which is Dirichlet
        bc_type = np.asarray(all_bc.size * ["neu"])
        bc_type[east_bc] = "dir"

        bc = pp.BoundaryCondition(sd, faces=all_bc, cond=bc_type)

        return bc

    def _bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define boundary condition types for the mechanics subproblem

        Args:
            sd: Subdomain grid.

        Returns:
            bc: Vectorial boundary condition representation.

        """

        # Inherit bc from parent class. This sets all bc faces as Dirichlet.
        super()._bc_type_mechanics(sd=sd)

        # Get boundary sides, retrieve data dict, and bc object
        _, east, west, north, south, *_ = self._domain_boundary_sides(sd)
        data = self.mdg.subdomain_data(sd)
        bc = data[pp.PARAMETERS][self.mechanics_parameter_key]["bc"]

        # East side: Stress-free
        bc.is_neu[:, east] = True
        bc.is_dir[:, east] = False

        # West side: Roller
        bc.is_neu[1, west] = True
        bc.is_dir[1, west] = False

        # North side: Roller
        bc.is_neu[0, north] = True
        bc.is_dir[0, north] = False

        # South side: Roller
        bc.is_neu[0, south] = True
        bc.is_dir[0, south] = False

        return bc

    def _bc_values_mechanics(self, sd: pp.Grid) -> np.ndarray:
        """Set boundary condition values for the mechanics subproblem.

        Args:
            sd: Subdomain grid.

        Returns:
            bc_values (sd.dim * sd.num_faces): Containing the boundary condition values.

        """

        # Retrieve boundary sides
        _, _, _, north, *_ = self._domain_boundary_sides(sd)

        # All zeros except vertical component of the north side
        sd = self.mdg.subdomains()[0]

        # Retrieve physical data
        F = self.params["applied_load"]
        nu_u = self.undrained_poisson_coefficient()
        mu_s = self.params["mu_lame"]

        # Retrieve geometrical data
        a = self.params["width"]
        b = self.params["height"]

        u0y = ((-F * b * (1 - nu_u)) / (2 * mu_s * a))
        bc_values = np.array([np.zeros(sd.num_faces), np.zeros(sd.num_faces)])
        bc_values[1, north] = u0y
        bc_values = bc_values.ravel("F")

        return bc_values

    def before_newton_loop(self) -> None:
        """Update time for time-stepping technique and bc values."""
        super().before_newton_loop()

        # Update time for the time-stepping control routine
        self.tsc.time += self.time_step

        # Update value of boundary conditions
        self.update_north_bc_values(self.time)

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        super().after_newton_convergence(solution, errors, iteration_counter)

        # Adjust time step
        self.time_step = self.tsc.next_time_step(5, recompute_solution=False)
        self._ad.time_step._value = self.time_step

        # Store solutions
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        if self.time in self.tsc.schedule[1:]:
            self.sol[self.time]["u_num"] = data[pp.STATE][self.displacement_variable]
            self.sol[self.time]["p_num"] = data[pp.STATE][self.scalar_variable]

    # Physical parameters
    def _permeability(self, sd: pp.Grid) -> np.ndarray:
        """Override permeability value [m^2]

        Args:
            sd: Subdomain grid.

        Returns:
            Permeability.

        """
        return self.params["permeability"] * np.ones(sd.num_cells)

    def _stiffness_tensor(self, sd: pp.Grid) -> pp.FourthOrderTensor:
        """Override stifness tensor.

        Args:
            sd: Subdomain grid.

        Returns:
            Fourth order tensorial representation of the stiffness tensor.

        """
        lam = (self.params["lambda_lame"] * np.ones(sd.num_cells)) / self.scalar_scale
        mu = (self.params["mu_lame"] * np.ones(sd.num_cells)) / self.scalar_scale
        return pp.FourthOrderTensor(mu, lam)

    def _viscosity(self, sd: pp.Grid) -> np.ndarray:
        """Override fluid viscosity values [Pa.s]

        Args:
            sd: Subdomain grid.

        Returns:
            Viscosity
        """
        return self.params["viscosity"] * np.ones(sd.num_cells)

    def _storativity(self, sd: pp.Grid) -> np.ndarray:
        """Override storativity value of the porous medium [1/Pa]

        Args:
            sd: Subdomain grid.

        Retunrs:
            Storativity
        """
        return self.params["storativity"] * np.ones(sd.num_cells)

    def _biot_alpha(self, sd: pp.Grid) -> np.ndarray:
        """Override value of Biot-Willis coefficient.

        Args:
            sd: Subdomain grid.

        Returns:
            Biot's coefficient.
        """
        return self.params["alpha_biot"] * np.ones(sd.num_cells)

    # Other physical parameters used specifically for Mandel's problem
    def bulk_modulus(self) -> float:
        """Set bulk modulus [Pa].

        Returns:
            K_s: Bulk modulus.

        """
        mu_s = self.params["mu_lame"]
        lambda_s = self.params["lambda_lame"]
        K_s = (2 / 3) * mu_s + lambda_s

        return K_s

    def young_modulus(self) -> float:
        """Set Young modulus [Pa]

        Returns:
            E_s: Young modulus.

        """
        mu_s = self.params["mu_lame"]
        K_s = self.bulk_modulus()
        E_s = mu_s * ((9 * K_s) / (3 * K_s + mu_s))

        return E_s

    def poisson_coefficient(self) -> float:
        """Set Poisson coefficient [-]

        Returns:
            nu_s: Poisson coefficient.

        """
        mu_s = self.params["mu_lame"]
        K_s = self.bulk_modulus()
        nu_s = (3 * K_s - 2 * mu_s) / (2 * (3 * K_s + mu_s))

        return nu_s

    def undrained_bulk_modulus(self) -> float:
        """Set undrained bulk modulus [Pa]

        Returns:
            K_u: Undrained bulk modulus.

        """
        alpha_biot = self.params["alpha_biot"]
        K_s = self.bulk_modulus()
        S_m = self.params["storativity"]
        K_u = K_s + (alpha_biot ** 2) / S_m

        return K_u

    def skempton_coefficient(self) -> float:
        """Set Skempton's coefficient [-]

        Returns:
            B: Skempton's coefficent.

        """
        alpha_biot = self.params["alpha_biot"]
        K_u = self.undrained_bulk_modulus()
        S_m = self.params["storativity"]
        B = alpha_biot / (S_m * K_u)

        return B

    def undrained_poisson_coefficient(self) -> float:
        """Set Poisson coefficient under undrained conditions [-]

        Returns:
            nu_u: Undrained Poisson coefficient.

        """
        nu_s = self.poisson_coefficient()
        B = self.skempton_coefficient()
        nu_u = (3 * nu_s + B * (1 - 2 * nu_s)) / (3 - B * (1 - 2 * nu_s))

        return nu_u

    def fluid_diffusivity(self) -> float:
        """Set fluid diffusivity [m^2/s]

        Returns:
            c_f: Fluid diffusivity.

        """
        k_s = self.params["permeability"]
        B = self.skempton_coefficient()
        mu_s = self.params["mu_lame"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        mu_f = self.params["viscosity"]
        c_f = (
                (2 * k_s * (B ** 2) * mu_s * (1 - nu_s) * (1 + nu_u) ** 2) /
                (9 * mu_f * (1 - nu_u) * (nu_u - nu_s))
        )
        return c_f

    # Analytical solutions
    def approximate_roots(self) -> np.ndarray:
        """
        Approximate roots to f(x) = 0, where f(x) = tan(x) - ((1-nu)/(nu_u-nu)) x

        Note that we have to solve the above equation numerically to get all positive
        solutions to the equation. Later, we will use them to compute the infinite series
        associated with the exact solutions. Experience has shown that 200 roots are enough
        to achieve accurate results.

        Implementation note:
            We find the roots using the bisection method. Thanks to Manuel Borregales who
            helped with the implementation of this part of the code.

        Returns:
            a_n: approximated roots of f(x) = 0.

        """

        # Retrieve physical data
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()

        # Define algebraic function
        def f(x):
            y = np.tan(x) - ((1 - nu_s) / (nu_u - nu_s)) * x
            return y

        n_series = 200  # number of approximated roots
        a_n = np.zeros(n_series)  # initializing roots array
        x0 = 0  # initial point
        for i in range(n_series):
            a_n[i] = opt.bisect(
                f,  # function
                x0 + np.pi / 4,  # left point
                # x0 + np.pi / 2 - 10000000 * 2.2204e-16,  # right point
                x0 + np.pi / 2 - 1E-6,  # right point
                xtol=1e-8,  # absolute tolerance
                rtol=1e-3,
            )  # relative tolerance
            x0 += np.pi  # apply a phase change of pi to get the next root

        return a_n

    def exact_p(self, t: Union[float, int]) -> np.ndarray:
        """
        Exact pressure solution for a given time `t`.

        Args:
            t: Time in seconds.

        Returns:
            p (sd.num_cells, ): Exact pressure solution.

        """

        sd = self.mdg.subdomains()[0]

        # Retrieve physical data
        F = self.params["applied_load"]
        B = self.skempton_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a = self.params["width"]
        xc = sd.cell_centers[0]

        # Auxiliary constant terms
        aa_n = self.approximate_roots()[:, np.newaxis]
        p0 = (2 * F * B * (1 + nu_u)) / (3 * a)

        # Compute exact solution
        p_sum = np.sum(
            ((np.sin(aa_n)) / (aa_n - (np.sin(aa_n) * np.cos(aa_n))))
            * (np.cos((aa_n * xc) / a) - np.cos(aa_n))
            * np.exp((-(aa_n ** 2) * c_f * t) / (a ** 2)),
            axis=0,
        )

        p = p0 * p_sum

        return p

    def exact_u(self, t: Union[float, int]) -> np.ndarray:
        """
        Exact displacement for a given time `t`.

        Args:
            t: Time in seconds.

        Returns:
            u (sd.dim * sd.num_cells, ): Exact displacement.

        """

        sd = self.mdg.subdomains()[0]

        # Retrieve physical data
        F = self.params["applied_load"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        mu_s = self.params["mu_lame"]
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a = self.params["width"]
        xc = sd.cell_centers[0]
        yc = sd.cell_centers[1]

        # Auxiliary constant terms
        aa_n = self.approximate_roots()[:, np.newaxis]

        ux0_1 = (F * nu_s) / (2 * mu_s * a)
        ux0_2 = -((F * nu_u) / (mu_s * a))
        ux0_3 = F / mu_s

        uy0_1 = (-F * (1 - nu_s)) / (2 * mu_s * a)
        uy0_2 = F * (1 - nu_u) / (mu_s * a)

        # Compute exact solutions
        ux_sum1 = np.sum(
            (np.sin(aa_n) * np.cos(aa_n))
            / (aa_n - np.sin(aa_n) * np.cos(aa_n))
            * np.exp((-(aa_n ** 2) * c_f * t) / (a ** 2)),
            axis=0,
        )

        ux_sum2 = np.sum(
            (np.cos(aa_n) / (aa_n - (np.sin(aa_n) * np.cos(aa_n))))
            * np.sin((aa_n * xc) / a)
            * np.exp((-(aa_n ** 2) * c_f * t) / (a ** 2)),
            axis=0,
        )

        uy_sum = np.sum(
            ((np.sin(aa_n) * np.cos(aa_n)) / (aa_n - np.sin(aa_n) * np.cos(aa_n)))
            * np.exp((-(aa_n ** 2) * c_f * t) / (a ** 2)),
            axis=0,
        )

        ux = (ux0_1 + ux0_2 * ux_sum1) * xc + ux0_3 * ux_sum2
        uy = (uy0_1 + uy0_2 * uy_sum) * yc
        u = np.array((ux, uy)).ravel("F")

        return u

    def exact_top_bc(self, t: Union[float, int]) -> np.ndarray:
        """
        Exact top boundary condition (displacement) for a givent time `t`.

        Args:
            t: Time in seconds.

        Returns:
            top_bc (num_top_bc_faces, ): Exact vertical displacements at the top boundary.

        """

        sd = self.mdg.subdomains()[0]

        # Retrieve physical data
        F = self.params["applied_load"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        mu_s = self.params["mu_lame"]
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a = self.params["width"]
        b = self.params["height"]
        yf = sd.face_centers[1]
        b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
        y_max = b_faces[yf[b_faces] > 0.9999 * b]

        # Auxiliary constant terms
        aa_n = self.approximate_roots()[:, np.newaxis]

        uy0_1 = (-F * (1 - nu_s)) / (2 * mu_s * a)
        uy0_2 = F * (1 - nu_u) / (mu_s * a)

        # Compute exact solution
        uy_sum = np.sum(
            ((np.sin(aa_n) * np.cos(aa_n)) / (aa_n - np.sin(aa_n) * np.cos(aa_n)))
            * np.exp((-(aa_n ** 2) * c_f * t) / (a ** 2)),
            axis=0,
        )

        top_bc = (uy0_1 + uy0_2 * uy_sum) * yf[y_max]

        return top_bc

    def update_north_bc_values(self, t: Union[float, int]) -> None:
        """
        Updates boundary condition value at the north boundary of the domain.

        Args:
            t: Time in seconds.

        Note:
            The key `bc_values` from data[pp.PARAMETERS][self.mechanics_parameter_key]
            will be updated accordingly.

        """
        sd = self.mdg.subdomains()[0]

        # Retrieve physical data
        F = self.params["applied_load"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        mu_s = self.params["mu_lame"]
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a = self.params["width"]
        b = self.params["height"]
        yf = sd.face_centers[1]
        b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
        y_max = b_faces[yf[b_faces] > 0.9999 * b]

        # Auxiliary constant terms
        aa_n = self.approximate_roots()[:, np.newaxis]

        uy0_1 = (-F * (1 - nu_s)) / (2 * mu_s * a)
        uy0_2 = F * (1 - nu_u) / (mu_s * a)

        # Compute exact north boundary condition for the given time `t`
        uy_sum = np.sum(
            ((np.sin(aa_n) * np.cos(aa_n)) / (aa_n - np.sin(aa_n) * np.cos(aa_n)))
            * np.exp((-(aa_n ** 2) * c_f * t) / (a ** 2)),
            axis=0,
        )

        north_bc = (uy0_1 + uy0_2 * uy_sum) * yf[y_max]

        # Update values
        data = self.mdg.subdomain_data(sd)
        data[pp.PARAMETERS][self.mechanics_parameter_key]["bc_values"][1::2][y_max] = north_bc


#%% Running the model
# Create time-stepping object. In this case, the time-step is fixed.
tsc = pp.TimeSteppingControl(
    schedule=[
        0.0,
        10.0,
        50.0,
        100.0,
        1000.0,
        5000.0,
        8000.0,
        10000.0,
        20000.0,
        30000.0,
        50000.0,
    ],  # [s]
    dt_init=10.0,  # [s]
    dt_min_max=(10.0, 100.0)  # [s]
)

# Model parameters
model_params = {
    "use_ad": True,
    "mu_lame": 2.475E9,  # [Pa]
    "lambda_lame": 1.65E9,  # [Pa]
    "permeability": 9.8692E-14,  # [m^2]
    "alpha_biot": 1.0,  # [-]
    "viscosity": 1E-3,  # [Pa.s]
    "storativity": 6.0606E-11,  # [1/Pa]
    "applied_load": 6.08E8,  # [N/m]
    "height": 10.0,  # [m]
    "width": 100.0,  # [m]
    "mesh_size": 2.0,  # [m]
    "time_stepping_object": tsc,
}

# Run model
print("Simulation started...")
tic = time()
model = Mandel(model_params)
pp.run_time_dependent_model(model, model_params)
print(f"Simulation finished in {round(time() - tic, 1)} seconds.")

#%% Retrieve useful data
mdg = model.mdg
sd = mdg.subdomains()[0]
data = mdg.subdomain_data(sd)

# Uncomment to plot pressure
pp.plot_grid(sd, model.sol[50000.0]["p_num"], plot_2d=True, figsize=(100, 10))
pp.plot_grid(sd, model.sol[50000.0]["u_num"][::2], plot_2d=True, figsize=(100, 10))
pp.plot_grid(sd, model.sol[50000.0]["u_num"][1::2], plot_2d=True, figsize=(100, 10))

#%%

