import numpy as np
import porepy as pp
import scipy.optimize as opt

from typing import Union


class MandelUtilities:

    def __init__(self,
                 model_parameters: dict,
                 mixed_dimensional_grid: pp.MixedDimensionalGrid
                 ):
        """
        Constructor for the MandelUtilities class.

        Args:
            model_parameters: input parameters for the model

        Mandatory model parameters are:
            mu_lame (float)
            lambda_lame (float)
            permeability (float)
            alpha_biot (float)
            viscosity (float)
            storativity (float)
            applied_load (float)
            height (float)
            width (float)

        """

        self.params: dict = model_parameters
        self.sd: pp.Grid = mixed_dimensional_grid.subdomains()[0]
        self.data: dict = mixed_dimensional_grid.subdomain_data(self.sd)

    # -----------> Physical properties
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

    # ----------> Analytical solutions
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

    def exact_p0(self) -> np.ndarray:
        """Exact initial pressure distribution

        Returns:
              p0 (sd.num_cells, ): Exact initial pressure.

        """

        # Retrieve physical data
        F = self.params["applied_load"]
        B = self.skempton_coefficient()
        nu_u = self.undrained_poisson_coefficient()

        # Retrieve geometrical data
        a = self.params["width"]

        p0 = F * B * (1 + nu_u) / (3 * a)

        return p0 * np.ones(self.sd.num_cells)

    def exact_u0(self) -> np.ndarray:
        """Exact initial displacement

        Returns:
            u0 (sd.dim * sd.num_cells, ): Exact initial displacement.

        """

        # Retrieve physical data
        F = self.params["applied_load"]
        mu_s = self.params["lame_mu"]
        nu_u = self.undrained_poisson_coefficient()

        # Retrieve geometrical data
        a = self.params["width"]
        b = self.params["height"]

        u0x = (F * nu_u) / (2 * mu_s)
        u0y = (-F * b * (1 - nu_u)) / (2 * mu_s * a)
        u0 = np.array((u0x, u0y)).ravel("F")

        return u0

    def exact_p(self, t: Union[float, int]) -> np.ndarray:
        """
        Exact pressure solution for a given time `t`.

        Args:
            t: Time in seconds.

        Returns:
            p (sd.num_cells, ): Exact pressure solution.

        """

        # Retrieve physical data
        F = self.params["applied_load"]
        B = self.skempton_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a = self.params["width"]
        xc = self.sd.cell_centers[0]

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

        # Retrieve physical data
        F = self.params["applied_load"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        mu_s = self.params["mu_lame"]
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a = self.params["width"]
        xc = self.sd.cell_centers[0]
        yc = self.sd.cell_centers[1]

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

        # Retrieve physical data
        F = self.params["applied_load"]
        nu_s = self.poisson_coefficient()
        nu_u = self.undrained_poisson_coefficient()
        mu_s = self.params["mu_lame"]
        c_f = self.fluid_diffusivity()

        # Retrieve geometrical data
        a = self.params["width"]
        b = self.params["height"]
        yf = self.sd.face_centers[1]
        b_faces = self.sd.tags["domain_boundary_faces"].nonzero()[0]
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
