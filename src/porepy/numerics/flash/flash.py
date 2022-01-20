from typing import Callable, Dict, List, Optional, Tuple, Literal

import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sps
from abc import ABC, abstractmethod

import porepy as pp


class KValueFlash(ABC):
    """Flash calculations based on known K-values and overall composition."""

    @abstractmethod
    def equilibrium_saturations(
        self, K: np.ndarray, z: np.ndarray, base_phase_order: List[int], **kwargs
    ) -> np.ndarray:
        """Find the equilibrium saturations for a system with known K-values and
        overall composition.
        """
        pass

    @abstractmethod
    def composition(
        self,
        K: np.ndarray,
        z: np.ndarray,
        saturation: np.ndarray,
        base_phase: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the distribution of components among phases given K-values, overall
        compositions and saturations.
        """
        pass


class EOSFlash(ABC):
    """Flash calculations based on equation of state"""

    pass


class TwoPhaseFlash(KValueFlash):
    """Two-phase flash calculation based on solving the Rashford-Rice equation."""

    def __init__(self, params: Optional[Dict] = None) -> None:
        if params is None:
            params = {}
        self._tol = params.get("tol", 1e-10)
        self._max_iter = params.get("max_iter", 30)
        self._solver = params.get("solver", "newton")

    def equilibrium_saturations(
        self,
        K: np.ndarray,
        z: np.ndarray,
        base_phase: Literal[0, 1],
    ) -> np.ndarray:
        """Compute equilibrium saturations for a two-phase system. Multiple cells
        can be solved simultaneously using vectorized computations.

        The equilibrium is found by solving the Rachford-Rice equation. For the
        time being, a standard Newton method is applied - more advanced method may
        be added as needed. For binary mixtures, an analytical expression is used.

        The K-values should be interpreted as the fraction of a component in a phase
        relative to that in the base phase. This perhaps unusual construction is used
        to be compatible with multistage approaches to multiphase flash calculations.
        For component i with volume phase fractions x_ij, we have

            K_{i, j} = x_{i,j} / x_{i, base_phase}.

        Thus, for the base phase all K-values (e.g. all values in K[:, base_phase, :])
        should be unity.

        The function has only been tested for mildly varying K-values. For extreme
        values, rounding errors may reduce the accuracy of the computations.

        Parameters:
            K (np.ndarray): K-values for the components. Size should be
                num_components x num_phases x num_cells, with unit K-values provided
                for the base phase.
            z (np.ndarray): Overall composition. Size should be
                num_components x num_cells, so that z.sum(axis=0) give a vector of
                ones.
            base_phase (Literal 0 or 1). Which phase to take as the base phase. Should
                correspond to an array of ones in corresponding K-values.

        Returns:
            np.ndarray, size num_phases x num_cells. Saturation of all phases in all
            cells.

            The composition corresponding to the calculated saturations can be obtained
            from the funciton composition().

        """

        num_components, num_phases, num_cells = K.shape

        if num_phases == 1:
            # Unit K-values for the base phase are likely not provided
            raise ValueError("K-values should be provided for all phases")
        elif num_phases > 2:
            raise ValueError("Use multiphase flash instead")

        # Define the base and other phase. The Rachford-Rice equation is formulated
        # so that the K-values gives the fraction of the composition of the other
        # face divided by that of the base phase. The equation is then solved for the
        # saturation of the other phase.
        other_phase = 0 if base_phase == 1 else 1

        K_frac = K[:, other_phase, :] / K[:, base_phase, :]

        def fun(s, active=None):
            if active is None:
                active = np.ones(s.size, dtype=bool)

            use_ad = False
            if isinstance(s, pp.ad.Ad_array):
                s = s.val
                use_ad = True

            val = sum(
                [
                    z[i, active]
                    * (K_frac[i, active] - 1)
                    * ((1 + (K_frac[i, active] - 1) * s) ** -1)
                    for i in range(num_components)
                ]
            )
            if use_ad:
                jac = sum(
                    [
                        -z[i, active]
                        * (K_frac[i, active] - 1) ** 2
                        * ((1 + (K_frac[i, active] - 1) * s) ** -1) ** 2
                        for i in range(num_components)
                    ]
                )
                return pp.ad.Ad_array(val, sps.diags(jac).tocsc())

            else:
                return val

        # In special cases (e.g. only one component present), the bounds may be
        # non-physical, in the sense that lower > upper. This corresponds to a
        # single-phase situation, and will be picked up below.
        lower, upper = self._bounds(K_frac, z)
        resid_lower = fun(lower)
        resid_upper = fun(upper)

        # Function is known to be monotonous in the full interval. If the function
        # has the same value in upper and lower bound, this is really a single phase
        # situation. The same applies if the bonds are in reverse order, which
        # corresponds to an empty interval of permissible solutions.
        non_empty_interval = upper > lower
        crosses_in_interval = np.logical_and(resid_lower > 0, resid_upper < 0)

        # Boolean array of whether cells are in a two-phase situation.
        two_phase = np.logical_and(non_empty_interval, crosses_in_interval)

        # Cases with non-empty intervals that is either fully below or above 0.
        # Test on whether it crosses may be superfluous here.
        all_below = np.logical_and.reduce(
            (non_empty_interval, np.logical_not(crosses_in_interval), resid_lower < 0)
        )
        all_above = np.logical_and.reduce(
            (non_empty_interval, np.logical_not(crosses_in_interval), resid_lower > 0)
        )
        # Cases of empty intervals. What to do here is not entirely clear, except that
        # there is no two-phase configuration which gives physically meaningful values
        # for the phase component fractions. The below logic is based on reasoning that
        # there are no crossing, so we will simply go with the sign of the residual
        # function at a saturation of 0.5.
        # resid_05 = fun(0.5 * np.ones(num_cells))
        # empty_above = np.logical_and(np.logical_not(non_empty_interval), resid_05 > 0)
        # empty_below = np.logical_and(np.logical_not(non_empty_interval), resid_05 < 0)

        # For the cells that have a real two-phase situation, the Rachford-Rice equation
        # is solved to get the saturation of the other phase, while that of the base
        # phase can be computed from the sum of phases.
        x_two_phase = np.zeros((2, two_phase.sum()))
        if num_components == 2:
            # In this case, an analytical solution is available
            # The saturation found here is that not of the base phase.
            sat_other = np.sum(
                z[:, two_phase] - K_frac[:, two_phase] * z[:, two_phase], axis=0
            ) / (
                np.sum(z[:, two_phase], axis=0)
                * (K_frac[0, two_phase] - 1)
                * (K_frac[1, two_phase] - 1)
            )
            x_two_phase[other_phase] = sat_other
            # Recover base phase saturation
            x_two_phase[base_phase] = 1 - sat_other

        else:
            # Use a Newton solver to find the zero.
            # FIXME: A combined Newton-bisection method may be more robust here.
            x0 = 0.5 * np.ones(two_phase.sum())
            x_two_phase[other_phase] = pp.nonlinear.scalar_newton(
                x0, fun, self._tol, self._max_iter
            )
            x_two_phase[base_phase] = 1 - x_two_phase[other_phase]

        # Variable for the full solution. To be filled up with saturation values for
        # the different cases.
        x = np.zeros((2, num_cells))

        # Calculated two-phase saturations.
        x[:, two_phase] = x_two_phase
        #        if not np.all(two_phase):
        #            breakpoint()
        # If the residual function is positive in the permissible interval, the
        # other phase is not present, thus base_phase saturation is unity.
        #        x[base_phase, all_above] = 1
        #        # The oposite case for negative residual function.
        #        x[other_phase, all_below] = 1

        not_twophase = np.logical_not(two_phase)
        x_not_twophase_0 = np.zeros((2, not_twophase.sum()))
        x_not_twophase_0[0] = 1
        x_not_twophase_1 = np.zeros_like(x_not_twophase_0)
        x_not_twophase_1[1] = 1

        x_0, y_0 = self.composition(
            K[:, :, not_twophase], z[:, not_twophase], x_not_twophase_0, base_phase
        )
        x_1, y_1 = self.composition(
            K[:, :, not_twophase], z[:, not_twophase], x_not_twophase_1, base_phase
        )

        sum_0 = np.sum(
            K[:, 0, not_twophase] * x_0 + K[:, 1, not_twophase] * y_0, axis=0
        )
        sum_1 = np.sum(
            K[:, 0, not_twophase] * x_1 + K[:, 1, not_twophase] * y_1, axis=0
        )

        not_twophase_ind = np.where(not_twophase)[0]
        pick_0 = sum_0 < sum_1

        index_0 = np.zeros_like(not_twophase)
        index_1 = np.zeros_like(index_0)
        index_0[not_twophase_ind[pick_0]] = True
        index_1[not_twophase_ind[np.logical_not(pick_0)]] = True

        x[0, index_0] = 1
        x[1, index_1] = 1

        return x

    def composition(
        self, K: np.ndarray, z: np.ndarray, sat: np.ndarray, base_phase: Literal[0, 1]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The composition can be recoved using the K-values, so that for component i

            x_{i, base} = z_i / (S_{base} + K_{i, other} * S_{other})
            x_{i, other} = x_{i, base} * K_{i, other}
                         = z_i * K_{i, other} / (S_{base} + K_{i, other} * S_{other})

        where S denotes saturation and {base, other} gives the index of the base phase
        and the other phase.

        """
        # Get phase composition from calculated saturations

        other_phase = 0 if base_phase == 1 else 1

        x_base = z / (1 + sat[other_phase] * (K[:, other_phase] - 1))
        x_other = K[:, other_phase] * x_base

        if base_phase == 0:
            return x_base, x_other
        else:
            return x_other, x_base

    def _bounds(self, K: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Simplified version of the multiphase case
        # Values from Michelsen and Mollerup (2004 textbook), page 221.
        # Correction: Values from Orr (2007, textbook), Eq 3.4.10

        upper = (K * z - 1) / (K - 1)
        lower = (1 - z) / (1 - K)

        upper[K < 1] = -np.inf
        lower[K > 1] = np.inf
        upper = 1 / (1 - K.min(axis=0)) - 1e-6
        lower = 1 / (1 - K.max(axis=0)) + 1e-6
        return lower, upper
        upper_bound = upper.max(axis=0)
        lower_bound = lower.min(axis=0)

        return lower_bound, upper_bound


class MultiphaseFlash(KValueFlash):
    """Class for implementation of multphase flash computations.

    For the moment, the number of phases is assumed to be 3, however, extension
    should be relatively straightforward.
    """

    def __init__(self, params: Optional[Dict] = None):
        self._params = params

        self._num_phases = 3

    def equilibrium_saturations(
        self,
        K: np.ndarray,
        z: np.ndarray,
        base_phase_order: np.ndarray,
        active_cells: Optional[np.ndarray] = None,
        domain_map: Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """Do a multistage negative flash for multiphase systems.

        The formulation is based on Iranshar et al 2010.

        Parameters:
            K (np.ndarray): K-values for the composition.
                Size: num_components x num_phases x num_cells.
            z (np.ndarray): Overall composition. Size: num_components x num_cells.
            derivative (bool): Formulate residual function in terms of AD variables.
                Use this if the target non-linear solver requries derivatives.

        Returns:
            np.ndarray, size: num_phases x num_cells: Saturations.

        """
        # TODO: Need to somehow mark the base phase for defining K-values

        num_comp, num_phases, num_cells = K.shape

        if active_cells is None:
            active_cells = np.ones(num_cells, dtype=bool)

        if domain_map is None:
            # Should loop here be over active cells?
            domain_map = {
                i: MultiphaseFlash.domain_vertexes([K[:, base_phase_order[1:], i]])[0]
                for i in range(num_cells)
            }

        resids = []
        for phase_ind in range(num_phases):
            resids.append(self._resid_factory(K, z, phase_ind, active_cells))

        s0, s1 = self._nested_bisection(resids, K, z, tol=1e-10)

        s2 = 1 - (s0 + s1)

        s_tmp = np.vstack((s0, s1, s2))

        s = np.zeros_like(s_tmp)

        # Need thresholding here
        num_phases = np.sum(s_tmp > 0, axis=0)
        s[:, num_phases == 3] = s_tmp[:, num_phases == 3]

        two_phase = TwoPhaseFlash(self._params)

        # Clear two-phase situations
        for phase_pair in [(0, 1), (0, 2), (1, 2)]:
            local_cells = np.logical_and(
                num_phases == 2, np.all(s_tmp[phase_pair] > 0, axis=0)
            )
            if local_cells.size == 0:
                continue

            if (
                np.where(base_phase_order == phase_pair[0]).ravel()
                < np.where(base_phase_order == phase_pair[1]).ravel()
            ):
                local_base_phase, other_phase = phase_pair
            else:
                local_base_phase, other_phase = phase_pair[::-1]

            K_loc = K[:, other_phase, local_cells] / K[:, local_base_phase, local_cells]

            two_phase = TwoPhaseFlash()

            s_other = two_phase.equilibrium_saturations(K_loc, z[:, local_cells])

            s[other_phase, local_cells] = s_other
            s[local_base_phase, local_cells] = 1 - s_other

        for pi in range(3):
            # Not sure what to do in single phase situations, but we probably need to do
            # two-phase with the two other phases
            local_cells = np.logical_and(num_phases == 1, s_tmp[pi] == 1)
            if local_cells.size == 0:
                continue

            s[pi, local_cells] = 1

            other_phase = np.setdiff1d(np.arange(3), pi)
            for oi in other_phase:
                K_loc = K[:, other_phase, local_cells] / K[:, pi, local_cells]

                s_other = two_phase.equilibrium_saturations(K_loc, z[:, local_cells])

                two_phase = s_other > 0
                assert np.allclose(s[local_cells[two_phase]], 0)

                s[oi, local_cells[two_phase]] = s_other[two_phase]
                s[oi, local_cells[two_phase]] = 1 - s_other[two_phase]

    def composition(self, K, z, saturations, base_phase_order):
        pass

    def _nested_bisection(self, resids: List[Callable], K: np.ndarray, z, tol: float):
        # Nested bisection solver for a three-phase system (two independent saturations)
        # Extension to more phases should be doable, but is not trivial

        counter = 0
        max_iter = 100

        # Lower, upper, middle value of the target phase
        lower, upper = self._domain
        f1_lower = resids[1]([lower[:, i] for i in range(lower.shape[1])])
        f1_upper = resids[1]([upper[:, i] for i in range(upper.shape[1])])
        s1_low, s1_up = lower[:, 1], upper[:, 1]

        while np.amax(f1_upper - f1_lower) > tol:

            s1_mid = 0.5 * (s1_low + s1_up)

            bounds_lower_0 = self._bounds(
                fixed_phase=1, target_phase=0, sat=s1_low, K=K
            )
            bounds_mid_0 = self._bounds(fixed_phase=1, target_phase=0, sat=s1_mid, K=K)

            f0_lower = self._resid_factory(
                K, z, 0, np.arange(K.shape[2]), fixed_sat=s1_low
            )
            f0_mid = self._resid_factory(
                K, z, 0, np.arange(K.shape[2]), fixed_sat=s1_mid
            )

            # Is there any reason not to use Newton here?
            s0_lower = pp.nonlinear.scalar_bisection(
                f0_lower, *bounds_lower_0, tol=1e-10
            )
            s0_mid = pp.nonlinear.scalar_bisection(f0_mid, *bounds_mid_0, tol=1e-10)

            f1_mid = resids[1]([s0_lower, s1_low])
            f1_mid = resids[1]([s0_mid, s1_mid])

            mid_same = np.sign(f1_lower * f1_mid) > 0
            mid_oposite = np.logical_not(mid_same)

            s1_low[mid_same] = s1_mid[mid_same]
            s1_up[mid_oposite] = s1_mid[mid_oposite]

            counter += 1
            if counter == max_iter:
                raise ValueError("Maximum number of iterations reached")

        return s0_mid, s1_mid

    def _bounds(
        self, fixed_phase: int, target_phase: int, sat: np.ndarray, K: np.ndarray
    ):
        # target_phase is the phase for which we need bounds
        # fixed_phase is the phase with a fixed value. sat is associated with fixed_phase

        K_target_less = np.ma.masked_less(K[:, target_phase], 1)
        K_target_greater = np.ma.masked_greater(K[:, target_phase], 1)

        K_fixed_less = np.ma.masked_array(K[:, fixed_phase], mask=K_target_less.mask)
        K_fixed_greater = np.ma.masked_array(
            K[:, fixed_phase], mask=K_target_greater.mask
        )

        # Not sure about axis argument here
        lower = np.min(
            (sat * (1 - K_fixed_greater) - 1) / (K_target_greater - 1), axis=2
        )
        upper = np.max((sat * (1 - K_fixed_less) - 1) / (K_target_less - 1), axis=2)

        return lower, upper

    def _resid_factory(
        self,
        K,
        z,
        pi: int,
        cells: np.ndarray,
        fixed_sat: Optional[List[np.ndarray]] = None,
    ):
        if fixed_sat is None:
            fixed_sat = []

        def func(sat):
            f = 0
            for comp in range(z.shape[0]):
                # Assume that the fixed saturations have the highest indexes
                for si, s in enumerate(sat + fixed_sat):
                    m_i = 1 + s * (K[comp, si, cells] - 1)
                f += z[comp, cells] * (1 - K[comp, pi, cells]) / m_i
            return f

        return func

    @staticmethod
    def domain_vertexes(K: List[np.ndarray]) -> List[np.ndarray]:
        # Find the max and min permissible values of a specific phase, given K-values
        num_cases = len(K)
        num_comp, num_phases = K[0].shape

        vertex_list: List[np.ndarray] = []

        for i in range(num_cases):

            # Use half space intersection method from scipy.
            # This first requires an interior point, which we find by linear programming

            A = 1 - K[i]
            b = -np.ones(num_comp)
            vertex_list.append(pp.half_space.vertexes_of_convex_domain(A, b).T)

        return vertex_list
