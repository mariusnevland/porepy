import numpy as np
import porepy as pp
import pytest

# Precomputed values, obtained with chemicals (pip install chemicals, v1.0.12), using
# the function chemicals.rachford_rice.Rachford_Rice_solution().
# One of the cases corresponds to a negative saturation (thus negative flash)
@pytest.mark.parametrize(
    "K,z,known_sat,known_x0,known_x1",
    [
        (
            # Binary mixture, simple.
            np.array([100, 0.1]),
            np.array([0.3, 0.7]),
            # EK: Comment to self, if we ever need to go back to recalculating the
            # values form the chemicals package: The return values from chemicals are
            # 1) The saturation of the vapour phase. This is placed in the first
            #    column here, corresponding to using the liquid phase as base phase,
            #    and taking this as phase 1.
            # 2) Further output from chemicals are the liquid and vapour molar fractions
            #    (in that order). In the known values below, there correspond respectively
            #    to x0 and x1 (since the liquid phase is the base, and has index 1).
            np.array([[0.32626262626262625], [0.6737373737373737]]),
            np.array([[0.9009009009009007], [0.0990990990990991]]),
            np.array([[0.009009009009009007], [0.990990990990991]]),
        ),
        (
            # Binary mixture, will trigger negative saturation value
            np.array([2, 0.1]),
            np.array([0.3, 0.7]),
            np.array(
                [
                    [-0.36666666666666664],
                    [1.3666666666666667],
                ],
            ),
            np.array([[0.9473684175007482], [0.052631579031937076]]),
            np.array([[0.4736842087503741], [0.5263157903193707]]),
        ),
        (
            # Ternary mixture, all numbers positive
            np.array([2, 0.1, 3]),
            np.array([0.3, 0.4, 0.3]),
            np.array([[0.3886230119517894], [0.6113769880482106]]),
            np.array(
                [[0.4320827141966095], [0.061515815272472255], [0.5064014705309183]]
            ),
            np.array(
                [[0.21604135709830474], [0.6151581527247225], [0.1688004901769728]]
            ),
        ),
        (
            # Formally ternary mixture, but one component missing.
            np.array([2, 0.1, 3]),
            np.array([0.3, 0.7, 0]),
            np.array(
                [
                    [-0.36666666666666664],
                    [1.3666666666666667],
                ],
            ),
            np.array([[0.9473684175007482], [0.052631579031937076], [0.0]]),
            np.array([[0.4736842087503741], [0.5263157903193707], [0.0]]),
        ),
        (
            # Ternary mixture, will result in negative value
            np.array([0.5, 0.1, 2]),
            np.array([0.3, 0.4, 0.3]),
            np.array([[-0.282902201930396], [1.282902201930396]]),
            np.array(
                [[0.13141167402892837], [0.03188236728347594], [0.8367059578416971]]
            ),
            np.array(
                [[0.26282334805785673], [0.3188236728347594], [0.41835297892084855]]
            ),
        ),
    ],
)
def test_twophase_binary_mixture(K, z, known_sat, known_x0, known_x1):
    two_phase = pp.TwoPhaseFlash()

    num_components = z.size
    K_base = np.zeros((z.size, 2))
    for row in range(num_components):
        K_base[row, 0] = K[row]
    K_base[:, 1] = 1

    z_base = z.reshape((-1, 1))

    K1 = np.atleast_3d(K_base)
    z1 = z_base
    sat_1 = two_phase.equilibrium_saturations(K1, z1, base_phase=1)

    assert np.allclose(sat_1, known_sat)

    x0, x1 = two_phase.composition(K1, z1, sat_1, base_phase=1)
    assert np.allclose(x0, known_x0)
    assert np.allclose(x1, known_x1)

    num_cells = 2
    K2 = np.zeros((num_components, 2, num_cells))
    for i in range(num_cells):
        K2[:, :, i] = K_base

    z2 = np.tile(z_base, (1, num_cells))
    sat_2 = two_phase.equilibrium_saturations(K2, z2, base_phase=1)
    assert np.allclose(sat_2, np.hstack([known_sat, known_sat]))

    # Flip the role of the base and other phase. This should switch saturations 0 and 1,
    # thus the known values. Similarly, component molar fractions for the two phases
    # should flip.

    # Switch places of base and other phase in the K-values
    K3 = K1[:, ::-1, :]
    # The composition is the same
    z3 = z1
    # Find equilibrium saturation with phase 0 as base phase
    sat_3 = two_phase.equilibrium_saturations(K3, z3, base_phase=0)
    # The saturation should be flipped.
    assert np.allclose(sat_3, known_sat[::-1])

    # NOTE: Order of return argumens are flipped, compensating for the new base_phase
    x1, x0 = two_phase.composition(K3, z3, sat_3, base_phase=0)
    assert np.allclose(x0, known_x0)
    assert np.allclose(x1, known_x1)


def test_twophase_binary_mixture_two_cells():
    # Use two cells with same composition but different K-values. The two cells
    # have saturations and compositions corresponding to the two cases tested
    # in the simple binary mixture case (above).

    K = np.zeros((2, 2, 2))
    K[0, 0, :] = [100, 2]
    K[1, 0, :] = [0.1, 0.1]
    K[:, 1, :] = 1

    z = np.array([[0.3, 0.3], [0.7, 0.7]])
    # Known saturation and componition values are taken from chemicals, see above
    # tests for details on how they were obtained.
    known_sat = np.array(
        [
            [0.32626262626262625, -0.36666666666666664],
            [0.6737373737373737, 1.3666666666666667],
        ]
    )
    known_x0 = (
        np.array(
            [
                [0.9009009009009007, 0.947368417500748],
                [0.0990990990990991, 0.052631579031937076],
            ]
        ),
    )
    known_x1 = (
        np.array(
            [
                [0.009009009009009007, 0.4736842087503741],
                [0.990990990990991, 0.5263157903193707],
            ]
        ),
    )

    two_phase = pp.TwoPhaseFlash()
    sat = two_phase.equilibrium_saturations(K, z, base_phase=1)
    assert np.allclose(sat, known_sat)

    x0, x1 = two_phase.composition(K, z, sat, base_phase=1)
    assert np.allclose(x0, known_x0)
    assert np.allclose(x1, known_x1)

    # Next, flip roles of base and other phase
    sat = two_phase.equilibrium_saturations(K[:, ::-1], z, base_phase=0)
    assert np.allclose(sat, known_sat[::-1])
    x1, x0 = two_phase.composition(K[:, ::-1], z, sat, base_phase=0)

    assert np.allclose(x0, known_x0)
    assert np.allclose(x1, known_x1)


def test_twophase_binary_mixture_mix_multiphase_and_singlephase():
    K = np.zeros((2, 2, 4))

    K[0, 0] = [100, 2, 100, 2]
    K[1, 0] = [0.1, 0.1, 0.1, 0.1]

    K[:, 1, :] = 1

    z = np.array([[0.3, 0.3, 1, 0], [0.7, 0.7, 0, 1]])

    known_sat = np.array(
        [
            [0.32626262626262625, -0.36666666666666664, 1, 0],
            [0.6737373737373737, 1.3666666666666667, 0, 1],
        ]
    )
    two_phase = pp.TwoPhaseFlash()

    sat = two_phase.equilibrium_saturations(K, z, base_phase=1)
    assert np.allclose(sat, known_sat)

    sat = two_phase.equilibrium_saturations(K[:, ::-1], z, base_phase=0)
    assert np.allclose(sat, known_sat[::-1])


@pytest.mark.parametrize(
    "Ky,Kz,z,sat_known,x_known,y_known,z_known",
    [
        (
            [2.0, 1.3, 0.3],
            [1.3, 2.2, 0.3],
            [0.3, 0.4, 0.3],
            [0.43243243243243223, -0.1548078690935835, 0.7223754366611512],
            [0.2825112107623318, 0.21973094170403581, 0.4977578475336324],
            [0.5650224215246636, 0.28565022421524655, 0.14932735426008972],
            [0.3672645739910314, 0.48340807174887884, 0.14932735426008972],
        )
    ],
)
def test_threephase_ternary_mixture(Ky, Kz, z, sat_known, x_known, y_known, z_known):

    num_components = len(Ky)
    num_phases = 3
    K = np.zeros((num_components, num_phases, 1))
    K[:, 1] = np.array(Ky).reshape((-1, 1))
    K[:, 2] = np.array(Kz).reshape((-1, 1))
    K[:, 0] = 1

    z = np.array(z).reshape((-1, 1))
    sat_known = np.array(sat_known).reshape((-1, 1))
    x_known = np.array(x_known).reshape((-1, 1))
    y_known = np.array(x_known).reshape((-1, 1))
    z_known = np.array(x_known).reshape((-1, 1))

    base_phase = 0
    other_phases = [1, 2]

    #    domain_map = {0: pp.MultiphaseFlash.domain_vertexes([K[:, other_phases, 0]])}
    flash = pp.MultiphaseFlash(params={})

    sat = flash.equilibrium_saturations(K, z, base_phase_order=np.array([0, 1, 2]))
    assert np.allclose(sat, sat_known)
