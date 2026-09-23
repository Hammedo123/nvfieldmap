import numpy as np
import pytest
from nvfieldmap import hamiltonian as H
from nvfieldmap.constants import GAMMA_NV_MHZ_PER_MT as G


def test_zero_field_gives_D_pm_E():
    fm, fp = H.resonances_nv_frame((0, 0, 0), D=2870, E=3.0)
    assert fm == pytest.approx(2867, abs=1e-6) and fp == pytest.approx(2873, abs=1e-6)


def test_axial_field_splitting_is_2_gamma_B():
    fm, fp = H.resonances_nv_frame((0, 0, 2.0))
    assert fp - fm == pytest.approx(2 * G * 2.0, rel=1e-9)


def test_splitting_formula_inverts_axial_field():
    for B in (0.5, 3.0, 8.0):
        fm, fp = H.resonances_nv_frame((0, 0, B), E=3.17)
        assert H.b_parallel_from_splitting(fp, fm, E=3.17) == pytest.approx(B, rel=1e-9)


def test_published_rule_is_factor_two_high():
    # a 170 MHz splitting corresponds to ~3.03 mT, not 6.07 mT
    assert H.b_parallel_from_splitting(170, 0) == pytest.approx(3.03, abs=0.01)


def test_nv_axes_are_unit_and_tetrahedral():
    n = H.NV_AXES
    assert np.allclose(np.linalg.norm(n, axis=1), 1)
    off = (n @ n.T)[~np.eye(4, dtype=bool)]
    assert np.allclose(off, -1 / 3)


def test_vector_fit_recovers_generic_field():
    B = np.array([2.0, -1.0, 3.5])
    obs = H.resonances(B).ravel()
    r = H.fit_vector(obs, fit_D=False, n_starts=60)
    # identifiable quantities: |B| and the sorted set of |projections|
    assert np.linalg.norm(r["B"]) == pytest.approx(np.linalg.norm(B), abs=1e-4)
    assert np.allclose(np.sort(np.abs(H.NV_AXES @ r["B"])), np.sort(np.abs(H.NV_AXES @ B)), atol=1e-4)


def test_approximation_error_small_for_aligned_field():
    assert np.abs(H.approximation_error(3.0 * H.NV_AXES[0])[0]) < 1e-9
