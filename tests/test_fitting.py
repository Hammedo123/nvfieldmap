import numpy as np
import pytest
from nvfieldmap.fitting import model, fit_spectrum, pair_dips


def test_fit_recovers_two_dips():
    f = np.arange(2600, 3100, 2.0)
    y = model(f, 1.0, 0.0, 2800, 12, 0.05, 2940, 12, 0.05)
    r = fit_spectrum(f, y, 2, init_width=10)
    assert r.success
    assert r.centers == pytest.approx([2800, 2940], abs=0.05)


def test_pairing_outer_with_outer():
    p = pair_dips([2790, 2860, 2920, 2960])
    assert p[0][:2] == (2790, 2960) and p[1][:2] == (2860, 2920)
