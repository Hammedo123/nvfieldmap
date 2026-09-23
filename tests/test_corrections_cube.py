import numpy as np
from nvfieldmap import corrections, cube


def test_apply_all_records_missing_calibration():
    c = np.ones((5, 4, 4))
    out, applied = corrections.apply_all(c, freqs=np.arange(5.0), off_windows=[(0, 1)])
    assert "dark:NOT_APPLIED" in applied and "off_resonance_norm" in applied
    assert np.allclose(out, 1)


def test_bin_cube_shape_and_mean():
    c = np.arange(2 * 4 * 6, dtype=float).reshape(2, 4, 6)
    b = cube.bin_cube(c, 2)
    assert b.shape == (2, 2, 3) and b[0, 0, 0] == np.mean(c[0, :2, :2])
