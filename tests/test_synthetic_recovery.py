"""Required synthetic-recovery test (brief). Noise/contrast here are placeholders
until measured values from the raw data replace them."""
import numpy as np
from nvfieldmap.uncertainty import synthetic_recovery


def test_recovery_fine_sampling_is_unbiased():
    r = synthetic_recovery(B_values=[2, 5, 8], step=2.0, fwhm=20, contrast=0.1,
                           noise=0.002, n_trials=5)
    assert np.all(r["success"] > 0.6)
    assert np.all(np.abs(r["bias"]) < 0.05)


def test_recovery_10MHz_sampling_runs_and_is_bounded():
    r = synthetic_recovery(B_values=[3, 6], step=10.0, fwhm=20, contrast=0.1,
                           noise=0.002, n_trials=5)
    assert np.all(r["success"] > 0.6)
    assert np.all(np.abs(r["bias"]) < 0.2)
