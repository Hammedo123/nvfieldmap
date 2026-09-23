"""Per-bin uncertainty budget (brief: 'Uncertainty budget') and the required
synthetic-recovery test. Fields in mT, frequencies in MHz."""
from __future__ import annotations

import numpy as np

from . import hamiltonian as H
from .fitting import fit_spectrum, model, pair_dips
from .constants import GAMMA_NV_MHZ_PER_MT as GAMMA


def rss(terms: dict) -> float:
    """Combined standard uncertainty: root-sum-square of independent terms."""
    v = np.array([t for t in terms.values() if np.isfinite(t)], float)
    return float(np.sqrt(np.sum(v ** 2)))


def sigma_fit(f_plus, f_minus, s_plus, s_minus, E=0.0):
    """Statistical term from dip-centre covariances (independent centres assumed)."""
    half = abs(f_plus - f_minus) / 2
    root = np.sqrt(max(half ** 2 - E ** 2, 1e-12))
    dBdhalf = half / (GAMMA * root)
    return float(dBdhalf * 0.5 * np.hypot(s_plus, s_minus))


def sigma_strain(f_plus, f_minus, E=3.17, sE=0.2):
    """Propagate E = 3.17 +/- 0.2 MHz (thesis p. 128) through the splitting formula.
    NOTE: this E value is itself disputed (see project notes); pass the value
    the reanalysis actually supports."""
    b = lambda e: H.b_parallel_from_splitting(f_plus, f_minus, e)  # noqa: E731
    return float(abs(b(E + sE) - b(E - sE)) / 2)


def sigma_frequency_accuracy(f_plus, f_minus, frac_ppm):
    """Microwave source frequency accuracy. frac_ppm MUST come from the LimeSDR
    reference-oscillator datasheet (cite it); no default is provided on purpose."""
    return float(abs(f_plus - f_minus) / 2 * frac_ppm * 1e-6 / GAMMA)


def sigma_miscut(observed, max_deg=3.0, n=50, seed=0, **fitkw):
    """Spread of the vector-fit |B| when NV axes are rotated by up to max_deg
    (thesis p. 93: +/-3 degree misorientation)."""
    rng = np.random.default_rng(seed)
    mags = []
    for i in range(n):
        ax = rng.normal(size=3)
        R = H.rotation_matrix(ax, np.deg2rad(rng.uniform(0, max_deg)))
        r = H.fit_vector(observed, axes=H.NV_AXES @ R.T, seed=i, n_starts=20, **fitkw)
        mags.append(np.linalg.norm(r["B"]))
    return float(np.std(mags, ddof=1))


def sigma_model(B_vec, E=0.0):
    """Max |splitting-formula - exact| over axes, for the fitted vector."""
    return float(np.max(np.abs(H.approximation_error(B_vec, E=E))))


def bootstrap_repeats(spectra, f, n_dips, n_boot=200, seed=0, **fitkw):
    """Bootstrap over repeated sweeps (rows of `spectra`): resample repeats with
    replacement, average, fit, compute outer-pair B||. Returns array of B||."""
    rng = np.random.default_rng(seed)
    spectra = np.asarray(spectra, float)
    out = []
    for _ in range(n_boot):
        pick = spectra[rng.integers(0, len(spectra), len(spectra))].mean(axis=0)
        r = fit_spectrum(f, pick, n_dips, **fitkw)
        if r.success:
            fm, fp, *_ = pair_dips(r.centers)[0]
            out.append(H.b_parallel_from_splitting(fp, fm))
    return np.array(out)


# ----------------------------------------------------------------- synthetic test
def synthetic_spectrum(B_vec, f, fwhm=20.0, contrast=0.1, D=H.D_ZFS_MHZ, E=0.0,
                       noise=0.0, rng=None):
    """Spectrum from the EXACT Hamiltonian: one Lorentzian dip per transition
    (8 total, equal contrast per NV axis/4). Labelled synthetic; never mixed
    with measured data."""
    fr = H.resonances(B_vec, D=D, E=E).ravel()
    dips = []
    for c in fr:
        dips += [c, fwhm, contrast / 4]
    y = model(f, 1.0, 0.0, *dips)
    if noise > 0:
        rng = np.random.default_rng() if rng is None else rng
        y = y + rng.normal(0, noise, size=np.shape(f))
    return y


def synthetic_recovery(B_values=np.arange(1, 11), step=10.0, fwhm=20.0,
                       contrast=0.1, noise=0.005, span=(2600.0, 3150.0),
                       direction=(1, 1, 1), n_trials=20, seed=0):
    """Field along `direction` (default: one NV axis, so 4 dips: the aligned
    pair plus the degenerate pair of the other three axes at B/3). Returns a
    dict of arrays: true, mean recovered, bias, spread, success fraction."""
    rng = np.random.default_rng(seed)
    f = np.arange(span[0], span[1] + 1e-9, step)
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    true_proj = np.abs(H.NV_AXES @ d).max()
    res = {"B_true": [], "mean": [], "bias": [], "spread": [], "success": []}
    for B in B_values:
        Bv = B * d
        truth = B * true_proj
        rec = []
        for _ in range(n_trials):
            y = synthetic_spectrum(Bv, f, fwhm, contrast, noise=noise, rng=rng)
            r = fit_spectrum(f, y, 4, init_width=fwhm)
            if r.success:
                fm, fp, *_ = pair_dips(r.centers)[0]
                rec.append(H.b_parallel_from_splitting(fp, fm))
        rec = np.array(rec)
        res["B_true"].append(truth)
        res["mean"].append(rec.mean() if rec.size else np.nan)
        res["bias"].append(rec.mean() - truth if rec.size else np.nan)
        res["spread"].append(rec.std(ddof=1) if rec.size > 1 else np.nan)
        res["success"].append(rec.size / n_trials)
    return {k: np.array(v) for k, v in res.items()}
