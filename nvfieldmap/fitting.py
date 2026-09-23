"""Multi-Lorentzian CW-ODMR model and per-spectrum fitting.

Model (thesis Eq. 6.1, printed p. 122, extended to N dips and a linear baseline):

    I(f) = I0 (1 + s (f - fc)) [1 - sum_k C_k (G_k/2)^2 / ((f - f_k)^2 + (G_k/2)^2)]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def model(f, I0, s, *dips, fc=2870.0):
    f = np.asarray(f, float)
    out = np.ones_like(f)
    for k in range(0, len(dips), 3):
        fk, gk, ck = dips[k:k + 3]
        out -= ck * (gk / 2) ** 2 / ((f - fk) ** 2 + (gk / 2) ** 2)
    return I0 * (1 + s * (f - fc)) * out


@dataclass
class FitResult:
    success: bool
    centers: np.ndarray = field(default_factory=lambda: np.array([]))
    widths: np.ndarray = field(default_factory=lambda: np.array([]))
    contrasts: np.ndarray = field(default_factory=lambda: np.array([]))
    center_sigma: np.ndarray = field(default_factory=lambda: np.array([]))
    params: np.ndarray = field(default_factory=lambda: np.array([]))
    cov: np.ndarray | None = None
    red_chi2: float = np.nan
    residuals: np.ndarray | None = None
    message: str = ""


def guess_dips(f, y, n, min_sep_MHz=15.0, smooth_MHz=0.0):
    """Initial dip centres from the n most prominent minima, after an optional
    moving-average smoothing of width smooth_MHz (used only for the guess)."""
    f = np.asarray(f, float)
    step = np.median(np.diff(f))
    dist = max(int(round(min_sep_MHz / step)), 1)
    w = int(round(smooth_MHz / step))
    ys = np.convolve(y, np.ones(w) / w, mode="same") if w > 1 else np.asarray(y, float)
    yy = (np.max(ys) - ys)
    idx, props = find_peaks(yy, distance=dist, prominence=0)
    order = np.argsort(props["prominences"])[::-1][:n]
    return np.sort(f[idx[order]])


def fit_spectrum(f, y, n_dips, init_centers=None, init_width=20.0, sigma=None,
                 width_bounds=(1.0, 80.0), fc=2870.0, max_red_chi2=np.inf,
                 min_contrast=0.0, min_contrast_snr=3.0) -> FitResult:
    """Fit N Lorentzian dips. Covariance is scaled by reduced chi^2
    (curve_fit with absolute_sigma=False). Quality flags: convergence,
    finite covariance, red_chi2 <= max_red_chi2, all contrasts >= min_contrast."""
    f = np.asarray(f, float)
    y = np.asarray(y, float)
    good = np.isfinite(y)
    f, y = f[good], y[good]
    sig = None if sigma is None else np.asarray(sigma, float)[good]
    if init_centers is None:
        init_centers = guess_dips(f, y, n_dips, smooth_MHz=init_width / 2)
    init_centers = np.asarray(init_centers, float)
    if len(init_centers) < n_dips:
        return FitResult(False, message="fewer candidate dips than requested")
    base = np.percentile(y, 90)
    depth = max(base - np.min(y), 1e-6) / base
    p0 = [base, 0.0]
    lo = [0.0, -1e-2]
    hi = [np.inf, 1e-2]
    for c in init_centers[:n_dips]:
        p0 += [c, init_width, min(depth, 0.9)]
        lo += [f.min(), width_bounds[0], 0.0]
        hi += [f.max(), width_bounds[1], 1.0]
    fun = lambda ff, *p: model(ff, *p, fc=fc)  # noqa: E731
    try:
        popt, pcov = curve_fit(fun, f, y, p0=p0, bounds=(lo, hi), sigma=sig,
                               absolute_sigma=False, maxfev=20000)
    except (RuntimeError, ValueError) as e:
        return FitResult(False, message=str(e))
    res = y - fun(f, *popt)
    dof = max(len(y) - len(popt), 1)
    w = 1.0 if sig is None else sig
    red = float(np.sum((res / w) ** 2) / dof)
    d = popt[2:].reshape(-1, 3)
    order = np.argsort(d[:, 0])
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))[2:].reshape(-1, 3)
    c, w = d[order, 0], d[order, 1]
    msgs = []
    if not np.all(np.isfinite(pcov)):
        msgs.append("non-finite covariance")
    if red > max_red_chi2:
        msgs.append("reduced chi2 above threshold")
    if np.any(d[:, 2] < min_contrast):
        msgs.append("contrast below threshold")
    # every dip must be statistically significant: contrast / sigma_contrast
    if np.any(d[:, 2] < min_contrast_snr * np.maximum(perr[:, 2], 1e-15)):
        msgs.append("dip not significant")
    # a dip whose centre sits within half a linewidth of the swept window edge is
    # not measured, only extrapolated
    if np.any(c - w / 2 < f.min()) or np.any(c + w / 2 > f.max()):
        msgs.append("dip at window edge")
    # two fitted dips closer than half a linewidth are not independently resolved
    # Lorentzian resolution criterion: separation > FWHM/sqrt(3)
    if len(c) > 1 and np.any(np.diff(c) < np.minimum(w[:-1], w[1:]) / np.sqrt(3)):
        msgs.append("dips not resolved")
    ok = not msgs
    return FitResult(ok, c, w, d[order, 2], perr[order, 0],
                     popt, pcov, red, res, "; ".join(msgs))


def pair_dips(centers, center_sigma=None):
    """Pair sorted dip centres symmetrically (outermost with outermost, ...).
    Returns list of (f_minus, f_plus, sigma_minus, sigma_plus)."""
    c = np.sort(np.asarray(centers, float))
    s = np.zeros_like(c) if center_sigma is None else np.asarray(center_sigma, float)[np.argsort(centers)]
    n = len(c)
    return [(c[i], c[n - 1 - i], s[i], s[n - 1 - i]) for i in range(n // 2)]
