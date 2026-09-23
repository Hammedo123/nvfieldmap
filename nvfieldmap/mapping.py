"""Assemble per-bin fits into one field-map Dataset (brief: 'Output format')."""
from __future__ import annotations

import subprocess

import numpy as np
import xarray as xr

from . import __version__
from . import hamiltonian as H
from .fitting import fit_spectrum, pair_dips
from .uncertainty import sigma_fit, rss


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def map_cube(cube, freqs, n_dips, init_centers, pixel_mm, bin_size, E=0.0,
             extra_terms=None, fit_kwargs=None, mask=None, meta=None):
    """Fit every bin of a (n_freq, ny, nx) binned cube.

    Returns xr.Dataset with B_parallel(pair, y, x), sigma_fit, sigma_B,
    fit_quality, mask, on a y/x grid in millimetres. `extra_terms` maps
    budget-term name -> scalar or (ny, nx) array (strain, frequency accuracy,
    sampling bias, model error ...), combined with sigma_fit by RSS.
    """
    fit_kwargs = fit_kwargs or {}
    nf, ny, nx = cube.shape
    npair = n_dips // 2
    Bp = np.full((npair, ny, nx), np.nan)
    sfit = np.full((npair, ny, nx), np.nan)
    q = np.full((ny, nx), np.nan)
    ok = np.zeros((ny, nx), bool)
    for iy in range(ny):
        for ix in range(nx):
            if mask is not None and not mask[iy, ix]:
                continue
            r = fit_spectrum(freqs, cube[:, iy, ix], n_dips, init_centers=init_centers, **fit_kwargs)
            q[iy, ix] = r.red_chi2
            if not r.success:
                continue
            ok[iy, ix] = True
            for k, (fm, fp, sm, sp) in enumerate(pair_dips(r.centers, r.center_sigma)):
                Bp[k, iy, ix] = H.b_parallel_from_splitting(fp, fm, E)
                sfit[k, iy, ix] = sigma_fit(fp, fm, sp, sm, E)
    extra = extra_terms or {}
    sig = np.sqrt(sfit ** 2 + sum(np.asarray(v, float) ** 2 for v in extra.values()))
    y_mm = (np.arange(ny) + 0.5) * bin_size * pixel_mm
    x_mm = (np.arange(nx) + 0.5) * bin_size * pixel_mm
    ds = xr.Dataset(
        {"B_parallel": (("pair", "y", "x"), Bp, {"units": "mT"}),
         "sigma_fit": (("pair", "y", "x"), sfit, {"units": "mT"}),
         "sigma_B": (("pair", "y", "x"), sig, {"units": "mT"}),
         "fit_quality": (("y", "x"), q, {"description": "reduced chi-squared"}),
         "mask": (("y", "x"), ok)},
        coords={"pair": np.arange(npair), "y": y_mm, "x": x_mm})
    for name, v in extra.items():
        ds[f"sigma_{name}"] = (("y", "x"), np.broadcast_to(np.asarray(v, float), (ny, nx)).copy(), {"units": "mT"})
    ds.attrs.update({"bin_size_px": bin_size, "pixel_scale_mm": pixel_mm,
                     "software_version": __version__, "git_commit": _git_commit(),
                     "E_MHz_used": E, **(meta or {})})
    return ds
