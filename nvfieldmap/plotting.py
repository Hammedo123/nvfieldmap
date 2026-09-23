"""Figures for the preprint. Every figure is regenerated from data by examples/reproduce_preprint.py."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .fitting import model  # noqa: E402


def spectrum_with_fit(f, y, fit, ax=None, label="data"):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(5, 4.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a0, a1 = ax
    a0.plot(f / 1e3, y, "o", ms=3, label=label)
    ff = np.linspace(f.min(), f.max(), 800)
    a0.plot(ff / 1e3, model(ff, *fit.params), "-", label="fit")
    a0.set_ylabel("Normalised fluorescence"); a0.legend()
    a1.plot(f / 1e3, fit.residuals, "o", ms=3); a1.axhline(0, color="k", lw=0.5)
    a1.set_xlabel("Microwave frequency (GHz)"); a1.set_ylabel("Residual")
    return fig


def field_map(ds, pair=0, path=None):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    for a, var, lab in zip(ax, ["B_parallel", "sigma_B"], ["$B_\\parallel$ (mT)", "$\\sigma_B$ (mT)"]):
        im = a.pcolormesh(ds.x, ds.y, ds[var].sel(pair=pair).where(ds.mask), shading="auto")
        fig.colorbar(im, ax=a, label=lab); a.set_aspect("equal"); a.set_xlabel("x (mm)"); a.set_ylabel("y (mm)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def synthetic_recovery(results_by_step, path=None):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for step, r in results_by_step.items():
        ax[0].errorbar(r["B_true"], r["mean"], yerr=r["spread"], fmt="o-", ms=3, capsize=2, label=f"{step:g} MHz step")
        ax[1].plot(r["B_true"], r["bias"], "o-", ms=3, label=f"{step:g} MHz step")
    lim = [0, max(np.nanmax(r["B_true"]) for r in results_by_step.values()) * 1.05]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set_xlabel("True $B_\\parallel$ (mT)"); ax[0].set_ylabel("Recovered $B_\\parallel$ (mT)"); ax[0].legend(fontsize=8)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("True $B_\\parallel$ (mT)"); ax[1].set_ylabel("Bias (mT)"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig
