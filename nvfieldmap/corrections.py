"""Camera frame corrections (thesis Fig. 5.6, printed pp. 104-107).

Every function records what it did in the returned `applied` list so the
provenance ends up in the output metadata. Nothing is 'cleaned up' beyond
these documented operations.
"""
from __future__ import annotations

import numpy as np


def subtract_dark(frames, dark):
    """Subtract a dark frame taken at the same temperature and exposure."""
    return np.asarray(frames, float) - np.asarray(dark, float)


def subtract_bias(frames, bias):
    """Subtract a bias frame (dark frame at the camera's fastest exposure).
    Note: a dark frame at the science exposure already contains the bias; apply
    only one of subtract_dark/subtract_bias to the same data unless the dark was
    itself bias-subtracted (thesis applies both; record which convention was used)."""
    return np.asarray(frames, float) - np.asarray(bias, float)


def flat_field(frames, flat, flat_dark=None):
    """Divide by the normalised flat (mean = 1)."""
    fl = np.asarray(flat, float) - (0 if flat_dark is None else np.asarray(flat_dark, float))
    fl = fl / np.nanmean(fl)
    fl = np.where(fl > 0.05, fl, np.nan)
    return np.asarray(frames, float) / fl


def normalise_off_resonance(cube, freqs, off_windows):
    """Per-pixel normalisation by the mean of frames in off-resonance windows.
    cube shape (n_freq, y, x); off_windows list of (fmin, fmax) in MHz.
    Used when calibration frames are missing (brief, Phase 3.2)."""
    freqs = np.asarray(freqs, float)
    sel = np.zeros_like(freqs, bool)
    for lo, hi in off_windows:
        sel |= (freqs >= lo) & (freqs <= hi)
    if sel.sum() < 2:
        raise ValueError("need at least two off-resonance frames")
    ref = np.nanmean(np.asarray(cube, float)[sel], axis=0)
    return np.asarray(cube, float) / ref


def apply_all(cube, dark=None, bias=None, flat=None, flat_dark=None,
              freqs=None, off_windows=None):
    """Apply the available corrections in thesis order; return (cube, applied)."""
    applied = []
    out = np.asarray(cube, float)
    if dark is not None:
        out = subtract_dark(out, dark); applied.append("dark")
    if bias is not None:
        out = subtract_bias(out, bias); applied.append("bias")
    if flat is not None:
        out = flat_field(out, flat, flat_dark); applied.append("flat")
    if off_windows is not None:
        out = normalise_off_resonance(out, freqs, off_windows); applied.append("off_resonance_norm")
    for name in ("dark", "bias", "flat"):
        if name not in applied:
            applied.append(f"{name}:NOT_APPLIED")
    return out, applied
