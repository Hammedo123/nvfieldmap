"""Cube utilities: repeat averaging, spatial binning, masks. Cube shape (n_freq, y, x)."""
from __future__ import annotations

import numpy as np


def average_repeats(cubes):
    """Mean and standard error over repeated sweeps (list of equal-shape cubes)."""
    st = np.stack([np.asarray(c, float) for c in cubes])
    n = st.shape[0]
    sem = np.nanstd(st, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(st.shape[1:], np.nan)
    return np.nanmean(st, axis=0), sem


def bin_cube(cube, by, bx=None):
    """Mean-bin in y and x by integer factors; trailing pixels are dropped."""
    bx = by if bx is None else bx
    c = np.asarray(cube, float)
    nf, ny, nx = c.shape
    ny2, nx2 = ny // by, nx // bx
    c = c[:, :ny2 * by, :nx2 * bx].reshape(nf, ny2, by, nx2, bx)
    return np.nanmean(c, axis=(2, 4))


def roi_mask(image, threshold_fraction=0.3):
    """Boolean mask of pixels brighter than a fraction of the image maximum
    (e.g. the illuminated diamond in a mean-fluorescence frame)."""
    im = np.asarray(image, float)
    return im >= threshold_fraction * np.nanmax(im)
