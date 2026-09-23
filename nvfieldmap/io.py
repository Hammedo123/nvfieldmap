"""Loading frames, attaching frequencies, saving cubes/maps (NetCDF via h5netcdf)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def load_frame(path):
    p = Path(path)
    if p.suffix.lower() == ".npy":
        a = np.load(p)
    else:  # TIFF / PNG through Pillow (a matplotlib dependency)
        from PIL import Image
        a = np.asarray(Image.open(p))
    a = np.asarray(a, float)
    if a.ndim == 3:            # colour camera (thesis: CMLN-13S2C, Bayer colour)
        a = a.mean(axis=2)     # documented choice: sum-of-channels proxy
    return a


def frequencies_from_sweep(start_MHz, stop_MHz, step_MHz):
    """Reconstruct the frame->frequency map from sweep settings (brief Phase 1.3).
    Use ONLY if no log exists, and state the assumption in the paper."""
    n = int(round((stop_MHz - start_MHz) / step_MHz)) + 1
    return start_MHz + step_MHz * np.arange(n)


def build_cube(frames, freqs_MHz, attrs=None):
    frames = np.asarray(frames, float)
    if frames.shape[0] != len(freqs_MHz):
        raise ValueError(f"{frames.shape[0]} frames but {len(freqs_MHz)} frequencies")
    ny, nx = frames.shape[1:]
    return xr.DataArray(frames, dims=("frequency", "y", "x"),
                        coords={"frequency": np.asarray(freqs_MHz, float),
                                "y": np.arange(ny), "x": np.arange(nx)},
                        name="intensity", attrs=attrs or {}).assign_attrs(freq_units="MHz")


def save(ds, path):
    ds.to_netcdf(path, engine="h5netcdf")


def load(path):
    return xr.open_dataset(path, engine="h5netcdf")
