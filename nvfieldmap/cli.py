"""Command line: `nvfieldmap run config.yaml` (processes one run end to end)."""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
import yaml

from . import corrections, cube as C, io, mapping


def run(cfg):
    fr = sorted(glob.glob(cfg["frames_glob"]))
    frames = np.stack([io.load_frame(p) for p in fr])
    if "frequencies_file" in cfg:
        freqs = np.loadtxt(cfg["frequencies_file"])
    else:
        s = cfg["sweep"]
        freqs = io.frequencies_from_sweep(s["start_MHz"], s["stop_MHz"], s["step_MHz"])
        print("WARNING: frequencies reconstructed from sweep settings (state in paper)", file=sys.stderr)
    cal = {k: (io.load_frame(cfg[k]) if cfg.get(k) else None) for k in ("dark", "bias", "flat")}
    data, applied = corrections.apply_all(frames, freqs=freqs, off_windows=cfg.get("off_resonance_windows"), **cal)
    binned = C.bin_cube(data, cfg["bin"])
    ds = mapping.map_cube(binned, freqs, cfg["n_dips"], cfg.get("init_centers"),
                          cfg["pixel_mm"], cfg["bin"], E=cfg.get("E_MHz", 0.0),
                          meta={"corrections": ",".join(applied), "run_label": cfg.get("label", ""),
                                "sweep": str(cfg.get("sweep", "from log"))})
    io.save(ds, cfg["output"])
    print(f"wrote {cfg['output']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="nvfieldmap")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("config")
    a = p.parse_args(argv)
    if a.cmd == "run":
        with open(a.config) as fh:
            run(yaml.safe_load(fh))


if __name__ == "__main__":
    main()
