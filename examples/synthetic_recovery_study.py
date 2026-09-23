"""Synthetic-recovery study (brief: required test). SYNTHETIC DATA ONLY.
Noise and contrast are PLACEHOLDERS until measured from the raw frames."""
import json, sys
import numpy as np
from nvfieldmap.uncertainty import synthetic_recovery
from nvfieldmap import plotting

B = np.arange(1, 11)
common = dict(fwhm=20.0, contrast=0.2, n_trials=8, seed=1, noise=0.005)
res = {s: synthetic_recovery(B, step=s, span=(2550, 3200), **common) for s in (2.0, 10.0)}
plotting.synthetic_recovery(res, path="examples/fig_synthetic_recovery.png")
pub = synthetic_recovery(B, step=10.0, span=(2600, 3000), **common)
out = {"wide_window": {str(k): {kk: np.round(v, 3).tolist() for kk, v in r.items()} for k, r in res.items()},
       "published_window_2600_3000_step10": {kk: np.round(v, 3).tolist() for kk, v in pub.items()}}
json.dump(out, open("examples/synthetic_recovery_results.json", "w"), indent=1)
for k, r in res.items():
    print(f"wide window, step {k}: bias {np.round(r['bias'],3)}\n   spread {np.round(r['spread'],3)}\n   success {r['success']}")
print("published window 2.60-3.00 GHz, 10 MHz: mean", np.round(pub['mean'],2), "\n   success", pub['success'])
