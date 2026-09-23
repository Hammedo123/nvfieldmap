"""NV ground-state spin Hamiltonian, resonance frequencies and field inversion.

Units: frequencies in MHz, fields in mT.

    H/h = D Sz^2 + E (Sx^2 - Sy^2) + gamma * B . S        (NV frame)

This extends thesis Eq. 5.1 (printed p. 91) with the strain term E.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .constants import GAMMA_NV_MHZ_PER_MT, D_ZFS_MHZ

_s = 1 / np.sqrt(2)
SX = _s * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
SY = _s * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex)
SZ = np.diag([1.0, 0.0, -1.0]).astype(complex)

# Four NV axes in the crystal frame of a (100)-cut plate (z = plate normal).
NV_AXES = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3)


def rotation_matrix(axis, angle_rad):
    """Rodrigues rotation matrix."""
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * K @ K


def _nv_frame(axis):
    z = np.asarray(axis, float) / np.linalg.norm(axis)
    ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = ref - ref.dot(z) * z
    x /= np.linalg.norm(x)
    return x, np.cross(z, x), z


def resonances_nv_frame(b_nv, D=D_ZFS_MHZ, E=0.0, gamma=GAMMA_NV_MHZ_PER_MT):
    """Exact (f-, f+) in MHz for field b_nv (mT) given in the NV frame.
    Valid below the ground-state level anti-crossing (~100 mT)."""
    bx, by, bz = b_nv
    H = D * SZ @ SZ + E * (SX @ SX - SY @ SY) + gamma * (bx * SX + by * SY + bz * SZ)
    ev = np.sort(np.linalg.eigvalsh(H))
    return ev[1] - ev[0], ev[2] - ev[0]


def resonances(B, axes=NV_AXES, D=D_ZFS_MHZ, E=0.0, gamma=GAMMA_NV_MHZ_PER_MT):
    """Array (n_axes, 2) of (f-, f+) in MHz for crystal-frame field B (mT)."""
    B = np.asarray(B, float)
    rows = []
    for a in axes:
        x, y, z = _nv_frame(a)
        rows.append(resonances_nv_frame((B @ x, B @ y, B @ z), D, E, gamma))
    return np.array(rows)


def b_parallel_from_splitting(f_plus, f_minus, E=0.0, gamma=GAMMA_NV_MHZ_PER_MT):
    """Projection on one NV axis from a resolved dip pair, ignoring transverse
    field: B|| = sqrt(((f+ - f-)/2)^2 - E^2) / gamma. Temperature drift of D cancels."""
    half = np.abs(np.asarray(f_plus, float) - np.asarray(f_minus, float)) / 2
    return np.sqrt(np.clip(half ** 2 - E ** 2, 0, None)) / gamma


def approximation_error(B, axes=NV_AXES, D=D_ZFS_MHZ, E=0.0):
    """Per-axis error (mT) of the splitting formula relative to the true |projection|."""
    B = np.asarray(B, float)
    f = resonances(B, axes, D, E)
    return b_parallel_from_splitting(f[:, 1], f[:, 0], E) - np.abs(axes @ B)


def fit_vector(observed, D=D_ZFS_MHZ, E=0.0, axes=NV_AXES, fit_D=True,
               n_starts=64, b_max=15.0, seed=0, window=None, merge_MHz=0.0):
    """Least-squares field vector (mT) from resolved dip frequencies (MHz), using
    the exact Hamiltonian. Each observed dip is matched to its nearest predicted
    transition, so degenerate/unresolved dips need no manual assignment.

    window=(fmin, fmax): if given, every PREDICTED dip inside the swept window
    must also lie within merge_MHz of an observed dip (no 'invisible' dips);
    violations add residuals. Use merge_MHz ~ linewidth/2 to allow merged dips.

    Returns dict(B, D, residuals_MHz, cov, rank, n_params).
    IDENTIFIABILITY: an ensemble spectrum fixes |B| and the SET of |projections|
    on the four axes. B -> -B and the tetrahedral permutations of the axes give
    identical spectra, so the lab-frame direction of B is not determined unless
    extra information breaks the symmetry (known crystal orientation plus
    orientation-dependent contrast, a bias field, etc.). Report |B| and the
    sorted projections; do not claim a direction without that information.
    """
    obs = np.sort(np.asarray(observed, float))
    rng = np.random.default_rng(seed)

    def resid(p):
        d = p[3] if fit_D else D
        pred = resonances(p[:3], axes, d, E).ravel()
        r = [np.min(np.abs(pred - o)) for o in obs]
        if window is not None:
            inside = pred[(pred > window[0]) & (pred < window[1])]
            r += [max(np.min(np.abs(obs - q)) - merge_MHz, 0.0) for q in inside]
            r += [0.0] * (8 - len(inside))   # fixed-length residual vector
        return np.array(r)

    best = None
    for _ in range(n_starts):
        b0 = rng.normal(size=3)
        b0 *= rng.uniform(0.3, b_max) / np.linalg.norm(b0)
        r = least_squares(resid, np.r_[b0, D] if fit_D else b0)
        if best is None or r.cost < best.cost:
            best = r
    J = best.jac
    npar = len(best.x)
    dof = max(len(obs) - npar, 1)
    cov = np.linalg.pinv(J.T @ J) * (2 * best.cost / dof)
    return {"B": best.x[:3], "D": best.x[3] if fit_D else D,
            "residuals_MHz": best.fun, "cov": cov,
            "rank": int(np.linalg.matrix_rank(J, tol=1e-6)),
            "n_params": npar}
