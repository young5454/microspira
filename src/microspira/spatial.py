"""Global residual spatial statistics and multiple-testing correction."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def morans_i(values: np.ndarray, W: sparse.csr_matrix) -> float:
    """Compute Moran's I for a finite one-dimensional vector."""
    x = np.asarray(values, dtype=float).ravel()
    if W.shape != (x.size, x.size):
        raise ValueError("W dimensions must match values.")
    if not np.all(np.isfinite(x)):
        return np.nan
    z = x - x.mean()
    denom = float(np.dot(z, z))
    s0 = float(W.sum())
    if denom <= 0.0 or s0 <= 0.0:
        return np.nan
    numerator = float(z @ (W @ z))
    return float((x.size / s0) * (numerator / denom))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values, preserving NaN entries."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return q
    pv = np.clip(p[valid], 0.0, 1.0)
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(n, dtype=float)
    restored[order] = np.minimum(adjusted, 1.0)
    q[valid] = restored
    return q

