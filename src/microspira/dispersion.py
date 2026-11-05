"""Unit-level overdispersion for the conditional technical null (MF4).

The working null allows independent multiplicative heterogeneity around the
technical mean: each unit's probability is multiplied by an independent
Gamma(1/kappa, kappa) factor (mean 1, variance kappa) before the taxon total is
allocated. kappa = 0 is the multinomial null of 0.1.0.dev2.

kappa is estimated from differences between graph neighbours, so spatially
smooth structure contributes little to the estimate. It is re-estimated on every
bootstrap draw, so the null distribution includes its estimation.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

OVERDISPERSION_METHODS = ("neighbour", "none")
MAX_KAPPA = 1e3


def graph_edges(W: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    """Unordered neighbour pairs (i < l) of a symmetric graph."""
    upper = sparse.triu(sparse.csr_matrix(W), k=1).tocoo()
    return upper.row.astype(np.int64), upper.col.astype(np.int64)


def neighbour_dispersion(counts: np.ndarray, mu: np.ndarray, edges: tuple[np.ndarray, np.ndarray]) -> float:
    """Neighbour-difference estimate of kappa (>= 0).

    With a_i = y_i / mu_i, D_il = (a_i - a_l)^2 - y_i / mu_i^2 - y_l / mu_l^2 has
    expectation 2 * kappa under the working null (plus the squared difference of
    local rate ratios, which is small for smooth structure). Pairs are weighted
    by (1/mu_i + 1/mu_l)^-2, which down-weights pairs with small fitted means.
    """
    i, l = edges
    if not len(i):
        return 0.0
    y = np.asarray(counts, dtype=float).ravel()
    m = np.clip(np.asarray(mu, dtype=float).ravel(), 1e-12, None)
    a = y / m
    D = (a[i] - a[l]) ** 2 - y[i] / m[i] ** 2 - y[l] / m[l] ** 2
    w = 1.0 / (1.0 / m[i] + 1.0 / m[l]) ** 2
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0:
        return 0.0
    kappa = float(np.sum(w * D) / (2.0 * total))
    if not np.isfinite(kappa):
        return 0.0
    return float(min(max(kappa, 0.0), MAX_KAPPA))


def overdispersed_residual(counts: np.ndarray, fitted: np.ndarray, kappa: float) -> np.ndarray:
    """(y - m) / sqrt(N p (1 - p) + kappa m^2); identical to the dev2 residual when kappa = 0.

    ``fitted`` must already be normalised to the taxon total (see model._normalise_mu).
    """
    y = np.asarray(counts, dtype=float).ravel()
    total = y.sum()
    if total <= 0:
        return np.zeros_like(y)
    p = fitted / fitted.sum()
    variance = total * p * (1.0 - p) + kappa * fitted ** 2
    return (y - fitted) / np.sqrt(np.clip(variance, 1e-10, None))


def draw_null(generator: np.random.Generator, total: int, probability: np.ndarray, kappa: float) -> np.ndarray:
    """One draw from the working null. kappa = 0 consumes the stream exactly as dev2."""
    if kappa > 0:
        weights = probability * generator.gamma(1.0 / kappa, kappa, size=probability.size)
        s = weights.sum()
        if s > 0 and np.isfinite(s):
            return generator.multinomial(total, weights / s).astype(float)
    return generator.multinomial(total, probability).astype(float)
