"""Localisation and physical-scale summaries for residual hotspots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import binom
import warnings


CLUSTER_COLUMNS = [
    "rank",
    "taxon",
    "cluster_id",
    "n_centres",
    "n_member_units",
    "member_reads",
    "member_area",
    "member_area_unit",
    "n_shared_members",
    "centroid_x",
    "centroid_y",
    "peak_residual",
    "localization_method",
    "local_max_adjusted_p_min",
    "peak_neighborhood_reads",
    "peak_neighborhood_expected",
    "component_supported",
    "area_kind",
]


def neighborhood_operator(W: sparse.csr_matrix) -> sparse.csr_matrix:
    """Fixed center-plus-one-hop sets, with binary membership and no double counting."""
    A = W.tocsr().copy()
    A.sum_duplicates()
    A.data = (A.data > 0).astype(float)
    A = A.maximum(A.T)
    A.setdiag(1.0)
    A.eliminate_zeros()
    return A


def neighborhood_enrichment(counts, fitted_mean, operator):
    """Binomial upper-tail scores for fixed neighborhoods conditional on total reads.

    These scores are ranking statistics, not calibrated local p-values when
    the mean was fitted. model.residual_spatial_test calibrates their maximum
    with the same plug-in/refitted bootstrap used for the global statistic.
    """
    y = np.asarray(counts, dtype=float).ravel()
    mu = np.asarray(fitted_mean, dtype=float).ravel()
    if y.size != mu.size or operator.shape != (y.size, y.size):
        raise ValueError("Counts, fitted mean and neighborhood operator must align.")
    if not np.isfinite(mu).all() or np.any(mu < 0) or (y.sum() > 0 and mu.sum() <= 0):
        raise ValueError("Fitted mean must be finite, non-negative and positive for nonzero counts.")
    total = int(np.rint(y.sum()))
    neighborhood_reads = np.asarray(operator @ y).ravel()
    if total <= 0:
        return np.zeros_like(y), neighborhood_reads, np.zeros_like(y)
    probability = np.clip(np.asarray(operator @ (mu / mu.sum())).ravel(), 0., 1.)
    log_tail = binom.logsf(np.rint(neighborhood_reads).astype(int) - 1, total, probability)
    # A fixed finite cap handles numerical underflow in very strong positives.
    score = -np.maximum(log_tail, np.log(np.finfo(float).tiny)) / np.log(10.)
    return score, neighborhood_reads, total * probability


def localize_hotspots(
    counts: np.ndarray,
    residual: np.ndarray,
    W: sparse.csr_matrix,
    coords: np.ndarray,
    *,
    rank: str,
    taxon: str,
    z_threshold: float = 1.96,
    unit_area: float | None = None,
    coordinate_unit: str | None = None,
    local_p: np.ndarray | None = None,
    neighborhood_reads: np.ndarray | None = None,
    neighborhood_expected: np.ndarray | None = None,
    local_alpha: float = 0.05,
    min_neighborhood_reads: int = 3,
    min_cluster_units: int = 2,
    member_z: float = 1.96,
) -> pd.DataFrame:
    """Summarise neighborhoods passing within-taxon bootstrap maximum adjustment.

    Components are connected candidate centres. Their member units are the
    units inside the components' neighbourhood sets whose own standardized
    residual is at least ``member_z``; size and support are counted in member
    units, not centres. A member falling in several components is assigned to
    the component with the smallest minimum local p (then larger peak
    neighbourhood reads, then lower component index). Members are units
    carrying excess counts, not colony boundaries. The old residual cutoff
    remains a deprecated compatibility mode when local_p is absent.
    """
    y = np.asarray(counts, dtype=float).ravel()
    r = np.asarray(residual, dtype=float).ravel()
    xy = np.asarray(coords, dtype=float)
    if y.size != r.size or xy.shape[0] != y.size or W.shape != (y.size, y.size):
        raise ValueError("counts, residual, coordinates, and W must be aligned.")
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive.")
    if unit_area is not None and (unit_area <= 0 or coordinate_unit is None):
        raise ValueError("Positive unit_area requires an explicit coordinate_unit.")
    if not 0 < local_alpha < 1 or min_neighborhood_reads < 1 or min_cluster_units < 1:
        raise ValueError("Local alpha/support settings are invalid.")
    if not np.isfinite(member_z) or member_z <= 0:
        raise ValueError("member_z must be positive and finite.")

    if local_p is None:
        warnings.warn("Legacy residual-threshold localization is descriptive and uncalibrated.",
                      DeprecationWarning, stacklevel=2)
        active = np.isfinite(r) & (r >= z_threshold)
        method = "legacy_residual_threshold"
        local_p = np.full(y.size, np.nan)
        neighborhood_reads = np.full(y.size, np.nan)
        neighborhood_expected = np.full(y.size, np.nan)
    else:
        local_p = np.asarray(local_p, dtype=float)
        neighborhood_reads = np.asarray(neighborhood_reads, dtype=float)
        neighborhood_expected = np.asarray(neighborhood_expected, dtype=float)
        if any(a.shape != y.shape for a in (local_p, neighborhood_reads, neighborhood_expected)):
            raise ValueError("Local enrichment arrays must align to counts.")
        active = (np.isfinite(local_p) & (local_p <= local_alpha)
                  & (neighborhood_reads >= min_neighborhood_reads))
        method = "bootstrap_neighborhood_max"
    if not np.any(active):
        return pd.DataFrame(columns=CLUSTER_COLUMNS)

    subgraph = W[active][:, active]
    n_components, labels = connected_components(subgraph, directed=False)
    active_positions = np.flatnonzero(active)
    legacy = method == "legacy_residual_threshold"
    centres = [active_positions[labels == k] for k in range(n_components)]

    # Spot-level membership (bootstrap mode). Legacy mode: active units are members.
    if legacy:
        members = centres
        shared = [0] * n_components
    else:
        A = neighborhood_operator(W)
        excess = np.isfinite(r) & (r >= member_z)
        candidate = []
        for positions in centres:
            region = np.flatnonzero(np.asarray(A[positions].sum(axis=0)).ravel() > 0)
            candidate.append(region[excess[region]])
        priority = [(float(np.min(local_p[c])), -float(np.max(neighborhood_reads[c])), k)
                    for k, c in enumerate(centres)]
        owner: dict[int, int] = {}
        claims: dict[int, int] = {}
        for k, units in enumerate(candidate):
            for unit in units.tolist():
                claims[unit] = claims.get(unit, 0) + 1
                if unit not in owner or priority[k] < priority[owner[unit]]:
                    owner[unit] = k
        members = [np.array(sorted(u for u, k in owner.items() if k == comp), dtype=int)
                   for comp in range(n_components)]
        shared = [int(sum(claims[u] > 1 for u in m.tolist())) for m in members]

    rows = []
    for component in range(n_components):
        positions, member = centres[component], members[component]
        located = member if len(member) else positions
        if unit_area is not None:
            member_area = float(len(member) * unit_area)
            member_area_unit = f"{coordinate_unit}^2"
        else:
            member_area, member_area_unit = np.nan, "not_available"
        rows.append(
            {
                "rank": rank,
                "taxon": taxon,
                "cluster_id": f"{rank}:{taxon}:{component + 1}",
                "n_centres": np.nan if legacy else int(len(positions)),
                "n_member_units": int(len(member)),
                "member_reads": float(y[member].sum()) if len(member) else 0.0,
                "member_area": member_area,
                "member_area_unit": member_area_unit,
                "n_shared_members": shared[component],
                "centroid_x": float(xy[located, 0].mean()),
                "centroid_y": float(xy[located, 1].mean()),
                "peak_residual": float(np.nanmax(r[located])),
                "localization_method": method,
                "local_max_adjusted_p_min": float(np.min(local_p[positions])) if np.isfinite(local_p[positions]).any() else np.nan,
                "peak_neighborhood_reads": float(np.max(neighborhood_reads[positions])),
                "peak_neighborhood_expected": float(np.max(neighborhood_expected[positions])),
                "component_supported": bool(len(member) >= min_cluster_units),
                "area_kind": "residual_exceedance_footprint" if legacy else "excess_member_units",
            }
        )
    return pd.DataFrame(rows, columns=CLUSTER_COLUMNS)

