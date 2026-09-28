"""Localisation and physical-scale summaries for residual hotspots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


CLUSTER_COLUMNS = [
    "rank",
    "taxon",
    "cluster_id",
    "n_units",
    "cluster_reads",
    "area",
    "area_unit",
    "equivalent_diameter",
    "diameter_unit",
    "centroid_x",
    "centroid_y",
    "peak_residual",
]


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
) -> pd.DataFrame:
    """Summarise connected positive residual components as descriptive hotspots."""
    y = np.asarray(counts, dtype=float).ravel()
    r = np.asarray(residual, dtype=float).ravel()
    xy = np.asarray(coords, dtype=float)
    if y.size != r.size or xy.shape[0] != y.size or W.shape != (y.size, y.size):
        raise ValueError("counts, residual, coordinates, and W must be aligned.")
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive.")

    active = np.isfinite(r) & (r >= z_threshold)
    if not np.any(active):
        return pd.DataFrame(columns=CLUSTER_COLUMNS)

    subgraph = W[active][:, active]
    n_components, labels = connected_components(subgraph, directed=False)
    active_positions = np.flatnonzero(active)
    rows = []
    for component in range(n_components):
        positions = active_positions[labels == component]
        if not len(positions):
            continue
        if unit_area is not None:
            area = float(len(positions) * unit_area)
            area_unit = f"{coordinate_unit or 'coordinate'}^2"
            diameter = float(np.sqrt(4.0 * area / np.pi))
            diameter_unit = coordinate_unit or "coordinate"
        else:
            area = float(len(positions))
            area_unit = "units"
            diameter = np.nan
            diameter_unit = coordinate_unit or "not_available"
        rows.append(
            {
                "rank": rank,
                "taxon": taxon,
                "cluster_id": f"{rank}:{taxon}:{component + 1}",
                "n_units": int(len(positions)),
                "cluster_reads": float(y[positions].sum()),
                "area": area,
                "area_unit": area_unit,
                "equivalent_diameter": diameter,
                "diameter_unit": diameter_unit,
                "centroid_x": float(xy[positions, 0].mean()),
                "centroid_y": float(xy[positions, 1].mean()),
                "peak_residual": float(np.nanmax(r[positions])),
            }
        )
    return pd.DataFrame(rows, columns=CLUSTER_COLUMNS)

