"""Minimal optional plotting helpers for SPIRA diagnostics."""

from __future__ import annotations

import numpy as np


def residual_map(coords, residual, *, ax=None, size: float = 8.0, cmap: str = "coolwarm"):
    """Plot a residual map; matplotlib is imported only when plotting is requested."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Install microspira[plot] to use plotting helpers.") from error
    xy = np.asarray(coords, dtype=float)
    r = np.asarray(residual, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    vmax = float(np.nanquantile(np.abs(r), 0.99)) if np.isfinite(r).any() else 1.0
    artist = ax.scatter(xy[:, 0], xy[:, 1], c=r, s=size, cmap=cmap, vmin=-vmax, vmax=vmax, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax, artist

