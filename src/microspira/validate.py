"""Input validation for count-aware spatial microbial analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


class ValidationError(ValueError):
    """Raised when an object cannot safely enter the SPIRA workflow."""


@dataclass(frozen=True)
class ValidationSummary:
    n_units: int
    n_taxa: int
    spatial_key: str
    taxonomy_columns: tuple[str, ...]
    host_aligned: bool


def _as_array(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def _is_integer_like(values: np.ndarray, atol: float = 1e-8) -> bool:
    if values.size == 0:
        return True
    return bool(np.all(np.isfinite(values)) and np.all(np.abs(values - np.rint(values)) <= atol))


def _matrix_values(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.data, dtype=float)
    return np.asarray(x, dtype=float).ravel()


def _index_of(adata) -> pd.Index:
    if hasattr(adata, "obs_names"):
        return pd.Index(adata.obs_names)
    if hasattr(adata, "obs"):
        return pd.Index(adata.obs.index)
    raise ValidationError("AnnData-like input must expose obs_names or obs.index.")


def validate(
    microbial_adata,
    host_adata=None,
    *,
    spatial_key: str = "spatial",
    taxonomy_columns: Iterable[str] = (),
    require_integer_counts: bool = True,
) -> ValidationSummary:
    """Validate a microbial AnnData-like object before SPIRA analysis.

    The function intentionally uses duck typing so that lightweight test
    objects can be used, while regular :class:`anndata.AnnData` works directly.
    """
    if not hasattr(microbial_adata, "X") or not hasattr(microbial_adata, "obs"):
        raise ValidationError("microbial_adata must provide X and obs attributes.")
    if not hasattr(microbial_adata, "var"):
        raise ValidationError("microbial_adata must provide a var table.")
    if not hasattr(microbial_adata, "obsm") or spatial_key not in microbial_adata.obsm:
        raise ValidationError(
            f"microbial_adata.obsm[{spatial_key!r}] is required for spatial inference."
        )

    X = microbial_adata.X
    if len(X.shape) != 2:
        raise ValidationError("X must be a two-dimensional spot/bin × taxon count matrix.")
    n_units, n_taxa = X.shape
    if len(microbial_adata.obs) != n_units:
        raise ValidationError("obs row count does not match X.shape[0].")
    if len(microbial_adata.var) != n_taxa:
        raise ValidationError("var row count does not match X.shape[1].")
    if n_units < 3:
        raise ValidationError("At least three spatial units are required.")

    values = _matrix_values(X)
    if not np.all(np.isfinite(values)):
        raise ValidationError("Raw microbial count matrix contains non-finite values.")
    if np.any(values < 0):
        raise ValidationError("Raw microbial count matrix contains negative values.")
    if require_integer_counts and not _is_integer_like(values):
        raise ValidationError("SPIRA requires raw integer-like counts, not normalized values.")

    coords = np.asarray(microbial_adata.obsm[spatial_key], dtype=float)
    if coords.ndim != 2 or coords.shape[0] != n_units or coords.shape[1] < 2:
        raise ValidationError(
            f"obsm[{spatial_key!r}] must have shape (n_units, >=2)."
        )
    if not np.all(np.isfinite(coords[:, :2])):
        raise ValidationError("Spatial coordinates contain non-finite values.")
    if np.unique(coords[:, :2], axis=0).shape[0] < 3:
        raise ValidationError("At least three unique spatial coordinates are required.")

    requested = tuple(taxonomy_columns)
    missing = [c for c in requested if c not in microbial_adata.var.columns]
    if missing:
        raise ValidationError(f"Missing requested taxonomy columns: {missing}")

    host_aligned = False
    if host_adata is not None:
        microbial_index, host_index = _index_of(microbial_adata), _index_of(host_adata)
        if not microbial_index.equals(host_index):
            raise ValidationError(
                "Host and microbial objects must share the same barcodes in the same order."
            )
        if hasattr(host_adata, "obsm") and spatial_key in host_adata.obsm:
            host_coords = np.asarray(host_adata.obsm[spatial_key], dtype=float)
            if host_coords.shape != coords.shape or not np.allclose(host_coords, coords):
                raise ValidationError("Host and microbial spatial coordinates are not aligned.")
        host_aligned = True

    return ValidationSummary(
        n_units=n_units,
        n_taxa=n_taxa,
        spatial_key=spatial_key,
        taxonomy_columns=requested,
        host_aligned=host_aligned,
    )

