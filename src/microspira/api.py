"""High-level SPIRA analysis API."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .clusters import CLUSTER_COLUMNS, localize_hotspots
from .config import SPIRAConfig
from .context import assess_context_association
from .graph import build_knn_graph
from .model import design_matrix, fit_poisson_gam, residual_spatial_test
from .report import SPIRAResult
from .spatial import benjamini_hochberg
from .support import NOT_TESTABLE, assess_support
from .taxonomy import aggregate_counts
from .validate import validate


NO_EVIDENCE = "NO_EVIDENCE_OF_SPATIAL_STRUCTURE"
STRUCTURED = "SPATIALLY_STRUCTURED"


def run(
    microbial_adata,
    *,
    technical_covariates: Sequence[str] = (),
    context_covariates: Mapping[str, str] | None = None,
    ranks: Sequence[str] = (),
    spatial_key: str = "spatial",
    graph: sparse.csr_matrix | None = None,
    n_neighbors: int = 6,
    coordinate_unit: str | None = None,
    unit_area: float | None = None,
    spline_df: int = 3,
    n_boot: int = 199,
    refit_bootstrap: bool = False,
    alpha: float = 0.05,
    seed: int = 0,
    localize: bool = True,
    hotspot_z: float = 1.96,
    config: SPIRAConfig | None = None,
) -> SPIRAResult:
    """Run support, spatial-pattern, context, taxonomy, and cluster reporting.

    Parameters
    ----------
    microbial_adata
        An AnnData-like object with raw microbial counts in ``.X``, spot/bin
        metadata in ``.obs``, taxonomic labels in ``.var``, and coordinates in
        ``.obsm[spatial_key]``.
    technical_covariates
        Variables used in the *primary* technical mean model. Choose variables
        whose adjustment is justified as measurement/exposure control.
    context_covariates
        Mapping from a human-readable flag name to an ``.obs`` column. These
        are reported as independent associations and never remove a spatial
        call. Example: ``{"host_density": "host_density"}``.
    ranks
        Optional rank columns in ``.var`` to aggregate and test in addition to
        native input features, e.g. ``("species", "genus", "family")``.
    refit_bootstrap
        Use model refitting for every null draw. This is slower but should be
        enabled for final calibration and manuscript-grade inference.

    Returns
    -------
    SPIRAResult
        Tidy taxon and cluster tables, reproducibility metadata, and per-taxon
        residual diagnostics.
    """
    if config is not None:
        config = _merge_config(
            config,
            technical_covariates=technical_covariates,
            context_covariates=context_covariates,
            ranks=ranks,
            spatial_key=spatial_key,
            n_neighbors=n_neighbors,
            coordinate_unit=coordinate_unit,
            unit_area=unit_area,
            spline_df=spline_df,
            n_boot=n_boot,
            refit_bootstrap=refit_bootstrap,
            alpha=alpha,
            seed=seed,
            localize=localize,
            hotspot_z=hotspot_z,
        )
    else:
        config = SPIRAConfig(
            technical_covariates=tuple(technical_covariates),
            context_covariates=dict(context_covariates or {}),
            ranks=tuple(ranks),
            spatial_key=spatial_key,
            coordinate_unit=coordinate_unit,
            unit_area=unit_area,
            n_neighbors=n_neighbors,
            spline_df=spline_df,
            n_boot=n_boot,
            refit_bootstrap=refit_bootstrap,
            alpha=alpha,
            seed=seed,
            localize=localize,
            hotspot_z=hotspot_z,
        )

    rank_columns = tuple(r for r in config.ranks if r != "feature")
    validation = validate(
        microbial_adata,
        spatial_key=config.spatial_key,
        taxonomy_columns=rank_columns,
    )
    obs = microbial_adata.obs.copy()
    coords = np.asarray(microbial_adata.obsm[config.spatial_key], dtype=float)[:, :2]
    technical = tuple(config.technical_covariates)
    context = dict(config.context_covariates)
    _require_obs_columns(obs, technical)

    W = graph.tocsr() if graph is not None else build_knn_graph(coords, config.n_neighbors)
    if W.shape != (len(obs), len(obs)):
        raise ValueError("Provided graph dimensions do not match microbial_adata.obs.")
    X_technical = design_matrix(obs, technical, df=config.spline_df)
    native_names = _feature_names(microbial_adata)
    raw_matrix = microbial_adata.X
    analyses = [aggregate_counts(raw_matrix, microbial_adata.var, rank="feature", feature_names=native_names)]
    analyses.extend(
        aggregate_counts(raw_matrix, microbial_adata.var, rank=rank, feature_names=native_names)
        for rank in rank_columns
    )

    rng = np.random.default_rng(config.seed)
    rows: list[dict] = []
    diagnostics: dict[str, dict] = {}
    counts_cache: dict[str, np.ndarray] = {}
    for aggregation in analyses:
        for index, taxon in enumerate(aggregation.names):
            y = _column(aggregation.matrix, index)
            support = assess_support(y, config.support)
            key = _diagnostic_key(aggregation.rank, str(taxon))
            row = {
                "rank": aggregation.rank,
                "taxon": str(taxon),
                "n_source_features": len(aggregation.members[str(taxon)]),
                **support.as_dict(),
                "spatial_I_residual": np.nan,
                "spatial_null_mean": np.nan,
                "spatial_null_sd": np.nan,
                "spatial_p": np.nan,
                "spatial_q": np.nan,
                "spatial_call": NOT_TESTABLE if not support.testable else NO_EVIDENCE,
                "fit_converged": False,
                "bootstrap_refit": config.refit_bootstrap,
            }
            if support.testable:
                test = residual_spatial_test(
                    y,
                    X_technical,
                    W,
                    n_draws=config.n_boot,
                    rng=rng,
                    refit_bootstrap=config.refit_bootstrap,
                )
                row.update(
                    {
                        "spatial_I_residual": test.observed_i,
                        "spatial_null_mean": test.null_mean,
                        "spatial_null_sd": test.null_sd,
                        "spatial_p": test.p_value,
                        "fit_converged": test.fit_converged,
                    }
                )
                baseline_fit = fit_poisson_gam(y, X_technical)
                for label, column in context.items():
                    safe = _safe_label(label)
                    association = assess_context_association(
                        y,
                        X_technical,
                        obs,
                        label=label,
                        column=column,
                        baseline_fit=baseline_fit,
                        spline_df=config.spline_df,
                    )
                    row.update(
                        {
                            f"context_{safe}_column": column,
                            f"context_{safe}_available": association.available,
                            f"context_{safe}_delta_deviance": association.delta_deviance,
                            f"context_{safe}_log_rate_range": association.log_rate_range,
                            f"context_{safe}_p": association.p_value,
                            f"context_{safe}_q": np.nan,
                            f"context_{safe}_associated": False,
                        }
                    )
                diagnostics[key] = {
                    "residual": test.residual,
                    "fitted_mean": test.fitted_mean,
                    "members": aggregation.members[str(taxon)],
                }
                counts_cache[key] = y
            else:
                for label, column in context.items():
                    safe = _safe_label(label)
                    row.update(
                        {
                            f"context_{safe}_column": column,
                            f"context_{safe}_available": False,
                            f"context_{safe}_delta_deviance": np.nan,
                            f"context_{safe}_log_rate_range": np.nan,
                            f"context_{safe}_p": np.nan,
                            f"context_{safe}_q": np.nan,
                            f"context_{safe}_associated": False,
                        }
                    )
            rows.append(row)

    taxa = pd.DataFrame(rows)
    taxa["spatial_q"] = benjamini_hochberg(taxa["spatial_p"].to_numpy())
    tested = taxa["support_tier"].ne(NOT_TESTABLE)
    taxa.loc[tested & (taxa["spatial_q"] <= config.alpha), "spatial_call"] = STRUCTURED
    taxa.loc[tested & ~(taxa["spatial_q"] <= config.alpha), "spatial_call"] = NO_EVIDENCE

    for label in context:
        safe = _safe_label(label)
        p_col, q_col, flag_col = (
            f"context_{safe}_p",
            f"context_{safe}_q",
            f"context_{safe}_associated",
        )
        taxa[q_col] = benjamini_hochberg(taxa[p_col].to_numpy())
        taxa[flag_col] = taxa[q_col] <= config.alpha

    clusters = _collect_clusters(taxa, diagnostics, counts_cache, W, coords, config)
    taxa = taxa.sort_values(["rank", "spatial_q", "taxon"], na_position="last").reset_index(drop=True)
    return SPIRAResult(
        taxa=taxa,
        clusters=clusters,
        config=config.as_dict(),
        validation=asdict(validation),
        diagnostics=diagnostics,
    )


def _collect_clusters(taxa, diagnostics, counts_cache, W, coords, config) -> pd.DataFrame:
    if not config.localize:
        return pd.DataFrame(columns=CLUSTER_COLUMNS)
    frames = []
    for row in taxa.loc[taxa["spatial_call"] == STRUCTURED].itertuples(index=False):
        key = _diagnostic_key(row.rank, row.taxon)
        diagnostic = diagnostics.get(key)
        if diagnostic is None:
            continue
        frames.append(
            localize_hotspots(
                counts_cache[key],
                diagnostic["residual"],
                W,
                coords,
                rank=row.rank,
                taxon=row.taxon,
                z_threshold=config.hotspot_z,
                unit_area=config.unit_area,
                coordinate_unit=config.coordinate_unit,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CLUSTER_COLUMNS)


def _merge_config(config: SPIRAConfig, **overrides) -> SPIRAConfig:
    """Apply explicitly supplied non-default function arguments to a config."""
    # ``run(config=...)`` is intentionally compact; users should set desired
    # values on the config itself. Passing the default function values leaves
    # the supplied config untouched.
    updates = {}
    default = SPIRAConfig()
    for name, value in overrides.items():
        if value is None:
            continue
        if value != getattr(default, name):
            updates[name] = value
    return replace(config, **updates)


def _require_obs_columns(obs: pd.DataFrame, names: Sequence[str]) -> None:
    missing = [name for name in names if name not in obs.columns]
    if missing:
        raise KeyError(f"Technical covariates absent from adata.obs: {missing}")
    for name in names:
        values = obs[name].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Technical covariate {name!r} contains non-finite values.")


def _feature_names(adata) -> pd.Index:
    if hasattr(adata, "var_names"):
        return pd.Index(adata.var_names).astype(str)
    return pd.Index(adata.var.index).astype(str)


def _column(matrix, index: int) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.getcol(index).toarray()).ravel()
    return np.asarray(matrix[:, index]).ravel()


def _safe_label(label: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(label)).strip("_")


def _diagnostic_key(rank: str, taxon: str) -> str:
    return f"{rank}::{taxon}"
