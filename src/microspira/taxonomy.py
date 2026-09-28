"""Taxonomic aggregation for species-to-phylum SPIRA reports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class AggregatedTaxa:
    rank: str
    matrix: np.ndarray | sparse.spmatrix
    names: pd.Index
    members: dict[str, tuple[str, ...]]


def aggregate_counts(X, var: pd.DataFrame, *, rank: str, feature_names) -> AggregatedTaxa:
    """Aggregate a count matrix at one requested taxonomic rank.

    Missing or empty rank labels are omitted rather than silently assigned to a
    misleading parent. The omitted-feature count is recoverable from the input
    report and can later be surfaced explicitly in a user-facing warning.
    """
    names = pd.Index(feature_names).astype(str)
    if rank == "feature":
        return AggregatedTaxa(
            rank="feature",
            matrix=X,
            names=names,
            members={name: (name,) for name in names},
        )
    if rank not in var.columns:
        raise KeyError(f"Taxonomic rank {rank!r} is absent from adata.var.")

    labels = var[rank].astype("string")
    valid = labels.notna() & labels.str.strip().ne("")
    if not valid.any():
        raise ValueError(f"Taxonomic rank {rank!r} has no non-empty labels.")
    source_positions = np.flatnonzero(valid.to_numpy())
    source_labels = labels[valid].astype(str).to_numpy()
    group_names = pd.Index(pd.unique(source_labels))
    group_codes = pd.Categorical(source_labels, categories=group_names).codes

    membership = sparse.csr_matrix(
        (
            np.ones(len(source_positions), dtype=float),
            (source_positions, group_codes),
        ),
        shape=(len(names), len(group_names)),
    )
    matrix = X @ membership
    members: dict[str, tuple[str, ...]] = {}
    for label in group_names:
        members[str(label)] = tuple(names[(labels.astype(str) == label).to_numpy()])
    return AggregatedTaxa(
        rank=rank,
        matrix=matrix,
        names=group_names.astype(str),
        members=members,
    )


def available_ranks(var: pd.DataFrame) -> list[str]:
    """Return conventional taxonomic rank columns present in a var table."""
    conventional = ["species", "genus", "family", "order", "class", "phylum"]
    return [rank for rank in conventional if rank in var.columns]

