"""Context association across independent analysis domains (sign test)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .api import _safe_label
from .spatial import benjamini_hochberg


REPLICATE_COLUMNS = [
    "rank", "taxon", "context", "n_domains_total", "n_domains_available",
    "n_units_compared", "n_positive", "n_negative", "median_log_contrast",
    "consensus_direction", "sign_test_p", "sign_test_q", "replicated",
]


def replicate_context(
    results: Sequence,
    label: str,
    *,
    groups: Sequence | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Two-sided exact sign test of context direction across independent domains.

    ``results`` holds one :class:`SPIRAResult` per independent analysis domain
    (one tissue section per ``run`` call), all run with the same context
    definition and orientation. With ``groups`` (e.g. animal IDs, one per
    result), contrasts are averaged within each group first and groups are the
    compared units. Taxa that are not testable or have no available contrast
    in a domain are excluded for that domain. BH adjustment is applied across
    all taxon rows (all ranks) with a finite sign-test p-value.

    The null assumes independent compared units and equally probable positive
    and negative nonzero contrasts. Availability filtering must not select a
    preferred direction. Independence and matching context orientation are
    analysis-design requirements; this function cannot establish them.

    At least six concordant units are needed for a two-sided p below 0.05
    (2 x 0.5**6 = 0.03125); five concordant units give 0.0625. This is a raw
    p-value threshold: six units do not guarantee significance after BH.
    Exact-zero contrasts do not enter the sign test. Taxa with no available
    domain are absent from the returned table; available all-zero contrasts
    give an unavailable p/q and ``replicated=False``.
    """
    if not len(results):
        raise ValueError("At least one SPIRAResult is required.")
    if groups is not None and len(groups) != len(results):
        raise ValueError("groups must have one entry per result.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1).")
    safe = _safe_label(label)
    contrast_col, available_col = f"context_{safe}_log_contrast", f"context_{safe}_available"
    frames = []
    for d, result in enumerate(results):
        taxa = result.taxa
        if contrast_col not in taxa.columns:
            raise KeyError(f"Result {d} has no context {label!r} descriptors.")
        keep = taxa[available_col].astype(bool) & np.isfinite(taxa[contrast_col].astype(float))
        frame = taxa.loc[keep, ["rank", "taxon", contrast_col]].copy()
        frame.columns = ["rank", "taxon", "log_contrast"]
        frame["unit"] = groups[d] if groups is not None else d
        frame["domain"] = d
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    rows = []
    for (rank, taxon), sub in long.groupby(["rank", "taxon"], sort=True):
        unit_contrast = sub.groupby("unit")["log_contrast"].mean()
        signs = np.sign(unit_contrast.to_numpy())
        n_pos, n_neg = int((signs > 0).sum()), int((signs < 0).sum())
        n = n_pos + n_neg
        p = float(binomtest(max(n_pos, n_neg), n, 0.5, alternative="two-sided").pvalue) if n else np.nan
        rows.append({
            "rank": rank, "taxon": taxon, "context": label,
            "n_domains_total": len(results),
            "n_domains_available": int(sub["domain"].nunique()),
            "n_units_compared": n, "n_positive": n_pos, "n_negative": n_neg,
            "median_log_contrast": float(np.median(unit_contrast)),
            "consensus_direction": "+" if n_pos > n_neg else ("-" if n_neg > n_pos else "tie"),
            "sign_test_p": p,
        })
    out = pd.DataFrame(rows, columns=REPLICATE_COLUMNS[:-2])
    out["sign_test_q"] = benjamini_hochberg(out["sign_test_p"].to_numpy()) if len(out) else []
    out["replicated"] = out["sign_test_q"] <= alpha
    return out.sort_values(["sign_test_q", "rank", "taxon"], na_position="last").reset_index(drop=True)
