"""Contextual association annotations that never delete a spatial call."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .model import PoissonFit, design_matrix, fit_poisson_gam, spline_basis


@dataclass(frozen=True)
class ContextAssociation:
    """Association between a taxon's counts and one user-labelled context."""

    label: str
    column: str
    delta_deviance: float
    degrees_of_freedom: int
    p_value: float
    log_rate_range: float
    available: bool = True
    message: str = ""


def assess_context_association(
    counts: np.ndarray,
    technical_X: np.ndarray,
    obs: pd.DataFrame,
    *,
    label: str,
    column: str,
    baseline_fit: PoissonFit | None = None,
    spline_df: int = 3,
) -> ContextAssociation:
    """Fit a nested mean model and report a descriptive context association.

    The likelihood-ratio p-value quantifies count association conditional on the
    selected technical design. It does *not* establish causality or prove that
    the spatial pattern is explained by the context; SPIRA exposes it as a
    report flag only.
    """
    if column not in obs.columns:
        return ContextAssociation(
            label=label,
            column=column,
            delta_deviance=np.nan,
            degrees_of_freedom=0,
            p_value=np.nan,
            log_rate_range=np.nan,
            available=False,
            message=f"obs column {column!r} is unavailable",
        )

    context_basis = spline_basis(obs[column].to_numpy(dtype=float), df=spline_df)
    if context_basis.shape[1] == 0:
        return ContextAssociation(
            label=label,
            column=column,
            delta_deviance=0.0,
            degrees_of_freedom=0,
            p_value=1.0,
            log_rate_range=0.0,
            available=False,
            message="context covariate is constant",
        )

    base = baseline_fit or fit_poisson_gam(counts, technical_X)
    augmented_X = np.hstack([technical_X, context_basis])
    augmented = fit_poisson_gam(counts, augmented_X)
    if not base.converged or not augmented.converged:
        return ContextAssociation(label=label, column=column, delta_deviance=np.nan,
            degrees_of_freedom=context_basis.shape[1], p_value=np.nan,
            log_rate_range=np.nan, available=False, message="context model did not converge")
    delta = max(0.0, float(base.deviance - augmented.deviance))
    df = int(context_basis.shape[1])
    p_value = float(chi2.sf(delta, df)) if df else 1.0

    context_effect = context_basis @ augmented.coefficients[-df:]
    log_rate_range = float(
        np.quantile(context_effect, 0.95) - np.quantile(context_effect, 0.05)
    )
    return ContextAssociation(
        label=label,
        column=column,
        delta_deviance=delta,
        degrees_of_freedom=df,
        p_value=p_value,
        log_rate_range=log_rate_range,
        available=True,
    )

