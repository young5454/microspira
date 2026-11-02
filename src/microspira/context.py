"""Contextual descriptors that never delete a spatial call.

Within one analysis domain, spatial units are dependent and contextual
covariates are usually spatially smooth, so an independence-based reference
distribution (for example chi-square on the deviance difference) is not
calibrated for spatially structured taxa. Each call therefore reports only
descriptors: the deviance improvement and a signed decile contrast of the
fitted context term. Association is assessed across independent domains with
:func:`microspira.replicate_context`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import PoissonFit, fit_poisson_gam, spline_basis


@dataclass(frozen=True)
class ContextAssociation:
    """Descriptive association between a taxon's counts and one context."""

    label: str
    column: str
    delta_deviance: float
    degrees_of_freedom: int
    log_rate_range: float
    log_contrast: float = np.nan
    direction: int = 0
    available: bool = True
    message: str = ""


def decile_contrast(context_values: np.ndarray, context_effect: np.ndarray,
                    lower: float = 0.1, upper: float = 0.9) -> float:
    """Signed contrast of a fitted context term between context extremes.

    Mean of the fitted term over units with context >= the ``upper`` quantile
    minus its mean over units with context <= the ``lower`` quantile, on the
    natural-log rate scale. Positive values mean higher counts at high context
    values. Returns NaN when the two groups cannot be separated.
    """
    x = np.asarray(context_values, dtype=float).ravel()
    effect = np.asarray(context_effect, dtype=float).ravel()
    lo, hi = np.quantile(x, lower), np.quantile(x, upper)
    if not hi > lo:
        return np.nan
    top, bottom = x >= hi, x <= lo
    if not top.any() or not bottom.any():
        return np.nan
    return float(effect[top].mean() - effect[bottom].mean())


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
    """Fit a nested mean model and report descriptive context summaries.

    No p-value is produced here. The deviance difference and the signed
    decile contrast describe count association conditional on the selected
    technical design; they do not establish causality.
    """
    unavailable = dict(label=label, column=column, delta_deviance=np.nan,
                       degrees_of_freedom=0, log_rate_range=np.nan,
                       log_contrast=np.nan, direction=0, available=False)
    if column not in obs.columns:
        return ContextAssociation(**unavailable, message=f"obs column {column!r} is unavailable")

    values = obs[column].to_numpy(dtype=float)
    context_basis = spline_basis(values, df=spline_df)
    if context_basis.shape[1] == 0:
        return ContextAssociation(**unavailable, message="context covariate is constant")

    base = baseline_fit or fit_poisson_gam(counts, technical_X)
    augmented_X = np.hstack([technical_X, context_basis])
    augmented = fit_poisson_gam(counts, augmented_X)
    df = int(context_basis.shape[1])
    if not base.converged or not augmented.converged:
        return ContextAssociation(**{**unavailable, "degrees_of_freedom": df},
                                  message="context model did not converge")
    delta = max(0.0, float(base.deviance - augmented.deviance))

    context_effect = context_basis @ augmented.coefficients[-df:]
    log_rate_range = float(
        np.quantile(context_effect, 0.95) - np.quantile(context_effect, 0.05)
    )
    contrast = decile_contrast(values, context_effect)
    if not np.isfinite(contrast):
        return ContextAssociation(label=label, column=column, delta_deviance=delta,
            degrees_of_freedom=df, log_rate_range=log_rate_range, log_contrast=np.nan,
            direction=0, available=False, message="context deciles cannot be separated")
    return ContextAssociation(
        label=label,
        column=column,
        delta_deviance=delta,
        degrees_of_freedom=df,
        log_rate_range=log_rate_range,
        log_contrast=contrast,
        direction=int(np.sign(contrast)),
        available=True,
    )
