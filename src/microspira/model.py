"""Spline Poisson mean models and conditional residual bootstrap tests.

The implementation deliberately avoids a mandatory heavy modelling dependency.
It uses a small ridge-stabilised Poisson IRLS solver because SPIRA needs many
taxon-wise fits with a modest number of spline columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.interpolate import BSpline

from .spatial import morans_i


@dataclass(frozen=True)
class PoissonFit:
    mu: np.ndarray
    coefficients: np.ndarray
    deviance: float
    converged: bool
    n_iter: int
    message: str = ""


@dataclass(frozen=True)
class ResidualSpatialTest:
    observed_i: float
    null_mean: float
    null_sd: float
    p_value: float
    residual: np.ndarray
    fitted_mean: np.ndarray
    fit_converged: bool
    bootstrap_refit: bool
    n_draws: int


def spline_basis(values: np.ndarray, df: int = 3) -> np.ndarray:
    """Return a compact cubic B-spline basis with quantile knots.

    Degenerate or low-cardinality covariates fall back to one z-scored linear
    term. The intercept is deliberately not included here.
    """
    x = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("Spline covariates must be finite.")
    if df < 1:
        raise ValueError("df must be at least one.")
    unique = np.unique(x)
    if unique.size <= 1:
        return np.empty((x.size, 0), dtype=float)
    if unique.size < df + 2:
        return _zscore(x)[:, None]

    knots = np.unique(np.quantile(x, np.linspace(0.0, 1.0, df + 1)))
    if knots.size < 4:
        return _zscore(x)[:, None]
    t = np.r_[[knots[0]] * 3, knots, [knots[-1]] * 3]
    basis = BSpline.design_matrix(
        np.clip(x, knots[0], knots[-1]), t, k=3, extrapolate=False
    ).toarray()
    return basis[:, 1:]


def design_matrix(obs: pd.DataFrame, covariates: tuple[str, ...] | list[str], df: int = 3) -> np.ndarray:
    """Build an intercept-plus-spline design matrix from selected columns."""
    cols = [np.ones((len(obs), 1), dtype=float)]
    for name in covariates:
        if name not in obs.columns:
            raise KeyError(f"Covariate {name!r} is absent from adata.obs.")
        basis = spline_basis(obs[name].to_numpy(dtype=float), df=df)
        if basis.shape[1]:
            cols.append(basis)
    return np.hstack(cols)


def conditional_residual(counts: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Residual standardised for a total-read-conditioned multinomial null."""
    y = np.asarray(counts, dtype=float).ravel()
    fitted = _normalise_mu(mu, y.sum())
    if y.sum() <= 0:
        return np.zeros_like(y)
    p = fitted / fitted.sum()
    variance = y.sum() * p * (1.0 - p)
    return (y - fitted) / np.sqrt(np.clip(variance, 1e-10, None))


def fit_poisson_gam(
    counts: np.ndarray,
    X: np.ndarray,
    *,
    offset: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    ridge: float = 1e-6,
) -> PoissonFit:
    """Fit a ridge-stabilised Poisson GLM by iteratively reweighted least squares."""
    y = np.asarray(counts, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X must be a two-dimensional matrix aligned to counts.")
    if np.any(y < 0) or not np.all(np.isfinite(y)):
        raise ValueError("counts must be finite and non-negative.")
    if y.sum() <= 0:
        return PoissonFit(
            mu=np.zeros_like(y),
            coefficients=np.zeros(X.shape[1]),
            deviance=0.0,
            converged=True,
            n_iter=0,
            message="zero total count",
        )

    if offset is None:
        off = np.zeros_like(y)
    else:
        off = np.asarray(offset, dtype=float).ravel()
        if off.size != y.size or not np.all(np.isfinite(off)):
            raise ValueError("offset must be finite and aligned to counts.")

    beta = np.zeros(X.shape[1], dtype=float)
    exposure = np.exp(np.clip(off, -20.0, 20.0))
    beta[0] = np.log(max(y.sum() / exposure.sum(), 1e-12))
    penalty = np.eye(X.shape[1]) * ridge
    penalty[0, 0] = 0.0
    previous_deviance = np.inf

    for iteration in range(1, max_iter + 1):
        eta = np.clip(off + X @ beta, -30.0, 30.0)
        mu = np.exp(eta)
        weights = np.clip(mu, 1e-10, None)
        working = eta + (y - mu) / weights

        lhs = X.T @ (X * weights[:, None]) + penalty
        rhs = X.T @ (weights * (working - off))
        try:
            candidate = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            candidate = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        # A conservative half-step prevents rare taxa from making unstable IRLS jumps.
        step = 1.0
        candidate_deviance = np.inf
        while step >= 1.0 / 64.0:
            proposed = beta + step * (candidate - beta)
            proposed_mu = np.exp(np.clip(off + X @ proposed, -30.0, 30.0))
            candidate_deviance = poisson_deviance(y, proposed_mu)
            if candidate_deviance <= previous_deviance + 1e-8:
                break
            step *= 0.5

        beta_new = beta + step * (candidate - beta)
        change = float(np.max(np.abs(beta_new - beta)))
        beta = beta_new
        previous_deviance = candidate_deviance
        if change < tol:
            mu = np.exp(np.clip(off + X @ beta, -30.0, 30.0))
            return PoissonFit(
                mu=_normalise_mu(mu, y.sum()),
                coefficients=beta,
                deviance=poisson_deviance(y, mu),
                converged=True,
                n_iter=iteration,
            )

    mu = np.exp(np.clip(off + X @ beta, -30.0, 30.0))
    return PoissonFit(
        mu=_normalise_mu(mu, y.sum()),
        coefficients=beta,
        deviance=poisson_deviance(y, mu),
        converged=False,
        n_iter=max_iter,
        message="maximum iterations reached",
    )


def poisson_deviance(counts: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance with the conventional zero-count limit."""
    y = np.asarray(counts, dtype=float)
    fitted = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
    nonzero = y > 0
    terms = np.empty_like(y, dtype=float)
    terms[nonzero] = y[nonzero] * np.log(y[nonzero] / fitted[nonzero]) - (y[nonzero] - fitted[nonzero])
    terms[~nonzero] = fitted[~nonzero]
    return float(2.0 * terms.sum())


def residual_spatial_test(
    counts: np.ndarray,
    X: np.ndarray,
    W: sparse.csr_matrix,
    *,
    n_draws: int = 199,
    rng: np.random.Generator | None = None,
    refit_bootstrap: bool = False,
    offset: np.ndarray | None = None,
) -> ResidualSpatialTest:
    """Test positive residual spatial clustering under a conditional null.

    ``refit_bootstrap=True`` is slower but captures mean-model fitting
    uncertainty and should be used for final calibration. The fast mode is
    useful while exploring a dataset and is labelled in the returned object.
    """
    if n_draws < 19:
        raise ValueError("n_draws must be at least 19 for a meaningful bootstrap.")
    generator = rng or np.random.default_rng()
    y = np.asarray(counts, dtype=float).ravel()
    fit = fit_poisson_gam(y, X, offset=offset)
    residual = conditional_residual(y, fit.mu)
    observed = morans_i(residual, W)
    if not np.isfinite(observed):
        return ResidualSpatialTest(
            observed_i=np.nan,
            null_mean=np.nan,
            null_sd=np.nan,
            p_value=np.nan,
            residual=residual,
            fitted_mean=fit.mu,
            fit_converged=fit.converged,
            bootstrap_refit=refit_bootstrap,
            n_draws=n_draws,
        )

    probability = fit.mu / fit.mu.sum()
    null_i = np.empty(n_draws, dtype=float)
    for b in range(n_draws):
        draw = generator.multinomial(int(np.rint(y.sum())), probability).astype(float)
        if refit_bootstrap:
            null_fit = fit_poisson_gam(draw, X, offset=offset)
            null_residual = conditional_residual(draw, null_fit.mu)
        else:
            null_residual = conditional_residual(draw, fit.mu)
        null_i[b] = morans_i(null_residual, W)

    valid = np.isfinite(null_i)
    null_i = null_i[valid]
    if not len(null_i):
        p_value = np.nan
        null_mean = np.nan
        null_sd = np.nan
    else:
        p_value = float((1 + np.sum(null_i >= observed)) / (1 + len(null_i)))
        null_mean = float(null_i.mean())
        null_sd = float(null_i.std(ddof=1)) if len(null_i) > 1 else 0.0

    return ResidualSpatialTest(
        observed_i=float(observed),
        null_mean=null_mean,
        null_sd=null_sd,
        p_value=p_value,
        residual=residual,
        fitted_mean=fit.mu,
        fit_converged=fit.converged,
        bootstrap_refit=refit_bootstrap,
        n_draws=n_draws,
    )


def _normalise_mu(mu: np.ndarray, total: float) -> np.ndarray:
    fitted = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
    if total <= 0:
        return np.zeros_like(fitted)
    return fitted * (total / fitted.sum())


def _zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()
