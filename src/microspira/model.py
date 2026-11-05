"""Spline Poisson mean models and conditional residual bootstrap tests.

The implementation deliberately avoids a mandatory heavy modelling dependency.
It profiles the free intercept and solves the equivalent ridge-stabilised
conditional likelihood with Newton steps. Total counts are preserved exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.special import logsumexp

from .spatial import morans_i
from .clusters import neighborhood_operator, neighborhood_enrichment
from .dispersion import (OVERDISPERSION_METHODS, draw_null, graph_edges, neighbour_dispersion,
                         overdispersed_residual)


@dataclass(frozen=True)
class PoissonFit:
    mu: np.ndarray
    coefficients: np.ndarray
    deviance: float
    converged: bool
    n_iter: int
    message: str = ""
    objective: float = np.nan
    objective_relative_change: float = np.nan
    mean_relative_change: float = np.nan
    scaled_score: float = np.nan


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
    fit: PoissonFit | None = None
    inference_valid: bool = False
    bootstrap_n_valid: int = 0
    bootstrap_fit_failures: int = 0
    bootstrap_invalid_statistics: int = 0
    bootstrap_max_n_iter: int = 0
    local_score: np.ndarray | None = None
    local_p: np.ndarray | None = None
    neighborhood_reads: np.ndarray | None = None
    neighborhood_expected: np.ndarray | None = None
    failure_reason: str = ""
    overdispersion: str = "none"
    dispersion_kappa: float = 0.0
    dispersion_null_median: float = np.nan


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
    max_iter: int = 200,
    tol: float = 1e-8,
    ridge: float = 1e-6,
) -> PoissonFit:
    """Fit a Poisson spline mean with its free intercept profiled out.

    This is the same intercept-plus-slope Poisson likelihood with a ridge on
    slopes only. Log-sum-exp avoids clipped rates; the stopping rule requires
    a small scaled score and stable fitted means/objective. Large changes in
    poorly identified coefficients alone are not treated as fitting failures.
    """
    y = np.asarray(counts, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X must be a two-dimensional matrix aligned to counts.")
    if np.any(y < 0) or not np.all(np.isfinite(y)):
        raise ValueError("counts must be finite and non-negative.")
    if X.shape[1] < 1 or not np.all(np.isfinite(X)) or not np.allclose(X[:, 0], 1.0):
        raise ValueError("X must be finite and its first column must be an intercept of ones.")
    if max_iter < 1 or tol <= 0 or ridge < 0 or not np.isfinite([tol, ridge]).all():
        raise ValueError("max_iter/tol must be positive and ridge must be finite and non-negative.")
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

    total = float(y.sum())
    Z = X[:, 1:]
    theta = np.zeros(Z.shape[1])

    def evaluate(slopes):
        eta = off + Z @ slopes
        normalizer = logsumexp(eta)
        log_probability = eta - normalizer
        probability = np.exp(log_probability)
        mu = total * probability
        objective = float(-y @ log_probability + 0.5 * ridge * np.dot(slopes, slopes))
        score = Z.T @ (mu - y) + ridge * slopes
        return objective, mu, score, normalizer

    objective, mu, score, normalizer = evaluate(theta)
    objective_change = mean_change = 0.0
    scaled_score = float(np.max(np.abs(score), initial=0.0) / max(total, 1.0))
    converged = Z.shape[1] == 0
    message = "intercept profiled" if converged else "maximum iterations reached"
    iteration = 0
    score_tolerance = max(1e-7, 10.0 * tol)
    for iteration in range(1, max_iter + 1) if not converged else ():
        probability = mu / total
        mean_Z = probability @ Z
        hessian = Z.T @ (Z * mu[:, None]) - total * np.outer(mean_Z, mean_Z)
        hessian += np.eye(Z.shape[1]) * ridge
        try:
            direction = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(hessian, score, rcond=None)[0]
        if not np.isfinite(direction).all():
            message = "non-finite Newton direction"
            break
        step, accepted = 1.0, False
        allowance = 1e-12 * (1.0 + abs(objective))
        for _ in range(30):
            proposed = theta - step * direction
            new_objective, new_mu, new_score, new_normalizer = evaluate(proposed)
            if np.isfinite(new_objective) and new_objective <= objective + allowance:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            message = "line search did not find an admissible step"
            break
        objective_change = abs(new_objective - objective) / (1.0 + abs(objective))
        mean_change = float(np.abs(new_mu - mu).sum() / max(total, 1.0))
        theta, objective, mu, score, normalizer = proposed, new_objective, new_mu, new_score, new_normalizer
        scaled_score = float(np.max(np.abs(score), initial=0.0) / max(total, 1.0))
        if scaled_score <= score_tolerance and objective_change <= tol and mean_change <= tol:
            converged, message = True, "stable mean/objective and small scaled score"
            break
    beta = np.r_[np.log(total) - normalizer, theta]
    return PoissonFit(mu=mu, coefficients=beta, deviance=poisson_deviance(y, mu),
        converged=converged, n_iter=iteration, message=message, objective=objective,
        objective_relative_change=objective_change, mean_relative_change=mean_change,
        scaled_score=scaled_score)


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
    localize: bool = False,
    overdispersion: str = "none",
    edges: tuple[np.ndarray, np.ndarray] | None = None,
) -> ResidualSpatialTest:
    """Test positive residual spatial clustering under a conditional null.

    ``refit_bootstrap=True`` is slower but captures mean-model fitting
    uncertainty and should be used for final calibration. The fast mode is
    useful while exploring a dataset and is labelled in the returned object.

    ``overdispersion="neighbour"`` (the ``run()`` default since 0.1.0.dev3) allows
    independent unit-level Gamma heterogeneity around the technical mean: the
    dispersion is estimated from neighbour differences, residuals are scaled by
    the overdispersed variance, null draws include Gamma factors, and the
    dispersion is re-estimated on every draw. ``"none"`` is the dev2 multinomial
    null and is this low-level function's default.
    """
    if n_draws < 19:
        raise ValueError("n_draws must be at least 19 for a meaningful bootstrap.")
    if overdispersion not in OVERDISPERSION_METHODS:
        raise ValueError(f"overdispersion must be one of {OVERDISPERSION_METHODS}.")
    estimate = overdispersion == "neighbour"
    if estimate and edges is None:
        edges = graph_edges(W)
    generator = rng or np.random.default_rng()
    y = np.asarray(counts, dtype=float).ravel()
    fit = fit_poisson_gam(y, X, offset=offset)
    fitted_total = _normalise_mu(fit.mu, y.sum())
    kappa = neighbour_dispersion(y, fitted_total, edges) if (estimate and fit.converged) else 0.0
    residual = overdispersed_residual(y, fitted_total, kappa) if estimate else conditional_residual(y, fit.mu)
    observed = morans_i(residual, W)
    if not np.isfinite(observed) or not fit.converged:
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
            fit=fit,
            failure_reason="observed_fit_not_converged" if not fit.converged else "observed_statistic_non_finite",
            overdispersion=overdispersion,
            dispersion_kappa=kappa,
        )

    probability = fit.mu / fit.mu.sum()
    null_i = np.empty(n_draws, dtype=float)
    local_operator = neighborhood_operator(W) if localize else None
    local_score = neighborhood_reads = neighborhood_expected = None
    local_maxima = np.empty(n_draws) if localize else None
    if localize:
        local_score, neighborhood_reads, neighborhood_expected = neighborhood_enrichment(y, fit.mu, local_operator)
    fit_failures = 0
    maximum_iterations = 0
    fit_valid = np.ones(n_draws, dtype=bool)
    null_kappa = np.zeros(n_draws)
    total = int(np.rint(y.sum()))
    for b in range(n_draws):
        draw = draw_null(generator, total, probability, kappa)
        if refit_bootstrap:
            null_fit = fit_poisson_gam(draw, X, offset=offset)
            maximum_iterations = max(maximum_iterations, null_fit.n_iter)
            fit_valid[b] = null_fit.converged
            fit_failures += int(not null_fit.converged)
            null_mu = null_fit.mu
        else:
            null_mu = fit.mu
        if estimate:
            null_total = _normalise_mu(null_mu, draw.sum())
            null_kappa[b] = neighbour_dispersion(draw, null_total, edges)
            null_residual = overdispersed_residual(draw, null_total, null_kappa[b])
        else:
            null_residual = conditional_residual(draw, null_mu)
        null_i[b] = morans_i(null_residual, W)
        if localize:
            scores, _, _ = neighborhood_enrichment(draw, null_mu, local_operator)
            local_maxima[b] = np.max(scores)

    finite = np.isfinite(null_i)
    if localize:
        finite &= np.isfinite(local_maxima)
    valid = finite & fit_valid
    inference_valid = bool(valid.all())
    valid_i = null_i[valid]
    if not len(valid_i):
        p_value = np.nan
        null_mean = np.nan
        null_sd = np.nan
    else:
        p_value = float((1 + np.sum(null_i >= observed)) / (1 + n_draws)) if inference_valid else np.nan
        null_mean = float(valid_i.mean())
        null_sd = float(valid_i.std(ddof=1)) if len(valid_i) > 1 else 0.0
    local_p = None
    if localize:
        local_p = np.full_like(local_score, np.nan)
        if inference_valid:
            # Sort maxima once: no B x n_units dense comparison matrix.
            position = np.searchsorted(np.sort(local_maxima), local_score, side="left")
            local_p = (1.0 + n_draws - position) / (1.0 + n_draws)

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
        fit=fit,
        inference_valid=inference_valid,
        bootstrap_n_valid=int(valid.sum()),
        bootstrap_fit_failures=fit_failures,
        bootstrap_invalid_statistics=int((~finite).sum()),
        bootstrap_max_n_iter=maximum_iterations,
        local_score=local_score,
        local_p=local_p,
        neighborhood_reads=neighborhood_reads,
        neighborhood_expected=neighborhood_expected,
        failure_reason=",".join(reason for condition, reason in (
            (fit_failures > 0, "bootstrap_fit_not_converged"),
            ((~finite).any(), "bootstrap_statistic_non_finite")) if condition),
        overdispersion=overdispersion,
        dispersion_kappa=kappa,
        dispersion_null_median=float(np.median(null_kappa)) if estimate else 0.0,
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
