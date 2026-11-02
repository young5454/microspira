import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

import microspira as ms
from microspira.clusters import localize_hotspots, neighborhood_operator, neighborhood_enrichment
from microspira.graph import build_knn_graph
from microspira.model import fit_poisson_gam, residual_spatial_test


class Carrier:
    def __init__(self, y, coords):
        self.X = np.asarray(y)[:, None]
        self.obs = pd.DataFrame(index=[f"u{i}" for i in range(len(y))])
        self.var = pd.DataFrame(index=["taxon"])
        self.var_names = self.var.index
        self.obsm = {"spatial": coords}


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        xx, yy = np.meshgrid(np.arange(10), np.arange(10))
        self.coords = np.column_stack([xx.ravel(), yy.ravel()])
        self.W = build_knn_graph(self.coords, 4)

    def test_profiled_solver_matches_independent_conditional_optimizer(self):
        rng = np.random.default_rng(812)
        z = np.linspace(-1, 1, 150)
        X = np.column_stack([np.ones(len(z)), z, z * z])
        eta = 1.2 * z - 0.7 * z * z
        y = rng.multinomial(30, np.exp(eta - logsumexp(eta)))
        fit = fit_poisson_gam(y, X)
        def objective(beta):
            logp = X[:, 1:] @ beta
            logp -= logsumexp(logp)
            return -y @ logp + .5e-6 * beta @ beta
        reference = minimize(objective, np.zeros(2), method="BFGS", tol=1e-9)
        p = np.exp(X[:, 1:] @ reference.x - logsumexp(X[:, 1:] @ reference.x))
        self.assertTrue(fit.converged)
        np.testing.assert_allclose(fit.mu, 30 * p, atol=1e-5)
        self.assertLess(fit.scaled_score, 1e-7)
        self.assertAlmostEqual(fit.mu.sum(), 30)

    def test_large_offset_is_stable_without_clipped_exposure(self):
        offset = np.linspace(1000, 1004, 100)
        fit = fit_poisson_gam(np.ones(100), np.ones((100, 1)), offset=offset)
        expected = 100 * np.exp(offset - logsumexp(offset))
        self.assertTrue(fit.converged)
        np.testing.assert_allclose(fit.mu, expected)

    def test_invalid_bootstrap_fit_is_not_silently_discarded(self):
        y = np.where(self.coords[:, 0] < 3, 8, 0)
        original = fit_poisson_gam
        counter = [0]
        def one_failure(*args, **kwargs):
            fit = original(*args, **kwargs)
            counter[0] += 1
            return replace(fit, converged=False) if counter[0] == 3 else fit
        with patch("microspira.model.fit_poisson_gam", side_effect=one_failure):
            result = residual_spatial_test(y, np.ones((100, 1)), self.W,
                n_draws=19, refit_bootstrap=True, localize=True, rng=np.random.default_rng(8))
        self.assertFalse(result.inference_valid)
        self.assertTrue(np.isnan(result.p_value))
        self.assertEqual(result.bootstrap_fit_failures, 1)
        self.assertEqual(result.bootstrap_n_valid, 18)
        self.assertTrue(np.isnan(result.local_p).all())

    def test_observed_fit_failure_has_an_explicit_api_state(self):
        y = np.where(self.coords[:, 0] < 3, 8, 0)
        original = fit_poisson_gam
        with patch("microspira.model.fit_poisson_gam",
                   side_effect=lambda *a, **k: replace(original(*a, **k), converged=False)):
            report = ms.run(Carrier(y, self.coords), graph=self.W, n_boot=19)
        self.assertEqual(report.taxa.spatial_call.iloc[0], "INFERENCE_FAILED")
        self.assertTrue(np.isnan(report.taxa.spatial_q.iloc[0]))

    def test_single_read_is_not_a_supported_local_hotspot(self):
        y = np.zeros(100)
        y[44] = 1
        A = neighborhood_operator(self.W)
        _, nr, ne = neighborhood_enrichment(y, np.full(100, .01), A)
        # Even a supplied small local p cannot override the count-support rule.
        clusters = localize_hotspots(y, y * 100, self.W, self.coords,
            rank="feature", taxon="one_read", local_p=np.full(100, .001),
            neighborhood_reads=nr, neighborhood_expected=ne)
        self.assertEqual(len(clusters), 0)

    def test_strong_cluster_has_calibrated_neighborhood_candidates(self):
        y = ((self.coords[:, 0] < 3) & (self.coords[:, 1] < 3)).astype(int) * 80
        report = ms.run(Carrier(y, self.coords), graph=self.W,
                        n_boot=99, refit_bootstrap=True, seed=18)
        self.assertEqual(report.taxa.spatial_call.iloc[0], "SPATIALLY_STRUCTURED")
        self.assertGreater(len(report.clusters), 0)
        self.assertTrue(report.clusters.component_supported.all())
        self.assertTrue(report.clusters.localization_method.eq("bootstrap_neighborhood_max").all())
        self.assertTrue(report.clusters.equivalent_diameter.isna().all())
        self.assertTrue(report.clusters.diameter_unit.eq("not_available").all())

    def test_intercept_only_fast_and_refit_use_identical_nulls(self):
        y = ((self.coords[:, 0] < 3) & (self.coords[:, 1] < 3)).astype(int) * 20
        kwargs = dict(counts=y, X=np.ones((100, 1)), W=self.W, n_draws=39)
        fast = residual_spatial_test(**kwargs, rng=np.random.default_rng(48))
        refit = residual_spatial_test(**kwargs, refit_bootstrap=True, rng=np.random.default_rng(48))
        self.assertEqual(fast.p_value, refit.p_value)
        self.assertAlmostEqual(fast.null_mean, refit.null_mean)


if __name__ == "__main__":
    unittest.main()
