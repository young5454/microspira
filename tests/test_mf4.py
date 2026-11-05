"""Tests for MF4: unit-level overdispersion in the conditional technical null."""

import inspect
import unittest

import numpy as np
import pandas as pd

import microspira as ms
from microspira.config import SPIRAConfig
from microspira.dispersion import draw_null, graph_edges, neighbour_dispersion, overdispersed_residual
from microspira.graph import build_knn_graph
from microspira.model import _normalise_mu, conditional_residual, residual_spatial_test


def grid(side):
    xx, yy = np.meshgrid(np.arange(side), np.arange(side))
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(float)


class Carrier:
    def __init__(self, Y, coords, obs=None):
        n = len(coords)
        self.X = np.asarray(Y)
        self.obs = obs if obs is not None else pd.DataFrame(
            {"depth": np.log1p(np.arange(n) % 7)}, index=[f"u{i}" for i in range(n)])
        self.var = pd.DataFrame(index=[f"t{j}" for j in range(self.X.shape[1])])
        self.var_names = self.var.index
        self.obsm = {"spatial": coords}


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        self.coords = grid(40)
        self.W = build_knn_graph(self.coords, 4)
        self.edges = graph_edges(self.W)
        self.n = len(self.coords)
        self.rng = np.random.default_rng(1)

    def _nb2(self, kappa, mean=5.0):
        p = np.full(self.n, 1 / self.n)
        g = self.rng.gamma(1 / kappa, kappa, self.n) if kappa > 0 else np.ones(self.n)
        w = p * g
        y = self.rng.multinomial(int(mean * self.n), w / w.sum()).astype(float)
        return y, np.full(self.n, mean)

    def test_recovers_kappa_for_independent_heterogeneity(self):
        for kappa, tol in ((0.0, 0.03), (0.5, 0.12), (2.0, 0.5)):
            y, mu = self._nb2(kappa)
            self.assertAlmostEqual(neighbour_dispersion(y, mu, self.edges), kappa, delta=tol)

    def test_smooth_gradient_is_not_read_as_overdispersion(self):
        x = self.coords[:, 0]
        lam = np.exp(0.06 * x)
        y = self.rng.multinomial(8000, lam / lam.sum()).astype(float)
        mu = np.full(self.n, y.sum() / self.n)                # technical model ignores the gradient
        self.assertLess(neighbour_dispersion(y, mu, self.edges), 0.03)

    def test_estimate_is_non_negative_and_finite_with_tiny_means(self):
        y = np.zeros(self.n)
        y[0] = 3
        mu = np.full(self.n, 1e-12)
        mu[0] = 3.0
        k = neighbour_dispersion(y, mu, self.edges)
        self.assertTrue(np.isfinite(k) and k >= 0)


class Dev2CompatibilityTests(unittest.TestCase):
    def test_residual_with_zero_kappa_equals_dev2_residual(self):
        rng = np.random.default_rng(2)
        y = rng.poisson(2.0, 100).astype(float)
        mu = rng.uniform(0.5, 3.0, 100)
        fitted = _normalise_mu(mu, y.sum())
        np.testing.assert_array_equal(overdispersed_residual(y, fitted, 0.0), conditional_residual(y, mu))

    def test_zero_kappa_draw_consumes_the_dev2_stream(self):
        p = np.full(50, 1 / 50)
        a = draw_null(np.random.default_rng(5), 300, p, 0.0)
        b = np.random.default_rng(5).multinomial(300, p).astype(float)
        np.testing.assert_array_equal(a, b)

    def test_none_reproduces_dev2_results_exactly(self):
        # Golden values computed with microspira 0.1.0.dev2 on the same input and seed.
        side = 14
        coords = grid(side)
        rng = np.random.default_rng(11)
        n = len(coords)
        Y = np.column_stack([rng.multinomial(400, (lambda w: w / w.sum())(rng.gamma(0.5, 2.0, n)))
                             for _ in range(3)] + [rng.multinomial(400, np.full(n, 1 / n))])
        t = ms.run(Carrier(Y, coords), technical_covariates=["depth"], n_neighbors=4, n_boot=49, seed=3,
                   overdispersion="none").taxa.sort_values("taxon")
        self.assertEqual(t.spatial_p.round(12).tolist(), [0.2, 0.36, 0.12, 0.98])
        np.testing.assert_allclose(t.spatial_I_residual.to_numpy(),
                                   [0.034608514376, 0.002858271252, 0.03854808291, -0.078918529099], atol=1e-11)
        self.assertTrue((t.dispersion_model == "multinomial").all())
        self.assertTrue((t.dispersion_kappa == 0).all())


class InterfaceTests(unittest.TestCase):
    def test_defaults_and_columns(self):
        self.assertEqual(SPIRAConfig().overdispersion, "neighbour")
        self.assertEqual(inspect.signature(ms.run).parameters["overdispersion"].default, "neighbour")
        self.assertEqual(inspect.signature(residual_spatial_test).parameters["overdispersion"].default, "none")
        coords = grid(10)
        Y = np.random.default_rng(3).poisson(2, size=(100, 2))
        t = ms.run(Carrier(Y, coords), technical_covariates=["depth"], n_neighbors=4, n_boot=19).taxa
        for col in ("dispersion_model", "dispersion_kappa", "dispersion_lambda", "dispersion_null_median"):
            self.assertIn(col, t.columns)
        self.assertTrue((t.dispersion_model == "gamma_multinomial_neighbour").all())
        self.assertTrue((t.dispersion_kappa >= 0).all())

    def test_config_none_is_respected(self):
        coords = grid(10)
        Y = np.random.default_rng(3).poisson(2, size=(100, 1))
        t = ms.run(Carrier(Y, coords), config=SPIRAConfig(technical_covariates=("depth",), n_neighbors=4,
                                                          n_boot=19, overdispersion="none")).taxa
        self.assertEqual(t.dispersion_model.iloc[0], "multinomial")

    def test_invalid_option_raises(self):
        coords = grid(6)
        Y = np.ones((36, 1))
        with self.assertRaises(ValueError):
            ms.run(Carrier(Y, coords), n_neighbors=4, n_boot=19, overdispersion="gamma")
        W = build_knn_graph(coords, 4)
        with self.assertRaises(ValueError):
            residual_spatial_test(Y[:, 0], np.ones((36, 1)), W, n_draws=19, overdispersion="x")


class CalibrationSmokeTests(unittest.TestCase):
    def test_local_null_is_calibrated_under_overdispersion(self):
        coords = grid(20)
        W = build_knn_graph(coords, 4)
        n = len(coords)
        X = np.ones((n, 1))
        gen = np.random.default_rng(8)
        counts = {"none": 0, "neighbour": 0}
        for _ in range(30):
            w = gen.gamma(1.0, 1.0, n)
            y = gen.multinomial(2000, w / w.sum()).astype(float)          # kappa = 1, no structure
            for method in counts:
                t = residual_spatial_test(y, X, W, n_draws=49, rng=np.random.default_rng(9), refit_bootstrap=True,
                                          localize=True, overdispersion=method)
                counts[method] += int(np.any((t.local_p <= 0.05) & (t.neighborhood_reads >= 3)))
        self.assertGreaterEqual(counts["none"], 25)        # dev2 null: nearly every null taxon has a hotspot
        self.assertLessEqual(counts["neighbour"], 5)       # MF4 null: about 5% expected


if __name__ == "__main__":
    unittest.main()
