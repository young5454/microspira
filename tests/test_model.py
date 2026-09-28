import unittest

import numpy as np

from microspira.graph import build_knn_graph
from microspira.model import design_matrix, fit_poisson_gam, residual_spatial_test


class ModelTests(unittest.TestCase):
    def setUp(self):
        side = 10
        xx, yy = np.meshgrid(np.arange(side), np.arange(side))
        self.coords = np.column_stack([xx.ravel(), yy.ravel()])
        self.W = build_knn_graph(self.coords, n_neighbors=4)

    def test_fit_preserves_total(self):
        y = np.full(100, 2.0)
        X = np.ones((100, 1))
        fit = fit_poisson_gam(y, X)
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.mu.sum(), y.sum(), places=6)

    def test_compact_cluster_has_residual_spatial_evidence(self):
        y = np.zeros(100)
        in_cluster = (self.coords[:, 0] < 4) & (self.coords[:, 1] < 4)
        y[in_cluster] = 20
        X = np.ones((100, 1))
        result = residual_spatial_test(
            y, X, self.W, n_draws=99, rng=np.random.default_rng(4)
        )
        self.assertLessEqual(result.p_value, 0.05)
        self.assertGreater(result.observed_i, result.null_mean)


if __name__ == "__main__":
    unittest.main()

