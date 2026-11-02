"""Tests for the dev1 must-fixes: context descriptors (MF1), member-unit hotspots (MF2), refit default (MF3)."""

import inspect
import unittest

import numpy as np
import pandas as pd

import microspira as ms
from microspira.clusters import localize_hotspots, neighborhood_operator, neighborhood_enrichment
from microspira.config import SPIRAConfig
from microspira.context import assess_context_association, decile_contrast
from microspira.graph import build_knn_graph
from microspira.model import conditional_residual, fit_poisson_gam, residual_spatial_test


def grid(side):
    xx, yy = np.meshgrid(np.arange(side), np.arange(side))
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(float)


class Carrier:
    def __init__(self, y, coords, obs=None):
        self.X = np.asarray(y)[:, None]
        self.obs = obs if obs is not None else pd.DataFrame(index=[f"u{i}" for i in range(len(y))])
        self.var = pd.DataFrame(index=["taxon"])
        self.var_names = self.var.index
        self.obsm = {"spatial": coords}


class MemberUnitTests(unittest.TestCase):
    def setUp(self):
        self.coords = grid(10)
        self.W = build_knn_graph(self.coords, 4)
        self.n = len(self.coords)
        self.X = np.ones((self.n, 1))

    def _localize(self, y, **kwargs):
        res = residual_spatial_test(y, self.X, self.W, n_draws=99, refit_bootstrap=True,
                                    localize=True, rng=np.random.default_rng(5))
        return localize_hotspots(y, res.residual, self.W, self.coords, rank="feature", taxon="t",
                                 local_p=res.local_p, neighborhood_reads=res.neighborhood_reads,
                                 neighborhood_expected=res.neighborhood_expected, **kwargs)

    def test_single_hot_unit_is_one_member_and_unsupported(self):
        y = np.zeros(self.n)
        y[44] = 20
        clusters = self._localize(y)
        self.assertEqual(len(clusters), 1)
        self.assertGreater(clusters.n_centres.iloc[0], 1)   # one-hop dilation of centres
        self.assertEqual(clusters.n_member_units.iloc[0], 1)
        self.assertFalse(clusters.component_supported.iloc[0])

    def test_planted_block_members_match_block(self):
        y = np.zeros(self.n)
        block = (self.coords[:, 0] >= 3) & (self.coords[:, 0] < 6) & (self.coords[:, 1] >= 3) & (self.coords[:, 1] < 6)
        y[block] = 10
        clusters = self._localize(y)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters.n_member_units.iloc[0], int(block.sum()))
        self.assertEqual(clusters.member_reads.iloc[0], 10.0 * block.sum())
        self.assertTrue(clusters.component_supported.iloc[0])

    def test_member_area_requires_unit_area(self):
        y = np.zeros(self.n)
        y[[44, 45]] = 15
        clusters = self._localize(y, unit_area=2500.0, coordinate_unit="um")
        self.assertEqual(clusters.member_area.iloc[0], 2 * 2500.0)
        self.assertEqual(clusters.member_area_unit.iloc[0], "um^2")

    def test_diffuse_component_without_excess_units_is_kept_unsupported(self):
        y = np.ones(self.n)
        residual = np.zeros(self.n)
        local_p = np.ones(self.n)
        local_p[[44, 45]] = 0.01
        reads = np.full(self.n, 5.0)
        clusters = localize_hotspots(y, residual, self.W, self.coords, rank="feature", taxon="t",
                                     local_p=local_p, neighborhood_reads=reads,
                                     neighborhood_expected=np.full(self.n, 4.0))
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters.n_member_units.iloc[0], 0)
        self.assertFalse(clusters.component_supported.iloc[0])

    def test_shared_member_goes_to_stronger_component(self):
        # Centres 22 and 24 are not adjacent but both neighbour unit 23.
        y = np.zeros(self.n)
        y[23] = 9
        residual = np.zeros(self.n)
        residual[23] = 5.0
        local_p = np.ones(self.n)
        local_p[22], local_p[24] = 0.02, 0.01
        reads = np.full(self.n, 9.0)
        clusters = localize_hotspots(y, residual, self.W, self.coords, rank="feature", taxon="t",
                                     local_p=local_p, neighborhood_reads=reads,
                                     neighborhood_expected=np.full(self.n, 1.0))
        self.assertEqual(len(clusters), 2)
        owner = clusters.loc[clusters.n_member_units == 1]
        self.assertEqual(len(owner), 1)
        self.assertEqual(owner.local_max_adjusted_p_min.iloc[0], 0.01)
        self.assertEqual(int(clusters.n_shared_members.sum()), 1)


class ContextDescriptorTests(unittest.TestCase):
    def test_decile_contrast_sign_and_binary_context(self):
        x = np.arange(100.0)
        self.assertGreater(decile_contrast(x, 0.02 * x), 0)
        self.assertLess(decile_contrast(x, -0.02 * x), 0)
        binary = (x >= 70).astype(float)
        self.assertAlmostEqual(decile_contrast(binary, np.log(3.0) * binary), np.log(3.0))
        self.assertTrue(np.isnan(decile_contrast(np.r_[np.zeros(95), np.ones(5)], np.zeros(100))))

    def test_context_has_no_p_value_and_recovers_direction(self):
        coords = grid(20)
        x = coords[:, 0]
        rng = np.random.default_rng(3)
        lam = np.exp(0.15 * x)
        y = rng.multinomial(2000, lam / lam.sum()).astype(float)
        obs = pd.DataFrame({"dist": x})
        X0 = np.ones((len(y), 1))
        assoc = assess_context_association(y, X0, obs, label="d", column="dist",
                                           baseline_fit=fit_poisson_gam(y, X0))
        self.assertFalse(hasattr(assoc, "p_value"))
        self.assertTrue(assoc.available)
        self.assertEqual(assoc.direction, 1)
        # True contrast between x >= 17.1 and x <= 1.9 deciles: 0.15 * (18 - 1) = 2.55
        self.assertAlmostEqual(assoc.log_contrast, 2.55, delta=0.3)


class ReplicateContextTests(unittest.TestCase):
    def _result(self, contrast, available=True):
        taxa = pd.DataFrame({"rank": ["feature"], "taxon": ["t"],
                             "context_d_available": [available],
                             "context_d_log_contrast": [contrast]})
        return ms.SPIRAResult(taxa=taxa, clusters=pd.DataFrame(), config={}, validation={})

    def test_six_concordant_domains(self):
        out = ms.replicate_context([self._result(0.5)] * 6, "d")
        self.assertAlmostEqual(out.sign_test_p.iloc[0], 2 * 0.5 ** 6)
        self.assertTrue(out.replicated.iloc[0])
        self.assertEqual(out.consensus_direction.iloc[0], "+")

    def test_five_concordant_domains_do_not_replicate(self):
        out = ms.replicate_context([self._result(0.5)] * 5, "d")
        self.assertAlmostEqual(out.sign_test_p.iloc[0], 0.0625)
        self.assertFalse(out.replicated.iloc[0])

    def test_groups_are_compared_units_and_unavailable_excluded(self):
        results = [self._result(c) for c in (0.4, -0.1, 0.3, 0.2)] + [self._result(np.nan, False)]
        out = ms.replicate_context(results, "d", groups=["m1", "m1", "m2", "m3", "m3"])
        self.assertEqual(out.n_units_compared.iloc[0], 3)
        self.assertEqual(out.n_domains_available.iloc[0], 4)
        self.assertEqual(out.n_positive.iloc[0], 3)


class RefitDefaultTests(unittest.TestCase):
    def test_refit_is_default_in_config_and_run(self):
        self.assertTrue(SPIRAConfig().refit_bootstrap)
        self.assertTrue(inspect.signature(ms.run).parameters["refit_bootstrap"].default)

    def test_explicit_fast_mode_still_overrides(self):
        coords = grid(8)
        y = ((coords[:, 0] < 3) & (coords[:, 1] < 3)).astype(int) * 10
        report = ms.run(Carrier(y, coords), n_neighbors=4, n_boot=19, refit_bootstrap=False)
        self.assertFalse(report.taxa.bootstrap_refit.iloc[0])
        report = ms.run(Carrier(y, coords), config=SPIRAConfig(refit_bootstrap=False), n_boot=19)
        self.assertFalse(report.taxa.bootstrap_refit.iloc[0])


if __name__ == "__main__":
    unittest.main()
