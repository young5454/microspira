"""Boundary cases for the MF1-MF3 output definitions and multiplicity family."""

import inspect
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

import microspira as ms
from microspira.clusters import CLUSTER_COLUMNS, localize_hotspots
from microspira.context import assess_context_association
from microspira.model import fit_poisson_gam, residual_spatial_test


class ContextAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.y = np.ones(100)
        self.X = np.ones((100, 1))
        self.obs = pd.DataFrame({"x": np.arange(100.0)})
        self.base = fit_poisson_gam(self.y, self.X)

    def assess(self, obs=None, **kwargs):
        return assess_context_association(
            self.y, self.X, self.obs if obs is None else obs,
            label="d", column="x", baseline_fit=kwargs.get("baseline", self.base),
        )

    def test_missing_and_constant_context_are_unavailable(self):
        for obs, message in ((pd.DataFrame(index=range(100)), "unavailable"),
                             (pd.DataFrame({"x": np.zeros(100)}), "constant")):
            with self.subTest(message=message):
                result = self.assess(obs)
                self.assertFalse(result.available)
                self.assertEqual(result.direction, 0)
                self.assertTrue(np.isnan(result.log_contrast))
                self.assertIn(message, result.message)

    def test_failed_base_or_augmented_fit_is_unavailable(self):
        failed = replace(self.base, converged=False)
        self.assertFalse(self.assess(baseline=failed).available)
        with patch("microspira.context.fit_poisson_gam", return_value=failed):
            result = self.assess()
        self.assertFalse(result.available)
        self.assertIn("did not converge", result.message)

    def test_inseparable_deciles_retain_fit_descriptor_without_direction(self):
        result = self.assess(pd.DataFrame({"x": np.r_[np.zeros(95), np.ones(5)]}))
        self.assertFalse(result.available)
        self.assertEqual(result.direction, 0)
        self.assertTrue(np.isfinite(result.delta_deviance))
        self.assertIn("deciles cannot be separated", result.message)


class MemberAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.W = sparse.diags([np.ones(4), np.ones(4)], [-1, 1], shape=(5, 5)).tocsr()
        self.xy = np.column_stack([np.arange(5), np.zeros(5)])
        self.y = np.array([0, 9, 0, 0, 0])
        self.r = np.array([0, 5, 0, 0, 0])
        self.p = np.array([0.01, 1, 0.01, 1, 1])

    def localize(self, reads=None, **kwargs):
        return localize_hotspots(
            self.y, self.r, self.W, self.xy, rank="feature", taxon="t",
            local_p=self.p, neighborhood_reads=np.full(5, 9.0) if reads is None else reads,
            neighborhood_expected=np.ones(5), **kwargs,
        )

    def test_shared_member_tie_uses_peak_reads_then_component_index(self):
        reads = np.full(5, 9.0)
        reads[2] = 10
        clusters = self.localize(reads)
        self.assertEqual(clusters.loc[1, "n_member_units"], 1)
        self.assertEqual(clusters.loc[0, "n_member_units"], 0)
        tied = self.localize()
        self.assertEqual(tied.loc[0, "n_member_units"], 1)
        self.assertEqual(tied.loc[1, "n_member_units"], 0)
        self.assertEqual(tied.n_shared_members.sum(), 1)
        self.assertEqual(tied.member_reads.sum(), 9)

    def test_member_centroid_and_empty_component_fallback(self):
        clusters = self.localize()
        self.assertEqual(clusters.loc[0, "centroid_x"], 1)
        self.assertEqual(clusters.loc[0, "peak_residual"], 5)
        self.assertEqual(clusters.loc[1, "centroid_x"], 2)
        self.assertEqual(clusters.loc[1, "peak_residual"], 0)
        self.assertFalse(clusters.component_supported.any())

    def test_custom_member_z_retains_zero_member_components(self):
        clusters = self.localize(member_z=6)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters.n_member_units.sum(), 0)
        self.assertEqual(clusters.member_reads.sum(), 0)
        self.assertFalse(clusters.component_supported.any())

    def test_invalid_member_threshold_and_missing_physical_unit_rejected(self):
        for value in (0, -1, np.nan, np.inf):
            with self.subTest(member_z=value), self.assertRaises(ValueError):
                self.localize(member_z=value)
        with self.assertRaises(ValueError):
            self.localize(unit_area=2500)

    def test_legacy_members_are_residual_exceedances(self):
        residual = np.array([0, 5, 4, 0, 0])
        with self.assertWarns(DeprecationWarning):
            clusters = localize_hotspots(
                self.y, residual, self.W, self.xy, rank="feature", taxon="t",
            )
        self.assertEqual(clusters.n_member_units.iloc[0], 2)
        self.assertTrue(clusters.component_supported.iloc[0])
        self.assertTrue(np.isnan(clusters.n_centres.iloc[0]))
        self.assertEqual(clusters.area_kind.iloc[0], "residual_exceedance_footprint")

    def test_empty_cluster_table_has_only_new_schema(self):
        self.p[:] = 1
        clusters = self.localize()
        self.assertEqual(list(clusters.columns), CLUSTER_COLUMNS)
        self.assertEqual(len(clusters), 0)
        for removed in ("n_units", "cluster_reads", "area", "area_unit",
                        "equivalent_diameter", "diameter_unit"):
            self.assertNotIn(removed, clusters.columns)


class ReplicateBoundaryTests(unittest.TestCase):
    @staticmethod
    def result(contrasts, available=None, ranks=None):
        count = len(contrasts)
        taxa = pd.DataFrame({
            "rank": ranks if ranks is not None else ["feature"] * count,
            "taxon": [f"t{i}" for i in range(count)],
            "context_d_available": available if available is not None else [True] * count,
            "context_d_log_contrast": contrasts,
        })
        return ms.SPIRAResult(taxa, pd.DataFrame(), {}, {})

    def test_bh_family_includes_all_requested_ranks(self):
        results = [self.result([0.5, 0.5 if i < 3 else -0.5], ranks=["feature", "genus"])
                   for i in range(6)]
        out = ms.replicate_context(results, "d").set_index("rank")
        self.assertEqual(out.loc["feature", "sign_test_p"], 0.03125)
        self.assertEqual(out.loc["feature", "sign_test_q"], 0.0625)
        self.assertFalse(out.loc["feature", "replicated"])

    def test_exact_zeros_and_nonfinite_contrasts_are_excluded(self):
        values = [0.5, -0.5, 0, np.nan, np.inf]
        out = ms.replicate_context([self.result([value]) for value in values], "d")
        row = out.iloc[0]
        self.assertEqual(row.n_domains_total, 5)
        self.assertEqual(row.n_domains_available, 3)
        self.assertEqual(row.n_units_compared, 2)
        self.assertEqual(row.sign_test_p, 1)
        self.assertEqual(row.consensus_direction, "tie")

    def test_zero_group_mean_is_excluded_after_averaging(self):
        results = [self.result([value]) for value in (1, -1, 0.5)]
        row = ms.replicate_context(results, "d", groups=["a", "a", "b"]).iloc[0]
        self.assertEqual(row.n_domains_available, 3)
        self.assertEqual(row.n_units_compared, 1)
        self.assertEqual(row.n_positive, 1)
        self.assertEqual(row.median_log_contrast, 0.25)

    def test_all_unavailable_or_zero_have_no_replicated_association(self):
        unavailable = ms.replicate_context([self.result([np.nan], [False])], "d")
        self.assertEqual(len(unavailable), 0)
        self.assertIn("sign_test_q", unavailable.columns)
        zero = ms.replicate_context([self.result([0.0])] * 6, "d").iloc[0]
        self.assertEqual(zero.n_units_compared, 0)
        self.assertTrue(np.isnan(zero.sign_test_p))
        self.assertTrue(np.isnan(zero.sign_test_q))
        self.assertFalse(zero.replicated)

    def test_negative_concordance_and_unavailable_domain_count(self):
        results = [self.result([-0.5])] * 6 + [self.result([2.0], [False])]
        row = ms.replicate_context(results, "d").iloc[0]
        self.assertEqual(row.n_domains_total, 7)
        self.assertEqual(row.n_domains_available, 6)
        self.assertEqual(row.n_negative, 6)
        self.assertEqual(row.consensus_direction, "-")
        self.assertEqual(row.sign_test_p, 0.03125)
        self.assertTrue(row.replicated)

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            ms.replicate_context([], "d")
        with self.assertRaises(ValueError):
            ms.replicate_context([self.result([1])], "d", groups=[])
        with self.assertRaises(ValueError):
            ms.replicate_context([self.result([1])], "d", alpha=1)
        with self.assertRaises(KeyError):
            ms.replicate_context([self.result([1])], "missing")


class CompatibilityTests(unittest.TestCase):
    def test_version_and_low_level_default(self):
        self.assertEqual(ms.__version__, "0.1.0.dev2")
        self.assertIs(inspect.signature(residual_spatial_test)
                      .parameters["refit_bootstrap"].default, False)

    def test_not_testable_context_schema_and_config_member_threshold(self):
        side = 8
        x, y = np.meshgrid(np.arange(side), np.arange(side))
        coords = np.column_stack([x.ravel(), y.ravel()])

        class Carrier:
            X = sparse.csr_matrix(np.zeros((side * side, 1), dtype=int))
            obs = pd.DataFrame({"dist": coords[:, 0]})
            var = pd.DataFrame(index=["t"])
            var_names = var.index
            obsm = {"spatial": coords}

        config = ms.SPIRAConfig(context_covariates={"d": "dist"}, member_z=4)
        result = ms.run(Carrier(), config=config, n_boot=19)
        row = result.taxa.iloc[0]
        self.assertEqual(row.spatial_call, "NOT_TESTABLE")
        self.assertFalse(row.context_d_available)
        self.assertEqual(row.context_d_message, "not testable")
        self.assertEqual(row.context_d_direction, 0)
        self.assertTrue(np.isnan(row.context_d_fold_contrast))
        self.assertEqual(result.config["member_z"], 4)
        for suffix in ("p", "q", "associated"):
            self.assertNotIn(f"context_d_{suffix}", result.taxa.columns)


if __name__ == "__main__":
    unittest.main()
