import unittest

import numpy as np
import pandas as pd

import microspira as ms


class FakeAnnData:
    def __init__(self, X, obs, var, coords):
        self.X = X
        self.obs = obs
        self.var = var
        self.var_names = pd.Index(var.index)
        self.obsm = {"spatial": coords}


class ApiTests(unittest.TestCase):
    def test_run_returns_rank_and_context_reports(self):
        side = 8
        xx, yy = np.meshgrid(np.arange(side), np.arange(side))
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        n = len(coords)
        cluster = (coords[:, 0] < 3) & (coords[:, 1] < 3)
        rng = np.random.default_rng(2)
        X = np.column_stack(
            [
                np.where(cluster, 12, 0),
                rng.poisson(1.0, n),
                np.where(coords[:, 0] > 5, 7, 0),
            ]
        )
        obs = pd.DataFrame(
            {
                "technical_depth": rng.uniform(0.2, 1.0, n),
                "host_density": cluster.astype(float),
            },
            index=[f"u{i}" for i in range(n)],
        )
        var = pd.DataFrame(
            {
                "species": ["species_a", "species_b", "species_c"],
                "genus": ["genus_a", "genus_a", "genus_c"],
            },
            index=["species_a", "species_b", "species_c"],
        )
        adata = FakeAnnData(X, obs, var, coords)
        result = ms.run(
            adata,
            technical_covariates=["technical_depth"],
            context_covariates={"host": "host_density"},
            ranks=["genus"],
            n_neighbors=4,
            n_boot=29,
            seed=3,
        )
        self.assertIn("feature", set(result.taxa["rank"]))
        self.assertIn("genus", set(result.taxa["rank"]))
        self.assertIn("context_host_associated", result.taxa.columns)
        self.assertIn("support_tier", result.taxa.columns)
        self.assertIn("spatial_call", result.taxa.columns)


if __name__ == "__main__":
    unittest.main()
