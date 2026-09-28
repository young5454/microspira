"""Minimal microSPIRA example.

Replace the placeholder AnnData object with a microbial raw-count AnnData.
Required fields:
  - adata.X: spot/bin × taxon raw count matrix
  - adata.obsm["spatial"]: spatial coordinates
  - adata.obs: technical/context covariates
  - adata.var: taxonomy columns
"""

import microspira as ms


def run_spira(microbial_adata):
    return ms.run(
        microbial_adata,
        technical_covariates=["capture_depth", "mito_frac"],
        context_covariates={
            "host_density": "host_density",
            "capture_boundary": "boundary_dist",
        },
        ranks=["species", "genus", "family", "phylum"],
        coordinate_unit="um",
        n_neighbors=6,
        n_boot=199,
        refit_bootstrap=True,
        seed=2026,
    )


if __name__ == "__main__":
    raise SystemExit(
        "Load a microbial raw-count AnnData and call run_spira(microbial_adata)."
    )

