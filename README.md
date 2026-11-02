# microSPIRA

**microSPIRA** implements **SPIRA** (*Spatial Pattern Inference with Residual
Adjustment*): a count-aware toolkit for prioritising spatial microbial signals
in host–microbe sequencing data.

> **Status:** pre-alpha research software. The API and statistical defaults are
> under active calibration through controlled simulation and real-background
> semi-simulation. Do not use it for clinical or causal claims.

## What SPIRA reports

SPIRA does not treat host- or tissue-associated taxa as automatic contaminants.
For every taxon and requested taxonomic rank, it reports:

1. **Support / testability** — total reads, occupied units, effective support,
   and a transparent support tier.
2. **Spatial evidence** — a residual Moran's I test after a user-defined
   *technical* mean model.
3. **Context** — host, capture-boundary, tissue-interface, or cell-type
   associations as independent, non-exclusive flags.
4. **Taxonomic resolution** — native feature, species, genus, family, and
   other rank-wise reports when taxonomy columns are supplied.
5. **Physical scale** — hotspot count, location, size, read support, and
   equivalent diameter for FDR-positive spatial taxa.

The primary output has three states:

```text
NOT_TESTABLE
NO_EVIDENCE_OF_SPATIAL_STRUCTURE
SPATIALLY_STRUCTURED
```

`NO_EVIDENCE_OF_SPATIAL_STRUCTURE` does not mean biological absence. Context
flags are associations, not causal explanations.

Version 0.1.0.dev1 additionally uses `INFERENCE_FAILED` for a supported taxon
whose observed fit or requested bootstrap draws could not be validated. This
is a numerical status, not negative biological evidence. Such rows publish
unavailable p/q and remain in the BH family through p=1 placeholders.

## Installation

```bash
pip install -e .
```

For plotting helpers:

```bash
pip install -e ".[plot]"
```

## Minimal example

```python
import microspira as ms

ms.validate(
    microbial_adata,
    spatial_key="spatial",
    taxonomy_columns=["species", "genus", "family"],
)

result = ms.run(
    microbial_adata,
    # Only variables justified as measurement/exposure controls belong here.
    technical_covariates=["capture_depth", "mito_frac"],
    # These remain interpretable flags; they never remove a spatial call.
    context_covariates={
        "host_density": "host_density",
        "capture_boundary": "boundary_dist",
        "tissue_interface": "tissue_dist",
    },
    ranks=["species", "genus", "family", "phylum"],
    coordinate_unit="um",
    unit_area=2500.0,
    n_neighbors=6,
    n_boot=199,
    # Use True for final calibration; False is a faster exploratory mode.
    refit_bootstrap=True,
)

print(result.taxa.head())
print(result.clusters.head())
result.write("spira_report")
```

## Statistical core

For taxon \(t\) at spatial unit \(i\), SPIRA fits a technical mean model:

\[
Y_{it} \sim \mathrm{Poisson}(\mu_{it}), \qquad
\log \mu_{it} = \beta_{0t} + \sum_k f_{kt}(x_{ik}).
\]

The \(f_k\) terms are cubic B-spline effects for selected *technical*
covariates. SPIRA then calculates a total-read-conditioned residual

\[
r_{it} = \frac{Y_{it}-\hat\mu_{it}}
{\sqrt{\mathrm{Var}(Y_{it}\mid\sum_iY_{it})}}
\]

and tests positive Moran's I of that residual against conditional multinomial
bootstrap draws. ``refit_bootstrap=True`` refits the mean model on each draw to
capture fitted-model uncertainty.

The solver profiles the free intercept and optimizes the same Poisson
likelihood with a small ridge on spline coefficients. It records iteration
count, fitted-mean/objective change and scaled score. Bootstrap columns include
`bootstrap_n_valid`, `bootstrap_fit_failures`, `bootstrap_invalid_statistics`
and `bootstrap_max_n_iter`. Failed draws are not silently discarded.

## Neighborhood localization (development version)

The default uses fixed center-plus-one-hop neighborhoods. Their count
enrichment is scored by a conditional binomial upper tail; the score itself
is not treated as a calibrated p-value. Bootstrap maxima across all centers
provide within-taxon adjusted candidate p-values, with model refitting when
the primary test uses refitting. The same null draws serve both statistics.

Only globally structured taxa receive cluster-table entries. Minimum local
read support is required; small components remain marked by
`component_supported=False`. `area_kind` states that reported areas are
enriched-neighborhood-center footprints, not measured colony boundaries.
This does not establish across-taxon local FDR control. Calibration and
localization accuracy remain research validation tasks.

Use `SPIRAConfig(localization_method="legacy_residual")` only to reproduce the
previous descriptive residual cutoff. `hotspot_z` applies to that mode.

## Covariate roles

| Role | Examples | Consequence |
|---|---|---|
| Technical adjustment | exposure, platform QC, batch | enters primary residual test |
| Context annotation | host density, tissue interface, cell type | independent association flag |

The distinction is intentional. Tissue interface and host-density patterns can
represent real microbial biology; SPIRA reports them rather than censoring
them.

## Current scope

- Raw, non-negative integer count matrices only.
- Coordinates in ``adata.obsm["spatial"]`` by default.
- Taxonomy columns in ``adata.var``.
- kNN graph as the default spatial graph.
- Local hotspots are descriptive summaries after a global FDR-positive call.

## Development priorities

1. Controlled calibration of support tiers and residual-bootstrap FDR.
2. Real-background semi-simulation on intestinal Visium sections A–D.
3. Stereo-seq fixed-physical-binning and tissue/lumen-interface support.
4. Multi-scale localisation, runtime benchmarks, and HTML report generation.

## License

MIT.

