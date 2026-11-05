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
   descriptors (deviance improvement and a signed decile contrast) per
   analysis domain; association is tested across independent domains with
   `ms.replicate_context`.
4. **Taxonomic resolution** — native feature, species, genus, family, and
   other rank-wise reports when taxonomy columns are supplied.
5. **Hotspot scale** — hotspot count, location, number of member units and
   member read support for FDR-positive spatial taxa.

The primary output has three states:

```text
NOT_TESTABLE
NO_EVIDENCE_OF_SPATIAL_STRUCTURE
SPATIALLY_STRUCTURED
```

`NO_EVIDENCE_OF_SPATIAL_STRUCTURE` does not mean biological absence. Context
descriptors and replicated associations are not causal explanations.

Within one analysis domain, spatial units are dependent and context covariates
are usually smooth, so no within-domain context p-value is reported: a
spatially structured taxon would align partly with almost any smooth covariate
by chance. Run one tissue section per `ms.run` call and combine calls:

```python
results = [ms.run(section, context_covariates={"boundary": "boundary_dist"})
           for section in sections]
ms.replicate_context(results, "boundary", groups=animal_ids)  # exact sign test + BH
```

At least six concordant independent units are needed for a two-sided sign-test
p below 0.05; this does not guarantee a BH-adjusted q below 0.05. The sign-test
null requires independent compared units with equally probable positive and
negative nonzero contrasts. Availability filtering must not favour a direction.

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
    # Per-domain descriptors; they never remove a spatial call.
    context_covariates={
        "host_density": "host_density",
        "capture_boundary": "boundary_dist",
        "tissue_interface": "tissue_dist",
    },
    ranks=["species", "genus", "family", "phylum"],
    coordinate_unit="um",
    unit_area=2500.0,
    n_neighbors=6,
    n_boot=999,
    # Default True (Methods). False is for faster exploration only.
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

and tests positive Moran's I of that residual against conditional bootstrap
draws. ``refit_bootstrap=True`` refits the mean model on each draw to capture
fitted-model uncertainty.

Since 0.1.0.dev3 the null allows independent unit-level overdispersion
(``overdispersion="neighbour"``, the default): each unit's probability carries
an independent Gamma factor with mean 1 and variance \(\kappa_t\). \(\kappa_t\)
is estimated from differences between graph neighbours (insensitive to smooth
spatial structure), the residual variance becomes
\(N_t p_{it}(1-p_{it}) + \kappa_t\hat\mu_{it}^2\), bootstrap draws include
the Gamma factors, and \(\kappa_t\) is re-estimated on every draw.
``overdispersion="none"`` reproduces the dev2 multinomial null. Without this,
unit-level overdispersion makes the local max-T step report isolated
excess units as hotspots and makes the global test mildly liberal for abundant
taxa.

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
read support is required. Connected significant centres form a component; its
**member units** are the units inside the centres' neighbourhoods whose own
standardized residual is at least `member_z` (default 1.96). Size and support
are counted in member units (`n_member_units`, `member_reads`), so a single
hot unit is one member and is marked `component_supported=False`, even though
its excess makes several neighbouring centres significant. `member_area` is
reported only when `unit_area` and `coordinate_unit` are supplied; no
equivalent diameter is computed. Members are units carrying excess counts,
not measured colony boundaries. This does not establish across-taxon local
FDR control.

Set `member_z` and `min_cluster_units` on `SPIRAConfig` (defaults 1.96 and 2).
Shared members are assigned once: smallest component minimum local p, then
largest peak neighbourhood reads, then lowest component index. The winning
component records them in `n_shared_members`. A component with no qualifying
members remains visible with support false; its centroid and peak residual
fall back to the centre positions.

Use `SPIRAConfig(localization_method="legacy_residual")` only to reproduce the
previous descriptive residual cutoff. `hotspot_z` applies to that mode.

## Covariate roles

| Role | Examples | Consequence |
|---|---|---|
| Technical adjustment | exposure, platform QC, batch | enters primary residual test |
| Context annotation | host density, tissue interface, cell type | per-domain descriptor; association tested across domains |

The distinction is intentional. Tissue interface and host-density patterns can
represent real microbial biology; SPIRA reports them rather than censoring
them.

## Current scope

- Raw, non-negative integer count matrices only.
- Coordinates in ``adata.obsm["spatial"]`` by default.
- Taxonomy columns in ``adata.var``.
- kNN graph as the default spatial graph.
- Local hotspots are descriptive summaries after a global FDR-positive call.

## Analysis settings and migration from dev1

- Analyze one tissue section per `run` call. Split capture areas containing
  several sections before graph construction. Group sections from the same
  animal with `replicate_context(..., groups=animal_ids)`.
- Use the same context definition, units and orientation in every section.
  Positive contrast for a distance means enrichment farther from the boundary.
- Square bins require a justified graph (e.g. `n_neighbors=4` or `8`); the
  default 6 is intended for a Visium-like hexagonal lattice. This is a setting,
  not automatic platform detection.
- Supply direct, non-cumulative taxonomic counts before rank aggregation.
- `n_boot` remains 199 by default. The example uses 999, but the resolution
  `1 / (n_boot + 1)` must be assessed against the planned BH family. For a
  lone discovery among 300 tests at alpha 0.05, at least 5999 draws are needed
  to reach the first BH threshold. There is no automatic choice or stopping.
- `run` and `SPIRAConfig` now default to refitting; the low-level
  `model.residual_spatial_test` default remains false for compatibility.
  Explicit `refit_bootstrap=False` is still supported for exploration. Fast
  mode can be conservative; calibration depends on the analysis setting.
- Replace per-domain `context_*_p`, `_q` and `_associated` with descriptive
  `context_*_log_contrast`, `_fold_contrast`, `_direction` and `_available`.
  Replicated association has a separate table with `sign_test_p`,
  `sign_test_q` and `replicated`.
- Cluster columns `n_units`, `cluster_reads`, `area`, `area_unit`,
  `equivalent_diameter` and `diameter_unit` are removed. Use `n_centres`,
  `n_member_units`, `member_reads`, `member_area` and `member_area_unit`.
  This is a change in definition, not a simple column rename. Update cached
  cluster tables with the dev2 summarizer or rerun the analysis.

## Development priorities

1. Controlled calibration of support tiers and residual-bootstrap FDR.
2. Real-background semi-simulation on intestinal Visium sections A–D.
3. Stereo-seq fixed-physical-binning and tissue/lumen-interface support.
4. Multi-scale localisation, runtime benchmarks, and HTML report generation.

## License

MIT.

