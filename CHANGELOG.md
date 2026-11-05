# Changelog

## 0.1.0.dev3 — unreleased (MF4)

- Overdispersion (MF4): the conditional technical null allows independent
  unit-level Gamma heterogeneity (variance kappa) around the technical mean.
  kappa is estimated per taxon from neighbour differences
  (`microspira.dispersion.neighbour_dispersion`), residuals use the variance
  N p (1 - p) + kappa mu^2, null draws include Gamma factors, and kappa is
  re-estimated on every draw. Global and local (max-T) inference share the
  draws as before. New `run(overdispersion="neighbour")` default and
  `SPIRAConfig.overdispersion`; `"none"` reproduces 0.1.0.dev2 exactly.
- New taxon columns: `dispersion_model`, `dispersion_kappa`,
  `dispersion_lambda` (kappa x mean reads per unit), `dispersion_null_median`.
- Member units (`member_z`) use the overdispersion-scaled residual.
- The low-level `residual_spatial_test(overdispersion="none")` default is
  unchanged, as is its `refit_bootstrap=False` default.

## 0.1.0.dev2 — 2026-11-02

- Context (MF1): remove the independence-based chi-square p/q and the
  `context_*_associated` flag from per-domain output. Add a signed decile
  contrast (`context_*_log_contrast`, `_fold_contrast`, `_direction`) and
  `ms.replicate_context`, an exact sign test across independent domains with
  optional grouping (e.g. animals) and BH across taxa.
- Hotspots (MF2): count size and support in member units (units inside the
  significant neighbourhoods with residual >= `member_z`), not centres. New
  columns `n_centres`, `n_member_units`, `member_reads`, `member_area`,
  `n_shared_members`; centroid and peak residual use member units. Equivalent
  diameter removed.
- Refit (MF3): `refit_bootstrap=True` is the default in `run()` and
  `SPIRAConfig`, matching the Methods.
- Preserve the low-level model default and the global/local bootstrap
  algorithms. Document sign-test assumptions, unavailable outputs, schema
  migration and finite-bootstrap resolution.

## 0.1.0.dev1 — 2026-10-19

- Profile the free Poisson intercept, use stable conditional-likelihood Newton
  steps, and require stable mean/objective plus a small scaled score.
- Report observed-fit diagnostics and every requested bootstrap draw's validity.
  An unresolved fit/statistic produces `INFERENCE_FAILED` and unavailable p/q.
  Failed testable items retain p=1 placeholders in the BH family.
- Replace default residual-cutoff hotspots with fixed-neighborhood count-tail
  scores calibrated against bootstrap maxima within each taxon. Small components
  remain visible with support flags. Areas describe neighborhood-center footprints.
- Keep the previous residual threshold as explicit `localization_method="legacy_residual"`.
- Make missing physical diameter units explicit and expose context-fit failures.
- Add failure-path, likelihood-reference, offset and neighborhood-localization tests.

## 0.1.0.dev0 — 2026-09-28

Initial pre-alpha implementation of the SPIRA design:

- input validation for raw microbial counts, coordinates, and taxonomy columns;
- transparent support/testability tiers;
- cubic-spline Poisson mean model with conditional residual Moran's I;
- fast and refitted parametric-bootstrap modes;
- independent context-association flags;
- species-to-phylum count aggregation;
- residual-hotspot cluster summaries with physical-scale metadata;
- tidy taxon/cluster report export.

