# Changelog

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

