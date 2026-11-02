"""User-facing configuration objects for SPIRA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SupportConfig:
    """Transparent first-pass rules for whether a taxon is testable.

    These defaults are deliberately reported with every result; they are not
    universal biological thresholds. Semi-simulation calibration should inform
    dataset-specific recommendations in later releases.
    """

    min_total_reads: int = 10
    min_occupied_units: int = 3
    min_effective_units: float = 2.0
    limited_total_reads: int = 50
    limited_occupied_units: int = 10
    strong_total_reads: int = 250
    strong_occupied_units: int = 50


@dataclass(frozen=True)
class SPIRAConfig:
    """Configuration for :func:`microspira.run`.

    ``technical_covariates`` are adjusted in the primary residual spatial test.
    ``context_covariates`` receive per-domain descriptors without p/q-values
    or association flags and are *not* used to drop a taxon. Context direction
    can be tested across independent domains with ``replicate_context``.
    """

    technical_covariates: Sequence[str] = field(default_factory=tuple)
    context_covariates: Mapping[str, str] = field(default_factory=dict)
    ranks: Sequence[str] = field(default_factory=tuple)
    spatial_key: str = "spatial"
    coordinate_unit: str | None = None
    unit_area: float | None = None
    n_neighbors: int = 6
    spline_df: int = 3
    n_boot: int = 199
    refit_bootstrap: bool = True
    alpha: float = 0.05
    seed: int = 0
    localize: bool = True
    hotspot_z: float = 1.96
    localization_method: str = "bootstrap_neighborhood"
    local_alpha: float = 0.05
    min_neighborhood_reads: int = 3
    min_cluster_units: int = 2
    member_z: float = 1.96
    support: SupportConfig = field(default_factory=SupportConfig)

    def as_dict(self) -> dict:
        """Return a JSON-serialisable configuration record."""
        return asdict(self)

