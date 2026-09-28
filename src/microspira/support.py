"""Count support and testability assessment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import SupportConfig


NOT_TESTABLE = "NOT_TESTABLE"
LIMITED = "LIMITED"
ADEQUATE = "ADEQUATE"
STRONG = "STRONG"


@dataclass(frozen=True)
class SupportResult:
    total_reads: int
    occupied_units: int
    effective_units: float
    max_read_fraction: float
    support_tier: str
    testable: bool

    def as_dict(self) -> dict:
        return asdict(self)


def assess_support(counts: np.ndarray, config: SupportConfig | None = None) -> SupportResult:
    """Assess whether a taxon has enough distributed evidence for testing.

    ``effective_units`` is the inverse Simpson number of read-bearing units. It
    prevents a large read total concentrated in one spatial unit from appearing
    equally testable to the same total spread over many units.
    """
    config = config or SupportConfig()
    y = np.asarray(counts, dtype=float).ravel()
    if np.any(y < 0) or not np.all(np.isfinite(y)):
        raise ValueError("counts must be finite and non-negative.")

    total = int(np.rint(y.sum()))
    occupied = int(np.count_nonzero(y > 0))
    if total == 0:
        return SupportResult(0, 0, 0.0, np.nan, NOT_TESTABLE, False)

    p = y[y > 0] / y.sum()
    effective = float(1.0 / np.sum(p**2))
    max_fraction = float(p.max())

    if (
        total < config.min_total_reads
        or occupied < config.min_occupied_units
        or effective < config.min_effective_units
    ):
        tier = NOT_TESTABLE
    elif total < config.limited_total_reads or occupied < config.limited_occupied_units:
        tier = LIMITED
    elif total >= config.strong_total_reads and occupied >= config.strong_occupied_units:
        tier = STRONG
    else:
        tier = ADEQUATE

    return SupportResult(
        total_reads=total,
        occupied_units=occupied,
        effective_units=effective,
        max_read_fraction=max_fraction,
        support_tier=tier,
        testable=tier != NOT_TESTABLE,
    )

