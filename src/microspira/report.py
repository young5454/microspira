"""Portable SPIRA result objects and report writing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SPIRAResult:
    """All tabular evidence returned by one :func:`microspira.run` call."""

    taxa: pd.DataFrame
    clusters: pd.DataFrame
    config: dict
    validation: dict
    diagnostics: dict[str, dict] = field(default_factory=dict, repr=False)

    def structured_taxa(self) -> pd.DataFrame:
        """Return taxa with FDR-positive residual spatial evidence."""
        return self.taxa.loc[self.taxa["spatial_call"] == "SPATIALLY_STRUCTURED"].copy()

    def write(self, directory: str | Path) -> Path:
        """Write tidy CSV tables and a machine-readable run manifest."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.taxa.to_csv(out / "spira_taxa.csv", index=False)
        self.clusters.to_csv(out / "spira_clusters.csv", index=False)
        manifest = {
            "config": _json_safe(self.config),
            "validation": _json_safe(self.validation),
            "n_taxa_rows": int(len(self.taxa)),
            "n_cluster_rows": int(len(self.clusters)),
        }
        (out / "spira_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return out


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

