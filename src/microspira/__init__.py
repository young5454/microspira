"""microSPIRA: count-aware spatial inference for microbial reads.

SPIRA (Spatial Pattern Inference with Residual Adjustment) is deliberately
small at its decision layer:

1. Is a taxon sufficiently supported to test?
2. Does it show residual spatial structure after technical adjustment?

Host, tissue-interface, and capture-boundary relationships are returned as
per-domain descriptors, with association assessed across independent domains.
"""

from .api import run
from .config import SPIRAConfig, SupportConfig
from .report import SPIRAResult
from .validate import ValidationError, validate
from .replicate import replicate_context

__all__ = ["run", "validate", "ValidationError", "SPIRAConfig", "SupportConfig", "SPIRAResult",
           "replicate_context"]
__version__ = "0.1.0.dev2"

