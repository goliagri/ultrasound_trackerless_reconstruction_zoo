"""TUS-REC2025 Challenge baseline algorithm implementation."""

from usrec_zoo.algorithms.tusrec_baseline.algorithm import TUSRECBaseline
from usrec_zoo.algorithms.tusrec_baseline.config import (
    get_default_config,
    get_config_schema,
)

__all__ = [
    "TUSRECBaseline",
    "get_default_config",
    "get_config_schema",
]
