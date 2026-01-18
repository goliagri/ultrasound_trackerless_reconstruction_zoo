"""Algorithm interface and registry for reconstruction methods."""

from usrec_zoo.algorithms.base import AlgorithmInterface, ConfigSchema
from usrec_zoo.algorithms.registry import (
    register_algorithm,
    get_algorithm,
    get_algorithm_class,
    list_algorithms,
)

# Import algorithm modules to trigger registration
from usrec_zoo.algorithms import tusrec_baseline  # noqa: F401

__all__ = [
    "AlgorithmInterface",
    "ConfigSchema",
    "register_algorithm",
    "get_algorithm",
    "get_algorithm_class",
    "list_algorithms",
]
