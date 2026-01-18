"""
Algorithm registry for discovering and instantiating reconstruction methods.

The registry provides a centralized way to:
- Register new algorithms
- Look up algorithms by name
- List all available algorithms

Algorithms are automatically registered when their modules are imported.
"""

from typing import Any, Callable, Dict, List, Optional, Type

from usrec_zoo.algorithms.base import AlgorithmInterface
from usrec_zoo.calibration import CalibrationData

# Global registry mapping algorithm names to classes
_ALGORITHM_REGISTRY: Dict[str, Type[AlgorithmInterface]] = {}


def register_algorithm(
    name: Optional[str] = None,
) -> Callable[[Type[AlgorithmInterface]], Type[AlgorithmInterface]]:
    """
    Decorator to register an algorithm class in the registry.

    Args:
        name: Optional name override. If not provided, uses the class's
              get_name() method.

    Returns:
        Decorator function.

    Example:
        >>> @register_algorithm("my_algo")
        ... class MyAlgorithm(AlgorithmInterface):
        ...     @classmethod
        ...     def get_name(cls):
        ...         return "my_algo"
        ...     # ... rest of implementation
    """
    def decorator(cls: Type[AlgorithmInterface]) -> Type[AlgorithmInterface]:
        algo_name = name if name is not None else cls.get_name()

        if algo_name in _ALGORITHM_REGISTRY:
            existing = _ALGORITHM_REGISTRY[algo_name]
            if existing is not cls:
                raise ValueError(
                    f"Algorithm name '{algo_name}' is already registered by "
                    f"{existing.__module__}.{existing.__name__}"
                )
            # Same class registered twice (e.g., module reload), ignore
            return cls

        _ALGORITHM_REGISTRY[algo_name] = cls
        return cls

    return decorator


def get_algorithm(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    calibration: Optional[CalibrationData] = None,
) -> AlgorithmInterface:
    """
    Get an algorithm instance by name.

    Args:
        name: Registered algorithm name.
        config: Optional configuration dict. If not provided, uses defaults.
        calibration: CalibrationData instance. Required for instantiation.

    Returns:
        Instantiated algorithm object.

    Raises:
        KeyError: If algorithm name is not registered.
        ValueError: If calibration is not provided.

    Example:
        >>> calibration = CalibrationData.from_csv("calib_matrix.csv")
        >>> algo = get_algorithm("tusrec_baseline", calibration=calibration)
    """
    if name not in _ALGORITHM_REGISTRY:
        available = ", ".join(sorted(_ALGORITHM_REGISTRY.keys()))
        raise KeyError(
            f"Unknown algorithm: '{name}'. Available: {available}"
        )

    if calibration is None:
        raise ValueError("calibration is required to instantiate an algorithm")

    cls = _ALGORITHM_REGISTRY[name]

    if config is None:
        config = cls.get_default_config()

    return cls(config=config, calibration=calibration)


def get_algorithm_class(name: str) -> Type[AlgorithmInterface]:
    """
    Get an algorithm class by name without instantiating.

    Args:
        name: Registered algorithm name.

    Returns:
        Algorithm class.

    Raises:
        KeyError: If algorithm name is not registered.
    """
    if name not in _ALGORITHM_REGISTRY:
        available = ", ".join(sorted(_ALGORITHM_REGISTRY.keys()))
        raise KeyError(
            f"Unknown algorithm: '{name}'. Available: {available}"
        )

    return _ALGORITHM_REGISTRY[name]


def list_algorithms() -> List[str]:
    """
    List all registered algorithm names.

    Returns:
        Sorted list of algorithm names.
    """
    return sorted(_ALGORITHM_REGISTRY.keys())


def get_algorithm_info(name: str) -> Dict[str, str]:
    """
    Get information about a registered algorithm.

    Args:
        name: Registered algorithm name.

    Returns:
        Dictionary with algorithm metadata.
    """
    cls = get_algorithm_class(name)
    return {
        "name": cls.get_name(),
        "description": cls.get_description(),
        "module": f"{cls.__module__}.{cls.__name__}",
    }


def list_algorithms_with_info() -> List[Dict[str, str]]:
    """
    List all registered algorithms with their metadata.

    Returns:
        List of dictionaries with algorithm information.
    """
    return [get_algorithm_info(name) for name in list_algorithms()]
