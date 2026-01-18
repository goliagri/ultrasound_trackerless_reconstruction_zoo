"""
usrec_zoo - A zoo of methods for trackerless 3D freehand ultrasound reconstruction.

This package provides:
- Core types and data structures for ultrasound reconstruction
- Calibration utilities for coordinate system transformations
- Transform utilities for 6DOF and 4x4 matrix conversions
- DDF (Displacement Field) conversion for evaluation
- Algorithm interface and registry for implementing reconstruction methods
- Data loading utilities
"""

from usrec_zoo.types import (
    ScanData,
    TransformPrediction,
    DDFOutput,
    ValidationResult,
    TrainResult,
    EvaluationResult,
)
from usrec_zoo.constants import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_PIXELS,
    DEFAULT_LANDMARKS_PER_SCAN,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "ScanData",
    "TransformPrediction",
    "DDFOutput",
    "ValidationResult",
    "TrainResult",
    "EvaluationResult",
    # Constants
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "NUM_PIXELS",
    "DEFAULT_LANDMARKS_PER_SCAN",
]
