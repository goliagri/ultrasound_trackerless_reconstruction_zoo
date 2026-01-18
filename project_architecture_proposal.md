# Architecture Proposal: Freehand 3D Ultrasound Reconstruction Algorithm Hub

**Version**: 0.2 (Revised after design review)
**Date**: 2025-01-17
**Status**: Proposal - Pending Review

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Design Goals](#2-design-goals)
3. [Architecture Overview](#3-architecture-overview)
4. [Directory Structure](#4-directory-structure)
5. [Core Abstractions](#5-core-abstractions)
6. [Interface Contracts](#6-interface-contracts)
7. [Coordinate Systems & Conventions](#7-coordinate-systems--conventions)
8. [Configuration System](#8-configuration-system)
9. [Key Design Decisions & Rationales](#9-key-design-decisions--rationales)
10. [Migration Path](#10-migration-path)
11. [Open Questions](#11-open-questions)
12. [Appendix: Current Codebase Analysis](#appendix-current-codebase-analysis)

---

## 1. Problem Statement

### 1.1 Domain

**Trackerless 3D freehand ultrasound reconstruction**: Estimate spatial transformations between consecutive 2D ultrasound frames to reconstruct 3D volumes, without relying on external optical/magnetic tracking devices.

### 1.2 Current State

The repository contains a single algorithm (TUS-REC2025 Challenge baseline) with:
- Tightly coupled components
- Hardcoded paths and magic numbers
- No clear separation between algorithm-specific and shared code
- Calibration logic spread across multiple files

### 1.3 Goal

Transform this into a **modular research hub** where:
- Multiple algorithms can be implemented, trained, and compared fairly
- Shared infrastructure (data loading, transforms, evaluation) is reusable
- New algorithms can be added with minimal boilerplate
- Results are reproducible and comparable

---

## 2. Design Goals

### 2.1 Primary Goals

| Goal | Description | Priority |
|------|-------------|----------|
| **Modularity** | Algorithms are self-contained; changes to one don't break others | P0 |
| **Fair Comparison** | Identical data splits, metrics, and evaluation procedures | P0 |
| **Extensibility** | Adding a new algorithm requires minimal changes to core code | P0 |
| **Reproducibility** | Same code + config + seed = same results | P0 |

### 2.2 Secondary Goals

| Goal | Description | Priority |
|------|-------------|----------|
| **Debuggability** | Clear error messages; easy to trace data flow | P1 |
| **Type Safety** | Catch interface mismatches at development time | P1 |
| **Rapid Iteration** | Support quick experiments without full boilerplate | P1 |
| **Documentation** | Self-documenting interfaces via type hints and docstrings | P2 |

### 2.3 Non-Goals

- **Production deployment**: This is a research codebase, not a clinical system
- **Multi-GPU distributed training**: Out of scope for initial version
- **Real-time inference**: Not optimizing for latency
- **Supporting arbitrary datasets**: Initially targeting TUS-REC data only

---

## 3. Architecture Overview

### 3.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SCRIPTS LAYER                               │
│   train.py | evaluate.py | compare.py | quick_train.py              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
┌─────────────────────────┐             ┌─────────────────────────────┐
│    ALGORITHMS LAYER     │             │      EXPERIMENTAL LAYER     │
│  (full interface)       │             │   (quick iteration)         │
│  tusrec_baseline/       │             │   single-file experiments   │
│  recurrent_net/         │             │   relaxed interface rules   │
└─────────────────────────┘             └─────────────────────────────┘
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SHARED COMPONENTS LAYER                          │
│   (opt-in, used by 2+ algorithms but not truly universal)           │
│   attention_blocks.py | custom_losses.py | augmentation.py          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          CORE LAYER                                 │
│   data/ | transforms/ | evaluation/ | calibration/ | types/         │
│   Shared, algorithm-agnostic infrastructure                         │
└─────────────────────────────────────────────────────────────────────┘
```

**Dependency rule:** Scripts → Algorithms/Experimental → Shared Components → Core (never upward)

### 3.2 Key Architectural Principle: Algorithms Predict Transforms, Framework Converts to DDFs

```
                    TRAINING FLOW

H5 Files ──► DataLoader ──► Algorithm.train() ──► Checkpoint
                │                   │
                │                   └──► SharedTrainer (default)
                │                        OR custom loop (escape hatch)
                │
                └──► Canonical Splits (reproducible)


                    INFERENCE FLOW

                    ┌─────────────────────┐
Scan ──► Algorithm.predict() ──► TransformPrediction
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │  Framework Layer    │
                              │  (DDF Converter)    │
                              └─────────────────────┘
                                         │
                                         ▼
                                    DDFOutput ──► Evaluation Metrics


                    EVALUATION FLOW

TransformPrediction ──► DDFConverter ──► DDFOutput ──► Metrics
        │                    │                            │
        │                    └── Calibration              └── Ground Truth
        │
        └── Available for debugging/visualization
```

---

## 4. Directory Structure

```
ultrasound_trackerless_reconstruction_zoo/
│
├── src/                               # Installable package (src layout)
│   └── usrec_zoo/                     # Package name
│       ├── __init__.py
│       ├── types.py                   # Dataclasses: ScanData, TransformPrediction, etc.
│       ├── constants.py               # IMAGE_HEIGHT, IMAGE_WIDTH, etc.
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py              # ScanDataset class
│       │   ├── splits.py              # Canonical train/val/test splits
│       │   └── h5_utils.py            # H5 file reading utilities
│       │
│       ├── calibration/
│       │   ├── __init__.py
│       │   └── calibration.py         # CalibrationData class (single source)
│       │
│       ├── transforms/
│       │   ├── __init__.py
│       │   ├── rigid.py               # 6DOF ↔ 4×4 conversions
│       │   ├── coordinates.py         # Pixel ↔ mm ↔ tool conversions
│       │   └── accumulation.py        # Sequential transform composition
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── ddf.py                 # Transform → DDF conversion (FRAMEWORK OWNS THIS)
│       │   ├── metrics.py             # Distance metrics
│       │   └── benchmark.py           # Cross-algorithm comparison
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── shared_trainer.py      # Default training loop
│       │   ├── callbacks.py           # Checkpointing, logging, early stopping
│       │   └── results.py             # TrainResult, ValidationResult dataclasses
│       │
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── seeding.py             # Reproducibility utilities
│       │   └── checkpointing.py       # Versioned checkpoint save/load
│       │
│       ├── algorithms/                # Algorithm implementations
│       │   ├── __init__.py
│       │   ├── base.py                # AlgorithmInterface ABC
│       │   ├── registry.py            # Algorithm registration
│       │   │
│       │   ├── tusrec_baseline/       # EfficientNet-B1 baseline
│       │   │   ├── __init__.py
│       │   │   ├── algorithm.py       # TUSRECBaseline(AlgorithmInterface)
│       │   │   ├── model.py           # Network architecture
│       │   │   └── config.py          # Default hyperparameters + schema
│       │   │
│       │   └── [future_algorithm]/    # Same structure
│       │
│       ├── experimental/              # Quick experiments (relaxed rules)
│       │   ├── __init__.py
│       │   └── README.md              # Guidelines for experimental code
│       │
│       └── shared_components/         # Opt-in code used by 2+ algorithms
│           ├── __init__.py
│           ├── attention.py           # Attention blocks
│           └── augmentation.py        # Shared augmentation utilities
│
├── configs/                           # YAML configuration files
│   ├── data/
│   │   └── tusrec2024.yaml            # Data paths, splits config
│   ├── algorithms/
│   │   └── tusrec_baseline.yaml       # Algorithm hyperparameters
│   └── env/
│       ├── local.yaml                 # Local development paths
│       ├── cluster.yaml               # HPC cluster paths
│       └── docker.yaml                # Docker submission paths
│
├── scripts/                           # Entry points
│   ├── train.py                       # python scripts/train.py --algorithm tusrec_baseline
│   ├── evaluate.py                    # python scripts/evaluate.py --algorithm tusrec_baseline
│   ├── compare.py                     # python scripts/compare.py --algorithms baseline,transformer
│   ├── quick_train.py                 # For experimental/ code
│   ├── generate_submission.py         # Docker submission builder
│   └── download_data.py               # Data download utility
│
├── experiments/                       # Training outputs (gitignored)
│   └── {algorithm}_{timestamp}/
│       ├── checkpoints/
│       ├── logs/
│       └── config.yaml                # Frozen config for reproducibility
│
├── tests/                             # Unit and integration tests
│   ├── unit/
│   │   ├── test_transforms.py
│   │   ├── test_ddf_conversion.py
│   │   └── test_config_validation.py
│   ├── integration/
│   │   └── test_training_loop.py
│   └── conftest.py                    # Shared fixtures
│
├── notebooks/                         # Exploration and visualization
│   └── visualize_predictions.ipynb
│
├── docs/                              # Documentation
│   └── adding_algorithms.md
│
├── data -> ../tus-rec-data/           # Symlink to data (gitignored)
│
├── TUS-REC2025-Challenge_baseline/    # Original code (to be deprecated)
│
├── pyproject.toml                     # Package configuration
├── CLAUDE.md                          # Project instructions
├── style.md                           # Code style guide
└── project_architecture_proposal.md   # This document
```

### 4.1 Rationale for Structure

| Directory | Rationale |
|-----------|-----------|
| `src/usrec_zoo/` | src-layout prevents import issues; project-specific name avoids confusion |
| `algorithms/` | Each algorithm is a self-contained package with consistent structure |
| `experimental/` | Escape hatch for quick iteration without full interface compliance |
| `shared_components/` | Code used by 2+ algorithms but not universal enough for core |
| `configs/` | Separates configuration from code; enables experiment tracking |
| `experiments/` | Keeps training artifacts organized and reproducible |
| `scripts/` | Single entry points reduce confusion about how to run things |

---

## 5. Core Abstractions

### 5.1 Type Definitions (`src/usrec_zoo/types.py`)

```python
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================

IMAGE_HEIGHT: int = 480
IMAGE_WIDTH: int = 640
NUM_PIXELS: int = IMAGE_HEIGHT * IMAGE_WIDTH  # 307,200
DEFAULT_LANDMARKS_PER_SCAN: int = 20  # Actual count in TUS-REC data


# ============================================================================
# INPUT DATA TYPES
# ============================================================================

@dataclass(frozen=True)
class ScanData:
    """
    A single ultrasound scan with all frames and metadata.

    This is the canonical input format for algorithms.
    All algorithms receive data in this format.

    WARNING: While this dataclass is frozen, numpy array contents can still
    be modified in-place. Treat arrays as immutable by convention.
    Use .copy() methods if you need to modify array data.
    """
    frames: np.ndarray          # Shape: [N, 480, 640], dtype: uint8
    transforms: np.ndarray      # Shape: [N, 4, 4], dtype: float32 (tracker space)
    landmarks: np.ndarray       # Shape: [num_landmarks, 3], dtype: int64
                                # Each row: (frame_idx, x, y)
                                # frame_idx is 0-indexed
                                # x, y are 1-indexed (calibration convention)
    scan_id: str                # e.g., "LH_Par_C_DtP"
    subject_id: str             # e.g., "000"

    def __post_init__(self):
        """Validate shapes and types at construction time."""
        # Use if/raise instead of assert (assertions can be disabled with -O)
        if self.frames.ndim != 3:
            raise ValueError(f"frames must be 3D, got {self.frames.ndim}D")
        if self.frames.shape[1:] != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(f"frames must be [N, {IMAGE_HEIGHT}, {IMAGE_WIDTH}], got {self.frames.shape}")
        if self.transforms.shape[0] != self.frames.shape[0]:
            raise ValueError(f"transforms count ({self.transforms.shape[0]}) must match frames count ({self.frames.shape[0]})")
        if self.transforms.shape[1:] != (4, 4):
            raise ValueError(f"transforms must be [N, 4, 4], got {self.transforms.shape}")
        if self.landmarks.ndim != 2 or self.landmarks.shape[1] != 3:
            raise ValueError(f"landmarks must be [num_landmarks, 3], got {self.landmarks.shape}")
        if self.frames.shape[0] < 2:
            raise ValueError(f"Need at least 2 frames, got {self.frames.shape[0]}")

    @property
    def num_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def num_landmarks(self) -> int:
        return self.landmarks.shape[0]

    def copy(self, **changes) -> 'ScanData':
        """Create a copy with optional field changes and deep-copied arrays."""
        return ScanData(
            frames=changes.get('frames', self.frames.copy()),
            transforms=changes.get('transforms', self.transforms.copy()),
            landmarks=changes.get('landmarks', self.landmarks.copy()),
            scan_id=changes.get('scan_id', self.scan_id),
            subject_id=changes.get('subject_id', self.subject_id),
        )


# ============================================================================
# ALGORITHM OUTPUT TYPES
# ============================================================================

@dataclass(frozen=True)
class TransformPrediction:
    """
    Predicted transforms between frames.

    This is what algorithms return from predict().
    The framework converts this to DDFOutput for evaluation.

    All transforms are in image-mm coordinate space.
    """
    local_transforms: np.ndarray   # Shape: [N-1, 4, 4], frame i → frame i-1
    global_transforms: np.ndarray  # Shape: [N-1, 4, 4], frame i → frame 0

    def __post_init__(self):
        if self.local_transforms.ndim != 3:
            raise ValueError(f"local_transforms must be 3D, got {self.local_transforms.ndim}D")
        if self.global_transforms.ndim != 3:
            raise ValueError(f"global_transforms must be 3D, got {self.global_transforms.ndim}D")
        if self.local_transforms.shape != self.global_transforms.shape:
            raise ValueError(f"local and global transforms must have same shape")
        if self.local_transforms.shape[1:] != (4, 4):
            raise ValueError(f"transforms must be [N-1, 4, 4], got {self.local_transforms.shape}")
        if self.local_transforms.dtype != np.float32:
            raise TypeError(f"transforms must be float32, got {self.local_transforms.dtype}")

    @property
    def num_frame_pairs(self) -> int:
        return self.local_transforms.shape[0]


# ============================================================================
# EVALUATION OUTPUT TYPES (Framework-generated, not algorithm-generated)
# ============================================================================

@dataclass(frozen=True)
class DDFOutput:
    """
    Displacement Dense Fields - the final output format for evaluation.

    This is generated by the FRAMEWORK from TransformPrediction, not by algorithms.
    The evaluation system uses this format for computing metrics.

    All displacements are in millimeters (mm).

    Pixel Ordering Convention:
        Pixels are stored in row-major order: pixel[y, x] maps to index y*640 + x
        Index 0 corresponds to pixel (x=1, y=1) in 1-based coordinates
        Use the helper methods to_2d() and get_pixel() for safe access.
    """
    global_pixels: np.ndarray     # GP: [N-1, 3, NUM_PIXELS], frame i → frame 0
    global_landmarks: np.ndarray  # GL: [3, num_landmarks], aggregated over frames
    local_pixels: np.ndarray      # LP: [N-1, 3, NUM_PIXELS], frame i → frame i-1
    local_landmarks: np.ndarray   # LL: [3, num_landmarks], aggregated over frames

    # Metadata for reshaping and validation
    image_shape: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)

    def __post_init__(self):
        """Validate output shapes strictly."""
        n_minus_1 = self.global_pixels.shape[0]
        expected_pixels = self.image_shape[0] * self.image_shape[1]

        # Validate shapes
        if self.global_pixels.shape != (n_minus_1, 3, expected_pixels):
            raise ValueError(f"GP shape must be [{n_minus_1}, 3, {expected_pixels}], got {self.global_pixels.shape}")
        if self.local_pixels.shape != (n_minus_1, 3, expected_pixels):
            raise ValueError(f"LP shape must be [{n_minus_1}, 3, {expected_pixels}], got {self.local_pixels.shape}")
        if self.global_landmarks.ndim != 2 or self.global_landmarks.shape[0] != 3:
            raise ValueError(f"GL must be [3, num_landmarks], got {self.global_landmarks.shape}")
        if self.local_landmarks.ndim != 2 or self.local_landmarks.shape[0] != 3:
            raise ValueError(f"LL must be [3, num_landmarks], got {self.local_landmarks.shape}")

        # Validate dtypes
        if self.global_pixels.dtype != np.float32:
            raise TypeError(f"GP must be float32, got {self.global_pixels.dtype}")

    def validate_values(self) -> 'DDFOutput':
        """
        Check for NaN/Inf values. Call explicitly when needed (expensive for large arrays).
        Returns self for chaining.
        """
        if not np.isfinite(self.global_pixels).all():
            raise ValueError("GP contains NaN or Inf values")
        if not np.isfinite(self.global_landmarks).all():
            raise ValueError("GL contains NaN or Inf values")
        if not np.isfinite(self.local_pixels).all():
            raise ValueError("LP contains NaN or Inf values")
        if not np.isfinite(self.local_landmarks).all():
            raise ValueError("LL contains NaN or Inf values")
        return self

    def global_pixels_2d(self) -> np.ndarray:
        """Reshape GP to [N-1, 3, H, W] for spatial operations."""
        return self.global_pixels.reshape(-1, 3, *self.image_shape)

    def local_pixels_2d(self) -> np.ndarray:
        """Reshape LP to [N-1, 3, H, W] for spatial operations."""
        return self.local_pixels.reshape(-1, 3, *self.image_shape)

    def get_pixel_displacement(self, frame_idx: int, x: int, y: int,
                                local: bool = False) -> np.ndarray:
        """
        Get displacement for a specific pixel (safe accessor).

        Args:
            frame_idx: 0-indexed frame number (0 to N-2)
            x: 1-indexed x coordinate (1 to 640)
            y: 1-indexed y coordinate (1 to 480)
            local: If True, return local displacement; else global

        Returns:
            [3,] array of (dx, dy, dz) displacement in mm
        """
        flat_idx = (y - 1) * self.image_shape[1] + (x - 1)
        if local:
            return self.local_pixels[frame_idx, :, flat_idx]
        return self.global_pixels[frame_idx, :, flat_idx]


# ============================================================================
# TRAINING RESULT TYPES
# ============================================================================

@dataclass
class ValidationResult:
    """Result from running validation."""
    metrics: Dict[str, float]                           # e.g., {'loss': 0.5, 'point_distance': 2.3}
    per_scan_metrics: Optional[Dict[str, Dict[str, float]]] = None  # Per-scan breakdown

    @property
    def loss(self) -> float:
        return self.metrics.get('loss', float('inf'))


@dataclass
class TrainResult:
    """Result from training an algorithm."""
    best_checkpoint_path: str
    final_checkpoint_path: str
    train_history: Dict[str, List[float]]   # e.g., {'loss': [0.9, 0.7, ...]}
    val_history: Dict[str, List[float]]     # e.g., {'loss': [0.8, 0.6, ...]}
    total_epochs: int
    training_time_seconds: float
    best_epoch: int
    final_val_metrics: Dict[str, float]

    def summary(self) -> str:
        """Human-readable training summary."""
        return (
            f"Training completed: {self.total_epochs} epochs in {self.training_time_seconds:.1f}s\n"
            f"Best epoch: {self.best_epoch}\n"
            f"Final validation metrics: {self.final_val_metrics}\n"
            f"Best checkpoint: {self.best_checkpoint_path}"
        )


# ============================================================================
# EVALUATION RESULT TYPES
# ============================================================================

@dataclass
class EvaluationResult:
    """Evaluation metrics for a single scan."""
    algorithm_name: str
    scan_id: str
    subject_id: str

    # Distance errors in mm
    global_pixel_error: float      # GPE
    global_landmark_error: float   # GLE
    local_pixel_error: float       # LPE
    local_landmark_error: float    # LLE

    # Timing
    inference_time_seconds: float

    # Metadata
    num_frames: int = 0


@dataclass
class BenchmarkResult:
    """Aggregated evaluation across multiple scans/algorithms."""
    results: List[EvaluationResult]

    def to_dataframe(self):
        """Convert to pandas DataFrame for analysis."""
        import pandas as pd
        return pd.DataFrame([vars(r) for r in self.results])

    def summary_by_algorithm(self):
        """Return mean metrics per algorithm."""
        df = self.to_dataframe()
        return df.groupby('algorithm_name').agg({
            'global_pixel_error': ['mean', 'std'],
            'global_landmark_error': ['mean', 'std'],
            'local_pixel_error': ['mean', 'std'],
            'local_landmark_error': ['mean', 'std'],
            'inference_time_seconds': 'mean'
        })
```

### 5.2 Rationale for Type Design

| Design Choice | Rationale |
|---------------|-----------|
| `frozen=True` dataclasses | Documents immutability intent; prevents attribute reassignment |
| Warning about numpy mutability | Frozen doesn't prevent `arr[0] = x`; users must understand this |
| `copy()` method | Explicit way to create modified copies |
| Validation via `if/raise` | Assertions can be disabled with `-O`; explicit raises always work |
| Separate `TransformPrediction` and `DDFOutput` | Algorithms predict transforms; framework converts to DDFs |
| `DDFOutput` has helper methods | Safe accessors prevent pixel ordering bugs |
| Optional `validate_values()` | NaN checking is expensive; call only when needed |
| `TrainResult` and `ValidationResult` | Structured returns instead of `Dict[str, Any]` |

---

## 6. Interface Contracts

### 6.1 Algorithm Interface (`src/usrec_zoo/algorithms/base.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Set, Type
from pathlib import Path
import torch

from usrec_zoo.types import ScanData, TransformPrediction, TrainResult, ValidationResult
from usrec_zoo.calibration import CalibrationData


@dataclass
class ConfigSchema:
    """
    Lightweight schema for config validation.

    Provides typo detection and basic type checking without
    requiring heavy dependencies like Pydantic.
    """
    required: Set[str]
    optional: Set[str]
    types: Dict[str, Type]
    descriptions: Dict[str, str] = None  # For documentation

    def __post_init__(self):
        if self.descriptions is None:
            self.descriptions = {}

    def validate(self, config: Dict[str, Any]) -> None:
        """
        Validate config dict against schema.

        Raises:
            ValueError: If required keys missing or unknown keys present
            TypeError: If values have wrong types
        """
        # Check for missing required keys
        missing = self.required - set(config.keys())
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

        # Check for unknown keys (typo detection)
        valid_keys = self.required | self.optional
        unknown = set(config.keys()) - valid_keys
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}. Valid keys: {valid_keys}")

        # Check types
        for key, expected_type in self.types.items():
            if key in config and not isinstance(config[key], expected_type):
                raise TypeError(
                    f"Config '{key}' expected {expected_type.__name__}, "
                    f"got {type(config[key]).__name__}"
                )


class AlgorithmInterface(ABC):
    """
    Base class that all algorithms must implement.

    Design Principles:
    1. Calibration is stored at construction (not passed to every method)
    2. Algorithms return TransformPrediction (not DDFOutput)
    3. Framework converts TransformPrediction → DDFOutput for evaluation
    4. Training uses shared trainer by default, with escape hatch for custom
    5. Checkpoints are self-contained (include config + calibration hash)
    """

    def __init__(self, config: Optional[Dict[str, Any]], calibration: CalibrationData):
        """
        Initialize algorithm with configuration and calibration.

        Args:
            config: Algorithm-specific configuration. If None, uses get_default_config().
            calibration: Calibration data for coordinate transforms. Bound to instance.
        """
        self.config = config if config is not None else self.get_default_config()
        self.calibration = calibration

        # Validate config against schema
        schema = self.get_config_schema()
        schema.validate(self.config)

    # ========================================================================
    # REQUIRED: These methods MUST be implemented
    # ========================================================================

    @abstractmethod
    def predict(self, scan: ScanData, device: torch.device) -> TransformPrediction:
        """
        Predict frame-to-frame transforms.

        Args:
            scan: Full scan data (all frames)
            device: GPU/CPU device

        Returns:
            TransformPrediction with local and global transforms

        Notes:
            - Returns TRANSFORMS, not DDFs
            - Framework handles DDF conversion for evaluation
            - Uses self.calibration internally as needed
        """
        pass

    @abstractmethod
    def validate(
        self,
        val_scans: List[ScanData],
        device: torch.device
    ) -> ValidationResult:
        """
        Run validation loop without updating weights.

        Args:
            val_scans: Validation data
            device: GPU/CPU device

        Returns:
            ValidationResult with metrics
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: Path, device: torch.device) -> None:
        """
        Load model weights from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load weights onto

        Notes:
            - Should verify calibration hash matches if stored in checkpoint
            - Should raise clear error if checkpoint is incompatible
        """
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Save model weights to a checkpoint.

        Args:
            checkpoint_path: Where to save

        Notes:
            - Must save: weights, config, calibration hash, training step
            - Recommend format:
              {'model_state': ..., 'config': ..., 'calibration_hash': ..., 'epoch': ...}
        """
        pass

    @classmethod
    @abstractmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Return default hyperparameters for this algorithm.

        Returns:
            Dict of hyperparameter names to default values
        """
        pass

    @classmethod
    @abstractmethod
    def get_config_schema(cls) -> ConfigSchema:
        """
        Return schema for configuration validation.

        Returns:
            ConfigSchema with required/optional keys and types
        """
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Return the canonical name of this algorithm.

        Returns:
            String identifier (e.g., "tusrec_baseline", "recurrent_transformer")
        """
        pass

    # ========================================================================
    # TRAINING: Default uses shared trainer, override for custom
    # ========================================================================

    def train(
        self,
        train_scans: List[ScanData],
        val_scans: List[ScanData],
        output_dir: Path,
        device: torch.device
    ) -> TrainResult:
        """
        Train the algorithm.

        Default implementation uses SharedTrainer. Override for custom training loops.

        Args:
            train_scans: Training data (full scans)
            val_scans: Validation data
            output_dir: Where to save checkpoints and logs
            device: GPU/CPU device

        Returns:
            TrainResult with training summary
        """
        from usrec_zoo.training.shared_trainer import SharedTrainer
        trainer = SharedTrainer(self, self.config, output_dir, device)
        return trainer.train(train_scans, val_scans)

    @property
    def requires_custom_training(self) -> bool:
        """
        Override to return True if this algorithm needs custom training.

        If True, must override train() with custom implementation.
        Should also provide custom_training_justification.
        """
        return False

    @property
    def custom_training_justification(self) -> str:
        """
        If requires_custom_training is True, explain why.

        This creates documentation/audit trail for non-standard training.
        """
        return ""

    # ========================================================================
    # OPTIONAL: Can be overridden for custom behavior
    # ========================================================================

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Path,
        calibration: CalibrationData,
        device: torch.device,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> 'AlgorithmInterface':
        """
        Factory method to load a pretrained algorithm.

        Args:
            checkpoint_path: Path to saved checkpoint
            calibration: Calibration data (validated against checkpoint)
            device: Device to load onto
            config_overrides: Optional config values to override

        Returns:
            Initialized algorithm with loaded weights
        """
        # Load checkpoint to get config
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', cls.get_default_config())

        if config_overrides:
            config = {**config, **config_overrides}

        instance = cls(config, calibration)
        instance.load_checkpoint(checkpoint_path, device)
        return instance

    def get_model(self) -> Optional[torch.nn.Module]:
        """
        Return underlying model for inspection.

        WARNING: Modifications to the returned model may break algorithm state.
        Returns None if algorithm doesn't expose a single model.
        """
        return None

    def predict_batch(
        self,
        scans: List[ScanData],
        device: torch.device,
        batch_size: int = 8
    ) -> List[TransformPrediction]:
        """
        Run batched inference. Override for efficiency.

        Default implementation calls predict() in a loop.
        """
        return [self.predict(scan, device) for scan in scans]
```

### 6.2 Why This Interface?

| Design Decision | Rationale | Change from v0.1 |
|-----------------|-----------|------------------|
| **Calibration at construction** | Prevents train/predict mismatch; cleaner signatures | NEW |
| **`predict()` returns `TransformPrediction`** | Algorithms return their natural output; framework converts to DDF | CHANGED |
| **Separate `validate()` method** | Needed for checkpoint selection and HP tuning | NEW |
| **Default `train()` uses SharedTrainer** | Most algorithms use similar training; escape hatch available | NEW |
| **`ConfigSchema` for validation** | Catches typos and type errors at config load time | NEW |
| **`from_pretrained()` factory** | Cleaner than construct-then-load pattern | NEW |
| **Structured return types** | `TrainResult`/`ValidationResult` instead of `Dict` | NEW |

### 6.3 Calibration Interface (`src/usrec_zoo/calibration/calibration.py`)

```python
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import torch


@dataclass(frozen=True)
class CalibrationData:
    """
    Calibration matrices for coordinate system conversions.

    Coordinate Systems:
    1. Image (pixels): [0, 640) × [0, 480) integer coordinates
    2. Image (mm): Physical coordinates in millimeters
    3. Tool: Optical tracker coordinate system

    Transform Chain:
        pixels → mm: tform_pixel_to_mm
        mm → tool: tform_mm_to_tool
        pixels → tool: tform_pixel_to_tool (= mm_to_tool @ pixel_to_mm)

    All transforms use LEFT multiplication:
        point_in_target = T @ point_in_source
    """

    tform_pixel_to_mm: np.ndarray      # [4, 4] Scale transform
    tform_mm_to_tool: np.ndarray       # [4, 4] Rigid transform
    tform_pixel_to_tool: np.ndarray    # [4, 4] Combined

    # Pre-computed inverses
    tform_mm_to_pixel: np.ndarray      # [4, 4]
    tform_tool_to_mm: np.ndarray       # [4, 4]
    tform_tool_to_pixel: np.ndarray    # [4, 4]

    def __post_init__(self):
        """Validate all matrices are 4×4 float32."""
        for name in ['tform_pixel_to_mm', 'tform_mm_to_tool', 'tform_pixel_to_tool',
                     'tform_mm_to_pixel', 'tform_tool_to_mm', 'tform_tool_to_pixel']:
            matrix = getattr(self, name)
            if matrix.shape != (4, 4):
                raise ValueError(f"{name} must be 4×4, got {matrix.shape}")
            if matrix.dtype != np.float32:
                raise TypeError(f"{name} must be float32, got {matrix.dtype}")

    def content_hash(self) -> str:
        """
        Return hash of calibration content for checkpoint validation.

        Used to verify that loaded models were trained with compatible calibration.
        """
        data = np.concatenate([
            self.tform_pixel_to_mm.flatten(),
            self.tform_mm_to_tool.flatten()
        ])
        return hashlib.sha256(data.tobytes()).hexdigest()[:16]

    @classmethod
    def from_csv(cls, csv_path: Path) -> 'CalibrationData':
        """
        Load calibration from CSV file.

        Expected format (TUS-REC convention):
            Row 0: Header
            Rows 1-4: tform_pixel_to_mm (4×4)
            Row 5: Header
            Rows 6-9: tform_mm_to_tool (4×4)
        """
        data = np.empty((8, 4), dtype=np.float32)
        with open(csv_path, 'r') as f:
            lines = [line.strip().split(',') for line in f.readlines()]
            data[0:4, :] = np.array(lines[1:5]).astype(np.float32)
            data[4:8, :] = np.array(lines[6:10]).astype(np.float32)

        pixel_to_mm = data[0:4, :]
        mm_to_tool = data[4:8, :]
        pixel_to_tool = mm_to_tool @ pixel_to_mm

        return cls(
            tform_pixel_to_mm=pixel_to_mm,
            tform_mm_to_tool=mm_to_tool,
            tform_pixel_to_tool=pixel_to_tool,
            tform_mm_to_pixel=np.linalg.inv(pixel_to_mm).astype(np.float32),
            tform_tool_to_mm=np.linalg.inv(mm_to_tool).astype(np.float32),
            tform_tool_to_pixel=np.linalg.inv(pixel_to_tool).astype(np.float32)
        )

    def to_torch(self, device: torch.device) -> 'CalibrationDataTorch':
        """Convert to PyTorch tensors on specified device."""
        return CalibrationDataTorch(
            tform_pixel_to_mm=torch.from_numpy(self.tform_pixel_to_mm).to(device),
            tform_mm_to_tool=torch.from_numpy(self.tform_mm_to_tool).to(device),
            tform_pixel_to_tool=torch.from_numpy(self.tform_pixel_to_tool).to(device),
            tform_mm_to_pixel=torch.from_numpy(self.tform_mm_to_pixel).to(device),
            tform_tool_to_mm=torch.from_numpy(self.tform_tool_to_mm).to(device),
            tform_tool_to_pixel=torch.from_numpy(self.tform_tool_to_pixel).to(device)
        )


@dataclass(frozen=True)
class CalibrationDataTorch:
    """PyTorch version of CalibrationData for GPU operations."""
    tform_pixel_to_mm: torch.Tensor
    tform_mm_to_tool: torch.Tensor
    tform_pixel_to_tool: torch.Tensor
    tform_mm_to_pixel: torch.Tensor
    tform_tool_to_mm: torch.Tensor
    tform_tool_to_pixel: torch.Tensor
```

### 6.4 DDF Converter (`src/usrec_zoo/evaluation/ddf.py`)

```python
"""
DDF Conversion Module

This module is OWNED BY THE FRAMEWORK, not by algorithms.
All transform-to-DDF conversion goes through here to ensure consistency.
"""

import numpy as np
import torch
from typing import Tuple

from usrec_zoo.types import TransformPrediction, DDFOutput, IMAGE_HEIGHT, IMAGE_WIDTH
from usrec_zoo.calibration import CalibrationData


def reference_image_points(
    image_shape: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)
) -> torch.Tensor:
    """
    Generate reference points for all pixels in homogeneous coordinates.

    Returns:
        Tensor of shape [4, H*W] where each column is [x, y, 0, 1]^T
        Coordinates are 1-indexed to match calibration convention.

    Pixel Ordering:
        Points are in row-major order: index i corresponds to
        y = (i // W) + 1, x = (i % W) + 1
    """
    h, w = image_shape

    # Create 1-indexed coordinate grids
    y_coords = torch.arange(1, h + 1, dtype=torch.float32)
    x_coords = torch.arange(1, w + 1, dtype=torch.float32)

    # Create meshgrid in row-major order (y changes slower than x)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Flatten to [H*W] arrays
    x_flat = xx.flatten()  # [H*W]
    y_flat = yy.flatten()  # [H*W]

    # Stack into homogeneous coordinates [4, H*W]
    points = torch.stack([
        x_flat,
        y_flat,
        torch.zeros_like(x_flat),
        torch.ones_like(x_flat)
    ], dim=0)

    return points


def transforms_to_ddf(
    prediction: TransformPrediction,
    landmarks: np.ndarray,
    calibration: CalibrationData,
    device: torch.device = torch.device('cpu')
) -> DDFOutput:
    """
    Convert transform predictions to Displacement Dense Fields.

    This function is the SINGLE SOURCE OF TRUTH for DDF conversion.
    All algorithms use this function via the evaluation framework.

    Args:
        prediction: TransformPrediction from algorithm
        landmarks: [num_landmarks, 3] array with (frame_idx, x, y), 1-indexed x/y
        calibration: Calibration matrices
        device: Device for computation

    Returns:
        DDFOutput ready for evaluation
    """
    calib = calibration.to_torch(device)

    # Get reference points [4, H*W]
    image_points = reference_image_points().to(device)

    # Convert transforms to torch
    global_transforms = torch.from_numpy(prediction.global_transforms).to(device)
    local_transforms = torch.from_numpy(prediction.local_transforms).to(device)

    # Scale points to mm
    points_mm = calib.tform_pixel_to_mm @ image_points  # [4, H*W]
    reference_points_mm = points_mm[:3, :]  # [3, H*W]

    # Compute global DDFs
    # transformed_points[i] = global_transforms[i] @ points_mm
    transformed_global = torch.matmul(global_transforms, points_mm)  # [N-1, 4, H*W]
    global_pixels_ddf = transformed_global[:, :3, :] - reference_points_mm  # [N-1, 3, H*W]

    # Compute local DDFs
    transformed_local = torch.matmul(local_transforms, points_mm)  # [N-1, 4, H*W]
    local_pixels_ddf = transformed_local[:, :3, :] - reference_points_mm  # [N-1, 3, H*W]

    # Extract landmark DDFs
    # landmarks[:, 0] is frame_idx (0-indexed), landmarks[:, 1:] is (x, y) (1-indexed)
    landmarks_torch = torch.from_numpy(landmarks).to(device)
    h, w = IMAGE_HEIGHT, IMAGE_WIDTH

    # Compute flat indices for landmarks
    # frame_idx - 1 because we index into [N-1] array (frame 0 has no displacement)
    landmark_frame_indices = landmarks_torch[:, 0].long() - 1  # Adjust for 0-indexed DDF
    landmark_flat_indices = ((landmarks_torch[:, 2] - 1) * w + (landmarks_torch[:, 1] - 1)).long()

    # Handle edge case: landmarks on frame 0 have no displacement
    valid_mask = landmark_frame_indices >= 0

    global_landmarks = torch.zeros((3, landmarks.shape[0]), dtype=torch.float32, device=device)
    local_landmarks = torch.zeros((3, landmarks.shape[0]), dtype=torch.float32, device=device)

    if valid_mask.any():
        valid_frames = landmark_frame_indices[valid_mask]
        valid_pixels = landmark_flat_indices[valid_mask]
        valid_indices = torch.arange(landmarks.shape[0], device=device)[valid_mask]

        for i, (frame_idx, pixel_idx, out_idx) in enumerate(zip(valid_frames, valid_pixels, valid_indices)):
            global_landmarks[:, out_idx] = global_pixels_ddf[frame_idx, :, pixel_idx]
            local_landmarks[:, out_idx] = local_pixels_ddf[frame_idx, :, pixel_idx]

    return DDFOutput(
        global_pixels=global_pixels_ddf.cpu().numpy().astype(np.float32),
        global_landmarks=global_landmarks.cpu().numpy().astype(np.float32),
        local_pixels=local_pixels_ddf.cpu().numpy().astype(np.float32),
        local_landmarks=local_landmarks.cpu().numpy().astype(np.float32),
        image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )
```

---

## 7. Coordinate Systems & Conventions

### 7.1 Coordinate System Definitions

```
IMAGE SPACE (PIXELS)                    IMAGE SPACE (MM)
┌─────────────────────────┐             ┌─────────────────────────┐
│ Origin: top-left        │             │ Origin: top-left        │
│ X: 1 → 640 (right)      │  ───────►   │ X: scaled (right)       │
│ Y: 1 → 480 (down)       │  pixel_to_mm│ Y: scaled (down)        │
│ Z: 0 (into page)        │             │ Z: 0 (into page)        │
│ (1-indexed for calib)   │             │                         │
└─────────────────────────┘             └─────────────────────────┘
        │                                        │
        │ pixel_to_tool                          │ mm_to_tool
        ▼                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TOOL SPACE (TRACKER)                        │
│ Origin: Optical tracker reference point                         │
│ Axes: Defined by tracker manufacturer                           │
│ Units: millimeters                                              │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Transform Conventions

| Convention | Definition |
|------------|------------|
| **Left multiplication** | `point_target = T @ point_source` |
| **Transform naming** | `T_source_to_target` or `T_{target←source}` |
| **Homogeneous coordinates** | Points are [x, y, z, 1]ᵀ |
| **Euler angles** | ZYX convention (yaw, pitch, roll) via pytorch3d |
| **Reference frame** | Frame 0 is always the reference |

### 7.3 DDF Conventions

| Convention | Definition |
|------------|------------|
| **Displacement direction** | From current frame TO reference (not from reference) |
| **Global DDF** | Displacement from frame i to frame 0 |
| **Local DDF** | Displacement from frame i to frame i-1 |
| **Units** | All DDFs are in millimeters |
| **Pixel ordering** | Row-major: index = (y-1) * 640 + (x-1) for 1-indexed coords |

### 7.4 Landmark Conventions

| Field | Convention |
|-------|------------|
| `frame_idx` | 0-indexed (0 to N-1) |
| `x` | 1-indexed (1 to 640) — calibration convention |
| `y` | 1-indexed (1 to 480) — calibration convention |

**Important**: When indexing into arrays, subtract 1 from x and y.

---

## 8. Configuration System

### 8.1 Configuration Hierarchy

```
configs/
├── data/
│   └── tusrec2024.yaml          # Data-specific config
├── algorithms/
│   └── tusrec_baseline.yaml     # Algorithm hyperparameters
└── env/
    ├── local.yaml               # Local development
    ├── cluster.yaml             # HPC cluster
    └── docker.yaml              # Docker submission
```

### 8.2 Data Configuration (`configs/data/tusrec2024.yaml`)

```yaml
# Data configuration for TUS-REC 2024 dataset

dataset:
  name: "tusrec2024"
  root_path: "${DATA_ROOT:data}/frames_transfs"
  calibration_path: "${DATA_ROOT:data}/calib_matrix.csv"
  landmarks_path: "${DATA_ROOT:data}/landmarks"

  # Image properties (fixed for this dataset)
  image_height: 480
  image_width: 640

splits:
  # Subject-level splits for reproducibility
  # 50 subjects total: 40 train, 5 val, 5 test
  seed: 42
  # Subjects are shuffled with this seed, then split
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

  # For cross-validation (secondary metric)
  cv_folds: 5

validation:
  # Primary: fixed split for rapid iteration
  primary: "fixed_split"
  # Secondary: 5-fold CV for publication claims
  secondary: "cross_validation"
```

### 8.3 Algorithm Configuration (`configs/algorithms/tusrec_baseline.yaml`)

```yaml
# Configuration for TUS-REC Baseline (EfficientNet-B1)

algorithm:
  name: "tusrec_baseline"
  version: "1.0"

model:
  backbone: "efficientnet_b1"
  pretrained: false
  num_input_frames: 2
  prediction_type: "parameter"  # 6DOF output

training:
  epochs: 100
  batch_size: 4                 # Reduced for 12GB GPU
  learning_rate: 0.0001
  optimizer: "adam"
  weight_decay: 0.0

  # Frame sampling
  sample_range: 2
  num_samples: 2

  # Validation
  val_frequency: 5              # Validate every N epochs

  # Checkpointing
  save_frequency: 10
  keep_last_n: 3

  # Early stopping
  early_stopping_patience: 20

logging:
  tensorboard: true
  log_frequency: 100            # Log every N batches
```

### 8.4 Environment Configuration (`configs/env/local.yaml`)

```yaml
# Local development environment
paths:
  data_root: "/home/user/data/tus-rec-2024"
  experiments_root: "./experiments"

compute:
  device: "cuda:0"
  num_workers: 4
```

### 8.5 Configuration Loading with Validation

```python
# src/usrec_zoo/config/loader.py
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig


def load_config(
    config_path: Path,
    env_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    Load configuration with inheritance and CLI overrides.

    Args:
        config_path: Primary config file
        env_path: Environment-specific overrides
        cli_overrides: Command-line overrides

    Returns:
        Merged configuration
    """
    # Load base config
    config = OmegaConf.load(config_path)

    # Merge environment config if provided
    if env_path and env_path.exists():
        env_config = OmegaConf.load(env_path)
        config = OmegaConf.merge(config, env_config)

    # Apply CLI overrides
    if cli_overrides:
        cli_config = OmegaConf.create(cli_overrides)
        config = OmegaConf.merge(config, cli_config)

    # Resolve interpolations (${VAR} syntax)
    OmegaConf.resolve(config)

    return config
```

### 8.6 Rationale for Configuration System

| Decision | Rationale |
|----------|-----------|
| **YAML format** | Human-readable, version-control friendly, supports comments |
| **OmegaConf for merging** | Handles interpolation, CLI overrides, hierarchical configs |
| **Separate env configs** | Different paths for local/cluster/docker without code changes |
| **ConfigSchema validation** | Catches typos and type errors before training starts |
| **Algorithm configs separate** | Each algorithm can have different hyperparameters |

---

## 9. Key Design Decisions & Rationales

### 9.1 Algorithms Return Transforms, Framework Converts to DDFs

**Decision**: `predict()` returns `TransformPrediction`; framework handles DDF conversion.

**Rationale**:
- Algorithms naturally predict transforms (their core task)
- DDF conversion requires calibration and is error-prone—centralizing ensures consistency
- Transforms are more interpretable for debugging
- Some future algorithms might predict DDFs directly (separate interface)

**Trade-off**: Framework has more responsibility; algorithms have simpler interface.

### 9.2 Calibration Stored at Construction

**Decision**: Calibration is passed to `__init__()`, stored as `self.calibration`.

**Rationale**:
- Prevents train/predict calibration mismatch (major bug source)
- Cleaner method signatures
- Hash stored in checkpoints for validation

**Trade-off**: Can't easily test same algorithm with different calibrations without new instance.

### 9.3 Hybrid Training (Shared Default + Escape Hatch)

**Decision**: Default `train()` uses `SharedTrainer`; can be overridden.

**Rationale**:
- 90% of algorithms use similar training loops
- Shared trainer ensures consistent logging, checkpointing, metrics
- Escape hatch available for GAN training, meta-learning, etc.
- `requires_custom_training` property documents exceptions

**Trade-off**: Two code paths to maintain; custom training needs auditing for fairness.

### 9.4 Subject-Level Splits with Cross-Validation Option

**Decision**: Primary split is subject-level; 5-fold CV available as secondary metric.

**Rationale**:
- Subject-level prevents data leakage (standard in medical imaging)
- Fixed split enables rapid iteration
- Cross-validation provides statistical validity for publication claims

**Trade-off**: 5 subjects per val/test is small; CV helps characterize variance.

### 9.5 Lightweight ConfigSchema Instead of Pydantic

**Decision**: Custom `ConfigSchema` class for validation, not Pydantic.

**Rationale**:
- Catches typos (unknown keys) and type errors
- No external dependency
- Simple enough to understand immediately
- Can upgrade to Pydantic later if needed

**Trade-off**: Less powerful validation than Pydantic; no auto-documentation.

### 9.6 src/ Layout with Project-Specific Package Name

**Decision**: Use `src/usrec_zoo/` layout.

**Rationale**:
- Prevents import shadowing issues
- Enables proper `pip install -e .`
- `usrec_zoo` is distinctive (not generic like `core`)
- Follows Python packaging best practices

**Trade-off**: One extra directory level; slightly longer imports.

### 9.7 Frozen Dataclasses with Documented Limitations

**Decision**: Use `@dataclass(frozen=True)` but document numpy mutability.

**Rationale**:
- Documents immutability intent
- Prevents attribute reassignment
- Explicit `copy()` method for modifications

**Trade-off**: `frozen=True` doesn't prevent `arr[0] = x`; must rely on convention.

### 9.8 Validation via if/raise, Not assert

**Decision**: Use `if condition: raise ValueError()` instead of `assert`.

**Rationale**:
- Assertions can be disabled with `python -O`
- Explicit raises always execute
- Clearer stack traces

**Trade-off**: Slightly more verbose.

### 9.9 Experimental Directory for Quick Iteration

**Decision**: Add `experimental/` with relaxed interface rules.

**Rationale**:
- Research requires quick experiments
- Full interface boilerplate slows iteration
- Successful experiments graduate to `algorithms/`

**Trade-off**: Two classes of code; must maintain quality boundary.

---

## 10. Migration Path

### Phase 1: Extract Core Infrastructure (Week 1-2)

1. Create `src/usrec_zoo/` directory structure
2. Implement `types.py` and `constants.py`
3. Extract `CalibrationData` with `content_hash()`
4. Implement `transforms_to_ddf()` as single source of truth
5. Add comprehensive tests for DDF conversion

**Checkpoint**: Core layer works independently; DDF conversion tested against baseline.

### Phase 2: Define Algorithm Interface (Week 2-3)

1. Create `algorithms/base.py` with `AlgorithmInterface`
2. Implement `ConfigSchema` validation
3. Create `SharedTrainer` default implementation
4. Implement `TrainResult`, `ValidationResult` types
5. Create `algorithms/registry.py`

**Checkpoint**: Interface defined; can instantiate mock algorithms.

### Phase 3: Refactor Baseline Algorithm (Week 3-4)

1. Move baseline to `algorithms/tusrec_baseline/`
2. Implement `AlgorithmInterface` for baseline
3. Change `predict()` to return `TransformPrediction`
4. Verify results match original code exactly
5. Write comparison tests

**Checkpoint**: Baseline works through new interface; results identical.

### Phase 4: Unify Entry Points (Week 4-5)

1. Create `scripts/train.py` using new interface
2. Create `scripts/evaluate.py` with DDF conversion
3. Implement configuration loading with OmegaConf
4. Create YAML configs for data and baseline
5. Deprecate `TUS-REC2025-Challenge_baseline/train.py`

**Checkpoint**: Can train and evaluate via new scripts.

### Phase 5: Add Benchmark Infrastructure (Week 5-6)

1. Implement `evaluation/benchmark.py`
2. Create `scripts/compare.py`
3. Add visualization utilities
4. Implement 5-fold cross-validation option
5. Document benchmark procedures

**Checkpoint**: Can run systematic algorithm comparisons.

### Phase 6: Validate with Second Algorithm (Week 6-7)

1. Implement a different architecture (e.g., recurrent)
2. Verify it integrates cleanly
3. Run comparison benchmark
4. Identify any interface gaps

**Checkpoint**: Architecture validated with multiple algorithms.

### Phase 7: Cleanup and Documentation (Week 7-8)

1. Remove deprecated code
2. Write user documentation
3. Add integration tests
4. Set up CI/CD

**Checkpoint**: Repository is clean and documented.

---

## 11. Open Questions

### 11.1 Resolved Questions (from v0.1)

| Question | Resolution |
|----------|------------|
| Should algorithms own training loop? | Hybrid: shared default + escape hatch |
| Where does DDF conversion happen? | Framework (evaluation layer) |
| How is calibration passed? | At construction, stored in instance |
| Config validation approach? | Lightweight ConfigSchema |
| Directory layout? | src/ layout with usrec_zoo package |

### 11.2 Remaining Questions

| Question | Options | Current Leaning |
|----------|---------|-----------------|
| **Checkpoint format** | (A) PyTorch .pt (B) SafeTensors | (A) PyTorch |
| **Experiment tracking** | (A) TensorBoard only (B) W&B integration | (A) Start simple |
| **Multi-GPU support** | (A) Out of scope (B) Basic DataParallel | (A) Out of scope |
| **Should experimental/ have any interface?** | (A) None (B) Minimal | (B) Minimal |

### 11.3 Future Considerations

- **Multi-dataset support**: May need `DatasetInterface` abstraction
- **Hyperparameter optimization**: Optuna/Ray Tune integration
- **Model export**: ONNX for deployment
- **Distributed training**: If datasets grow significantly

---

## Appendix: Current Codebase Analysis

### A.1 Files to Extract into Core

| Current Location | Target Location | Notes |
|------------------|-----------------|-------|
| `utils/loader.py` → `Dataset` | `usrec_zoo/data/loader.py` | Generalize frame sampling |
| `utils/transform.py` → coordinate logic | `usrec_zoo/transforms/` | Split into modules |
| `utils/Transf2DDFs.py` | `usrec_zoo/evaluation/ddf.py` | Rewrite with clear pixel ordering |
| `utils/plot_functions.py` → calibration | `usrec_zoo/calibration/` | Add content_hash() |
| `utils/loss.py` | `usrec_zoo/evaluation/metrics.py` | Rename for clarity |

### A.2 Files to Keep Algorithm-Specific

| File | Target | Reason |
|------|--------|--------|
| `utils/network.py` | `algorithms/tusrec_baseline/model.py` | Architecture-specific |
| `utils/Prediction.py` | `algorithms/tusrec_baseline/` | Inference logic |
| `train.py` | `algorithms/tusrec_baseline/` | Training details |

### A.3 Constants to Extract

| Value | Current | Constant Name | Location |
|-------|---------|---------------|----------|
| `480` | Everywhere | `IMAGE_HEIGHT` | `constants.py` |
| `640` | Everywhere | `IMAGE_WIDTH` | `constants.py` |
| `307200` | DDF shapes | `NUM_PIXELS` | `constants.py` |
| `20` | Landmark count | `DEFAULT_LANDMARKS_PER_SCAN` | `constants.py` |

### A.4 Incorrect Documentation to Fix

| Issue | Location | Fix |
|-------|----------|-----|
| "100 landmarks" | Multiple comments | Actual count is 20 |
| Pixel ordering comments | `Transf2DDFs.py` | Document row-major clearly |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2025-01-17 | Initial draft |
| 0.2 | 2025-01-17 | Major revision based on design review: calibration at construction, algorithms return transforms, hybrid training, ConfigSchema, src/ layout, structured return types |

---

*End of Architecture Proposal*
