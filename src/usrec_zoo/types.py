"""
Core type definitions for the usrec_zoo package.

These dataclasses define the standard interfaces for data flow between
components in the reconstruction pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch

from usrec_zoo.constants import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_PIXELS


@dataclass(frozen=True)
class ScanData:
    """
    Container for a single ultrasound scan's data.

    This is the canonical input format for algorithms. All algorithms receive
    data in this format.

    WARNING: While this dataclass is frozen (attribute reassignment prevented),
    numpy array contents can still be modified in-place. Treat arrays as
    immutable by convention. Use copy() method if you need to modify data.

    Attributes:
        frames: Ultrasound images, shape [N, H, W] where N is number of frames.
                dtype: uint8, values 0-255.
        transforms: Ground-truth 4x4 transformation matrices from tracker,
                    shape [N, 4, 4]. dtype: float32. May be None for inference.
        subject_id: Subject identifier (e.g., "000", "001").
        scan_name: Scan identifier within subject (e.g., "2024_10_08_12_35_41").
        landmarks: Optional landmark coordinates, shape [L, 3] where L is number
                   of landmarks. Each row is (frame_index, x, y) where frame_index
                   is 0-based and x,y are 1-based pixel coordinates.
    """
    frames: np.ndarray
    transforms: Optional[np.ndarray]
    subject_id: str
    scan_name: str
    landmarks: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate input data shapes and types."""
        if self.frames.ndim != 3:
            raise ValueError(
                f"frames must be 3D [N, H, W], got shape {self.frames.shape}"
            )
        if self.frames.shape[1] != IMAGE_HEIGHT or self.frames.shape[2] != IMAGE_WIDTH:
            raise ValueError(
                f"frames must have shape [N, {IMAGE_HEIGHT}, {IMAGE_WIDTH}], "
                f"got shape {self.frames.shape}"
            )
        if self.frames.shape[0] < 2:
            raise ValueError(
                f"Need at least 2 frames for reconstruction, got {self.frames.shape[0]}"
            )
        if self.transforms is not None:
            if self.transforms.ndim != 3:
                raise ValueError(
                    f"transforms must be 3D [N, 4, 4], got shape {self.transforms.shape}"
                )
            if self.transforms.shape[1:] != (4, 4):
                raise ValueError(
                    f"transforms must have shape [N, 4, 4], got {self.transforms.shape}"
                )
            if self.transforms.shape[0] != self.frames.shape[0]:
                raise ValueError(
                    f"Number of transforms ({self.transforms.shape[0]}) must match "
                    f"number of frames ({self.frames.shape[0]})"
                )
        if self.landmarks is not None:
            if self.landmarks.ndim != 2 or self.landmarks.shape[1] != 3:
                raise ValueError(
                    f"landmarks must have shape [L, 3], got {self.landmarks.shape}"
                )

    @property
    def num_frames(self) -> int:
        """Return the number of frames in this scan."""
        return self.frames.shape[0]

    def copy(self, **changes: Any) -> "ScanData":
        """
        Create a copy with optional field changes and deep-copied arrays.

        Args:
            **changes: Field names and new values to override.

        Returns:
            New ScanData instance with copied arrays.

        Example:
            new_scan = scan.copy(subject_id="modified_001")
        """
        return ScanData(
            frames=changes.get("frames", self.frames.copy()),
            transforms=changes.get(
                "transforms",
                self.transforms.copy() if self.transforms is not None else None,
            ),
            subject_id=changes.get("subject_id", self.subject_id),
            scan_name=changes.get("scan_name", self.scan_name),
            landmarks=changes.get(
                "landmarks",
                self.landmarks.copy() if self.landmarks is not None else None,
            ),
        )


@dataclass(frozen=True)
class TransformPrediction:
    """
    Output from an algorithm's predict() method.

    Algorithms predict transformations between frames. This container holds
    both local transforms (frame N to frame N-1) and global transforms
    (frame N to frame 0).

    This is what algorithms return from predict(). The framework converts
    this to DDFOutput for evaluation.

    WARNING: While this dataclass is frozen, tensor contents can still be
    modified in-place. Treat tensors as immutable by convention.

    Attributes:
        local_transforms: Transformations from each frame to the previous frame,
                          shape [N-1, 4, 4]. Transform[i] maps frame i+1 to frame i.
                          Units are millimeters (in image-mm coordinate system).
                          dtype: torch.float32.
        global_transforms: Transformations from each frame to the first frame,
                           shape [N-1, 4, 4]. Transform[i] maps frame i+1 to frame 0.
                           Units are millimeters (in image-mm coordinate system).
                           dtype: torch.float32.
        metadata: Optional dictionary for algorithm-specific outputs (e.g.,
                  confidence scores, intermediate representations).
    """
    local_transforms: torch.Tensor
    global_transforms: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate transform shapes and dtypes."""
        if self.local_transforms.ndim != 3:
            raise ValueError(
                f"local_transforms must be 3D [N-1, 4, 4], "
                f"got shape {self.local_transforms.shape}"
            )
        if self.local_transforms.shape[1:] != (4, 4):
            raise ValueError(
                f"local_transforms must have shape [N-1, 4, 4], "
                f"got {self.local_transforms.shape}"
            )
        if self.global_transforms.ndim != 3:
            raise ValueError(
                f"global_transforms must be 3D [N-1, 4, 4], "
                f"got shape {self.global_transforms.shape}"
            )
        if self.global_transforms.shape[1:] != (4, 4):
            raise ValueError(
                f"global_transforms must have shape [N-1, 4, 4], "
                f"got {self.global_transforms.shape}"
            )
        if self.local_transforms.shape[0] != self.global_transforms.shape[0]:
            raise ValueError(
                f"local_transforms ({self.local_transforms.shape[0]}) and "
                f"global_transforms ({self.global_transforms.shape[0]}) must have "
                f"same number of transforms"
            )
        if self.local_transforms.dtype != torch.float32:
            raise TypeError(
                f"local_transforms must be float32, got {self.local_transforms.dtype}"
            )
        if self.global_transforms.dtype != torch.float32:
            raise TypeError(
                f"global_transforms must be float32, got {self.global_transforms.dtype}"
            )

    @property
    def num_frame_pairs(self) -> int:
        """Return the number of frame pairs (N-1 for N frames)."""
        return self.local_transforms.shape[0]


@dataclass(frozen=True)
class DDFOutput:
    """
    Displacement Dense Fields - the final output format for evaluation.

    This is generated by the FRAMEWORK from TransformPrediction, not by algorithms.
    The evaluation system uses this format for computing metrics.

    All displacements are in millimeters (mm). dtype: float32.

    Attributes:
        global_pixels: GP: Global DDF for all pixels, shape [N-1, 3, 307200].
                       Displacement from each frame to frame 0.
        global_landmarks: GL: Global DDF for landmarks, shape [3, L] where L is
                          number of landmarks. Each column is (dx, dy, dz).
        local_pixels: LP: Local DDF for all pixels, shape [N-1, 3, 307200].
                      Displacement from each frame to the previous frame.
        local_landmarks: LL: Local DDF for landmarks, shape [3, L].
        image_shape: Shape of image (height, width) for reshaping operations.

    Pixel Ordering Convention:
        Pixels are stored in row-major order: pixel[y, x] maps to index y*640 + x
        Index 0 corresponds to pixel (x=1, y=1) in 1-based coordinates.
        Use the helper methods global_pixels_2d() and get_pixel_displacement()
        for safe access.
    """
    global_pixels: np.ndarray
    global_landmarks: np.ndarray
    local_pixels: np.ndarray
    local_landmarks: np.ndarray
    image_shape: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)

    def __post_init__(self) -> None:
        """Validate DDF shapes and dtypes."""
        expected_pixels = self.image_shape[0] * self.image_shape[1]

        # Validate global_pixels
        if self.global_pixels.ndim != 3:
            raise ValueError(
                f"global_pixels must be 3D [N-1, 3, num_pixels], "
                f"got shape {self.global_pixels.shape}"
            )
        if self.global_pixels.shape[1] != 3:
            raise ValueError(
                f"global_pixels second dim must be 3 (x,y,z), "
                f"got {self.global_pixels.shape[1]}"
            )
        if self.global_pixels.shape[2] != expected_pixels:
            raise ValueError(
                f"global_pixels must have {expected_pixels} pixels, "
                f"got {self.global_pixels.shape[2]}"
            )

        # Validate local_pixels matches global_pixels
        if self.local_pixels.shape != self.global_pixels.shape:
            raise ValueError(
                f"local_pixels shape {self.local_pixels.shape} must match "
                f"global_pixels shape {self.global_pixels.shape}"
            )

        # Validate landmark DDFs
        if self.global_landmarks.ndim != 2 or self.global_landmarks.shape[0] != 3:
            raise ValueError(
                f"global_landmarks must have shape [3, L], "
                f"got {self.global_landmarks.shape}"
            )
        if self.local_landmarks.shape != self.global_landmarks.shape:
            raise ValueError(
                f"local_landmarks shape {self.local_landmarks.shape} must match "
                f"global_landmarks shape {self.global_landmarks.shape}"
            )

        # Validate dtypes
        if self.global_pixels.dtype != np.float32:
            raise TypeError(
                f"global_pixels must be float32, got {self.global_pixels.dtype}"
            )
        if self.local_pixels.dtype != np.float32:
            raise TypeError(
                f"local_pixels must be float32, got {self.local_pixels.dtype}"
            )
        if self.global_landmarks.dtype != np.float32:
            raise TypeError(
                f"global_landmarks must be float32, got {self.global_landmarks.dtype}"
            )
        if self.local_landmarks.dtype != np.float32:
            raise TypeError(
                f"local_landmarks must be float32, got {self.local_landmarks.dtype}"
            )

    def validate_values(self) -> "DDFOutput":
        """
        Check for NaN/Inf values. Call explicitly when needed (expensive for large arrays).

        Returns:
            self for method chaining.

        Raises:
            ValueError: If any array contains NaN or Inf values.
        """
        if not np.isfinite(self.global_pixels).all():
            raise ValueError("global_pixels contains NaN or Inf values")
        if not np.isfinite(self.global_landmarks).all():
            raise ValueError("global_landmarks contains NaN or Inf values")
        if not np.isfinite(self.local_pixels).all():
            raise ValueError("local_pixels contains NaN or Inf values")
        if not np.isfinite(self.local_landmarks).all():
            raise ValueError("local_landmarks contains NaN or Inf values")
        return self

    def global_pixels_2d(self) -> np.ndarray:
        """
        Reshape global_pixels to [N-1, 3, H, W] for spatial operations.

        Returns:
            Array with shape [N-1, 3, image_height, image_width].
        """
        return self.global_pixels.reshape(-1, 3, *self.image_shape)

    def local_pixels_2d(self) -> np.ndarray:
        """
        Reshape local_pixels to [N-1, 3, H, W] for spatial operations.

        Returns:
            Array with shape [N-1, 3, image_height, image_width].
        """
        return self.local_pixels.reshape(-1, 3, *self.image_shape)

    def get_pixel_displacement(
        self, frame_idx: int, x: int, y: int, local: bool = False
    ) -> np.ndarray:
        """
        Get displacement for a specific pixel (safe accessor).

        Args:
            frame_idx: 0-indexed frame number (0 to N-2).
            x: 1-indexed x coordinate (1 to 640).
            y: 1-indexed y coordinate (1 to 480).
            local: If True, return local displacement; else global.

        Returns:
            [3,] array of (dx, dy, dz) displacement in mm.

        Raises:
            IndexError: If coordinates are out of bounds.
        """
        if not (1 <= x <= self.image_shape[1]):
            raise IndexError(f"x must be in [1, {self.image_shape[1]}], got {x}")
        if not (1 <= y <= self.image_shape[0]):
            raise IndexError(f"y must be in [1, {self.image_shape[0]}], got {y}")
        if not (0 <= frame_idx < self.global_pixels.shape[0]):
            raise IndexError(
                f"frame_idx must be in [0, {self.global_pixels.shape[0] - 1}], "
                f"got {frame_idx}"
            )

        flat_idx = (y - 1) * self.image_shape[1] + (x - 1)
        if local:
            return self.local_pixels[frame_idx, :, flat_idx]
        return self.global_pixels[frame_idx, :, flat_idx]

    @property
    def num_frame_pairs(self) -> int:
        """Return the number of frame pairs (N-1 for N frames)."""
        return self.global_pixels.shape[0]

    @property
    def num_landmarks(self) -> int:
        """Return the number of landmarks."""
        return self.global_landmarks.shape[1]


@dataclass
class ValidationResult:
    """
    Results from validating an algorithm on a validation set.

    Attributes:
        mean_landmark_error: Mean Euclidean distance error across all landmarks,
                             in millimeters.
        per_scan_errors: Dictionary mapping scan identifiers to their mean
                         landmark errors.
        ddf_outputs: Optional dictionary mapping scan identifiers to DDFOutput
                     objects for further analysis.
        metadata: Additional algorithm-specific validation metrics.
    """
    mean_landmark_error: float
    per_scan_errors: Dict[str, float] = field(default_factory=dict)
    ddf_outputs: Optional[Dict[str, DDFOutput]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    """
    Results from training an algorithm.

    Attributes:
        final_epoch: The epoch number at which training ended.
        final_train_loss: Training loss at the final epoch.
        final_val_loss: Validation loss at the final epoch (if validation was run).
        best_val_epoch: Epoch with best validation performance.
        best_val_metric: Best validation metric achieved.
        checkpoint_path: Path to the saved best checkpoint.
        training_history: List of per-epoch metrics for plotting training curves.
        training_time_seconds: Total wall-clock training time in seconds.
    """
    final_epoch: int
    final_train_loss: float
    final_val_loss: Optional[float] = None
    best_val_epoch: Optional[int] = None
    best_val_metric: Optional[float] = None
    checkpoint_path: Optional[str] = None
    training_history: List[Dict[str, float]] = field(default_factory=list)
    training_time_seconds: Optional[float] = None

    def summary(self) -> str:
        """
        Generate a human-readable training summary.

        Returns:
            Multi-line string summarizing training results.
        """
        lines = [f"Training completed: {self.final_epoch} epochs"]
        if self.training_time_seconds is not None:
            lines[0] += f" in {self.training_time_seconds:.1f}s"
        lines.append(f"Final train loss: {self.final_train_loss:.4f}")
        if self.final_val_loss is not None:
            lines.append(f"Final val loss: {self.final_val_loss:.4f}")
        if self.best_val_epoch is not None:
            lines.append(f"Best epoch: {self.best_val_epoch}")
        if self.best_val_metric is not None:
            lines.append(f"Best val metric: {self.best_val_metric:.4f}")
        if self.checkpoint_path is not None:
            lines.append(f"Best checkpoint: {self.checkpoint_path}")
        return "\n".join(lines)


@dataclass
class EvaluationResult:
    """
    Complete evaluation results for a test/validation set.

    Attributes:
        global_landmark_error: Mean error using global transforms.
        local_landmark_error: Mean error using local transforms.
        per_subject_results: Dictionary of per-subject metrics.
        metadata: Additional evaluation details.
    """
    global_landmark_error: float
    local_landmark_error: float
    per_subject_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
