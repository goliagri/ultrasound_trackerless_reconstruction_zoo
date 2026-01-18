"""
Calibration data for coordinate system transformations.

The ultrasound reconstruction pipeline uses three coordinate systems:
1. Image coordinate (pixels): 480x640 frame space
2. Image coordinate (mm): Scaled from pixels using calibration
3. Tracker tool coordinate: From optical tracker

The calibration matrix relates these coordinate systems.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union
import hashlib

import numpy as np
import torch


@dataclass(frozen=True)
class CalibrationData:
    """
    Immutable container for calibration matrices.

    This class holds the calibration data loaded from the calib_matrix.csv file.
    The matrices define transformations between coordinate systems.

    Coordinate Systems:
        1. Image (pixels): [0, 640) x [0, 480) integer coordinates
        2. Image (mm): Physical coordinates in millimeters
        3. Tool: Optical tracker coordinate system

    Transform Chain:
        pixels -> mm: tform_pixel_to_mm
        mm -> tool: tform_mm_to_tool
        pixels -> tool: tform_pixel_to_tool (= mm_to_tool @ pixel_to_mm)

    All transforms use LEFT multiplication:
        point_in_target = T @ point_in_source

    Attributes:
        tform_pixel_to_mm: 4x4 matrix transforming image pixel coordinates to
                           image millimeter coordinates. Shape [4, 4], dtype float32.
        tform_mm_to_tool: 4x4 matrix transforming image millimeter coordinates
                          to tracker tool coordinates. Shape [4, 4], dtype float32.
        tform_pixel_to_tool: Combined transformation from pixel to tool coordinates.
                             Computed as tform_mm_to_tool @ tform_pixel_to_mm.
                             Shape [4, 4], dtype float32.
        tform_mm_to_pixel: Inverse of tform_pixel_to_mm. Shape [4, 4], dtype float32.
        tform_tool_to_mm: Inverse of tform_mm_to_tool. Shape [4, 4], dtype float32.
        tform_tool_to_pixel: Inverse of tform_pixel_to_tool. Shape [4, 4], dtype float32.

    Note:
        All matrices use homogeneous coordinates (4x4) for 3D transformations.
        The z-coordinate in image space is typically 0 (2D images).
        While this dataclass is frozen, numpy array contents can still be modified
        in-place. Treat arrays as immutable by convention.
    """
    tform_pixel_to_mm: np.ndarray
    tform_mm_to_tool: np.ndarray
    tform_pixel_to_tool: np.ndarray
    tform_mm_to_pixel: np.ndarray
    tform_tool_to_mm: np.ndarray
    tform_tool_to_pixel: np.ndarray

    def __post_init__(self) -> None:
        """Validate matrix shapes and dtypes."""
        matrices = [
            ("tform_pixel_to_mm", self.tform_pixel_to_mm),
            ("tform_mm_to_tool", self.tform_mm_to_tool),
            ("tform_pixel_to_tool", self.tform_pixel_to_tool),
            ("tform_mm_to_pixel", self.tform_mm_to_pixel),
            ("tform_tool_to_mm", self.tform_tool_to_mm),
            ("tform_tool_to_pixel", self.tform_tool_to_pixel),
        ]
        for name, matrix in matrices:
            if matrix.shape != (4, 4):
                raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
            if matrix.dtype != np.float32:
                raise TypeError(f"{name} must have dtype float32, got {matrix.dtype}")

    @classmethod
    def from_csv(cls, filepath: Union[str, Path]) -> "CalibrationData":
        """
        Load calibration data from a CSV file.

        The CSV file format (calib_matrix.csv) contains:
        - Row 0: Header for first matrix
        - Rows 1-4: First 4x4 matrix (pixel to mm transformation)
        - Row 5: Header for second matrix
        - Rows 6-9: Second 4x4 matrix (mm to tool transformation)

        Args:
            filepath: Path to the calib_matrix.csv file.

        Returns:
            CalibrationData instance with loaded matrices.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is invalid.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Calibration file not found: {filepath}")

        tform_calib = np.empty((8, 4), np.float32)

        with open(filepath, "r") as csv_file:
            lines = [line.strip("\n").split(",") for line in csv_file.readlines()]

            if len(lines) < 10:
                raise ValueError(
                    f"Calibration file must have at least 10 rows, got {len(lines)}"
                )

            # First matrix: rows 1-4 (after header row 0)
            tform_calib[0:4, :] = np.array(lines[1:5]).astype(np.float32)
            # Second matrix: rows 6-9 (after header row 5)
            tform_calib[4:8, :] = np.array(lines[6:10]).astype(np.float32)

        tform_pixel_to_mm = tform_calib[0:4, :]
        tform_mm_to_tool = tform_calib[4:8, :]
        tform_pixel_to_tool = (tform_mm_to_tool @ tform_pixel_to_mm).astype(np.float32)

        # Pre-compute inverses
        tform_mm_to_pixel = np.linalg.inv(tform_pixel_to_mm).astype(np.float32)
        tform_tool_to_mm = np.linalg.inv(tform_mm_to_tool).astype(np.float32)
        tform_tool_to_pixel = np.linalg.inv(tform_pixel_to_tool).astype(np.float32)

        return cls(
            tform_pixel_to_mm=tform_pixel_to_mm,
            tform_mm_to_tool=tform_mm_to_tool,
            tform_pixel_to_tool=tform_pixel_to_tool,
            tform_mm_to_pixel=tform_mm_to_pixel,
            tform_tool_to_mm=tform_tool_to_mm,
            tform_tool_to_pixel=tform_tool_to_pixel,
        )

    def content_hash(self) -> str:
        """
        Compute a hash of the calibration data for checkpoint validation.

        Used to verify that loaded models were trained with compatible calibration.

        Returns:
            First 16 characters of SHA256 hash of the concatenated matrix bytes.
        """
        data = np.concatenate([
            self.tform_pixel_to_mm.flatten(),
            self.tform_mm_to_tool.flatten(),
        ])
        return hashlib.sha256(data.tobytes()).hexdigest()[:16]

    def to_torch(self, device: torch.device) -> "CalibrationDataTorch":
        """
        Convert to PyTorch tensors on specified device.

        Args:
            device: PyTorch device for tensor storage (e.g., torch.device("cuda:0")).

        Returns:
            CalibrationDataTorch instance with all matrices as tensors on the device.
        """
        return CalibrationDataTorch(self, device)


class CalibrationDataTorch:
    """
    GPU-compatible calibration data for efficient batch processing.

    This class wraps CalibrationData and provides PyTorch tensors
    on the specified device for use in neural network pipelines.

    Attributes:
        device: The PyTorch device (CPU or CUDA).
        tform_pixel_to_mm: [4, 4] tensor, pixel to mm transform.
        tform_mm_to_tool: [4, 4] tensor, mm to tool transform.
        tform_pixel_to_tool: [4, 4] tensor, pixel to tool transform.
        tform_mm_to_pixel: [4, 4] tensor, mm to pixel transform (inverse).
        tform_tool_to_mm: [4, 4] tensor, tool to mm transform (inverse).
        tform_tool_to_pixel: [4, 4] tensor, tool to pixel transform (inverse).

    Note:
        Unlike CalibrationData, this class is NOT frozen. Tensor attributes can
        be reassigned or mutated in-place. Treat tensors as immutable by convention.
    """

    def __init__(
        self,
        calibration: CalibrationData,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initialize with calibration data on the specified device.

        Args:
            calibration: The CalibrationData instance to convert.
            device: PyTorch device for tensor storage.

        Raises:
            TypeError: If calibration is not a CalibrationData instance.
        """
        if not isinstance(calibration, CalibrationData):
            raise TypeError(
                f"calibration must be CalibrationData, got {type(calibration).__name__}"
            )

        self.device = device
        self._calibration = calibration

        # Convert all matrices to tensors (using pre-computed numpy arrays)
        self.tform_pixel_to_mm = torch.tensor(
            calibration.tform_pixel_to_mm, dtype=torch.float32, device=device
        )
        self.tform_mm_to_tool = torch.tensor(
            calibration.tform_mm_to_tool, dtype=torch.float32, device=device
        )
        self.tform_pixel_to_tool = torch.tensor(
            calibration.tform_pixel_to_tool, dtype=torch.float32, device=device
        )
        self.tform_mm_to_pixel = torch.tensor(
            calibration.tform_mm_to_pixel, dtype=torch.float32, device=device
        )
        self.tform_tool_to_mm = torch.tensor(
            calibration.tform_tool_to_mm, dtype=torch.float32, device=device
        )
        self.tform_tool_to_pixel = torch.tensor(
            calibration.tform_tool_to_pixel, dtype=torch.float32, device=device
        )

    def to(self, device: torch.device) -> "CalibrationDataTorch":
        """
        Create a new CalibrationDataTorch on a different device.

        Args:
            device: Target device.

        Returns:
            New CalibrationDataTorch instance on the specified device.
        """
        return CalibrationDataTorch(self._calibration, device)

    @property
    def numpy_calibration(self) -> CalibrationData:
        """Return the underlying NumPy calibration data."""
        return self._calibration

    def content_hash(self) -> str:
        """Delegate to underlying CalibrationData."""
        return self._calibration.content_hash()
