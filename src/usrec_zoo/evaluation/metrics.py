"""
Evaluation metrics for ultrasound reconstruction.

This module provides metrics for evaluating the quality of
transformation predictions, primarily through landmark-based
error measurements.
"""

from typing import Dict, Optional
import numpy as np

from usrec_zoo.types import DDFOutput


def compute_landmark_error(
    predicted_ddf: DDFOutput,
    ground_truth_ddf: DDFOutput,
    use_global: bool = True,
) -> float:
    """
    Compute mean Euclidean distance error between predicted and GT landmark DDFs.

    Args:
        predicted_ddf: DDFOutput from algorithm prediction.
        ground_truth_ddf: DDFOutput from ground truth transforms.
        use_global: If True, use global DDFs; otherwise use local DDFs.

    Returns:
        Mean Euclidean distance error in millimeters across all landmarks.

    Raises:
        ValueError: If predicted and ground truth have different numbers of landmarks.
    """
    if use_global:
        pred_landmarks = predicted_ddf.global_landmarks
        gt_landmarks = ground_truth_ddf.global_landmarks
    else:
        pred_landmarks = predicted_ddf.local_landmarks
        gt_landmarks = ground_truth_ddf.local_landmarks

    # Validate matching shapes
    if pred_landmarks.shape != gt_landmarks.shape:
        raise ValueError(
            f"Predicted landmarks shape {pred_landmarks.shape} does not match "
            f"ground truth shape {gt_landmarks.shape}"
        )

    # Both have shape [3, L] where L is number of landmarks
    # Compute Euclidean distance for each landmark
    diff = pred_landmarks - gt_landmarks  # [3, L]
    distances = np.linalg.norm(diff, axis=0)  # [L]

    return float(np.mean(distances))


def compute_per_frame_landmark_error(
    predicted_ddf: DDFOutput,
    ground_truth_ddf: DDFOutput,
    landmarks: np.ndarray,
    use_global: bool = True,
) -> Dict[int, float]:
    """
    Compute landmark error grouped by frame index.

    Args:
        predicted_ddf: DDFOutput from algorithm prediction.
        ground_truth_ddf: DDFOutput from ground truth transforms.
        landmarks: Landmark coordinates, shape [L, 3] with (frame_idx, x, y).
                   frame_idx is 0-based, x and y are 1-based.
        use_global: If True, use global DDFs; otherwise use local DDFs.

    Returns:
        Dictionary mapping frame index to mean error for landmarks in that frame.

    Raises:
        ValueError: If landmarks shape doesn't match DDF landmark count.
        ValueError: If landmarks has invalid shape.
    """
    # Validate inputs
    if landmarks.ndim != 2 or landmarks.shape[1] != 3:
        raise ValueError(
            f"landmarks must have shape [L, 3], got {landmarks.shape}"
        )

    if use_global:
        pred_landmarks = predicted_ddf.global_landmarks
        gt_landmarks = ground_truth_ddf.global_landmarks
    else:
        pred_landmarks = predicted_ddf.local_landmarks
        gt_landmarks = ground_truth_ddf.local_landmarks

    # Validate landmark count matches DDF
    if pred_landmarks.shape[1] != landmarks.shape[0]:
        raise ValueError(
            f"Number of landmarks ({landmarks.shape[0]}) does not match "
            f"DDF landmark count ({pred_landmarks.shape[1]})"
        )

    # Compute per-landmark errors
    diff = pred_landmarks - gt_landmarks  # [3, L]
    distances = np.linalg.norm(diff, axis=0)  # [L]

    # Group by frame index (0-based)
    frame_indices = landmarks[:, 0]
    unique_frames = np.unique(frame_indices)

    per_frame_errors: Dict[int, float] = {}
    for frame_idx in unique_frames:
        mask = frame_indices == frame_idx
        per_frame_errors[int(frame_idx)] = float(np.mean(distances[mask]))

    return per_frame_errors


def compute_pixel_error(
    predicted_ddf: DDFOutput,
    ground_truth_ddf: DDFOutput,
    use_global: bool = True,
) -> Dict[str, float]:
    """
    Compute error statistics for all pixel DDFs.

    Args:
        predicted_ddf: DDFOutput from algorithm prediction.
        ground_truth_ddf: DDFOutput from ground truth transforms.
        use_global: If True, use global DDFs; otherwise use local DDFs.

    Returns:
        Dictionary with error statistics:
        - mean: Mean error across all pixels and frames
        - std: Standard deviation
        - max: Maximum error
        - median: Median error

    Raises:
        ValueError: If predicted and ground truth have different shapes.
    """
    if use_global:
        pred_pixels = predicted_ddf.global_pixels
        gt_pixels = ground_truth_ddf.global_pixels
    else:
        pred_pixels = predicted_ddf.local_pixels
        gt_pixels = ground_truth_ddf.local_pixels

    # Validate matching shapes
    if pred_pixels.shape != gt_pixels.shape:
        raise ValueError(
            f"Predicted pixels shape {pred_pixels.shape} does not match "
            f"ground truth shape {gt_pixels.shape}"
        )

    # Both have shape [N-1, 3, NUM_PIXELS]
    diff = pred_pixels - gt_pixels
    # Compute Euclidean distance per pixel per frame
    distances = np.linalg.norm(diff, axis=1)  # [N-1, NUM_PIXELS]

    return {
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "max": float(np.max(distances)),
        "median": float(np.median(distances)),
    }


def ddf_to_landmark_positions(
    ddf: DDFOutput,
    landmarks: np.ndarray,
    calibration_scale: np.ndarray,
    use_global: bool = True,
) -> np.ndarray:
    """
    Convert DDF values to predicted landmark positions in mm.

    This reconstructs the absolute positions of landmarks after applying
    the predicted displacements.

    Args:
        ddf: DDFOutput containing the displacement fields.
        landmarks: Original landmark coordinates, shape [L, 3] with
                   (frame_idx, x, y) where x,y are 1-based pixel coords.
        calibration_scale: Pixel to mm transformation matrix, shape [4, 4].
        use_global: If True, use global DDFs; otherwise use local DDFs.

    Returns:
        Predicted landmark positions in mm, shape [L, 3] with (x_mm, y_mm, z_mm).

    Raises:
        ValueError: If landmarks has invalid shape.
        ValueError: If calibration_scale has invalid shape.
    """
    # Validate inputs
    if landmarks.ndim != 2 or landmarks.shape[1] != 3:
        raise ValueError(
            f"landmarks must have shape [L, 3], got {landmarks.shape}"
        )
    if calibration_scale.shape != (4, 4):
        raise ValueError(
            f"calibration_scale must have shape [4, 4], got {calibration_scale.shape}"
        )

    if use_global:
        landmark_ddf = ddf.global_landmarks
    else:
        landmark_ddf = ddf.local_landmarks

    # Convert original landmark pixel coordinates to mm
    # landmarks[:, 1:3] are (x, y) in 1-based pixel coordinates
    # Calibration matrix expects [x, y, z, 1]^T as input
    num_landmarks = landmarks.shape[0]
    homogeneous_coords = np.zeros((4, num_landmarks), dtype=np.float32)
    homogeneous_coords[0, :] = landmarks[:, 1]  # x (column, 1-based)
    homogeneous_coords[1, :] = landmarks[:, 2]  # y (row, 1-based)
    homogeneous_coords[2, :] = 0  # z (image plane)
    homogeneous_coords[3, :] = 1  # homogeneous coordinate

    # Transform to mm
    ref_positions_mm = calibration_scale @ homogeneous_coords  # [4, L]

    # Add DDF to get predicted positions
    # DDF is [3, L], ref_positions is [4, L]
    predicted_positions = ref_positions_mm[0:3, :] + landmark_ddf  # [3, L]

    return predicted_positions.T  # [L, 3]
