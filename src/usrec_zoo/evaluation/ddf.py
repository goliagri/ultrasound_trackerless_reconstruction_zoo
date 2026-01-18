"""
Displacement vector field (DDF) conversion for evaluation.

DDFs represent the displacement of pixels/landmarks from their position
in one frame to their position in another frame, measured in millimeters.

This module provides the single source of truth for DDF computation,
ensuring consistency across all algorithms and evaluation code.
"""

import torch
import numpy as np

from usrec_zoo.types import TransformPrediction, DDFOutput
from usrec_zoo.calibration.calibration import CalibrationDataTorch
from usrec_zoo.constants import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_PIXELS


def reference_image_points(
    height: int = IMAGE_HEIGHT,
    width: int = IMAGE_WIDTH,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate reference image points in homogeneous coordinates.

    Creates a grid of all pixel coordinates for the image, suitable for
    transformation by 4x4 matrices. Points are 1-indexed to match the
    calibration convention.

    This function replicates the exact behavior of the original TUS-REC
    implementation's reference_image_points function:
        torch.flip(torch.cartesian_prod(
            torch.linspace(1, height, height),
            torch.linspace(1, width, width)
        ).t(), [0])

    Args:
        height: Image height in pixels (default: 480).
        width: Image width in pixels (default: 640).
        device: PyTorch device for the output tensor.

    Returns:
        Tensor of shape [4, height*width] containing homogeneous coordinates.
        Each column is [x, y, 0, 1] where x is column (1 to width) and
        y is row (1 to height).

    Note on pixel ordering:
        Pixels are ordered with x (column) varying fastest, y (row) varying slowest.
        For a 480x640 image:
        - Index 0: pixel at (x=1, y=1) i.e., top-left
        - Index 1: pixel at (x=2, y=1)
        - Index 639: pixel at (x=640, y=1) i.e., top-right of first row
        - Index 640: pixel at (x=1, y=2) i.e., first column, second row

        The linear index for pixel at (row y, col x) with 1-based indexing is:
            index = (y-1) * width + (x-1)

    Note on coordinate convention:
        The returned coordinates use the convention where:
        - First coordinate (row 0 of output) is x (column index, 1-based)
        - Second coordinate (row 1 of output) is y (row index, 1-based)
        - Third coordinate is z = 0 (2D images)
        - Fourth coordinate is 1 (homogeneous)

        This matches the original TUS-REC implementation exactly.
    """
    # Replicate original: torch.flip(torch.cartesian_prod(...).t(), [0])
    # cartesian_prod(y_range, x_range) produces pairs with x varying fastest
    y_coords = torch.linspace(1, height, height, dtype=torch.float32, device=device)
    x_coords = torch.linspace(1, width, width, dtype=torch.float32, device=device)

    # cartesian_prod: for each y, iterate all x values
    # Result shape: [height*width, 2] with columns [y, x]
    pairs = torch.cartesian_prod(y_coords, x_coords)

    # Transpose to [2, height*width] with rows [y_values, x_values]
    pairs_t = pairs.t()

    # Flip rows: [x_values, y_values]
    pairs_flipped = torch.flip(pairs_t, [0])

    # Add z=0 and homogeneous coordinate
    z_row = torch.zeros(1, pairs_flipped.shape[1], dtype=torch.float32, device=device)
    ones_row = torch.ones(1, pairs_flipped.shape[1], dtype=torch.float32, device=device)

    image_points = torch.cat([pairs_flipped, z_row, ones_row], dim=0)

    return image_points


def transforms_to_ddf(
    prediction: TransformPrediction,
    landmarks: np.ndarray,
    calibration: CalibrationDataTorch,
) -> DDFOutput:
    """
    Convert transformation predictions to displacement vector fields.

    This is the single source of truth for DDF computation. All algorithms
    should use this function to ensure consistent evaluation.

    Args:
        prediction: TransformPrediction containing local and global transforms.
                    Transforms should be in image-mm coordinate system.
        landmarks: Landmark coordinates, shape [L, 3] where each row is
                   (frame_index, x, y). frame_index is 0-based, x and y are
                   1-based pixel coordinates.
        calibration: CalibrationDataTorch instance for coordinate conversion.

    Returns:
        DDFOutput containing global and local DDFs for pixels and landmarks.

    Raises:
        ValueError: If landmarks has invalid shape.
        ValueError: If landmark frame indices are out of bounds.

    Note:
        The DDF represents displacement in millimeters from a point's position
        in frame N to its corresponding position in frame 0 (global) or
        frame N-1 (local).
    """
    # Validate inputs
    if landmarks.ndim != 2 or landmarks.shape[1] != 3:
        raise ValueError(
            f"landmarks must have shape [L, 3], got {landmarks.shape}"
        )

    device = calibration.device
    num_transforms = prediction.num_frame_pairs

    # Get reference image points [4, NUM_PIXELS]
    image_points = reference_image_points(
        IMAGE_HEIGHT, IMAGE_WIDTH, device=device
    )

    # Get calibration scale matrix (pixel to mm)
    tform_scale = calibration.tform_pixel_to_mm

    # Compute reference points in mm: [4, NUM_PIXELS]
    ref_points_mm = torch.matmul(tform_scale, image_points)

    # Ensure transforms are on the correct device
    global_transforms = prediction.global_transforms.to(device)
    local_transforms = prediction.local_transforms.to(device)

    # Compute global DDFs
    # For each transform, compute: T_global @ scale @ image_points - scale @ image_points
    # Shape: [N-1, 4, NUM_PIXELS]
    global_allpts = torch.matmul(
        global_transforms,
        ref_points_mm.unsqueeze(0).expand(num_transforms, -1, -1)
    )

    # DDF = transformed position - reference position
    # Shape: [N-1, 3, NUM_PIXELS] (drop homogeneous coordinate)
    global_allpts_ddf = global_allpts[:, 0:3, :] - ref_points_mm[0:3, :].unsqueeze(0)

    # Compute local DDFs
    local_allpts = torch.matmul(
        local_transforms,
        ref_points_mm.unsqueeze(0).expand(num_transforms, -1, -1)
    )
    local_allpts_ddf = local_allpts[:, 0:3, :] - ref_points_mm[0:3, :].unsqueeze(0)

    # Extract landmark DDFs
    # Reshape to [N-1, 3, H, W] for indexing
    global_ddf_reshaped = global_allpts_ddf.reshape(num_transforms, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    local_ddf_reshaped = local_allpts_ddf.reshape(num_transforms, 3, IMAGE_HEIGHT, IMAGE_WIDTH)

    # Convert landmarks to torch tensor
    landmarks_tensor = torch.tensor(landmarks, dtype=torch.long, device=device)

    # Extract frame indices and coordinates from landmarks
    # Landmarks format: [frame_index, x, y] where:
    # - frame_index is 0-based (0 to N-1)
    # - x and y are 1-based pixel coordinates (calibration convention)
    #
    # For DDF indexing:
    # - DDF arrays have shape [N-1, ...] where index 0 corresponds to frame 1
    # - Frame 0 is the reference and has no DDF (displacement is zero by definition)
    # - So landmark on frame i indexes into DDF at position i-1
    #
    # IMPORTANT: Landmarks on frame 0 have zero displacement by definition
    # since frame 0 is the reference frame.
    frame_indices_raw = landmarks_tensor[:, 0]  # 0-based frame indices
    x_coords = landmarks_tensor[:, 1] - 1  # Convert to 0-based pixel column
    y_coords = landmarks_tensor[:, 2] - 1  # Convert to 0-based pixel row

    # Validate frame indices
    max_frame = num_transforms  # N-1 transforms means N frames (0 to N-1)
    if (frame_indices_raw < 0).any() or (frame_indices_raw > max_frame).any():
        invalid_frames = frame_indices_raw[(frame_indices_raw < 0) | (frame_indices_raw > max_frame)]
        raise ValueError(
            f"Landmark frame indices must be in [0, {max_frame}], "
            f"got invalid indices: {invalid_frames.cpu().numpy()}"
        )

    # Validate coordinate bounds
    if (x_coords < 0).any() or (x_coords >= IMAGE_WIDTH).any():
        raise ValueError(
            f"Landmark x coordinates must be in [1, {IMAGE_WIDTH}] (1-based)"
        )
    if (y_coords < 0).any() or (y_coords >= IMAGE_HEIGHT).any():
        raise ValueError(
            f"Landmark y coordinates must be in [1, {IMAGE_HEIGHT}] (1-based)"
        )

    # Handle landmarks: frame 0 landmarks have zero displacement
    # For frames 1 to N-1, DDF index is frame_index - 1
    num_landmarks = landmarks.shape[0]
    global_landmark_ddf = torch.zeros((3, num_landmarks), dtype=torch.float32, device=device)
    local_landmark_ddf = torch.zeros((3, num_landmarks), dtype=torch.float32, device=device)

    # Mask for landmarks NOT on frame 0 (these have non-zero DDF)
    non_ref_mask = frame_indices_raw > 0

    if non_ref_mask.any():
        # DDF index = frame_index - 1 (since frame 0 is reference)
        ddf_indices = frame_indices_raw[non_ref_mask] - 1
        x_non_ref = x_coords[non_ref_mask]
        y_non_ref = y_coords[non_ref_mask]

        # Extract DDFs for non-reference frame landmarks
        # Index: [ddf_index, :, y, x] for each landmark
        # Reshape gives [N-1, 3, H, W] so indexing is [ddf_idx, channel, row, col]
        global_landmark_ddf[:, non_ref_mask] = global_ddf_reshaped[
            ddf_indices, :, y_non_ref, x_non_ref
        ].T
        local_landmark_ddf[:, non_ref_mask] = local_ddf_reshaped[
            ddf_indices, :, y_non_ref, x_non_ref
        ].T

    # Convert to numpy
    return DDFOutput(
        global_pixels=global_allpts_ddf.cpu().numpy(),
        global_landmarks=global_landmark_ddf.cpu().numpy(),
        local_pixels=local_allpts_ddf.cpu().numpy(),
        local_landmarks=local_landmark_ddf.cpu().numpy(),
    )


def compute_ddf_from_tracker_transforms(
    tracker_transforms: torch.Tensor,
    calibration: CalibrationDataTorch,
    landmarks: np.ndarray,
    pairs_global: torch.Tensor,
    pairs_local: torch.Tensor,
) -> DDFOutput:
    """
    Compute DDFs directly from tracker (ground truth) transforms.

    This function is useful for computing ground truth DDFs from the
    tracker data, or for evaluating methods that directly predict
    tracker-space transforms.

    Args:
        tracker_transforms: Ground truth tracker transforms, shape [N, 4, 4].
                            Transform[i] is T_{tool_i → world}.
        calibration: CalibrationDataTorch instance.
        landmarks: Landmark coordinates, shape [L, 3].
        pairs_global: Frame pairs for global transforms, shape [M, 2].
                      Each row [i, j] indicates transform from frame j to frame i.
        pairs_local: Frame pairs for local transforms, shape [M, 2].

    Returns:
        DDFOutput with computed DDFs.

    Raises:
        ValueError: If tracker_transforms has invalid shape.
        ValueError: If pairs have mismatched lengths.

    Note:
        This function converts tracker transforms to image-mm transforms
        internally using the calibration matrices.
    """
    # Validate inputs
    if tracker_transforms.ndim != 3 or tracker_transforms.shape[1:] != (4, 4):
        raise ValueError(
            f"tracker_transforms must have shape [N, 4, 4], "
            f"got {tracker_transforms.shape}"
        )
    if pairs_global.shape[0] != pairs_local.shape[0]:
        raise ValueError(
            f"pairs_global ({pairs_global.shape[0]}) and pairs_local "
            f"({pairs_local.shape[0]}) must have the same number of pairs"
        )

    device = calibration.device
    num_frames = tracker_transforms.shape[0]

    # Ensure transforms are on device
    tracker_transforms = tracker_transforms.to(device)

    # Pre-compute inverses
    tracker_transforms_inv = torch.linalg.inv(tracker_transforms)

    # Compute image-mm transforms
    # T_{img1_mm → img0_mm} = T_{tool → img_mm} @ T_{tool0 → world}^{-1} @ T_{tool1 → world} @ T_{img_mm → tool}
    tform_tool_to_img_mm = calibration.tform_tool_to_mm
    tform_img_mm_to_tool = calibration.tform_mm_to_tool

    # Build transforms from tracker data
    def compute_transform(pairs: torch.Tensor) -> torch.Tensor:
        """Compute transforms for given frame pairs."""
        num_pairs = pairs.shape[0]
        transforms = []

        for i in range(num_pairs):
            idx0, idx1 = pairs[i, 0].item(), pairs[i, 1].item()
            # T_{tool1 → tool0} = T_{world → tool0} @ T_{tool1 → world}
            t_tool1_to_tool0 = torch.matmul(
                tracker_transforms_inv[idx0],
                tracker_transforms[idx1]
            )
            # T_{img1_mm → img0_mm}
            t_img = torch.matmul(
                tform_tool_to_img_mm,
                torch.matmul(t_tool1_to_tool0, tform_img_mm_to_tool)
            )
            transforms.append(t_img)

        return torch.stack(transforms, dim=0)

    # Compute global and local transforms
    global_transforms = compute_transform(pairs_global)
    local_transforms = compute_transform(pairs_local)

    # Create prediction object
    prediction = TransformPrediction(
        local_transforms=local_transforms,
        global_transforms=global_transforms,
    )

    return transforms_to_ddf(prediction, landmarks, calibration)


def data_pairs_global(num_frames: int) -> torch.Tensor:
    """
    Generate frame pairs for global transforms (all frames to frame 0).

    Args:
        num_frames: Total number of frames in the scan.

    Returns:
        Tensor of shape [num_frames-1, 2] where each row is [0, i] for i in [1, num_frames-1].
        This indicates transform from frame i to frame 0.
    """
    # Note: We skip frame 0 since it's the reference
    return torch.tensor([[0, i] for i in range(1, num_frames)], dtype=torch.long)


def data_pairs_local(num_frames: int) -> torch.Tensor:
    """
    Generate frame pairs for local transforms (each frame to previous).

    Args:
        num_frames: Total number of frames in the scan.

    Returns:
        Tensor of shape [num_frames-1, 2] where each row is [i, i+1].
        This indicates transform from frame i+1 to frame i.
    """
    return torch.tensor([[i, i + 1] for i in range(num_frames - 1)], dtype=torch.long)
