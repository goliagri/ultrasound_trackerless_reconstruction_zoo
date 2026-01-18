"""
Rigid transformation utilities for 6DOF parameter and 4x4 matrix conversions.

This module provides functions to convert between:
- 6DOF parameters (rx, ry, rz, tx, ty, tz) where r is Euler angles and t is translation
- 4x4 homogeneous transformation matrices

The Euler angle convention is ZYX (rotate around Z first, then Y, then X).
"""

import torch
import pytorch3d.transforms

from usrec_zoo.constants import EULER_CONVENTION


def params_to_matrix(params: torch.Tensor) -> torch.Tensor:
    """
    Convert 6DOF parameters to 4x4 transformation matrices.

    Args:
        params: 6DOF parameters with shape [..., 6] where the last dimension
                contains (rx, ry, rz, tx, ty, tz). Euler angles in radians.
                Supports any number of batch dimensions.

    Returns:
        4x4 transformation matrices with shape [..., 4, 4].

    Raises:
        ValueError: If params does not have 6 elements in the last dimension.

    Example:
        >>> params = torch.zeros(10, 6)  # 10 identity transforms
        >>> matrices = params_to_matrix(params)
        >>> matrices.shape
        torch.Size([10, 4, 4])
    """
    if params.shape[-1] != 6:
        raise ValueError(
            f"params must have 6 elements in last dimension, got {params.shape[-1]}"
        )

    # Extract rotation (Euler angles) and translation
    euler_angles = params[..., 0:3]  # rx, ry, rz
    translation = params[..., 3:6]  # tx, ty, tz

    # Convert Euler angles to rotation matrix using ZYX convention
    rotation_matrix = pytorch3d.transforms.euler_angles_to_matrix(
        euler_angles, EULER_CONVENTION
    )

    # Build 4x4 transformation matrix
    # Shape: [..., 3, 4] by concatenating rotation and translation
    transform_3x4 = torch.cat(
        [rotation_matrix, translation[..., None]],
        dim=-1
    )

    # Create the last row [0, 0, 0, 1]
    batch_shape = params.shape[:-1]
    last_row = torch.zeros(
        (*batch_shape, 1, 4),
        dtype=params.dtype,
        device=params.device
    )
    last_row[..., 0, 3] = 1.0

    # Concatenate to get 4x4 matrix
    transform_4x4 = torch.cat([transform_3x4, last_row], dim=-2)

    return transform_4x4


def matrix_to_params(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert 4x4 transformation matrices to 6DOF parameters.

    Note: This function assumes the rotation matrix is orthogonal (valid rotation).
    Non-orthogonal matrices will produce undefined results.

    Args:
        matrix: 4x4 transformation matrices with shape [..., 4, 4].
                Must be valid rigid transformations (orthogonal rotation part).

    Returns:
        6DOF parameters with shape [..., 6] containing (rx, ry, rz, tx, ty, tz).
        Euler angles are in radians, using ZYX convention.

    Raises:
        ValueError: If matrix does not have shape [..., 4, 4].

    Example:
        >>> matrix = torch.eye(4).unsqueeze(0)  # [1, 4, 4] identity
        >>> params = matrix_to_params(matrix)
        >>> params.shape
        torch.Size([1, 6])
    """
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(
            f"matrix must have shape [..., 4, 4], got shape ending with {matrix.shape[-2:]}"
        )

    # Extract rotation matrix (top-left 3x3)
    rotation_matrix = matrix[..., 0:3, 0:3]

    # Extract translation (top-right 3x1)
    translation = matrix[..., 0:3, 3]

    # Convert rotation matrix to Euler angles
    euler_angles = pytorch3d.transforms.matrix_to_euler_angles(
        rotation_matrix, EULER_CONVENTION
    )

    # Concatenate to get 6DOF parameters
    params = torch.cat([euler_angles, translation], dim=-1)

    return params


def compose_transforms(
    transform_a_to_b: torch.Tensor,
    transform_b_to_c: torch.Tensor,
) -> torch.Tensor:
    """
    Compose two transformations to get transform from A to C.

    Given:
    - T_{B←A}: transforms points from A to B
    - T_{C←B}: transforms points from B to C

    Returns:
    - T_{C←A} = T_{C←B} @ T_{B←A}: transforms points from A to C

    Args:
        transform_a_to_b: Transformation matrices from A to B, shape [..., 4, 4].
        transform_b_to_c: Transformation matrices from B to C, shape [..., 4, 4].
                          Must be broadcastable with transform_a_to_b.

    Returns:
        Composed transformation matrices from A to C, shape [..., 4, 4].

    Raises:
        ValueError: If transforms do not have shape [..., 4, 4].

    Note:
        This uses the convention that transformations are applied by
        left-multiplication: p_C = T_{C←A} @ p_A
    """
    if transform_a_to_b.shape[-2:] != (4, 4):
        raise ValueError(
            f"transform_a_to_b must have shape [..., 4, 4], "
            f"got shape ending with {transform_a_to_b.shape[-2:]}"
        )
    if transform_b_to_c.shape[-2:] != (4, 4):
        raise ValueError(
            f"transform_b_to_c must have shape [..., 4, 4], "
            f"got shape ending with {transform_b_to_c.shape[-2:]}"
        )
    return torch.matmul(transform_b_to_c, transform_a_to_b)


def invert_transform(transform: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a rigid transformation.

    For a rigid transformation T, this computes T^{-1} such that
    T @ T^{-1} = T^{-1} @ T = I.

    Args:
        transform: Transformation matrices with shape [..., 4, 4].

    Returns:
        Inverse transformation matrices with shape [..., 4, 4].

    Raises:
        ValueError: If transform does not have shape [..., 4, 4].
    """
    if transform.shape[-2:] != (4, 4):
        raise ValueError(
            f"transform must have shape [..., 4, 4], "
            f"got shape ending with {transform.shape[-2:]}"
        )
    return torch.linalg.inv(transform)


def transform_points(
    points: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    """
    Apply transformation to points.

    Args:
        points: Points in homogeneous coordinates, shape [..., 4, N] where
                N is the number of points. Each column is (x, y, z, 1).
        transform: Transformation matrices, shape [..., 4, 4].
                   Must be broadcastable with points.

    Returns:
        Transformed points, shape [..., 4, N].

    Raises:
        ValueError: If points do not have 4 in the second-to-last dimension,
                   or if transform does not have shape [..., 4, 4].
    """
    if points.shape[-2] != 4:
        raise ValueError(
            f"points must have shape [..., 4, N], "
            f"got {points.shape[-2]} in second-to-last dimension"
        )
    if transform.shape[-2:] != (4, 4):
        raise ValueError(
            f"transform must have shape [..., 4, 4], "
            f"got shape ending with {transform.shape[-2:]}"
        )
    return torch.matmul(transform, points)
