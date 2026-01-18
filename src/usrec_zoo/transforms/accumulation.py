"""
Transform accumulation for computing global transforms from local transforms.

This module provides utilities for accumulating sequential local transformations
(frame N to frame N-1) into global transformations (frame N to frame 0).

Notation Convention:
    T_{B←A} means "transformation that maps points from coordinate system A to B"
    Applied as: point_in_B = T_{B←A} @ point_in_A (left multiplication)

Indexing Convention:
    Given N frames (indexed 0 to N-1), we have N-1 local transforms.
    - local_transforms[0] = T_{0←1}: maps frame 1 to frame 0
    - local_transforms[i] = T_{i←i+1}: maps frame i+1 to frame i

    After accumulation:
    - global_transforms[i] = T_{0←i+1}: maps frame i+1 to frame 0
"""

import torch

from usrec_zoo.transforms.rigid import compose_transforms


class TransformAccumulator:
    """
    Accumulates local transformations into global transformations.

    Given N frames (indexed 0 to N-1), we have N-1 local transforms where:
    - local_transforms[i] maps frame i+1 to frame i (i.e., T_{i←i+1})

    The accumulation computes global transforms where:
    - global_transforms[i] maps frame i+1 to frame 0 (i.e., T_{0←i+1})

    The accumulation follows (using T_{B←A} notation):
        global[0] = local[0]                           # T_{0←1}
        global[1] = global[0] @ local[1]               # T_{0←1} @ T_{1←2} = T_{0←2}
        global[2] = global[1] @ local[2]               # T_{0←2} @ T_{2←3} = T_{0←3}
        ...
        global[i] = global[i-1] @ local[i]             # T_{0←i} @ T_{i←i+1} = T_{0←i+1}

    Attributes:
        None (this is a stateless utility class with static methods).
    """

    @staticmethod
    def accumulate_local_to_global(
        local_transforms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert local transforms to global transforms.

        Args:
            local_transforms: Local transformation matrices, shape [N-1, 4, 4].
                              local_transforms[i] = T_{i←i+1}: maps frame i+1 to frame i.
                              These are in image-mm coordinate system.

        Returns:
            Global transformation matrices, shape [N-1, 4, 4].
            global_transforms[i] = T_{0←i+1}: maps frame i+1 to frame 0.

        Raises:
            ValueError: If local_transforms has wrong shape.

        Example:
            >>> local = torch.eye(4).unsqueeze(0).repeat(5, 1, 1)  # 5 identity transforms
            >>> global_transforms = TransformAccumulator.accumulate_local_to_global(local)
            >>> global_transforms.shape
            torch.Size([5, 4, 4])
        """
        if local_transforms.ndim != 3 or local_transforms.shape[1:] != (4, 4):
            raise ValueError(
                f"local_transforms must have shape [N-1, 4, 4], "
                f"got {local_transforms.shape}"
            )

        num_transforms = local_transforms.shape[0]
        if num_transforms == 0:
            return local_transforms.clone()

        # Initialize output tensor
        global_transforms = torch.zeros_like(local_transforms)

        # First global transform equals first local transform
        # global[0] = local[0] = T_{0←1}
        global_transforms[0] = local_transforms[0]

        # Accumulate: global[i] = global[i-1] @ local[i]
        # i.e., T_{0←i+1} = T_{0←i} @ T_{i←i+1}
        # LOOP INVARIANT: After iteration i, global_transforms[i] contains
        # T_{0←i+1}, the transformation that maps frame i+1 to frame 0.
        for i in range(1, num_transforms):
            global_transforms[i] = compose_transforms(
                transform_a_to_b=local_transforms[i],      # T_{i←i+1}
                transform_b_to_c=global_transforms[i - 1],  # T_{0←i}
            )
            # Result: T_{0←i} @ T_{i←i+1} = T_{0←i+1}

        return global_transforms

    @staticmethod
    def accumulate_batched(
        local_transforms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert local transforms to global transforms with batch dimension.

        Args:
            local_transforms: Local transformation matrices, shape [B, N-1, 4, 4].
                              B is batch size, N-1 is number of transform pairs.
                              local_transforms[:, i] = T_{i←i+1}: maps frame i+1 to frame i.

        Returns:
            Global transformation matrices, shape [B, N-1, 4, 4].
            global_transforms[:, i] = T_{0←i+1}: maps frame i+1 to frame 0.

        Raises:
            ValueError: If local_transforms has wrong shape.
        """
        if local_transforms.ndim != 4 or local_transforms.shape[2:] != (4, 4):
            raise ValueError(
                f"local_transforms must have shape [B, N-1, 4, 4], "
                f"got {local_transforms.shape}"
            )

        batch_size, num_transforms = local_transforms.shape[:2]
        if num_transforms == 0:
            return local_transforms.clone()

        global_transforms = torch.zeros_like(local_transforms)
        global_transforms[:, 0] = local_transforms[:, 0]

        # LOOP INVARIANT: After iteration i, global_transforms[:, i] contains
        # T_{0←i+1}, the transformation that maps frame i+1 to frame 0, for all batches.
        for i in range(1, num_transforms):
            global_transforms[:, i] = compose_transforms(
                transform_a_to_b=local_transforms[:, i],      # T_{i←i+1}
                transform_b_to_c=global_transforms[:, i - 1],  # T_{0←i}
            )
            # Result: T_{0←i} @ T_{i←i+1} = T_{0←i+1}

        return global_transforms


def local_to_global_transforms(
    local_transforms: torch.Tensor,
) -> torch.Tensor:
    """
    Convenience function to convert local to global transforms.

    This is a wrapper around TransformAccumulator.accumulate_local_to_global
    for simpler usage.

    Args:
        local_transforms: Local transformation matrices, shape [N-1, 4, 4].

    Returns:
        Global transformation matrices, shape [N-1, 4, 4].
    """
    return TransformAccumulator.accumulate_local_to_global(local_transforms)
