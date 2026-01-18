"""
Neural network model for the TUS-REC baseline algorithm.

This module provides the EfficientNet-B1 based architecture used in the
TUS-REC2025 Challenge baseline implementation.
"""

from typing import Dict, Any

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1


def build_model(
    config: Dict[str, Any],
    in_frames: int,
    pred_dim: int,
) -> nn.Module:
    """
    Build the neural network model for transformation prediction.

    The model is based on EfficientNet-B1 with modified input and output layers
    to handle multi-frame input and 6DOF parameter output.

    Args:
        config: Configuration dictionary containing 'model_name'.
        in_frames: Number of input frames (channels).
        pred_dim: Output dimension (typically 6 for 6DOF parameters).

    Returns:
        PyTorch model ready for training/inference.

    Raises:
        ValueError: If the model_name in config is not recognized.

    Example:
        >>> config = {"model_name": "efficientnet_b1"}
        >>> model = build_model(config, in_frames=2, pred_dim=6)
        >>> output = model(torch.randn(1, 2, 480, 640))
        >>> output.shape
        torch.Size([1, 6])
    """
    model_name = config.get("model_name", "efficientnet_b1")

    if model_name == "efficientnet_b1":
        model = efficientnet_b1(weights=None)

        # Modify input layer to accept variable number of input channels
        original_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels=in_frames,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Modify output layer for prediction dimension
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features=in_features,
            out_features=pred_dim,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


class TransformPredictor(nn.Module):
    """
    Wrapper module for transform prediction with pre/post processing.

    This module wraps the base network and handles:
    - Input normalization
    - Multi-pair prediction (if needed)
    - Output reshaping

    Attributes:
        backbone: The underlying network (e.g., EfficientNet-B1).
        num_pairs: Number of frame pairs to predict transforms for.
        pred_dim: Dimension of each transform prediction (6 for 6DOF).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        num_frames: int = 2,
        num_pairs: int = 1,
        pred_dim: int = 6,
    ) -> None:
        """
        Initialize the transform predictor.

        Args:
            config: Model configuration dictionary.
            num_frames: Number of input frames.
            num_pairs: Number of transform pairs to predict.
            pred_dim: Prediction dimension per pair (6 for 6DOF).
        """
        super().__init__()

        if num_frames < 2:
            raise ValueError(f"num_frames must be >= 2, got {num_frames}")
        if num_pairs < 1:
            raise ValueError(f"num_pairs must be >= 1, got {num_pairs}")

        self.num_frames = num_frames
        self.num_pairs = num_pairs
        self.pred_dim = pred_dim

        # Total output dimension
        total_pred_dim = num_pairs * pred_dim

        # Build backbone
        self.backbone = build_model(
            config,
            in_frames=num_frames,
            pred_dim=total_pred_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [B, num_frames, H, W].
               Values should be in range [0, 255] (uint8 or float32).
               The model normalizes to [0, 1] internally by dividing by 255.
               If values are already in [0, 1] range, caller should multiply
               by 255 before passing to this method.

        Returns:
            Prediction tensor of shape [B, num_pairs, pred_dim].

        Raises:
            ValueError: If input tensor has wrong number of dimensions or channels.

        Note:
            The normalization matches the original TUS-REC baseline which uses
            frames/255. No ImageNet-style mean/std normalization is applied.
        """
        # Input validation
        if x.ndim != 4:
            raise ValueError(
                f"Input must be 4D [B, num_frames, H, W], got {x.ndim}D"
            )
        if x.shape[1] != self.num_frames:
            raise ValueError(
                f"Input must have {self.num_frames} frames (channels), "
                f"got {x.shape[1]}"
            )

        # Normalize input to [0, 1] range
        # The original TUS-REC baseline divides by 255 unconditionally.
        # No ImageNet-style mean/std normalization is applied.
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x / 255.0

        # Forward through backbone
        output = self.backbone(x)  # [B, num_pairs * pred_dim]

        # Reshape to [B, num_pairs, pred_dim]
        output = output.view(-1, self.num_pairs, self.pred_dim)

        return output
