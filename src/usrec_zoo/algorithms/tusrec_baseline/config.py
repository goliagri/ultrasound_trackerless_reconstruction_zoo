"""
Configuration schema and defaults for the TUS-REC baseline algorithm.
"""

from typing import Any, Dict

from usrec_zoo.algorithms.base import ConfigSchema


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration for the TUS-REC baseline.

    Returns:
        Dictionary of default configuration values.

    Configuration keys:
        model_name: Neural network architecture name.
        pred_type: Prediction output type ('parameter', 'transform', 'quaternion').
        label_type: Label format for training ('point', 'parameter', 'transform').
        num_samples: Number of input frames per sample.
        sample_range: Range from which to sample frames.
        num_pred: Number of transform predictions per sample.
        batch_size: Training batch size.
        learning_rate: Optimizer learning rate.
        num_epochs: Maximum training epochs.
        freq_info: Frequency of logging (iterations).
        freq_save: Frequency of checkpoint saving (iterations).
        val_freq: Frequency of validation (epochs).
    """
    return {
        # Model architecture
        "model_name": "efficientnet_b1",

        # Prediction/label types
        "pred_type": "parameter",  # 6DOF parameters
        "label_type": "point",  # Point-based loss

        # Sampling configuration
        "num_samples": 2,  # Number of input frames
        "sample_range": 2,  # Frames sampled from this range
        "num_pred": 1,  # Number of predictions (frame pairs)

        # Training hyperparameters
        "batch_size": 16,  # Default; use 4-8 for 12GB GPU
        "learning_rate": 1e-4,
        "num_epochs": int(1e8),  # Train until convergence

        # Logging/checkpointing
        "freq_info": 10,
        "freq_save": 100,
        "val_freq": 1,

        # Data paths (can be overridden)
        "data_path": "data/frames_transfs",
        "calib_path": "data/calib_matrix.csv",
        "landmark_path": "data/landmarks",
    }


def get_config_schema() -> ConfigSchema:
    """
    Get the configuration schema for validation.

    Returns:
        ConfigSchema instance defining required and optional keys.
    """
    return ConfigSchema(
        required={
            "model_name": str,
        },
        optional={
            "pred_type": (str, "parameter"),
            "label_type": (str, "point"),
            "num_samples": (int, 2),
            "sample_range": (int, 2),
            "num_pred": (int, 1),
            "batch_size": (int, 16),
            "learning_rate": (float, 1e-4),
            "num_epochs": (int, int(1e8)),
            "freq_info": (int, 10),
            "freq_save": (int, 100),
            "val_freq": (int, 1),
            "data_path": (str, "data/frames_transfs"),
            "calib_path": (str, "data/calib_matrix.csv"),
            "landmark_path": (str, "data/landmarks"),
        },
        constraints={
            "pred_type": lambda v: v in ("parameter", "transform", "quaternion", "point"),
            "label_type": lambda v: v in ("point", "parameter", "transform"),
            "num_samples": lambda v: v >= 2,
            "sample_range": lambda v: v >= 2,
            "num_pred": lambda v: v >= 1,
            "batch_size": lambda v: v >= 1,
            "learning_rate": lambda v: v > 0,
            "num_epochs": lambda v: v >= 1,
            "freq_info": lambda v: v >= 1,
            "freq_save": lambda v: v >= 1,
            "val_freq": lambda v: v >= 1,
        },
    )
