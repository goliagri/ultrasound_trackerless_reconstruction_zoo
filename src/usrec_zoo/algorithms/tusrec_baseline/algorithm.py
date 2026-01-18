"""
TUS-REC2025 Challenge baseline algorithm implementation.

This module implements the AlgorithmInterface for the baseline method,
which uses an EfficientNet-B1 to predict 6DOF transformation parameters
between consecutive ultrasound frames.
"""

from typing import Any, Dict, List, Optional
import warnings

import torch

from usrec_zoo.algorithms.base import AlgorithmInterface, ConfigSchema
from usrec_zoo.algorithms.registry import register_algorithm
from usrec_zoo.algorithms.tusrec_baseline.model import TransformPredictor
from usrec_zoo.algorithms.tusrec_baseline.config import (
    get_default_config as _get_default_config,
    get_config_schema as _get_config_schema,
)
from usrec_zoo.calibration import CalibrationData, CalibrationDataTorch
from usrec_zoo.types import ScanData, TransformPrediction, TrainResult
from usrec_zoo.transforms import params_to_matrix
from usrec_zoo.transforms.accumulation import TransformAccumulator


@register_algorithm("tusrec_baseline")
class TUSRECBaseline(AlgorithmInterface):
    """
    TUS-REC2025 Challenge baseline algorithm.

    This algorithm uses an EfficientNet-B1 backbone to predict 6DOF
    transformation parameters (rx, ry, rz, tx, ty, tz) between
    consecutive ultrasound frames.

    The workflow is:
    1. Input N consecutive frames
    2. Network predicts 6DOF parameters for frame pairs
    3. Convert 6DOF to 4x4 transformation matrices
    4. Accumulate local transforms to get global transforms

    Attributes:
        model: The neural network predictor.
        config: Validated configuration dictionary.
        calibration: CalibrationData for coordinate transforms.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        calibration: CalibrationData,
    ) -> None:
        """
        Initialize the baseline algorithm.

        Args:
            config: Algorithm configuration (validated against schema).
            calibration: CalibrationData instance.
        """
        super().__init__(config, calibration)

        # Build model
        self.model = TransformPredictor(
            config=self.config,
            num_frames=self.config["num_samples"],
            num_pairs=self.config["num_pred"],
            pred_dim=6,  # 6DOF parameters
        )

        # Cache for calibration tensors on different devices
        self._calib_cache: Dict[torch.device, CalibrationDataTorch] = {}

    def _get_calib_torch(self, device: torch.device) -> CalibrationDataTorch:
        """Get CalibrationDataTorch for the specified device."""
        if device not in self._calib_cache:
            self._calib_cache[device] = CalibrationDataTorch(self.calibration, device)
        return self._calib_cache[device]

    def predict(
        self,
        scan: ScanData,
        device: torch.device,
    ) -> TransformPrediction:
        """
        Run inference on a scan to predict transformations.

        This method:
        1. Prepares frame pairs from the scan
        2. Runs the neural network to get 6DOF predictions
        3. Converts predictions to 4x4 matrices
        4. Accumulates local transforms to global transforms

        Args:
            scan: ScanData containing frames.
            device: PyTorch device for inference.

        Returns:
            TransformPrediction with local and global transforms.

        Raises:
            ValueError: If scan has fewer than 2 frames.
        """
        # Input validation
        if scan.num_frames < 2:
            raise ValueError(
                f"Scan must have at least 2 frames for prediction, "
                f"got {scan.num_frames}"
            )

        self.model.to(device)
        self.model.eval()

        num_frames = scan.num_frames
        num_pairs = num_frames - 1

        # Prepare frames tensor
        frames = torch.tensor(scan.frames, dtype=torch.float32, device=device)

        # Predict transforms for each consecutive frame pair
        local_transforms_list = []

        # LOOP INVARIANT: After iteration i, local_transforms_list contains
        # i+1 transformation matrices, each mapping frame j+1 to frame j
        # for j in [0, i].
        with torch.no_grad():
            for i in range(num_pairs):
                # Get frame pair [frame_i, frame_{i+1}]
                # Shape: [1, 2, H, W]
                frame_pair = frames[i:i + 2].unsqueeze(0)

                # Predict 6DOF parameters
                # Output shape: [1, 1, 6]
                params = self.model(frame_pair)
                params = params.squeeze(0).squeeze(0)  # [6]

                # Convert to 4x4 matrix in image-mm space
                # The network predicts transform from frame i+1 to frame i
                transform = params_to_matrix(params)  # [4, 4]

                local_transforms_list.append(transform)

        # Stack local transforms: [N-1, 4, 4]
        local_transforms = torch.stack(local_transforms_list, dim=0)

        # Compute global transforms by accumulation
        global_transforms = TransformAccumulator.accumulate_local_to_global(
            local_transforms
        )

        return TransformPrediction(
            local_transforms=local_transforms,
            global_transforms=global_transforms,
        )

    def forward_batch(
        self,
        frames_batch: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Run batched forward pass for training (returns raw 6DOF parameters).

        This is a lower-level method used by the training loop. For inference
        on complete scans, use predict() instead.

        Args:
            frames_batch: Batch of frame sequences, shape [B, num_samples, H, W].
                          Values should be in [0, 255] range (uint8 or float32).
                          The model normalizes to [0, 1] internally.
            device: PyTorch device.

        Returns:
            Predicted 6DOF parameters, shape [B, num_pred, 6].
            Parameters are (rx, ry, rz, tx, ty, tz) where r is Euler angles
            in radians (ZYX convention) and t is translation in mm.

        Raises:
            ValueError: If frames_batch has incorrect shape.
        """
        if frames_batch.ndim != 4:
            raise ValueError(
                f"frames_batch must be 4D [B, num_samples, H, W], "
                f"got shape {frames_batch.shape}"
            )

        self.model.to(device)
        frames_batch = frames_batch.to(device)
        return self.model(frames_batch)

    def train(
        self,
        train_scans: List[ScanData],
        val_scans: Optional[List[ScanData]] = None,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> TrainResult:
        """
        Train the algorithm using the baseline training procedure.

        This overrides the default SharedTrainer to use the original
        TUS-REC training loop logic.

        Args:
            train_scans: List of training scans.
            val_scans: Optional validation scans.
            device: Training device.
            **kwargs: Additional arguments (e.g., num_epochs override).

        Returns:
            TrainResult with training history.
        """
        # For now, delegate to SharedTrainer
        # In the future, this can implement the exact original training loop
        from usrec_zoo.training import SharedTrainer

        trainer = SharedTrainer(self, self.config)
        return trainer.train(train_scans, val_scans, device, **kwargs)

    def _get_state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary for checkpointing.

        Returns:
            Dictionary containing:
            - model_state_dict: Neural network weights
            - config: Algorithm configuration
            - calibration_hash: Hash of calibration data for validation
        """
        return {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "calibration_hash": self.calibration.content_hash(),
        }

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state from checkpoint.

        Args:
            state_dict: Dictionary from checkpoint file.

        Raises:
            ValueError: If calibration hash doesn't match (warning only).

        Note:
            Calibration hash mismatch logs a warning but does not raise
            an exception to allow loading checkpoints trained with
            different calibration (at user's risk).
        """
        # Validate calibration hash if present
        if "calibration_hash" in state_dict:
            expected_hash = state_dict["calibration_hash"]
            actual_hash = self.calibration.content_hash()
            if expected_hash != actual_hash:
                warnings.warn(
                    f"Calibration hash mismatch: checkpoint was trained with "
                    f"calibration hash '{expected_hash}', but current calibration "
                    f"has hash '{actual_hash}'. Results may be invalid.",
                    UserWarning
                )

        self.model.load_state_dict(state_dict["model_state_dict"])

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration."""
        return _get_default_config()

    @classmethod
    def get_config_schema(cls) -> ConfigSchema:
        """Get configuration schema."""
        return _get_config_schema()

    @classmethod
    def get_name(cls) -> str:
        """Get algorithm name."""
        return "tusrec_baseline"

    @classmethod
    def get_description(cls) -> str:
        """Get algorithm description."""
        return (
            "TUS-REC2025 Challenge baseline: EfficientNet-B1 predicting "
            "6DOF transformation parameters between consecutive frames."
        )

    def get_model(self) -> torch.nn.Module:
        """
        Return the underlying PyTorch model for training.

        Returns:
            The TransformPredictor model instance.
        """
        return self.model
