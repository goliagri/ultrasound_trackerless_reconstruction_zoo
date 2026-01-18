"""
Base algorithm interface for ultrasound reconstruction methods.

All reconstruction algorithms must implement the AlgorithmInterface ABC
to ensure compatibility with the training, evaluation, and prediction pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Type

import torch

from usrec_zoo.types import (
    ScanData,
    TransformPrediction,
    ValidationResult,
    TrainResult,
)
from usrec_zoo.calibration import CalibrationData


@dataclass
class ConfigSchema:
    """
    Schema definition for algorithm configuration validation.

    This class defines the expected configuration keys, their types,
    default values, and constraints for an algorithm.

    Attributes:
        required: Dictionary of required config keys with their expected types.
        optional: Dictionary of optional config keys with (type, default_value).
        constraints: Dictionary of key -> validation function returning bool.
    """
    required: Dict[str, Type] = field(default_factory=dict)
    optional: Dict[str, tuple] = field(default_factory=dict)  # key -> (type, default)
    constraints: Dict[str, Callable[[Any], bool]] = field(default_factory=dict)

    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize a configuration dictionary.

        Args:
            config: Configuration dictionary to validate.

        Returns:
            Normalized configuration with defaults filled in.

        Raises:
            ValueError: If required keys are missing or types don't match.
            ValueError: If constraints are violated.
        """
        validated = {}

        # Check required keys
        for key, expected_type in self.required.items():
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
            value = config[key]
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"Config key '{key}' must be {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
            validated[key] = value

        # Check optional keys and fill defaults
        for key, (expected_type, default) in self.optional.items():
            if key in config:
                value = config[key]
                if not isinstance(value, expected_type):
                    raise ValueError(
                        f"Config key '{key}' must be {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
                validated[key] = value
            else:
                validated[key] = default

        # Check constraints
        for key, constraint_fn in self.constraints.items():
            if key in validated:
                if not constraint_fn(validated[key]):
                    raise ValueError(f"Constraint violated for config key: {key}")

        # Check for unknown keys (typo detection)
        valid_keys = set(self.required.keys()) | set(self.optional.keys())
        unknown_keys = set(config.keys()) - valid_keys
        if unknown_keys:
            raise ValueError(
                f"Unknown config keys: {unknown_keys}. "
                f"Valid keys are: {valid_keys}"
            )

        return validated


class AlgorithmInterface(ABC):
    """
    Abstract base class for ultrasound reconstruction algorithms.

    All reconstruction algorithms must inherit from this class and implement
    the required abstract methods. This ensures compatibility with the
    shared training, evaluation, and prediction infrastructure.

    Attributes:
        config: Validated configuration dictionary.
        calibration: CalibrationData instance for coordinate transforms.
        _checkpoint_path: Path to loaded checkpoint (if any).

    Example:
        >>> class MyAlgorithm(AlgorithmInterface):
        ...     def predict(self, scan, device):
        ...         # Implementation
        ...         pass
        ...
        >>> algo = MyAlgorithm(config, calibration)
        >>> prediction = algo.predict(scan_data, device)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        calibration: CalibrationData,
    ) -> None:
        """
        Initialize the algorithm with configuration and calibration data.

        Args:
            config: Algorithm configuration dictionary. Will be validated
                    against get_config_schema().
            calibration: CalibrationData instance for coordinate transforms.
        """
        # Validate config against schema
        schema = self.get_config_schema()
        self.config = schema.validate(config)
        self.calibration = calibration
        self._checkpoint_path: Optional[Path] = None

    @abstractmethod
    def predict(
        self,
        scan: ScanData,
        device: torch.device,
    ) -> TransformPrediction:
        """
        Run inference on a single scan to predict transformations.

        This is the core method that algorithms must implement. It takes
        a scan's frames and predicts the transformations between them.

        Args:
            scan: ScanData containing frames (and optionally ground truth).
            device: PyTorch device to run inference on.

        Returns:
            TransformPrediction containing local and global transforms.

        Note:
            The returned transforms should be in image-mm coordinate system.
            Algorithms that predict in other spaces (e.g., tracker space)
            should convert internally before returning.
        """
        pass

    def validate(
        self,
        val_scans: List[ScanData],
        device: torch.device,
    ) -> ValidationResult:
        """
        Validate the algorithm on a set of validation scans.

        Default implementation runs predict() on each scan and computes
        metrics. Algorithms may override for custom validation logic.

        Args:
            val_scans: List of ScanData objects for validation.
            device: PyTorch device for inference.

        Returns:
            ValidationResult with computed metrics.

        Raises:
            ValueError: If val_scans is empty.
        """
        if not val_scans:
            raise ValueError("val_scans cannot be empty")

        from usrec_zoo.evaluation import transforms_to_ddf, compute_landmark_error
        from usrec_zoo.calibration import CalibrationDataTorch

        calib_torch = CalibrationDataTorch(self.calibration, device)
        per_scan_errors: Dict[str, float] = {}
        all_errors: List[float] = []

        # LOOP INVARIANT: per_scan_errors contains error for all processed scans
        # with landmarks and transforms; all_errors contains the same values as list
        for scan in val_scans:
            if scan.landmarks is None:
                continue

            # Get prediction
            prediction = self.predict(scan, device)

            # Compute DDF
            ddf = transforms_to_ddf(prediction, scan.landmarks, calib_torch)

            # Compute ground truth DDF (requires ground truth transforms)
            if scan.transforms is not None:
                gt_prediction = self._compute_gt_prediction(scan, device)
                gt_ddf = transforms_to_ddf(gt_prediction, scan.landmarks, calib_torch)
                error = compute_landmark_error(ddf, gt_ddf, use_global=True)
                scan_id = f"{scan.subject_id}/{scan.scan_name}"
                per_scan_errors[scan_id] = error
                all_errors.append(error)

        mean_error = sum(all_errors) / len(all_errors) if all_errors else 0.0

        return ValidationResult(
            mean_landmark_error=mean_error,
            per_scan_errors=per_scan_errors,
        )

    def _compute_gt_prediction(
        self,
        scan: ScanData,
        device: torch.device,
    ) -> TransformPrediction:
        """
        Compute ground truth TransformPrediction from tracker transforms.

        This is a helper method for validation that converts ground truth
        tracker transforms to the same format as algorithm predictions.

        Args:
            scan: ScanData with ground truth transforms.
            device: PyTorch device.

        Returns:
            TransformPrediction from ground truth.

        Raises:
            ValueError: If scan does not have ground truth transforms.
        """
        from usrec_zoo.calibration import CalibrationDataTorch

        if scan.transforms is None:
            raise ValueError("Scan must have ground truth transforms")

        calib_torch = CalibrationDataTorch(self.calibration, device)

        # Convert numpy transforms to torch
        tracker_transforms = torch.tensor(
            scan.transforms, dtype=torch.float32, device=device
        )
        tracker_transforms_inv = torch.linalg.inv(tracker_transforms)

        num_frames = scan.num_frames
        local_transforms: List[torch.Tensor] = []
        global_transforms: List[torch.Tensor] = []

        # Compute transforms in image-mm space
        tform_tool_to_mm = calib_torch.tform_tool_to_mm
        tform_mm_to_tool = calib_torch.tform_mm_to_tool

        # LOOP INVARIANT: After iteration i, local_transforms contains transforms
        # for frames 1..i (each mapping to previous frame), and global_transforms
        # contains transforms for frames 1..i (each mapping to frame 0)
        for i in range(1, num_frames):
            # Local: frame i to frame i-1
            t_tool_local = torch.matmul(
                tracker_transforms_inv[i - 1],
                tracker_transforms[i]
            )
            t_local = torch.matmul(
                tform_tool_to_mm,
                torch.matmul(t_tool_local, tform_mm_to_tool)
            )
            local_transforms.append(t_local)

            # Global: frame i to frame 0
            t_tool_global = torch.matmul(
                tracker_transforms_inv[0],
                tracker_transforms[i]
            )
            t_global = torch.matmul(
                tform_tool_to_mm,
                torch.matmul(t_tool_global, tform_mm_to_tool)
            )
            global_transforms.append(t_global)

        return TransformPrediction(
            local_transforms=torch.stack(local_transforms),
            global_transforms=torch.stack(global_transforms),
        )

    def train(
        self,
        train_scans: List[ScanData],
        val_scans: Optional[List[ScanData]] = None,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> TrainResult:
        """
        Train the algorithm on a set of training scans.

        Default implementation uses SharedTrainer. Algorithms may override
        for custom training logic.

        Args:
            train_scans: List of ScanData objects for training.
            val_scans: Optional list of validation scans.
            device: PyTorch device for training.
            **kwargs: Additional training arguments.

        Returns:
            TrainResult with training history and metrics.
        """
        from usrec_zoo.training import SharedTrainer

        trainer = SharedTrainer(self, self.config)
        return trainer.train(train_scans, val_scans, device, **kwargs)

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file.
            device: Device to load the checkpoint onto.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self._load_state_dict(checkpoint)
        self._checkpoint_path = checkpoint_path

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save model weights to a checkpoint file.

        Args:
            checkpoint_path: Path to save the checkpoint.
            metadata: Optional metadata to include (e.g., epoch, metrics).
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = self._get_state_dict()
        if metadata:
            state_dict["metadata"] = metadata

        torch.save(state_dict, checkpoint_path)

    @abstractmethod
    def _get_state_dict(self) -> Dict[str, Any]:
        """
        Get the state dictionary for checkpointing.

        Returns:
            Dictionary containing model state.
        """
        pass

    @abstractmethod
    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state from a state dictionary.

        Args:
            state_dict: Dictionary containing model state.
        """
        pass

    @classmethod
    @abstractmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get the default configuration for this algorithm.

        Returns:
            Dictionary of default configuration values.
        """
        pass

    @classmethod
    @abstractmethod
    def get_config_schema(cls) -> ConfigSchema:
        """
        Get the configuration schema for validation.

        Returns:
            ConfigSchema instance defining required and optional keys.
        """
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Get the human-readable name of this algorithm.

        Returns:
            Algorithm name string (e.g., "tusrec_baseline").
        """
        pass

    @classmethod
    def get_description(cls) -> str:
        """
        Get a brief description of this algorithm.

        Returns:
            Description string.
        """
        return cls.__doc__ or "No description available."

    def get_model(self) -> Optional[torch.nn.Module]:
        """
        Return the underlying PyTorch model for training.

        This method should be overridden by subclasses to return their
        trainable model. The SharedTrainer uses this to access model
        parameters for optimization.

        Returns:
            The PyTorch model, or None if the algorithm doesn't expose
            a single trainable model (e.g., ensemble methods).

        Warning:
            Modifications to the returned model may affect algorithm state.
            Use with caution.
        """
        return None
