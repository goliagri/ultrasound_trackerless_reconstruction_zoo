"""
Shared training infrastructure for reconstruction algorithms.

This module provides a generic trainer that can be used by algorithms
that don't need custom training logic. It handles:
- Data loading and batching
- Optimization loop
- Validation
- Checkpointing
- Logging
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from usrec_zoo.types import ScanData, TrainResult

if TYPE_CHECKING:
    from usrec_zoo.algorithms.base import AlgorithmInterface


class SharedTrainer:
    """
    Generic trainer for reconstruction algorithms.

    This trainer provides a standard training loop that works with any
    algorithm implementing AlgorithmInterface. Algorithms can override
    the train() method if they need custom logic.

    Attributes:
        algorithm: The algorithm instance to train.
        config: Training configuration dictionary.

    Expected config keys:
        num_epochs (int): Number of training epochs. Default: 100.
        learning_rate (float): Learning rate for optimizer. Default: 1e-4.
        batch_size (int): Batch size for training. Default: 16.
        freq_info (int): Frequency of info logging in epochs. Default: 10.
        freq_save (int): Frequency of checkpoint saving in epochs. Default: 100.
        val_freq (int): Frequency of validation in epochs. Default: 1.
        save_path (str): Directory to save checkpoints. Default: "experiments/default".
    """

    def __init__(
        self,
        algorithm: "AlgorithmInterface",
        config: Dict[str, Any],
    ) -> None:
        """
        Initialize the trainer.

        Args:
            algorithm: Algorithm instance with a trainable model. Must implement
                       AlgorithmInterface and return a non-None model from get_model().
            config: Configuration dictionary with training hyperparameters.

        Raises:
            TypeError: If algorithm is None.
            TypeError: If config is not a dictionary.
        """
        if algorithm is None:
            raise TypeError("algorithm cannot be None")
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")

        self.algorithm = algorithm
        self.config = config

    def train(
        self,
        train_scans: List[ScanData],
        val_scans: Optional[List[ScanData]] = None,
        device: torch.device = torch.device("cpu"),
        **kwargs: Any,
    ) -> TrainResult:
        """
        Run the training loop.

        This is a stub implementation. The full implementation will:
        1. Create data loaders from scans
        2. Initialize optimizer and scheduler
        3. Run training epochs
        4. Validate periodically
        5. Save checkpoints

        Args:
            train_scans: List of training scans. Must be non-empty.
            val_scans: Optional validation scans. If provided, validation will
                       run at the frequency specified by val_freq config.
            device: Training device (CPU or CUDA).
            **kwargs: Override config values. Keys must match expected config keys.

        Returns:
            TrainResult with training history and best checkpoint path.

        Raises:
            ValueError: If train_scans is empty.
            RuntimeError: If algorithm.get_model() returns None.

        Note:
            This is currently a stub implementation that demonstrates the
            training loop structure. It processes a limited number of epochs
            and uses placeholder loss computation. Full implementation will
            include proper data loading and loss computation.
        """
        # Input validation
        if not train_scans:
            raise ValueError("train_scans cannot be empty")

        # Merge kwargs into config
        config: Dict[str, Any] = {**self.config, **kwargs}

        # Extract config values with defaults
        num_epochs: int = config.get("num_epochs", 100)
        learning_rate: float = config.get("learning_rate", 1e-4)
        batch_size: int = config.get("batch_size", 16)
        freq_info: int = config.get("freq_info", 10)
        freq_save: int = config.get("freq_save", 100)
        val_freq: int = config.get("val_freq", 1)
        save_path: Path = Path(config.get("save_path", "experiments/default"))

        # Validate config values
        if num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {num_epochs}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Create save directory
        save_path.mkdir(parents=True, exist_ok=True)

        # Get model via the interface method
        model: Optional[nn.Module] = self.algorithm.get_model()
        if model is None:
            raise RuntimeError(
                f"Algorithm {type(self.algorithm).__name__}.get_model() returned None. "
                "SharedTrainer requires a trainable model."
            )

        model.to(device)
        model.train()

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training history
        history: List[Dict[str, Any]] = []
        best_val_metric: float = float("inf")
        best_val_epoch: int = 0
        start_time: float = time.time()

        print(f"Starting training with {len(train_scans)} scans on {device}")
        print(f"Config: num_epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")

        # STUB: Limit to 3 epochs for demonstration purposes
        # TODO: Remove this limit and implement proper data loading
        max_stub_epochs: int = min(num_epochs, 3)

        # LOOP INVARIANT: At the start of each iteration:
        #   - epoch is the current epoch index (0-based)
        #   - history contains records for epochs 0 to epoch-1
        #   - best_val_metric is the minimum validation metric seen so far
        #   - best_val_epoch is the epoch (1-based) that achieved best_val_metric
        for epoch in range(max_stub_epochs):
            epoch_start: float = time.time()
            epoch_loss: float = 0.0
            num_batches: int = 0

            # STUB: Placeholder training step
            # Full implementation would iterate over batches from a DataLoader
            scans_to_process: int = min(batch_size, len(train_scans))
            for scan in train_scans[:scans_to_process]:
                optimizer.zero_grad()

                # STUB: Placeholder forward pass
                # This creates a minimal computation graph for demonstration
                # Real implementation would:
                # 1. Extract frame pairs from scan
                # 2. Run through model
                # 3. Compute actual loss
                dummy_param = next(model.parameters())
                loss: torch.Tensor = dummy_param.sum() * 0  # Zero loss for stub

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss: float = epoch_loss / max(num_batches, 1)
            epoch_time: float = time.time() - epoch_start

            # Log progress
            if (epoch + 1) % freq_info == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.6f}, "
                      f"time={epoch_time:.2f}s")

            # Validation
            val_loss: Optional[float] = None
            if val_scans and (epoch + 1) % val_freq == 0:
                model.eval()
                # Limit validation scans for stub
                scans_to_validate = val_scans[:min(3, len(val_scans))]
                val_result = self.algorithm.validate(scans_to_validate, device)
                val_loss = val_result.mean_landmark_error
                model.train()

                if val_loss < best_val_metric:
                    best_val_metric = val_loss
                    best_val_epoch = epoch + 1

                    # Save best checkpoint
                    best_checkpoint_path: Path = save_path / "best_model.pt"
                    self.algorithm.save_checkpoint(
                        best_checkpoint_path,
                        metadata={"epoch": epoch + 1, "val_metric": val_loss}
                    )

            # Record history
            history.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": val_loss,
            })

            # Save periodic checkpoint
            if (epoch + 1) % freq_save == 0:
                periodic_checkpoint_path: Path = save_path / f"checkpoint_epoch_{epoch + 1:08d}.pt"
                self.algorithm.save_checkpoint(
                    periodic_checkpoint_path,
                    metadata={"epoch": epoch + 1}
                )

        total_time: float = time.time() - start_time
        print(f"Training complete in {total_time:.1f}s. "
              f"Best val metric: {best_val_metric:.6f} at epoch {best_val_epoch}")

        # Determine final values
        final_train_loss: float = history[-1]["train_loss"] if history else 0.0
        final_val_loss: Optional[float] = history[-1]["val_loss"] if history else None

        return TrainResult(
            final_epoch=len(history),
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            best_val_epoch=best_val_epoch if best_val_metric < float("inf") else None,
            best_val_metric=best_val_metric if best_val_metric < float("inf") else None,
            checkpoint_path=str(save_path / "best_model.pt"),
            training_history=history,
            training_time_seconds=total_time,
        )
