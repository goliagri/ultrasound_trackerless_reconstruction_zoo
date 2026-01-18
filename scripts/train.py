#!/usr/bin/env python3
"""
Training script for ultrasound reconstruction algorithms.

Usage:
    python scripts/train.py --algorithm tusrec_baseline --data_path data/frames_transfs

    # With custom batch size for limited GPU memory:
    python scripts/train.py --algorithm tusrec_baseline --batch_size 4
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from usrec_zoo.algorithms import get_algorithm, list_algorithms
from usrec_zoo.calibration import CalibrationData
from usrec_zoo.data import ScanDataset, get_canonical_splits


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an ultrasound reconstruction algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Algorithm selection
    parser.add_argument(
        "--algorithm",
        type=str,
        default="tusrec_baseline",
        choices=list_algorithms(),
        help="Algorithm to train",
    )

    # Data paths
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/frames_transfs",
        help="Path to training data directory",
    )
    parser.add_argument(
        "--calib_path",
        type=str,
        default="data/calib_matrix.csv",
        help="Path to calibration matrix CSV",
    )
    parser.add_argument(
        "--landmark_path",
        type=str,
        default="data/landmarks",
        help="Path to landmarks directory",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size (use 4-8 for 12GB GPU)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Maximum number of training epochs",
    )

    # Output
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save checkpoints (default: experiments/<algorithm>)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main() -> int:
    """Main training function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    print(f"Training algorithm: {args.algorithm}")
    print(f"Device: {args.device}")

    # Load calibration
    calib_path = Path(args.calib_path)
    if not calib_path.exists():
        # Try relative to TUS-REC baseline directory
        alt_path = Path("TUS-REC2025-Challenge_baseline") / args.calib_path
        if alt_path.exists():
            calib_path = alt_path
        else:
            print(f"ERROR: Calibration file not found: {args.calib_path}")
            return 1

    print(f"Loading calibration from: {calib_path}")
    calibration = CalibrationData.from_csv(calib_path)

    # Create algorithm
    config = {
        "model_name": "efficientnet_b1",
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
    }

    if args.save_path:
        config["save_path"] = args.save_path
    else:
        config["save_path"] = f"experiments/{args.algorithm}"

    algorithm = get_algorithm(args.algorithm, config=config, calibration=calibration)
    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Config: {algorithm.config}")

    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        alt_path = Path("TUS-REC2025-Challenge_baseline") / args.data_path
        if alt_path.exists():
            data_path = alt_path
        else:
            print(f"ERROR: Data path not found: {args.data_path}")
            return 1

    landmark_path = Path(args.landmark_path)
    if not landmark_path.exists():
        alt_path = Path("TUS-REC2025-Challenge_baseline") / args.landmark_path
        if alt_path.exists():
            landmark_path = alt_path
        else:
            landmark_path = None  # Landmarks optional

    print(f"Loading data from: {data_path}")

    # Create dataset and split
    dataset = ScanDataset(
        data_path=data_path,
        num_samples=2,
        landmark_path=landmark_path,
    )

    # Split into train/val/test (60/20/20)
    train_dataset, val_dataset, test_dataset = dataset.partition_by_ratio(
        [0.6, 0.2, 0.2],
        randomize=True,
        seed=args.seed,
    )

    print(f"Train: {len(train_dataset)} scans")
    print(f"Val: {len(val_dataset)} scans")
    print(f"Test: {len(test_dataset)} scans")

    # Load scans for training
    # Note: For full training, we'd use DataLoader with proper batching
    train_scans = [train_dataset[i] for i in range(min(len(train_dataset), 100))]
    val_scans = [val_dataset[i] for i in range(min(len(val_dataset), 20))]

    # Train
    device = torch.device(args.device)
    print(f"\nStarting training on {device}...")

    result = algorithm.train(
        train_scans=train_scans,
        val_scans=val_scans,
        device=device,
    )

    print(f"\nTraining complete!")
    print(f"Final epoch: {result.final_epoch}")
    print(f"Final train loss: {result.final_train_loss:.6f}")
    if result.final_val_loss:
        print(f"Final val loss: {result.final_val_loss:.6f}")
    if result.best_val_epoch:
        print(f"Best val epoch: {result.best_val_epoch}")
        print(f"Best val metric: {result.best_val_metric:.6f}")
    print(f"Checkpoint saved to: {result.checkpoint_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
