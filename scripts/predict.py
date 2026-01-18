#!/usr/bin/env python3
"""
Prediction script for ultrasound reconstruction algorithms.

Loads a trained model and generates DDF predictions for validation/test data.

Usage:
    python scripts/predict.py --checkpoint experiments/tusrec_baseline/best_model.pt

    # Generate DDFs for challenge submission:
    python scripts/predict.py --checkpoint model.pt --output_dir submission/
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from usrec_zoo.algorithms import get_algorithm, get_algorithm_class
from usrec_zoo.calibration import CalibrationData, CalibrationDataTorch
from usrec_zoo.data import ScanDataset, load_all_scans
from usrec_zoo.evaluation import transforms_to_ddf


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate DDF predictions from a trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="tusrec_baseline",
        help="Algorithm name (if not saved in checkpoint)",
    )

    # Data paths
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/validation/frames",
        help="Path to data directory (validation or test)",
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
        default="data/validation/landmark",
        help="Path to landmarks directory",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Directory to save DDF predictions",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )

    return parser.parse_args()


def save_ddf(ddf, output_path: Path, scan_name: str) -> None:
    """Save DDF outputs to numpy files."""
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / f"{scan_name}_GP.npy", ddf.global_pixels)
    np.save(output_path / f"{scan_name}_GL.npy", ddf.global_landmarks)
    np.save(output_path / f"{scan_name}_LP.npy", ddf.local_pixels)
    np.save(output_path / f"{scan_name}_LL.npy", ddf.local_landmarks)


def main() -> int:
    """Main prediction function."""
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return 1

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load calibration
    calib_path = Path(args.calib_path)
    if not calib_path.exists():
        alt_path = Path("TUS-REC2025-Challenge_baseline") / args.calib_path
        if alt_path.exists():
            calib_path = alt_path
        else:
            print(f"ERROR: Calibration file not found: {args.calib_path}")
            return 1

    print(f"Loading calibration from: {calib_path}")
    calibration = CalibrationData.from_csv(calib_path)
    calib_torch = CalibrationDataTorch(calibration, device)

    # Create algorithm and load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")

    # First, load checkpoint to get config if available
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", None)
    if config is None:
        # Use default config
        algo_class = get_algorithm_class(args.algorithm)
        config = algo_class.get_default_config()

    algorithm = get_algorithm(args.algorithm, config=config, calibration=calibration)
    algorithm._load_state_dict(checkpoint)
    algorithm.model.to(device)
    algorithm.model.eval()

    print(f"Algorithm: {algorithm.get_name()}")

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
            print(f"WARNING: Landmark path not found: {args.landmark_path}")
            landmark_path = None

    print(f"Loading data from: {data_path}")

    # Load all scans (with all frames)
    scans = load_all_scans(
        data_path=data_path,
        landmark_path=landmark_path,
    )

    print(f"Loaded {len(scans)} scans")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each scan
    print("\nGenerating predictions...")
    for i, scan in enumerate(scans):
        print(f"  [{i + 1}/{len(scans)}] {scan.subject_id}/{scan.scan_name}")

        # Get prediction
        with torch.no_grad():
            prediction = algorithm.predict(scan, device)

        # Convert to DDF
        if scan.landmarks is not None:
            ddf = transforms_to_ddf(prediction, scan.landmarks, calib_torch)

            # Save
            scan_id = f"{scan.subject_id}_{scan.scan_name}"
            save_ddf(ddf, output_dir, scan_id)

            print(f"    Saved DDF: {scan_id}")
        else:
            print(f"    WARNING: No landmarks for {scan.subject_id}/{scan.scan_name}")

    print(f"\nPredictions saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
