"""
Data loading utilities for ultrasound scans.

This module provides dataset classes for loading H5 scan files and
converting them to ScanData objects for use with the reconstruction pipeline.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
from torch.utils.data import Dataset as TorchDataset

from usrec_zoo.types import ScanData


class ScanDataset(TorchDataset):
    """
    PyTorch Dataset for loading ultrasound scans.

    This dataset loads scans from H5 files and returns ScanData objects.
    It supports subject-level indexing and frame sampling for training.

    Attributes:
        data_path: Path to the data directory.
        subjects: List of subject directory names.
        scans: List of scan filenames.
        indices_in_use: List of (subject_idx, scan_idx) tuples to use.
        num_samples: Number of frames to sample (-1 for all frames).
        sample_range: Range from which to sample frames.
        landmark_path: Optional path to landmark files.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        num_samples: int = 2,
        sample_range: Optional[int] = None,
        indices_in_use: Optional[List[Tuple[int, int]]] = None,
        landmark_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data_path: Path to the directory containing subject folders.
            num_samples: Number of frames to sample per scan. Use -1 to
                         load all frames in a scan.
            sample_range: Range from which frames are sampled. If None,
                          defaults to num_samples.
            indices_in_use: List of (subject_idx, scan_idx) tuples specifying
                            which scans to include. If None, uses all scans.
            landmark_path: Optional path to directory containing landmark files.

        Raises:
            ValueError: If num_samples < 2 and not -1.
            FileNotFoundError: If data_path doesn't exist.
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        # Get list of subjects (subdirectories)
        self.subjects = sorted([
            d.name for d in self.data_path.iterdir()
            if d.is_dir()
        ])

        if not self.subjects:
            raise ValueError(f"No subject directories found in {self.data_path}")

        # Get list of scans (H5 files in first subject directory)
        first_subject_path = self.data_path / self.subjects[0]
        self.scans = sorted([
            f.name for f in first_subject_path.iterdir()
            if f.suffix == ".h5"
        ])

        if not self.scans:
            raise ValueError(f"No H5 files found in {first_subject_path}")

        # Set up indices
        if indices_in_use is None:
            self.indices_in_use = [
                (i_sub, i_scn)
                for i_sub in range(len(self.subjects))
                for i_scn in range(len(self.scans))
            ]
        else:
            self.indices_in_use = sorted(indices_in_use)

        # Validate num_samples
        if num_samples < 2 and num_samples != -1:
            raise ValueError(
                "num_samples must be >= 2 or -1 (for all frames), "
                f"got {num_samples}"
            )
        self.num_samples = num_samples

        # Set sample_range
        if sample_range is None:
            self.sample_range = num_samples if num_samples > 0 else None
        else:
            self.sample_range = sample_range

        # Landmark path
        self.landmark_path = Path(landmark_path) if landmark_path else None

    def __len__(self) -> int:
        """Return number of scans in the dataset."""
        return len(self.indices_in_use)

    def __getitem__(self, idx: int) -> ScanData:
        """
        Load a scan by index.

        Args:
            idx: Index into indices_in_use.

        Returns:
            ScanData object containing frames and transforms.

        Raises:
            ValueError: If H5 data has unexpected format or types.
            IOError: If H5 file cannot be read.

        Note:
            When num_samples != -1, only a subset of frames are loaded.
            Landmarks are NOT adjusted - they still reference original frame
            indices. For training with sampled frames, landmarks should not
            be used (pass landmark_path=None).
        """
        subject_idx, scan_idx = self.indices_in_use[idx]
        subject_id = self.subjects[subject_idx]
        scan_name = self.scans[scan_idx][:-3]  # Remove .h5 extension

        # Load H5 file
        h5_path = self.data_path / subject_id / self.scans[scan_idx]
        try:
            with h5py.File(h5_path, "r") as h5file:
                if "frames" not in h5file:
                    raise ValueError(f"H5 file missing 'frames' dataset: {h5_path}")
                if "tforms" not in h5file:
                    raise ValueError(f"H5 file missing 'tforms' dataset: {h5_path}")
                frames = h5file["frames"][()]
                tforms = h5file["tforms"][()]
        except OSError as e:
            raise IOError(f"Failed to read H5 file {h5_path}: {e}") from e

        # Validate loaded data types and shapes
        if frames.ndim != 3:
            raise ValueError(
                f"Expected frames with 3 dimensions [N, H, W], got {frames.ndim}D "
                f"from {h5_path}"
            )
        if tforms.ndim != 3 or tforms.shape[1:] != (4, 4):
            raise ValueError(
                f"Expected tforms with shape [N, 4, 4], got {tforms.shape} "
                f"from {h5_path}"
            )
        if frames.shape[0] != tforms.shape[0]:
            raise ValueError(
                f"Mismatch: {frames.shape[0]} frames but {tforms.shape[0]} transforms "
                f"in {h5_path}"
            )

        # Ensure correct dtypes
        frames = frames.astype(np.uint8)
        tforms = tforms.astype(np.float32)

        # Sample frames if needed
        if self.num_samples == -1:
            # Use all frames
            pass
        else:
            frame_indices = self._sample_frames(len(frames))
            frames = frames[frame_indices]
            tforms = tforms[frame_indices]

        # Load landmarks if available
        # Note: landmarks are NOT adjusted for frame sampling
        landmarks = self._load_landmarks(subject_id, scan_name)

        return ScanData(
            frames=frames,
            transforms=tforms,
            subject_id=subject_id,
            scan_name=scan_name,
            landmarks=landmarks,
        )

    def _sample_frames(self, total_frames: int) -> List[int]:
        """
        Sample frame indices for training.

        Args:
            total_frames: Total number of frames in the scan.

        Returns:
            List of frame indices to use.

        Raises:
            ValueError: If total_frames is less than num_samples.
        """
        if total_frames < self.num_samples:
            raise ValueError(
                f"Scan has {total_frames} frames but num_samples={self.num_samples}. "
                f"Cannot sample more frames than available."
            )

        if self.sample_range is None:
            # Sample from all frames
            return sorted(random.sample(range(total_frames), self.num_samples))

        # Sample start position
        max_start = total_frames - self.sample_range
        if max_start < 0:
            max_start = 0

        start_idx = random.randint(0, max_start)
        end_idx = min(start_idx + self.sample_range, total_frames)

        # Sample within the range
        available_frames = end_idx - start_idx
        samples_to_take = min(self.num_samples, available_frames)
        indices = sorted(random.sample(range(start_idx, end_idx), samples_to_take))

        return indices

    def _load_landmarks(
        self,
        subject_id: str,
        scan_name: str,
    ) -> Optional[np.ndarray]:
        """
        Load landmarks for a scan if available.

        Args:
            subject_id: Subject identifier.
            scan_name: Scan name (without .h5 extension).

        Returns:
            Landmarks array of shape [L, 3] or None if not found.

        Raises:
            IOError: If landmark file exists but cannot be read.
            ValueError: If landmark data has unexpected shape.
        """
        if self.landmark_path is None:
            return None

        landmark_file = self.landmark_path / f"landmark_{subject_id}.h5"
        if not landmark_file.exists():
            return None

        try:
            with h5py.File(landmark_file, "r") as h5file:
                if scan_name not in h5file:
                    return None
                landmarks = h5file[scan_name][()]
                # Validate shape: expect [L, 3] where each row is (frame_idx, x, y)
                if landmarks.ndim != 2 or landmarks.shape[1] != 3:
                    raise ValueError(
                        f"Landmark data for {subject_id}/{scan_name} has unexpected "
                        f"shape {landmarks.shape}, expected [L, 3]"
                    )
                return landmarks.astype(np.int64)
        except OSError as e:
            raise IOError(
                f"Failed to read landmark file {landmark_file}: {e}"
            ) from e

    def get_subject_ids(self) -> List[str]:
        """Get list of subject IDs in the dataset."""
        subject_indices = set(idx[0] for idx in self.indices_in_use)
        return [self.subjects[i] for i in sorted(subject_indices)]

    def partition_by_ratio(
        self,
        ratios: List[float],
        randomize: bool = False,
        seed: int = 4,
    ) -> List["ScanDataset"]:
        """
        Partition the dataset into multiple subsets by subject.

        This performs a subject-level split to ensure all scans from
        a subject are in the same partition.

        Args:
            ratios: List of ratios for each partition (will be normalized).
            randomize: Whether to shuffle subjects before splitting.
            seed: Random seed for reproducibility.

        Returns:
            List of ScanDataset objects, one per partition.
        """
        num_sets = len(ratios)
        total_ratio = sum(ratios)
        ratios = [r / total_ratio for r in ratios]

        # Get unique subjects in this dataset
        subject_indices = sorted(set(idx[0] for idx in self.indices_in_use))
        num_subjects = len(subject_indices)

        # Compute partition sizes
        set_sizes = [int(num_subjects * r) for r in ratios]
        # Distribute remainder
        remainder = num_subjects - sum(set_sizes)
        for i in range(remainder):
            set_sizes[i] += 1

        # Optionally shuffle
        if randomize:
            random.Random(seed).shuffle(subject_indices)

        # Split subject indices
        partitions = []
        start = 0
        # LOOP INVARIANT: start is the index of the first subject for the current
        # partition. partitions contains completed ScanDataset objects for all
        # previous partitions. sum(set_sizes[:i]) == start at iteration i.
        for size in set_sizes:
            end = start + size
            partition_subjects = set(subject_indices[start:end])

            # Get indices for this partition
            partition_indices = [
                idx for idx in self.indices_in_use
                if idx[0] in partition_subjects
            ]

            partitions.append(ScanDataset(
                data_path=self.data_path,
                num_samples=self.num_samples,
                sample_range=self.sample_range,
                indices_in_use=partition_indices,
                landmark_path=self.landmark_path,
            ))

            start = end

        return partitions


def load_all_scans(
    data_path: Union[str, Path],
    landmark_path: Optional[Union[str, Path]] = None,
    subject_ids: Optional[List[str]] = None,
) -> List[ScanData]:
    """
    Load all scans from a directory as ScanData objects.

    This is a convenience function for loading complete scans
    (all frames) for evaluation.

    Args:
        data_path: Path to data directory.
        landmark_path: Optional path to landmarks.
        subject_ids: Optional list of subject IDs to load.

    Returns:
        List of ScanData objects.
    """
    dataset = ScanDataset(
        data_path=data_path,
        num_samples=-1,  # Load all frames
        landmark_path=landmark_path,
    )

    if subject_ids is not None:
        # Filter to specified subjects
        subject_set = set(subject_ids)
        valid_indices = [
            idx for idx in dataset.indices_in_use
            if dataset.subjects[idx[0]] in subject_set
        ]
        dataset.indices_in_use = valid_indices

    return [dataset[i] for i in range(len(dataset))]
