"""
Dataset splitting utilities.

This module provides canonical train/validation/test splits
for reproducible experiments.
"""

from dataclasses import dataclass
from typing import List, Tuple
import random


@dataclass
class SubjectSplit:
    """
    Container for dataset split subject IDs.

    Attributes:
        train: List of subject IDs for training.
        val: List of subject IDs for validation.
        test: List of subject IDs for testing.
    """
    train: List[str]
    val: List[str]
    test: List[str]

    def __post_init__(self) -> None:
        """Validate that splits are disjoint."""
        train_set = set(self.train)
        val_set = set(self.val)
        test_set = set(self.test)

        if train_set & val_set:
            raise ValueError(
                f"Train and val sets overlap: {train_set & val_set}"
            )
        if train_set & test_set:
            raise ValueError(
                f"Train and test sets overlap: {train_set & test_set}"
            )
        if val_set & test_set:
            raise ValueError(
                f"Val and test sets overlap: {val_set & test_set}"
            )


def get_canonical_splits(
    num_subjects: int = 50,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> SubjectSplit:
    """
    Get canonical train/val/test splits for reproducibility.

    This function generates deterministic subject splits based on
    the random seed. The default seed=42 provides the canonical
    splits for benchmarking.

    Args:
        num_subjects: Total number of subjects (default: 50 for TUS-REC).
        train_ratio: Fraction of subjects for training.
        val_ratio: Fraction of subjects for validation.
        test_ratio: Fraction of subjects for testing.
        seed: Random seed for reproducibility.

    Returns:
        SubjectSplit with train, val, and test subject ID lists.

    Raises:
        ValueError: If ratios don't sum to approximately 1.0.

    Example:
        >>> splits = get_canonical_splits()
        >>> len(splits.train), len(splits.val), len(splits.test)
        (30, 10, 10)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total_ratio}"
        )

    # Generate subject IDs (zero-padded 3-digit strings)
    subject_ids = [f"{i:03d}" for i in range(num_subjects)]

    # Shuffle with seed
    rng = random.Random(seed)
    shuffled = subject_ids.copy()
    rng.shuffle(shuffled)

    # Compute split sizes
    num_train = int(num_subjects * train_ratio)
    num_val = int(num_subjects * val_ratio)
    # Test gets the remainder
    num_test = num_subjects - num_train - num_val

    # Split
    train_subjects = sorted(shuffled[:num_train])
    val_subjects = sorted(shuffled[num_train:num_train + num_val])
    test_subjects = sorted(shuffled[num_train + num_val:])

    return SubjectSplit(
        train=train_subjects,
        val=val_subjects,
        test=test_subjects,
    )


def get_validation_subjects() -> List[str]:
    """
    Get the official validation subject IDs from TUS-REC challenge.

    The challenge provides validation data for subjects 050, 051, 052.

    Returns:
        List of validation subject IDs.
    """
    return ["050", "051", "052"]


def get_kfold_splits(
    num_subjects: int = 50,
    num_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """
    Generate k-fold cross-validation splits.

    Args:
        num_subjects: Total number of subjects.
        num_folds: Number of folds.
        seed: Random seed.

    Returns:
        List of (train_subjects, val_subjects) tuples for each fold.
    """
    subject_ids = [f"{i:03d}" for i in range(num_subjects)]

    rng = random.Random(seed)
    shuffled = subject_ids.copy()
    rng.shuffle(shuffled)

    # Distribute subjects into folds
    fold_size = num_subjects // num_folds
    remainder = num_subjects % num_folds

    folds = []
    start = 0
    # LOOP INVARIANT: start is the index into shuffled for the first subject
    # of fold i. folds contains i completed fold lists. First `remainder` folds
    # have size fold_size+1, remaining folds have size fold_size.
    for i in range(num_folds):
        size = fold_size + (1 if i < remainder else 0)
        folds.append(shuffled[start:start + size])
        start += size

    # Generate train/val splits for each fold
    splits = []
    for i in range(num_folds):
        val_subjects = sorted(folds[i])
        train_subjects = sorted([
            s for j, fold in enumerate(folds)
            for s in fold if j != i
        ])
        splits.append((train_subjects, val_subjects))

    return splits
