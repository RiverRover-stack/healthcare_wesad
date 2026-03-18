"""
Splitting: Subject-Level Data Splitting Strategies

Responsibility:
    Implement LOSO and fixed subject-level train/test splits.

Inputs:
    Feature matrix, labels, subject IDs

Outputs:
    Train/test indices or generator for cross-validation

Assumptions:
    - All windows from a subject go to same split
    - No random window-level splitting allowed

Failure Modes:
    - Subject not found: Raises error
"""

import numpy as np
from typing import Generator, Tuple, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEBUG_TEST_SUBJECTS


def loso_split(subject_ids: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray, str], None, None]:
    """Leave-One-Subject-Out cross-validation generator."""
    unique_subjects = sorted(set(subject_ids))

    for test_subject in unique_subjects:
        train_mask = subject_ids != test_subject
        test_mask = subject_ids == test_subject
        yield train_mask, test_mask, test_subject


def fixed_subject_split(subject_ids: np.ndarray,
                        test_subjects: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed subject-level train/test split."""
    if test_subjects is None:
        test_subjects = DEBUG_TEST_SUBJECTS

    test_mask = np.isin(subject_ids, test_subjects)
    train_mask = ~test_mask
    return train_mask, test_mask


def get_split_info(subject_ids: np.ndarray, labels: np.ndarray,
                   train_mask: np.ndarray, test_mask: np.ndarray) -> dict:
    """Get information about a split."""
    return {
        'train_samples': int(np.sum(train_mask)),
        'test_samples': int(np.sum(test_mask)),
        'train_subjects': len(set(subject_ids[train_mask])),
        'test_subjects': len(set(subject_ids[test_mask])),
    }
