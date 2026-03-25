"""
LOSO DataLoader Factory

Responsibility:
    Given a pre-built WESADDataset and a held-out test subject, return
    balanced train and test DataLoaders for one fold of LOSO cross-validation.

    Encapsulates the split + sampler logic that would otherwise be duplicated
    in trainer.py and distillation.py.

Usage:
    from src.data.dl_dataloader import get_loso_dataloaders, build_subject_index

    full_dataset = WESADDataset(windowed_data)
    subject_ids, labels = build_subject_index(full_dataset)

    train_loader, test_loader, meta = get_loso_dataloaders(
        full_dataset, subject_ids, labels, test_subject='S5'
    )
    # meta: {'n_train': int, 'n_test': int, 'n_baseline': int, 'n_stress': int}
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from typing import Tuple, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DL_CONFIG


def build_subject_index(full_dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract parallel arrays from a WESADDataset for LOSO splitting.

    Returns:
        subject_ids:  (N,) str array — subject label per window
        labels:       (N,) int array — class label per window (0=baseline, 1=stress)
    """
    subject_ids = np.array([s   for _, _, s in full_dataset.samples])
    labels      = np.array([lbl for _, lbl, _ in full_dataset.samples])
    return subject_ids, labels


def _make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Inverse-frequency sampler that forces ~50/50 class balance in every batch."""
    counts = np.bincount(labels, minlength=2).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    sample_weights = np.where(labels == 1, 1.0 / counts[1], 1.0 / counts[0])
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(labels),
        replacement=True,
    )


def get_loso_dataloaders(
    full_dataset,
    subject_ids:  np.ndarray,
    labels:       np.ndarray,
    test_subject: str,
    batch_size:   int = None,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and test DataLoaders for one LOSO fold.

    Args:
        full_dataset:  WESADDataset containing all subjects.
        subject_ids:   Subject label per window (from build_subject_index).
        labels:        Class label per window   (from build_subject_index).
        test_subject:  Subject ID to hold out (e.g. 'S5').
        batch_size:    Override default from DL_CONFIG if provided.

    Returns:
        train_loader:  DataLoader with WeightedRandomSampler (balanced batches).
        test_loader:   DataLoader with sequential sampling (no shuffle).
        meta:          Dict with split statistics for logging.

    Raises:
        ValueError:  If the test split contains only one class (fold is
                     not usable for binary evaluation — caller should skip).
    """
    if batch_size is None:
        batch_size = DL_CONFIG['batch_size']

    train_idx = np.where(subject_ids != test_subject)[0].tolist()
    test_idx  = np.where(subject_ids == test_subject)[0].tolist()

    if len(test_idx) == 0:
        raise ValueError(f"No windows found for test subject '{test_subject}'.")

    test_labels  = labels[test_idx]
    train_labels = labels[train_idx]

    if len(np.unique(test_labels)) < 2:
        raise ValueError(
            f"Test subject '{test_subject}' has only one class "
            f"({np.unique(test_labels)}) — fold cannot be evaluated."
        )

    sampler = _make_balanced_sampler(train_labels)

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=batch_size,
        sampler=sampler,     # replaces shuffle=True; enforces class balance
        num_workers=0,
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    meta = {
        'n_train':    len(train_idx),
        'n_test':     len(test_idx),
        'n_baseline': int(np.sum(train_labels == 0)),
        'n_stress':   int(np.sum(train_labels == 1)),
        'train_labels': train_labels,   # full array — used for class weight calc
    }
    return train_loader, test_loader, meta
