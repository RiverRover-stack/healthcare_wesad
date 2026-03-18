"""
Subject Data: Data Container for WESAD Subject

Responsibility:
    Define the SubjectData dataclass that holds all data for one subject.

Inputs:
    Raw signals, labels during construction

Outputs:
    SubjectData object with organized signals and metadata

Assumptions:
    - Chest signals at 700 Hz
    - Labels aligned with chest sampling rate

Failure Modes:
    - Empty arrays: Statistics will be zero
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHEST_SAMPLING_RATE


@dataclass
class SubjectData:
    """Container for a single subject's data after loading and curation."""

    subject_id: str

    # Chest signals (all at 700 Hz)
    chest_ecg: np.ndarray = field(default_factory=lambda: np.array([]))
    chest_eda: np.ndarray = field(default_factory=lambda: np.array([]))
    chest_emg: np.ndarray = field(default_factory=lambda: np.array([]))
    chest_resp: np.ndarray = field(default_factory=lambda: np.array([]))
    chest_temp: np.ndarray = field(default_factory=lambda: np.array([]))
    chest_acc: np.ndarray = field(default_factory=lambda: np.array([]))

    # Labels (at chest sampling rate)
    labels_raw: np.ndarray = field(default_factory=lambda: np.array([]))
    labels_binary: np.ndarray = field(default_factory=lambda: np.array([]))
    valid_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    # Processed signals (added by signal_processor)
    processed_signals: Dict[str, np.ndarray] = field(default_factory=dict)

    # Computed statistics
    num_samples: int = 0
    num_valid_samples: int = 0
    duration_seconds: float = 0.0
    baseline_samples: int = 0
    stress_samples: int = 0

    def __post_init__(self):
        """Calculate derived statistics after initialization."""
        self._compute_statistics()

    def _compute_statistics(self):
        """Compute statistics from loaded data."""
        if len(self.labels_raw) > 0:
            self.num_samples = len(self.labels_raw)
            self.duration_seconds = self.num_samples / CHEST_SAMPLING_RATE

        if len(self.valid_mask) > 0:
            self.num_valid_samples = int(np.sum(self.valid_mask))

        if len(self.labels_binary) > 0 and len(self.valid_mask) > 0:
            valid_labels = self.labels_binary[self.valid_mask]
            self.baseline_samples = int(np.sum(valid_labels == 0))
            self.stress_samples = int(np.sum(valid_labels == 1))
