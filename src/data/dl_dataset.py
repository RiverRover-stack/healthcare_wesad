"""
DL Dataset: PyTorch Dataset for Windowed Physiological Signals

Responsibility:
    Wrap WindowedData objects into a PyTorch Dataset for DL training.

Inputs:
    Dict[str, WindowedData] from the segmentation pipeline

Outputs:
    (tensor, label, subject_id) per window
    tensor shape: (C=6, T=3840) — 6 channels, 64 Hz × 60s
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from scipy.signal import resample

import sys
from pathlib import Path

# Ensure src/ is on the path however this module is invoked
_src_root = Path(__file__).parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from segmentation.window_data import WindowedData

# Signal channels in fixed order
SIGNAL_CHANNELS = ['chest_ecg', 'chest_eda', 'chest_emg', 'chest_resp', 'chest_temp', 'chest_acc']
TARGET_FS = 64       # Hz — downsample target
WINDOW_SEC = 60      # seconds
TARGET_LENGTH = TARGET_FS * WINDOW_SEC  # 3840 samples


class WESADDataset(Dataset):
    """PyTorch Dataset wrapping windowed WESAD signals."""

    def __init__(self, windowed_data: Dict[str, WindowedData], subject_ids: List[str] = None):
        """
        Args:
            windowed_data: Dict mapping subject_id -> WindowedData
            subject_ids: Subset of subjects to include (None = all)
        """
        self.samples: List[Tuple[np.ndarray, int, str]] = []

        subjects = subject_ids if subject_ids is not None else list(windowed_data.keys())

        for sid in subjects:
            wd = windowed_data[sid]
            if wd.num_windows == 0:
                continue

            n = wd.num_windows
            # Build (n, 6, T_original) array
            channels = []
            for ch in SIGNAL_CHANNELS:
                if ch in wd.windows:
                    channels.append(wd.windows[ch])  # shape (n, T_original)
                else:
                    # fill with zeros if a channel is missing
                    t = next(iter(wd.windows.values())).shape[1]
                    channels.append(np.zeros((n, t), dtype=np.float32))

            signal_array = np.stack(channels, axis=1)  # (n, 6, T_original)

            for i in range(n):
                window = signal_array[i]  # (6, T_original)

                # Replace NaN/Inf before resampling — FFT resampler spreads NaN everywhere
                window = np.nan_to_num(window, nan=0.0, posinf=5.0, neginf=-5.0)

                # Downsample each channel from 700 Hz -> 64 Hz
                resampled = resample(window, TARGET_LENGTH, axis=1)

                # Final sanitise after resampling (Gibbs ringing can produce large values)
                # Signals are z-scored so ±5 covers everything real; beyond that is artifact
                resampled = np.nan_to_num(resampled, nan=0.0, posinf=5.0, neginf=-5.0)
                resampled = np.clip(resampled, -5.0, 5.0).astype(np.float32)

                label = int(wd.labels[i])
                self.samples.append((resampled, label, sid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        window, label, subject_id = self.samples[idx]
        return torch.from_numpy(window), label, subject_id


def build_subject_arrays(windowed_data: Dict[str, WindowedData]) -> Tuple[np.ndarray, np.ndarray]:
    """Return parallel arrays of (subject_id_per_window, label_per_window) for LOSO splitting."""
    subject_ids_list = []
    labels_list = []
    for sid, wd in windowed_data.items():
        subject_ids_list.extend([sid] * wd.num_windows)
        labels_list.extend(wd.labels.tolist())
    return np.array(subject_ids_list), np.array(labels_list)
