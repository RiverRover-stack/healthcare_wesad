"""
Filters: Signal Filtering Functions

Responsibility:
    Provide reusable signal filtering functions using Butterworth filters.

Inputs:
    Raw signal arrays, filter parameters

Outputs:
    Filtered signal arrays

Assumptions:
    - Signals are 1D numpy arrays
    - Sampling frequency is known

Failure Modes:
    - Invalid filter type: ValueError
    - Frequencies outside Nyquist: Automatically clamped
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Optional


def butter_filter(signal: np.ndarray,
                  filter_type: str,
                  fs: float,
                  lowcut: Optional[float] = None,
                  highcut: Optional[float] = None,
                  cutoff: Optional[float] = None,
                  order: int = 4) -> np.ndarray:
    """Apply Butterworth filter to signal."""
    nyq = 0.5 * fs

    if filter_type == 'bandpass':
        low = max(0.001, min(lowcut / nyq, 0.999))
        high = max(low + 0.001, min(highcut / nyq, 0.999))
        b, a = butter(order, [low, high], btype='band')

    elif filter_type == 'lowpass':
        freq = max(0.001, min((cutoff or highcut) / nyq, 0.999))
        b, a = butter(order, freq, btype='low')

    elif filter_type == 'highpass':
        freq = max(0.001, min((cutoff or lowcut) / nyq, 0.999))
        b, a = butter(order, freq, btype='high')

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return filtfilt(b, a, signal)


def z_score_normalize(signal: np.ndarray, robust: bool = True) -> np.ndarray:
    """Apply z-score normalization."""
    if robust:
        center = np.median(signal)
        scale = np.median(np.abs(signal - center)) * 1.4826
    else:
        center = np.mean(signal)
        scale = np.std(signal)

    if scale < 1e-10:
        return signal - center

    return (signal - center) / scale
