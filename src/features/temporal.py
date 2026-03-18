"""
Temporal Features: Time-Domain Feature Extraction

Responsibility:
    Extract temporal features (derivatives, zero crossings, trends).

Inputs:
    1D signal array, sampling frequency

Outputs:
    Dictionary of feature_name -> float value

Assumptions:
    - Signal is preprocessed
    - Sampling frequency is known (default 700 Hz)

Failure Modes:
    - Very short signal: Returns 0 for undefined features
"""

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHEST_SAMPLING_RATE


def extract_temporal_features(signal: np.ndarray,
                              fs: float = CHEST_SAMPLING_RATE) -> dict:
    """Extract temporal features from a signal window."""
    if np.any(np.isnan(signal)):
        signal = np.nan_to_num(signal, nan=np.nanmedian(signal))

    features = {}

    derivative = np.diff(signal) * fs
    features['deriv_mean'] = float(np.mean(derivative))
    features['deriv_std'] = float(np.std(derivative))
    features['deriv_max'] = float(np.max(np.abs(derivative)))

    deriv2 = np.diff(derivative)
    features['deriv2_mean'] = float(np.mean(deriv2))
    features['deriv2_std'] = float(np.std(deriv2))

    zero_crossings = np.sum(np.abs(np.diff(np.sign(signal - np.mean(signal)))) > 0)
    features['zero_crossing_rate'] = float(zero_crossings / len(signal))
    features['mean_crossing_rate'] = float(zero_crossings / len(signal))

    x = np.arange(len(signal))
    slope, _ = np.polyfit(x, signal, 1)
    features['slope'] = float(slope)

    if len(signal) > 1:
        autocorr = np.corrcoef(signal[:-1], signal[1:])[0, 1]
        features['autocorr_lag1'] = float(autocorr) if np.isfinite(autocorr) else 0.0
    else:
        features['autocorr_lag1'] = 0.0

    for key, val in features.items():
        if not np.isfinite(val):
            features[key] = 0.0

    return features
