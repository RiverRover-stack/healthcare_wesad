"""
Frequency Features: Spectral Feature Extraction

Responsibility:
    Extract frequency-domain features using FFT and Welch's method.

Inputs:
    1D signal array, sampling frequency

Outputs:
    Dictionary of feature_name -> float value

Assumptions:
    - Signal is preprocessed
    - Sufficient length for spectral analysis

Failure Modes:
    - Very short signal: Uses smaller nperseg
"""

import numpy as np
from scipy.signal import welch
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHEST_SAMPLING_RATE


def extract_frequency_features(signal: np.ndarray,
                               fs: float = CHEST_SAMPLING_RATE) -> dict:
    """Extract frequency domain features."""
    if np.any(np.isnan(signal)):
        signal = np.nan_to_num(signal, nan=np.nanmedian(signal))

    features = {}
    n = len(signal)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        freq, psd = welch(signal, fs=fs, nperseg=min(256, n // 2))

    total_power = np.sum(psd)
    features['total_power'] = float(total_power)

    bands = [('power_vlf', 0, 0.04), ('power_lf', 0.04, 0.15),
             ('power_hf', 0.15, 0.4), ('power_vhf', 0.4, 2.0)]

    for name, low, high in bands:
        mask = (freq >= low) & (freq < high)
        band_power = np.sum(psd[mask]) if np.any(mask) else 0.0
        features[name] = float(band_power)
        features[f'{name}_rel'] = float(band_power / max(total_power, 1e-10))

    if len(psd) > 0 and total_power > 0:
        features['peak_freq'] = float(freq[np.argmax(psd)])
        features['spectral_centroid'] = float(np.sum(freq * psd) / total_power)
    else:
        features['peak_freq'] = 0.0
        features['spectral_centroid'] = 0.0

    psd_norm = psd / (np.sum(psd) + 1e-10)
    psd_pos = psd_norm[psd_norm > 0]
    features['spectral_entropy'] = float(-np.sum(psd_pos * np.log2(psd_pos + 1e-10))) if len(psd_pos) > 0 else 0.0

    for key, val in features.items():
        if not np.isfinite(val):
            features[key] = 0.0

    return features
