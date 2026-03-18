"""
EDA Features: Electrodermal Activity Specific Features

Responsibility:
    Extract EDA-specific features (tonic/phasic decomposition, SCR).

Inputs:
    EDA signal array, sampling frequency

Outputs:
    Dictionary of feature_name -> float value

Assumptions:
    - Signal is EDA (skin conductance)

Failure Modes:
    - No peaks detected: SCR features are zero
"""

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHEST_SAMPLING_RATE


def extract_eda_features(signal: np.ndarray,
                         fs: float = CHEST_SAMPLING_RATE) -> dict:
    """Extract EDA-specific features."""
    if np.any(np.isnan(signal)):
        signal = np.nan_to_num(signal, nan=np.nanmedian(signal))

    features = {}

    window_size = min(int(10 * fs), len(signal) // 2)
    window_size = max(window_size, 1)
    kernel = np.ones(window_size) / window_size
    tonic = np.convolve(signal, kernel, mode='same')
    phasic = signal - tonic

    features['eda_tonic_mean'] = float(np.mean(tonic))
    features['eda_tonic_std'] = float(np.std(tonic))
    features['eda_tonic_min'] = float(np.min(tonic))
    features['eda_tonic_max'] = float(np.max(tonic))
    features['eda_phasic_mean'] = float(np.mean(phasic))
    features['eda_phasic_std'] = float(np.std(phasic))
    features['eda_phasic_max'] = float(np.max(phasic))

    derivative = np.diff(phasic)
    peaks = []
    threshold = np.std(phasic) * 0.5

    for i in range(1, len(derivative)):
        if derivative[i - 1] > 0 and derivative[i] <= 0:
            if phasic[i] > threshold:
                peaks.append(i)

    features['eda_scr_count'] = float(len(peaks))
    features['eda_scr_rate'] = float(len(peaks) / (len(signal) / fs))

    if len(peaks) > 0:
        amplitudes = [phasic[p] for p in peaks]
        features['eda_scr_amplitude_mean'] = float(np.mean(amplitudes))
        features['eda_scr_amplitude_max'] = float(np.max(amplitudes))
    else:
        features['eda_scr_amplitude_mean'] = 0.0
        features['eda_scr_amplitude_max'] = 0.0

    for key, val in features.items():
        if not np.isfinite(val):
            features[key] = 0.0

    return features
