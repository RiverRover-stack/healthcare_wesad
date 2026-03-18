"""
Statistical Features: Basic Statistical Feature Extraction

Responsibility:
    Extract statistical features from a signal window.

Inputs:
    1D signal array (numpy)

Outputs:
    Dictionary of feature_name -> float value

Assumptions:
    - Signal is already preprocessed
    - NaN values are handled

Failure Modes:
    - All NaN signal: Returns zeros
"""

import numpy as np
from scipy import stats
import warnings


def extract_statistical_features(signal: np.ndarray) -> dict:
    """Extract statistical features from a signal window."""
    if np.any(np.isnan(signal)):
        signal = np.nan_to_num(signal, nan=np.nanmedian(signal))

    features = {}

    features['mean'] = float(np.mean(signal))
    features['std'] = float(np.std(signal))
    features['var'] = float(np.var(signal))
    features['min'] = float(np.min(signal))
    features['max'] = float(np.max(signal))
    features['range'] = float(np.ptp(signal))

    features['p10'] = float(np.percentile(signal, 10))
    features['p25'] = float(np.percentile(signal, 25))
    features['p50'] = float(np.percentile(signal, 50))
    features['p75'] = float(np.percentile(signal, 75))
    features['p90'] = float(np.percentile(signal, 90))
    features['iqr'] = features['p75'] - features['p25']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features['skewness'] = float(stats.skew(signal))
        features['kurtosis'] = float(stats.kurtosis(signal))

    features['rms'] = float(np.sqrt(np.mean(signal ** 2)))
    features['energy'] = float(np.sum(signal ** 2))

    for key, val in features.items():
        if not np.isfinite(val):
            features[key] = 0.0

    return features
