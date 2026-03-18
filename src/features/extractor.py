"""
Feature Extractor: Main Feature Extraction Pipeline

Responsibility:
    Orchestrate feature extraction from all signal windows.

Inputs:
    WindowedData dictionary (from segmentation module)

Outputs:
    Feature matrix, labels, subject IDs, feature names

Assumptions:
    - Windows are already created and valid

Failure Modes:
    - Empty windows: Returns empty arrays
"""

import numpy as np
from typing import Dict, List, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHEST_SAMPLING_RATE
from .statistical import extract_statistical_features
from .temporal import extract_temporal_features
from .frequency import extract_frequency_features
from .eda import extract_eda_features
from utils import print_section_header


def extract_features_from_window(window: np.ndarray,
                                 signal_name: str,
                                 fs: float = CHEST_SAMPLING_RATE) -> dict:
    """Extract all features from a single window."""
    features = {}
    prefix = signal_name

    for k, v in extract_statistical_features(window).items():
        features[f'{prefix}_{k}'] = v

    for k, v in extract_temporal_features(window, fs).items():
        features[f'{prefix}_{k}'] = v

    for k, v in extract_frequency_features(window, fs).items():
        features[f'{prefix}_{k}'] = v

    if 'eda' in signal_name.lower():
        for k, v in extract_eda_features(window, fs).items():
            features[f'{prefix}_{k}'] = v

    return features


def extract_all_features(windowed_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Extract features from all windows across all subjects."""
    print_section_header("PHASE 4: FEATURE ENGINEERING")

    all_features, all_labels, all_subject_ids = [], [], []
    feature_names = None

    for subject_id, wd in windowed_data.items():
        print(f"  {subject_id}...", end=" ")

        subject_features = []
        for window_idx in range(wd.num_windows):
            window_features = {}
            for signal_name, windows in wd.windows.items():
                feats = extract_features_from_window(windows[window_idx], signal_name)
                window_features.update(feats)
            subject_features.append(window_features)

        if feature_names is None and subject_features:
            feature_names = sorted(subject_features[0].keys())

        for wf in subject_features:
            all_features.append([wf.get(fn, 0.0) for fn in feature_names])

        all_labels.extend(wd.labels.tolist())
        all_subject_ids.extend([subject_id] * wd.num_windows)
        print(f"{wd.num_windows} windows, {len(feature_names)} features")

    feature_matrix = np.nan_to_num(np.array(all_features, dtype=np.float32))
    labels = np.array(all_labels, dtype=np.int32)
    subject_ids = np.array(all_subject_ids)

    print(f"\n[SUMMARY] Shape: {feature_matrix.shape}, Features: {len(feature_names)}")
    return feature_matrix, labels, subject_ids, feature_names
