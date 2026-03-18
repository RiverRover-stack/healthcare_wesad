"""
Baselines: Heuristic Baseline Detectors

Responsibility:
    Implement non-ML baseline detectors for reference performance.

Inputs:
    Feature matrix, labels, subject IDs

Outputs:
    Baseline predictions and metrics

Assumptions:
    - Majority class is baseline (label 0)

Failure Modes:
    - No EDA features: EDA threshold returns zeros
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RANDOM_SEED
from utils import print_section_header


def random_baseline(labels: np.ndarray, seed: int = RANDOM_SEED) -> np.ndarray:
    """Predict based on class prior probability."""
    np.random.seed(seed)
    class_probs = np.bincount(labels) / len(labels)
    return np.random.choice(len(class_probs), size=len(labels), p=class_probs)


def majority_baseline(labels: np.ndarray) -> np.ndarray:
    """Always predict the majority class."""
    return np.full(len(labels), np.argmax(np.bincount(labels)))


def eda_threshold_baseline(features: np.ndarray, feature_names: List[str],
                           labels: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    """Threshold-based detector using EDA mean."""
    eda_idx = next((i for i, n in enumerate(feature_names) if 'chest_eda_mean' in n), None)

    if eda_idx is None:
        return np.zeros(len(labels), dtype=np.int32)

    eda_values = features[:, eda_idx]
    predictions = np.zeros(len(labels), dtype=np.int32)

    for subject in np.unique(subject_ids):
        mask = subject_ids == subject
        subject_eda = eda_values[mask]
        subject_labels = labels[mask]

        baseline_mask = subject_labels == 0
        if np.sum(baseline_mask) > 0:
            threshold = np.mean(subject_eda[baseline_mask]) + np.std(subject_eda[baseline_mask])
        else:
            threshold = np.median(subject_eda)

        predictions[mask] = (subject_eda > threshold).astype(np.int32)

    return predictions


def compute_baseline_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute basic metrics for baseline predictions."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def run_all_baselines(features: np.ndarray, labels: np.ndarray,
                      subject_ids: np.ndarray, feature_names: List[str]) -> Dict[str, Dict]:
    """Run all baseline detectors and compute metrics."""
    print_section_header("PHASE 4.5: HEURISTIC BASELINES")

    results = {}

    preds = random_baseline(labels)
    results['random'] = compute_baseline_metrics(labels, preds)
    print(f"  Random: Acc={results['random']['accuracy']:.3f}")

    preds = majority_baseline(labels)
    results['majority'] = compute_baseline_metrics(labels, preds)
    print(f"  Majority: Acc={results['majority']['accuracy']:.3f}")

    preds = eda_threshold_baseline(features, feature_names, labels, subject_ids)
    results['eda_threshold'] = compute_baseline_metrics(labels, preds)
    print(f"  EDA Threshold: Recall={results['eda_threshold']['recall']:.3f}")

    return results
