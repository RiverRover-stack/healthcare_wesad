"""
Metrics: Classification Metrics Computation

Responsibility:
    Compute classification metrics and generate evaluation reports.

Inputs:
    Predictions, true labels, probabilities

Outputs:
    Metrics dictionary, aggregated results

Assumptions:
    - Binary classification (0=baseline, 1=stress)

Failure Modes:
    - All same class: Some metrics undefined (return 0)
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import print_section_header


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> Dict[str, float]:
    """Compute all classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['roc_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0

    return metrics


def aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across cross-validation folds."""
    if not fold_metrics:
        return {}

    aggregated = {}
    for metric in fold_metrics[0].keys():
        values = [fm[metric] for fm in fold_metrics]
        aggregated[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }
    return aggregated


def print_aggregated_summary(agg: Dict, title: str = "LOSO Results") -> None:
    """Print aggregated cross-validation results."""
    print_section_header(title)
    print(f"  {'Metric':<12} {'Mean':>8} {'Std':>8}")
    print("  " + "-" * 30)
    for metric, vals in agg.items():
        print(f"  {metric:<12} {vals['mean']:>8.4f} {vals['std']:>8.4f}")
