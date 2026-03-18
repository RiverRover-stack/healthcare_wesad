"""
Main Pipeline: End-to-End Stress Anomaly Detection

Responsibility:
    Orchestrate the complete pipeline from data loading to evaluation.

Inputs:
    Command-line arguments (optional)

Outputs:
    Console output with results

Assumptions:
    - WESAD dataset is available at configured path

Failure Modes:
    - Missing data: Exits with error
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import RANDOM_SEED, create_directories
from utils import set_all_seeds, print_section_header
from data import load_all_subjects
from preprocessing import process_all_subjects
from segmentation import create_all_windows
from features import extract_all_features
from models import run_all_baselines, create_logistic_regression, create_random_forest, train_and_predict
from evaluation import loso_split, compute_metrics, aggregate_fold_metrics, print_aggregated_summary


def run_loso_evaluation(features: np.ndarray, labels: np.ndarray,
                        subject_ids: np.ndarray, model_name: str = "LogReg"):
    """Run LOSO cross-validation with specified model."""
    print_section_header(f"PHASE 5: LOSO EVALUATION - {model_name}")

    fold_metrics = []

    for train_mask, test_mask, test_subject in loso_split(subject_ids):
        X_train, y_train = features[train_mask], labels[train_mask]
        X_test, y_test = features[test_mask], labels[test_mask]

        if len(np.unique(y_test)) < 2:
            print(f"  {test_subject}: SKIP (single class)")
            continue

        model = create_logistic_regression() if model_name == "LogReg" else create_random_forest()
        preds, probs, _, _ = train_and_predict(model, X_train, y_train, X_test)
        metrics = compute_metrics(y_test, preds, probs)
        fold_metrics.append(metrics)

        print(f"  {test_subject}: Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

    aggregated = aggregate_fold_metrics(fold_metrics)
    print_aggregated_summary(aggregated, f"{model_name} LOSO Summary")
    return aggregated


def main():
    """Main pipeline execution."""
    print_section_header("BINARY STRESS ANOMALY DETECTION PIPELINE")

    set_all_seeds(RANDOM_SEED)
    create_directories()

    # Phase 2-4
    subjects = load_all_subjects()
    subjects = process_all_subjects(subjects)
    windowed = create_all_windows(subjects)
    features, labels, subject_ids, feature_names = extract_all_features(windowed)

    # Phase 4.5 & 5
    baseline_results = run_all_baselines(features, labels, subject_ids, feature_names)
    logreg_results = run_loso_evaluation(features, labels, subject_ids, "LogReg")
    rf_results = run_loso_evaluation(features, labels, subject_ids, "RandomForest")

    # Summary
    print_section_header("FINAL RESULTS SUMMARY")
    print(f"\n  BASELINES:")
    for name, metrics in baseline_results.items():
        print(f"    {name}: F1={metrics['f1']:.3f}, Recall={metrics['recall']:.3f}")

    print(f"\n  MODELS (LOSO Mean ± Std):")
    print(f"    LogReg: Recall={logreg_results['recall']['mean']:.3f}±{logreg_results['recall']['std']:.3f}")
    print(f"    RF:     Recall={rf_results['recall']['mean']:.3f}±{rf_results['recall']['std']:.3f}")
    print("\n  Pipeline completed successfully!")


if __name__ == "__main__":
    main()
