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

import csv
import json
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    RANDOM_SEED, create_directories, REPORTS_DIR,
    CHEST_SAMPLING_RATE, WINDOW_LENGTH_SEC, WINDOW_OVERLAP,
)
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
    per_fold_rows = []

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
        per_fold_rows.append({'model': model_name, 'test_subject': test_subject, **metrics})

        print(f"  {test_subject}: Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

    aggregated = aggregate_fold_metrics(fold_metrics)
    print_aggregated_summary(aggregated, f"{model_name} LOSO Summary")
    return aggregated, per_fold_rows


# ─────────────────────────────────────────────────────────────────────────────
# Persist what this run actually computed (Task 3: nothing hand-transcribed)
# ─────────────────────────────────────────────────────────────────────────────

def _save_baseline_csv(baseline_results: dict) -> None:
    path = REPORTS_DIR / "baseline_results.csv"
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'accuracy', 'precision', 'recall', 'f1'])
        for name, m in baseline_results.items():
            writer.writerow([name, m['accuracy'], m['precision'], m['recall'], m['f1']])
    print(f"  Saved -> {path}")


def _save_classical_csvs(aggregated_by_model: dict, per_fold_rows: list) -> None:
    loso_path = REPORTS_DIR / "classical_loso.csv"
    with open(loso_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'accuracy_mean', 'accuracy_std', 'precision_mean', 'precision_std',
                          'recall_mean', 'recall_std', 'f1_mean', 'f1_std', 'roc_auc_mean', 'roc_auc_std'])
        for name, agg in aggregated_by_model.items():
            writer.writerow([
                name,
                agg['accuracy']['mean'], agg['accuracy']['std'],
                agg['precision']['mean'], agg['precision']['std'],
                agg['recall']['mean'], agg['recall']['std'],
                agg['f1']['mean'], agg['f1']['std'],
                agg['roc_auc']['mean'], agg['roc_auc']['std'],
            ])
    print(f"  Saved -> {loso_path}")

    per_fold_path = REPORTS_DIR / "classical_per_fold.csv"
    with open(per_fold_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model', 'test_subject', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
        ])
        writer.writeheader()
        writer.writerows(per_fold_rows)
    print(f"  Saved -> {per_fold_path}")


def _save_dataset_stats(windowed: dict, features: np.ndarray, feature_names: list) -> None:
    per_subject = {
        sid: {
            'n_windows': wd.num_windows,
            'n_baseline': wd.num_baseline,
            'n_stress': wd.num_stress,
        }
        for sid, wd in windowed.items()
    }
    total_windows = sum(s['n_windows'] for s in per_subject.values())
    total_baseline = sum(s['n_baseline'] for s in per_subject.values())
    total_stress = sum(s['n_stress'] for s in per_subject.values())

    channels = sorted(next(iter(windowed.values())).windows.keys()) if windowed else []

    stats = {
        'total_windows': total_windows,
        'per_subject': per_subject,
        'class_balance': {
            'baseline': total_baseline,
            'stress': total_stress,
            'baseline_fraction': total_baseline / total_windows if total_windows else 0.0,
            'stress_fraction': total_stress / total_windows if total_windows else 0.0,
        },
        'feature_matrix_shape': list(features.shape),
        'n_features': len(feature_names),
        'channels': channels,
        'sampling_rate_hz': CHEST_SAMPLING_RATE,
        'window_length_sec': WINDOW_LENGTH_SEC,
        'window_overlap': WINDOW_OVERLAP,
    }

    path = REPORTS_DIR / "dataset_stats.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved -> {path}")


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
    logreg_results, logreg_per_fold = run_loso_evaluation(features, labels, subject_ids, "LogReg")
    rf_results, rf_per_fold = run_loso_evaluation(features, labels, subject_ids, "RandomForest")

    # ── Persist everything this run actually computed ──────────────────────────
    print_section_header("SAVING REPORTS")
    _save_baseline_csv(baseline_results)
    _save_classical_csvs(
        {'LogReg': logreg_results, 'RandomForest': rf_results},
        logreg_per_fold + rf_per_fold,
    )
    _save_dataset_stats(windowed, features, feature_names)

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
