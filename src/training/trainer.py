"""
Trainer: Nested LOSO Training Loop for Teacher CNN

Responsibility:
    Train the TeacherCNN using nested Leave-One-Subject-Out cross-validation:
    each fold holds out one test subject and two inner validation subjects.
    Model selection (best epoch, LR scheduling) is driven by the validation
    fold only -- the test fold is scored exactly once, after selection.
    Uses WeightedRandomSampler to force balanced batches (fixes class collapse).

Inputs:
    windowed_data: Dict[str, WindowedData] from the preprocessing pipeline

Outputs:
    Aggregated LOSO metrics dict (of the selected-checkpoint test scores)
    Per-fold model checkpoints in outputs/models/
    Per-fold rows in outputs/reports/per_fold_results.csv
    Comparison CSV in outputs/reports/model_comparison.csv
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Dict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RANDOM_SEED, MODELS_DIR, REPORTS_DIR, create_directories, DL_CONFIG
from data.dl_dataset import WESADDataset
from models.teacher import create_teacher_cnn
from segmentation.window_data import WindowedData
from utils import set_all_seeds
from training.loso import (
    DEVICE, SMOKE_FOLDS, SMOKE_EPOCHS,
    make_fold_split, make_balanced_sampler, compute_class_weights,
    evaluate_model, loader_subject_ids, append_fold_record, default_run_tag,
    write_manifest,
)
import csv
from datetime import datetime, timezone

# Training hyperparameters — sourced from config.DL_CONFIG (single source of truth)
BATCH_SIZE   = DL_CONFIG['batch_size']
EPOCHS       = DL_CONFIG['teacher_epochs']
LR           = DL_CONFIG['lr']
WEIGHT_DECAY = DL_CONFIG['weight_decay']


def train_teacher_loso(windowed_data: Dict[str, WindowedData], smoke: bool = False) -> Dict:
    """
    Run nested LOSO cross-validation training for TeacherCNN.
    Returns aggregated metrics (mean/std of the selected-checkpoint test scores).
    """
    create_directories()
    start_time = datetime.now(timezone.utc).isoformat()
    run_tag = default_run_tag()

    all_subjects = sorted(windowed_data.keys())
    n_epochs = SMOKE_EPOCHS if smoke else EPOCHS
    folds_to_run = all_subjects[:SMOKE_FOLDS] if smoke else all_subjects

    print(f"\n{'='*60}")
    print(f"  TEACHER CNN -- NESTED LOSO CROSS-VALIDATION{'  [SMOKE]' if smoke else ''}")
    print(f"  Device: {DEVICE}  |  Epochs: {n_epochs}  |  LR: {LR}")
    print(f"  Batch: {BATCH_SIZE}  |  WeightDecay: {WEIGHT_DECAY}")
    print(f"  Imbalance fix: WeightedRandomSampler + class-weighted loss")
    print(f"{'='*60}")

    print("  Building dataset (downsampling 700->64 Hz)...")
    full_dataset = WESADDataset(windowed_data)
    print(f"  Total windows: {len(full_dataset)}")

    subject_per_window = np.array([s for _, _, s in full_dataset.samples])
    label_per_window = np.array([lbl for _, lbl, _ in full_dataset.samples])

    fold_metrics = []

    for fold_idx, test_subject in enumerate(folds_to_run):
        split = make_fold_split(fold_idx, test_subject, all_subjects,
                                 subject_per_window, label_per_window)
        if split is None:
            continue

        seed = RANDOM_SEED + fold_idx
        set_all_seeds(seed)

        train_labels = label_per_window[split.train_idx]
        n_stress = int(np.sum(train_labels == 1))
        n_base = int(np.sum(train_labels == 0))

        sampler = make_balanced_sampler(train_labels)
        class_weights = compute_class_weights(train_labels)

        train_loader = DataLoader(
            Subset(full_dataset, split.train_idx),
            batch_size=BATCH_SIZE,
            sampler=sampler,   # replaces shuffle=True
            num_workers=0,
        )
        val_loader = DataLoader(
            Subset(full_dataset, split.val_idx),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )
        test_loader = DataLoader(
            Subset(full_dataset, split.test_idx),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )

        assert loader_subject_ids(val_loader).isdisjoint(loader_subject_ids(test_loader)), (
            f"Fold {fold_idx} [{test_subject}]: val/test subject leak"
        )

        model = create_teacher_cnn().to(DEVICE)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_f1 = -1.0
        best_state = None
        best_epoch = 0
        best_val_metrics = None

        print(f"  Fold {fold_idx+1:02d} [{test_subject}] val={split.val_subjects} "
              f"training ({n_epochs} epochs)...")
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for x, y, _ in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x), y)

                # Skip NaN loss batches (guard against any surviving NaN in data)
                if torch.isnan(loss):
                    continue

                loss.backward()
                # Clip gradients — prevents explosion on the rare noisy batch
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            val_metrics = evaluate_model(model, val_loader)
            scheduler.step(val_metrics['f1'])

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                best_val_metrics = val_metrics

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                lr_now = optimizer.param_groups[0]['lr']
                print(f"    ep {epoch+1:02d}/{n_epochs}  loss={avg_loss:.4f}"
                      f"  val_F1={val_metrics['f1']:.3f}"
                      f"  val_Recall={val_metrics['recall']:.3f}"
                      f"  lr={lr_now:.2e}")

        final_epoch_metrics = evaluate_model(model, test_loader)   # last-epoch weights FIRST
        model.load_state_dict(best_state)
        test_metrics = evaluate_model(model, test_loader)          # then the selected checkpoint
        fold_metrics.append(test_metrics)

        ckpt_path = MODELS_DIR / f"teacher_loso_{test_subject}.pt"
        torch.save({
            'model_state': best_state,
            'subject': test_subject,
            'mode': 'teacher',
            'val_subjects': split.val_subjects,
            'metrics': test_metrics,
            'val_metrics': best_val_metrics,
            'final_epoch_metrics': final_epoch_metrics,
            'best_epoch': best_epoch,
            'n_epochs': n_epochs,
            'seed': seed,
        }, ckpt_path)

        append_fold_record({
            'model': 'Teacher', 'mode': 'teacher', 'run_tag': run_tag,
            'fold_idx': fold_idx, 'test_subject': test_subject,
            'val_subjects': '|'.join(split.val_subjects),
            'n_train_windows': len(split.train_idx),
            'n_val_windows': len(split.val_idx),
            'n_test_windows': len(split.test_idx),
            'n_train_baseline': n_base, 'n_train_stress': n_stress,
            'best_epoch': best_epoch, 'n_epochs': n_epochs, 'seed': seed,
            'val_accuracy': best_val_metrics['accuracy'],
            'val_precision': best_val_metrics['precision'],
            'val_recall': best_val_metrics['recall'],
            'val_f1': best_val_metrics['f1'],
            'val_roc_auc': best_val_metrics['roc_auc'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'test_roc_auc': test_metrics['roc_auc'],
            'final_epoch_accuracy': final_epoch_metrics['accuracy'],
            'final_epoch_precision': final_epoch_metrics['precision'],
            'final_epoch_recall': final_epoch_metrics['recall'],
            'final_epoch_f1': final_epoch_metrics['f1'],
            'final_epoch_roc_auc': final_epoch_metrics['roc_auc'],
            'temperature': '', 'alpha': '',
        })

        print(f"  Fold {fold_idx+1:02d} [{test_subject}]"
              f" (train B={n_base}/S={n_stress})  best_ep={best_epoch}/{n_epochs}: "
              f"valF1={best_val_metrics['f1']:.3f} | testF1={test_metrics['f1']:.3f}  "
              f"Acc={test_metrics['accuracy']:.3f}  "
              f"Recall={test_metrics['recall']:.3f}  "
              f"AUC={test_metrics['roc_auc']:.3f}")

    if not fold_metrics:
        print("  ERROR: No folds completed.")
        return {}

    aggregated = {}
    for metric in fold_metrics[0].keys():
        vals = [fm[metric] for fm in fold_metrics]
        aggregated[metric] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    print(f"\n  Teacher CNN LOSO Summary (n={len(fold_metrics)} folds):")
    print(f"  {'Metric':<12} {'Mean':>8} {'Std':>8}")
    print("  " + "-" * 30)
    for metric, vals in aggregated.items():
        print(f"  {metric:<12} {vals['mean']:>8.4f} {vals['std']:>8.4f}")

    if not smoke:
        _save_comparison_csv(aggregated)

    write_manifest('teacher', start_time, extra={
        'smoke': smoke, 'n_epochs': n_epochs, 'n_folds_run': len(fold_metrics),
    })
    return aggregated


def _save_comparison_csv(cnn_results: Dict) -> None:
    """
    Update model comparison CSV with baseline/classical/teacher rows.
    Appends-and-dedupes by label instead of overwriting the file, so a
    teacher run after students no longer wipes the student rows that
    distillation.py's _append_to_comparison_csv already added.
    """
    csv_path = REPORTS_DIR / "model_comparison.csv"
    header = ["Model", "Params", "Accuracy", "Recall", "F1", "ROC-AUC"]

    new_rows = [
        ["Random Baseline",          "--",         "~0.50",          "0.358",             "0.364",             "--"],
        ["Majority Baseline",        "--",         "~0.67",          "0.000",             "0.000",             "--"],
        ["EDA Threshold",            "--",         "~0.60",          "0.866",             "0.792",             "--"],
        ["Logistic Regression",  "~150 feat.", "0.964 +/- 0.072", "0.954 +/- 0.134", "0.947 +/- 0.105", "0.976 +/- 0.062"],
        ["Random Forest",        "~150 feat.", "0.966 +/- 0.077", "0.965 +/- 0.081", "0.956 +/- 0.086", "0.996 +/- 0.011"],
        [
            "1D-CNN Teacher (Multi-Scale)",
            f"~{create_teacher_cnn().count_parameters() // 1000}K",
            f"{cnn_results['accuracy']['mean']:.3f} +/- {cnn_results['accuracy']['std']:.3f}",
            f"{cnn_results['recall']['mean']:.3f} +/- {cnn_results['recall']['std']:.3f}",
            f"{cnn_results['f1']['mean']:.3f} +/- {cnn_results['f1']['std']:.3f}",
            f"{cnn_results['roc_auc']['mean']:.3f} +/- {cnn_results['roc_auc']['std']:.3f}",
        ],
    ]
    new_labels = {row[0] for row in new_rows}

    existing_rows = []
    if csv_path.exists():
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = list(csv.reader(f))
        if reader:
            existing_rows = [row for row in reader[1:] if row and row[0] not in new_labels]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(existing_rows)
        writer.writerows(new_rows)

    print(f"\n  Comparison table saved -> {csv_path}")
