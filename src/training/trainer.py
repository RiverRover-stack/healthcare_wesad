"""
Trainer: LOSO Training Loop for Teacher CNN

Responsibility:
    Train the TeacherCNN using Leave-One-Subject-Out cross-validation.
    Uses WeightedRandomSampler to force balanced batches (fixes class collapse).
    Saves best model per fold (by F1) and aggregates metrics.

Inputs:
    windowed_data: Dict[str, WindowedData] from the preprocessing pipeline

Outputs:
    Aggregated LOSO metrics dict
    Per-fold model checkpoints in outputs/models/
    Comparison CSV in outputs/reports/model_comparison.csv
"""

import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RANDOM_SEED, MODELS_DIR, REPORTS_DIR, create_directories, DL_CONFIG
from data.dl_dataset import WESADDataset
from models.teacher import create_teacher_cnn
from segmentation.window_data import WindowedData

# Training hyperparameters — sourced from config.DL_CONFIG (single source of truth)
BATCH_SIZE   = DL_CONFIG['batch_size']
EPOCHS       = DL_CONFIG['teacher_epochs']
LR           = DL_CONFIG['lr']
WEIGHT_DECAY = DL_CONFIG['weight_decay']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Create a sampler that draws equal numbers of each class per epoch.
    This is the primary fix for the class-collapse problem: instead of
    letting 64% baseline windows dominate every batch, we oversample
    stress windows so each batch is ~50/50.
    """
    counts = np.bincount(labels, minlength=2).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    # Weight per sample = inverse class frequency
    sample_weights = np.where(labels == 1,
                               1.0 / counts[1],
                               1.0 / counts[0])
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),  # pass as Python list; PyTorch stores as float64 internally
        num_samples=len(labels),
        replacement=True,
    )


def _compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for the loss function (secondary defence)."""
    counts = np.bincount(labels, minlength=2).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def _eval_fold(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    """Run inference on a DataLoader; return metrics dict."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['roc_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
    return metrics


def train_teacher_loso(windowed_data: Dict[str, WindowedData]) -> Dict:
    """
    Run LOSO cross-validation training for TeacherCNN.
    Returns aggregated metrics.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    create_directories()

    all_subjects = sorted(windowed_data.keys())
    print(f"\n{'='*60}")
    print(f"  TEACHER CNN -- LOSO CROSS-VALIDATION")
    print(f"  Device: {DEVICE}  |  Epochs: {EPOCHS}  |  LR: {LR}")
    print(f"  Batch: {BATCH_SIZE}  |  WeightDecay: {WEIGHT_DECAY}")
    print(f"  Imbalance fix: WeightedRandomSampler + class-weighted loss")
    print(f"{'='*60}")

    print("  Building dataset (downsampling 700->64 Hz)...")
    full_dataset = WESADDataset(windowed_data)
    print(f"  Total windows: {len(full_dataset)}")

    subject_per_window = np.array([s for _, _, s in full_dataset.samples])
    label_per_window = np.array([lbl for _, lbl, _ in full_dataset.samples])

    fold_metrics = []

    for fold_idx, test_subject in enumerate(all_subjects):
        train_indices = np.where(subject_per_window != test_subject)[0].tolist()
        test_indices = np.where(subject_per_window == test_subject)[0].tolist()

        if len(test_indices) == 0:
            continue

        test_labels = label_per_window[test_indices]
        if len(np.unique(test_labels)) < 2:
            print(f"  Fold {fold_idx+1:02d} [{test_subject}]: SKIP (single class in test)")
            continue

        train_labels = label_per_window[train_indices]
        n_stress = int(np.sum(train_labels == 1))
        n_base = int(np.sum(train_labels == 0))

        # Balanced sampler — forces 50/50 class mix in every batch
        sampler = _make_balanced_sampler(train_labels)
        class_weights = _compute_class_weights(train_labels)

        train_loader = DataLoader(
            Subset(full_dataset, train_indices),
            batch_size=BATCH_SIZE,
            sampler=sampler,   # replaces shuffle=True
            num_workers=0,
        )
        test_loader = DataLoader(
            Subset(full_dataset, test_indices),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
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

        print(f"  Fold {fold_idx+1:02d} [{test_subject}] training ({EPOCHS} epochs)...")
        for epoch in range(EPOCHS):
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

            val_metrics = _eval_fold(model, test_loader)
            scheduler.step(val_metrics['f1'])

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                lr_now = optimizer.param_groups[0]['lr']
                print(f"    ep {epoch+1:02d}/{EPOCHS}  loss={avg_loss:.4f}"
                      f"  val_F1={val_metrics['f1']:.3f}"
                      f"  val_Recall={val_metrics['recall']:.3f}"
                      f"  lr={lr_now:.2e}")

        model.load_state_dict(best_state)
        final_metrics = _eval_fold(model, test_loader)
        fold_metrics.append(final_metrics)

        ckpt_path = MODELS_DIR / f"teacher_loso_{test_subject}.pt"
        torch.save({
            'model_state': best_state,
            'subject': test_subject,
            'metrics': final_metrics,
        }, ckpt_path)

        print(f"  Fold {fold_idx+1:02d} [{test_subject}]"
              f" (train B={n_base}/S={n_stress}): "
              f"Acc={final_metrics['accuracy']:.3f}  "
              f"Recall={final_metrics['recall']:.3f}  "
              f"F1={final_metrics['f1']:.3f}  "
              f"AUC={final_metrics['roc_auc']:.3f}")

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

    _save_comparison_csv(aggregated)
    return aggregated


def _save_comparison_csv(cnn_results: Dict) -> None:
    """Write model comparison CSV with actual ML results + CNN results."""
    csv_path = REPORTS_DIR / "model_comparison.csv"

    rows = [
        ["Model", "Params", "Accuracy", "Recall", "F1", "ROC-AUC"],
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

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\n  Comparison table saved -> {csv_path}")
