"""
Knowledge Distillation (KD) Pipeline

Trains lightweight student models using soft probability targets from a
frozen multi-scale teacher (response-based KD, Hinton et al. 2015).

KD Loss formula:
    L = alpha * T^2 * KL( log_softmax(s/T) || softmax(t/T) )
      + (1 - alpha) * CE(s, y)

    s      = student logits  (will be updated)
    t      = teacher logits  (frozen)
    T      = temperature     (>1 softens the distribution, transferring 'dark knowledge')
    alpha  = weight on soft targets (higher = trust teacher more)
    T^2    = scale correction so KD and CE losses are on the same magnitude

Two training modes (run both for the ablation comparison in the paper):
    'standalone' -- student trained directly on hard labels, no teacher
    'distilled'  -- student trained with KD loss (soft teacher targets)

Usage:
    from src.models.distillation import train_student_kd_loso
    results = train_student_kd_loso(windowed_data, MicroCNN, 'MicroCNN', mode='distilled')
"""

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from typing import Dict, Type
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RANDOM_SEED, MODELS_DIR, REPORTS_DIR, create_directories, DL_CONFIG
from data.dl_dataset import WESADDataset
from models.teacher import create_teacher_cnn
from segmentation.window_data import WindowedData

# ── Hyperparameters — sourced from config.DL_CONFIG ──────────────────────────
KD_TEMPERATURE = DL_CONFIG['kd_temperature']
KD_ALPHA       = DL_CONFIG['kd_alpha']
BATCH_SIZE     = DL_CONFIG['batch_size']
EPOCHS         = DL_CONFIG['student_epochs']
LR             = DL_CONFIG['lr']
WEIGHT_DECAY   = DL_CONFIG['weight_decay']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────

class KDLoss(nn.Module):
    """
    Response-based Knowledge Distillation loss.

    Combines:
        - Soft loss: KL divergence between student and teacher soft probabilities
          (temperature-scaled to transfer 'dark knowledge' about inter-class similarity)
        - Hard loss: standard Cross-Entropy on ground-truth labels
          (keeps the student grounded in the actual task)

    The T^2 scaling factor restores gradient magnitudes that are reduced by T
    (otherwise alpha effectively becomes much smaller than intended).
    """

    def __init__(self, temperature: float = KD_TEMPERATURE,
                 alpha: float = KD_ALPHA,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.class_weights = class_weights
        # batchmean: sum over classes, mean over batch -- correct KL normalisation
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_logits: (B, C)  raw logits from student (no softmax)
            teacher_logits: (B, C)  raw logits from teacher (no softmax, frozen)
            labels:         (B,)    integer ground-truth class indices
        Returns:
            Scalar loss value.
        """
        # Soft targets: KL( student || teacher )
        log_p_student = F.log_softmax(student_logits / self.T, dim=1)
        p_teacher     = F.softmax(teacher_logits    / self.T, dim=1)
        soft_loss = self.kl_div(log_p_student, p_teacher) * (self.T ** 2)

        # Hard targets: CE on ground-truth
        hard_loss = F.cross_entropy(student_logits, labels, weight=self.class_weights)

        return self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (mirrors trainer.py)
# ─────────────────────────────────────────────────────────────────────────────

def _make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=2).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    sample_weights = np.where(labels == 1, 1.0 / counts[1], 1.0 / counts[0])
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(labels),
        replacement=True,
    )


def _compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def _eval_fold(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['roc_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_student_kd_loso(
    windowed_data: Dict[str, WindowedData],
    student_class: Type[nn.Module],
    model_name: str,
    mode: str = 'distilled',
) -> Dict:
    """
    LOSO cross-validation for a student model.

    Args:
        windowed_data:  Output of the segmentation pipeline (same as teacher trainer).
        student_class:  The student class (e.g. MicroCNN), not an instance.
                        Will be instantiated fresh for each fold.
        model_name:     Short name for logging and CSV (e.g. 'MicroCNN').
        mode:           'distilled'   -- KD loss (teacher soft targets + hard labels)
                        'standalone'  -- CE loss only (for ablation: does KD actually help?)

    Returns:
        Dict of aggregated metrics: {metric: {'mean': float, 'std': float}}

    Side effects:
        - Saves per-fold checkpoints to outputs/models/{model_name}_{mode}_loso_{subject}.pt
        - Appends results row to outputs/reports/model_comparison.csv
    """
    if mode not in ('distilled', 'standalone'):
        raise ValueError(f"mode must be 'distilled' or 'standalone', got '{mode}'")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    create_directories()

    all_subjects = sorted(windowed_data.keys())

    print(f"\n{'='*60}")
    print(f"  STUDENT: {model_name} [{mode.upper()}] -- LOSO")
    print(f"  Device: {DEVICE}  |  Epochs: {EPOCHS}  |  LR: {LR}")
    if mode == 'distilled':
        print(f"  KD: T={KD_TEMPERATURE}, alpha={KD_ALPHA}")
    print(f"{'='*60}")

    print("  Building dataset (reusing downsampled signals)...")
    full_dataset = WESADDataset(windowed_data)
    print(f"  Total windows: {len(full_dataset)}")

    subject_per_window = np.array([s   for _, _, s in full_dataset.samples])
    label_per_window   = np.array([lbl for _, lbl, _ in full_dataset.samples])

    fold_metrics = []

    for fold_idx, test_subject in enumerate(all_subjects):
        train_indices = np.where(subject_per_window != test_subject)[0].tolist()
        test_indices  = np.where(subject_per_window == test_subject)[0].tolist()

        if len(test_indices) == 0:
            continue

        test_labels = label_per_window[test_indices]
        if len(np.unique(test_labels)) < 2:
            print(f"  Fold {fold_idx+1:02d} [{test_subject}]: SKIP (single class in test)")
            continue

        train_labels = label_per_window[train_indices]
        n_stress = int(np.sum(train_labels == 1))
        n_base   = int(np.sum(train_labels == 0))

        sampler       = _make_balanced_sampler(train_labels)
        class_weights = _compute_class_weights(train_labels)

        train_loader = DataLoader(
            Subset(full_dataset, train_indices),
            batch_size=BATCH_SIZE, sampler=sampler, num_workers=0,
        )
        test_loader = DataLoader(
            Subset(full_dataset, test_indices),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )

        # ── Load frozen teacher (distilled mode only) ─────────────────────────
        teacher = None
        if mode == 'distilled':
            ckpt_path = MODELS_DIR / f"teacher_loso_{test_subject}.pt"
            if not ckpt_path.exists():
                print(f"  Fold {fold_idx+1:02d} [{test_subject}]: SKIP"
                      f" -- teacher checkpoint not found at {ckpt_path}")
                print("  Run train_teacher.py first to generate teacher checkpoints.")
                continue
            teacher = create_teacher_cnn().to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            teacher.load_state_dict(ckpt['model_state'])
            teacher.freeze()  # sets eval() + requires_grad=False

        # ── Build student ──────────────────────────────────────────────────────
        student   = student_class().to(DEVICE)
        optimizer = torch.optim.Adam(
            student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
        )

        if mode == 'distilled':
            criterion = KDLoss(
                temperature=KD_TEMPERATURE,
                alpha=KD_ALPHA,
                class_weights=class_weights,
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_f1    = -1.0
        best_state = None

        print(f"  Fold {fold_idx+1:02d} [{test_subject}]"
              f" (train B={n_base}/S={n_stress}) ...")

        for epoch in range(EPOCHS):
            student.train()
            epoch_loss = 0.0
            n_batches  = 0

            for x, y, _ in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                student_logits = student(x)

                if mode == 'distilled':
                    with torch.no_grad():
                        teacher_logits = teacher(x)
                    loss = criterion(student_logits, teacher_logits, y)
                else:
                    loss = criterion(student_logits, y)

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            val_metrics = _eval_fold(student, test_loader)
            scheduler.step(val_metrics['f1'])

            if val_metrics['f1'] > best_f1:
                best_f1    = val_metrics['f1']
                best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                lr_now   = optimizer.param_groups[0]['lr']
                print(f"    ep {epoch+1:02d}/{EPOCHS}  loss={avg_loss:.4f}"
                      f"  F1={val_metrics['f1']:.3f}"
                      f"  Recall={val_metrics['recall']:.3f}"
                      f"  lr={lr_now:.2e}")

        student.load_state_dict(best_state)
        final_metrics = _eval_fold(student, test_loader)
        fold_metrics.append(final_metrics)

        ckpt_out = MODELS_DIR / f"{model_name}_{mode}_loso_{test_subject}.pt"
        torch.save({
            'model_state': best_state,
            'subject':     test_subject,
            'mode':        mode,
            'metrics':     final_metrics,
        }, ckpt_out)

        print(f"  Fold {fold_idx+1:02d} [{test_subject}]:"
              f"  Acc={final_metrics['accuracy']:.3f}"
              f"  Recall={final_metrics['recall']:.3f}"
              f"  F1={final_metrics['f1']:.3f}"
              f"  AUC={final_metrics['roc_auc']:.3f}")

    if not fold_metrics:
        print("  ERROR: No folds completed.")
        return {}

    aggregated = {}
    for metric in fold_metrics[0].keys():
        vals = [fm[metric] for fm in fold_metrics]
        aggregated[metric] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    n = len(fold_metrics)
    print(f"\n  {model_name} [{mode}] LOSO Summary (n={n} folds):")
    print(f"  {'Metric':<12} {'Mean':>8} {'Std':>8}")
    print("  " + "-" * 30)
    for metric, vals in aggregated.items():
        print(f"  {metric:<12} {vals['mean']:>8.4f} {vals['std']:>8.4f}")

    _append_to_comparison_csv(model_name, mode, student_class, aggregated)
    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _append_to_comparison_csv(
    model_name: str,
    mode: str,
    student_class: Type[nn.Module],
    results: Dict,
) -> None:
    """
    Append one row for a trained student to the existing comparison CSV.
    Reads current rows, removes any previous row with the same label,
    then appends the fresh result.
    """
    csv_path = REPORTS_DIR / "model_comparison.csv"

    label = f"{model_name} ({mode})"

    # Count student parameters
    try:
        n_params = student_class().count_parameters()
        params_str = f"~{n_params // 1000}K" if n_params >= 1000 else f"~{n_params}"
    except Exception:
        params_str = "--"

    new_row = [
        label,
        params_str,
        f"{results['accuracy']['mean']:.3f} +/- {results['accuracy']['std']:.3f}",
        f"{results['recall']['mean']:.3f} +/- {results['recall']['std']:.3f}",
        f"{results['f1']['mean']:.3f} +/- {results['f1']['std']:.3f}",
        f"{results['roc_auc']['mean']:.3f} +/- {results['roc_auc']['std']:.3f}",
    ]

    # Read existing rows; drop stale row for this label if present
    existing_rows = []
    if csv_path.exists():
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            for row in csv.reader(f):
                if row and row[0] != label:
                    existing_rows.append(row)

    existing_rows.append(new_row)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(existing_rows)

    print(f"\n  Results appended to {csv_path}")
