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

Model selection uses nested LOSO: each fold holds out one test subject and
two inner validation subjects; the validation fold drives LR scheduling and
best-checkpoint selection, and the test fold is scored exactly once after
selection (see training.loso for the fold-splitting logic).

Usage:
    from src.models.distillation import train_student_kd_loso
    results = train_student_kd_loso(windowed_data, MicroCNN, 'MicroCNN', mode='distilled')
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Dict, Type
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RANDOM_SEED, MODELS_DIR, REPORTS_DIR, create_directories, DL_CONFIG
from data.dl_dataset import WESADDataset
from models.teacher import create_teacher_cnn
from segmentation.window_data import WindowedData
from utils import set_all_seeds
from training.loso import (
    DEVICE,
    make_fold_split, make_balanced_sampler, compute_class_weights,
    evaluate_model, loader_subject_ids, append_fold_record, default_run_tag,
)
import csv

# ── Hyperparameters — sourced from config.DL_CONFIG ──────────────────────────
BATCH_SIZE   = DL_CONFIG['batch_size']
EPOCHS       = DL_CONFIG['student_epochs']
LR           = DL_CONFIG['lr']
WEIGHT_DECAY = DL_CONFIG['weight_decay']


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

    def __init__(self, temperature: float, alpha: float,
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
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_student_kd_loso(
    windowed_data: Dict[str, WindowedData],
    student_class: Type[nn.Module],
    model_name: str,
    mode: str = 'distilled',
    *,
    dataset: WESADDataset = None,
    temperature: float = None,
    alpha: float = None,
    epochs: int = None,
    max_folds: int = None,
    run_tag: str = None,
) -> Dict:
    """
    Nested LOSO cross-validation for a student model.

    Args:
        windowed_data:  Output of the segmentation pipeline (same as teacher trainer).
        student_class:  The student class (e.g. MicroCNN), not an instance.
                        Will be instantiated fresh for each fold.
        model_name:     Short name for logging and CSV (e.g. 'MicroCNN').
        mode:           'distilled'   -- KD loss (teacher soft targets + hard labels)
                        'standalone'  -- CE loss only (for ablation: does KD actually help?)
        dataset:        Pre-built WESADDataset to reuse instead of rebuilding one from
                        windowed_data (lets callers share one dataset across many runs).
        temperature:    KD temperature override (default: DL_CONFIG['kd_temperature']).
        alpha:          KD alpha override (default: DL_CONFIG['kd_alpha']).
        epochs:         Epoch count override (default: DL_CONFIG['student_epochs']).
        max_folds:      Only run the first N folds (smoke tests / quick checks).
        run_tag:        Distinguishes checkpoint filenames for sweep configs
                        (e.g. 'T4_a0.7') and, when set, suppresses the
                        model_comparison.csv append -- ablation configs must
                        not pollute the headline model comparison table.

    Returns:
        Dict of aggregated metrics: {metric: {'mean': float, 'std': float}}

    Side effects:
        - Saves per-fold checkpoints to outputs/models/{model_name}_{mode}[_{run_tag}]_loso_{subject}.pt
        - Appends a row per fold to outputs/reports/per_fold_results.csv
        - Appends results row to outputs/reports/model_comparison.csv (unless run_tag is set)
    """
    if mode not in ('distilled', 'standalone'):
        raise ValueError(f"mode must be 'distilled' or 'standalone', got '{mode}'")

    temperature = temperature if temperature is not None else DL_CONFIG['kd_temperature']
    alpha       = alpha       if alpha       is not None else DL_CONFIG['kd_alpha']
    n_epochs    = epochs      if epochs      is not None else EPOCHS

    create_directories()

    all_subjects = sorted(windowed_data.keys())
    folds_to_run = all_subjects[:max_folds] if max_folds else all_subjects

    print(f"\n{'='*60}")
    print(f"  STUDENT: {model_name} [{mode.upper()}] -- NESTED LOSO"
          f"{f'  [run_tag={run_tag}]' if run_tag else ''}")
    print(f"  Device: {DEVICE}  |  Epochs: {n_epochs}  |  LR: {LR}")
    if mode == 'distilled':
        print(f"  KD: T={temperature}, alpha={alpha}")
    print(f"{'='*60}")

    if dataset is not None:
        full_dataset = dataset
    else:
        print("  Building dataset (reusing downsampled signals)...")
        full_dataset = WESADDataset(windowed_data)
    print(f"  Total windows: {len(full_dataset)}")

    subject_per_window = np.array([s   for _, _, s in full_dataset.samples])
    label_per_window   = np.array([lbl for _, lbl, _ in full_dataset.samples])

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
        n_base   = int(np.sum(train_labels == 0))

        sampler       = make_balanced_sampler(train_labels)
        class_weights = compute_class_weights(train_labels)

        train_loader = DataLoader(
            Subset(full_dataset, split.train_idx),
            batch_size=BATCH_SIZE, sampler=sampler, num_workers=0,
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
            # weights_only=False: these checkpoints carry val_subjects/metrics
            # dicts alongside the tensors. Our own artifact, so this is safe;
            # a future pass can move this to weights_only=True once every
            # checkpoint in circulation was written by the current format.
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
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
                temperature=temperature,
                alpha=alpha,
                class_weights=class_weights,
            )
            if fold_idx == 0:
                print(f"  [fold 0 guard] criterion.T={criterion.T}  criterion.alpha={criterion.alpha}")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_f1        = -1.0
        best_state     = None
        best_epoch     = 0
        best_val_metrics = None

        print(f"  Fold {fold_idx+1:02d} [{test_subject}] val={split.val_subjects}"
              f" (train B={n_base}/S={n_stress}) ...")

        for epoch in range(n_epochs):
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

            val_metrics = evaluate_model(student, val_loader)
            scheduler.step(val_metrics['f1'])

            if val_metrics['f1'] > best_f1:
                best_f1        = val_metrics['f1']
                best_state     = {k: v.cpu().clone() for k, v in student.state_dict().items()}
                best_epoch     = epoch + 1
                best_val_metrics = val_metrics

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                lr_now   = optimizer.param_groups[0]['lr']
                print(f"    ep {epoch+1:02d}/{n_epochs}  loss={avg_loss:.4f}"
                      f"  val_F1={val_metrics['f1']:.3f}"
                      f"  val_Recall={val_metrics['recall']:.3f}"
                      f"  lr={lr_now:.2e}")

        final_epoch_metrics = evaluate_model(student, test_loader)  # last-epoch weights FIRST
        student.load_state_dict(best_state)
        test_metrics = evaluate_model(student, test_loader)         # then the selected checkpoint
        fold_metrics.append(test_metrics)

        suffix = f"_{run_tag}" if run_tag else ""
        ckpt_out = MODELS_DIR / f"{model_name}_{mode}{suffix}_loso_{test_subject}.pt"
        torch.save({
            'model_state': best_state,
            'subject':     test_subject,
            'mode':        mode,
            'val_subjects': split.val_subjects,
            'metrics':      test_metrics,
            'val_metrics':  best_val_metrics,
            'final_epoch_metrics': final_epoch_metrics,
            'best_epoch':   best_epoch,
            'n_epochs':     n_epochs,
            'seed':         seed,
        }, ckpt_out)

        append_fold_record({
            'model': model_name, 'mode': mode, 'run_tag': run_tag or default_run_tag(),
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
            'temperature': temperature if mode == 'distilled' else '',
            'alpha': alpha if mode == 'distilled' else '',
        })

        print(f"  Fold {fold_idx+1:02d} [{test_subject}]  best_ep={best_epoch}/{n_epochs}:"
              f"  valF1={best_val_metrics['f1']:.3f} | testF1={test_metrics['f1']:.3f}"
              f"  Acc={test_metrics['accuracy']:.3f}"
              f"  Recall={test_metrics['recall']:.3f}"
              f"  AUC={test_metrics['roc_auc']:.3f}")

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

    if run_tag is None:
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
    header = ["Model", "Params", "Accuracy", "Recall", "F1", "ROC-AUC"]

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

    # Read existing rows; drop stale row for this label and any pre-existing
    # header (rewritten below) if present -- a file created by this function
    # alone (e.g. students trained before the teacher) must still end up
    # with a header, or anything parsing by column name misreads it.
    existing_rows = []
    if csv_path.exists():
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        if rows and rows[0] and rows[0][0].strip().lower() == 'model':
            rows = rows[1:]
        existing_rows = [row for row in rows if row and row[0] != label]

    existing_rows.append(new_row)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(existing_rows)

    print(f"\n  Results appended to {csv_path}")
