"""
LOSO Fold Splitting and Shared Training Helpers

Responsibility:
    Single owner of nested-LOSO fold construction (test subject + 2 inner
    validation subjects + remaining inner-train subjects) and the helpers
    that trainer.py and distillation.py both need. Previously duplicated
    byte-for-byte between the two trainers.

Inputs:
    subject_per_window / label_per_window: parallel arrays describing every
    window's subject and binary label (see WESADDataset.samples).

Outputs:
    FoldSplit per fold (or None if the fold must be skipped), plus
    sampler/class-weight/evaluation/logging helpers shared by both trainers.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from torch.utils.data import DataLoader, WeightedRandomSampler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import REPORTS_DIR, OUTPUT_DIR, RANDOM_SEED, DL_CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOKE_FOLDS = 2
SMOKE_EPOCHS = 3

PER_FOLD_FIELDS = [
    'model', 'mode', 'run_tag', 'fold_idx', 'test_subject', 'val_subjects',
    'n_train_windows', 'n_val_windows', 'n_test_windows',
    'n_train_baseline', 'n_train_stress', 'best_epoch', 'n_epochs', 'seed',
    'val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_roc_auc',
    'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc',
    'final_epoch_accuracy', 'final_epoch_precision', 'final_epoch_recall',
    'final_epoch_f1', 'final_epoch_roc_auc',
    'temperature', 'alpha', 'timestamp',
]


class FoldSplit(NamedTuple):
    test_subject: str
    val_subjects: List[str]
    train_idx: List[int]
    val_idx: List[int]
    test_idx: List[int]


def make_fold_split(fold_idx: int, test_subject: str, all_subjects: List[str],
                     subject_per_window: np.ndarray,
                     label_per_window: np.ndarray) -> Optional[FoldSplit]:
    """
    Build the (train, val, test) index split for one LOSO fold.

    val_subjects is a pair of subjects, rotated by fold_idx so every fold
    (mostly) sees a different validation pair:
        (train_subjects[i % n], train_subjects[(i+1) % n]),  i = fold_idx
    advanced past any pair that is single-class in this fold's window set.

    sorted() is lexicographic, so fold order is S10, S11, S13, ... S2, S3, ...
    -- that is today's behaviour and must not be "fixed", or the fold->val
    mapping shifts for every downstream checkpoint and CSV row.
    """
    test_idx = np.where(subject_per_window == test_subject)[0].tolist()
    if len(test_idx) == 0:
        print(f"  Fold {fold_idx+1:02d} [{test_subject}]: SKIP (no windows)")
        return None

    test_labels = label_per_window[test_idx]
    if len(np.unique(test_labels)) < 2:
        print(f"  Fold {fold_idx+1:02d} [{test_subject}]: SKIP (single-class test)")
        return None

    train_subjects = sorted(s for s in all_subjects if s != test_subject)
    n = len(train_subjects)

    val_subjects = None
    for advance in range(n):
        i = fold_idx + advance
        pair = (train_subjects[i % n], train_subjects[(i + 1) % n])
        pair_labels = label_per_window[np.isin(subject_per_window, pair)]
        if len(np.unique(pair_labels)) < 2:
            continue
        if advance > 0:
            print(f"  Fold {fold_idx+1:02d} [{test_subject}]: val pair advanced "
                  f"by {advance} (nominal pair was single-class)")
        val_subjects = list(pair)
        break

    if val_subjects is None:
        print(f"  Fold {fold_idx+1:02d} [{test_subject}]: SKIP (no valid val pair)")
        return None

    val_idx = np.where(np.isin(subject_per_window, val_subjects))[0].tolist()

    inner_train_subjects = [s for s in train_subjects if s not in val_subjects]
    train_idx = np.where(np.isin(subject_per_window, inner_train_subjects))[0].tolist()

    train_labels = label_per_window[train_idx]
    if len(np.unique(train_labels)) < 2:
        print(f"  Fold {fold_idx+1:02d} [{test_subject}]: SKIP (single-class inner-train)")
        return None

    n_windows = len(subject_per_window)
    partition = sorted(train_idx + val_idx + test_idx)
    assert partition == list(range(n_windows)), (
        f"Fold {fold_idx} [{test_subject}]: train/val/test indices are not "
        f"a partition of all {n_windows} windows"
    )

    return FoldSplit(test_subject=test_subject, val_subjects=val_subjects,
                      train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Create a sampler that draws equal numbers of each class per epoch.
    This is the primary fix for the class-collapse problem: instead of
    letting the majority-class windows dominate every batch, we oversample
    the minority class so each batch is ~50/50.
    """
    counts = np.bincount(labels, minlength=2).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    sample_weights = np.where(labels == 1, 1.0 / counts[1], 1.0 / counts[0])
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),  # pass as Python list; PyTorch stores as float64 internally
        num_samples=len(labels),
        replacement=True,
    )


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for the loss function (secondary defence)."""
    counts = np.bincount(labels, minlength=2).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def evaluate_model(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
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
    # Cast to plain Python floats: sklearn returns numpy scalars, and those
    # make the saved checkpoint unloadable under weights_only=True.
    return {k: float(v) for k, v in metrics.items()}


def loader_subject_ids(loader: DataLoader) -> set:
    """Subject ids actually present in a DataLoader wrapping a Subset(WESADDataset)."""
    subset = loader.dataset
    underlying = subset.dataset
    return {underlying.samples[i][2] for i in subset.indices}


def default_run_tag() -> str:
    """
    Fallback run tag derived from the output directory name (e.g. 'run_20260906_101500'
    or 'smoke_20260905_123712'), so no per_fold_results.csv row is ever left with a
    blank run_tag even outside the ablation sweep, which supplies its own (e.g. 'T4_a0.7').
    """
    return OUTPUT_DIR.name


def _fold_record_key(record: Dict) -> tuple:
    return (str(record.get('model', '')), str(record.get('mode', '')),
            str(record.get('run_tag', '')), str(record.get('fold_idx', '')))


def append_fold_record(row: Dict) -> None:
    """
    Upsert one fold's results into REPORTS_DIR/per_fold_results.csv, keyed on
    (model, mode, run_tag, fold_idx). Written immediately after each fold so a
    crash leaves completed folds on disk. Re-running the same fold (e.g. an
    invocation restarted into the same WESAD_OUTPUT_DIR) replaces its row
    instead of duplicating it -- teacher, students and every ablation config
    all share this one file across separate script invocations, so unlike
    ablation_results.csv it must never be wiped wholesale at process start.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "per_fold_results.csv"

    record = {k: row.get(k, '') for k in PER_FOLD_FIELDS}
    if not record.get('timestamp'):
        record['timestamp'] = datetime.now(timezone.utc).isoformat()
    key = _fold_record_key(record)

    existing_rows = []
    if path.exists():
        with open(path, newline='', encoding='utf-8') as f:
            for existing in csv.DictReader(f):
                if _fold_record_key(existing) != key:
                    existing_rows.append(existing)

    existing_rows.append(record)

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=PER_FOLD_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)


def _get_git_sha() -> str:
    try:
        import subprocess
        repo_root = Path(__file__).resolve().parent.parent.parent
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(repo_root), text=True,
        ).strip()
    except Exception:
        return 'unknown'


def write_manifest(phase: str, start_time: str, extra: Optional[Dict] = None) -> None:
    """
    Merge one phase's provenance into OUTPUT_DIR/manifest.json: git SHA,
    torch/device/CUDA info, seed (+ the per-fold seeding scheme), the full
    DL_CONFIG, and start/end timestamps. Keyed by phase, so teacher/students/
    ablation runs sharing one WESAD_OUTPUT_DIR accumulate into a single file
    rather than overwriting each other.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "manifest.json"
    manifest = {}
    if path.exists():
        try:
            with open(path, encoding='utf-8') as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    entry = {
        'phase': phase,
        'git_sha': _get_git_sha(),
        'torch_version': torch.__version__,
        'device': str(DEVICE),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'seed': RANDOM_SEED,
        'seed_scheme': 'RANDOM_SEED + fold_idx, set per fold (so --smoke reproduces '
                       'the first folds of a full run bit-for-bit)',
        'dl_config': DL_CONFIG,
        'start_time': start_time,
        'end_time': datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        entry.update(extra)
    manifest[phase] = entry

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Manifest updated -> {path}")
