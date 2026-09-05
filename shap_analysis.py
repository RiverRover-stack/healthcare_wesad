"""
SHAP Explainability Analysis for WESAD Multi-Scale CNN Teacher and MicroCNN Student

Tasks:
    1. Channel-level SHAP       -- which physiological signal matters most,
                                   computed over all 15 LOSO folds (each fold's
                                   own checkpoints, its own held-out test subject)
    2. Grad-CAM temporal        -- where in the 60-second window the model focuses
                                   (applied to branch_small conv output before GAP;
                                   single representative fold)
    3. Teacher vs Student SHAP  -- side-by-side channel importance comparison,
                                   with per-fold Jensen-Shannon divergence and
                                   Spearman rank correlation between the two
                                   channel-importance distributions
    4. Per-class SHAP           -- stress vs baseline channel importance breakdown
                                   (single representative fold)

Usage:
    python shap_analysis.py

Respects WESAD_OUTPUT_DIR (via src/config.py) for both checkpoints (MODELS_DIR)
and outputs (REPORTS_DIR) -- it previously shadowed both off its own ROOT, so it
always read outputs/models/ regardless of which run's checkpoints were wanted.

Outputs (saved to REPORTS_DIR):
    shap_channel_importance.csv    -- fold, model, channel, mean_abs_shap, normalised_importance
    shap_divergence.csv            -- fold, test_subject, js_divergence, spearman_rho
    shap_channel_importance.png    -- regenerated from the CSV above, no re-run needed
    shap_teacher_vs_student.png    -- regenerated from the CSV above, no re-run needed
    shap_gradcam_temporal.png
    shap_per_class.png
"""

import csv
import sys
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless / Windows
import matplotlib.pyplot as plt
import torch
import shap
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT   = Path(__file__).parent
SRC    = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import MODELS_DIR, REPORTS_DIR
from data import load_all_subjects
from preprocessing import process_all_subjects
from segmentation import create_all_windows
from data.dl_dataset import WESADDataset
from models.teacher import MultiScaleTeacherCNN
from models.student  import MicroCNN

# ── constants ─────────────────────────────────────────────────────────────────
SIGNAL_NAMES    = ["ECG", "EDA", "EMG", "Resp", "Temp", "ACC"]
GRADCAM_SUBJECT = "S2"          # representative fold for Grad-CAM / per-class SHAP
N_BG            = 50            # background samples
N_TEST          = 50            # test samples
SAMPLE_RATE     = 64            # Hz
WINDOW_SEC      = 60            # seconds
DPI             = 300
SEED            = 42

# colour palette
C_TEACHER = "#4C72B0"   # blue
C_STUDENT = "#DD8452"   # orange
C_STRESS  = "#C44E52"   # red
C_BASE    = "#55A868"   # green

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":    100,
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    """
    Load a checkpoint that may be either:
        - a bare state_dict  (OrderedDict)
        - a dict with a 'state_dict' or 'model_state_dict' key
        - a full serialised model object
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model_state", "model"):
            if key in obj:
                model.load_state_dict(obj[key])
                return model
        model.load_state_dict(obj)
    elif isinstance(obj, torch.nn.Module):
        return obj
    else:
        raise ValueError(f"Unrecognised checkpoint format in {ckpt_path}")

    return model


def build_datasets(windowed, test_subject: str):
    """
    Build training (all subjects except test_subject) and
    test (test_subject only) datasets.
    """
    all_sids  = list(windowed.keys())
    train_ids = [s for s in all_sids if s != test_subject]
    test_ids  = [test_subject]

    train_ds = WESADDataset(windowed, subject_ids=train_ids)
    test_ds  = WESADDataset(windowed, subject_ids=test_ids)
    return train_ds, test_ds


def sample_tensors(dataset, n: int, seed: int = SEED):
    """
    Randomly sample n tensors and their labels from a WESADDataset.
    Returns:
        X  : float32 tensor  (n, 6, 3840)
        y  : int numpy array (n,)
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    tensors, labels = [], []
    for i in indices:
        x, label, _ = dataset[int(i)]
        tensors.append(x)
        labels.append(label)
    X = torch.stack(tensors, dim=0).float()   # (n, 6, 3840)
    y = np.array(labels, dtype=np.int32)
    return X, y


def _extract_stress_shap(shap_output, n_samples: int, n_channels: int = 6):
    """
    Robustly extract the stress-class SHAP values from whatever
    shap.GradientExplainer.shap_values() returns.

    SHAP 0.50 format (confirmed by debug):
        list of n arrays, each shaped (C, T, num_classes)
        → stack → (n, C, T, 2)  → select class 1 → (n, C, T)

    Also handles the classic format:
        list of num_classes arrays, each (n, C, T)
        → shap_output[1] → (n, C, T)

    Returns:
        sv_stress : ndarray of shape (n_samples, n_channels, T)
    """
    if isinstance(shap_output, list):
        first = np.array(shap_output[0])

        if first.ndim == 3 and first.shape[-1] == 2:
            stacked = np.stack(shap_output, axis=0)   # (n, C, T, 2)
            sv = stacked[..., 1]                       # (n, C, T)
        elif first.ndim == 2 and len(shap_output) == 2:
            sv = np.array(shap_output[1])[np.newaxis]  # (1, C, T)
        elif first.ndim == 3 and first.shape[0] == n_samples:
            sv = np.array(shap_output[1])              # (n, C, T)
        else:
            stacked = np.stack(shap_output, axis=0)
            if stacked.ndim == 4 and stacked.shape[-1] == 2:
                sv = stacked[..., 1]
            elif stacked.ndim == 4 and stacked.shape[0] == 2:
                sv = stacked[1]
            else:
                sv = stacked
    elif isinstance(shap_output, np.ndarray):
        if shap_output.ndim == 4 and shap_output.shape[-1] == 2:
            sv = shap_output[..., 1]
        elif shap_output.ndim == 4 and shap_output.shape[0] == 2:
            sv = shap_output[1]
        elif shap_output.ndim == 3:
            sv = shap_output
        else:
            sv = shap_output
    else:
        sv = np.array(shap_output)

    sv = np.nan_to_num(sv, nan=0.0, posinf=0.0, neginf=0.0)

    if sv.ndim == 3 and sv.shape[1] != n_channels and sv.shape[2] == n_channels:
        sv = sv.transpose(0, 2, 1)

    return sv


def compute_shap_channel(model: torch.nn.Module,
                          background: torch.Tensor,
                          test_x:    torch.Tensor):
    """
    Run shap.GradientExplainer and return:
        importance : ndarray (6,)  — mean |SHAP| per channel
        sv_stress  : ndarray (n, 6, 3840) — raw stress-class SHAP values
    """
    model.eval()
    explainer   = shap.GradientExplainer(model, background)
    shap_raw    = explainer.shap_values(test_x)

    sv_stress   = _extract_stress_shap(shap_raw, n_samples=len(test_x))
    importance  = np.mean(np.abs(sv_stress), axis=(0, 2))
    assert importance.shape == (6,), (
        f"Channel importance shape {importance.shape} != (6,). "
        f"sv_stress shape was {sv_stress.shape}"
    )
    return importance, sv_stress


def normalize_simplex(v: np.ndarray) -> np.ndarray:
    """Normalise a non-negative importance vector to sum to 1."""
    total = v.sum()
    return v / total if total > 0 else np.full_like(v, 1.0 / len(v))


def save_figure(fig: plt.Figure, name: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 1+3 — Teacher vs Student channel attribution over all 15 LOSO folds
# ─────────────────────────────────────────────────────────────────────────────

CHANNEL_IMPORTANCE_FIELDS = ['fold', 'test_subject', 'model', 'channel',
                             'mean_abs_shap', 'normalised_importance']
DIVERGENCE_FIELDS = ['fold', 'test_subject', 'js_divergence', 'spearman_rho']


def run_all_folds_channel_attribution(windowed) -> None:
    """
    For every LOSO fold: load that fold's teacher and student checkpoints,
    sample background from that fold's training subjects and test windows
    from its held-out subject, compute channel-level SHAP importance for
    both models, then the Jensen-Shannon divergence and Spearman rank
    correlation between their (simplex-normalised) importance vectors.

    Writes shap_channel_importance.csv and shap_divergence.csv immediately
    per fold, so a crash partway through leaves completed folds on disk.
    """
    all_subjects = sorted(windowed.keys())
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ci_path = REPORTS_DIR / "shap_channel_importance.csv"
    div_path = REPORTS_DIR / "shap_divergence.csv"

    with open(ci_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(CHANNEL_IMPORTANCE_FIELDS)
    with open(div_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(DIVERGENCE_FIELDS)

    n_completed = 0
    for fold_idx, test_subject in enumerate(all_subjects):
        teacher_ckpt = MODELS_DIR / f"teacher_loso_{test_subject}.pt"
        student_ckpt = MODELS_DIR / f"MicroCNN_distilled_loso_{test_subject}.pt"
        if not (teacher_ckpt.exists() and student_ckpt.exists()):
            print(f"  Fold {fold_idx+1:02d} [{test_subject}]: SKIP (missing checkpoint)")
            continue

        train_ds, test_ds = build_datasets(windowed, test_subject)
        bg_x, _ = sample_tensors(train_ds, N_BG, seed=SEED)
        test_x, test_y = sample_tensors(test_ds, N_TEST, seed=SEED + 1)

        teacher = MultiScaleTeacherCNN(in_channels=6, num_classes=2)
        teacher = load_checkpoint(teacher, teacher_ckpt)
        teacher.eval()

        student = MicroCNN(in_channels=6, num_classes=2)
        student = load_checkpoint(student, student_ckpt)
        student.eval()

        imp_teacher, _ = compute_shap_channel(teacher, bg_x, test_x)
        imp_student, _ = compute_shap_channel(student, bg_x, test_x)

        p = normalize_simplex(imp_teacher)
        q = normalize_simplex(imp_student)

        js_dist = jensenshannon(p, q, base=2)
        js_div = float(js_dist ** 2) if np.isfinite(js_dist) else float('nan')
        rho, _ = spearmanr(imp_teacher, imp_student)

        with open(ci_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for name, mean_abs, norm in zip(SIGNAL_NAMES, imp_teacher, p):
                writer.writerow([fold_idx, test_subject, 'teacher', name, mean_abs, norm])
            for name, mean_abs, norm in zip(SIGNAL_NAMES, imp_student, q):
                writer.writerow([fold_idx, test_subject, 'student', name, mean_abs, norm])

        with open(div_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([fold_idx, test_subject, js_div, rho])

        print(f"  Fold {fold_idx+1:02d} [{test_subject}]: JS divergence={js_div:.4f}  "
              f"Spearman rho={rho:.4f}")
        n_completed += 1

    if n_completed == 0:
        print("  ERROR: no folds had both checkpoints -- nothing computed.")
        return

    # Summary across folds
    js_vals, rho_vals = [], []
    with open(div_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            js_vals.append(float(row['js_divergence']))
            rho_vals.append(float(row['spearman_rho']))
    print(f"\n  Channel attribution summary ({n_completed} folds):")
    print(f"    JS divergence : {np.mean(js_vals):.4f} +/- {np.std(js_vals):.4f}")
    print(f"    Spearman rho  : {np.mean(rho_vals):.4f} +/- {np.std(rho_vals):.4f}")


def plot_channel_importance_from_csv() -> None:
    """Regenerate shap_channel_importance.png purely from the CSV -- no SHAP re-run."""
    path = REPORTS_DIR / "shap_channel_importance.csv"
    if not path.exists():
        print("  Skipping shap_channel_importance.png -- CSV not found")
        return

    # mean normalised importance per (model, channel) across folds
    sums, counts = {}, {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (row['model'], row['channel'])
            sums[key] = sums.get(key, 0.0) + float(row['normalised_importance'])
            counts[key] = counts.get(key, 0) + 1

    teacher_vals = [sums[('teacher', ch)] / counts[('teacher', ch)] for ch in SIGNAL_NAMES]
    ranked = sorted(zip(SIGNAL_NAMES, teacher_vals), key=lambda t: t[1], reverse=True)
    order = [x[0] for x in ranked]
    vals = [x[1] for x in ranked]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(order, vals, color=C_TEACHER, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_title("Teacher CNN -- Mean Normalised SHAP Importance per Channel\n"
                 "(stress class, mean across all LOSO folds)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Physiological Signal", fontsize=11)
    ax.set_ylabel("Mean normalised |SHAP| (simplex)", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    save_figure(fig, "shap_channel_importance.png")


def plot_teacher_vs_student_from_csv() -> None:
    """Regenerate shap_teacher_vs_student.png purely from the CSV -- no SHAP re-run."""
    path = REPORTS_DIR / "shap_channel_importance.csv"
    if not path.exists():
        print("  Skipping shap_teacher_vs_student.png -- CSV not found")
        return

    sums, counts = {}, {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (row['model'], row['channel'])
            sums[key] = sums.get(key, 0.0) + float(row['normalised_importance'])
            counts[key] = counts.get(key, 0) + 1

    imp_teacher = np.array([sums[('teacher', ch)] / counts[('teacher', ch)] for ch in SIGNAL_NAMES])
    imp_student = np.array([sums[('student', ch)] / counts[('student', ch)] for ch in SIGNAL_NAMES])

    x = np.arange(len(SIGNAL_NAMES))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_t = ax.bar(x - w/2, imp_teacher, width=w, color=C_TEACHER,
                    label="Teacher (MultiScaleTeacherCNN, ~266K params)",
                    edgecolor="white", linewidth=0.8)
    bars_s = ax.bar(x + w/2, imp_student, width=w, color=C_STUDENT,
                    label="Student (MicroCNN, ~5.3K params)",
                    edgecolor="white", linewidth=0.8)
    ax.bar_label(bars_t, fmt="%.4f", padding=3, fontsize=7, rotation=45)
    ax.bar_label(bars_s, fmt="%.4f", padding=3, fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(SIGNAL_NAMES, fontsize=11)
    # Fixed: the old title asserted the paper's conclusion ("Preserved
    # physiological attention after 50x compression") as a caption instead of
    # a finding -- state what the chart is, let the numbers make the claim.
    ax.set_title(
        "Teacher vs Student Channel Attribution (stress class)\n"
        "Mean normalised SHAP importance across all LOSO folds",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Physiological Signal Channel", fontsize=11)
    ax.set_ylabel("Mean normalised |SHAP value| (simplex)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(imp_teacher.max(), imp_student.max()) * 1.25)
    fig.tight_layout()
    save_figure(fig, "shap_teacher_vs_student.png")


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Grad-CAM on branch_small conv output (single representative fold)
# ─────────────────────────────────────────────────────────────────────────────

def task2_gradcam_temporal(teacher, test_x, test_y):
    """
    Grad-CAM applied to the output of the SECOND Conv1d in branch_small
    (layer index 3 in the Sequential), just before AdaptiveAvgPool1d.
    """
    print("\n[Task 2] Grad-CAM temporal analysis on branch_small ...")

    teacher.eval()
    activations_store: dict = {}
    gradients_store:   dict = {}

    target_layer = teacher.branch_small[3]   # second Conv1d

    def fwd_hook(module, inp, out):
        activations_store["act"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients_store["grad"] = grad_out[0]

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    n_samples      = min(len(test_x), N_TEST)
    time_len_ref   = None
    cam_stress     = []
    cam_baseline   = []

    for i in range(n_samples):
        teacher.zero_grad()
        x_i = test_x[i:i+1].clone().requires_grad_(True)
        logits = teacher(x_i)

        logits[0, 1].backward()

        act  = activations_store["act"].detach()
        grad = gradients_store["grad"].detach()

        weights  = grad.mean(dim=-1, keepdim=True)
        cam_map  = (weights * act).sum(dim=1)
        cam_map  = torch.clamp(cam_map, min=0)
        cam_map  = cam_map.squeeze(0).numpy()

        cam_min, cam_max = cam_map.min(), cam_map.max()
        if cam_max > cam_min:
            cam_map = (cam_map - cam_min) / (cam_max - cam_min)

        if time_len_ref is None:
            time_len_ref = len(cam_map)

        if test_y[i] == 1:
            cam_stress.append(cam_map)
        else:
            cam_baseline.append(cam_map)

    h_fwd.remove()
    h_bwd.remove()

    T = time_len_ref
    t_axis = np.linspace(0, WINDOW_SEC, T)

    mean_stress   = np.mean(np.stack(cam_stress),   axis=0) if cam_stress   else np.zeros(T)
    mean_baseline = np.mean(np.stack(cam_baseline), axis=0) if cam_baseline else np.zeros(T)

    print(f"  Stress windows used:   {len(cam_stress)}")
    print(f"  Baseline windows used: {len(cam_baseline)}")

    from numpy.lib.stride_tricks import sliding_window_view
    def smooth(arr, k=64):
        pad = k // 2
        arr_p = np.pad(arr, (pad, pad), mode="edge")
        return sliding_window_view(arr_p, k).mean(axis=-1)[:len(arr)]

    sm_stress   = smooth(mean_stress)
    sm_baseline = smooth(mean_baseline)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_axis, sm_stress,   color=C_STRESS, lw=1.8, label="Stress windows")
    ax.plot(t_axis, sm_baseline, color=C_BASE,   lw=1.8, label="Baseline windows",
            linestyle="--")
    ax.fill_between(t_axis, sm_stress, sm_baseline,
                    where=(sm_stress >= sm_baseline),
                    alpha=0.15, color=C_STRESS, label="Stress > Baseline")
    ax.set_title(
        "Grad-CAM Temporal Activation — Teacher branch_small (k=8, fast timescale)\n"
        "Mean normalised activation across stress vs baseline windows",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Normalised Grad-CAM activation", fontsize=11)
    ax.set_xlim(0, WINDOW_SEC)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    fig.tight_layout()
    save_figure(fig, "shap_gradcam_temporal.png")


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Per-class SHAP (stress vs baseline, single representative fold)
# ─────────────────────────────────────────────────────────────────────────────

def task4_per_class_shap(teacher, background, test_x, test_y):
    print("\n[Task 4] Per-class SHAP (stress vs baseline) ...")

    teacher.eval()
    explainer     = shap.GradientExplainer(teacher, background)
    shap_raw      = explainer.shap_values(test_x)
    sv_stress_raw = _extract_stress_shap(shap_raw, n_samples=len(test_x))

    stress_mask   = test_y == 1
    base_mask     = test_y == 0

    def chan_importance(sv, mask):
        if mask.sum() == 0:
            return np.zeros(6)
        subset = sv[mask]
        return np.mean(np.abs(subset), axis=(0, 2))

    imp_stress = chan_importance(sv_stress_raw, stress_mask)
    imp_base   = chan_importance(sv_stress_raw, base_mask)

    print("\n  Mean |SHAP| per channel — Stress vs Baseline windows (teacher, stress class):")
    print(f"  {'Channel':>6}  {'Stress':>10}  {'Baseline':>10}")
    for name, vs, vb in zip(SIGNAL_NAMES, imp_stress, imp_base):
        print(f"  {name:>6}  {vs:10.6f}  {vb:10.6f}")

    x = np.arange(len(SIGNAL_NAMES))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_s = ax.bar(x - w/2, imp_stress, width=w, color=C_STRESS,
                    label="Stress windows",
                    edgecolor="white", linewidth=0.8)
    bars_b = ax.bar(x + w/2, imp_base,   width=w, color=C_BASE,
                    label="Baseline windows",
                    edgecolor="white", linewidth=0.8)
    ax.bar_label(bars_s, fmt="%.5f", padding=3, fontsize=7, rotation=45)
    ax.bar_label(bars_b, fmt="%.5f", padding=3, fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(SIGNAL_NAMES, fontsize=11)
    ax.set_title(
        "Teacher CNN — Per-Class SHAP: Stress vs Baseline Windows\n"
        "Mean |SHAP| for stress-class output, split by true label",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Physiological Signal Channel", fontsize=11)
    ax.set_ylabel("Mean |SHAP value| (stress class output)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(imp_stress.max(), imp_base.max()) * 1.25)
    fig.tight_layout()
    save_figure(fig, "shap_per_class.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed(SEED)

    print("=" * 60)
    print("  WESAD SHAP Explainability Analysis")
    print("=" * 60)
    print(f"  MODELS_DIR : {MODELS_DIR}")
    print(f"  REPORTS_DIR: {REPORTS_DIR}")

    print("\n[Data] Loading subjects and running pipeline ...")
    subjects  = load_all_subjects()
    subjects  = process_all_subjects(subjects)
    windowed  = create_all_windows(subjects)

    # ── Task 1+3: channel attribution over all 15 folds ───────────────────────
    print("\n[Task 1+3] Teacher vs Student channel attribution over all LOSO folds ...")
    run_all_folds_channel_attribution(windowed)
    plot_channel_importance_from_csv()
    plot_teacher_vs_student_from_csv()

    # ── Task 2 & 4: single representative fold ────────────────────────────────
    train_ds, test_ds = build_datasets(windowed, GRADCAM_SUBJECT)
    bg_x, _ = sample_tensors(train_ds, N_BG, seed=SEED)
    test_x, test_y = sample_tensors(test_ds, N_TEST, seed=SEED + 1)

    teacher = MultiScaleTeacherCNN(in_channels=6, num_classes=2)
    teacher = load_checkpoint(teacher, MODELS_DIR / f"teacher_loso_{GRADCAM_SUBJECT}.pt")
    teacher.eval()

    task2_gradcam_temporal(teacher, test_x, test_y)
    task4_per_class_shap(teacher, bg_x, test_x, test_y)

    print("\n" + "=" * 60)
    print(f"  All figures saved to {REPORTS_DIR}")
    print("=" * 60)
