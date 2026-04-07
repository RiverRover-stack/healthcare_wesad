"""
SHAP Explainability Analysis for WESAD Multi-Scale CNN Teacher and MicroCNN Student

Tasks:
    1. Channel-level SHAP       -- which physiological signal matters most
    2. Grad-CAM temporal        -- where in the 60-second window the model focuses
                                   (applied to branch_small conv output before GAP)
    3. Teacher vs Student SHAP  -- side-by-side channel importance comparison
    4. Per-class SHAP           -- stress vs baseline channel importance breakdown

Usage:
    python shap_analysis.py

Outputs (saved to outputs/reports/):
    shap_channel_importance.png
    shap_gradcam_temporal.png
    shap_teacher_vs_student.png
    shap_per_class.png
"""

import sys
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless / Windows
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import shap

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT   = Path(__file__).parent
SRC    = ROOT / "src"
sys.path.insert(0, str(SRC))

from data import load_all_subjects
from preprocessing import process_all_subjects
from segmentation import create_all_windows
from data.dl_dataset import WESADDataset
from models.teacher import MultiScaleTeacherCNN
from models.student  import MicroCNN

# ── constants ─────────────────────────────────────────────────────────────────
SIGNAL_NAMES    = ["ECG", "EDA", "EMG", "Resp", "Temp", "ACC"]
LOSO_SUBJECT    = "S2"          # fold used for SHAP analysis
N_BG            = 50            # background samples
N_TEST          = 50            # test samples
SAMPLE_RATE     = 64            # Hz
WINDOW_SEC      = 60            # seconds
REPORTS_DIR     = ROOT / "outputs" / "reports"
MODELS_DIR      = ROOT / "outputs" / "models"
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

    obj = torch.load(ckpt_path, map_location="cpu")

    if isinstance(obj, dict):
        # try common key names
        for key in ("state_dict", "model_state_dict", "model_state", "model"):
            if key in obj:
                model.load_state_dict(obj[key])
                return model
        # assume the dict itself is the state_dict
        model.load_state_dict(obj)
    elif isinstance(obj, torch.nn.Module):
        return obj
    else:
        raise ValueError(f"Unrecognised checkpoint format in {ckpt_path}")

    return model


def build_datasets(windowed):
    """
    Build training (all subjects except LOSO_SUBJECT) and
    test (LOSO_SUBJECT only) datasets.
    """
    all_sids  = list(windowed.keys())
    train_ids = [s for s in all_sids if s != LOSO_SUBJECT]
    test_ids  = [LOSO_SUBJECT]

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
            # SHAP 0.50: list of n arrays, each (C, T, 2)
            # Stack → (n, C, T, 2), take stress class (index 1) on last axis
            stacked = np.stack(shap_output, axis=0)   # (n, C, T, 2)
            sv = stacked[..., 1]                       # (n, C, T)

        elif first.ndim == 2 and len(shap_output) == 2:
            # Classic: list of 2 arrays each (C, T) — single sample, 2 classes
            sv = np.array(shap_output[1])[np.newaxis]  # (1, C, T)

        elif first.ndim == 3 and first.shape[0] == n_samples:
            # Classic: list of 2 arrays each (n, C, T) — 2 classes
            sv = np.array(shap_output[1])              # (n, C, T)

        else:
            # Fallback: try stacking and see if last dim is 2
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

    # Ensure (n, C, T) — transpose if axes are swapped
    if sv.ndim == 3 and sv.shape[1] != n_channels and sv.shape[2] == n_channels:
        sv = sv.transpose(0, 2, 1)

    print(f"    [debug] stress SHAP shape after extraction: {sv.shape}")
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
    # mean |SHAP| over samples (axis 0) and time (axis 2) → (6,)
    importance  = np.mean(np.abs(sv_stress), axis=(0, 2))
    assert importance.shape == (6,), (
        f"Channel importance shape {importance.shape} != (6,). "
        f"sv_stress shape was {sv_stress.shape}"
    )
    return importance, sv_stress


def save_figure(fig: plt.Figure, name: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Channel-level SHAP (teacher)
# ─────────────────────────────────────────────────────────────────────────────

def task1_channel_shap(teacher, background, test_x):
    print("\n[Task 1] Channel-level SHAP (GradientExplainer) ...")
    importance, shap_values = compute_shap_channel(teacher, background, test_x)

    # ranked printout for the paper
    ranked = sorted(zip(SIGNAL_NAMES, importance), key=lambda t: t[1], reverse=True)
    print("\n  Mean |SHAP| per channel (stress class, teacher):")
    for rank, (name, val) in enumerate(ranked, 1):
        print(f"    {rank}. {name:>5}: {val:.6f}")

    # bar chart
    order   = [x[0] for x in ranked]
    vals    = [x[1] for x in ranked]
    colours = [C_TEACHER] * 6

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(order, vals, color=colours, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
    ax.set_title("Teacher CNN — Mean |SHAP| per Signal Channel (Stress Class)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Physiological Signal", fontsize=11)
    ax.set_ylabel("Mean |SHAP value|", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    save_figure(fig, "shap_channel_importance.png")
    return shap_values   # pass to task 4


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 (REPLACEMENT) — Grad-CAM on branch_small conv output
# ─────────────────────────────────────────────────────────────────────────────

def task2_gradcam_temporal(teacher, test_x, test_y):
    """
    Grad-CAM applied to the output of the SECOND Conv1d in branch_small
    (layer index 3 in the Sequential), just before AdaptiveAvgPool1d.

    This gives genuine temporal localisation: a heatmap of shape (B, 64, T')
    where T' ≈ 3840 (same-length padding was used).

    We visualise the mean Grad-CAM activation across stress and baseline windows.
    """
    print("\n[Task 2] Grad-CAM temporal analysis on branch_small ...")

    teacher.eval()
    activations_store: dict = {}
    gradients_store:   dict = {}

    # ── register hooks on branch_small[3] (second Conv1d, before GAP) ────────
    # branch_small is nn.Sequential: [Conv, BN, ReLU, Conv, BN, ReLU, GAP]
    #                                  0    1    2    3    4    5     6
    target_layer = teacher.branch_small[3]   # second Conv1d

    def fwd_hook(module, inp, out):
        activations_store["act"] = out   # (B, 64, T')

    def bwd_hook(module, grad_in, grad_out):
        gradients_store["grad"] = grad_out[0]  # (B, 64, T')

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    n_samples      = min(len(test_x), N_TEST)
    time_len_ref   = None
    cam_stress     = []
    cam_baseline   = []

    for i in range(n_samples):
        teacher.zero_grad()
        x_i = test_x[i:i+1].clone().requires_grad_(True)   # (1, 6, 3840)
        logits = teacher(x_i)                               # (1, 2)

        # grad w.r.t. stress class (index 1)
        logits[0, 1].backward()

        act  = activations_store["act"].detach()    # (1, 64, T')
        grad = gradients_store["grad"].detach()     # (1, 64, T')

        # Grad-CAM: weight channels by global-average-pooled gradient
        weights  = grad.mean(dim=-1, keepdim=True)  # (1, 64, 1)
        cam_map  = (weights * act).sum(dim=1)        # (1, T')
        cam_map  = torch.clamp(cam_map, min=0)       # ReLU
        cam_map  = cam_map.squeeze(0).numpy()        # (T',)

        # min-max normalise per window
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

    # smooth for readability
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
# Task 3 — Teacher vs Student SHAP comparison
# ─────────────────────────────────────────────────────────────────────────────

def task3_teacher_vs_student(teacher, student, background_t, background_s,
                              test_x, test_x_s):
    print("\n[Task 3] Teacher vs Student SHAP comparison ...")

    # teacher importance (already computed but re-run for independence)
    imp_teacher, _ = compute_shap_channel(teacher, background_t, test_x)
    imp_student, _ = compute_shap_channel(student, background_s, test_x_s)

    print("\n  Mean |SHAP| per channel — Teacher vs Student (stress class):")
    print(f"  {'Channel':>6}  {'Teacher':>10}  {'Student':>10}")
    for name, vt, vs in zip(SIGNAL_NAMES, imp_teacher, imp_student):
        print(f"  {name:>6}  {vt:10.6f}  {vs:10.6f}")

    x   = np.arange(len(SIGNAL_NAMES))
    w   = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_t = ax.bar(x - w/2, imp_teacher, width=w, color=C_TEACHER,
                    label="Teacher (MultiScaleTeacherCNN, ~266K params)",
                    edgecolor="white", linewidth=0.8)
    bars_s = ax.bar(x + w/2, imp_student, width=w, color=C_STUDENT,
                    label="Student (MicroCNN, ~5.3K params)",
                    edgecolor="white", linewidth=0.8)
    ax.bar_label(bars_t, fmt="%.5f", padding=3, fontsize=7, rotation=45)
    ax.bar_label(bars_s, fmt="%.5f", padding=3, fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(SIGNAL_NAMES, fontsize=11)
    ax.set_title(
        "Knowledge Distillation — SHAP Channel Importance: Teacher vs Student\n"
        "Preserved physiological attention after 50× compression",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Physiological Signal Channel", fontsize=11)
    ax.set_ylabel("Mean |SHAP value| (stress class)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(imp_teacher.max(), imp_student.max()) * 1.25)
    fig.tight_layout()
    save_figure(fig, "shap_teacher_vs_student.png")


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Per-class SHAP (stress vs baseline)
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
        subset = sv[mask]        # (k, 6, 3840)
        return np.mean(np.abs(subset), axis=(0, 2))   # (6,)

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

    # ── 1. Load data pipeline ─────────────────────────────────────────────────
    print("=" * 60)
    print("  WESAD SHAP Explainability Analysis")
    print("=" * 60)

    print("\n[Data] Loading subjects and running pipeline ...")
    subjects  = load_all_subjects()
    subjects  = process_all_subjects(subjects)
    windowed  = create_all_windows(subjects)
    train_ds, test_ds = build_datasets(windowed)

    print(f"  Training set : {len(train_ds):,} windows")
    print(f"  Test set (S2): {len(test_ds):,} windows")

    # ── 2. Sample background + test batches ───────────────────────────────────
    bg_x, _     = sample_tensors(train_ds, N_BG,   seed=SEED)
    test_x, test_y = sample_tensors(test_ds, N_TEST, seed=SEED + 1)

    print(f"\n  background shape : {tuple(bg_x.shape)}")
    print(f"  test shape       : {tuple(test_x.shape)}")
    print(f"  stress in test   : {test_y.sum()} / {len(test_y)}")

    # ── 3. Load teacher ───────────────────────────────────────────────────────
    print(f"\n[Models] Loading teacher checkpoint (LOSO {LOSO_SUBJECT}) ...")
    teacher = MultiScaleTeacherCNN(in_channels=6, num_classes=2)
    teacher = load_checkpoint(teacher, MODELS_DIR / f"teacher_loso_{LOSO_SUBJECT}.pt")
    teacher.eval()
    print(f"  Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")

    # ── 4. Load student ───────────────────────────────────────────────────────
    print(f"\n[Models] Loading student checkpoint (LOSO {LOSO_SUBJECT}) ...")
    student = MicroCNN(in_channels=6, num_classes=2)
    student = load_checkpoint(student, MODELS_DIR / f"MicroCNN_distilled_loso_{LOSO_SUBJECT}.pt")
    student.eval()
    print(f"  Student params: {sum(p.numel() for p in student.parameters()):,}")

    # ── 5. Run tasks ──────────────────────────────────────────────────────────
    # Task 1: Channel-level SHAP (teacher)
    teacher_shap_values = task1_channel_shap(teacher, bg_x, test_x)

    # Task 2 (replacement): Grad-CAM temporal on branch_small
    task2_gradcam_temporal(teacher, test_x, test_y)

    # Task 3: Teacher vs Student SHAP comparison
    # Use SAME background for both (same signal space, same N_BG)
    task3_teacher_vs_student(teacher, student,
                             background_t=bg_x,
                             background_s=bg_x,
                             test_x=test_x,
                             test_x_s=test_x)

    # Task 4: Per-class SHAP (stress vs baseline)
    task4_per_class_shap(teacher, bg_x, test_x, test_y)

    print("\n" + "=" * 60)
    print("  All figures saved to outputs/reports/")
    print("=" * 60)
