"""
Teacher vs Student — Per-Subject LOSO Performance Comparison

Extracts accuracy, F1, and ROC-AUC from saved checkpoints for every
LOSO fold and plots side-by-side grouped bar charts.

Usage:
    python plot_teacher_vs_student.py

Output:
    outputs/reports/teacher_vs_student_per_subject.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ── paths & constants ─────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
MODELS_DIR  = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"
DPI         = 300

ALL_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9',
                'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

MODELS = {
    "Teacher (Multi-Scale CNN)":  "teacher_loso_{}.pt",
    "MicroCNN (distilled)":       "MicroCNN_distilled_loso_{}.pt",
    "MicroCNN (standalone)":      "MicroCNN_standalone_loso_{}.pt",
}

COLOURS = {
    "Teacher (Multi-Scale CNN)":  "#4C72B0",
    "MicroCNN (distilled)":       "#DD8452",
    "MicroCNN (standalone)":      "#937860",
}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         100,
})


def load_metrics(pattern: str, subjects: list) -> dict:
    """Load per-subject metrics from checkpoints matching `pattern`."""
    results = {}
    for sid in subjects:
        path = MODELS_DIR / pattern.format(sid)
        if not path.exists():
            print(f"  [WARN] Missing: {path.name}")
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "metrics" in ckpt:
            results[sid] = ckpt["metrics"]
        else:
            print(f"  [WARN] No metrics in {path.name}")
    return results


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Extract metrics ────────────────────────────────────────────────────
    all_metrics = {}
    for label, pattern in MODELS.items():
        print(f"Loading {label} ...")
        all_metrics[label] = load_metrics(pattern, ALL_SUBJECTS)

    # only keep subjects present in ALL models
    common = set(ALL_SUBJECTS)
    for label in MODELS:
        common &= set(all_metrics[label].keys())
    subjects = sorted(common, key=lambda s: int(s[1:]))
    print(f"\nCommon subjects ({len(subjects)}): {subjects}")

    # ── 2. Build arrays ───────────────────────────────────────────────────────
    metric_keys = ["accuracy", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "F1 Score", "ROC-AUC"]

    data = {}  # model_label -> metric_key -> list of values per subject
    for label in MODELS:
        data[label] = {}
        for mk in metric_keys:
            data[label][mk] = [all_metrics[label][s].get(mk, 0) for s in subjects]

    # ── 3. Print summary table ────────────────────────────────────────────────
    print(f"\n{'Subject':>8}", end="")
    for label in MODELS:
        print(f"  {label[:20]:>20}", end="")
    print()
    print("-" * (8 + 22 * len(MODELS)))
    for i, sid in enumerate(subjects):
        print(f"{sid:>8}", end="")
        for label in MODELS:
            acc = data[label]["accuracy"][i]
            f1  = data[label]["f1"][i]
            print(f"  Acc={acc:.3f} F1={f1:.3f}", end="")
        print()

    # means
    print("-" * (8 + 22 * len(MODELS)))
    print(f"{'Mean':>8}", end="")
    for label in MODELS:
        acc = np.mean(data[label]["accuracy"])
        f1  = np.mean(data[label]["f1"])
        print(f"  Acc={acc:.3f} F1={f1:.3f}", end="")
    print()

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    n_subjects = len(subjects)
    n_models   = len(MODELS)
    model_labels = list(MODELS.keys())

    fig, axes = plt.subplots(len(metric_keys), 1,
                              figsize=(14, 4 * len(metric_keys)),
                              sharex=True)
    if len(metric_keys) == 1:
        axes = [axes]

    x = np.arange(n_subjects)
    total_w = 0.75
    bar_w   = total_w / n_models

    for ax, mk, ml in zip(axes, metric_keys, metric_labels):
        for j, label in enumerate(model_labels):
            vals = data[label][mk]
            offset = (j - (n_models - 1) / 2) * bar_w
            bars = ax.bar(x + offset, vals, width=bar_w,
                          label=label, color=COLOURS[label],
                          edgecolor="white", linewidth=0.6)

        ax.set_ylabel(ml, fontsize=11, fontweight="bold")
        ax.set_ylim(
            max(0, min(min(data[l][mk]) for l in model_labels) - 0.05),
            1.02
        )
        ax.axhline(y=np.mean(data[model_labels[0]][mk]),
                    color=COLOURS[model_labels[0]], ls="--", lw=0.9,
                    alpha=0.5, label=f"Teacher mean ({np.mean(data[model_labels[0]][mk]):.3f})")
        ax.axhline(y=np.mean(data[model_labels[1]][mk]),
                    color=COLOURS[model_labels[1]], ls="--", lw=0.9,
                    alpha=0.5, label=f"Distilled mean ({np.mean(data[model_labels[1]][mk]):.3f})")
        ax.legend(fontsize=8, loc="lower left", ncol=2)
        ax.grid(axis="y", alpha=0.2)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(subjects, fontsize=10)
    axes[-1].set_xlabel("Left-Out Subject (LOSO fold)", fontsize=11)

    fig.suptitle(
        "Teacher vs Student — Per-Subject LOSO Performance\n"
        "Multi-Scale CNN (266K) vs MicroCNN Distilled (5.3K) vs MicroCNN Standalone (5.3K)",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    out_path = REPORTS_DIR / "teacher_vs_student_per_subject.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
