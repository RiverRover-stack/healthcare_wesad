"""
Reporter: Generate Publication-Quality Result Figures

Outputs:
    fig1_model_comparison.png  — Recall / F1 / AUC bar chart for all models
    fig2_loso_per_subject.png  — Per-subject LOSO breakdown (LogReg & RF)
    fig3_summary_table.png     — Clean metrics table for printing / slides
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import REPORTS_DIR


# ── Colour palette (accessible, prints well in B&W too) ─────────────────────
COLORS = {
    'recall':    '#2196F3',   # blue
    'f1':        '#4CAF50',   # green
    'roc_auc':   '#FF9800',   # orange
    'highlight': '#E53935',   # red accent
    'grid':      '#EEEEEE',
    'text':      '#212121',
}

SUBJECT_ORDER = ['S2','S3','S4','S5','S6','S7','S8','S9',
                 'S10','S11','S13','S14','S15','S16','S17']


def _style():
    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'font.size':         11,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'axes.grid.axis':    'y',
        'grid.color':        COLORS['grid'],
        'grid.linewidth':    0.8,
        'figure.dpi':        150,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(results: Dict, save: bool = True) -> plt.Figure:
    """
    Bar chart comparing Recall / F1 / AUC across all models.

    results format:
        {
          'model_name': {
              'recall': float, 'f1': float, 'roc_auc': float,
              'recall_std': float (optional), 'f1_std': float (optional),
          },
          ...
        }
    """
    _style()
    model_names = list(results.keys())
    n = len(model_names)
    metrics = ['recall', 'f1', 'roc_auc']
    metric_labels = ['Recall', 'F1', 'ROC-AUC']
    colors = [COLORS['recall'], COLORS['f1'], COLORS['roc_auc']]

    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [results[m].get(metric, 0) for m in model_names]
        errs = [results[m].get(f'{metric}_std', 0) for m in model_names]
        bars = ax.bar(x + i * width, vals, width,
                      label=label, color=color, alpha=0.88,
                      yerr=errs, capsize=3, error_kw={'linewidth': 1.2})

        # Value labels on bars
        for bar, val, err in zip(bars, vals, errs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + err + 0.012,
                        f'{val:.2f}', ha='center', va='bottom',
                        fontsize=8.5, color=COLORS['text'])

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=18, ha='right', fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Stress Detection — Model Comparison (LOSO Cross-Validation)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.axhline(y=0.9, color='#BDBDBD', linestyle='--', linewidth=0.8, zorder=0)

    fig.tight_layout()

    if save:
        path = REPORTS_DIR / 'fig1_model_comparison.png'
        fig.savefig(path, bbox_inches='tight')
        print(f'  Saved -> {path}')

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Per-Subject LOSO Breakdown
# ─────────────────────────────────────────────────────────────────────────────

def plot_loso_per_subject(per_subject: Dict[str, Dict[str, Dict]],
                          save: bool = True) -> plt.Figure:
    """
    Per-subject Recall and F1 for multiple models side by side.

    per_subject format:
        {
          'LogReg':       {'S2': {'recall': 1.0, 'f1': 0.74}, ...},
          'RandomForest': {'S2': {'recall': 1.0, 'f1': 0.97}, ...},
        }
    """
    _style()
    model_names = list(per_subject.keys())
    subjects = SUBJECT_ORDER
    n_subj = len(subjects)
    n_models = len(model_names)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    metric_pairs = [('recall', 'Recall'), ('f1', 'F1 Score')]
    model_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    x = np.arange(n_subj)
    width = 0.8 / n_models

    for ax, (metric, ylabel) in zip(axes, metric_pairs):
        for i, (model, color) in enumerate(zip(model_names, model_colors)):
            vals = [per_subject[model].get(s, {}).get(metric, 0) for s in subjects]
            offset = (i - (n_models - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=model,
                          color=color, alpha=0.85)

        ax.set_ylim(0, 1.15)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc='lower left', fontsize=9)
        ax.axhline(y=1.0, color='#BDBDBD', linestyle='--', linewidth=0.8, zorder=0)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(subjects, fontsize=10)
    axes[-1].set_xlabel('Subject', fontsize=11)

    fig.suptitle('LOSO Per-Subject Performance', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()

    if save:
        path = REPORTS_DIR / 'fig2_loso_per_subject.png'
        fig.savefig(path, bbox_inches='tight')
        print(f'  Saved -> {path}')

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Summary Table
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_table(rows: List[List[str]], save: bool = True) -> plt.Figure:
    """
    Render a clean metrics table as a PNG.

    rows: list of [Model, Params, Accuracy, Recall, F1, AUC]
          First row is the header.
    """
    _style()
    headers = rows[0]
    data = rows[1:]
    n_rows = len(data)
    n_cols = len(headers)

    fig, ax = plt.subplots(figsize=(12, 0.55 * n_rows + 1.6))
    ax.axis('off')

    col_widths = [0.28, 0.10, 0.14, 0.14, 0.14, 0.14]

    table = ax.table(
        cellText=data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.9)

    # Header style
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold')

    # Alternating row colours; highlight best rows
    best_section_start = None
    for i, row in enumerate(data):
        is_ml = any(x in row[0] for x in ['Logistic', 'Random Forest', 'CNN'])
        for j in range(n_cols):
            cell = table[i + 1, j]
            if is_ml:
                cell.set_facecolor('#E3F2FD')
            elif i % 2 == 0:
                cell.set_facecolor('#FAFAFA')
            else:
                cell.set_facecolor('#FFFFFF')
            cell.set_edgecolor('#E0E0E0')

    ax.set_title('WESAD Stress Detection — Results Summary',
                 fontsize=13, fontweight='bold', pad=10)

    fig.tight_layout()

    if save:
        path = REPORTS_DIR / 'fig3_summary_table.png'
        fig.savefig(path, bbox_inches='tight', dpi=180)
        print(f'  Saved -> {path}')

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Pareto Front: Accuracy vs Model Size
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto_front(
    efficiency_data: Dict[str, Dict],
    accuracy_data:   Dict[str, Dict],
    save: bool = True,
) -> plt.Figure:
    """
    Scatter plot showing the accuracy vs model size trade-off (Pareto front).

    efficiency_data format:
        {'ModelName': {'size_kb': float, 'params': int, 'latency_ms': float}, ...}
    accuracy_data format:
        {'ModelName': {'f1': {'mean': float, 'std': float}}, ...}
    """
    _style()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Color map: blue=baselines/ML, green=teacher, orange=standalone, red=distilled
    color_map = {
        'teacher':    '#1565C0',
        'standalone': '#FF9800',
        'distilled':  '#E53935',
        'ml':         '#4CAF50',
    }
    marker_map = {
        'teacher': '*', 'standalone': 'o', 'distilled': 's', 'ml': '^',
    }

    plotted = []
    for name, eff in efficiency_data.items():
        size_kb = eff.get('size_kb', None)
        if size_kb is None or name not in accuracy_data:
            continue
        f1 = accuracy_data[name].get('f1', {}).get('mean', None)
        if f1 is None:
            continue

        # Determine category for styling
        low = name.lower()
        if 'teacher' in low:
            cat = 'teacher'
        elif 'distill' in low:
            cat = 'distilled'
        elif 'standalone' in low:
            cat = 'standalone'
        else:
            cat = 'ml'

        ax.scatter(size_kb, f1,
                   s=120, zorder=5,
                   color=color_map[cat],
                   marker=marker_map[cat],
                   edgecolors='white', linewidths=0.8)
        ax.annotate(name, (size_kb, f1),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=8.5, color=COLORS['text'])
        plotted.append((size_kb, f1, cat))

    # Draw Pareto front (non-dominated points: larger size, higher F1)
    if plotted:
        pts = sorted(plotted, key=lambda p: p[0])
        pareto = []
        best_f1 = -1.0
        for sz, f1, _ in pts:
            if f1 > best_f1:
                pareto.append((sz, f1))
                best_f1 = f1
        if len(pareto) > 1:
            px, py = zip(*pareto)
            ax.step(px, py, where='post', linestyle='--',
                    color='#BDBDBD', linewidth=1.2, zorder=1, label='Pareto front')

    # Legend patches
    patches = [
        mpatches.Patch(color=color_map['teacher'],    label='Teacher (Multi-Scale)'),
        mpatches.Patch(color=color_map['standalone'], label='Student (standalone)'),
        mpatches.Patch(color=color_map['distilled'],  label='Student (distilled)'),
        mpatches.Patch(color=color_map['ml'],         label='Traditional ML'),
    ]
    ax.legend(handles=patches, fontsize=9, loc='lower right')

    ax.set_xlabel('Model Size (KB, FP32)', fontsize=12)
    ax.set_ylabel('F1 Score (LOSO mean)', fontsize=12)
    ax.set_title('Accuracy vs Model Size — Pareto Front', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.axhline(y=0.9, color='#EEEEEE', linestyle='--', linewidth=0.8, zorder=0)

    fig.tight_layout()
    if save:
        path = REPORTS_DIR / 'fig4_pareto_front.png'
        fig.savefig(path, bbox_inches='tight')
        print(f'  Saved -> {path}')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — KD Improvement: Standalone vs Distilled
# ─────────────────────────────────────────────────────────────────────────────

def plot_kd_improvement(
    standalone_results: Dict[str, Dict],
    distilled_results:  Dict[str, Dict],
    save: bool = True,
) -> plt.Figure:
    """
    Grouped bar chart comparing standalone vs distilled F1 for each student.

    standalone_results / distilled_results format:
        {'MicroCNN': {'f1': {'mean': float, 'std': float}}, ...}
    """
    _style()
    model_names = list(standalone_results.keys())
    n = len(model_names)
    x = np.arange(n)
    width = 0.35

    sa_vals = [standalone_results[m]['f1']['mean'] for m in model_names]
    sa_errs = [standalone_results[m]['f1']['std']  for m in model_names]
    kd_vals = [distilled_results.get(m, {}).get('f1', {}).get('mean', 0) for m in model_names]
    kd_errs = [distilled_results.get(m, {}).get('f1', {}).get('std',  0) for m in model_names]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_sa = ax.bar(x - width/2, sa_vals, width, label='Standalone',
                     color='#FF9800', alpha=0.88,
                     yerr=sa_errs, capsize=4, error_kw={'linewidth': 1.2})
    bars_kd = ax.bar(x + width/2, kd_vals, width, label='Distilled (KD)',
                     color='#E53935', alpha=0.88,
                     yerr=kd_errs, capsize=4, error_kw={'linewidth': 1.2})

    # Improvement delta labels
    for i, (sa, kd) in enumerate(zip(sa_vals, kd_vals)):
        delta = kd - sa
        if abs(delta) > 0.001:
            sign = '+' if delta >= 0 else ''
            ax.text(i + width/2 + 0.02, max(sa, kd) + max(sa_errs[i], kd_errs[i]) + 0.02,
                    f'{sign}{delta:.3f}', ha='center', fontsize=9,
                    color='#1565C0', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('F1 Score (LOSO mean)', fontsize=12)
    ax.set_title('Knowledge Distillation Improvement over Standalone Training',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(y=0.9, color='#EEEEEE', linestyle='--', linewidth=0.8, zorder=0)

    fig.tight_layout()
    if save:
        path = REPORTS_DIR / 'fig5_kd_improvement.png'
        fig.savefig(path, bbox_inches='tight')
        print(f'  Saved -> {path}')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Per-Subject F1 Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_loso_heatmap(
    per_subject_data: Dict[str, Dict[str, Dict]],
    metric: str = 'f1',
    save: bool = True,
) -> plt.Figure:
    """
    Heatmap of per-subject F1 scores across multiple models.
    Rows = subjects, columns = models. Color = metric value.

    per_subject_data format:
        {'ModelName': {'S2': {'f1': float, 'recall': float}, 'S3': {...}, ...}, ...}
    """
    _style()
    model_names = list(per_subject_data.keys())
    subjects    = SUBJECT_ORDER

    # Build matrix
    matrix = np.full((len(subjects), len(model_names)), np.nan)
    for j, model in enumerate(model_names):
        for i, subj in enumerate(subjects):
            val = per_subject_data[model].get(subj, {}).get(metric, None)
            if val is not None:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.8), 7))

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

    # Annotate cells
    for i in range(len(subjects)):
        for j in range(len(model_names)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.4 or val > 0.85 else COLORS['text']
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8.5, color=color)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=25, ha='right', fontsize=10)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(metric.upper() + ' Score', fontsize=10)

    label_map = {'f1': 'F1', 'recall': 'Recall', 'accuracy': 'Accuracy'}
    ax.set_title(f'Per-Subject LOSO {label_map.get(metric, metric)} — All Models',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Subject', fontsize=11)

    fig.tight_layout()
    if save:
        path = REPORTS_DIR / 'fig6_loso_heatmap.png'
        fig.savefig(path, bbox_inches='tight')
        print(f'  Saved -> {path}')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Ablation: KD Temperature and Alpha Sweep
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation(
    temp_results:  Dict[float, float],
    alpha_results: Dict[float, float],
    save: bool = True,
) -> plt.Figure:
    """
    Two-panel ablation plot:
      Left:  F1 vs KD Temperature
      Right: F1 vs KD Alpha (teacher weight)

    temp_results  format: {1: 0.82, 2: 0.85, 4: 0.88, 8: 0.87}
    alpha_results format: {0.3: 0.82, 0.5: 0.85, 0.7: 0.88, 0.9: 0.86}
    """
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    def _plot_sweep(ax, data, xlabel, best_marker_x, title):
        xs = sorted(data.keys())
        ys = [data[x] for x in xs]
        ax.plot(xs, ys, 'o-', color='#2196F3', linewidth=2, markersize=7)
        ax.axvline(x=best_marker_x, color='#E53935', linestyle='--',
                   linewidth=1.2, label=f'Best ({best_marker_x})')
        for x, y in zip(xs, ys):
            ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('F1 Score', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)

    if temp_results:
        best_t = max(temp_results, key=temp_results.get)
        _plot_sweep(ax1, temp_results, 'KD Temperature (T)', best_t,
                    'Temperature Ablation')

    if alpha_results:
        best_a = max(alpha_results, key=alpha_results.get)
        _plot_sweep(ax2, alpha_results, 'KD Alpha (teacher weight)', best_a,
                    'Alpha Ablation')

    fig.suptitle('KD Hyperparameter Ablation — MicroCNN on WESAD',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()

    if save:
        path = REPORTS_DIR / 'fig7_ablation.png'
        fig.savefig(path, bbox_inches='tight')
        print(f'  Saved -> {path}')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate all three figures at once
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(comparison_results: Dict,
                         per_subject: Dict[str, Dict],
                         table_rows: List[List[str]]) -> None:
    """Generate and save the three core report figures (always available)."""
    from config import create_directories
    create_directories()
    print('\n  Generating report figures...')
    plot_model_comparison(comparison_results)
    plot_loso_per_subject(per_subject)
    plot_summary_table(table_rows)
    print('  All figures saved to outputs/reports/')


def generate_advanced_figures(
    per_subject_all: Optional[Dict[str, Dict[str, Dict]]] = None,
    efficiency_data: Optional[Dict[str, Dict]] = None,
    accuracy_data:   Optional[Dict[str, Dict]] = None,
    standalone_res:  Optional[Dict[str, Dict]] = None,
    distilled_res:   Optional[Dict[str, Dict]] = None,
    temp_ablation:   Optional[Dict[float, float]] = None,
    alpha_ablation:  Optional[Dict[float, float]] = None,
) -> None:
    """
    Generate advanced figures (fig4-fig7) when the relevant data is available.
    Each figure is skipped gracefully if its required data is missing.
    """
    from config import create_directories
    create_directories()
    print('\n  Generating advanced figures...')

    if efficiency_data and accuracy_data:
        plot_pareto_front(efficiency_data, accuracy_data)
    else:
        print('  Skipping fig4 (pareto front) — run efficiency benchmarks first')

    if standalone_res and distilled_res:
        plot_kd_improvement(standalone_res, distilled_res)
    else:
        print('  Skipping fig5 (KD improvement) — run train_students.py first')

    if per_subject_all:
        plot_loso_heatmap(per_subject_all)
    else:
        print('  Skipping fig6 (heatmap) — per-subject data needed')

    if temp_ablation or alpha_ablation:
        plot_ablation(temp_ablation or {}, alpha_ablation or {})
