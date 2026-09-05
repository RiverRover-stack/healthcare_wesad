"""
Generate Report: Produce All Result Figures from Existing Pipeline Results

Usage:
    python generate_report.py

Reads outputs/reports/*.csv (baseline_results.csv, classical_loso.csv,
classical_per_fold.csv, model_comparison.csv, per_fold_results.csv,
ablation_results.csv) and generates publication-ready PNG figures. Nothing
here is hand-transcribed -- every number comes from a file a pipeline run
actually wrote. No re-training needed -- just run after each training step.

Always generated (core figures):
    fig1_model_comparison.png    — Recall / F1 / AUC bar chart
    fig2_loso_per_subject.png    — Per-subject ML breakdown
    fig3_summary_table.png       — Clean metrics table

Generated when student results are available (run train_students.py first):
    fig4_pareto_front.png        — Accuracy vs Model Size scatter
    fig5_kd_improvement.png      — Standalone vs Distilled comparison
    fig6_loso_heatmap.png        — Subjects x Models F1 heatmap

Generated when ablation results are available (run run_ablation.py first):
    fig7_ablation.png            — Temperature x alpha F1 heatmap
"""

import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import create_directories, REPORTS_DIR
from evaluation.reporter import (
    generate_all_figures,
    generate_advanced_figures,
)
from evaluation.efficiency import get_efficiency_report
from models.teacher import create_teacher_cnn
from models.student import STUDENT_REGISTRY

BASELINE_DISPLAY = {
    'random': ('Random\nBaseline', 'Random Baseline'),
    'majority': ('Majority\nBaseline', 'Majority Baseline'),
    'eda_threshold': ('EDA\nThreshold', 'EDA Threshold'),
}
CLASSICAL_DISPLAY = {
    'LogReg': ('Logistic\nRegression', 'Logistic Regression'),
    'RandomForest': ('Random\nForest', 'Random Forest'),
}


# ─────────────────────────────────────────────────────────────────────────────
# CSV / JSON reader helpers -- every value traces back to a file a run wrote
# ─────────────────────────────────────────────────────────────────────────────

def _read_baseline_results():
    """Read baseline_results.csv (written by main.py). {} if not yet run."""
    path = REPORTS_DIR / 'baseline_results.csv'
    if not path.exists():
        return {}
    rows = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows[row['model']] = {k: float(row[k]) for k in
                                   ('accuracy', 'precision', 'recall', 'f1')}
    return rows


def _read_classical_loso():
    """Read classical_loso.csv (LogReg/RF aggregates, written by main.py)."""
    path = REPORTS_DIR / 'classical_loso.csv'
    if not path.exists():
        return {}
    rows = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows[row['model']] = {
                metric: {'mean': float(row[f'{metric}_mean']), 'std': float(row[f'{metric}_std'])}
                for metric in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
            }
    return rows


def _read_classical_per_fold():
    """Read classical_per_fold.csv into {'LogReg': {'S2': {'recall':.., 'f1':..}}, ...}."""
    path = REPORTS_DIR / 'classical_per_fold.csv'
    if not path.exists():
        return {}
    per_subject = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            per_subject.setdefault(row['model'], {})[row['test_subject']] = {
                'recall': float(row['recall']), 'f1': float(row['f1']),
            }
    return per_subject


def _read_dataset_stats():
    """Read dataset_stats.json (written by main.py). None if not yet run."""
    path = REPORTS_DIR / 'dataset_stats.json'
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _read_all_dl_results():
    """Read all DL (teacher/student) rows from model_comparison.csv."""
    csv_path = REPORTS_DIR / "model_comparison.csv"
    if not csv_path.exists():
        return {}

    dl_rows = {}
    try:
        with open(csv_path, encoding='utf-8') as f:
            rows = list(csv.reader(f))
    except Exception:
        return {}

    for row in rows:
        if len(row) < 6:
            continue
        name = row[0].strip()
        # Skip the baseline/ML rows -- those come from the dedicated CSVs above.
        if any(x in name for x in ('Baseline', 'EDA Threshold', 'Logistic', 'Random Forest')):
            continue

        def parse(cell):
            cell = cell.strip()
            if '+/-' in cell:
                parts = cell.split('+/-')
                try:
                    return float(parts[0].strip()), float(parts[1].strip())
                except ValueError:
                    return None, None
            try:
                return float(cell), 0.0
            except ValueError:
                return None, None

        params       = row[1].strip()
        acc_m, acc_s = parse(row[2]) if len(row) > 2 else (None, None)
        rec_m, rec_s = parse(row[3]) if len(row) > 3 else (None, None)
        f1_m,  f1_s  = parse(row[4]) if len(row) > 4 else (None, None)
        auc_m, auc_s = parse(row[5]) if len(row) > 5 else (None, None)

        if rec_m is None:
            continue

        dl_rows[name] = {
            'params':  params,
            'accuracy':  {'mean': acc_m, 'std': acc_s},
            'recall':    {'mean': rec_m, 'std': rec_s},
            'f1':        {'mean': f1_m,  'std': f1_s},
            'roc_auc':   {'mean': auc_m, 'std': auc_s},
        }
    return dl_rows


def _read_dl_per_fold():
    """Read per_fold_results.csv into {'MicroCNN (distilled)': {'S2': {'f1':.., 'recall':..}}, ...}."""
    path = REPORTS_DIR / 'per_fold_results.csv'
    if not path.exists():
        return {}
    per_subject = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['mode'] == 'teacher':
                label = 'Teacher'
            else:
                label = f"{row['model']} ({row['mode']})"
            try:
                per_subject.setdefault(label, {})[row['test_subject']] = {
                    'recall': float(row['test_recall']), 'f1': float(row['test_f1']),
                }
            except (KeyError, ValueError):
                continue
    return per_subject


def _read_ablation_results():
    """
    Read ablation_results.csv (one row per (temperature, alpha) configuration)
    into a grid dict plus the sorted axis values, derived entirely from the
    CSV so the ablation figure is regenerable without re-running the sweep.
    """
    path = REPORTS_DIR / 'ablation_results.csv'
    if not path.exists():
        return {}, [], []
    grid = {}
    try:
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                t = float(row['temperature'])
                a = float(row['alpha'])
                grid[(t, a)] = float(row['f1_mean'])
    except Exception:
        return {}, [], []
    temperatures = sorted({t for t, _ in grid})
    alphas = sorted({a for _, a in grid})
    return grid, temperatures, alphas


# ─────────────────────────────────────────────────────────────────────────────
# Data assembly for figures / tables
# ─────────────────────────────────────────────────────────────────────────────

def _build_comparison_and_table(baseline_rows, classical_agg, dl_results, n_features):
    """Build the fig1 bar-chart dict and the fig3 table rows from real, computed data."""
    results = {}
    table = [['Model', 'Params', 'Accuracy', 'Recall', 'F1', 'ROC-AUC']]
    feat_str = f'{n_features} feat.' if n_features else '--'

    for key, (chart_name, table_name) in BASELINE_DISPLAY.items():
        if key not in baseline_rows:
            continue
        m = baseline_rows[key]
        results[chart_name] = {'recall': m['recall'], 'f1': m['f1'], 'roc_auc': 0.0}
        # EDA threshold's accuracy doesn't summarise a per-subject adaptive
        # threshold meaningfully -- report '--' rather than a misleading number.
        acc_str = '--' if key == 'eda_threshold' else f"{m['accuracy']:.3f}"
        table.append([table_name, '--', acc_str, f"{m['recall']:.3f}", f"{m['f1']:.3f}", '--'])

    for name, (chart_name, table_name) in CLASSICAL_DISPLAY.items():
        if name not in classical_agg:
            continue
        agg = classical_agg[name]
        results[chart_name] = {
            'recall': agg['recall']['mean'], 'recall_std': agg['recall']['std'],
            'f1': agg['f1']['mean'], 'f1_std': agg['f1']['std'],
            'roc_auc': agg['roc_auc']['mean'], 'roc_auc_std': agg['roc_auc']['std'],
        }
        table.append([
            table_name, feat_str,
            f"{agg['accuracy']['mean']:.3f} +/- {agg['accuracy']['std']:.3f}",
            f"{agg['recall']['mean']:.3f} +/- {agg['recall']['std']:.3f}",
            f"{agg['f1']['mean']:.3f} +/- {agg['f1']['std']:.3f}",
            f"{agg['roc_auc']['mean']:.3f} +/- {agg['roc_auc']['std']:.3f}",
        ])

    teacher_param_count = create_teacher_cnn().count_parameters()

    if dl_results:
        for name, r in dl_results.items():
            if r['recall']['mean'] == 0.0 and r['f1']['mean'] == 0.0:
                continue  # skip collapsed training runs

            display = name.replace(' (', '\n(')
            results[display] = {
                'recall':      r['recall']['mean'],
                'recall_std':  r['recall']['std'],
                'f1':          r['f1']['mean'],
                'f1_std':      r['f1']['std'],
                'roc_auc':     r['roc_auc']['mean'],
                'roc_auc_std': r['roc_auc']['std'],
            }

            params = r['params']
            if '1D-CNN Teacher' in name or name == 'Teacher':
                params = f"~{teacher_param_count // 1000}K"

            table.append([
                name, params,
                f"{r['accuracy']['mean']:.3f} +/- {r['accuracy']['std']:.3f}",
                f"{r['recall']['mean']:.3f} +/- {r['recall']['std']:.3f}",
                f"{r['f1']['mean']:.3f} +/- {r['f1']['std']:.3f}",
                f"{r['roc_auc']['mean']:.3f} +/- {r['roc_auc']['std']:.3f}",
            ])
    else:
        table.append([
            '1D-CNN Teacher (Multi-Scale)', f'~{teacher_param_count // 1000}K',
            'Run train_teacher.py', 'Run train_teacher.py',
            'Run train_teacher.py', 'Run train_teacher.py',
        ])

    return results, table


def _build_pareto_points(dl_results, classical_agg):
    """
    Explicit (label, size_kb, f1_mean, category) points for fig4 -- no
    substring guessing on model names, so standalone and distilled students
    of the same architecture (same size_kb, different F1) both show up.
    """
    print("  Running efficiency benchmarks (CPU)...")
    efficiency = {'Teacher (Multi-Scale)': get_efficiency_report(create_teacher_cnn(), input_shape=(6, 3840))}
    for name, cls in STUDENT_REGISTRY.items():
        efficiency[name] = get_efficiency_report(cls(), input_shape=(6, 3840))

    points = []

    teacher_key = next((k for k in dl_results if 'Teacher' in k), None)
    if teacher_key and efficiency['Teacher (Multi-Scale)'].get('size_kb') is not None:
        points.append({
            'label': 'Teacher (Multi-Scale)',
            'size_kb': efficiency['Teacher (Multi-Scale)']['size_kb'],
            'f1_mean': dl_results[teacher_key]['f1']['mean'],
            'category': 'teacher',
        })

    for model_name in STUDENT_REGISTRY:
        size_kb = efficiency.get(model_name, {}).get('size_kb')
        if size_kb is None:
            continue
        for mode in ('standalone', 'distilled'):
            key = f'{model_name} ({mode})'
            if key not in dl_results:
                continue
            points.append({
                'label': key,
                'size_kb': size_kb,
                'f1_mean': dl_results[key]['f1']['mean'],
                'category': mode,
            })

    # Traditional ML: no serialized model size to measure here, so these are
    # rough estimates (LogReg coefficients / RF tree ensemble), clearly not
    # measured the way DL model sizes are.
    ml_approx_size_kb = {'LogReg': 0.5, 'RandomForest': 45.0}
    for name, size_kb in ml_approx_size_kb.items():
        if name in classical_agg:
            points.append({
                'label': name,
                'size_kb': size_kb,
                'f1_mean': classical_agg[name]['f1']['mean'],
                'category': 'ml',
            })

    return points, efficiency


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    create_directories()

    baseline_rows = _read_baseline_results()
    classical_agg = _read_classical_loso()
    classical_per_fold = _read_classical_per_fold()
    dataset_stats = _read_dataset_stats()
    dl_results = _read_all_dl_results()

    teacher_done   = any('Teacher' in k for k in dl_results)
    students_done  = any('standalone' in k or 'distilled' in k for k in dl_results)

    if teacher_done:
        print(f"  Found {len(dl_results)} DL model result(s) in model_comparison.csv")
    else:
        print("  No DL results yet (run train_teacher.py)")
    if not baseline_rows or not classical_agg:
        print("  No classical/baseline results yet (run main.py)")

    # ── Core figures (always generated) ──────────────────────────────────────
    n_features = dataset_stats['n_features'] if dataset_stats else None
    comparison_results, table_rows = _build_comparison_and_table(
        baseline_rows, classical_agg, dl_results, n_features)

    per_subject_core = dict(classical_per_fold)  # {'LogReg': {...}, 'RandomForest': {...}}
    generate_all_figures(comparison_results, per_subject_core, table_rows)

    # ── Advanced figures (only when student results exist) ────────────────────
    if students_done:
        print("\n  Student results found — generating advanced figures...")

        pareto_points, _ = _build_pareto_points(dl_results, classical_agg)

        standalone_res = {}
        distilled_res  = {}
        for csv_name, r in dl_results.items():
            if '(standalone)' in csv_name:
                standalone_res[csv_name.replace(' (standalone)', '')] = r
            elif '(distilled)' in csv_name:
                distilled_res[csv_name.replace(' (distilled)', '')] = r

        per_subject_all = dict(classical_per_fold)
        per_subject_all.update(_read_dl_per_fold())

        ablation_grid, ablation_temps, ablation_alphas = _read_ablation_results()

        generate_advanced_figures(
            per_subject_all=per_subject_all,
            pareto_points=pareto_points or None,
            standalone_res=standalone_res or None,
            distilled_res=distilled_res   or None,
            ablation_grid=ablation_grid or None,
            ablation_temperatures=ablation_temps,
            ablation_alphas=ablation_alphas,
        )
    else:
        print("\n  Run train_students.py to unlock advanced figures (fig4-fig7)")

    print('\nDone! Output files:')
    print('  outputs/reports/fig1_model_comparison.png   (always)')
    print('  outputs/reports/fig2_loso_per_subject.png   (always)')
    print('  outputs/reports/fig3_summary_table.png      (always)')
    if students_done:
        print('  outputs/reports/fig4_pareto_front.png       (new)')
        print('  outputs/reports/fig5_kd_improvement.png     (new)')
        print('  outputs/reports/fig6_loso_heatmap.png       (new)')
