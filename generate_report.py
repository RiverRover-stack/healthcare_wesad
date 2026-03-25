"""
Generate Report: Produce All Result Figures from Existing Pipeline Results

Usage:
    python generate_report.py

Reads model_comparison.csv for all results and generates publication-ready
PNG figures. No re-training needed — just run after each training step.

Always generated (core figures):
    fig1_model_comparison.png    — Recall / F1 / AUC bar chart
    fig2_loso_per_subject.png    — Per-subject ML breakdown
    fig3_summary_table.png       — Clean metrics table

Generated when student results are available (run train_students.py first):
    fig4_pareto_front.png        — Accuracy vs Model Size scatter
    fig5_kd_improvement.png      — Standalone vs Distilled comparison
    fig6_loso_heatmap.png        — Subjects x Models F1 heatmap

Generated when ablation results are available (run run_ablation.py first):
    fig7_ablation.png            — Temperature and alpha sweep plots
"""

import csv
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

# ─────────────────────────────────────────────────────────────────────────────
# Traditional ML + baseline results (from main.py run, March 24 2026)
# ─────────────────────────────────────────────────────────────────────────────

COMPARISON_RESULTS = {
    'Random\nBaseline': {
        'recall': 0.358, 'f1': 0.364, 'roc_auc': 0.50,
    },
    'Majority\nBaseline': {
        'recall': 0.000, 'f1': 0.000, 'roc_auc': 0.00,
    },
    'EDA\nThreshold': {
        'recall': 0.866, 'f1': 0.792, 'roc_auc': 0.00,
    },
    'Logistic\nRegression': {
        'recall': 0.954, 'recall_std': 0.134,
        'f1':     0.947, 'f1_std':     0.105,
        'roc_auc': 0.976, 'roc_auc_std': 0.062,
    },
    'Random\nForest': {
        'recall': 0.965, 'recall_std': 0.081,
        'f1':     0.956, 'f1_std':     0.086,
        'roc_auc': 0.996, 'roc_auc_std': 0.011,
    },
}

PER_SUBJECT = {
    'LogReg': {
        'S2':  {'recall': 1.000, 'f1': 0.745},
        'S3':  {'recall': 0.474, 'f1': 0.643},
        'S4':  {'recall': 1.000, 'f1': 1.000},
        'S5':  {'recall': 1.000, 'f1': 1.000},
        'S6':  {'recall': 1.000, 'f1': 1.000},
        'S7':  {'recall': 1.000, 'f1': 1.000},
        'S8':  {'recall': 1.000, 'f1': 0.909},
        'S9':  {'recall': 0.842, 'f1': 0.914},
        'S10': {'recall': 1.000, 'f1': 1.000},
        'S11': {'recall': 1.000, 'f1': 1.000},
        'S13': {'recall': 1.000, 'f1': 1.000},
        'S14': {'recall': 1.000, 'f1': 1.000},
        'S15': {'recall': 1.000, 'f1': 1.000},
        'S16': {'recall': 1.000, 'f1': 1.000},
        'S17': {'recall': 1.000, 'f1': 1.000},
    },
    'RandomForest': {
        'S2':  {'recall': 1.000, 'f1': 0.974},
        'S3':  {'recall': 0.789, 'f1': 0.857},
        'S4':  {'recall': 1.000, 'f1': 1.000},
        'S5':  {'recall': 1.000, 'f1': 1.000},
        'S6':  {'recall': 1.000, 'f1': 1.000},
        'S7':  {'recall': 1.000, 'f1': 0.974},
        'S8':  {'recall': 0.950, 'f1': 0.691},
        'S9':  {'recall': 0.737, 'f1': 0.848},
        'S10': {'recall': 1.000, 'f1': 1.000},
        'S11': {'recall': 1.000, 'f1': 1.000},
        'S13': {'recall': 1.000, 'f1': 1.000},
        'S14': {'recall': 1.000, 'f1': 1.000},
        'S15': {'recall': 1.000, 'f1': 1.000},
        'S16': {'recall': 1.000, 'f1': 1.000},
        'S17': {'recall': 1.000, 'f1': 1.000},
    },
}

# Base table rows (CNN + student rows added dynamically from CSV)
BASE_TABLE_ROWS = [
    ['Model',                'Params',       'Accuracy',        'Recall',            'F1',                'ROC-AUC'],
    ['Random Baseline',      '--',           '~0.50',           '0.358',             '0.364',             '--'],
    ['Majority Baseline',    '--',           '~0.67',           '0.000',             '0.000',             '--'],
    ['EDA Threshold',        '--',           '~0.60',           '0.866',             '0.792',             '--'],
    ['Logistic Regression',  '~150 feat.',   '0.964 +/- 0.072', '0.954 +/- 0.134',   '0.947 +/- 0.105',   '0.976 +/- 0.062'],
    ['Random Forest',        '~150 feat.',   '0.966 +/- 0.077', '0.965 +/- 0.081',   '0.956 +/- 0.086',   '0.996 +/- 0.011'],
]


# ─────────────────────────────────────────────────────────────────────────────
# CSV reader helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_all_dl_results():
    """Read all DL model rows from model_comparison.csv."""
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
        # Skip the baseline/ML rows we already have hardcoded
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


def _build_comparison_and_table(dl_results):
    """Merge DL results into COMPARISON_RESULTS dict and table rows."""
    results = dict(COMPARISON_RESULTS)
    table   = [row[:] for row in BASE_TABLE_ROWS]

    teacher_param_count = create_teacher_cnn().count_parameters()

    if dl_results:
        for name, r in dl_results.items():
            if r['recall']['mean'] == 0.0 and r['f1']['mean'] == 0.0:
                continue  # skip collapsed training runs

            # Display name for chart (shorter)
            display = name.replace(' (', '\n(')
            results[display] = {
                'recall':      r['recall']['mean'],
                'recall_std':  r['recall']['std'],
                'f1':          r['f1']['mean'],
                'f1_std':      r['f1']['std'],
                'roc_auc':     r['roc_auc']['mean'],
                'roc_auc_std': r['roc_auc']['std'],
            }

            # Update param count for teacher dynamically
            params = r['params']
            if '1D-CNN Teacher' in name:
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


def _build_advanced_data(dl_results):
    """Build data structures needed for advanced figures (fig4-fig6)."""
    # ── Per-subject data for heatmap ──────────────────────────────────────────
    # Start with the hardcoded ML per-subject data
    per_subject_all = dict(PER_SUBJECT)

    # ── Efficiency benchmarks (quick, CPU-only) ───────────────────────────────
    efficiency_data = {}
    models_to_bench = {'Teacher (Multi-Scale)': create_teacher_cnn()}
    for name, cls in STUDENT_REGISTRY.items():
        models_to_bench[name] = cls()

    print("  Running efficiency benchmarks (CPU)...")
    for mname, model in models_to_bench.items():
        efficiency_data[mname] = get_efficiency_report(model, input_shape=(6, 3840))

    # ── Accuracy data for pareto ──────────────────────────────────────────────
    # Map efficiency model names to CSV result names
    accuracy_data = {}
    for csv_name, r in dl_results.items():
        if 'Teacher' in csv_name:
            accuracy_data['Teacher (Multi-Scale)'] = r
        elif 'MicroCNN (standalone)' in csv_name:
            accuracy_data['MicroCNN'] = r    # default accuracy = standalone
        elif 'MicroCNN (distilled)' in csv_name:
            accuracy_data['MicroCNN (distilled)'] = r
        elif 'TinyCNN (standalone)' in csv_name:
            accuracy_data['TinyCNN'] = r
        elif 'TinyCNN (distilled)' in csv_name:
            accuracy_data['TinyCNN (distilled)'] = r
        elif 'MiniCNN-LSTM (standalone)' in csv_name:
            accuracy_data['MiniCNN-LSTM'] = r
        elif 'MiniCNN-LSTM (distilled)' in csv_name:
            accuracy_data['MiniCNN-LSTM (distilled)'] = r

    # Add ML models to pareto (they have no size_kb, estimate from param count)
    ml_approximate_sizes = {'LogReg': 0.5, 'RandomForest': 45.0}
    for mname, approx_kb in ml_approximate_sizes.items():
        efficiency_data[mname] = {
            'params': 0, 'size_kb': approx_kb,
            'latency_ms': 0.0, 'flops': None,
        }
    accuracy_data['LogReg']      = {'f1': {'mean': 0.947, 'std': 0.105}}
    accuracy_data['RandomForest'] = {'f1': {'mean': 0.956, 'std': 0.086}}

    # ── KD improvement data ───────────────────────────────────────────────────
    standalone_res = {}
    distilled_res  = {}
    for csv_name, r in dl_results.items():
        if '(standalone)' in csv_name:
            base = csv_name.replace(' (standalone)', '')
            standalone_res[base] = r
        elif '(distilled)' in csv_name:
            base = csv_name.replace(' (distilled)', '')
            distilled_res[base] = r

    return per_subject_all, efficiency_data, accuracy_data, standalone_res, distilled_res


def _read_ablation_results():
    """Read ablation_results.csv if it exists."""
    path = REPORTS_DIR / 'ablation_results.csv'
    if not path.exists():
        return {}, {}
    temp_res, alpha_res = {}, {}
    try:
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                val = float(row['value'])
                f1  = float(row['f1'])
                if row['sweep'] == 'temperature':
                    temp_res[val] = f1
                else:
                    alpha_res[val] = f1
    except Exception:
        pass
    return temp_res, alpha_res


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    create_directories()

    dl_results = _read_all_dl_results()
    teacher_done   = any('Teacher' in k for k in dl_results)
    students_done  = any('standalone' in k or 'distilled' in k for k in dl_results)

    if teacher_done:
        print(f"  Found {len(dl_results)} DL model result(s) in model_comparison.csv")
    else:
        print("  No DL results yet (run train_teacher.py)")

    # ── Core figures (always generated) ──────────────────────────────────────
    comparison_results, table_rows = _build_comparison_and_table(dl_results)
    generate_all_figures(comparison_results, PER_SUBJECT, table_rows)

    # ── Advanced figures (only when student results exist) ────────────────────
    if students_done:
        print("\n  Student results found — generating advanced figures...")
        per_subject_all, eff_data, acc_data, sa_res, kd_res = \
            _build_advanced_data(dl_results)
        temp_res, alpha_res = _read_ablation_results()

        generate_advanced_figures(
            per_subject_all=per_subject_all,
            efficiency_data=eff_data,
            accuracy_data=acc_data,
            standalone_res=sa_res  or None,
            distilled_res=kd_res   or None,
            temp_ablation=temp_res  or None,
            alpha_ablation=alpha_res or None,
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
