"""
Results Aggregation and LaTeX Table Generator

Reads outputs/reports/model_comparison.csv and produces:
    - A structured Python dict for downstream use
    - LaTeX-formatted table strings ready to paste into the paper

Two table formats:
    accuracy_table()   — Model | Params | Accuracy | Recall | F1 | ROC-AUC
    efficiency_table() — Model | Params | Size(KB) | Latency(ms) | FLOPs | F1

Usage:
    from src.evaluation.results import load_results, accuracy_latex, efficiency_latex

    results = load_results()
    print(accuracy_latex(results))
    print(efficiency_latex(efficiency_data))   # efficiency_data from efficiency.py
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import REPORTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# CSV reader
# ─────────────────────────────────────────────────────────────────────────────

def _parse_cell(cell: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse 'mean +/- std' or plain float. Returns (mean, std) or (None, None)."""
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


def load_results(csv_path: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Load model_comparison.csv into a structured dict.

    Returns:
        {
          'model_name': {
              'params':    str,
              'accuracy':  {'mean': float, 'std': float},
              'recall':    {'mean': float, 'std': float},
              'f1':        {'mean': float, 'std': float},
              'roc_auc':   {'mean': float, 'std': float},
          },
          ...
        }
    """
    path = csv_path or (REPORTS_DIR / "model_comparison.csv")
    if not path.exists():
        return {}

    results = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)   # skip header row
        for row in reader:
            if len(row) < 6:
                continue
            name   = row[0].strip()
            params = row[1].strip()
            acc_m,  acc_s  = _parse_cell(row[2])
            rec_m,  rec_s  = _parse_cell(row[3])
            f1_m,   f1_s   = _parse_cell(row[4])
            auc_m,  auc_s  = _parse_cell(row[5])

            results[name] = {
                'params':   params,
                'accuracy': {'mean': acc_m,  'std': acc_s},
                'recall':   {'mean': rec_m,  'std': rec_s},
                'f1':       {'mean': f1_m,   'std': f1_s},
                'roc_auc':  {'mean': auc_m,  'std': auc_s},
            }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table builders
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(mean: Optional[float], std: Optional[float], decimals: int = 3) -> str:
    """Format mean±std for LaTeX. Returns '--' for None values."""
    if mean is None:
        return '--'
    if std is None or std == 0:
        return f'{mean:.{decimals}f}'
    return f'${mean:.{decimals}f} \\pm {std:.{decimals}f}$'


def accuracy_latex(
    results: Dict[str, Dict],
    caption: str = 'Binary stress detection results under LOSO cross-validation on WESAD.',
    label: str = 'tab:results',
) -> str:
    """
    Generate the main accuracy comparison table as a LaTeX string.

    Columns: Model | Params | Accuracy | Recall | F1 | ROC-AUC
    """
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\begin{tabular}{lrcccc}',
        r'\toprule',
        r'Model & Params & Accuracy & Recall & F1 & ROC-AUC \\',
        r'\midrule',
    ]

    # Group separators
    baseline_names = {'Random Baseline', 'Majority Baseline', 'EDA Threshold'}
    ml_names       = {'Logistic Regression', 'Random Forest'}
    prev_group     = None

    for name, r in results.items():
        if name in baseline_names:
            group = 'baseline'
        elif name in ml_names:
            group = 'ml'
        else:
            group = 'dl'

        if prev_group is not None and group != prev_group:
            lines.append(r'\midrule')
        prev_group = group

        acc  = _fmt(r['accuracy']['mean'], r['accuracy']['std'])
        rec  = _fmt(r['recall']['mean'],   r['recall']['std'])
        f1   = _fmt(r['f1']['mean'],       r['f1']['std'])
        auc  = _fmt(r['roc_auc']['mean'],  r['roc_auc']['std'])
        params = r['params']

        # Escape underscores and special chars in model name
        safe_name = name.replace('_', r'\_').replace('&', r'\&')
        lines.append(f'{safe_name} & {params} & {acc} & {rec} & {f1} & {auc} \\\\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def efficiency_latex(
    efficiency_data: Dict[str, Dict],
    accuracy_data:   Optional[Dict[str, Dict]] = None,
    caption: str = 'Model efficiency comparison. Latency measured on CPU (single sample).',
    label: str = 'tab:efficiency',
) -> str:
    """
    Generate the efficiency comparison table as a LaTeX string.

    Columns: Model | Params | Size(KB) | Latency(ms) | FLOPs | F1
    The F1 column is populated from accuracy_data if provided.
    """
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\begin{tabular}{lrrrrl}',
        r'\toprule',
        r'Model & Params & Size (KB) & Latency (ms) & FLOPs & F1 \\',
        r'\midrule',
    ]

    for name, eff in efficiency_data.items():
        params     = f"{eff.get('params', 0):,}"
        size_kb    = f"{eff.get('size_kb', 0):.1f}"
        latency    = f"{eff.get('latency_ms', 0):.1f}"
        flops      = f"{eff['flops']:,}" if eff.get('flops') else 'N/A'

        f1_str = '--'
        if accuracy_data and name in accuracy_data:
            f1d = accuracy_data[name]['f1']
            f1_str = _fmt(f1d['mean'], f1d['std'])

        safe_name = name.replace('_', r'\_')
        lines.append(
            f'{safe_name} & {params} & {size_kb} & {latency} & {flops} & {f1_str} \\\\'
        )

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def save_latex_tables(
    output_dir: Optional[Path] = None,
    accuracy_data:   Optional[Dict] = None,
    efficiency_data: Optional[Dict] = None,
) -> None:
    """
    Write LaTeX table files to outputs/reports/.

    Files created:
        table_accuracy.tex   — main results table
        table_efficiency.tex — efficiency metrics table (if efficiency_data provided)
    """
    out = output_dir or REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    if accuracy_data is None:
        accuracy_data = load_results()

    if accuracy_data:
        tex = accuracy_latex(accuracy_data)
        path = out / 'table_accuracy.tex'
        path.write_text(tex, encoding='utf-8')
        print(f'  Saved -> {path}')

    if efficiency_data:
        tex = efficiency_latex(efficiency_data, accuracy_data)
        path = out / 'table_efficiency.tex'
        path.write_text(tex, encoding='utf-8')
        print(f'  Saved -> {path}')
