"""
Ablation Study: KD Hyperparameter Sweep

Sweeps KD temperature and alpha on MicroCNN to show sensitivity analysis.
Results are the key evidence for the paper's hyperparameter section.

Usage:
    python run_ablation.py                       # full sweep (temperature + alpha)
    python run_ablation.py --sweep temperature   # temperature only
    python run_ablation.py --sweep alpha         # alpha only
    python run_ablation.py --model TinyCNN       # use a different student

Prerequisite:
    Teacher checkpoints must exist:  python train_teacher.py

Outputs:
    outputs/reports/ablation_results.csv   — numerical results
    outputs/reports/fig7_ablation.png      — plot of both sweeps
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import RANDOM_SEED, REPORTS_DIR, create_directories, DL_CONFIG
from utils import set_all_seeds, print_section_header
from data import load_all_subjects
from preprocessing import process_all_subjects
from segmentation import create_all_windows
from models.student import STUDENT_REGISTRY
from models.distillation import train_student_kd_loso, KD_TEMPERATURE, KD_ALPHA
from evaluation.reporter import plot_ablation


def parse_args():
    parser = argparse.ArgumentParser(description='KD hyperparameter ablation study')
    parser.add_argument('--sweep', choices=['temperature', 'alpha', 'both'],
                        default='both')
    parser.add_argument('--model', choices=list(STUDENT_REGISTRY.keys()),
                        default='MicroCNN',
                        help='Student model to use for ablation (default: MicroCNN)')
    return parser.parse_args()


def run_sweep(windowed, model_name, model_cls, param_name, values, fixed_params):
    """
    Run one sweep: vary one KD parameter across `values`, hold others fixed.

    Returns:
        {value: f1_mean}  — F1 score for each parameter value
    """
    results = {}
    for val in values:
        print(f"\n  [{param_name}={val}]")

        # Temporarily override DL_CONFIG for this run
        import src.models.distillation as dist_module
        if param_name == 'temperature':
            dist_module.KD_TEMPERATURE = val
            dist_module.KD_ALPHA       = fixed_params['alpha']
        else:
            dist_module.KD_TEMPERATURE = fixed_params['temperature']
            dist_module.KD_ALPHA       = val

        metrics = train_student_kd_loso(
            windowed, model_cls, f'{model_name}_ablation', mode='distilled'
        )
        f1 = metrics.get('f1', {}).get('mean', 0.0) if metrics else 0.0
        results[val] = f1
        print(f"    -> F1={f1:.4f}")

    # Restore defaults
    import src.models.distillation as dist_module
    dist_module.KD_TEMPERATURE = KD_TEMPERATURE
    dist_module.KD_ALPHA       = KD_ALPHA

    return results


def save_ablation_csv(temp_results, alpha_results):
    """Write ablation results to CSV."""
    path = REPORTS_DIR / 'ablation_results.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sweep', 'value', 'f1'])
        for t, f1 in sorted(temp_results.items()):
            writer.writerow(['temperature', t, f'{f1:.4f}'])
        for a, f1 in sorted(alpha_results.items()):
            writer.writerow(['alpha', a, f'{f1:.4f}'])
    print(f"\n  Ablation results saved -> {path}")


def main():
    args = parse_args()
    set_all_seeds(RANDOM_SEED)
    create_directories()

    model_name = args.model
    model_cls  = STUDENT_REGISTRY[model_name]

    print_section_header(f"KD ABLATION STUDY — {model_name}")
    print(f"  Sweep: {args.sweep}")
    print(f"  Temperature values : {DL_CONFIG['ablation_temperatures']}")
    print(f"  Alpha values       : {DL_CONFIG['ablation_alphas']}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print_section_header("LOADING DATA")
    subjects = load_all_subjects()
    subjects = process_all_subjects(subjects)
    windowed = create_all_windows(subjects)

    fixed = {
        'temperature': DL_CONFIG['kd_temperature'],
        'alpha':       DL_CONFIG['kd_alpha'],
    }

    temp_results  = {}
    alpha_results = {}

    # ── Temperature sweep ──────────────────────────────────────────────────────
    if args.sweep in ('temperature', 'both'):
        print_section_header("TEMPERATURE SWEEP")
        temp_results = run_sweep(
            windowed, model_name, model_cls,
            param_name='temperature',
            values=DL_CONFIG['ablation_temperatures'],
            fixed_params=fixed,
        )
        best_t = max(temp_results, key=temp_results.get)
        print(f"\n  Best T={best_t}  F1={temp_results[best_t]:.4f}")

    # ── Alpha sweep ────────────────────────────────────────────────────────────
    if args.sweep in ('alpha', 'both'):
        print_section_header("ALPHA SWEEP")
        alpha_results = run_sweep(
            windowed, model_name, model_cls,
            param_name='alpha',
            values=DL_CONFIG['ablation_alphas'],
            fixed_params=fixed,
        )
        best_a = max(alpha_results, key=alpha_results.get)
        print(f"\n  Best alpha={best_a}  F1={alpha_results[best_a]:.4f}")

    # ── Save results ───────────────────────────────────────────────────────────
    if temp_results or alpha_results:
        save_ablation_csv(temp_results, alpha_results)
        print_section_header("GENERATING ABLATION PLOT")
        plot_ablation(temp_results, alpha_results)
        print("  Saved -> outputs/reports/fig7_ablation.png")

    print_section_header("DONE")


if __name__ == '__main__':
    main()
