"""
Ablation Study: KD Hyperparameter Sweep

Sweeps KD temperature and alpha on a student model as a real 4x4 grid
(T in {1,2,4,8} x alpha in {0.3,0.5,0.7,0.9} = 16 configurations by default),
showing sensitivity analysis. Results are the key evidence for the paper's
hyperparameter section.

Usage:
    python run_ablation.py                       # full 4x4 grid (16 configs)
    python run_ablation.py --sweep temperature   # temperature only, alpha fixed (4 configs)
    python run_ablation.py --sweep alpha         # alpha only, temperature fixed (4 configs)
    python run_ablation.py --model TinyCNN       # use a different student
    python run_ablation.py --smoke               # 2 configs x 2 folds x 3 epochs
    python run_ablation.py --folds 5             # limit every config to 5 LOSO folds

Prerequisite:
    Teacher checkpoints must exist:  python train_teacher.py

Outputs:
    outputs/reports/ablation_results.csv   — one row per configuration
    outputs/reports/fig7_ablation.png      — heatmap of the grid
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
from data.dl_dataset import WESADDataset
from models.student import STUDENT_REGISTRY
from models.distillation import train_student_kd_loso
from training.loso import SMOKE_FOLDS, SMOKE_EPOCHS
from evaluation.reporter import plot_ablation

ABLATION_CSV_FIELDS = [
    'temperature', 'alpha', 'f1_mean', 'f1_std', 'accuracy_mean', 'recall_mean', 'auc_mean',
]


def parse_args():
    parser = argparse.ArgumentParser(description='KD hyperparameter ablation study')
    parser.add_argument('--sweep', choices=['temperature', 'alpha', 'both'],
                        default='both',
                        help="'both' = full 4x4 grid; otherwise vary one param at the other's default")
    parser.add_argument('--model', choices=list(STUDENT_REGISTRY.keys()),
                        default='MicroCNN',
                        help='Student model to use for ablation (default: MicroCNN)')
    parser.add_argument('--smoke', action='store_true',
                        help='Run only the first 2 configs, 2 folds, 3 epochs each')
    parser.add_argument('--folds', type=int, default=None,
                        help='Limit every configuration to the first N LOSO folds (default: all 15)')
    return parser.parse_args()


def build_configs(sweep: str) -> list:
    """Return the list of (temperature, alpha) configurations to run."""
    temps  = DL_CONFIG['ablation_temperatures']
    alphas = DL_CONFIG['ablation_alphas']
    fixed_t = DL_CONFIG['kd_temperature']
    fixed_a = DL_CONFIG['kd_alpha']

    if sweep == 'both':
        return [(t, a) for t in temps for a in alphas]
    if sweep == 'temperature':
        return [(t, fixed_a) for t in temps]
    return [(fixed_t, a) for a in alphas]  # sweep == 'alpha'


def append_ablation_row(temperature: float, alpha: float, metrics: dict) -> None:
    """Write one configuration's aggregated results to the CSV immediately."""
    path = REPORTS_DIR / 'ablation_results.csv'
    write_header = not path.exists()
    row = {
        'temperature': temperature,
        'alpha': alpha,
        'f1_mean': metrics['f1']['mean'],
        'f1_std': metrics['f1']['std'],
        'accuracy_mean': metrics['accuracy']['mean'],
        'recall_mean': metrics['recall']['mean'],
        'auc_mean': metrics['roc_auc']['mean'],
    }
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ABLATION_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    set_all_seeds(RANDOM_SEED)
    create_directories()

    model_name = args.model
    model_cls  = STUDENT_REGISTRY[model_name]

    configs = build_configs(args.sweep)
    if args.smoke:
        configs = configs[:2]
    max_folds = SMOKE_FOLDS if args.smoke else args.folds
    epochs    = SMOKE_EPOCHS if args.smoke else None

    print_section_header(f"KD ABLATION STUDY — {model_name}{'  [SMOKE]' if args.smoke else ''}")
    print(f"  Sweep: {args.sweep}  ({len(configs)} configurations)")
    print(f"  Configs: {configs}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print_section_header("LOADING DATA")
    subjects = load_all_subjects()
    subjects = process_all_subjects(subjects)
    windowed = create_all_windows(subjects)

    print("  Building dataset once, reused across all configurations...")
    dataset = WESADDataset(windowed)

    # ── Sweep ──────────────────────────────────────────────────────────────────
    grid_results = {}  # {(temperature, alpha): f1_mean}

    for temperature, alpha in configs:
        print_section_header(f"T={temperature}  alpha={alpha}")
        run_tag = f"T{temperature}_a{alpha}"
        metrics = train_student_kd_loso(
            windowed, model_cls, model_name, mode='distilled',
            dataset=dataset, temperature=temperature, alpha=alpha,
            epochs=epochs, max_folds=max_folds, run_tag=run_tag,
        )
        if not metrics:
            print(f"    -> no folds completed, skipping")
            continue

        f1 = metrics['f1']['mean']
        grid_results[(temperature, alpha)] = f1
        print(f"    -> F1={f1:.4f}")
        append_ablation_row(temperature, alpha, metrics)

    if not grid_results:
        print_section_header("DONE (no results)")
        return

    if len(grid_results) > 1 and len(set(grid_results.values())) == 1:
        print("\n  WARNING: all configurations produced the identical F1 score.")
        print("  This indicates the sweep is still not varying hyperparameters -- stop and report.")

    print_section_header("GENERATING ABLATION PLOT")
    plot_ablation(
        grid_results,
        temperatures=DL_CONFIG['ablation_temperatures'],
        alphas=DL_CONFIG['ablation_alphas'],
        model_name=model_name,
    )

    print_section_header("DONE")


if __name__ == '__main__':
    main()
