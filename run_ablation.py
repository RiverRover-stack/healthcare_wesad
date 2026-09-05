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

Resuming: by default, any (temperature, alpha) configuration already present
in this output dir's ablation_results.csv is skipped -- a run killed at
config 12/16 and restarted into the same WESAD_OUTPUT_DIR picks up at 13,
not from scratch. Pass --force to ignore the existing CSV and rerun
everything (e.g. after a code change that should invalidate old results).

Prerequisite:
    Teacher checkpoints must exist:  python train_teacher.py

Outputs:
    outputs/reports/ablation_results.csv   — one row per configuration
    outputs/reports/fig7_ablation.png      — heatmap of the grid
"""

import argparse
import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from config import RANDOM_SEED, REPORTS_DIR, MODELS_DIR, create_directories, DL_CONFIG
from utils import set_all_seeds, print_section_header
from data import load_all_subjects
from preprocessing import process_all_subjects
from segmentation import create_all_windows
from data.dl_dataset import WESADDataset
from models.student import STUDENT_REGISTRY
from models.distillation import train_student_kd_loso
from training.loso import SMOKE_FOLDS, SMOKE_EPOCHS, write_manifest
from evaluation.reporter import plot_ablation
from evaluation.results import read_ablation_grid

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
    parser.add_argument('--force', action='store_true',
                        help='Ignore any existing ablation_results.csv in this output dir '
                             'and rerun every requested configuration from scratch')
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


def _hash_state_dict(state_dict: dict) -> str:
    """Order-independent hash of a model's weights, for the identical-config check below."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode())
        h.update(state_dict[key].cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _weight_signature(model_name: str, run_tag: str, test_subjects: list) -> tuple:
    """
    Hash each fold's saved checkpoint weights for one configuration.

    F1 on 15 windows saturates: two genuinely different models routinely
    produce identical F1 to full precision, which is exactly what happened
    in the smoke test (T=1,a=0.3 vs T=1,a=0.5 -- confirmed by hand to be
    different weights). Comparing weight hashes instead of the metric is
    the correct way to detect a real "the sweep isn't varying anything" bug.
    """
    hashes = []
    for subject in test_subjects:
        ckpt_path = MODELS_DIR / f"{model_name}_distilled_{run_tag}_loso_{subject}.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        hashes.append(_hash_state_dict(ckpt['model_state']))
    return tuple(hashes)


def _write_fresh_ablation_csv() -> None:
    """Wipe ablation_results.csv down to just the header (--force path only)."""
    path = REPORTS_DIR / 'ablation_results.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(ABLATION_CSV_FIELDS)


def _ensure_ablation_csv_header() -> None:
    """Create ablation_results.csv with just a header if it doesn't exist yet -- does
    not touch an existing file, so already-completed configurations survive a resume."""
    path = REPORTS_DIR / 'ablation_results.csv'
    if not path.exists():
        _write_fresh_ablation_csv()


def append_ablation_row(temperature: float, alpha: float, metrics: dict) -> None:
    """Append one configuration's aggregated results to the CSV."""
    path = REPORTS_DIR / 'ablation_results.csv'
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
        csv.DictWriter(f, fieldnames=ABLATION_CSV_FIELDS).writerow(row)


def main():
    args = parse_args()
    start_time = datetime.now(timezone.utc).isoformat()
    set_all_seeds(RANDOM_SEED)
    create_directories()

    model_name = args.model
    model_cls  = STUDENT_REGISTRY[model_name]

    configs = build_configs(args.sweep)
    if args.smoke:
        # First and last of the grid, not two adjacent points -- exercises
        # both the temperature and alpha axes instead of leaving T fixed.
        configs = [configs[0], configs[-1]] if len(configs) > 1 else configs
    max_folds = SMOKE_FOLDS if args.smoke else args.folds
    epochs    = SMOKE_EPOCHS if args.smoke else None

    # ── Resume support: skip configs already in this dir's ablation_results.csv ──
    if args.force:
        _write_fresh_ablation_csv()
        existing_grid = {}
    else:
        _ensure_ablation_csv_header()
        existing_grid = read_ablation_grid()

    already_done = [c for c in configs if c in existing_grid]
    configs_to_run = [c for c in configs if c not in existing_grid]
    if already_done:
        print(f"  Resuming: {len(already_done)} configuration(s) already in "
              f"ablation_results.csv, skipping: {already_done}")

    print_section_header(f"KD ABLATION STUDY — {model_name}{'  [SMOKE]' if args.smoke else ''}")
    print(f"  Sweep: {args.sweep}  ({len(configs_to_run)} configuration(s) to run"
          f" of {len(configs)} requested)")
    print(f"  Configs to run: {configs_to_run}")

    if not configs_to_run and not existing_grid:
        print_section_header("DONE (no results)")
        return

    # ── Load data (skipped entirely if every requested config is already done) ──
    dataset = None
    windowed = {}
    if configs_to_run:
        print_section_header("LOADING DATA")
        subjects = load_all_subjects()
        subjects = process_all_subjects(subjects)
        windowed = create_all_windows(subjects)

        print("  Building dataset once, reused across all configurations...")
        dataset = WESADDataset(windowed)

    all_subjects = sorted(windowed.keys())
    folds_to_run = all_subjects[:max_folds] if max_folds else all_subjects

    # ── Sweep ──────────────────────────────────────────────────────────────────
    grid_results = {}      # {(temperature, alpha): f1_mean} -- this invocation only
    weight_signatures = {}  # {(temperature, alpha): tuple of per-fold weight hashes}

    for temperature, alpha in configs_to_run:
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
        weight_signatures[(temperature, alpha)] = _weight_signature(model_name, run_tag, folds_to_run)
        print(f"    -> F1={f1:.4f}")
        append_ablation_row(temperature, alpha, metrics)

    # F1 saturates on small fold sizes -- two genuinely different models can
    # land on identical F1. The real "sweep isn't doing anything" signal is
    # identical *weights* across configs run in this invocation, not identical F1.
    if len(weight_signatures) > 1 and len(set(weight_signatures.values())) == 1:
        print("\n  WARNING: all configurations produced IDENTICAL model weights.")
        print("  This means the sweep is still not varying hyperparameters -- stop and report.")
    elif len(grid_results) > 1 and len(set(grid_results.values())) == 1:
        print("\n  Note: all configurations produced the same F1, but weight hashes differ")
        print("  (metric saturation on this many folds/windows) -- not a bug.")

    print_section_header("GENERATING ABLATION PLOT")
    # Read back from disk rather than plotting grid_results directly -- the
    # figure must reflect exactly what's in the CSV (this run's new rows plus
    # any resumed-from-existing ones), or the two can drift apart (the
    # original failure mode: figures and CSVs from different runs).
    csv_grid = read_ablation_grid()
    if not csv_grid:
        print_section_header("DONE (no results)")
        return

    plot_ablation(
        csv_grid,
        temperatures=DL_CONFIG['ablation_temperatures'],
        alphas=DL_CONFIG['ablation_alphas'],
        model_name=model_name,
    )

    write_manifest('ablation', start_time, extra={
        'smoke': args.smoke, 'sweep': args.sweep, 'model': model_name,
        'n_configs_run_this_invocation': len(grid_results),
        'n_configs_total_in_csv': len(csv_grid),
    })

    print_section_header("DONE")


if __name__ == '__main__':
    main()
