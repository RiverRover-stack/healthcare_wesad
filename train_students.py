"""
Train Students: Entry Point for All Student Model Training

Usage:
    python train_students.py              # train all 3 models x 2 modes = 6 runs
    python train_students.py --model MicroCNN          # single model, both modes
    python train_students.py --mode distilled          # all models, distilled only
    python train_students.py --model TinyCNN --mode standalone

Prerequisite:
    Teacher checkpoints must exist in outputs/models/ before running distilled mode.
    Run: python train_teacher.py

Outputs:
    outputs/models/{model}_{mode}_loso_{subject}.pt  — per-fold checkpoints
    outputs/reports/model_comparison.csv             — updated with student results
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import RANDOM_SEED, create_directories
from utils import set_all_seeds, print_section_header
from data import load_all_subjects
from preprocessing import process_all_subjects
from segmentation import create_all_windows
from models.student import STUDENT_REGISTRY
from models.distillation import train_student_kd_loso
from evaluation.efficiency import get_efficiency_report, print_efficiency_table
from evaluation.results import save_latex_tables


def parse_args():
    parser = argparse.ArgumentParser(description='Train student models via KD')
    parser.add_argument('--model', choices=list(STUDENT_REGISTRY.keys()) + ['all'],
                        default='all', help='Which student to train (default: all)')
    parser.add_argument('--mode',  choices=['standalone', 'distilled', 'both'],
                        default='both', help='Training mode (default: both)')
    parser.add_argument('--skip-efficiency', action='store_true',
                        help='Skip efficiency benchmarking after training')
    return parser.parse_args()


def main():
    args = parse_args()
    set_all_seeds(RANDOM_SEED)
    create_directories()

    # ── Determine which models and modes to run ────────────────────────────────
    if args.model == 'all':
        models_to_run = list(STUDENT_REGISTRY.items())
    else:
        models_to_run = [(args.model, STUDENT_REGISTRY[args.model])]

    if args.mode == 'both':
        modes_to_run = ['standalone', 'distilled']
    else:
        modes_to_run = [args.mode]

    total_runs = len(models_to_run) * len(modes_to_run)
    print_section_header("STUDENT MODEL TRAINING PIPELINE")
    print(f"  Models : {[n for n, _ in models_to_run]}")
    print(f"  Modes  : {modes_to_run}")
    print(f"  Total  : {total_runs} training runs x 15 LOSO folds each")

    # ── Load & preprocess data (same as train_teacher.py) ─────────────────────
    print_section_header("LOADING DATA")
    subjects = load_all_subjects()
    subjects = process_all_subjects(subjects)
    windowed = create_all_windows(subjects)

    # ── Training loop ──────────────────────────────────────────────────────────
    all_results = {}   # {(model_name, mode): aggregated_metrics}

    for model_name, model_cls in models_to_run:
        for mode in modes_to_run:
            print_section_header(f"{model_name} [{mode.upper()}]")
            results = train_student_kd_loso(windowed, model_cls, model_name, mode=mode)
            if results:
                all_results[(model_name, mode)] = results

    # ── Summary table ──────────────────────────────────────────────────────────
    print_section_header("TRAINING COMPLETE — SUMMARY")
    print(f"\n  {'Model':<20} {'Mode':<12} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("  " + "-" * 60)
    for (name, mode), r in all_results.items():
        print(f"  {name:<20} {mode:<12}"
              f"  {r['recall']['mean']:.3f}"
              f"  {r['f1']['mean']:.3f}"
              f"  {r['roc_auc']['mean']:.3f}")

    # ── Efficiency benchmarks ──────────────────────────────────────────────────
    if not args.skip_efficiency:
        print_section_header("EFFICIENCY BENCHMARKS")
        from models.teacher import create_teacher_cnn
        bench_models = {'Teacher (Multi-Scale)': create_teacher_cnn()}
        for name, cls in STUDENT_REGISTRY.items():
            bench_models[name] = cls()
        eff_results = {}
        for mname, model in bench_models.items():
            from evaluation.efficiency import get_efficiency_report
            eff_results[mname] = get_efficiency_report(model)
        print_efficiency_table(eff_results)

        # ── Generate LaTeX tables ──────────────────────────────────────────────
        from evaluation.results import load_results
        acc_results = load_results()
        save_latex_tables(accuracy_data=acc_results, efficiency_data=eff_results)
        print("\n  LaTeX tables saved to outputs/reports/")

    print_section_header("DONE")
    print("  Next steps:")
    print("    python generate_report.py  -- update all figures with new results")
    print("    python run_ablation.py     -- run KD hyperparameter ablation")


if __name__ == '__main__':
    main()
