"""
Train Teacher: Entry Point for 1D-CNN Teacher Training

Usage:
    python train_teacher.py

Runs the full preprocessing pipeline, then trains the TeacherCNN
under LOSO cross-validation. Results saved to outputs/reports/model_comparison.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import RANDOM_SEED, create_directories
from utils import set_all_seeds, print_section_header
from data import load_all_subjects
from preprocessing import process_all_subjects
from segmentation import create_all_windows
from training.trainer import train_teacher_loso


def main():
    print_section_header("TEACHER CNN TRAINING PIPELINE")
    set_all_seeds(RANDOM_SEED)
    create_directories()

    subjects = load_all_subjects()
    subjects = process_all_subjects(subjects)
    windowed = create_all_windows(subjects)

    results = train_teacher_loso(windowed)

    print_section_header("DONE")
    if not results:
        print("  Training produced no results. Check that subjects have both classes.")
        return
    print(f"  Recall (mean): {results['recall']['mean']:.3f} +/- {results['recall']['std']:.3f}")
    print(f"  F1     (mean): {results['f1']['mean']:.3f} +/- {results['f1']['std']:.3f}")
    print(f"  AUC    (mean): {results['roc_auc']['mean']:.3f} +/- {results['roc_auc']['std']:.3f}")


if __name__ == "__main__":
    main()
