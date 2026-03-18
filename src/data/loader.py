"""
Data Loader: Load and Validate WESAD Pickle Files

Responsibility:
    Load WESAD pickle files and curate labels for binary classification.

Inputs:
    Subject IDs (list of strings like 'S2', 'S3')

Outputs:
    Dictionary mapping subject_id -> SubjectData object

Assumptions:
    - Pickle files use 'latin1' encoding
    - Structure: {'signal': {...}, 'label': array, 'subject': str}

Failure Modes:
    - Missing pickle file: FileNotFoundError
    - Invalid structure: ValueError
"""

import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, ALL_SUBJECTS, LABEL_BASELINE, LABEL_STRESS, VALID_LABELS
from .subject_data import SubjectData
from utils import print_section_header, format_percentage


def load_subject_pickle(subject_id: str) -> Dict:
    """Load a subject's pickle file with error handling."""
    pkl_path = DATA_DIR / subject_id / f"{subject_id}.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

    print(f"  Loading {subject_id}...", end=" ")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    for key in ['signal', 'label', 'subject']:
        if key not in data:
            raise ValueError(f"{subject_id}: Missing key '{key}'")

    print(f"OK ({len(data['label']):,} samples)")
    return data


def curate_labels(labels_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Filter labels to baseline/stress and create binary labels."""
    valid_mask = np.isin(labels_raw, VALID_LABELS)
    binary_labels = np.zeros(len(labels_raw), dtype=np.int32)
    binary_labels[labels_raw == LABEL_STRESS] = 1
    return binary_labels, valid_mask


def load_single_subject(subject_id: str) -> SubjectData:
    """Load and curate data for a single subject."""
    data = load_subject_pickle(subject_id)
    chest = data['signal']['chest']
    labels_raw = data['label'].flatten().astype(np.int32)
    labels_binary, valid_mask = curate_labels(labels_raw)

    return SubjectData(
        subject_id=subject_id,
        chest_ecg=chest.get('ECG', np.array([])).flatten(),
        chest_eda=chest.get('EDA', np.array([])).flatten(),
        chest_emg=chest.get('EMG', np.array([])).flatten(),
        chest_resp=chest.get('Resp', np.array([])).flatten(),
        chest_temp=chest.get('Temp', np.array([])).flatten(),
        chest_acc=chest.get('ACC', np.array([])),
        labels_raw=labels_raw,
        labels_binary=labels_binary,
        valid_mask=valid_mask,
    )


def load_all_subjects(subject_ids: Optional[List[str]] = None) -> Dict[str, SubjectData]:
    """Load and curate data for multiple subjects."""
    if subject_ids is None:
        subject_ids = ALL_SUBJECTS

    print_section_header("PHASE 2: DATA LOADING AND LABEL CURATION")
    print(f"Loading {len(subject_ids)} subjects from: {DATA_DIR}")

    subjects = {}
    total_valid, total_baseline, total_stress = 0, 0, 0

    for subject_id in subject_ids:
        try:
            sd = load_single_subject(subject_id)
            subjects[subject_id] = sd
            total_valid += sd.num_valid_samples
            total_baseline += sd.baseline_samples
            total_stress += sd.stress_samples
        except Exception as e:
            print(f"  ERROR loading {subject_id}: {e}")

    print(f"\n[SUMMARY] {len(subjects)} subjects, {total_valid:,} valid samples")
    print(f"  Baseline: {total_baseline:,} ({format_percentage(total_baseline/total_valid)})")
    print(f"  Stress: {total_stress:,} ({format_percentage(total_stress/total_valid)})")

    return subjects
