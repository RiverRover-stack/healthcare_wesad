import os
import sys
import tempfile
from pathlib import Path

# config.py reads WESAD_OUTPUT_DIR at import time; set it before any test
# module can import config, so training smoke tests never write into the
# real outputs/ directory.
os.environ.setdefault("WESAD_OUTPUT_DIR", tempfile.mkdtemp(prefix="wesad_pytest_outputs_"))

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

DEFAULT_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9',
                     'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']


def fake_index(all_subjects=None, n_per_subject=20, single_class_subjects=None):
    """
    Synthesize (subject_per_window, label_per_window) arrays for testing
    fold-splitting logic without touching torch, WESADDataset, or the real
    17 GB pickle dataset.

    single_class_subjects: optional {subject: label} forcing every window
    of that subject to carry only `label` (0 or 1), to exercise the
    single-class skip/advance paths in make_fold_split.
    """
    all_subjects = all_subjects if all_subjects is not None else DEFAULT_SUBJECTS
    single_class_subjects = single_class_subjects or {}

    subjects, labels = [], []
    rng = np.random.RandomState(0)
    for sid in all_subjects:
        subjects.extend([sid] * n_per_subject)
        if sid in single_class_subjects:
            labels.extend([single_class_subjects[sid]] * n_per_subject)
        else:
            half = n_per_subject // 2
            sid_labels = [0] * half + [1] * (n_per_subject - half)
            rng.shuffle(sid_labels)
            labels.extend(sid_labels)

    return np.array(subjects), np.array(labels)
