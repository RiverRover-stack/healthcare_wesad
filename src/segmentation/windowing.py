"""
Windowing: Sliding Window Segmentation and Label Assignment

Responsibility:
    Create fixed-duration windows with proper label assignment.

Inputs:
    Processed signals and raw labels from SubjectData

Outputs:
    WindowedData objects containing signal windows and binary labels

Assumptions:
    - Window length = 60 seconds, overlap = 50%
    - Purity threshold = 80% for label assignment

Failure Modes:
    - All windows rejected: Empty WindowedData returned
"""

import numpy as np
from typing import Dict, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CHEST_SAMPLING_RATE, CHEST_WINDOW_SIZE, CHEST_STEP_SIZE,
    PURITY_THRESHOLD, TRANSITION_BUFFER_SAMPLES,
    LABEL_BASELINE, LABEL_STRESS, WINDOW_LENGTH_SEC, WINDOW_OVERLAP
)
from .window_data import WindowInfo, WindowedData
from utils import print_section_header, format_percentage


def find_label_transitions(labels: np.ndarray) -> np.ndarray:
    """Find sample indices where labels change."""
    return np.where(np.diff(labels) != 0)[0] + 1


def is_near_transition(start: int, end: int, transitions: np.ndarray, buffer: int) -> bool:
    """Check if window overlaps with transition buffer zone."""
    for trans in transitions:
        if start < trans + buffer and end > trans - buffer:
            return True
    return False


def compute_label_purity(labels_window: np.ndarray) -> Tuple[int, float]:
    """Compute majority label and purity for a window."""
    valid_mask = (labels_window == LABEL_BASELINE) | (labels_window == LABEL_STRESS)
    valid_labels = labels_window[valid_mask]

    if len(valid_labels) == 0:
        return -1, 0.0

    baseline_count = np.sum(valid_labels == LABEL_BASELINE)
    stress_count = np.sum(valid_labels == LABEL_STRESS)
    total = baseline_count + stress_count

    if baseline_count >= stress_count:
        return LABEL_BASELINE, baseline_count / total
    return LABEL_STRESS, stress_count / total


def create_windows_for_subject(subject_data, processed: Dict[str, np.ndarray]) -> WindowedData:
    """Create sliding windows for a single subject."""
    labels_raw = subject_data.labels_raw
    transitions = find_label_transitions(labels_raw)

    all_windows = {sig: [] for sig in processed.keys()}
    all_labels, all_info = [], []

    start = 0
    while start + CHEST_WINDOW_SIZE <= len(labels_raw):
        end = start + CHEST_WINDOW_SIZE
        window_labels = labels_raw[start:end]
        majority_label, purity = compute_label_purity(window_labels)

        is_valid = True
        reason = None

        if majority_label not in [LABEL_BASELINE, LABEL_STRESS]:
            is_valid, reason = False, "invalid_label"
        elif purity < PURITY_THRESHOLD:
            is_valid, reason = False, f"low_purity_{purity:.2f}"
        elif is_near_transition(start, end, transitions, TRANSITION_BUFFER_SAMPLES):
            is_valid, reason = False, "near_transition"

        if is_valid:
            for sig_name, sig_data in processed.items():
                all_windows[sig_name].append(sig_data[start:end])

            binary_label = 0 if majority_label == LABEL_BASELINE else 1
            all_labels.append(binary_label)
            all_info.append(WindowInfo(
                subject_id=subject_data.subject_id, window_idx=len(all_labels)-1,
                start_sample=start, end_sample=end,
                start_time_sec=start/CHEST_SAMPLING_RATE, end_time_sec=end/CHEST_SAMPLING_RATE,
                label_binary=binary_label, label_purity=purity, is_valid=True
            ))

        start += CHEST_STEP_SIZE

    windows_dict = {k: np.array(v) for k, v in all_windows.items() if len(v) > 0}
    labels_array = np.array(all_labels, dtype=np.int32)

    return WindowedData(
        subject_id=subject_data.subject_id, windows=windows_dict, labels=labels_array,
        window_info=all_info, num_windows=len(all_labels),
        num_baseline=int(np.sum(labels_array == 0)), num_stress=int(np.sum(labels_array == 1))
    )


def create_all_windows(subjects: Dict) -> Dict[str, WindowedData]:
    """Create windows for all subjects."""
    print_section_header("PHASE 3: WINDOWING AND SYNCHRONIZATION")
    print(f"  Window: {WINDOW_LENGTH_SEC}s, Overlap: {WINDOW_OVERLAP*100}%")

    all_windowed = {}
    total_windows, total_baseline, total_stress = 0, 0, 0

    for subject_id, sd in subjects.items():
        if not hasattr(sd, 'processed_signals') or not sd.processed_signals:
            print(f"  {subject_id}: SKIP (no processed signals)")
            continue

        wd = create_windows_for_subject(sd, sd.processed_signals)
        all_windowed[subject_id] = wd
        total_windows += wd.num_windows
        total_baseline += wd.num_baseline
        total_stress += wd.num_stress
        print(f"  {subject_id}: {wd.num_windows} windows (B={wd.num_baseline}, S={wd.num_stress})")

    print(f"\n[SUMMARY] {total_windows} windows, ratio={total_baseline/max(total_stress,1):.2f}:1")
    return all_windowed
