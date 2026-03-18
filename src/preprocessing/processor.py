"""
Signal Processor: Per-Modality Signal Conditioning

Responsibility:
    Apply task-specific signal conditioning for each sensor modality.

Inputs:
    Raw signal arrays from SubjectData

Outputs:
    Dictionary of processed (filtered, normalized) signal arrays

Assumptions:
    - Chest signals at 700 Hz
    - ECG, EDA, EMG, Resp, Temp require different filtering

Failure Modes:
    - NaN in signals: Replaced with median
"""

import numpy as np
from typing import Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHEST_SAMPLING_RATE, FILTER_PARAMS
from .filters import butter_filter, z_score_normalize
from utils import print_section_header


def _handle_nan(signal: np.ndarray) -> np.ndarray:
    """Replace NaN with median."""
    if np.any(np.isnan(signal)):
        signal = np.nan_to_num(signal, nan=np.nanmedian(signal))
    return signal


def process_ecg(signal: np.ndarray, fs: float = CHEST_SAMPLING_RATE) -> np.ndarray:
    """Process ECG: bandpass 0.5-40 Hz, z-score normalize."""
    signal = _handle_nan(signal)
    params = FILTER_PARAMS['ECG']
    filtered = butter_filter(signal, params['type'], fs,
                             lowcut=params['lowcut'], highcut=params['highcut'],
                             order=params['order'])
    return z_score_normalize(filtered)


def process_eda(signal: np.ndarray, fs: float = CHEST_SAMPLING_RATE) -> np.ndarray:
    """Process EDA: lowpass 1 Hz, robust z-score normalize."""
    signal = _handle_nan(signal)
    params = FILTER_PARAMS['EDA']
    filtered = butter_filter(signal, params['type'], fs,
                             cutoff=params['cutoff'], order=params['order'])
    return z_score_normalize(filtered, robust=True)


def process_emg(signal: np.ndarray, fs: float = CHEST_SAMPLING_RATE) -> np.ndarray:
    """Process EMG: bandpass 20-300 Hz, z-score normalize."""
    signal = _handle_nan(signal)
    params = FILTER_PARAMS['EMG']
    filtered = butter_filter(signal, params['type'], fs,
                             lowcut=params['lowcut'], highcut=params['highcut'],
                             order=params['order'])
    return z_score_normalize(filtered)


def process_resp(signal: np.ndarray, fs: float = CHEST_SAMPLING_RATE) -> np.ndarray:
    """Process Resp: bandpass 0.1-0.5 Hz, z-score normalize."""
    signal = _handle_nan(signal)
    params = FILTER_PARAMS['Resp']
    filtered = butter_filter(signal, params['type'], fs,
                             lowcut=params['lowcut'], highcut=params['highcut'],
                             order=params['order'])
    return z_score_normalize(filtered)


def process_temp(signal: np.ndarray, fs: float = CHEST_SAMPLING_RATE) -> np.ndarray:
    """Process Temp: lowpass 0.1 Hz, robust z-score normalize."""
    signal = _handle_nan(signal)
    params = FILTER_PARAMS['Temp']
    filtered = butter_filter(signal, params['type'], fs,
                             cutoff=params['cutoff'], order=params['order'])
    return z_score_normalize(filtered, robust=True)


def process_acc(signal: np.ndarray) -> np.ndarray:
    """Process ACC: compute magnitude, z-score normalize."""
    if signal.ndim > 1:
        signal = np.sqrt(np.sum(signal ** 2, axis=1))
    return z_score_normalize(_handle_nan(signal))


def process_subject_signals(subject_data) -> Dict[str, np.ndarray]:
    """Process all signals for a subject."""
    processed = {}
    if len(subject_data.chest_ecg) > 0:
        processed['chest_ecg'] = process_ecg(subject_data.chest_ecg)
    if len(subject_data.chest_eda) > 0:
        processed['chest_eda'] = process_eda(subject_data.chest_eda)
    if len(subject_data.chest_emg) > 0:
        processed['chest_emg'] = process_emg(subject_data.chest_emg)
    if len(subject_data.chest_resp) > 0:
        processed['chest_resp'] = process_resp(subject_data.chest_resp)
    if len(subject_data.chest_temp) > 0:
        processed['chest_temp'] = process_temp(subject_data.chest_temp)
    if len(subject_data.chest_acc) > 0:
        processed['chest_acc'] = process_acc(subject_data.chest_acc)
    return processed


def process_all_subjects(subjects: Dict) -> Dict:
    """Process signals for all subjects."""
    print_section_header("PHASE 2.5: SIGNAL CONDITIONING")

    for subject_id, subject_data in subjects.items():
        print(f"  Processing {subject_id}...", end=" ")
        subject_data.processed_signals = process_subject_signals(subject_data)
        print("OK")

    print(f"\n  Conditioning complete for {len(subjects)} subjects")
    return subjects
