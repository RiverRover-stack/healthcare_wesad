"""
Configuration: Pipeline Hyperparameters and Paths

Responsibility:
    Central configuration file containing all hyperparameters,
    paths, and constants for the stress detection pipeline.

Inputs:
    None (static configuration)

Outputs:
    Constants imported by other modules:
    - Path configurations
    - Window parameters
    - Label mappings
    - Filter parameters
    - Model hyperparameters

Assumptions:
    - WESAD dataset is at d:\\WESAD\\WESAD
    - All 15 subjects (S2-S17, no S12) are available
    - Chest signals sampled at 700 Hz

Failure Modes:
    - Paths don't exist: create_directories() will create them
    - Invalid parameters: Will cause downstream errors
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(r"d:\WESAD")
DATA_DIR = PROJECT_ROOT / "WESAD"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_DIR = OUTPUT_DIR / "features"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# SUBJECTS
# =============================================================================

ALL_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9',
                'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

DEBUG_TEST_SUBJECTS = ['S15', 'S16', 'S17']
DEBUG_TRAIN_SUBJECTS = [s for s in ALL_SUBJECTS if s not in DEBUG_TEST_SUBJECTS]

# =============================================================================
# LABELS
# =============================================================================

LABEL_BASELINE = 1  # Normal class
LABEL_STRESS = 2    # Anomaly class
VALID_LABELS = [LABEL_BASELINE, LABEL_STRESS]
BINARY_LABEL_MAP = {LABEL_BASELINE: 0, LABEL_STRESS: 1}

# =============================================================================
# SAMPLING RATES
# =============================================================================

CHEST_SAMPLING_RATE = 700  # Hz

# =============================================================================
# WINDOWING
# =============================================================================

WINDOW_LENGTH_SEC = 60
WINDOW_OVERLAP = 0.5
WINDOW_STEP_SEC = WINDOW_LENGTH_SEC * (1 - WINDOW_OVERLAP)
CHEST_WINDOW_SIZE = int(WINDOW_LENGTH_SEC * CHEST_SAMPLING_RATE)
CHEST_STEP_SIZE = int(WINDOW_STEP_SEC * CHEST_SAMPLING_RATE)
PURITY_THRESHOLD = 0.8
TRANSITION_BUFFER_SEC = 5
TRANSITION_BUFFER_SAMPLES = int(TRANSITION_BUFFER_SEC * CHEST_SAMPLING_RATE)

# =============================================================================
# SIGNAL FILTERING
# =============================================================================

FILTER_PARAMS = {
    'ECG': {'type': 'bandpass', 'lowcut': 0.5, 'highcut': 40.0, 'order': 4},
    'EDA': {'type': 'lowpass', 'cutoff': 1.0, 'order': 4},
    'EMG': {'type': 'bandpass', 'lowcut': 20.0, 'highcut': 300.0, 'order': 4},
    'Resp': {'type': 'bandpass', 'lowcut': 0.1, 'highcut': 0.5, 'order': 4},
    'Temp': {'type': 'lowpass', 'cutoff': 0.1, 'order': 2},
}

FLATLINE_THRESHOLD_SEC = 1.0
SATURATION_PERCENTILE = 99.9

# =============================================================================
# UTILITY
# =============================================================================


# =============================================================================
# DEEP LEARNING CONFIGURATION
# =============================================================================
# Single source of truth for all DL hyperparameters.
# trainer.py, distillation.py, and dl_dataset.py all import from here.

DL_CONFIG = {
    # Signal resampling
    'target_sr':        64,      # Hz — 700 -> 64 Hz downsample before CNN input

    # Shared training
    'batch_size':       16,      # Small batches: total dataset is ~380 windows
    'lr':               3e-4,    # AdamW learning rate
    'weight_decay':     1e-3,    # L2 regularisation — prevents overfitting on tiny data

    # Epoch counts (teacher needs more, students converge faster)
    'teacher_epochs':   40,
    'student_epochs':   30,

    # Knowledge Distillation
    'kd_temperature':   4.0,     # Softens teacher distribution (higher = softer)
    'kd_alpha':         0.7,     # Weight on soft (teacher) targets; 1-alpha on hard labels

    # Ablation sweep ranges (used by run_ablation.py)
    'ablation_temperatures': [1, 2, 4, 8],
    'ablation_alphas':       [0.3, 0.5, 0.7, 0.9],
}


def create_directories():
    """Create all required output directories."""
    for d in [OUTPUT_DIR, FEATURES_DIR, MODELS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_subject_pkl_path(subject_id: str) -> Path:
    """Get path to a subject's pickle file."""
    return DATA_DIR / subject_id / f"{subject_id}.pkl"
