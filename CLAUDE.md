# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Binary stress detection pipeline using the WESAD dataset (15 subjects, chest-worn RespiBAN sensors at 700 Hz). The pipeline goes: load pickles → preprocess signals → segment into windows → extract hand-crafted features → train classifiers → LOSO cross-validation.

## Commands

### Run the full pipeline
```bash
python main.py
```

### Run the EDA visualization
```bash
python eda_visualization.py
# Output: outputs/eda_visualization.png
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Data Flow
```
WESAD/{S2..S17}/S{id}.pkl
    → src/data/loader.py          # load_all_subjects() → List[SubjectData]
    → src/preprocessing/          # Butterworth filtering + z-score per subject
    → src/segmentation/           # 60s sliding windows, 50% overlap, 80% purity threshold
    → src/features/extractor.py   # ~150 hand-crafted features per window
    → src/models/classifiers.py   # LogisticRegression & RandomForest (class_weight='balanced')
    → src/evaluation/             # LOSO cross-validation, aggregate metrics
```

### Key Configuration (`src/config.py`)
All pipeline parameters are centralized here. Important values:
- `DATA_DIR`: Points to `d:\WESAD\WESAD\` (raw dataset, ~15 GB, not in git)
- `ALL_SUBJECTS`: 15 subjects S2–S17 (S12 excluded — corrupted)
- `CHEST_SAMPLING_RATE = 700` Hz
- `WINDOW_LENGTH_SEC = 60`, `WINDOW_OVERLAP = 0.5`, `PURITY_THRESHOLD = 0.8`
- `TRANSITION_BUFFER_SEC = 5` — discards gradual state-change boundaries
- `RANDOM_SEED = 42`
- `FILTER_PARAMS`: Per-signal Butterworth specs (ECG bandpass 0.5–40 Hz, EDA lowpass 1 Hz, etc.)

### Module Responsibilities
| Module | Responsibility |
|--------|---------------|
| `src/data/loader.py` | Load `.pkl` files with `encoding='latin1'`; returns `SubjectData` containers |
| `src/data/subject_data.py` | `SubjectData` dataclass holding `chest`, `wrist`, `labels`, `subject_id` |
| `src/preprocessing/filters.py` | `scipy.signal.butter` + `filtfilt` Butterworth filters |
| `src/preprocessing/processor.py` | Per-modality filtering → z-score normalization per subject |
| `src/segmentation/windowing.py` | Sliding windows; rejects windows with <80% label purity |
| `src/features/` | `statistical.py`, `temporal.py`, `frequency.py`, `eda.py` — each adds features to windows |
| `src/features/extractor.py` | Orchestrates feature extraction across all signal modalities |
| `src/models/classifiers.py` | `create_logistic_regression()` (L2, C=1.0), `create_random_forest()` (100 trees, depth=10) |
| `src/models/baselines.py` | Heuristic baselines (EDA threshold-based) |
| `src/evaluation/splitting.py` | `loso_split()` generator — yields (train_idx, test_idx) per subject fold |
| `src/evaluation/metrics.py` | Per-fold accuracy/precision/recall/F1/ROC-AUC; `aggregate_fold_metrics()` for mean±std |

### Binary Classification Task
Labels in the raw pickle: 0=unlabeled, 1=baseline, 2=stress, 3=amusement, 4=meditation, 5-7=other. The pipeline keeps only labels 1 and 2, remapping baseline→0, stress→1.

### Signals Used
Only chest signals (6 modalities @ 700 Hz): ECG, EDA, EMG, Resp, Temp, ACC (3-axis). Wrist signals (Empatica E4) are loaded but not used in the current implementation.

## Dataset
Raw data lives in `WESAD/` (gitignored, ~15 GB). Each subject folder `S{id}/` contains `S{id}.pkl` (~975 MB). The dataset is not redistributable; see the WESAD paper for access.

Known issue: some subjects have NaN values in respiration signal — handled in preprocessing.

## Documentation
- `docs/decision_log.md` — rationale for windowing, filter, and modeling choices
- `docs/known_unknowns.md` — open questions and unvalidated assumptions
- `docs/technical_report.md` — full methodology and results
- `WESAD_Learning_Guide.md` — practical guide to the data and pipeline challenges
