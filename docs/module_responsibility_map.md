# Module Responsibility Map

This document maps each source file to its responsibility, inputs, and outputs.

---

## Directory Structure

```
src/
├── __init__.py              # Root package
├── config.py                # Central configuration
├── utils.py                 # Shared utilities
│
├── data/                    # Data Loading
│   ├── __init__.py
│   ├── subject_data.py      # SubjectData container
│   └── loader.py            # WESAD pickle loading
│
├── preprocessing/           # Signal Conditioning
│   ├── __init__.py
│   ├── filters.py           # Butterworth filtering
│   └── processor.py         # Per-modality processing
│
├── segmentation/            # Windowing
│   ├── __init__.py
│   ├── window_data.py       # WindowInfo/WindowedData containers
│   └── windowing.py         # Sliding window creation
│
├── features/                # Feature Extraction
│   ├── __init__.py
│   ├── statistical.py       # Statistical features
│   ├── temporal.py          # Time-domain features
│   ├── frequency.py         # Spectral features
│   ├── eda.py               # EDA-specific features
│   └── extractor.py         # Main extraction pipeline
│
├── models/                  # ML Models
│   ├── __init__.py
│   ├── classifiers.py       # LogReg, RandomForest
│   └── baselines.py         # Heuristic baselines
│
└── evaluation/              # Metrics & Splitting
    ├── __init__.py
    ├── metrics.py           # Classification metrics
    └── splitting.py         # LOSO, fixed splits
```

---

## Package Summaries

| Package | Purpose | Key Exports |
|---------|---------|-------------|
| `data` | Load WESAD pickle files | `load_all_subjects`, `SubjectData` |
| `preprocessing` | Signal conditioning | `process_all_subjects`, `butter_filter` |
| `segmentation` | Create labeled windows | `create_all_windows`, `WindowedData` |
| `features` | Extract features | `extract_all_features` |
| `models` | Train/predict | `train_and_predict`, `run_all_baselines` |
| `evaluation` | Compute metrics | `loso_split`, `compute_metrics` |

---

## Data Flow

```
WESAD/*.pkl
    │
    ▼
┌─────────────────────────┐
│  data.load_all_subjects │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ preprocessing.process_all_subjects │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  segmentation.create_all_windows  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  features.extract_all_features   │
└─────────────────────────────────┘
    │
    ├──► models.run_all_baselines
    │
    ▼
┌─────────────────────────────────┐
│    evaluation.loso_split         │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   models.train_and_predict       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   evaluation.compute_metrics     │
└─────────────────────────────────┘
```

---

## Usage

```python
# Import from packages
from data import load_all_subjects
from preprocessing import process_all_subjects
from segmentation import create_all_windows
from features import extract_all_features
from models import run_all_baselines, train_and_predict
from evaluation import loso_split, compute_metrics
```

---

## Compliance

- ✅ Each file has one responsibility
- ✅ All files under 150 lines
- ✅ Proper file-level contracts
- ✅ Organized into logical packages
