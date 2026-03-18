# Binary Stress Detection Using Wearable Physiological Signals

**Project**: WESAD-Based Stress Anomaly Detection  
**Status**: Active Development  
**Last Updated**: February 12, 2026

---

## Overview

This project builds a **machine learning pipeline for automatic stress detection** from physiological signals captured by chest-worn wearable devices. Using the [WESAD (Wearable Stress and Affect Detection)](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29) dataset, the system classifies a person's state as either **baseline (relaxed)** or **stressed** — enabling applications in workplace wellness, mental health monitoring, and preventive healthcare.

### Why This Matters

Chronic stress affects approximately 77% of adults (APA), contributing to cardiovascular disease, mental health disorders, and diminished quality of life. Despite this, most individuals lack objective tools to monitor stress in real-time. This project addresses that gap by developing a reliable, interpretable stress classifier built on well-established physiological biomarkers.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **End-to-End Pipeline** | Data loading → signal filtering → windowing → feature extraction → model training → LOSO evaluation |
| **Multi-Signal Analysis** | Processes ECG, EDA, EMG, respiration, temperature, and accelerometer data |
| **Hand-Crafted Features** | ~150 interpretable features (statistical, temporal, frequency-domain, EDA-specific) |
| **Robust Validation** | Leave-One-Subject-Out (LOSO) cross-validation for true generalization testing |
| **Multiple Baselines** | Logistic Regression and Random Forest classifiers with balanced class weighting |

---

## Dataset

The project uses the **WESAD dataset** collected by researchers at TU Darmstadt:

- **15 subjects** (S2–S17, excluding S12) wearing a RespiBAN Professional chest device
- **Sampling rate**: 700 Hz across all chest signals
- **Protocol**: Subjects undergo baseline reading (~20 min) and stress induction via the Trier Social Stress Test (~10 min)
- **Binary labels**: Baseline (0) vs. Stress (1)

> **Note**: The raw WESAD data files (~15 GB total) are not included in this repository. Download them separately from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29) and place them in the `WESAD/` directory.

---

## Project Structure

```
WESAD/
├── main.py                  # Pipeline orchestrator
├── requirements.txt         # Python dependencies
├── src/
│   ├── config.py            # Global configuration & constants
│   ├── utils.py             # Shared utilities (seeding, logging)
│   ├── data/                # Data loading from pickle files
│   ├── preprocessing/       # Signal filtering & normalization
│   ├── segmentation/        # Windowing with overlap & purity checks
│   ├── features/            # Feature extraction (~150 features)
│   ├── models/              # Model definitions & training logic
│   └── evaluation/          # LOSO splits, metrics, aggregation
├── docs/
│   ├── project_introduction.md   # ← You are here
│   ├── technical_report.md       # Full methodology & results
│   ├── decision_log.md           # Design rationale for key choices
│   ├── known_unknowns.md         # Open questions & assumptions
│   └── module_responsibility_map.md  # Module-level documentation
├── outputs/                 # Generated results & artifacts
└── WESAD/                   # Raw dataset (not included)
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- WESAD dataset downloaded and placed in the `WESAD/` folder

### Installation

```bash
pip install -r requirements.txt
```

### Running the Pipeline

```bash
python main.py
```

This executes the full pipeline: data loading, signal preprocessing, windowing, feature extraction, baseline evaluation, and LOSO cross-validation for both Logistic Regression and Random Forest models.

---

## Results at a Glance

| Model | Accuracy | Recall | F1-Score | ROC-AUC |
|-------|----------|--------|----------|---------|
| Logistic Regression | 0.72 ± 0.08 | **0.74 ± 0.15** | 0.68 ± 0.10 | 0.78 ± 0.09 |
| Random Forest | 0.75 ± 0.07 | 0.71 ± 0.14 | 0.69 ± 0.09 | 0.80 ± 0.08 |

> Both models exceed the **70% recall target** for stress detection. Logistic Regression achieves higher recall (fewer missed stress events), while Random Forest provides better overall accuracy.

---

## Technical Highlights

- **Signal Conditioning**: Per-modality bandpass filtering (ECG: 0.5–40 Hz, EDA: <1 Hz, EMG: 20–300 Hz, Resp: 0.1–0.5 Hz)
- **Per-Subject Normalization**: Z-score normalization removes inter-subject baseline differences
- **Windowing Strategy**: 60-second windows with 50% overlap, 80% purity threshold, and 5-second transition buffers
- **Class Imbalance**: Handled via `class_weight='balanced'` in all models
- **Reproducibility**: Fixed random seed (42) across all components

---

## Documentation

| Document | Purpose |
|----------|---------|
| [Technical Report](technical_report.md) | Full methodology, results, and discussion |
| [Decision Log](decision_log.md) | Rationale behind key design choices |
| [Known Unknowns](known_unknowns.md) | Open questions and assumptions |
| [Module Responsibility Map](module_responsibility_map.md) | Per-module documentation and interfaces |

---

## Future Directions

- **Wrist signal integration** for consumer wearable compatibility
- **Multi-class classification** (stress, amusement, meditation, baseline)
- **Deep learning approaches** (1D-CNN, LSTM) for automatic feature learning
- **Real-time deployment** on edge devices for live monitoring
- **Personalization** through user-specific model fine-tuning

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Scientific Computing | NumPy, SciPy, Pandas |
| Machine Learning | Scikit-learn |
| Signal Processing | SciPy (Butterworth filters) |
| Visualization | Matplotlib, Seaborn |

---

## References

1. Schmidt, P., Reiss, A., Duerichen, R., Marber, C., & Van Laerhoven, K. (2018). *Introducing WESAD, a multimodal dataset for wearable stress and affect detection.* ICMI 2018.
2. Kirschbaum, C., Pirke, K. M., & Hellhammer, D. H. (1993). *The 'Trier Social Stress Test' – a tool for investigating psychobiological stress responses.* Neuropsychobiology.
