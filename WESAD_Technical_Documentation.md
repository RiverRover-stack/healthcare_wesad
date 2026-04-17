# WESAD Stress Detection — Complete Technical Documentation

**Author**: Kaustubh Agrawal  
**Project**: Binary Stress Detection Using Wearable Physiological Signals  
**Dataset**: WESAD (Wearable Stress and Affect Detection)  
**Last Updated**: April 8, 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Dataset Description](#3-dataset-description)
4. [System Architecture](#4-system-architecture)
5. [Phase 1: Traditional ML Pipeline](#5-phase-1-traditional-ml-pipeline)
6. [Phase 2: Deep Learning & Knowledge Distillation](#6-phase-2-deep-learning--knowledge-distillation)
7. [Explainability & Interpretability](#7-explainability--interpretability)
8. [Results & Analysis](#8-results--analysis)
9. [Design Decisions & Rationale](#9-design-decisions--rationale)
10. [Known Limitations & Open Questions](#10-known-limitations--open-questions)
11. [Deployment Considerations](#11-deployment-considerations)
12. [How to Reproduce](#12-how-to-reproduce)
13. [References](#13-references)

---

## 1. Executive Summary

This project builds a complete stress detection system — from raw physiological signals to edge-deployable models — using the WESAD dataset (15 subjects, chest-worn RespiBAN sensors at 700 Hz).

The pipeline progresses through two phases:

**Phase 1 (Traditional ML):** Load raw pickles → Butterworth filtering → z-score normalization → 60-second sliding windows → ~150 hand-crafted features → Logistic Regression and Random Forest classifiers → LOSO cross-validation. Result: **0.96+ accuracy, 0.95+ recall** for both classifiers.

**Phase 2 (Deep Learning + Knowledge Distillation):** Train a multi-scale 1D-CNN teacher (266K params, 3 parallel branches at different temporal scales) → distill into three student architectures optimized for edge deployment → LOSO validation. Result: **MicroCNN (distilled, ~5K params) achieves 0.998 accuracy and 0.996 F1** — matching the teacher at 50x compression. This is the headline result.

The project also includes SHAP-based channel importance analysis, Grad-CAM temporal activation maps, KD hyperparameter ablation studies, and INT8 quantization benchmarks.

---

## 2. Problem Statement & Motivation

### The Problem

Chronic stress affects ~77% of adults (APA) and contributes to cardiovascular disease, mental health disorders, and reduced quality of life. Most individuals lack objective, continuous tools to monitor stress. Commercial wearables provide basic metrics (heart rate, steps) but lack sophisticated, validated stress classification.

### The Core Question

> Can we build a model that reliably detects stress from wearable physiological signals, generalizes to unseen users, and runs on resource-constrained edge devices?

### Why This Matters

The wearable health technology market is projected to exceed $100 billion by 2030. Stress monitoring is a high-demand feature, but the gap between research prototypes and deployable systems remains wide. This project bridges that gap by going from raw signals to quantized edge models in a single, reproducible pipeline.

### Hypotheses

- **H1**: Physiological signals (ECG, EDA, respiration, etc.) contain measurable patterns that reliably distinguish stress from baseline states.
- **H2**: Traditional ML with hand-crafted features can achieve >70% recall for stress detection under LOSO validation.
- **H3**: Knowledge distillation can compress a large teacher CNN into models small enough for microcontrollers without significant performance loss.

All three hypotheses were validated. H2 was exceeded by a wide margin (recall >95%).

---

## 3. Dataset Description

### WESAD Overview

| Attribute | Details |
|-----------|---------|
| Source | TU Darmstadt (Schmidt et al., 2018) |
| Subjects | 15 participants (S2–S17, S12 excluded — corrupted) |
| Duration | ~2 hours per subject |
| Device | RespiBAN Professional (chest-worn) |
| Sampling Rate | 700 Hz (chest) |
| File Format | Python pickle (.pkl), ~975 MB each |
| Total Size | ~15 GB |

### Signals Used

| Signal | Description | Stress Relevance |
|--------|-------------|-----------------|
| ECG | Electrocardiogram | Heart rate variability, cardiac stress response |
| EDA | Electrodermal Activity | Skin conductance, sympathetic arousal (primary stress marker) |
| EMG | Electromyography | Muscle tension under stress |
| Resp | Respiration | Breathing rate and depth changes |
| Temp | Body Temperature | Thermoregulation under stress |
| ACC | 3-axis Accelerometer | Motion artifact detection |

Only chest signals are used in the current implementation. Wrist signals (Empatica E4, 4–64 Hz) are loaded but reserved for future work.

### Data Collection Protocol

Subjects underwent a structured protocol:
- **Baseline** (Label 1): Neutral reading task (~20 minutes)
- **Stress** (Label 2): Trier Social Stress Test (TSST) — public speaking + mental arithmetic (~10 minutes)
- Other states (Label 3: amusement, Label 4: meditation, Labels 5–7: other) are excluded.
- Label 0 (unlabeled/transition periods) is discarded.

### Binary Classification Task

Raw labels are remapped: baseline (1) → 0, stress (2) → 1. This creates a ~64/36 class split (baseline-heavy), which is handled via balanced class weights and weighted sampling.

---

## 4. System Architecture

### Data Flow

```
WESAD/{S2..S17}/S{id}.pkl
    → src/data/loader.py              # Load pickles, curate binary labels
    → src/preprocessing/processor.py  # Per-signal Butterworth filtering + z-score normalization
    → src/segmentation/windowing.py   # 60s sliding windows, 50% overlap, 80% purity threshold
    → [Phase 1] src/features/         # ~150 hand-crafted features per window
        → src/models/classifiers.py   # LogReg & RandomForest
    → [Phase 2] src/data/dl_dataset.py   # 700→64 Hz downsampling, PyTorch Dataset
        → src/models/teacher.py           # Multi-Scale Teacher CNN (266K params)
        → src/models/student.py           # MicroCNN / TinyCNN / MiniCNN-LSTM
        → src/models/distillation.py      # Knowledge distillation training
    → src/evaluation/                 # LOSO cross-validation, metrics aggregation
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `src/data/loader.py` | Load `.pkl` files with `encoding='latin1'`; returns `SubjectData` containers |
| `src/data/subject_data.py` | `SubjectData` dataclass: chest signals, labels, subject_id |
| `src/data/dl_dataset.py` | PyTorch Dataset: 700→64 Hz FFT-based resampling, NaN sanitization |
| `src/data/dl_dataloader.py` | LOSO DataLoader factory with WeightedRandomSampler |
| `src/preprocessing/filters.py` | `scipy.signal.butter` + `filtfilt` Butterworth filters |
| `src/preprocessing/processor.py` | Per-modality filtering → z-score normalization per subject |
| `src/segmentation/windowing.py` | Sliding windows; rejects windows with <80% label purity |
| `src/features/` | `statistical.py`, `temporal.py`, `frequency.py`, `eda.py` |
| `src/features/extractor.py` | Orchestrates feature extraction across all signal modalities |
| `src/models/classifiers.py` | LogisticRegression (L2, C=1.0), RandomForest (100 trees, depth=10) |
| `src/models/baselines.py` | Random, majority, EDA threshold baselines |
| `src/models/teacher.py` | MultiScaleTeacherCNN — 3 parallel conv branches |
| `src/models/student.py` | MicroCNN, TinyCNN, MiniCNN-LSTM |
| `src/models/distillation.py` | KDLoss, student training (standalone + distilled modes) |
| `src/models/quantization.py` | INT8 dynamic quantization |
| `src/evaluation/splitting.py` | LOSO split generator |
| `src/evaluation/metrics.py` | Per-fold metrics + `aggregate_fold_metrics()` for mean±std |
| `src/evaluation/efficiency.py` | Params, FP32/INT8 size, CPU latency, FLOPs |
| `src/evaluation/reporter.py` | Publication-quality figure generation (fig1–fig7) |
| `shap_analysis.py` | SHAP channel importance, Grad-CAM, per-class analysis |

### Key Configuration (`src/config.py`)

All pipeline parameters are centralized:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `CHEST_SAMPLING_RATE` | 700 Hz | Raw data sampling rate |
| `WINDOW_LENGTH_SEC` | 60 s | Window duration |
| `WINDOW_OVERLAP` | 0.5 | 50% overlap between windows |
| `PURITY_THRESHOLD` | 0.8 | Minimum label purity per window |
| `TRANSITION_BUFFER_SEC` | 5 s | Discard zone around state transitions |
| `RANDOM_SEED` | 42 | Reproducibility |
| `DL_CONFIG.downsample_target_hz` | 64 Hz | Target sampling rate for DL models |
| `DL_CONFIG.teacher_epochs` | 40 | Teacher training epochs |
| `DL_CONFIG.student_epochs` | 30 | Student training epochs |
| `DL_CONFIG.lr` | 3e-4 | AdamW learning rate |
| `DL_CONFIG.kd_temperature` | 4.0 | KD softmax temperature |
| `DL_CONFIG.kd_alpha` | 0.7 | Teacher weight in KD loss |

---

## 5. Phase 1: Traditional ML Pipeline

### 5.1 Signal Preprocessing

**Step 1 — Filtering:** Each signal gets a modality-specific Butterworth filter (order 4, `filtfilt` for zero-phase distortion):

| Signal | Filter Type | Cutoff | Rationale |
|--------|------------|--------|-----------|
| ECG | Bandpass | 0.5–40 Hz | Remove baseline wander + HF noise |
| EDA | Lowpass | <1 Hz | EDA is a slow-varying signal |
| EMG | Bandpass | 20–300 Hz | Muscle activity frequency range |
| Resp | Bandpass | 0.1–0.5 Hz | Normal breathing frequency range |
| Temp | Lowpass | <0.1 Hz | Very slow thermal changes |
| ACC | None (magnitude) | — | 3-axis → magnitude, then z-score |

**Step 2 — Normalization:** Z-score per subject (`z = (x - μ_subject) / σ_subject`). This removes inter-subject baseline differences while preserving within-subject stress response patterns. Robust z-score (using median/IQR) is used for EDA and Temp due to outlier sensitivity.

**Step 3 — NaN Handling:** NaN values (primarily in respiration for some subjects) are replaced with per-signal median values.

### 5.2 Windowing & Segmentation

Signals are segmented into 60-second windows (42,000 samples at 700 Hz) with 50% overlap (21,000-sample step). Windows are filtered by two quality criteria:

1. **Purity Threshold (80%):** At least 80% of samples in a window must share the same label. Mixed-state windows are discarded.
2. **Transition Buffer (5s):** Samples within ±5 seconds of a label change are excluded before windowing.

This produces clean, unambiguous windows for training and evaluation.

### 5.3 Feature Engineering

~150 hand-crafted features are extracted per window, organized into four categories:

**Statistical Features (~10 per signal):** mean, std, variance, min, max, range, percentiles (10th, 25th, 50th, 75th, 90th), IQR, skewness, kurtosis, RMS, energy.

**Temporal Features (~5 per signal):** zero-crossing rate, peak count, slope features (mean/std of first derivative).

**Frequency Features (~5 per signal):** Welch's PSD estimate, power in bands (VLF, LF, HF, VHF), peak frequency, spectral centroid, spectral entropy.

**EDA-Specific Features (~5):** tonic/phasic decomposition (moving average method), SCR count, SCR amplitude statistics.

Features are extracted per signal and concatenated into a single feature vector per window. NaN values in features are replaced with 0.

### 5.4 Models

**Logistic Regression:** L2 penalty, C=1.0, balanced class weights, lbfgs solver, max_iter=1000. Provides interpretable coefficients and a strong linear baseline.

**Random Forest:** 100 trees, max_depth=10, min_samples_split=5, balanced class weights. Captures non-linear patterns and is robust to noise.

Both models use StandardScaler on the feature matrix (fit on training data only).

### 5.5 Baselines

Three baselines establish the floor:
- **Random Baseline:** Samples predictions from class prior distribution.
- **Majority Baseline:** Always predicts the majority class (baseline → 0).
- **EDA Threshold:** Per-subject threshold on mean EDA (mean + 1×std of baseline EDA values). A simple physiological heuristic.

---

## 6. Phase 2: Deep Learning & Knowledge Distillation

### 6.1 Motivation

While Phase 1 achieved strong results, the hand-crafted features require domain expertise and may miss complex temporal patterns. Phase 2 explores whether 1D-CNNs can learn features directly from raw signals, and whether knowledge distillation can compress a large teacher model into edge-deployable students.

### 6.2 Data Preparation for Deep Learning

Raw 700 Hz signals are downsampled to 64 Hz using FFT-based resampling (`scipy.signal.resample`), producing 3,840-sample windows (6 channels × 60 seconds × 64 Hz). Input tensor shape: `(batch, 6, 3840)`.

NaN sanitization is critical: `nan_to_num(nan→0, posinf→5, neginf→-5)` is applied before resampling (FFT spreads NaN), after resampling (Gibbs ringing), and with ±5 clipping.

### 6.3 Teacher Architecture: MultiScaleTeacherCNN

The teacher uses **3 parallel convolutional branches** at different kernel sizes to capture physiological phenomena at different timescales:

| Branch | Kernel Size | Captures | Physiological Target |
|--------|-------------|----------|---------------------|
| Small | k=8 | Fast patterns | ECG heartbeats, motion artifacts |
| Medium | k=32 | Mid-range patterns | Respiration cycles |
| Large | k=64 | Slow trends | EDA rise, thermal drift |

Each branch: Conv1d → BatchNorm → ReLU → Conv1d → BatchNorm → ReLU → GlobalAvgPool1d.

The three 64-dim branch outputs are concatenated (192-dim) and passed through: FC(192→128) → ReLU → Dropout(0.3) → FC(128→64) → ReLU → FC(64→2).

Total parameters: **~266K**. The model also exposes `forward_with_features()` which returns both logits and the 192-dim embedding for feature-based distillation.

### 6.4 Student Architectures

Three student models target different deployment scenarios:

#### MicroCNN (~5.3K params) — Microcontroller Target

Uses **depthwise-separable convolutions** (MobileNet-style), achieving ~8x parameter reduction over standard Conv1d.

Architecture: DS-Conv(6→16, k=7) → MaxPool → DS-Conv(16→32, k=5) → MaxPool → DS-Conv(32→64, k=3) → GlobalAvgPool → FC(64→32) → ReLU → FC(32→2).

Target: ARM Cortex-M4 (<32KB flash).

#### TinyCNN (~15.2K params) — Smartphone/BLE SoC Target

Standard 3-layer Conv1d, straightforward and quantization-friendly.

Architecture: Conv1d(6→24, k=7) → BN → ReLU → MaxPool → Conv1d(24→48, k=5) → BN → ReLU → MaxPool → Conv1d(48→56, k=3) → BN → ReLU → GlobalAvgPool → FC(56→2).

Target: nRF52840 (<256KB flash).

#### MiniCNN-LSTM (~28.8K params) — ESP32 Target

Hybrid architecture: 2 Conv blocks compress the time dimension (3840→240), then an LSTM(40) processes the temporal sequence. This captures long-range dependencies like slow EDA trends.

Target: ESP32-S3 (<512KB, has PSRAM).

### 6.5 Knowledge Distillation

The KD loss (Hinton et al., 2015) combines soft targets from the teacher with hard labels:

```
L_KD = α × T² × KL(softmax(student_logits / T) || softmax(teacher_logits / T))
     + (1 - α) × CrossEntropy(student_logits, y_true)
```

- **T = 4.0** (temperature): Softens the teacher's output distribution, exposing inter-class similarities ("dark knowledge").
- **α = 0.7**: 70% of the loss comes from the teacher's soft targets, 30% from ground truth hard labels.
- **T² scaling**: Restores gradient magnitude that would otherwise be suppressed by the softmax temperature.

Training uses:
- **AdamW optimizer** (lr=3e-4, weight_decay=1e-3)
- **ReduceLROnPlateau** scheduler (monitors F1, patience=5)
- **Gradient clipping** (max_norm=1.0)
- **WeightedRandomSampler** — forces ~50/50 class balance in every batch
- **Class-weighted CrossEntropy** — secondary imbalance defense
- Best checkpoint saved per fold (by F1 score)

### 6.6 LOSO Evaluation

All models (Phase 1 and Phase 2) are evaluated using Leave-One-Subject-Out cross-validation:

```
For each subject S_i (i = 1 to 15):
    Train on all subjects except S_i
    Test on S_i
    Record: accuracy, precision, recall, F1, ROC-AUC
Aggregate: mean ± std across 15 folds
```

This is the gold standard for physiological signal classification — it simulates real-world deployment where the model encounters unseen users.

---

## 7. Explainability & Interpretability

### 7.1 SHAP Channel Importance

SHAP (SHapley Additive exPlanations) analysis reveals which physiological signal channels contribute most to the teacher CNN's stress predictions.

**Key Finding (Figure: SHAP Channel Importance):** ECG dominates with mean |SHAP| = 0.00093, followed by Temp (0.00029), EDA (0.00025), ACC (0.00023), and EMG (0.00016). Respiration contributes near-zero (0.00000), suggesting the CNN finds minimal predictive value in the filtered respiration signal.

### 7.2 Per-Class SHAP Analysis

**Key Finding (Figure: Per-Class SHAP):** ECG importance is asymmetric — it matters ~1.75x more for classifying stress windows (0.00128) than baseline windows (0.00073). EDA shows 2.4x higher importance for stress (0.00039 vs 0.00016). Interestingly, Temp and ACC are more important for classifying baseline (0.00041 and 0.00026) than stress (0.00010 and 0.00019), suggesting the model uses temperature/motion stability as a baseline indicator.

### 7.3 Grad-CAM Temporal Activation

**Key Finding (Figure: Grad-CAM Temporal):** The teacher's fast branch (k=8) shows consistently higher activation for stress windows (~0.33) versus baseline windows (~0.17) across the entire 60-second window. The activation is relatively uniform temporally — the model doesn't focus on a specific time segment but uses the entire window.

### 7.4 Knowledge Distillation — Preserved Attention

**Key Finding (Figure: Teacher vs Student SHAP):** After 50x compression (266K → 5.3K params), the MicroCNN student preserves the teacher's general attention pattern but redistributes it. The student relies less on ECG (0.00012 vs 0.00093) and more on Temp (0.00047 vs 0.00028) and EDA (0.00035 vs 0.00024). This suggests the student learns a complementary feature representation that is more efficient for its smaller capacity.

---

## 8. Results & Analysis

### 8.1 Complete Results Table

| Model | Params | Accuracy | Recall | F1 | ROC-AUC |
|-------|--------|----------|--------|----|---------|
| Random Baseline | -- | ~0.50 | 0.358 | 0.364 | -- |
| Majority Baseline | -- | ~0.67 | 0.000 | 0.000 | -- |
| EDA Threshold | -- | ~0.60 | 0.866 | 0.792 | -- |
| **Logistic Regression** | ~150 feat. | 0.964 ± 0.072 | 0.954 ± 0.134 | 0.947 ± 0.105 | 0.976 ± 0.062 |
| **Random Forest** | ~150 feat. | 0.966 ± 0.077 | 0.965 ± 0.081 | 0.956 ± 0.086 | 0.996 ± 0.011 |
| **1D-CNN Teacher (Multi-Scale)** | ~266K | 0.987 ± 0.024 | 0.986 ± 0.041 | 0.981 ± 0.035 | 0.995 ± 0.014 |
| MicroCNN (standalone) | ~5K | 0.968 ± 0.065 | 0.975 ± 0.092 | 0.957 ± 0.084 | 0.985 ± 0.034 |
| **MicroCNN (distilled)** | ~5K | **0.998 ± 0.006** | **0.996 ± 0.013** | **0.996 ± 0.009** | **1.000 ± 0.002** |
| TinyCNN (standalone) | ~15K | 0.988 ± 0.024 | 0.986 ± 0.041 | 0.983 ± 0.035 | 0.996 ± 0.012 |
| TinyCNN (distilled) | ~15K | 0.986 ± 0.026 | 0.979 ± 0.046 | 0.979 ± 0.037 | 0.995 ± 0.012 |
| MiniCNN-LSTM (standalone) | ~28K | 0.986 ± 0.026 | 0.982 ± 0.066 | 0.979 ± 0.042 | 0.995 ± 0.016 |
| **MiniCNN-LSTM (distilled)** | ~28K | 0.995 ± 0.012 | 0.993 ± 0.026 | 0.993 ± 0.018 | 0.997 ± 0.011 |

*All results are LOSO cross-validation (15 folds). Values are mean ± std.*

### 8.2 Key Findings

**Finding 1 — Traditional ML is surprisingly strong.** Logistic Regression and Random Forest achieve 0.95+ recall with ~150 hand-crafted features. This far exceeds the initial 70% recall target and shows that domain-informed feature engineering remains competitive.

**Finding 2 — The multi-scale teacher improves over traditional ML.** The teacher CNN (0.987 accuracy, 0.981 F1) improves on Random Forest (0.966 accuracy, 0.956 F1), with notably lower variance across folds — meaning more consistent performance across subjects.

**Finding 3 — Knowledge distillation produces the best model overall.** MicroCNN (distilled) achieves 0.998 accuracy and 0.996 F1 with only ~5K parameters — *surpassing its own teacher*. This is the best result in the pipeline at 50x compression.

**Finding 4 — Distillation helps small models most.** The KD improvement chart (Figure: KD Improvement) shows MicroCNN gains +0.039 F1 from distillation, MiniCNN-LSTM gains +0.014, while TinyCNN shows -0.004 (negligible). The smallest model benefits most from the teacher's soft targets.

**Finding 5 — KD hyperparameters are insensitive.** The ablation study (Figure: KD Ablation) shows F1 converges to 0.9965 across all tested temperatures (1, 2, 4, 8) and alphas (0.3, 0.5, 0.7, 0.9). This is a strong signal — it means the task has high signal-to-noise ratio and the student model has sufficient capacity.

### 8.3 Per-Subject Analysis

**Problematic Subjects (Figure: Per-Subject LOSO, LOSO Heatmap):**

- **S2**: LogReg F1 = 0.74, RF F1 = 0.97. LogReg struggles here while RF handles it well. The near-zero recall for LogReg on S2 suggests this subject's stress pattern is non-linear.
- **S3**: LogReg F1 = 0.64, RF F1 = 0.86. Both models underperform relative to other subjects.
- **S8**: RF F1 = 0.69, the worst RF fold. This subject may have atypical physiological responses.
- **S9**: RF F1 = 0.85. Slightly below average.
- **S4–S7, S10–S17**: Both models achieve F1 ≥ 0.97, often 1.00.

The deep learning models largely eliminate these per-subject weaknesses — the distilled MicroCNN achieves near-perfect performance across all subjects.

### 8.4 Pareto Front — Accuracy vs. Model Size

**Key Finding (Figure: Pareto Front):** The Pareto frontier shows a clear efficiency curve. Traditional ML models (LogReg, RF) cluster at low model size but slightly lower F1 (~0.95). Student models (distilled) dominate the upper-left corner — high F1 (>0.99) at minimal size (<60KB FP32). The teacher sits at ~1MB, achieving 0.98 F1 — efficient for a research model but overkill for deployment when distilled students are available.

### 8.5 Hypothesis Validation

| Hypothesis | Verdict | Evidence |
|------------|---------|---------|
| **H1**: Physio signals distinguish stress | Supported | ROC-AUC 0.976–1.000 across models |
| **H2**: ML achieves >70% recall (LOSO) | Strongly Supported | All trained models exceed 0.95 recall |
| **H3**: KD compresses without significant loss | Supported | MicroCNN (5K) surpasses teacher (266K) |

---

## 9. Design Decisions & Rationale

### 9.1 Why 60-Second Windows?

EDA has a latency of 1–3 seconds and a full response of 10–15 seconds. A 60-second window captures the complete stress response cycle including onset, peak, and partial recovery. 30-second windows risk truncating slower physiological responses; 120-second windows reduce sample count without proportional benefit.

### 9.2 Why 50% Overlap?

Balances data augmentation (more training samples) against window independence. 0% overlap produces too few samples for 15 subjects; 75% overlap creates highly correlated adjacent windows that inflate evaluation metrics.

### 9.3 Why 80% Purity Threshold?

Ensures each window has an unambiguous physiological state. 100% is too strict (loses many windows near transitions); 60% allows mixed-state windows that introduce label noise.

### 9.4 Why Per-Subject Z-Score?

Physiological baselines vary dramatically between individuals (resting heart rate alone can vary 40–100 BPM). Per-subject normalization removes these inter-individual differences while preserving within-subject stress response magnitude. Global normalization would conflate inter-subject variation with stress response.

### 9.5 Why Multi-Scale CNN (Not Sequential)?

Different physiological phenomena operate at different timescales. A single kernel size forces a compromise. Parallel branches with k=8 (captures ECG R-peaks at ~1 Hz), k=32 (captures respiration at ~0.2 Hz), and k=64 (captures EDA trends at ~0.05 Hz) allow each branch to specialize.

### 9.6 Why Depthwise-Separable Convolutions for MicroCNN?

Standard Conv1d with C_in=6 and C_out=16 at k=7 requires 6×16×7 = 672 parameters per layer. Depthwise-separable splits this into depthwise (6×1×7 = 42) + pointwise (6×16×1 = 96) = 138 parameters — a ~5x reduction. At the full model scale, MicroCNN achieves ~8x overall parameter reduction versus a standard equivalent, making it viable for MCUs with <32KB flash.

### 9.7 Why Response-Based KD (Not Feature-Based)?

Response-based KD (soft target matching) is simpler, architecture-agnostic (teacher and student don't need matching intermediate dimensions), and empirically sufficient for this binary task. Feature-based KD adds complexity without clear benefit when the task has high signal-to-noise ratio.

### 9.8 Why WeightedRandomSampler + Balanced Class Weights?

Double defense against the 64/36 class imbalance. WeightedRandomSampler ensures each batch is approximately 50/50 (prevents the model from ignoring the minority class during training). Balanced class weights in the loss function provide a secondary correction. Without both, early experiments showed recall collapsing to 0 (the model learned to predict only baseline).

---

## 10. Known Limitations & Open Questions

### Limitations

1. **Controlled Lab Setting:** TSST-induced stress may not generalize to everyday stressors (work, social, financial).
2. **Small Sample Size:** 15 subjects cannot capture full population diversity (age, ethnicity, fitness level, medical conditions).
3. **Chest-Only Signals:** Chest-worn sensors are impractical for daily use. Wrist-based models are needed for consumer adoption.
4. **Binary Classification:** Only baseline vs. stress. Amusement, meditation, and other affective states are discarded.
5. **Offline Processing:** No real-time streaming implementation — current pipeline processes complete sessions.
6. **Respiration Signal Quality:** Some subjects show NaN values after filtering (root cause unclear — sensor or filter instability).

### Open Questions

| ID | Question | Priority | Status |
|----|----------|----------|--------|
| 1 | Are literature-based filter cutoffs optimal for stress detection? | Medium | Open |
| 2 | Root cause of respiration NaN (sensor vs. filter instability)? | High | Open |
| 3 | EMG highcut at 450 Hz vs. Nyquist at 350 Hz — impact of capping? | Low | Documented |
| 4 | Simple moving average vs. neurokit2 for EDA decomposition accuracy? | Medium | Open |
| 5 | How many of the ~150 features are redundant (correlation analysis)? | Medium | Open |
| 6 | Why do S2, S3, S8 consistently underperform? Physiological outliers? | High | Open |
| 7 | Would cross-dataset validation (SWELL, AMIGOS) hold up? | High | Future work |

---

## 11. Deployment Considerations

### Target Hardware

| Model | Params | FP32 Size | Target Device | Flash Budget |
|-------|--------|-----------|---------------|-------------|
| MicroCNN (distilled) | ~5.3K | ~21 KB | ARM Cortex-M4 | <32 KB |
| TinyCNN (distilled) | ~15.2K | ~60 KB | nRF52840 BLE SoC | <256 KB |
| MiniCNN-LSTM (distilled) | ~28.8K | ~113 KB | ESP32-S3 | <512 KB |

### INT8 Quantization

Dynamic INT8 quantization (PyTorch `torch.quantization.quantize_dynamic`) achieves 3–4x compression over FP32 with negligible accuracy loss. This brings MicroCNN to ~6 KB — well within MCU flash budgets.

### Inference Requirements

Input: 60 seconds of 6-channel data at 64 Hz = 3,840 × 6 = 23,040 float values (~92 KB at FP32, ~23 KB at INT8). This fits comfortably in RAM for all target devices.

---

## 12. How to Reproduce

### Prerequisites

```bash
pip install -r requirements.txt
```

Dependencies: Python 3.x, NumPy, SciPy, scikit-learn, PyTorch, matplotlib, seaborn.

### Run Phase 1 (Traditional ML)

```bash
python main.py
```

This runs the full pipeline: load → preprocess → window → extract features → train LogReg/RF → LOSO evaluation → print results.

### Run Phase 2 (Deep Learning)

```bash
# Train the multi-scale teacher CNN
python train_teacher.py

# Train all student models (standalone + distilled)
python train_students.py

# Run KD hyperparameter ablation
python run_ablation.py
```

### Generate Reports & Figures

```bash
# Generate all publication figures (fig1-fig7)
python generate_report.py

# Run SHAP/Grad-CAM explainability analysis
python shap_analysis.py

# EDA visualization
python eda_visualization.py
```

### Output Files

Results: `outputs/reports/model_comparison.csv`, `outputs/reports/ablation_results.csv`

Figures: `outputs/reports/fig1_model_comparison.png` through `fig7_ablation.png`, plus `shap_*.png`.

### Dataset

The WESAD dataset is not redistributable. Request access from the original authors (Schmidt et al., 2018, TU Darmstadt). Place extracted data in `WESAD/` directory.

---

## 13. References

1. Schmidt, P., Reiss, A., Duerichen, R., Marber, C., & Van Laerhoven, K. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. *ICMI 2018*.
2. Kirschbaum, C., Pirke, K. M., & Hellhammer, D. H. (1993). The 'Trier Social Stress Test' — a tool for investigating psychobiological stress responses. *Neuropsychobiology*.
3. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*.
4. Howard, A. G., et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint arXiv:1704.04861*.
5. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.

---

## Appendix A: Figure Reference

| Figure | File | Section |
|--------|------|---------|
| Model Comparison (bar chart) | `fig1_model_comparison.png` | 8.1 |
| LOSO Per-Subject (LogReg vs RF) | `fig2_loso_per_subject.png` | 8.3 |
| Results Summary Table | `fig3_summary_table.png` | 8.1 |
| Pareto Front (F1 vs Size) | `fig4_pareto_front.png` | 8.4 |
| KD Improvement | `fig5_kd_improvement.png` | 8.2 |
| LOSO Heatmap (All Models) | `fig6_loso_heatmap.png` | 8.3 |
| KD Ablation (T and α) | `fig7_ablation.png` | 8.5 |
| SHAP Channel Importance | `shap_channel_importance.png` | 7.1 |
| Grad-CAM Temporal | `shap_gradcam_temporal.png` | 7.3 |
| Teacher vs Student SHAP | `shap_teacher_vs_student.png` | 7.4 |
| Per-Class SHAP | `shap_per_class.png` | 7.2 |

---

*Generated: April 8, 2026*
