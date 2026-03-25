# WESAD Stress Detection — Project Documentation

**Last updated:** March 2026
**Project type:** Academic / Conference paper
**Status:** Traditional ML complete ✅ | Deep learning (teacher CNN) built ✅ | Knowledge distillation (future work) 🔜

---

## Table of Contents

1. [What This Project Is (Plain English)](#1-what-this-project-is-plain-english)
2. [The Dataset](#2-the-dataset)
3. [The Problem We Are Solving](#3-the-problem-we-are-solving)
4. [The Overall Strategy](#4-the-overall-strategy)
5. [Step-by-Step: How the Pipeline Works](#5-step-by-step-how-the-pipeline-works)
6. [The Machine Learning Models](#6-the-machine-learning-models)
7. [The Deep Learning Models](#7-the-deep-learning-models)
8. [Knowledge Distillation — The Research Contribution](#8-knowledge-distillation--the-research-contribution)
9. [Results So Far](#9-results-so-far)
10. [Technology Stack](#10-technology-stack)
11. [File and Folder Structure](#11-file-and-folder-structure)
12. [How to Run Everything](#12-how-to-run-everything)
13. [Key Design Decisions and Why](#13-key-design-decisions-and-why)
14. [Glossary](#14-glossary)

---

## 1. What This Project Is (Plain English)

This project builds a system that can **automatically detect when a person is stressed**, by analysing physiological signals recorded from sensors worn on their body.

Think of it like this: when you are stressed, your heart beats differently, your skin becomes slightly more conductive (you sweat a little more), your breathing changes, and your muscles tense. These changes are invisible to the eye but are measurable by sensors. This project reads those sensor recordings, learns the patterns that correspond to stress, and builds a classifier — a program that looks at new recordings and says "this person is stressed" or "this person is calm."

The broader goal is not just to build a model that works on a powerful laptop, but to eventually shrink that model down so it can run directly on a small wearable device (like a smartwatch) — in real time, without sending data to a server. That is the research angle for the conference paper.

---

## 2. The Dataset

**Name:** WESAD (Wearable Stress and Affect Detection)
**Published by:** Researchers at the University of Augsburg, Germany (2018)
**Size:** ~15 GB on disk

### What is in the dataset

The dataset contains recordings from **15 volunteers** (labelled S2 through S17; S12 is missing because it was corrupted). Each volunteer wore two sensor devices simultaneously:

- **RespiBAN (chest strap):** Measures ECG, EDA, EMG, Respiration, Temperature, and Accelerometer. Samples 700 times per second.
- **Empatica E4 (wrist band):** Measures EDA, BVP, Temperature, Accelerometer. Samples at lower rates.

Each volunteer went through a **structured experiment** in a lab:
1. Baseline (sitting quietly, reading neutral magazines) — label **1**
2. Stress task (Trier Social Stress Test: public speaking + mental arithmetic under evaluation) — label **2**
3. Amusement (watching funny video clips) — label **3**
4. Meditation — label **4**

### What this project uses

Only the **chest sensor signals** are used, and only the **baseline** and **stress** conditions are kept. The other conditions (amusement, meditation) are discarded. The project therefore becomes a **binary classification** problem: given a sensor recording, is this person in a baseline (calm) state or a stressed state?

**This project does NOT use** the wrist sensor — it is loaded but ignored, because the chest sensor has higher quality signals and a consistent sampling rate.

---

## 3. The Problem We Are Solving

The core task is:

> "Given a 60-second window of physiological sensor data from the chest, predict: is this person stressed (1) or calm (0)?"

Some things that make this problem hard:

- **People are different.** What looks like "stress" in one person's heart rate may look like normal calm for another person. Models need to generalise across people they have never seen before.
- **Class imbalance.** In the dataset, roughly 36% of windows are stress and 64% are baseline. If the model just always predicts "calm", it would be right 64% of the time but completely useless. We therefore care more about **recall** (does the model catch actual stress events?) than raw accuracy.
- **Noisy signals.** Real physiological data contains artefacts, motion noise, and sensor dropouts.

---

## 4. The Overall Strategy

The project is built in **three stages**, each building on the previous:

```
Stage 1 (DONE)     Traditional ML      Hand-crafted features → Logistic Regression / Random Forest
Stage 2 (BUILT)    Deep Learning       Raw signals → 1D Convolutional Neural Network (Teacher)
Stage 3 (FUTURE)   Knowledge Distill.  Teach a tiny model to mimic the big one → runs on a wearable
```

The reason for this progression:
- Stage 1 establishes a strong baseline and proves the problem is solvable.
- Stage 2 builds a more capable model that learns from raw data, not hand-crafted rules.
- Stage 3 is the research novelty: compressing a large model into a small one without much performance loss.

---

## 5. Step-by-Step: How the Pipeline Works

The pipeline has 6 phases. Running `python main.py` executes phases 1–5. Running `python train_teacher.py` re-runs phases 1–3 then executes the deep learning training.

---

### Phase 1 — Setup

Creates output folders (`outputs/models/`, `outputs/reports/`, etc.) and sets a fixed random seed (42) so results are reproducible — the same run will always give the same numbers.

---

### Phase 2 — Data Loading

Each subject's data is stored as a `.pkl` file (a Python-format binary file). The loader reads these files, extracts the chest sensor channels, and **strips out everything except baseline and stress labels**.

After loading, each subject has:
- 6 signal arrays (ECG, EDA, EMG, Respiration, Temperature, Accelerometer), each ~1.5 million samples long
- A label array of the same length, where each sample is tagged 0 (calm) or 1 (stressed)

---

### Phase 3 — Signal Conditioning (Preprocessing)

Raw sensor signals contain noise that has nothing to do with stress — mains electrical interference (50 Hz hum), muscle movement artefacts, baseline drift, etc. This phase cleans each signal using **Butterworth filters** (a standard signal processing technique):

| Signal | What we keep | Why |
|--------|-------------|-----|
| ECG (heart) | 0.5 – 40 Hz | Removes drift and high-frequency muscle noise |
| EDA (skin conductance) | below 1 Hz | EDA changes very slowly; keep only slow changes |
| EMG (muscle) | 20 – 300 Hz | Muscle electrical activity lives in this range |
| Respiration | 0.1 – 0.5 Hz | Normal breath rate is 0.1–0.4 Hz (6–24 breaths/min) |
| Temperature | below 0.1 Hz | Skin temperature changes extremely slowly |

After filtering, each signal is **z-score normalised** — this means we subtract the average and divide by the standard deviation so every signal has a mean of 0 and a standard deviation of 1. This removes the difference between people who just run warmer or have naturally higher skin conductance.

---

### Phase 4 — Windowing

The 1.5 million-sample signal is too long to feed into a model as-is. Instead, it is sliced into **60-second windows** with **50% overlap**:

```
|←—————— 60 sec ——————→|
                |←—————— 60 sec ——————→|
                        |←—————— 60 sec ——————→|
```

At 700 samples per second, each window is **42,000 samples** long.

Each window is then checked:
1. **Purity check:** At least 80% of the samples in the window must be the same label. If a window straddles a stress-to-calm transition, it is discarded.
2. **Transition buffer:** Samples within 5 seconds of any label change are discarded entirely — because those transition moments are ambiguous.

This gives roughly **20–30 usable windows per subject**, for a total of **~400 windows** across all 15 subjects.

---

### Phase 5 — Feature Extraction (Traditional ML path only)

For the traditional ML models, the 42,000 raw samples in each window cannot be fed directly to Logistic Regression or Random Forest — these models need a fixed-size row of numbers. So we compute **~150 summary statistics** per window, per signal channel:

- **Statistical features:** Mean, standard deviation, minimum, maximum, percentiles, skewness (asymmetry), kurtosis (peakedness)
- **Temporal features:** Root mean square, zero-crossing rate (how often the signal crosses zero), signal range
- **Frequency features:** Power in different frequency bands (slow vs fast oscillations), dominant frequency, spectral entropy
- **EDA-specific features:** Number of skin conductance responses (sudden spikes), total rise area, recovery time

Each window therefore becomes a row of ~150 numbers (features), and the entire dataset becomes a matrix of ~400 rows × ~150 columns.

---

### Phase 6 — Model Training and Evaluation

Models are trained and evaluated using **Leave-One-Subject-Out (LOSO) cross-validation**. This is the gold standard for wearable biosignal studies.

**What LOSO means:** We train on 14 subjects and test on the 1 remaining subject. We repeat this 15 times, each time holding out a different subject. Final performance is the average across all 15 test subjects.

**Why LOSO and not a random train/test split?** Because a random split would let data from the same person appear in both training and testing. The model would essentially "memorise" that person and report inflated numbers. LOSO forces the model to generalise to a completely new person it has never seen — which is what actually matters for a real wearable application.

---

## 6. The Machine Learning Models

Three categories of models are built, from dumbest to smartest:

### Baselines (sanity checks)

These are not real models — they are simple rules used to set a floor for performance.

| Baseline | What it does | Why it exists |
|----------|-------------|---------------|
| Random | Randomly predicts stressed/calm with the same probability as the data | Sets the absolute floor |
| Majority | Always predicts "calm" (the bigger class) | A naive model beats this trivially |
| EDA Threshold | If a person's skin conductance is higher than their own average + 1 std dev, predict stress | Simple physiologically-motivated rule |

### Logistic Regression

A linear model that learns a weighted combination of the ~150 features. Think of it as: "if ECG mean is high AND EDA slope is positive AND respiration rate is elevated, lean towards stress." Uses **L2 regularisation** (prevents the model from over-relying on any single feature) and `class_weight='balanced'` (automatically compensates for the 64/36 class imbalance).

### Random Forest

An ensemble of 100 decision trees. Each tree learns a series of yes/no questions about the features ("Is chest_eda_mean > 0.3? → If yes, go left..."). The 100 trees vote, and the majority wins. Random Forests tend to handle noisy features better than logistic regression and automatically capture non-linear relationships.

Both models are trained fresh for each LOSO fold — there is no single "saved" traditional ML model.

---

## 7. The Deep Learning Models

The deep learning path does **not** use hand-crafted features. Instead, raw signals are fed directly into a neural network that learns its own patterns.

### Data Preparation for Deep Learning

The 42,000-sample window (at 700 Hz) is **downsampled to 64 Hz**, giving a 3,840-sample window. This reduces computation without losing the frequency content that matters (EDA changes happen below 1 Hz, respiration below 0.5 Hz — 64 Hz is more than enough). The 6 signal channels are stacked into a single tensor of shape `(6 channels × 3,840 time steps)`.

### Teacher CNN (`src/models/teacher.py`)

The "teacher" is the large, accurate model — the one we train properly. It is a **1D Convolutional Neural Network**:

```
Input: (batch of windows, 6 channels, 3,840 time steps)
    ↓
Block 1: Conv1D(6→32 filters, kernel=7) → BatchNorm → ReLU → MaxPool(÷2)
    ↓  [now 1,920 time steps]
Block 2: Conv1D(32→64 filters, kernel=5) → BatchNorm → ReLU → MaxPool(÷2)
    ↓  [now 960 time steps]
Block 3: Conv1D(64→128 filters, kernel=3) → BatchNorm → ReLU → GlobalAvgPool
    ↓  [now 128 numbers]
Classifier: FC(128→64) → ReLU → Dropout(30%) → FC(64→2)
    ↓
Output: [probability of calm, probability of stress]
```

**What each part does in plain English:**
- **Conv1D layers:** Slide a small detector window across the time series looking for local patterns (a spike, a dip, a slope). The "filters" are learned pattern detectors.
- **BatchNorm:** Keeps numbers in a healthy range during training, making learning faster and more stable.
- **ReLU:** A simple non-linearity (set negatives to zero) that lets the network learn non-linear patterns.
- **MaxPool:** Shrinks the time dimension by 2, forcing the network to focus on the most prominent features at increasing scales.
- **GlobalAvgPool:** Collapses the entire time dimension into a single average number per filter — gives a summary of "how much of this pattern was present across the whole window."
- **Dropout:** Randomly turns off 30% of neurons during training to prevent overfitting (memorising the training data).
- **FC (Fully Connected):** Standard classifier layer — learns which combination of the 128 summary features predicts stress.

**Size:** ~45,000 trainable parameters. Small for a deep learning model, but large relative to what a wearable chip can handle.

### Student Models (`src/models/student.py`) — Stubs for future work

Three miniature versions are defined but not yet trained:

| Model | Parameters | Architecture |
|-------|-----------|--------------|
| TinyCNN | ~4,000 | 2 conv blocks (6→16→32 filters) |
| MicroCNN | ~1,000 | 2 conv blocks (6→8→16 filters) |
| NanoCNN | ~200 | 1 conv block (6→4 filters) |

These are **shells** — the architecture is written, but they contain no learned weights yet. They will be trained via knowledge distillation (Section 8).

---

## 8. Knowledge Distillation — The Research Contribution

This is the novel part of the project and the core argument of the conference paper.

### The problem

A NanoCNN with 200 parameters, if trained normally from scratch, would perform poorly. It is simply too small to learn complex patterns directly from the data. Yet 200 parameters is what a low-power microcontroller on a wearable can handle.

### The solution: knowledge distillation

The idea, originally from Google (Hinton et al., 2015), is:

> **Don't train the small model directly. Instead, train it to mimic the large model.**

The large teacher model does not just output "stressed/calm" — it outputs a **probability distribution** (e.g., "87% stress, 13% calm"). These "soft" probabilities contain more information than a hard label: they tell you how confident the model was, and which stress-like patterns it noticed. The small student model is trained to match these soft outputs, not just the binary label.

Think of it like: instead of learning from the exam answer key (hard labels), the student learns from a tutor who explains their reasoning and level of confidence (soft probabilities).

The loss function used during student training is:

```
Total Loss = α × KL-divergence(student output, teacher soft output)
           + (1 - α) × CrossEntropy(student output, true label)
```

Where α controls how much the student relies on the teacher versus the ground truth.

### Why this matters for the paper

The research question is: **how much can we compress a stress detection model while maintaining clinically useful performance?** Specifically, can a 200-parameter model on a wearable chip achieve comparable recall to a 45,000-parameter model — when the small model learns via distillation instead of from scratch?

This is a practical and publishable contribution because wearable stress monitoring has real applications in mental health, occupational health, and sports science.

---

## 9. Results So Far

### Pipeline run completed — March 24, 2026

| Model | Accuracy | Recall | F1 | ROC-AUC |
|-------|----------|--------|----|---------|
| Random Baseline | ~50% | 0.358 | 0.364 | — |
| Majority Baseline | ~67% | 0.000 | 0.000 | — |
| EDA Threshold | ~60% | **0.866** | 0.792 | — |
| **Logistic Regression (LOSO)** | **96.4% ± 7.2%** | **95.4% ± 13.4%** | **94.7% ± 10.5%** | **0.976** |
| **Random Forest (LOSO)** | **96.6% ± 7.7%** | **96.5% ± 8.1%** | **95.6% ± 8.6%** | **0.996** |
| 1D-CNN Teacher (LOSO) | *pending* | *pending* | *pending* | *pending* |

**Key observations:**
- Both ML models perform very strongly — LOSO recall above 95%.
- The main weak subjects are **S3** and **S9** — both models struggled with them, suggesting high inter-subject variability for these individuals.
- **Random Forest has a notably higher AUC (0.996)**, meaning it ranks stress windows almost perfectly even before choosing a threshold.
- The EDA threshold baseline is surprisingly competitive at 0.866 recall — skin conductance is a strong stress indicator on its own.

### Visualisations generated

Three figures are saved in `outputs/reports/`:
- `fig1_model_comparison.png` — bar chart of all models
- `fig2_loso_per_subject.png` — per-subject breakdown showing S3 and S9 as harder cases
- `fig3_summary_table.png` — printable summary table

---

## 10. Technology Stack

| Tool | What it does in this project |
|------|------------------------------|
| **Python 3.x** | Main programming language |
| **NumPy** | Numerical array operations (the backbone of all data handling) |
| **SciPy** | Signal filtering (Butterworth filters), resampling |
| **scikit-learn** | Logistic Regression, Random Forest, metrics, StandardScaler |
| **PyTorch** | Deep learning framework — defines and trains neural networks |
| **Matplotlib** | Generates the result figures |
| **pickle** | Reads the raw WESAD `.pkl` dataset files |

All dependencies are listed in `requirements.txt`. Install with `pip install -r requirements.txt`.

---

## 11. File and Folder Structure

```
WESAD/
├── WESAD/                          ← Raw dataset (15 GB, not in git)
│   ├── S2/S2.pkl
│   ├── S3/S3.pkl
│   └── ...
│
├── src/                            ← All source code
│   ├── config.py                   ← All parameters in one place (sampling rate, window size, paths)
│   ├── utils.py                    ← Shared helpers (print formatting, seed setting)
│   │
│   ├── data/
│   │   ├── loader.py               ← Reads .pkl files, extracts signals and labels
│   │   ├── subject_data.py         ← Data container for one subject's signals
│   │   └── dl_dataset.py           ← PyTorch Dataset for deep learning training
│   │
│   ├── preprocessing/
│   │   ├── filters.py              ← Butterworth filter implementation
│   │   └── processor.py            ← Applies filters + z-score to each signal
│   │
│   ├── segmentation/
│   │   ├── windowing.py            ← Cuts signals into 60s windows, checks purity
│   │   └── window_data.py          ← Data container for windowed data
│   │
│   ├── features/
│   │   ├── statistical.py          ← Mean, std, percentiles, skewness, kurtosis
│   │   ├── temporal.py             ← RMS, zero-crossing rate, signal range
│   │   ├── frequency.py            ← FFT-based power bands, spectral entropy
│   │   ├── eda.py                  ← EDA-specific features (skin conductance responses)
│   │   └── extractor.py            ← Orchestrates all feature extraction
│   │
│   ├── models/
│   │   ├── classifiers.py          ← Logistic Regression and Random Forest
│   │   ├── baselines.py            ← Random, majority, EDA threshold baselines
│   │   ├── teacher.py              ← 1D-CNN Teacher (~45K parameters)
│   │   └── student.py              ← Tiny student models (stubs, not yet trained)
│   │
│   ├── training/
│   │   └── trainer.py              ← LOSO training loop for Teacher CNN
│   │
│   └── evaluation/
│       ├── splitting.py            ← LOSO split generator
│       ├── metrics.py              ← Accuracy, recall, F1, AUC computation
│       └── reporter.py             ← Generates the result figures (PNG)
│
├── outputs/
│   ├── models/                     ← Saved model checkpoints (.pt files)
│   └── reports/                    ← Generated figures and CSVs
│       ├── fig1_model_comparison.png
│       ├── fig2_loso_per_subject.png
│       ├── fig3_summary_table.png
│       └── model_comparison.csv
│
├── main.py                         ← Runs the full traditional ML pipeline
├── train_teacher.py                ← Trains the 1D-CNN Teacher (30–60 min)
├── generate_report.py              ← Generates result figures from saved results
├── eda_visualization.py            ← Exploratory data analysis plots
└── requirements.txt                ← Python dependencies
```

---

## 12. How to Run Everything

### Install dependencies (first time only)
```bash
pip install -r requirements.txt
```

### Run the traditional ML pipeline (phases 1–5)
```bash
python main.py
```
Takes ~10–15 minutes. Prints results to console.

### Generate result figures (can run immediately — no re-training needed)
```bash
python generate_report.py
```
Takes ~5 seconds. Saves 3 PNG files in `outputs/reports/`.

### Train the Teacher CNN (deep learning)
```bash
python train_teacher.py
```
Takes ~30–60 minutes on CPU. Saves model checkpoints and updates `model_comparison.csv`.

### View exploratory data analysis
```bash
python eda_visualization.py
```
Saves a signal visualisation to `outputs/eda_visualization.png`.

---

## 13. Key Design Decisions and Why

### Why only chest signals, not wrist?
The chest RespiBAN gives all 6 signals at a consistent 700 Hz. The Empatica E4 wrist device records different signals at different rates (e.g., 4 Hz for temperature, 64 Hz for BVP). Combining them would require complex resampling and synchronisation. The chest signals alone are sufficient for strong results, and simplifying the pipeline is sensible for a class project.

### Why 60-second windows?
Short windows (5–10 seconds) do not give enough data to compute meaningful respiration rate or HRV features. Very long windows (120+ seconds) reduce the number of windows per subject drastically and risk spanning multiple conditions. 60 seconds is the standard in the WESAD literature.

### Why 80% purity threshold?
A window that is 60% stress and 40% baseline is genuinely ambiguous — the body is transitioning between states. Keeping only high-purity windows makes the labels reliable. The 5-second transition buffer removes the gradual physiological transition that occurs as stress builds or fades.

### Why LOSO instead of random split?
See Section 5, Phase 6. LOSO is the only evaluation method that tests genuine generalisation to new people — which is the real task.

### Why class_weight='balanced' in the ML models?
There are roughly 1.8× more baseline windows than stress windows. Without compensation, both Logistic Regression and Random Forest would bias toward predicting "calm" because that is the more common class. `class_weight='balanced'` automatically upweights the minority class (stress) during training.

### Why downsample to 64 Hz for deep learning?
The Nyquist theorem says you need at least 2× the highest frequency you care about. The highest-frequency content that matters here is EMG (~300 Hz) — but EMG is already filtered and summarised by the time we use it. For the time-series classification task, 64 Hz captures all relevant physiological dynamics and keeps the sequence length manageable (3,840 vs 42,000 samples).

---

## 14. Glossary

**AUC (Area Under the ROC Curve):** A metric that measures how well a model ranks stressed windows above calm windows. 1.0 is perfect, 0.5 is random. Does not depend on a threshold.

**Butterworth filter:** A type of signal filter that smoothly removes frequencies outside a specified range. "Bandpass" keeps a range; "lowpass" keeps only slow signals.

**Class imbalance:** When one category appears more often than another in the data (here: 64% calm, 36% stress).

**Cross-validation:** A technique for measuring model performance more reliably by testing on multiple different held-out subsets. LOSO is a form of cross-validation.

**Dropout:** A training technique where random neurons are temporarily deactivated to prevent the network from becoming over-reliant on any single feature.

**EDA (Electrodermal Activity):** Skin conductance — a measure of how much the skin conducts electricity, which increases with emotional arousal and stress.

**ECG (Electrocardiogram):** Electrical signal of the heart. Used to compute heart rate and heart rate variability.

**EMG (Electromyogram):** Electrical signal of muscles. Muscle tension increases under stress.

**F1 Score:** The harmonic mean of precision and recall. A balanced metric — useful when false positives and false negatives both matter.

**Feature extraction:** The process of computing summary statistics from raw signals to represent each window as a fixed-length vector of numbers.

**Knowledge distillation:** Training a small model to replicate a large model's soft output probabilities, rather than learning directly from labels.

**LOSO (Leave-One-Subject-Out):** A cross-validation strategy where one person is held out as the test set and the model trains on all others. Repeated for each person.

**Overfitting:** When a model memorises training data instead of learning general patterns. It performs well on training data but poorly on new data.

**Precision:** Of all the windows the model labelled "stressed", what fraction were actually stressed?

**Recall:** Of all the windows that were actually stressed, what fraction did the model catch? This is the most important metric for a stress alarm — we do not want to miss real stress events.

**ReLU (Rectified Linear Unit):** An activation function used in neural networks: output = max(0, input). Introduces non-linearity cheaply.

**ROC curve:** A plot of recall vs false-alarm rate as the decision threshold varies. AUC summarises the whole curve as one number.

**Soft probabilities / soft labels:** The full output distribution from a classifier (e.g., [0.13, 0.87]) rather than just the winning class (stress). Contains richer information for knowledge distillation.

**Z-score normalisation:** Subtract the mean, divide by the standard deviation. Result has mean 0 and std 1. Removes scale differences between people and between signals.
