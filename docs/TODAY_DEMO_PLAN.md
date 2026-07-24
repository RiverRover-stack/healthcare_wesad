# WESAD Demo Plan — Professor Meeting (March 24, 2026)

**Objective**: Show a working end-to-end system + first DL model to demonstrate clear direction toward the knowledge distillation research goal.

---

## What's Already Built & Working

The entire **traditional ML pipeline** is complete, clean, and well-documented:

| Module | Files | Status |
|--------|-------|--------|
| Data Loading | `src/data/loader.py`, `subject_data.py` | ✅ Complete |
| Signal Preprocessing | `src/preprocessing/filters.py`, `processor.py` | ✅ Complete |
| Windowing / Segmentation | `src/segmentation/windowing.py`, `window_data.py` | ✅ Complete |
| Feature Extraction | `src/features/extractor.py`, `statistical.py`, `temporal.py`, `frequency.py`, `eda.py` | ✅ Complete |
| Heuristic Baselines | `src/models/baselines.py` | ✅ Complete |
| ML Classifiers (LOSO) | `src/models/classifiers.py` | ✅ Complete |
| Evaluation | `src/evaluation/metrics.py`, `splitting.py` | ✅ Complete |
| EDA Visualization | `eda_visualization.py` | ✅ Complete |
| Documentation | `docs/technical_report.md`, `decision_log.md`, etc. | ✅ Complete |

**Achieved so far**: ~72–75% accuracy, ~74% stress recall, ROC-AUC ~0.78–0.80 under LOSO cross-validation across 15 subjects.

---

## What's Missing (Gap Analysis)

The full implementation plan targets a **conference-paper pipeline** involving:
- Teacher CNN (multi-scale 1D-CNN)
- 3 student models (MicroCNN, TinyCNN, NanoCNN)
- Knowledge distillation training loop
- Quantization + efficiency benchmarking
- Ablation studies

**None of that exists yet.** `outputs/models/` and `outputs/features/` are both empty — the pipeline may not have been fully run recently to save artifacts.

---

## Today's Demo Plan (Priority Order)

> Estimated total time: **4–5 hours**

---

### Step 1 — Run existing pipeline, capture outputs (~15 min)

Run `python main.py` end-to-end and save results. Also run `eda_visualization.py` to generate the EDA plot.

**Goal**: Have concrete printed numbers and at least one visualization to show — not just code.

```bash
python main.py
python eda_visualization.py
```

Capture the LOSO summary table printed to terminal. Screenshot or save to `outputs/reports/`.

---

### Step 2 — Build a basic 1D-CNN teacher model (~2–3 hours)

This is the **highest-impact thing** to add today. You don't need the full multi-scale architecture — a simple 3-layer 1D-CNN on raw windowed signals is sufficient for a demo.

#### File 1: `src/data/dl_dataset.py` — PyTorch Dataset

- Takes existing `WindowedData` objects (already built by the preprocessing pipeline)
- Stacks the 6 signal channels into a `(C, T)` tensor
- Downsamples from 700 Hz → 64 Hz (reduces sequence length from 42,000 → 3,840)
- Returns `(tensor, label, subject_id)` per window

#### File 2: `src/models/teacher.py` — Simple 1D-CNN

Architecture:
```
Conv1D(6, 32, k=7) → BN → ReLU → MaxPool
Conv1D(32, 64, k=5) → BN → ReLU → MaxPool
Conv1D(64, 128, k=3) → BN → ReLU → AdaptiveAvgPool
FC(128, 64) → ReLU → Dropout(0.3)
FC(64, 2)
```
~100–200K parameters. Do not overcomplicate — default values, no hyperparameter tuning.

#### File 3: `src/training/trainer.py` — Training loop with LOSO

- Reuse existing `loso_split()` logic for subject-level splits
- Adam optimizer, lr=1e-3, 20 epochs per fold
- Save best model per fold based on validation recall
- Record metrics per fold, aggregate at the end

---

### Step 3 — Generate comparison table (~1 hour)

Save a printed + CSV comparison to `outputs/reports/model_comparison.csv`:

| Model | Params | Accuracy | Recall | F1 | ROC-AUC |
|-------|--------|----------|--------|----|---------|
| Random Baseline | — | ~0.50 | ~0.50 | ~0.50 | — |
| Majority Baseline | — | ~0.67 | 0.00 | 0.00 | — |
| EDA Threshold | — | ~0.60 | ~0.55 | ~0.57 | — |
| Logistic Regression (LOSO) | ~150 feat. | 0.72 | 0.74 | 0.68 | 0.78 |
| Random Forest (LOSO) | ~150 feat. | 0.75 | 0.71 | 0.69 | 0.80 |
| **1D-CNN Teacher (LOSO)** | ~150K | **TBD** | **TBD** | **TBD** | **TBD** |

This table **is the story** — it shows the trajectory from baselines → traditional ML → deep learning, and directly motivates knowledge distillation.

---

### Step 4 (If time permits) — Stub student model (~30 min)

Even without training it, having `src/models/student.py` with a defined `MicroCNN` architecture (~5K params) makes the KD direction concrete and shows the professor the next step is already scoped.

```python
# MicroCNN: ~5K params
Conv1D(6, 8, k=7) → BN → ReLU → MaxPool
Conv1D(8, 16, k=5) → BN → ReLU → AdaptiveAvgPool
FC(16, 2)
```

---

## What NOT to Do Today

- ❌ Knowledge distillation training loop
- ❌ Quantization or edge deployment
- ❌ Ablation studies
- ❌ Multi-scale teacher architecture
- ❌ Hyperparameter tuning
- ❌ LaTeX tables or paper figures
- ❌ Wrist signal integration

---

## What to Tell Your Professor

> "The traditional ML pipeline is complete with LOSO cross-validation achieving ~74% stress recall across 15 subjects. I've now started the deep learning phase — a 1D-CNN teacher model is training on raw signals. The next steps are building lightweight student models (MicroCNN, TinyCNN) and implementing the knowledge distillation pipeline, which is the core contribution of this project."

---

## Broader Project Trajectory (Reference)

```
[DONE]         [TODAY]            [NEXT 2 WEEKS]
Traditional ML → 1D-CNN Teacher → KD Pipeline → Student Models → Benchmarking
```

| Phase | Deliverable | Timeline |
|-------|-------------|----------|
| ✅ Traditional ML baseline | LogReg + RF, LOSO eval | Complete |
| 🔧 Teacher CNN | Simple 1D-CNN, LOSO | Today |
| ⬜ Student models | 3 lightweight CNNs | Week 1 |
| ⬜ Knowledge distillation | KD training loop | Week 1–2 |
| ⬜ Efficiency benchmarking | Latency, params, accuracy tradeoff | Week 2 |
| ⬜ Ablation studies | Feature/signal contribution | Week 3 |
| ⬜ Paper writeup | Results, figures, analysis | Week 3–4 |
