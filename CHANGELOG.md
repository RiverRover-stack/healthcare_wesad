# CHANGELOG

All notable changes to the WESAD stress detection project.

---

## [Phase 2] Deep Learning + Knowledge Distillation — 2026-03-24

### Added
- **Multi-scale Teacher CNN** (`src/models/teacher.py`)
  - Replaced sequential 3-block CNN (45K params) with 3 parallel branches (k=8/32/64)
  - 266K parameters; adds `forward_with_features()` for feature distillation
  - `freeze()` method for use as frozen teacher during KD

- **Knowledge Distillation pipeline** (`src/models/distillation.py`)
  - `KDLoss`: response-based KD (Hinton 2015) with T² scaling and class-weight support
  - `train_student_kd_loso()`: full LOSO training in `standalone` or `distilled` mode
  - Appends per-student results to `model_comparison.csv`

- **Student models** (`src/models/student.py`) — 3 architectures
  - `MicroCNN` (~5.3K): depthwise-separable Conv1D (MobileNet-style), edge MCU target
  - `TinyCNN` (~15.2K): 3-layer standard Conv1D, smartphone/BLE target
  - `MiniCNN-LSTM` (~28.8K): Conv1D + LSTM hybrid, captures long-range patterns
  - `STUDENT_REGISTRY` dict for programmatic access

- **LOSO DataLoader factory** (`src/data/dl_dataloader.py`)
  - `build_subject_index()` + `get_loso_dataloaders()` — encapsulates split logic

- **Efficiency metrics** (`src/evaluation/efficiency.py`)
  - `get_efficiency_report()`: params, FP32 size (KB), CPU latency (ms), FLOPs (optional via thop)
  - `run_all_efficiency_benchmarks()` for batch measurement
  - `print_efficiency_table()` for console display

- **INT8 quantization** (`src/models/quantization.py`)
  - `quantize_model()`: PyTorch dynamic INT8 quantization
  - `compare_fp32_vs_int8()`: size and latency comparison report

- **Results aggregation** (`src/evaluation/results.py`)
  - `load_results()`: parse `model_comparison.csv` into structured dict
  - `accuracy_latex()`: generate main results LaTeX table
  - `efficiency_latex()`: generate efficiency LaTeX table
  - `save_latex_tables()`: write `.tex` files to `outputs/reports/`

- **New report figures** (`src/evaluation/reporter.py`)
  - `plot_pareto_front()` → `fig4_pareto_front.png`
  - `plot_kd_improvement()` → `fig5_kd_improvement.png`
  - `plot_loso_heatmap()` → `fig6_loso_heatmap.png`
  - `plot_ablation()` → `fig7_ablation.png`
  - `generate_advanced_figures()` convenience wrapper

- **`DL_CONFIG`** in `src/config.py` — single source of truth for all DL hyperparameters
- **`train_students.py`** — entry point for all student training (standalone + distilled)
- **`run_ablation.py`** — KD temperature and alpha sweep script
- **Updated `generate_report.py`** — auto-detects available results, generates fig1-fig7

### Changed
- `src/training/trainer.py`: reads hyperparams from `DL_CONFIG` instead of hardcoding
- `src/models/distillation.py`: reads hyperparams from `DL_CONFIG`
- `src/training/__init__.py`: exports `train_student_kd_loso`
- `trainer.py` CSV writer: teacher param count now computed dynamically

---

## [Phase 1] Traditional ML Pipeline — 2026-03-24

### Results (from `python main.py`)

| Model               | Accuracy     | Recall       | F1           | ROC-AUC      |
|---------------------|--------------|--------------|--------------|--------------|
| Random Baseline     | ~0.50        | 0.358        | 0.364        | --           |
| Majority Baseline   | ~0.67        | 0.000        | 0.000        | --           |
| EDA Threshold       | ~0.60        | 0.866        | 0.792        | --           |
| Logistic Regression | 0.964±0.072  | 0.954±0.134  | 0.947±0.105  | 0.976±0.062  |
| Random Forest       | 0.966±0.077  | 0.965±0.081  | 0.956±0.086  | 0.996±0.011  |

### Added
- Full data pipeline: `src/data/`, `src/preprocessing/`, `src/segmentation/`
- Hand-crafted feature extraction (~150 features): `src/features/`
- Traditional ML classifiers: `src/models/classifiers.py`
- LOSO evaluation: `src/evaluation/splitting.py`, `src/evaluation/metrics.py`
- EDA visualization: `eda_visualization.py`
- Documentation: `docs/decision_log.md`, `docs/known_unknowns.md`, `docs/technical_report.md`

---

## [Phase 1.5] Teacher CNN (Sequential) — 2026-03-24

### Results (from `python train_teacher.py` — sequential 3-block CNN)

| Model         | Accuracy     | Recall       | F1           | ROC-AUC      |
|---------------|--------------|--------------|--------------|--------------|
| 1D-CNN Teacher| 0.988±0.024  | 0.986±0.041  | 0.983±0.035  | 0.995±0.016  |

> Note: This was the first CNN training run. The architecture has since been
> replaced by the multi-scale parallel-branch design. **Re-run `train_teacher.py`
> to get updated results with the 266K multi-scale teacher.**

### Added
- `src/data/dl_dataset.py`: PyTorch Dataset with 700→64 Hz downsampling + NaN sanitisation
- `src/models/teacher.py` (sequential version, now replaced)
- `src/training/trainer.py`: LOSO training loop with WeightedRandomSampler
- `src/evaluation/reporter.py`: 3 publication figures (fig1-fig3)
- `train_teacher.py`, `generate_report.py`

### Fixed
- Class collapse (Recall=0): WeightedRandomSampler + lower LR
- NaN loss: `nan_to_num` before/after FFT resample + NaN batch skip
- Unicode crash: replaced `→` / `±` with ASCII equivalents for Windows cp1252
