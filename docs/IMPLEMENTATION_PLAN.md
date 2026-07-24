# Plan: Resource-Efficient Health Anomaly Detection on Wearable Devices

## Context

**Current State**: The project has a well-architected traditional ML pipeline (handcrafted features + LogReg/RF) achieving 72-75% accuracy on LOSO cross-validation for binary stress detection on WESAD. It has excellent documentation, modular code, and proper validation. However, it has **no deep learning models** despite the project title mentioning "Lightweight Deep Models."

**Problem**: The project is stuck at Phase 5 (traditional ML) and lacks the novelty needed for a conference paper. The current results (72-75% LOSO accuracy) are significantly below SOTA (~88-93% for cross-subject). The title promises lightweight deep models and efficiency, but the codebase delivers neither.

**Research Gap Identified**: After surveying 20+ recent papers (2023-2026), we identified that **Knowledge Distillation for model-size compression on WESAD** is essentially untried. Existing KD work (PULSE, Apple) focuses on cross-modality transfer, not compressing a large stress model into a TinyML-deployable one. Additionally, **deep anomaly detection** for stress is unexplored on WESAD.

**Goal**: Transform this into a conference-paper-worthy project with a clear novelty: **a unified pipeline that trains a strong teacher model, distills it into a TinyML-ready student, and benchmarks efficiency vs. accuracy trade-offs for wearable deployment.**

---

## Proposed Novelty Statement (Paper Angle)

> "We propose a Knowledge-Distillation-based pipeline for resource-efficient stress detection on wearable devices. A multi-scale 1D-CNN teacher achieves state-of-the-art cross-subject accuracy on WESAD, which is then distilled into a sub-100KB student model suitable for microcontroller deployment, with less than 3% accuracy degradation. We further introduce efficiency-aware evaluation metrics (Params, FLOPs, inference latency, model size) alongside traditional classification metrics, establishing a new benchmark for lightweight stress detection."

---

## Implementation Plan

### Phase A: Deep Learning Foundation (Priority: CRITICAL)
> Transition from handcrafted features to learned representations

#### A1. Raw Signal Data Pipeline for Deep Learning
- **File**: `src/data/dl_dataset.py` (NEW)
- Create a PyTorch `Dataset` class that:
  - Takes windowed raw signals (not extracted features) as input
  - Stacks multi-modal signals into a multi-channel tensor: shape `(C, T)` where C=6 channels (ECG, EDA, EMG, Resp, Temp, ACC_mag), T=samples per window
  - Downsamples from 700Hz to 64Hz or 128Hz (configurable) to reduce input size while preserving stress-relevant info (stress signals are <40Hz)
  - Applies the existing preprocessing (filtering + normalization) before windowing
  - Returns `(signal_tensor, label, subject_id)`
- **File**: `src/data/dl_dataloader.py` (NEW)
  - LOSO DataLoader factory: given a held-out subject, returns train/val/test loaders
  - Implements within-train subject-wise split for validation (e.g., 1 of 14 train subjects)

#### A2. Teacher Model: Multi-Scale 1D-CNN (`src/models/teacher.py` NEW)
- **Architecture**: Multi-Scale Convolutional Network
  - 3 parallel branches with different kernel sizes (small=8, medium=32, large=64) to capture multi-scale temporal patterns
  - Each branch: Conv1D -> BatchNorm -> ReLU -> Conv1D -> BatchNorm -> ReLU -> AdaptiveAvgPool
  - Concatenate branch outputs -> FC layers -> Binary classification
  - ~500K-1M parameters (large enough to learn well, small enough to train on this dataset)
- **Why multi-scale**: Stress manifests at different timescales (heartbeat=fast, EDA=slow, respiration=medium). Multi-scale kernels capture all of them. This is a defensible architectural choice for the paper.
- **Training**: Binary cross-entropy, AdamW optimizer, cosine annealing LR, early stopping on validation loss

#### A3. Lightweight Student Models (`src/models/student.py` NEW)
Three student architectures at different efficiency points:

| Model | Params | Target Size | Architecture |
|-------|--------|-------------|--------------|
| **MicroCNN** | ~5K | <20 KB | 2-layer depthwise-separable Conv1D + GAP + FC |
| **TinyCNN** | ~15K | <60 KB | 3-layer standard Conv1D + GAP + FC |
| **MiniCNN-LSTM** | ~30K | <120 KB | 2-layer Conv1D + single-layer LSTM(32) + FC |

- Use depthwise-separable convolutions (MobileNet-style) for the smallest model
- Global Average Pooling (GAP) instead of flatten to reduce FC parameters
- All models should be quantization-friendly (no complex operations)

#### A4. Knowledge Distillation Pipeline (`src/models/distillation.py` NEW)
- **Method**: Response-based KD (Hinton et al.) + optional feature-based KD
  - Loss = alpha * KL_div(student_logits/T, teacher_logits/T) + (1-alpha) * CE(student_logits, labels)
  - Temperature T=3-5 (softens teacher probabilities to transfer "dark knowledge")
  - Alpha=0.7 (weight more on teacher's knowledge)
- **Feature distillation** (optional, for paper ablation): align intermediate representations via MSE loss between teacher and student feature maps (with a projection layer to match dimensions)
- Teacher is frozen during distillation

---

### Phase B: Efficiency Benchmarking Framework (Priority: HIGH)
> The key differentiator - no one benchmarks efficiency properly on WESAD

#### B1. Efficiency Metrics Module (`src/evaluation/efficiency.py` NEW)
- **Parameter count**: Total trainable parameters
- **Model size**: Saved model file size in KB (FP32 and INT8-quantized)
- **FLOPs**: Multiply-accumulate operations per inference (use `thop` or manual calculation)
- **Inference latency**: Mean inference time per window on CPU (simulate edge device)
- **Memory footprint**: Peak memory during inference
- **Efficiency score**: Custom composite metric, e.g., `F1 / log(params)` or `F1 * (1 / model_size_KB)`

#### B2. Post-Training Quantization (`src/models/quantization.py` NEW)
- INT8 dynamic quantization via PyTorch's `torch.quantization`
- Measure accuracy drop after quantization
- Report quantized model size (target: <50KB for MicroCNN)

#### B3. Comprehensive Comparison Table
- Compare across all models: Traditional ML (existing), Teacher, 3 Students (standalone + distilled), Quantized students
- Metrics: Accuracy, F1, Recall, ROC-AUC, Params, FLOPs, Size(KB), Latency(ms)
- This table IS the paper's main contribution

---

### Phase C: Training & Evaluation Infrastructure (Priority: HIGH)

#### C1. Training Script (`src/training/trainer.py` NEW)
- Unified trainer for teacher, standalone student, and distillation
- LOSO cross-validation loop (reuse existing subject split logic)
- Logging: loss curves, per-fold metrics, early stopping
- Checkpointing: save best model per fold
- Config-driven (extend existing `src/config.py`)

#### C2. Extended Config (`src/config.py` MODIFY)
Add deep learning configuration:
```python
DL_CONFIG = {
    'target_sr': 64,           # Downsampled sampling rate
    'batch_size': 32,
    'epochs': 100,
    'patience': 15,            # Early stopping
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'kd_temperature': 4.0,
    'kd_alpha': 0.7,
}
```

#### C3. Results Aggregation (`src/evaluation/results.py` NEW)
- Collect all metrics into a unified results table
- Generate LaTeX-ready tables for the paper
- Generate comparison plots (accuracy vs. model size Pareto front)

---

### Phase D: Ablation Studies (Priority: MEDIUM)
> Strengthen the paper with controlled experiments

1. **KD temperature sweep**: T in {1, 2, 3, 4, 5, 8} - show impact on student accuracy
2. **KD alpha sweep**: alpha in {0.0, 0.3, 0.5, 0.7, 0.9, 1.0}
3. **Student standalone vs. distilled**: Train same student architecture with and without KD
4. **Downsampling rate**: 32Hz vs 64Hz vs 128Hz impact on accuracy and speed
5. **Modality ablation**: Which sensor channels matter most? (drop one at a time)
6. **Feature-based vs response-based KD**: Does adding intermediate feature alignment help?

---

### Phase E: Visualization & Paper Assets (Priority: MEDIUM)

#### E1. Figures to Generate
- **Pareto front**: Accuracy vs Model Size (KB) scatter plot showing all models
- **KD improvement**: Bar chart showing student accuracy with/without distillation
- **Per-subject LOSO heatmap**: Which subjects are hard/easy for each model?
- **Ablation plots**: Temperature and alpha sweeps
- **Architecture diagram**: Teacher + Student + KD pipeline (can be done in draw.io)
- **t-SNE/UMAP**: Learned embeddings colored by stress/baseline

#### E2. Documentation
- **File**: `docs/CHANGELOG.md` (NEW) - Track all major changes
- **File**: `docs/deep_learning_report.md` (NEW) - Results and analysis
- Update `docs/technical_report.md` with deep learning sections

---

### Phase F: Stretch Goals (Priority: LOW, if time permits)

1. **Self-supervised pretraining**: Masked signal modeling on unlabeled windows, then fine-tune. Would strengthen the paper significantly but adds complexity.
2. **ONNX export**: Convert best student to ONNX for cross-platform deployment demo
3. **Anomaly detection variant**: Train autoencoder on "normal" (baseline) data, detect stress as reconstruction anomaly. Novel angle but separate from KD story.

---

## Implementation Order & Dependencies

```
Week 1: Phase A1 + A2 (data pipeline + teacher model)
         Phase C1 + C2 (trainer + config)
         -> Milestone: Teacher model trained with LOSO, baseline DL accuracy established

Week 2: Phase A3 + A4 (student models + KD)
         Phase B1 (efficiency metrics)
         -> Milestone: KD pipeline working, students trained, efficiency measured

Week 3: Phase B2 + B3 (quantization + comparison table)
         Phase D (ablation studies)
         -> Milestone: Full results table, ablation data

Week 4: Phase E (visualization + documentation)
         Phase F (stretch goals if time permits)
         -> Milestone: Paper-ready figures and tables
```

---

## Key Files to Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/config.py` | MODIFY | Add DL hyperparameters |
| `src/data/dl_dataset.py` | NEW | PyTorch dataset for raw signals |
| `src/data/dl_dataloader.py` | NEW | LOSO DataLoader factory |
| `src/models/teacher.py` | NEW | Multi-scale 1D-CNN teacher |
| `src/models/student.py` | NEW | 3 lightweight student architectures |
| `src/models/distillation.py` | NEW | KD training logic |
| `src/models/quantization.py` | NEW | INT8 quantization |
| `src/training/trainer.py` | NEW | Unified training loop |
| `src/evaluation/efficiency.py` | NEW | Params, FLOPs, size, latency |
| `src/evaluation/results.py` | NEW | Results aggregation + LaTeX tables |
| `requirements.txt` | MODIFY | Add torch, thop |
| `main.py` | MODIFY | Add DL pipeline entry points |

---

## Verification Plan

1. **Unit test**: Teacher forward pass with dummy tensor `(B=4, C=6, T=4096)` produces `(4, 2)` output
2. **Unit test**: Each student forward pass with same input shape
3. **Integration test**: Full LOSO on 2 subjects (fast sanity check)
4. **Full run**: LOSO on all 15 subjects for teacher, then KD for each student
5. **Efficiency check**: Verify MicroCNN < 20KB after quantization
6. **Comparison check**: Distilled student should beat standalone student (if not, debug KD hyperparams)

---

## Expected Paper Contribution

1. **First KD-based model compression study on WESAD** for stress detection
2. **Efficiency-aware benchmarking** (Pareto analysis of accuracy vs. model size)
3. **Three deployment-ready student models** at different efficiency points (5K-30K params)
4. **Comprehensive ablation** of KD hyperparameters for physiological signals
5. **Reproducible, modular codebase** (already a strength of this project)

**Target venue**: ACM/IEEE conferences on health informatics, wearable computing, or edge AI (e.g., UbiComp, PerCom, BSN, EMBC, or a workshop at NeurIPS/AAAI)
