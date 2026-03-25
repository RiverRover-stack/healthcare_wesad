# Deep Learning Report: Knowledge Distillation for Stress Detection on WESAD

**Project**: Resource-Efficient Health Anomaly Detection on Wearable Devices
**Dataset**: WESAD (15 subjects, RespiBAN chest sensor, 700 Hz)
**Task**: Binary stress classification (baseline=0 vs stress=1) under LOSO CV

---

## 1. Motivation

The traditional ML pipeline (Phase 1) achieves strong results (RF: F1=0.956, AUC=0.996) but relies on ~150 hand-crafted features. This approach:
- Requires domain expertise to engineer
- Cannot be end-to-end trained
- Has no path to TinyML deployment (the feature extractor alone is too large)

The deep learning phase replaces hand-crafted features with learned representations and adds a **knowledge distillation** compression step to produce models small enough for microcontroller deployment.

---

## 2. Architecture

### 2.1 Teacher: Multi-Scale 1D-CNN (~266K parameters)

**Key insight**: Stress manifests at multiple physiological timescales simultaneously:

| Branch   | Kernel | Captures                              | Example signal |
|----------|--------|---------------------------------------|----------------|
| Small    | k=8    | Fast local patterns (heartbeat peaks) | ECG R-peaks    |
| Medium   | k=32   | Mid-range patterns (respiration)      | Resp cycles    |
| Large    | k=64   | Slow trends (EDA rise, temp drift)    | EDA, Temp      |

Three parallel branches process the input independently, then their outputs are concatenated and passed through a shared FC classification head.

```
Input (B, 6, 3840)
├── Branch Small  (k=8):  Conv->BN->ReLU->Conv->BN->ReLU->GAP -> (B, 64)
├── Branch Medium (k=32): Conv->BN->ReLU->Conv->BN->ReLU->GAP -> (B, 64)
└── Branch Large  (k=64): Conv->BN->ReLU->Conv->BN->ReLU->GAP -> (B, 64)
                                                                    ↓
                                                              Concat (B, 192)
                                                         FC(192→128)→ReLU→Drop
                                                           FC(128→64)→ReLU
                                                              FC(64→2)
```

### 2.2 Student Models

Three student architectures target different deployment scenarios:

| Model        | Params  | Key technique             | Target device         |
|--------------|---------|---------------------------|-----------------------|
| MicroCNN     | ~5.3K   | Depthwise-separable Conv  | Microcontroller (MCU) |
| TinyCNN      | ~15.2K  | 3-layer standard Conv1D   | BLE SoC / wearable    |
| MiniCNN-LSTM | ~28.8K  | Conv1D + LSTM(40)         | Smartphone / edge hub |

**MicroCNN** uses depthwise-separable convolutions (MobileNet-style): each standard Conv1d(in, out, k) is replaced by a depthwise Conv1d(in, in, k, groups=in) + pointwise Conv1d(in, out, 1). This reduces FLOP count ~8x for the convolutional layers.

**MiniCNN-LSTM** is the most expressive student. After two Conv+MaxPool blocks that compress the sequence to 240 time steps, an LSTM processes the temporal sequence to capture long-range dependencies that a pure CNN would miss.

### 2.3 Knowledge Distillation (Response-Based)

Following Hinton et al. (2015), the student is trained on a combination of:
- **Soft targets** from the frozen teacher (carry "dark knowledge" about inter-class relationships)
- **Hard targets** from ground-truth labels

```
L_KD = α × T² × KL( softmax(s/T) ‖ softmax(t/T) )
      + (1-α) × CE(s, y)

T = 4.0   (temperature: higher = softer teacher distribution)
α = 0.7   (weight: 70% knowledge from teacher, 30% from ground-truth)
T² factor restores gradient scale reduced by temperature (Hinton 2015)
```

---

## 3. Training Details

### 3.1 Class Imbalance

The dataset is ~64% baseline / 36% stress. Without correction, models collapse to predicting all-baseline (Recall=0, F1=0). Two-layer defence:

1. **WeightedRandomSampler**: forces ~50/50 class balance in every mini-batch
2. **Class-weighted CrossEntropyLoss**: inverse-frequency weights as secondary defence

### 3.2 NaN Sanitisation

Some subjects have all-NaN respiration segments. NaN values corrupt FFT-based resampling, propagate through BatchNorm, and cause `loss=NaN` → all gradients=NaN → `argmax(NaN)=0` (class collapse). Fix:

```python
window = np.nan_to_num(window, nan=0.0, posinf=5.0, neginf=-5.0)
resampled = resample(window, TARGET_LENGTH, axis=1)
resampled = np.nan_to_num(resampled, nan=0.0)
resampled = np.clip(resampled, -5.0, 5.0)
```

Gradient clipping (`max_norm=1.0`) is also applied as a final safety net.

### 3.3 Hyperparameters

| Parameter     | Teacher | Student | Notes                              |
|---------------|---------|---------|-------------------------------------|
| Batch size    | 16      | 16      | Small: ~380 total windows           |
| Epochs        | 40      | 30      | Students converge faster            |
| LR            | 3e-4    | 3e-4    | AdamW; 1e-3 was too aggressive      |
| Weight decay  | 1e-3    | 1e-3    | L2 regularization                   |
| Scheduler     | ReduceLROnPlateau (F1, patience=5) | same |           |
| KD Temperature| —       | 4.0     | Ablation range: [1, 2, 4, 8]       |
| KD Alpha      | —       | 0.7     | Ablation range: [0.3, 0.5, 0.7, 0.9]|

All hyperparameters are in `src/config.py::DL_CONFIG`.

---

## 4. Results

### 4.1 Current Results (after Phase 1.5 teacher training)

| Model                   | Accuracy     | Recall       | F1           | ROC-AUC      | Params |
|-------------------------|--------------|--------------|--------------|--------------|--------|
| Random Baseline         | ~0.50        | 0.358        | 0.364        | --           | --     |
| Majority Baseline       | ~0.67        | 0.000        | 0.000        | --           | --     |
| EDA Threshold           | ~0.60        | 0.866        | 0.792        | --           | --     |
| Logistic Regression     | 0.964±0.072  | 0.954±0.134  | 0.947±0.105  | 0.976±0.062  | ~150 feat |
| Random Forest           | 0.966±0.077  | 0.965±0.081  | 0.956±0.086  | 0.996±0.011  | ~150 feat |
| 1D-CNN Teacher (seq.)   | 0.988±0.024  | 0.986±0.041  | 0.983±0.035  | 0.995±0.016  | ~45K   |
| **Teacher (Multi-Scale)**| TBD         | TBD          | TBD          | TBD          | ~266K  |
| MicroCNN (standalone)   | TBD          | TBD          | TBD          | TBD          | ~5.3K  |
| MicroCNN (distilled)    | TBD          | TBD          | TBD          | TBD          | ~5.3K  |
| TinyCNN (standalone)    | TBD          | TBD          | TBD          | TBD          | ~15.2K |
| TinyCNN (distilled)     | TBD          | TBD          | TBD          | TBD          | ~15.2K |
| MiniCNN-LSTM (standalone)| TBD         | TBD          | TBD          | TBD          | ~28.8K |
| MiniCNN-LSTM (distilled) | TBD         | TBD          | TBD          | TBD          | ~28.8K |

> TBD rows: run `python train_teacher.py` then `python train_students.py`

### 4.2 Expected Findings (from prior literature)

- Distilled students should outperform standalone students by 2-5% F1
- Temperature T=4 typically outperforms T=1 (too sharp) and T=8 (too soft)
- The teacher (266K) should match or exceed the sequential (45K) — more capacity helps with multi-scale capture
- MiniCNN-LSTM is expected to be the best student (recurrent memory helps with EDA trends)

---

## 5. Efficiency Analysis

> Requires `python train_students.py` to populate, then run `generate_report.py`

### Planned Efficiency Table

| Model             | Params  | FP32 Size (KB) | INT8 Size (KB) | CPU Latency (ms) |
|-------------------|---------|----------------|----------------|------------------|
| Teacher (MS)      | ~266K   | ~1,050 KB      | ~270 KB        | ~120 ms          |
| MicroCNN          | ~5.3K   | ~21 KB         | ~6 KB          | ~5 ms            |
| TinyCNN           | ~15.2K  | ~60 KB         | ~16 KB         | ~8 ms            |
| MiniCNN-LSTM      | ~28.8K  | ~115 KB        | ~30 KB         | ~15 ms           |

> Estimates. Actual values depend on hardware. See `fig4_pareto_front.png`.

### Deployment Targets

| Model        | Target        | Constraint             | Status   |
|--------------|---------------|------------------------|----------|
| Teacher      | GPU server    | Training/research only | Ready    |
| MicroCNN     | ARM Cortex-M4 | < 32KB flash           | Pending  |
| TinyCNN      | nRF52840      | < 256KB flash          | Pending  |
| MiniCNN-LSTM | ESP32-S3      | < 512KB, has PSRAM     | Pending  |

---

## 6. How to Run

```bash
# Step 1: Traditional ML baseline
python main.py

# Step 2: Train multi-scale teacher (LOSO, ~2-4 hours CPU)
python train_teacher.py

# Step 3: Train all students (standalone + distilled, ~4-8 hours CPU)
python train_students.py

# Step 4: Ablation study (optional, ~2 hours CPU)
python run_ablation.py

# Step 5: Generate all report figures
python generate_report.py
# -> outputs/reports/fig1-fig7.png
# -> outputs/reports/table_accuracy.tex
# -> outputs/reports/table_efficiency.tex
```

---

## 7. Open Questions

1. **Will KD improve small students on WESAD?** The dataset is tiny (~380 windows). With limited data, the teacher may not produce reliable soft targets. We expect yes, but the ablation will tell.

2. **Does LSTM help?** MiniCNN-LSTM adds recurrent memory. EDA has slow temporal structure that a pure CNN may miss. Compare MiniCNN-LSTM vs TinyCNN to test this.

3. **Subject S3 and S9**: Both consistently score lower across all models (LogReg F1=0.643/0.914, RF F1=0.857/0.848). Understanding why may inform preprocessing. Possible causes: unusual stress protocol response, signal quality, mislabeled data.

4. **Multi-scale vs sequential teacher**: The new 266K multi-scale teacher should perform better than the 45K sequential one, but the evidence isn't in yet. If it doesn't, the architectural claim weakens — we'd need to document this honestly.

5. **Quantization efficacy**: Dynamic INT8 quantization has limited effect on Conv1d layers. Static quantization (with calibration data) would give better compression. This is a known limitation to address in future work.
