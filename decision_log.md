# Decision Log - Binary Stress Anomaly Detection Pipeline

This document records major decisions, rationale, alternatives considered, and assumptions for each component.

---

## Phase 1: Problem Formalization

### Decision: 60-Second Window Length
- **Why this choice**: 60 seconds provides sufficient time for physiological stress responses (EDA latency ~1-3s, full response ~10-15s)
- **Alternatives rejected**: 
  - 30 seconds: May miss slower stress responses
  - 120 seconds: Fewer samples, less granular detection
- **Assumptions**: Stress states persist for at least 60 seconds in WESAD protocol

### Decision: 50% Window Overlap
- **Why this choice**: Balances sample count vs. window independence
- **Alternatives rejected**:
  - 0% overlap: Too few training samples
  - 75% overlap: High correlation between adjacent windows
- **Assumptions**: 50% provides reasonable trade-off for this dataset size

### Decision: 80% Purity Threshold
- **Why this choice**: Ensures window has clear physiological state
- **Alternatives rejected**:
  - 100%: Too strict, loses many windows near transitions
  - 60%: Allows ambiguous mixed-state windows
- **Assumptions**: Majority label represents dominant physiological state

### Decision: 5-Second Transition Buffer
- **Why this choice**: Physiological transitions are gradual, not instantaneous
- **Assumptions**: 5 seconds is sufficient for transition dynamics

---

## Phase 2: Data Loading

### Decision: Use Pickle Files (Not Raw Text)
- **Why this choice**: Preprocessed and synchronized by dataset authors
- **Alternatives rejected**: Raw `.txt` files require manual synchronization
- **Assumptions**: Authors' preprocessing is correct

### Decision: Binary Classification (Baseline vs Stress Only)
- **Why this choice**: Matches problem statement - anomaly detection
- **Alternatives rejected**: Multi-class (amusement, meditation) - out of scope
- **Assumptions**: Other states are excluded, not mislabeled

---

## Phase 2.5: Signal Conditioning

### Decision: Per-Modality Bandpass Filtering
- **Why this choice**: Each signal has different frequency characteristics
- **Filter choices**:
  - ECG: 0.5-40 Hz (remove baseline wander and HF noise)
  - EDA: 1 Hz lowpass (slow-varying signal)
  - EMG: 20-450 Hz (muscle activity range)
  - Resp: 0.1-0.5 Hz (breathing frequency)
- **Assumptions**: Filter parameters are appropriate for 700 Hz sampling

### Decision: Z-Score Normalization Per Subject
- **Why this choice**: Removes inter-subject baseline differences
- **Alternatives rejected**: Global normalization (loses subject-specific patterns)
- **Assumptions**: Within-subject variation is meaningful

---

## Phase 3: Windowing

### Decision: Chest Signals Only (for initial implementation)
- **Why this choice**: Higher quality, single sampling rate (700 Hz)
- **Alternatives rejected**: Wrist signals have lower quality, different rates
- **Future work**: Add wrist signals after baseline validation

### Decision: Exclude Transition Windows
- **Why this choice**: Ambiguous physiological state
- **Assumptions**: Transitions don't carry useful signal

---

## Phase 4: Feature Engineering

### Decision: Hand-Crafted Features (Not Deep Learning)
- **Why this choice**: 
  - Interpretable for edge deployment
  - Small dataset (15 subjects)
  - Matches project goals
- **Alternatives rejected**: CNN/LSTM (black box, overfitting risk)
- **Assumptions**: Traditional features capture stress patterns

### Decision: Statistical + Temporal + Frequency Features
- **Why this choice**: Comprehensive coverage of signal characteristics
- **Feature count**: ~150 features per window
- **Future work**: Feature selection for edge deployment

---

## Phase 5: Model Selection

### Decision: Logistic Regression + Random Forest as Baselines
- **Why this choice**: 
  - Simple, interpretable
  - Good baselines for comparison
  - Fast training
- **Alternatives rejected**: SVM, XGBoost (more complex, less interpretable)
- **Assumptions**: Simple models can capture stress patterns

---

## Data Splitting Strategy

### Decision: LOSO as Primary Evaluation
- **Why this choice**: Measures true generalization to unseen subjects
- **Alternatives rejected**: Random split (data leakage within subject)
- **Assumptions**: Subject independence is critical for real-world deployment

### Decision: Fixed Subject Split for Debug
- **Why this choice**: Faster iteration during development
- **Test subjects**: S15, S16, S17 (held out consistently)
