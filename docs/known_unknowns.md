# Known Unknowns

This document tracks implementation details that are acknowledged but not fully understood yet.
These items should be revisited for deeper understanding.

---

## Signal Processing

### 1. Optimal Filter Parameters
- **Status**: Using literature-based defaults
- **Unknown**: Are these optimal for stress detection?
- **To revisit**: After baseline results, experiment with different filter cutoffs

### 2. Respiration Signal NaN Issues
- **Status**: Some subjects show NaN after filtering
- **Unknown**: Root cause (sensor issues? filter instability?)
- **To revisit**: Investigate specific subjects, consider robust filtering

### 3. EMG Highcut Frequency
- **Status**: Using 450 Hz but Nyquist is 350 Hz at 700 Hz sampling
- **Unknown**: Impact of capping at Nyquist
- **To revisit**: Verify EMG frequency content in this dataset

---

## Feature Engineering

### 4. EDA Decomposition Accuracy
- **Status**: Using simple moving average for tonic/phasic split
- **Unknown**: How accurate vs. dedicated libraries (neurokit2)?
- **To revisit**: Compare with proper decomposition methods

### 5. Frequency Band Definitions
- **Status**: Using generic bands (VLF, LF, HF)
- **Unknown**: Optimal bands for stress detection
- **To revisit**: Literature review on stress-specific frequency bands

### 6. Feature Correlation
- **Status**: Not checked yet
- **Unknown**: How many features are redundant?
- **To revisit**: Correlation analysis and feature selection

---

## Windowing

### 7. Transition Buffer Duration
- **Status**: Using 5 seconds
- **Unknown**: Is this physiologically justified?
- **To revisit**: Literature on stress response dynamics

### 8. Purity Threshold Sensitivity
- **Status**: Using 80%
- **Unknown**: Impact on model performance
- **To revisit**: Sensitivity analysis on threshold value

---

## Modeling

### 9. Class Imbalance Handling
- **Status**: Using class_weight='balanced'
- **Unknown**: Is this sufficient? Should we try SMOTE?
- **To revisit**: Compare balanced weights vs. resampling

### 10. LOSO Variance
- **Status**: Expected high variance due to subject differences
- **Unknown**: Which subjects are outliers?
- **To revisit**: Subject-wise analysis after LOSO

---

## Dataset-Specific

### 11. Subject S12 Missing
- **Status**: Dataset has 15 subjects (S2-S17, no S12)
- **Unknown**: Why? Corrupted data?
- **Impact**: One less subject for training/evaluation

### 12. Temperature Sensor Notes
- **Status**: S2_readme.txt mentions sensor attachment issues
- **Unknown**: Impact on temperature features
- **To revisit**: Check temperature data quality per subject

---

## Tracking

| ID | Item | Priority | Status |
|----|------|----------|--------|
| 1 | Filter parameters | Medium | Open |
| 2 | Resp NaN | High | Open |
| 3 | EMG Nyquist | Low | Open |
| 4 | EDA decomposition | Medium | Open |
| 5 | Frequency bands | Medium | Open |
| 6 | Feature correlation | Medium | Open |
| 7 | Transition buffer | Low | Open |
| 8 | Purity threshold | Low | Open |
| 9 | Class imbalance | High | Open |
| 10 | LOSO variance | High | Open |
| 11 | Subject S12 | Low | Documented |
| 12 | Temp sensor | Low | Open |
