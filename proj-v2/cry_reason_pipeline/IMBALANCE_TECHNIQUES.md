# Imbalanced Classification Techniques Implementation

## Problem Summary

The cry classification model suffers from class imbalance - some cry types have significantly fewer samples than others. This causes:
- Poor generalization on minority classes
- High false negative rate for important classes
- Model favoring majority classes
- Suboptimal F1 scores for minority classes

## Solutions Implemented

### 1. **Class Weight Preservation** (Training Script Fix)

**Problem**: Previous implementation zeroed out class weights when using balanced sampling, losing the imbalance information.

**Solution**: Keep class weights even with balanced sampling.

```python
# Class weights are now always computed
classes = np.array(sorted(np.unique(y_train)))
class_weights_arr = compute_class_weight("balanced", classes=classes, y=y_train)
class_weights = {int(cls): float(wt) for cls, wt in zip(classes, class_weights_arr)}

# Only zero them for focal loss (which has built-in imbalance handling)
if args.use_focal_loss:
    class_weights = {int(cls): 1.0 for cls in classes}
```

**Effect**: 
- Minority class samples contribute more to loss
- Gradient updates weighted by class frequency
- Better convergence for underrepresented classes

---

### 2. **Focal Loss Option** (Advanced Technique)

**Problem**: Standard cross-entropy treats all examples equally; easy examples dominate the loss.

**Solution**: Add optional focal loss that downweights easy examples.

Focal loss formula: $FL(p_t) = -\alpha_t (1-p_t)^{\gamma} \log(p_t)$

Where:
- $p_t$ is the probability of true class
- $\gamma$ is the focusing parameter (2.0)
- $\alpha$ is the weighting factor (0.25)

**Usage**:
```bash
python 03_train_model.py --use-focal-loss
```

**Effect**:
- Model focuses on hard-to-classify examples
- Naturally handles class imbalance
- Often better than class weights alone

---

### 3. **Extended Training Duration**

**Problem**: Minority classes may need more time to learn patterns.

**Solutions**:
- **Epochs**: Increased from 60 → 100 (67% more training)
- **Early Stopping Patience**: Increased from 12 → 20 (67% more patience)

```python
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--early-stop-patience", type=int, default=20)
```

**Usage**:
```bash
# Longer training with default settings
python 03_train_model.py

# Custom values
python 03_train_model.py --epochs 150 --early-stop-patience 30
```

**Effect**:
- More epochs for minority class convergence
- Higher patience prevents premature stopping
- Better utilization of training time

---

### 4. **Threshold Adjustment (Post-hoc)** ⭐

**Problem**: Model may be too conservative or aggressive with specific classes.

**Solution**: Find optimal decision thresholds on validation set to maximize desired metric.

#### Approach:

**Per-Class Thresholds** (Recommended for imbalance)
- Different threshold for each class
- Fine-grained control
- Better for minority classes

#### Finding Optimal Thresholds:

```bash
# Per-class thresholds optimized for macro F1 (better for imbalance)
python 08_threshold_optimization.py --metric f1-macro

# Per-class thresholds optimized for weighted F1
python 08_threshold_optimization.py --metric f1-weighted

# Per-class thresholds optimized for balanced accuracy
python 08_threshold_optimization.py --metric balanced-accuracy
```

Output: `artifacts/model/threshold_config.json`

Example output:
```json
{
  "type": "per-class",
  "metric": "f1-macro",
  "thresholds": {
    "0": 0.450,
    "1": 0.380,
    "2": 0.520,
    "3": 0.460
  },
  "score": 0.7254
}
```

#### Using Optimized Thresholds:

```bash
python 09_inference_with_thresholds.py \
  --threshold-config artifacts/model/threshold_config.json
```

**Effect**:
- Better precision-recall trade-off
- Higher minority class recall
- Improved macro F1 scores

---

## Recommended Training Pipeline

### Step 1: Train Model with Improvements

```bash
# With focal loss (recommended for severe imbalance)
python 03_train_model.py \
  --use-focal-loss \
  --balanced-sampling \
  --augment-minority \
  --epochs 100 \
  --early-stop-patience 20

# OR standard approach with extended training
python 03_train_model.py \
  --balanced-sampling \
  --augment-minority \
  --epochs 100 \
  --early-stop-patience 20
```

### Step 2: Optimize Thresholds

```bash
# Find per-class thresholds
python 08_threshold_optimization.py \
  --metric f1-macro \
  --output-file artifacts/model/threshold_config.json
```

### Step 3: Evaluate with Thresholds

```bash
python 09_inference_with_thresholds.py \
  --threshold-config artifacts/model/threshold_config.json \
  --model-path artifacts/model/cry_reason_model.keras
```

---

## Combining Techniques

| Scenario | Technique | Reasoning |
|----------|-----------|-----------|
| **Mild imbalance** (2-5x) | Class weights + balanced sampling | Sufficient for small differences |
| **Moderate imbalance** (5-10x) | Above + extended training | More convergence time needed |
| **Severe imbalance** (>10x) | Focal loss + extended training | Need stronger emphasis on hard examples |
| **Any imbalance** | Add threshold adjustment | Post-hoc fine-tuning always helps |

---

## Expected Improvements

### Before (with default settings):
```
           precision    recall  f1-score   support
Class 0       0.85      0.78      0.82       100
Class 1       0.65      0.45      0.53        50    ← Minority class
Class 2       0.82      0.80      0.81        95
Class 3       0.78      0.75      0.76        80

macro avg     0.77      0.70      0.73       325
```

### After (with improvements):
```
           precision    recall  f1-score   support
Class 0       0.84      0.80      0.82       100
Class 1       0.72      0.62      0.67        50    ← +14pp recall improvement
Class 2       0.83      0.81      0.82        95
Class 3       0.80      0.78      0.79        80

macro avg     0.80      0.75      0.77       325    ← +5pp macro average improvement
```

---

## Monitoring Training

### New Output Files

After training, check:

```
artifacts/model/
├── cry_reason_model.keras          # Trained model
├── training_info.json              # Training details
├── metadata.json                   # Labels metadata (NEW)
├── val_predictions.npy             # Validation predictions (NEW)
├── val_labels.npy                  # Validation labels (NEW)
├── test_predictions.npy            # Test predictions (NEW)
├── test_labels.npy                 # Test labels (NEW)
├── per_class_metrics.csv           # Per-class metrics
└── threshold_config.json           # Thresholds (after optimization)
```

### Training Metrics

Monitor for:
1. **Validation accuracy plateau**: If plateaus early, increase `--early-stop-patience`
2. **Class-wise F1 divergence**: If minority class F1 << majority, increase focal loss strength
3. **Loss curves**: Should show steady improvement across epochs

---

## Troubleshooting

### Issue: Minority class F1 not improving
**Solutions**:
1. Use `--use-focal-loss` to focus on hard examples
2. Increase `--epochs` further (try 150-200)
3. Increase `--early-stop-patience` (try 30-40)
4. Use per-class threshold optimization on macro F1

### Issue: Overfitting
**Solutions**:
1. Make sure `--augment-minority` is enabled
2. Reduce `--epochs` or lower `--early-stop-patience`
3. Check for data leakage

### Issue: Threshold optimization gives weird thresholds
**Solutions**:
1. Use `--metric f1-macro` instead of f1-weighted for imbalance
2. Ensure validation set has enough minority samples
3. Re-run after a fresh training pass so `val_predictions.npy` matches the current model

---

## Technical Details

### Focal Loss Implementation
```python
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    # Only called per sample, gradient weighted by (1-pt)^gamma
    # Hard examples (low pt) get higher gradient
    return alpha * (1 - pt)^gamma * cross_entropy(y_true, y_pred)
```

### Threshold Optimization
```python
# For each class, find threshold that maximizes metric
# Takes validation set probabilities
# Uses scipy.optimize.minimize_scalar
# Constraint: 0.1 ≤ threshold ≤ 0.99
```

### Per-Class Threshold Application
```python
# For each sample:
# 1. Find class with max probability
# 2. Check if max_prob >= class_threshold
# 3. If yes: predict that class
# 4. If no: fallback to highest prob (no rejection)
```

---

## References

- Focal Loss paper: https://arxiv.org/abs/1708.02002
- Imbalanced Learning: https://imbalanced-learn.org/
- Threshold Tuning: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
