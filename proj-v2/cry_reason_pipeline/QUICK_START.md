# Quick Reference: Running Imbalance Improvements

## One-Liner Commands

### Option 1: Extended Training (Simplest)
```bash
python 03_train_model.py --balanced-sampling --augment-minority
```
- Uses improved defaults (100 epochs, patience 20)
- No extra configs needed

### Option 2: With Focal Loss (For Severe Imbalance)
```bash
python 03_train_model.py --use-focal-loss --balanced-sampling --augment-minority
```
- Better for highly imbalanced data
- Automatically scales training appropriately

### Option 3: Extended Custom Settings
```bash
python 03_train_model.py \
  --balanced-sampling \
  --augment-minority \
  --epochs 150 \
  --early-stop-patience 30
```
- Full control over training duration

---

## Threshold Optimization

After training completes, optimize decision boundaries:

```bash
# Best for imbalanced data (macro F1)
python 08_threshold_optimization.py --metric f1-macro

# Alternative: balanced accuracy
python 08_threshold_optimization.py --metric balanced-accuracy
```

Creates: `artifacts/model/threshold_config.json`

---

## Inference with Optimized Thresholds

```bash
python 09_inference_with_thresholds.py
```

Shows:
- Accuracy on test set
- Per-class precision/recall
- Confusion matrix

---

## Full Pipeline (Complete Solution)

```bash
# 1. Train with improvements
python 03_train_model.py \
  --use-focal-loss \
  --balanced-sampling \
  --augment-minority \
  --epochs 100

# 2. Optimize thresholds (after training completes)
python 08_threshold_optimization.py --metric f1-macro

# 3. Evaluate
python 09_inference_with_thresholds.py
```

---

## Key Hyperparameters

| Parameter | Default | Recommended | Notes |
|-----------|---------|-------------|-------|
| `--epochs` | 100 | 100-150 | More epochs = better minority convergence |
| `--early-stop-patience` | 20 | 20-30 | Higher patience = more training time |
| `--use-focal-loss` | False | True | Use for severe imbalance (>10x) |
| `--balanced-sampling` | False | True | Always use for imbalance |
| `--augment-minority` | False | True | Increases minority samples |

---

## Expected Results

### Training Metrics In Console

```
Epoch 45/100
...
val_accuracy: 0.8234

Epoch 100/100 (best model restored)
Test accuracy: 0.8145
```

### After Threshold Optimization

```
Loaded threshold configuration from: artifacts/model/threshold_config.json
  Type: per-class
  Metric: f1-macro

EVALUATION ON TEST SET
Accuracy: 0.8145

Classification Report:
              precision    recall  f1-score   support
    class_0       0.85      0.82      0.84       100
    class_1       0.70      0.68      0.69        50    ← Minority now performs better
    class_2       0.82      0.84      0.83        95
    class_3       0.81      0.79      0.80        80

  macro avg       0.80      0.78      0.79       325    ← Balanced across classes
weighted avg       0.80      0.81      0.81       325
```

---

## Monitoring & Debugging

### Check if training is working:
- Validation accuracy increasing? ✓ Good convergence
- Training loss decreasing? ✓ Model learning
- Per-class metrics improving? ✓ Balanced learning

### If minority class F1 is still low:
1. Increase `--epochs` to 150+
2. Add `--use-focal-loss`
3. Try `--metric f1-macro` for threshold optimization
4. Check if minority samples have good augmentation

### If overfitting:
1. Make sure `--augment-minority` is ON
2. Reduce `--epochs`
3. Increase `--early-stop-patience` slightly

---

## Output Files Location

After complete pipeline, check `artifacts/model/`:

```
artifacts/model/
├── cry_reason_model.keras          # Main model
├── training_info.json              # Training summary
├── metadata.json                   # Classes & mapping
├── val_predictions.npy             # Used for threshold optimization
├── test_predictions.npy            # Test set predictions
├── per_class_metrics.csv           # Per-class F1, precision, recall
└── threshold_config.json           # Optimized thresholds ← Load this in inference
```

---

## Tips & Tricks

1. **For severe imbalance** (>20 minority samples with 100+ majority):
   - Use `--use-focal-loss` + `--epochs 150`
  - Run `08_threshold_optimization.py --metric f1-macro`

2. **For mild imbalance** (>50 minority samples):
   - Start with extended training only
   - Add threshold tuning if F1 still low

3. **For production**:
   - Save `threshold_config.json` with model
   - Load and apply in inference pipeline
   - Monitor minority class predictions separately

4. **Iterate quickly**:
   ```bash
   # Test pipeline
   python 03_train_model.py --epochs 20 --early-stop-patience 5
  python 08_threshold_optimization.py --metric f1-macro
   python 09_inference_with_thresholds.py
   ```

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| "No thresholds found" | Threshold script folder doesn't exist | Create `artifacts/model/` before running |
| Minority F1 still 0.4 | Model not training enough | Use `--epochs 150 --use-focal-loss` |
| Training very slow | Too high augmentation | Reduce minority augmentation settings |
| Unstable thresholds | Noisy validation set | Ensure good train/val split |
| Overfitting after epoch 20 | Model has too much capacity | Reduce `--epochs` or check data for leakage |

