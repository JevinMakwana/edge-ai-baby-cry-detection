# Baby Cry Reason Pipeline (proj-v2)

This pipeline is designed for classifying cry reason from the cleaned Donate-a-Cry WAV dataset and exporting a deployable TFLite model.

## Should you use notebooks or Python files?

Use `.py` files as the primary workflow for this project.

Why:
- You need repeatable training and export runs for deployment.
- Script-based steps are easier to automate and version-control.
- Deployment preparation for Nicla boards is cleaner with scripts.

Recommended approach:
- Use `.py` scripts for the full pipeline (already provided here).
- Optionally use one small notebook only for quick EDA and visualization.

## Dataset facts (from the cleaned dataset)

- Total WAV files: 457
- Classes: `belly_pain`, `burping`, `discomfort`, `hungry`, `tired`
- Class counts:
  - belly_pain: 16
  - burping: 8
  - discomfort: 27
  - hungry: 382
  - tired: 24
- Audio format: 8 kHz, mono, about 7 seconds per clip

Important note:
- The dataset is heavily imbalanced, so training uses class weights.
- You can now enable balanced sampling and class-aware augmentation for minority classes.

## Setup

```powershell
cd d:/IISc/sem2/EdgeAI/Project/proj-v2/cry_reason_pipeline
d:/IISc/sem2/EdgeAI/Project/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Step-by-step code flow

### 1) Audit and metadata export

```powershell
d:/IISc/sem2/EdgeAI/Project/.venv/Scripts/python.exe 01_dataset_audit.py
```

Outputs:
- `artifacts/metadata.csv`
- `artifacts/dataset_summary.json`

### 2) Create stratified train/val/test splits

```powershell
d:/IISc/sem2/EdgeAI/Project/.venv/Scripts/python.exe 02_prepare_splits.py
```

Outputs:
- `artifacts/splits/train.csv`
- `artifacts/splits/val.csv`
- `artifacts/splits/test.csv`

### 3) Train the classifier

```powershell
d:/IISc/sem2/EdgeAI/Project/.venv/Scripts/python.exe 03_train_model.py --epochs 100 --batch-size 16 --early-stop-patience 20 --lr-plateau-patience 4 --balanced-sampling --augment-minority
```

Outputs:
- `artifacts/model/cry_reason_model.keras`
- `artifacts/model/training_info.json`
- `artifacts/model/per_class_metrics.csv`
- `artifacts/model/x_train_features.npy` (used for quantization)

Notes:
- The training script keeps only the best model (by validation accuracy) as `cry_reason_model.keras`.
- Per-class precision/recall/F1 is written to `per_class_metrics.csv`.
- For imbalance handling, `--balanced-sampling` oversamples classes to equal counts.
- `--augment-minority` applies stronger augmentations to underrepresented classes.
- You can enable `--use-focal-loss` and then run `08_threshold_optimization.py --metric f1-macro` to tune per-class decision thresholds.

### 4) Export to TFLite

Dynamic range quantization:

```powershell
d:/IISc/sem2/EdgeAI/Project/.venv/Scripts/python.exe 04_export_tflite.py
```

Full int8 quantization (recommended for MCU deployment):

```powershell
d:/IISc/sem2/EdgeAI/Project/.venv/Scripts/python.exe 04_export_tflite.py --quantize
```

Output:
- `artifacts/model/cry_reason_model.tflite`

### 5) Test inference on one clip

Keras model inference:

```powershell
d:/IISc/sem2/EdgeAI/Project/.venv/Scripts/python.exe 05_infer_clip.py --clip ../donateacry-corpus-master/donateacry_corpus_cleaned_and_updated_data/hungry/004-f-48-hu.wav
```

TFLite model inference:

```powershell
d:/IISc/sem2/EdgeAI/Project/.venv/Scripts/python.exe 05_infer_clip.py --use-tflite --clip ../donateacry-corpus-master/donateacry_corpus_cleaned_and_updated_data/hungry/004-f-48-hu.wav
```

## Deployment direction: Nicla Vision vs Nicla Voice

- If your board has native audio-focused support and enough RAM for features + model, prefer Nicla Voice for direct audio workflow.
- If using Nicla Vision, audio can still be done but usually needs a more custom pipeline and tighter optimization.

Model-side recommendations for either board:
- Keep input features small (`n_mels=40`, short clip or frame stack).
- Use int8 quantization.
- Start with a tiny CNN and prune complexity if latency or RAM is too high.

## Practical next improvement

Given class imbalance, after baseline training:
- add weighted sampling or class-aware augmentation,
- or merge labels to a smaller taxonomy if your deployment goal is robust real-time behavior over fine-grained classes.

## One-command pipeline runner

Use `06_run_pipeline.py` to run all steps in sequence with tunable hyperparameters.

Example:

```powershell
d:/IISc/sem2/EdgeAI/Project/.venv/Scripts/python.exe 06_run_pipeline.py --epochs 100 --batch-size 16 --n-mels 40 --n-fft 512 --hop-length 160 --balanced-sampling --augment-minority --optimize-thresholds --quantize
```

This runs:
1. dataset audit,
2. split creation,
3. model training,
4. TFLite export,
5. optional threshold optimization,
6. optional single-clip smoke inference.

You can re-run with different hyperparameters without changing code.

Useful augmentation hyperparameters:
- `--aug-noise-std` (default: 0.002)
- `--aug-time-shift-max` in seconds (default: 0.02)
- `--aug-gain-min`, `--aug-gain-max` (default: 0.85 to 1.15)
- `--aug-time-mask-max` and `--aug-freq-mask-max` (default: 6, 6)
