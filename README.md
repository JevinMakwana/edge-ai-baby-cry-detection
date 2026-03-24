# Edge AI Baby Cry Detection

## Project Structure

```
edge-ai-baby-cry-detection/
├── 1_dataset_generation/
│   └── audio_samples/
│       ├── generate.py          # AudioLDM dataset generation script
│       ├── generate.ipynb       # Notebook version for testing
│       ├── generation_log.jsonl # Per-file generation log (seed, RMS, silent flag)
│       └── prompts/
│           ├── baby_cry_prompts.txt
│           └── background_noise_prompts.txt
├── 2_model_training/
├── 3_openmv_firmware/
├── 4_integration_demo/
├── scripts/
│   ├── check_generated_wavs.py  # Validates WAV files (count, sample rate, RMS)
│   └── archive_slurm_job.sh     # Archives SLURM logs and outputs
├── run_generate.sh              # Shell wrapper for generate.py
├── run_generate.slurm           # SLURM job submission script
└── requirements.txt
```

---

## Member 1 — Dataset Generation

### Cluster Workflow (SLURM, GPU node)

#### 1) One-time setup

```bash
cd /scratch/ashishp/edge-ai-baby-cry-detection
.venv/bin/python -m pip install -r requirements.txt
chmod +x run_generate.sh scripts/archive_slurm_job.sh
```

#### 2) One-time prefetch on internet-enabled session

This downloads required Hugging Face + AudioLDM assets into project-local cache.

```bash
cd /scratch/ashishp/edge-ai-baby-cry-detection
rm -rf .hf-cache/transformers/models--roberta-base

HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
HF_HOME="$PWD/.hf-cache" \
TRANSFORMERS_CACHE="$PWD/.hf-cache/transformers" \
HUGGINGFACE_HUB_CACHE="$PWD/.hf-cache/hub" \
.venv/bin/python -c "
from transformers import RobertaTokenizer, RobertaModel
RobertaTokenizer.from_pretrained('roberta-base', force_download=True)
RobertaModel.from_pretrained('roberta-base', force_download=True)
print('roberta prefetch ok')
"

AUDIOLDM_CACHE_DIR="$PWD/.audioldm-cache" \
.venv/bin/python -c "
from audioldm.utils import download_checkpoint
download_checkpoint('audioldm-s-full')
print('audioldm checkpoint prefetch ok')
"
```

#### 3) Submit generation job

```bash
cd /scratch/ashishp/edge-ai-baby-cry-detection
sbatch run_generate.slurm
```

#### 4) Monitor job

```bash
squeue -u $USER
tail -f out_<jobid>.txt
tail -f err_<jobid>.txt
```

#### 5) Validate generated WAV files

```bash
cd /scratch/ashishp/edge-ai-baby-cry-detection
.venv/bin/python scripts/check_generated_wavs.py --require-nonempty
```

Optional — override class folders or expected sample rate:

```bash
.venv/bin/python scripts/check_generated_wavs.py \
  --root 1_dataset_generation/audio_samples \
  --classes baby_cry background_noise \
  --sample-rate 16000
```

#### 6) Archive outputs and logs per SLURM job

Archive logs and WAV outputs:

```bash
./scripts/archive_slurm_job.sh <jobid>
```

Archive only logs (no audio files):

```bash
./scripts/archive_slurm_job.sh <jobid> --no-audio
```

Archive then remove original logs and WAV files:

```bash
./scripts/archive_slurm_job.sh <jobid> --delete-originals
```

Archives are saved under:

```
archives/job_<jobid>_<timestamp>.tar.gz
```

#### 7) To regenerate from scratch

```bash
cd /scratch/ashishp/edge-ai-baby-cry-detection
sbatch run_generate.slurm
# or run directly:
.venv/bin/python 1_dataset_generation/audio_samples/generate.py \
  --samples-per-prompt 3 \
  --base-seed 42
```

Generation is **resumable** — already-generated files are automatically skipped.

---

## Dataset

| Class              | Files | Format        | Duration | Sample Rate |
|--------------------|-------|---------------|----------|-------------|
| `baby_cry`         | 570   | WAV, PCM-16   | 5s each  | 16000 Hz    |
| `background_noise` | 471   | WAV, PCM-16   | 5s each  | 16000 Hz    |
| **Total**          | **1041** |            |          |             |

- Generated using [AudioLDM](https://github.com/haoheliu/AudioLDM) (`audioldm-s-full`)
- 190 baby cry prompts × 3 seeds + 157 background noise prompts × 3 seeds
- WAV files are **not committed to git** — download from Google Drive below

**Google Drive:** `https://drive.google.com/drive/folders/1he9v7fSK9GHA7CfeW1HjqMbHArcD-Qi1?usp=sharing`

> For Member 2: download and extract both `.tar.gz` files, then upload the
> `baby_cry/` and `background_noise/` folders directly into Edge Impulse
> as two separate classes. All files are already 16kHz mono — no conversion needed.

---

## Notes

- Silent or near-silent files (RMS < 0.005) are flagged in `generation_log.jsonl`
- To list all flagged silent files:
  ```bash
  python -c "
  import json
  flagged = [e['file'] for e in (json.loads(l) for l in open('generation_log.jsonl')) if e['silent']]
  print('\n'.join(flagged))
  "
  ```
- Manually listen to ~20% of files as a quality check before handing off to Member 2
- Do **not** commit `.audioldm-cache/`, `.hf-cache/`, `.venv/`, or `*.wav` files to git