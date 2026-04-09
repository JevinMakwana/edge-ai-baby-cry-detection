#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p .tmp .pip-cache
export TMPDIR="$ROOT_DIR/.tmp"
export PIP_CACHE_DIR="$ROOT_DIR/.pip-cache"
export AUDIOLDM_CACHE_DIR="$ROOT_DIR/.audioldm-cache"
export HF_HOME="$ROOT_DIR/.hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$AUDIOLDM_CACHE_DIR" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/1_dataset_generation/audio_samples/generate.py" \
    --samples-per-prompt 3 \
    --base-seed 42 \
    --rms-threshold 0.005