#!/bin/bash
#SBATCH --job-name=dataset_gen              # Job name
#SBATCH --nodes=1                           # Number of nodes (do not change)
#SBATCH --partition=debug                   # Use your cluster's GPU-enabled partition if different
#SBATCH --nodelist=node5                    # Use one valid GPU node (change to node2..node7 if needed)
#SBATCH --ntasks=1                          # Number of tasks
#SBATCH --cpus-per-task=8                   # CPU cores for data + model loading
#SBATCH --mem=32G                           # RAM request
#SBATCH --gres=gpu:1                        # Request one GPU
#SBATCH --time=12:00:00                     # Maximum runtime (HH:MM:SS)
#SBATCH --output=out_%j.txt                 # Standard output
#SBATCH --error=err_%j.txt                  # Standard error

set -euo pipefail

cd /scratch/ashishp/edge-ai-baby-cry-detection

mkdir -p .tmp .pip-cache .audioldm-cache
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.pip-cache"
export AUDIOLDM_CACHE_DIR="$PWD/.audioldm-cache"
export HF_HOME="$PWD/.hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE"

# Optional GPU visibility check
echo "Running on host: $(hostname)"
date
nvidia-smi || true

./run_generate.sh
