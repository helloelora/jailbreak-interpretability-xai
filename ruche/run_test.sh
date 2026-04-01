#!/bin/bash
#SBATCH --job-name=jailbreak-test
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

# Quick smoke test: 1 seed, 5 generations, population 5
# Expected runtime: ~20-30 min (model loading ~5min + 25 evals x 15s)

set -euo pipefail

PROJECT_NAME="jailbreak-interpretability-slm"
SIF_NAME="jailbreak_xai.sif"
PROJECT_DIR="$WORKDIR/$PROJECT_NAME"
SIF_PATH="$WORKDIR/$SIF_NAME"
OUTPUT_DIR="$WORKDIR/jailbreak_xai_runs/results/test_${SLURM_JOB_ID}"

module purge
module load apptainer/1.4.4/gcc-15.1.0

# HuggingFace cache in WORKDIR to avoid HOME quota (50GB limit)
export HF_HOME="$WORKDIR/.cache/huggingface"
mkdir -p "$HF_HOME" "$OUTPUT_DIR"

echo "=== TEST JOB $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi

apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$WORKDIR:$WORKDIR:rw" \
    --env HF_HOME="$HF_HOME" \
    --env TORCHDYNAMO_DISABLE=1 \
    --pwd "$PROJECT_DIR" \
    "$SIF_PATH" \
    python -m src.fuzzer.run \
        --output-dir "$OUTPUT_DIR" \
        --category cybersecurity \
        --population-size 5 \
        --generations 5 \
        --mutation-rate 0.3 \
        --max-new-tokens 256

echo "=== TEST JOB $SLURM_JOB_ID finished ==="
