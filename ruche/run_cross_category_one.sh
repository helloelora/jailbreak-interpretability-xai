#!/bin/bash
#SBATCH --job-name=xai-one
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

set -euo pipefail

# Run cross-category XAI analysis on ONE HarmBench-validated directory.
# Designed for parallel submission: each category runs on its own A100.
#
# Usage:
#   sbatch --export=ALL,HB_DIR=<harmbench_validated_dir>,OUT_DIR=<shared_output_dir> \
#          ruche/run_cross_category_one.sh
#
# All parallel jobs should share the SAME OUT_DIR so the plot job sees
# every analysis_*.json file.

if [ -z "${HB_DIR:-}" ]; then
    echo "ERROR: HB_DIR env var not set"
    exit 1
fi
if [ -z "${OUT_DIR:-}" ]; then
    echo "ERROR: OUT_DIR env var not set"
    exit 1
fi

PROJECT_NAME="jailbreak-interpretability-slm"
SIF_NAME="jailbreak_xai.sif"
PROJECT_DIR="$WORKDIR/$PROJECT_NAME"
SIF_PATH="$WORKDIR/$SIF_NAME"

module purge
module load apptainer/1.4.4/gcc-15.1.0

export HF_HOME="$WORKDIR/.cache/huggingface"
mkdir -p "$HF_HOME" "$OUT_DIR"

echo "=== Job $SLURM_JOB_ID starting on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Input: $HB_DIR"
echo "Shared output: $OUT_DIR"

apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$WORKDIR:$WORKDIR:rw" \
    --env HF_HOME="$HF_HOME" \
    --env TORCHDYNAMO_DISABLE=1 \
    --pwd "$PROJECT_DIR" \
    "$SIF_PATH" \
    python -m scripts.run_cross_category \
        --harmbench-dirs "$HB_DIR" \
        --output-dir "$OUT_DIR" \
        --n-ig-steps 50

echo "=== Job $SLURM_JOB_ID finished ==="
