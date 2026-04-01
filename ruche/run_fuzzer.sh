#!/bin/bash
#SBATCH --job-name=jailbreak-fuzz
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=23:59:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

set -euo pipefail

# --- Configuration ---
PROJECT_NAME="jailbreak-interpretability-slm"
SIF_NAME="jailbreak_xai.sif"
PROJECT_DIR="$WORKDIR/$PROJECT_NAME"
SIF_PATH="$WORKDIR/$SIF_NAME"
OUTPUT_DIR="$WORKDIR/jailbreak_xai_runs/results/fuzz_${SLURM_JOB_ID}"

# --- Environment ---
module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p "$OUTPUT_DIR"

echo "=== Job $SLURM_JOB_ID starting on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Output: $OUTPUT_DIR"

# --- Run fuzzer ---
apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$WORKDIR:$WORKDIR:rw" \
    --pwd "$PROJECT_DIR" \
    "$SIF_PATH" \
    python -m src.fuzzer.run \
        --output-dir "$OUTPUT_DIR" \
        --population-size 20 \
        --generations 50 \
        --mutation-rate 0.3 \
        --max-new-tokens 256 \
        --device cuda

echo "=== Job $SLURM_JOB_ID finished ==="
