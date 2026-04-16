#!/bin/bash
#SBATCH --job-name=xai-defense
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

set -euo pipefail

if [ -z "${ANALYSIS_DIR:-}" ]; then
    echo "ERROR: ANALYSIS_DIR env var not set"
    exit 1
fi

PROJECT_NAME="jailbreak-interpretability-slm"
SIF_NAME="jailbreak_xai.sif"
PROJECT_DIR="$WORKDIR/$PROJECT_NAME"
SIF_PATH="$WORKDIR/$SIF_NAME"

if [ -z "${OUT_DIR:-}" ]; then
    OUT_DIR="$ANALYSIS_DIR/internal_defense"
fi

module purge
module load apptainer/1.4.4/gcc-15.1.0

export HF_HOME="$WORKDIR/.cache/huggingface"
mkdir -p "$HF_HOME" "$OUT_DIR"

echo "=== Job $SLURM_JOB_ID starting on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Input analyses: $ANALYSIS_DIR"
echo "Output dir: $OUT_DIR"
echo "Extra args: ${EXTRA_ARGS:-<none>}"

apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$WORKDIR:$WORKDIR:rw" \
    --env HF_HOME="$HF_HOME" \
    --env TORCHDYNAMO_DISABLE=1 \
    --pwd "$PROJECT_DIR" \
    "$SIF_PATH" \
    python -m scripts.run_internal_defense \
        --input-dir "$ANALYSIS_DIR" \
        --output-dir "$OUT_DIR" \
        --top-k-values 1 3 5 \
        --random-trials 5 \
        ${EXTRA_ARGS:-}

apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$WORKDIR:$WORKDIR:rw" \
    --env HF_HOME="$HF_HOME" \
    --env TORCHDYNAMO_DISABLE=1 \
    --pwd "$PROJECT_DIR" \
    "$SIF_PATH" \
    python -m scripts.plot_internal_defense \
        "$OUT_DIR" \
        --focus-k 5 \
        --output-dir "$OUT_DIR/figures"

echo "=== Job $SLURM_JOB_ID finished ==="
