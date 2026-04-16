#!/bin/bash
#SBATCH --job-name=xai-style
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

set -euo pipefail

if [ -z "${OUT_DIR:-}" ]; then
    echo "ERROR: OUT_DIR env var not set"
    exit 1
fi

PROJECT_NAME="jailbreak-interpretability-slm"
SIF_NAME="jailbreak_xai.sif"
PROJECT_DIR="$WORKDIR/$PROJECT_NAME"
SIF_PATH="$WORKDIR/$SIF_NAME"

if [ -z "${ANALYSIS_DIR:-}" ]; then
    ANALYSIS_DIR="$WORKDIR/jailbreak_xai_runs/results/cross_category_20260416_124911"
fi

if [ -z "${HB_DIRS:-}" ]; then
    HB_DIRS="$WORKDIR/jailbreak_xai_runs/results/cyber_467665/harmbench_validated \
$WORKDIR/jailbreak_xai_runs/results/malware_467668/harmbench_validated \
$WORKDIR/jailbreak_xai_runs/results/fuzz_505235/harmbench_validated"
fi

module purge
module load apptainer/1.4.4/gcc-15.1.0

export HF_HOME="$WORKDIR/.cache/huggingface"
mkdir -p "$HF_HOME" "$OUT_DIR"

echo "=== Job $SLURM_JOB_ID starting on $(hostname) ==="
echo "Analysis dir: $ANALYSIS_DIR"
echo "HarmBench dirs: $HB_DIRS"
echo "Output dir: $OUT_DIR"

apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$WORKDIR:$WORKDIR:rw" \
    --env HF_HOME="$HF_HOME" \
    --pwd "$PROJECT_DIR" \
    "$SIF_PATH" \
    python -m scripts.analyze_jailbreak_styles \
        --analysis-dir "$ANALYSIS_DIR" \
        --harmbench-dirs $HB_DIRS \
        --output-dir "$OUT_DIR"

echo "=== Job $SLURM_JOB_ID finished ==="
