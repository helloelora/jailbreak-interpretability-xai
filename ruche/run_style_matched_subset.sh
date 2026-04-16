#!/bin/bash
#SBATCH --job-name=xai-run6
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

set -euo pipefail

if [ -z "${INPUT_DIR:-}" ]; then
    echo "ERROR: INPUT_DIR env var not set"
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

echo "=== Run 6 job $SLURM_JOB_ID on $(hostname) ==="
echo "Input dir: $INPUT_DIR"
echo "Output dir: $OUT_DIR"

apptainer exec \
    --writable-tmpfs \
    --bind "$WORKDIR:$WORKDIR:rw" \
    --pwd "$PROJECT_DIR" \
    "$SIF_PATH" \
    python -m scripts.run_style_matched_subset \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUT_DIR" \
        --styles research override

echo "=== Run 6 finished ==="
