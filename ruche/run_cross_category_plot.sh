#!/bin/bash
#SBATCH --job-name=xai-plot
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=gpua100
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

set -euo pipefail

# Generate cross-category plots and report from shared output dir.
# Usage: sbatch --export=ALL,OUT_DIR=<shared_output_dir> ruche/run_cross_category_plot.sh

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

echo "=== Plot job $SLURM_JOB_ID on $(hostname) ==="
echo "Output dir: $OUT_DIR"
ls -la "$OUT_DIR" | head

apptainer exec \
    --writable-tmpfs \
    --bind "$WORKDIR:$WORKDIR:rw" \
    --pwd "$PROJECT_DIR" \
    "$SIF_PATH" \
    python -m scripts.plot_cross_category "$OUT_DIR" --top-k 5

echo "=== Figures: $OUT_DIR/figures/ ==="
