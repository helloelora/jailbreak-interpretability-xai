#!/bin/bash
#SBATCH --job-name=xai-cross-cat
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
RESULTS_BASE="$WORKDIR/jailbreak_xai_runs/results"
OUTPUT_DIR="$RESULTS_BASE/cross_category_${SLURM_JOB_ID}"

# --- HarmBench-validated run directories to include ---
# Each should contain seed_*.json with "jailbreaks" (HarmBench-validated).
# Edit this list if you want to include/exclude categories.
HB_DIRS=(
    "$RESULTS_BASE/cyber_467665/harmbench_validated"
    "$RESULTS_BASE/malware_467668/harmbench_validated"
    "$RESULTS_BASE/fuzz_505235/harmbench_validated"
)
# Add chembio dir here once that HarmBench run is available, e.g.:
#     "$RESULTS_BASE/chembio_XXXXX/harmbench_validated"

# --- Environment ---
module purge
module load apptainer/1.4.4/gcc-15.1.0

export HF_HOME="$WORKDIR/.cache/huggingface"
mkdir -p "$HF_HOME" "$OUTPUT_DIR"

echo "=== Job $SLURM_JOB_ID starting on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Output: $OUTPUT_DIR"
echo "HarmBench directories:"
for d in "${HB_DIRS[@]}"; do
    echo "  - $d"
done

# --- Run cross-category analysis (logit lens + activation patching + IG) ---
apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$WORKDIR:$WORKDIR:rw" \
    --env HF_HOME="$HF_HOME" \
    --env TORCHDYNAMO_DISABLE=1 \
    --pwd "$PROJECT_DIR" \
    "$SIF_PATH" \
    python -m scripts.run_cross_category \
        --harmbench-dirs "${HB_DIRS[@]}" \
        --output-dir "$OUTPUT_DIR" \
        --n-ig-steps 50

# --- Generate figures and report ---
apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$WORKDIR:$WORKDIR:rw" \
    --env HF_HOME="$HF_HOME" \
    --pwd "$PROJECT_DIR" \
    "$SIF_PATH" \
    python -m scripts.plot_cross_category "$OUTPUT_DIR" --top-k 5

echo ""
echo "=== Job $SLURM_JOB_ID finished ==="
echo "Figures: $OUTPUT_DIR/figures/"
