#!/bin/bash
#SBATCH --job-name=harmbench-reannotate
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

# --- Configuration ---
PROJECT_NAME="jailbreak-interpretability-slm"
SIF_NAME="jailbreak_xai.sif"
PROJECT_DIR="$WORKDIR/$PROJECT_NAME"
SIF_PATH="$WORKDIR/$SIF_NAME"
RESULTS_BASE="$WORKDIR/jailbreak_xai_runs/results"

# --- Directories to re-annotate ---
# Pass directories as positional args to override auto-discovery:
#   sbatch ruche/run_reannotate_all.sh "$WORKDIR/.../cyber_123" "$WORKDIR/.../malware_456"
if [ "$#" -gt 0 ]; then
    DIRS=("$@")
else
    DIRS=()
    for pattern in cyber_* malware_* illegal_* chembio_* fuzz_*; do
        for dir in "$RESULTS_BASE"/$pattern; do
            if [ -d "$dir" ]; then
                DIRS+=("$dir")
            fi
        done
    done
fi

# --- Environment ---
module purge
module load apptainer/1.4.4/gcc-15.1.0

export HF_HOME="$WORKDIR/.cache/huggingface"
mkdir -p "$HF_HOME"

echo "=== Job $SLURM_JOB_ID starting on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

if [ "${#DIRS[@]}" -eq 0 ]; then
    echo "No run directories found under $RESULTS_BASE"
    exit 1
fi

echo "Run directories to re-annotate:"
for dir in "${DIRS[@]}"; do
    echo "  - $dir"
done

for INPUT_DIR in "${DIRS[@]}"; do
    echo ""
    echo "=== Re-annotating: $INPUT_DIR ==="
    apptainer exec \
        --nv \
        --writable-tmpfs \
        --bind "$WORKDIR:$WORKDIR:rw" \
        --env HF_HOME="$HF_HOME" \
        --env TORCHDYNAMO_DISABLE=1 \
        --pwd "$PROJECT_DIR" \
        "$SIF_PATH" \
        python -m scripts.reannotate_harmbench "$INPUT_DIR"
done

echo ""
echo "=== Job $SLURM_JOB_ID finished ==="
