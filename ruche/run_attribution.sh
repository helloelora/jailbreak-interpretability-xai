#!/bin/bash
#SBATCH --job-name=ig-attribution
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=20:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

set -euo pipefail

# --- Configuration ---
PROJECT_NAME="jailbreak-interpretability-slm"
SIF_NAME="jailbreak_xai.sif"
PROJECT_DIR="$WORKDIR/$PROJECT_NAME"
SIF_PATH="$WORKDIR/$SIF_NAME"
RESULTS_BASE="$WORKDIR/jailbreak_xai_runs/results"
OUTPUT_DIR="$RESULTS_BASE/attributions_${SLURM_JOB_ID}"

# --- Environment ---
module purge
module load apptainer/1.4.4/gcc-15.1.0

export HF_HOME="$WORKDIR/.cache/huggingface"
mkdir -p "$HF_HOME" "$OUTPUT_DIR"

echo "=== Job $SLURM_JOB_ID starting on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Output: $OUTPUT_DIR"

# Run IG attribution on supplied directories, or auto-discover re-annotated runs.
# Pass either run dirs or harmbench_validated dirs:
#   sbatch ruche/run_attribution.sh "$WORKDIR/.../cyber_123"
#   sbatch ruche/run_attribution.sh "$WORKDIR/.../cyber_123/harmbench_validated"
if [ "$#" -gt 0 ]; then
    DIRS=()
    for arg in "$@"; do
        if [ -d "$arg/harmbench_validated" ]; then
            DIRS+=("$arg/harmbench_validated")
        else
            DIRS+=("$arg")
        fi
    done
else
    DIRS=()
    for pattern in cyber_* malware_* illegal_* chembio_* fuzz_*; do
        for dir in "$RESULTS_BASE"/$pattern/harmbench_validated; do
            if [ -d "$dir" ]; then
                DIRS+=("$dir")
            fi
        done
    done
fi

if [ "${#DIRS[@]}" -eq 0 ]; then
    echo "No harmbench_validated directories found under $RESULTS_BASE"
    exit 1
fi

echo "Input directories:"
for dir in "${DIRS[@]}"; do
    echo "  - $dir"
done

# float16 Mistral 24B (~48GB) uses device_map=auto to split GPU+CPU
for DIR in "${DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        echo ""
        echo "=== Attribution: $DIR ==="
        apptainer exec \
            --nv \
            --writable-tmpfs \
            --bind "$WORKDIR:$WORKDIR:rw" \
            --env HF_HOME="$HF_HOME" \
            --env TORCHDYNAMO_DISABLE=1 \
            --pwd "$PROJECT_DIR" \
            "$SIF_PATH" \
            python -m scripts.run_attribution "$DIR" \
                --output-dir "$OUTPUT_DIR" \
                --n-steps 50 \
                --max-pairs 3
    else
        echo "Skipping $DIR (not found)"
    fi
done

echo ""
echo "=== Job $SLURM_JOB_ID finished ==="
