#!/bin/bash
#SBATCH --job-name=style-fill
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

if [ -z "${CATEGORY:-}" ]; then
    echo "ERROR: CATEGORY env var not set"
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
echo "Category: $CATEGORY"
echo "Output dir: $OUT_DIR"
echo "Existing dirs: ${EXISTING_DIRS:-}"
echo "Validated dirs: ${VALIDATED_DIRS:-}"

apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$WORKDIR:$WORKDIR:rw" \
    --env HF_HOME="$HF_HOME" \
    --env TORCHDYNAMO_DISABLE=1 \
    --pwd "$PROJECT_DIR" \
    "$SIF_PATH" \
    python -m src.fuzzer.run_style_completion \
        --output-dir "$OUT_DIR" \
        --category "$CATEGORY" \
        --existing-dirs ${EXISTING_DIRS:-} \
        --validated-dirs ${VALIDATED_DIRS:-} \
        --target-styles fiction direct research technical_spec \
        --target-per-style ${TARGET_PER_STYLE:-2} \
        --oversample-factor ${OVERSAMPLE_FACTOR:-4} \
        --max-attempts-per-seed-style ${MAX_ATTEMPTS_PER_SEED_STYLE:-40} \
        --min-keep-score ${MIN_KEEP_SCORE:-0.55} \
        --population-size ${POPULATION_SIZE:-10} \
        --generations 1 \
        --mutation-rate ${MUTATION_RATE:-0.3} \
        --max-new-tokens 256

echo "=== Job $SLURM_JOB_ID finished ==="
