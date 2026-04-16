#!/bin/bash
# Build an expanded style-matched subset, then run cross-category XAI in parallel.

set -euo pipefail

PROJECT_NAME="jailbreak-interpretability-slm"
SIF_NAME="jailbreak_xai.sif"
PROJECT_DIR="$WORKDIR/$PROJECT_NAME"
SIF_PATH="$WORKDIR/$SIF_NAME"
RESULTS_BASE="$WORKDIR/jailbreak_xai_runs/results"

OUT_DIR="$RESULTS_BASE/run6_expanded_$(date +%Y%m%d_%H%M%S)"
SUBSET_DIR="$OUT_DIR/subsets"
mkdir -p "$SUBSET_DIR"

SOURCE_DIRS=(
    "$RESULTS_BASE/cyber_467665/harmbench_validated"
    "$RESULTS_BASE/malware_467668/harmbench_validated"
    "$RESULTS_BASE/fuzz_505235/harmbench_validated"
    "$RESULTS_BASE/stylefill_cybersecurity_20260416_181629"
    "$RESULTS_BASE/stylefill_illegal_20260416_181630"
    "$RESULTS_BASE/stylefill_malware_20260416_181630"
)

module purge
module load apptainer/1.4.4/gcc-15.1.0

echo "=== Building expanded subset into $SUBSET_DIR ==="
apptainer exec \
    --writable-tmpfs \
    --bind "$WORKDIR:$WORKDIR:rw" \
    --pwd "$PROJECT_DIR" \
    "$SIF_PATH" \
    python -m scripts.build_style_matched_validated_subset \
        --source-dirs "${SOURCE_DIRS[@]}" \
        --output-base "$SUBSET_DIR" \
        --styles research override technical_spec \
        --max-per-style 2

echo ""
echo "=== Submitting expanded cross-category jobs ==="

JOB_IDS=()
for CATEGORY in cybersecurity illegal malware; do
    HB_DIR="$SUBSET_DIR/$CATEGORY"
    if [ ! -d "$HB_DIR" ]; then
        echo "SKIP: $HB_DIR (not found)"
        continue
    fi
    jid=$(sbatch --parsable \
        --job-name="xai-$CATEGORY-exp" \
        --export=ALL,HB_DIR="$HB_DIR",OUT_DIR="$OUT_DIR" \
        ruche/run_cross_category_one.sh)
    echo "  $CATEGORY -> $jid"
    JOB_IDS+=("$jid")
done

if [ "${#JOB_IDS[@]}" -eq 0 ]; then
    echo "No analysis jobs submitted."
    exit 1
fi

DEP=$(IFS=:; echo "${JOB_IDS[*]}")
plot_jid=$(sbatch --parsable \
    --dependency="afterany:$DEP" \
    --export=ALL,OUT_DIR="$OUT_DIR" \
    ruche/run_cross_category_plot.sh)

echo ""
echo "Output dir: $OUT_DIR"
echo "Analysis jobs: ${JOB_IDS[*]}"
echo "Plot job: $plot_jid"
echo "Monitor with: squeue -u \$USER"
