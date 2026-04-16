#!/bin/bash
# Submit parallel cross-category XAI analysis jobs, then a dependent plot job.
#
# Usage (from the project root on La Ruche):
#   bash ruche/submit_cross_category_parallel.sh
#
# Each category runs on its own A100 simultaneously. The plot job is
# scheduled with --dependency=afterany:... so it runs once all analysis
# jobs complete (even if one fails, so you still get partial results).

set -euo pipefail

RESULTS_BASE="$WORKDIR/jailbreak_xai_runs/results"
OUT_DIR="$RESULTS_BASE/cross_category_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

echo "Shared output directory: $OUT_DIR"
echo ""

# List HarmBench-validated dirs to analyze in parallel.
# Edit this list to add/remove categories.
HB_DIRS=(
    "$RESULTS_BASE/cyber_467665/harmbench_validated"
    "$RESULTS_BASE/malware_467668/harmbench_validated"
    "$RESULTS_BASE/fuzz_505235/harmbench_validated"
)
# Add chembio here once HarmBench-validated, e.g.:
#     "$RESULTS_BASE/chembio_XXXXX/harmbench_validated"

JOB_IDS=()

for HB_DIR in "${HB_DIRS[@]}"; do
    if [ ! -d "$HB_DIR" ]; then
        echo "SKIP: $HB_DIR (not found)"
        continue
    fi
    category=$(basename "$(dirname "$HB_DIR")")
    echo "Submitting analysis for $category ($HB_DIR)..."
    jid=$(sbatch --parsable \
        --job-name="xai-$category" \
        --export=ALL,HB_DIR="$HB_DIR",OUT_DIR="$OUT_DIR" \
        ruche/run_cross_category_one.sh)
    echo "  -> job $jid"
    JOB_IDS+=("$jid")
done

if [ "${#JOB_IDS[@]}" -eq 0 ]; then
    echo "No analysis jobs submitted. Aborting."
    exit 1
fi

DEP=$(IFS=:; echo "${JOB_IDS[*]}")
echo ""
echo "Submitting plot job with dependency afterany:$DEP..."

plot_jid=$(sbatch --parsable \
    --dependency="afterany:$DEP" \
    --export=ALL,OUT_DIR="$OUT_DIR" \
    ruche/run_cross_category_plot.sh)

echo "  -> plot job $plot_jid (waits for analysis jobs to finish)"
echo ""
echo "Submitted jobs:"
echo "  Analysis: ${JOB_IDS[*]}"
echo "  Plot:     $plot_jid"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Output dir:   $OUT_DIR"
echo "Figures will land in: $OUT_DIR/figures/"
