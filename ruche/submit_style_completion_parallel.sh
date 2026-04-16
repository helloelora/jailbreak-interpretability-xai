#!/bin/bash
# Submit 3 category-specific style-completion jobs in parallel, then
# schedule HarmBench reannotation for each result directory.

set -euo pipefail

RESULTS_BASE="$WORKDIR/jailbreak_xai_runs/results"
TARGET_PER_STYLE="${TARGET_PER_STYLE:-2}"
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-4}"
MAX_ATTEMPTS_PER_SEED_STYLE="${MAX_ATTEMPTS_PER_SEED_STYLE:-40}"
MIN_KEEP_SCORE="${MIN_KEEP_SCORE:-0.55}"

declare -A RAW_DIRS=(
    [cybersecurity]="$RESULTS_BASE/cyber_467665"
    [malware]="$RESULTS_BASE/malware_467668"
    [illegal]="$RESULTS_BASE/fuzz_505235"
)

declare -A VALIDATED_DIRS=(
    [cybersecurity]="$RESULTS_BASE/cyber_467665/harmbench_validated"
    [malware]="$RESULTS_BASE/malware_467668/harmbench_validated"
    [illegal]="$RESULTS_BASE/fuzz_505235/harmbench_validated"
)

CATEGORIES=(cybersecurity malware illegal)

echo "Submitting targeted style-completion jobs"
echo "Target styles: fiction direct research technical_spec"
echo "Target per style: $TARGET_PER_STYLE"
echo ""

for CATEGORY in "${CATEGORIES[@]}"; do
    TS=$(date +%Y%m%d_%H%M%S)
    OUT_DIR="$RESULTS_BASE/stylefill_${CATEGORY}_${TS}"
    mkdir -p "$OUT_DIR"

    echo "Submitting $CATEGORY -> $OUT_DIR"
    fuzz_jid=$(sbatch --parsable \
        --job-name="style-$CATEGORY" \
        --export=ALL,\
CATEGORY="$CATEGORY",\
OUT_DIR="$OUT_DIR",\
EXISTING_DIRS="${RAW_DIRS[$CATEGORY]}",\
VALIDATED_DIRS="${VALIDATED_DIRS[$CATEGORY]}",\
TARGET_PER_STYLE="$TARGET_PER_STYLE",\
OVERSAMPLE_FACTOR="$OVERSAMPLE_FACTOR",\
MAX_ATTEMPTS_PER_SEED_STYLE="$MAX_ATTEMPTS_PER_SEED_STYLE",\
MIN_KEEP_SCORE="$MIN_KEEP_SCORE" \
        ruche/run_style_completion_one.sh)

    echo "  -> fuzz job $fuzz_jid"

    reann_jid=$(sbatch --parsable \
        --dependency="afterok:$fuzz_jid" \
        ruche/run_reannotate.sh "$OUT_DIR")

    echo "  -> reannotate job $reann_jid"
    echo ""
done

echo "Monitor with: squeue -u \$USER"
