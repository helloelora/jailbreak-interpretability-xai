#!/bin/bash
# One-time setup for La Ruche: build the Apptainer container
# Run this on a login node after cloning the repo.

set -euo pipefail

PROJECT_NAME="jailbreak-interpretability-slm"
SIF_NAME="jailbreak_xai.sif"

echo "=== Setting up $PROJECT_NAME on La Ruche ==="

# Load Apptainer
module purge
module load apptainer/1.4.4/gcc-15.1.0

# Use WORKDIR to avoid HOME quota issues
export APPTAINER_CACHEDIR="$WORKDIR/.apptainer_cache"
export APPTAINER_TMPDIR="$WORKDIR/apptainer_tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# Build the container image
SIF_PATH="$WORKDIR/$SIF_NAME"
DEF_PATH="$WORKDIR/$PROJECT_NAME/ruche/jailbreak_xai.def"

echo "Building container from $DEF_PATH..."
apptainer build "$SIF_PATH" "$DEF_PATH"

# Create output directories
mkdir -p "$WORKDIR/jailbreak_xai_runs/results"
mkdir -p "$WORKDIR/jailbreak_xai_runs/data"

echo ""
echo "=== Setup complete ==="
echo "Container: $SIF_PATH"
echo "Results:   $WORKDIR/jailbreak_xai_runs/"
echo ""
echo "Next: sbatch ruche/run_fuzzer.sh"
