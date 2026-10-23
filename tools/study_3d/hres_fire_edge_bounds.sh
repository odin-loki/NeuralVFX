#!/usr/bin/env bash
# Build hres fire edge teacher runs and calibrated bounds (study R-style; nvfx_3d --sim --runs).
# Use before re-running edgecheck/choose with --bounds. Does not touch model edge CSVs.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
B=$(BUILD="${BUILD:-$REPO/build/nvfx_3d}" bash "$REPO/tools/study_3d/nvfx_bin_link.sh")
ROOT=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/fire
R=results/3d/hres/fire
H="--study hres --root $ROOT --effect fire"
W="--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30"
RUNS=${EDGE_RUNS:-4}
EDGE_B=${EDGE_B:-4}
EDGE_C=${EDGE_C:-2}
mkdir -p "$R"
echo "teacher runs=$RUNS -> $R/edge_teacher.csv"
nvfx() { "$B" "$@"; }
nvfx edge $H --sim --runs "$RUNS" $W --out "$R/edge_teacher.csv"
nvfx edgecal $H --csv "$R/edge_teacher.csv" --a 2 --b "$EDGE_B" --c "$EDGE_C" --out "$R/edge_bounds.csv" | tee "$R/edge_calibration.md"
echo "done: $R/edge_bounds.csv (use edgecheck --bounds and choose --bounds)"
