#!/usr/bin/env bash
# Run hres_fire_validate.sh from a snapshot copy under the data root (edits to the repo file cannot touch the run).
# NVFX_REPO tells the copy where the repo is; without it the tables land under $NEURALVFX_DATA/hres_3d/results.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export BUILD="${BUILD:-$REPO/build/nvfx_3d}"
export NVFX_REPO="$REPO"
B=$(BUILD="$BUILD" bash "$REPO/tools/study_3d/nvfx_bin_link.sh")
FIRE_ROOT=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/fire
mkdir -p "$FIRE_ROOT/logs"
SNAP="$FIRE_ROOT/logs/hres_fire_validate.snapshot.sh"
cp "$REPO/tools/study_3d/hres_fire_validate.sh" "$SNAP"
exec bash "$SNAP" "$B"
