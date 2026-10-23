#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export BUILD="${BUILD:-$REPO/build/nvfx_3d}"
B=$(BUILD="$BUILD" bash "$REPO/tools/study_3d/nvfx_bin_link.sh")
FIRE_ROOT=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/fire
mkdir -p "$FIRE_ROOT/logs"
SNAP="$FIRE_ROOT/logs/hres_fire_validate.snapshot.sh"
cp "$REPO/tools/study_3d/hres_fire_validate.sh" "$SNAP"
exec bash "$SNAP" "$B"
