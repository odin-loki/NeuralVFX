#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
if [ -f "$REPO/tools/wsl_env.sh" ]; then
  # shellcheck source=/dev/null
  . "$REPO/tools/wsl_env.sh"
fi
cd "$REPO"
export NVFX_REPO="$REPO"
B=$(BUILD="${BUILD:-${BUILD_DIR:-$REPO/build-cuda}/nvfx_3d}" bash "$REPO/tools/study_3d/nvfx_bin_link.sh")
ROOT=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/smoke_r2
mkdir -p "$ROOT/logs"
SNAP="$ROOT/logs/hres_smoke_r2_validate.snapshot.sh"
cp "$REPO/tools/study_3d/hres_smoke_r2_validate.sh" "$SNAP"
exec bash "$SNAP" "$B" "$ROOT"
