#!/usr/bin/env bash
# Run study hres end-to-end: fire (record, train, finish, validate) then smoke. One train job at a time.
# Usage: hres_run_all.sh [NVFX_3D_BINARY] [NEURALVFX_DATA]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
if [ -n "${1:-}" ] && [ -x "$1" ]; then B=$1; else
  B=$(BUILD="${BUILD:-$REPO/build/nvfx_3d}" bash "$REPO/tools/study_3d/nvfx_bin_link.sh")
fi
export NEURALVFX_DATA=${2:-$HOME/nvfx-data}
LOG=$NEURALVFX_DATA/hres_run_all.log
exec > >(tee -a "$LOG") 2>&1
echo "=== hres_run_all $(date -Iseconds) NEURALVFX_DATA=$NEURALVFX_DATA ==="
cd "$REPO"
chmod +x tools/study_3d/*.sh
"$REPO/tools/study_3d/hres_fire.sh" "$B" "$NEURALVFX_DATA/hres_3d/fire" 3 2
"$REPO/tools/study_3d/hres_fire_validate.sh" "$B" "$NEURALVFX_DATA/hres_3d/fire" || echo "fire validate skipped or failed (need n2_3d baseline?)"
"$REPO/tools/study_3d/hres_smoke.sh" "$B" "$NEURALVFX_DATA/hres_3d/smoke" 3 2
"$REPO/tools/study_3d/hres_smoke_validate.sh" "$B" "$NEURALVFX_DATA/hres_3d/smoke" || true
echo "=== hres_run_all done $(date -Iseconds) ==="
