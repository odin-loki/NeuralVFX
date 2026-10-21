#!/bin/sh
# Run study hres end-to-end: fire (record, train, finish, validate) then smoke. One train job at a time.
# Usage: hres_run_all.sh [BUILD_DIR] [NEURALVFX_DATA]
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
B=${1:-$REPO/build/nvfx_3d}
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
