#!/usr/bin/env bash
# Wait for smoke_r2 CUDA train to finish, then run finish, bounds, edge (STUDY_3D_HRES.md §10).
set -euo pipefail
REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
if [ -f "$REPO/tools/wsl_env.sh" ]; then
  # shellcheck source=/dev/null
  . "$REPO/tools/wsl_env.sh"
fi
ROOT=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/smoke_r2
LOG="$ROOT/logs/smoke_r2_pipeline.log"
BUILD_DIR=${BUILD_DIR:-$REPO/build-cuda}
mkdir -p "$ROOT/logs"
exec >>"$LOG" 2>&1
echo "=== smoke_r2 after_train watcher $(date -Iseconds) pid=$$ ==="
while pgrep -f 'nvfx_3d train --study hres --effect smoke_r2' >/dev/null 2>&1; do
  sleep 120
done
while ! grep -q 'done at iteration 1750' "$ROOT/logs/train_smoke_r2.log" 2>/dev/null; do
  echo "train not complete (no 'done at iteration 1750'); waiting 120s ($(date -Iseconds))"
  sleep 120
done
echo "train complete; starting finish,bounds,edge $(date -Iseconds)"
export FORCE=1 STAGE=finish,bounds,edge TRAIN_DEVICE=cuda TRAIN_SHARDS="${TRAIN_SHARDS:-4}"
export BUILD_DIR NVFX_3D="${NVFX_3D:-$BUILD_DIR/nvfx_3d}" NVFX_REPO="$REPO"
exec bash "$REPO/tools/study_3d/hres_smoke_r2.sh"
