#!/usr/bin/env bash
# Detached driver: train (CUDA) -> finish -> bounds -> edge. Logs to $ROOT/logs/smoke_r2_pipeline.log
set -euo pipefail
REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
ROOT=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/smoke_r2
BUILD_DIR=${BUILD_DIR:-$REPO/build-cuda}
mkdir -p "$ROOT/logs"
exec >>"$ROOT/logs/smoke_r2_pipeline.log" 2>&1
echo "=== smoke_r2 daemon $(date -Iseconds) pid=$$ ==="
export FORCE=1 STAGE=train,finish,bounds,edge TRAIN_DEVICE=cuda TRAIN_SHARDS="${TRAIN_SHARDS:-4}"
export BUILD_DIR NVFX_3D="${NVFX_3D:-$BUILD_DIR/nvfx_3d}" NVFX_REPO="$REPO"
exec bash "$REPO/tools/study_3d/hres_smoke_r2.sh"
