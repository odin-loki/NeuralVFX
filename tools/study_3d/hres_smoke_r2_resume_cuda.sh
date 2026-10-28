#!/usr/bin/env bash
# Stop CPU smoke_r2 train (if any) and resume from ckpt/smoke_r2.ckpt on the CUDA trainer (spread/edge on CPU in-window).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
if [ -f "$REPO/tools/wsl_env.sh" ]; then
  # shellcheck source=/dev/null
  . "$REPO/tools/wsl_env.sh"
fi
ROOT=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/smoke_r2
BUILD_DIR=${BUILD_DIR:-$REPO/build-cuda}
BUILD_DIR=${BUILD_DIR//$'\r'/}
NVFX_3D_BIN="${NVFX_3D:-$BUILD_DIR/nvfx_3d}"
NVFX_3D_BIN=${NVFX_3D_BIN//$'\r'/}
LOG="$ROOT/logs/train_smoke_r2.log"
mkdir -p "$ROOT/logs"

if pgrep -f "nvfx_3d train.*hres_3d/smoke_r2" >/dev/null 2>&1; then
  echo "stopping smoke_r2 train…"
  pkill -f "nvfx_3d train.*hres_3d/smoke_r2" || true
  pkill -f hres_smoke_r2.sh || true
  sleep 5
fi
rm -f "$ROOT/logs/train.lock"

if [ ! -x "$NVFX_3D_BIN" ]; then
  cmake -S "$REPO" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 \
    -DNEURALFX_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-86}" -DNEURALFX_BUILD_TESTS=ON
fi
ninja -C "$BUILD_DIR" nvfx_3d

export TRAIN_DEVICE=cuda TRAIN_SHARDS="${TRAIN_SHARDS:-4}" FORCE=1 STAGE=train
export NVFX_3D="$NVFX_3D_BIN"
echo "=== resume smoke_r2 CUDA $(date -Iseconds) NVFX_3D=$NVFX_3D ===" | tee -a "$LOG"
FORCE=1 STAGE=train TRAIN_DEVICE=cuda TRAIN_SHARDS="${TRAIN_SHARDS:-4}" BUILD_DIR="$BUILD_DIR" NVFX_3D="$NVFX_3D_BIN" \
  bash "$REPO/tools/study_3d/hres_smoke_r2.sh" 2>&1 | tee -a "$LOG"
