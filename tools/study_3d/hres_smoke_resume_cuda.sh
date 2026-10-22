#!/usr/bin/env bash
# Stop CPU smoke train (if any) and resume from ckpt/smoke.ckpt on the CUDA trainer.
# Never run this while a second nvfx_3d train on smoke is already active.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/smoke
BUILD_DIR=${BUILD_DIR:-$REPO/build-cuda}
LOG="$ROOT/logs/train_smoke.log"
mkdir -p "$ROOT/logs"

if pgrep -f "nvfx_3d train.*hres_3d/smoke" >/dev/null 2>&1; then
  echo "stopping CPU smoke train…"
  pkill -f "nvfx_3d train.*hres_3d/smoke" || true
  sleep 5
fi

if [ ! -f "$BUILD_DIR/nvfx_3d" ]; then
  cmake -S "$REPO" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 \
    -DNEURALFX_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-86}" -DNEURALFX_BUILD_TESTS=OFF
fi
ninja -C "$BUILD_DIR" nvfx_3d
B=$(BUILD="$BUILD_DIR/nvfx_3d" bash "$REPO/tools/study_3d/nvfx_bin_link.sh")
H="--study hres --root $ROOT --effect smoke"
ALSO="--runs-dir train --also $ROOT/runs/long"
echo "=== resume smoke CUDA $(date -Iseconds) nvfx=$B ===" | tee -a "$LOG"
# Progress goes to $LOG via log_line; do not tee stdout into the same file (duplicate lines).
export NVFX_TRAIN_QUIET_STDOUT=1
nohup "$B" train $H --name smoke $ALSO --threads 1 --device cuda --shards "${TRAIN_SHARDS:-4}" >>"$LOG" 2>&1 &
echo "started CUDA smoke train pid=$! (log $LOG)"
bash "$REPO/tools/study_3d/start_smoke_watcher.sh" 2>/dev/null || true
