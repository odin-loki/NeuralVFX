#!/usr/bin/env bash
# Finish study hres on a big CPU box (record → train with volume shards → finish → validate → git publish).
# CPU default (AVX2/512). Optional GPU training: TRAIN_DEVICE=cuda TRAIN_SHARDS=4 and build with -DNEURALFX_CUDA=ON.
#
# Usage (repo root, WSL):
#   export NEURALVFX_DATA=~/nvfx-data
#   nohup bash tools/study_3d/hres_machine_run.sh >> ~/nvfx-data/hres_machine_run.log 2>&1 &
#
# Env:
#   RECORD_THREADS  default min(8, nproc/4) for sim recording
#   TRAIN_THREADS default 2 (one hres train at a time; ~12 GiB)
#   SKIP_FIRE=1   skip fire record/train if already at 1750 iters
#   SKIP_GIT=1    skip commit/push at the end
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
export NEURALVFX_DATA=${NEURALVFX_DATA:-$HOME/nvfx-data}
B=${BUILD:-$REPO/build/nvfx_3d}
RECORD_THREADS=${RECORD_THREADS:-$(nproc)}
if [ "$RECORD_THREADS" -gt 8 ]; then RECORD_THREADS=8; fi
TRAIN_THREADS=${TRAIN_THREADS:-2}
TRAIN_DEVICE=${TRAIN_DEVICE:-cpu}
TRAIN_SHARDS=${TRAIN_SHARDS:-4}
TRAIN_EXTRA="--device $TRAIN_DEVICE --shards $TRAIN_SHARDS"
export NVFX_TRAIN_QUIET_STDOUT=${NVFX_TRAIN_QUIET_STDOUT:-1}
LOG="$NEURALVFX_DATA/hres_machine_run.log"
FIRE_ROOT="$NEURALVFX_DATA/hres_3d/fire"
SMOKE_ROOT="$NEURALVFX_DATA/hres_3d/smoke"

exec > >(tee -a "$LOG") 2>&1
echo "=== hres_machine_run $(date -Iseconds) REPO=$REPO DATA=$NEURALVFX_DATA RECORD_THREADS=$RECORD_THREADS TRAIN_THREADS=$TRAIN_THREADS ==="

cd "$REPO"
if command -v dos2unix >/dev/null 2>&1; then dos2unix tools/study_3d/*.sh 2>/dev/null || true; fi
chmod +x tools/study_3d/*.sh

if [ -x "$REPO/models/restore.sh" ]; then
  echo "--- restore shipped models into NEURALVFX_DATA (for N2 baselines) ---"
  "$REPO/models/restore.sh" "$NEURALVFX_DATA" || echo "restore.sh: some paths differ (ok if 2d tree missing)"
fi

echo "--- build nvfx_3d ---"
if [ "$TRAIN_DEVICE" = cuda ]; then
  BUILD_DIR=${BUILD_DIR:-$REPO/build-cuda}
  if [ ! -f "$BUILD_DIR/nvfx_3d" ]; then
    cmake -S "$REPO" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 \
      -DNEURALFX_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DNEURALFX_BUILD_TESTS=OFF
  fi
else
  BUILD_DIR=${BUILD_DIR:-$REPO/build}
fi
if [ -d "$BUILD_DIR" ]; then
  (cd "$BUILD_DIR" && ninja nvfx_3d)
  B=$(BUILD="$BUILD_DIR/nvfx_3d" bash "$REPO/tools/study_3d/nvfx_bin_link.sh")
else
  echo "missing build/; run cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 && ninja nvfx_3d"
  exit 1
fi

fire_done() {
  [ -f "$FIRE_ROOT/logs/train_hres.log" ] && grep -q 'done at iteration 1750' "$FIRE_ROOT/logs/train_hres.log"
}

smoke_done() {
  [ -f "$SMOKE_ROOT/logs/train_smoke.log" ] && grep -q 'done at iteration 1750' "$SMOKE_ROOT/logs/train_smoke.log"
}

H_FIRE="--study hres --root $FIRE_ROOT --effect fire"
H_SMOKE="--study hres --root $SMOKE_ROOT --effect smoke"
ALSO_FIRE="--runs-dir train --also $FIRE_ROOT/runs/long"
ALSO_SMOKE="--runs-dir train --also $SMOKE_ROOT/runs/long"

if [ "${SKIP_FIRE:-0}" != 1 ] && ! fire_done; then
  echo "--- fire: record (teacher sim3d) ---"
  "$B" record $H_FIRE --runs 40 --frames 240 --threads "$RECORD_THREADS"
  "$B" record $H_FIRE --runs-dir long --from 0 --runs 24 --frames 540 --threads "$RECORD_THREADS"
  echo "--- fire: train (resumable; volume playback uses 8 s shards in the runtime) ---"
  "$B" train $H_FIRE --name hres $ALSO_FIRE --threads "$TRAIN_THREADS" $TRAIN_EXTRA
else
  echo "--- fire: train complete or SKIP_FIRE=1; ensure finish exports ---"
fi

if ! [ -f "$FIRE_ROOT/models/hres_fire.nvfx3" ]; then
  echo "--- fire: finish stage snapshots + hres_fire ---"
  for k in 1 2 3 4; do
    if [ -f "$FIRE_ROOT/ckpt/hres_stage$k.ckpt" ]; then
      "$B" finish $H_FIRE --ckpt hres_stage"$k" --out C"$k" $ALSO_FIRE
    fi
  done
  "$B" finish $H_FIRE --ckpt hres --out hres_fire $ALSO_FIRE
fi

if ! smoke_done; then
  if [ ! -d "$SMOKE_ROOT/runs/train" ] || [ -z "$(ls -A "$SMOKE_ROOT/runs/train" 2>/dev/null)" ]; then
    echo "--- smoke: record ---"
    "$B" record $H_SMOKE --runs 40 --frames 240 --threads "$RECORD_THREADS"
    "$B" record $H_SMOKE --runs-dir long --from 0 --runs 24 --frames 540 --threads "$RECORD_THREADS"
  else
    echo "--- smoke: runs present; skip record ---"
  fi
  if pgrep -f "nvfx_3d train.*hres_3d/smoke" >/dev/null 2>&1; then
    echo "--- smoke: train already running (do not start a second); watcher: bash tools/study_3d/start_smoke_watcher.sh ---"
  else
    echo "--- smoke: train (resumable) ---"
    "$B" train $H_SMOKE --name smoke $ALSO_SMOKE --threads "$TRAIN_THREADS" $TRAIN_EXTRA
  fi
else
  echo "--- smoke: train already at 1750 ---"
fi

if ! [ -f "$SMOKE_ROOT/models/smoke_hres.nvfx3" ]; then
  echo "--- smoke: finish ---"
  for k in 1 2 3 4; do
    if [ -f "$SMOKE_ROOT/ckpt/smoke_stage$k.ckpt" ]; then
      "$B" finish $H_SMOKE --ckpt smoke_stage"$k" --out C"$k" $ALSO_SMOKE
    fi
  done
  "$B" finish $H_SMOKE --ckpt smoke --out smoke_hres $ALSO_SMOKE
fi

echo "--- fire validate (long: edge sim + val; may take hours) ---"
"$REPO/tools/study_3d/hres_fire_validate.sh" "$B" "$FIRE_ROOT" || echo "fire validate failed (see log)"

echo "--- smoke validate ---"
"$REPO/tools/study_3d/hres_smoke_validate.sh" "$B" "$SMOKE_ROOT" || echo "smoke validate failed (see log)"

echo "--- publish models into repo models/ ---"
"$REPO/tools/study_3d/hres_publish_models.sh" "$FIRE_ROOT" "$SMOKE_ROOT"

if [ "${SKIP_GIT:-0}" != 1 ]; then
  echo "--- git commit and push ---"
  "$REPO/tools/study_3d/hres_publish_git.sh"
fi

echo "=== hres_machine_run done $(date -Iseconds) ==="
