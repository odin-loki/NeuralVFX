#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/smoke
mkdir -p "$ROOT/logs"
if [ -f "$ROOT/logs/after_train.pid" ] && kill -0 "$(cat "$ROOT/logs/after_train.pid")" 2>/dev/null; then
  echo "watcher already running pid=$(cat "$ROOT/logs/after_train.pid")"
  exit 0
fi
nohup bash "$REPO/tools/study_3d/hres_smoke_after_train.sh" >>"$ROOT/logs/after_train.log" 2>&1 &
sleep 1
tail -3 "$ROOT/logs/after_train.log" 2>/dev/null || true
echo "started watcher"
