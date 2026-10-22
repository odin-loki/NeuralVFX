#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/fire
mkdir -p "$ROOT/logs"
if [ -f "$ROOT/logs/after_validate.pid" ] && kill -0 "$(cat "$ROOT/logs/after_validate.pid")" 2>/dev/null; then
  echo "fire publish watcher already running pid=$(cat "$ROOT/logs/after_validate.pid")"
  exit 0
fi
nohup bash "$REPO/tools/study_3d/hres_fire_after_validate.sh" >>"$ROOT/logs/after_validate.log" 2>&1 &
sleep 1
tail -3 "$ROOT/logs/after_validate.log" 2>/dev/null || true
echo "started fire publish watcher"
