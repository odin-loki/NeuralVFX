#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/fire
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/validate.log"
LOCK="$ROOT/logs/validate.lock"
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "hres fire validate: another start in progress"
  exit 0
fi
if pgrep -f 'hres_fire_validate(\.sh|\.snapshot\.sh)|run_fire_validate\.sh' >/dev/null 2>&1; then
  echo "hres fire validate already running"
  exit 0
fi
echo "=== validate start $(date -Iseconds) threads=${THREADS:-default} nice=${NICE_LEVEL:-0} ===" >>"$LOG"
if [ -n "${NICE_LEVEL:-}" ]; then
  nohup nice -n "$NICE_LEVEL" bash "$REPO/tools/study_3d/run_fire_validate.sh" >>"$LOG" 2>&1 &
else
  nohup bash "$REPO/tools/study_3d/run_fire_validate.sh" >>"$LOG" 2>&1 &
fi
pid=$!
disown "$pid" 2>/dev/null || true
echo "started fire validate pid=$pid log=$LOG"
