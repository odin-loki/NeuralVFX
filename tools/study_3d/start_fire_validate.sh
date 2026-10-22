#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/fire
mkdir -p "$ROOT/logs"
if pgrep -f "run_fire_validate.sh" >/dev/null 2>&1; then
  echo "hres fire validate already running"
  exit 0
fi
LOG="$ROOT/logs/validate.log"
echo "=== validate start $(date -Iseconds) ===" >>"$LOG"
bash "$REPO/tools/study_3d/run_fire_validate.sh" >>"$LOG" 2>&1 &
pid=$!
disown "$pid" 2>/dev/null || true
echo "started fire validate pid=$pid log=$LOG"
