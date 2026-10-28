#!/usr/bin/env bash
# Detached: validate (resume) then publish. Log: $NEURALVFX_DATA/hres_3d/smoke_r2/logs/validate_pipeline.log
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
if [ -f "$REPO/tools/wsl_env.sh" ]; then
  # shellcheck source=/dev/null
  . "$REPO/tools/wsl_env.sh"
fi
ROOT="${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/smoke_r2"
mkdir -p "$ROOT/logs"
PIPE="$REPO/tools/study_3d/hres_smoke_r2_validate_pipeline.sh"

if pgrep -f 'hres_smoke_r2_validate_pipeline\.sh|run_smoke_r2_validate\.sh|hres_smoke_r2_validate\.snapshot\.sh' >/dev/null 2>&1; then
  echo "smoke_r2 validate pipeline already running"
  pgrep -af 'hres_smoke_r2_validate|run_smoke_r2_validate' || true
  exit 0
fi

if [ -n "${NICE_LEVEL:-}" ]; then
  nohup nice -n "$NICE_LEVEL" bash "$PIPE" >>"$ROOT/logs/validate_pipeline.log" 2>&1 &
else
  nohup bash "$PIPE" >>"$ROOT/logs/validate_pipeline.log" 2>&1 &
fi
pid=$!
disown "$pid" 2>/dev/null || true
echo "started smoke_r2 validate pipeline pid=$pid"
echo "  validate log: $ROOT/logs/validate.log"
echo "  pipeline log: $ROOT/logs/validate_pipeline.log"
