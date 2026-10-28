#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
if [ -f "$REPO/tools/wsl_env.sh" ]; then
  # shellcheck source=/dev/null
  . "$REPO/tools/wsl_env.sh"
fi
R2="${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/smoke_r2"
LOG="$R2/logs/validate.log"
DONE_SH="$REPO/tools/study_3d/hres_validate_done.sh"
POLL=${POLL_SEC:-180}
PIDFILE="$R2/logs/after_validate.pid"
echo $$ >"$PIDFILE"
echo "=== hres_smoke_r2_after_validate $(date -Iseconds) pid=$$ ==="
cd "$REPO"
while true; do
  if bash "$DONE_SH" "$LOG" 2>/dev/null; then break; fi
  if ! pgrep -f 'hres_smoke_r2_validate(\.sh|\.snapshot\.sh)|run_smoke_r2_validate\.sh' >/dev/null 2>&1 && \
     ! pgrep -af nvfx_3d 2>/dev/null | grep -q 'hres_3d/smoke_r2'; then
    if bash "$DONE_SH" "$LOG" 2>/dev/null; then break; fi
    echo "$(date -Iseconds) warn: validate stopped without done: chosen (see $LOG)"
  else
    echo "$(date -Iseconds) waiting for smoke_r2 validate…"
  fi
  sleep "$POLL"
done
bash "$REPO/tools/study_3d/hres_smoke_r2_publish.sh"
echo "=== hres_smoke_r2_after_validate done $(date -Iseconds) ==="
