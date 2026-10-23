#!/usr/bin/env bash
# Wait for hres fire validation to finish, then publish fire model into repo models/.
# Does not start validation. Safe while hres_fire_validate.sh is running.
#
# Usage (repo root):
#   nohup bash tools/study_3d/hres_fire_after_validate.sh >> ~/nvfx-data/hres_3d/fire/logs/after_validate.log 2>&1 &
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${NEURALVFX_DATA:-$HOME/nvfx-data}
FIRE="$DATA/hres_3d/fire"
SMOKE="$DATA/hres_3d/smoke"
LOG="$FIRE/logs/validate.log"
DONE_SH="$REPO/tools/study_3d/hres_validate_done.sh"
POLL=${POLL_SEC:-180}
PIDFILE="$FIRE/logs/after_validate.pid"
echo $$ >"$PIDFILE"
echo "=== hres_fire_after_validate $(date -Iseconds) pid=$$ ==="
cd "$REPO"
while true; do
  if bash "$DONE_SH" "$LOG" 2>/dev/null; then
    break
  fi
  if ! pgrep -f 'hres_fire_validate(\.sh|\.snapshot\.sh)|run_fire_validate\.sh' >/dev/null 2>&1 && \
     ! pgrep -af nvfx_3d 2>/dev/null | grep -q 'hres_3d/fire'; then
    if bash "$DONE_SH" "$LOG" 2>/dev/null; then
      break
    fi
    echo "$(date -Iseconds) warn: validate stopped without done: chosen (see $LOG); still waiting or restart start_fire_validate.sh"
  else
    echo "$(date -Iseconds) waiting for fire validate (done: chosen after latest start)…"
  fi
  sleep "$POLL"
done
echo "--- publish fire model ---"
bash "$REPO/tools/study_3d/hres_publish_models.sh" "$FIRE" "$SMOKE"
echo "=== hres_fire_after_validate done $(date -Iseconds) ==="
