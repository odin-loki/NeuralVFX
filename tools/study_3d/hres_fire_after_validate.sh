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
TARGET='done: chosen'
POLL=${POLL_SEC:-180}
PIDFILE="$FIRE/logs/after_validate.pid"
echo $$ >"$PIDFILE"
echo "=== hres_fire_after_validate $(date -Iseconds) pid=$$ ==="
cd "$REPO"
while true; do
  if [ -f "$LOG" ] && grep -q "$TARGET" "$LOG"; then
    break
  fi
  if ! pgrep -f 'hres_fire_validate.sh' >/dev/null 2>&1 && \
     ! pgrep -af nvfx_3d 2>/dev/null | grep -q 'hres_3d/fire'; then
    if [ -f "$LOG" ] && grep -q "$TARGET" "$LOG"; then
      break
    fi
    echo "$(date -Iseconds) warn: no validate process and log missing '$TARGET'; still waiting"
  else
    echo "$(date -Iseconds) waiting for fire validate ($TARGET)…"
  fi
  sleep "$POLL"
done
echo "--- publish fire model ---"
"$REPO/tools/study_3d/hres_publish_models.sh" "$FIRE" "$SMOKE"
echo "=== hres_fire_after_validate done $(date -Iseconds) ==="
