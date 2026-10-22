#!/usr/bin/env bash
# Live hres pipeline monitor (60 s). Usage: bash tools/study_3d/monitor_hres.sh
REPO=$(cd "$(dirname "$0")/../.." && pwd)
echo "=== hres monitor (Ctrl+C to stop) ==="
while true; do
  bash "$REPO/tools/study_3d/hres_status.sh" | sed 's/^/  /'
  if ! pgrep -f 'nvfx_3d train.*hres_3d|hres_fire_validate|hres_run_all|hres_smoke_after_train|hres_fire_after_validate' >/dev/null 2>&1; then
    echo "  No active hres train/validate/watchers — monitor exiting."
    break
  fi
  sleep 60
done
