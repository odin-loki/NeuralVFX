#!/bin/sh
# Live hres pipeline monitor (60 s). Usage: monitor_hres.sh
LOG_FIRE="$HOME/nvfx-data/hres_3d/fire/logs/train_hres.log"
LOG_SMOKE="$HOME/nvfx-data/hres_3d/smoke/logs/train_smoke.log"
ALL="$HOME/nvfx-data/hres_run_all.log"
echo "=== hres monitor (Ctrl+C to stop) ==="
while true; do
  TS=$(date +%H:%M:%S)
  if pgrep -f 'nvfx_3d train.*hres_3d/fire' >/dev/null 2>&1; then
    PHASE=FIRE_TRAIN
    LAST=$(grep '^it ' "$LOG_FIRE" 2>/dev/null | tail -1)
  elif pgrep -f 'nvfx_3d train.*hres_3d/smoke' >/dev/null 2>&1; then
    PHASE=SMOKE_TRAIN
    LAST=$(grep '^it ' "$LOG_SMOKE" 2>/dev/null | tail -1)
  elif pgrep -f hres_run_all >/dev/null 2>&1; then
    PHASE=PIPELINE
    LAST=$(tail -1 "$ALL" 2>/dev/null)
  else
    PHASE=STOPPED
    LAST=
  fi
  MEM=$(free -h | awk '/^Mem:/{printf "%s / %s used", $3, $1}')
  echo "[$TS] $PHASE | $MEM"
  [ -n "$LAST" ] && echo "  $LAST"
  [ "$PHASE" = STOPPED ] && echo "No hres_run_all or nvfx_3d train — monitor exiting." && break
  sleep 60
done
