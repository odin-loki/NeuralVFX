#!/usr/bin/env bash
# Babysit hres: smoke CUDA train → finish/validate/publish → fire validate → publish.
# Fire training must already be at 1750. Does not start a second smoke train.
#
# Usage (repo root):
#   nohup bash tools/study_3d/hres_train_until_done.sh >> ~/nvfx-data/hres_until_done.log 2>&1 &
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${NEURALVFX_DATA:-$HOME/nvfx-data}
FIRE="$DATA/hres_3d/fire"
SMOKE="$DATA/hres_3d/smoke"
LOG="$DATA/hres_until_done.log"
POLL=${POLL_SEC:-300}

log() { echo "$(date -Iseconds) $*"; }

cd "$REPO"
if ! grep -q 'done at iteration 1750' "$FIRE/logs/train_hres.log" 2>/dev/null; then
  log "error: fire train not at 1750 — run hres_fire.sh first"
  exit 1
fi

if ! pgrep -f 'nvfx_3d train.*hres_3d/smoke' >/dev/null 2>&1; then
  if ! grep -q 'done at iteration 1750' "$SMOKE/logs/train_smoke.log" 2>/dev/null; then
    log "starting smoke train (CUDA) — no process and not done"
    bash "$REPO/tools/study_3d/hres_smoke_resume_cuda.sh"
  fi
else
  log "smoke train already running"
fi

if ! [ -f "$SMOKE/logs/after_train.pid" ] || ! kill -0 "$(cat "$SMOKE/logs/after_train.pid")" 2>/dev/null; then
  bash "$REPO/tools/study_3d/start_smoke_watcher.sh"
fi

while ! grep -q 'done at iteration 1750' "$SMOKE/logs/train_smoke.log" 2>/dev/null; do
  last=$(grep '^it ' "$SMOKE/logs/train_smoke.log" 2>/dev/null | awk '!seen[$0]++' | tail -1 || echo '?')
  log "smoke training… $last"
  sleep "$POLL"
done
log "smoke train at 1750"

while [ ! -f "$SMOKE/models/smoke_hres.nvfx3" ]; do
  log "waiting for smoke finish (watcher)…"
  sleep "$POLL"
done
log "smoke_hres.nvfx3 ready"

if ! grep -q 'done: chosen' "$FIRE/logs/validate.log" 2>/dev/null; then
  if pgrep -f 'hres_fire_validate.sh' >/dev/null 2>&1; then
    log "fire validate already running"
  else
    log "starting fire validate"
    bash "$REPO/tools/study_3d/start_fire_validate.sh"
  fi
  bash "$REPO/tools/study_3d/start_fire_publish_watcher.sh"
  while ! grep -q 'done: chosen' "$FIRE/logs/validate.log" 2>/dev/null; do
    log "fire validate in progress…"
    sleep "$POLL"
  done
fi
log "fire validate complete"

bash "$REPO/tools/study_3d/hres_publish_models.sh" "$FIRE" "$SMOKE" | tee -a "$LOG"

log "=== hres_train_until_done: all training and validation finished ==="
