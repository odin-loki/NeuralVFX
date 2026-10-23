#!/usr/bin/env bash
# Wait for hres smoke training to finish, then finish + validate (docs/STUDY_3D_HRES.md §6).
# Does not start training. Safe to run while CPU train is in progress; do not start a second train.
#
# Usage (repo root):
#   BUILD=build/nvfx_3d nohup bash tools/study_3d/hres_smoke_after_train.sh >> ~/nvfx-data/hres_3d/smoke/logs/after_train.log 2>&1 &
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/smoke
mkdir -p "$ROOT/logs"
LOCK="$ROOT/logs/after_train.lock"
DONE="$ROOT/logs/after_train.done"
exec 8>"$LOCK"
if ! flock -n 8; then
  echo "smoke watcher already running (lock $LOCK)"
  exit 0
fi
LOG="$ROOT/logs/train_smoke.log"
TARGET="done at iteration 1750"
POLL=${POLL_SEC:-120}
PIDFILE="$ROOT/logs/after_train.pid"
echo $$ >"$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
if [ -f "$DONE" ] && [ -f "$LOG" ] && [ "$DONE" -nt "$LOG" ] && [ -z "${FORCE:-}" ]; then
  echo "smoke after-train already done ($DONE); set FORCE=1 to rerun"
  exit 0
fi
B=$(BUILD="${BUILD:-$REPO/build/nvfx_3d}" bash "$REPO/tools/study_3d/nvfx_bin_link.sh")
echo "=== hres_smoke_after_train $(date -Iseconds) B=$B ROOT=$ROOT pid=$$ ==="
while [ -f "$LOG" ] && ! grep -q "$TARGET" "$LOG"; do
  last_it=$(grep '^it ' "$LOG" 2>/dev/null | awk '!seen[$0]++' | tail -1 || true)
  echo "$(date -Iseconds) waiting for smoke train ($TARGET)… last: ${last_it:-?}"
  sleep "$POLL"
done
if [ ! -f "$LOG" ] || ! grep -q "$TARGET" "$LOG"; then
  echo "error: $LOG missing or training not complete"
  exit 1
fi

H="--study hres --root $ROOT --effect smoke"
ALSO="--runs-dir train --also $ROOT/runs/long"
cd "$REPO"
for k in 1 2 3 4; do
  if [ -f "$ROOT/ckpt/smoke_stage$k.ckpt" ]; then
    echo "--- finish stage $k ---"
    "$B" finish $H --ckpt smoke_stage"$k" --out C"$k" $ALSO
  fi
done
echo "--- finish smoke_hres ---"
"$B" finish $H --ckpt smoke --out smoke_hres $ALSO
echo "--- validate ---"
"$REPO/tools/study_3d/hres_smoke_validate.sh" "$B" "$ROOT"
FIRE=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/fire
echo "--- publish models ---"
"$REPO/tools/study_3d/hres_publish_models.sh" "$FIRE" "$ROOT"
VAL_LOG="$FIRE/logs/validate.log"
if ! bash "$REPO/tools/study_3d/hres_validate_done.sh" "$VAL_LOG" 2>/dev/null; then
  echo "--- resume fire validate (smoke train finished) ---"
  if pgrep -f 'hres_fire_validate(\.sh|\.snapshot\.sh)|run_fire_validate\.sh' >/dev/null 2>&1; then
    echo "fire validate already running — skip start"
  else
    bash "$REPO/tools/study_3d/start_fire_validate.sh" 8>&-
  fi
  bash "$REPO/tools/study_3d/start_fire_publish_watcher.sh" 8>&-
fi
touch "$DONE"
echo "=== hres_smoke_after_train done $(date -Iseconds) ==="
