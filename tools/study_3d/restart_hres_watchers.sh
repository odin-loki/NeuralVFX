#!/usr/bin/env bash
# Restart hres watcher scripts so they pick up on-disk fixes. Does NOT kill the smoke CUDA trainer.
# Usage: TRAIN_PID=33217 DRY=1 bash tools/study_3d/restart_hres_watchers.sh
#        DRY=0 bash tools/study_3d/restart_hres_watchers.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
S="$REPO/tools/study_3d"
DATA=${NEURALVFX_DATA:-$HOME/nvfx-data}
SMOKE="$DATA/hres_3d/smoke"
FIRE="$DATA/hres_3d/fire"
TRAIN_PID=${TRAIN_PID:-$(pgrep -f 'nvfx_3d train.*hres_3d/smoke' 2>/dev/null | head -1 || true)}
DRY=${DRY:-1}

die() { echo "ABORT: $*" >&2; exit 1; }
run() { if [ "$DRY" = 1 ]; then echo "DRY> $*"; else echo "+ $*"; "$@"; fi; }

[ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null || die "smoke train not running (set TRAIN_PID if needed)"
cmd=$(tr '\0' ' ' </proc/"$TRAIN_PID"/cmdline)
[[ "$cmd" == *nvfx_3d*" train "*hres_3d/smoke* ]] || die "pid $TRAIN_PID is not smoke train: $cmd"
MATCH=$(pgrep -f 'nvfx_3d train.*hres_3d/smoke' | wc -l)
[ "$MATCH" -eq 1 ] || die "expected exactly one smoke train, pgrep count=$MATCH"

WATCH_RE='hres_train_until_done\.sh|hres_smoke_after_train\.sh|hres_fire_after_validate\.sh'
mapfile -t WATCHERS < <(pgrep -f "$WATCH_RE" || true)
echo "smoke train pid=$TRAIN_PID; watchers to stop: ${WATCHERS[*]:-none}"

for p in "${WATCHERS[@]}"; do
  [ "$p" = "$TRAIN_PID" ] && die "train pid in watcher list?"
  c=$(tr '\0' ' ' </proc/"$p"/cmdline 2>/dev/null || echo '')
  [[ "$c" == *bash* ]] || continue
  run kill -TERM -- "$p"
done
if [ "$DRY" != 1 ]; then
  sleep 2
  for p in "${WATCHERS[@]}"; do kill -0 "$p" 2>/dev/null && kill -KILL -- "$p" || true; done
fi

AT="$SMOKE/logs/after_train.log"
if [ -f "$AT" ] && grep -q 'hres_smoke_after_train done' "$AT"; then
  run mv "$AT" "$AT.pre_restart.$(date +%Y%m%dT%H%M%S)"
fi
run rm -f "$SMOKE/logs/after_train.pid" "$SMOKE/logs/after_train.done" "$FIRE/logs/after_validate.pid"

cd "$REPO"
export BUILD="${BUILD:-$REPO/build/nvfx_3d}"
if [ "$DRY" != 1 ]; then
  bash "$S/nvfx_bin_link.sh" "$BUILD" >/dev/null
else
  echo "DRY> bash $S/nvfx_bin_link.sh $BUILD"
fi
run bash "$S/start_smoke_watcher.sh"
run bash "$S/start_fire_publish_watcher.sh"
if [ "$DRY" = 1 ]; then
  echo "DRY> setsid -f bash $S/hres_train_until_done.sh >>$DATA/hres_until_done.log 2>&1"
else
  setsid -f bash "$S/hres_train_until_done.sh" >>"$DATA/hres_until_done.log" 2>&1
fi
echo "done DRY=$DRY; train pid $TRAIN_PID should still be alive"
