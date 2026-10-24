#!/usr/bin/env bash
# Restart hres watcher scripts so they pick up on-disk fixes. Never starts, nor kills, a smoke train.
# Idempotent: with the smoke train done and the pipeline finished (after_train.done and fire validate done) it is a
# no-op; finished work is not redone (after_train.done is kept) unless FORCE=1, which moves after_train.log aside
# and lets the smoke watcher re-run finish + validate + publish.
# Usage: TRAIN_PID=33217 DRY=1 bash tools/study_3d/restart_hres_watchers.sh
#        DRY=0 bash tools/study_3d/restart_hres_watchers.sh
#        DRY=0 FORCE=1 bash tools/study_3d/restart_hres_watchers.sh   # pipeline finished; redo smoke finish/validate
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
S="$REPO/tools/study_3d"
DATA=${NEURALVFX_DATA:-$HOME/nvfx-data}
SMOKE="$DATA/hres_3d/smoke"
FIRE="$DATA/hres_3d/fire"
TRAIN_RE='nvfx_3d train.*hres_3d/smoke'
TRAIN_PID=${TRAIN_PID:-$(pgrep -f "$TRAIN_RE" 2>/dev/null | head -1 || true)}
DRY=${DRY:-1}
FORCE=${FORCE:-0}

die() { echo "ABORT: $*" >&2; exit 1; }
run() { if [ "$DRY" = 1 ]; then echo "DRY> $*"; else echo "+ $*"; "$@"; fi; }

SMOKE_TRAIN_DONE=0
grep -q 'done at iteration 1750' "$SMOKE/logs/train_smoke.log" 2>/dev/null && SMOKE_TRAIN_DONE=1
MATCH=$({ pgrep -f "$TRAIN_RE" 2>/dev/null || true; } | wc -l)
[ "$MATCH" -le 1 ] || die "pgrep count=$MATCH: more than one smoke train is running; stop the duplicate by hand first"

if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
  cmd=$(tr '\0' ' ' </proc/"$TRAIN_PID"/cmdline)
  [[ "$cmd" == *nvfx_3d*" train "*hres_3d/smoke* ]] || die "pid $TRAIN_PID is not smoke train: $cmd"
  [ "$MATCH" -eq 1 ] || die "TRAIN_PID=$TRAIN_PID given but pgrep count=$MATCH"
  echo "smoke train pid=$TRAIN_PID running"
elif [ "$SMOKE_TRAIN_DONE" = 1 ]; then
  TRAIN_PID=""
  echo "smoke train done (no process); restarting watchers only"
else
  die "smoke train not running and not at 1750 — this script never starts a train (use hres_smoke_resume_cuda.sh)"
fi

PIPELINE_DONE=0
if [ -f "$SMOKE/logs/after_train.done" ] && bash "$S/hres_validate_done.sh" "$FIRE/logs/validate.log" 2>/dev/null; then
  PIPELINE_DONE=1
fi

WATCH_RE='hres_train_until_done\.sh|hres_smoke_after_train\.sh|hres_fire_after_validate\.sh'
mapfile -t WATCHERS < <(pgrep -f "$WATCH_RE" || true)
echo "watchers to stop: ${WATCHERS[*]:-none}"

if [ "$PIPELINE_DONE" = 1 ] && [ "$FORCE" != 1 ]; then
  if [ "${#WATCHERS[@]}" -eq 0 ]; then
    echo "hres pipeline finished (after_train.done + fire validate done) and no watcher running: nothing to do (FORCE=1 to redo smoke finish/validate)"
    exit 0
  fi
  echo "hres pipeline finished: stopping stale watchers only; not restarting (FORCE=1 to redo smoke finish/validate)"
fi

for p in "${WATCHERS[@]}"; do
  [ -n "$TRAIN_PID" ] && [ "$p" = "$TRAIN_PID" ] && die "train pid in watcher list?"
  c=$(tr '\0' ' ' </proc/"$p"/cmdline 2>/dev/null || echo '')
  [[ "$c" == *bash* ]] || continue
  run kill -TERM -- "$p"
done
if [ "$DRY" != 1 ] && [ "${#WATCHERS[@]}" -gt 0 ]; then
  sleep 2
  for p in "${WATCHERS[@]}"; do kill -0 "$p" 2>/dev/null && kill -KILL -- "$p" || true; done
fi

if [ "$PIPELINE_DONE" = 1 ] && [ "$FORCE" != 1 ]; then
  run rm -f "$SMOKE/logs/after_train.pid" "$FIRE/logs/after_validate.pid"
  echo "done DRY=$DRY (pipeline finished; watchers not restarted)"
  exit 0
fi

AT="$SMOKE/logs/after_train.log"
if [ "$FORCE" = 1 ]; then
  if [ -f "$AT" ] && grep -q 'hres_smoke_after_train done' "$AT"; then
    run mv "$AT" "$AT.pre_restart.$(date +%Y%m%dT%H%M%S)"
  fi
  run rm -f "$SMOKE/logs/after_train.done"
else
  echo "keeping $SMOKE/logs/after_train.done if present (finished work is not redone; FORCE=1 to redo)"
fi
run rm -f "$SMOKE/logs/after_train.pid" "$FIRE/logs/after_validate.pid"

cd "$REPO"
export BUILD="${BUILD:-$REPO/build/nvfx_3d}"
if [ "$DRY" != 1 ]; then
  bash "$S/nvfx_bin_link.sh" "$BUILD" >/dev/null
else
  echo "DRY> bash $S/nvfx_bin_link.sh $BUILD"
fi
run bash "$S/start_smoke_watcher.sh"
run bash "$S/start_fire_publish_watcher.sh"
# hres_train_until_done.sh starts a smoke train only when none runs and the log is not at 1750; both are excluded
# above, so launching it here cannot start a second train. Still, never run two of it.
if [ "$DRY" = 1 ]; then
  echo "DRY> setsid -f bash $S/hres_train_until_done.sh >>$DATA/hres_until_done.log 2>&1"
elif pgrep -f 'hres_train_until_done\.sh' >/dev/null 2>&1; then
  echo "hres_train_until_done.sh still running (not killed above?); not launching another"
else
  setsid -f bash "$S/hres_train_until_done.sh" >>"$DATA/hres_until_done.log" 2>&1
fi
echo "done DRY=$DRY${TRAIN_PID:+; train pid $TRAIN_PID should still be alive}"
