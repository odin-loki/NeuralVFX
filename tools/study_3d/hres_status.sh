#!/usr/bin/env bash
# One-line hres smoke/fire training status (data under NEURALVFX_DATA).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
DATA=${NEURALVFX_DATA:-$HOME/nvfx-data}
FIRE="$DATA/hres_3d/fire"
SMOKE="$DATA/hres_3d/smoke"
echo "=== hres status $(date -Iseconds) ==="
if [ -f "$FIRE/logs/train_hres.log" ] && grep -q 'done at iteration 1750' "$FIRE/logs/train_hres.log"; then
  echo "fire: train DONE"
else
  echo "fire: train $(tail -1 "$FIRE/logs/train_hres.log" 2>/dev/null || echo '?')"
fi
if [ -f "$FIRE/models/hres_fire.nvfx3" ]; then echo "fire: model hres_fire.nvfx3 OK"; fi
R=results/3d/hres/fire
if pgrep -f 'hres_fire_validate(\.sh|\.snapshot\.sh)|run_fire_validate\.sh' >/dev/null 2>&1 || pgrep -af 'nvfx_3d' 2>/dev/null | grep -q 'hres_3d/fire'; then
  echo "fire: validate RUNNING ($(pgrep -af nvfx_3d 2>/dev/null | grep hres_3d/fire | tail -1 || echo '?'))"
elif bash "$REPO/tools/study_3d/hres_validate_done.sh" "$FIRE/logs/validate.log" 2>/dev/null; then
  echo "fire: validate DONE"
elif [ -f "$FIRE/logs/validate.log" ]; then
  last_ev=$(grep -E 'validate (start|paused for smoke CUDA)' "$FIRE/logs/validate.log" | tail -1 || true)
  if [[ "$last_ev" == *paused* ]]; then
    echo "fire: validate PAUSED (resumes via smoke watcher at iter 1750, or start_fire_validate.sh)"
  else
    echo "fire: validate not started or incomplete (start: bash tools/study_3d/start_fire_validate.sh)"
  fi
else
  echo "fire: validate not started or incomplete (start: bash tools/study_3d/start_fire_validate.sh)"
fi
if [ -d "$R" ]; then
  n=$(find "$R" -maxdepth 1 -type f 2>/dev/null | wc -l)
  echo "fire: $R files: $n"
fi
if [ -f "$FIRE/logs/after_validate.pid" ] && kill -0 "$(cat "$FIRE/logs/after_validate.pid")" 2>/dev/null; then
  echo "fire: publish watcher pid=$(cat "$FIRE/logs/after_validate.pid")"
elif pgrep -f 'hres_fire_validate(\.sh|\.snapshot\.sh)|run_fire_validate\.sh' >/dev/null 2>&1; then
  echo "fire: publish watcher not running (start: bash tools/study_3d/start_fire_publish_watcher.sh)"
fi
if [ -f "$SMOKE/logs/train_smoke.log" ] && grep -q 'done at iteration 1750' "$SMOKE/logs/train_smoke.log"; then
  echo "smoke: train DONE"
else
  last=$(grep '^it ' "$SMOKE/logs/train_smoke.log" 2>/dev/null | awk '!seen[$0]++' | tail -1 || echo '?')
  echo "smoke: train $last"
  if grep -q "device cuda" "$SMOKE/logs/train_smoke.log" 2>/dev/null; then echo "smoke: device cuda"; fi
  mapfile -t _itlines < <(grep '^it ' "$SMOKE/logs/train_smoke.log" 2>/dev/null | awk '!seen[$0]++' | tail -2)
  if [ "${#_itlines[@]}" -ge 2 ] && [[ "${_itlines[1]}" =~ ^it[[:space:]]+([0-9]+).*\(([0-9]+)[[:space:]]s, ]]; then
    it="${BASH_REMATCH[1]}"
    sec2="${BASH_REMATCH[2]}"
    if [[ "${_itlines[0]}" =~ ^it[[:space:]]+([0-9]+).*\(([0-9]+)[[:space:]]s, ]]; then
      it0="${BASH_REMATCH[1]}"
      sec0="${BASH_REMATCH[2]}"
      dit=$((it - it0))
      if [ "$dit" -gt 0 ] && [ "$it" -lt 1750 ]; then
        spi=$(( (sec2 - sec0) / dit ))
        rem=$(( (1750 - it) * spi ))
        echo "smoke: ~$(( rem / 3600 )) h $(( (rem % 3600) / 60 )) m remaining (~${spi} s/iter from log cadence)"
      fi
    fi
  fi
  if pgrep -f 'nvfx_3d train.*hres_3d/smoke' >/dev/null 2>&1; then
    echo "smoke: trainer running (log every 10 iters; ~30 min between lines on CUDA is normal)"
  fi
  if pgrep -f 'hres_fire_validate(\.sh|\.snapshot\.sh)|run_fire_validate\.sh' >/dev/null 2>&1 && grep -q "device cuda" "$SMOKE/logs/train_smoke.log" 2>/dev/null; then
    echo "smoke: note fire validate also running (may slow train; pause_fire_validate.sh)"
  fi
  if [ -f "$SMOKE/logs/train_smoke.log" ]; then
    echo "smoke: log updated $(stat -c %y "$SMOKE/logs/train_smoke.log" 2>/dev/null || echo '?')"
  fi
fi
B="$HOME/.cache/neuralvfx/nvfx_3d"
if [ -L "$B" ] && [ -x "$B" ]; then
  echo "nvfx_3d link OK -> $(readlink -f "$B")"
elif [ -L "$B" ]; then
  echo "nvfx_3d link BROKEN ($(readlink "$B" 2>/dev/null || echo '?')) — run: BUILD=build/nvfx_3d bash tools/study_3d/nvfx_bin_link.sh"
fi
pgrep -af 'nvfx_3d train' 2>/dev/null || echo "no nvfx_3d train process"
if [ -f "$SMOKE/logs/after_train.pid" ] && kill -0 "$(cat "$SMOKE/logs/after_train.pid")" 2>/dev/null; then
  echo "smoke: finish watcher pid=$(cat "$SMOKE/logs/after_train.pid")"
else
  echo "smoke: finish watcher not running (start: bash tools/study_3d/start_smoke_watcher.sh)"
fi
if [ -f "$SMOKE/models/smoke_hres.nvfx3" ]; then echo "smoke: model smoke_hres.nvfx3 OK"; fi
