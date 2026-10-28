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
  # The test-phase tables (edgecheck_test.md, compare_test_*) belong beside the val ones. A snapshot run of
  # hres_fire_validate.sh before NVFX_REPO existed wrote them under the data root instead; present since 24 October.
  if [ -f "$R/edgecheck_test.md" ] && [ -f "$R/compare_test_learned_vs_coarse.md" ] && [ -f "$R/compare_test_learned_vs_N2.md" ]; then
    verdict=$(grep -E '^learned:' "$R/edgecheck_test.md" | tail -1 || true)
    if grep -q 'calibrated bound' "$R/edgecheck_test.md"; then
      echo "fire: test tables present (${verdict:-edgecheck_test.md has no learned verdict line}; calibrated bounds, edge_bounds.csv)"
    elif [ -f "$R/edge_bounds.csv" ]; then
      echo "fire: test tables present (${verdict:-edgecheck_test.md has no learned verdict line}; single-run bounds: judge with the calibrated ones: bash tools/study_3d/hres_fire_recheck_test_edges.sh)"
    else
      echo "fire: test tables present (${verdict:-edgecheck_test.md has no learned verdict line}; single-run bounds, no edge_bounds.csv yet: bash tools/study_3d/hres_fire_edge_bounds.sh)"
    fi
  else
    STRAY="$DATA/hres_3d/results/3d/hres/fire"
    if [ -f "$STRAY/edgecheck_test.md" ]; then
      echo "fire: test tables missing from $R; found under $STRAY (copy: bash tools/study_3d/hres_sync_fire_test_results.sh)"
    else
      echo "fire: test tables (edgecheck_test.md, compare_test_*) missing from $R (rerun the test step: bash tools/study_3d/start_fire_validate.sh)"
    fi
  fi
fi
if [ -f "models/3d/fire_hres.nvfx3" ]; then echo "fire: published models/3d/fire_hres.nvfx3"; fi
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
RS=results/3d/hres/smoke
SMOKE_VAL_DONE=0
if [ -f "$RS/edgecheck.md" ] && [ -f "$RS/edge.csv" ] && grep -q 'edge check' "$RS/edgecheck.md"; then
  SMOKE_VAL_DONE=1
fi
if [ -f "$SMOKE/logs/after_train.pid" ] && kill -0 "$(cat "$SMOKE/logs/after_train.pid")" 2>/dev/null; then
  echo "smoke: finish watcher pid=$(cat "$SMOKE/logs/after_train.pid")"
elif [ "$SMOKE_VAL_DONE" = 1 ]; then
  verdict=$(grep -E '^smoke_hres:' "$RS/edgecheck.md" | tail -1 || true)
  echo "smoke: validate DONE ($RS/edgecheck.md: ${verdict:-no smoke_hres verdict line}; after_train.done $([ -f "$SMOKE/logs/after_train.done" ] && echo present || echo missing))"
elif [ -f "$SMOKE/logs/after_train.done" ]; then
  echo "smoke: watcher finished (after_train.done) but $RS/edgecheck.md missing (rerun: bash tools/study_3d/hres_smoke_validate.sh)"
else
  echo "smoke: finish watcher not running (start: bash tools/study_3d/start_smoke_watcher.sh)"
fi
if [ -f "$SMOKE/models/smoke_hres.nvfx3" ]; then echo "smoke: model smoke_hres.nvfx3 OK"; fi
if [ -f "models/3d/smoke_hres.nvfx3" ]; then echo "smoke: published models/3d/smoke_hres.nvfx3"; fi
# smoke on study R2's teacher: the workflow is docs/STUDY_3D_HRES.md §8 (why), §10 (the plan, the rule, the driver's
# stages and files); the driver is tools/study_3d/hres_smoke_r2.sh (hres_smoke_r2_train.sh = its record,train,finish).
# Reports: runs, training, the driver or trainer running, candidates, bounds, edge-check verdicts. Same root as the driver.
R2="${HRES_R2_ROOT:-${HRES_SMOKE_R2_ROOT:-$DATA/hres_3d/smoke_r2}}"
RR2=results/3d/hres/smoke_r2
R2_DOC="docs/STUDY_3D_HRES.md §10"
if [ -d "$R2" ] || [ -d "$RR2" ]; then
  ns=$(ls -- "$R2/runs/train/"*.n3run 2>/dev/null | wc -l || true); nl=$(ls -- "$R2/runs/long/"*.n3run 2>/dev/null | wc -l || true)
  echo "smoke_r2: runs ${ns:-0}/40 short, ${nl:-0}/24 long ($R2; $R2_DOC)"
  if drv=$(pgrep -af 'hres_smoke_r2(_train)?\.sh' 2>/dev/null | head -1) && [ -n "$drv" ]; then
    echo "smoke_r2: driver running ($drv)"
  fi
  if grep -q 'done at iteration 1750' "$R2/logs/train_smoke_r2.log" 2>/dev/null; then
    echo "smoke_r2: train DONE"
  else
    last=$(grep '^it ' "$R2/logs/train_smoke_r2.log" 2>/dev/null | awk '!seen[$0]++' | tail -1 || true)
    echo "smoke_r2: train ${last:-not started}"
    if pgrep -f "nvfx_3d[^ ]* train --study hres --root $R2" >/dev/null 2>&1; then echo "smoke_r2: trainer running (CPU; spread and edge losses)"; fi
  fi
  for c in C2 C3 C4 smoke_r2_hres; do [ -f "$R2/models/$c.nvfx3" ] && echo "smoke_r2: model $c.nvfx3 OK"; done
  if [ -f "$RR2/edge_bounds.csv" ]; then echo "smoke_r2: bounds $RR2/edge_bounds.csv (calibrated on the teacher's runs)"; fi
  if [ -f "$RR2/edgecheck.md" ]; then
    echo "smoke_r2: edge check $RR2/edgecheck.md: $(grep -E '^C[234]:' "$RR2/edgecheck.md" | tr '\n' ' ' || true)"
  elif [ -f "$RR2/edge_bounds.csv" ] && [ -f "$R2/models/C4.nvfx3" ]; then
    echo "smoke_r2: bounds and candidates ready, edge check pending (FORCE=1 STAGE=edge bash tools/study_3d/hres_smoke_r2.sh)"
  fi
  if pgrep -f 'hres_smoke_r2_validate_pipeline\.sh|hres_smoke_r2_validate(\.sh|\.snapshot\.sh)|run_smoke_r2_validate\.sh' >/dev/null 2>&1 || \
     pgrep -af nvfx_3d 2>/dev/null | grep -q 'hres_3d/smoke_r2'; then
    echo "smoke_r2: validate RUNNING (log $R2/logs/validate.log)"
  elif bash "$REPO/tools/study_3d/hres_validate_done.sh" "$R2/logs/validate.log" 2>/dev/null; then
    echo "smoke_r2: validate DONE (chosen under $R2/models/; publish: bash tools/study_3d/hres_smoke_r2_publish.sh)"
    [ -f "$REPO/models/3d/smoke_r2_hres.nvfx3" ] && echo "smoke_r2: published models/3d/smoke_r2_hres.nvfx3"
  elif [ -f "$R2/logs/validate.log" ]; then
    echo "smoke_r2: validate not complete (start: bash tools/study_3d/start_smoke_r2_validate_pipeline.sh)"
  elif grep -qE '^C[23]: passes' "$RR2/edgecheck.md" 2>/dev/null; then
    echo "smoke_r2: validation not started (start: bash tools/study_3d/start_smoke_r2_validate_pipeline.sh)"
  fi
  echo "smoke_r2: resume or next stage: FORCE=1 bash tools/study_3d/hres_smoke_r2.sh (the plan without FORCE=1; $R2_DOC)"
elif [ -f "$SMOKE/models/smoke_hres.nvfx3" ]; then
  echo "smoke_r2: not started (plan: bash tools/study_3d/hres_smoke_r2.sh; FORCE=1 to run; $R2_DOC, the reason in §8)"
fi
