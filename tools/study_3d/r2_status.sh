#!/usr/bin/env bash
# Study R2: where the study stands, in a few lines (data under NEURALVFX_DATA/r2_smoke3d; hres_status.sh's form).
#   bash tools/study_3d/r2_status.sh
set -euo pipefail
REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
. "$REPO/tools/study_3d/r2_env.sh"
cd "$REPO"
echo "=== r2 status $(date -Iseconds) ==="
echo "data: $D$([ -d "$D" ] || echo ' (MISSING: r2_bootstrap.sh)')"
if [ -n "${NVFX_3D:-}" ]; then echo "tool: $NVFX_3D (NVFX_3D)"
elif [ -L "$D/bin/nvfx_3d" ] && [ -x "$D/bin/nvfx_3d" ]; then echo "tool: $(readlink -f "$D/bin/nvfx_3d")"
elif [ -L "$D/bin/nvfx_3d" ]; then echo "tool: link BROKEN ($(readlink "$D/bin/nvfx_3d")); REPIN=1 bash tools/study_3d/r2_bootstrap.sh"
else echo "tool: not pinned (r2_bootstrap.sh)"; fi

# 1. the runs and training
nl=$(r2_runs_count long); ns=$(r2_runs_count short)
echo "runs: $nl long, $ns short of 24 each"
if [ -f "$MARK/candidates_done" ]; then echo "train: DONE; candidates C2 C3 C4 ($MARK/candidates_done)"
elif r2_train_done; then echo "train: DONE at iteration $TOTAL_IT; candidates $(r2_nfiles "$D/models/"C?.nvfx3) of 3 (r2_resume.sh finishes them)"
else
  last=$(r2_train_last)
  echo "train: ${last:-not started}"
  mapfile -t itl < <(grep '^it ' "$D/logs/train_r2.log" 2>/dev/null | awk '!seen[$0]++' | tail -2 || true)
  if [ "${#itl[@]}" -ge 2 ] && [[ "${itl[1]}" =~ ^it[[:space:]]+([0-9]+).*\(([0-9]+)[[:space:]]s, ]]; then
    it="${BASH_REMATCH[1]}"; sec2="${BASH_REMATCH[2]}"
    if [[ "${itl[0]}" =~ ^it[[:space:]]+([0-9]+).*\(([0-9]+)[[:space:]]s, ]]; then
      it0="${BASH_REMATCH[1]}"; sec0="${BASH_REMATCH[2]}"; dit=$((it - it0))
      if [ "$dit" -gt 0 ] && [ "$it" -lt "$TOTAL_IT" ] && [ "$sec2" -gt "$sec0" ]; then
        spi=$(( (sec2 - sec0) / dit )); rem=$(( (TOTAL_IT - it) * spi ))
        echo "train: ~$(( rem / 3600 )) h $(( (rem % 3600) / 60 )) m remaining (~${spi} s/iter from the log's cadence)"
      fi
    fi
  fi
  if r2_running "$R2_PAT_TRAIN"; then echo "train: trainer RUNNING ($(pgrep -af -- "$R2_PAT_TRAIN" | head -1 | cut -c1-120))"
  elif r2_running "$R2_PAT_RECORD"; then echo "train: recording RUNNING"
  elif r2_running "$R2_PAT_TRAIN_SH"; then echo "train: r2_train.sh RUNNING (between steps)"
  else echo "train: NOT running (start: bash tools/study_3d/r2_resume.sh)"; fi
  [ -f "$D/logs/train_r2.log" ] && echo "train: log updated $(stat -c %y "$D/logs/train_r2.log" 2>/dev/null | cut -c1-19)"
fi

# 2. the plume measurement (§1.6)
np=$(r2_plume_count)
if [ -f "$MARK/plume_done" ]; then echo "plume: DONE ($np of $PLUME_RUNS runs; $R/plume_fit.md)"
elif r2_running "$R2_PAT_PLUME_SH"; then echo "plume: RUNNING ($np of $PLUME_RUNS runs)"
else echo "plume: $np of $PLUME_RUNS runs, not running (r2_resume.sh)"; fi

# 3. the rule
if [ -f "$MARK/rule_committed" ]; then echo "rule: committed"
elif r2_rule_in_git; then echo "rule: in git; marker missing (r2_bootstrap.sh or r2_resume.sh makes it)"
else echo "rule: NOT committed ($R/margins.csv, edge_bounds.csv)"; fi

# 4. validation, the choice and the freeze
if [ -f "$MARK/frozen" ]; then
  echo "validate: DONE; chosen $(cat "$D/models/chosen.txt" 2>/dev/null || echo '?') ($R/choice_val.md; frozen in $R/frozen.csv)"
  [ -f "$MARK/frozen_committed" ] && echo "validate: frozen_committed" || echo "validate: waiting for the hand step (commit and push $R/choice_val.md frozen.csv edge_val.csv edgecheck_val.md compare_val_*; touch $MARK/frozen_committed; r2_resume.sh)"
elif [ -f "$MARK/val_done" ]; then echo "validate: runs DONE, choice not frozen (r2_resume.sh)"
elif r2_running "$R2_PAT_VAL_SH"; then
  echo "validate: RUNNING ($(pgrep -af -- "$R2_PAT_ANY_TOOL" | grep -c -- '--split val' || true) tool processes; tail $D/logs/job_val.log)"
elif [ -f "$MARK/candidates_done" ] && [ -f "$MARK/rule_committed" ]; then echo "validate: not started or cut (start: bash tools/study_3d/r2_resume.sh)"
else echo "validate: waits for the candidates and the committed rule"; fi
if [ -d "$D/eval/val" ]; then
  echo "validate: truth $(r2_nfiles "$D/eval/val/truth/"*.bin) of 18; edge $(for m in C2 C3 C4; do printf '%s:%s ' "$m" "$(r2_edge_rows "$D/results/edge_val_$m.csv")"; done)of $EDGE_ROWS each"
fi

# 5. the test
if [ -f "$MARK/test_done" ]; then
  echo "test: DONE ($(cat "$MARK/test_done"))"
  grep -E '^\*\*The verdict' "$R/verdict_test.md" 2>/dev/null | tail -1 || echo "test: $R/verdict_test.md has no verdict line"
elif r2_running "$R2_PAT_TEST_SH"; then
  echo "test: RUNNING ($(pgrep -af -- "$R2_PAT_ANY_TOOL" | grep -c -- '--split test' || true) tool processes; tail $D/logs/job_test.log)"
  [ -d "$D/eval/test" ] && echo "test: truth $(r2_nfiles "$D/eval/test/truth/"*.bin) of 24; edge $(r2_edge_rows "$D/results/edge_test.csv") of $EDGE_ROWS"
elif [ -f "$MARK/frozen_committed" ]; then echo "test: not started or cut (start: bash tools/study_3d/r2_resume.sh)"
else echo "test: waits (frozen_committed, or AUTO_TEST=1)"; fi

echo "markers: $(ls "$MARK" 2>/dev/null | tr '\n' ' ')"
[ -d "$R" ] && echo "tables: $R holds $(find "$R" -maxdepth 1 -type f | wc -l) files"
if [ -d "$D" ]; then
  echo "disk: $(( $(df --output=avail -k "$D" | tail -1) / 1024 )) MB free; r2_smoke3d holds $(( $(du -sk "$D" 2>/dev/null | cut -f1) / 1024 )) MB"
fi
pgrep -af -- "$R2_PAT_ANY_TOOL" 2>/dev/null | cut -c1-140 || echo "no nvfx_3d process on $D"
