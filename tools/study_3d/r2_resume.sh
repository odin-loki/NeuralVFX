#!/usr/bin/env bash
# Study R2 (docs/STUDY_R2.md §0): restart everything that is not done, after a restart or at any time. A job already
# running is left alone; every job skips what it has already written. Run it again whenever in doubt:
#   bash tools/study_3d/r2_resume.sh
# Portable form of the container's jobs/resume.sh (tools/study_r2/resume.sh). Stages, each behind the one before,
# the hand steps marked:
#   1. the training runs and training, then the candidates C2..C4 (r2_train.sh; markers/candidates_done)
#   2. the plume measurement's 18 teacher runs (§1.6) and their fits (in parallel with 1; markers/plume_done)
#   3. the rule's numbers (§2.10): committed in results/3d/r2 already; if the marker rule_committed is missing and git
#      has them, it is made (r2_bootstrap.sh does the same); otherwise say what is missing (by hand)
#   4. validation, the choice and the freeze (r2_validate.sh, once rule_committed and candidates_done; markers val_done,
#      frozen); then by hand: commit and push validation's files, touch markers/frozen_committed
#   5. the test, once (r2_test.sh, once frozen_committed; marker test_done)
# Jobs start detached (setsid nohup) from snapshot copies under $D/logs (r2_env.sh: r2_launch), logging to
# $D/logs/job_<name>.log. Env (r2_env.sh): NEURALVFX_DATA, NVFX_3D, BUILD, T, J, NICE_LEVEL; AUTO_TEST=1 lets the test
# start without the frozen_committed hand step.
#
# The cleaning first: outputs the tool appends line by line (tracking rows, edge runs) that a kill could have cut are
# trimmed to whole rows and whole 61-row edge runs; half-written *.tmp files are deleted. Everything else (a run's
# rows, units, profiles, checkpoints) is written whole through a temporary file renamed, so nothing partial is used.
set -euo pipefail

# --plume-worker: the detached job for stage 2 (started by main below; not for the hand).
plume_worker() {
  local B n
  exec 7>"$D/logs/plume.lock"
  if ! flock -n 7; then echo "r2 plume: another plume job holds $D/logs/plume.lock"; return 0; fi
  B=$(r2_bin)
  echo "=== r2 plume $(date -Iseconds) B=$B D=$D J=$J pid=$$ ==="
  r2_disk_need 1
  r2_plume_tasks | r2_run_many
  n=$(r2_plume_count)
  [ "$n" -ge "$PLUME_RUNS" ] || { echo "error: $n of $PLUME_RUNS plume runs written"; return 1; }
  echo "--- the fits (§1.6, §7): results/3d/r2/plume_fit.md, plume_profiles.csv ---"
  nice -n "$NICE" "$B" plumefit $H --configs v_i20,v_i50,v_i80,v_t00,v_t100 --strengths v_i20,v_i50,v_i80 \
    --momentum v_i20,v_i50,v_i80,v_t00,v_t100 --budget v_i50,v_t100 --fit-lo 24 --fit-hi 92 --out "$R"
  touch "$MARK/plume_done"
  echo "PLUME_DONE $(date -Iseconds): $R/plume_fit.md (commit it with the write-up of §1.6)"
  flock -u 7
}

clean_cut_outputs() {
  local f
  for f in "$D"/eval/*/track/*.csv; do
    [ -f "$f" ] || continue
    awk -F, 'NF >= 26' "$f" >"$f.clean" && mv -f "$f.clean" "$f"
  done
  for f in "$D"/results/edge_*.csv; do
    [ -f "$f" ] || continue
    awk -F, 'NR == FNR { if (FNR > 1 && NF >= 10) n[$1 "," $2]++; next } FNR == 1 || (NF >= 10 && n[$1 "," $2] >= 61)' "$f" "$f" >"$f.clean" && mv -f "$f.clean" "$f"
  done
  find "$D/results" "$D/eval" "$D/plume" "$D/models" "$D/ckpt" -name '*.tmp' -type f -print -delete 2>/dev/null | sed 's/^/resume: half-written, deleted: /' || true
  find "$D/results" -name '*.clean' -type f -delete 2>/dev/null || true
}

main() {
  local REPO nl ns np
  REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
  [ -f "$REPO/tools/study_3d/r2_env.sh" ] || { echo "error: repo not found at $REPO (running a copy? set NVFX_REPO)" >&2; return 1; }
  . "$REPO/tools/study_3d/r2_env.sh"
  cd "$REPO"
  if [ "${1:-}" = --plume-worker ]; then plume_worker; return $?; fi
  [ -d "$D/runs" ] || [ -d "$D/markers" ] || { echo "resume: $D is not set up (bash tools/study_3d/r2_bootstrap.sh first)"; return 2; }
  r2_dirs
  echo "=== r2 resume $(date -Iseconds) D=$D repo=$REPO ==="
  r2_disk_need 0.6
  clean_cut_outputs

  # 1. the training runs, training and the candidates
  nl=$(r2_runs_count long); ns=$(r2_runs_count short)
  if [ ! -f "$MARK/candidates_done" ]; then
    if r2_running "$R2_PAT_TRAIN_SH" || r2_running "$R2_PAT_TRAIN" || r2_running "$R2_PAT_RECORD"; then
      echo "training: running (runs $nl long, $ns short of 24 each; last: $(r2_train_last))"
    else
      echo "training: not done (runs $nl long, $ns short of 24 each; last: $(r2_train_last))"
      r2_launch r2_train "$D/logs/job_train.log"
    fi
  else
    echo "training: done ($MARK/candidates_done; models/C2.nvfx3 C3 C4)"
  fi

  # 2. the plume measurement's teacher runs (§1.6) and the fits
  np=$(r2_plume_count)
  if [ -f "$MARK/plume_done" ]; then echo "plume (§1.6): done ($np of $PLUME_RUNS runs; $R/plume_fit.md)"
  elif r2_running "$R2_PAT_PLUME_SH"; then echo "plume (§1.6): running ($np of $PLUME_RUNS runs)"
  else
    echo "plume (§1.6): $np of $PLUME_RUNS runs; starting the rest and the fits"
    r2_launch r2_resume "$D/logs/job_plume.log" --plume-worker
  fi

  # 3. the rule's numbers (§2.10), committed before validation
  if [ ! -f "$MARK/rule_committed" ]; then
    if r2_rule_in_git; then touch "$MARK/rule_committed"; echo "rule: $R/margins.csv and edge_bounds.csv are committed; $MARK/rule_committed made"
    else echo "rule: not marked committed ($MARK/rule_committed): commit and push $R/margins.csv and edge_bounds.csv (§2.10), then touch the marker; validation waits"; fi
  fi

  # 4. validation, the choice and the freeze
  if [ -f "$MARK/rule_committed" ] && [ -f "$MARK/candidates_done" ]; then
    if [ -f "$MARK/frozen" ]; then echo "validation: done ($R/choice_val.md; chosen $(cat "$D/models/chosen.txt" 2>/dev/null || echo '?') frozen)"
    elif r2_running "$R2_PAT_VAL_SH"; then echo "validation: running (log $D/logs/job_val.log)"
    else r2_launch r2_validate "$D/logs/job_val.log"; fi
  elif [ -f "$MARK/candidates_done" ]; then echo "validation: waits for $MARK/rule_committed"
  else echo "validation: waits for the candidates"; fi

  # 5. the test, once
  if [ -f "$MARK/test_done" ]; then echo "test: done ($(cat "$MARK/test_done"); $R/verdict_test.md)"
  elif [ -f "$MARK/frozen" ]; then
    if [ -f "$MARK/frozen_committed" ] || [ "${AUTO_TEST:-0}" = 1 ]; then
      if r2_running "$R2_PAT_TEST_SH"; then echo "test: running (log $D/logs/job_test.log)"
      else r2_launch r2_test "$D/logs/job_test.log"; fi
    else
      echo "test: waits for $MARK/frozen_committed (by hand: commit and push $R/choice_val.md, frozen.csv, edge_val.csv, edgecheck_val.md, compare_val_*; then touch it and run this again; or AUTO_TEST=1)"
    fi
  else echo "test: waits for the frozen chosen model"; fi
  echo "markers: $(ls "$MARK" 2>/dev/null | tr '\n' ' ')"
  echo "status: bash tools/study_3d/r2_status.sh"
}

main "$@"; exit $?
