#!/usr/bin/env bash
# Study R2 §3: validation, the choice and the freeze (STUDY_R2.md §2.8). Portable form of tools/study_r2/val.sh.
# The truth and the real runs per setting, each candidate's edge check (validation's seeds), tracking and plays (one
# process per method and setting, J at a time), the coarse simulation's and holding the first frame's, the comparisons,
# `choose` -> results/3d/r2/choice_val.md (marker val_done); then the choice copied to models/chosen.nvfx3 (+ chosen.txt)
# and frozen by SHA-256 into results/3d/r2/frozen.csv (marker frozen). The test is r2_test.sh, after the hand step.
#
# Usage (repo root; usually started by r2_resume.sh, detached and from a snapshot copy):
#   bash tools/study_3d/r2_validate.sh
# Needs the rule committed (results/3d/r2/{edge_bounds,margins}.csv and markers/rule_committed) and the candidates
# (markers/candidates_done). Resumable: every run already written is skipped by the tool (truth, real, track rows,
# plays, whole edge runs). Env (r2_env.sh): J (8), CHOSEN (override the parsed choice), FORCE=1 (the analysis steps
# again; the runs themselves are always skipped when written), NICE_LEVEL, NVFX_3D, NEURALVFX_DATA.
set -euo pipefail

main() {
  local REPO B m s n CHOSEN_
  REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
  [ -f "$REPO/tools/study_3d/r2_env.sh" ] || { echo "error: repo not found at $REPO (running a copy? set NVFX_REPO)" >&2; return 1; }
  . "$REPO/tools/study_3d/r2_env.sh"
  cd "$REPO"
  r2_dirs
  exec 8>"$D/logs/val.lock"
  if ! flock -n 8; then echo "r2 validate: another validation holds $D/logs/val.lock"; return 0; fi
  [ -f "$MARK/candidates_done" ] || { echo "error: no candidates yet ($MARK/candidates_done); r2_train.sh first"; return 1; }
  [ -f "$MARK/rule_committed" ] || { echo "error: the rule is not marked committed ($MARK/rule_committed; r2_bootstrap.sh makes it when git has §2.10's files)"; return 1; }
  for m in "$R/edge_bounds.csv" "$R/margins.csv"; do [ -f "$m" ] || { echo "error: the rule's numbers are missing: $m (STUDY_R2.md §2.10)"; return 1; }; done
  for m in C2 C3 C4; do [ -f "$D/models/$m.nvfx3" ] || { echo "error: missing $D/models/$m.nvfx3"; return 1; }; done
  B=$(r2_bin)
  echo "=== r2 validate $(date -Iseconds) B=$B D=$D J=$J pid=$$ ==="
  r2_disk_need 2

  if [ ! -f "$MARK/val_done" ] || [ "${FORCE:-0}" = 1 ]; then
    echo "--- the truth and the real runs (6 settings) ---"
    for s in 0 1 2 3 4 5; do echo "truth --split val --setting $s"; echo "real --split val --setting $s"; done | r2_run_many
    echo "--- each candidate's edge check (validation's seeds), tracking and plays; the coarse simulation and holding the first frame ---"
    {
      for m in C2 C3 C4; do echo "edge --model $D/models/$m.nvfx3 --label $m $W --out $D/results/edge_val_$m.csv"; done
      for m in C2 C3 C4; do for s in 0 1 2 3 4 5; do
        echo "track --split val --method $m --model $D/models/$m.nvfx3 --setting $s"
        echo "play --split val --method $m --model $D/models/$m.nvfx3 --setting $s"
      done; done
      for s in 0 1 2 3 4 5; do
        echo "track --split val --method coarse --model $D/models/C2.nvfx3 --setting $s"
        echo "play --split val --method coarse --model $D/models/C2.nvfx3 --setting $s"
        echo "track --split val --method hold --model $D/models/C2.nvfx3 --setting $s"
      done
    } | r2_run_many
    for m in C2 C3 C4; do
      n=$(r2_edge_rows "$D/results/edge_val_$m.csv")
      [ "$n" -ge "$EDGE_ROWS" ] || { echo "error: $D/results/edge_val_$m.csv holds $n rows of $EDGE_ROWS (5 winds x 61)"; return 1; }
    done
    r2_combine_edge "$D/results/edge_val.csv" "$D/results/edge_val_C2.csv" "$D/results/edge_val_C3.csv" "$D/results/edge_val_C4.csv"
    cp -f "$D/results/edge_val.csv" "$R/edge_val.csv"
    echo "--- the edge check, the comparisons and the choice ---"
    "$B" edgecheck $H --csv "$R/edge_val.csv" --bounds "$R/edge_bounds.csv" >"$R/edgecheck_val.md"
    for m in C2 C3 C4; do "$B" compare $H --split val --a "$m" --b coarse --results "$R" >/dev/null; done
    "$B" compare $H --split val --a C4 --b C3 --results "$R" >/dev/null
    "$B" compare $H --split val --a C4 --b hold --results "$R" >/dev/null
    "$B" choose $H --split val --cands C2,C3,C4 --edge "$R/edge_val.csv" --bounds "$R/edge_bounds.csv" --margins "$R/margins.csv" >"$R/choice_val.md"
    touch "$MARK/val_done"
    echo "VAL_DONE $(date -Iseconds): $R/choice_val.md"
  else
    echo "validation done ($MARK/val_done); FORCE=1 runs its analysis again"
  fi

  # the choice and the freeze (§2.8: the chosen file is frozen by SHA-256 before the test)
  CHOSEN_=$(r2_chosen)
  [ -n "$CHOSEN_" ] || { echo "error: no 'chosen: C?' line in $R/choice_val.md (or set CHOSEN)"; return 1; }
  [ -f "$D/models/$CHOSEN_.nvfx3" ] || { echo "error: missing $D/models/$CHOSEN_.nvfx3"; return 1; }
  if [ -n "${CHOSEN:-}" ]; then echo "CHOSEN from env: $CHOSEN_"; else echo "CHOSEN parsed from $R/choice_val.md: $CHOSEN_"; fi
  if [ -f "$MARK/frozen" ] && r2_frozen_ok && [ "$(cat "$D/models/chosen.txt")" = "$CHOSEN_" ]; then
    echo "frozen already: $CHOSEN_ as $D/models/chosen.nvfx3 ($R/frozen.csv)"
  else
    if [ -f "$MARK/frozen" ]; then echo "error: $MARK/frozen exists but the chosen model or its sha changed; remove the marker by hand if that is meant"; return 1; fi
    cp -f "$D/models/$CHOSEN_.nvfx3" "$D/models/chosen.nvfx3"
    echo "$CHOSEN_" >"$D/models/chosen.txt"
    r2_freeze_once "$D/models/chosen.nvfx3"
    touch "$MARK/frozen"
    echo "FROZEN $(date -Iseconds): $CHOSEN_ -> $D/models/chosen.nvfx3, sha in $R/frozen.csv"
  fi
  if [ ! -f "$MARK/frozen_committed" ]; then
    cat <<EOF
Next (by hand, STUDY_R2.md §0): commit and push validation's files, then let the test run:
  git add $R/choice_val.md $R/frozen.csv $R/edge_val.csv $R/edgecheck_val.md $R/compare_val_*
  git commit -m "Study R2: validation, the choice ($CHOSEN_) and the freeze" && git push
  touch $MARK/frozen_committed && bash tools/study_3d/r2_resume.sh     # (or AUTO_TEST=1 bash tools/study_3d/r2_test.sh)
EOF
  fi
  flock -u 8
}

main "$@"; exit $?
