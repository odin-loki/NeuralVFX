#!/usr/bin/env bash
# Study R2 §4: the test, run once on the frozen model (STUDY_R2.md §2.9). Portable form of tools/study_r2/test.sh.
# The chosen model's edge check with the test's seeds (34,730,000 + j), the truth and the real runs per setting (no
# floor, §2.5), the learned, coarse and hold tracking and plays (J at a time), the comparisons and `verdict` ->
# results/3d/r2/verdict_test.md; marker test_done. With test_done it only says where the verdict is.
#
# Usage (repo root; usually started by r2_resume.sh, detached and from a snapshot copy):
#   bash tools/study_3d/r2_test.sh
# Needs the frozen chosen model (markers/frozen, r2_validate.sh) and markers/frozen_committed (by hand: validation's
# files committed and pushed, §0) or AUTO_TEST=1. The tool refuses the learned runs on any file whose SHA-256 is not
# results/3d/r2/frozen.csv's. Resumable: every run already written is skipped. Env (r2_env.sh): J (8), NICE_LEVEL,
# NVFX_3D, NEURALVFX_DATA.
set -euo pipefail

main() {
  local REPO B M F s n rc=0
  REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
  [ -f "$REPO/tools/study_3d/r2_env.sh" ] || { echo "error: repo not found at $REPO (running a copy? set NVFX_REPO)" >&2; return 1; }
  . "$REPO/tools/study_3d/r2_env.sh"
  cd "$REPO"
  r2_dirs
  exec 9>"$D/logs/test.lock"
  if ! flock -n 9; then echo "r2 test: another test holds $D/logs/test.lock"; return 0; fi
  if [ -f "$MARK/test_done" ]; then echo "the test ran once already ($MARK/test_done): $R/verdict_test.md"; return 0; fi
  [ -f "$MARK/val_done" ] && [ -f "$MARK/frozen" ] && r2_frozen_ok || { echo "error: no frozen chosen model (r2_validate.sh first: markers val_done, frozen; models/chosen.nvfx3 and chosen.txt; $R/frozen.csv)"; return 1; }
  [ -f "$MARK/frozen_committed" ] || [ "${AUTO_TEST:-0}" = 1 ] || { echo "the test waits for $MARK/frozen_committed (commit and push validation's files first) or AUTO_TEST=1"; return 0; }
  B=$(r2_bin)
  M="$D/models/chosen.nvfx3"; F="--frozen $R/frozen.csv"
  echo "=== r2 test $(date -Iseconds) B=$B D=$D J=$J chosen=$(cat "$D/models/chosen.txt") pid=$$ ==="
  r2_disk_need 2

  echo "--- the edge check (the test's seeds), the truth and the real runs (8 settings, no floor) ---"
  {
    echo "edge --model $M --label learned --seed-base 34730000 $W --out $D/results/edge_test.csv"
    for s in 0 1 2 3 4 5 6 7; do echo "truth --split test --setting $s"; echo "real --split test --setting $s --no-floor"; done
  } | r2_run_many
  echo "--- learned, coarse and hold: tracking and plays ---"
  {
    for s in 0 1 2 3 4 5 6 7; do
      echo "track --split test --method learned --model $M $F --setting $s"; echo "play --split test --method learned --model $M $F --setting $s"
      echo "track --split test --method coarse --model $M --setting $s"; echo "play --split test --method coarse --model $M --setting $s"
      echo "track --split test --method hold --model $M --setting $s"
    done
  } | r2_run_many
  n=$(r2_edge_rows "$D/results/edge_test.csv")
  [ "$n" -ge "$EDGE_ROWS" ] || { echo "error: $D/results/edge_test.csv holds $n rows of $EDGE_ROWS"; return 1; }
  cp -f "$D/results/edge_test.csv" "$R/edge_test.csv"
  echo "--- the edge check, the comparisons and the verdict ---"
  "$B" edgecheck $H --csv "$R/edge_test.csv" --bounds "$R/edge_bounds.csv" >"$R/edgecheck_test.md"
  "$B" compare $H --split test --a learned --b coarse --results "$R" >/dev/null
  "$B" compare $H --split test --a learned --b hold --results "$R" >/dev/null
  "$B" compare $H --split test --a hold --b coarse --results "$R" >/dev/null
  "$B" verdict $H --split test --a learned --b coarse --margins "$R/margins.csv" --bounds "$R/edge_bounds.csv" \
    --edge-val "$R/edge_val.csv" --val-label "$(cat "$D/models/chosen.txt")" --edge-test "$R/edge_test.csv" --out "$R/verdict_test.md" || rc=$?
  case $rc in
    0|3) echo "done $(date -Iseconds) $(cat "$D/models/chosen.txt") verdict-exit $rc" >"$MARK/test_done";;
  esac
  case $rc in
    0) echo "TEST_DONE $(date -Iseconds): PASSES ($R/verdict_test.md); ship per STUDY_R2.md §2.9 (models/3d/smoke.nvfx3, layout.txt, SHA256SUMS)";;
    3) echo "TEST_DONE $(date -Iseconds): FAILS ($R/verdict_test.md); study R's demo stays; write up the numbers";;
    *) echo "error: verdict exited $rc (nothing marked done; see $R/verdict_test.md if written)"; return "$rc";;
  esac
  echo "next: commit and push $R (edge_test.csv, edgecheck_test.md, compare_test_*, verdict_test.md); the write-up (§0, §3, §4)"
  flock -u 9
}

main "$@"; exit $?
