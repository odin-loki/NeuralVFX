#!/bin/bash
# Study Q2 (docs/STUDY_Q2.md §4.4, §4.8; §0 "to resume"): after one effect's validation, the rule's choice is frozen and
# then tested once. By hand, never from resume.sh. Usage:
#   tools/study_q2/after_val.sh <effect> [threads]
# 1. The choice: the "<effect>,chosen" row of q2_val_choice.csv (its last field; "none" ends the effect as a null).
# 2. The freeze: the chosen file copied to $Q/models/<effect>.nvfx (read-only), its bytes and SHA-256 appended to
#    results/experiments/q2_fix/q2_frozen.csv with the binary's SHA-256, committed and pushed (Q2_NO_PUSH=1 freezes
#    without committing; the test then waits, §4.6: the frozen row is pushed before the test runs).
# 3. The test, once: `nvfx_ordered --study q2 test`, which refuses any file whose hash is not frozen for the effect and
#    writes $Q/test_<effect>.done ("done <sha> met|not met"), q2_test_*.csv and the decision row.
# Then the ship steps, if met, are printed (§0: links, models/2d, SHA256SUMS, frozen rows, videos, the write-up).
. "$(dirname "$0")/common.sh"
e=${1:?effect}
T=${2:-$T}
cd "$REPO" || exit 2
[ -f "$Q/jobs/val_$e.done" ] || grep -q "^$e,chosen," "$R/q2_val_choice.csv" || { log "after_val $e: its validation is not done ($Q/jobs/val_$e.done, or a chosen row in $R/q2_val_choice.csv)"; exit 2; }
[ -f "$Q/test_$e.done" ] && { log "after_val $e: its test has run: $(cat "$Q/test_$e.done")"; exit 0; }
[ -x "$B" ] || { log "after_val $e: no tool at $B (bootstrap.sh)"; exit 2; }

# 1. the choice
row=$(grep "^$e,chosen," "$R/q2_val_choice.csv" | tail -1)
name=${row##*,}
if [ -z "$name" ] || [ "$name" = none ]; then
  echo "done $(date -u +%FT%TZ) none: no candidate met the rule on validation" > "$Q/jobs/after_val_$e.done"
  log "after_val $e: no candidate met the rule (a null, §4.3); nothing frozen, no test. Write it up (§0, §1, §4.7, §8)."
  exit 0
fi
val_sha=$(grep "^$e,$name," "$R/q2_val_choice.csv" | tail -1 | awk -F, '{ print $NF }')
src=$Q/cand/$e/$name.nvfx
[ -f "$src" ] || { log "after_val $e: the chosen candidate's file $src is missing"; exit 2; }
[ "$(sha_of "$src")" = "$val_sha" ] || { log "after_val $e: $src has SHA-256 $(sha_of "$src"), but validation's $name row says $val_sha"; exit 2; }
cur=$(current_file "$e") || exit 2

# 2. the freeze
frozen=$Q/models/$e.nvfx
mkdir -p "$Q/models"
if [ -f "$frozen" ]; then
  [ "$(sha_of "$frozen")" = "$val_sha" ] || { log "after_val $e: $frozen exists with another hash ($(sha_of "$frozen")); not overwritten"; exit 2; }
else
  cp "$src" "$frozen" && chmod a-w "$frozen"
fi
bytes=$(stat -c %s "$frozen")
if ! grep -q "^$e,$name,q2_fix/models/$e.nvfx,$bytes,$val_sha," "$R/q2_frozen.csv"; then
  grep -q "^$e," "$R/q2_frozen.csv" && { log "after_val $e: $R/q2_frozen.csv already has another row for $e; not changed"; exit 2; }
  echo "$e,$name,q2_fix/models/$e.nvfx,$bytes,$val_sha,nvfx_ordered_q2 $(sha_of "$B")" >> "$R/q2_frozen.csv"
  log "after_val $e: frozen $name ($bytes bytes, $val_sha) in $R/q2_frozen.csv"
fi
if [ -n "$(git -C "$REPO" status --porcelain -- results/experiments/q2_fix/q2_frozen.csv)" ]; then
  if [ "${Q2_NO_PUSH:-0}" = 1 ]; then log "after_val $e: Q2_NO_PUSH=1: commit and push q2_frozen.csv by hand, then run this again for the test"; exit 0; fi
  git -C "$REPO" add results/experiments/q2_fix/q2_frozen.csv &&
    git -C "$REPO" commit -q -m "Study Q2: the $e's chosen candidate ($name) frozen before its test (SHA-256 ${val_sha:0:8})" -- results/experiments/q2_fix/q2_frozen.csv &&
    git -C "$REPO" push -q || { log "after_val $e: the frozen row is not committed and pushed; the test does not start (§4.6)"; exit 3; }
  log "after_val $e: frozen row committed and pushed ($(git -C "$REPO" rev-parse --short HEAD))"
fi

# 3. the test, once
check_disk
log "test $e: current $cur; candidate $name=$frozen; $T threads; tool $(sha_of "$B")"
nice -n 10 "$B" --study q2 --root "$D" --threads "$T" --results "$R" --rule "$RULE" \
  test --effect "$e" --current "$cur" --cand "$name=$frozen" --frozen "$R/q2_frozen.csv" 2>&1 | tee -a "$Q/logs/test_$e.log"
rc=${PIPESTATUS[0]}
[ -f "$Q/test_$e.done" ] || { log "test $e: the tool exited $rc and left no marker"; exit $rc; }
log "test $e: $(cat "$Q/test_$e.done")"
log "next: commit and push $R (q2_test_*.csv; only that folder)."
if grep -qE '^done [0-9a-f]+ met$' "$Q/test_$e.done"; then
  cat <<EOF
$e is replaced (§4.4, §0): by hand, in this order
  - the shared links: $D/library/models/$e.nvfx and the scene folder's (models/layout.txt) -> $frozen (old targets logged in $Q/logs/links_before.txt)
  - cp $frozen $REPO/models/2d/$e.nvfx; regenerate models/SHA256SUMS; models/restore.sh against the data root (0 different)
  - the frozen rows: a ${e}_q2 row in the round's fx_*_frozen.csv and the library's frozen_*.csv (float, --isa avx2); their ctests
  - the videos that show it (at most 5 MB each) and their rows in videos/README.md; the figure docs/figures/q2_${e}_$name.png
  - the write-up: STUDY_Q2.md §0, §1, §4.7, §4.8, §5, §7, §8; REPORT.md §23; TODO.md; the full checks after merging main; push
EOF
else
  log "$e keeps its current model (the test did not meet the rule); write it up (§0, §1, §4.8, §8)."
fi
