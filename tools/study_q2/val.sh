#!/bin/bash
# Study Q2 (docs/STUDY_Q2.md §4.4, §4.7): one effect's validation by the committed rule (q2_rule.csv): the shipped file
# against every candidate of §4.1 on study G's validation settings extended to the effect's N, study Q2's seeds (valq2).
# Resumable: the tool caches each setting's statistics under $Q/vt, keyed by the seeds and each file's SHA-256, so a run
# started again recomputes only what is missing (the tracking runs are not cached). Writes q2_val_*.csv and the choice
# row to $R, then $Q/jobs/val_<effect>.done. Starts no freeze and no test (§0). Usage:
#   tools/study_q2/val.sh geyser|heavygas|mushroom
. "$(dirname "$0")/common.sh"
e=${1:?effect}
done_marker=$Q/jobs/val_$e.done
[ -f "$done_marker" ] && { log "val $e: done ($(cat "$done_marker"))"; exit 0; }
[ -x "$B" ] || { log "val $e: no tool at $B (bootstrap.sh)"; exit 2; }
grep -q "^$e," "$RULE" || { log "val $e: no row in $RULE (the rule's numbers are committed before validation, §4.6)"; exit 2; }
cur=$(current_file "$e") || exit 2
cands=$(cand_list "$e") || exit 2
check_disk
cd "$REPO" || exit 2
log "val $e: current $cur; candidates $cands; $T threads; tool $(sha_of "$B")"
nice -n 10 "$B" --study q2 --root "$D" --threads "$T" --results "$R" --rule "$RULE" \
  val --effect "$e" --split valq2 --current "$cur" --cand "$cands" --prefix q2_val
rc=$?
if [ $rc -ne 0 ]; then log "val $e: the tool exited $rc; nothing marked done"; exit $rc; fi
grep -q "^$e,chosen," "$R/q2_val_choice.csv" || { log "val $e: no choice row written to $R/q2_val_choice.csv"; exit 3; }
echo "done $(date -u +%FT%TZ) $(grep "^$e,chosen," "$R/q2_val_choice.csv")" > "$done_marker"
log "val $e: $(cat "$done_marker")"
log "next: commit and push $R (q2_val_*.csv; only that folder), then by hand: tools/study_q2/after_val.sh $e"
