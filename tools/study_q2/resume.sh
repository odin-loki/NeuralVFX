#!/bin/bash
# Study Q2 (docs/STUDY_Q2.md §0): after a container restart, or at any time: the cleaning, then every validation that
# is not done is started (a job already running is left alone; each skips what is done). It does NOT start
# after_val.sh (the freeze and the one test are by hand, §0). Usage:
#   NEURALVFX_DATA=/root/nvfx-data tools/study_q2/resume.sh
# The cleaning: the statistics caches ($Q/vt) are written whole and renamed, but one written in the moments before a
# restart may not have reached the disk whole, so every cache written in the ten minutes before this boot is deleted
# (logged to $Q/logs/resume_deleted.txt; a step that runs again recomputes it, the same numbers); half-written files
# (*.tmp); a stale git lock; and a test cut off before its result (its marker says "started", not "done") is set aside
# so that the test can run (it has made no decision; its caches were deleted with the rest).
. "$(dirname "$0")/common.sh"
[ -d "$Q" ] || { echo "resume: $Q does not exist (tools/study_q2/bootstrap.sh first)"; exit 2; }
mkdir -p "$Q/logs" "$Q/jobs"
cd "$Q" || exit 2

boot=$(uptime -s 2>/dev/null)
if [ -n "$boot" ] && [ -d vt ]; then
  since=$(date -d "$boot -10 minutes" +%FT%T)
  n=0
  while IFS= read -r f; do
    echo "$(date -u +%FT%TZ) boot $boot: $f" >> logs/resume_deleted.txt
    rm -f -- "$f"; n=$((n + 1))
  done < <(find vt -type f \( -newermt "$since" ! -newermt "$boot" \) 2>/dev/null)
  log "resume: boot $boot; $n cache files written in its last ten minutes deleted (logs/resume_deleted.txt)"
fi
find vt cand -name '*.tmp' -type f -print -delete 2>/dev/null | sed 's/^/resume: half-written, deleted: /'
if [ -f "$REPO/.git/index.lock" ] && ! pgrep -x git > /dev/null; then rm -f "$REPO/.git/index.lock"; log "resume: stale git lock removed"; fi
for e in $EFFECTS; do
  m=test_$e.done
  if [ -f "$m" ] && grep -q '^started' "$m" && ! grep -q '^done' "$m"; then
    mv "$m" "$m.cut.$(date -u +%Y%m%dT%H%M%SZ)"; log "resume: the test of $e was cut off before its result; its marker set aside (run after_val.sh $e again)"
  fi
done

# The validations (§0): each in its own process, 2 threads (Q2_THREADS), skipping what is done.
for e in $EFFECTS; do
  if [ -f "jobs/val_$e.done" ]; then log "resume: val $e done"
  elif [ -f "test_$e.done" ]; then log "resume: $e tested ($(cat "test_$e.done"))"
  elif running "^(nice -n 10 )?$B --study q2 .* val --effect $e "; then log "resume: val $e running"
  else launch "logs/val_$e.log" bash "$REPO/tools/study_q2/val.sh" "$e"; fi
done
log "resume: when jobs/val_<effect>.done appears: tools/study_q2/after_val.sh <effect> (the freeze and the one test, by hand)"
ls jobs
