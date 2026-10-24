#!/bin/bash
# Study Q2 (docs/STUDY_Q2_RESTORE.md): copy the container's q2_fix onto this machine's data root so that the git scripts
# (tools/study_q2/) can run the three validations here. Copies cand/ (the candidates, which exist nowhere else) and,
# with Q2_RESTORE_VT=1, vt/ (the statistics caches: optional, they only save recomputing). Nothing else is copied: the
# tables are in git, the tool is built here, the shipped files are git's models/2d. A DRY RUN unless APPLY=1. Usage:
#   RSYNC_SRC=root@HOST:/root/nvfx-data/q2_fix NEURALVFX_DATA=/path/to/nvfx-data tools/study_q2/restore_from.sh
#   RSYNC_SRC=... APPLY=1 tools/study_q2/restore_from.sh            # really copy
#   RSYNC_SRC=... APPLY=1 Q2_RESTORE_VT=1 tools/study_q2/restore_from.sh
#   RSYNC_SRC=/mnt/backup/q2_fix ...                                # a local folder works too (rsync, no ssh)
#   RSYNC_OPTS="ssh -p 2222 -i ~/.ssh/container"                   # the remote shell (rsync -e): the ssh port, a key
. "$(dirname "$0")/common.sh"
SRC=${RSYNC_SRC:?"set RSYNC_SRC to the container's q2_fix, e.g. root@HOST:/root/nvfx-data/q2_fix"}
SRC=${SRC%/}
APPLY=${APPLY:-0}
command -v rsync > /dev/null || { echo "restore_from: rsync is not installed (scp by hand: docs/STUDY_Q2_RESTORE.md §2)"; exit 2; }
mkdir -p "$Q"
what="cand"
[ "${Q2_RESTORE_VT:-0}" = 1 ] && what="cand vt"
[ "$APPLY" = 1 ] && mode="copying" || mode="DRY RUN (set APPLY=1 to copy)"
echo "restore_from: $SRC -> $Q; $what; $mode"
rc=0
for d in $what; do
  # -a keeps times and modes; --partial resumes a cut copy; --exclude '*.tmp' leaves out a half-written file (resume.sh
  # would delete it anyway). No --delete: nothing of this machine's is removed.
  flags=(-a --partial --info=stats1,progress2 --exclude='*.tmp')
  [ "$APPLY" = 1 ] || flags+=(--dry-run --itemize-changes)
  [ -n "${RSYNC_OPTS:-}" ] && flags+=(-e "$RSYNC_OPTS")  # RSYNC_OPTS is the remote shell, e.g. "ssh -p 2222 -i KEY"
  mkdir -p "$Q/$d"
  rsync "${flags[@]}" "$SRC/$d/" "$Q/$d/" || { echo "restore_from: rsync of $d failed"; rc=1; }
done
[ $rc -ne 0 ] && exit $rc
if [ "$APPLY" = 1 ]; then
  echo "restore_from: the candidates now here (SHA-256 against q2_val_choice.csv's rows once those exist):"
  for e in $EFFECTS; do
    for n in $(candidates "$e"); do
      f=$Q/cand/$e/$n.nvfx
      if [ -f "$f" ]; then echo "  $e $n $(sha_of "$f") $(stat -c %s "$f") bytes"; else echo "  $e $n MISSING"; fi
    done
  done
  echo "next: NEURALVFX_DATA=$D tools/study_q2/bootstrap.sh (then resume.sh; docs/STUDY_Q2_RESTORE.md §3)"
fi
