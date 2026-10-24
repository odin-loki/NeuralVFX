#!/bin/bash
# Study Q2 (docs/STUDY_Q2.md §0, docs/STUDY_Q2_RESTORE.md §3): make the data root's q2_fix ready on a machine that has
# the repository and a build but no q2_fix yet (the owner's PC: its ~/nvfx-data holds hres_3d and n2_3d, the 3D studies',
# and nothing of study Q2's), or check one that has: the folders, the tool copied to $Q/bin/nvfx_ordered_q2 (its SHA-256
# against §4.6's), the shipped files (the data root's or git's models/2d, by their §4.1 hashes), the candidates present,
# and what each effect still needs. Ends with a checklist. Makes no numbers; touches nothing outside $Q. Usage:
#   NEURALVFX_DATA=$HOME/nvfx-data tools/study_q2/bootstrap.sh              # make and check
#   NEURALVFX_DATA=$HOME/nvfx-data Q2_DRY=1 tools/study_q2/bootstrap.sh     # dry run: check only, write nothing
# Exit 0 when every line of the checklist is ok (resume.sh can start), 1 when something is still to do.
. "$(dirname "$0")/common.sh"
DRY=${Q2_DRY:-0}
todo=()                                   # the checklist's open items
note() { echo "  - $*"; todo+=("$*"); }   # one open item, said now and again at the end
hdr() { echo; echo "## $*"; }

hdr "the data root"
echo "data root $D; q2_fix $Q; repository $REPO; tables $R$([ "$DRY" = 1 ] && echo '; DRY RUN (Q2_DRY=1): nothing is written')"
if [ -d "$D" ]; then
  others=$(cd "$D" && ls -d */ 2>/dev/null | sed 's#/$##' | grep -vx q2_fix | tr '\n' ' ')
  [ -n "$others" ] && echo "data root: also holds ${others% } (other studies'; untouched; q2_fix sits next to them)"
  free_mb=$(df -m "$D" | awk 'NR == 2 { print $4 }')
  if [ "${free_mb:-0}" -lt 2048 ]; then note "disk: $D has only ${free_mb} MB free; the study wants about 2 GB (q2_fix stays under 1.5 GB, check_disk pauses below 1 GB free on /)"
  else echo "disk: ${free_mb} MB free at $D (about 2 GB is enough; q2_fix stays under 1.5 GB)"; fi
  case $D in /mnt/[a-z]/*) note "data root: $D is a Windows drive mounted in WSL; a Linux path (e.g. \$HOME/nvfx-data) is faster and locks files properly" ;; esac
else
  note "data root: $D does not exist (set NEURALVFX_DATA to a folder with about 2 GB free, e.g. \$HOME/nvfx-data; the default is the container's /root/nvfx-data)"
fi
for t in sha256sum pgrep setsid df du; do command -v "$t" > /dev/null || note "tool: $t is not installed (the scripts are bash on Linux or WSL, not PowerShell)"; done
command -v rsync > /dev/null || echo "rsync: not installed (restore_from.sh needs it; scp by hand works too, docs/STUDY_Q2_RESTORE.md §2)"

hdr "the folders"
if [ -d "$Q" ]; then echo "q2_fix: $Q exists ($(du -sm "$Q" 2>/dev/null | cut -f1) MB)"
elif [ "$DRY" = 1 ]; then echo "q2_fix: $Q is absent; would be made (jobs logs bin models vt res scratch cand/<effect>)"
else echo "q2_fix: $Q is absent; making it"; fi
if [ "$DRY" != 1 ]; then
  if ! { mkdir -p "$Q" && mkdir -p "$Q/jobs" "$Q/logs" "$Q/bin" "$Q/models" "$Q/vt" "$Q/res" "$Q/scratch"; }; then
    echo "bootstrap: cannot make $Q (q2_fix under the data root $D); mkdir's error is above."
    if [ ! -d "$D" ]; then echo "  the data root $D does not exist: mkdir -p \"$D\" (a Linux path with about 2 GB free, e.g. \$HOME/nvfx-data), or set NEURALVFX_DATA to one that does"
    elif [ ! -w "$D" ]; then echo "  $D is not writable by $(id -un): chown or chmod it, or set NEURALVFX_DATA to a folder of yours"
    else echo "  $D exists and is writable; see the error above (a file in the way of $Q, a full or read-only disk: df -h \"$D\")"; fi
    echo "  nothing is written outside \$NEURALVFX_DATA/q2_fix; Q2_DRY=1 runs the checks without making anything"
    exit 2
  fi
  for e in $EFFECTS; do mkdir -p "$Q/cand/$e"; done
fi

# The tool: one binary for validation and the test (§4.6). A different build is allowed to run but is said out loud,
# and the frozen row records whatever binary tested the file.
hdr "the tool"
src=${Q2_BUILD_BIN:-$REPO/build/nvfx_ordered}
if [ ! -x "$B" ]; then
  if [ -x "$src" ]; then
    if [ "$DRY" = 1 ]; then echo "tool: $B is missing; would copy $src (SHA-256 $(sha_of "$src"))"
    else cp "$src" "$B" && chmod +x "$B" && echo "tool: copied $src to $B"; fi
  else note "tool: $B is missing and so is $src (build it: cmake --build build --target nvfx_ordered; or point Q2_BUILD_BIN at a build)"; fi
fi
if [ -x "$B" ]; then
  have=$(sha_of "$B")
  if [ "$have" = "$BIN_SHA_Q2" ]; then echo "tool: $B is §4.6's binary ($have)"
  else echo "tool: $B is NOT §4.6's binary (SHA-256 $have, §4.6's $BIN_SHA_Q2): allowed; the numbers depend only on the seeds and the files (§6), but say so in STUDY_Q2.md §0 and q2_frozen.csv's last field"; fi
  if [ -x "$src" ] && [ "$(sha_of "$src")" != "$have" ]; then
    echo "tool: the build at $src differs from $B (built later, or another commit); bootstrap.sh never replaces a copied tool: rm $B and run again to take the build, or leave it (val.sh and after_val.sh log the hash they ran with)"
  fi
fi

hdr "the tables"
if [ -f "$RULE" ]; then echo "rule: $RULE ok ($(grep -c . "$RULE") lines)"
else note "rule: $RULE is missing (the committed rule; valq2 and testq2 refuse an effect without a row; git checkout results/experiments/q2_fix)"; fi

# Each effect: its shipped file, its candidates, and where it stands.
hdr "the effects"
any_missing=0; missing_files=""
for e in $EFFECTS; do
  if cur=$(current_file "$e" 2>&1); then
    case $cur in "$REPO"/models/2d/*) cs="ok (git's $cur; the data root has no fx_*/q_fix copy, which is fine)" ;; *) cs="ok ($cur)" ;; esac
  else cs="MISSING: $cur"; note "$e: shipped file: $cur"; fi
  missing=""; present=""
  for n in $(candidates "$e"); do
    f=$Q/cand/$e/$n.nvfx
    if [ -f "$f" ]; then present="$present $n($(stat -c %s "$f") bytes)"; else missing="$missing $n"; fi
  done
  if [ -f "$Q/test_$e.done" ]; then st="tested: $(cat "$Q/test_$e.done")"
  elif [ -f "$Q/jobs/val_$e.done" ]; then st="validated; next: tools/study_q2/after_val.sh $e"
  elif grep -q "^$e,chosen," "$R/q2_val_choice.csv" 2>/dev/null; then st="validated in the tables (no local marker; touch $Q/jobs/val_$e.done, then after_val.sh $e)"
  else st="at validation (not done)"; fi
  if [ -n "$missing" ]; then
    cands="MISSING:$missing${present:+; present:$present}"; any_missing=1
    for n in $missing; do missing_files="$missing_files${missing_files:+ }cand/$e/$n.nvfx"; done
  else cands="all present:$present"; fi
  echo "$e: current $cs; candidates $cands; $st"
done
if [ $any_missing = 1 ]; then
  # The one thing this machine cannot make: the candidates are the container's fine-tunes (§4.1). Say exactly which
  # files, where they are, and the two ways to fetch them (the script, or rsync by hand); HOST is the container.
  n_missing=$(echo "$missing_files" | wc -w)
  note "candidates: $n_missing missing under $Q/cand/ (they exist only on the container, /root/nvfx-data/q2_fix/cand/; nothing here can make them):
       missing: $missing_files
       with the script (dry run first, then APPLY=1; HOST is the container's address, RSYNC_OPTS=\"ssh -p PORT -i KEY\" if needed):
         RSYNC_SRC=root@HOST:/root/nvfx-data/q2_fix NEURALVFX_DATA=$D tools/study_q2/restore_from.sh
         RSYNC_SRC=root@HOST:/root/nvfx-data/q2_fix APPLY=1 NEURALVFX_DATA=$D tools/study_q2/restore_from.sh
       or by hand, the same copy in one line (add -n to see it first; Q2_RESTORE_VT or a second line for vt/ is optional):
         rsync -a --partial --info=progress2 --exclude='*.tmp' root@HOST:/root/nvfx-data/q2_fix/cand/ $Q/cand/
       no ssh between the two: scp, a push from the container, or a tarball (docs/STUDY_Q2_RESTORE.md §2); then run bootstrap.sh again"
fi
if [ -d "$Q/vt" ] && [ -n "$(ls -A "$Q/vt" 2>/dev/null)" ]; then echo "vt: $Q/vt has $(find "$Q/vt" -type f | wc -l) cached statistics files (reused as they are; keyed by seeds and file hashes)"
else echo "vt: $Q/vt is empty or absent (fine: the validations recompute; a copy of the container's vt/ only saves time, §2)"; fi

hdr "checklist"
if [ ${#todo[@]} -eq 0 ]; then
  echo "ready: yes; everything reads ok"
  if [ "$DRY" = 1 ] && { [ ! -d "$Q" ] || [ ! -x "$B" ]; }; then
    echo "next: NEURALVFX_DATA=$D tools/study_q2/bootstrap.sh (without Q2_DRY) to make the folders and copy the tool, then resume.sh"
  else echo "next: NEURALVFX_DATA=$D tools/study_q2/resume.sh (starts the validations that are not done; after_val.sh is by hand, §0)"; fi
  exit 0
fi
echo "ready: no; ${#todo[@]} to do, in order:"
i=0; for t in "${todo[@]}"; do i=$((i + 1)); echo "  $i. $t"; done
echo "then: NEURALVFX_DATA=$D tools/study_q2/bootstrap.sh again until it reads ready: yes; then resume.sh"
exit 1
