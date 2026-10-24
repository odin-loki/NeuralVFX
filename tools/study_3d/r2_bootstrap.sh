#!/usr/bin/env bash
# Study R2 (docs/STUDY_R2.md §0): make the data root's r2_smoke3d ready on a machine that has the repository and a
# build, or check one that has the container's data copied in. Makes no numbers and starts no job: the folders, the
# tool pinned to $D/bin/nvfx_3d (from BUILD, default build/nvfx_3d), the rule's numbers (results/3d/r2, committed:
# markers/rule_committed is made when git has them), and what is there of each stage against what the container had
# (STUDY_R2.md §0: the 48 runs, 1.1 GB; the checkpoints; the candidates; the teacher's runs). Usage:
#   NEURALVFX_DATA=$HOME/nvfx-data bash tools/study_3d/r2_bootstrap.sh
# What to copy from the container's /root/nvfx-data/r2_smoke3d (its layout is this one's): runs/ (long, short),
# ckpt/, logs/train_r2.log (the trainer's own log; "done at iteration 1750" is how training is known to be done),
# models/ (C2..C4 if written), markers/, results/edge_teacher_w*.csv, eval/ (the calibration and any validation
# runs), plume/ (§1.6's runs). Never bin/ (the tool is pinned from this machine's build). The one-line rsync, the
# checks and the order (this script before and after the copy; r2_resume.sh only after): docs/STUDY_R2_RESTORE.md.
# When the data is missing it says so plainly (STUDY_R2.md §0, "Without the container's data"): a data root whose
# folder does not exist stops it (exit 2; mkdir -p it, or MKDATA=1 makes it), an r2_smoke3d with none of the
# container's stages is reported with what to copy (docs/STUDY_R2_RESTORE.md, and its rsync line ready to paste with
# HOST filled in) and what starting from nothing costs, a partial copy with what is not there and the same rsync line
# (--partial resumes it). Exit 0 otherwise, data or no data: an empty r2_smoke3d is a legitimate start (r2_resume.sh
# fills it).
set -euo pipefail
REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
. "$REPO/tools/study_3d/r2_env.sh"
cd "$REPO"

# The data root's folder must be there before anything is made under it: a mistyped NEURALVFX_DATA (or R2_ROOT) must
# not quietly grow a second, empty r2_smoke3d somewhere else while the real one sits unseen.
parent=$(dirname "$D")
if [ ! -d "$parent" ]; then
  if [ "${MKDATA:-0}" = 1 ]; then mkdir -p "$parent"; echo "data root: $parent made (MKDATA=1)"
  else
    cat >&2 <<EOF
r2_bootstrap: the data root's folder $parent does not exist; nothing made.
  NEURALVFX_DATA=${NEURALVFX_DATA:-<unset; the default is \$HOME/nvfx-data>}${R2_ROOT:+; R2_ROOT=$R2_ROOT}
  Is this the machine's data root (the one that holds hres_3d, n2_3d, or the container's r2_smoke3d)? Then set
  NEURALVFX_DATA to it. A new data root on purpose: mkdir -p "$parent", or MKDATA=1 on this command.
  docs/STUDY_R2.md §0, "Where the files are (portable)".
EOF
    exit 2
  fi
fi
had_d=1; [ -d "$D" ] || had_d=0
r2_dirs
echo "data root $DATA; r2_smoke3d $D; repository $REPO; tables $REPO/$R"

# The container's data: which of its stages are here at all. None: say what to copy, or what starting from nothing
# costs (STUDY_R2.md §0). Some: say what is not there, so a half-finished copy is not mistaken for a fresh start.
nl=$(r2_runs_count long); ns=$(r2_runs_count short)
nck=$(r2_nfiles "$D/ckpt/"*); nmod=$(r2_nfiles "$D/models/"*.nvfx3)
ne=0; for j in 0 1 2 3 4; do [ "$(r2_edge_rows "$D/results/edge_teacher_w$j.csv")" -ge 244 ] && ne=$((ne + 1)); done
np=$(r2_plume_count)
have=(); lack=()
if [ "$nl" -gt 0 ] || [ "$ns" -gt 0 ]; then have+=("runs/ ($nl long, $ns short)"); else lack+=("runs/long and runs/short (the 48 training runs, 1.1 GB)"); fi
if [ "$nck" -gt 0 ]; then have+=("ckpt/ ($nck)"); else lack+=("ckpt/ (training's checkpoints)"); fi
if [ -f "$D/logs/train_r2.log" ]; then have+=("logs/train_r2.log"); else lack+=("logs/train_r2.log (the trainer's log: where training stood)"); fi
if [ "$nmod" -gt 0 ]; then have+=("models/ ($nmod)"); else lack+=("models/ (C2..C4, if training had written them)"); fi
if [ "$ne" -gt 0 ]; then have+=("results/edge_teacher_w*.csv ($ne of 5)"); else lack+=("results/edge_teacher_w*.csv (the edge bounds' teacher runs; the record only)"); fi
if [ "$np" -gt 0 ]; then have+=("plume/ ($np of $PLUME_RUNS)"); else lack+=("plume/ (§1.6's runs)"); fi
if [ -d "$D/eval" ]; then have+=("eval/"); else lack+=("eval/ (the calibration's and validation's runs, if any)"); fi
join() { local s="" x; for x in "$@"; do s="$s${s:+; }$x"; done; echo "$s"; }
# The copy, in one line: docs/STUDY_R2_RESTORE.md's rsync (every folder but bin/; --partial resumes a cut copy; no
# --delete, so nothing of this machine's is removed). Printed whenever the container's data is missing or partial.
RSYNC_LINE="rsync -a --partial --info=progress2 --exclude='bin/' root@HOST:/root/nvfx-data/r2_smoke3d/ \"$D/\""
if [ ${#have[@]} -eq 0 ]; then
  cat <<EOF
data: NONE of the container's r2_smoke3d is here$([ $had_d = 1 ] || echo " ($D was not there; its empty folders are made now)"): no training run, checkpoint, trainer log, candidate, edge teacher run or plume run.
  Meant to have the container's data? docs/STUDY_R2_RESTORE.md says what to copy (every folder but bin/) and what to
  run after; the copy in one line (HOST is the container; add -n to see it first; --partial resumes a cut copy):
    $RSYNC_LINE
  then run this again (nothing here is overwritten; only the empty folders are filled). Or is the data under another
  root? Then NEURALVFX_DATA=<that root> (or R2_ROOT=<its r2_smoke3d>) on this command.
  Starting from nothing is also fine: bash tools/study_3d/r2_resume.sh records the 48 training runs (about 8 CPU-hours
  at 10 CPU-minutes a run, J=$J at a time), trains on the CPU (hours), runs the §1.6 plume runs beside it; the rule's
  numbers (§2.10) come from the repository. docs/STUDY_R2.md §0, "Without the container's data".
EOF
elif [ ${#lack[@]} -gt 0 ]; then
  echo "data: PART of the container's r2_smoke3d is here: $(join "${have[@]}")"
  echo "  not here: $(join "${lack[@]}")"
  echo "  (a copy still under way, or stages the container never reached? r2_resume.sh makes what is not written; docs/STUDY_R2.md §0)"
  echo "  a cut copy resumes with the same line (docs/STUDY_R2_RESTORE.md; HOST is the container; add -n to see it first):"
  echo "    $RSYNC_LINE"
else
  echo "data: the container's r2_smoke3d is here: $(join "${have[@]}")"
fi

# The tool: one pinned copy for training and the teacher's runs, validation and the test (the container used two
# builds of the same solver and trainer, cb31ccb and a5637ed; a different build is allowed and is said out loud).
if B=$(r2_bin 2>/dev/null); then
  echo "tool: $B -> $(readlink -f "$B") (SHA-256 $(sha256sum "$B" | cut -c1-16)...; REPIN=1 refreshes it from BUILD)"
else
  echo "tool: MISSING: no $D/bin/nvfx_3d and no build at ${BUILD:-$REPO/build/nvfx_3d} (cmake --build build --target nvfx_3d, or set BUILD or NVFX_3D)"
fi
( r2_disk_need 0 ) || true

# The rule's numbers (§2.10), committed before any validation run.
for f in margins.csv edge_bounds.csv edge_teacher.csv edge_calibration.md margins.md; do
  [ -f "$R/$f" ] || echo "rule: $R/$f is MISSING (it is a committed file of the repository, not the data's: git checkout, or STUDY_R2.md §2.10 says how it was made)"
done
if [ -f "$MARK/rule_committed" ]; then echo "rule: committed ($MARK/rule_committed)"
elif r2_rule_in_git; then touch "$MARK/rule_committed"; echo "rule: $R/margins.csv and edge_bounds.csv are committed in git; $MARK/rule_committed made"
else echo "rule: NOT marked committed; commit and push $R/margins.csv and edge_bounds.csv, then touch $MARK/rule_committed"; fi

# Each stage: what is there (the counts from above).
echo "training runs: $nl long, $ns short of 24 each$([ "$nl" -ge 24 ] && [ "$ns" -ge 24 ] && echo ' (complete)' || echo ' (r2_train.sh records the rest: about 10 CPU-minutes a run; or copy the container'\''s tree first: docs/STUDY_R2_RESTORE.md §2)')"
if r2_train_done; then echo "training: DONE (logs/train_r2.log: done at iteration $TOTAL_IT)"
else
  last=$(r2_train_last)
  echo "training: ${last:-not started (no logs/train_r2.log)}; checkpoints: $(ls "$D/ckpt" 2>/dev/null | tr '\n' ' ')"
fi
for m in C2 C3 C4; do [ -f "$D/models/$m.nvfx3" ] && echo "candidate $m: $D/models/$m.nvfx3" || echo "candidate $m: not written"; done
[ -f "$MARK/candidates_done" ] && echo "candidates: done (marker)" || echo "candidates: not marked done (r2_train.sh writes them after training)"
echo "edge bounds' teacher runs: $ne of 5 wind files whole under results/ (for the record only: the committed $R/edge_teacher.csv is what the rule uses)"
echo "plume (§1.6): $np of $PLUME_RUNS runs$([ -f "$MARK/plume_done" ] && echo ' (fits done)' || echo ' (r2_resume.sh runs the rest and the fits)')"
echo "markers: $(ls "$MARK" 2>/dev/null | tr '\n' ' ')"
if [ -f "$MARK/frozen" ]; then
  r2_frozen_ok && echo "frozen: $(cat "$D/models/chosen.txt") as models/chosen.nvfx3 (sha in $R/frozen.csv)" || echo "frozen: MARKER but models/chosen.nvfx3, chosen.txt or $R/frozen.csv do not agree"
fi
echo "next: bash tools/study_3d/r2_resume.sh (starts what is not done; the hand steps are said as they come); bash tools/study_3d/r2_status.sh"
