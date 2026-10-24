#!/usr/bin/env bash
# Study R2 (docs/STUDY_R2.md §0): what tools/study_3d/r2_*.sh share. Sourced, not run; each script does
#   REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}; . "$REPO/tools/study_3d/r2_env.sh"
# The portable form of the data root's jobs/ (tools/study_r2/*, written for /root/nvfx-data and /home/user/wt-r2):
# the same steps against environment variables, so another machine runs them without edits.
#
#   NEURALVFX_DATA  the data root (default $HOME/nvfx-data, the tool's own default); study R2's files are under
#                   $NEURALVFX_DATA/r2_smoke3d (D; or R2_ROOT if set): runs/, ckpt/, models/, eval/, plume/, results/,
#                   logs/, bin/, markers/
#   NVFX_REPO       the repository (set by r2_launch for the snapshot copies, which live under $D/logs; default: the
#                   script's own location). Tables go to $REPO/results/3d/r2 (R), where the rule's numbers are committed.
#   NVFX_3D         the tool. Default: the pinned copy $D/bin/nvfx_3d, made on first use from BUILD (default
#                   $REPO/build/nvfx_3d) through nvfx_bin_link.sh (which also refreshes ~/.cache/neuralvfx/nvfx_3d, a
#                   path without spaces) and named by the git hash, so a rebuild mid-study does not change the solver or
#                   the trainer under a running job; REPIN=1 refreshes it.
#   T               training threads (default 8: the batch of 8 windows runs in parallel)
#   J               processes at a time for the runs (default 8; each nvfx_3d on one thread)
#   NICE_LEVEL      nice for every tool process (default 10)
#   R2_BUDGET_MB    if set, a job stops when r2_smoke3d holds more (the old container's limit was 2500)
#   MKDATA=1        r2_bootstrap.sh only: make the data root's folder when it is not there (otherwise it stops, exit 2,
#                   so a mistyped root does not grow a second, empty r2_smoke3d)
#
# Markers ($D/markers), each made when its stage is whole: candidates_done (r2_train.sh), plume_done (the §1.6 runs and
# fits, r2_resume.sh), rule_committed (by hand, or r2_bootstrap.sh when git has the rule's numbers), val_done and
# frozen (r2_validate.sh), frozen_committed (by hand: validation's files committed and pushed), test_done (r2_test.sh).
: "${REPO:?set REPO to the repo root before sourcing r2_env.sh}"
DATA=${NEURALVFX_DATA:-$HOME/nvfx-data}
D=${R2_ROOT:-$DATA/r2_smoke3d}
R=results/3d/r2
MARK=$D/markers
H="--study r2 --root $D"
W="--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30"
NICE=${NICE_LEVEL:-10}
J=${J:-8}
TOTAL_IT=1750    # 900 + 350 + 250 + 250 (STUDY_R2.md §2.4)
EDGE_ROWS=305    # one model's edge table: 5 winds x 61 rows (a header besides)
PLUME_RUNS=18    # §1.6: v_i50, v_i20, v_i80 (4 seeds each), v_t00, v_t100 (3 each)
PLUME_LABELS="v_i50 v_i20 v_i80 v_t00 v_t100"

r2_dirs() { mkdir -p "$D/logs" "$D/markers" "$D/models" "$D/results" "$D/runs" "$D/ckpt" "$D/bin" "$D/plume"; }
r2_log() { echo "$(date -Iseconds) $*"; }

# The tool (see NVFX_3D above).
r2_bin() {
  if [ -n "${NVFX_3D:-}" ] && [ -x "${NVFX_3D}" ]; then echo "$NVFX_3D"; return 0; fi
  local pinned="$D/bin/nvfx_3d"
  if [ -x "$pinned" ] && [ "${REPIN:-0}" != 1 ]; then echo "$pinned"; return 0; fi
  local src hash
  src=$(BUILD="${BUILD:-$REPO/build/nvfx_3d}" bash "$REPO/tools/study_3d/nvfx_bin_link.sh") || return 1
  src=$(readlink -f "$src")
  hash=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo nogit)
  git -C "$REPO" diff --quiet HEAD -- src include tools/nvfx_3d.cpp tools/plume3d.cpp 2>/dev/null || hash="$hash-dirty"
  mkdir -p "$D/bin" "$D/logs"
  cp -f "$src" "$D/bin/nvfx_3d_$hash.tmp" && mv -f "$D/bin/nvfx_3d_$hash.tmp" "$D/bin/nvfx_3d_$hash"
  chmod +x "$D/bin/nvfx_3d_$hash"
  ln -sfn "nvfx_3d_$hash" "$pinned"
  echo "$(date -Iseconds) pinned $src as bin/nvfx_3d_$hash" >>"$D/logs/bin.log" 2>/dev/null || true
  echo "$pinned"
}

# The disk (jobs/disk.sh's rule): at least G GB free where the data root lives, else exit 9. The old container's
# budget (R2 under 2.5 GB) is reported, and enforced only with R2_BUDGET_MB set.
r2_disk_need() {
  local g=$1 free_kb used_kb lim
  mkdir -p "$D"
  free_kb=$(df --output=avail -k "$D" | tail -1)
  used_kb=$(du -sk "$D" 2>/dev/null | cut -f1); used_kb=${used_kb:-0}
  echo "disk: $((free_kb / 1024)) MB free under $D; r2_smoke3d holds $((used_kb / 1024)) MB"
  lim=$(awk -v g="$g" 'BEGIN { printf "%d", g * 1048576 }')
  if [ "$free_kb" -lt "$lim" ]; then echo "DISK SHORT: under $g GB free; stopping"; exit 9; fi
  if [ -n "${R2_BUDGET_MB:-}" ] && [ "$used_kb" -gt $((R2_BUDGET_MB * 1024)) ]; then echo "DISK BUDGET: r2_smoke3d over $R2_BUDGET_MB MB; stopping"; exit 9; fi
}

# Is a job running? Patterns are matched against whole command lines (pgrep -f) and anchored on the program, so a
# shell that merely names a job (an editor, a grep) does not count. The scripts run as snapshot copies
# ($D/logs/<name>.snapshot.sh) when r2_resume.sh starts them, or as the repo file by hand: both forms match.
r2_running() { pgrep -f -- "$1" >/dev/null 2>&1; }
R2_PAT_TRAIN="nvfx_3d[^ ]* (train|finish) --study r2 --root $D"
R2_PAT_RECORD="nvfx_3d[^ ]* record --study r2 --root $D"
R2_PAT_TRAIN_SH="bash [^ ]*r2_train(\.snapshot)?\.sh"
R2_PAT_VAL_SH="bash [^ ]*r2_validate(\.snapshot)?\.sh"
R2_PAT_TEST_SH="bash [^ ]*r2_test(\.snapshot)?\.sh"
R2_PAT_PLUME_SH="bash [^ ]*r2_resume(\.snapshot)?\.sh --plume-worker"
R2_PAT_ANY_TOOL="nvfx_3d[^ ]* [a-z]+ .*--study r2 --root $D"

# Start tools/study_3d/<name>.sh detached (setsid nohup, as the container's jobs), from a snapshot copy under $D/logs:
# an edit to the repo file during the hours-long run cannot touch the running script (the copy is renamed into place,
# so a copy still running keeps its own inode). NVFX_REPO tells the copy where the repo is.
#   r2_launch NAME LOG [ARGS...]
r2_launch() {
  local name=$1 lg=$2; shift 2
  local snap="$D/logs/$name.snapshot.sh"
  mkdir -p "$D/logs"
  cp -f "$REPO/tools/study_3d/$name.sh" "$snap.tmp" && mv -f "$snap.tmp" "$snap"
  r2_log "starting: $name $* (log $lg)"
  NVFX_REPO=$REPO setsid nohup bash "$snap" "$@" >>"$lg" 2>&1 </dev/null &
}

# One nvfx_3d task per line on stdin ("cmd args..."), J at a time, each on one thread, niced. Needs B set.
r2_run_many() { xargs -P "$J" -I{} sh -c "nice -n $NICE $B {} $H --threads 1"; }

# The last training line, de-duplicated (the tool logs every 10 iterations; a resumed run logs some again).
r2_train_last() { grep '^it ' "$D/logs/train_r2.log" 2>/dev/null | awk '!seen[$0]++' | tail -1 || true; }
r2_train_done() { grep -q "done at iteration $TOTAL_IT" "$D/logs/train_r2.log" 2>/dev/null; }
r2_nfiles() { local n; n=$(ls -- "$@" 2>/dev/null | wc -l) || true; echo "${n:-0}"; }  # how many of a glob exist (0 is not an error under pipefail)
r2_runs_count() { r2_nfiles "$D/runs/$1/"*.n3run; }
r2_edge_rows() { local n=0; [ -f "$1" ] && { n=$(tail -n +2 "$1" | wc -l) || true; }; echo "${n:-0}"; }
r2_plume_count() {  # the §1.6 runs written (seed_??.bin under plume/<label>)
  local n=0 c
  for c in $PLUME_LABELS; do n=$((n + $(r2_nfiles "$D/plume/$c/"seed_*.bin))); done
  echo "$n"
}
r2_plume_tasks() {  # the 18 runs as single-seed tasks (jobs/teacher_tasks.txt), each skipped by the tool if written
  local c k n base a
  for c in $PLUME_LABELS; do
    case $c in
      v_i50) n=4; base=34010000; a="--intensity 0.5";;
      v_i20) n=4; base=34010100; a="--intensity 0.2";;
      v_i80) n=4; base=34010200; a="--intensity 0.8";;
      v_t00) n=3; base=34010300; a="--turbulence 0";;
      v_t100) n=3; base=34010400; a="--turbulence 1";;
    esac
    for k in $(seq 0 $((n - 1))); do
      echo "plume --effect smoke_r2 --spin 6 --seconds 16 --label $c $a --from $k --seeds $((k + 1)) --seed-base $base"
    done
  done
}

# The rule's numbers (§2.10) are in the repo and committed (no uncommitted change to them): then validation may run.
r2_rule_in_git() {
  [ -f "$REPO/$R/margins.csv" ] && [ -f "$REPO/$R/edge_bounds.csv" ] &&
    git -C "$REPO" ls-files --error-unmatch "$R/margins.csv" "$R/edge_bounds.csv" >/dev/null 2>&1 &&
    [ -z "$(git -C "$REPO" status --porcelain -- "$R/margins.csv" "$R/edge_bounds.csv")" ]
}

r2_combine_edge() {  # out, then the parts: one header, every part's rows (written whole, then renamed)
  local out=$1; shift
  local tmp="$out.tmp" first=1 p
  rm -f "$tmp"
  for p in "$@"; do
    [ -f "$p" ] || continue
    if [ $first = 1 ]; then cp "$p" "$tmp"; first=0; else tail -n +2 "$p" >>"$tmp"; fi
  done
  [ -f "$tmp" ] && mv -f "$tmp" "$out"
}

r2_freeze_once() {  # a model's row in frozen.csv, once (§2.8: the chosen file frozen by SHA-256 before the test)
  local model=$1 base sha
  base=$(basename "$model"); sha=$(sha256sum "$model" | awk '{print $1}')
  if [ -f "$R/frozen.csv" ] && grep -q "^$base,$sha," "$R/frozen.csv"; then echo "skip freeze $base ($sha): already in $R/frozen.csv"; return 0; fi
  "$B" freeze --model "$model" --frozen "$R/frozen.csv"
}

r2_frozen_ok() {  # the chosen model is there and its SHA-256 is the frozen one
  [ -f "$D/models/chosen.nvfx3" ] && [ -f "$D/models/chosen.txt" ] && [ -f "$REPO/$R/frozen.csv" ] &&
    grep -q "^chosen.nvfx3,$(sha256sum "$D/models/chosen.nvfx3" | awk '{print $1}')," "$REPO/$R/frozen.csv"
}

r2_chosen() {  # the choice: CHOSEN from the environment, else the "chosen: C?" line of choice_val.md
  if [ -n "${CHOSEN:-}" ]; then echo "$CHOSEN"; return 0; fi
  grep -E '^chosen: C[234]' "$REPO/$R/choice_val.md" 2>/dev/null | tail -1 | awk '{print $2}'
}
