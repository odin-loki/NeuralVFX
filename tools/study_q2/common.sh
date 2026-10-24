#!/bin/bash
# Study Q2 (docs/STUDY_Q2.md §0): what every job script shares. Sourced, not run. The container's copies live in the
# data root's q2_fix/jobs/; these are the same steps written against environment variables so that another machine
# (or a fresh container) runs them without edits.
#
#   NEURALVFX_DATA  the data root (default /root/nvfx-data); study Q2's files are under $NEURALVFX_DATA/q2_fix (Q)
#   Q2_REPO         the git worktree holding results/experiments/q2_fix (default: this script's repository)
#   Q2_BIN          the tool for validation and the test (default $Q/bin/nvfx_ordered_q2; bootstrap.sh copies it there
#                   from $Q2_BUILD_BIN, default $Q2_REPO/build/nvfx_ordered, and says whether its SHA-256 is §4.6's)
#   Q2_DRY=1        bootstrap.sh only: check and report, write nothing (no folders, no copy of the tool)
#   Q2_THREADS      threads per validation or test (default 2, the container's)
#   Q2_NO_PUSH=1    after_val.sh freezes without committing and pushing q2_frozen.csv (the test then does not start:
#                   §4.6 wants the frozen row pushed before the test runs)
set -u
D=${NEURALVFX_DATA:-/root/nvfx-data}
Q=$D/q2_fix
REPO=${Q2_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
R=$REPO/results/experiments/q2_fix
B=${Q2_BIN:-$Q/bin/nvfx_ordered_q2}
T=${Q2_THREADS:-2}
RULE=$R/q2_rule.csv
BIN_SHA_Q2=420131ac5e3455943401bda68199660191558cacbc8deb568a48bbe551dd265a  # §4.6: the probe's, validation's and the test's binary

# The three effects still at validation (§0's table); the three replaced ones are kept for the record and skip.
EFFECTS="geyser heavygas mushroom"
shipped_rel() {  # the shipped file under the data root (the same bytes as models/2d/<name>.nvfx in git)
  case $1 in
    geyser) echo fx_elements/models/d/geyser.nvfx ;;
    haboob) echo fx_elements/models/d/haboob.nvfx ;;
    smokering) echo fx_air/models/d/smokering.nvfx ;;
    heavygas) echo fx_air/models/d/heavygas.nvfx ;;
    avalanche) echo q_fix/models/avalanche.nvfx ;;
    mushroom) echo q_fix/models/mushroom.nvfx ;;
    *) return 1 ;;
  esac
}
shipped_sha() {  # §4.1's SHA-256 of each shipped file (the full hash, from models/SHA256SUMS at the study's start)
  case $1 in
    geyser) echo 43095b69b808b87e45b13410f831bc81e3e55361fdad6f754f6457934e2c9c9f ;;
    heavygas) echo 8d436f906cc2fe939aff4830d1368144680c8c2ad4c268dcb59d0b7a9d089e16 ;;
    mushroom) echo c1cd25a276bd3b2a9e0578e631ed85ad9cd9286d5625d30b2c987b2cd9004209 ;;
    *) echo "" ;;
  esac
}
candidates() {  # §4.1's candidates, every one of which goes to validation (the avalanche's msg was dropped by §3)
  case $1 in
    geyser) echo "u30a u30al" ;;
    haboob | smokering) echo "rft rfs" ;;
    avalanche) echo "s48" ;;
    heavygas) echo "ms msg s24" ;;
    mushroom) echo "u60a u90a" ;;
    *) return 1 ;;
  esac
}
# The current (shipped) file: the data root's if it is there, else git's models/2d copy; its SHA-256 must be §4.1's.
current_file() {
  local e=$1 p
  p=$D/$(shipped_rel "$e")
  [ -f "$p" ] || p=$REPO/models/2d/$e.nvfx
  [ -f "$p" ] || { echo "current_file: no shipped file for $e ($D/$(shipped_rel "$e") or $REPO/models/2d/$e.nvfx)" >&2; return 1; }
  local want have
  want=$(shipped_sha "$e")
  if [ -n "$want" ]; then
    have=$(sha256sum "$p" | cut -c1-64)
    [ "$have" = "$want" ] || { echo "current_file: $p has SHA-256 $have, not §4.1's $want" >&2; return 1; }
  fi
  echo "$p"
}
cand_list() {  # --cand NAME=PATH,... from $Q/cand/<effect>/<name>.nvfx; fails if one is missing
  local e=$1 n out=""
  for n in $(candidates "$e"); do
    [ -f "$Q/cand/$e/$n.nvfx" ] || { echo "cand_list: $Q/cand/$e/$n.nvfx is missing (the candidates are made on the container, §4.1)" >&2; return 1; }
    out="$out${out:+,}$n=$Q/cand/$e/$n.nvfx"
  done
  echo "$out"
}
sha_of() { sha256sum "$1" | cut -c1-64; }
log() { echo "$(date -u +%FT%TZ) $*"; }
running() { pgrep -f -- "$1" > /dev/null; }  # patterns anchored at the program, so a shell that merely names a job does not count
launch() { local lg=$1; shift; log "starting: $* (log $lg)"; setsid nohup "$@" > "$lg" 2>&1 < /dev/null & }

# Disk (§0, the coordinator's limit): q2_fix stays under 1.5 GB and / keeps 1 GB free; a data-making step waits while
# either fails and logs PAUSED to logs/disk.log.
check_disk() {
  local used free paused=0
  while :; do
    used=$(du -sm "$Q" 2>/dev/null | cut -f1); used=${used:-0}
    free=$(df -m / | awk 'NR == 2 { print $4 }')
    if [ "$used" -le 1500 ] && [ "$free" -ge 1024 ]; then
      [ $paused -eq 1 ] && log "RESUMED (q2_fix ${used} MB, / ${free} MB free)" >> "$Q/logs/disk.log"
      return 0
    fi
    [ $paused -eq 0 ] && log "PAUSED: q2_fix ${used} MB (limit 1500), / ${free} MB free (needs 1024)" >> "$Q/logs/disk.log"
    paused=1
    sleep 120
  done
}
