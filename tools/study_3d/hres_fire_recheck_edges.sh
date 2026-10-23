#!/usr/bin/env bash
# Rebuild edge_val.csv / edgecheck / choose with calibrated bounds (after hres_fire_edge_bounds.sh).
# Does not rerun model edge tables or val track/play. Safe while validate is on test phase if val edges exist.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export USE_EDGE_BOUNDS=1
B=$(BUILD="${BUILD:-$REPO/build/nvfx_3d}" bash "$REPO/tools/study_3d/nvfx_bin_link.sh")
ROOT=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/fire
R=results/3d/hres/fire
H="--study hres --root $ROOT --effect fire"
W="--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30"
EDGE_FULL=306
RUNS=${EDGE_RUNS:-4}
nvfx() { "$B" "$@"; }

edge_done() { [ -f "$1" ] && [ "$(wc -l <"$1")" -ge "$EDGE_FULL" ]; }
edge_done_multi() {
  local f=$1 runs=${2:-$RUNS}
  [ -f "$f" ] && [ "$(wc -l <"$f")" -ge $((1 + runs * 5 * 61)) ]
}

combine_edge() {
  out=$1
  shift
  rm -f "$out"
  for p in "$@"; do
    [ -f "$p" ] || continue
    if [ ! -f "$out" ]; then cp "$p" "$out"; else tail -n +2 "$p" >>"$out"; fi
  done
}

[ -f "$R/edge_bounds.csv" ] && edge_done_multi "$R/edge_teacher.csv" || {
  echo "run: bash tools/study_3d/hres_fire_edge_bounds.sh"; exit 1
}
for m in N2 C2 C3 C4; do
  edge_done "$R/edge_val_$m.csv" || { echo "missing edge_val_$m.csv"; exit 1; }
done
combine_edge "$R/edge_val.csv" "$R/edge_teacher.csv" "$R/edge_val_N2.csv" "$R/edge_val_C2.csv" "$R/edge_val_C3.csv" "$R/edge_val_C4.csv"
nvfx edgecheck --csv "$R/edge_val.csv" --bounds "$R/edge_bounds.csv" | tee "$R/edgecheck_val.md"
nvfx choose $H --split val --cands C2,C3,C4 --edge "$R/edge_val.csv" --bounds "$R/edge_bounds.csv" | tee "$R/choose_val.md"
echo "recheck done; re-run freeze/test from hres_fire_validate.sh if CHOSEN changed"
