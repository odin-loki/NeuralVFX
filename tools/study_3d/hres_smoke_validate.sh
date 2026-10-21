#!/bin/sh
# Study hres smoke: demo validation (docs/STUDY_3D_HRES.md §6).
# Run from the repo root after hres_smoke.sh has produced smoke_hres.nvfx3 under the data root.
# Usage: hres_smoke_validate.sh BUILD_DIR [DATA_ROOT]
# Lighter than hres_fire_validate.sh: 30 s edge sim + learned vs smoke2, edgecheck; no held-out test.
# Optional: set TRUTH=1 to also run val truth/real, track/play, and compare smoke_hres vs smoke2.
# Resumes from files already written (edge CSVs append; delete them to rerun edge).
set -e

B=${1:-build/nvfx_3d}
ROOT=${2:-${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/smoke}
DATA=${NEURALVFX_DATA:-$HOME/nvfx-data}
R=results/3d/hres/smoke
H="--study hres --root $ROOT --effect smoke"
W="--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30 --effect smoke"
N2_SRC=$DATA/n2_3d/models/smoke2.nvfx3
LEARNED=${LEARNED:-smoke_hres}

mkdir -p "$R" "$ROOT/models"
cp "$N2_SRC" "$ROOT/models/smoke2.nvfx3"

combine_edge() {
  out=$1
  shift
  rm -f "$out"
  for p in "$@"; do
    [ -f "$p" ] || continue
    if [ ! -f "$out" ]; then
      cp "$p" "$out"
    else
      tail -n +2 "$p" >> "$out"
    fi
  done
}

echo "hres smoke validate: data $ROOT, results $R"

$B edge $H --sim $W --out "$R/edge_sim.csv"
$B edge $H --model "$ROOT/models/smoke2.nvfx3" --label smoke2 $W --out "$R/edge_smoke2.csv"
$B edge $H --model "$ROOT/models/$LEARNED.nvfx3" --label "$LEARNED" $W --out "$R/edge_learned.csv"
combine_edge "$R/edge.csv" "$R/edge_sim.csv" "$R/edge_smoke2.csv" "$R/edge_learned.csv"
$B edgecheck --csv "$R/edge.csv" | tee "$R/edgecheck.md"

if [ -n "${TRUTH:-}" ]; then
  $B truth $H --split val
  $B real $H --split val
  for m in smoke2 "$LEARNED"; do
    $B track $H --split val --method "$m" --model "$ROOT/models/$m.nvfx3"
    $B play $H --split val --method "$m" --model "$ROOT/models/$m.nvfx3"
  done
  $B compare $H --split val --a "$LEARNED" --b smoke2 --results "$R"
fi

echo "done: edge tables under $R (baseline smoke2 from $N2_SRC; learned $ROOT/models/$LEARNED.nvfx3)"
