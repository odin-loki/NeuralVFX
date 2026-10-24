#!/usr/bin/env bash
# Copy hres fire test-phase tables from the data root into results/3d/hres/fire/ (snapshot runs wrote them under
# $NEURALVFX_DATA/hres_3d/results/...). Copies only what the repo lacks; FORCE=1 overwrites the repo's copies.
# Usage: bash tools/study_3d/hres_sync_fire_test_results.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${NEURALVFX_DATA:-$HOME/nvfx-data}
SRC="$DATA/hres_3d/results/3d/hres/fire"
DST="$REPO/results/3d/hres/fire"
FORCE=${FORCE:-0}
mkdir -p "$DST"
if [ ! -d "$SRC" ]; then
  echo "no stray dir $SRC (nothing to copy)"
  exit 0
fi
n=0
kept=0
for f in edge_test_sim.csv edge_test_learned.csv edge_test_N2.csv edge_test.csv edgecheck_test.md \
  compare_test_learned_vs_coarse.md compare_test_learned_vs_coarse.csv \
  compare_test_learned_vs_N2.md compare_test_learned_vs_N2.csv; do
  [ -f "$SRC/$f" ] || continue
  if [ -f "$DST/$f" ] && [ "$FORCE" != 1 ]; then kept=$((kept + 1)); continue; fi
  cp -v "$SRC/$f" "$DST/"
  n=$((n + 1))
done
echo "copied $n files to $DST ($kept already there, kept; FORCE=1 overwrites)"
# The stray edgecheck_test.md is the 22 October single-run table; the repo's is judged with the calibrated bounds.
if [ -f "$DST/edgecheck_test.md" ] && ! grep -q 'calibrated bound' "$DST/edgecheck_test.md" && [ -f "$DST/edge_bounds.csv" ]; then
  echo "edgecheck_test.md is the single-run table; judge it with the calibrated bounds: bash tools/study_3d/hres_fire_recheck_test_edges.sh"
fi
