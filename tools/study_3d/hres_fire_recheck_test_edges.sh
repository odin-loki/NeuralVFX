#!/usr/bin/env bash
# Judge the hres fire's test edge check with the calibrated bounds (after hres_fire_edge_bounds.sh).
#
# The 22 October test step ran before edge_bounds.csv existed, so edgecheck_test.md judged the learned model against a
# single simulated run per wind (study N2's rule) and marked two winds as runaway. This rebuilds the table from the
# same edge_test*.csv (no model or simulation is rerun) against results/3d/hres/fire/edge_bounds.csv, the rule the
# validation was judged by (edgecheck_val.md). A single-run table found in edgecheck_test.md is kept as
# edgecheck_test_single.md.
#
#   bash tools/study_3d/hres_fire_recheck_test_edges.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
B=$(BUILD="${BUILD:-$REPO/build/nvfx_3d}" bash "$REPO/tools/study_3d/nvfx_bin_link.sh")
R=results/3d/hres/fire
EDGE_FULL=306
nvfx() { "$B" "$@"; }

edge_done() { [ -f "$1" ] && [ "$(wc -l <"$1")" -ge "$EDGE_FULL" ]; }

combine_edge() {
  out=$1
  shift
  rm -f "$out"
  for p in "$@"; do
    [ -f "$p" ] || continue
    if [ ! -f "$out" ]; then cp "$p" "$out"; else tail -n +2 "$p" >>"$out"; fi
  done
}

[ -f "$R/edge_bounds.csv" ] || { echo "missing $R/edge_bounds.csv (run: bash tools/study_3d/hres_fire_edge_bounds.sh)"; exit 1; }
for m in sim learned N2; do
  edge_done "$R/edge_test_$m.csv" || {
    echo "missing or partial $R/edge_test_$m.csv (copy: bash tools/study_3d/hres_sync_fire_test_results.sh; or rerun the test step)"
    exit 1
  }
done
if [ -f "$R/edgecheck_test.md" ] && ! grep -q 'calibrated bound' "$R/edgecheck_test.md"; then
  cp -p "$R/edgecheck_test.md" "$R/edgecheck_test_single.md"
  echo "kept the single-run table as $R/edgecheck_test_single.md"
fi
combine_edge "$R/edge_test.csv" "$R/edge_test_sim.csv" "$R/edge_test_learned.csv" "$R/edge_test_N2.csv"
nvfx edgecheck --csv "$R/edge_test.csv" --bounds "$R/edge_bounds.csv" | tee "$R/edgecheck_test.md"
