#!/bin/sh
# Study hres, fire: finer coarse rollouts for study N2's C4 lineage (docs/STUDY_3D_HRES.md).
# Usage: hres_fire.sh BUILD_DIR [DATA_ROOT] [THREADS]
# Resumes every step from files already under DATA_ROOT.
set -e
B=${1:-build/nvfx_3d}
ROOT=${2:-${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/fire}
TREC=${3:-3}
TTRAIN=${4:-2}
H="--study hres --root $ROOT --effect fire"
W="--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30"
N2=${NEURALVFX_DATA:-$HOME/nvfx-data}/n2_3d/models

echo "hres fire: data $ROOT"
$B record $H --runs 40 --frames 240 --threads $TREC
$B record $H --runs-dir long --runs 24 --frames 540 --threads $TREC
$B train $H --name hres --runs-dir train --also "$ROOT/runs/long" --threads $TTRAIN
for k in 1 2 3 4; do
  if [ -f "$ROOT/ckpt/hres_stage$k.ckpt" ]; then $B finish $H --ckpt hres_stage$k --out C$k --runs-dir train --also "$ROOT/runs/long"; fi
done
$B finish $H --ckpt hres --out hres_fire --runs-dir train --also "$ROOT/runs/long"

echo "validation (after choosing a candidate and freezing):"
echo "  $B truth $H --split val; $B real $H --split val"
echo "  $B edge $H --sim $W --out results/3d/hres/fire/edge_sim.csv"
echo "  compare hres C* against N2 baseline: $N2/chosen.nvfx3 (16³ coarse; picture metrics only)"
