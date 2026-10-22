#!/bin/sh
# Study hres, smoke: finer coarse rollouts for study N2's smoke2 lineage (docs/STUDY_3D_HRES.md §6).
# Usage: hres_smoke.sh BUILD_DIR [DATA_ROOT] [THREADS]
set -e
B=${1:-build/nvfx_3d}
ROOT=${2:-${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/smoke}
TREC=${3:-3}
TTRAIN=${4:-2}
TRAIN_DEVICE=${TRAIN_DEVICE:-cpu}
TRAIN_SHARDS=${TRAIN_SHARDS:-4}
H="--study hres --root $ROOT --effect smoke"
TD="--device $TRAIN_DEVICE --shards $TRAIN_SHARDS"
W="--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30 --effect smoke"

echo "hres smoke: data $ROOT"
$B record $H --runs 40 --frames 240 --threads $TREC
$B record $H --runs-dir long --runs 24 --frames 540 --threads $TREC
$B train $H --name smoke --runs-dir train --also "$ROOT/runs/long" --threads $TTRAIN $TD
$B finish $H --ckpt smoke --out smoke_hres --runs-dir train --also "$ROOT/runs/long"

echo "smoke checks (demo, not a full study rule):"
echo "  $B edge $H --sim $W --out results/3d/hres/smoke/edge_sim.csv"
echo "  $B edge $H --model $ROOT/models/smoke_hres.nvfx3 --label smoke_hres $W --out results/3d/hres/smoke/edge_learned.csv"
echo "  $B video --kind orbit --model $ROOT/models/smoke_hres.nvfx3 --title 'LEARNED 3D SMOKE (hres)' --out videos/3d_smoke_hres_orbit.mp4 --seconds 12 --controls 0.7,0.5,0.5 --seed 9961005"
