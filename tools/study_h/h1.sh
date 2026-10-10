#!/bin/sh
# Study H, H1: products on LZ78-, RePair- and CSR-coded data against the dense AVX2 loop (thread CPU time, least of
# REPS, one pinned core). Usage: h1.sh BUILD_DIR OUT_CSV [CORE] [REPS]
B=${1:-build}; OUT=${2:-results/experiments/h1_compressed_products.csv}; CORE=${3:-3}; REPS=${4:-15}
M=${NEURALVFX_DATA:-/root/nvfx-data}/experiments/models
echo "load before: $(cat /proc/loadavg)"
taskset -c $CORE nice -n 10 $B/nvfx_study_h --h1 --reps $REPS --out $OUT \
  $M/b/fire_grid_k8.nvfx $M/b/fire_grid_k16.nvfx $M/c/fire_variation_k8.nvfx $M/c/fire_variation_k24.nvfx $M/a/fire_0_grid_m8.nvfx \
  $M/a/fire_0_grid_m16.nvfx $M/d/explosion.nvfx $M/d/smoke.nvfx
echo "load after: $(cat /proc/loadavg)"
