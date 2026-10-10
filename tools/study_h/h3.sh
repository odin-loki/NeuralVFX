#!/bin/sh
# Study H, H3: sizes and decode times of every coder configuration (thread CPU time, least and median of N interleaved
# repetitions, one pinned core). Usage: h3.sh BUILD_DIR OUT_DIR [CORE] [REPS]
B=${1:-build}; OUT=${2:-results/compression}; CORE=${3:-3}; REPS=${4:-15}
M=${NEURALVFX_DATA:-/root/nvfx-data}/experiments/models
FILES="$M/d/fire.nvfx $M/d/smoke.nvfx $M/d/explosion.nvfx"
for c in grid_s8 grid_m8 grid_mt8 grid_l8 conv_s8 conv_m8 grid_s16 grid_m16 grid_mt16 grid_l16 conv_s16 conv_m16; do FILES="$FILES $M/a/fire_0_$c.nvfx"; done
FILES="$FILES $M/a/smoke_0_grid_m8.nvfx $M/a/explosion_0_grid_m8.nvfx $M/b/fire_grid_k8.nvfx $M/c/fire_variation_k8.nvfx"
echo "load before: $(cat /proc/loadavg)"
taskset -c $CORE nice -n 10 $B/nvfx_pack --h3 --reps $REPS --out $OUT --csv h3_decode.csv $FILES
echo "load after: $(cat /proc/loadavg)"
# Segment sizes for seekable files: sizes and one-slice decode times (rollout effects and one 1 MB model)
for s in 4096 16384; do
  taskset -c $CORE nice -n 10 $B/nvfx_pack --h3 --reps 5 --segment $s --out $OUT --csv h3_segment_$s.csv $M/d/fire.nvfx $M/d/smoke.nvfx $M/d/explosion.nvfx $M/b/fire_grid_k8.nvfx
done
