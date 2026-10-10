#!/bin/sh
# Study H, H2: how empty the fireball's fine fields are (every third frame, every active module). Deterministic.
# Usage: h2_occupancy.sh BUILD_DIR OUT_CSV [CORE]
B=${1:-build}; OUT=${2:-/tmp/h2_occupancy.csv}; CORE=${3:-2}
taskset -c $CORE nice -n 10 $B/nvfx_fireball --models ${NEURALVFX_DATA:-/root/nvfx-data}/experiments/models/d --no-video --threads 1 --occupancy $OUT > /dev/null
