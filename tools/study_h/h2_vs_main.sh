#!/bin/sh
# Study H, H2: the rollout step of main's runtime against this branch's (skipping on), separate processes alternated,
# thread CPU time of 90 steps from start point 0, REPS runs each. AB_MAIN and AB_NEW are tools/study_h/ab_step.cpp
# built against main's and this branch's libraries (see its header; main's sources: git archive <main commit>). Usage: h2_vs_main.sh AB_MAIN AB_NEW OUT_CSV [CORE] [REPS]
A=$1; N=$2; OUT=$3; CORE=${4:-2}; REPS=${5:-15}; D=${NEURALVFX_DATA:-/root/nvfx-data}/experiments/models/d
: > $OUT
for cfg in "fire.nvfx 192" "fire.nvfx 128" "explosion.nvfx 384" "explosion.nvfx 256" "smoke.nvfx 384"; do
  set -- $cfg
  for i in $(seq 1 $REPS); do
    taskset -c $CORE nice -n 10 $A $D/$1 $2 0 90 | sed "s/^0/main/" | awk -v c="$1,$2" '{print c "," $1 "," $2}' >> $OUT
    taskset -c $CORE nice -n 10 $N $D/$1 $2 1 90 | sed "s/^1/study-h/" | awk -v c="$1,$2" '{print c "," $1 "," $2}' >> $OUT
  done
done
