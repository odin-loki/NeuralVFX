#!/bin/sh
# Study H, H2: (1) the runtime's rollout step and learned renderer with skipping on and off (thread CPU time, least of
# REPS per frame, interleaved, fields and pictures compared to the bit); (2) the fireball with and without skipping,
# PAIRS interleaved runs at 1 thread pinned, and with THREADS4=1 also at 4 threads (the documented configuration; on a
# quiet machine only); SKIP_MICRO=1 runs the fireball pairs alone. The fireball's profiles carry each frame's thread CPU
# time (tools/study_h/h2_summary.sh sums it). Usage: [THREADS4=1] [SKIP_MICRO=1] h2_time.sh BUILD_DIR OUT_DIR [CORE] [REPS] [PAIRS]
B=${1:-build}; OUT=${2:-/tmp/h2}; CORE=${3:-2}; REPS=${4:-15}; PAIRS=${5:-5}
M=${NEURALVFX_DATA:-/root/nvfx-data}/experiments/models/d
mkdir -p $OUT
echo "load before: $(cat /proc/loadavg)"
[ "$SKIP_MICRO" = 1 ] || for isa in avx2 baseline avx512; do
  taskset -c $CORE nice -n 10 $B/nvfx_study_h --h2 --models $M --reps $REPS --frames 90 --isa $isa --out $OUT/h2_step_$isa.csv
done
echo "load after micro: $(cat /proc/loadavg)"
for p in $(seq 1 $PAIRS); do
  for mode in skip no-skip; do
    flag=""; [ $mode = no-skip ] && flag="--no-skip"
    taskset -c $CORE nice -n 10 $B/nvfx_fireball --models $M --no-video --threads 1 $flag --profile $OUT/fb1_${mode}_$p.csv > $OUT/fb1_${mode}_$p.log 2>&1
    [ "$THREADS4" = 1 ] && nice -n 10 $B/nvfx_fireball --models $M --no-video --threads 4 $flag --profile $OUT/fb4_${mode}_$p.csv > $OUT/fb4_${mode}_$p.log 2>&1
    echo "pair $p $mode: load $(cat /proc/loadavg)"
  done
done
