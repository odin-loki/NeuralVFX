#!/bin/sh
# Quick quiet re-timing of half size's saving (docs/MANAGER.md §7.2, docs/REPORT.md §22.8, both provisional: the ten-round
# A/B ran under other agents' load). The two configurations where the saving shows directly, with no budget (with one the
# planner spends it on other simulations): the close-up scene, where about five 256 px pictures are drawn at 128 at a
# time, and the crowd, where about one is. Two arms, off (--half-px 0) and on (--half-px 256, the engine's sampler doing
# the 2x: what an engine that sets nothing gets since NVFX_MANAGER_VERSION 3), back to back in each round, their order
# swapped from round to round; then `nvfx_crowd --ab` gives the medians, 95% bootstrap intervals and on / off per round.
# Before each round the one-minute load average is checked, and the round waits until it is at most MAX_LOAD, so that
# the figures are a quiet machine's; the load at each round's start is written to OUT/load.txt beside the runs.
#
#   tools/manager/time_half_128.sh [BUILD] [OUT] [ROUNDS] [SECONDS]
#
# Defaults: BUILD=build, OUT=/tmp/nvfx_half_128_timing, ROUNDS=5, SECONDS=10 (about 2 x 2 x 5 x 10 s = 3.5 minutes). The
# effects come from $NEURALVFX_DATA (default /root/nvfx-data; v2/models fire, smoke, explosion and fx/models/d steam), as
# for time_crowd.sh and time_halfsize.sh. Environment: CORES (default 0,1,2,3: the four cores every run is pinned to;
# the project's machine has four, set e.g. CORES=60,61,62,63 on a larger one), MAX_LOAD (default 1.0), MAX_SIZE (default
# 256: 256 px pictures drawn at 128, the level measured; MAX_SIZE=128 times 128 px pictures drawn at 64, which REPORT.md
# §22.9 lists as not measured: with --max-size 128 and --half-px 128 every 128 px picture is a candidate). Writes
# OUT/runs.csv (one line per run, labels CONFIG.ARM), OUT/<label>_r<round>.txt, OUT/ab_summary.csv and OUT/ab_summary.txt.
# For the full picture (budgets, the manager's own 2x, one thread and two) run tools/manager/time_halfsize.sh instead.
set -eu
BUILD=${1:-build}
OUT=${2:-/tmp/nvfx_half_128_timing}
ROUNDS=${3:-5}
SECS=${4:-10}
CORES=${CORES:-0,1,2,3}
MAX_LOAD=${MAX_LOAD:-1.0}
MAX_SIZE=${MAX_SIZE:-256}
HALF_PX=$MAX_SIZE
mkdir -p "$OUT"
rm -f "$OUT/runs.csv" "$OUT/load.txt"

# config|options (no budget, four threads on the pinned cores; the close scene as in time_halfsize.sh: seed 35095001 and
# four 256 px slots per effect)
CONFIGS="close_unlimited_t4|--scene close --seed 35095001 --slots -1,-1,-1,4 --unlimited --threads 4 --cores $CORES --max-size $MAX_SIZE
unlimited_t4|--unlimited --threads 4 --cores $CORES --max-size $MAX_SIZE"
ARMS="off|--half-px 0
on|--half-px $HALF_PX --half-upscale off"

load1() { awk '{ print $1 }' /proc/loadavg; }

r=1
while [ "$r" -le "$ROUNDS" ]; do
  while [ "$(awk -v l="$(load1)" -v m="$MAX_LOAD" 'BEGIN { print (l > m) ? 1 : 0 }')" = 1 ]; do
    echo "round $r: load $(load1) above $MAX_LOAD, waiting 30 s"
    sleep 30
  done
  echo "round $r load $(load1)" | tee -a "$OUT/load.txt"
  printf '%s\n' "$CONFIGS" | while IFS='|' read -r config copts; do
    [ -n "$config" ] || continue
    # the arms in an order swapped by the round
    printf '%s\n' "$ARMS" | awk -v r="$r" '{ a[NR - 1] = $0 } END { for (i = 0; i < NR; i++) print a[(i + r) % NR] }' | while IFS='|' read -r arm aopts; do
      label="$config.$arm"
      echo "round $r: $label"
      # shellcheck disable=SC2086
      "$BUILD/nvfx_crowd" --seconds "$SECS" --label "$label" --summary "$OUT/runs.csv" $copts $aopts > "$OUT/${label}_r$r.txt"
    done
  done
  r=$((r + 1))
done

"$BUILD/nvfx_crowd" --ab "$OUT/runs.csv" --out "$OUT/ab_summary.csv" | tee "$OUT/ab_summary.txt"
