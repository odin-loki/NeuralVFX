#!/bin/sh
# Drawn at half size where it pays (docs/MANAGER.md §2.10, docs/REPORT.md §22): the effect manager's crowd benchmark
# (docs/MANAGER.md §7) and its close-up scene (§7.2) with the policy off and on, interleaved A/B. Every round runs every
# configuration's three arms back to back, their order rotated from round to round, each on its own pinned cores; then
# `nvfx_crowd --ab` gives per configuration and arm the median over the rounds with a 95% bootstrap interval, and each arm
# against `off` round by round (the ratio and the difference of the CPU per tick).
#
#   tools/manager/time_halfsize.sh [BUILD] [OUT] [ROUNDS] [SECONDS]
#
# Defaults: BUILD=build, OUT=/tmp/nvfx_halfsize_timing, ROUNDS=10, SECONDS=20. The effects come from $NEURALVFX_DATA
# (default /root/nvfx-data), as for tools/manager/time_crowd.sh. Writes OUT/runs.csv (one line per run, labels CONFIG.ARM),
# OUT/<label>_r<round>.txt, OUT/ab_summary.csv and OUT/ab_summary.txt. The machine has 4 cores: one-thread runs use
# core 3, two threads cores 2-3, four threads all. The close scene: seed 35095001 (docs/REPORT.md §22's block) and four
# 256 px slots per effect (an engine that plays large effects sizes that pool; the default is one).
set -eu
BUILD=${1:-build}
OUT=${2:-/tmp/nvfx_halfsize_timing}
ROUNDS=${3:-10}
SECS=${4:-20}
mkdir -p "$OUT"
rm -f "$OUT/runs.csv"

# config|options (the crowd's 160 instances, or the close scene's 32, at 60 Hz)
CONFIGS='b4_t1|--budget 4 --threads 1 --cores 3
b8_t2|--budget 8 --threads 2 --cores 2,3
b16_t4|--budget 16 --threads 4 --cores 0,1,2,3
unlimited_t4|--unlimited --threads 4 --cores 0,1,2,3
close_b4_t1|--scene close --seed 35095001 --slots -1,-1,-1,4 --budget 4 --threads 1 --cores 3
close_b8_t2|--scene close --seed 35095001 --slots -1,-1,-1,4 --budget 8 --threads 2 --cores 2,3
close_unlimited_t4|--scene close --seed 35095001 --slots -1,-1,-1,4 --unlimited --threads 4 --cores 0,1,2,3'
# arm|options: off; on with the engine's sampler doing the 2x (the default mode); on with the manager's bilinear 2x
ARMS='off|--half-px 0
on|--half-px 256 --half-upscale off
cpu2x|--half-px 256 --half-upscale bilinear'

r=1
while [ "$r" -le "$ROUNDS" ]; do
  printf '%s\n' "$CONFIGS" | while IFS='|' read -r config copts; do
    [ -n "$config" ] || continue
    # the arms in an order rotated by the round
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
