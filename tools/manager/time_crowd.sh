#!/bin/sh
# Re-time the effect manager crowd benchmark (docs/MANAGER.md section 7) on a quiet machine.
#
#   tools/manager/time_crowd.sh [BUILD] [OUT] [ROUNDS] [SECONDS]
#
# Defaults: BUILD=build, OUT=/tmp/nvfx_crowd_timing, ROUNDS=3, SECONDS=20.
# Every run has half size off (--half-px 0); see tools/manager/time_halfsize.sh for half-size A/B.
set -eu
BUILD=${1:-build}
OUT=${2:-/tmp/nvfx_crowd_timing}
ROUNDS=${3:-3}
SECS=${4:-20}
mkdir -p "$OUT"
rm -f "$OUT/runs.csv"

CONFIGS='b2_t1|--budget 2 --threads 1 --cores 3
b4_t1|--budget 4 --threads 1 --cores 3
b4_t1_hold|--budget 4 --threads 1 --cores 3 --interp hold
b4_t1_crossfade|--budget 4 --threads 1 --cores 3 --interp crossfade
b4_t1_motion|--budget 4 --threads 1 --cores 3 --interp motion
b4_t1_calibrated|--budget 4 --threads 1 --cores 3 --calibrate
b8_t2|--budget 8 --threads 2 --cores 2,3
b16_t4|--budget 16 --threads 4 --cores 0,1,2,3
unlimited_t4|--unlimited --threads 4 --cores 0,1,2,3
unmanaged64_t1|--unmanaged 64 --cores 3'

r=1
while [ "$r" -le "$ROUNDS" ]; do
  printf '%s\n' "$CONFIGS" | while IFS='|' read -r label opts; do
    [ -n "$label" ] || continue
    echo "round $r: $label"
    # shellcheck disable=SC2086
    "$BUILD/nvfx_crowd" --seconds "$SECS" --label "$label" --summary "$OUT/runs.csv" --half-px 0 $opts > "$OUT/${label}_r$r.txt"
  done
  r=$((r + 1))
done

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
awk -F, -f "$SCRIPT_DIR/time_crowd_summary.awk" "$OUT/runs.csv" | tee "$OUT/summary.csv"
