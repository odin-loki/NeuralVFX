#!/usr/bin/env bash
# Re-time study K (docs/REPORT.md §13) on a quiet machine: rollout effects at float and at int8, everything interleaved
# (each round runs every configuration once, in order), then each figure's median over the rounds with the least and
# the greatest.
#
#   tools/study_k/time.sh [BUILD] [OUT] [ROUNDS] [SECONDS]
#
# Defaults: BUILD=build, OUT=/tmp/nvfx_study_k_timing, ROUNDS=5, SECONDS=20 (the crowd's). Effects from $NEURALVFX_DATA
# (default /root/nvfx-data). Steps (about 30 minutes at 5 rounds):
#   1. the eleven effects through nvfx_render, float and both int8 candidates, on AVX2 and AVX-512, at 64, 128 and 256
#      px, on core 3 (nvfx_study_k timing): OUT/effects.csv, and the table with the ratios: OUT/effects_summary.md;
#   2. where a frame's time goes (nvfx_study_k profile), float on AVX2 and AVX-512 and both candidates on AVX-512 and
#      AVX2, on core 3: OUT/profile.csv;
#   3. the fireball (study D's v1 models, as COMPOSE.md §7), float against int8, alternately, at 1280 x 720 on 4 threads
#      and on 1: OUT/fireball_*.csv and their medians (tools/opt2/summary.sh): OUT/fireball_summary.txt;
#   4. the effect manager's crowd (nvfx_crowd, as tools/manager/time_crowd.sh), float against int8, alternately, for
#      three configurations: OUT/crowd_runs.csv and OUT/crowd_summary.csv.
set -euo pipefail
BUILD=${1:-build}
OUT=${2:-/tmp/nvfx_study_k_timing}
ROUNDS=${3:-5}
SECS=${4:-20}
DATA=${NEURALVFX_DATA:-/root/nvfx-data}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
rm -f "$OUT/effects.csv" "$OUT/profile.csv" "$OUT/crowd_runs.csv"

echo "1. the eleven effects"
"$BUILD/nvfx_study_k" timing --rounds "$ROUNDS" --core 3 --root "$DATA" --out "$OUT" --csv effects.csv --label quiet
"$BUILD/nvfx_study_k" timing-summary --root "$DATA" --out "$OUT" --csv effects.csv --label quiet | tee "$OUT/effects_summary.md"

echo "2. the stages"
r=1
while [ "$r" -le "$ROUNDS" ]; do
  for isa in avx2 avx512; do
    "$BUILD/nvfx_study_k" profile --root "$DATA" --isa "$isa" --precision both --convs second --frames 200 --core 3 --out "$OUT/profile.csv" > /dev/null
    "$BUILD/nvfx_study_k" profile --root "$DATA" --isa "$isa" --precision int8 --convs both --frames 200 --core 3 --out "$OUT/profile.csv" > /dev/null
  done
  r=$((r + 1))
done
"$BUILD/nvfx_study_k" profile-summary --csv "$OUT/profile.csv" | tee "$OUT/profile_summary.md"

echo "3. the fireball"
r=1
while [ "$r" -le "$ROUNDS" ]; do
  for threads in 4 1; do
    for p in float int8; do
      "$BUILD/nvfx_fireball" --models "$DATA/experiments/models/d" --no-video --threads "$threads" --precision "$p" \
        --profile "$OUT/fireball_${p}_t${threads}_r$r.csv" > "$OUT/fireball_${p}_t${threads}_r$r.txt"
    done
  done
  r=$((r + 1))
done
"$HERE/../opt2/summary.sh" "$OUT"/fireball_*_r*.csv | tee "$OUT/fireball_summary.txt"

echo "4. the crowd"
CONFIGS='b4_t1|--budget 4 --threads 1 --cores 3
b4_t1_calibrated|--budget 4 --threads 1 --cores 3 --calibrate
unmanaged64_t1|--unmanaged 64 --cores 3'
r=1
while [ "$r" -le "$ROUNDS" ]; do
  printf '%s\n' "$CONFIGS" | while IFS='|' read -r label opts; do
    for p in float int8; do
      # shellcheck disable=SC2086
      "$BUILD/nvfx_crowd" --root "$DATA" --seconds "$SECS" --label "${label}_$p" --summary "$OUT/crowd_runs.csv" --precision "$p" $opts > "$OUT/crowd_${label}_${p}_r$r.txt"
    done
  done
  r=$((r + 1))
done
# Per label and figure: the median, least and greatest over the rounds (as tools/manager/time_crowd.sh).
awk -F, '
NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
{
  l = $1; if (!(l in n)) { order[++labels] = l; n[l] = 0 }
  k = ++n[l]
  split("cpu_mean cpu_median cpu_p99 wall_median wall_p99 plan_mean sims proxied memory_mb", f, " ")
  for (j in f) v[l, f[j], k] = $(col[f[j]])
}
function stat(l, name,   m, i, j, t, a) {
  m = n[l]
  for (i = 1; i <= m; i++) a[i] = v[l, name, i] + 0
  for (i = 2; i <= m; i++) { t = a[i]; for (j = i - 1; j >= 1 && a[j] > t; j--) a[j + 1] = a[j]; a[j + 1] = t }
  return sprintf("%.2f (%.2f to %.2f)", (m % 2) ? a[(m + 1) / 2] : (a[m / 2] + a[m / 2 + 1]) / 2, a[1], a[m])
}
END {
  print "label,rounds,cpu_mean_ms,cpu_median_ms,cpu_p99_ms,wall_median_ms,wall_p99_ms,plan_ms,sims,proxied,memory_mb"
  for (o = 1; o <= labels; o++) {
    l = order[o]
    printf "%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", l, n[l], stat(l, "cpu_mean"), stat(l, "cpu_median"), stat(l, "cpu_p99"), stat(l, "wall_median"),
           stat(l, "wall_p99"), stat(l, "plan_mean"), stat(l, "sims"), stat(l, "proxied"), stat(l, "memory_mb")
  }
}' "$OUT/crowd_runs.csv" | tee "$OUT/crowd_summary.csv"
