#!/bin/sh
# Re-time the effect manager's crowd benchmark (docs/MANAGER.md §7) on a quiet machine. Every configuration below runs
# ROUNDS times, interleaved (each round runs them all once, in order), each on its own pinned cores; then each figure's
# median over the rounds, with the least and the greatest (the spread).
#
#   tools/manager/time_crowd.sh [BUILD] [OUT] [ROUNDS] [SECONDS]
#
# Defaults: BUILD=build, OUT=/tmp/nvfx_crowd_timing, ROUNDS=3, SECONDS=20. The effects come from $NEURALVFX_DATA
# (default /root/nvfx-data): v2/models (fire, smoke, explosion) and fx/models/d (steam). Writes OUT/runs.csv (one line
# per run, every figure nvfx_crowd reports), OUT/<label>_r<round>.txt (its printout) and OUT/summary.csv; prints the
# summary. The machine has 4 cores: the one-thread runs use core 3, two threads cores 2-3, four threads all.
set -eu
BUILD=${1:-build}
OUT=${2:-/tmp/nvfx_crowd_timing}
ROUNDS=${3:-3}
SECS=${4:-20}
mkdir -p "$OUT"
rm -f "$OUT/runs.csv"

# label|options (160 instances at 60 Hz unless the options say otherwise)
CONFIGS='b2_t1|--budget 2 --threads 1 --cores 3
b4_t1|--budget 4 --threads 1 --cores 3
b4_t1_hold|--budget 4 --threads 1 --cores 3 --interp hold
b4_t1_crossfade|--budget 4 --threads 1 --cores 3 --interp crossfade
b4_t1_motion|--budget 4 --threads 1 --cores 3 --interp motion
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
    "$BUILD/nvfx_crowd" --seconds "$SECS" --label "$label" --summary "$OUT/runs.csv" $opts > "$OUT/${label}_r$r.txt"
  done
  r=$((r + 1))
done

# Per label and figure: the median, least and greatest over the rounds.
awk -F, '
NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
{
  l = $1; if (!(l in n)) { order[++labels] = l; n[l] = 0 }
  k = ++n[l]
  split("cpu_mean cpu_median cpu_p99 wall_median wall_p99 plan_mean overhead_mean sims proxied memory_mb rss_mb", f, " ")
  for (j in f) v[l, f[j], k] = $(col[f[j]])
  budget[l] = $(col["budget_ms"])
}
function stat(l, name,   m, i, j, t, a) {
  m = n[l]
  for (i = 1; i <= m; i++) a[i] = v[l, name, i] + 0
  for (i = 2; i <= m; i++) { t = a[i]; for (j = i - 1; j >= 1 && a[j] > t; j--) a[j + 1] = a[j]; a[j + 1] = t }
  return sprintf("%.2f (%.2f to %.2f)", (m % 2) ? a[(m + 1) / 2] : (a[m / 2] + a[m / 2 + 1]) / 2, a[1], a[m])
}
END {
  print "label,budget_ms,rounds,cpu_mean_ms,cpu_median_ms,cpu_p99_ms,wall_median_ms,wall_p99_ms,plan_ms,overhead_ms,sims,proxied,memory_mb,rss_mb"
  for (o = 1; o <= labels; o++) {
    l = order[o]
    printf "%s,%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", l, budget[l], n[l], stat(l, "cpu_mean"), stat(l, "cpu_median"), stat(l, "cpu_p99"),
           stat(l, "wall_median"), stat(l, "wall_p99"), stat(l, "plan_mean"), stat(l, "overhead_mean"), stat(l, "sims"), stat(l, "proxied"), stat(l, "memory_mb"), stat(l, "rss_mb")
  }
}' "$OUT/runs.csv" | tee "$OUT/summary.csv"
