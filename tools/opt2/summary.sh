#!/usr/bin/env bash
# Medians of an nvfx_fireball --profile CSV over the frames after the detonation (frame 36 on, when everything runs),
# one line per CSV:   tools/opt2/summary.sh run1.csv [run2.csv ...]
# Columns that a profile lacks (the code before the second optimisation round has no period) are left out; there,
# the frame time is `total` (the sum of the stages, which ran one after another).
set -euo pipefail
cols="step couple bus light particles_update shade background draw particles_draw distort bloom finish total period wait_images wait_picture frame_thread_cpu_ms step_thread_cpu_ms"
for csv in "$@"; do
  header=$(head -1 "$csv")
  line="$(basename "$csv")"
  for c in $cols; do
    idx=$(echo "$header" | tr ',' '\n' | grep -nx "$c" | cut -d: -f1 || true)
    [ -z "$idx" ] && continue
    med=$(awk -F, -v i="$idx" 'NR > 1 && $1 >= 36 { print $i }' "$csv" | sort -g | awk '{ v[NR] = $1 } END { if (NR == 0) print "nan"; else if (NR % 2) print v[(NR + 1) / 2]; else printf "%.3f\n", (v[NR / 2] + v[NR / 2 + 1]) / 2 }')
    line="$line $c=$med"
  done
  echo "$line"
done
