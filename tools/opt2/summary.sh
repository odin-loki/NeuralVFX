#!/usr/bin/env bash
# Medians of an nvfx_fireball --profile CSV over the frames after the detonation (frame 36 on, when everything runs),
# one line per CSV:   tools/opt2/summary.sh run1.csv [run2.csv ...]
# `frame` is the frame time: the profile's `period` (the wall time between finished frames) where it has one, else
# `total` (the code before the second optimisation round: the sum of the stages, which ran one after another); with
# its 90th percentile, maximum and frames per second (1000 / median). Columns a profile lacks are left out.
set -euo pipefail
cols="step couple bus light particles_update shade background draw particles_draw distort bloom finish total period wait_images wait_picture frame_thread_cpu_ms step_thread_cpu_ms shade_thread_cpu_ms"
stat() {  # median, p90, max of a column from frame 36 on
  awk -F, -v i="$2" 'NR > 1 && $1 >= 36 { print $i }' "$1" | sort -g |
    awk '{ v[NR] = $1 } END { if (NR == 0) { print "nan nan nan"; exit } m = (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2;
                              p = v[int(0.9 * (NR - 1) + 0.5) + 1]; printf "%.3f %.3f %.3f\n", m, p, v[NR] }'
}
for csv in "$@"; do
  header=$(head -1 "$csv")
  idx() { echo "$header" | tr ',' '\n' | grep -nx "$1" | cut -d: -f1 || true; }
  line="$(basename "$csv")"
  f=$(idx period)
  [ -z "$f" ] && f=$(idx total)
  read -r m p x <<< "$(stat "$csv" "$f")"
  line="$line frame=$m p90=$p max=$x fps=$(awk -v m="$m" 'BEGIN { printf "%.1f", 1000 / m }')"
  # the modules' step and shading summed over the modules (each timed on whichever thread ran it): the CPU time of the
  # learned models, as §7's tables give it
  for kind in step shade; do
    m=$(awk -F, -v k="_$kind" 'NR == 1 { for (i = 1; i <= NF; i++) if (substr($i, length($i) - length(k) + 1) == k && $i != k) c[i] = 1; next }
                               $1 >= 36 { s = 0; for (i in c) s += $i; print s }' "$csv" | sort -g | awk '{ v[NR] = $1 } END { if (NR) printf "%.3f", (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2 }')
    line="$line model_${kind}_cpu=$m"
  done
  for c in $cols; do
    i=$(idx "$c")
    [ -z "$i" ] && continue
    read -r m _ _ <<< "$(stat "$csv" "$i")"
    line="$line $c=$m"
  done
  echo "$line"
done
