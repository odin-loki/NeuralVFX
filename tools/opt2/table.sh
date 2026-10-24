#!/usr/bin/env bash
# §7's table of configurations, before and after, interleaved: for each configuration, ROUNDS rounds of the old build
# then the new one. Prints the least over the rounds of each run's median frame time (frames 36 on), with the new
# build's p90, maximum, model step and shading CPU (thread CPU summed over modules) of that run, and the speed-up.
#   tools/opt2/table.sh OLD_BIN NEW_BIN OUT_DIR [ROUNDS (3)]
# OLD_BIN: nvfx_fireball of the code before (for the round-2 comparison, commit d41adff); NEW_BIN: this branch's.
# The new build runs with --no-checksums (the old one keeps its checksum outside its timed stages). Models: $MODELS
# (default $NEURALVFX_DATA/experiments/models/d). On a shared machine the numbers are provisional: the load average is
# written to OUT_DIR/runs.log before each run.
set -euo pipefail
old=$1 new=$2 out=$3 rounds=${4:-3}
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
configs=(
  "720_t4|--threads 4"
  "720_t2|--threads 2"
  "720_t1|--threads 1"
  "720_t4_sse2|--threads 4 --isa baseline"
  "720_t4_avx512|--threads 4 --isa avx512"
  "720_t4_halftiles|--threads 4 --quality 0.5"
  "360_t4|--threads 4 --width 640 --height 360"
  "1080_t4|--threads 4 --width 1920 --height 1080"
)
args=()
for c in "${configs[@]}"; do
  IFS='|' read -r name opts <<< "$c"
  args+=("old_$name|$old|$opts" "new_$name|$new|$opts --no-checksums")
done
"$here/time_fireball.sh" "$out" "$rounds" "${args[@]}" > "$out/summaries.txt"
field() { tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }
best() {  # the run with the least median frame time among the rounds: its summary line
  for r in $(seq 1 "$rounds"); do "$here/summary.sh" "$out/$1_$r.csv"; done | awk '{ split($2, a, "="); print a[2], $0 }' | sort -g | head -1 | cut -d' ' -f2-
}
printf "%-18s %9s %9s %8s %9s %9s %7s %10s %10s\n" config before_ms after_ms speedup after_p90 after_max fps step_cpu shade_cpu
for c in "${configs[@]}"; do
  IFS='|' read -r name _ <<< "$c"
  o=$(best "old_$name") n=$(best "new_$name")
  om=$(echo "$o" | field frame) nm=$(echo "$n" | field frame)
  printf "%-18s %9s %9s %7.2fx %9s %9s %7s %10s %10s\n" "$name" "$om" "$nm" "$(awk -v a="$om" -v b="$nm" 'BEGIN { print a / b }')" \
    "$(echo "$n" | field p90)" "$(echo "$n" | field max)" "$(echo "$n" | field fps)" "$(echo "$n" | field step_thread_cpu_ms)" "$(echo "$n" | field shade_thread_cpu_ms)"
done
