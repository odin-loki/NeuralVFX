#!/usr/bin/env bash
# Are the frames the same as the old build's, in every configuration and mode? For each configuration the old build
# writes its per-frame RGB checksums once, then the new build runs in each picture mode and thread count and is
# compared with it (tools/opt2/same_frames.sh).
#   tools/opt2/verify.sh OLD_BIN NEW_BIN OUT_DIR       (exit 0 when every run matches)
set -euo pipefail
old=$1 new=$2 out=$3
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
models=${MODELS:-${NEURALVFX_DATA:-$HOME/nvfx-data}/experiments/models/d}
configs=(
  "720|"
  "720_sse2|--isa baseline"
  "720_avx512|--isa avx512"
  "720_halftiles|--quality 0.5"
  "360|--width 640 --height 360"
  "1080|--width 1920 --height 1080"
)
modes=(
  "t1|--threads 1"
  "t1_stages|--threads 1 --stages"
  "t2|--threads 2"
  "t3|--threads 3"
  "t4|--threads 4"
  "t4_captured|--threads 4 --no-overlap"
  "t4_stages|--threads 4 --stages"
  "t4_baseline_kernels|--threads 4 --baseline-kernels"
)
fail=0
for c in "${configs[@]}"; do
  IFS='|' read -r name opts <<< "$c"
  # shellcheck disable=SC2086
  "$old" --models "$models" --no-video --threads 4 $opts --profile "$out/old_$name.csv" > /dev/null
  for m in "${modes[@]}"; do
    IFS='|' read -r mode mopts <<< "$m"
    [ "$name" != 720 ] && [ "$mode" != t4 ] && [ "$mode" != t1 ] && continue  # every mode at 720p; 1 and 4 threads elsewhere
    # shellcheck disable=SC2086
    "$new" --models "$models" --no-video $opts $mopts --profile "$out/new_${name}_$mode.csv" > /dev/null
    "$here/same_frames.sh" "$out/old_$name.csv" "$out/new_${name}_$mode.csv" || fail=1
  done
done
exit $fail
