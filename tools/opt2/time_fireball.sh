#!/usr/bin/env bash
# Interleaved timing runs of nvfx_fireball builds, for before/after comparisons on a shared machine: each round runs
# every configuration once, in turn, and the load average is recorded before each run.
#   tools/opt2/time_fireball.sh OUT_DIR ROUNDS "label|path/to/nvfx_fireball|options" ...
# e.g.
#   tools/opt2/time_fireball.sh /tmp/t 3 "before|$OLD/nvfx_fireball|--threads 4" "after|build/nvfx_fireball|--threads 4 --no-checksums"
# Writes OUT_DIR/<label>_<round>.csv (with --profile; --no-video is added) and OUT_DIR/runs.log, then prints
# tools/opt2/summary.sh of every run. Models: $MODELS (default $NEURALVFX_DATA/experiments/models/d).
set -euo pipefail
out=$1 rounds=$2
shift 2
models=${MODELS:-${NEURALVFX_DATA:-$HOME/nvfx-data}/experiments/models/d}
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
for r in $(seq 1 "$rounds"); do
  for cfg in "$@"; do
    IFS='|' read -r label bin opts <<< "$cfg"
    load=$(cut -d' ' -f1 /proc/loadavg)
    # shellcheck disable=SC2086
    "$bin" --models "$models" --no-video $opts --profile "$out/${label}_$r.csv" > "$out/${label}_$r.log"
    echo "$label round $r load $load $(date +%T)" | tee -a "$out/runs.log"
  done
done
for cfg in "$@"; do
  IFS='|' read -r label _ _ <<< "$cfg"
  for r in $(seq 1 "$rounds"); do "$here/summary.sh" "$out/${label}_$r.csv"; done
done
