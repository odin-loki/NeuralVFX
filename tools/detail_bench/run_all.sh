#!/bin/sh
# The detail layer's measurements (docs/REPORT.md §19, results/detail/): every 2D model at 128, 256 and 512 px on AVX2
# and AVX-512, the 3D model at 1 and 4 threads, and the bit-identity runs, each old (OLD.so) against new (NEW.so) in one
# process. A step whose CSV exists is skipped, so the script resumes after an interruption; delete a CSV to redo it.
#   tools/detail_bench/run_all.sh OLD.so NEW.so [DATA] [OUT]
# DATA defaults to /root/nvfx-data, OUT to results/detail. Run from the repository root with the build in build/.
set -e
OLD=$1
NEW=$2
DATA=${3:-/root/nvfx-data}
OUT=${4:-results/detail}
BENCH=${BENCH:-build/nvfx_detail_bench}
MODELS=$DATA/library/models
VOL=$DATA/n2_3d/models/chosen.nvfx3
mkdir -p "$OUT"
run() {  # run CSV command...: the command writes CSV (--out); skipped when CSV exists
  csv=$1
  shift
  if [ -s "$OUT/$csv" ]; then
    echo "skip $csv (exists)"
    return
  fi
  echo "== $csv: $* (load: $(cut -d' ' -f1-3 /proc/loadavg))"
  nice -n 10 "$BENCH" "$@" --out "$OUT/$csv.part"
  mv "$OUT/$csv.part" "$OUT/$csv"
}
# timing (and every frame compared): 2D, per size so that a restart loses at most one size
for size in 128 256 512; do
  run frames_2d_$size.csv 2d --old "$OLD" --new "$NEW" --models "$MODELS" --sizes $size --isas avx2,avx512 --rounds 12 --frames 16
done
run volume_3d.csv 3d --old "$OLD" --new "$NEW" --model "$VOL" --threads 1,4 --isas avx2,avx512 --rounds 10 --frames 6
run detail_3d.csv 3d-detail --new "$NEW" --model "$VOL" --threads 1,4 --isas avx2,avx512 --rounds 10 --frames 6
# bit identity beyond the timing runs: the runners at int8, and pictures through the C API (shards, crossfades)
run identity_2d_int8.csv 2d --old "$OLD" --new "$NEW" --models "$MODELS" --sizes 256 --isas avx2,avx512 --precision int8 --rounds 2 --frames 30
run identity_api.csv api --old "$OLD" --new "$NEW" --models "$MODELS" --sizes 128 --isas avx2,avx512 --precision default --frames 300
# summaries
for f in frames_2d_128 frames_2d_256 frames_2d_512 volume_3d detail_3d identity_2d_int8; do
  "$BENCH" summary --in "$OUT/$f.csv" --out "$OUT/$f.md" > /dev/null
done
echo done
