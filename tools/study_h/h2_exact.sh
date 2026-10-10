#!/bin/sh
# Study H, H2: the fireball with and without skipping, on every ISA: every frame's picture checksum must be the same.
# Usage: h2_exact.sh BUILD_DIR OUT_DIR
B=${1:-build}; OUT=${2:-/tmp/h2}; M=${NEURALVFX_DATA:-/root/nvfx-data}/experiments/models/d
mkdir -p $OUT
for isa in avx2 baseline avx512; do
  for mode in skip no-skip; do
    flag=""; [ $mode = no-skip ] && flag="--no-skip"
    nice -n 10 $B/nvfx_fireball --models $M --no-video --threads 2 --isa $isa $flag --profile $OUT/exact_${isa}_$mode.csv > $OUT/exact_${isa}_$mode.log 2>&1
  done
  a=$(awk -F, 'NR>1{print $NF}' $OUT/exact_${isa}_skip.csv | md5sum); b=$(awk -F, 'NR>1{print $NF}' $OUT/exact_${isa}_no-skip.csv | md5sum)
  n=$(awk -F, 'NR>1' $OUT/exact_${isa}_skip.csv | wc -l)
  if [ "$a" = "$b" ]; then echo "$isa: all $n frames the same"; else echo "$isa: DIFFERENT"; fi
done
