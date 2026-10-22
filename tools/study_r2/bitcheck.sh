#!/bin/bash
# Study R2 §1.5: the fire, study N's smoke and study R's smoke_r recorded by main's tool (fc3d5f6) and by this branch's,
# compared byte for byte (2 s each at 64 x 96 x 64 from an empty grid, one run of study N2's changing controls too).
OLD=/root/nvfx-data/r2_smoke3d/bin/nvfx_3d_main_fc3d5f6
NEW=${NEW:-/root/nvfx-data/r2_smoke3d/bin/nvfx_3d_3cdee7c}
D=/root/nvfx-data/r2_smoke3d/scratch/bitcheck
for who in old new; do
  B=$OLD; [ $who = new ] && B=$NEW
  nice -n 10 $B record --study n --root $D/$who/fire --runs 1 --frames 60 --seed-base 34099200 --threads 2 > /dev/null
  nice -n 10 $B record --study n --effect smoke --root $D/$who/smoke --runs 1 --frames 60 --seed-base 34099201 --threads 2 > /dev/null
  nice -n 10 $B record --study r --effect smoke_r --root $D/$who/smoke_r --runs 1 --frames 60 --seed-base 34099202 --threads 2 > /dev/null
  nice -n 10 $B record --study n2 --root $D/$who/fire_n2 --runs 2 --frames 60 --seed-base 34099203 --threads 2 > /dev/null
done
for e in fire smoke smoke_r fire_n2; do
  for f in $(cd $D/old/$e/runs/train && ls); do
    if cmp -s $D/old/$e/runs/train/$f $D/new/$e/runs/train/$f; then echo "$e $f: identical ($(stat -c %s $D/new/$e/runs/train/$f) bytes)"; else echo "$e $f: DIFFERENT"; fi
  done
done
