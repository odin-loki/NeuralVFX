#!/bin/bash
# Study R2 §3: validation. The truth and the real runs per setting, then every method's tracking and plays (one process
# per method and setting, J at a time), the edge check of each candidate (validation's seeds), the comparisons and the
# choice. Resumable. Needs the rule committed (results/3d/r2: edge_bounds.csv, margins.csv) and the candidates.
B=${B:?set B to the tool}
R=/root/nvfx-data/r2_smoke3d
J=${J:-4}
. /root/nvfx-data/r2_smoke3d/jobs/disk.sh; need 0.8
cd /home/user/wt-r2
for s in 0 1 2 3 4 5; do echo "truth --split val --setting $s"; echo "real --split val --setting $s"; done | xargs -P $J -I{} sh -c "nice -n 10 $B {} --study r2 --threads 1"
W="--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30"
{
  for m in C2 C3 C4; do echo "edge --model $R/models/$m.nvfx3 --label $m $W --out $R/results/edge_val_$m.csv"; done
  for m in C2 C3 C4; do for s in 0 1 2 3 4 5; do
    echo "track --split val --method $m --model $R/models/$m.nvfx3 --setting $s"; echo "play --split val --method $m --model $R/models/$m.nvfx3 --setting $s"; done; done
  for s in 0 1 2 3 4 5; do
    echo "track --split val --method coarse --model $R/models/C2.nvfx3 --setting $s"; echo "play --split val --method coarse --model $R/models/C2.nvfx3 --setting $s"
    echo "track --split val --method hold --model $R/models/C2.nvfx3 --setting $s"; done
} | xargs -P $J -I{} sh -c "nice -n 10 $B {} --study r2 --threads 1"
head -1 $R/results/edge_val_C2.csv > $R/results/edge_val.csv
for m in C2 C3 C4; do tail -n +2 $R/results/edge_val_$m.csv >> $R/results/edge_val.csv; done
cp $R/results/edge_val.csv results/3d/r2/edge_val.csv
$B edgecheck --study r2 --csv results/3d/r2/edge_val.csv --bounds results/3d/r2/edge_bounds.csv > results/3d/r2/edgecheck_val.md
for m in C2 C3 C4; do $B compare --study r2 --split val --a $m --b coarse --results results/3d/r2 > /dev/null; done
$B compare --study r2 --split val --a C4 --b C3 --results results/3d/r2 > /dev/null
$B compare --study r2 --split val --a C4 --b hold --results results/3d/r2 > /dev/null
$B choose --study r2 --split val --cands C2,C3,C4 --edge results/3d/r2/edge_val.csv --bounds results/3d/r2/edge_bounds.csv --margins results/3d/r2/margins.csv > results/3d/r2/choice_val.md
touch $R/markers/val_done; echo VAL_DONE
