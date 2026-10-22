#!/bin/bash
# Study R2 §4: the test, run once on the frozen model (results/3d/r2/frozen.csv): the truth, the real runs and the floor
# per setting, the edge check with the test's seeds, the chosen model's and the coarse simulation's tracking and plays,
# holding the first frame; then the comparisons and the verdict. Resumable (a run already written is skipped).
B=${B:?set B to the tool}
R=/root/nvfx-data/r2_smoke3d
J=${J:-4}
M=$R/models/chosen.nvfx3
F="--frozen results/3d/r2/frozen.csv"
. /root/nvfx-data/r2_smoke3d/jobs/disk.sh; need 0.8
cd /home/user/wt-r2
W="--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30"
{
  echo "edge --model $M --label learned --seed-base 34730000 $W --out $R/results/edge_test.csv"
  for s in 0 1 2 3 4 5 6 7; do echo "truth --split test --setting $s"; echo "real --split test --setting $s --no-floor"; done
} | xargs -P $J -I{} sh -c "nice -n 10 $B {} --study r2 --threads 1"
{
  for s in 0 1 2 3 4 5 6 7; do
    echo "track --split test --method learned --model $M $F --setting $s"; echo "play --split test --method learned --model $M $F --setting $s"
    echo "track --split test --method coarse --model $M --setting $s"; echo "play --split test --method coarse --model $M --setting $s"
    echo "track --split test --method hold --model $M --setting $s"; done
} | xargs -P $J -I{} sh -c "nice -n 10 $B {} --study r2 --threads 1"
cp $R/results/edge_test.csv results/3d/r2/edge_test.csv
$B edgecheck --study r2 --csv results/3d/r2/edge_test.csv --bounds results/3d/r2/edge_bounds.csv > results/3d/r2/edgecheck_test.md
$B compare --study r2 --split test --a learned --b coarse --results results/3d/r2 > /dev/null
$B compare --study r2 --split test --a learned --b hold --results results/3d/r2 > /dev/null
$B compare --study r2 --split test --a hold --b coarse --results results/3d/r2 > /dev/null
$B verdict --study r2 --split test --a learned --b coarse --margins results/3d/r2/margins.csv --bounds results/3d/r2/edge_bounds.csv \
   --edge-val results/3d/r2/edge_val.csv --val-label $(cat $R/models/chosen.txt) --edge-test results/3d/r2/edge_test.csv --out results/3d/r2/verdict_test.md
touch $R/markers/test_done; echo TEST_DONE
