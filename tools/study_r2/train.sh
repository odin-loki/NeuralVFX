#!/bin/bash
# Study R2 §2.4: training (N2's four stages; stage 4's burn-ins 48 to 480 frames; the spread and edge losses from stage 2),
# then the candidates C2, C3, C4. Resumable from its checkpoints.
B=${B:?set B to the tool}
R=/root/nvfx-data/r2_smoke3d
. /root/nvfx-data/r2_smoke3d/jobs/disk.sh; need 0.8
cd /home/user/wt-r2
RUNS="--runs-dir long --also $R/runs/short"
nice -n 10 $B train --study r2 --name r2 $RUNS --iterations 900 --finetune 350 --activity 250 --long 250 --batch 8 --unroll 12 --burn 24 \
  --burn-long-min 48 --burn-long-max 360 --spread ${SPREAD:-2} --edge ${EDGE:-10} --seed 34200000 --threads ${T:-3} --checkpoint-every 10 || exit 1
for k in 2 3 4; do nice -n 10 $B finish --study r2 --ckpt r2_stage$k --out C$k $RUNS; done
touch $R/markers/candidates_done; echo TRAIN_DONE
