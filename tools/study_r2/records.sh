#!/bin/bash
# Study R2 §2.2: the training runs (24 of 18 s, N2's kinds, seeds 34,200,000 + i; 24 of 8 s at random controls held,
# seeds 34,100,000 + i). One process per run, J at a time; resumable.
B=${B:?set B to the tool}
. /root/nvfx-data/r2_smoke3d/jobs/disk.sh; need 1.8
cd /home/user/wt-r2
R="nice -n 10 $B record --study r2 --threads 1"
{
  for i in $(seq 0 23); do echo "--runs-dir long --from $i --runs $((i + 1))"; done
  for i in $(seq 0 23); do echo "--runs-dir short --from $i --runs $((i + 1))"; done
} | xargs -P ${J:-4} -I{} sh -c "$R {}"
echo RECORDS_DONE
