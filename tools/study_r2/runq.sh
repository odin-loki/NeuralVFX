#!/bin/bash
# Study R2: run the tasks of a list (one nvfx_3d command line each) with N workers, each task once across every runner
# working on the same list (a task is claimed by making its lock directory). More runners may join at any time.
#   B=tool runq.sh LIST N
B=${B:?set B to the tool}
LIST=$1; N=${2:-2}
LOCKS=$LIST.locks
mkdir -p $LOCKS
. /root/nvfx-data/r2_smoke3d/jobs/disk.sh; need 0.6
cd /home/user/wt-r2
worker() {
  while IFS= read -r task; do
    [ -z "$task" ] && continue
    id=$(echo "$task" | sha1sum | cut -c1-16)
    mkdir $LOCKS/$id 2>/dev/null || continue
    echo "$(date -u +%H:%M:%S) start: $task"
    nice -n 10 $B $task --study r2 --threads 1 > /dev/null 2>> $LIST.err
    echo "$(date -u +%H:%M:%S) done ($?): $task"
  done < $LIST
}
for w in $(seq 1 $N); do worker & sleep 1; done
wait
echo "RUNQ_DONE $LIST"
