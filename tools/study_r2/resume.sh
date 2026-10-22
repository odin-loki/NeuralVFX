#!/bin/bash
# Study R2: restart everything that is not done, after a container restart or at any time (a job already running is
# left alone; every job skips what it has already written). Run it again whenever in doubt:
#   /root/nvfx-data/r2_smoke3d/jobs/resume.sh
# Stages, each behind the one before: the training runs; training and its candidates (markers/candidates_done); the
# teacher's runs for the rule and the plume measurement (jobs/teacher_tasks.txt); the rule's numbers (results/3d/r2,
# from the teacher's runs alone); validation (only once markers/rule_committed exists, made by hand when the rule's
# numbers are committed and pushed; then markers/val_done); the test (only once the chosen model is frozen; then
# markers/test_done). The choice, the freeze and every commit are by hand (STUDY_R2.md §0).
D=/root/nvfx-data/r2_smoke3d
B=$D/bin/nvfx_3d_cb31ccb   # training and the teacher's runs
BT=$D/bin/nvfx_3d_a5637ed  # validation and the test: the same solver and trainer (and real --no-floor)
WT=/home/user/wt-r2
cd $D
running() { pgrep -f -- "$1" > /dev/null; }  # patterns anchored at the program, so no shell that merely names a job counts
launch() { local log=$1; shift; echo "starting: $* (log $log)"; setsid nohup "$@" > "$log" 2>&1 < /dev/null & }
( . $D/jobs/disk.sh; need 0.6 ) || { echo "resume: the disk is short; nothing started"; exit 9; }

# Line-appended outputs a kill could have cut: keep only whole tracking rows, and only whole edge runs (61 rows each).
for f in $D/eval/*/track/*.csv; do
  [ -f "$f" ] || continue
  awk -F, 'NF >= 26' "$f" > "$f.clean" && mv "$f.clean" "$f"
done
for f in $D/results/edge_*.csv; do
  [ -f "$f" ] || continue
  awk -F, 'NR == FNR { if (FNR > 1 && NF >= 10) n[$1 "," $2]++; next } FNR == 1 || (NF >= 10 && n[$1 "," $2] >= 61)' "$f" "$f" > "$f.clean" && mv "$f.clean" "$f"
done

# 1. the training runs (24 + 24)
nl=$(ls runs/long/*.n3run 2>/dev/null | wc -l); ns=$(ls runs/short/*.n3run 2>/dev/null | wc -l)
if [ "$nl" -lt 24 ] || [ "$ns" -lt 24 ]; then
  running "^$D/bin/nvfx_3d_[0-9a-f]+ record --study r2" || launch logs/job_records.log env B=$B J=4 jobs/records.sh
  echo "training runs: $nl long, $ns short of 24 each; nothing after them started"; exit 0
fi

# 2. training and the candidates
if [ ! -f markers/candidates_done ]; then
  if running "^$D/bin/nvfx_3d_[0-9a-f]+ (train|finish) --study r2" || running "^/bin/bash jobs/train.sh"; then echo "training: running ($(tail -1 logs/train_r2.log))"
  else launch logs/job_train.log env B=$B T=3 jobs/train.sh; fi
fi

# 3. the teacher's runs (the edge bounds' 20, the margins' 45, the plume measurement's 18)
edge_ok=1; for j in 0 1 2 3 4; do [ "$(tail -n +2 results/edge_teacher_w$j.csv 2>/dev/null | wc -l)" -ge 244 ] || edge_ok=0; done
nu=$(ls eval/cal/endless/real/*.unit 2>/dev/null | wc -l)
np=0; for c in v_i50 v_i20 v_i80; do np=$((np + $(ls plume/$c/seed_*.bin 2>/dev/null | wc -l))); done
for c in v_t00 v_t100; do np=$((np + $(ls plume/$c/seed_*.bin 2>/dev/null | wc -l))); done
echo "teacher's runs: edge $edge_ok (all 20 whole), margins $nu of 45, plume $np of 18"
if [ $edge_ok -eq 0 ] || [ "$nu" -lt 45 ] || [ "$np" -lt 18 ]; then
  if running "^/bin/bash $D/jobs/runq.sh $D/jobs/teacher_tasks.txt"; then echo "teacher's runs: running"
  else rm -rf $D/jobs/teacher_tasks.txt.locks; launch logs/job_runq_teacher.log env B=$B $D/jobs/runq.sh $D/jobs/teacher_tasks.txt 3; fi
fi

# 4. the rule's numbers (from the teacher's runs alone), once their runs are whole
if [ $edge_ok -eq 1 ] && [ "$nu" -ge 45 ] && [ ! -f $WT/results/3d/r2/margins.csv ]; then
  env B=$B $D/jobs/rule_numbers.sh && echo "rule's numbers written: commit them (STUDY_R2 §2.10), push, then touch $D/markers/rule_committed"
fi

# 5. validation: after the rule is committed and the candidates exist
if [ -f markers/rule_committed ] && [ -f markers/candidates_done ] && [ ! -f markers/val_done ]; then
  running "^/bin/bash jobs/val.sh" || launch logs/job_val.log env B=$BT J=4 jobs/val.sh
fi

# 6. the test: once the chosen model is frozen (by hand: models/chosen.nvfx3, models/chosen.txt, results/3d/r2/frozen.csv)
if [ -f markers/val_done ] && [ -f models/chosen.nvfx3 ] && [ -f models/chosen.txt ] && [ -f $WT/results/3d/r2/frozen.csv ] && [ ! -f markers/test_done ]; then
  running "^/bin/bash jobs/test.sh" || launch logs/job_test.log env B=$BT J=4 jobs/test.sh
fi
ls markers
