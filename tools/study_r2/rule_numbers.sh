#!/bin/bash
# Study R2 §2.10: the rule's numbers from the teacher's runs alone: the edge bounds (leave one out, as study R's) and the
# margins (the teacher against itself). Run once the edge and calibration runs exist.
B=${B:?set B to the tool}
D=/root/nvfx-data/r2_smoke3d
cd /home/user/wt-r2
for j in 0 1 2 3 4; do [ -f $D/results/edge_teacher_w$j.csv ] || { echo "missing edge_teacher_w$j.csv"; exit 1; }; done
head -1 $D/results/edge_teacher_w0.csv > $D/results/edge_teacher.csv
for j in 0 1 2 3 4; do tail -n +2 $D/results/edge_teacher_w$j.csv >> $D/results/edge_teacher.csv; done
cp $D/results/edge_teacher.csv results/3d/r2/edge_teacher.csv
$B edgecal --study r2 --csv results/3d/r2/edge_teacher.csv --out results/3d/r2/edge_bounds.csv > results/3d/r2/edge_calibration.md || exit 1
$B nullcal --study r2 --out results/3d/r2/margins.csv > /dev/null || exit 1
echo RULE_NUMBERS_DONE
