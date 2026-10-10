#!/bin/sh
# Study H, H2: per fireball run of h2_time.sh, the median frame time, step stage and summed module steps (ms), over
# all frames and over the frames after the detonation (1.2 s on, frame 36), and the thread CPU time summed over the
# frames (the main thread's: the whole frame at one thread; and the module steps'); one CSV line per run.
# Usage: h2_summary.sh OUT_DIR
OUT=${1:-/tmp/h2}
echo "run,threads,mode,pair,frames,median_total_ms,median_step_ms,median_module_step_cpu_ms,after_detonation_median_total_ms,after_detonation_median_step_ms,frame_thread_cpu_total_s,step_thread_cpu_total_s"
for f in $OUT/fb*_*.csv; do
  b=$(basename $f .csv); th=$(echo $b | sed 's/fb\([0-9]*\)_.*/\1/'); mode=$(echo $b | sed 's/fb[0-9]*_\(.*\)_[0-9]*$/\1/'); pair=${b##*_}
  awk -F, -v run=$b -v th=$th -v mode=$mode -v pair=$pair '
    NR == 1 { for (i = 1; i <= NF; ++i) { if ($i == "total") tc = i; if ($i == "step") sc = i; if ($i ~ /_step$/) ms[++nm] = i; if ($i == "frame_thread_cpu_ms") fc = i; if ($i == "step_thread_cpu_ms") pc = i } next }
    { n++; tot[n] = $tc; st[n] = $sc; s = 0; for (k = 1; k <= nm; ++k) s += $ms[k]; mod[n] = s; if ($1 >= 36) { m++; tot2[m] = $tc; st2[m] = $sc }; if (fc) fcpu += $fc; if (pc) pcpu += $pc }
    function median(a, k,   i, j, t) { for (i = 2; i <= k; ++i) { t = a[i]; for (j = i - 1; j >= 1 && a[j] > t; --j) a[j + 1] = a[j]; a[j + 1] = t } return k % 2 ? a[(k + 1) / 2] : (a[k / 2] + a[k / 2 + 1]) / 2 }
    END { printf "%s,%s,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", run, th, mode, pair, n, median(tot, n), median(st, n), median(mod, n), median(tot2, m), median(st2, m), fcpu / 1000, pcpu / 1000 }' $f
done
