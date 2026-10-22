#!/usr/bin/env bash
# The summaries of quiet round 7 (tools/quiet/round7.sh): from the session's folder OUT to DEST (results/quiet/round7).
#   tools/quiet/summary7.sh BUILD OUT DEST
# Medians over rounds come with a distribution-free 95% interval from order statistics (tools/quiet/stats.awk: with
# n rounds, (x_k, x_{n+1-k}) for the largest k with coverage >= 95%; the coverage is given); ratios over models are
# means or geometric means with a 95% bootstrap interval over the models (10,000 resamples).
set -euo pipefail
BUILD=$(cd "${1:?BUILD}" && pwd)
OUT=$(cd "${2:?OUT}" && pwd)
DEST=${3:?DEST}
HERE=$(cd "$(dirname "$0")" && pwd)
ST=$HERE/stats.awk
mkdir -p "$DEST"
R=$OUT/raw

cp "$OUT/blocks.csv" "$DEST/blocks.csv"

# --- 1. the 33 models (K: 64, 128, 256 px) and 12 at 512 px (L): rounds renumbered over the blocks ------------------
merge_k() {  # PREFIX OUTCSV: blocks PREFIX1, PREFIX2, ..., each with rounds 0.. -> one numbering over the blocks
  local p=$1 o=$2 b off=0 n
  echo "label,round,effect,size,isa,candidate,median_ms,mean_ms,scratch_kb" > "$o"
  for b in $(seq 1 18); do
    [ -s "$R/$p$b/timing.csv" ] || continue
    awk -F, -v off=$off 'NR > 1 { $2 = $2 + off; print }' OFS=, "$R/$p$b/timing.csv" >> "$o"
    n=$(awk -F, 'NR > 1 { r[$2] = 1 } END { print length(r) }' "$R/$p$b/timing.csv")
    off=$((off + n))
  done
}
merge_k K "$DEST/lib33.csv"
merge_k L "$DEST/lib512.csv"
for f in lib33 lib512; do
  "$BUILD/nvfx_study_k" timing-summary --out "$DEST" --csv "$f.csv" --label r7 > "$DEST/${f}_summary.md"
  # per configuration: the median over the rounds with its interval
  awk -F, -f "$ST" -f /dev/stdin "$DEST/$f.csv" > "$DEST/${f}_medians.csv" << 'EOF'
NR == 1 { next }
{ k = $3 "," $4 "," $5 "," $6; if (!(k in n)) order[++K] = k; v[k, ++n[k]] = $7 }
END {
  print "effect,size,isa,candidate,rounds,median_ms,ci_lo,ci_hi,coverage"
  for (i = 1; i <= K; i++) { k = order[i]; m = n[k]; for (j = 1; j <= m; j++) a[j] = v[k, j]; mci(a, m)
    printf "%s,%d,%.4f,%.4f,%.4f,%.3f\n", k, m, MED, LO, HI, COV }
}
EOF
done

# --- 2. the round documents' tables: d-timing, 9 rounds --------------------------------------------------------------
{
  echo "round,model,isa,size,median_ms,p90_ms,restart_ms,macs_px,stored_kb,resident_kb,scratch_kb"
  r=0
  for b in 1 2 3; do
    for i in 1 2 3; do
      f=$R/D$b/r$i/d_timing.csv
      [ -s "$f" ] || continue
      r=$((r + 1))
      awk -F, -v r=$r 'NR > 1 { print r "," $0 }' "$f"
    done
  done
} > "$DEST/d_timing_rounds.csv"
awk -F, -f "$ST" -f /dev/stdin "$DEST/d_timing_rounds.csv" > "$DEST/d_timing_summary.csv" << 'EOF'
NR == 1 { next }
{ k = $2 "," $3 "," $4; if (!(k in n)) order[++K] = k; c = ++n[k]; v[k, c] = $5; rs[k, c] = $7; kb[k] = $9 "," $10 "," $11 }
END {
  print "model,isa,size,rounds,median_ms,ci_lo,ci_hi,coverage,restart_ms_median,restart_lo,restart_hi,stored_kb,resident_kb,scratch_kb"
  for (i = 1; i <= K; i++) { k = order[i]; m = n[k]; for (j = 1; j <= m; j++) a[j] = v[k, j]; mci(a, m); line = sprintf("%s,%d,%.4f,%.4f,%.4f,%.3f", k, m, MED, LO, HI, COV)
    if (rs[k, 1] != "") { for (j = 1; j <= m; j++) a[j] = rs[k, j]; mci(a, m); line = line sprintf(",%.3f,%.3f,%.3f", MED, LO, HI) } else line = line ",,,"
    print line "," kb[k] }
}
EOF

# --- 3. the fireball: one line per run (tools/opt2/summary.sh), then per configuration and size --------------------
fbsum() {  # BLOCKS... -> lines "cfg size round frame step_cpu shade_cpu"
  local b f
  for b in "$@"; do
    for f in "$R/$b"/fireball_*_r*.csv; do
      [ -s "$f" ] || continue
      "$HERE/../opt2/summary.sh" "$f"
    done
  done
}
fbsum F1 F2 F3 F4 F5 E1 E2 > "$DEST/fireball_runs.txt"
awk -f "$ST" -f /dev/stdin "$DEST/fireball_runs.txt" > "$DEST/fireball_summary.csv" << 'EOF'
{
  name = $1; sub(/^fireball_/, "", name); sub(/\.csv$/, "", name)
  n = split(name, p, "_"); round = p[n]; sub(/^r/, "", round); size = p[n - 1]; cfg = p[1]; for (i = 2; i <= n - 2; i++) cfg = cfg "_" p[i]
  for (i = 2; i <= NF; i++) { split($i, kv, "="); val[kv[1]] = kv[2] }
  end = (round + 0 >= 11) ? "end" : "main"
  k = cfg "," size "," end; if (!(k in cnt)) order[++K] = k
  c = ++cnt[k]; fr[k, c] = val["frame"]; st[k, c] = val["model_step_cpu"]; sh[k, c] = val["model_shade_cpu"]; rd[k, c] = round
  byround[cfg, size, round] = val["frame"]
  rounds[size, round] = 1
}
END {
  print "config,size,set,runs,frame_ms_median,ci_lo,ci_hi,coverage,model_step_cpu_ms,model_shade_cpu_ms"
  for (i = 1; i <= K; i++) { k = order[i]; m = cnt[k]
    for (j = 1; j <= m; j++) a[j] = st[k, j]; mci(a, m); s = MED
    for (j = 1; j <= m; j++) a[j] = sh[k, j]; mci(a, m); h = MED
    for (j = 1; j <= m; j++) a[j] = fr[k, j]; mci(a, m)
    printf "%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", k, m, MED, LO, HI, COV, s, h }
  # paired ratios per round (main set): int8 / float, float / float on AVX2, int8 / float on AVX2
  print ""
  print "ratio,size,rounds,median,ci_lo,ci_hi,coverage"
  split("int8/float float/float_avx2 int8/float_avx2", rq, " ")
  for (q = 1; q <= 3; q++) { split(rq[q], ab, "/")
    for (size = 720; size <= 1080; size += 360) { m = 0
      for (key in rounds) { split(key, kk, SUBSEP); if (kk[1] != size || kk[2] + 0 >= 11) continue
        x = byround[ab[1], size, kk[2]]; y = byround[ab[2], size, kk[2]]; if (x != "" && y != "") a[++m] = x / y }
      if (m) { mci(a, m); printf "%s,%d,%d,%.4f,%.4f,%.4f,%.3f\n", rq[q], size, m, MED, LO, HI, COV } } }
}
EOF

# --- 4. the detail layer: the bench's own CSVs and summaries ---------------------------------------------------------
for b in X1 X2 X3 X4 X5 X6; do
  for f in "$R/$b"/*.csv "$R/$b"/*.md; do [ -e "$f" ] && cp "$f" "$DEST/detail_$(basename "$f")"; done
done

# --- 5. study P: per effect, size and arm, the median ms over the 30 rounds; each arm's ratio to N by round --------
if [ -s "$R/P1/p_cost.csv" ]; then
  cp "$R/P1/p_cost.csv" "$DEST/p_cost.csv"
  awk -F, -f "$ST" -f /dev/stdin "$R/P1/p_cost.csv" > "$DEST/p_cost_summary.csv" << 'EOF'
NR == 1 { next }
{ k = $2 "," $3; if (!(k in seen)) { seen[k] = 1; order[++K] = k } ms[k, $5, $4] = $6; if ($4 + 1 > nr[k]) nr[k] = $4 + 1 }
END {
  print "effect,size,arm,rounds,ms_median,ci_lo,ci_hi,ratio_to_N_median,ratio_lo,ratio_hi,coverage"
  split("N H B L", arms, " ")
  for (i = 1; i <= K; i++) { k = order[i]; m = nr[k]
    for (q = 1; q <= 4; q++) { x = arms[q]
      for (r = 0; r < m; r++) a[r + 1] = ms[k, x, r]; mci(a, m); line = sprintf("%s,%s,%d,%.4f,%.4f,%.4f", k, x, m, MED, LO, HI)
      for (r = 0; r < m; r++) a[r + 1] = ms[k, x, r] / ms[k, "N", r]; mci(a, m)
      printf "%s,%.4f,%.4f,%.4f,%.3f\n", line, MED, LO, HI, COV } }
}
EOF
fi

# --- 6. the 3D steps: nvfx_3d timing's rows, per model, row and thread count over the rounds ------------------------
{
  echo "model,round,what,threads,ms_median,ms_min,ms_max,n,load"
  for b in V1 V2 V3 V4 V5 V6; do
    for f in "$R/$b"/timing_*_r*.csv; do
      [ -s "$f" ] || continue
      n=$(basename "$f" .csv)
      m=${n#timing_}; m=${m%_r*}; r=${n##*_r}
      awk -F, -v m="$m" -v r="$r" 'NR > 1 { w = $1; for (i = 2; i <= NF - 6; i++) w = w ";" $i; printf "%s,%s,%s", m, r, w; for (i = NF - 5; i <= NF; i++) printf ",%s", $i; print "" }' "$f"
    done
  done
} > "$DEST/volume_timing.csv"
awk -F, -f "$ST" -f /dev/stdin "$DEST/volume_timing.csv" > "$DEST/volume_summary.csv" << 'EOF'
NR == 1 { next }
{ k = $1 "," $3 "," $4; if (!(k in n)) order[++K] = k; c = ++n[k]; v[k, c] = $5; mn[k, c] = $6 }
END {
  print "model,what,threads,rounds,ms_median,ci_lo,ci_hi,coverage,ms_min_median"
  for (i = 1; i <= K; i++) { k = order[i]; m = n[k]; for (j = 1; j <= m; j++) a[j] = mn[k, j]; mci(a, m); lo = MED
    for (j = 1; j <= m; j++) a[j] = v[k, j]; mci(a, m); printf "%s,%d,%.2f,%.2f,%.2f,%.3f,%.2f\n", k, m, MED, LO, HI, COV, lo }
}
EOF

# --- 7. the scenes: per run the median and mean frame period (frames 1 on) and the step stage; then over the rounds -
SCENE_RUN=$(mktemp -p "$OUT")
cat > "$SCENE_RUN" << 'EOF2'
NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
$1 >= 1 { m++; per[m] = $(c["period"]); stp[m] = $(c["step"]); tot[m] = 0; for (q = c["script"]; q <= c["finish"]; q++) tot[m] += $q; sum += $(c["period"]); al += $(c["allocations"]) }
END { mci(per, m); a = MED; mci(stp, m); b = MED; mci(tot, m); printf "%s,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%d\n", s, p, r, m, a, sum / m, b, MED, al }
EOF2
{
  echo "scene,precision,round,frames,period_median_ms,period_mean_ms,step_median_ms,total_median_ms,allocations"
  for b in S1 S2 S3 S4 W1 W2 W3 W4 W5 W6; do
    for f in "$R/$b"/*_r*.csv; do
      [ -s "$f" ] || continue
      n=$(basename "$f" .csv)
      r=${n##*_r}; rest=${n%_r*}; p=${rest##*_}; s=${rest%_*}
      awk -F, -v s="$s" -v p="$p" -v r="$r" -f "$ST" -f "$SCENE_RUN" "$f"
    done
  done
} > "$DEST/scenes_runs.csv"
rm -f "$SCENE_RUN"
awk -F, -f "$ST" -f /dev/stdin "$DEST/scenes_runs.csv" > "$DEST/scenes_summary.csv" << 'EOF'
NR == 1 { next }
{ k = $1 "," $2; if (!(k in n)) order[++K] = k; c = ++n[k]; md[k, c] = $5; mn[k, c] = $6; st[k, c] = $7; fr[k] = $4; al[k] += $9 }
END {
  print "scene,precision,runs,frames,period_median_ms,ci_lo,ci_hi,coverage,period_mean_ms_median,mean_lo,mean_hi,step_median_ms,allocations"
  for (i = 1; i <= K; i++) { k = order[i]; m = n[k]
    for (j = 1; j <= m; j++) a[j] = st[k, j]; mci(a, m); s = MED
    for (j = 1; j <= m; j++) a[j] = mn[k, j]; mci(a, m); x = MED; xl = LO; xh = HI
    for (j = 1; j <= m; j++) a[j] = md[k, j]; mci(a, m)
    printf "%s,%d,%d,%.2f,%.2f,%.2f,%.3f,%.2f,%.2f,%.2f,%.2f,%d\n", k, m, fr[k], MED, LO, HI, COV, x, xl, xh, s, al[k] }
}
EOF

# --- 7b. the lift-off scene (round7_extra.sh S4) as REPORT §16.6 measured it: frames from the ignition at 2.5 s (75) on
{
  echo "precision,round,frames,step_median_ms,period_median_ms,allocations"
  for f in "$R"/S4/liftoff_*_r*.csv; do
    [ -s "$f" ] || continue
    n=$(basename "$f" .csv); r=${n##*_r}; p=${n%_r*}; p=${p#liftoff_}
    awk -F, -v p="$p" -v r="$r" -f "$ST" -f /dev/stdin "$f" << 'EOF2'
NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
$1 >= 75 { m++; per[m] = $(c["period"]); stp[m] = $(c["step"]); al += $(c["allocations"]) }
END { mci(stp, m); s = MED; mci(per, m); printf "%s,%s,%d,%.3f,%.3f,%d\n", p, r, m, s, MED, al }
EOF2
  done
} > "$DEST/liftoff_runs.csv"

# --- 7c. REPORT §16.6: the second detail pass and the replaced files (round7_extra.sh O1, O2), as nvfx_ordered wrote them
for b in O1 O2; do
  for f in "$R/$b"/*.csv; do [ -e "$f" ] && cp "$f" "$DEST/$(basename "$f")"; done
done

# --- 8. the FUSE demo: each run's two plays (3 workers, the game thread alone) ---------------------------------------
{
  echo "run,workers,frames,wall_s,allocations,frame_errors,update_mean_ms,update_worst_ms,worst_at_s,volume_job_ms,volume_frames,volume_worst_ms,scene_ms,stage_ms,volume_step_ms,volume_picture_ms,volume_fields_ms,manager_jobs_ms,manager_planned_ms,upload_mb_frame,load_ms"
  for b in U1 U2 U3; do
    f=$R/$b/fuse.txt
    [ -s "$f" ] || continue
    awk -v run="$b" '
      / frames on .* workers? \+ the game thread in / { if (w != "") print line; w = $4; fr = $1; wall = $11; al = $13; err = $22
        line = "" }
      /update \(game thread, wall\)/ { gsub(/[(),;]/, " "); for (i = 1; i <= NF; i++) { if ($i == "mean" && !um) um = $(i + 1); if ($i == "worst" && !uw) { uw = $(i + 1); at = $(i + 4) } }
        for (i = 1; i <= NF; i++) { if ($i == "volume") { vj = $(i + 2); vf = $(i + 5); vw = $(i + 8) } if ($i == "scene") sc = $(i + 1); if ($i == "stage") sg = $(i + 1) } }
      /the volume.s job: step/ { gsub(/,/, " "); vs = $5; vp = $8; vfr = $12 }
      /manager: .* of jobs per tick/ { mj = $2; mp = $9 }
      /uploaded .* MB a frame/ { x = $0; sub(/.*\(/, "", x); sub(/ MB a frame.*/, "", x); up = x }
      /loaded in .* ms;/ { ld = $3
        line = sprintf("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s", run, w, fr, wall, al, err, um, uw, at, vj, vf, vw, sc, sg, vs, vp, vfr, mj, mp, up, ld)
        um = uw = "" }
      END { if (line != "") print line }' "$f"
  done
} > "$DEST/fuse_runs.csv"

echo "summaries in $DEST"
