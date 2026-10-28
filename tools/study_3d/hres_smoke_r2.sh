#!/usr/bin/env bash
# Study hres, smoke on study R2's teacher (docs/STUDY_3D_HRES.md §10; the plan of §8): smoke_r2 (momentum fixed,
# STUDY_R2.md §1) recorded at 64 x 96 x 64 and block-averaged to the 32 x 48 x 32 coarse grid, trained with study N2's
# four stages, stage 4's long burn-ins (48 to 360 frames) and study R2's two losses (--spread 2 --edge 10; STUDY_R2.md
# §2.4), on the CPU or CUDA (CUDA: MSE/profile/activity on GPU; spread/edge on CPU in the same step). Then the
# candidates C2 to C4 and the final finish, the edge bounds calibrated on four teacher runs per wind, and the
# candidates' 30 s edge check judged against those bounds. Its own seed block (the tool's defaults for
# --study hres --effect smoke_r2): training runs 9,630,000 + i and 9,631,000 + i (long), initial weights 9,630,100,
# trainer 9,630,000, the bounds' teacher runs 9,633,000 + 100 j + k, edge-check plays 9,662,900 + j.
#
# Usage (repo root):
#   bash tools/study_3d/hres_smoke_r2.sh            # the plan and what is there; starts nothing
#   FORCE=1 bash tools/study_3d/hres_smoke_r2.sh    # runs every stage not done (hours to days on a CPU)
#   FORCE=1 STAGE=record,train,finish bash tools/study_3d/hres_smoke_r2.sh   # some stages only (record|train|finish|bounds|edge)
# Env: NEURALVFX_DATA (data root, default $HOME/nvfx-data; this study's files under hres_3d/smoke_r2), HRES_R2_ROOT
#   (that folder, if elsewhere; HRES_SMOKE_R2_ROOT is the same), BUILD or NVFX_3D (the tool; default build/nvfx_3d
#   through nvfx_bin_link.sh), T (training threads, 4; TTRAIN is the same), J (recording processes at a time, 4, each
#   on one thread; TREC is the same), NICE_LEVEL (10), SPREAD and EDGE
#   B and C; the hres fire's 4 and 2 by default: STUDY_3D_HRES.md §10), TRAIN_DEVICE (cpu or cuda; cuda uses
#   build-cuda/nvfx_3d and TRAIN_SHARDS, spread/edge on CPU in the CUDA trainer), BUILD_DIR (build-cuda when cuda).
# Resumable: a run already recorded is skipped by the tool, training resumes from ckpt/smoke_r2.ckpt (a checkpoint
# every 10 iterations), a candidate already written is kept, teacher edge runs already written are skipped. Never
# starts a second trainer on this data root (logs/train.lock and a pgrep).
#
# Layout: main(), then one `main "$@"` line, so an edit to this file while a run is under way cannot corrupt it
# (hres_fire_validate.sh's lesson, STUDY_3D_HRES.md §7). After any edit: bash -n tools/study_3d/hres_smoke_r2.sh
set -euo pipefail

main() {
  # STAGE, T and J come from the environment and are not declared local (a `local` would shadow them with empty values)
  local REPO DATA ROOT R B H W RUNS HARGS WARGS RUNSARGS TOTAL_IT nshort nlong k last
  REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
  [ -f "$REPO/tools/study_3d/nvfx_bin_link.sh" ] || { echo "error: repo not found at $REPO (running a copy? set NVFX_REPO)" >&2; return 1; }
  cd "$REPO"
  DATA=${NEURALVFX_DATA:-$HOME/nvfx-data}
  ROOT=${HRES_R2_ROOT:-${HRES_SMOKE_R2_ROOT:-$DATA/hres_3d/smoke_r2}}
  R=results/3d/hres/smoke_r2
  HARGS=(--study hres --root "$ROOT" --effect smoke_r2)
  WARGS=(--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30)
  RUNSARGS=(--runs-dir train --also "$ROOT/runs/long")
  H="--study hres --root $ROOT --effect smoke_r2"
  W="--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30"
  RUNS="--runs-dir train --also $ROOT/runs/long"
  T=${T:-${TTRAIN:-4}}
  J=${J:-${TREC:-4}}
  NICE=${NICE_LEVEL:-10}
  STAGE=${STAGE:-all}
  TOTAL_IT=1750  # 900 + 350 + 250 + 250 (STUDY_3D_HRES.md §3)
  for k in ${STAGE//,/ }; do
    case "$k" in all|record|train|finish|bounds|edge) ;; *) echo "error: STAGE must be all or a comma list of record, train, finish, bounds, edge (got '$k')" >&2; return 1;; esac
  done
  TRAIN_DEVICE=${TRAIN_DEVICE:-cpu}
  TRAIN_SHARDS=${TRAIN_SHARDS:-4}

  nshort=$(count_runs "$ROOT/runs/train"); nlong=$(count_runs "$ROOT/runs/long")
  last=$(grep '^it ' "$ROOT/logs/train_smoke_r2.log" 2>/dev/null | awk '!seen[$0]++' | tail -1 || true)
  echo "=== hres smoke_r2 $(date -Iseconds) data $ROOT results $R stage $STAGE ==="
  echo "runs: $nshort of 40 short (240 frames), $nlong of 24 long (540 frames)"
  echo "train: ${last:-not started} $(train_done "$ROOT" "$TOTAL_IT" && echo '(DONE)' || true)"
  for k in C2 C3 C4 smoke_r2_hres; do [ -f "$ROOT/models/$k.nvfx3" ] && echo "model: $ROOT/models/$k.nvfx3"; done
  [ -f "$R/edge_bounds.csv" ] && echo "bounds: $R/edge_bounds.csv"
  [ -f "$R/edgecheck.md" ] && echo "edge check: $R/edgecheck.md"
  if [ "${FORCE:-0}" != 1 ]; then
    cat <<EOF
plan (nothing started; FORCE=1 runs the stages not done, STAGE=record,train,... some of them):
  1. record   $nshort/40 + $nlong/24 teacher runs of smoke_r2 at 64 x 96 x 64 (seeds 9,630,000 + i; long 9,631,000 + i), $J at a time
  2. train    nvfx_3d train $H --name smoke_r2 $RUNS
                --iterations 900 --finetune 350 --activity 250 --long 250 --batch 8 --unroll 12 --burn 24
                --burn-long-min 48 --burn-long-max 360 --spread ${SPREAD:-2} --edge ${EDGE:-10} --checkpoint-every 10
              (cuda: --device cuda --shards $TRAIN_SHARDS --threads 1 and BUILD_DIR/build-cuda; cpu: --device cpu --threads $T)
              (trainer seed 9,630,000 and initial weights 9,630,100: the tool's defaults for --study hres --effect smoke_r2)
  3. finish   C2, C3, C4 from ckpt/smoke_r2_stage{2,3,4}.ckpt and smoke_r2_hres from ckpt/smoke_r2.ckpt
  4. bounds   nvfx_3d edge $H --sim --runs ${EDGE_RUNS:-4} $W -> $R/edge_teacher.csv; edgecal --a 2 --b ${EDGE_B:-4} --c ${EDGE_C:-2} -> $R/edge_bounds.csv
  5. edge     the candidates' 30 s edge check (plays 9,662,900 + j) against the bounds -> $R/edge.csv, edgecheck.md
status: bash tools/study_3d/hres_status.sh
EOF
    return 0
  fi

  B=$(tool)
  mkdir -p "$ROOT/logs" "$ROOT/runs" "$ROOT/ckpt" "$ROOT/models" "$R"
  exec 8>"$ROOT/logs/train.lock"
  if ! flock -n 8; then echo "hres smoke_r2: another hres_smoke_r2.sh holds $ROOT/logs/train.lock"; return 0; fi
  if pgrep -f "nvfx_3d[^ ]* (train|finish|record) --study hres --root $ROOT" >/dev/null 2>&1; then
    echo "hres smoke_r2: a tool is already running on $ROOT ($(pgrep -af "nvfx_3d[^ ]* (train|finish|record) --study hres --root $ROOT" | head -1)); not starting another"
    return 0
  fi
  echo "tool $B; train_device $TRAIN_DEVICE; threads $T; J $J; nice $NICE; stage $STAGE"

  # 1. the teacher's runs (the tool skips a run file that exists); one process per run, J at a time, each on one thread
  if want record; then
    if [ "$nshort" -lt 40 ] || [ "$nlong" -lt 24 ]; then
      echo "--- recording ($nshort/40 short, $nlong/24 long; $J at a time) ---"
      {
        for k in $(seq 0 39); do echo "record --runs-dir train --from $k --runs $((k + 1)) --frames 240"; done
        for k in $(seq 0 23); do echo "record --runs-dir long --from $k --runs $((k + 1)) --frames 540"; done
      } | xargs -P "$J" -I{} sh -c 'nice -n "$1" "$2" {} --study hres --root "$3" --effect smoke_r2 --threads 1' _ "$NICE" "$B" "$ROOT"
      nshort=$(count_runs "$ROOT/runs/train"); nlong=$(count_runs "$ROOT/runs/long")
      [ "$nshort" -ge 40 ] && [ "$nlong" -ge 24 ] || { echo "error: recording left $nshort short, $nlong long runs" >&2; return 1; }
      echo "RECORDS_DONE $(date -Iseconds) ($nshort short, $nlong long)"
    fi
  fi

  # 2. training, resumable from ckpt/smoke_r2.ckpt; the tool writes logs/train_smoke_r2.log itself
  if want train; then
    [ "$nshort" -ge 40 ] && [ "$nlong" -ge 24 ] || { echo "error: record first ($nshort/40 short, $nlong/24 long)" >&2; return 1; }
    if train_done "$ROOT" "$TOTAL_IT"; then
      echo "--- train done at iteration $TOTAL_IT ---"
    else
      echo "--- training smoke_r2 (log $ROOT/logs/train_smoke_r2.log; last: ${last:-none}) ---"
      export NVFX_TRAIN_QUIET_STDOUT=1
      # shellcheck disable=SC2086
      TRAIN_EXTRA=(--device cpu --threads "$T")
      if [ "$TRAIN_DEVICE" = cuda ]; then TRAIN_EXTRA=(--device cuda --shards "$TRAIN_SHARDS" --threads 1); fi
      nice -n "$NICE" "$B" train "${HARGS[@]}" --name smoke_r2 "${RUNSARGS[@]}" --iterations 900 --finetune 350 --activity 250 --long 250 --batch 8 --unroll 12 --burn 24 \
        --burn-long-min 48 --burn-long-max 360 --spread "${SPREAD:-2}" --edge "${EDGE:-10}" "${TRAIN_EXTRA[@]}" --checkpoint-every 10
      unset NVFX_TRAIN_QUIET_STDOUT
      train_done "$ROOT" "$TOTAL_IT" || { echo "error: training ended without 'done at iteration $TOTAL_IT' (see $ROOT/logs/train_smoke_r2.log)" >&2; return 1; }
      echo "TRAIN_DONE $(date -Iseconds)"
    fi
  fi

  # 3. the candidates (the stepper after stages 2, 3 and 4) and the final finish
  if want finish; then
    for k in 2 3 4; do
      if [ -f "$ROOT/models/C$k.nvfx3" ]; then continue; fi
      [ -f "$ROOT/ckpt/smoke_r2_stage$k.ckpt" ] || { echo "error: missing $ROOT/ckpt/smoke_r2_stage$k.ckpt" >&2; return 1; }
      nice -n "$NICE" "$B" finish "${HARGS[@]}" --ckpt "smoke_r2_stage$k" --out "C$k" "${RUNSARGS[@]}"
    done
    if [ ! -f "$ROOT/models/smoke_r2_hres.nvfx3" ]; then
      [ -f "$ROOT/ckpt/smoke_r2.ckpt" ] || { echo "error: missing $ROOT/ckpt/smoke_r2.ckpt" >&2; return 1; }
      nice -n "$NICE" "$B" finish "${HARGS[@]}" --ckpt smoke_r2 --out smoke_r2_hres "${RUNSARGS[@]}"
    fi
    echo "FINISH_DONE $(date -Iseconds): $ROOT/models/C2.nvfx3 C3 C4 smoke_r2_hres"
  fi

  # 4. the edge bounds from the teacher's own runs (four per wind; study R2 §2.7, as hres_fire_edge_bounds.sh), before any
  # model is judged. edge --sim --runs K writes a run's rows at once and skips runs already in the file.
  if want bounds; then
    if [ -f "$R/edge_bounds.csv" ]; then
      echo "--- bounds present: $R/edge_bounds.csv (delete it and edge_teacher.csv to calibrate again) ---"
    else
      echo "--- teacher edge runs (${EDGE_RUNS:-4} per wind) -> $R/edge_teacher.csv ---"
      nice -n "$NICE" "$B" edge "${HARGS[@]}" --sim --runs "${EDGE_RUNS:-4}" "${WARGS[@]}" --threads "$T" --out "$R/edge_teacher.csv"
      "$B" edgecal "${HARGS[@]}" --csv "$R/edge_teacher.csv" --a 2 --b "${EDGE_B:-4}" --c "${EDGE_C:-2}" --out "$R/edge_bounds.csv" | tee "$R/edge_calibration.md"
      echo "BOUNDS_DONE $(date -Iseconds): $R/edge_bounds.csv"
    fi
  fi

  # 5. the candidates' edge check against the bounds (a model's CSV is appended to by the tool: written afresh here)
  if want edge; then
    [ -f "$R/edge_bounds.csv" ] || { echo "error: no $R/edge_bounds.csv (STAGE=bounds first)" >&2; return 1; }
    for k in C2 C3 C4; do
      [ -f "$ROOT/models/$k.nvfx3" ] || { echo "error: missing $ROOT/models/$k.nvfx3 (STAGE=finish first)" >&2; return 1; }
      rm -f "$R/edge_$k.csv"
      nice -n "$NICE" "$B" edge "${HARGS[@]}" --model "$ROOT/models/$k.nvfx3" --label "$k" "${WARGS[@]}" --threads "$T" --out "$R/edge_$k.csv"
    done
    combine_edge "$R/edge.csv" "$R/edge_teacher.csv" "$R/edge_C2.csv" "$R/edge_C3.csv" "$R/edge_C4.csv"
    "$B" edgecheck "${HARGS[@]}" --csv "$R/edge.csv" --bounds "$R/edge_bounds.csv" | tee "$R/edgecheck.md"
    echo "EDGE_DONE $(date -Iseconds): $R/edgecheck.md"
  fi
  echo "next (STUDY_3D_HRES.md §10): if a candidate passes, the held-out validation and test with --study hres --effect smoke_r2 (seeds 9,660,000 and 9,870,000 blocks)"
}

tool() {  # NVFX_3D, else build or build-cuda/nvfx_3d (BUILD / BUILD_DIR) through nvfx_bin_link.sh
  if [ -n "${NVFX_3D:-}" ] && [ -x "${NVFX_3D}" ]; then echo "$NVFX_3D"; return 0; fi
  local b="$REPO/build/nvfx_3d"
  if [ "${TRAIN_DEVICE:-cpu}" = cuda ]; then b="${BUILD_DIR:-$REPO/build-cuda/nvfx_3d}"; elif [ -n "${BUILD:-}" ]; then b="$BUILD"; fi
  BUILD="$b" bash "$REPO/tools/study_3d/nvfx_bin_link.sh"
}
want() { case ",$STAGE," in *,all,*|*,"$1",*) return 0;; *) return 1;; esac; }  # is a stage asked for (STAGE: all or a comma list)
count_runs() { local n; n=$(ls -- "$1"/*.n3run 2>/dev/null | wc -l) || true; echo "${n:-0}"; }
train_done() { grep -q "done at iteration $2" "$1/logs/train_smoke_r2.log" 2>/dev/null; }
combine_edge() {  # out, then the parts: one header, every part's rows
  local out=$1 tmp first=1 p; shift
  tmp="$out.tmp"; rm -f "$tmp"
  for p in "$@"; do
    [ -f "$p" ] || continue
    if [ $first = 1 ]; then cp "$p" "$tmp"; first=0; else tail -n +2 "$p" >>"$tmp"; fi
  done
  [ -f "$tmp" ] && mv -f "$tmp" "$out"
}

main "$@"; exit $?
