#!/usr/bin/env bash
# Study R2 §2.2 and §2.4: the 48 training runs (if the data root lacks them), training (N2's four stages, 1750
# iterations; stage 4's burn-ins 48 to 360 frames; the spread and edge losses from stage 2), then the candidates C2, C3
# and C4, and markers/candidates_done. Resumable: runs already recorded are skipped, training resumes from
# ckpt/r2.ckpt (a checkpoint every 10 iterations). Portable form of tools/study_r2/{records,train}.sh.
#
# Usage (repo root; usually started by r2_resume.sh, detached and from a snapshot copy):
#   bash tools/study_3d/r2_train.sh
# Env (tools/study_3d/r2_env.sh): T (training threads, 8), J (recording processes, 8), FORCE=1 (run again although
#   markers/candidates_done exists: training is then a no-op at 1750 and the candidates are written again), SPREAD,
#   EDGE (the losses' weights; the rule's are 2 and 10), NICE_LEVEL, BUILD, NVFX_3D, NEURALVFX_DATA.
# Never starts a second trainer on this data root (logs/train.lock). CPU only: the CUDA trainer has no spread or edge
# loss (the tool refuses --device cuda with them), so TRAIN_DEVICE=cuda is an error here.
#
# Layout: main(), then one `main "$@"` line: bash parses a function body in full before running any of it, so the
# hours-long run is not corrupted when this file is edited (hres_fire_validate.sh's lesson).
set -euo pipefail

main() {
  local REPO B T nl ns RUNS k
  REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
  [ -f "$REPO/tools/study_3d/r2_env.sh" ] || { echo "error: repo not found at $REPO (running a copy? set NVFX_REPO)" >&2; return 1; }
  . "$REPO/tools/study_3d/r2_env.sh"
  cd "$REPO"
  r2_dirs
  exec 8>"$D/logs/train.lock"
  if ! flock -n 8; then echo "r2 train: another r2_train.sh holds $D/logs/train.lock"; return 0; fi
  if [ -f "$MARK/candidates_done" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "r2 train: candidates done ($MARK/candidates_done); set FORCE=1 to write them again"; return 0
  fi
  if r2_running "$R2_PAT_TRAIN"; then echo "r2 train: a trainer is already running on $D ($(pgrep -af -- "$R2_PAT_TRAIN" | head -1)); not starting another"; return 0; fi
  if [ "${TRAIN_DEVICE:-cpu}" != cpu ]; then echo "error: study R2 trains on the CPU (the CUDA trainer has no spread or edge loss)"; return 1; fi
  B=$(r2_bin)
  T=${T:-8}
  echo "=== r2_train $(date -Iseconds) B=$B D=$D threads=$T J=$J pid=$$ ==="
  r2_disk_need 2

  # 1. the training runs: 24 of 18 s (N2's kinds, seeds 34,200,000 + i) and 24 of 8 s (random controls held, seeds
  # 34,100,000 + i); one process per run, J at a time (record skips a run file that exists)
  nl=$(r2_runs_count long); ns=$(r2_runs_count short)
  if [ "$nl" -lt 24 ] || [ "$ns" -lt 24 ]; then
    echo "--- recording the training runs ($nl long, $ns short of 24 each; $J at a time) ---"
    {
      for k in $(seq 0 23); do echo "record --runs-dir long --from $k --runs $((k + 1))"; done
      for k in $(seq 0 23); do echo "record --runs-dir short --from $k --runs $((k + 1))"; done
    } | r2_run_many
    nl=$(r2_runs_count long); ns=$(r2_runs_count short)
    [ "$nl" -ge 24 ] && [ "$ns" -ge 24 ] || { echo "error: recording left $nl long, $ns short runs"; return 1; }
    echo "RECORDS_DONE $(date -Iseconds) ($nl long, $ns short)"
  fi

  # 2. training (STUDY_R2.md §2.4 and §7), resumable from ckpt/r2.ckpt; the tool writes logs/train_r2.log itself
  RUNS="--runs-dir long --also $D/runs/short"
  echo "--- training r2 (see $D/logs/train_r2.log; last: $(r2_train_last)) ---"
  export NVFX_TRAIN_QUIET_STDOUT=1
  nice -n "$NICE" "$B" train $H --name r2 $RUNS --iterations 900 --finetune 350 --activity 250 --long 250 --batch 8 --unroll 12 --burn 24 \
    --burn-long-min 48 --burn-long-max 360 --spread "${SPREAD:-2}" --edge "${EDGE:-10}" --seed 34200000 --threads "$T" --checkpoint-every 10 \
    --device cpu
  unset NVFX_TRAIN_QUIET_STDOUT
  r2_train_done || { echo "error: training ended without 'done at iteration $TOTAL_IT' (see $D/logs/train_r2.log)"; return 1; }

  # 3. the candidates: the stepper after stages 2, 3 and 4, the same start points and detail constants
  for k in 2 3 4; do
    [ -f "$D/ckpt/r2_stage$k.ckpt" ] || { echo "error: missing $D/ckpt/r2_stage$k.ckpt"; return 1; }
    nice -n "$NICE" "$B" finish $H --ckpt "r2_stage$k" --out "C$k" $RUNS
  done
  touch "$MARK/candidates_done"
  echo "TRAIN_DONE $(date -Iseconds): $D/models/C2.nvfx3 C3 C4; $MARK/candidates_done"
  echo "next: bash tools/study_3d/r2_resume.sh (starts validation once markers/rule_committed exists)"
}

main "$@"; exit $?
