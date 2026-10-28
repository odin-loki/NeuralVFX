#!/usr/bin/env bash
# Sequential hres smoke_r2 promotion (STUDY_3D_HRES.md §10): validate (resume) -> publish models/3d/smoke_r2_hres.nvfx3.
# Run from repo root or via start_smoke_r2_validate_pipeline.sh (detached). One log: validate.log (for hres_validate_done.sh).
set -euo pipefail

main() {
  local REPO ROOT VALIDATE_LOG PIPE_LOG DONE_SH
  REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
  if [ -f "$REPO/tools/wsl_env.sh" ]; then
    # shellcheck source=/dev/null
    . "$REPO/tools/wsl_env.sh"
  fi
  ROOT="${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/smoke_r2"
  VALIDATE_LOG="$ROOT/logs/validate.log"
  PIPE_LOG="$ROOT/logs/validate_pipeline.log"
  DONE_SH="$REPO/tools/study_3d/hres_validate_done.sh"
  mkdir -p "$ROOT/logs"
  cd "$REPO"
  exec 200>"$ROOT/logs/validate.lock"
  flock -n 200 || { echo "another smoke_r2 validate pipeline holds the lock"; return 0; }

  exec >>"$PIPE_LOG" 2>&1
  echo "=== smoke_r2 validate pipeline $(date -Iseconds) pid=$$ ==="

  if bash "$DONE_SH" "$VALIDATE_LOG" 2>/dev/null; then
    echo "validate already complete ($(grep '^done: chosen' "$VALIDATE_LOG" | tail -1)); publish only"
  else
    echo "=== validate start $(date -Iseconds) threads=${THREADS:-default} nice=${NICE_LEVEL:-0} ===" | tee -a "$VALIDATE_LOG"
    if [ -n "${NICE_LEVEL:-}" ]; then
      nice -n "$NICE_LEVEL" bash "$REPO/tools/study_3d/run_smoke_r2_validate.sh" 2>&1 | tee -a "$VALIDATE_LOG"
    else
      bash "$REPO/tools/study_3d/run_smoke_r2_validate.sh" 2>&1 | tee -a "$VALIDATE_LOG"
    fi
    bash "$DONE_SH" "$VALIDATE_LOG" || {
      echo "error: validate did not finish (no done: chosen after latest start in $VALIDATE_LOG)" >&2
      return 1
    }
  fi

  echo "--- publish frozen chosen -> models/3d/smoke_r2_hres.nvfx3 ---"
  bash "$REPO/tools/study_3d/hres_smoke_r2_publish.sh"
  echo "=== smoke_r2 validate pipeline done $(date -Iseconds) ==="
}

main "$@"; exit $?
