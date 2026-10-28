#!/usr/bin/env bash
set -euo pipefail
REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
# shellcheck source=/dev/null
. "$REPO/tools/wsl_env.sh"
ROOT="$NEURALVFX_DATA/hres_3d/smoke_r2"
LOG="$ROOT/logs/smoke_r2_pipeline.log"
rm -f "$ROOT/logs/train.lock"
export FORCE=1 STAGE=finish,bounds,edge TRAIN_DEVICE=cuda TRAIN_SHARDS="${TRAIN_SHARDS:-4}"
cd "$REPO"
exec >>"$LOG" 2>&1
echo "=== smoke_r2 finish pipeline $(date -Iseconds) pid=$$ ==="
exec bash "$REPO/tools/study_3d/hres_smoke_r2.sh"
