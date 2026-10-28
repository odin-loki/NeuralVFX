#!/usr/bin/env bash
# Source wsl_env and resume smoke_r2 CUDA train (for nohup from Windows-hosted WSL).
set -euo pipefail
REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
# shellcheck source=/dev/null
. "$REPO/tools/wsl_env.sh"
exec bash "$REPO/tools/study_3d/hres_smoke_r2_resume_cuda.sh"
