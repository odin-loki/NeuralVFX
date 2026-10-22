#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export BUILD="${BUILD:-$REPO/build-cuda/nvfx_3d}"
exec bash tools/study_3d/hres_fire_validate.sh
