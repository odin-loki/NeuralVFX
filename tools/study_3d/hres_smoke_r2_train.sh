#!/usr/bin/env bash
# Study hres, smoke on study R2's teacher: the record, train and finish stages of tools/study_3d/hres_smoke_r2.sh
# (docs/STUDY_3D_HRES.md §10), kept under this name for the commands that used it. Everything (the seeds, the losses,
# the CPU-only trainer, the dry run without FORCE=1, the lock) is hres_smoke_r2.sh's; this forwards to it with
# STAGE=record,train,finish unless STAGE is set. TREC, TTRAIN and HRES_SMOKE_R2_ROOT are read by hres_smoke_r2.sh too.
#
#   bash tools/study_3d/hres_smoke_r2_train.sh            # the plan; starts nothing
#   FORCE=1 bash tools/study_3d/hres_smoke_r2_train.sh    # record, train (CPU, --spread 2 --edge 10), finish C2..C4 + smoke_r2_hres
set -euo pipefail
REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
STAGE=${STAGE:-record,train,finish} exec bash "$REPO/tools/study_3d/hres_smoke_r2.sh" "$@"
