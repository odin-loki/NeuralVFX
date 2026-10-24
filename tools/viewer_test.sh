#!/usr/bin/env bash
# Run all viewer tests (windowless GTest + Xvfb GUI screenshots when available).
#   tools/viewer_test.sh [BUILD_DIR]
#
# Needs library models for data-tagged tests:  models/restore.sh --link
set -euo pipefail
REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
BUILD=${1:-${BUILD_DIR:-$REPO/build}}
if [[ ! -d "$BUILD" ]]; then
  echo "viewer_test.sh: no build at $BUILD (cmake -DNEURALFX_BUILD_VIEWER=ON -DNEURALFX_BUILD_TESTS=ON)" >&2
  exit 2
fi
export NEURALVFX_DATA=${NEURALVFX_DATA:-$HOME/nvfx-data}
if [[ ! -f "$NEURALVFX_DATA/library/models/fire.nvfx" ]]; then
  echo "viewer_test.sh: linking library models via models/restore.sh --link" >&2
  bash "$REPO/models/restore.sh" --link "$NEURALVFX_DATA"
fi
cd "$BUILD"
ctest -R 'neuralfx\.ViewerHeadless|neuralfx\.Viewer' --output-on-failure "$@"
