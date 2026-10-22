#!/usr/bin/env bash
# Build (if needed) and run CUDA 3D trainer tests. Requires a CUDA toolkit and, for non-skipped tests, a GPU.
# Usage (repo root): bash tools/study_3d/cuda_train_check.sh [BUILD_DIR]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
BUILD_DIR=${1:-$REPO/build-cuda}
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  cmake -S "$REPO" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 \
    -DNEURALFX_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-86}" -DNEURALFX_BUILD_TESTS=ON
fi
ninja -C "$BUILD_DIR" neuralfx_3d_tests nvfx_3d
"$BUILD_DIR/neuralfx_3d_tests" --gtest_filter='Rollout3d.Cuda*'
echo "CUDA trainer check OK ($(date -Iseconds))"
