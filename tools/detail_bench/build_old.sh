#!/bin/sh
# The detail layer's A/B bench (tools/nvfx_detail_bench.cpp): the old probe, this tree's probe.cpp built against the
# runtime of an earlier commit, with that commit's own build flags.
#   tools/detail_bench/build_old.sh [REV] [DIR]
# REV defaults to the merge base of HEAD with main; DIR (default ../nvfx-detail-old) receives the commit's sources, its
# build (the runtime only: about 60 MB) and DIR/nvfx_detail_probe_old.so. Needs g++-14, cmake and ninja.
set -e
SRC=$(git rev-parse --show-toplevel)
REV=${1:-$(git -C "$SRC" merge-base HEAD main)}
DIR=${2:-$SRC/../nvfx-detail-old}
CXX=${CXX:-g++-14}
mkdir -p "$DIR/src"
git -C "$SRC" archive "$REV" CMakeLists.txt cmake include src examples | tar -x -C "$DIR/src"
cmake -S "$DIR/src" -B "$DIR/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" -DBUILD_TESTING=OFF \
  -DNEURALFX_BUILD_TESTS=OFF -DNEURALFX_BUILD_TOOLS=OFF -DNEURALFX_BUILD_BENCH=OFF -DNEURALFX_BUILD_SHARED=OFF \
  -DNEURALFX_FETCH_ENCODERS=OFF > "$DIR/cmake.log"
ninja -C "$DIR/build" ${JOBS:+-j"$JOBS"} nvfx
# as CMake builds this tree's probe (CMakeLists.txt, nvfx_detail_probe): only the probe_* functions exported
"$CXX" -O3 -DNDEBUG -std=c++23 -fPIC -shared -fvisibility=hidden -fvisibility-inlines-hidden -I"$DIR/src/include" -I"$DIR/src/src/runtime" \
  "$SRC/tools/detail_bench/probe.cpp" "$DIR/build/libnvfx.a" "$DIR/build/libneuralfx_model.a" -Wl,--exclude-libs,ALL -Wl,-Bsymbolic -lpthread \
  -o "$DIR/nvfx_detail_probe_old.so"
echo "$DIR/nvfx_detail_probe_old.so: the runtime of $(git -C "$SRC" rev-parse --short "$REV")"
