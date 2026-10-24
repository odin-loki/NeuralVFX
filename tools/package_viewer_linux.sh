#!/usr/bin/env bash
# Linux nvfx_viewer + bundled models (zip). Uses an existing native build, or configures one.
#   tools/package_viewer_linux.sh
set -euo pipefail
REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
VER="${NVFX_VIEWER_VERSION:-1.0.0}"
BUILD="${BUILD_DIR:-$HOME/build-viewer}"
if [[ ! -f "$BUILD/build.ninja" && -f "$REPO/build-viewer/build.ninja" ]]; then
  BUILD="$REPO/build-viewer"
fi
STAGE="$REPO/dist/NeuralVFX-Viewer-${VER}-linux-x64"
ZIP="$REPO/dist/NeuralVFX-Viewer-${VER}-linux-x64.zip"

if [[ ! -f "$BUILD/build.ninja" ]]; then
  cmake -S "$REPO" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DNEURALFX_BUILD_VIEWER=ON \
    -DNEURALFX_BUILD_SHARED=OFF \
    -DNEURALFX_BUILD_TESTS=OFF \
    -DNEURALFX_BUILD_TOOLS=OFF \
    -DNEURALFX_BUILD_BENCH=OFF \
    -DNEURALFX_BUILD_FUSE=OFF \
    -DNEURALFX_FETCH_ENCODERS=OFF
fi
cmake --build "$BUILD" --target nvfx_viewer
EXE="$BUILD/viewer/nvfx_viewer"
[[ -f "$EXE" ]] || { echo "build failed: no $EXE" >&2; exit 1; }

rm -rf "$STAGE"
mkdir -p "$STAGE/data"
cp "$EXE" "$STAGE/"
cp -a "$REPO/library" "$STAGE/"
mkdir -p "$STAGE/models"
cp -a "$REPO/models/3d" "$STAGE/models/"
mkdir -p "$STAGE/examples/scenes"
cp -a "$REPO/examples/scenes/." "$STAGE/examples/scenes/"
bash "$REPO/models/restore.sh" "$STAGE/data"

cat > "$STAGE/run-viewer.sh" <<'EOF'
#!/bin/sh
cd "$(dirname "$0")"
export NEURALVFX_DATA="$PWD/data"
exec ./nvfx_viewer --library "$@"
EOF
chmod +x "$STAGE/run-viewer.sh"

cat > "$STAGE/README.txt" <<EOF
NeuralVFX Viewer ${VER} — Linux x86-64 (CPU AVX2, OpenGL 3 window).

Run ./run-viewer.sh or ./nvfx_viewer --library
Keep library/, models/, data/, and examples/ next to the binary.
Needs libglfw3 and OpenGL 3 (GPU drivers). The effect itself runs on the CPU.
EOF

rm -f "$ZIP"
mkdir -p "$REPO/dist"
(cd "$REPO/dist" && zip -r "$(basename "$ZIP")" "$(basename "$STAGE")")
echo "package_viewer_linux.sh: $ZIP ($(du -h "$ZIP" | cut -f1))"
