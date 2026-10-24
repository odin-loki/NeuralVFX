#!/usr/bin/env bash
# Native Windows nvfx_viewer.exe + bundled models (zip). Uses llvm-mingw from WSL (cached once).
#   tools/package_viewer_windows.sh
set -euo pipefail
REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
# llvm-mingw breaks on spaces in -I paths; use a symlink under $HOME.
LINK="${NVFX_BUILD_LINK:-$HOME/nvfx-src}"
ln -sfn "$REPO" "$LINK"
REPO="$LINK"
BUILD="${BUILD_DIR:-$HOME/build-mingw-viewer}"
VER="${NVFX_VIEWER_VERSION:-1.0.0}"
STAGE="$REPO/dist/NeuralVFX-Viewer-${VER}-win64"
ZIP="$REPO/dist/NeuralVFX-Viewer-${VER}-win64.zip"

CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/neuralfx-llvm-mingw"
LLVM_MINGW="$CACHE/llvm-mingw"
if [[ ! -x "$LLVM_MINGW/bin/x86_64-w64-mingw32-clang++" ]]; then
  echo "package_viewer_windows.sh: fetching llvm-mingw …" >&2
  mkdir -p "$CACHE"
  url=$(curl -fsSL https://api.github.com/repos/mstorsjo/llvm-mingw/releases/latest |
    grep -oE 'https://[^"]+ucrt-ubuntu-22.04-x86_64.tar.xz' | head -1)
  tgz="$CACHE/$(basename "$url")"
  [[ -f "$tgz" ]] || curl -fL "$url" -o "$tgz"
  rm -rf "$LLVM_MINGW"
  tar -xJf "$tgz" -C "$CACHE"
  mv "$(find "$CACHE" -maxdepth 1 -type d -name 'llvm-mingw-*' | head -1)" "$LLVM_MINGW"
fi
export PATH="$LLVM_MINGW/bin:$PATH"

cmake -S "$REPO" -B "$BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$REPO/cmake/toolchain-llvm-mingw.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEURALFX_BUILD_VIEWER=ON \
  -DNEURALFX_BUILD_SHARED=OFF \
  -DNEURALFX_BUILD_TESTS=OFF \
  -DNEURALFX_BUILD_TOOLS=OFF \
  -DNEURALFX_BUILD_BENCH=OFF \
  -DNEURALFX_BUILD_FUSE=OFF \
  -DNEURALFX_FETCH_ENCODERS=OFF

cmake --build "$BUILD" --target nvfx_viewer
EXE="$BUILD/viewer/nvfx_viewer.exe"
[[ -f "$EXE" ]] || { echo "build failed: no $EXE" >&2; exit 1; }

rm -rf "$STAGE"
mkdir -p "$STAGE/data"
cp "$EXE" "$STAGE/"
MINGW_BIN="$LLVM_MINGW/x86_64-w64-mingw32/bin"
for dll in libc++.dll libunwind.dll; do
  cp "$MINGW_BIN/$dll" "$STAGE/"
done
cp -a "$REPO/library" "$STAGE/"
mkdir -p "$STAGE/models"
cp -a "$REPO/models/3d" "$STAGE/models/"
mkdir -p "$STAGE/examples/scenes"
cp -a "$REPO/examples/scenes/." "$STAGE/examples/scenes/"
bash "$REPO/models/restore.sh" "$STAGE/data"

cat > "$STAGE/Run NeuralVFX Viewer.bat" <<'EOF'
@echo off
cd /d "%~dp0"
set NEURALVFX_DATA=%~dp0data
start "" "%~dp0nvfx_viewer.exe" --library
EOF

cat > "$STAGE/README.txt" <<EOF
NeuralVFX Viewer ${VER} — native Windows (CPU AVX2, OpenGL 3 window).

Double-click nvfx_viewer.exe or Run NeuralVFX Viewer.bat.
Keep library\, models\, data\, examples\, libc++.dll, and libunwind.dll next to the exe.
AMD/NVIDIA/Intel GPU drivers must support OpenGL 3.
EOF

rm -f "$ZIP"
mkdir -p "$REPO/dist"
(cd "$REPO/dist" && zip -r "$(basename "$ZIP")" "$(basename "$STAGE")")
echo "package_viewer_windows.sh: $ZIP ($(du -h "$ZIP" | cut -f1))"
