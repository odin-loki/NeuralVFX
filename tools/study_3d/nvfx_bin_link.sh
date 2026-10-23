#!/usr/bin/env bash
# Symlink nvfx_3d to a path without spaces (OneDrive "Neural VFX" breaks some nohup wrappers).
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SRC=${1:-}
if [ -z "$SRC" ]; then SRC=${BUILD:-$REPO/build/nvfx_3d}; fi
if [[ "$SRC" != /* ]]; then SRC="$REPO/$SRC"; fi
SRC=$(readlink -f "$SRC") || { echo "error: nvfx_3d binary not found: $SRC" >&2; exit 1; }
CACHE="$HOME/.cache/neuralvfx/nvfx_3d"
mkdir -p "$(dirname "$CACHE")"
ln -sfn "$SRC" "$CACHE"
echo "$CACHE"
