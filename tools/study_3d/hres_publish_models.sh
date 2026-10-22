#!/usr/bin/env bash
# Copy finished hres .nvfx3 into models/3d/ and refresh SHA256SUMS + layout.txt entries.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
FIRE_ROOT=${1:-${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/fire}
SMOKE_ROOT=${2:-${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/smoke}
MODELS="$REPO/models/3d"
SUMS="$REPO/models/SHA256SUMS"
LAYOUT="$REPO/models/layout.txt"
mkdir -p "$MODELS"

update_sha() {
  local rel=$1 hash=$2
  grep -v "  $rel$" "$SUMS" > "$SUMS.tmp" || true
  echo "$hash  $rel" >> "$SUMS.tmp"
  sort -k2 "$SUMS.tmp" -o "$SUMS"
  rm -f "$SUMS.tmp"
}

ensure_layout() {
  local data_path=$1 rel=$2
  if ! grep -q "^$data_path " "$LAYOUT"; then
    echo "$data_path 3d/$rel" >> "$LAYOUT"
  fi
}

publish() {
  local src=$1 dst=$2 data_path=$3
  [ -f "$src" ] || { echo "skip missing $src"; return 0; }
  cp "$src" "$MODELS/$dst"
  hash=$(sha256sum "$MODELS/$dst" | awk '{print $1}')
  update_sha "$dst" "$hash"
  ensure_layout "$data_path" "$dst"
  echo "published 3d/$dst ($hash)"
}

publish "$FIRE_ROOT/models/hres_fire.nvfx3" fire_hres.nvfx3 hres_3d/fire/models/hres_fire.nvfx3
publish "$SMOKE_ROOT/models/smoke_hres.nvfx3" smoke_hres.nvfx3 hres_3d/smoke/models/smoke_hres.nvfx3
