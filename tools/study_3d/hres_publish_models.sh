#!/usr/bin/env bash
# Copy finished hres .nvfx3 into models/3d/ and refresh SHA256SUMS + layout.txt entries.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
FIRE_ROOT=${1:-${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/fire}
SMOKE_ROOT=${2:-${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/smoke}
FROZEN=${3:-${HRES_FROZEN:-$REPO/results/3d/hres/fire/frozen.csv}}
MODELS="$REPO/models/3d"
SUMS="$REPO/models/SHA256SUMS"
LAYOUT="$REPO/models/layout.txt"
LOCK="$REPO/models/.publish.lock"
mkdir -p "$MODELS"
exec 9>"$LOCK"
flock -w 120 9 || { echo "error: could not lock $LOCK"; exit 1; }

update_sha() {
  local rel=$1 hash=$2
  grep -v "  $rel$" "$SUMS" > "$SUMS.tmp" || true
  echo "$hash  $rel" >> "$SUMS.tmp"
  sort -k2 "$SUMS.tmp" -o "$SUMS"
  rm -f "$SUMS.tmp"
}

ensure_layout() {
  local data_path=$1 rel=$2 tmp="$LAYOUT.tmp"
  grep -vF " 3d/$rel" "$LAYOUT" >"$tmp" 2>/dev/null || true
  echo "$data_path 3d/$rel" >>"$tmp"
  mv "$tmp" "$LAYOUT"
}

verify_frozen() {
  local src=$1 base sha
  base=$(basename "$src")
  [ -f "$FROZEN" ] || { echo "error: frozen.csv missing: $FROZEN"; return 1; }
  sha=$(sha256sum "$src" | awk '{print $1}')
  grep -q "^$base,$sha," "$FROZEN" || {
    echo "error: $base sha $sha not listed in $FROZEN"
    return 1
  }
  echo "$sha"
}

publish() {
  local src=$1 dst=$2 data_path=$3 verify=${4:-0}
  [ -f "$src" ] || { echo "skip missing $src"; return 0; }
  local hash
  if [ "$verify" = 1 ]; then
    hash=$(verify_frozen "$src")
  else
    hash=$(sha256sum "$src" | awk '{print $1}')
  fi
  cp "$src" "$MODELS/$dst"
  local got
  got=$(sha256sum "$MODELS/$dst" | awk '{print $1}')
  [ "$got" = "$hash" ] || { echo "error: copy sha mismatch for 3d/$dst"; exit 1; }
  update_sha "3d/$dst" "$hash"
  ensure_layout "$data_path" "$dst"
  echo "published 3d/$dst ($hash)"
}

if [ -f "$FIRE_ROOT/models/chosen.nvfx3" ]; then
  publish "$FIRE_ROOT/models/chosen.nvfx3" fire_hres.nvfx3 hres_3d/fire/models/chosen.nvfx3 1
else
  publish "$FIRE_ROOT/models/hres_fire.nvfx3" fire_hres.nvfx3 hres_3d/fire/models/hres_fire.nvfx3 0
fi
publish "$SMOKE_ROOT/models/smoke_hres.nvfx3" smoke_hres.nvfx3 hres_3d/smoke/models/smoke_hres.nvfx3 0
