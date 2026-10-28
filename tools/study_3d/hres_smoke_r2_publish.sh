#!/usr/bin/env bash
# Publish frozen smoke_r2 chosen model to models/3d/smoke_r2_hres.nvfx3 (after validate + frozen.csv).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
if [ -f "$REPO/tools/wsl_env.sh" ]; then
  # shellcheck source=/dev/null
  . "$REPO/tools/wsl_env.sh"
fi
R2_ROOT=${1:-${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/smoke_r2}
FROZEN=${HRES_R2_FROZEN:-$REPO/results/3d/hres/smoke_r2/frozen.csv}
SRC="$R2_ROOT/models/chosen.nvfx3"
DST="$REPO/models/3d/smoke_r2_hres.nvfx3"
MODELS="$REPO/models/3d"
SUMS="$REPO/models/SHA256SUMS"
LAYOUT="$REPO/models/layout.txt"
[ -f "$SRC" ] || { echo "error: missing $SRC (run validate first)"; exit 1; }
base=$(basename "$SRC")
sha=$(sha256sum "$SRC" | awk '{print $1}')
grep -q "^$base,$sha," "$FROZEN" || { echo "error: $base not in $FROZEN"; exit 1; }
mkdir -p "$MODELS"
cp "$SRC" "$DST"
grep -v "  3d/smoke_r2_hres.nvfx3$" "$SUMS" >"$SUMS.tmp" || true
echo "$sha  3d/smoke_r2_hres.nvfx3" >>"$SUMS.tmp"
sort -k2 "$SUMS.tmp" -o "$SUMS"
rm -f "$SUMS.tmp"
grep -vF " 3d/smoke_r2_hres.nvfx3" "$LAYOUT" >"$LAYOUT.tmp" 2>/dev/null || true
echo "hres_3d/smoke_r2/models/chosen.nvfx3 3d/smoke_r2_hres.nvfx3" >>"$LAYOUT.tmp"
mv "$LAYOUT.tmp" "$LAYOUT"
echo "published 3d/smoke_r2_hres.nvfx3 ($sha)"
