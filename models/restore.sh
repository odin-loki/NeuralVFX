#!/usr/bin/env bash
# Rebuild the model files of a data root (NEURALVFX_DATA) from this folder, so the tests, scenes, library and tools
# that look for models there find them. Every line of layout.txt names a path under the data root and the file here
# that belongs at it. Checks SHA256SUMS first. An existing file at a path is left alone when it is the same file,
# reported when it differs (replaced only with --force).
#
#   models/restore.sh [DATA_ROOT] [--link] [--force]
#
# DATA_ROOT defaults to $NEURALVFX_DATA, else $HOME/nvfx-data. --link makes symbolic links into this folder instead
# of copies.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${NEURALVFX_DATA:-$HOME/nvfx-data}"
link=0
force=0
for a in "$@"; do
  case "$a" in
    --link) link=1 ;;
    --force) force=1 ;;
    -*) echo "restore.sh: unknown option $a" >&2; exit 2 ;;
    *) root="$a" ;;
  esac
done

(cd "$here" && sha256sum --quiet -c SHA256SUMS) || { echo "restore.sh: a file here does not match SHA256SUMS" >&2; exit 1; }

placed=0 same=0 differ=0
while read -r path file; do
  [[ -z "$path" || "$path" == \#* ]] && continue
  src="$here/$file"
  dst="$root/$path"
  if [[ -e "$dst" ]]; then
    if cmp -s "$src" "$dst"; then same=$((same + 1)); continue; fi
    if [[ $force -eq 0 ]]; then echo "differs, left alone: $dst"; differ=$((differ + 1)); continue; fi
    rm -f "$dst"
  fi
  mkdir -p "$(dirname "$dst")"
  if [[ $link -eq 1 ]]; then ln -s "$src" "$dst"; else cp "$src" "$dst"; fi
  placed=$((placed + 1))
done < "$here/layout.txt"

echo "restore.sh: $placed placed, $same already there, $differ different (left alone) under $root"
