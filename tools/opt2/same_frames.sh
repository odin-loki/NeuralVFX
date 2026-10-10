#!/usr/bin/env bash
# Do two nvfx_fireball --profile CSVs hold the same pictures? Compares the per-frame RGB checksums (column rgb_fnv):
#   tools/opt2/same_frames.sh reference.csv other.csv     (exit 0 when all frames match)
set -euo pipefail
col() { local idx; idx=$(head -1 "$1" | tr ',' '\n' | grep -nx rgb_fnv | cut -d: -f1); awk -F, -v i="$idx" 'NR > 1 { print $1, $i }' "$1"; }
a=$(col "$1"); b=$(col "$2")
n=$(echo "$a" | wc -l)
same=$(paste -d' ' <(echo "$a") <(echo "$b") | awk '$1 == $3 && $2 == $4 { s++ } END { print s + 0 }')
echo "$2: $same of $n frames have the same RGB checksum as $1"
[ "$same" -eq "$n" ]
