#!/usr/bin/env bash
# Dedupe a CSV by full line (keeps first occurrence). Usage: dedupe_edge_csv.sh FILE
set -euo pipefail
f=$1
awk '!seen[$0]++' "$f" > "${f}.tmp"
mv "${f}.tmp" "$f"
wc -l "$f"
