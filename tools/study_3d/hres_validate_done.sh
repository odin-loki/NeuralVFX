#!/usr/bin/env bash
# True when validate.log has "done: chosen" after the latest "=== validate start" line.
set -euo pipefail
LOG=${1:?validate.log path}
[ -f "$LOG" ] || exit 1
start=$(grep -n '=== validate start' "$LOG" | tail -1 | cut -d: -f1 || true)
done=$(grep -n 'done: chosen' "$LOG" | tail -1 | cut -d: -f1 || true)
[ -n "$start" ] && [ -n "$done" ] && [ "$done" -gt "$start" ]
