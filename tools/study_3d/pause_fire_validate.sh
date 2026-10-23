#!/usr/bin/env bash
# Stop in-progress hres fire validate (e.g. while smoke CUDA train runs). Partial edge CSVs for
# incomplete models are removed so a later run does not append duplicates.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
R=results/3d/hres/fire
REF=0
[ -f "$R/edge_val_N2.csv" ] && REF=$(wc -l <"$R/edge_val_N2.csv")
for m in C2 C3 C4; do
  f="$R/edge_val_$m.csv"
  if [ -f "$f" ]; then
    n=$(wc -l <"$f")
    if [ "$REF" -gt 0 ] && [ "$n" -lt "$REF" ]; then
      echo "remove partial $f ($n lines, need $REF)"
      rm -f "$f"
    fi
  fi
done
pkill -f 'nvfx_3d.*hres_3d/fire' 2>/dev/null || true
pkill -f 'hres_fire_validate(\.sh|\.snapshot\.sh)' 2>/dev/null || true
pkill -f 'run_fire_validate.sh' 2>/dev/null || true
sleep 1
LOG=${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/fire/logs/validate.log
echo "=== validate paused for smoke CUDA $(date -Iseconds) ===" >>"$LOG"
echo "fire validate paused (resume: bash tools/study_3d/start_fire_validate.sh or after smoke watcher)"
