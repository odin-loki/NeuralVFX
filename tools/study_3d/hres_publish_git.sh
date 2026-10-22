#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
git add models/ results/3d/hres/ docs/STUDY_3D_HRES.md videos/README.md 2>/dev/null || true
git add -u models/ results/3d/hres/ docs/STUDY_3D_HRES.md 2>/dev/null || true
if git diff --staged --quiet; then
  echo "nothing to commit"
  exit 0
fi
git -c user.name="${GIT_AUTHOR_NAME:-NeuralVFX}" -c user.email="${GIT_AUTHOR_EMAIL:-odin-loki@users.noreply.github.com}" \
  commit -m "Update hres 3D models and validation results after machine training run."
git push origin main
