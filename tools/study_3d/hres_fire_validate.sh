#!/usr/bin/env bash
# Study hres fire: validation and test (mirror study N2 §14.13; docs/STUDY_3D_HRES.md §7).
# Run from the repo root after hres_fire.sh has produced C2..C4 under the data root.
# Usage: hres_fire_validate.sh [NVFX_3D_BINARY] [DATA_ROOT]
# Default binary: ~/.cache/neuralvfx/nvfx_3d via tools/study_3d/nvfx_bin_link.sh (avoids "Neural VFX" path splits).
# Resumes from files already written (edge CSVs append; delete them to rerun edge).
# Long runs use a snapshot copy via run_fire_validate.sh; still avoid editing during a direct bash of this file.
# Override chosen candidate: CHOSEN=C3 (default: parse from choose_val.md, else C4).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
if [ -n "${1:-}" ] && [ -x "$1" ]; then
  B=$1
else
  B=$(BUILD="${BUILD:-$REPO/build/nvfx_3d}" bash "$REPO/tools/study_3d/nvfx_bin_link.sh")
fi
ROOT=${2:-${NEURALVFX_DATA:-$HOME/nvfx-data}/hres_3d/fire}
DATA=${NEURALVFX_DATA:-$HOME/nvfx-data}
R=results/3d/hres/fire
TARGS=()
if [ -n "${THREADS:-}" ]; then TARGS=(--threads "$THREADS"); fi
nvfx() {
  case "$1" in freeze|edgecheck|choose) "$B" "$@";;
  *) "$B" "$@" "${TARGS[@]}";;
  esac
}
H="--study hres --root $ROOT --effect fire"
W="--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30"
N2_SRC=$DATA/n2_3d/models/chosen.nvfx3
HAVE_N2=0
[ -f "$N2_SRC" ] && HAVE_N2=1

# Full hres/N2 edge tables: 1 header + 5 winds × 61 rows = 306 lines.
EDGE_FULL=306
USE_EDGE_BOUNDS=${USE_EDGE_BOUNDS:-0}
RUNS=${EDGE_RUNS:-4}
BARGS=()
EDGE_HEAD="$R/edge_sim.csv"
if [ "$USE_EDGE_BOUNDS" = 1 ]; then
  BARGS=(--bounds "$R/edge_bounds.csv")
  EDGE_HEAD="$R/edge_teacher.csv"
fi

mkdir -p "$R" "$ROOT/models"
if [ "$HAVE_N2" = 1 ]; then cp "$N2_SRC" "$ROOT/models/N2.nvfx3"; else echo "warn: no $N2_SRC - skipping N2 baseline copies"; fi

dedupe_edge_csv() {
  local f=$1
  [ -f "$f" ] || return 0
  local n
  n=$(wc -l <"$f")
  [ "$n" -le "$EDGE_FULL" ] && return 0
  awk '!s[$0]++' "$f" >"$f.tmp" && mv "$f.tmp" "$f"
  echo "deduped $f ($(wc -l <"$f") lines)"
}

combine_edge() {
  out=$1
  shift
  rm -f "$out"
  for p in "$@"; do
    [ -f "$p" ] || continue
    if [ ! -f "$out" ]; then
      cp "$p" "$out"
    else
      tail -n +2 "$p" >> "$out"
    fi
  done
}

freeze_once() {
  local model=$1
  local sha bytes
  sha=$(sha256sum "$model" | awk '{print $1}')
  bytes=$(wc -c <"$model")
  local base
  base=$(basename "$model")
  if [ -f "$R/frozen.csv" ] && grep -q "^$base,$sha," "$R/frozen.csv"; then
    echo "skip freeze $base ($sha)"
    return 0
  fi
  nvfx freeze --model "$model" --frozen "$R/frozen.csv"
}

edge_done() {
  local f=$1
  [ -f "$f" ] && [ "$(wc -l <"$f")" -ge "$EDGE_FULL" ]
}

edge_done_multi() {
  local f=$1 runs=${2:-$RUNS}
  [ -f "$f" ] && [ "$(wc -l <"$f")" -ge $((1 + runs * 5 * 61)) ]
}

echo "hres fire validate: data $ROOT, results $R"
dedupe_edge_csv "$R/edge_sim.csv"

if [ "$USE_EDGE_BOUNDS" = 1 ]; then
  [ -f "$R/edge_bounds.csv" ] && edge_done_multi "$EDGE_HEAD" || {
    echo "run: bash tools/study_3d/hres_fire_edge_bounds.sh"; exit 1
  }
  echo "USE_EDGE_BOUNDS=1: teacher $EDGE_HEAD, bounds $R/edge_bounds.csv"
elif [ -f "$R/edge_sim.csv" ] && [ "$(wc -l <"$R/edge_sim.csv")" -ge "$EDGE_FULL" ]; then
  echo "skip edge sim ($R/edge_sim.csv)"
else
  rm -f "$R/edge_sim.csv"
  nvfx edge $H --sim $W --out "$R/edge_sim.csv"
fi
for m in N2 C2 C3 C4; do
  out="$R/edge_val_$m.csv"
  if edge_done "$out"; then
    echo "skip edge $m ($out)"
  else
    rm -f "$out"
    nvfx edge $H --model "$ROOT/models/$m.nvfx3" --label "$m" $W --out "$out"
  fi
done
combine_edge "$R/edge_val.csv" "$EDGE_HEAD" "$R/edge_val_N2.csv" "$R/edge_val_C2.csv" "$R/edge_val_C3.csv" "$R/edge_val_C4.csv"
nvfx edgecheck --csv "$R/edge_val.csv" "${BARGS[@]}" | tee "$R/edgecheck_val.md"

for m in C2 C3 C4; do
  nvfx track $H --split val --method "$m" --model "$ROOT/models/$m.nvfx3"
  nvfx play $H --split val --method "$m" --model "$ROOT/models/$m.nvfx3"
done
nvfx play $H --split val --method N2 --model "$ROOT/models/N2.nvfx3"
# N2 track at hres coarse size is invalid (16^3 model vs 32^3 truth); endless compare still uses N2 plays.

nvfx track $H --split val --method coarse --model "$ROOT/models/C4.nvfx3"
nvfx play $H --split val --method coarse --model "$ROOT/models/C4.nvfx3"
nvfx track $H --split val --method hold --model "$ROOT/models/N2.nvfx3"

for a in C2 C3 C4; do
  nvfx compare $H --split val --a "$a" --b N2 --results "$R"
  nvfx compare $H --split val --a "$a" --b coarse --results "$R"
done
nvfx compare $H --split val --a C4 --b hold --results "$R"
nvfx compare $H --split val --a N2 --b coarse --results "$R"
nvfx compare $H --split val --a hold --b coarse --results "$R"

nvfx choose $H --split val --cands C2,C3,C4 --edge "$R/edge_val.csv" "${BARGS[@]}" | tee "$R/choose_val.md"

if [ -n "${CHOSEN:-}" ]; then
  echo "CHOSEN from env: $CHOSEN"
else
  CHOSEN=$(grep '^chosen:' "$R/choose_val.md" 2>/dev/null | tail -1 | awk '{print $2}' || true)
  CHOSEN=${CHOSEN:-C4}
  echo "CHOSEN parsed: $CHOSEN"
fi
[ -f "$ROOT/models/$CHOSEN.nvfx3" ] || { echo "error: missing $ROOT/models/$CHOSEN.nvfx3"; exit 1; }

cp "$ROOT/models/$CHOSEN.nvfx3" "$ROOT/models/chosen.nvfx3"
freeze_once "$ROOT/models/chosen.nvfx3"
if [ "$HAVE_N2" = 1 ]; then freeze_once "$ROOT/models/N2.nvfx3"; fi

nvfx truth $H --split test
nvfx real $H --split test

W_TEST="--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30"
rm -f "$R/edge_test_sim.csv" "$R/edge_test_learned.csv" "$R/edge_test_N2.csv"
nvfx edge $H --sim $W_TEST --seed-base 9861900 --out "$R/edge_test_sim.csv"
nvfx edge $H --model "$ROOT/models/chosen.nvfx3" --label learned $W_TEST --seed-base 9862900 --out "$R/edge_test_learned.csv"
if [ "$HAVE_N2" = 1 ]; then
  nvfx edge $H --model "$ROOT/models/N2.nvfx3" --label N2 $W_TEST --seed-base 9862900 --out "$R/edge_test_N2.csv"
fi
combine_edge "$R/edge_test.csv" "$R/edge_test_sim.csv" "$R/edge_test_learned.csv" "$R/edge_test_N2.csv"
nvfx edgecheck --csv "$R/edge_test.csv" "${BARGS[@]}" | tee "$R/edgecheck_test.md"

FZ="--frozen $R/frozen.csv"
for m in learned N2 coarse hold; do
  case $m in
    learned) M="$ROOT/models/chosen.nvfx3" ;;
    N2) M="$ROOT/models/N2.nvfx3" ;;
    coarse) M="$ROOT/models/$CHOSEN.nvfx3" ;;
    hold) M="$ROOT/models/N2.nvfx3" ;;
  esac
  if [ "$m" = N2 ]; then
    nvfx play $H --split test --method N2 --model "$M" $FZ
    continue
  fi
  nvfx track $H --split test --method "$m" --model "$M" $FZ
  if [ "$m" != hold ]; then
    nvfx play $H --split test --method "$m" --model "$M" $FZ
  fi
done

nvfx compare $H --split test --a learned --b N2 --results "$R"
nvfx compare $H --split test --a learned --b coarse --results "$R"

echo "done: chosen $CHOSEN -> $ROOT/models/chosen.nvfx3; tables under $R"
