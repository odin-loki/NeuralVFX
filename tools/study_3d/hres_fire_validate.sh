#!/usr/bin/env bash
# Study hres fire: validation and test (mirror study N2 §14.13; docs/STUDY_3D_HRES.md §7).
# Run from the repo root after hres_fire.sh has produced C2..C4 under the data root.
# Usage: hres_fire_validate.sh [NVFX_3D_BINARY] [DATA_ROOT]
# Default binary: ~/.cache/neuralvfx/nvfx_3d via tools/study_3d/nvfx_bin_link.sh (avoids "Neural VFX" path splits).
# Resumes from files already written (edge CSVs append; delete them to rerun edge).
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
H="--study hres --root $ROOT --effect fire"
W="--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30"
N2_SRC=$DATA/n2_3d/models/chosen.nvfx3
CHOSEN=${CHOSEN:-C4}
HAVE_N2=0
[ -f "$N2_SRC" ] && HAVE_N2=1

mkdir -p "$R" "$ROOT/models"
if [ "$HAVE_N2" = 1 ]; then cp "$N2_SRC" "$ROOT/models/N2.nvfx3"; else echo "warn: no $N2_SRC — skipping N2 baseline copies"; fi

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

echo "hres fire validate: data $ROOT, results $R"

"$B" truth $H --split val
"$B" real $H --split val

edge_ref_lines() {
  local f="$R/edge_val_N2.csv"
  if [ -f "$f" ]; then wc -l <"$f"; else echo 0; fi
}
edge_done() {
  local f=$1 need
  need=$(edge_ref_lines)
  [ "$need" -gt 1 ] && [ -f "$f" ] && [ "$(wc -l <"$f")" -ge "$need" ]
}

if [ -f "$R/edge_sim.csv" ] && [ "$(wc -l <"$R/edge_sim.csv")" -ge 700 ]; then
  echo "skip edge sim ($R/edge_sim.csv)"
else
  "$B" edge $H --sim $W --out "$R/edge_sim.csv"
fi
for m in N2 C2 C3 C4; do
  out="$R/edge_val_$m.csv"
  if edge_done "$out"; then
    echo "skip edge $m ($out)"
  else
    "$B" edge $H --model "$ROOT/models/$m.nvfx3" --label "$m" $W --out "$out"
  fi
done
combine_edge "$R/edge_val.csv" "$R/edge_sim.csv" "$R/edge_val_N2.csv" "$R/edge_val_C2.csv" "$R/edge_val_C3.csv" "$R/edge_val_C4.csv"
"$B" edgecheck --csv "$R/edge_val.csv" | tee "$R/edgecheck_val.md"

for m in N2 C2 C3 C4; do
  "$B" track $H --split val --method "$m" --model "$ROOT/models/$m.nvfx3"
  "$B" play $H --split val --method "$m" --model "$ROOT/models/$m.nvfx3"
done
"$B" track $H --split val --method coarse --model "$ROOT/models/C4.nvfx3"
"$B" play $H --split val --method coarse --model "$ROOT/models/C4.nvfx3"
"$B" track $H --split val --method hold --model "$ROOT/models/N2.nvfx3"

for a in C2 C3 C4; do
  "$B" compare $H --split val --a "$a" --b N2 --results "$R"
  "$B" compare $H --split val --a "$a" --b coarse --results "$R"
done
"$B" compare $H --split val --a C4 --b hold --results "$R"
"$B" compare $H --split val --a N2 --b coarse --results "$R"
"$B" compare $H --split val --a hold --b coarse --results "$R"

"$B" choose $H --split val --cands C2,C3,C4 --edge "$R/edge_val.csv" | tee "$R/choose_val.md"

cp "$ROOT/models/$CHOSEN.nvfx3" "$ROOT/models/chosen.nvfx3"
"$B" freeze --model "$ROOT/models/chosen.nvfx3" --frozen "$R/frozen.csv"
"$B" freeze --model "$ROOT/models/N2.nvfx3" --frozen "$R/frozen.csv"

"$B" truth $H --split test
"$B" real $H --split test

"$B" edge $H --model "$ROOT/models/chosen.nvfx3" --label learned $W --out "$R/edge_test_learned.csv"
for m in learned N2 coarse hold; do
  case $m in
    learned) M="$ROOT/models/chosen.nvfx3" ;;
    N2) M="$ROOT/models/N2.nvfx3" ;;
    coarse) M="$ROOT/models/C4.nvfx3" ;;
    hold) M="$ROOT/models/N2.nvfx3" ;;
  esac
  "$B" track $H --split test --method "$m" --model "$M" --frozen "$R/frozen.csv"
  if [ "$m" != hold ]; then
    "$B" play $H --split test --method "$m" --model "$M" --frozen "$R/frozen.csv"
  fi
done

"$B" compare $H --split test --a learned --b N2 --results "$R"
"$B" compare $H --split test --a learned --b coarse --results "$R"

echo "done: chosen $CHOSEN -> $ROOT/models/chosen.nvfx3; tables under $R"
