#!/usr/bin/env bash
# Study hres smoke_r2: validation and test after C2/C3 pass the §10 edge gate (docs/STUDY_3D_HRES.md §10; mirror hres_fire_validate.sh).
# Run from the repo root when models/C2.nvfx3 and C3.nvfx3 exist under the data root.
# Usage: hres_smoke_r2_validate.sh [NVFX_3D_BINARY] [DATA_ROOT]
# Default binary: nvfx_bin_link.sh. Resumes from files already written. USE_EDGE_BOUNDS=1 by default (edge_bounds.csv).
# Override chosen: CHOSEN=C2|C3. C4 is excluded (failed the §10 edge gate).
#
# Layout: helpers, main(), then one `main "$@"; exit $?` line (STUDY_3D_HRES.md §7).
set -euo pipefail

EDGE_FULL=306
B= R= ROOT= RUNS= TARGS=() BARGS=() HARGS=() WARGS=()

nvfx() {
  case "$1" in freeze|edgecheck|choose) "$B" "$@";;
  *) "$B" "$@" "${TARGS[@]}";;
  esac
}

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
  local out=$1
  shift
  rm -f "$out"
  for p in "$@"; do
    [ -f "$p" ] || continue
    if [ ! -f "$out" ]; then cp "$p" "$out"; else tail -n +2 "$p" >>"$out"; fi
  done
}

freeze_once() {
  local model=$1 sha bytes base
  sha=$(sha256sum "$model" | awk '{print $1}')
  bytes=$(wc -c <"$model")
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

cleanup() {
  local rc=$?
  if [ -n "$R" ]; then rm -f "$R"/edge_*.csv.tmp; fi
  if [ "$rc" -ne 0 ]; then
    echo "hres smoke_r2 validate: FAILED (exit $rc) $(date -Iseconds); rerun resumes from files already written" >&2
  fi
  return "$rc"
}

main() {
  local REPO DATA SMOKE2_SRC HRES_SMOKE_SRC HAVE_SMOKE2 HAVE_HRES_SMOKE USE_EDGE_BOUNDS EDGE_HEAD
  local m a out M COARSE_REF SEED_TEST_SIM SEED_TEST_MODEL
  REPO=${NVFX_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
  if [ -f "$REPO/tools/wsl_env.sh" ]; then
    # shellcheck source=/dev/null
    . "$REPO/tools/wsl_env.sh"
  fi
  [ -f "$REPO/tools/study_3d/nvfx_bin_link.sh" ] || { echo "error: repo not found at $REPO" >&2; return 1; }
  cd "$REPO"
  trap cleanup EXIT

  if [ -n "${1:-}" ] && [ -x "$1" ]; then B=$1; else
    B=$(BUILD="${BUILD:-${BUILD_DIR:-$REPO/build-cuda}/nvfx_3d}" bash "$REPO/tools/study_3d/nvfx_bin_link.sh")
  fi
  DATA=${NEURALVFX_DATA:-$HOME/nvfx-data}
  ROOT=${2:-$DATA/hres_3d/smoke_r2}
  R=results/3d/hres/smoke_r2
  TARGS=()
  [ -n "${THREADS:-}" ] && TARGS=(--threads "$THREADS")
  HARGS=(--study hres --root "$ROOT" --effect smoke_r2)
  WARGS=(--winds 0.05,0.25,0.5,0.75,0.95 --seconds 30)
  SMOKE2_SRC=$DATA/n2_3d/models/smoke2.nvfx3
  HRES_SMOKE_SRC=$DATA/hres_3d/smoke/models/smoke_hres.nvfx3
  [ -f "$HRES_SMOKE_SRC" ] || HRES_SMOKE_SRC=$REPO/models/3d/smoke_hres.nvfx3
  HAVE_SMOKE2=0; HAVE_HRES_SMOKE=0
  [ -f "$SMOKE2_SRC" ] && HAVE_SMOKE2=1
  [ -f "$HRES_SMOKE_SRC" ] && HAVE_HRES_SMOKE=1
  COARSE_REF=C3
  SEED_TEST_SIM=9871900
  SEED_TEST_MODEL=9872900

  USE_EDGE_BOUNDS=${USE_EDGE_BOUNDS:-1}
  RUNS=${EDGE_RUNS:-4}
  BARGS=()
  EDGE_HEAD="$R/edge_teacher.csv"
  if [ "$USE_EDGE_BOUNDS" = 1 ]; then
    BARGS=(--bounds "$R/edge_bounds.csv")
    [ -f "$R/edge_bounds.csv" ] && edge_done_multi "$EDGE_HEAD" "$RUNS" || {
      echo "error: missing bounds or teacher runs (run FORCE=1 STAGE=bounds on hres_smoke_r2.sh)"; return 1
    }
    echo "USE_EDGE_BOUNDS=1: teacher $EDGE_HEAD, bounds $R/edge_bounds.csv"
  fi

  mkdir -p "$R" "$ROOT/models"
  [ "$HAVE_SMOKE2" = 1 ] && cp "$SMOKE2_SRC" "$ROOT/models/smoke2.nvfx3"
  [ "$HAVE_HRES_SMOKE" = 1 ] && cp "$HRES_SMOKE_SRC" "$ROOT/models/smoke_hres.nvfx3"

  echo "hres smoke_r2 validate: data $ROOT, results $REPO/$R"

  for m in C2 C3; do
    out="$R/edge_val_$m.csv"
    if edge_done "$out"; then echo "skip edge $m ($out)"; else
      rm -f "$out"
      nvfx edge "${HARGS[@]}" --model "$ROOT/models/$m.nvfx3" --label "$m" "${WARGS[@]}" --out "$out"
    fi
  done
  if [ "$HAVE_SMOKE2" = 1 ]; then
    out="$R/edge_val_smoke2.csv"
    if edge_done "$out"; then echo "skip edge smoke2"; else
      rm -f "$out"
      nvfx edge "${HARGS[@]}" --model "$ROOT/models/smoke2.nvfx3" --label smoke2 "${WARGS[@]}" --out "$out"
    fi
  fi
  if [ "$HAVE_HRES_SMOKE" = 1 ]; then
    out="$R/edge_val_smoke_hres.csv"
    if edge_done "$out"; then echo "skip edge smoke_hres"; else
      rm -f "$out"
      nvfx edge "${HARGS[@]}" --model "$ROOT/models/smoke_hres.nvfx3" --label smoke_hres "${WARGS[@]}" --out "$out"
    fi
  fi
  combine_edge "$R/edge_val.csv" "$EDGE_HEAD" "$R/edge_val_C2.csv" "$R/edge_val_C3.csv" \
    "$R/edge_val_smoke2.csv" "$R/edge_val_smoke_hres.csv"
  nvfx edgecheck --csv "$R/edge_val.csv" "${BARGS[@]}" | tee "$R/edgecheck_val.md"

  nvfx truth "${HARGS[@]}" --split val
  nvfx real "${HARGS[@]}" --split val

  for m in C2 C3; do
    nvfx track "${HARGS[@]}" --split val --method "$m" --model "$ROOT/models/$m.nvfx3"
    nvfx play "${HARGS[@]}" --split val --method "$m" --model "$ROOT/models/$m.nvfx3"
  done
  [ "$HAVE_SMOKE2" = 1 ] && nvfx play "${HARGS[@]}" --split val --method smoke2 --model "$ROOT/models/smoke2.nvfx3"
  [ "$HAVE_HRES_SMOKE" = 1 ] && nvfx play "${HARGS[@]}" --split val --method smoke_hres --model "$ROOT/models/smoke_hres.nvfx3"

  nvfx track "${HARGS[@]}" --split val --method coarse --model "$ROOT/models/$COARSE_REF.nvfx3"
  nvfx play "${HARGS[@]}" --split val --method coarse --model "$ROOT/models/$COARSE_REF.nvfx3"
  [ "$HAVE_SMOKE2" = 1 ] && nvfx track "${HARGS[@]}" --split val --method hold --model "$ROOT/models/smoke2.nvfx3"

  for a in C2 C3; do
    [ "$HAVE_SMOKE2" = 1 ] && nvfx compare "${HARGS[@]}" --split val --a "$a" --b smoke2 --results "$R"
    nvfx compare "${HARGS[@]}" --split val --a "$a" --b coarse --results "$R"
  done
  [ "$HAVE_SMOKE2" = 1 ] && [ "$HAVE_HRES_SMOKE" = 1 ] && \
    nvfx compare "${HARGS[@]}" --split val --a smoke_hres --b smoke2 --results "$R"
  [ "$HAVE_SMOKE2" = 1 ] && nvfx compare "${HARGS[@]}" --split val --a "$COARSE_REF" --b hold --results "$R"
  [ "$HAVE_SMOKE2" = 1 ] && nvfx compare "${HARGS[@]}" --split val --a smoke2 --b coarse --results "$R"

  nvfx choose "${HARGS[@]}" --split val --cands C2,C3 --edge "$R/edge_val.csv" "${BARGS[@]}" | tee "$R/choose_val.md"

  if [ -n "${CHOSEN:-}" ]; then echo "CHOSEN from env: $CHOSEN"; else
    CHOSEN=$(grep '^chosen:' "$R/choose_val.md" 2>/dev/null | tail -1 | awk '{print $2}' || true)
    CHOSEN=${CHOSEN:-C3}
    echo "CHOSEN parsed: $CHOSEN"
  fi
  [ -f "$ROOT/models/$CHOSEN.nvfx3" ] || { echo "error: missing $ROOT/models/$CHOSEN.nvfx3"; return 1; }

  cp "$ROOT/models/$CHOSEN.nvfx3" "$ROOT/models/chosen.nvfx3"
  freeze_once "$ROOT/models/chosen.nvfx3"
  [ "$HAVE_SMOKE2" = 1 ] && freeze_once "$ROOT/models/smoke2.nvfx3"

  nvfx truth "${HARGS[@]}" --split test
  nvfx real "${HARGS[@]}" --split test

  rm -f "$R/edge_test_sim.csv" "$R/edge_test_learned.csv" "$R/edge_test_smoke2.csv"
  nvfx edge "${HARGS[@]}" --sim "${WARGS[@]}" --seed-base "$SEED_TEST_SIM" --out "$R/edge_test_sim.csv"
  nvfx edge "${HARGS[@]}" --model "$ROOT/models/chosen.nvfx3" --label learned "${WARGS[@]}" --seed-base "$SEED_TEST_MODEL" --out "$R/edge_test_learned.csv"
  [ "$HAVE_SMOKE2" = 1 ] && nvfx edge "${HARGS[@]}" --model "$ROOT/models/smoke2.nvfx3" --label smoke2 "${WARGS[@]}" --seed-base "$SEED_TEST_MODEL" --out "$R/edge_test_smoke2.csv"
  combine_edge "$R/edge_test.csv" "$R/edge_test_sim.csv" "$R/edge_test_learned.csv" "$R/edge_test_smoke2.csv"
  nvfx edgecheck --csv "$R/edge_test.csv" "${BARGS[@]}" | tee "$R/edgecheck_test.md"

  FZ=(--frozen "$R/frozen.csv")
  for m in learned smoke2 coarse hold; do
    case $m in
      learned) M="$ROOT/models/chosen.nvfx3" ;;
      smoke2) M="$ROOT/models/smoke2.nvfx3"; [ "$HAVE_SMOKE2" = 1 ] || continue ;;
      coarse) M="$ROOT/models/$CHOSEN.nvfx3" ;;
      hold) M="$ROOT/models/smoke2.nvfx3"; [ "$HAVE_SMOKE2" = 1 ] || continue ;;
    esac
    if [ "$m" = smoke2 ]; then
      nvfx play "${HARGS[@]}" --split test --method smoke2 --model "$M" "${FZ[@]}"
      continue
    fi
    nvfx track "${HARGS[@]}" --split test --method "$m" --model "$M" "${FZ[@]}"
    [ "$m" != hold ] && nvfx play "${HARGS[@]}" --split test --method "$m" --model "$M" "${FZ[@]}"
  done

  [ "$HAVE_SMOKE2" = 1 ] && nvfx compare "${HARGS[@]}" --split test --a learned --b smoke2 --results "$R"
  nvfx compare "${HARGS[@]}" --split test --a learned --b coarse --results "$R"

  echo "done: chosen $CHOSEN -> $ROOT/models/chosen.nvfx3; tables under $R"
}

main "$@"; exit $?
