#!/usr/bin/env bash
# Quiet round 7, the blocks after tools/quiet/round7.sh's (the same session folder and harness): REPORT §16.6's costs.
#   tools/quiet/round7_extra.sh BUILD OUT [BLOCK ...]
#   O1  the detail layer's second pass (study O): the waterfall (swirl 0) and launch (swirl 1, scale 9), one pass
#       against two, 128 and 256 px, int8 and float (nvfx_ordered cost, 9 interleaved repetitions, core 3)
#   O2  the replaced mushroom, splash, avalanche and dust explosion: the round's file against study O's and study Q's,
#       the files that ship now, the same way
#   S4  the lift-off scene as shipped (three launch tiles, two passes) on 2 threads, int8 and float in turn, 3 rounds
#   K10-K18  nine more rounds of round7.sh's K blocks (the 33 models at 64, 128 and 256 px): the host switched between a
#       fast and a slow state for single configurations, so the per-model medians needed more than nine rounds
#   E2  the fireball at 1280 x 720 once more, each configuration twice (as round7.sh's E1): the session's very end
set -uo pipefail
BUILD=$(cd "${1:?BUILD}" && pwd)
OUT=${2:?OUT}
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../.." && pwd)
DATA=${NEURALVFX_DATA:-/root/nvfx-data}
export NEURALVFX_DATA=$DATA
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
# shellcheck source=quiet.sh
. "$HERE/quiet.sh" "$OUT"

ordered() {  # CSV EFFECT CURRENT CANDS
  local p
  for p in int8 float; do
    taskset -c 3 "$BUILD/nvfx_ordered" cost --effect "$2" --current "$3" --cand "$4" --sizes 128,256 --precision "$p" --reps 9 \
      --out "$QUIET_BLOCK_DIR/$1" || return 1
  done
}
passes() {
  ordered o_cost_passes.csv waterfall "$DATA/o_fix/cost/waterfall_sw0.nvfx" "passes2=$DATA/o_fix/cost/waterfall_sw0_p2.nvfx" &&
    ordered o_cost_passes.csv launch "$DATA/o_fix/cost/launch_lib.nvfx" "passes2=$DATA/o_fix/cost/launch_lib_p2.nvfx"
}
models() {
  ordered o_cost_models.csv mushroom "$DATA/fx_blaze/models/d/mushroom.nvfx" "o=$DATA/o_fix/models/mushroom.nvfx,q=$DATA/q_fix/models/mushroom.nvfx" &&
    ordered o_cost_models.csv splash "$DATA/fx_water/models/d/splash.nvfx" "o=$DATA/o_fix/models/splash.nvfx" &&
    ordered o_cost_models.csv avalanche "$DATA/fx_earth/models/d/avalanche.nvfx" "q=$DATA/q_fix/models/avalanche.nvfx" &&
    ordered o_cost_models.csv dustexplosion "$DATA/fx_fire2/models/d/dustexplosion.nvfx" "q=$DATA/q_fix/models/dustexplosion.nvfx"
}
MODELS=$DATA/library/models
ALL33=avalanche,candle,cloud,coals,crownfire,dust,dustexplosion,eruption,explosion,fire,fog,fountain,geyser,haboob,heavygas,ink,jet,launch,lava,magic,mushroom,orb,puffs,rainshaft,seaspray,smoke,smokering,splash,steam,vortex,waterfall,wildfire,windowfire
studyk() {  # one round of round7.sh's K blocks
  "$BUILD/nvfx_study_k" timing --effects "$ALL33" --models "$MODELS" --sizes 64,128,256 --isas avx2,avx512 --candidates int8_second \
    --rounds 1 --frames 200 --core 3 --root "$DATA" --out "$QUIET_BLOCK_DIR" --csv timing.csv --label r7
}
fireball_end() {
  local r k cfg cfgs=(float float_avx2 int8)
  for r in 13 14; do
    for ((k = 0; k < 3; k++)); do
      cfg=${cfgs[$(((k + r) % 3))]}
      local opt=(--precision float)
      [ "$cfg" = int8 ] && opt=(--precision int8)
      [ "$cfg" = float_avx2 ] && opt=(--precision float --isa avx2)
      "$BUILD/nvfx_fireball" --models "$DATA/experiments/models/d" --no-video --threads 4 --width 1280 --height 720 --seconds 9 "${opt[@]}" \
        --profile "$QUIET_BLOCK_DIR/fireball_${cfg}_720_r$r.csv" > "$QUIET_BLOCK_DIR/fireball_${cfg}_720_r$r.txt" || return 1
    done
  done
}
liftoff() {
  local r k p
  for r in 1 2 3; do
    for k in 0 1; do
      p=default
      [ $(((k + r) % 2)) = 1 ] && p=float
      local opt=()
      [ "$p" = float ] && opt=(--precision float)
      "$BUILD/nvfx_scene_script" --script "$SRC/examples/scenes/liftoff.nvfxs" --models "$DATA/fx_flow/models/scene" --no-video --threads 2 \
        "${opt[@]}" --profile "$QUIET_BLOCK_DIR/liftoff_${p}_r$r.csv" > "$QUIET_BLOCK_DIR/liftoff_${p}_r$r.txt" 2>&1 || return 1
    done
  done
}

ORDER=(O1 O2 S4 K10 K11 K12 K13 K14 K15 K16 K17 K18 E2)
[ $# -gt 0 ] && ORDER=("$@")
quiet_start
trap quiet_stop EXIT
for b in "${ORDER[@]}"; do
  echo "$(date -u +%FT%TZ) block $b: load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/run.log"
  case $b in
    O1) block O1 passes ;;
    O2) block O2 models ;;
    S4) block S4 liftoff ;;
    K1[0-8]) block "$b" studyk ;;
    E2) block E2 fireball_end ;;
    *) echo "unknown block $b" >&2 ;;
  esac
  echo "$(date -u +%FT%TZ) block $b done: rc $? load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/run.log"
done
