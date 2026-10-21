#!/usr/bin/env bash
# Quiet re-timing, round 7 (results/quiet/README.md, "Round 7"): every cost measured on the busy machine since round 9,
# in one session, each block run only when the machine is idle (tools/quiet/quiet.sh: the gate, the foreign load).
#
#   tools/quiet/round7.sh BUILD OUT [BLOCK ...]
#
# BUILD: a Release build of this tree (with nvfx_detail_probe.so); OUT: the session's folder (blocks.csv, monitor.csv,
# logs/, raw/BLOCK/ with each block's CSVs). With no BLOCK every block runs, in the order below; blocks named run alone
# (a block that ran is run again). OLD_PROBE is the detail layer's old runtime (tools/detail_bench/build_old.sh
# c846c80 DIR; default DIR/nvfx_detail_probe_old.so under OUT/../detail-old). DRY=1 runs each with tiny counts (a check
# of the commands, not a timing). Data from NEURALVFX_DATA (default /root/nvfx-data).
#
# The blocks (each interleaves what it compares; the rounds' counts give 95% order-statistic intervals for medians):
#   F1-F5  the fireball (study D's v1 models, COMPOSE.md §7) at 1280 x 720 and 1920 x 1080, 4 threads: float (the
#          AVX-512 build), float on the AVX2 build (round 6's float) and int8, in turn (the order rotated), 2 rounds a
#          block; as round 6 ran it (nvfx_fireball --no-video --profile, checksums on)
#   K1-K9  the 33 shipped 2D models (DATA/library/models) at 64, 128 and 256 px, float and int8 (the stepper's second
#          convolution, the default with VNNI) on AVX2 and AVX-512, one pinned core, a round a block (nvfx_study_k timing)
#   L1-L3  12 of them at 512 px, the same, 3 rounds a block
#   D1-D3  the round documents' tables (EFFECTS_EARTH, _AIR, _FIRE2, _ELEMENTS §5): nvfx_experiment d-timing on the
#          twelve rounds' models as shipped and study D's fire and explosion, 3 rounds a block
#   P1     study P's cost: N, the half-size instance, bilinear and learned 2x at 256 and 512 px (nvfx_study_p cost,
#          30 rounds of 20 frames, one pinned core)
#   X1-X6  the detail layer, old (c846c80) against new in one process (nvfx_detail_bench, as REPORT §19): 2D at 128,
#          256 and 512 px (AVX2, then AVX-512; 12 rounds of 16 frames), the volume (10 rounds of 6 steps), the 3D layer
#          alone
#   V1-V6  the 3D steps: N2's C4 fire, study hres's fire and study R's smoke through the volume API on 1 and 4 threads
#          (nvfx_3d timing: 5 starts of 60 frames, pictures 256 and 512), the models in turn (rotated), a round a block
#   S1-S3  the scenes volcano, crypt, forest edge and coast on 2 threads, the default (int8) and float in turn,
#          2 rounds a block (nvfx_scene_script --no-video --profile)
#   W1-W6  the weather day scene, likewise, a round a block
#   U1-U3  the FUSE demo (nvfx_fuse_demo --check: 3 workers and the game thread alone), one run a block
#   E1     the fireball at 1280 x 720 again (each configuration twice): the session's end against its start
set -uo pipefail
BUILD=$(cd "${1:?BUILD}" && pwd)
OUT=${2:?OUT}
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../.." && pwd)
DATA=${NEURALVFX_DATA:-/root/nvfx-data}
export NEURALVFX_DATA=$DATA
OLD_PROBE=${OLD_PROBE:-$(dirname "$OUT")/detail-old/nvfx_detail_probe_old.so}
NEW_PROBE=$BUILD/nvfx_detail_probe.so
DRY=${DRY:-0}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
# shellcheck source=quiet.sh
. "$HERE/quiet.sh" "$OUT"

MODELS=$DATA/library/models
ALL33=avalanche,candle,cloud,coals,crownfire,dust,dustexplosion,eruption,explosion,fire,fog,fountain,geyser,haboob,heavygas,ink,jet,launch,lava,magic,mushroom,orb,puffs,rainshaft,seaspray,smoke,smokering,splash,steam,vortex,waterfall,wildfire,windowfire
SUB512=fire,smoke,explosion,mushroom,wildfire,launch,waterfall,fog,orb,geyser,haboob,dustexplosion
ROUNDS12=lava,eruption,avalanche,candle,smokering,heavygas,crownfire,windowfire,dustexplosion,geyser,haboob,seaspray
if [ "$DRY" = 1 ]; then K_ROUNDS=1 K_FRAMES=10 FB_SECONDS=2 X_ROUNDS=1 X_FRAMES=2 P_ROUNDS=2 V_REPS=1 V_FRAMES=3 SC_FRAMES=20 BLOCK_ROUNDS=1
else K_ROUNDS=3 K_FRAMES=200 FB_SECONDS=9 X_ROUNDS=0 X_FRAMES=0 P_ROUNDS=30 V_REPS=5 V_FRAMES=60 SC_FRAMES=0 BLOCK_ROUNDS=0; fi

# d-timing reads DIR/models/d/NAME.nvfx: the shipped files of the twelve, study D's v1 fire and explosion
DT=$OUT/dtime_models
mkdir -p "$DT/models/d"
for m in ${ROUNDS12//,/ }; do ln -sfn "$(readlink -f "$MODELS/$m.nvfx")" "$DT/models/d/$m.nvfx"; done
for m in fire explosion; do ln -sfn "$DATA/experiments/models/d/$m.nvfx" "$DT/models/d/$m.nvfx"; done

# rounds of a block: R0 is the first round's index, N the count (BLOCK_ROUNDS overrides it in a dry run)
nrounds() { [ "$BLOCK_ROUNDS" -gt 0 ] && echo "$BLOCK_ROUNDS" || echo "$1"; }

fireball_rounds() {  # R0 N SIZES
  local r0=$1 n sizes=$3 r k i w h cfg
  n=$(nrounds "$2")
  local cfgs=(float float_avx2 int8)
  for ((i = 0; i < n; i++)); do
    r=$((r0 + i))
    for size in $sizes; do
      if [ "$size" = 1080 ]; then w=1920 h=1080; else w=1280 h=720; fi
      for ((k = 0; k < 3; k++)); do
        cfg=${cfgs[$(((k + r) % 3))]}
        local opt=(--precision float)
        [ "$cfg" = int8 ] && opt=(--precision int8)
        [ "$cfg" = float_avx2 ] && opt=(--precision float --isa avx2)
        echo "fireball $cfg ${size}p round $r load $(cut -d' ' -f1-3 /proc/loadavg)"
        "$BUILD/nvfx_fireball" --models "$DATA/experiments/models/d" --no-video --threads 4 --width $w --height $h --seconds "$FB_SECONDS" \
          "${opt[@]}" --profile "$QUIET_BLOCK_DIR/fireball_${cfg}_${size}_r$r.csv" > "$QUIET_BLOCK_DIR/fireball_${cfg}_${size}_r$r.txt" || return 1
      done
    done
  done
}

studyk_rounds() {  # EFFECTS SIZES ROUNDS: rounds of nvfx_study_k timing (renumbered when merged)
  "$BUILD/nvfx_study_k" timing --effects "$1" --models "$MODELS" --sizes "$2" --isas avx2,avx512 --candidates int8_second \
    --rounds "$(nrounds "$3")" --frames "$K_FRAMES" --core 3 --root "$DATA" --out "$QUIET_BLOCK_DIR" --csv timing.csv --label r7
}

dtiming_rounds() {  # N
  local n i
  n=$(nrounds "$1")
  for ((i = 1; i <= n; i++)); do
    echo "d-timing round $i load $(cut -d' ' -f1-3 /proc/loadavg)"
    "$BUILD/nvfx_experiment" d-timing --root "$DT" --results "$QUIET_BLOCK_DIR/r$i" --effects "$ROUNDS12,fire,explosion" || return 1
  done
}

detail() {  # MODE ARGS...: nvfx_detail_bench (the old build against this one), its summary beside its CSV
  local mode=$1 csv=$QUIET_BLOCK_DIR/$2.csv
  shift 2
  local extra=()
  [ "$X_ROUNDS" -gt 0 ] && extra=(--rounds "$X_ROUNDS" --frames "$X_FRAMES")
  if [ "$mode" = 3d-detail ]; then
    "$BUILD/nvfx_detail_bench" "$mode" --new "$NEW_PROBE" "$@" "${extra[@]}" --out "$csv" || return 1
  else
    "$BUILD/nvfx_detail_bench" "$mode" --old "$OLD_PROBE" --new "$NEW_PROBE" "$@" "${extra[@]}" --out "$csv" || return 1
  fi
  "$BUILD/nvfx_detail_bench" summary --in "$csv" --out "${csv%.csv}.md" > /dev/null
}

studyp() {
  "$BUILD/nvfx_study_p" cost --net cand_shared --sizes 256,512 --rounds "$P_ROUNDS" --frames 20 --core 3 --label quiet7 --out "$QUIET_BLOCK_DIR/p_cost.csv"
}

volume_rounds() {  # R0 N: the three 3D models in turn (rotated), 1 and 4 threads
  local r0=$1 n r i k
  n=$(nrounds "$2")
  local names=(fire hres smoke)
  for ((i = 0; i < n; i++)); do
    r=$((r0 + i))
    for ((k = 0; k < 3; k++)); do
      local m=${names[$(((k + r) % 3))]} study model
      case $m in
        fire) study=n2 model=$DATA/n2_3d/models/chosen.nvfx3 ;;
        hres) study=hres model=$DATA/hres_3d/fire/models/hres_fire.nvfx3 ;;
        smoke) study=r model=$DATA/r_smoke3d/models/chosen.nvfx3 ;;
      esac
      echo "3d $m round $r load $(cut -d' ' -f1-3 /proc/loadavg)"
      "$BUILD/nvfx_3d" --study "$study" timing --model "$model" --threads 1,4 --reps "$V_REPS" --frames "$V_FRAMES" \
        --out "$QUIET_BLOCK_DIR/timing_${m}_r$r.csv" || return 1
    done
  done
}

scene_rounds() {  # R0 N SCENES: each scene at the default precision (int8 here) and at float, in turn (rotated)
  local r0=$1 n r i k s
  n=$(nrounds "$2")
  for ((i = 0; i < n; i++)); do
    r=$((r0 + i))
    for s in $3; do
      local models
      case $s in
        volcano) models=$DATA/fx_earth/models/scene ;;
        crypt) models=$DATA/fx_air/models/scene ;;
        forestedge) models=$DATA/fx_fire2/models/scene ;;
        coast) models=$DATA/fx_elements/models/scene ;;
        weather_day) models=$MODELS ;;
      esac
      for ((k = 0; k < 2; k++)); do
        local p=default opt=()
        [ $(((k + r) % 2)) = 1 ] && p=float opt=(--precision float)
        [ "$SC_FRAMES" -gt 0 ] && opt+=(--frames "$SC_FRAMES")
        echo "scene $s $p round $r load $(cut -d' ' -f1-3 /proc/loadavg)"
        "$BUILD/nvfx_scene_script" --script "$SRC/examples/scenes/$s.nvfxs" --models "$models" --no-video --threads 2 "${opt[@]}" \
          --profile "$QUIET_BLOCK_DIR/${s}_${p}_r$r.csv" > "$QUIET_BLOCK_DIR/${s}_${p}_r$r.txt" 2>&1 || return 1
      done
    done
  done
}

fuse() {
  local opt=()
  [ "$DRY" = 1 ] && opt=(--seconds 1)
  "$BUILD/engines/fuse/nvfx_fuse_demo" --pack "$BUILD/engines/fuse/fuse_demo.nvfxp" --check --workers 3 "${opt[@]}" > "$QUIET_BLOCK_DIR/fuse.txt" 2>&1
}

run_named() {
  local b=$1 i
  case $b in
    F[1-5]) i=${b#F}; block "$b" fireball_rounds $((2 * i - 1)) 2 "720 1080" ;;
    K[1-9]) block "$b" studyk_rounds "$ALL33" 64,128,256 1 ;;
    L[1-3]) block "$b" studyk_rounds "$SUB512" 512 3 ;;
    D[1-3]) block "$b" dtiming_rounds 3 ;;
    X1) block X1 detail 2d frames_2d_128 --models "$MODELS" --sizes 128 --isas avx2,avx512 ;;
    X2) block X2 detail 2d frames_2d_256 --models "$MODELS" --sizes 256 --isas avx2,avx512 ;;
    X3) block X3 detail 2d frames_2d_512_avx2 --models "$MODELS" --sizes 512 --isas avx2 ;;
    X4) block X4 detail 2d frames_2d_512_avx512 --models "$MODELS" --sizes 512 --isas avx512 ;;
    X5) block X5 detail 3d volume_3d --model "$DATA/n2_3d/models/chosen.nvfx3" --threads 1,4 --isas avx2,avx512 ;;
    X6) block X6 detail 3d-detail detail_3d --model "$DATA/n2_3d/models/chosen.nvfx3" --threads 1,4 --isas avx2,avx512 ;;
    P1) block P1 studyp ;;
    V[1-6]) i=${b#V}; block "$b" volume_rounds "$i" 1 ;;
    S[1-3]) i=${b#S}; block "$b" scene_rounds $((2 * i - 1)) 2 "volcano crypt forestedge coast" ;;
    W[1-6]) i=${b#W}; block "$b" scene_rounds "$i" 1 weather_day ;;
    U[1-3]) block "$b" fuse ;;
    E1) block E1 fireball_rounds 11 2 720 ;;
    *) echo "unknown block $b" >&2; return 2 ;;
  esac
}

ORDER=(F1 F2 F3 F4 F5 K1 K2 K3 K4 K5 K6 K7 K8 K9 L1 L2 L3 D1 D2 D3 P1 X1 X2 X3 X4 X5 X6 V1 V2 V3 V4 V5 V6 S1 S2 S3 W1 W2 W3 W4 W5 W6 U1 U2 U3 E1)
[ $# -gt 0 ] && ORDER=("$@")
quiet_start
trap quiet_stop EXIT
for b in "${ORDER[@]}"; do
  echo "$(date -u +%FT%TZ) block $b: load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/run.log"
  run_named "$b"
  echo "$(date -u +%FT%TZ) block $b done: rc $? load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/run.log"
done
