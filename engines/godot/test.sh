#!/usr/bin/env bash
# The Godot 4 GDExtension, built and tested headless (docs/ENGINES.md §8):
#   1. the official Godot 4.4.1 Linux binary, downloaded once (pinned, SHA-512 checked) into $DATA/godot (not git);
#   2. libnvfx, the extension (NEURALFX_BUILD_GODOT=ON: godot-cpp fetched at its pinned tag) and nvfx_scene_script,
#      built in build-godot/ (the extension and libnvfx are copied into demo/bin/);
#   3. the script runner's frame N of the fireball (when N is a keyframe), as the reference; the references and the
#      nodes run their effects at float (rollout effects run at int8 by default since docs/REPORT.md §13), and on one
#      build: neither forces an ISA, so both take the runtime's float build for this CPU (AVX-512 where it has it, else
#      AVX2, §13.10), from the same code (keep it so: the nodes have no ISA setting, so the references take none);
#   4. the demo project imported (which registers the extension), then demo/test.gd run headless: the fireball for N
#      frames, the frame saved as PNG and checked (not empty, the same pixels as the reference, heat where the fire
#      is), a script error with its line and column, and one effect drawn;
#   5. the effect library (docs/LIBRARY.md), when its effects are in DATA/library/models: nvfx_gallery --shot draws a
#      stage with five library effects spawned at given times and a rocket trail moved every frame as the reference,
#      and demo/test_library.gd plays the same stage in a NeuralVFXStage with the engine layers on (listing, spawns, a
#      parameter, a move, a stop, the moving trail, the layer textures, the same pixels to the byte).
#
#   engines/godot/test.sh [--frames 45] [--out DIR] [--jobs 2]
#
# DATA is $NEURALVFX_DATA (default ~/nvfx-data): the download, the effects (experiments/models/d, library/models) and
# the output (DATA/godot/fireball_godot.png, library_godot.png). Exit code: 0 when every check passed.
set -euo pipefail

GODOT_VERSION=4.4.1-stable
GODOT_ZIP=Godot_v${GODOT_VERSION}_linux.x86_64.zip
GODOT_SHA512=ef4e76880a514257175544952c61191106fdef3095b909bafed9fcbeb230c3e5533920a0f3012882dd4bbde83028a67549825794e2d2c3cf76eba7918b71370e

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
data=${NEURALVFX_DATA:-$HOME/nvfx-data}
frames=45
out=$data/godot
jobs=2
while [[ $# -gt 0 ]]; do
  case $1 in
    --frames) frames=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    *) echo "unknown option $1 (see the header of $0)" >&2; exit 2 ;;
  esac
done
models=$data/experiments/models/d
for f in fire.nvfx smoke.nvfx explosion.nvfx; do
  [[ -f $models/$f ]] || { echo "$models/$f not found: the study D effects are needed" >&2; exit 77; }
done
mkdir -p "$out"

# 1. Godot
godot_dir=$data/godot/bin
godot=$godot_dir/Godot_v${GODOT_VERSION}_linux.x86_64
if [[ ! -x $godot ]]; then
  mkdir -p "$data/godot/dl" "$godot_dir"
  zip=$data/godot/dl/$GODOT_ZIP
  [[ -f $zip ]] || curl -fsSL -o "$zip" "https://github.com/godotengine/godot/releases/download/$GODOT_VERSION/$GODOT_ZIP"
  echo "$GODOT_SHA512  $zip" | sha512sum -c --quiet
  unzip -o -q "$zip" -d "$godot_dir"
fi
"$godot" --headless --version

# 2. the extension and the script runner
build=$repo/build-godot
cxx=${CXX:-$(command -v g++-14 || echo c++)}
cc=${CC:-$(command -v gcc-14 || echo cc)}
nice -n 10 cmake -S "$repo" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_C_COMPILER="$cc" -DNEURALFX_WERROR=ON \
  -DNEURALFX_BUILD_GODOT=ON -DNEURALFX_BUILD_TESTS=OFF -DNEURALFX_BUILD_BENCH=OFF -DNEURALFX_BUILD_TOOLS=ON > "$out/configure.log"
nice -n 10 cmake --build "$build" -j "$jobs" --target nvfx_godot nvfx_scene_script nvfx_gallery > "$out/build.log"
ls -l "$here/demo/bin"

# 3. the reference: the script runner's keyframes up to frame N (frame_NNN.png exists when N is a keyframe)
rm -rf "$out/reference"
nice -n 10 "$build/nvfx_scene_script" --script "$repo/examples/scenes/fireball.nvfxs" --models "$models" --keyframes "$out/reference" \
  --no-video --frames $((frames + 1)) --threads 2 --precision float > "$out/reference.log"
head -1 "$out/reference.log"  # the build the reference ran on (the nodes take the same)
reference=$(printf '%s/reference/frame_%03d.png' "$out" "$frames")
[[ -f $reference ]] || { echo "frame $frames is not a keyframe of the script: no reference"; reference=""; }

# 4. the demo project, headless
(cd "$here/demo" && nice -n 10 "$godot" --headless --path . --import > "$out/import.log" 2>&1) || true
grep -q neuralvfx.gdextension "$here/demo/.godot/extension_list.cfg" || { echo "the import did not register the extension (see $out/import.log)" >&2; exit 1; }
cd "$here/demo"
set +e  # the test's own errors (the broken script it reports on purpose) are filtered out
nice -n 10 "$godot" --headless --path . --script res://test.gd -- --script "$repo/examples/scenes/fireball.nvfxs" --models "$models" \
  --out "$out/fireball_godot.png" --frames "$frames" --reference "$reference" 2>&1 | grep -v -e '^ERROR: NeuralVFXScene: broken' -e 'at: push_error'
status=${PIPESTATUS[0]}
set -e
echo "frame: $out/fireball_godot.png"

# 5. the effect library: the same stage and spawns as demo/test_library.gd (keep them the same)
library_models=$data/library/models
if [[ -f $library_models/fire.nvfx && -f $library_models/steam.nvfx && -f $library_models/explosion.nvfx && -f $library_models/orb.nvfx ]]; then
  nice -n 10 "$build/nvfx_gallery" --library "$repo/library" --models "$library_models" --shot 960x540 --ground 470 \
    --spawn campfire@220,470,1,0.3 --spawn steam_vent@480,470,0.8,0.4 --spawn magic_burst@760,260,1,0.6 \
    --spawn electric_arc@620,470,1,0.2 --spawn plasma_orb@110,150,0.8,0.5 --spawn rocket_trail@100,330,0.8,0.1,250,-60 \
    --frames "$frames" --threads 2 --precision float --png "$out/library_reference.png" > "$out/library_reference.log"
  set +e
  nice -n 10 "$godot" --headless --path . --script res://test_library.gd -- --library "$repo/library" --library_models "$library_models" \
    --out "$out/library_godot.png" --frames "$frames" --reference "$out/library_reference.png" 2>&1
  library_status=$?
  set -e
  echo "library frame: $out/library_godot.png"
  status=$((status + library_status))
else
  echo "the library's effects are not in $library_models: the library test is skipped"
fi
exit "$status"
