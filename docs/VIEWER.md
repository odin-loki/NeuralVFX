# The viewer

Status: **current**. Audience: dev, artists.

`nvfx_viewer` plays trained effects live with a slider for every control, and shows them next to the reference
clip, a flipbook of that clip at the effect's memory, and the fluid simulation running at the same controls, each
with its cost per frame. It renders through the same runtime a game uses.

![The viewer: a fire model next to its reference clip and a BC3 flipbook of the same memory](figures/viewer.png)

## Build and run

The viewer is optional (it needs GLFW and OpenGL and fetches Dear ImGui with git at a pinned tag):

```sh
sudo apt install libglfw3-dev libgl-dev        # plus xvfb to run it headless
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 -DNEURALFX_BUILD_VIEWER=ON
cmake --build build --target nvfx_viewer
build/viewer/nvfx_viewer fire.nvfx smoke.nvfx --clip fire_reference.nfxclip
build/viewer/nvfx_viewer --demo                 # an untrained demo effect, no files needed
```

Headless (for tests and screenshots): `xvfb-run -a build/viewer/nvfx_viewer fire.nvfx --screenshot out.png --frames 30`.

## Panels

| panel | what |
|---|---|
| learned controls | one slider per control the effect was trained with (names stored in the `.nvfx` file) |
| variation | seed, "new seed", a training variation to replay, and the drift time between variations |
| exact runtime controls | playback speed, hue rotation, brightness |
| view | size (any multiple of 16 for grid effects), background, pause and restart, the comparison panels |
| cost | ms per evaluation (rolling mean), MAC per pixel, ISA, stored and resident weight size, scratch per instance, and the simulation's ms per frame when it runs |

The flipbook panel keeps as many BC3 frames of the reference clip as fit in the effect's stored size, so the two
cost the same memory.
