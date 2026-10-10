# NeuralVFX

Neural visual effects for games that run on **one CPU core**. Each effect (fire, smoke, an explosion) is small and
trained offline. At run time it draws sprites (64-256 px) from time, artist controls (intensity, wind, turbulence) and
a seed, with no GPU. The idea follows NVIDIA's small-network-per-asset work (RTX Neural Shaders, Neural Texture
Compression), but runs on a CPU instead of tensor cores.

Two kinds of effect:

- **Frame models** (studies A to C): a network that stores an effect's frames in feature volumes and reproduces them.
- **Rollout effects** (study D): a few stored simulation states (start points) and a small network that moves the
  effect forward one frame at a time, driven by the same kind of noise that forces the simulation. The effect is a
  chaotic system: after about a second the start point no longer decides the picture, the noise does. So the frames
  in between are not stored, only made to look right, and every seed gives a new run that never repeats.

Rollout effects can also run **on one another** (a prototype, [docs/COMPOSE.md](docs/COMPOSE.md)): an explosion's
blast bends a fire, its cloud is handed to the smoke model, embers light new fires, and scripted force fields shape
them. `nvfx_fireball` renders a 9-second fireball scene that way and profiles every stage.

C++23, no third-party runtime dependencies, a C API for engines, no Python.

- **Results:** [docs/REPORT.md](docs/REPORT.md) covers quality against memory and milliseconds, flipbooks at
  matched memory, NVIDIA's claims and the honest limits. The generated tables with intervals are in
  [results/experiments/SUMMARY.md](results/experiments/SUMMARY.md).
- **Plan and decisions:** [docs/PLAN.md](docs/PLAN.md).
- **Engine integration:** [docs/ENGINES.md](docs/ENGINES.md) (C API, threading, Unreal, Unity, Godot).
- **Data and licences:** [docs/DATA.md](docs/DATA.md).
- **Viewer:** [docs/VIEWER.md](docs/VIEWER.md).

## Results in brief

From [docs/REPORT.md](docs/REPORT.md), measured on simulated fire, smoke and explosions at 128 x 128 on one AVX2 core:

- **Rollout effects (study D):** endless, controllable, never-repeating effects in **82 to 274 KB** per effect. At
  settings and seeds never seen in training, their detail statistically ties a 45 MB flipbook library and beats the
  1 MB control model on smoke. The learned dynamics follow a real run 2 to 5 dB better than the same simulation at
  the same resolution for the first second. Played in 6 s shards from the start points, a minute looks like the
  first 10 s. The price is **2.6 to 4.1 ms per 128 x 128 frame** (1.2 to 1.3 ms at 64 x 64) and 3.4 MB per playing
  instance.
- **Memory:** for equal quality, a network per effect clip needs **3.6 to 7.4 times less memory** than a flipbook;
  at equal memory it scores +2.8 to +7.0 dB higher (every 95% interval above zero). NVIDIA claims "up to 8x" for
  Neural Texture Compression; our flipbooks use our own BC3-layout encoder, so the ratio against BC7 would be lower.
- **Controls:** one 1 MB model per effect plays unseen settings better than a 45 MB flipbook library
  (+1.0 to +2.5 dB), but its unseen-setting flames look softer than the real simulation.
- **Variation:** endless non-repeating playback by drifting between learned variations; variations are softer than
  real ones and are blends of the training seeds.
- **Cost of frame models:** 0.38 ms (73 KB model) to 1.1 ms (132 KB model) per 128 x 128 frame; 0.1-0.3 ms at 64 x 64; 8-30 times
  cheaper than simulating, 50-150 times dearer than playing a flipbook.

![D: fire, 10 s at a setting never seen in training. Rows: the simulation; the 82 KB rollout effect with a new seed; the 1 MB control model; the nearest of 45 training flipbooks](docs/figures/d_fire_endless.png)

![A: rows are the reference smoke clip, the 132 KB network, a 128 KB flipbook at 64 px, and a 144 KB flipbook with 8 frames and motion vectors](docs/figures/a_smoke_compare.png)

![The viewer: the 1 MB fire control model (one model for every setting) next to the reference clip of one setting and a 1 MB flipbook of that one setting](docs/figures/viewer.png)

## How it works

| piece | where | what |
|---|---|---|
| simulation | `src/sim`, `nvfx_sim` | a 2D stable-fluids solver (temperature, soot, vorticity, curl noise, combustion expansion) that renders fire, smoke and explosions to premultiplied RGBA clips with labelled controls: unlimited training and test data |
| ingest | `src/core/ingest.cpp`, `nvfx_ingest` | owner footage into clips via ffmpeg, with a licence check and a licence register |
| models | `src/core/model.cpp` | a **grid** family (learned feature volumes over x, y and t, blended by the controls, sampled bilinearly, then a small MLP with FiLM conditioning) and a **conv** family (a learned latent, three upsampling convolutions); `.nvfx` files store features at 8 or 16 bits |
| rollout effects | `src/core/rollout.cpp`, `src/train/rollout_train.cpp`, `src/runtime/rt_rollout.hpp` | start points plus a learned stepper on a 32 x 32 grid (advection and pressure projection built in; the network supplies forces and the sub-grid closure, conditioned on the controls and fed the simulation's kind of noise), a detail layer that carries full-resolution heat and soot with the learned flow, and a per-pixel renderer; trained by backpropagation through time, then on its own rollouts with statistical losses |
| trainer | `src/train`, `nvfx_train` | hand-written gradients (checked against finite differences), Adam, multithreaded; per-clip variation codes |
| runtime | `src/runtime`, `include/neuralfx/nvfx.h` | the shipping library: C API, per-ISA SIMD (SSE2, AVX2, AVX-512), no allocation per frame, seeds and endless drift, exact hue, brightness and speed, bake to flipbook |
| baselines | `src/core/flipbook.cpp` | flipbooks at matched memory: frame count, resolution, raw or BC1/BC4 (BC3-layout) compression, motion vectors |
| lossless packing | `src/core/cm.cpp`, `nvfx_pack` | a context-mixing coder (context models, logistic mixer, APMs, binary arithmetic coder) that knows the tensors' shapes: `.nvfx` to `.nvfz` and back, bit-exact; flipbooks coded the same way for comparison ([results/compression](results/compression/README.md)) |
| metrics | `src/core/metrics.cpp` | PSNR (full and active-region), SSIM, temporal PSNR, flicker, spectrum and motion statistics, paired bootstrap |
| evaluation | `nvfx_experiment` | the whole study end to end: compression, controls, variation, timing, figures, report |
| viewer | `viewer/`, `nvfx_viewer` | Dear ImGui: sliders for every control, side by side with the reference, a flipbook and the live simulation |
| composed effects | `src/compose`, `nvfx_fireball` | a prototype: rollout effects coupled through their fields (tiles of one domain, hand-over between models, pushes, transfers, force fields), with particles, light, distortion and bloom; [docs/COMPOSE.md](docs/COMPOSE.md) |
| context mixing | `src/dcm`, `nvfx_dcm` | the owner's diffusion-context mixing (DCM) from CameraDetector: a PAQ8-style mixer with a frozen, versioned inference copy and a nested mixer search, k-means contexts without LibTorch, and a value-domain mixer (a value and a Laplace scale per prediction) for generating and coding effects; docs/DCM.md (study G) |

## Build and test

Ubuntu 24.04: `g++-14`, CMake 3.25+, Ninja, `libgtest-dev`, `zlib1g-dev` (and `ffmpeg` for videos and the video test).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14
cmake --build build
ctest --test-dir build                     # 115 tests: sim, metrics, codecs, gradients, runtime parity, composition, context mixing, lossless coder, allocation, C API
```

Options: `NEURALFX_BUILD_VIEWER` (GLFW + OpenGL; fetches Dear ImGui), `NEURALFX_BUILD_SHARED` (libnvfx.so for engines),
`NEURALFX_BUILD_TOOLS`, `NEURALFX_BUILD_TESTS`, `NEURALFX_BUILD_BENCH`, `NEURALFX_WERROR` (CI).

The trainer and the tools need AVX2 + FMA (checked at start). The runtime falls back to SSE2 code on older x86-64.

## Quick start

```sh
export NEURALVFX_DATA=$HOME/nvfx-data              # clips, weights and videos live here, never in git
build/nvfx_sim --effect fire --out $NEURALVFX_DATA/fire.nfxclip --sheet fire.png
build/nvfx_train --clips $NEURALVFX_DATA/fire.nfxclip --out $NEURALVFX_DATA/fire.nvfx --bits 8
build/nvfx_eval baselines --clip $NEURALVFX_DATA/fire.nfxclip           # flipbooks at every memory size
build/nvfx_eval model --clip $NEURALVFX_DATA/fire.nfxclip --model $NEURALVFX_DATA/fire.nvfx
build/nvfx_c_host $NEURALVFX_DATA/fire.nvfx 128 5                       # what an engine does, with timings
build/nvfx_train --rollout fire --out $NEURALVFX_DATA/fire_rollout.nvfx   # a rollout effect (about 40 min on 4 cores)
```

The whole evaluation of studies A to C (about two hours on 4 cores): `build/nvfx_experiment all`, then on an idle
machine `build/nvfx_experiment timing` and `build/nvfx_experiment report`. Study D: `build/nvfx_experiment d` (chaos,
training, evaluation; about three hours), then `d-timing` on an idle machine and `report`.

## Tools

| tool | what |
|---|---|
| `nvfx_sim` | simulate one clip; `--bench` reports the solver's cost per frame |
| `nvfx_ingest` | footage into a clip, after a licence check; adds a row to the licence register |
| `nvfx_train` | train a frame model from one or more clips (controls and variation codes come from the clips), or with `--rollout` a rollout effect from the simulation |
| `nvfx_eval` | score the flipbook ladder or a model against a reference clip |
| `nvfx_experiment` | the full study: `data`, `a`, `b`, `c`, `media`, `timing`, `report`; study D: `d-chaos`, `d-train`, `d-tune`, `d-finish`, `d-eval`, `d-timing`; study G (to come): `g-data`, `g-pilot`, `g-search`, `g-eval`, `g-timing` |
| `nvfx_c_host` | the engine loop in plain C, with timings; `--self-test` checks the error paths |
| `nvfx_viewer` | live viewer with sliders |
| `nvfx_fireball` | a scripted scene of composed effects (a fireball with smoke, fires and embers) to video, with a profile of every stage |
| `nvfx_dcm` | context mixing: `selftest` runs a small synthetic mixer search under both objectives (ROC-AUC and Laplace bits); `version FILE` prints the SHA-256 version of a serialised mixer |
| `nvfx_pack` | pack a `.nvfx` into a `.nvfz` and back (`--unpack`), bit-exact; `--report DIR` for sizes and ratios; `--study` for the measurement in results/compression |
| `neuralfx_arch_bench` | Phase 0 architecture microbenchmark |

## Layout

| path | what |
|---|---|
| `include/neuralfx/` | public headers: `nvfx.h` (C API), `clip`, `sim`, `model`, `train`, `rollout`, `rollout_train`, `metrics`, `flipbook`, `ingest`, `image_io`, `noise`, `cm` (lossless coder); `dcm/` (context mixing) |
| `src/core`, `src/sim`, `src/train`, `src/runtime`, `src/compose`, `src/dcm`, `src/common`, `src/proto` | libraries (see the table above); `src/proto` holds the Phase 0 prototypes |
| `tools/`, `examples/`, `viewer/`, `bench/` | executables |
| `tests/` | GoogleTest suites, the allocation test, the C host self-test, the viewer screenshot test |
| `docs/` | plan, report, composed effects, engines, data, viewer, figures |
| `results/` | small text results (CSVs and generated summaries); no images, clips or weights |

## Rules

- C++23 and the CPU only, for the runtime and the product. No GPU. No Python.
- No pretrained networks without the owner's approval.
- No raw video, frames, clips or model weights in git. Data lives under `NEURALVFX_DATA`.
- No AI model names or email addresses in the repository.
- Measure honestly: clips and control settings held out from training, 95% intervals in square brackets, ties
  reported as ties.
- Documentation in plain English with British spelling.
