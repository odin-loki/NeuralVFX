# NeuralVFX

Neural visual effects for games that run on **one CPU core**. Each effect (fire, smoke, an explosion) is a small
network trained offline from clips of that effect. At run time it draws sprites (64-256 px) from time, artist
controls (intensity, wind, turbulence) and a seed, with no GPU. The idea follows NVIDIA's small-network-per-asset
work (RTX Neural Shaders, Neural Texture Compression), but runs on a CPU instead of tensor cores.

C++23, no third-party runtime dependencies, a C API for engines, no Python.

- **Results:** [docs/REPORT.md](docs/REPORT.md) covers quality against memory and milliseconds, flipbooks at
  matched memory, NVIDIA's claims and the honest limits. The generated tables with intervals are in
  [results/experiments/SUMMARY.md](results/experiments/SUMMARY.md).
- **Plan and decisions:** [docs/PLAN.md](docs/PLAN.md).
- **Engine integration:** [docs/ENGINES.md](docs/ENGINES.md) (C API, threading, Unreal, Unity, Godot).
- **Data and licences:** [docs/DATA.md](docs/DATA.md).
- **Viewer:** [docs/VIEWER.md](docs/VIEWER.md).

![The viewer: a neural fire next to its reference clip and a flipbook of the same memory](docs/figures/viewer.png)

## How it works

| piece | where | what |
|---|---|---|
| simulation | `src/sim`, `nvfx_sim` | a 2D stable-fluids solver (temperature, soot, vorticity, curl noise, combustion expansion) that renders fire, smoke and explosions to premultiplied RGBA clips with labelled controls: unlimited training and test data |
| ingest | `src/core/ingest.cpp`, `nvfx_ingest` | owner footage into clips via ffmpeg, with a licence check and a licence register |
| models | `src/core/model.cpp` | a **grid** family (learned feature volumes over x, y and t, blended by the controls, sampled bilinearly, then a small MLP with FiLM conditioning) and a **conv** family (a learned latent, three upsampling convolutions); `.nvfx` files store features at 8 or 16 bits |
| trainer | `src/train`, `nvfx_train` | hand-written gradients (checked against finite differences), Adam, multithreaded; per-clip variation codes |
| runtime | `src/runtime`, `include/neuralfx/nvfx.h` | the shipping library: C API, per-ISA SIMD (SSE2, AVX2, AVX-512), no allocation per frame, seeds and endless drift, exact hue, brightness and speed, bake to flipbook |
| baselines | `src/core/flipbook.cpp` | flipbooks at matched memory: frame count, resolution, raw or BC1/BC4 (BC3-layout) compression, motion vectors |
| metrics | `src/core/metrics.cpp` | PSNR (full and active-region), SSIM, temporal PSNR, flicker, spectrum and motion statistics, paired bootstrap |
| evaluation | `nvfx_experiment` | the whole study end to end: compression, controls, variation, timing, figures, report |
| viewer | `viewer/`, `nvfx_viewer` | Dear ImGui: sliders for every control, side by side with the reference, a flipbook and the live simulation |

## Build and test

Ubuntu 24.04: `g++-14`, CMake 3.25+, Ninja, `libgtest-dev`, `zlib1g-dev` (and `ffmpeg` for videos and the video test).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14
cmake --build build
ctest --test-dir build                     # 51 tests: sim, metrics, codecs, gradients, runtime parity, allocation, C API
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
```

The whole evaluation (about two hours on 4 cores): `build/nvfx_experiment all`, then on an idle machine
`build/nvfx_experiment timing` and `build/nvfx_experiment report`.

## Tools

| tool | what |
|---|---|
| `nvfx_sim` | simulate one clip; `--bench` reports the solver's cost per frame |
| `nvfx_ingest` | footage into a clip, after a licence check; adds a row to the licence register |
| `nvfx_train` | train an effect from one or more clips (controls and variation codes come from the clips) |
| `nvfx_eval` | score the flipbook ladder or a model against a reference clip |
| `nvfx_experiment` | the full study: `data`, `a`, `b`, `c`, `media`, `timing`, `report` |
| `nvfx_c_host` | the engine loop in plain C, with timings; `--self-test` checks the error paths |
| `nvfx_viewer` | live viewer with sliders |
| `neuralfx_arch_bench` | Phase 0 architecture microbenchmark |

## Layout

| path | what |
|---|---|
| `include/neuralfx/` | public headers: `nvfx.h` (C API), `clip`, `sim`, `model`, `train`, `metrics`, `flipbook`, `ingest`, `image_io`, `noise` |
| `src/core`, `src/sim`, `src/train`, `src/runtime`, `src/common`, `src/proto` | libraries (see the table above); `src/proto` holds the Phase 0 prototypes |
| `tools/`, `examples/`, `viewer/`, `bench/` | executables |
| `tests/` | GoogleTest suites, the allocation test, the C host self-test, the viewer screenshot test |
| `docs/` | plan, report, engines, data, viewer, figures |
| `results/` | small text results (CSVs and generated summaries); no images, clips or weights |

## Rules

- C++23 and the CPU only, for the runtime and the product. No GPU. No Python.
- No pretrained networks without the owner's approval.
- No raw video, frames, clips or model weights in git. Data lives under `NEURALVFX_DATA`.
- No AI model names or email addresses in the repository.
- Measure honestly: clips and control settings held out from training, 95% intervals in square brackets, ties
  reported as ties.
- Documentation in plain English with British spelling.
