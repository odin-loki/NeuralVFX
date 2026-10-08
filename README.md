# NeuralVFX

CPU-only neural visual effects for games: small per-effect networks ("neural flipbooks") trained offline from clips
of an effect, then evaluated on one CPU core at sprite resolution (64–256 px) from time, artist controls (intensity,
colour, wind, speed) and a seed. The idea follows NVIDIA's small-network-per-asset work (RTX Neural Shaders, Neural
Texture Compression), but on a CPU instead of tensor cores.

The plan, targets, evaluation, risks and Phase 0 results are in [docs/PLAN.md](docs/PLAN.md).

**Status:** Phase 0 (feasibility) is done. Only the architecture prototypes and their benchmark exist; nothing is trained
yet. On one AVX2 core, the leading candidates render a 128×128 sprite in 0.3–0.6 ms with 130–270 KB of weights
([docs/PLAN.md §8](docs/PLAN.md#8-phase-0-results)).

## Build and test

Ubuntu 24.04 with `g++-14`, CMake 3.25 or newer, Ninja and `libgtest-dev`.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14
cmake --build build
ctest --test-dir build        # label: neuralfx
```

Options: `NEURALFX_BUILD_TESTS`, `NEURALFX_BUILD_BENCH`, `NEURALFX_WERROR` (CI). No runtime dependencies.

## Layout

| Path | What |
|---|---|
| `docs/PLAN.md` | the plan: goal, what NVIDIA ships, feasibility, architectures and cost, design, evaluation, phases, risks |
| `include/neuralfx/proto.hpp` | Phase 0 prototype interface: `Spec`, `Controls`, `Model`, `make_model`, ISA selection |
| `src/proto/kernels_impl.hpp` | dense and 3×3 conv kernels and the five candidate families, compiled once per ISA |
| `src/proto/kernels_{base,avx2,avx512}.cpp` | the ISA builds (SSE2, x86-64-v3, x86-64-v4), switched on by `#pragma GCC target` after the standard headers |
| `src/proto/dispatch.cpp` | CPU detection and the factory |
| `bench/arch_bench.cpp` | `neuralfx_arch_bench`: ms per frame for each family |
| `tests/` | ISA parity, determinism, separable-layer equality, cost bookkeeping |
| `results/phase0/` | benchmark outputs (tables and CSVs; no images or weights) |

## Phase 0 benchmark

```sh
build/neuralfx_arch_bench --isa avx2 --sizes 64,128,256 --frames 200 --core 3 --csv out.csv
```

Pin to a quiet core: on a shared VM, a busy neighbour can double some timings.

## Rules

- C++23 and the CPU only, for the runtime and the product. No GPU.
- No pretrained networks without the owner's approval.
- No raw video, frames, images or model weights in git. Data lives under `NEURALVFX_DATA`, outside the repository.
- No AI model names or email addresses in the repository.
- Measure honestly: clips and parameter settings held out from training, 95% intervals in square brackets, ties
  reported as ties.
- Documentation in plain English with British spelling.
