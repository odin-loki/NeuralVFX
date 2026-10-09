# Neural visual effects on one CPU core: plan

Status: **current** (Phases 0–5 carried out 8–9 October 2026; results in [REPORT.md](REPORT.md)). Audience: new, dev, research.

Code: [README.md](../README.md). Phase 0 evidence: [results/phase0/](../results/phase0/).

Origin: the project started on 8 October 2026 inside [CameraDetector](https://github.com/odin-loki/CameraDetector), a
C++23 CPU-only machine-learning stack for roadside seatbelt detection, and moved here as its own repository the same
day. "Phase 12" below is CameraDetector's diffusion study. Its SIMD kernels and ISA dispatch, profiling tools, Dear
ImGui app and evaluation habits are the starting points this project borrows from.

## 1. Goal

The owner's request: "We are going to use this model for graphics effects in games. We are going to show the model
videos of effects and train the model to generate them. This should be like Nvidia's Neural effects but mine runs on
a single CPU."

In engineering terms: **neural flipbooks**. One small network per effect (fire, smoke, explosion, sparks, magic),
trained offline on the CPU from reference clips of that effect, and evaluated at run time on one CPU core at sprite
resolution (64–256 px). The network takes time, artist controls (intensity, colour, wind, speed) and a seed, so one
effect gives endless non-repeating variations. Compared with a texture flipbook it should use far less memory, play
at any frame rate without blending ghosts, and respond to controls that a flipbook cannot.

What this is not: a model that invents new effects from a text prompt, or a full-frame neural renderer. Neither fits
one CPU core in real time (§3).

## 2. What NVIDIA actually ships (checked 8 October 2026)

There is no NVIDIA product called "Neural Effects". The nearest things are:

- **RTX Neural Shaders** put "small neural networks into programmable shaders" and run them on the GPU's tensor cores.
  Built on them: **Neural Texture Compression** ("up to 8x" less texture memory than block compression, "at similar
  visual fidelity"), **Neural Materials** ("up to 5X faster material processing"; SDK not yet released) and the
  experimental **Neural Radiance Cache**. The post gives no network sizes or per-pixel costs.
  Source: [Get started with neural rendering using NVIDIA RTX Kit](https://developer.nvidia.com/blog/get-started-with-neural-rendering-using-nvidia-rtx-kit/).
- **DLSS 5**, announced at GTC on 16 March 2026 for release in autumn 2026: a real-time neural rendering model that
  takes each frame's colour and motion vectors and changes lighting and materials across the whole frame. The
  reported demo ran it on a second RTX 5090.
  Source: [NVIDIA newsroom](https://nvidianews.nvidia.com/news/nvidia-dlss-5-delivers-ai-powered-breakthrough-in-visual-fidelity-for-games)
  and press coverage of the keynote.

So "like NVIDIA's" means the **small network per asset** idea (neural shaders, NTC), not DLSS 5. Our target to
compare against is NTC's claim: the same quality as block compression at about 1/8 of the memory, here for animated
effects rather than static textures, and on a CPU core instead of tensor cores.

## 3. Feasibility

### 3.1 The budget

Measured in Phase 0 (§8): on one core of this machine (Intel Xeon, 2.8 GHz, cloud VM), the hand-vectorised kernels
sustain **about 25–35 GMAC/s** on AVX2 for dense layers and 3×3 convolutions. One 128×128 frame is 16,384 pixels, so:

| Cost per pixel | MACs per 128×128 frame | ms per frame at 30 GMAC/s |
|---:|---:|---:|
| 250 | 4.1 M | 0.14 |
| 1,000 | 16 M | 0.55 |
| 2,000 | 33 M | 1.1 |
| 5,000 | 82 M | 2.7 |

The starting estimate ("a few thousand multiply-adds per pixel at 128×128 is about 1 ms per effect on one AVX2 core")
holds: **about 2,000 MAC per pixel per millisecond at 128×128**. At 256×256 everything costs four times as much.

A 60 fps game has 16.7 ms per frame for everything. Ten distinct effects at 1 ms each would take most of a core, so
the runtime must amortise (§5.4): an effect animates at 20–30 Hz, not 60; instances with the same controls share one
evaluation; distant effects use 64 px. With those, 1 ms per evaluation supports a scene with many effects.

### 3.2 Realistic on one core in real time

- Per-effect networks of 200–2,000 MAC per pixel at 64–128 px, evaluated at the effect's own frame rate.
- Controls and a seed as inputs, with continuous time.
- Diffusion or other large generative models **offline only**: to author or augment training clips, never per frame.

### 3.3 Not realistic

- Prompt-to-effect generation at run time.
- Full-frame neural rendering in the style of DLSS 5 on a CPU (a 1080p frame is 127 times as many pixels as a
  128×128 sprite).
- Large pretrained generative models: they are excluded by the owner's rule anyway unless approved.

### 3.4 The Phase 12 lesson

CameraDetector's Phase 12 ([DIFFUSION_MIXER.md](https://github.com/odin-loki/CameraDetector/blob/main/cabinlab/results/product/DIFFUSION_MIXER.md) §1) trained a diffusion model from
scratch on the CPU without trouble, but its features did not beat simpler methods. The same risk applies here in a
sharper form: **a neural flipbook must beat a plain flipbook at the same memory, or offer something a flipbook cannot
(controls, variation, no frame-blend ghosting) at a quality an artist accepts.** Phase 3 has an explicit stop rule
(§7) and the baselines come first (Phase 2).

## 4. Architecture options and cost

Five families were built and timed in Phase 0 with random weights (cost does not depend on the values). All of them
write RGBA8 sprites. "Memory" is learnable parameters at 2 bytes each (fp16). The reference flipbook is 64 frames of
128×128 RGBA8.

| Family | How it works | Memory | MAC/px | ms, 128² AVX2 | Notes |
|---|---|---:|---:|---:|---|
| A. Coordinate MLP, naive | Fourier(x, y, t) + controls → MLP | 2–14 KB | 1,000–7,000 | 1.6–4.3 | per-pixel encoding dominates small nets |
| B. Coordinate MLP, separable first layer | first layer split into column, row and per-frame tables | 2–22 KB | 330–8,500 | 0.26–4.0 | same function as A, 2–6x cheaper |
| C. Feature grid + tiny MLP | learned (x, y, t) feature volume, trilinear, then MLP | 128 KB–1 MB | 200–1,700 | 0.33–1.6 | most capacity in the grid; cheapest per quality (expected) |
| D. Multiresolution hash grid + MLP | Instant-NGP-style hashed levels | 129 KB–1 MB | 580–6,900 | 6.7–18 | random gathers, scalar: 10x slower per MAC than C |
| E. Convolutional decoder | (t-sliced, control-modulated) latent 16², three up-conv stages | 36–290 KB | 500–2,300 | 0.37–1.6 | spatial detail from convolutions; seed noise is natural here |

Medians of three runs of 300 frames on one pinned core ([avx2_128_r*.csv](../results/phase0/)); full sweeps
at 64, 128 and 256 px for the baseline ISA, AVX2 and AVX-512 are in
[sweep_*.md](../results/phase0/). Each configuration is listed there separately.

Memory against the flipbook (128×128, 64 frames):

| Storage | Size |
|---|---:|
| RGBA8, uncompressed | 4,096 KB |
| BC7 / BC3 / ASTC 4×4 (8 bits per pixel) | 1,024 KB |
| BC1 (4 bpp, 1-bit alpha) | 512 KB |
| ASTC 8×8 (2 bpp) | 256 KB |
| 1/8 of BC7 (the NTC-style target) | 128 KB |

Reading the tables together:

- **B (separable MLP)** is tiny in memory but holds little detail: 2,500 parameters cannot store 64 frames of
  turbulent fire. It is the "procedural" end: smooth shapes, strong controls, poor fine detail. Worth keeping as the
  smallest baseline and as the head of C and E.
- **C (grid + MLP)** and **E (conv decoder)** reach 128–270 KB at 0.4–0.6 ms per 128² frame: in the right range for
  the NTC-style target. They are the primary candidates.
- **D (hash grid)** is the standard GPU choice, but on a CPU the hashed gathers make it 6–18 ms per 128² frame here. It
  stays in training (it learns fast) only if its result can be **baked into a dense grid** (C) for run time.
- AVX-512 gave no speed-up over AVX2 on this machine (ties or slower; [sweep_avx512.md](../results/phase0/sweep_avx512.md)).
  AVX2 is the target ISA; AVX-512 stays as a runtime option.

Optimisation headroom not yet used (Phase 4): weights pre-broadcast or packed for the FMA loop (the dense kernel is at
about 40% of the AVX2 peak of about 90 GMAC/s); int8 weights with VNNI (`avx512_vnni` is present here; AVX-VNNI on
newer desktop parts); vectorised grid sampling (the x-expansion in C is scalar); fusing the RGBA8 conversion.

## 5. Proposed design

### 5.1 The model

Start with C and E side by side, sharing one head:

- **Inputs.** Time t in [0, 1) (or looping phase), controls (intensity, colour or hue, wind direction and strength,
  speed) and a seed. Controls modulate features per channel (FiLM: scale and shift), which costs nothing per pixel
  because they are constant over a frame.
- **Variation from a seed.** Options to compare in Phase 3: (a) a learned latent per training clip with a seed mapped
  into that latent space (an auto-decoder; new seeds interpolate between clips); (b) procedural noise channels
  (seeded value or simplex noise) fed to the decoder alongside the learned features, so the network learns to shape
  noise into flames; (c) domain warping of the learned features by seeded noise. (b) and (c) give endless variation at
  run time for almost no cost.
- **Output.** Premultiplied RGBA (or RGB + alpha, matching the engine's blend mode: additive fire and alpha-blended
  smoke need different handling). Optional extra channels later (emissive, normal for lit smoke).
- **Looping.** Time enters as sin/cos of the loop phase when the effect loops, so frame 0 and the last frame match.

### 5.2 Training

- On the CPU, from scratch, no pretrained networks. LibTorch (CPU) gives a C++ trainer with no Python; CameraDetector
  already downloads and checksums it at configure time (`cmake/LibTorch.cmake`), and the same recipe can be copied. If the owner allows Python for offline tools only, PyTorch on the CPU is the
  same library with faster iteration; nothing in Python would ship.
- Losses: L1 on premultiplied RGBA, a gradient (edge) term, and a temporal term on frame differences to fight
  flicker. No perceptual losses that need pretrained networks (LPIPS, VGG) unless the owner approves.
- The trained weights are exported to a small flat file (fp16 or int8) that the runtime library reads. Weights never
  enter git.

### 5.3 Data

- **Procedural ground truth (Phase 1a).** A C++ stable-fluids solver (semi-Lagrangian advection, pressure
  projection, buoyancy from temperature, vorticity confinement) with density and temperature, rendered to RGBA with a
  blackbody-like colour ramp for fire and a lit density for smoke. Its parameters (source size, buoyancy, wind,
  turbulence, cooling, colour) are the same controls the model exposes, so every clip is labelled and unlimited, and
  test clips can use seeds and parameter settings never seen in training.
- **The owner's videos (Phase 1b).** Each video's licence and source is recorded before use. Raw video, extracted
  frames and weights stay outside git (under a data root named by the environment variable `NEURALVFX_DATA`). Effects filmed or rendered
  on black give alpha from luminance for additive effects; others need a matte.

### 5.4 Runtime

- A C++23 library with a C API: `nfx_effect_load`, `nfx_instance_create(controls, seed)`,
  `nfx_render(instance, t, rgba_out)`, `nfx_effect_free`. Single-threaded per call, no allocation per frame,
  deterministic. The engine owns threading.
- An evaluation cache: an effect instance renders at its own frame rate (20–30 Hz) into a two-frame ring, and the
  engine blends or reprojects between them. Instances with equal controls share frames. Level of detail picks 64, 128
  or 256 px.
- A Dear ImGui viewer, adapted from CameraDetector's `cabinlab/src/gui/`, that plays effects live with sliders for every control and a
  readout of milliseconds per frame.
- A second mode for engines that prefer textures: **bake at load time.** The network expands into an ordinary
  flipbook (with fresh seeds) when the level loads. This saves download and disk size rather than run-time memory,
  and costs nothing per frame.

## 6. Evaluation

The habits of [ROADSIDE_STUDY.md](https://github.com/odin-loki/CameraDetector/blob/main/cabinlab/results/product/ROADSIDE_STUDY.md) §3 carry over:

- **Held-out data.** Test clips use simulation seeds and parameter settings that training never saw: interior points
  of the parameter grid (interpolation) and a few outside it (extrapolation), reported separately. For the owner's
  videos, whole clips are held out, never frames of a training clip.
- **Metrics.** PSNR and SSIM on premultiplied RGBA per frame; a temporal-flicker metric (the error of frame-to-frame
  differences, so a model that is right on average but jitters is penalised); memory in bytes; milliseconds per frame
  (median and 90th percentile on one pinned core). Side-by-side videos for every comparison, written outside git.
- **Matched comparisons.** Every neural model is compared with a flipbook of the same memory: fewer frames, lower
  resolution, or block compression (BC7/ASTC-size equivalents, using an open-source encoder fetched at build time).
- **Intervals.** Paired bootstrap over held-out clips, 95% intervals in square brackets. When an interval covers zero,
  the result is reported as a tie.
- **Variation.** For the seed: diversity between seeds against realism (distance to the nearest training clip, so
  that copying is visible), plus artist review.

## 7. Phases, deliverables and stop rules

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0. Feasibility | this plan; `neuralfx_arch_bench`; [phase0 results](../results/phase0/) | done: ms per frame known for five families |
| 1a. Procedural data | `neuralfx_sim`: stable-fluids smoke and fire to RGBA clips with labelled parameters; tests | clips look like fire and smoke; cost per frame measured |
| 1b. Owner data | ingest tool for videos; licence register; frames outside git | each clip's licence recorded |
| 2. Baselines | flipbooks at matched memory (raw, fewer frames, lower resolution, BC7/ASTC-size); the simulation's own cost per frame | baseline table at 64/128/256 px |
| 3. Models | trainer; C and E (and B as the smallest) trained per effect; evaluation on held-out clips and parameters | **continue** only if a model beats the flipbook of equal memory on held-out clips (PSNR or SSIM interval above zero) or ties it at ≤ 1/4 of the memory, within 1 ms per 128² frame on AVX2; otherwise report and stop or change the family |
| 4. Runtime | C API library, zero allocation per frame, SIMD, viewer with sliders | meets the ms budget with the cache; tests for determinism and ISA parity |
| 5. Report | quality against memory and ms, against flipbooks and against NVIDIA's published claims; honest limits | owner review |

Status on 9 October 2026: every phase delivered (the code map is in the README). Phase 1b is built and tested but has
no owner footage yet; Phase 3's continuation rule is met ([REPORT.md](REPORT.md) §1); Phase 4 meets the 0.5 ms budget
with the small grid model only. Two names in the table changed in the build: the simulator tool is `nvfx_sim`, and
the flipbook baseline uses a BC3-layout encoder (BC1 + BC4) rather than BC7 or ASTC sizes.

Phases 0, 1a, 2 and the runtime skeleton do not depend on the owner's answers below; 1b and the choice of effects,
sizes and budgets do.

## 8. Phase 0 results

Machine: Intel Xeon at 2.8 GHz, 4 vCPU cloud VM, AVX2, FMA and AVX-512 (with VNNI); GCC 14.2, `-O3`, multiply-add
contraction on. One pinned core, flush-to-zero on. Tool: `neuralfx_arch_bench` ([README.md](../README.md)).

AVX2, 128×128, median ms per frame of 300 frames, three runs:

| Model | Params | KB fp16 | MAC/px | run 1 | run 2 | run 3 |
|---|---:|---:|---:|---:|---:|---:|
| mlp_naive H16 L2 F6 | 1,012 | 2.0 | 976 | 1.638 | 1.613 | 1.605 |
| mlp_naive H32 L2 F6 | 2,532 | 4.9 | 2,464 | 2.347 | 2.307 | 2.301 |
| mlp_naive H64 L2 F6 | 7,108 | 13.9 | 6,976 | 4.292 | 4.335 | 4.325 |
| mlp_sep H16 L2 F6 | 1,012 | 2.0 | 328 | 0.258 | 0.257 | 0.266 |
| mlp_sep H32 L2 F6 | 2,532 | 4.9 | 1,168 | 0.687 | 0.675 | 0.725 |
| mlp_sep H32 L3 F6 | 3,588 | 7.0 | 2,192 | 1.245 | 1.233 | 1.214 |
| mlp_sep H64 L2 F6 | 7,108 | 13.9 | 4,384 | 2.180 | 2.175 | 2.174 |
| mlp_sep H64 L3 F6 | 11,268 | 22.0 | 8,480 | 4.016 | 4.172 | 3.991 |
| grid_mlp 32x32x16 C8 H16 L1 | 131,364 | 256.6 | 208 | 0.333 | 0.351 | 0.335 |
| grid_mlp 32x32x16 C8 H32 L1 | 131,652 | 257.1 | 400 | 0.513 | 0.579 | 0.516 |
| grid_mlp 32x32x32 C16 H32 L2 | 526,180 | 1,027.7 | 1,696 | 1.633 | 1.629 | 1.584 |
| grid_mlp 64x64x16 C8 H32 L2 | 525,924 | 1,027.2 | 1,424 | 1.296 | 1.334 | 1.280 |
| hash_mlp L8 F2 T2^12 H16 L1 | 65,956 | 128.8 | 576 | 6.882 | 6.808 | 6.723 |
| hash_mlp L8 F2 T2^14 H32 L2 | 264,036 | 515.7 | 1,920 | 8.083 | 8.118 | 8.156 |
| hash_mlp L16 F2 T2^14 H64 L2 | 531,140 | 1,037.4 | 6,912 | 18.323 | 18.152 | 18.877 |
| conv_dec 16^2x16 16-8-8-4 | 67,764 | 132.4 | 504 | 0.388 | 0.366 | 0.364 |
| conv_dec 16^2x16 32-16-8-4 | 137,532 | 268.6 | 864 | 0.640 | 0.625 | 0.612 |
| conv_dec 16^2x16 32-32-16-4 | 145,908 | 285.0 | 2,304 | 1.563 | 1.555 | 1.529 |

Playing back a 64-frame RGBA8 flipbook with blending costs 0.007 ms per 128² frame: the neural models are about 35–2,700
times more expensive per frame than a flipbook. Their case rests on memory, controls and variation, not speed.

Findings:

1. **The budget estimate holds**: about 2,000 MAC per pixel per millisecond at 128² on AVX2.
2. **Separating the first layer** of a coordinate MLP by input group makes it 2–6x cheaper with identical output (a
   unit test checks the equality).
3. **Grid + MLP and the conv decoder** fit 0.3–0.6 ms per 128² frame at 130–270 KB fp16.
4. **Hash grids are a poor fit for CPU inference** here (6.7–18 ms per 128² frame): use them, if at all, for training
   and bake to a dense grid.
5. **AVX-512 is a tie with AVX2** on this machine; AVX2 is the target.
6. The first kernel version reached only about 1 GMAC/s because GCC lowered 16-float vectors badly on AVX2 (spills
   to the stack). Native-width vectors fixed it (10–30x). Kept here as a warning for Phase 4.

Limits of these numbers: one cloud VM, whose neighbours add noise (one early run on another core gave 1.23 ms for a
configuration that measures 0.69 ms on a quiet core); random weights (real weights do not change cost but sparsity
or int8 could); no engine around the code (cache effects of a real game frame are not included).

## 9. Risks

| Risk | Mitigation |
|---|---|
| Neural model does not beat a flipbook of equal memory (the Phase 12 pattern) | baselines first; stop rule in Phase 3; value can still come from controls and variation, judged separately and honestly |
| Fine turbulent detail is lost (blurry flames) | noise channels and conv decoders; temporal loss; compare at equal memory, not equal parameters |
| Flicker between frames | temporal loss and the flicker metric; continuous time input |
| CPU budget with many effects | evaluation cache at the effect's frame rate, sharing between instances, level of detail, bake-at-load mode |
| Simulated data does not resemble the owner's effects | train on the owner's videos once supplied; sim used for controlled tests and pretraining |
| Video licences | licence register before any use; nothing raw in git |
| Engine integration cost | C API, no dependencies, engine-owned threading; target engine chosen by the owner |

## 10. Decisions

The owner was asked five questions and answered (8 October 2026): "You pick all. Do all. Be comprehensive." The
answers below were chosen on that instruction; each can be revisited.

1. **Run-time generation, offline authoring, or both?** Both. The default is run-time evaluation through the C API
   (`nvfx_render`), and `nvfx_bake` expands an effect into an ordinary flipbook at load time for engines or platforms
   that prefer textures (it saves download size rather than memory).
2. **Effects, resolution and budget.** Fire and smoke (looping) and explosions (one-shot) first, because the simulator
   makes unlimited labelled data for them; sparks and magic follow once owner footage exists. Native size 128 x 128,
   with 64 and 256 as levels of detail. Budget: at most 0.5 ms per 128 x 128 evaluation on one AVX2 core, effects
   evaluated at 30 Hz and shared between instances with equal controls, so 16 distinct live effects cost about
   8 ms of one core per 33 ms. Memory target: at most 128 KB per effect, one eighth of a 1 MB BC7/BC3 flipbook (the
   ratio NVIDIA claims for Neural Texture Compression).
3. **Training videos and licences.** None were supplied, so training uses the project's own simulation (no
   licence question). Owner footage enters through `nvfx_ingest`, which accepts own footage, CC0, CC-BY (author
   recorded), CC-BY-SA (flagged) and commercial footage whose agreement allows ML training, refuses NonCommercial,
   NoDerivatives and unknown licences, and records every clip in a licence register ([DATA.md](DATA.md)).
4. **Engine.** Engine-neutral: a C API with no dependencies (`include/neuralfx/nvfx.h`), a static and a shared
   library, a plain-C example host, and integration notes for Unreal, Unity and Godot ([ENGINES.md](ENGINES.md)).
   No engine plugin is built here (none can be tested in this environment).
5. **Python.** No. Training is C++23 with hand-written gradients (no LibTorch, no Python), checked against finite
   differences; the repository keeps its "no Python" rule.

Language: C++23 throughout (owner's instruction, 8 October 2026): `std::print`, `std::expected`, `std::span`, ranges
and views, multidimensional `operator[]`, `std::float16_t`, `std::byteswap`, `std::ispanstream`, `std::jthread`. The
engine-facing header is C so that every engine can call it.

## 11. Study D: start points and learned dynamics (9 October 2026)

After the first report the owner pointed the work in a new direction: "we can store the saved points of the
simulation from training and shard the inference from the start points. We don't have to store all the data, just the
start points. Essentially we are modelling chaotic systems, so the right start point matters but the data in between
doesn't. A lossy algorithm with some noise that benefits the simulation would be good ... Use those characteristics as
strengths. Don't fight the algorithm." Studies A to C had fought it: they asked a network to reproduce every frame of a
chaotic run, and the squared-error optimum for turbulence it cannot predict is a blur.

Decisions:

1. **Measure the chaos first** (`nvfx_experiment d-chaos`). Runs of the simulation from the same stored state with a
   tiny nudge stay close for seconds; the same state with another noise seed is unrelated within a second; another
   state with the same seed drifts towards the first. The seed's noise, not the state, decides most of what is seen.
2. **A second kind of effect, `rollout`** (`include/neuralfx/rollout.hpp`): start points (coarse simulation states)
   plus a learned stepper that moves a 32 x 32 state one frame at a time. What physics does cheaply and exactly is
   built in (advection, pressure projection); the network supplies what is local and learnable (forces, sources,
   decay, the sub-grid closure), conditioned on the controls and **driven by the same procedural noise as the
   simulation's forcing**, so the noise is an input, not something to memorise.
3. **Detail from the dynamics, not from the network**: full-resolution heat and soot are carried by the learned flow
   (stretching makes filaments), their block averages locked to the coarse state (scaling existing structure where
   it can, so peaks stay peaks, and adding the rest as new material), new material broken up by the flicker noise,
   plus a small sub-grid swirl. Its few constants are calibrated (decision 5). A per-pixel MLP renders the fields,
   gated so empty pixels are exactly transparent.
4. **Training for the long run**: backpropagation through up to 16 frames with noise on the inputs; then fine-tuning
   on windows that start from the stepper's own rollout, with a loss on where the heat and soot are (row and column
   profiles), which still means something after the chaos horizon; then a third stage that also matches how much the
   state changes from frame to frame (the first two left the heat changing about 60% as much as the simulation's: a
   smooth average in time). Data: 160 runs of 8 s per looping effect, 240 runs of 3 s of explosions, random controls
   and seeds (55 minutes of simulation, nearly six times the 9.7 minutes of clips A to C used; 13 times study B's data
   per effect).
5. **Calibrated as deployed**: the detail layer's constants (contrast, swirl, how far the lock may scale existing
   structure) are chosen by the statistics of endless runs from the start points with new seeds against real runs at
   training controls (detail spectrum, motion, light, cover), not by tracking a true run for three seconds, which
   favoured settings that faded later.
6. **Shards** (the owner's proposal): playback is a chain of 6 s rollouts, each from a start point chosen by the seed,
   the shard and the nearest controls, with a seed of its own, rolled ahead and crossfaded in over half a second. One
   continuous rollout drifted (fire froze, smoke filled the frame) within a minute; shards keep every minute like the
   first, make every frame a function of time, and make any time reachable in bounded cost.
7. **Judged as a chaotic system**: tracking from true start points is scored frame by frame against the true run
   only as far as the chaos allows (against the simulation's own nudged-start curve); beyond that, held-out settings
   and new seeds are scored by frame statistics (detail spectrum, motion, coverage, light) against real runs, with a
   second real seed as the floor, against study B's control model, the nearest flipbook and the simulation on the same
   coarse grid with the same detail layer (the traditional cheap alternative).
