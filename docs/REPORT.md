# Neural visual effects on one CPU core: report

Status: **canonical** for results (9 October 2026). Audience: new, research, dev. Generated tables with every number
and interval: [results/experiments/SUMMARY.md](../results/experiments/SUMMARY.md) (rerun `nvfx_experiment report`).
Plan and decisions: [PLAN.md](PLAN.md).

## 1. Bottom line

- **Compression works, and in the range NVIDIA claims.** On 12 effect clips, a network per clip beats flipbooks
  of the same memory by **+2.8 to +7.0 dB** of active-region PSNR at every budget from 128 to 512 KB (every 95%
  interval above zero; SSIM better or tied). For equal quality the networks need **3.6 to 7.4 times less memory**:
  the 132 KB grid model equals a 772 KB flipbook (5.9x). NVIDIA claims "up to 8x" for Neural Texture Compression
  against block compression. Our flipbooks use our own BC3-layout encoder, so against a production BC7 encoder the
  ratios would be lower (§7).
- **It depends on the effect.** At 132 KB the network beats the full 1 MB flipbook on fire, not on smoke or
  explosions (§3).
- **Controls work, at a fidelity cost.** One 1 MB model per effect plays control settings it never saw better than a
  45 MB library of flipbooks: **+2.49 [+2.22, +2.77] dB** against the nearest setting, **+0.99 [+0.65, +1.36] dB**
  against blending the two nearest, with higher SSIM and matching motion. By eye its held-out flames are softer and
  dimmer than the simulation; every method scores low there because turbulence is chaotic (§4).
- **Endless variation works, as morphs of what it saw.** Looping effects drift between learned variation codes and
  never repeat. Generated variations are about as diverse as real seeds but softer, and they are blends of the
  training seeds rather than new turbulence (§5).
- **Cost: 0.38 to 1.1 ms per 128 x 128 sprite on one AVX2 core** (0.10 to 0.27 ms at 64 x 64). A scene of 24 effects
  (8 near, 16 far, each at 30 Hz) costs 2.6 to 7.2 ms of one core per 60 fps frame. The networks are 8 to 30 times
  cheaper than running the simulation and 50 to 150 times more expensive than playing a flipbook. Only the small
  grid model (73 KB, 0.38 ms) meets the 0.5 ms target set in the plan (§6).
- **The plan's continuation rule is met** (PLAN.md §7: beat the flipbook of equal memory on held-out data, interval above
  zero, within 1 ms per 128² frame): on held-out settings (study B, against a 45 times larger flipbook library) and on
  held-out frames against a BC3 flipbook with four times the memory (study A; motion-vector flipbooks still win there). grid_s, conv_s and conv_m are within 1 ms;
  grid_m is 6% over.
- **Not done:** owner footage (none supplied), a BC7 baseline, int8 kernels, an engine plugin (§8, §9).

## 2. How it was measured

**Data.** Every clip comes from the project's fluid simulation (`src/sim`): 128 x 128 pixels, 64 frames at 30 fps,
premultiplied RGBA. Fire and smoke loop (warmed up to a steady state, then crossfaded into a seamless loop);
explosions play once. Three learned controls in [0, 1]: intensity, wind, turbulence; a seed sets the source noise
and the turbulence field. 273 clips in all, regenerated deterministically by `nvfx_experiment data`.

**Three questions, three studies.**

| study | question | training | test (never seen in training) |
|---|---|---|---|
| A. compression | does a network store an effect in less memory than a flipbook at the same quality? | one model per clip (12 clips: 4 per effect, different controls and seeds) | the clip itself, as for any compression; plus a held-out variant: only even frames available, odd frames scored |
| B. controls | can one model play settings it never saw? | one model per effect on a 3 x 5 x 3 grid of settings (45 clips, seed 1) | 10 settings per effect, off the grid in every control |
| C. variation | can it produce new, realistic variations? | one model per effect on 24 seeds (fixed controls), each with a learned code | 8 held-out seeds, compared as distributions |

**Models.** Grid family (feature volumes over x, y and t; bilinear sampling; an MLP with FiLM) and conv family (a
latent volume and three upsampling convolutions), described in [PLAN.md](PLAN.md) §5.1 and `include/neuralfx/model.hpp`.
Study A trains a ladder of sizes; B and C use grid models with 8 to 24 blended feature volumes. Training: Adam with
cosine decay, 2,000 steps (A) or 12,000 steps (B, C) of 8 frames, 4 threads, from random initialisation. Every model
is scored **through the shipping runtime** at its stored precision (8-bit or fp16 features).

**Baselines.**
- A: flipbooks of the same clip at matched memory, 30 configurations: 4 to 64 kept frames, 32 to 128 px, raw RGBA8
  (32 bits per pixel) or BC3-layout compression (8 bits per pixel; our own BC1 + BC4 encoder, a lower bound on what a
  production BC7 encoder reaches), with or without motion vectors (block-matched flow at a quarter resolution, 8 bits
  per component). Playback blends neighbouring kept frames with bilinear filtering, as a game does.
- B: a library of 45 BC3 flipbooks (one per training setting, 45 MB per effect) played at the nearest setting, or
  blended from the two nearest; and the oracle "closest training clip".
- C: the natural spread between real seeds (a generator should match it, not beat it) and the training clips
  themselves (a flipbook library replays them).
- The simulation itself, at its cost per frame.

**Metrics.** PSNR over all pixels and over *active* pixels (visible in either clip: empty background cannot
flatter a method); SSIM (Gaussian 11, sigma 1.5, four channels); temporal PSNR of frame-to-frame differences; a
flicker ratio (second temporal difference energy against the reference); a sharpness statistic (mean absolute
difference of the radially averaged log luminance spectrum: blur raises it); a motion ratio (frame-to-frame change
against the reference). Differences between methods are paired over clips with 95% bootstrap intervals (10,000
resamples) in square brackets; an interval covering zero is a tie.

**Machine.** One cloud VM: Intel Xeon at 2.8 GHz, 4 vCPUs, AVX2, FMA and AVX-512 (VNNI); 32 KB L1 and 1 MB L2 per
core; GCC 14.2 at `-O3`. Timings are medians over 200 frames on one pinned core with nothing else running
(`nvfx_experiment timing`); the VM adds noise, so 90th percentiles are reported too.

## 3. A. Compression: one effect clip, less memory than a flipbook

Means over 12 clips (4 fire, 4 smoke, 4 explosions). Neural models at 8-bit feature storage (fp16 storage scores the
same within 0.05 dB at twice the memory); ms per 128 x 128 frame on one AVX2 core (quiet machine).

| method | KB | PSNR | active PSNR | SSIM | temporal PSNR | spectrum | motion | ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural conv_s (latent 16², 16-8-8) | 69 | 36.02 | 29.43 | 0.9668 | 36.33 | 0.266 | 0.93 | 0.57 |
| neural grid_s (G24 C8, H16 L1) | 73 | 34.32 | 28.07 | 0.9627 | 34.97 | 0.171 | 0.95 | 0.38 |
| neural grid_m (G32 C8, H32 L2) | 132 | 39.10 | 32.62 | 0.9873 | 39.83 | 0.096 | 0.98 | 1.06 |
| neural conv_m (latent 16², 32-16-8) | 142 | 37.96 | 31.31 | 0.9737 | 37.73 | 0.193 | 0.95 | 0.76 |
| neural grid_mt (G32, 32 time slices) | 260 | 40.26 | 33.76 | 0.9897 | 41.22 | 0.089 | 1.01 | 1.09 |
| neural grid_l (G48 C8, H32 L2) | 292 | 42.73 | 36.23 | 0.9930 | 41.42 | 0.039 | 0.98 | 1.09 |
| flipbook BC3 16 frames 64 px + motion vectors | 72 | 32.71 | 26.69 | 0.9618 | 34.50 | 0.574 | 0.83 | 0.007 |
| flipbook BC3 32 frames 64 px | 128 | 32.38 | 26.39 | 0.9593 | 33.46 | 0.647 | 0.81 | 0.007 |
| flipbook BC3 64 frames 64 px | 256 | 33.63 | 27.59 | 0.9671 | 35.60 | 0.602 | 0.89 | 0.007 |
| flipbook BC3 16 frames 128 px + motion vectors | 288 | 35.61 | 29.25 | 0.9811 | 35.20 | 0.123 | 0.93 | 0.007 |
| flipbook BC3 32 frames 128 px | 512 | 36.36 | 29.92 | 0.9838 | 34.50 | 0.103 | 0.94 | 0.007 |
| flipbook BC3 64 frames 128 px (every frame) | 1024 | 41.02 | 34.48 | 0.9924 | 39.39 | 0.102 | 1.03 | 0.007 |

**Matched memory.** The best neural configuration against the best flipbook configuration within each budget,
paired over the 12 clips (active PSNR, then SSIM):

| budget | neural | flipbook | active PSNR difference | SSIM difference |
|---|---|---|---:|---:|
| 128 KB | conv_s, 69 KB | BC3 16f 64 px + MV, 72 KB | +2.78 [+2.27, +3.37] | +0.0045 [-0.0009, +0.0099] (tie) |
| 160 KB | grid_m, 132 KB | BC3 16f 64 px + MV, 72 KB | +5.93 [+4.99, +7.01] | +0.0255 [+0.0202, +0.0307] |
| 256 KB | grid_mt, 260 KB | BC3 64f 64 px, 256 KB | +6.16 [+4.82, +7.50] | +0.0226 [+0.0182, +0.0271] |
| 320 KB | grid_l, 292 KB | BC3 16f 128 px + MV, 288 KB | +6.98 [+5.70, +8.22] | +0.0119 [+0.0083, +0.0158] |
| 512 KB | grid_l, 292 KB | BC3 32f 128 px, 512 KB | +6.30 [+4.87, +7.70] | +0.0092 [+0.0065, +0.0123] |

**Memory at equal quality** (flipbook memory for the same mean active PSNR, along the best-flipbook envelope):
conv_s 7.4x, grid_m 5.9x, conv_m 4.5x, grid_s 3.6x, grid_mt 3.5x; grid_l (292 KB) is above every flipbook up to the
full 1 MB BC3 one (36.23 against 34.48), so its ratio is more than 3.5x.

**By effect** (grid_m, 132 KB, against the 1 MB BC3 flipbook of every frame, active PSNR): fire 31.47 against 30.44,
smoke 31.86 against 36.48, explosion 34.53 against 36.51. The average hides this: the network beats the full flipbook
on fire at an eighth of the memory, but not on smoke or explosions, where the 4x larger grid_l or more is needed.

**Held-out frames.** Given only the even frames, scored on the odd ones (active PSNR): grid_m (132 KB) 29.52; BC3 even
frames (512 KB) 28.10, 1.43 [0.07, 2.94] dB below the network; raw RGBA8 even frames (2 MB) 28.81, a tie; BC3 even
frames with motion vectors (576 KB) 33.40, **3.88 [2.22, 5.35] dB above the network**. A trained network interpolates
time better than frame blending, but motion-vector flipbooks interpolate better still.

**Flicker and motion.** The networks are temporally smooth: flicker ratios 0.6-1.0 (1 = as much frame-to-frame jitter
as the reference) and motion ratios 0.93-1.01. Low-frame-count flipbooks lose motion (ratios 0.6-0.9) because frame
blending averages it away.

![A: smoke. Rows: reference; neural grid_m, 132 KB; BC3 32 frames at 64 px, 128 KB; BC3 8 frames at 128 px with motion vectors, 144 KB](figures/a_smoke_compare.png)

![A: fire, same rows](figures/a_fire_compare.png)

## 4. B. Controls: settings the model never saw

One 1 MB model per effect (grid, 8 feature volumes blended by the controls) against a library of 45 BC3 flipbooks
(45 MB per effect), on 10 held-out settings per effect (30 in all):

| method | MB per effect | active PSNR | PSNR | SSIM | spectrum | motion | ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| neural grid, 8 volumes (k8) | 1.0 | 17.12 | 23.37 | 0.7750 | 0.218 | 0.99 | 1.07 |
| neural grid, 16 volumes, H48 (k16) | 2.0 | 17.28 | 23.42 | 0.7759 | 0.239 | 0.99 | 1.75 |
| library: nearest training setting | 45 | 14.64 | 20.73 | 0.7687 | 0.184 | 1.14 | 0.007 |
| library: blend of the two nearest | 45 | 16.13 | 21.81 | 0.7531 | 0.220 | 1.05 | ~0.014 (two lookups) |
| oracle: the closest training clip | (180) | 15.09 | - | - | - | - | - |

Paired over the 30 held-out settings (active PSNR): k8 minus nearest **+2.49 [+2.22, +2.77]**, k8 minus blend
**+0.99 [+0.65, +1.36]**, k8 minus oracle +2.03 [+1.72, +2.37]; k16 adds 0.15 dB at twice the memory and 1.6 times the
cost. By effect (k8 against the blend): fire 19.16 against 18.23, smoke 17.06 against 16.96 (close to a tie),
explosion 15.15 against 13.21.

Read these numbers with care:
- **Every method scores low** (active PSNR 15-17 dB): at a new setting the turbulence is chaotic, so no method can
  place the fine detail pixel by pixel.
- **Pixel scores reward averaging.** The network's lead comes partly from predicting a safer, softer flame. Its
  sharpness statistic equals the blend's and its motion is right (0.99), but by eye its held-out flames are dimmer and
  more broken up than the simulation's, while the nearest flipbook is crisp but the wrong shape (figure below).
- **Capacity limits the fit.** On its own training settings the 1 MB model reaches 23.7-26.5 dB active PSNR, well
  below the 31-33 dB of a 132 KB model trained on one clip (study A): one small model for 45 settings trades detail
  for control.

![B: fire at a held-out setting. Rows: simulation (truth); neural k8 (1 MB, all settings); the nearest training flipbook (45 MB library)](figures/b_fire_heldout.png)

## 5. C. Variation: new seeds

Models with 8-dimensional variation codes on 24 training seeds per effect; 8 generated seeds against 8 held-out
simulated seeds (k8: 1 MB; k24: 3 MB; a library of the 24 training flipbooks would be 24 MB):

| effect | real seeds: spectrum spread | generated vs held-out: spectrum | motion | diversity: active PSNR between two generated (real pairs) | generated vs nearest training (held-out real vs nearest training) | training seed replayed |
|---|---:|---:|---:|---:|---:|---:|
| fire | 0.13 | 0.33 | 0.86 | 13.7 (12.9) | 20.5 (14.1) | 24.9 |
| smoke | 0.13 | 0.42 | 0.85 | 15.3 (13.7) | 21.1 (14.9) | 26.1 |
| explosion | 0.07 | 0.21 | 0.94 | 15.8 (13.9) | 24.1 (15.3) | 29.8 |

(k8 shown. k24 is within 0.05 of k8 on the spectrum and motion statistics and within 0.4 dB on the PSNR-based ones, and up to 0.6 dB better at replaying training seeds, at three times the memory.)

- **Diversity is close to real**: two generated variations differ from each other almost as much as two real seeds.
- **They are softer than real**: 2.5 to 3.2 times the real spectrum spread, and 6-15% less motion.
- **They are morphs of the training seeds, not new ones**: each generated variation is far closer to some training
  seed (20-24 dB) than a fresh real seed is (14-15 dB). That is what blending two training codes produces. For a game
  this means a looping effect that never repeats exactly (the runtime drifts between codes), drawn from what it was
  trained on.

![C: fire. Rows 1-3: held-out simulated seeds; rows 4-6: generated variations (k8, 1 MB); rows 7-9: generated variations (k24, 3 MB)](figures/c_fire_variations.png)

## 6. Runtime cost and budget

Median ms per frame through `nvfx_render`, one pinned AVX2 core (90th percentiles and every configuration in
[SUMMARY.md](../results/experiments/SUMMARY.md)):

| model | KB | 64 px | 128 px | 256 px | MAC/px |
|---|---:|---:|---:|---:|---:|
| grid_s | 73 | 0.10 | **0.38** | 1.49 | 209 |
| grid_m | 132 | 0.27 | 1.06 | 4.12 | 1,425 |
| grid_l | 292 | 0.27 | 1.09 | 4.31 | 1,426 |
| conv_s | 69 | 0.62 | 0.57 | - | 504 |
| control model k8 (B) | 1,031 | 0.29 | 1.07 | 4.11 | 1,432 |
| variation model k8 (C) | 1,033 | 0.30 | 1.15 | 4.23 | 1,432 |
| fluid simulation (solver + render) | - | - | 8.8-11.7 | - | - |
| flipbook playback | 128-1,024 | - | 0.007 | - | - |

- Cost depends on the network, not on the memory: grid_l stores twice grid_m's features at the same cost; the 1 MB
  control and variation models cost the same as grid_m.
- **The 0.5 ms budget is met by grid_s (0.38 ms) only.** grid_m takes 1.06 ms (0.27 ms at 64 px). conv_s takes 0.57 ms
  at every size: its level of detail renders the native frame and filters it down, so distant copies save nothing.
- AVX-512 is slower than AVX2 on this machine (grid_m 1.36 against 1.06 ms); the baseline SSE2 build is 2.2 times
  slower. The default is AVX2.
- The networks are 8 to 30 times cheaper per frame than running the simulation, and 50 to 150 times more expensive
  than playing a flipbook.

**A scene** (`nvfx_scene`): 8 instances at 128 px and 16 at 64 px, each updated at 30 Hz, staggered over a 60 fps
game, on one core: grid_m 7.2 ms per game frame on average (43% of 16.7 ms; 99th percentile 22.5 ms, from VM
preemption spikes), grid_s 2.6 ms (16%; p99 8.7 ms), conv_s 9.0 ms (54%). Sharing instances between copies with the
same controls and seed lowers this further; the game's other CPU work has to fit around it.

## 7. Against NVIDIA's published claims and traditional methods

| | NVIDIA (published) | NeuralVFX (measured here) |
|---|---|---|
| what | small networks in shaders (RTX Neural Shaders; Neural Texture Compression; Neural Materials) on GPU tensor cores; DLSS 5, a full-frame model | a small network per effect on one CPU core |
| memory | NTC: "up to 8x" less texture memory than block compression "at similar visual fidelity" | 3.6x to 7.4x less memory than flipbooks at equal mean active PSNR (study A) |
| speed | no per-pixel costs published; DLSS 5's demo reportedly used a second RTX 5090 | 0.38-1.1 ms per 128 x 128 sprite on one CPU core |
| controls and variation | not claimed for effects | continuous learned controls and endless drift (studies B, C), at a fidelity cost |

The memory ratios are in the range NVIDIA claims for static textures, now for animated effects on a CPU. They are
measured against our own BC3-layout encoder plus motion-vector flipbooks; a production BC7 encoder reaches higher
quality at the same 8 bits per pixel, so the ratio against a production BC7 flipbook would be lower than these
figures (not measured here). DLSS 5 is a different problem (whole frames on a GPU) and is not comparable.

Against traditional methods:
- **Flipbooks** remain far cheaper to play (0.007 ms) and, with motion vectors, interpolate time better than the
  network. The network's advantages are memory (study A), controls (study B) and variation (study C).
- **Simulation** is exact at any setting but costs 8.8-11.7 ms per frame plus warm-up; the network is 8 to 30 times
  cheaper and needs no warm-up.

## 8. Limits

1. **Simulated data only.** No owner footage was supplied; real fire, smoke and explosions have more detail than the
   2D solver, and results may differ. The ingest path and its licence checks are built and tested.
2. **The flipbook baseline is our own BC3-layout encoder** (BC1 + BC4); BC7 and ASTC were not available here.
   Production flipbooks would score higher at the same size, narrowing the memory ratios.
3. **Pixel metrics reward blur where detail is unpredictable** (studies B and C). The networks' held-out outputs are
   visibly softer or dimmer than the truth; an artist would see this.
4. **Variations are morphs of the training seeds**, softer than real ones, not new turbulence.
5. **The runtime misses the 0.5 ms target for the better models** (grid_m 1.06 ms). The kernels run at about 45% of the
   core's FMA peak; int8 or VNNI kernels and a projected first layer were not done. The conv family's level of
   detail saves no time.
6. **One cloud VM.** Timings carry VM jitter (90th percentiles usually 4-10% above the medians, up to 70% in a few cells; scene p99 three times the mean).
7. **Small samples.** 12 clips (A) and 30 settings (B) from one simulator; intervals are over those, not over the
   variety of effects a game has.
8. **No engine plugin was built or tested in an engine.** The C API is engine-neutral and its example host is tested.

## 9. Recommendations

- For a game today: use **grid_s** (73 KB, 0.38 ms) where memory matters most and some softness is acceptable, or
  **grid_m** (132 KB, 1.06 ms at 128 px, 0.27 ms at 64 px) when quality matters, evaluated at 20-30 Hz and shared
  between instances; keep motion-vector flipbooks where per-frame cost must be near zero.
- Train one model per effect *and* setting for hero effects (study A quality); use one controllable model per effect
  (study B) where artists need sliders, accepting softer detail.
- Next work, in order: owner footage; a BC7 baseline (an open-source encoder fetched at build time); faster kernels
  (int8, a projected first layer, a cheaper conv level of detail); a loss that keeps detail (spectral or adversarial)
  for controls and variation; an engine plugin (Godot is the cheapest to test).

## 10. Reproduce

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 && cmake --build build
build/nvfx_experiment all --threads 4      # data, A, B, C, media, report: about 2 hours on 4 cores
build/nvfx_experiment timing               # on an idle machine
build/nvfx_experiment report               # results/experiments/SUMMARY.md
build/nvfx_scene $NEURALVFX_DATA/experiments/models/a/fire_0_grid_m8.nvfx --near 8 --far 16
```

Clips are deterministic, so the data regenerates identically; training uses fixed seeds, but thread scheduling makes
the last digits of trained weights, and so of the scores, vary slightly between runs.
