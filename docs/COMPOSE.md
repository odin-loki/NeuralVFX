# Composed effects: modules on modules

Status: **future feature, with a working prototype** (9 October 2026). Audience: dev, research, artists. The prototype
lives in `src/compose` (a library over the runtime's internals) and `tools/nvfx_fireball` (a scripted scene). It is
not part of the C API yet. Section 8 sketches what would make it a product feature.

## 1. The idea

The goal is effects that run on one another. Each effect is a module, and the modules read and write each other's
fields: velocity, heat and soot. A script wires them together and changes the wiring over time. Examples:
- an explosion's blast bends a nearby fire;
- the explosion's cloud is handed to the smoke model when the fireball dies down;
- a fire's smoke rises into the sky above it;
- embers that land hot light new fires;
- a scripted **force field** (a ceiling, a vortex, a gust of wind) shapes any effect it covers.

This is possible because of how rollout effects work (`include/neuralfx/rollout.hpp`, [REPORT.md](REPORT.md) §6):
- **The state is physical.** It is a coarse grid of velocity, heat and soot. Every effect trained on the project's
  simulator uses the same units, so one effect's velocity can push another effect's material, and one effect's
  state can be handed to another model.
- **Moving and projecting the flow is built into each step.** Advection and the pressure projection are not
  learned. When something outside writes into the state, the next step moves that material with the flow and makes
  the flow consistent again, with no retraining. The learned part (forces, sources, sub-grid closure) then reacts
  to whatever it finds.
- **Each step is a function of the state, the controls and the noise.** A script can change any of the three
  between steps.

Frame models (studies A to C) store frames rather than a state, so they can only be drawn, not coupled. In a
composed scene they can still appear as decoration.

## 2. Modules

| module | what it is | reads | writes |
|---|---|---|---|
| rollout effect (`compose::Module`) | a learned effect in a square tile of the world, stepped with its controls and seed | its own state | its state (velocity, heat, soot; fine fields), published to the bus |
| field bus (`FieldBus`) | every module's velocity, heat and soot resampled into world space, kept per group | all modules | (read by everything else) |
| force field (`ForceField`) | a force the script places in the world: `ceiling`, `vortex`, `wind` | the bus position | the velocity of the modules it covers |
| particles (`Particles`) | sparks, embers, debris and soot flakes, carried by the bus flow, cooling, landing | the bus flow | landing events (for triggers) |
| light (`Light`) | light from everything hot on the bus, spread by a pyramid of blurs, plus flashes | the bus heat | the light that lights soot and the ground |
| distortion | shock fronts and heat haze, bending what is seen through hot air | the bus heat, shocks | the picture |
| look | each module's learned renderer, or the field shader (light from heat, soot that absorbs and is lit by the scene and the moon) | the module's fields, light | the picture |

## 3. Couplings

These are applied between steps. Each reads and writes the runners' states through
`rt::RolloutRunner::coarse_mut()` and the related accessors. Nothing allocates.

| coupling | what it does | in the fireball |
|---|---|---|
| `blend_band(a, b, side, cells)` | two tiles of one domain share a band of cells and both take the weighted mean there; their ownership weights sum to 1 everywhere, so drawing and publishing are seamless | six tiles of the explosion model make one domain 2.5 times wider and 1.75 times taller than the model was trained on |
| `hand_over(from, to)` | a module continues from another's physical state and fine fields (memory channels start from rest) | after 2.4 s the smoke model takes the cloud over from the explosion model, tile by tile |
| `push(m, bus, gain)` | the other modules' flow moves this module's material for one step (added before the step, taken out after, so momentum does not build up) | the blast bends the wreck's fire and the new fires |
| `transfer(from, to, fraction, rows, gains)` | material moves from one module into others, conserving the amount (gains can convert one model's thin soot into another's smoke) | the second blast's cloud joins the first; the fires' smoke leaves through their tile tops into the sky |
| `suppress(m, region)` | material is removed from a region | the smoke model's learned source is kept only in the crater's tile |
| `apply(m, ForceField)` | a scripted force: `ceiling` damps rising above a height, `vortex` swirls, `wind` blows | a ceiling stops the cloud rising so it spreads into a cap and stays in view |
| triggers | rules on the bus, the shocks and the particles: when a condition first holds, an action runs | the wreck explodes when the shock front reaches it; embers that land hot light fires |
| controls from the script | any module's controls, seed, look and opacity, at any time | the crater's smoke fades; new fires grow; the cloud's look turns from fireball to smoke |

## 4. Scripts

In the prototype the script is C++ (`tools/nvfx_fireball.cpp`). It has three parts:
- modules created at the start, so the frame loop allocates nothing;
- rules (`when` → `then`, each fires once);
- per-frame couplings and control curves.

The future form is a data file that an artist edits and the runtime runs. A sketch of the fireball in that form:

```text
module crater  = tiles(explosion, 3 x 2, tile 576, band 8, look shader:cloud)  at (-80, -324)
module sky     = tiles(smoke,     3 x 2, same placement,      look shader:smoke)
module wreck   = fire(192) at (1010, ground)  controls intensity 0.62
module blast2  = explosion(320) at (1010, 560)
field  ceiling = ceiling(y -30, soft 150, damping 0.3)       from 2.0 s on crater, sky

at 0.2 s       particles fuse from (250, ground) to (640, ground) until 1.2 s
at 1.2 s       start crater from start 9; shock at (640, 500); embers 1400; debris 160; scorch
when shock(1).reaches(blast2)   start blast2; shock; embers 450
at 3.6 s       hand_over crater -> sky; crossfade look over 1.2 s
every frame    push wreck by others 0.22; transfer wreck.top -> sky soot x6
every frame    after blast2 + 0.7 s: transfer blast2 -> crater 5%
when ember lands hot, away from crater and wreck, at most 2:  start fire(144) there, grow over 1.5 s
```

## 5. Field effects

Force fields are modules too: the script places them in the world and they act on whatever they cover. The
prototype has three. Others that would fit the same interface:
- **walls and obstacles**: velocity into a mask removed, so smoke flows around a building;
- **attractors and repulsors**: a spell drawing fire towards a point;
- **pressure waves**: a radial impulse from any explosion, felt by every effect nearby;
- **inversion layers and wind shear**: what the ceiling does, as weather;
- **heat sources from gameplay**: a burning object adding heat that the fire model then turns into flames.

Because advection and projection are built into each step, any of these acts like a physical force: material moves
and the flow stays consistent.

## 6. The demonstration: a fireball

`nvfx_fireball --models DIR --out fireball.mp4` renders 9 seconds at 1280 x 720 and 30 fps from the three rollout
effects of study D (fire 82 KB, smoke 146 KB, explosion 274 KB):

| time | what happens | how |
|---|---|---|
| 0 - 1.2 s | night; a wreck burns at the right; a fuse runs along the ground | fire model (learned renderer); sparks |
| 1.2 s | the charge explodes: flash, shock front, embers, debris, camera shake, scorch | explosion model from its strongest start point, in the centre tile of six; the other five start empty and receive the fireball through their bands |
| 1.2 - 3.6 s | the fireball grows and rises; the blast bends the wreck's fire | the six tiles step together (`blend_band`), shaded by the field shader; `push` |
| 1.6 s | the shock front reaches the wreck; the wreck explodes | a trigger on the shock radius starts a second explosion |
| 2.3 s on | the second cloud drifts into the first | `transfer` |
| 2 - 4 s | embers land; two of them light new fires | particle landings trigger fire modules |
| 2 s on | the cloud stops rising and spreads into a cap | a `ceiling` field |
| 3.6 s on | the smoke model takes the cloud over; a smoke column grows from the crater; the fires' smoke joins the sky | `hand_over`; `suppress`; `transfer` |
| 8.4 - 9 s | fade out | |

![The fireball scene: keyframes at 0.9, 1.27, 1.5, 2.0, 2.9, 4.2, 6.0 and 8.2 s](figures/fireball_keyframes.png)

The video itself is not in git (data rules): it is written where `--out` says.

## 7. Cost

Measured with `nvfx_fireball --profile`, which times every stage of every frame. The machine is the report's
4-core machine, with AVX2 unless the row says otherwise. The figures are medians over the frames after the detonation,
when everything runs.

### 7.1 After the first optimisation round

Three passes, each on its own branch and each checked against the picture before it:
- **The runtime's rollout step and renderer** (4.7 times faster step, renderer 2 to 4 times): separable interpolation,
  cheaper flicker noise, vectorised advection, and a row pipeline with small rings instead of full-frame buffers. The
  runtime's parity tests are unchanged (worst difference 0). Keyframes differ from the old ones by float order only
  (82 to 111 dB PSNR).
- **Shading, light, field bus and couplings:** a field shader that skips empty spans and is vectorised, the light
  computed on the thread pool, and a 4-channel bus. Shading CPU time fell about 7 times. The light, the bus and every
  module's state are bit-exact; keyframes are within one level on at most two pixels.
- **The final-picture stages** (background, draw, distortion, bloom, tone mapping), 4.6 times less work and
  **bit-exact**: all 270 frames have the same RGB checksum as before, at 1 and at 4 threads. Work that depends on a
  row or a column alone is computed once, empty spans are skipped (skipping only ever adds exactly zero), and pixels
  use 4-lane vectors with each channel's operations in their original order.

Measured one configuration at a time, before and after on the same machine in the same session (load about 1 at the
start of each batch): the code before the first pass (commit 1d14d4a) was built beside the current code. The machine
was faster in this session than when the prototype was first profiled (§7.2): the old code ran at 143 ms per frame
here against 196 ms then, so the speed-ups below compare like with like and are smaller than 196 / 24 would suggest.

| configuration | before: ms per frame (median) | after: ms per frame (median) | p90 | max | frames per second | speed-up | model step, CPU ms | model shading, CPU ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1280 x 720, 4 threads | 143 | **24.1** | 29.9 | 77 | **41.6** | 5.9x | 55 → 13.2 | 35 → 6.7 |
| 1280 x 720, 2 threads | 205 | 38.2 | 43.5 | 57 | 26.2 | 5.4x | 54 → 12.5 | 35 → 6.5 |
| 1280 x 720, 1 thread | 332 | 67.8 | 78.6 | 105 | 14.8 | 4.9x | 53 → 11.9 | 34 → 6.2 |
| 1280 x 720, 4 threads, baseline ISA (SSE2) | 148 | 26.7 | 31.1 | 41 | 37.5 | 5.6x | 71 → 22.3 | 37 → 11.2 |
| 1280 x 720, 4 threads, AVX-512 | 145 | 23.7 | 29.0 | 37 | 42.3 | 6.1x | 56 → 14.3 | 38 → 7.3 |
| 1280 x 720, 4 threads, tiles at half size | 129 | 18.9 | 21.3 | 31 | 52.9 | 6.8x | 21 → 6.4 | 10 → 2.5 |
| 640 x 360, 4 threads | 48 | 9.5 | 10.9 | 17 | 105 | 5.0x | 21 → 6.6 | 10 → 2.5 |
| 1920 x 1080, 4 threads | 318 | 46.5 | 54.2 | 61 | 21.5 | 6.8x | 119 → 24.7 | 81 → 13.4 |

A second run of the first row, interleaved with the old code's runs, gave 22.6 ms (44 frames per second).

Stages at 1280 x 720 (median ms per frame, same session):

| stage | 4 threads, before | 4 threads, after | 1 thread, before | 1 thread, after |
|---|---:|---:|---:|---:|
| step the learned models (up to 10 at once) | 16.3 | 3.7 | 53.1 | 11.9 |
| couplings | 1.6 | 1.1 | 1.7 | 1.1 |
| field bus | 2.1 | 0.7 | 2.0 | 0.7 |
| light | 5.1 | 0.6 | 5.1 | 0.6 |
| particles (update and draw) | 0.4 | 0.4 | 0.4 | 0.4 |
| shade the models | 11.2 | 2.0 | 34.2 | 6.2 |
| background | 57.0 | 2.2 | 56.7 | 7.0 |
| draw the modules | 13.3 | 3.0 | 48.1 | 9.1 |
| distortion | 19.9 | 3.7 | 74.4 | 13.5 |
| bloom | 8.7 | 3.6 | 32.5 | 8.6 |
| tone mapping and grain | 6.1 | 2.3 | 22.7 | 8.1 |

![Stage times per frame, 1280 x 720, 4 threads, after the first round](figures/fireball_profile_after1.svg)

What changed in the picture of the cost:
- **The scene runs at 42 frames per second at 1280 x 720 on 4 threads,** 21 at 1920 x 1080 and 105 at 640 x 360.
  The 77 ms maximum is one frame at 2.9 s whose light and shading stages took 28 and 19 ms (normally 0.6 and 2).
  It does not repeat: no other run's light stage exceeded 3 ms, and the other runs' maxima are 37 to 41 ms at
  1280 x 720. Single-stage stalls like this one (bloom 31 ms in one frame, drawing 23 ms in another) look like a
  worker thread losing its core, which the pool then waits for.
- **The learned models are still the small part:** stepping and shading take 5.7 of 24 ms. The five picture stages
  take 15 ms, about 3 ms each, which is close to the cost of reading and writing a 1280 x 720 float image a few
  times; the next gains there need fewer full-screen passes (bloom's last add folded into tone mapping, light and
  distortion at half resolution) rather than faster arithmetic.
- **SIMD now matters for the frame:** the baseline SSE2 build is 11% slower overall (before: 4%) and its model step
  1.7 times slower. AVX-512 is still no faster than AVX2.
- **Threads:** one to four threads is 2.8 times faster (before: 2.3, same session).
- **Memory:** unchanged in kind. The 16 modules' working memory grew from 115 to 127 MB, mostly the faster runner's
  per-instance buffers (4.0 MB against 3.4 MB for one 128 px instance), and peak resident memory from 231 to 247 MB.
  The frame loop still allocates nothing (0 in 269 frames).
- **perf** (6 s of the scene): distortion 14%, tone mapping 12%, the runtime's detail step 11%, background 10%,
  drawing 9%, bloom 13% over its passes, the runtime's coarse convolution 3%, shading 3%, `sinf` 3% (the haze).

The summaries are in `results/compose/fireball_profile_after1.csv` (before and after, same session) and
`results/compose/fireball_profile_before.csv` (the first profile, §7.2), with that profile's per-frame data in
`results/compose/fireball_frames_before.csv`.

### 7.2 Before: the prototype as first written

The first profile, taken in an earlier session of the machine that ran the same code about 1.4 times slower than the
session of §7.1. Its absolute times are kept as measured; compare across sections with §7.1's same-session rows.

| configuration | ms per frame (median) | p90 | max | frames per second | model step, CPU ms | model shading, CPU ms |
|---|---:|---:|---:|---:|---:|---:|
| 1280 x 720, 4 threads | 196 | 232 | 335 | 5.1 | 78 | 49 |
| 1280 x 720, 2 threads | 276 | 325 | 419 | 3.6 | 75 | 46 |
| 1280 x 720, 1 thread | 456 | 554 | 680 | 2.2 | 73 | 44 |
| 1280 x 720, 4 threads, baseline ISA (SSE2) | 203 | 239 | 295 | 4.9 | 100 | 50 |
| 1280 x 720, 4 threads, AVX-512 | 198 | 240 | 295 | 5.0 | 82 | 52 |
| 1280 x 720, 4 threads, tiles at half size | 164 | 195 | 251 | 6.1 | 25 | 13 |
| 640 x 360, 4 threads | 58 | 70 | 96 | 17.3 | 24 | 12 |

Stages at 1280 x 720 (median ms per frame):

| stage | 4 threads | 1 thread |
|---|---:|---:|
| step the learned models (up to 10 at once) | 22.1 | 73.4 |
| couplings | 2.1 | 2.0 |
| field bus | 2.8 | 2.8 |
| light | 6.2 | 6.2 |
| particles (update and draw) | 0.6 | 0.6 |
| shade the models | 15.1 | 44.3 |
| background (sky, stars, hills, ground) | 74.1 | 74.8 |
| draw the modules | 20.2 | 72.0 |
| distortion (shock fronts, heat haze) | 27.6 | 98.5 |
| bloom | 11.8 | 37.2 |
| tone mapping and grain | 8.9 | 31.3 |

![Stage times per frame, 1280 x 720, 4 threads, before optimisation](figures/fireball_profile_before.svg)

What the profile shows:
- **The learned models are the small part.** Stepping and shading them takes 37 ms of the 196 (19%), or 127 ms of CPU
  time over up to ten models. The hand-written picture stages take the rest. The background alone takes 74 ms
  because it runs on one thread and evaluates the light, a sine and a hash for every pixel.
- **The models scale with threads; the prototype's picture stages scale badly.** One to four threads is 2.3 times
  faster overall: the model step 3.3 times, the background not at all.
- **SIMD matters for the models only.** AVX2 against the baseline SSE2 build: the step takes 22 ms against 28 ms, but
  the frame barely changes, because the compositing is plain code. AVX-512 is no faster than AVX2 (as in the report).
- **Smaller tiles make the models 3 to 4 times cheaper but the frame only 16% faster**, for the same reason.
- **At 640 x 360 the whole scene runs at 17 frames per second** on 4 threads.
- **Memory:** the three effects take 1.4 MB resident. The 16 modules take 115 MB of working memory: 8.7 MB per
  384-pixel tile, created up front so the frame loop allocates nothing (0 allocations in 269 frames). Peak resident
  memory is 231 MB, and setup takes 0.45 s.
- **perf** (6 s of the scene): the runtime's rollout step 13%, the compositor's bilinear helper 18%, tone mapping 8%,
  drawing 8% plus its ownership weights 6%, distortion 7%, background 6% plus `sinf` 5%, bloom 8%, shading 3%.

§7.1 has the same measurements after the first optimisation round.

## 8. Limits and what a product feature needs

- **Couplings are outside the models' training.** Each model was trained alone. A hand-over or a transfer gives it
  states it never saw, and a push gives it flows it never felt. It copes because the physics it relies on is built
  in, but its learned part was not trained for this. Training the stepper with random outside forcing and random
  hand-overs would put couplings inside its training.
- **A model's learned source cannot be switched off**, only removed afterwards (`suppress`), which costs a little
  material that passes through.
- **Tiles solve their pressure separately.** Bands exchange the state, not a global solve. The seams hold in the
  scene, but a strong flow across a band can show one.
- **The look is hand-tuned.** The field shader and its constants were set by eye for this scene. Each learned renderer
  has its own look, so tiles of different models drawn by their renderers would not match. A learned renderer shared
  by all models would fix that.
- **The units are shared because the simulator is shared.** Effects trained on other data (footage, another solver)
  would need a map between their units.
- **Cost:** see §7. After the first round the scene runs at 42 frames per second at 1280 x 720 on 4 CPU threads; the
  compositing still runs on every pixel at full resolution.

What it needs to become a product feature:
1. A C API: `nvfx_scene_create`, modules placed in it, `nvfx_scene_couple(...)`, `nvfx_scene_field(...)`, one
   `nvfx_scene_step` and `nvfx_scene_render` per frame, and fields readable by the game (for gameplay: is this tile
   on fire?).
2. A script format (section 4), with a viewer to edit it live.
3. Training with couplings in the loop (above).
4. A cheaper compositor: SIMD, half-resolution light and distortion, and the engine's own renderer doing the drawing
   (the fields can be uploaded as textures).

## 9. Couplings in training (study I)

Status: **design and rule written before the test** (10 October 2026). Code: the simulator's hooks
(`sim::Fluid::push`, `add_material`), forced and hand-over runs and the coupled loss (`src/train/rollout_train.cpp`),
and the study's steps (`nvfx_experiment i-data | i-probe | i-train | i-val | i-test`, `tools/experiment_i.cpp`).

### 9.1 The question

Each rollout effect of study D (v1) was trained alone. In a scene a push, a force field, a transfer or a hand-over
gives it states and flows it never saw (§8). Does putting these couplings into training make a model follow the
simulator better when it is coupled, without making it worse when it runs alone?

### 9.2 What the fireball does to each model

`nvfx_fireball` was run once with every coupling measured on the coarse grid, per module and frame (velocities in
cells of the 32-cell grid per frame; heat and soot in the simulator's units):

| module | its own flow (RMS) | push (strongest cell: median / max) | ceiling (velocity change) | material in, per frame | material out, per frame |
|---|---:|---:|---:|---:|---|
| burning wreck (fire) | 0.35-0.48 | 0.18 / 0.49 | - | - | half of the top 4 rows (transfer to the sky) |
| new fires (fire) | 0.39-0.58 | 0.14-0.22 / 0.32 | - | - | half of the top 4 rows |
| explosion tiles | 0.20-0.49 | 0 | up to 0.06 | heat up to 0.056, soot up to 0.047 | - |
| second explosion | 0.15-0.36 | 0.07 / 0.09 | - | - | 5% everywhere (transfer to the cloud) |
| smoke tiles | 0.12-0.46 | 0 | up to 0.10 | soot up to 0.017 | everything over the source in two tiles (suppress) |

So the wreck's fire is pushed by up to its own speed, and the clouds are damped and fed. The smoke model also starts
from the explosion model's state (hand-over at 2.4 s after the detonation).

### 9.3 Design

- **Ground truth for coupled runs is the simulator with the same operations.** `Fluid::push(du, dv)` adds a velocity
  field; `push`, `step_frame()`, `push(-du, -dv)` is what `compose::push` does to a learned effect (the flow moves
  material for one frame and does not build up). `add_material(dT, dD)` adds heat and soot, clamped at zero. Push then
  unpush without a step restores the state up to one float rounding of each sum (exactly where the addition is exact;
  a zero push changes nothing, bit for bit); the tests check this, determinism, and that material moves with a push.
- **Forced runs.** Each run has a random spec (`random_forcing`, seeded): one to four events, each with a random onset,
  duration, place and amplitude, of five kinds: a temporary push (35%: a gust, a vortex or a drifting wave; up to 0.5
  cells per frame on fire, 0.3 on the others; 10 to 120 frames), a lasting force (15%: a kick of 2 to 12 frames that
  stays in the flow), a ceiling (15%: v multiplied by down to 0.7 per frame above a height, 30 to 150 frames),
  material in (15%: up to 0.06 heat and soot per frame in a blob) and material out (20%: 2% to 60% per frame, or all,
  in a disc, over the source, or through the top). The fields are evaluated at the simulator's resolution and
  applied to it; their coarse version is averaged and scaled exactly as `coarse_from_sim` averages the state, and
  stored with the run for the state it produced. The amplitudes cover §9.2 and go somewhat beyond it.
- **Hand-over runs (smoke).** An explosion simulated for 0.8 to 3 s, its state set into a smoke simulation
  (`set_state`) and recorded from there: the smoke model's training then contains explosion states.
- **The coupled loss.** In `window_loss` (and its burn-in) the operations are applied to the model's state before
  each step exactly as compose applies them (material, then the ceiling's v multiplier, then force and push), and the
  push is taken out after the step. The gradient passes through the offsets unchanged, is scaled by the multipliers
  and stops where heat or soot is clamped at zero (the finite-difference test covers it). As in compose, the state
  is clamped to the training range inside the step, before the push is taken out.
- **Fine-tuning from v1, not retraining.** v1 is loaded (frozen: never overwritten), its normalisation kept, and only
  the stepper trained, with the recipe's last stage (windows of 16 frames, half after up to 48 frames of the model's
  own rollout, 32 for explosions; the profile and activity losses) at a lower learning rate, on a mix: a window comes
  from a coupled run (forced or hand-over) with probability `share`, otherwise from a plain run. Renderer, detail
  constants and start points stay v1's, so only the stepper differs. The plain runs are v1's own training runs
  (salt 1, the first 96 of fire and smoke and 144 of the explosion); the forced runs are new salt-1 runs (96 fire, 64
  smoke plus 64 hand-overs, 144 explosion).
- **Choices on validation only.** A 2 x 2 grid (share 0.5 and 0.8, learning rate 1e-4 and 3e-4; v1's last stage
  used 7e-4), 400 iterations of 16 windows each, with a model saved every 100 iterations, is scored on validation:
  salt-3 runs, study G's 10 validation settings, validation seeds and validation forcing seeds, with the measures of
  §9.4. The chosen candidate, **v2c**, has the best coupled tracking (mean PSNR at 8 and 30 frames over the forced
  cases, and for smoke the hand-over cases too) among the candidates that stay within guards against v1 on
  validation: plain tracking (mean over 1, 8, 30 and 60 frames) at most 0.1 dB lower, spectrum distance at most 0.01
  higher, mean |log motion ratio| at most 0.03 higher, coverage distance at most 0.005 higher and mean-frame PSNR at
  most 0.3 dB lower (if none stays within them, the best coupled score is taken anyway, and the test decides). The
  same fine-tuning with share 0 (plain runs only) at v2c's learning rate, stopped at v2c's checkpoint, is the control
  **v2p**: it shows how much of any change comes from the couplings and how much from more training.

### 9.4 Evaluation

- **Forced tracking.** Study B's 10 held-out settings, two new seeds each, random couplings from held-out forcing
  seeds (onsets in the first half second). The truth is the simulator with the couplings; the model starts from the
  true state (coarse and fine, as stored) with the run's noise seed and gets the same couplings, through the
  runtime's runner as compose drives it (the material also enters the fine fields). Active PSNR against the truth at
  1, 8, 30 and 60 frames.
- **Hand-over tracking (smoke).** Explosions of salt 2 handed to the smoke simulation after 0.8 to 3 s, at B's
  settings with new seeds; the smoke model takes the true explosion state over as `compose::hand_over` does.
- **No regression.** Plain tracking as study D's (salt-2 runs from their true states, here 16 runs, D's 8 and 8
  more of the same kind), and the endless statistics exactly as study D's test (B's settings, D's seeds, shards):
  spectrum distance, motion ratio (as |log ratio|), coverage distance and mean-frame PSNR.
- **A diagnostic outside the rule:** the forced cases again without the couplings (same settings, seeds and start
  states), which shows what the couplings cost each model.
- **Intervals:** 95% paired bootstrap (10,000 resamples) over runs or settings, model minus v1.

### 9.5 The rule (written before the test)

v2c is kept for an effect when all of these hold on the test, which is run once:
1. **Coupled tracking is better:** forced tracking is better than v1 at 8 and at 30 frames, both intervals above
   zero. For smoke, either the forced or the hand-over tracking is better at 8 and 30 frames in this sense, and the
   other is not worse at 8 or 30 frames (no interval entirely below zero).
2. **Plain tracking is not worse:** at 1, 8, 30 and 60 frames no interval lies entirely below zero.
3. **The endless statistics are not worse:** for spectrum distance, |log motion ratio|, coverage distance and
   mean-frame PSNR, no interval lies entirely on the worse side.

Otherwise v1 stays. Every effect is reported, nulls with their numbers.
