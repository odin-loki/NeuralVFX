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
when everything runs, and are for the prototype as first written. The per-frame data is in
`results/compose/fireball_frames_before.csv` and the summary in `results/compose/fireball_profile_before.csv`.

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

Optimisation passes on these stages follow, each measured before and after with the same picture.

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
- **Cost:** see §7. The prototype's compositing is plain C++ and runs on every pixel.

What it needs to become a product feature:
1. A C API: `nvfx_scene_create`, modules placed in it, `nvfx_scene_couple(...)`, `nvfx_scene_field(...)`, one
   `nvfx_scene_step` and `nvfx_scene_render` per frame, and fields readable by the game (for gameplay: is this tile
   on fire?).
2. A script format (section 4), with a viewer to edit it live.
3. Training with couplings in the loop (above).
4. A cheaper compositor: SIMD, half-resolution light and distortion, and the engine's own renderer doing the drawing
   (the fields can be uploaded as textures).
