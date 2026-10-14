# The effect manager: many rollout effects within a CPU budget

Status: **current** (14 October 2026; the benchmark in §7 was provisional, §7.1 re-times it on a quiet machine). Audience: dev. The C API is `include/neuralfx/nvfx_manager.h` (in libnvfx);
the code is `src/manager/`; the tools are `nvfx_crowd` (§7) and `nvfx_lod` (§5).

One rollout effect playing on its own costs about 0.75 to 1.1 ms per 128 x 128 frame on one core and 2.4 MB of working
memory (docs/REPORT.md §6.7): fine for a handful of hero effects, not for a battlefield. The manager plays a whole
scene's rollout effects within a CPU budget per game frame: it decides every tick which simulations update, at what
size and rate, and on which thread, draws the frames between updates, shares simulations between copies, pauses what
is off screen, and, when the budget is short, shows unimportant effects from a few shared proxy simulations.

## 1. Bottom line

- **A battlefield of 165 effects fits a 4 ms budget per 60 Hz frame on one core, once the manager has measured the
  machine** (§7.1: with `nvfx_manager_calibrate`, 3.8 to 4.3 ms of CPU per tick on a quiet machine; with the default
  costs on a machine 1.8 times slower than they assume, 6.7 ms). First measured on the shared machine (§7): fires, smoke
  columns, steam vents, rows of torches and explosions, 133 on screen on average in a fly-through, took 4.2 ms of CPU
  per tick (105% of the budget; 99th percentile 9.0 ms) and 120 MB. Playing every one of them with `nvfx.h` alone, each
  its own instance at 64 px, took 47.7 ms per tick (11 times as much) and 257 MB (398 MB resident). At 8 ms on two
  threads and 16 ms on four the manager used 8.5 and 16.7 ms.
- **What it looks like at 4 ms**: about 13 simulations run at once (plus 9 paused off screen); the 5 or so nearest or
  largest effects (explosions among them) have their own, at the size they are seen and mostly 30 Hz, and the other 148 instances are shown
  from 8 or so proxies, copies of a few runs at 64 px and 7.5 Hz, offset in time so that they do not move in step but
  plainly the same shapes when several are in view (rows of identical torches, smoke columns that repeat). At 16 ms on
  four threads 46 simulations run, 15 of them proxies, and the far field looks varied. The sheets and videos are in
  `$NEURALVFX_DATA/scale/crowd`.
- **The floor is the coarse step**: every simulation steps 30 times a second whatever its size and rate (0.32 ms a
  step on one core), so a core runs 15 to 18 simulations within 4 ms per 60 Hz tick, and a crowd beyond that is
  sharing. A lower update rate saves only the pictures: 8 to 18% of a simulation at 64 px and below, 19 to 41% at 128
  and 256 px; a smaller size saves 16 to 67%.
- **Quality** (test run once, held-out settings, fresh seeds, §5.4): between updates the default **crossfade** draws
  frames 40, 36 and 34 dB (active PSNR per frame) from the full-rate frames at 32 px for 15, 10 and 7.5 Hz, and 35,
  32 and 29 dB at 128 px, keeping the motion within 4 to 11% and softening detail a little. It beats holding the last
  update by 7.74 dB [+7.31, +8.21] and the motion-compensated warp along the effect's own flow by 3.44 dB [+1.87,
  +5.11] (the warp moves a flame's base with the gas; it tied only on the explosion). A smaller picture at its own size
  is statistically the same effect (spectrum and motion close) but drawn larger than itself it blurs, the largest loss
  measured. A proxy's picture ties another run of the effect on fire and smoke on every statistic; steam's average
  picture is 2.6 dB further (snapped controls) and magic's proxies are worse.
- **Exact where it can be**: at a locked level the manager's frames are `nvfx_render`'s to the bit; the pictures are
  the same on any number of threads (the plan comes from a cost model, never from measured time); a tick allocates
  nothing; planning costs 0.03 to 0.08 ms of the calling thread.
- **New in the runtime**: a runner can take another's state at another size (the coarse state exactly), and lean
  runners keep only state (186 KB less each): a simulation slot is 0.5 MB at 32 px against 1.3 MB for an `nvfx_instance`.

## 2. Design

### 2.1 Instances and simulations

An **instance** is what the engine plays: an effect at a place with its controls, seed and priority. A **simulation**
is what runs: one shard timeline of a rollout effect, the same as `nvfx_render` plays (6 s shards from start points
chosen by the seed, each rolled ahead and crossfaded into the next; docs/ENGINES.md §5). An instance views one
simulation. Instances with the same effect, controls and seed view the same simulation (`share`, the default), and a
copy may view it `time_offset` seconds behind (§2.5).

A simulation is advanced to **keyframes** only. At its update rate (every 1, 2, 3 or 4 frames of the effect's 30 fps:
30, 15, 10 or 7.5 Hz) it is stepped to the next keyframe and drawn there, at its size (32, 64, 128 or 256 px); every
frame is still stepped (the stepper is trained for one frame), so a lower rate saves pictures, not steps. The
simulation keyframes one interval ahead of what it shows, and the frames between the last two keyframes are drawn by
interpolation (§2.3). At 30 Hz the keyframe is the frame: **at a locked level (any size, 30 Hz) the manager's frames are
`nvfx_render`'s to the bit** (`ctest -R Manager`: the small effects of the tests through shard changes, warm-ups and a
one-shot's end, and the trained fire, explosion and steam at 32 and 128 px).

The machinery of a simulation at one size, two runners (the shard on screen and the next) and its keyframe pictures,
is a **slot** from a pool per effect and size, made by `nvfx_manager_add_model`. A size change moves the state to a
slot of the new size (`RolloutRunner::transfer_from`, new in the runtime): the coarse state, the pressure, the flow,
the time and the seed exactly, the fine fields resampled (block averages down, bilinear up). The coarse state does not
depend on the size (checked: 200 steps of fire and smoke at 32, 64 and 128 px give the same coarse state to the bit),
so after a size change the simulation goes on exactly as before; only the detail layer starts from a resampled field.

### 2.2 Levels of detail, importance and hysteresis

Every tick each visible instance gets an **on-screen size** in pixels, from its position and world size through the
camera (`nvfx_manager_set_camera`, with a frustum test for visibility) or given directly (`set_screen_size`), and an
**importance**: on-screen size times its priority (default 1). A simulation takes the largest size and importance of
its visible viewers.

- **Size.** The smallest level that holds the on-screen size times `texel_scale` (default 1): an effect 100 px across
  is drawn at 128 px. It moves up only when the size exceeds the level by `hysteresis` (15%), down only when it fits
  the level below with 15% to spare, and upward changes wait `min_dwell` (1 s) after the last one: a size that wobbles
  within 15% of a threshold never changes the level (`Manager.LevelsDoNotFlickerAroundAThreshold`).
- **Rate.** 30 Hz wherever the budget allows (`rate_px`, default 0, 0, 0). An engine can ask for lower rates by
  on-screen size (`rate_px`: the sizes from which 30, 15 and 10 Hz are wanted), with the same hysteresis.
- **Locks.** `nvfx_manager_set_lod_lock` pins a simulation's size and rate whatever the budget.

### 2.3 Drawing between keyframes

Three ways, chosen by `desc.interp` (and per effect by the model's `interp`):
- **hold**: the last keyframe until the next (a stutter at the update rate; costs nothing);
- **crossfade**: the two keyframes blended by the fraction of the interval (integer arithmetic; 0.009 ms at 128 px);
- **motion**: both keyframes warped along the displacement of the effect's own coarse flow between them (the flow of
  the shard on screen, summed over the interval's steps, at the detail layer's scale: S / 32 pixels per cell), a pixel
  of the frame at fraction w taken from the first keyframe at x - w d(x) and from the second at x + (1 - w) d(x),
  bilinear, then blended (AVX2 gathers where the CPU has them; 0.34 ms at 128 px, about one keyframe picture).

The quality study (§5) chose **crossfade** as the default.

### 2.4 The budget

The **budget** is CPU milliseconds per tick for all the manager's work, summed over its threads (`budget_ms`, default
4). The planner works from a **cost model**, never from measured time: milliseconds per simulation step and per
keyframe picture at each size (defaults measured on the project's machine, §6; `nvfx_manager_set_cost`, or
`nvfx_manager_calibrate` to measure this machine), and per composed picture. A simulation's cost per tick is its steps
per second (30, plus the next shard's pre-roll: 25% more for effects that warm up, 8% otherwise), its pictures per
second (30 / rate divisor, plus the shard crossfades) and its interpolated pictures, divided by the game's tick rate.

Each tick:
1. **Wanted levels**: every simulation's size and rate from its on-screen size (above); downward changes at once,
   upward ones only through step 3.
2. **Over budget**: the simulations are lowered **least important first**, each fully before the next is touched.
   A simulation at the proxies' size or smaller (64 px) goes to a **proxy** at once if it may (looping effect, every
   viewer allows it, not locked, its viewers have kept their source for `proxy_dwell`): at those sizes a lower rate
   saves at most 17% (the steps cost the same at every rate, §6). A larger one lowers its rate step by step to 7.5 Hz,
   then its size one level at a time, then goes to a proxy if it may. If proxies themselves are over the budget, the
   proxy with the fewest viewers merges into the nearest one.
3. **Under 85% of the budget**: the most important simulations below their wanted level go up one step each (size
   first, then rate), within 85% of the budget and after `min_dwell`; then a proxy is added for the cell whose
   instances were sent to another cell's proxy, if there is room; then proxied instances, most important first, get a
   simulation of their own again (it warms up while they still show the proxy, then they fade across).
4. **Admission**: the updates due this tick (a simulation whose display has reached its last keyframe, or one warming
   up) run in order of how long they have waited, then importance, until the tick's planned cost reaches `burst` times
   the budget (1.5); proxies' pre-rolls go first (others wait for them). An update that does not fit waits: its
   simulation shows its last keyframe and its clock waits too, so a starved effect slows down rather than jumping.
   A warm-up or pre-roll takes at most `catchup_frames` (8) steps per tick.

New instances are attached in the same terms, most important first: to a simulation of the same effect, controls and
seed if there is one; else to a simulation of their own while the plan (last tick's, plus what joined this tick) has
room, or if they matter more than what the planner last sent to proxies; else to a proxy.

### 2.5 Sharing: copies, time offsets and proxies

- **Copies** (same effect, controls and seed, `share` on) view one simulation, at the largest size any of them needs.
- **Copies offset in time** view it `time_offset` seconds behind, from a **ring** of its past keyframes kept at
  `copy_size` (64 px; `history` keyframes, default 16, so up to about 14 updates back: 0.5 s at 30 Hz, 1.9 s at 7.5 Hz). A
  simulation gets a ring from the effect's pool (`offset_sims`, default 8) when a copy with an offset joins it; a new
  one pre-rolls far enough to fill it before it is shown. Offsets are capped at the ring's span. At 30 Hz and the
  copy's size the copy shows exactly the frame its simulation showed that many frames before
  (`Manager.CopiesShareOneSimulationAndOffsetsAreEarlierFrames`).
- **Proxies** (looping effects): an instance shown from a proxy views a shared simulation at `proxy_size` (64 px) and
  `proxy_rate_div` (every 4 frames, 7.5 Hz) whose controls are the centres of the instance's **cell** (each control
  snapped to one of `proxy_steps` = 3 values), with a seed of the proxy's own, at an offset in its ring drawn from the
  instance's seed, so the copies of a proxy do not move in step. The number of proxies follows the budget (§2.4): a
  new cell gets a proxy of its own only if there is room for one more (else its instances view the nearest cell's),
  proxies merge when over budget and split when under. A new proxy pre-rolls its ring (1.9 s of simulation, after a
  1 s warm-up for the effects that grow their fine fields; at most 8 steps a tick) before anyone sees it. An instance changes source with a **fade** of `fade_frames` (8) from a
  snapshot of its last picture.

### 2.6 Off screen

A simulation none of whose viewers is visible **pauses**: its clock stops and it costs nothing; seen again it resumes
where it stopped (warm: `Manager.PausedSimulationsResumeWarm` gets the same picture as a run that never paused). After
`park_after` (1 s) it moves to the smallest size, freeing its slot; after `cull_after` (30 s), or when a new instance
needs a simulation and none is free (the one paused longest goes first), it is **culled**: released; seen again, a
looping instance starts a new run, a one-shot is finished.

### 2.7 Threads and determinism

A tick is `tick_begin` (planning, on the calling thread), jobs, and `tick_end`. A **job** is one simulation: its level
change, its update, its picture and the pictures of the copies and proxied instances that view it; it touches nothing
else, and uses the step scratch of the thread that runs it (a step reads nothing an earlier step left in its scratch).
Jobs are taken largest first from an atomic counter by the manager's `threads` (the caller and threads - 1 workers),
or, with `threads = 0`, by the engine's own job system: `nvfx_manager_run_jobs(manager, slot)` from up to `job_slots`
threads. Every decision depends only on the calls made and the cost model, so the pictures are the same to the bit on
any number of threads and whoever runs the jobs (`Manager.SamePicturesOnAnyNumberOfThreads`: a scene with level
changes, a short budget, proxies, offset copies, pauses, culls, spawns, stops and forks, on 1 and 3 threads and on two
engine threads, hashed tick by tick).

### 2.8 Memory

Everything is allocated by `nvfx_manager_create` and `nvfx_manager_add_model`; spawning, stopping, setting and
ticking allocate nothing (`nvfx_alloc_test`: 399 ticks of a crowd with every call a game makes, on the manager's
threads and on the engine's: 0 allocations).

- **Lean runners** (new in the runtime, `make_rollout(..., lean)`): the buffers a step or a picture fills and uses up
  (the coarse step's double buffer, the pressure's second buffer, the divergence, the noise samples, the lock factors,
  the swirl lattice, the renderer's per-cell layer and soot sums) come from the step scratch of the thread instead of
  the runner: 186 KB less per runner, the same frames to the bit. Every other user of the runtime keeps the runners as
  they were.
- **Per slot** (two lean runners and four pictures): 0.50 MB at 32 px, 0.61 MB at 64, 1.0 MB at 128, 2.6 MB at 256
  (an `nvfx_instance`, which also holds its own step scratch: 1.3, 1.6, 2.4 and 5.7 MB).
- **Per effect** (`nvfx_manager_add_model`): `slots[level]` slots per size (default `max_sims` = 16, 8, 4 and 1), 8
  more at the proxies' size, rings for the proxies and `offset_sims` (0.29 MB each at the defaults): about 29 MB at the
  defaults for a looping effect, 21 MB for a one-shot (no proxies).
- **Per manager**: one step scratch per thread for the largest runner (3.7 MB at 256 px), and two 64-px pictures per
  instance for offset copies, proxies and fades (32 KB; 8 MB for the default 256 instances). A manager with one fire at
  the defaults takes 40.6 MB in all.

### 2.9 What is new in the runtime

Two additions to `src/runtime`, neither changing any existing output (every bit-exact test passes): the runner's
`transfer_from` and the lean runners. The manager's shard timeline is a copy of `nvfx.cpp`'s (`pick_start`,
`plan_shard`, `bring`), advanced to keyframes; its tests hold it to `nvfx_render` to the bit.

## 3. The API

```c
nvfx_manager_desc d;
nvfx_manager_desc_init(&d);              /* budget 4 ms, 1 thread, crossfade, sizes 32-256 px, 30 Hz */
d.threads = 2;
d.budget_ms = 6.f;                       /* CPU ms per tick, summed over threads */
nvfx_manager* m;
nvfx_manager_create(&d, &m);

int fire;
nvfx_manager_add_model(m, fire_effect, NULL, &fire);   /* NULL: the model defaults; allocates the pools */

nvfx_manager_set_camera(m, cam_pos, cam_forward, cam_up, fov_y, 1920, 1080);
nvfx_spawn_desc s;
nvfx_spawn_desc_init(&s);
s.model = fire;
s.controls = c; s.n_controls = 3;
s.seed = 7;
s.position[0] = x; s.position[1] = y; s.position[2] = z; s.world_size = 2.5f;
nvfx_handle h;
nvfx_manager_spawn(m, &s, &h);

/* every game frame */
nvfx_manager_set_camera(m, ...);
nvfx_manager_set_position(m, h, p, 2.5f);          /* if it moves */
nvfx_manager_tick(m, dt);
nvfx_manager_picture pic;
nvfx_manager_get_picture(m, h, &pic);
if (pic.rgba && (pic.flags & NVFX_PICTURE_NEW)) upload(pic.rgba, pic.size, pic.stride);   /* premultiplied RGBA */
```

| what | calls |
|---|---|
| create | `nvfx_manager_desc_init`, `nvfx_manager_create`, `nvfx_manager_free` |
| effects | `nvfx_manager_model_desc_init`, `nvfx_manager_add_model` (rollout effects; the effect outlives the manager), `nvfx_manager_set_cost`, `nvfx_manager_get_cost`, `nvfx_manager_calibrate` |
| the view | `nvfx_manager_set_camera` (position, forward, up, vertical field of view, viewport in pixels), `nvfx_manager_set_budget` |
| instances | `nvfx_spawn_desc_init`, `nvfx_manager_spawn` (model, controls, seed, share, time offset, proxies allowed, priority, position and world size or on-screen size), `nvfx_manager_stop`, `nvfx_manager_set_controls` (in place for a simulation of its own; a shared one forks with a copy of the state, no restart), `nvfx_manager_set_position`, `nvfx_manager_set_screen_size`, `nvfx_manager_set_priority`, `nvfx_manager_set_lod_lock` |
| ticks | `nvfx_manager_tick(dt)`; or `nvfx_manager_tick_begin(dt, &jobs)`, `nvfx_manager_run_jobs(slot)` from the engine's threads, `nvfx_manager_tick_end` |
| results | `nvfx_manager_get_picture` (pixels, size, stride, the simulation's size and rate, flags: shared, proxy, interpolated, held, fading, finished, new; the effect time; visible; on-screen size), `nvfx_manager_get_stats` (wall and job milliseconds, the planner's, the planning on the calling thread; instances, visible, simulations, paused, proxies, proxied, sharing; updated, skipped, interpolated, held; steps and pictures; simulations per size and rate; memory) |

Rules: one thread calls the manager at a time (its own workers aside); spawn, stop and the setters go between ticks; a
picture stays valid until the next tick; handles of stopped instances are never reused (16-bit generations).

## 4. In an engine

- **Threads.** With the engine's job system, create the manager with `threads = 0` and `job_slots` = the most worker
  threads that may run its jobs at once; each frame call `nvfx_manager_tick_begin` on the game thread, schedule
  `nvfx_manager_run_jobs(m, slot)` on up to `job_slots` workers (each with its own slot number), and call
  `nvfx_manager_tick_end` when they have all returned. Spawns, stops and setters go before `tick_begin` or after
  `tick_end`. The budget counts the jobs' CPU time on all threads: with 4 ms and 4 workers the effects take about 1 ms
  of wall time.
- **Uploads.** A picture changes only when `NVFX_PICTURE_NEW` is set (at most at the effect's 30 fps; a 60 Hz game sees
  it every other frame). Copies without an offset share one picture (the same pointer): upload it once per simulation
  if the engine keys its textures by pointer. Draw with premultiplied alpha, as for `nvfx_render` (docs/ENGINES.md §1),
  stretched to the sprite: pictures are 32 to 256 px whatever the sprite's size.
- **Sizes.** `world_size` is the sprite's extent in world units; with `set_screen_size` the engine gives the on-screen
  size itself (and visibility) instead. `texel_scale` below 1 trades sharpness for CPU at every distance.
- **Budget.** The cost model's defaults are one AVX2 core of the project's machine; call `nvfx_manager_calibrate` once
  per effect at load (tens of milliseconds) to plan with this machine's speed, at the price of plans that depend on
  the machine (and its load at that moment). `nvfx_manager_get_stats` reports what the jobs took (`jobs_ms`) against
  the plan (`planned_ms`).
- **Pools.** Size `max_sims` and `slots` to what the budget can run (at 4 ms per 60 Hz tick, 15 to 18 simulations at
  32 to 64 px and 30 Hz, §6), `proxy_sims` to the number of distinct looks the far field needs, and `history` and `offset_sims`
  to the copies offset in time the scene uses.
- **One-shots.** Explosions are never proxied; stop them when their picture has `NVFX_PICTURE_FINISHED`.

## 5. Quality: what the levels of detail cost

### 5.1 Design (written before any run)

The manager draws a simulation's frames between its updates by interpolation, and draws distant effects at smaller
sizes. Both cost quality; this section measures how much, against **full-rate, full-resolution playback of the same
runs**, with the project's protocol (docs/REPORT.md §2 and §6).

- **Effects.** Fire, smoke and explosion (v2, `$NEURALVFX_DATA/v2/models`), steam and magic (study D's recipe,
  `$NEURALVFX_DATA/fx/models/d`).
- **Settings and seeds.** Validation: study G's ten validation settings (docs/DCM.md §4). Test: study B's ten held-out
  settings. Fresh seeds, never used before in the project: validation 7,300,000 + 100 s + e (the studies' salt-3 split),
  test 7,700,000 + 100 s + e, for setting s and effect e.
- **Runs.** 300 frames (10 s) for the looping effects, the explosion's 90; each played through the manager with its level
  locked (`nvfx_manager_set_lod_lock`), one tick per frame, keeping the frame shown at every tick (`nvfx_lod`).
- **Rate.** At 32, 64 and 128 px, updates every 2, 3 and 4 frames (15, 10 and 7.5 Hz), the frames between drawn by
  **hold** (the last update), **crossfade** (the two updates blended) or **motion** (both warped along the
  displacement of the effect's coarse flow between them, then blended); against the full-rate run at the same size. The
  keyframes are the full-rate run's frames to the bit (checked on every run), so only the frames between differ.
- **Size.** 32 and 64 px at 30 Hz against 128 px averaged down to the size shown, and drawn at 128 px (bilinear) against
  128 px.
- **Proxies.** 64 px at 7.5 Hz with the controls snapped to the centres of three cells per control and another seed, and,
  as the floor, another seed at the same controls at 64 px and 30 Hz; their statistics against 128 px averaged to 64.
- **Metrics.** Active PSNR per frame (the active-pixel PSNR of each frame against the reference frame, docs/REPORT.md §2;
  the mean over the frames drawn between updates for rate, over every frame otherwise), and the endless statistics of the
  whole run against the reference run: the detail spectrum L1 distance and the motion ratio (with the mean-frame PSNR).
- **Intervals.** Paired bootstrap over settings, 10,000 resamples, 95% percentile intervals; an interval covering zero is
  a tie.

### 5.2 The rule for the default interpolation (written before validation, committed before the test)

For each (effect, setting) pair, each method's metric is averaged over six cells: 64 and 128 px at 15, 10 and 7.5 Hz.
Paired bootstrap over the 50 pairs (ten settings of five effects).

1. **Motion** becomes the default if, on validation, (a) its active PSNR per frame minus crossfade's has a pooled interval
   entirely above zero, (b) no effect's interval for (a) lies entirely below zero, and (c) it is not worse on either
   endless statistic: the pooled interval of its |ln motion ratio| minus crossfade's has an upper bound below +0.02, and
   that of its spectrum L1 minus crossfade's an upper bound below +0.01.
2. Otherwise **crossfade** is the default (the cheaper of the two that move between updates), unless its active PSNR per
   frame minus hold's has a pooled interval entirely below zero, in which case **hold**.
3. The test runs once, after the choice is committed: the chosen default against each of the other two, per effect and
   pooled, with intervals; ties are reported as ties. The test does not change the choice.

The rate thresholds and sizes (§6) are chosen on validation from the same tables and the cost model, and reported on the
test.

### 5.3 Validation and the choice

`nvfx_lod val` (results/manager/lod_val.csv and lod_val_summary.md; 50 pairs: five effects at ten settings), the rule's
comparisons, mean over the six cells, paired over settings:

| comparison | active PSNR per frame (dB) | motion error, abs(ln ratio) | spectrum L1 |
|---|---|---|---|
| motion - crossfade, pooled | -3.10 [-4.85, -1.49] | +0.208 [+0.101, +0.326] | +0.087 [+0.072, +0.102] |
| crossfade - hold, pooled | +7.73 [+7.31, +8.17] | -0.003 [-0.003, -0.002] | +0.049 [+0.046, +0.052] |

Motion fails the rule's first condition (its interval is below zero), so by rule 2 **crossfade is the default**
(`NVFX_INTERP_DEFAULT`). Per effect, motion loses on fire (-1.10 [-2.05, -0.30]), smoke, steam and magic (-14.46
[-15.28, -13.67]) and wins on the explosion (+2.19 [+1.21, +3.03]): the flow carries an expanding fireball's picture,
but a flame or a plume keeps its shape where the gas flows through it (warping along the gas shifts the flame's base),
and the magic portal spins faster than a straight warp over four frames can follow. The test (§5.4) runs once with this
choice.

### 5.4 The test (run once)

`nvfx_lod test` (results/manager/lod_test.csv and lod_test_summary.md): study B's ten held-out settings, fresh seeds,
the five effects, after the choice was committed. (Between validation and the test the manager's runners became lean,
§2.8: the same frames to the bit, checked against `nvfx_render` on the trained effects.)

**Crossfade, the default, against the others** (mean over 64 and 128 px at 15, 10 and 7.5 Hz; paired over settings):

| effect | crossfade - motion, active PSNR per frame | crossfade - hold, active PSNR per frame | crossfade - hold, spectrum L1 | crossfade - hold, motion error |
|---|---|---|---|---|
| fire | +1.73 [+0.87, +2.51] | +6.25 [+5.83, +6.64] | +0.054 [+0.048, +0.060] | -0.004 [-0.005, -0.002] |
| smoke | +0.88 [+0.04, +1.83] | +6.78 [+6.42, +7.13] | +0.049 [+0.044, +0.054] | -0.003 [-0.004, -0.001] |
| explosion | -1.20 [-2.68, +0.21] (tie) | +10.47 [+9.82, +11.08] | +0.053 [+0.048, +0.057] | -0.000 (tie) |
| steam | +1.41 [+0.02, +2.71] | +7.12 [+6.69, +7.50] | +0.047 [+0.041, +0.053] | -0.003 [-0.004, -0.002] |
| magic | +14.38 [+13.70, +15.03] | +8.07 [+7.90, +8.25] | +0.034 [+0.033, +0.035] | -0.004 [-0.005, -0.003] |
| pooled | **+3.44 [+1.87, +5.11]** | **+7.74 [+7.31, +8.21]** | +0.047 [+0.045, +0.050] | -0.003 [-0.003, -0.002] |

Against motion, crossfade is also closer on the endless statistics: motion's |ln motion ratio| is +0.211 [+0.105,
+0.327] higher (it moves too much: ratio 1.3 to 1.8) and its spectrum L1 +0.088 [+0.074, +0.102] higher. Against hold,
crossfade draws every frame between updates 6 to 10 dB closer and keeps the motion as well (a hair better), but softens
the detail a little (spectrum L1 +0.047): a ghost of two pictures rather than a stutter. On the explosion, where
validation had motion ahead by 2.2 dB, the test has a tie.

**What the levels cost** (test, five effects pooled, crossfade): the frames between updates are 40.0, 36.5 and 33.6 dB
from the full-rate frames at 32 px (15, 10, 7.5 Hz), 38.2, 34.4 and 31.7 dB at 64 px, 35.0, 31.5 and 29.1 dB at 128
px; the motion ratio over the whole run is 0.96, 0.93 and 0.89; the spectrum L1 0.02 to 0.08. A smaller picture at its
own size: 30.7 dB per frame at 32 px and 28.3 dB at 64 px against 128 px averaged down, with spectrum L1 0.03 and 0.07 and
motion 0.98 and 0.93 (the same effect, a different detail layer); drawn at 128 px it is blurred (spectrum L1 1.17 and
0.71). The numbers are within 0.3 dB of validation's.

**Proxies** (64 px, 7.5 Hz, snapped controls, another seed) against the floor (another seed at the instance's own
controls, 64 px, 30 Hz), both against 128 px averaged to 64:

| effect | spectrum L1 | abs(ln motion ratio) | mean-frame PSNR |
|---|---|---|---|
| fire | -0.005 [-0.062, +0.059] (tie) | +0.028 [-0.222, +0.274] (tie) | -0.89 [-3.39, +1.89] (tie) |
| smoke | +0.012 [-0.022, +0.046] (tie) | +0.102 [-0.029, +0.239] (tie) | -0.07 [-1.43, +1.08] (tie) |
| explosion (no proxies in use) | +0.062 [-0.007, +0.131] (tie) | +0.131 [-0.176, +0.426] (tie) | -2.41 [-5.99, +1.56] (tie) |
| steam | -0.004 [-0.027, +0.022] (tie) | +0.102 [-0.020, +0.225] (tie) | **-2.59 [-4.36, -0.67]** |
| magic | **+0.063 [+0.040, +0.085]** | **+0.108 [+0.052, +0.163]** | **-3.29 [-5.52, -0.93]** |

A proxy's picture is statistically another run of the effect for fire and smoke (ties on all three); steam's average
picture moves with the snapped controls (-2.6 dB), and magic's proxies are worse on everything (its spin at 7.5 Hz).

## 6. Defaults, and why

Chosen on validation (§5.3) and the cost model; the test (§5.4) reports them once.

**What a level saves.** The coarse step costs the same at every size and runs every frame whatever the update rate,
so a lower rate saves only pictures. From the cost model (one AVX2 core; ms per frame at 30 Hz: step plus picture):

| size | ms per frame at 30 Hz | 15 Hz | 10 Hz | 7.5 Hz | one size down |
|---:|---:|---:|---:|---:|---:|
| 32 px | 0.39 | -8% | -11% | -12% | - |
| 64 px | 0.47 | -12% | -16% | -18% | -16% |
| 128 px | 0.78 | -19% | -26% | -29% | -40% |
| 256 px | 2.35 | -28% | -37% | -41% | -67% |

**What it costs in quality** (validation, five effects pooled, crossfade; active PSNR of the frames between updates
against the full-rate frames at the same size; the keyframes are exact):

| size | 15 Hz | 10 Hz | 7.5 Hz | at 30 Hz against 128 px: at its size / drawn at 128 px |
|---:|---:|---:|---:|---|
| 32 px | 39.8 dB | 36.2 dB | 33.3 dB | 31.0 dB / 23.7 dB (spectrum L1 0.03 / 1.12) |
| 64 px | 38.0 dB | 34.2 dB | 31.5 dB | 28.7 dB / 26.7 dB (spectrum L1 0.07 / 0.67) |
| 128 px | 34.8 dB | 31.3 dB | 29.0 dB | - |

The crossfade damps motion between updates (motion ratio 0.96 at 15 Hz, 0.93 at 10 Hz, 0.89 at 7.5 Hz) and softens
detail a little (spectrum L1 +0.02 to +0.08); holding keeps the detail but stutters (7.7 dB worse per frame). At its
own size a smaller picture is a different detail layer over the same coarse state (28.7 dB against 128 px averaged to
64, but spectrum L1 0.07 and motion 0.93: statistically the same effect); drawn larger than itself it is visibly blurred
(spectrum L1 0.67 at 64 px drawn at 128, 1.12 at 32 px), the largest loss in these tables.

The defaults that follow:
- **Size follows the screen** (`texel_scale` 1: the smallest level that holds the on-screen size, hysteresis 15%,
  upward changes at most once a second): drawing a picture larger than itself costs the most quality for what it saves.
- **30 Hz wherever the budget allows** (`rate_px` 0, 0, 0): at 64 px and below a lower rate saves 8 to 18% of a
  simulation and draws the frames between updates 31 to 40 dB from the real ones; only at 128 px and up does it save a
  fifth to two fifths.
- **Under a short budget**, least important first: a simulation of 64 px or less goes to a proxy (a lower rate would
  save little; a proxy saves nearly all of it, and its picture is statistically the effect's: §5); a larger one lowers
  its rate to 15, 10, then 7.5 Hz (29 to 35 dB per frame between updates), then its size (blur), then goes to a proxy.
- **Crossfade** between updates (the rule, §5.3), for every effect. Motion was ahead on the explosion in validation
  and tied with crossfade on the test (§5.4); an engine can set it per effect (`nvfx_manager_model_desc.interp`), at
  about the cost of a keyframe picture per picture composed, which a crowd of proxied copies cannot afford (§7).
- **Proxies**: 64 px, 7.5 Hz, controls snapped to three cells per control, eight proxies per effect, a ring of 16
  keyframes (offsets spread over 1.7 s between the copies).
- **Budget** 4 ms per tick, one thread; **burst** 1.5; **fade** 8 frames; **pause** off screen, **park** after 1 s,
  **cull** after 30 s.

## 7. The crowd benchmark (provisional)

`nvfx_crowd`: 160 instances to start with: rows of 8 identical torches (15%, offset in time by 0.37 s each) and fires,
smoke columns and steam vents (53, 29 and 18% of the rest) with random controls and seeds, over a field 80 m wide and
160 m deep; a camera 2 m up flies through at 2 m/s, yawing 25 degrees either way every 12 s, so effects come close,
leave the screen and come back. Explosions go off ahead at 1.5 a second and are stopped when they finish; a fire dies
and another starts every half second. 60 Hz, 20 s (the first second not counted), default pools (`max_sims` 16 and 8
proxies per effect). CPU is the process's CPU time per tick (every thread), wall the tick's elapsed time.

**Provisional**: measured on the shared 4-core machine while two other agents trained (load average 9 to 14), three
interleaved rounds per configuration with pinned cores (`tools/manager/time_crowd.sh build OUT 3 20`; every run in
results/manager/crowd_timing_runs_provisional.csv, the summary in crowd_timing_summary_provisional.csv); the median of
the three rounds, with the least and greatest in brackets. The planner's figures (plan, simulations, proxied) were the same
in every round: the plan does not depend on timing. Re-time on a quiet machine with the same script.

| configuration | CPU per tick, mean | CPU p99 | wall, median | wall p99 | plan | simulations (paused) | proxied | memory (resident) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 ms, 1 thread | 2.22 [2.14, 2.26] | 6.0 | 2.1 | 13.0 | 2.03 | 15.3 (7.0) | 153 | 120 MB (133) |
| 4 ms, 1 thread | **4.19 [4.19, 4.27]** | 9.0 | 5.6 | 23.3 | 3.71 | 22.2 (9.1) | 148 | 120 MB (133) |
| 4 ms, 1 thread, hold | 4.44 [4.43, 4.56] | 10.2 | 5.6 | 20.0 | 3.69 | 23.9 (10.1) | 147 | 120 MB (133) |
| 4 ms, 1 thread, motion | 6.19 [6.17, 6.26] | 10.6 | 7.8 | 22.7 | 12.10 | 12.1 (5.4) | 155 | 120 MB (133) |
| 8 ms, 2 threads | 8.52 [8.29, 8.78] | 17.2 | 12.8 | 35.1 | 7.11 | 38.4 (16.6) | 133 | 124 MB (137) |
| 16 ms, 4 threads | 16.68 [16.48, 16.69] | 30.9 | 27.8 | 62.7 | 13.89 | 62.7 (16.5) | 107 | 131 MB (144) |
| no budget, default pools, 4 threads | 22.78 [22.37, 22.91] | 37.1 | 36.2 | 71.8 | 21.07 | 70.2 (5.2) | 69 | 131 MB (144) |
| every instance its own `nvfx_instance` at 64 px, 30 Hz while on screen, 1 thread | **47.67 [47.54, 49.07]** | 140.6 | 61.5 | 211.0 | - | 134 instances | - | 257 MB (398) |

Simulations running at once by size and rate (mean over the ticks; the rest are paused off screen):

| configuration | 32 px | 64 px | 128 px | 256 px |
|---|---|---|---|---|
| 4 ms | 1.4 at 30 Hz, 0.9 at 7.5 | 0.8 at 30, 0.3 at 10, 7.5 at 7.5 (the proxies) | 0.9 at 30, 0.2 at 15, 0.7 at 7.5 | 0.3 |
| 8 ms, 2 threads | 3.1 at 30 | 4.6 at 30, 0.2 at 10, 7.5 at 7.5 | 3.7 at 30, 1.0 at 15, 0.9 slower | 0.8 |
| 16 ms, 4 threads | 5.3 at 30, 2.3 slower | 13.5 at 30, 16.0 slower (15.5 proxies) | 8.3 at 30 | 0.8 |

What the figures say:
- **The budget holds, with the cost model's error.** The jobs took 6 to 16% more CPU than the plan (the defaults
  were measured one simulation at a time; a crowd shares the caches, and the machine was loaded), so the manager used
  104 to 111% of its budget. `nvfx_manager_calibrate` measures this machine; a quiet machine should land closer.
- **Wall time on several threads** was far above CPU / threads here because the other agents' training held two of the
  four cores; on a quiet machine the jobs (one per simulation with work, largest first) spread over the threads.
- **Motion-compensated interpolation does not fit a crowd**: every proxied copy composes its own picture, 0.08 ms
  each at 64 px against 0.002 ms for a crossfade; its plan alone was 12 ms against the 4 ms budget, and the planner sent
  fewer simulations to their own runs to pay for it.
- **Spikes**: the 99th percentile is 2 to 3 times the mean, from spawns (a fire's 1 s warm-up spreads over four
  ticks at 8 steps each), explosions coming on screen, proxies pre-rolling and the next shard's pre-roll lining up.
  `burst` (1.5) caps what a tick admits; updates beyond it wait a tick.
- **Memory** is the pools: 120 MB for four effects at the default pools (three looping effects' slots, proxies and
  rings at about 29 MB each, the explosion's 21 MB, and 12 MB for the instances' pictures and the thread scratch),
  against 257 MB for 134 instances of `nvfx.h` at 64 px (1.6 MB each with its own step scratch; 2.4 MB at 128 px).

The pictures: `$NEURALVFX_DATA/scale/crowd/crowd_b4.mp4`, `crowd_b8.mp4` and `crowd_b16.mp4` (the composited view at 30
fps), `b4_view_10s.png` and the like (the view at 5, 10 and 15 s) and `b4_sheet_10s.png` and the like (every visible
instance's picture, largest on screen first, with a bar for its level: its length the size, its colour the rate, green
30 Hz to red 7.5 Hz, blue a proxy). At 4 ms the near effects are sharp but the far field repeats a handful of runs
(identical torches in a row; smoke columns of one shape); at 16 ms the far fires differ from one another. Some
sprites show a hard edge where a plume leaves the 128-pixel domain the effect was trained in (smoke's top, the
explosion's), as with `nvfx_render`.


### 7.1 Re-timed on a quiet machine

The same script one job at a time (14 October 2026; `results/manager/crowd_timing_runs_quiet.csv`, summary in
`crowd_timing_summary_quiet.csv`), on a session of the machine that ran slower than the one the cost model's defaults
were measured in (the fireball took 26 ms per frame at 720p on 4 threads, against 19 to 20 ms on 13 October). Median of
three rounds, least and greatest in brackets:

| configuration | CPU per tick, mean | CPU p99 | wall, median | plan | simulations | proxied |
|---|---:|---:|---:|---:|---:|---:|
| 2 ms, 1 thread | 3.27 [3.03, 3.37] | 8.5 | 2.9 | 2.03 | 15.3 | 153 |
| 4 ms, 1 thread | 6.71 [6.18, 7.17] | 14.8 | 6.5 | 3.71 | 22.2 | 148 |
| 4 ms, 1 thread, hold | 6.45 [6.41, 6.91] | 14.8 | 6.6 | 3.69 | 23.9 | 147 |
| 4 ms, 1 thread, motion | 8.96 [8.20, 9.07] | 16.6 | 8.6 | 12.10 | 12.1 | 155 |
| 8 ms, 2 threads | 13.09 [12.62, 15.38] | 27.6 | 6.5 | 7.11 | 38.4 | 133 |
| 16 ms, 4 threads | 26.66 [25.17, 26.96] | 51.9 | 7.3 | 13.89 | 62.7 | 107 |
| no budget, default pools, 4 threads | 36.51 [35.97, 36.67] | 65.8 | 9.6 | 21.07 | 70.2 | 69 |
| every instance its own `nvfx_instance` at 64 px, 1 thread | 69.91 [69.00, 70.66] | 195.3 | 63.5 | - | 134 instances | - |

- **Without calibration the budget is exceeded on a slower machine, by as much as the machine is slower.** The plan is
  the same as before (it does not depend on timing), but every job took about 1.8 times its default cost here, so the
  4 ms budget used 6.7 ms. The managed crowd still took a tenth of the unmanaged one's CPU (6.7 against 69.9 ms).
- **With `nvfx_manager_calibrate` it holds.** `nvfx_crowd --calibrate` measures each model's step and pictures on this
  machine before the run (`results/manager/crowd_calibrate_quiet.csv`, three alternating pairs, 4 ms on one thread):
  calibrated, 3.77, 4.29 and 3.88 ms of CPU per tick (94 to 107% of the budget); with the defaults, 6.07 to 6.24 ms. An
  engine should calibrate once at start (here it added 1.2 s to the start for the benchmark's four effects) or set costs measured on its target
  hardware with `nvfx_manager_set_cost`; the defaults are one core of the project's machine on a fast day.
- **On several threads the wall time now spreads**: 6.5 ms per tick on 2 threads for 13.1 ms of CPU and 7.3 ms on 4
  for 26.7 ms, where the shared machine's other jobs had held two cores.

## 8. Limits

- **The coarse step is the floor.** Every simulation steps 30 times a second at any size and rate (0.32 ms a step on
  one core): at 4 ms per 60 Hz tick a core runs about 15 to 18 simulations, however small. Beyond that the manager can
  only share: copies, and proxies, which are copies of a few runs at snapped controls. A battlefield of a hundred
  distinct-looking effects needs more CPU, a cheaper stepper (fewer channels; study K's int8 path, now the default,
  takes a 64 px step from 0.36 to 0.20 ms with AVX-512 VNNI, 0.07 ms of it int8's and the rest the AVX-512 build's,
  and the 4 ms crowd's CPU from 4.1 to 2.8 ms per tick with the same plan; docs/REPORT.md §13), or a learned stepper
  that takes longer steps (not trained).
- **Proxies are copies.** Their copies are offset in time, but near one another they repeat; their controls are
  snapped (2.6 to 3.2 dB of mean-frame PSNR lost against another run at the instance's own controls, §5); magic's
  proxies are visibly worse (its spin at 7.5 Hz). One-shot effects have none.
- **Motion-compensated interpolation** follows the gas, not the picture: it loses on flames, plumes and the portal and
  costs about a keyframe picture; only the expanding explosion gains. A warp along the motion of the picture itself
  (estimated between keyframes) was not tried.
- **Rates save little at small sizes** (§6); the update rate is mostly a lever for large effects.
- **Size changes** start the detail layer from a resampled field (the coarse state is exact): a simulation that grows
  looks soft for a moment, as fire does after a cold start.
- **The cost model is fixed** (deterministic plans); on a slower or busier machine the jobs take longer than planned
  unless `nvfx_manager_calibrate` measured that machine. There is no feedback from measured time.
- **Memory** is the pools': a simulation slot holds two runners (the shard on screen and the next), though the second
  is used only during the half second of a shard change (and a warm-up); sharing it between simulations would save
  about a third, not done.
- **Not done:** colour (hue, brightness) per instance, the prior against drift, frame models (grid and conv effects;
  `nvfx_manager_add_model` refuses them), GPU-side interpolation (the API returns composed pictures only).

## 9. Reproduce

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_C_COMPILER=gcc-14 && cmake --build build
ctest --test-dir build -R Manager                   # the manager's tests (the trained effects' from $NEURALVFX_DATA)
build/nvfx_alloc_test                               # 0 allocations, the manager included
build/nvfx_lod val --threads 3                      # §5.3: results/manager/lod_val.csv, lod_val_summary.md (12 min, 3 threads)
build/nvfx_lod test --threads 3                     # §5.4, once
build/nvfx_lod sheet --effect fire --size 64 --div 4                 # $NEURALVFX_DATA/scale/figures
tools/manager/time_crowd.sh build /tmp/crowd 3 20   # §7: every configuration, 3 interleaved rounds, pinned cores
build/nvfx_crowd --budget 4 --video $NEURALVFX_DATA/scale/crowd/crowd_b4.mp4 --sheet $NEURALVFX_DATA/scale/crowd
```
