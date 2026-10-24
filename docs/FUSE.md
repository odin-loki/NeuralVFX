# NeuralVFX in FUSE: a sample integration

Status: **current** (21 October 2026). Audience: FUSE's engine programmers. The code is in `engines/fuse/`: the
interface FUSE implements (`fuse_engine.hpp`), the adapter that plays NeuralVFX through it (`nvfx_fuse.hpp`,
`nvfx_fuse.cpp`, C++23), and a headless demo on a stand-in FUSE (`demo/`) with its test. Background:
[INTEGRATION.md](INTEGRATION.md) (which API when, threads, textures, packs), [ENGINES.md](ENGINES.md) (the C API, the
Godot plugin, lit sprites, volumes), [MANAGER.md](MANAGER.md) (the budget), [LIBRARY.md](LIBRARY.md) (the catalogue).

NeuralVFX is meant to replace FUSE's flipbooks: a flipbook on a particle becomes a learned effect playing on the CPU,
uploaded as a texture when it changes. **FUSE source** lives in the owner's **`Desktop/FUSE`** git repo; this tree
holds the adapter (`engines/fuse/`) against `fuse_engine.hpp`, which FUSE implements over its own systems (§2).
Everything else is NeuralVFX's public C API
through its C++ wrapper. **No call was added to the C API** for this (`NVFX_VERSION` 7 then; it is 8 since the effect
manager draws large sprites at half size, which the adapter exposes, §5; 10 since the volume API has a job hook, §9,
which the adapter does not use yet).

![The demo at 7 s: torches, a portal, smoke columns and an explosion as sprites; the campfire script on a card at the
left; library entries on a stage at the right; the 3D fire in the middle](../videos/fuse_demo.jpg)

*The demo ([videos/fuse_demo.mp4](../videos/fuse_demo.mp4)), drawn by the stand-in's software renderer: every effect is
a quad FUSE would draw, blended premultiplied over the stand-in's sky and ground.*

## 1. What the adapter plays

One object, `nvfx_fuse::System`, made while a level loads, plays four kinds of effect from **one effect pack**:

| kind | NeuralVFX API | in FUSE's world | budget |
|---|---|---|---|
| **sprites**: the pack's learned models (fire, smoke, explosion, magic, ...) spawned and despawned anywhere | the effect manager (`nvfx_manager.h`), its jobs on FUSE's job system | an upright quad facing the camera, one 256 x 256 texture each: what replaces a flipbook on a particle | CPU milliseconds per frame (`budget_ms`), sprites at once (`max_sprites`), simulations per model (`max_sims`) |
| **a scene**: a scene script of the pack, driven by FUSE through its inputs and named rules | `nvfx_scene.h` (`nvfx_scene_create_from_pack`), its engine layers | a card standing in the world showing the scene's effects alone (the colour layer: no sky, no ground) | its script (what it plays), the texture's size |
| **a stage**: entries of the pack's effect library (`campfire`, `steam_vent`, `magic_burst`, ...) spawned and stopped | `nvfx_library.h` (`nvfx_library_open_pack`, a stage) | a second card | rooms: how many of each entry at once |
| **a volume**: the learned 3D fire, stored in the pack | `nvfx_volume.h` | a quad facing the camera with the volume's CPU picture, and optionally its fields as two 3D textures for FUSE's own ray marcher | the picture's size; one volume costs what STUDY_3D.md §8 says (§6) |

What FUSE gets: textures it uploads (premultiplied RGBA8, sRGB; half-float 3D textures for the volume's fields),
quads it draws with one blend mode, and fields it reads for gameplay (`heat_at`: is the player standing in fire?).

## 2. What FUSE implements

`engines/fuse/fuse_engine.hpp`, five small interfaces and a struct of quads. Nothing in them may allocate in a frame.

| interface | what FUSE does | called |
|---|---|---|
| `FrameClock` | `now()`: the frame index, the game time and dt (pauses and slow motion included) | by FUSE, passed to `update` |
| `JobSystem` | `run(jobs, fn, context)`: run `fn(context, j)` for every j on its workers and return when all have; jobs started in index order; `fn` is a plain function pointer | game thread, once per frame |
| `View` | the frame's camera: `width`, `height`, `world_to_screen(p)` (pixels and view depth), `pose()` (eye, forward, up, vertical fov) and `matrices()` (world to view and view to clip, column-major, OpenGL conventions: the camera looks down -z, clip z in [-1, 1]; FUSE converts its own) | game thread, in `update` |
| `GpuUpload` | `create_texture` / `destroy_texture` at load and unload (`Rgba8Srgb`: premultiplied RGBA8 with rgb decoded as sRGB; `R16Float3d`: half floats, a 3D texture); `update(texture, region, pixels, row_bytes)` on the render thread | load; render thread |
| `RenderHooks` | `add(fn, context)` at load, `remove` at unload; FUSE's render thread calls `fn(context, {frame, sink})` once for every frame it draws, in order, after the game thread finished updating it, before it draws the frame's transparent quads | render thread |
| `DrawSink` | `draw(quad)`: copy a quad into FUSE's transparent pass | render thread, in the hook |

A `Quad` is a textured rectangle in the world (a centre and two half-axes), a UV rectangle, an opacity, a view depth
for sorting and `edge`, a soft-edge width (§3.5). FUSE draws every quad the same way:

- **blend** `ONE, ONE_MINUS_SRC_ALPHA` (premultiplied: `dst = src.rgb + dst * (1 - src.a)`), no depth writes, depth
  test on, sorted back to front by `depth` with its other transparents;
- **colour**: textures are created sRGB (`VK_FORMAT_R8G8B8A8_SRGB`, `GL_SRGB8_ALPHA8`): the GPU decodes rgb, not alpha,
  before filtering and blending, which is what NeuralVFX's own compositor does (INTEGRATION.md §4); in a renderer that
  blends in display space, create them UNORM instead;
- **soft edges**: texel times `smoothstep(0, edge, 1 - |s|) * smoothstep(0, edge, 1 - |t|)` with (s, t) the position on
  the quad in [-1, 1]; 0 turns it off.

## 3. Step by step

### 3.1 Build

Add `engines/fuse/nvfx_fuse.cpp` to FUSE (or link the `nvfx_fuse` static library this repository builds), with
`engines/fuse` and `include/` on the include path, C++23 (`std::expected`, `std::span`), and link libnvfx: the shared
library (`libnvfx.so`, `nvfx.dll`; the adapter checks its version against the headers) or the static ones
(`nvfx_pack_api` pulls in the rest). The adapter compiles with `-fno-exceptions -fno-rtti` (as the wrapper does);
errors are returned, never thrown.

### 3.2 Make the level's pack

```sh
nvfx_pack pack level.nvfxp --library library --models $DATA/library/models \
    --add examples/scenes/campfire.nvfxs,fire3d.nvfx3 --threads 2      # scripts and the 3D model ride along as members
nvfx_pack verify level.nvfxp                                           # what a load checks
```

The catalogue's entries need their models (a missing one is an error at pack time, and a pack missing one is refused at
load); scene scripts and the `.nvfx3` are added as members by their file names. On fixed hardware add manager costs
measured there (`--costs target.csv` or `--calibrate` run on the target, INTEGRATION.md §6), so the game does not
calibrate at start.

### 3.3 Load the level (game thread, loading screen)

```cpp
#include "nvfx_fuse.hpp"

const nvfx_fuse::SpriteModel models[] = {{"fire", 8}, {"smoke", 4}, {"explosion", 4}, {"magic", 2}};
const nvfx_stage_room rooms[] = {{"steam_vent", 2, 0}, {"magic_burst", 2, 1}};   // the stage's budget: rooms
nvfx_fuse::Config cfg;
cfg.pack_path = "levels/forest/effects.nvfxp";
cfg.sprites.models = models;
cfg.sprites.max_sprites = 48;
cfg.sprites.budget_ms = 4.f;              // CPU ms per frame for every sprite, summed over the workers
cfg.sprites.job_slots = 3;                // workers that may run the manager's jobs at once
cfg.scene.script = "campfire.nvfxs";      // a member of the pack
cfg.scene.card = {{-3.3f, 0.f, -3.f}, {0.94f, 0.f, 0.34f}, {0.f, 1.f, 0.f}, 0.0085f};  // base, right, up, metres per pixel
cfg.stage.rooms = rooms;
cfg.stage.card = {{3.4f, 0.f, -3.3f}, {0.94f, 0.f, -0.34f}, {0.f, 1.f, 0.f}, 0.009f};
cfg.volume.member = "fire3d.nvfx3";
cfg.volume.position = {0.2f, 0.f, -0.9f};
cfg.volume.scale = 2.4f;                  // metres per volume unit
cfg.volume.fields = true;                 // also its temperature and extinction for FUSE's ray marcher

auto fx = nvfx_fuse::System::create({&jobs, &render_hooks, &gpu}, cfg);
if (!fx) log_error("NeuralVFX: %s", fx.error().message.c_str());   // the level plays on without its effects
nvfx_fuse::System& vfx = **fx;
```

`create` does every allocation and every slow thing, in this order, and stops at the first failure with a message
(`fx.error()`: the status, the text, and the line and column for a script or catalogue):

1. **The library and the headers.** `nvfx_version` must equal the header's `NVFX_VERSION` ("libnvfx is version 7,
   the headers 8"), and the build must have the features the level uses: `NVFX_FEATURE_PACK`, and `_MANAGER`,
   `_SCENE`, `_LIBRARY` or `_VOLUME` for what is configured.
2. **The pack**, read once: its header, length, manifest checksum, every member's two SHA-256 and the entries' needs
   (`effects.nvfxp: truncated: 300000 of its 613865 bytes`). It is freed at the end of `create`: the library, the scene
   and the effects keep what they need.
3. **Sprites**: each model loaded from the pack (must be a rollout effect), the manager made with `threads = 0` (FUSE's
   jobs run its work), its costs from the pack, else measured here (`Costs::pack_or_calibrate`, about 0.25 s a model,
   INTEGRATION.md §6) or the defaults (`Costs::pack_or_default`: plans that do not depend on the machine, for replays
   and tests); one 256 x 256 texture per sprite.
4. **The scene** from the pack on one thread (it runs as one FUSE job), overlap off, its engine layers on; its first
   picture drawn; its texture.
5. **The stage**: the pack's catalogue opened as a library, a stage with the rooms (its learned effects warm up here).
6. **The volume**: its file read from the pack's member, created, given its controls, seed and shards, stepped to
   frame 0 (its warm-up, about 1.7 s on one thread: STUDY_3D.md §8); its picture's texture and the two 3D textures.
7. **The packets** for the render thread (§4), and the render hook registered.

### 3.4 Each frame (game thread)

```cpp
// gameplay first: spawns, despawns, moves, inputs, rules (never during update)
nvfx_fuse::Sprite torch = vfx.spawn({.model = vfx.model("fire"), .position = {1.f, 0.f, 2.f}, .size = 1.1f,
                                     .controls = {0.8f, 0.5f, 0.45f}, .seed = 7, .time_offset = 0.37f});
vfx.despawn(old_torch);
vfx.trigger("blow");                                   // a rule of the scene script, `when 0 as blow`
vfx.set_input("lean", 0.5f + 0.3f * wind);             // an input of the script
vfx.spawn_entry("magic_burst", world_point, 0.8f);     // a library entry on the stage's card
const float volume_controls[3] = {0.7f, 0.5f + 0.25f * wind, 0.6f};
vfx.set_volume_controls(volume_controls);

vfx.update(clock.now(), camera);                       // everything steps on FUSE's jobs; returns when done

if (vfx.heat_at(player.position) > 0.1f) player.burn();   // the scene's and the stage's field buses
```

`update` does, in order: waits only if the render thread is `frames_in_flight - 1` frames behind; plans the volume
(the box's rectangle on screen, a projection cropped to it, the quad facing the camera that will show the picture),
gives the manager the camera and plans its tick (`nvfx_manager_tick_begin`, 0.03 to 0.08 ms); runs **one
`JobSystem::run`** with the volume first (when its 30 fps frame changes: step, picture, fields), then the scene and the
stage (one job each: `nvfx_scene_step` and, when a frame was computed, the colour layer), then up to `job_slots`
manager jobs (`nvfx_manager_run_jobs(slot)`); finishes the tick; reads the sprites' pictures (a one-shot whose picture
says `NVFX_PICTURE_FINISHED` is despawned; copies sharing one picture share one texture); and publishes the frame's
quads and whatever pixels changed (§4).

One-shots (explosions) despawn themselves when they finish (`despawn_when_finished`); handles of despawned sprites are
never valid again (`alive(sprite)`). Spawning when `max_sprites` play returns an empty handle and records why.

### 3.5 The render thread

FUSE calls the hook once per frame it draws. It takes the packet the game thread published for that frame, gives
`GpuUpload::update` only what changed since the texture was last updated (a sprite's top-left `size` x `size`, 32 to
256 px as its level of detail changes; the cards' pictures; the volume's picture and fields), and draws the frame's
quads into the `DrawSink`. Sprite UVs cover their picture's `size` with half a texel in from each edge (the texture
beyond holds older pixels). Sprite quads carry `edge = 0.08` (`SpriteConfig::edge_fade`): plumes that reach the top of
their simulated domain (smoke columns, the explosion's cloud) show a hard line there (MANAGER.md §7), which a soft edge
hides.

### 3.6 Unload

Destroy the `System` on the game thread, after the last `update` and before FUSE's job system, render hooks and GPU
upload go: it removes its hook (after `remove` returns FUSE does not call it again), destroys its textures and frees
everything. FUSE must not draw quads it was given after that.

## 4. Threads

```
game thread   │ gameplay N: spawn, inputs │ update N: plan ─ JobSystem::run ─────────────── publish N │ gameplay N+1 │ update N+1 ...
FUSE workers  │                           │             volume │ scene │ stage │ manager × job_slots │
render thread │ draw N-1: hook (uploads, quads) ...                                     │ draw N: hook ...
```

- **Who calls what.** Every call of `nvfx_fuse::System` is made on the game thread, never during `update`. The work
  runs as FUSE jobs inside `update`, which returns when they have all returned. The render hook runs on FUSE's render
  thread and touches only the packets and `GpuUpload`.
- **Jobs.** The manager's jobs use FUSE's workers directly (`threads = 0`, `job_slots` at once: MANAGER.md §4). A scene,
  a stage and a volume each run as one job: the scene and the stage on one thread (`threads = 1`, overlap off), the
  volume on `VolumeConfig::threads` threads (1: only the job's; above 1 it starts threads of its own inside that job).
  The volume API can run on an engine's job system instead (`nvfx_volume_create_with_jobs`, NVFX_VERSION 10), but the
  adapter does not use it yet: the volume's loops would call `JobSystem::run` from inside the volume's own job, which §2
  forbids (§9). With the defaults one update runs up to 3 + `job_slots` jobs, longest first.
- **Packets.** `frames_in_flight` (default 2) packets carry frames from the game thread to the render thread: frame N's
  quads, and the newest pixels of every texture with a version per texture. A packet is filled only after the render
  thread has drawn the one `frames_in_flight` frames before it; pixels are copied into it only when its copy is older than the newest
  (a new sprite picture is copied into each packet once), and uploaded only when the texture's copy is older. So a
  frame the render thread skips loses nothing, and the manager's pictures (valid until its next tick) are never read
  while it ticks. A hook for a frame `update` never published draws nothing.
- **Determinism.** The same calls at the same frames give the same pictures to the bit on any number of FUSE workers
  and whoever runs the jobs: the manager, scenes, stages and volumes promise it (INTEGRATION.md §7), and the adapter
  adds nothing that depends on timing. The demo's test checks it on the whole sequence (§8). Two conditions: fixed
  manager costs (`Costs::pack_or_default`, or costs in the pack; calibration measures the machine and its load), and
  the same instruction set (`nvfx_set_isa` for lockstep across machines).

## 5. Memory and budgets

Everything is allocated by `create`; `update`, the render hook and every call of §3.4 allocate nothing (frame errors
too: §7). The demo's test counts heap allocations on every thread (the game thread, FUSE's workers, the render thread
and libnvfx's own) over all 600 frames: 0.

**The budgets a level sets.**

| what | knob | what it bounds |
|---|---|---|
| sprites' CPU | `SpriteConfig::budget_ms` | the manager's planned CPU per frame summed over workers (MANAGER.md §2.4); it lowers sizes and rates, shares and proxies to fit |
| sprites at once | `max_sprites` | the manager's instances and the sprite textures |
| simulations per model | `SpriteModel::max_sims` | the manager's pools for that model, the most it plays with runs of their own |
| large sprites at half size | `SpriteConfig::half_px`, `half_share`, `half_upscale`; `SpriteModel::half_px` | sprites at least that large on screen are drawn at the level below theirs (MANAGER.md §2.10; below) |
| library entries | `StageConfig::rooms` | instances of each entry at once (`steal`: a spawn when all play replaces the oldest) |
| a scene | its script | what it plays; the texture is `SceneConfig::width` x `height` |
| the volume | `picture_scale`, `max_picture` | its picture's pixels (it is drawn at a fraction of its size on screen and sampled up) |

**What the demo's level holds** (from `Stats::memory`; the demo's configuration, §8):

| part | MB | from |
|---|---:|---|
| the sprites' four models | 1.5 | `nvfx_effect_info.resident_bytes` |
| the manager (four models, `max_sims` 4, 3, 3, 2; 24 instances) | 65.5 | `nvfx_manager_stats.memory_bytes`: the pools, rings and threads' scratch (MANAGER.md §2.8) |
| the scene (campfire at 480 x 270) and the stage (640 x 360, five rooms) | 17.8 | `nvfx_scene_info.scratch_bytes` and their pictures |
| the volume | 34.1 | `nvfx_volume_scratch_bytes` (31 MB), its model, its picture and fields |
| two packets | 18.5 | 24 sprite pictures of 256 KB, the cards' and the volume's pictures and fields, twice |
| **on the CPU** | **137.5** | |
| textures on FUSE's side | 9.3 | 24 x 256 KB, the two cards, the volume's picture and its two 3D textures (0.75 MB each) |

The manager's pools dominate: at its defaults (`max_sims` 16) a looping model takes about 29 MB and a one-shot 21 MB
(MANAGER.md §2.8); size `max_sims` to what the budget can run (15 to 18 simulations per core at 4 ms, MANAGER.md §4).
Packets cost `frames_in_flight` x (`max_sprites` x 256 KB + the cards' and the volume's pixels).

**Large sprites at half size** (MANAGER.md §2.10, REPORT.md §22). The manager draws a sprite that is large on screen at
the level below its level: a 256 px picture at 128. How FUSE turns it on or off, at load:

```cpp
cfg.sprites.half_px = -1.f;                          // the default: the manager's per-effect default (MANAGER.md §6)
cfg.sprites.half_px = 0.f;                           // off (with half_share 0)
cfg.sprites.half_px = 192.f;                         // or its own threshold, in pixels on FUSE's view
cfg.sprites.half_share = 0.25f;                      // or a quarter of the view's height (either suffices)
const nvfx_fuse::SpriteModel models[] = {{"fire", 8}, {"magic", 2, 0, 0.f}};   // per model: 0 never, > 0 its own
```

`SpriteConfig::half_px` below 0 passes the manager's default through (NVFX_MANAGER_VERSION 3, REPORT.md §22.11): by
the model's name, fire, smoke, steam and explosions at least 256 px on FUSE's view are drawn at 128, magic and every
effect not measured at full size; so the demo's fire, smoke and explosions take it and its magic does not. 0 is off for
every sprite model, a positive value one threshold for all of them; `SpriteModel::half_px` still comes first (0 opts a
model out, a positive value opts it in at its own threshold).

`SpriteConfig::half_upscale` is `NVFX_UPSCALE_OFF` by default: the picture uploaded is the half-size drawing, and the
quad's bilinear sampling (FUSE's, in linear light after the sRGB decode) is the 2x; the render hook already uploads only
a picture's `size` x `size` and sets the quad's UVs to it, so nothing else changes, the upload is a quarter and the 2x
costs no CPU. `NVFX_UPSCALE_BILINEAR` has the manager do the 2x on FUSE's jobs instead (the picture at its level's size).
The manager never uses the learned 2x. Turning it on or off changes the plans, so a replay or a lockstep test compares
runs made with the same setting.

## 6. What it costs

Measured numbers from the docs (one thread unless said; the rows that name round 7 were re-timed on a quiet machine, the
others come from the shared 4-core machine and their sources say how provisional):

| part | cost | source |
|---|---|---|
| loading a pack | the 11-model library (614 kB, fast coder): 124 ms on one thread, 52 ms to open on 3 | INTEGRATION.md §5 |
| a stage's creation (warm-ups) | four effects: 130 ms on 2 threads | INTEGRATION.md §2 |
| calibrating a model for the manager | about 0.25 s | INTEGRATION.md §6 |
| a sprite simulation | per frame at 30 Hz, step and picture: 0.39 ms at 32 px, 0.47 at 64, 0.78 at 128, 2.35 at 256 (one AVX2 core, float: the cost model's defaults); half that per 60 Hz frame | MANAGER.md §6 |
| the manager's floor | every simulation steps 30 times a second (0.32 ms a step): 15 to 18 simulations per core within 4 ms at 60 Hz; beyond that it shares | MANAGER.md §1, §8 |
| a crowd | 165 effects in 4 ms per 60 Hz frame on one core, calibrated (3.8 to 4.3 ms measured); 6.7 ms uncalibrated on a machine 1.8 times slower than the defaults assume | MANAGER.md §7.1 |
| planning a tick | 0.03 to 0.08 ms on the game thread | MANAGER.md §1 |
| one learned effect alone | 0.63 ms per 128 px frame at int8 with AVX-512 VNNI, the mean over the 33 shipped models (0.46 to 1.12; 0.37 at 64 px, 1.51 at 256); round 7 | REPORT.md §13.11 |
| a stage | about 2.5 ms to draw at 320 x 240 (30 ms at 1120 x 840), plus 3 ms per 128 px fire flame, 1 ms per 64 px one, 15 ms per 256 px smoke or steam tile | LIBRARY.md §7 |
| a later spawn of a learned entry | its warm-up in that frame: 16 to 20 ms for a 128 px fire or steam module | LIBRARY.md §7 |
| the engine layers | 3.5 to 6 ms more per picture at 960 x 540; reading all four about 7 ms (the adapter reads one, the colour, as RGBA8) | ENGINES.md §11 |
| a scene | the fireball at 1280 x 720: 12.5 ms per frame on 4 threads, 23.7 on 2; round 7: 15.0 on 4 threads (12.0 at best), 31 at 1920 x 1080; the round scenes 10 to 19 ms on 2 threads | ENGINES.md §7, REPORT.md §13.11 |
| the volume | study N2's fire, which the demo plays: a step 21 ms (9 on 4 threads), a 256 x 256 picture 56 ms (14), a start (its warm-up) 0.55 s (0.22); a quiet machine, round 7 of the quiet re-timing | STUDY_3D.md §14.12 |
| uploads | a sprite at most 256 KB, at most 30 times a second; a card w x h x 4 per new picture; the volume's fields 2 x 0.75 MB per volume frame | INTEGRATION.md §4, this adapter |

What that means for a 60 Hz FUSE frame:

- **Sprites are the real-time path.** At 4 ms on one worker the manager plays about 15 to 18 simulations and shows
  far more through sharing and proxies; this is what replaces flipbooks at scale.
- **A stage with a few small entries fits on one worker**: a 640 x 360 stage with a 128 px fire and a steam module is
  about 15 ms a picture on one thread by LIBRARY.md §7's numbers, half that per 60 Hz frame.
- **A scene costs what its script plays**: the campfire script's sky is four 256 px smoke tiles, about 60 ms a picture
  on one thread by the same numbers, so as written it is a showcase, not a budget scene. A scene for a game is written
  small, or given threads of its own (`threads` in the C API; the adapter uses 1).
- **The volume is not real time on the CPU**: a step and a picture are about 77 ms every 30 fps frame on one thread and
  23 ms on four (STUDY_3D.md §14.12, a quiet machine; 120 ms on one busy thread before the detail layer ran on
  vectors). It suits one hero effect on a machine with cores to spare, a cut scene, or baking; FUSE's GPU should
  ray-march its fields (`fields = true`) rather than show the CPU picture, which saves the picture's 56 ms (14 on four
  threads).

The demo itself ran at about 4 frames a second on the shared machine (load average 6 to 17 while four other agents built
and trained) and at 14 on three workers (11 on the game thread alone) on a quiet one (round 7 of the quiet re-timing,
§8), most of it the volume's job.

## 7. Errors

- **At load**: `create` returns `std::unexpected(nvfx::Error)`: the status and the C API's message (with the file,
  line and column for a script or catalogue), or the adapter's own ("sprites: the pack has no model 'fyre'", "FUSE
  could not create its texture"). Nothing is left behind. A FUSE level should play on without its effects.
- **In a frame**: the C calls return a status; a failed part skips its work for that frame (the rest goes on), and the
  error is counted and kept in fixed storage, never allocated: `stats().errors` and `last_error()` (the status, the
  frame and a line such as `scene: trigger blwo: invalid argument`). FUSE logs it at its own pace.
- **Calls**: `spawn` returns an empty handle and records why (no room, an unknown model); `despawn`, `move` and
  `set_controls` on a stale handle return false; `trigger` returns false for a rule the script does not name (or a landing rule); a rule with no firings left is simply not fired.

## 8. The demo and its test

```sh
cmake --build build --target nvfx_fuse_demo nvfx_pack
ctest --test-dir build -R Fuse                     # the self-test (no models needed), and the demo's --check (skips without them)
build/engines/fuse/nvfx_fuse_demo --pack build/engines/fuse/fuse_demo.nvfxp --out OUT --video fuse_demo.mp4
```

**The stand-in FUSE** (`demo/standin.hpp`): a thread pool of N workers plus the calling thread (`JobSystem`), a fixed
60 Hz clock, a look-at camera, and a software renderer with a render thread of its own (`GpuUpload`, `RenderHooks`):
textures in CPU memory, FUSE's pipeline (frame N drawn while N + 1 updates, at most one frame queued), the hooks called,
quads sorted back to front and drawn into a linear-light float framebuffer with premultiplied blending, texels decoded
as sRGB and filtered bilinearly, soft edges, over a sky and a ground grid of its own, then encoded to sRGB bytes. It is
the stand-in, not a renderer: one thread, no depth buffer, no lighting.

**The level** (`demo/fuse_demo.cpp`): a pack of the library's 27 models and 22 catalogue files, the campfire script and
study N2's 3D fire (2.2 MB, 51 members); sprites of four models (24 at most, 5 ms on 2 job slots, the default costs),
the campfire script on a card, a stage with five rooms, the volume with its picture and fields.

**The sequence** (10 s at 60 Hz; the HUD names each event):

| time | what the game does |
|---|---|
| 0 | six torches (fire sprites, one simulation shown 0.37 s apart along two rows), two smoke columns (their own runs) |
| 0.3 to 1 s | the stage: a brazier, a steam vent, an electric arc |
| 1.5 s | the scene's rule `blow`: a gust leans the campfire |
| 2 s | an explosion sprite, despawned when it finishes |
| 3 s | the scene's rule `light_log`: the log beside the campfire catches |
| 3.6 s | the stage: a magic burst (a one-shot entry) |
| 4.2 s | a magic portal sprite |
| 5 to 7.4 s | the game's wind: the scene's inputs `lean` and `breeze`, the smoke columns' wind control, the volume's wind control |
| 5.9 s | the stage: a lightning strike |
| 6.3 s | a second explosion |
| 7 s | the two far torches despawned |
| 7.6 s | the scene's rule `douse`: the campfire is put out |
| 8.4 s | the steam vent stopped |
| 8.8 s | the scene's rule `relight`, its input `fuel` set to 1 |

The camera dollies in slowly throughout, so the manager's levels of detail change with distance.

**The test** (`neuralfx.FuseDemoPack`, then `neuralfx.FuseDemoHeadless`): `nvfx_pack` makes the pack (exit 77, a
skip, when a model or the 3D fire is missing: CI has no models), then `nvfx_fuse_demo --check` plays the whole sequence
twice, on three workers and then on the game thread alone, and checks: no heap allocation in any frame on any thread
(the counter replaces `operator new`, and must have seen libnvfx's allocations at load); **the same 600 frames to the
bit** on both (a hash of each frame drawn); no frame error; the render hook drew every frame; sprites were drawn; the
explosions were despawned; heat on the campfire's card; textures uploaded. Without the pack it exits 77.

**The self-test, without models** (`neuralfx.FuseSelfTestPack`, then `neuralfx.FuseSelfTest`; it runs in CI, in about
2 s): a pack of `demo/selftest/sparks.nvfxl` (three procedural entries copied from the library: no learned model), and
`nvfx_fuse_demo --self-test` on it. First the load errors, each refused with its status and leaving no texture or hook
behind: a missing pack, a damaged one, `frames_in_flight` below 2, an engine without a job system, and a model, a
script, a volume member or a room's entry the pack does not hold (the last two after the stage was made). Then a stage of
the three entries played for 2 s on two workers and on the game thread alone, with spawns, a stop and, in the middle of
a frame, calls that must fail (an unknown entry, a sprite in a level without sprites, a rule without a scene, a stale
handle): no heap allocation in any frame, failed calls included; the same frames to the bit; every frame drawn; the
failed calls kept as frame errors (`spawn: no sprite model 0: invalid argument`).

**What the test found** (21 October 2026: one run of the same `--check` by hand, then `ctest -R Fuse`, which passed
every check), **re-timed on a quiet machine**: round 7 of the quiet re-timing
([results/quiet/README.md](../results/quiet/README.md); three runs of `nvfx_fuse_demo --check --workers 3`, nothing else
running; the median of the runs and their range, [fuse_runs.csv](../results/quiet/round7/fuse_runs.csv)). The first
run's times were wall clock on the shared 4-core machine at a load average of 6 to 17 (four other agents building and
training) and are kept in the last column (every time there differs from round 7's by more than 15%). §6 has the costs
of the parts:

| | 3 workers and the game thread | the game thread alone | the busy machine: 3 workers / alone |
|---|---:|---:|---:|
| frames, and the time they took | 600 in 41.8 s (39.7 to 42.3) | 600 in 55.3 s (51.7 to 55.3) | 168 s / 138 s |
| heap allocations in the frames, every thread | **0** | **0** | 0 / 0 |
| frames that differ between the two plays of a run | **0 of 600** | | 0 of 600 |
| frame errors | 0 | 0 | 0 / 0 |
| `update`, mean and worst (the worst is the volume's shard change at 7.5 s) | 69.1 and 621 ms (65.5 to 69.6; 616 to 639) | 91.9 and 647 ms (85.9 to 91.9; 642 to 763) | 276 and 2,981 / 228 and 2,874 ms |
| the volume's job per volume frame: step, picture, fields | 26.9, 38.7 and 50.7 ms | 26.5, 38.9 and 51.4 ms | 272, 105 and 111 / 190, 85 and 86 ms |
| the scene's job per picture (the campfire script) | 32.7 ms | 30.0 ms | 122 / 37 ms |
| the stage's job per picture | 14.8 ms | 11.7 ms | 50 / 14 ms |
| the manager's jobs per tick, against its plan (budget 5 ms) | 4.07 against 4.07 ms | 3.54 against 4.07 ms | 7.0 / 4.8 against 4.1 ms |
| uploads | 1.88 MB a frame | the same | 1.87 MB a frame |
| `create` | 1.1 s (1.06 to 1.31) | 1.0 s (1.01 to 1.24) | 3.1 s / 2.9 s |

Every run's worst update came at 7.50 s, where the volume's next shard starts (§9). The volume's step is a tenth of the
busy run's: the detail layer runs on vectors since (REPORT.md §19), besides the quiet.

On the busy machine each job took longer on three workers than on the game thread alone: it had no free core, so more
threads only competed. On the quiet one the three workers play the 600 frames in three quarters of the time the game
thread alone takes (14 frames a second against 11), though a job itself is no faster on them (the scene's and the
stage's a little slower). The manager's jobs were at or under its plan on both; its default costs are one AVX2 core on a
fast day (MANAGER.md §7.1): a game calibrates at load, or ships costs measured on its target in the pack.

**Checks** (21 October 2026, the branch on main at c846c80): the full build with warnings as errors; the full ctest,
492 of 492 (main's 488 and the four here), none skipped; `nvfx_alloc_test`; `nvfx_gallery --check --require-models`
(213 of 213 played); the Godot test (`engines/godot/test.sh`, 0 failures in both parts).

## 9. Honest limits

- **FUSE lives on the Desktop.** Implement `fuse_engine.hpp` in `Desktop/FUSE`; keep the adapter here in sync. The
  stand-in demo approximates job system, render thread and texture uploads (for example, creating textures
  on the render thread only). The stand-in is a test harness, not a renderer.
- **The volume is slow on the CPU** (§6), and a shard change starts the next shard's rollout in one frame: every 8 s a
  step costs a start, about 1.7 s on one thread (STUDY_3D.md §8; the demo's worst frame is that one). The volume API
  has a job hook since 24 October 2026 (`nvfx_volume_create_with_jobs(fx, run, context, &vol)`, NVFX_VERSION 10,
  ENGINES.md §12): every parallel loop of a step, a read or a picture becomes one fork-join `run(context, jobs, job,
  job_context)` on the engine, the same frames to the bit (`VolumeApi.SameFieldsAndPicturesOnAnEnginesJobSystem`,
  `nvfx_alloc_test`: no allocation on that path). **This adapter does not use it yet**: it steps the volume inside one
  FUSE job, from which `JobSystem::run` may not be called (§2: game thread only; the stand-in's `ThreadPool::run` is not
  re-entrant, a nested call from a worker would corrupt the run in progress). The adapter makes four volume calls in
  all: `nvfx_volume_step_to` at load (the warm-up, frame 0) and, in its job, `nvfx_volume_step_to`,
  `nvfx_volume_render` and `nvfx_volume_read`; every parallel loop of those (a start's warm-up steps, a step, a
  crossfade, a picture's prepare and rows, a field's slices) goes through the volume's one pool, which the hook
  replaces whole, so the library side needs nothing more. Wiring it means either stepping the volume on the game thread
  before `JobSystem::run` (its loops then fan out over FUSE's workers, but the cards and the manager wait for it) or a
  `JobSystem` that allows a nested `run` from a worker: the creation call changes (`nvfx_volume_create_with_jobs` in
  place of `nvfx_volume_create`, `VolumeConfig::threads` then unused) and the volume leaves the job list; the hook also
  does not spread a start over frames (a step is still one call). Play one continuous rollout (`shard_seconds = 0`) to
  avoid the start where the model holds (study N2's fire held 30 s, STUDY_3D.md §14), or bake.
- **The volume's quad is an impostor**: its CPU picture is drawn for the camera at the frame its volume frame changed
  (30 Hz) and placed in the world facing that camera, so between pictures a moving camera sees it slightly off, and FUSE's
  geometry does not cut it (no depth inside the box). When the camera is inside or beside the box (a corner behind the
  camera) there is no picture. FUSE's own ray marcher over the fields has none of these.
- **Cards are flat.** A scene or a stage is 2D: its effects are drawn on a card in the world, as a flipbook would be,
  lit by the scene's own lights (the colour layer), not FUSE's; FUSE could light them with the albedo, emission and
  normal layers (ENGINES.md §11), which this adapter does not read. A card shows the colour layer only, but
  `nvfx_scene_render_layer` draws the scene's whole picture too, so a card costs its picture plus the layers. Gameplay
  reads its fields only on the card's plane (`heat_at`, within half a unit of it).
- **Scenes and stages run on one thread each**, as one job; the C API can give them threads of their own (overlap
  included), which this adapter does not use.
- **Sprites**: upright quads facing the camera around the world's y axis (fire and smoke stay upright), at most 256 px
  pictures stretched over the quad, sorted by their centre's depth; a texture per sprite (an atlas would save memory
  and draw calls); soft edges hide, but do not fix, plumes that reach their domain's top; the manager's limits are
  MANAGER.md §8's (proxies are copies, the coarse step is the floor, no colour per instance).
- **Copies**: the packets copy every new picture once per packet (about 1.9 MB a frame in the demo, most of it the
  volume's fields and the cards) so that the manager can tick while the render thread uploads; an engine that uploads
  on the game thread could skip them.
- **The render hook draws what was published**: a frame FUSE draws twice, or one `update` never published, draws
  nothing new; a headless server that never draws would block `update` after `frames_in_flight` frames (do not create
  the adapter there: use the C API for the fields).
- **Not measured here.** The demo's timings were taken on a shared machine at a load of 8 to 17 and are not costs; §6
  quotes the measured ones.
