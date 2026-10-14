# Putting NeuralVFX in your engine

Status: **current**. Audience: engine programmers. What to call, in what order, on which thread, and what to measure.
The code is the C++ wrapper `include/neuralfx/nvfx.hpp` (header-only, C++23) over the C API; every call has a C
twin of the same name (`stage.spawn(...)` is `nvfx_stage_spawn(stage, ...)`). The whole flow runs in
`examples/pack_host.cpp` (`nvfx_pack_host PACK --out sheet.png`). Background: [ENGINES.md](ENGINES.md) (the C API,
Unreal, Unity, Godot), [LIBRARY.md](LIBRARY.md) (the 70 effects), [MANAGER.md](MANAGER.md) (the budget).

## 1. Which API

| you want | API | what you get |
|---|---|---|
| one effect as a sprite (a torch, a fire on a billboard) | `nvfx.h`: `Effect`, `Instance` | a premultiplied RGBA8 sprite per `render(t)`; you place and blend it |
| a composed set piece from a script (the fireball) | `nvfx_scene.h`: `Scene` | an opaque picture (the scene's own sky and ground) and a field bus (heat, soot, velocity) for gameplay |
| named, ready-made effects spawned at run time (campfire, grenade, portal) | `nvfx_library.h`: `Library`, `Stage` | a stage: a scene with rooms; spawn, move, set parameters, stop; same picture and fields as a scene |
| many rollout effects (tens to hundreds) within a CPU budget | `nvfx_manager.h`: `Manager` | one premultiplied sprite per instance, levels of detail chosen per tick to fit the budget |
| one file with the models and the catalogue | `nvfx_pack.h`: `Pack` | the models for all of the above, checked at load (§5) |

A stage or scene draws a finished picture (it composites its effects, particles and light itself); `Instance` and
`Manager` give sprites that your renderer composites. The library's entries exist only as stages.

## 2. Start

```cpp
#include <neuralfx/nvfx.hpp>

// 1. The shared library is the one the headers describe (§8).
if (!nvfx::version().matches_headers()) fail("libnvfx does not match its headers");

// 2. One pack: every member's SHA-256 checked, every model decoded (§5).
auto pack = nvfx::Pack::open("effects.nvfxp", /*threads*/ 2);
if (!pack) fail(pack.error().message);          // "effects.nvfxp: member fire.nvfx is damaged (...)"

// 3a. The effect library, from the pack's catalogue; a stage with room for what the level uses.
auto lib = nvfx::Library::open_pack(*pack);     // keeps what it needs: the pack may be freed after step 3
nvfx::StageRooms rooms = {{"campfire", 4, 0}, {"grenade", 8, 1}};   // 8 grenades, the oldest stolen
auto sd = nvfx::stage_desc();                   // 960 x 540, 30 fps, 2 threads
sd.rooms = rooms.data();
sd.n_rooms = int(rooms.size());
auto stage = nvfx::Stage::create(*lib, sd);     // loads and warms up the models: all its memory, now
if (!stage) fail(stage.error().message);        // a catalogue error: "fire.nvfxl:12:7: ..." (error().line, .column)

// 3b. Models for the manager or for instances: an Effect of its own (it does not need the pack afterwards).
auto fire = pack->load_effect("fire");          // must outlive the manager it is given to

// 3c. A scene script stored in the pack, its effects from the pack.
auto desc = nvfx::scene_desc();
desc.script_path = "fireball.nvfxs";            // a member of the pack (else a file)
auto scene = nvfx::Scene::create_from_pack(*pack, desc);
```

Creation is where everything is allocated and where models warm up: a stage of four effects took 130 ms on two threads
of the shared machine (`nvfx_pack_host`); the fireball scene about 0.2 s (ENGINES.md §7). Do it while the level
loads.

## 3. The frame loop and threads

**Stages and scenes** run on threads of their own (`desc.threads`, the calling thread included; 1: everything on the
caller). Call them from one thread at a time, in this order:

```cpp
auto s = stage->scene();                        // nvfx::SceneView: step, render, sample
const int computed = *s.step(dt);               // the scene's clock (30 fps): 0, 1 or 2 frames at 60 Hz
if (s.sample(player.x, player.y)->heat > 0.1f) player.burn();       // fields of the frame computed
stage->set_param(fx, "intensity", 0.4f);        // inputs act from the next frame computed
if (computed > 0) { s.render(rgba); upload(rgba); }                 // std::span<std::uint8_t>, width * height * 4
```

With `overlap` (the default with two threads or more) the picture is drawn on a thread of its own while the next
frame's state is computed: the same pictures, faster, and settings act one frame later (ENGINES.md §7).

**Instances** (`nvfx_render`) are single-threaded and touch only their own memory: run each on any worker of your job
system, one thread per instance at a time. Share an `Effect` between threads freely.

```cpp
job_system.parallel_for(instances, [&](Sprite& sp) { sp.inst.render(time * sp.speed, sp.pixels); });
```

**The manager** runs its jobs on its own threads (`desc.threads`), or on yours with `threads = 0`:

```cpp
auto md = nvfx::manager_desc();
md.threads = 0;                                 // the engine's job system runs the work
md.job_slots = 4;                               // at most 4 of its workers at once
md.budget_ms = 4.f;                             // CPU ms per tick, summed over threads
auto mgr = nvfx::Manager::create(md);
const int model = *mgr->add_model(*fire);
// each frame, between spawns and setters:
const int jobs = *mgr->tick_begin(dt);          // planning, on the calling thread (0.03 to 0.08 ms)
job_system.run_on(std::min(jobs, 4), [&](int slot) { mgr->run_jobs(slot); });   // slots 0 .. job_slots - 1
mgr->tick_end();
for (auto& e : effects) {
  const auto pic = *mgr->get_picture(e.handle);  // valid until the next tick
  if (pic.rgba && pic.is_new()) upload(e.texture, pic.pixels(), pic.size, pic.stride);
}
```

Spawns, stops and setters go before `tick_begin` or after `tick_end`. A frame allocates nothing in any of these
(`nvfx_alloc_test`; through the wrapper, `Wrapper.AFrameAllocatesNothing`), and errors are returned, never thrown.

## 4. Textures

| source | pixels | size | a new picture |
|---|---|---|---|
| `Instance::render` | premultiplied RGBA8, rows top to bottom | `size` x `size`: rollout effects any multiple of 32 from 32 to 1024 | when `floor(t * info.fps)` changes (30 fps for the eleven models) |
| `Manager::get_picture` | premultiplied RGBA8, rows `stride` apart | 32, 64, 128 or 256: **changes** with the level of detail | `NVFX_PICTURE_NEW` (at most 30 a second) |
| `Scene::render`, a stage's scene | RGBA8, alpha 255 (opaque) | `desc.width` x `desc.height` (a stage: 960 x 540 by default) | `step` computed a frame (30 fps) |

- **Blend sprites premultiplied**: `dst = src.rgb + dst * (1 - src.a)` (source ONE, destination ONE_MINUS_SRC_ALPHA).
  One mode covers fire (colour above alpha: emission) and smoke. Stretch the sprite over its quad; the manager's
  pictures are 32 to 256 px whatever the quad's size.
- **Colour space.** The RGB values are display-referred: the effects were trained on sprites drawn as display values,
  and the runtime's own compositor (`src/compose/compose.cpp`, `Gamma`) reads them as gamma 2.2 (`pow(v / 255, 2.2)`,
  alpha left linear), composites premultiplied in linear light, and writes scene pictures back with `1 / 2.2`. In a
  linear-light renderer, create sprite and scene textures as sRGB (`R8G8B8A8_UNORM_SRGB`, `VK_FORMAT_R8G8B8A8_SRGB`,
  `GL_SRGB8_ALPHA8`): the GPU decodes RGB, not alpha, before filtering and blending, as the compositor does (the sRGB
  curve and 2.2 differ only near black). In a renderer that blends in display space, upload them as UNORM.
- **Uploads.** At 60 Hz a sprite or scene changes every other frame: upload only then. Copies of a manager instance
  without a time offset share one picture (the same pointer): upload it once. A manager sprite's size changes with its
  level: keep a 256 x 256 texture and update its top-left `size` x `size`, scaling the UVs. A 720p scene picture is
  3.7 MB a frame.

## 5. Effect packs

One file, `.nvfxp`, with everything a set of effects needs: the `.nvfx` models, the catalogue (`library/*.nvfxl`),
optionally scene scripts and licence or credit text, and a manifest with the format, a SHA-256 per member, which
catalogue entries need which models, and optional default costs per model for the manager.

```sh
nvfx_pack pack effects.nvfxp --library library --models $DATA/library/models --licence CREDITS.txt
nvfx_pack list effects.nvfxp          # members, sizes, entries by the models they need, costs
nvfx_pack verify effects.nvfxp        # what a load checks, and the library opened and every model loaded
nvfx_pack extract effects.nvfxp out/  # every member back, byte for byte
```

`pack` takes the models the catalogue's entries need from `--models` (an entry without its model is an error), checks
that each loads, codes each member, decodes it again and compares before writing. `--add FILES` adds scene scripts or
other files; `--costs CSV` (rows `model,size,step_ms,render_ms`) or `--calibrate` (measured on the machine that packs)
records manager costs; `--coder` picks the coder (below).

**The format.** A 96-byte header (`NVFXPACK`, the format, the sizes of the manifest and the data, the file's length,
the manifest's SHA-256), a text manifest, then the members' bytes:

```
nvfx-pack 1
member cloud.nvfx model cm 149287 48311 0 b1a886dd...e221c8 9d20eeed...00816c      NAME KIND METHOD BYTES STORED OFFSET
member fire.nvfx model cm 42895 25787 173816 b0f7ad32...61c827 b583782a...862931       SHA256 STORED_SHA256
member fire.nvfxl catalogue cm 15930 5579 583640 9442165d...f824a8 89b7cce7...5d39a9
needs campfire fire.nvfx                                                              ENTRY MODELS...
needs grenade explosion.nvfx
needs lantern
cost fire.nvfx 64 0.32 0.11                                                           MODEL SIZE STEP_MS RENDER_MS
```

Why so: one file to ship and one read at load; a text manifest a person can read (`list --manifest`) and a later
version can extend (readers ignore records they do not know; the format number changes only for what old readers would
misread); SHA-256 of each member as stored (checked before it is decoded) and as decoded (what `extract` and an audit
compare); the entries' needs recorded so that a pack missing a model is refused at load, not when an effect is spawned.
Members are 8-byte aligned. Code: `src/pack/` (format, reader, writer, C API), the tool in `tools/pack_files.cpp`.

**Loading.** `Pack::open(path)` or `Pack::open_memory(bytes)` checks the header, the length (a truncated file is
refused), the manifest's checksum, every member's two checksums and the entries' needs, and decodes every member;
errors name what is wrong (`effects.nvfxp: truncated: 300000 of its 613865 bytes`, `entry 'campfire' needs smoke.nvfx,
which the pack does not hold`). `Library::open_pack` then checks the catalogue itself against the pack. A pack is
read-only once open and may be shared between threads; the members' decoded bytes stay in memory (1.5 MB for the full
library) while the pack or a library from it lives.

**The coder.** Members are coded by the project's lossless context-mixing coder (`include/neuralfx/cm.hpp`) when that
saves at least 5% and 256 bytes, else stored. Which of its models to use was chosen by measurement, with the rule
fixed before measuring: *the smallest pack whose load of the full library on one thread (open with every check, the
library opened, all eleven models loaded) takes at most 250 ms; among packs within 2% of that size, the fastest.* The
full library: the 11 models and 11 catalogue files (70 entries), 1,538,529 bytes as loose files
(`nvfx_pack measure --reps 9`, the median of 9 interleaved rounds on the shared 4-core machine at a load average of 5 to
6: `results/pack/pack_measure.csv`, per member `pack_members.csv`; a first run of 5 rounds at a load of 6 to 7 gave the
same order and 135 ms for the fast pack):

| | bytes | of the loose files | open, 1 thread | open, 3 threads | library + 11 models | load in all, 1 thread |
|---|---:|---:|---:|---:|---:|---:|
| loose files (library from the folder, models from files) | 1,538,529 | 1 | - | - | 38 ms | **38 ms** |
| pack, stored | 1,544,591 | 1.004 | 7 ms | 4 ms | 32 ms | 41 ms |
| **pack, fast** (the default) | **613,865** | **0.399** | 93 ms | 52 ms | 31 ms | **124 ms** |
| pack, light | 519,872 | 0.338 | 518 ms | 344 ms | 32 ms | 558 ms |
| pack, full | 507,096 | 0.330 | 2,683 ms | 1,775 ms | 31 ms | 2,714 ms |

By the rule the default is the **fast** model (a LOCO-I-like residual code with LZ tokens): 614 kB, 40% of the loose
files, for about 0.09 s more at start (half that on 3 threads). The light model would save another 94 kB for 5.6 times
the decoding; the full model 107 kB for 29 times. Models shrink to 20 to 77% of their bytes (the large start-point
files most), the catalogue's text to 35 to 61%. `--coder store` gives a pack as fast to load as the loose files, at
their size; `light` suits a download where 0.5 s at start does not matter.

**Same frames.** Unpacking reproduces every member byte for byte, so a pack plays what the folder plays: the gallery's
frozen keyframes of eleven entries, drawn from the pack, have their SHA-256 in `results/library/frozen.csv`
(`ctest -R PackLibraryKeyframesFrozen`), and stages from a pack and from the folder give the same frames
(`Pack.TheFullLibraryPlaysTheSameFramesFromAPack`).

## 6. The manager's budget: calibrate at start

The manager plans from a cost model, never from measured time. Its default costs are one AVX2 core of the project's
machine on a fast day: **on a machine 1.8 times slower the 4 ms budget used 6.7 ms (1.7 times the budget); calibrated,
3.8 to 4.3 ms** ([MANAGER.md](MANAGER.md) §7.1). So measure the machine once per model at start:

```cpp
for (int m : models) mgr->calibrate(m);         // about 0.25 s per model on the shared machine; do it while loading
```

On fixed hardware (a console), measure there once and ship the costs in the pack (`nvfx_pack pack ... --costs
target.csv`, or `--calibrate` run on the target); then `mgr->set_costs_from(*pack, model, "fire")` sets them without
measuring. Check the result in play: `get_stats()->jobs_ms` against `planned_ms` and `budget_ms`.

## 7. Determinism and memory

- **Determinism.** The same calls at the same times give the same pictures, to the bit, on any number of threads and
  whoever runs the jobs: instances, scenes, stages (overlapped or not) and the manager. Across machines, on the same
  instruction set: the AVX2, AVX-512 and baseline code differ in the last bits (`nvfx_set_isa` forces one, for
  lockstep). The manager's plan depends
  only on the calls and its cost model: calibration makes it depend on the machine (and its load at that moment);
  fixed costs (`set_cost`, a pack's costs) keep the plans the same everywhere, for replays or lockstep. Packs change
  nothing: the members are the files.
- **Memory**, all of it allocated at creation:

| what | size |
|---|---|
| a model (`Effect`) | `get_info()->resident_bytes`: 0.17 to 0.82 MB for the eleven |
| an `Instance` of a rollout effect | `scratch_bytes()`: 2.35 MB at 128 x 128, 1.5 MB at 64 |
| a scene or stage | `get_info()->scratch_bytes`: the fireball 127 MB at 1280 x 720 |
| the manager | `get_stats()->memory_bytes`: about 29 MB per looping model and 21 MB per one-shot at the default pools, plus the instances' pictures; 44 MB for one fire and six instances in `nvfx_pack_host` |
| a pack | its members decoded: 1.5 MB for the full library |

## 8. Version and ABI

`nvfx_version(&features)` returns the library's `NVFX_VERSION` (3: packs and `nvfx_version`), which changes whenever
a call or a public struct changes: an engine that loads `libnvfx.so` / `nvfx.dll` checks it against the header's
(`nvfx::version().matches_headers()`). `features` says what the build holds (`NVFX_FEATURE_SCENE`, `_LIBRARY`,
`_MANAGER`, `_PACK`, `_SHARED`) and what this CPU runs (`_AVX2`, `_AVX512`, `_VNNI`). The headers compile as C99 and
as C++ (with `-fno-exceptions -fno-rtti` for the wrapper). On 64-bit platforms (x86-64 Linux and Windows alike: no
`long` in any struct) the structs a binding declares have these sizes in bytes, checked at compile time by
`nvfx_abi_test` (`ctest -R Abi`):

| struct | bytes | struct | bytes | struct | bytes |
|---|---:|---|---:|---|---:|
| `nvfx_effect_info` | 80 | `nvfx_library_file` | 16 | `nvfx_manager_desc` | 88 |
| `nvfx_scene_effect` | 16 | `nvfx_library_desc` | 32 | `nvfx_manager_model_desc` | 52 |
| `nvfx_scene_input` | 16 | `nvfx_library_effect` | 80 | `nvfx_spawn_desc` | 72 |
| `nvfx_scene_desc` | 80 | `nvfx_library_param` | 32 | `nvfx_manager_picture` | 56 |
| `nvfx_scene_error` | 1032 | `nvfx_stage_room` | 16 | `nvfx_manager_stats` | 184 |
| `nvfx_scene_info` | 80 | `nvfx_stage_desc` | 88 | `nvfx_pack_desc` | 40 |
| `nvfx_scene_fields` | 16 | `nvfx_stage_fx_info` | 28 | `nvfx_pack_info` | 48 |
| `nvfx_scene_module_info` | 60 | enums, `nvfx_fx`, `nvfx_handle` | 4 | `nvfx_pack_member` | 48 |

Always fill a desc with its `_desc_init` (the wrapper's `scene_desc()`, `stage_desc()`, ...) before setting fields.

## 9. What to measure in your engine

- **Start**: pack open, stage and scene creation (warm-ups), `calibrate`: once per level, on the loading screen.
- **Per frame**: the effects' CPU time on every thread, mean and 99th percentile (shard changes, spawns and the prior's
  passes make spikes: ENGINES.md §5, MANAGER.md §7); for the manager, `jobs_ms` against `planned_ms` and the budget.
- **Uploads**: bytes per frame (only new pictures).
- **Memory**: the table of §7, against your budget.
- **Allocations**: none per frame; to check in your build, count `operator new` around the frame as
  `tests/alloc_test.cpp` does.
- **Pictures**: the same calls give the same frames; a golden-image test of a few frames per effect catches a broken
  integration (wrong blend, wrong colour space, a stale upload) at once.
