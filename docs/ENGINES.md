# Using NeuralVFX effects in a game engine

Status: **current**. Audience: dev. The API is `include/neuralfx/nvfx.h` (C, no dependencies) for single effects and
`include/neuralfx/nvfx_scene.h` for composed scenes played from scripts (§7); the reference host is `examples/c_host.c`,
and a Godot 4 plugin is in `engines/godot` (§8).

## 1. What the runtime does

An effect is a `.nvfx` file: a small network trained for one effect. The runtime evaluates it on the CPU and writes
a premultiplied RGBA8 sprite, which the engine uploads to a texture and draws on a particle or billboard.

```c
nvfx_effect* fire;
nvfx_effect_load("fire.nvfx", &fire);           /* once, at level load; shareable between threads */

nvfx_instance* inst;
nvfx_instance_create(fire, 128, &inst);         /* per playing copy; all memory allocated here */
float controls[3] = {0.8f, 0.3f, 0.6f};          /* intensity, wind, turbulence (see nvfx_effect_control_name) */
nvfx_instance_set_controls(inst, controls, 3);
nvfx_instance_set_seed(inst, 1234);             /* which variation; looping effects drift between variations */

/* every effect update (for example at 30 Hz): */
nvfx_render(inst, time_seconds, pixels, 128 * 4);  /* no allocation, no locks */
upload_to_texture(pixels);
```

Blend the sprite with **premultiplied alpha**: `dst = src.rgb + dst * (1 - src.a)` (in most engines: source factor
ONE, destination factor ONE_MINUS_SRC_ALPHA). The same blend covers additive light (fire: colour above alpha) and
smoke (colour at most alpha).

## 2. Budget and threading

- `nvfx_render` is single-threaded and touches only the instance's own scratch memory; run different instances on
  different worker threads freely. One instance must not be rendered from two threads at once.
- Cost scales with pixels: measured costs per configuration are in [REPORT.md](REPORT.md) §7 (about 1 ms per
  128 x 128 frame for the default grid model on one AVX2 core, a quarter of that at 64 x 64).
- Evaluate effects at their own rate (20-30 Hz is plenty for fire and smoke) and blend the last two results in the
  shader if needed; share one instance between all copies of an effect that use the same controls and seed; use
  64 x 64 for distant effects (`nvfx_instance_create(effect, 64, ...)`; a grid effect renders at any multiple of 16,
  a conv effect at its native size or a half or quarter of it).
- `nvfx_bake(inst, frames, buffer)` expands an effect into an ordinary flipbook (for example at load time, or on a
  platform without the CPU budget). This saves download size, not run-time memory.

## 3. Controls

| control | how | exact? |
|---|---|---|
| learned controls (intensity, wind, turbulence for simulated effects) | `nvfx_instance_set_controls` | learned: interpolates between training settings |
| variation | `nvfx_instance_set_seed`, `nvfx_instance_set_variation` (replay a training clip), `nvfx_instance_set_drift` | learned |
| playback speed | scale the time you pass to `nvfx_render` | exact |
| hue and brightness | `nvfx_instance_set_colour` | exact (applied to the output colour) |
| precision (grid frame models) | `nvfx_instance_set_precision`: `NVFX_PRECISION_INT8` (the default for the grid family) or `NVFX_PRECISION_FLOAT` | int8 changes a few pixels by a few levels (REPORT.md §7); float is the reference network |

Looping effects wrap time; one-shot effects (explosions) hold their last frame after their duration
(`info.frames / info.fps` seconds).

## 4. Engines

Nothing here is engine-specific; these notes say where the calls go. The Godot plugin is in this repository and
tested headless (§8); the Unreal and Unity notes are not built or tested.

### Unreal Engine

- Build `nvfx_shared` (`libnvfx.so` / `nvfx.dll`) or link the static `nvfx` into a runtime module; add
  `include/` to `PublicIncludePaths`.
- Own effects in a `UObject` (or an asset type that wraps the `.nvfx` bytes and calls `nvfx_effect_load_memory`).
- Render into a `UTexture2D` created with `PF_R8G8B8A8` and updated with `UpdateTextureRegions`, or into a
  `FRHITexture` from a render-thread task. Call `nvfx_render` on a worker (`UE::Tasks`) and upload on the render thread.
- Use the texture as a Niagara sprite's material texture with a premultiplied blend (Translucent, with the material
  outputting premultiplied colour, or Additive for pure emission).

### Unity

- Build `nvfx_shared` and place the library in `Assets/Plugins` (per platform). Declare the functions with
  `[DllImport("nvfx")]`; the API uses only C types and opaque pointers (`IntPtr`).
- Render into a `NativeArray<byte>` from a Burst-free `IJob` (the call is native), then `Texture2D.LoadRawTextureData`
  + `Apply(false)`; or update a texture from a native render plugin.
- Use a particle material with `Blend One OneMinusSrcAlpha`.

### Godot 4

- The GDExtension in `engines/godot` (§8): `NeuralVFXScene` plays a scene script and `NeuralVFXEffect` one effect,
  each into an `ImageTexture`.
- Draw an effect's texture with premultiplied-alpha blending (`CanvasItemMaterial.blend_mode = BLEND_MODE_PREMULT_ALPHA`,
  or a spatial material's `blend_mode = premul_alpha`). A scene's picture is opaque.

### Custom engines

`examples/c_host.c` is the whole integration: load, create, set controls, render in the update loop, upload. Build
the static library with your engine's compiler flags, or link the shared one. The library needs x86-64; it picks
AVX2 at run time when present and falls back to SSE2 code otherwise (AVX-512 code is included and can be forced
with `nvfx_set_isa`, but it was not faster on the benchmark machine).

## 5. Rollout effects

A `.nvfx` file can also hold a **rollout effect** (`nvfx_effect_info.arch == 3`; [REPORT.md](REPORT.md) §6): a few
stored simulation states (start points) and a small network that moves the effect forward one frame at a time. The API
is the same; what differs:

- **Shards.** A looping effect plays as a chain of shards (6 s by default, `nvfx_instance_set_drift` sets the length),
  each a fresh rollout from a start point chosen by the seed, the shard and the nearest controls, with a seed of its
  own; the next shard is rolled ahead of its turn and crossfades in over half a second. Drift never builds up beyond a
  shard (one continuous rollout, `set_drift(0)`, wanders off after 20 s or so, unless a prior is attached: below), and the frame shown depends only on the
  time and the controls.
- **One continuous run: the prior against drift (optional, fire only).** A second file, a small denoiser trained for
  the effect (`fire.ddpm`: 1.57 MB, about 19 times the 82 KB fire effect, whose own file is unchanged), keeps one
  continuous rollout alive for a minute or more without shards. Attach it once, before sharing the effect between
  threads and creating instances: `nvfx_effect_attach_prior(fire, "fire.ddpm")` (or `_memory`); then
  `nvfx_instance_set_drift(inst, 0)`. Every 16th frame one pass of the denoiser (89.5 million multiply-adds) moves the
  coarse state towards its estimate of a clean state ([DCM.md](DCM.md) G2.6, G2.13). Study G found it **ties** the 6 s
  shards on every statistic: it buys continuity (no restarts, no crossfades), not better pictures, and it was
  validated on fire only (smoke tied without it; explosions do not run long enough to drift).
  `nvfx_instance_set_prior(inst, every_frames, t, beta)` changes it (default 16, 100, 1; 0 frames turns it off); with
  shards it does nothing. **Cost** (provisional, one AVX2 core, 128 x 128): on average what shards cost (0.91 against
  0.93 ms per frame), but not evenly: the frame where the pass runs costs about 3.9 ms instead of 0.7 ms, once every
  16 frames (worst frame 4.1 ms, against 1.85 ms with shards). Running `nvfx_render` on a worker hides it from the game thread; with many such instances, start them on
  different frames so their passes fall on different frames (the pass runs on frames 16, 32, ... of each instance's
  own timeline). Memory: 1.57 MB per effect, 1.1 MB more per instance (in `nvfx_instance_scratch_bytes`).
- **Time moves forward.** `nvfx_render(inst, t, ...)` steps the shards to frame `floor(t * fps)`: normal playback
  costs one step per new frame, two during the half second before each shard change (the next shard rolling ahead) and
  two renders during the crossfade. Going backwards, or more than 2 s forwards, restarts the shard that contains `t`
  (a one-off cost of up to one shard of steps). Playback speed is exact as before: scale the time you pass.
- **It never repeats.** Every shard starts afresh, and a new seed (`nvfx_instance_set_seed`) gives new runs; changing
  it mid-play takes effect from the next shard, without a jump.
- **Variations are start points.** `nvfx_instance_set_variation(i)` restarts from start point `i` with the seed of the
  run it came from (`info.n_variations` start points).
- **Controls change the dynamics**, so a change shows within a few frames rather than at once.
- **Sizes**: any multiple of 32 from 32 to 1024. The coarse step costs the same at every size; the detail layer and the
  renderer scale with pixels.
- **One instance per playing copy.** The state belongs to the instance; two copies that should look different need two
  instances (sharing one instance shares the look, as before).
- **Memory**: the effect holds the weights and the start points (stored and resident sizes in `nvfx_effect_info`);
  each instance holds two shards' states and one set of work buffers that both shards' steps share
  (`nvfx_instance_scratch_bytes`: about 2.4 MB at 128 x 128, 1.5 MB at 64 x 64; 4.0 and 2.3 MB before they shared it).
- `nvfx_bake` renders consecutive frames from a fresh start and crossfades a few extra frames into the first ones, so
  the flipbook loops.

## 6. Memory

| what | where | size |
|---|---|---|
| effect weights | `nvfx_effect` | `info.resident_bytes`: features stay at their stored precision (8 or 16 bits); the small MLP is widened to floats |
| per instance | `nvfx_instance` | `nvfx_instance_scratch_bytes`: a few tens of KB at float precision; at int8 (the grid family's default) 60 to 400 KB, most of it the first layer evaluated at every grid point (grid_m 0.18 MB, grid_l 0.38 MB); rollout effects about 2.4 MB at 128 x 128, 1.1 MB more with a prior |
| prior against drift (optional, rollout effects) | `nvfx_effect`, after `nvfx_effect_attach_prior` | the denoiser's weights as floats: 1.57 MB for fire; counted in `info.resident_bytes` |
| output | the engine's buffer | size x size x 4 bytes |

## 7. Scenes: composed effects from a script

A scene ([COMPOSE.md](COMPOSE.md)) is a script (`.nvfxs`) and the rollout effects it names: modules that run on one
another through their fields (velocity, heat, soot), force fields, particles, light and rules that change the scene
over time. `include/neuralfx/nvfx_scene.h` plays one for an engine: a finished picture per frame (opaque: the scene's
own sky and ground), the field bus for gameplay, and calls with which the game drives the scene while it plays.

```c
nvfx_scene_desc d;
nvfx_scene_desc_init(&d);                  /* threads 2, overlap on, the script's size */
d.script_path = "fireball.nvfxs";          /* or d.script = text */
d.effects_dir = "effects/";                /* or loaded effects in d.effects (by the file name the script uses) */
nvfx_scene_error err;
nvfx_scene* scene;
if (nvfx_scene_create(&d, &scene, &err) != NVFX_OK)
  printf("%s\n", err.message);             /* "fireball.nvfxs:12:7: unknown module 'wrek' (did you mean 'wreck'?)" */

/* every game frame: */
nvfx_scene_step(scene, dt, NULL);          /* the scene's clock: computes the frames whose time has come */
nvfx_scene_fields f;
nvfx_scene_sample(scene, x, y, &f);        /* gameplay: f.heat > 0.1 means fire here */
nvfx_scene_set_input(scene, "wind", w);    /* a value the script reads */
nvfx_scene_trigger(scene, "explode");      /* a rule fired by name */
nvfx_scene_render(scene, rgba, stride);    /* the picture of the current frame */
upload_to_texture(rgba);
```

| what | calls |
|---|---|
| create | `nvfx_scene_desc_init`, `nvfx_scene_create` (script text or file; effects from a folder or passed in; threads, overlap, output size), `nvfx_scene_free`, `nvfx_scene_get_info` (sizes, fps, length, the bus's grid, counts, memory) |
| check | `nvfx_scene_check` (parse and check without effects, for an editor's quick loop), `nvfx_scene_list_effects` (the files a script names, for hosts that read them from a package and pass them in) |
| clock | `nvfx_scene_step(dt)`, `nvfx_scene_step_frames(n)`, `nvfx_scene_seek(seconds)`, `nvfx_scene_restart`, `nvfx_scene_time`, `nvfx_scene_frame` |
| picture | `nvfx_scene_render(rgba, stride)` |
| fields | `nvfx_scene_sample` (heat, soot, velocity at a point), `nvfx_scene_sample_grid` (a field on a grid of points: a tile map), `nvfx_scene_field_region` (largest and mean value in a rectangle) |
| inputs | `nvfx_scene_set_input`, `nvfx_scene_get_input`, `nvfx_scene_input_name` |
| rules | `nvfx_scene_trigger`, `nvfx_scene_rule_state` (times fired, last time), `nvfx_scene_rule_name` |
| modules | `nvfx_scene_module_get_info` (where, active, controls), `nvfx_scene_module_place`, `nvfx_scene_module_set_control`, `nvfx_scene_module_name`, `nvfx_scene_module_control_name` |

**The clock.** A scene runs at its script's frame rate (30 fps for the examples); frame f is at f / fps seconds. A new
scene is at frame 0, with its state computed (the modules' warm-ups run in `nvfx_scene_create`). `nvfx_scene_step`
moves the clock by the game's dt and computes the state of every frame whose time has come; `nvfx_scene_render` draws
the current frame, or copies it if it is drawn already. A game at 60 Hz draws a new picture every other call; a game
at 20 Hz computes one or two frames per call and draws the last. Frames computed without a picture change none of the
pictures after them (shading keeps no state): the fireball drawn only at its eight keyframes gives the frozen
keyframes to the bit. `nvfx_scene_seek` forwards steps; backwards, and `nvfx_scene_restart`, rebuild the scene from its
script and effects (kept in memory: no file is read) and play it to the time. A rebuild costs what creation costs and
allocates: for the fireball, 0.2 s (created in 202 ms with its files read, restarted in 173 ms, on 2 threads of a busy
machine). The scene may play on after its `length`
(the game decides: restart, stop or carry on).

**Overlap.** With 2 threads or more, each picture is drawn on a thread of its own while the next frame's state is
computed (docs/COMPOSE.md §7.3): the same pictures, faster. The state then runs a frame ahead of the picture: after
`nvfx_scene_render` of frame f, field reads see frame f + 1, and inputs, triggers and module settings given after it
act from frame f + 2 (without overlap, from the next frame). In the usual order (step, read fields, set inputs, render)
the fields read are those of the frame drawn.

**The picture.** RGBA8, rows top to bottom, alpha 255. `desc.width` and `desc.height` set another output size: the
scene is drawn at its script's size and resampled (bilinear), which does not make it cheaper; to draw fewer pixels,
write the script at a smaller size.

**Fields, for gameplay.** The field bus holds every active module's velocity, heat and soot in world space (world
pixels, y down, as in the script), in cells of `info.bus_cell` pixels (8 for the fireball), sampled bilinearly; zero
outside the bus. Heat and soot are the simulation's units (a fire's flames are about 0.2 to 2; the fireball's blast
1.4); velocity is in world pixels per second. For a tile map, `nvfx_scene_sample_grid` with the first tile's centre
and the tile size gives one value per tile; `nvfx_scene_field_region` answers "is anything in this rectangle
burning?". They read the bus of the last frame computed and cost a few bilinear lookups each.

**Driving the scene.** Three ways, each acting from the next frame computed:
- **Inputs:** a script declares `input NAME = value` and uses NAME in any expression that may change over time (a
  module's place or controls, a field's strength, a rule's condition). `nvfx_scene_set_input` sets it; starting values
  can be given in `desc.inputs`. A restart keeps the inputs' current values.
- **Triggers:** `nvfx_scene_trigger(scene, "NAME")` fires the rule `as NAME` at the start of the next frame, as if its
  condition held, if it has firings left (`at most N`; `repeat`: always). A rule only the game fires is written
  `when 0 as NAME:`. Landing rules cannot be triggered.
- **Modules:** `nvfx_scene_module_place` moves a module that is not tiled; `nvfx_scene_module_set_control` sets one of
  its learned controls. A place or control the script changes over time is set by the script again next frame, so
  give the game an input for those instead.

**Errors.** Script errors (`NVFX_ERROR_SCRIPT`) carry the line and the column, in `nvfx_scene_error` and in the
message (`"name:line:column: message"`, with a suggestion for a misspelt word). The script is checked before any
effect is loaded; an effect that cannot be read (`NVFX_ERROR_IO`) or is not a rollout effect (`NVFX_ERROR_FORMAT`)
names its `effect` statement's line. Exceptions never cross the C boundary.

**Threads and memory.** A scene is used from one thread at a time; it runs `desc.threads` threads of its own, the
calling thread included (the picture thread is one of them). Scenes are independent of one another. Everything is
allocated by `nvfx_scene_create`: stepping, drawing, field reads and the run-time settings allocate nothing
(`nvfx_alloc_test` drives a scripted scene through the API at 60 Hz, with inputs, triggers, module moves and field
reads between frames, overlapped and not: 0 allocations). The fireball's modules take 127 MB at 1280 x 720
(`info.scratch_bytes`).

**Cost.** What the scene costs (docs/COMPOSE.md §7.3): the fireball at 1280 x 720 in 12.5 ms per frame on 4 threads
and 23.7 ms on 2 (a quiet machine), plus the copy into RGBA (0.9 ms at 1280 x 720).

**Exactness.** The API plays a script as `nvfx_scene_script` does, to the bit: every one of the fireball's 270 frames,
played from C through `libnvfx.so` (`nvfx_c_host --scene ... --expect PROFILE`), has the RGB checksum the script runner
writes in its profile (`ctest -R SceneApiFireballBitExact`); the fireball's eight keyframes through the API match the
hand-written scene's frozen ones (`SceneApi.FireballKeyframesMatchTheFrozenOnes`); and a scene that uses every
statement gives the runner's frames on 1 and 2 threads, overlapped or not, with skipped pictures, seeks and restarts
(`SceneApi.*`).

**In the shared library.** `libnvfx.so` holds the scene API with the composition code built in (position-independent,
hidden symbols, compose's own compiler flags, so its frames are the static build's). Of the library's own code it
exports the `nvfx_*` C functions and nothing else (the static libraries inside are linked with `--exclude-libs`; what
remains are a few instances of standard-library templates). The static build has the scene API in `nvfx_scene_api`
(with `neuralfx_compose`).

```sh
build/nvfx_c_host --scene examples/scenes/fireball.nvfxs --effects $NEURALVFX_DATA/experiments/models/d --out frame.pam
```

## 8. Godot 4

`engines/godot` is a GDExtension over the C API: two nodes, a demo project and a headless test. It is built against
godot-cpp at the pinned tag `godot-4.4.1-stable` (fetched by CMake) and tested with the official Godot 4.4.1 Linux
binary. It is outside the default build: `-DNEURALFX_BUILD_GODOT=ON` (needs Python 3 for godot-cpp's binding
generator; a build profile, `engines/godot/build_profile.json`, binds only the engine classes the nodes use, so
godot-cpp, libnvfx and the extension build in about 3.5 minutes on 2 cores of a busy machine). The extension (`libneuralvfx_godot.so`) and `libnvfx.so` are copied
into `engines/godot/demo/bin`, where `demo/neuralvfx.gdextension` finds them (the extension finds libnvfx next to
itself; an export copies it along). Linux x86-64 only so far.

```sh
cmake -S . -B build-godot -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 -DNEURALFX_BUILD_GODOT=ON
cmake --build build-godot --target nvfx_godot
engines/godot/test.sh          # downloads Godot 4.4.1 (checked), builds, and runs the demo's test headless
```

**`NeuralVFXScene`** (a `Node`): plays a scene script into `texture` (an `ImageTexture`, updated when a new frame is
drawn; `get_image()` gives the CPU image).
- Properties: `scene_script` (a `.nvfxs`: `res://`, `user://` or an absolute path), `effects_dir` (default: the
  script's folder), `threads` (2), `overlap` (on), `output_size` (0: the script's), `playing` (on), `looping`,
  `speed`.
- The scene: `load_scene()` (done in `_ready` when `scene_script` is set), `advance(seconds)`, `seek(seconds)`,
  `restart()`, `get_time()`, `get_frame()`, `get_length()`, `get_fps()`, `get_scene_size()`; errors in `get_error()`,
  `get_error_line()`, `get_error_column()` (and pushed to Godot's error log).
- Fields: `get_heat(position)`, `get_soot(position)`, `get_velocity(position)`, `sample(position)` (a dictionary),
  `get_field_grid(field, origin, step, count)` (a `PackedFloat32Array`, one value per tile), `get_field_max(field,
  rect)`, `get_field_mean(field, rect)`; `field` is `FIELD_HEAT`, `FIELD_SOOT`, `FIELD_U` or `FIELD_V`.
- Driving: `set_input(name, value)`, `get_input(name)`, `trigger(rule)`, `get_rule_count(rule)`,
  `move_module(module, position)`, `set_module_control(module, control, value)`, `get_module_info(module)`, and the
  names: `get_input_names()`, `get_rule_names()`, `get_module_names()`.
- Signals: `frame_drawn(frame)`, `finished` (at the script's length, when not looping; playing stops).

**`NeuralVFXEffect`** (a `Node`): one effect (an `.nvfx`: a frame model or a rollout effect) into `texture`, with
premultiplied alpha. Properties `effect_path`, `size` (128), `controls` (a `PackedFloat32Array`), `seed`, `drift`
(seconds; negative: the effect's default), `hue`, `brightness`, `playing`, `speed`, `time`; methods `load_effect()`,
`render_at(seconds)`, `get_control_names()`, `get_info()`.

The nodes read files with `FileAccess` and pass them to the C API in memory, so they work from an exported package:
the scene node lists the script's effects (`nvfx_scene_list_effects`) and reads each from `effects_dir`. They do not
play in the editor.

```gdscript
var scene := NeuralVFXScene.new()
scene.scene_script = "res://scenes/fireball.nvfxs"
scene.effects_dir = "res://effects"
add_child(scene)                                   # loads and plays
$Sprite2D.texture = scene.texture
...
if scene.get_heat(player.position) > 0.1:          # gameplay: the player stands in fire
    player.burn()
```

**The headless test** (`engines/godot/test.sh`, all in the container): downloads the official Godot 4.4.1 Linux binary
(pinned, SHA-512 checked) into `$NEURALVFX_DATA/godot` (not in git), builds the extension and `nvfx_scene_script` in
`build-godot/`, renders the script runner's keyframe of frame 45 as the reference, imports the demo project (which
registers the extension) and runs `demo/test.gd` headless: a `NeuralVFXScene` plays the fireball for 45 frames (1.5 s,
0.3 s after the detonation) and saves the frame as PNG. Its checks, all passed:
- the frame is 1280 x 720, not black, bright where the fireball is and varied, and **its pixels are the script
  runner's frame 45, byte for byte**;
- the heat is 1.42 at the blast's centre, 0.93 over the burning wreck and 0 in the far sky; 194 of 880 tiles of 32
  pixels are burning (a heat grid as a tile map would read it);
- `frame_drawn` fired for frame 0 and each of the 45 frames, and `texture` is the scene's picture;
- the detonation rule fired once; the wreck's fire is active where the script put it, (1010, 600), with its controls;
  an unknown rule cannot be triggered;
- a broken script is reported as `broken.nvfxs:2:34: 'strenght' is not a property of field (did you mean
  'strength'?)`, line 2, column 34;
- a `NeuralVFXEffect` loads `fire.nvfx` and draws a fire at 128 x 128.

The 45 frames took 17 to 45 ms each on 2 threads in Godot, depending on what else ran on the shared machine. The
saved frame is `$NEURALVFX_DATA/godot/fireball_godot.png`. `demo/main.tscn` is the same scene for a window: the
fireball full screen, looping, with the heat under the mouse shown in a label.

What the plugin does not do yet: other platforms than Linux x86-64; drawing in the editor; uploading the fields as
textures for the engine's own renderer (the picture is composed on the CPU and uploaded: 3.7 MB per frame at
1280 x 720); a transparent background, so that a scene could be drawn over the game's world.
