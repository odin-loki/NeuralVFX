# Using NeuralVFX effects in a game engine

Status: **current**. Audience: dev. The API is `include/neuralfx/nvfx.h` (C, no dependencies); the reference host is
`examples/c_host.c`.

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

Looping effects wrap time; one-shot effects (explosions) hold their last frame after their duration
(`info.frames / info.fps` seconds).

## 4. Engines

Nothing here is engine-specific; these notes say where the calls go. None of the three plugins below is part of
this repository yet (no engine is available in the build environment to test one).

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

- Write a small GDExtension (C or C++) that links `nvfx` and exposes `load`, `create`, `set_controls` and `render` to
  GDScript; render into an `Image` (`FORMAT_RGBA8`) and update an `ImageTexture`.
- Use a `CanvasItemMaterial` or spatial material with premultiplied-alpha blending (`blend_mode = premul_alpha`).

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
  shard (one continuous rollout, `set_drift(0)`, wanders off after 20 s or so), and the frame shown depends only on the
  time and the controls.
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
  each instance holds two shards' states and work buffers (`nvfx_instance_scratch_bytes`: about 4.0 MB at 128 x 128,
  2.3 MB at 64 x 64).
- `nvfx_bake` renders consecutive frames from a fresh start and crossfades a few extra frames into the first ones, so
  the flipbook loops.

## 6. Memory

| what | where | size |
|---|---|---|
| effect weights | `nvfx_effect` | `info.resident_bytes`: features stay at their stored precision (8 or 16 bits); the small MLP is widened to floats |
| per instance | `nvfx_instance` | `nvfx_instance_scratch_bytes`: a few tens of KB |
| output | the engine's buffer | size x size x 4 bytes |
