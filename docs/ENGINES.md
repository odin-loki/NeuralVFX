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
- Cost scales with pixels: measured costs per configuration are in [REPORT.md](REPORT.md) §5 (about 1 ms per
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

## 5. Memory

| what | where | size |
|---|---|---|
| effect weights | `nvfx_effect` | `info.resident_bytes`: features stay at their stored precision (8 or 16 bits); the small MLP is widened to floats |
| per instance | `nvfx_instance` | `nvfx_instance_scratch_bytes`: a few tens of KB |
| output | the engine's buffer | size x size x 4 bytes |
