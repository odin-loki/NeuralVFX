/* NeuralVFX runtime: evaluate trained neural effects on the CPU. C API for game engines (docs/ENGINES.md); composed
 * scenes played from scripts are in nvfx_scene.h.
 *
 * An effect (nvfx_effect) is a loaded .nvfx file: immutable, shareable between threads and instances. An instance
 * (nvfx_instance) is one playing copy with its own controls, seed and scratch memory; it is used from one thread
 * at a time. nvfx_render() never allocates, locks or blocks: every buffer is created with the instance.
 *
 * Output: premultiplied RGBA8, rows top to bottom. Blend with dst = src.rgb + dst * (1 - src.a): one blend mode
 * covers additive light (fire) and alpha-blended smoke.
 *
 * Controls: the effect's learned controls (for simulated effects: intensity, wind, turbulence, each in [0, 1]).
 * Exact runtime controls: playback speed (scale the time you pass), hue rotation and brightness (applied to the
 * output colour), and the seed (which variation plays; looping effects drift slowly between variations so they never
 * repeat).
 */
#ifndef NEURALVFX_NVFX_H
#define NEURALVFX_NVFX_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(NVFX_SHARED_BUILD)
#define NVFX_API __declspec(dllexport)
#elif defined(__GNUC__)
#define NVFX_API __attribute__((visibility("default")))
#else
#define NVFX_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NVFX_VERSION 7 /* 2: scenes (nvfx_scene.h); 3: nvfx_version(), effect packs (nvfx_pack.h); 4: engine layers and trails; 5: precision (study K); 6: 3D volumes (nvfx_volume.h); 7: drawn at half size and upscaled (study P) */

typedef enum nvfx_status {
  NVFX_OK = 0,
  NVFX_ERROR_ARGUMENT = 1,   /* a null pointer or a value out of range */
  NVFX_ERROR_IO = 2,         /* the file could not be read */
  NVFX_ERROR_FORMAT = 3,     /* not a valid .nvfx file */
  NVFX_ERROR_MEMORY = 4,     /* allocation failed (only at load or instance creation) */
  NVFX_ERROR_UNSUPPORTED = 5, /* size or ISA not supported */
  NVFX_ERROR_SCRIPT = 6      /* a scene script is wrong (nvfx_scene.h: the error has its line and column) */
} nvfx_status;

typedef enum nvfx_isa { NVFX_ISA_AUTO = 0, NVFX_ISA_BASELINE = 1, NVFX_ISA_AVX2 = 2, NVFX_ISA_AVX512 = 3 } nvfx_isa;

typedef struct nvfx_effect nvfx_effect;
typedef struct nvfx_instance nvfx_instance;

typedef struct nvfx_effect_info {
  int arch;            /* 1 = grid (any size), 2 = conv (native size and native / 2, / 4), 3 = rollout (multiples of 32),
                          4 = multi-level grid (any size) */
  int native_size;     /* training sprite size in pixels */
  int frames;          /* frames of the training clip */
  float fps;           /* frames per second of the training clip */
  int loops;           /* 1 = seamless loop, 0 = plays once */
  int n_controls;      /* learned controls */
  int n_variations;    /* training variations that can be replayed (0 if none); rollout: start points */
  size_t stored_bytes; /* size of the weights as shipped */
  size_t resident_bytes; /* memory the loaded effect occupies (features stay in their stored format) */
  char name[32];
} nvfx_effect_info;

NVFX_API const char* nvfx_status_string(nvfx_status status);

/* Effects ------------------------------------------------------------------------------------------------------ */
NVFX_API nvfx_status nvfx_effect_load(const char* path, nvfx_effect** out);
NVFX_API nvfx_status nvfx_effect_load_memory(const void* data, size_t bytes, nvfx_effect** out);
NVFX_API void nvfx_effect_free(nvfx_effect* effect);
NVFX_API nvfx_status nvfx_effect_get_info(const nvfx_effect* effect, nvfx_effect_info* info);
/* Name of learned control i (e.g. "intensity"), or NULL. */
NVFX_API const char* nvfx_effect_control_name(const nvfx_effect* effect, int i);

/* Instances ---------------------------------------------------------------------------------------------------- */
/* `size`: output side in pixels (grid: any multiple of 16 from 16 to 1024; conv: native, native / 2 or native / 4;
 * rollout: any multiple of 32 from 32 to 1024). */
NVFX_API nvfx_status nvfx_instance_create(const nvfx_effect* effect, int size, nvfx_instance** out);
NVFX_API void nvfx_instance_free(nvfx_instance* instance);
NVFX_API size_t nvfx_instance_scratch_bytes(const nvfx_instance* instance);
NVFX_API nvfx_status nvfx_instance_set_controls(nvfx_instance* instance, const float* controls, int count);
/* Which variation plays: a seed (any value), or a training variation by index (seed ignored; -1 returns to seeds). */
NVFX_API nvfx_status nvfx_instance_set_seed(nvfx_instance* instance, uint64_t seed);
NVFX_API nvfx_status nvfx_instance_set_variation(nvfx_instance* instance, int training_index);
/* Frame models: seconds per drift between variations for looping effects (default: 4 loops; 0 = no drift).
 * Rollout effects: seconds per shard, each a fresh rollout from a start point, crossfaded into the next (default 6;
 * 0 = one continuous rollout, which drifts after 20 s or so). */
NVFX_API nvfx_status nvfx_instance_set_drift(nvfx_instance* instance, float seconds);
/* Exact colour controls applied to the output: hue rotation in radians, brightness multiplier (default 0, 1). */
NVFX_API nvfx_status nvfx_instance_set_colour(nvfx_instance* instance, float hue_radians, float brightness);

/* Precision of an instance's network.
 * Frame models (docs/REPORT.md §7). INT8: the hidden layers multiply 8-bit activations (one scale per pixel) by 8-bit
 * weights (one scale per unit) in integers, and the first layer is evaluated per grid point and interpolated; AVX-512
 * VNNI where the CPU has it (with no ISA forced, int8 instances of models with a hidden layer then use the AVX-512
 * build). FLOAT: the float network (the reference). DEFAULT is INT8 for the grid family: on study A's 12 clips it
 * changes active PSNR by -0.008 dB [-0.015, -0.001] with VNNI and -0.018 dB [-0.031, -0.008] with AVX2 (whose
 * activations get 7 bits), within the -0.05 dB allowed for a default; no pixel channel moves by more than 6 to 10
 * levels of 255 there (up to 20 on studies B and C). It costs 0.41 to 0.51 ms instead of 0.75 to 0.78 ms per 128 x 128
 * frame for the larger models (provisional). The conv family is float either way (INT8 returns NVFX_ERROR_UNSUPPORTED).
 * Rollout effects (study K, docs/REPORT.md §13). INT8: the stepper's second 3 x 3 convolution (two thirds of its
 * multiply-adds) multiplies 7-bit activations (one scale per channel over the grid) by 8-bit weights (one scale per
 * output channel) in integers; every ISA computes the same integer sums, and with no ISA forced the AVX-512 build is
 * used where the CPU has VNNI. The renderer and the detail layer stay float. Its frames are not float's: the effects
 * are chaotic, so int8 and float runs drift apart after a while, as two seeds do. DEFAULT is INT8 for rollout effects
 * where the CPU has AVX-512 VNNI and no other ISA is forced, FLOAT elsewhere: on held-out settings and seeds int8
 * tracked real runs and played as float does on all eleven trained effects (study K's rule), and with VNNI a frame
 * costs about a tenth less than float in the same build at 128 x 128. FLOAT is the reference; with no ISA forced it
 * takes the AVX-512 build where the CPU has AVX-512 (0.80 of AVX2's time, docs/REPORT.md §13.10), else AVX2; the
 * builds may differ in the last bits (nvfx_set_isa forces one). A change starts the instance's timeline again.
 * A set-up call: a change allocates the instance's new buffers (at INT8 a frame model's instance holds the first layer
 * at every grid point: 0.2 to 0.4 MB). */
typedef enum nvfx_precision { NVFX_PRECISION_DEFAULT = 0, NVFX_PRECISION_FLOAT = 1, NVFX_PRECISION_INT8 = 2 } nvfx_precision;
NVFX_API nvfx_status nvfx_instance_set_precision(nvfx_instance* instance, nvfx_precision precision);

/* Rollout effects: the prior against drift (docs/ENGINES.md §5, docs/DCM.md G2.13) ------------------------------
 * An optional second file, a small denoiser trained for the effect (fire.ddpm: 1.57 MB, about 20 times the 82 KB
 * fire effect, whose file is unchanged). With it one continuous rollout (nvfx_instance_set_drift(instance, 0)) plays
 * for a minute or more without drifting: every N frames one pass of the denoiser (89.5 M multiply-adds, a few ms on
 * one AVX2 core, so that frame costs that much more) moves the coarse state towards the denoiser's one-step estimate
 * of a clean state. Study G found it ties the 6 s shards on every statistic: it buys continuity (no restarts, no
 * crossfades), not better pictures. Validated on fire only.
 * Attach once, before the effect is shared between threads and before creating instances. Errors:
 * NVFX_ERROR_ARGUMENT (not a rollout effect, or a prior is already attached), NVFX_ERROR_IO, NVFX_ERROR_FORMAT (not a
 * denoiser file, or one for another grid or condition). info.resident_bytes grows by the prior's weights. */
NVFX_API nvfx_status nvfx_effect_attach_prior(nvfx_effect* effect, const char* path);
NVFX_API nvfx_status nvfx_effect_attach_prior_memory(nvfx_effect* effect, const void* data, size_t bytes);
/* One pass every `every_frames` frames (0 = off) at noise level `t` (1 to the denoiser's T), blended with weight
 * `beta` in [0, 1]. Default once a prior is attached: 16, 100, 1 (study G's tested setting). It acts only while the
 * instance plays one continuous rollout (drift 0, or a one-shot effect); with shards it is idle. A change replays the
 * timeline from the start at the next render. Instances created after the attach hold the prior's buffers
 * (in nvfx_instance_scratch_bytes); for an instance created before it, this call allocates them. */
NVFX_API nvfx_status nvfx_instance_set_prior(nvfx_instance* instance, int every_frames, int t, float beta);

/* Rollout effects: the simulator's look for the first frames (docs/COMPOSE.md §10) ------------------------------
 * Draw the first `frames` frames of the instance's timeline (frame 0 is the start point) in the look of the simulator
 * the effect was trained on, drawn from the instance's own fine fields, then crossfade to the learned renderer over
 * `fade` frames. Meant for explosions, whose learned renderer draws a fresh fireball poorly. 0, 0 (the default) is
 * off. Only the picture changes, not the state. Costs one more drawing per frame while it shows; no allocation.
 * Experimental: it did not pass its rule (first-second active PSNR +0.86 dB on validation, but coverage distance
 * slightly worse in 23 of 24 configurations), and the extra drawing costs 2.8 to 7 times a frame while it shows.
 * Errors: NVFX_ERROR_ARGUMENT (not a rollout effect, negative counts, or an effect the simulator does not make). */
NVFX_API nvfx_status nvfx_instance_set_handoff(nvfx_instance* instance, int frames, int fade);

/* Rollout effects: drawn at half size and upscaled (study P, docs/STUDY_P.md) --------------------------------------
 * An instance can draw its effect at half its size and bring the frame to full size: the detail layer and the renderer,
 * whose work grows with the pixels, then do a quarter of it. NVFX_UPSCALE_BILINEAR is the bilinear 2x (what a GPU's
 * bilinear sampler gives; needs no file). NVFX_UPSCALE_LEARNED adds a small learned 2x (two small convolutions on the
 * half-size frame) and seeded micro-detail (fine noise whose amplitude the network gives from the frame), deterministic
 * from the instance's seed and the time; it needs an upscaler file attached to the effect (nvfx_effect_attach_upscaler,
 * once, before the effect is shared between threads; one file can serve every effect). The instance's size must be a
 * multiple of 64 (twice the coarse grid). A set-up call: it allocates the instance's new buffers and starts its
 * timeline again; nvfx_render then allocates nothing. The instance's colour is applied after the upscale. Pixels the
 * half-size frame leaves empty stay exactly transparent.
 * Experimental: LEARNED did not pass its rule (study P): at 512 px it costs 0.45 to 0.59 of drawing at full size (more
 * than the 0.55 allowed for two of six effects; provisional, a busy machine) and against the full-size frames it was
 * not better than BILINEAR on PSNR, SSIM and fine detail together (on the held-out test, docs/STUDY_P.md §6). The
 * half-size instance alone costs 0.29 to 0.34 of a full-size one at 512 px and is within the null's margins of the
 * full-size one against the simulator's frames: an engine gets most of the saving by creating the instance at half size
 * and letting its GPU sample it larger.
 * Errors: NVFX_ERROR_ARGUMENT (not a rollout effect, an unknown mode, LEARNED with no upscaler attached),
 * NVFX_ERROR_UNSUPPORTED (a size that is not a multiple of 64). */
typedef enum nvfx_upscale { NVFX_UPSCALE_OFF = 0, NVFX_UPSCALE_BILINEAR = 1, NVFX_UPSCALE_LEARNED = 2 } nvfx_upscale;
NVFX_API nvfx_status nvfx_effect_attach_upscaler(nvfx_effect* effect, const char* path);
NVFX_API nvfx_status nvfx_effect_attach_upscaler_memory(nvfx_effect* effect, const void* data, size_t bytes);
NVFX_API nvfx_status nvfx_instance_set_upscale(nvfx_instance* instance, nvfx_upscale mode);

/* Render the frame at `time_seconds` (looping effects wrap; one-shot effects hold their last frame) into `rgba`,
 * whose rows are `stride_bytes` apart (>= size * 4). No allocation. */
NVFX_API nvfx_status nvfx_render(nvfx_instance* instance, double time_seconds, uint8_t* rgba, size_t stride_bytes);

/* Bake `frames` frames covering one loop (or the whole one-shot effect) into a flipbook: frame f at
 * rgba + f * size * size * 4. The instance's controls, seed and colour are used. For engines that prefer textures. */
NVFX_API nvfx_status nvfx_bake(nvfx_instance* instance, int frames, uint8_t* rgba);

/* Version and features -------------------------------------------------------------------------------------------
 * For an engine to check, at start, a shared library against the headers it was built with (docs/INTEGRATION.md §8):
 * returns the library's NVFX_VERSION, which changes whenever a call or a public struct changes (require it to equal
 * the header's), and sets `features` (may be NULL) to NVFX_FEATURE_* bits: the APIs the library holds and the
 * instruction sets this CPU runs. */
#define NVFX_FEATURE_SCENE 0x1u     /* nvfx_scene.h */
#define NVFX_FEATURE_LIBRARY 0x2u   /* nvfx_library.h */
#define NVFX_FEATURE_MANAGER 0x4u   /* nvfx_manager.h */
#define NVFX_FEATURE_PACK 0x8u      /* nvfx_pack.h (packs of format NVFX_PACK_FORMAT and older) */
#define NVFX_FEATURE_SHARED 0x10u   /* this is the shared library (libnvfx.so, nvfx.dll) */
#define NVFX_FEATURE_VOLUME 0x20u   /* nvfx_volume.h (3D effects, study N) */
#define NVFX_FEATURE_AVX2 0x100u    /* this CPU runs the AVX2 code (used by default when present) */
#define NVFX_FEATURE_AVX512 0x200u  /* this CPU runs the AVX-512 code (rollout effects use it; nvfx_set_isa) */
#define NVFX_FEATURE_VNNI 0x400u    /* this CPU has AVX-512 VNNI (int8 frame models and rollout effects use it) */
NVFX_API uint32_t nvfx_version(uint32_t* features);

/* Testing and benchmarking -------------------------------------------------------------------------------------- */
NVFX_API nvfx_isa nvfx_get_isa(void);                 /* the default ISA: the one forced, else AVX2 when present (frame
                                                         models; rollout effects take AVX-512 where the CPU has it: see
                                                         nvfx_instance_set_precision) */
NVFX_API nvfx_status nvfx_set_isa(nvfx_isa isa);      /* force an ISA for instances created afterwards */
NVFX_API double nvfx_instance_macs_per_pixel(const nvfx_instance* instance);

#ifdef __cplusplus
}
#endif

#endif /* NEURALVFX_NVFX_H */
