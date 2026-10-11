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

#define NVFX_VERSION 2 /* 2: scenes (nvfx_scene.h) */

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
  int arch;            /* 1 = grid (any size), 2 = conv (native size and native / 2, / 4), 3 = rollout (multiples of 32) */
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

/* Render the frame at `time_seconds` (looping effects wrap; one-shot effects hold their last frame) into `rgba`,
 * whose rows are `stride_bytes` apart (>= size * 4). No allocation. */
NVFX_API nvfx_status nvfx_render(nvfx_instance* instance, double time_seconds, uint8_t* rgba, size_t stride_bytes);

/* Bake `frames` frames covering one loop (or the whole one-shot effect) into a flipbook: frame f at
 * rgba + f * size * size * 4. The instance's controls, seed and colour are used. For engines that prefer textures. */
NVFX_API nvfx_status nvfx_bake(nvfx_instance* instance, int frames, uint8_t* rgba);

/* Testing and benchmarking -------------------------------------------------------------------------------------- */
NVFX_API nvfx_isa nvfx_get_isa(void);                 /* the ISA instances are created with */
NVFX_API nvfx_status nvfx_set_isa(nvfx_isa isa);      /* force an ISA for instances created afterwards */
NVFX_API double nvfx_instance_macs_per_pixel(const nvfx_instance* instance);

#ifdef __cplusplus
}
#endif

#endif /* NEURALVFX_NVFX_H */
