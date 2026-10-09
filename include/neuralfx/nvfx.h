/* NeuralVFX runtime: evaluate trained neural effects on the CPU. C API for game engines (docs/ENGINES.md).
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

#define NVFX_VERSION 1

typedef enum nvfx_status {
  NVFX_OK = 0,
  NVFX_ERROR_ARGUMENT = 1,   /* a null pointer or a value out of range */
  NVFX_ERROR_IO = 2,         /* the file could not be read */
  NVFX_ERROR_FORMAT = 3,     /* not a valid .nvfx file */
  NVFX_ERROR_MEMORY = 4,     /* allocation failed (only at load or instance creation) */
  NVFX_ERROR_UNSUPPORTED = 5 /* size or ISA not supported */
} nvfx_status;

typedef enum nvfx_isa { NVFX_ISA_AUTO = 0, NVFX_ISA_BASELINE = 1, NVFX_ISA_AVX2 = 2, NVFX_ISA_AVX512 = 3 } nvfx_isa;

typedef struct nvfx_effect nvfx_effect;
typedef struct nvfx_instance nvfx_instance;

typedef struct nvfx_effect_info {
  int arch;            /* 1 = grid (any size), 2 = conv (native size and native / 2, / 4) */
  int native_size;     /* training sprite size in pixels */
  int frames;          /* frames of the training clip */
  float fps;           /* frames per second of the training clip */
  int loops;           /* 1 = seamless loop, 0 = plays once */
  int n_controls;      /* learned controls */
  int n_variations;    /* training variations that can be replayed (0 if none) */
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
/* `size`: output side in pixels (grid: any multiple of 16 from 16 to 1024; conv: native, native / 2 or native / 4). */
NVFX_API nvfx_status nvfx_instance_create(const nvfx_effect* effect, int size, nvfx_instance** out);
NVFX_API void nvfx_instance_free(nvfx_instance* instance);
NVFX_API size_t nvfx_instance_scratch_bytes(const nvfx_instance* instance);
NVFX_API nvfx_status nvfx_instance_set_controls(nvfx_instance* instance, const float* controls, int count);
/* Which variation plays: a seed (any value), or a training variation by index (seed ignored; -1 returns to seeds). */
NVFX_API nvfx_status nvfx_instance_set_seed(nvfx_instance* instance, uint64_t seed);
NVFX_API nvfx_status nvfx_instance_set_variation(nvfx_instance* instance, int training_index);
/* Seconds per drift between variations for looping effects (default: 4 loops; 0 = no drift). */
NVFX_API nvfx_status nvfx_instance_set_drift(nvfx_instance* instance, float seconds);
/* Exact colour controls applied to the output: hue rotation in radians, brightness multiplier (default 0, 1). */
NVFX_API nvfx_status nvfx_instance_set_colour(nvfx_instance* instance, float hue_radians, float brightness);

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
