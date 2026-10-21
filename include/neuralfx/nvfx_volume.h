/* NeuralVFX volumes: learned 3D effects for engines (study N, docs/STUDY_3D.md; docs/ENGINES.md §12).
 *
 * A volume effect (nvfx_volume_effect) is a loaded .nvfx3 file: a 3D rollout effect, immutable and shareable between
 * threads. A volume (nvfx_volume) is one playing copy with its own controls, seed, threads and buffers; it is used from
 * one thread at a time. Every buffer is made when the volume is created: stepping, reading and rendering never
 * allocate.
 *
 * What an engine gets, both from the same state:
 *   - the volume itself as 3D textures, for the engine's own volume renderer: temperature, soot, extinction (per world
 *     unit, as this library's renderer sees it) and velocity (world units per second), as float or half;
 *   - a picture for a camera, ray-marched on the CPU (emission by Planck's law, absorption and single scattering by
 *     soot), as premultiplied RGBA8: blend with dst = src.rgb + dst * (1 - src.a).
 *
 * Space: the volume fills the box [-ex/2, ex/2] x [0, ey] x [-ez/2, ez/2] of its own world units, with
 * (ex, ey, ez) = nvfx_volume_info.extent (one unit across): y up, the floor at y = 0, centred on x = z = 0. Place it in a
 * scene by folding its transform into the view matrix. Textures are x fastest, then y (rows, up), then z (slices):
 * texel (x, y, z) at (z * size[1] + y) * size[0] + x.
 *
 * Time: nvfx_volume_step_to(t) brings the volume to frame floor(t * fps). Every frame is a function of the time, the
 * seed, the controls and the variation: the effect plays shards (default 8 s), each a fresh rollout from a stored start
 * point chosen by the seed, the shard and the nearest controls, with noise of its own, rolled ahead and crossfaded in
 * over 0.5 s, so it plays forever without repeating. Going back in time, or far ahead, restarts the shard's rollout
 * (a start costs the effect's warm-up, about a second of steps). The same frames on any number of threads.
 *
 * Experimental (study N is a small demo; see docs/STUDY_3D.md for what it was measured to do).
 */
#ifndef NEURALVFX_NVFX_VOLUME_H
#define NEURALVFX_NVFX_VOLUME_H

#include <neuralfx/nvfx.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nvfx_volume_effect nvfx_volume_effect;
typedef struct nvfx_volume nvfx_volume;

typedef struct nvfx_volume_info {
  int coarse[3];         /* the learned state's grid (x, y, z) */
  int size[3];           /* the fields (3D textures): x, y, z */
  float extent[3];       /* the volume's box in world units (x, y, z) */
  float fps;             /* frames per second of the rollout */
  int n_controls;        /* learned controls (intensity, wind, turbulence) */
  int n_start_points;    /* stored start points (variations) */
  size_t stored_bytes;   /* the effect as shipped */
  size_t resident_bytes; /* memory the loaded effect occupies */
  char name[32];
} nvfx_volume_info;

typedef enum nvfx_volume_field {
  NVFX_VOLUME_TEMPERATURE = 0, /* the model's temperature (0 cold; about 1 to 1.5 in the flame; warm smoke, "smoke3d":
                                  0 to about 0.45, its excess over the air, about 98 K per unit) */
  NVFX_VOLUME_SOOT = 1,        /* soot density */
  NVFX_VOLUME_EXTINCTION = 2,  /* extinction per world unit, as this library's renderer has it */
  NVFX_VOLUME_VELOCITY = 3     /* three components per texel (x, y, z), world units per second */
} nvfx_volume_field;

typedef enum nvfx_volume_texel { NVFX_VOLUME_FLOAT = 0, NVFX_VOLUME_HALF = 1 } nvfx_volume_texel;

/* A camera as an engine has it: column-major matrices, OpenGL conventions (the camera looks down its -z axis; clip z in
 * [-1, 1]). Any camera works: perspective or orthographic, from any side. */
typedef struct nvfx_volume_camera {
  float view[16]; /* world to view */
  float proj[16]; /* view to clip */
} nvfx_volume_camera;

/* Or a look-at description, made into a camera by nvfx_volume_camera_look_at(). */
typedef struct nvfx_volume_look_at {
  float eye[3];
  float target[3];
  float up[3];
  float fov_y_degrees; /* vertical field of view */
  float aspect;        /* width / height of the picture */
  float near_plane;    /* 0: 0.02 */
  float far_plane;     /* 0: 50 */
} nvfx_volume_look_at;

/* Effects --------------------------------------------------------------------------------------------------------- */
NVFX_API nvfx_status nvfx_volume_effect_load(const char* path, nvfx_volume_effect** out);
NVFX_API nvfx_status nvfx_volume_effect_load_memory(const void* data, size_t bytes, nvfx_volume_effect** out);
NVFX_API void nvfx_volume_effect_free(nvfx_volume_effect* effect);
NVFX_API nvfx_status nvfx_volume_effect_get_info(const nvfx_volume_effect* effect, nvfx_volume_info* info);
/* Name of learned control i (e.g. "intensity"), or NULL. */
NVFX_API const char* nvfx_volume_effect_control_name(const nvfx_volume_effect* effect, int i);

/* Volumes --------------------------------------------------------------------------------------------------------- */
/* `threads`: threads the volume's steps and pictures use, the calling thread included (1: only the caller; 0: one per
 * hardware thread). The volume owns its worker threads. The frames are the same on any number. */
NVFX_API nvfx_status nvfx_volume_create(const nvfx_volume_effect* effect, int threads, nvfx_volume** out);
NVFX_API void nvfx_volume_free(nvfx_volume* volume);
NVFX_API size_t nvfx_volume_scratch_bytes(const nvfx_volume* volume);
/* A change of seed, variation, shard length or controls applies from the next step; the seed, variation and shard
 * length start the timeline again (the next step_to rolls out from the start). */
NVFX_API nvfx_status nvfx_volume_set_controls(nvfx_volume* volume, const float* controls, int count);
NVFX_API nvfx_status nvfx_volume_set_seed(nvfx_volume* volume, uint64_t seed);
/* Play stored start point `start_point` in every shard (-1, the default: chosen by the seed among the nearest). */
NVFX_API nvfx_status nvfx_volume_set_variation(nvfx_volume* volume, int start_point);
/* Seconds per shard (default 8); 0 plays one continuous rollout. */
NVFX_API nvfx_status nvfx_volume_set_shard(nvfx_volume* volume, float seconds);
/* Step to the frame at `time_seconds` (>= 0). No allocation. */
NVFX_API nvfx_status nvfx_volume_step_to(nvfx_volume* volume, double time_seconds);
/* Read a field at the volume's current frame into `out` (bytes >= texels * components * 4 for FLOAT, * 2 for HALF). */
NVFX_API nvfx_status nvfx_volume_read(nvfx_volume* volume, nvfx_volume_field field, nvfx_volume_texel texel, void* out, size_t bytes);
/* Draw the current frame for a camera into premultiplied RGBA8, rows top to bottom, `stride_bytes` apart. */
NVFX_API nvfx_status nvfx_volume_render(nvfx_volume* volume, const nvfx_volume_camera* camera, int width, int height, uint8_t* rgba,
                                        size_t stride_bytes);
NVFX_API nvfx_status nvfx_volume_camera_look_at(const nvfx_volume_look_at* desc, nvfx_volume_camera* out);

#ifdef __cplusplus
}
#endif

#endif /* NEURALVFX_NVFX_VOLUME_H */
