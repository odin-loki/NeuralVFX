/* NeuralVFX effect library: a catalogue of named, ready-to-use effects that a game lists, spawns, drives and stops
 * (docs/LIBRARY.md). Built on scenes (nvfx_scene.h): a stage is a scene with room for instances of chosen effects.
 *
 * A library is a catalogue (a folder of .nvfxl files, or their texts) and the folder of the learned effects it plays.
 * Opening it reads and checks the catalogue only; a stage loads the effects its rooms need.
 *
 *   nvfx_library_desc ld;
 *   nvfx_library_desc_init(&ld);
 *   ld.catalogue_dir = "library/";                   // the catalogue
 *   ld.models_dir = "effects/";                      // fire.nvfx, smoke.nvfx, ...
 *   nvfx_library* lib;
 *   nvfx_scene_error err;
 *   if (nvfx_library_open(&ld, &lib, &err) != NVFX_OK) printf("%s\n", err.message);   // "fire.nvfxl:12:7: ..."
 *   for (int i = 0; i < nvfx_library_count(lib); ++i) { nvfx_library_effect e; nvfx_library_get_effect(lib, i, &e); ... }
 *
 *   nvfx_stage_room rooms[] = {{"campfire", 4, 0}, {"grenade", 8, 1}};   // up to 4 campfires and 8 grenades at once
 *   nvfx_stage_desc sd;
 *   nvfx_stage_desc_init(&sd);
 *   sd.rooms = rooms;
 *   sd.n_rooms = 2;
 *   nvfx_stage* stage;
 *   nvfx_stage_create(lib, &sd, &stage, &err);       // builds the scene, warms the learned effects up
 *   nvfx_scene* scene = nvfx_stage_scene(stage);     // step, render and read fields with nvfx_scene_*
 *
 *   nvfx_fx fire;
 *   nvfx_stage_spawn(stage, "campfire", 300, 470, 1.0f, -1.0, &fire);   // at (300, 470), scale 1, now
 *   nvfx_stage_set_param(stage, fire, "intensity", 0.4f);
 *   every game frame: nvfx_scene_step(scene, dt, NULL); nvfx_scene_render(scene, rgba, stride);
 *   nvfx_stage_stop(stage, fire, -1.0);
 *
 * Spawning, moving, setting parameters and stopping set the scene's inputs: they act from the next frame computed
 * (with overlap, as nvfx_scene_render says, the frame after), or at the time given. They allocate nothing, and a
 * stage's frames allocate nothing: everything is made by nvfx_stage_create. The same calls give the same frames.
 *
 * Threads: a library is read-only once open and may be shared between threads; a stage (and its scene) is used from
 * one thread at a time, and runs desc.threads threads of its own, as a scene does.
 */
#ifndef NEURALVFX_NVFX_LIBRARY_H
#define NEURALVFX_NVFX_LIBRARY_H

#include "nvfx_scene.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nvfx_library nvfx_library;
typedef struct nvfx_stage nvfx_stage;
/* An instance on a stage; 0 is none. A handle goes stale when its room is used by a later spawn. */
typedef uint32_t nvfx_fx;

/* A catalogue file in memory. */
typedef struct nvfx_library_file {
  const char* name;  /* for error messages: "fire.nvfxl" */
  const char* text;  /* NUL-terminated */
} nvfx_library_file;

typedef struct nvfx_library_desc {
  const char* catalogue_dir;      /* every *.nvfxl in it, in name order (or NULL: `files`) */
  const nvfx_library_file* files; /* catalogue files in memory, in this order (n_files of them) */
  int n_files;
  const char* models_dir;         /* where a stage reads the learned effects' files (default: catalogue_dir; files
                                     named with absolute paths are read as they are) */
} nvfx_library_desc;

NVFX_API void nvfx_library_desc_init(nvfx_library_desc* desc);
/* Read and check the catalogue (every entry, without loading any effect). Errors: NVFX_ERROR_SCRIPT (the catalogue is
 * wrong: `error` has the file, line and column), NVFX_ERROR_IO, NVFX_ERROR_ARGUMENT, NVFX_ERROR_MEMORY. */
NVFX_API nvfx_status nvfx_library_open(const nvfx_library_desc* desc, nvfx_library** out, nvfx_scene_error* error);
NVFX_API void nvfx_library_free(nvfx_library* library);

typedef enum nvfx_anchor {
  NVFX_ANCHOR_BASE = 0,   /* the spawn point is the bottom centre of the effect's box (a fire on the ground) */
  NVFX_ANCHOR_CENTRE = 1, /* the spawn point is the box's centre (a burst in the air, a portal) */
  NVFX_ANCHOR_LEFT = 2    /* the spawn point is the middle of the box's left edge (a jet's nozzle: it blows right) */
} nvfx_anchor;

/* An effect of the catalogue. The strings belong to the library. */
typedef struct nvfx_library_effect {
  const char* name;        /* "campfire" */
  const char* category;    /* "fire", "smoke", "explosions", "steam", "magic", "ambient", "sparks", "light", ... */
  const char* description; /* one line */
  const char* models;      /* the learned effects it plays, as the catalogue names them ("fire, smoke"; "" if none) */
  const char* elements;    /* the procedural elements it uses ("embers, lamp, look coals"; "" if none) */
  const char* file;        /* the catalogue file it is in */
  float width, height;     /* its box at scale 1, world pixels */
  nvfx_anchor anchor;
  int looping;             /* 1: plays until stopped; 0: a one-shot of `duration` seconds, then its room is free */
  float duration;
  float show;              /* a moment that shows it well, seconds after its spawn */
  int n_params;
} nvfx_library_effect;

/* A parameter of an effect: a value the game sets per instance (intensity, wind, colour, ...). */
typedef struct nvfx_library_param {
  const char* name;
  const char* description;
  float value;             /* its default */
  float min, max;          /* its range: values set are clamped to it */
} nvfx_library_param;

NVFX_API int nvfx_library_count(const nvfx_library* library);
NVFX_API int nvfx_library_find(const nvfx_library* library, const char* name); /* its index, or -1 */
NVFX_API nvfx_status nvfx_library_get_effect(const nvfx_library* library, int index, nvfx_library_effect* out);
NVFX_API nvfx_status nvfx_library_get_param(const nvfx_library* library, int effect, int param, nvfx_library_param* out);
NVFX_API int nvfx_library_category_count(const nvfx_library* library);
NVFX_API const char* nvfx_library_category(const nvfx_library* library, int i); /* i < count, else NULL */
/* The files of the learned effects an effect plays (for hosts that read files themselves and pass them in
 * nvfx_stage_desc.effects): fn(user, name, file) once per effect, as nvfx_scene_list_effects does. */
NVFX_API nvfx_status nvfx_library_list_models(const nvfx_library* library, int effect, nvfx_scene_effect_fn fn, void* user);

/* Stages ------------------------------------------------------------------------------------------------------------ */

/* How many instances of an effect may play at once on a stage. With `steal`, a spawn when all are playing replaces
 * the oldest; without, it fails (NVFX_ERROR_MEMORY: no room). */
typedef struct nvfx_stage_room {
  const char* effect;
  int count;
  int steal;
} nvfx_stage_room;

typedef struct nvfx_stage_desc {
  int width, height;          /* the picture (default 960 x 540): the world from the camera's corner */
  float fps;                  /* the scene's clock (default 30) */
  float ground;               /* world y of the ground (default: height * 0.87) */
  float bus_x, bus_y, bus_width, bus_height; /* where effects play and the fields are kept, world pixels (default, all 0:
                                                the picture with an eighth of its width each side and half its height
                                                above); effects outside are drawn but not on the bus */
  const nvfx_stage_room* rooms;
  int n_rooms;
  int particles;              /* sparks, embers, debris and glows at once (default 8192) */
  float light_gain, exposure, haze, bloom; /* the stage's light and picture (default 0.14, 1.2, 1, 1) */
  int threads, overlap;       /* as nvfx_scene_desc (default 2, -1) */
  const nvfx_scene_effect* effects; /* learned effects passed in (by their file's name), instead of read */
  int n_effects;
  int precision;              /* the learned effects' nvfx_precision, as nvfx_scene_desc's (default 0: the runtime's) */
} nvfx_stage_desc;

typedef enum nvfx_fx_state {
  NVFX_FX_NONE = 0,     /* not an instance (or a stale handle) */
  NVFX_FX_WAITING = 1,  /* spawned for a time not reached yet */
  NVFX_FX_PLAYING = 2,
  NVFX_FX_ENDED = 3     /* stopped, or a one-shot that has played: its room is free */
} nvfx_fx_state;

typedef struct nvfx_stage_fx_info {
  nvfx_fx_state state;
  int effect;               /* its index in the library */
  float x, y, scale;        /* where it is and its scale */
  float spawned;            /* scene time it spawned (INFINITY: not yet) */
  float age;                /* seconds since (0 before) */
} nvfx_stage_fx_info;

NVFX_API void nvfx_stage_desc_init(nvfx_stage_desc* desc);
/* Build the stage's scene (one scene script with a room per instance) and load the learned effects it needs; the
 * effects with start points warm up here. Errors as nvfx_scene_create, and NVFX_ERROR_ARGUMENT for an unknown effect
 * in a room. */
NVFX_API nvfx_status nvfx_stage_create(const nvfx_library* library, const nvfx_stage_desc* desc, nvfx_stage** out, nvfx_scene_error* error);
NVFX_API void nvfx_stage_free(nvfx_stage* stage);
/* The stage's scene, owned by the stage: nvfx_scene_step, _render, _sample and the rest. Do not restart it or seek it
 * backwards (nvfx_stage_clear does that). */
NVFX_API nvfx_scene* nvfx_stage_scene(nvfx_stage* stage);
/* The stage's scene script (what nvfx_scene_script and the viewer can play): writes up to `capacity` bytes with a NUL
 * and returns the length the whole script needs (without its NUL). */
NVFX_API size_t nvfx_stage_script(const nvfx_stage* stage, char* out, size_t capacity);

/* Spawn an instance of `effect` with its anchor at world (x, y), at `scale` (1: its catalogue size; its learned
 * effects keep their pixels, so far above 1 they blur), at scene time `at` (< 0: now). Its parameters take their
 * defaults. NVFX_ERROR_MEMORY: no free room for it (and no `steal`). */
NVFX_API nvfx_status nvfx_stage_spawn(nvfx_stage* stage, const char* effect, float x, float y, float scale, double at, nvfx_fx* out);
NVFX_API nvfx_status nvfx_stage_set_param(nvfx_stage* stage, nvfx_fx fx, const char* param, float value);
NVFX_API nvfx_status nvfx_stage_get_param(const nvfx_stage* stage, nvfx_fx fx, const char* param, float* value);
NVFX_API nvfx_status nvfx_stage_move(nvfx_stage* stage, nvfx_fx fx, float x, float y);
/* Stop it at scene time `at` (< 0: now): its learned effects stop and it emits nothing more (particles already out
 * fly on). A spawn that has not happened yet is cancelled. */
NVFX_API nvfx_status nvfx_stage_stop(nvfx_stage* stage, nvfx_fx fx, double at);
NVFX_API nvfx_status nvfx_stage_get_fx(const nvfx_stage* stage, nvfx_fx fx, nvfx_stage_fx_info* out);
/* Rooms of an effect free for a spawn now (-1: the stage has no room for it). */
NVFX_API int nvfx_stage_free_rooms(const nvfx_stage* stage, const char* effect);
/* The camera: the world position of the picture's top-left corner (shakes of instances are added). */
NVFX_API nvfx_status nvfx_stage_set_camera(nvfx_stage* stage, float x, float y);
/* Every instance gone and the scene back at time 0 (rebuilt: allocates, as nvfx_scene_restart). Handles go stale. */
NVFX_API nvfx_status nvfx_stage_clear(nvfx_stage* stage);

#ifdef __cplusplus
}
#endif

#endif /* NEURALVFX_NVFX_LIBRARY_H */
