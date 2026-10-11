/* NeuralVFX scenes: composed effects played from a scene script (docs/COMPOSE.md §4), for game engines
 * (docs/ENGINES.md §7).
 *
 * A scene (nvfx_scene) is a script (.nvfxs) with the rollout effects it names: modules that run on one another
 * through their fields, force fields, particles, light and rules that change the scene over time. It draws a finished,
 * opaque picture (its own sky and ground) and keeps a field bus that the game can read (is this tile on fire?).
 *
 *   nvfx_scene_desc d;
 *   nvfx_scene_desc_init(&d);
 *   d.script_path = "fireball.nvfxs";
 *   d.effects_dir = "effects/";                     // or pass loaded effects in d.effects
 *   nvfx_scene_error err;
 *   nvfx_scene* scene;
 *   if (nvfx_scene_create(&d, &scene, &err) != NVFX_OK) printf("%s\n", err.message);   // "fireball.nvfxs:12:7: ..."
 *
 *   every game frame:
 *     nvfx_scene_step(scene, dt, NULL);              // the scene's clock: computes the frames whose time has come
 *     nvfx_scene_sample(scene, x, y, &s);            // gameplay: s.heat > 0.1 means fire here
 *     nvfx_scene_render(scene, rgba, stride);        // the picture of the current frame (copied if already drawn)
 *
 * Threads: a scene is used from one thread at a time, like an instance; it runs `threads` threads of its own (the
 * calling thread is one of them). Scenes are independent of one another. Effects passed in are copied at creation.
 * Memory: everything is allocated by nvfx_scene_create; nvfx_scene_step, nvfx_scene_render, the field reads and the
 * run-time settings allocate nothing. nvfx_scene_restart and seeking backwards rebuild the scene (they allocate and
 * cost what creation costs, without reading files again).
 */
#ifndef NEURALVFX_NVFX_SCENE_H
#define NEURALVFX_NVFX_SCENE_H

#include "nvfx.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nvfx_scene nvfx_scene;

/* A loaded effect for the scene, instead of reading its file: `name` is the file as the script writes it
 * (effect fire = "fire.nvfx" -> "fire.nvfx"), or that file's name without folder and extension ("fire"). It must be a
 * rollout effect (nvfx_effect_info.arch 3). It is copied: the caller may free it after nvfx_scene_create. */
typedef struct nvfx_scene_effect {
  const char* name;
  const nvfx_effect* effect;
} nvfx_scene_effect;

/* A starting value for one of the script's inputs (`input NAME = value`). */
typedef struct nvfx_scene_input {
  const char* name;
  float value;
} nvfx_scene_input;

typedef struct nvfx_scene_desc {
  const char* script;       /* the script's text (NUL-terminated), or NULL to read script_path */
  const char* script_path;  /* a .nvfxs file (when script is NULL) */
  const char* source_name;  /* the script's name in error messages (default: the file's name, or "script") */
  const char* effects_dir;  /* where the effects' files are (default: the script file's folder, else the current one);
                               absolute paths in the script are used as they are */
  const nvfx_scene_effect* effects; /* effects passed in (n_effects of them); the others are read from effects_dir */
  int n_effects;
  const nvfx_scene_input* inputs;   /* starting values of inputs (n_inputs of them) */
  int n_inputs;
  int threads;              /* threads the scene runs on, the caller's included (default 2; 1: all on the caller) */
  int overlap;              /* 1: each picture is drawn on a thread of its own while the next frame's state is computed
                               (the same pictures, faster; the state runs a frame ahead of the picture, see
                               nvfx_scene_render); 0: off; -1 (default): on with 2 threads or more */
  int width, height;        /* the output's size in pixels (0: the script's `scene size`). The scene is drawn at its
                               own size and resampled (bilinear) when they differ: this does not make it cheaper */
} nvfx_scene_desc;

/* Where and why a scene could not be created. */
typedef struct nvfx_scene_error {
  int line, column;   /* in the script, 1-based (0: not about a line of the script) */
  char message[1024]; /* "name:line:column: message" for script errors, else the message alone (NUL-terminated) */
} nvfx_scene_error;

typedef struct nvfx_scene_info {
  int width, height;             /* the output (nvfx_scene_render writes width x height RGBA) */
  int scene_width, scene_height; /* the picture as the script draws it */
  float fps;                     /* frames per second of the scene's clock */
  float length;                  /* seconds (the script's `length`; the scene may play on after it) */
  int frames;                    /* length * fps */
  float ground;                  /* world y of the ground */
  float bus_x, bus_y, bus_cell;  /* the field bus: nx x ny cells of bus_cell world pixels from (bus_x, bus_y) */
  int bus_nx, bus_ny;
  int n_modules, n_rules, n_inputs;
  int threads, overlap;
  size_t scratch_bytes;          /* the modules' working memory */
} nvfx_scene_info;

/* Fields at a point of the world (world pixels, y down). */
typedef struct nvfx_scene_fields {
  float u, v;      /* velocity, world pixels per second (y down) */
  float heat;      /* the simulation's heat (0: cold; a fire's flames are about 0.2 to 2) */
  float soot;      /* the simulation's soot (smoke) */
} nvfx_scene_fields;

typedef enum nvfx_scene_field { NVFX_FIELD_HEAT = 0, NVFX_FIELD_SOOT = 1, NVFX_FIELD_U = 2, NVFX_FIELD_V = 3 } nvfx_scene_field;

typedef struct nvfx_scene_module_info {
  float x, y;       /* where it stands: the centre of its tile (or domain of tiles) and the y it stands on */
  float width;      /* world pixels its tile (or domain) covers across */
  float started;    /* scene time it last started, woke or took over (INFINITY: not yet) */
  int active;       /* 1: stepped, drawn and published to the bus */
  int tiles;        /* 1, or the tiles of a domain */
  int n_controls;   /* the effect's learned controls (names: nvfx_scene_module_control_name) */
  float controls[8];/* their values (the first tile's; the first 8) */
} nvfx_scene_module_info;

/* Creation ------------------------------------------------------------------------------------------------------- */
NVFX_API void nvfx_scene_desc_init(nvfx_scene_desc* desc);
/* Parse and check the script, load its effects, build every module and buffer, and compute frame 0's state (its
 * modules' warm-ups run here). Errors: NVFX_ERROR_SCRIPT (the script is wrong: `error` has the line and column),
 * NVFX_ERROR_IO (the script or an effect's file cannot be read; for an effect, the line of its `effect` statement),
 * NVFX_ERROR_FORMAT (not a rollout effect), NVFX_ERROR_ARGUMENT, NVFX_ERROR_MEMORY. `error` may be NULL. */
NVFX_API nvfx_status nvfx_scene_create(const nvfx_scene_desc* desc, nvfx_scene** out, nvfx_scene_error* error);
/* Parse and check a script without loading any effect (an editor's quick loop): NVFX_OK or NVFX_ERROR_SCRIPT. */
NVFX_API nvfx_status nvfx_scene_check(const char* script, const char* source_name, nvfx_scene_error* error);
/* The effects a script names, for hosts that read the files themselves (from a package, say) and pass them in
 * nvfx_scene_desc.effects: fn(user, name, file) once per `effect NAME = "FILE"` statement, in order. The script is
 * parsed, not checked: NVFX_OK or NVFX_ERROR_SCRIPT. */
typedef void (*nvfx_scene_effect_fn)(void* user, const char* name, const char* file);
NVFX_API nvfx_status nvfx_scene_list_effects(const char* script, const char* source_name, nvfx_scene_effect_fn fn, void* user, nvfx_scene_error* error);
NVFX_API void nvfx_scene_free(nvfx_scene* scene);
NVFX_API nvfx_status nvfx_scene_get_info(const nvfx_scene* scene, nvfx_scene_info* info);

/* The clock ------------------------------------------------------------------------------------------------------
 * The scene runs at its script's fps; frame f is at f / fps seconds. A new scene is at frame 0, computed. */
/* Move the clock forward by dt seconds (>= 0) and compute the state of every frame whose time has come (rules,
 * modules, couplings, bus, light, particles), without drawing them. `frames` (may be NULL): how many were computed. */
NVFX_API nvfx_status nvfx_scene_step(nvfx_scene* scene, double dt_seconds, int* frames);
/* Move forward by whole frames (the clock to the start of the frame). */
NVFX_API nvfx_status nvfx_scene_step_frames(nvfx_scene* scene, int frames);
/* Go to a time: forwards, as nvfx_scene_step; backwards, the scene is rebuilt (nvfx_scene_restart) and played to it. */
NVFX_API nvfx_status nvfx_scene_seek(nvfx_scene* scene, double seconds);
/* Back to frame 0: the scene is rebuilt from its script and effects (no files are read), with the inputs' current
 * values as their starting values. Module moves and controls set by the game are not kept. */
NVFX_API nvfx_status nvfx_scene_restart(nvfx_scene* scene);
NVFX_API double nvfx_scene_time(const nvfx_scene* scene);  /* seconds */
NVFX_API int nvfx_scene_frame(const nvfx_scene* scene);    /* floor(time * fps) */

/* The picture ------------------------------------------------------------------------------------------------------
 * The current frame into `rgba` (info.width x info.height, rows top to bottom, `stride_bytes` apart, alpha 255):
 * drawn if it has not been, else copied. With overlap, drawing frame f also computes frame f + 1's state (on the
 * calling thread, while the picture is drawn on another): field reads then see frame f + 1, and inputs, triggers and
 * module settings given after this call act from frame f + 2. Without overlap they act from the next frame. */
NVFX_API nvfx_status nvfx_scene_render(nvfx_scene* scene, uint8_t* rgba, size_t stride_bytes);

/* Fields, for gameplay ----------------------------------------------------------------------------------------------
 * The field bus of the last frame computed: every active module's velocity, heat and soot in world space, cells of
 * info.bus_cell pixels, sampled bilinearly. Zero outside the bus. */
NVFX_API nvfx_status nvfx_scene_sample(const nvfx_scene* scene, float x, float y, nvfx_scene_fields* out);
/* A grid of samples: out[j * row_stride + i] = the field at (x0 + i * dx, y0 + j * dy), i < nx, j < ny
 * (row_stride in floats, >= nx). For a tile map, x0, y0 the first tile's centre and dx, dy the tile size. */
NVFX_API nvfx_status nvfx_scene_sample_grid(const nvfx_scene* scene, nvfx_scene_field field, float x0, float y0, float dx, float dy,
                                            int nx, int ny, float* out, size_t row_stride);
/* The largest and the mean value of a field over the bus cells whose centres are in [x0, x1) x [y0, y1) (world
 * pixels): "is anything in this rectangle burning?". Both 0 when no cell is inside. Either pointer may be NULL. */
NVFX_API nvfx_status nvfx_scene_field_region(const nvfx_scene* scene, nvfx_scene_field field, float x0, float y0, float x1, float y1,
                                             float* max, float* mean);

/* Run-time settings -------------------------------------------------------------------------------------------------
 * Names are those of the script. Each acts from the next frame computed (see nvfx_scene_render). */
/* Inputs: values the script reads like `t` (declared with `input NAME = value`). */
NVFX_API nvfx_status nvfx_scene_set_input(nvfx_scene* scene, const char* name, float value);
NVFX_API nvfx_status nvfx_scene_get_input(const nvfx_scene* scene, const char* name, float* value);
NVFX_API const char* nvfx_scene_input_name(const nvfx_scene* scene, int i); /* i < n_inputs, else NULL */
/* Fire a rule by its name (`as NAME`) at the start of the next frame, as if its condition held, if it has firings
 * left (`at most N`; `repeat`: always). A rule the game alone fires can be written `when 0 as NAME:`. Landing rules
 * cannot be triggered (NVFX_ERROR_ARGUMENT). */
NVFX_API nvfx_status nvfx_scene_trigger(nvfx_scene* scene, const char* rule);
/* How often a rule has fired, and when it last did (INFINITY: never). Either pointer may be NULL. */
NVFX_API nvfx_status nvfx_scene_rule_state(const nvfx_scene* scene, const char* rule, int* count, float* last_time);
NVFX_API const char* nvfx_scene_rule_name(const nvfx_scene* scene, int i); /* i < n_rules, else NULL */
/* Modules (a domain of tiles is one): where they are, and the game moving them or setting their controls. A module
 * whose place or control the script changes over time is set by the script again in the next frame; tiles cannot
 * be moved. */
NVFX_API const char* nvfx_scene_module_name(const nvfx_scene* scene, int i); /* i < n_modules, else NULL */
NVFX_API nvfx_status nvfx_scene_module_get_info(const nvfx_scene* scene, const char* module, nvfx_scene_module_info* info);
NVFX_API const char* nvfx_scene_module_control_name(const nvfx_scene* scene, const char* module, int i);
NVFX_API nvfx_status nvfx_scene_module_place(nvfx_scene* scene, const char* module, float x, float y);
NVFX_API nvfx_status nvfx_scene_module_set_control(nvfx_scene* scene, const char* module, const char* control, float value);

#ifdef __cplusplus
}
#endif

#endif /* NEURALVFX_NVFX_SCENE_H */
