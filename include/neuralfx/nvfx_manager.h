/* NeuralVFX effect manager: many rollout effects in a game frame within a CPU budget (docs/MANAGER.md).
 *
 * The manager owns every playing copy of the scene's rollout effects (instances) and the simulations behind them. Each
 * tick it decides, from the on-screen size and priority of each instance and a CPU budget in milliseconds, which
 * simulations update, at what level of detail (picture size 32, 64, 128 or 256 px; updates at 30, 15, 10 or 7.5 Hz)
 * and on which thread; the frames between updates are drawn by temporal interpolation. Copies with the same effect,
 * controls and seed share one simulation (optionally offset in time); offscreen simulations pause and resume warm;
 * under a tight budget unimportant looping instances are shown from shared proxy simulations.
 *
 * Determinism: every decision depends only on the calls made (sizes, priorities, the tick's dt) and the cost model
 * (nvfx_manager_set_cost), never on measured time, and a simulation's frames do not depend on which thread computes
 * them: the same calls give the same pictures to the bit on any number of threads.
 *
 * Memory: everything is allocated by nvfx_manager_create and nvfx_manager_add_model; spawning, stopping, setting and
 * ticking allocate nothing. Use a manager from one thread at a time (its own workers aside). The pictures returned are
 * premultiplied RGBA8 (blend as in nvfx.h) and stay valid until the next tick.
 */
#ifndef NEURALVFX_NVFX_MANAGER_H
#define NEURALVFX_NVFX_MANAGER_H

#include <neuralfx/nvfx.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NVFX_MANAGER_VERSION 1

typedef struct nvfx_manager nvfx_manager;
typedef uint32_t nvfx_handle; /* an instance; 0 is never a valid handle, and a stopped instance's handle is not reused */

/* Drawing the frames between a simulation's updates. */
typedef enum nvfx_interp {
  NVFX_INTERP_DEFAULT = 0,   /* the measured default (docs/MANAGER.md §5) */
  NVFX_INTERP_HOLD = 1,      /* show the last update until the next (no interpolation) */
  NVFX_INTERP_CROSSFADE = 2, /* blend the two updates around the frame */
  NVFX_INTERP_MOTION = 3     /* warp both along the effect's own coarse flow between them, then blend */
} nvfx_interp;

typedef struct nvfx_manager_desc {
  int max_instances;    /* instances at once (default 256) */
  int threads;          /* threads that run a tick's work, the calling thread included (default 1); 0: the engine runs
                           the work itself on up to job_slots threads (nvfx_manager_tick_begin / run_jobs / tick_end) */
  int job_slots;        /* threads = 0 only: engine threads that may run jobs at once (default 4) */
  float budget_ms;      /* CPU milliseconds per tick for every effect, summed over threads (default 4) */
  float burst;          /* a tick may plan up to burst x budget before due updates wait for the next tick (1.5) */
  nvfx_interp interp;   /* default: NVFX_INTERP_DEFAULT */
  int max_size;         /* largest picture size the planner uses: 32, 64, 128 or 256 (default 256) */
  float texel_scale;    /* picture pixels wanted per on-screen pixel (default 1) */
  float rate_px[3];     /* on-screen sizes in pixels from which 30, 15 and 10 Hz updates are wanted; below the last,
                           7.5 Hz (defaults in docs/MANAGER.md §6) */
  float hysteresis;     /* a level changes only when the size is this fraction past its threshold (default 0.15) */
  float min_dwell;      /* seconds between upward level changes of a simulation (default 1) */
  float catchup_frames; /* simulation steps per tick at most for one simulation (a spawn's warm-up, a pre-roll) (8) */
  int copy_size;        /* picture size of time-offset copies, proxies and fades (default 64) */
  int fade_frames;      /* frames over which an instance crossfades when its picture changes source (default 8) */
  float park_after;     /* seconds offscreen before a paused simulation drops to the smallest size (default 1) */
  float cull_after;     /* seconds offscreen before a paused simulation is released (default 30; 0: never) */
  int proxies;          /* 1 (default): unimportant looping instances may be shown from shared proxy simulations */
  float proxy_dwell;    /* seconds an instance stays a proxy (or its own) before switching back (default 2) */
  uint64_t seed;        /* phases of the simulations' updates and proxies' time offsets (default 1) */
} nvfx_manager_desc;

NVFX_API void nvfx_manager_desc_init(nvfx_manager_desc* desc);
NVFX_API nvfx_status nvfx_manager_create(const nvfx_manager_desc* desc, nvfx_manager** out);
NVFX_API void nvfx_manager_free(nvfx_manager* manager);

/* Models ------------------------------------------------------------------------------------------------------- */
typedef struct nvfx_manager_model_desc {
  int max_sims;       /* simulations of this effect at once, not counting proxies (default 16); an instance that finds
                         none free starts as a proxy, takes one paused longest, or waits */
  int slots[4];       /* simulations at once at 32, 64, 128 and 256 px (-1: the default max_sims, max_sims / 2,
                         max_sims / 4, max_sims / 16, at least 1; 0: never that size). What a level costs in memory is
                         mostly these: docs/MANAGER.md §7 */
  int history;        /* keyframes kept at copy_size by proxies and by simulations with copies offset in time
                         (default 16; 0: none: offsets are ignored and the effect has no proxies) */
  int proxy_sims;     /* proxy simulations at once (default 8; at most proxy_steps ^ controls) */
  int offset_sims;    /* simulations with copies offset in time at once (default 8): their history's rings */
  int proxy_steps;    /* proxy simulations per control: controls are snapped to the centres of this many cells (3) */
  int proxy_size;     /* picture size of proxy simulations (default 64) */
  int proxy_rate_div; /* proxy simulations update every this many frames (default 4: 7.5 Hz) */
  float shard_seconds; /* looping effects: seconds per shard (default 6, as nvfx_instance_set_drift) */
} nvfx_manager_model_desc;

NVFX_API void nvfx_manager_model_desc_init(nvfx_manager_model_desc* desc);
/* A rollout effect (info.arch 3) for the manager's instances; the effect must outlive the manager. Allocates the
 * model's simulations and their buffers. NVFX_ERROR_UNSUPPORTED for frame models. */
NVFX_API nvfx_status nvfx_manager_add_model(nvfx_manager* manager, const nvfx_effect* effect, const nvfx_manager_model_desc* desc,
                                            int* model);
/* The planner's cost model: milliseconds of one simulation step and of one picture at `size` px on one core. Defaults:
 * docs/MANAGER.md §6 (measured on the project's machine). */
NVFX_API nvfx_status nvfx_manager_set_cost(nvfx_manager* manager, int model, int size, float step_ms, float render_ms);
/* Measures the costs on this machine (about `frames` steps and pictures per size; tens of milliseconds) and sets them.
 * A set-up call; the measurement makes the plan depend on this machine's speed (and load) from then on. */
NVFX_API nvfx_status nvfx_manager_calibrate(nvfx_manager* manager, int model, int frames);
NVFX_API nvfx_status nvfx_manager_get_cost(const nvfx_manager* manager, int model, int size, float* step_ms, float* render_ms);

/* Camera: on-screen sizes and visibility of instances placed in the world (nvfx_manager_set_position). */
NVFX_API nvfx_status nvfx_manager_set_camera(nvfx_manager* manager, const float position[3], const float forward[3], const float up[3],
                                             float vertical_fov_radians, float viewport_width_px, float viewport_height_px);
NVFX_API nvfx_status nvfx_manager_set_budget(nvfx_manager* manager, float budget_ms);

/* Instances ---------------------------------------------------------------------------------------------------- */
typedef struct nvfx_spawn_desc {
  int model;
  const float* controls; /* the effect's learned controls (NULL: 0.5 each) */
  int n_controls;
  uint64_t seed;         /* which run plays (as nvfx_instance_set_seed) */
  int share;             /* 1 (default): share a simulation with live instances of the same model, controls and seed */
  float time_offset;     /* seconds this copy plays behind its shared simulation (needs the model's history) */
  int allow_proxy;       /* 1 (default): may be shown from a proxy simulation when the budget is short (looping only) */
  float priority;        /* multiplies the on-screen size in the importance (default 1) */
  float position[3];     /* in the world, with nvfx_manager_set_camera */
  float world_size;      /* the effect's extent in world units (default 1) */
  float screen_px;       /* > 0: the on-screen size in pixels, given directly instead of a position */
  int visible;           /* with screen_px: on screen (default 1) */
} nvfx_spawn_desc;

NVFX_API void nvfx_spawn_desc_init(nvfx_spawn_desc* desc);
/* Starts an instance; its first picture appears after the next tick (a few ticks for effects that warm up, or a
 * pre-roll for time-offset copies). NVFX_ERROR_MEMORY when max_instances are playing. */
NVFX_API nvfx_status nvfx_manager_spawn(nvfx_manager* manager, const nvfx_spawn_desc* desc, nvfx_handle* out);
NVFX_API nvfx_status nvfx_manager_stop(nvfx_manager* manager, nvfx_handle instance);
/* New controls: an instance with its own simulation changes in place (as nvfx_instance_set_controls); a shared one
 * leaves its group with a copy of the simulation's state (no restart). */
NVFX_API nvfx_status nvfx_manager_set_controls(nvfx_manager* manager, nvfx_handle instance, const float* controls, int count);
NVFX_API nvfx_status nvfx_manager_set_position(nvfx_manager* manager, nvfx_handle instance, const float position[3], float world_size);
NVFX_API nvfx_status nvfx_manager_set_screen_size(nvfx_manager* manager, nvfx_handle instance, float pixels, int visible);
NVFX_API nvfx_status nvfx_manager_set_priority(nvfx_manager* manager, nvfx_handle instance, float priority);
/* Pins the instance's simulation at a size (32 to 256) and an update every rate_div frames (1 to 4: 30 to 7.5 Hz),
 * whatever the budget (0, 0: free again). */
NVFX_API nvfx_status nvfx_manager_set_lod_lock(nvfx_manager* manager, nvfx_handle instance, int size, int rate_div);

/* Ticks ---------------------------------------------------------------------------------------------------------- */
/* Advances every playing instance by dt seconds and computes the pictures of this tick on the manager's threads. */
NVFX_API nvfx_status nvfx_manager_tick(nvfx_manager* manager, double dt_seconds);
/* The same in three parts, for engines that run the work on their own job system: tick_begin plans (on the calling
 * thread) and returns the number of jobs; run_jobs(slot) runs jobs until none is left, from up to job_slots threads at
 * once, each with its own slot in [0, job_slots); tick_end (after every run_jobs has returned) finishes the tick. */
NVFX_API nvfx_status nvfx_manager_tick_begin(nvfx_manager* manager, double dt_seconds, int* jobs);
NVFX_API nvfx_status nvfx_manager_run_jobs(nvfx_manager* manager, int slot);
NVFX_API nvfx_status nvfx_manager_tick_end(nvfx_manager* manager);

enum {
  NVFX_PICTURE_SHARED = 1,       /* the simulation is shared with other instances */
  NVFX_PICTURE_PROXY = 2,        /* shown from a proxy simulation */
  NVFX_PICTURE_INTERPOLATED = 4, /* drawn between two updates */
  NVFX_PICTURE_HELD = 8,         /* an update was due and waited (budget), or the instance is paused */
  NVFX_PICTURE_FADING = 16,      /* crossfading from a previous source */
  NVFX_PICTURE_FINISHED = 32,    /* a one-shot effect has reached its last frame */
  NVFX_PICTURE_NEW = 64          /* the pixels changed in this tick */
};

typedef struct nvfx_manager_picture {
  const uint8_t* rgba; /* NULL until the instance's first picture, or while it is culled */
  int size;            /* side in pixels */
  size_t stride;       /* bytes per row */
  int lod_size;        /* the simulation's picture size */
  float lod_rate_hz;   /* the simulation's updates per second (0 while paused) */
  int flags;           /* NVFX_PICTURE_* */
  double effect_time;  /* seconds of the effect shown */
  int visible;
  float screen_px;     /* the on-screen size the planner used */
} nvfx_manager_picture;

NVFX_API nvfx_status nvfx_manager_get_picture(const nvfx_manager* manager, nvfx_handle instance, nvfx_manager_picture* out);

typedef struct nvfx_manager_stats {
  double wall_ms;          /* the last tick, from tick_begin to tick_end */
  double jobs_ms;          /* time spent in its jobs, summed over threads */
  double planned_ms;       /* the cost model's estimate of its jobs */
  double plan_ms;          /* the cost model's mean cost per tick of the levels chosen */
  double budget_ms;
  int instances, visible, finished;
  int sims, sims_paused, proxy_sims;
  int updated;             /* simulations that made a new keyframe */
  int skipped;             /* simulations whose due update waited (budget) */
  int interpolated;        /* pictures drawn between keyframes */
  int held;                /* visible pictures that did not change although their frame did */
  int shared_viewers;      /* instances showing a simulation another instance also shows */
  int proxied;             /* instances shown from proxies */
  int steps, renders;      /* simulation steps and keyframe pictures in the tick */
  int lod[4][4];           /* simulations per [size 32, 64, 128, 256][rate 30, 15, 10, 7.5 Hz] */
  size_t memory_bytes;     /* what the manager allocated (simulations, buffers, threads' scratch) */
  int threads;
} nvfx_manager_stats;

NVFX_API nvfx_status nvfx_manager_get_stats(const nvfx_manager* manager, nvfx_manager_stats* out);

#ifdef __cplusplus
}
#endif

#endif /* NEURALVFX_NVFX_MANAGER_H */
