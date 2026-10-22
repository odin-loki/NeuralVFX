/* NeuralVFX effect packs: one file that holds everything a set of effects needs, for an engine to ship and load at
 * start (docs/INTEGRATION.md §5).
 *
 * A pack (.nvfxp, made by `nvfx_pack pack`) holds the rollout effects' .nvfx models, the effect library's catalogue
 * (.nvfxl files), optionally scene scripts and licence or credit text, and a manifest: the format, a SHA-256 per member,
 * which catalogue entries need which models, and optional default costs per model for the effect manager. Members are
 * compressed by the project's lossless coder when that pays, else stored; opening a pack checks every checksum and
 * decodes every member, which reproduces each byte for byte.
 *
 *   nvfx_pack_desc pd;
 *   nvfx_pack_desc_init(&pd);
 *   pd.path = "effects.nvfxp";                       // or pd.data, pd.bytes: a pack in memory
 *   nvfx_pack* pack;
 *   nvfx_scene_error err;
 *   if (nvfx_pack_open(&pd, &pack, &err) != NVFX_OK) printf("%s\n", err.message);  // "effects.nvfxp: truncated: ..."
 *
 *   nvfx_library* lib;
 *   nvfx_library_open_pack(pack, &lib, &err);        // the catalogue; stages load their models from the pack
 *   nvfx_effect* fire;
 *   nvfx_pack_load_effect(pack, "fire", &fire);      // a model for nvfx_instance_create or the effect manager
 *   nvfx_pack_free(pack);                            // the library keeps what it needs; effects are the caller's
 *
 * Threads: a pack is read-only once open and may be shared between threads. Memory: a pack holds its members' decoded
 * bytes (about 1.6 MB for the eleven models and the catalogue); a library opened from it shares them.
 */
#ifndef NEURALVFX_NVFX_PACK_H
#define NEURALVFX_NVFX_PACK_H

#include "nvfx_library.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NVFX_PACK_FORMAT 1 /* the newest pack format this header's library reads */

typedef struct nvfx_pack nvfx_pack;

typedef enum nvfx_pack_kind {
  NVFX_PACK_MODEL = 0,     /* a .nvfx effect */
  NVFX_PACK_CATALOGUE = 1, /* a catalogue file of the effect library (.nvfxl) */
  NVFX_PACK_SCRIPT = 2,    /* a scene script (.nvfxs) */
  NVFX_PACK_LICENCE = 3,   /* licence or credit text */
  NVFX_PACK_FILE = 4       /* anything else */
} nvfx_pack_kind;

typedef struct nvfx_pack_desc {
  const char* path;  /* a pack file, or NULL: the pack in data */
  const void* data;  /* a pack in memory (bytes long); read during nvfx_pack_open only */
  size_t bytes;
  const char* name;  /* the pack's name in error messages (default: the file's name, or "pack") */
  int threads;       /* threads that decode members, the calling thread included (default 1) */
} nvfx_pack_desc;

typedef struct nvfx_pack_info {
  int format;            /* the file's format version */
  int n_members;
  int n_models;          /* members of kind NVFX_PACK_MODEL */
  int n_catalogue;       /* catalogue files */
  int n_entries;         /* catalogue entries the manifest lists, with the models each needs */
  int n_costs;           /* default costs (model and size) for the effect manager */
  size_t file_bytes;     /* the pack */
  size_t unpacked_bytes; /* its members, decoded */
  const char* licence;   /* the text of its first licence member, or "" (the pack's) */
} nvfx_pack_info;

typedef struct nvfx_pack_member {
  const char* name;      /* "fire.nvfx", "fire.nvfxl", ... (a relative path) */
  nvfx_pack_kind kind;
  int compressed;        /* 1: coded by the lossless coder in the pack; 0: stored */
  const uint8_t* data;   /* its bytes, decoded (the pack's; a NUL follows them, so text members are C strings) */
  size_t bytes;
  size_t stored_bytes;   /* its size in the pack */
  const char* sha256;    /* the SHA-256 of its bytes, 64 hex digits */
} nvfx_pack_member;

NVFX_API void nvfx_pack_desc_init(nvfx_pack_desc* desc);
/* Read a pack and check it: the header, the size (a truncated file is refused), the manifest's SHA-256, every
 * member's SHA-256 before and after decoding, and that every catalogue entry's models are in it. Errors:
 * NVFX_ERROR_IO (the file cannot be read), NVFX_ERROR_FORMAT (not a pack, or damaged: `error` says what, and for the
 * manifest its line), NVFX_ERROR_UNSUPPORTED (a newer format), NVFX_ERROR_ARGUMENT, NVFX_ERROR_MEMORY. */
NVFX_API nvfx_status nvfx_pack_open(const nvfx_pack_desc* desc, nvfx_pack** out, nvfx_scene_error* error);
NVFX_API void nvfx_pack_free(nvfx_pack* pack);
NVFX_API nvfx_status nvfx_pack_get_info(const nvfx_pack* pack, nvfx_pack_info* info);
/* A member by its name ("fire.nvfx"), or a model by its name without folder and extension ("fire"): its index, or -1. */
NVFX_API int nvfx_pack_find(const nvfx_pack* pack, const char* name);
NVFX_API nvfx_status nvfx_pack_get_member(const nvfx_pack* pack, int index, nvfx_pack_member* out);
/* Catalogue entry i of the manifest (i < info.n_entries): its name (or NULL), and the model members it needs:
 * fn(user, name, file) once per model, as nvfx_library_list_models. */
NVFX_API const char* nvfx_pack_entry_name(const nvfx_pack* pack, int i);
NVFX_API nvfx_status nvfx_pack_entry_models(const nvfx_pack* pack, int i, nvfx_scene_effect_fn fn, void* user);

/* Models ----------------------------------------------------------------------------------------------------------- */
/* Load a model member (by name, as nvfx_pack_find) as nvfx_effect_load_memory does; the effect is the caller's
 * (nvfx_effect_free) and does not need the pack afterwards. For instances (nvfx.h), the effect manager
 * (nvfx_manager_add_model) and scenes (nvfx_scene_desc.effects). NVFX_ERROR_IO: no such model in the pack. */
NVFX_API nvfx_status nvfx_pack_load_effect(const nvfx_pack* pack, const char* name, nvfx_effect** out);
/* The pack's default cost of a model at a size (32, 64, 128 or 256) for nvfx_manager_set_cost, when the pack was made
 * with costs (measured on the target machine). NVFX_ERROR_ARGUMENT: none for that model and size. Calibrating at start
 * (nvfx_manager_calibrate) measures the machine the game runs on instead. */
NVFX_API nvfx_status nvfx_pack_model_cost(const nvfx_pack* pack, const char* model, int size, float* step_ms, float* render_ms);

/* The library and scenes ------------------------------------------------------------------------------------------ */
/* Open the effect library from the pack's catalogue files (in name order, as a folder is read); its stages load their
 * models from the pack. The library keeps the members it needs: the pack may be freed before it. Errors as
 * nvfx_library_open, and NVFX_ERROR_FORMAT when an entry needs a model the pack does not hold, NVFX_ERROR_ARGUMENT when
 * the pack has no catalogue. */
NVFX_API nvfx_status nvfx_library_open_pack(const nvfx_pack* pack, nvfx_library** out, nvfx_scene_error* error);
/* nvfx_scene_create with the pack: desc->script_path may name a script member of the pack (else it is a file), and
 * every effect the script names that desc->effects does not pass is loaded from the pack (by file name, or by name
 * without folder and extension), else read from desc->effects_dir. Errors as nvfx_scene_create. */
NVFX_API nvfx_status nvfx_scene_create_from_pack(const nvfx_pack* pack, const nvfx_scene_desc* desc, nvfx_scene** out, nvfx_scene_error* error);

#ifdef __cplusplus
}
#endif

#endif /* NEURALVFX_NVFX_PACK_H */
