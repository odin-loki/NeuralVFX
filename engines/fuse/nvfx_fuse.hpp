// NeuralVFX in FUSE: the sample integration's adapter (docs/FUSE.md). C++23, CPU only, written against the engine
// interface of fuse_engine.hpp and the C++ wrapper over the C API (include/neuralfx/nvfx.hpp, nvfx_volume.h).
//
// One object, made while a level loads, plays four kinds of effect from one effect pack (.nvfxp):
//   sprites  the pack's learned models through the effect manager (nvfx_manager.h): spawned and despawned at run time
//            anywhere in the world, within a CPU budget in milliseconds and a count, each an upright camera-facing
//            quad with its own texture: what replaces a flipbook on a particle or a billboard;
//   scene    a scene script of the pack (nvfx_scene.h), driven by FUSE through its inputs and named rules, its effects
//            (the engine layers' colour, docs/ENGINES.md §11) drawn on a card standing in the world;
//   stage    entries of the pack's effect library (nvfx_library.h: "campfire", "steam_vent", ...) spawned and stopped
//            on a second card, within the rooms the level asked for;
//   volume   a learned 3D effect stored in the pack (nvfx_volume.h), drawn for FUSE's camera as a quad facing it, and
//            optionally given to FUSE's own ray marcher as 3D textures.
//
//   auto fx = nvfx_fuse::System::create(engine, config);           // at load: every allocation, warm-up and texture
//   if (!fx) log(fx.error().message);                              // "effects.nvfxp: member fire.nvfx is damaged ..."
//   auto torch = (*fx)->spawn({.model = 0, .position = {1, 0, 2}, .size = 0.9f});
//   every frame, on the game thread:
//     (*fx)->trigger("blow");                                       // spawns, moves, inputs: between updates
//     (*fx)->update(clock.now(), camera);                           // steps everything on FUSE's jobs; no allocation
//   and FUSE's render thread calls the hook the adapter registered: it uploads what changed and draws the quads.
//
// Threads: create, the destructor and every call here are made on the game thread, never during update(); update runs
// the work as FUSE jobs and returns when they are done. The render hook runs on FUSE's render thread and reads only
// what update() published for the frame it draws (frames_in_flight packets, each holding the newest pixels of
// everything: a frame the render thread skips loses nothing).
// Memory: everything is allocated by create(); update(), the render hook and every call below allocate nothing (errors
// in a frame are kept in fixed storage: last_error()).
#pragma once

#include "fuse_engine.hpp"

#include <neuralfx/nvfx.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string_view>

namespace nvfx_fuse {

using fuse::Vec3;

// --- configuration (load time) -----------------------------------------------------------------------------------------

// A learned model of the pack for sprites.
struct SpriteModel {
  const char* name = nullptr;  // the pack's model, without folder and extension ("fire")
  int max_sims = 8;            // its simulations at once (each holds pool slots: docs/MANAGER.md §2.8); <= 0: the manager's 16
  int precision = NVFX_PRECISION_DEFAULT;  // the orb plays at NVFX_PRECISION_FLOAT (study K's rule, docs/ENGINES.md §5)
  float half_px = -1.f;        // drawn at half size where it pays (docs/MANAGER.md §2.10): < 0 the sprites' threshold
                               // (SpriteConfig::half_px), 0 never for this model, > 0 its own (pixels on screen)
};

enum class Costs : std::uint8_t {
  pack_or_calibrate,  // the pack's costs when it has them, else measured on this machine at load (docs/INTEGRATION.md §6)
  pack_or_default,    // the pack's, else the manager's defaults: plans that never depend on the machine (replays, tests)
};

struct SpriteConfig {
  std::span<const SpriteModel> models;  // none: no sprites
  int max_sprites = 32;                 // sprites at once: the manager's instances, one 256 x 256 texture each
  float budget_ms = 4.f;                // CPU milliseconds per update for every sprite, summed over FUSE's workers
  int job_slots = 2;                    // FUSE workers that may run the manager's jobs at once
  Costs costs = Costs::pack_or_calibrate;
  float texel_scale = 1.f;              // sprite pixels per screen pixel (below 1: softer and cheaper at every distance)
  // Drawn at half size where it pays (docs/MANAGER.md §2.10, FUSE.md §5): a sprite at least half_px pixels on screen
  // (or half_share of the view's height) is drawn at the level below its level. half_px < 0: the manager's default
  // (nvfx_manager_desc_init); 0 with half_share 0: off. half_upscale NVFX_UPSCALE_OFF (the default): the
  // half-size picture is uploaded and FUSE's bilinear sampler draws it on the quad (a quarter of the upload, no CPU for
  // the 2x); NVFX_UPSCALE_BILINEAR: the manager's bilinear 2x on the CPU first, the picture at its level's size.
  float half_px = -1.f;
  float half_share = 0.f;
  nvfx_upscale half_upscale = NVFX_UPSCALE_OFF;
  float edge_fade = 0.08f;              // soft edges of the sprites' quads (fuse::Quad::edge): plumes that touch their picture's border
};

// Where a 2D scene stands in the world: a card whose picture is the scene's world pixels.
struct Card {
  Vec3 base;                             // the middle of the scene's ground line, in the world
  Vec3 right = {1.f, 0.f, 0.f};          // the card's axes: unit vectors at right angles (the card faces right x up)
  Vec3 up = {0.f, 1.f, 0.f};
  float units_per_px = 0.01f;            // world units per scene pixel
};

struct SceneConfig {
  const char* script = nullptr;                // a script member of the pack ("campfire.nvfxs"); nullptr: no scene
  std::span<const nvfx_scene_input> inputs;    // starting values of its inputs
  int width = 0, height = 0;                   // its texture (the scene is drawn at its script's size and resampled); 0: that size
  int precision = 0;                           // its learned effects' nvfx_precision (0: the runtime's)
  Card card;
};

struct StageConfig {
  std::span<const nvfx_stage_room> rooms;      // the library entries it may play and how many at once; empty: no stage
  int width = 640, height = 360;               // its picture (world pixels at scale 1) and texture
  int precision = 0;
  Card card;
};

struct VolumeConfig {
  const char* member = nullptr;           // a 3D effect (.nvfx3) stored in the pack; nullptr: no volume
  Vec3 position;                          // the centre of its floor in the world
  float scale = 1.f;                      // world units per volume unit (its box is nvfx_volume_info.extent units)
  std::array<float, 3> controls = {0.7f, 0.5f, 0.6f};  // intensity, wind (0.5 none; keep it in 0.4 to 0.8), turbulence
  std::uint64_t seed = 1;
  float shard_seconds = 8.f;              // 0: one continuous rollout
  int threads = 1;                        // the volume's own threads, the FUSE job's included (above 1 it runs threads of
                                          // its own inside that job; nvfx_volume_create_with_jobs is not used: the volume
                                          // steps inside a FUSE job, from which JobSystem::run may not be called, FUSE.md §9)
  bool picture = true;                    // drawn on the CPU for FUSE's camera, shown as a quad facing the camera
  bool fields = false;                    // its temperature and extinction as 3D textures (half floats) for FUSE's ray marcher
  float picture_scale = 0.5f;             // picture pixels per screen pixel
  int max_picture = 320;                  // the picture's largest side in pixels (its texture's size)
};

struct Config {
  const char* pack_path = nullptr;        // the pack, or pack_bytes (read during create only)
  std::span<const std::uint8_t> pack_bytes;
  int load_threads = 2;                   // threads that decode the pack
  int frames_in_flight = 2;               // packets between the game thread and the render thread (>= 2)
  SpriteConfig sprites;
  SceneConfig scene;
  StageConfig stage;
  VolumeConfig volume;
};

// --- run time ----------------------------------------------------------------------------------------------------------

struct SpriteDesc {
  int model = 0;                          // an index in Config::sprites.models
  Vec3 position;                          // its base (anchor_base) or its centre
  float size = 1.f;                       // world units across (the quad is size x size)
  bool anchor_base = true;                // a fire stands on the ground; an explosion is centred in the air
  std::array<float, 3> controls = {0.5f, 0.5f, 0.5f};  // the model's learned controls (intensity, wind, turbulence, ...)
  std::uint64_t seed = 1;
  bool share = true;                      // copies of the same model, controls and seed share one simulation
  float time_offset = 0.f;                // seconds this copy plays behind its shared simulation
  bool allow_proxy = true;                // may be shown from a shared proxy simulation when the budget is short
  float priority = 1.f;
  bool despawn_when_finished = true;      // a one-shot (an explosion) is despawned when its last frame has shown
};

// A sprite: 0 is none; a despawned sprite's handle is never valid again.
struct Sprite {
  std::uint32_t id = 0;
  explicit operator bool() const noexcept { return id != 0; }
};

// An entry playing on the stage: 0 is none.
struct StageFx {
  std::uint32_t id = 0;
  explicit operator bool() const noexcept { return id != 0; }
};

// An error in a frame, kept without allocating (the frame goes on without the part that failed).
struct FrameError {
  nvfx_status status = NVFX_OK;
  std::uint64_t frame = 0;
  char what[192] = {};
};

struct Stats {
  std::uint64_t frame = 0;                // the last frame updated
  int sprites = 0, sprites_drawn = 0;     // live sprites; quads drawn for them in the last frame
  nvfx_manager_stats manager{};           // the last tick (planned_ms, jobs_ms, sims, proxied, ...)
  double update_ms = 0;                   // the last update(), wall clock, on the game thread (jobs included)
  double volume_ms = 0, scene_ms = 0, stage_ms = 0;  // their jobs in the last update (0 when nothing was due)
  double volume_step_ms = 0, volume_picture_ms = 0, volume_fields_ms = 0;  // the volume's job: its step, picture and reads
  std::size_t copied_bytes = 0;           // pixels copied into the frame's packet by the last update
  std::uint64_t uploaded_bytes = 0;       // texels given to GpuUpload::update by the render hook, in all (read when stats() is called)
  std::uint64_t frames_drawn = 0;         // render hooks run
  std::uint64_t errors = 0;               // frame errors in all (last_error() has the last)
  std::size_t memory_bytes = 0;           // what this level holds on the CPU, all of it allocated by create():
  struct Memory {
    std::size_t models = 0;               // the sprites' models (nvfx_effect_info.resident_bytes)
    std::size_t manager = 0;              // the manager's pools and threads' scratch (nvfx_manager_stats.memory_bytes, from the first update)
    std::size_t scenes = 0;               // the scene and the stage (nvfx_scene_info.scratch_bytes and their pictures)
    std::size_t volume = 0;               // the volume (nvfx_volume_scratch_bytes, its model, its picture and fields)
    std::size_t packets = 0;              // the frames_in_flight packets for the render thread
  } memory;
  std::size_t texture_bytes = 0;          // what the textures made at load hold on FUSE's side
};

class System {
 public:
  // At load, on the game thread: checks libnvfx against the headers and its features, opens and checks the pack, loads
  // the models, makes the manager (costs from the pack, else calibrated or the defaults), the scene, the stage and the
  // volume (warm-ups included), creates every texture and registers the render hook. The pack is not kept.
  static std::expected<std::unique_ptr<System>, nvfx::Error> create(const fuse::Engine& engine, const Config& config);
  ~System();
  System(const System&) = delete;
  System& operator=(const System&) = delete;

  // Sprites (the effect manager). spawn returns {} when max_sprites play or the manager refuses (last_error says).
  int model(std::string_view name) const noexcept;  // an index in Config::sprites.models, or -1
  Sprite spawn(const SpriteDesc& desc) noexcept;
  bool despawn(Sprite sprite) noexcept;
  bool move(Sprite sprite, Vec3 position) noexcept;
  bool set_controls(Sprite sprite, std::span<const float> controls) noexcept;  // a shared one leaves its group
  bool alive(Sprite sprite) const noexcept;

  // The scripted scene: its inputs and named rules (act from the next frame it computes).
  bool set_input(const char* name, float value) noexcept;
  bool trigger(const char* rule) noexcept;
  nvfx::SceneView scene() noexcept;  // everything else (fields, modules): between updates only; empty without a scene

  // The stage: library entries spawned at a point of the world (projected onto its card) and stopped.
  StageFx spawn_entry(const char* entry, Vec3 at, float scale = 1.f) noexcept;
  bool stop_entry(StageFx fx) noexcept;
  bool set_entry_param(StageFx fx, const char* param, float value) noexcept;
  nvfx::SceneView stage_scene() noexcept;

  // Gameplay: the heat at a point of the world, from the scene's and the stage's field buses, where the point is on a
  // card (within half a world unit of its plane); 0 elsewhere. The frame last computed.
  float heat_at(Vec3 p) const noexcept;

  // The volume.
  bool set_volume_controls(std::span<const float> controls) noexcept;
  void place_volume(Vec3 position, float scale) noexcept;

  // Once per game frame, on the game thread: steps every effect to the frame's time on FUSE's jobs (the volume, the
  // scene, the stage and the manager's jobs together), and publishes the frame's quads and new pixels for the render
  // hook. Waits only if the render thread is frames_in_flight - 1 frames behind. No allocation.
  void update(const fuse::FrameTime& time, const fuse::View& view) noexcept;

  const Stats& stats() const noexcept;
  const FrameError& last_error() const noexcept;

  struct Impl;

 private:
  explicit System(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nvfx_fuse
