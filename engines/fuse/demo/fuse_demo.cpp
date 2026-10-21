// nvfx_fuse_demo: NeuralVFX in a stand-in FUSE, headless (docs/FUSE.md §8).
//
//   nvfx_fuse_demo --pack fuse_demo.nvfxp [--out DIR] [--video FILE.mp4] [--seconds 10] [--workers 3] [--calibrate]
//   nvfx_fuse_demo --pack fuse_demo.nvfxp --check [--seconds 10] [--workers 3]
//   nvfx_fuse_demo --self-test PACK      PACK: a pack of demo/selftest (procedural entries only: runs without models)
//
// Plays a 10 s sequence at 60 Hz through the adapter (engines/fuse/nvfx_fuse.hpp): the pack's learned models as sprites
// through the effect manager (torches sharing one simulation offset in time, smoke columns whose wind the game sets,
// two explosions despawned when they finish, a magic portal), the campfire scene script driven by its inputs and named
// rules (a gust, a log catching, the game's wind, doused and relit), library entries on a stage (a brazier, a steam vent,
// an electric arc, a magic burst and a lightning strike), and the learned 3D fire volume, all from one effect pack. The
// stand-in (standin.hpp) runs the jobs on a thread pool and draws every frame on a render thread of its own.
//
// --out writes PNG keyframes (at 1, 3, 5, 7 and 9 s); --video pipes every other frame (30 fps) to ffmpeg (H.264,
// yuv420p, +faststart). --check plays the sequence twice, on `--workers` worker threads and then on the calling thread
// alone, and checks: no heap allocation in any frame (game thread, workers, render thread), the same frames to the bit
// (a hash of each), no frame error, and that the parts did their work (heat on the campfire's card, sprites drawn, the
// volume's picture and fields uploaded, the one-shots despawned). Exit 0 when every check passed, 1 when one failed, 77
// when the pack is not there (a build without the models: tests skip).
//
// --self-test needs no learned model: the adapter's load errors (a missing pack, a damaged one, an unknown model,
// script, member or entry, bad settings; each leaving no texture or hook behind), then a stage of the pack's procedural
// entries played for 2 s twice (on two workers and on the game thread alone) with spawns, stops, a failed spawn and
// calls on parts the level does not have: no allocation in any frame, the same frames, the frame error kept.
#include "nvfx_fuse.hpp"
#include "standin.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <new>
#include <string>
#include <vector>

// --- counting heap allocations ------------------------------------------------------------------------------------------

namespace {
std::atomic<long> g_allocations{0};
std::atomic<bool> g_counting{false};
thread_local bool t_uncounted = false;  // the capture (PNG, video) is the demo's, not the engine's frame
}  // namespace

void* operator new(std::size_t n) {
  if (g_counting.load(std::memory_order_relaxed) && !t_uncounted) g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) {
  if (g_counting.load(std::memory_order_relaxed) && !t_uncounted) g_allocations.fetch_add(1, std::memory_order_relaxed);
  const std::size_t al = static_cast<std::size_t>(a);
  if (void* p = std::aligned_alloc(al, (n + al - 1) / al * al + (n ? 0 : al))) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n, std::align_val_t a) { return operator new(n, a); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace {

using fuse::Vec3;
using Clock = std::chrono::steady_clock;

constexpr int kWidth = 1280, kHeight = 720;
constexpr double kHz = 60.0;

struct Options {
  std::string pack, out, video;
  double seconds = 10.0;
  int workers = 3;
  bool check = false, calibrate = false;
};

float smooth(float x) {
  x = std::clamp(x, 0.f, 1.f);
  return x * x * (3.f - 2.f * x);
}

// --- what the demo's capture does with each frame drawn (the render thread) ---------------------------------------------

struct Capture {
  std::vector<std::uint64_t>* hashes = nullptr;
  std::FILE* video = nullptr;
  std::string out;
  std::uint64_t keyframes[5] = {60, 180, 300, 420, 540};
  int png_written = 0;
};

void frame_done(void* ctx, std::uint64_t frame, const std::uint8_t* rgb, int w, int h) noexcept {
  auto& c = *static_cast<Capture*>(ctx);
  const std::size_t n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3;
  if (frame < c.hashes->size()) (*c.hashes)[static_cast<std::size_t>(frame)] = standin::fnv1a({rgb, n});
  t_uncounted = true;
  if (c.video && frame % 2 == 0) std::fwrite(rgb, 1, n, c.video);
  if (!c.out.empty())
    for (const std::uint64_t k : c.keyframes)
      if (k == frame) {
        char name[64];
        std::snprintf(name, sizeof(name), "/fuse_demo_%02d.png", static_cast<int>(frame / 60));
        if (standin::write_png(c.out + name, rgb, w, h)) ++c.png_written;
      }
  t_uncounted = false;
}

// --- the world --------------------------------------------------------------------------------------------------------

const nvfx_fuse::Card kSceneCard{{-3.3f, 0.f, -3.0f}, {0.940f, 0.f, 0.342f}, {0.f, 1.f, 0.f}, 0.0085f};   // the campfire script, 960 x 540
const nvfx_fuse::Card kStageCard{{3.4f, 0.f, -3.3f}, {0.940f, 0.f, -0.342f}, {0.f, 1.f, 0.f}, 0.0090f};    // the stage, 640 x 360
constexpr float kStageGround = 0.87f * 360.f;

// A point of a card from its scene's world pixels (x across, y down to its ground line at `ground`).
Vec3 on_card(const nvfx_fuse::Card& c, float x_from_middle, float above_ground) {
  const float k = c.units_per_px;
  return {c.base.x + c.right.x * x_from_middle * k, c.base.y + above_ground * k, c.base.z + c.right.z * x_from_middle * k};
}

struct Result {
  std::vector<std::uint64_t> hashes;
  long allocations = 0, load_allocations = 0;
  std::uint64_t errors = 0, uploaded = 0, frames_drawn = 0;
  std::string last_error;
  float max_heat = 0.f;
  int max_sprites_drawn = 0, explosions_left = 0;
  double update_ms = 0, update_ms_max = 0, volume_ms = 0, volume_ms_max = 0, scene_ms = 0, stage_ms = 0, jobs_ms = 0, planned_ms = 0, wall_s = 0;
  double volume_step_ms = 0, volume_picture_ms = 0, volume_fields_ms = 0;
  int volume_frames = 0, scene_frames = 0, stage_frames = 0;
  std::size_t memory = 0, textures = 0;
  nvfx_fuse::Stats::Memory mem{};
  double load_ms = 0;
  nvfx_manager_stats manager{};
  int status = 0;  // 0 ran, 1 failed to load, 77 no pack
};

Result play(const Options& o, int workers, bool capture) {
  Result res;
  const int frames = static_cast<int>(std::lround(o.seconds * kHz));
  res.hashes.assign(static_cast<std::size_t>(frames), 0);

  // the stand-in FUSE
  Capture cap;
  cap.hashes = &res.hashes;
  if (capture) {
    cap.out = o.out;
    if (!o.video.empty()) {
      const std::string cmd = "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s " + std::to_string(kWidth) + "x" + std::to_string(kHeight) +
                              " -r 30 -i - -c:v libx264 -preset slow -crf 24 -pix_fmt yuv420p -movflags +faststart \"" + o.video + "\"";
      cap.video = popen(cmd.c_str(), "w");
      if (!cap.video) std::fprintf(stderr, "cannot start ffmpeg: no video\n");
    }
  }
  standin::ThreadPool pool(workers);
  standin::SoftRenderer gpu(kWidth, kHeight, &frame_done, &cap);
  standin::FixedClock clock(kHz);
  standin::LookAtCamera camera(kWidth, kHeight, 0.82f);
  const fuse::Engine engine{&pool, &gpu, &gpu};

  // the level: what NeuralVFX plays, from one pack
  const nvfx_fuse::SpriteModel models[] = {{"fire", 4, 0}, {"smoke", 3, 0}, {"explosion", 3, 0}, {"magic", 2, 0}};
  const nvfx_stage_room rooms[] = {{"brazier", 1, 0}, {"steam_vent", 1, 0}, {"electric_arc", 1, 0}, {"magic_burst", 1, 1}, {"lightning_strike", 1, 1}};
  const nvfx_scene_input inputs[] = {{"fuel", 0.75f}};
  nvfx_fuse::Config cfg;
  cfg.pack_path = o.pack.c_str();
  cfg.sprites.models = models;
  cfg.sprites.max_sprites = 24;
  cfg.sprites.budget_ms = 5.f;
  cfg.sprites.job_slots = 2;
  cfg.sprites.costs = o.calibrate ? nvfx_fuse::Costs::pack_or_calibrate : nvfx_fuse::Costs::pack_or_default;
  cfg.scene.script = "campfire.nvfxs";
  cfg.scene.inputs = inputs;
  cfg.scene.width = 480;
  cfg.scene.height = 270;
  cfg.scene.card = kSceneCard;
  cfg.stage.rooms = rooms;
  cfg.stage.card = kStageCard;
  cfg.volume.member = "fire3d.nvfx3";
  cfg.volume.position = {0.2f, 0.f, -0.9f};
  cfg.volume.scale = 2.4f;
  cfg.volume.seed = 9950001;
  cfg.volume.fields = true;
  cfg.volume.picture_scale = 0.4f;

  g_allocations.store(0);
  g_counting.store(true);
  const auto t_load = Clock::now();
  auto sys = nvfx_fuse::System::create(engine, cfg);
  g_counting.store(false);
  res.load_allocations = g_allocations.load();
  if (!sys) {
    std::fprintf(stderr, "nvfx_fuse_demo: %s\n", sys.error().message.c_str());
    res.status = 1;
    if (cap.video) pclose(cap.video);
    return res;
  }
  nvfx_fuse::System& fx = **sys;
  res.load_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_load).count();
  const int fire = fx.model("fire"), smoke = fx.model("smoke"), explosion = fx.model("explosion"), magic = fx.model("magic");

  // the game's state
  nvfx_fuse::Sprite torches[6], columns[2], blasts[2], portal;
  nvfx_fuse::StageFx vent;
  const char* caption = "";
  double caption_until = 0.0;
  char hud[320];
  const Vec3 campfire = on_card(kSceneCard, 440.f - 480.f, 30.f);  // the scene's campfire (x 440, 30 px above its ground)

  g_allocations.store(0);
  g_counting.store(true);
  const auto t_run = Clock::now();
  for (int f = 0; f < frames; ++f) {
    const fuse::FrameTime ft = clock.now();
    const float t = static_cast<float>(ft.time);
    const auto at = [&](double s) { return f == static_cast<int>(std::lround(s * kHz)); };
    const auto say = [&](const char* text) {
      caption = text;
      caption_until = ft.time + 2.5;
    };

    // --- the game: spawns, despawns, inputs, rules (between updates)
    if (f == 0) {
      const float zs[3] = {3.0f, 1.2f, -0.6f};
      for (int i = 0; i < 6; ++i) {  // six torches, one simulation shown 0.37 s apart
        const bool left = i < 3;
        nvfx_fuse::SpriteDesc d;
        d.model = fire;
        d.position = {left ? -1.6f - 0.3f * static_cast<float>(i) : 2.2f + 0.3f * static_cast<float>(i - 3), 0.f, zs[i % 3]};
        d.size = 1.1f;
        d.controls = {0.8f, 0.5f, 0.45f};
        d.seed = 7;
        d.time_offset = 0.37f * static_cast<float>(i);
        torches[i] = fx.spawn(d);
      }
      const Vec3 at_col[2] = {{-0.9f, 0.f, -4.6f}, {1.5f, 0.f, -4.9f}};
      for (int i = 0; i < 2; ++i) {  // two smoke columns, each its own run
        nvfx_fuse::SpriteDesc d;
        d.model = smoke;
        d.position = at_col[i];
        d.size = 3.6f;
        d.controls = {0.4f, 0.5f, 0.5f};
        d.seed = 31 + static_cast<std::uint64_t>(i);
        columns[i] = fx.spawn(d);
      }
      say("SPRITES: SIX TORCHES (ONE SIMULATION, OFFSET IN TIME), TWO SMOKE COLUMNS");
    }
    if (at(0.3)) fx.spawn_entry("brazier", on_card(kStageCard, -210.f, 0.f), 0.8f);
    if (at(0.6)) vent = fx.spawn_entry("steam_vent", on_card(kStageCard, -40.f, 0.f), 0.8f);
    if (at(1.0)) {
      fx.spawn_entry("electric_arc", on_card(kStageCard, 170.f, 0.f), 1.f);
      say("STAGE: LIBRARY ENTRIES SPAWNED: BRAZIER, STEAM VENT, ELECTRIC ARC");
    }
    if (at(1.5)) {
      fx.trigger("blow");
      say("SCENE: RULE 'BLOW' FIRED BY THE GAME (A GUST)");
    }
    if (at(2.0)) {
      nvfx_fuse::SpriteDesc d;
      d.model = explosion;
      d.position = {1.7f, 0.f, 1.6f};
      d.size = 2.2f;
      d.controls = {0.6f, 0.5f, 0.5f};
      d.seed = 101;
      d.allow_proxy = false;
      d.priority = 2.f;
      blasts[0] = fx.spawn(d);
      say("SPRITES: AN EXPLOSION, DESPAWNED WHEN IT FINISHES");
    }
    if (at(3.0)) {
      fx.trigger("light_log");
      say("SCENE: RULE 'LIGHT_LOG': THE LOG CATCHES");
    }
    if (at(3.6)) {
      fx.spawn_entry("magic_burst", on_card(kStageCard, 60.f, 170.f), 0.8f);
      say("STAGE: A MAGIC BURST (A ONE-SHOT ENTRY)");
    }
    if (at(4.2)) {
      nvfx_fuse::SpriteDesc d;
      d.model = magic;
      d.position = {-0.9f, 1.6f, 0.6f};
      d.anchor_base = false;
      d.size = 1.4f;
      d.controls = {0.7f, 0.6f, 0.5f};
      d.seed = 55;
      portal = fx.spawn(d);
      say("SPRITES: A MAGIC PORTAL");
    }
    if (at(5.0)) say("THE GAME'S WIND: SCENE INPUTS, SMOKE CONTROLS, THE VOLUME'S WIND");
    // the wind, from 5 to 7.4 s: one game value into all three
    const float wind = smooth((t - 5.f) / 0.8f) * (1.f - smooth((t - 6.6f) / 0.8f));
    if (f % 6 == 0 || at(7.4)) {  // the scene's inputs and the volume's controls, ten times a second
      fx.set_input("lean", 0.5f + 0.3f * wind);
      fx.set_input("breeze", 2.f * wind);
      const float vc[3] = {0.7f, 0.5f + 0.25f * wind, 0.6f};
      fx.set_volume_controls(vc);
    }
    if (at(5.0) || at(7.4)) {
      const float sc[3] = {0.4f, at(5.0) ? 0.85f : 0.5f, 0.5f};
      for (auto& c : columns) fx.set_controls(c, sc);
    }
    if (at(5.9)) {
      fx.spawn_entry("lightning_strike", on_card(kStageCard, 190.f, 0.f), 0.8f);
      say("STAGE: A LIGHTNING STRIKE");
    }
    if (at(6.3)) {
      nvfx_fuse::SpriteDesc d;
      d.model = explosion;
      d.position = {-1.3f, 0.f, 2.2f};
      d.size = 2.0f;
      d.controls = {0.55f, 0.5f, 0.5f};
      d.seed = 202;
      d.allow_proxy = false;
      d.priority = 2.f;
      blasts[1] = fx.spawn(d);
      say("SPRITES: A SECOND EXPLOSION");
    }
    if (at(7.0)) {
      fx.despawn(torches[2]);
      fx.despawn(torches[5]);
      say("SPRITES: THE TWO FAR TORCHES DESPAWNED");
    }
    if (at(7.6)) {
      fx.trigger("douse");
      say("SCENE: RULE 'DOUSE': THE CAMPFIRE IS PUT OUT");
    }
    if (at(8.4)) {
      fx.stop_entry(vent);
      say("STAGE: THE STEAM VENT STOPPED");
    }
    if (at(8.8)) {
      fx.set_input("fuel", 1.f);
      fx.trigger("relight");
      say("SCENE: RULE 'RELIGHT', INPUT FUEL = 1");
    }

    // --- the camera: a slow dolly in
    const float k = smooth(t / 10.f);
    camera.look_at({-0.5f + 1.2f * k, 1.7f + 0.3f * k, 7.8f - 2.0f * k}, {0.f + 0.3f * k, 1.5f, -1.5f});

    // --- the frame
    fx.update(ft, camera);
    const nvfx_fuse::Stats& st = fx.stats();
    res.max_heat = std::max(res.max_heat, fx.heat_at(campfire));
    res.max_sprites_drawn = std::max(res.max_sprites_drawn, st.sprites_drawn);
    res.update_ms += st.update_ms;
    res.update_ms_max = std::max(res.update_ms_max, st.update_ms);
    if (st.volume_ms > 0) {
      res.volume_ms += st.volume_ms;
      res.volume_ms_max = std::max(res.volume_ms_max, st.volume_ms);
      res.volume_step_ms += st.volume_step_ms;
      res.volume_picture_ms += st.volume_picture_ms;
      res.volume_fields_ms += st.volume_fields_ms;
      ++res.volume_frames;
    }
    if (st.scene_ms > 0) {
      res.scene_ms += st.scene_ms;
      ++res.scene_frames;
    }
    if (st.stage_ms > 0) {
      res.stage_ms += st.stage_ms;
      ++res.stage_frames;
    }
    res.jobs_ms += st.manager.jobs_ms;
    res.planned_ms += st.manager.planned_ms;
    if (ft.time >= caption_until) caption = "";
    std::snprintf(hud, sizeof(hud), "NEURALVFX IN A STAND-IN FUSE   T %4.1f S|SPRITES %d  SIMULATIONS %d  PROXIED %d  PLAN %.1f OF %.0f MS|%s", ft.time,
                  st.sprites_drawn, st.manager.sims, st.manager.proxied, st.manager.plan_ms, st.manager.budget_ms, caption);
    gpu.submit(ft.frame, camera, hud);
    clock.advance();
  }
  gpu.finish();
  g_counting.store(false);
  res.wall_s = std::chrono::duration<double>(Clock::now() - t_run).count();
  res.allocations = g_allocations.load();

  const nvfx_fuse::Stats& st = fx.stats();
  res.errors = st.errors;
  res.last_error = fx.last_error().what;
  res.uploaded = st.uploaded_bytes;
  res.frames_drawn = st.frames_drawn;
  res.memory = st.memory_bytes;
  res.mem = st.memory;
  res.textures = gpu.texture_bytes();
  res.manager = st.manager;
  for (const auto& b : blasts) res.explosions_left += fx.alive(b) ? 1 : 0;
  if (cap.video) {
    t_uncounted = true;
    if (pclose(cap.video) != 0) std::fprintf(stderr, "ffmpeg failed\n");
    t_uncounted = false;
  }
  if (capture && !o.out.empty()) std::printf("keyframes: %d PNG in %s\n", cap.png_written, o.out.c_str());
  return res;
}

// --- the self-test: no learned model needed ----------------------------------------------------------------------------

struct StageRun {
  std::vector<std::uint64_t> hashes;
  long allocations = 0;
  std::uint64_t errors = 0, frames_drawn = 0, uploaded = 0;
  std::string last_error;
  bool calls_ok = true;  // every call returned what it should
  int status = 0;
};

// The pack's procedural entries on a stage, 2 s at 60 Hz, with spawns, stops and calls that must fail.
StageRun play_stage(const std::string& pack, int workers) {
  constexpr int kFrames = 120, kW = 320, kH = 180;
  StageRun r;
  r.hashes.assign(kFrames, 0);
  Capture cap;
  cap.hashes = &r.hashes;
  standin::ThreadPool pool(workers);
  standin::SoftRenderer gpu(kW, kH, &frame_done, &cap);
  standin::FixedClock clock(kHz);
  standin::LookAtCamera camera(kW, kH, 0.9f);
  camera.look_at({0.f, 1.1f, 3.6f}, {0.f, 0.9f, 0.f});
  const nvfx_stage_room rooms[] = {{"sparks", 2, 0}, {"electric_arc", 1, 0}, {"lightning_strike", 1, 1}};
  nvfx_fuse::Config cfg;
  cfg.pack_path = pack.c_str();
  cfg.stage.rooms = rooms;
  cfg.stage.width = 320;
  cfg.stage.height = 180;
  cfg.stage.card = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, 0.01f};
  auto sys = nvfx_fuse::System::create({&pool, &gpu, &gpu}, cfg);
  if (!sys) {
    std::fprintf(stderr, "nvfx_fuse_demo: %s\n", sys.error().message.c_str());
    r.status = 1;
    return r;
  }
  nvfx_fuse::System& fx = **sys;
  nvfx_fuse::StageFx sparks;
  const float no_controls[3] = {0.5f, 0.5f, 0.5f};
  g_allocations.store(0);
  g_counting.store(true);
  for (int f = 0; f < kFrames; ++f) {
    const fuse::FrameTime ft = clock.now();
    if (f == 0) sparks = fx.spawn_entry("sparks", {-0.8f, 0.f, 0.f});
    if (f == 10) r.calls_ok &= static_cast<bool>(fx.spawn_entry("electric_arc", {0.6f, 0.f, 0.f}));
    if (f == 30 || f == 70) r.calls_ok &= static_cast<bool>(fx.spawn_entry("lightning_strike", {0.2f, 0.f, 0.f}, 0.4f));
    if (f == 40) {  // calls that fail, in a frame, without allocating
      r.calls_ok &= !fx.spawn_entry("no_such_entry", {0.f, 0.f, 0.f});  // a frame error, kept
      r.calls_ok &= !fx.spawn(nvfx_fuse::SpriteDesc{});                   // this level has no sprites: an error too
      r.calls_ok &= !fx.trigger("blow") && !fx.set_input("fuel", 1.f);    // nor a scene: false
      r.calls_ok &= !fx.set_volume_controls(no_controls);                 // nor a volume
      r.calls_ok &= !fx.despawn(nvfx_fuse::Sprite{12345}) && fx.model("fire") == -1;
    }
    if (f == 90) r.calls_ok &= fx.stop_entry(sparks);
    fx.update(ft, camera);
    gpu.submit(ft.frame, camera, "SELF-TEST");
    clock.advance();
  }
  gpu.finish();
  g_counting.store(false);
  r.allocations = g_allocations.load();
  const nvfx_fuse::Stats& st = fx.stats();
  r.errors = st.errors;
  r.frames_drawn = st.frames_drawn;
  r.uploaded = st.uploaded_bytes;
  r.last_error = fx.last_error().what;
  return r;
}

int self_test(const std::string& pack) {
  if (!std::filesystem::exists(pack)) {
    std::printf("nvfx_fuse_demo --self-test: no pack (%s): skipped\n", pack.c_str());
    return 77;
  }
  int failures = 0;
  const auto expect = [&](bool ok, const char* what) {
    std::printf("  %s: %s\n", ok ? "ok    " : "FAILED", what);
    failures += ok ? 0 : 1;
  };
  std::printf("load errors:\n");
  {
    std::vector<std::uint64_t> none;
    Capture cap;
    cap.hashes = &none;
    standin::ThreadPool pool(1);
    standin::SoftRenderer gpu(64, 36, &frame_done, &cap);
    const fuse::Engine engine{&pool, &gpu, &gpu};
    const auto refused = [&](const nvfx_fuse::Config& c, const fuse::Engine& e, nvfx_status want, const char* what) {
      const auto s = nvfx_fuse::System::create(e, c);
      expect(!s && s.error().status == want && gpu.live_textures() == 0 && gpu.hooks() == 0, what);
      if (!s) std::printf("          (%s)\n", s.error().message.c_str());
    };
    const nvfx_stage_room rooms[] = {{"sparks", 1, 0}};
    nvfx_fuse::Config base;
    base.pack_path = pack.c_str();
    base.stage.rooms = rooms;
    nvfx_fuse::Config c = base;
    c.pack_path = "/nonexistent/level.nvfxp";
    refused(c, engine, NVFX_ERROR_IO, "a missing pack is refused");
    static const std::uint8_t junk[96] = {'N', 'V', 'F', 'X', 'P', 'A', 'C', 'K'};
    c = base;
    c.pack_path = nullptr;
    c.pack_bytes = junk;
    refused(c, engine, NVFX_ERROR_FORMAT, "a damaged pack is refused");
    c = base;
    c.frames_in_flight = 1;
    refused(c, engine, NVFX_ERROR_ARGUMENT, "frames_in_flight below 2 is refused");
    refused(base, fuse::Engine{nullptr, &gpu, &gpu}, NVFX_ERROR_ARGUMENT, "an engine without a job system is refused");
    const nvfx_fuse::SpriteModel models[] = {{"fire", 4, 0}};
    c = base;
    c.sprites.models = models;
    refused(c, engine, NVFX_ERROR_IO, "a model the pack does not hold is refused");
    c = base;
    c.scene.script = "no_such_script.nvfxs";
    refused(c, engine, NVFX_ERROR_IO, "a script the pack does not hold is refused");
    c = base;
    c.volume.member = "fire3d.nvfx3";
    refused(c, engine, NVFX_ERROR_IO, "a volume the pack does not hold is refused (after the stage was made: nothing left behind)");
    const nvfx_stage_room unknown[] = {{"no_such_entry", 1, 0}};
    c = base;
    c.stage.rooms = unknown;
    refused(c, engine, NVFX_ERROR_ARGUMENT, "a room for an entry the catalogue lacks is refused");
  }
  std::printf("a stage of procedural entries, 120 frames on 2 workers and on the game thread alone:\n");
  const StageRun a = play_stage(pack, 2), b = play_stage(pack, 0);
  expect(a.status == 0 && b.status == 0, "the level loads and plays");
  expect(a.allocations == 0 && b.allocations == 0, "no heap allocation in any frame, on any thread (failed calls included)");
  expect(a.hashes == b.hashes, "the same frames to the bit on 2 workers and on the game thread alone");
  expect(a.hashes.front() != a.hashes.back(), "the frames change");
  expect(a.frames_drawn == 120 && b.frames_drawn == 120, "the render hook drew every frame");
  expect(a.uploaded > 0, "the stage's texture uploaded");
  expect(a.calls_ok && b.calls_ok, "spawns, a stop, and calls on parts the level lacks return what they should");
  expect(a.errors == 2 && a.last_error.starts_with("spawn: no sprite model"), "the failed calls are kept as frame errors");
  std::printf("  (%llu errors, the last: %s)\n", static_cast<unsigned long long>(a.errors), a.last_error.c_str());
  std::printf("%s\n", failures ? "FAILED" : "all checks passed");
  return failures ? 1 : 0;
}

void report(const Result& r, int workers, int frames) {
  const auto mean = [](double s, int n) { return n > 0 ? s / n : 0.0; };
  std::printf("%d frames on %d worker%s + the game thread in %.1f s: %ld heap allocations in the frames (%ld at load), %llu frame errors\n", frames, workers,
              workers == 1 ? "" : "s", r.wall_s, r.allocations, r.load_allocations, static_cast<unsigned long long>(r.errors));
  std::printf("  update (game thread, wall): mean %.1f ms, worst %.1f ms; volume job %.1f ms mean (%d frames, worst %.1f); scene %.1f ms (%d); stage %.1f ms (%d)\n",
              mean(r.update_ms, frames), r.update_ms_max, mean(r.volume_ms, r.volume_frames), r.volume_frames, r.volume_ms_max, mean(r.scene_ms, r.scene_frames),
              r.scene_frames, mean(r.stage_ms, r.stage_frames), r.stage_frames);
  std::printf("  the volume's job: step %.1f ms, picture %.1f ms, fields read %.1f ms (means)\n", mean(r.volume_step_ms, r.volume_frames),
              mean(r.volume_picture_ms, r.volume_frames), mean(r.volume_fields_ms, r.volume_frames));
  std::printf("  manager: %.2f ms of jobs per tick against %.2f planned (budget %.0f ms); last tick %d sims, %d proxied, %.1f MB\n", mean(r.jobs_ms, frames),
              mean(r.planned_ms, frames), r.manager.budget_ms, r.manager.sims, r.manager.proxied, static_cast<double>(r.manager.memory_bytes) / 1048576.0);
  std::printf("  uploaded %.1f MB in all (%.2f MB a frame); heat on the campfire up to %.2f; up to %d sprites drawn\n", static_cast<double>(r.uploaded) / 1048576.0,
              static_cast<double>(r.uploaded) / 1048576.0 / std::max(1, frames), r.max_heat, r.max_sprites_drawn);
  const auto mb = [](std::size_t b) { return static_cast<double>(b) / 1048576.0; };
  std::printf("  loaded in %.0f ms; %.1f MB on the CPU (models %.1f, manager %.1f, scene and stage %.1f, volume %.1f, packets %.1f), textures %.1f MB\n", r.load_ms,
              mb(r.memory), mb(r.mem.models), mb(r.mem.manager), mb(r.mem.scenes), mb(r.mem.volume), mb(r.mem.packets), mb(r.textures));
  if (r.errors) std::printf("  last error: %s\n", r.last_error.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--pack") o.pack = next();
    else if (a == "--out") o.out = next();
    else if (a == "--video") o.video = next();
    else if (a == "--seconds") o.seconds = std::atof(next().c_str());
    else if (a == "--workers") o.workers = std::atoi(next().c_str());
    else if (a == "--check") o.check = true;
    else if (a == "--self-test") return self_test(next());
    else if (a == "--calibrate") o.calibrate = true;
    else {
      std::fprintf(stderr, "nvfx_fuse_demo --pack PACK [--out DIR] [--video FILE.mp4] [--seconds 10] [--workers 3] [--calibrate] [--check] | --self-test PACK\n");
      return 2;
    }
  }
  if (o.pack.empty() || !std::filesystem::exists(o.pack)) {
    std::printf("nvfx_fuse_demo: no pack (%s): skipped\n", o.pack.c_str());
    return 77;
  }
  if (!o.out.empty()) std::filesystem::create_directories(o.out);
  const nvfx::Version v = nvfx::version();
  std::printf("libnvfx %u (headers %d), features 0x%x%s%s%s\n", static_cast<unsigned>(v.version), NVFX_VERSION, static_cast<unsigned>(v.features),
              v.has(NVFX_FEATURE_SHARED) ? ", shared" : "", v.has(NVFX_FEATURE_AVX512) ? ", AVX-512" : "", v.has(NVFX_FEATURE_VNNI) ? ", VNNI" : "");
  const int frames = static_cast<int>(std::lround(o.seconds * kHz));

  const Result a = play(o, o.workers, true);
  if (a.status) return a.status;
  report(a, o.workers, frames);
  if (!o.check) return a.allocations == 0 && a.errors == 0 ? 0 : 1;

  const Result b = play(o, 0, false);
  if (b.status) return b.status;
  report(b, 0, frames);
  int failures = 0;
  const auto expect = [&](bool ok, const char* what) {
    std::printf("  %s: %s\n", ok ? "ok    " : "FAILED", what);
    failures += ok ? 0 : 1;
  };
  int differ = 0;
  for (std::size_t i = 0; i < a.hashes.size(); ++i) differ += a.hashes[i] != b.hashes[i] ? 1 : 0;
  std::printf("checks:\n");
  expect(a.load_allocations > 0, "the allocation counter sees libnvfx's allocations (at load)");
  expect(a.allocations == 0 && b.allocations == 0, "no heap allocation in any frame, on any thread");
  expect(differ == 0, differ ? "the same frames on any number of workers (some differ)" : "the same frames to the bit on 3 workers and on the game thread alone");
  expect(a.errors == 0 && b.errors == 0, "no frame error");
  expect(a.frames_drawn == static_cast<std::uint64_t>(frames), "the render hook drew every frame");
  expect(a.max_sprites_drawn >= 8, "sprites drawn (torches, smoke, explosions, the portal)");
  expect(o.seconds < 9.0 || a.explosions_left == 0, "the explosions were despawned when they finished");
  expect(a.max_heat > 0.1f, "heat over the campfire's card (gameplay reads the scene's fields)");
  expect(a.uploaded > 0, "textures uploaded (sprites, cards, the volume's picture and fields)");
  std::printf("%s\n", failures ? "FAILED" : "all checks passed");
  return failures ? 1 : 0;
}
