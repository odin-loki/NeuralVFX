// The effect manager (include/neuralfx/nvfx_manager.h, docs/MANAGER.md): its frames against nvfx_render at a fixed
// level, the same pictures on any number of threads, level changes that keep the coarse state, sharing, copies offset
// in time, pauses that resume warm, the budget, hysteresis, the interpolation kernels and the C API's errors. Small
// effects built in memory; nothing here needs the data root.
#include <neuralfx/nvfx.h>
#include <neuralfx/nvfx_manager.h>
#include <neuralfx/rollout.hpp>

#include "manager/interp.hpp"
#include "runtime/nvfx_internal.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using Bytes = std::vector<std::uint8_t>;

// A small rollout effect: 16 x 16 coarse cells; start points with fine fields and without (grown by a warm-up).
std::string tiny_effect(bool loop, std::uint64_t seed = 3) {
  namespace ro = nfx::rollout;
  ro::Hyper h;
  h.res = 16;
  h.hidden = 8;
  h.memory = 2;
  h.jacobi = 10;
  h.render_hidden = 6;
  h.warmup = 4;
  h.start_fine = 16;
  if (!loop) {
    h.n_age = 2;
    h.frames = 40;
  }
  ro::Model m = ro::init_model(h, seed);
  m.effect = loop ? "tinyloop" : "tinyshot";
  m.loop = loop;
  m.control_names = {"intensity", "wind", "turbulence"};
  m.scale = {0.2f, 0.2f, 0.4f, 0.3f};
  m.lo = {-2.f, -2.f, 0.f, 0.f};
  m.hi = {2.f, 2.f, 3.f, 3.f};
  m.detail.swirl_control = 2;
  for (int k = 0; k < 4; ++k) {
    ro::StartPoint sp;
    sp.controls = {0.2f + 0.2f * static_cast<float>(k), 0.5f, 0.5f};
    sp.seed = 10 + static_cast<std::uint64_t>(k);
    sp.coarse.resize(static_cast<std::size_t>(h.res) * static_cast<std::size_t>(h.res) * ro::kPhys);
    for (int y = 0; y < h.res; ++y) {
      for (int x = 0; x < h.res; ++x) {
        float* c = sp.coarse.data() + (static_cast<std::size_t>(y) * static_cast<std::size_t>(h.res) + static_cast<std::size_t>(x)) * ro::kPhys;
        c[0] = 0.1f * std::sin(0.5f * static_cast<float>(x + k));
        c[1] = 0.2f + 0.1f * std::cos(0.4f * static_cast<float>(y));
        c[2] = std::max(0.f, 1.2f - 0.08f * static_cast<float>(std::abs(x - 8) + y));
        c[3] = std::max(0.f, 0.7f - 0.05f * static_cast<float>(std::abs(x - 7) + std::abs(y - 6)));
      }
    }
    if (k % 2 == 1) {  // fine fields kept with the start (the others grow theirs in a warm-up)
      sp.fine_t.resize(static_cast<std::size_t>(h.start_fine) * static_cast<std::size_t>(h.start_fine));
      sp.fine_d.resize(sp.fine_t.size());
      for (std::size_t i = 0; i < sp.fine_t.size(); ++i) {
        sp.fine_t[i] = sp.coarse[i * ro::kPhys + 2];
        sp.fine_d[i] = sp.coarse[i * ro::kPhys + 3];
      }
    }
    m.starts.push_back(sp);
  }
  std::ostringstream os;
  EXPECT_TRUE(ro::save_model(os, m));
  return os.str();
}

struct Effect {
  nvfx_effect* fx = nullptr;
  explicit Effect(const std::string& bytes) { EXPECT_EQ(nvfx_effect_load_memory(bytes.data(), bytes.size(), &fx), NVFX_OK); }
  ~Effect() { nvfx_effect_free(fx); }
};

struct Mgr {
  nvfx_manager* m = nullptr;
  explicit Mgr(const nvfx_manager_desc& d) { EXPECT_EQ(nvfx_manager_create(&d, &m), NVFX_OK); }
  ~Mgr() { nvfx_manager_free(m); }
};

nvfx_manager_desc small_desc(int threads = 1) {
  nvfx_manager_desc d;
  nvfx_manager_desc_init(&d);
  d.threads = threads;
  d.max_instances = 64;
  d.max_size = 128;
  d.budget_ms = 1000.f;
  return d;
}

nvfx_manager_model_desc small_model(float shard_seconds = 2.5f) {
  nvfx_manager_model_desc d;
  nvfx_manager_model_desc_init(&d);
  d.max_sims = 24;
  d.slots[0] = d.slots[1] = 24;
  d.slots[2] = 12;
  d.shard_seconds = shard_seconds;
  return d;
}

nvfx_handle spawn(nvfx_manager* m, int model, std::uint64_t seed, float px, std::array<float, 3> c = {0.45f, 0.55f, 0.6f}, float offset = 0.f, int share = 1) {
  nvfx_spawn_desc s;
  nvfx_spawn_desc_init(&s);
  s.model = model;
  s.controls = c.data();
  s.n_controls = 3;
  s.seed = seed;
  s.screen_px = px;
  s.time_offset = offset;
  s.share = share;
  nvfx_handle h = 0;
  EXPECT_EQ(nvfx_manager_spawn(m, &s, &h), NVFX_OK);
  return h;
}

Bytes picture(nvfx_manager* m, nvfx_handle h, int* size = nullptr) {
  nvfx_manager_picture p;
  EXPECT_EQ(nvfx_manager_get_picture(m, h, &p), NVFX_OK);
  if (size) *size = p.size;
  if (!p.rgba) return {};
  return Bytes(p.rgba, p.rgba + static_cast<std::size_t>(p.size) * static_cast<std::size_t>(p.size) * 4);
}

std::uint64_t hash(const Bytes& b, std::uint64_t h = 1469598103934665603ULL) {
  for (const std::uint8_t c : b) h = (h ^ c) * 1099511628211ULL;
  return h;
}

// At a fixed level (locked) and 30 Hz in a 30 Hz game, the manager's frames are nvfx_render's to the bit: the same
// shard timeline, keyframe by keyframe (through shard changes and their crossfades, and warm-ups). At a precision: the
// model's (nvfx_manager_model_desc.precision) and the instance's (study K's int8 path, docs/REPORT.md §13).
void expect_like_render(bool loop, int size, nvfx_precision precision = NVFX_PRECISION_DEFAULT) {
  const Effect e(tiny_effect(loop));
  Mgr g(small_desc());
  int model = -1;
  auto md = small_model(2.5f);
  md.precision = precision;
  ASSERT_EQ(nvfx_manager_add_model(g.m, e.fx, &md, &model), NVFX_OK);
  const std::array<float, 3> c{0.45f, 0.55f, 0.6f};
  const nvfx_handle h = spawn(g.m, model, 77, static_cast<float>(size), c);
  ASSERT_EQ(nvfx_manager_set_lod_lock(g.m, h, size, 1), NVFX_OK);
  nvfx_instance* in = nullptr;
  ASSERT_EQ(nvfx_instance_create(e.fx, size, &in), NVFX_OK);
  ASSERT_EQ(nvfx_instance_set_precision(in, precision), NVFX_OK);
  nvfx_instance_set_controls(in, c.data(), 3);
  nvfx_instance_set_seed(in, 77);
  nvfx_instance_set_drift(in, 2.5f);
  Bytes ref(static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4);
  int frame = 0, shown = 0;
  for (int t = 0; t < 200; ++t) {
    ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
    nvfx_manager_picture p;
    ASSERT_EQ(nvfx_manager_get_picture(g.m, h, &p), NVFX_OK);
    if (!p.rgba) continue;  // warming up
    ASSERT_EQ(p.size, size);
    nvfx_render(in, frame / 30.0, ref.data(), static_cast<std::size_t>(size) * 4);
    ASSERT_EQ(std::memcmp(ref.data(), p.rgba, ref.size()), 0) << "frame " << frame;
    ++frame;
    ++shown;
  }
  EXPECT_GT(shown, 150);
  nvfx_instance_free(in);
}

// The same on the project's trained effects (outside git; skipped without them): the manager's lean runners and level
// transfers against nvfx_render's runners, at three sizes, through a shard change for the looping ones.
TEST(Manager, TrainedEffectsAreNvfxRender) {
  const char* env = std::getenv("NEURALVFX_DATA");
  const std::string root = env ? env : "/root/nvfx-data";
  int checked = 0;
  for (const std::string file : {"v2/models/fire.nvfx", "v2/models/explosion.nvfx", "fx/models/d/steam.nvfx"}) {
    nvfx_effect* fx = nullptr;
    if (nvfx_effect_load((root + "/" + file).c_str(), &fx) != NVFX_OK) continue;
    nvfx_effect_info info{};
    nvfx_effect_get_info(fx, &info);
    for (const int size : {32, 128}) {
      Mgr g(small_desc());
      int model = -1;
      nvfx_manager_model_desc md;
      nvfx_manager_model_desc_init(&md);
      md.max_sims = 2;
      md.shard_seconds = 2.5f;
      ASSERT_EQ(nvfx_manager_add_model(g.m, fx, &md, &model), NVFX_OK);
      const std::array<float, 3> c{0.62f, 0.31f, 0.55f};
      const nvfx_handle h = spawn(g.m, model, 4242, static_cast<float>(size), c);
      ASSERT_EQ(nvfx_manager_set_lod_lock(g.m, h, size, 1), NVFX_OK);
      nvfx_instance* in = nullptr;
      ASSERT_EQ(nvfx_instance_create(fx, size, &in), NVFX_OK);
      nvfx_instance_set_controls(in, c.data(), 3);
      nvfx_instance_set_seed(in, 4242);
      nvfx_instance_set_drift(in, 2.5f);
      Bytes ref(static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4);
      int frame = 0;
      for (int t = 0; t < 110; ++t) {
        ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
        nvfx_manager_picture p;
        ASSERT_EQ(nvfx_manager_get_picture(g.m, h, &p), NVFX_OK);
        if (!p.rgba) continue;
        nvfx_render(in, frame / 30.0, ref.data(), static_cast<std::size_t>(size) * 4);
        ASSERT_EQ(std::memcmp(ref.data(), p.rgba, ref.size()), 0) << file << " " << size << " frame " << frame;
        ++frame;
      }
      EXPECT_GT(frame, 80);
      nvfx_instance_free(in);
      ++checked;
    }
    nvfx_effect_free(fx);
  }
  if (checked == 0) GTEST_SKIP() << "no trained effects under " << root;
}

// Drawn at half size (docs/MANAGER.md §2.10): a simulation at least half_px on screen is drawn at the level below its
// level, and its picture is the runtime's bilinear 2x of the drawing: to the bit what an instance of the level's size
// draws with nvfx_instance_set_upscale(NVFX_UPSCALE_BILINEAR), through shard changes, crossfades and warm-ups. With
// NVFX_UPSCALE_OFF the picture is the drawing itself: an instance of half the size.
void expect_half_like_render(bool loop, nvfx_upscale mode, nvfx_precision precision = NVFX_PRECISION_DEFAULT) {
  const Effect e(tiny_effect(loop));
  nvfx_manager_desc d = small_desc();
  d.half_px = 100.f;
  d.half_upscale = mode;
  Mgr g(d);
  int model = -1;
  auto md = small_model(2.5f);
  md.precision = precision;
  ASSERT_EQ(nvfx_manager_add_model(g.m, e.fx, &md, &model), NVFX_OK);
  const std::array<float, 3> c{0.45f, 0.55f, 0.6f};
  const nvfx_handle h = spawn(g.m, model, 77, 120.f, c);  // a 128 px picture, above half_px on screen: drawn at 64
  const int size = mode == NVFX_UPSCALE_OFF ? 64 : 128;
  nvfx_instance* in = nullptr;
  ASSERT_EQ(nvfx_instance_create(e.fx, size, &in), NVFX_OK);
  ASSERT_EQ(nvfx_instance_set_precision(in, precision), NVFX_OK);
  if (mode == NVFX_UPSCALE_BILINEAR) {
    ASSERT_EQ(nvfx_instance_set_upscale(in, NVFX_UPSCALE_BILINEAR), NVFX_OK);
  }
  nvfx_instance_set_controls(in, c.data(), 3);
  nvfx_instance_set_seed(in, 77);
  nvfx_instance_set_drift(in, 2.5f);
  Bytes ref(static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4);
  int frame = 0;
  for (int t = 0; t < 200; ++t) {
    ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
    nvfx_manager_picture p;
    ASSERT_EQ(nvfx_manager_get_picture(g.m, h, &p), NVFX_OK);
    if (!p.rgba) continue;  // warming up
    ASSERT_EQ(p.size, size);
    ASSERT_EQ(p.lod_size, 64);
    ASSERT_NE(p.flags & NVFX_PICTURE_HALF, 0);
    nvfx_render(in, frame / 30.0, ref.data(), static_cast<std::size_t>(size) * 4);
    ASSERT_EQ(std::memcmp(ref.data(), p.rgba, ref.size()), 0) << "frame " << frame;
    ++frame;
  }
  EXPECT_GT(frame, 150);
  nvfx_manager_stats st;
  nvfx_manager_get_stats(g.m, &st);
  EXPECT_EQ(st.half, 1);
  EXPECT_EQ(st.lod[1][0], 1);  // counted by the size it is drawn at
  nvfx_instance_free(in);
}

TEST(Manager, HalfSizeIsNvfxRenderWithTheBilinear2x) {
  expect_half_like_render(true, NVFX_UPSCALE_BILINEAR);
  expect_half_like_render(false, NVFX_UPSCALE_BILINEAR);
  expect_half_like_render(true, NVFX_UPSCALE_BILINEAR, NVFX_PRECISION_INT8);
}
TEST(Manager, HalfSizeWithoutTheUpscaleIsTheDrawing) {
  expect_half_like_render(true, NVFX_UPSCALE_OFF);
  expect_half_like_render(false, NVFX_UPSCALE_OFF);
}

// The same on the project's trained effects (outside git; skipped without them): a 256 px picture drawn at 128 and
// brought up by the bilinear 2x is an instance at 256 px with NVFX_UPSCALE_BILINEAR, frame for frame.
TEST(Manager, HalfSizeTrainedEffectsAreNvfxRender) {
  const char* env = std::getenv("NEURALVFX_DATA");
  const std::string root = env ? env : "/root/nvfx-data";
  int checked = 0;
  for (const std::string file : {"v2/models/fire.nvfx", "v2/models/explosion.nvfx", "fx/models/d/steam.nvfx"}) {
    nvfx_effect* fx = nullptr;
    if (nvfx_effect_load((root + "/" + file).c_str(), &fx) != NVFX_OK) continue;
    nvfx_manager_desc d = small_desc();
    d.max_size = 256;
    d.half_px = 256.f;
    d.half_upscale = NVFX_UPSCALE_BILINEAR;
    Mgr g(d);
    int model = -1;
    nvfx_manager_model_desc md;
    nvfx_manager_model_desc_init(&md);
    md.max_sims = 2;
    md.shard_seconds = 2.5f;
    ASSERT_EQ(nvfx_manager_add_model(g.m, fx, &md, &model), NVFX_OK);
    const std::array<float, 3> c{0.62f, 0.31f, 0.55f};
    const nvfx_handle h = spawn(g.m, model, 4242, 300.f, c);
    nvfx_instance* in = nullptr;
    ASSERT_EQ(nvfx_instance_create(fx, 256, &in), NVFX_OK);
    ASSERT_EQ(nvfx_instance_set_upscale(in, NVFX_UPSCALE_BILINEAR), NVFX_OK);
    nvfx_instance_set_controls(in, c.data(), 3);
    nvfx_instance_set_seed(in, 4242);
    nvfx_instance_set_drift(in, 2.5f);
    Bytes ref(256 * 256 * 4);
    int frame = 0;
    for (int t = 0; t < 100; ++t) {
      ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
      nvfx_manager_picture p;
      ASSERT_EQ(nvfx_manager_get_picture(g.m, h, &p), NVFX_OK);
      if (!p.rgba) continue;
      ASSERT_EQ(p.size, 256);
      ASSERT_EQ(p.lod_size, 128);
      nvfx_render(in, frame / 30.0, ref.data(), 256 * 4);
      ASSERT_EQ(std::memcmp(ref.data(), p.rgba, ref.size()), 0) << file << " frame " << frame;
      ++frame;
    }
    EXPECT_GT(frame, 70);
    nvfx_instance_free(in);
    nvfx_effect_free(fx);
    ++checked;
  }
  if (checked == 0) GTEST_SKIP() << "no trained effects under " << root;
}

// The threshold follows the size on screen with hysteresis: a size wobbling just around it does not switch, a size
// well below it draws at full size again (after min_dwell, the budget allowing), and above it at half size at once. The
// picture keeps its level's size throughout (the bilinear 2x).
TEST(Manager, HalfSizeFollowsTheScreenWithHysteresis) {
  const Effect e(tiny_effect(true));
  nvfx_manager_desc d = small_desc();
  d.half_px = 100.f;
  d.half_upscale = NVFX_UPSCALE_BILINEAR;  // the picture keeps its level's size
  d.min_dwell = 0.2f;
  Mgr g(d);
  int model = -1;
  const auto md = small_model();
  ASSERT_EQ(nvfx_manager_add_model(g.m, e.fx, &md, &model), NVFX_OK);
  const nvfx_handle h = spawn(g.m, model, 5, 120.f);
  int last = -1;  // drawn at half size in the last picture (across the runs)
  const auto run = [&](int ticks, auto px_of) {
    int switches = 0;
    nvfx_manager_picture p{};
    for (int t = 0; t < ticks; ++t) {
      nvfx_manager_set_screen_size(g.m, h, px_of(t), 1);
      EXPECT_EQ(nvfx_manager_tick(g.m, 1.0 / 60.0), NVFX_OK);
      nvfx_manager_get_picture(g.m, h, &p);
      if (!p.rgba) continue;
      EXPECT_EQ(p.size, 128);
      const int half = (p.flags & NVFX_PICTURE_HALF) ? 1 : 0;
      if (last >= 0 && half != last) ++switches;
      last = half;
    }
    return std::pair{switches, p};
  };
  auto [sw0, p0] = run(60, [](int) { return 120.f; });
  EXPECT_NE(p0.flags & NVFX_PICTURE_HALF, 0);
  EXPECT_EQ(p0.lod_size, 64);
  // around the threshold, within the hysteresis below it (85 to 100 keeps it on): no switch
  auto [sw1, p1] = run(240, [](int t) { return 100.f * (1.f + 0.1f * std::sin(0.7f * static_cast<float>(t))); });
  EXPECT_EQ(sw1, 0);
  EXPECT_NE(p1.flags & NVFX_PICTURE_HALF, 0);
  // well below: its full size again (the picture's level stays 128 px)
  auto [sw2, p2] = run(120, [](int) { return 80.f; });
  EXPECT_EQ(sw2, 1);
  EXPECT_EQ(p2.flags & NVFX_PICTURE_HALF, 0);
  EXPECT_EQ(p2.lod_size, 128);
  // above it again: at once
  auto [sw3, p3] = run(4, [](int) { return 130.f; });
  EXPECT_EQ(sw3, 1);
  EXPECT_NE(p3.flags & NVFX_PICTURE_HALF, 0);
  EXPECT_EQ(p3.lod_size, 64);
}

// Never for a simulation locked to a level, nor for a model that opts out (half_px 0); a model's own threshold; a share
// of the viewport's height, once there is a camera; the plan counts a drawing at half size at the level below's cost.
TEST(Manager, HalfSizeLocksModelsShareAndCost) {
  const Effect e(tiny_effect(true));
  nvfx_manager_desc d = small_desc();
  d.half_px = 100.f;
  Mgr g(d);
  int follows = -1, never = -1, own = -1;
  auto md = small_model();
  ASSERT_EQ(nvfx_manager_add_model(g.m, e.fx, &md, &follows), NVFX_OK);
  md.half_px = 0.f;
  ASSERT_EQ(nvfx_manager_add_model(g.m, e.fx, &md, &never), NVFX_OK);
  md.half_px = 60.f;
  ASSERT_EQ(nvfx_manager_add_model(g.m, e.fx, &md, &own), NVFX_OK);
  const nvfx_handle a = spawn(g.m, follows, 1, 120.f), locked = spawn(g.m, follows, 2, 120.f), b = spawn(g.m, never, 3, 120.f), c = spawn(g.m, own, 4, 70.f),
                    small = spawn(g.m, follows, 6, 70.f);
  ASSERT_EQ(nvfx_manager_set_lod_lock(g.m, locked, 128, 1), NVFX_OK);
  for (int t = 0; t < 30; ++t) ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 60.0), NVFX_OK);
  const auto half = [&](nvfx_handle h) {
    nvfx_manager_picture p;
    EXPECT_EQ(nvfx_manager_get_picture(g.m, h, &p), NVFX_OK);
    EXPECT_NE(p.rgba, nullptr);
    return std::pair{(p.flags & NVFX_PICTURE_HALF) != 0, p.lod_size};
  };
  EXPECT_EQ(half(a), std::pair(true, 64));
  EXPECT_EQ(half(locked), std::pair(false, 128));
  EXPECT_EQ(half(b), std::pair(false, 128));
  EXPECT_EQ(half(c), std::pair(true, 64));       // its own threshold, 60
  EXPECT_EQ(half(small), std::pair(false, 128));  // 70 < the manager's 100
  nvfx_manager_stats st;
  nvfx_manager_get_stats(g.m, &st);
  EXPECT_EQ(st.half, 2);
  // a share of the viewport's height: 0.2 of 360 px is 72, so the 70 px one is not, until the viewport is 400 px high
  nvfx_manager_desc ds = small_desc();
  ds.half_px = 0.f;
  ds.half_share = 0.2f;
  Mgr gs(ds);
  int ms = -1;
  ASSERT_EQ(nvfx_manager_add_model(gs.m, e.fx, nullptr, &ms), NVFX_OK);
  const nvfx_handle q = spawn(gs.m, ms, 7, 75.f), r = spawn(gs.m, ms, 8, 70.f);
  const auto half_s = [&](nvfx_handle h) {
    nvfx_manager_picture p;
    nvfx_manager_get_picture(gs.m, h, &p);
    return (p.flags & NVFX_PICTURE_HALF) != 0;
  };
  for (int t = 0; t < 20; ++t) ASSERT_EQ(nvfx_manager_tick(gs.m, 1.0 / 60.0), NVFX_OK);
  EXPECT_FALSE(half_s(q));  // no camera: no viewport to take a share of
  const float pos[3] = {0.f, 0.f, 0.f}, fwd[3] = {0.f, 0.f, 1.f}, up[3] = {0.f, 1.f, 0.f};
  ASSERT_EQ(nvfx_manager_set_camera(gs.m, pos, fwd, up, 1.f, 640.f, 360.f), NVFX_OK);
  for (int t = 0; t < 20; ++t) ASSERT_EQ(nvfx_manager_tick(gs.m, 1.0 / 60.0), NVFX_OK);
  EXPECT_TRUE(half_s(q));
  EXPECT_FALSE(half_s(r));
  // the plan: four 128 px pictures drawn at 64 cost what four drawings at 64 do, and the 2x
  double off_plan = 0.0;
  for (const bool on : {false, true}) {
    nvfx_manager_desc dc = small_desc();
    dc.half_px = on ? 100.f : 0.f;
    dc.half_upscale = NVFX_UPSCALE_BILINEAR;  // the 2x counted too
    Mgr gc(dc);
    int mc = -1;
    ASSERT_EQ(nvfx_manager_add_model(gc.m, e.fx, nullptr, &mc), NVFX_OK);
    ASSERT_EQ(nvfx_manager_set_cost(gc.m, mc, 64, 0.1f, 0.1f), NVFX_OK);
    ASSERT_EQ(nvfx_manager_set_cost(gc.m, mc, 128, 0.4f, 0.4f), NVFX_OK);
    for (int i = 0; i < 4; ++i) spawn(gc.m, mc, static_cast<std::uint64_t>(20 + i), 120.f);
    for (int t = 0; t < 10; ++t) ASSERT_EQ(nvfx_manager_tick(gc.m, 1.0 / 60.0), NVFX_OK);
    nvfx_manager_get_stats(gc.m, &st);
    if (!on) {
      off_plan = st.plan_ms;
      EXPECT_EQ(st.half, 0);
    } else {
      EXPECT_EQ(st.half, 4);
      EXPECT_LT(st.plan_ms, 0.3 * off_plan);
      EXPECT_GT(st.plan_ms, 0.25 * off_plan);
    }
  }
}

// Under a short budget a simulation drawn at half size goes down with its drawing (a 128 px picture drawn at 64, then a
// 64 px one drawn at 32), within the budget; with room again it comes back; drawing at full size again waits for room.
TEST(Manager, HalfSizeUnderAShortBudget) {
  const Effect e(tiny_effect(true));
  nvfx_manager_desc d = small_desc();
  d.half_px = 100.f;
  d.half_upscale = NVFX_UPSCALE_BILINEAR;
  d.proxies = 0;
  d.min_dwell = 0.1f;
  d.budget_ms = 0.45f;
  Mgr g(d);
  int model = -1;
  const auto md = small_model();
  ASSERT_EQ(nvfx_manager_add_model(g.m, e.fx, &md, &model), NVFX_OK);
  for (const auto& [size, step] : {std::pair{32, 0.05f}, std::pair{64, 0.2f}, std::pair{128, 0.8f}}) ASSERT_EQ(nvfx_manager_set_cost(g.m, model, size, step, step), NVFX_OK);
  std::vector<nvfx_handle> hs;
  for (int i = 0; i < 4; ++i) hs.push_back(spawn(g.m, model, static_cast<std::uint64_t>(40 + i), 120.f));
  nvfx_manager_set_priority(g.m, hs[0], 10.f);
  const auto run = [&](int ticks) {
    nvfx_manager_stats st{};
    for (int t = 0; t < ticks; ++t) {
      EXPECT_EQ(nvfx_manager_tick(g.m, 1.0 / 60.0), NVFX_OK);
      nvfx_manager_get_stats(g.m, &st);
    }
    return st;
  };
  const auto pic = [&](nvfx_handle h) {
    nvfx_manager_picture p;
    EXPECT_EQ(nvfx_manager_get_picture(g.m, h, &p), NVFX_OK);
    return p;
  };
  nvfx_manager_stats st = run(90);
  EXPECT_LE(st.plan_ms, 0.45 * 1.01);
  EXPECT_EQ(st.half, 4);  // every one is still drawn at half size
  const nvfx_manager_picture top = pic(hs[0]);
  EXPECT_EQ(top.lod_size, 64);  // the important one keeps its 128 px picture drawn at 64
  EXPECT_EQ(top.size, 128);
  int lowered = 0;
  for (const nvfx_handle h : hs) {
    const nvfx_manager_picture p = pic(h);
    EXPECT_NE(p.flags & NVFX_PICTURE_HALF, 0);
    EXPECT_EQ(p.size, 2 * p.lod_size);
    lowered += p.lod_size == 32 ? 1 : 0;
  }
  EXPECT_GT(lowered, 0);  // some went down to a 64 px picture drawn at 32
  // room again: back up, drawn at half size
  ASSERT_EQ(nvfx_manager_set_budget(g.m, 1000.f), NVFX_OK);
  st = run(240);
  EXPECT_EQ(st.half, 4);
  for (const nvfx_handle h : hs) EXPECT_EQ(pic(h).lod_size, 64);
  // smaller on screen, with a short budget again: drawing them at full size would not fit, so they wait at half size
  ASSERT_EQ(nvfx_manager_set_budget(g.m, 0.2f), NVFX_OK);
  for (const nvfx_handle h : hs) nvfx_manager_set_screen_size(g.m, h, 80.f, 1);
  st = run(60);
  for (const nvfx_handle h : hs) EXPECT_LE(pic(h).lod_size, 64);
  EXPECT_LT(st.lod[2][0] + st.lod[2][1] + st.lod[2][2] + st.lod[2][3], 4);
  // and with room they are drawn at full size, one rung a tick
  ASSERT_EQ(nvfx_manager_set_budget(g.m, 1000.f), NVFX_OK);
  st = run(240);
  EXPECT_EQ(st.half, 0);
  for (const nvfx_handle h : hs) {
    EXPECT_EQ(pic(h).lod_size, 128);
    EXPECT_EQ(pic(h).flags & NVFX_PICTURE_HALF, 0);
  }
}

TEST(Manager, FixedLevelIsNvfxRenderLooping) { expect_like_render(true, 64); }
TEST(Manager, FixedLevelIsNvfxRenderOneShot) { expect_like_render(false, 32); }
TEST(Manager, FixedLevelIsNvfxRender128) { expect_like_render(true, 128); }
TEST(Manager, FixedLevelIsNvfxRenderAtInt8) {
  expect_like_render(true, 64, NVFX_PRECISION_INT8);
  expect_like_render(false, 32, NVFX_PRECISION_INT8);
}

// A scene that exercises everything (sizes moving across levels, a tight budget, proxies, copies offset in time,
// pauses, spawns and stops, controls changed on shared and own simulations) gives the same pictures on 1 and 3 threads
// and with the engine running the jobs on two threads of its own.
struct Seen {
  int proxied = 0, skipped = 0, paused = 0, interpolated = 0, shared = 0, levels = 0, half = 0;
};

std::vector<std::uint64_t> busy_scene(int threads, bool external, Seen* seen = nullptr, float half_px = 0.f, nvfx_upscale half_upscale = NVFX_UPSCALE_BILINEAR) {
  const Effect loopfx(tiny_effect(true)), shotfx(tiny_effect(false, 5));
  nvfx_manager_desc d = small_desc(external ? 0 : threads);
  d.half_px = half_px;
  d.half_upscale = half_upscale;
  d.job_slots = 2;
  d.budget_ms = 1.2f;
  d.proxy_dwell = 0.3f;
  d.min_dwell = 0.2f;
  d.park_after = 0.3f;
  d.cull_after = 1.5f;
  Mgr g(d);
  int ml = -1, ms = -1;
  const auto md = small_model(2.f);
  EXPECT_EQ(nvfx_manager_add_model(g.m, loopfx.fx, &md, &ml), NVFX_OK);
  EXPECT_EQ(nvfx_manager_add_model(g.m, shotfx.fx, &md, &ms), NVFX_OK);
  std::vector<nvfx_handle> hs;
  for (int i = 0; i < 14; ++i) {
    const float px = 20.f + 9.f * static_cast<float>(i);
    hs.push_back(spawn(g.m, ml, static_cast<std::uint64_t>(i % 3), px, {0.3f + 0.05f * static_cast<float>(i % 3), 0.5f, 0.5f}, i % 4 == 3 ? 0.4f : 0.f));
  }
  std::vector<std::uint64_t> out;
  for (int t = 0; t < 160; ++t) {
    for (std::size_t i = 0; i < hs.size(); ++i) {
      const float px = 16.f + 60.f * (1.f + std::sin(0.03f * static_cast<float>(t) + static_cast<float>(i)));
      nvfx_manager_set_screen_size(g.m, hs[i], px, (t / 40 + static_cast<int>(i)) % 5 != 0);
    }
    if (t == 30) nvfx_manager_set_priority(g.m, hs[2], 4.f);
    if (t == 50) {
      const float c[3] = {0.8f, 0.4f, 0.5f};
      nvfx_manager_set_controls(g.m, hs[0], c, 3);  // shared: forks
      nvfx_manager_set_controls(g.m, hs[13], c, 3);
    }
    if (t % 25 == 10) hs.push_back(spawn(g.m, ms, static_cast<std::uint64_t>(t), 70.f, {0.5f, 0.5f, 0.5f}));
    if (t == 90) {
      nvfx_manager_stop(g.m, hs[5]);
      hs.erase(hs.begin() + 5);
    }
    if (external) {
      int jobs = 0;
      EXPECT_EQ(nvfx_manager_tick_begin(g.m, 1.0 / 60.0, &jobs), NVFX_OK);
      std::thread other([&] { nvfx_manager_run_jobs(g.m, 1); });
      nvfx_manager_run_jobs(g.m, 0);
      other.join();
      EXPECT_EQ(nvfx_manager_tick_end(g.m), NVFX_OK);
    } else {
      EXPECT_EQ(nvfx_manager_tick(g.m, 1.0 / 60.0), NVFX_OK);
    }
    std::uint64_t h = 1469598103934665603ULL;
    for (const nvfx_handle x : hs) {
      nvfx_manager_picture p;
      EXPECT_EQ(nvfx_manager_get_picture(g.m, x, &p), NVFX_OK);
      if (p.rgba) h = hash(Bytes(p.rgba, p.rgba + static_cast<std::size_t>(p.size) * static_cast<std::size_t>(p.size) * 4), h ^ static_cast<std::uint64_t>(p.flags));
    }
    nvfx_manager_stats st;
    nvfx_manager_get_stats(g.m, &st);
    h ^= static_cast<std::uint64_t>(st.updated) * 31u + static_cast<std::uint64_t>(st.proxied) * 7u + static_cast<std::uint64_t>(st.skipped);
    out.push_back(h);
    if (seen) {
      seen->proxied = std::max(seen->proxied, st.proxied);
      seen->skipped += st.skipped;
      seen->paused = std::max(seen->paused, st.sims_paused);
      seen->interpolated += st.interpolated;
      seen->shared = std::max(seen->shared, st.shared_viewers);
      int levels = 0;
      for (const auto& row : st.lod) {
        for (const int n : row) levels += n > 0 ? 1 : 0;
      }
      seen->levels = std::max(seen->levels, levels);
      seen->half = std::max(seen->half, st.half);
    }
  }
  return out;
}

TEST(Manager, SamePicturesOnAnyNumberOfThreads) {
  Seen seen;
  const auto one = busy_scene(1, false, &seen);
  EXPECT_GT(seen.proxied, 0);  // the scene does exercise what it should
  EXPECT_GT(seen.skipped, 0);
  EXPECT_GT(seen.paused, 0);
  EXPECT_GT(seen.interpolated, 0);
  EXPECT_GT(seen.shared, 0);
  EXPECT_GE(seen.levels, 3);
  EXPECT_EQ(one, busy_scene(3, false));
  EXPECT_EQ(one, busy_scene(0, true));
}

// The same with large simulations drawn at half size (with the bilinear 2x and without it): what is drawn at half size
// depends only on the sizes given, so the pictures are the same on any number of threads.
TEST(Manager, SamePicturesOnAnyNumberOfThreadsAtHalfSize) {
  for (const nvfx_upscale mode : {NVFX_UPSCALE_BILINEAR, NVFX_UPSCALE_OFF}) {
    Seen seen;
    const auto one = busy_scene(1, false, &seen, 70.f, mode);
    EXPECT_GT(seen.half, 0);
    EXPECT_GT(seen.proxied, 0);
    EXPECT_GT(seen.skipped, 0);
    EXPECT_EQ(one, busy_scene(3, false, nullptr, 70.f, mode));
    EXPECT_EQ(one, busy_scene(0, true, nullptr, 70.f, mode));
    EXPECT_NE(one, busy_scene(1, false));  // and it does draw something else than the policy off
  }
}

// RolloutRunner::transfer_from: the coarse state continues exactly at another size.
TEST(Manager, TransferKeepsTheCoarseStateExactly) {
  const Effect e(tiny_effect(true));
  const nfx::rt::RolloutEffect* re = nfx::rt::rollout_of(e.fx);
  ASSERT_NE(re, nullptr);
  const std::vector<float> c{0.4f, 0.6f, 0.5f};
  auto a = nfx::rt::isa_base::make_rollout(*re, 64);
  a->start(1, c, 9);
  for (int i = 0; i < 12; ++i) a->step(c, 9);
  for (const int size : {32, 64, 128}) {
    auto b = nfx::rt::isa_base::make_rollout(*re, size);
    auto ref = nfx::rt::isa_base::make_rollout(*re, 64);
    b->start(0, c, 1);
    b->transfer_from(*a);
    ref->transfer_from(*a);
    EXPECT_EQ(b->time(), a->time());
    for (int i = 0; i < 20; ++i) {
      b->step(c, 9);
      ref->step(c, 9);
    }
    const auto x = b->coarse(), y = ref->coarse();
    ASSERT_EQ(x.size(), y.size());
    EXPECT_EQ(std::memcmp(x.data(), y.data(), x.size() * sizeof(float)), 0) << size;
    if (size == 64) {
      EXPECT_EQ(std::memcmp(b->fine_heat().data(), ref->fine_heat().data(), b->fine_heat().size() * sizeof(float)), 0);
    }
  }
  auto other = nfx::rt::isa_avx2::make_rollout(*re, 64);
  EXPECT_THROW(other->transfer_from(*a), std::invalid_argument);  // another ISA's runner
}

// Copies with the same model, controls and seed share one simulation; a copy offset in time shows the frame its
// simulation showed that many frames before (from the ring, at copy size: exact at 30 Hz and the copy's size). An
// instance that does not share plays the same run (same model, controls and seed) on a simulation of its own.
TEST(Manager, CopiesShareOneSimulationAndOffsetsAreEarlierFrames) {
  const Effect e(tiny_effect(true));
  Mgr g(small_desc());
  int model = -1;
  const auto md = small_model();
  ASSERT_EQ(nvfx_manager_add_model(g.m, e.fx, &md, &model), NVFX_OK);
  const nvfx_handle a = spawn(g.m, model, 4, 64.f), b = spawn(g.m, model, 4, 30.f), c = spawn(g.m, model, 4, 64.f, {0.45f, 0.55f, 0.6f}, 0.2f);
  const nvfx_handle own = spawn(g.m, model, 4, 64.f, {0.45f, 0.55f, 0.6f}, 0.f, 0);
  for (const nvfx_handle h : {a, b, c, own}) ASSERT_EQ(nvfx_manager_set_lod_lock(g.m, h, 64, 1), NVFX_OK);
  std::map<long, Bytes> shared, alone;
  std::vector<Bytes> history;
  int checked = 0;
  for (int t = 0; t < 90; ++t) {
    ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
    nvfx_manager_stats st;
    nvfx_manager_get_stats(g.m, &st);
    EXPECT_LE(st.sims, 2);  // the three copies, and the one that does not share
    nvfx_manager_picture pa, po;
    nvfx_manager_get_picture(g.m, a, &pa);
    nvfx_manager_get_picture(g.m, own, &po);
    const Bytes ba = picture(g.m, a), bb = picture(g.m, b), bc = picture(g.m, c), bo = picture(g.m, own);
    if (!bo.empty()) alone[std::lround(po.effect_time * 30.0)] = bo;
    if (ba.empty()) continue;
    shared[std::lround(pa.effect_time * 30.0)] = ba;
    EXPECT_EQ(ba, bb);
    EXPECT_NE(pa.flags & NVFX_PICTURE_SHARED, 0);
    history.push_back(ba);
    if (history.size() > 6) {
      EXPECT_EQ(bc, history[history.size() - 7]);  // 0.2 s at 30 fps: 6 frames before
      ++checked;
    }
  }
  EXPECT_GT(checked, 50);
  int same = 0;
  for (const auto& [f, px] : shared) {
    if (const auto it = alone.find(f); it != alone.end()) {
      EXPECT_EQ(px, it->second) << f;
      ++same;
    }
  }
  EXPECT_GT(same, 50);
}

// Offscreen, a simulation pauses: its clock stops, and when it is seen again it continues where it stopped.
TEST(Manager, PausedSimulationsResumeWarm) {
  const Effect e(tiny_effect(true));
  const auto run = [&](bool hide) {
    nvfx_manager_desc d = small_desc();
    d.park_after = 10.f;
    Mgr g(d);
    int model = -1;
    const auto md = small_model();
    EXPECT_EQ(nvfx_manager_add_model(g.m, e.fx, &md, &model), NVFX_OK);
    const nvfx_handle h = spawn(g.m, model, 11, 64.f);
    nvfx_manager_set_lod_lock(g.m, h, 64, 2);
    int visible_ticks = 0;
    for (int t = 0; visible_ticks < 70; ++t) {
      const bool vis = !(hide && t >= 30 && t < 50);
      nvfx_manager_set_screen_size(g.m, h, 64.f, vis ? 1 : 0);
      EXPECT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
      visible_ticks += vis ? 1 : 0;
      if (hide && t == 40) {
        nvfx_manager_stats st;
        nvfx_manager_get_stats(g.m, &st);
        EXPECT_EQ(st.sims_paused, 1);
      }
    }
    return picture(g.m, h);
  };
  const Bytes straight = run(false), resumed = run(true);
  ASSERT_FALSE(straight.empty());
  EXPECT_EQ(straight, resumed);
}

// Off screen long enough, a simulation is culled (released); seen again, a looping instance starts a new run. With no
// simulation free, a new instance takes the one paused longest.
TEST(Manager, CullAndEvict) {
  const Effect e(tiny_effect(true));
  nvfx_manager_desc d = small_desc();
  d.cull_after = 0.5f;
  d.proxies = 0;
  Mgr g(d);
  int model = -1;
  auto md = small_model();
  md.max_sims = 2;
  ASSERT_EQ(nvfx_manager_add_model(g.m, e.fx, &md, &model), NVFX_OK);
  const nvfx_handle a = spawn(g.m, model, 1, 64.f), b = spawn(g.m, model, 2, 64.f);
  for (int t = 0; t < 10; ++t) ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
  nvfx_manager_set_screen_size(g.m, a, 64.f, 0);  // a leaves the screen
  for (int t = 0; t < 5; ++t) ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
  nvfx_manager_stats st;
  nvfx_manager_get_stats(g.m, &st);
  EXPECT_EQ(st.sims, 2);
  EXPECT_EQ(st.sims_paused, 1);
  const nvfx_handle c = spawn(g.m, model, 3, 64.f);  // none free: takes a's
  for (int t = 0; t < 5; ++t) ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
  nvfx_manager_picture pa, pc;
  nvfx_manager_get_picture(g.m, a, &pa);
  nvfx_manager_get_picture(g.m, c, &pc);
  EXPECT_EQ(pa.rgba, nullptr);
  EXPECT_NE(pc.rgba, nullptr);
  nvfx_manager_set_screen_size(g.m, b, 64.f, 0);  // b leaves and stays away past cull_after
  for (int t = 0; t < 30; ++t) ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
  nvfx_manager_get_stats(g.m, &st);
  EXPECT_EQ(st.sims, 1);  // only c's
  nvfx_manager_set_screen_size(g.m, b, 64.f, 1);  // back: a new run
  for (int t = 0; t < 5; ++t) ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
  nvfx_manager_picture pb;
  nvfx_manager_get_picture(g.m, b, &pb);
  EXPECT_NE(pb.rgba, nullptr);
  EXPECT_LT(pb.effect_time, 0.5);
}

// The planner keeps the plan within the budget by lowering the least important simulations first.
TEST(Manager, TheBudgetLowersTheLeastImportantFirst) {
  const Effect e(tiny_effect(true));
  for (const float budget : {1000.f, 4.f}) {
    nvfx_manager_desc d = small_desc();
    d.budget_ms = budget;
    d.proxies = 0;
    Mgr g(d);
    int model = -1;
    const auto md = small_model();
    ASSERT_EQ(nvfx_manager_add_model(g.m, e.fx, &md, &model), NVFX_OK);
    std::vector<nvfx_handle> hs;
    for (int i = 0; i < 8; ++i) hs.push_back(spawn(g.m, model, static_cast<std::uint64_t>(100 + i), 128.f));
    nvfx_manager_set_priority(g.m, hs[0], 10.f);
    nvfx_manager_stats st{};
    for (int t = 0; t < 60; ++t) {
      ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
      nvfx_manager_get_stats(g.m, &st);
    }
    nvfx_manager_picture top, low;
    nvfx_manager_get_picture(g.m, hs[0], &top);
    nvfx_manager_get_picture(g.m, hs[7], &low);
    if (budget > 100.f) {
      EXPECT_EQ(st.lod[2][0], 8);  // everything at 128 px and 30 Hz
    } else {
      EXPECT_LE(st.plan_ms, static_cast<double>(budget) * 1.01);
      EXPECT_EQ(top.lod_size, 128);  // the important one keeps its level
      EXPECT_EQ(top.lod_rate_hz, 30.f);
      EXPECT_LT(low.lod_rate_hz * static_cast<float>(low.lod_size), top.lod_rate_hz * static_cast<float>(top.lod_size));
    }
  }
}

// With proxies, a tight budget shows unimportant looping instances from shared proxy simulations.
TEST(Manager, ProxiesTakeTheUnimportantWhenTheBudgetIsShort) {
  const Effect e(tiny_effect(true));
  nvfx_manager_desc d = small_desc();
  d.budget_ms = 1.5f;
  d.proxy_dwell = 0.2f;
  Mgr g(d);
  int model = -1;
  const auto md = small_model();
  ASSERT_EQ(nvfx_manager_add_model(g.m, e.fx, &md, &model), NVFX_OK);
  std::vector<nvfx_handle> hs;
  for (int i = 0; i < 12; ++i) hs.push_back(spawn(g.m, model, static_cast<std::uint64_t>(200 + i), 20.f + 5.f * static_cast<float>(i)));
  nvfx_manager_stats st{};
  for (int t = 0; t < 120; ++t) {
    ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
    nvfx_manager_get_stats(g.m, &st);
  }
  EXPECT_GT(st.proxied, 0);
  EXPECT_GT(st.proxy_sims, 0);
  EXPECT_LE(st.plan_ms, 1.5 * 1.05);
  nvfx_handle locked = 0;
  for (const nvfx_handle h : hs) {
    nvfx_manager_picture p;
    nvfx_manager_get_picture(g.m, h, &p);
    EXPECT_NE(p.rgba, nullptr);  // everyone is shown
    if ((p.flags & NVFX_PICTURE_PROXY) && !locked) locked = h;
  }
  // a proxied instance locked to a level gets a simulation of its own (and fades across once it is ready)
  ASSERT_NE(locked, 0u);
  ASSERT_EQ(nvfx_manager_set_lod_lock(g.m, locked, 64, 2), NVFX_OK);
  for (int t = 0; t < 30; ++t) ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
  nvfx_manager_picture p;
  nvfx_manager_get_picture(g.m, locked, &p);
  EXPECT_EQ(p.flags & NVFX_PICTURE_PROXY, 0);
  EXPECT_EQ(p.lod_size, 64);
  EXPECT_EQ(p.lod_rate_hz, 15.f);
}

// A size wobbling around a threshold, within the hysteresis, does not change the level.
TEST(Manager, LevelsDoNotFlickerAroundAThreshold) {
  const Effect e(tiny_effect(true));
  Mgr g(small_desc());
  int model = -1;
  const auto md = small_model();
  ASSERT_EQ(nvfx_manager_add_model(g.m, e.fx, &md, &model), NVFX_OK);
  const nvfx_handle h = spawn(g.m, model, 1, 64.f);
  int changes = 0, last = -1;
  for (int t = 0; t < 300; ++t) {
    nvfx_manager_set_screen_size(g.m, h, 64.f * (1.f + 0.08f * std::sin(0.7f * static_cast<float>(t))), 1);
    ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 60.0), NVFX_OK);
    nvfx_manager_picture p;
    nvfx_manager_get_picture(g.m, h, &p);
    const int lod = p.lod_size * 10 + static_cast<int>(p.lod_rate_hz);
    if (p.rgba && last >= 0 && lod != last) ++changes;
    if (p.rgba) last = lod;
  }
  EXPECT_EQ(changes, 0);
  // and a real move changes it
  for (int t = 0; t < 120; ++t) {
    nvfx_manager_set_screen_size(g.m, h, 120.f, 1);
    ASSERT_EQ(nvfx_manager_tick(g.m, 1.0 / 60.0), NVFX_OK);
  }
  nvfx_manager_picture p;
  nvfx_manager_get_picture(g.m, h, &p);
  EXPECT_EQ(p.lod_size, 128);
}

TEST(Manager, InterpolationKernels) {
  const int S = 64;
  Bytes a(S * S * 4), b(a.size()), o(a.size()), q(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<std::uint8_t>((i * 37) % 251);
    b[i] = static_cast<std::uint8_t>((i * 11) % 241);
  }
  nfx::mgr::crossfade(a.data(), b.data(), S, 0.f, o.data());
  EXPECT_EQ(o, a);
  nfx::mgr::crossfade(a.data(), b.data(), S, 1.f, o.data());
  EXPECT_EQ(o, b);
  std::vector<float> uv(16 * 16 * 2, 0.f);
  const nfx::mgr::Motion still{uv.data(), 16, 4.f};
  nfx::mgr::motion_blend(a.data(), b.data(), S, 0.4f, still, o.data());
  nfx::mgr::crossfade(a.data(), b.data(), S, 0.4f, q.data());
  for (std::size_t i = 0; i < o.size(); ++i) ASSERT_LE(std::abs(o[i] - q[i]), 1) << i;
  // a uniform displacement of 2 cells right (8 pixels at 4 per cell) moves a's content: at w = 1, b is shown, and at
  // w = 0, a; at w = 0.5, a pixel comes from a 4 pixels left and from b 4 pixels right
  for (std::size_t i = 0; i < uv.size(); i += 2) uv[i] = 2.f;
  const nfx::mgr::Motion right{uv.data(), 16, 4.f};
  nfx::mgr::motion_blend(a.data(), a.data(), S, 0.5f, right, o.data());
  for (int y = 0; y < S; ++y) {
    for (int x = 8; x < S - 8; ++x) {
      for (int c = 0; c < 4; ++c) {
        const int want = (a[(static_cast<std::size_t>(y) * S + static_cast<std::size_t>(x - 4)) * 4 + static_cast<std::size_t>(c)] +
                          a[(static_cast<std::size_t>(y) * S + static_cast<std::size_t>(x + 4)) * 4 + static_cast<std::size_t>(c)] + 1) / 2;
        ASSERT_LE(std::abs(o[(static_cast<std::size_t>(y) * S + static_cast<std::size_t>(x)) * 4 + static_cast<std::size_t>(c)] - want), 1);
      }
    }
  }
  // the AVX2 kernel and the portable one agree (to a level: fused multiply-adds)
  for (std::size_t i = 0; i < uv.size(); ++i) uv[i] = 0.7f * std::sin(0.37f * static_cast<float>(i));
  const nfx::mgr::Motion wavy{uv.data(), 16, 4.f};
  nfx::mgr::motion_blend(a.data(), b.data(), S, 0.3f, wavy, o.data());
  nfx::mgr::force_portable_kernels(true);
  nfx::mgr::motion_blend(a.data(), b.data(), S, 0.3f, wavy, q.data());
  nfx::mgr::force_portable_kernels(false);
  int worst = 0;
  for (std::size_t i = 0; i < o.size(); ++i) worst = std::max(worst, std::abs(o[i] - q[i]));
  EXPECT_LE(worst, 1);
  // resampling: a box average down, then a copy
  Bytes small(32 * 32 * 4);
  nfx::mgr::resample(a.data(), S, small.data(), 32);
  EXPECT_EQ(small[0], static_cast<std::uint8_t>((a[0] + a[4] + a[S * 4] + a[S * 4 + 4] + 2) / 4));
}

TEST(Manager, ApiErrors) {
  const Effect e(tiny_effect(true));
  nvfx_manager* m = nullptr;
  nvfx_manager_desc d = small_desc();
  d.copy_size = 48;
  EXPECT_EQ(nvfx_manager_create(&d, &m), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_manager_create(nullptr, &m), NVFX_OK);  // defaults
  nvfx_manager_free(m);
  for (const auto& bad : std::vector<std::function<void(nvfx_manager_desc&)>>{
           [](nvfx_manager_desc& x) { x.half_upscale = NVFX_UPSCALE_LEARNED; },  // experimental: never the manager's
           [](nvfx_manager_desc& x) { x.half_upscale = static_cast<nvfx_upscale>(3); },
           [](nvfx_manager_desc& x) { x.half_px = -1.f; },
           [](nvfx_manager_desc& x) { x.half_px = std::nanf(""); },
           [](nvfx_manager_desc& x) { x.half_share = -0.5f; }}) {
    nvfx_manager_desc x = small_desc();
    bad(x);
    EXPECT_EQ(nvfx_manager_create(&x, &m), NVFX_ERROR_ARGUMENT);
  }
  {
    Mgr ok(small_desc());
    nvfx_manager_model_desc md;
    nvfx_manager_model_desc_init(&md);
    EXPECT_EQ(md.half_px, -1.f);  // the manager's thresholds
    md.half_px = std::nanf("");
    int mi = -1;
    EXPECT_EQ(nvfx_manager_add_model(ok.m, e.fx, &md, &mi), NVFX_ERROR_ARGUMENT);
  }
  Mgr g(small_desc());
  int model = -1;
  EXPECT_EQ(nvfx_manager_add_model(g.m, nullptr, nullptr, &model), NVFX_ERROR_ARGUMENT);
  ASSERT_EQ(nvfx_manager_add_model(g.m, e.fx, nullptr, &model), NVFX_OK);
  nvfx_spawn_desc s;
  nvfx_spawn_desc_init(&s);
  s.model = 7;
  nvfx_handle h = 0;
  EXPECT_EQ(nvfx_manager_spawn(g.m, &s, &h), NVFX_ERROR_ARGUMENT);
  s.model = model;
  ASSERT_EQ(nvfx_manager_spawn(g.m, &s, &h), NVFX_OK);
  EXPECT_NE(h, 0u);
  EXPECT_EQ(nvfx_manager_set_lod_lock(g.m, h, 48, 1), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_manager_set_lod_lock(g.m, h, 256, 1), NVFX_ERROR_ARGUMENT);  // max_size 128: no slots
  EXPECT_EQ(nvfx_manager_set_lod_lock(g.m, h, 64, 5), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_manager_tick(g.m, -1.0), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
  EXPECT_EQ(nvfx_manager_stop(g.m, h), NVFX_OK);
  nvfx_manager_picture p;
  EXPECT_EQ(nvfx_manager_get_picture(g.m, h, &p), NVFX_ERROR_ARGUMENT);  // stopped
  EXPECT_EQ(nvfx_manager_stop(g.m, h), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_manager_tick(g.m, 1.0 / 30.0), NVFX_OK);
  nvfx_handle h2 = 0;
  ASSERT_EQ(nvfx_manager_spawn(g.m, &s, &h2), NVFX_OK);
  EXPECT_NE(h2, h);  // the slot is reused with another generation
  EXPECT_EQ(nvfx_manager_get_picture(g.m, h, &p), NVFX_ERROR_ARGUMENT);
  int jobs = 0;
  EXPECT_EQ(nvfx_manager_tick_begin(g.m, 0.0, &jobs), NVFX_OK);
  EXPECT_EQ(nvfx_manager_spawn(g.m, &s, &h2), NVFX_ERROR_ARGUMENT);  // not during a tick
  EXPECT_EQ(nvfx_manager_tick_begin(g.m, 0.0, &jobs), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_manager_run_jobs(g.m, 0), NVFX_OK);
  EXPECT_EQ(nvfx_manager_tick_end(g.m), NVFX_OK);
  EXPECT_EQ(nvfx_manager_tick_end(g.m), NVFX_ERROR_ARGUMENT);
}

}  // namespace
