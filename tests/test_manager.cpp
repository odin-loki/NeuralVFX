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
// shard timeline, keyframe by keyframe (through shard changes and their crossfades, and warm-ups).
void expect_like_render(bool loop, int size) {
  const Effect e(tiny_effect(loop));
  Mgr g(small_desc());
  int model = -1;
  const auto md = small_model(2.5f);
  ASSERT_EQ(nvfx_manager_add_model(g.m, e.fx, &md, &model), NVFX_OK);
  const std::array<float, 3> c{0.45f, 0.55f, 0.6f};
  const nvfx_handle h = spawn(g.m, model, 77, static_cast<float>(size), c);
  ASSERT_EQ(nvfx_manager_set_lod_lock(g.m, h, size, 1), NVFX_OK);
  nvfx_instance* in = nullptr;
  ASSERT_EQ(nvfx_instance_create(e.fx, size, &in), NVFX_OK);
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

TEST(Manager, FixedLevelIsNvfxRenderLooping) { expect_like_render(true, 64); }
TEST(Manager, FixedLevelIsNvfxRenderOneShot) { expect_like_render(false, 32); }
TEST(Manager, FixedLevelIsNvfxRender128) { expect_like_render(true, 128); }

// A scene that exercises everything (sizes moving across levels, a tight budget, proxies, copies offset in time,
// pauses, spawns and stops, controls changed on shared and own simulations) gives the same pictures on 1 and 3 threads
// and with the engine running the jobs on two threads of its own.
struct Seen {
  int proxied = 0, skipped = 0, paused = 0, interpolated = 0, shared = 0, levels = 0;
};

std::vector<std::uint64_t> busy_scene(int threads, bool external, Seen* seen = nullptr) {
  const Effect loopfx(tiny_effect(true)), shotfx(tiny_effect(false, 5));
  nvfx_manager_desc d = small_desc(external ? 0 : threads);
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
