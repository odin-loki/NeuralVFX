// Study I round 2 (docs/COMPOSE.md §10): the explosion's first frames in the simulator's look. The look drawn from the
// runtime's fine fields is the simulator's renderer to the bit, and nvfx_instance_set_handoff shows it first and
// crossfades to the learned renderer, exactly as the runner's frames blended by hand.
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>
#include <neuralfx/sim.hpp>

#include "rt_common.hpp"
#include "rt_handoff.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace nfx;
namespace ro = nfx::rollout;

namespace {

// A small one-shot rollout effect named after the explosion (so the simulator has a look for it), with a start point
// that keeps fine fields (as tests/test_rollout.cpp's runtime model).
ro::Model explosion_model(const std::string& name = "explosion") {
  ro::Hyper h;
  h.res = 16;
  h.hidden = 8;
  h.memory = 2;
  h.jacobi = 10;
  h.render_hidden = 6;
  h.start_fine = 16;
  h.warmup = 3;
  h.n_age = 2;
  h.frames = 12;
  ro::Model m = ro::init_model(h, 21);
  std::mt19937_64 rng(22);
  std::normal_distribution<float> nd(0.f, 0.1f);
  for (float& w : m.step_w) w += nd(rng);
  for (float& w : m.render_w) w = 0.5f * w + 0.05f;
  m.effect = name;
  m.loop = false;
  m.control_names = {"intensity", "wind", "turbulence"};
  m.scale = {0.2f, 0.2f, 0.4f, 0.3f};
  m.lo = {-2.f, -2.f, 0.f, 0.f};
  m.hi = {2.f, 2.f, 3.f, 3.f};
  m.render_scale = {1.f, 1.f};
  const int R = h.res;
  ro::StartPoint sp;
  sp.controls = {0.6f, 0.5f, 0.6f};
  sp.seed = 1001;
  sp.coarse.resize(static_cast<std::size_t>(R) * R * ro::kPhys);
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      float* c = sp.coarse.data() + (static_cast<std::size_t>(y) * R + x) * ro::kPhys;
      c[0] = 0.1f * std::sin(0.5f * static_cast<float>(x + 1));
      c[1] = 0.2f + 0.1f * std::cos(0.4f * static_cast<float>(y));
      c[2] = std::max(0.f, 0.8f - 0.05f * static_cast<float>(std::abs(x - 8) + y));
      c[3] = std::max(0.f, 0.5f - 0.04f * static_cast<float>(std::abs(x - 7) + std::abs(y - 6)));
    }
  }
  sp.fine_t.resize(256);
  sp.fine_d.resize(256);
  for (int i = 0; i < 256; ++i) {
    sp.fine_t[static_cast<std::size_t>(i)] = std::max(0.f, 0.9f - 0.03f * static_cast<float>(std::abs(i % 16 - 8) + i / 16));
    sp.fine_d[static_cast<std::size_t>(i)] = std::max(0.f, 0.3f + 0.2f * std::sin(0.7f * static_cast<float>(i)));
  }
  m.starts.push_back(sp);
  ro::quantise_like_storage(m);
  return m;
}

std::string bytes_of(const ro::Model& m) {
  std::ostringstream os;
  EXPECT_TRUE(ro::save_model(os, m));
  return os.str();
}

struct Fx {
  nvfx_effect* e = nullptr;
  explicit Fx(const ro::Model& m) {
    const std::string b = bytes_of(m);
    EXPECT_EQ(nvfx_effect_load_memory(b.data(), b.size(), &e), NVFX_OK);
  }
  ~Fx() { nvfx_effect_free(e); }
  Fx(const Fx&) = delete;
  Fx& operator=(const Fx&) = delete;
};

}  // namespace

// Study I round 2 (docs/COMPOSE.md §10): the simulator's look drawn from the runtime's fine fields is the simulator's
// renderer to the bit, and the hand-off shows it first and crossfades to the learned renderer.
TEST(Handoff, SimLookIsTheSimulatorsRenderer) {
  for (const auto e : sim::kEffects) {
    sim::Params p;
    p.effect = e;
    p.size = 64;
    p.seed = 31;
    sim::Fluid run(p);
    for (int i = 0; i < (e == sim::Effect::explosion ? 6 : 40); ++i) run.step_frame();
    const sim::State st = run.state();
    sim::Fluid fresh(p);  // a solver of the fields' size holding them (study G's SimRenderer)
    fresh.set_state(st);
    std::vector<std::uint8_t> ref(static_cast<std::size_t>(64) * 64 * 4);
    fresh.render(ref);
    const std::size_t stride = 64 * 4 + 12;
    std::vector<std::uint8_t> out(stride * 64, 0xAB);
    rt::draw_sim_look(rt::sim_look_for(sim::effect_name(e)), st.temp, st.soot, 64, out.data(), stride);
    std::size_t lit = 0;
    for (int y = 0; y < 64; ++y) {
      for (int i = 0; i < 64 * 4; ++i) {
        const std::uint8_t a = ref[static_cast<std::size_t>(y) * 256 + static_cast<std::size_t>(i)], b = out[static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(i)];
        ASSERT_EQ(a, b) << sim::effect_name(e) << " row " << y << " byte " << i;
        lit += a > 0;
      }
      EXPECT_EQ(out[static_cast<std::size_t>(y) * stride + 256], 0xAB) << "wrote past the row";
    }
    EXPECT_GT(lit, 500u) << sim::effect_name(e) << ": the check would be vacuous";
  }
  EXPECT_EQ(rt::sim_look_for("test"), rt::SimLook::none);
}

TEST(Handoff, WeightsAndBlend) {
  EXPECT_EQ(rt::handoff_weight(0, 3, 2), 0.f);
  EXPECT_EQ(rt::handoff_weight(2, 3, 2), 0.f);
  EXPECT_FLOAT_EQ(rt::handoff_weight(3, 3, 2), 1.f / 3.f);
  EXPECT_FLOAT_EQ(rt::handoff_weight(4, 3, 2), 2.f / 3.f);
  EXPECT_EQ(rt::handoff_weight(5, 3, 2), 1.f);
  EXPECT_EQ(rt::handoff_weight(3, 3, 0), 1.f);
  EXPECT_EQ(rt::handoff_weight(0, 0, 0), 1.f);
  std::vector<std::uint8_t> learned{0, 10, 200, 255, 7, 8, 9, 10}, sim{255, 0, 100, 255, 7, 0, 0, 0};
  auto a = learned;
  rt::blend_handoff(a.data(), 8, sim.data(), 8, 2, 0.f);
  EXPECT_EQ(a, sim);
  a = learned;
  rt::blend_handoff(a.data(), 8, sim.data(), 8, 2, 1.f);
  EXPECT_EQ(a, learned);
  a = learned;
  rt::blend_handoff(a.data(), 8, sim.data(), 8, 2, 0.5f);
  EXPECT_EQ(a[0], 128);
  EXPECT_EQ(a[2], 150);
}

TEST(Handoff, RuntimeShowsTheSimulatorsLookFirst) {
  const ro::Model m = explosion_model();
  Fx fx(m);
  const int size = 32, N = 3, M = 2;
  ASSERT_EQ(nvfx_set_isa(NVFX_ISA_BASELINE), NVFX_OK);
  nvfx_instance *with = nullptr, *without = nullptr;
  ASSERT_EQ(nvfx_instance_create(fx.e, size, &with), NVFX_OK);
  ASSERT_EQ(nvfx_instance_create(fx.e, size, &without), NVFX_OK);
  EXPECT_EQ(nvfx_instance_set_handoff(with, -1, 2), NVFX_ERROR_ARGUMENT);
  ASSERT_EQ(nvfx_instance_set_handoff(with, N, M), NVFX_OK);
  for (nvfx_instance* in : {with, without}) {
    nvfx_instance_set_controls(in, m.starts[0].controls.data(), 3);
    ASSERT_EQ(nvfx_instance_set_variation(in, 0), NVFX_OK);
  }
  // the reference: the same runner stepped by hand, its learned frame and the simulator's look of its fields blended
  rt::RolloutEffect re;
  {
    std::istringstream is(bytes_of(m));
    auto loaded = ro::load_model(is);
    ASSERT_TRUE(loaded);
    re.m = std::move(*loaded);
  }
  auto runner = rt::isa_base::make_rollout(re, size);
  runner->begin(0, m.starts[0].seed);
  const std::size_t bytes = static_cast<std::size_t>(size) * size * 4, row = static_cast<std::size_t>(size) * 4;
  std::vector<std::uint8_t> a(bytes), b(bytes), learned(bytes), look(bytes);
  int differ = 0;
  for (int f = 0; f <= N + M + 1; ++f) {
    if (f > 0) runner->step(m.starts[0].controls, m.starts[0].seed);
    runner->render(rt::FrameInput{}, learned.data(), row);
    rt::draw_sim_look(rt::SimLook::explosion, runner->fine_heat(), runner->fine_soot(), size, look.data(), row);
    auto expect = learned;
    rt::blend_handoff(expect.data(), row, look.data(), row, size, rt::handoff_weight(f, N, M));
    ASSERT_EQ(nvfx_render(with, f / 30.0, a.data(), row), NVFX_OK);
    ASSERT_EQ(nvfx_render(without, f / 30.0, b.data(), row), NVFX_OK);
    EXPECT_EQ(b, learned) << "frame " << f;
    EXPECT_EQ(a, expect) << "frame " << f;
    if (f < N) {
      EXPECT_EQ(a, look) << "frame " << f;
    }
    if (f >= N + M) {
      EXPECT_EQ(a, b) << "frame " << f;
    }
    differ += a != b;
  }
  EXPECT_GE(differ, N + M - 1);
  ASSERT_EQ(nvfx_instance_set_handoff(with, 0, 0), NVFX_OK);  // off again
  ASSERT_EQ(nvfx_render(with, 1 / 30.0, a.data(), row), NVFX_OK);
  ASSERT_EQ(nvfx_render(without, 1 / 30.0, b.data(), row), NVFX_OK);
  EXPECT_EQ(a, b);
  nvfx_instance_free(with);
  nvfx_instance_free(without);
  nvfx_set_isa(NVFX_ISA_AUTO);
  // an effect the simulator does not make has no look to hand off from
  Fx other(explosion_model("test"));
  nvfx_instance* in = nullptr;
  ASSERT_EQ(nvfx_instance_create(other.e, size, &in), NVFX_OK);
  EXPECT_EQ(nvfx_instance_set_handoff(in, 3, 0), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_instance_set_handoff(in, 0, 0), NVFX_OK);
  nvfx_instance_free(in);
}
