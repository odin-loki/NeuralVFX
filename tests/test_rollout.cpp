// Rollout effects: the noise matches the simulation's forcing, the format round-trips, the trainer's forward pass is
// the reference's, its gradients (through time, burn-in and the profile loss) match finite differences, and the
// reference rollout stays finite.
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <random>
#include <sstream>

using namespace nfx;
using namespace nfx::rollout;

namespace {

Hyper tiny_hyper() {
  Hyper h;
  h.res = 8;
  h.hidden = 6;
  h.memory = 2;
  h.jacobi = 7;
  h.render_hidden = 5;
  h.warmup = 3;
  return h;
}

// A synthetic run on the tiny grid: smooth fields that move, so every operation is exercised.
Run tiny_run(const Hyper& h, int frames) {
  Run r;
  r.p.seed = 7;
  r.p.intensity = 0.3f;
  r.p.wind = 0.6f;
  r.p.turbulence = 0.8f;
  r.frames = frames;
  const int R = h.res;
  r.coarse.resize(static_cast<std::size_t>(frames) * R * R * kPhys);
  for (int f = 0; f < frames; ++f) {
    for (int y = 0; y < R; ++y) {
      for (int x = 0; x < R; ++x) {
        float* c = r.coarse.data() + ((static_cast<std::size_t>(f) * R + y) * R + x) * kPhys;
        const float t = 0.3f * static_cast<float>(f);
        // a steady drift keeps backtraces inside cells (bilinear interpolation has kinks at whole-cell positions)
        c[0] = 0.4f + 0.08f * std::sin(0.7f * static_cast<float>(x) + t);
        c[1] = 0.45f + 0.08f * std::cos(0.6f * static_cast<float>(y) - 0.5f * t);
        c[2] = 0.6f + 0.5f * std::sin(0.9f * static_cast<float>(x + y) + t);
        c[3] = 0.3f + 0.25f * std::cos(0.8f * static_cast<float>(x) - 0.4f * static_cast<float>(y) + t);
      }
    }
  }
  return r;
}

Model tiny_model(std::uint64_t seed = 3) {
  Model m = init_model(tiny_hyper(), seed);
  // Larger output weights than a fresh model, so the stepper's contribution is not negligible in the checks.
  std::mt19937_64 rng(seed + 1);
  std::normal_distribution<float> nd(0.f, 0.15f);
  for (float& w : m.step_w) w += nd(rng);
  const StepLayout L = step_layout(m.h);  // a modest output layer: velocity changes stay well below a cell
  for (std::size_t i = L.wo; i < L.bo + static_cast<std::size_t>(m.h.outputs()); ++i) m.step_w[i] *= 0.3f;
  m.scale = {0.4f, 0.4f, 0.5f, 0.3f};
  m.lo = {-5.f, -5.f, 0.f, 0.f};
  m.hi = {5.f, 5.f, 5.f, 5.f};
  m.effect = "tiny";
  m.control_names = {"intensity", "wind", "turbulence"};
  return m;
}

}  // namespace

TEST(Rollout, NoiseIsTheSimulationsForcing) {
  sim::Params p;
  p.seed = 4242;
  NoiseSpec n;
  for (const auto e : sim::kEffects) {
    p.effect = e;
    n.flicker_rate = e == sim::Effect::fire ? 2.6f : 1.4f;
    for (int i = 0; i < 50; ++i) {
      const float X = 2.7f * static_cast<float>(i), Y = 1.9f * static_cast<float>(i % 13), t = 0.13f * static_cast<float>(i);
      // equal up to rounding (X * (1 / 14) against X / 14)
      EXPECT_NEAR(noise_curl(n, p.seed, X, Y, t), sim::curl_potential(p, X, Y, t), 1e-5f);
      EXPECT_NEAR(noise_flicker(n, p.seed, X, Y, t), sim::source_flicker(p, X, Y, t), 1e-5f);
    }
  }
}

TEST(Rollout, CoarseFromSimAveragesBlocks) {
  sim::State st;
  st.n = 16;
  const std::size_t nn = 256;
  st.u.assign(nn, 60.f);  // 60 cells per second
  st.v.assign(nn, -30.f);
  st.temp.resize(nn);
  st.soot.resize(nn);
  for (std::size_t i = 0; i < nn; ++i) {
    st.temp[i] = static_cast<float>(i % 16);  // x
    st.soot[i] = static_cast<float>(i / 16);  // y
  }
  std::vector<float> c(4 * 4 * kPhys);
  coarse_from_sim(st, 4, 30.f, c);
  EXPECT_FLOAT_EQ(c[0], 60.f / 4.f / 30.f);   // coarse cells per frame
  EXPECT_FLOAT_EQ(c[1], -30.f / 4.f / 30.f);
  EXPECT_FLOAT_EQ(c[2], 1.5f);                // mean of x = 0..3
  EXPECT_FLOAT_EQ(c[(4 * 2 + 1) * kPhys + 2], 5.5f);  // block (1, 2): x = 4..7
  EXPECT_FLOAT_EQ(c[(4 * 2 + 1) * kPhys + 3], 9.5f);  // y = 8..11
}

TEST(Rollout, SaveLoadRoundTrips) {
  Model m = tiny_model();
  m.h.start_fine = 4;
  const auto r = tiny_run(m.h, 4);
  for (int k = 0; k < 2; ++k) {
    StartPoint sp;
    sp.controls = r.controls();
    sp.seed = 99 + static_cast<std::uint64_t>(k);
    sp.time = 1.5f;
    sp.coarse.assign(r.coarse.begin(), r.coarse.begin() + 8 * 8 * kPhys);
    if (k == 1) {
      sp.fine_t.assign(16, 0.25f);
      sp.fine_d.assign(16, 0.75f);
      sp.fine_t[3] = 1.f;
    }
    m.starts.push_back(sp);
  }
  quantise_like_storage(m);
  std::stringstream ss;
  ASSERT_TRUE(save_model(ss, m));
  const std::string bytes = ss.str();
  EXPECT_TRUE(is_rollout_file(std::span(bytes.data(), bytes.size())));
  EXPECT_NEAR(static_cast<double>(bytes.size()), static_cast<double>(m.storage_bytes()), 0.2 * static_cast<double>(bytes.size()));
  auto back = load_model(ss);
  ASSERT_TRUE(back) << back.error();
  EXPECT_EQ(back->h.res, m.h.res);
  EXPECT_EQ(back->control_names, m.control_names);
  EXPECT_EQ(back->step_w, m.step_w);
  EXPECT_EQ(back->render_w, m.render_w);
  ASSERT_EQ(back->starts.size(), 2u);
  EXPECT_EQ(back->starts[0].coarse, m.starts[0].coarse);
  EXPECT_TRUE(back->starts[0].fine_t.empty());
  EXPECT_EQ(back->starts[1].fine_t, m.starts[1].fine_t);
  EXPECT_EQ(back->starts[1].seed, 100u);
  EXPECT_EQ(back->detail.swirl_control, m.detail.swirl_control);
  std::stringstream bad("NVFXMDL1 not this kind");
  EXPECT_FALSE(load_model(bad));
}

TEST(Rollout, TrainerForwardMatchesTheReference) {
  const Model m = tiny_model();
  const auto r = tiny_run(m.h, 6);
  const int R = m.h.res, N = R * R, C = m.h.channels();
  // Reference: one coarse_step from state 2, scored against state 3.
  std::vector<float> in(static_cast<std::size_t>(N) * C, 0.f), out(in.size()), flow(static_cast<std::size_t>(N) * 2), p(static_cast<std::size_t>(N), 0.f);
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < kPhys; ++k) in[static_cast<std::size_t>(j) * C + k] = r.coarse[(2 * static_cast<std::size_t>(N) + j) * kPhys + k];
  }
  std::vector<float> cond(static_cast<std::size_t>(m.h.cond())), noise(static_cast<std::size_t>(N) * kNoise);
  condition(m, r.controls(), 3.f / r.p.fps, cond);
  coarse_noise(m, r.p.seed, 3.f / r.p.fps, noise);
  coarse_step(m, in, noise, cond, p, out, flow);
  double ref = 0;
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < kPhys; ++k) {
      const double e = out[static_cast<std::size_t>(j) * C + k] - r.coarse[(3 * static_cast<std::size_t>(N) + j) * kPhys + k];
      ref += e * e / (static_cast<double>(m.scale[k]) * m.scale[k]);
    }
  }
  ref /= N * kPhys;
  const double tr = window_loss(m, r, 2, 1, 0, 0.f, 0.f, 1, nullptr);
  EXPECT_NEAR(tr, ref, 1e-5 * ref + 1e-9);
}

namespace {

void rollout_gradient_check(int unroll, int burn, float profile) {
  Model m = tiny_model(11 + static_cast<std::uint64_t>(unroll + burn));
  const auto r = tiny_run(m.h, 12);
  std::vector<float> grad(m.step_w.size(), 0.f);
  const double l0 = window_loss(m, r, 1, unroll, burn, 0.f, profile, 5, &grad);
  ASSERT_GT(l0, 0.0);
  const auto eval = [&] { return window_loss(m, r, 1, unroll, burn, 0.f, profile, 5, nullptr); };
  // Central differences at three step sizes, and one-sided differences at the two smaller ones. A parameter passes
  // when one of them matches within 3%: the loss is summed in float, and ReLUs, clamps and bilinear interpolation make
  // it only piecewise smooth, so a kink just beside the current point spoils the central differences while the
  // difference on the far side of it stays exact. A wrong gradient matches none. Scattered estimates without a match
  // mark a kink and are skipped (at most 10%).
  const auto diffs = [&](float& slot) {
    const float keep = slot;
    const double base = eval();
    std::vector<double> out;
    for (const float eps : {1e-2f, 3e-3f, 1e-3f}) {
      slot = keep + eps;
      const double lp = eval();
      slot = keep - eps;
      const double lm = eval();
      slot = keep;
      out.push_back((lp - lm) / (2.0 * static_cast<double>(eps)));
      if (eps < 5e-3f) {
        out.push_back((lp - base) / static_cast<double>(eps));
        out.push_back((base - lm) / static_cast<double>(eps));
      }
    }
    return out;
  };
  const auto near = [](double a, double b) { return std::abs(a - b) <= 3e-2 * std::max(std::abs(a), std::abs(b)) + 2e-5; };
  int checked = 0, bad = 0, live = 0, kinks = 0;
  for (std::size_t i = 0; i < m.step_w.size(); i += std::max<std::size_t>(1, m.step_w.size() / 250)) {
    const std::vector<double> n = diffs(m.step_w[i]);
    const double ana = grad[i];
    if (std::ranges::any_of(n, [&](double v) { return near(v, ana); })) {
      ++checked;
      live += std::abs(n[0]) > 2e-5;
      continue;
    }
    const auto [lo, hi] = std::ranges::minmax(n);
    if (hi - lo > 0.1 * std::max(std::abs(lo), std::abs(hi)) + 1e-6) {
      ++kinks;
      continue;
    }
    ++checked;
    if (++bad <= 6) ADD_FAILURE() << std::format("param {}: analytic {} central {} {} {}", i, ana, n[0], n[1], n[4]);
  }
  EXPECT_GT(checked, 100);
  EXPECT_LE(kinks, (checked + kinks) / 10) << "too many parameters at kinks";
  EXPECT_GT(live, checked / 2) << "most gradients are zero: the check would be vacuous";
}

}  // namespace

TEST(Rollout, GradientsMatchFiniteDifferencesOneStep) { rollout_gradient_check(1, 0, 0.f); }
TEST(Rollout, GradientsMatchFiniteDifferencesThroughTime) { rollout_gradient_check(4, 0, 0.f); }
// The burn-in frames of a fine-tuning window are a stop-gradient by design (the window starts from wherever the
// stepper's own rollout went), so finite differences, which would re-run them, check the profile loss without them.
TEST(Rollout, GradientsMatchFiniteDifferencesWithProfiles) { rollout_gradient_check(3, 0, 1.f); }

TEST(Rollout, BurnInStartsFromTheStepperRollout) {
  const Model m = tiny_model();
  const auto r = tiny_run(m.h, 12);
  // Same window after burn-in, against a window from true states: different starts, so different losses, both finite.
  const double a = window_loss(m, r, 1, 2, 3, 0.f, 0.f, 5, nullptr), b = window_loss(m, r, 4, 2, 0, 0.f, 0.f, 5, nullptr);
  EXPECT_TRUE(std::isfinite(a) && std::isfinite(b));
  EXPECT_NE(a, b);
  std::vector<float> g(m.step_w.size(), 0.f);
  EXPECT_DOUBLE_EQ(window_loss(m, r, 1, 2, 3, 0.f, 0.f, 5, &g), a);
}

TEST(Rollout, ReferenceRolloutStaysFiniteAndRenders) {
  Model m = tiny_model();
  const auto r = tiny_run(m.h, 4);
  StartPoint sp;
  sp.controls = r.controls();
  sp.seed = 5;
  sp.time = 2.f;
  sp.coarse.assign(r.coarse.begin(), r.coarse.begin() + 8 * 8 * kPhys);
  m.starts.push_back(sp);
  m.render_scale = {1.f, 1.f};
  for (const int size : {16, 32}) {
    State s = start(m, 0, size, sp.controls, 77);
    EXPECT_NEAR(s.time, 2.f + static_cast<float>(m.h.warmup) / m.fps, 1e-5f);
    for (int f = 0; f < 20; ++f) step(m, s, sp.controls, 77);
    for (const float v : s.coarse) ASSERT_TRUE(std::isfinite(v));
    for (const float v : s.fine_t) ASSERT_TRUE(std::isfinite(v) && v >= 0.f);
    for (const float v : s.fine_d) ASSERT_TRUE(std::isfinite(v) && v >= 0.f);
    std::vector<float> rgba(static_cast<std::size_t>(size) * size * 4);
    render(m, s, rgba);
    for (const float v : rgba) ASSERT_TRUE(v >= 0.f && v <= 1.f);
  }
}

TEST(Rollout, DetailLayerCarriesFieldsWithTheFlow) {
  // No swirl, no contrast, a uniform flow of one coarse cell per frame to the right, a coarse state that matches the
  // fine field everywhere: a blob moves right by size / res pixels per frame and keeps its mass.
  Model m = tiny_model();
  m.detail.swirl = 0.f;
  m.detail.contrast = 0.f;
  const int R = m.h.res, S = 32, C = m.h.channels();
  State s;
  s.res = R;
  s.size = S;
  s.coarse.assign(static_cast<std::size_t>(R) * R * C, 0.f);
  s.flow.assign(static_cast<std::size_t>(R) * R * 2, 0.f);
  for (int i = 0; i < R * R; ++i) s.flow[static_cast<std::size_t>(i) * 2] = 0.25f;  // a quarter cell = 1 pixel
  s.fine_t.assign(static_cast<std::size_t>(S) * S, 0.f);
  s.fine_d.assign(s.fine_t.size(), 0.f);
  for (int y = 12; y < 20; ++y) {
    for (int x = 8; x < 16; ++x) s.fine_t[static_cast<std::size_t>(y) * S + x] = 1.f;
  }
  // coarse target: the block averages after the move
  std::vector<float> moved(s.fine_t.size(), 0.f);
  for (int y = 12; y < 20; ++y) {
    for (int x = 9; x < 17; ++x) moved[static_cast<std::size_t>(y) * S + x] = 1.f;
  }
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) s.coarse[(static_cast<std::size_t>(y / 4) * R + x / 4) * C + 2] += moved[static_cast<std::size_t>(y) * S + x] / 16.f;
  }
  detail_step(m, s, 1, {});
  double mass = 0, err = 0;
  for (std::size_t i = 0; i < moved.size(); ++i) {
    mass += s.fine_t[i];
    err += std::abs(s.fine_t[i] - moved[i]);
  }
  EXPECT_NEAR(mass, 64.0, 1.0);
  EXPECT_LT(err, 2.0);  // the edge columns are within a pixel's interpolation of exact
}

// --- the runtime (C API) on rollout effects ----------------------------------------------------------------------------

#include <neuralfx/nvfx.h>

namespace {

// A small rollout effect with two start points: one grown from its coarse state, one with fine fields.
Model runtime_model() {
  Hyper h;
  h.res = 16;
  h.hidden = 8;
  h.memory = 2;
  h.jacobi = 10;
  h.render_hidden = 6;
  h.start_fine = 16;
  h.warmup = 3;
  Model m = init_model(h, 21);
  std::mt19937_64 rng(22);
  std::normal_distribution<float> nd(0.f, 0.1f);
  for (float& w : m.step_w) w += nd(rng);
  for (float& w : m.render_w) w = 0.5f * w + 0.05f;
  m.effect = "test";
  m.control_names = {"intensity", "wind", "turbulence"};
  m.scale = {0.2f, 0.2f, 0.4f, 0.3f};
  m.lo = {-2.f, -2.f, 0.f, 0.f};
  m.hi = {2.f, 2.f, 3.f, 3.f};
  m.render_scale = {1.f, 1.f};
  m.detail.swirl_control = 2;
  m.detail.swirl = 0.6f;
  const int R = h.res;
  for (int k = 0; k < 2; ++k) {
    StartPoint sp;
    sp.controls = {0.3f + 0.4f * static_cast<float>(k), 0.5f, 0.6f};
    sp.seed = 1000 + static_cast<std::uint64_t>(k);
    sp.time = 2.f;
    sp.coarse.resize(static_cast<std::size_t>(R) * R * kPhys);
    for (int y = 0; y < R; ++y) {
      for (int x = 0; x < R; ++x) {
        float* c = sp.coarse.data() + (static_cast<std::size_t>(y) * R + x) * kPhys;
        c[0] = 0.1f * std::sin(0.5f * static_cast<float>(x + k));
        c[1] = 0.2f + 0.1f * std::cos(0.4f * static_cast<float>(y));
        c[2] = std::max(0.f, 0.8f - 0.05f * static_cast<float>(std::abs(x - 8) + y));
        c[3] = std::max(0.f, 0.5f - 0.04f * static_cast<float>(std::abs(x - 7) + std::abs(y - 6)));
      }
    }
    if (k == 1) {
      sp.fine_t.resize(256);
      sp.fine_d.resize(256);
      for (int i = 0; i < 256; ++i) {
        sp.fine_t[static_cast<std::size_t>(i)] = std::max(0.f, 0.9f - 0.03f * static_cast<float>(std::abs(i % 16 - 8) + i / 16));
        sp.fine_d[static_cast<std::size_t>(i)] = 0.3f + 0.2f * std::sin(0.7f * static_cast<float>(i));
      }
    }
    m.starts.push_back(sp);
  }
  quantise_like_storage(m);
  return m;
}

struct Fx {
  nvfx_effect* e = nullptr;
  explicit Fx(const Model& m) {
    std::ostringstream os;
    EXPECT_TRUE(save_model(os, m));
    const std::string b = os.str();
    EXPECT_EQ(nvfx_effect_load_memory(b.data(), b.size(), &e), NVFX_OK);
  }
  ~Fx() { nvfx_effect_free(e); }
  Fx(const Fx&) = delete;
  Fx& operator=(const Fx&) = delete;
};

std::vector<std::uint8_t> reference_frame(const Model& m, int start_index, int size, int frames) {
  State s = rollout::start(m, start_index, size, m.starts[static_cast<std::size_t>(start_index)].controls, m.starts[static_cast<std::size_t>(start_index)].seed);
  for (int f = 0; f < frames; ++f) rollout::step(m, s, m.starts[static_cast<std::size_t>(start_index)].controls, m.starts[static_cast<std::size_t>(start_index)].seed);
  std::vector<float> rgba(static_cast<std::size_t>(size) * size * 4);
  rollout::render(m, s, rgba);
  std::vector<std::uint8_t> out(rgba.size());
  for (std::size_t i = 0; i < rgba.size(); ++i) out[i] = static_cast<std::uint8_t>(rgba[i] * 255.f + 0.5f);
  return out;
}

}  // namespace

TEST(RolloutRuntime, InfoAndControls) {
  const Model m = runtime_model();
  Fx fx(m);
  nvfx_effect_info info{};
  ASSERT_EQ(nvfx_effect_get_info(fx.e, &info), NVFX_OK);
  EXPECT_EQ(info.arch, 3);
  EXPECT_EQ(info.n_variations, 2);
  EXPECT_EQ(info.n_controls, 3);
  EXPECT_STREQ(nvfx_effect_control_name(fx.e, 2), "turbulence");
  nvfx_instance* in = nullptr;
  EXPECT_EQ(nvfx_instance_create(fx.e, 24, &in), NVFX_ERROR_UNSUPPORTED);  // not a multiple of the coarse grid
  ASSERT_EQ(nvfx_instance_create(fx.e, 32, &in), NVFX_OK);
  EXPECT_GT(nvfx_instance_scratch_bytes(in), 0u);
  EXPECT_GT(nvfx_instance_macs_per_pixel(in), 0.0);
  EXPECT_EQ(nvfx_instance_set_variation(in, 2), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_instance_set_variation(in, 1), NVFX_OK);
  nvfx_instance_free(in);
}

TEST(RolloutRuntime, MatchesTheReferenceOnEveryIsa) {
  const Model m = runtime_model();
  Fx fx(m);
  for (const nvfx_isa isa : {NVFX_ISA_BASELINE, NVFX_ISA_AVX2, NVFX_ISA_AVX512}) {
    if (nvfx_set_isa(isa) != NVFX_OK) continue;
    for (const int start_index : {0, 1}) {
      for (const int size : {32, 64}) {
        nvfx_instance* in = nullptr;
        ASSERT_EQ(nvfx_instance_create(fx.e, size, &in), NVFX_OK);
        nvfx_instance_set_controls(in, m.starts[static_cast<std::size_t>(start_index)].controls.data(), 3);
        ASSERT_EQ(nvfx_instance_set_variation(in, start_index), NVFX_OK);
        std::vector<std::uint8_t> rt(static_cast<std::size_t>(size) * size * 4);
        for (const int frames : {0, 4}) {
          ASSERT_EQ(nvfx_render(in, frames / 30.0, rt.data(), static_cast<std::size_t>(size) * 4), NVFX_OK);
          const auto ref = reference_frame(m, start_index, size, frames);
          int worst = 0;
          std::size_t off = 0;
          for (std::size_t i = 0; i < ref.size(); ++i) {
            const int d = std::abs(int(ref[i]) - int(rt[i]));
            worst = std::max(worst, d);
            off += d > 1;
          }
          EXPECT_LE(worst, 3) << "isa " << isa << " start " << start_index << " size " << size << " frame " << frames;
          EXPECT_LE(off, ref.size() / 100) << "isa " << isa << " start " << start_index << " size " << size << " frame " << frames;
        }
        nvfx_instance_free(in);
      }
    }
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
}

TEST(RolloutRuntime, SeeksAreDeterministicWithinAShard) {
  const Model m = runtime_model();
  Fx fx(m);
  const int size = 32;
  std::vector<std::uint8_t> a(static_cast<std::size_t>(size) * size * 4), b(a.size());
  nvfx_instance *played = nullptr, *fresh = nullptr;
  ASSERT_EQ(nvfx_instance_create(fx.e, size, &played), NVFX_OK);
  ASSERT_EQ(nvfx_instance_create(fx.e, size, &fresh), NVFX_OK);
  for (nvfx_instance* in : {played, fresh}) nvfx_instance_set_seed(in, 4242);
  for (int f = 0; f <= 40; ++f) nvfx_render(played, f / 30.0, a.data(), size * 4);  // played frame by frame
  nvfx_render(fresh, 40 / 30.0, b.data(), size * 4);                                // one seek
  EXPECT_EQ(a, b);
  nvfx_render(played, 10 / 30.0, a.data(), size * 4);  // backwards: restarts from the start point
  nvfx_render(fresh, 10 / 30.0, b.data(), size * 4);
  EXPECT_EQ(a, b);
  nvfx_instance_set_seed(fresh, 7);  // another seed: another run
  nvfx_render(fresh, 0.0, b.data(), size * 4);
  nvfx_render(fresh, 30 / 30.0, b.data(), size * 4);
  nvfx_render(played, 30 / 30.0, a.data(), size * 4);
  EXPECT_NE(a, b);
  nvfx_instance_free(played);
  nvfx_instance_free(fresh);
}

TEST(RolloutRuntime, OneShotsHoldTheirLastFrameAndBakesLoop) {
  Model m = runtime_model();
  m.loop = false;
  m.h.frames = 12;
  m.h.n_age = 2;
  m.step_w = init_model(m.h, 5).step_w;  // the FiLM input grew by the age features
  {
    Fx fx(m);
    const int size = 32;
    nvfx_instance* in = nullptr;
    ASSERT_EQ(nvfx_instance_create(fx.e, size, &in), NVFX_OK);
    std::vector<std::uint8_t> a(static_cast<std::size_t>(size) * size * 4), b(a.size());
    nvfx_render(in, 11 / 30.0, a.data(), size * 4);
    nvfx_render(in, 5.0, b.data(), size * 4);
    EXPECT_EQ(a, b);
    nvfx_instance_free(in);
  }
  const Model loop = runtime_model();
  Fx fx(loop);
  const int size = 32, frames = 20;
  nvfx_instance* in = nullptr;
  ASSERT_EQ(nvfx_instance_create(fx.e, size, &in), NVFX_OK);
  std::vector<std::uint8_t> baked(static_cast<std::size_t>(size) * size * 4 * frames), one(static_cast<std::size_t>(size) * size * 4);
  ASSERT_EQ(nvfx_bake(in, frames, baked.data()), NVFX_OK);
  // frames after the crossfade are the plain rollout
  nvfx_render(in, 15 / 30.0, one.data(), size * 4);
  EXPECT_TRUE(std::equal(one.begin(), one.end(), baked.begin() + static_cast<std::ptrdiff_t>(one.size() * 15)));
  nvfx_instance_free(in);
}
