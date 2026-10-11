// Rollout effects: the noise matches the simulation's forcing, the format round-trips, the trainer's forward pass is
// the reference's, its gradients (through time, burn-in and the profile loss) match finite differences, and the
// reference rollout stays finite.
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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

TEST(Rollout, QuantisedStartStatesRoundTrip) {
  // Version 3 files: coarse start states at 2 to 8 bits per channel plane, with and without the seed's dither. What
  // loads equals quantise_like_storage; the error stays within a step (half a step undithered); empty heat stays
  // empty; files without quantised starts keep the version 2 layout.
  Model m = tiny_model();
  const auto r = tiny_run(m.h, 4);
  for (int k = 0; k < 2; ++k) {
    StartPoint sp;
    sp.controls = r.controls();
    sp.seed = 7 + static_cast<std::uint64_t>(k);
    sp.time = 1.f;
    sp.coarse.assign(r.coarse.begin() + k * 8 * 8 * kPhys, r.coarse.begin() + (k + 1) * 8 * 8 * kPhys);
    for (int i = 0; i < 8; ++i) sp.coarse[static_cast<std::size_t>(i) * kPhys + 2] = 0.f;  // some empty heat
    m.starts.push_back(sp);
  }
  {
    std::stringstream ss;
    ASSERT_TRUE(save_model(ss, m));
    EXPECT_EQ(ss.str()[8], 2);  // version 2 layout when start states are fp16
  }
  for (const int bits : {8, 6, 4, 3}) {
    for (const bool dither : {false, true}) {
      Model q = m;
      q.start_bits = bits;
      q.start_dither = dither;
      std::stringstream ss;
      ASSERT_TRUE(save_model(ss, q));
      const std::string bytes = ss.str();
      EXPECT_EQ(bytes[8], 3);
      EXPECT_NEAR(static_cast<double>(bytes.size()), static_cast<double>(q.storage_bytes()), 0.2 * static_cast<double>(bytes.size()));
      auto back = load_model(ss);
      ASSERT_TRUE(back) << back.error();
      EXPECT_EQ(back->start_bits, bits);
      EXPECT_EQ(back->start_dither, dither);
      Model ref = q;
      quantise_like_storage(ref);
      for (std::size_t k = 0; k < 2; ++k) {
        ASSERT_EQ(back->starts[k].coarse, ref.starts[k].coarse) << bits << " " << dither;
        for (int c = 0; c < kPhys; ++c) {
          float mn = 1e9f, mx = -1e9f;
          for (std::size_t i = static_cast<std::size_t>(c); i < m.starts[k].coarse.size(); i += kPhys) {
            mn = std::min(mn, m.starts[k].coarse[i]);
            mx = std::max(mx, m.starts[k].coarse[i]);
          }
          const float step = (mx - mn) / static_cast<float>((1 << bits) - 1);
          for (std::size_t i = static_cast<std::size_t>(c); i < m.starts[k].coarse.size(); i += kPhys) {
            EXPECT_LE(std::abs(back->starts[k].coarse[i] - m.starts[k].coarse[i]), (dither ? 1.01f : 0.51f) * step + 2e-3f * std::abs(mx) + 1e-6f);
          }
        }
        for (int i = 0; i < 8; ++i) EXPECT_EQ(back->starts[k].coarse[static_cast<std::size_t>(i) * kPhys + 2], 0.f);
      }
    }
  }
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

// Couplings on the tiny run (docs/COMPOSE.md §9): every operation before two steps in three, small enough that the
// backtraces stay inside their cells.
Run tiny_forced_run(const Hyper& h, int frames) {
  Run r = tiny_run(h, frames);
  const int N = h.res * h.res;
  r.forcing_at.assign(static_cast<std::size_t>(frames), -1);
  for (int i = 0; i < frames; ++i) {
    if (i % 3 == 0) continue;
    r.forcing_at[static_cast<std::size_t>(i)] = static_cast<int>(r.forcing.size() / (static_cast<std::size_t>(N) * kForce));
    for (int j = 0; j < N; ++j) {
      const float a = 0.3f * static_cast<float>(i) + 0.7f * static_cast<float>(j);
      r.forcing.insert(r.forcing.end(), {0.03f * std::sin(a), 0.025f * std::cos(a), 0.012f * std::sin(1.3f * a), -0.01f * std::cos(0.7f * a),
                                         0.9f + 0.08f * std::sin(a), 0.85f + 0.1f * std::cos(a), 0.03f * (1.f + std::sin(a)), 0.02f * (1.f + std::cos(a))});
    }
  }
  return r;
}

void rollout_gradient_check(int unroll, int burn, float profile, float activity = 0.f, bool forced = false, const Model* anchor = nullptr,
                            float anchor_weight = 0.f) {
  Model m = tiny_model(11 + static_cast<std::uint64_t>(unroll + burn));
  const auto r = forced ? tiny_forced_run(m.h, 12) : tiny_run(m.h, 12);
  std::vector<float> grad(m.step_w.size(), 0.f);
  const double l0 = window_loss(m, r, 1, unroll, burn, 0.f, profile, 5, &grad, activity, anchor, anchor_weight);
  ASSERT_GT(l0, 0.0);
  const auto eval = [&] { return window_loss(m, r, 1, unroll, burn, 0.f, profile, 5, nullptr, activity, anchor, anchor_weight); };
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
TEST(Rollout, GradientsMatchFiniteDifferencesWithActivity) { rollout_gradient_check(4, 0, 0.5f, 50.f); }
// Through the couplings: the push is added and taken out, forces and material added, v and material multiplied.
TEST(Rollout, GradientsMatchFiniteDifferencesWithCouplings) { rollout_gradient_check(4, 0, 0.5f, 50.f, true); }

// Round 2 (docs/COMPOSE.md §10): the anchor to another model's first step, alone and with couplings.
TEST(Rollout, GradientsMatchFiniteDifferencesWithTheAnchor) {
  const Model a = tiny_model(99);
  rollout_gradient_check(3, 0, 0.5f, 20.f, false, &a, 5.f);
}
TEST(Rollout, GradientsMatchFiniteDifferencesWithTheAnchorAndCouplings) {
  const Model a = tiny_model(98);
  rollout_gradient_check(2, 0, 0.f, 0.f, true, &a, 3.f);
}

TEST(Rollout, AnchorAddsTheFirstStepsDistanceFromATrueState) {
  const Model m = tiny_model(), other = tiny_model(42);
  const auto r = tiny_run(m.h, 12);
  const double plain = window_loss(m, r, 1, 3, 0, 0.f, 0.5f, 5, nullptr);
  EXPECT_EQ(window_loss(m, r, 1, 3, 0, 0.f, 0.5f, 5, nullptr, 0.f, &m, 7.f), plain);  // anchored to itself: no distance
  const double d1 = window_loss(m, r, 1, 3, 0, 0.f, 0.5f, 5, nullptr, 0.f, &other, 1.f) - plain;
  const double d4 = window_loss(m, r, 1, 3, 0, 0.f, 0.5f, 5, nullptr, 0.f, &other, 4.f) - plain;
  EXPECT_GT(d1, 0.0);
  EXPECT_NEAR(d4, 4.0 * d1, 1e-6 * std::abs(plain) + 1e-9);
  // after a burn-in the window does not start at a true state: no anchor
  EXPECT_EQ(window_loss(m, r, 1, 3, 2, 0.f, 0.5f, 5, nullptr, 0.f, &other, 4.f), window_loss(m, r, 1, 3, 2, 0.f, 0.5f, 5, nullptr));
}

TEST(Rollout, WindowsKnowWhereCouplingsAct) {
  const Hyper h = tiny_hyper();
  const auto forced = tiny_forced_run(h, 12), plain = tiny_run(h, 12);
  // forced_run leaves out every third state (0, 3, 6, 9): steps that produce states 3 and 4 include state 4's coupling
  EXPECT_TRUE(window_has_coupling(forced, 2, 2));
  EXPECT_FALSE(window_has_coupling(forced, 2, 1));
  EXPECT_FALSE(window_has_coupling(plain, 0, 10));
}

TEST(Rollout, SceneForcingIsDeterministicAndStrong) {
  int rings = 0, removes = 0;
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    for (const auto e : {sim::Effect::fire, sim::Effect::smoke}) {
      const ForcingSpec a = scene_forcing(e, 240, seed, 15, 180), b = scene_forcing(e, 240, seed, 15, 180);
      ASSERT_EQ(a.events.size(), b.events.size());
      ASSERT_GE(a.events.size(), 1u);
      const Coupling& gale = a.events[0];
      EXPECT_EQ(gale.kind, Coupling::Kind::push);
      EXPECT_EQ(gale.shape, Coupling::Shape::wave);
      EXPECT_GE(gale.onset, 15);
      EXPECT_LE(gale.onset, 25);
      EXPECT_LE(gale.amp, e == sim::Effect::fire ? 0.6f : 0.4f);
      for (std::size_t k = 0; k < a.events.size(); ++k) {
        const Coupling& c = a.events[k];
        EXPECT_EQ(c.onset, b.events[k].onset);
        EXPECT_EQ(c.amp, b.events[k].amp);
        EXPECT_LE(c.onset + c.duration, 240);
        if (c.shape == Coupling::Shape::vortex) {
          EXPECT_LE(std::abs(c.amp), 0.7f);
          ++rings;
        }
        removes += c.kind == Coupling::Kind::remove;
      }
    }
  }
  EXPECT_GT(rings, 300);  // two vortices per ring, in about 60% of 400 specs
  EXPECT_GT(removes, 50);
}

TEST(Rollout, CouplingsChangeTheWindowAndBurnInFollowsThem) {
  const Model m = tiny_model();
  const auto plain = tiny_run(m.h, 12), forced = tiny_forced_run(m.h, 12);
  EXPECT_NE(window_loss(m, plain, 1, 3, 0, 0.f, 0.f, 5, nullptr), window_loss(m, forced, 1, 3, 0, 0.f, 0.f, 5, nullptr));
  const double a = window_loss(m, plain, 1, 2, 3, 0.f, 0.f, 5, nullptr), b = window_loss(m, forced, 1, 2, 3, 0.f, 0.f, 5, nullptr);
  EXPECT_TRUE(std::isfinite(a) && std::isfinite(b));
  EXPECT_NE(a, b);
  // a forced run whose slots are all empty is the plain run
  rollout::Run none = forced;
  std::ranges::fill(none.forcing_at, -1);
  EXPECT_EQ(window_loss(m, plain, 1, 3, 2, 0.f, 0.5f, 5, nullptr, 20.f), window_loss(m, none, 1, 3, 2, 0.f, 0.5f, 5, nullptr, 20.f));
}

TEST(Rollout, ApplyForcingAndRemovePushMirrorCompose) {
  const int C = 6;
  std::vector<float> s = {0.5f, -0.2f, 0.8f, 0.3f, 0.1f, 0.2f};
  const std::vector<float> f = {0.25f, 0.125f, 0.5f, -0.25f, 0.5f, 0.75f, 0.125f, 0.0625f};
  apply_forcing(s, C, f);
  EXPECT_FLOAT_EQ(s[0], 0.5f + 0.5f + 0.25f);
  EXPECT_FLOAT_EQ(s[1], -0.2f * 0.5f - 0.25f + 0.125f);
  EXPECT_FLOAT_EQ(s[2], 0.8f * 0.75f + 0.125f);
  EXPECT_FLOAT_EQ(s[3], 0.3f * 0.75f + 0.0625f);
  EXPECT_EQ(s[4], 0.1f);  // memory channels are not touched
  remove_push(s, C, f);
  EXPECT_FLOAT_EQ(s[0], 0.5f + 0.5f);  // the force stays, the push goes
  EXPECT_FLOAT_EQ(s[1], -0.2f * 0.5f - 0.25f);
  std::vector<float> t = {0.f, 0.f, 0.1f, 0.1f};
  apply_forcing(t, 4, std::vector<float>{0, 0, 0, 0, 1, 1, -0.5f, 0});
  EXPECT_EQ(t[2], 0.f);  // heat stays at or above zero
}

TEST(Rollout, ForcedRunWithoutEventsIsThePlainRun) {
  sim::Params p;
  p.effect = sim::Effect::fire;
  p.size = 32;
  p.pressure_iters = 10;
  p.seed = 9;
  const rollout::Run a = record_run(p, 12, 8), b = record_forced_run(p, ForcingSpec{}, 12, 8);
  EXPECT_EQ(a.coarse, b.coarse);
  EXPECT_TRUE(std::ranges::all_of(b.forcing_at, [](int k) { return k < 0; }));
  // with couplings: deterministic, different from the plain run, the slots where the spec is active
  const ForcingSpec spec = random_forcing(sim::Effect::fire, 12, 77, 2, 6);
  const rollout::Run c = record_forced_run(p, spec, 12, 8), d = record_forced_run(p, spec, 12, 8);
  EXPECT_EQ(c.coarse, d.coarse);
  EXPECT_EQ(c.forcing, d.forcing);
  EXPECT_NE(c.coarse, a.coarse);
  for (int i = 0; i < 12; ++i) EXPECT_EQ(c.forcing_at[static_cast<std::size_t>(i)] >= 0, spec.active(i)) << i;
}

TEST(Rollout, CoarseForcingIsAveragedLikeTheState) {
  ForcingSpec spec;
  Coupling g;
  g.kind = Coupling::Kind::push;
  g.shape = Coupling::Shape::vortex;
  g.duration = 5;
  g.amp = 0.4f;
  g.radius = 0.2f;
  spec.events.push_back(g);
  Coupling a;
  a.kind = Coupling::Kind::add;
  a.duration = 5;
  a.amp = 0.05f;
  a.amp2 = 0.02f;
  spec.events.push_back(a);
  sim::Params p;
  SimForcing sf;
  forcing_fields(spec, p, 64, 2, sf);
  std::vector<float> cf(16 * 16 * kForce), cs(16 * 16 * kPhys);
  coarse_forcing(sf, 16, p.fps, cf);
  sim::State st;
  st.n = 64;
  st.u = sf.pu;
  st.v = sf.pv;
  st.temp = sf.ah;
  st.soot = sf.as;
  coarse_from_sim(st, 16, p.fps, cs);
  float peak = 0.f;
  for (int i = 0; i < 16 * 16; ++i) {
    for (int k = 0; k < 2; ++k) EXPECT_EQ(cf[static_cast<std::size_t>(i) * kForce + k], cs[static_cast<std::size_t>(i) * kPhys + k]);
    EXPECT_EQ(cf[static_cast<std::size_t>(i) * kForce + 6], cs[static_cast<std::size_t>(i) * kPhys + 2]);
    EXPECT_EQ(cf[static_cast<std::size_t>(i) * kForce + 7], cs[static_cast<std::size_t>(i) * kPhys + 3]);
    EXPECT_FLOAT_EQ(cf[static_cast<std::size_t>(i) * kForce + 4], 1.f);  // no ceiling
    peak = std::max(peak, std::hypot(cf[static_cast<std::size_t>(i) * kForce], cf[static_cast<std::size_t>(i) * kForce + 1]));
  }
  // the vortex's peak speed is `amp` cells of a 32-cell grid per frame: 0.2 cells of this 16-cell grid
  EXPECT_NEAR(peak, 0.4f * 16.f / 32.f, 0.03f);
}

TEST(Rollout, RandomForcingIsDeterministicAndCoversEveryKind) {
  std::array<int, 5> kinds{};
  for (std::uint64_t seed = 1; seed <= 300; ++seed) {
    const ForcingSpec a = random_forcing(sim::Effect::smoke, 240, seed, 15, 230), b = random_forcing(sim::Effect::smoke, 240, seed, 15, 230);
    ASSERT_EQ(a.events.size(), b.events.size());
    ASSERT_GE(a.events.size(), 1u);
    for (std::size_t k = 0; k < a.events.size(); ++k) {
      const Coupling& c = a.events[k];
      EXPECT_EQ(c.onset, b.events[k].onset);
      EXPECT_EQ(c.amp, b.events[k].amp);
      EXPECT_GE(c.onset, 15);
      EXPECT_LE(c.onset + c.duration, 240);
      ++kinds[static_cast<std::size_t>(c.kind)];
    }
  }
  for (const int n : kinds) EXPECT_GT(n, 30);
}

TEST(Rollout, HandoverRunStartsFromTheExplosionsState) {
  sim::Params ex, sm;
  ex.effect = sim::Effect::explosion;
  ex.size = 32;
  ex.pressure_iters = 10;
  sm = ex;
  sm.effect = sim::Effect::smoke;
  sm.seed = 3;
  const rollout::Run r = record_handover_run(ex, 6, sm, 5, 8);
  sim::Fluid f(ex);
  for (int i = 0; i < 6; ++i) f.step_frame();
  std::vector<float> c(8 * 8 * kPhys);
  coarse_from_sim(f.state(), 8, ex.fps, c);
  EXPECT_EQ(std::vector<float>(r.coarse.begin(), r.coarse.begin() + static_cast<std::ptrdiff_t>(c.size())), c);
  EXPECT_EQ(r.p.effect, sim::Effect::smoke);
  EXPECT_FLOAT_EQ(r.t0 + 1.f / ex.fps, 6.f / ex.fps);  // state 0 at the explosion's time
  // and then the smoke simulation continues from it
  sim::Params q = sm;
  q.sim_res = 32;
  sim::Fluid g(q);
  g.set_state(f.state());
  g.step_frame();
  coarse_from_sim(g.state(), 8, ex.fps, c);
  EXPECT_EQ(std::vector<float>(r.coarse.begin() + static_cast<std::ptrdiff_t>(c.size()), r.coarse.begin() + static_cast<std::ptrdiff_t>(2 * c.size())), c);
}

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

TEST(RolloutRuntime, ShardsMakeEveryFrameAFunctionOfTime) {
  const Model m = runtime_model();
  Fx fx(m);
  const int size = 32;
  std::vector<std::uint8_t> a(static_cast<std::size_t>(size) * size * 4), b(a.size()), c(a.size());
  nvfx_instance *played = nullptr, *fresh = nullptr;
  ASSERT_EQ(nvfx_instance_create(fx.e, size, &played), NVFX_OK);
  ASSERT_EQ(nvfx_instance_create(fx.e, size, &fresh), NVFX_OK);
  for (nvfx_instance* in : {played, fresh}) {
    nvfx_instance_set_seed(in, 99);
    nvfx_instance_set_drift(in, 2.f);  // 60-frame shards: frames 45-59 crossfade into the second shard
  }
  for (int f = 0; f <= 130; ++f) {
    nvfx_render(played, f / 30.0, a.data(), size * 4);
    if (f == 50 || f == 75 || f == 130) {
      nvfx_render(fresh, f / 30.0, b.data(), size * 4);  // a seek straight to the frame
      EXPECT_EQ(a, b) << "frame " << f;
    }
  }
  nvfx_render(fresh, 44 / 30.0, b.data(), size * 4);  // the last frame of the first shard alone
  nvfx_render(fresh, 52 / 30.0, c.data(), size * 4);  // inside the crossfade
  EXPECT_NE(b, c);
  nvfx_instance_free(played);
  nvfx_instance_free(fresh);
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
