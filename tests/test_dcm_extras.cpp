// Study G's extras (include/neuralfx/dcm/extras.hpp, docs/DCM.md §10): the renderer mixer (G4a), the shard critic (G5a)
// and the update mixer (G5b).
#include <neuralfx/dcm/extras.hpp>
#include <neuralfx/dcm/fine.hpp>
#include <neuralfx/dcm/search.hpp>
#include <neuralfx/rollout.hpp>
#include <neuralfx/sim.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

namespace ex = nfx::dcm::extras;
namespace rollout = nfx::rollout;
namespace sim = nfx::sim;

namespace {

std::size_t zs(int v) { return static_cast<std::size_t>(v); }

// A small rollout model (random stepper) and a state with material in it: 8 coarse cells, 32 pixels.
rollout::Model small_model() {
  rollout::Hyper h;
  h.res = 8;
  h.hidden = 8;
  h.memory = 2;
  h.jacobi = 10;
  rollout::Model m = rollout::init_model(h, 3);
  m.render_scale = {0.8f, 0.5f};
  m.lo = {-3.f, -3.f, 0.f, 0.f};
  m.hi = {3.f, 3.f, 4.f, 4.f};
  m.scale = {0.3f, 0.3f, 0.5f, 0.3f};
  return m;
}

rollout::State small_state(const rollout::Model& m, unsigned seed) {
  const int R = m.h.res, S = 32, C = m.h.channels();
  rollout::State s;
  s.res = R;
  s.size = S;
  s.time = 2.f;
  s.since_start = 0.3f;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(0.f, 1.f);
  s.coarse.assign(zs(R * R * C), 0.f);
  for (int i = 0; i < R * R; ++i) {
    const int x = i % R, y = i / R;
    const float blob = std::exp(-0.15f * static_cast<float>((x - 4) * (x - 4) + (y - 3) * (y - 3)));
    s.coarse[zs(i * C)] = 0.2f * (u(rng) - 0.5f);
    s.coarse[zs(i * C + 1)] = 0.2f * (u(rng) - 0.5f);
    s.coarse[zs(i * C + 2)] = blob * u(rng);
    s.coarse[zs(i * C + 3)] = blob > 0.2f ? 0.5f * blob * u(rng) : 0.f;
  }
  s.pressure.assign(zs(R * R), 0.f);
  s.flow.assign(zs(R * R * 2), 0.f);
  s.fine_t.assign(zs(S * S), 0.f);
  s.fine_d.assign(zs(S * S), 0.f);
  for (int i = 0; i < S * S; ++i) {
    const int x = i % S, y = i / S;
    if (x > 6 && x < 26 && y > 4 && y < 22) {
      s.fine_t[zs(i)] = u(rng);
      s.fine_d[zs(i)] = u(rng) < 0.5f ? 0.4f * u(rng) : 0.f;
    }
  }
  return s;
}

}  // namespace

// G5b: with the mixer at its start (weight 1 on the stepper's update), a mixed step is rollout::step to the bit, and so
// is a fixed blend with a = 0; a blend with a = 1 takes the solver's physical channels.
TEST(DcmExtras, UpdateMixerAtItsStartIsTheStepper) {
  const rollout::Model m = small_model();
  const std::vector<float> ctl{0.4f, 0.6f, 0.5f};
  for (const int which : {0, 1}) {
    rollout::State a = small_state(m, 5), b = a;
    ex::CoarseSolver solver(sim::Effect::smoke, ctl, 11, m.h.res, m.fps);
    ex::UpdateMixer mix = which == 0 ? ex::UpdateMixer() : ex::UpdateMixer::blend(0.0);
    if (which == 1) mix.freeze();
    for (int f = 0; f < 3; ++f) {
      rollout::step(m, a, ctl, 11);
      ex::mixed_step(m, b, ctl, 11, solver, mix);
    }
    EXPECT_EQ(a.coarse, b.coarse);
    EXPECT_EQ(a.flow, b.flow);
    EXPECT_EQ(a.fine_t, b.fine_t);
    EXPECT_EQ(a.fine_d, b.fine_d);
    EXPECT_EQ(a.time, b.time);
  }
  rollout::State a = small_state(m, 6), b = a;
  ex::CoarseSolver s1(sim::Effect::smoke, ctl, 11, m.h.res, m.fps), s2(sim::Effect::smoke, ctl, 11, m.h.res, m.fps);
  std::vector<float> want(zs(m.h.res * m.h.res * rollout::kPhys));
  s1.step(m, a, want);
  ex::UpdateMixer full = ex::UpdateMixer::blend(1.0);
  ex::mixed_step(m, b, ctl, 11, s2, full);
  const int C = m.h.channels();
  for (int i = 0; i < m.h.res * m.h.res; ++i) {
    for (int k = 0; k < rollout::kPhys; ++k) {
      const float w = std::clamp(want[zs(i * rollout::kPhys + k)], m.lo[zs(k)], m.hi[zs(k)]);
      EXPECT_NEAR(b.coarse[zs(i * C + k)], w, 1e-5f);
    }
  }
}

// G5b: the coarse solver steps a rollout state as the simulation steps itself on the coarse grid (from zero pressure).
TEST(DcmExtras, CoarseSolverIsTheCoarseSimulation) {
  const rollout::Model m = small_model();
  sim::Params p;
  p.effect = sim::Effect::fire;
  p.size = 32;
  p.sim_res = 8;
  p.seed = 4;
  sim::Fluid f(p);
  for (int i = 0; i < 12; ++i) f.step_frame();
  sim::State st = f.state();
  std::ranges::fill(st.pressure, 0.f);
  rollout::State s = small_state(m, 1);
  s.time = st.time;
  const int C = m.h.channels();
  for (int i = 0; i < 64; ++i) {
    s.coarse[zs(i * C)] = st.u[zs(i)] / p.fps;
    s.coarse[zs(i * C + 1)] = st.v[zs(i)] / p.fps;
    s.coarse[zs(i * C + 2)] = st.temp[zs(i)];
    s.coarse[zs(i * C + 3)] = st.soot[zs(i)];
  }
  sim::Fluid g(p);
  g.set_state(st);
  g.step_frame();
  const sim::State want = g.state();
  ex::CoarseSolver solver(sim::Effect::fire, std::vector<float>{p.intensity, p.wind, p.turbulence}, p.seed, 8, p.fps);
  std::vector<float> out(64 * rollout::kPhys);
  solver.step(m, s, out);
  for (int i = 0; i < 64; ++i) {
    EXPECT_NEAR(out[zs(i * 4)] * p.fps, want.u[zs(i)], 1e-3f * (1.f + std::abs(want.u[zs(i)])));
    EXPECT_NEAR(out[zs(i * 4 + 2)], want.temp[zs(i)], 1e-5f);
    EXPECT_NEAR(out[zs(i * 4 + 3)], want.soot[zs(i)], 1e-5f);
  }
}

// G4a: the simulator's renderer on a state's fine fields draws the simulation's own frame (it is the truth's renderer).
TEST(DcmExtras, SimRendererDrawsTheSimulationsFrame) {
  for (const auto e : sim::kEffects) {
    sim::Params p;
    p.effect = e;
    p.size = 32;
    p.seed = 9;
    sim::Fluid f(p);
    for (int i = 0; i < 20; ++i) f.step_frame();
    std::vector<std::uint8_t> want(32 * 32 * 4);
    f.render(want);
    const sim::State st = f.state();
    rollout::State s;
    s.size = 32;
    s.fine_t = st.temp;
    s.fine_d = st.soot;
    ex::SimRenderer r(e, 32);
    std::vector<float> got(want.size());
    r.render(s, got);
    for (std::size_t i = 0; i < want.size(); ++i) ASSERT_EQ(static_cast<int>(std::lround(got[i] * 255.f)), want[i]) << i;
  }
}

// G4a: a frozen renderer mixer at its start draws the learned renderer exactly; the field shader leaves empty pixels
// transparent and makes hot ones glow; a saved mixer loads to the same pictures.
TEST(DcmExtras, RenderMixerStartsAsTheLearnedRenderer) {
  const rollout::Model m = small_model();
  rollout::State s = small_state(m, 3);
  ex::SimRenderer sr(sim::Effect::fire, 32);
  ex::RenderPlanes planes;
  ex::render_experts(m, s, sr, ex::ShaderLook{}, &m, nullptr, planes);
  ex::RenderMixer mix(ex::RenderMixConfig{{ex::kLearned, ex::kSim, ex::kShader, ex::kOther1}});
  mix.freeze();
  std::vector<float> out(planes.p[0].size());
  ex::render_mixed(mix, m, s, planes, out);
  EXPECT_EQ(out, planes.p[ex::kLearned]);
  EXPECT_EQ(planes.p[ex::kOther1], planes.p[ex::kLearned]);
  const ex::RenderMixer back = ex::RenderMixer::load(mix.config(), mix.serialise());
  std::vector<float> out2(out.size());
  ex::render_mixed(back, m, s, planes, out2);
  EXPECT_EQ(out, out2);
  EXPECT_EQ(back.version(), mix.version());
  // the field shader: pixel (0, 0) is empty, pixel (16, 12) (y up) is hot
  const auto& sh = planes.p[ex::kShader];
  for (int c = 0; c < 4; ++c) EXPECT_EQ(sh[zs(((32 - 1) * 32) * 4 + c)], 0.f);
  s.fine_t[zs(12 * 32 + 16)] = 3.f;
  ex::field_shader(m, s, ex::ShaderLook{}, planes.p[ex::kShader]);
  EXPECT_GT(planes.p[ex::kShader][zs(((32 - 1 - 12) * 32 + 16) * 4)], 0.5f);
  // training moves it towards a target the simulator's renderer draws
  ex::RenderMixer learn(ex::RenderMixConfig{{ex::kLearned, ex::kSim}}, 0.2);
  std::array<float, ex::kRenderExperts> x{};
  double before = 0, after = 0;
  std::mt19937 rng(2);
  std::uniform_real_distribution<float> u(0.f, 1.f);
  for (int it = 0; it < 8000; ++it) {
    x[ex::kLearned] = u(rng);
    x[ex::kSim] = u(rng);
    const double p = learn.predict(x, it % 4, 1, 1, 1);
    if (it < 400) before += std::abs(p - x[ex::kSim]);
    if (it >= 7600) after += std::abs(p - x[ex::kSim]);
    learn.update(x[ex::kSim]);
  }
  EXPECT_LT(after, 0.25 * before);
}

// G5a: the critic tells shifted features apart on fresh windows, is deterministic, and loads from its text.
TEST(DcmExtras, ShardCriticSeparatesRealFromModelWindows) {
  std::mt19937_64 rng(4);
  std::normal_distribution<double> g(0.0, 1.0);
  std::uniform_real_distribution<float> u(0.f, 1.f);
  const auto draw = [&](int n) {
    std::vector<ex::CriticSample> v;
    for (int i = 0; i < n; ++i) {
      ex::CriticSample s;
      s.real = i % 2 == 0;
      s.controls = {u(rng), u(rng), u(rng)};
      for (auto& f : s.f) f = g(rng);
      if (s.real) s.f[2] += 1.0 + s.controls[0];  // real windows move more, more so at high intensity
      v.push_back(s);
    }
    return v;
  };
  const auto fit = draw(400), train = draw(400), test = draw(400);
  ex::ShardCritic a, b;
  a.fit(fit, train, 3, 7);
  b.fit(fit, train, 3, 7);
  EXPECT_EQ(a.version(), b.version());
  std::vector<float> score;
  std::vector<int> y;
  for (const auto& s : test) {
    score.push_back(static_cast<float>(a.p_real(s.f, s.controls)));
    y.push_back(s.real ? 1 : 0);
  }
  EXPECT_GT(nfx::dcm::roc_auc(score, y), 0.8);
  ex::ShardCritic c = ex::ShardCritic::load(a.serialise());
  EXPECT_EQ(c.version(), a.version());
  for (int i = 0; i < 20; ++i) EXPECT_DOUBLE_EQ(c.p_real(test[zs(i)].f, test[zs(i)].controls), a.p_real(test[zs(i)].f, test[zs(i)].controls));
  EXPECT_EQ(ex::ShardCritic::load(c.serialise()).serialise(), a.serialise());
}

// G5a: candidate 0 is the runtime's own choice; the others pick among the three nearest start points.
TEST(DcmExtras, ShardCandidates) {
  rollout::Model m = small_model();
  for (int k = 0; k < 6; ++k) {
    rollout::StartPoint sp;
    sp.controls = {0.2f * static_cast<float>(k), 0.5f, 0.5f};
    m.starts.push_back(sp);
  }
  const std::vector<float> ctl{0.45f, 0.5f, 0.5f};  // nearest: starts 2, 3, then 1
  for (std::int64_t shard = 0; shard < 4; ++shard) {
    const ex::Candidate c0 = ex::shard_candidate(m, ctl, 77, shard, 0);
    EXPECT_EQ(c0.start, ex::runtime_start(m, ctl, 77, shard));
    EXPECT_EQ(c0.seed, ex::runtime_shard_seed(77, shard));
    for (int j = 0; j < 8; ++j) {
      const ex::Candidate c = ex::shard_candidate(m, ctl, 77, shard, j);
      EXPECT_TRUE(c.start == 1 || c.start == 2 || c.start == 3);
      if (j > 0) {
        EXPECT_NE(c.seed, c0.seed);
      }
    }
  }
  EXPECT_EQ(ex::runtime_shard_seed(77, 0), 77u);
}

// G5a: the critic's features of a clip are finite, also for an empty one.
TEST(DcmExtras, CriticFeaturesAreFinite) {
  nfx::Clip c;
  c.allocate(32, 6);
  for (const double f : ex::critic_features(c)) EXPECT_TRUE(std::isfinite(f));
  for (int t = 0; t < 6; ++t) {
    auto fr = c.frame(t);
    for (std::size_t i = 0; i < fr.size(); ++i) fr[i] = static_cast<std::uint8_t>((i * 7 + static_cast<std::size_t>(t) * 13) % 200);
  }
  const auto f = ex::critic_features(c);
  for (const double v : f) EXPECT_TRUE(std::isfinite(v));
  EXPECT_GT(f[9], 0.0);
  EXPECT_LT(f[9], 1.0);
}
