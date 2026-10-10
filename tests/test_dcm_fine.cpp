// DCM-fine (include/neuralfx/dcm/fine.hpp, docs/DCM.md G1): the mixer in place of the rollout detail layer's lock.
#include <neuralfx/dcm/fine.hpp>
#include <neuralfx/dcm/search.hpp>
#include <neuralfx/rollout.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <random>
#include <vector>

namespace fine = nfx::dcm::fine;
namespace rollout = nfx::rollout;

namespace {

// A small rollout model (random stepper) and a state with material in it: 8 coarse cells, 32 pixels.
rollout::Model small_model() {
  rollout::Hyper h;
  h.res = 8;
  h.hidden = 8;
  h.memory = 2;
  h.jacobi = 10;
  rollout::Model m = rollout::init_model(h, 3);
  m.render_scale = {0.8f, 0.5f};
  m.detail.grow = 1.5f;
  m.detail.swirl_control = 2;
  return m;
}

rollout::State small_state(const rollout::Model& m, unsigned seed) {
  const int R = m.h.res, S = 32, C = m.h.channels();
  rollout::State s;
  s.res = R;
  s.size = S;
  s.time = 2.f;
  s.since_start = 0.3f;  // the swirl still ramping in
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(0.f, 1.f);
  s.coarse.assign(static_cast<std::size_t>(R * R * C), 0.f);
  for (int i = 0; i < R * R; ++i) {
    const int x = i % R, y = i / R;
    const float blob = std::exp(-0.15f * static_cast<float>((x - 4) * (x - 4) + (y - 3) * (y - 3)));
    s.coarse[static_cast<std::size_t>(i * C)] = 0.2f * (u(rng) - 0.5f);
    s.coarse[static_cast<std::size_t>(i * C + 1)] = 0.2f * (u(rng) - 0.5f);
    s.coarse[static_cast<std::size_t>(i * C + 2)] = blob * u(rng);
    s.coarse[static_cast<std::size_t>(i * C + 3)] = blob > 0.2f ? 0.5f * blob * u(rng) : 0.f;
  }
  s.pressure.assign(static_cast<std::size_t>(R * R), 0.f);
  s.flow.assign(static_cast<std::size_t>(R * R * 2), 0.f);
  for (auto& v : s.flow) v = 0.3f * (u(rng) - 0.5f);
  s.fine_t.assign(static_cast<std::size_t>(S * S), 0.f);
  s.fine_d.assign(static_cast<std::size_t>(S * S), 0.f);
  for (int i = 0; i < S * S; ++i) {
    const int x = i % S, y = i / S;
    if (x > 6 && x < 26 && y > 4 && y < 22) {
      s.fine_t[static_cast<std::size_t>(i)] = u(rng);
      s.fine_d[static_cast<std::size_t>(i)] = u(rng) < 0.5f ? 0.4f * u(rng) : 0.f;
    }
  }
  return s;
}

fine::Spec small_spec(const rollout::Model& m) {
  fine::Spec sp;
  sp.size = 32;
  sp.s = {m.render_scale[0], m.render_scale[1]};
  sp.eps = {0.02f * sp.s[0], 0.02f * sp.s[1]};
  for (int q = 0; q < 2; ++q) {
    sp.lvl[static_cast<std::size_t>(q)] = {0.01f, 0.05f, 0.2f, 0.5f};
    sp.ratio[static_cast<std::size_t>(q)] = {0.5f, 1.f, 1.5f};
  }
  sp.macro4 = std::vector<double>(16, 0.0);
  for (int k = 0; k < 4; ++k) sp.macro4[static_cast<std::size_t>(k * 4 + k)] = 1.0;
  return sp;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  return d;
}

}  // namespace

TEST(DcmFine, DcmFineWithTheV1WeightsIsV1) {
  const rollout::Model m = small_model();
  const std::vector<float> controls{0.6f, 0.4f, 0.7f};
  for (const auto domain : {fine::Domain::linear, fine::Domain::log}) {
    fine::Spec sp = small_spec(m);
    sp.domain = domain;
    // with and without context-selected mixers: every weight set starts as the rule
    for (const std::vector<int>& ctx : {std::vector<int>{}, std::vector<int>{fine::kLvl, fine::kFlow, fine::kMacro4}}) {
      const fine::Mixer mix = fine::v1_mixer(sp, ctx);
      // one detail step from the same state
      rollout::State a = small_state(m, 5), b = a;
      rollout::detail_step(m, a, 77, controls);
      fine::Frame f;
      fine::detail_step(m, mix, b, 77, controls, {}, f);
      const double tol = domain == fine::Domain::linear ? 1e-6 : 2e-6;
      EXPECT_LE(max_diff(a.fine_t, b.fine_t), tol);
      EXPECT_LE(max_diff(a.fine_d, b.fine_d), tol);
      // and frame.L is the reference's output exactly
      rollout::State c = small_state(m, 5);
      fine::compute_frame(m, c, 77, controls, f);
      EXPECT_EQ(f.L[0], a.fine_t);
      EXPECT_EQ(f.L[1], a.fine_d);
      // whole frames (stepper, then detail), several in a row
      rollout::State x = small_state(m, 9), y = x;
      for (int k = 0; k < 5; ++k) {
        rollout::step(m, x, controls, 123);
        fine::step(m, mix, y, controls, 123, {}, f);
      }
      EXPECT_EQ(x.coarse, y.coarse);
      EXPECT_LE(max_diff(x.fine_t, y.fine_t), 5.0 * tol);
      EXPECT_LE(max_diff(x.fine_d, y.fine_d), 5.0 * tol);
    }
  }
}

TEST(DcmFine, RenderIsTheReferenceRender) {
  rollout::Model m = small_model();
  m.h.res = 8;
  rollout::State s = small_state(m, 3);
  std::vector<float> a(32 * 32 * 4), b(a.size());
  rollout::render(m, s, a);
  fine::render(m, s, b);
  EXPECT_EQ(a, b);
}

TEST(DcmFine, ExpertsAndContextsOfAFrame) {
  const rollout::Model m = small_model();
  const fine::Spec sp = small_spec(m);
  const std::vector<float> controls{0.9f, 0.5f, 0.1f};
  rollout::State s = small_state(m, 4);
  fine::Frame f;
  fine::compute_frame(m, s, 11, controls, f);
  int active = 0;
  for (int y = 0; y < 32; ++y) {
    for (int x = 0; x < 32; ++x) {
      for (int q = 0; q < 2; ++q) {
        const fine::Row r = fine::row_of(f, s, x, y, q);
        EXPECT_FLOAT_EQ(r.e[fine::kAdvDiff], r.e[fine::kAdv] - r.e[fine::kAdvSl]);
        EXPECT_GE(r.e[fine::kAdv], 0.f);
        const auto c = fine::contexts_of(sp, r, controls, 32);
        for (int k = 0; k < fine::kContexts; ++k) {
          EXPECT_GE(c[static_cast<std::size_t>(k)], 0);
          EXPECT_LT(c[static_cast<std::size_t>(k)], fine::context_size(sp, k));
        }
        EXPECT_EQ(c[fine::kChannel], q);
        EXPECT_EQ(c[fine::kCtrl], 2 * 3 + 0);
        EXPECT_EQ(c[fine::kAge], 0);  // a looping effect
        if (!fine::skip_pixel(sp, f, static_cast<std::size_t>(y * 32 + x), q)) ++active;
      }
    }
  }
  EXPECT_GT(active, 200);
}

TEST(DcmFine, ValueBitsAreLaplaceBitsInTheLinearDomain) {
  const rollout::Model m = small_model();
  fine::Spec sp = small_spec(m);
  for (const double c : {0.0, 0.05, 0.6}) {
    for (const double v : {0.0, 0.01, 0.3, 1.2}) {
      const double y = fine::normalise_value(sp, 0, v, c), w = fine::bin_width(sp, 0, v, c);
      EXPECT_NEAR(w, sp.s[0] / 256.0 / (c + sp.eps[0]), 1e-12);
      for (const double mu : {y - 0.3, y, y + 2.0}) {
        EXPECT_NEAR(fine::value_bits(sp, 0, v, mu, 0.2, c), nfx::dcm::laplace_bits(y, mu, 0.2, w), 1e-9);
      }
      EXPECT_NEAR(fine::value_of(sp, 0, y, c), v, 1e-12);
    }
  }
  sp.domain = fine::Domain::log;
  for (const double v : {0.0, 0.01, 0.3, 1.2}) EXPECT_NEAR(fine::value_of(sp, 1, fine::normalise_value(sp, 1, v, 0.1), 0.1), v, 1e-12);
  // a sharper prediction at the truth codes it in fewer bits
  EXPECT_LT(fine::value_bits(sp, 0, 0.3, fine::normalise_value(sp, 0, 0.3, 0.0), 0.01, 0.0),
            fine::value_bits(sp, 0, 0.3, fine::normalise_value(sp, 0, 0.3, 0.0), 0.1, 0.0));
}

TEST(DcmFine, GrainIsLaplaceAndRepeats) {
  const rollout::Model m = small_model();
  const fine::Spec sp = small_spec(m);
  double mean = 0.0, absmean = 0.0;
  const int n = 4000;
  for (int i = 0; i < n; ++i) {
    const float X = static_cast<float>(i % 64) * 1.7f, Y = static_cast<float>(i / 64) * 1.9f;
    const float g = fine::grain(sp, 5, X, Y, 0.4f);
    EXPECT_EQ(g, fine::grain(sp, 5, X, Y, 0.4f));
    mean += g / n;
    absmean += std::abs(g) / n;
  }
  EXPECT_NEAR(mean, 0.0, 0.25);
  EXPECT_NEAR(absmean, 1.0, 0.25);  // Laplace(0, 1): E|x| = 1
  EXPECT_NE(fine::grain(sp, 5, 3.f, 4.f, 0.4f), fine::grain(sp, 6, 3.f, 4.f, 0.4f));
}

TEST(DcmFine, MixerFileRoundTrip) {
  const rollout::Model m = small_model();
  fine::Spec sp = small_spec(m);
  sp.domain = fine::Domain::log;
  sp.one_shot = true;
  const fine::Mixer a = fine::v1_mixer(sp, std::vector<int>{fine::kHeight, fine::kAge});
  const auto path = std::filesystem::temp_directory_path() / "nvfx_dcm_fine_test.mixer";
  fine::save_mixer(path, a);
  const fine::Mixer b = fine::load_mixer(path);
  std::filesystem::remove(path);
  EXPECT_EQ(a.version(), b.version());
  EXPECT_EQ(b.config.mixer_contexts, (std::vector<int>{fine::kHeight, fine::kAge}));
  EXPECT_EQ(b.spec.domain, fine::Domain::log);
  EXPECT_TRUE(b.spec.one_shot);
  EXPECT_EQ(b.spec.lvl, a.spec.lvl);
  EXPECT_EQ(b.spec.macro4, a.spec.macro4);
  std::vector<double> x(fine::kExperts, 0.25), z(fine::kScaleFeatures, 0.1);
  x[fine::kLock] = 0.7;
  std::vector<int> c(fine::kContexts, 1);
  EXPECT_EQ(a.predict(x, c, z).mu, 0.7);
  EXPECT_EQ(b.predict(x, c, z).mu, 0.7);
}

// Determinism: 100 frames of a mixer that is not the v1 rule, with grain and each relock, give one hash.
TEST(DcmFine, HundredStepsGiveOneHash) {
  const rollout::Model m = small_model();
  const fine::Spec sp = small_spec(m);
  fine::Config c;
  c.experts = {fine::kAdv, fine::kAdvSl, fine::kLock, fine::kPrev, fine::kBias};
  c.mixer_contexts = {fine::kLvl};
  c.spec.context_sizes = {1, fine::context_size(sp, fine::kLvl)};
  c.spec.loss = nfx::dcm::ValueLoss::laplace;
  nfx::dcm::ValueNet net(5, fine::kScaleFeatures, c.spec);
  net.set_weights(std::vector<double>{0.5, 0.3, 0.3, -0.1, 0.01});
  net.freeze();
  const fine::Mixer mix = fine::make_mixer(sp, c, net);
  const std::vector<float> controls{0.4f, 0.6f, 0.5f};
  for (const int relock : {0, 1, 2}) {
    std::string first;
    for (int run = 0; run < 2; ++run) {
      rollout::State s = small_state(m, 21);
      fine::Frame f;
      for (int k = 0; k < 100; ++k) fine::step(m, mix, s, controls, 99, {0.7, relock, {}}, f);
      std::string bytes(reinterpret_cast<const char*>(s.fine_t.data()), s.fine_t.size() * sizeof(float));
      bytes.append(reinterpret_cast<const char*>(s.fine_d.data()), s.fine_d.size() * sizeof(float));
      const std::string h = nfx::dcm::sha256_hex(bytes);
      if (run == 0) first = h;
      EXPECT_EQ(h, first);
    }
  }
}

// The search's value_rule: a configuration that includes the rule's column starts training as the rule.
TEST(DcmSearch, ValueRuleStartsTrainingAtTheRule) {
  namespace dcm = nfx::dcm;
  dcm::SearchProblem p;
  p.input_names = {"rule", "other", "bias"};
  p.groups = {{"rule", {0}}, {"other", {1, 2}}};
  p.context_names = {"c"};
  p.context_sizes = {2};
  p.site_names = {"a", "b"};
  p.holdout = {true, true};
  p.delta = 0.01;
  std::mt19937 rng(2);
  std::normal_distribution<double> nd;
  for (int i = 0; i < 400; ++i) {
    const double r = nd(rng);
    p.x.push_back({r, nd(rng), 1.0});
    p.contexts.push_back({i % 2});
    p.z.push_back({std::abs(r)});
    p.y.push_back(r);  // the rule is exact
    p.site.push_back(i % 2);
    p.fold.push_back(-1);
  }
  dcm::SearchSpace space;
  dcm::SearchConfig c;
  c.groups = {0, 1};
  c.mixer_contexts = {0};
  c.loss = 1;  // laplace
  c.epochs = 1;
  std::vector<std::size_t> rows(p.y.size());
  for (std::size_t i = 0; i < rows.size(); ++i) rows[i] = i;
  const auto predict = [&](dcm::ValueNet& net, std::size_t i) {
    return net.predict(p.x[i], std::vector<int>{0, p.contexts[i][0], 0, 0, 0}, p.z[i]).mu;
  };
  p.value_rule = 0;
  auto net = dcm::train_value_config(p, space, c, rows, 1);
  for (std::size_t i = 0; i < 50; ++i) EXPECT_EQ(predict(net, i), p.y[i]);  // no error, so nothing moved
  p.value_rule = -1;
  auto avg = dcm::train_value_config(p, space, c, rows, 1);
  double err = 0.0;
  for (std::size_t i = 0; i < 50; ++i) err += std::abs(predict(avg, i) - p.y[i]);
  EXPECT_GT(err, 1e-3);
  p.value_rule = 3;
  EXPECT_THROW(dcm::validate(p, dcm::Objective::laplace_bits), std::invalid_argument);
}
