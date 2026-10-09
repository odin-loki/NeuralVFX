// Diffusion-context mixing (docs/DCM.md): the PAQ mixer, its compact copy, the mixer search and the k-means contexts.
// The mixer, search and k-means tests are ported from the owner's CameraDetector (cabinlab/src/diffusion/tests:
// test_diffusion.cpp, test_optimise.cpp, test_search.cpp, test_kmeans.cpp) with their data and thresholds; the
// value-domain tests are new.
#include <neuralfx/dcm/compact.hpp>
#include <neuralfx/dcm/kmeans.hpp>
#include <neuralfx/dcm/mixer.hpp>
#include <neuralfx/dcm/search.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace dcm = nfx::dcm;

namespace {

bool has(const std::vector<int>& v, int x) { return std::ranges::find(v, x) != v.end(); }

}  // namespace

// --- the PAQ mixer --------------------------------------------------------------------------------------------------

TEST(DcmMixer, LearnsToTrustTheInformativeInput) {
  std::mt19937 rng(3);
  std::normal_distribution<double> nd;
  dcm::MixerNetSpec spec;
  spec.context_sizes = {1, 2};
  spec.final_contexts = 1;
  spec.apm_contexts = 1;
  dcm::MixerNet net(2, spec);
  int correct = 0;
  for (int i = 0; i < 4000; ++i) {
    const int y = static_cast<int>(rng() % 2);
    const double good = (y ? 1.5 : -1.5) + nd(rng);
    const double noise = nd(rng) * 2.0;
    const std::vector<double> x{good, noise};
    const std::vector<int> ctx{0, i % 2, 0, 0};
    const double p = net.predict(x, ctx);
    if (i >= 3000 && ((p > 0.5) == (y == 1))) ++correct;
    net.update(y);
  }
  EXPECT_GT(correct, 850);  // about 93% achievable
  const auto w = net.mixer(0).weights(0);
  EXPECT_GT(std::abs(w[0]), 3.0 * std::abs(w[1]));
}

TEST(DcmMixer, FrozenNetDoesNotChangeAndVersionIsStable) {
  dcm::MixerNetSpec spec;
  spec.context_sizes = {2};
  dcm::MixerNet net(3, spec);
  const std::vector<double> x{0.5, -1.0, 2.0};
  const std::vector<int> ctx{1, 0, 0};
  (void)net.predict(x, ctx);
  net.update(1);
  net.freeze();
  const auto v = net.version();
  for (int i = 0; i < 10; ++i) {
    (void)net.predict(x, ctx);
    net.update(i % 2);
  }
  EXPECT_EQ(net.version(), v);
  EXPECT_EQ(v.size(), 64u);
  EXPECT_TRUE(net.serialise().starts_with("nvfx-paq-mixer v1\n"));
}

TEST(DcmMixer, ApmStartsAsIdentityAndSha256Vectors) {
  dcm::APM apm(1, 0.02);
  for (const double p : {0.1, 0.5, 0.9}) EXPECT_NEAR(apm.refine(p, 0), p, 0.03);
  EXPECT_EQ(dcm::sha256_hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(dcm::sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  // FIPS 180-4's two-block example (448 bits: the padding needs a second block)
  EXPECT_EQ(dcm::sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  EXPECT_NEAR(dcm::squash(dcm::stretch(0.3)), 0.3, 1e-12);
}

namespace {

// test_optimise.cpp's trained_net: five first-layer mixers (MixerNet's fused first layer: a block of four and one).
dcm::MixerNet trained_net(double apm_weight, unsigned seed, std::vector<std::vector<double>>& xs,
                          std::vector<std::vector<int>>& cs) {
  dcm::MixerNetSpec s;
  s.context_sizes = {1, 3, 4, 2, 5};
  s.apm_contexts = 3;
  s.apm_weight = apm_weight;
  dcm::MixerNet net(7, s);
  std::mt19937 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  xs.clear();
  cs.clear();
  for (int i = 0; i < 600; ++i) {
    std::vector<double> x(7);
    for (auto& v : x) v = g(rng);
    std::vector<int> c{0, i % 3, i % 4, i % 2, i % 5, 0, i % 3};
    const int y = x[0] + 0.5 * x[1] * (c[1] == 1 ? -1 : 1) + 0.3 * g(rng) > 0 ? 1 : 0;
    (void)net.predict(x, c);
    net.update(y);
    xs.push_back(x);
    cs.push_back(c);
  }
  net.freeze();
  return net;
}

}  // namespace

TEST(DcmMixer, CompactMixerPredictsAsTheReleasedMixer) {
  for (const double apm : {0.0, 0.4}) {
    std::vector<std::vector<double>> xs;
    std::vector<std::vector<int>> cs;
    auto net = trained_net(apm, 2, xs, cs);
    const auto text = net.serialise();
    const dcm::CompactMixer<double> d(text);
    const dcm::CompactMixer<float> f(text);
    EXPECT_EQ(d.version(), net.version());
    EXPECT_EQ(d.inputs(), 7);
    EXPECT_EQ(d.first_layer(), 5);
    const std::size_t values = 7u * (1 + 3 + 4 + 2 + 5) + 5u + (apm > 0.0 ? 3u * 33u : 0u);
    EXPECT_EQ(d.values(), values);
    EXPECT_EQ(d.bytes(), values * 8);
    EXPECT_EQ(f.bytes(), values * 4);
    const auto fb = dcm::CompactMixer<float>::from_bytes(f.to_bytes());
    EXPECT_EQ(fb.version(), net.version());
    double worst_d = 0.0, worst_f = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
      const double p = net.predict(xs[i], cs[i]);
      worst_d = std::max(worst_d, std::abs(d.predict(xs[i], cs[i]) - p));
      worst_f = std::max(worst_f, std::abs(f.predict(xs[i], cs[i]) - p));
      EXPECT_EQ(fb.predict(xs[i], cs[i]), f.predict(xs[i], cs[i]));
    }
    EXPECT_LT(worst_d, 1e-8);
    EXPECT_LT(worst_f, 1e-5);
    EXPECT_THROW((void)d.predict(xs[0], std::vector<int>{0, 0}), std::invalid_argument);
    EXPECT_THROW((void)d.predict(xs[0], std::vector<int>{0, 9, 0, 0, 0, 0, 0}), std::out_of_range);
  }
  EXPECT_THROW(dcm::CompactMixer<float>("not a mixer"), std::runtime_error);
}

// --- the value domain -----------------------------------------------------------------------------------------------

TEST(DcmValue, LaplaceBitsAreTheBinsProbabilityMass) {
  // a narrow bin: about −log2(delta · density); a bin holding the centre; the far tail stays finite
  const double b = 0.5, delta = 1e-3;
  EXPECT_NEAR(dcm::laplace_bits(0.7, 0.2, b, delta), -std::log2(delta * std::exp(-0.5 / b) / (2.0 * b)), 1e-6);
  EXPECT_NEAR(dcm::laplace_bits(-0.3, 0.2, b, delta), dcm::laplace_bits(0.7, 0.2, b, delta), 1e-9);  // symmetric
  EXPECT_NEAR(dcm::laplace_bits(0.0, 0.0, 1.0, 2.0), -std::log2(1.0 - std::exp(-1.0)), 1e-12);
  EXPECT_LT(dcm::laplace_bits(0.0, 0.0, 1e-3, 1.0), 1e-9);  // nearly all the mass: nearly 0 bits
  EXPECT_TRUE(std::isfinite(dcm::laplace_bits(1e6, 0.0, 1e-3, 1e-3)));
  EXPECT_THROW((void)dcm::laplace_bits(0.0, 0.0, 0.0, 1.0), std::invalid_argument);
}

TEST(DcmValue, ValueNetLearnsALinearTarget) {
  // y = 2 a − b + 0.5 + noise; inputs (a, b, 1), contexts that carry nothing
  std::mt19937 rng(4);
  std::normal_distribution<double> nd;
  dcm::ValueNetSpec spec;
  spec.context_sizes = {1, 3};
  spec.lr1 = 0.1;
  spec.lr2 = 0.05;
  dcm::ValueNet net(3, 0, spec);
  double sq = 0.0, var = 0.0;
  for (int i = 0; i < 6000; ++i) {
    const double a = nd(rng), b = nd(rng);
    const double y = 2.0 * a - b + 0.5 + 0.1 * nd(rng);
    const std::vector<double> x{a, b, 1.0};
    const std::vector<int> ctx{0, i % 3, 0, 0, 0};
    const auto pr = net.predict(x, ctx, {});
    if (i >= 5000) {
      sq += (y - pr.mu) * (y - pr.mu);
      var += y * y;
    }
    net.update(y);
  }
  EXPECT_LT(sq / 1000.0, 0.05) << "mean squared error over the last 1000 rows";  // noise alone: 0.01; y: about 5.3
  EXPECT_GT(var / 1000.0, 4.0);
  // the scale net followed the residuals (Laplace scale of 0.1 Gaussian noise: about 0.08)
  EXPECT_LT(net.last().b, 0.3);
}

TEST(DcmValue, ValueNetLearnsTheLaplaceScale) {
  // y = a + Laplace noise of scale 0.2 (z = 0) or 1.0 (z = 1); the net starts as the rule "y = a" and learns the
  // scale through the scale feature z alone (one scale context)
  std::mt19937 rng(5);
  std::normal_distribution<double> nd;
  std::exponential_distribution<double> ex(1.0);
  std::bernoulli_distribution coin(0.5);
  dcm::ValueNetSpec spec;
  spec.context_sizes = {1};
  spec.loss = dcm::ValueLoss::laplace;
  spec.lr1 = spec.lr2 = 0.005;
  dcm::ValueNet net(2, 1, spec);
  net.set_rule(0);
  double learned_bits = 0.0, fixed_bits = 0.0;
  for (int i = 0; i < 8000; ++i) {
    const double a = nd(rng);
    const double s = coin(rng) ? 1.0 : 0.0;
    const double scale = s > 0.0 ? 1.0 : 0.2;
    const double y = a + scale * ex(rng) * (coin(rng) ? 1.0 : -1.0);
    const std::vector<double> x{a, 1.0};
    const std::vector<double> z{s};
    (void)net.predict(x, std::vector<int>{0, 0, 0, 0}, z);
    if (i >= 6000) {
      learned_bits += net.bits(y, 0.01);
      fixed_bits += dcm::laplace_bits(y, net.last().mu, 0.6, 0.01);  // the best single scale for both: mean |e|
    }
    net.update(y);
  }
  net.freeze();
  for (const double s : {0.0, 1.0}) {
    const std::vector<double> z{s};
    const auto pr = net.predict(std::vector<double>{0.0, 1.0}, std::vector<int>{0, 0, 0, 0}, z);
    const double truth = s > 0.0 ? 1.0 : 0.2;
    EXPECT_NEAR(pr.b, truth, 0.25 * truth) << "z = " << s;
  }
  EXPECT_LT(learned_bits, fixed_bits - 0.2 * 2000) << "at least 0.2 bits per row better than one fixed scale";
}

TEST(DcmValue, ValueNetStartsAsTheRuleItCopies) {
  dcm::ValueNetSpec spec;
  spec.context_sizes = {1, 3, 2};
  spec.avm_contexts = 2;
  spec.avm_weight = 0.3;
  spec.loss = dcm::ValueLoss::laplace;
  dcm::ValueNet net(4, 2, spec);
  net.set_rule(2);
  std::mt19937 rng(6);
  std::normal_distribution<double> nd;
  for (int i = 0; i < 200; ++i) {
    const std::vector<double> x{nd(rng) * 3.0, nd(rng), nd(rng) * 10.0, nd(rng)};
    const std::vector<int> ctx{0, i % 3, i % 2, 0, i % 2, 0};
    const std::vector<double> z{nd(rng), nd(rng)};
    EXPECT_EQ(net.predict(x, ctx, z).mu, x[2]);  // exactly the rule, whatever the contexts and the AVM
  }
  // a frozen copy stays the rule; a learning one moves away from it once it sees a target
  const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
  const std::vector<int> ctx{0, 1, 0, 0, 0, 0};
  const std::vector<double> z{0.0, 0.0};
  auto frozen = net;
  frozen.freeze();
  (void)frozen.predict(x, ctx, z);
  frozen.update(10.0);
  EXPECT_EQ(frozen.predict(x, ctx, z).mu, 3.0);
  (void)net.predict(x, ctx, z);
  net.update(10.0);
  EXPECT_GT(net.predict(x, ctx, z).mu, 3.0);
  // general weights: the net predicts w · x
  dcm::ValueNet lin(3, 0, spec);
  lin.set_weights(std::vector<double>{0.5, -2.0, 1.0});
  EXPECT_NEAR(lin.predict(std::vector<double>{2.0, 1.0, 4.0}, std::vector<int>{0, 2, 1, 0, 0, 0}, {}).mu, 3.0, 1e-12);
  EXPECT_THROW(lin.set_rule(3), std::out_of_range);
}

namespace {

// A trained ValueNet with three first-layer mixers, the AVM and a scale net with two features and two contexts.
dcm::ValueNet trained_value_net(std::vector<std::vector<double>>& xs, std::vector<std::vector<int>>& cs,
                                std::vector<std::vector<double>>& zs) {
  dcm::ValueNetSpec s;
  s.context_sizes = {1, 3, 2};
  s.avm_contexts = 2;
  s.avm_weight = 0.4;
  s.avm_lo = -6.0;
  s.avm_hi = 6.0;
  s.avm_rate = 0.05;
  s.scale_contexts = 2;
  s.loss = dcm::ValueLoss::laplace;
  dcm::ValueNet net(4, 2, s);
  std::mt19937 rng(7);
  std::normal_distribution<double> g(0.0, 1.0);
  xs.clear();
  cs.clear();
  zs.clear();
  for (int i = 0; i < 1500; ++i) {
    std::vector<double> x(4);
    for (auto& v : x) v = g(rng);
    std::vector<int> c{0, i % 3, i % 2, 0, (i / 2) % 2, i % 2};
    std::vector<double> z{g(rng), std::abs(g(rng))};
    const double y = 1.5 * x[0] * (c[2] == 1 ? -1.0 : 1.0) + 0.5 * x[1] + (0.1 + 0.3 * z[1]) * g(rng);
    (void)net.predict(x, c, z);
    net.update(y);
    xs.push_back(x);
    cs.push_back(c);
    zs.push_back(z);
  }
  net.freeze();
  return net;
}

}  // namespace

TEST(DcmCompact, CompactValueNetMatchesTrained) {
  std::vector<std::vector<double>> xs, zs;
  std::vector<std::vector<int>> cs;
  auto net = trained_value_net(xs, cs, zs);
  const auto text = net.serialise();
  ASSERT_TRUE(text.starts_with("nvfx-value-mixer v1\n"));
  const dcm::CompactValueNet<double> d(text);
  const dcm::CompactValueNet<float> f(text);
  EXPECT_EQ(d.version(), net.version());
  EXPECT_EQ(f.version(), net.version());
  EXPECT_EQ(d.inputs(), 4);
  EXPECT_EQ(d.scale_features(), 2);
  EXPECT_EQ(d.first_layer(), 3);
  const std::size_t values = 4u * (1 + 3 + 2) + 3u + 2u * 33u + 3u * 2u;  // mixers, final, AVM, scale net
  EXPECT_EQ(d.values(), values);
  EXPECT_EQ(d.bytes(), values * 8);
  EXPECT_EQ(f.bytes(), values * 4);
  double worst_d = 0.0, worst_f = 0.0, worst_b = 0.0;
  std::vector<dcm::ValuePrediction> want(xs.size());
  for (std::size_t i = 0; i < xs.size(); ++i) {
    const auto p = net.predict(xs[i], cs[i], zs[i]);
    const auto pd = d.predict(xs[i], cs[i], zs[i]);
    const auto pf = f.predict(xs[i], cs[i], zs[i]);
    worst_d = std::max({worst_d, std::abs(pd.mu - p.mu), std::abs(pd.b - p.b)});
    worst_f = std::max(worst_f, std::abs(pf.mu - p.mu));
    worst_b = std::max(worst_b, std::abs(pf.b - p.b) / p.b);
    want[i] = pf;
  }
  EXPECT_LE(worst_d, 1e-6);
  EXPECT_LE(worst_f, 1e-5);  // float weights and sums (about 7e-7 here)
  EXPECT_LE(worst_b, 1e-5);
  // the flat binary form: the same predictions bit for bit, and the version carried along
  for (const bool use_float : {false, true}) {
    const auto bytes = use_float ? f.to_bytes() : d.to_bytes();
    if (use_float) {
      const auto r = dcm::CompactValueNet<float>::from_bytes(bytes);
      EXPECT_EQ(r.version(), net.version());
      EXPECT_EQ(r.to_bytes(), bytes);
      for (std::size_t i = 0; i < xs.size(); ++i) {
        const auto p = r.predict(xs[i], cs[i], zs[i]);
        EXPECT_EQ(p.mu, want[i].mu);
        EXPECT_EQ(p.b, want[i].b);
      }
    } else {
      const auto r = dcm::CompactValueNet<double>::from_bytes(bytes);
      EXPECT_EQ(r.version(), net.version());
      EXPECT_EQ(r.to_bytes(), bytes);
      for (std::size_t i = 0; i < xs.size(); i += 7) EXPECT_EQ(r.predict(xs[i], cs[i], zs[i]).mu, d.predict(xs[i], cs[i], zs[i]).mu);
      EXPECT_THROW((void)dcm::CompactValueNet<float>::from_bytes(bytes), std::runtime_error);  // 8-byte weights
      EXPECT_THROW((void)dcm::CompactMixer<double>::from_bytes(bytes), std::runtime_error);    // another kind
      auto cut = bytes;
      cut.pop_back();
      EXPECT_THROW((void)dcm::CompactValueNet<double>::from_bytes(cut), std::runtime_error);
      cut = bytes;
      cut.push_back(0);
      EXPECT_THROW((void)dcm::CompactValueNet<double>::from_bytes(cut), std::runtime_error);
    }
  }
  // one instance serves several threads (no learning state)
  std::vector<std::vector<double>> got(4, std::vector<double>(xs.size()));
  {
    std::vector<std::jthread> pool;
    for (std::size_t t = 0; t < got.size(); ++t) {
      pool.emplace_back([&, t] {
        for (std::size_t i = 0; i < xs.size(); ++i) got[t][i] = f.predict(xs[i], cs[i], zs[i]).mu;
      });
    }
  }
  for (const auto& g : got) {
    for (std::size_t i = 0; i < xs.size(); ++i) EXPECT_EQ(g[i], want[i].mu);
  }
  EXPECT_THROW((void)d.predict(xs[0], cs[0], std::vector<double>{1.0}), std::invalid_argument);
  EXPECT_THROW((void)d.predict(xs[0], std::vector<int>{0, 5, 0, 0, 0, 0}, zs[0]), std::out_of_range);
  EXPECT_THROW(dcm::CompactValueNet<double>(net.serialise().substr(1)), std::runtime_error);
  EXPECT_THROW(dcm::CompactValueNet<double>(trained_net(0.0, 1, xs, cs).serialise()), std::runtime_error);
}

// --- the mixer search -----------------------------------------------------------------------------------------------

namespace {

// Synthetic search problem: a training pool (site 0, folds 0-2) and four test sites of `per` rows each. Input
// groups: "good" (column 0, informative), "noise" (columns 1-2) and "junk" (column 3), pure noise. Context columns:
// "shuffle" (3 values, random), "flip" (2 values: the sign of the good input's evidence flips with it, so the good
// input helps only through a mixer selected by "flip") and "spare" (4 values, random). max_mixer_contexts is set to 1
// by the tests, so the default mixer uses "shuffle", not "flip".
dcm::SearchProblem make_problem(int per, unsigned seed) {
  dcm::SearchProblem p;
  p.input_names = {"good", "noise_a", "noise_b", "junk"};
  p.groups = {{"good", {0}}, {"noise", {1, 2}}, {"junk", {3}}};
  p.context_names = {"shuffle", "flip", "spare"};
  p.context_sizes = {3, 2, 4};
  p.site_names = {"pool", "cam_a", "cam_b", "cam_c", "cam_d"};
  p.holdout = {false, true, true, true, true};
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd;
  std::bernoulli_distribution viol(0.4), half(0.5);
  for (int s = 0; s < 5; ++s) {
    const int n = s == 0 ? 3 * per : per;
    for (int i = 0; i < n; ++i) {
      const int y = viol(rng) ? 1 : 0;
      const int flip = half(rng) ? 1 : 0;
      const double evidence = (y ? 1.2 : -1.2) * (flip ? -1.0 : 1.0);
      p.x.push_back({evidence + nd(rng), nd(rng), nd(rng), nd(rng)});
      p.contexts.push_back({static_cast<int>(rng() % 3), flip, static_cast<int>(rng() % 4)});
      p.y.push_back(y);
      p.site.push_back(s);
      p.fold.push_back(s == 0 ? i % 3 : -1);
    }
  }
  return p;
}

dcm::SearchOptions small_options() {
  dcm::SearchOptions o;
  o.configs = 16;
  o.refine_rounds = 2;
  o.threads = 2;
  o.seed = 5;
  o.space.max_mixer_contexts = 1;
  return o;
}

}  // namespace

TEST(DcmSearch, FindsTheInformativeGroupAndContext) {
  const auto p = make_problem(160, 1);
  const auto o = small_options();
  const auto def = dcm::default_config(p, o.space);
  ASSERT_EQ(def.mixer_contexts, std::vector<int>{0});  // the default mixer does not use "flip"
  const auto r = dcm::run_search(p, o);

  ASSERT_FALSE(r.global_top.empty());
  const auto& g = r.global_top.front();
  SCOPED_TRACE(dcm::describe(p, o.space, g.config));
  EXPECT_TRUE(has(g.config.groups, 0));
  EXPECT_TRUE(has(g.config.mixer_contexts, 1));
  EXPECT_GT(g.score, 0.85);
  for (std::size_t i = 1; i < r.global_top.size(); ++i) EXPECT_GE(r.global_top[i - 1].score, r.global_top[i].score);
  EXPECT_EQ(r.global_version.size(), 64u);
  EXPECT_EQ(dcm::sha256_hex(r.global_mixer), r.global_version);

  ASSERT_EQ(r.nested_choices.size(), 4u);
  for (const auto& c : r.nested_choices) {
    SCOPED_TRACE(dcm::describe(p, o.space, c.config));
    EXPECT_TRUE(has(c.config.groups, 0));
    EXPECT_TRUE(has(c.config.mixer_contexts, 1));
    EXPECT_EQ(c.inner_sites, 3);
  }
  EXPECT_TRUE(has(r.rfonly_best.config.groups, 0));
  EXPECT_TRUE(has(r.rfonly_best.config.mixer_contexts, 1));

  // every protocol scores every test row and no training-pool row, and the scores separate the classes
  for (const auto* scores : {&r.nested, &r.global, &r.rfonly}) {
    ASSERT_EQ(scores->size(), p.y.size());
    for (std::size_t i = 0; i < p.y.size(); ++i) EXPECT_EQ(std::isnan((*scores)[i]), p.site[i] == 0);
    const auto auc = dcm::per_site_auc(p, *scores);
    EXPECT_TRUE(std::isnan(auc[0]));
    for (int s = 1; s < 5; ++s) EXPECT_GT(auc[static_cast<std::size_t>(s)], 0.8);
  }
  EXPECT_TRUE(r.nested_b.empty());  // value objectives only
}

TEST(DcmSearch, NestedNeverTrainsOnTheSiteItScores) {
  const auto p = make_problem(100, 2);
  auto o = small_options();
  o.configs = 10;
  o.refine_rounds = 1;
  o.global = false;
  o.rfonly = false;
  std::vector<dcm::TrainRecord> records;
  o.on_train = [&](const dcm::TrainRecord& rec) { records.push_back(rec); };
  const auto r = dcm::run_search(p, o);
  ASSERT_EQ(static_cast<std::int64_t>(records.size()), r.trainings);
  ASSERT_GT(records.size(), 10u * 6u);  // at least every pair of test sites for every random configuration
  std::set<int> scored_alone;
  for (const auto& rec : records) {
    EXPECT_EQ(rec.stage, "nested");
    EXPECT_TRUE(has(rec.trained_sites, 0));  // the training pool is always trained on
    ASSERT_FALSE(rec.scored_sites.empty());
    for (const int s : rec.scored_sites) {
      EXPECT_FALSE(has(rec.trained_sites, s)) << "site " << s << " scored by a model trained on it";
      EXPECT_TRUE(has(rec.excluded, s));
    }
    if (rec.excluded.size() == 1) scored_alone.insert(rec.scored_sites.front());
  }
  EXPECT_EQ(scored_alone, (std::set<int>{1, 2, 3, 4}));  // one final model per held-out site

  // black-box check of the same property: flipping a test site's labels cannot change that site's nested scores
  auto q = p;
  for (std::size_t i = 0; i < q.y.size(); ++i) {
    if (q.site[i] == 2) q.y[i] = 1 - q.y[i];
  }
  o.on_train = nullptr;
  const auto r2 = dcm::run_search(q, o);
  for (std::size_t i = 0; i < p.y.size(); ++i) {
    if (p.site[i] == 2) {
      EXPECT_EQ(r.nested[i], r2.nested[i]);
    }
  }
  EXPECT_EQ(dcm::describe(p, o.space, r.nested_choices[1].config), dcm::describe(q, o.space, r2.nested_choices[1].config));
  EXPECT_EQ(r.nested_choices[1].inner_auc, r2.nested_choices[1].inner_auc);
}

TEST(DcmSearch, NestedTeachersNeverSeeTheirSitesLabels) {
  const auto p = make_problem(80, 9);
  auto o = small_options();
  o.configs = 8;
  o.refine_rounds = 1;
  o.global = false;
  o.rfonly = false;
  const auto plain = dcm::run_search(p, o);
  EXPECT_TRUE(plain.nested_teachers.empty());
  o.nested_teachers = true;
  const auto r = dcm::run_search(p, o);
  ASSERT_EQ(r.nested.size(), plain.nested.size());
  for (std::size_t i = 0; i < r.nested.size(); ++i) {  // the teachers do not change the nested scores (NaN: unscored)
    if (std::isnan(plain.nested[i])) {
      EXPECT_TRUE(std::isnan(r.nested[i]));
    } else {
      EXPECT_EQ(r.nested[i], plain.nested[i]);
    }
  }
  ASSERT_EQ(r.nested_teachers.size(), r.nested_choices.size());
  for (std::size_t c = 0; c < r.nested_teachers.size(); ++c) {
    const auto& t = r.nested_teachers[c];
    ASSERT_EQ(t.size(), p.y.size());
    for (std::size_t i = 0; i < p.y.size(); ++i) {
      ASSERT_TRUE(std::isfinite(t[i]));  // every row, the training pool included
      if (p.site[i] == r.nested_choices[c].site) {
        EXPECT_EQ(t[i], r.nested[i]);  // its own site: the nested score
      }
    }
  }
  // flipping site 2's labels cannot change the teacher of site 2 on any row
  auto q = p;
  for (std::size_t i = 0; i < q.y.size(); ++i) {
    if (q.site[i] == 2) q.y[i] = 1 - q.y[i];
  }
  const auto r2 = dcm::run_search(q, o);
  ASSERT_EQ(r.nested_choices[1].site, 2);
  EXPECT_EQ(r.nested_teachers[1], r2.nested_teachers[1]);
  EXPECT_NE(r.nested_teachers[0], r2.nested_teachers[0]);  // site 1's teacher did train on site 2
}

TEST(DcmSearch, RfOnlyUsesOnlyTheTrainingPoolsLabels) {
  const auto p = make_problem(100, 3);
  auto o = small_options();
  o.configs = 10;
  o.nested = false;
  o.global = false;
  std::vector<dcm::TrainRecord> records;
  o.on_train = [&](const dcm::TrainRecord& rec) { records.push_back(rec); };
  const auto r = dcm::run_search(p, o);
  ASSERT_FALSE(records.empty());
  for (const auto& rec : records) {
    EXPECT_EQ(rec.stage, "rfonly");
    EXPECT_EQ(rec.trained_sites, std::vector<int>{0});
    if (rec.excluded_fold >= 0) {
      EXPECT_EQ(rec.scored_sites, std::vector<int>{0});
    }
  }
  EXPECT_EQ(records.back().scored_sites, (std::vector<int>{1, 2, 3, 4}));

  auto q = p;  // every test site's labels flipped: nothing of the rfonly protocol may change
  for (std::size_t i = 0; i < q.y.size(); ++i) {
    if (q.site[i] != 0) q.y[i] = 1 - q.y[i];
  }
  o.on_train = nullptr;
  const auto r2 = dcm::run_search(q, o);
  EXPECT_EQ(r.rfonly_version, r2.rfonly_version);
  for (std::size_t i = 0; i < p.y.size(); ++i) {
    if (p.site[i] != 0) {
      EXPECT_EQ(r.rfonly[i], r2.rfonly[i]);
    }
  }
}

TEST(DcmSearch, NeighboursConfigsAndTraining) {
  const auto p = make_problem(30, 4);
  dcm::SearchSpace space;
  space.max_mixer_contexts = 2;
  const auto def = dcm::default_config(p, space);
  EXPECT_EQ(def.groups, (std::vector<int>{0, 1, 2}));
  EXPECT_EQ(def.mixer_contexts, (std::vector<int>{0, 1}));
  EXPECT_EQ(dcm::describe(p, space, def),
            "groups good+noise+junk | mixers -,shuffle,flip | apm shuffle 0.3 | lr 0.02/0.01 | ep 4");

  const auto spec = dcm::mixer_spec(p, space, def);
  EXPECT_EQ(spec.context_sizes, (std::vector<int>{1, 3, 2}));
  EXPECT_EQ(spec.apm_contexts, 3);

  const auto nb = dcm::neighbours(p, space, def);
  std::set<std::string> seen;
  int group_flips = 0;
  for (const auto& c : nb) {
    EXPECT_TRUE(seen.insert(dcm::describe(p, space, c)).second);  // distinct
    EXPECT_NE(dcm::describe(p, space, c), dcm::describe(p, space, def));
    EXPECT_FALSE(c.groups.empty());
    EXPECT_LE(static_cast<int>(c.mixer_contexts.size()), space.max_mixer_contexts);
    EXPECT_TRUE(std::ranges::is_sorted(c.mixer_contexts));
    if (c.groups.size() == 2) ++group_flips;
  }
  // 3 group flips, 2 context drops (adding a third exceeds the cap), 2 swaps to "spare", 3 other APM contexts,
  // lr1 +/- 1, lr2 +/- 1, epochs +/- 1, APM weight +/- 1
  EXPECT_EQ(group_flips, 3);
  EXPECT_EQ(nb.size(), 3u + 2u + 2u + 3u + 8u);
  // under a value objective the scale context (4 others) and the loss (1 other) are neighbours too
  const auto vdef = dcm::default_config(p, space, dcm::Objective::laplace_bits);
  EXPECT_EQ(dcm::describe(p, space, vdef, dcm::Objective::laplace_bits),
            "groups good+noise+junk | mixers -,shuffle,flip | avm shuffle 0.3 | scale - | loss laplace | lr 0.02/0.01 | ep 4");
  EXPECT_EQ(dcm::neighbours(p, space, vdef, dcm::Objective::laplace_bits).size(), 3u + 2u + 2u + 3u + 3u + 8u + 1u);

  // a single-group configuration never loses its last group; with the APM off its context is canonical (-1)
  dcm::SearchConfig one = def;
  one.groups = {1};
  one.apm_weight = 0;
  for (const auto& c : dcm::neighbours(p, space, one)) {
    EXPECT_FALSE(c.groups.empty());
    if (space.apm_weight[static_cast<std::size_t>(c.apm_weight)] == 0.0) {
      EXPECT_EQ(c.apm_context, -1);
    }
  }

  std::vector<std::size_t> rows;
  for (std::size_t i = 0; i < p.y.size(); ++i) rows.push_back(i);
  const auto a = dcm::train_config(p, space, def, rows, 1);
  const auto b = dcm::train_config(p, space, def, rows, 1);
  EXPECT_TRUE(a.frozen());
  EXPECT_EQ(a.version(), b.version());
  EXPECT_EQ(a.first_layer(), 3);
  EXPECT_NE(a.version(), dcm::train_config(p, space, def, rows, 2).version());
}

TEST(DcmSearch, TrainsSiteBySite) {
  // The global prediction of a held-out site is the default configuration trained on the rows of the other sites in
  // site order (each in row order) with the search seed.
  const auto p = make_problem(60, 7);
  auto o = small_options();
  o.configs = 1;
  o.refine_rounds = 0;
  o.top = 1;
  o.nested = false;
  o.rfonly = false;
  const auto r = dcm::run_search(p, o);
  const auto def = dcm::default_config(p, o.space);
  ASSERT_EQ(dcm::describe(p, o.space, r.global_top.front().config), dcm::describe(p, o.space, def));
  std::vector<std::size_t> rows;
  for (const int s : {0, 2, 3, 4}) {
    for (std::size_t i = 0; i < p.y.size(); ++i) {
      if (p.site[i] == s) rows.push_back(i);
    }
  }
  auto net = dcm::train_config(p, o.space, def, rows, o.seed);
  int checked = 0;
  for (std::size_t i = 0; i < p.y.size(); ++i) {
    if (p.site[i] != 1) continue;
    const std::vector<int> ctx{0, p.contexts[i][0], 0, p.contexts[i][0]};  // mixers -, shuffle; final; APM shuffle
    EXPECT_EQ(r.global[i], static_cast<double>(static_cast<float>(dcm::stretch(net.predict(p.x[i], ctx)))));
    ++checked;
  }
  EXPECT_EQ(checked, 60);
}

TEST(DcmSearch, ResultsDoNotDependOnThreads) {
  const auto p = make_problem(80, 8);
  auto o = small_options();
  o.configs = 8;
  o.refine_rounds = 1;
  o.threads = 1;
  const auto a = dcm::run_search(p, o);
  o.threads = 3;
  const auto b = dcm::run_search(p, o);
  for (std::size_t i = 0; i < p.y.size(); ++i) {
    if (p.site[i] == 0) continue;
    EXPECT_EQ(a.nested[i], b.nested[i]);
    EXPECT_EQ(a.global[i], b.global[i]);
    EXPECT_EQ(a.rfonly[i], b.rfonly[i]);
  }
  ASSERT_EQ(a.nested_choices.size(), b.nested_choices.size());
  for (std::size_t k = 0; k < a.nested_choices.size(); ++k) {
    EXPECT_EQ(dcm::describe(p, o.space, a.nested_choices[k].config), dcm::describe(p, o.space, b.nested_choices[k].config));
    EXPECT_EQ(a.nested_choices[k].inner_auc, b.nested_choices[k].inner_auc);
  }
  EXPECT_EQ(a.global_version, b.global_version);
  EXPECT_EQ(a.rfonly_version, b.rfonly_version);
  EXPECT_EQ(a.trainings, b.trainings);
}

TEST(DcmSearch, StratifiedAucComparesPairsWithinStrata) {
  // one test site, two strata: perfectly ordered within each, but stratum 1 sits far below stratum 0 and holds more
  // positives, so pooled pairs across strata disagree
  auto p = make_problem(10, 6);
  std::vector<double> scores(p.y.size(), std::nan(""));
  p.stratum.assign(p.y.size(), 0);
  std::vector<std::size_t> rows;
  for (std::size_t i = 0; i < p.y.size(); ++i) {
    if (p.site[i] == 1) rows.push_back(i);
  }
  for (std::size_t k = 0; k < rows.size(); ++k) {
    const auto i = rows[k];
    const int stratum = k % 2 == 0 ? 0 : 1;
    p.stratum[i] = stratum;
    p.y[i] = k % 4 < 2 ? 1 : (stratum == 1 && k % 8 == 3 ? 1 : 0);
    scores[i] = (stratum == 0 ? 10.0 : 0.0) + (p.y[i] > 0.0 ? 1.0 : 0.0) + 0.01 * static_cast<double>(k);
  }
  EXPECT_DOUBLE_EQ(dcm::per_site_auc(p, scores)[1], 1.0);
  auto plain = p;
  plain.stratum.clear();
  EXPECT_LT(dcm::per_site_auc(plain, scores)[1], 0.95);
  p.stratum.pop_back();
  EXPECT_THROW(dcm::validate(p), std::invalid_argument);
}

TEST(DcmSearch, RejectsInconsistentProblems) {
  const auto good = make_problem(20, 5);
  EXPECT_NO_THROW(dcm::validate(good));
  auto p = good;
  p.fold.pop_back();
  EXPECT_THROW(dcm::validate(p), std::invalid_argument);
  p = good;
  p.contexts[3][1] = 2;  // "flip" has two values
  EXPECT_THROW(dcm::validate(p), std::invalid_argument);
  p = good;
  p.groups.push_back({"bad", {7}});
  EXPECT_THROW(dcm::validate(p), std::invalid_argument);
  p = good;
  p.fold.back() = 0;  // a test-site row with a training fold
  EXPECT_THROW(dcm::validate(p), std::invalid_argument);
  p = good;
  p.x[0][0] = std::nan("");
  EXPECT_THROW(dcm::validate(p), std::invalid_argument);
  p = good;
  p.holdout = {false, true, false, false, false};  // one test site: no inner leave-one-site-out
  dcm::SearchOptions o;
  o.configs = 2;
  o.global = false;
  o.rfonly = false;
  EXPECT_THROW((void)dcm::run_search(p, o), std::invalid_argument);
  // the auc objective needs 0/1 labels; value objectives need finite targets, matching scale features and a delta
  p = good;
  p.y[0] = 0.5;
  EXPECT_THROW(dcm::validate(p), std::invalid_argument);
  EXPECT_NO_THROW(dcm::validate(p, dcm::Objective::mse));
  p.y[0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(dcm::validate(p, dcm::Objective::mse), std::invalid_argument);
  p = good;
  p.z.assign(p.y.size(), {1.0, 2.0});
  EXPECT_NO_THROW(dcm::validate(p, dcm::Objective::laplace_bits));
  p.z[4].pop_back();
  EXPECT_THROW(dcm::validate(p, dcm::Objective::laplace_bits), std::invalid_argument);
  p.z.pop_back();
  EXPECT_THROW(dcm::validate(p, dcm::Objective::laplace_bits), std::invalid_argument);
  p = good;
  p.delta = 0.0;
  EXPECT_THROW(dcm::validate(p, dcm::Objective::laplace_bits), std::invalid_argument);
  EXPECT_NO_THROW(dcm::validate(p, dcm::Objective::mse));
}

namespace {

// "good" needs the expensive component (10 ms) shared with "junk"; "noise" a cheap one (1 ms); the "flip" context
// costs 5 ms; base 0.5 ms.
dcm::SearchCost make_cost(const dcm::SearchProblem& p) {
  dcm::SearchCost c;
  c.components = {"encoder", "cheap", "scene"};
  c.ms = {10.0, 1.0, 5.0};
  c.group_components = {{0}, {1}, {0}};
  c.context_components = {{}, {2}, {}};
  c.base_ms = 0.5;
  EXPECT_EQ(c.group_components.size(), p.groups.size());
  return c;
}

}  // namespace

TEST(DcmSearch, CostCountsSharedComponentsOnce) {
  const auto p = make_problem(20, 1);
  const auto c = make_cost(p);
  dcm::SearchConfig cfg;
  cfg.groups = {0, 2};  // good + junk share the encoder: paid once
  EXPECT_DOUBLE_EQ(dcm::config_cost(c, cfg), 10.5);
  cfg.groups = {1};
  cfg.mixer_contexts = {1};
  EXPECT_DOUBLE_EQ(dcm::config_cost(c, cfg), 6.5);
  cfg.mixer_contexts = {};
  cfg.apm_context = 1;  // the APM's context is paid too
  EXPECT_DOUBLE_EQ(dcm::config_cost(c, cfg), 6.5);
  cfg.apm_context = -1;
  cfg.scale_context = 1;  // and the scale net's
  EXPECT_DOUBLE_EQ(dcm::config_cost(c, cfg), 6.5);
  EXPECT_DOUBLE_EQ(dcm::config_cost(dcm::SearchCost{}, cfg), 0.0);
  cfg.groups = {7};
  EXPECT_THROW((void)dcm::config_cost(c, cfg), std::invalid_argument);
}

TEST(DcmSearch, BudgetKeepsEveryChoiceWithinIt) {
  const auto p = make_problem(80, 2);  // the original uses 120 rows per site; every check here holds for any data
  auto o = small_options();
  const auto plain = dcm::run_search(p, o);
  // an inactive budget and no penalty change nothing
  o.cost = make_cost(p);
  const auto same = dcm::run_search(p, o);
  EXPECT_EQ(same.global_version, plain.global_version);
  ASSERT_EQ(same.nested.size(), plain.nested.size());
  for (std::size_t i = 0; i < same.nested.size(); ++i) {
    if (!std::isnan(plain.nested[i])) {
      EXPECT_EQ(same.nested[i], plain.nested[i]);
    }
  }
  // a 7 ms budget rules out the encoder groups and leaves the cheap group and the scene context
  o.cost.budget_ms = 7.0;
  const auto r = dcm::run_search(p, o);
  for (const auto& c : r.nested_choices) {
    EXPECT_LE(dcm::config_cost(o.cost, c.config), 7.0) << dcm::describe(p, o.space, c.config);
    EXPECT_FALSE(has(c.config.groups, 0));
  }
  for (const auto& s : r.global_top) EXPECT_LE(dcm::config_cost(o.cost, s.config), 7.0);
  EXPECT_LE(dcm::config_cost(o.cost, r.global_best.config), 7.0);
  // nothing fits a budget below the base cost
  o.cost.budget_ms = 0.1;
  EXPECT_THROW((void)dcm::run_search(p, o), std::invalid_argument);
}

TEST(DcmSearch, CostPenaltyPrefersCheapConfigurations) {
  const auto p = make_problem(120, 3);
  auto o = small_options();
  o.nested = false;
  o.rfonly = false;
  o.cost = make_cost(p);
  o.cost.penalty = 1.0;  // one AUC point per ms: only the cheapest configuration can win
  const auto r = dcm::run_search(p, o);
  EXPECT_LE(dcm::config_cost(o.cost, r.global_best.config), 1.5) << dcm::describe(p, o.space, r.global_best.config);
  for (std::size_t i = 1; i < r.global_top.size(); ++i) EXPECT_GE(r.global_top[i - 1].score, r.global_top[i].score);
  // a malformed cost table is refused
  o.cost.group_components.pop_back();
  EXPECT_THROW((void)dcm::run_search(p, o), std::invalid_argument);
  o.cost = make_cost(p);
  o.cost.ms[1] = -1.0;
  EXPECT_THROW((void)dcm::run_search(p, o), std::invalid_argument);
}

namespace {

// A value-domain problem: y = slope · good + Laplace noise, where the "slope" context column selects the slope (+1.5
// or −0.5) and the noise scale grows with the scale feature u (z = {u}). The same sites, folds and other columns as
// make_problem.
dcm::SearchProblem make_value_problem(int per, unsigned seed) {
  dcm::SearchProblem p;
  p.input_names = {"good", "noise_a", "noise_b", "junk"};
  p.groups = {{"good", {0}}, {"noise", {1, 2}}, {"junk", {3}}};
  p.context_names = {"shuffle", "slope", "spare"};
  p.context_sizes = {3, 2, 4};
  p.site_names = {"pool", "bin_a", "bin_b", "bin_c", "bin_d"};
  p.holdout = {false, true, true, true, true};
  p.delta = 0.05;
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd;
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::exponential_distribution<double> ex(1.0);
  std::bernoulli_distribution half(0.5);
  for (int s = 0; s < 5; ++s) {
    const int n = s == 0 ? 3 * per : per;
    for (int i = 0; i < n; ++i) {
      const int slope = half(rng) ? 1 : 0;
      const double good = nd(rng), u = uni(rng);
      const double noise = (0.1 + 0.3 * u) * ex(rng) * (half(rng) ? 1.0 : -1.0);
      p.x.push_back({good, nd(rng), nd(rng), nd(rng)});
      p.contexts.push_back({static_cast<int>(rng() % 3), slope, static_cast<int>(rng() % 4)});
      p.z.push_back({u});
      p.y.push_back((slope ? 1.5 : -0.5) * good + noise);
      p.site.push_back(s);
      p.fold.push_back(s == 0 ? i % 3 : -1);
    }
  }
  return p;
}

}  // namespace

TEST(DcmSearch, SearchMinimisesBitsWithAnInformativeContext) {
  const auto p = make_value_problem(120, 11);
  auto o = small_options();
  o.objective = dcm::Objective::laplace_bits;
  o.space.lr1 = o.space.lr2 = {0.01, 0.03, 0.1};
  o.rfonly = false;
  const auto r = dcm::run_search(p, o);
  ASSERT_FALSE(r.global_top.empty());
  const auto& g = r.global_best;
  SCOPED_TRACE(dcm::describe(p, o.space, g.config, o.objective));
  EXPECT_TRUE(has(g.config.groups, 0));
  EXPECT_TRUE(has(g.config.mixer_contexts, 1));
  EXPECT_LT(g.score, 0.0);  // a negated code length
  EXPECT_TRUE(r.global_mixer.starts_with("nvfx-value-mixer v1\n"));
  EXPECT_EQ(dcm::CompactValueNet<float>(r.global_mixer).version(), r.global_version);
  ASSERT_EQ(r.nested_choices.size(), 4u);
  for (const auto& c : r.nested_choices) {
    SCOPED_TRACE(dcm::describe(p, o.space, c.config, o.objective));
    EXPECT_TRUE(has(c.config.mixer_contexts, 1));
  }
  // held-out bits of the nested predictions against the same configurations without mixer contexts, each trained
  // without the site it codes
  ASSERT_EQ(r.nested_b.size(), p.y.size());
  const auto bits = dcm::per_site_loss(p, o.objective, r.nested, r.nested_b);
  double chosen = 0.0, plain = 0.0;
  for (const auto& c : r.nested_choices) {
    auto no_ctx = c.config;
    no_ctx.mixer_contexts.clear();
    std::vector<std::size_t> train, test;
    for (std::size_t i = 0; i < p.y.size(); ++i) (p.site[i] == c.site ? test : train).push_back(i);
    std::ranges::stable_sort(train, {}, [&](std::size_t i) { return p.site[i]; });
    auto net = dcm::train_value_config(p, o.space, no_ctx, train, o.seed);
    const auto v = dcm::value_spec(p, o.space, no_ctx);
    double sum = 0.0;
    for (const auto i : test) {
      std::vector<int> ctx(v.context_sizes.size() + 3, 0);
      if (no_ctx.apm_context >= 0) ctx[v.context_sizes.size() + 1] = p.contexts[i][static_cast<std::size_t>(no_ctx.apm_context)];
      if (no_ctx.scale_context >= 0) ctx[v.context_sizes.size() + 2] = p.contexts[i][static_cast<std::size_t>(no_ctx.scale_context)];
      std::vector<double> x;  // the groups' columns, ascending (as the search gathers them; these groups do not overlap)
      for (const int gi : no_ctx.groups) {
        for (const int col : p.groups[static_cast<std::size_t>(gi)].columns) x.push_back(p.x[i][static_cast<std::size_t>(col)]);
      }
      (void)net.predict(x, ctx, p.z[i]);
      sum += net.bits(p.y[i], p.delta);
    }
    chosen += bits[static_cast<std::size_t>(c.site)];
    plain += sum / static_cast<double>(test.size());
  }
  chosen /= 4.0;
  plain /= 4.0;
  EXPECT_LT(chosen, plain - 0.5) << "held-out bits per row: chosen " << chosen << ", without context " << plain;

  // the squared-error objective finds the same context
  o.objective = dcm::Objective::mse;
  o.nested = false;
  const auto m = dcm::run_search(p, o);
  EXPECT_TRUE(has(m.global_best.config.mixer_contexts, 1)) << dcm::describe(p, o.space, m.global_best.config, o.objective);
  const auto mse = dcm::per_site_loss(p, o.objective, m.global, {});
  for (int s = 1; s < 5; ++s) EXPECT_LT(mse[static_cast<std::size_t>(s)], 0.4);  // noise alone: about 0.14; without the context: about 1
}

// --- k-means contexts -----------------------------------------------------------------------------------------------

namespace {

// Four well-separated Gaussian blobs in 5-D (centres 20 apart, unit spread), `per` rows each; truth[i] is the blob.
std::vector<double> blobs(int per, unsigned seed, std::vector<int>& truth) {
  const std::vector<std::vector<double>> centres{
      {0, 0, 0, 0, 0}, {20, 0, 0, 0, 0}, {0, 20, 0, 0, 0}, {0, 0, 20, 20, 0}};
  std::mt19937 rng(seed);
  std::normal_distribution<double> noise(0.0, 1.0);
  std::vector<double> x(static_cast<std::size_t>(4 * per) * 5);
  truth.clear();
  for (int i = 0; i < 4 * per; ++i) {
    const int b = i % 4;  // interleaved, so cluster order is not the row order
    truth.push_back(b);
    const auto& centre = centres[static_cast<std::size_t>(b)];
    for (int j = 0; j < 5; ++j) x[static_cast<std::size_t>(i * 5 + j)] = centre[static_cast<std::size_t>(j)] + noise(rng);
  }
  return x;
}

}  // namespace

TEST(DcmKmeans, RecoversSeparatedBlobs) {
  std::vector<int> truth;
  const auto x = blobs(50, 1, truth);
  dcm::KMeansOptions o;
  o.k = 4;
  const auto km = dcm::fit_kmeans(x, 5, o);
  ASSERT_EQ(km.labels.size(), truth.size());
  EXPECT_DOUBLE_EQ(dcm::adjusted_rand_index(km.labels, truth), 1.0);
  EXPECT_DOUBLE_EQ(dcm::normalized_mutual_info(km.labels, truth), 1.0);
  EXPECT_EQ(km.clusters(), 4);
  EXPECT_EQ(km.dims, 5u);
  EXPECT_LT(km.inertia, 200.0 * 5 * 2.0);  // about n · d · σ² = 1000 for the right partition
  EXPECT_EQ(km.assign(x, 5), km.labels);
  // new rows from the same blobs land in the cluster of their blob
  std::vector<int> truth2;
  const auto x2 = blobs(10, 2, truth2);
  const auto a2 = km.assign(x2, 5);
  std::vector<int> joined = km.labels, joined_truth = truth;
  joined.insert(joined.end(), a2.begin(), a2.end());
  joined_truth.insert(joined_truth.end(), truth2.begin(), truth2.end());
  EXPECT_DOUBLE_EQ(dcm::adjusted_rand_index(joined, joined_truth), 1.0);
  EXPECT_THROW((void)km.assign(x2, 4), std::invalid_argument);
}

TEST(DcmKmeans, RepeatsForASeedAndNumbersBySize) {
  std::mt19937 rng(7);
  std::normal_distribution<double> noise(0.0, 1.0);
  std::vector<double> x(900);  // one shapeless cloud of 300 rows in 3-D: the partition depends on the starts
  for (auto& v : x) v = noise(rng);
  dcm::KMeansOptions o;
  o.k = 5;
  o.seed = 3;
  const auto a = dcm::fit_kmeans(x, 3, o), b = dcm::fit_kmeans(x, 3, o);
  EXPECT_EQ(a.labels, b.labels);
  EXPECT_EQ(a.centroids, b.centroids);
  EXPECT_EQ(a.inertia, b.inertia);
  std::vector<int> size(5, 0);
  for (const int l : a.labels) ++size[static_cast<std::size_t>(l)];
  for (std::size_t j = 1; j < size.size(); ++j) EXPECT_GE(size[j - 1], size[j]);
  // more starts never end with a higher inertia (the first start of both runs is the same)
  o.restarts = 1;
  EXPECT_GE(dcm::fit_kmeans(x, 3, o).inertia, a.inertia);
}

TEST(DcmKmeans, HandlesDuplicateRowsAndRejectsBadK) {
  std::vector<double> x(12, 0.0);  // 6 x 2: two distinct points, three copies each
  for (std::size_t i = 3; i < 6; ++i) x[i * 2] = 1.0;
  dcm::KMeansOptions o;
  o.k = 3;  // more clusters than distinct points: one cluster is repaired from a duplicate
  const auto km = dcm::fit_kmeans(x, 2, o);
  EXPECT_DOUBLE_EQ(km.inertia, 0.0);
  std::vector<int> size(3, 0);
  for (const int l : km.labels) ++size[static_cast<std::size_t>(l)];
  for (const int s : size) EXPECT_GT(s, 0);
  o.k = 7;
  EXPECT_THROW((void)dcm::fit_kmeans(x, 2, o), std::invalid_argument);
  o.k = 0;
  EXPECT_THROW((void)dcm::fit_kmeans(x, 2, o), std::invalid_argument);
  o.k = 2;
  EXPECT_THROW((void)dcm::fit_kmeans(x, 5, o), std::invalid_argument);  // not whole rows
  // a NaN or an infinity is refused (a NaN distance is never the nearest, which left rows without a cluster)
  auto bad = x;
  bad[2 * 2 + 1] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW((void)dcm::fit_kmeans(bad, 2, o), std::invalid_argument);
  EXPECT_THROW((void)km.assign(bad, 2), std::invalid_argument);
  EXPECT_THROW((void)dcm::fit_pca(bad, 2, 1), std::invalid_argument);
  bad[2 * 2 + 1] = std::numeric_limits<double>::infinity();
  EXPECT_THROW((void)dcm::fit_kmeans(bad, 2, o), std::invalid_argument);
  EXPECT_THROW((void)dcm::fit_pca(x, 2, 0), std::invalid_argument);
  EXPECT_THROW((void)dcm::fit_pca(std::vector<double>{1.0, 2.0}, 2, 1), std::invalid_argument);  // one row
}

TEST(DcmKmeans, PcaKeepsTheDominantDirection) {
  std::mt19937 rng(11);
  std::normal_distribution<double> noise(0.0, 1.0);
  const std::vector<double> dir{0.6, -0.8, 0.0, 0.0};  // unit; largest entry negative, so the sign rule flips it
  std::vector<double> x(500 * 4);
  for (int i = 0; i < 500; ++i) {
    const double s = 10.0 * noise(rng);
    for (int j = 0; j < 4; ++j) {
      x[static_cast<std::size_t>(i * 4 + j)] = 5.0 + s * dir[static_cast<std::size_t>(j)] + 0.3 * noise(rng);
    }
  }
  const auto pca = dcm::fit_pca(x, 4, 2);
  ASSERT_EQ(pca.count(), 2u);
  EXPECT_NEAR(pca.components[0], -0.6, 0.01);
  EXPECT_NEAR(pca.components[1], 0.8, 0.01);
  double dot = 0.0, norm0 = 0.0;
  for (std::size_t j = 0; j < 4; ++j) {
    dot += pca.components[j] * pca.components[4 + j];
    norm0 += pca.components[j] * pca.components[j];
  }
  EXPECT_NEAR(dot, 0.0, 1e-9);
  EXPECT_NEAR(norm0, 1.0, 1e-9);
  EXPECT_GT(pca.variance[0], 50.0 * pca.variance[1]);
  EXPECT_NEAR(pca.mean[0], 5.0, 1.0);
  const auto z = pca.project(x, 4);
  ASSERT_EQ(z.size(), 500u * 2u);
  double mean = 0.0, var = 0.0;
  for (std::size_t i = 0; i < 500; ++i) mean += z[i * 2];
  mean /= 500.0;
  for (std::size_t i = 0; i < 500; ++i) var += (z[i * 2] - mean) * (z[i * 2] - mean);
  var /= 499.0;
  EXPECT_NEAR(mean, 0.0, 1e-9);
  EXPECT_NEAR(var, pca.variance[0], 1e-6);
  EXPECT_EQ(dcm::fit_pca(x, 4, 9).count(), 4u);  // capped at d
  EXPECT_EQ(dcm::fit_pca(x, 4, 2).components, pca.components);
  EXPECT_THROW((void)pca.project(x, 2), std::invalid_argument);
}

TEST(DcmKmeans, AdjustedRandIndexKnownValues) {
  const std::vector<int> a{0, 0, 0, 1, 1, 1}, renamed{5, 5, 5, 2, 2, 2}, b{0, 0, 1, 1, 2, 2};
  EXPECT_DOUBLE_EQ(dcm::adjusted_rand_index(a, renamed), 1.0);
  // pairs together in both: 2; expected 6 · 3 / 15 = 1.2; max (6 + 3) / 2 = 4.5 → 0.8 / 3.3
  EXPECT_NEAR(dcm::adjusted_rand_index(a, b), 0.8 / 3.3, 1e-12);
  EXPECT_NEAR(dcm::adjusted_rand_index(b, a), 0.8 / 3.3, 1e-12);
  const std::vector<int> one(6, 0), singletons{0, 1, 2, 3, 4, 5};
  EXPECT_DOUBLE_EQ(dcm::adjusted_rand_index(one, one), 1.0);
  EXPECT_DOUBLE_EQ(dcm::adjusted_rand_index(one, singletons), 0.0);
  const std::vector<int> shorter{0, 1};
  EXPECT_THROW((void)dcm::adjusted_rand_index(a, shorter), std::invalid_argument);
}

TEST(DcmKmeans, NormalizedMutualInfoKnownValues) {
  const std::vector<int> a{0, 0, 0, 1, 1, 1}, renamed{5, 5, 5, 2, 2, 2}, b{0, 0, 1, 1, 2, 2};
  const std::vector<int> one(6, 0);
  EXPECT_DOUBLE_EQ(dcm::normalized_mutual_info(one, one), 1.0);
  EXPECT_DOUBLE_EQ(dcm::normalized_mutual_info(one, a), 0.0);
  EXPECT_NEAR(dcm::normalized_mutual_info(a, renamed), 1.0, 1e-12);
  // I(a; b) = (2/3) ln 2, H(a) = ln 2, H(b) = ln 3: NMI = (4/3) ln 2 / ln 6
  EXPECT_NEAR(dcm::normalized_mutual_info(a, b), 4.0 / 3.0 * std::log(2.0) / std::log(6.0), 1e-12);
  EXPECT_NEAR(dcm::normalized_mutual_info(b, a), dcm::normalized_mutual_info(a, b), 1e-15);
  // a finer labelling that splits each cluster of a: H grows, I stays H(a)
  const std::vector<int> split{0, 0, 1, 2, 2, 3};
  EXPECT_GT(dcm::normalized_mutual_info(a, split), 0.5);
  EXPECT_LT(dcm::normalized_mutual_info(a, split), 1.0);
  EXPECT_THROW((void)dcm::normalized_mutual_info(a, std::vector<int>{0}), std::invalid_argument);
}
