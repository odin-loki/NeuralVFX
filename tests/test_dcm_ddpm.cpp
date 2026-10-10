// The coarse-state denoiser (include/neuralfx/dcm/ddpm.hpp): the fast kernels equal the plain patterns, the
// hand-written gradients match finite differences on an 8 x 8 toy, training lowers the loss, a short training gives one
// SHA-256 over repeats (and for any thread count), and the helpers (features at a fixed noise level, Tweedie's denoise,
// the prior step, DDIM, SDEdit, contexts) behave as documented.
#include <neuralfx/dcm/ddpm.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <set>
#include <string>
#include <vector>

using namespace nfx::dcm;
using namespace nfx::dcm::ddpm;

namespace {

Config toy_config() {
  Config c;
  c.res = 8;
  c.channels = 4;
  c.c0 = 8;
  c.c1 = 8;
  c.c2 = 16;
  c.cond = 2;
  c.freqs = 4;
  c.film_hidden = 8;
  c.timesteps = 100;
  return c;
}

// Every weight random (the zero-initialised layers too), so every path carries a gradient.
std::vector<float> random_weights(const Config& c, std::uint64_t seed, float sd) {
  std::vector<float> w(layout(c).size);
  gaussian(seed, w);
  for (float& v : w) v *= sd;
  return w;
}

// Smooth blobs that move with the condition: structure a denoiser can learn quickly.
Dataset toy_set(const Config& c, int count, std::uint64_t seed) {
  Dataset d;
  d.count = count;
  d.values = c.res * c.res * c.channels;
  d.conds = c.cond;
  std::vector<float> g(4);
  for (int i = 0; i < count; ++i) {
    gaussian(seed * 1000 + static_cast<std::uint64_t>(i), g);
    const float cx = 3.5f + 1.5f * std::tanh(g[0]), cy = 3.5f + 1.5f * std::tanh(g[1]);
    const float a = 1.f + 0.3f * g[2];
    for (int y = 0; y < c.res; ++y) {
      for (int x = 0; x < c.res; ++x) {
        const float r2 = (static_cast<float>(x) - cx) * (static_cast<float>(x) - cx) + (static_cast<float>(y) - cy) * (static_cast<float>(y) - cy);
        const float blob = a * std::exp(-r2 / 4.f);
        d.x.push_back(0.5f * (static_cast<float>(y) - cy) * blob);
        d.x.push_back(-0.5f * (static_cast<float>(x) - cx) * blob);
        d.x.push_back(2.f * blob);
        d.x.push_back(blob * blob);
      }
    }
    d.cond.push_back((cx - 2.f) / 3.f);
    d.cond.push_back((cy - 2.f) / 3.f);
  }
  return d;
}

}  // namespace

TEST(Ddpm, FastKernelsEqualThePlainPatterns) {
  // shapes of the network (stem 6 -> 32, blocks 32 and 64 wide, output 32 -> 4) and odd ones (dot path, odd inputs)
  const std::vector<std::array<int, 3>> shapes = {{32, 6, 32}, {16, 64, 64}, {32, 32, 4}, {8, 4, 32}, {8, 8, 16}, {5, 3, 5}, {7, 9, 24}, {4, 7, 8}};
  for (const auto& [res, ci, co] : shapes) {
    EXPECT_LT(kernel_max_error(res, ci, co, 3), 2e-3) << std::format("res {} in {} out {}", res, ci, co);
  }
}

TEST(Ddpm, LayoutAndCost) {
  const Config c;  // the network of the study
  const Layout L = layout(c);
  EXPECT_EQ(L.film_size, static_cast<std::size_t>(2 * (32 + 64 + 64 + 64 + 64 + 32)));
  EXPECT_GT(L.size, 300000u);
  EXPECT_LT(L.size, 500000u);
  EXPECT_GT(forward_macs(c), 80e6);
  EXPECT_LT(forward_macs(c), 100e6);
  const Denoiser d = init_denoiser(c, 1);
  EXPECT_EQ(d.parameters(), L.size);
  // the output layer and the second convolution of every block start at zero
  for (std::size_t i = L.out_w; i < L.out_b + static_cast<std::size_t>(c.channels); ++i) ASSERT_EQ(d.w[i], 0.f);
  for (const BlockLayout& B : L.blocks) {
    for (std::size_t i = B.wb; i < B.bb + static_cast<std::size_t>(B.width); ++i) ASSERT_EQ(d.w[i], 0.f);
  }
  const auto ab = cosine_alpha_bar(1000);
  EXPECT_EQ(ab[0], 1.0);
  EXPECT_GT(ab[1], 0.999);
  EXPECT_LT(ab[1000], 1e-3);
  for (std::size_t t = 1; t < ab.size(); ++t) ASSERT_LT(ab[t], ab[t - 1]);
}

TEST(Ddpm, GradientsMatchFiniteDifferences) {
  const Config c = toy_config();
  std::vector<float> w = random_weights(c, 11, 0.12f);  // loss of order 1: larger weights blow up through six blocks
  const std::size_t n = static_cast<std::size_t>(c.res * c.res * c.channels);
  std::vector<float> x0(n), eps(n);
  gaussian(21, x0);
  gaussian(22, eps);
  const std::vector<float> cond = {0.3f, 0.8f};
  const int t = 37;
  std::vector<float> grad(w.size(), 0.f);
  const double l0 = example_loss(c, w, x0, t, eps, cond, grad);
  ASSERT_GT(l0, 0.0);
  const auto eval = [&] { return example_loss(c, w, x0, t, eps, cond, {}); };
  // As in tests/test_rollout.cpp: central differences at three step sizes and one-sided ones at the two smaller; a
  // parameter passes when one of them matches within 3%. The loss is summed from float activations, so the smallest
  // steps are noisy and the largest are curved; a wrong gradient matches none of them.
  const auto diffs = [&](float& slot) {
    const float keep = slot;
    const double base = eval();
    std::vector<double> out;
    for (const float e : {1e-2f, 3e-3f, 1e-3f}) {
      slot = keep + e;
      const double lp = eval();
      slot = keep - e;
      const double lm = eval();
      slot = keep;
      out.push_back((lp - lm) / (2.0 * static_cast<double>(e)));
      if (e < 5e-3f) {
        out.push_back((lp - base) / static_cast<double>(e));
        out.push_back((base - lm) / static_cast<double>(e));
      }
    }
    return out;
  };
  const auto near = [](double a, double b) { return std::abs(a - b) <= 3e-2 * std::max(std::abs(a), std::abs(b)) + 2e-5; };
  const Layout L = layout(c);
  // every part of the network, sampled
  std::vector<std::pair<std::string, std::size_t>> parts = {{"stem", L.stem_w}, {"down1", L.down1_w}, {"down2", L.down2_w}, {"up2", L.up2_w},
                                                            {"up1", L.up1_w},   {"out", L.out_w},     {"out_b", L.out_b},     {"mlp1", L.mlp1_w},
                                                            {"mlp2", L.mlp2_w}, {"mlp2_b", L.mlp2_b}};
  for (int k = 0; k < kBlocks; ++k) {
    parts.emplace_back(std::format("{}.a", kBlockNames[static_cast<std::size_t>(k)]), L.blocks[static_cast<std::size_t>(k)].wa);
    parts.emplace_back(std::format("{}.b", kBlockNames[static_cast<std::size_t>(k)]), L.blocks[static_cast<std::size_t>(k)].wb);
    parts.emplace_back(std::format("{}.ba", kBlockNames[static_cast<std::size_t>(k)]), L.blocks[static_cast<std::size_t>(k)].ba);
  }
  std::set<std::size_t> picks;
  for (const auto& [name, at] : parts) {
    for (std::size_t j = 0; j < 6; ++j) picks.insert(std::min(w.size() - 1, at + j * 7));
  }
  for (std::size_t i = 0; i < w.size(); i += std::max<std::size_t>(1, w.size() / 200)) picks.insert(i);
  int checked = 0, bad = 0, live = 0;
  for (const std::size_t i : picks) {
    const std::vector<double> nd = diffs(w[i]);
    const double ana = grad[i];
    ++checked;
    live += std::abs(nd[0]) > 2e-5;
    if (std::ranges::any_of(nd, [&](double v) { return near(v, ana); })) continue;
    if (++bad <= 8) ADD_FAILURE() << std::format("param {}: analytic {} central {} {} {}", i, ana, nd[0], nd[1], nd[4]);
  }
  EXPECT_GT(checked, 200);
  EXPECT_EQ(bad, 0);
  EXPECT_GT(live, checked * 3 / 4) << "most gradients are zero: the check would be vacuous";
}

TEST(Ddpm, TrainingLowersTheLoss) {
  const Config c = toy_config();
  Denoiser d = init_denoiser(c, 4);
  const Dataset data = toy_set(c, 96, 1), held = toy_set(c, 48, 2);
  set_range(d, data);
  const double before = eval_loss(d, held, 30, 5, 0, 2);
  TrainOptions o;
  o.steps = 400;
  o.batch = 16;
  o.lr = 2e-3f;
  o.warmup = 50;
  o.threads = 2;
  o.log_every = 100;
  o.eval_t = {30};
  o.eval_count = 48;
  const TrainResult r = train(d, data, o, &held);
  ASSERT_EQ(r.curve.size(), 4u);
  ASSERT_EQ(r.curve.back().eval.size(), 1u);
  EXPECT_DOUBLE_EQ(eval_loss(d, held, 30, o.eval_seed, 48, 1), r.curve.back().eval[0]);  // the log scored the EMA weights
  const double after = eval_loss(d, held, 30, 5, 0, 2);
  EXPECT_LT(after, 0.8 * before) << std::format("held-out loss at t = 30: {} before, {} after", before, after);
  EXPECT_LT(r.curve.back().loss, r.curve.front().loss);
}

TEST(Ddpm, ShortTrainingGivesOneHash) {
  const Config c = toy_config();
  const Dataset data = toy_set(c, 32, 3);
  std::set<std::string> hashes;
  for (int rep = 0; rep < 10; ++rep) {
    Denoiser d = init_denoiser(c, 9);
    set_range(d, data);
    TrainOptions o;
    o.steps = 6;
    o.batch = 8;
    o.warmup = 2;
    o.threads = rep % 2 ? 2 : 1;  // the thread count must not matter either
    o.log_every = 100;
    (void)train(d, data, o);
    hashes.insert(version(d));
  }
  EXPECT_EQ(hashes.size(), 1u);
  // and a different seed gives different weights
  Denoiser d = init_denoiser(c, 9);
  set_range(d, data);
  TrainOptions o;
  o.steps = 6;
  o.batch = 8;
  o.warmup = 2;
  o.seed = 2;
  (void)train(d, data, o);
  EXPECT_FALSE(hashes.contains(version(d)));
}

TEST(Ddpm, SerialisationRoundTrips) {
  const Config c = toy_config();
  Denoiser d = init_denoiser(c, 2);
  d.w = random_weights(c, 3, 0.1f);
  d.scale = {0.5f, 0.6f, 0.7f, 0.8f};
  const auto back = parse(serialise(d));
  ASSERT_TRUE(back.has_value()) << back.error();
  EXPECT_EQ(back->w, d.w);
  EXPECT_EQ(back->scale, d.scale);
  EXPECT_EQ(version(*back), version(d));
  EXPECT_EQ(version(d).size(), 64u);
  std::string bad = serialise(d);
  bad[4] = 'Q';
  EXPECT_FALSE(parse(bad).has_value());
  EXPECT_FALSE(parse(serialise(d).substr(0, 60)).has_value());
}

TEST(Ddpm, TweedieAndThePriorStep) {
  const Config c = toy_config();
  Denoiser d = init_denoiser(c, 2);
  const std::size_t n = static_cast<std::size_t>(c.res * c.res * c.channels);
  std::vector<float> xt(n), x0(n), eps(n);
  gaussian(5, xt);
  const std::vector<float> cond = {0.2f, 0.4f};
  const auto ab = cosine_alpha_bar(c.timesteps);
  // a fresh network predicts zero noise, so Tweedie's denoise only rescales
  tweedie(d, xt, 20, cond, x0);
  for (std::size_t i = 0; i < n; ++i) ASSERT_NEAR(x0[i], xt[i] / std::sqrt(ab[20]), 1e-5);
  // with any weights, the denoise and the predicted noise recompose x_t
  d.w = random_weights(c, 6, 0.2f);
  for (const int t : {5, 50, 99}) {
    tweedie(d, xt, t, cond, x0);
    predict_eps(d, xt, t, cond, eps);
    for (std::size_t i = 0; i < n; ++i) ASSERT_NEAR(std::sqrt(ab[static_cast<std::size_t>(t)]) * x0[i] + std::sqrt(1.0 - ab[static_cast<std::size_t>(t)]) * eps[i], xt[i], 1e-4);
  }
  // the prior step: beta 0 changes nothing; beta 1 moves x by sqrt((1 - ab) / ab) eps_hat(sqrt(ab) x)
  std::vector<float> x(xt), y(xt), xs(n);
  prior_step(d, x, 10, 0.f, cond);
  EXPECT_EQ(x, xt);
  prior_step(d, y, 10, 1.f, cond);
  for (std::size_t i = 0; i < n; ++i) xs[i] = static_cast<float>(std::sqrt(ab[10])) * xt[i];
  predict_eps(d, xs, 10, cond, eps);
  const double r = std::sqrt((1.0 - ab[10]) / ab[10]);
  for (std::size_t i = 0; i < n; ++i) ASSERT_NEAR(y[i], std::clamp(static_cast<float>(xt[i] - r * eps[i]), d.lo[i % 4], d.hi[i % 4]), 1e-5);
}

TEST(Ddpm, FeaturesAtAFixedNoiseLevel) {
  const Config c = toy_config();
  Denoiser d = init_denoiser(c, 2);
  d.w = random_weights(c, 8, 0.2f);
  const std::size_t n = static_cast<std::size_t>(c.res * c.res * c.channels);
  std::vector<float> x0(n);
  gaussian(9, x0);
  const std::vector<float> cond = {0.5f, 0.5f};
  const Features a = features_at(d, x0, 40, cond, 7), b = features_at(d, x0, 40, cond, 7);
  EXPECT_EQ(a.r1, 4);
  EXPECT_EQ(a.r2, 2);
  EXPECT_EQ(a.enc1.size(), static_cast<std::size_t>(4 * 4 * c.c1));
  EXPECT_EQ(a.mid.size(), static_cast<std::size_t>(2 * 2 * c.c2));
  EXPECT_EQ(a.dec1.size(), static_cast<std::size_t>(4 * 4 * c.c1));
  EXPECT_EQ(a.enc1, b.enc1);  // bit-identical: the noise image is fixed by (t, seed)
  EXPECT_EQ(a.mid, b.mid);
  EXPECT_EQ(a.dec1, b.dec1);
  EXPECT_NE(features_at(d, x0, 60, cond, 7).mid, a.mid);
  EXPECT_NE(features_at(d, x0, 40, cond, 8).mid, a.mid);
  std::vector<float> nz(n), nz2(n);
  fixed_noise(c, 40, 7, nz);
  fixed_noise(c, 40, 7, nz2);
  EXPECT_EQ(nz, nz2);
}

TEST(Ddpm, SamplingIsDeterministicAndInRange) {
  const Config c = toy_config();
  Denoiser d = init_denoiser(c, 2);
  d.w = random_weights(c, 10, 0.1f);
  const Dataset data = toy_set(c, 16, 4);
  set_range(d, data);
  const std::size_t n = static_cast<std::size_t>(c.res * c.res * c.channels);
  std::vector<float> a(n), b(n), e(n);
  const std::vector<float> cond = {0.4f, 0.6f};
  sample(d, cond, 10, 3, a);
  sample(d, cond, 10, 3, b);
  EXPECT_EQ(a, b);
  for (std::size_t i = 0; i < n; ++i) ASSERT_TRUE(a[i] >= d.lo[i % 4] - 1e-4f && a[i] <= d.hi[i % 4] + 1e-4f);
  sample(d, cond, 10, 4, b);
  EXPECT_NE(a, b);
  sdedit(d, data.state(0), 40, 10, cond, 5, e);
  for (const float v : e) ASSERT_TRUE(std::isfinite(v));
  EXPECT_EQ(ddim_passes(c, 100, 10), 10);
  EXPECT_EQ(ddim_passes(c, 40, 10), 4);
  EXPECT_EQ(ddim_passes(Config{}, 400, 25), 10);
  EXPECT_EQ(sample_start(Config{}, 25), 961);
  EXPECT_EQ(ddim_passes(Config{}, sample_start(Config{}, 25), 25), 25);
}

TEST(Ddpm, ContextPlanes) {
  const Config c = toy_config();
  Denoiser d = init_denoiser(c, 2);
  d.w = random_weights(c, 12, 0.2f);
  const Dataset data = toy_set(c, 40, 6);
  std::vector<int> states(40);
  for (int i = 0; i < 40; ++i) states[static_cast<std::size_t>(i)] = i;
  for (const bool diffusion : {true, false}) {
    ContextSpec s;
    s.diffusion = diffusion;
    s.ts = {30, 60};
    s.pca = 4;
    s.ks = {2, 4};
    const ContextModel cm = fit_contexts(&d, s, data, states, c.res, c.channels, 2);
    ASSERT_EQ(cm.km.size(), 2u);
    std::vector<float> phys(data.state(3).begin(), data.state(3).end());
    const std::vector<float> scale(4, 1.f);
    const auto planes = context_planes(&d, cm, phys, 4, scale, data.condition(3), c.res);
    ASSERT_EQ(planes.size(), 2u);
    for (std::size_t k = 0; k < 2; ++k) {
      ASSERT_EQ(planes[k].size(), static_cast<std::size_t>(kRegions * kRegions));
      for (const int v : planes[k]) ASSERT_TRUE(v >= 0 && v < s.ks[k]);
    }
    EXPECT_EQ(planes, context_planes(&d, cm, phys, 4, scale, data.condition(3), c.res));
  }
  const HandMade h = fit_handmade(data, states, c.res, c.channels);
  const auto hm = handmade_contexts(h, data.state(0), std::vector<float>{0.7f, 0.2f, 0.9f}, c.res, c.channels);
  ASSERT_EQ(hm.size(), static_cast<std::size_t>(kRegions * kRegions));
  for (const auto& r : hm) {
    EXPECT_TRUE(r[0] >= 0 && r[0] < 4 && r[1] >= 0 && r[1] < 4 && r[2] >= 0 && r[2] < 3);
    EXPECT_EQ(r[3], 1 + 4);
  }
}
