// Phase 3: the model description and file format, the trainer's forward pass against the reference, its gradients
// against finite differences, and that training actually learns.
#include <neuralfx/metrics.hpp>
#include <neuralfx/model.hpp>
#include <neuralfx/train.hpp>

#include "../src/train/net.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <format>
#include <algorithm>
#include <array>
#include <random>
#include <sstream>
#include <unistd.h>

using namespace nfx;
using train::detail::Grads;
using train::detail::Net;

namespace {

Hyper tiny_grid() {
  Hyper h;
  h.arch = Arch::grid;
  h.size = 8;
  h.frames = 4;
  h.loop = true;
  h.n_controls = 2;
  h.n_latent = 1;
  h.bases = 2;
  h.grid_t = 3;
  h.grid = 4;
  h.channels = 3;
  h.hidden = 5;
  h.layers = 2;
  return h;
}

Hyper tiny_conv() {
  Hyper h;
  h.arch = Arch::conv;
  h.latent = 2;
  h.size = 16;
  h.frames = 4;
  h.loop = false;
  h.n_controls = 2;
  h.n_latent = 1;
  h.bases = 2;
  h.grid_t = 3;
  h.c0 = 3;
  h.c1 = 3;
  h.c2 = 2;
  return h;
}

// Give the zero-initialised FiLM layers some weight so their gradients are exercised away from identity.
Model randomised(const Hyper& h, std::uint64_t seed) {
  Model m = init_model(h, seed);
  std::mt19937_64 rng(seed + 1);
  std::uniform_real_distribution<float> u(-0.3f, 0.3f);
  for (auto& f : m.films) {
    for (float& w : f.w) w = u(rng);
    for (float& b : f.b) b = u(rng);
  }
  for (auto& l : m.layers) {
    for (float& b : l.b) b = u(rng);
  }
  for (float& w : m.layers.back().w) w *= 10.f;  // undo the small output gain so every gradient is sizeable
  for (float& f : m.features) f *= 8.f;  // wake the relus: a dead network passes any gradient check trivially
  return m;
}

std::vector<float> all_params(Model& m, std::vector<float*>& ptrs) {
  ptrs.clear();
  for (float& v : m.features) ptrs.push_back(&v);
  for (auto* d : {&m.basis}) {
    for (float& v : d->w) ptrs.push_back(&v);
    for (float& v : d->b) ptrs.push_back(&v);
  }
  for (auto* group : {&m.layers, &m.films}) {
    for (auto& d : *group) {
      for (float& v : d.w) ptrs.push_back(&v);
      for (float& v : d.b) ptrs.push_back(&v);
    }
  }
  std::vector<float> vals;
  for (float* p : ptrs) vals.push_back(*p);
  return vals;
}

void gradient_check(const Hyper& h) {
  Model m = randomised(h, 3);
  const int size = h.size;
  std::mt19937_64 rng(5);
  std::vector<std::uint8_t> target(static_cast<std::size_t>(size) * size * 4);
  for (auto& v : target) v = static_cast<std::uint8_t>(rng() % 256);
  std::vector<int> pixels(static_cast<std::size_t>(size) * size);
  for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = static_cast<int>(i);
  std::vector<float> c = {0.3f, -0.7f, 0.5f};
  const float t = 0.4f;
  Net net(m);
  const auto loss = [&](const Model& mm, std::span<const float> cc) {
    Grads g(mm);
    std::vector<float> dc(cc.size());
    return net.step(mm, g, t, cc, target, pixels, size, 1.f, dc);
  };
  Grads g(m);
  std::vector<float> dc(c.size(), 0.f);
  net.step(m, g, t, c, target, pixels, size, 1.f, dc);
  Model gm = g.g;
  std::vector<float*> pp, gp;
  all_params(m, pp);
  all_params(gm, gp);
  ASSERT_EQ(pp.size(), gp.size());
  // Central differences at three step sizes. Float round-off spoils small steps and relu kinks spoil large ones, so a
  // parameter passes when the analytic gradient matches at one of them; a wrong gradient is wrong at every step and
  // fails. Where the three estimates scatter by more than 10% without a match, the parameter sits on a kink and is
  // skipped (at most a tenth may be).
  const auto numeric = [&](float& slot, float eps, auto&& eval) {
    const float keep = slot;
    slot = keep + eps;
    const double lp = eval();
    slot = keep - eps;
    const double lm = eval();
    slot = keep;
    return (lp - lm) / (2.0 * static_cast<double>(eps));
  };
  const auto near = [](double a, double b) { return std::abs(a - b) <= 1e-2 * std::max(std::abs(a), std::abs(b)) + 3e-4; };
  int checked = 0, bad = 0, live = 0, kinks = 0;
  const auto check_one = [&](float& slot, double ana, std::string_view what, auto&& eval) {
    const std::array<double, 3> n = {numeric(slot, 1e-2f, eval), numeric(slot, 3e-3f, eval), numeric(slot, 1e-3f, eval)};
    if (std::ranges::any_of(n, [&](double v) { return near(v, ana); })) {
      ++checked;
      live += std::abs(n[0]) > 1e-3;
      return;
    }
    const auto [lo, hi] = std::ranges::minmax(n);
    if (hi - lo > 0.1 * std::max(std::abs(lo), std::abs(hi)) + 1e-3) {
      ++kinks;
      return;
    }
    ++checked;
    if (++bad <= 6) ADD_FAILURE() << std::format("{}: analytic {} numeric {} {} {}", what, ana, n[0], n[1], n[2]);
  };
  const auto model_loss = [&] { return loss(m, c); };
  for (std::size_t i = 0; i < pp.size(); i += std::max<std::size_t>(1, pp.size() / 300)) {
    check_one(*pp[i], *gp[i], std::format("param {}", i), model_loss);
  }
  for (std::size_t j = 0; j < c.size(); ++j) check_one(c[j], dc[j], std::format("condition {}", j), model_loss);
  EXPECT_GT(checked, 50);
  EXPECT_LE(kinks, (checked + kinks) / 10) << "too many parameters at relu kinks";
  EXPECT_GT(live, checked / 2) << "most gradients are zero: the check would be vacuous";
}

Clip smooth_clip(int size, int frames, float phase) {
  Clip c;
  c.allocate(size, frames);
  c.loop = true;
  for (int f = 0; f < frames; ++f) {
    auto fr = c.frame(f);
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        const double t = 2 * 3.14159265 * f / frames + phase;
        const double a = std::clamp(0.5 + 0.45 * std::sin(0.25 * x + t) * std::cos(0.2 * y - 0.5 * t), 0.0, 1.0);
        const std::size_t i = (static_cast<std::size_t>(y) * size + x) * 4;
        fr[i] = static_cast<std::uint8_t>(255 * a);
        fr[i + 1] = static_cast<std::uint8_t>(160 * a);
        fr[i + 2] = static_cast<std::uint8_t>(60 * a);
        fr[i + 3] = static_cast<std::uint8_t>(255 * a);
      }
    }
  }
  return c;
}

}  // namespace

TEST(Model, ShapesCountsAndCost) {
  Hyper h = tiny_grid();
  const Model m = init_model(h, 1);
  EXPECT_EQ(m.features.size(), 2u * 3 * 3 * 4 * 4);
  EXPECT_EQ(m.layers.size(), 3u);
  EXPECT_EQ(m.films.size(), 1u);
  EXPECT_EQ(m.films[0].out, 10);
  EXPECT_GT(m.macs_per_pixel(8), 0.0);
  const Model c = init_model(tiny_conv(), 1);
  EXPECT_EQ(c.layers[0].in, 27);
  EXPECT_EQ(c.films.size(), 2u);
  Hyper bad = tiny_conv();
  bad.size = 20;
  EXPECT_THROW(init_model(bad, 1), std::invalid_argument);
}

TEST(Model, SaveLoadRoundTripsAtEveryPrecision) {
  for (const int bits : {16, 8, 6, 5, 4, 3, 2}) {
    Model m = randomised(tiny_grid(), 7);
    m.effect = "fire";
    m.feature_bits = bits;
    m.z_train = {{0.1f}, {-0.2f}};
    m.z_mean = {-0.05f};
    m.z_std = {0.15f};
    const auto path = std::filesystem::temp_directory_path() / std::format("nfx_model_{}_{}.nvfx", ::getpid(), bits);
    ASSERT_TRUE(save_model(path, m).has_value());
    const auto r = load_model(path);
    ASSERT_TRUE(r.has_value()) << r.error();
    Model q = m;
    quantise_like_storage(q);
    EXPECT_EQ(r->features, q.features);
    EXPECT_EQ(r->layers[1].w, q.layers[1].w);
    EXPECT_EQ(r->films[0].b, q.films[0].b);
    EXPECT_EQ(r->effect, "fire");
    EXPECT_EQ(r->z_train.size(), 2u);
    EXPECT_NEAR(r->z_std[0], 0.15f, 1e-3f);
    // The storage formula agrees with the file within the fixed header.
    const auto file = std::filesystem::file_size(path);
    EXPECT_LE(m.storage_bytes(), file);
    EXPECT_LT(file - m.storage_bytes(), 256u);
    std::filesystem::remove(path);
  }
}

TEST(Model, EightBitFeaturesHalveFeatureStorage) {
  Hyper h = tiny_grid();
  h.grid = 32;
  Model m = init_model(h, 1);
  const auto b16 = m.storage_bytes();
  m.feature_bits = 8;
  EXPECT_LT(m.storage_bytes(), b16 * 6 / 10);
}

TEST(Model, PackedFeaturesUseTheirBitsAndRoundTrip) {
  // Codes of every width survive packing, whatever the plane length (planes may end inside a byte).
  for (const int bits : {2, 3, 4, 5, 6, 7}) {
    for (const std::size_t n : {1u, 7u, 13u, 144u, 169u, 1024u}) {
      std::vector<std::uint8_t> buf(packed_plane_bytes(n, bits) + 1, 0);
      buf.back() = 0xa5;  // a guard byte: nothing writes past the plane
      std::mt19937 rng(static_cast<unsigned>(bits * 1000 + n));
      std::vector<unsigned> codes(n);
      for (auto& c : codes) c = static_cast<unsigned>(rng()) & ((1u << bits) - 1u);
      for (std::size_t j = 0; j < n; ++j) put_packed_code(buf.data(), j, bits, codes[j]);
      EXPECT_EQ(buf.back(), 0xa5);
      for (std::size_t j = 0; j < n; ++j) ASSERT_EQ(packed_code(buf.data(), j, bits), codes[j]) << bits << " " << n << " " << j;
    }
  }
  // Storage shrinks with the bits: a 32 x 32 grid's features at 4 bits take a quarter of fp16 (plus the ranges).
  Hyper h = tiny_grid();
  h.grid = 32;
  Model m = init_model(h, 1);
  m.feature_bits = 16;
  const auto b16 = m.storage_bytes();
  m.feature_bits = 4;
  const std::size_t planes = static_cast<std::size_t>(h.bases) * h.grid_t * h.channels;
  EXPECT_EQ(b16 - m.storage_bytes(), m.features.size() * 2 - (m.features.size() / 2 + planes * 4));
}

TEST(Train, ForwardMatchesTheReference) {
  for (const Hyper& h : {tiny_grid(), tiny_conv()}) {
    const Model m = randomised(h, 11);
    const std::vector<float> c = {0.2f, 0.9f, -0.4f};
    std::vector<float> ref(static_cast<std::size_t>(h.size) * h.size * 4), got(ref.size());
    reference_render(m, 0.63f, c, h.size, ref);
    Net net(m);
    net.render(m, 0.63f, c, h.size, got);
    for (std::size_t i = 0; i < ref.size(); ++i) ASSERT_NEAR(ref[i], got[i], 1e-4f) << h.describe() << " at " << i;
  }
}

TEST(Train, GridGradientsMatchFiniteDifferences) { gradient_check(tiny_grid()); }
TEST(Train, ConvGradientsMatchFiniteDifferences) { gradient_check(tiny_conv()); }

TEST(Train, LearnsASmoothClip) {
  const Clip clip = smooth_clip(32, 8, 0.f);
  Hyper h;
  h.arch = Arch::grid;
  h.size = 32;
  h.frames = 8;
  h.grid = 16;
  h.grid_t = 8;
  h.channels = 4;
  h.hidden = 16;
  h.layers = 1;
  train::Options o;
  o.iterations = 400;
  o.batch_frames = 4;
  o.pixels = 512;
  o.threads = 2;
  o.log_every = 0;
  const train::Example ex{&clip, {}};
  const auto r = train::train(h, std::span(&ex, 1), o);
  const Clip out = train::render_clip(r.model, {}, {}, 8, 32);
  EXPECT_GT(metrics::score(clip, out).psnr, 28.0);
  EXPECT_LT(r.final_loss, 2e-3);
}

TEST(Train, ConditioningSeparatesTwoClips) {
  const Clip a = smooth_clip(32, 8, 0.f), b = smooth_clip(32, 8, 2.5f);
  Hyper h;
  h.arch = Arch::grid;
  h.size = 32;
  h.frames = 8;
  h.grid = 16;
  h.grid_t = 8;
  h.channels = 4;
  h.hidden = 16;
  h.layers = 2;
  h.n_controls = 1;
  h.bases = 2;
  train::Options o;
  o.iterations = 600;
  o.batch_frames = 4;
  o.pixels = 512;
  o.threads = 2;
  o.log_every = 0;
  const train::Example ex[] = {{&a, {0.f}}, {&b, {1.f}}};
  const auto r = train::train(h, ex, o);
  const Clip ra = train::render_clip(r.model, std::array{0.f}, {}, 8, 32);
  const Clip rb = train::render_clip(r.model, std::array{1.f}, {}, 8, 32);
  EXPECT_GT(metrics::score(a, ra).psnr, metrics::score(b, ra).psnr + 5.0);
  EXPECT_GT(metrics::score(b, rb).psnr, metrics::score(a, rb).psnr + 5.0);
}

TEST(Train, ConvFamilyLearnsToo) {
  const Clip clip = smooth_clip(32, 8, 1.f);
  Hyper h;
  h.arch = Arch::conv;
  h.latent = 4;
  h.size = 32;
  h.frames = 8;
  h.grid_t = 8;
  h.c0 = 8;
  h.c1 = 8;
  h.c2 = 8;
  train::Options o;
  o.iterations = 400;
  o.batch_frames = 4;
  o.threads = 2;
  o.log_every = 0;
  const train::Example ex{&clip, {}};
  const auto r = train::train(h, std::span(&ex, 1), o);
  EXPECT_GT(metrics::score(clip, train::render_clip(r.model, {}, {}, 8, 32)).psnr, 24.0);
}

TEST(Train, FakeQuantiseIsTheStorageRounding) {
  for (const bool trim : {false, true}) {
    for (const int bits : {8, 6, 5, 4, 3}) {
      Hyper h = tiny_grid();
      h.grid = 8;
      Model m = randomised(h, 11);
      Model a = m, b = m;
      train::fake_quantise(a, bits, trim);
      b.feature_bits = bits;
      b.feature_trim = trim;
      quantise_like_storage(b);
      EXPECT_EQ(a.features, b.features) << bits << " " << trim;
      EXPECT_EQ(a.layers[0].w, m.layers[0].w);  // only the features change
      // ...and it is what a saved file holds.
      std::stringstream ss;
      ASSERT_TRUE(save_model(ss, b));
      const auto back = load_model(ss);
      ASSERT_TRUE(back.has_value());
      EXPECT_EQ(back->features, a.features) << bits << " " << trim;
    }
  }
}

TEST(Model, TrimmedRangesQuantiseHeavyTailsBetter) {
  // A bell-shaped plane with a few outliers: at 4 bits, clipping the tails lowers the squared error.
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> p(1024);
  for (float& v : p) v = nd(rng);
  p[5] = 9.f;
  p[700] = -7.f;
  const auto err = [&](bool trim) {
    const auto [lo, hi] = feature_plane_range(p, 4, trim);
    double e = 0;
    for (const float v : p) {
      const float q = std::clamp(std::round((v - lo) / (hi - lo) * 15.f), 0.f, 15.f);
      e += std::pow(v - (lo + q / 15.f * (hi - lo)), 2.f);
    }
    return e;
  };
  EXPECT_LT(err(true), 0.8 * err(false));
  const auto [lo8, hi8] = feature_plane_range(p, 8, false);
  EXPECT_EQ(lo8, -7.f);
  EXPECT_EQ(hi8, 9.f);
}

TEST(Train, RateGradientMatchesFiniteDifferences) {
  // The rate estimate is piecewise smooth (min over predictors, |r|); check away from the kinks.
  Hyper h = tiny_grid();
  h.grid = 6;
  Model m = randomised(h, 13);
  std::vector<float> g(m.features.size(), 0.f);
  const double r0 = train::feature_rate(m, 5, g, 1.f);
  EXPECT_GT(r0, 0.0);
  int checked = 0, bad = 0;
  std::string first_bad;
  for (std::size_t i = 0; i < m.features.size(); i += 3) {
    const float keep = m.features[i];
    const float eps = 1e-4f * std::max(1.f, std::abs(keep));
    std::array<double, 2> n{};
    for (int s = 0; s < 2; ++s) {
      m.features[i] = keep + (s ? -eps : eps);
      n[static_cast<std::size_t>(s)] = train::feature_rate(m, 5);
    }
    m.features[i] = keep;
    const double num = (n[0] - n[1]) / (2.0 * static_cast<double>(eps));
    // A plane's min or max moves the step itself, which the estimate holds constant: skip those values.
    const std::size_t plane = static_cast<std::size_t>(h.grid) * h.grid, p0 = i / plane * plane;
    const auto [mn, mx] = std::minmax_element(m.features.begin() + static_cast<std::ptrdiff_t>(p0),
                                              m.features.begin() + static_cast<std::ptrdiff_t>(p0 + plane));
    if (keep == *mn || keep == *mx) continue;
    ++checked;
    if (std::abs(num - g[i]) > 2e-2 * std::max(std::abs(num), std::abs(static_cast<double>(g[i]))) + 1e-2) {
      if (++bad <= 5) first_bad += std::format(" feature {}: analytic {} numeric {};", i, g[i], num);
    }
  }
  EXPECT_GT(checked, 20);
  EXPECT_LE(bad, checked / 20) << "a few values sit on a kink of the predictor choice; more means a wrong gradient:" << first_bad;
}

TEST(Train, QuantisationAwareAndRateAwareTrainingWork) {
  const Clip clip = smooth_clip(32, 8, 0.f);
  Hyper h;
  h.arch = Arch::grid;
  h.size = 32;
  h.frames = 8;
  h.grid = 16;
  h.grid_t = 8;
  h.channels = 4;
  h.hidden = 16;
  h.layers = 1;
  train::Options o;
  o.iterations = 400;
  o.batch_frames = 4;
  o.pixels = 512;
  o.threads = 2;
  o.log_every = 0;
  o.qat_bits = 3;
  const train::Example ex{&clip, {}};
  const auto qat = train::train(h, std::span(&ex, 1), o);
  o.qat_bits = 0;
  const auto plain = train::train(h, std::span(&ex, 1), o);
  // At 3 bits, the model trained for them beats the float model rounded to them.
  const auto at_bits = [&](Model m) {
    m.feature_bits = 3;
    quantise_like_storage(m);
    return metrics::score(clip, train::render_clip(m, {}, {}, 8, 32)).psnr;
  };
  EXPECT_GT(at_bits(qat.model), at_bits(plain.model) + 1.0);
  EXPECT_GT(at_bits(qat.model), 26.0);
  // A rate term lowers the estimated bits per value.
  o.qat_bits = 4;
  const auto r0 = train::train(h, std::span(&ex, 1), o);
  o.rate_lambda = 3e-3f;
  o.rate_bits = 4;
  const auto r1 = train::train(h, std::span(&ex, 1), o);
  EXPECT_LT(train::feature_rate(r1.model, 4), 0.8 * train::feature_rate(r0.model, 4));
}
