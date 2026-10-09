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

TEST(Model, SaveLoadRoundTripsAtBothPrecisions) {
  for (const int bits : {16, 8}) {
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
