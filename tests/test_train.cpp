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

// The multi family (study F4): a still level, two time-varying ones, and Fourier features of position and time.
Hyper tiny_multi(bool loop) {
  Hyper h;
  h.arch = Arch::multi;
  h.size = 8;
  h.frames = 4;
  h.loop = loop;
  h.n_controls = 2;
  h.n_latent = 1;
  h.bases = 2;
  h.levels = {{4, 3, 2}, {3, 1, 2}, {6, 4, 1}};
  h.pe_xy = 2;
  h.pe_t = 1;
  h.hidden = 5;
  h.layers = 2;
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
  // Central differences at four step sizes. Float round-off spoils small steps and relu kinks spoil large ones, so a
  // parameter passes when the analytic gradient matches at one of them; a wrong gradient is wrong at every step and
  // fails. Where the estimates scatter by more than 10% without a match, the parameter sits on a kink and is skipped
  // (at most a tenth may be). (The smallest step was added for the multi family, whose test model has a unit within
  // about 1e-3 of its kink at one grid point: there the larger steps are 1 to 3% off and 3e-4 matches.)
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
    const std::array<double, 4> n = {numeric(slot, 1e-2f, eval), numeric(slot, 3e-3f, eval), numeric(slot, 1e-3f, eval), numeric(slot, 3e-4f, eval)};
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
    if (++bad <= 6) ADD_FAILURE() << std::format("{}: analytic {} numeric {} {} {} {}", what, ana, n[0], n[1], n[2], n[3]);
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

TEST(Model, MixedPrecisionFilesRoundTrip) {
  // Every plane at its own width (study F3, file version 3), 0 to 8 bits, for both families and odd plane sizes.
  Hyper odd = tiny_grid();
  odd.grid = 5;  // planes of 25 values: packed planes end inside a byte
  for (const Hyper& h : {tiny_grid(), odd, tiny_conv()}) {
    Model m = randomised(h, 9);
    m.effect = "smoke";
    m.feature_bits = 8;
    m.z_train = {{0.1f}, {-0.2f}};
    m.z_mean = {-0.05f};
    m.z_std = {0.15f};
    const std::size_t side2 = static_cast<std::size_t>(h.feature_side()) * h.feature_side();
    m.plane_bits.resize(m.features.size() / side2);
    for (std::size_t k = 0; k < m.plane_bits.size(); ++k) m.plane_bits[k] = static_cast<std::uint8_t>((k * 7 + 3) % 9);
    std::stringstream ss;
    ASSERT_TRUE(save_model(ss, m).has_value());
    const std::string bytes = ss.str();
    const auto back = load_model(ss);
    ASSERT_TRUE(back.has_value()) << back.error();
    Model q = m;
    quantise_like_storage(q);
    EXPECT_EQ(back->plane_bits, m.plane_bits);
    EXPECT_EQ(back->features, q.features) << h.describe();
    EXPECT_EQ(back->storage_bytes(), m.storage_bytes());
    EXPECT_LE(m.storage_bytes(), bytes.size());
    EXPECT_LT(bytes.size() - m.storage_bytes(), 256u);
    // A plane at 0 bits is its mean; saving the loaded model again gives the same bytes.
    for (std::size_t k = 0; k < m.plane_bits.size(); ++k) {
      if (m.plane_bits[k] != 0) continue;
      const float* p = back->features.data() + k * side2;
      EXPECT_TRUE(std::all_of(p, p + side2, [&](float v) { return v == p[0]; }));
    }
    std::stringstream again;
    ASSERT_TRUE(save_model(again, *back).has_value());
    EXPECT_EQ(again.str(), bytes);
    // Widths above 8, a wrong count, or a damaged width byte are refused.
    Model bad = m;
    bad.plane_bits[1] = 9;
    std::stringstream sb;
    EXPECT_FALSE(save_model(sb, bad).has_value());
    bad.plane_bits.pop_back();
    EXPECT_FALSE(save_model(sb, bad).has_value());
    std::string damaged = bytes;
    const std::size_t at = 8 + 4 + 4 + 15 * 4 + 32 + 4 + 4 + 16 * static_cast<std::size_t>(h.n_controls);  // the first width byte
    damaged[at] = static_cast<char>(12);
    std::istringstream sd(damaged);
    EXPECT_FALSE(load_model(sd).has_value());
  }
}

TEST(Train, MixedPrecisionAllocationAndTraining) {
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
  const auto plain = train::train(h, std::span(&ex, 1), o);
  // The allocation spends the budget within the range, and its summed distortion is no worse than every plane at the
  // average width (the uniform allocation is one the search could have made).
  std::vector<double> D;
  const auto bits = train::allocate_plane_bits(plain.model, std::span(&ex, 1), {}, 3.0, 1, 6, false, 2, 0, &D);
  const std::size_t planes = plain.model.features.size() / 256;
  ASSERT_EQ(bits.size(), planes);
  ASSERT_EQ(D.size(), planes * 6);
  long total = 0;
  double mixed = 0, uniform = 0;
  for (std::size_t p = 0; p < planes; ++p) {
    EXPECT_GE(bits[p], 1);
    EXPECT_LE(bits[p], 6);
    total += bits[p];
    mixed += D[p * 6 + bits[p] - 1];
    uniform += D[p * 6 + 2];
    EXPECT_GE(D[p * 6], D[p * 6 + 5]);  // one bit distorts more than six
  }
  EXPECT_LE(total, static_cast<long>(3 * planes));
  EXPECT_LE(mixed, uniform);
  // A plane the frames never see (a constant feature channel is still seen, so: zero weights on channel 0 of the
  // first layer) gets the fewest bits.
  Model blind = plain.model;
  for (int o2 = 0; o2 < blind.layers[0].out; ++o2) blind.layers[0].w[static_cast<std::size_t>(o2) * blind.layers[0].in] = 0.f;
  const auto bits2 = train::allocate_plane_bits(blind, std::span(&ex, 1), {}, 3.0, 0, 8, false, 2);
  for (std::size_t p = 0; p < planes; p += 4) EXPECT_EQ(bits2[p], 0) << p;  // channel 0 of every slice
  // Mixed-precision QAT: the result carries its widths, the saved file renders what the trainer saw, and it beats the
  // float model quantised to the same widths afterwards.
  o.mixed_bits = 3.f;
  o.mixed_min = 1;
  o.mixed_max = 6;
  o.qat_start = 0.5f;
  const auto r = train::train(h, std::span(&ex, 1), o);
  ASSERT_EQ(r.model.plane_bits.size(), planes);
  std::stringstream ss;
  ASSERT_TRUE(save_model(ss, r.model).has_value());
  const auto back = load_model(ss);
  ASSERT_TRUE(back.has_value()) << back.error();
  Model seen = r.model;
  train::fake_quantise(seen, seen.plane_bits);
  EXPECT_EQ(back->features, seen.features);
  const double qat_psnr = metrics::score(clip, train::render_clip(*back, {}, {}, 8, 32)).psnr;
  Model ptq = plain.model;
  ptq.plane_bits = r.model.plane_bits;
  quantise_like_storage(ptq);
  EXPECT_GT(qat_psnr, metrics::score(clip, train::render_clip(ptq, {}, {}, 8, 32)).psnr);
  EXPECT_GT(qat_psnr, 26.0);
  EXPECT_THROW(train::train(h, std::span(&ex, 1), [&] {
                 train::Options bad = o;
                 bad.qat_bits = 4;
                 return bad;
               }()),
               std::invalid_argument);
}

TEST(Train, SparseFeaturesKeepWhatTheFramesNeed) {
  // A clip that is empty but for a square that moves right: each slice's mask holds the grid points under the square
  // in the frames that blend it (and their bilinear neighbours), nothing far from it.
  Clip clip;
  clip.allocate(32, 8);
  clip.loop = true;
  for (int f = 0; f < 8; ++f) {
    auto fr = clip.frame(f);
    for (int y = 12; y < 18; ++y) {
      for (int x = 2 + 3 * f; x < 8 + 3 * f; ++x) {
        const std::size_t i = (static_cast<std::size_t>(y) * 32 + x) * 4;
        fr[i] = 200;
        fr[i + 1] = 120;
        fr[i + 2] = 40;
        fr[i + 3] = 220;
      }
    }
  }
  Hyper h;
  h.arch = Arch::grid;
  h.size = 32;
  h.frames = 8;
  h.grid = 16;
  h.grid_t = 8;
  h.channels = 4;
  h.hidden = 16;
  h.layers = 1;
  const train::Example ex{&clip, {}};
  const auto mask = train::feature_support(h, std::span(&ex, 1), 0);
  ASSERT_EQ(mask.size(), 8u * 256u);
  // Slice 0 is blended by frame 0 only (8 frames, 8 slices, looping): the square covers pixels 2..7 x 12..17, sampled
  // by grid points 0..4 x 5..9 (pixel x samples point floor(x / 2 - 0.25) and the next, both with weight above zero).
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      EXPECT_EQ(mask[static_cast<std::size_t>(y) * 16 + x], (x <= 4 && y >= 5 && y <= 9) ? 1 : 0) << x << " " << y;
    }
  }
  const auto grown = train::feature_support(h, std::span(&ex, 1), 0, 1);
  EXPECT_EQ(grown[4 * 16 + 4], 1);  // one point further
  EXPECT_EQ(grown[2 * 16 + 4], 0);
  // Sparse 4-bit training: the model keeps the mask, stores less than the dense model at the same bits, the file
  // renders what the trainer saw, and it learns the square.
  train::Options o;
  o.iterations = 400;
  o.batch_frames = 4;
  o.pixels = 512;
  o.threads = 2;
  o.log_every = 0;
  o.qat_bits = 4;
  o.sparse = true;
  const auto r = train::train(h, std::span(&ex, 1), o);
  EXPECT_EQ(r.model.feature_mask, mask);
  ASSERT_EQ(r.model.plane_bits.size(), 32u);
  Model m = r.model;
  m.feature_bits = 4;
  Model dense = m;
  dense.plane_bits.clear();
  dense.feature_mask.clear();
  EXPECT_LT(m.storage_bytes(), dense.storage_bytes() * 6 / 10);
  std::stringstream ss;
  ASSERT_TRUE(save_model(ss, m).has_value());
  const auto back = load_model(ss);
  ASSERT_TRUE(back.has_value()) << back.error();
  Model seen = r.model;
  train::fake_quantise(seen, seen.plane_bits);
  EXPECT_EQ(back->features, seen.features);
  o.sparse = false;
  const auto d = train::train(h, std::span(&ex, 1), o);
  Model dq = d.model;
  dq.feature_bits = 4;
  quantise_like_storage(dq);
  const double sparse_psnr = metrics::score(clip, train::render_clip(*back, {}, {}, 8, 32)).active_psnr;
  EXPECT_GT(sparse_psnr, 20.0);
  EXPECT_GT(sparse_psnr, metrics::score(clip, train::render_clip(dq, {}, {}, 8, 32)).active_psnr - 2.0);
}

TEST(Train, ForwardMatchesTheReference) {
  for (const Hyper& h : {tiny_grid(), tiny_conv(), tiny_multi(true), tiny_multi(false)}) {
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
TEST(Train, MultiLevelGradientsMatchFiniteDifferences) {
  gradient_check(tiny_multi(true));
  gradient_check(tiny_multi(false));
}

TEST(Model, MaskRunsRoundTripAndRefuseMalformedInput) {
  std::mt19937_64 rng(7);
  for (const std::size_t n : {1u, 2u, 7u, 200u, 4096u}) {
    for (int pattern = 0; pattern < 4; ++pattern) {
      std::vector<std::uint8_t> mask(n);
      for (std::size_t j = 0; j < n; ++j) mask[j] = pattern == 0 ? 0 : pattern == 1 ? 1 : pattern == 2 ? static_cast<std::uint8_t>(rng() % 2) : static_cast<std::uint8_t>(j > n / 3);
      const auto runs = mask_runs(mask);
      std::vector<std::uint8_t> back(n, 9);
      ASSERT_EQ(read_mask_runs(runs, back), runs.size()) << n << " " << pattern;
      EXPECT_EQ(back, mask);
      if (runs.size() > 1) {
        std::vector<std::uint8_t> cut(runs.begin(), runs.end() - 1);
        EXPECT_EQ(read_mask_runs(cut, back), 0u) << "truncated runs";
      }
    }
  }
  std::vector<std::uint8_t> m4(4);
  EXPECT_EQ(read_mask_runs(std::vector<std::uint8_t>{5}, m4), 0u);        // longer than the mask
  EXPECT_EQ(read_mask_runs(std::vector<std::uint8_t>{1, 0, 3}, m4), 0u);  // an empty run after the first
  EXPECT_EQ(read_mask_runs(std::vector<std::uint8_t>{0, 4}, m4), 2u);     // starting with stored points
  EXPECT_EQ(m4, (std::vector<std::uint8_t>{1, 1, 1, 1}));
  EXPECT_EQ(mask_runs(std::vector<std::uint8_t>(300, 0)), (std::vector<std::uint8_t>{0xac, 0x02}));  // 300 as a varint
}

TEST(Model, MultiLevelFilesRoundTrip) {
  // File version 4: levels, a width per level, masks as runs, the MLP at 8 or 16 bits. Saving what was loaded gives the
  // same bytes; the loaded model computes what quantise_like_storage computes; damaged files are refused.
  for (const bool masked : {false, true}) {
    for (const int mlp_bits : {16, 8}) {
      Model m = randomised(tiny_multi(false), 4);
      m.mlp_bits = mlp_bits;
      const auto vols = volumes(m.h);
      for (std::size_t l = 0; l < vols.size(); ++l) {
        const std::size_t n = static_cast<std::size_t>(m.h.bases) * vols[l].slices * vols[l].channels;
        for (std::size_t k = 0; k < n; ++k) m.plane_bits.push_back(static_cast<std::uint8_t>(l == 0 ? 3 : l == 1 ? 8 : 5));
      }
      if (masked) {
        m.feature_mask.resize(vols.back().mask0 + static_cast<std::size_t>(vols.back().slices) * vols.back().plane_values());
        for (std::size_t j = 0; j < m.feature_mask.size(); ++j) m.feature_mask[j] = static_cast<std::uint8_t>((j * 5 / 3) % 2);
      }
      std::stringstream ss;
      ASSERT_TRUE(save_model(ss, m).has_value());
      const std::string bytes = ss.str();
      std::stringstream in(bytes);
      const auto back = load_model(in);
      ASSERT_TRUE(back.has_value()) << back.error();
      Model q = m;
      quantise_like_storage(q);
      EXPECT_EQ(back->features, q.features);
      for (std::size_t l = 0; l < q.layers.size(); ++l) EXPECT_EQ(back->layers[l].w, q.layers[l].w) << "layer " << l;
      EXPECT_EQ(back->plane_bits, m.plane_bits);
      EXPECT_EQ(back->feature_mask, m.feature_mask);
      EXPECT_EQ(back->h.levels, m.h.levels);
      EXPECT_EQ(back->storage_bytes(), m.storage_bytes());
      std::stringstream again;
      ASSERT_TRUE(save_model(again, *back).has_value());
      EXPECT_EQ(again.str(), bytes);
      for (const std::size_t cut : {bytes.size() - 1, bytes.size() / 2, std::size_t{80}}) {
        std::stringstream part(bytes.substr(0, cut));
        EXPECT_FALSE(load_model(part).has_value()) << "truncated at " << cut;
      }
      std::string bad = bytes;
      bad[12] = 1;  // the grid family's code in a version 4 file
      std::stringstream bs(bad);
      EXPECT_FALSE(load_model(bs).has_value());
      if (mlp_bits == 8) {
        Model wide = m;
        wide.mlp_bits = 16;
        EXPECT_LT(m.storage_bytes(), wide.storage_bytes());
      }
    }
  }
  Model bad = randomised(tiny_multi(true), 5);
  bad.plane_bits.assign(3, 4);  // not one width per plane
  std::stringstream ss;
  EXPECT_FALSE(save_model(ss, bad).has_value());
}

TEST(Train, MultiLevelSparseQuantisedTrainingWorks) {
  // Two levels at their own widths, sparse, the MLP trained for 8-bit weights: the model keeps one mask per level, its
  // file renders what the trainer saw, and it learns the clip.
  const Clip clip = smooth_clip(32, 8, 0.f);
  Clip holes = clip;  // empty corners, so that the masks drop points
  for (int f = 0; f < 8; ++f) {
    auto fr = holes.frame(f);
    for (int y = 0; y < 32; ++y) {
      for (int x = 0; x < 32; ++x) {
        if (x + y < 14 || x + y > 50) std::fill_n(fr.begin() + (static_cast<std::ptrdiff_t>(y) * 32 + x) * 4, 4, std::uint8_t{0});
      }
    }
  }
  Hyper h;
  h.arch = Arch::multi;
  h.size = 32;
  h.frames = 8;
  h.levels = {{8, 4, 4}, {16, 8, 2}};
  h.hidden = 16;
  h.layers = 2;
  h.pe_xy = 1;
  const train::Example ex{&holes, {}};
  train::Options o;
  o.iterations = 500;
  o.batch_frames = 4;
  o.pixels = 512;
  o.threads = 2;
  o.log_every = 0;
  o.qat_bits = 4;
  o.level_bits = {6, 3};
  o.mlp_qat = true;
  o.sparse = true;
  const auto r = train::train(h, std::span(&ex, 1), o);
  EXPECT_EQ(r.model.feature_mask, train::feature_support(h, std::span(&ex, 1), 0));
  ASSERT_EQ(r.model.plane_bits.size(), 4u * 4u + 8u * 2u);
  EXPECT_EQ(r.model.plane_bits.front(), 6);
  EXPECT_EQ(r.model.plane_bits.back(), 3);
  Model m = r.model;
  m.mlp_bits = 8;
  std::stringstream ss;
  ASSERT_TRUE(save_model(ss, m).has_value());
  const auto back = load_model(ss);
  ASSERT_TRUE(back.has_value()) << back.error();
  Model seen = r.model;
  train::store_planes(seen, seen.plane_bits);
  EXPECT_EQ(back->features, seen.features);
  const double psnr = metrics::score(holes, train::render_clip(*back, {}, {}, 8, 32, 2)).active_psnr;
  EXPECT_GT(psnr, 24.0);
}

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

TEST(Train, VectorQuantisedTrainingWorks) {
  // Codebooks fitted halfway, then trained through: the saved model renders what the trainer saw, and at one bit per
  // feature value (8-bit indices for 8 channels) it still learns the clip.
  const Clip clip = smooth_clip(32, 8, 0.f);
  Hyper h;
  h.arch = Arch::grid;
  h.size = 32;
  h.frames = 8;
  h.grid = 16;
  h.grid_t = 8;
  h.channels = 8;
  h.hidden = 16;
  h.layers = 1;
  train::Options o;
  o.iterations = 400;
  o.batch_frames = 4;
  o.pixels = 512;
  o.threads = 2;
  o.log_every = 0;
  o.vq_bits = 8;
  o.vq_dim = 8;
  const train::Example ex{&clip, {}};
  const auto r = train::train(h, std::span(&ex, 1), o);
  ASSERT_EQ(r.model.vq_codebook.size(), 256u * 8u);
  Model m = r.model;
  m.feature_bits = 8;
  std::stringstream ss;
  ASSERT_TRUE(save_model(ss, m));
  const auto back = load_model(ss);
  ASSERT_TRUE(back.has_value()) << back.error();
  Model q = m;
  quantise_like_storage(q);
  EXPECT_EQ(back->features, q.features);
  EXPECT_LT(back->storage_bytes(), 16u * 16u * 8u + 256u * 8u * 2u + 2048u);
  EXPECT_GT(metrics::score(clip, train::render_clip(*back, {}, {}, 8, 32)).psnr, 26.0);
  EXPECT_THROW(train::train(h, std::span(&ex, 1), [&] {
                 train::Options bad = o;
                 bad.vq_dim = 3;
                 return bad;
               }()),
               std::invalid_argument);
}
