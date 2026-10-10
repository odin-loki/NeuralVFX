// The context-mixing coder: exact round trips on random and smooth tensors of many shapes, on frame models and
// rollout effects saved in memory, on empty, tiny and malformed inputs; smooth data must compress well; damaged
// packed data must be refused, not decoded into something else.
#include <neuralfx/cm.hpp>
#include <neuralfx/model.hpp>
#include <neuralfx/rollout.hpp>

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <stdfloat>

using namespace nfx;

namespace {

std::uint16_t f16(float v) { return std::bit_cast<std::uint16_t>(static_cast<std::float16_t>(v)); }

cm::Tensor random_tensor(cm::Kind kind, int width, std::vector<std::uint32_t> dims, bool channels, std::uint64_t seed) {
  cm::Tensor t;
  t.shape.kind = kind;
  t.shape.width = width;
  t.shape.dims = std::move(dims);
  t.shape.channels = channels;
  std::mt19937_64 rng(seed);
  t.values.resize(t.shape.size());
  for (auto& v : t.values) v = static_cast<std::uint16_t>(width == 1 ? rng() & 255 : rng() & 0xffff);
  return t;
}

// Packs, unpacks and compares; returns the packed size.
std::size_t round_trip(const std::vector<cm::Tensor>& in) {
  const cm::Packed p = cm::pack_tensors(in);
  const auto back = cm::unpack_tensors(p.data);
  EXPECT_TRUE(back) << (back ? "" : back.error());
  if (!back) return 0;
  EXPECT_EQ(back->size(), in.size());
  for (std::size_t k = 0; k < std::min(back->size(), in.size()); ++k) {
    EXPECT_EQ((*back)[k].shape.kind, in[k].shape.kind);
    EXPECT_EQ((*back)[k].shape.width, in[k].shape.width);
    EXPECT_EQ((*back)[k].shape.dims, in[k].shape.dims);
    EXPECT_EQ((*back)[k].shape.channels, in[k].shape.channels);
    EXPECT_EQ((*back)[k].shape.lo, in[k].shape.lo);
    EXPECT_EQ((*back)[k].shape.hi, in[k].shape.hi);
    EXPECT_TRUE((*back)[k].values == in[k].values) << "tensor " << k;
  }
  return p.data.size();
}

std::vector<std::uint8_t> bytes_of(const std::string& s) { return {s.begin(), s.end()}; }

void expect_model_round_trip(const std::vector<std::uint8_t>& file) {
  const cm::Packed p = cm::pack_model(file);
  const auto back = cm::unpack_model(p.data);
  ASSERT_TRUE(back) << back.error();
  EXPECT_TRUE(*back == file);
  std::size_t covered = 0;
  for (const auto& [k, b] : cm::split_model(file)) covered += b.size();
  EXPECT_EQ(covered, file.size());
}

}  // namespace

TEST(Cm, Fp16OrderIsAMonotoneBijection) {
  std::vector<bool> seen(65536, false);
  for (std::uint32_t b = 0; b < 65536; ++b) {
    const auto u = cm::f16_order(static_cast<std::uint16_t>(b));
    EXPECT_FALSE(seen[u]);
    seen[u] = true;
    EXPECT_EQ(cm::f16_unorder(u), b);
  }
  // Finite values in increasing order map to increasing codes.
  float prev = -std::numeric_limits<float>::infinity();
  std::uint16_t prev_code = 0;
  for (std::uint32_t u = 0; u < 65536; ++u) {
    const float v = static_cast<float>(std::bit_cast<std::float16_t>(cm::f16_unorder(static_cast<std::uint16_t>(u))));
    if (std::isnan(v)) continue;
    EXPECT_GE(v, prev) << "code " << u;
    if (v > prev) prev_code = static_cast<std::uint16_t>(u);
    prev = v;
  }
  EXPECT_GT(prev_code, 0x8000);
}

TEST(Cm, RandomTensorsOfManyShapesRoundTrip) {
  using K = cm::Kind;
  std::vector<cm::Tensor> ts;
  std::uint64_t seed = 1;
  for (const int width : {1, 2}) {
    ts.push_back(random_tensor(K::bytes, width, {1}, false, seed++));
    ts.push_back(random_tensor(K::weights, width, {37}, false, seed++));
    ts.push_back(random_tensor(K::weights, width, {5, 9}, false, seed++));
    ts.push_back(random_tensor(K::features, width, {3, 4, 7, 6}, false, seed++));
    ts.push_back(random_tensor(K::features, width, {2, 3, 2, 5, 5}, false, seed++));
    ts.push_back(random_tensor(K::coarse, width, {2, 6, 5, 4}, true, seed++));
    ts.push_back(random_tensor(K::rgba, width, {3, 1, 8, 4}, true, seed++));
    ts.push_back(random_tensor(K::flow, width, {7, 2}, true, seed++));
  }
  ts.push_back(random_tensor(K::bc3, 1, {3, 4, 4, 16}, true, seed++));
  // 8-bit planes with affine maps, including a degenerate plane (hi == lo) and a reversed one.
  cm::Tensor m = random_tensor(K::features, 1, {2, 3, 6, 6}, false, seed++);
  for (int p = 0; p < 6; ++p) {
    m.shape.lo.push_back(-(int64_t{1} << 20) * p);
    m.shape.hi.push_back(p == 2 ? m.shape.lo.back() : p == 4 ? m.shape.lo.back() - 5 : (int64_t{1} << 22) * (p + 1));
  }
  ts.push_back(m);
  round_trip(ts);
  for (const auto& t : ts) round_trip({t});  // each alone too (a fresh model per stream)
}

TEST(Cm, SmoothTensorsCompressWell) {
  // Eight planes of a smooth 8-bit field drifting over the planes.
  cm::Tensor a;
  a.shape.kind = cm::Kind::features;
  a.shape.dims = {8, 48, 48};
  for (int t = 0; t < 8; ++t) {
    for (int y = 0; y < 48; ++y) {
      for (int x = 0; x < 48; ++x) {
        const double v = 128 + 90 * std::sin(0.11 * x + 0.3 * t) * std::cos(0.07 * y - 0.2 * t);
        a.values.push_back(static_cast<std::uint16_t>(std::lround(v)));
      }
    }
  }
  const std::size_t na = round_trip({a});
  EXPECT_LT(static_cast<double>(na), static_cast<double>(a.values.size()) / 4.0) << "smooth 8-bit field: " << na << " bytes";
  // A smooth fp16 field with channels (like the coarse start states), crossing zero.
  cm::Tensor b;
  b.shape.kind = cm::Kind::coarse;
  b.shape.width = 2;
  b.shape.channels = true;
  b.shape.dims = {2, 32, 32, 4};
  for (int s = 0; s < 2; ++s) {
    for (int y = 0; y < 32; ++y) {
      for (int x = 0; x < 32; ++x) {
        for (int c = 0; c < 4; ++c) b.values.push_back(f16(static_cast<float>(std::sin(0.2 * x + c + s) * std::cos(0.15 * y) * (c + 1))));
      }
    }
  }
  const std::size_t nb = round_trip({b});
  EXPECT_LT(static_cast<double>(nb), 2.0 * static_cast<double>(b.values.size()) / 1.6) << "smooth fp16 field: " << nb << " bytes";
  // Constant data costs almost nothing.
  cm::Tensor z;
  z.shape.kind = cm::Kind::bc3;
  z.shape.channels = true;
  z.shape.dims = {16, 8, 8, 16};
  z.values.assign(z.shape.size(), 0);
  EXPECT_LT(round_trip({z}), 200u);
}

TEST(Cm, FrameModelsRoundTrip) {
  for (const Arch arch : {Arch::grid, Arch::conv}) {
    for (const int bits : {8, 16}) {
      Hyper h;
      h.arch = arch;
      h.size = 32;
      h.latent = 4;
      h.frames = 8;
      h.grid = 6;
      h.channels = 3;
      h.hidden = 5;
      h.c0 = 3;
      h.c1 = 4;
      h.c2 = 2;
      h.grid_t = 3;
      h.bases = 2;
      h.n_controls = 2;
      h.n_latent = 2;
      Model m = init_model(h, 7);
      m.feature_bits = bits;
      m.effect = "smoke";
      m.control_names = {"intensity", "wind"};
      m.z_train = {{0.5f, -1.f}, {0.25f, 2.f}, {0.f, 0.f}};
      m.z_mean = {0.1f, 0.2f};
      m.z_std = {1.f, 0.5f};
      std::ostringstream os;
      ASSERT_TRUE(save_model(os, m));
      const auto file = bytes_of(os.str());
      expect_model_round_trip(file);
      // It is parsed as a model: features, weights and codes are coded as tensors.
      const cm::Packed p = cm::pack_model(file);
      std::vector<cm::Kind> kinds;
      for (const auto& part : p.parts) kinds.push_back(part.kind);
      EXPECT_NE(std::ranges::find(kinds, cm::Kind::features), kinds.end());
      EXPECT_NE(std::ranges::find(kinds, cm::Kind::weights), kinds.end());
      EXPECT_NE(std::ranges::find(kinds, cm::Kind::codes), kinds.end());
      EXPECT_EQ(std::ranges::find(kinds, cm::Kind::ranges) != kinds.end(), bits == 8);
    }
  }
}

TEST(Cm, RolloutEffectsRoundTrip) {
  rollout::Hyper h;
  h.res = 8;
  h.hidden = 6;
  h.memory = 2;
  h.render_hidden = 5;
  h.start_fine = 16;
  h.n_age = 1;
  rollout::Model m = rollout::init_model(h, 3);
  m.effect = "explosion";
  m.control_names = {"intensity", "wind", "turbulence"};
  for (int k = 0; k < 3; ++k) {
    rollout::StartPoint sp;
    sp.controls = {0.1f * static_cast<float>(k), 0.5f, 0.9f};
    sp.seed = 40 + static_cast<std::uint64_t>(k);
    sp.time = 0.5f * static_cast<float>(k);
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        for (int c = 0; c < rollout::kPhys; ++c) sp.coarse.push_back(std::sin(0.4f * static_cast<float>(x + c) + static_cast<float>(k)) * std::cos(0.3f * static_cast<float>(y)));
      }
    }
    if (k != 1) {  // start points with and without fine fields
      for (int i = 0; i < 16 * 16; ++i) {
        sp.fine_t.push_back(std::max(0.f, std::sin(0.1f * static_cast<float>(i))));
        sp.fine_d.push_back(std::max(0.f, std::cos(0.13f * static_cast<float>(i))));
      }
    }
    m.starts.push_back(sp);
  }
  rollout::quantise_like_storage(m);
  std::ostringstream os;
  ASSERT_TRUE(rollout::save_model(os, m));
  const auto file = bytes_of(os.str());
  expect_model_round_trip(file);
  const cm::Packed p = cm::pack_model(file);
  std::vector<cm::Kind> kinds;
  for (const auto& part : p.parts) kinds.push_back(part.kind);
  for (const cm::Kind k : {cm::Kind::weights, cm::Kind::biases, cm::Kind::coarse, cm::Kind::fine, cm::Kind::scales}) {
    EXPECT_NE(std::ranges::find(kinds, k), kinds.end()) << cm::kind_name(k);
  }
}

TEST(Cm, EmptyTinyAndMalformedInputsRoundTrip) {
  expect_model_round_trip({});
  expect_model_round_trip({42});
  expect_model_round_trip(bytes_of("NVFXMDL1"));
  expect_model_round_trip(bytes_of("NVFXROL1 but not really a rollout effect"));
  // A model with its tail cut off is no model: coded as plain bytes, still exact.
  Model m = init_model(Hyper{.size = 16, .frames = 4, .grid_t = 2, .grid = 4, .channels = 2, .hidden = 3, .layers = 1}, 1);
  std::ostringstream os;
  ASSERT_TRUE(save_model(os, m));
  auto file = bytes_of(os.str());
  expect_model_round_trip(file);
  file.resize(file.size() - 3);
  expect_model_round_trip(file);
  file.resize(file.size() + 10, 7);  // and with trailing bytes
  expect_model_round_trip(file);
  // Tensors: none, and empty ones.
  EXPECT_GT(round_trip({}), 0u);
  cm::Tensor e;
  e.shape.dims = {};
  cm::Tensor e2;
  e2.shape.dims = {4, 0, 3};
  e2.shape.width = 2;
  round_trip({e, e2, random_tensor(cm::Kind::bytes, 1, {1}, false, 9)});
}

TEST(Cm, DamagedDataIsRefused) {
  EXPECT_FALSE(cm::unpack_model({}));
  EXPECT_FALSE(cm::unpack_model(bytes_of("NVFZ but nothing else of use here")));
  Model m = init_model(Hyper{.size = 16, .frames = 4, .grid_t = 2, .grid = 8, .channels = 2, .hidden = 3, .layers = 1}, 2);
  m.feature_bits = 8;
  std::ostringstream os;
  ASSERT_TRUE(save_model(os, m));
  const auto file = bytes_of(os.str());
  const cm::Packed p = cm::pack_model(file);
  // Damage anywhere is refused, or (late in the stream, where the last bits may not depend on it) gives the exact
  // original: never a different file.
  int refused = 0;
  for (std::size_t at = 0; at < p.data.size(); at += 7) {
    auto bad = p.data;
    bad[at] ^= 0x5a;
    const auto r = cm::unpack_model(bad);
    EXPECT_TRUE(!r || *r == file) << "damage at " << at << " gave another file";
    refused += r ? 0 : 1;
  }
  EXPECT_GT(refused, static_cast<int>(p.data.size() / 7) * 3 / 4);
  auto cut = p.data;
  cut.resize(cut.size() / 2);
  EXPECT_FALSE(cm::unpack_model(cut));
  // Random payloads behind a valid header, for every kind of content, and a header claiming a huge size: refused,
  // whatever shapes the decoder makes of them.
  std::mt19937_64 rng(11);
  for (std::uint8_t mode = 0; mode < 4; ++mode) {
    for (int trial = 0; trial < 6; ++trial) {
      std::vector<std::uint8_t> junk(p.data.begin(), p.data.begin() + 22);
      junk[5] = mode;
      junk[6] = static_cast<std::uint8_t>(rng());
      junk[7] = static_cast<std::uint8_t>(rng() & 7);
      for (int k = 8; k < 14; ++k) junk[static_cast<std::size_t>(k)] = 0;
      for (int k = 0; k < 400; ++k) junk.push_back(static_cast<std::uint8_t>(rng()));
      EXPECT_FALSE(mode == 3 ? cm::unpack_tensors(junk).has_value() : cm::unpack_model(junk).has_value());
    }
  }
  auto huge = p.data;
  huge[13] = 0x40;
  EXPECT_FALSE(cm::unpack_model(huge));
  // A model stream is not a tensor stream and the other way round.
  EXPECT_FALSE(cm::unpack_tensors(p.data));
  EXPECT_FALSE(cm::unpack_model(cm::pack_tensors({}).data));
  // Bad tensors are refused when packing.
  cm::Tensor t;
  t.shape.dims = {2};
  t.values = {1, 300};
  EXPECT_THROW(cm::pack_tensors(std::span(&t, 1)), std::invalid_argument);
  t.values = {1};
  EXPECT_THROW(cm::pack_tensors(std::span(&t, 1)), std::invalid_argument);
}

TEST(Cm, PackingIsDeterministic) {
  Model m = init_model(Hyper{.size = 16, .frames = 4, .grid_t = 2, .grid = 8, .channels = 2, .hidden = 3, .layers = 2}, 5);
  std::ostringstream os;
  ASSERT_TRUE(save_model(os, m));
  const auto file = bytes_of(os.str());
  EXPECT_EQ(cm::pack_model(file).data, cm::pack_model(file).data);
}
