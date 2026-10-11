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
    for (const int bits : {8, 16, 6, 5, 4, 3}) {
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
      // It is parsed as a model: features, weights and codes are coded as tensors. (Bit-packed planes that end
      // inside a byte are not parsed; the file is coded as plain bytes, still exactly.)
      const int side = arch == Arch::grid ? h.grid : h.latent;
      if (bits < 8 && side * side * bits % 8 != 0) continue;
      const cm::Packed p = cm::pack_model(file);
      std::vector<cm::Kind> kinds;
      for (const auto& part : p.parts) kinds.push_back(part.kind);
      EXPECT_NE(std::ranges::find(kinds, cm::Kind::features), kinds.end()) << bits;
      EXPECT_NE(std::ranges::find(kinds, cm::Kind::weights), kinds.end());
      EXPECT_NE(std::ranges::find(kinds, cm::Kind::codes), kinds.end());
      EXPECT_EQ(std::ranges::find(kinds, cm::Kind::ranges) != kinds.end(), bits < 16);
    }
  }
}

TEST(Cm, PackedFeaturesCodeOnlyTheirBits) {
  // A 4-bit model is parsed (its features are coded as 4-bit values: under 4 bits each), and a model whose packed
  // planes end inside a byte is still restored exactly (coded as plain bytes).
  for (const int grid : {16, 5}) {
    Hyper h;
    h.arch = Arch::grid;
    h.size = 32;
    h.frames = 8;
    h.grid = grid;
    h.channels = 4;
    h.hidden = 6;
    h.grid_t = 4;
    Model m = init_model(h, 9);
    m.feature_bits = grid == 16 ? 4 : 5;
    m.effect = "fire";
    std::ostringstream os;
    ASSERT_TRUE(save_model(os, m));
    const auto file = bytes_of(os.str());
    expect_model_round_trip(file);
    const cm::Packed p = cm::pack_model(file);
    const auto feat = std::ranges::find(p.parts, cm::Kind::features, &cm::Part::kind);
    if (grid == 16) {
      ASSERT_NE(feat, p.parts.end());
      EXPECT_EQ(feat->bytes, m.features.size() / 2);
      EXPECT_LT(8.0 * feat->coded_bytes / static_cast<double>(feat->values), 4.5);
    } else {
      EXPECT_EQ(feat, p.parts.end());
    }
  }
}

TEST(Cm, MixedPrecisionModelsRoundTrip) {
  // Study F3's file version 3 (a width per feature plane, 0 to 8 bits) is parsed: ranges as one tensor, the codes of
  // each width as their own tensor with only their bits coded, planes at 0 bits as their range only. In every format.
  // Planes that end inside a byte are not parsed (plain bytes), and are still restored exactly.
  for (const int grid : {16, 5}) {
    Hyper h;
    h.arch = Arch::grid;
    h.size = 32;
    h.frames = 8;
    h.grid = grid;
    h.channels = 4;
    h.hidden = 6;
    h.grid_t = 5;
    h.bases = 2;
    Model m = init_model(h, 13);
    m.feature_bits = 8;
    m.effect = "explosion";
    const std::size_t side2 = static_cast<std::size_t>(grid) * grid;
    m.plane_bits.resize(m.features.size() / side2);
    std::size_t coded = 0;
    for (std::size_t k = 0; k < m.plane_bits.size(); ++k) {
      m.plane_bits[k] = static_cast<std::uint8_t>((k * 4 + 1) % 9);
      if (m.plane_bits[k] > 0) coded += side2;
    }
    std::ostringstream os;
    ASSERT_TRUE(save_model(os, m));
    const auto file = bytes_of(os.str());
    expect_model_round_trip(file);
    for (const cm::Options o : {cm::Options{}, cm::Options{cm::Literal::fast, true, false}, cm::Options{cm::Literal::light, true, true, 512}}) {
      const auto back = cm::unpack_model(cm::pack_model(file, o).data);
      ASSERT_TRUE(back) << back.error();
      EXPECT_TRUE(*back == file);
    }
    const cm::Packed p = cm::pack_model(file);
    const auto feat = std::ranges::find(p.parts, cm::Kind::features, &cm::Part::kind);
    if (grid == 16) {
      ASSERT_NE(feat, p.parts.end());
      EXPECT_EQ(feat->values, coded);
      EXPECT_NE(std::ranges::find(p.parts, cm::Kind::ranges, &cm::Part::kind), p.parts.end());
    } else {
      EXPECT_EQ(feat, p.parts.end());
    }
    // With a mask: the stored points of each width one after another (runs of any length, padded to whole bytes),
    // fills as a tensor of their own. Parsed for both plane sizes.
    m.feature_mask.resize(static_cast<std::size_t>(h.grid_t) * side2);
    std::size_t stored = 0;
    for (std::size_t j = 0; j < m.feature_mask.size(); ++j) m.feature_mask[j] = static_cast<std::uint8_t>((j * j + j / 3) % 3 == 0);
    for (std::size_t k = 0; k < m.plane_bits.size(); ++k) {
      const std::size_t t = (k / static_cast<std::size_t>(h.channels)) % static_cast<std::size_t>(h.grid_t);
      if (m.plane_bits[k] > 0) stored += static_cast<std::size_t>(std::count(m.feature_mask.begin() + static_cast<std::ptrdiff_t>(t * side2), m.feature_mask.begin() + static_cast<std::ptrdiff_t>((t + 1) * side2), std::uint8_t{1}));
    }
    std::ostringstream ms;
    ASSERT_TRUE(save_model(ms, m));
    const auto masked = bytes_of(ms.str());
    expect_model_round_trip(masked);
    for (const cm::Options o : {cm::Options{}, cm::Options{cm::Literal::fast, true, false}, cm::Options{cm::Literal::light, true, true, 512}}) {
      const auto back = cm::unpack_model(cm::pack_model(masked, o).data);
      ASSERT_TRUE(back) << back.error();
      EXPECT_TRUE(*back == masked);
    }
    const cm::Packed pm = cm::pack_model(masked);
    const auto mf = std::ranges::find(pm.parts, cm::Kind::features, &cm::Part::kind);
    ASSERT_NE(mf, pm.parts.end()) << grid;
    EXPECT_EQ(mf->values, stored);
  }
}

TEST(Cm, MultiLevelModelsRoundTrip) {
  // Study F4's file version 4 is parsed: per level the ranges (and fills) as tensors and the codes at the level's width
  // (whole planes when they fill whole bytes, else and when masked one plane after another), the mask runs as plain
  // bytes, 8-bit MLP weights as a tensor with their scales. In every format; a damaged stream is refused.
  for (const bool masked : {false, true}) {
    for (const int mlp_bits : {16, 8}) {
      Hyper h;
      h.arch = Arch::multi;
      h.size = 32;
      h.frames = 8;
      h.bases = 2;
      h.n_latent = 1;
      h.levels = {{16, 4, 3}, {5, 1, 2}, {8, 6, 2}};  // 5 x 5 planes at 3 bits end inside a byte
      h.pe_xy = 1;
      h.pe_t = 1;
      h.hidden = 6;
      Model m = init_model(h, 17);
      m.effect = "smoke";
      m.mlp_bits = mlp_bits;
      const auto vols = volumes(h);
      std::size_t values = 0;
      for (std::size_t l = 0; l < vols.size(); ++l) {
        const std::size_t n = static_cast<std::size_t>(h.bases) * vols[l].slices * vols[l].channels;
        for (std::size_t k = 0; k < n; ++k) m.plane_bits.push_back(static_cast<std::uint8_t>(l == 0 ? 4 : l == 1 ? 3 : 8));
        values += n * vols[l].plane_values();
      }
      if (masked) {
        m.feature_mask.resize(vols.back().mask0 + static_cast<std::size_t>(vols.back().slices) * vols.back().plane_values());
        values = 0;
        for (std::size_t j = 0; j < m.feature_mask.size(); ++j) m.feature_mask[j] = static_cast<std::uint8_t>((j * j + j / 3) % 3 != 0);
        for (const Volume& v : vols) {
          for (int t = 0; t < v.slices; ++t) {
            const auto first = m.feature_mask.begin() + static_cast<std::ptrdiff_t>(v.mask0 + static_cast<std::size_t>(t) * v.plane_values());
            values += static_cast<std::size_t>(h.bases) * v.channels * static_cast<std::size_t>(std::count(first, first + static_cast<std::ptrdiff_t>(v.plane_values()), std::uint8_t{1}));
          }
        }
      }
      m.z_train = {{0.3f}, {-0.2f}};
      m.z_mean = {0.05f};
      m.z_std = {0.25f};
      std::ostringstream os;
      ASSERT_TRUE(save_model(os, m));
      const auto file = bytes_of(os.str());
      expect_model_round_trip(file);
      for (const cm::Options o : {cm::Options{}, cm::Options{cm::Literal::fast, true, false}, cm::Options{cm::Literal::light, true, true, 512}}) {
        const auto back = cm::unpack_model(cm::pack_model(file, o).data);
        ASSERT_TRUE(back) << back.error();
        EXPECT_TRUE(*back == file);
      }
      const cm::Packed p = cm::pack_model(file);
      const auto feat = std::ranges::find(p.parts, cm::Kind::features, &cm::Part::kind);
      ASSERT_NE(feat, p.parts.end()) << "masked " << masked << " mlp " << mlp_bits;
      EXPECT_EQ(feat->values, values);
      const auto w = std::ranges::find(p.parts, cm::Kind::weights, &cm::Part::kind);
      ASSERT_NE(w, p.parts.end());
      int refused = 0;
      for (std::size_t at = 30; at < p.data.size(); at += std::max<std::size_t>(11, p.data.size() / 10)) {
        auto bad = p.data;
        bad[at] ^= 0x3c;
        const auto r = cm::unpack_model(bad);
        EXPECT_TRUE(!r || *r == file) << "damage at " << at << " gave another file";
        refused += r ? 0 : 1;
      }
      EXPECT_GT(refused, 0);
    }
  }
}

TEST(Cm, VectorQuantisedModelsRoundTrip) {
  // Codebooks and index planes are parsed: the indices are coded with their bits, the codebooks as fp16 weights.
  for (const int bits : {8, 5, 3}) {
    Hyper h;
    h.arch = Arch::grid;
    h.size = 32;
    h.frames = 8;
    h.grid = 16;
    h.channels = 4;
    h.hidden = 6;
    h.grid_t = 4;
    Model m = init_model(h, 9);
    m.effect = "fire";
    m.feature_bits = 8;
    m.vq_bits = bits;
    m.vq_dim = 2;
    m.vq_codebook.resize(static_cast<std::size_t>(m.vq_groups()) * (std::size_t{1} << bits) * 2);
    for (std::size_t i = 0; i < m.vq_codebook.size(); ++i) m.vq_codebook[i] = 0.1f * std::sin(0.37f * static_cast<float>(i));
    std::ostringstream os;
    ASSERT_TRUE(save_model(os, m));
    const auto file = bytes_of(os.str());
    expect_model_round_trip(file);
    const cm::Packed p = cm::pack_model(file);
    const auto feat = std::ranges::find(p.parts, cm::Kind::features, &cm::Part::kind);
    ASSERT_NE(feat, p.parts.end()) << bits;
    EXPECT_EQ(feat->values, m.features.size() / 2);
    EXPECT_LT(8.0 * feat->coded_bytes / static_cast<double>(feat->values), bits + 0.5);
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
  for (const int bits : {16, 8, 6, 4}) {  // fp16 start states (versions 1 and 2), and quantised ones (version 3)
    rollout::Model q = m;
    q.start_bits = bits;
    q.start_dither = bits == 6;
    std::ostringstream os;
    ASSERT_TRUE(rollout::save_model(os, q));
    const auto file = bytes_of(os.str());
    expect_model_round_trip(file);
    const cm::Packed p = cm::pack_model(file);
    std::vector<cm::Kind> kinds;
    for (const auto& part : p.parts) kinds.push_back(part.kind);
    for (const cm::Kind k : {cm::Kind::weights, cm::Kind::biases, cm::Kind::coarse, cm::Kind::fine, cm::Kind::scales}) {
      EXPECT_NE(std::ranges::find(kinds, k), kinds.end()) << cm::kind_name(k) << " " << bits;
    }
    EXPECT_EQ(std::ranges::find(kinds, cm::Kind::ranges) != kinds.end(), bits < 16);
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

// --- format 2 (study H, H3): LZ tokens, the light and fast models, seekable segments ----------------------------------

namespace {

std::vector<cm::Options> format2_options() {
  std::vector<cm::Options> v;
  for (const cm::Literal lit : {cm::Literal::full, cm::Literal::light, cm::Literal::fast}) {
    for (const bool lz : {false, true}) {
      for (const bool seek : {false, true}) {
        if (seek && lit == cm::Literal::full) continue;
        v.push_back({lit, lz, seek, std::size_t{200}});  // small segments: many per tensor
      }
    }
  }
  return v;
}

std::string describe(const cm::Options& o) {
  return std::to_string(static_cast<int>(o.literal)) + (o.lz ? " lz" : "") + (o.seekable ? " seekable" : "");
}

std::size_t round_trip2(const std::vector<cm::Tensor>& in, const cm::Options& o) {
  const cm::Packed p = cm::pack_tensors(in, o);
  EXPECT_EQ(p.data[4], 2) << "format 2";
  const auto back = cm::unpack_tensors(p.data);
  EXPECT_TRUE(back) << describe(o) << ": " << (back ? "" : back.error());
  if (!back) return 0;
  EXPECT_EQ(back->size(), in.size());
  for (std::size_t k = 0; k < std::min(back->size(), in.size()); ++k) {
    EXPECT_EQ((*back)[k].shape.dims, in[k].shape.dims);
    EXPECT_TRUE((*back)[k].values == in[k].values) << describe(o) << ", tensor " << k;
  }
  return p.data.size();
}

void expect_model_round_trip2(const std::vector<std::uint8_t>& file, const cm::Options& o) {
  const cm::Packed p = cm::pack_model(file, o);
  const auto back = cm::unpack_model(p.data);
  ASSERT_TRUE(back) << describe(o) << ": " << back.error();
  EXPECT_TRUE(*back == file) << describe(o);
}

std::vector<std::uint8_t> small_rollout_file() {
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
  for (int k = 0; k < 4; ++k) {
    rollout::StartPoint sp;
    sp.controls = {0.1f * static_cast<float>(k), 0.5f, 0.9f};
    sp.seed = 40 + static_cast<std::uint64_t>(k);
    sp.time = 0.5f * static_cast<float>(k);
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        for (int c = 0; c < rollout::kPhys; ++c) sp.coarse.push_back(std::sin(0.4f * static_cast<float>(x + c) + static_cast<float>(k)) * std::cos(0.3f * static_cast<float>(y)));
      }
    }
    if (k != 1) {  // mostly empty fine fields, as the effects' are
      for (int i = 0; i < 16 * 16; ++i) {
        const bool on = (i % 16) > 5 && (i % 16) < 10 && i / 16 > 4;
        sp.fine_t.push_back(on ? std::max(0.f, std::sin(0.1f * static_cast<float>(i))) : 0.f);
        sp.fine_d.push_back(on ? std::max(0.f, std::cos(0.13f * static_cast<float>(i))) : 0.f);
      }
    }
    m.starts.push_back(sp);
  }
  rollout::quantise_like_storage(m);
  std::ostringstream os;
  EXPECT_TRUE(rollout::save_model(os, m));
  return bytes_of(os.str());
}

std::vector<std::uint8_t> small_frame_file(int bits) {
  Hyper h;
  h.arch = Arch::grid;
  h.size = 32;
  h.frames = 8;
  h.grid = 6;
  h.channels = 3;
  h.hidden = 5;
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
  EXPECT_TRUE(save_model(os, m));
  return bytes_of(os.str());
}

}  // namespace

TEST(Cm, Format2RoundTripsInEveryConfiguration) {
  using K = cm::Kind;
  std::vector<cm::Tensor> ts;
  std::uint64_t seed = 100;
  for (const int width : {1, 2}) {
    ts.push_back(random_tensor(K::bytes, width, {1}, false, seed++));
    ts.push_back(random_tensor(K::weights, width, {5, 9}, false, seed++));
    ts.push_back(random_tensor(K::features, width, {2, 3, 2, 5, 5}, false, seed++));
    ts.push_back(random_tensor(K::coarse, width, {3, 6, 5, 4}, true, seed++));
    ts.push_back(random_tensor(K::rgba, width, {3, 1, 8, 4}, true, seed++));
  }
  cm::Tensor m = random_tensor(K::features, 1, {2, 3, 6, 6}, false, seed++);
  for (int p = 0; p < 6; ++p) {
    m.shape.lo.push_back(-(int64_t{1} << 20) * p);
    m.shape.hi.push_back(p == 2 ? m.shape.lo.back() : p == 4 ? m.shape.lo.back() - 5 : (int64_t{1} << 22) * (p + 1));
  }
  ts.push_back(m);
  cm::Tensor z;  // long repeats: what LZ tokens are for
  z.shape.kind = K::fine;
  z.shape.dims = {4, 2, 16, 16};
  z.values.assign(z.shape.size(), 0);
  for (std::size_t i = 0; i < z.values.size(); i += 37) z.values[i] = static_cast<std::uint16_t>(i % 251);
  ts.push_back(z);
  const auto frame8 = small_frame_file(8), frame16 = small_frame_file(16), roll = small_rollout_file();
  for (const cm::Options& o : format2_options()) {
    round_trip2(ts, o);
    round_trip2({}, o);
    for (const auto* f : {&frame8, &frame16, &roll}) expect_model_round_trip2(*f, o);
    expect_model_round_trip2({}, o);
    expect_model_round_trip2({42}, o);
    expect_model_round_trip2(bytes_of("NVFXROL1 but not really a rollout effect"), o);
    // Format 1 files still unpack, and format 2 is deterministic.
    EXPECT_EQ(cm::pack_model(roll, o).data, cm::pack_model(roll, o).data);
  }
  EXPECT_EQ(cm::pack_model(roll).data[4], 1);
}

TEST(Cm, Format2RoundTripsBitPackedFeatures) {
  // Low-bit features (study F2) in format 2: coded as packed values in a single stream; a seekable file, whose segments
  // are placed byte by byte, falls back to plain bytes. Both restore the file exactly.
  for (const int grid : {16, 5}) {
    Hyper h;
    h.arch = Arch::grid;
    h.size = 32;
    h.frames = 8;
    h.grid = grid;
    h.channels = 4;
    h.hidden = 6;
    h.grid_t = 4;
    Model m = init_model(h, 11);
    m.feature_bits = grid == 16 ? 4 : 5;
    m.effect = "smoke";
    std::ostringstream os;
    ASSERT_TRUE(save_model(os, m));
    const auto file = bytes_of(os.str());
    for (const cm::Options& o : format2_options()) {
      expect_model_round_trip2(file, o);
      const cm::Packed p = cm::pack_model(file, o);
      const bool packed_planes = grid == 16 && !o.seekable;
      EXPECT_EQ(std::ranges::find(p.parts, cm::Kind::features, &cm::Part::kind) != p.parts.end(), packed_planes);
    }
  }
}

TEST(Cm, Format2SizesAndLzTokens) {
  // Exact repeats (one noisy patch copied into every plane of an empty field) cost much less with LZ tokens; noise
  // costs about the same.
  cm::Tensor z;
  z.shape.kind = cm::Kind::fine;
  z.shape.dims = {8, 2, 32, 32};
  z.values.assign(z.shape.size(), 0);
  std::mt19937_64 rng(5);
  std::array<std::uint16_t, 12 * 12> patch{};
  for (auto& v : patch) v = static_cast<std::uint16_t>(rng() & 255);
  for (std::size_t p = 0; p < 16; ++p) {
    for (std::size_t y = 0; y < 12; ++y) {
      for (std::size_t x = 0; x < 12; ++x) z.values[p * 1024 + (y + 10) * 32 + x + 4 + p] = patch[y * 12 + x];
    }
  }
  const cm::Tensor noise = random_tensor(cm::Kind::features, 1, {4, 32, 32}, false, 77);
  for (const cm::Literal lit : {cm::Literal::full, cm::Literal::light, cm::Literal::fast}) {
    const std::size_t with = round_trip2({z}, {lit, true, false}), without = round_trip2({z}, {lit, false, false});
    EXPECT_LT(static_cast<double>(with), 0.8 * static_cast<double>(without)) << static_cast<int>(lit) << ": " << with << " against " << without;
    const std::size_t nw = round_trip2({noise}, {lit, true, false}), nwo = round_trip2({noise}, {lit, false, false});
    EXPECT_LT(static_cast<double>(nw), 1.01 * static_cast<double>(nwo)) << static_cast<int>(lit);
  }
  // The levels are ordered by size on a smooth field: full <= light <= fast, and all far below the raw size.
  cm::Tensor a;
  a.shape.kind = cm::Kind::features;
  a.shape.dims = {8, 48, 48};
  for (int t = 0; t < 8; ++t) {
    for (int y = 0; y < 48; ++y) {
      for (int x = 0; x < 48; ++x) {
        const double v = 128 + 90 * std::sin(0.11 * x + 0.3 * t) * std::cos(0.07 * y - 0.2 * t) + 3.0 * std::sin(1.7 * x * y + t);
        a.values.push_back(static_cast<std::uint16_t>(std::lround(v)));
      }
    }
  }
  const std::size_t full = round_trip2({a}, {cm::Literal::full, true, false}), light = round_trip2({a}, {cm::Literal::light, true, false});
  const std::size_t fast = round_trip2({a}, {cm::Literal::fast, true, false});
  EXPECT_LE(full, light + 64);
  EXPECT_LE(light, fast + 64);
  EXPECT_LT(static_cast<double>(fast), 0.6 * static_cast<double>(a.values.size()));
}

TEST(Cm, SeekableSlicesDecodeAlone) {
  for (const auto& file : {small_rollout_file(), small_frame_file(8), small_frame_file(16)}) {
    for (const cm::Literal lit : {cm::Literal::light, cm::Literal::fast}) {
      const cm::Packed p = cm::pack_model(file, {lit, true, true, 128});
      const auto list = cm::list_slices(p.data);
      ASSERT_TRUE(list) << list.error();
      ASSERT_GT(list->size(), 3u);
      std::size_t values = 0, packed = 0;
      for (std::size_t k = 0; k < list->size(); ++k) {
        const auto s = cm::unpack_slice(p.data, k);
        ASSERT_TRUE(s) << s.error();
        const cm::SliceInfo& info = (*list)[k];
        EXPECT_EQ(s->tensor.shape.kind, info.kind);
        EXPECT_EQ(s->first_plane, info.first_plane);
        ASSERT_EQ(s->tensor.values.size(), info.values);
        ASSERT_EQ(s->at.size(), info.values);
        for (std::size_t q = 0; q < s->at.size(); ++q) {
          const int w = s->tensor.shape.width;
          const auto want = static_cast<std::uint16_t>(file[s->at[q]] | (w == 2 ? file[s->at[q] + 1] << 8 : 0));
          ASSERT_EQ(s->tensor.values[q], want) << "slice " << k << " value " << q;
        }
        values += info.values * static_cast<std::size_t>(s->tensor.shape.width);
        packed += info.packed_bytes;
      }
      EXPECT_LT(values, file.size());  // headers and small tensors are in segment 0
      EXPECT_LT(packed, p.data.size());
      EXPECT_FALSE(cm::unpack_slice(p.data, list->size()));
      // Damage inside a segment's stream is refused when that segment is decoded.
      for (std::size_t k = 0; k < list->size(); k += 3) {
        std::size_t at = p.data.size();
        for (std::size_t j = list->size(); j-- > k;) at -= (*list)[j].packed_bytes;
        auto bad = p.data;
        bad[at + (*list)[k].packed_bytes / 2] ^= 0x41;
        const auto s = cm::unpack_slice(bad, k);
        const auto ok = cm::unpack_slice(p.data, k);
        EXPECT_TRUE(!s || s->tensor.values == ok->tensor.values) << "slice " << k;
        EXPECT_FALSE(cm::unpack_model(bad).has_value() && *cm::unpack_model(bad) != file);
      }
    }
  }
  // Not seekable: refused.
  const auto f = small_rollout_file();
  EXPECT_FALSE(cm::list_slices(cm::pack_model(f).data));
  EXPECT_FALSE(cm::unpack_slice(cm::pack_model(f, {cm::Literal::light, true, false}).data, 0));
}

TEST(Cm, Format2DamagedDataIsRefused) {
  const auto file = small_frame_file(8);
  for (const cm::Options& o : format2_options()) {
    const cm::Packed p = cm::pack_model(file, o);
    int refused = 0, tries = 0;
    for (std::size_t at = 0; at < p.data.size(); at += 5, ++tries) {
      auto bad = p.data;
      bad[at] ^= 0x5a;
      const auto r = cm::unpack_model(bad);
      EXPECT_TRUE(!r || *r == file) << describe(o) << ": damage at " << at << " gave another file";
      refused += r ? 0 : 1;
    }
    EXPECT_GT(refused, tries * 3 / 4) << describe(o);
    auto cut = p.data;
    cut.resize(cut.size() / 2);
    EXPECT_FALSE(cm::unpack_model(cut)) << describe(o);
    auto flags = p.data;
    flags[22] = 0xff;  // unknown options
    EXPECT_FALSE(cm::unpack_model(flags)) << describe(o);
  }
}
