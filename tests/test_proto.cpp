// Phase 0 prototypes: every ISA renders the same frame as the baseline (within rounding; the SIMD builds fuse
// multiply-adds), rendering is deterministic, and the cost bookkeeping matches the shapes.
#include <neuralfx/proto.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

using nfx::proto::Isa;
using nfx::proto::Spec;

namespace {

std::vector<Spec> specs() {
  std::vector<Spec> v;
  Spec s;
  s.size = 64;
  s.kind = "mlp_naive";
  v.push_back(s);
  s.kind = "mlp_sep";
  v.push_back(s);
  s.kind = "grid_mlp";
  s.grid = 16;
  s.grid_t = 4;
  s.channels = 6;  // not a multiple of the tile: exercises the remainder path
  s.hidden = 20;
  v.push_back(s);
  s.kind = "hash_mlp";
  s.levels = 4;
  s.log2_table = 10;
  v.push_back(s);
  s.kind = "conv_dec";
  s.latent = 8;
  s.c0 = 12;
  s.c1 = 10;
  s.c2 = 6;
  v.push_back(s);
  return v;
}

std::vector<std::uint8_t> render(const Spec& s, Isa isa, float t) {
  auto m = nfx::proto::make_model(s, isa);
  std::vector<std::uint8_t> px(static_cast<std::size_t>(s.size) * s.size * 4);
  nfx::proto::Controls c;
  c.t = t;
  c.wind = 0.3f;
  m->render(c, px.data());
  return px;
}

}  // namespace

TEST(NeuralfxProto, EveryIsaMatchesTheBaseline) {
  for (const Spec& s : specs()) {
    const auto ref = render(s, Isa::base, 0.37f);
    for (const Isa isa : {Isa::avx2, Isa::avx512}) {
      if (!nfx::proto::isa_supported(isa)) continue;
      const auto got = render(s, isa, 0.37f);
      int worst = 0, nonzero = 0;
      for (std::size_t i = 0; i < ref.size(); ++i) {
        worst = std::max(worst, std::abs(int(ref[i]) - int(got[i])));
        nonzero += ref[i] != 0;
      }
      EXPECT_LE(worst, 2) << s.describe() << " on " << nfx::proto::isa_name(isa);
      EXPECT_GT(nonzero, 0) << s.describe() << ": an all-zero frame would hide differences";
    }
  }
}

TEST(NeuralfxProto, RenderingIsDeterministicAndDependsOnTime) {
  for (const Spec& s : specs()) {
    const Isa isa = nfx::proto::best_isa();
    EXPECT_EQ(render(s, isa, 0.5f), render(s, isa, 0.5f)) << s.describe();
    EXPECT_NE(render(s, isa, 0.1f), render(s, isa, 0.9f)) << s.describe();
  }
}

TEST(NeuralfxProto, SeparableFirstLayerEqualsTheNaiveMlp) {
  // Same seed, same weights: splitting the first layer by input group must not change the image.
  Spec s;
  s.size = 64;
  s.kind = "mlp_naive";
  const auto a = render(s, Isa::base, 0.25f);
  s.kind = "mlp_sep";
  const auto b = render(s, Isa::base, 0.25f);
  int worst = 0;
  for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(int(a[i]) - int(b[i])));
  EXPECT_LE(worst, 1);
}

TEST(NeuralfxProto, CostBookkeeping) {
  Spec s;
  s.kind = "mlp_naive";
  s.hidden = 32;
  s.layers = 2;
  s.freqs = 6;
  const auto m = nfx::proto::make_model(s, Isa::base);
  const int nin = 6 * 6 + 5;
  EXPECT_EQ(m->param_count(), static_cast<std::size_t>(nin * 32 + 32 + 32 * 32 + 32 + 32 * 4 + 4));
  EXPECT_DOUBLE_EQ(m->macs_per_pixel(), nin * 32 + 32 * 32 + 32 * 4);

  s.kind = "conv_dec";
  s.size = 128;
  s.latent = 16;
  s.c0 = 32;
  s.c1 = 16;
  s.c2 = 8;
  const auto c = nfx::proto::make_model(s, Isa::base);
  EXPECT_DOUBLE_EQ(c->macs_per_pixel(), (9.0 * 32 * 16 * 32 * 32 + 9.0 * 16 * 8 * 64 * 64 + 9.0 * 8 * 4 * 128 * 128) / (128 * 128));
}

TEST(NeuralfxProto, BadSpecsAreRejected) {
  Spec s;
  s.size = 100;
  EXPECT_THROW(nfx::proto::make_model(s, Isa::base), std::invalid_argument);
  s.size = 128;
  s.kind = "nope";
  EXPECT_THROW(nfx::proto::make_model(s, Isa::base), std::invalid_argument);
  s.kind = "conv_dec";
  s.latent = 8;  // 8 * 8 != 128
  EXPECT_THROW(nfx::proto::make_model(s, Isa::base), std::invalid_argument);
  Isa isa{};
  EXPECT_FALSE(nfx::proto::parse_isa("sse9", isa));
  EXPECT_TRUE(nfx::proto::parse_isa("avx2", isa));
  EXPECT_EQ(isa, Isa::avx2);
}
