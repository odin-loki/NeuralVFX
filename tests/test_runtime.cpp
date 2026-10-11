// Phase 4: the runtime (C API) against the reference forward pass on every ISA the CPU has, its time, variation,
// colour and bake logic, and its error handling.
#include <neuralfx/model.hpp>
#include <neuralfx/nvfx.h>

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

using namespace nfx;

namespace {

struct EffectDeleter {
  void operator()(nvfx_effect* e) const { nvfx_effect_free(e); }
};
struct InstanceDeleter {
  void operator()(nvfx_instance* i) const { nvfx_instance_free(i); }
};
using EffectPtr = std::unique_ptr<nvfx_effect, EffectDeleter>;
using InstancePtr = std::unique_ptr<nvfx_instance, InstanceDeleter>;

Hyper grid_hyper() {
  Hyper h;
  h.arch = Arch::grid;
  h.size = 32;
  h.frames = 12;
  h.loop = true;
  h.n_controls = 3;
  h.n_latent = 2;
  h.bases = 3;
  h.grid_t = 5;
  h.grid = 12;
  h.channels = 6;
  h.hidden = 20;  // not a multiple of the register tile: exercises the remainder paths
  h.layers = 2;
  return h;
}

Hyper conv_hyper() {
  Hyper h;
  h.arch = Arch::conv;
  h.latent = 4;
  h.size = 32;
  h.frames = 12;
  h.loop = false;
  h.n_controls = 3;
  h.n_latent = 2;
  h.bases = 2;
  h.grid_t = 5;
  h.c0 = 6;
  h.c1 = 10;
  h.c2 = 5;
  return h;
}

// A model with every weight non-trivial, codes kept, and weights rounded to their storage precision.
Model make_model(const Hyper& h, int bits) {
  Model m = init_model(h, 21);
  std::mt19937_64 rng(22);
  std::uniform_real_distribution<float> u(-0.4f, 0.4f);
  for (auto& f : m.films) {
    for (float& w : f.w) w = u(rng);
    for (float& b : f.b) b = u(rng);
  }
  for (float& v : m.features) v *= 6.f;
  for (float& w : m.layers.back().w) w *= 8.f;
  for (float& b : m.layers.back().b) b = 0.3f + u(rng);
  m.effect = "test";
  m.feature_bits = bits;
  m.control_names = {"intensity", "wind", "turbulence"};
  m.z_train = {{0.2f, -0.5f}, {-0.4f, 0.3f}, {0.1f, 0.1f}};
  m.z_mean = {0.f, 0.f};
  m.z_std = {0.3f, 0.3f};
  quantise_like_storage(m);
  return m;
}

EffectPtr load(const Model& m) {
  std::ostringstream os;
  EXPECT_TRUE(save_model(os, m).has_value());
  const std::string bytes = os.str();
  nvfx_effect* e = nullptr;
  EXPECT_EQ(nvfx_effect_load_memory(bytes.data(), bytes.size(), &e), NVFX_OK);
  return EffectPtr(e);
}

InstancePtr instance(const nvfx_effect* e, int size) {
  nvfx_instance* in = nullptr;
  EXPECT_EQ(nvfx_instance_create(e, size, &in), NVFX_OK);
  return InstancePtr(in);
}

std::vector<std::uint8_t> render(nvfx_instance* in, double seconds, int size) {
  std::vector<std::uint8_t> out(static_cast<std::size_t>(size) * size * 4);
  EXPECT_EQ(nvfx_render(in, seconds, out.data(), static_cast<std::size_t>(size) * 4), NVFX_OK);
  return out;
}

// The reference at the same moment: model time for frame index `f`, condition from controls and a training code.
std::vector<std::uint8_t> reference(const Model& m, int f, std::span<const float> controls, int code, int size) {
  const int native = m.h.arch == Arch::conv ? m.h.size : size;
  std::vector<float> rgba(static_cast<std::size_t>(native) * native * 4);
  reference_render(m, frame_time(m.h, f, m.h.frames), condition(m, controls, m.z_train[static_cast<std::size_t>(code)]), native, rgba);
  if (native != size) {  // the runtime's level of detail: box filter
    const int k = native / size;
    std::vector<float> small(static_cast<std::size_t>(size) * size * 4, 0.f);
    for (int y = 0; y < native; ++y) {
      for (int x = 0; x < native; ++x) {
        for (int c = 0; c < 4; ++c) {
          small[((static_cast<std::size_t>(y / k) * size + x / k) * 4) + static_cast<std::size_t>(c)] +=
              rgba[(static_cast<std::size_t>(y) * native + x) * 4 + static_cast<std::size_t>(c)] / static_cast<float>(k * k);
        }
      }
    }
    rgba = small;
  }
  std::vector<std::uint8_t> out(rgba.size());
  to_rgba8(rgba, out);
  return out;
}

int max_diff(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
  int d = 0;
  for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, std::abs(int(a[i]) - int(b[i])));
  return d;
}

const nvfx_isa kIsas[] = {NVFX_ISA_BASELINE, NVFX_ISA_AVX2, NVFX_ISA_AVX512};

}  // namespace

TEST(Runtime, MatchesTheReferenceOnEveryIsaFamilyPrecisionAndSize) {  // precisions: fp16, 8 bits, 3 to 6 bits packed
  const float controls[3] = {0.8f, 0.1f, 0.6f};
  struct Case {
    Hyper h;
    int bits, size;
  };
  Hyper odd = grid_hyper();  // planes of 13 x 13: bit-packed planes end inside a byte
  odd.grid = 13;
  const Case cases[] = {{grid_hyper(), 16, 32}, {grid_hyper(), 8, 32}, {grid_hyper(), 16, 48}, {grid_hyper(), 6, 32},
                        {grid_hyper(), 5, 48},  {grid_hyper(), 4, 32}, {odd, 5, 32},           {odd, 3, 32},
                        {conv_hyper(), 16, 32}, {conv_hyper(), 8, 32}, {conv_hyper(), 16, 16}, {conv_hyper(), 4, 32},
                        {conv_hyper(), 6, 32}};
  for (const nvfx_isa isa : kIsas) {
    if (nvfx_set_isa(isa) != NVFX_OK) continue;  // the CPU lacks it
    for (const Case& c : cases) {
      const Model m = make_model(c.h, c.bits);
      auto e = load(m);
      auto in = instance(e.get(), c.size);
      ASSERT_EQ(nvfx_instance_set_precision(in.get(), NVFX_PRECISION_FLOAT), NVFX_OK);  // the float network (int8: below)
      nvfx_instance_set_controls(in.get(), controls, 3);
      nvfx_instance_set_variation(in.get(), 1);
      for (const int f : {0, 5, 11}) {
        const auto got = render(in.get(), f / static_cast<double>(m.fps), c.size);
        EXPECT_LE(max_diff(got, reference(m, f, controls, 1, c.size)), 2)
            << m.h.describe() << " bits " << c.bits << " size " << c.size << " frame " << f << " isa " << isa;
      }
    }
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
}

TEST(Runtime, VectorQuantisedFeaturesMatchTheReference) {
  // Codebooks of 2^bits vectors per group of channels: the runtime decodes the index planes as the reference sees the
  // features after quantise_like_storage (nearest codewords).
  const float controls[3] = {0.3f, 0.6f, 0.1f};
  struct Case {
    Hyper h;
    int bits, dim, size;
  };
  const Case cases[] = {{grid_hyper(), 4, 2, 32}, {grid_hyper(), 8, 6, 48}, {grid_hyper(), 5, 3, 32}, {conv_hyper(), 6, 3, 32}};
  for (const nvfx_isa isa : kIsas) {
    if (nvfx_set_isa(isa) != NVFX_OK) continue;
    for (const Case& c : cases) {
      Model m = make_model(c.h, 16);
      m.feature_bits = 8;
      m.vq_bits = c.bits;
      m.vq_dim = c.dim;
      std::mt19937_64 rng(5);
      std::uniform_real_distribution<float> u(-0.6f, 0.6f);
      m.vq_codebook.resize(static_cast<std::size_t>(m.vq_groups()) * (std::size_t{1} << c.bits) * static_cast<std::size_t>(c.dim));
      for (float& v : m.vq_codebook) v = u(rng);
      quantise_like_storage(m);
      auto e = load(m);
      auto in = instance(e.get(), c.size);
      ASSERT_EQ(nvfx_instance_set_precision(in.get(), NVFX_PRECISION_FLOAT), NVFX_OK);
      nvfx_instance_set_controls(in.get(), controls, 3);
      nvfx_instance_set_variation(in.get(), 1);
      for (const int f : {0, 7}) {
        const auto got = render(in.get(), f / static_cast<double>(m.fps), c.size);
        EXPECT_LE(max_diff(got, reference(m, f, controls, 1, c.size)), 2) << m.h.describe() << " vq " << c.bits << "x" << c.dim << " isa " << isa;
      }
      nvfx_effect_info info{};
      nvfx_effect_get_info(e.get(), &info);
      EXPECT_EQ(info.stored_bytes, m.storage_bytes());
      EXPECT_LT(info.stored_bytes, make_model(c.h, 8).storage_bytes());
    }
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
}

// The int8 path (nvfx_instance_set_precision; src/runtime/rt_int8.hpp) is not the float network: the hidden layers'
// activations and weights are rounded to 8 bits (activations to 7 with AVX2's pmaddubsw). It is held to the float
// reference within a stated tolerance on models whose every weight is non-trivial (larger outputs than trained models,
// so larger errors): no channel more than kInt8Max levels of 255 away, and on average less than kInt8Mean levels. The
// worst case here is two hidden layers with AVX2's 7-bit activations (8 levels, 0.51 on average); with one hidden
// layer, as every trained grid model has, the mean stays below 0.3. Shapes: hidden widths that are not a multiple of 4
// (the words are padded), one and two hidden layers and none, the first layer projected (frame at least as wide as the
// grid) or per pixel (narrower), 16- and 8-bit features. On the trained models of studies A to C the measured changes
// are in results/experiments/int8_summary.csv.
constexpr int kInt8Max = 8;
constexpr double kInt8Mean = 0.6;

TEST(Runtime, Int8StaysWithinItsToleranceOfTheFloatReferenceOnEveryIsa) {
  const float controls[3] = {0.8f, 0.1f, 0.6f};
  struct Case {
    Hyper h;
    int bits, size;
  };
  Hyper deep = grid_hyper(), shallow = grid_hyper(), wide = grid_hyper(), quad = grid_hyper(), odd = grid_hyper();
  deep.layers = 3;
  shallow.layers = 1;
  wide.grid = 40;  // more grid points across than pixels at 32: the first layer per pixel
  quad.hidden = 24;
  odd.hidden = 21;  // a last word in part (pairs and quads)
  const Case cases[] = {{grid_hyper(), 16, 32}, {grid_hyper(), 8, 48}, {grid_hyper(), 16, 16}, {deep, 16, 32}, {shallow, 8, 32},
                        {wide, 16, 32},         {wide, 8, 64},         {quad, 16, 48},         {odd, 16, 32}};
  int compared = 0;
  for (const nvfx_isa isa : kIsas) {
    if (nvfx_set_isa(isa) != NVFX_OK) continue;
    for (const Case& c : cases) {
      const Model m = make_model(c.h, c.bits);
      auto e = load(m);
      auto in = instance(e.get(), c.size);
      ASSERT_EQ(nvfx_instance_set_precision(in.get(), NVFX_PRECISION_INT8), NVFX_OK);
      nvfx_instance_set_controls(in.get(), controls, 3);
      nvfx_instance_set_variation(in.get(), 1);
      for (const int f : {0, 5, 11}) {
        const auto got = render(in.get(), f / static_cast<double>(m.fps), c.size);
        const auto want = reference(m, f, controls, 1, c.size);
        double sum = 0;
        for (std::size_t i = 0; i < got.size(); ++i) sum += std::abs(int(got[i]) - int(want[i]));
        EXPECT_LE(max_diff(got, want), kInt8Max) << m.h.describe() << " size " << c.size << " frame " << f << " isa " << isa;
        EXPECT_LE(sum / static_cast<double>(got.size()), kInt8Mean) << m.h.describe() << " size " << c.size << " frame " << f << " isa " << isa;
        ++compared;
      }
    }
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
  EXPECT_GE(compared, 24);
}

TEST(Runtime, PrecisionIsASettingOfTheInstance) {
  const Model g = make_model(grid_hyper(), 16), c = make_model(conv_hyper(), 16);
  auto eg = load(g), ec = load(c);
  auto def = instance(eg.get(), 32), q = instance(eg.get(), 32), fl = instance(eg.get(), 32);
  ASSERT_EQ(nvfx_instance_set_precision(q.get(), NVFX_PRECISION_INT8), NVFX_OK);
  ASSERT_EQ(nvfx_instance_set_precision(fl.get(), NVFX_PRECISION_FLOAT), NVFX_OK);
  for (auto* in : {def.get(), q.get(), fl.get()}) nvfx_instance_set_variation(in, 1);
  EXPECT_EQ(render(def.get(), 0.1, 32), render(q.get(), 0.1, 32));  // the default for the grid family is int8
  const auto f0 = render(fl.get(), 0.1, 32);
  EXPECT_NE(render(q.get(), 0.1, 32), f0);  // (the test model's weights make the difference visible)
  ASSERT_EQ(nvfx_instance_set_precision(q.get(), NVFX_PRECISION_FLOAT), NVFX_OK);  // and back: the float network exactly
  EXPECT_EQ(render(q.get(), 0.1, 32), f0);
  EXPECT_GT(nvfx_instance_scratch_bytes(def.get()), 0u);
  // the conv family and rollout effects have no int8 path
  auto cv = instance(ec.get(), 32);
  nvfx_instance_set_variation(cv.get(), 0);
  const auto conv_default = render(cv.get(), 0.1, 32);
  EXPECT_EQ(nvfx_instance_set_precision(cv.get(), NVFX_PRECISION_INT8), NVFX_ERROR_UNSUPPORTED);
  EXPECT_EQ(nvfx_instance_set_precision(cv.get(), NVFX_PRECISION_FLOAT), NVFX_OK);
  EXPECT_EQ(render(cv.get(), 0.1, 32), conv_default);
  const int bad = 3;  // not a precision
  EXPECT_EQ(nvfx_instance_set_precision(cv.get(), static_cast<nvfx_precision>(bad)), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_instance_set_precision(nullptr, NVFX_PRECISION_FLOAT), NVFX_ERROR_ARGUMENT);
}

TEST(Runtime, InfoNamesAndMemory) {
  const Model m = make_model(grid_hyper(), 16);
  auto e = load(m);
  nvfx_effect_info info{};
  ASSERT_EQ(nvfx_effect_get_info(e.get(), &info), NVFX_OK);
  EXPECT_EQ(info.arch, 1);
  EXPECT_EQ(info.native_size, 32);
  EXPECT_EQ(info.loops, 1);
  EXPECT_EQ(info.n_controls, 3);
  EXPECT_EQ(info.n_variations, 3);
  EXPECT_EQ(info.stored_bytes, m.storage_bytes());
  EXPECT_STREQ(info.name, "test");
  EXPECT_STREQ(nvfx_effect_control_name(e.get(), 1), "wind");
  EXPECT_EQ(nvfx_effect_control_name(e.get(), 3), nullptr);
  // Features stay at storage precision; only the small MLP is widened to floats.
  const std::size_t feature_bytes = m.features.size() * 2;
  EXPECT_LT(info.resident_bytes, feature_bytes + 4 * (m.param_count() - m.features.size()) + 64);
}

TEST(Runtime, SeedsDriftAndVariationsBehave) {
  const Model m = make_model(grid_hyper(), 16);
  auto e = load(m);
  auto a = instance(e.get(), 32), b = instance(e.get(), 32);
  nvfx_instance_set_drift(a.get(), 0.f);
  nvfx_instance_set_drift(b.get(), 0.f);
  nvfx_instance_set_seed(a.get(), 1);
  nvfx_instance_set_seed(b.get(), 1);
  EXPECT_EQ(render(a.get(), 0.1, 32), render(b.get(), 0.1, 32));  // same seed, same frame
  nvfx_instance_set_seed(b.get(), 2);
  EXPECT_GT(max_diff(render(a.get(), 0.1, 32), render(b.get(), 0.1, 32)), 0);  // another seed, another variation
  // Drift is continuous: across a waypoint, a millisecond changes the frame no more than the animation itself does.
  const int still = max_diff(render(a.get(), 0.2495, 32), render(a.get(), 0.2505, 32));  // drift off
  nvfx_instance_set_drift(a.get(), 0.25f);
  EXPECT_LE(max_diff(render(a.get(), 0.2495, 32), render(a.get(), 0.2505, 32)), still + 2);
  // ...and the loop does not repeat exactly once drifting.
  const double loop = m.h.frames / static_cast<double>(m.fps);
  EXPECT_GT(max_diff(render(a.get(), 0.1, 32), render(a.get(), 0.1 + 3 * loop, 32)), 0);
  EXPECT_EQ(nvfx_instance_set_variation(a.get(), 3), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_instance_set_variation(a.get(), -1), NVFX_OK);
}

TEST(Runtime, TimeWrapsForLoopsAndHoldsForOneShots) {
  const Model g = make_model(grid_hyper(), 16), c = make_model(conv_hyper(), 16);
  auto eg = load(g), ec = load(c);
  auto ig = instance(eg.get(), 32), ic = instance(ec.get(), 32);
  nvfx_instance_set_variation(ig.get(), 0);
  nvfx_instance_set_variation(ic.get(), 0);
  const double loop = g.h.frames / static_cast<double>(g.fps);
  EXPECT_LE(max_diff(render(ig.get(), 0.05, 32), render(ig.get(), 0.05 + 2 * loop, 32)), 1);
  const double end = (c.h.frames - 1) / static_cast<double>(c.fps);
  EXPECT_EQ(render(ic.get(), end, 32), render(ic.get(), end + 5.0, 32));
}

TEST(Runtime, ColourControlsAreExact) {
  const Model m = make_model(grid_hyper(), 16);
  auto e = load(m);
  auto in = instance(e.get(), 32);
  nvfx_instance_set_variation(in.get(), 2);
  const auto plain = render(in.get(), 0.2, 32);
  nvfx_instance_set_colour(in.get(), 0.f, 1.f);
  EXPECT_EQ(render(in.get(), 0.2, 32), plain);
  nvfx_instance_set_colour(in.get(), 0.f, 0.f);
  const auto black = render(in.get(), 0.2, 32);
  for (std::size_t i = 0; i < black.size(); i += 4) {
    EXPECT_EQ(black[i], 0);
    EXPECT_EQ(black[i + 3], plain[i + 3]);  // coverage untouched
  }
  // A full turn of hue is the identity (within rounding).
  nvfx_instance_set_colour(in.get(), 2.f * 3.14159265f, 1.f);
  EXPECT_LE(max_diff(render(in.get(), 0.2, 32), plain), 1);
}

TEST(Runtime, BakeEqualsRenderingEachFrame) {
  const Model m = make_model(grid_hyper(), 16);
  auto e = load(m);
  auto in = instance(e.get(), 32);
  nvfx_instance_set_seed(in.get(), 7);
  nvfx_instance_set_drift(in.get(), 0.f);
  const int frames = 6;
  std::vector<std::uint8_t> baked(static_cast<std::size_t>(frames) * 32 * 32 * 4);
  ASSERT_EQ(nvfx_bake(in.get(), frames, baked.data()), NVFX_OK);
  const double loop = m.h.frames / static_cast<double>(m.fps);
  for (int f = 0; f < frames; ++f) {
    const auto one = render(in.get(), loop * f / frames, 32);
    EXPECT_EQ(std::vector<std::uint8_t>(baked.begin() + f * 4096, baked.begin() + (f + 1) * 4096), one) << f;
  }
}

TEST(Runtime, ErrorsAreStatusesNotCrashes) {
  const Model g = make_model(grid_hyper(), 16), c = make_model(conv_hyper(), 16);
  auto eg = load(g), ec = load(c);
  nvfx_instance* in = nullptr;
  EXPECT_EQ(nvfx_instance_create(eg.get(), 40, &in), NVFX_ERROR_UNSUPPORTED);  // not a multiple of 16
  EXPECT_EQ(nvfx_instance_create(ec.get(), 24, &in), NVFX_ERROR_UNSUPPORTED);  // conv: native or a power-of-two fraction
  EXPECT_EQ(nvfx_effect_load("/nonexistent.nvfx", nullptr), NVFX_ERROR_ARGUMENT);
  nvfx_effect* e = nullptr;
  EXPECT_EQ(nvfx_effect_load("/nonexistent.nvfx", &e), NVFX_ERROR_IO);
  const char junk[] = "NVFXMDL1 but then nothing useful";
  EXPECT_EQ(nvfx_effect_load_memory(junk, sizeof junk, &e), NVFX_ERROR_FORMAT);
  auto ok = instance(eg.get(), 32);
  std::vector<std::uint8_t> buf(32 * 32 * 4);
  EXPECT_EQ(nvfx_render(ok.get(), 0.0, buf.data(), 16), NVFX_ERROR_ARGUMENT);  // stride too small
  EXPECT_EQ(nvfx_render(ok.get(), std::nan(""), buf.data(), 128), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_instance_set_colour(ok.get(), 0.f, -1.f), NVFX_ERROR_ARGUMENT);
}

TEST(Runtime, MixedPrecisionFeaturesMatchTheReference) {
  // Every plane at its own width, 0 to 8 bits (study F3): the runtime decodes each plane as the reference sees the
  // features after quantise_like_storage, on every ISA, for both families and for planes that end inside a byte.
  const float controls[3] = {0.4f, 0.9f, 0.2f};
  Hyper odd = grid_hyper();
  odd.grid = 13;
  const Hyper hypers[] = {grid_hyper(), odd, conv_hyper()};
  for (const nvfx_isa isa : kIsas) {
    if (nvfx_set_isa(isa) != NVFX_OK) continue;
    for (const Hyper& h : hypers) {
      Model m = make_model(h, 16);
      m.feature_bits = 8;
      const std::size_t side2 = static_cast<std::size_t>(h.feature_side()) * h.feature_side();
      m.plane_bits.resize(m.features.size() / side2);
      for (std::size_t k = 0; k < m.plane_bits.size(); ++k) m.plane_bits[k] = static_cast<std::uint8_t>((k * 5) % 9);
      quantise_like_storage(m);
      auto e = load(m);
      auto in = instance(e.get(), 32);
      ASSERT_EQ(nvfx_instance_set_precision(in.get(), NVFX_PRECISION_FLOAT), NVFX_OK);  // the float reference
      nvfx_instance_set_controls(in.get(), controls, 3);
      nvfx_instance_set_variation(in.get(), 2);
      for (const int f : {0, 6, 13}) {
        const auto got = render(in.get(), f / static_cast<double>(m.fps), 32);
        EXPECT_LE(max_diff(got, reference(m, f, controls, 2, 32)), 2) << h.describe() << " mixed, frame " << f << " isa " << isa;
      }
      nvfx_effect_info info{};
      nvfx_effect_get_info(e.get(), &info);
      EXPECT_EQ(info.stored_bytes, m.storage_bytes());
      EXPECT_LT(info.stored_bytes, make_model(h, 8).storage_bytes());  // 4 bits on average plus a byte per plane
    }
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
}

TEST(Runtime, SparseFeaturesMatchTheReference) {
  // A mask per time slice (study F3): the runtime decodes the stored points of each plane in raster order and gives
  // every other point the plane's fill, as the reference sees the features after quantise_like_storage. Planes of
  // 13 x 13 points end inside a byte; one slice stores no point at all.
  const float controls[3] = {0.7f, 0.3f, 0.5f};
  Hyper odd = grid_hyper();
  odd.grid = 13;
  for (const nvfx_isa isa : kIsas) {
    if (nvfx_set_isa(isa) != NVFX_OK) continue;
    for (const Hyper& h : {grid_hyper(), odd}) {
      Model m = make_model(h, 16);
      m.feature_bits = 4;
      const std::size_t side2 = static_cast<std::size_t>(h.grid) * h.grid;
      m.plane_bits.resize(m.features.size() / side2);
      for (std::size_t k = 0; k < m.plane_bits.size(); ++k) m.plane_bits[k] = static_cast<std::uint8_t>((k * 5) % 9);
      std::mt19937_64 rng(31);
      m.feature_mask.resize(static_cast<std::size_t>(h.grid_t) * side2);
      for (std::size_t j = 0; j < m.feature_mask.size(); ++j) m.feature_mask[j] = j < side2 ? 0 : static_cast<std::uint8_t>(rng() % 2);
      quantise_like_storage(m);
      auto e = load(m);
      auto in = instance(e.get(), 32);
      ASSERT_EQ(nvfx_instance_set_precision(in.get(), NVFX_PRECISION_FLOAT), NVFX_OK);  // the float reference
      nvfx_instance_set_controls(in.get(), controls, 3);
      nvfx_instance_set_variation(in.get(), 0);
      for (const int f : {0, 4, 9}) {
        const auto got = render(in.get(), f / static_cast<double>(m.fps), 32);
        EXPECT_LE(max_diff(got, reference(m, f, controls, 0, 32)), 2) << h.describe() << " sparse, frame " << f << " isa " << isa;
      }
      nvfx_effect_info info{};
      nvfx_effect_get_info(e.get(), &info);
      EXPECT_EQ(info.stored_bytes, m.storage_bytes());
      Model dense = m;
      dense.feature_mask.clear();
      EXPECT_LT(info.stored_bytes, dense.storage_bytes());
    }
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
}
