// Study P (docs/STUDY_P.md): a rollout effect drawn at half size and upscaled. The upscaler's format, its float reference
// (src/core/upscale.cpp) against the runtime's copy on every ISA this CPU runs, the micro-detail's noise, the C API
// (nvfx_effect_attach_upscaler, nvfx_instance_set_upscale) on the tests' small rollout effect: what an instance that
// upscales draws, determinism, errors, no heap allocation per frame (this program replaces operator new to count them, as
// nvfx_alloc_test does), and frozen frames on the AVX2 build at float.
#include "compose_scene.hpp"
#include "nvfx_internal.hpp"
#include "rt_upscale.hpp"

#include <neuralfx/nvfx.h>
#include <neuralfx/nvfx.hpp>
#include <neuralfx/noise.hpp>
#include <neuralfx/rollout.hpp>
#include <neuralfx/upscale.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <new>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::atomic<long> g_allocations{0};
std::atomic<bool> g_counting{false};
}  // namespace

void* operator new(std::size_t n) {
  if (g_counting.load(std::memory_order_relaxed)) g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) {
  if (g_counting.load(std::memory_order_relaxed)) g_allocations.fetch_add(1, std::memory_order_relaxed);
  const std::size_t al = static_cast<std::size_t>(a);
  if (void* p = std::aligned_alloc(al, (n + al - 1) / al * al + (n ? 0 : al))) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n, std::align_val_t a) { return operator new(n, a); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

namespace up = nfx::upscale;
namespace rt = nfx::rt;
using Bytes = std::vector<std::uint8_t>;
std::size_t sz(int v) { return static_cast<std::size_t>(v); }

// A net with every output in play: the residuals and the amplitudes pushed away from zero.
up::Net test_net(int c1 = 8, int c2 = 8, int k2 = 1, std::uint64_t seed = 11) {
  up::Net n = up::init_net(c1, c2, k2, seed);
  const auto L = n.layout();
  std::mt19937_64 rng(seed + 1);
  std::normal_distribution<float> d(0.f, 0.15f);
  for (std::size_t i = L.wo; i < L.bo; ++i) n.w[i] = d(rng);
  for (std::size_t i = L.bo; i < L.size; ++i) n.w[i] = 0.02f;
  n.gain = 0.8f;
  n.rate = 9.f;
  n.tag = "test";
  return n;
}

// A half-size frame with soft blobs, hard edges and empty space (premultiplied, alpha above colour where it is smoke).
Bytes test_frame(int n, std::uint64_t seed = 5) {
  Bytes f(sz(n) * sz(n) * 4, 0);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> u(0.f, 1.f);
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      const float cx = static_cast<float>(x) / static_cast<float>(n) - 0.45f, cy = static_cast<float>(y) / static_cast<float>(n) - 0.55f;
      const float r = std::sqrt(cx * cx + cy * cy);
      if (r > 0.38f) continue;  // empty around
      const float a = std::clamp(1.2f - 3.f * r + 0.25f * u(rng), 0.f, 1.f);
      std::uint8_t* p = f.data() + (sz(y) * sz(n) + sz(x)) * 4;
      p[3] = static_cast<std::uint8_t>(a * 255.f);
      p[0] = static_cast<std::uint8_t>(std::min(255.f, a * 255.f * (0.6f + 0.6f * u(rng))));  // emission may exceed alpha
      p[1] = static_cast<std::uint8_t>(a * 160.f);
      p[2] = static_cast<std::uint8_t>(a * 90.f * u(rng));
    }
  }
  return f;
}

struct IsaUp {
  nvfx_isa isa;
  std::unique_ptr<rt::Upscaler> (*make)(const up::Net*, int);
};
std::vector<IsaUp> isas() {
  std::vector<IsaUp> v{{NVFX_ISA_BASELINE, rt::isa_base::make_upscaler}};
  std::uint32_t f = 0;
  nvfx_version(&f);
  if (f & NVFX_FEATURE_AVX2) v.push_back({NVFX_ISA_AVX2, rt::isa_avx2::make_upscaler});
  if (f & NVFX_FEATURE_AVX512) v.push_back({NVFX_ISA_AVX512, rt::isa_avx512::make_upscaler});
  return v;
}

struct Diff {
  int max = 0;
  double over1 = 0;  // fraction of channels more than one level apart
};
Diff diff(const Bytes& a, const Bytes& b) {
  Diff d;
  long over = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const int e = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
    d.max = std::max(d.max, e);
    over += e > 1;
  }
  d.over1 = static_cast<double>(over) / static_cast<double>(a.size());
  return d;
}

std::uint64_t fnv(const Bytes& b) {
  std::uint64_t h = 1469598103934665603ULL;
  for (const std::uint8_t c : b) h = (h ^ c) * 1099511628211ULL;
  return h;
}

// The tests' small rollout effect (16 coarse cells, so sizes are multiples of 32) as an nvfx_effect.
struct Fx {
  nvfx_effect* e = nullptr;
  explicit Fx(std::uint64_t seed = 3) {
    const auto t = nfx::compose::testing::tiny_effect(seed);
    std::ostringstream os;
    EXPECT_TRUE(nfx::rollout::save_model(os, t.m));
    const std::string b = os.str();
    EXPECT_EQ(nvfx_effect_load_memory(b.data(), b.size(), &e), NVFX_OK);
  }
  ~Fx() { nvfx_effect_free(e); }
  Fx(const Fx&) = delete;
  Fx& operator=(const Fx&) = delete;
};
std::string net_bytes(const up::Net& n) {
  std::ostringstream os;
  EXPECT_TRUE(up::save_net(os, n));
  return os.str();
}
nvfx_instance* make_instance(nvfx_effect* e, int size, std::uint64_t seed) {
  nvfx_instance* in = nullptr;
  EXPECT_EQ(nvfx_instance_create(e, size, &in), NVFX_OK);
  const float ctl[3] = {0.6f, 0.4f, 0.7f};
  nvfx_instance_set_controls(in, ctl, 3);
  nvfx_instance_set_seed(in, seed);
  nvfx_instance_set_drift(in, 2.f);  // short shards: changes and crossfades within the tests' frames
  return in;
}

}  // namespace

// --- the format ------------------------------------------------------------------------------------------------------------

TEST(Upscale, FormatRoundTripsAndRejectsBadFiles) {
  for (const int k2 : {1, 3}) {
    const up::Net n = test_net(6, 5, k2);
    const std::string b = net_bytes(n);
    auto r = up::parse_net(std::span(b.data(), b.size()));
    ASSERT_TRUE(r) << r.error();
    EXPECT_EQ(r->c1, 6);
    EXPECT_EQ(r->c2, 5);
    EXPECT_EQ(r->k2, k2);
    EXPECT_EQ(r->gain, n.gain);
    EXPECT_EQ(r->rate, n.rate);
    EXPECT_EQ(r->tag, "test");
    EXPECT_EQ(r->w, n.w);
    EXPECT_EQ(r->w.size(), r->layout().size);
  }
  const std::string good = net_bytes(test_net());
  EXPECT_FALSE(up::parse_net(std::span(good.data(), 7)));                     // truncated magic
  EXPECT_FALSE(up::parse_net(std::span(good.data(), good.size() - 1)));       // truncated weights
  std::string bad = good;
  bad[0] = 'X';
  EXPECT_FALSE(up::parse_net(std::span(bad.data(), bad.size())));             // magic
  bad = good;
  bad[8] = 2;
  EXPECT_FALSE(up::parse_net(std::span(bad.data(), bad.size())));             // version
  bad = good;
  const std::int32_t k2 = 2;
  std::memcpy(bad.data() + 20, &k2, 4);
  EXPECT_FALSE(up::parse_net(std::span(bad.data(), bad.size())));             // kernel
  bad = good;
  const float nan = std::nanf("");
  std::memcpy(bad.data() + bad.size() - 4, &nan, 4);
  EXPECT_FALSE(up::parse_net(std::span(bad.data(), bad.size())));             // weights not finite
}

// --- the reference ----------------------------------------------------------------------------------------------------------

TEST(Upscale, BilinearIsTheSamplersAndEmptyStaysEmpty) {
  const int n = 32;
  const Bytes lo = test_frame(n);
  std::vector<float> f(sz(2 * n) * sz(2 * n) * 4);
  up::bilinear2x(lo, n, f);
  // interior: 9/16, 3/16, 3/16, 1/16 of the four nearest half-size pixels; edges clamped
  const auto L = [&](int x, int y, int c) { return static_cast<float>(lo[(sz(std::clamp(y, 0, n - 1)) * sz(n) + sz(std::clamp(x, 0, n - 1))) * 4 + sz(c)]) / 255.f; };
  for (const auto& [X, Y] : std::vector<std::pair<int, int>>{{0, 0}, {13, 20}, {14, 21}, {63, 63}, {40, 1}}) {
    const int xi = X / 2, yi = Y / 2, xj = X % 2 ? xi + 1 : xi - 1, yj = Y % 2 ? yi + 1 : yi - 1;
    for (int c = 0; c < 4; ++c) {
      const float want = 0.5625f * L(xi, yi, c) + 0.1875f * (L(xj, yi, c) + L(xi, yj, c)) + 0.0625f * L(xj, yj, c);
      EXPECT_NEAR(f[(sz(Y) * sz(2 * n) + sz(X)) * 4 + sz(c)], want, 1e-6f);
    }
  }
  // the learned upscale: where the bilinear is empty in every channel the output is exactly zero
  const up::Net net = test_net();
  Bytes hi(sz(2 * n) * sz(2 * n) * 4);
  up::upscale(&net, lo, n, 42, 1.3, hi.data(), sz(2 * n) * 4);
  int empty = 0, drawn = 0;
  for (std::size_t p = 0; p < f.size() / 4; ++p) {
    const bool z = f[p * 4] == 0.f && f[p * 4 + 1] == 0.f && f[p * 4 + 2] == 0.f && f[p * 4 + 3] == 0.f;
    if (!z) {
      drawn += hi[p * 4 + 3] > 0;
      continue;
    }
    ++empty;
    for (int c = 0; c < 4; ++c) EXPECT_EQ(hi[p * 4 + sz(c)], 0);
  }
  EXPECT_GT(empty, 1000);
  EXPECT_GT(drawn, 1000);
}

TEST(Upscale, NoiseIsSeededSmoothInTimeAndHasNoLowFrequencies) {
  const int N = 128;
  std::vector<float> a(sz(N) * sz(N)), b(a.size()), c(a.size()), d(a.size());
  up::noise_frame(7, 2.0, 8.f, N, a);
  up::noise_frame(7, 2.0, 8.f, N, b);
  up::noise_frame(8, 2.0, 8.f, N, c);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
  // unit variance, zero mean
  double s = 0, s2 = 0;
  for (const float v : a) {
    s += v;
    s2 += static_cast<double>(v) * v;
  }
  const double m = s / static_cast<double>(a.size()), var = s2 / static_cast<double>(a.size()) - m * m;
  EXPECT_NEAR(m, 0.0, 0.03);
  EXPECT_NEAR(var, 1.0, 0.05);
  // at a slice boundary the noise is that slice; between two slices the variance stays near 1
  const up::NoiseBlend at = up::noise_blend(2.0, 8.f), mid = up::noise_blend(2.0 + 0.5 / 8.0, 8.f);
  EXPECT_EQ(at.s, 0.f);
  EXPECT_NEAR(mid.s, 0.5f, 1e-5f);
  for (int y = 0; y < 4; ++y) EXPECT_FLOAT_EQ(a[sz(y) * sz(N) + 3], up::noise_slice(7, 3, y, at.k));
  up::noise_frame(7, 2.0 + 0.5 / 8.0, 8.f, N, d);
  double s3 = 0;
  for (const float v : d) s3 += static_cast<double>(v) * v;
  EXPECT_NEAR(s3 / static_cast<double>(d.size()), 1.0, 0.06);
  // a frame later it has moved a little, not jumped (8 slices a second at 30 frames a second)
  up::noise_frame(7, 2.0 + 1.0 / 30.0, 8.f, N, b);
  double dd = 0;
  for (std::size_t i = 0; i < a.size(); ++i) dd += (a[i] - b[i]) * (a[i] - b[i]);
  EXPECT_LT(dd / static_cast<double>(a.size()), 0.5);
  // its 4 x 4 means (the frequencies a half-size frame draws) hold little of its energy
  double low = 0;
  for (int y = 0; y + 4 <= N; y += 4) {
    for (int x = 0; x + 4 <= N; x += 4) {
      double q = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) q += a[sz(y + j) * sz(N) + sz(x + i)];
      }
      low += (q / 16) * (q / 16);
    }
  }
  EXPECT_LT(low / (N / 4 * N / 4), 0.02);
}

// --- the runtime against the reference ------------------------------------------------------------------------------------------

TEST(Upscale, RuntimeMatchesTheReferenceOnEveryIsa) {
  // The runtime keeps the reference's operations in its order per value; the AVX2 and AVX-512 builds contract some
  // multiply-adds, so a byte may land one level away on a rounding boundary. Measured: at most 1 level, in under 0.1% of
  // the channels; allowed: 2 levels, 0.5% of the channels more than one apart (none on the baseline build).
  const int n = 64;
  const Bytes lo = test_frame(n, 9);
  const std::array<float, 9> colour = {0.9f, 0.1f, 0.f, 0.05f, 0.8f, 0.1f, 0.f, 0.2f, 1.1f};
  for (const int k2 : {1, 3}) {
    const up::Net net = test_net(8, 6, k2, 21);
    for (const bool with_colour : {false, true}) {
      for (const up::Net* p : {static_cast<const up::Net*>(nullptr), &net}) {
        for (const double t : {0.0, 0.37, 5.2}) {
          Bytes ref(sz(2 * n) * sz(2 * n) * 4), got(ref.size());
          up::upscale(p, lo, n, 1234, t, ref.data(), sz(2 * n) * 4, with_colour ? colour.data() : nullptr);
          for (const IsaUp& isa : isas()) {
            auto u = isa.make(p, n);
            rt::FrameInput fi;
            fi.colour = colour;
            fi.apply_colour = with_colour;
            u->run(lo.data(), sz(n) * 4, 1234, t, fi, got.data(), sz(2 * n) * 4);
            const Diff d = diff(ref, got);
            SCOPED_TRACE(testing::Message() << "isa " << isa.isa << " k2 " << k2 << " colour " << with_colour << " net " << (p != nullptr) << " t " << t);
            if (isa.isa == NVFX_ISA_BASELINE && !p) {
              EXPECT_EQ(d.max, 0);
            }
            EXPECT_LE(d.max, 2);
            EXPECT_LE(d.over1, 0.005);
          }
        }
      }
    }
  }
}

TEST(Upscale, RuntimeKeepsNoiseSlicesAcrossFramesAndSeeds) {
  // The cached slices give the same frames as a fresh upscaler, whatever order the times and seeds come in.
  const int n = 32;
  const Bytes lo = test_frame(n, 3);
  const up::Net net = test_net();
  for (const IsaUp& isa : isas()) {
    auto kept = isa.make(&net, n);
    const rt::FrameInput fi;
    for (const auto& [seed, t] : std::vector<std::pair<std::uint64_t, double>>{{1, 0.0}, {1, 0.04}, {1, 0.13}, {1, 0.2}, {2, 0.2}, {1, 0.2}, {1, 7.0}, {1, 0.01}}) {
      Bytes a(sz(2 * n) * sz(2 * n) * 4), b(a.size());
      kept->run(lo.data(), sz(n) * 4, seed, t, fi, a.data(), sz(2 * n) * 4);
      isa.make(&net, n)->run(lo.data(), sz(n) * 4, seed, t, fi, b.data(), sz(2 * n) * 4);
      EXPECT_EQ(a, b) << "isa " << isa.isa << " seed " << seed << " t " << t;
    }
  }
}

// --- the C API --------------------------------------------------------------------------------------------------------------------

TEST(Upscale, InstanceDrawsTheHalfSizeFrameUpscaled) {
  // An instance with NVFX_UPSCALE_LEARNED draws exactly what an instance at half its size draws, upscaled by the runtime's
  // upscaler with the instance's seed, the frame's time and its colour; with NVFX_UPSCALE_BILINEAR the bilinear 2x.
  Fx fx;
  const up::Net net = test_net();
  const std::string b = net_bytes(net);
  ASSERT_EQ(nvfx_effect_attach_upscaler_memory(fx.e, b.data(), b.size()), NVFX_OK);
  for (const nvfx_upscale mode : {NVFX_UPSCALE_BILINEAR, NVFX_UPSCALE_LEARNED}) {
    nvfx_instance* full = make_instance(fx.e, 128, 77);
    nvfx_instance* half = make_instance(fx.e, 64, 77);
    ASSERT_EQ(nvfx_instance_set_upscale(full, mode), NVFX_OK);
    ASSERT_EQ(nvfx_instance_set_colour(full, 0.4f, 1.2f), NVFX_OK);
    ASSERT_EQ(nvfx_instance_set_colour(half, 0.f, 1.f), NVFX_OK);
    std::array<float, 9> colour{};
    {  // the instance's colour matrix (nvfx_instance_set_colour: a rotation about the grey axis, times the brightness)
      const float c = std::cos(0.4f), s = std::sin(0.4f), k = (1.f - c) / 3.f, r = 0.57735026919f * s;
      colour = {c + k, k - r, k + r, k + r, c + k, k - r, k - r, k + r, c + k};
      for (float& v : colour) v *= 1.2f;
    }
    auto u = rt::isa_base::make_upscaler(nullptr, 64);  // replaced below by the instance's ISA's
    const nvfx_isa isa = rt::rollout_instance_isa(half);
    u = isa == NVFX_ISA_AVX512 ? rt::isa_avx512::make_upscaler(mode == NVFX_UPSCALE_LEARNED ? &net : nullptr, 64)
        : isa == NVFX_ISA_AVX2 ? rt::isa_avx2::make_upscaler(mode == NVFX_UPSCALE_LEARNED ? &net : nullptr, 64)
                               : rt::isa_base::make_upscaler(mode == NVFX_UPSCALE_LEARNED ? &net : nullptr, 64);
    Bytes a(128 * 128 * 4), h(64 * 64 * 4), want(a.size());
    rt::FrameInput fi;
    fi.colour = colour;
    fi.apply_colour = true;
    for (int f = 0; f < 140; f += 3) {  // shard changes every 60 frames, crossfades
      const double t = f / 30.0;
      ASSERT_EQ(nvfx_render(full, t, a.data(), 128 * 4), NVFX_OK);
      ASSERT_EQ(nvfx_render(half, t, h.data(), 64 * 4), NVFX_OK);
      u->run(h.data(), 64 * 4, 77, static_cast<double>(f) / 30.0, fi, want.data(), 128 * 4);
      ASSERT_EQ(a, want) << "mode " << mode << " frame " << f;
    }
    nvfx_instance_free(full);
    nvfx_instance_free(half);
  }
}

TEST(Upscale, InstancesAreDeterministicAndSeekable) {
  Fx fx;
  const std::string b = net_bytes(test_net());
  ASSERT_EQ(nvfx_effect_attach_upscaler_memory(fx.e, b.data(), b.size()), NVFX_OK);
  nvfx_instance* x = make_instance(fx.e, 128, 5);
  nvfx_instance* y = make_instance(fx.e, 128, 5);
  nvfx_instance* z = make_instance(fx.e, 128, 6);
  for (nvfx_instance* in : {x, y, z}) ASSERT_EQ(nvfx_instance_set_upscale(in, NVFX_UPSCALE_LEARNED), NVFX_OK);
  Bytes a(128 * 128 * 4), c(a.size()), d(a.size());
  std::vector<Bytes> seq;
  for (int f = 0; f < 100; ++f) {
    ASSERT_EQ(nvfx_render(x, f / 30.0, a.data(), 128 * 4), NVFX_OK);
    ASSERT_EQ(nvfx_render(y, f / 30.0, c.data(), 128 * 4), NVFX_OK);
    ASSERT_EQ(nvfx_render(z, f / 30.0, d.data(), 128 * 4), NVFX_OK);
    EXPECT_EQ(a, c) << "frame " << f;
    if (f > 5) {
      EXPECT_NE(a, d) << "frame " << f;  // another seed, another run and other micro-detail
    }
    seq.push_back(a);
  }
  // seeking: backwards and forwards give the frames of a straight play
  for (const int f : {80, 10, 99, 3, 61}) {
    ASSERT_EQ(nvfx_render(y, f / 30.0, c.data(), 128 * 4), NVFX_OK);
    EXPECT_EQ(c, seq[sz(f)]) << "frame " << f;
  }
  // turned off and on: the timeline starts again and plays the same frames
  ASSERT_EQ(nvfx_instance_set_upscale(y, NVFX_UPSCALE_OFF), NVFX_OK);
  ASSERT_EQ(nvfx_render(y, 0.5, c.data(), 128 * 4), NVFX_OK);
  EXPECT_NE(c, seq[15]);
  ASSERT_EQ(nvfx_instance_set_upscale(y, NVFX_UPSCALE_LEARNED), NVFX_OK);
  ASSERT_EQ(nvfx_render(y, 0.5, c.data(), 128 * 4), NVFX_OK);
  EXPECT_EQ(c, seq[15]);
  // baked flipbooks too
  Bytes bake(sz(128) * 128 * 4 * 12), bake2(bake.size());
  ASSERT_EQ(nvfx_bake(x, 12, bake.data()), NVFX_OK);
  ASSERT_EQ(nvfx_bake(y, 12, bake2.data()), NVFX_OK);
  EXPECT_EQ(bake, bake2);
  for (nvfx_instance* in : {x, y, z}) nvfx_instance_free(in);
}

TEST(Upscale, ErrorsAndMemory) {
  Fx fx;
  const std::string b = net_bytes(test_net());
  EXPECT_EQ(nvfx_effect_attach_upscaler(nullptr, "x"), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_effect_attach_upscaler(fx.e, nullptr), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_effect_attach_upscaler(fx.e, "/nonexistent/p.nvfxup"), NVFX_ERROR_IO);
  EXPECT_EQ(nvfx_effect_attach_upscaler_memory(fx.e, "garbage!garbage!", 16), NVFX_ERROR_FORMAT);
  EXPECT_EQ(nvfx_effect_attach_upscaler_memory(fx.e, nullptr, 3), NVFX_ERROR_ARGUMENT);
  nvfx_instance* in = make_instance(fx.e, 128, 1);
  EXPECT_EQ(nvfx_instance_set_upscale(in, NVFX_UPSCALE_LEARNED), NVFX_ERROR_ARGUMENT);  // nothing attached yet
  EXPECT_EQ(nvfx_instance_set_upscale(in, static_cast<nvfx_upscale>(3)), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_instance_set_upscale(nullptr, NVFX_UPSCALE_BILINEAR), NVFX_ERROR_ARGUMENT);
  nvfx_effect_info info{};
  ASSERT_EQ(nvfx_effect_get_info(fx.e, &info), NVFX_OK);
  const std::size_t resident = info.resident_bytes, before = nvfx_instance_scratch_bytes(in);
  ASSERT_EQ(nvfx_effect_attach_upscaler_memory(fx.e, b.data(), b.size()), NVFX_OK);
  EXPECT_EQ(nvfx_effect_attach_upscaler_memory(fx.e, b.data(), b.size()), NVFX_ERROR_ARGUMENT);  // once
  ASSERT_EQ(nvfx_effect_get_info(fx.e, &info), NVFX_OK);
  EXPECT_EQ(info.resident_bytes, resident + test_net().w.size() * 4);
  // an instance made before the attach can switch to it; it then holds half-size runners, so it needs less memory
  ASSERT_EQ(nvfx_instance_set_upscale(in, NVFX_UPSCALE_LEARNED), NVFX_OK);
  EXPECT_LT(nvfx_instance_scratch_bytes(in), before);
  ASSERT_EQ(nvfx_instance_set_upscale(in, NVFX_UPSCALE_OFF), NVFX_OK);
  EXPECT_EQ(nvfx_instance_scratch_bytes(in), before);
  nvfx_instance_free(in);
  // sizes: twice the coarse grid (32 for the tests' 16-cell effect)
  nvfx_instance* odd = make_instance(fx.e, 80, 1);
  EXPECT_EQ(nvfx_instance_set_upscale(odd, NVFX_UPSCALE_BILINEAR), NVFX_ERROR_UNSUPPORTED);
  nvfx_instance_free(odd);
  nvfx_instance* small = make_instance(fx.e, 32, 1);
  EXPECT_EQ(nvfx_instance_set_upscale(small, NVFX_UPSCALE_BILINEAR), NVFX_OK);
  Bytes px(32 * 32 * 4);
  EXPECT_EQ(nvfx_render(small, 0.3, px.data(), 32 * 4), NVFX_OK);
  nvfx_instance_free(small);
}

TEST(Upscale, NoAllocationPerFrame) {
  Fx fx;
  const std::string b = net_bytes(test_net(8, 8, 3));
  ASSERT_EQ(nvfx_effect_attach_upscaler_memory(fx.e, b.data(), b.size()), NVFX_OK);
  for (const nvfx_upscale mode : {NVFX_UPSCALE_BILINEAR, NVFX_UPSCALE_LEARNED}) {
    nvfx_instance* in = make_instance(fx.e, 128, 9);
    ASSERT_EQ(nvfx_instance_set_upscale(in, mode), NVFX_OK);
    ASSERT_EQ(nvfx_instance_set_colour(in, 0.2f, 0.9f), NVFX_OK);
    Bytes px(128 * 128 * 4);
    ASSERT_EQ(nvfx_render(in, 0.0, px.data(), 128 * 4), NVFX_OK);
    g_allocations.store(0);
    g_counting.store(true);
    for (int f = 1; f < 200; ++f) nvfx_render(in, f / 30.0, px.data(), 128 * 4);  // shards, crossfades, noise slices
    nvfx_render(in, 1.0, px.data(), 128 * 4);                                      // a seek back
    nvfx_instance_set_seed(in, 10);
    nvfx_render(in, 1.1, px.data(), 128 * 4);  // a new seed: new noise slices
    g_counting.store(false);
    EXPECT_EQ(g_allocations.load(), 0) << "mode " << mode;
    nvfx_instance_free(in);
  }
}

TEST(Upscale, FrozenFramesOnAvx2AtFloat) {
  // The tests' small effect and a fixed net, played at float on the AVX2 build (pinned, as the other frozen tests are), with
  // a colour, shards and crossfades: the frames' FNV-1a hashes, frozen when this test was written (15 October 2026). A
  // change to the upscaler, its noise or the instance's path changes them.
  if (nvfx_set_isa(NVFX_ISA_AVX2) != NVFX_OK) GTEST_SKIP() << "no AVX2";
  Fx fx;
  const std::string b = net_bytes(test_net(8, 8, 1, 31));
  ASSERT_EQ(nvfx_effect_attach_upscaler_memory(fx.e, b.data(), b.size()), NVFX_OK);
  const std::array<std::pair<nvfx_upscale, std::array<std::uint64_t, 4>>, 2> frozen = {{
      {NVFX_UPSCALE_LEARNED, {0x65a2dc09b4a0d627ULL, 0x4f3f67b1e041a015ULL, 0x15ed9451605f6ca6ULL, 0xd4d45d0624327b42ULL}},
      {NVFX_UPSCALE_BILINEAR, {0x4ca7f1238f5cf9ceULL, 0x4a8cdcd83447eaf1ULL, 0xb0f2c0104352d43aULL, 0xac1d085ca557005dULL}},
  }};
  for (const auto& [mode, want] : frozen) {
    nvfx_instance* in = make_instance(fx.e, 128, 2024);
    ASSERT_EQ(nvfx_instance_set_precision(in, NVFX_PRECISION_FLOAT), NVFX_OK);
    ASSERT_EQ(nvfx_instance_set_upscale(in, mode), NVFX_OK);
    ASSERT_EQ(nvfx_instance_set_colour(in, 0.3f, 1.1f), NVFX_OK);
    Bytes px(128 * 128 * 4);
    const std::array<int, 4> at = {0, 37, 75, 119};
    for (std::size_t k = 0; k < at.size(); ++k) {
      ASSERT_EQ(nvfx_render(in, at[k] / 30.0, px.data(), 128 * 4), NVFX_OK);
      EXPECT_EQ(fnv(px), want[k]) << "mode " << mode << " frame " << at[k] << ": 0x" << std::hex << fnv(px) << "ULL";
    }
    nvfx_instance_free(in);
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
}

TEST(Upscale, TheCppWrapperCallsTheCApi) {
  const auto t = nfx::compose::testing::tiny_effect(3);
  std::ostringstream os;
  ASSERT_TRUE(nfx::rollout::save_model(os, t.m));
  const std::string mb = os.str(), nb = net_bytes(test_net());
  const auto bytes = [](const std::string& s) { return std::span(reinterpret_cast<const std::uint8_t*>(s.data()), s.size()); };
  auto e = nvfx::Effect::load_memory(bytes(mb));
  ASSERT_TRUE(e);
  EXPECT_FALSE(e->attach_upscaler("/nonexistent/p.nvfxup"));
  ASSERT_TRUE(e->attach_upscaler_memory(bytes(nb)));
  EXPECT_EQ(e->attach_upscaler_memory(bytes(nb)).error().status, NVFX_ERROR_ARGUMENT);
  auto in = nvfx::Instance::create(*e, 128);
  ASSERT_TRUE(in);
  ASSERT_TRUE(in->set_seed(4));
  ASSERT_TRUE(in->set_upscale(NVFX_UPSCALE_LEARNED));
  EXPECT_EQ(in->set_upscale(static_cast<nvfx_upscale>(3)).error().status, NVFX_ERROR_ARGUMENT);
  Fx fx;  // the same effect through the C API
  ASSERT_EQ(nvfx_effect_attach_upscaler_memory(fx.e, nb.data(), nb.size()), NVFX_OK);
  nvfx_instance* c = nullptr;
  ASSERT_EQ(nvfx_instance_create(fx.e, 128, &c), NVFX_OK);
  nvfx_instance_set_seed(c, 4);
  ASSERT_EQ(nvfx_instance_set_upscale(c, NVFX_UPSCALE_LEARNED), NVFX_OK);
  Bytes a(128 * 128 * 4), b(a.size());
  for (const double s : {0.0, 0.7, 2.1}) {
    ASSERT_TRUE(in->render(s, a, 128 * 4));
    ASSERT_EQ(nvfx_render(c, s, b.data(), 128 * 4), NVFX_OK);
    EXPECT_EQ(a, b);
  }
  nvfx_instance_free(c);
}
