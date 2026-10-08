// Phase 1a: clips, noise and the fluid simulation.
#include <neuralfx/clip.hpp>
#include <neuralfx/image_io.hpp>
#include <neuralfx/noise.hpp>
#include <neuralfx/sim.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <numeric>
#include <unistd.h>

using namespace nfx;

namespace {

sim::Params small(sim::Effect e) {
  sim::Params p;
  p.effect = e;
  p.size = 48;
  p.frames = 12;
  p.warmup = 20;
  p.loop_blend = 4;
  p.pressure_iters = 12;
  return p;
}

double mean_abs_diff(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); ++i) s += std::abs(int(a[i]) - int(b[i]));
  return s / static_cast<double>(a.size());
}

std::filesystem::path temp_path(std::string_view name) {
  return std::filesystem::temp_directory_path() / std::format("nfx_test_{}_{}", ::getpid(), name);
}

}  // namespace

TEST(Clip, RoundTripKeepsPixelsAndMetadata) {
  Clip c;
  c.allocate(16, 3);
  for (std::size_t i = 0; i < c.rgba.size(); ++i) c.rgba[i] = static_cast<std::uint8_t>(i * 7);
  c.effect = "fire";
  c.source = "sim";
  c.loop = true;
  c.fps = 24.f;
  c.n_controls = 3;
  c.controls[1] = 0.25f;
  c.seed = 0x1234567890abcdefULL;
  const auto path = temp_path("clip.nfxclip");
  ASSERT_TRUE(write_clip(path, c).has_value());
  const auto r = read_clip(path);
  ASSERT_TRUE(r.has_value()) << r.error();
  EXPECT_EQ(r->rgba, c.rgba);
  EXPECT_EQ(r->effect, "fire");
  EXPECT_EQ(r->source, "sim");
  EXPECT_TRUE(r->loop);
  EXPECT_EQ(r->fps, 24.f);
  EXPECT_EQ(r->n_controls, 3);
  EXPECT_EQ(r->controls[1], 0.25f);
  EXPECT_EQ(r->seed, c.seed);
  const std::uint8_t last = (*r)[2, 15, 15, 3];  // multidimensional subscript: frame, y, x, channel
  EXPECT_EQ(last, c.rgba.back());
  std::filesystem::remove(path);
}

TEST(Clip, ReadErrorsAreValuesNotExceptions) {
  EXPECT_FALSE(read_clip("/nonexistent/clip.nfxclip").has_value());
  const auto path = temp_path("junk.nfxclip");
  { std::ofstream(path) << "not a clip"; }
  const auto r = read_clip(path);
  ASSERT_FALSE(r.has_value());
  EXPECT_NE(r.error().find("not a clip"), std::string::npos);
  std::filesystem::remove(path);
}

TEST(Clip, SliceCopiesTheRightFrames) {
  Clip c;
  c.allocate(4, 5);
  c.loop = true;
  for (int f = 0; f < 5; ++f) std::ranges::fill(c.frame(f), static_cast<std::uint8_t>(f));
  const Clip s = slice_clip(c, 1, 3);
  EXPECT_EQ(s.frames, 3);
  EXPECT_FALSE(s.loop);
  EXPECT_EQ(s.frame(0)[0], 1);
  EXPECT_EQ(s.frame(2)[0], 3);
  EXPECT_THROW(slice_clip(c, 4, 2), std::invalid_argument);
}

TEST(Noise, DeterministicBoundedAndSeedDependent) {
  float lo = 1, hi = -1;
  for (int i = 0; i < 1000; ++i) {
    const float x = 0.37f * static_cast<float>(i), v = value_noise(x, 0.5f * x, 0.1f * x, 9);
    lo = std::min(lo, v);
    hi = std::max(hi, v);
    EXPECT_EQ(v, value_noise(x, 0.5f * x, 0.1f * x, 9));
  }
  EXPECT_GE(lo, -1.f);
  EXPECT_LE(hi, 1.f);
  EXPECT_LT(lo, -0.5f);
  EXPECT_GT(hi, 0.5f);
  EXPECT_NE(value_noise(1.3f, 2.7f, 0.2f, 1), value_noise(1.3f, 2.7f, 0.2f, 2));
}

TEST(Sim, FieldSamplesBilinearly) {
  sim::Field f(4);
  f[2, 2] = 1.f;
  EXPECT_FLOAT_EQ(f.sample(2.f, 2.f), 1.f);
  EXPECT_FLOAT_EQ(f.sample(2.5f, 2.f), 0.5f);
  EXPECT_FLOAT_EQ(f.sample(2.5f, 2.5f), 0.25f);
  EXPECT_FLOAT_EQ(f.sample(-10.f, -10.f), 0.f);  // clamped to the border
}

TEST(Sim, SimulationIsDeterministic) {
  for (const sim::Effect e : sim::kEffects) {
    const auto p = small(e);
    EXPECT_EQ(sim::simulate(p).rgba, sim::simulate(p).rgba) << sim::effect_name(e);
  }
}

TEST(Sim, EveryEffectDrawsSomething) {
  for (const sim::Effect e : sim::kEffects) {
    const Clip c = sim::simulate(small(e));
    const auto lit = std::ranges::count_if(c.rgba, [](std::uint8_t v) { return v > 16; });
    EXPECT_GT(lit, static_cast<long>(c.rgba.size() / 100)) << sim::effect_name(e);
    EXPECT_EQ(c.loop, sim::effect_loops(e));
    EXPECT_EQ(c.n_controls, sim::kControls);
  }
}

TEST(Sim, SeedAndEveryControlChangeTheOutput) {
  const auto p = small(sim::Effect::fire);
  const Clip base = sim::simulate(p);
  auto q = p;
  q.seed = 2;
  EXPECT_GT(mean_abs_diff(base.rgba, sim::simulate(q).rgba), 0.5);
  for (const int k : {0, 1, 2}) {
    q = p;
    float* c[] = {&q.intensity, &q.wind, &q.turbulence};
    *c[k] = 1.f;
    EXPECT_GT(mean_abs_diff(base.rgba, sim::simulate(q).rgba), 0.1) << sim::kControlNames[k];  // measurable, not large
  }
}

TEST(Sim, FireIsMostlyEmissionAndSmokeIsPremultiplied) {
  const Clip fire = sim::simulate(small(sim::Effect::fire));
  const Clip smoke = sim::simulate(small(sim::Effect::smoke));
  long emissive = 0, lit = 0, bad = 0;
  for (std::size_t i = 0; i < fire.rgba.size(); i += 4) {
    if (fire.rgba[i] > 64) {
      ++lit;
      emissive += fire.rgba[i] > fire.rgba[i + 3];
    }
  }
  for (std::size_t i = 0; i < smoke.rgba.size(); i += 4) {
    for (int c = 0; c < 3; ++c) bad += smoke.rgba[i + c] > smoke.rgba[i + 3] + 1;
  }
  EXPECT_GT(lit, 0);
  EXPECT_GT(emissive, lit / 2);  // additive light: colour above coverage
  EXPECT_EQ(bad, 0);             // smoke reflects light only: colour <= alpha
}

TEST(Sim, LoopSeamIsNoWorseThanAnOrdinaryFrameStep) {
  auto p = small(sim::Effect::smoke);
  p.frames = 24;
  p.loop_blend = 8;
  const Clip c = sim::simulate(p);
  double step = 0;
  for (int f = 1; f < c.frames; ++f) step += mean_abs_diff(c.frame(f - 1), c.frame(f));
  step /= c.frames - 1;
  EXPECT_LE(mean_abs_diff(c.frame(c.frames - 1), c.frame(0)), 1.5 * step + 0.5);
}

TEST(Sim, ParamsRoundTripThroughClipMetadata) {
  auto p = small(sim::Effect::explosion);
  p.intensity = 0.3f;
  p.wind = 0.9f;
  p.turbulence = 0.1f;
  p.seed = 42;
  const Clip c = sim::simulate(p);
  const auto q = sim::params_of(c);
  EXPECT_EQ(q.effect, p.effect);
  EXPECT_EQ(q.intensity, 0.3f);
  EXPECT_EQ(q.wind, 0.9f);
  EXPECT_EQ(q.turbulence, 0.1f);
  EXPECT_EQ(q.seed, 42u);
}

TEST(ImageIo, PngHasSignatureAndSheetHasExpectedSize) {
  const Clip c = sim::simulate(small(sim::Effect::fire));
  const Image sheet = contact_sheet(c, 4, 3, Background::black, 1);
  EXPECT_EQ(sheet.width, 4 * (48 + 2));
  EXPECT_EQ(sheet.height, 1 * (48 + 2));
  const auto path = temp_path("sheet.png");
  ASSERT_TRUE(write_png(path, sheet).has_value());
  std::ifstream in(path, std::ios::binary);
  char sig[8];
  in.read(sig, 8);
  EXPECT_EQ(std::string_view(sig + 1, 3), "PNG");
  std::filesystem::remove(path);
}
