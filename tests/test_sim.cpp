// Phase 1a: clips, noise and the fluid simulation.
#include <neuralfx/clip.hpp>
#include <neuralfx/image_io.hpp>
#include <neuralfx/noise.hpp>
#include <neuralfx/sim.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <numeric>
#include <span>
#include <utility>
#include <vector>
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

// Adding effects (docs/EFFECTS.md, docs/EFFECTS_ENERGY.md, docs/EFFECTS_ATMOSPHERE.md, docs/EFFECTS_BLAZE.md,
// docs/EFFECTS_FLOW.md, docs/EFFECTS_WATER.md) must not change the earlier ones: these hashes are of clips made before
// steam and magic existed (fire, smoke, explosion), before jet, orb, vortex, fog, dust and cloud existed (steam, magic),
// and before wildfire, coals and mushroom, launch, puffs and ink, and the water effects existed (jet, orb, vortex, fog,
// dust, cloud).
// FNV-1a of the RGBA bytes; the solver is built at the baseline ISA without contraction.
TEST(Sim, StudyEffectsAreUnchangedByTheNewOnes) {
  const auto fnv = [](std::span<const std::uint8_t> b) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const auto v : b) {
      h ^= v;
      h *= 1099511628211ULL;
    }
    return h;
  };
  // the first eleven's hashes as before; the ten after them (docs/EFFECTS_BLAZE.md, EFFECTS_FLOW.md, EFFECTS_WATER.md) made
  // before the air effects (docs/EFFECTS_AIR.md) were added
  const std::array<std::pair<sim::Effect, std::uint64_t>, 21> want{{{sim::Effect::fire, 0xecb25de8e62e7379ULL},
                                                                     {sim::Effect::smoke, 0xea602bfa2409bf3fULL},
                                                                     {sim::Effect::explosion, 0xbb823d31fd8c4eb4ULL},
                                                                     {sim::Effect::steam, 0xa57b2dc109882ec3ULL},
                                                                     {sim::Effect::magic, 0xf79b687a9614f399ULL},
                                                                     {sim::Effect::jet, 0x9cdcde02a8631bf6ULL},
                                                                     {sim::Effect::orb, 0x5fef6cca235e06f1ULL},
                                                                     {sim::Effect::vortex, 0xfa46f5fc13b9cb20ULL},
                                                                     {sim::Effect::fog, 0x36573f12c43055b2ULL},
                                                                     {sim::Effect::dust, 0xa20af83462f66930ULL},
                                                                     {sim::Effect::cloud, 0x8e6664474fe37d58ULL},
                                                                     {sim::Effect::wildfire, 0xad693da9553819c7ULL},
                                                                     {sim::Effect::coals, 0x6f038944bdd12910ULL},
                                                                     {sim::Effect::mushroom, 0x93f24f8ef0b28ae6ULL},
                                                                     {sim::Effect::launch, 0x8a85441c04182281ULL},
                                                                     {sim::Effect::puffs, 0xcff7f9fb95e106fdULL},
                                                                     {sim::Effect::ink, 0x46f9e4a0c8a075b4ULL},
                                                                     {sim::Effect::waterfall, 0x20ace39e7ed5b5b1ULL},
                                                                     {sim::Effect::fountain, 0x897564a37a1bf472ULL},
                                                                     {sim::Effect::splash, 0x4c634291b30f7d58ULL},
                                                                     {sim::Effect::rainshaft, 0xbc6f9c2a34332725ULL}}};
  for (const auto& [e, h] : want) {
    auto p = small(e);
    p.intensity = 0.8f;
    p.wind = 0.3f;
    p.turbulence = 0.7f;
    p.seed = 7;
    const std::uint64_t got = fnv(sim::simulate(p).rgba);
    EXPECT_EQ(got, h) << sim::effect_name(e) << std::format(": 0x{:016x}", got);
  }
}

TEST(Sim, NewEffectsAreNamedAndLoop) {
  for (const sim::Effect e : sim::kNewEffects) {
    sim::Effect back{};
    ASSERT_TRUE(sim::parse_effect(sim::effect_name(e), back));
    EXPECT_EQ(back, e);
    EXPECT_EQ(sim::effect_loops(e), e != sim::Effect::dust && e != sim::Effect::mushroom &&
                                       e != sim::Effect::splash && e != sim::Effect::smokering);  // they play once, as the explosion
    EXPECT_EQ(sim::simulate(small(e)).rgba, sim::simulate(small(e)).rgba) << sim::effect_name(e);
  }
  EXPECT_EQ(sim::kAllEffects.size(), sim::kEffects.size() + sim::kNewEffects.size());
  EXPECT_EQ(sim::control_names(sim::Effect::magic)[1], "spin");
  EXPECT_EQ(sim::control_names(sim::Effect::steam), sim::kControlNames);
  EXPECT_EQ(sim::control_names(sim::Effect::fire), sim::kControlNames);
}

TEST(Sim, SteamIsWhiteVapourAndMagicGivesBlueLight) {
  const Clip steam = sim::simulate(small(sim::Effect::steam));
  const Clip magic = sim::simulate(small(sim::Effect::magic));
  long bad = 0, white = 0, lit = 0, emissive = 0;
  double r = 0, b = 0;
  for (std::size_t i = 0; i < steam.rgba.size(); i += 4) {
    for (int c = 0; c < 3; ++c) bad += steam.rgba[i + c] > steam.rgba[i + 3] + 1;  // reflected light only
    if (steam.rgba[i + 3] > 64) white += steam.rgba[i] * 2 > steam.rgba[i + 3];   // and bright: more than half its cover
  }
  for (std::size_t i = 0; i < magic.rgba.size(); i += 4) {
    if (std::max({magic.rgba[i], magic.rgba[i + 1], magic.rgba[i + 2]}) > 64) {
      ++lit;
      emissive += magic.rgba[i + 2] > magic.rgba[i + 3];
      r += magic.rgba[i];
      b += magic.rgba[i + 2];
    }
  }
  EXPECT_EQ(bad, 0);
  EXPECT_GT(white, 50);
  EXPECT_GT(lit, 50);
  EXPECT_GT(emissive, lit / 2);  // additive light
  EXPECT_GT(b, 1.5 * r);         // violet to blue
}

TEST(Sim, MagicSpinsFasterWithItsSpinControl) {
  // angular momentum about the middle of the frame (counter-clockwise positive, y up)
  const auto momentum = [](float spin) {
    auto p = small(sim::Effect::magic);
    p.wind = spin;
    sim::Fluid f(p);
    for (int i = 0; i < 30; ++i) f.step_frame();
    const sim::State s = f.state();
    const float c = 0.5f * static_cast<float>(s.n - 1);
    double l = 0;
    for (int y = 0; y < s.n; ++y) {
      for (int x = 0; x < s.n; ++x) {
        const std::size_t i = static_cast<std::size_t>(y * s.n + x);
        l += (static_cast<float>(x) - c) * s.v[i] - (static_cast<float>(y) - c) * s.u[i];
      }
    }
    return l;
  };
  const double slow = momentum(0.f), fast = momentum(1.f);
  EXPECT_GT(slow, 0.0);
  EXPECT_GT(fast, 2.0 * slow);
}

// jet, orb and vortex (docs/EFFECTS_ENERGY.md): named, with their own second control, and drawn as what they are.
namespace {

sim::Params energy(sim::Effect e, float second, std::uint64_t seed = 3) {
  sim::Params p;
  p.effect = e;
  p.size = 64;
  p.wind = second;
  p.seed = seed;
  p.pressure_iters = 16;
  return p;
}

}  // namespace

TEST(Sim, JetOrbAndVortexHaveTheirOwnControls) {
  const std::array<std::pair<sim::Effect, std::string_view>, 3> second{{{sim::Effect::jet, "aim"}, {sim::Effect::orb, "pulse"}, {sim::Effect::vortex, "sway"}}};
  for (const auto& [e, name] : second) {
    sim::Effect back{};
    ASSERT_TRUE(sim::parse_effect(sim::effect_name(e), back));
    EXPECT_EQ(back, e);
    EXPECT_TRUE(sim::effect_loops(e));
    EXPECT_NE(std::ranges::find(sim::kNewEffects, e), sim::kNewEffects.end());
    EXPECT_FALSE(std::ranges::contains(sim::kEffects, e));
    const auto names = sim::control_names(e);
    EXPECT_EQ(names[0], "intensity");
    EXPECT_EQ(names[1], name);
    EXPECT_EQ(names[2], "turbulence");
  }
  EXPECT_TRUE(std::ranges::contains(sim::kAllEffects, sim::Effect::vortex));
}

TEST(Sim, JetIsAFlameFromTheLeftThatItsAimRaises) {
  // the jet's heat: where it is (mean x and y over the frame, cells from the bottom left) and its light
  const auto heat_centre = [](float aim) {
    sim::Fluid f(energy(sim::Effect::jet, aim));
    for (int i = 0; i < 40; ++i) f.step_frame();
    const sim::State s = f.state();
    double w = 0, x = 0, y = 0;
    for (int j = 0; j < s.n; ++j) {
      for (int i = 0; i < s.n; ++i) {
        const double t = s.temp[static_cast<std::size_t>(j * s.n + i)];
        w += t;
        x += t * i;
        y += t * j;
      }
    }
    return std::make_pair(x / w / s.n, y / w / s.n);
  };
  const auto [x_low, y_low] = heat_centre(0.f);
  const auto [x_high, y_high] = heat_centre(1.f);
  EXPECT_LT(x_low, 0.5);  // it comes from the left
  EXPECT_LT(x_high, 0.5);
  EXPECT_GT(y_high, y_low + 0.08);  // aimed higher
  const Clip c = sim::simulate(energy(sim::Effect::jet, 0.5f));
  double r = 0, b = 0;
  long lit = 0, smoke = 0;
  for (std::size_t i = 0; i < c.rgba.size(); i += 4) {
    if (std::max({c.rgba[i], c.rgba[i + 1], c.rgba[i + 2]}) > 96) {
      ++lit;
      r += c.rgba[i];
      b += c.rgba[i + 2];
    }
    smoke += c.rgba[i + 3] > 64 && c.rgba[i] < 48;  // dark smoke
  }
  EXPECT_GT(lit, 500);
  EXPECT_GT(r, 1.5 * b);  // fire colours
  EXPECT_GT(smoke, 100);
}

TEST(Sim, OrbIsAContainedBallOfCyanLightThatBreathesWithItsPulse) {
  // lit pixels per frame (more than a quarter of full brightness), and how far they reach from the middle
  const auto run = [](float pulse) {
    auto p = energy(sim::Effect::orb, pulse);
    p.frames = 90;
    const Clip c = sim::simulate(p);
    std::vector<double> area;
    long lit = 0, outside = 0;
    double r = 0, g = 0, b = 0;
    for (int k = 0; k < c.frames; ++k) {
      const auto fr = c.frame(k);
      double a = 0;
      for (int y = 0; y < c.size; ++y) {
        for (int x = 0; x < c.size; ++x) {
          const auto* q = &fr[static_cast<std::size_t>((y * c.size + x) * 4)];
          if (std::max({q[0], q[1], q[2]}) <= 64) continue;
          ++lit;
          ++a;
          r += q[0];
          g += q[1];
          b += q[2];
          outside += std::hypot(x - 31.5, y - 31.5) > 0.45 * c.size;
        }
      }
      area.push_back(a);
    }
    const double mean = std::accumulate(area.begin(), area.end(), 0.0) / static_cast<double>(area.size());
    double var = 0;
    for (const double a : area) var += (a - mean) * (a - mean) / static_cast<double>(area.size());
    return std::array<double, 7>{static_cast<double>(lit), static_cast<double>(outside), r, g, b, mean, std::sqrt(var) / mean};
  };
  const auto calm = run(0.f), strong = run(1.f);
  EXPECT_GT(calm[0], 2000);
  EXPECT_LT(calm[1], calm[0] / 50);  // contained
  EXPECT_GT(calm[4], 1.2 * calm[2]);  // blue and green over red: cyan
  EXPECT_GT(calm[3], 1.2 * calm[2]);
  EXPECT_GT(strong[6], 1.5 * calm[6]);  // the lit area swings more with a strong pulse
}

TEST(Sim, VortexIsAColumnOfDustThatSwaysWithItsControl) {
  // dust (soot) per row: its spread across the frame over time, in the upper half
  const auto run = [](float sway) {
    sim::Fluid f(energy(sim::Effect::vortex, sway, 5));
    for (int i = 0; i < 120; ++i) f.step_frame();
    std::vector<double> xs;
    double low = 0, high = 0;
    for (int i = 0; i < 90; ++i) {
      f.step_frame();
      const sim::State s = f.state();
      double w = 0, x = 0;
      for (int y = 0; y < s.n; ++y) {
        for (int k = 0; k < s.n; ++k) {
          const double d = s.soot[static_cast<std::size_t>(y * s.n + k)];
          (y < s.n / 2 ? low : high) += d;
          if (y >= s.n / 2) {
            w += d;
            x += d * k;
          }
        }
      }
      xs.push_back(x / w / s.n);
    }
    const double mean = std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
    double dev = 0;
    for (const double x : xs) dev += std::abs(x - mean) / static_cast<double>(xs.size());
    return std::array<double, 3>{dev, low, high};
  };
  const auto still = run(0.f), wandering = run(1.f);
  EXPECT_GT(still[2], 0.1 * still[1]);    // the dust reaches the upper half
  EXPECT_GT(wandering[0], 1.5 * still[0]);  // and wanders more with its sway
  const Clip c = sim::simulate(energy(sim::Effect::vortex, 0.5f));
  long bad = 0, dust = 0;
  double r = 0, b = 0;
  for (std::size_t i = 0; i < c.rgba.size(); i += 4) {
    for (int k = 0; k < 3; ++k) bad += c.rgba[i + k] > c.rgba[i + 3] + 1;  // reflected light only
    if (c.rgba[i + 3] > 64) {
      ++dust;
      r += c.rgba[i];
      b += c.rgba[i + 2];
    }
  }
  EXPECT_EQ(bad, 0);
  EXPECT_GT(dust, 1000);
  EXPECT_GT(r, 1.2 * b);  // brown-grey
}

// --- fog, dust and cloud (docs/EFFECTS_ATMOSPHERE.md) ---------------------------------------------------------------

namespace {

// Where the material (soot) is, as fractions of the frame (y up): its amount, centroid, and the extent of cells above
// a tenth of its peak.
struct Spread {
  double amount = 0, cx = 0, cy = 0, left = 1, right = 0, top = 0;
  double width() const { return right - left; }
};

Spread spread_of(const sim::State& s) {
  Spread o;
  float peak = 0.f;
  for (const float d : s.soot) peak = std::max(peak, d);
  const double n = s.n;
  for (int y = 0; y < s.n; ++y) {
    for (int x = 0; x < s.n; ++x) {
      const double d = s.soot[static_cast<std::size_t>(y * s.n + x)];
      o.amount += d;
      o.cx += d * (x + 0.5) / n;
      o.cy += d * (y + 0.5) / n;
      if (d > 0.1 * peak) {
        o.left = std::min(o.left, x / n);
        o.right = std::max(o.right, (x + 1) / n);
        o.top = std::max(o.top, (y + 1) / n);
      }
    }
  }
  if (o.amount > 0) {
    o.cx /= o.amount;
    o.cy /= o.amount;
  }
  return o;
}

sim::Params atmos_params(sim::Effect e, float intensity, float wind, float turbulence) {
  auto p = small(e);
  p.size = 64;
  p.intensity = intensity;
  p.wind = wind;
  p.turbulence = turbulence;
  return p;
}

// The spread after each of `frames` frames.
std::vector<Spread> spreads(const sim::Params& p, int frames) {
  sim::Fluid f(p);
  std::vector<Spread> v;
  for (int i = 0; i < frames; ++i) {
    f.step_frame();
    v.push_back(spread_of(f.state()));
  }
  return v;
}

}  // namespace

TEST(Sim, AtmosphereEffectsHaveNoHeatAndGiveNoLight) {
  for (const sim::Effect e : {sim::Effect::fog, sim::Effect::dust, sim::Effect::cloud}) {
    const auto name = sim::effect_name(e);
    EXPECT_EQ(sim::control_names(e), sim::kControlNames) << name;
    const Clip c = sim::simulate(atmos_params(e, 0.7f, 0.5f, 0.5f));
    EXPECT_EQ(c.loop, e != sim::Effect::dust) << name;
    long bad = 0, covered = 0;
    double r = 0, g = 0, b = 0, a = 0;
    for (std::size_t i = 0; i < c.rgba.size(); i += 4) {
      for (int k = 0; k < 3; ++k) bad += c.rgba[i + static_cast<std::size_t>(k)] > c.rgba[i + 3] + 1;  // reflected light only
      if (c.rgba[i + 3] > 64) {
        ++covered;
        r += c.rgba[i];
        g += c.rgba[i + 1];
        b += c.rgba[i + 2];
        a += c.rgba[i + 3];
      }
    }
    EXPECT_EQ(bad, 0) << name;
    ASSERT_GT(covered, 100) << name;
    if (e == sim::Effect::dust) {  // brown-grey: warmer than grey, and darker than the white vapours
      EXPECT_GT(r, 1.25 * b) << name;
      EXPECT_LT(r, 0.65 * a) << name;
    } else {  // pale: near grey, bright
      EXPECT_LT(std::abs(r - b), 0.1 * b) << name;
      EXPECT_GT(g, 0.6 * a) << name;
    }
    sim::Fluid f(atmos_params(e, 0.7f, 0.5f, 0.5f));
    for (int i = 0; i < 20; ++i) f.step_frame();
    const sim::State st = f.state();
    EXPECT_TRUE(std::ranges::all_of(st.temp, [](float t) { return t == 0.f; })) << name;  // no heat
  }
}

TEST(Sim, FogSinksAndSpreadsAlongTheGround) {
  const auto v = spreads(atmos_params(sim::Effect::fog, 0.6f, 0.5f, 0.4f), 150);
  EXPECT_LT(v[149].cy, 0.15);                 // below its release (a fifth of the frame up)
  EXPECT_LT(v[149].top, 0.4);                 // a layer on the ground
  EXPECT_GT(v[149].width(), 0.42);            // spread beyond the release (about 0.3 of the frame wide)
  EXPECT_GT(v[149].width(), v[29].width() + 0.1) << v[29].width();
  EXPECT_GT(v[149].amount, 0.5 * v[99].amount);  // released all the time
  // the wind control carries it to one side or the other
  const auto left = spreads(atmos_params(sim::Effect::fog, 0.6f, 0.f, 0.4f), 120);
  const auto right = spreads(atmos_params(sim::Effect::fog, 0.6f, 1.f, 0.4f), 120);
  EXPECT_LT(left[119].cx, 0.45);
  EXPECT_GT(right[119].cx, 0.55);
}

TEST(Sim, DustRisesSpreadsAndSettles) {
  const auto v = spreads(atmos_params(sim::Effect::dust, 0.6f, 0.5f, 0.5f), 90);  // its 3 s
  const auto highest = std::ranges::max_element(v, {}, &Spread::top);
  EXPECT_GT(highest->top, v[0].top + 0.05);  // thrown up a little
  EXPECT_LT(highest->top, 0.5);              // but stays low
  EXPECT_LT(v[89].top, highest->top);        // and settles
  EXPECT_LT(v[89].cy, 0.12);
  EXPECT_GT(v[89].width(), 2.5 * v[0].width());  // running out along the ground
  EXPECT_GT(v[89].width(), 0.6);
  EXPECT_LT(v[89].amount, std::ranges::max_element(v, {}, &Spread::amount)->amount);  // and thinning
  // more energy, a bigger burst
  const auto weak = spreads(atmos_params(sim::Effect::dust, 0.f, 0.5f, 0.5f), 30);
  const auto strong = spreads(atmos_params(sim::Effect::dust, 1.f, 0.5f, 0.5f), 30);
  EXPECT_GT(strong[29].amount, 2.0 * weak[29].amount);
  EXPECT_GT(strong[29].top, weak[29].top);
}

TEST(Sim, CloudStaysAliveAndDriftsWithTheWind) {
  const auto v = spreads(atmos_params(sim::Effect::cloud, 0.5f, 0.5f, 0.5f), 360);
  for (const int i : {149, 239, 359}) {
    EXPECT_GT(v[static_cast<std::size_t>(i)].amount, 0.8 * v[149].amount) << i;  // it does not fade away
    EXPECT_GT(v[static_cast<std::size_t>(i)].cy, 0.4) << i;                       // and stays in the middle of the frame
    EXPECT_LT(v[static_cast<std::size_t>(i)].cy, 0.65) << i;
    EXPECT_GT(v[static_cast<std::size_t>(i)].width(), 0.4) << i;
  }
  // more intensity, more cover
  const auto thin = spreads(atmos_params(sim::Effect::cloud, 0.f, 0.5f, 0.5f), 150);
  const auto thick = spreads(atmos_params(sim::Effect::cloud, 1.f, 0.5f, 0.5f), 150);
  EXPECT_GT(thick[149].amount, 2.0 * thin[149].amount);
  EXPECT_GT(thick[149].width(), thin[149].width() + 0.1);
  // the wind drifts it
  const auto left = spreads(atmos_params(sim::Effect::cloud, 0.5f, 0.f, 0.5f), 150);
  const auto right = spreads(atmos_params(sim::Effect::cloud, 0.5f, 1.f, 0.5f), 150);
  EXPECT_GT(right[149].cx, left[149].cx + 0.04);
}

// --- wildfire, coals and mushroom (docs/EFFECTS_BLAZE.md) ---------------------------------------------------------------

namespace {

// Where the heat and the soot are, as fractions of the frame (y up): amounts, centroids, and for the heat the share of
// columns between 12% and 88% of the width that burn in the bottom rows (below `low`), the height under which 98% of it
// lies, and its extent along x (cells above a tenth of its peak).
struct Blaze {
  double heat = 0, soot = 0, hx = 0, hy = 0, sx = 0, sy = 0, burning = 0, heat_top = 0, left = 1, right = 0;
};

Blaze blaze_of(const sim::State& s, double low = 0.12) {
  Blaze o;
  const double n = s.n;
  float peak = 0.f;
  for (const float t : s.temp) peak = std::max(peak, t);
  std::vector<double> rows(static_cast<std::size_t>(s.n), 0.0);
  int cols = 0, lit = 0;
  for (int x = 0; x < s.n; ++x) {
    double base = 0;
    for (int y = 0; y < s.n; ++y) {
      const auto i = static_cast<std::size_t>(y * s.n + x);
      const double t = s.temp[i], d = s.soot[i];
      o.heat += t;
      o.soot += d;
      o.hx += t * (x + 0.5) / n;
      o.hy += t * (y + 0.5) / n;
      o.sx += d * (x + 0.5) / n;
      o.sy += d * (y + 0.5) / n;
      rows[static_cast<std::size_t>(y)] += t;
      if ((y + 0.5) / n < low) base = std::max(base, t);
      if (t > 0.1 * peak) {
        o.left = std::min(o.left, x / n);
        o.right = std::max(o.right, (x + 1) / n);
      }
    }
    if ((x + 0.5) / n > 0.12 && (x + 0.5) / n < 0.88) {
      ++cols;
      lit += base > 0.15;
    }
  }
  o.burning = static_cast<double>(lit) / cols;
  double acc = 0;
  for (int y = 0; y < s.n; ++y) {
    acc += rows[static_cast<std::size_t>(y)];
    if (acc >= 0.98 * o.heat) {
      o.heat_top = (y + 1) / n;
      break;
    }
  }
  if (o.heat > 0) {
    o.hx /= o.heat;
    o.hy /= o.heat;
  }
  if (o.soot > 0) {
    o.sx /= o.soot;
    o.sy /= o.soot;
  }
  return o;
}

// The state after each of `frames` frames, at 64 cells.
std::vector<sim::State> blaze_run(sim::Effect e, float intensity, float wind, float turbulence, int frames) {
  sim::Fluid f(atmos_params(e, intensity, wind, turbulence));
  std::vector<sim::State> v;
  for (int i = 0; i < frames; ++i) {
    f.step_frame();
    v.push_back(f.state());
  }
  return v;
}

}  // namespace

TEST(Sim, BlazeEffectsAreNamedAndOnlyTheMushroomPlaysOnce) {
  const std::array blaze{sim::Effect::wildfire, sim::Effect::coals, sim::Effect::mushroom};
  for (const sim::Effect e : blaze) {
    sim::Effect back{};
    ASSERT_TRUE(sim::parse_effect(sim::effect_name(e), back));
    EXPECT_EQ(back, e);
    EXPECT_TRUE(std::ranges::contains(sim::kNewEffects, e));
    EXPECT_FALSE(std::ranges::contains(sim::kEffects, e));
    EXPECT_EQ(sim::control_names(e), sim::kControlNames);
  }
  EXPECT_EQ(sim::effect_name(sim::Effect::wildfire), "wildfire");
  EXPECT_EQ(sim::effect_name(sim::Effect::coals), "coals");
  EXPECT_EQ(sim::effect_name(sim::Effect::mushroom), "mushroom");
  // after cloud, in this order, before launch, puffs and ink
  const auto at = [](sim::Effect e) { return std::ranges::find(sim::kAllEffects, e) - sim::kAllEffects.begin(); };
  EXPECT_EQ(at(sim::Effect::wildfire), at(sim::Effect::cloud) + 1);
  EXPECT_EQ(at(sim::Effect::coals), at(sim::Effect::cloud) + 2);
  EXPECT_EQ(at(sim::Effect::mushroom), at(sim::Effect::cloud) + 3);
  EXPECT_LT(at(sim::Effect::mushroom), at(sim::Effect::launch));
  EXPECT_TRUE(sim::effect_loops(sim::Effect::wildfire));
  EXPECT_TRUE(sim::effect_loops(sim::Effect::coals));
  EXPECT_FALSE(sim::effect_loops(sim::Effect::mushroom));
  EXPECT_EQ(sim::one_shot_frames(sim::Effect::explosion), 90);
  EXPECT_EQ(sim::one_shot_frames(sim::Effect::dust), 90);
  EXPECT_EQ(sim::one_shot_frames(sim::Effect::mushroom), 150);
  EXPECT_EQ(sim::one_shot_frames(sim::Effect::fire), 0);
  EXPECT_EQ(sim::one_shot_frames(sim::Effect::wildfire), 0);
  // all three give light: emission (colour above alpha) somewhere
  for (const sim::Effect e : blaze) {
    auto p = atmos_params(e, 0.7f, 0.5f, 0.5f);
    p.frames = 20;
    const Clip c = sim::simulate(p);
    long glowing = 0;
    for (std::size_t i = 0; i < c.rgba.size(); i += 4) glowing += c.rgba[i] > c.rgba[i + 3] + 8;
    EXPECT_GT(glowing, 200) << sim::effect_name(e);
  }
}

TEST(Sim, WildfireBurnsAcrossTheWholeWidthWithSmokeAboveAndLeansWithTheWind) {
  const auto v = blaze_run(sim::Effect::wildfire, 0.6f, 0.5f, 0.5f, 150);
  double burning = 0;
  for (int i = 90; i < 150; ++i) {
    const Blaze b = blaze_of(v[static_cast<std::size_t>(i)]);
    burning += b.burning / 60.0;
    EXPECT_GT(b.sy, b.hy + 0.1) << i;  // the smoke rolls off above the flames
  }
  EXPECT_GT(burning, 0.85);  // the whole width burns, not one flame
  const Blaze late = blaze_of(v[149]);
  EXPECT_LT(late.left, 0.15);
  EXPECT_GT(late.right, 0.85);
  // more intensity, taller flames
  const auto low = blaze_run(sim::Effect::wildfire, 0.f, 0.5f, 0.5f, 120), high = blaze_run(sim::Effect::wildfire, 1.f, 0.5f, 0.5f, 120);
  double top_low = 0, top_high = 0;
  for (int i = 60; i < 120; ++i) {
    top_low += blaze_of(low[static_cast<std::size_t>(i)]).heat_top / 60.0;
    top_high += blaze_of(high[static_cast<std::size_t>(i)]).heat_top / 60.0;
  }
  EXPECT_GT(top_high, top_low + 0.1);
  // the wind leans the flames and carries the smoke to its side
  const auto left = blaze_run(sim::Effect::wildfire, 0.6f, 0.f, 0.5f, 120), right = blaze_run(sim::Effect::wildfire, 0.6f, 1.f, 0.5f, 120);
  EXPECT_GT(blaze_of(right[119]).sx, blaze_of(left[119]).sx + 0.15);
  EXPECT_GT(blaze_of(right[119]).hx, blaze_of(left[119]).hx + 0.03);
}

TEST(Sim, CoalsGlowLowAndWideAndTheirGlowCrawls) {
  const auto v = blaze_run(sim::Effect::coals, 0.6f, 0.5f, 0.5f, 240);
  double above = 0;
  for (int i = 120; i < 240; ++i) {
    const sim::State& s = v[static_cast<std::size_t>(i)];
    const Blaze b = blaze_of(s, 0.1);
    EXPECT_LT(b.hy, 0.12) << i;       // low: the glow stays in the bed
    EXPECT_LT(b.sy, 0.12) << i;       // and so does its char
    EXPECT_LT(b.heat_top, 0.3) << i;  // the flames are short
    EXPECT_LT(b.left, 0.2) << i;      // and wide
    EXPECT_GT(b.right, 0.8) << i;
    for (int y = s.n / 5; y < s.n; ++y) {
      for (int x = 0; x < s.n; ++x) above += s.temp[static_cast<std::size_t>(y * s.n + x)];
    }
  }
  EXPECT_GT(above, 1.0);  // small flames lick up from the bed now and then
  // the glow crawls: the bed's heat a second apart differs by a good part of itself
  const auto bed_change = [&](std::size_t a, std::size_t b) {
    double diff = 0, sum = 0;
    for (std::size_t i = 0; i < v[a].temp.size(); ++i) {
      diff += std::abs(v[a].temp[i] - v[b].temp[i]);
      sum += v[a].temp[i];
    }
    return diff / sum;
  };
  EXPECT_GT(bed_change(150, 180), 0.25);
  // hotter and livelier with intensity
  const auto cool = blaze_run(sim::Effect::coals, 0.f, 0.5f, 0.5f, 120), hot = blaze_run(sim::Effect::coals, 1.f, 0.5f, 0.5f, 120);
  EXPECT_GT(blaze_of(hot[119]).heat, 1.5 * blaze_of(cool[119]).heat);
}

TEST(Sim, MushroomRisesIntoACapThatSpreadsAndDarkens) {
  const auto v = blaze_run(sim::Effect::mushroom, 0.6f, 0.5f, 0.5f, 150);  // its 5 s
  const Blaze first = blaze_of(v[14]), last = blaze_of(v[149]);
  EXPECT_LT(first.sy, 0.3);  // a fireball near the ground
  EXPECT_GT(last.sy, 0.5);   // that rose
  EXPECT_LT(last.heat / last.soot, 0.3 * first.heat / first.soot);  // and turned to smoke
  // a cap wider than its stem: soot above 60% of the height spans more than twice as many columns as at 15 to 30%
  const auto span = [](const sim::State& s, double lo, double hi) {
    int cols = 0;
    for (int x = 0; x < s.n; ++x) {
      double m = 0;
      for (int y = 0; y < s.n; ++y) {
        const double h = (y + 0.5) / s.n;
        if (h >= lo && h < hi) m = std::max(m, static_cast<double>(s.soot[static_cast<std::size_t>(y * s.n + x)]));
      }
      cols += m > 0.2;
    }
    return cols;
  };
  EXPECT_GT(span(v[119], 0.6, 1.0), 2 * span(v[119], 0.15, 0.3));
  EXPECT_GT(span(v[119], 0.15, 0.3), 0);  // with a stem
  // a bigger yield, a bigger cloud; the wind leans it
  const auto small_one = blaze_run(sim::Effect::mushroom, 0.f, 0.5f, 0.5f, 90), big = blaze_run(sim::Effect::mushroom, 1.f, 0.5f, 0.5f, 90);
  EXPECT_GT(blaze_of(big[89]).soot, 2.0 * blaze_of(small_one[89]).soot);
  EXPECT_GT(blaze_of(big[89]).sy, blaze_of(small_one[89]).sy);
  const auto left = blaze_run(sim::Effect::mushroom, 0.6f, 0.f, 0.5f, 120), right = blaze_run(sim::Effect::mushroom, 0.6f, 1.f, 0.5f, 120);
  EXPECT_GT(blaze_of(right[119]).sx, blaze_of(left[119]).sx + 0.1);
}

// --- launch, puffs and ink (docs/EFFECTS_FLOW.md) ---------------------------------------------------------------------

TEST(Sim, FlowEffectsHaveTheirOwnControls) {
  const std::array<std::pair<sim::Effect, std::string_view>, 3> second{
      {{sim::Effect::launch, "wind"}, {sim::Effect::puffs, "rate"}, {sim::Effect::ink, "current"}}};
  for (const auto& [e, name] : second) {
    sim::Effect back{};
    ASSERT_TRUE(sim::parse_effect(sim::effect_name(e), back));
    EXPECT_EQ(back, e);
    EXPECT_TRUE(sim::effect_loops(e));
    EXPECT_TRUE(std::ranges::contains(sim::kNewEffects, e));
    EXPECT_FALSE(std::ranges::contains(sim::kEffects, e));
    const auto names = sim::control_names(e);
    EXPECT_EQ(names[0], "intensity");
    EXPECT_EQ(names[1], name);
    EXPECT_EQ(names[2], "turbulence");
  }
  // added after cloud (and after wildfire, coals and mushroom), in this order
  const auto at = [](sim::Effect e) { return std::ranges::find(sim::kAllEffects, e) - sim::kAllEffects.begin(); };
  EXPECT_LT(at(sim::Effect::cloud), at(sim::Effect::launch));
  EXPECT_LT(at(sim::Effect::launch), at(sim::Effect::puffs));
  EXPECT_LT(at(sim::Effect::puffs), at(sim::Effect::ink));
}

namespace {

// The material's spread after `frames` frames at 64 cells, and what was drawn.
struct FlowRun {
  Spread spread;
  sim::State state;
  std::vector<std::uint8_t> rgba;
};
FlowRun flow_run(sim::Effect e, float intensity, float second, float turbulence, int frames) {
  sim::Fluid f(atmos_params(e, intensity, second, turbulence));
  for (int i = 0; i < frames; ++i) f.step_frame();
  FlowRun r{spread_of(f.state()), f.state(), std::vector<std::uint8_t>(64 * 64 * 4)};
  f.render(r.rgba);
  return r;
}

// Soot in a band of rows [y0, y1) and columns [x0, x1) of a state (fractions of the frame, y up).
double soot_in(const sim::State& s, double x0, double x1, double y0, double y1) {
  double a = 0;
  for (int y = static_cast<int>(y0 * s.n); y < static_cast<int>(y1 * s.n); ++y) {
    for (int x = static_cast<int>(x0 * s.n); x < static_cast<int>(x1 * s.n); ++x) a += s.soot[static_cast<std::size_t>(y * s.n + x)];
  }
  return a;
}

}  // namespace

TEST(Sim, LaunchBlastsTheGroundAndBillowsOutBothWays) {
  const FlowRun r = flow_run(sim::Effect::launch, 0.6f, 0.5f, 0.5f, 150);
  // the clouds run out along the ground both ways, well beyond the jet (about a fifth of the frame wide)
  EXPECT_GT(soot_in(r.state, 0.0, 0.3, 0.0, 0.3), 0.15 * r.spread.amount);
  EXPECT_GT(soot_in(r.state, 0.7, 1.0, 0.0, 0.3), 0.15 * r.spread.amount);
  EXPECT_GT(r.spread.width(), 0.8);
  // the heat reaches the ground under the nozzle
  double ground_heat = 0, heat = 0;
  for (int y = 0; y < r.state.n; ++y) {
    for (int x = 0; x < r.state.n; ++x) {
      const float t = r.state.temp[static_cast<std::size_t>(y * r.state.n + x)];
      heat += t;
      if (y < r.state.n / 8 && std::abs(x - r.state.n / 2) < r.state.n / 8) ground_heat += t;
    }
  }
  EXPECT_GT(ground_heat, 0.05 * heat);  // (8% of it in the bottom eighth of the middle quarter of the frame)
  // bright at the impact: light (colour above coverage) near the ground in the middle, reflected light at the sides
  long lit = 0, lit_side = 0;
  for (int py = 0; py < 64; ++py) {
    for (int px = 0; px < 64; ++px) {
      const auto* o = &r.rgba[static_cast<std::size_t>(py * 64 + px) * 4];
      if (o[0] <= o[3] + 8) continue;
      ++lit;
      if (std::abs(px - 32) > 20) ++lit_side;
    }
  }
  EXPECT_GT(lit, 60);
  EXPECT_LT(lit_side, lit / 10);
  // the clouds rise as they cool: their centre climbs above the ground band
  EXPECT_GT(r.spread.cy, 0.15);
  EXPECT_GT(r.spread.top, 0.5);
  // the wind drifts them
  const FlowRun left = flow_run(sim::Effect::launch, 0.6f, 0.f, 0.5f, 150), right = flow_run(sim::Effect::launch, 0.6f, 1.f, 0.5f, 150);
  EXPECT_LT(left.spread.cx, 0.45);
  EXPECT_GT(right.spread.cx, 0.55);
}

TEST(Sim, PuffsComeInPulsesThatTheRateControlQuickens) {
  // puffs past a line a little above the mouth (the soot there rising past a level and falling back)
  const auto puffs = [](float rate, std::uint64_t seed) {
    auto p = atmos_params(sim::Effect::puffs, 0.7f, rate, 0.4f);
    p.seed = seed;
    sim::Fluid f(p);
    int n = 0;
    bool open = false;
    for (int i = 0; i < 300; ++i) {
      f.step_frame();
      const double m = soot_in(f.state(), 0.28, 0.45, 0.14, 0.2) / (0.17 * 0.06 * 64 * 64);
      if (!open && m > 0.3) {
        open = true;
        ++n;
      }
      if (open && m < 0.1) open = false;
    }
    return n;
  };
  int slow = 0, fast = 0;
  for (const std::uint64_t seed : {1, 2, 3}) {
    slow += puffs(0.1f, seed);
    fast += puffs(0.9f, seed);
  }
  EXPECT_GE(slow, 2);              // distinct pulses, a few in 10 s at the slow rate
  EXPECT_GT(fast, slow + slow / 2);  // more at the fast rate
  const FlowRun r = flow_run(sim::Effect::puffs, 0.7f, 0.5f, 0.4f, 240);
  EXPECT_GT(r.spread.cx, 46.5 / 128 + 0.05);  // the trail drifts right of the mouth
  EXPECT_GT(r.spread.top, 0.6);                // and rises
  long bad = 0, covered = 0;
  for (std::size_t i = 0; i < r.rgba.size(); i += 4) {
    for (int k = 0; k < 3; ++k) bad += r.rgba[i + static_cast<std::size_t>(k)] > r.rgba[i + 3] + 1;  // reflected light only
    covered += r.rgba[i + 3] > 64;
  }
  EXPECT_EQ(bad, 0);
  EXPECT_GT(covered, 200);
}

TEST(Sim, InkIsSlowDarkDyeThatTheCurrentCarries) {
  const FlowRun r = flow_run(sim::Effect::ink, 0.7f, 0.5f, 0.5f, 240);
  EXPECT_TRUE(std::ranges::all_of(r.state.temp, [](float t) { return t == 0.f; }));  // no heat
  long bad = 0, covered = 0;
  double rgb = 0, alpha = 0;
  for (std::size_t i = 0; i < r.rgba.size(); i += 4) {
    for (int k = 0; k < 3; ++k) bad += r.rgba[i + static_cast<std::size_t>(k)] > r.rgba[i + 3] + 1;  // no light
    if (r.rgba[i + 3] > 64) {
      ++covered;
      rgb += r.rgba[i] + r.rgba[i + 1] + r.rgba[i + 2];
      alpha += 3.0 * r.rgba[i + 3];
    }
  }
  EXPECT_EQ(bad, 0);
  EXPECT_GT(covered, 300);
  EXPECT_LT(rgb, 0.25 * alpha);  // dark
  // slow: where there is dye, the water moves at a few cells (of a 128-cell frame) a second
  std::vector<float> speed;
  for (std::size_t i = 0; i < r.state.soot.size(); ++i) {
    if (r.state.soot[i] > 0.05f) speed.push_back(2.f * std::hypot(r.state.u[i], r.state.v[i]));
  }
  ASSERT_FALSE(speed.empty());
  std::ranges::sort(speed);
  EXPECT_LT(speed[speed.size() / 2], 20.f);
  EXPECT_LT(speed[speed.size() * 99 / 100], 50.f);
  // the current carries it left or right
  const FlowRun left = flow_run(sim::Effect::ink, 0.7f, 0.f, 0.5f, 240), right = flow_run(sim::Effect::ink, 0.7f, 1.f, 0.5f, 240);
  EXPECT_LT(left.spread.cx, 0.4);
  EXPECT_GT(right.spread.cx, 0.6);
}

// --- waterfall, fountain, splash and rainshaft (docs/EFFECTS_WATER.md) -----------------------------------------------

namespace {

sim::Params water_params(sim::Effect e, float intensity, float second, float turbulence, std::uint64_t seed = 3) {
  sim::Params p;
  p.effect = e;
  p.size = 64;
  p.intensity = intensity;
  p.wind = second;
  p.turbulence = turbulence;
  p.seed = seed;
  p.pressure_iters = 16;
  return p;
}

// Where the water (soot) and the mist (heat) are, as fractions of the frame (y up), after `frames` frames.
struct WaterSpread {
  double water = 0, mist = 0, wx = 0, wy = 0, mx = 0, my = 0, top = 0, low_x = 0, low = 0;
};

WaterSpread water_spread(const sim::Params& p, int frames) {
  sim::Fluid f(p);
  for (int i = 0; i < frames; ++i) f.step_frame();
  const sim::State s = f.state();
  WaterSpread o;
  float peak = 0.f;
  for (const float d : s.soot) peak = std::max(peak, d);
  const double n = s.n;
  for (int y = 0; y < s.n; ++y) {
    for (int x = 0; x < s.n; ++x) {
      const std::size_t i = static_cast<std::size_t>(y * s.n + x);
      const double d = s.soot[i], t = s.temp[i];
      o.water += d;
      o.wx += d * (x + 0.5) / n;
      o.wy += d * (y + 0.5) / n;
      o.mist += t;
      o.mx += t * (x + 0.5) / n;
      o.my += t * (y + 0.5) / n;
      if (d > 0.1 * peak) o.top = std::max(o.top, (y + 1) / n);
      if (y < s.n / 2) {  // the lower half: water and mist
        o.low += d + t;
        o.low_x += (d + t) * (x + 0.5) / n;
      }
    }
  }
  if (o.water > 0) {
    o.wx /= o.water;
    o.wy /= o.water;
  }
  if (o.mist > 0) {
    o.mx /= o.mist;
    o.my /= o.mist;
  }
  if (o.low > 0) o.low_x /= o.low;
  return o;
}

}  // namespace

TEST(Sim, WaterEffectsAreNamedWithTheirControls) {
  for (const sim::Effect e : {sim::Effect::waterfall, sim::Effect::fountain, sim::Effect::splash, sim::Effect::rainshaft}) {
    const auto name = sim::effect_name(e);
    sim::Effect back{};
    ASSERT_TRUE(sim::parse_effect(name, back));
    EXPECT_EQ(back, e);
    EXPECT_EQ(sim::effect_loops(e), e != sim::Effect::splash) << name;  // a splash plays once, as the explosion and dust
    EXPECT_TRUE(std::ranges::contains(sim::kNewEffects, e)) << name;
    EXPECT_FALSE(std::ranges::contains(sim::kEffects, e)) << name;
    EXPECT_EQ(sim::control_names(e)[1], e == sim::Effect::fountain ? "aim" : "wind") << name;
  }
  // after cloud, in this order
  // the four after round 7's six (wildfire, coals, mushroom, launch, puffs, ink)
  const auto at = [](sim::Effect e) { return std::ranges::find(sim::kAllEffects, e) - sim::kAllEffects.begin(); };
  EXPECT_EQ(at(sim::Effect::waterfall), at(sim::Effect::ink) + 1);
  EXPECT_EQ(at(sim::Effect::rainshaft), at(sim::Effect::ink) + 4);
}

TEST(Sim, WaterIsReflectedLightBlueWhiteAndTranslucent) {
  for (const sim::Effect e : {sim::Effect::waterfall, sim::Effect::fountain, sim::Effect::splash, sim::Effect::rainshaft}) {
    const auto name = sim::effect_name(e);
    auto p = water_params(e, 0.7f, 0.5f, 0.5f);
    p.frames = e == sim::Effect::splash ? 30 : 24;
    const Clip c = sim::simulate(p);
    long bad = 0, covered = 0, translucent = 0;
    double r = 0, b = 0, a = 0;
    for (std::size_t i = 0; i < c.rgba.size(); i += 4) {
      for (std::size_t k = 0; k < 3; ++k) bad += c.rgba[i + k] > c.rgba[i + 3] + 1;  // no emission
      if (c.rgba[i + 3] > 32) {
        ++covered;
        translucent += c.rgba[i + 3] < 230;
        r += c.rgba[i];
        b += c.rgba[i + 2];
        a += c.rgba[i + 3];
      }
    }
    EXPECT_EQ(bad, 0) << name;
    ASSERT_GT(covered, 200) << name;
    EXPECT_GT(b, 1.05 * r) << name;           // blue-white (the rain grey-blue)
    EXPECT_GT(b, 0.6 * a) << name;            // and light, not dark
    EXPECT_GT(translucent, covered / 2) << name;  // mostly see-through
  }
}

TEST(Sim, WaterfallPoursFromItsLipIntoAMistThatTheWindCarries) {
  const auto v = water_spread(water_params(sim::Effect::waterfall, 0.6f, 0.5f, 0.5f), 150);
  EXPECT_GT(v.top, 0.85);    // the water comes over the lip near the top of the frame
  EXPECT_LT(v.my, 0.4);      // the mist is where it lands, low down
  EXPECT_LT(v.my, v.wy);
  EXPECT_GT(v.mist, 0.2 * v.water);
  EXPECT_NEAR(v.wx, 0.5, 0.06);  // the curtain falls straight
  const auto left = water_spread(water_params(sim::Effect::waterfall, 0.6f, 0.f, 0.5f), 150);
  const auto right = water_spread(water_params(sim::Effect::waterfall, 0.6f, 1.f, 0.5f), 150);
  EXPECT_GT(right.mx, left.mx + 0.04);  // the wind carries the mist
  const auto thin = water_spread(water_params(sim::Effect::waterfall, 0.f, 0.5f, 0.5f), 150);
  const auto heavy = water_spread(water_params(sim::Effect::waterfall, 1.f, 0.5f, 0.5f), 150);
  EXPECT_GT(heavy.water, 1.5 * thin.water);  // more flow
}

TEST(Sim, FountainRisesFallsBackAndLeansWithItsAim) {
  const auto low = water_spread(water_params(sim::Effect::fountain, 0.f, 0.5f, 0.5f), 120);
  const auto high = water_spread(water_params(sim::Effect::fountain, 1.f, 0.5f, 0.5f), 120);
  EXPECT_GT(low.top, 0.3);
  EXPECT_GT(high.top, low.top + 0.15);  // intensity: how high
  EXPECT_GT(high.mist, 0.f);            // it falls back as spray
  const auto left = water_spread(water_params(sim::Effect::fountain, 0.6f, 0.f, 0.5f), 120);
  const auto right = water_spread(water_params(sim::Effect::fountain, 0.6f, 1.f, 0.5f), 120);
  EXPECT_LT(left.wx, 0.47);  // aim: leaning left or right
  EXPECT_GT(right.wx, 0.53);
}

TEST(Sim, SplashThrowsUpACrownThatFallsBackIntoMist) {
  const auto p = water_params(sim::Effect::splash, 0.7f, 0.5f, 0.5f);
  const auto crown = water_spread(p, 12), late = water_spread(p, 90);
  EXPECT_GT(crown.top, 0.3);                  // the crown stands up within half a second
  EXPECT_LT(late.water, 0.3 * crown.water);   // and falls back
  EXPECT_GT(late.mist, 0.f);                  // leaving mist
  EXPECT_LT(late.my, 0.3);                    // low down
  const auto small_one = water_spread(water_params(sim::Effect::splash, 0.f, 0.5f, 0.5f), 12);
  const auto big_one = water_spread(water_params(sim::Effect::splash, 1.f, 0.5f, 0.5f), 12);
  EXPECT_GT(big_one.top, small_one.top + 0.1);  // intensity: size
  EXPECT_GT(big_one.water, 1.5 * small_one.water);
}

TEST(Sim, RainFallsFromTheCloudBaseAndTheWindSlantsIt) {
  const auto v = water_spread(water_params(sim::Effect::rainshaft, 0.6f, 0.5f, 0.5f), 120);
  EXPECT_GT(v.top, 0.9);        // from the top of the frame
  EXPECT_NEAR(v.wy, 0.5, 0.15);  // all the way down
  const auto left = water_spread(water_params(sim::Effect::rainshaft, 0.6f, 0.f, 0.5f), 120);
  const auto right = water_spread(water_params(sim::Effect::rainshaft, 0.6f, 1.f, 0.5f), 120);
  EXPECT_GT(right.low_x, left.low_x + 0.1);  // blown aside lower down
  const auto light = water_spread(water_params(sim::Effect::rainshaft, 0.f, 0.5f, 0.5f), 120);
  const auto heavy = water_spread(water_params(sim::Effect::rainshaft, 1.f, 0.5f, 0.5f), 120);
  EXPECT_GT(heavy.water, 2.0 * light.water);  // intensity: how heavy
}

// --- candle, smokering and heavygas (docs/EFFECTS_AIR.md) ------------------------------------------------------------

TEST(Sim, AirEffectsHaveTheirOwnControlsAndOnlyTheRingPlaysOnce) {
  const std::array<std::pair<sim::Effect, std::string_view>, 3> second{
      {{sim::Effect::candle, "draught"}, {sim::Effect::smokering, "wind"}, {sim::Effect::heavygas, "flow"}}};
  for (const auto& [e, name] : second) {
    sim::Effect back{};
    ASSERT_TRUE(sim::parse_effect(sim::effect_name(e), back));
    EXPECT_EQ(back, e);
    EXPECT_TRUE(std::ranges::contains(sim::kNewEffects, e));
    EXPECT_FALSE(std::ranges::contains(sim::kEffects, e));
    const auto names = sim::control_names(e);
    EXPECT_EQ(names[0], "intensity");
    EXPECT_EQ(names[1], name);
    EXPECT_EQ(names[2], "turbulence");
    EXPECT_EQ(sim::effect_loops(e), e != sim::Effect::smokering);
  }
  EXPECT_EQ(sim::effect_name(sim::Effect::smokering), "smokering");
  EXPECT_EQ(sim::effect_name(sim::Effect::heavygas), "heavygas");
  EXPECT_EQ(sim::one_shot_frames(sim::Effect::smokering), 120);  // 4 s
  EXPECT_EQ(sim::one_shot_frames(sim::Effect::candle), 0);
  // after the water effects, in this order
  const auto at = [](sim::Effect e) { return std::ranges::find(sim::kAllEffects, e) - sim::kAllEffects.begin(); };
  EXPECT_GT(at(sim::Effect::candle), at(sim::Effect::rainshaft));
  EXPECT_EQ(at(sim::Effect::smokering), at(sim::Effect::candle) + 1);
  EXPECT_EQ(at(sim::Effect::heavygas), at(sim::Effect::candle) + 2);
}

namespace {

// The state after each of `frames` frames, at 64 cells.
std::vector<sim::State> air_run(sim::Effect e, float intensity, float second, float turbulence, int frames, std::uint64_t seed = 1) {
  auto p = atmos_params(e, intensity, second, turbulence);
  p.seed = seed;
  sim::Fluid f(p);
  std::vector<sim::State> v;
  for (int i = 0; i < frames; ++i) {
    f.step_frame();
    v.push_back(f.state());
  }
  return v;
}

// Heat in the flame (above the visible threshold of the candle's light), its centroid and extent, as fractions of the
// frame (y up).
struct Flame {
  double heat = 0, x = 0, y = 0, left = 1, right = 0, top = 0, bottom = 1;
};
Flame flame_of(const sim::State& s) {
  Flame o;
  const double n = s.n;
  for (int y = 0; y < s.n; ++y) {
    for (int x = 0; x < s.n; ++x) {
      const double t = s.temp[static_cast<std::size_t>(y * s.n + x)];
      if (t < 0.4) continue;
      o.heat += t;
      o.x += t * (x + 0.5) / n;
      o.y += t * (y + 0.5) / n;
      o.left = std::min(o.left, x / n);
      o.right = std::max(o.right, (x + 1) / n);
      o.bottom = std::min(o.bottom, y / n);
      o.top = std::max(o.top, (y + 1) / n);
    }
  }
  if (o.heat > 0) {
    o.x /= o.heat;
    o.y /= o.heat;
  }
  return o;
}

}  // namespace

TEST(Sim, CandleIsASmallSteadyTeardropWithAWispThatLeansInADraught) {
  const auto v = air_run(sim::Effect::candle, 0.5f, 0.5f, 0.1f, 150);
  for (int i = 90; i < 150; i += 10) {
    const Flame fl = flame_of(v[static_cast<std::size_t>(i)]);
    EXPECT_GT(fl.heat, 1.0) << i;
    EXPECT_LT(fl.right - fl.left, 0.16) << i;  // small and narrow, on its wick in the middle of the frame
    EXPECT_GT(fl.top - fl.bottom, 1.8 * (fl.right - fl.left)) << i;  // taller than it is wide
    EXPECT_LT(fl.top, 0.5) << i;
    EXPECT_NEAR(fl.x, 0.5, 0.03) << i;
    const Spread sm = spread_of(v[static_cast<std::size_t>(i)]);
    EXPECT_GT(sm.cy, fl.y + 0.1) << i;  // its smoke rises above it
    EXPECT_GT(sm.top, 0.8) << i;
  }
  // steady in still air: the flame's heat changes little from frame to frame
  const auto change = [](const sim::State& a, const sim::State& b) {
    double d = 0, sum = 0;
    for (std::size_t i = 0; i < a.temp.size(); ++i) {
      if (a.temp[i] < 0.4f && b.temp[i] < 0.4f) continue;
      d += std::abs(a.temp[i] - b.temp[i]);
      sum += a.temp[i];
    }
    return d / sum;
  };
  double calm = 0, gusty = 0;
  const auto t = air_run(sim::Effect::candle, 0.5f, 0.5f, 1.f, 150), d = air_run(sim::Effect::candle, 0.5f, 0.9f, 0.1f, 150);
  for (int i = 100; i < 140; ++i) {
    calm += change(v[static_cast<std::size_t>(i)], v[static_cast<std::size_t>(i + 1)]) / 40;
    gusty += change(t[static_cast<std::size_t>(i)], t[static_cast<std::size_t>(i + 1)]) / 40;
  }
  EXPECT_LT(calm, 0.1);
  EXPECT_GT(gusty, 1.5 * calm);  // turbulence makes it flicker
  // the draught leans it and its smoke to the side it blows to
  const auto left = air_run(sim::Effect::candle, 0.5f, 0.1f, 0.1f, 120), right = air_run(sim::Effect::candle, 0.5f, 0.9f, 0.1f, 120);
  EXPECT_LT(flame_of(left[119]).x, 0.5 - 0.02);
  EXPECT_GT(flame_of(right[119]).x, 0.5 + 0.02);
  EXPECT_GT(spread_of(right[119]).cx, spread_of(left[119]).cx + 0.15);
  EXPECT_GT(flame_of(d[149]).x, 0.5 + 0.02);
  // a bigger flame with intensity; it gives light (colour above coverage)
  const auto small_one = air_run(sim::Effect::candle, 0.f, 0.5f, 0.1f, 90), big = air_run(sim::Effect::candle, 1.f, 0.5f, 0.1f, 90);
  EXPECT_GT(flame_of(big[89]).top - flame_of(big[89]).bottom, 1.4 * (flame_of(small_one[89]).top - flame_of(small_one[89]).bottom));
  EXPECT_GT(flame_of(big[89]).heat, 2.0 * flame_of(small_one[89]).heat);
  auto p = atmos_params(sim::Effect::candle, 0.6f, 0.5f, 0.3f);
  p.frames = 12;
  const Clip c = sim::simulate(p);
  long glowing = 0;
  for (std::size_t i = 0; i < c.rgba.size(); i += 4) glowing += c.rgba[i] > c.rgba[i + 3] + 8;
  EXPECT_GT(glowing, 100);
}

namespace {

// Circulation (cells squared per second, y up) of the left and the right half of a state.
std::pair<double, double> halves_circulation(const sim::State& s) {
  double l = 0, r = 0;
  for (int y = 1; y < s.n - 1; ++y) {
    for (int x = 1; x < s.n - 1; ++x) {
      const auto at = [&](int xx, int yy) { return static_cast<std::size_t>(yy * s.n + xx); };
      const double w = 0.5 * ((s.v[at(x + 1, y)] - s.v[at(x - 1, y)]) - (s.u[at(x, y + 1)] - s.u[at(x, y - 1)]));
      (x < s.n / 2 ? l : r) += w;
    }
  }
  return {l, r};
}

}  // namespace

TEST(Sim, SmokeRingRisesAsAVortexPairThatSpreadsAndBreaksUp) {
  const auto v = air_run(sim::Effect::smokering, 0.5f, 0.5f, 0.3f, sim::one_shot_frames(sim::Effect::smokering));
  // a pair of vortices turning in opposite senses, the left one anticlockwise: the air between them rises
  const auto [l, r] = halves_circulation(v[0]);
  EXPECT_GT(l, 0.0);
  EXPECT_LT(r, 0.0);
  EXPECT_NEAR(l, -r, 0.2 * l);
  const Spread first = spread_of(v[0]), mid = spread_of(v[59]), last = spread_of(v[119]);
  EXPECT_LT(first.cy, 0.3);  // puffed near the bottom
  EXPECT_GT(mid.cy, first.cy + 0.15);  // it rises
  EXPECT_GT(last.cy, mid.cy);
  EXPECT_GT(last.width(), first.width() + 0.1);  // and spreads
  EXPECT_NEAR(mid.cx, 0.5, 0.05);
  // slowly: it rises more slowly at the end than at the start
  EXPECT_LT(last.cy - spread_of(v[89]).cy, spread_of(v[29]).cy - first.cy);
  // its two cores hold together for a second or two: soot in both halves, little between them below the top
  const auto halves = [](const sim::State& s) {
    double a = 0, b = 0;
    for (int y = 0; y < s.n; ++y) {
      for (int x = 0; x < s.n; ++x) (x < s.n / 2 ? a : b) += s.soot[static_cast<std::size_t>(y * s.n + x)];
    }
    return std::pair{a, b};
  };
  const auto [ha, hb] = halves(v[45]);
  EXPECT_NEAR(ha, hb, 0.25 * (ha + hb));
  // then breaks up: with turbulence its smoke is scattered further about its middle at the end
  const auto rough = air_run(sim::Effect::smokering, 0.5f, 0.5f, 1.f, 120);
  const auto scatter = [](const sim::State& s) {
    const Spread o = spread_of(s);
    double r2 = 0;
    for (int y = 0; y < s.n; ++y) {
      for (int x = 0; x < s.n; ++x) {
        const double dx = (x + 0.5) / s.n - o.cx, dy = (y + 0.5) / s.n - o.cy;
        r2 += s.soot[static_cast<std::size_t>(y * s.n + x)] * (dx * dx + dy * dy);
      }
    }
    return std::sqrt(r2 / o.amount);
  };
  EXPECT_GT(scatter(rough[119]), 1.1 * scatter(v[119]));
  // it is smoke: reflected light only
  auto p = atmos_params(sim::Effect::smokering, 0.6f, 0.5f, 0.3f);
  p.frames = 30;
  const Clip c = sim::simulate(p);
  EXPECT_FALSE(c.loop);
  long bad = 0, covered = 0;
  for (std::size_t i = 0; i < c.rgba.size(); i += 4) {
    for (int k = 0; k < 3; ++k) bad += c.rgba[i + static_cast<std::size_t>(k)] > c.rgba[i + 3] + 1;
    covered += c.rgba[i + 3] > 64;
  }
  EXPECT_EQ(bad, 0);
  EXPECT_GT(covered, 200);
  // a stronger puff, a bigger ring that rises faster; the wind drifts it
  const auto weak = air_run(sim::Effect::smokering, 0.f, 0.5f, 0.3f, 60), strong = air_run(sim::Effect::smokering, 1.f, 0.5f, 0.3f, 60);
  EXPECT_GT(spread_of(strong[59]).cy, spread_of(weak[59]).cy + 0.05);
  EXPECT_GT(spread_of(strong[59]).amount, 1.5 * spread_of(weak[59]).amount);
  const auto left = air_run(sim::Effect::smokering, 0.5f, 0.f, 0.3f, 90), right = air_run(sim::Effect::smokering, 0.5f, 1.f, 0.3f, 90);
  EXPECT_GT(spread_of(right[89]).cx, spread_of(left[89]).cx + 0.15);
}

TEST(Sim, HeavyGasSpillsAlongTheFloorAndPoolsInTheHollows) {
  const auto v = air_run(sim::Effect::heavygas, 0.6f, 0.5f, 0.3f, 240);
  EXPECT_TRUE(std::ranges::all_of(v[239].temp, [](float t) { return t == 0.f; }));  // no heat: its weight moves it
  const Spread early = spread_of(v[59]), late = spread_of(v[239]);
  EXPECT_LT(late.cy, 0.2);                       // a layer on the floor, under its release (a fifth of the frame up)
  EXPECT_LT(late.top, 0.45);
  EXPECT_GT(late.width(), 0.75);                 // that has spread over the floor
  EXPECT_GT(late.width(), early.width() + 0.3);  // from its release (about a sixth of the frame wide)
  // the gas runs out as a current with a front: the edge of the layer moves along the floor
  const Spread s90 = spread_of(v[89]), s120 = spread_of(v[119]);
  EXPECT_GT(s120.right, s90.right + 0.04);
  EXPECT_LT(s120.left, s90.left - 0.04);
  // it pools in the hollows: in the bottom rows there is more gas over the hollows than over the mound
  const auto low = [](const sim::State& s, double x0, double x1) {
    double a = 0;
    int cells = 0;
    for (int y = 0; y < s.n * 12 / 100; ++y) {
      for (int x = static_cast<int>(x0 * s.n); x < static_cast<int>(x1 * s.n); ++x) {
        a += s.soot[static_cast<std::size_t>(y * s.n + x)];
        ++cells;
      }
    }
    return a / cells;
  };
  double hollows = 0, mound = 0;
  for (int i = 180; i < 240; i += 10) {
    hollows += low(v[static_cast<std::size_t>(i)], 0.1, 0.25) + low(v[static_cast<std::size_t>(i)], 0.75, 0.9);
    mound += 2 * low(v[static_cast<std::size_t>(i)], 0.42, 0.58);
  }
  EXPECT_GT(hollows, 1.5 * mound);
  // the flow control sends it left or right; more intensity, more gas
  const auto left = air_run(sim::Effect::heavygas, 0.6f, 0.f, 0.3f, 150), right = air_run(sim::Effect::heavygas, 0.6f, 1.f, 0.3f, 150);
  EXPECT_LT(spread_of(left[149]).cx, 0.45);
  EXPECT_GT(spread_of(right[149]).cx, 0.55);
  const auto thin = air_run(sim::Effect::heavygas, 0.f, 0.5f, 0.3f, 150), thick = air_run(sim::Effect::heavygas, 1.f, 0.5f, 0.3f, 150);
  EXPECT_GT(spread_of(thick[149]).amount, 1.5 * spread_of(thin[149]).amount);
  // a pale vapour: reflected light only
  auto p = atmos_params(sim::Effect::heavygas, 0.6f, 0.5f, 0.3f);
  p.frames = 12;
  const Clip c = sim::simulate(p);
  long bad = 0, covered = 0;
  double rgb = 0, alpha = 0;
  for (std::size_t i = 0; i < c.rgba.size(); i += 4) {
    for (int k = 0; k < 3; ++k) bad += c.rgba[i + static_cast<std::size_t>(k)] > c.rgba[i + 3] + 1;
    if (c.rgba[i + 3] > 64) {
      ++covered;
      rgb += c.rgba[i] + c.rgba[i + 1] + c.rgba[i + 2];
      alpha += 3.0 * c.rgba[i + 3];
    }
  }
  EXPECT_EQ(bad, 0);
  EXPECT_GT(covered, 200);
  EXPECT_GT(rgb, 0.6 * alpha);  // pale
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

// Couplings from outside (docs/COMPOSE.md §9): push and add_material.
namespace {

sim::Fluid running(sim::Effect e, int frames) {
  sim::Fluid f(small(e));
  for (int i = 0; i < frames; ++i) f.step_frame();
  return f;
}

std::vector<float> wave(int n, float amp, float phase) {
  std::vector<float> v(static_cast<std::size_t>(n) * n);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = amp * std::sin(0.37f * static_cast<float>(i % 97) + phase);
  return v;
}

}  // namespace

TEST(Sim, PushThenUnpushRestoresTheState) {
  for (const sim::Effect e : sim::kEffects) {
    sim::Fluid f = running(e, 15);
    const sim::State a = f.state();
    const auto du = wave(a.n, 37.f, 0.f), dv = wave(a.n, 23.f, 1.f);
    std::vector<float> mu(du.size()), mv(dv.size());
    std::ranges::transform(du, mu.begin(), [](float x) { return -x; });
    std::ranges::transform(dv, mv.begin(), [](float x) { return -x; });
    f.push(du, dv);
    const sim::State b = f.state();
    EXPECT_NE(a.u, b.u);
    f.push(mu, mv);
    const sim::State c = f.state();
    // (u + du) - du is u up to one rounding of the sum: exact wherever float addition is
    for (std::size_t i = 0; i < a.u.size(); ++i) {
      ASSERT_NEAR(c.u[i], a.u[i], 1e-6f * (std::abs(a.u[i]) + std::abs(du[i]))) << i;
      ASSERT_NEAR(c.v[i], a.v[i], 1e-6f * (std::abs(a.v[i]) + std::abs(dv[i]))) << i;
    }
    EXPECT_EQ(c.temp, a.temp);
    EXPECT_EQ(c.soot, a.soot);
    EXPECT_EQ(c.pressure, a.pressure);
    // a zero push changes nothing, and the run continues bit for bit as if nothing had been done
    sim::Fluid g = running(e, 15), h = running(e, 15);
    const std::vector<float> zero(du.size(), 0.f);
    g.push(zero, zero);
    g.add_material(zero, zero);
    for (int i = 0; i < 5; ++i) {
      g.step_frame();
      h.step_frame();
    }
    EXPECT_EQ(g.state().u, h.state().u) << sim::effect_name(e);
    EXPECT_EQ(g.state().soot, h.state().soot) << sim::effect_name(e);
  }
}

TEST(Sim, PushIsDeterministicAndMovesMaterial) {
  const auto run = [](bool pushed) {
    sim::Fluid f = running(sim::Effect::smoke, 20);
    const int n = f.state().n;
    const std::vector<float> du(static_cast<std::size_t>(n) * n, 300.f), zero(du.size(), 0.f);
    std::vector<float> back(du.size(), -300.f);
    for (int i = 0; i < 4; ++i) {
      if (pushed) f.push(du, zero);
      f.step_frame();
      if (pushed) f.push(back, zero);
    }
    return f.state();
  };
  const sim::State a = run(true), b = run(true), plain = run(false);
  EXPECT_EQ(a.u, b.u);
  EXPECT_EQ(a.soot, b.soot);
  // the soot's centre of mass moved right (a push of 300 cells per second for 4 frames)
  const auto cx = [](const sim::State& s) {
    double m = 0, x = 0;
    for (std::size_t i = 0; i < s.soot.size(); ++i) {
      m += s.soot[i];
      x += s.soot[i] * static_cast<double>(i % static_cast<std::size_t>(s.n));
    }
    return x / m;
  };
  EXPECT_GT(cx(a), cx(plain) + 1.0);
}

TEST(Sim, AddMaterialClampsAtZero) {
  sim::Fluid f = running(sim::Effect::fire, 20);
  const sim::State a = f.state();
  std::vector<float> minus(a.temp.size(), -0.3f), plus(a.temp.size(), 0.25f);
  f.add_material(minus, plus);
  const sim::State b = f.state();
  for (std::size_t i = 0; i < a.temp.size(); ++i) {
    ASSERT_EQ(b.temp[i], std::max(0.f, a.temp[i] - 0.3f));
    ASSERT_EQ(b.soot[i], std::max(0.f, a.soot[i] + 0.25f));
  }
  EXPECT_EQ(b.u, a.u);
  const std::vector<float> wrong(5, 0.f);
  EXPECT_THROW(f.add_material(wrong, wrong), std::invalid_argument);
  EXPECT_THROW(f.push(wrong, wrong), std::invalid_argument);
}
