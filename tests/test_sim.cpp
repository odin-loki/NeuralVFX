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

// Adding effects (docs/EFFECTS.md, docs/EFFECTS_ENERGY.md) must not change the earlier ones: these hashes are of clips
// made before steam and magic existed (fire, smoke, explosion) and before jet, orb and vortex existed (steam, magic).
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
  const std::array<std::pair<sim::Effect, std::uint64_t>, 5> want{{{sim::Effect::fire, 0xecb25de8e62e7379ULL},
                                                                    {sim::Effect::smoke, 0xea602bfa2409bf3fULL},
                                                                    {sim::Effect::explosion, 0xbb823d31fd8c4eb4ULL},
                                                                    {sim::Effect::steam, 0xa57b2dc109882ec3ULL},
                                                                    {sim::Effect::magic, 0xf79b687a9614f399ULL}}};
  for (const auto& [e, h] : want) {
    auto p = small(e);
    p.intensity = 0.8f;
    p.wind = 0.3f;
    p.turbulence = 0.7f;
    p.seed = 7;
    EXPECT_EQ(fnv(sim::simulate(p).rgba), h) << sim::effect_name(e);
  }
}

TEST(Sim, NewEffectsAreNamedAndLoop) {
  for (const sim::Effect e : sim::kNewEffects) {
    sim::Effect back{};
    ASSERT_TRUE(sim::parse_effect(sim::effect_name(e), back));
    EXPECT_EQ(back, e);
    EXPECT_TRUE(sim::effect_loops(e));
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
  EXPECT_EQ(sim::kAllEffects.back(), sim::Effect::vortex);
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
