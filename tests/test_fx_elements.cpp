// The procedural elements added for the library's third round (docs/COMPOSE.md §4.10, src/compose/fx_*.cpp): hail, a
// burst's spatter and shards, and the lines of light of aurora, rainbow, meteors, beams, shields and glints.
// No data is needed: the scenes have no learned module, or the tests' untrained one.
#include "compose_scene.hpp"
#include "library.hpp"
#include "script.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace {

namespace sc = nfx::compose::script;
namespace cm = nfx::compose;
using cm::Kind;

sc::Scene scene_of(const std::string& text, int threads = 2, bool overlap = false, bool layers = false) {
  sc::Options o;
  o.threads = threads;
  o.overlap = overlap;
  o.layers = layers;
  return sc::Scene(sc::parse(text, "t"), [](const std::string&) { return cm::testing::tiny_effect(3); }, o);
}

std::string error_of(const std::string& text) {
  try {
    sc::Scene s = scene_of(text);
  } catch (const sc::Error& e) {
    return e.what();
  }
  return "no error";
}

struct Seen {
  Kind kind;
  float x, y, vx, vy;
};
std::vector<Seen> census(cm::Particles& p) {
  std::vector<Seen> out;
  int i = 0;
  p.each([&](float x, float y, float& vx, float& vy) { out.push_back({p.kind(i++), x, y, vx, vy}); });
  return out;
}
int count(const std::vector<Seen>& s, Kind k) {
  int n = 0;
  for (const Seen& e : s) n += e.kind == k;
  return n;
}

constexpr const char* kHead = "scene size 200 x 120, fps 30, length 4 s, ground 110\nparticles capacity 8192\n";

// Every new element at once, small, over the tests' module (for the layers and the threads).
std::string everything_new() {
  return std::string(kHead) +
         "effect tiny = \"tiny\"\nlook gas = shader, heat_scale 1, emission 2\n"
         "module m = tiny, size 32, width 64, at (100, 110), look gas, start 1, seed 4\n"
         "emit hail, in (0, 0), size 200 x 110, density 2, seed 7\n"
         "emit aurora, in (0, 0), size 200 x 60, seed 8\n"
         "emit rainbow, at (100, 110), radius 80\n"
         "emit meteors, in (0, 0), size 100 x 30, rate 6, seed 9\n"
         "emit beam, along (10, 60) to (190, 70), seed 10\n"
         "emit shield, at (60, 70), radius 30, rate 4, hit t - 0.3, angle 90, seed 11\n"
         "emit glints, in (0, 80), size 200 x 20, count 30, rate 2, seed 12\n"
         "at 0.2, repeat:\n"
         "  burst at (120, 90), radius 4, embers 0, spatter 14, shards 6, speed 200, stain 1.5, if rand < 0.3\n";
}

}  // namespace

// --- what each does ------------------------------------------------------------------------------------------------------

TEST(FxElements, SpatterFliesFallsAndStainsWhereItLands) {
  const auto play = [](const std::string& burst) {
    sc::Scene scene = scene_of(std::string(kHead) + "at 0.1:\n  burst at (100, 60), radius 3, embers 0, speed 220, direction (1, -1), cone 60" + burst + "\n");
    std::vector<std::uint8_t> rgb(200 * 120 * 3);
    std::vector<std::vector<Seen>> frames;
    for (int f = 0; f < 45; ++f) {
      scene.render(f, rgb);
      frames.push_back(census(scene.particles()));
    }
    return frames;
  };
  const auto stained = play(", spatter 30, colour (0.2, 0.01, 0.01), stain 2, floor 100");
  int flying = 0, lying = 0;
  for (const Seen& s : stained[8]) {
    ASSERT_EQ(s.kind, Kind::spatter);
    flying += s.vy != 0.f;
  }
  for (const Seen& s : stained.back()) {
    ASSERT_EQ(s.kind, Kind::spatter);
    if (s.vx == 0.f && s.vy == 0.f) {
      ++lying;
      EXPECT_FLOAT_EQ(s.y, 100.f) << "a stain lies on its floor";
    } else {
      EXPECT_LE(s.y, 100.f);
    }
  }
  EXPECT_GT(flying, 20) << "thrown, falling";
  EXPECT_GT(lying, 20) << "stains where they landed";
  const auto gone = play(", spatter 30");
  EXPECT_EQ(gone.back().size(), 0u) << "without a stain, a drop is gone where it lands";
  EXPECT_EQ(gone[8].size(), 30u);
}

TEST(FxElements, HailBouncesAndLiesAMoment) {
  sc::Scene scene = scene_of(std::string(kHead) + "emit hail, in (0, 0), size 200 x 100, density 3, seed 1\n");
  std::vector<std::uint8_t> rgb(200 * 120 * 3);
  int up = 0, lying = 0;
  for (int f = 0; f < 45; ++f) {
    scene.render(f, rgb);
    for (const Seen& s : census(scene.particles())) {
      if (s.kind != Kind::hail) continue;
      EXPECT_LE(s.y, 100.f) << "never below its floor (the region's bottom)";
      up += s.vy < 0.f;
      lying += s.vy == 0.f && s.y == 100.f;
    }
  }
  EXPECT_GT(up, 20) << "bounced";
  EXPECT_GT(lying, 20) << "settled";
}

TEST(FxElements, ABurstsSpatterAndShardsComeAfterItsOtherDraws) {
  // the debris and sparks of a burst are the same with spatter and shards after them or without (embers would not be:
  // their flight draws from the scene's generator, which the spatter and shards have moved on)
  const auto run = [](const std::string& more) {
    sc::Scene scene = scene_of(std::string(kHead) + "at 0.1:\n  burst at (100, 60), radius 5, embers 0, debris 6, sparks 8, speed 200" + more + "\n");
    std::vector<std::uint8_t> rgb(200 * 120 * 3);
    for (int f = 0; f < 12; ++f) scene.render(f, rgb);
    return census(scene.particles());
  };
  const auto plain = run(""), more = run(", spatter 12, shards 8, colour (0.5, 0.05, 0.05), stain 1");
  std::vector<Seen> others, before = plain;
  for (const Seen& s : more)
    if (s.kind != Kind::spatter && s.kind != Kind::shard) others.push_back(s);
  const auto order = [](const Seen& a, const Seen& b) { return std::tie(a.kind, a.x, a.y) < std::tie(b.kind, b.x, b.y); };
  std::ranges::sort(others, order);  // (a particle that dies takes the last one's place: the orders differ)
  std::ranges::sort(before, order);
  ASSERT_EQ(others.size(), before.size());
  ASSERT_GT(before.size(), 6u);
  for (std::size_t i = 0; i < before.size(); ++i) {
    EXPECT_EQ(others[i].kind, before[i].kind);
    EXPECT_EQ(others[i].x, before[i].x);
    EXPECT_EQ(others[i].y, before[i].y);
  }
  EXPECT_GT(count(more, Kind::shard), 6);
  EXPECT_GT(count(more, Kind::spatter), 10);
}

TEST(FxElements, TheyKeepTheScenesRandomNumbers) {
  const auto run = [](const std::string& extra) {
    sc::Scene scene = scene_of(std::string(kHead) + "emit sparks, along (20, 70) to (20, 70), count 3\n" + extra);
    std::vector<std::uint8_t> rgb(200 * 120 * 3);
    for (int f = 0; f < 30; ++f) scene.render(f, rgb);
    return std::pair{scene.particles().alive(), scene.particles().rng};
  };
  const auto [n0, rng0] = run("");
  const auto [n1, rng1] = run("emit hail, in (0, 0), size 200 x 110\nemit aurora, in (0, 0), size 200 x 60\n"
                              "emit rainbow, at (100, 110), radius 80\nemit meteors, in (0, 0), size 100 x 30, rate 6\nemit beam, along (10, 60) to (190, 70)\n"
                              "emit shield, at (60, 70), radius 30, rate 4\nemit glints, at (100, 60), spread 50\n");
  EXPECT_EQ(rng0, rng1) << "their shapes and timings come from hashes";
  EXPECT_GT(n1, n0 + 500);
}

TEST(FxElements, LinesOfLightAreLightOnly) {
  // aurora, rainbow, meteors, beams, shields and glints give light (the colour and the emission layers) and cover nothing
  sc::Scene scene = scene_of(std::string(kHead) +
                                 "emit aurora, in (0, 0), size 200 x 60\nemit rainbow, at (100, 110), radius 80\n"
                                 "emit meteors, in (0, 0), size 100 x 30, rate 6\nemit beam, along (10, 60) to (190, 70)\n"
                                 "emit shield, at (60, 70), radius 30, rate 4\nemit glints, at (100, 60), spread 50, count 40, rate 2\n",
                             2, false, true);
  std::vector<std::uint8_t> rgb(200 * 120 * 3);
  for (int f = 0; f < 20; ++f) scene.render(f, rgb);
  const auto& C = scene.frame().layer(cm::Frame::kColour).px;
  const auto& E = scene.frame().layer(cm::Frame::kEmission).px;
  const auto& N = scene.frame().layer(cm::Frame::kNormal).px;
  int emitted = 0;
  for (std::size_t i = 0; i < C.size(); i += 4) {
    ASSERT_EQ(C[i + 3], 0.f);
    ASSERT_EQ(N[i + 3], 0.f);
    for (std::size_t c = 0; c < 3; ++c) ASSERT_FLOAT_EQ(C[i + c], E[i + c]) << "their colour is their light";
    emitted += E[i] > 0.01f;
  }
  EXPECT_GT(emitted, 2000);
}

// --- the layers, the threads and the picture ----------------------------------------------------------------------------

namespace {
struct Frames {
  std::vector<std::vector<std::uint8_t>> rgb;
  std::vector<std::array<std::vector<float>, 4>> layer;
};
Frames play(const std::string& text, int threads, bool overlap, bool layers, int frames) {
  sc::Scene scene = scene_of(text, threads, overlap, layers);
  Frames out;
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(scene.width()) * static_cast<std::size_t>(scene.height()) * 3);
  for (int f = 0; f < frames; ++f) {
    scene.render(f, rgb);
    out.rgb.push_back(rgb);
    if (layers) {
      std::array<std::vector<float>, 4> l;
      for (int k = 0; k < 4; ++k) l[static_cast<std::size_t>(k)] = scene.frame().layer(static_cast<cm::Frame::Layer>(k)).px;
      out.layer.push_back(std::move(l));
    }
  }
  return out;
}
}  // namespace

TEST(FxElements, TheSameFramesOnAnyThreadsAndWithLayersOrWithout) {
  const std::string text = everything_new();
  const Frames a = play(text, 1, false, false, 45), b = play(text, 3, true, true, 45);
  for (std::size_t f = 0; f < a.rgb.size(); ++f) ASSERT_EQ(a.rgb[f], b.rgb[f]) << "frame " << f;
  const Frames c = play(text, 2, false, true, 45);
  for (std::size_t f = 0; f < c.layer.size(); ++f)
    for (std::size_t k = 0; k < 4; ++k) ASSERT_EQ(b.layer[f][k], c.layer[f][k]) << "frame " << f << ", layer " << k;
}

TEST(FxElements, TheirLayersAgree) {
  const Frames fr = play(everything_new(), 2, false, true, 45);
  int covered = 0, emitted = 0;
  for (const auto& l : fr.layer) {
    const auto &C = l[0], &A = l[1], &E = l[2], &N = l[3];
    for (std::size_t i = 0; i < C.size(); i += 4) {
      const float a = C[i + 3], thick = N[i + 3];
      ASSERT_EQ(a == 0.f, thick == 0.f) << "thickness is 0 exactly where alpha is 0 (pixel " << i / 4 << ")";
      ASSERT_EQ(A[i + 3], a);
      ASSERT_GE(a, 0.f);
      ASSERT_LE(a, 1.f);
      ASSERT_EQ(E[i + 3], 0.f);
      for (std::size_t c = 0; c < 3; ++c) {
        ASSERT_GE(E[i + c], 0.f);
        ASSERT_LE(A[i + c], a + 1e-5f) << "albedo at most its alpha";
      }
      const float n = std::sqrt(N[i] * N[i] + N[i + 1] * N[i + 1] + N[i + 2] * N[i + 2]);
      ASSERT_LE(n, a + 1e-4f);
      covered += a > 0.f;
      emitted += E[i] > 0.f;
    }
  }
  EXPECT_GT(covered, 3000) << "hail, spatter, stains and shards cover what is under them";
  EXPECT_GT(emitted, 10000) << "the lines of light, the glints and the shards' glints give light";
}

// --- the script ------------------------------------------------------------------------------------------------------

TEST(FxElements, TheCheckerSaysWhatIsWrong) {
  EXPECT_NE(error_of(std::string(kHead) + "emit shield, radius 3\n").find("emit shield needs 'at'"), std::string::npos);
  EXPECT_NE(error_of(std::string(kHead) + "emit aurora, in (0, 0)\n").find("needs 'size'"), std::string::npos);
  EXPECT_NE(error_of(std::string(kHead) + "emit beam, along (0, 0) to (9, 9), hit 3\n").find("'hit' does not apply to emit beam"), std::string::npos);
  EXPECT_NE(error_of(std::string(kHead) + "emit glints, count 3\n").find("needs 'at'"), std::string::npos);
  EXPECT_NE(error_of(std::string(kHead) + "emit beam, along (0, 0) to (9, 9), flaer 3\n").find("did you mean 'flare'"), std::string::npos);
  EXPECT_NE(error_of(std::string(kHead) + "emit beam, along (0, 0) to (9, 9), colour (1, 1)\n").find("'colour' has 3 values"), std::string::npos);
  EXPECT_NE(error_of(std::string(kHead) + "emit sparkles, at (1, 1)\n").find("cannot be emitted"), std::string::npos);
  // their keys and names reserve no word: a module may be named after them, and an older emitter does not take them
  EXPECT_EQ(error_of(std::string(kHead) + "effect tiny = \"tiny\"\nmodule flare = tiny, size 32, start 1\nmodule hit = tiny, size 32, start 1\n"
                                          "emit embers, on flare, hit\nemit beam, along (flare.x, 10) to (hit.x, 90), flare 4\nlet beam = 2\n"),
            "no error");
  EXPECT_NE(error_of(std::string(kHead) + "emit glow, at (1, 1), radius 2, flare 3\n").find("'flare' is not a property of emit"), std::string::npos);
  // the canonical form reads back the same
  const sc::Script s = sc::parse(everything_new(), "t");
  EXPECT_EQ(sc::print(sc::parse(sc::print(s), "t")), sc::print(s));
}

TEST(FxElements, ALibraryEntryUsesThemAsAnyOther) {
  const std::string cat = R"(
entry storm, category weather, size 300 x 200, anchor base, loop, show 1, about "rain and hail, glints on the ground"
param rate = 1, range (0.2, 4), about "glints a second"
emit hail, in (X - 150 * scale, Y - 200 * scale), size 300 * scale x 200 * scale, density 1
emit rain, in (X - 150 * scale, Y - 200 * scale), size 300 * scale x 200 * scale, density 2
emit glints, in (X - 120 * scale, Y - 10 * scale), size 240 * scale x 10 * scale, rate rate, radius 10 * scale
entry hit, category combat, size 200 x 200, anchor centre, once 1, show 0.3, about "a shield hit"
emit shield, at (X, Y), radius 80 * scale, hit t - 0.1, angle 180, seed 3
at 0:
  burst at (X, Y), embers 0, shards 10, spatter 5, speed 200
)";
  const std::pair<std::string, std::string> file{"t.nvfxl", cat};
  const nfx::library::Catalogue c = nfx::library::parse(std::span(&file, 1));
  ASSERT_EQ(c.entries.size(), 2u);
  EXPECT_NE(std::ranges::find(c.entries[0].elements, "hail"), c.entries[0].elements.end());
  EXPECT_NE(std::ranges::find(c.entries[1].elements, "shield"), c.entries[1].elements.end());
  nfx::library::StageSpec spec;
  spec.rooms = {{0, 2}, {1, 1}};
  const nfx::library::Stage st = nfx::library::build_stage(c, spec);
  EXPECT_NO_THROW(sc::validate(st.script));
}
