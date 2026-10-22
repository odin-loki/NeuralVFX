// Water (docs/EFFECTS_WATER.md), with the tests' untrained tiny effect for its mist, so no data is needed:
//   - determinism: the same script gives the same frames on 1, 2 and 3 threads, overlapped or not; another seed others;
//   - the engine layers: the picture the same with them or without, the same on any threads, consistent with each other;
//   - the simulation: waves spread and die and keep the pool's water, drops land, splashes crown, mist is given;
//   - the script: water statements and the splash and wave actions checked, errors with their line and column.
#include "compose_scene.hpp"
#include "script.hpp"
#include "water.hpp"
#include "water_scene.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {

namespace sc = nfx::compose::script;
namespace cm = nfx::compose;

sc::Options options(int threads, bool overlap, bool layers) {
  sc::Options o;
  o.threads = threads;
  o.overlap = overlap;
  o.layers = layers;
  return o;
}

sc::Scene scene_of(const std::string& text, int threads = 2, bool overlap = false, bool layers = false) {
  return sc::Scene(sc::parse(text, "t"), [](const std::string&) { return cm::testing::tiny_effect(3); }, options(threads, overlap, layers));
}

std::string error_of(const std::string& text) {
  try {
    sc::Scene s = scene_of(text);
  } catch (const sc::Error& e) {
    return e.what();
  }
  return "no error";
}

std::vector<std::vector<std::uint8_t>> play(const std::string& text, int threads, bool overlap, int frames) {
  sc::Scene scene = scene_of(text, threads, overlap);
  std::vector<std::vector<std::uint8_t>> out;
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(scene.width()) * static_cast<std::size_t>(scene.height()) * 3);
  for (int f = 0; f < frames; ++f) {
    scene.render(f, rgb);
    out.push_back(rgb);
  }
  return out;
}

}  // namespace

TEST(Water, TheSameFramesOnAnyNumberOfThreads) {
  const std::string text = cm::testing::kWaterScene;
  const auto one = play(text, 1, false, 60);
  const auto two = play(text, 2, false, 60);
  const auto three = play(text, 3, true, 60);
  for (std::size_t f = 0; f < one.size(); ++f) {
    ASSERT_EQ(one[f], two[f]) << "frame " << f << ": 1 and 2 threads";
    ASSERT_EQ(one[f], three[f]) << "frame " << f << ": 1 and 3 threads, overlapped";
  }
  // and the water is in them: the same scene without its water statements draws other frames
  std::string dry;
  for (std::size_t a = 0; a < text.size();) {
    const std::size_t b = text.find('\n', a);
    const std::string line = text.substr(a, b - a + 1);
    if (line.rfind("water", 0) != 0 && line.rfind("  ", 0) != 0 && line.rfind("              ", 0) != 0 && line.rfind("at ", 0) != 0 &&
        line.rfind("every", 0) != 0)
      dry += line;
    a = b + 1;
  }
  const auto without = play(dry, 2, false, 60);
  int differ = 0;
  for (std::size_t f = 0; f < one.size(); ++f) differ += one[f] != without[f];
  EXPECT_GT(differ, 55);
}

TEST(Water, AnotherSeedOtherFrames) {
  std::string text = cm::testing::kWaterScene, other = text;
  const std::size_t at = other.find("seed 4");
  ASSERT_NE(at, std::string::npos);
  other.replace(at, 6, "seed 44");
  const auto a = play(text, 2, false, 30), b = play(other, 2, false, 30);
  EXPECT_NE(a.back(), b.back());
}

TEST(Water, ItsPartsMoveAndItsMistIsGiven) {
  sc::Scene scene = scene_of(cm::testing::kWaterScene, 2);
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(scene.width()) * static_cast<std::size_t>(scene.height()) * 3);
  ASSERT_NE(scene.water(), nullptr);
  int most_crowns = 0, most_rings = 0, most_foam = 0, most_bubbles = 0, most_drips = 0, most_streaks = 0, most_nodes = 0, most_drops = 0;
  for (int f = 0; f < scene.frames(); ++f) {
    scene.render(f, rgb);
    const cm::Water::Counts c = scene.water()->counts();
    most_crowns = std::max(most_crowns, c.crowns);
    most_rings = std::max(most_rings, c.rings);
    most_foam = std::max(most_foam, c.foam);
    most_bubbles = std::max(most_bubbles, c.bubbles);
    most_drips = std::max(most_drips, c.drips);
    most_streaks = std::max(most_streaks, c.streaks);
    most_nodes = std::max(most_nodes, c.nodes);
    most_drops = std::max(most_drops, c.drops);
  }
  EXPECT_EQ(most_crowns, 2);
  EXPECT_GT(most_rings, 5);
  EXPECT_GT(most_foam, 5);
  EXPECT_GT(most_bubbles, 2);
  EXPECT_GT(most_drips, 0);
  EXPECT_GT(most_streaks, 10);
  EXPECT_GT(most_nodes, 100);
  EXPECT_GT(most_drops, 50);
  EXPECT_GT(scene.water()->mist_given(), 0.f);
  const cm::Module* mist = scene.module("mist");
  ASSERT_NE(mist, nullptr);
}

// The random detail of a learned water effect: spray thrown off where the bus has water (here the tiny effect's soot,
// which the mist module gets from what lands), and none where it has none.
TEST(Water, SprayIsThrownOffTheBusesWaterOnly) {
  const auto sprayed = [](const std::string& text) {
    sc::Scene scene = scene_of(text, 2);
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(scene.width()) * static_cast<std::size_t>(scene.height()) * 3);
    for (int f = 0; f < scene.frames(); ++f) scene.render(f, rgb);
    return scene.water()->sprayed();
  };
  const std::string base = cm::testing::kWaterScene;
  std::string none = base, off = base;
  const std::string spray = "water flecks = spray, in (60, 30), size 80 x 70,";
  ASSERT_NE(base.find(spray), std::string::npos);
  none.replace(none.find(spray), spray.size(), "water flecks = spray, in (60, 30), size 80 x 70, if 0,");
  off.replace(off.find(spray), spray.size(), "water flecks = spray, in (0, 0), size 2 x 2,");
  EXPECT_GT(sprayed(base), 100) << "spray thrown off the module's soot";
  EXPECT_EQ(sprayed(none), 0);
  EXPECT_EQ(sprayed(off), 0) << "nothing where the bus has no water";
}

// The simulation alone: a pool kicked in its middle.
TEST(Water, WavesSpreadDieAndKeepThePoolsWater) {
  cm::WaterLayout pool;
  pool.kind = cm::WaterKind::pool;
  pool.cells = 128;
  const cm::WaterLayout layout[] = {pool};
  cm::Water w(layout, 16, 1, 16);
  cm::WaterSpec& s = w.spec(0);
  s.x0 = 0;
  s.x1 = 512;
  s.y0 = s.y1 = 100;
  s.depth = 40;
  s.rate = 0;
  s.damping = 1.5f;
  w.wave(0, 256, 300, 12);
  const auto sum = [&] {
    double total = 0, most = 0;
    for (int x = 2; x < 512; x += 4) {
      total += w.pool_height(0, static_cast<float>(x));
      most = std::max(most, std::abs(static_cast<double>(w.pool_height(0, static_cast<float>(x)))));
    }
    return std::pair{total, most};
  };
  const float dt = 1.f / 30.f;
  for (int f = 0; f < 15; ++f) w.step(static_cast<float>(f) * dt, dt, f);
  EXPECT_GT(std::abs(w.pool_height(0, 256 + 60)), 0.05f) << "the waves have spread 60 pixels in half a second";
  const auto [total, most] = sum();
  EXPECT_GT(most, 0.5);
  EXPECT_LT(std::abs(total), 0.2 * most * 128) << "pushed aside, not lost";
  for (int f = 15; f < 300; ++f) w.step(static_cast<float>(f) * dt, dt, f);
  EXPECT_LT(sum().second, 0.05 * most) << "and they die";
}

TEST(Water, ASplashThrowsACrownAndDropsThatLand) {
  cm::WaterLayout pool;
  pool.kind = cm::WaterKind::pool;
  pool.cells = 64;
  const cm::WaterLayout layout[] = {pool};
  cm::Water w(layout, 400, 2, 32);
  cm::WaterSpec& s = w.spec(0);
  s.x0 = 0;
  s.x1 = 400;
  s.y0 = s.y1 = 300;
  s.rate = 0;
  const float dt = 1.f / 30.f;
  w.step(0, dt, 0);
  w.splash(200, 300, 1, 0, nullptr, 1, 0, 7, dt, 1);
  int most = 0, nodes = 0;
  for (int f = 1; f < 150; ++f) {
    w.step(static_cast<float>(f) * dt, dt, f);
    most = std::max(most, w.drops_alive());
    nodes = std::max(nodes, w.counts().nodes);
  }
  EXPECT_GT(most, 30);
  EXPECT_GT(nodes, 10) << "the crown's walls and the crater's jet";
  EXPECT_EQ(w.drops_alive(), 0) << "all landed in five seconds";
  EXPECT_EQ(w.counts().crowns, 0);
  EXPECT_EQ(w.mist_given(), 0.f) << "no mist without a module";
}

TEST(Water, ThePictureHasRoomForEverything) {
  sc::Scene scene = scene_of(cm::testing::kWaterScene, 2);
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(scene.width()) * static_cast<std::size_t>(scene.height()) * 3);
  cm::WaterPicture pic;
  scene.water()->reserve(pic, scene.width(), scene.height());
  const std::size_t room = pic.prims_.capacity();
  for (int f = 0; f < scene.frames(); ++f) {
    scene.render(f, rgb);
    scene.water()->picture(pic, 0, 0, scene.width(), scene.height());
    ASSERT_EQ(pic.prims_.capacity(), room) << "frame " << f;
    ASSERT_LT(pic.prims_.size(), room) << "frame " << f << ": nothing left out";
  }
}

// --- engine layers -----------------------------------------------------------------------------------------------------

namespace {

struct Layers {
  std::vector<std::uint8_t> rgb;
  std::array<std::vector<float>, 4> layer;
};

std::vector<Layers> play_layers(int threads, bool overlap, bool layers, int frames) {
  sc::Scene scene = scene_of(cm::testing::kWaterScene, threads, overlap, layers);
  std::vector<Layers> out;
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(scene.width()) * static_cast<std::size_t>(scene.height()) * 3);
  for (int f = 0; f < frames; ++f) {
    scene.render(f, rgb);
    Layers l;
    l.rgb = rgb;
    if (layers)
      for (int k = 0; k < 4; ++k) l.layer[static_cast<std::size_t>(k)] = scene.frame().layer(static_cast<cm::Frame::Layer>(k)).px;
    out.push_back(std::move(l));
  }
  return out;
}

}  // namespace

TEST(Water, ThePictureIsTheSameWithLayersOrWithout) {
  for (const bool overlap : {false, true}) {
    const auto off = play_layers(2, overlap, false, 45);
    const auto on = play_layers(2, overlap, true, 45);
    for (std::size_t f = 0; f < off.size(); ++f) ASSERT_EQ(off[f].rgb, on[f].rgb) << "frame " << f << (overlap ? ", overlapped" : "");
  }
}

TEST(Water, TheSameLayersOnAnyNumberOfThreads) {
  const auto a = play_layers(1, false, true, 45);
  const auto b = play_layers(3, true, true, 45);
  for (std::size_t f = 0; f < a.size(); ++f)
    for (std::size_t k = 0; k < 4; ++k) ASSERT_EQ(a[f].layer[k], b[f].layer[k]) << "frame " << f << ", layer " << k;
}

TEST(Water, ItsLayersAgreeWithEachOther) {
  const auto frames = play_layers(2, false, true, 45);
  int covered = 0, facing_up = 0;
  for (const Layers& l : frames) {
    const auto& C = l.layer[0];
    const auto& A = l.layer[1];
    const auto& E = l.layer[2];
    const auto& N = l.layer[3];
    for (std::size_t i = 0; i < C.size(); i += 4) {
      const float a = C[i + 3], thick = N[i + 3];
      ASSERT_EQ(a == 0.f, thick == 0.f) << "thickness is 0 exactly where alpha is 0 (pixel " << i / 4 << ")";
      ASSERT_EQ(A[i + 3], a);
      ASSERT_GE(a, 0.f);
      ASSERT_LE(a, 1.f);
      ASSERT_EQ(E[i + 3], 0.f);
      for (int c = 0; c < 3; ++c) {
        ASSERT_GE(E[i + static_cast<std::size_t>(c)], 0.f);
        ASSERT_LE(A[i + static_cast<std::size_t>(c)], a + 1e-5f);
      }
      const float n = std::sqrt(N[i] * N[i] + N[i + 1] * N[i + 1] + N[i + 2] * N[i + 2]);
      ASSERT_LE(n, a + 1e-4f);
      ASSERT_GE(N[i + 2], 0.f);
      covered += a > 0.f;
      facing_up += a > 0.1f && N[i + 1] > 0.3f * a;  // the pool's surface seen from above faces the sky
    }
  }
  EXPECT_GT(covered, 5000);
  EXPECT_GT(facing_up, 500) << "the pool's surface gives a normal that faces up";
}

// --- the script --------------------------------------------------------------------------------------------------------

TEST(Water, ScriptsAreCheckedWithLineAndColumn) {
  const std::string head = "scene size 100 x 80, fps 30, length 1 s, ground 70\neffect tiny = \"tiny\"\nmodule m = tiny, size 32\n";
  EXPECT_NE(error_of(head + "water w = lake, along (0, 50) to (90, 50)\n").find("t:4:11:"), std::string::npos)
      << error_of(head + "water w = lake, along (0, 50) to (90, 50)\n");
  EXPECT_NE(error_of(head + "water w = pool, depth 10\n").find("needs 'along'"), std::string::npos);
  EXPECT_NE(error_of(head + "water w = jet, at (10, 10)\n").find("needs 'speed'"), std::string::npos);
  EXPECT_NE(error_of(head + "water w = pool, along (0, 50) to (90, 50), speed 100, density 3\n").find("t:4:"), std::string::npos);
  EXPECT_NE(error_of(head + "water w = sheet, along (0, 5) to (9, 5), onto m\n").find("t:4:"), std::string::npos);
  EXPECT_NE(error_of(head + "water a = jet, at (1, 1), speed 9\nwater w = sheet, along (0, 5) to (9, 5), onto a\n").find("t:5:"), std::string::npos);
  EXPECT_NE(error_of(head + "water w = sheet, along (0, 5) to (9, 5), mist nothing\n").find("t:4:"), std::string::npos);
  EXPECT_NE(error_of(head + "water w = sheet, along (0, 5) to (9, 5), strands 999\n").find("t:4:"), std::string::npos);
  EXPECT_NE(error_of(head + "water w = sheet, along (0, 5) to (9, 5), strands t\n").find("t:4:"), std::string::npos);
  EXPECT_NE(error_of(head + "water w = pool, along (0, 50) to (90, 50)\nevery frame: splash at (5, 50), onto w\n").find("t:5:"), std::string::npos);
  EXPECT_NE(error_of(head + "water w = jet, at (1, 1), speed 9\nat 0.5: wave w at (5, 50)\n").find("t:5:"), std::string::npos);
  EXPECT_NE(error_of(head + "water w = pool, along (0, 50) to (90, 50)\nat 0.5: wave w, amount 3\n").find("t:5:"), std::string::npos);
  EXPECT_NE(error_of(head + "water m = pool, along (0, 50) to (90, 50)\n").find("t:4:"), std::string::npos) << "a name taken";
  EXPECT_EQ(error_of(head + "water w = pool, along (0, 50) to (90, 50), surface 8, rate 2\nat 0.5: splash at (5, 50), onto w, mist m, power 0.5\n"
                            "every frame: wave w at (5 + t, 50), amount 3, width 4\n"),
            "no error");
}

TEST(Water, ThePrinterGivesBackWhatItRead) {
  const sc::Script s = sc::parse(cm::testing::kWaterScene, "t");
  const std::string printed = sc::print(s);
  EXPECT_EQ(sc::print(sc::parse(printed, "t")), printed);
  EXPECT_NE(printed.find("water falls = sheet"), std::string::npos);
  EXPECT_NE(printed.find("splash at (110, 100)"), std::string::npos);
  EXPECT_NE(printed.find("wave lake, at (30 + 60 * t, 100)"), std::string::npos);
}
