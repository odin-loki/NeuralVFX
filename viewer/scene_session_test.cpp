// The viewer's scene mode without a display (scene_session.hpp): edits are debounced, checked and rebuilt off the
// frame; a broken edit, whether the check or the build finds it, keeps the last good scene playing and says where;
// inputs set by the user carry over a rebuild; seeks and restarts run on the worker; a frame allocates nothing.
// Untrained stand-in effects, so no data is needed.
#include "scene_session.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <new>
#include <sstream>
#include <string>
#include <unistd.h>

// Heap allocations are counted (the global operator new is replaced, as in tests/alloc_test.cpp) for the test that a
// frame of the scene mode allocates nothing.
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

namespace v = nfx::viewer;
using namespace std::chrono_literals;

namespace {

// A folder with stand-in effect files, read as a game reads its effects.
struct Folder {
  std::filesystem::path dir;
  Folder() {
    static std::atomic<int> n{0};
    dir = std::filesystem::temp_directory_path() / ("nvfx_viewer_test_" + std::to_string(::getpid()) + "_" + std::to_string(n++));
    std::filesystem::create_directories(dir);
    for (const char* name : {"fire", "smoke"}) {
      std::ofstream f(dir / (std::string(name) + ".nvfx"), std::ios::binary);
      f << v::stand_in_bytes(name);
    }
  }
  ~Folder() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};

v::SessionSettings settings(const Folder& f, v::StandIns stand_ins = v::StandIns::off) {
  v::SessionSettings s;
  s.effects_dir = f.dir.string();
  s.stand_ins = stand_ins;
  s.threads = 1;  // a scene on the worker and one on this thread: two threads
  s.debounce = 300ms;
  return s;
}

constexpr const char* kScript = R"(scene size 160 x 90, fps 30, length 3, ground 80
effect fire = "fire.nvfx"
input power = 0.4
module a = fire, size 64, width 64, at (60, ground), start 1, seed 5, intensity power
module b = fire, size 32, at (120, ground), start 0, seed 6, waiting
when 0 as boom, repeat:
  shock at (80, 50), speed 600
when t > 100 as never:
  wake b
)";

std::string replaced(std::string s, const std::string& from, const std::string& to) {
  const std::size_t at = s.find(from);
  EXPECT_NE(at, std::string::npos) << from;
  if (at != std::string::npos) s.replace(at, from.size(), to);
  return s;
}

float input_of(const v::SceneSession& s, const char* name) {
  float value = -1.f;
  EXPECT_EQ(nvfx_scene_get_input(s.scene(), name, &value), NVFX_OK);
  return value;
}

// The luma spread of the session's picture: a flat picture has none.
double spread(const std::vector<std::uint8_t>& rgba) {
  double sum = 0.0, sum2 = 0.0;
  const std::size_t n = rgba.size() / 4;
  for (std::size_t i = 0; i < n; ++i) {
    const double l = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
    sum += l;
    sum2 += l * l;
  }
  const double mean = sum / static_cast<double>(n);
  return std::sqrt(std::max(0.0, sum2 / static_cast<double>(n) - mean * mean));
}

}  // namespace

TEST(ViewerScene, DebounceFiresOnceWhenEditsPause) {
  v::Debouncer d(300ms);
  const v::Clock::time_point t0 = v::Clock::time_point{} + 10s;
  EXPECT_FALSE(d.due(t0));
  d.touch(t0);
  EXPECT_TRUE(d.pending());
  EXPECT_FALSE(d.due(t0 + 299ms));
  d.touch(t0 + 200ms);  // another edit restarts the wait
  EXPECT_FALSE(d.due(t0 + 450ms));
  EXPECT_TRUE(d.due(t0 + 500ms));
  EXPECT_FALSE(d.due(t0 + 900ms));  // once
  EXPECT_FALSE(d.pending());
  d.touch(t0 + 1s);
  d.cancel();
  EXPECT_FALSE(d.due(t0 + 2s));
}

TEST(ViewerScene, ABrokenEditKeepsTheLastGoodScene) {
  const Folder f;
  v::SceneSession s(settings(f));
  s.open(kScript, "t.nvfxs");
  ASSERT_TRUE(s.settle(120s));
  ASSERT_TRUE(s.scene());
  ASSERT_FALSE(s.error()) << s.error()->message;
  const std::uint64_t gen = s.generation();
  EXPECT_EQ(gen, 1u);
  ASSERT_EQ(s.info().width, 160);
  ASSERT_EQ(s.info().height, 90);
  s.step(0.2, 10);
  EXPECT_EQ(s.frame(), 6);
  ASSERT_TRUE(s.draw());
  EXPECT_FALSE(s.draw());  // the same frame is not drawn again
  EXPECT_GT(spread(s.picture()), 1.0);

  // a typo: found by the check once the edits pause, and nothing is built
  const v::Clock::time_point t0 = v::Clock::now();
  s.edit(replaced(kScript, "intensity power", "intensity powr"), t0);
  s.update(t0 + 100ms);
  EXPECT_FALSE(s.error());
  EXPECT_TRUE(s.edit_pending());
  s.update(t0 + 400ms);
  ASSERT_TRUE(s.error());
  EXPECT_TRUE(s.error()->from_check);
  EXPECT_EQ(s.error()->line, 4);
  EXPECT_GT(s.error()->column, 60);
  EXPECT_EQ(s.error()->message.rfind("t.nvfxs:4:", 0), 0u) << s.error()->message;
  EXPECT_NE(s.error()->message.find("did you mean 'power'"), std::string::npos) << s.error()->message;
  EXPECT_FALSE(s.building());
  EXPECT_EQ(s.generation(), gen);
  // the last good scene plays on
  ASSERT_TRUE(s.scene());
  s.step(0.1, 10);
  EXPECT_EQ(s.frame(), 9);
  EXPECT_TRUE(s.draw());

  // an error only the build finds (a tile size that is not a multiple of the effect's grid): the same
  s.edit(replaced(kScript, "size 64, width 64", "size 48, width 64"), v::Clock::now());
  ASSERT_TRUE(s.settle(120s));
  ASSERT_TRUE(s.error());
  EXPECT_FALSE(s.error()->from_check);
  EXPECT_EQ(s.error()->line, 4);
  EXPECT_NE(s.error()->message.find("multiple of its effect's grid"), std::string::npos) << s.error()->message;
  EXPECT_EQ(s.generation(), gen);
  EXPECT_EQ(s.frame(), 9);

  // a missing effect file: at the line of its `effect` statement
  s.edit(replaced(kScript, "fire.nvfx", "wood.nvfx"), v::Clock::now());
  ASSERT_TRUE(s.settle(120s));
  ASSERT_TRUE(s.error());
  EXPECT_EQ(s.error()->line, 2);
  EXPECT_NE(s.error()->message.find("cannot load effect 'fire' (wood.nvfx)"), std::string::npos) << s.error()->message;
  ASSERT_EQ(s.effects().size(), 1u);
  EXPECT_EQ(s.effects()[0].from, v::EffectUse::From::missing);
  EXPECT_EQ(s.generation(), gen);

  // fixed: a new scene, played to the time of the edit off the frame
  const double before = s.time();
  s.edit(replaced(kScript, "seed 5", "seed 9"), v::Clock::now());
  ASSERT_TRUE(s.settle(120s));
  EXPECT_FALSE(s.error());
  EXPECT_EQ(s.generation(), gen + 1);
  ASSERT_TRUE(s.scene());
  EXPECT_DOUBLE_EQ(s.time(), before);
  EXPECT_EQ(s.frame(), 9);
  ASSERT_EQ(s.effects().size(), 1u);
  EXPECT_EQ(s.effects()[0].from, v::EffectUse::From::file);
  EXPECT_TRUE(s.draw());
  EXPECT_EQ(s.builds(), 4);  // the first, the build error, the missing file and the fix (the typo was never built)
}

TEST(ViewerScene, OpeningABrokenScriptGivesNoScene) {
  const Folder f;
  v::SceneSession s(settings(f));
  s.open("scene size 160 x 90\nmodule m = nothing, size 32\n", "bad.nvfxs");
  ASSERT_TRUE(s.settle(30s));
  EXPECT_FALSE(s.has_scene());
  EXPECT_FALSE(s.scene());
  ASSERT_TRUE(s.error());
  EXPECT_EQ(s.error()->line, 2);
  EXPECT_FALSE(s.draw());
  s.seek(1.0);  // nothing to seek or restart
  s.restart();
  s.step(0.5);
  EXPECT_FALSE(s.busy());
  // a good edit gives the first scene
  s.edit(kScript, v::Clock::now());
  ASSERT_TRUE(s.settle(120s));
  EXPECT_TRUE(s.has_scene());
  EXPECT_FALSE(s.error());
}

TEST(ViewerScene, InputsCarryOverARebuildUnlessTheScriptChangesThem) {
  const Folder f;
  v::SceneSession s(settings(f));
  s.open(kScript, "t.nvfxs");
  ASSERT_TRUE(s.settle(120s));
  ASSERT_EQ(s.inputs().size(), 1u);
  EXPECT_EQ(s.inputs()[0].name, "power");
  EXPECT_FLOAT_EQ(s.inputs()[0].script, 0.4f);
  s.inputs()[0].value = 0.9f;  // the slider
  s.update(v::Clock::now());
  EXPECT_FLOAT_EQ(input_of(s, "power"), 0.9f);
  // an edit elsewhere: the user's value stays
  s.edit(replaced(kScript, "seed 5", "seed 8"), v::Clock::now());
  ASSERT_TRUE(s.settle(120s));
  EXPECT_EQ(s.generation(), 2u);
  EXPECT_FLOAT_EQ(input_of(s, "power"), 0.9f);
  EXPECT_FLOAT_EQ(s.inputs()[0].value, 0.9f);
  EXPECT_FLOAT_EQ(s.inputs()[0].script, 0.4f);
  // the script's starting value changed: it wins
  s.edit(replaced(kScript, "input power = 0.4", "input power = 0.2"), v::Clock::now());
  ASSERT_TRUE(s.settle(120s));
  EXPECT_EQ(s.generation(), 3u);
  EXPECT_FLOAT_EQ(input_of(s, "power"), 0.2f);
  EXPECT_FLOAT_EQ(s.inputs()[0].value, 0.2f);
}

TEST(ViewerScene, RulesProbeSeekAndRestart) {
  const Folder f;
  v::SceneSession s(settings(f));
  s.open(kScript, "t.nvfxs");
  ASSERT_TRUE(s.settle(120s));
  ASSERT_EQ(s.rules(), (std::vector<std::string>{"boom", "never"}));
  EXPECT_TRUE(s.trigger("boom"));
  EXPECT_FALSE(s.trigger("nope"));
  s.step_frames(1);
  int count = 0;
  ASSERT_EQ(nvfx_scene_rule_state(s.scene(), "boom", &count, nullptr), NVFX_OK);
  EXPECT_EQ(count, 1);
  // the probe: the fire's heat under its tile, none in the sky far from it (no camera: world = picture)
  float wx = 0.f, wy = 0.f;
  nvfx_scene_fields hot{}, cold{};
  ASSERT_TRUE(s.probe(60.f, 76.f, wx, wy, hot));
  EXPECT_EQ(wx, 60.f);
  EXPECT_EQ(wy, 76.f);
  ASSERT_TRUE(s.probe(150.f, 5.f, wx, wy, cold));
  EXPECT_GT(hot.heat, 0.01f);
  EXPECT_EQ(cold.heat, 0.f);

  // a seek forward on the worker; then back (a rebuild and replay); a restart
  s.seek(1.0);
  EXPECT_TRUE(s.seeking());
  EXPECT_DOUBLE_EQ(s.time(), 1.0);
  ASSERT_TRUE(s.settle(120s));
  ASSERT_TRUE(s.scene());
  EXPECT_EQ(s.frame(), 30);
  EXPECT_DOUBLE_EQ(s.time(), 1.0);
  s.seek(0.5);
  ASSERT_TRUE(s.settle(120s));
  EXPECT_EQ(s.frame(), 15);
  s.seek(0.5 + 1.0 / 30.0);  // a frame forward: here, at once
  EXPECT_FALSE(s.busy());
  EXPECT_EQ(s.frame(), 16);
  s.restart();
  ASSERT_TRUE(s.settle(120s));
  EXPECT_EQ(s.frame(), 0);
  EXPECT_EQ(s.time(), 0.0);
  EXPECT_EQ(s.generation(), 1u);  // the same scene throughout
  // a later seek supersedes an earlier one
  s.seek(2.0);
  s.update(v::Clock::now());
  s.seek(0.2);
  ASSERT_TRUE(s.settle(120s));
  EXPECT_EQ(s.frame(), 6);
}

TEST(ViewerScene, StandInsForMissingEffects) {
  const Folder f;
  const std::string script = replaced(kScript, "fire.nvfx", "explosion.nvfx");  // not in the folder
  v::SceneSession s(settings(f, v::StandIns::off));
  s.open(script, "t.nvfxs");
  ASSERT_TRUE(s.settle(120s));
  EXPECT_FALSE(s.has_scene());
  ASSERT_TRUE(s.error());
  EXPECT_EQ(s.error()->line, 2);
  v::SessionSettings set = s.settings();
  set.stand_ins = v::StandIns::missing;
  s.set_settings(set);  // rebuilds
  ASSERT_TRUE(s.settle(120s));
  ASSERT_TRUE(s.has_scene());
  EXPECT_FALSE(s.error());
  ASSERT_EQ(s.effects().size(), 1u);
  EXPECT_EQ(s.effects()[0].from, v::EffectUse::From::stand_in);
  // the fireball of the examples, every effect a stand-in: it builds and plays
  std::ifstream in(std::filesystem::path(NEURALFX_SOURCE_DIR) / "examples/scenes/fireball.nvfxs");
  std::stringstream ss;
  ss << in.rdbuf();
  set.stand_ins = v::StandIns::all;
  s.set_settings(set);
  s.open(ss.str(), "fireball.nvfxs", 1.3);
  ASSERT_TRUE(s.settle(300s));
  ASSERT_TRUE(s.scene()) << (s.error() ? s.error()->message : "");
  EXPECT_EQ(s.info().width, 1280);
  EXPECT_EQ(s.frame(), 39);
  EXPECT_EQ(s.effects().size(), 3u);
  for (const v::EffectUse& e : s.effects()) EXPECT_EQ(e.from, v::EffectUse::From::stand_in) << e.name;
  ASSERT_TRUE(s.draw());
  EXPECT_GT(spread(s.picture()), 2.0);
  int fired = 0;
  ASSERT_EQ(nvfx_scene_rule_state(s.scene(), "detonate", &fired, nullptr), NVFX_OK);
  EXPECT_EQ(fired, 1);
}

TEST(ViewerScene, ANewerEditSupersedesABuildInProgress) {
  const Folder f;
  v::SceneSession s(settings(f));
  s.open(kScript, "t.nvfxs", 2.0);  // a build that replays 60 frames
  s.update(v::Clock::now());
  EXPECT_TRUE(s.building());
  s.edit(std::string(kScript) + "at 0.5 as later:\n  wake b\n", v::Clock::now() - 1s);  // already quiet for long enough
  ASSERT_TRUE(s.settle(120s));
  ASSERT_TRUE(s.has_scene());
  EXPECT_EQ(s.rules(), (std::vector<std::string>{"boom", "never", "later"}));
  EXPECT_EQ(s.frame(), 0);  // the second build: no scene was playing when it was asked for, so from 0
  EXPECT_FALSE(s.error());
}

// A UI frame of the scene mode through the session allocates nothing once the scene is built: the work between edits,
// the clock, inputs, a rule fired, the probe and the picture, overlapped (the scene API's own frames are counted by
// nvfx_alloc_test).
TEST(ViewerScene, AFrameAllocatesNothing) {
  const Folder f;
  v::SessionSettings set = settings(f);
  set.threads = 2;
  v::SceneSession s(set);
  s.open(kScript, "t.nvfxs");
  ASSERT_TRUE(s.settle(120s));
  ASSERT_TRUE(s.draw());
  ASSERT_EQ(s.info().overlap, 1);
  ASSERT_EQ(s.inputs().size(), 1u);
  const std::string boom = "boom";
  float wx = 0.f, wy = 0.f, heat = 0.f;
  nvfx_scene_fields fields{};
  g_allocations = 0;
  g_counting = true;
  for (int i = 0; i < 120; ++i) {
    s.update(v::Clock::now());
    s.inputs()[0].value = 0.3f + 0.005f * static_cast<float>(i);
    if (i % 30 == 0) s.trigger(boom);
    s.step(1.0 / 60.0);
    s.probe(60.f, 70.f, wx, wy, fields);
    heat = std::max(heat, fields.heat);
    s.draw();
  }
  g_counting = false;
  EXPECT_EQ(g_allocations.load(), 0);
  EXPECT_EQ(s.frame(), 60);
  EXPECT_GT(heat, 0.f);
  int count = 0;
  ASSERT_EQ(nvfx_scene_rule_state(s.scene(), "boom", &count, nullptr), NVFX_OK);
  EXPECT_EQ(count, 4);
}
