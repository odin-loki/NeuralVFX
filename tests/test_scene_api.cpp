// The C API for scenes (include/neuralfx/nvfx_scene.h): it plays a script as the script runner does, frame for frame
// and to the bit (skipped pictures, restarts and seeks included), its clock computes the frames whose time has come,
// the game reads the fields and drives the scene with inputs, triggers and module settings, and errors say where.
// With the study D effects (outside git), the fireball through the API reproduces the frozen keyframes.
#include "compose_scene.hpp"
#include "script.hpp"

#include <neuralfx/dcm/mixer.hpp>
#include <neuralfx/image_io.hpp>
#include <neuralfx/model.hpp>
#include <neuralfx/nvfx_scene.h>
#include <neuralfx/rollout.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "nvfx_internal.hpp"

namespace {

namespace sc = nfx::compose::script;
using nfx::compose::testing::kEverything;

std::size_t z(int v) { return static_cast<std::size_t>(v); }

struct EffectDeleter {
  void operator()(nvfx_effect* e) const { nvfx_effect_free(e); }
};
using EffectPtr = std::unique_ptr<nvfx_effect, EffectDeleter>;
struct SceneDeleter {
  void operator()(nvfx_scene* s) const { nvfx_scene_free(s); }
};
using ScenePtr = std::unique_ptr<nvfx_scene, SceneDeleter>;

// The tests' untrained rollout effects, through the file format and nvfx_effect_load_memory, as a game would load them.
EffectPtr tiny_nvfx(std::uint64_t seed) {
  const nfx::rt::RolloutEffect t = nfx::compose::testing::tiny_effect(seed);
  std::ostringstream os;
  EXPECT_TRUE(nfx::rollout::save_model(os, t.m));
  const std::string bytes = os.str();
  nvfx_effect* e = nullptr;
  EXPECT_EQ(nvfx_effect_load_memory(bytes.data(), bytes.size(), &e), NVFX_OK);
  return EffectPtr(e);
}

struct Tiny {
  EffectPtr tiny = tiny_nvfx(3), other = tiny_nvfx(4);
  std::array<nvfx_scene_effect, 2> list() const { return {{{"tiny", tiny.get()}, {"other", other.get()}}}; }
  sc::EffectLoader loader() const {
    return [this](const std::string& f) { return *nfx::rt::rollout_of(f == "other" ? other.get() : tiny.get()); };
  }
};

ScenePtr make(const Tiny& fx, const char* text, int threads, int overlap, nvfx_scene_error* err = nullptr, int w = 0, int h = 0) {
  const auto list = fx.list();
  nvfx_scene_desc d;
  nvfx_scene_desc_init(&d);
  d.script = text;
  d.source_name = "t";
  d.effects = list.data();
  d.n_effects = 2;
  d.threads = threads;
  d.overlap = overlap;
  d.width = w;
  d.height = h;
  nvfx_scene* s = nullptr;
  nvfx_scene_error e;
  const nvfx_status st = nvfx_scene_create(&d, &s, err ? err : &e);
  if (!err) {
    EXPECT_EQ(st, NVFX_OK) << e.message;
  }
  return ScenePtr(s);
}

nvfx_scene_info info_of(const nvfx_scene* s) {
  nvfx_scene_info i{};
  EXPECT_EQ(nvfx_scene_get_info(s, &i), NVFX_OK);
  return i;
}

// The current frame as RGB (the alpha is 255 everywhere).
std::vector<std::uint8_t> picture(nvfx_scene* s) {
  const nvfx_scene_info i = info_of(s);
  std::vector<std::uint8_t> rgba(z(i.width) * z(i.height) * 4), rgb(z(i.width) * z(i.height) * 3);
  EXPECT_EQ(nvfx_scene_render(s, rgba.data(), z(i.width) * 4), NVFX_OK);
  for (std::size_t p = 0; p < z(i.width) * z(i.height); ++p) {
    for (std::size_t c = 0; c < 3; ++c) rgb[p * 3 + c] = rgba[p * 4 + c];
    EXPECT_EQ(rgba[p * 4 + 3], 255);
  }
  return rgb;
}

// A scene the game drives: inputs, a rule only the game fires, a landing rule, a waiting module.
constexpr const char* kGameScript = R"(
scene size 160 x 90, fps 30, length 4, ground 80
effect tiny = "tiny"
input spot = 40                       # where the game wants the fire
input power = 0.3
module fire = tiny, size 32, at (spot, ground), start 1, seed 5, intensity power
module torch = tiny, size 32, at (120, ground), start 0, seed 6
module spare = tiny, size 32, at (20, ground), start 0, seed 7, waiting
when 0 as boom, repeat:               # only the game fires it
  shock at (80, 50), speed 600
when ember lands:
  stop spare
at 100 as late:
  wake spare
)";

// Every frame of the scene that uses every statement, through the API and through the script runner: the same bits,
// on 1 and 2 threads, overlapped or not.
TEST(SceneApi, PlaysTheSameFramesAsTheScriptRunner) {
  const Tiny fx;
  const sc::Script script = sc::parse(kEverything, "t");
  for (const auto& [threads, overlap] : {std::pair{1, 0}, std::pair{2, 0}, std::pair{2, 1}}) {
    ScenePtr api = make(fx, kEverything, threads, overlap);
    ASSERT_TRUE(api);
    sc::Scene ref(script, fx.loader(), {threads, nfx::compose::best_isa(), overlap != 0});
    const nvfx_scene_info i = info_of(api.get());
    ASSERT_EQ(i.width, 160);
    ASSERT_EQ(i.height, 90);
    ASSERT_EQ(i.frames, ref.frames());
    EXPECT_EQ(i.overlap, overlap);
    EXPECT_EQ(i.n_modules, 4);
    EXPECT_EQ(i.n_rules, 7);
    std::vector<std::uint8_t> want(160 * 90 * 3);
    for (int f = 0; f < i.frames; ++f) {
      if (f > 0) {
        ASSERT_EQ(nvfx_scene_step_frames(api.get(), 1), NVFX_OK);
      }
      EXPECT_EQ(nvfx_scene_frame(api.get()), f);
      ref.render(f, want);
      ASSERT_EQ(picture(api.get()), want) << "frame " << f << ", " << threads << " threads, overlap " << overlap;
    }
    int count = 0;
    float when = 0.f;
    EXPECT_EQ(nvfx_scene_rule_state(api.get(), "boom", &count, &when), NVFX_OK);
    EXPECT_EQ(count, 1);
    EXPECT_NEAR(when, 3.f / 30.f, 1e-6f);
  }
}

// Frames computed and not drawn change nothing in the frames drawn after them; drawing a frame twice copies it.
TEST(SceneApi, SkippedPicturesChangeNothing) {
  const Tiny fx;
  for (const int overlap : {0, 1}) {
    ScenePtr every = make(fx, kEverything, 2, overlap), some = make(fx, kEverything, 2, overlap);
    std::map<int, std::vector<std::uint8_t>> all;
    for (int f = 0; f < 45; ++f) {
      if (f > 0) nvfx_scene_step_frames(every.get(), 1);
      all[f] = picture(every.get());
    }
    for (const int f : {0, 3, 4, 9, 20, 21, 44}) {
      ASSERT_EQ(nvfx_scene_step_frames(some.get(), f - nvfx_scene_frame(some.get())), NVFX_OK);
      EXPECT_EQ(picture(some.get()), all[f]) << "frame " << f << ", overlap " << overlap;
      EXPECT_EQ(picture(some.get()), all[f]) << "frame " << f << " again";
    }
  }
}

// Scenes are independent: two played at once on two threads give the frames each gives alone.
TEST(SceneApi, TwoScenesOnTwoThreadsAreIndependent) {
  const Tiny fx;
  const auto play = [&](nvfx_scene* s, std::vector<std::vector<std::uint8_t>>& out) {
    for (int f = 0; f < 30; ++f) {
      if (f > 0) nvfx_scene_step(s, 1.0 / 30.0, nullptr);
      if (f % 3 == 0) out.push_back(picture(s));
    }
  };
  std::vector<std::vector<std::uint8_t>> alone_a, alone_b, a, b;
  {
    ScenePtr x = make(fx, kEverything, 2, 1), y = make(fx, kGameScript, 2, 0);
    play(x.get(), alone_a);
    play(y.get(), alone_b);
  }
  ScenePtr x = make(fx, kEverything, 2, 1), y = make(fx, kGameScript, 2, 0);
  std::thread ta([&] { play(x.get(), a); });
  std::thread tb([&] { play(y.get(), b); });
  ta.join();
  tb.join();
  EXPECT_EQ(a, alone_a);
  EXPECT_EQ(b, alone_b);
}

// The clock: a step computes the frames whose time has come (frame f at f / fps); seeking back rebuilds the scene and
// gives the same frames; a restart starts again from frame 0.
TEST(SceneApi, ClockSeekAndRestart) {
  const Tiny fx;
  ScenePtr s = make(fx, kEverything, 2, 1);
  std::vector<std::vector<std::uint8_t>> first;
  int n = -1;
  EXPECT_EQ(nvfx_scene_step(s.get(), 1.0 / 60.0, &n), NVFX_OK);
  EXPECT_EQ(n, 0);
  EXPECT_EQ(nvfx_scene_frame(s.get()), 0);
  EXPECT_EQ(nvfx_scene_step(s.get(), 1.0 / 60.0, &n), NVFX_OK);
  EXPECT_EQ(n, 1);  // two half frames make one, whatever the rounding of 1/60
  EXPECT_EQ(nvfx_scene_step(s.get(), 0.5, &n), NVFX_OK);
  EXPECT_EQ(n, 15);
  EXPECT_EQ(nvfx_scene_frame(s.get()), 16);
  EXPECT_NEAR(nvfx_scene_time(s.get()), 1.0 / 30.0 + 0.5, 1e-12);
  EXPECT_EQ(nvfx_scene_step(s.get(), -0.1, nullptr), NVFX_ERROR_ARGUMENT);
  const std::vector<std::uint8_t> at16 = picture(s.get());
  ASSERT_EQ(nvfx_scene_seek(s.get(), 30.0 / 30.0), NVFX_OK);  // forwards
  EXPECT_EQ(nvfx_scene_frame(s.get()), 30);
  const std::vector<std::uint8_t> at30 = picture(s.get());
  int boom = 0;
  ASSERT_EQ(nvfx_scene_rule_state(s.get(), "boom", &boom, nullptr), NVFX_OK);
  EXPECT_EQ(boom, 1);
  ASSERT_EQ(nvfx_scene_seek(s.get(), 16.0 / 30.0), NVFX_OK);  // backwards: rebuilt, played to frame 16
  EXPECT_EQ(nvfx_scene_frame(s.get()), 16);
  EXPECT_EQ(picture(s.get()), at16);
  ASSERT_EQ(nvfx_scene_restart(s.get()), NVFX_OK);
  EXPECT_EQ(nvfx_scene_frame(s.get()), 0);
  EXPECT_EQ(nvfx_scene_time(s.get()), 0.0);
  ASSERT_EQ(nvfx_scene_rule_state(s.get(), "boom", &boom, nullptr), NVFX_OK);
  EXPECT_EQ(boom, 0);
  ASSERT_EQ(nvfx_scene_step_frames(s.get(), 30), NVFX_OK);
  EXPECT_EQ(picture(s.get()), at30);
}

// The game drives the scene: inputs read by expressions, rules fired by name, modules moved and their controls set.
TEST(SceneApi, InputsTriggersAndModules) {
  const Tiny fx;
  for (const int overlap : {0, 1}) {
    ScenePtr s = make(fx, kGameScript, 2, overlap);
    ASSERT_TRUE(s);
    const nvfx_scene_info i = info_of(s.get());
    ASSERT_EQ(i.n_inputs, 2);
    EXPECT_STREQ(nvfx_scene_input_name(s.get(), 0), "spot");
    EXPECT_STREQ(nvfx_scene_input_name(s.get(), 1), "power");
    EXPECT_EQ(nvfx_scene_input_name(s.get(), 2), nullptr);
    EXPECT_STREQ(nvfx_scene_module_name(s.get(), 1), "torch");
    EXPECT_STREQ(nvfx_scene_rule_name(s.get(), 0), "boom");
    nvfx_scene_module_info m{};
    ASSERT_EQ(nvfx_scene_module_get_info(s.get(), "fire", &m), NVFX_OK);
    EXPECT_FLOAT_EQ(m.x, 40.f);
    EXPECT_FLOAT_EQ(m.y, 80.f);
    EXPECT_EQ(m.active, 1);
    EXPECT_EQ(m.tiles, 1);
    ASSERT_EQ(m.n_controls, 3);
    EXPECT_FLOAT_EQ(m.controls[0], 0.3f);
    EXPECT_STREQ(nvfx_scene_module_control_name(s.get(), "fire", 0), "intensity");
    EXPECT_EQ(nvfx_scene_module_control_name(s.get(), "fire", 3), nullptr);
    // inputs act from the next frame computed (overlapped: the one after it)
    ASSERT_EQ(nvfx_scene_set_input(s.get(), "spot", 100.f), NVFX_OK);
    ASSERT_EQ(nvfx_scene_set_input(s.get(), "power", 0.9f), NVFX_OK);
    float v = 0.f;
    ASSERT_EQ(nvfx_scene_get_input(s.get(), "spot", &v), NVFX_OK);
    EXPECT_EQ(v, 100.f);
    EXPECT_EQ(nvfx_scene_set_input(s.get(), "nope", 1.f), NVFX_ERROR_ARGUMENT);
    (void)picture(s.get());
    nvfx_scene_step_frames(s.get(), 2);
    ASSERT_EQ(nvfx_scene_module_get_info(s.get(), "fire", &m), NVFX_OK);
    EXPECT_FLOAT_EQ(m.x, 100.f);
    EXPECT_FLOAT_EQ(m.controls[0], 0.9f);
    // a rule only the game fires; `repeat` lets it fire again
    int count = -1;
    ASSERT_EQ(nvfx_scene_rule_state(s.get(), "boom", &count, nullptr), NVFX_OK);
    EXPECT_EQ(count, 0);
    ASSERT_EQ(nvfx_scene_trigger(s.get(), "boom"), NVFX_OK);
    (void)picture(s.get());
    nvfx_scene_step_frames(s.get(), 2);
    float when = 0.f;
    ASSERT_EQ(nvfx_scene_rule_state(s.get(), "boom", &count, &when), NVFX_OK);
    EXPECT_EQ(count, 1);
    EXPECT_GE(when, 2.f / 30.f);
    ASSERT_EQ(nvfx_scene_trigger(s.get(), "boom"), NVFX_OK);
    nvfx_scene_step_frames(s.get(), 2);
    ASSERT_EQ(nvfx_scene_rule_state(s.get(), "boom", &count, nullptr), NVFX_OK);
    EXPECT_EQ(count, 2);
    // a rule on time fired early by the game; a waiting module woken by it
    ASSERT_EQ(nvfx_scene_trigger(s.get(), "late"), NVFX_OK);
    nvfx_scene_step_frames(s.get(), 2);
    ASSERT_EQ(nvfx_scene_module_get_info(s.get(), "spare", &m), NVFX_OK);
    EXPECT_EQ(m.active, 1);
    EXPECT_LT(m.started, 1.f);
    EXPECT_EQ(nvfx_scene_trigger(s.get(), "line 11"), NVFX_ERROR_ARGUMENT);  // a landing rule
    EXPECT_EQ(nvfx_scene_trigger(s.get(), "nope"), NVFX_ERROR_ARGUMENT);
    // modules moved and their controls set by the game
    ASSERT_EQ(nvfx_scene_module_place(s.get(), "torch", 60.f, 70.f), NVFX_OK);
    ASSERT_EQ(nvfx_scene_module_set_control(s.get(), "torch", "wind", 0.8f), NVFX_OK);
    ASSERT_EQ(nvfx_scene_module_get_info(s.get(), "torch", &m), NVFX_OK);
    EXPECT_FLOAT_EQ(m.x, 60.f);
    EXPECT_FLOAT_EQ(m.y, 70.f);
    EXPECT_FLOAT_EQ(m.controls[1], 0.8f);
    EXPECT_EQ(nvfx_scene_module_set_control(s.get(), "torch", "colour", 1.f), NVFX_ERROR_ARGUMENT);
    EXPECT_EQ(nvfx_scene_module_place(s.get(), "nope", 1.f, 1.f), NVFX_ERROR_ARGUMENT);
    // a restart keeps the inputs' values as the starting ones
    ASSERT_EQ(nvfx_scene_restart(s.get()), NVFX_OK);
    ASSERT_EQ(nvfx_scene_module_get_info(s.get(), "fire", &m), NVFX_OK);
    EXPECT_FLOAT_EQ(m.x, 100.f);
    ASSERT_EQ(nvfx_scene_rule_state(s.get(), "boom", &count, nullptr), NVFX_OK);
    EXPECT_EQ(count, 0);
  }
  // starting values of inputs from the description; unknown ones are an error
  const auto list = fx.list();
  const nvfx_scene_input start[] = {{"spot", 70.f}};
  nvfx_scene_desc d;
  nvfx_scene_desc_init(&d);
  d.script = kGameScript;
  d.effects = list.data();
  d.n_effects = 2;
  d.inputs = start;
  d.n_inputs = 1;
  nvfx_scene* s = nullptr;
  nvfx_scene_error err;
  ASSERT_EQ(nvfx_scene_create(&d, &s, &err), NVFX_OK) << err.message;
  nvfx_scene_module_info m{};
  ASSERT_EQ(nvfx_scene_module_get_info(s, "fire", &m), NVFX_OK);
  EXPECT_FLOAT_EQ(m.x, 70.f);
  nvfx_scene_free(s);
  const nvfx_scene_input bad[] = {{"spott", 70.f}};
  d.inputs = bad;
  EXPECT_EQ(nvfx_scene_create(&d, &s, &err), NVFX_ERROR_ARGUMENT);
  EXPECT_NE(std::string(err.message).find("no input 'spott' (did you mean 'spot'?)"), std::string::npos) << err.message;
}

// Fields for gameplay: heat where the fire is, none far from it; as a point, a grid and a region.
TEST(SceneApi, FieldsShowHeatWhereTheFireIs) {
  const Tiny fx;
  ScenePtr s = make(fx, R"(
scene size 160 x 90, fps 30, length 2, ground 80
bus at (-16, -16), size 192 x 128, cell 4
effect tiny = "tiny"
module fire = tiny, size 32, width 32, at (40, ground), start 1, seed 5
)", 1, 0);
  ASSERT_TRUE(s);
  nvfx_scene_step_frames(s.get(), 10);
  nvfx_scene_fields hot{}, cold{};
  ASSERT_EQ(nvfx_scene_sample(s.get(), 40.f, 70.f, &hot), NVFX_OK);
  ASSERT_EQ(nvfx_scene_sample(s.get(), 140.f, 20.f, &cold), NVFX_OK);
  EXPECT_GT(hot.heat, 0.05f);
  EXPECT_EQ(cold.heat, 0.f);
  EXPECT_EQ(cold.soot, 0.f);
  // a grid of 8 x 4 tiles of 20 pixels: the heat is in the tiles over the fire
  std::vector<float> grid(8 * 4, -1.f);
  ASSERT_EQ(nvfx_scene_sample_grid(s.get(), NVFX_FIELD_HEAT, 10.f, 10.f, 20.f, 20.f, 8, 4, grid.data(), 8), NVFX_OK);
  float near_fire = 0.f, far = 0.f;
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 8; ++i) {
      const float v = grid[z(j * 8 + i)];
      if (i <= 2 && j >= 2) near_fire = std::max(near_fire, v);
      if (i >= 5) far = std::max(far, v);
    }
  }
  EXPECT_GT(near_fire, 0.05f);
  EXPECT_EQ(far, 0.f);
  float mx = -1.f, mean = -1.f;
  ASSERT_EQ(nvfx_scene_field_region(s.get(), NVFX_FIELD_HEAT, 24.f, 48.f, 56.f, 80.f, &mx, &mean), NVFX_OK);
  EXPECT_GT(mx, 0.05f);
  EXPECT_GT(mean, 0.f);
  EXPECT_LE(mean, mx);
  ASSERT_EQ(nvfx_scene_field_region(s.get(), NVFX_FIELD_HEAT, 120.f, 0.f, 160.f, 40.f, &mx, &mean), NVFX_OK);
  EXPECT_EQ(mx, 0.f);
  ASSERT_EQ(nvfx_scene_field_region(s.get(), NVFX_FIELD_HEAT, 500.f, 500.f, 600.f, 600.f, &mx, &mean), NVFX_OK);  // off the bus
  EXPECT_EQ(mx, 0.f);
  EXPECT_EQ(mean, 0.f);
  // the fire's flow rises (velocity in world pixels per second, y down)
  ASSERT_EQ(nvfx_scene_field_region(s.get(), NVFX_FIELD_V, 24.f, 40.f, 56.f, 80.f, &mx, &mean), NVFX_OK);
  EXPECT_LT(mean, 0.f);
  EXPECT_EQ(nvfx_scene_sample_grid(s.get(), NVFX_FIELD_HEAT, 0, 0, 1, 1, 2, 1, grid.data(), 1), NVFX_ERROR_ARGUMENT);  // row stride < nx
}

// The camera of the last frame computed, for a pointer over the picture (the viewer's field probe): overlapped, drawing
// frame f computes frame f + 1, and the camera moves on with the field reads.
TEST(SceneApi, CameraOfTheLastFrameComputed) {
  const Tiny fx;
  constexpr const char* kCamera = R"(
scene size 160 x 90, fps 30, length 2, ground 80
effect tiny = "tiny"
module fire = tiny, size 32, at (40, ground), start 1, seed 5
camera x 90 * t, y -7
)";
  for (const int overlap : {0, 1}) {
    ScenePtr s = make(fx, kCamera, 2, overlap);
    ASSERT_TRUE(s);
    float x = -1.f, y = -1.f;
    ASSERT_EQ(nvfx_scene_camera(s.get(), &x, &y), NVFX_OK);
    EXPECT_EQ(x, 0.f);
    EXPECT_EQ(y, -7.f);
    ASSERT_EQ(nvfx_scene_step_frames(s.get(), 10), NVFX_OK);
    ASSERT_EQ(nvfx_scene_camera(s.get(), &x, nullptr), NVFX_OK);
    EXPECT_NEAR(x, 30.f, 1e-4f);  // 3 world pixels a frame
    std::vector<std::uint8_t> rgba(160 * 90 * 4);
    ASSERT_EQ(nvfx_scene_render(s.get(), rgba.data(), 160 * 4), NVFX_OK);
    ASSERT_EQ(nvfx_scene_camera(s.get(), &x, &y), NVFX_OK);
    EXPECT_NEAR(x, overlap ? 33.f : 30.f, 1e-4f) << "overlap " << overlap;
    EXPECT_EQ(y, -7.f);
    EXPECT_EQ(nvfx_scene_camera(s.get(), nullptr, nullptr), NVFX_OK);
  }
  float x = 0.f;
  EXPECT_EQ(nvfx_scene_camera(nullptr, &x, nullptr), NVFX_ERROR_ARGUMENT);
}

// Errors: the script's line and column, the effect's statement for a file that cannot be read, wrong effects.
TEST(SceneApi, ErrorsSayWhere) {
  const Tiny fx;
  nvfx_scene_error err;
  ScenePtr s = make(fx, "scene size 160 x 90\neffect tiny = \"tiny\"\nmodule m = tiny, size 40", 1, 0, &err);
  EXPECT_FALSE(s);
  EXPECT_EQ(err.line, 3);
  EXPECT_EQ(err.column, 1);
  EXPECT_NE(std::string(err.message).find("t:3:1: the size of 'm' must be a multiple of its effect's grid"), std::string::npos) << err.message;
  s = make(fx, "effect tiny = \"tiny\"\nmodule m = tiny, size 32\nat 1:\n  wake m", 1, 0, &err);
  EXPECT_FALSE(s);
  EXPECT_EQ(err.line, 4);
  EXPECT_EQ(err.column, 3);
  EXPECT_EQ(nvfx_scene_check("module m = tiny, size 32,\n   at (1, 2", "c", &err), NVFX_ERROR_SCRIPT);
  EXPECT_EQ(err.line, 2);
  EXPECT_EQ(err.column, 7);
  EXPECT_EQ(nvfx_scene_check(kEverything, "c", &err), NVFX_OK);
  // an effect that is not a rollout effect
  nfx::Hyper g;  // as nvfx_alloc_test's grid effect
  g.arch = nfx::Arch::grid;
  g.size = 64;
  g.frames = 16;
  g.n_controls = 3;
  g.n_latent = 4;
  g.bases = 2;
  g.grid_t = 4;
  g.grid = 16;
  g.channels = 8;
  g.hidden = 16;
  nfx::Model fm = nfx::init_model(g, 1);
  std::ostringstream os;
  ASSERT_TRUE(nfx::save_model(os, fm));
  const std::string bytes = os.str();
  nvfx_effect* frame_fx = nullptr;
  ASSERT_EQ(nvfx_effect_load_memory(bytes.data(), bytes.size(), &frame_fx), NVFX_OK);
  const nvfx_scene_effect list[] = {{"tiny.nvfx", frame_fx}};
  nvfx_scene_desc d;
  nvfx_scene_desc_init(&d);
  d.script = "scene size 64 x 64\neffect tiny = \"tiny.nvfx\"";
  d.effects = list;
  d.n_effects = 1;
  nvfx_scene* out = nullptr;
  EXPECT_EQ(nvfx_scene_create(&d, &out, &err), NVFX_ERROR_FORMAT);
  EXPECT_EQ(err.line, 2);
  EXPECT_NE(std::string(err.message).find("not a rollout effect"), std::string::npos) << err.message;
  nvfx_effect_free(frame_fx);
  d.effects = nullptr;
  EXPECT_EQ(nvfx_scene_create(&d, &out, &err), NVFX_ERROR_ARGUMENT);  // n_effects without effects
  d.n_effects = 0;
  d.effects_dir = "/nonexistent";
  EXPECT_EQ(nvfx_scene_create(&d, &out, &err), NVFX_ERROR_IO);
  EXPECT_EQ(err.line, 2);
  EXPECT_NE(std::string(err.message).find("/nonexistent/tiny.nvfx"), std::string::npos) << err.message;
}

// A host that reads the effects itself (from a package) lists the ones a script names, then passes them in.
TEST(SceneApi, ListsTheEffectsAScriptNames) {
  using List = std::vector<std::pair<std::string, std::string>>;
  List got;
  const nvfx_scene_effect_fn fn = [](void* user, const char* name, const char* file) { static_cast<List*>(user)->emplace_back(name, file); };
  EXPECT_EQ(nvfx_scene_list_effects(kEverything, "t", fn, &got, nullptr), NVFX_OK);
  EXPECT_EQ(got, (List{{"tiny", "tiny"}, {"other", "other"}}));
  nvfx_scene_error err;
  EXPECT_EQ(nvfx_scene_list_effects("effect e = fire.nvfx", "t", fn, &got, &err), NVFX_ERROR_SCRIPT);
  EXPECT_EQ(err.line, 1);
  EXPECT_EQ(err.column, 12);
  EXPECT_EQ(nvfx_scene_list_effects(nullptr, "t", fn, &got, &err), NVFX_ERROR_ARGUMENT);
}

// An output size other than the script's: the picture resampled, opaque.
TEST(SceneApi, OutputSizeResamplesThePicture) {
  const Tiny fx;
  ScenePtr full = make(fx, kEverything, 1, 0), half = make(fx, kEverything, 1, 0, nullptr, 80, 45);
  ASSERT_TRUE(full && half);
  const nvfx_scene_info i = info_of(half.get());
  EXPECT_EQ(i.width, 80);
  EXPECT_EQ(i.height, 45);
  EXPECT_EQ(i.scene_width, 160);
  nvfx_scene_step_frames(full.get(), 20);
  nvfx_scene_step_frames(half.get(), 20);
  const std::vector<std::uint8_t> a = picture(full.get()), b = picture(half.get());
  ASSERT_EQ(b.size(), 80u * 45u * 3u);
  double ma = 0, mb = 0;
  for (const auto v : a) ma += v;
  for (const auto v : b) mb += v;
  ma /= static_cast<double>(a.size());
  mb /= static_cast<double>(b.size());
  EXPECT_GT(ma, 1.0);
  EXPECT_NEAR(mb, ma, 0.05 * ma + 0.5);
  std::vector<std::uint8_t> small(80 * 45 * 4);
  EXPECT_EQ(nvfx_scene_render(half.get(), small.data(), 79 * 4), NVFX_ERROR_ARGUMENT);  // stride too small
}

// --- with the study D effects: the fireball through the API --------------------------------------------------------

std::filesystem::path models_d() {
  if (const char* m = std::getenv("NEURALFX_MODELS_D")) return m;
  if (const char* d = std::getenv("NEURALVFX_DATA")) return std::filesystem::path(d) / "experiments" / "models" / "d";
  const char* home = std::getenv("HOME");
  return std::filesystem::path(home ? home : ".") / "nvfx-data" / "experiments" / "models" / "d";
}

// The fireball script played through the API, drawing only its eight keyframes (the other frames computed without
// pictures), gives the hand-written scene's frozen keyframes to the bit (results/experiments/v1_frozen.csv).
TEST(SceneApi, FireballKeyframesMatchTheFrozenOnes) {
  const std::filesystem::path models = models_d(), src = NEURALFX_SOURCE_DIR;
  for (const char* f : {"fire.nvfx", "smoke.nvfx", "explosion.nvfx"})
    if (!std::filesystem::exists(models / f)) GTEST_SKIP() << "the study D effects are not in " << models.string();
  std::map<std::string, std::string> frozen;  // frame_NNN -> sha256
  {
    std::ifstream in(src / "results/experiments/v1_frozen.csv");
    for (std::string line; std::getline(in, line);) {
      std::vector<std::string> cols;
      std::stringstream ls(line);
      for (std::string c; std::getline(ls, c, ',');) cols.push_back(c);
      if (cols.size() >= 5 && cols[0].ends_with("_keyframe")) frozen[cols[1]] = cols[4];
    }
  }
  ASSERT_EQ(frozen.size(), 8u);
  const std::string script = (src / "examples/scenes/fireball.nvfxs").string(), dir = models.string();
  nvfx_scene_desc d;
  nvfx_scene_desc_init(&d);
  d.script_path = script.c_str();
  d.effects_dir = dir.c_str();
  d.threads = 2;
  nvfx_scene* s = nullptr;
  nvfx_scene_error err;
  ASSERT_EQ(nvfx_scene_create(&d, &s, &err), NVFX_OK) << err.message;
  ScenePtr scene(s);
  const nvfx_scene_info i = info_of(s);
  ASSERT_EQ(i.width, 1280);
  std::vector<std::uint8_t> rgba(z(i.width) * z(i.height) * 4);
  const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "nvfx_scene_api_keyframe.png";
  int same = 0;
  for (const int f : {27, 38, 45, 60, 87, 126, 180, 246}) {  // the keyframes 0.9, 1.27, ..., 8.2 s at 30 fps
    ASSERT_EQ(nvfx_scene_seek(s, f / 30.0), NVFX_OK);
    ASSERT_EQ(nvfx_scene_frame(s), f);
    ASSERT_EQ(nvfx_scene_render(s, rgba.data(), z(i.width) * 4), NVFX_OK);
    nfx::Image img;
    img.allocate(i.width, i.height);
    std::copy(rgba.begin(), rgba.end(), img.rgba.begin());
    ASSERT_TRUE(nfx::write_png(tmp, img).has_value());
    std::ifstream in(tmp, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string name = std::format("frame_{:03d}", f);
    ASSERT_TRUE(frozen.contains(name)) << name;
    EXPECT_EQ(nfx::dcm::sha256_hex(ss.str()), frozen[name]) << name;
    same += nfx::dcm::sha256_hex(ss.str()) == frozen[name];
  }
  std::filesystem::remove(tmp);
  EXPECT_EQ(same, 8);
  nvfx_scene_fields hot{};
  ASSERT_EQ(nvfx_scene_sample(s, 1010.f, 560.f, &hot), NVFX_OK);  // the wreck burns at 8.2 s
  EXPECT_GT(hot.heat, 0.05f);
}

}  // namespace
