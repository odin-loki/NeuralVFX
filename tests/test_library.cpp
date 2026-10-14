// The effect library (docs/LIBRARY.md): its catalogue parses and every entry checks, errors say where, stages are
// scene scripts that compile, and instances spawn, play, end and come back as a game drives them (with the tests'
// untrained rollout effects, so no data is needed). And the composer's procedural elements the library added: glows
// (lightning, rings, discs), lamps, the shader's hue, a module's width that follows an input, room for shock fronts,
// sparks in bursts.
#include "compose_scene.hpp"
#include "library.hpp"
#include "script.hpp"

#include <neuralfx/nvfx_library.h>
#include <neuralfx/rollout.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace sc = nfx::compose::script;
namespace lib = nfx::library;
using nfx::compose::Frame;
using nfx::compose::Image4;
using nfx::compose::Particles;

const std::string kDir = std::string(NEURALFX_SOURCE_DIR) + "/library";

struct EffectDeleter {
  void operator()(nvfx_effect* e) const { nvfx_effect_free(e); }
};
using EffectPtr = std::unique_ptr<nvfx_effect, EffectDeleter>;
struct LibraryDeleter {
  void operator()(nvfx_library* l) const { nvfx_library_free(l); }
};
using LibraryPtr = std::unique_ptr<nvfx_library, LibraryDeleter>;
struct StageDeleter {
  void operator()(nvfx_stage* s) const { nvfx_stage_free(s); }
};
using StagePtr = std::unique_ptr<nvfx_stage, StageDeleter>;

EffectPtr tiny_nvfx(std::uint64_t seed) {
  const nfx::rt::RolloutEffect t = nfx::compose::testing::tiny_effect(seed);
  std::ostringstream os;
  EXPECT_TRUE(nfx::rollout::save_model(os, t.m));
  const std::string bytes = os.str();
  nvfx_effect* e = nullptr;
  EXPECT_EQ(nvfx_effect_load_memory(bytes.data(), bytes.size(), &e), NVFX_OK);
  return EffectPtr(e);
}

std::string parse_error(const std::string& text) {
  try {
    const std::vector<std::pair<std::string, std::string>> files = {{"t.nvfxl", text}};
    (void)lib::parse(files);
  } catch (const sc::Error& e) {
    return e.what();
  }
  return "no error";
}

// A small catalogue over the tiny effect: a looping entry with a module, a once-rule and a parameter; a one-shot; a
// procedural one.
constexpr const char* kTinyCatalogue = R"(
effect tiny = "tiny.nvfx"
look hot = shader, heat_scale 1, emission 2

entry blob, category fire, size 64 x 64, anchor base, loop, show 0.5, about "a tiny looping effect"
param intensity = 0.5, range (0, 1), about "its control"
module m = tiny, size 32, width 48 * scale, at (X, Y), look hot, start 1, seed 7, controls (0.5, 0.5, 0.5),
           intensity intensity
at 0.2 as poke:
  scorch at (X, Y), radius 10 * scale
emit embers, on m, tries 1, chance 0.5, spread 10

entry pop, category explosions, size 64 x 64, anchor centre, once 0.5, about "a tiny one-shot"
module m = tiny, size 32, width 40 * scale, at (X, Y + 20 * scale), start 0, seed 3
at 0: shock at (X, Y), speed 300

entry zap, category sparks, size 80 x 40, anchor left, loop, about "a procedural arc"
emit arc, along (X - 30, Y - 10) to (X + 30, Y - 10), rate 10, seed 2
field shine = lamp, at (X, Y - 10), radius 20, strength 0.5
)";

struct TinyLibrary {
  EffectPtr tiny = tiny_nvfx(3);
  LibraryPtr library;
  TinyLibrary() {
    const nvfx_library_file file{"tiny.nvfxl", kTinyCatalogue};
    nvfx_library_desc d;
    nvfx_library_desc_init(&d);
    d.files = &file;
    d.n_files = 1;
    nvfx_library* l = nullptr;
    nvfx_scene_error err;
    EXPECT_EQ(nvfx_library_open(&d, &l, &err), NVFX_OK) << err.message;
    library.reset(l);
  }
  StagePtr stage(std::vector<nvfx_stage_room> rooms, int threads = 2, int overlap = -1) {
    const nvfx_scene_effect fx{"tiny.nvfx", tiny.get()};
    nvfx_stage_desc d;
    nvfx_stage_desc_init(&d);
    d.width = 160;
    d.height = 96;
    d.ground = 80;
    d.rooms = rooms.data();
    d.n_rooms = static_cast<int>(rooms.size());
    d.effects = &fx;
    d.n_effects = 1;
    d.threads = threads;
    d.overlap = overlap;
    nvfx_stage* s = nullptr;
    nvfx_scene_error err;
    EXPECT_EQ(nvfx_stage_create(library.get(), &d, &s, &err), NVFX_OK) << err.message;
    return StagePtr(s);
  }
};

nvfx_scene_module_info module(nvfx_stage* s, const char* name) {
  nvfx_scene_module_info m{};
  EXPECT_EQ(nvfx_scene_module_get_info(nvfx_stage_scene(s), name, &m), NVFX_OK) << name;
  return m;
}

int rule_count(nvfx_stage* s, const char* name) {
  int n = -1;
  EXPECT_EQ(nvfx_scene_rule_state(nvfx_stage_scene(s), name, &n, nullptr), NVFX_OK) << name;
  return n;
}

}  // namespace

TEST(Library, TheCatalogueParsesAndEveryEntryChecks) {
  const lib::Catalogue c = lib::load_dir(kDir);
  EXPECT_GE(c.entries.size(), 30u);
  const std::vector<std::string> cats = c.categories();
  for (const char* want : {"fire", "smoke", "explosions", "steam", "magic", "ambient", "sparks", "light", "dust", "weather"})
    EXPECT_NE(std::ranges::find(cats, want), cats.end()) << want;
  for (const lib::Entry& e : c.entries) {
    EXPECT_FALSE(e.about.empty()) << e.name;
    EXPECT_GT(e.width, 0.f) << e.name;
    EXPECT_TRUE(!e.effects.empty() || !e.elements.empty()) << e.name << " plays nothing";
    for (const std::string& f : e.effects) EXPECT_FALSE(c.effect_file(f).empty()) << e.name;
  }
  // a stage with two of every entry is one script that compiles, and prints back as it parses
  lib::StageSpec spec;
  for (std::size_t i = 0; i < c.entries.size(); ++i) spec.rooms.push_back({static_cast<int>(i), 2});
  const lib::Stage st = lib::build_stage(c, spec);
  EXPECT_NO_THROW(sc::validate(st.script));
  const std::string text = st.text();
  EXPECT_EQ(sc::print(sc::parse(text, "stage")), text);
  EXPECT_EQ(st.slots.size(), 2 * c.entries.size());
}

TEST(Library, ErrorsSayFileLineAndColumn) {
  const std::string head = "effect fire = \"fire.nvfx\"\n";
  const std::string e = "entry torch, category fire, size 90 x 190, loop, about \"x\"\n";
  EXPECT_EQ(parse_error(head + e + "field f = lamp, at (X, Y), strenght 1\n"), "t.nvfxl:3:28: 'strenght' is not a property of field (did you mean 'strength'?)");
  EXPECT_EQ(parse_error(head + "entry torch, size 90 x 190, loop, about \"x\"\n"), "t.nvfxl:2:1: entry 'torch' needs a category (fire, smoke, explosions, steam, magic, ambient, sparks, light, ...)");
  EXPECT_EQ(parse_error(head + "entry torch, category fire, size 90 x 190, about \"x\"\n"),
            "t.nvfxl:2:1: entry 'torch' plays either until stopped ('loop') or once for a time ('once 2.5 s')");
  EXPECT_EQ(parse_error(head + e + "scene size 64 x 64\n"), "t.nvfxl:3:1: 'scene' is the stage's (its picture, clock and bus): an entry cannot set it");
  EXPECT_EQ(parse_error(head + e + "module flame = fire, size 64\n"),
            "t.nvfxl:3:1: module 'flame' stands relative to the anchor: give it 'at (X, Y)' (or an offset from it)");
  EXPECT_EQ(parse_error(head + e + "module flame = fire, tiles 2 x 1, band 4, size 64, at (X, Y)\n"),
            "t.nvfxl:3:1: tiles stay where they are declared, so they cannot be spawned: use single modules side by side");
  EXPECT_EQ(parse_error(head + e + "let X = 3\n"), "t.nvfxl:3:5: 'X' is the instance's own (X, Y, scale, variant, alive) or starts with '_': choose another name");
  EXPECT_EQ(parse_error(head + e + "when heat(X, Y) > shock(1): scorch at (X, Y)\n"), "t.nvfxl:3:19: shock fronts are numbered in the whole stage: an entry cannot read them");
  EXPECT_EQ(parse_error(head + e + "at 1, at most 2: scorch at (X, Y)\n"), "t.nvfxl:3:15: a rule of an entry fires once per spawn, or every time with 'repeat'");
  EXPECT_EQ(parse_error(head + e + "param fuel = 2, range (0, 1), about \"x\"\n"), "t.nvfxl:3:14: the default of 'fuel' is outside its range (0 to 1)");
  EXPECT_EQ(parse_error(e + "module flame = fire, size 64, at (X, Y)\n"), "t.nvfxl:2:1: unknown effect 'fire' (declare it before the entries: effect fire = \"FILE\")");
  EXPECT_EQ(parse_error(e + head), "t.nvfxl:2:1: declare effects before the first entry of a file: they are shared by every entry");
  EXPECT_EQ(parse_error(head + e + "emit sparks, along (X, Y) to (X, Y), count speed\n"), "t.nvfxl:3:44: unknown name 'speed'");
  EXPECT_EQ(parse_error("module m = fire, size 32\n"), "t.nvfxl:1:1: 'module' belongs in an entry: start one with 'entry NAME, category ..., size W x H, ...' (only effects and looks come before the first)");
  // a scene cannot hold an entry
  EXPECT_THROW(sc::validate(sc::parse(e, "s")), sc::Error);
}

TEST(Library, InstancesSpawnPlayEndAndComeBack) {
  TinyLibrary T;
  StagePtr s = T.stage({{"blob", 2, 0}, {"pop", 1, 1}});
  ASSERT_TRUE(s);
  nvfx_scene* scene = nvfx_stage_scene(s.get());
  // the modules are asleep until a spawn, warm (started when the stage was made)
  EXPECT_FALSE(module(s.get(), "s0_m").active);
  nvfx_fx a = 0;
  ASSERT_EQ(nvfx_stage_spawn(s.get(), "blob", 50.f, 80.f, 1.5f, 0.3, &a), NVFX_OK);
  nvfx_scene_step(scene, 0.25, nullptr);
  EXPECT_FALSE(module(s.get(), "s0_m").active) << "not before its time";
  nvfx_scene_step(scene, 0.1, nullptr);
  const nvfx_scene_module_info m = module(s.get(), "s0_m");
  EXPECT_TRUE(m.active);
  EXPECT_NEAR(m.started, 0.3f, 1e-5f);
  EXPECT_FLOAT_EQ(m.width, 72.f) << "48 world pixels at scale 1.5";
  EXPECT_FLOAT_EQ(m.x, 50.f);
  EXPECT_FLOAT_EQ(m.y, 80.f);
  ASSERT_EQ(nvfx_stage_set_param(s.get(), a, "intensity", 0.9f), NVFX_OK);
  ASSERT_EQ(nvfx_stage_move(s.get(), a, 70.f, 80.f), NVFX_OK);
  nvfx_scene_step(scene, 0.3, nullptr);
  EXPECT_NEAR(module(s.get(), "s0_m").controls[0], 0.9f, 1e-6f);
  EXPECT_FLOAT_EQ(module(s.get(), "s0_m").x, 70.f);
  EXPECT_EQ(rule_count(s.get(), "s0_poke"), 1) << "a rule fires once a spawn: at 0.2 s of its life";
  EXPECT_EQ(rule_count(s.get(), "s1_poke"), 0) << "the other room's never spawned";
  // stopped: its module stops; the room is free and a new spawn starts it afresh, its rule firing again
  ASSERT_EQ(nvfx_stage_stop(s.get(), a, -1.0), NVFX_OK);
  nvfx_scene_step(scene, 0.1, nullptr);
  EXPECT_FALSE(module(s.get(), "s0_m").active);
  nvfx_stage_fx_info info{};
  ASSERT_EQ(nvfx_stage_get_fx(s.get(), a, &info), NVFX_OK);
  EXPECT_EQ(info.state, NVFX_FX_ENDED);
  EXPECT_EQ(nvfx_stage_free_rooms(s.get(), "blob"), 2);
  nvfx_fx b = 0;
  ASSERT_EQ(nvfx_stage_spawn(s.get(), "blob", 30.f, 80.f, 1.f, -1.0, &b), NVFX_OK);
  EXPECT_NE(a, b);
  EXPECT_EQ(nvfx_stage_get_fx(s.get(), a, &info), NVFX_ERROR_ARGUMENT) << "a's handle is stale: its room plays b";
  nvfx_scene_step(scene, 0.5, nullptr);
  EXPECT_TRUE(module(s.get(), "s0_m").active);
  float v = 0.f;
  ASSERT_EQ(nvfx_stage_get_param(s.get(), b, "intensity", &v), NVFX_OK);
  EXPECT_FLOAT_EQ(v, 0.5f) << "a spawn's parameters take their defaults";
  EXPECT_EQ(rule_count(s.get(), "s0_poke"), 2);
  // a one-shot ends by itself; with steal, a spawn when it plays replaces it
  nvfx_fx p = 0, q = 0;
  ASSERT_EQ(nvfx_stage_spawn(s.get(), "pop", 80.f, 40.f, 1.f, -1.0, &p), NVFX_OK);
  nvfx_scene_step(scene, 0.2, nullptr);
  EXPECT_TRUE(module(s.get(), "s2_m").active);
  ASSERT_EQ(nvfx_stage_spawn(s.get(), "pop", 90.f, 40.f, 1.f, -1.0, &q), NVFX_OK);
  EXPECT_EQ(nvfx_stage_get_fx(s.get(), p, &info), NVFX_ERROR_ARGUMENT);
  nvfx_scene_step(scene, 0.2, nullptr);
  ASSERT_EQ(nvfx_stage_get_fx(s.get(), q, &info), NVFX_OK);
  EXPECT_EQ(info.state, NVFX_FX_PLAYING);
  nvfx_scene_step(scene, 0.6, nullptr);
  ASSERT_EQ(nvfx_stage_get_fx(s.get(), q, &info), NVFX_OK);
  EXPECT_EQ(info.state, NVFX_FX_ENDED);
  EXPECT_FALSE(module(s.get(), "s2_m").active);
  // cleared: nothing plays, handles are stale
  ASSERT_EQ(nvfx_stage_clear(s.get()), NVFX_OK);
  EXPECT_EQ(nvfx_scene_time(scene), 0.0);
  EXPECT_FALSE(module(s.get(), "s0_m").active);
  EXPECT_EQ(nvfx_stage_get_fx(s.get(), b, &info), NVFX_ERROR_ARGUMENT);
}

TEST(Library, ErrorsFoundWithTheEffectsSayWhereInTheCatalogue) {
  EffectPtr tiny = tiny_nvfx(3);
  const nvfx_library_file file{"bad.nvfxl", "effect tiny = \"tiny.nvfx\"\nentry blob, category fire, size 64 x 64, loop, about \"x\"\n"
                                            "module m = tiny, size 32, at (X, Y), start 0, intensty 0.4\n"};
  nvfx_library_desc d;
  nvfx_library_desc_init(&d);
  d.files = &file;
  d.n_files = 1;
  nvfx_library* l = nullptr;
  nvfx_scene_error err;
  ASSERT_EQ(nvfx_library_open(&d, &l, &err), NVFX_OK) << "a control's name is checked against the effect, which is not loaded yet";
  LibraryPtr library(l);
  const nvfx_scene_effect fx{"tiny.nvfx", tiny.get()};
  const nvfx_stage_room room{"blob", 2, 0};
  nvfx_stage_desc sd;
  nvfx_stage_desc_init(&sd);
  sd.rooms = &room;
  sd.n_rooms = 1;
  sd.effects = &fx;
  sd.n_effects = 1;
  nvfx_stage* s = nullptr;
  EXPECT_EQ(nvfx_stage_create(l, &sd, &s, &err), NVFX_ERROR_SCRIPT);
  EXPECT_EQ(s, nullptr);
  EXPECT_STREQ(err.message, "bad.nvfxl:3:47: 'intensty' is not a property of a module nor a control of effect 'tiny' (its controls: intensity, wind, "
                            "turbulence) (did you mean 'intensity'?)");
  EXPECT_EQ(err.line, 3);
}

TEST(Library, NoRoomWithoutSteal) {
  TinyLibrary T;
  StagePtr s = T.stage({{"blob", 1, 0}, {"zap", 1, 0}});
  nvfx_fx a = 0, b = 0;
  ASSERT_EQ(nvfx_stage_spawn(s.get(), "blob", 50.f, 80.f, 1.f, -1.0, &a), NVFX_OK);
  EXPECT_EQ(nvfx_stage_spawn(s.get(), "blob", 50.f, 80.f, 1.f, -1.0, &b), NVFX_ERROR_MEMORY);
  EXPECT_EQ(b, 0u);
  EXPECT_EQ(nvfx_stage_spawn(s.get(), "pop", 50.f, 80.f, 1.f, -1.0, &b), NVFX_ERROR_ARGUMENT) << "no room for it on this stage";
  EXPECT_EQ(nvfx_stage_free_rooms(s.get(), "blob"), 0);
  // a spawn not yet due is cancelled by a stop
  ASSERT_EQ(nvfx_stage_spawn(s.get(), "zap", 80.f, 80.f, 1.f, 5.0, &b), NVFX_OK);
  nvfx_stage_fx_info info{};
  ASSERT_EQ(nvfx_stage_get_fx(s.get(), b, &info), NVFX_OK);
  EXPECT_EQ(info.state, NVFX_FX_WAITING);
  ASSERT_EQ(nvfx_stage_stop(s.get(), b, -1.0), NVFX_OK);
  nvfx_scene_step(nvfx_stage_scene(s.get()), 5.5, nullptr);
  int fired = -1;
  nvfx_scene_rule_state(nvfx_stage_scene(s.get()), "s1__go", &fired, nullptr);
  EXPECT_EQ(fired, 0);
}

TEST(Library, TheSameCallsGiveTheSameFramesOnAnyThreads) {
  TinyLibrary T;
  std::vector<std::vector<std::uint8_t>> frames;
  for (const int threads : {1, 3}) {
    for (const int overlap : {0, 1}) {
      StagePtr s = T.stage({{"blob", 2, 0}, {"pop", 1, 1}, {"zap", 1, 0}}, threads, overlap);
      nvfx_scene* scene = nvfx_stage_scene(s.get());
      std::vector<std::uint8_t> rgba(160 * 96 * 4);
      nvfx_scene_render(scene, rgba.data(), 160 * 4);
      nvfx_fx a = 0, b = 0, c = 0;
      nvfx_stage_spawn(s.get(), "blob", 40.f, 80.f, 1.f, 0.2, &a);
      nvfx_stage_spawn(s.get(), "pop", 100.f, 40.f, 1.2f, 0.4, &b);
      nvfx_stage_spawn(s.get(), "zap", 120.f, 80.f, 1.f, 0.3, &c);
      for (int f = 1; f <= 30; ++f) {
        if (f == 18) nvfx_stage_set_param(s.get(), a, "intensity", 1.f);
        nvfx_scene_step_frames(scene, 1);
        nvfx_scene_render(scene, rgba.data(), 160 * 4);
      }
      frames.push_back(rgba);
    }
  }
  for (std::size_t i = 1; i < frames.size(); ++i) EXPECT_EQ(frames[i], frames[0]) << "run " << i;
}

TEST(Library, AnchorsAreTheBaseTheCentreOrTheLeftEdge) {
  TinyLibrary t;
  const auto anchor = [&](const char* name) {
    nvfx_library_effect e{};
    EXPECT_EQ(nvfx_library_get_effect(t.library.get(), nvfx_library_find(t.library.get(), name), &e), NVFX_OK) << name;
    return e.anchor;
  };
  EXPECT_EQ(anchor("blob"), NVFX_ANCHOR_BASE);
  EXPECT_EQ(anchor("pop"), NVFX_ANCHOR_CENTRE);
  EXPECT_EQ(anchor("zap"), NVFX_ANCHOR_LEFT);  // a jet's nozzle: the box lies to the right of the spawn point
  EXPECT_EQ(parse_error("entry torch, category fire, size 90 x 190, anchor top, loop, about \"x\"\n"),
            "t.nvfxl:1:51: an anchor is 'base' (X, Y at the bottom centre of its box), 'centre' or 'left' (the middle of its left edge), not 'top'");
}

TEST(Library, AnEntryAloneIsItsOwnScene) {
  const std::vector<std::pair<std::string, std::string>> files = {{"tiny.nvfxl", kTinyCatalogue}};
  const lib::Catalogue c = lib::parse(files);
  lib::StageSpec spec;
  const lib::Stage one = lib::single(c, c.index("blob"), spec);
  const std::string text = one.text();
  // its names are as written, its instance's own inputs and rules start with '_'
  EXPECT_NE(text.find("module m = tiny"), std::string::npos) << text;
  EXPECT_NE(text.find("input _X = 0"), std::string::npos) << text;
  EXPECT_NE(text.find("when t >= _at and (_go < _at or _go == infinity) as _go, repeat:"), std::string::npos) << text;
  EXPECT_NO_THROW(sc::validate(one.script));
}

// --- the composer's procedural elements ---------------------------------------------------------------------------

TEST(Glows, PiecesOfALineJoinWithoutOverlapping) {
  // one glow from (10, 20.5) to (50, 20.5), against two pieces that meet at (30, 20.5): the same picture
  const auto draw = [](int pieces) {
    Particles p(8);
    if (pieces == 1) {
      p.spawn_glow(10.f, 20.5f, 50.f, 20.5f, 3.f, {1.f, 0.5f, 0.25f}, 2.f, 1.f, true, true);
    } else {
      p.spawn_glow(10.f, 20.5f, 30.f, 20.5f, 3.f, {1.f, 0.5f, 0.25f}, 2.f, 1.f, true, false);
      p.spawn_glow(30.f, 20.5f, 50.f, 20.5f, 3.f, {1.f, 0.5f, 0.25f}, 2.f, 1.f, false, true);
    }
    Image4 img;
    img.allocate(64, 40);
    p.draw(img, 0.f, 0.f);
    return img.px;
  };
  const auto one = draw(1), two = draw(2);
  float most = 0.f;
  for (std::size_t i = 0; i < one.size(); ++i) {
    EXPECT_NEAR(one[i], two[i], 1e-5f) << i;
    most = std::max(most, one[i]);
  }
  EXPECT_NEAR(most, 2.f, 0.05f) << "brightness times colour at the core";
  // a disc: both caps, no length
  Particles d(2);
  d.spawn_glow(20.f, 20.f, 20.f, 20.f, 6.f, {0.f, 1.f, 0.f}, 1.f, 1.f, true, true);
  Image4 img;
  img.allocate(40, 40);
  d.draw(img, 0.f, 0.f);
  EXPECT_GT(img.row(19)[19 * 4 + 1], 0.9f);
  EXPECT_EQ(img.row(19)[19 * 4 + 0], 0.f);
  EXPECT_EQ(img.row(19)[27 * 4 + 1], 0.f) << "nothing beyond its radius";
  // glows stay put for their life, then go
  d.update(0.02f, nullptr, 0.f, 1000.f);
  EXPECT_EQ(d.alive(), 1);
  d.update(1.f, nullptr, 0.f, 1000.f);
  EXPECT_EQ(d.alive(), 0);
}

TEST(Glows, EmittersMakeBoltsRingsAndMotesAndKeepTheScenesRandomNumbers) {
  const auto run = [](const std::string& extra) {
    const std::string text = "scene size 160 x 90, fps 30, length 1 s, ground 80\nparticles capacity 4096\n" + extra;
    sc::Scene scene(sc::parse(text, "t"), [](const std::string&) { return nfx::compose::testing::tiny_effect(3); }, {});
    std::vector<std::uint8_t> rgb(160 * 90 * 3);
    for (int f = 0; f < 10; ++f) scene.render(f, rgb);
    return std::pair{scene.particles().alive(), scene.particles().rng};
  };
  const auto [n0, rng0] = run("emit sparks, along (20, 70) to (20, 70), count 3\n");
  const auto [n1, rng1] = run("emit sparks, along (20, 70) to (20, 70), count 3\n"
                              "emit arc, along (40, 10) to (60, 70), rate 10, branches 2, seed 4\n"
                              "emit ring, at (100, 60), radius 30, squash 0.3\n"
                              "emit glow, at (120, 30), radius 2, count 5, spread 20\n");
  EXPECT_EQ(rng0, rng1) << "glows draw their shapes from a hash, not from the scene's generator";
  EXPECT_GT(n1, n0 + 50);
  // a bolt keeps its shape between strikes: the same pieces at frames of the same strike
  const auto pieces = [](int frames) {
    sc::Scene scene(sc::parse("scene size 100 x 100, fps 30, length 1 s\nemit arc, along (10, 10) to (90, 90), rate 2, seed 9, branches 0\n", "t"),
                    [](const std::string&) { return nfx::compose::testing::tiny_effect(3); }, {});
    for (int f = 0; f < frames; ++f) scene.advance(f);
    std::vector<float> xy;
    scene.particles().each([&](float x, float y, float&, float&) {
      xy.push_back(x);
      xy.push_back(y);
    });
    return xy;
  };
  EXPECT_EQ(pieces(3), pieces(5)) << "frames 2 and 4: the same strike (rate 2: a new one every 15 frames)";
  EXPECT_NE(pieces(3), pieces(17));
}

TEST(Lamps, ALampLightsAroundItInItsColour) {
  nfx::compose::FieldBus bus(0.f, 0.f, 40, 30, 8.f, 1);
  nfx::compose::Light light(bus);
  nfx::compose::Pool pool(1);
  light.update(bus, 0.14f, {0.f, 0.f, 0.f}, pool);
  EXPECT_EQ(light.at(160.f, 120.f)[0], 0.f);
  const nfx::compose::Lamp lamp{160.f, 120.f, 30.f, 1.f, {0.2f, 0.4f, 1.f}};
  light.update(bus, 0.14f, {0.f, 0.f, 0.f}, pool, std::span(&lamp, 1));
  const auto near = light.at(160.f, 120.f), far = light.at(20.f, 20.f);
  EXPECT_GT(near[2], 0.05f);
  EXPECT_NEAR(near[0] / near[2], 0.2f, 1e-3f);
  EXPECT_NEAR(near[1] / near[2], 0.4f, 1e-3f);
  EXPECT_LT(far[2], 0.1f * near[2]);
  // and in a scene: a lamp field, with no module to act on
  sc::Scene scene(sc::parse("scene size 160 x 90, fps 30, length 1 s\nfield l = lamp, at (80, 40), radius 20, strength 2, colour (1, 0, 0)\n", "t"),
                  [](const std::string&) { return nfx::compose::testing::tiny_effect(3); }, {});
  std::vector<std::uint8_t> rgb(160 * 90 * 3);
  scene.render(0, rgb);
  EXPECT_EQ(scene.bus().at(80.f, 40.f).heat, 0.f) << "a lamp is not heat";
}

TEST(Shader, HueTurnsTheEmittedColoursAroundTheGreyAxis) {
  const auto m0 = nfx::compose::hue_matrix(0.f), m120 = nfx::compose::hue_matrix(120.f), m240 = nfx::compose::hue_matrix(240.f);
  const std::array<float, 9> id{1, 0, 0, 0, 1, 0, 0, 0, 1}, red_to_green{0, 0, 1, 1, 0, 0, 0, 1, 0}, red_to_blue{0, 1, 0, 0, 0, 1, 1, 0, 0};
  EXPECT_EQ(m0, id);
  EXPECT_EQ(m120, red_to_green);
  EXPECT_EQ(m240, red_to_blue);
  // a module shaded with hue 120: its emission's red is now green, the soot unchanged
  const auto shade = [](float hue) {
    nfx::rt::RolloutEffect e = nfx::compose::testing::tiny_effect(3);
    nfx::compose::Module m("m", e, 32, {0.f, 0.f, 1.f}, nfx::compose::best_isa());
    m.look = nfx::compose::Look::shader;
    m.spec.soot_density = 0.f;
    m.spec.hue = hue;
    m.start(0, 1);
    m.shade(nullptr);
    double r = 0, g = 0;
    for (std::size_t i = 0; i < m.image().px.size(); i += 4) {
      r += m.image().px[i];
      g += m.image().px[i + 1];
    }
    return std::pair{r, g};
  };
  const auto [r0, g0] = shade(0.f);
  const auto [r1, g1] = shade(120.f);
  EXPECT_GT(r0, 1.2 * g0) << "fire colours";
  EXPECT_NEAR(g1, r0, 1e-3 * r0) << "red drawn as green";
  EXPECT_NEAR(r1 + g1, r0 + g0, 0.6 * (r0 + g0));
}

TEST(Script, ASingleModulesWidthFollowsAnInput) {
  const std::string text = "scene size 160 x 90, fps 30, length 1 s, ground 80\ninput k = 1\neffect e = \"e\"\n"
                           "module m = e, size 32, width 40 * k, at (80, ground), start 0\n";
  sc::Scene scene(sc::parse(text, "t"), [](const std::string&) { return nfx::compose::testing::tiny_effect(3); }, {});
  scene.advance(0);
  EXPECT_FLOAT_EQ(scene.module_info(0).width, 40.f);
  scene.set_input(0, 2.f);
  scene.advance(1);
  const auto m = scene.module_info(0);
  EXPECT_FLOAT_EQ(m.width, 80.f);
  EXPECT_FLOAT_EQ(m.x, 80.f);
  EXPECT_FLOAT_EQ(scene.module("m")->at.scale, 2.5f);
  EXPECT_FLOAT_EQ(scene.module("m")->at.x, 40.f) << "still standing on its centre";
  // tiles keep a constant width
  try {
    sc::validate(sc::parse("input k = 1\neffect e = \"e\"\nmodule m = e, tiles 2 x 1, band 4, size 32, width 40 * k\n", "t"));
    ADD_FAILURE();
  } catch (const sc::Error& e) {
    EXPECT_EQ(std::string(e.what()), "t:3:54: the width of tiles must be a constant: it cannot depend on time, rules, modules, inputs or rand");
  }
}

TEST(Script, RepeatingShocksAndScorchMarksMakeRoom) {
  const std::string text = "scene size 160 x 90, fps 30, length 10 s\n"
                           "when 1 as boom, repeat:\n  shock at (80, 40), decay 0.01\n  scorch at (80, 80), radius 5\n";
  sc::Scene scene(sc::parse(text, "t"), [](const std::string&) { return nfx::compose::testing::tiny_effect(3); }, {});
  std::vector<std::uint8_t> rgb(160 * 90 * 3);
  for (int f = 0; f < 100; ++f) scene.render(f, rgb);  // 100 shocks: more than the 32 a repeating rule has room for
  EXPECT_EQ(scene.rule_count("boom"), 100);
}

TEST(Script, BurstSparksFlyWithinTheirCone) {
  const std::string text = "scene size 160 x 90, fps 30, length 1 s, ground 1000\nparticles capacity 1000\n"
                           "at 0: burst at (80, 40), embers 0, sparks 200, direction (1, 0), cone 30, speed 300\n";
  sc::Scene scene(sc::parse(text, "t"), [](const std::string&) { return nfx::compose::testing::tiny_effect(3); }, {});
  scene.advance(0);
  int n = 0, outside = 0;
  scene.particles().each([&](float x, float y, float&, float&) {
    ++n;
    const float a = std::atan2(y - 40.f, x - 80.f) * 180.f / 3.1415927f;
    outside += std::fabs(a) > 15.f + 6.f ? 1 : 0;  // a frame of gravity bends them a little
  });
  EXPECT_EQ(n, 200);
  EXPECT_EQ(outside, 0);
}

TEST(Script, NamedRulesMayHaveNoActions) {
  const sc::Script s = sc::parse("when t > 0.5 as half:\nlet h = half\n", "t");
  EXPECT_NO_THROW(sc::validate(s));
  try {
    (void)sc::parse("when t > 0.5:\n", "t");
    ADD_FAILURE();
  } catch (const sc::Error& e) {
    EXPECT_EQ(std::string(e.what()), "t:1:1: this rule has no actions (indent them under it)");
  }
}
