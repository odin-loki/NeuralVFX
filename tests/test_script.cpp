// Scene scripts (src/compose/script.hpp, docs/COMPOSE.md §4) and the field effects of §5: the parser's errors point at
// the right line and column, scripts survive a round trip through the printer, a scripted scene matches the same scene
// written in C++ to the bit, rules fire as often as they say, and each field effect does what it promises.
#include "compose_scene.hpp"
#include "script.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>

namespace {

using namespace nfx;
using namespace nfx::compose;
namespace sc = nfx::compose::script;
using nfx::compose::testing::tiny_effect;

std::size_t z(int v) { return static_cast<std::size_t>(v); }

sc::EffectLoader tiny_loader() {
  return [](const std::string& file) { return tiny_effect(file == "other" ? 4 : 3); };
}

std::string read(const std::filesystem::path& p) {
  std::ifstream f(p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// --- the parser and the checks -----------------------------------------------------------------------------------------

struct Bad {
  const char* text;
  int line, col;
  const char* says;
};

// Problems found by the parser (syntax and properties).
TEST(Script, ParserErrorsHaveLineAndColumn) {
  const Bad cases[] = {
      {"modul x = fire", 1, 1, "unknown statement 'modul'"},
      {"scene size 1280 x 720\nfield f = wind, velocity (1, 0), on a, strenght 3", 2, 40, "'strenght' is not a property of field"},
      {"let a = 1 +\n", 1, 12, "found the end of the line"},
      {"let a = 1 2", 1, 11, "unexpected '2' at the end of the let statement"},
      {"at 1.0\n  stop a", 1, 7, "expected ':' to end the rule"},
      {"  stop a", 1, 3, "indented, but no rule"},
      {"let a = 1 < 2 < 3", 1, 15, "comparisons do not chain"},
      {"let a = 3 $ 4", 1, 11, "unexpected character '$'"},
      {"let a = 3 \u00a7 4", 1, 11, "unexpected character '\u00a7'"},
      {"scene size 1280 x", 1, 18, "expected a value"},
      {"scene size 1280 720", 1, 17, "expected 'x' between the two sizes"},
      {"field f = whirl, at (1, 2)", 1, 11, "'whirl' is not a kind of field"},
      {"at 1:\n  explode m", 2, 3, "expected an action"},
      {"push m, gain 1", 1, 1, "'push' is an action"},
      {"at 1:\n", 1, 1, "this rule has no actions"},
      {"let a = (1 +\n 2", 1, 9, "this '(' is not closed"},
      {"module m = fire, size 64, size 32", 1, 27, "'size' is given twice"},
      {"effect e = fire.nvfx", 1, 12, "expected the effect's file in quotes"},
      {"emit rain, in (0, 0)", 1, 6, "'rain' cannot be emitted"},
      {"when ember lands where x > 1, at most:\n  stop m", 1, 38, "expected a value"},
  };
  for (const Bad& b : cases) {
    try {
      (void)sc::parse(b.text, "t");
      ADD_FAILURE() << "no error for: " << b.text;
    } catch (const sc::Error& e) {
      EXPECT_EQ(e.pos.line, b.line) << b.text << "\n  " << e.what();
      EXPECT_EQ(e.pos.col, b.col) << b.text << "\n  " << e.what();
      EXPECT_NE(e.message.find(b.says), std::string::npos) << b.text << "\n  " << e.what();
      EXPECT_EQ(std::string(e.what()).rfind(std::format("t:{}:{}: ", b.line, b.col), 0), 0u) << e.what();
    }
  }
}

// Problems found by validate(): names, contexts and constants, before anything is loaded.
TEST(Script, ValidationErrorsHaveLineAndColumn) {
  const Bad cases[] = {
      {"let x = 1", 1, 5, "'x' is a word of the script language"},
      {"effect e = \"e\"\nmodule weight = e, size 32", 2, 8, "'weight' is a word of the script language"},
      {"let a = foo(1)", 1, 9, "unknown function 'foo'"},
      {"let a = b + 1", 1, 9, "unknown name 'b'"},
      {"let a = a + 1\nlet b = a", 1, 9, "defined in terms of itself"},
      {"effect e = \"e\"\nmodule m = f, size 32", 2, 1, "unknown effect 'f'"},
      {"effect e = \"e\"\nmodule m = e, size 32\nevery frame:\n  start m", 4, 3, "'start' happens once"},
      {"at 1: stop m", 1, 7, "unknown module 'm'"},
      {"effect e = \"e\"\nmodule m = e, size 32\nwhen ember lands, at most 0:\n  stop m", 3, 27, "'at most' must be a whole number"},
      {"camera x temp", 1, 10, "'temp' is the landing particle's"},
      {"keyframes 20", 1, 11, "keyframe 20 is outside the scene"},
      {"scene length t", 1, 14, "cannot depend on time"},
      {"effect e = \"e\"\nmodule m = e, size 32\nmodule m = e, size 32", 3, 8, "already the name of a module"},
      {"effect e = \"e\"\nmodule m = e, size 32, tiles 2 x 2, width 64", 2, 1, "add 'band 8'"},
      {"effect e = \"e\"\nmodule m = e, size 32\nat 1: wake m", 3, 7, "'m' is not waiting"},
      {"effect e = \"e\"\nmodule m = e, size 32\nfield f = vortex, at (0, 0), on m", 3, 1, "a vortex field needs 'strength'"},
      {"effect e = \"e\"\nmodule m = e, size 32\nfield f = ceiling, level 3, radius 4, on m", 3, 29, "'radius' does not apply to a ceiling field"},
      {"effect e = \"e\"\nmodule m = e, size 32\nfield f = heat, at (0, 0), strength 1, on m, particles", 3, 43, "only a flow"},
      {"effect e = \"e\"\nmodule m = e, size 32\nlet v = m + 1", 3, 9, "a module is not a value"},
      {"effect e = \"e\"\nmodule m = e, size 32, tiles 2 x 1, band 4, at (t, 0)", 2, 49, "must be a constant"},
  };
  for (const Bad& b : cases) {
    try {
      sc::validate(sc::parse(b.text, "t"));
      ADD_FAILURE() << "no error for: " << b.text;
    } catch (const sc::Error& e) {
      EXPECT_EQ(e.pos.line, b.line) << b.text << "\n  " << e.what();
      EXPECT_EQ(e.pos.col, b.col) << b.text << "\n  " << e.what();
      EXPECT_NE(e.message.find(b.says), std::string::npos) << b.text << "\n  " << e.what();
    }
  }
}

// A word within two edits of exactly one known word gets a suggestion.
TEST(Script, ErrorsSuggestTheNearestWord) {
  const std::pair<const char*, const char*> cases[] = {
      {"effect e = \"e\"\nmodule m = e, size 32\nevery frame:\n  transfer m -> m, fractio 0.5", "(did you mean 'fraction'?)"},
      {"modle m = e", "(did you mean 'module'?)"},
      {"at 1:\n  wak m", "(did you mean 'wake'?)"},
      {"field f = vortx, at (0, 0)", "(did you mean 'vortex'?)"},
      {"let since = t - 1\nlet b = sinse * 2", "(did you mean 'since'?)"},
      {"let b = smoth(t)", "(did you mean 'smooth'?)"},
      {"effect fire = \"f\"\nmodule m = fier, size 32", "(did you mean 'fire'?)"},
      {"effect e = \"e\"\nmodule wreck = e, size 32\nat 1: stop wrek", "(did you mean 'wreck'?)"},
  };
  for (const auto& [text, says] : cases) {
    try {
      sc::validate(sc::parse(text, "t"));
      ADD_FAILURE() << "no error for: " << text;
    } catch (const sc::Error& e) {
      EXPECT_NE(e.message.find(says), std::string::npos) << text << "\n  " << e.what();
    }
  }
  EXPECT_EQ(sc::did_you_mean("zzzzzz", std::vector<std::string>{"fraction", "top"}), "");
}

// Problems that need the effects: sizes, controls, start points.
TEST(Script, BuildErrorsNameTheModule) {
  const Bad cases[] = {
      {"effect e = \"e\"\nmodule m = e, size 40", 2, 1, "multiple of its effect's grid"},
      {"effect e = \"e\"\nmodule m = e, size 32, heat_power 3", 2, 24, "not a property of a module nor a control of effect 'e'"},
      {"effect e = \"e\"\nmodule m = e, size 32, controls (1, 2)", 2, 1, "has 3 controls"},
      {"effect e = \"e\"\nmodule m = e, size 32, start 7", 2, 1, "has 2 start points"},
      {"effect e = \"e\"\nmodule m = e, size 32\nat 1: start m, from 5", 3, 7, "has 2 start points"},
      {"effect e = \"e\"\nmodule a = e, size 32\nmodule b = e, size 64\nat 1: hand_over a -> b", 4, 7, "same size and grid"},
  };
  for (const Bad& b : cases) {
    try {
      sc::Scene s(sc::parse(b.text, "t"), tiny_loader(), {1, Isa::base});
      ADD_FAILURE() << "no error for: " << b.text;
    } catch (const sc::Error& e) {
      EXPECT_EQ(e.pos.line, b.line) << b.text << "\n  " << e.what();
      EXPECT_EQ(e.pos.col, b.col) << b.text << "\n  " << e.what();
      EXPECT_NE(e.message.find(b.says), std::string::npos) << b.text << "\n  " << e.what();
    }
  }
}

using nfx::compose::testing::kEverything;

TEST(Script, RoundTripThroughThePrinter) {
  for (const std::string& text : {std::string(kEverything), read(std::filesystem::path(NEURALFX_SOURCE_DIR) / "examples/scenes/fireball.nvfxs")}) {
    ASSERT_FALSE(text.empty());
    const sc::Script a = sc::parse(text, "a");
    const std::string pa = sc::print(a);
    const sc::Script b = sc::parse(pa, "b");
    EXPECT_EQ(sc::print(b), pa);
    ASSERT_EQ(a.statements.size(), b.statements.size());
    for (std::size_t i = 0; i < a.statements.size(); ++i) {
      EXPECT_EQ(a.statements[i].keyword, b.statements[i].keyword);
      EXPECT_EQ(a.statements[i].props.size(), b.statements[i].props.size());
      EXPECT_EQ(a.statements[i].body.size(), b.statements[i].body.size());
    }
    sc::validate(a);
    sc::validate(b);
  }
  // numbers print as the shortest text that reads back as the same float
  const sc::Script s = sc::parse("let a = 0.1 + 1e-7 * 3.4028235e38 - 684 - 0.32 * 576\nlet b = (1 - 2) - (3 - 4) / (5 * (6 / 7))\nlet c = not (1 < 2) or 3 > 4 and 5 == 5", "n");
  EXPECT_EQ(sc::print(s), "let a = 0.1 + 1e-07 * 3.4028235e+38 - 684 - 0.32 * 576\nlet b = 1 - 2 - (3 - 4) / (5 * (6 / 7))\nlet c = not 1 < 2 or 3 > 4 and 5 == 5\n");
}

// The scene the round trip uses plays: every statement does something, and nothing allocates (checked in
// nvfx_alloc_test); on 1 and 2 threads the picture is the same.
TEST(Script, EverythingPlaysTheSameOnAnyNumberOfThreads) {
  const sc::Script s = sc::parse(kEverything, "everything");
  sc::Scene one(s, tiny_loader(), {1, Isa::base}), two(s, tiny_loader(), {2, Isa::base});
  ASSERT_EQ(one.frames(), 45);
  std::vector<std::uint8_t> a(160 * 90 * 3), b(a.size());
  for (int f = 0; f < one.frames(); ++f) {
    one.render(f, a);
    two.render(f, b);
    ASSERT_EQ(a, b) << "frame " << f;
  }
  EXPECT_GT(std::accumulate(a.begin(), a.end(), 0L), 0L);
  EXPECT_EQ(one.rule_count("boom"), 1);
  EXPECT_NEAR(one.rule_time("boom"), 3.f / 30.f, 1e-6f);
  EXPECT_EQ(one.rule_count("hit"), 1);
  EXPECT_EQ(one.rule_count("later"), 1);
  EXPECT_FALSE(one.module("pair", 0)->active);  // handed over
  EXPECT_TRUE(one.module("sky", 1)->active);
  EXPECT_EQ(one.module("sky", 1)->seed, 41u);    // seed + 10 row + column
  EXPECT_EQ(one.rule_count("lit"), 1);           // the torch heated the fire's air enough
  EXPECT_EQ(one.rule_count("nonexistent"), -1);
  EXPECT_EQ(one.keyframes().size(), 2u);
}

// Options::overlap draws each picture while the next frame's state is computed: the same frames.
TEST(Script, OverlapGivesTheSameFrames) {
  const sc::Script s = sc::parse(kEverything, "everything");
  sc::Scene one(s, tiny_loader(), {1, Isa::base}), two(s, tiny_loader(), {2, Isa::base, true}), three(s, tiny_loader(), {3, Isa::base, true});
  std::vector<std::uint8_t> a(160 * 90 * 3), b(a.size()), c(a.size());
  for (int f = 0; f < one.frames(); ++f) {
    one.render(f, a);
    two.render(f, b);
    three.render(f, c);
    ASSERT_EQ(a, b) << "frame " << f;
    ASSERT_EQ(a, c) << "frame " << f;
  }
  EXPECT_EQ(three.rule_count("boom"), 1);
}

// Inputs (`input NAME = value`): values the game sets while the scene plays, read like t, never where a constant is
// needed; a rule on an input fires when the game sets it.
TEST(Script, InputsAreValuesTheGameSets) {
  const Bad cases[] = {
      {"input a = t", 1, 11, "an input's starting value must be a constant"},
      {"input a = 1\nscene length a", 2, 14, "the scene's settings cannot depend on inputs"},
      {"effect e = \"e\"\ninput a = 1\nmodule m = e, size 32, tiles 2 x 1, band 4, at (a, 0)", 3, 49, "cannot depend on time, rules, modules, inputs or rand"},
      {"input a = 1\ninput a = 2", 2, 7, "'a' is already the name of an input (line 1)"},
      {"input t = 1", 1, 7, "'t' is a word of the script language"},
      {"input wind_in = 1\nlet b = wind_im + 1", 2, 9, "(did you mean 'wind_in'?)"},
  };
  for (const Bad& b : cases) {
    try {
      sc::validate(sc::parse(b.text, "t"));
      ADD_FAILURE() << "no error for: " << b.text;
    } catch (const sc::Error& e) {
      EXPECT_EQ(e.pos.line, b.line) << b.text << "\n  " << e.what();
      EXPECT_EQ(e.pos.col, b.col) << b.text << "\n  " << e.what();
      EXPECT_NE(e.message.find(b.says), std::string::npos) << b.text << "\n  " << e.what();
    }
  }
  EXPECT_EQ(sc::print(sc::parse("input wind_in = 0.5 * 2   # from the game\nlet w = wind_in + 1", "t")), "input wind_in = 0.5 * 2\nlet w = wind_in + 1\n");
  constexpr const char* text = R"(
scene size 160 x 90, fps 30, length 1, ground 80
effect tiny = "tiny"
input spot = 40
input go = 0
module fire = tiny, size 32, at (spot, ground), start 1, seed 5
when go > 0 as lit:
  shock at (80, 50)
)";
  sc::Scene scene(sc::parse(text, "t"), tiny_loader(), {1, Isa::base});
  ASSERT_EQ(scene.inputs(), 2);
  EXPECT_EQ(scene.input_name(1), "go");
  EXPECT_EQ(scene.input_index("spot"), 0);
  EXPECT_EQ(scene.input_index("nope"), -1);
  EXPECT_EQ(scene.input(0), 40.f);
  std::vector<std::uint8_t> rgb(160 * 90 * 3);
  scene.render(0, rgb);
  scene.render(1, rgb);
  EXPECT_EQ(scene.rule_count("lit"), 0);
  EXPECT_EQ(scene.module_info(0).x, 40.f);
  scene.set_input(0, 100.f);
  scene.set_input(1, 1.f);
  scene.render(2, rgb);
  EXPECT_EQ(scene.module_info(0).x, 100.f);
  EXPECT_EQ(scene.rule_count("lit"), 1);
  EXPECT_EQ(scene.rule_time("lit"), 2.f / 30.f);
  sc::Scene started(sc::parse(text, "t"), tiny_loader(), {1, Isa::base, false, {{"spot", 70.f}}});
  EXPECT_EQ(started.module_info(0).x, 70.f);
  EXPECT_THROW(sc::Scene(sc::parse(text, "t"), tiny_loader(), {1, Isa::base, false, {{"spto", 70.f}}}), std::invalid_argument);
}

// --- the runner matches the same scene written in C++ ------------------------------------------------------------------

constexpr const char* kMini = R"(
scene size 160 x 90, fps 30, length 1, ground 80
bus at (-16, -16), size 192 x 128, cell 8
particles capacity 512, flow 1
effect tiny = "tiny"
look gas = shader, heat_scale 1.2, emission 2
module pair = tiny, tiles 2 x 1, band 4, size 32, width 64, at (80, 74), look gas
module fire = tiny, size 32, width 32, at (66, 72), feather 0.125, start 1, seed 99, intensity 0.3 + 0.2 * smooth(t / 0.5)
field cap = ceiling, level 20, soft 10, damping 0.3, on pair
field swirl = vortex, at (60, 40), radius 20, strength 1, on pair
emit embers, on fire, tries 2, chance 0.6, spread 10
light gain 0.2, flash (0.1, 0.05, 0)
frame haze 1, bloom 1, bloom_threshold 0.8
camera x (rand - 0.5) * 2
at 0.1 as boom:
  start pair, from 0, seed 7
  shock at (80, 50), speed 600, decay 0.2, amp 3, width 8
  scorch at (80, 82), radius 20, glow 0.5
  burst at (80, 60), radius 5, embers 20, debris 4, speed 100
every frame:
  transfer fire -> pair, top 4, fraction 0.2, heat 0, soot 2
  push fire, gain 0.2
)";

// kMini by hand, in the order of script_run.cpp's frame, with the random draws written as the hand-written fireball
// writes them (spawn() arguments in one call: the compiler's order of evaluation, which the runner reproduces).
struct MiniByHand {
  rt::RolloutEffect e = tiny_effect(3);
  std::vector<std::unique_ptr<Module>> owned;
  Module *p0, *p1, *fire;
  FieldBus bus{-16.f, -16.f, 24, 16, 8.f, 2};
  Light light{bus};
  Particles parts{512};
  Frame frame{160, 90};
  Pool pool;
  std::vector<Module*> active;
  std::vector<Shock> shocks;
  std::vector<std::array<float, 4>> scorch;
  bool boom = false;

  explicit MiniByHand(int threads) : pool(threads) {
    ShaderSpec gas;
    gas.heat_scale = 1.2f;
    gas.emission = 2.f;
    for (int c = 0; c < 2; ++c) {  // stride 64 * 12 / 16 = 48, domain 112 wide centred on 80, standing on 74
      owned.push_back(std::make_unique<Module>("p", e, 32, Placement{24.f + 48.f * static_cast<float>(c), 10.f, 2.f}, Isa::base));
      Module* m = owned.back().get();
      m->group = 0;
      m->band = {c == 1 ? 4 : 0, c == 0 ? 4 : 0, 0, 0};
      m->look = Look::shader;
      m->spec = gas;
    }
    owned.push_back(std::make_unique<Module>("fire", e, 32, Placement{50.f, 40.f, 1.f}, Isa::base));
    p0 = owned[0].get();
    p1 = owned[1].get();
    fire = owned[2].get();
    fire->group = 1;
    fire->feather = 0.125f * 32.f;
    fire->start(1, 99);
    frame.ground_y = 80.f;
  }

  void step(int f, std::span<std::uint8_t> rgb) {
    const float t = static_cast<float>(f) / 30.f;
    if (!boom && t >= 0.1f) {
      boom = true;
      for (int c = 0; c < 2; ++c) {
        Module* m = owned[z(c)].get();
        m->controls = e.m.starts[0].controls;
        if (c == 1) m->start(0, 7);
        else m->start_empty(e.m.starts[0].time, 7 + static_cast<std::uint64_t>(c));
      }
      shocks.push_back({80.f, 50.f, t, 600.f, 0.2f, 3.f, 8.f});
      scorch.push_back({80.f, 82.f, 20.f, 0.5f});
      const float x = 80.f, y = 60.f, r = 5.f, speed = 100.f;
      for (int i = 0; i < 20; ++i) {  // the fireball's burst(), as written there
        const float a = parts.uniform() * 6.2831853f, d = std::sqrt(parts.uniform()) * r;
        const float s = speed * (0.3f + 0.7f * parts.uniform());
        parts.spawn(Kind::ember, x + d * std::cos(a), y + d * std::sin(a), s * std::cos(a), s * std::sin(a) - 0.45f * speed, 0.8f + 0.4f * parts.uniform(),
                    0.7f + 1.1f * parts.uniform(), 2.f + 3.f * parts.uniform());
      }
      for (int i = 0; i < 4; ++i) {
        const float a = 3.1415927f + parts.uniform() * 3.1415927f, s = speed * (0.4f + 0.9f * parts.uniform());
        parts.spawn(Kind::debris, x, y, s * std::cos(a), s * std::sin(a), 0.5f + 0.4f * parts.uniform(), 2.f + 3.f * parts.uniform(), 6.f);
      }
    }
    if (fire->active) {  // the fireball's embers from its fires, as written there
      const float cx = fire->at.x + 0.5f * static_cast<float>(fire->size()) * fire->at.scale;
      for (int i = 0; i < 2; ++i) {
        if (parts.uniform() < 0.6f) {
          parts.spawn(Kind::ember, cx + (parts.uniform() - 0.5f) * 10.f, 72.f - 20.f, (parts.uniform() - 0.5f) * 30.f, -(60.f + 90.f * parts.uniform()), 0.7f,
                      0.6f + 0.5f * parts.uniform(), 1.5f + 1.5f * parts.uniform());
        }
      }
    }
    fire->controls[0] = 0.3f + 0.2f * [](float v) {
      v = std::clamp(v, 0.f, 1.f);
      return v * v * (3.f - 2.f * v);
    }(t / 0.5f);
    frame.exposure = 1.f;
    frame.fade = 1.f;
    frame.haze = 1.f;
    frame.time = t;
    frame.cam_x = (parts.uniform() - 0.5f) * 2.f;
    frame.cam_y = 0.f;
    active.clear();
    for (const auto& m : owned)
      if (m->active) active.push_back(m.get());
    pool.run(static_cast<int>(active.size()), [&](int i) { active[z(i)]->step(); });
    if (p0->active) blend_band(*p0, *p1, Side::right, 4);
    std::vector<Module*> to;
    for (Module* m : {p0, p1})
      if (m->active) to.push_back(m);
    if (!to.empty()) transfer(*fire, to, 0.2f, 16 - 4, 0.f, 2.f);
    bus.clear();
    for (Module* m : active) bus.publish(*m);
    push(*fire, bus, 0.2f);
    ForceField cap, swirl;
    cap.kind = ForceField::Kind::ceiling;
    cap.y = 20.f;
    cap.soft = 10.f;
    cap.damping = 0.3f;
    swirl.kind = ForceField::Kind::vortex;
    swirl.x = 60.f;
    swirl.y = 40.f;
    swirl.radius = 20.f;
    swirl.strength = 1.f;
    for (Module* m : {p0, p1}) apply(*m, cap, 1.f);
    for (Module* m : {p0, p1}) apply(*m, swirl, 1.f);
    light.update(bus, 0.2f, {0.1f, 0.05f, 0.f}, pool);
    parts.update(1.f / 30.f, &bus, 1.f, 80.f);
    pool.run(static_cast<int>(active.size()), [&](int i) { active[z(i)]->shade(&light); });
    frame.background(light, scorch, pool);
    frame.draw(active, pool);
    frame.particles(parts);
    frame.distort(shocks, bus, pool);
    frame.bloom(0.8f, 1.f, pool);
    frame.finish(rgb, pool);
  }
};

TEST(Script, MatchesTheSameSceneWrittenInCpp) {
  sc::Scene scene(sc::parse(kMini, "mini"), tiny_loader(), {2, Isa::base});
  MiniByHand hand(2);
  std::vector<std::uint8_t> a(160 * 90 * 3), b(a.size());
  ASSERT_EQ(scene.frames(), 30);
  for (int f = 0; f < scene.frames(); ++f) {
    scene.render(f, a);
    hand.step(f, b);
    ASSERT_EQ(a, b) << "frame " << f;
    ASSERT_EQ(scene.particles().alive(), hand.parts.alive()) << "frame " << f;
  }
  EXPECT_GT(std::accumulate(a.begin(), a.end(), 0L), 0L);
  EXPECT_GT(hand.parts.alive(), 0);
}

// The script computes in float, operation by operation, as C++ does: 1.2 + 2.4 is a little more than 3.6 in float, so a
// rule at 1.2 + 2.4 fires a frame after a rule at 3.6 (the hand-written fireball's hand-over is at 3.63 s for this).
TEST(Script, ExpressionsComputeInFloatAsCppDoes) {
  ASSERT_GT(1.2f + 2.4f, 3.6f);
  const char* text = R"(
scene size 64 x 64, length 4
effect e = "e"
module m = e, size 32
let t_det = 1.2
at t_det + 2.4 as sum: stop m
at 3.6 as literal: stop m
)";
  sc::Scene s(sc::parse(text, "f"), tiny_loader(), {1, Isa::base});
  std::vector<std::uint8_t> rgb(64 * 64 * 3);
  for (int f = 0; f < s.frames(); ++f) s.render(f, rgb);
  EXPECT_EQ(s.rule_time("literal"), 108.f / 30.f);
  EXPECT_EQ(s.rule_time("sum"), 109.f / 30.f);
}

TEST(Script, RulesFireOnceRepeatOrAtMostN) {
  const char* text = R"(
scene size 64 x 64, length 1
effect e = "e"
module m = e, size 32, start 0
when t >= 0.5 as once: stop m
when t >= 0.5 as again, repeat: stop m
when t >= 0.5 as three, at most 3: stop m
when once < infinity and t - once > 0.2 as after: stop m
when 1 > 2 as never: stop m
)";
  sc::Scene s(sc::parse(text, "r"), tiny_loader(), {1, Isa::base});
  std::vector<std::uint8_t> rgb(64 * 64 * 3);
  for (int f = 0; f < s.frames(); ++f) s.render(f, rgb);
  EXPECT_EQ(s.rule_count("once"), 1);
  EXPECT_EQ(s.rule_count("again"), 15);  // frames 15 to 29
  EXPECT_EQ(s.rule_count("three"), 3);
  EXPECT_EQ(s.rule_count("after"), 1);
  EXPECT_NEAR(s.rule_time("after"), 22.f / 30.f, 1e-6f);
  EXPECT_EQ(s.rule_count("never"), 0);
  EXPECT_TRUE(std::isinf(s.rule_time("never")));
  EXPECT_FALSE(s.module("m")->active);
}

// `when shock N reaches M` fires in the first frame whose shock radius is at least the distance to where M stands.
TEST(Script, ShockFrontReachingAModuleTriggers) {
  const char* text = R"(
scene size 160 x 90, length 1, ground 80
effect e = "e"
module far = e, size 32, at (150, 80)
at 0.1: shock at (10, 80), speed 400, decay 0.2
when shock 1 reaches far as hit: start far
)";
  sc::Scene s(sc::parse(text, "s"), tiny_loader(), {1, Isa::base});
  std::vector<std::uint8_t> rgb(160 * 90 * 3);
  int expect = -1;
  Shock sh;
  sh.x = 10.f;
  sh.y = 80.f;
  sh.t0 = 3.f / 30.f;
  sh.speed = 400.f;
  sh.decay = 0.2f;
  for (int f = 0; f < 30 && expect < 0; ++f)
    if (sh.radius(static_cast<float>(f) / 30.f) >= std::hypot(150.f - 10.f, 80.f - 80.f)) expect = f;
  ASSERT_GT(expect, 3);
  for (int f = 0; f < s.frames(); ++f) s.render(f, rgb);
  EXPECT_EQ(s.rule_time("hit"), static_cast<float>(expect) / 30.f);
  EXPECT_TRUE(s.module("far")->active);
}

// Embers that land hot wake a waiting module where they land, at most as often as the rule says.
TEST(Script, EmbersLandingWakeWaitingModules) {
  const char* text = R"(
scene size 160 x 90, length 2, ground 80
particles capacity 4096, flow 0
effect e = "e"
module a = e, size 32, width 24, sink 0.1, start 0, waiting, opacity smooth(a.age / 0.5)
module b = e, size 32, width 24, sink 0.1, start 1, waiting
at 0.05: burst at (80, 60), radius 4, embers 300, speed 150
when ember lands where temp >= 0.3 and not near(a, x, 30), at most 2 as lit:
  wake a, b at (x, ground)
)";
  sc::Scene s(sc::parse(text, "l"), tiny_loader(), {1, Isa::base});
  std::vector<std::uint8_t> rgb(160 * 90 * 3);
  s.render(0, rgb);
  EXPECT_FALSE(s.module("a")->active);
  EXPECT_EQ(s.module("a")->opacity, 0.f);  // smooth(-infinity): not woken yet
  for (int f = 1; f < s.frames(); ++f) s.render(f, rgb);
  EXPECT_EQ(s.rule_count("lit"), 2);
  for (const char* n : {"a", "b"}) {
    const Module* m = s.module(n);
    EXPECT_TRUE(m->active) << n;
    EXPECT_FLOAT_EQ(m->at.y, 80.f + 0.1f * 24.f - 24.f) << n;  // standing on the ground, sunk by 10% of its width
  }
  EXPECT_GE(std::fabs((s.module("a")->at.x + 12.f) - (s.module("b")->at.x + 12.f)), 30.f);
  EXPECT_GT(s.module("a")->opacity, 0.f);
}

// --- field effects -----------------------------------------------------------------------------------------------------

// A module with soot (and heat) in a blob, nothing moving.
std::unique_ptr<Module> blob(const rt::RolloutEffect& e, float cx_cells, float cy_cells) {
  auto m = std::make_unique<Module>("m", e, 32, Placement{0.f, 0.f, 4.f}, Isa::base);  // 128 world pixels, 8 per cell
  m->start_empty(0.f, 1);
  auto co = m->runner().coarse_mut();
  const int C = m->channels();
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      const float d2 = (static_cast<float>(x) - cx_cells) * (static_cast<float>(x) - cx_cells) + (static_cast<float>(y) - cy_cells) * (static_cast<float>(y) - cy_cells);
      co[z(y * 16 + x) * z(C) + 2] = 0.8f * std::exp(-d2 / 6.f);
      co[z(y * 16 + x) * z(C) + 3] = 0.6f * std::exp(-d2 / 6.f);
    }
  }
  auto ft = m->runner().fine_heat_mut();
  auto fd = m->runner().fine_soot_mut();
  for (int y = 0; y < 32; ++y) {
    for (int x = 0; x < 32; ++x) {
      const float cx = (static_cast<float>(x) + 0.5f) / 2.f - 0.5f, cy = (static_cast<float>(y) + 0.5f) / 2.f - 0.5f;
      const float d2 = (cx - cx_cells) * (cx - cx_cells) + (cy - cy_cells) * (cy - cy_cells);
      ft[z(y * 32 + x)] = 0.8f * std::exp(-d2 / 6.f);
      fd[z(y * 32 + x)] = 0.6f * std::exp(-d2 / 6.f);
    }
  }
  return m;
}

double total(std::span<const float> v, int stride = 1, int offset = 0) {
  double s = 0;
  for (std::size_t i = z(offset); i < v.size(); i += z(stride)) s += v[i];
  return s;
}

// Centre of the fine soot in world pixels (y down).
std::array<double, 2> centroid(const Module& m) {
  double sx = 0, sy = 0, s = 0;
  const auto fd = m.runner().fine_soot();
  const int S = m.size();
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      const double v = fd[z(y * S + x)];
      sx += v * (m.at.x + (x + 0.5) * m.at.scale);
      sy += v * (m.at.y + (S - y - 0.5) * m.at.scale);
      s += v;
    }
  }
  return {sx / s, sy / s};
}

TEST(Fields, GustVariesAroundTheWindAndTravelsWithIt) {
  ForceField g;
  g.kind = ForceField::Kind::gust;
  g.u = 2.f;
  g.v = 0.f;
  g.amount = 0.6f;
  g.soft = 50.f;
  g.rate = 0.5f;
  double su = 0, su2 = 0, sv = 0;
  int n = 0;
  float lo = 1e9f, hi = -1e9f;
  for (float y = 0.f; y < 400.f; y += 13.f) {
    for (float x = 0.f; x < 400.f; x += 11.f) {
      const auto uv = field_flow(g, x, y, 1.f);
      su += uv[0];
      su2 += static_cast<double>(uv[0]) * uv[0];
      sv += uv[1];
      lo = std::min(lo, uv[0]);
      hi = std::max(hi, uv[0]);
      ++n;
      ASSERT_GE(uv[0], 0.f);  // gusts change the wind's speed; they do not blow it backwards
    }
  }
  const double mean = su / n, sd = std::sqrt(su2 / n - mean * mean);
  EXPECT_NEAR(mean, 2.0, 0.4);
  EXPECT_GT(sd, 0.2);  // gusty, not uniform
  EXPECT_LT(hi - lo, 2.f * 2.f * 0.6f * 2.f + 0.1f);
  EXPECT_NEAR(sv / n, 0.0, 0.3);
  // the pattern travels with the wind: one second later it is 60 pixels downwind (2 pixels a frame, 30 frames)
  ForceField later = g;
  later.time = 1.f;
  later.rate = 0.f;
  g.rate = 0.f;
  for (float x = 0.f; x < 200.f; x += 17.f) {
    const auto a = field_flow(g, x, 100.f, 1.f), b = field_flow(later, x + 60.f, 100.f, 1.f);
    EXPECT_NEAR(a[0], b[0], 1e-4f);
  }
  // applied to a module: a push for one step, the wind's size on average
  const rt::RolloutEffect e = tiny_effect();
  auto m = blob(e, 8.f, 8.f);
  apply(*m, g, 1.f);
  EXPECT_NEAR(total(m->pushed(), 2, 0) / 256.0, 2.0 / 8.0, 0.1);  // 2 world pixels a frame = 1/4 cell
}

TEST(Fields, VortexRingBlowsThroughItsCentre) {
  ForceField r;
  r.kind = ForceField::Kind::ring;
  r.x = 100.f;
  r.y = 100.f;
  r.u = 0.f;
  r.v = -1.f;  // upwards on screen
  r.radius = 30.f;
  r.soft = 12.f;
  r.strength = 3.f;
  const auto c = field_flow(r, 100.f, 100.f, 1.f);
  EXPECT_NEAR(c[0], 0.f, 1e-5f);
  EXPECT_LT(c[1], -1.f);  // strongly along the direction at the centre
  // outside the cores the flow turns back (a vortex ring's return flow), and far away it fades
  const auto out = field_flow(r, 100.f + 60.f, 100.f, 1.f);
  EXPECT_GT(out[1], 0.f);
  const auto far = field_flow(r, 400.f, 400.f, 1.f);
  EXPECT_LT(std::hypot(far[0], far[1]), 1e-3f);
  // mirror symmetry about its axis
  const auto a = field_flow(r, 90.f, 120.f, 1.f), b = field_flow(r, 110.f, 120.f, 1.f);
  EXPECT_NEAR(a[0], -b[0], 1e-5f);
  EXPECT_NEAR(a[1], b[1], 1e-5f);
  // a ring carries material: a blob of soot on its axis moves along it over a few steps, faster than without it
  const rt::RolloutEffect e = tiny_effect();
  auto m = blob(e, 8.f, 6.f), ref = blob(e, 8.f, 6.f);
  r.x = 64.f;
  r.y = 80.f;
  r.radius = 20.f;
  r.soft = 10.f;
  r.strength = 4.f;
  for (int i = 0; i < 4; ++i) {
    apply(*m, r, 1.f);
    m->step();
    ref->step();
  }
  EXPECT_LT(centroid(*m)[1], centroid(*ref)[1] - 1.0);  // further up (y down)
}

TEST(Fields, AttractorPullsMaterialInAndKeepsTheAmount) {
  const rt::RolloutEffect e = tiny_effect();
  auto m = blob(e, 4.f, 8.f);  // left of the middle
  ForceField a;
  a.kind = ForceField::Kind::attract;
  a.x = 96.f;  // right of the blob
  a.y = 64.f;
  a.radius = 50.f;
  a.strength = 6.f;
  a.damping = 0.f;
  const int C = m->channels();
  const double h0 = total(m->runner().coarse(), C, 2), d0 = total(m->runner().coarse(), C, 3);
  const double fh0 = total(m->runner().fine_heat()), fd0 = total(m->runner().fine_soot());
  const auto c0 = centroid(*m);
  std::vector<float> scratch(pull_scratch(*m));
  for (int i = 0; i < 5; ++i) pull(*m, a, 1.f, scratch);
  EXPECT_NEAR(total(m->runner().coarse(), C, 2), h0, 1e-4 * h0);
  EXPECT_NEAR(total(m->runner().coarse(), C, 3), d0, 1e-4 * d0);
  EXPECT_NEAR(total(m->runner().fine_heat()), fh0, 1e-4 * fh0);
  EXPECT_NEAR(total(m->runner().fine_soot()), fd0, 1e-4 * fd0);
  const auto c1 = centroid(*m);
  EXPECT_GT(c1[0], c0[0] + 15.0);  // drawn towards x = 96
  EXPECT_LT(std::fabs(c1[1] - 64.0), std::fabs(c0[1] - 64.0) + 1.0);
  // negative strength pushes away; the swirl is a push around the point
  auto r = blob(e, 4.f, 8.f);
  a.strength = -4.f;
  for (int i = 0; i < 3; ++i) pull(*r, a, 1.f, scratch);
  EXPECT_LT(centroid(*r)[0], c0[0] - 3.0);
  a.damping = 2.f;
  const auto s = field_flow(a, 96.f, 64.f - 50.f, 1.f);  // above the point: clockwise on screen is to the right
  EXPECT_GT(s[0], 1.f);
  EXPECT_LT(std::fabs(s[1]), 1e-4f);
}

TEST(Fields, HeatSourceAddsHeatWhereItIs) {
  const rt::RolloutEffect e = tiny_effect();
  auto m = blob(e, 8.f, 8.f);
  ForceField h;
  h.kind = ForceField::Kind::heat;
  h.x = 32.f;
  h.y = 96.f;
  h.radius = 12.f;
  h.strength = 0.1f;
  const int C = m->channels();
  const std::vector<float> before(m->runner().coarse().begin(), m->runner().coarse().end());
  const std::vector<float> fine0(m->runner().fine_heat().begin(), m->runner().fine_heat().end());
  apply(*m, h, 1.f);
  const auto co = m->runner().coarse();
  // the cell under the point (world 32, 96: cell x 3.5, y up 3.5) gets nearly the full rate; one far away nothing
  const std::size_t near = z(3 * 16 + 3) * z(C), far = z(14 * 16 + 14) * z(C);
  EXPECT_NEAR(co[near + 2] - before[near + 2], 0.1f, 0.03f);
  EXPECT_EQ(co[far + 2], before[far + 2]);
  EXPECT_EQ(co[near + 3], before[near + 3]);  // soot untouched
  EXPECT_GT(total(m->runner().fine_heat()), total(fine0) + 1.0);
  // the weight scales it; a module that is not active is left alone
  m->active = false;
  const std::vector<float> idle(m->runner().coarse().begin(), m->runner().coarse().end());
  apply(*m, h, 1.f);
  EXPECT_TRUE(std::ranges::equal(idle, m->runner().coarse()));
}

TEST(Fields, ColdPutsOutHeatAndMakesSteam) {
  const rt::RolloutEffect e = tiny_effect();
  auto m = blob(e, 8.f, 8.f);
  ForceField c;
  c.kind = ForceField::Kind::cold;
  c.x = 64.f;
  c.y = 64.f;
  c.radius = 20.f;
  c.strength = 0.5f;
  c.damping = 0.4f;  // steam: 40% of the heat removed becomes soot
  const int C = m->channels();
  const std::vector<float> before(m->runner().coarse().begin(), m->runner().coarse().end());
  apply(*m, c, 1.f);
  const auto co = m->runner().coarse();
  double removed = 0, added = 0;
  for (int i = 0; i < 256; ++i) {
    const std::size_t q = z(i) * z(C);
    ASSERT_LE(co[q + 2], before[q + 2]);
    removed += before[q + 2] - co[q + 2];
    added += co[q + 3] - before[q + 3];
  }
  const std::size_t mid = z(7 * 16 + 7) * z(C);
  EXPECT_NEAR(co[mid + 2], before[mid + 2] * 0.5f, 0.05f * before[mid + 2]);  // half the heat at the centre
  EXPECT_NEAR(added, 0.4 * removed, 1e-4 * removed);
  EXPECT_GT(removed, 0.5);
  // far from the centre almost nothing changes
  const std::size_t edge = z(0 * 16 + 0) * z(C);
  EXPECT_NEAR(co[edge + 2], before[edge + 2], 1e-3f * std::max(1e-3f, before[edge + 2]) + 1e-6f);
}

// In a script, a heat source warms the air until a rule on the bus's heat lights the fuel there.
TEST(Fields, HeatSourceIgnitesFuelThroughARule) {
  const char* text = R"(
scene size 160 x 90, length 2, ground 80
effect e = "e"
module air = e, size 32, width 128, at (80, 100), start 0, seed 3
module fuel = e, size 32, width 24, start 1, waiting
field torch = heat, at (40, 60), radius 12, strength 0.08, on air, until 1
when heat(40, 60) > 0.6 as ignite:
  wake fuel at (40, ground)
)";
  sc::Scene s(sc::parse(text, "h"), tiny_loader(), {1, Isa::base});
  std::vector<std::uint8_t> rgb(160 * 90 * 3);
  for (int f = 0; f < s.frames(); ++f) s.render(f, rgb);
  EXPECT_EQ(s.rule_count("ignite"), 1);
  EXPECT_GT(s.rule_time("ignite"), 0.f);
  EXPECT_TRUE(s.module("fuel")->active);
}

// In a script, a cold field without `steam` makes no soot, and an attractor without `swirl` does not push.
TEST(Fields, ScriptedFieldsDefaultToNoSteamAndNoSwirl) {
  const std::string base = R"(
scene size 160 x 90, length 0.2, ground 80
effect e = "e"
module m = e, size 32, width 128, at (64, 128), start 0
field a = attract, at (64, 64), radius 40, strength 0.001, on m
)";
  const std::string cold = base + "field c = cold, at (64, 64), radius 40, strength 0.5, on m\n";
  sc::Scene with(sc::parse(cold, "c"), tiny_loader(), {1, Isa::base}), without(sc::parse(base, "b"), tiny_loader(), {1, Isa::base});
  std::vector<std::uint8_t> rgb(160 * 90 * 3);
  with.render(0, rgb);  // step, then the fields: what is pushed now is what the fields added
  without.render(0, rgb);
  Module* m = with.module("m");
  const int C = m->channels();
  EXPECT_EQ(total(m->pushed(), 1, 0), 0.0);  // no swirl: nothing pushed
  EXPECT_LT(total(m->runner().coarse(), C, 2), total(without.module("m")->runner().coarse(), C, 2));  // heat taken
  EXPECT_EQ(total(m->runner().coarse(), C, 3), total(without.module("m")->runner().coarse(), C, 3));  // no steam
}

// Flow fields can act on particles too: a wind blows embers sideways.
TEST(Fields, WindOnParticlesBlowsThem) {
  const char* base = R"(
scene size 160 x 90, length 0.5, ground 80
particles capacity 64, flow 0
effect e = "e"
module m = e, size 32
at 0: burst at (80, 40), radius 1, embers 20, speed 1
)";
  const std::string blown = std::string(base) + "field w = wind, velocity (3, 0), on particles\n";
  sc::Scene still(sc::parse(base, "a"), tiny_loader(), {1, Isa::base}), windy(sc::parse(blown, "b"), tiny_loader(), {1, Isa::base});
  std::vector<std::uint8_t> rgb(160 * 90 * 3);
  for (int f = 0; f < 10; ++f) {
    still.render(f, rgb);
    windy.render(f, rgb);
  }
  double vs = 0, vw = 0;
  still.particles().each([&](float, float, float& vx, float&) { vs += vx; });
  windy.particles().each([&](float, float, float& vx, float&) { vw += vx; });
  EXPECT_GT(vw / 20.0, vs / 20.0 + 20.0);  // pixels per second
}

}  // namespace
