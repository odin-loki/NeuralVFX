// Three features for engines (docs/COMPOSE.md §5.3 and §4.6, docs/ENGINES.md §11, docs/LIBRARY.md §9), with the tests'
// untrained rollout effects, so no data is needed:
//   - trails: a moving source leaves material in tiles that stay in the world, are taken again as it moves on, fade;
//   - the engine layers (lit sprites): deterministic, consistent with each other, the picture unchanged;
//   - weather: procedural rain, snow, ash and leaves that keep the scene's random numbers and couple to the bus.
#include "compose_scene.hpp"
#include "library.hpp"
#include "script.hpp"

#include <neuralfx/nvfx_library.h>
#include <neuralfx/rollout.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace sc = nfx::compose::script;
namespace cm = nfx::compose;

sc::Scene scene_of(const std::string& text, int threads = 2, bool overlap = false, bool layers = false) {
  sc::Options o;
  o.threads = threads;
  o.overlap = overlap;
  o.layers = layers;
  return sc::Scene(sc::parse(text, "t"), [](const std::string& f) { return cm::testing::tiny_effect(f == "other" ? 4 : 3); }, o);
}

std::string error_of(const std::string& text) {
  try {
    sc::Scene s = scene_of(text);
  } catch (const sc::Error& e) {
    return e.what();
  }
  return "no error";
}

// A source that crosses the scene from left to right in a second and a half, leaving soot in tiles of the tiny effect.
constexpr const char* kTrail = R"(
scene size 240 x 120, fps 30, length 3 s, ground 110
effect tiny = "tiny"
look gas = shader, heat_scale 1, emission 2
let sx = 20 + 140 * min(t, 1.5)
trail streak = tiny, size 32, width 64, band 4, tiles 12, at (sx, 60), look gas, age 3, hold 0.4, fade 0.3,
      soot 0.5, heat 0.1, radius 6, kick -0.2, velocity (0, -0.3), seed 5, if t < 1.5
)";

struct EffectDeleter {
  void operator()(nvfx_effect* e) const { nvfx_effect_free(e); }
};
using EffectPtr = std::unique_ptr<nvfx_effect, EffectDeleter>;
EffectPtr tiny_nvfx(std::uint64_t seed) {
  const nfx::rt::RolloutEffect t = cm::testing::tiny_effect(seed);
  std::ostringstream os;
  EXPECT_TRUE(nfx::rollout::save_model(os, t.m));
  const std::string bytes = os.str();
  nvfx_effect* e = nullptr;
  EXPECT_EQ(nvfx_effect_load_memory(bytes.data(), bytes.size(), &e), NVFX_OK);
  return EffectPtr(e);
}

}  // namespace

// --- trails ------------------------------------------------------------------------------------------------------------

TEST(Trails, TilesLieOnAGridInTheWorldAlongThePath) {
  sc::Scene scene = scene_of(kTrail);
  std::vector<std::uint8_t> rgb(240 * 120 * 3);
  int most = 0;
  for (int f = 0; f < 40; ++f) {
    scene.render(f, rgb);
    int on = 0;
    for (const cm::Module* m : scene.modules()) {
      if (!m->active) continue;
      ++on;
      // the stride is 64 * (16 - 4) / 16 = 48: every tile's corner is on the grid (row 0 stands on the ground)
      EXPECT_FLOAT_EQ(std::fmod(m->at.x, 48.f), 0.f) << m->name();
      EXPECT_FLOAT_EQ(std::fmod(m->at.y - (110.f - 64.f) + 4800.f, 48.f), 0.f) << m->name();
    }
    most = std::max(most, on);
  }
  EXPECT_GE(most, 3) << "a path 180 pixels long over tiles 48 apart";
  const sc::Scene::ModuleInfo info = scene.module_info(0);
  EXPECT_EQ(info.tiles, 12);
  EXPECT_NEAR(info.x, 20.f + 140.f * 39.f / 30.f, 1e-3f) << "the trail's place is its source's";
}

TEST(Trails, MaterialStaysWhereItWasLeftAndFades) {
  sc::Scene scene = scene_of(kTrail);
  std::vector<std::uint8_t> rgb(240 * 120 * 3);
  for (int f = 0; f <= 30; ++f) scene.render(f, rgb);
  // the source passed x = 60 at 0.29 s and is at 160 now: its soot is still there, in the world
  float behind = 0.f;
  for (int y = 40; y <= 80; y += 4) behind = std::max(behind, scene.bus().at(60.f, static_cast<float>(y)).soot);
  EXPECT_GT(behind, 0.01f) << "soot left at x = 60 stays in the world";
  // the source stops leaving anything at 1.5 s; its tiles fade (hold 0.4 + 5 x fade 0.3) and are switched off
  for (int f = 31; f < 4 * 30; ++f) scene.render(f, rgb);
  int on = 0;
  for (const cm::Module* m : scene.modules()) on += m->active ? 1 : 0;
  EXPECT_EQ(on, 0);
}

TEST(Trails, AShortPoolTakesTheTilesFedLongestAgo) {
  std::string text = kTrail;
  text.replace(text.find("tiles 12"), 8, "tiles 3 ");
  sc::Scene scene = scene_of(text);
  std::vector<std::uint8_t> rgb(240 * 120 * 3);
  int most = 0;
  float first_x = 1e9f;
  for (int f = 0; f < 45; ++f) {
    scene.render(f, rgb);
    int on = 0;
    for (const cm::Module* m : scene.modules()) {
      on += m->active ? 1 : 0;
      if (m->active && f == 44) first_x = std::min(first_x, m->at.x);
    }
    most = std::max(most, on);
  }
  EXPECT_EQ(most, 3);
  EXPECT_GE(first_x, 96.f) << "the tiles at the start of the path were taken for its end";
}

TEST(Trails, TheSameFramesOnAnyNumberOfThreads) {
  const auto frames = [](int threads, bool overlap) {
    sc::Scene scene = scene_of(kTrail, threads, overlap);
    std::vector<std::uint8_t> rgb(240 * 120 * 3), all;
    for (int f = 0; f < 50; ++f) {
      scene.render(f, rgb);
      all.insert(all.end(), rgb.begin(), rgb.end());
    }
    return all;
  };
  const auto a = frames(1, false);
  EXPECT_EQ(a, frames(2, false));
  EXPECT_EQ(a, frames(3, true));
}

TEST(Trails, TheSourcesVelocityAndWhatTheCheckerSays) {
  sc::Scene scene = scene_of(std::string(kTrail) + "let lean = streak.vx\n");
  std::vector<std::uint8_t> rgb(240 * 120 * 3);
  for (int f = 0; f < 20; ++f) scene.render(f, rgb);
  const auto n = static_cast<int>(scene.modules().size());
  EXPECT_EQ(n, 12);
  EXPECT_NE(error_of(std::string(kTrail) + "module m = tiny, size 32, at (10, 10)\nlet v = m.vx\nfield w = wind, velocity (v, 0), on m\n").find("only a trail has a velocity"),
            std::string::npos);
  EXPECT_NE(error_of(std::string(kTrail) + "at 1: start streak\n").find("a trail"), std::string::npos);
  EXPECT_NE(error_of("effect tiny = \"tiny\"\ntrail w = tiny, size 32, band 8, at (0, 0)\n").find("less than half"), std::string::npos);
  EXPECT_NE(error_of("effect tiny = \"tiny\"\ntrail w = tiny, size 32, at (0, 0), tiles 65\n").find("1 to 64"), std::string::npos);
  EXPECT_NE(error_of("effect tiny = \"tiny\"\ntrail w = tiny, size 32\n").find("needs its source"), std::string::npos);
  EXPECT_NE(error_of("effect tiny = \"tiny\"\ntrail w = tiny, size 32, band 4, at (0, 0), intensty 0.5\n").find("did you mean 'intensity'"), std::string::npos);
  // a module can still be called trail (the word is not reserved)
  EXPECT_EQ(error_of("effect tiny = \"tiny\"\nmodule trail = tiny, size 32, at (40, 40)\n"), "no error");
}

TEST(Trails, AStagesInstanceMovedEveryFrameLeavesItsTrailBehind) {
  const EffectPtr tiny = tiny_nvfx(3);
  const char* catalogue = R"(
effect tiny = "tiny.nvfx"
look gas = shader, heat_scale 1, emission 2
entry streak, category trails, size 160 x 60, anchor centre, loop, moving (120, 0), about "a moving source"
param density = 0.5, range (0, 1), about "how much"
trail smoke = tiny, size 32, width 64, band 4, tiles 10, at (X, Y), look gas, hold 0.5, fade 0.5, soot density, radius 6 * scale
)";
  const nvfx_library_file file{"t.nvfxl", catalogue};
  nvfx_library_desc ld;
  nvfx_library_desc_init(&ld);
  ld.files = &file;
  ld.n_files = 1;
  nvfx_library* lib = nullptr;
  nvfx_scene_error err;
  ASSERT_EQ(nvfx_library_open(&ld, &lib, &err), NVFX_OK) << err.message;
  nvfx_library_effect e{};
  ASSERT_EQ(nvfx_library_get_effect(lib, 0, &e), NVFX_OK);
  EXPECT_FLOAT_EQ(e.move_x, 120.f);
  EXPECT_FLOAT_EQ(e.move_y, 0.f);
  EXPECT_STREQ(e.elements, "trail, look gas");
  const nvfx_scene_effect fx{"tiny.nvfx", tiny.get()};
  const nvfx_stage_room room{"streak", 2, 0};
  nvfx_stage_desc sd;
  nvfx_stage_desc_init(&sd);
  sd.width = 240;
  sd.height = 120;
  sd.ground = 110;
  sd.rooms = &room;
  sd.n_rooms = 1;
  sd.effects = &fx;
  sd.n_effects = 1;
  nvfx_stage* st = nullptr;
  ASSERT_EQ(nvfx_stage_create(lib, &sd, &st, &err), NVFX_OK) << err.message;
  nvfx_scene* scene = nvfx_stage_scene(st);
  nvfx_fx a = 0;
  ASSERT_EQ(nvfx_stage_spawn(st, "streak", 30.f, 60.f, 1.f, -1.0, &a), NVFX_OK);
  for (int f = 1; f <= 30; ++f) {
    nvfx_stage_move(st, a, 30.f + 5.f * static_cast<float>(f), 60.f);
    nvfx_scene_step_frames(scene, 1);
  }
  nvfx_stage_fx_info info{};
  ASSERT_EQ(nvfx_stage_get_fx(st, a, &info), NVFX_OK);
  EXPECT_FLOAT_EQ(info.x, 180.f);
  float max = 0.f;
  ASSERT_EQ(nvfx_scene_field_region(scene, NVFX_FIELD_SOOT, 40.f, 40.f, 90.f, 80.f, &max, nullptr), NVFX_OK);
  EXPECT_GT(max, 0.01f) << "soot behind it, where it passed";
  // stopped: the trail stays and fades by itself
  nvfx_stage_stop(st, a, -1.0);
  nvfx_scene_step_frames(scene, 2);
  ASSERT_EQ(nvfx_scene_field_region(scene, NVFX_FIELD_SOOT, 40.f, 40.f, 200.f, 80.f, &max, nullptr), NVFX_OK);
  EXPECT_GT(max, 0.f) << "its trail outlives it for a moment";
  nvfx_stage_free(st);
  nvfx_library_free(lib);
}

TEST(Trails, AMovedInstanceGivesTheSameFramesOnAnyNumberOfThreads) {
  // Moves are inputs: they act from the next frame computed, and with overlap (a frame's state computed while the last
  // picture is drawn) from the frame after (docs/ENGINES.md §7). For one overlap setting, the frames do not depend on
  // the number of threads.
  const EffectPtr tiny = tiny_nvfx(3);
  const char* catalogue = R"(
effect tiny = "tiny.nvfx"
look gas = shader, heat_scale 1, emission 2, relief 2
entry streak, category trails, size 160 x 60, anchor centre, loop, moving (120, 0), about "a moving source"
let lean = clamp(smoke.vx / 400, -0.3, 0.3)
module head = tiny, size 32, width 40 * scale, at (X, Y + 20 * scale), start 1, seed 7, wind 0.5 - lean
trail smoke = tiny, size 32, width 64, band 4, tiles 6, at (X, Y), look gas, hold 0.3, fade 0.3, soot 0.5, heat 0.1, radius 6,
      kick -0.2
at 0, repeat: burst at (X, Y), embers 0, sparks 1, speed 100, direction (-smoke.vx, 0.01), cone 30
)";
  const nvfx_library_file file{"t.nvfxl", catalogue};
  nvfx_library_desc ld;
  nvfx_library_desc_init(&ld);
  ld.files = &file;
  ld.n_files = 1;
  nvfx_library* lib = nullptr;
  nvfx_scene_error err;
  ASSERT_EQ(nvfx_library_open(&ld, &lib, &err), NVFX_OK) << err.message;
  const auto frames = [&](int threads, int overlap) {
    const nvfx_scene_effect fx{"tiny.nvfx", tiny.get()};
    const nvfx_stage_room room{"streak", 1, 0};
    nvfx_stage_desc sd;
    nvfx_stage_desc_init(&sd);
    sd.width = 240;
    sd.height = 120;
    sd.ground = 110;
    sd.rooms = &room;
    sd.n_rooms = 1;
    sd.effects = &fx;
    sd.n_effects = 1;
    sd.threads = threads;
    sd.overlap = overlap;
    nvfx_stage* st = nullptr;
    EXPECT_EQ(nvfx_stage_create(lib, &sd, &st, &err), NVFX_OK) << err.message;
    nvfx_scene* scene = nvfx_stage_scene(st);
    std::vector<std::uint8_t> rgba(240 * 120 * 4), all;
    nvfx_fx a = 0;
    nvfx_stage_spawn(st, "streak", 20.f, 60.f, 1.f, -1.0, &a);
    for (int f = 1; f <= 45; ++f) {
      nvfx_stage_move(st, a, 20.f + 4.f * static_cast<float>(f), 60.f + 10.f * std::sin(0.2f * static_cast<float>(f)));
      nvfx_scene_step_frames(scene, 1);
      nvfx_scene_render(scene, rgba.data(), 240 * 4);
      all.insert(all.end(), rgba.begin(), rgba.end());
    }
    nvfx_stage_free(st);
    return all;
  };
  const auto one = frames(1, 0);
  EXPECT_EQ(one, frames(2, 0));
  EXPECT_EQ(one, frames(3, 0));
  const auto two = frames(2, 1);
  EXPECT_EQ(two, frames(3, 1));
  EXPECT_EQ(two, frames(4, 1));
  nvfx_library_free(lib);
}

// --- engine layers -----------------------------------------------------------------------------------------------------

namespace {

// Everything a scene draws (modules of both looks, particles of every kind, glows), and weather that covers pixels.
std::string everything_with_weather() {
  return std::string(cm::testing::kEverything) + "emit snow, in (0, 0), size 160 x 80, density 6, radius 1.5, seed 3\n" +
         "emit ash, in (0, 0), size 160 x 80, density 3, glowing 0.5, seed 4\nemit leaves, in (0, 0), size 160 x 80, density 1, seed 5\n" +
         "emit rain, in (0, 0), size 160 x 80, density 4, seed 6\n";
}

struct Layers {
  std::vector<std::uint8_t> rgb;
  std::array<std::vector<float>, 4> layer;
};

std::vector<Layers> play_layers(const std::string& text, int threads, bool overlap, bool layers, int frames) {
  sc::Scene scene = scene_of(text, threads, overlap, layers);
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

TEST(Layers, ThePictureIsTheSameWithLayersOrWithout) {
  const std::string text = everything_with_weather();
  for (const bool overlap : {false, true}) {
    const auto off = play_layers(text, 2, overlap, false, 40);
    const auto on = play_layers(text, 2, overlap, true, 40);
    for (std::size_t f = 0; f < off.size(); ++f) ASSERT_EQ(off[f].rgb, on[f].rgb) << "frame " << f << (overlap ? ", overlapped" : "");
  }
}

TEST(Layers, TheSameLayersOnAnyNumberOfThreads) {
  const std::string text = everything_with_weather();
  const auto a = play_layers(text, 1, false, true, 40);
  const auto b = play_layers(text, 3, true, true, 40);
  for (std::size_t f = 0; f < a.size(); ++f)
    for (std::size_t k = 0; k < 4; ++k) ASSERT_EQ(a[f].layer[k], b[f].layer[k]) << "frame " << f << ", layer " << k;
}

TEST(Layers, TheyAgreeWithEachOther) {
  const auto frames = play_layers(everything_with_weather(), 2, false, true, 40);
  int covered = 0, lit = 0, emitted = 0;
  for (const Layers& l : frames) {
    const auto& C = l.layer[0];
    const auto& A = l.layer[1];
    const auto& E = l.layer[2];
    const auto& N = l.layer[3];
    for (std::size_t i = 0; i < C.size(); i += 4) {
      const float a = C[i + 3], thick = N[i + 3];
      ASSERT_EQ(a == 0.f, thick == 0.f) << "thickness is 0 exactly where alpha is 0 (pixel " << i / 4 << ")";
      ASSERT_EQ(A[i + 3], a) << "the albedo's alpha is the colour's";
      ASSERT_GE(a, 0.f);
      ASSERT_LE(a, 1.f);
      ASSERT_GE(thick, 0.f);
      ASSERT_EQ(E[i + 3], 0.f);
      for (int c = 0; c < 3; ++c) {
        ASSERT_GE(E[i + static_cast<std::size_t>(c)], 0.f);
        ASSERT_LE(A[i + static_cast<std::size_t>(c)], a + 1e-5f) << "albedo is a reflectance at most 1, premultiplied";
      }
      // the normal times alpha: no longer than alpha (a mix of unit vectors weighted by coverage), facing the viewer
      const float n = std::sqrt(N[i] * N[i] + N[i + 1] * N[i + 1] + N[i + 2] * N[i + 2]);
      ASSERT_LE(n, a + 1e-4f);
      ASSERT_GE(N[i + 2], 0.f);
      covered += a > 0.f;
      lit += C[i] > 0.f;
      emitted += E[i] > 0.f;
    }
  }
  EXPECT_GT(covered, 1000);
  EXPECT_GT(emitted, 100) << "embers, sparks, glows and hot gas give light";
  EXPECT_GT(lit, covered / 2);
}

TEST(Layers, TheCApiEncodesThemForEngines) {
  const EffectPtr tiny = tiny_nvfx(3), other = tiny_nvfx(4);
  const nvfx_scene_effect list[] = {{"tiny", tiny.get()}, {"other", other.get()}};
  const std::string text = everything_with_weather();
  const auto make = [&](int layers, int w, int h) {
    nvfx_scene_desc d;
    nvfx_scene_desc_init(&d);
    d.script = text.c_str();
    d.effects = list;
    d.n_effects = 2;
    d.layers = layers;
    d.width = w;
    d.height = h;
    nvfx_scene* s = nullptr;
    nvfx_scene_error err;
    EXPECT_EQ(nvfx_scene_create(&d, &s, &err), NVFX_OK) << err.message;
    return s;
  };
  nvfx_scene* plain = make(0, 0, 0);
  nvfx_scene* s = make(1, 0, 0);
  nvfx_scene_info info{};
  nvfx_scene_get_info(s, &info);
  EXPECT_EQ(info.layers, 1);
  const int W = info.width, H = info.height;
  std::vector<float> f32(static_cast<std::size_t>(W) * static_cast<std::size_t>(H) * 4);
  std::vector<std::uint16_t> f16(f32.size());
  std::vector<std::uint8_t> u8(f32.size()), rgba(f32.size()), rgba_plain(f32.size());
  EXPECT_EQ(nvfx_scene_render_layer(plain, NVFX_LAYER_COLOUR, NVFX_FORMAT_RGBA32F, f32.data(), static_cast<std::size_t>(W) * 16), NVFX_ERROR_ARGUMENT)
      << "no layers without the option";
  EXPECT_EQ(nvfx_scene_render_layer(s, NVFX_LAYER_COLOUR, NVFX_FORMAT_RGBA32F, f32.data(), static_cast<std::size_t>(W) * 8), NVFX_ERROR_ARGUMENT)
      << "a stride too short";
  EXPECT_EQ(nvfx_pixel_format_bytes(NVFX_FORMAT_RGBA8), 4u);
  EXPECT_EQ(nvfx_pixel_format_bytes(NVFX_FORMAT_RGBA16F), 8u);
  EXPECT_EQ(nvfx_pixel_format_bytes(NVFX_FORMAT_RGBA32F), 16u);
  const auto half = [](std::uint16_t h) {  // binary16 to float
    const int e = (h >> 10) & 0x1f, m = h & 0x3ff;
    const float v = e == 0 ? std::ldexp(static_cast<float>(m), -24) : std::ldexp(1.f + static_cast<float>(m) / 1024.f, e - 15);
    return (h & 0x8000) ? -v : v;
  };
  for (int f = 0; f < 30; ++f) {
    nvfx_scene_step_frames(s, 1);
    nvfx_scene_step_frames(plain, 1);
    nvfx_scene_render(plain, rgba_plain.data(), static_cast<std::size_t>(W) * 4);
    // the layers first: they draw the frame, and the picture is then the same as without them
    for (int l = 0; l < 4; ++l) {
      const auto layer = static_cast<nvfx_scene_layer>(l);
      ASSERT_EQ(nvfx_scene_render_layer(s, layer, NVFX_FORMAT_RGBA32F, f32.data(), static_cast<std::size_t>(W) * 16), NVFX_OK);
      ASSERT_EQ(nvfx_scene_render_layer(s, layer, NVFX_FORMAT_RGBA16F, f16.data(), static_cast<std::size_t>(W) * 8), NVFX_OK);
      ASSERT_EQ(nvfx_scene_render_layer(s, layer, NVFX_FORMAT_RGBA8, u8.data(), static_cast<std::size_t>(W) * 4), NVFX_OK);
      for (std::size_t i = 0; i < f32.size(); ++i) {
        ASSERT_NEAR(half(f16[i]), f32[i], 1e-3f + 1e-3f * std::abs(f32[i])) << "layer " << l << ", value " << i;
      }
      for (std::size_t i = 0; i < f32.size(); i += 4) {
        const float* v = f32.data() + i;
        const std::uint8_t* b = u8.data() + i;
        if (layer == NVFX_LAYER_NORMAL) {
          ASSERT_NEAR(std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), 1.f, 1e-4f) << "a unit normal";
          if (v[3] == 0.f) {
            ASSERT_EQ(b[0], 128);
            ASSERT_EQ(b[1], 128);
            ASSERT_EQ(b[2], 255);
            ASSERT_EQ(b[3], 0);
          }
          ASSERT_NEAR(static_cast<float>(b[3]) / 255.f, v[3] / (1.f + v[3]), 0.5f / 255.f + 1e-6f);
        } else if (layer == NVFX_LAYER_EMISSION) {  // RGBM
          for (int c = 0; c < 3; ++c) ASSERT_NEAR(static_cast<float>(b[c]) / 255.f * static_cast<float>(b[3]) / 255.f * 8.f, std::min(v[c], 8.f), 0.04f);
        } else {
          ASSERT_EQ(b[3], static_cast<std::uint8_t>(std::lround(255.f * std::clamp(v[3], 0.f, 1.f))));
          for (int c = 0; c < 3; ++c) ASSERT_NEAR(std::pow(static_cast<float>(b[c]) / 255.f, 2.2f), std::clamp(v[c], 0.f, 1.f), 0.02f);
        }
      }
    }
    ASSERT_EQ(nvfx_scene_render(s, rgba.data(), static_cast<std::size_t>(W) * 4), NVFX_OK);
    ASSERT_EQ(rgba, rgba_plain) << "frame " << f;
  }
  // at another output size the layers are resampled as the picture is
  nvfx_scene* small = make(1, 80, 45);
  std::vector<float> s32(80 * 45 * 4);
  nvfx_scene_step_frames(small, 10);
  EXPECT_EQ(nvfx_scene_render_layer(small, NVFX_LAYER_NORMAL, NVFX_FORMAT_RGBA32F, s32.data(), 80 * 16), NVFX_OK);
  nvfx_scene_free(small);
  nvfx_scene_free(s);
  nvfx_scene_free(plain);
}

TEST(Layers, AShaderLooksAlbedoAndEmissionAreItsOwn) {
  // one module of a shader look, alone: where it covers, its albedo is soot_albedo * tint times alpha, and its emission
  // the shader's heat ramp
  const char* text = R"(
scene size 96 x 96, fps 30, length 1 s, ground 95
effect tiny = "tiny"
look gas = shader, heat_scale 1, emission 2, soot_albedo 0.5, tint (1, 0.8, 0.6), relief 3
module m = tiny, size 32, width 64, at (48, 80), look gas, start 1, seed 4
)";
  const auto frames = play_layers(text, 1, false, true, 8);
  const Layers& l = frames.back();
  int checked = 0, glowing = 0, tilted = 0;
  for (std::size_t i = 0; i < l.layer[0].size(); i += 4) {
    const float a = l.layer[0][i + 3];
    if (a <= 0.f) continue;
    EXPECT_NEAR(l.layer[1][i], 0.5f * a, 1e-4f);
    EXPECT_NEAR(l.layer[1][i + 1], 0.4f * a, 1e-4f);
    EXPECT_NEAR(l.layer[1][i + 2], 0.3f * a, 1e-4f);
    glowing += l.layer[2][i] > 0.f;
    tilted += std::abs(l.layer[3][i]) > 0.05f * a;
    ++checked;
  }
  EXPECT_GT(checked, 200);
  EXPECT_GT(glowing, 100);
  EXPECT_GT(tilted, 20) << "the relief tilts the normals";
}

// --- weather -----------------------------------------------------------------------------------------------------------

TEST(Weather, ItFillsItsRegionAndFallsThroughIt) {
  sc::Scene scene = scene_of("scene size 200 x 100, fps 30, length 2 s, ground 100\nparticles capacity 8192\n"
                             "emit snow, in (0, 0), size 200 x 100, density 50, wind (40, 0), seed 3\n");
  std::vector<std::uint8_t> rgb(200 * 100 * 3);
  scene.render(0, rgb);
  EXPECT_NEAR(static_cast<float>(scene.particles().alive()), 50.f * (200.f + 40.f * 100.f / 55.f) * 100.f / 10000.f, 4.f)
      << "the region is filled at once";
  float low = 0.f;
  scene.particles().each([&](float, float y, float&, float&) { low = std::max(low, y); });
  for (int f = 1; f < 45; ++f) scene.render(f, rgb);
  const int n = scene.particles().alive();
  EXPECT_GT(n, 100);
  EXPECT_LT(n, 180) << "fed from the top as fast as it leaves";
  scene.particles().each([&](float x, float y, float& vx, float& vy) {
    EXPECT_GE(y, -10.f);
    EXPECT_LE(y, 100.f);
    EXPECT_GT(x, -120.f);
    EXPECT_GT(vy, 0.f) << "falling";
    (void)vx;
  });
}

TEST(Weather, ItKeepsTheScenesRandomNumbers) {
  const auto run = [](const std::string& extra) {
    sc::Scene scene = scene_of("scene size 160 x 90, fps 30, length 1 s, ground 80\nparticles capacity 4096\n"
                               "emit sparks, along (20, 70) to (20, 70), count 3\n" +
                               extra);
    std::vector<std::uint8_t> rgb(160 * 90 * 3);
    for (int f = 0; f < 20; ++f) scene.render(f, rgb);
    return std::pair{scene.particles().alive(), scene.particles().rng};
  };
  const auto [n0, rng0] = run("");
  const auto [n1, rng1] = run("emit rain, in (0, 0), size 160 x 80, density 30\nemit leaves, in (0, 0), size 160 x 80, density 10\n");
  EXPECT_EQ(rng0, rng1);
  EXPECT_GT(n1, n0 + 20);
}

TEST(Weather, RainWashesSmokeOutAndSnowMeltsOverHeat) {
  const char* base = R"(
scene size 120 x 100, fps 30, length 2 s, ground 95
effect tiny = "tiny"
module m = tiny, size 32, width 64, at (60, 90), start 1, seed 4
)";
  const auto soot = [&](const std::string& extra) {
    sc::Scene scene = scene_of(std::string(base) + extra);
    std::vector<std::uint8_t> rgb(120 * 100 * 3);
    for (int f = 0; f < 20; ++f) scene.render(f, rgb);
    double sum = 0;
    for (const float v : scene.module("m")->runner().fine_soot()) sum += v;
    return sum;
  };
  const double dry = soot(""), wet = soot("emit rain, in (0, 0), size 120 x 100, density 4, wash 2\n"), drizzle = soot("emit rain, in (0, 0), size 120 x 100, density 4\n");
  EXPECT_LT(wet, 0.7 * dry) << "rain with wash 2 thins the smoke";
  EXPECT_DOUBLE_EQ(drizzle, dry) << "without wash, rain leaves the modules alone";
  // snow over a heat source melts: none left where the bus is hot
  sc::Scene scene = scene_of("scene size 120 x 100, fps 30, length 2 s, ground 95\nparticles capacity 4096\neffect tiny = \"tiny\"\n"
                             "module m = tiny, size 32, width 64, at (60, 90), start 1, seed 4\n"
                             "field hot = heat, at (60, 60), radius 25, strength 0.5, on m\n"
                             "emit snow, in (0, 0), size 120 x 100, density 60, melt 0.05, seed 2\n");
  std::vector<std::uint8_t> rgb(120 * 100 * 3);
  for (int f = 0; f < 30; ++f) scene.render(f, rgb);
  int over_heat = 0, total = 0;
  scene.particles().each([&](float x, float y, float&, float&) {
    ++total;
    over_heat += scene.bus().at(x, y).heat > 0.05f ? 1 : 0;
  });
  EXPECT_GT(total, 20);
  EXPECT_LE(over_heat, total / 10) << "flakes melt where the bus is hot (those that just arrived may still be there)";
}

TEST(Weather, ItRidesTheWindAndTheFieldsOnParticles) {
  const auto mean_x = [](const std::string& extra) {
    sc::Scene scene = scene_of("scene size 200 x 100, fps 30, length 2 s, ground 100\nparticles capacity 4096\n"
                               "emit leaves, in (0, 0), size 200 x 100, density 20, seed 3\n" +
                               extra);
    std::vector<std::uint8_t> rgb(200 * 100 * 3);
    for (int f = 0; f < 30; ++f) scene.render(f, rgb);
    double sum = 0;
    int n = 0;
    scene.particles().each([&](float, float, float& vx, float&) {
      sum += vx;
      ++n;
    });
    return sum / std::max(1, n);
  };
  EXPECT_GT(mean_x("field breeze = wind, velocity (5, 0), on particles\n"), mean_x("") + 30.0) << "a wind field carries the leaves";
}
