// The C++ wrapper (include/neuralfx/nvfx.hpp): every wrapped call against the C call it wraps, on the tests' small
// rollout effects (no data needed): the same results, the same pixels, the C API's errors with their messages and
// positions, spans checked against the picture, handles that free and move; and no heap allocation in a game's frame
// through the wrapper (this program replaces operator new to count them, as nvfx_alloc_test does).
#include "compose_scene.hpp"
#include "pack.hpp"

#include <neuralfx/nvfx.hpp>
#include <neuralfx/rollout.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <new>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

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

namespace {

namespace fs = std::filesystem;
namespace pk = nfx::pack;
using Bytes = std::vector<std::uint8_t>;

Bytes bytes_of(std::string_view s) { return {s.begin(), s.end()}; }

Bytes tiny_model(std::uint64_t seed) {
  const nfx::rt::RolloutEffect t = nfx::compose::testing::tiny_effect(seed);
  std::ostringstream os;
  EXPECT_TRUE(nfx::rollout::save_model(os, t.m));
  return bytes_of(os.str());
}

// The wrapper's result is the C call's status.
template <class T>
void same(const nvfx::Result<T>& r, nvfx_status c, const char* what) {
  EXPECT_EQ(r.has_value(), c == NVFX_OK) << what;
  if (!r) {
    EXPECT_EQ(r.error().status, c) << what;
    EXPECT_EQ(r.error().message, nvfx_status_string(c)) << what;
  }
}

constexpr const char* kCatalogue = R"(
effect tiny = "tiny.nvfx"
look hot = shader, heat_scale 1, emission 2

entry blob, category fire, size 64 x 64, anchor base, loop, show 0.5, about "a tiny looping effect"
param intensity = 0.5, range (0, 1), about "its control"
module m = tiny, size 32, width 48 * scale, at (X, Y), look hot, start 1, seed 7, controls (0.5, 0.5, 0.5),
           intensity intensity
emit embers, on m, tries 1, chance 0.5, spread 10

entry zap, category sparks, size 80 x 40, anchor left, loop, about "a procedural arc"
emit arc, along (X - 30, Y - 10) to (X + 30, Y - 10), rate 10, seed 2
)";

constexpr const char* kScript = R"(
scene size 160 x 90, fps 30, length 2, ground 80
effect tiny = "tiny.nvfx"
input spot = 40
module fire = tiny, size 32, at (spot, ground), start 1, seed 5
module torch = tiny, size 32, at (120, ground), start 0, seed 6
when 0 as boom, repeat:
  shock at (80, 50), speed 600
)";

Bytes tiny_pack() {
  const std::vector<pk::Input> members = {{"tiny.nvfx", pk::Kind::model, tiny_model(3)},
                                          {"tiny.nvfxl", pk::Kind::catalogue, bytes_of(kCatalogue)},
                                          {"camp.nvfxs", pk::Kind::script, bytes_of(kScript)},
                                          {"CREDITS.txt", pk::Kind::licence, bytes_of("credits")}};
  return pk::write(members, {{"blob", {"tiny.nvfx"}}, {"zap", {}}}, {{"tiny.nvfx", 32, 0.3f, 0.06f}}, {}).file;
}

struct TempFile {
  fs::path path;
  explicit TempFile(std::span<const std::uint8_t> b, const char* ext) {
    std::random_device rd;
    path = fs::temp_directory_path() / std::format("nvfx_wrapper_{:08x}{}", rd(), ext);
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
  }
  ~TempFile() {
    std::error_code ec;
    fs::remove(path, ec);
  }
};

}  // namespace

TEST(Wrapper, VersionAndTheCpu) {
  const nvfx::Version v = nvfx::version();
  std::uint32_t f = 0;
  EXPECT_EQ(v.version, nvfx_version(&f));
  EXPECT_EQ(v.features, f);
  EXPECT_EQ(nvfx_version(nullptr), static_cast<std::uint32_t>(NVFX_VERSION));
  EXPECT_TRUE(v.matches_headers());
  EXPECT_TRUE(v.has(NVFX_FEATURE_SCENE | NVFX_FEATURE_LIBRARY | NVFX_FEATURE_MANAGER | NVFX_FEATURE_PACK));
  EXPECT_FALSE(v.has(NVFX_FEATURE_SHARED)) << "this program links the static libraries";
  EXPECT_EQ(v.has(NVFX_FEATURE_AVX2), nvfx_get_isa() == NVFX_ISA_AVX2);
  EXPECT_EQ(nvfx::get_isa(), nvfx_get_isa());
  EXPECT_TRUE(nvfx::set_isa(NVFX_ISA_AUTO));
  if (nvfx_set_isa(NVFX_ISA_AVX512) == NVFX_OK) {  // a CPU with AVX-512: forcing works through the wrapper too
    EXPECT_TRUE(nvfx::set_isa(NVFX_ISA_AVX512));
    EXPECT_EQ(nvfx::get_isa(), NVFX_ISA_AVX512);
  } else {
    same(nvfx::set_isa(NVFX_ISA_AVX512), NVFX_ERROR_UNSUPPORTED, "set_isa");
  }
  EXPECT_TRUE(nvfx::set_isa(NVFX_ISA_AUTO));
  EXPECT_STREQ(nvfx::status_string(NVFX_ERROR_IO), nvfx_status_string(NVFX_ERROR_IO));
}

TEST(Wrapper, EffectsAndInstancesAreTheCApis) {
  const Bytes model = tiny_model(3);
  const TempFile file(model, ".nvfx");
  auto loaded = nvfx::Effect::load(file.path.string());
  ASSERT_TRUE(loaded) << loaded.error().message;
  auto effect = nvfx::Effect::load_memory(model);
  ASSERT_TRUE(effect);
  same(nvfx::Effect::load("/nonexistent/x.nvfx"), NVFX_ERROR_IO, "load");
  same(nvfx::Effect::load_memory(bytes_of("not a model")), NVFX_ERROR_FORMAT, "load_memory");
  const auto info = effect->get_info();
  ASSERT_TRUE(info);
  nvfx_effect_info ci;
  nvfx_effect_get_info(loaded->get(), &ci);
  EXPECT_EQ(info->arch, 3);
  EXPECT_EQ(info->arch, ci.arch);
  EXPECT_EQ(info->resident_bytes, ci.resident_bytes);
  same(nvfx::Effect().get_info(), NVFX_ERROR_ARGUMENT, "get_info of none");
  EXPECT_STREQ(effect->control_name(0), "intensity");
  EXPECT_EQ(effect->control_name(3), nullptr);
  same(loaded->attach_prior("/nonexistent/fire.ddpm"), NVFX_ERROR_IO, "attach_prior");
  same(loaded->attach_prior_memory(bytes_of("no denoiser")), NVFX_ERROR_FORMAT, "attach_prior_memory");

  // an instance through the wrapper and one through the C API: the same pixels for the same calls
  auto inst = nvfx::Instance::create(*effect, 64);
  ASSERT_TRUE(inst);
  same(nvfx::Instance::create(*effect, 50), NVFX_ERROR_UNSUPPORTED, "create at a size off the grid");
  nvfx_instance* c = nullptr;
  ASSERT_EQ(nvfx_instance_create(effect->get(), 64, &c), NVFX_OK);
  EXPECT_EQ(inst->size(), 64);
  EXPECT_EQ(inst->frame_bytes(), 64u * 64u * 4u);
  EXPECT_EQ(inst->scratch_bytes(), nvfx_instance_scratch_bytes(c));
  EXPECT_EQ(inst->macs_per_pixel(), nvfx_instance_macs_per_pixel(c));
  const float controls[3] = {0.7f, 0.4f, 0.6f};
  same(inst->set_controls(controls), nvfx_instance_set_controls(c, controls, 3), "set_controls");
  same(inst->set_seed(99), nvfx_instance_set_seed(c, 99), "set_seed");
  same(inst->set_drift(2.f), nvfx_instance_set_drift(c, 2.f), "set_drift");
  same(inst->set_colour(0.3f, 1.2f), nvfx_instance_set_colour(c, 0.3f, 1.2f), "set_colour");
  same(inst->set_precision(NVFX_PRECISION_INT8), nvfx_instance_set_precision(c, NVFX_PRECISION_INT8), "set_precision int8");
  same(inst->set_precision(NVFX_PRECISION_FLOAT), nvfx_instance_set_precision(c, NVFX_PRECISION_FLOAT), "set_precision float");
  same(inst->set_prior(16, 100, 1.f), nvfx_instance_set_prior(c, 16, 100, 1.f), "set_prior");
  same(inst->set_handoff(4, 2), nvfx_instance_set_handoff(c, 4, 2), "set_handoff");
  same(inst->set_handoff(0, 0), nvfx_instance_set_handoff(c, 0, 0), "set_handoff off");
  Bytes a(inst->frame_bytes()), b(a.size());
  for (const double t : {0.0, 0.1, 0.5, 1.3}) {
    ASSERT_TRUE(inst->render(t, a));
    ASSERT_EQ(nvfx_render(c, t, b.data(), 64 * 4), NVFX_OK);
    EXPECT_EQ(a, b) << "t " << t;
  }
  same(inst->set_variation(1), nvfx_instance_set_variation(c, 1), "set_variation");
  ASSERT_TRUE(inst->render(0.4, a));
  ASSERT_EQ(nvfx_render(c, 0.4, b.data(), 64 * 4), NVFX_OK);
  EXPECT_EQ(a, b);
  Bytes wide(64 * 300 + 64 * 4);  // rows 300 bytes apart
  ASSERT_TRUE(inst->render(0.5, wide, 300));
  ASSERT_EQ(nvfx_render(c, 0.5, b.data(), 64 * 4), NVFX_OK);
  for (int y = 0; y < 64; ++y) EXPECT_TRUE(std::equal(b.begin() + y * 256, b.begin() + y * 256 + 256, wide.begin() + y * 300)) << y;
  same(inst->render(0.5, std::span(a).first(a.size() - 1)), NVFX_ERROR_ARGUMENT, "render into too small a buffer");
  same(inst->render(0.5, a, 100), NVFX_ERROR_ARGUMENT, "render with a stride under a row");
  Bytes flip(inst->frame_bytes() * 3), cflip(flip.size());
  ASSERT_TRUE(inst->bake(3, flip));
  ASSERT_EQ(nvfx_bake(c, 3, cflip.data()), NVFX_OK);
  EXPECT_EQ(flip, cflip);
  same(inst->bake(4, flip), NVFX_ERROR_ARGUMENT, "bake into too small a buffer");
  nvfx_instance_free(c);

  // handles: move, release, reset
  nvfx::Instance moved = std::move(*inst);
  EXPECT_FALSE(*inst);
  EXPECT_TRUE(moved);
  EXPECT_EQ(moved.size(), 64);
  nvfx_instance* raw = moved.release();
  EXPECT_FALSE(moved);
  nvfx::Instance adopted(raw, 64);
  EXPECT_TRUE(adopted.render(0.2, a));
  adopted.reset();
  EXPECT_FALSE(adopted);
  nvfx::Effect e2 = std::move(*effect);
  EXPECT_FALSE(*effect);
  *effect = std::move(e2);
  EXPECT_TRUE(*effect);
}

TEST(Wrapper, ScenesAreTheCApis) {
  auto tiny = nvfx::Effect::load_memory(tiny_model(3));
  ASSERT_TRUE(tiny);
  const nvfx_scene_effect fx{"tiny.nvfx", tiny->get()};
  nvfx_scene_desc d = nvfx::scene_desc();
  nvfx_scene_desc cd;
  nvfx_scene_desc_init(&cd);
  EXPECT_EQ(d.threads, cd.threads);
  EXPECT_EQ(d.overlap, cd.overlap);
  d.script = kScript;
  d.effects = &fx;
  d.n_effects = 1;
  d.threads = 1;
  auto scene = nvfx::Scene::create(d);
  ASSERT_TRUE(scene) << scene.error().message;
  nvfx_scene* c = nullptr;
  ASSERT_EQ(nvfx_scene_create(&d, &c, nullptr), NVFX_OK);
  // errors: the C API's message, line and column
  nvfx_scene_desc bad = d;
  bad.script = "scene size 160 x 90\nmodule m = tiny, size 32, at (40, 80), strat 0\n";
  bad.source_name = "bad.nvfxs";
  const auto err = nvfx::Scene::create(bad);
  ASSERT_FALSE(err);
  nvfx_scene_error ce;
  nvfx_scene* none = nullptr;
  EXPECT_EQ(err.error().status, nvfx_scene_create(&bad, &none, &ce));
  EXPECT_EQ(err.error().status, NVFX_ERROR_SCRIPT);
  EXPECT_EQ(err.error().message, ce.message);
  EXPECT_EQ(err.error().line, ce.line);
  EXPECT_EQ(err.error().column, ce.column);
  EXPECT_EQ(err.error().line, 2);
  EXPECT_TRUE(nvfx::Scene::check(kScript, "camp.nvfxs"));
  const auto chk = nvfx::Scene::check("scene size 160 x 90\nmodul m\n", "c.nvfxs");
  ASSERT_FALSE(chk);
  EXPECT_EQ(chk.error().line, 2);
  EXPECT_EQ(chk.error().message.rfind("c.nvfxs:2:", 0), 0u) << chk.error().message;
  std::vector<std::string> listed;
  EXPECT_TRUE(nvfx::Scene::list_effects(kScript, "camp.nvfxs", [&](const char* n, const char* f) { listed.push_back(std::string(n) + "=" + f); }));
  EXPECT_EQ(listed, std::vector<std::string>{"tiny=tiny.nvfx"});
  EXPECT_FALSE(nvfx::Scene::list_effects("effect = \n", "x", [](const char*, const char*) {}));
  // info and the clock
  const auto info = scene->get_info();
  ASSERT_TRUE(info);
  nvfx_scene_info ci;
  nvfx_scene_get_info(c, &ci);
  EXPECT_EQ(info->width, ci.width);
  EXPECT_EQ(info->n_modules, ci.n_modules);
  int frames = 0;
  const auto stepped = scene->step(0.1);
  nvfx_scene_step(c, 0.1, &frames);
  ASSERT_TRUE(stepped);
  EXPECT_EQ(*stepped, frames);
  same(scene->step_frames(2), nvfx_scene_step_frames(c, 2), "step_frames");
  EXPECT_EQ(scene->time(), nvfx_scene_time(c));
  EXPECT_EQ(scene->frame(), nvfx_scene_frame(c));
  same(scene->step(-1.0), nvfx_scene_step(c, -1.0, nullptr), "step backwards");
  // run-time settings
  same(scene->set_input("spot", 70.f), nvfx_scene_set_input(c, "spot", 70.f), "set_input");
  same(scene->set_input("nothing", 1.f), nvfx_scene_set_input(c, "nothing", 1.f), "set_input unknown");
  const auto spot = scene->get_input(std::string("spot"));
  ASSERT_TRUE(spot);
  EXPECT_EQ(*spot, 70.f);
  EXPECT_STREQ(scene->input_name(0), nvfx_scene_input_name(c, 0));
  EXPECT_EQ(scene->input_name(5), nullptr);
  same(scene->trigger("boom"), nvfx_scene_trigger(c, "boom"), "trigger");
  same(scene->trigger("nothing"), nvfx_scene_trigger(c, "nothing"), "trigger unknown");
  EXPECT_STREQ(scene->rule_name(0), nvfx_scene_rule_name(c, 0));
  EXPECT_STREQ(scene->module_name(1), nvfx_scene_module_name(c, 1));
  EXPECT_STREQ(scene->module_control_name("fire", 1), nvfx_scene_module_control_name(c, "fire", 1));
  same(scene->module_place("torch", 100.f, 80.f), nvfx_scene_module_place(c, "torch", 100.f, 80.f), "module_place");
  same(scene->module_set_control("fire", "wind", 0.8f), nvfx_scene_module_set_control(c, "fire", "wind", 0.8f), "module_set_control");
  same(scene->module_set_control("fire", "nothing", 0.8f), nvfx_scene_module_set_control(c, "fire", "nothing", 0.8f), "module_set_control unknown");
  // frames, fields and states, against the C API's
  Bytes a(160 * 90 * 4), b(a.size());
  for (int f = 0; f < 12; ++f) {
    ASSERT_TRUE(scene->step_frames(1));
    nvfx_scene_step_frames(c, 1);
    ASSERT_TRUE(scene->render(a));
    nvfx_scene_render(c, b.data(), 160 * 4);
    ASSERT_EQ(a, b) << "frame " << f;
  }
  const auto rs = scene->rule_state("boom");
  int count = 0;
  float last = 0;
  nvfx_scene_rule_state(c, "boom", &count, &last);
  ASSERT_TRUE(rs);
  EXPECT_EQ(rs->count, count);
  EXPECT_EQ(rs->last_time, last);
  EXPECT_EQ(count, 1);
  same(scene->rule_state("nothing"), nvfx_scene_rule_state(c, "nothing", nullptr, nullptr), "rule_state unknown");
  const auto mi = scene->module_get_info("torch");
  nvfx_scene_module_info cmi;
  nvfx_scene_module_get_info(c, "torch", &cmi);
  ASSERT_TRUE(mi);
  EXPECT_EQ(mi->x, cmi.x);
  EXPECT_EQ(mi->x, 100.f);
  same(scene->module_get_info("nothing"), nvfx_scene_module_get_info(c, "nothing", &cmi), "module_get_info unknown");
  const auto s = scene->sample(70.f, 70.f);
  nvfx_scene_fields cs;
  nvfx_scene_sample(c, 70.f, 70.f, &cs);
  ASSERT_TRUE(s);
  EXPECT_EQ(s->heat, cs.heat);
  EXPECT_EQ(s->u, cs.u);
  std::vector<float> grid(5 * 3), cgrid(grid.size());
  ASSERT_TRUE(scene->sample_grid(NVFX_FIELD_HEAT, 10.f, 50.f, 30.f, 10.f, 5, 3, grid));
  nvfx_scene_sample_grid(c, NVFX_FIELD_HEAT, 10.f, 50.f, 30.f, 10.f, 5, 3, cgrid.data(), 5);
  EXPECT_EQ(grid, cgrid);
  same(scene->sample_grid(NVFX_FIELD_HEAT, 10.f, 50.f, 30.f, 10.f, 5, 4, grid), NVFX_ERROR_ARGUMENT, "sample_grid into too small a buffer");
  std::vector<float> strided(7 * 3);
  ASSERT_TRUE(scene->sample_grid(NVFX_FIELD_HEAT, 10.f, 50.f, 30.f, 10.f, 5, 3, strided, 7));
  EXPECT_EQ(strided[7], cgrid[5]);
  const auto region = scene->field_region(NVFX_FIELD_HEAT, 0.f, 0.f, 160.f, 90.f);
  float mx = 0, mean = 0;
  nvfx_scene_field_region(c, NVFX_FIELD_HEAT, 0.f, 0.f, 160.f, 90.f, &mx, &mean);
  ASSERT_TRUE(region);
  EXPECT_EQ(region->max, mx);
  EXPECT_EQ(region->mean, mean);
  EXPECT_GT(mx, 0.f);
  const auto cam = scene->camera();
  float cx = 1, cy = 1;
  nvfx_scene_camera(c, &cx, &cy);
  ASSERT_TRUE(cam);
  EXPECT_EQ((*cam)[0], cx);
  EXPECT_EQ((*cam)[1], cy);
  same(scene->render(std::span(a).first(1000)), NVFX_ERROR_ARGUMENT, "render into too small a buffer");
  // seek and restart
  same(scene->seek(1.0), nvfx_scene_seek(c, 1.0), "seek");
  same(scene->restart(), nvfx_scene_restart(c), "restart");
  ASSERT_TRUE(scene->render(a));
  nvfx_scene_render(c, b.data(), 160 * 4);
  EXPECT_EQ(a, b);
  // views and ownership
  nvfx::SceneView view(scene->get());
  EXPECT_EQ(view.frame(), scene->frame());
  nvfx::Scene moved = std::move(*scene);
  EXPECT_FALSE(*scene);
  same(scene->get_info(), NVFX_ERROR_ARGUMENT, "get_info of none");
  nvfx_scene* raw = moved.release();
  nvfx::Scene again(raw);
  EXPECT_TRUE(again);
  nvfx_scene_free(c);
}

TEST(Wrapper, TheLibraryAndStagesAreTheCApis) {
  auto tiny = nvfx::Effect::load_memory(tiny_model(3));
  ASSERT_TRUE(tiny);
  const nvfx_library_file file{"tiny.nvfxl", kCatalogue};
  nvfx_library_desc ld = nvfx::library_desc();
  ld.files = &file;
  ld.n_files = 1;
  auto lib = nvfx::Library::open(ld);
  ASSERT_TRUE(lib) << lib.error().message;
  nvfx_library* c = nullptr;
  ASSERT_EQ(nvfx_library_open(&ld, &c, nullptr), NVFX_OK);
  const nvfx_library_file broken{"bad.nvfxl", "entry x, category fire, size 10 x 10, loop, about \"x\"\nmodul m\n"};
  ld.files = &broken;
  const auto err = nvfx::Library::open(ld);
  ASSERT_FALSE(err);
  EXPECT_EQ(err.error().status, NVFX_ERROR_SCRIPT);
  EXPECT_EQ(err.error().line, 2);
  EXPECT_EQ(err.error().message.rfind("bad.nvfxl:2:", 0), 0u) << err.error().message;
  EXPECT_EQ(lib->count(), nvfx_library_count(c));
  EXPECT_EQ(lib->find("zap"), nvfx_library_find(c, "zap"));
  EXPECT_EQ(lib->find("nothing"), -1);
  const auto e = lib->get_effect(0);
  ASSERT_TRUE(e);
  EXPECT_STREQ(e->name, "blob");
  EXPECT_EQ(e->n_params, 1);
  same(lib->get_effect(5), NVFX_ERROR_ARGUMENT, "get_effect past the last");
  const auto p = lib->get_param(0, 0);
  ASSERT_TRUE(p);
  EXPECT_STREQ(p->name, "intensity");
  same(lib->get_param(0, 1), NVFX_ERROR_ARGUMENT, "get_param past the last");
  EXPECT_EQ(lib->category_count(), nvfx_library_category_count(c));
  EXPECT_STREQ(lib->category(1), nvfx_library_category(c, 1));
  std::vector<std::string> models;
  EXPECT_TRUE(lib->list_models(0, [&](const char* n, const char* f) { models.push_back(std::string(n) + "=" + f); }));
  EXPECT_EQ(models, std::vector<std::string>{"tiny=tiny.nvfx"});
  same(lib->list_models(9, [](const char*, const char*) {}), NVFX_ERROR_ARGUMENT, "list_models past the last");

  // a stage through the wrapper and one through the C API: the same frames for the same calls
  const nvfx::StageRooms rooms = {{"blob", 2, 1}, {"zap", 1, 0}};
  const nvfx_scene_effect fx{"tiny.nvfx", tiny->get()};
  nvfx_stage_desc sd = nvfx::stage_desc();
  sd.width = 160;
  sd.height = 96;
  sd.ground = 80;
  sd.rooms = rooms.data();
  sd.n_rooms = static_cast<int>(rooms.size());
  sd.effects = &fx;
  sd.n_effects = 1;
  sd.threads = 1;
  auto stage = nvfx::Stage::create(*lib, sd);
  ASSERT_TRUE(stage) << stage.error().message;
  nvfx_stage* cs = nullptr;
  ASSERT_EQ(nvfx_stage_create(c, &sd, &cs, nullptr), NVFX_OK);
  nvfx::StageRooms unknown = {{"nothing", 1, 0}};
  sd.rooms = unknown.data();
  sd.n_rooms = 1;
  const auto no_stage = nvfx::Stage::create(*lib, sd);
  ASSERT_FALSE(no_stage);
  EXPECT_EQ(no_stage.error().status, NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(no_stage.error().message, "the library has no effect 'nothing'");
  const std::string script = stage->script();
  std::string cscript(nvfx_stage_script(cs, nullptr, 0), '\0');
  nvfx_stage_script(cs, cscript.data(), cscript.size() + 1);
  EXPECT_EQ(script, cscript);
  std::array<char, 16> head{};
  EXPECT_EQ(stage->script(head), script.size());
  EXPECT_EQ(std::string(head.data()), script.substr(0, 15));
  const auto f1 = stage->spawn("blob", 50.f, 80.f, 1.f, 0.1);
  nvfx_fx c1 = 0;
  nvfx_stage_spawn(cs, "blob", 50.f, 80.f, 1.f, 0.1, &c1);
  ASSERT_TRUE(f1);
  EXPECT_EQ(*f1, c1);
  const auto f2 = stage->spawn("zap", 100.f, 60.f);
  nvfx_fx c2 = 0;
  nvfx_stage_spawn(cs, "zap", 100.f, 60.f, 1.f, -1.0, &c2);
  ASSERT_TRUE(f2);
  same(stage->spawn("zap", 100.f, 60.f), nvfx_stage_spawn(cs, "zap", 100.f, 60.f, 1.f, -1.0, nullptr), "spawn with no room");
  same(stage->spawn("nothing", 1.f, 1.f), NVFX_ERROR_ARGUMENT, "spawn unknown");
  EXPECT_EQ(stage->free_rooms("blob"), nvfx_stage_free_rooms(cs, "blob"));
  EXPECT_EQ(stage->free_rooms("nothing"), -1);
  same(stage->set_param(*f1, "intensity", 2.f), nvfx_stage_set_param(cs, c1, "intensity", 2.f), "set_param");
  const auto got = stage->get_param(*f1, "intensity");
  ASSERT_TRUE(got);
  EXPECT_EQ(*got, 1.f) << "clamped to the range";
  same(stage->get_param(*f1, "nothing"), NVFX_ERROR_ARGUMENT, "get_param unknown");
  same(stage->set_camera(4.f, -2.f), nvfx_stage_set_camera(cs, 4.f, -2.f), "set_camera");
  nvfx::SceneView scene = stage->scene();
  nvfx::SceneView cscene(nvfx_stage_scene(cs));
  Bytes a(160 * 96 * 4), b(a.size());
  for (int f = 0; f < 30; ++f) {
    if (f == 10) {
      same(stage->move(*f1, 70.f, 80.f), nvfx_stage_move(cs, c1, 70.f, 80.f), "move");
    }
    if (f == 20) {
      same(stage->stop(*f2), nvfx_stage_stop(cs, c2, -1.0), "stop");
    }
    ASSERT_TRUE(scene.step_frames(1));
    ASSERT_TRUE(cscene.step_frames(1));
    ASSERT_TRUE(scene.render(a));
    ASSERT_TRUE(cscene.render(b));
    ASSERT_EQ(a, b) << "frame " << f;
  }
  const auto fi = stage->get_fx(*f1);
  nvfx_stage_fx_info cfi;
  nvfx_stage_get_fx(cs, c1, &cfi);
  ASSERT_TRUE(fi);
  EXPECT_EQ(fi->state, cfi.state);
  EXPECT_EQ(fi->state, NVFX_FX_PLAYING);
  EXPECT_EQ(fi->x, 70.f);
  same(stage->get_fx(0), NVFX_ERROR_ARGUMENT, "get_fx of none");
  same(stage->clear(), nvfx_stage_clear(cs), "clear");
  same(stage->get_fx(*f1), NVFX_ERROR_ARGUMENT, "a handle after clear");
  nvfx_stage_free(cs);
  nvfx_library_free(c);
}

TEST(Wrapper, TheManagerIsTheCApi) {
  auto fx = nvfx::Effect::load_memory(tiny_model(3));
  ASSERT_TRUE(fx);
  nvfx_manager_desc d = nvfx::manager_desc();
  d.max_instances = 16;
  d.max_size = 64;
  d.budget_ms = 1000.f;
  d.threads = 0;  // the "engine" runs the jobs, in both
  d.job_slots = 2;
  auto m = nvfx::Manager::create(d);
  ASSERT_TRUE(m);
  nvfx_manager* c = nullptr;
  ASSERT_EQ(nvfx_manager_create(&d, &c), NVFX_OK);
  nvfx_manager_desc bad = d;
  bad.max_instances = -1;
  nvfx_manager* none = nullptr;
  same(nvfx::Manager::create(bad), nvfx_manager_create(&bad, &none), "create with a bad desc");
  nvfx_manager_model_desc md = nvfx::manager_model_desc();
  md.max_sims = 4;
  const auto model = m->add_model(*fx, &md);
  int cmodel = -1;
  ASSERT_EQ(nvfx_manager_add_model(c, fx->get(), &md, &cmodel), NVFX_OK);
  ASSERT_TRUE(model);
  EXPECT_EQ(*model, cmodel);
  same(m->add_model(nvfx::Effect()), NVFX_ERROR_ARGUMENT, "add_model of none");
  same(m->set_cost(*model, 64, 0.3f, 0.1f), nvfx_manager_set_cost(c, cmodel, 64, 0.3f, 0.1f), "set_cost");
  same(m->set_cost(*model, 48, 0.3f, 0.1f), nvfx_manager_set_cost(c, cmodel, 48, 0.3f, 0.1f), "set_cost at a size it has not");
  const auto cost = m->get_cost(*model, 64);
  ASSERT_TRUE(cost);
  EXPECT_EQ(cost->step_ms, 0.3f);
  EXPECT_EQ(cost->render_ms, 0.1f);
  auto calibrated = nvfx::Manager::create(d);
  ASSERT_TRUE(calibrated);
  const auto cm = calibrated->add_model(*fx, &md);
  ASSERT_TRUE(cm);
  EXPECT_TRUE(calibrated->calibrate(*cm, 4));
  same(calibrated->calibrate(7, 4), NVFX_ERROR_ARGUMENT, "calibrate an unknown model");
  EXPECT_GT(calibrated->get_cost(*cm, 32)->step_ms, 0.f);
  // the pack's costs
  auto pack = nvfx::Pack::open_memory(tiny_pack());
  ASSERT_TRUE(pack);
  const auto set = calibrated->set_costs_from(*pack, *cm, "tiny");
  ASSERT_TRUE(set);
  EXPECT_EQ(*set, 1);
  EXPECT_EQ(calibrated->get_cost(*cm, 32)->step_ms, 0.3f);
  const std::array<float, 3> pos{0.f, 0.f, 0.f}, fwd{0.f, 0.f, 1.f}, up{0.f, 1.f, 0.f};
  same(m->set_camera(pos, fwd, up, 1.f, 640.f, 360.f), nvfx_manager_set_camera(c, pos.data(), fwd.data(), up.data(), 1.f, 640.f, 360.f), "set_camera");
  same(m->set_budget(500.f), nvfx_manager_set_budget(c, 500.f), "set_budget");
  same(m->set_budget(-1.f), nvfx_manager_set_budget(c, -1.f), "set_budget below 0");
  // two instances, the same calls on both managers; the engine runs the jobs on two threads
  nvfx_spawn_desc s = nvfx::spawn_desc();
  const float controls[3] = {0.4f, 0.5f, 0.6f};
  s.model = *model;
  s.controls = controls;
  s.n_controls = 3;
  s.seed = 11;
  s.screen_px = 60.f;
  const auto h1 = m->spawn(s);
  nvfx_handle c1 = 0;
  nvfx_manager_spawn(c, &s, &c1);
  s.seed = 12;
  s.position[2] = 10.f;
  s.screen_px = 0.f;
  s.world_size = 2.f;
  const auto h2 = m->spawn(s);
  nvfx_handle c2 = 0;
  nvfx_manager_spawn(c, &s, &c2);
  ASSERT_TRUE(h1);
  ASSERT_TRUE(h2);
  EXPECT_EQ(*h1, c1);
  s.model = 9;
  nvfx_handle no_handle = 0;
  same(m->spawn(s), nvfx_manager_spawn(c, &s, &no_handle), "spawn an unknown model");
  const auto tick = [](nvfx_manager* mgr, nvfx::Manager* w) {
    if (w) {
      const auto jobs = w->tick_begin(1.0 / 30.0);
      ASSERT_TRUE(jobs);
      std::jthread other([&] { EXPECT_TRUE(w->run_jobs(1)); });
      EXPECT_TRUE(w->run_jobs(0));
      other.join();
      EXPECT_TRUE(w->tick_end());
    } else {
      int jobs = 0;
      nvfx_manager_tick_begin(mgr, 1.0 / 30.0, &jobs);
      std::jthread other([&] { nvfx_manager_run_jobs(mgr, 1); });
      nvfx_manager_run_jobs(mgr, 0);
      other.join();
      nvfx_manager_tick_end(mgr);
    }
  };
  int pictures = 0;
  for (int t = 0; t < 20; ++t) {
    if (t == 5) {
      const float c2c[3] = {0.8f, 0.2f, 0.5f};
      same(m->set_controls(*h1, c2c), nvfx_manager_set_controls(c, c1, c2c, 3), "set_controls");
    }
    if (t == 8) same(m->set_position(*h2, std::array<float, 3>{0.5f, 0.f, 6.f}, 2.f), nvfx_manager_set_position(c, c2, std::array<float, 3>{0.5f, 0.f, 6.f}.data(), 2.f), "set_position");
    if (t == 10) same(m->set_screen_size(*h1, 120.f), nvfx_manager_set_screen_size(c, c1, 120.f, 1), "set_screen_size");
    if (t == 11) same(m->set_priority(*h1, 3.f), nvfx_manager_set_priority(c, c1, 3.f), "set_priority");
    if (t == 12) same(m->set_lod_lock(*h1, 32, 2), nvfx_manager_set_lod_lock(c, c1, 32, 2), "set_lod_lock");
    tick(nullptr, &*m);
    tick(c, nullptr);
    for (const auto& [h, ch] : {std::pair{*h1, c1}, std::pair{*h2, c2}}) {
      const auto pic = m->get_picture(h);
      nvfx_manager_picture cp;
      ASSERT_EQ(nvfx_manager_get_picture(c, ch, &cp), NVFX_OK);
      ASSERT_TRUE(pic);
      EXPECT_EQ(pic->flags, cp.flags) << "tick " << t;
      EXPECT_EQ(pic->is_new(), (cp.flags & NVFX_PICTURE_NEW) != 0);
      EXPECT_EQ(pic->finished(), false);
      ASSERT_EQ(pic->rgba != nullptr, cp.rgba != nullptr);
      if (cp.rgba) {
        ++pictures;
        ASSERT_EQ(pic->size, cp.size);
        EXPECT_EQ(pic->pixels().size(), cp.stride * static_cast<std::size_t>(cp.size - 1) + static_cast<std::size_t>(cp.size) * 4);
        EXPECT_TRUE(std::equal(pic->pixels().begin(), pic->pixels().end(), cp.rgba)) << "tick " << t;
      } else {
        EXPECT_TRUE(pic->pixels().empty());
      }
    }
  }
  EXPECT_GT(pictures, 20) << "the instances have pictures";
  const auto st = m->get_stats();
  nvfx_manager_stats cst;
  nvfx_manager_get_stats(c, &cst);
  ASSERT_TRUE(st);
  EXPECT_EQ(st->instances, cst.instances);
  EXPECT_EQ(st->steps, cst.steps);
  EXPECT_EQ(st->memory_bytes, cst.memory_bytes);
  same(m->stop(*h2), nvfx_manager_stop(c, c2), "stop");
  same(m->stop(*h2), nvfx_manager_stop(c, c2), "stop again");
  same(m->get_picture(*h2), NVFX_ERROR_ARGUMENT, "get_picture of a stopped instance");
  // the manager's own threads
  nvfx_manager_desc own = d;
  own.threads = 2;
  auto m2 = nvfx::Manager::create(own);
  ASSERT_TRUE(m2);
  ASSERT_TRUE(m2->add_model(*fx, &md));
  s.model = 0;
  ASSERT_TRUE(m2->spawn(s));
  EXPECT_TRUE(m2->tick(1.0 / 60.0));
  same(m2->tick(-1.0), nvfx_manager_tick(c, -1.0), "tick backwards");
  nvfx_manager_free(c);
}

TEST(Wrapper, PacksAreTheCApi) {
  const Bytes file = tiny_pack();
  const TempFile tmp(file, ".nvfxp");
  auto from_file = nvfx::Pack::open(tmp.path.string(), 2);
  ASSERT_TRUE(from_file) << from_file.error().message;
  auto pack = nvfx::Pack::open_memory(file, 1, "tiny.nvfxp");
  ASSERT_TRUE(pack);
  nvfx_pack_desc pd = nvfx::pack_desc();
  EXPECT_EQ(pd.threads, 1);
  pd.data = file.data();
  pd.bytes = file.size();
  ASSERT_TRUE(nvfx::Pack::open(pd));
  Bytes broken = file;
  broken.resize(file.size() - 10);
  const auto err = nvfx::Pack::open_memory(broken, 1, "tiny.nvfxp");
  ASSERT_FALSE(err);
  EXPECT_EQ(err.error().status, NVFX_ERROR_FORMAT);
  EXPECT_EQ(err.error().message, std::format("tiny.nvfxp: truncated: {} of its {} bytes", broken.size(), file.size()));
  const auto missing = nvfx::Pack::open("/nonexistent/p.nvfxp");
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().status, NVFX_ERROR_IO);
  const auto info = pack->get_info();
  ASSERT_TRUE(info);
  EXPECT_EQ(info->n_members, 4);
  EXPECT_EQ(info->n_models, 1);
  EXPECT_STREQ(info->licence, "credits");
  EXPECT_EQ(pack->find("tiny"), 0);
  EXPECT_EQ(pack->find(std::string("camp.nvfxs")), 2);
  const auto m = pack->get_member(1);
  ASSERT_TRUE(m);
  EXPECT_STREQ(m->name, "tiny.nvfxl");
  EXPECT_EQ(m->kind, NVFX_PACK_CATALOGUE);
  EXPECT_TRUE(std::ranges::equal(nvfx::bytes(*m), bytes_of(kCatalogue)));
  same(pack->get_member(4), NVFX_ERROR_ARGUMENT, "get_member past the last");
  EXPECT_STREQ(pack->entry_name(0), "blob");
  EXPECT_EQ(pack->entry_name(2), nullptr);
  std::vector<std::string> needs;
  EXPECT_TRUE(pack->entry_models(0, [&](const char* n, const char* f) { needs.push_back(std::string(n) + "=" + f); }));
  EXPECT_EQ(needs, std::vector<std::string>{"tiny=tiny.nvfx"});
  same(pack->entry_models(5, [](const char*, const char*) {}), NVFX_ERROR_ARGUMENT, "entry_models past the last");
  auto effect = pack->load_effect("tiny");
  ASSERT_TRUE(effect);
  EXPECT_EQ(effect->get_info()->arch, 3);
  same(pack->load_effect("nothing"), NVFX_ERROR_IO, "load_effect unknown");
  const auto cost = pack->model_cost("tiny", 32);
  ASSERT_TRUE(cost);
  EXPECT_EQ(cost->step_ms, 0.3f);
  same(pack->model_cost("tiny", 64), NVFX_ERROR_ARGUMENT, "model_cost at a size it has not");
  // a library and a scene from the pack (the pack may go first)
  auto lib = nvfx::Library::open_pack(*pack);
  ASSERT_TRUE(lib) << lib.error().message;
  EXPECT_EQ(lib->count(), 2);
  nvfx_scene_desc sd = nvfx::scene_desc();
  sd.script_path = "camp.nvfxs";
  sd.threads = 1;
  auto scene = nvfx::Scene::create_from_pack(*pack, sd);
  ASSERT_TRUE(scene) << scene.error().message;
  pack->reset();
  EXPECT_FALSE(*pack);
  const nvfx::StageRooms rooms = {{"blob", 1, 0}};
  nvfx_stage_desc st = nvfx::stage_desc();
  st.width = 128;
  st.height = 96;
  st.rooms = rooms.data();
  st.n_rooms = 1;
  st.threads = 1;
  auto stage = nvfx::Stage::create(*lib, st);
  ASSERT_TRUE(stage) << stage.error().message;
  EXPECT_TRUE(stage->spawn("blob", 64.f, 80.f));
  Bytes rgba(128 * 96 * 4);
  EXPECT_TRUE(stage->scene().step(0.2));
  EXPECT_TRUE(stage->scene().render(rgba));
  sd.script_path = "nothing.nvfxs";
  const auto no_scene = nvfx::Scene::create_from_pack(*from_file, sd);
  ASSERT_FALSE(no_scene);
  EXPECT_EQ(no_scene.error().status, NVFX_ERROR_IO);
  auto models_only = nvfx::Pack::open_memory(pk::write({{"tiny.nvfx", pk::Kind::model, tiny_model(3)}}, {}, {}, {}).file);
  ASSERT_TRUE(models_only);
  const auto no_lib = nvfx::Library::open_pack(*models_only);
  ASSERT_FALSE(no_lib);
  EXPECT_EQ(no_lib.error().message, "pack: the pack has no catalogue");
}

// A game's frame through the wrapper allocates nothing: an instance rendered, a scene stepped, driven, read and drawn,
// a stage's effects spawned, moved, set and stopped, and the manager ticked and read. (Errors aside: an Error carries a
// std::string.)
TEST(Wrapper, AFrameAllocatesNothing) {
  auto pack = nvfx::Pack::open_memory(tiny_pack());
  ASSERT_TRUE(pack);
  auto fx = pack->load_effect("tiny");
  ASSERT_TRUE(fx);
  auto inst = nvfx::Instance::create(*fx, 64);
  ASSERT_TRUE(inst);
  nvfx_scene_desc sd = nvfx::scene_desc();
  sd.script_path = "camp.nvfxs";
  sd.threads = 2;
  auto scene = nvfx::Scene::create_from_pack(*pack, sd);
  ASSERT_TRUE(scene);
  auto lib = nvfx::Library::open_pack(*pack);
  ASSERT_TRUE(lib);
  const nvfx::StageRooms rooms = {{"blob", 2, 1}, {"zap", 1, 0}};
  nvfx_stage_desc st = nvfx::stage_desc();
  st.width = 128;
  st.height = 96;
  st.rooms = rooms.data();
  st.n_rooms = 2;
  auto stage = nvfx::Stage::create(*lib, st);
  ASSERT_TRUE(stage);
  nvfx_manager_desc md = nvfx::manager_desc();
  md.max_instances = 8;
  md.max_size = 64;
  md.threads = 2;
  auto mgr = nvfx::Manager::create(md);
  ASSERT_TRUE(mgr);
  nvfx_manager_model_desc mm = nvfx::manager_model_desc();
  mm.max_sims = 4;
  const auto model = mgr->add_model(*fx, &mm);
  ASSERT_TRUE(model);
  nvfx_spawn_desc s = nvfx::spawn_desc();
  s.model = *model;
  s.screen_px = 48.f;
  const auto handle = mgr->spawn(s);
  ASSERT_TRUE(handle);
  Bytes sprite(inst->frame_bytes()), picture(160 * 90 * 4), stage_rgba(128 * 96 * 4);
  std::vector<float> grid(16);
  const float controls[3] = {0.6f, 0.5f, 0.5f};
  long allocations = 0, failures = 0;
  for (int f = 0; f < 60; ++f) {
    g_allocations.store(0);
    g_counting.store(f >= 2);
    const double t = f / 60.0;
    failures += !inst->render(t, sprite);
    failures += !scene->step(1.0 / 60.0);
    failures += !scene->sample(60.f, 70.f);
    failures += !scene->sample_grid(NVFX_FIELD_HEAT, 10.f, 10.f, 40.f, 20.f, 4, 4, grid);
    failures += !scene->field_region(NVFX_FIELD_SOOT, 0.f, 0.f, 80.f, 90.f);
    failures += !scene->set_input("spot", 40.f + static_cast<float>(f));
    if (f == 30) failures += !scene->trigger("boom");
    failures += !scene->render(picture);
    if (f % 20 == 3) {
      const auto h = stage->spawn("blob", 40.f + static_cast<float>(f), 80.f);
      failures += !h;
      if (h) {
        failures += !stage->set_param(*h, "intensity", 0.8f);
        failures += !stage->move(*h, 50.f, 80.f);
        failures += !stage->get_fx(*h);
        if (f > 40) failures += !stage->stop(*h);
      }
    }
    failures += !stage->scene().step(1.0 / 60.0);
    failures += !stage->scene().render(stage_rgba);
    failures += !mgr->tick(1.0 / 60.0);
    failures += !mgr->get_picture(*handle);
    failures += !mgr->get_stats();
    failures += !mgr->set_controls(*handle, controls);
    g_counting.store(false);
    allocations += g_allocations.load();
  }
  EXPECT_EQ(failures, 0);
  EXPECT_EQ(allocations, 0) << "allocations in a frame through the wrapper";
  std::uint8_t lo = 255, hi = 0;  // the frames show something
  for (std::size_t i = 0; i < stage_rgba.size(); i += 4) {
    lo = std::min(lo, stage_rgba[i]);
    hi = std::max(hi, stage_rgba[i]);
  }
  EXPECT_GT(hi - lo, 20);
  EXPECT_GT(*std::max_element(sprite.begin(), sprite.end()), 0);
  g_allocations.store(0);  // and the count counts
  g_counting.store(true);
  const std::string longer(100, 'x');
  g_counting.store(false);
  EXPECT_GE(g_allocations.load(), 1) << longer.size();
}
