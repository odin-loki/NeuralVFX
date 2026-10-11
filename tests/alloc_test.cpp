// nvfx_render must not allocate (docs/PLAN.md §5.4), nor may a frame of a composed scene (docs/COMPOSE.md). This
// program replaces the global operator new to count heap allocations, builds small grid, conv and rollout effects in
// memory, and renders 200 frames of each with every feature switched on (controls, seeded drift, colour; grid models at
// float and int8 precision), a rollout effect as one continuous rollout with the prior against drift, then 99 frames of
// a small composed scene (its modules with their own step scratch, and sharing one per thread), 44 frames of a scripted
// one, and the scripted one through the scene C API (nvfx_scene.h) with the game's calls between frames: steps, field
// reads, inputs, triggers, module moves and controls. Exit code 0 = no allocation during rendering.
#include "compose_scene.hpp"
#include "script.hpp"

#include <neuralfx/dcm/ddpm.hpp>
#include <neuralfx/model.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/nvfx_scene.h>
#include <neuralfx/rollout.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <sstream>
#include <string>
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
void* operator new(std::size_t n, std::align_val_t a) {  // aligned allocations (the prior's buffers) count too
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

int run(nfx::Hyper h, int size, const char* name, int bits = 16, int vq_bits = 0, nvfx_precision precision = NVFX_PRECISION_DEFAULT) {
  nfx::Model m = nfx::init_model(h, 1);
  m.effect = name;
  m.feature_bits = bits;
  if (vq_bits > 0) {  // vector-quantised features: an index per two channels
    m.vq_bits = vq_bits;
    m.vq_dim = 2;
    m.vq_codebook.assign(static_cast<std::size_t>(m.vq_groups()) * (std::size_t{1} << vq_bits) * 2, 0.f);
    for (std::size_t i = 0; i < m.vq_codebook.size(); ++i) m.vq_codebook[i] = 0.01f * static_cast<float>(i % 97);
  }
  m.z_train = {std::vector<float>(static_cast<std::size_t>(h.n_latent), 0.1f), std::vector<float>(static_cast<std::size_t>(h.n_latent), -0.2f)};
  m.z_mean.assign(static_cast<std::size_t>(h.n_latent), 0.f);
  m.z_std.assign(static_cast<std::size_t>(h.n_latent), 0.1f);
  std::ostringstream os;
  if (!nfx::save_model(os, m)) return 1;
  const std::string bytes = os.str();
  nvfx_effect* e = nullptr;
  nvfx_instance* in = nullptr;
  if (nvfx_effect_load_memory(bytes.data(), bytes.size(), &e) != NVFX_OK) return 1;
  if (nvfx_instance_create(e, size, &in) != NVFX_OK) return 1;
  if (nvfx_instance_set_precision(in, precision) != NVFX_OK) return 1;
  const float controls[3] = {0.7f, 0.2f, 0.9f};
  nvfx_instance_set_controls(in, controls, 3);
  nvfx_instance_set_seed(in, 99);
  nvfx_instance_set_drift(in, 0.5f);
  nvfx_instance_set_colour(in, 0.4f, 1.2f);
  std::vector<std::uint8_t> rgba(static_cast<std::size_t>(size) * size * 4);
  nvfx_render(in, 0.0, rgba.data(), static_cast<std::size_t>(size) * 4);  // warm-up outside the count
  g_allocations = 0;
  g_counting = true;
  for (int f = 0; f < 200; ++f) nvfx_render(in, f / 30.0, rgba.data(), static_cast<std::size_t>(size) * 4);
  g_counting = false;
  const long n = g_allocations.load();
  std::printf("%s %dx%d, %d-bit features%s%s: %ld allocations in 200 frames\n", name, size, size, bits, vq_bits ? " (vector-quantised)" : "",
              precision == NVFX_PRECISION_FLOAT ? ", float" : precision == NVFX_PRECISION_INT8 ? ", int8" : "", n);
  nvfx_instance_free(in);
  nvfx_effect_free(e);
  return n == 0 ? 0 : 1;
}

// A rollout effect: the counted frames include a seek backwards (a restart from a start point, with warm-up steps).
int run_rollout(int size) {
  nfx::rollout::Hyper h;
  h.res = 16;
  h.hidden = 8;
  h.memory = 2;
  h.jacobi = 10;
  h.render_hidden = 6;
  h.warmup = 4;
  nfx::rollout::Model m = nfx::rollout::init_model(h, 3);
  m.effect = "rollout";
  m.control_names = {"intensity", "wind", "turbulence"};
  m.scale = {0.2f, 0.2f, 0.4f, 0.3f};
  m.lo = {-2.f, -2.f, 0.f, 0.f};
  m.hi = {2.f, 2.f, 3.f, 3.f};
  m.detail.swirl_control = 2;
  for (int k = 0; k < 3; ++k) {
    nfx::rollout::StartPoint sp;
    sp.controls = {0.2f * static_cast<float>(k), 0.5f, 0.5f};
    sp.seed = static_cast<std::uint64_t>(k);
    sp.coarse.assign(static_cast<std::size_t>(h.res) * h.res * nfx::rollout::kPhys, 0.2f);
    m.starts.push_back(sp);
  }
  std::ostringstream os;
  if (!nfx::rollout::save_model(os, m)) return 1;
  const std::string bytes = os.str();
  nvfx_effect* e = nullptr;
  nvfx_instance* in = nullptr;
  if (nvfx_effect_load_memory(bytes.data(), bytes.size(), &e) != NVFX_OK) return 1;
  if (nvfx_instance_create(e, size, &in) != NVFX_OK) return 1;
  const float controls[3] = {0.7f, 0.2f, 0.9f};
  nvfx_instance_set_controls(in, controls, 3);
  nvfx_instance_set_seed(in, 99);
  nvfx_instance_set_colour(in, 0.4f, 1.2f);
  std::vector<std::uint8_t> rgba(static_cast<std::size_t>(size) * size * 4);
  nvfx_render(in, 0.0, rgba.data(), static_cast<std::size_t>(size) * 4);  // warm-up outside the count
  g_allocations = 0;
  g_counting = true;
  for (int f = 0; f < 200; ++f) nvfx_render(in, f / 30.0, rgba.data(), static_cast<std::size_t>(size) * 4);
  nvfx_render(in, 1.0, rgba.data(), static_cast<std::size_t>(size) * 4);  // backwards: restart
  nvfx_instance_set_seed(in, 5);
  nvfx_render(in, 1.5, rgba.data(), static_cast<std::size_t>(size) * 4);
  g_counting = false;
  const long n = g_allocations.load();
  std::printf("rollout %dx%d: %ld allocations in 202 frames (with a restart)\n", size, size, n);
  nvfx_instance_free(in);
  nvfx_effect_free(e);
  return n == 0 ? 0 : 1;
}

// A rollout effect played as one continuous rollout with the prior against drift (docs/DCM.md G2.13) acting every 4
// frames: 50 passes of the denoiser in the counted frames, then a seek backwards (a replay from the start, with passes)
// and a new seed.
int run_rollout_prior(int size) {
  nfx::rollout::Hyper h;
  h.res = 16;
  h.hidden = 8;
  h.memory = 2;
  h.jacobi = 10;
  h.render_hidden = 6;
  h.warmup = 4;
  nfx::rollout::Model m = nfx::rollout::init_model(h, 3);
  m.effect = "rollout";
  m.control_names = {"intensity", "wind", "turbulence"};
  m.scale = {0.2f, 0.2f, 0.4f, 0.3f};
  m.lo = {-2.f, -2.f, 0.f, 0.f};
  m.hi = {2.f, 2.f, 3.f, 3.f};
  for (int k = 0; k < 3; ++k) {
    nfx::rollout::StartPoint sp;
    sp.controls = {0.2f * static_cast<float>(k), 0.5f, 0.5f};
    sp.seed = static_cast<std::uint64_t>(k);
    sp.coarse.assign(static_cast<std::size_t>(h.res) * h.res * nfx::rollout::kPhys, 0.2f);
    m.starts.push_back(sp);
  }
  namespace dd = nfx::dcm::ddpm;
  dd::Config c;
  c.res = h.res;
  c.c0 = 8;
  c.c1 = 16;
  c.c2 = 16;
  dd::Denoiser d = dd::init_denoiser(c, 4);
  std::vector<float> g(d.w.size());
  dd::gaussian(5, g);
  for (std::size_t i = 0; i < g.size(); ++i) d.w[i] += 0.03f * g[i];
  d.scale = {0.2f, 0.2f, 0.4f, 0.3f};
  const std::string prior = dd::serialise(d);
  std::ostringstream os;
  if (!nfx::rollout::save_model(os, m)) return 1;
  const std::string bytes = os.str();
  nvfx_effect* e = nullptr;
  nvfx_instance* in = nullptr;
  if (nvfx_effect_load_memory(bytes.data(), bytes.size(), &e) != NVFX_OK) return 1;
  if (nvfx_effect_attach_prior_memory(e, prior.data(), prior.size()) != NVFX_OK) return 1;
  if (nvfx_instance_create(e, size, &in) != NVFX_OK) return 1;
  const float controls[3] = {0.7f, 0.2f, 0.9f};
  nvfx_instance_set_controls(in, controls, 3);
  nvfx_instance_set_seed(in, 99);
  nvfx_instance_set_colour(in, 0.4f, 1.2f);
  nvfx_instance_set_drift(in, 0.f);
  if (nvfx_instance_set_prior(in, 4, 100, 1.f) != NVFX_OK) return 1;
  std::vector<std::uint8_t> rgba(static_cast<std::size_t>(size) * size * 4);
  nvfx_render(in, 0.0, rgba.data(), static_cast<std::size_t>(size) * 4);  // warm-up outside the count
  g_allocations = 0;
  g_counting = true;
  for (int f = 0; f < 200; ++f) nvfx_render(in, f / 30.0, rgba.data(), static_cast<std::size_t>(size) * 4);
  nvfx_render(in, 1.0, rgba.data(), static_cast<std::size_t>(size) * 4);  // backwards: replay from the start
  nvfx_instance_set_seed(in, 5);
  nvfx_render(in, 1.5, rgba.data(), static_cast<std::size_t>(size) * 4);
  g_counting = false;
  const long n = g_allocations.load();
  // the prior did act: the same frame without it differs
  std::vector<std::uint8_t> plain(rgba.size());
  nvfx_instance_set_prior(in, 0, 100, 1.f);
  nvfx_render(in, 1.5, plain.data(), static_cast<std::size_t>(size) * 4);
  const bool acted = plain != rgba;
  std::printf("rollout %dx%d, one continuous rollout with the prior every 4 frames: %ld allocations in 202 frames (with a replay)%s\n", size, size, n,
              acted ? "" : "; the prior did not act");
  nvfx_instance_free(in);
  nvfx_effect_free(e);
  return n == 0 && acted ? 0 : 1;
}

// A composed scene (src/compose): stepping, couplings, the bus, light, particles and the whole frame, on 2 threads.
int run_compose(bool shared_scratch) {
  nfx::compose::testing::MiniScene scene(2, shared_scratch);
  std::vector<std::uint8_t> rgb(160 * 90 * 3);
  scene.step(0, rgb);  // warm-up outside the count
  g_allocations = 0;
  g_counting = true;
  for (int f = 1; f < 100; ++f) scene.step(f, rgb);
  g_counting = false;
  const long n = g_allocations.load();
  std::printf("composed scene 160x90%s: %ld allocations in 99 frames\n", shared_scratch ? ", step scratch shared" : "", n);
  // the same with the picture drawn on its own thread while the next frame's state is computed
  nfx::compose::testing::MiniScene over(2, shared_scratch);
  nfx::compose::PictureThread picture(over.frame, over.pool);
  std::array<std::vector<std::uint8_t>, 2> bufs{rgb, rgb};
  const auto frame = [&](int f, bool count_from_start) {
    over.capture();
    picture.start(bufs[static_cast<std::size_t>(f % 2)], 0.8f, 1.f);
    if (count_from_start) g_counting = true;  // the picture thread's first picture is counted too
    over.state(f + 1);
    picture.wait_images();
    over.shade();
    picture.wait();
  };
  over.state(0);
  over.shade();
  g_allocations = 0;
  frame(0, true);  // (the capture of the first frame sizes what the pictures need: outside the count)
  for (int f = 1; f < 100; ++f) frame(f, false);
  g_counting = false;
  const long m = g_allocations.load();
  std::printf("composed scene 160x90%s, picture overlapped: %ld allocations in 100 pictures and 99 frames\n", shared_scratch ? ", step scratch shared" : "", m);
  return n == 0 && m == 0 ? 0 : 1;
}

// A scripted scene (src/compose/script.hpp): rules on time, shocks, the bus and landings, emitters, every kind of field,
// hand-over, wake, transfers and pushes, all on 2 threads.
int run_script() {
  namespace sc = nfx::compose::script;
  const sc::Script s = sc::parse(nfx::compose::testing::kEverything, "everything");
  int failures = 0;
  for (const bool overlap : {false, true}) {
    sc::Scene scene(s, [](const std::string& file) { return nfx::compose::testing::tiny_effect(file == "other" ? 4 : 3); }, {2, nfx::compose::best_isa(), overlap});
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(scene.width()) * static_cast<std::size_t>(scene.height()) * 3);
    scene.render(0, rgb);  // warm-up outside the count
    g_allocations = 0;
    g_counting = true;
    for (int f = 1; f < scene.frames(); ++f) scene.render(f, rgb);
    g_counting = false;
    const long n = g_allocations.load();
    int fired = 0;
    for (const auto& r : scene.rules_fired()) fired += r.second < 1e30f;
    std::printf("scripted scene %dx%d%s: %ld allocations in %d frames (%d of %zu rules fired)\n", scene.width(), scene.height(), overlap ? ", overlapped" : "", n,
                scene.frames() - 1, fired, scene.rules_fired().size());
    failures += n == 0 && fired >= 6 ? 0 : 1;
  }
  return failures;
}

// The scene C API: a scripted scene with an input and a rule the game fires, driven as a game drives it. Everything
// is made by nvfx_scene_create; the frames, at 30 Hz and at 60 Hz (a frame every other step), allocate nothing.
int run_scene_api() {
  std::string tiny, other;
  for (const auto& [seed, bytes] : {std::pair<std::uint64_t, std::string*>{3, &tiny}, {4, &other}}) {
    std::ostringstream os;
    if (!nfx::rollout::save_model(os, nfx::compose::testing::tiny_effect(seed).m)) return 1;
    *bytes = os.str();
  }
  nvfx_effect* et = nullptr;
  nvfx_effect* eo = nullptr;
  if (nvfx_effect_load_memory(tiny.data(), tiny.size(), &et) != NVFX_OK || nvfx_effect_load_memory(other.data(), other.size(), &eo) != NVFX_OK) return 1;
  const nvfx_scene_effect list[] = {{"tiny", et}, {"other", eo}};
  const std::string script = std::string(nfx::compose::testing::kEverything) +
                             "input breeze_in = 0.3\nfield breeze2 = wind, velocity (breeze_in, 0), on fire\nwhen 0 as poke, repeat:\n  shock at (80, 50)\n";
  int failures = 0;
  for (const int overlap : {0, 1}) {
    nvfx_scene_desc d;
    nvfx_scene_desc_init(&d);
    d.script = script.c_str();
    d.effects = list;
    d.n_effects = 2;
    d.threads = 2;
    d.overlap = overlap;
    nvfx_scene* s = nullptr;
    nvfx_scene_error err;
    if (nvfx_scene_create(&d, &s, &err) != NVFX_OK) {
      std::printf("scene API: %s\n", err.message);
      return 1;
    }
    nvfx_scene_info info{};
    nvfx_scene_get_info(s, &info);
    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(info.width) * static_cast<std::size_t>(info.height) * 4);
    std::vector<float> grid(8 * 4);
    nvfx_scene_render(s, rgba.data(), static_cast<std::size_t>(info.width) * 4);  // warm-up outside the count
    g_allocations = 0;
    g_counting = true;
    float heat = 0.f;
    for (int f = 1; f < 2 * info.frames; ++f) {
      nvfx_scene_step(s, 1.0 / 60.0, nullptr);
      nvfx_scene_fields sm{};
      nvfx_scene_sample(s, 66.f, 70.f, &sm);
      heat = std::max(heat, sm.heat);
      nvfx_scene_sample_grid(s, NVFX_FIELD_HEAT, 10.f, 10.f, 20.f, 20.f, 8, 4, grid.data(), 8);
      float mx = 0.f;
      nvfx_scene_field_region(s, NVFX_FIELD_SOOT, 0.f, 0.f, 160.f, 80.f, &mx, nullptr);
      nvfx_scene_set_input(s, "breeze_in", 0.01f * static_cast<float>(f % 50));
      if (f % 20 == 0) nvfx_scene_trigger(s, "poke");
      if (f == 30) nvfx_scene_module_place(s, "spare", 50.f, 80.f);
      if (f == 31) nvfx_scene_module_set_control(s, "spare", "wind", 0.7f);
      int count = 0;
      nvfx_scene_rule_state(s, "poke", &count, nullptr);
      nvfx_scene_module_info mi{};
      nvfx_scene_module_get_info(s, "fire", &mi);
      nvfx_scene_render(s, rgba.data(), static_cast<std::size_t>(info.width) * 4);
    }
    g_counting = false;
    const long n = g_allocations.load();
    int pokes = 0;
    nvfx_scene_rule_state(s, "poke", &pokes, nullptr);
    std::printf("scene API %dx%d%s: %ld allocations in %d steps at 60 Hz (%d frames; %d triggered rules fired; most heat %.2f)\n", info.width, info.height,
                overlap ? ", overlapped" : "", n, 2 * info.frames - 1, nvfx_scene_frame(s), pokes, static_cast<double>(heat));
    failures += n == 0 && pokes >= 4 ? 0 : 1;
    nvfx_scene_free(s);
  }
  nvfx_effect_free(et);
  nvfx_effect_free(eo);
  return failures;
}

}  // namespace

int main() {
  nfx::Hyper g;
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
  nfx::Hyper c;
  c.arch = nfx::Arch::conv;
  c.latent = 8;
  c.size = 64;
  c.frames = 16;
  c.loop = false;
  c.n_controls = 3;
  c.n_latent = 4;
  c.bases = 2;
  c.grid_t = 4;
  c.c0 = 8;
  c.c1 = 8;
  c.c2 = 8;
  int failures = run(g, 64, "grid") + run(g, 128, "grid") + run(c, 64, "conv") + run(c, 32, "conv") + run_rollout(64) + run_rollout(128) + run_compose(false) +
                 run_compose(true) + run_script();
  nfx::Hyper deep = g;  // int8 with two hidden layers, and with the first layer per pixel (a grid wider than the frame)
  deep.layers = 3;
  nfx::Hyper wide = g;
  wide.grid = 40;
  failures += run(g, 128, "grid", 16, 0, NVFX_PRECISION_FLOAT) + run(g, 128, "grid", 16, 0, NVFX_PRECISION_INT8) + run(deep, 64, "grid", 8, 0, NVFX_PRECISION_INT8) +
              run(wide, 32, "grid", 16, 0, NVFX_PRECISION_INT8);
  failures += run_scene_api();
  failures += run_rollout_prior(64) + run_rollout_prior(128);  // the prior against drift
  failures += run(g, 128, "grid", 8) + run(g, 128, "grid", 5) + run(g, 64, "grid", 4) + run(c, 64, "conv", 4);  // packed features
  failures += run(g, 128, "grid", 8, 6) + run(c, 64, "conv", 8, 8);  // vector-quantised features
  std::printf("%s\n", failures ? "FAILED: nvfx_render allocated" : "ok: no allocation per frame");
  return failures ? 1 : 0;
}
