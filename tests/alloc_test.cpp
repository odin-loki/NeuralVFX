// nvfx_render must not allocate (docs/PLAN.md §5.4), nor may a frame of a composed scene (docs/COMPOSE.md). This
// program replaces the global operator new to count heap allocations, builds small grid, conv and rollout effects in
// memory, and renders 200 frames of each with every feature switched on (controls, seeded drift, colour; grid models at
// float and int8 precision), a rollout effect as one continuous rollout with the prior against drift, then 99 frames of
// a small composed scene (its modules with their own step scratch, and sharing one per thread), 44 frames of a scripted
// one, and the scripted one through the scene C API (nvfx_scene.h) with the game's calls between frames: steps, field
// reads, inputs, triggers, module moves and controls; then a stage of the effect library (nvfx_library.h): learned and
// procedural effects (glows, lamps, bursts of sparks, shards) spawned, moved, set, stopped and stolen between frames,
// overlapped and not; a stage with the engine layers on (read every frame in every format), a trail moved every frame
// (its tiles switched on, grown, faded and taken again) and procedural weather (rain washing, snow melting); a stage of
// the third round's procedural elements (hail, aurora, rainbow, meteors, a beam, a shield, glints, bursts of spatter
// and shards), its layers read every frame; and the effect manager (nvfx_manager.h) ticking a
// crowd on its threads and on the engine's, with spawns, stops, controls, camera moves, a tight budget, proxies, copies
// offset in time, pauses and culls; and a 3D volume (nvfx_volume.h, study N) stepped, read and drawn on 1 and 3 threads.
// Exit code 0 = no allocation during rendering.
#include "compose_scene.hpp"
#include "nvfx_internal.hpp"
#include "script.hpp"

#include <neuralfx/dcm/ddpm.hpp>
#include <neuralfx/model.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/nvfx_library.h>
#include <neuralfx/nvfx_manager.h>
#include <neuralfx/nvfx_scene.h>
#include <neuralfx/nvfx_volume.h>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout3d.hpp>

#include <algorithm>
#include <cmath>
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

// A frame model with a width per feature plane (every width from 0 to 8 bits among the planes; study F3), and with
// `masked` (grid family) only about half of each slice's grid points stored.
int run_per_plane(nfx::Hyper h, int size, const char* name, bool masked) {
  nfx::Model m = nfx::init_model(h, 1);
  m.effect = name;
  m.feature_bits = 8;
  m.plane_bits.resize(m.features.size() / (static_cast<std::size_t>(h.feature_side()) * h.feature_side()));
  for (std::size_t k = 0; k < m.plane_bits.size(); ++k) m.plane_bits[k] = static_cast<std::uint8_t>(k % 9);
  if (masked) {
    m.feature_mask.resize(static_cast<std::size_t>(h.grid_t) * h.grid * h.grid);
    for (std::size_t j = 0; j < m.feature_mask.size(); ++j) m.feature_mask[j] = static_cast<std::uint8_t>((j * 7 / 3) % 2);
  }
  m.z_train = {std::vector<float>(static_cast<std::size_t>(h.n_latent), 0.1f)};
  m.z_mean.assign(static_cast<std::size_t>(h.n_latent), 0.f);
  m.z_std.assign(static_cast<std::size_t>(h.n_latent), 0.1f);
  std::ostringstream os;
  if (!nfx::save_model(os, m)) return 1;
  const std::string bytes = os.str();
  nvfx_effect* e = nullptr;
  nvfx_instance* in = nullptr;
  if (nvfx_effect_load_memory(bytes.data(), bytes.size(), &e) != NVFX_OK) return 1;
  if (nvfx_instance_create(e, size, &in) != NVFX_OK) return 1;
  const float controls[3] = {0.7f, 0.2f, 0.9f};
  nvfx_instance_set_controls(in, controls, 3);
  nvfx_instance_set_seed(in, 99);
  nvfx_instance_set_drift(in, 0.5f);
  std::vector<std::uint8_t> rgba(static_cast<std::size_t>(size) * size * 4);
  nvfx_render(in, 0.0, rgba.data(), static_cast<std::size_t>(size) * 4);  // warm-up outside the count
  g_allocations = 0;
  g_counting = true;
  for (int f = 0; f < 200; ++f) nvfx_render(in, f / 30.0, rgba.data(), static_cast<std::size_t>(size) * 4);
  g_counting = false;
  const long n = g_allocations.load();
  std::printf("%s %dx%d, 0 to 8 bits per feature plane%s: %ld allocations in 200 frames\n", name, size, size, masked ? ", sparse" : "", n);
  nvfx_instance_free(in);
  nvfx_effect_free(e);
  return n == 0 ? 0 : 1;
}

// A multi-level frame model (study F4): three levels at their own widths, optionally masked (stored as runs), the MLP
// at 8 bits, Fourier features of position and time.
int run_multi(int size, bool masked, nvfx_precision precision = NVFX_PRECISION_DEFAULT) {
  nfx::Hyper h;
  h.arch = nfx::Arch::multi;
  h.size = 64;
  h.frames = 16;
  h.n_controls = 3;
  h.n_latent = 4;
  h.bases = 2;
  h.levels = {{16, 4, 4}, {8, 1, 4}, {32, 8, 2}};  // int8: two projected levels, one per pixel
  h.pe_xy = 2;
  h.pe_t = 2;
  h.hidden = 16;
  nfx::Model m = nfx::init_model(h, 1);
  m.effect = "multi";
  m.mlp_bits = 8;
  const auto vols = nfx::volumes(h);
  for (std::size_t l = 0; l < vols.size(); ++l) {
    const std::size_t n = static_cast<std::size_t>(h.bases) * vols[l].slices * vols[l].channels;
    for (std::size_t k = 0; k < n; ++k) m.plane_bits.push_back(static_cast<std::uint8_t>(3 + l * 2));
  }
  if (masked) {
    m.feature_mask.resize(vols.back().mask0 + static_cast<std::size_t>(vols.back().slices) * vols.back().plane_values());
    for (std::size_t j = 0; j < m.feature_mask.size(); ++j) m.feature_mask[j] = static_cast<std::uint8_t>((j * 7 / 3) % 2);
  }
  m.z_train = {std::vector<float>(static_cast<std::size_t>(h.n_latent), 0.1f)};
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
  std::printf("multi-level %dx%d%s%s: %ld allocations in 200 frames\n", size, size, masked ? ", sparse" : "",
              precision == NVFX_PRECISION_FLOAT ? ", float" : ", int8", n);
  nvfx_instance_free(in);
  nvfx_effect_free(e);
  return n == 0 ? 0 : 1;
}

// A rollout effect: the counted frames include a seek backwards (a restart from a start point, with warm-up steps).
// With `handoff`, an explosion whose first 30 frames are drawn in the simulator's look, crossfading over 10
// (docs/COMPOSE.md §10). With `d2`, study D2's detail constants (advection gain and softening, docs/REPORT.md §6.9).
// With `int8`, study K's int8 path (both convolutions with `both`; docs/REPORT.md §13).
int run_rollout(int size, bool handoff = false, bool d2 = false, bool int8 = false, bool both = false) {
  nfx::rollout::Hyper h;
  h.res = 16;
  h.hidden = 8;
  h.memory = 2;
  h.jacobi = 10;
  h.render_hidden = 6;
  h.warmup = 4;
  nfx::rollout::Model m = nfx::rollout::init_model(h, 3);
  m.effect = handoff ? "explosion" : "rollout";
  m.control_names = {"intensity", "wind", "turbulence"};
  m.scale = {0.2f, 0.2f, 0.4f, 0.3f};
  m.lo = {-2.f, -2.f, 0.f, 0.f};
  m.hi = {2.f, 2.f, 3.f, 3.f};
  m.detail.swirl_control = 2;
  if (d2) {
    m.detail.advect = 1.2f;
    m.detail.soften = 0.05f;
  }
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
  if (int8 && both && nfx::rt::set_rollout_int8_convs(in, nfx::rt::Int8Convs::both) != NVFX_OK) return 1;
  if (nvfx_instance_set_precision(in, int8 ? NVFX_PRECISION_INT8 : NVFX_PRECISION_FLOAT) != NVFX_OK) return 1;
  const float controls[3] = {0.7f, 0.2f, 0.9f};
  nvfx_instance_set_controls(in, controls, 3);
  nvfx_instance_set_seed(in, 99);
  nvfx_instance_set_colour(in, 0.4f, 1.2f);
  if (handoff && nvfx_instance_set_handoff(in, 30, 10) != NVFX_OK) return 1;
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
  std::printf("rollout %dx%d%s: %ld allocations in 202 frames (with a restart)\n", size, size,
              handoff ? " with the hand-off" : (d2 ? " with study D2's detail constants" : int8 ? (both ? " at int8 (both convolutions)" : " at int8") : ""), n);
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
      float cam_x = 0.f, cam_y = 0.f;
      nvfx_scene_camera(s, &cam_x, &cam_y);
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

// A stage of the effect library: a looping learned effect in shards, a one-shot, procedural glows and a lamp. Every
// call a game makes while it plays (spawn, set_param, move, stop, get_fx, a stolen room) and every frame allocate nothing.
int run_library() {
  std::string bytes;
  {
    std::ostringstream os;
    if (!nfx::rollout::save_model(os, nfx::compose::testing::tiny_effect(3).m)) return 1;
    bytes = os.str();
  }
  nvfx_effect* et = nullptr;
  if (nvfx_effect_load_memory(bytes.data(), bytes.size(), &et) != NVFX_OK) return 1;
  const char* catalogue = R"(
effect tiny = "tiny.nvfx"
look hot = shader, heat_scale 1, emission 2, hue 200
entry blob, category fire, size 64 x 64, loop, about "a loop in shards"
param intensity = 0.5, range (0, 1), about "its control"
module m = tiny, size 32, width 48 * scale, at (X, Y), look hot, start 1, seed 7, controls (0.5, 0.5, 0.5), intensity intensity,
           shards 1
emit embers, on m, tries 1, chance 0.5, spread 10
at 0.2: scorch at (X, Y), radius 10 * scale, glow 0.5 * alive
entry pop, category explosions, size 64 x 64, anchor centre, once 0.6, about "a one-shot"
module m = tiny, size 32, width 40 * scale, at (X, Y + 20 * scale), start 0, seed 3
light flash (exp(-t / 0.1), 0, 0)
camera x (rand - 0.5) * 4 * exp(-t / 0.2), y 0
at 0:
  shock at (X, Y), speed 300
  burst at (X, Y), embers 20, debris 5, sparks 20, direction (1, 0), cone 40, speed 200
entry zap, category sparks, size 80 x 40, loop, about "glows"
emit arc, along (X - 30, Y - 10) to (X + 30, Y - 10), rate 10, seed 2, branches 2
emit ring, at (X, Y), radius 10 + 40 * t, squash 0.3
emit glow, at (X, Y - 20), radius 3, count 8, spread 20
field shine = lamp, at (X, Y - 10), radius 20, strength 0.5, colour (0.5, 0.7, 1)
)";
  const nvfx_library_file file{"alloc.nvfxl", catalogue};
  nvfx_library_desc ld;
  nvfx_library_desc_init(&ld);
  ld.files = &file;
  ld.n_files = 1;
  nvfx_library* lib = nullptr;
  nvfx_scene_error err;
  if (nvfx_library_open(&ld, &lib, &err) != NVFX_OK) {
    std::printf("library: %s\n", err.message);
    return 1;
  }
  int failures = 0;
  const nvfx_scene_effect fx{"tiny.nvfx", et};
  const nvfx_stage_room rooms[] = {{"blob", 2, 0}, {"pop", 2, 1}, {"zap", 1, 0}};
  for (const int overlap : {0, 1}) {
    nvfx_stage_desc sd;
    nvfx_stage_desc_init(&sd);
    sd.width = 160;
    sd.height = 96;
    sd.ground = 80;
    sd.rooms = rooms;
    sd.n_rooms = 3;
    sd.effects = &fx;
    sd.n_effects = 1;
    sd.overlap = overlap;
    nvfx_stage* st = nullptr;
    if (nvfx_stage_create(lib, &sd, &st, &err) != NVFX_OK) {
      std::printf("stage: %s\n", err.message);
      ++failures;
      continue;
    }
    nvfx_scene* scene = nvfx_stage_scene(st);
    std::vector<std::uint8_t> rgba(160 * 96 * 4);
    nvfx_scene_render(scene, rgba.data(), 160 * 4);  // warm-up outside the count
    g_allocations.store(0);
    g_counting.store(true);
    nvfx_fx a = 0, b = 0, z = 0;
    int spawned = 0;
    for (int f = 1; f < 150; ++f) {
      if (f == 2) spawned += nvfx_stage_spawn(st, "blob", 40.f, 80.f, 1.f, -1.0, &a) == NVFX_OK;
      if (f == 3) spawned += nvfx_stage_spawn(st, "zap", 120.f, 70.f, 1.f, 0.3, &z) == NVFX_OK;
      if (f % 9 == 4) {  // two rooms: the third and later spawns steal the oldest
        nvfx_fx p = 0;
        spawned += nvfx_stage_spawn(st, "pop", 80.f + static_cast<float>(f % 30), 40.f, 1.f, -1.0, &p) == NVFX_OK;
      }
      if (f == 20) nvfx_stage_set_param(st, a, "intensity", 0.9f);
      if (f == 25) nvfx_stage_move(st, a, 50.f, 80.f);
      if (f == 40) spawned += nvfx_stage_spawn(st, "blob", 100.f, 80.f, 1.2f, -1.0, &b) == NVFX_OK;
      if (f == 70) nvfx_stage_stop(st, a, -1.0);
      if (f == 90) spawned += nvfx_stage_spawn(st, "blob", 60.f, 80.f, 0.8f, -1.0, &a) == NVFX_OK;
      if (f == 120) nvfx_stage_stop(st, z, 4.5);
      nvfx_stage_fx_info info{};
      nvfx_stage_get_fx(st, b, &info);
      (void)nvfx_stage_free_rooms(st, "pop");
      nvfx_stage_set_camera(st, 0.f, 0.f);
      nvfx_scene_step_frames(scene, 1);
      if (f % 2 == 0) nvfx_scene_render(scene, rgba.data(), 160 * 4);
    }
    g_counting.store(false);
    const long n = g_allocations.load();
    std::printf("library stage%s: %ld allocations in 149 frames with %d spawns, moves, parameters and stops\n", overlap ? ", overlapped" : "", n, spawned);
    failures += n == 0 && spawned >= 18 ? 0 : 1;
    nvfx_stage_free(st);
  }
  nvfx_library_free(lib);
  nvfx_effect_free(et);
  return failures;
}

// The engine features (docs/ENGINES.md §11, docs/COMPOSE.md §5.3 and §4.6): a stage with the engine layers on, read every
// frame in every format; a trail whose instance is moved every frame along a path long enough for its pool of tiles to
// be taken again, stopped and spawned again; rain that washes a smoke module, snow that melts over it, ash and leaves.
int run_engine_features() {
  std::string bytes;
  {
    std::ostringstream os;
    if (!nfx::rollout::save_model(os, nfx::compose::testing::tiny_effect(3).m)) return 1;
    bytes = os.str();
  }
  nvfx_effect* et = nullptr;
  if (nvfx_effect_load_memory(bytes.data(), bytes.size(), &et) != NVFX_OK) return 1;
  const char* catalogue = R"(
effect tiny = "tiny.nvfx"
look gas = shader, heat_scale 1, emission 2, relief 2
entry streak, category trails, size 160 x 60, anchor centre, loop, moving (150, 0), about "a moving source"
param density = 0.5, range (0, 1), about "how much"
let lean = clamp(smoke.vx / 400, -0.3, 0.3)
module head = tiny, size 32, width 40 * scale, at (X, Y + 20 * scale), start 1, seed 7, wind 0.5 - lean
trail smoke = tiny, size 32, width 48, band 4, tiles 6, at (X, Y), look gas, hold 0.3, fade 0.3, soot density, heat 0.1,
      radius 5 * scale, kick -0.2, velocity (0, -0.3)
entry storm, category weather, size 160 x 96, anchor base, loop, about "weather over a smoke"
module puff = tiny, size 32, width 64, at (X, Y), look gas, start 1, seed 9
emit rain, in (X - 80, Y - 96), size 160 x 96, density 4, wind (60, 0), wash 0.5, seed 3
emit snow, in (X - 80, Y - 96), size 160 x 96, density 4, melt 0.05, seed 4
emit ash, in (X - 80, Y - 96), size 160 x 96, density 2, glowing 0.3, seed 5
emit leaves, in (X - 80, Y - 96), size 160 x 96, density 0.5, seed 6
)";
  const nvfx_library_file file{"engine.nvfxl", catalogue};
  nvfx_library_desc ld;
  nvfx_library_desc_init(&ld);
  ld.files = &file;
  ld.n_files = 1;
  nvfx_library* lib = nullptr;
  nvfx_scene_error err;
  if (nvfx_library_open(&ld, &lib, &err) != NVFX_OK) {
    std::printf("engine features: %s\n", err.message);
    return 1;
  }
  int failures = 0;
  const nvfx_scene_effect fx{"tiny.nvfx", et};
  const nvfx_stage_room rooms[] = {{"streak", 1, 1}, {"storm", 1, 0}};
  for (const int overlap : {0, 1}) {
    nvfx_stage_desc sd;
    nvfx_stage_desc_init(&sd);
    sd.width = 160;
    sd.height = 96;
    sd.ground = 90;
    sd.rooms = rooms;
    sd.n_rooms = 2;
    sd.effects = &fx;
    sd.n_effects = 1;
    sd.overlap = overlap;
    sd.layers = 1;
    nvfx_stage* st = nullptr;
    if (nvfx_stage_create(lib, &sd, &st, &err) != NVFX_OK) {
      std::printf("engine features stage: %s\n", err.message);
      ++failures;
      continue;
    }
    nvfx_scene* scene = nvfx_stage_scene(st);
    std::vector<std::uint8_t> rgba(160 * 96 * 4);
    std::vector<float> f32(160 * 96 * 4);
    nvfx_scene_render(scene, rgba.data(), 160 * 4);  // warm-up outside the count
    for (int l = 0; l < 4; ++l) nvfx_scene_render_layer(scene, static_cast<nvfx_scene_layer>(l), NVFX_FORMAT_RGBA32F, f32.data(), 160 * 16);
    g_allocations.store(0);
    g_counting.store(true);
    nvfx_fx a = 0, w = 0;
    int spawned = 0, layers = 0;
    spawned += nvfx_stage_spawn(st, "streak", 10.f, 50.f, 1.f, -1.0, &a) == NVFX_OK;
    spawned += nvfx_stage_spawn(st, "storm", 80.f, 90.f, 1.f, -1.0, &w) == NVFX_OK;
    for (int f = 1; f < 150; ++f) {
      const float t = static_cast<float>(f) / 30.f;
      nvfx_stage_move(st, a, 10.f + 60.f * std::fmod(t, 3.f), 50.f + 12.f * std::sin(2.f * t));  // a jump back every 3 s
      if (f == 100) nvfx_stage_stop(st, a, -1.0);
      if (f == 110) spawned += nvfx_stage_spawn(st, "streak", 20.f, 40.f, 1.f, -1.0, &a) == NVFX_OK;
      nvfx_scene_step_frames(scene, 1);
      const auto layer = static_cast<nvfx_scene_layer>(f % 4);
      if (f % 3 == 0) layers += nvfx_scene_render_layer(scene, layer, NVFX_FORMAT_RGBA8, rgba.data(), 160 * 4) == NVFX_OK;
      else if (f % 3 == 1) layers += nvfx_scene_render_layer(scene, layer, NVFX_FORMAT_RGBA16F, f32.data(), 160 * 8) == NVFX_OK;
      else layers += nvfx_scene_render_layer(scene, layer, NVFX_FORMAT_RGBA32F, f32.data(), 160 * 16) == NVFX_OK;
      nvfx_scene_render(scene, rgba.data(), 160 * 4);
    }
    g_counting.store(false);
    const long n = g_allocations.load();
    nvfx_scene_module_info mi{};
    nvfx_scene_module_get_info(scene, "s0_smoke", &mi);
    std::printf("engine features%s: %ld allocations in 149 frames (a trail moved every frame, %d of its tiles, %d spawns; weather; %d layers read)\n",
                overlap ? ", overlapped" : "", n, mi.tiles, spawned, layers);
    failures += n == 0 && spawned == 3 && layers == 149 ? 0 : 1;
    nvfx_stage_free(st);
  }
  nvfx_library_free(lib);
  nvfx_effect_free(et);
  return failures;
}

// The procedural elements of the library's third round (docs/COMPOSE.md §4.10): a stage with the engine layers on, read
// every frame, playing hail, an aurora, a rainbow, meteors, a beam, a shield hit again and again, glints, and bursts of
// spatter and shards, spawned, stopped and spawned again.
int run_fx_elements() {
  const char* catalogue = R"(
entry pellets, category weather, size 160 x 96, anchor base, loop, about "hail"
emit hail, in (X - 80, Y - 96), size 160 x 96, density 2, seed 6
entry sky, category weather, size 160 x 96, anchor base, loop, about "light"
emit aurora, in (X - 80, Y - 96), size 160 x 50, seed 7
emit rainbow, at (X, Y), radius 70, count 2
emit meteors, in (X - 80, Y - 96), size 100 x 20, rate 5, seed 8
emit glints, in (X - 80, Y - 30), size 160 x 20, count 20, rate 2, seed 9
entry hit, category combat, size 160 x 96, anchor base, once 1.2, show 0.3, about "a beam, a shield, bursts"
emit beam, along (X - 70, Y - 50) to (X + 60, Y - 40), seed 10
emit shield, at (X, Y - 48), radius 40, rate 3, hit t - 0.1, angle 180, seed 11
at 0, repeat:
  burst at (X, Y - 30), radius 4, embers 0, spatter 6, shards 3, speed 160, stain 1, if rand < 0.3
)";
  const nvfx_library_file file{"fx.nvfxl", catalogue};
  nvfx_library_desc ld;
  nvfx_library_desc_init(&ld);
  ld.files = &file;
  ld.n_files = 1;
  nvfx_library* lib = nullptr;
  nvfx_scene_error err;
  if (nvfx_library_open(&ld, &lib, &err) != NVFX_OK) {
    std::printf("fx elements: %s\n", err.message);
    return 1;
  }
  int failures = 0;
  const nvfx_stage_room rooms[] = {{"pellets", 1, 0}, {"sky", 1, 0}, {"hit", 2, 1}};
  for (const int overlap : {0, 1}) {
    nvfx_stage_desc sd;
    nvfx_stage_desc_init(&sd);
    sd.width = 160;
    sd.height = 96;
    sd.ground = 92;
    sd.rooms = rooms;
    sd.n_rooms = 3;
    sd.overlap = overlap;
    sd.layers = 1;
    nvfx_stage* st = nullptr;
    if (nvfx_stage_create(lib, &sd, &st, &err) != NVFX_OK) {
      std::printf("fx elements stage: %s\n", err.message);
      ++failures;
      continue;
    }
    nvfx_scene* scene = nvfx_stage_scene(st);
    std::vector<std::uint8_t> rgba(160 * 96 * 4);
    std::vector<float> f32(160 * 96 * 4);
    nvfx_scene_render(scene, rgba.data(), 160 * 4);  // warm-up outside the count
    for (int l = 0; l < 4; ++l) nvfx_scene_render_layer(scene, static_cast<nvfx_scene_layer>(l), NVFX_FORMAT_RGBA32F, f32.data(), 160 * 16);
    g_allocations.store(0);
    g_counting.store(true);
    nvfx_fx w = 0, s = 0, h = 0;
    int spawned = 0, layers = 0;
    spawned += nvfx_stage_spawn(st, "pellets", 80.f, 92.f, 1.f, -1.0, &w) == NVFX_OK;
    spawned += nvfx_stage_spawn(st, "sky", 80.f, 92.f, 1.f, -1.0, &s) == NVFX_OK;
    for (int f = 1; f < 150; ++f) {
      if (f % 25 == 1) spawned += nvfx_stage_spawn(st, "hit", 60.f + static_cast<float>(f % 40), 92.f, 1.f, -1.0, &h) == NVFX_OK;
      if (f == 70) nvfx_stage_stop(st, w, -1.0);
      if (f == 90) spawned += nvfx_stage_spawn(st, "pellets", 70.f, 92.f, 0.8f, -1.0, &w) == NVFX_OK;
      if (f == 100) nvfx_stage_stop(st, s, -1.0);
      nvfx_scene_step_frames(scene, 1);
      const auto layer = static_cast<nvfx_scene_layer>(f % 4);
      layers += nvfx_scene_render_layer(scene, layer, f % 2 ? NVFX_FORMAT_RGBA8 : NVFX_FORMAT_RGBA16F, f % 2 ? static_cast<void*>(rgba.data()) : f32.data(),
                                        f % 2 ? 160 * 4 : 160 * 8) == NVFX_OK;
      nvfx_scene_render(scene, rgba.data(), 160 * 4);
    }
    g_counting.store(false);
    const long n = g_allocations.load();
    std::printf("fx elements%s: %ld allocations in 149 frames (hail, sky and hits; %d spawns; %d layers read)\n", overlap ? ", overlapped" : "", n, spawned, layers);
    failures += n == 0 && spawned == 9 && layers == 149 ? 0 : 1;
    nvfx_stage_free(st);
  }
  nvfx_library_free(lib);
  return failures;
}

// The effect manager: everything is made by nvfx_manager_create and nvfx_manager_add_model; 400 ticks at 60 Hz of a
// crowd of two looping effects and a one-shot, with every call a game makes between ticks, allocate nothing. Run with
// its own threads (2), and with the engine running the jobs (here both job slots on the calling thread).
int run_manager(bool external) {
  std::string loop_bytes, shot_bytes;
  for (const bool loop : {true, false}) {
    nfx::rollout::Hyper h;
    h.res = 16;
    h.hidden = 8;
    h.memory = 2;
    h.jacobi = 10;
    h.render_hidden = 6;
    h.warmup = 4;
    if (!loop) {
      h.n_age = 2;
      h.frames = 50;
    }
    nfx::rollout::Model m = nfx::rollout::init_model(h, loop ? 3 : 4);
    m.effect = "rollout";
    m.loop = loop;
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
    std::ostringstream os;
    if (!nfx::rollout::save_model(os, m)) return 1;
    (loop ? loop_bytes : shot_bytes) = os.str();
  }
  nvfx_effect* fl = nullptr;
  nvfx_effect* fs = nullptr;
  if (nvfx_effect_load_memory(loop_bytes.data(), loop_bytes.size(), &fl) != NVFX_OK) return 1;
  if (nvfx_effect_load_memory(shot_bytes.data(), shot_bytes.size(), &fs) != NVFX_OK) return 1;
  nvfx_manager_desc d;
  nvfx_manager_desc_init(&d);
  d.threads = external ? 0 : 2;
  d.job_slots = 2;
  d.max_instances = 48;
  d.budget_ms = 1.5f;
  d.proxy_dwell = 0.3f;
  d.min_dwell = 0.2f;
  d.park_after = 0.3f;
  d.cull_after = 2.f;
  d.interp = NVFX_INTERP_MOTION;
  nvfx_manager* g = nullptr;
  if (nvfx_manager_create(&d, &g) != NVFX_OK) return 1;
  nvfx_manager_model_desc md;
  nvfx_manager_model_desc_init(&md);
  md.max_sims = 24;
  md.shard_seconds = 2.f;
  int ml = -1, ms = -1;
  if (nvfx_manager_add_model(g, fl, &md, &ml) != NVFX_OK || nvfx_manager_add_model(g, fs, &md, &ms) != NVFX_OK) return 1;
  const float pos0[3] = {0.f, 0.f, 0.f}, fwd[3] = {0.f, 0.f, 1.f}, up[3] = {0.f, 1.f, 0.f};
  nvfx_manager_set_camera(g, pos0, fwd, up, 1.f, 640.f, 360.f);
  std::array<nvfx_handle, 48> hs{};
  int n = 0;
  const auto add = [&](int model, std::uint64_t seed, float x, float zz, float offset) {
    nvfx_spawn_desc s;
    nvfx_spawn_desc_init(&s);
    const float c[3] = {0.3f + 0.1f * static_cast<float>(seed % 3), 0.5f, 0.6f};
    s.model = model;
    s.controls = c;
    s.n_controls = 3;
    s.seed = seed % 4;
    s.time_offset = offset;
    s.position[0] = x;
    s.position[2] = zz;
    s.world_size = 2.f;
    nvfx_handle h = 0;
    if (n < 48 && nvfx_manager_spawn(g, &s, &h) == NVFX_OK) hs[static_cast<std::size_t>(n++)] = h;
  };
  for (int i = 0; i < 20; ++i) add(ml, static_cast<std::uint64_t>(i), -10.f + static_cast<float>(i), 4.f + static_cast<float>(i % 7) * 3.f, i % 5 == 4 ? 0.3f : 0.f);
  const auto tick = [&](int t) {
    const float cam[3] = {2.f * std::sin(0.01f * static_cast<float>(t)), 0.f, -2.f + 0.02f * static_cast<float>(t % 200)};
    nvfx_manager_set_camera(g, cam, fwd, up, 1.f, 640.f, 360.f);
    if (t % 37 == 5) add(ms, static_cast<std::uint64_t>(t), static_cast<float>(t % 9) - 4.f, 6.f, 0.f);
    if (t % 53 == 20 && n > 25) {
      nvfx_manager_stop(g, hs[static_cast<std::size_t>(n - 1)]);
      --n;
    }
    if (t % 41 == 7) {
      const float c[3] = {0.8f, 0.3f + 0.001f * static_cast<float>(t), 0.5f};
      nvfx_manager_set_controls(g, hs[static_cast<std::size_t>(t % 20)], c, 3);
    }
    if (t % 29 == 3) nvfx_manager_set_priority(g, hs[static_cast<std::size_t>(t % 20)], 1.f + static_cast<float>(t % 4));
    if (t % 31 == 0) nvfx_manager_set_screen_size(g, hs[static_cast<std::size_t>(t % 20)], static_cast<float>(t % 140), t % 3 != 0);
    if (external) {
      int jobs = 0;
      nvfx_manager_tick_begin(g, 1.0 / 60.0, &jobs);
      nvfx_manager_run_jobs(g, 0);
      nvfx_manager_run_jobs(g, 1);
      nvfx_manager_tick_end(g);
    } else {
      nvfx_manager_tick(g, 1.0 / 60.0);
    }
    for (int i = 0; i < n; ++i) {
      nvfx_manager_picture p;
      nvfx_manager_get_picture(g, hs[static_cast<std::size_t>(i)], &p);
    }
    nvfx_manager_stats st;
    nvfx_manager_get_stats(g, &st);
    return st;
  };
  tick(0);  // the first tick outside the count
  g_allocations = 0;
  g_counting = true;
  int proxied = 0, skipped = 0, paused = 0, interpolated = 0;
  for (int t = 1; t < 400; ++t) {
    const nvfx_manager_stats st = tick(t);
    proxied = std::max(proxied, st.proxied);
    skipped += st.skipped;
    paused = std::max(paused, st.sims_paused);
    interpolated += st.interpolated;
  }
  g_counting = false;
  const long a = g_allocations.load();
  std::printf("effect manager, %s: %ld allocations in 399 ticks (most proxied %d, skipped updates %d, most paused %d, interpolated pictures %d)\n",
              external ? "jobs on the engine's thread" : "2 threads", a, proxied, skipped, paused, interpolated);
  nvfx_manager_free(g);
  nvfx_effect_free(fl);
  nvfx_effect_free(fs);
  return a == 0 && proxied > 0 && interpolated > 0 ? 0 : 1;
}

}  // namespace

// Study N (docs/STUDY_3D.md): a 3D volume (nvfx_volume.h) on 1 and 3 threads, in shards with crossfades, stepped, read
// as float and half textures in every field, drawn for a moving camera, seeded again and stepped back (restarts).
int run_volume(int threads) {
  nfx::rollout3d::Hyper h;
  h.rx = 8;
  h.ry = 12;
  h.rz = 8;
  h.fine = 2;
  h.hidden = 8;
  h.memory = 2;
  h.jacobi = 8;
  h.warmup = 4;
  nfx::rollout3d::Model m = nfx::rollout3d::init_model(h, 4);
  m.effect = "volume";
  m.scale = {0.08f, 0.15f, 0.08f, 0.4f, 0.06f};
  m.lo = {-0.6f, -0.6f, -0.6f, 0.f, 0.f};
  m.hi = {0.6f, 0.9f, 0.6f, 2.f, 0.5f};
  for (int k = 0; k < 3; ++k) {
    nfx::rollout3d::StartPoint sp;
    sp.controls = {0.2f + 0.3f * static_cast<float>(k), 0.5f, 0.5f};
    sp.seed = 9'000'600 + static_cast<std::uint64_t>(k);
    sp.time = 3.f;
    sp.coarse.resize(static_cast<std::size_t>(h.cells()) * nfx::rollout3d::kPhys);
    for (std::size_t i = 0; i < sp.coarse.size(); ++i) {
      const float v = 0.5f + 0.5f * std::sin(0.37f * static_cast<float>(i) + static_cast<float>(k));
      sp.coarse[i] = i % 5 < 3 ? 0.1f * (v - 0.5f) : (i % 5 == 3 ? 1.2f * v : 0.1f * v);
    }
    m.starts.push_back(std::move(sp));
  }
  std::ostringstream os;
  if (!nfx::rollout3d::save_model(os, m)) return 1;
  const std::string bytes = os.str();
  nvfx_volume_effect* e = nullptr;
  nvfx_volume* v = nullptr;
  if (nvfx_volume_effect_load_memory(bytes.data(), bytes.size(), &e) != NVFX_OK) return 1;
  if (nvfx_volume_create(e, threads, &v) != NVFX_OK) return 1;
  const float controls[3] = {0.6f, 0.3f, 0.8f};
  nvfx_volume_set_controls(v, controls, 3);
  nvfx_volume_set_seed(v, 99);
  if (nvfx_volume_set_shard(v, 1.5f) != NVFX_OK) return 1;
  const std::size_t n = static_cast<std::size_t>(h.fx()) * h.fy() * h.fz();
  std::vector<float> field(3 * n);
  std::vector<std::uint16_t> half(3 * n);
  std::vector<std::uint8_t> rgba(64 * 48 * 4);
  nvfx_volume_camera cam{};
  const auto frame = [&](double t, int f) {
    long bad = 0;
    bad += nvfx_volume_step_to(v, t) != NVFX_OK;
    for (const nvfx_volume_field fld : {NVFX_VOLUME_TEMPERATURE, NVFX_VOLUME_SOOT, NVFX_VOLUME_EXTINCTION, NVFX_VOLUME_VELOCITY}) {
      bad += nvfx_volume_read(v, fld, NVFX_VOLUME_FLOAT, field.data(), field.size() * 4) != NVFX_OK;
      bad += nvfx_volume_read(v, fld, NVFX_VOLUME_HALF, half.data(), half.size() * 2) != NVFX_OK;
    }
    const float a = 0.1f * static_cast<float>(f);
    const nvfx_volume_look_at la{{2.f * std::sin(a), 1.f, 2.f * std::cos(a)}, {0.f, 0.7f, 0.f}, {0.f, 1.f, 0.f}, 45.f, 64.f / 48.f, 0.f, 0.f};
    bad += nvfx_volume_camera_look_at(&la, &cam) != NVFX_OK;
    bad += nvfx_volume_render(v, &cam, 64, 48, rgba.data(), 64 * 4) != NVFX_OK;
    return bad;
  };
  long bad = frame(0.0, 0);  // warm-up outside the count
  g_allocations = 0;
  g_counting = true;
  for (int f = 1; f <= 80; ++f) bad += frame(f / 30.0, f);  // across a shard change and its crossfade
  nvfx_volume_set_seed(v, 7);                                 // a new seed: the timeline starts again
  bad += frame(0.5, 81);
  bad += frame(0.2, 82);  // back in time: a restart
  nvfx_volume_set_variation(v, 2);
  bad += frame(1.0, 83);
  g_counting = false;
  const long a = g_allocations.load();
  std::printf("volume (3D, study N) on %d thread%s: %ld allocations in 84 frames%s\n", threads, threads == 1 ? "" : "s", a, bad ? " (calls failed)" : "");
  nvfx_volume_free(v);
  nvfx_volume_effect_free(e);
  return a == 0 && bad == 0 ? 0 : 1;
}

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
  failures += run_library();
  failures += run_engine_features();
  failures += run_fx_elements();  // the procedural elements of the library's third round
  failures += run_rollout_prior(64) + run_rollout_prior(128);  // the prior against drift
  failures += run_rollout(64, true);                             // the simulator's look for the first frames
  failures += run_rollout(64, false, true);                      // study D2's advection gain and softening
  failures += run_rollout(64, false, false, true) + run_rollout(128, false, false, true) + run_rollout(64, false, false, true, true);  // study K
  failures += run(g, 128, "grid", 8) + run(g, 128, "grid", 5) + run(g, 64, "grid", 4) + run(c, 64, "conv", 4);  // packed features
  failures += run(g, 128, "grid", 8, 6) + run(c, 64, "conv", 8, 8);  // vector-quantised features
  failures += run_per_plane(g, 128, "grid", false) + run_per_plane(c, 64, "conv", false);  // a width per feature plane
  failures += run_per_plane(g, 128, "grid", true) + run_per_plane(g, 64, "grid", true);       // sparse features
  failures += run_multi(128, false) + run_multi(64, true) + run_multi(128, true);                // multi-level models (study F4), int8
  failures += run_multi(128, false, NVFX_PRECISION_FLOAT) + run_multi(64, true, NVFX_PRECISION_FLOAT);  // and float
  failures += run_manager(false) + run_manager(true);                                                    // the effect manager
  failures += run_volume(1) + run_volume(3);                                                             // 3D volumes (study N)
  std::printf("%s\n", failures ? "FAILED: nvfx_render allocated" : "ok: no allocation per frame");
  return failures ? 1 : 0;
}
