// nvfx_render must not allocate (docs/PLAN.md §5.4). This program replaces the global operator new to count heap
// allocations, builds small grid and conv effects in memory, and renders 200 frames of each with every feature
// switched on (controls, seeded drift, colour). Exit code 0 = no allocation during rendering.
#include <neuralfx/model.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <sstream>
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
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

int run(nfx::Hyper h, int size, const char* name) {
  nfx::Model m = nfx::init_model(h, 1);
  m.effect = name;
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
  std::printf("%s %dx%d: %ld allocations in 200 frames\n", name, size, size, n);
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
  int failures = run(g, 64, "grid") + run(g, 128, "grid") + run(c, 64, "conv") + run(c, 32, "conv") + run_rollout(64) + run_rollout(128);
  std::printf("%s\n", failures ? "FAILED: nvfx_render allocated" : "ok: no allocation per frame");
  return failures ? 1 : 0;
}
