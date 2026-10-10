// nvfx_fireball: a scripted scene of composed effects (docs/COMPOSE.md), rendered to video and profiled.
//
//   nvfx_fireball --models DIR [--out fireball.mp4] [--width 1280 --height 720] [--quality 1] [--threads 4]
//                 [--isa avx2|avx512|baseline] [--seconds 9] [--profile profile.csv] [--keyframes DIR]
//                 [--sheet sheet.png] [--no-video]
//
// DIR holds the rollout effects fire.nvfx, smoke.nvfx and explosion.nvfx (nvfx_experiment d-train). The scene: a
// burning wreck at night; a fuse runs to a charge; the charge explodes. Six tiles of the explosion model make one
// domain larger than the model was trained on (blend_band); after 1.5 s the smoke model takes the cloud over
// (hand_over) and grows a smoke column from the crater; the blast bends the wreck's fire (push) and sets the wreck
// off (a trigger on the shock front); the second blast's cloud joins the first (transfer), and the wreck's smoke
// joins the sky above it; embers that land hot light new fires. Light from everything hot lights the smoke and the
// ground; the shock front and hot air bend what is seen through them.
//
// Every stage is timed per frame (--profile), heap allocations during the frame loop are counted (this program
// replaces operator new), and the video and keyframes go to paths outside git.
#include "compose.hpp"

#include <neuralfx/image_io.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <memory>
#include <new>
#include <print>
#include <string>
#include <sys/resource.h>
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

using namespace nfx;
using namespace nfx::compose;
using Clock = std::chrono::steady_clock;

double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
float smooth(float t) {
  t = std::clamp(t, 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}

struct Args {
  std::filesystem::path models, out = "fireball.mp4", profile, keyframes, sheet;
  int width = 1280, height = 720, threads = 4;
  float quality = 1.f, seconds = 9.f;
  std::string isa;
  bool video = true;
};

Args parse(int argc, char** argv) {
  Args a;
  if (const char* d = std::getenv("NEURALVFX_DATA")) a.models = std::filesystem::path(d) / "experiments" / "models" / "d";
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(k + " needs a value");
      return argv[++i];
    };
    if (k == "--models") a.models = next();
    else if (k == "--out") a.out = next();
    else if (k == "--profile") a.profile = next();
    else if (k == "--keyframes") a.keyframes = next();
    else if (k == "--sheet") a.sheet = next();
    else if (k == "--width") a.width = std::stoi(next());
    else if (k == "--height") a.height = std::stoi(next());
    else if (k == "--threads") a.threads = std::stoi(next());
    else if (k == "--quality") a.quality = std::stof(next());
    else if (k == "--seconds") a.seconds = std::stof(next());
    else if (k == "--isa") a.isa = next();
    else if (k == "--no-video") a.video = false;
    else throw std::invalid_argument("unknown option " + k + " (see the source header)");
  }
  if (a.models.empty()) throw std::invalid_argument("--models DIR (or NEURALVFX_DATA) is needed");
  return a;
}

rt::RolloutEffect load(const std::filesystem::path& p) {
  auto m = rollout::load_model(p);
  if (!m) throw std::runtime_error(std::format("{}: {}", p.string(), m.error()));
  rt::RolloutEffect e;
  e.m = std::move(*m);
  e.stored_bytes = e.m.storage_bytes();
  return e;
}

int round32(float v) { return std::max(32, 32 * static_cast<int>(std::lround(v / 32.f))); }

// A rule of the script: when `when` first holds, `then` runs (once).
struct Rule {
  std::string name;
  std::function<bool(float)> when;
  std::function<void(float)> then;
  float fired_at = -1.f;
};

// Per-frame stage times (ms).
enum Stage { kScript, kStep, kCouple, kBus, kLight, kPartUpdate, kShade, kBackground, kDraw, kPartDraw, kDistort, kBloom, kFinish, kEncode, kStages };
constexpr const char* kStageNames[kStages] = {"script", "step", "couple", "bus", "light", "particles_update", "shade",
                                              "background", "draw", "particles_draw", "distort", "bloom", "finish", "encode"};

}  // namespace

int main(int argc, char** argv) try {
  const Args A = parse(argc, argv);
  if (!A.isa.empty()) {
    const nvfx_isa want = A.isa == "avx512" ? NVFX_ISA_AVX512 : A.isa == "avx2" ? NVFX_ISA_AVX2 : NVFX_ISA_BASELINE;
    if (nvfx_set_isa(want) != NVFX_OK) throw std::runtime_error("this CPU cannot run --isa " + A.isa);
  }
  const Isa isa = best_isa();
  const auto setup0 = Clock::now();
  rt::RolloutEffect EX = load(A.models / "explosion.nvfx"), SM = load(A.models / "smoke.nvfx"), FI = load(A.models / "fire.nvfx");
  // Scripted settings of the models' detail layers (the script may set any of them): at this size the cloud wants
  // larger, stronger sub-grid eddies and fully broken-up new material.
  EX.m.detail.swirl = 2.2f;
  EX.m.detail.swirl_scale = 9.f;
  EX.m.detail.contrast = 1.f;
  SM.m.detail.swirl = 1.4f;
  SM.m.detail.swirl_scale = 9.f;

  // --- the world: reference coordinates for 1280 x 720, scaled by u; render sizes scaled by quality ----------------
  const float u = static_cast<float>(A.width) / 1280.f, q = A.quality;
  const float W = [&](float v) { return v; }(0.f);
  (void)W;
  const float ground = 600.f * u;
  const int T = round32(384.f * q * u);           // main tiles: render size
  const float tile = 576.f * u, sc = tile / static_cast<float>(T), stride = 432.f * u;  // world size, scale, step
  constexpr int kBand = 8;
  std::vector<std::unique_ptr<Module>> owned;
  const auto make = [&](std::string name, const rt::RolloutEffect& e, int size, Placement p) {
    owned.push_back(std::make_unique<Module>(std::move(name), e, size, p, isa));
    return owned.back().get();
  };
  std::array<std::array<Module*, 3>, 2> ex{}, sm{};
  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 3; ++c) {
      const Placement p{(-80.f * u) + stride * static_cast<float>(c), 108.f * u - stride * static_cast<float>(r), sc};
      ex[zs(r)][zs(c)] = make(std::format("explosion_r{}c{}", r, c), EX, T, p);
      sm[zs(r)][zs(c)] = make(std::format("smoke_r{}c{}", r, c), SM, T, p);
      for (Module* m : {ex[zs(r)][zs(c)], sm[zs(r)][zs(c)]}) {
        m->group = 0;
        m->band = {c > 0 ? kBand : 0, c < 2 ? kBand : 0, r > 0 ? kBand : 0, r < 1 ? kBand : 0};
        m->look = Look::shader;
      }
    }
  }
  const int T2 = round32(256.f * q * u);
  const float sec_world = 320.f * u;
  Module* sec = make("explosion_wreck", EX, T2, {1010.f * u - 0.5f * sec_world, 560.f * u + 0.32f * sec_world - sec_world, sec_world / static_cast<float>(T2)});
  sec->group = 1;
  sec->look = Look::shader;
  sec->feather = 0.12f * static_cast<float>(T2);
  const int TF = round32(192.f * q * u), TF2 = round32(128.f * q * u);
  const float fw = 192.f * u, fw2 = 144.f * u;
  Module* wreck = make("fire_wreck", FI, TF, {1010.f * u - 0.5f * fw, ground + 0.08f * fw - fw, fw / static_cast<float>(TF)});
  wreck->group = 2;
  wreck->feather = 0.15f * static_cast<float>(TF);
  std::array<Module*, 2> fires{make("fire_a", FI, TF2, {0, 0, fw2 / static_cast<float>(TF2)}), make("fire_b", FI, TF2, {0, 0, fw2 / static_cast<float>(TF2)})};
  for (int i = 0; i < 2; ++i) {
    fires[zs(i)]->group = 3 + i;
    fires[zs(i)]->feather = 0.15f * static_cast<float>(TF2);
  }
  // shader looks: hot gas glows, soot absorbs and is lit
  ShaderSpec cloud;
  cloud.heat_scale = 1.6f;
  cloud.emission = 1.6f;
  cloud.emission_power = 3.0f;
  cloud.soot_density = 3.0f;
  cloud.relief = 4.f;
  cloud.soot_albedo = 0.3f;
  cloud.sky = 0.18f;
  cloud.shadow = 0.3f;
  cloud.scene_light = 0.6f;
  cloud.tint = {1.f, 0.93f, 0.86f};
  for (auto& row : ex)
    for (Module* m : row) m->spec = cloud;
  ShaderSpec smoke = cloud;
  smoke.tint = {0.92f, 0.92f, 0.95f};
  smoke.soot_density = 4.0f;
  smoke.soot_albedo = 0.45f;
  smoke.emission = 0.6f;
  smoke.heat_scale = 2.5f;
  smoke.emission_power = 3.f;
  smoke.sky = 0.3f;
  smoke.relief = 2.5f;
  for (auto& row : sm)
    for (Module* m : row) m->spec = smoke;
  sec->spec = cloud;

  // fires are started now, so their warm-up is not in the frame loop; they wait (not stepped) until lit
  wreck->controls = {0.62f, 0.5f, 0.55f};
  wreck->start(0, 4242);
  for (int i = 0; i < 2; ++i) {
    fires[zs(i)]->controls = {0.35f, 0.5f, 0.5f};
    fires[zs(i)]->start(i == 0 ? 0 : 6, 777u + static_cast<std::uint64_t>(i));
    fires[zs(i)]->active = false;
  }

  // the field bus over the whole world the scene uses
  const float bus_cell = 8.f * u;
  const float bx0 = -96.f * u, by0 = -340.f * u;
  FieldBus bus(bx0, by0, static_cast<int>(std::ceil((1392.f * u) / bus_cell)), static_cast<int>(std::ceil((1080.f * u) / bus_cell)), bus_cell, 6);
  Light light(bus);
  Particles parts(12000);
  Frame frame(A.width, A.height);
  Pool pool(A.threads);
  frame.ground_y = ground;
  std::vector<Shock> shocks;
  shocks.reserve(4);
  std::vector<std::array<float, 4>> scorch;
  scorch.reserve(4);

  // --- the script ----------------------------------------------------------------------------------------------------
  const float t_det = 1.2f, t_hand = t_det + 2.4f, t_end = A.seconds;
  const float burst_x = 640.f * u, burst_y = 684.f * u - 0.32f * tile;
  float t_sec = -1.f, sec_x = 1010.f * u, sec_y = 560.f * u;
  int lit = 0;
  std::array<float, 2> lit_at{};
  const int sp_main = 9;  // the strongest burst among the start points (intensity 0.98)
  const std::vector<float> main_controls = EX.m.starts[sp_main].controls;
  const auto burst = [&](float x, float y, float r, int embers, int debris, float speed) {
    for (int i = 0; i < embers; ++i) {
      const float a = parts.uniform() * 6.2831853f, d = std::sqrt(parts.uniform()) * r;
      const float s = speed * (0.3f + 0.7f * parts.uniform());
      parts.spawn(Kind::ember, x + d * std::cos(a), y + d * std::sin(a), s * std::cos(a), s * std::sin(a) - 0.45f * speed, 0.8f + 0.4f * parts.uniform(),
                  0.7f + 1.1f * parts.uniform(), 2.f + 3.f * parts.uniform());
    }
    for (int i = 0; i < debris; ++i) {
      const float a = 3.1415927f + parts.uniform() * 3.1415927f, s = speed * (0.4f + 0.9f * parts.uniform());
      parts.spawn(Kind::debris, x, y, s * std::cos(a), s * std::sin(a), 0.5f + 0.4f * parts.uniform(), 2.f + 3.f * parts.uniform(), 6.f);
    }
  };
  std::vector<Rule> rules;
  rules.push_back({"detonate", [&](float t) { return t >= t_det; }, [&](float t) {
                     for (int r = 0; r < 2; ++r) {
                       for (int c = 0; c < 3; ++c) {
                         Module* m = ex[zs(r)][zs(c)];
                         m->controls = main_controls;
                         if (r == 0 && c == 1) m->start(sp_main, 9001);
                         else m->start_empty(EX.m.starts[sp_main].time, 9001 + static_cast<std::uint64_t>(10 * r + c));
                       }
                     }
                     shocks.push_back({burst_x, burst_y, t, 1500.f * u, 0.3f, 7.f * u, 30.f * u});
                     scorch.push_back({burst_x, ground + 6.f * u, 190.f * u, 1.f});
                     burst(burst_x, burst_y, 70.f * u, 1400, 160, 900.f * u);
                   }});
  rules.push_back({"wreck explodes when the shock front reaches it", [&](float t) {
                     return !shocks.empty() && shocks[0].radius(t) >= std::hypot(sec_x - burst_x, sec_y - burst_y);
                   },
                   [&](float t) {
                     t_sec = t;
                     sec->controls = EX.m.starts[0].controls;
                     sec->start(0, 31337);
                     shocks.push_back({sec_x, sec_y, t, 900.f * u, 0.25f, 4.f * u, 22.f * u});
                     scorch.push_back({sec_x, ground + 4.f * u, 95.f * u, 0.8f});
                     burst(sec_x, sec_y, 35.f * u, 450, 60, 650.f * u);
                   }});
  rules.push_back({"the smoke model takes the cloud over", [&](float t) { return t >= t_hand; }, [&](float) {
                     for (int r = 0; r < 2; ++r) {
                       for (int c = 0; c < 3; ++c) {
                         sm[zs(r)][zs(c)]->controls = {0.75f, 0.56f, 0.55f};
                         sm[zs(r)][zs(c)]->seed = 5150 + static_cast<std::uint64_t>(10 * r + c);
                         sm[zs(r)][zs(c)]->take_over(*ex[zs(r)][zs(c)]);
                         ex[zs(r)][zs(c)]->active = false;
                       }
                     }
                   }});

  std::vector<Module*> drawn;  // back to front
  for (auto& row : ex)
    for (Module* m : row) drawn.push_back(m);
  for (auto& row : sm)
    for (Module* m : row) drawn.push_back(m);
  drawn.push_back(sec);
  drawn.push_back(wreck);
  drawn.push_back(fires[0]);
  drawn.push_back(fires[1]);
  std::vector<Module*> main_tiles;  // whichever set is active receives transfers
  main_tiles.reserve(12);
  std::vector<Module*> active;
  active.reserve(owned.size());

  // --- output --------------------------------------------------------------------------------------------------------
  std::FILE* video = nullptr;
  if (A.video) {
    const std::string cmd = std::format("ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgb24 -s {}x{} -r 30 -i - -c:v libx264 -preset slow -crf 19 "
                                        "-pix_fmt yuv420p -movflags +faststart \"{}\"",
                                        A.width, A.height, A.out.string());
    video = popen(cmd.c_str(), "w");
    if (!video) throw std::runtime_error("cannot start ffmpeg");
  }
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(A.width) * static_cast<std::size_t>(A.height) * 3);
  const std::array<float, 8> key_times = {0.9f, 1.27f, 1.5f, 2.0f, 2.9f, 4.2f, 6.0f, 8.2f};
  std::vector<Image> keys;
  keys.reserve(key_times.size());
  const int frames = static_cast<int>(std::lround(t_end * 30.f));
  std::vector<std::array<double, kStages>> prof(static_cast<std::size_t>(frames));
  std::vector<std::vector<std::pair<double, double>>> mod_prof(static_cast<std::size_t>(frames), std::vector<std::pair<double, double>>(owned.size()));
  std::vector<int> n_active(static_cast<std::size_t>(frames)), n_parts(static_cast<std::size_t>(frames));
  std::vector<long> allocs(static_cast<std::size_t>(frames));
  const double setup_ms = ms(setup0, Clock::now());
  std::size_t scratch = 0, resident = 0;
  for (const auto& m : owned) scratch += m->runner().scratch_bytes();
  for (const auto* e : {&EX, &SM, &FI}) {
    std::size_t n = (e->m.step_w.size() + e->m.render_w.size()) * 4;
    for (const auto& s : e->m.starts) n += (s.coarse.size() + s.fine_t.size() + s.fine_d.size()) * 4;
    resident += n;
  }
  std::println("nvfx_fireball: {}x{} at quality {} ({} threads, {}); main tiles {} px, {} modules; setup {:.0f} ms", A.width, A.height, q, pool.threads(),
               isa_name(isa), T, owned.size(), setup_ms);

  for (int f = 0; f < frames; ++f) {
    const float t = static_cast<float>(f) / 30.f;
    if (f == 1) g_counting.store(true);  // the first frame may still touch lazily sized buffers
    const long alloc0 = g_allocations.load();
    std::array<double, kStages>& P = prof[zs(f)];
    auto c0 = Clock::now();
    // script: rules, then controls and the scene's time-driven settings
    for (Rule& r : rules) {
      if (r.fired_at < 0.f && r.when(t)) {
        r.then(t);
        r.fired_at = t;
      }
    }
    if (t >= 0.2f && t < t_det) {  // the fuse
      const float s = (t - 0.2f) / (t_det - 0.2f);
      const float x = (250.f + (640.f - 250.f) * s) * u;
      for (int i = 0; i < 22; ++i) {
        parts.spawn(Kind::spark, x, ground - 2.f * u, (parts.uniform() - 0.5f) * 160.f * u, -(60.f + 240.f * parts.uniform()) * u, 1.f, 1.2f, 0.2f + 0.35f * parts.uniform());
      }
    }
    for (Module* fm : {wreck, fires[0], fires[1]}) {  // fires shed embers
      if (!fm->active) continue;
      const float cx = fm->at.x + 0.5f * static_cast<float>(fm->size()) * fm->at.scale;
      for (int i = 0; i < 2; ++i) {
        if (parts.uniform() < 0.6f) {
          parts.spawn(Kind::ember, cx + (parts.uniform() - 0.5f) * 40.f * u, ground - 20.f * u, (parts.uniform() - 0.5f) * 30.f * u, -(60.f + 90.f * parts.uniform()) * u,
                      0.7f, 0.6f + 0.5f * parts.uniform(), 1.5f + 1.5f * parts.uniform());
        }
      }
    }
    if (t > t_det + 2.f) {  // soot flakes fall out of the cloud
      for (int i = 0; i < 40; ++i) {
        const float x = (parts.uniform() * 1280.f) * u, y = (-200.f + parts.uniform() * 700.f) * u;
        if (bus.at(x, y).soot > 0.35f && parts.uniform() < 0.25f) parts.spawn(Kind::flake, x, y, 0.f, 15.f * u, 0.f, 1.f + parts.uniform(), 5.f);
      }
    }
    {  // the crater's smoke fades; the cloud's look turns from fireball to smoke over a second after the hand-over
      const float w = smooth((t - t_hand) / 1.2f);
      ShaderSpec mix = smoke;
      const auto lerp = [w](float a, float b) { return a + w * (b - a); };
      mix.heat_scale = lerp(cloud.heat_scale, smoke.heat_scale);
      mix.emission = lerp(cloud.emission, smoke.emission);
      mix.emission_power = lerp(cloud.emission_power, smoke.emission_power);
      mix.soot_density = lerp(cloud.soot_density, smoke.soot_density);
      mix.soot_albedo = lerp(cloud.soot_albedo, smoke.soot_albedo);
      mix.sky = lerp(cloud.sky, smoke.sky);
      mix.scene_light = lerp(cloud.scene_light, smoke.scene_light);
      mix.relief = lerp(cloud.relief, smoke.relief);
      for (int c = 0; c < 3; ++c) mix.tint[zs(c)] = lerp(cloud.tint[zs(c)], smoke.tint[zs(c)]);
      for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) {
          sm[zs(r)][zs(c)]->controls[0] = 0.75f - 0.4f * smooth((t - t_hand) / 5.f);
          sm[zs(r)][zs(c)]->spec = mix;
        }
      }
    }
    for (int i = 0; i < lit; ++i) {  // new fires grow
      Module* fm = fires[zs(i)];
      fm->controls[0] = 0.15f + 0.6f * smooth((t - lit_at[zs(i)]) / 1.5f);
      fm->opacity = smooth((t - lit_at[zs(i)]) / 0.5f);
    }
    if (t_sec >= 0.f) {  // the wreck burns harder once it has gone up
      wreck->controls[0] = 0.62f + 0.3f * smooth((t - t_sec) / 1.f);
      sec->opacity = 1.f - smooth((t - t_sec - 2.4f) / 0.6f);
      if (t - t_sec > 3.f) sec->active = false;
    }
    const float since = t - t_det;
    const float flash = since >= 0.f ? std::exp(-since / 0.09f) : 0.f;
    const float flash2 = t_sec >= 0.f && t >= t_sec ? 0.5f * std::exp(-(t - t_sec) / 0.07f) : 0.f;
    const float shake = (since >= 0.f ? 10.f * u * std::exp(-since / 0.22f) : 0.f) + (flash2 > 0.f ? 6.f * u * std::exp(-(t - t_sec) / 0.18f) : 0.f);
    frame.cam_x = (parts.uniform() - 0.5f) * 2.f * shake;
    frame.cam_y = -125.f * u * smooth((since - 0.4f) / 5.5f) + (parts.uniform() - 0.5f) * 2.f * shake;
    frame.exposure = 1.25f * (1.f + 0.3f * flash + 0.15f * flash2);
    frame.fade = smooth(t / 0.35f) * smooth((t_end - t) / 0.6f);
    frame.time = t;
    for (auto& s : scorch) s[3] = (s == scorch.front() ? 1.f : 0.8f) * (0.25f + 0.75f * std::exp(-std::max(0.f, since) / 2.5f));
    auto c1 = Clock::now();
    P[kScript] = ms(c0, c1);

    // step every active module, in parallel
    active.clear();
    for (const auto& m : owned) {
      if (m->active && !((m.get() == fires[0] || m.get() == fires[1]) && m->opacity <= 0.f && lit == 0)) active.push_back(m.get());
    }
    for (Module* m : active) m->step_ms = m->shade_ms = 0.0;
    pool.run(static_cast<int>(active.size()), [&](int i) { active[zs(i)]->step(); });
    auto c2 = Clock::now();
    P[kStep] = ms(c1, c2);

    // couplings
    main_tiles.clear();
    for (const auto* set : {&ex, &sm}) {
      for (const auto& row : *set)
        for (Module* m : row)
          if (m->active) main_tiles.push_back(m);
    }
    for (const auto* set : {&ex, &sm}) {
      const auto& g = *set;
      if (!g[0][1]->active) continue;
      for (int r = 0; r < 2; ++r) {
        blend_band(*g[zs(r)][0], *g[zs(r)][1], Side::right, kBand);
        blend_band(*g[zs(r)][1], *g[zs(r)][2], Side::right, kBand);
      }
      for (int c = 0; c < 3; ++c) blend_band(*g[0][zs(c)], *g[1][zs(c)], Side::top, kBand);
    }
    if (sm[0][0]->active) {  // the smoke model's source belongs to the crater only
      suppress(*sm[0][0], 8, 0, 24, 7);
      suppress(*sm[0][2], 8, 0, 24, 7);
    }
    if (!main_tiles.empty()) {
      transfer(*wreck, main_tiles, 0.5f, wreck->res() - 4, 0.f, 6.f);  // the wreck's smoke leaves through its top into the sky
      for (Module* fm : fires)
        if (fm->active) transfer(*fm, main_tiles, 0.5f, fm->res() - 4, 0.f, 6.f);
      if (sec->active && t - t_sec > 0.7f) transfer(*sec, main_tiles, 0.05f);  // the second cloud joins the first
    }
    auto c3 = Clock::now();
    P[kCouple] = ms(c2, c3);

    // the bus, and the pushes it carries
    bus.clear();
    for (Module* m : active) bus.publish(*m);
    push(*wreck, bus, 0.22f);
    for (Module* fm : fires) push(*fm, bus, 0.22f);
    push(*sec, bus, 0.1f);
    if (since > 0.8f) {  // the cloud stops rising where it stops being buoyant, and spreads into a cap
      const ForceField ceiling{ForceField::Kind::ceiling, 0.f, -30.f * u, 0.f, 0.f, 150.f * u, 0.f, 0.f, 0.3f};
      for (Module* m : main_tiles) apply(*m, ceiling, smooth((since - 0.8f) / 1.f));
    }
    auto c4 = Clock::now();
    P[kBus] = ms(c3, c4);

    light.update(bus, 0.14f, {0.75f * flash + 0.5f * flash2, 0.55f * flash + 0.36f * flash2, 0.37f * flash + 0.22f * flash2}, pool);
    auto c5 = Clock::now();
    P[kLight] = ms(c4, c5);

    parts.update(1.f / 30.f, &bus, 0.9f, ground);
    for (const auto& l : parts.landings()) {  // hot embers light new fires, away from the crater and the wreck
      if (lit >= 2 || t < t_det + 0.5f || l.kind != Kind::ember || l.temp < 0.4f) continue;
      if (std::fabs(l.x - burst_x) < 230.f * u || std::fabs(l.x - sec_x) < 150.f * u || l.x < 120.f * u || l.x > 1160.f * u) continue;
      if (lit == 1 && std::fabs(l.x - lit_at[0]) < 1.f) continue;
      bool near = false;
      for (int i = 0; i < lit; ++i) near |= std::fabs(fires[zs(i)]->at.x + 0.5f * fw2 - l.x) < 200.f * u;
      if (near) continue;
      Module* fm = fires[zs(lit)];
      fm->at.x = l.x - 0.5f * fw2;
      fm->at.y = ground + 0.08f * fw2 - fw2;
      fm->active = true;
      fm->opacity = 0.f;
      lit_at[zs(lit)] = t;
      ++lit;
    }
    auto c6 = Clock::now();
    P[kPartUpdate] = ms(c5, c6);

    pool.run(static_cast<int>(active.size()), [&](int i) { active[zs(i)]->shade(&light); });
    auto c7 = Clock::now();
    P[kShade] = ms(c6, c7);

    frame.background(light, scorch, pool);
    auto c8 = Clock::now();
    P[kBackground] = ms(c7, c8);
    frame.draw(drawn, pool);
    auto c9 = Clock::now();
    P[kDraw] = ms(c8, c9);
    frame.particles(parts);
    auto c10 = Clock::now();
    P[kPartDraw] = ms(c9, c10);
    frame.haze = 1.2f * u;
    frame.distort(shocks, bus, pool);
    auto c11 = Clock::now();
    P[kDistort] = ms(c10, c11);
    frame.bloom(1.0f, 1.0f, pool);
    auto c12 = Clock::now();
    P[kBloom] = ms(c11, c12);
    frame.finish(rgb, pool);
    auto c13 = Clock::now();
    P[kFinish] = ms(c12, c13);
    if (video) std::fwrite(rgb.data(), 1, rgb.size(), video);
    auto c14 = Clock::now();
    P[kEncode] = ms(c13, c14);
    allocs[zs(f)] = g_allocations.load() - alloc0;
    for (std::size_t i = 0; i < owned.size(); ++i) mod_prof[zs(f)][i] = {owned[i]->active ? owned[i]->step_ms : 0.0, owned[i]->active ? owned[i]->shade_ms : 0.0};
    n_active[zs(f)] = static_cast<int>(active.size());
    n_parts[zs(f)] = parts.alive();
    g_counting.store(false);
    for (const float kt : key_times) {
      if (f == static_cast<int>(std::lround(kt * 30.f))) {
        Image img;
        img.allocate(A.width, A.height);
        for (std::size_t i = 0, j = 0; i < rgb.size(); i += 3, j += 4) {
          img.rgba[j] = rgb[i];
          img.rgba[j + 1] = rgb[i + 1];
          img.rgba[j + 2] = rgb[i + 2];
          img.rgba[j + 3] = 255;
        }
        if (!A.keyframes.empty()) {
          std::filesystem::create_directories(A.keyframes);
          (void)write_png(A.keyframes / std::format("frame_{:03d}.png", f), img);
        }
        keys.push_back(std::move(img));
      }
    }
    if (f % 30 == 0) {
      double tot = 0;
      for (const double v : P) tot += v;
      std::println("  t {:4.1f} s: {:6.1f} ms ({} modules, {} particles)", t, tot, active.size(), parts.alive());
      if (std::getenv("NVFX_FIREBALL_STATS")) {
        for (Module* m : active) {
          auto co = m->runner().coarse();
          float hmax = 0, dmax = 0, hsum = 0, dsum = 0, vmax = 0;
          const int C = m->channels();
          for (std::size_t i = 0; i < co.size(); i += zs(C)) {
            hmax = std::max(hmax, co[i + 2]);
            dmax = std::max(dmax, co[i + 3]);
            hsum += co[i + 2];
            dsum += co[i + 3];
            vmax = std::max(vmax, std::hypot(co[i], co[i + 1]));
          }
          const float n = static_cast<float>(co.size() / zs(C));
          std::println("      {:<18} heat max {:5.2f} mean {:6.3f}  soot max {:5.2f} mean {:6.3f}  |v| max {:5.2f}", m->name(), hmax, hsum / n, dmax, dsum / n, vmax);
        }
      }
      std::fflush(stdout);
    }
  }
  if (video && pclose(video) != 0) throw std::runtime_error("ffmpeg failed");

  // --- profile -------------------------------------------------------------------------------------------------------
  const auto stats = [](std::vector<double> v) {
    std::ranges::sort(v);
    double mean = 0;
    for (const double x : v) mean += x;
    mean /= static_cast<double>(std::max<std::size_t>(1, v.size()));
    const auto pct = [&](double p) { return v.empty() ? 0.0 : v[std::min(v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size() - 1) + 0.5))]; };
    return std::array<double, 4>{mean, pct(0.5), pct(0.9), v.empty() ? 0.0 : v.back()};
  };
  std::println("\nstage                 mean   median      p90      max   (ms per frame, {} frames)", frames);
  std::vector<double> total(zs(frames)), step_cpu(zs(frames)), shade_cpu(zs(frames));
  for (int s = 0; s < kStages; ++s) {
    std::vector<double> v(zs(frames));
    for (int f = 0; f < frames; ++f) {
      v[zs(f)] = prof[zs(f)][zs(s)];
      total[zs(f)] += v[zs(f)];
    }
    const auto st = stats(v);
    std::println("{:<18} {:8.2f} {:8.2f} {:8.2f} {:8.2f}", kStageNames[s], st[0], st[1], st[2], st[3]);
  }
  for (int f = 0; f < frames; ++f) {
    for (const auto& [a, b] : mod_prof[zs(f)]) {
      step_cpu[zs(f)] += a;
      shade_cpu[zs(f)] += b;
    }
  }
  const auto tt = stats(total), sc_ = stats(step_cpu), sh_ = stats(shade_cpu);
  std::println("{:<18} {:8.2f} {:8.2f} {:8.2f} {:8.2f}", "total", tt[0], tt[1], tt[2], tt[3]);
  std::println("{:<18} {:8.2f} {:8.2f} {:8.2f} {:8.2f}   (sum over modules, all threads)", "step cpu", sc_[0], sc_[1], sc_[2], sc_[3]);
  std::println("{:<18} {:8.2f} {:8.2f} {:8.2f} {:8.2f}", "shade cpu", sh_[0], sh_[1], sh_[2], sh_[3]);
  long alloc_frames = 0, alloc_total = 0;
  for (int f = 1; f < frames; ++f) {
    alloc_frames += allocs[zs(f)] > 0;
    alloc_total += allocs[zs(f)];
  }
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  std::println("allocations in the frame loop: {} in {} of {} frames", alloc_total, alloc_frames, frames - 1);
  std::println("memory: effects resident {:.0f} KB, module scratch {:.1f} MB, peak RSS {:.1f} MB", static_cast<double>(resident) / 1024.0,
               static_cast<double>(scratch) / 1048576.0, static_cast<double>(ru.ru_maxrss) / 1024.0);
  for (const Rule& r : rules) std::println("rule '{}' fired at {:.2f} s", r.name, r.fired_at);
  std::println("fires lit: {}", lit);

  if (!A.profile.empty()) {
    std::filesystem::create_directories(A.profile.parent_path().empty() ? "." : A.profile.parent_path());
    std::ofstream o(A.profile);
    o << "frame,t,modules,particles,allocations";
    for (const char* n : kStageNames) o << ',' << n;
    o << ",total";
    for (const auto& m : owned) o << ',' << m->name() << "_step," << m->name() << "_shade";
    o << '\n';
    for (int f = 0; f < frames; ++f) {
      o << f << ',' << std::format("{:.3f}", static_cast<double>(f) / 30.0) << ',' << n_active[zs(f)] << ',' << n_parts[zs(f)] << ',' << allocs[zs(f)];
      for (const double v : prof[zs(f)]) o << std::format(",{:.3f}", v);
      o << std::format(",{:.3f}", total[zs(f)]);
      for (const auto& [a, b] : mod_prof[zs(f)]) o << std::format(",{:.3f},{:.3f}", a, b);
      o << '\n';
    }
    std::ofstream meta(A.profile.string() + ".meta");
    meta << std::format("width,{}\nheight,{}\nquality,{}\nthreads,{}\nisa,{}\nmain_tile_px,{}\nmodules,{}\nsetup_ms,{:.1f}\nresident_kb,{:.1f}\nscratch_mb,{:.2f}\npeak_rss_mb,{:.1f}\n"
                        "alloc_total,{}\nalloc_frames,{}\nfires_lit,{}\n",
                        A.width, A.height, q, pool.threads(), isa_name(isa), T, owned.size(), setup_ms, static_cast<double>(resident) / 1024.0,
                        static_cast<double>(scratch) / 1048576.0, static_cast<double>(ru.ru_maxrss) / 1024.0, alloc_total, alloc_frames, lit);
    for (const auto& m : owned) meta << std::format("module,{},{},{},{:.2f}\n", m->name(), m->effect().m.effect, m->size(), static_cast<double>(m->runner().scratch_bytes()) / 1048576.0);
    for (const Rule& r : rules) meta << std::format("rule,{},{:.3f}\n", r.name, r.fired_at);
  }
  if (!A.sheet.empty() && !keys.empty()) {  // keyframes, 4 per row, at a quarter size
    const int kw = A.width / 4, kh = A.height / 4, cols = 4, rows_n = (static_cast<int>(keys.size()) + cols - 1) / cols;
    Image sheet;
    sheet.allocate(cols * kw, rows_n * kh);
    for (std::size_t k = 0; k < keys.size(); ++k) {
      const int ox = static_cast<int>(k) % cols * kw, oy = static_cast<int>(k) / cols * kh;
      for (int y = 0; y < kh; ++y) {
        for (int x = 0; x < kw; ++x) {
          int acc[3] = {0, 0, 0};
          for (int dy = 0; dy < 4; ++dy)
            for (int dx = 0; dx < 4; ++dx)
              for (int c = 0; c < 3; ++c) acc[c] += keys[k].pixel(4 * x + dx, 4 * y + dy)[c];
          std::uint8_t* d = sheet.pixel(ox + x, oy + y);
          for (int c = 0; c < 3; ++c) d[c] = static_cast<std::uint8_t>(acc[c] / 16);
          d[3] = 255;
        }
      }
    }
    if (auto w = write_png(A.sheet, sheet); !w) std::println("sheet: {}", w.error());
  }
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_fireball: {}", e.what());
  return 1;
}
