// Study G (docs/DCM.md): diffusion-context mixing for generating effects. Part of nvfx_experiment. A skeleton: the
// steps are registered as g-data, g-pilot, g-search, g-eval and g-timing, and later stages fill them in.
//
// Stage S5 (docs/DCM.md, section G2) adds g-diff and g-diff-test: the coarse-state denoiser (nvfx_dcm ddpm-train) as a
// prior against long-run drift (G2b) and as a source of start points (G2c), each against its alternative without
// diffusion, on validation settings; g-diff-test runs the test protocol once for a use that passed. G2a's first check
// (mutual information of the denoiser's contexts with hand-made ones) is nvfx_dcm contexts.
#include "experiment_g.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/dcm/ddpm.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <functional>
#include <cstdio>
#include <map>
#include <numeric>
#include <print>
#include <set>
#include <random>
#include <sched.h>
#include <sstream>
#include <thread>

namespace nfx::study_g {

namespace {

void not_yet(std::string_view step) { std::println("{}: not yet implemented", step); }

}  // namespace

void step_data(const Ctx&) { not_yet("g-data"); }
void step_pilot(const Ctx&) { not_yet("g-pilot"); }
void step_search(const Ctx&) { not_yet("g-search"); }
void step_eval(const Ctx&) { not_yet("g-eval"); }
void step_timing(const Ctx&) { not_yet("g-timing"); }

void report(const Ctx&, std::ostream&) {}

}  // namespace nfx::study_g

// --- g-diff: stage S5, diffusion for the macro features ------------------------------------------------------------

namespace nfx::study_g {

namespace {

namespace dd = nfx::dcm::ddpm;
namespace fs = std::filesystem;
constexpr int kSize = 128;
using Setting = std::array<float, 3>;

struct DiffPaths {
  fs::path root, out, denoiser, dmodel;
};
DiffPaths diff_paths(const Ctx& c) {
  DiffPaths p;
  p.root = c.quick ? c.data.parent_path().parent_path() : c.data.parent_path();
  p.out = c.data / "diff";
  p.denoiser = p.root / "g" / "diff" / "fire.ddpm";
  p.dmodel = p.root / "experiments" / "models" / "d" / "fire.nvfx";
  return p;
}

// Study B's held-out settings (the generator of nvfx_experiment's b_test_settings and study D's copy).
std::vector<Setting> b_test_settings() {
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < 10) {
    const Setting s{u(rng), u(rng), u(rng)};
    const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
    if (off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f})) v.push_back(s);
  }
  return v;
}

// Study G's validation settings (docs/DCM.md §4): the same kind of draw from std::mt19937_64(2027), off study B's
// training grid, and at least 0.05 (Euclidean, in control space) from every one of B's held-out settings.
std::vector<Setting> validation_settings() {
  const auto test = b_test_settings();
  std::mt19937_64 rng(2027);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < 10) {
    const Setting s{u(rng), u(rng), u(rng)};
    const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
    const bool far = std::ranges::all_of(test, [&](const Setting& t) {
      return std::sqrt((s[0] - t[0]) * (s[0] - t[0]) + (s[1] - t[1]) * (s[1] - t[1]) + (s[2] - t[2]) * (s[2] - t[2])) >= 0.05f;
    });
    if (off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f}) && far) v.push_back(s);
  }
  return v;
}

// The detail score of study D's calibration (rollout_train.cpp): detail spectrum distance plus the absolute log ratios
// of motion, mean light and mean cover. Lower is better.
double detail_score(const metrics::ClipStats& ref, const metrics::ClipStats& test) {
  const metrics::StatDistance d = metrics::distance(ref, test);
  const auto mean = [](const std::vector<double>& v) {
    double s = 0;
    for (const double x : v) s += x;
    return v.empty() ? 0.0 : s / static_cast<double>(v.size());
  };
  const auto log_ratio = [](double a, double b) { return std::abs(std::log(std::max(1e-6, a) / std::max(1e-6, b))); };
  const double em = mean(ref.emission);
  return d.spectrum_l1 + std::abs(std::log(std::max(1e-3, d.motion_ratio))) + (em > 1e-3 ? log_ratio(mean(test.emission), em) : 0.0) +
         log_ratio(mean(test.coverage), mean(ref.coverage));
}

struct Scored {
  metrics::StatDistance d;
  double score = 0;
};
Scored score_clip(const metrics::ClipStats& ref, const Clip& clip) {
  const metrics::ClipStats st = metrics::stats(clip);
  return {metrics::distance(ref, st), detail_score(ref, st)};
}
std::string scored_csv(const Scored& s) {
  return std::format("{:.4f},{:.4f},{:.4f},{:.4f},{:.3f},{:.4f}", s.d.spectrum_l1, s.d.motion_ratio, s.d.coverage_l1, s.d.emission_l1, s.d.mean_frame_psnr, s.score);
}
constexpr const char* kScoredHeader = "spectrum_l1,motion_ratio,coverage_l1,emission_l1,mean_frame_psnr,detail_score";

// A real run at a setting: warmed up, then `frames` frames at 128 x 128.
Clip real_clip(const Setting& s, std::uint64_t seed, int warm, int frames) {
  sim::Params p;
  p.effect = sim::Effect::fire;
  p.intensity = s[0];
  p.wind = s[1];
  p.turbulence = s[2];
  p.seed = seed;
  p.size = kSize;
  sim::Fluid f(p);
  for (int i = 0; i < warm; ++i) f.step_frame();
  Clip c;
  c.allocate(kSize, frames);
  c.fps = 30.f;
  for (int i = 0; i < frames; ++i) {
    f.step_frame();
    f.render(c.frame(i));
  }
  return c;
}

// The coarse state (kPhys channels) of a real run at frame `frame` (the d-train recording, at a setting).
std::vector<float> real_coarse(const Setting& s, std::uint64_t seed, int frame) {
  sim::Params p;
  p.effect = sim::Effect::fire;
  p.intensity = s[0];
  p.wind = s[1];
  p.turbulence = s[2];
  p.seed = seed;
  p.size = kSize;
  const rollout::Run r = rollout::record_run(p, frame + 1, 32);
  const std::size_t per = 32 * 32 * rollout::kPhys;
  return {r.coarse.begin() + static_cast<std::ptrdiff_t>(per * static_cast<std::size_t>(frame)), r.coarse.begin() + static_cast<std::ptrdiff_t>(per * static_cast<std::size_t>(frame + 1))};
}

// rollout::render without its per-pixel allocations, the directional soot sums taken once per coarse cell: the same
// arithmetic in the same order (this file, like src/core/rollout.cpp, is built without multiply-add contraction), so
// the same pixels. play() checks that on the first frame of every run.
void render_fast(const rollout::Model& m, const rollout::State& s, std::vector<float>& rgba) {
  const int R = m.h.res, S = s.size, C = m.h.channels(), H = m.h.render_hidden;
  const auto sz = [](int v) { return static_cast<std::size_t>(v); };
  const float k = static_cast<float>(S) / static_cast<float>(R);
  const float it = 1.f / m.render_scale[0], id = 1.f / m.render_scale[1];
  static constexpr std::array<std::array<int, 2>, rollout::kDirs> dirs{{{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}};
  std::vector<float> D(sz(R) * sz(R) * rollout::kDirs);
  for (int cy = 0; cy < R; ++cy) {
    for (int cx = 0; cx < R; ++cx) {
      for (int j = 0; j < rollout::kDirs; ++j) {
        float sum = 0.f;
        for (int st = 1; st <= rollout::kDirSteps; ++st) {
          const int xx = cx + st * dirs[sz(j)][0], yy = cy + st * dirs[sz(j)][1];
          if (xx >= 0 && yy >= 0 && xx < R && yy < R) sum += s.coarse[(sz(yy) * sz(R) + sz(xx)) * sz(C) + 3];
        }
        D[(sz(cy) * sz(R) + sz(cx)) * rollout::kDirs + sz(j)] = sum;
      }
    }
  }
  const auto bil = [&](int c, float x, float y) {
    x = std::clamp(x, 0.f, static_cast<float>(R - 1));
    y = std::clamp(y, 0.f, static_cast<float>(R - 1));
    const int x0 = std::min(static_cast<int>(x), R - 2), y0 = std::min(static_cast<int>(y), R - 2);
    const float fx = x - static_cast<float>(x0), fy = y - static_cast<float>(y0);
    const auto at = [&](int xx, int yy) { return s.coarse[(sz(yy) * sz(R) + sz(xx)) * sz(C) + sz(c)]; };
    return (1.f - fy) * ((1.f - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1.f - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
  };
  const rollout::RenderLayout L = rollout::render_layout(m.h);
  const float* w = m.render_w.data();
  std::array<float, rollout::kRenderIn> f{};
  std::vector<float> h1(sz(H)), h2(sz(H));
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      const float xc = (static_cast<float>(x) + 0.5f) / k - 0.5f, yc = (static_cast<float>(y) + 0.5f) / k - 0.5f;
      f[0] = s.fine_t[sz(y) * sz(S) + sz(x)] * it;
      f[1] = s.fine_d[sz(y) * sz(S) + sz(x)] * id;
      f[2] = bil(2, xc, yc) * it;
      f[3] = bil(3, xc, yc) * id;
      const float cx = std::clamp(xc, 0.f, static_cast<float>(R - 1)), cy = std::clamp(yc, 0.f, static_cast<float>(R - 1));
      const int x0 = std::min(static_cast<int>(cx), R - 2), y0 = std::min(static_cast<int>(cy), R - 2);
      const float fx = cx - static_cast<float>(x0), fy = cy - static_cast<float>(y0);
      const float* d00 = &D[(sz(y0) * sz(R) + sz(x0)) * rollout::kDirs];
      const float* d10 = d00 + rollout::kDirs;
      const float* d01 = d00 + sz(R) * rollout::kDirs;
      const float* d11 = d01 + rollout::kDirs;
      for (int j = 0; j < rollout::kDirs; ++j) {
        const float a = (1.f - fx) * d00[j] + fx * d10[j];
        const float b = (1.f - fx) * d01[j] + fx * d11[j];
        f[4 + sz(j)] = ((1.f - fy) * a + fy * b) * id;
      }
      for (int j = 0; j < H; ++j) {
        float v = w[L.b1 + sz(j)];
        for (int i = 0; i < rollout::kRenderIn; ++i) v += w[L.w1 + sz(j) * rollout::kRenderIn + sz(i)] * f[sz(i)];
        h1[sz(j)] = std::max(0.f, v);
      }
      for (int j = 0; j < H; ++j) {
        float v = w[L.b2 + sz(j)];
        for (int i = 0; i < H; ++i) v += w[L.w2 + sz(j) * sz(H) + sz(i)] * h1[sz(i)];
        h2[sz(j)] = std::max(0.f, v);
      }
      const float g = rollout::render_gate(f[0], f[1]);
      float* p = rgba.data() + (sz(S - 1 - y) * sz(S) + sz(x)) * 4;
      for (int c = 0; c < 4; ++c) {
        float v = w[L.bo + sz(c)];
        for (int i = 0; i < H; ++i) v += w[L.wo + sz(c) * sz(H) + sz(i)] * h2[sz(i)];
        p[c] = std::clamp(v * g, 0.f, 1.f);
      }
    }
  }
}

// `frames` frames of the reference rollout from s, each rendered; after(frame, s) runs after each step.
void play(const rollout::Model& m, rollout::State& s, std::span<const float> ctl, std::uint64_t seed, Clip& out, int frames,
          const std::function<void(rollout::State&)>& after) {
  out.allocate(kSize, frames);
  out.fps = 30.f;
  std::vector<float> rgba(static_cast<std::size_t>(kSize) * kSize * 4), check(rgba.size());
  for (int f = 0; f < frames; ++f) {
    rollout::step(m, s, ctl, seed);
    if (after) after(s);
    render_fast(m, s, rgba);
    if (f == 0) {
      rollout::render(m, s, check);
      if (check != rgba) throw std::logic_error("g-diff: the fast renderer differs from rollout::render");
    }
    const auto o = out.frame(f);
    for (std::size_t i = 0; i < rgba.size(); ++i) o[i] = static_cast<std::uint8_t>(rgba[i] * 255.f + 0.5f);
  }
}

int nearest_start(const rollout::Model& m, std::span<const float> ctl) {
  int best = 0;
  float bd = 1e30f;
  for (std::size_t k = 0; k < m.starts.size(); ++k) {
    float d = 0;
    for (std::size_t q = 0; q < 3; ++q) d += (m.starts[k].controls[q] - ctl[q]) * (m.starts[k].controls[q] - ctl[q]);
    if (d < bd) {
      bd = d;
      best = static_cast<int>(k);
    }
  }
  return best;
}

// The physical coarse channels of a rollout state, in the denoiser's units, and back (memory channels untouched).
std::vector<float> state_to_x(const dd::Denoiser& d, const rollout::Model& m, const rollout::State& s) {
  std::vector<float> x(static_cast<std::size_t>(d.cfg.res * d.cfg.res * d.cfg.channels));
  dd::to_network(d, s.coarse, m.h.channels(), x);
  return x;
}
void x_to_state(const dd::Denoiser& d, const rollout::Model& m, std::span<const float> x, rollout::State& s) {
  dd::to_physical(d, x, m.h.channels(), s.coarse);
  const int C = m.h.channels();
  for (std::size_t i = 0; i < s.coarse.size(); ++i) {
    const auto k = i % static_cast<std::size_t>(C);
    if (k < rollout::kPhys) s.coarse[i] = std::clamp(s.coarse[i], m.lo[k], m.hi[k]);
  }
}

// G2b: the prior, applied after every `every`-th frame.
struct Prior {
  int every = 0, t = 0;
  float beta = 0;
  std::string name() const { return every == 0 ? "none" : std::format("prior_n{}_t{}_b{}", every, t, beta); }
};

// Six 10 s windows of one continuous 60 s rollout from the start point nearest the controls (fine fields grown by the
// effect's warm-up), each scored against the real 10 s run.
std::vector<Scored> long_run(const rollout::Model& m, const dd::Denoiser* d, const Prior& pr, const Setting& s, std::uint64_t seed,
                             const metrics::ClipStats& ref, int windows) {
  const std::vector<float> ctl(s.begin(), s.end());
  rollout::State st = rollout::start(m, nearest_start(m, ctl), kSize, ctl, seed);
  int frame = 0;
  std::vector<float> cond(static_cast<std::size_t>(m.h.cond()));
  const auto after = [&](rollout::State& x) {
    ++frame;
    if (pr.every <= 0 || frame % pr.every != 0) return;
    std::vector<float> v = state_to_x(*d, m, x);
    rollout::condition(m, ctl, x.time, cond);
    dd::prior_step(*d, v, pr.t, pr.beta, cond);
    x_to_state(*d, m, v, x);
  };
  std::vector<Scored> out;
  Clip w;
  for (int k = 0; k < windows; ++k) {
    play(m, st, ctl, seed, w, 300, after);
    out.push_back(score_clip(ref, w));
  }
  return out;
}

// The runtime's shards (one per 6 s, crossfaded), as d-eval plays them: the reference for what a restart policy gives.
std::vector<Scored> shard_run(const rollout::Model& m, const Setting& s, std::uint64_t seed, const metrics::ClipStats& ref, int windows) {
  std::ostringstream os;
  if (auto r = rollout::save_model(os, m); !r) throw std::runtime_error(r.error());
  const std::string b = os.str();
  nvfx_effect* e = nullptr;
  if (nvfx_effect_load_memory(b.data(), b.size(), &e) != NVFX_OK) throw std::runtime_error("runtime load failed");
  nvfx_instance* in = nullptr;
  if (nvfx_instance_create(e, kSize, &in) != NVFX_OK) throw std::runtime_error("instance");
  nvfx_instance_set_controls(in, s.data(), 3);
  nvfx_instance_set_seed(in, seed);
  std::vector<Scored> out;
  Clip w;
  w.allocate(kSize, 300);
  for (int k = 0; k < windows; ++k) {
    for (int f = 0; f < 300; ++f) nvfx_render(in, (k * 300 + f) / 30.0, w.frame(f).data(), kSize * 4);
    out.push_back(score_clip(ref, w));
  }
  nvfx_instance_free(in);
  nvfx_effect_free(e);
  return out;
}

// Runs jobs 0..n-1 on `threads` threads.
void parallel(int n, int threads, const std::function<void(int)>& job) {
  std::atomic<int> next{0};
  std::vector<std::jthread> pool;
  for (int t = 0; t < std::max(1, std::min(threads, n)); ++t) {
    pool.emplace_back([&] {
      for (int i; (i = next++) < n;) job(i);
    });
  }
}

double load_average() {
  std::ifstream f("/proc/loadavg");
  double v = 99;
  f >> v;
  return v;
}

// Milliseconds of one denoiser pass on one pinned core (median of 60), and whether the machine was quiet (load < 1.5).
std::pair<double, bool> pass_ms(const dd::Denoiser& d) {
  const double load = load_average();
  cpu_set_t old, one;
  sched_getaffinity(0, sizeof(old), &old);
  CPU_ZERO(&one);
  CPU_SET(3, &one);
  sched_setaffinity(0, sizeof(one), &one);
  std::vector<float> x(static_cast<std::size_t>(d.cfg.res * d.cfg.res * d.cfg.channels)), eps(x.size());
  dd::gaussian(5, x);
  const std::vector<float> cond(static_cast<std::size_t>(d.cfg.cond), 0.5f);
  for (int i = 0; i < 5; ++i) dd::predict_eps(d, x, 50, cond, eps);
  std::vector<double> ms;
  for (int i = 0; i < 60; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    dd::predict_eps(d, x, 50, cond, eps);
    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  sched_setaffinity(0, sizeof(old), &old);
  std::ranges::sort(ms);
  return {ms[ms.size() / 2], load < 1.5 && load_average() < 1.5};
}

std::string interval_csv(const metrics::Interval& iv) { return std::format("{:.4f},{:.4f},{:.4f}", iv.mean, iv.lo, iv.hi); }
std::string interval_text(const metrics::Interval& iv) {
  return iv.covers_zero() ? std::format("{:+.4f} [{:+.4f}, {:+.4f}] (tie)", iv.mean, iv.lo, iv.hi) : std::format("{:+.4f} [{:+.4f}, {:+.4f}]", iv.mean, iv.lo, iv.hi);
}

void write_rows(const fs::path& path, const std::string& header, const std::vector<std::string>& rows) {
  fs::create_directories(path.parent_path());
  std::ofstream o(path);
  o << header << "\n";
  for (const auto& r : rows) o << r << "\n";
}

// Writes rows, keeping the rows of other phases already in the file (the phase is the first column), so validation and
// test runs add up.
void merge_rows(const fs::path& path, const std::string& header, const std::vector<std::string>& rows) {
  std::set<std::string> phases;
  for (const auto& r : rows) phases.insert(r.substr(0, r.find(',')));
  std::vector<std::string> keep;
  if (std::ifstream in(path); in) {
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      if (!line.empty() && !phases.contains(line.substr(0, line.find(',')))) keep.push_back(line);
    }
  }
  keep.insert(keep.end(), rows.begin(), rows.end());
  write_rows(path, header, keep);
}

struct Loaded {
  rollout::Model m;
  dd::Denoiser d;
};
Loaded load_both(const DiffPaths& p) {
  auto m = rollout::load_model(p.dmodel);
  if (!m) throw std::runtime_error(p.dmodel.string() + ": " + m.error());
  auto d = dd::load(p.denoiser);
  if (!d) throw std::runtime_error(p.denoiser.string() + ": " + d.error() + " (train it with nvfx_dcm ddpm-train)");
  return {std::move(*m), std::move(*d)};
}

// --- G2b ------------------------------------------------------------------------------------------------------------

struct PriorResult {
  Prior chosen;
  metrics::Interval delta;  // chosen minus none, per (setting, window)
  double ms_per_frame = 0;
  bool quiet = false, pass = false;
};

// One phase: rollouts of `priors` (plus none and the runtime's shards) at `settings`, seeds base + setting index.
std::map<std::string, std::vector<double>> prior_phase(const Ctx& c, const Loaded& L, const std::string& phase, const std::vector<Setting>& settings,
                                                     std::uint64_t real_seed, std::uint64_t seed, const std::vector<Prior>& priors, bool shards,
                                                     std::vector<std::string>& rows) {
  const int windows = c.quick ? 2 : 6;
  std::vector<metrics::ClipStats> refs(settings.size());
  parallel(static_cast<int>(settings.size()), c.threads, [&](int si) {
    refs[static_cast<std::size_t>(si)] = metrics::stats(real_clip(settings[static_cast<std::size_t>(si)], real_seed + static_cast<std::uint64_t>(si), 150, 300));
  });
  struct Job {
    int setting;
    Prior prior;
    bool shards;
  };
  std::vector<Job> jobs;
  for (std::size_t si = 0; si < settings.size(); ++si) {
    jobs.push_back({static_cast<int>(si), Prior{}, false});
    for (const Prior& p : priors) jobs.push_back({static_cast<int>(si), p, false});
    if (shards) jobs.push_back({static_cast<int>(si), Prior{}, true});
  }
  std::vector<std::vector<Scored>> res(jobs.size());
  std::atomic<int> done{0};
  const auto t0 = std::chrono::steady_clock::now();
  parallel(static_cast<int>(jobs.size()), c.threads, [&](int j) {
    const Job& jb = jobs[static_cast<std::size_t>(j)];
    const Setting& s = settings[static_cast<std::size_t>(jb.setting)];
    const std::uint64_t sd = seed + static_cast<std::uint64_t>(jb.setting);
    res[static_cast<std::size_t>(j)] = jb.shards ? shard_run(L.m, s, sd, refs[static_cast<std::size_t>(jb.setting)], windows)
                                                 : long_run(L.m, &L.d, jb.prior, s, sd, refs[static_cast<std::size_t>(jb.setting)], windows);
    const int k = ++done;
    std::println("  g-diff {} {}/{} {} at setting {} ({:.0f} s)", phase, k, jobs.size(), jb.shards ? "shards" : jb.prior.name(), jb.setting,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::fflush(stdout);
  });
  std::map<std::string, std::vector<double>> scores;  // method -> per (setting, window) detail scores, in order
  for (std::size_t j = 0; j < jobs.size(); ++j) {
    const std::string name = jobs[j].shards ? "shards" : jobs[j].prior.name();
    for (std::size_t w = 0; w < res[j].size(); ++w) {
      rows.push_back(std::format("{},{},{},{},{},{}", phase, jobs[j].setting, seed + static_cast<std::uint64_t>(jobs[j].setting), name, w, scored_csv(res[j][w])));
      scores[name].push_back(res[j][w].score);
    }
  }
  return scores;
}

// --- G2c ------------------------------------------------------------------------------------------------------------

// A start-point method: "rolled" (the nearest stored start rolled ahead 0.5 s at the requested controls), "stored" (the
// nearest stored start as it is), "fresh" (a DDIM sample at the requested controls) or "sdedit" (the nearest stored
// start moved to the requested controls by SDEdit from t0).
struct StartMethod {
  std::string kind;
  int t0 = 0;
  std::string name() const { return kind == "sdedit" ? std::format("sdedit_t{}", t0) : kind; }
};
constexpr int kDdimSteps = 25, kRollAhead = 15, kScoredFrames = 60;

// The coarse start (physical, kPhys channels) a method gives at a setting.
std::vector<float> start_state(const Loaded& L, const StartMethod& sm, const Setting& s, std::uint64_t seed) {
  const std::vector<float> ctl(s.begin(), s.end());
  const int idx = nearest_start(L.m, ctl);
  const rollout::StartPoint& sp = L.m.starts[static_cast<std::size_t>(idx)];
  std::vector<float> cond(static_cast<std::size_t>(L.m.h.cond()));
  rollout::condition(L.m, ctl, sp.time, cond);
  const std::size_t n = static_cast<std::size_t>(L.d.cfg.res * L.d.cfg.res * L.d.cfg.channels);
  std::vector<float> x(n), out(n), phys(sp.coarse.size());
  if (sm.kind == "stored") return sp.coarse;
  if (sm.kind == "rolled") {  // the stepper alone, at the requested controls (the coarse half of rolling a shard ahead)
    rollout::State st;
    st.res = L.m.h.res;
    st.size = kSize;
    st.time = sp.time;
    st.coarse.assign(static_cast<std::size_t>(L.m.h.res * L.m.h.res * L.m.h.channels()), 0.f);
    for (int i = 0; i < L.m.h.res * L.m.h.res; ++i) {
      for (int k = 0; k < rollout::kPhys; ++k) st.coarse[static_cast<std::size_t>(i * L.m.h.channels() + k)] = sp.coarse[static_cast<std::size_t>(i * rollout::kPhys + k)];
    }
    st.pressure.assign(static_cast<std::size_t>(L.m.h.res * L.m.h.res), 0.f);
    st.flow.assign(static_cast<std::size_t>(L.m.h.res * L.m.h.res * 2), 0.f);
    std::vector<float> noise(static_cast<std::size_t>(L.m.h.res * L.m.h.res * rollout::kNoise)), next(st.coarse.size());
    for (int f = 0; f < kRollAhead; ++f) {
      rollout::condition(L.m, ctl, st.time, cond);
      rollout::coarse_noise(L.m, seed, st.time, noise);
      rollout::coarse_step(L.m, st.coarse, noise, cond, st.pressure, next, st.flow);
      st.coarse.swap(next);
      st.time += 1.f / L.m.fps;
    }
    for (int i = 0; i < L.m.h.res * L.m.h.res; ++i) {
      for (int k = 0; k < rollout::kPhys; ++k) phys[static_cast<std::size_t>(i * rollout::kPhys + k)] = st.coarse[static_cast<std::size_t>(i * L.m.h.channels() + k)];
    }
    return phys;
  }
  if (sm.kind == "fresh") {
    dd::sample(L.d, cond, kDdimSteps, seed, out);
  } else {
    dd::to_network(L.d, sp.coarse, rollout::kPhys, x);
    dd::sdedit(L.d, x, sm.t0, kDdimSteps, cond, seed, out);
  }
  dd::to_physical(L.d, out, rollout::kPhys, phys);
  for (std::size_t i = 0; i < phys.size(); ++i) phys[i] = std::clamp(phys[i], L.m.lo[i % rollout::kPhys], L.m.hi[i % rollout::kPhys]);
  return phys;
}

// The first 2 s of play from a start: the effect's own start-up (fire grows its fine fields in a 1 s warm-up), then
// kScoredFrames frames scored against the real run. The rolled start has already been rolled ahead.
Scored first_seconds(const Loaded& L, const StartMethod& sm, const Setting& s, std::uint64_t seed, const metrics::ClipStats& ref) {
  const std::vector<float> ctl(s.begin(), s.end());
  const int idx = nearest_start(L.m, ctl);
  rollout::Model mm = L.m;
  rollout::StartPoint sp = L.m.starts[static_cast<std::size_t>(idx)];
  sp.coarse = start_state(L, sm, s, seed);
  sp.controls = ctl;
  if (sm.kind == "rolled") sp.time += static_cast<float>(kRollAhead) / L.m.fps;
  mm.starts = {sp};
  rollout::State st = rollout::start(mm, 0, kSize, ctl, seed);
  Clip clip;
  play(mm, st, ctl, seed, clip, kScoredFrames, {});
  return score_clip(ref, clip);
}

double mean_pairwise_rms(const std::vector<std::vector<float>>& xs) {
  double s = 0;
  int n = 0;
  for (std::size_t a = 0; a < xs.size(); ++a) {
    for (std::size_t b = a + 1; b < xs.size(); ++b) {
      double e = 0;
      for (std::size_t i = 0; i < xs[a].size(); ++i) e += (static_cast<double>(xs[a][i]) - xs[b][i]) * (static_cast<double>(xs[a][i]) - xs[b][i]);
      s += std::sqrt(e / static_cast<double>(xs[a].size()));
      ++n;
    }
  }
  return n ? s / n : 0.0;
}

// One phase of G2c: every method at every setting with `seeds` seeds; returns method -> per-setting mean score.
std::map<std::string, std::vector<double>> start_phase(const Ctx& c, const Loaded& L, const std::string& phase, const std::vector<Setting>& settings,
                                                       std::uint64_t real_seed, std::uint64_t seed, int seeds, const std::vector<StartMethod>& methods,
                                                       std::vector<std::string>& rows) {
  std::vector<metrics::ClipStats> refs(settings.size());
  parallel(static_cast<int>(settings.size()), c.threads, [&](int si) {
    refs[static_cast<std::size_t>(si)] = metrics::stats(real_clip(settings[static_cast<std::size_t>(si)], real_seed + static_cast<std::uint64_t>(si), 150, 300));
  });
  const int jobs = static_cast<int>(settings.size() * methods.size()) * seeds;
  std::vector<Scored> res(static_cast<std::size_t>(jobs));
  parallel(jobs, c.threads, [&](int j) {
    const int si = j / (static_cast<int>(methods.size()) * seeds), mi = (j / seeds) % static_cast<int>(methods.size()), k = j % seeds;
    const std::uint64_t sd = seed + static_cast<std::uint64_t>(si * 10 + k);
    res[static_cast<std::size_t>(j)] = first_seconds(L, methods[static_cast<std::size_t>(mi)], settings[static_cast<std::size_t>(si)], sd, refs[static_cast<std::size_t>(si)]);
  });
  std::map<std::string, std::vector<double>> per_setting;
  for (int j = 0; j < jobs; ++j) {
    const int si = j / (static_cast<int>(methods.size()) * seeds), mi = (j / seeds) % static_cast<int>(methods.size()), k = j % seeds;
    const std::string name = methods[static_cast<std::size_t>(mi)].name();
    rows.push_back(std::format("{},{},{},{},{}", phase, si, seed + static_cast<std::uint64_t>(si * 10 + k), name, scored_csv(res[static_cast<std::size_t>(j)])));
    auto& v = per_setting[name];
    if (v.size() < settings.size()) v.resize(settings.size(), 0.0);
    v[static_cast<std::size_t>(si)] += res[static_cast<std::size_t>(j)].score / seeds;
  }
  std::println("  g-diff {}: {} first-2-s runs done", phase, jobs);
  return per_setting;
}

const std::string kDecisionHeader = "phase,use,comparison,mean,lo,hi,cost_ms,cost_bound_ms,timing,rule,decision";

}  // namespace

void step_diff(const Ctx& c) {
  const DiffPaths P = diff_paths(c);
  const Loaded L = load_both(P);
  const auto val = validation_settings();
  std::println("g-diff: denoiser {} ({}), {} parameters, {:.1f} M multiply-adds per pass", P.denoiser.string(), dd::version(L.d).substr(0, 16),
               L.d.parameters(), dd::forward_macs(L.d.cfg) / 1e6);
  {
    std::vector<std::string> rows;
    for (std::size_t i = 0; i < val.size(); ++i) rows.push_back(std::format("{},{:.4f},{:.4f},{:.4f}", i, val[i][0], val[i][1], val[i][2]));
    write_rows(c.results / "g_diff_settings.csv", "setting,intensity,wind,turbulence", rows);
  }
  const auto [ms, quiet] = pass_ms(L.d);
  std::println("g-diff: one denoiser pass {:.2f} ms on one core ({})", ms, quiet ? "quiet machine" : "machine busy: an upper bound, unmeasured by the rules");
  std::vector<std::string> decisions;
  // ---- G2b: tune on seeds 1'960'000 + setting, decide on fresh seeds 1'970'000 + setting
  {
    const std::vector<Setting> two(val.begin(), val.begin() + 2);
    std::vector<Prior> grid;
    for (const int n : {4, 8, 16}) {
      for (const int t : {20, 50, 100}) {
        for (const float b : {0.25f, 0.5f, 1.f}) grid.push_back({n, t, b});
      }
    }
    if (c.quick) grid = {{8, 50, 0.5f}, {16, 100, 1.f}};
    std::vector<std::string> rows;
    const auto tune = prior_phase(c, L, "tune", two, 1950000, 1960000, grid, true, rows);
    Prior best;
    double best_mean = 1e30;
    for (const Prior& p : grid) {
      const auto& v = tune.at(p.name());
      const double m = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
      if (m < best_mean) {
        best_mean = m;
        best = p;
      }
    }
    const auto mean_of = [](const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); };
    std::println("g-diff G2b tuning: none {:.4f}, shards {:.4f}, best prior {} {:.4f}", mean_of(tune.at("none")), mean_of(tune.at("shards")), best.name(), best_mean);
    const auto v = prior_phase(c, L, "validate", two, 1950000, 1970000, {best}, true, rows);
    const metrics::Interval iv = metrics::paired_bootstrap(v.at(best.name()), v.at("none"), 10000, 1);
    const metrics::Interval sh = metrics::paired_bootstrap(v.at(best.name()), v.at("shards"), 10000, 1);
    const double cost = ms / best.every;
    const bool pass = iv.hi < 0 && cost <= 0.5;
    std::println("g-diff G2b validation: {} minus none {}; minus shards {}; {:.3f} ms per frame -> {}", best.name(), interval_text(iv), interval_text(sh), cost,
                 pass ? "KEEP" : "STOP");
    decisions.push_back(std::format("validate,G2b,{} - none,{},{:.3f},0.5,{},interval below zero and cost within bound,{}", best.name(), interval_csv(iv), cost,
                                    quiet ? "quiet" : "busy", pass ? "keep" : "stop"));
    decisions.push_back(std::format("validate,G2b,{} - shards,{},{:.3f},,{},reference only,", best.name(), interval_csv(sh), cost, quiet ? "quiet" : "busy"));
    merge_rows(c.results / "g_diff_prior.csv", std::string("phase,setting,seed,method,window,") + kScoredHeader, rows);
  }
  // ---- G2c: tune on seeds 1'981'000 + 10 setting + k, decide on fresh seeds 1'982'000 + ...
  {
    std::vector<StartMethod> methods = {{"rolled", 0}, {"stored", 0}, {"fresh", 0}, {"sdedit", 300}, {"sdedit", 400}, {"sdedit", 500}};
    std::vector<std::string> rows;
    const std::vector<Setting> sv = c.quick ? std::vector<Setting>(val.begin(), val.begin() + 3) : val;
    const auto tune = start_phase(c, L, "tune", sv, 1980000, 1981000, c.quick ? 1 : 2, methods, rows);
    const auto mean_of = [](const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); };
    StartMethod best = methods[2];
    for (std::size_t i = 2; i < methods.size(); ++i) {
      if (mean_of(tune.at(methods[i].name())) < mean_of(tune.at(best.name()))) best = methods[i];
    }
    for (const auto& m : methods) std::println("g-diff G2c tuning: {:10} {:.4f}", m.name(), mean_of(tune.at(m.name())));
    const auto v = start_phase(c, L, "validate", sv, 1980000, 1982000, c.quick ? 1 : 2, {methods[0], best}, rows);
    const metrics::Interval iv = metrics::paired_bootstrap(v.at(best.name()), v.at("rolled"), 10000, 1);
    const int passes = best.kind == "fresh" ? dd::ddim_passes(L.d.cfg, L.d.cfg.timesteps, kDdimSteps)
                                            : dd::ddim_passes(L.d.cfg, std::clamp((best.t0 + 20) / 40 * 40, 40, L.d.cfg.timesteps), kDdimSteps);
    const double cost = passes * ms;
    const bool pass = iv.hi < 0 && cost <= 100.0;
    std::println("g-diff G2c validation: {} minus rolled {}; {} passes, {:.1f} ms per shard -> {}", best.name(), interval_text(iv), passes, cost, pass ? "KEEP" : "STOP");
    decisions.push_back(std::format("validate,G2c,{} - rolled,{},{:.2f},100,{},interval below zero and cost within bound,{}", best.name(), interval_csv(iv), cost,
                                    quiet ? "quiet" : "busy", pass ? "keep" : "stop"));
    merge_rows(c.results / "g_diff_starts.csv", std::string("phase,setting,seed,method,") + kScoredHeader, rows);
    // diversity at three settings: 8 starts each, mean pairwise RMS distance in the denoiser's units
    std::vector<std::string> div;
    for (int si = 0; si < (c.quick ? 1 : 3); ++si) {
      const Setting& s = val[static_cast<std::size_t>(si)];
      std::map<std::string, std::vector<std::vector<float>>> sets;
      std::vector<std::vector<float>> reals(8);
      parallel(8, c.threads, [&](int k) { reals[static_cast<std::size_t>(k)] = real_coarse(s, 1990000 + static_cast<std::uint64_t>(si * 10 + k), 150); });
      for (const StartMethod& m : {methods[0], methods[2], best}) {
        auto& xs = sets[m.name()];
        xs.resize(8);
        parallel(8, c.threads, [&](int k) { xs[static_cast<std::size_t>(k)] = start_state(L, m, s, 1991000 + static_cast<std::uint64_t>(si * 10 + k)); });
      }
      sets["real"] = reals;
      for (auto& [name, xs] : sets) {
        for (auto& x : xs) {
          std::vector<float> n(x.size());
          dd::to_network(L.d, x, rollout::kPhys, n);
          x = n;
        }
        const double d = mean_pairwise_rms(xs);
        div.push_back(std::format("{},{},{:.4f}", si, name, d));
        std::println("g-diff G2c diversity at setting {}: {:10} {:.4f}", si, name, d);
      }
    }
    write_rows(c.results / "g_diff_diversity.csv", "setting,method,mean_pairwise_rms", div);
  }
  merge_rows(c.results / "g_diff_decisions.csv", kDecisionHeader, decisions);
  std::println("g-diff: wrote {}", (c.results / "g_diff_decisions.csv").string());
}

void step_diff_test(const Ctx& c) {
  // The test protocol, once, for each use that passed validation (g_diff_decisions.csv says "keep").
  std::map<std::string, std::string> passed;  // use -> comparison
  if (std::ifstream in(c.results / "g_diff_decisions.csv"); in) {
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      std::vector<std::string> f;
      std::stringstream ss(line);
      for (std::string x; std::getline(ss, x, ',');) f.push_back(x);
      if (f.size() >= 11 && f[0] == "validate" && f[10] == "keep") passed[f[1]] = f[2];
    }
  }
  if (passed.empty()) {
    std::println("g-diff-test: no use passed validation; nothing is tested");
    return;
  }
  const DiffPaths P = diff_paths(c);
  const Loaded L = load_both(P);
  const auto test = b_test_settings();
  const auto [ms, quiet] = pass_ms(L.d);
  std::vector<std::string> decisions;
  if (passed.contains("G2b")) {
    const std::string name = passed["G2b"].substr(0, passed["G2b"].find(' '));
    Prior p;
    std::sscanf(name.c_str(), "prior_n%d_t%d_b%f", &p.every, &p.t, &p.beta);
    std::vector<std::string> rows;
    const std::vector<Setting> two(test.begin(), test.begin() + 2);
    const auto v = prior_phase(c, L, "test", two, 2950000, 2970000, {p}, true, rows);
    const metrics::Interval iv = metrics::paired_bootstrap(v.at(p.name()), v.at("none"), 10000, 1);
    const double cost = ms / p.every;
    std::println("g-diff-test G2b: {} minus none {}", p.name(), interval_text(iv));
    decisions.push_back(std::format("test,G2b,{} - none,{},{:.3f},0.5,{},interval below zero and cost within bound,{}", p.name(), interval_csv(iv), cost,
                                    quiet ? "quiet" : "busy", iv.hi < 0 && cost <= 0.5 ? "keep" : "stop"));
    merge_rows(c.results / "g_diff_prior.csv", std::string("phase,setting,seed,method,window,") + kScoredHeader, rows);
  }
  if (passed.contains("G2c")) {
    const std::string name = passed["G2c"].substr(0, passed["G2c"].find(' '));
    StartMethod m{name.starts_with("sdedit") ? "sdedit" : name, 0};
    if (m.kind == "sdedit") std::sscanf(name.c_str(), "sdedit_t%d", &m.t0);
    std::vector<std::string> rows;
    const auto v = start_phase(c, L, "test", test, 2980000, 2982000, 2, {{"rolled", 0}, m}, rows);
    const metrics::Interval iv = metrics::paired_bootstrap(v.at(m.name()), v.at("rolled"), 10000, 1);
    std::println("g-diff-test G2c: {} minus rolled {}", m.name(), interval_text(iv));
    decisions.push_back(std::format("test,G2c,{} - rolled,{},,100,{},interval below zero and cost within bound,{}", m.name(), interval_csv(iv),
                                    quiet ? "quiet" : "busy", iv.hi < 0 ? "keep" : "stop"));
    merge_rows(c.results / "g_diff_starts.csv", std::string("phase,setting,seed,method,") + kScoredHeader, rows);
  }
  merge_rows(c.results / "g_diff_decisions.csv", kDecisionHeader, decisions);
}

}  // namespace nfx::study_g
