// nvfx_d2: study D2 (docs/REPORT.md §6.9): the detail layer's constants tuned on the endless statistics.
//
// Study D's rollout effects reach the real floor on neither detail nor motion (REPORT §6.5: motion 10-13% low on fire
// and smoke), and every earlier attempt to improve one statistic cost motion. Here the detail layer's constants are
// tuned by a black-box search (CMA-ES) against an objective that holds both the detail spectrum distance and
// |ln motion ratio|, with penalties for coverage, emission and mean-frame PSNR falling behind v2, on endless runs of
// salt-1 training settings. Candidates are chosen on study G's 10 validation settings and tested once on study B's 10
// held-out settings with fresh seeds. Only the detail layer's constants change; the stepper, the renderer and the start
// points stay v2's.
//
//   nvfx_d2 probe  --effect E [--set k=v,k=v]          statistics of v2 and of v2 with constants changed (training)
//   nvfx_d2 tune   --effect E [--evals N]              CMA-ES on the training objective; candidates to DATA/d2/cand
//   nvfx_d2 val    --effect E                          v2 and the candidates on the validation settings; the choice
//   nvfx_d2 test   --effect E                          once: v2 and the chosen candidate on B's settings, fresh seeds
//   nvfx_d2 write  --effect E                          the v3 candidate file (v2's with the chosen constants)
//   nvfx_d2 cost   --effect E                          ms per 128 x 128 frame, v2 against the candidate
//
// Options: --root DATA (default $NEURALVFX_DATA or /root/nvfx-data), --results DIR (results/experiments),
// --threads N (1).
#include "args.hpp"

#include <neuralfx/clip.hpp>
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
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <numeric>
#include <print>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace nfx;

namespace {

constexpr int kSize = 128;
std::size_t sz(int v) { return static_cast<std::size_t>(v); }
std::string ename(sim::Effect e) { return std::string(sim::effect_name(e)); }
double seconds_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

struct Ctx {
  fs::path data, results;
  int threads = 1;
};

template <class F>
void parallel(int threads, int n, F&& f) {
  std::atomic<int> next{0};
  std::vector<std::jthread> pool;
  for (int t = 0; t < std::max(1, threads); ++t) {
    pool.emplace_back([&] {
      for (int i; (i = next++) < n;) f(i);
    });
  }
}

// --- settings and seeds -------------------------------------------------------------------------------------------------

using Setting = std::array<float, 3>;
bool off_grid(const Setting& s) {
  const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
  return off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f});
}
// Study B's ten held-out settings (the test) and study G's ten validation settings (docs/DCM.md §4).
std::vector<Setting> test_settings() {
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < 10) {
    const Setting s{u(rng), u(rng), u(rng)};
    if (off_grid(s)) v.push_back(s);
  }
  return v;
}
std::vector<Setting> validation_settings() {
  const auto test = test_settings();
  std::mt19937_64 rng(2027);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < 10) {
    const Setting s{u(rng), u(rng), u(rng)};
    const bool far = std::ranges::all_of(test, [&](const Setting& t) {
      return std::sqrt((s[0] - t[0]) * (s[0] - t[0]) + (s[1] - t[1]) * (s[1] - t[1]) + (s[2] - t[2]) * (s[2] - t[2])) >= 0.05f;
    });
    if (off_grid(s) && far) v.push_back(s);
  }
  return v;
}
// Training settings: the controls of the effect's salt-1 recipe runs 0 to n - 1 (the runs the stepper was trained on).
std::vector<Setting> training_settings(sim::Effect e, int n) {
  rollout::SimRecipe r = rollout::recipe_for(e);
  r.salt = 1;
  std::vector<Setting> v;
  for (int i = 0; i < n; ++i) {
    const sim::Params p = rollout::recipe_run(r, static_cast<std::uint64_t>(i));
    v.push_back({p.intensity, p.wind, p.turbulence});
  }
  return v;
}

// Seeds (docs/REPORT.md §6.9): per split a base; real run k of setting s has seed base + 100 s + k, the second set of
// real runs (the floor) base + 50 + 100 s + k, the model's runs model base + 100 s + k.
struct Split {
  std::string name;
  std::vector<Setting> settings;
  int reps = 1;
  std::uint64_t real = 0, model = 0;
  bool floor = false;
};
Split split(sim::Effect e, const std::string& name) {
  Split s;
  s.name = name;
  if (name == "train") {
    s.settings = training_settings(e, 12);
    s.reps = 3;
    s.real = 4'100'000;
    s.model = 4'200'000;
  } else if (name == "val") {
    s.settings = validation_settings();
    s.reps = 8;
    s.real = 4'300'000;
    s.model = 4'400'000;
    s.floor = true;
  } else if (name == "test") {
    s.settings = test_settings();
    s.reps = 8;
    s.real = 4'700'000;
    s.model = 4'800'000;
    s.floor = true;
  } else {
    throw std::invalid_argument("unknown split " + name);
  }
  return s;
}

int frames_of(sim::Effect e) { return e == sim::Effect::explosion ? 89 : 300; }
int warm_of(sim::Effect e) { return e == sim::Effect::explosion ? 1 : 150; }

// --- statistics of runs, cached -------------------------------------------------------------------------------------------

void put_vec(std::ostream& o, const std::vector<double>& v) {
  const auto n = static_cast<std::uint32_t>(v.size());
  o.write(reinterpret_cast<const char*>(&n), 4);
  o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(double)));
}
bool get_vec(std::istream& i, std::vector<double>& v) {
  std::uint32_t n = 0;
  i.read(reinterpret_cast<char*>(&n), 4);
  if (!i || n > 100000) return false;
  v.resize(n);
  i.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(double)));
  return static_cast<bool>(i);
}
void save_stats(const fs::path& p, const metrics::ClipStats& s) {
  fs::create_directories(p.parent_path());
  std::ofstream o(p, std::ios::binary);
  put_vec(o, s.coverage);
  put_vec(o, s.emission);
  put_vec(o, s.spectrum);
  std::vector<double> mf(s.mean_frame.begin(), s.mean_frame.end());
  put_vec(o, mf);
  o.write(reinterpret_cast<const char*>(&s.motion), sizeof(double));
}
bool load_stats(const fs::path& p, metrics::ClipStats& s) {
  std::ifstream i(p, std::ios::binary);
  if (!i) return false;
  std::vector<double> mf;
  if (!get_vec(i, s.coverage) || !get_vec(i, s.emission) || !get_vec(i, s.spectrum) || !get_vec(i, mf)) return false;
  s.mean_frame.assign(mf.begin(), mf.end());
  i.read(reinterpret_cast<char*>(&s.motion), sizeof(double));
  return static_cast<bool>(i);
}

metrics::ClipStats real_stats(const Ctx& c, sim::Effect e, const Setting& s, std::uint64_t seed) {
  const fs::path p = c.data / "d2" / "real" / std::format("{}_{}.stats", ename(e), seed);
  metrics::ClipStats st;
  if (load_stats(p, st)) return st;
  sim::Params q;
  q.effect = e;
  q.intensity = s[0];
  q.wind = s[1];
  q.turbulence = s[2];
  q.seed = seed;
  q.size = kSize;
  sim::Fluid f(q);
  for (int i = 0; i < warm_of(e); ++i) f.step_frame();
  Clip cl;
  cl.allocate(kSize, frames_of(e));
  cl.fps = 30.f;
  for (int i = 0; i < cl.frames; ++i) {
    f.step_frame();
    f.render(cl.frame(i));
  }
  st = metrics::stats(cl);
  save_stats(p, st);
  return st;
}

// The statistics of several runs pooled: per-frame coverage and emission, log spectrum, mean frame and motion averaged
// over the runs. With one run this is the run's own statistics (study D's protocol).
metrics::ClipStats pool(const std::vector<metrics::ClipStats>& v) {
  metrics::ClipStats p = v.at(0);
  const double n = static_cast<double>(v.size());
  for (std::size_t k = 1; k < v.size(); ++k) {
    for (std::size_t i = 0; i < p.coverage.size(); ++i) p.coverage[i] += v[k].coverage[i];
    for (std::size_t i = 0; i < p.emission.size(); ++i) p.emission[i] += v[k].emission[i];
    for (std::size_t i = 0; i < p.spectrum.size(); ++i) p.spectrum[i] += v[k].spectrum[i];
    for (std::size_t i = 0; i < p.mean_frame.size(); ++i) p.mean_frame[i] += v[k].mean_frame[i];
    p.motion += v[k].motion;
  }
  for (double& x : p.coverage) x /= n;
  for (double& x : p.emission) x /= n;
  for (double& x : p.spectrum) x /= n;
  for (float& x : p.mean_frame) x /= static_cast<float>(n);
  p.motion /= n;
  return p;
}

// --- the detail layer's constants ---------------------------------------------------------------------------------------

// The constants searched, by name. kappa follows edge0 and edge1 (1 / mean of the contrast curve, as
// rollout::calibrate_detail sets it).
struct Knob {
  std::string name;
  float lo, hi;
  bool log;  // searched on a log scale
};
const std::vector<Knob>& all_knobs() {
  static const std::vector<Knob> k = {
      {"contrast", 0.f, 1.f, false},   {"edge0", -0.6f, 0.3f, false},     {"edge1", 0.2f, 1.2f, false}, {"swirl", 0.05f, 4.f, true},
      {"swirl_scale", 3.f, 32.f, true}, {"swirl_rate", 0.3f, 6.f, true}, {"grow", 1.f, 8.f, true},  {"advect", 0.7f, 2.f, true},
      {"soften", 0.f, 0.2f, false},
  };
  return k;
}
float get_knob(const rollout::Model& m, const std::string& k) {
  const rollout::DetailSpec& d = m.detail;
  if (k == "contrast") return d.contrast;
  if (k == "edge0") return d.edge0;
  if (k == "edge1") return d.edge1;
  if (k == "swirl") return d.swirl;
  if (k == "swirl_scale") return d.swirl_scale;
  if (k == "swirl_rate") return d.swirl_rate;
  if (k == "grow") return d.grow;
  if (k == "kappa") return d.kappa;
  if (k == "advect") return d.advect;
  if (k == "soften") return d.soften;
  throw std::invalid_argument("unknown constant " + k);
}
void set_knob(rollout::Model& m, const std::string& k, float v) {
  rollout::DetailSpec& d = m.detail;
  if (k == "contrast") d.contrast = v;
  else if (k == "edge0") d.edge0 = v;
  else if (k == "edge1") d.edge1 = v;
  else if (k == "swirl") d.swirl = v;
  else if (k == "swirl_scale") d.swirl_scale = v;
  else if (k == "swirl_rate") d.swirl_rate = v;
  else if (k == "grow") d.grow = v;
  else if (k == "advect") d.advect = v;
  else if (k == "soften") d.soften = v;
  else throw std::invalid_argument("unknown constant " + k);
}
float contrast_kappa(const rollout::Model& m) {  // as rollout::calibrate_detail
  double mean = 0;
  constexpr int n = 20000;
  for (int i = 0; i < n; ++i) {
    const float phi = rollout::noise_flicker(m.noise, 9, static_cast<float>(i % 137) * 0.7f, static_cast<float>(i / 137) * 0.9f, 0.3f);
    const float t = std::clamp((phi - m.detail.edge0) / (m.detail.edge1 - m.detail.edge0), 0.f, 1.f);
    mean += static_cast<double>(t * t * (3.f - 2.f * t));
  }
  return static_cast<float>(n / std::max(1e-6, mean));
}
using Values = std::map<std::string, float>;
rollout::Model with_values(const rollout::Model& base, const Values& v) {
  rollout::Model m = base;
  bool edges = false;
  for (const auto& [k, x] : v) {
    set_knob(m, k, x);
    edges = edges || k == "edge0" || k == "edge1";
  }
  if (m.detail.edge1 <= m.detail.edge0 + 0.05f) m.detail.edge1 = m.detail.edge0 + 0.05f;
  if (edges) m.detail.kappa = contrast_kappa(m);
  return m;
}
std::string values_str(const rollout::Model& m) {
  std::string s;
  for (const Knob& k : all_knobs()) s += std::format("{}{}={:.4g}", s.empty() ? "" : ";", k.name, get_knob(m, k.name));
  s += std::format(";kappa={:.4g}", m.detail.kappa);
  return s;
}
Values parse_values(const std::string& text) {
  Values v;
  std::size_t a = 0;
  while (a < text.size()) {
    std::size_t b = text.find_first_of(",;", a);
    if (b == std::string::npos) b = text.size();
    const std::string kv = text.substr(a, b - a);
    const std::size_t eq = kv.find('=');
    if (eq != std::string::npos && kv.substr(0, eq) != "kappa") v[kv.substr(0, eq)] = std::stof(kv.substr(eq + 1));
    a = b + 1;
  }
  return v;
}

// --- model runs through the runtime -----------------------------------------------------------------------------------

struct Runtime {
  nvfx_effect* e = nullptr;
  explicit Runtime(const rollout::Model& m) {
    std::ostringstream os;
    if (auto r = rollout::save_model(os, m); !r) throw std::runtime_error(r.error());
    const std::string b = os.str();
    if (nvfx_effect_load_memory(b.data(), b.size(), &e) != NVFX_OK) throw std::runtime_error("runtime load failed");
  }
  ~Runtime() { nvfx_effect_free(e); }
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
};

// `frames` frames of the runtime at controls with a seed, played as shipped (shards from start points).
Clip runtime_clip(nvfx_effect* e, const Setting& s, std::uint64_t seed, int frames) {
  nvfx_instance* in = nullptr;
  if (nvfx_instance_create(e, kSize, &in) != NVFX_OK) throw std::runtime_error("instance");
  nvfx_instance_set_controls(in, s.data(), 3);
  nvfx_instance_set_seed(in, seed);
  Clip c;
  c.allocate(kSize, frames);
  c.fps = 30.f;
  for (int f = 0; f < frames; ++f) nvfx_render(in, f / 30.0, c.frame(f).data(), kSize * 4);
  nvfx_instance_free(in);
  return c;
}

// Per setting, the pooled distance of the model's runs from the real runs.
struct Eval {
  std::vector<metrics::StatDistance> d;   // per setting
  std::vector<metrics::StatDistance> d1;  // per setting, study D's protocol: the first play against the first real run
  std::vector<double> mot;                // per setting: pooled model motion (for the record)
  std::vector<std::vector<double>> spec;  // per setting: pooled model log spectrum (diagnostics)
};
struct RealSet {
  std::vector<metrics::ClipStats> ref, other;  // per setting, pooled
  std::vector<metrics::ClipStats> ref1;        // per setting, the first real run alone
  std::vector<double> ref_motion;
};
RealSet real_set(const Ctx& c, sim::Effect e, const Split& sp) {
  RealSet r;
  const int n = static_cast<int>(sp.settings.size());
  r.ref.resize(sz(n));
  r.other.resize(sz(n));
  parallel(c.threads, n * sp.reps * (sp.floor ? 2 : 1), [&](int j) {  // fill the cache
    const int si = j % n, k = (j / n) % sp.reps, o = j / (n * sp.reps);
    (void)real_stats(c, e, sp.settings[sz(si)], sp.real + (o ? 50 : 0) + 100 * static_cast<std::uint64_t>(si) + static_cast<std::uint64_t>(k));
  });
  for (int si = 0; si < n; ++si) {
    for (int o = 0; o < (sp.floor ? 2 : 1); ++o) {
      std::vector<metrics::ClipStats> v;
      for (int k = 0; k < sp.reps; ++k) {
        v.push_back(real_stats(c, e, sp.settings[sz(si)], sp.real + (o ? 50 : 0) + 100 * static_cast<std::uint64_t>(si) + static_cast<std::uint64_t>(k)));
      }
      (o ? r.other : r.ref)[sz(si)] = pool(v);
      if (!o) r.ref1.push_back(v[0]);
    }
    r.ref_motion.push_back(r.ref[sz(si)].motion);
  }
  return r;
}
Eval evaluate(const Ctx& c, sim::Effect e, const rollout::Model& m, const Split& sp, const RealSet& real, int threads) {
  Runtime rt(m);
  const int n = static_cast<int>(sp.settings.size());
  std::vector<metrics::ClipStats> runs(sz(n * sp.reps));
  parallel(threads, n * sp.reps, [&](int j) {
    const int si = j / sp.reps, k = j % sp.reps;
    runs[sz(j)] = metrics::stats(runtime_clip(rt.e, sp.settings[sz(si)], sp.model + 100 * static_cast<std::uint64_t>(si) + static_cast<std::uint64_t>(k), frames_of(e)));
  });
  (void)c;
  Eval ev;
  for (int si = 0; si < n; ++si) {
    const metrics::ClipStats p = pool(std::vector<metrics::ClipStats>(runs.begin() + si * sp.reps, runs.begin() + (si + 1) * sp.reps));
    ev.d.push_back(metrics::distance(real.ref[sz(si)], p));
    ev.d1.push_back(metrics::distance(real.ref1[sz(si)], runs[sz(si * sp.reps)]));
    ev.mot.push_back(p.motion);
    ev.spec.push_back(p.spectrum);
  }
  return ev;
}

// Means over settings.
struct Summary {
  double spec = 0, lnmot = 0, mot = 0, cov = 0, emi = 0, psnr = 0;
};
Summary summarise(const std::vector<metrics::StatDistance>& d) {
  Summary s;
  for (const auto& x : d) {
    s.spec += x.spectrum_l1;
    s.lnmot += std::abs(std::log(std::max(1e-3, x.motion_ratio)));
    s.mot += x.motion_ratio;
    s.cov += x.coverage_l1;
    s.emi += x.emission_l1;
    s.psnr += x.mean_frame_psnr;
  }
  const double n = static_cast<double>(std::max<std::size_t>(1, d.size()));
  s.spec /= n;
  s.lnmot /= n;
  s.mot /= n;
  s.cov /= n;
  s.emi /= n;
  s.psnr /= n;
  return s;
}
std::string summary_str(const Summary& s) {
  return std::format("spec {:.4f} |ln mot| {:.4f} (mot {:.3f}) cov {:.5f} emi {:.5f} psnr {:.3f}", s.spec, s.lnmot, s.mot, s.cov, s.emi, s.psnr);
}

rollout::Model load_or_throw(const fs::path& p) {
  auto m = rollout::load_model(p);
  if (!m) throw std::runtime_error(std::format("{}: {}", p.string(), m.error()));
  return std::move(*m);
}
fs::path v2_path(const Ctx& c, sim::Effect e) { return c.data / "v2" / "models" / std::format("{}.nvfx", ename(e)); }

sim::Effect effect_of(const std::string& s) {
  for (const auto e : sim::kEffects) {
    if (ename(e) == s) return e;
  }
  throw std::invalid_argument("unknown effect " + s);
}

// --- probe ------------------------------------------------------------------------------------------------------------

void step_probe(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const rollout::Model base = load_or_throw(v2_path(c, e));
  const Split sp = split(e, a.str("split", "train"));
  auto t0 = std::chrono::steady_clock::now();
  const RealSet real = real_set(c, e, sp);
  std::println("probe: {} {} real runs ready in {:.0f} s", ename(e), sp.name, seconds_since(t0));
  std::println("v2: {}", values_str(base));
  std::vector<std::string> variants = {""};
  if (a.has("set")) {
    std::string all = a.str("set");
    std::size_t p = 0;
    while (p <= all.size()) {
      const std::size_t q = std::min(all.find('/', p), all.size());
      variants.push_back(all.substr(p, q - p));
      p = q + 1;
    }
  }
  for (const std::string& v : variants) {
    t0 = std::chrono::steady_clock::now();
    const rollout::Model m = with_values(base, parse_values(v));
    const Eval ev = evaluate(c, e, m, sp, real, c.threads);
    std::println("{:<40} {}  ({:.0f} s)", v.empty() ? "v2" : v, summary_str(summarise(ev.d)), seconds_since(t0));
    {  // where the spectrum differs: mean over settings of model - real log10 power, by band of radial bins
      std::string bands;
      for (const auto& [lo, hi] : {std::pair{1, 4}, {5, 8}, {9, 16}, {17, 32}, {33, 48}, {49, 64}}) {
        double sum = 0;
        int n = 0;
        for (std::size_t si = 0; si < ev.spec.size(); ++si) {
          for (int b = lo; b <= hi && b <= static_cast<int>(ev.spec[si].size()); ++b, ++n) sum += ev.spec[si][sz(b - 1)] - real.ref[si].spectrum[sz(b - 1)];
        }
        bands += std::format(" {}-{}: {:+.3f}", lo, hi, n ? sum / n : 0.0);
      }
      std::println("   spectrum, model - real (log10 power) by radial band:{}", bands);
    }
    if (a.flag("verbose")) {
      for (std::size_t si = 0; si < ev.d.size(); ++si) {
        std::println("   setting {} spec {:.4f} mot {:.3f} cov {:.5f} psnr {:.2f}", si, ev.d[si].spectrum_l1, ev.d[si].motion_ratio, ev.d[si].coverage_l1, ev.d[si].mean_frame_psnr);
      }
    }
    std::fflush(stdout);
  }
  if (sp.floor) {
    std::vector<metrics::StatDistance> fl;
    for (std::size_t si = 0; si < real.ref.size(); ++si) fl.push_back(metrics::distance(real.ref[si], real.other[si]));
    std::println("{:<40} {}", "real, other seeds (floor)", summary_str(summarise(fl)));
  }
}

// --- CMA-ES -----------------------------------------------------------------------------------------------------------

// A plain (mu / mu_w, lambda) CMA-ES (Hansen's tutorial, 2016) on [0, 1]^n: points outside the box are evaluated at
// their projection, plus a penalty of the squared distance.
class Cma {
 public:
  Cma(std::vector<double> mean, double sigma, std::uint64_t seed) : n_(static_cast<int>(mean.size())), mean_(std::move(mean)), sigma_(sigma), rng_(seed) {
    const double n = n_;
    lambda_ = 4 + static_cast<int>(std::floor(3.0 * std::log(n)));
    mu_ = lambda_ / 2;
    double sw = 0, sw2 = 0;
    for (int i = 0; i < mu_; ++i) {
      w_.push_back(std::log(mu_ + 0.5) - std::log(i + 1.0));
      sw += w_.back();
    }
    for (double& x : w_) {
      x /= sw;
      sw2 += x * x;
    }
    mueff_ = 1.0 / sw2;
    cs_ = (mueff_ + 2) / (n + mueff_ + 5);
    ds_ = 1 + 2 * std::max(0.0, std::sqrt((mueff_ - 1) / (n + 1)) - 1) + cs_;
    cc_ = (4 + mueff_ / n) / (n + 4 + 2 * mueff_ / n);
    c1_ = 2 / ((n + 1.3) * (n + 1.3) + mueff_);
    cmu_ = std::min(1 - c1_, 2 * (mueff_ - 2 + 1 / mueff_) / ((n + 2) * (n + 2) + mueff_));
    chin_ = std::sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));
    ps_.assign(sz(n_), 0.0);
    pc_.assign(sz(n_), 0.0);
    C_.assign(sz(n_ * n_), 0.0);
    B_.assign(sz(n_ * n_), 0.0);
    D_.assign(sz(n_), 1.0);
    for (int i = 0; i < n_; ++i) C_[sz(i * n_ + i)] = B_[sz(i * n_ + i)] = 1.0;
  }
  int lambda() const { return lambda_; }
  double sigma() const { return sigma_; }
  const std::vector<double>& mean() const { return mean_; }
  std::vector<std::vector<double>> ask() {
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<std::vector<double>> xs;
    for (int k = 0; k < lambda_; ++k) {
      std::vector<double> z(sz(n_)), x(mean_);
      for (double& v : z) v = nd(rng_);
      for (int i = 0; i < n_; ++i) {
        double y = 0;
        for (int j = 0; j < n_; ++j) y += B_[sz(i * n_ + j)] * D_[sz(j)] * z[sz(j)];
        x[sz(i)] += sigma_ * y;
      }
      xs.push_back(std::move(x));
    }
    return xs;
  }
  void tell(const std::vector<std::vector<double>>& xs, const std::vector<double>& f) {
    std::vector<int> order(xs.size());
    std::iota(order.begin(), order.end(), 0);
    std::ranges::stable_sort(order, [&](int a, int b) { return f[sz(a)] < f[sz(b)]; });
    const std::vector<double> old = mean_;
    std::ranges::fill(mean_, 0.0);
    for (int i = 0; i < mu_; ++i) {
      for (int j = 0; j < n_; ++j) mean_[sz(j)] += w_[sz(i)] * xs[sz(order[sz(i)])][sz(j)];
    }
    std::vector<double> yw(sz(n_)), t(sz(n_)), cy(sz(n_));
    for (int j = 0; j < n_; ++j) yw[sz(j)] = (mean_[sz(j)] - old[sz(j)]) / sigma_;
    for (int a = 0; a < n_; ++a) {  // C^{-1/2} yw = B D^-1 B^T yw
      double s = 0;
      for (int j = 0; j < n_; ++j) s += B_[sz(j * n_ + a)] * yw[sz(j)];
      t[sz(a)] = s / D_[sz(a)];
    }
    for (int i = 0; i < n_; ++i) {
      double s = 0;
      for (int a = 0; a < n_; ++a) s += B_[sz(i * n_ + a)] * t[sz(a)];
      cy[sz(i)] = s;
    }
    double psn = 0;
    for (int i = 0; i < n_; ++i) {
      ps_[sz(i)] = (1 - cs_) * ps_[sz(i)] + std::sqrt(cs_ * (2 - cs_) * mueff_) * cy[sz(i)];
      psn += ps_[sz(i)] * ps_[sz(i)];
    }
    psn = std::sqrt(psn);
    ++gen_;
    const bool hsig = psn / std::sqrt(1 - std::pow(1 - cs_, 2.0 * gen_)) / chin_ < 1.4 + 2.0 / (n_ + 1);
    for (int i = 0; i < n_; ++i) pc_[sz(i)] = (1 - cc_) * pc_[sz(i)] + (hsig ? std::sqrt(cc_ * (2 - cc_) * mueff_) : 0.0) * yw[sz(i)];
    for (int i = 0; i < n_; ++i) {
      for (int j = 0; j < n_; ++j) {
        double rank_mu = 0;
        for (int k = 0; k < mu_; ++k) {
          const auto& x = xs[sz(order[sz(k)])];
          rank_mu += w_[sz(k)] * (x[sz(i)] - old[sz(i)]) / sigma_ * (x[sz(j)] - old[sz(j)]) / sigma_;
        }
        double& c = C_[sz(i * n_ + j)];
        c = (1 - c1_ - cmu_) * c + c1_ * (pc_[sz(i)] * pc_[sz(j)] + (hsig ? 0.0 : cc_ * (2 - cc_) * c)) + cmu_ * rank_mu;
      }
    }
    sigma_ *= std::exp((cs_ / ds_) * (psn / chin_ - 1));
    eigen();
  }

 private:
  // C = B diag(D^2) B^T by Jacobi rotations (n is small).
  void eigen() {
    std::vector<double> a = C_, v(sz(n_ * n_), 0.0);
    for (int i = 0; i < n_; ++i) v[sz(i * n_ + i)] = 1.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
      double off = 0;
      for (int p = 0; p < n_; ++p) {
        for (int q = p + 1; q < n_; ++q) off += a[sz(p * n_ + q)] * a[sz(p * n_ + q)];
      }
      if (off < 1e-30) break;
      for (int p = 0; p < n_; ++p) {
        for (int q = p + 1; q < n_; ++q) {
          const double apq = a[sz(p * n_ + q)];
          if (std::abs(apq) < 1e-300) continue;
          const double theta = (a[sz(q * n_ + q)] - a[sz(p * n_ + p)]) / (2 * apq);
          const double tt = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1));
          const double cth = 1 / std::sqrt(tt * tt + 1), sth = tt * cth;
          for (int k = 0; k < n_; ++k) {
            const double akp = a[sz(k * n_ + p)], akq = a[sz(k * n_ + q)];
            a[sz(k * n_ + p)] = cth * akp - sth * akq;
            a[sz(k * n_ + q)] = sth * akp + cth * akq;
          }
          for (int k = 0; k < n_; ++k) {
            const double apk = a[sz(p * n_ + k)], aqk = a[sz(q * n_ + k)];
            a[sz(p * n_ + k)] = cth * apk - sth * aqk;
            a[sz(q * n_ + k)] = sth * apk + cth * aqk;
          }
          for (int k = 0; k < n_; ++k) {
            const double vkp = v[sz(k * n_ + p)], vkq = v[sz(k * n_ + q)];
            v[sz(k * n_ + p)] = cth * vkp - sth * vkq;
            v[sz(k * n_ + q)] = sth * vkp + cth * vkq;
          }
        }
      }
    }
    B_ = v;
    for (int i = 0; i < n_; ++i) D_[sz(i)] = std::sqrt(std::max(1e-20, a[sz(i * n_ + i)]));
  }

  int n_, lambda_ = 0, mu_ = 0, gen_ = 0;
  std::vector<double> mean_, w_, ps_, pc_, C_, B_, D_;
  double sigma_, mueff_ = 0, cs_ = 0, ds_ = 0, cc_ = 0, c1_ = 0, cmu_ = 0, chin_ = 0;
  std::mt19937_64 rng_;
};

// --- tune -------------------------------------------------------------------------------------------------------------

// The knob sets searched: "all" (every constant, the two new ones included) and "old" (the constants v2's file
// already has, so a candidate needs no new file version).
std::vector<Knob> knob_set(const std::string& name) {
  std::vector<Knob> v;
  for (const Knob& k : all_knobs()) {
    if (name == "all" || (name == "old" && k.name != "advect" && k.name != "soften")) v.push_back(k);
  }
  if (v.empty()) throw std::invalid_argument("unknown knob set " + name);
  return v;
}
double to_unit(const Knob& k, float v) {
  const double x = k.log ? (std::log(v) - std::log(k.lo)) / (std::log(k.hi) - std::log(k.lo)) : (v - k.lo) / (k.hi - k.lo);
  return std::clamp(x, 0.0, 1.0);
}
float from_unit(const Knob& k, double x) {
  x = std::clamp(x, 0.0, 1.0);
  const double v = k.log ? std::exp(std::log(k.lo) + x * (std::log(k.hi) - std::log(k.lo))) : k.lo + x * (k.hi - k.lo);
  return std::stof(std::format("{:.4g}", v));  // what is applied, written and keyed is the same 4-digit value
}

// The training objective (docs/REPORT.md §6.9), on means over the training settings (pooled runs): with r_s and r_m
// the detail spectrum distance and |ln motion ratio| as fractions of v2's, J = (r_s + r_m) / 2 + max(r_s, r_m), so
// neither can be traded for the other (2 at v2); plus 20 times the relative excess of coverage and emission distance
// over v2's and 4 times the mean-frame PSNR lost in dB, so the picture cannot be traded for either.
double objective(const Summary& s, const Summary& v2) {
  const double rs = s.spec / std::max(1e-6, v2.spec), rm = s.lnmot / std::max(1e-6, v2.lnmot);
  double j = 0.5 * (rs + rm) + std::max(rs, rm);
  j += 20 * std::max(0.0, s.cov - v2.cov) / std::max(1e-5, v2.cov);
  j += 20 * std::max(0.0, s.emi - v2.emi) / std::max(1e-5, v2.emi);
  j += 4 * std::max(0.0, v2.psnr - s.psnr);
  return j;
}

struct EvalRow {
  std::string values;
  Summary s;
};
std::map<std::string, EvalRow> read_evals(const fs::path& p) {
  std::map<std::string, EvalRow> m;
  std::ifstream in(p);
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    std::vector<std::string> cell;
    std::stringstream ss(line);
    for (std::string x; std::getline(ss, x, ',');) cell.push_back(x);
    if (cell.size() < 8) continue;
    EvalRow r;
    r.values = cell[1];
    r.s = {std::stod(cell[2]), std::stod(cell[3]), std::stod(cell[4]), std::stod(cell[5]), std::stod(cell[6]), std::stod(cell[7])};
    m[r.values] = r;
  }
  return m;
}

void step_tune(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const rollout::Model base = load_or_throw(v2_path(c, e));
  const Split sp = split(e, "train");
  const std::string set = a.str("knobs", "all");
  const std::vector<Knob> knobs = knob_set(set);
  const int evals = a.i("evals", 120);
  const fs::path dir = c.data / "d2" / "cand";
  fs::create_directories(dir);
  const fs::path log = dir / std::format("{}_{}_evals.csv", ename(e), set);
  std::map<std::string, EvalRow> cache = read_evals(log);
  if (!fs::exists(log)) std::ofstream(log) << "n,values,spec,lnmot,mot,cov,emi,psnr,objective\n";
  auto t0 = std::chrono::steady_clock::now();
  const RealSet real = real_set(c, e, sp);
  std::println("tune: {} {} knobs, real runs ready in {:.0f} s", ename(e), set, seconds_since(t0));
  int done = 0;
  const auto run = [&](const Values& v) -> Summary {
    const rollout::Model m = with_values(base, v);
    const std::string key = values_str(m);
    if (const auto it = cache.find(key); it != cache.end()) return it->second.s;
    const auto t1 = std::chrono::steady_clock::now();
    const Summary s = summarise(evaluate(c, e, m, sp, real, c.threads).d);
    cache[key] = {key, s};
    std::ofstream(log, std::ios::app) << std::format("{},{},{:.6f},{:.6f},{:.6f},{:.7f},{:.7f},{:.5f},\n", ++done, key, s.spec, s.lnmot, s.mot, s.cov, s.emi, s.psnr);
    std::println("  eval {} ({:.0f} s): {}", key, seconds_since(t1), summary_str(s));
    std::fflush(stdout);
    return s;
  };
  const Summary v2 = run({});
  std::println("tune: v2 {} objective {:.4f}", summary_str(v2), objective(v2, v2));
  std::vector<double> x0;
  for (const Knob& k : knobs) x0.push_back(to_unit(k, get_knob(base, k.name)));
  Cma cma(x0, a.f("sigma", 0.2f), 20261011);
  const auto values_of = [&](const std::vector<double>& x) {
    Values v;
    for (std::size_t i = 0; i < knobs.size(); ++i) v[knobs[i].name] = from_unit(knobs[i], x[i]);
    return v;
  };
  double best = objective(v2, v2);
  Values best_v;
  int used = 1, gen = 0;
  std::vector<std::string> gen_rows;
  while (used + cma.lambda() <= evals) {
    const auto xs = cma.ask();
    std::vector<double> f;
    for (const auto& x : xs) {
      double out = 0;
      for (const double q : x) out += q < 0 ? q * q : (q > 1 ? (q - 1) * (q - 1) : 0.0);
      const Values v = values_of(x);
      const double j = objective(run(v), v2) + out;
      f.push_back(j);
      if (j < best) {
        best = j;
        best_v = v;
      }
      ++used;
    }
    cma.tell(xs, f);
    ++gen;
    const Summary sm = run(values_of(cma.mean()));
    ++used;
    std::println("tune: {} generation {} sigma {:.3f}: best {:.4f}, mean {:.4f} ({})", ename(e), gen, cma.sigma(), best, objective(sm, v2),
                 values_str(with_values(base, values_of(cma.mean()))));
    std::fflush(stdout);
  }
  std::println("tune: {} done after {} evaluations in {:.0f} s; best objective {:.4f}: {}", ename(e), used, seconds_since(t0), best,
               values_str(with_values(base, best_v)));
}

// --- candidates -------------------------------------------------------------------------------------------------------

struct Candidate {
  std::string name, values;
};
std::vector<Candidate> read_candidates(const fs::path& p) {
  std::vector<Candidate> v;
  std::ifstream in(p);
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    const auto a = line.find(','), b = line.find(',', a + 1);
    if (a == std::string::npos) continue;
    v.push_back({line.substr(0, a), line.substr(a + 1, b == std::string::npos ? std::string::npos : b - a - 1)});
  }
  return v;
}
fs::path candidates_path(const Ctx& c, sim::Effect e) { return c.data / "d2" / "cand" / std::format("{}_candidates.csv", ename(e)); }

// From each search's evaluations (knob sets "all" and "old"): the one with the lowest training objective, and the
// lowest among those no worse than v2 in any of the five statistics on the training settings (when it is another).
void step_candidates(const Ctx& c, sim::Effect e) {
  const rollout::Model base = load_or_throw(v2_path(c, e));
  const std::string v2key = values_str(base);
  std::vector<std::string> rows;
  std::vector<std::string> summary;
  for (const std::string set : {"all", "old"}) {
    const auto evals = read_evals(c.data / "d2" / "cand" / std::format("{}_{}_evals.csv", ename(e), set));
    const auto it = evals.find(v2key);
    if (it == evals.end()) continue;
    const Summary v2 = it->second.s;
    const EvalRow *best = nullptr, *feasible = nullptr;
    double fb = 1e9, ff = 1e9;
    for (const auto& [k, r] : evals) {
      if (k == v2key) continue;
      const double j = objective(r.s, v2);
      if (j < fb) {
        fb = j;
        best = &r;
      }
      const bool ok = r.s.spec < v2.spec && r.s.lnmot < v2.lnmot && r.s.cov <= v2.cov && r.s.emi <= v2.emi && r.s.psnr >= v2.psnr;
      if (ok && j < ff) {
        ff = j;
        feasible = &r;
      }
    }
    if (best) {
      rows.push_back(std::format("{}_best,{},{:.5f},{}", set, best->values, fb, evals.size()));
      summary.push_back(std::format("{} best {:.4f} ({}): {}", set, fb, summary_str(best->s), best->values));
    }
    if (feasible && feasible != best) {
      rows.push_back(std::format("{}_feasible,{},{:.5f},{}", set, feasible->values, ff, evals.size()));
      summary.push_back(std::format("{} feasible {:.4f} ({}): {}", set, ff, summary_str(feasible->s), feasible->values));
    }
    summary.push_back(std::format("{} v2 {:.4f} ({})", set, objective(v2, v2), summary_str(v2)));
  }
  std::ofstream o(candidates_path(c, e));
  o << "name,values,training_objective,evaluations\n";
  for (const auto& r : rows) o << r << "\n";
  for (const auto& s : summary) std::println("candidates: {}: {}", ename(e), s);
}

// --- validation and test ----------------------------------------------------------------------------------------------

std::vector<double> stat_of(const std::vector<metrics::StatDistance>& d, int k) {
  std::vector<double> v;
  for (const auto& x : d) {
    switch (k) {
      case 0: v.push_back(x.spectrum_l1); break;
      case 1: v.push_back(std::abs(std::log(std::max(1e-3, x.motion_ratio)))); break;
      case 2: v.push_back(x.coverage_l1); break;
      case 3: v.push_back(x.emission_l1); break;
      default: v.push_back(x.mean_frame_psnr); break;
    }
  }
  return v;
}
const std::array<std::string, 5> kStatNames = {"spectrum_l1", "abs_ln_motion_ratio", "coverage_l1", "emission_l1", "mean_frame_psnr"};

// The rule (docs/REPORT.md §6.9), candidate - v2 paired over the settings: spectrum distance and |ln motion ratio| lower
// with intervals below zero; coverage and emission distance not worse (no interval above zero); mean-frame PSNR not
// worse (no interval below zero).
struct RuleResult {
  std::array<metrics::Interval, 5> iv{};
  bool spec = false, motion = false, others = false;
  bool pass() const { return spec && motion && others; }
};
RuleResult apply_rule(const std::vector<metrics::StatDistance>& cand, const std::vector<metrics::StatDistance>& v2) {
  RuleResult r;
  for (int k = 0; k < 5; ++k) {
    const auto a = stat_of(cand, k), b = stat_of(v2, k);
    r.iv[sz(k)] = metrics::paired_bootstrap(a, b, 10000, 2027);
  }
  r.spec = r.iv[0].hi < 0;
  r.motion = r.iv[1].hi < 0;
  r.others = !(r.iv[2].lo > 0) && !(r.iv[3].lo > 0) && !(r.iv[4].hi < 0);
  return r;
}

void merge_rows(const fs::path& path, const std::string& header, const std::vector<std::string>& rows, const std::string& effect) {
  std::vector<std::string> keep;
  if (std::ifstream in(path); in) {
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      if (!line.empty() && line.substr(0, line.find(',')) != effect) keep.push_back(line);
    }
  }
  for (const auto& r : rows) keep.push_back(r);
  std::ranges::stable_sort(keep, [](const std::string& a, const std::string& b) {
    const auto rank = [](const std::string& s) {
      const std::string e = s.substr(0, s.find(','));
      return e == "fire" ? 0 : e == "smoke" ? 1 : 2;
    };
    return rank(a) < rank(b);
  });
  fs::create_directories(path.parent_path());
  std::ofstream o(path);
  o << header << "\n";
  for (const auto& r : keep) o << r << "\n";
}

struct Scored {
  std::string name, values;
  std::vector<metrics::StatDistance> d, d1;  // pooled runs (the rule); one run each (study D's protocol, reported)
};
std::vector<Scored> score_models(const Ctx& c, sim::Effect e, const Split& sp, const RealSet& real, const std::vector<Candidate>& cands, std::vector<std::string>& rows) {
  const rollout::Model base = load_or_throw(v2_path(c, e));
  std::vector<Scored> out;
  std::vector<Candidate> all = {{"v2", ""}};
  all.insert(all.end(), cands.begin(), cands.end());
  for (const Candidate& cd : all) {
    const auto t0 = std::chrono::steady_clock::now();
    const rollout::Model m = with_values(base, parse_values(cd.values));
    const Eval ev = evaluate(c, e, m, sp, real, c.threads);
    Scored s{cd.name, values_str(m), ev.d, ev.d1};
    std::println("{}: {} {}: {} ({:.0f} s)", sp.name, ename(e), cd.name, summary_str(summarise(s.d)), seconds_since(t0));
    std::fflush(stdout);
    out.push_back(std::move(s));
  }
  if (sp.floor) {
    Scored f{"real_other_seeds", "", {}, {}};
    for (std::size_t si = 0; si < real.ref.size(); ++si) f.d.push_back(metrics::distance(real.ref[si], real.other[si]));
    std::println("{}: {} floor: {}", sp.name, ename(e), summary_str(summarise(f.d)));
    out.push_back(std::move(f));
  }
  for (const Scored& s : out) {
    for (int pooled = 1; pooled >= 0; --pooled) {
      const auto& d = pooled ? s.d : s.d1;
      for (std::size_t si = 0; si < d.size(); ++si) {
        const auto& x = d[si];
        rows.push_back(std::format("{},{},{},{},{:.5f},{:.5f},{:.5f},{:.6f},{:.6f},{:.4f}", ename(e), s.name, pooled ? std::format("pooled{}", sp.reps) : "one",
                                   si, x.spectrum_l1, x.motion_ratio, std::abs(std::log(std::max(1e-3, x.motion_ratio))), x.coverage_l1, x.emission_l1,
                                   x.mean_frame_psnr));
      }
    }
  }
  return out;
}
const char* kStatsHeader = "effect,model,runs,setting,spectrum_l1,motion_ratio,abs_ln_motion_ratio,coverage_l1,emission_l1,mean_frame_psnr";
const char* kCompareHeader = "effect,model,statistic,mean_difference,lo,hi,candidate_mean,v2_mean";

std::vector<std::string> compare_rows(sim::Effect e, const Scored& cand, const Scored& v2, const RuleResult& r) {
  std::vector<std::string> rows;
  for (int k = 0; k < 5; ++k) {
    const auto a = stat_of(cand.d, k), b = stat_of(v2.d, k);
    const double ma = std::accumulate(a.begin(), a.end(), 0.0) / static_cast<double>(a.size());
    const double mb = std::accumulate(b.begin(), b.end(), 0.0) / static_cast<double>(b.size());
    rows.push_back(std::format("{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}", ename(e), cand.name, kStatNames[sz(k)], r.iv[sz(k)].mean, r.iv[sz(k)].lo,
                               r.iv[sz(k)].hi, ma, mb));
  }
  return rows;
}

void step_val(const Ctx& c, sim::Effect e) {
  const Split sp = split(e, "val");
  const auto cands = read_candidates(candidates_path(c, e));
  if (cands.empty()) throw std::runtime_error("no candidates: run tune and candidates first");
  auto t0 = std::chrono::steady_clock::now();
  const RealSet real = real_set(c, e, sp);
  std::println("val: {} real runs ready in {:.0f} s", ename(e), seconds_since(t0));
  std::vector<std::string> stat_rows, cmp_rows, choice_rows;
  const auto scored = score_models(c, e, sp, real, cands, stat_rows);
  const Scored& v2 = scored[0];
  int chosen = -1;
  double best = 1e9;
  for (std::size_t m = 1; m <= cands.size(); ++m) {
    const RuleResult r = apply_rule(scored[m].d, v2.d);
    for (auto& row : compare_rows(e, scored[m], v2, r)) cmp_rows.push_back(row);
    const Summary s = summarise(scored[m].d);
    const double score = s.spec + s.lnmot;
    choice_rows.push_back(std::format("{},{},{},{},{},{},{:.5f},{}", ename(e), scored[m].name, r.spec ? 1 : 0, r.motion ? 1 : 0, r.others ? 1 : 0,
                                      r.pass() ? 1 : 0, score, scored[m].values));
    std::println("val: {} {}: spectrum {:+.4f} [{:+.4f}, {:+.4f}], |ln motion| {:+.4f} [{:+.4f}, {:+.4f}], coverage {:+.5f} [{:+.5f}, {:+.5f}], "
                 "emission {:+.5f} [{:+.5f}, {:+.5f}], PSNR {:+.3f} [{:+.3f}, {:+.3f}] -> {}",
                 ename(e), scored[m].name, r.iv[0].mean, r.iv[0].lo, r.iv[0].hi, r.iv[1].mean, r.iv[1].lo, r.iv[1].hi, r.iv[2].mean, r.iv[2].lo, r.iv[2].hi,
                 r.iv[3].mean, r.iv[3].lo, r.iv[3].hi, r.iv[4].mean, r.iv[4].lo, r.iv[4].hi, r.pass() ? "meets the rule" : "does not meet it");
    if (r.pass() && score < best) {
      best = score;
      chosen = static_cast<int>(m);
    }
  }
  const fs::path dir = c.data / "d2" / "chosen";
  fs::create_directories(dir);
  if (chosen > 0) {
    std::ofstream(dir / std::format("{}.txt", ename(e))) << scored[sz(chosen)].name << "," << scored[sz(chosen)].values << "\n";
    std::println("val: {}: chosen {} ({})", ename(e), scored[sz(chosen)].name, scored[sz(chosen)].values);
  } else {
    fs::remove(dir / std::format("{}.txt", ename(e)));
    std::println("val: {}: no candidate meets the rule on validation; not tested", ename(e));
  }
  merge_rows(c.results / "d2_val_stats.csv", kStatsHeader, stat_rows, ename(e));
  merge_rows(c.results / "d2_val_compare.csv", kCompareHeader, cmp_rows, ename(e));
  choice_rows.push_back(std::format("{},chosen,,,,,,{}", ename(e), chosen > 0 ? scored[sz(chosen)].name : "none"));
  merge_rows(c.results / "d2_val_choice.csv", "effect,model,spectrum_better,motion_better,others_not_worse,meets_rule,val_spec_plus_lnmot,values", choice_rows,
             ename(e));
}

// --- cost ---------------------------------------------------------------------------------------------------------------

double thread_ms() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) * 1e3 + static_cast<double>(ts.tv_nsec) * 1e-6;
}
// Thread CPU time per 128 x 128 frame through nvfx_render, v2 against the candidate, interleaved repetitions: per
// repetition the median over frames; returned: the median over repetitions of each and of their paired difference.
std::array<double, 3> frame_cost(sim::Effect e, const rollout::Model& v2, const rollout::Model& cand, int reps) {
  Runtime a(v2), b(cand);
  const int frames = e == sim::Effect::explosion ? 88 : 200;
  std::vector<double> ma, mb, md;
  for (int r = 0; r < reps; ++r) {
    std::array<double, 2> med{};
    for (int which = 0; which < 2; ++which) {
      nvfx_instance* in = nullptr;
      if (nvfx_instance_create((which ? b : a).e, kSize, &in) != NVFX_OK) throw std::runtime_error("instance");
      const float ctl[3] = {0.6f, 0.5f, 0.6f};
      nvfx_instance_set_controls(in, ctl, 3);
      nvfx_instance_set_seed(in, 77 + static_cast<std::uint64_t>(r));
      std::vector<std::uint8_t> buf(kSize * kSize * 4);
      nvfx_render(in, 0.0, buf.data(), kSize * 4);
      std::vector<double> ms;
      for (int f = 1; f <= frames; ++f) {
        const double t0 = thread_ms();
        nvfx_render(in, f / 30.0, buf.data(), kSize * 4);
        ms.push_back(thread_ms() - t0);
      }
      nvfx_instance_free(in);
      std::ranges::sort(ms);
      med[sz(which)] = ms[ms.size() / 2];
    }
    ma.push_back(med[0]);
    mb.push_back(med[1]);
    md.push_back(med[1] - med[0]);
  }
  const auto median = [](std::vector<double> v) {
    std::ranges::sort(v);
    return v[v.size() / 2];
  };
  return {median(ma), median(mb), median(md)};
}

// --- test (once) --------------------------------------------------------------------------------------------------------

void step_test(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const fs::path chosen_file = c.data / "d2" / "chosen" / std::format("{}.txt", ename(e));
  const fs::path done = c.data / "d2" / std::format("test_{}.done", ename(e));
  if (fs::exists(done) && !a.flag("again")) throw std::runtime_error("the test of " + ename(e) + " has run (it runs once)");
  std::vector<Candidate> chosen;
  if (std::ifstream in(chosen_file); in) {
    std::string line;
    std::getline(in, line);
    const auto k = line.find(',');
    chosen.push_back({line.substr(0, k), line.substr(k + 1)});
  }
  if (chosen.empty()) {
    std::println("test: {}: nothing chosen on validation; not tested", ename(e));
    return;
  }
  std::ofstream(done) << "started\n";
  const Split sp = split(e, "test");
  auto t0 = std::chrono::steady_clock::now();
  const RealSet real = real_set(c, e, sp);
  std::println("test: {} real runs ready in {:.0f} s", ename(e), seconds_since(t0));
  std::vector<std::string> stat_rows, cmp_rows;
  const auto scored = score_models(c, e, sp, real, chosen, stat_rows);
  const RuleResult r = apply_rule(scored[1].d, scored[0].d);
  cmp_rows = compare_rows(e, scored[1], scored[0], r);
  const rollout::Model base = load_or_throw(v2_path(c, e));
  const auto cost = frame_cost(e, base, with_values(base, parse_values(chosen[0].values)), a.i("cost-reps", 7));
  const bool cheap = cost[2] <= 0.2;
  const bool keep = r.pass() && cheap;
  std::println("test: {} {}: spectrum {:+.4f} [{:+.4f}, {:+.4f}], |ln motion| {:+.4f} [{:+.4f}, {:+.4f}], coverage {:+.5f} [{:+.5f}, {:+.5f}], "
               "emission {:+.5f} [{:+.5f}, {:+.5f}], PSNR {:+.3f} [{:+.3f}, {:+.3f}]; cost {:.3f} -> {:.3f} ms ({:+.3f}) -> {}",
               ename(e), chosen[0].name, r.iv[0].mean, r.iv[0].lo, r.iv[0].hi, r.iv[1].mean, r.iv[1].lo, r.iv[1].hi, r.iv[2].mean, r.iv[2].lo, r.iv[2].hi,
               r.iv[3].mean, r.iv[3].lo, r.iv[3].hi, r.iv[4].mean, r.iv[4].lo, r.iv[4].hi, cost[0], cost[1], cost[2], keep ? "KEPT" : "v2 stays");
  merge_rows(c.results / "d2_test_stats.csv", kStatsHeader, stat_rows, ename(e));
  merge_rows(c.results / "d2_test_compare.csv", kCompareHeader, cmp_rows, ename(e));
  merge_rows(c.results / "d2_decisions.csv", "effect,candidate,spectrum_better,motion_better,others_not_worse,v2_ms,candidate_ms,extra_ms,cost_ok,decision,values",
             {std::format("{},{},{},{},{},{:.4f},{:.4f},{:+.4f},{},{},{}", ename(e), chosen[0].name, r.spec ? 1 : 0, r.motion ? 1 : 0, r.others ? 1 : 0, cost[0],
                          cost[1], cost[2], cheap ? 1 : 0, keep ? "keep_candidate" : "keep_v2", scored[1].values)},
             ename(e));
  std::ofstream(done) << "done\n";
}

// --- cost alone, and the v3 candidate files ------------------------------------------------------------------------------

void step_cost(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const rollout::Model base = load_or_throw(v2_path(c, e));
  const rollout::Model cand = with_values(base, parse_values(a.need("values")));
  const auto cost = frame_cost(e, base, cand, a.i("cost-reps", 7));
  std::println("cost: {}: v2 {:.3f} ms, candidate {:.3f} ms, difference {:+.3f} ms (median of {} paired repetitions)", ename(e), cost[0], cost[1], cost[2],
               a.i("cost-reps", 7));
}

// v2's file with the decided constants (kept: the candidate's; not kept: v2's unchanged) under DATA/d2/models.
void step_write(const Ctx& c, sim::Effect e) {
  const fs::path out = c.data / "d2" / "models" / std::format("{}.nvfx", ename(e));
  fs::create_directories(out.parent_path());
  std::string decision, values;
  if (std::ifstream in(c.results / "d2_decisions.csv"); in) {
    std::string line;
    while (std::getline(in, line)) {
      if (line.rfind(ename(e) + ",", 0) != 0) continue;
      std::vector<std::string> cell;
      std::stringstream ss(line);
      for (std::string x; std::getline(ss, x, ',');) cell.push_back(x);
      decision = cell.at(9);
      values = cell.at(10);
    }
  }
  const rollout::Model base = load_or_throw(v2_path(c, e));
  if (decision == "keep_candidate") {
    const rollout::Model m = with_values(base, parse_values(values));
    if (auto r = rollout::save_model(out, m); !r) throw std::runtime_error(r.error());
    std::println("write: {}: v2 with {} -> {}", ename(e), values_str(m), out.string());
  } else {
    fs::copy_file(v2_path(c, e), out, fs::copy_options::overwrite_existing);
    std::println("write: {}: v2 unchanged ({}) -> {}", ename(e), decision.empty() ? "not tested" : decision, out.string());
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    tools::Args a(argc, argv, {"verbose", "again"});
    if (a.positional().empty()) {
      std::println("nvfx_d2 probe|tune|val|test|write|cost --effect E [--root DIR] [--results DIR] [--threads N]");
      return 2;
    }
    Ctx c;
    const char* env = std::getenv("NEURALVFX_DATA");
    c.data = a.str("root", env ? env : "/root/nvfx-data");
    c.results = a.str("results", "results/experiments");
    c.threads = a.i("threads", 1);
    const std::string step = a.positional()[0];
    const sim::Effect e = effect_of(a.need("effect"));
    if (step == "roundtrip") {  // v2's file, loaded and saved again, is the same bytes
      std::ifstream in(v2_path(c, e), std::ios::binary);
      const std::string file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      std::ostringstream os;
      if (auto r = rollout::save_model(os, load_or_throw(v2_path(c, e))); !r) throw std::runtime_error(r.error());
      std::println("{}: {} bytes, saved again {} bytes, {}", ename(e), file.size(), os.str().size(), file == os.str() ? "identical" : "DIFFERENT");
    } else if (step == "probe") step_probe(c, e, a);
    else if (step == "tune") step_tune(c, e, a);
    else if (step == "candidates") step_candidates(c, e);
    else if (step == "val") step_val(c, e);
    else if (step == "test") step_test(c, e, a);
    else if (step == "cost") step_cost(c, e, a);
    else if (step == "write") step_write(c, e);
    else throw std::invalid_argument("unknown step " + step);
    a.warn_unused();
  } catch (const std::exception& e) {
    std::println(stderr, "nvfx_d2: {}", e.what());
    return 1;
  }
  return 0;
}
