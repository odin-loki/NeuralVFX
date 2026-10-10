// nvfx_g_extras: study G's extras (docs/DCM.md §10, stage S8) on study D's rollout effects (v1, read only).
//
//   g4a-pilot            the renderers alone on validation tracking runs (the explosion's first second)
//   g4a-rows             renderer rows: 24 training runs per effect tracked from their true state
//   g4a-train            the renderer mixer, one per configuration, trained on every effect's rows and frozen
//   g4a-eval --split S   tracking and endless statistics, v1's learned renderer against the mixers (S: val or test)
//   g5a-train            the shard critic per effect (real windows against the model's own shard starts)
//   g5a-eval --split S   shards with K candidates each, scored against real runs
//   g5b-train            the update mixer for smoke: one step from the truth, then an own-rollout pass
//   g5b-eval --split S   tracking and endless statistics, v1 against fixed blends and the mixers
//   bench                thread CPU time of each design's parts
//   summary              paired bootstrap tables of every CSV, the choices and the decisions (results/.../g_extras_summary.md)
//
// Options: --effects fire,smoke,explosion  --threads 1  --data DIR (default NEURALVFX_DATA/g/extras)
//          --models DIR (default NEURALVFX_DATA/experiments/models/d)  --results DIR (default results/experiments)
//          --quick (2 settings, short runs, under DATA/quick)  --method NAME (g4a-eval/g5b-eval --split test: what to test)
// Every eval step appends finished units to its CSV and skips units already there, so a restarted run resumes.
#include "args.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/dcm/extras.hpp>
#include <neuralfx/dcm/fine.hpp>
#include <neuralfx/dcm/search.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <print>
#include <random>
#include <set>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;
using namespace nfx;
namespace ex = nfx::dcm::extras;

namespace {

std::size_t sz(int v) { return static_cast<std::size_t>(v); }
constexpr int kSize = 128;
using Setting = std::array<float, 3>;

struct Ctx {
  fs::path data, models, results;
  int threads = 1;
  bool quick = false;
  std::string effects;
  std::string method;
};

std::mutex g_log_mutex;
fs::path g_log_path;
const auto g_t0 = std::chrono::steady_clock::now();

void log(const std::string& s) {
  std::lock_guard lock(g_log_mutex);
  const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_t0).count();
  const std::string line = std::format("[{:7.0f} s] {}", t, s);
  std::println("{}", line);
  std::fflush(stdout);
  if (!g_log_path.empty()) {
    std::ofstream o(g_log_path, std::ios::app);
    o << line << "\n";
  }
}

double thread_cpu_s() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

std::string ename(sim::Effect e) { return std::string(sim::effect_name(e)); }
bool one_shot(sim::Effect e) { return !sim::effect_loops(e); }
bool wanted(const Ctx& c, sim::Effect e) {
  if (c.effects.empty()) return true;
  std::stringstream ss(c.effects);
  for (std::string w; std::getline(ss, w, ',');) {
    if (w == ename(e)) return true;
  }
  return false;
}

// Jobs 0..n-1 on `threads` threads (one: on the calling thread, so its thread CPU time counts them).
void parallel(int n, int threads, const std::function<void(int)>& job) {
  if (threads <= 1) {
    for (int i = 0; i < n; ++i) job(i);
    return;
  }
  std::atomic<int> next{0};
  std::vector<std::jthread> pool;
  for (int t = 0; t < std::max(1, std::min(threads, n)); ++t) {
    pool.emplace_back([&] {
      for (int i; (i = next++) < n;) job(i);
    });
  }
}

// --- settings and seeds (docs/DCM.md §10.1) -----------------------------------------------------------------------

bool off_grid(const Setting& s) {
  const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
  return off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f});
}
std::vector<Setting> test_settings() {  // study B's held-out settings
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < 10) {
    const Setting s{u(rng), u(rng), u(rng)};
    if (off_grid(s)) v.push_back(s);
  }
  return v;
}
std::vector<Setting> validation_settings() {  // study G's validation settings (docs/DCM.md §4)
  const auto test = test_settings();
  std::mt19937_64 rng(2027);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < 10) {
    const Setting s{u(rng), u(rng), u(rng)};
    const bool far = std::ranges::all_of(test, [&](const Setting& t) { return std::hypot(s[0] - t[0], s[1] - t[1], s[2] - t[2]) >= 0.05f; });
    if (off_grid(s) && far) v.push_back(s);
  }
  return v;
}

struct Seeds {
  std::uint64_t real, model, track;
};
// Seed bases (§10.1): validation 2'100'000 / 2'110'000 / 2'120'000, test 2'200'000 / 2'220'000 / 2'230'000, plus the
// setting's index (one-shot shard slots: + 100 x setting + slot).
// --quick (a check of the pipeline) adds 50'000 to every base, so it never sees a validation or test run.
Seeds seeds_for(const std::string& split, bool quick = false) {
  const std::uint64_t q = quick ? 50000 : 0;
  if (split == "val") return {2100000 + q, 2110000 + q, 2120000 + q};
  if (split == "test") return {2200000 + q, 2220000 + q, 2230000 + q};
  throw std::invalid_argument("--split val or test");
}
std::vector<Setting> settings_for(const Ctx& c, const std::string& split) {
  auto v = split == "val" ? validation_settings() : test_settings();
  if (c.quick) v.resize(2);
  return v;
}
constexpr std::uint64_t kTrainModelSeed = 2050000;  // model windows of training runs: + 16 x run + slot

sim::Params run_params(sim::Effect e, std::uint64_t salt, std::uint64_t index) {
  rollout::SimRecipe r;
  r.effect = e;
  r.salt = salt;
  return rollout::recipe_run(r, index);
}
sim::Params setting_params(sim::Effect e, const Setting& s, std::uint64_t seed) {
  sim::Params p;
  p.effect = e;
  p.intensity = s[0];
  p.wind = s[1];
  p.turbulence = s[2];
  p.seed = seed;
  p.size = kSize;
  return p;
}

rollout::Model load_v1(const Ctx& c, sim::Effect e) {
  auto m = rollout::load_model(c.models / (ename(e) + ".nvfx"));
  if (!m) throw std::runtime_error(m.error());
  return std::move(*m);
}

// --- frames, statistics, scores -------------------------------------------------------------------------------------

struct Protocol {
  int warm = 150, ref_frames = 300, frames = 180, track_warm = 100, track = 60;
};
Protocol protocol(const Ctx& c, sim::Effect e) {
  Protocol p = one_shot(e) ? Protocol{1, 89, 89, 1, 89} : Protocol{};
  if (c.quick) {
    p.ref_frames = std::min(p.ref_frames, 60);
    p.frames = std::min(p.frames, 60);
    p.track = std::min(p.track, 40);
  }
  return p;
}

void to_u8(std::span<const float> rgba, std::span<std::uint8_t> out) {
  for (std::size_t i = 0; i < rgba.size(); ++i) out[i] = static_cast<std::uint8_t>(rgba[i] * 255.f + 0.5f);
}

double apsnr(std::span<const std::uint8_t> r, std::span<const std::uint8_t> t) {
  double sum = 0, n = 0;
  for (std::size_t i = 0; i < r.size(); i += 4) {
    if (std::max({r[i], r[i + 1], r[i + 2], r[i + 3]}) <= 4 && std::max({t[i], t[i + 1], t[i + 2], t[i + 3]}) <= 4) continue;
    n += 4;
    for (std::size_t c = 0; c < 4; ++c) {
      const double d = (static_cast<double>(r[i + c]) - static_cast<double>(t[i + c])) / 255.0;
      sum += d * d;
    }
  }
  return n > 0 ? metrics::psnr_from_mse(sum / n) : metrics::kPsnrCap;
}

// rollout_train.cpp's calibration score (lower is better), as fine_study.cpp and experiment_g.cpp compute it.
double detail_score(const metrics::ClipStats& ref, const metrics::ClipStats& test) {
  const metrics::StatDistance d = metrics::distance(ref, test);
  const auto mean = [](const std::vector<double>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); };
  const auto log_ratio = [](double a, double b) { return std::abs(std::log(std::max(1e-6, a) / std::max(1e-6, b))); };
  const double em = mean(ref.emission);
  return d.spectrum_l1 + std::abs(std::log(std::max(1e-3, d.motion_ratio))) + (em > 1e-3 ? log_ratio(mean(test.emission), em) : 0.0) +
         log_ratio(mean(test.coverage), mean(ref.coverage));
}
std::string stats_cells(const metrics::ClipStats& ref, const Clip& clip) {
  const metrics::ClipStats st = metrics::stats(clip);
  const metrics::StatDistance d = metrics::distance(ref, st);
  return std::format("{:.5f},{:.5f},{:.5f},{:.6f},{:.6f},{:.4f}", detail_score(ref, st), d.spectrum_l1, d.motion_ratio, d.coverage_l1, d.emission_l1,
                     d.mean_frame_psnr);
}
constexpr const char* kStatsHeader = "score,spectrum_l1,motion_ratio,coverage_l1,emission_l1,mean_frame_psnr";

struct Reference {
  metrics::ClipStats all;
  Clip clip;
};
Reference real_run(const Ctx& c, sim::Effect e, const Setting& s, std::uint64_t seed) {
  const Protocol pr = protocol(c, e);
  sim::Fluid f(setting_params(e, s, seed));
  for (int i = 0; i < pr.warm; ++i) f.step_frame();
  Reference r;
  r.clip.allocate(kSize, pr.ref_frames);
  r.clip.fps = 30.f;
  for (int i = 0; i < pr.ref_frames; ++i) {
    f.step_frame();
    f.render(r.clip.frame(i));
  }
  r.all = metrics::stats(r.clip);
  return r;
}

int nearest_start(const rollout::Model& M, std::span<const float> ctl) {
  int best = 0;
  float bd = 1e30f;
  for (std::size_t k = 0; k < M.starts.size(); ++k) {
    float d = 0.f;
    for (std::size_t q = 0; q < 3 && q < M.starts[k].controls.size(); ++q) d += (M.starts[k].controls[q] - ctl[q]) * (M.starts[k].controls[q] - ctl[q]);
    if (d < bd) {
      bd = d;
      best = static_cast<int>(k);
    }
  }
  return best;
}

// A real run tracked from its true state (study D's d-eval): the simulation warmed up, its state (with the fine fields
// at 128 px) the only start point of a copy of the model, which replays the run's seed.
struct Track {
  sim::Fluid truth;
  rollout::Model mt;
  std::vector<float> ctl;
  std::uint64_t seed = 0;
  std::vector<std::uint8_t> ref;
  Track(const rollout::Model& M, const sim::Params& p, int warm) : truth(p), mt(M), ctl{p.intensity, p.wind, p.turbulence}, seed(p.seed) {
    for (int i = 0; i < warm; ++i) truth.step_frame();
    const sim::State st = truth.state();
    mt.h.start_fine = kSize;
    rollout::StartPoint sp;
    sp.controls = ctl;
    sp.seed = p.seed;
    sp.time = st.time;
    sp.coarse.resize(sz(mt.h.res) * sz(mt.h.res) * rollout::kPhys);
    rollout::coarse_from_sim(st, mt.h.res, p.fps, sp.coarse);
    sp.fine_t = st.temp;
    sp.fine_d = st.soot;
    mt.starts = {sp};
    ref.resize(sz(kSize) * sz(kSize) * 4);
  }
  rollout::State start() const { return rollout::start(mt, 0, kSize, ctl, seed); }
  void next() {  // the true frame after the next step
    truth.step_frame();
    truth.render(ref);
  }
};

// --- CSVs that resume -----------------------------------------------------------------------------------------------

// Rows of finished units: the first `key` cells of a row name its unit; units already in the file are skipped.
class UnitCsv {
 public:
  UnitCsv(fs::path path, std::string header, int key) : path_(std::move(path)), header_(std::move(header)), key_(key) {
    fs::create_directories(path_.parent_path());
    if (std::ifstream in(path_); in) {
      std::string line;
      std::getline(in, line);
      if (line != header_) throw std::runtime_error(path_.string() + ": another header (move the old file away)");
      while (std::getline(in, line)) {
        if (!line.empty()) done_.insert(unit_of(line));
      }
    } else {
      std::ofstream(path_) << header_ << "\n";
    }
  }
  bool done(const std::string& unit) const {
    std::lock_guard lock(m_);
    return done_.contains(unit);
  }
  void add(const std::vector<std::string>& rows) {
    std::lock_guard lock(m_);
    std::ofstream o(path_, std::ios::app);
    for (const auto& r : rows) {
      o << r << "\n";
      done_.insert(unit_of(r));
    }
  }

 private:
  std::string unit_of(const std::string& row) const {
    std::size_t at = 0;
    for (int k = 0; k < key_ && at != std::string::npos; ++k) at = row.find(',', at + 1);
    return row.substr(0, at);
  }
  fs::path path_;
  std::string header_;
  int key_;
  std::set<std::string> done_;
  mutable std::mutex m_;
};

std::vector<std::vector<std::string>> read_csv(const fs::path& p) {
  std::vector<std::vector<std::string>> rows;
  std::ifstream in(p);
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<std::string> cells;
    std::stringstream ss(line);
    for (std::string cell; std::getline(ss, cell, ',');) cells.push_back(cell);
    rows.push_back(cells);
  }
  return rows;
}

std::string read_text(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + p.string());
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}
void write_text(const fs::path& p, const std::string& s) {
  fs::create_directories(p.parent_path());
  std::ofstream(p, std::ios::binary) << s;
}

// =====================================================================================================================
// G4a: one renderer
// =====================================================================================================================

const std::vector<ex::RenderMixConfig>& g4a_configs() {
  static const std::vector<ex::RenderMixConfig> v = {
      {{ex::kLearned, ex::kSim}}, {{ex::kLearned, ex::kSim, ex::kShader}}, {{ex::kLearned, ex::kSim, ex::kShader, ex::kOther1, ex::kOther2}}};
  return v;
}
std::string cfg_tag(const ex::RenderMixConfig& c) {
  std::string n = c.name();
  std::ranges::replace(n, '+', '_');
  return "mix_" + n;
}

struct Models {
  std::array<rollout::Model, 3> m;
  explicit Models(const Ctx& c) {
    for (std::size_t k = 0; k < 3; ++k) m[k] = load_v1(c, sim::kEffects[k]);
  }
  const rollout::Model& of(sim::Effect e) const { return m[static_cast<std::size_t>(e)]; }
  std::pair<const rollout::Model*, const rollout::Model*> others(sim::Effect e) const {
    std::vector<const rollout::Model*> o;
    for (const auto f : sim::kEffects) {
      if (f != e) o.push_back(&of(f));
    }
    return {o[0], o[1]};
  }
};

// The per-pixel contexts of the renderer mixer at a state.
struct PixelCtx {
  int age = 0;
  std::vector<std::uint8_t> heat, soot;  // per pixel, rows top to bottom
};
PixelCtx pixel_ctx(const rollout::Model& m, const rollout::State& s) {
  PixelCtx p;
  const int S = s.size;
  p.age = ex::age_bin(s.since_start * m.fps);
  p.heat.resize(sz(S) * sz(S));
  p.soot.resize(sz(S) * sz(S));
  const float it = 1.f / m.render_scale[0], id = 1.f / m.render_scale[1];
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      const std::size_t fi = sz(y) * sz(S) + sz(x), pi = sz(S - 1 - y) * sz(S) + sz(x);
      p.heat[pi] = static_cast<std::uint8_t>(ex::level_bin(s.fine_t[fi] * it));
      p.soot[pi] = static_cast<std::uint8_t>(ex::level_bin(s.fine_d[fi] * id));
    }
  }
  return p;
}

// G4a pilot: each renderer alone on validation tracking runs.
void g4a_pilot(const Ctx& c) {
  const Models M(c);
  const auto settings = settings_for(c, "val");
  const Seeds sd = seeds_for("val", c.quick);
  UnitCsv csv(c.results / "g_extras_g4a_pilot.csv", "effect,setting,method,track_1_30,track_31_end", 2);
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e)) continue;
    const Protocol pr = protocol(c, e);
    const auto [o1, o2] = M.others(e);
    parallel(static_cast<int>(settings.size()), c.threads, [&](int i) {
      const std::string unit = std::format("{},{}", ename(e), i);
      if (csv.done(unit)) return;
      Track tr(M.of(e), setting_params(e, settings[sz(i)], sd.track + static_cast<std::uint64_t>(i)), pr.track_warm);
      rollout::State s = tr.start();
      ex::SimRenderer sr(e, kSize);
      ex::RenderPlanes planes;
      std::array<double, ex::kRenderExperts> a{}, b{};
      std::vector<std::uint8_t> u8(sz(kSize) * sz(kSize) * 4);
      for (int f = 1; f <= pr.track; ++f) {
        tr.next();
        rollout::step(tr.mt, s, tr.ctl, tr.seed);
        ex::render_experts(tr.mt, s, sr, ex::ShaderLook{}, o1, o2, planes);
        for (int k = 0; k < ex::kRenderExperts; ++k) {
          to_u8(planes.p[sz(k)], u8);
          (f <= 30 ? a : b)[sz(k)] += apsnr(tr.ref, u8) / (f <= 30 ? 30.0 : pr.track - 30.0);
        }
      }
      std::vector<std::string> rows;
      for (int k = 0; k < ex::kRenderExperts; ++k) rows.push_back(std::format("{},{},{:.4f},{:.4f}", unit, ex::render_expert_name(k), a[sz(k)], b[sz(k)]));
      csv.add(rows);
      log(std::format("g4a-pilot {} setting {}: learned {:.2f} sim {:.2f} shader {:.2f} (first second)", ename(e), i, a[0], a[1], a[2]));
    });
  }
}

// One pixel's training record: every expert's RGBA, the contexts, the true RGBA.
struct PixelRow {
  std::array<float, ex::kRenderExperts * 4> x{};
  std::array<float, 4> y{};
  std::uint8_t age = 0, heat = 0, soot = 0, effect = 0;
};
static_assert(std::is_trivially_copyable_v<PixelRow>);

// The renderer mixer's rates (docs/DCM.md §10.6, amendment 1): chosen on the training rows' own error before any
// validation run. With the mixer's default annealing (rates halved after 2,000 uses of a context) it hardly left the
// learned renderer: the experts are close to each other, and normalised LMS moves slowly along their difference.
constexpr double kG4aLr = 0.1, kG4aAnneal = 1e5;

fs::path g4a_rows_path(const Ctx& c, sim::Effect e) { return c.data / std::format("g4a_rows_{}.bin", ename(e)); }
fs::path g4a_mixer_path(const Ctx& c, const ex::RenderMixConfig& cfg) { return c.data / std::format("g4a_{}.mixer", cfg_tag(cfg)); }

void g4a_rows(const Ctx& c) {
  const Models M(c);
  const int runs = c.quick ? 2 : 24;
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e) || fs::exists(g4a_rows_path(c, e))) continue;
    const Protocol pr = protocol(c, e);
    const auto [o1, o2] = M.others(e);
    std::vector<std::vector<PixelRow>> per(sz(runs));
    const double t0 = thread_cpu_s();
    parallel(runs, c.threads, [&](int r) {
      Track tr(M.of(e), run_params(e, 1, static_cast<std::uint64_t>(r)), pr.track_warm);
      rollout::State s = tr.start();
      ex::SimRenderer sr(e, kSize);
      ex::RenderPlanes planes;
      std::mt19937_64 rng(1000 + static_cast<std::uint64_t>(r) + 7919 * static_cast<std::uint64_t>(e));
      for (int f = 1; f <= pr.track; ++f) {
        tr.next();
        rollout::step(tr.mt, s, tr.ctl, tr.seed);
        ex::render_experts(tr.mt, s, sr, ex::ShaderLook{}, o1, o2, planes);
        const PixelCtx pc = pixel_ctx(tr.mt, s);
        std::vector<int> on, off;
        for (int i = 0; i < kSize * kSize; ++i) {
          bool act = false;
          for (int q = 0; q < 4; ++q) {
            act = act || tr.ref[sz(i * 4 + q)] > 4 || planes.p[ex::kLearned][sz(i * 4 + q)] * 255.f > 4.f || planes.p[ex::kSim][sz(i * 4 + q)] * 255.f > 4.f;
          }
          (act ? on : off).push_back(i);
        }
        std::shuffle(on.begin(), on.end(), rng);
        std::shuffle(off.begin(), off.end(), rng);
        on.resize(std::min<std::size_t>(on.size(), 150));
        off.resize(std::min<std::size_t>(off.size(), 15));
        on.insert(on.end(), off.begin(), off.end());
        for (const int i : on) {
          PixelRow row;
          for (int k = 0; k < ex::kRenderExperts; ++k) {
            for (int q = 0; q < 4; ++q) row.x[sz(k * 4 + q)] = planes.p[sz(k)][sz(i * 4 + q)];
          }
          for (int q = 0; q < 4; ++q) row.y[sz(q)] = static_cast<float>(tr.ref[sz(i * 4 + q)]) / 255.f;
          row.age = static_cast<std::uint8_t>(pc.age);
          row.heat = pc.heat[sz(i)];
          row.soot = pc.soot[sz(i)];
          row.effect = static_cast<std::uint8_t>(e);
          per[sz(r)].push_back(row);
        }
      }
    });
    std::vector<PixelRow> all;
    for (auto& v : per) all.insert(all.end(), v.begin(), v.end());
    fs::create_directories(c.data);
    std::ofstream o(g4a_rows_path(c, e), std::ios::binary);
    o.write(reinterpret_cast<const char*>(all.data()), static_cast<std::streamsize>(all.size() * sizeof(PixelRow)));
    log(std::format("g4a-rows {}: {} pixel rows from {} runs ({:.0f} s thread CPU)", ename(e), all.size(), runs, thread_cpu_s() - t0));
  }
}

std::vector<PixelRow> g4a_load_rows(const Ctx& c) {
  std::vector<PixelRow> all;
  for (const auto e : sim::kEffects) {
    std::ifstream in(g4a_rows_path(c, e), std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("no rows for " + ename(e) + " (run g4a-rows)");
    const auto n = static_cast<std::size_t>(in.tellg()) / sizeof(PixelRow);
    in.seekg(0);
    const std::size_t at = all.size();
    all.resize(at + n);
    in.read(reinterpret_cast<char*>(all.data() + at), static_cast<std::streamsize>(n * sizeof(PixelRow)));
  }
  return all;
}

void g4a_train(const Ctx& c) {
  const std::vector<PixelRow> rows = g4a_load_rows(c);
  std::vector<std::string> out;
  for (const auto& cfg : g4a_configs()) {
    const double t0 = thread_cpu_s();
    ex::RenderMixer mix(cfg, kG4aLr, kG4aAnneal);
    std::vector<std::size_t> order(rows.size() * 4);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::mt19937_64 rng(1);
    std::array<float, ex::kRenderExperts> x{};
    for (int pass = 0; pass < 3; ++pass) {
      std::shuffle(order.begin(), order.end(), rng);
      double se = 0;
      for (const std::size_t k : order) {
        const PixelRow& r = rows[k / 4];
        const int q = static_cast<int>(k % 4);
        for (int j = 0; j < ex::kRenderExperts; ++j) x[sz(j)] = r.x[sz(j * 4 + q)];
        const double p = mix.predict(x, q, r.age, r.heat, r.soot);
        se += (p - r.y[sz(q)]) * (p - r.y[sz(q)]);
        mix.update(r.y[sz(q)]);
      }
      log(std::format("g4a-train {} pass {}: rmse {:.4f}", cfg.name(), pass, std::sqrt(se / static_cast<double>(order.size()))));
      mix.scale_lr(0.5);
    }
    std::vector<std::vector<double>> w;
    for (int q = 0; q < 4; ++q) w.push_back(mix.mean_weights(q));
    mix.freeze();
    write_text(g4a_mixer_path(c, cfg), mix.serialise());
    for (int q = 0; q < 4; ++q) {
      std::string row = std::format("{},{},{},{}", cfg.name(), mix.version().substr(0, 16), "RGBA"[q], rows.size());
      for (std::size_t k = 0; k < cfg.experts.size(); ++k) row += std::format(",{}={:.3f}", ex::render_expert_name(cfg.experts[k]), w[sz(q)][k]);
      out.push_back(row);
    }
    log(std::format("g4a-train {}: version {} ({:.0f} s thread CPU)", cfg.name(), mix.version(), thread_cpu_s() - t0));
  }
  std::ofstream o(c.results / "g_extras_g4a_mixers.csv");
  o << "config,version,channel,pixel_rows,weights (mean over contexts, through the final mixer)\n";
  for (const auto& r : out) o << r << "\n";
}

void g4a_eval(const Ctx& c, const std::string& split) {
  const Models M(c);
  std::vector<ex::RenderMixConfig> cfgs;
  if (split == "val") {
    cfgs = g4a_configs();
  } else {
    if (c.method.empty()) throw std::invalid_argument("g4a-eval --split test needs --method (the configuration chosen on validation)");
    for (const auto& cfg : g4a_configs()) {
      if (cfg_tag(cfg) == c.method) cfgs.push_back(cfg);
    }
    if (cfgs.empty()) throw std::invalid_argument("no configuration " + c.method);
  }
  std::vector<ex::RenderMixer> mixers;
  for (const auto& cfg : cfgs) {
    mixers.push_back(ex::RenderMixer::load(cfg, read_text(g4a_mixer_path(c, cfg))));
    log(std::format("g4a-eval {}: mixer {} version {}", split, cfg.name(), mixers.back().version()));
  }
  const auto settings = settings_for(c, split);
  const Seeds sd = seeds_for(split, c.quick);
  UnitCsv csv(c.results / std::format("g_extras_g4a_{}.csv", split), std::string("effect,setting,method,track_1_30,track_31_end,") + kStatsHeader, 2);
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e)) continue;
    const Protocol pr = protocol(c, e);
    const auto [o1, o2] = M.others(e);
    parallel(static_cast<int>(settings.size()), c.threads, [&](int i) {
      const std::string unit = std::format("{},{}", ename(e), i);
      if (csv.done(unit)) return;
      const std::size_t nm = mixers.size() + 1;  // v1, then each mixer
      std::vector<double> a(nm, 0.0), b(nm, 0.0);
      ex::SimRenderer sr(e, kSize);
      ex::RenderPlanes planes;
      std::vector<float> rgba(sz(kSize) * sz(kSize) * 4);
      std::vector<std::uint8_t> u8(rgba.size());
      {  // tracking from the true state
        Track tr(M.of(e), setting_params(e, settings[sz(i)], sd.track + static_cast<std::uint64_t>(i)), pr.track_warm);
        rollout::State s = tr.start();
        for (int f = 1; f <= pr.track; ++f) {
          tr.next();
          rollout::step(tr.mt, s, tr.ctl, tr.seed);
          ex::render_experts(tr.mt, s, sr, ex::ShaderLook{}, o1, o2, planes);
          const double w = f <= 30 ? 1.0 / 30.0 : 1.0 / (pr.track - 30.0);
          to_u8(planes.p[ex::kLearned], u8);
          (f <= 30 ? a : b)[0] += w * apsnr(tr.ref, u8);
          for (std::size_t k = 0; k < mixers.size(); ++k) {
            ex::render_mixed(mixers[k], tr.mt, s, planes, rgba);
            to_u8(rgba, u8);
            (f <= 30 ? a : b)[k + 1] += w * apsnr(tr.ref, u8);
          }
        }
      }
      // endless: one shard from the nearest start point against a real run
      const Reference ref = real_run(c, e, settings[sz(i)], sd.real + static_cast<std::uint64_t>(i));
      const rollout::Model& m = M.of(e);
      const std::vector<float> ctl(settings[sz(i)].begin(), settings[sz(i)].end());
      const std::uint64_t seed = sd.model + static_cast<std::uint64_t>(i);
      rollout::State s = rollout::start(m, nearest_start(m, ctl), kSize, ctl, seed);
      std::vector<Clip> clips(nm);
      for (auto& cl : clips) {
        cl.allocate(kSize, pr.frames);
        cl.fps = 30.f;
      }
      for (int f = 0; f < pr.frames; ++f) {
        rollout::step(m, s, ctl, seed);
        ex::render_experts(m, s, sr, ex::ShaderLook{}, o1, o2, planes);
        to_u8(planes.p[ex::kLearned], clips[0].frame(f));
        for (std::size_t k = 0; k < mixers.size(); ++k) {
          ex::render_mixed(mixers[k], m, s, planes, rgba);
          to_u8(rgba, clips[k + 1].frame(f));
        }
      }
      std::vector<std::string> rows;
      for (std::size_t k = 0; k < nm; ++k) {
        const std::string name = k == 0 ? "v1" : cfg_tag(cfgs[k - 1]);
        rows.push_back(std::format("{},{},{:.4f},{:.4f},{}", unit, name, a[k], b[k], stats_cells(ref.all, clips[k])));
      }
      csv.add(rows);
      log(std::format("g4a-eval {} {} setting {}: first second v1 {:.2f}, mixers {:.2f}{}", split, ename(e), i, a[0], a.size() > 1 ? a[1] : 0.0,
                      a.size() > 2 ? std::format(" {:.2f} {:.2f}", a[2], a.size() > 3 ? a[3] : 0.0) : ""));
    });
  }
}

// =====================================================================================================================
// G5a: shard critic
// =====================================================================================================================

constexpr int kLook = 30;  // frames of look-ahead the critic reads

fs::path g5a_windows_path(const Ctx& c, sim::Effect e) { return c.data / std::format("g5a_{}_train_windows.csv", ename(e)); }
fs::path g5a_critic_path(const Ctx& c, sim::Effect e) { return c.data / std::format("g5a_{}.critic", ename(e)); }

std::string feature_cells(const std::array<double, ex::kCriticFeatures>& f) {
  std::string s;
  for (const double v : f) s += std::format(",{:.6g}", v);
  return s;
}

// `frames` frames of a shard (start point, seed) through the reference, v1's renderer.
Clip shard_clip(const rollout::Model& m, int start, std::uint64_t seed, std::span<const float> ctl, int frames) {
  Clip cl;
  cl.allocate(kSize, frames);
  cl.fps = 30.f;
  rollout::State s = rollout::start(m, start, kSize, ctl, seed);
  std::vector<float> rgba(sz(kSize) * sz(kSize) * 4);
  for (int f = 0; f < frames; ++f) {
    rollout::step(m, s, ctl, seed);
    dcm::fine::render(m, s, rgba);
    to_u8(rgba, cl.frame(f));
  }
  return cl;
}

// The instance seed and shard of slot k at setting i: looping effects play shards 0..3 of one instance; a one-shot
// effect is one shard, so each slot is an instance of its own.
std::pair<std::uint64_t, std::int64_t> slot_of(sim::Effect e, std::uint64_t base, int i, int k) {
  if (one_shot(e)) return {base + 100 * static_cast<std::uint64_t>(i) + static_cast<std::uint64_t>(k), 0};
  return {base + static_cast<std::uint64_t>(i), k};
}

void g5a_train(const Ctx& c) {
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e)) continue;
    const rollout::Model M = load_v1(c, e);
    const Protocol pr = protocol(c, e);
    const int runs = c.quick ? 8 : (one_shot(e) ? 120 : 40);
    UnitCsv csv(g5a_windows_path(c, e), "run,kind,k,i,w,t" + [] {
      std::string h;
      for (int j = 0; j < ex::kCriticFeatures; ++j) h += std::format(",{}", ex::critic_feature_name(j));
      return h;
    }(), 1);
    const double t0 = thread_cpu_s();
    parallel(runs, c.threads, [&](int r) {
      if (csv.done(std::to_string(r))) return;
      const sim::Params p = run_params(e, 1, static_cast<std::uint64_t>(r));
      const std::vector<float> ctl{p.intensity, p.wind, p.turbulence};
      std::vector<std::string> rows;
      const std::string cc = std::format("{:.5f},{:.5f},{:.5f}", ctl[0], ctl[1], ctl[2]);
      {  // real windows: looping effects, 10 windows of a 10 s run after the warm-up; one-shot effects, the first second
        sim::Fluid f(p);
        for (int i = 0; i < pr.warm; ++i) f.step_frame();
        const int n = one_shot(e) ? 1 : (c.quick ? 2 : 10);
        Clip w;
        w.allocate(kSize, kLook);
        for (int k = 0; k < n; ++k) {
          for (int t = 0; t < kLook; ++t) {
            f.step_frame();
            f.render(w.frame(t));
          }
          rows.push_back(std::format("{},real,{},{}{}", r, k, cc, feature_cells(ex::critic_features(w))));
        }
      }
      for (int k = 0; k < 4; ++k) {  // the model's own shard starts at these controls
        const auto [seed, shard] = slot_of(e, kTrainModelSeed + 16 * static_cast<std::uint64_t>(r), 0, k);
        const ex::Candidate cand = ex::shard_candidate(M, ctl, seed, shard, 0);
        const Clip w = shard_clip(M, cand.start, cand.seed, ctl, kLook);
        rows.push_back(std::format("{},model,{},{}{}", r, k, cc, feature_cells(ex::critic_features(w))));
      }
      csv.add(rows);
    });
    log(std::format("g5a-train {}: windows of {} runs ({:.0f} s thread CPU)", ename(e), runs, thread_cpu_s() - t0));
    std::vector<ex::CriticSample> fit, train;
    for (const auto& row : read_csv(g5a_windows_path(c, e))) {
      ex::CriticSample s;
      const int run = std::stoi(row[0]);
      if (run >= runs) continue;
      s.real = row[1] == "real";
      for (int q = 0; q < 3; ++q) s.controls[sz(q)] = std::stof(row[sz(3 + q)]);
      for (int j = 0; j < ex::kCriticFeatures; ++j) s.f[sz(j)] = std::stod(row[sz(6 + j)]);
      (run < runs / 2 ? fit : train).push_back(s);
    }
    ex::ShardCritic critic;
    critic.fit(fit, train, 4, 1);
    write_text(g5a_critic_path(c, e), critic.serialise());
    std::vector<float> sc;
    std::vector<int> y;
    for (const auto& s : train) {
      sc.push_back(static_cast<float>(critic.p_real(s.f, s.controls)));
      y.push_back(s.real ? 1 : 0);
    }
    log(std::format("g5a-train {}: critic {} on {} + {} windows, AUC on its training windows {:.3f}", ename(e), critic.version(), fit.size(), train.size(),
                    dcm::roc_auc(sc, y)));
    std::ofstream o(c.results / "g_extras_g5a_critics.csv", std::ios::app);
    o << std::format("{},{},{},{},{:.4f}\n", ename(e), critic.version(), fit.size(), train.size(), dcm::roc_auc(sc, y));
  }
}

void g5a_eval(const Ctx& c, const std::string& split) {
  const auto settings = settings_for(c, split);
  const Seeds sd = seeds_for(split, c.quick);
  const int K = split == "val" ? 8 : 4, slots = c.quick ? 2 : 4;
  UnitCsv csv(c.results / std::format("g_extras_g5a_{}.csv", split), std::string("effect,setting,slot,cand,start,seed,p_real,") + kStatsHeader, 2);
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e)) continue;
    const rollout::Model M = load_v1(c, e);
    ex::ShardCritic critic = ex::ShardCritic::load(read_text(g5a_critic_path(c, e)));
    log(std::format("g5a-eval {} {}: critic {}", split, ename(e), critic.version()));
    const Protocol pr = protocol(c, e);
    parallel(static_cast<int>(settings.size()), c.threads, [&](int i) {
      const std::string unit = std::format("{},{}", ename(e), i);
      if (csv.done(unit)) return;
      ex::ShardCritic cr = critic;
      const std::vector<float> ctl(settings[sz(i)].begin(), settings[sz(i)].end());
      const Reference ref = real_run(c, e, settings[sz(i)], sd.real + static_cast<std::uint64_t>(i));
      std::vector<std::string> rows;
      // the real run's own windows (for the critic's AUC): looping effects every 30 frames, one-shot effects the first
      for (int w = 0; w + kLook <= ref.clip.frames && (w == 0 || !one_shot(e)); w += kLook) {
        rows.push_back(std::format("{},-1,{},-1,0,{:.6f},,,,,,", unit, w / kLook, cr.p_real(ex::critic_features(slice_clip(ref.clip, w, kLook)), ctl)));
      }
      for (int k = 0; k < slots; ++k) {
        const auto [seed, shard] = slot_of(e, sd.model, i, k);
        for (int j = 0; j < K; ++j) {
          const ex::Candidate cand = ex::shard_candidate(M, ctl, seed, shard, j);
          const Clip cl = shard_clip(M, cand.start, cand.seed, ctl, pr.frames);
          const double p = cr.p_real(ex::critic_features(slice_clip(cl, 0, kLook)), ctl);
          rows.push_back(std::format("{},{},{},{},{},{:.6f},{}", unit, k, j, cand.start, cand.seed, p, stats_cells(ref.all, cl)));
        }
      }
      csv.add(rows);
      log(std::format("g5a-eval {} {} setting {} done", split, ename(e), i));
    });
  }
}

// =====================================================================================================================
// G5b: the stepper's update and a cheap solver's update, mixed
// =====================================================================================================================

fs::path g5b_mixer_path(const Ctx& c, const std::string& name) { return c.data / std::format("g5b_{}.mixer", name); }

struct TargetRow {
  ex::UpdateRow r;
  float y = 0;
};

// One step from the truth: rows of training runs (smoke, salt 1), frames 100 to 238, 128 cells per step.
std::vector<TargetRow> g5b_truth_rows(const Ctx& c, const rollout::Model& M, int runs, std::vector<rollout::Run>& recorded) {
  const int R = M.h.res, C = M.h.channels();
  std::vector<std::vector<TargetRow>> per(sz(runs));
  recorded.resize(sz(runs));
  parallel(runs, c.threads, [&](int r) {
    const sim::Params p = run_params(sim::Effect::smoke, 1, static_cast<std::uint64_t>(r));
    recorded[sz(r)] = rollout::record_run(p, 240, R);
    const rollout::Run& run = recorded[sz(r)];
    const std::vector<float> ctl = run.controls();
    ex::CoarseSolver solver(sim::Effect::smoke, ctl, p.seed, R, M.fps);
    rollout::State s;
    s.res = R;
    s.size = kSize;
    s.coarse.assign(sz(R) * sz(R) * sz(C), 0.f);
    s.pressure.assign(sz(R) * sz(R), 0.f);
    s.flow.assign(sz(R) * sz(R) * 2, 0.f);
    std::vector<float> cond(sz(M.h.cond())), noise(sz(R) * sz(R) * rollout::kNoise), next(s.coarse.size()), solv(sz(R) * sz(R) * rollout::kPhys);
    std::mt19937_64 rng(500 + static_cast<std::uint64_t>(r));
    const std::size_t per_frame = sz(R) * sz(R) * rollout::kPhys;
    for (int t = 100; t < 239; ++t) {
      for (int i = 0; i < R * R; ++i) {
        for (int k = 0; k < rollout::kPhys; ++k) s.coarse[sz(i) * sz(C) + sz(k)] = run.coarse[sz(t) * per_frame + sz(i) * rollout::kPhys + sz(k)];
      }
      s.time = static_cast<float>(t + 1) / M.fps;
      rollout::condition(M, ctl, s.time, cond);
      rollout::coarse_noise(M, p.seed, s.time, noise);
      rollout::coarse_step(M, s.coarse, noise, cond, s.pressure, next, s.flow);
      solver.step(M, s, solv);
      std::vector<int> cells(sz(R) * sz(R));
      std::iota(cells.begin(), cells.end(), 0);
      std::shuffle(cells.begin(), cells.end(), rng);
      for (int q = 0; q < 128; ++q) {
        const int i = cells[sz(q)];
        const ex::CellCtx cx = ex::cell_context(M, s, i);
        for (int k = 0; k < rollout::kPhys; ++k) {
          const std::size_t j = sz(i) * sz(C) + sz(k);
          const float x = s.coarse[j];
          TargetRow tr;
          tr.r = {next[j] - x, solv[sz(i) * rollout::kPhys + sz(k)] - x, x, static_cast<std::uint8_t>(k), static_cast<std::uint8_t>(cx.heat),
                  static_cast<std::uint8_t>(cx.band), 0};
          tr.y = run.coarse[sz(t + 1) * per_frame + sz(i) * rollout::kPhys + sz(k)] - x;
          per[sz(r)].push_back(tr);
        }
      }
      for (int i = 0; i < R * R; ++i) {  // memory channels carried from the stepper's own step
        for (int k = rollout::kPhys; k < C; ++k) s.coarse[sz(i) * sz(C) + sz(k)] = next[sz(i) * sz(C) + sz(k)];
      }
    }
  });
  std::vector<TargetRow> all;
  for (auto& v : per) all.insert(all.end(), v.begin(), v.end());
  return all;
}

void train_pass(ex::UpdateMixer& mix, std::vector<TargetRow>& rows, std::mt19937_64& rng, double* rmse = nullptr) {
  std::shuffle(rows.begin(), rows.end(), rng);
  double se = 0;
  for (const TargetRow& t : rows) {
    const double p = mix.predict(t.r.dn, t.r.ds, t.r.channel, t.r.heat, t.r.band, t.r.age);
    se += (p - t.y) * (p - t.y);
    mix.update(t.y);
  }
  if (rmse) *rmse = std::sqrt(se / static_cast<double>(std::max<std::size_t>(1, rows.size())));
}

void g5b_train(const Ctx& c) {
  const rollout::Model M = load_v1(c, sim::Effect::smoke);
  const int runs = c.quick ? 2 : 24, R = M.h.res, C = M.h.channels();
  const double t0 = thread_cpu_s();
  std::vector<rollout::Run> recorded;
  std::vector<TargetRow> truth = g5b_truth_rows(c, M, runs, recorded);
  log(std::format("g5b-train: {} one-step rows from {} runs ({:.0f} s thread CPU)", truth.size(), runs, thread_cpu_s() - t0));
  // m1: the first-layer rate chosen among four by the training rows' own error after training (rates anneal over
  // 100,000 uses of a context; docs/DCM.md §10.6)
  std::mt19937_64 rng(1);
  ex::UpdateMixer mix;
  double best_rmse = 1e30, v1_rmse = 0;
  for (const TargetRow& t : truth) v1_rmse += (t.r.dn - t.y) * (t.r.dn - t.y);
  v1_rmse = std::sqrt(v1_rmse / static_cast<double>(truth.size()));
  for (const double lr : {0.02, 0.05, 0.1, 0.2}) {
    ex::UpdateMixer cand(lr, 1e5);
    std::mt19937_64 rr(1);
    std::vector<TargetRow> rows = truth;
    for (int pass = 0; pass < 3; ++pass) {
      train_pass(cand, rows, rr);
      cand.scale_lr(0.5);
    }
    ex::UpdateMixer probe = cand;
    probe.freeze();
    double se = 0;
    for (const TargetRow& t : truth) {
      const double p = probe.predict(t.r.dn, t.r.ds, t.r.channel, t.r.heat, t.r.band, t.r.age);
      se += (p - t.y) * (p - t.y);
    }
    const double rmse = std::sqrt(se / static_cast<double>(truth.size()));
    log(std::format("g5b-train m1 lr {}: training rmse {:.6f} (the stepper alone {:.6f})", lr, rmse, v1_rmse));
    if (rmse < best_rmse) {
      best_rmse = rmse;
      mix = cand;
    }
  }
  std::vector<std::string> wrows;
  const auto weights = [&](const std::string& name, const ex::UpdateMixer& m) {
    for (int k = 0; k < rollout::kPhys; ++k) {
      const auto w = m.mean_weights(k);
      wrows.push_back(std::format("{},{},{:.4f},{:.4f},{:.6f}", name, "uvhs"[k], w[0], w[1], w[2]));
    }
  };
  weights("m1", mix);
  ex::UpdateMixer m1 = mix;
  m1.freeze();
  write_text(g5b_mixer_path(c, "m1"), m1.serialise());
  log(std::format("g5b-train m1: version {}", m1.version()));
  // the own-rollout pass: windows from the true state at frames 100 and 160, 60 frames of the mixed coarse dynamics
  // (the frozen m1), rows against the true state at the same frame, then one pass at the rates training ended with,
  // mixed with as many one-step rows
  std::vector<TargetRow> own;
  std::mutex m_own;
  parallel(runs, c.threads, [&](int r) {
    const rollout::Run& run = recorded[sz(r)];
    const std::vector<float> ctl = run.controls();
    const std::size_t per_frame = sz(R) * sz(R) * rollout::kPhys;
    std::mt19937_64 rr(900 + static_cast<std::uint64_t>(r));
    std::vector<TargetRow> mine;
    for (const int t0w : {100, 160}) {
      ex::UpdateMixer frozen = m1;
      ex::CoarseSolver solver(sim::Effect::smoke, ctl, run.p.seed, R, M.fps);
      rollout::State s;
      s.res = R;
      s.size = kSize;
      s.coarse.assign(sz(R) * sz(R) * sz(C), 0.f);
      for (int i = 0; i < R * R; ++i) {
        for (int k = 0; k < rollout::kPhys; ++k) s.coarse[sz(i) * sz(C) + sz(k)] = run.coarse[sz(t0w) * per_frame + sz(i) * rollout::kPhys + sz(k)];
      }
      s.pressure.assign(sz(R) * sz(R), 0.f);
      s.flow.assign(sz(R) * sz(R) * 2, 0.f);
      s.time = static_cast<float>(t0w + 1) / M.fps;
      for (int f = 0; f < 60 && t0w + f + 1 < run.frames; ++f) {
        std::vector<float> before(s.coarse);
        std::vector<ex::UpdateRow> rows;
        ex::mixed_step(M, s, ctl, run.p.seed, solver, frozen, &rows, false);
        std::vector<int> cells(sz(R) * sz(R));
        std::iota(cells.begin(), cells.end(), 0);
        std::shuffle(cells.begin(), cells.end(), rr);
        for (int q = 0; q < 128; ++q) {
          const int i = cells[sz(q)];
          for (int k = 0; k < rollout::kPhys; ++k) {
            TargetRow tr;
            tr.r = rows[sz(i) * rollout::kPhys + sz(k)];
            tr.y = run.coarse[sz(t0w + f + 1) * per_frame + sz(i) * rollout::kPhys + sz(k)] - before[sz(i) * sz(C) + sz(k)];
            mine.push_back(tr);
          }
        }
      }
    }
    std::lock_guard lock(m_own);
    own.insert(own.end(), mine.begin(), mine.end());
  });
  std::shuffle(truth.begin(), truth.end(), rng);
  own.insert(own.end(), truth.begin(), truth.begin() + static_cast<std::ptrdiff_t>(std::min(truth.size(), own.size())));
  double rmse = 0;
  train_pass(mix, own, rng, &rmse);
  log(std::format("g5b-train m2 own-rollout pass on {} rows: rmse {:.5f}", own.size(), rmse));
  weights("m2", mix);
  mix.freeze();
  write_text(g5b_mixer_path(c, "m2"), mix.serialise());
  log(std::format("g5b-train m2: version {} ({:.0f} s thread CPU in all)", mix.version(), thread_cpu_s() - t0));
  std::ofstream o(c.results / "g_extras_g5b_mixers.csv");
  o << "mixer,channel,w_stepper,w_solver,bias\n";
  for (const auto& r : wrows) o << r << "\n";
  o << std::format("# m1 {}\n# m2 {}\n", m1.version(), mix.version());
}

void g5b_eval(const Ctx& c, const std::string& split) {
  const sim::Effect e = sim::Effect::smoke;
  const rollout::Model M = load_v1(c, e);
  std::vector<std::string> names;
  if (split == "val") {
    names = {"blend_0.1", "blend_0.25", "blend_0.5", "m1", "m2"};
  } else {
    if (c.method.empty()) throw std::invalid_argument("g5b-eval --split test needs --method");
    names = {c.method};
  }
  std::vector<ex::UpdateMixer> mixers;
  for (const auto& n : names) {
    if (n.starts_with("blend_")) {
      mixers.push_back(ex::UpdateMixer::blend(std::stod(n.substr(6))));
      mixers.back().freeze();
    } else {
      mixers.push_back(ex::UpdateMixer::load(read_text(g5b_mixer_path(c, n))));
    }
    log(std::format("g5b-eval {}: {} version {}", split, n, mixers.back().version()));
  }
  const auto settings = settings_for(c, split);
  const Seeds sd = seeds_for(split, c.quick);
  const Protocol pr = protocol(c, e);
  UnitCsv csv(c.results / std::format("g_extras_g5b_{}.csv", split), std::string("effect,setting,method,f1,f8,f30,f60,track_mean,") + kStatsHeader, 2);
  parallel(static_cast<int>(settings.size()), c.threads, [&](int i) {
    const std::string unit = std::format("{},{}", ename(e), i);
    if (csv.done(unit)) return;
    const std::size_t nm = mixers.size() + 1;
    std::vector<std::vector<double>> curve(nm);
    {
      Track tr(M, setting_params(e, settings[sz(i)], sd.track + static_cast<std::uint64_t>(i)), pr.track_warm);
      std::vector<rollout::State> st(nm, tr.start());
      std::vector<std::unique_ptr<ex::CoarseSolver>> solvers;
      for (std::size_t k = 0; k < nm; ++k) solvers.push_back(std::make_unique<ex::CoarseSolver>(e, tr.ctl, tr.seed, M.h.res, M.fps));
      std::vector<ex::UpdateMixer> mx = mixers;
      std::vector<float> rgba(sz(kSize) * sz(kSize) * 4);
      std::vector<std::uint8_t> u8(rgba.size());
      for (int f = 1; f <= pr.track; ++f) {
        tr.next();
        for (std::size_t k = 0; k < nm; ++k) {
          if (k == 0) rollout::step(tr.mt, st[k], tr.ctl, tr.seed);
          else ex::mixed_step(tr.mt, st[k], tr.ctl, tr.seed, *solvers[k], mx[k - 1]);
          dcm::fine::render(tr.mt, st[k], rgba);
          to_u8(rgba, u8);
          curve[k].push_back(apsnr(tr.ref, u8));
        }
      }
    }
    const Reference ref = real_run(c, e, settings[sz(i)], sd.real + static_cast<std::uint64_t>(i));
    const std::vector<float> ctl(settings[sz(i)].begin(), settings[sz(i)].end());
    const std::uint64_t seed = sd.model + static_cast<std::uint64_t>(i);
    std::vector<std::string> rows;
    for (std::size_t k = 0; k < nm; ++k) {
      Clip cl;
      cl.allocate(kSize, pr.frames);
      cl.fps = 30.f;
      rollout::State s = rollout::start(M, nearest_start(M, ctl), kSize, ctl, seed);
      ex::CoarseSolver solver(e, ctl, seed, M.h.res, M.fps);
      ex::UpdateMixer mx = k == 0 ? ex::UpdateMixer() : mixers[k - 1];
      std::vector<float> rgba(sz(kSize) * sz(kSize) * 4);
      for (int f = 0; f < pr.frames; ++f) {
        if (k == 0) rollout::step(M, s, ctl, seed);
        else ex::mixed_step(M, s, ctl, seed, solver, mx);
        dcm::fine::render(M, s, rgba);
        to_u8(rgba, cl.frame(f));
      }
      const auto at = [&](int h) { return h <= static_cast<int>(curve[k].size()) ? curve[k][sz(h - 1)] : std::nan(""); };
      const double mean = std::accumulate(curve[k].begin(), curve[k].end(), 0.0) / static_cast<double>(curve[k].size());
      rows.push_back(std::format("{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{}", unit, k == 0 ? "v1" : names[k - 1], at(1), at(8), at(30), at(60), mean,
                                 stats_cells(ref.all, cl)));
    }
    csv.add(rows);
    log(std::format("g5b-eval {} setting {}: f30 v1 {:.2f} vs {} {:.2f}", split, i, curve[0][std::min<std::size_t>(29, curve[0].size() - 1)], names[0],
                    curve[1][std::min<std::size_t>(29, curve[1].size() - 1)]));
  });
}

// =====================================================================================================================
// summary: paired bootstrap tables, the choices and the decisions by the rules of docs/DCM.md §10
// =====================================================================================================================

struct Csv {
  std::vector<std::string> header;
  std::vector<std::vector<std::string>> rows;
  int col(const std::string& name) const {
    for (std::size_t k = 0; k < header.size(); ++k) {
      if (header[k] == name) return static_cast<int>(k);
    }
    throw std::runtime_error("no column " + name);
  }
};
Csv load_csv(const fs::path& p) {
  Csv c;
  std::ifstream in(p);
  std::string line;
  if (!std::getline(in, line)) return c;
  std::stringstream hs(line);
  for (std::string cell; std::getline(hs, cell, ',');) c.header.push_back(cell);
  c.rows = read_csv(p);
  return c;
}

// A measure of a method: its value per setting (index = setting).
struct Measure {
  std::string name;
  bool higher_better = true;
};
const std::vector<Measure>& endless_measures() {
  static const std::vector<Measure> v = {{"score", false}, {"spectrum_l1", false}, {"abs_ln_motion", false}, {"coverage_l1", false}, {"mean_frame_psnr", true}};
  return v;
}
double cell_value(const Csv& c, const std::vector<std::string>& r, const std::string& name) {
  if (name == "abs_ln_motion") return std::abs(std::log(std::max(1e-6, std::stod(r[sz(c.col("motion_ratio"))]))));
  return std::stod(r[sz(c.col(name))]);
}
using Series = std::map<std::string, std::vector<double>>;  // measure -> per setting
// effect -> method -> measure -> values over settings (rows with a method column)
std::map<std::string, std::map<std::string, Series>> by_method(const Csv& c, const std::vector<std::string>& measures) {
  std::map<std::string, std::map<std::string, Series>> out;
  std::map<std::string, std::map<std::string, std::map<int, std::vector<double>>>> tmp;
  for (const auto& r : c.rows) {
    const std::string e = r[sz(c.col("effect"))], m = r[sz(c.col("method"))];
    const int s = std::stoi(r[sz(c.col("setting"))]);
    for (const auto& name : measures) out[e][m][name];
    for (std::size_t k = 0; k < measures.size(); ++k) tmp[e][m][s].push_back(cell_value(c, r, measures[k]));
  }
  for (auto& [e, ms] : tmp) {
    for (auto& [m, ss] : ms) {
      for (auto& [s, vals] : ss) {
        (void)s;
        for (std::size_t k = 0; k < measures.size(); ++k) out[e][m][measures[k]].push_back(vals[k]);
      }
    }
  }
  return out;
}

double mean_of(const std::vector<double>& v) { return v.empty() ? std::nan("") : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); }
std::string iv_text(const metrics::Interval& iv, int prec = 3) {
  return std::format("{:+.{}f} [{:+.{}f}, {:+.{}f}]{}", iv.mean, prec, iv.lo, prec, iv.hi, prec, iv.covers_zero() ? " (tie)" : "");
}
metrics::Interval diff(const std::vector<double>& a, const std::vector<double>& b) { return metrics::paired_bootstrap(a, b, 10000, 1); }
bool worse(const metrics::Interval& iv, bool higher_better) { return higher_better ? iv.hi < 0 : iv.lo > 0; }

struct Md {
  std::ostringstream o;
  void line(const std::string& s = "") { o << s << "\n"; }
};

// G4a: the pilot, validation (the choice), test (the decision).
void summary_g4a(const Ctx& c, Md& md) {
  const fs::path pilot = c.results / "g_extras_g4a_pilot.csv";
  if (fs::exists(pilot)) {
    const Csv p = load_csv(pilot);
    const auto bm = by_method(p, {"track_1_30", "track_31_end"});
    md.line("#### G4a pilot: each renderer alone (validation tracking, active PSNR in dB, means over 10 settings)");
    md.line();
    md.line("| effect | renderer | frames 1-30 | frames 31-60 (89) | minus learned, frames 1-30 |");
    md.line("|---|---|---:|---:|---|");
    for (const auto& [e, ms] : bm) {
      for (const auto& [m, s] : ms) {
        const std::string d = m == "learned" ? "" : iv_text(diff(s.at("track_1_30"), ms.at("learned").at("track_1_30")), 2);
        md.line(std::format("| {} | {} | {:.2f} | {:.2f} | {} |", e, m, mean_of(s.at("track_1_30")), mean_of(s.at("track_31_end")), d));
      }
    }
    md.line();
  }
  const std::vector<std::string> meas = {"track_1_30", "track_31_end", "score", "spectrum_l1", "abs_ln_motion", "coverage_l1", "mean_frame_psnr"};
  const std::vector<std::string> rule = {"track_1_30", "track_31_end", "score", "mean_frame_psnr"};
  const auto hb = [](const std::string& m) { return m.starts_with("track") || m == "mean_frame_psnr"; };
  for (const std::string split : {"val", "test"}) {
    const fs::path f = c.results / std::format("g_extras_g4a_{}.csv", split);
    if (!fs::exists(f)) continue;
    const auto bm = by_method(load_csv(f), meas);
    md.line(std::format("#### G4a {}: v1's learned renderer and the mixers (means over settings; differences paired over settings)", split));
    md.line();
    md.line("| effect | method | frames 1-30 | 31-60 (89) | score | spectrum | \\|ln motion\\| | coverage L1 | mean-frame PSNR |");
    md.line("|---|---|---:|---:|---:|---:|---:|---:|---:|");
    for (const auto& [e, ms] : bm) {
      for (const auto& [m, s] : ms) {
        std::string row = std::format("| {} | {} |", e, m);
        for (const auto& k : meas) row += std::format(" {:.4f} |", mean_of(s.at(k)));
        md.line(row);
      }
    }
    md.line();
    md.line("| effect | method - v1 | frames 1-30 | 31-60 (89) | score | spectrum | \\|ln motion\\| | coverage L1 | mean-frame PSNR | worse in a rule measure |");
    md.line("|---|---|---|---|---|---|---|---|---|---|");
    std::map<std::string, std::pair<bool, double>> ok;  // method -> (not worse anywhere, sum of first-second gains)
    for (const auto& [e, ms] : bm) {
      for (const auto& [m, s] : ms) {
        if (m == "v1") continue;
        std::string row = std::format("| {} | {} |", e, m);
        bool bad = false;
        for (const auto& k : meas) {
          const auto iv = diff(s.at(k), ms.at("v1").at(k));
          row += " " + iv_text(iv, k.starts_with("track") || k == "mean_frame_psnr" ? 2 : 4) + " |";
          if (std::ranges::find(rule, k) != rule.end() && worse(iv, hb(k))) bad = true;
        }
        md.line(row + (bad ? " **yes** |" : " no |"));
        auto& o = ok.try_emplace(m, std::make_pair(true, 0.0)).first->second;
        o.first = o.first && !bad;
        o.second += mean_of(s.at("track_1_30")) - mean_of(ms.at("v1").at("track_1_30"));
      }
    }
    md.line();
    if (split == "val") {
      std::string best;
      double gain = -1e30;
      for (const auto& cfg : g4a_configs()) {  // in order of inputs: ties go to fewer
        const auto it = ok.find(cfg_tag(cfg));
        if (it == ok.end() || !it->second.first) continue;
        if (it->second.second / 3.0 > gain + 1e-9) {
          gain = it->second.second / 3.0;
          best = it->first;
        }
      }
      md.line(best.empty() ? "**G4a validation: no configuration is free of a worse measure; G4a stops.**"
                           : std::format("**G4a choice (validation):** `{}`, mean first-second gain over the effects {:+.2f} dB.", best, gain));
      md.line();
    } else {
      for (const auto& [m, o] : ok) {
        const auto& ms = bm.at("explosion");
        const auto iv = diff(ms.at(m).at("track_1_30"), ms.at("v1").at("track_1_30"));
        const bool pass = iv.lo > 0 && o.first;
        md.line(std::format("**G4a decision (test), `{}`:** explosion first second {} dB; an effect worse in a rule measure: {}. **{}**", m, iv_text(iv, 2),
                            o.first ? "no" : "yes", pass ? "Kept." : "Not kept."));
      }
      md.line();
    }
  }
}

// G5a: per setting, the mean over slots of the shard each policy plays.
void summary_g5a(const Ctx& c, Md& md) {
  for (const std::string split : {"val", "test"}) {
    const fs::path f = c.results / std::format("g_extras_g5a_{}.csv", split);
    if (!fs::exists(f)) continue;
    const Csv t = load_csv(f);
    std::map<std::string, std::map<int, std::map<int, std::vector<std::vector<std::string>>>>> cand;  // effect, setting, slot -> rows by cand
    std::map<std::string, std::pair<std::vector<float>, std::vector<int>>> auc;
    for (const auto& r : t.rows) {
      const std::string e = r[0];
      const int s = std::stoi(r[1]), slot = std::stoi(r[2]), j = std::stoi(r[3]);
      const float p = std::stof(r[6]);
      if (slot < 0) {
        auc[e].first.push_back(p);
        auc[e].second.push_back(1);
        continue;
      }
      if (j == 0 || true) {
        auc[e].first.push_back(p);
        auc[e].second.push_back(0);
      }
      auto& v = cand[e][s][slot];
      if (static_cast<int>(v.size()) <= j) v.resize(sz(j + 1));
      v[sz(j)] = r;
    }
    md.line(std::format("#### G5a {}: shard policies (means over settings of the mean over 4 slots; differences paired over settings)", split));
    md.line();
    for (const auto& [e, sets] : cand) {
      const int kmax = static_cast<int>(sets.begin()->second.begin()->second.size());
      md.line(std::format("{}: critic AUC, real windows against every candidate's first second: {:.3f}", e, dcm::roc_auc(auc[e].first, auc[e].second)));
      md.line();
      md.line("| policy | score | spectrum | \\|ln motion\\| | coverage L1 | mean-frame PSNR |");
      md.line("|---|---:|---:|---:|---:|---:|");
      const auto& meas = endless_measures();
      std::map<std::string, std::map<std::string, std::vector<double>>> pol;  // policy -> measure -> per setting
      std::vector<std::string> order;
      const auto add = [&](const std::string& name, const std::function<int(const std::vector<std::vector<std::string>>&)>& pick, bool average) {
        if (!pol.contains(name)) order.push_back(name);
        for (const auto& [s, slots] : sets) {
          (void)s;
          for (const auto& m : meas) {
            double acc = 0;
            for (const auto& [slot, rows] : slots) {
              (void)slot;
              if (average) {
                double a = 0;
                for (const auto& r : rows) a += cell_value(t, r, m.name) / static_cast<double>(rows.size());
                acc += a;
              } else {
                acc += cell_value(t, rows[sz(pick(rows))], m.name);
              }
            }
            pol[name][m.name].push_back(acc / static_cast<double>(slots.size()));
          }
        }
      };
      add("runtime (candidate 0)", [](const auto&) { return 0; }, false);
      for (const int K : {2, 4, 8}) {
        if (K > kmax) continue;
        add(std::format("critic, K = {}", K), [K](const auto& rows) {
          int b = 0;
          for (int j = 1; j < K; ++j) {
            if (std::stod(rows[sz(j)][6]) > std::stod(rows[sz(b)][6])) b = j;
          }
          return b;
        }, false);
        add(std::format("oracle, K = {}", K), [K, &t](const auto& rows) {
          int b = 0;
          for (int j = 1; j < K; ++j) {
            if (cell_value(t, rows[sz(j)], "score") < cell_value(t, rows[sz(b)], "score")) b = j;
          }
          return b;
        }, false);
      }
      add(std::format("mean of {} candidates", kmax), [](const auto&) { return 0; }, true);
      for (const auto& name : order) {
        std::string row = "| " + name + " |";
        for (const auto& m : meas) row += std::format(" {:.4f} |", mean_of(pol[name][m.name]));
        md.line(row);
      }
      md.line();
      md.line("| policy - runtime | score | spectrum | \\|ln motion\\| | coverage L1 | mean-frame PSNR | passes the rule |");
      md.line("|---|---|---|---|---|---|---|");
      for (const auto& name : order) {
        if (name.starts_with("runtime")) continue;
        std::string row = "| " + name + " |";
        bool bad = false, better = false;
        for (const auto& m : meas) {
          const auto iv = diff(pol[name][m.name], pol["runtime (candidate 0)"][m.name]);
          row += " " + iv_text(iv, m.name == "mean_frame_psnr" ? 2 : 4) + " |";
          if (m.name == "score") better = iv.hi < 0;
          else if (worse(iv, m.higher_better)) bad = true;
        }
        md.line(row + (name == "critic, K = 4" ? (better && !bad ? " **yes** |" : " **no** |") : " |"));
      }
      md.line();
    }
  }
}

void summary_g5b(const Ctx& c, Md& md) {
  const std::vector<std::string> meas = {"f1", "f8", "f30", "f60", "track_mean", "score", "spectrum_l1", "abs_ln_motion", "coverage_l1", "mean_frame_psnr"};
  for (const std::string split : {"val", "test"}) {
    const fs::path f = c.results / std::format("g_extras_g5b_{}.csv", split);
    if (!fs::exists(f)) continue;
    const auto bm = by_method(load_csv(f), meas);
    md.line(std::format("#### G5b {}: smoke, v1 against the update mixers (means over settings; differences paired over settings)", split));
    md.line();
    md.line("| method | frame 1 | 8 | 30 | 60 | mean 1-60 | score | spectrum | \\|ln motion\\| | coverage L1 | mean-frame PSNR |");
    md.line("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    const auto& ms = bm.at("smoke");
    for (const auto& [m, s] : ms) {
      std::string row = "| " + m + " |";
      for (const auto& k : meas) row += std::format(" {:.4f} |", mean_of(s.at(k)));
      md.line(row);
    }
    md.line();
    md.line("| method - v1 | frame 30 | frame 60 | mean 1-60 | score | spectrum | \\|ln motion\\| | coverage L1 | mean-frame PSNR | endless worse |");
    md.line("|---|---|---|---|---|---|---|---|---|---|");
    std::string best;
    double gain = -1e30;
    for (const auto& [m, s] : ms) {
      if (m == "v1") continue;
      std::string row = "| " + m + " |";
      bool bad = false;
      for (const auto& k : {"f30", "f60", "track_mean", "score", "spectrum_l1", "abs_ln_motion", "coverage_l1", "mean_frame_psnr"}) {
        const std::string ks = k;
        const bool hi = ks.starts_with("f") || ks == "track_mean" || ks == "mean_frame_psnr";
        const auto iv = diff(s.at(ks), ms.at("v1").at(ks));
        row += " " + iv_text(iv, hi ? 2 : 4) + " |";
        if (!ks.starts_with("f") && ks != "track_mean" && worse(iv, hi)) bad = true;
      }
      md.line(row + (bad ? " **yes** |" : " no |"));
      const double g = 0.5 * (mean_of(s.at("f30")) - mean_of(ms.at("v1").at("f30")) + mean_of(s.at("f60")) - mean_of(ms.at("v1").at("f60")));
      if (split == "val" && !bad && g > gain) {
        gain = g;
        best = m;
      }
      if (split == "test") {
        const auto a = diff(s.at("f30"), ms.at("v1").at("f30")), b = diff(s.at("f60"), ms.at("v1").at("f60"));
        md.line();
        md.line(std::format("**G5b decision (test), `{}`:** frame 30 {}, frame 60 {}, endless worse: {}. **{}**", m, iv_text(a, 2), iv_text(b, 2),
                            bad ? "yes" : "no", a.lo > 0 && b.lo > 0 && !bad ? "Kept." : "Not kept."));
      }
    }
    md.line();
    if (split == "val") {
      md.line(best.empty() ? "**G5b validation: no candidate keeps the endless statistics; G5b stops.**"
                           : std::format("**G5b choice (validation):** `{}`, mean gain at frames 30 and 60 {:+.2f} dB.", best, gain));
      md.line();
    }
  }
}

void summary(const Ctx& c) {
  Md md;
  md.line("# Study G extras (stage S8): tables");
  md.line();
  md.line("Written by `nvfx_g_extras summary` from `results/experiments/g_extras_*.csv`; rules in docs/DCM.md §10. Intervals: 95% paired");
  md.line("bootstrap, 10,000 resamples, over settings; (tie): the interval covers zero.");
  md.line();
  summary_g4a(c, md);
  summary_g5a(c, md);
  summary_g5b(c, md);
  const std::string text = md.o.str();
  std::print("{}", text);
  write_text(c.results / "g_extras_summary.md", text);
}

// =====================================================================================================================
// costs: thread CPU time (the least of `reps`), provisional on a shared machine
// =====================================================================================================================

template <class F>
double least_ms(F&& fn, int reps) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    const double t0 = thread_cpu_s();
    fn();
    best = std::min(best, (thread_cpu_s() - t0) * 1e3);
  }
  return best;
}

double load_average() {
  std::ifstream f("/proc/loadavg");
  double v = 99;
  f >> v;
  return v;
}

void bench(const Ctx& c) {
  const Models M(c);
  const int reps = c.quick ? 3 : 15;
  std::vector<std::string> rows;
  const double load0 = load_average();
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e)) continue;
    const rollout::Model& m = M.of(e);
    const std::vector<float> ctl{0.5f, 0.5f, 0.5f};
    rollout::State s = rollout::start(m, nearest_start(m, ctl), kSize, ctl, 7);
    for (int f = 0; f < 20; ++f) rollout::step(m, s, ctl, 7);
    std::vector<float> rgba(sz(kSize) * sz(kSize) * 4);
    const auto [o1, o2] = M.others(e);
    ex::SimRenderer sr(e, kSize);
    ex::RenderPlanes planes;
    const double step_ms = least_ms([&] {
      rollout::State t = s;
      rollout::step(m, t, ctl, 7);
    }, reps);
    const double learned = least_ms([&] { dcm::fine::render(m, s, rgba); }, reps);
    const double simr = least_ms([&] { sr.render(s, rgba); }, reps);
    const double shader = least_ms([&] { ex::field_shader(m, s, ex::ShaderLook{}, rgba); }, reps);
    rows.push_back(std::format("{},reference_step,{:.3f}", ename(e), step_ms));
    rows.push_back(std::format("{},g4a_learned_renderer,{:.3f}", ename(e), learned));
    rows.push_back(std::format("{},g4a_sim_renderer,{:.3f}", ename(e), simr));
    rows.push_back(std::format("{},g4a_field_shader,{:.3f}", ename(e), shader));
    ex::render_experts(m, s, sr, ex::ShaderLook{}, o1, o2, planes);
    for (const auto& cfg : g4a_configs()) {
      const fs::path p = g4a_mixer_path(c, cfg);
      if (!fs::exists(p)) continue;
      const ex::RenderMixer mix = ex::RenderMixer::load(cfg, read_text(p));
      rows.push_back(std::format("{},g4a_mixing_{},{:.3f}", ename(e), cfg_tag(cfg), least_ms([&] { ex::render_mixed(mix, m, s, planes, rgba); }, reps)));
    }
    // G5a: the critic's part per candidate (features of 30 frames and the mixer); the look-ahead frames themselves are
    // runtime frames (REPORT §6.7)
    Clip w;
    w.allocate(kSize, kLook);
    for (int f = 0; f < kLook; ++f) {
      rollout::step(m, s, ctl, 7);
      dcm::fine::render(m, s, rgba);
      to_u8(rgba, w.frame(f));
    }
    if (fs::exists(g5a_critic_path(c, e))) {
      ex::ShardCritic cr = ex::ShardCritic::load(read_text(g5a_critic_path(c, e)));
      rows.push_back(std::format("{},g5a_critic_per_candidate,{:.3f}", ename(e), least_ms([&] { (void)cr.p_real(ex::critic_features(w), ctl); }, reps)));
    }
    if (e == sim::Effect::smoke) {
      for (const std::string n : {"m1", "m2"}) {
        if (!fs::exists(g5b_mixer_path(c, n))) continue;
        ex::UpdateMixer mx = ex::UpdateMixer::load(read_text(g5b_mixer_path(c, n)));
        rows.push_back(std::format("{},g5b_mixed_step_{},{:.3f}", ename(e), n, least_ms([&] {
          rollout::State t = s;
          ex::CoarseSolver solver(e, ctl, 7, m.h.res, m.fps);
          ex::mixed_step(m, t, ctl, 7, solver, mx);
        }, reps)));
      }
      std::vector<float> out(sz(m.h.res) * sz(m.h.res) * rollout::kPhys);
      ex::CoarseSolver solver(e, ctl, 7, m.h.res, m.fps);
      rows.push_back(std::format("{},g5b_coarse_solver_step,{:.3f}", ename(e), least_ms([&] { solver.step(m, s, out); }, reps)));
    }
    log(std::format("bench {}: done", ename(e)));
  }
  const double load1 = load_average();
  std::ofstream o(c.results / "g_extras_cost.csv");
  o << std::format("# thread CPU ms per 128 x 128 frame (or per call), least of {}; reference code, baseline ISA; load {:.1f} to {:.1f}: provisional\n", reps,
                   load0, load1);
  o << "effect,part,ms\n";
  for (const auto& r : rows) o << r << "\n";
}

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"help", "quick"});
  const auto& pos = a.positional();
  if (a.flag("help") || pos.empty()) {
    std::println("nvfx_g_extras g4a-pilot|g4a-rows|g4a-train|g4a-eval|g5a-train|g5a-eval|g5b-train|g5b-eval|bench [--split val|test] "
                 "[--method NAME] [--effects LIST] [--threads 1] [--quick]  (see the source's header)");
    return 0;
  }
  const char* env = std::getenv("NEURALVFX_DATA");
  const fs::path root = env ? fs::path(env) : fs::path("/root/nvfx-data");
  Ctx c;
  c.data = a.has("data") ? fs::path(a.str("data")) : root / "g" / "extras";
  c.models = a.has("models") ? fs::path(a.str("models")) : root / "experiments" / "models" / "d";
  c.results = a.str("results", "results/experiments");
  c.threads = a.i("threads", 1);
  c.quick = a.flag("quick");
  c.effects = a.str("effects", "");
  c.method = a.str("method", "");
  if (c.quick) {
    c.data /= "quick";
    c.results = c.data / "results";
  }
  fs::create_directories(c.data / "logs");
  fs::create_directories(c.results);
  g_log_path = c.data / "logs" / "g_extras.log";
  const std::string split = a.str("split", "val");
  a.warn_unused();
  const std::string cmd = pos[0];
  log(std::format("nvfx_g_extras {} --split {} (threads {}, load {:.1f})", cmd, split, c.threads, load_average()));
  const double t0 = thread_cpu_s();
  if (cmd == "g4a-pilot") g4a_pilot(c);
  else if (cmd == "g4a-rows") g4a_rows(c);
  else if (cmd == "g4a-train") g4a_train(c);
  else if (cmd == "g4a-eval") g4a_eval(c, split);
  else if (cmd == "g5a-train") g5a_train(c);
  else if (cmd == "g5a-eval") g5a_eval(c, split);
  else if (cmd == "g5b-train") g5b_train(c);
  else if (cmd == "g5b-eval") g5b_eval(c, split);
  else if (cmd == "bench") bench(c);
  else if (cmd == "summary") summary(c);
  else throw std::invalid_argument("unknown step " + cmd);
  log(std::format("nvfx_g_extras {} done: {:.0f} s thread CPU on the main thread", cmd, thread_cpu_s() - t0));
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_g_extras: {}", e.what());
  return 1;
}
