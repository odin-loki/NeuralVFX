// nvfx_study_k: study K (docs/REPORT.md §13): an 8-bit integer path for rollout effects.
//
//   nvfx_study_k profile [--effects LIST] [--sizes 64,128,256] [--isa avx2|avx512|baseline] [--precision float|int8|both]
//                        [--frames 200] [--core 3] [--out CSV]
//       where a frame's time goes: thread CPU per stage of a runner (the coarse step's noise and FiLM, its convolutions,
//       its projection and advection; the detail layer; the renderer's per-cell and per-pixel parts), medians over the
//       frames, and the instance's frame through nvfx_render (shards and crossfades included), means over two shards.
//   nvfx_study_k profile-summary [--csv profile.csv]
//       per ISA, precision and size: each stage's median over the rounds, its mean (least-greatest) over the effects.
//
//   nvfx_study_k track   --split train|val|test [--effects LIST] [--runs N] [--threads 2]
//       tracking from the true state: each run of the split is simulated, its state at the start becomes the effect's
//       only start point (fine fields at full size), and the runtime plays it with the run's seed at float and at each
//       int8 candidate; active PSNR against the simulation at 8 frames, 1 s and 2 s (k_track_<split>.csv).
//   nvfx_study_k endless --split train|val|test [--effects LIST] [--reps R] [--threads 2]
//       endless statistics: per setting R real runs and R plays (shards from the stored start points, as shipped) at
//       float and at each candidate, pooled, distances from the real runs (k_stats_<split>.csv); real runs' and plays'
//       statistics are cached under DATA/k_int8, so each step resumes.
//   nvfx_study_k decide  --split val|test [--effects LIST]
//       the rule (docs/REPORT.md §13): each candidate minus float, paired, 95% bootstrap (10,000 resamples), against
//       the margins fixed in advance (k_rule_<split>.csv; a table on stdout).
//   nvfx_study_k timing  [--effects LIST|all|every] [--models DIR] [--sizes 64,128,256] [--isas avx2,avx512] [--rounds 3] [--frames 200]
//                        [--core 3] [--csv k_timing.csv] [--label NAME]
//       ms per frame through nvfx_render, float and each candidate on each ISA, every configuration once per round
//       (interleaved), thread CPU time on one pinned core: the median and the mean over the frames (k_timing.csv).
//   nvfx_study_k timing-summary [--csv k_timing.csv] [--label NAME]
//       the table of the rounds' medians with their spread, and the int8 candidates' ratios to float.
//
// The splits (seeds: real run k of setting s has seed base + 100 s + k):
//   train  the controls of the effect's salt-1 training runs 0 to 9 (development probes only); tracking: salt-1 runs
//          1000 + r (other seeds); endless: real 5,100,000, plays 5,200,000;
//   val    tracking: the salt-3 validation runs 0 to 15 (as studies I and J); endless: study G's 10 validation settings
//          (docs/DCM.md §4), real 5,300,000, plays 5,400,000;
//   test   study B's 10 held-out settings: tracking from runs at them with fresh seeds 5,900,000 + 100 s + k (k = 0, 1);
//          endless: real 5,700,000, plays 5,800,000.
// Candidates: float, int8_second (the stepper's second convolution in 8 bits), int8_both (both convolutions);
// --candidates LIST keeps only those (with float) in track, endless, decide and timing.
//
// Options: --root DATA (default $NEURALVFX_DATA or /root/nvfx-data), --out DIR (CSVs; default DATA/k_int8).
#include "args.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include "nvfx_internal.hpp"
#include "rt_common.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <print>
#include <set>
#include <tuple>
#include <random>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <time.h>
#include <vector>

namespace fs = std::filesystem;
using namespace nfx;

namespace {

std::size_t sz(int v) { return static_cast<std::size_t>(v); }
std::string ename(sim::Effect e) { return std::string(sim::effect_name(e)); }

double thread_ms() {
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return static_cast<double>(t.tv_sec) * 1e3 + static_cast<double>(t.tv_nsec) * 1e-6;
}

void pin(int core) {
  if (core < 0) return;
  cpu_set_t one;
  CPU_ZERO(&one);
  CPU_SET(core, &one);
  sched_setaffinity(0, sizeof(one), &one);
}

double median(std::vector<double> v) {
  if (v.empty()) return std::nan("");
  std::ranges::sort(v);
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// The eleven rollout effects and where their shipped files are (docs/REPORT.md §6, §12; EFFECTS*.md).
struct Shipped {
  sim::Effect e;
  const char* dir;
};
const std::vector<Shipped>& shipped() {
  static const std::vector<Shipped> v = {
      {sim::Effect::fire, "v2/models"},           {sim::Effect::smoke, "v2/models"},           {sim::Effect::explosion, "v2/models"},
      {sim::Effect::steam, "fx/models/d"},        {sim::Effect::magic, "fx/models/d"},         {sim::Effect::jet, "fx_energy/models/d"},
      {sim::Effect::orb, "fx_energy/models/d"},   {sim::Effect::vortex, "fx_energy/models/d"}, {sim::Effect::fog, "fx_atmos/models/d"},
      {sim::Effect::dust, "fx_atmos/models/d"},   {sim::Effect::cloud, "fx_atmos/models/d"},
  };
  return v;
}
// The ten trained after study K (EFFECTS_BLAZE.md, EFFECTS_FLOW.md, EFFECTS_WATER.md): named one by one, or all with
// `--effects every`; `all` stays study K's eleven.
const std::vector<Shipped>& later() {
  static const std::vector<Shipped> v = {
      {sim::Effect::wildfire, "fx_blaze/models/d"},   {sim::Effect::coals, "fx_blaze/models/d"},  {sim::Effect::mushroom, "fx_blaze/models/d"},
      {sim::Effect::launch, "fx_flow/models/d"},      {sim::Effect::puffs, "fx_flow/models/d"},   {sim::Effect::ink, "fx_flow/models/d"},
      {sim::Effect::waterfall, "fx_water/models/d"},  {sim::Effect::fountain, "fx_water/models/d"}, {sim::Effect::splash, "fx_water/models/d"},
      {sim::Effect::rainshaft, "fx_water/models/d"},
  };
  return v;
}
// --models DIR: every effect's file from one folder (e.g. the library's, DATA/library/models: the models as shipped
// now) instead of each study's own.
fs::path g_models;
fs::path model_path(const fs::path& root, sim::Effect e) {
  if (!g_models.empty()) return g_models / std::format("{}.nvfx", ename(e));
  for (const auto* list : {&shipped(), &later()}) {
    for (const Shipped& s : *list) {
      if (s.e == e) return root / s.dir / std::format("{}.nvfx", ename(e));
    }
  }
  throw std::invalid_argument("not a rollout effect");
}
std::vector<sim::Effect> effects_of(const std::string& list) {
  std::vector<sim::Effect> v;
  if (list.empty() || list == "all" || list == "every") {
    for (const Shipped& s : shipped()) v.push_back(s.e);
    if (list == "every") {
      for (const Shipped& s : later()) v.push_back(s.e);
    }
    return v;
  }
  std::size_t a = 0;
  while (a <= list.size()) {
    const std::size_t b = std::min(list.find(',', a), list.size());
    sim::Effect e{};
    if (!sim::parse_effect(list.substr(a, b - a), e)) throw std::invalid_argument("unknown effect " + list.substr(a, b - a));
    v.push_back(e);
    a = b + 1;
  }
  return v;
}
std::vector<int> ints(const std::string& s) {
  std::vector<int> v;
  for (const float f : tools::parse_floats(s)) v.push_back(static_cast<int>(f));
  return v;
}

rt::RolloutEffect load_effect(const fs::path& p) {
  auto m = rollout::load_model(p);
  if (!m) throw std::runtime_error(m.error());
  rt::RolloutEffect e;
  e.m = std::move(*m);
  return e;
}

std::unique_ptr<rt::RolloutRunner> make_runner(const rt::RolloutEffect& e, int size, nvfx_isa isa, rt::Precision p, rt::Int8Convs c) {
  switch (isa) {
    case NVFX_ISA_AVX512: return rt::isa_avx512::make_rollout(e, size, false, p, c);
    case NVFX_ISA_AVX2: return rt::isa_avx2::make_rollout(e, size, false, p, c);
    default: return rt::isa_base::make_rollout(e, size, false, p, c);
  }
}
rt::Int8Convs parse_convs(const std::string& s) {
  if (s == "second") return rt::Int8Convs::second;
  if (s == "both") return rt::Int8Convs::both;
  throw std::invalid_argument("--convs second or both");
}
nvfx_isa parse_isa(const std::string& s) {
  if (s == "avx2") return NVFX_ISA_AVX2;
  if (s == "avx512") return NVFX_ISA_AVX512;
  if (s == "baseline") return NVFX_ISA_BASELINE;
  throw std::invalid_argument("--isa avx2, avx512 or baseline");
}
const char* isa_name(nvfx_isa i) { return i == NVFX_ISA_AVX512 ? "avx512" : i == NVFX_ISA_AVX2 ? "avx2" : "baseline"; }
std::vector<rt::Precision> precisions_of(const std::string& s) {
  if (s == "float") return {rt::Precision::float32};
  if (s == "int8") return {rt::Precision::int8};
  if (s == "both") return {rt::Precision::float32, rt::Precision::int8};
  throw std::invalid_argument("--precision float, int8 or both");
}

// --- profile ---------------------------------------------------------------------------------------------------------

constexpr std::array<const char*, rt::RolloutProfile::kStages> kStageNames = {"noise", "conv", "project", "detail", "shade_cells", "shade_pixels"};

void step_profile(const tools::Args& a, const fs::path& root) {
  const auto effects = effects_of(a.str("effects", "all"));
  const auto sizes = ints(a.str("sizes", "64,128,256"));
  const nvfx_isa isa = parse_isa(a.str("isa", "avx2"));
  const auto precs = precisions_of(a.str("precision", "float"));
  const int frames = a.i("frames", 200);
  const std::string cname = a.str("convs", "second");
  const rt::Int8Convs convs = parse_convs(cname);
  const fs::path out = a.str("out", (root / "k_int8" / "profile.csv").string());
  pin(a.i("core", 3));
  if (out.has_parent_path()) fs::create_directories(out.parent_path());
  const bool fresh = !fs::exists(out);
  std::ofstream csv(out, std::ios::app);
  if (fresh) {
    csv << "effect,isa,precision,size,frames";
    for (const char* s : kStageNames) csv << "," << s << "_ms";
    csv << ",runner_ms,instance_ms,shard_overhead_ms\n";
  }
  const std::array<float, 3> ctl{0.6f, 0.5f, 0.6f};
  for (const sim::Effect e : effects) {
    const fs::path path = model_path(root, e);
    const rt::RolloutEffect fx = load_effect(path);
    nvfx_effect* eff = nullptr;
    if (nvfx_effect_load(path.string().c_str(), &eff) != NVFX_OK) throw std::runtime_error("cannot load " + path.string());
    const bool once = !fx.m.loop;
    const int n = once ? std::min(frames, fx.m.h.frames - 12) : frames;
    for (const int size : sizes) {
      for (const rt::Precision prec : precs) {
        // the runner alone: per-frame stage times
        auto r = make_runner(fx, size, isa, prec, convs);
        r->start(0, ctl, 1);
        std::vector<std::uint8_t> rgba(sz(size) * sz(size) * 4);
        rt::FrameInput fi;
        for (int f = 0; f < 10; ++f) {  // warm caches (and grow fire's fields)
          r->step(ctl, 1);
          r->render(fi, rgba.data(), sz(size) * 4);
        }
        rt::RolloutProfile prof;
        r->set_profile(&prof);
        std::array<std::vector<double>, rt::RolloutProfile::kStages> per;
        std::vector<double> whole;
        for (int f = 0; f < n; ++f) {
          const auto before = prof.ms;
          const double t0 = thread_ms();
          r->step(ctl, 1);
          r->render(fi, rgba.data(), sz(size) * 4);
          whole.push_back(thread_ms() - t0);
          for (std::size_t s = 0; s < per.size(); ++s) per[s].push_back(prof.ms[s] - before[s]);
        }
        r->set_profile(nullptr);
        // the instance: nvfx_render over two whole shards (one-shots: their frames), mean per frame
        double inst = std::nan("");
        if (nvfx_set_isa(isa) == NVFX_OK) {
          nvfx_instance* in = nullptr;
          if (nvfx_instance_create(eff, size, &in) == NVFX_OK) {
            rt::set_rollout_int8_convs(in, convs);
            nvfx_instance_set_precision(in, prec == rt::Precision::int8 ? NVFX_PRECISION_INT8 : NVFX_PRECISION_FLOAT);
            nvfx_instance_set_controls(in, ctl.data(), 3);
            nvfx_instance_set_seed(in, 1);
            nvfx_render(in, 0.0, rgba.data(), sz(size) * 4);
            const int F = once ? n : 360;
            const double t0 = thread_ms();
            for (int f = 1; f <= F; ++f) nvfx_render(in, f / static_cast<double>(fx.m.fps), rgba.data(), sz(size) * 4);
            inst = (thread_ms() - t0) / F;
            nvfx_instance_free(in);
          }
          nvfx_set_isa(NVFX_ISA_AUTO);
        }
        double mean_whole = 0;
        for (const double w : whole) mean_whole += w / static_cast<double>(whole.size());
        const std::string pn = prec == rt::Precision::int8 ? "int8_" + cname : "float";
        csv << std::format("{},{},{},{},{}", ename(e), isa_name(isa), pn, size, n);
        std::string line = std::format("{:>9} {:>6} {:>11} {:>4}:", ename(e), isa_name(isa), pn, size);
        for (std::size_t s = 0; s < per.size(); ++s) {
          csv << std::format(",{:.4f}", median(per[s]));
          line += std::format(" {} {:.3f}", kStageNames[s], median(per[s]));
        }
        csv << std::format(",{:.4f},{:.4f},{:.4f}\n", median(whole), inst, inst - mean_whole);
        std::println("{} | runner {:.3f} (mean {:.3f}) | instance {:.3f}", line, median(whole), mean_whole, inst);
        csv.flush();
      }
    }
    nvfx_effect_free(eff);
  }
}


// --- quality: splits, candidates, real runs and plays ----------------------------------------------------------------

constexpr int kSize = 128, kRes = 32;

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

using Setting = std::array<float, 3>;
bool off_grid(const Setting& s) {
  const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
  return off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f});
}
// Study B's ten held-out settings and study G's ten validation settings (the generators of nvfx_d2 and docs/DCM.md §4).
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

struct Split {
  std::string name;
  std::vector<Setting> settings;     // endless
  std::uint64_t real = 0, play = 0;  // endless seeds' bases
};
Split split_of(sim::Effect e, const std::string& name) {
  Split s;
  s.name = name;
  if (name == "train") {
    s.settings = training_settings(e, 10);
    s.real = 5'100'000;
    s.play = 5'200'000;
  } else if (name == "val") {
    s.settings = validation_settings();
    s.real = 5'300'000;
    s.play = 5'400'000;
  } else if (name == "test") {
    s.settings = test_settings();
    s.real = 5'700'000;
    s.play = 5'800'000;
  } else {
    throw std::invalid_argument("--split train, val or test");
  }
  return s;
}

// The runs tracked from their true state.
std::vector<sim::Params> track_runs(sim::Effect e, const std::string& split, int runs) {
  std::vector<sim::Params> v;
  rollout::SimRecipe r = rollout::recipe_for(e);
  if (split == "train" || split == "val") {
    r.salt = split == "train" ? 1 : 3;
    for (int i = 0; i < runs; ++i) v.push_back(rollout::recipe_run(r, static_cast<std::uint64_t>(split == "train" ? 1000 + i : i)));
    return v;
  }
  if (split != "test") throw std::invalid_argument("--split train, val or test");
  const auto set = test_settings();
  for (int i = 0; i < runs; ++i) {
    const std::size_t si = static_cast<std::size_t>(i) % set.size();
    sim::Params p;
    p.effect = e;
    p.intensity = set[si][0];
    p.wind = set[si][1];
    p.turbulence = set[si][2];
    p.seed = 5'900'000 + 100 * si + static_cast<std::uint64_t>(i) / set.size();
    p.size = kSize;
    v.push_back(p);
  }
  return v;
}

struct Candidate {
  std::string name;
  nvfx_precision precision;
  rt::Int8Convs convs;
};
std::vector<Candidate>& candidate_list() {
  static std::vector<Candidate> v = {{"float", NVFX_PRECISION_FLOAT, rt::Int8Convs::second},
                                     {"int8_second", NVFX_PRECISION_INT8, rt::Int8Convs::second},
                                     {"int8_both", NVFX_PRECISION_INT8, rt::Int8Convs::both}};
  return v;
}
const std::vector<Candidate>& candidates() { return candidate_list(); }
// --candidates a,b: only these (and float, the reference), e.g. the test of the candidate chosen on validation.
void keep_candidates(const std::string& list) {
  if (list.empty()) return;
  std::erase_if(candidate_list(), [&](const Candidate& c) { return c.name != "float" && ("," + list + ",").find("," + c.name + ",") == std::string::npos; });
  if (candidate_list().size() < 2) throw std::invalid_argument("--candidates: none of int8_second, int8_both");
}

struct Loaded {
  nvfx_effect* e = nullptr;
  explicit Loaded(const fs::path& p) {
    if (nvfx_effect_load(p.string().c_str(), &e) != NVFX_OK) throw std::runtime_error("cannot load " + p.string());
  }
  explicit Loaded(const rollout::Model& m) {
    std::ostringstream os;
    if (auto r = rollout::save_model(os, m); !r) throw std::runtime_error(r.error());
    const std::string b = os.str();
    if (nvfx_effect_load_memory(b.data(), b.size(), &e) != NVFX_OK) throw std::runtime_error("runtime load failed");
  }
  ~Loaded() { nvfx_effect_free(e); }
  Loaded(const Loaded&) = delete;
  Loaded& operator=(const Loaded&) = delete;
};

// `frames` frames from the runtime at a candidate's precision: a start point (variation >= 0: its run's seed, one
// continuous rollout) or a seed (shards from the stored start points, as shipped).
Clip play(nvfx_effect* e, const Candidate& c, const Setting& ctl, int variation, std::uint64_t seed, int frames) {
  nvfx_instance* in = nullptr;
  if (nvfx_instance_create(e, kSize, &in) != NVFX_OK) throw std::runtime_error("instance");
  if (rt::set_rollout_int8_convs(in, c.convs) != NVFX_OK || nvfx_instance_set_precision(in, c.precision) != NVFX_OK) throw std::runtime_error("precision");
  nvfx_instance_set_controls(in, ctl.data(), 3);
  if (variation >= 0) {
    nvfx_instance_set_drift(in, 0.f);
    nvfx_instance_set_variation(in, variation);
  } else {
    nvfx_instance_set_seed(in, seed);
  }
  Clip clip;
  clip.allocate(kSize, frames);
  clip.fps = 30.f;
  for (int f = 0; f < frames; ++f) nvfx_render(in, f / 30.0, clip.frame(f).data(), kSize * 4);
  nvfx_instance_free(in);
  return clip;
}

double apsnr(std::span<const std::uint8_t> r, std::span<const std::uint8_t> t) {  // study D's active PSNR of one frame
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

void append_rows(const fs::path& path, const std::string& header, const std::vector<std::string>& rows) {
  if (path.has_parent_path()) fs::create_directories(path.parent_path());
  const bool fresh = !fs::exists(path) || fs::file_size(path) == 0;
  std::ofstream o(path, std::ios::app);
  if (fresh) o << header << "\n";
  for (const auto& r : rows) o << r << "\n";
}

// --- track ----------------------------------------------------------------------------------------------------------

const std::array<int, 3> kHorizons{8, 30, 60};  // frames after the start: 8 frames, 1 s, 2 s

void step_track(const tools::Args& a, const fs::path& root, const fs::path& out) {
  const std::string split = a.need("split");
  const int threads = a.i("threads", 2);
  const fs::path csv = out / std::format("k_track_{}.csv", split);
  for (const sim::Effect e : effects_of(a.str("effects", "all"))) {
    auto loaded = rollout::load_model(model_path(root, e));
    if (!loaded) throw std::runtime_error(loaded.error());
    const rollout::Model M = std::move(*loaded);
    const bool once = !sim::effect_loops(e);
    const int runs = a.i("runs", split == "test" ? 20 : 16), warm = once ? 1 : 100, K = kHorizons.back();
    const auto params = track_runs(e, split, runs);
    std::vector<std::string> rows(params.size() * candidates().size());
    parallel(threads, static_cast<int>(params.size()), [&](int r) {
      const sim::Params& p = params[sz(r)];
      sim::Fluid truth(p);
      for (int i = 0; i < warm; ++i) truth.step_frame();
      const sim::State st = truth.state();
      rollout::Model mt = M;  // the true state as the only start point, its fine fields at full size
      mt.h.start_fine = kSize;
      rollout::StartPoint sp;
      sp.controls = {p.intensity, p.wind, p.turbulence};
      sp.seed = p.seed;
      sp.time = st.time;
      sp.coarse.resize(sz(kRes) * kRes * rollout::kPhys);
      rollout::coarse_from_sim(st, kRes, p.fps, sp.coarse);
      sp.fine_t = st.temp;
      sp.fine_d = st.soot;
      mt.starts = {sp};
      Clip ref;
      ref.allocate(kSize, K);
      for (int i = 0; i < K; ++i) {
        truth.step_frame();
        truth.render(ref.frame(i));
      }
      const Loaded fx(mt);
      for (std::size_t c = 0; c < candidates().size(); ++c) {
        const Clip clip = play(fx.e, candidates()[c], {p.intensity, p.wind, p.turbulence}, 0, 0, K + 1);  // frame 0: the start
        std::string row = std::format("{},{},{},{}", ename(e), r, p.seed, candidates()[c].name);
        for (const int h : kHorizons) row += std::format(",{:.4f}", apsnr(ref.frame(h - 1), clip.frame(h)));
        rows[sz(r) * candidates().size() + c] = row;
      }
    });
    append_rows(csv, "effect,run,seed,candidate,f8,f30,f60", rows);
    std::println("track {}: {} runs", ename(e), runs);
  }
}

// --- endless --------------------------------------------------------------------------------------------------------

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
  const fs::path tmp = p.string() + ".tmp";
  {
    std::ofstream o(tmp, std::ios::binary);
    put_vec(o, s.coverage);
    put_vec(o, s.emission);
    put_vec(o, s.spectrum);
    put_vec(o, std::vector<double>(s.mean_frame.begin(), s.mean_frame.end()));
    o.write(reinterpret_cast<const char*>(&s.motion), sizeof(double));
  }
  fs::rename(tmp, p);
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
// The statistics of several runs pooled (study D2's protocol, docs/REPORT.md §6.9).
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
int frames_of(sim::Effect e) { return sim::effect_loops(e) ? 300 : 89; }
int warm_of(sim::Effect e) { return sim::effect_loops(e) ? 150 : 1; }

metrics::ClipStats real_stats(const fs::path& cache, sim::Effect e, const Setting& s, std::uint64_t seed) {
  const fs::path p = cache / "real" / std::format("{}_{}.stats", ename(e), seed);
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
metrics::ClipStats play_stats(const fs::path& cache, nvfx_effect* fx, sim::Effect e, const Candidate& c, const Setting& s, std::uint64_t seed) {
  const fs::path p = cache / "plays" / c.name / std::format("{}_{}.stats", ename(e), seed);
  metrics::ClipStats st;
  if (load_stats(p, st)) return st;
  st = metrics::stats(play(fx, c, s, -1, seed, frames_of(e)));
  save_stats(p, st);
  return st;
}

void step_endless(const tools::Args& a, const fs::path& root, const fs::path& out) {
  const std::string name = a.need("split");
  const int threads = a.i("threads", 2), reps = a.i("reps", 8);
  const fs::path cache = root / "k_int8" / "cache", csv = out / std::format("k_stats_{}.csv", name);
  for (const sim::Effect e : effects_of(a.str("effects", "all"))) {
    const Split sp = split_of(e, name);
    const int n = static_cast<int>(sp.settings.size());
    const auto seed = [&](std::uint64_t base, int si, int k) { return base + 100 * static_cast<std::uint64_t>(si) + static_cast<std::uint64_t>(k); };
    const Loaded fx(model_path(root, e));
    const int nc = static_cast<int>(candidates().size());
    // every real run and play into the cache (the longest jobs, real runs, first)
    parallel(threads, n * reps * (1 + nc), [&](int j) {
      const int si = j % n, k = (j / n) % reps, w = j / (n * reps);
      if (w == 0) (void)real_stats(cache, e, sp.settings[sz(si)], seed(sp.real, si, k));
      else (void)play_stats(cache, fx.e, e, candidates()[sz(w - 1)], sp.settings[sz(si)], seed(sp.play, si, k));
    });
    std::vector<std::string> rows;
    for (int si = 0; si < n; ++si) {
      std::vector<metrics::ClipStats> rv;
      for (int k = 0; k < reps; ++k) rv.push_back(real_stats(cache, e, sp.settings[sz(si)], seed(sp.real, si, k)));
      const metrics::ClipStats ref = pool(rv);
      for (const Candidate& c : candidates()) {
        std::vector<metrics::ClipStats> pv;
        for (int k = 0; k < reps; ++k) pv.push_back(play_stats(cache, fx.e, e, c, sp.settings[sz(si)], seed(sp.play, si, k)));
        const metrics::StatDistance d = metrics::distance(ref, pool(pv));
        rows.push_back(std::format("{},{},{},{},{:.6f},{:.6f},{:.6f},{:.7f},{:.7f},{:.4f}", ename(e), si, c.name, reps, d.spectrum_l1, d.motion_ratio,
                                   std::abs(std::log(std::max(1e-3, d.motion_ratio))), d.coverage_l1, d.emission_l1, d.mean_frame_psnr));
      }
    }
    append_rows(csv, "effect,setting,candidate,reps,spectrum_l1,motion_ratio,abs_ln_motion,coverage_l1,emission_l1,mean_frame_psnr", rows);
    std::println("endless {}: {} settings x {} runs", ename(e), n, reps);
  }
}

// --- decide: the rule (docs/REPORT.md §13), fixed before validation ----------------------------------------------------

// Non-inferiority margins: a candidate minus float, paired, 95% bootstrap interval (10,000 resamples). Tracking: the
// lower end at least -kTrack[h] dB. Endless: the upper end of the distances' change at most the margin (coverage and
// emission: a fraction of float's own mean distance for the effect); the lower end of the mean-frame PSNR's change at
// least -kMeanFrame dB.
constexpr std::array<double, 3> kTrack{0.10, 0.10, 0.20};  // 8 frames, 1 s, 2 s
constexpr double kSpectrum = 0.02, kLnMotion = 0.02, kCoverageFrac = 0.10, kEmissionFrac = 0.10, kMeanFrame = 0.10;

std::vector<std::vector<std::string>> read_csv(const fs::path& p) {
  std::vector<std::vector<std::string>> rows;
  std::ifstream in(p);
  if (!in) throw std::runtime_error("cannot read " + p.string());
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<std::string> cols;
    std::size_t a = 0;
    while (a <= line.size()) {
      const std::size_t b = std::min(line.find(',', a), line.size());
      cols.push_back(line.substr(a, b - a));
      a = b + 1;
    }
    rows.push_back(cols);
  }
  return rows;
}

void step_decide(const tools::Args& a, const fs::path& out) {
  const std::string split = a.need("split");
  const auto track = read_csv(out / std::format("k_track_{}.csv", split));
  const auto stats = read_csv(out / std::format("k_stats_{}.csv", split));
  std::vector<std::string> rows;
  std::println("| effect | candidate | 8 frames | 1 s | 2 s | spectrum | ln motion | coverage | emission | mean frame (dB) | rule |");
  std::println("|---|---|---|---|---|---|---|---|---|---|---|");
  std::map<std::string, int> passed;
  for (const sim::Effect e : effects_of(a.str("effects", "all"))) {
    const std::string en = ename(e);
    for (const Candidate& c : candidates()) {
      if (c.precision == NVFX_PRECISION_FLOAT) continue;
      // paired values: tracking by run (cols 4..6), endless by setting
      std::map<std::string, std::map<std::string, std::vector<double>>> v;  // measure -> candidate -> values
      for (const auto& r : track) {
        if (r[0] != en || (r[3] != c.name && r[3] != "float")) continue;
        for (int h = 0; h < 3; ++h) v[std::format("track_f{}", kHorizons[sz(h)])][r[3]].push_back(std::stod(r[sz(4 + h)]));
      }
      const std::array<std::string, 5> names{"spectrum", "ln_motion", "coverage", "emission", "mean_frame"};
      const std::array<int, 5> col{4, 6, 7, 8, 9};
      for (const auto& r : stats) {
        if (r[0] != en || (r[2] != c.name && r[2] != "float")) continue;
        for (std::size_t k = 0; k < names.size(); ++k) v[names[k]][r[2]].push_back(std::stod(r[sz(col[k])]));
      }
      bool all = true, any = false;
      std::string cells, failed;
      const auto judge = [&](const std::string& m, bool higher_better, double margin, int prec) {
        const auto& a1 = v[m][c.name];
        const auto& b1 = v[m]["float"];
        if (a1.empty() || a1.size() != b1.size()) {
          cells += " - |";
          all = false;
          return;
        }
        any = true;
        const metrics::Interval iv = metrics::paired_bootstrap(a1, b1, 10000, 1);
        const bool pass = higher_better ? iv.lo >= -margin : iv.hi <= margin;
        const bool worse = higher_better ? iv.hi < 0.0 : iv.lo > 0.0;  // the interval entirely on the worse side
        all = all && pass;
        if (!pass) failed += (failed.empty() ? "" : " ") + m;
        double fmean = 0;
        for (const double x : b1) fmean += x / static_cast<double>(b1.size());
        cells += std::format(" {:+.{}f} [{:+.{}f}, {:+.{}f}]{}{} |", iv.mean, prec, iv.lo, prec, iv.hi, prec, pass ? "" : " (fails)", worse ? " (worse)" : "");
        rows.push_back(std::format("{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{},{}", split, en, c.name, m, fmean, iv.mean, iv.lo, iv.hi,
                                   higher_better ? -margin : margin, pass ? 1 : 0, worse ? 1 : 0));
      };
      for (std::size_t h = 0; h < kHorizons.size(); ++h) judge(std::format("track_f{}", kHorizons[h]), true, kTrack[h], 2);
      double cov = 0, emi = 0;
      for (const double x : v["coverage"]["float"]) cov += x / std::max<double>(1, static_cast<double>(v["coverage"]["float"].size()));
      for (const double x : v["emission"]["float"]) emi += x / std::max<double>(1, static_cast<double>(v["emission"]["float"].size()));
      judge("spectrum", false, kSpectrum, 3);
      judge("ln_motion", false, kLnMotion, 3);
      judge("coverage", false, kCoverageFrac * cov, 4);
      judge("emission", false, kEmissionFrac * emi, 4);
      judge("mean_frame", true, kMeanFrame, 3);
      if (all && any) ++passed[c.name];
      std::println("| {} | {} |{} {} |", en, c.name, cells, all && any ? "passes" : "fails: " + failed);
    }
  }
  for (const auto& [c, n] : passed) std::println("{}: passes on {} effects", c, n);
  const fs::path csv = out / std::format("k_rule_{}.csv", split);
  if (fs::exists(csv)) fs::remove(csv);
  append_rows(csv, "split,effect,candidate,measure,float_mean,diff_mean,diff_lo,diff_hi,margin,pass,worse", rows);
}

// --- timing: float against int8, interleaved --------------------------------------------------------------------------

// Every configuration (effect, size, ISA, candidate) once per round, the rounds one after another, each through
// nvfx_render on one pinned core: thread CPU time per frame, the median and the mean over `frames` frames after the
// first (looping effects: 200 frames cross one shard change with its crossfade; one-shots: their 88 frames).
void step_timing(const tools::Args& a, const fs::path& root, const fs::path& out) {
  const auto effects = effects_of(a.str("effects", "all"));
  const auto sizes = ints(a.str("sizes", "64,128,256"));
  std::vector<nvfx_isa> isas;
  for (const std::string& n : {std::string("avx2"), std::string("avx512")}) {
    if (a.str("isas", "avx2,avx512").find(n) != std::string::npos) isas.push_back(parse_isa(n));
  }
  const int rounds = a.i("rounds", 3), frames = a.i("frames", 200);
  pin(a.i("core", 3));
  const fs::path csv = out / a.str("csv", "k_timing.csv");
  const std::string label = a.str("label", "");
  for (int r = 0; r < rounds; ++r) {
    std::vector<std::string> rows;
    for (const sim::Effect e : effects) {
      const Loaded fx(model_path(root, e));
      nvfx_effect_info info{};
      nvfx_effect_get_info(fx.e, &info);
      const int n = info.loops ? frames : std::min(frames, info.frames - 1);
      for (const int size : sizes) {
        for (const nvfx_isa isa : isas) {
          for (const Candidate& c : candidates()) {
            if (nvfx_set_isa(isa) != NVFX_OK) continue;
            nvfx_instance* in = nullptr;
            if (nvfx_instance_create(fx.e, size, &in) != NVFX_OK) throw std::runtime_error("instance");
            rt::set_rollout_int8_convs(in, c.convs);
            nvfx_instance_set_precision(in, c.precision);
            const std::array<float, 3> ctl{0.6f, 0.5f, 0.6f};
            nvfx_instance_set_controls(in, ctl.data(), 3);
            nvfx_instance_set_seed(in, 1);
            std::vector<std::uint8_t> buf(sz(size) * sz(size) * 4);
            nvfx_render(in, 0.0, buf.data(), sz(size) * 4);
            std::vector<double> ms;
            for (int f = 1; f <= n; ++f) {
              const double t0 = thread_ms();
              nvfx_render(in, f / 30.0, buf.data(), sz(size) * 4);
              ms.push_back(thread_ms() - t0);
            }
            double mean = 0;
            for (const double x : ms) mean += x / static_cast<double>(ms.size());
            rows.push_back(std::format("{},{},{},{},{},{},{:.4f},{:.4f},{:.1f}", label, r, ename(e), size, isa_name(isa), c.name, median(ms), mean,
                                       static_cast<double>(nvfx_instance_scratch_bytes(in)) / 1024.0));
            nvfx_instance_free(in);
          }
        }
      }
      nvfx_set_isa(NVFX_ISA_AUTO);
    }
    append_rows(csv, "label,round,effect,size,isa,candidate,median_ms,mean_ms,scratch_kb", rows);
    std::println("timing: round {} done", r + 1);
  }
}

// The timing table: per effect, size, ISA and candidate, the median over the rounds of each round's median frame time
// (and the least and greatest); and each int8 candidate's ratio to float on the same ISA and to float on AVX2 (the
// shipped default), the median over the rounds of the per-round ratio, with the mean over the effects and its 95%
// bootstrap interval.
void step_timing_summary(const tools::Args& a, const fs::path& out) {
  const auto rows = read_csv(out / a.str("csv", "k_timing.csv"));
  const std::string label = a.str("label", "");
  using Key = std::tuple<std::string, int, std::string, std::string>;  // effect, size, isa, candidate
  std::map<Key, std::map<int, double>> med;                            // -> round -> median ms
  std::vector<std::string> effects;
  for (const auto& r : rows) {
    if (r[0] != label) continue;
    med[{r[2], std::stoi(r[3]), r[4], r[5]}][std::stoi(r[1])] = std::stod(r[6]);
    if (std::ranges::find(effects, r[2]) == effects.end()) effects.push_back(r[2]);
  }
  const auto over = [](const std::map<int, double>& m) {
    std::vector<double> v;
    for (const auto& [k, x] : m) v.push_back(x);
    std::ranges::sort(v);
    return std::array<double, 3>{median(v), v.front(), v.back()};
  };
  const auto ratio = [&](const Key& a1, const Key& b1) {  // median over the rounds both have of a / b
    std::vector<double> v;
    for (const auto& [k, x] : med[a1]) {
      if (med[b1].contains(k)) v.push_back(x / med[b1][k]);
    }
    return v.empty() ? std::nan("") : median(v);
  };
  std::println("| effect | size | float AVX2 | float AVX-512 | int8_second AVX2 | int8_second AVX-512 | int8_both AVX2 | int8_both AVX-512 |");
  std::println("|---|---:|---:|---:|---:|---:|---:|---:|");
  std::set<int> sizes;
  for (const auto& [k, v] : med) sizes.insert(std::get<1>(k));
  for (const int size : sizes) {
    for (const auto& e : effects) {
      std::string line = std::format("| {} | {} |", e, size);
      for (const char* c : {"float", "int8_second", "int8_both"}) {
        for (const char* isa : {"avx2", "avx512"}) {
          const Key k{e, size, isa, c};
          if (!med.contains(k)) {
            line += " - |";
            continue;
          }
          const auto [m, lo, hi] = over(med[k]);
          line += std::format(" {:.3f} ({:.3f}-{:.3f}) |", m, lo, hi);
        }
      }
      std::println("{}", line);
    }
  }
  std::map<std::pair<int, std::string>, std::vector<double>> mem;  // (size, candidate) -> KB per effect and ISA
  for (const auto& r : rows) {
    if (r[0] == label && r[1] == "0" && r.size() > 8) mem[{std::stoi(r[3]), r[5]}].push_back(std::stod(r[8]));
  }
  for (const auto& [k, v] : mem) {
    double m = 0;
    for (const double x : v) m += x / static_cast<double>(v.size());
    std::println("instance memory, {} px, {}: {:.0f} KB (mean over effects and ISAs)", k.first, k.second, m);
  }
  std::println("\nratios (median over rounds), mean over the effects [95% bootstrap]:");
  for (const int size : sizes) {
    for (const char* c : {"int8_second", "int8_both"}) {
      for (const auto& [isa, base] : std::vector<std::pair<std::string, std::string>>{{"avx512", "avx512"}, {"avx2", "avx2"}, {"avx512", "avx2"}}) {
        std::vector<double> v;
        for (const auto& e : effects) {
          const double r = ratio({e, size, isa, c}, {e, size, base, "float"});
          if (std::isfinite(r)) v.push_back(r);
        }
        if (v.empty()) continue;
        const metrics::Interval iv = metrics::bootstrap_mean(v, 10000, 1);
        std::println("  {} px {} on {} / float on {}: {:.3f} [{:.3f}, {:.3f}] ({} effects)", size, c, isa, base, iv.mean, iv.lo, iv.hi, v.size());
      }
    }
    std::vector<double> v;
    for (const auto& e : effects) {
      const double r = ratio({e, size, "avx512", "float"}, {e, size, "avx2", "float"});
      if (std::isfinite(r)) v.push_back(r);
    }
    if (!v.empty()) {
      const metrics::Interval iv = metrics::bootstrap_mean(v, 10000, 1);
      std::println("  {} px float on avx512 / float on avx2: {:.3f} [{:.3f}, {:.3f}]", size, iv.mean, iv.lo, iv.hi);
    }
  }
}

// The profile's table: per ISA, precision and size, each stage's median over the rounds, then the mean, least and
// greatest over the effects; and the runner and the instance per frame.
void step_profile_summary(const tools::Args& a, const fs::path& root) {
  const fs::path csv = a.str("csv", (root / "k_int8" / "profile.csv").string());
  const auto rows = read_csv(csv);
  using Key = std::tuple<std::string, std::string, int>;  // isa, precision, size
  std::map<Key, std::map<std::string, std::vector<std::vector<double>>>> v;  // -> effect -> rounds of the 9 figures
  for (const auto& r : rows) {
    std::vector<double> x;
    for (std::size_t c = 5; c < 14 && c < r.size(); ++c) x.push_back(std::stod(r[c]));
    v[{r[1], r[2], std::stoi(r[3])}][r[0]].push_back(x);
  }
  std::println("| ISA | precision | size | effects | noise | conv | project | detail | shade cells | shade pixels | runner | instance | shards |");
  std::println("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
  for (const auto& [k, effects] : v) {
    std::array<std::vector<double>, 9> per;
    for (const auto& [e, rounds] : effects) {
      for (std::size_t c = 0; c < 9; ++c) {
        std::vector<double> m;
        for (const auto& x : rounds) {
          if (c < x.size()) m.push_back(x[c]);
        }
        per[c].push_back(median(m));
      }
    }
    std::string line = std::format("| {} | {} | {} | {} |", std::get<0>(k), std::get<1>(k), std::get<2>(k), effects.size());
    for (const auto& c : per) {
      double mean = 0;
      for (const double x : c) mean += x / static_cast<double>(c.size());
      line += std::format(" {:.3f} ({:.3f}-{:.3f}) |", mean, *std::ranges::min_element(c), *std::ranges::max_element(c));
    }
    std::println("{}", line);
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const tools::Args a(argc, argv, {});
    if (a.positional().empty()) {
      std::println(stderr, "usage: nvfx_study_k profile|... [options] (see the comment at the top of tools/nvfx_study_k.cpp)");
      return 2;
    }
    const char* env = std::getenv("NEURALVFX_DATA");
    const fs::path root = a.str("root", env ? env : "/root/nvfx-data");
    const std::string step = a.positional()[0];
    const fs::path out = a.str("out", (root / "k_int8").string());
    keep_candidates(a.str("candidates", ""));
    g_models = a.str("models", "");
    if (step == "profile") step_profile(a, root);
    else if (step == "profile-summary") step_profile_summary(a, root);
    else if (step == "track") step_track(a, root, out);
    else if (step == "endless") step_endless(a, root, out);
    else if (step == "decide") step_decide(a, out);
    else if (step == "timing") step_timing(a, root, out);
    else if (step == "timing-summary") step_timing_summary(a, out);
    else throw std::invalid_argument("unknown step " + step);
    a.warn_unused();
    return 0;
  } catch (const std::exception& ex) {
    std::println(stderr, "nvfx_study_k: {}", ex.what());
    return 1;
  }
}
