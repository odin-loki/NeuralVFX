// nvfx_experiment: the Phase 3 evaluation (docs/PLAN.md §6, docs/REPORT.md), end to end and cached.
//
//   nvfx_experiment data | a | b | c | report | all   [--root DIR] [--threads 4] [--quick]
//
//   data    simulate every clip (fire, smoke, explosion at 128 px, 64 frames)
//   a       compression: one model per clip, a ladder of model sizes against the flipbook ladder at matched memory,
//           plus the frame-interpolation test (only even frames available)
//   b       controls: one model per effect trained on a 3 x 5 x 3 grid of control settings, scored on held-out
//           settings against nearest-setting and two-setting-blend flipbook libraries
//   c       variation: one model per effect trained on 24 seeds with learned variation codes; new seeds against
//           held-out real clips by distribution statistics, diversity and copying checks
//   report  bootstrap intervals and tables: results/experiments/SUMMARY.md (generated)
//
// Everything trained is scored through the shipping runtime (nvfx.h) at its stored precision. Clips, models,
// sheets and videos go under the data root (never git); CSVs and the summary go under results/experiments.
#include "args.hpp"
#include "experiment_d.hpp"
#include "experiment_g.hpp"

#include <neuralfx/flipbook.hpp>
#include <neuralfx/image_io.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/model.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/sim.hpp>
#include <neuralfx/train.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <print>
#include <random>
#include <ranges>
#include <sched.h>
#include <sstream>

namespace fs = std::filesystem;
using namespace nfx;

namespace {

struct Ctx {
  fs::path data;     // clips, models, videos (outside git)
  fs::path results;  // CSVs and the summary (in git)
  int threads = 4;
  bool quick = false;
  int iters(int full) const { return quick ? std::max(100, full / 10) : full; }
};

constexpr int kSize = 128, kFrames = 64;

// --- data -----------------------------------------------------------------------------------------------------------

fs::path clip_path(const Ctx& c, std::string_view group, std::string_view name) { return c.data / "clips" / group / std::format("{}.nfxclip", name); }

Clip get_clip(const Ctx& c, std::string_view group, std::string_view name, const sim::Params& p) {
  const fs::path path = clip_path(c, group, name);
  if (fs::exists(path)) {
    if (auto r = read_clip(path)) return *r;
  }
  Clip clip = sim::simulate(p);
  if (auto r = write_clip(path, clip); !r) throw std::runtime_error(r.error());
  return clip;
}

sim::Params params(sim::Effect e, float i, float w, float t, std::uint64_t seed) {
  sim::Params p;
  p.effect = e;
  p.intensity = i;
  p.wind = w;
  p.turbulence = t;
  p.seed = seed;
  p.size = kSize;
  p.frames = kFrames;
  return p;
}

std::string ename(sim::Effect e) { return std::string(sim::effect_name(e)); }

// Task A clips: four per effect with different controls and seeds.
std::vector<std::pair<std::string, sim::Params>> a_clips() {
  std::vector<std::pair<std::string, sim::Params>> v;
  const std::array<std::array<float, 3>, 4> s = {{{0.5f, 0.5f, 0.5f}, {0.9f, 0.3f, 0.7f}, {0.2f, 0.7f, 0.2f}, {0.7f, 0.6f, 1.0f}}};
  for (const auto e : sim::kEffects) {
    for (std::size_t k = 0; k < s.size(); ++k) {
      v.emplace_back(std::format("{}_{}", ename(e), k), params(e, s[k][0], s[k][1], s[k][2], 101 + k));
    }
  }
  return v;
}

// Task B: the training grid and the held-out settings (off-grid, fixed seed list).
std::vector<std::array<float, 3>> b_train_settings() {
  std::vector<std::array<float, 3>> v;
  for (const float i : {0.f, 0.5f, 1.f}) {
    for (const float w : {0.f, 0.25f, 0.5f, 0.75f, 1.f}) {
      for (const float t : {0.f, 0.5f, 1.f}) v.push_back({i, w, t});
    }
  }
  return v;
}
std::vector<std::array<float, 3>> b_test_settings() {
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<std::array<float, 3>> v;
  while (v.size() < 10) {
    const std::array<float, 3> s{u(rng), u(rng), u(rng)};
    // off the grid in every coordinate (at least 0.05 from a training value)
    const auto off = [](float x, std::initializer_list<float> g) {
      return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; });
    };
    if (off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f})) v.push_back(s);
  }
  return v;
}

constexpr std::array<float, 3> kCControls = {0.6f, 0.5f, 0.6f};
constexpr int kCTrain = 24, kCTest = 8;

void step_data(const Ctx& c) {
  int n = 0;
  for (const auto& [name, p] : a_clips()) {
    get_clip(c, "a", name, p);
    ++n;
  }
  for (const auto e : sim::kEffects) {
    for (const auto& s : b_train_settings()) get_clip(c, "b", std::format("{}_train_{:.2f}_{:.2f}_{:.2f}", ename(e), s[0], s[1], s[2]), params(e, s[0], s[1], s[2], 1));
    for (const auto& s : b_test_settings()) get_clip(c, "b", std::format("{}_test_{:.2f}_{:.2f}_{:.2f}", ename(e), s[0], s[1], s[2]), params(e, s[0], s[1], s[2], 1));
    for (int k = 0; k < kCTrain + kCTest; ++k) {
      get_clip(c, "c", std::format("{}_seed{}", ename(e), 1000 + k), params(e, kCControls[0], kCControls[1], kCControls[2], 1000 + static_cast<std::uint64_t>(k)));
    }
    n += 55 + kCTrain + kCTest;
    std::println("data: {} done ({} clips so far)", ename(e), n);
  }
}

// --- runtime evaluation ---------------------------------------------------------------------------------------------

struct RtEffect {
  nvfx_effect* e = nullptr;
  explicit RtEffect(const Model& m) {
    std::ostringstream os;
    if (auto r = save_model(os, m); !r) throw std::runtime_error(r.error());
    const std::string b = os.str();
    if (nvfx_effect_load_memory(b.data(), b.size(), &e) != NVFX_OK) throw std::runtime_error("runtime load failed");
  }
  ~RtEffect() { nvfx_effect_free(e); }
  RtEffect(const RtEffect&) = delete;
  RtEffect& operator=(const RtEffect&) = delete;
};

// Render a clip through the runtime: controls, then a training variation (>= 0) or a seed (drift off).
Clip runtime_clip(const Model& m, std::span<const float> controls, int variation, std::uint64_t seed) {
  RtEffect fx(m);
  nvfx_instance* in = nullptr;
  if (nvfx_instance_create(fx.e, kSize, &in) != NVFX_OK) throw std::runtime_error("runtime instance failed");
  nvfx_instance_set_controls(in, controls.data(), static_cast<int>(controls.size()));
  nvfx_instance_set_drift(in, 0.f);
  if (variation >= 0) nvfx_instance_set_variation(in, variation);
  else nvfx_instance_set_seed(in, seed);
  Clip clip;
  clip.allocate(kSize, kFrames);
  clip.loop = m.h.loop;
  clip.fps = m.fps;
  for (int f = 0; f < kFrames; ++f) nvfx_render(in, f / static_cast<double>(m.fps), clip.frame(f).data(), kSize * 4);
  nvfx_instance_free(in);
  return clip;
}

// Median milliseconds per 128x128 frame on one pinned core.
double runtime_ms(const Model& m) {
  cpu_set_t old, one;
  sched_getaffinity(0, sizeof(old), &old);
  CPU_ZERO(&one);
  CPU_SET(3, &one);
  sched_setaffinity(0, sizeof(one), &one);
  RtEffect fx(m);
  nvfx_instance* in = nullptr;
  nvfx_instance_create(fx.e, kSize, &in);
  std::vector<std::uint8_t> buf(kSize * kSize * 4);
  std::vector<double> ms;
  for (int f = 0; f < 140; ++f) {
    const auto t0 = std::chrono::steady_clock::now();
    nvfx_render(in, f / 30.0, buf.data(), kSize * 4);
    if (f >= 20) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  nvfx_instance_free(in);
  sched_setaffinity(0, sizeof(old), &old);
  std::ranges::sort(ms);
  return ms[ms.size() / 2];
}

// --- CSV --------------------------------------------------------------------------------------------------------

struct Row {
  std::map<std::string, std::string> v;
  std::string s(const std::string& k) const { return v.at(k); }
  double d(const std::string& k) const { return std::stod(v.at(k)); }
};

class Csv {
 public:
  Csv(fs::path path, std::vector<std::string> cols) : path_(std::move(path)), cols_(std::move(cols)) {
    if (fs::exists(path_)) {
      std::ifstream in(path_);
      std::string line;
      std::getline(in, line);
      while (std::getline(in, line)) {
        if (line.empty()) continue;
        Row r;
        std::size_t k = 0;
        for (const auto part : std::views::split(line, ',')) {
          if (k < cols_.size()) r.v[cols_[k++]] = std::string(std::string_view(part));
        }
        rows_.push_back(std::move(r));
      }
    }
  }
  bool has(const std::string& key_col, const std::string& key) const {
    return std::ranges::any_of(rows_, [&](const Row& r) { return r.v.contains(key_col) && r.v.at(key_col) == key; });
  }
  void add(Row r) {
    rows_.push_back(std::move(r));
    save();
  }
  const std::vector<Row>& rows() const { return rows_; }

 private:
  void save() const {
    fs::create_directories(path_.parent_path());
    std::ofstream out(path_);
    for (std::size_t i = 0; i < cols_.size(); ++i) out << cols_[i] << (i + 1 < cols_.size() ? "," : "\n");
    for (const Row& r : rows_) {
      for (std::size_t i = 0; i < cols_.size(); ++i) out << (r.v.contains(cols_[i]) ? r.v.at(cols_[i]) : "") << (i + 1 < cols_.size() ? "," : "\n");
    }
  }
  fs::path path_;
  std::vector<std::string> cols_;
  std::vector<Row> rows_;
};

const std::vector<std::string> kScoreCols = {"task", "effect", "clip", "method", "family", "config", "bytes", "psnr",
                                             "active_psnr", "ssim", "tpsnr", "flicker", "ms", "train_s", "spectrum_l1", "motion_ratio"};

// Sharpness and motion against the reference (blur shows as a large spectrum distance and a motion ratio below 1).
void add_stats(Row& r, const metrics::ClipStats& ref, const Clip& test) {
  const auto d = metrics::distance(ref, metrics::stats(test));
  r.v["spectrum_l1"] = std::format("{:.4f}", d.spectrum_l1);
  r.v["motion_ratio"] = std::format("{:.4f}", d.motion_ratio);
}

Row score_row(std::string task, std::string effect, std::string clip, std::string method, std::string family, std::string config,
              std::size_t bytes, const metrics::ClipScores& s, double ms, double train_s) {
  Row r;
  r.v = {{"task", task}, {"effect", effect}, {"clip", clip}, {"method", method}, {"family", family}, {"config", config},
         {"bytes", std::to_string(bytes)}, {"psnr", std::format("{:.4f}", s.psnr)}, {"active_psnr", std::format("{:.4f}", s.active_psnr)},
         {"ssim", std::format("{:.5f}", s.ssim)}, {"tpsnr", std::format("{:.4f}", s.temporal_psnr)}, {"flicker", std::format("{:.4f}", s.flicker)},
         {"ms", std::format("{:.4f}", ms)}, {"train_s", std::format("{:.1f}", train_s)}};
  return r;
}

// --- Task A -----------------------------------------------------------------------------------------------------

struct Config {
  std::string name;
  Hyper h;
  std::vector<int> bits;  // storage precisions to score from one training
  int iters;
};

std::vector<Config> a_configs(const Ctx& c) {
  const auto grid = [](int G, int C, int H, int L, int T) {
    Hyper h;
    h.arch = Arch::grid;
    h.size = kSize;
    h.frames = kFrames;
    h.grid = G;
    h.channels = C;
    h.hidden = H;
    h.layers = L;
    h.grid_t = T;
    return h;
  };
  const auto conv = [](int c0, int c1, int c2, int T) {
    Hyper h;
    h.arch = Arch::conv;
    h.latent = kSize / 8;
    h.size = kSize;
    h.frames = kFrames;
    h.c0 = c0;
    h.c1 = c1;
    h.c2 = c2;
    h.grid_t = T;
    return h;
  };
  return {
      {"grid_s", grid(24, 8, 16, 1, 16), {16, 8}, c.iters(2000)},
      {"grid_m", grid(32, 8, 32, 2, 16), {16, 8}, c.iters(2000)},
      {"grid_l", grid(48, 8, 32, 2, 16), {16, 8}, c.iters(2000)},
      {"grid_mt", grid(32, 8, 32, 2, 32), {16, 8}, c.iters(2000)},
      {"conv_s", conv(16, 8, 8, 16), {16, 8}, c.iters(1500)},
      {"conv_m", conv(32, 16, 8, 16), {16, 8}, c.iters(1500)},
  };
}

void step_a(const Ctx& c) {
  Csv csv(c.results / "a_scores.csv", kScoreCols);
  const auto configs = a_configs(c);
  std::map<std::string, double> ms_cache;
  for (const auto& [name, p] : a_clips()) {
    const Clip ref = get_clip(c, "a", name, p);
    const std::string effect = ename(p.effect);
    const auto ref_stats = metrics::stats(ref);
    // baselines
    if (!csv.has("clip", name)) {
      for (const auto& spec : flipbook::ladder(kSize, kFrames)) {
        const auto fb = flipbook::build(ref, spec);
        const Clip played = flipbook::play(fb);
        Row row = score_row("a", effect, name, "flipbook", spec.flow_res > 0 ? "flipbook_mv" : spec.codec == flipbook::Codec::raw ? "flipbook_raw" : "flipbook_bc3",
                            spec.describe(), fb.bytes, metrics::score(ref, played), 0, 0);
        add_stats(row, ref_stats, played);
        csv.add(row);
      }
    }
    // models
    for (const Config& cfg : configs) {
      const std::string key = std::format("{}|{}", name, cfg.name);
      if (csv.has("method", key + "|16")) continue;
      train::Options o;
      o.iterations = cfg.iters;
      o.threads = c.threads;
      o.log_every = 0;
      const train::Example ex{&ref, {}};
      Hyper h = cfg.h;
      h.loop = ref.loop;
      auto r = train::train(h, std::span(&ex, 1), o);
      r.model.effect = effect;
      for (const int bits : cfg.bits) {
        Model m = r.model;
        m.feature_bits = bits;
        quantise_like_storage(m);
        m.pack_features();
        const std::string mk = std::format("{}|{}", cfg.name, bits);
        if (!ms_cache.contains(mk)) ms_cache[mk] = runtime_ms(m);
        const Clip out = runtime_clip(m, {}, 0, 0);
        Row row = score_row("a", effect, name, key + std::format("|{}", bits), cfg.h.arch == Arch::grid ? "neural_grid" : "neural_conv",
                            std::format("{} {}-bit", h.describe(), bits), m.storage_bytes(), metrics::score(ref, out), ms_cache[mk], r.seconds);
        add_stats(row, ref_stats, out);
        csv.add(row);
        if ((cfg.name == "grid_m" && bits == 8) || name.ends_with("_0")) {  // for videos, the viewer and the timing step
          save_model(c.data / "models" / "a" / std::format("{}_{}{}.nvfx", name, cfg.name, bits), m);
        }
      }
      std::println("A {} {}: {:.1f} s", name, cfg.name, r.seconds);
    }
    // frame interpolation: only even frames available to both
    const std::string ikey = std::format("{}|interp", name);
    if (!csv.has("method", ikey + "|grid_m")) {
      std::vector<int> even, odd;
      for (int f = 0; f < kFrames; ++f) (f % 2 ? odd : even).push_back(f);
      for (const auto& spec : {flipbook::Spec{0, 128, flipbook::Codec::bc3, 0}, flipbook::Spec{0, 128, flipbook::Codec::bc3, 32},
                               flipbook::Spec{0, 128, flipbook::Codec::raw, 0}}) {
        const auto fb = flipbook::build(ref, spec, even);
        csv.add(score_row("a_interp", effect, name, ikey + "|" + spec.describe(), spec.flow_res > 0 ? "flipbook_mv" : "flipbook", spec.describe(),
                          fb.bytes, metrics::score(ref, flipbook::play(fb), odd), 0, 0));
      }
      train::Options o;
      o.iterations = c.iters(2000);
      o.threads = c.threads;
      o.log_every = 0;
      o.frames = even;
      const train::Example ex{&ref, {}};
      Hyper h = configs[1].h;
      h.loop = ref.loop;
      auto r = train::train(h, std::span(&ex, 1), o);
      Model m = r.model;
      m.feature_bits = 8;
      quantise_like_storage(m);
      m.pack_features();
      csv.add(score_row("a_interp", effect, name, ikey + "|grid_m", "neural_grid", h.describe() + " 8-bit", m.storage_bytes(),
                        metrics::score(ref, runtime_clip(m, {}, 0, 0), odd), 0, r.seconds));
    }
  }
}

// --- Task B -----------------------------------------------------------------------------------------------------

void step_b(const Ctx& c) {
  Csv csv(c.results / "b_scores.csv", kScoreCols);
  for (const auto e : sim::kEffects) {
    const std::string effect = ename(e);
    std::vector<Clip> train_clips;
    std::vector<std::array<float, 3>> train_s = b_train_settings();
    for (const auto& s : train_s) {
      train_clips.push_back(get_clip(c, "b", std::format("{}_train_{:.2f}_{:.2f}_{:.2f}", effect, s[0], s[1], s[2]), params(e, s[0], s[1], s[2], 1)));
    }
    const auto test_s = b_test_settings();
    std::vector<Clip> test_clips;
    for (const auto& s : test_s) {
      test_clips.push_back(get_clip(c, "b", std::format("{}_test_{:.2f}_{:.2f}_{:.2f}", effect, s[0], s[1], s[2]), params(e, s[0], s[1], s[2], 1)));
    }
    // Baselines: libraries of flipbooks at the training settings (BC3, all frames, full size: the best per-setting
    // quality; memory = 45 such flipbooks), played at the nearest setting or blended from the two nearest.
    const std::size_t lib_bytes = train_clips.size() * static_cast<std::size_t>(kSize) * kSize * kFrames;
    if (!csv.has("method", effect + "|nearest")) {
      for (std::size_t t = 0; t < test_s.size(); ++t) {
        std::vector<std::pair<float, std::size_t>> d;
        for (std::size_t k = 0; k < train_s.size(); ++k) {
          float s = 0;
          for (int j = 0; j < 3; ++j) s += (test_s[t][static_cast<std::size_t>(j)] - train_s[k][static_cast<std::size_t>(j)]) * (test_s[t][static_cast<std::size_t>(j)] - train_s[k][static_cast<std::size_t>(j)]);
          d.emplace_back(std::sqrt(s), k);
        }
        std::ranges::sort(d);
        const Clip nearest = flipbook::play(flipbook::build(train_clips[d[0].second], {kFrames, kSize, flipbook::Codec::bc3, 0}));
        const Clip second = flipbook::play(flipbook::build(train_clips[d[1].second], {kFrames, kSize, flipbook::Codec::bc3, 0}));
        Clip blend = nearest;
        const float wa = d[1].first / std::max(1e-6f, d[0].first + d[1].first);
        for (std::size_t i = 0; i < blend.rgba.size(); ++i) {
          blend.rgba[i] = static_cast<std::uint8_t>(std::lround(wa * nearest.rgba[i] + (1.f - wa) * second.rgba[i]));
        }
        const std::string cl = std::format("{}_test{}", effect, t);
        const auto test_stats = metrics::stats(test_clips[t]);
        Row rn = score_row("b", effect, cl, effect + "|nearest", "flipbook_library", "nearest of 45 BC3", lib_bytes, metrics::score(test_clips[t], nearest), 0, 0);
        add_stats(rn, test_stats, nearest);
        csv.add(rn);
        Row rb = score_row("b", effect, cl, effect + "|blend2", "flipbook_library", "blend of 2 nearest BC3", lib_bytes, metrics::score(test_clips[t], blend), 0, 0);
        add_stats(rb, test_stats, blend);
        csv.add(rb);
        // How close is any training clip? (the oracle library pick, an upper bound for a library)
        double best = -1;
        for (const Clip& tc : train_clips) best = std::max(best, metrics::active_psnr(test_clips[t], tc));
        Row r = score_row("b", effect, cl, effect + "|oracle", "flipbook_library", "best of 45 raw (oracle)", lib_bytes * 4, metrics::score(test_clips[t], test_clips[t]), 0, 0);
        r.v["active_psnr"] = std::format("{:.4f}", best);
        r.v["psnr"] = "";
        r.v["ssim"] = "";
        csv.add(r);
      }
    }
    struct BConfig {
      std::string name;
      int bases, hidden, layers, grid;
    };
    for (const BConfig& bc : {BConfig{"grid_k8", 8, 32, 2, 32}, BConfig{"grid_k16", 16, 48, 2, 32}}) {
      const std::string key = effect + "|" + bc.name;
      if (csv.has("method", key)) continue;
      Hyper h;
      h.arch = Arch::grid;
      h.size = kSize;
      h.frames = kFrames;
      h.loop = sim::effect_loops(e);
      h.n_controls = 3;
      h.bases = bc.bases;
      h.grid = bc.grid;
      h.channels = 8;
      h.hidden = bc.hidden;
      h.layers = bc.layers;
      h.grid_t = 16;
      std::vector<train::Example> data;
      for (std::size_t k = 0; k < train_clips.size(); ++k) data.push_back({&train_clips[k], {train_s[k][0], train_s[k][1], train_s[k][2]}});
      train::Options o;
      o.iterations = c.iters(12000);
      o.threads = c.threads;
      o.log_every = 0;
      auto r = train::train(h, data, o);
      Model m = r.model;
      m.effect = effect;
      m.control_names = {"intensity", "wind", "turbulence"};
      m.feature_bits = 8;
      quantise_like_storage(m);
      m.pack_features();
      save_model(c.data / "models" / "b" / std::format("{}_{}.nvfx", effect, bc.name), m);
      const double ms = runtime_ms(m);
      for (std::size_t t = 0; t < test_s.size(); ++t) {
        const Clip out = runtime_clip(m, test_s[t], -1, 0);
        Row row = score_row("b", effect, std::format("{}_test{}", effect, t), key, "neural_grid", h.describe() + " 8-bit", m.storage_bytes(),
                            metrics::score(test_clips[t], out), ms, r.seconds);
        add_stats(row, metrics::stats(test_clips[t]), out);
        csv.add(row);
      }
      // Training-setting reconstruction (how well it fits what it saw)
      double fit = 0;
      for (std::size_t k = 0; k < train_clips.size(); k += 4) fit += metrics::active_psnr(train_clips[k], runtime_clip(m, train_s[k], -1, 0));
      std::println("B {} {}: {:.1f} s, train-setting active PSNR {:.2f}", effect, bc.name, r.seconds, fit / std::ceil(static_cast<double>(train_clips.size()) / 4.0));
      Row fr = score_row("b_fit", effect, effect + "_train", key + "|fit", "neural_grid", h.describe(), m.storage_bytes(), metrics::ClipScores{}, ms, r.seconds);
      fr.v["active_psnr"] = std::format("{:.4f}", fit / std::ceil(static_cast<double>(train_clips.size()) / 4.0));
      csv.add(fr);
    }
  }
}

// --- Task C -----------------------------------------------------------------------------------------------------

const std::vector<std::string> kCCols = {"effect", "config", "kind", "a", "b", "coverage_l1", "emission_l1", "spectrum_l1",
                                         "mean_frame_psnr", "motion_ratio", "active_psnr"};

void step_c(const Ctx& c) {
  Csv csv(c.results / "c_stats.csv", kCCols);
  struct CConfig {
    std::string name;
    int bases;
  };
  // k8: eight shared feature volumes (about 1 MB at 8 bits); k24: one volume per training seed (about 3 MB), which can
  // reproduce every training variation and morph between them.
  const CConfig configs[] = {{"k8", 8}, {"k24", 24}};
  for (const auto e : sim::kEffects) {
    const std::string effect = ename(e);
    std::vector<Clip> train_clips, test_clips;
    for (int k = 0; k < kCTrain + kCTest; ++k) {
      Clip cl = get_clip(c, "c", std::format("{}_seed{}", effect, 1000 + k),
                         params(e, kCControls[0], kCControls[1], kCControls[2], 1000 + static_cast<std::uint64_t>(k)));
      (k < kCTrain ? train_clips : test_clips).push_back(std::move(cl));
    }
    std::map<const Clip*, metrics::ClipStats> stats_cache;  // each clip's statistics computed once
    const auto stats_of = [&](const Clip& cl) -> const metrics::ClipStats& {
      auto it = stats_cache.find(&cl);
      if (it == stats_cache.end()) it = stats_cache.emplace(&cl, metrics::stats(cl)).first;
      return it->second;
    };
    const auto add = [&](const std::string& config, std::string kind, std::size_t a, std::size_t b, const Clip& ref, const Clip& test) {
      const auto d = metrics::distance(stats_of(ref), stats_of(test));
      Row row;
      row.v = {{"effect", effect}, {"config", config}, {"kind", kind}, {"a", std::to_string(a)}, {"b", std::to_string(b)},
               {"coverage_l1", std::format("{:.5f}", d.coverage_l1)}, {"emission_l1", std::format("{:.5f}", d.emission_l1)},
               {"spectrum_l1", std::format("{:.5f}", d.spectrum_l1)}, {"mean_frame_psnr", std::format("{:.4f}", d.mean_frame_psnr)},
               {"motion_ratio", std::format("{:.4f}", d.motion_ratio)}, {"active_psnr", std::format("{:.4f}", metrics::active_psnr(ref, test))}};
      csv.add(row);
    };
    const auto nearest_training = [&](const Clip& x) {
      double best = -1;
      std::size_t nk = 0;
      for (std::size_t k = 0; k < train_clips.size(); ++k) {
        const double p = metrics::active_psnr(train_clips[k], x);
        if (p > best) {
          best = p;
          nk = k;
        }
      }
      return nk;
    };
    // The natural spread between real seeds, once per effect.
    if (!csv.has("config", effect + "|real")) {
      for (std::size_t a = 0; a < test_clips.size(); ++a) {
        for (std::size_t b = 0; b < train_clips.size(); b += 3) add(effect + "|real", "training_vs_heldout", a, b, test_clips[a], train_clips[b]);
        for (std::size_t b = a + 1; b < test_clips.size(); ++b) add(effect + "|real", "heldout_vs_heldout", a, b, test_clips[a], test_clips[b]);
        const std::size_t nk = nearest_training(test_clips[a]);
        add(effect + "|real", "heldout_nearest_training", a, nk, train_clips[nk], test_clips[a]);
      }
    }
    for (const CConfig& cc : configs) {
      const std::string key = effect + "|" + cc.name;
      if (csv.has("config", key)) continue;
      Hyper h;
      h.arch = Arch::grid;
      h.size = kSize;
      h.frames = kFrames;
      h.loop = sim::effect_loops(e);
      h.n_latent = 8;
      h.bases = cc.bases;
      h.grid = 32;
      h.channels = 8;
      h.hidden = 32;
      h.layers = 2;
      h.grid_t = 16;
      std::vector<train::Example> data;
      for (const Clip& cl : train_clips) data.push_back({&cl, {}});
      train::Options o;
      o.iterations = c.iters(12000);
      o.threads = c.threads;
      o.log_every = 0;
      auto r = train::train(h, data, o);
      Model m = r.model;
      m.effect = effect;
      m.feature_bits = 8;
      quantise_like_storage(m);
      m.pack_features();
      save_model(c.data / "models" / "c" / std::format("{}_variation_{}.nvfx", effect, cc.name), m);
      std::vector<Clip> gen, recon;
      for (int s = 0; s < kCTest; ++s) gen.push_back(runtime_clip(m, {}, -1, 5000 + static_cast<std::uint64_t>(s)));
      for (int k = 0; k < kCTrain; ++k) recon.push_back(runtime_clip(m, {}, k, 0));
      for (std::size_t a = 0; a < test_clips.size(); ++a) {
        for (std::size_t b = 0; b < gen.size(); ++b) add(key, "generated_vs_heldout", a, b, test_clips[a], gen[b]);
      }
      for (std::size_t a = 0; a < gen.size(); ++a) {
        for (std::size_t b = a + 1; b < gen.size(); ++b) add(key, "generated_vs_generated", a, b, gen[a], gen[b]);
        const std::size_t nk = nearest_training(gen[a]);  // copying check
        add(key, "generated_nearest_training", a, nk, train_clips[nk], gen[a]);
      }
      for (std::size_t k = 0; k < train_clips.size(); ++k) add(key, "reconstruction", k, k, train_clips[k], recon[k]);
      Row info;
      info.v = {{"effect", effect}, {"config", key}, {"kind", "model"}, {"a", std::to_string(m.storage_bytes())}, {"b", std::format("{:.1f}", r.seconds)}};
      csv.add(info);
      std::println("C {} {}: trained {:.1f} s ({} KB)", effect, cc.name, r.seconds, m.storage_bytes() / 1024);
    }
  }
}

// --- timing -----------------------------------------------------------------------------------------------------

// Median and 90th percentile ms per frame through the runtime on one pinned core, for a size and an ISA.
std::pair<double, double> measure(const fs::path& model, int size, nvfx_isa isa, double& macs, nvfx_effect_info& info) {
  cpu_set_t old, one;
  sched_getaffinity(0, sizeof(old), &old);
  CPU_ZERO(&one);
  CPU_SET(3, &one);
  sched_setaffinity(0, sizeof(one), &one);
  nvfx_effect* e = nullptr;
  if (nvfx_effect_load(model.c_str(), &e) != NVFX_OK) throw std::runtime_error("cannot load " + model.string());
  nvfx_effect_get_info(e, &info);
  nvfx_set_isa(isa);
  nvfx_instance* in = nullptr;
  if (nvfx_instance_create(e, size, &in) != NVFX_OK) {
    nvfx_effect_free(e);
    nvfx_set_isa(NVFX_ISA_AUTO);
    sched_setaffinity(0, sizeof(old), &old);
    return {-1, -1};
  }
  macs = nvfx_instance_macs_per_pixel(in);
  std::vector<std::uint8_t> buf(static_cast<std::size_t>(size) * size * 4);
  std::vector<double> ms;
  for (int f = 0; f < 230; ++f) {
    const auto t0 = std::chrono::steady_clock::now();
    nvfx_render(in, f / 30.0, buf.data(), static_cast<std::size_t>(size) * 4);
    if (f >= 30) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  nvfx_instance_free(in);
  nvfx_effect_free(e);
  nvfx_set_isa(NVFX_ISA_AUTO);
  sched_setaffinity(0, sizeof(old), &old);
  std::ranges::sort(ms);
  return {ms[ms.size() / 2], ms[ms.size() * 9 / 10]};
}

// Re-measure every saved configuration on a quiet machine (run nothing else meanwhile): sizes, ISAs.
void step_timing(const Ctx& c) {
  const fs::path out = c.results / "timing.csv";
  fs::remove(out);
  Csv csv(out, {"model", "config", "isa", "size", "median_ms", "p90_ms", "macs_px", "stored_kb", "resident_kb"});
  std::vector<std::pair<std::string, fs::path>> models;
  for (const auto& e : fs::directory_iterator(c.data / "models" / "a")) {
    const std::string n = e.path().stem().string();
    if (n.starts_with("fire_0_")) models.emplace_back(n.substr(7), e.path());
  }
  for (const std::string group : {"b", "c"}) {
    if (!fs::exists(c.data / "models" / group)) continue;
    for (const auto& e : fs::directory_iterator(c.data / "models" / group)) {
      if (e.path().stem().string().starts_with("fire_")) models.emplace_back(group + ":" + e.path().stem().string(), e.path());
    }
  }
  std::ranges::sort(models);
  for (const auto& [name, path] : models) {
    for (const nvfx_isa isa : {NVFX_ISA_AVX2, NVFX_ISA_AVX512, NVFX_ISA_BASELINE}) {
      if (nvfx_set_isa(isa) != NVFX_OK) continue;
      nvfx_set_isa(NVFX_ISA_AUTO);
      for (const int size : {32, 64, 128, 256}) {
        if (isa != NVFX_ISA_AVX2 && size != 128) continue;
        double macs = 0;
        nvfx_effect_info info{};
        const auto [med, p90] = measure(path, size, isa, macs, info);
        if (med < 0) continue;
        Row r;
        r.v = {{"model", name}, {"config", name},
               {"isa", isa == NVFX_ISA_AVX2 ? "avx2" : isa == NVFX_ISA_AVX512 ? "avx512" : "baseline"}, {"size", std::to_string(size)},
               {"median_ms", std::format("{:.4f}", med)}, {"p90_ms", std::format("{:.4f}", p90)}, {"macs_px", std::format("{:.0f}", macs)},
               {"stored_kb", std::format("{:.1f}", static_cast<double>(info.stored_bytes) / 1024.0)},
               {"resident_kb", std::format("{:.1f}", static_cast<double>(info.resident_bytes) / 1024.0)}};
        csv.add(r);
      }
    }
    std::println("timing {} done", name);
  }
  // The simulation's own cost per output frame, for comparison (128 x 128, default solver settings).
  for (const auto e : sim::kEffects) {
    sim::Fluid f(params(e, 0.5f, 0.5f, 0.5f, 1));
    std::vector<std::uint8_t> frame(kSize * kSize * 4);
    std::vector<double> ms;
    for (int i = 0; i < 120; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      f.step_frame();
      f.render(frame);
      if (i >= 20) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::ranges::sort(ms);
    Row r;
    r.v = {{"model", "simulation:" + ename(e)}, {"config", "simulation"}, {"isa", "baseline"}, {"size", "128"},
           {"median_ms", std::format("{:.4f}", ms[ms.size() / 2])}, {"p90_ms", std::format("{:.4f}", ms[ms.size() * 9 / 10])},
           {"macs_px", ""}, {"stored_kb", ""}, {"resident_kb", ""}};
    csv.add(r);
  }
}

// --- figures and videos -----------------------------------------------------------------------------------------

void step_media(const Ctx& c) {
  const fs::path media = c.data / "media";
  const auto bg_of = [](sim::Effect e) { return e == sim::Effect::fire ? Background::black : Background::grey; };
  // A: reference, the neural model (8-bit, about 132 KB) and the two best flipbook layouts at about that memory.
  for (const auto& [name, p] : a_clips()) {
    if (!name.ends_with("_0")) continue;
    const Clip ref = get_clip(c, "a", name, p);
    auto m = load_model(c.data / "models" / "a" / std::format("{}_grid_m8.nvfx", name));
    if (!m) continue;
    const Clip neural = runtime_clip(*m, {}, 0, 0);
    const Clip low_res = flipbook::play(flipbook::build(ref, {32, kSize / 2, flipbook::Codec::bc3, 0}));      // 128 KB
    const Clip few = flipbook::play(flipbook::build(ref, {8, kSize, flipbook::Codec::bc3, kSize / 4}));       // 144 KB
    const Clip* rows[] = {&ref, &neural, &low_res, &few};
    write_png(media / std::format("a_{}_compare.png", name), comparison_sheet(rows, 6, bg_of(p.effect), 1));
    write_comparison_video(media / std::format("a_{}_compare.mp4", name), rows, bg_of(p.effect), 2, 3);
  }
  // B: a held-out control setting: the simulation's truth, the neural model, the nearest training flipbook.
  const auto train_s = b_train_settings();
  const auto test_s = b_test_settings();
  for (const auto e : sim::kEffects) {
    auto m = load_model(c.data / "models" / "b" / std::format("{}_grid_k8.nvfx", ename(e)));
    if (!m) continue;
    for (const std::size_t t : {std::size_t{0}, std::size_t{1}}) {
      const auto& s = test_s[t];
      const Clip truth = get_clip(c, "b", std::format("{}_test_{:.2f}_{:.2f}_{:.2f}", ename(e), s[0], s[1], s[2]), params(e, s[0], s[1], s[2], 1));
      const Clip neural = runtime_clip(*m, s, -1, 0);
      std::size_t best = 0;
      float bd = 1e9f;
      for (std::size_t k = 0; k < train_s.size(); ++k) {
        float d = 0;
        for (std::size_t j = 0; j < 3; ++j) d += (s[j] - train_s[k][j]) * (s[j] - train_s[k][j]);
        if (d < bd) {
          bd = d;
          best = k;
        }
      }
      const auto& n = train_s[best];
      const Clip near = flipbook::play(flipbook::build(
          get_clip(c, "b", std::format("{}_train_{:.2f}_{:.2f}_{:.2f}", ename(e), n[0], n[1], n[2]), params(e, n[0], n[1], n[2], 1)),
          {kFrames, kSize, flipbook::Codec::bc3, 0}));
      const Clip* rows[] = {&truth, &neural, &near};
      write_png(media / std::format("b_{}_test{}.png", ename(e), t), comparison_sheet(rows, 6, bg_of(e), 1));
      write_comparison_video(media / std::format("b_{}_test{}.mp4", ename(e), t), rows, bg_of(e), 2, 3);
    }
  }
  // C: held-out real seeds against generated variations (k8, then k24), and a long drifting run (endless variation).
  for (const auto e : sim::kEffects) {
    auto m8 = load_model(c.data / "models" / "c" / std::format("{}_variation_k8.nvfx", ename(e)));
    auto m = load_model(c.data / "models" / "c" / std::format("{}_variation_k24.nvfx", ename(e)));
    if (!m || !m8) continue;
    std::vector<Clip> clips;
    for (int k = kCTrain; k < kCTrain + 3; ++k) {
      clips.push_back(get_clip(c, "c", std::format("{}_seed{}", ename(e), 1000 + k), params(e, kCControls[0], kCControls[1], kCControls[2], 1000 + static_cast<std::uint64_t>(k))));
    }
    for (int s = 0; s < 3; ++s) clips.push_back(runtime_clip(*m8, {}, -1, 5000 + static_cast<std::uint64_t>(s)));
    for (int s = 0; s < 3; ++s) clips.push_back(runtime_clip(*m, {}, -1, 5000 + static_cast<std::uint64_t>(s)));
    std::vector<const Clip*> rows;
    for (const Clip& cl : clips) rows.push_back(&cl);
    write_png(media / std::format("c_{}_variations.png", ename(e)), comparison_sheet(rows, 6, bg_of(e), 1));
    if (e == sim::Effect::explosion) continue;  // drifting only makes sense for looping effects
    RtEffect fx(*m);
    nvfx_instance* in = nullptr;
    nvfx_instance_create(fx.e, kSize, &in);
    nvfx_instance_set_seed(in, 77);
    nvfx_instance_set_drift(in, 3.f);
    Clip drift;
    drift.allocate(kSize, 30 * 20);  // 20 seconds
    drift.fps = 30;
    for (int f = 0; f < drift.frames; ++f) nvfx_render(in, f / 30.0, drift.frame(f).data(), kSize * 4);
    nvfx_instance_free(in);
    const Clip* one[] = {&drift};
    write_comparison_video(media / std::format("c_{}_drift_20s.mp4", ename(e)), one, bg_of(e), 2, 1);
  }
  std::println("media written to {}", media.string());
}

// --- report -----------------------------------------------------------------------------------------------------

std::string fmt_iv(const metrics::Interval& iv, int prec = 2) {
  return std::format("{:+.{}f} [{:+.{}f}, {:+.{}f}]{}", iv.mean, prec, iv.lo, prec, iv.hi, prec, iv.covers_zero() ? " (tie)" : "");
}

void step_report(const Ctx& c) {
  std::ostringstream md;
  md << "# Experiment summary (generated)\n\nStatus: **generated** by `nvfx_experiment report` from the CSVs in this folder; rerun it rather "
        "than editing. 95% paired bootstrap intervals in square brackets (10,000 resamples over clips); an interval "
        "covering zero is reported as a tie. PSNR in dB; \"active\" = PSNR over pixels visible in either clip.\n\n";
  // Quiet-machine timings (the timing step), when present: model -> median ms at 128 px with AVX2.
  std::map<std::string, double> quiet_ms;
  const bool have_timing = fs::exists(c.results / "timing.csv");
  if (have_timing) {
    Csv t(c.results / "timing.csv", {"model", "config", "isa", "size", "median_ms", "p90_ms", "macs_px", "stored_kb", "resident_kb"});
    for (const Row& r : t.rows()) {
      if (r.s("isa") == "avx2" && r.s("size") == "128") quiet_ms[r.s("model")] = r.d("median_ms");
    }
  }
  // ---- A
  if (fs::exists(c.results / "a_scores.csv")) {
    Csv a(c.results / "a_scores.csv", kScoreCols);
    std::map<std::string, std::vector<const Row*>> by_cfg;  // config key -> rows (one per clip)
    for (const Row& r : a.rows()) {
      if (r.s("task") != "a") continue;
      const std::string key = r.s("family").starts_with("flipbook") ? r.s("config") : r.s("method").substr(r.s("method").find('|') + 1);
      by_cfg[key].push_back(&r);
    }
    struct Agg {
      std::string key, family;
      double kb, psnr, active, ssim, tpsnr, flicker, ms;
      std::size_t n;
      double spec = 0, mot = 0;
    };
    std::vector<Agg> aggs;
    for (const auto& [k, rows] : by_cfg) {
      Agg g{k, rows[0]->s("family"), 0, 0, 0, 0, 0, 0, 0, rows.size()};
      for (const Row* r : rows) {
        g.kb += r->d("bytes") / 1024.0 / static_cast<double>(rows.size());
        g.psnr += r->d("psnr") / static_cast<double>(rows.size());
        g.active += r->d("active_psnr") / static_cast<double>(rows.size());
        g.ssim += r->d("ssim") / static_cast<double>(rows.size());
        g.tpsnr += r->d("tpsnr") / static_cast<double>(rows.size());
        g.flicker += r->d("flicker") / static_cast<double>(rows.size());
        g.ms += r->d("ms") / static_cast<double>(rows.size());
        if (r->v.contains("spectrum_l1") && !r->s("spectrum_l1").empty()) {
          g.spec += r->d("spectrum_l1") / static_cast<double>(rows.size());
          g.mot += r->d("motion_ratio") / static_cast<double>(rows.size());
        }
      }
      if (const auto bar = k.find('|'); bar != std::string::npos) {
        const std::string tk = k.substr(0, bar) + k.substr(bar + 1);  // "grid_m|8" -> "grid_m8"
        if (quiet_ms.contains(tk)) g.ms = quiet_ms[tk];
      }
      aggs.push_back(g);
    }
    std::ranges::sort(aggs, {}, &Agg::kb);
    md << "## A. Compression: one model per clip against flipbooks of the same clip\n\n";
    md << std::format("Means over {} clips (fire, smoke, explosion; 128 x 128, 64 frames). ms = median per frame through the runtime on one AVX2 core{}.\n\n",
                      by_cfg.begin()->second.size(), have_timing ? ", measured on a quiet machine (timing step)" : ", measured between training runs");
    md << "Spectrum = mean |log power difference| of the radially averaged luminance spectrum against the reference (0 = same "
          "sharpness; blur raises it); motion = frame-to-frame change relative to the reference (1 = same).\n\n";
    md << "| method | family | KB | PSNR | active PSNR | SSIM | temporal PSNR | flicker | spectrum | motion | ms |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const Agg& g : aggs) {
      md << std::format("| {} | {} | {:.1f} | {:.2f} | {:.2f} | {:.4f} | {:.2f} | {:.2f} | {:.3f} | {:.2f} | {} |\n", g.key, g.family, g.kb, g.psnr, g.active, g.ssim,
                        g.tpsnr, g.flicker, g.spec, g.mot, g.family.starts_with("neural") ? std::format("{:.3f}", g.ms) : "-");
    }
    // Matched-memory comparison: the best neural and the best flipbook configuration (by mean active PSNR) within
    // each budget, paired over clips.
    md << "\n### Matched memory\n\nWithin each budget, the neural configuration and the flipbook configuration with the best mean active PSNR "
          "(chosen on the same clips they are scored on, which favours neither), compared clip by clip.\n\n";
    md << "| budget | neural | KB | flipbook | KB | delta active PSNR | delta PSNR | delta SSIM |\n|---|---|---:|---|---:|---:|---:|---:|\n";
    for (const double budget : {64.0, 128.0, 160.0, 256.0, 320.0, 512.0}) {
      const Agg* bn = nullptr;
      const Agg* bf = nullptr;
      for (const Agg& g : aggs) {
        if (g.kb > budget * 1.02) continue;
        auto& slot = g.family.starts_with("neural") ? bn : bf;
        if (!slot || g.active > slot->active) slot = &g;
      }
      if (!bn || !bf) continue;
      std::vector<double> na, fa, np, fp, ns, fs2;
      for (const Row* r : by_cfg[bn->key]) {
        for (const Row* q : by_cfg[bf->key]) {
          if (q->s("clip") != r->s("clip")) continue;
          na.push_back(r->d("active_psnr"));
          fa.push_back(q->d("active_psnr"));
          np.push_back(r->d("psnr"));
          fp.push_back(q->d("psnr"));
          ns.push_back(r->d("ssim"));
          fs2.push_back(q->d("ssim"));
        }
      }
      md << std::format("| {:.0f} KB | {} | {:.0f} | {} | {:.0f} | {} | {} | {} |\n", budget, bn->key, bn->kb, bf->key, bf->kb,
                        fmt_iv(metrics::paired_bootstrap(na, fa)), fmt_iv(metrics::paired_bootstrap(np, fp)), fmt_iv(metrics::paired_bootstrap(ns, fs2), 4));
    }
    // Memory ratio at equal quality: flipbook envelope (best active PSNR at or below each size), log-interpolated.
    std::vector<std::pair<double, double>> env;  // (kb, best active psnr at <= kb)
    for (const Agg& g : aggs) {
      if (!g.family.starts_with("flipbook")) continue;
      const double best = env.empty() ? g.active : std::max(env.back().second, g.active);
      env.emplace_back(g.kb, best);
    }
    md << "\n### Memory at equal quality\n\nFor each neural configuration: the flipbook memory needed for the same mean active PSNR "
          "(log-linear interpolation along the best-flipbook-at-each-size envelope; \">\" when no flipbook up to 1 MB reaches it).\n\n";
    md << "| neural | KB | active PSNR | flipbook KB for equal quality | ratio |\n|---|---:|---:|---:|---:|\n";
    for (const Agg& g : aggs) {
      if (!g.family.starts_with("neural")) continue;
      std::string fkb = std::format("> {:.0f}", env.back().first), ratio = std::format("> {:.1f}x", env.back().first / g.kb);
      for (std::size_t i = 1; i < env.size(); ++i) {
        if (env[i].second >= g.active && env[i - 1].second < g.active) {
          const double u = (g.active - env[i - 1].second) / (env[i].second - env[i - 1].second);
          const double kb = std::exp(std::log(env[i - 1].first) + u * (std::log(env[i].first) - std::log(env[i - 1].first)));
          fkb = std::format("{:.0f}", kb);
          ratio = std::format("{:.1f}x", kb / g.kb);
          break;
        }
        if (i == 1 && env[0].second >= g.active) {
          fkb = std::format("< {:.0f}", env[0].first);
          ratio = std::format("< {:.1f}x", env[0].first / g.kb);
          break;
        }
      }
      md << std::format("| {} | {:.0f} | {:.2f} | {} | {} |\n", g.key, g.kb, g.active, fkb, ratio);
    }
    // per effect for the headline configs
    md << "\n### By effect (grid_m 8-bit against BC3 16 frames 128 px and BC3 all frames)\n\n| effect | grid_m 8-bit active | bc3 16f 128px active | bc3 64f 128px active |\n|---|---:|---:|---:|\n";
    for (const auto e : sim::kEffects) {
      const auto mean_of = [&](const std::string& key) {
        double s = 0;
        int n = 0;
        for (const Row* r : by_cfg[key]) {
          if (r->s("effect") == ename(e)) {
            s += r->d("active_psnr");
            ++n;
          }
        }
        return n ? s / n : 0.0;
      };
      md << std::format("| {} | {:.2f} | {:.2f} | {:.2f} |\n", ename(e), mean_of("grid_m|8"), mean_of("bc3 16f 128px"), mean_of("bc3 64f 128px"));
    }
    // interpolation
    std::map<std::string, std::vector<double>> interp;
    std::map<std::string, double> interp_kb;
    std::map<std::string, std::vector<std::string>> interp_clip;
    for (const Row& r : a.rows()) {
      if (r.s("task") != "a_interp") continue;
      const std::string k = r.s("method").substr(r.s("method").rfind('|') + 1);
      interp[k].push_back(r.d("active_psnr"));
      interp_kb[k] = r.d("bytes") / 1024.0;
    }
    if (!interp.empty()) {
      md << "\n### Frame interpolation (only even frames available; odd frames scored)\n\n| method | KB | mean active PSNR on odd frames | method minus neural |\n|---|---:|---:|---:|\n";
      const auto& nv = interp["grid_m"];
      for (const auto& [k, v] : interp) {
        const double mean = std::ranges::fold_left(v, 0.0, std::plus{}) / static_cast<double>(v.size());
        std::string label = k == "grid_m" ? std::string("grid_m 8-bit (trained on the even frames)") : k;
        if (const auto at = label.find(" 0f "); at != std::string::npos) label.replace(at, 4, " even frames, ");  // explicit keep list
        md << std::format("| {} | {:.0f} | {:.2f} | {} |\n", label, interp_kb[k], mean, k == "grid_m" || v.size() != nv.size() ? "-" : fmt_iv(metrics::paired_bootstrap(v, nv)));
      }
    }
  }
  // ---- B
  if (fs::exists(c.results / "b_scores.csv")) {
    Csv b(c.results / "b_scores.csv", kScoreCols);
    md << "\n## B. Controls: held-out control settings\n\nOne model per effect trained on 45 settings (3 intensity x 5 wind x 3 turbulence, seed 1); "
          "scored on 10 off-grid settings per effect. Baselines are libraries of 45 BC3 flipbooks (one per training setting).\n\n";
    std::map<std::string, std::map<std::string, std::pair<double, std::string>>> v;  // method -> clip -> (active, effect)
    std::map<std::string, std::map<std::string, double>> vs, vp, vspec, vmot;
    std::map<std::string, double> kb, ms;
    for (const Row& r : b.rows()) {
      if (r.s("task") != "b") continue;
      const std::string k = r.s("method").substr(r.s("method").find('|') + 1);
      v[k][r.s("clip")] = {r.d("active_psnr"), r.s("effect")};
      if (!r.s("ssim").empty()) vs[k][r.s("clip")] = r.d("ssim");
      if (!r.s("psnr").empty()) vp[k][r.s("clip")] = r.d("psnr");
      if (r.v.contains("spectrum_l1") && !r.s("spectrum_l1").empty()) {
        vspec[k][r.s("clip")] = r.d("spectrum_l1");
        vmot[k][r.s("clip")] = r.d("motion_ratio");
      }
      kb[k] = r.d("bytes") / 1024.0;  // per effect (one model or one library per effect)
      ms[k] = r.d("ms");
    }
    md << "| method | KB per effect | mean active PSNR | mean PSNR | mean SSIM | spectrum | motion | ms |\n|---|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& [k, clips] : v) {
      double sa = 0, sp = 0, ss = 0, sx = 0, sm = 0;
      for (const auto& [cl, val] : clips) {
        sa += val.first;
        if (vp[k].contains(cl)) sp += vp[k][cl];
        if (vs[k].contains(cl)) ss += vs[k][cl];
        if (vspec[k].contains(cl)) {
          sx += vspec[k][cl];
          sm += vmot[k][cl];
        }
      }
      const double n = static_cast<double>(clips.size());
      md << std::format("| {} | {:.0f} | {:.2f} | {} | {} | {} | {} | {} |\n", k, kb[k], sa / n, vp[k].empty() ? "-" : std::format("{:.2f}", sp / n),
                        vs[k].empty() ? "-" : std::format("{:.4f}", ss / n), vspec[k].empty() ? "-" : std::format("{:.3f}", sx / n),
                        vspec[k].empty() ? "-" : std::format("{:.2f}", sm / n), ms[k] > 0 ? std::format("{:.3f}", ms[k]) : "-");
    }
    md << "\n| comparison (active PSNR, paired over 30 held-out settings) | difference |\n|---|---:|\n";
    for (const std::string nk : {"grid_k8", "grid_k16"}) {
      for (const std::string fk : {"nearest", "blend2", "oracle"}) {
        if (!v.contains(nk) || !v.contains(fk)) continue;
        std::vector<double> x, y;
        for (const auto& [cl, val] : v[nk]) {
          if (v[fk].contains(cl)) {
            x.push_back(val.first);
            y.push_back(v[fk][cl].first);
          }
        }
        md << std::format("| {} - {} | {} |\n", nk, fk, fmt_iv(metrics::paired_bootstrap(x, y)));
      }
    }
    md << "\n| effect | grid_k8 | grid_k16 | nearest | blend2 | oracle |\n|---|---:|---:|---:|---:|---:|\n";
    for (const auto e : sim::kEffects) {
      md << "| " << ename(e);
      for (const std::string k : {"grid_k8", "grid_k16", "nearest", "blend2", "oracle"}) {
        double s = 0;
        int n = 0;
        for (const auto& [cl, val] : v[k]) {
          if (val.second == ename(e)) {
            s += val.first;
            ++n;
          }
        }
        md << std::format(" | {}", n ? std::format("{:.2f}", s / n) : "-");
      }
      md << " |\n";
    }
    for (const Row& r : b.rows()) {
      if (r.s("task") == "b_fit") md << std::format("\nTraining-setting fit ({}): mean active PSNR {:.2f} on every 4th training setting.\n", r.s("method"), r.d("active_psnr"));
    }
  }
  // ---- C
  if (fs::exists(c.results / "c_stats.csv")) {
    Csv cs(c.results / "c_stats.csv", kCCols);
    md << "\n## C. Variation: new seeds against held-out real clips\n\nPer effect: models with 8-dimensional variation codes trained on 24 "
          "simulated seeds at fixed controls; k8 shares 8 feature volumes, k24 has one per training seed. 8 new seeds are generated through the "
          "runtime (each a random point between two training codes) and compared with 8 held-out simulated seeds. Distances are means over "
          "pairs; smaller is closer. The \"real\" rows are the natural spread between simulated seeds: a generator should match them, not beat "
          "them. Spectrum = |log power difference| (blur raises it); motion = frame-to-frame change relative to the first clip of the pair; "
          "reconstruction = a training seed replayed from its own code.\n\n";
    md << "| effect | model | pairs | coverage L1 | spectrum L1 | mean-frame PSNR | motion ratio | active PSNR |\n|---|---|---|---:|---:|---:|---:|---:|\n";
    for (const auto e : sim::kEffects) {
      for (const std::string cfg : {"real", "k8", "k24"}) {
        const std::string key = ename(e) + "|" + cfg;
        for (const std::string kind : {"heldout_vs_heldout", "training_vs_heldout", "heldout_nearest_training", "generated_vs_heldout",
                                       "generated_vs_generated", "generated_nearest_training", "reconstruction"}) {
          double cov = 0, spe = 0, mfp = 0, mot = 0, act = 0;
          int n = 0;
          for (const Row& r : cs.rows()) {
            if (r.s("config") != key || r.s("kind") != kind) continue;
            cov += r.d("coverage_l1");
            spe += r.d("spectrum_l1");
            mfp += r.d("mean_frame_psnr");
            mot += r.d("motion_ratio");
            act += r.d("active_psnr");
            ++n;
          }
          if (!n) continue;
          md << std::format("| {} | {} | {} | {:.4f} | {:.3f} | {:.2f} | {:.2f} | {:.2f} |\n", ename(e), cfg, kind, cov / n, spe / n, mfp / n, mot / n, act / n);
        }
      }
    }
    md << "\n| effect | model | KB stored | training s |\n|---|---|---:|---:|\n";
    for (const Row& r : cs.rows()) {
      if (r.s("kind") == "model") md << std::format("| {} | {} | {:.0f} | {} |\n", r.s("effect"), r.s("config"), r.d("a") / 1024.0, r.s("b"));
    }
  }
  // ---- timing
  if (have_timing) {
    Csv t(c.results / "timing.csv", {"model", "config", "isa", "size", "median_ms", "p90_ms", "macs_px", "stored_kb", "resident_kb"});
    md << "\n## Runtime cost\n\nMedian (90th percentile) ms per frame through `nvfx_render`, one pinned core, nothing else running. Models "
          "trained on the first fire clip (A), the fire control model (B) and the fire variation model (C). The simulation row is the "
          "solver plus its renderer for one 128 x 128 output frame.\n\n";
    std::map<std::string, std::map<std::string, std::string>> cell;  // model -> column -> text
    std::map<std::string, std::string> kb;
    std::vector<std::string> order;
    for (const Row& r : t.rows()) {
      const std::string m = r.s("model");
      if (!cell.contains(m)) order.push_back(m);
      const std::string col = r.s("isa") == "avx2" ? r.s("size") + " px" : r.s("isa") + " 128";
      cell[m][col] = std::format("{:.3f} ({:.3f})", r.d("median_ms"), r.d("p90_ms"));
      if (!r.s("stored_kb").empty()) kb[m] = std::format("{} / {}", r.s("stored_kb"), r.s("resident_kb"));
      if (!r.s("macs_px").empty() && r.s("size") == "128") cell[m]["MAC/px"] = r.s("macs_px");
    }
    const std::vector<std::string> cols = {"32 px", "64 px", "128 px", "256 px", "avx512 128", "baseline 128", "MAC/px"};
    md << "| model | KB stored / resident |";
    for (const auto& col : cols) md << " " << col << " |";
    md << "\n|---|---:|";
    for (std::size_t i = 0; i < cols.size(); ++i) md << "---:|";
    md << "\n";
    for (const auto& m : order) {
      md << "| " << m << " | " << (kb.contains(m) ? kb[m] : "-") << " |";
      for (const auto& col : cols) md << " " << (cell[m].contains(col) ? cell[m][col] : "-") << " |";
      md << "\n";
    }
  }
  {
    study_d::Ctx d;
    d.data = c.data;
    d.results = c.results;
    d.threads = c.threads;
    d.quick = c.quick;
    study_d::report(d, md);
  }
  {
    study_g::Ctx g;  // the report reads only the CSVs in results
    g.results = c.results;
    g.quick = c.quick;
    study_g::report(g, md);
  }
  fs::create_directories(c.results);
  std::ofstream(c.results / "SUMMARY.md") << md.str();
  std::println("wrote {}", (c.results / "SUMMARY.md").string());
}

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"quick", "help"});
  if (a.flag("help") || a.positional().empty()) {
    std::println("nvfx_experiment data|a|b|c|media|timing|report|all|d|d-chaos|d-train|d-tune|d-finish|d-eval|d-timing|g-data|g-pilot|g-search|g-eval|g-timing|g-fine|g-diff|g-diff-test [--root DIR] [--results DIR] [--threads 4] [--quick]");
    return 0;
  }
  Ctx c;
  c.data = a.has("root") ? fs::path(a.str("root")) : data_root() / "experiments";
  c.results = a.str("results", "results/experiments");
  c.threads = a.i("threads", 4);
  c.quick = a.flag("quick");
  if (c.quick) {
    c.data /= "quick";
    c.results /= "quick";
  }
  const std::string step = a.positional()[0];
  const auto t0 = std::chrono::steady_clock::now();
  if (step == "data" || step == "all") step_data(c);
  if (step == "a" || step == "all") step_a(c);
  if (step == "b" || step == "all") step_b(c);
  if (step == "c" || step == "all") step_c(c);
  if (step == "media" || step == "all") step_media(c);
  if (step == "timing") step_timing(c);  // separately, on a quiet machine
  {
    study_d::Ctx d;
    d.data = c.data;
    d.results = c.results;
    d.threads = c.threads;
    d.quick = c.quick;
    d.effects = a.str("effects");
    if (step == "d-chaos" || step == "d") study_d::step_chaos(d);
    if (step == "d-train" || step == "d") study_d::step_train(d);
    if (step == "d-tune") study_d::step_tune(d);
    if (step == "d-finish") study_d::step_finish(d);
    if (step == "d-eval" || step == "d") study_d::step_eval(d);
    if (step == "d-timing") study_d::step_timing(d);  // separately, on a quiet machine
  }
  {  // study G (docs/DCM.md): diffusion-context mixing
    study_g::Ctx g;  // its data under the data root's g/ (docs/DCM.md §4), or --root DIR/g
    g.data = (a.has("root") ? fs::path(a.str("root")) : data_root()) / "g";
    if (c.quick) g.data /= "quick";
    g.results = c.results;
    g.threads = c.threads;
    g.quick = c.quick;
    g.effects = a.str("effects");
    if (step == "g-data") study_g::step_data(g);
    if (step == "g-pilot") study_g::step_pilot(g);
    if (step == "g-search") study_g::step_search(g);
    if (step == "g-eval") study_g::step_eval(g);
    if (step == "g-timing") study_g::step_timing(g);  // separately, on a quiet machine
    if (step == "g-fine") study_g::step_fine(g);      // design G1 end to end (docs/DCM.md)
    if (step == "g-diff") study_g::step_diff(g);       // stage S5 (needs nvfx_dcm ddpm-train first)
    if (step == "g-diff-test") study_g::step_diff_test(g);
  }
  if (step == "report" || step == "all") step_report(c);
  std::println("{} finished in {:.1f} min", step, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 60.0);
  a.warn_unused();
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_experiment: {}", e.what());
  return 2;
}
