// nvfx_f2: study F2, compression of effects pushed as far as it honestly goes (results/compression/README.md, F2).
//
//   nvfx_f2 data                                    validation clips (2 per effect, settings and seeds not in study A)
//   nvfx_f2 flipbooks --set val|test                the study A flipbook ladder on every clip: scores, memory, packed bytes
//   nvfx_f2 train --set val|test --configs A,B,...  train, save, score through the runtime, pack (lossless coder)
//   nvfx_f2 rescore --set test --name N --pattern P score and pack existing models ("{clip}" in P is the clip name)
//   nvfx_f2 video --set val|test [--codecs x264,...] the video codecs' quality ladders on every clip
//   nvfx_f2 report --set val|test [--configs ...]   equal-quality ratios against flipbooks and each codec, memory and
//                                                   disk, with 95% bootstrap intervals over clips
//   nvfx_f2 timing --models a.nvfx,b.nvfx [--core 3] [--reps 5]
//                                                   thread CPU time per 128 x 128 frame, least of the repetitions
//   options: --root DIR (data root, default $NEURALVFX_DATA), --out DIR (results/compression), --threads 2,
//            --clips a,b (a subset of the set)
//
// Sets: "test" is study A's 12 clips (docs/REPORT.md §3), scored exactly as study A scores them (the network trained
// on the clip, rendered through the runtime at its stored precision, active-region PSNR over all 64 frames). "val"
// holds 6 other clips for every choice. Every step appends rows to its CSV as it goes and skips rows already there,
// so a stopped run continues where it stopped.
//
// Configurations are written as fields joined by "_": an architecture, then options.
//   g<G>c<C>h<H>l<L>t<T>     grid family: G x G feature planes, C channels, H hidden units, L hidden layers, T time slices
//   v<c0>.<c1>.<c2>t<T>      conv family at 128 px (latent 16 x 16)
//   b<bits>                  feature storage: 16, or 2 to 8 (default 8)
//   q                        quantisation-aware training at the storage bits (qs<f>: start after a fraction f)
//   t                        trimmed plane ranges (Model::feature_trim): the best-quantising range, tails clipped
//   r<lambda>                rate term in the loss (estimated bits per feature value, at the storage bits)
//   i<iterations>            training steps (default 2000 grid, 1500 conv, as study A)
//   s<seed>                  training seed (default 1)
//   e.g. g32c8h32l2t16_b8 is study A's grid_m at 8 bits; g32c8h32l2t16_b4_q_r3e-5 adds 4-bit QAT and a rate term.
#include "args.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/cm.hpp>
#include <neuralfx/flipbook.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/model.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/sim.hpp>
#include <neuralfx/train.hpp>
#include <neuralfx/video_codec.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <numeric>
#include <print>
#include <random>
#include <ranges>
#include <sched.h>
#include <set>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;
using namespace nfx;

namespace {

constexpr int kSize = 128, kFrames = 64;

struct Ctx {
  fs::path root;  // data root
  fs::path out;   // results/compression
  int threads = 2;
  std::string set = "val";
  std::set<std::string> only;  // clip subset
};

// --- clips --------------------------------------------------------------------------------------------------------

struct ClipRef {
  std::string name, effect;
  fs::path path;
  sim::Params params;
};

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

std::vector<ClipRef> clip_set(const Ctx& c) {
  std::vector<ClipRef> v;
  for (const auto e : sim::kEffects) {
    const std::string en(sim::effect_name(e));
    if (c.set == "test") {  // study A's clips (tools/nvfx_experiment.cpp, a_clips())
      const std::array<std::array<float, 3>, 4> s = {{{0.5f, 0.5f, 0.5f}, {0.9f, 0.3f, 0.7f}, {0.2f, 0.7f, 0.2f}, {0.7f, 0.6f, 1.0f}}};
      for (std::size_t k = 0; k < s.size(); ++k) {
        const std::string name = std::format("{}_{}", en, k);
        v.push_back({name, en, c.root / "experiments" / "clips" / "a" / (name + ".nfxclip"), params(e, s[k][0], s[k][1], s[k][2], 101 + k)});
      }
    } else if (c.set == "val") {  // other settings and seeds, used for every choice
      const std::array<std::array<float, 3>, 2> s = {{{0.4f, 0.4f, 0.6f}, {0.8f, 0.65f, 0.35f}}};
      for (std::size_t k = 0; k < s.size(); ++k) {
        const std::string name = std::format("{}_v{}", en, k);
        v.push_back({name, en, c.root / "f2" / "clips" / "val" / (name + ".nfxclip"), params(e, s[k][0], s[k][1], s[k][2], 201 + k)});
      }
    } else {
      throw std::invalid_argument("--set must be val or test");
    }
  }
  if (!c.only.empty()) std::erase_if(v, [&](const ClipRef& r) { return !c.only.contains(r.name); });
  return v;
}

Clip load_clip(const ClipRef& r) {
  auto clip = read_clip(r.path);
  if (!clip) throw std::runtime_error(std::format("{} (run `nvfx_f2 data`, or study A's data step for the test set)", clip.error()));
  return *clip;
}

// --- CSV ----------------------------------------------------------------------------------------------------------

class Csv {
 public:
  Csv(fs::path path, std::vector<std::string> cols) : path_(std::move(path)), cols_(std::move(cols)) {
    std::ifstream in(path_);
    std::string line;
    if (!std::getline(in, line)) return;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      std::map<std::string, std::string> r;
      std::size_t k = 0;
      for (const auto part : std::views::split(line, ',')) {
        if (k < cols_.size()) r[cols_[k++]] = std::string(std::string_view(part));
      }
      rows_.push_back(std::move(r));
    }
  }
  bool has(const std::map<std::string, std::string>& key) const {
    return std::ranges::any_of(rows_, [&](const auto& r) {
      return std::ranges::all_of(key, [&](const auto& kv) { return r.contains(kv.first) && r.at(kv.first) == kv.second; });
    });
  }
  void add(std::map<std::string, std::string> r) {
    const bool fresh = !fs::exists(path_);
    fs::create_directories(path_.parent_path());
    std::ofstream o(path_, std::ios::app);
    if (fresh) {
      for (std::size_t i = 0; i < cols_.size(); ++i) o << cols_[i] << (i + 1 < cols_.size() ? "," : "\n");
    }
    for (std::size_t i = 0; i < cols_.size(); ++i) o << (r.contains(cols_[i]) ? r.at(cols_[i]) : "") << (i + 1 < cols_.size() ? "," : "\n");
    rows_.push_back(std::move(r));
  }
  const std::vector<std::map<std::string, std::string>>& rows() const { return rows_; }

 private:
  fs::path path_;
  std::vector<std::string> cols_;
  std::vector<std::map<std::string, std::string>> rows_;
};

std::string f4(double v) { return std::format("{:.4f}", v); }

void put_scores(std::map<std::string, std::string>& r, const metrics::ClipScores& s) {
  r["psnr"] = f4(s.psnr);
  r["active_psnr"] = f4(s.active_psnr);
  r["ssim"] = std::format("{:.5f}", s.ssim);
  r["tpsnr"] = f4(s.temporal_psnr);
  r["flicker"] = f4(s.flicker);
}

// --- data -----------------------------------------------------------------------------------------------------------

void step_data(const Ctx& c0) {
  Ctx c = c0;
  c.set = "val";
  for (const ClipRef& r : clip_set(c)) {
    if (fs::exists(r.path)) continue;
    const Clip clip = sim::simulate(r.params);
    if (auto w = write_clip(r.path, clip); !w) throw std::runtime_error(w.error());
    std::println("data: {}", r.path.string());
  }
}

// --- flipbooks -------------------------------------------------------------------------------------------------------

// The stored form of a flipbook as tensors, exactly as tools/nvfx_pack.cpp codes them.
std::vector<cm::Tensor> flipbook_tensors(const Clip& ref, const flipbook::Flipbook& fb) {
  const flipbook::Spec& spec = fb.spec;
  const auto F = static_cast<std::uint32_t>(fb.kept.size()), R = static_cast<std::uint32_t>(spec.res);
  cm::Tensor img;
  img.shape.width = 1;
  img.shape.channels = true;
  for (const int f : fb.kept) {
    auto small = flipbook::downsample(ref.frame(f), ref.size, spec.res);
    if (spec.codec == flipbook::Codec::bc3) small = flipbook::compress_bc3(small, spec.res, spec.res);
    img.values.insert(img.values.end(), small.begin(), small.end());
  }
  if (spec.codec == flipbook::Codec::bc3) {
    img.shape.kind = cm::Kind::bc3;
    img.shape.dims = {F, R / 4, R / 4, 16};
  } else {
    img.shape.kind = cm::Kind::rgba;
    img.shape.dims = {F, R, R, 4};
  }
  std::vector<cm::Tensor> out{std::move(img)};
  if (spec.flow_res > 0) {
    cm::Tensor mv;
    mv.shape.kind = cm::Kind::flow;
    mv.shape.width = 1;
    mv.shape.channels = true;
    const auto fr = static_cast<std::uint32_t>(spec.flow_res);
    mv.shape.dims = {F, fr, fr, 2};
    for (const auto& field : fb.flow) {
      for (const float v : field) mv.values.push_back(static_cast<std::uint16_t>(std::clamp(std::lround(v * 4.f) + 128, 0L, 255L)));
    }
    out.push_back(std::move(mv));
  }
  return out;
}

const std::vector<std::string> kFlipCols = {"set", "clip", "effect", "config", "family", "memory_bytes", "packed_bytes", "psnr",
                                            "active_psnr", "ssim", "tpsnr", "flicker"};

void step_flipbooks(const Ctx& c) {
  Csv csv(c.out / std::format("f2_flipbooks_{}.csv", c.set), kFlipCols);
  for (const ClipRef& r : clip_set(c)) {
    const Clip ref = load_clip(r);
    for (const auto& spec : flipbook::ladder(kSize, kFrames)) {
      if (csv.has({{"set", c.set}, {"clip", r.name}, {"config", spec.describe()}})) continue;
      const auto fb = flipbook::build(ref, spec);
      const auto tensors = flipbook_tensors(ref, fb);
      std::size_t bytes = 0;
      for (const auto& t : tensors) bytes += t.values.size();
      if (bytes != fb.bytes) throw std::runtime_error("flipbook: stored size does not match");
      const cm::Packed p = cm::pack_tensors(tensors);
      const auto back = cm::unpack_tensors(p.data);
      if (!back || back->size() != tensors.size() || (*back)[0].values != tensors[0].values) throw std::runtime_error("flipbook round trip failed");
      std::map<std::string, std::string> row = {{"set", c.set}, {"clip", r.name}, {"effect", r.effect}, {"config", spec.describe()},
                                                {"family", spec.flow_res > 0 ? "flipbook_mv" : spec.codec == flipbook::Codec::raw ? "flipbook_raw" : "flipbook_bc3"},
                                                {"memory_bytes", std::to_string(fb.bytes)}, {"packed_bytes", std::to_string(p.data.size())}};
      put_scores(row, metrics::score(ref, flipbook::play(fb)));
      csv.add(row);
      std::println("flipbook {} {}: {} -> {} bytes, active {}", r.name, spec.describe(), fb.bytes, p.data.size(), row["active_psnr"]);
    }
  }
}

// --- networks -------------------------------------------------------------------------------------------------------

struct Config {
  std::string name;
  Hyper h;
  int bits = 8;
  bool qat = false;
  bool trim = false;
  float qat_start = 0.f;
  float lambda = 0.f;
  int iters = 0;
  std::uint64_t seed = 1;
};

Config parse_config(const std::string& name) {
  Config cf;
  cf.name = name;
  std::vector<std::string> parts;
  for (const auto p : std::views::split(name, '_')) parts.emplace_back(std::string_view(p));
  if (parts.empty()) throw std::invalid_argument("empty configuration");
  Hyper& h = cf.h;
  h.size = kSize;
  h.frames = kFrames;
  const std::string& a = parts[0];
  int G, C, H, L, T, c0, c1, c2;
  if (std::sscanf(a.c_str(), "g%dc%dh%dl%dt%d", &G, &C, &H, &L, &T) == 5) {
    h.arch = Arch::grid;
    h.grid = G;
    h.channels = C;
    h.hidden = H;
    h.layers = L;
    h.grid_t = T;
    cf.iters = 2000;
  } else if (std::sscanf(a.c_str(), "v%d.%d.%dt%d", &c0, &c1, &c2, &T) == 4) {
    h.arch = Arch::conv;
    h.latent = kSize / 8;
    h.c0 = c0;
    h.c1 = c1;
    h.c2 = c2;
    h.grid_t = T;
    cf.iters = 1500;
  } else {
    throw std::invalid_argument("configuration: unknown architecture " + a);
  }
  for (std::size_t k = 1; k < parts.size(); ++k) {
    const std::string& p = parts[k];
    if (p == "q") cf.qat = true;
    else if (p == "t") cf.trim = true;
    else if (p.starts_with("qs")) {
      cf.qat = true;
      cf.qat_start = std::stof(p.substr(2));
    } else if (p[0] == 'b') cf.bits = std::stoi(p.substr(1));
    else if (p[0] == 'r') cf.lambda = std::stof(p.substr(1));
    else if (p[0] == 'i') cf.iters = std::stoi(p.substr(1));
    else if (p[0] == 's') cf.seed = std::stoull(p.substr(1));
    else throw std::invalid_argument("configuration: unknown option " + p);
  }
  if (!valid_feature_bits(cf.bits)) throw std::invalid_argument("configuration: bits must be 16 or 2 to 8");
  if (cf.qat && cf.bits > 8) throw std::invalid_argument("configuration: QAT needs bits 2 to 8");
  return cf;
}

struct RtEffect {
  nvfx_effect* e = nullptr;
  explicit RtEffect(std::span<const std::uint8_t> bytes) {
    if (nvfx_effect_load_memory(bytes.data(), bytes.size(), &e) != NVFX_OK) throw std::runtime_error("runtime load failed");
  }
  ~RtEffect() { nvfx_effect_free(e); }
  RtEffect(const RtEffect&) = delete;
  RtEffect& operator=(const RtEffect&) = delete;
};

// The clip through the runtime, as study A renders it (training variation 0, no drift).
Clip runtime_clip(const RtEffect& fx, const Model& m, std::size_t& scratch) {
  nvfx_instance* in = nullptr;
  if (nvfx_instance_create(fx.e, kSize, &in) != NVFX_OK) throw std::runtime_error("runtime instance failed");
  nvfx_instance_set_drift(in, 0.f);
  nvfx_instance_set_variation(in, 0);
  scratch = nvfx_instance_scratch_bytes(in);
  Clip clip;
  clip.allocate(kSize, kFrames);
  clip.loop = m.h.loop;
  clip.fps = m.fps;
  for (int f = 0; f < kFrames; ++f) nvfx_render(in, f / static_cast<double>(m.fps), clip.frame(f).data(), kSize * 4);
  nvfx_instance_free(in);
  return clip;
}

const std::vector<std::string> kNetCols = {"set", "clip", "effect", "config", "arch", "bits", "qat", "lambda", "iters", "train_s",
                                           "stored_bytes", "file_bytes", "resident_bytes", "scratch_bytes", "packed_bytes",
                                           "packed_feature_bytes", "feature_values", "est_bits_per_value", "psnr", "active_psnr",
                                           "ssim", "tpsnr", "flicker"};

std::vector<std::uint8_t> read_bytes(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + p.string());
  return {std::istreambuf_iterator<char>(in), {}};
}

// Score a saved model through the runtime and pack it; fills the size and quality columns.
void score_and_pack(const Clip& ref, const fs::path& file, std::map<std::string, std::string>& row) {
  const auto bytes = read_bytes(file);
  auto loaded = load_model(file);
  if (!loaded) throw std::runtime_error(loaded.error());
  RtEffect fx(bytes);
  nvfx_effect_info info{};
  nvfx_effect_get_info(fx.e, &info);
  std::size_t scratch = 0;
  const Clip out = runtime_clip(fx, *loaded, scratch);
  put_scores(row, metrics::score(ref, out));
  const cm::Packed p = cm::pack_model(bytes);
  const auto back = cm::unpack_model(p.data);
  if (!back || *back != bytes) throw std::runtime_error("lossless round trip failed for " + file.string());
  double feat = 0;
  std::size_t values = 0;
  for (const auto& part : p.parts) {
    if (part.kind == cm::Kind::features) {
      feat = part.coded_bytes;
      values = part.values;
    }
  }
  row["stored_bytes"] = std::to_string(loaded->storage_bytes());
  row["file_bytes"] = std::to_string(bytes.size());
  row["resident_bytes"] = std::to_string(info.resident_bytes);
  row["scratch_bytes"] = std::to_string(scratch);
  row["packed_bytes"] = std::to_string(p.data.size());
  row["packed_feature_bytes"] = std::format("{:.1f}", feat);
  row["feature_values"] = std::to_string(values);
  row["est_bits_per_value"] = f4(train::feature_rate(*loaded, loaded->feature_bits < 16 ? loaded->feature_bits : 8) /
                                 static_cast<double>(loaded->features.size()));
}

void step_train(const Ctx& c, const std::vector<std::string>& configs) {
  Csv csv(c.out / std::format("f2_nets_{}.csv", c.set), kNetCols);
  const fs::path models = c.root / "f2" / "models" / c.set;
  for (const std::string& name : configs) {
    const Config cf = parse_config(name);
    for (const ClipRef& r : clip_set(c)) {
      if (csv.has({{"set", c.set}, {"clip", r.name}, {"config", name}})) continue;
      const Clip ref = load_clip(r);
      train::Options o;
      o.iterations = cf.iters;
      o.threads = c.threads;
      o.log_every = 0;
      o.seed = cf.seed;
      if (cf.qat) {
        o.qat_bits = cf.bits;
        o.qat_start = cf.qat_start;
        o.qat_trim = cf.trim;
      }
      o.rate_lambda = cf.lambda;
      o.rate_bits = std::min(cf.bits, 8);
      Hyper h = cf.h;
      h.loop = ref.loop;
      const train::Example ex{&ref, {}};
      auto res = train::train(h, std::span(&ex, 1), o);
      res.model.effect = r.effect;
      res.model.fps = ref.fps;
      res.model.feature_bits = cf.bits;
      res.model.feature_trim = cf.trim;
      const fs::path file = models / std::format("{}__{}.nvfx", r.name, name);
      fs::create_directories(models);
      if (auto w = save_model(file, res.model); !w) throw std::runtime_error(w.error());
      std::map<std::string, std::string> row = {{"set", c.set}, {"clip", r.name}, {"effect", r.effect}, {"config", name},
                                                {"arch", h.arch == Arch::grid ? "grid" : "conv"}, {"bits", std::to_string(cf.bits)},
                                                {"qat", cf.qat ? std::format("{}", cf.qat_start) : "-"}, {"lambda", std::format("{}", cf.lambda)},
                                                {"iters", std::to_string(cf.iters)}, {"train_s", std::format("{:.1f}", res.seconds)}};
      score_and_pack(ref, file, row);
      csv.add(row);
      std::println("train {} {}: {:.0f} s, {} -> {} bytes packed, active {}, est {} bits/value", r.name, name, res.seconds, row["stored_bytes"],
                   row["packed_bytes"], row["active_psnr"], row["est_bits_per_value"]);
    }
  }
}

void step_rescore(const Ctx& c, const std::string& name, const std::string& pattern) {
  Csv csv(c.out / std::format("f2_nets_{}.csv", c.set), kNetCols);
  for (const ClipRef& r : clip_set(c)) {
    if (csv.has({{"set", c.set}, {"clip", r.name}, {"config", name}})) continue;
    std::string p = pattern;
    if (const auto at = p.find("{clip}"); at != std::string::npos) p.replace(at, 6, r.name);
    if (!fs::exists(p)) {
      std::println("rescore: {} missing, skipped", p);
      continue;
    }
    const Clip ref = load_clip(r);
    auto loaded = load_model(p);
    if (!loaded) throw std::runtime_error(loaded.error());
    std::map<std::string, std::string> row = {{"set", c.set}, {"clip", r.name}, {"effect", r.effect}, {"config", name},
                                              {"arch", loaded->h.arch == Arch::grid ? "grid" : "conv"}, {"bits", std::to_string(loaded->feature_bits)},
                                              {"qat", "-"}, {"lambda", "0"}, {"iters", ""}, {"train_s", ""}};
    score_and_pack(ref, p, row);
    csv.add(row);
    std::println("rescore {} {}: {} bytes, {} packed, active {}", r.name, name, row["stored_bytes"], row["packed_bytes"], row["active_psnr"]);
  }
}

// --- video codecs -----------------------------------------------------------------------------------------------------

const std::vector<std::string> kVideoCols = {"set", "clip", "effect", "codec", "q", "payload_bytes", "file_bytes", "psnr", "active_psnr",
                                             "ssim", "tpsnr", "flicker", "encode_cpu_s", "decode_cpu_s", "decode_max_rss_kb"};

void step_video(const Ctx& c, const std::set<std::string>& which) {
  Csv csv(c.out / std::format("f2_video_{}.csv", c.set), kVideoCols);
  const fs::path work = c.root / "f2" / "video_work";
  for (const ClipRef& r : clip_set(c)) {
    const Clip ref = load_clip(r);
    for (const auto& codec : video::codecs()) {
      if (!which.empty() && !which.contains(codec.name)) continue;
      for (const int q : codec.ladder) {
        if (csv.has({{"set", c.set}, {"clip", r.name}, {"codec", codec.name}, {"q", std::to_string(q)}})) continue;
        auto coded = video::code(ref, codec, q, work);
        if (!coded) throw std::runtime_error(coded.error());
        std::map<std::string, std::string> row = {{"set", c.set}, {"clip", r.name}, {"effect", r.effect}, {"codec", codec.name},
                                                  {"q", std::to_string(q)}, {"payload_bytes", std::to_string(coded->payload_bytes)},
                                                  {"file_bytes", std::to_string(coded->file_bytes)},
                                                  {"encode_cpu_s", std::format("{:.3f}", coded->encode_cpu_s)},
                                                  {"decode_cpu_s", std::format("{:.3f}", coded->decode_cpu_s)},
                                                  {"decode_max_rss_kb", std::to_string(coded->decode_max_rss_kb)}};
        put_scores(row, metrics::score(ref, coded->decoded));
        csv.add(row);
        std::println("video {} {} q{}: {} bytes, active {}", r.name, codec.name, q, coded->payload_bytes, row["active_psnr"]);
      }
    }
  }
}

// --- report ---------------------------------------------------------------------------------------------------------

// One method's points on a set: per clip (bytes, quality) for one or more size measures.
struct Point {
  std::vector<double> q;                         // active PSNR per clip (clip order of the set)
  std::map<std::string, std::vector<double>> b;  // size measure -> bytes per clip
};

// A family of configurations (a flipbook ladder, a codec's quality ladder): config -> point.
using Family = std::map<std::string, Point>;

double mean_at(const std::vector<double>& v, const std::vector<std::size_t>& idx) {
  double s = 0;
  for (const std::size_t i : idx) s += v[i];
  return s / static_cast<double>(idx.size());
}

// The best-of-family envelope: best mean quality at or below each mean size, one point per size.
std::vector<std::pair<double, double>> envelope(const Family& fam, const std::string& measure, const std::vector<std::size_t>& idx) {
  std::vector<std::pair<double, double>> pts;
  for (const auto& [k, p] : fam) {
    if (!p.b.contains(measure)) continue;
    pts.emplace_back(mean_at(p.b.at(measure), idx), mean_at(p.q, idx));
  }
  std::ranges::sort(pts);
  std::vector<std::pair<double, double>> env;
  for (const auto& [kb, q] : pts) {
    const double best = env.empty() ? q : std::max(env.back().second, q);
    if (!env.empty() && env.back().first == kb) env.back().second = best;
    else env.emplace_back(kb, best);
  }
  return env;
}

// Size along an envelope for a quality: log-linear between points. censor: -1 below the smallest, +1 above the largest.
double size_for(const std::vector<std::pair<double, double>>& env, double q, int& censor) {
  censor = 0;
  if (env.empty()) return std::nan("");
  if (env.front().second >= q) {
    censor = -1;
    return env.front().first;
  }
  for (std::size_t i = 1; i < env.size(); ++i) {
    if (env[i].second >= q && env[i - 1].second < q) {
      const double u = (q - env[i - 1].second) / (env[i].second - env[i - 1].second);
      return std::exp(std::log(env[i - 1].first) + u * (std::log(env[i].first) - std::log(env[i - 1].first)));
    }
  }
  censor = 1;
  return env.back().first;
}

// Quality along an envelope at a size (the best at or below it, log-linear between points); NaN below the smallest.
double quality_at(const std::vector<std::pair<double, double>>& env, double bytes) {
  if (env.empty() || bytes < env.front().first) return std::nan("");
  for (std::size_t i = 1; i < env.size(); ++i) {
    if (bytes < env[i].first) {
      const double u = (std::log(bytes) - std::log(env[i - 1].first)) / (std::log(env[i].first) - std::log(env[i - 1].first));
      return env[i - 1].second + u * (env[i].second - env[i - 1].second);
    }
  }
  return env.back().second;
}

struct Ratio {
  double point = 0, lo = 0, hi = 0;
  int censor = 0;            // of the point estimate
  double censored_share = 0; // share of resamples outside the envelope
  double other_kb = 0;       // the baseline's size at equal quality (point estimate)
  double dq = 0, dq_lo = 0, dq_hi = 0;  // quality difference at the network's size (network minus baseline envelope)
};

// Equal-quality ratio of a network against a family, with a bootstrap over clips (the same resample on both sides).
Ratio equal_quality(const Point& net, const std::string& net_measure, const Family& fam, const std::string& fam_measure, int resamples = 10000) {
  const std::size_t n = net.q.size();
  std::vector<std::size_t> all(n);
  std::iota(all.begin(), all.end(), 0);
  Ratio r;
  const auto one = [&](const std::vector<std::size_t>& idx, int& censor, double& other, double& dq) {
    const auto env = envelope(fam, fam_measure, idx);
    const double nb = mean_at(net.b.at(net_measure), idx), nq = mean_at(net.q, idx);
    other = size_for(env, nq, censor);
    dq = nq - quality_at(env, nb);
    return other / nb;
  };
  double dq0 = 0;
  r.point = one(all, r.censor, r.other_kb, dq0);
  r.other_kb /= 1024.0;
  r.dq = dq0;
  std::mt19937_64 rng(1);
  std::uniform_int_distribution<std::size_t> pick(0, n - 1);
  std::vector<double> ratios, dqs;
  int cens = 0;
  std::vector<std::size_t> idx(n);
  for (int b = 0; b < resamples; ++b) {
    for (auto& i : idx) i = pick(rng);
    int ce = 0;
    double other = 0, dq = 0;
    const double ratio = one(idx, ce, other, dq);
    if (!std::isnan(ratio)) ratios.push_back(ratio);
    if (!std::isnan(dq)) dqs.push_back(dq);
    cens += ce != 0;
  }
  std::ranges::sort(ratios);
  std::ranges::sort(dqs);
  const auto pct = [](const std::vector<double>& v, double p) {
    return v.empty() ? std::nan("") : v[static_cast<std::size_t>(p * static_cast<double>(v.size() - 1) + 0.5)];
  };
  r.lo = pct(ratios, 0.025);
  r.hi = pct(ratios, 0.975);
  r.dq_lo = pct(dqs, 0.025);
  r.dq_hi = pct(dqs, 0.975);
  r.censored_share = static_cast<double>(cens) / resamples;
  return r;
}

std::string ratio_cell(const Ratio& r) {
  const std::string mark = r.censor > 0 ? ">" : r.censor < 0 ? "<" : "";
  std::string s = std::format("{}{:.1f}x [{:.1f}, {:.1f}]", mark, r.point, r.lo, r.hi);
  if (r.censored_share > 0.025) s += std::format(" ({:.0f}% censored)", 100 * r.censored_share);
  return s;
}

void step_report(const Ctx& c, const std::vector<std::string>& only_configs) {
  const auto clips = clip_set(c);
  std::map<std::string, std::size_t> ci;
  for (std::size_t i = 0; i < clips.size(); ++i) ci[clips[i].name] = i;
  const std::size_t n = clips.size();
  const auto fill = [&](Point& p, const std::string& clip) {
    if (p.q.empty()) p.q.assign(n, std::nan(""));
    return ci.at(clip);
  };
  // Flipbooks.
  Family flips;
  {
    Csv f(c.out / std::format("f2_flipbooks_{}.csv", c.set), kFlipCols);
    for (const auto& r : f.rows()) {
      if (!ci.contains(r.at("clip"))) continue;
      Point& p = flips[r.at("config")];
      const std::size_t i = fill(p, r.at("clip"));
      p.q[i] = std::stod(r.at("active_psnr"));
      for (const std::string m : {"memory", "packed"}) {
        auto& v = p.b[m];
        if (v.empty()) v.assign(n, std::nan(""));
        v[i] = std::stod(r.at(m + "_bytes"));
      }
    }
  }
  // Video codecs, each its own family, and all of them together.
  std::map<std::string, Family> videos;
  Family all_video;
  if (fs::exists(c.out / std::format("f2_video_{}.csv", c.set))) {
    Csv v(c.out / std::format("f2_video_{}.csv", c.set), kVideoCols);
    for (const auto& r : v.rows()) {
      if (!ci.contains(r.at("clip"))) continue;
      for (Family* fam : {&videos[r.at("codec")], &all_video}) {
        Point& p = (*fam)[r.at("codec") + " q" + r.at("q")];
        const std::size_t i = fill(p, r.at("clip"));
        p.q[i] = std::stod(r.at("active_psnr"));
        auto& b = p.b["payload"];
        if (b.empty()) b.assign(n, std::nan(""));
        b[i] = std::stod(r.at("payload_bytes"));
      }
    }
  }
  // Networks.
  std::map<std::string, Point> nets;
  std::map<std::string, std::map<std::string, double>> net_extra;  // config -> resident, scratch means
  {
    Csv f(c.out / std::format("f2_nets_{}.csv", c.set), kNetCols);
    for (const auto& r : f.rows()) {
      if (!ci.contains(r.at("clip"))) continue;
      if (!only_configs.empty() && std::ranges::find(only_configs, r.at("config")) == only_configs.end()) continue;
      Point& p = nets[r.at("config")];
      const std::size_t i = fill(p, r.at("clip"));
      p.q[i] = std::stod(r.at("active_psnr"));
      for (const std::string m : {"stored", "packed", "resident"}) {
        auto& v = p.b[m];
        if (v.empty()) v.assign(n, std::nan(""));
        v[i] = std::stod(r.at(m + "_bytes"));
      }
      net_extra[r.at("config")]["scratch"] = std::stod(r.at("scratch_bytes"));
    }
  }
  const auto complete = [&](const Point& p) {
    return std::ranges::none_of(p.q, [](double v) { return std::isnan(v); });
  };
  std::erase_if(flips, [&](const auto& kv) { return !complete(kv.second); });
  for (auto& [k, fam] : videos) std::erase_if(fam, [&](const auto& kv) { return !complete(kv.second); });
  std::erase_if(all_video, [&](const auto& kv) { return !complete(kv.second); });
  std::vector<std::size_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);

  std::println("## Study F2 on the {} set ({} clips)\n", c.set, n);
  if (!flips.empty()) {
    for (const std::string m : {"memory", "packed"}) {
      std::string s;
      for (const auto& [kb, q] : envelope(flips, m, idx)) s += std::format(" {:.1f}:{:.2f}", kb / 1024, q);
      std::println("Flipbook envelope ({}, KB:dB):{}", m, s);
    }
  }
  for (const auto& [name, fam] : videos) {
    std::string s;
    for (const auto& [kb, q] : envelope(fam, "payload", idx)) s += std::format(" {:.1f}:{:.2f}", kb / 1024, q);
    std::println("{} envelope (payload, KB:dB):{}", name, s);
  }
  std::println("\n| network | active PSNR | stored KB | resident KB | packed KB | memory vs flipbooks | disk: packed vs packed flipbooks | dB vs flipbooks at equal memory |");
  std::println("|---|---:|---:|---:|---:|---|---|---|");
  std::ofstream eq(c.out / std::format("f2_equal_quality_{}.csv", c.set));
  eq << "set,network,clips,active_psnr,stored_kb,resident_kb,scratch_kb,packed_kb,baseline,net_measure,baseline_measure,baseline_kb,ratio,ratio_lo,ratio_hi,censor,censored_share,"
        "delta_db_at_net_size,delta_db_lo,delta_db_hi\n";
  std::vector<std::pair<std::string, const Point*>> order;
  for (const auto& [k, p] : nets) {
    if (complete(p)) order.emplace_back(k, &p);
  }
  std::ranges::sort(order, [&](const auto& a, const auto& b) { return mean_at(a.second->b.at("stored"), idx) < mean_at(b.second->b.at("stored"), idx); });
  const auto write = [&](const std::string& k, const Point& p, const std::string& base, const std::string& nm, const std::string& bm, const Ratio& r) {
    eq << std::format("{},{},{},{:.3f},{:.2f},{:.2f},{:.2f},{:.2f},{},{},{},{:.2f},{:.3f},{:.3f},{:.3f},{},{:.3f},{:.3f},{:.3f},{:.3f}\n", c.set, k, n,
                      mean_at(p.q, idx), mean_at(p.b.at("stored"), idx) / 1024, mean_at(p.b.at("resident"), idx) / 1024,
                      net_extra[k]["scratch"] / 1024, mean_at(p.b.at("packed"), idx) / 1024, base, nm, bm, r.other_kb, r.point, r.lo, r.hi, r.censor,
                      r.censored_share, r.dq, r.dq_lo, r.dq_hi);
  };
  for (const auto& [k, pp] : order) {
    const Point& p = *pp;
    std::string mem = "-", disk = "-", ddb = "-";
    if (!flips.empty()) {
      const Ratio rm = equal_quality(p, "stored", flips, "memory");
      const Ratio rd = equal_quality(p, "packed", flips, "packed");
      mem = ratio_cell(rm);
      disk = ratio_cell(rd);
      ddb = std::format("{:+.2f} [{:+.2f}, {:+.2f}]", rm.dq, rm.dq_lo, rm.dq_hi);
      write(k, p, "flipbook", "stored", "memory", rm);
      write(k, p, "flipbook", "packed", "packed", rd);
    }
    std::println("| {} | {:.2f} | {:.1f} | {:.1f} | {:.1f} | {} | {} | {} |", k, mean_at(p.q, idx), mean_at(p.b.at("stored"), idx) / 1024,
                 mean_at(p.b.at("resident"), idx) / 1024, mean_at(p.b.at("packed"), idx) / 1024, mem, disk, ddb);
  }
  if (!videos.empty()) {
    std::println("\nOn disk against video codecs (network packed by the lossless coder; codec payload bytes; equal mean active PSNR):\n");
    std::string head = "| network | packed KB |";
    std::string rule = "|---|---:|";
    for (const auto& [name, fam] : videos) {
      head += " " + name + " |";
      rule += "---|";
    }
    head += " best codec |";
    rule += "---|";
    std::println("{}\n{}", head, rule);
    for (const auto& [k, pp] : order) {
      const Point& p = *pp;
      std::string line = std::format("| {} | {:.1f} |", k, mean_at(p.b.at("packed"), idx) / 1024);
      for (const auto& [name, fam] : videos) {
        if (fam.empty()) {
          line += " - |";
          continue;
        }
        const Ratio r = equal_quality(p, "packed", fam, "payload");
        write(k, p, name, "packed", "payload", r);
        line += " " + ratio_cell(r) + " |";
      }
      if (!all_video.empty()) {
        const Ratio r = equal_quality(p, "packed", all_video, "payload");
        write(k, p, "best_video", "packed", "payload", r);
        line += " " + ratio_cell(r) + " |";
      } else {
        line += " - |";
      }
      std::println("{}", line);
    }
  }
}

// --- timing ----------------------------------------------------------------------------------------------------------

double thread_seconds() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

void step_timing(const std::vector<std::string>& models, int core, int reps) {
  cpu_set_t one;
  CPU_ZERO(&one);
  CPU_SET(core, &one);
  if (sched_setaffinity(0, sizeof(one), &one) != 0) std::println("timing: could not pin to core {}", core);
  double load[3] = {0, 0, 0};
  {
    std::ifstream la("/proc/loadavg");
    la >> load[0] >> load[1] >> load[2];
  }
  std::println("timing on core {}, load average {:.2f} {:.2f} {:.2f} (provisional: the machine is shared)", core, load[0], load[1], load[2]);
  std::println("| model | stored KB | resident KB | ms per 128 x 128 frame (thread CPU, median of 200 frames, least of {} runs) |", reps);
  std::println("|---|---:|---:|---:|");
  for (const std::string& path : models) {
    const auto bytes = read_bytes(path);
    RtEffect fx(bytes);
    nvfx_effect_info info{};
    nvfx_effect_get_info(fx.e, &info);
    nvfx_instance* in = nullptr;
    if (nvfx_instance_create(fx.e, kSize, &in) != NVFX_OK) throw std::runtime_error("instance failed");
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(kSize) * kSize * 4);
    double best = 1e9;
    for (int r = 0; r < reps; ++r) {
      std::vector<double> ms;
      for (int f = 0; f < 220; ++f) {
        const double t0 = thread_seconds();
        nvfx_render(in, f / 30.0, buf.data(), kSize * 4);
        if (f >= 20) ms.push_back(1e3 * (thread_seconds() - t0));
      }
      std::ranges::sort(ms);
      best = std::min(best, ms[ms.size() / 2]);
    }
    nvfx_instance_free(in);
    std::println("| {} | {:.1f} | {:.1f} | {:.3f} |", fs::path(path).filename().string(), static_cast<double>(info.stored_bytes) / 1024,
                 static_cast<double>(info.resident_bytes) / 1024, best);
  }
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  for (const auto p : std::views::split(s, ',')) {
    if (!std::string_view(p).empty()) v.emplace_back(std::string_view(p));
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"help"});
  const auto& pos = a.positional();
  if (a.flag("help") || pos.empty()) {
    std::println("nvfx_f2 data | flipbooks | train --configs A,B | rescore --name N --pattern P | video [--codecs ...] | report | timing --models a,b\n"
                 "        [--set val|test] [--root DIR] [--out DIR] [--threads 2] [--clips a,b]");
    return 0;
  }
  Ctx c;
  c.root = a.has("root") ? fs::path(a.str("root")) : data_root();
  c.out = a.str("out", "results/compression");
  c.threads = a.i("threads", 2);
  c.set = a.str("set", "val");
  for (const auto& s : split(a.str("clips", ""))) c.only.insert(s);
  const std::string step = pos[0];
  if (step == "data") step_data(c);
  else if (step == "flipbooks") step_flipbooks(c);
  else if (step == "train") step_train(c, split(a.need("configs")));
  else if (step == "rescore") step_rescore(c, a.need("name"), a.need("pattern"));
  else if (step == "video") {
    const auto v = split(a.str("codecs", ""));
    step_video(c, std::set<std::string>(v.begin(), v.end()));
  } else if (step == "report") step_report(c, split(a.str("configs", "")));
  else if (step == "timing") step_timing(split(a.need("models")), a.i("core", 3), a.i("reps", 5));
  else throw std::invalid_argument("unknown step " + step);
  a.warn_unused();
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_f2: {}", e.what());
  return 1;
}
