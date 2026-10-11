// nvfx_f2: study F2, compression of effects pushed as far as it honestly goes (results/compression/README.md, F2).
//
//   nvfx_f2 data                                    validation clips (2 per effect, settings and seeds not in study A)
//   nvfx_f2 flipbooks --set val|test                the study A flipbook ladder on every clip: scores, memory, packed bytes
//   nvfx_f2 trim --set val|test                     the same flipbooks without their empty space (study F3, below)
//   nvfx_f2 pairs --set val|test --pairs a:b,c:d    paired differences of network configurations over the clips
//   nvfx_f2 train --set val|test --configs A,B,...  train, save, score through the runtime, pack (lossless coder)
//   nvfx_f2 rescore --set test --name N --pattern P score and pack existing models ("{clip}" in P is the clip name)
//   nvfx_f2 video --set val|test [--codecs x264,...] the video codecs' quality ladders on every clip
//   nvfx_f2 report --set val|test [--configs ...]   equal-quality ratios against flipbooks and each codec, memory and
//                                                   disk, with 95% bootstrap intervals over clips
//   nvfx_f2 g3c --split val|test [--variants v1,b6_d,...] [--base DIR --tag NAME]  (another base than the v1 files)
//                                                   design G3c: study D's rollout effects with quantised, dithered and
//                                                   fewer start points, scored as study D's endless runs
//   nvfx_f2 g3c-report --split val|test             each variant against v1, paired over the settings
//   nvfx_f2 timing --models a.nvfx,b.nvfx [--core 3] [--reps 5]
//                                                   thread CPU time per 128 x 128 frame, least of the repetitions
//   options: --root DIR (data root, default $NEURALVFX_DATA), --out DIR (results/compression), --threads 2,
//            --clips a,b (a subset of the set), --study NAME (default f2: networks in $NEURALVFX_DATA/NAME/models and
//            NAME_nets_<set>.csv; study F3 uses f3, and its report also reads F2's networks), --teacher CONFIG (the
//            distillation teacher, a configuration trained by the same study or F2; default the same architecture at
//            8 bits), --flipbooks a.csv,b.csv (report: more flipbook rows, e.g. other encoders, besides f2_flipbooks_<set>.csv),
//            --suffix S (report: its table as <study>_equal_quality_<set>S.csv), --pareto (report: envelopes of the points
//            that improve on every smaller one only; see g_pareto)
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
//   vq<bits>x<dim>           vector-quantised features: an index of <bits> bits per <dim> channels (vs<f>: from a
//                            fraction f of the training on, default 0.5)
//   i<iterations>            training steps (default 2000 grid, 1500 conv, as study A)
//   s<seed>                  training seed (default 1)
//   m<bits>                  mixed precision (study F3): every feature plane its own bits, an average of <bits> per value,
//                            chosen by distortion when quantisation starts (default halfway; qs<f> moves it), then QAT
//   mr<lo>-<hi>              the range of the mixed widths (default 0-8; 0 bits: the plane is one value)
//   sp<d>                    sparse features (study F3, grid family): only the grid points each time slice needs are
//                            stored (the clip's support, grown by d points, default 0); the others take a fill per plane
//   d<alpha>                 distillation (study F3): the target is (1 - alpha) x the clip + alpha x the teacher's frames
//                            (for a squared error, the same as weighting the two losses); scored against the clip
//   e.g. g32c8h32l2t16_b8 is study A's grid_m at 8 bits; g32c8h32l2t16_b4_q_r3e-5 adds 4-bit QAT and a rate term.
#include "args.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/cm.hpp>
#include <neuralfx/flipbook.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/model.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>
#include <neuralfx/sim.hpp>
#include <neuralfx/train.hpp>
#include <neuralfx/video_codec.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <limits>
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
#include <tuple>

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
  fs::path base;               // g3c: the rollout effects to start from (default: the frozen v1 files)
  std::string tag;             // g3c: prefix of the variant names written for that base
  std::string study = "f2";    // prefix of the network table and folder of the models
  std::string teacher;         // distillation teacher configuration (empty: the architecture at 8 bits)
  std::vector<fs::path> flipbooks;  // report: more flipbook tables
  std::string suffix;          // report: written as <study>_equal_quality_<set><suffix>.csv
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
  // An existing file is read, and appended to, by its own header (older tables may have fewer columns).
  Csv(fs::path path, std::vector<std::string> cols) : path_(std::move(path)), cols_(std::move(cols)) {
    std::ifstream in(path_);
    std::string line;
    if (!std::getline(in, line)) return;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    cols_.clear();
    for (const auto part : std::views::split(line, ',')) cols_.emplace_back(std::string_view(part));
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
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

// Flipbooks without their empty space (study F3), as a fairer baseline for networks that store only the grid points
// their frames need. Two variants of every ladder flipbook, at the same quality (the dropped blocks decode to zero, so
// playback is unchanged; scores and packed sizes are F2's):
//   trim    each kept frame cropped to the bounding box of its non-zero blocks (BC3: 4 x 4 texels; raw: pixels), as
//           sprite atlases are packed in production, plus 8 bytes per frame for its rectangle (packing assumed perfect);
//   sparse  only the non-zero 4 x 4 blocks of each frame, plus one bit per block (a mask, as the networks keep).
// Motion vectors are cropped to the frame's rectangle scaled to their resolution in both variants.
void step_trim(const Ctx& c) {
  Csv f2(c.out / std::format("f2_flipbooks_{}.csv", c.set), kFlipCols);
  Csv trim(c.out / std::format("f3_flipbooks_trim_{}.csv", c.set), kFlipCols);
  Csv sparse(c.out / std::format("f3_flipbooks_sparse_{}.csv", c.set), kFlipCols);
  for (const ClipRef& r : clip_set(c)) {
    const Clip ref = load_clip(r);
    for (const auto& spec : flipbook::ladder(kSize, kFrames)) {
      const std::string name = spec.describe();
      if (trim.has({{"set", c.set}, {"clip", r.name}, {"config", name + " trim"}})) continue;
      const auto src = std::ranges::find_if(f2.rows(), [&](const auto& row) { return row.at("set") == c.set && row.at("clip") == r.name && row.at("config") == name; });
      if (src == f2.rows().end()) throw std::runtime_error("trim: run the flipbooks step first (" + r.name + ", " + name + ")");
      const auto fb = flipbook::build(ref, spec);
      const int R = spec.res, B = R / 4;
      const bool bc3 = spec.codec == flipbook::Codec::bc3;
      std::size_t trim_bytes = 0, sparse_bytes = 0;
      for (std::size_t k = 0; k < fb.frames.size(); ++k) {
        const auto& img = fb.frames[k];
        int x0 = R, y0 = R, x1 = -1, y1 = -1;  // bounding box of non-zero texels
        std::size_t blocks = 0;
        for (int by = 0; by < B; ++by) {
          for (int bx = 0; bx < B; ++bx) {
            bool any = false;
            for (int y = 4 * by; y < 4 * by + 4; ++y) {
              for (int x = 4 * bx; x < 4 * bx + 4; ++x) {
                const std::size_t i = (static_cast<std::size_t>(y) * R + x) * 4;
                if (img[i] | img[i + 1] | img[i + 2] | img[i + 3]) {
                  any = true;
                  x0 = std::min(x0, x);
                  y0 = std::min(y0, y);
                  x1 = std::max(x1, x);
                  y1 = std::max(y1, y);
                }
              }
            }
            blocks += any;
          }
        }
        std::size_t w = 0, hgt = 0;  // the rectangle, in texels (BC3: whole blocks)
        if (x1 >= 0) {
          if (bc3) {
            w = static_cast<std::size_t>(4 * (x1 / 4 - x0 / 4 + 1));
            hgt = static_cast<std::size_t>(4 * (y1 / 4 - y0 / 4 + 1));
          } else {
            w = static_cast<std::size_t>(x1 - x0 + 1);
            hgt = static_cast<std::size_t>(y1 - y0 + 1);
          }
        }
        const std::size_t texel = bc3 ? 1 : 4;  // bytes per texel (BC3: 16 bytes per 4 x 4 block)
        trim_bytes += w * hgt * texel + 8;
        sparse_bytes += blocks * 16 * texel + static_cast<std::size_t>(B * B + 7) / 8;
        if (spec.flow_res > 0) {  // vectors of the rectangle at the flow's resolution, 2 bytes each
          const double sc = static_cast<double>(spec.flow_res) / R;
          const auto fw = static_cast<std::size_t>(std::ceil(static_cast<double>(w) * sc)), fh = static_cast<std::size_t>(std::ceil(static_cast<double>(hgt) * sc));
          trim_bytes += 2 * fw * fh;
          sparse_bytes += 2 * fw * fh;
        }
      }
      for (auto [table, suffix, bytes] : {std::tuple{&trim, " trim", trim_bytes}, std::tuple{&sparse, " sparse", sparse_bytes}}) {
        std::map<std::string, std::string> row = *src;
        row["config"] = name + suffix;
        row["family"] = row["family"] + std::string(suffix == std::string(" trim") ? "_trim" : "_sparse");
        row["memory_bytes"] = std::to_string(bytes);
        table->add(row);
      }
      std::println("trim {} {}: {} -> {} (trim), {} (sparse) bytes", r.name, name, fb.bytes, trim_bytes, sparse_bytes);
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
  int vq_bits = 0, vq_dim = 0;
  float vq_start = 0.5f;
  float mixed = 0.f;  // average bits per value of mixed precision
  int mixed_min = 0, mixed_max = 8;
  float distill = 0.f;
  bool sparse = false;
  int sparse_dilate = 0;
  std::string arch;   // the architecture field
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
  cf.arch = a;
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
  const bool has_qs = std::ranges::any_of(parts, [](const std::string& p) { return p.starts_with("qs"); });
  bool q_flag = false;
  for (std::size_t k = 1; k < parts.size(); ++k) {
    const std::string& p = parts[k];
    if (p == "q") {
      cf.qat = true;
      q_flag = true;
    } else if (p == "t") cf.trim = true;
    else if (p.starts_with("mr")) {
      if (std::sscanf(p.c_str(), "mr%d-%d", &cf.mixed_min, &cf.mixed_max) != 2) throw std::invalid_argument("configuration: mr<lo>-<hi>");
    } else if (p[0] == 'm') {
      cf.mixed = std::stof(p.substr(1));
      if (!has_qs) cf.qat_start = 0.5f;
    } else if (p[0] == 'd') cf.distill = std::stof(p.substr(1));
    else if (p.starts_with("sp")) {
      cf.sparse = true;
      if (p.size() > 2) cf.sparse_dilate = std::stoi(p.substr(2));
    }
    else if (p.starts_with("vq")) {
      if (std::sscanf(p.c_str(), "vq%dx%d", &cf.vq_bits, &cf.vq_dim) != 2) throw std::invalid_argument("configuration: vq<bits>x<dim>");
    } else if (p.starts_with("vs")) cf.vq_start = std::stof(p.substr(2));
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
  if (cf.mixed > 0.f && !q_flag) cf.qat = false;  // qs<f> only moves the allocation and the quantisation
  if (cf.mixed > 0.f && (cf.qat || cf.vq_bits || cf.mixed < static_cast<float>(cf.mixed_min) || cf.mixed > static_cast<float>(cf.mixed_max))) {
    throw std::invalid_argument("configuration: m<bits> within mr<lo>-<hi>, without q or vq");
  }
  if (cf.distill < 0.f || cf.distill > 1.f) throw std::invalid_argument("configuration: d<alpha> with alpha in [0, 1]");
  if (cf.sparse && (h.arch != Arch::grid || cf.vq_bits || cf.lambda > 0.f || (!cf.qat && cf.mixed == 0.f))) {
    throw std::invalid_argument("configuration: sp needs the grid family and q or m<bits>, without vq or a rate term");
  }
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
                                           "ssim", "tpsnr", "flicker", "alloc_s", "plane_bits", "mask_share"};

// Planes per width of a mixed-precision model ("0:3;2:40;4:85"), or empty.
std::string bits_histogram(const Model& m) {
  if (!m.per_plane()) return "";
  std::array<int, 9> n{};
  for (const std::uint8_t b : m.plane_bits) ++n[b];
  std::string s;
  for (std::size_t b = 0; b < n.size(); ++b) {
    if (n[b] > 0) s += std::format("{}{}:{}", s.empty() ? "" : ";", b, n[b]);
  }
  return s;
}

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
  row["est_bits_per_value"] = f4(train::feature_rate(*loaded, loaded->feature_bits < 16 ? loaded->feature_bits : 8, {}, 0.f, false, loaded->plane_bits) /
                                 static_cast<double>(loaded->features.size()));
  row["plane_bits"] = bits_histogram(*loaded);
  if (loaded->masked()) {
    row["mask_share"] = f4(static_cast<double>(std::ranges::count(loaded->feature_mask, std::uint8_t{1})) / static_cast<double>(loaded->feature_mask.size()));
  }
}

// The distillation teacher's frames for a clip: its model file from this study or F2, through the runtime.
Clip teacher_clip(const Ctx& c, const ClipRef& r, const std::string& teacher) {
  for (const std::string& study : {c.study, std::string("f2")}) {
    const fs::path file = c.root / study / "models" / c.set / std::format("{}__{}.nvfx", r.name, teacher);
    if (!fs::exists(file)) continue;
    auto m = load_model(file);
    if (!m) throw std::runtime_error(m.error());
    RtEffect fx(read_bytes(file));
    std::size_t scratch = 0;
    return runtime_clip(fx, *m, scratch);
  }
  throw std::runtime_error(std::format("distillation: no teacher {} for {} (train it first)", teacher, r.name));
}

void step_train(const Ctx& c, const std::vector<std::string>& configs) {
  Csv csv(c.out / std::format("{}_nets_{}.csv", c.study, c.set), kNetCols);
  const fs::path models = c.root / c.study / "models" / c.set;
  for (const std::string& name : configs) {
    const Config cf = parse_config(name);
    for (const ClipRef& r : clip_set(c)) {
      if (csv.has({{"set", c.set}, {"clip", r.name}, {"config", name}})) continue;
      const Clip ref = load_clip(r);
      Clip target = ref;  // what training sees: the clip, or with distillation its blend with the teacher's frames
      if (cf.distill > 0.f) {
        const Clip t = teacher_clip(c, r, c.teacher.empty() ? cf.arch + "_b8" : c.teacher);
        for (std::size_t i = 0; i < target.rgba.size(); ++i) {
          const float v = (1.f - cf.distill) * static_cast<float>(ref.rgba[i]) + cf.distill * static_cast<float>(t.rgba[i]);
          target.rgba[i] = static_cast<std::uint8_t>(std::clamp(std::lround(v), 0L, 255L));
        }
      }
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
      o.vq_bits = cf.vq_bits;
      o.vq_dim = cf.vq_dim;
      o.vq_start = cf.vq_start;
      if (cf.sparse) {
        o.sparse = true;
        o.sparse_dilate = cf.sparse_dilate;
      }
      if (cf.mixed > 0.f) {
        o.mixed_bits = cf.mixed;
        o.mixed_min = cf.mixed_min;
        o.mixed_max = cf.mixed_max;
        o.qat_start = cf.qat_start;
        o.qat_trim = cf.trim;
      }
      Hyper h = cf.h;
      h.loop = ref.loop;
      const train::Example ex{&target, {}};
      auto res = train::train(h, std::span(&ex, 1), o);
      res.model.effect = r.effect;
      res.model.fps = ref.fps;
      res.model.feature_bits = cf.bits;
      res.model.feature_trim = cf.trim;
      const fs::path file = models / std::format("{}__{}.nvfx", r.name, name);
      fs::create_directories(models);
      if (auto w = save_model(file, res.model); !w) throw std::runtime_error(w.error());
      std::map<std::string, std::string> row = {{"set", c.set}, {"clip", r.name}, {"effect", r.effect}, {"config", name},
                                                {"arch", h.arch == Arch::grid ? "grid" : "conv"},
                                                {"bits", cf.mixed > 0.f ? std::format("{}", cf.mixed) : std::to_string(cf.bits)},
                                                {"qat", cf.qat || cf.mixed > 0.f ? std::format("{}", cf.qat_start) : cf.vq_bits ? std::format("vq{}x{}", cf.vq_bits, cf.vq_dim) : "-"}, {"lambda", std::format("{}", cf.lambda)},
                                                {"iters", std::to_string(cf.iters)}, {"train_s", std::format("{:.1f}", res.seconds)},
                                                {"alloc_s", std::format("{:.1f}", res.alloc_seconds)}};
      score_and_pack(ref, file, row);
      csv.add(row);
      std::println("train {} {}: {:.0f} s, {} -> {} bytes packed, active {}, est {} bits/value {}", r.name, name, res.seconds, row["stored_bytes"],
                   row["packed_bytes"], row["active_psnr"], row["est_bits_per_value"], row["plane_bits"]);
      std::fflush(stdout);
    }
  }
}

void step_rescore(const Ctx& c, const std::string& name, const std::string& pattern) {
  Csv csv(c.out / std::format("{}_nets_{}.csv", c.study, c.set), kNetCols);
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

// With --pareto (study F3), the envelope keeps only the points that improve on every smaller one: a dominated point
// otherwise stays a vertex at the running best and draws a flat step that the log-linear interpolation then crosses
// steeply (it inflates the size at equal quality). F2's envelopes are the default.
bool g_pareto = false;

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
    else if (!(g_pareto && !env.empty() && q <= env.back().second)) env.emplace_back(kb, best);
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
// With several families (the video codecs), the best of them: the smallest size at the network's quality, and the
// best quality at its size, each codec along its own ladder.
Ratio equal_quality(const Point& net, const std::string& net_measure, const std::vector<const Family*>& fams, const std::string& fam_measure,
                    int resamples = 10000) {
  const std::size_t n = net.q.size();
  std::vector<std::size_t> all(n);
  std::iota(all.begin(), all.end(), 0);
  Ratio r;
  const auto one = [&](const std::vector<std::size_t>& idx, int& censor, double& other, double& dq) {
    const double nb = mean_at(net.b.at(net_measure), idx), nq = mean_at(net.q, idx);
    other = std::numeric_limits<double>::infinity();
    double best_q = -std::numeric_limits<double>::infinity();
    for (const Family* fam : fams) {
      const auto env = envelope(*fam, fam_measure, idx);
      int ce = 0;
      const double kb = size_for(env, nq, ce);
      if (!std::isnan(kb) && kb < other) {
        other = kb;
        censor = ce;
      }
      const double q = quality_at(env, nb);
      if (!std::isnan(q)) best_q = std::max(best_q, q);
    }
    if (std::isinf(other)) other = std::nan("");
    dq = std::isinf(best_q) ? std::nan("") : nq - best_q;
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

Ratio equal_quality(const Point& net, const std::string& net_measure, const Family& fam, const std::string& fam_measure) {
  return equal_quality(net, net_measure, std::vector<const Family*>{&fam}, fam_measure);
}

std::string ratio_cell(const Ratio& r) {
  const std::string mark = r.censor > 0 ? ">" : r.censor < 0 ? "<" : "";
  const int prec = r.point < 1 ? 2 : 1;  // the codecs' side: 0.13x reads better than 0.1x
  std::string s = std::format("{}{:.{}f}x [{:.{}f}, {:.{}f}]", mark, r.point, prec, r.lo, prec, r.hi, prec);
  if (r.censored_share > 0.025) s += std::format(" ({:.0f}% censored)", 100 * r.censored_share);
  return s;
}

// A short label for a network configuration ("G32 4-bit", "G48 VQ 8 bits / 8 ch, rate 1e-4").
std::string short_label(const std::string& config) {
  Config cf;
  try {
    cf = parse_config(config);
  } catch (const std::exception&) {
    return config;  // a rescored file (e.g. study A's own models)
  }
  std::string s = cf.h.arch == Arch::grid ? std::format("G{}", cf.h.grid) : std::string("conv");
  if (cf.h.arch == Arch::grid && cf.h.channels != 8) s += std::format(" C{}", cf.h.channels);
  if (cf.h.grid_t != 16) s += std::format(" T{}", cf.h.grid_t);
  if (cf.h.arch == Arch::grid && cf.h.hidden != 32) s += std::format(" H{}", cf.h.hidden);
  if (cf.vq_bits) s += std::format(" VQ{}/{}", cf.vq_bits, cf.vq_dim);
  else if (cf.mixed > 0.f) s += std::format(" mixed {:g}-bit", cf.mixed);
  else s += std::format(" {}-bit", cf.bits);
  if (cf.sparse) s += " sparse";
  if (cf.distill > 0.f) s += std::format(" distil {:g}", cf.distill);
  if (cf.lambda > 0) s += " + rate";
  if ((cf.h.arch == Arch::grid && cf.iters != 2000) || (cf.h.arch == Arch::conv && cf.iters != 1500)) s += std::format(" {}k it", cf.iters / 1000);
  return s;
}

// Two panels as SVG: memory against quality (flipbooks, networks) and disk against quality (flipbooks and networks
// packed by the lossless coder, video codecs' payload). Log size axis; one quality axis per panel.
void write_figure(const fs::path& path, const std::string& title, const Family& flips, const Family* flips_more, const std::map<std::string, Family>& videos,
                  const std::vector<std::pair<std::string, const Point*>>& nets, const std::vector<std::size_t>& idx) {
  constexpr double W = 1040, H = 470, top = 74, bottom = 58, left = 62, gap = 70;
  const double pw = (W - left - gap - 24) / 2, ph = H - top - bottom;
  const char* ink = "#0b0b0b";
  const char* ink2 = "#52514e";
  const char* grid = "#e4e3df";
  // Categorical slots in fixed order (the reference palette): flipbooks, networks, then the codecs shown.
  const std::array<const char*, 8> slot = {"#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"};
  double qlo = 1e9, qhi = -1e9;
  const auto qrange = [&](double q) {
    qlo = std::min(qlo, q);
    qhi = std::max(qhi, q);
  };
  for (const auto& [k, p] : nets) qrange(mean_at(p->q, idx));
  qlo = std::floor(std::min(qlo, 26.0)) - 1;
  qhi = std::ceil(std::max(qhi, 36.0)) + 1;
  const double klo = 2, khi = 2048;
  std::ostringstream o;
  o << std::format("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" font-family=\"Helvetica, Arial, sans-serif\">\n", W, H, W, H);
  o << std::format("<rect width=\"{}\" height=\"{}\" fill=\"#fcfcfb\"/>\n", W, H);
  o << std::format("<text x=\"{}\" y=\"24\" font-size=\"16\" font-weight=\"600\" fill=\"{}\">{}</text>\n", left, ink, title);
  const std::vector<std::string> shown = {"aom", "x265", "vp9a"};
  for (int panel = 0; panel < 2; ++panel) {
    const double x0 = left + panel * (pw + gap), y0 = top;
    const auto X = [&](double kb) { return x0 + (std::log(std::clamp(kb, klo, khi)) - std::log(klo)) / (std::log(khi) - std::log(klo)) * pw; };
    const auto Y = [&](double q) { return y0 + ph - (std::clamp(q, qlo, qhi) - qlo) / (qhi - qlo) * ph; };
    o << std::format("<text x=\"{}\" y=\"{}\" font-size=\"13\" font-weight=\"600\" fill=\"{}\">{}</text>\n", x0, y0 - 30, ink,
                     panel == 0 ? "Memory while playing: flipbook texture, network as stored" : "Disk: both sides losslessly packed; video bitstreams");
    for (double kb = klo; kb <= khi * 1.01; kb *= 4) {
      o << std::format("<line x1=\"{:.1f}\" y1=\"{}\" x2=\"{:.1f}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1\"/>\n", X(kb), y0, X(kb), y0 + ph, grid);
      o << std::format("<text x=\"{:.1f}\" y=\"{}\" font-size=\"11\" fill=\"{}\" text-anchor=\"middle\">{:g}</text>\n", X(kb), y0 + ph + 16, ink2, kb);
    }
    for (double q = std::ceil(qlo / 2) * 2; q <= qhi; q += 2) {
      o << std::format("<line x1=\"{}\" y1=\"{:.1f}\" x2=\"{}\" y2=\"{:.1f}\" stroke=\"{}\" stroke-width=\"1\"/>\n", x0, Y(q), x0 + pw, Y(q), grid);
      o << std::format("<text x=\"{}\" y=\"{:.1f}\" font-size=\"11\" fill=\"{}\" text-anchor=\"end\">{:g}</text>\n", x0 - 6, Y(q) + 4, ink2, q);
    }
    o << std::format("<text x=\"{}\" y=\"{}\" font-size=\"12\" fill=\"{}\" text-anchor=\"middle\">KB per effect (log scale)</text>\n", x0 + pw / 2, y0 + ph + 36, ink2);
    o << std::format("<text transform=\"translate({},{}) rotate(-90)\" font-size=\"12\" fill=\"{}\" text-anchor=\"middle\">mean active PSNR (dB)</text>\n", x0 - 42, y0 + ph / 2, ink2);
    const auto line = [&](const std::vector<std::pair<double, double>>& env, const char* colour) {
      std::string d;
      for (std::size_t i = 0; i < env.size(); ++i) {
        const double kb = env[i].first / 1024;
        if (kb < klo || kb > khi || env[i].second < qlo || env[i].second > qhi) continue;  // drawn inside the frame only
        d += std::format("{}{:.1f},{:.1f} ", d.empty() ? "M" : "L", X(kb), Y(env[i].second));
      }
      o << std::format("<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2\" stroke-linejoin=\"round\"/>\n", d, colour);
    };
    // legend
    std::vector<std::pair<std::string, const char*>> legend = {{panel == 0 ? "best flipbook at each size" : "best packed flipbook", slot[0]}, {"networks", slot[1]}};
    line(envelope(flips, panel == 0 ? "memory" : "packed", idx), slot[0]);
    if (panel == 0 && flips_more) {  // with the extra tables (F3: flipbooks without their empty space)
      line(envelope(*flips_more, "memory", idx), slot[6]);
      legend.emplace_back("flipbooks trimmed to their content", slot[6]);
    }
    if (panel == 1) {
      for (std::size_t v = 0; v < shown.size(); ++v) {
        if (!videos.contains(shown[v]) || videos.at(shown[v]).empty()) continue;
        line(envelope(videos.at(shown[v]), "payload", idx), slot[2 + v]);
        legend.emplace_back(shown[v] == "aom" ? "AV1 (libaom, 4:4:4)" : shown[v] == "x265" ? "HEVC (x265, 4:4:4)" : "VP9 with alpha (4:2:0)", slot[2 + v]);
      }
    }
    double lx = x0;
    for (const auto& [name, colour] : legend) {
      o << std::format("<rect x=\"{:.1f}\" y=\"{}\" width=\"14\" height=\"4\" rx=\"2\" fill=\"{}\"/>\n", lx, y0 - 16, colour);
      o << std::format("<text x=\"{:.1f}\" y=\"{}\" font-size=\"11\" fill=\"{}\">{}</text>\n", lx + 18, y0 - 11, ink2, name);
      lx += 26 + 6.2 * static_cast<double>(name.size());
    }
    std::vector<std::array<double, 4>> placed;  // label boxes: x0, y0, x1, y1
    for (const auto& [k, p] : nets) {
      if (short_label(k) == k) continue;  // rescored copies are not drawn twice
      const double kb = mean_at(p->b.at(panel == 0 ? "stored" : "packed"), idx) / 1024, q = mean_at(p->q, idx);
      o << std::format("<circle cx=\"{:.1f}\" cy=\"{:.1f}\" r=\"4.5\" fill=\"{}\" stroke=\"#fcfcfb\" stroke-width=\"2\"/>\n", X(kb), Y(q), slot[1]);
      // The label right of its point, moved up or down until it overlaps no other label.
      const std::string text = short_label(k);
      const double w = 5.6 * static_cast<double>(text.size()), lx0 = X(kb) + 7;
      double ly = Y(q) + 3.5;
      for (const double dy : {0.0, 12.0, -12.0, 24.0, -24.0, 36.0, -36.0}) {
        const double y = Y(q) + 3.5 + dy;
        const bool clash = std::ranges::any_of(placed, [&](const auto& b) { return lx0 < b[2] && lx0 + w > b[0] && y - 9 < b[3] && y + 2 > b[1]; });
        if (!clash) {
          ly = y;
          break;
        }
      }
      placed.push_back({lx0, ly - 9, lx0 + w, ly + 2});
      o << std::format("<text x=\"{:.1f}\" y=\"{:.1f}\" font-size=\"10\" fill=\"{}\">{}</text>\n", lx0, ly, ink, text);
    }
  }
  o << "</svg>\n";
  std::ofstream f(path);
  f << o.str();
}

void step_report(const Ctx& c, const std::vector<std::string>& only_configs, const std::string& figure) {
  const auto clips = clip_set(c);
  std::map<std::string, std::size_t> ci;
  for (std::size_t i = 0; i < clips.size(); ++i) ci[clips[i].name] = i;
  const std::size_t n = clips.size();
  const auto fill = [&](Point& p, const std::string& clip) {
    if (p.q.empty()) p.q.assign(n, std::nan(""));
    return ci.at(clip);
  };
  // Flipbooks.
  Family flips, flips_stored;  // every table; F2's table only
  std::vector<fs::path> flip_tables = {c.out / std::format("f2_flipbooks_{}.csv", c.set)};
  for (const fs::path& extra : c.flipbooks) flip_tables.push_back(extra);
  for (const fs::path& table : flip_tables) {
    if (!fs::exists(table)) throw std::runtime_error("no flipbook table " + table.string());
    Csv f(table, kFlipCols);
    for (const auto& r : f.rows()) {
      if (r.contains("set") && r.at("set") != c.set) continue;
      if (!ci.contains(r.at("clip"))) continue;
      for (Family* fam : {&flips, table == flip_tables.front() ? &flips_stored : nullptr}) {
        if (!fam) continue;
        Point& p = (*fam)[r.at("config")];
        const std::size_t i = fill(p, r.at("clip"));
        p.q[i] = std::stod(r.at("active_psnr"));
        for (const std::string m : {"memory", "packed"}) {
          auto& v = p.b[m];
          if (v.empty()) v.assign(n, std::nan(""));
          v[i] = std::stod(r.at(m + "_bytes"));
        }
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
  std::vector<std::string> net_studies = {"f2"};
  if (c.study != "f2") net_studies.push_back(c.study);  // a later study's report shows F2's networks beside its own
  for (const std::string& study : net_studies) {
    const fs::path table = c.out / std::format("{}_nets_{}.csv", study, c.set);
    if (!fs::exists(table)) continue;
    Csv f(table, kNetCols);
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
  std::erase_if(flips_stored, [&](const auto& kv) { return !complete(kv.second); });
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
  std::ofstream eq(c.out / std::format("{}_equal_quality_{}{}.csv", c.study, c.set, c.suffix));
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
  if (!figure.empty()) {
    std::string study = c.study;
    std::ranges::transform(study, study.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    write_figure(figure, std::format("Study {}, {} clips ({} set): quality against bytes", study, n, c.set), c.flipbooks.empty() ? flips : flips_stored,
                 c.flipbooks.empty() ? nullptr : &flips, videos, order, idx);
    std::println("figure: {}", figure);
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
        std::vector<const Family*> fams;
        for (const auto& [name, fam] : videos) {
          if (!fam.empty()) fams.push_back(&fam);
        }
        const Ratio r = equal_quality(p, "packed", fams, "payload");
        write(k, p, "best_video", "packed", "payload", r);
        line += " " + ratio_cell(r) + " |";
      } else {
        line += " - |";
      }
      std::println("{}", line);
    }
  }
}

// Paired differences of two network configurations over the clips of a set (study F3): active PSNR, stored and packed
// KB, each a - b with a 95% bootstrap interval (10,000 resamples). Networks from F2's table and this study's.
void step_pairs(const Ctx& c, const std::vector<std::string>& pairs) {
  std::map<std::string, std::map<std::string, std::map<std::string, double>>> v;  // config -> clip -> column -> value
  for (const std::string& study : {std::string("f2"), c.study}) {
    const fs::path table = c.out / std::format("{}_nets_{}.csv", study, c.set);
    if (!fs::exists(table)) continue;
    Csv f(table, kNetCols);
    for (const auto& r : f.rows()) {
      for (const std::string col : {"active_psnr", "stored_bytes", "packed_bytes"}) v[r.at("config")][r.at("clip")][col] = std::stod(r.at(col));
    }
  }
  const auto clips = clip_set(c);
  std::println("| a - b ({} set) | clips | active PSNR dB | stored KB | packed KB |", c.set);
  std::println("|---|---:|---|---|---|");
  for (const std::string& pr : pairs) {
    const auto colon = pr.find(':');
    if (colon == std::string::npos) throw std::invalid_argument("--pairs a:b,c:d");
    const std::string a = pr.substr(0, colon), b = pr.substr(colon + 1);
    std::string cells;
    std::size_t n = 0;
    for (const std::string col : {"active_psnr", "stored_bytes", "packed_bytes"}) {
      std::vector<double> x, y;
      for (const ClipRef& r : clips) {
        if (!v[a].contains(r.name) || !v[b].contains(r.name)) continue;
        const double k = col == "active_psnr" ? 1.0 : 1.0 / 1024.0;
        x.push_back(v[a][r.name][col] * k);
        y.push_back(v[b][r.name][col] * k);
      }
      n = x.size();
      if (x.empty()) {
        cells += " - |";
        continue;
      }
      const auto iv = metrics::paired_bootstrap(x, y);
      cells += std::format(" {:+.2f} [{:+.2f}, {:+.2f}] |", iv.mean, iv.lo, iv.hi);
    }
    std::println("| {} - {} | {} |{}", a, b, n, cells);
  }
}

// --- G3c: rollout start points quantised, dithered, coded; fewer of them -------------------------------------------

// Study B's held-out settings (study D's test) and study G's validation settings (docs/DCM.md §4), as the tools draw
// them (tools/experiment_d.cpp, tools/experiment_g.cpp).
using Setting = std::array<float, 3>;
bool off_grid(const Setting& s) {
  const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
  return off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f});
}
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

// A variant of a rollout effect: "v1" (as trained), or fields joined by "_": b<bits> (coarse start states at that many
// bits per channel plane), d (dithered by the start's seed), h (half the start points, the most spread in controls).
rollout::Model g3c_variant(const rollout::Model& v1, const std::string& name) {
  rollout::Model m = v1;
  if (name == "v1") return m;
  for (const auto part : std::views::split(name, '_')) {
    const std::string p(std::string_view{part});
    if (p == "d") m.start_dither = true;
    else if (p == "h") {
      // Farthest-point sampling in control space from the start nearest the middle of the controls.
      const std::size_t n = m.starts.size(), keep = std::max<std::size_t>(1, n / 2);
      const auto dist = [&](std::size_t a, const std::vector<float>& b) {
        float d = 0;
        for (std::size_t k = 0; k < b.size(); ++k) d += (m.starts[a].controls[k] - b[k]) * (m.starts[a].controls[k] - b[k]);
        return d;
      };
      std::vector<std::size_t> chosen;
      std::size_t first = 0;
      for (std::size_t k = 1; k < n; ++k) {
        if (dist(k, {0.5f, 0.5f, 0.5f}) < dist(first, {0.5f, 0.5f, 0.5f})) first = k;
      }
      chosen.push_back(first);
      while (chosen.size() < keep) {
        std::size_t best = 0;
        float best_d = -1;
        for (std::size_t k = 0; k < n; ++k) {
          if (std::ranges::find(chosen, k) != chosen.end()) continue;
          float dmin = 1e30f;
          for (const std::size_t c : chosen) dmin = std::min(dmin, dist(k, m.starts[c].controls));
          if (dmin > best_d) {
            best_d = dmin;
            best = k;
          }
        }
        chosen.push_back(best);
      }
      std::ranges::sort(chosen);
      std::vector<rollout::StartPoint> kept;
      for (const std::size_t k : chosen) kept.push_back(m.starts[k]);
      m.starts = std::move(kept);
    } else if (p[0] == 'b') {
      m.start_bits = std::stoi(p.substr(1));
    } else {
      throw std::invalid_argument("g3c: unknown variant field " + p);
    }
  }
  return m;
}

Clip rollout_clip(const std::vector<std::uint8_t>& bytes, const Setting& s, std::uint64_t seed, int frames) {
  RtEffect fx(bytes);
  nvfx_instance* in = nullptr;
  if (nvfx_instance_create(fx.e, kSize, &in) != NVFX_OK) throw std::runtime_error("instance");
  nvfx_instance_set_controls(in, s.data(), 3);
  nvfx_instance_set_seed(in, seed);
  Clip c;
  c.allocate(kSize, frames);
  c.fps = 30.f;
  for (int f = 0; f < frames; ++f) nvfx_render(in, f / 30.0, c.frame(f).data(), kSize * 4);
  nvfx_instance_free(in);
  return c;
}

const std::vector<std::string> kG3cCols = {"split", "effect", "variant", "setting", "starts", "start_bits", "dither", "file_bytes",
                                           "packed_bytes", "packed_coarse_bytes", "resident_bytes", "spectrum_l1", "motion_ratio",
                                           "coverage_l1", "emission_l1", "mean_frame_psnr"};

// Endless runs at held-out settings with new seeds, scored by frame statistics against a real run, as study D's
// evaluation (tools/experiment_d.cpp, d-eval): test = study B's held-out settings with study D's seeds; val = study G's
// validation settings with seeds of their own.
void step_g3c(const Ctx& c, const std::string& split, const std::vector<std::string>& variants) {
  Csv csv(c.out / "f2_g3c.csv", kG3cCols);
  const auto settings = split == "test" ? test_settings() : validation_settings();
  const std::uint64_t real_seed = split == "test" ? 900000 : 960000, net_seed = split == "test" ? 920000 : 970000;
  const fs::path dir = c.root / "f2" / "g3c";
  fs::create_directories(dir);
  for (const auto e : sim::kEffects) {
    const std::string en(sim::effect_name(e));
    if (!c.only.empty() && !c.only.contains(en)) continue;
    const bool ex = e == sim::Effect::explosion;
    const fs::path base = c.base.empty() ? c.root / "experiments" / "models" / "d" : c.base;
    auto loaded = rollout::load_model(base / (en + ".nvfx"));
    if (!loaded) throw std::runtime_error(loaded.error());
    const int F = ex ? 89 : 300, warm = ex ? 1 : 150;
    std::vector<std::string> todo;
    const auto named = [&](const std::string& v) { return c.tag.empty() ? v : c.tag + "_" + v; };
    for (const auto& v : variants) {
      if (!csv.has({{"split", split}, {"effect", en}, {"variant", named(v)}})) todo.push_back(v);
    }
    if (todo.empty()) continue;
    // The real runs, one per setting (threads over settings).
    std::vector<metrics::ClipStats> real(settings.size());
    {
      std::atomic<std::size_t> next{0};
      std::vector<std::jthread> pool;
      for (int t = 0; t < c.threads; ++t) {
        pool.emplace_back([&] {
          for (std::size_t si; (si = next++) < settings.size();) {
            sim::Params p;
            p.effect = e;
            p.intensity = settings[si][0];
            p.wind = settings[si][1];
            p.turbulence = settings[si][2];
            p.seed = real_seed + si;
            p.size = kSize;
            sim::Fluid f(p);
            for (int i = 0; i < warm; ++i) f.step_frame();
            Clip cl;
            cl.allocate(kSize, F);
            cl.fps = 30.f;
            for (int i = 0; i < F; ++i) {
              f.step_frame();
              f.render(cl.frame(i));
            }
            real[si] = metrics::stats(cl);
          }
        });
      }
    }
    for (const std::string& v : todo) {
      rollout::Model m = g3c_variant(*loaded, v);
      const fs::path file = dir / std::format("{}__{}.nvfx", en, named(v));
      std::ostringstream os;
      if (auto w = rollout::save_model(os, m); !w) throw std::runtime_error(w.error());
      const std::string str = os.str();
      const std::vector<std::uint8_t> bytes(str.begin(), str.end());
      if (v != "v1" || !c.tag.empty()) {
        std::ofstream of(file, std::ios::binary);
        of.write(str.data(), static_cast<std::streamsize>(str.size()));
      }
      const cm::Packed packed = cm::pack_model(bytes);
      const auto back = cm::unpack_model(packed.data);
      if (!back || *back != bytes) throw std::runtime_error("g3c: lossless round trip failed");
      double coarse = 0;
      for (const auto& part : packed.parts) {
        if (part.kind == cm::Kind::coarse || part.kind == cm::Kind::ranges) coarse += part.coded_bytes;
      }
      RtEffect probe(bytes);
      nvfx_effect_info info{};
      nvfx_effect_get_info(probe.e, &info);
      std::vector<metrics::StatDistance> d(settings.size());
      {
        std::atomic<std::size_t> next{0};
        std::vector<std::jthread> pool;
        for (int t = 0; t < c.threads; ++t) {
          pool.emplace_back([&] {
            for (std::size_t si; (si = next++) < settings.size();) {
              d[si] = metrics::distance(real[si], metrics::stats(rollout_clip(bytes, settings[si], net_seed + si, F)));
            }
          });
        }
      }
      for (std::size_t si = 0; si < settings.size(); ++si) {
        csv.add({{"split", split}, {"effect", en}, {"variant", named(v)}, {"setting", std::to_string(si)}, {"starts", std::to_string(m.starts.size())},
                 {"start_bits", std::to_string(m.start_bits)}, {"dither", m.start_dither ? "1" : "0"}, {"file_bytes", std::to_string(bytes.size())},
                 {"packed_bytes", std::to_string(packed.data.size())}, {"packed_coarse_bytes", std::format("{:.1f}", coarse)},
                 {"resident_bytes", std::to_string(info.resident_bytes)}, {"spectrum_l1", f4(d[si].spectrum_l1)},
                 {"motion_ratio", f4(d[si].motion_ratio)}, {"coverage_l1", f4(d[si].coverage_l1)}, {"emission_l1", f4(d[si].emission_l1)},
                 {"mean_frame_psnr", std::format("{:.3f}", d[si].mean_frame_psnr)}});
      }
      std::println("g3c {} {} {}: {} bytes, {} packed (coarse {:.0f})", split, en, named(v), bytes.size(), packed.data.size(), coarse);
      std::fflush(stdout);
    }
  }
}

// Variant against v1 on one split: per effect and statistic, the paired difference of distances to the real run
// (spectrum, coverage, emission: lower is better; |log motion ratio|: lower is better; mean-frame PSNR: higher is
// better), signed so that positive means the variant is worse.
void step_g3c_report(const Ctx& c, const std::string& split) {
  Csv csv(c.out / "f2_g3c.csv", kG3cCols);
  std::map<std::string, std::map<std::string, std::map<int, std::map<std::string, std::string>>>> rows;  // effect, variant, setting
  for (const auto& r : csv.rows()) {
    if (r.at("split") == split) rows[r.at("effect")][r.at("variant")][std::stoi(r.at("setting"))] = r;
  }
  std::println("## G3c on {} settings: variant minus v1, positive = worse (95% paired bootstrap over 10 settings)\n", split);
  std::println("| effect | variant | starts | KB | packed KB | spectrum | abs log motion | coverage | emission | mean-frame PSNR (v1 minus variant) | verdict |");
  std::println("|---|---|---:|---:|---:|---|---|---|---|---|---|");
  std::ofstream out(c.out / std::format("f2_g3c_{}_decisions.csv", split));
  out << "split,effect,variant,starts,file_bytes,packed_bytes,stat,mean,lo,hi,verdict\n";
  for (const auto& [effect, vars] : rows) {
    if (!vars.contains("v1")) continue;
    const auto& base = vars.at("v1");
    for (const auto& [v, sets] : vars) {
      if (v == "v1") continue;
      std::string line = std::format("| {} | {} | {} | {:.1f} | {:.1f} |", effect, v, sets.begin()->second.at("starts"),
                                     std::stod(sets.begin()->second.at("file_bytes")) / 1024, std::stod(sets.begin()->second.at("packed_bytes")) / 1024);
      bool worse = false, better = false;
      for (const std::string stat : {"spectrum_l1", "motion_ratio", "coverage_l1", "emission_l1", "mean_frame_psnr"}) {
        std::vector<double> a, b;
        for (const auto& [si, r] : sets) {
          if (!base.contains(si)) continue;
          const auto val = [&](const std::map<std::string, std::string>& row) {
            const double x = std::stod(row.at(stat));
            if (stat == "motion_ratio") return std::abs(std::log(x));
            if (stat == "mean_frame_psnr") return -x;
            return x;
          };
          a.push_back(val(r));
          b.push_back(val(base.at(si)));
        }
        const auto iv = metrics::paired_bootstrap(a, b);
        const std::string verdict = iv.covers_zero() ? "tie" : iv.lo > 0 ? "worse" : "better";
        worse = worse || verdict == "worse";
        better = better || verdict == "better";
        line += std::format(" {:+.4f} [{:+.4f}, {:+.4f}] |", iv.mean, iv.lo, iv.hi);
        out << std::format("{},{},{},{},{},{},{},{:.5f},{:.5f},{:.5f},{}\n", split, effect, v, sets.begin()->second.at("starts"),
                           sets.begin()->second.at("file_bytes"), sets.begin()->second.at("packed_bytes"), stat, iv.mean, iv.lo, iv.hi, verdict);
      }
      line += worse ? " worse |" : better ? " better |" : " tie |";
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
  const tools::Args a(argc, argv, {"help", "pareto"});
  const auto& pos = a.positional();
  if (a.flag("help") || pos.empty()) {
    std::println("nvfx_f2 data | flipbooks | train --configs A,B | rescore --name N --pattern P | video [--codecs ...] | report | timing --models a,b\n"
                 "        [--set val|test] [--root DIR] [--out DIR] [--threads 2] [--clips a,b] [--study f2|f3] [--teacher CONFIG]\n"
                 "        [--flipbooks more.csv,...]");
    return 0;
  }
  Ctx c;
  c.root = a.has("root") ? fs::path(a.str("root")) : data_root();
  c.out = a.str("out", "results/compression");
  c.threads = a.i("threads", 2);
  c.set = a.str("set", "val");
  for (const auto& s : split(a.str("clips", ""))) c.only.insert(s);
  c.base = a.str("base", "");
  c.tag = a.str("tag", "");
  c.study = a.str("study", "f2");
  c.teacher = a.str("teacher", "");
  for (const auto& f : split(a.str("flipbooks", ""))) c.flipbooks.emplace_back(f);
  c.suffix = a.str("suffix", "");
  g_pareto = a.flag("pareto");
  const std::string step = pos[0];
  if (step == "data") step_data(c);
  else if (step == "flipbooks") step_flipbooks(c);
  else if (step == "trim") step_trim(c);
  else if (step == "pairs") step_pairs(c, split(a.need("pairs")));
  else if (step == "train") step_train(c, split(a.need("configs")));
  else if (step == "rescore") step_rescore(c, a.need("name"), a.need("pattern"));
  else if (step == "video") {
    const auto v = split(a.str("codecs", ""));
    step_video(c, std::set<std::string>(v.begin(), v.end()));
  } else if (step == "report") step_report(c, split(a.str("configs", "")), a.str("figure", ""));
  else if (step == "timing") step_timing(split(a.need("models")), a.i("core", 3), a.i("reps", 5));
  else if (step == "g3c") step_g3c(c, a.str("split", "val"), split(a.str("variants", "v1,b8,b6,b6_d,b4_d,h,b6_d_h")));
  else if (step == "g3c-report") step_g3c_report(c, a.str("split", "val"));
  else throw std::invalid_argument("unknown step " + step);
  a.warn_unused();
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_f2: {}", e.what());
  return 1;
}
