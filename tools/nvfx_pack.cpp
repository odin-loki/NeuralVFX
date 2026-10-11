// nvfx_pack: lossless context-mixing compression of .nvfx files (include/neuralfx/cm.hpp), and its measurement.
//
//   nvfx_pack in.nvfx out.nvfz                pack a frame model or a rollout effect
//   nvfx_pack --unpack in.nvfz out.nvfx       restore the exact bytes
//   nvfx_pack --report DIR                    every .nvfx under DIR: sizes, ratio, bits per stored value, zlib -9 as a
//                                             general-purpose reference, decoding speed; every round trip is checked
//   nvfx_pack --study [--data DATA] [--scores CSV] [--out OUT] [--threads N] [--flipbooks-only]
//                                             the measurement of results/compression: the models of studies A to D,
//                                             the study A flipbooks coded the same way, and the equal-quality ratios of
//                                             docs/REPORT.md §3 on disk; writes OUT/cm.csv and OUT/cm_equal_quality.csv
//                                             and prints the tables. DATA defaults to $NEURALVFX_DATA/experiments, CSV
//                                             to results/experiments/a_scores.csv, OUT to results/compression.
//                                             --flipbooks-only keeps OUT/cm.csv and adds the flipbooks it lacks (the
//                                             BC7 and ASTC ladders); --threads codes that many flipbooks at a time
//   nvfx_pack --tables [--scores CSV] [--out OUT]
//                                             the tables again from OUT/cm.csv, without coding anything
//   nvfx_pack in.nvfx out.nvfz [--light | --fast] [--lz] [--seekable [--segment N]]
//                                             format 2 (study H): LZ tokens, the light model, seekable segments
//   nvfx_pack --h3 [--reps N] [--segment N] [--out OUT] [--csv NAME] FILES_OR_DIRS
//                                             study H, H3: every configuration of the coder on each file, decode
//                                             times (thread CPU time, interleaved repetitions), one-slice decodes
//
// Packing changes the bytes on disk only: the runtime loads the unpacked .nvfx, so resident memory is unchanged.
// Single-threaded throughout; decoding speed is the unpacked size over the time to unpack.
#include "args.hpp"
#include "baselines.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/cm.hpp>
#include <neuralfx/flipbook.hpp>
#include <neuralfx/sim.hpp>

#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <print>
#include <ranges>
#include <sstream>

namespace fs = std::filesystem;
using namespace nfx;

namespace {

std::vector<std::uint8_t> read_file(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error(std::format("cannot open {}", p.string()));
  return {std::istreambuf_iterator<char>(in), {}};
}

void write_file(const fs::path& p, std::span<const std::uint8_t> b) {
  if (p.has_parent_path()) fs::create_directories(p.parent_path());
  std::ofstream out(p, std::ios::binary);
  out.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
  if (!out) throw std::runtime_error(std::format("cannot write {}", p.string()));
}

std::size_t zlib9(std::span<const std::uint8_t> b) {
  uLongf n = compressBound(static_cast<uLong>(b.size()));
  std::vector<std::uint8_t> out(n);
  if (compress2(out.data(), &n, b.data(), static_cast<uLong>(b.size()), 9) != Z_OK) throw std::runtime_error("zlib failed");
  return n;
}

double kb(std::size_t bytes) { return static_cast<double>(bytes) / 1024.0; }

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// One coded item: its total and its parts (by kind).
struct PartRow {
  std::string part;
  std::size_t values = 0, original = 0, zlib = 0;
  double cm = 0;
};
struct Item {
  std::string set, name, config;
  std::size_t values = 0, original = 0, packed = 0, zlib = 0;
  double decode_s = 0;
  std::vector<PartRow> parts;
};

// A model file: packed, unpacked (checked, timed), and zlib on the whole and on each part.
Item measure_model(const fs::path& path) {
  const auto file = read_file(path);
  Item it;
  it.name = path.filename().string();
  it.original = file.size();
  const cm::Packed p = cm::pack_model(file);
  it.packed = p.data.size();
  const auto t0 = std::chrono::steady_clock::now();
  const auto back = cm::unpack_model(p.data);
  it.decode_s = seconds_since(t0);
  if (!back || *back != file) throw std::runtime_error(std::format("round trip failed for {}", path.string()));
  it.zlib = zlib9(file);
  const auto split = cm::split_model(file);
  for (const cm::Part& q : p.parts) {
    PartRow r{cm::kind_name(q.kind), q.values, q.bytes, 0, q.coded_bytes};
    for (const auto& [k, bytes] : split) {
      if (k == q.kind) r.zlib = zlib9(bytes);
    }
    if (q.kind != cm::Kind::bytes) it.values += q.values;
    it.parts.push_back(r);
  }
  return it;
}

// --- the flipbook baselines, coded the same way ---------------------------------------------------------------------

// The study A clips (tools/nvfx_experiment.cpp, a_clips()): four per effect.
std::vector<std::string> a_clip_names() {
  std::vector<std::string> v;
  for (const auto e : sim::kEffects) {
    for (int k = 0; k < 4; ++k) v.push_back(std::format("{}_{}", sim::effect_name(e), k));
  }
  return v;
}

// The stored form of a flipbook as tensors: 16-byte blocks of any block format (our BC3 layout, BC7, ASTC) as
// [frame][block row][block column][16] (the coder's block kind; its endpoint contexts are BC3's, its contexts of the
// same byte in the neighbouring blocks suit any format), RGBA pixels as [frame][y][x][4], and motion vectors as
// [frame][y][x][2] (quarter pixels, offset by 128), as flipbook.cpp keeps them.
std::vector<cm::Tensor> flipbook_tensors(const Clip& ref, const flipbook::Spec& spec, std::size_t& bytes, flipbook::FrameCache* cache) {
  const flipbook::Flipbook fb = flipbook::build(ref, spec, {}, cache);
  const auto F = static_cast<std::uint32_t>(fb.kept.size()), R = static_cast<std::uint32_t>(spec.res);
  cm::Tensor img;
  img.shape.width = 1;
  img.shape.channels = true;
  for (const auto& stored : fb.stored) img.values.insert(img.values.end(), stored.begin(), stored.end());
  if (spec.codec != flipbook::Codec::raw) {
    const auto B = static_cast<std::uint32_t>((spec.res + flipbook::block_dim(spec.codec) - 1) / flipbook::block_dim(spec.codec));
    img.shape.kind = cm::Kind::bc3;
    img.shape.dims = {F, B, B, 16};
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
  bytes = 0;
  for (const auto& t : out) bytes += t.values.size();
  if (bytes != fb.bytes) throw std::runtime_error("flipbook: stored size does not match flipbook::memory_bytes");
  return out;
}

Item measure_flipbook(const std::string& clip, const Clip& ref, const flipbook::Spec& spec, flipbook::FrameCache* cache = nullptr) {
  Item it;
  it.set = "flipbook";
  it.name = clip;
  it.config = spec.describe();
  const auto tensors = flipbook_tensors(ref, spec, it.original, cache);
  const cm::Packed p = cm::pack_tensors(tensors);
  it.packed = p.data.size();
  const auto t0 = std::chrono::steady_clock::now();
  const auto back = cm::unpack_tensors(p.data);
  it.decode_s = seconds_since(t0);
  if (!back || back->size() != tensors.size()) throw std::runtime_error("flipbook round trip failed");
  std::vector<std::uint8_t> all;
  for (std::size_t k = 0; k < tensors.size(); ++k) {
    if ((*back)[k].values != tensors[k].values) throw std::runtime_error("flipbook round trip failed");
    std::vector<std::uint8_t> b(tensors[k].values.begin(), tensors[k].values.end());
    const cm::Part* q = nullptr;
    for (const auto& pp : p.parts) {
      if (pp.kind == tensors[k].shape.kind) q = &pp;
    }
    it.parts.push_back({cm::kind_name(tensors[k].shape.kind), b.size(), b.size(), zlib9(b), q ? q->coded_bytes : 0.0});
    all.insert(all.end(), b.begin(), b.end());
    it.values += b.size();
  }
  it.zlib = zlib9(all);
  return it;
}

// --- output -----------------------------------------------------------------------------------------------------------

void write_csv(const fs::path& path, const std::vector<Item>& items) {
  fs::create_directories(path.parent_path());
  std::ofstream o(path);
  o << "set,item,config,part,values,original_bytes,cm_bytes,zlib_bytes,decode_mb_per_s\n";
  for (const Item& it : items) {
    o << std::format("{},{},{},total,{},{},{},{},{:.3f}\n", it.set, it.name, it.config, it.values, it.original, it.packed, it.zlib,
                     it.decode_s > 0 ? static_cast<double>(it.original) / 1e6 / it.decode_s : 0.0);
    for (const PartRow& r : it.parts) {
      o << std::format("{},{},{},{},{},{},{:.1f},{},\n", it.set, it.name, it.config, r.part, r.values, r.original, r.cm, r.zlib);
    }
  }
}

// "fire_0_grid_m8.nvfx" -> ("grid_m", 8); "fire_grid_k8.nvfx" -> ("grid_k8", 8); rollout files -> ("rollout", 16).
std::string model_config(const std::string& set, const std::string& file) {
  std::string stem = fs::path(file).stem().string();
  if (set == "model_d") return "rollout";
  if (set == "model_a") {
    // effect_k_config: drop the first two fields
    for (int i = 0; i < 2; ++i) stem = stem.substr(stem.find('_') + 1);
    return stem;
  }
  return stem.substr(stem.find('_') + 1);  // effect_config
}

void print_item_row(const Item& it) {
  std::println("| {} | {} | {:.1f} | {:.1f} | {:.2f}x | {:.2f} | {:.2f}x | {:.2f} |", it.name, it.config, kb(it.original), kb(it.packed),
               static_cast<double>(it.original) / static_cast<double>(it.packed), 8.0 * static_cast<double>(it.packed) / static_cast<double>(it.values),
               static_cast<double>(it.original) / static_cast<double>(it.zlib), static_cast<double>(it.original) / 1e6 / it.decode_s);
}

int report(const fs::path& dir) {
  std::vector<fs::path> files;
  for (const auto& e : fs::recursive_directory_iterator(dir)) {
    if (e.is_regular_file() && e.path().extension() == ".nvfx") files.push_back(e.path());
  }
  std::ranges::sort(files);
  std::println("| file | config | KB | packed KB | ratio | bits per value | zlib -9 ratio | decode MB/s |");
  std::println("|---|---|---:|---:|---:|---:|---:|---:|");
  std::size_t orig = 0, packed = 0, z = 0;
  for (const auto& f : files) {
    Item it = measure_model(f);
    it.config = "";
    print_item_row(it);
    for (const PartRow& r : it.parts) {
      std::println("|  | {} | {:.1f} | {:.1f} | {:.2f}x | {:.2f} | {:.2f}x | |", r.part, kb(r.original), r.cm / 1024.0,
                   static_cast<double>(r.original) / r.cm, 8.0 * r.cm / static_cast<double>(r.values),
                   static_cast<double>(r.original) / static_cast<double>(r.zlib));
    }
    orig += it.original;
    packed += it.packed;
    z += it.zlib;
  }
  if (!files.empty()) {
    std::println("\n{} files, {} round trips exact: {:.1f} KB -> {:.1f} KB ({:.2f}x); zlib -9 {:.2f}x", files.size(), files.size(), kb(orig),
                 kb(packed), static_cast<double>(orig) / static_cast<double>(packed), static_cast<double>(orig) / static_cast<double>(z));
  }
  return 0;
}

// --- the study ----------------------------------------------------------------------------------------------------------

struct Mean {
  double sum = 0;
  int n = 0;
  void add(double v) {
    sum += v;
    ++n;
  }
  double get() const { return n ? sum / n : 0.0; }
};

// Mean active PSNR per configuration of study A (results/experiments/a_scores.csv): flipbooks by their config
// ("bc3 64f 128px"), networks by "grid_m|8".
std::map<std::string, Mean> a_quality(const fs::path& scores) {
  std::ifstream in(scores);
  if (!in) throw std::runtime_error(std::format("cannot open {}", scores.string()));
  std::string line;
  std::getline(in, line);
  std::vector<std::string> cols;
  for (const auto part : std::views::split(line, ',')) cols.emplace_back(std::string_view(part));
  const auto col = [&](std::string_view name) { return static_cast<std::size_t>(std::ranges::find(cols, name) - cols.begin()); };
  const std::size_t c_task = col("task"), c_method = col("method"), c_family = col("family"), c_config = col("config"), c_active = col("active_psnr");
  std::map<std::string, Mean> q;
  while (std::getline(in, line)) {
    std::vector<std::string> v;
    for (const auto part : std::views::split(line, ',')) v.emplace_back(std::string_view(part));
    if (v.size() < cols.size() || v[c_task] != "a") continue;
    const std::string key = v[c_family].starts_with("flipbook") ? v[c_config] : v[c_method].substr(v[c_method].find('|') + 1);
    q[key].add(std::stod(v[c_active]));
  }
  return q;
}

// Flipbook size (KB) for a given quality along the best-flipbook envelope (best quality at or below each size),
// log-linear between neighbours; "<" or ">" when outside it. As in nvfx_experiment's report.
std::pair<std::string, double> envelope_kb(const std::vector<std::pair<double, double>>& env, double quality) {
  if (env.empty()) return {"-", 0};
  if (env.front().second >= quality) return {"<", env.front().first};
  for (std::size_t i = 1; i < env.size(); ++i) {
    if (env[i].second >= quality && env[i - 1].second < quality) {
      const double u = (quality - env[i - 1].second) / (env[i].second - env[i - 1].second);
      return {"", std::exp(std::log(env[i - 1].first) + u * (std::log(env[i].first) - std::log(env[i - 1].first)))};
    }
  }
  return {">", env.back().first};
}

int tables(const fs::path& out, const fs::path& scores);

std::vector<Item> read_csv(const fs::path& path);

int study(const tools::Args& a) {
  const fs::path data = a.has("data") ? fs::path(a.str("data")) : data_root() / "experiments";
  const fs::path scores = a.str("scores", "results/experiments/a_scores.csv");
  const fs::path out = a.str("out", "results/compression");
  const int threads = a.i("threads", 1);
  // --flipbooks-only: keep cm.csv as it is and add the flipbooks it lacks (the production formats, when added).
  const bool flipbooks_only = a.flag("flipbooks-only");
  std::vector<Item> items;
  if (flipbooks_only) items = read_csv(out / "cm.csv");
  // Models of every study.
  for (const std::string g : {"a", "b", "c", "d"}) {
    if (flipbooks_only) break;
    const fs::path dir = data / "models" / g;
    if (!fs::exists(dir)) continue;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir)) {
      if (e.path().extension() == ".nvfx") files.push_back(e.path());
    }
    std::ranges::sort(files);
    for (const auto& f : files) {
      Item it = measure_model(f);
      it.set = "model_" + g;
      it.config = model_config(it.set, it.name);
      std::println(stderr, "{} {}: {} -> {} ({:.2f}x), zlib {:.2f}x, decode {:.2f} MB/s", it.set, it.name, it.original, it.packed,
                   static_cast<double>(it.original) / static_cast<double>(it.packed), static_cast<double>(it.original) / static_cast<double>(it.zlib),
                   static_cast<double>(it.original) / 1e6 / it.decode_s);
      items.push_back(std::move(it));
    }
  }
  // The study A flipbook ladder (tools/baselines.hpp: our BC3 layout and raw, then BC7 and ASTC when this build has
  // them), every clip; --threads flipbooks at a time (their decode speeds are then measured side by side).
  for (const std::string& clip : a_clip_names()) {
    const auto ref = read_clip(data / "clips" / "a" / (clip + ".nfxclip"));
    if (!ref) throw std::runtime_error(ref.error());
    std::vector<flipbook::Spec> todo;
    for (const auto& spec : tools::flipbook_ladder(ref->size, ref->frames)) {
      const bool have = std::ranges::any_of(items, [&](const Item& it) { return it.set == "flipbook" && it.name == clip && it.config == spec.describe(); });
      if (!have) todo.push_back(spec);
    }
    std::vector<Item> done(todo.size());
    flipbook::FrameCache cache;
    tools::parallel_for(todo.size(), threads, [&](std::size_t i) { done[i] = measure_flipbook(clip, *ref, todo[i], &cache); });
    for (Item& it : done) {
      std::println(stderr, "flipbook {} {}: {} -> {} ({:.2f}x), zlib {:.2f}x", clip, it.config, it.original, it.packed,
                   static_cast<double>(it.original) / static_cast<double>(it.packed), static_cast<double>(it.original) / static_cast<double>(it.zlib));
      items.push_back(std::move(it));
    }
    if (!todo.empty()) write_csv(out / "cm.csv", items);  // a stopped run keeps what it measured
  }
  write_csv(out / "cm.csv", items);
  return tables(out, scores);
}

// Items back from cm.csv (the tables can be made again without coding anything).
std::vector<Item> read_csv(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error(std::format("cannot open {}", path.string()));
  std::string line;
  std::getline(in, line);
  std::vector<Item> items;
  while (std::getline(in, line)) {
    std::vector<std::string> v;
    for (const auto part : std::views::split(line, ',')) v.emplace_back(std::string_view(part));
    if (v.size() < 8) continue;
    if (v[3] == "total") {
      Item it;
      it.set = v[0];
      it.name = v[1];
      it.config = v[2];
      it.values = std::stoull(v[4]);
      it.original = std::stoull(v[5]);
      it.packed = std::stoull(v[6]);
      it.zlib = std::stoull(v[7]);
      const double mbs = v.size() > 8 && !v[8].empty() ? std::stod(v[8]) : 0.0;
      it.decode_s = mbs > 0 ? static_cast<double>(it.original) / 1e6 / mbs : 0.0;
      items.push_back(std::move(it));
    } else if (!items.empty()) {
      items.back().parts.push_back({v[3], std::stoull(v[4]), std::stoull(v[5]), std::stoull(v[7]), std::stod(v[6])});
    }
  }
  return items;
}

int tables(const fs::path& out, const fs::path& scores) {
  const std::vector<Item> items = read_csv(out / "cm.csv");
  // Tables. Models by set and configuration (means over the files of each).
  const auto ratio = [](double x, double y) { return y > 0 ? x / y : 0.0; };
  struct Agg {
    Mean orig, packed, zlib, values, mbs;
    std::map<std::string, std::array<Mean, 3>> parts;  // part -> original, cm, zlib
    int n = 0;
  };
  std::map<std::pair<std::string, std::string>, Agg> agg;
  for (const Item& it : items) {
    Agg& g = agg[{it.set, it.config}];
    g.orig.add(static_cast<double>(it.original));
    g.packed.add(static_cast<double>(it.packed));
    g.zlib.add(static_cast<double>(it.zlib));
    g.values.add(static_cast<double>(it.values));
    g.mbs.add(static_cast<double>(it.original) / 1e6 / it.decode_s);
    for (const PartRow& r : it.parts) {
      auto& p = g.parts[r.part];
      p[0].add(static_cast<double>(r.original));
      p[1].add(r.cm);
      p[2].add(static_cast<double>(r.zlib));
    }
    ++g.n;
  }
  std::println("\n### Models (means over the files of each configuration)\n");
  std::println("| set | config | files | KB | packed KB | ratio | bits per value (stored / packed) | zlib -9 ratio | decode MB/s |");
  std::println("|---|---|---:|---:|---:|---:|---:|---:|---:|");
  for (const auto& [key, g] : agg) {
    if (key.first == "flipbook") continue;
    std::println("| {} | {} | {} | {:.1f} | {:.1f} | {:.2f}x | {:.2f} / {:.2f} | {:.2f}x | {:.2f} |", key.first.substr(6), key.second, g.n, g.orig.get() / 1024,
                 g.packed.get() / 1024, ratio(g.orig.get(), g.packed.get()), 8 * g.orig.get() / g.values.get(), 8 * g.packed.get() / g.values.get(),
                 ratio(g.orig.get(), g.zlib.get()), g.mbs.get());
  }
  std::println("\n### Models by part (means per file)\n");
  std::println("| set | config | part | KB | packed KB | ratio | zlib -9 ratio |");
  std::println("|---|---|---|---:|---:|---:|---:|");
  for (const auto& [key, g] : agg) {
    if (key.first == "flipbook") continue;
    for (const auto& [part, m] : g.parts) {
      std::println("| {} | {} | {} | {:.1f} | {:.1f} | {:.2f}x | {:.2f}x |", key.first.substr(6), key.second, part, m[0].get() / 1024, m[1].get() / 1024,
                   ratio(m[0].get(), m[1].get()), ratio(m[0].get(), m[2].get()));
    }
  }
  std::println("\n### Flipbooks (means over the 12 study A clips)\n");
  std::println("| config | KB | packed KB | ratio | zlib -9 KB | zlib ratio | decode MB/s |");
  std::println("|---|---:|---:|---:|---:|---:|---:|");
  std::vector<std::pair<std::string, const Agg*>> fbs;
  for (const auto& [key, g] : agg) {
    if (key.first == "flipbook") fbs.emplace_back(key.second, &g);
  }
  std::ranges::sort(fbs, [](const auto& x, const auto& y) { return x.second->orig.get() > y.second->orig.get(); });
  for (const auto& [config, g] : fbs) {
    std::println("| {} | {:.0f} | {:.1f} | {:.2f}x | {:.1f} | {:.2f}x | {:.2f} |", config, g->orig.get() / 1024, g->packed.get() / 1024,
                 ratio(g->orig.get(), g->packed.get()), g->zlib.get() / 1024, ratio(g->orig.get(), g->zlib.get()), g->mbs.get());
  }

  // Equal quality on disk. Quality: mean active PSNR over the 12 clips (study A scores). Size: in memory (as the
  // report), coded by this coder, and by zlib -9, each side coded the same way.
  const auto quality = a_quality(scores);
  struct Side {
    double mem = 0, cm = 0, zlib = 0;
  };
  std::map<std::string, Side> fsize;  // flipbook config -> mean sizes in KB
  for (const auto& [config, g] : fbs) fsize[config] = {g->orig.get() / 1024, g->packed.get() / 1024, g->zlib.get() / 1024};
  // One envelope per flipbook baseline (tools/baselines.hpp): the original ladder (our BC3 layout and raw), with BC7,
  // with BC7 and ASTC.
  using tools::Baseline;
  const auto envelope = [&](double Side::*field, Baseline b) {
    std::vector<std::pair<double, double>> pts;
    for (const auto& [config, s] : fsize) {
      if (quality.contains(config) && tools::in_baseline(config, b)) pts.emplace_back(s.*field, quality.at(config).get());
    }
    std::ranges::sort(pts);
    std::vector<std::pair<double, double>> env;  // one point per size: the best quality at or below it
    for (const auto& [kb, q] : pts) {
      const double best = env.empty() ? q : std::max(env.back().second, q);
      if (!env.empty() && env.back().first == kb) env.back().second = best;
      else env.emplace_back(kb, best);
    }
    return env;
  };
  std::ofstream eq(out / "cm_equal_quality.csv");
  eq << "network,files,active_psnr,net_kb,net_cm_kb,net_zlib_kb,flipbook_kb,flipbook_cm_kb,flipbook_zlib_kb,ratio_memory,ratio_cm,ratio_zlib,baseline\n";
  for (const Baseline b : {Baseline::bc3_layout, Baseline::desktop, Baseline::all}) {
    const bool more = std::ranges::any_of(fsize, [&](const auto& kv) { return quality.contains(kv.first) && !tools::in_baseline(kv.first, Baseline::bc3_layout) &&
                                                                              tools::in_baseline(kv.first, b); });
    if (b != Baseline::bc3_layout && !more) continue;
    const auto env_mem = envelope(&Side::mem, b), env_cm = envelope(&Side::cm, b), env_z = envelope(&Side::zlib, b);
    std::println("\n### Equal quality, flipbooks {}: flipbook size for the network's mean active PSNR, and the ratio\n", tools::baseline_name(b));
    std::println("| network | files | active PSNR | in memory: net / flipbook KB, ratio | packed (this coder): net / flipbook KB, ratio | zlib -9: net / flipbook KB, ratio |");
    std::println("|---|---:|---:|---|---|---|");
    for (const auto& [key, g] : agg) {
      if (key.first != "model_a") continue;
      // "grid_m8" -> "grid_m|8"
      const std::string& c = key.second;
      const std::size_t digits = c.find_first_of("0123456789");
      if (digits == std::string::npos) continue;
      const std::string qkey = c.substr(0, digits) + "|" + c.substr(digits);
      if (!quality.contains(qkey)) continue;
      const double q = quality.at(qkey).get();
      // The networks: every saved file of this configuration (grid_m 8-bit: all 12 clips; the others: clip 0 of
      // each effect), sizes as stored, packed and zlib.
      const Side net{g.orig.get() / 1024, g.packed.get() / 1024, g.zlib.get() / 1024};
      const auto [om, fm] = envelope_kb(env_mem, q);
      const auto [oc, fc] = envelope_kb(env_cm, q);
      const auto [oz, fz] = envelope_kb(env_z, q);
      const auto cell = [](double n, const std::string& o, double f) { return std::format("{:.0f} / {}{:.0f}, {}{:.1f}x", n, o, f, o, f / n); };
      std::println("| {} | {} | {:.2f} | {} | {} | {} |", qkey, g.n, q, cell(net.mem, om, fm), cell(net.cm, oc, fc), cell(net.zlib, oz, fz));
      eq << std::format("{},{},{:.3f},{:.1f},{:.1f},{:.1f},{}{:.1f},{}{:.1f},{}{:.1f},{:.3f},{:.3f},{:.3f},{}\n", qkey, g.n, q, net.mem, net.cm, net.zlib, om, fm, oc, fc,
                        oz, fz, fm / net.mem, fc / net.cm, fz / net.zlib, tools::baseline_key(b));
    }
    std::println("\n(\"<\": every flipbook in the ladder is at least as good, so the ratio is at most that; \">\": none up to the largest is as good.)");
    std::println("\nFlipbook envelope (KB in memory, packed, zlib; best mean active PSNR at or below): ");
    for (const auto* e : {&env_mem, &env_cm, &env_z}) {
      std::string s;
      for (const auto& [kb, qq] : *e) s += std::format(" {:.0f}:{:.2f}", kb, qq);
      std::println("  {}", s);
    }
  }
  return 0;
}

// --- study H, H3: LZ tokens and a light model in the coder (docs/DCM.md §9) ---------------------------------------------

double thread_seconds() {
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return static_cast<double>(t.tv_sec) + 1e-9 * static_cast<double>(t.tv_nsec);
}

struct H3Config {
  std::string name;
  int kind;  // 0: format 1; 1: format 2 with options; 2: zlib -9
  cm::Options o;
};

std::vector<H3Config> h3_configs(std::size_t segment) {
  std::vector<H3Config> c;
  c.push_back({"cm", 0, {}});
  using L = cm::Literal;
  c.push_back({"cm+lz", 1, {L::full, true, false, segment}});
  c.push_back({"light", 1, {L::light, false, false, segment}});
  c.push_back({"light+lz", 1, {L::light, true, false, segment}});
  c.push_back({"light+lz seekable", 1, {L::light, true, true, segment}});
  c.push_back({"fast", 1, {L::fast, false, false, segment}});
  c.push_back({"fast+lz", 1, {L::fast, true, false, segment}});
  c.push_back({"fast+lz seekable", 1, {L::fast, true, true, segment}});
  c.push_back({"zlib-9", 2, {}});
  return c;
}

std::vector<std::uint8_t> zlib_pack(std::span<const std::uint8_t> b) {
  uLongf n = compressBound(static_cast<uLong>(b.size()));
  std::vector<std::uint8_t> out(n);
  if (compress2(out.data(), &n, b.data(), static_cast<uLong>(b.size()), 9) != Z_OK) throw std::runtime_error("zlib failed");
  out.resize(n);
  return out;
}

std::vector<std::uint8_t> zlib_unpack(std::span<const std::uint8_t> b, std::size_t size) {
  std::vector<std::uint8_t> out(size);
  uLongf n = static_cast<uLongf>(size);
  if (uncompress(out.data(), &n, b.data(), static_cast<uLong>(b.size())) != Z_OK || n != size) throw std::runtime_error("zlib failed");
  return out;
}

// Every file under the arguments packed in every configuration (round trips checked), then decoded `reps` times with
// the configurations interleaved within each repetition; thread CPU time, least and median. For seekable files, each
// segment is also decoded alone (least of the repetitions per segment; the mean and largest over segments are kept).
int h3(const tools::Args& a) {
  const int reps = a.i("reps", 15);
  const std::size_t segment = static_cast<std::size_t>(a.i("segment", 1 << 16));
  const fs::path out = a.str("out", "results/compression");
  std::vector<fs::path> files;
  for (const auto& p : a.positional()) {
    if (fs::is_directory(p)) {
      for (const auto& e : fs::recursive_directory_iterator(p)) {
        if (e.is_regular_file() && e.path().extension() == ".nvfx") files.push_back(e.path());
      }
    } else {
      files.push_back(p);
    }
  }
  std::ranges::sort(files);
  const auto configs = h3_configs(segment);
  fs::create_directories(out);
  const std::string name = a.str("csv", "h3_decode.csv");
  std::ofstream csv(out / name);
  std::ofstream parts(out / (fs::path(name).stem().string() + "_parts.csv"));
  parts << "file,config,part,values,original_bytes,coded_bytes\n";
  csv << "file,config,original_bytes,packed_bytes,ratio,decode_ms_min,decode_ms_median,decode_mb_per_s,segments,slice_ms_mean,slice_ms_max,"
         "slice_values_max\n";
  std::println("| file | config | KB | packed KB | ratio | decode ms (least) | median | MB/s | segments | one slice ms (mean / max) |");
  std::println("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|");
  for (const auto& f : files) {
    const auto file = read_file(f);
    std::vector<std::vector<std::uint8_t>> packed;
    for (const auto& c : configs) {
      if (c.kind == 2) {
        packed.push_back(zlib_pack(file));
        continue;
      }
      const cm::Packed p = c.kind == 0 ? cm::pack_model(file) : cm::pack_model(file, c.o);
      for (const cm::Part& q : p.parts) {
        parts << std::format("{},{},{},{},{},{:.1f}\n", f.filename().string(), c.name, cm::kind_name(q.kind), q.values, q.bytes, q.coded_bytes);
      }
      packed.push_back(p.data);
    }
    std::vector<std::vector<double>> t(configs.size());
    for (int r = 0; r < reps; ++r) {
      for (std::size_t k = 0; k < configs.size(); ++k) {
        const double t0 = thread_seconds();
        std::vector<std::uint8_t> back;
        if (configs[k].kind == 2) {
          back = zlib_unpack(packed[k], file.size());
        } else {
          auto u = cm::unpack_model(packed[k]);
          if (!u) throw std::runtime_error(std::format("{} {}: {}", f.string(), configs[k].name, u.error()));
          back = std::move(*u);
        }
        t[k].push_back(thread_seconds() - t0);
        if (back != file) throw std::runtime_error(std::format("round trip failed: {} {}", f.string(), configs[k].name));
      }
    }
    for (std::size_t k = 0; k < configs.size(); ++k) {
      auto v = t[k];
      std::ranges::sort(v);
      const double least = v.front(), median = v[v.size() / 2];
      std::size_t nseg = 0, max_values = 0;
      double slice_mean = 0, slice_max = 0;
      if (configs[k].kind == 1 && configs[k].o.seekable) {
        const auto list = cm::list_slices(packed[k]);
        if (!list) throw std::runtime_error(list.error());
        nseg = list->size();
        for (std::size_t j = 0; j < list->size(); ++j) {
          double best = 1e9;
          for (int r = 0; r < std::max(3, reps / 3); ++r) {
            const double t0 = thread_seconds();
            const auto sl = cm::unpack_slice(packed[k], j);
            best = std::min(best, thread_seconds() - t0);
            if (!sl) throw std::runtime_error(sl.error());
            for (std::size_t q = 0; q < sl->at.size(); ++q) {
              const int w = sl->tensor.shape.width;
              const auto want = static_cast<std::uint16_t>(file[sl->at[q]] | (w == 2 ? file[sl->at[q] + 1] << 8 : 0));
              if (sl->tensor.values[q] != want) throw std::runtime_error("slice differs from the file");
            }
          }
          slice_mean += best;
          slice_max = std::max(slice_max, best);
          max_values = std::max(max_values, (*list)[j].values);
        }
        if (nseg) slice_mean /= static_cast<double>(nseg);
      }
      const double ratio = static_cast<double>(file.size()) / static_cast<double>(packed[k].size());
      csv << std::format("{},{},{},{},{:.4f},{:.3f},{:.3f},{:.3f},{},{:.3f},{:.3f},{}\n", f.filename().string(), configs[k].name, file.size(),
                         packed[k].size(), ratio, 1e3 * least, 1e3 * median, static_cast<double>(file.size()) / 1e6 / least, nseg, 1e3 * slice_mean,
                         1e3 * slice_max, max_values);
      std::println("| {} | {} | {:.1f} | {:.1f} | {:.3f}x | {:.2f} | {:.2f} | {:.2f} | {} | {} |", f.filename().string(), configs[k].name, kb(file.size()),
                   kb(packed[k].size()), ratio, 1e3 * least, 1e3 * median, static_cast<double>(file.size()) / 1e6 / least, nseg,
                   nseg ? std::format("{:.2f} / {:.2f}", 1e3 * slice_mean, 1e3 * slice_max) : std::string());
    }
    csv.flush();
    parts.flush();
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"help", "unpack", "study", "tables", "h3", "light", "fast", "lz", "seekable", "flipbooks-only"});
  const auto& pos = a.positional();
  if (a.flag("help") || (pos.empty() && !a.has("report") && !a.flag("study") && !a.flag("tables"))) {
    std::println("nvfx_pack in.nvfx out.nvfz [--light | --fast] [--lz] [--seekable [--segment N]] | --unpack in.nvfz out.nvfx | --report DIR | "
                 "--study [--data DIR] [--scores CSV] [--out DIR] [--threads N] [--flipbooks-only] | --tables [--scores CSV] [--out DIR] | "
                 "--h3 [--reps N] [--segment N] [--out DIR] [--csv NAME] FILES_OR_DIRS");
    return 0;
  }
  if (a.flag("h3")) {
    const int r = h3(a);
    a.warn_unused();
    return r;
  }
  if (a.flag("tables")) return tables(a.str("out", "results/compression"), a.str("scores", "results/experiments/a_scores.csv"));
  if (a.has("report")) return report(a.str("report"));
  if (a.flag("study")) {
    const int r = study(a);
    a.warn_unused();
    return r;
  }
  if (pos.size() != 2) throw std::invalid_argument("expected an input and an output file");
  const auto in = read_file(pos[0]);
  if (a.flag("unpack")) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = cm::unpack_model(in);
    if (!r) throw std::runtime_error(r.error());
    const double s = seconds_since(t0);
    write_file(pos[1], *r);
    std::println("{}: {} bytes -> {} bytes in {:.3f} s ({:.2f} MB/s)", pos[0], in.size(), r->size(), s, static_cast<double>(r->size()) / 1e6 / s);
    return 0;
  }
  const bool format2 = a.flag("light") || a.flag("fast") || a.flag("lz") || a.flag("seekable");
  const cm::Literal lit = a.flag("fast") ? cm::Literal::fast : a.flag("light") ? cm::Literal::light : cm::Literal::full;
  const cm::Options o{lit, a.flag("lz"), a.flag("seekable"), static_cast<std::size_t>(a.i("segment", 1 << 16))};
  const cm::Packed p = format2 ? cm::pack_model(in, o) : cm::pack_model(in);
  write_file(pos[1], p.data);
  std::println("{}: {} bytes -> {} bytes ({:.3f}x)", pos[0], in.size(), p.data.size(), static_cast<double>(in.size()) / static_cast<double>(p.data.size()));
  for (const cm::Part& q : p.parts) {
    std::println("  {:<15} {:>9} bytes -> {:>11.1f} ({:.3f}x, {:.2f} bits per value)", cm::kind_name(q.kind), q.bytes, q.coded_bytes,
                 static_cast<double>(q.bytes) / q.coded_bytes, 8.0 * q.coded_bytes / static_cast<double>(q.values));
  }
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_pack: {}", e.what());
  return 1;
}
