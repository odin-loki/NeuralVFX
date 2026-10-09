// nvfx_pack: lossless context-mixing compression of .nvfx files (include/neuralfx/cm.hpp), and its measurement.
//
//   nvfx_pack in.nvfx out.nvfz                pack a frame model or a rollout effect
//   nvfx_pack --unpack in.nvfz out.nvfx       restore the exact bytes
//   nvfx_pack --report DIR                    every .nvfx under DIR: sizes, ratio, bits per stored value, zlib -9 as a
//                                             general-purpose reference, decoding speed; every round trip is checked
//   nvfx_pack --study [--data DATA] [--scores CSV] [--out OUT]
//                                             the measurement of results/compression: the models of studies A to D,
//                                             the study A flipbooks coded the same way, and the equal-quality ratios of
//                                             docs/REPORT.md §3 on disk; writes OUT/cm.csv and OUT/cm_equal_quality.csv
//                                             and prints the tables. DATA defaults to $NEURALVFX_DATA/experiments, CSV
//                                             to results/experiments/a_scores.csv, OUT to results/compression
//   nvfx_pack --tables [--scores CSV] [--out OUT]
//                                             the tables again from OUT/cm.csv, without coding anything
//
// Packing changes the bytes on disk only: the runtime loads the unpacked .nvfx, so resident memory is unchanged.
// Single-threaded throughout; decoding speed is the unpacked size over the time to unpack.
#include "args.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/cm.hpp>
#include <neuralfx/flipbook.hpp>
#include <neuralfx/sim.hpp>

#include <zlib.h>

#include <algorithm>
#include <chrono>
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

// The stored form of a flipbook as tensors: BC3 blocks [frame][block row][block column][16] or RGBA pixels
// [frame][y][x][4], and motion vectors [frame][y][x][2] (quarter pixels, offset by 128), as flipbook.cpp keeps them.
std::vector<cm::Tensor> flipbook_tensors(const Clip& ref, const flipbook::Spec& spec, std::size_t& bytes) {
  const flipbook::Flipbook fb = flipbook::build(ref, spec);
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
  bytes = 0;
  for (const auto& t : out) bytes += t.values.size();
  if (bytes != fb.bytes) throw std::runtime_error("flipbook: stored size does not match flipbook::memory_bytes");
  return out;
}

Item measure_flipbook(const std::string& clip, const Clip& ref, const flipbook::Spec& spec) {
  Item it;
  it.set = "flipbook";
  it.name = clip;
  it.config = spec.describe();
  const auto tensors = flipbook_tensors(ref, spec, it.original);
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

int study(const tools::Args& a) {
  const fs::path data = a.has("data") ? fs::path(a.str("data")) : data_root() / "experiments";
  const fs::path scores = a.str("scores", "results/experiments/a_scores.csv");
  const fs::path out = a.str("out", "results/compression");
  std::vector<Item> items;
  // Models of every study.
  for (const std::string g : {"a", "b", "c", "d"}) {
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
  // The study A flipbook ladder, every clip.
  for (const std::string& clip : a_clip_names()) {
    const auto ref = read_clip(data / "clips" / "a" / (clip + ".nfxclip"));
    if (!ref) throw std::runtime_error(ref.error());
    for (const auto& spec : flipbook::ladder(ref->size, ref->frames)) {
      items.push_back(measure_flipbook(clip, *ref, spec));
      const Item& it = items.back();
      std::println(stderr, "flipbook {} {}: {} -> {} ({:.2f}x), zlib {:.2f}x", clip, it.config, it.original, it.packed,
                   static_cast<double>(it.original) / static_cast<double>(it.packed), static_cast<double>(it.original) / static_cast<double>(it.zlib));
    }
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
  const auto envelope = [&](double Side::*field) {
    std::vector<std::pair<double, double>> pts;
    for (const auto& [config, s] : fsize) {
      if (quality.contains(config)) pts.emplace_back(s.*field, quality.at(config).get());
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
  const auto env_mem = envelope(&Side::mem), env_cm = envelope(&Side::cm), env_z = envelope(&Side::zlib);
  std::ofstream eq(out / "cm_equal_quality.csv");
  eq << "network,files,active_psnr,net_kb,net_cm_kb,net_zlib_kb,flipbook_kb,flipbook_cm_kb,flipbook_zlib_kb,ratio_memory,ratio_cm,ratio_zlib\n";
  std::println("\n### Equal quality: flipbook size for the network's mean active PSNR, and the ratio\n");
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
    eq << std::format("{},{},{:.3f},{:.1f},{:.1f},{:.1f},{}{:.1f},{}{:.1f},{}{:.1f},{:.3f},{:.3f},{:.3f}\n", qkey, g.n, q, net.mem, net.cm, net.zlib, om, fm, oc, fc, oz,
                      fz, fm / net.mem, fc / net.cm, fz / net.zlib);
  }
  std::println("\n(\"<\": every flipbook in the ladder is at least as good, so the ratio is at most that; \">\": none up to the largest is as good.)");
  std::println("\nFlipbook envelope (KB in memory, packed, zlib; best mean active PSNR at or below): ");
  for (const auto* e : {&env_mem, &env_cm, &env_z}) {
    std::string s;
    for (const auto& [kb, qq] : *e) s += std::format(" {:.0f}:{:.2f}", kb, qq);
    std::println("  {}", s);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"help", "unpack", "study", "tables"});
  const auto& pos = a.positional();
  if (a.flag("help") || (pos.empty() && !a.has("report") && !a.flag("study") && !a.flag("tables"))) {
    std::println("nvfx_pack in.nvfx out.nvfz | --unpack in.nvfz out.nvfx | --report DIR | --study [--data DIR] [--scores CSV] [--out DIR] | "
                 "--tables [--scores CSV] [--out DIR]");
    return 0;
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
  const cm::Packed p = cm::pack_model(in);
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
