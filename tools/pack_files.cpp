// nvfx_pack's modes for effect packs (docs/INTEGRATION.md §5):
//
//   nvfx_pack pack OUT.nvfxp --library DIR [--models DIR] [--all-models] [--add FILE,...] [--licence FILE,...]
//                  [--costs CSV | --calibrate [--calibrate-frames 24]] [--coder fast] [--threads 1] [--require-models]
//       the catalogue (every *.nvfxl in DIR), the models its entries need (from --models, default DIR; --all-models:
//       every *.nvfx there), scene scripts or other files (--add), licence or credit text (--licence) and default costs
//       for the effect manager (--costs: rows model,size,step_ms,render_ms; --calibrate: measured on this machine).
//       Without --library: every *.nvfx in --models. --coder: store, full, light or fast (the context-mixing coder's
//       models; a member is coded only when that saves 5% and 256 bytes). Exit 77 with --require-models when a model
//       or a file to --add is missing.
//   nvfx_pack list PACK [--manifest]          members, entries and costs (--manifest: the manifest as written)
//   nvfx_pack verify PACK [--threads N] [--skip-missing]
//                                             every check an engine's load makes: the pack, its catalogue as a
//                                             library, every model loaded (--skip-missing: exit 77 when PACK is not
//                                             there, as for a test whose pack was not built without the models)
//   nvfx_pack extract PACK DIR                every member into DIR, byte for byte
//   nvfx_pack measure --library DIR --models DIR [--reps 7] [--threads 3] [--out results/pack]
//                                             each coder's pack size and loading time against the loose files
//                                             (OUT/pack_measure.csv, OUT/pack_members.csv)
#include "pack_files.hpp"

#include "pack.hpp"

#include <neuralfx/nvfx_manager.h>
#include <neuralfx/nvfx_pack.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <print>
#include <set>
#include <sstream>

namespace nfx::tools {

namespace {

namespace fs = std::filesystem;
namespace pk = nfx::pack;
using Clock = std::chrono::steady_clock;

struct Missing : std::runtime_error {
  using std::runtime_error::runtime_error;
};

std::vector<std::uint8_t> read_bytes(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error(std::format("cannot read {}", p.string()));
  return {std::istreambuf_iterator<char>(in), {}};
}

void write_bytes(const fs::path& p, std::span<const std::uint8_t> b) {
  if (p.has_parent_path()) fs::create_directories(p.parent_path());
  std::ofstream out(p, std::ios::binary);
  out.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
  if (!out) throw std::runtime_error(std::format("cannot write {}", p.string()));
}

std::vector<std::string> list_arg(const Args& a, const std::string& k) {
  std::vector<std::string> v;
  if (!a.has(k)) return v;
  std::stringstream ss(a.str(k));
  for (std::string x; std::getline(ss, x, ',');)
    if (!x.empty()) v.push_back(x);
  return v;
}

std::vector<fs::path> files_with(const fs::path& dir, const std::string& ext) {
  std::vector<fs::path> v;
  for (const auto& de : fs::directory_iterator(dir))
    if (de.is_regular_file() && de.path().extension() == ext) v.push_back(de.path());
  std::ranges::sort(v);
  return v;
}

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// What goes into a pack: the members in order, the entries' needs and the loose files' total size.
struct Contents {
  std::vector<pk::Input> members;
  std::vector<pk::Need> needs;
  std::size_t loose_bytes = 0;
};

Contents collect(const Args& a) {
  Contents c;
  const fs::path library = a.str("library");
  const fs::path models = a.has("models") ? fs::path(a.str("models")) : library;
  if (library.empty() && models.empty()) throw std::invalid_argument("pack needs --library DIR or --models DIR");
  std::set<std::string> model_files;
  std::vector<pk::Input> catalogue;
  if (!library.empty()) {
    for (const fs::path& p : files_with(library, ".nvfxl")) catalogue.push_back({p.filename().string(), pk::Kind::catalogue, read_bytes(p)});
    if (catalogue.empty()) throw std::runtime_error(std::format("no catalogue files (*.nvfxl) in {}", library.string()));
    std::vector<std::string> texts;
    std::vector<nvfx_library_file> files;
    for (const pk::Input& in : catalogue) texts.emplace_back(in.data.begin(), in.data.end());
    for (std::size_t i = 0; i < catalogue.size(); ++i) files.push_back({catalogue[i].name.c_str(), texts[i].c_str()});
    nvfx_library_desc d;
    nvfx_library_desc_init(&d);
    d.files = files.data();
    d.n_files = static_cast<int>(files.size());
    nvfx_library* lib = nullptr;
    nvfx_scene_error err;
    if (nvfx_library_open(&d, &lib, &err) != NVFX_OK) throw std::runtime_error(err.message);
    for (int i = 0; i < nvfx_library_count(lib); ++i) {
      nvfx_library_effect e;
      nvfx_library_get_effect(lib, i, &e);
      pk::Need n{e.name, {}};
      nvfx_library_list_models(lib, i, [](void* u, const char*, const char* f) { static_cast<pk::Need*>(u)->models.emplace_back(f); }, &n);
      for (const std::string& f : n.models) model_files.insert(f);
      c.needs.push_back(std::move(n));
    }
    nvfx_library_free(lib);
  }
  if (library.empty() || a.flag("all-models"))
    for (const fs::path& p : files_with(models, ".nvfx")) model_files.insert(p.filename().string());
  for (const std::string& f : model_files) {
    const fs::path p = models / f;
    if (!fs::exists(p)) throw Missing(std::format("{} is not in {} (--models DIR)", f, models.string()));
    pk::Input in{f, pk::Kind::model, read_bytes(p)};
    nvfx_effect* e = nullptr;
    if (const nvfx_status st = nvfx_effect_load_memory(in.data.data(), in.data.size(), &e); st != NVFX_OK)
      throw std::runtime_error(std::format("{}: {}", p.string(), nvfx_status_string(st)));
    nvfx_effect_free(e);
    c.members.push_back(std::move(in));
  }
  for (pk::Input& in : catalogue) c.members.push_back(std::move(in));
  for (const std::string& f : list_arg(a, "add")) {
    if (!fs::exists(f)) throw Missing(std::format("{} is not there (--add)", f));  // a 3D model, say, outside git
    c.members.push_back({fs::path(f).filename().string(), pk::kind_of_file(f), read_bytes(f)});
  }
  for (const std::string& f : list_arg(a, "licence")) c.members.push_back({fs::path(f).filename().string(), pk::Kind::licence, read_bytes(f)});
  for (const pk::Input& in : c.members) c.loose_bytes += in.data.size();
  return c;
}

std::vector<pk::Cost> costs_from_csv(const fs::path& p, const std::vector<pk::Input>& members) {
  std::ifstream in(p);
  if (!in) throw std::runtime_error(std::format("cannot read {}", p.string()));
  std::vector<pk::Cost> out;
  for (std::string line; std::getline(in, line);) {
    std::stringstream ss(line);
    std::vector<std::string> f;
    for (std::string x; std::getline(ss, x, ',');) f.push_back(x);
    if (f.size() < 4 || f[1].empty() || !std::isdigit(static_cast<unsigned char>(f[1][0]))) continue;  // a header
    std::string model = f[0];
    if (!model.ends_with(".nvfx")) model += ".nvfx";
    if (std::ranges::none_of(members, [&](const pk::Input& m) { return m.name == model; })) throw std::runtime_error(std::format("{}: {} is not a model of the pack", p.string(), f[0]));
    out.push_back({model, std::stoi(f[1]), std::stof(f[2]), std::stof(f[3])});
  }
  return out;
}

std::vector<pk::Cost> calibrate(const std::vector<pk::Input>& members, int frames) {
  std::vector<pk::Cost> out;
  for (const pk::Input& in : members) {
    if (in.kind != pk::Kind::model) continue;
    nvfx_effect* e = nullptr;
    nvfx_effect_load_memory(in.data.data(), in.data.size(), &e);
    nvfx_effect_info info{};
    nvfx_effect_get_info(e, &info);
    if (info.arch == 3) {
      nvfx_manager_desc d;
      nvfx_manager_desc_init(&d);
      nvfx_manager* m = nullptr;
      int model = 0;
      if (nvfx_manager_create(&d, &m) != NVFX_OK || nvfx_manager_add_model(m, e, nullptr, &model) != NVFX_OK || nvfx_manager_calibrate(m, model, frames) != NVFX_OK)
        throw std::runtime_error(std::format("cannot calibrate {}", in.name));
      for (const int size : {32, 64, 128, 256}) {
        pk::Cost c{in.name, size, 0.f, 0.f};
        nvfx_manager_get_cost(m, model, size, &c.step_ms, &c.render_ms);
        out.push_back(c);
      }
      nvfx_manager_free(m);
      std::println("  calibrated {}", in.name);
    }
    nvfx_effect_free(e);
  }
  return out;
}

std::string ratio(std::uint64_t a, std::uint64_t b) { return b ? std::format("{:.3f}", static_cast<double>(a) / static_cast<double>(b)) : "-"; }

int do_pack(const Args& a) {
  const auto& pos = a.positional();
  if (pos.size() != 2) throw std::invalid_argument("pack OUT.nvfxp --library DIR [--models DIR] ...");
  Contents c;
  try {
    c = collect(a);
  } catch (const Missing& e) {
    std::println(stderr, "nvfx_pack: {}", e.what());
    return a.flag("require-models") ? 77 : 1;
  }
  pk::PackOptions o;
  const std::string coder = a.str("coder", "fast");
  o.coder = pk::coder_from(coder).value_or(pk::Coder::fast);
  if (!pk::coder_from(coder)) throw std::invalid_argument("--coder takes store, full, light or fast");
  o.threads = a.i("threads", 1);
  std::vector<pk::Cost> costs;
  if (a.has("costs")) costs = costs_from_csv(a.str("costs"), c.members);
  if (a.flag("calibrate")) costs = calibrate(c.members, a.i("calibrate-frames", 24));
  const auto t0 = Clock::now();
  const pk::Written w = pk::write(c.members, c.needs, costs, o);
  const double ms = ms_since(t0);
  write_bytes(pos[1], w.file);
  std::println("{}: {} members ({} models, {} entries{}), {} bytes from {} loose ({}), coder {}, written in {:.0f} ms", pos[1], w.manifest.members.size(),
               std::ranges::count_if(w.manifest.members, [](const pk::Record& r) { return r.kind == pk::Kind::model; }), w.manifest.needs.size(),
               costs.empty() ? "" : std::format(", {} costs", costs.size()), w.file.size(), c.loose_bytes, ratio(w.file.size(), c.loose_bytes), coder, ms);
  for (const pk::Record& r : w.manifest.members)
    std::println("  {:<24} {:<9} {:<6} {:>9} -> {:>9} ({})", r.name, pk::kind_name(r.kind), r.method == pk::Method::cm ? "cm" : "stored", r.bytes, r.stored_bytes,
                 ratio(r.stored_bytes, r.bytes));
  return 0;
}

pk::Pack read_pack(const std::string& path, int threads) {
  const auto bytes = read_bytes(path);
  try {
    return pk::read(bytes, pk::ReadOptions{threads});
  } catch (const pk::Error& e) {
    throw std::runtime_error(std::format("{}: {}", path, e.what()));
  }
}

int do_list(const Args& a) {
  const auto& pos = a.positional();
  if (pos.size() != 2) throw std::invalid_argument("list PACK");
  const pk::Pack p = read_pack(pos[1], 1);
  if (a.flag("manifest")) {
    std::print("{}", p.manifest_text);
    return 0;
  }
  std::uint64_t unpacked = 0;
  for (const pk::Member& m : p.members) unpacked += m.rec.bytes;
  std::println("{}: format {}, {} bytes, {} members ({} bytes unpacked, {}), {} entries, {} costs", pos[1], p.manifest.format, p.file_bytes, p.members.size(),
               unpacked, ratio(p.file_bytes, unpacked), p.manifest.needs.size(), p.manifest.costs.size());
  std::println("| member | kind | method | bytes | in the pack | ratio | SHA-256 |");
  std::println("|---|---|---|---:|---:|---:|---|");
  for (const pk::Member& m : p.members)
    std::println("| {} | {} | {} | {} | {} | {} | {} |", m.rec.name, pk::kind_name(m.rec.kind), m.rec.method == pk::Method::cm ? "cm" : "stored", m.rec.bytes,
                 m.rec.stored_bytes, ratio(m.rec.stored_bytes, m.rec.bytes), pk::hex(m.rec.sha256));
  std::map<std::string, std::vector<std::string>> by_models;
  for (const pk::Need& n : p.manifest.needs) {
    std::string k;
    for (const std::string& f : n.models) k += (k.empty() ? "" : " + ") + f;
    by_models[k.empty() ? "(procedural)" : k].push_back(n.entry);
  }
  if (!by_models.empty()) std::println("entries by the models they need:");
  for (const auto& [k, v] : by_models) {
    std::string names;
    for (const std::string& e : v) names += (names.empty() ? "" : ", ") + e;
    std::println("  {}: {}", k, names);
  }
  for (const pk::Cost& c : p.manifest.costs) std::println("cost {} at {} px: step {:.4f} ms, picture {:.4f} ms", c.model, c.size, c.step_ms, c.render_ms);
  for (const pk::Member& m : p.members)
    if (m.rec.kind == pk::Kind::licence) std::println("--- {} ---\n{}", m.rec.name, m.text());
  return 0;
}

int do_verify(const Args& a) {
  const auto& pos = a.positional();
  if (pos.size() != 2) throw std::invalid_argument("verify PACK");
  const auto t0 = Clock::now();
  nvfx_pack_desc d;
  nvfx_pack_desc_init(&d);
  d.path = pos[1].c_str();
  d.threads = a.i("threads", 1);
  if (a.flag("skip-missing") && !fs::exists(pos[1])) {
    std::println(stderr, "verify: {} is not there: skipped", pos[1]);
    return 77;
  }
  nvfx_pack* pack = nullptr;
  nvfx_scene_error err;
  if (nvfx_pack_open(&d, &pack, &err) != NVFX_OK) {
    std::println(stderr, "verify: {}", err.message);
    return 1;
  }
  const double open_ms = ms_since(t0);
  nvfx_pack_info info;
  nvfx_pack_get_info(pack, &info);
  int failures = 0;
  for (int i = 0; i < info.n_members; ++i) {
    nvfx_pack_member m;
    nvfx_pack_get_member(pack, i, &m);
    if (m.kind != NVFX_PACK_MODEL) continue;
    nvfx_effect* e = nullptr;
    if (const nvfx_status st = nvfx_pack_load_effect(pack, m.name, &e); st != NVFX_OK) {
      std::println(stderr, "verify: {} does not load: {}", m.name, nvfx_status_string(st));
      ++failures;
    }
    nvfx_effect_free(e);
  }
  if (info.n_catalogue > 0) {
    nvfx_library* lib = nullptr;
    if (nvfx_library_open_pack(pack, &lib, &err) != NVFX_OK) {
      std::println(stderr, "verify: {}", err.message);
      ++failures;
    } else {
      std::println("library: {} entries in {} categories", nvfx_library_count(lib), nvfx_library_category_count(lib));
    }
    nvfx_library_free(lib);
  }
  std::println("{}: {} members, every SHA-256 checked, {} models loaded, opened in {:.1f} ms: {}", pos[1], info.n_members, info.n_models, open_ms,
               failures ? "FAILED" : "ok");
  nvfx_pack_free(pack);
  return failures ? 1 : 0;
}

int do_extract(const Args& a) {
  const auto& pos = a.positional();
  if (pos.size() != 3) throw std::invalid_argument("extract PACK DIR");
  const pk::Pack p = read_pack(pos[1], a.i("threads", 1));
  for (const pk::Member& m : p.members) write_bytes(fs::path(pos[2]) / m.rec.name, m.bytes());
  std::println("{}: {} members into {}", pos[1], p.members.size(), pos[2]);
  return 0;
}

// --- measure: the coders' sizes and loading times --------------------------------------------------------------

double median(std::vector<double> v) {
  std::ranges::sort(v);
  return v.empty() ? 0.0 : v[v.size() / 2];
}

int do_measure(const Args& a) {
  const Contents c = collect(a);
  const int reps = a.i("reps", 7), threads = a.i("threads", 3);
  const fs::path out = a.str("out", "results/pack");
  fs::create_directories(out);
  const std::vector<pk::Coder> coders = {pk::Coder::store, pk::Coder::fast, pk::Coder::light, pk::Coder::full};
  std::vector<pk::Written> packs;
  for (const pk::Coder k : coders) {
    pk::PackOptions o;
    o.coder = k;
    o.threads = threads;
    packs.push_back(pk::write(c.members, c.needs, {}, o));
    std::println("  {}: {} bytes", pk::coder_name(k), packs.back().file.size());
  }
  // the loose files, as an engine reads them today: every file read, the library opened from the folder, every model
  // loaded; and each pack: opened (checked and decoded, on 1 and on `threads` threads), the library opened from it,
  // every model loaded. Interleaved repetitions; the median.
  const fs::path library = a.str("library"), models = a.has("models") ? fs::path(a.str("models")) : library;
  struct Times {
    std::vector<double> open1, openN, library, models, total1;
  };
  std::vector<Times> t(coders.size() + 1);
  const auto load_models = [](const auto& load_one, const std::vector<pk::Input>& members) {
    for (const pk::Input& in : members) {
      if (in.kind != pk::Kind::model) continue;
      nvfx_effect* e = load_one(in.name);
      if (!e) throw std::runtime_error("cannot load " + in.name);
      nvfx_effect_free(e);
    }
  };
  for (int r = 0; r < reps; ++r) {
    {  // loose files
      Times& T = t.back();
      const auto t0 = Clock::now();
      std::size_t n = 0;
      for (const pk::Input& in : c.members) n += read_bytes((in.kind == pk::Kind::model ? models : library) / in.name).size();
      const double read_ms = ms_since(t0);
      const auto t1 = Clock::now();
      nvfx_library_desc d;
      nvfx_library_desc_init(&d);
      const std::string dir = library.string(), mdir = models.string();
      d.catalogue_dir = dir.c_str();
      d.models_dir = mdir.c_str();
      nvfx_library* lib = nullptr;
      nvfx_scene_error err;
      if (nvfx_library_open(&d, &lib, &err) != NVFX_OK) throw std::runtime_error(err.message);
      const double lib_ms = ms_since(t1);
      const auto t2 = Clock::now();
      load_models([&](const std::string& f) {
        nvfx_effect* e = nullptr;
        nvfx_effect_load((models / f).string().c_str(), &e);
        return e;
      }, c.members);
      const double model_ms = ms_since(t2);
      nvfx_library_free(lib);
      T.open1.push_back(read_ms);
      T.openN.push_back(read_ms);
      T.library.push_back(lib_ms);
      T.models.push_back(model_ms);
      T.total1.push_back(lib_ms + model_ms);  // what an engine does today: the library from the folder, the models from files
      if (n != c.loose_bytes) throw std::runtime_error("the loose files changed");
    }
    for (std::size_t k = 0; k < coders.size(); ++k) {
      Times& T = t[k];
      double open1 = 0;
      for (const int th : {1, threads}) {
        nvfx_pack_desc d;
        nvfx_pack_desc_init(&d);
        d.data = packs[k].file.data();
        d.bytes = packs[k].file.size();
        d.threads = th;
        nvfx_pack* pack = nullptr;
        nvfx_scene_error err;
        const auto t0 = Clock::now();
        if (nvfx_pack_open(&d, &pack, &err) != NVFX_OK) throw std::runtime_error(err.message);
        const double ms = ms_since(t0);
        if (th == 1) {
          open1 = ms;
          T.open1.push_back(ms);
          const auto t1 = Clock::now();
          nvfx_library* lib = nullptr;
          if (nvfx_library_open_pack(pack, &lib, &err) != NVFX_OK) throw std::runtime_error(err.message);
          const double lib_ms = ms_since(t1);
          const auto t2 = Clock::now();
          load_models([&](const std::string& f) {
            nvfx_effect* e = nullptr;
            nvfx_pack_load_effect(pack, f.c_str(), &e);
            return e;
          }, c.members);
          const double model_ms = ms_since(t2);
          nvfx_library_free(lib);
          T.library.push_back(lib_ms);
          T.models.push_back(model_ms);
          T.total1.push_back(open1 + lib_ms + model_ms);
        } else {
          T.openN.push_back(ms);
        }
        nvfx_pack_free(pack);
      }
    }
  }
  std::ofstream csv(out / "pack_measure.csv");
  csv << "what,bytes,ratio_to_loose,encode_s,open_ms_1_thread,open_ms_n_threads,threads,library_ms,models_ms,total_ms_1_thread,reps\n";
  std::println("\n| what | bytes | against the loose files | open, 1 thread (ms) | open, {} threads (ms) | library (ms) | models (ms) | in all, 1 thread (ms) |", threads);
  std::println("|---|---:|---:|---:|---:|---:|---:|---:|");
  const auto row = [&](const std::string& what, std::size_t bytes, double enc, const Times& T) {
    const double r = static_cast<double>(bytes) / static_cast<double>(c.loose_bytes);
    csv << std::format("{},{},{:.4f},{:.2f},{:.2f},{:.2f},{},{:.2f},{:.2f},{:.2f},{}\n", what, bytes, r, enc, median(T.open1), median(T.openN), threads, median(T.library),
                       median(T.models), median(T.total1), reps);
    std::println("| {} | {} | {:.3f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} |", what, bytes, r, median(T.open1), median(T.openN), median(T.library), median(T.models),
                 median(T.total1));
  };
  row("loose files", c.loose_bytes, 0.0, t.back());
  for (std::size_t k = 0; k < coders.size(); ++k) row(std::format("pack, {}", pk::coder_name(coders[k])), packs[k].file.size(), packs[k].encode_seconds, t[k]);
  std::ofstream mcsv(out / "pack_members.csv");
  mcsv << "member,kind,bytes";
  for (const pk::Coder k : coders) mcsv << "," << pk::coder_name(k);
  mcsv << "\n";
  for (std::size_t i = 0; i < c.members.size(); ++i) {
    mcsv << std::format("{},{},{}", c.members[i].name, pk::kind_name(c.members[i].kind), c.members[i].data.size());
    for (const pk::Written& w : packs) mcsv << "," << w.manifest.members[i].stored_bytes;
    mcsv << "\n";
  }
  std::println("\n{} and {}", (out / "pack_measure.csv").string(), (out / "pack_members.csv").string());
  return 0;
}

}  // namespace

bool is_pack_mode(std::string_view w) { return w == "pack" || w == "list" || w == "verify" || w == "extract" || w == "measure"; }

int pack_mode(const Args& a) {
  const std::string mode = a.positional().front();
  int r = 1;
  if (mode == "pack") r = do_pack(a);
  else if (mode == "list") r = do_list(a);
  else if (mode == "verify") r = do_verify(a);
  else if (mode == "extract") r = do_extract(a);
  else if (mode == "measure") r = do_measure(a);
  a.warn_unused();
  return r;
}

}  // namespace nfx::tools
