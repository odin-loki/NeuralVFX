// The C API of effect packs (include/neuralfx/nvfx_pack.h): a pack read and checked by pack::read, its members, models
// loaded from it, and the effect library and scenes over it. Exceptions never cross the C boundary.
#include <neuralfx/nvfx_pack.h>

#include "nvfx_library_internal.hpp"
#include "pack.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace pk = nfx::pack;

struct nvfx_pack {
  std::shared_ptr<const pk::Pack> pack;  // shared with the libraries opened from it
  std::string name;
  std::vector<std::string> sha;          // per member, hex
  int licence = -1;                      // the first licence member
};

namespace {

void set_error(nvfx_scene_error* err, int line, std::string_view message) {
  if (!err) return;
  err->line = line;
  err->column = 0;
  const std::size_t n = std::min(message.size(), sizeof(err->message) - 1);
  std::memcpy(err->message, message.data(), n);
  err->message[n] = '\0';
}

template <class F>
nvfx_status guarded(nvfx_scene_error* err, const std::string& name, F&& f) {
  try {
    return f();
  } catch (const pk::Error& e) {
    set_error(err, e.line, name + ": " + e.what());
    return e.status;
  } catch (const std::bad_alloc&) {
    set_error(err, 0, name + ": out of memory");
    return NVFX_ERROR_MEMORY;
  } catch (const std::exception& e) {
    set_error(err, 0, name + ": " + e.what());
    return NVFX_ERROR_ARGUMENT;
  } catch (...) {
    set_error(err, 0, name + ": unknown error");
    return NVFX_ERROR_ARGUMENT;
  }
}

nvfx_status load_member(const pk::Pack& p, int i, nvfx_effect** out) {
  if (i < 0 || p.members[static_cast<std::size_t>(i)].rec.kind != pk::Kind::model) return NVFX_ERROR_IO;
  const auto b = p.members[static_cast<std::size_t>(i)].bytes();
  return nvfx_effect_load_memory(b.data(), b.size(), out);
}

std::string read_text(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return {};
  return {std::istreambuf_iterator<char>(f), {}};
}

}  // namespace

extern "C" {

void nvfx_pack_desc_init(nvfx_pack_desc* d) {
  if (!d) return;
  *d = {};
  d->threads = 1;
}

nvfx_status nvfx_pack_open(const nvfx_pack_desc* d, nvfx_pack** out, nvfx_scene_error* err) {
  set_error(err, 0, "");
  if (out) *out = nullptr;
  if (!d || !out || (!d->path && (!d->data || d->bytes == 0))) {
    set_error(err, 0, "invalid pack description (a path, or data and bytes)");
    return NVFX_ERROR_ARGUMENT;
  }
  const std::string name = d->name ? d->name : d->path ? std::filesystem::path(d->path).filename().string() : "pack";
  return guarded(err, name, [&]() -> nvfx_status {
    std::vector<std::uint8_t> file;
    std::span<const std::uint8_t> bytes(static_cast<const std::uint8_t*>(d->data), d->data ? d->bytes : 0);
    if (d->path) {
      std::error_code ec;
      const bool regular = std::filesystem::is_regular_file(d->path, ec);
      const auto n = regular ? static_cast<std::size_t>(std::filesystem::file_size(d->path, ec)) : 0;
      std::ifstream f(d->path, std::ios::binary);
      if (!regular || ec || !f) throw pk::Error(NVFX_ERROR_IO, std::format("cannot read {}", d->path));  // not there, or a folder
      file.resize(n);
      if (!f.read(reinterpret_cast<char*>(file.data()), static_cast<std::streamsize>(n))) throw pk::Error(NVFX_ERROR_IO, std::format("cannot read {}", d->path));
      bytes = file;
    }
    auto p = std::make_unique<nvfx_pack>();
    p->name = name;
    p->pack = std::make_shared<const pk::Pack>(pk::read(bytes, pk::ReadOptions{std::max(1, d->threads)}));
    for (const pk::Member& m : p->pack->members) {
      p->sha.push_back(pk::hex(m.rec.sha256));
      if (p->licence < 0 && m.rec.kind == pk::Kind::licence) p->licence = static_cast<int>(p->sha.size()) - 1;
    }
    *out = p.release();
    return NVFX_OK;
  });
}

void nvfx_pack_free(nvfx_pack* p) { delete p; }

nvfx_status nvfx_pack_get_info(const nvfx_pack* p, nvfx_pack_info* o) {
  if (!p || !o) return NVFX_ERROR_ARGUMENT;
  *o = {};
  const pk::Pack& k = *p->pack;
  o->format = k.manifest.format;
  o->n_members = static_cast<int>(k.members.size());
  for (const pk::Member& m : k.members) {
    o->n_models += m.rec.kind == pk::Kind::model ? 1 : 0;
    o->n_catalogue += m.rec.kind == pk::Kind::catalogue ? 1 : 0;
    o->unpacked_bytes += m.bytes().size();
  }
  o->n_entries = static_cast<int>(k.manifest.needs.size());
  o->n_costs = static_cast<int>(k.manifest.costs.size());
  o->file_bytes = k.file_bytes;
  o->licence = p->licence >= 0 ? reinterpret_cast<const char*>(k.members[static_cast<std::size_t>(p->licence)].buf.data()) : "";
  return NVFX_OK;
}

int nvfx_pack_find(const nvfx_pack* p, const char* name) { return p && name ? p->pack->find(name) : -1; }

nvfx_status nvfx_pack_get_member(const nvfx_pack* p, int i, nvfx_pack_member* o) {
  if (!p || !o || i < 0 || i >= static_cast<int>(p->pack->members.size())) return NVFX_ERROR_ARGUMENT;
  const pk::Member& m = p->pack->members[static_cast<std::size_t>(i)];
  *o = {};
  o->name = m.rec.name.c_str();
  o->kind = static_cast<nvfx_pack_kind>(m.rec.kind);
  o->compressed = m.rec.method == pk::Method::cm ? 1 : 0;
  o->data = m.buf.data();
  o->bytes = m.bytes().size();
  o->stored_bytes = static_cast<size_t>(m.rec.stored_bytes);
  o->sha256 = p->sha[static_cast<std::size_t>(i)].c_str();
  return NVFX_OK;
}

const char* nvfx_pack_entry_name(const nvfx_pack* p, int i) {
  if (!p || i < 0 || i >= static_cast<int>(p->pack->manifest.needs.size())) return nullptr;
  return p->pack->manifest.needs[static_cast<std::size_t>(i)].entry.c_str();
}

nvfx_status nvfx_pack_entry_models(const nvfx_pack* p, int i, nvfx_scene_effect_fn fn, void* user) {
  if (!p || !fn || i < 0 || i >= static_cast<int>(p->pack->manifest.needs.size())) return NVFX_ERROR_ARGUMENT;
  for (const std::string& f : p->pack->manifest.needs[static_cast<std::size_t>(i)].models) {
    const std::string stem = std::filesystem::path(f).stem().string();
    fn(user, stem.c_str(), f.c_str());
  }
  return NVFX_OK;
}

nvfx_status nvfx_pack_load_effect(const nvfx_pack* p, const char* name, nvfx_effect** out) {
  if (out) *out = nullptr;
  if (!p || !name || !out) return NVFX_ERROR_ARGUMENT;
  try {
    return load_member(*p->pack, p->pack->find(name), out);
  } catch (const std::bad_alloc&) {
    return NVFX_ERROR_MEMORY;
  } catch (...) {
    return NVFX_ERROR_FORMAT;
  }
}

nvfx_status nvfx_pack_model_cost(const nvfx_pack* p, const char* model, int size, float* step_ms, float* render_ms) {
  if (!p || !model) return NVFX_ERROR_ARGUMENT;
  const int i = p->pack->find(model);
  if (i < 0) return NVFX_ERROR_ARGUMENT;
  const std::string& file = p->pack->members[static_cast<std::size_t>(i)].rec.name;
  for (const pk::Cost& c : p->pack->manifest.costs) {
    if (c.model != file || c.size != size) continue;
    if (step_ms) *step_ms = c.step_ms;
    if (render_ms) *render_ms = c.render_ms;
    return NVFX_OK;
  }
  return NVFX_ERROR_ARGUMENT;
}

nvfx_status nvfx_library_open_pack(const nvfx_pack* p, nvfx_library** out, nvfx_scene_error* err) {
  set_error(err, 0, "");
  if (out) *out = nullptr;
  if (!p || !out) {
    set_error(err, 0, "invalid arguments");
    return NVFX_ERROR_ARGUMENT;
  }
  std::vector<nvfx_library_file> files;
  for (const pk::Member& m : p->pack->members)
    if (m.rec.kind == pk::Kind::catalogue) files.push_back({m.rec.name.c_str(), reinterpret_cast<const char*>(m.buf.data())});
  std::ranges::sort(files, [](const nvfx_library_file& a, const nvfx_library_file& b) { return std::strcmp(a.name, b.name) < 0; });
  if (files.empty()) {
    set_error(err, 0, p->name + ": the pack has no catalogue");
    return NVFX_ERROR_ARGUMENT;
  }
  nvfx_library_desc d;
  nvfx_library_desc_init(&d);
  d.files = files.data();
  d.n_files = static_cast<int>(files.size());
  nvfx_library* lib = nullptr;
  if (const nvfx_status st = nvfx_library_open(&d, &lib, err); st != NVFX_OK) return st;
  return guarded(err, p->name, [&]() -> nvfx_status {
    std::unique_ptr<nvfx_library, void (*)(nvfx_library*)> l(lib, nvfx_library_free);
    // every entry's models must be in the pack (the manifest's records say so; the catalogue itself is the authority)
    struct Missing {
      const pk::Pack* pack;
      std::string file;
    } miss{p->pack.get(), {}};
    for (int i = 0; i < nvfx_library_count(lib); ++i) {
      nvfx_library_list_models(lib, i,
                               [](void* u, const char*, const char* file) {
                                 auto* m = static_cast<Missing*>(u);
                                 const int k = m->pack->manifest.find(file);
                                 if (m->file.empty() && (k < 0 || m->pack->members[static_cast<std::size_t>(k)].rec.kind != pk::Kind::model)) m->file = file;
                               },
                               &miss);
      if (!miss.file.empty()) {
        nvfx_library_effect e;
        nvfx_library_get_effect(lib, i, &e);
        throw pk::Error(NVFX_ERROR_FORMAT, std::format("entry '{}' needs {}, which the pack does not hold", e.name, miss.file));
      }
    }
    std::shared_ptr<const pk::Pack> keep = p->pack;
    nfx::library::set_model_source(lib, [keep](const std::string& file, nvfx_effect** e) { return load_member(*keep, keep->manifest.find(file), e); });
    *out = l.release();
    return NVFX_OK;
  });
}

nvfx_status nvfx_scene_create_from_pack(const nvfx_pack* p, const nvfx_scene_desc* d, nvfx_scene** out, nvfx_scene_error* err) {
  set_error(err, 0, "");
  if (out) *out = nullptr;
  if (!p || !d || !out || (!d->script && !d->script_path) || d->n_effects < 0 || (d->n_effects > 0 && !d->effects)) {
    set_error(err, 0, "invalid scene description");
    return NVFX_ERROR_ARGUMENT;
  }
  return guarded(err, p->name, [&]() -> nvfx_status {
    nvfx_scene_desc sd = *d;
    std::string text;  // the script read from disk, to list its effects (nvfx_scene_create reads it again, by its path)
    if (!d->script) {
      const int i = p->pack->manifest.find(d->script_path);
      if (i >= 0 && p->pack->members[static_cast<std::size_t>(i)].rec.kind == pk::Kind::script) {
        sd.script = reinterpret_cast<const char*>(p->pack->members[static_cast<std::size_t>(i)].buf.data());
        if (!sd.source_name) sd.source_name = d->script_path;
      } else {
        text = read_text(d->script_path);
      }
    }
    const char* script = sd.script ? sd.script : text.c_str();
    struct Listed {
      std::vector<std::pair<std::string, std::string>> v;
    } listed;
    if (sd.script || !text.empty()) {
      const nvfx_status st = nvfx_scene_list_effects(script, sd.source_name ? sd.source_name : "script",
                                                     [](void* u, const char* n, const char* f) { static_cast<Listed*>(u)->v.emplace_back(n, f); }, &listed, err);
      if (st != NVFX_OK) return st;
    }
    std::vector<nvfx_effect*> loaded;
    struct Free {
      std::vector<nvfx_effect*>& v;
      ~Free() {
        for (nvfx_effect* e : v) nvfx_effect_free(e);
      }
    } free_loaded{loaded};
    std::vector<nvfx_scene_effect> list(d->effects, d->effects + d->n_effects);
    std::vector<std::string> names;
    names.reserve(listed.v.size());
    for (const auto& [n, file] : listed.v) {
      const std::string stem = std::filesystem::path(file).stem().string();
      bool given = false;
      for (int i = 0; i < d->n_effects && !given; ++i) given = d->effects[i].name && (file == d->effects[i].name || stem == d->effects[i].name);
      for (const std::string& s : names) given = given || s == file;
      if (given) continue;
      int k = p->pack->manifest.find(file);
      if (k < 0) k = p->pack->find(stem);
      if (k < 0 || p->pack->members[static_cast<std::size_t>(k)].rec.kind != pk::Kind::model) continue;  // read from effects_dir
      nvfx_effect* e = nullptr;
      if (const nvfx_status st = load_member(*p->pack, k, &e); st != NVFX_OK) {
        set_error(err, 0, std::format("{}: cannot load {}: {}", p->name, file, nvfx_status_string(st)));
        return st;
      }
      loaded.push_back(e);
      names.push_back(file);
    }
    for (std::size_t i = 0; i < loaded.size(); ++i) list.push_back({names[i].c_str(), loaded[i]});
    sd.effects = list.data();
    sd.n_effects = static_cast<int>(list.size());
    return nvfx_scene_create(&sd, out, err);
  });
}

}  // extern "C"
