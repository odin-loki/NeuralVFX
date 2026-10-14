// The C API of the effect library (include/neuralfx/nvfx_library.h, docs/LIBRARY.md): a catalogue (library.hpp) and
// stages, each a scene (nvfx_scene.h) whose script library::build_stage writes. A stage keeps, per instance's room, the
// times it asked for (spawn, stop) and reads the scene's own record of the spawn (the room's _go rule); everything a
// game calls while the stage plays sets the scene's inputs, so nothing allocates after nvfx_stage_create.
#include <neuralfx/nvfx_library.h>

#include "library.hpp"
#include "nvfx_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace lib = nfx::library;
namespace sc = nfx::compose::script;

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

void set_error(nvfx_scene_error* err, int line, int col, std::string_view message) {
  if (!err) return;
  err->line = line;
  err->column = col;
  const std::size_t n = std::min(message.size(), sizeof(err->message) - 1);
  std::memcpy(err->message, message.data(), n);
  err->message[n] = '\0';
}

template <class F>
nvfx_status guarded(nvfx_scene_error* err, F&& f) {
  try {
    return f();
  } catch (const sc::Error& e) {
    set_error(err, e.pos.line, e.pos.col, e.what());
    return NVFX_ERROR_SCRIPT;
  } catch (const std::bad_alloc&) {
    set_error(err, 0, 0, "out of memory");
    return NVFX_ERROR_MEMORY;
  } catch (const std::filesystem::filesystem_error& e) {
    set_error(err, 0, 0, e.what());
    return NVFX_ERROR_IO;
  } catch (const std::runtime_error& e) {
    set_error(err, 0, 0, e.what());
    return NVFX_ERROR_IO;
  } catch (const std::exception& e) {
    set_error(err, 0, 0, e.what());
    return NVFX_ERROR_ARGUMENT;
  } catch (...) {
    set_error(err, 0, 0, "unknown error");
    return NVFX_ERROR_ARGUMENT;
  }
}

std::string join(const std::vector<std::string>& v) {
  std::string o;
  for (const std::string& s : v) o += (o.empty() ? "" : ", ") + s;
  return o;
}

}  // namespace

struct nvfx_library {
  lib::Catalogue cat;
  std::filesystem::path models_dir;
  std::vector<std::string> models, elements;  // per entry, joined (the strings nvfx_library_get_effect gives)
};

struct nvfx_stage {
  const nvfx_library* library = nullptr;
  lib::Stage stage;
  std::string text;
  nvfx_scene* scene = nullptr;
  float fps = 30.f;
  struct Room {
    int entry = 0, first = 0, count = 0;
    bool steal = false;
  };
  std::vector<Room> rooms;
  struct Slot {
    float at = kInf, end = kInf;  // the spawn's and the stop's times asked for (the scene's inputs _at and _end)
    float x = 0.f, y = 0.f, scale = 1.f;
    std::uint16_t gen = 0;        // spawns so far (a handle carries it)
    std::vector<float> params;    // the parameters' values
  };
  std::vector<Slot> slots;
  std::uint32_t spawns = 0;

  ~nvfx_stage() {
    if (scene) nvfx_scene_free(scene);
  }
  const lib::Entry& entry(int slot) const { return library->cat.entries[static_cast<std::size_t>(stage.slots[static_cast<std::size_t>(slot)].entry)]; }
  float go(int slot) const {  // when the room last spawned (infinity: never)
    float t = kInf;
    nvfx_scene_rule_state(scene, stage.slots[static_cast<std::size_t>(slot)].go.c_str(), nullptr, &t);
    return t;
  }
  float now() const { return static_cast<float>(nvfx_scene_time(scene)); }
  nvfx_fx_state state(int i) const {
    const Slot& s = slots[static_cast<std::size_t>(i)];
    if (s.at == kInf) return s.gen == 0 ? NVFX_FX_NONE : NVFX_FX_ENDED;
    const float g = go(i);
    if (!(g >= s.at) || g == kInf) return NVFX_FX_WAITING;
    const lib::Entry& e = entry(i);
    const float t = now();
    if ((t >= s.end && s.end >= g) || (!e.loop && t >= g + e.duration)) return NVFX_FX_ENDED;
    return NVFX_FX_PLAYING;
  }
  bool free(int i) const {
    const nvfx_fx_state st = state(i);
    return st == NVFX_FX_NONE || st == NVFX_FX_ENDED;
  }
  int slot_of(nvfx_fx fx) const {  // -1: not a live handle
    const int i = static_cast<int>(fx & 0xffffu) - 1;
    if (i < 0 || i >= static_cast<int>(slots.size())) return -1;
    return slots[static_cast<std::size_t>(i)].gen == static_cast<std::uint16_t>(fx >> 16) ? i : -1;
  }
  void input(const std::string& name, float v) { nvfx_scene_set_input(scene, name.c_str(), v); }
  int room_of(const char* effect) const {
    if (!effect) return -1;
    for (std::size_t r = 0; r < rooms.size(); ++r)
      if (library->cat.entries[static_cast<std::size_t>(rooms[r].entry)].name == effect) return static_cast<int>(r);
    return -1;
  }
};

extern "C" {

void nvfx_library_desc_init(nvfx_library_desc* d) {
  if (d) *d = {};
}

nvfx_status nvfx_library_open(const nvfx_library_desc* d, nvfx_library** out, nvfx_scene_error* err) {
  set_error(err, 0, 0, "");
  if (!d || !out || (!d->catalogue_dir && d->n_files <= 0) || d->n_files < 0 || (d->n_files > 0 && !d->files)) {
    set_error(err, 0, 0, "invalid library description");
    return NVFX_ERROR_ARGUMENT;
  }
  *out = nullptr;
  return guarded(err, [&] {
    auto l = std::make_unique<nvfx_library>();
    std::vector<std::pair<std::string, std::string>> files;
    if (d->catalogue_dir) files = lib::read_dir(d->catalogue_dir);
    for (int i = 0; i < d->n_files; ++i) {
      if (!d->files[i].text) throw std::invalid_argument("a catalogue file without its text");
      files.emplace_back(d->files[i].name ? d->files[i].name : std::format("file{}", i), d->files[i].text);
    }
    l->cat = lib::parse(files);
    l->models_dir = d->models_dir ? std::filesystem::path(d->models_dir) : d->catalogue_dir ? std::filesystem::path(d->catalogue_dir) : std::filesystem::path();
    for (const lib::Entry& e : l->cat.entries) {
      l->models.push_back(join(e.effects));
      l->elements.push_back(join(e.elements));
    }
    *out = l.release();
    return NVFX_OK;
  });
}

void nvfx_library_free(nvfx_library* l) { delete l; }

int nvfx_library_count(const nvfx_library* l) { return l ? static_cast<int>(l->cat.entries.size()) : 0; }

int nvfx_library_find(const nvfx_library* l, const char* name) { return l && name ? l->cat.index(name) : -1; }

nvfx_status nvfx_library_get_effect(const nvfx_library* l, int i, nvfx_library_effect* o) {
  if (!l || !o || i < 0 || i >= nvfx_library_count(l)) return NVFX_ERROR_ARGUMENT;
  const lib::Entry& e = l->cat.entries[static_cast<std::size_t>(i)];
  *o = {};
  o->name = e.name.c_str();
  o->category = e.category.c_str();
  o->description = e.about.c_str();
  o->models = l->models[static_cast<std::size_t>(i)].c_str();
  o->elements = l->elements[static_cast<std::size_t>(i)].c_str();
  o->file = e.file.c_str();
  o->width = e.width;
  o->height = e.height;
  o->anchor = e.anchor == lib::Anchor::centre ? NVFX_ANCHOR_CENTRE : e.anchor == lib::Anchor::left ? NVFX_ANCHOR_LEFT : NVFX_ANCHOR_BASE;
  o->looping = e.loop ? 1 : 0;
  o->duration = e.duration;
  o->show = e.show;
  o->n_params = static_cast<int>(e.params.size());
  return NVFX_OK;
}

nvfx_status nvfx_library_get_param(const nvfx_library* l, int effect, int param, nvfx_library_param* o) {
  if (!l || !o || effect < 0 || effect >= nvfx_library_count(l)) return NVFX_ERROR_ARGUMENT;
  const lib::Entry& e = l->cat.entries[static_cast<std::size_t>(effect)];
  if (param < 0 || param >= static_cast<int>(e.params.size())) return NVFX_ERROR_ARGUMENT;
  const lib::Param& p = e.params[static_cast<std::size_t>(param)];
  o->name = p.name.c_str();
  o->description = p.about.c_str();
  o->value = p.value;
  o->min = p.lo;
  o->max = p.hi;
  return NVFX_OK;
}

int nvfx_library_category_count(const nvfx_library* l) { return l ? static_cast<int>(l->cat.categories().size()) : 0; }

const char* nvfx_library_category(const nvfx_library* l, int i) {
  if (!l || i < 0) return nullptr;
  // the categories, in order (the first entry of each, so the string is the library's)
  const std::vector<std::string> cats = l->cat.categories();
  if (i >= static_cast<int>(cats.size())) return nullptr;
  for (const lib::Entry& e : l->cat.entries)
    if (e.category == cats[static_cast<std::size_t>(i)]) return e.category.c_str();
  return nullptr;
}

nvfx_status nvfx_library_list_models(const nvfx_library* l, int effect, nvfx_scene_effect_fn fn, void* user) {
  if (!l || !fn || effect < 0 || effect >= nvfx_library_count(l)) return NVFX_ERROR_ARGUMENT;
  for (const std::string& n : l->cat.entries[static_cast<std::size_t>(effect)].effects) {
    const std::string file = l->cat.effect_file(n);
    fn(user, n.c_str(), file.c_str());
  }
  return NVFX_OK;
}

void nvfx_stage_desc_init(nvfx_stage_desc* d) {
  if (!d) return;
  *d = {};
  d->width = 960;
  d->height = 540;
  d->fps = 30.f;
  d->particles = 8192;
  d->light_gain = 0.14f;
  d->exposure = 1.2f;
  d->haze = 1.f;
  d->bloom = 1.f;
  d->threads = 2;
  d->overlap = -1;
}

nvfx_status nvfx_stage_create(const nvfx_library* l, const nvfx_stage_desc* d, nvfx_stage** out, nvfx_scene_error* err) {
  set_error(err, 0, 0, "");
  if (!l || !d || !out || d->n_rooms < 0 || (d->n_rooms > 0 && !d->rooms) || d->n_effects < 0 || (d->n_effects > 0 && !d->effects) || d->width < 16 ||
      d->height < 16 || !(d->fps > 0.f) || d->particles < 1) {
    set_error(err, 0, 0, "invalid stage description");
    return NVFX_ERROR_ARGUMENT;
  }
  *out = nullptr;
  return guarded(err, [&]() -> nvfx_status {
    auto s = std::make_unique<nvfx_stage>();
    s->library = l;
    lib::StageSpec spec;
    spec.width = d->width;
    spec.height = d->height;
    spec.fps = d->fps;
    spec.ground = d->ground != 0.f ? d->ground : std::round(0.87f * static_cast<float>(d->height));
    if (d->bus_width > 0.f && d->bus_height > 0.f) {
      spec.has_bus = true;
      spec.bus_x = d->bus_x;
      spec.bus_y = d->bus_y;
      spec.bus_w = d->bus_width;
      spec.bus_h = d->bus_height;
    }
    spec.particles = d->particles;
    spec.light_gain = d->light_gain;
    spec.exposure = d->exposure;
    spec.haze = d->haze;
    spec.bloom = d->bloom;
    int first = 0;
    for (int r = 0; r < d->n_rooms; ++r) {
      const nvfx_stage_room& room = d->rooms[r];
      const int e = room.effect ? l->cat.index(room.effect) : -1;
      if (e < 0) {
        set_error(err, 0, 0, std::format("the library has no effect '{}'", room.effect ? room.effect : "(null)"));
        return NVFX_ERROR_ARGUMENT;
      }
      if (s->room_of(room.effect) >= 0 || room.count < 1) {
        set_error(err, 0, 0, std::format("the room for '{}' is given twice or holds no instance", room.effect));
        return NVFX_ERROR_ARGUMENT;
      }
      spec.rooms.push_back({e, room.count});
      s->rooms.push_back({e, first, room.count, room.steal != 0});
      first += room.count;
    }
    s->stage = lib::build_stage(l->cat, spec);
    s->text = s->stage.text();
    // the learned effects: passed in, or read from the models' folder
    std::vector<nvfx_effect*> loaded;
    std::vector<nvfx_scene_effect> list;
    struct Free {
      std::vector<nvfx_effect*>& v;
      ~Free() {
        for (nvfx_effect* e : v) nvfx_effect_free(e);
      }
    } free_loaded{loaded};
    for (const std::string& file : s->stage.files) {
      const std::string stem = std::filesystem::path(file).stem().string();
      bool given = false;
      for (int i = 0; i < d->n_effects && !given; ++i) {
        const char* n = d->effects[i].name;
        given = n && (file == n || stem == n);
      }
      if (given) continue;
      const std::filesystem::path p = std::filesystem::path(file).is_absolute() ? std::filesystem::path(file) : l->models_dir / file;
      nvfx_effect* e = nullptr;
      const nvfx_status st = nvfx_effect_load(p.string().c_str(), &e);
      if (st != NVFX_OK) {
        set_error(err, 0, 0, std::format("cannot load {}: {}", p.string(), nvfx_status_string(st)));
        return st;
      }
      loaded.push_back(e);
      list.push_back({file.c_str(), e});
    }
    for (int i = 0; i < d->n_effects; ++i) list.push_back(d->effects[i]);
    nvfx_scene_desc sd;
    nvfx_scene_desc_init(&sd);
    sd.script = s->text.c_str();
    sd.source_name = "stage";
    sd.effects = list.data();
    sd.n_effects = static_cast<int>(list.size());
    sd.threads = d->threads > 0 ? d->threads : 2;
    sd.overlap = d->overlap;
    const nvfx_status st = nvfx_scene_create(&sd, &s->scene, err);
    if (st == NVFX_ERROR_SCRIPT) {
      // An entry checks without its effects (nvfx_library_open) but not with them (a control it does not have, a start
      // point past its last, a size off its grid): find which, and say where in its catalogue file.
      const auto load = [&](const std::string& file) {
        for (const nvfx_scene_effect& e : list)
          if (file == e.name || std::filesystem::path(file).stem().string() == e.name)
            if (const nfx::rt::RolloutEffect* r = nfx::rt::rollout_of(e.effect)) return *r;
        throw std::runtime_error("cannot load " + file);
      };
      for (const nvfx_stage::Room& room : s->rooms) {
        try {
          sc::Options o;
          o.threads = 1;
          const sc::Scene alone(lib::single(l->cat, room.entry, spec).script, load, o);
        } catch (const sc::Error& e) {
          set_error(err, e.pos.line, e.pos.col, e.what());
          return NVFX_ERROR_SCRIPT;
        } catch (const std::exception&) {
        }
      }
    }
    if (st != NVFX_OK) return st;
    nvfx_scene_info info{};
    nvfx_scene_get_info(s->scene, &info);
    s->fps = info.fps;
    for (const lib::Slot& slot : s->stage.slots) {
      nvfx_stage::Slot x;
      for (const lib::Param& p : l->cat.entries[static_cast<std::size_t>(slot.entry)].params) x.params.push_back(p.value);
      s->slots.push_back(std::move(x));
    }
    *out = s.release();
    return NVFX_OK;
  });
}

void nvfx_stage_free(nvfx_stage* s) { delete s; }

nvfx_scene* nvfx_stage_scene(nvfx_stage* s) { return s ? s->scene : nullptr; }

size_t nvfx_stage_script(const nvfx_stage* s, char* out, size_t capacity) {
  if (!s) return 0;
  if (out && capacity > 0) {
    const std::size_t n = std::min(capacity - 1, s->text.size());
    std::memcpy(out, s->text.data(), n);
    out[n] = '\0';
  }
  return s->text.size();
}

nvfx_status nvfx_stage_spawn(nvfx_stage* s, const char* effect, float x, float y, float scale, double at, nvfx_fx* out) {
  if (out) *out = 0;
  if (!s || !std::isfinite(x) || !std::isfinite(y) || !(scale > 0.f) || !std::isfinite(scale) || std::isnan(at)) return NVFX_ERROR_ARGUMENT;
  const int r = s->room_of(effect);
  if (r < 0) return NVFX_ERROR_ARGUMENT;
  const nvfx_stage::Room& room = s->rooms[static_cast<std::size_t>(r)];
  const float now = s->now(), frame = 1.f / s->fps;
  int pick = -1;
  for (int i = room.first; i < room.first + room.count && pick < 0; ++i)
    if (s->free(i)) pick = i;
  if (pick < 0) {
    if (!room.steal) return NVFX_ERROR_MEMORY;
    float oldest = kInf;  // the instance that spawned first (or a waiting one, last)
    for (int i = room.first; i < room.first + room.count; ++i) {
      const nvfx_stage::Slot& sl = s->slots[static_cast<std::size_t>(i)];
      const float g = s->state(i) == NVFX_FX_PLAYING ? s->go(i) : sl.at + 1e6f;
      if (pick < 0 || g < oldest) {
        oldest = g;
        pick = i;
      }
    }
    nvfx_stage::Slot& sl = s->slots[static_cast<std::size_t>(pick)];
    if (s->state(pick) == NVFX_FX_PLAYING) {  // it ends now; the new one spawns a frame later
      sl.end = now;
      s->input(s->stage.slots[static_cast<std::size_t>(pick)].end, sl.end);
    }
  }
  nvfx_stage::Slot& sl = s->slots[static_cast<std::size_t>(pick)];
  const lib::Slot& names = s->stage.slots[static_cast<std::size_t>(pick)];
  const lib::Entry& e = s->entry(pick);
  // when: not before now, after the room's last spawn (its rule fires once per time asked for) and after a stop of the
  // instance before (whose end must come before this spawn)
  float when = at < 0.0 ? now : std::max(now, static_cast<float>(at));
  const float g = s->go(pick);
  if (g != kInf) when = std::max(when, g + 0.5f * frame);
  if (sl.end != kInf) when = std::max(when, sl.end + 0.5f * frame);
  sl.at = when;
  sl.x = x;
  sl.y = y;
  sl.scale = scale;
  sl.gen = static_cast<std::uint16_t>(sl.gen == 0xffff ? 1 : sl.gen + 1);
  ++s->spawns;
  const std::uint32_t h = s->spawns * 0x9e3779b9u ^ static_cast<std::uint32_t>(pick) * 0x85ebca6bu;
  const float variant = static_cast<float>((h ^ (h >> 15)) % 65536u) / 65536.f;
  s->input(names.X, x);
  s->input(names.Y, y);
  s->input(names.scale, scale);
  s->input(names.variant, variant);
  for (std::size_t p = 0; p < e.params.size(); ++p) {
    sl.params[p] = e.params[p].value;
    s->input(names.params[p], sl.params[p]);
  }
  s->input(names.at, sl.at);
  if (out) *out = (static_cast<nvfx_fx>(sl.gen) << 16) | static_cast<nvfx_fx>(pick + 1);
  return NVFX_OK;
}

nvfx_status nvfx_stage_set_param(nvfx_stage* s, nvfx_fx fx, const char* param, float value) {
  const int i = s ? s->slot_of(fx) : -1;
  if (i < 0 || !param || !std::isfinite(value)) return NVFX_ERROR_ARGUMENT;
  const lib::Entry& e = s->entry(i);
  for (std::size_t p = 0; p < e.params.size(); ++p) {
    if (e.params[p].name != param) continue;
    const float v = std::clamp(value, e.params[p].lo, e.params[p].hi);
    s->slots[static_cast<std::size_t>(i)].params[p] = v;
    s->input(s->stage.slots[static_cast<std::size_t>(i)].params[p], v);
    return NVFX_OK;
  }
  return NVFX_ERROR_ARGUMENT;
}

nvfx_status nvfx_stage_get_param(const nvfx_stage* s, nvfx_fx fx, const char* param, float* value) {
  const int i = s ? s->slot_of(fx) : -1;
  if (i < 0 || !param || !value) return NVFX_ERROR_ARGUMENT;
  const lib::Entry& e = s->entry(i);
  for (std::size_t p = 0; p < e.params.size(); ++p) {
    if (e.params[p].name != param) continue;
    *value = s->slots[static_cast<std::size_t>(i)].params[p];
    return NVFX_OK;
  }
  return NVFX_ERROR_ARGUMENT;
}

nvfx_status nvfx_stage_move(nvfx_stage* s, nvfx_fx fx, float x, float y) {
  const int i = s ? s->slot_of(fx) : -1;
  if (i < 0 || !std::isfinite(x) || !std::isfinite(y)) return NVFX_ERROR_ARGUMENT;
  nvfx_stage::Slot& sl = s->slots[static_cast<std::size_t>(i)];
  sl.x = x;
  sl.y = y;
  s->input(s->stage.slots[static_cast<std::size_t>(i)].X, x);
  s->input(s->stage.slots[static_cast<std::size_t>(i)].Y, y);
  return NVFX_OK;
}

nvfx_status nvfx_stage_stop(nvfx_stage* s, nvfx_fx fx, double at) {
  const int i = s ? s->slot_of(fx) : -1;
  if (i < 0 || std::isnan(at)) return NVFX_ERROR_ARGUMENT;
  nvfx_stage::Slot& sl = s->slots[static_cast<std::size_t>(i)];
  const lib::Slot& names = s->stage.slots[static_cast<std::size_t>(i)];
  const nvfx_fx_state st = s->state(i);
  if (st == NVFX_FX_WAITING) {  // not spawned yet: cancelled
    sl.at = kInf;
    s->input(names.at, sl.at);
    return NVFX_OK;
  }
  if (st != NVFX_FX_PLAYING) return NVFX_OK;
  const float now = s->now();
  sl.end = at < 0.0 ? now : std::max(now, static_cast<float>(at));
  sl.end = std::max(sl.end, s->go(i));
  s->input(names.end, sl.end);
  return NVFX_OK;
}

nvfx_status nvfx_stage_get_fx(const nvfx_stage* s, nvfx_fx fx, nvfx_stage_fx_info* o) {
  if (!s || !o) return NVFX_ERROR_ARGUMENT;
  *o = {};
  o->spawned = kInf;
  const int i = s->slot_of(fx);
  if (i < 0) return NVFX_ERROR_ARGUMENT;
  const nvfx_stage::Slot& sl = s->slots[static_cast<std::size_t>(i)];
  o->state = s->state(i);
  o->effect = s->stage.slots[static_cast<std::size_t>(i)].entry;
  o->x = sl.x;
  o->y = sl.y;
  o->scale = sl.scale;
  if (o->state == NVFX_FX_PLAYING || o->state == NVFX_FX_ENDED) {
    const float g = s->go(i);
    if (g != kInf && g >= sl.at) {
      o->spawned = g;
      o->age = std::max(0.f, s->now() - g);
    }
  }
  return NVFX_OK;
}

int nvfx_stage_free_rooms(const nvfx_stage* s, const char* effect) {
  const int r = s ? s->room_of(effect) : -1;
  if (r < 0) return -1;
  const nvfx_stage::Room& room = s->rooms[static_cast<std::size_t>(r)];
  int n = 0;
  for (int i = room.first; i < room.first + room.count; ++i) n += s->free(i) ? 1 : 0;
  return n;
}

nvfx_status nvfx_stage_set_camera(nvfx_stage* s, float x, float y) {
  if (!s || !std::isfinite(x) || !std::isfinite(y)) return NVFX_ERROR_ARGUMENT;
  nvfx_scene_set_input(s->scene, "_camera_x", x);
  nvfx_scene_set_input(s->scene, "_camera_y", y);
  return NVFX_OK;
}

nvfx_status nvfx_stage_clear(nvfx_stage* s) {
  if (!s) return NVFX_ERROR_ARGUMENT;
  return guarded(nullptr, [&] {
    for (std::size_t i = 0; i < s->slots.size(); ++i) {  // the inputs' values become the rebuilt scene's starting values
      nvfx_stage::Slot& sl = s->slots[i];
      const lib::Slot& names = s->stage.slots[i];
      const lib::Entry& e = s->entry(static_cast<int>(i));
      sl.at = sl.end = kInf;
      sl.gen = static_cast<std::uint16_t>(sl.gen == 0xffff ? 1 : sl.gen + 1);
      s->input(names.at, kInf);
      s->input(names.end, kInf);
      s->input(names.X, 0.f);
      s->input(names.Y, 0.f);
      s->input(names.scale, 1.f);
      s->input(names.variant, 0.f);
      for (std::size_t p = 0; p < e.params.size(); ++p) {
        sl.params[p] = e.params[p].value;
        s->input(names.params[p], sl.params[p]);
      }
    }
    s->spawns = 0;
    return nvfx_scene_restart(s->scene);
  });
}

}  // extern "C"
