// NeuralVFXStage (nvfx_nodes.hpp): a stage of the effect library (include/neuralfx/nvfx_library.h, docs/LIBRARY.md),
// played as a NeuralVFXScene plays its scene. The catalogue's files and the effects' files are read with FileAccess
// (res://, user:// or absolute paths) and passed to the C API in memory, so it works from an exported package too.
#include "nvfx_nodes.hpp"

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace godot {

namespace {

CharString utf8(const String& s) { return s.utf8(); }

void collect_model(void* user, const char* name, const char* file) {
  static_cast<std::vector<std::pair<std::string, std::string>>*>(user)->emplace_back(name, file);
}

const char* state_name(nvfx_fx_state s) {
  switch (s) {
    case NVFX_FX_WAITING: return "waiting";
    case NVFX_FX_PLAYING: return "playing";
    case NVFX_FX_ENDED: return "ended";
    default: return "none";
  }
}

}  // namespace

NeuralVFXStage::~NeuralVFXStage() {
  release_scene();
  close_library();
}

void NeuralVFXStage::close_library() {
  if (library_) nvfx_library_free(library_);
  library_ = nullptr;
}

bool NeuralVFXStage::open_library() {
  release_scene();  // a stage refers to its library
  close_library();
  if (library_dir_.is_empty()) {
    fail("no library_dir");
    return false;
  }
  PackedStringArray names = DirAccess::get_files_at(library_dir_);
  names.sort();
  std::vector<CharString> texts, files;
  for (const String& n : names) {
    if (n.get_extension() != "nvfxl") continue;
    texts.push_back(utf8(FileAccess::get_file_as_string(library_dir_.path_join(n))));
    files.push_back(utf8(n));
  }
  if (texts.empty()) {
    fail("no catalogue files (*.nvfxl) in " + library_dir_);
    return false;
  }
  std::vector<nvfx_library_file> list;
  for (std::size_t i = 0; i < texts.size(); ++i) list.push_back({files[i].get_data(), texts[i].get_data()});
  nvfx_library_desc d;
  nvfx_library_desc_init(&d);
  d.files = list.data();
  d.n_files = static_cast<int>(list.size());
  nvfx_scene_error err{};
  if (nvfx_library_open(&d, &library_, &err) != NVFX_OK) {
    library_ = nullptr;
    fail(String::utf8(err.message), err.line, err.column);
    return false;
  }
  return true;
}

nvfx_scene* NeuralVFXStage::create_scene() {
  if (!open_library()) return nullptr;
  // the rooms, in the dictionary's order (the stage's script, and so its frames, follow it)
  std::vector<CharString> room_names;
  std::vector<nvfx_stage_room> rooms;
  const Array keys = rooms_.keys();
  for (int64_t i = 0; i < keys.size(); ++i) {
    const Variant v = rooms_[keys[i]];
    int count = 1, steal = 0;
    if (v.get_type() == Variant::ARRAY) {
      const Array a = v;
      if (a.size() > 0) count = static_cast<int>(a[0]);
      if (a.size() > 1) steal = static_cast<bool>(a[1]) ? 1 : 0;
    } else {
      count = static_cast<int>(v);
    }
    room_names.push_back(utf8(String(keys[i])));
    rooms.push_back({nullptr, count, steal});
  }
  for (std::size_t i = 0; i < rooms.size(); ++i) rooms[i].effect = room_names[i].get_data();
  // the learned effects the rooms need, read from effects_dir
  std::vector<std::pair<std::string, std::string>> models;
  for (const nvfx_stage_room& r : rooms) {
    const int e = nvfx_library_find(library_, r.effect);
    if (e < 0) {
      fail(vformat("the library has no effect '%s'", String::utf8(r.effect)));
      return nullptr;
    }
    nvfx_library_list_models(library_, e, collect_model, &models);
  }
  const String dir = get_effects_dir().is_empty() ? library_dir_ : get_effects_dir();
  std::vector<nvfx_effect*> loaded;
  std::vector<CharString> loaded_names;
  const auto free_loaded = [&] {
    for (nvfx_effect* e : loaded) nvfx_effect_free(e);
  };
  for (const auto& [name, file] : models) {
    const String f = String::utf8(file.c_str());
    bool have = false;
    for (const CharString& n : loaded_names) have = have || file == n.get_data();
    if (have) continue;
    const String path = f.is_absolute_path() ? f : dir.path_join(f);
    const PackedByteArray bytes = FileAccess::get_file_as_bytes(path);
    nvfx_effect* e = nullptr;
    const nvfx_status s = bytes.is_empty() ? NVFX_ERROR_IO : nvfx_effect_load_memory(bytes.ptr(), static_cast<size_t>(bytes.size()), &e);
    if (s != NVFX_OK) {
      free_loaded();
      fail(vformat("effect '%s': cannot load %s (%s)", String::utf8(name.c_str()), path, String(nvfx_status_string(s))));
      return nullptr;
    }
    loaded.push_back(e);
    loaded_names.push_back(utf8(f));
  }
  std::vector<nvfx_scene_effect> list;
  for (std::size_t i = 0; i < loaded.size(); ++i) list.push_back({loaded_names[i].get_data(), loaded[i]});
  nvfx_stage_desc d;
  nvfx_stage_desc_init(&d);
  d.width = stage_size_.x;
  d.height = stage_size_.y;
  d.ground = static_cast<float>(ground_);
  d.rooms = rooms.data();
  d.n_rooms = static_cast<int>(rooms.size());
  d.effects = list.data();
  d.n_effects = static_cast<int>(list.size());
  d.threads = get_threads() > 0 ? get_threads() : 2;
  d.overlap = get_overlap() ? 1 : 0;
  d.layers = get_layers() ? 1 : 0;
  d.precision = get_precision();
  nvfx_scene_error err{};
  const nvfx_status s = nvfx_stage_create(library_, &d, &stage_, &err);
  free_loaded();  // the stage's scene keeps copies
  if (s != NVFX_OK) {
    stage_ = nullptr;
    fail(String::utf8(err.message), err.line, err.column);
    return nullptr;
  }
  return nvfx_stage_scene(stage_);
}

void NeuralVFXStage::release_scene() {
  if (stage_) nvfx_stage_free(stage_);  // with its scene
  stage_ = nullptr;
  scene_ = nullptr;
}

void NeuralVFXStage::restart_scene() {
  if (stage_) nvfx_stage_clear(stage_);
}

PackedStringArray NeuralVFXStage::list_effects() const {
  PackedStringArray out;
  for (int i = 0; library_ && i < nvfx_library_count(library_); ++i) {
    nvfx_library_effect e;
    nvfx_library_get_effect(library_, i, &e);
    out.push_back(String::utf8(e.name));
  }
  return out;
}

PackedStringArray NeuralVFXStage::get_categories() const {
  PackedStringArray out;
  for (int i = 0; library_ && i < nvfx_library_category_count(library_); ++i) out.push_back(String::utf8(nvfx_library_category(library_, i)));
  return out;
}

Dictionary NeuralVFXStage::get_effect_info(const String& effect) const {
  Dictionary d;
  const int i = library_ ? nvfx_library_find(library_, utf8(effect).get_data()) : -1;
  nvfx_library_effect e;
  if (i < 0 || nvfx_library_get_effect(library_, i, &e) != NVFX_OK) return d;
  d["name"] = String::utf8(e.name);
  d["category"] = String::utf8(e.category);
  d["description"] = String::utf8(e.description);
  d["models"] = String::utf8(e.models);
  d["elements"] = String::utf8(e.elements);
  d["size"] = Vector2(e.width, e.height);
  d["anchor"] = e.anchor == NVFX_ANCHOR_CENTRE ? "centre" : e.anchor == NVFX_ANCHOR_LEFT ? "left" : "base";
  d["looping"] = e.looping != 0;
  d["duration"] = e.duration;
  d["show"] = e.show;
  d["moving"] = Vector2(e.move_x, e.move_y);  // a moving effect (a trail): move it every frame (move_fx)
  Array params;
  for (int p = 0; p < e.n_params; ++p) {
    nvfx_library_param pr;
    nvfx_library_get_param(library_, i, p, &pr);
    Dictionary q;
    q["name"] = String::utf8(pr.name);
    q["description"] = String::utf8(pr.description);
    q["value"] = pr.value;
    q["min"] = pr.min;
    q["max"] = pr.max;
    params.push_back(q);
  }
  d["params"] = params;
  return d;
}

int64_t NeuralVFXStage::spawn(const String& effect, const Vector2& at, double scale, double time) {
  nvfx_fx fx = 0;
  if (!stage_ || nvfx_stage_spawn(stage_, utf8(effect).get_data(), static_cast<float>(at.x), static_cast<float>(at.y), static_cast<float>(scale), time, &fx) != NVFX_OK) {
    return 0;
  }
  return static_cast<int64_t>(fx);
}

bool NeuralVFXStage::set_param(int64_t fx, const String& param, double value) {
  return stage_ && nvfx_stage_set_param(stage_, static_cast<nvfx_fx>(fx), utf8(param).get_data(), static_cast<float>(value)) == NVFX_OK;
}

double NeuralVFXStage::get_param(int64_t fx, const String& param) const {
  float v = 0.f;
  if (stage_) nvfx_stage_get_param(stage_, static_cast<nvfx_fx>(fx), utf8(param).get_data(), &v);
  return v;
}

bool NeuralVFXStage::move_fx(int64_t fx, const Vector2& at) {
  return stage_ && nvfx_stage_move(stage_, static_cast<nvfx_fx>(fx), static_cast<float>(at.x), static_cast<float>(at.y)) == NVFX_OK;
}

bool NeuralVFXStage::stop(int64_t fx, double time) { return stage_ && nvfx_stage_stop(stage_, static_cast<nvfx_fx>(fx), time) == NVFX_OK; }

Dictionary NeuralVFXStage::get_fx_info(int64_t fx) const {
  Dictionary d;
  nvfx_stage_fx_info info{};
  if (!stage_ || nvfx_stage_get_fx(stage_, static_cast<nvfx_fx>(fx), &info) != NVFX_OK) {
    d["state"] = "none";
    return d;
  }
  nvfx_library_effect e;
  nvfx_library_get_effect(library_, info.effect, &e);
  d["state"] = state_name(info.state);
  d["effect"] = String::utf8(e.name);
  d["position"] = Vector2(info.x, info.y);
  d["scale"] = info.scale;
  d["spawned"] = std::isinf(info.spawned) ? -1.0 : static_cast<double>(info.spawned);
  d["age"] = info.age;
  return d;
}

int NeuralVFXStage::free_rooms(const String& effect) const { return stage_ ? nvfx_stage_free_rooms(stage_, utf8(effect).get_data()) : -1; }

bool NeuralVFXStage::set_camera(const Vector2& at) { return stage_ && nvfx_stage_set_camera(stage_, static_cast<float>(at.x), static_cast<float>(at.y)) == NVFX_OK; }

String NeuralVFXStage::get_stage_script() const {
  if (!stage_) return String();
  std::string text(nvfx_stage_script(stage_, nullptr, 0) + 1, '\0');
  nvfx_stage_script(stage_, text.data(), text.size());
  return String::utf8(text.c_str());
}

void NeuralVFXStage::_bind_methods() {
  ClassDB::bind_method(D_METHOD("set_library_dir", "path"), &NeuralVFXStage::set_library_dir);
  ClassDB::bind_method(D_METHOD("get_library_dir"), &NeuralVFXStage::get_library_dir);
  ClassDB::bind_method(D_METHOD("set_stage_size", "size"), &NeuralVFXStage::set_stage_size);
  ClassDB::bind_method(D_METHOD("get_stage_size"), &NeuralVFXStage::get_stage_size);
  ClassDB::bind_method(D_METHOD("set_ground", "y"), &NeuralVFXStage::set_ground);
  ClassDB::bind_method(D_METHOD("get_ground"), &NeuralVFXStage::get_ground);
  ClassDB::bind_method(D_METHOD("set_rooms", "rooms"), &NeuralVFXStage::set_rooms);
  ClassDB::bind_method(D_METHOD("get_rooms"), &NeuralVFXStage::get_rooms);
  ADD_PROPERTY(PropertyInfo(Variant::STRING, "library_dir", PROPERTY_HINT_DIR), "set_library_dir", "get_library_dir");
  ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "stage_size"), "set_stage_size", "get_stage_size");
  ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ground"), "set_ground", "get_ground");
  ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "rooms"), "set_rooms", "get_rooms");

  ClassDB::bind_method(D_METHOD("open_library"), &NeuralVFXStage::open_library);
  ClassDB::bind_method(D_METHOD("list_effects"), &NeuralVFXStage::list_effects);
  ClassDB::bind_method(D_METHOD("get_categories"), &NeuralVFXStage::get_categories);
  ClassDB::bind_method(D_METHOD("get_effect_info", "effect"), &NeuralVFXStage::get_effect_info);
  ClassDB::bind_method(D_METHOD("spawn", "effect", "position", "scale", "time"), &NeuralVFXStage::spawn, DEFVAL(1.0), DEFVAL(-1.0));
  ClassDB::bind_method(D_METHOD("set_param", "fx", "param", "value"), &NeuralVFXStage::set_param);
  ClassDB::bind_method(D_METHOD("get_param", "fx", "param"), &NeuralVFXStage::get_param);
  ClassDB::bind_method(D_METHOD("move_fx", "fx", "position"), &NeuralVFXStage::move_fx);
  ClassDB::bind_method(D_METHOD("stop", "fx", "time"), &NeuralVFXStage::stop, DEFVAL(-1.0));
  ClassDB::bind_method(D_METHOD("get_fx_info", "fx"), &NeuralVFXStage::get_fx_info);
  ClassDB::bind_method(D_METHOD("free_rooms", "effect"), &NeuralVFXStage::free_rooms);
  ClassDB::bind_method(D_METHOD("set_camera", "position"), &NeuralVFXStage::set_camera);
  ClassDB::bind_method(D_METHOD("get_stage_script"), &NeuralVFXStage::get_stage_script);
}

}  // namespace godot
