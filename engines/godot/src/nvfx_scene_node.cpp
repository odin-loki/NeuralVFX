// NeuralVFXScene (nvfx_nodes.hpp): a scene script played through the scene C API (include/neuralfx/nvfx_scene.h).
#include "nvfx_nodes.hpp"

#include <godot_cpp/classes/engine.hpp>
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

void collect_effect(void* user, const char* name, const char* file) {
  static_cast<std::vector<std::pair<std::string, std::string>>*>(user)->emplace_back(name, file);
}

}  // namespace

NeuralVFXScene::~NeuralVFXScene() { unload_scene(); }

void NeuralVFXScene::fail(const String& message, int line, int column) {
  error_ = message;
  error_line_ = line;
  error_column_ = column;
  UtilityFunctions::push_error("NeuralVFXScene: ", message);
}

void NeuralVFXScene::unload_scene() {
  if (scene_) nvfx_scene_free(scene_);
  scene_ = nullptr;
  drawn_ = -1;
}

bool NeuralVFXScene::load_scene() {
  unload_scene();
  tried_ = true;
  error_ = String();
  error_line_ = error_column_ = 0;
  finished_ = false;
  if (scene_script_.is_empty()) {
    fail("no scene_script");
    return false;
  }
  // The script and its effects are read here (FileAccess: res://, user:// or absolute paths) and passed in memory.
  const String text = FileAccess::get_file_as_string(scene_script_);
  if (text.is_empty()) {
    fail("cannot read " + scene_script_);
    return false;
  }
  const CharString script = utf8(text), source = utf8(scene_script_.get_file());
  std::vector<std::pair<std::string, std::string>> files;
  nvfx_scene_error err{};
  if (nvfx_scene_list_effects(script.get_data(), source.get_data(), collect_effect, &files, &err) != NVFX_OK) {
    fail(String::utf8(err.message), err.line, err.column);
    return false;
  }
  const String dir = effects_dir_.is_empty() ? scene_script_.get_base_dir() : effects_dir_;
  std::vector<nvfx_effect*> loaded;
  std::vector<nvfx_scene_effect> list;
  const auto free_loaded = [&] {
    for (nvfx_effect* e : loaded) nvfx_effect_free(e);
  };
  for (const auto& [name, file] : files) {
    const String f = String::utf8(file.c_str());
    const String path = f.is_absolute_path() ? f : dir.path_join(f);
    const PackedByteArray bytes = FileAccess::get_file_as_bytes(path);
    nvfx_effect* e = nullptr;
    const nvfx_status s = bytes.is_empty() ? NVFX_ERROR_IO : nvfx_effect_load_memory(bytes.ptr(), static_cast<size_t>(bytes.size()), &e);
    if (s != NVFX_OK) {
      free_loaded();
      fail(vformat("effect '%s': cannot load %s (%s)", String::utf8(name.c_str()), path, String(nvfx_status_string(s))));
      return false;
    }
    loaded.push_back(e);
    list.push_back({file.c_str(), e});
  }
  nvfx_scene_desc d;
  nvfx_scene_desc_init(&d);
  d.script = script.get_data();
  d.source_name = source.get_data();
  d.effects = list.data();
  d.n_effects = static_cast<int>(list.size());
  d.threads = threads_ > 0 ? threads_ : 2;
  d.overlap = overlap_ ? 1 : 0;
  d.width = output_size_.x > 0 ? output_size_.x : 0;
  d.height = output_size_.y > 0 ? output_size_.y : 0;
  const nvfx_status s = nvfx_scene_create(&d, &scene_, &err);
  free_loaded();  // the scene keeps copies
  if (s != NVFX_OK) {
    scene_ = nullptr;
    fail(String::utf8(err.message), err.line, err.column);
    return false;
  }
  nvfx_scene_info info{};
  nvfx_scene_get_info(scene_, &info);
  width_ = info.width;
  height_ = info.height;
  pixels_.resize(static_cast<int64_t>(width_) * height_ * 4);
  image_ = Image::create_empty(width_, height_, false, Image::FORMAT_RGBA8);
  texture_ = ImageTexture::create_from_image(image_);
  draw();
  return true;
}

void NeuralVFXScene::draw() {
  if (!scene_) return;
  nvfx_scene_render(scene_, pixels_.ptrw(), static_cast<size_t>(width_) * 4);
  image_->set_data(width_, height_, false, Image::FORMAT_RGBA8, pixels_);
  texture_->update(image_);
  drawn_ = nvfx_scene_frame(scene_);
  emit_signal("frame_drawn", drawn_);
}

void NeuralVFXScene::advance(double seconds) {
  if (!scene_ || !(seconds >= 0.0)) return;
  nvfx_scene_step(scene_, seconds, nullptr);
  if (nvfx_scene_frame(scene_) != drawn_) draw();
  if (get_time() >= get_length()) {
    if (looping_) {
      restart();
    } else if (!finished_) {
      finished_ = true;
      playing_ = false;
      emit_signal("finished");
    }
  }
}

void NeuralVFXScene::seek(double seconds) {
  if (!scene_ || !(seconds >= 0.0)) return;
  nvfx_scene_seek(scene_, seconds);
  finished_ = false;
  draw();
}

void NeuralVFXScene::restart() {
  if (!scene_) return;
  nvfx_scene_restart(scene_);
  finished_ = false;
  draw();
}

double NeuralVFXScene::get_time() const { return scene_ ? nvfx_scene_time(scene_) : 0.0; }
int NeuralVFXScene::get_frame() const { return scene_ ? nvfx_scene_frame(scene_) : 0; }

double NeuralVFXScene::get_length() const {
  nvfx_scene_info i{};
  return scene_ && nvfx_scene_get_info(scene_, &i) == NVFX_OK ? static_cast<double>(i.length) : 0.0;
}

double NeuralVFXScene::get_fps() const {
  nvfx_scene_info i{};
  return scene_ && nvfx_scene_get_info(scene_, &i) == NVFX_OK ? static_cast<double>(i.fps) : 0.0;
}

Vector2i NeuralVFXScene::get_scene_size() const {
  nvfx_scene_info i{};
  return scene_ && nvfx_scene_get_info(scene_, &i) == NVFX_OK ? Vector2i(i.scene_width, i.scene_height) : Vector2i();
}

Dictionary NeuralVFXScene::sample(const Vector2& at) const {
  Dictionary d;
  nvfx_scene_fields s{};
  if (scene_) nvfx_scene_sample(scene_, static_cast<float>(at.x), static_cast<float>(at.y), &s);
  d["heat"] = s.heat;
  d["soot"] = s.soot;
  d["velocity"] = Vector2(s.u, s.v);
  return d;
}

float NeuralVFXScene::get_heat(const Vector2& at) const {
  nvfx_scene_fields s{};
  if (scene_) nvfx_scene_sample(scene_, static_cast<float>(at.x), static_cast<float>(at.y), &s);
  return s.heat;
}

float NeuralVFXScene::get_soot(const Vector2& at) const {
  nvfx_scene_fields s{};
  if (scene_) nvfx_scene_sample(scene_, static_cast<float>(at.x), static_cast<float>(at.y), &s);
  return s.soot;
}

Vector2 NeuralVFXScene::get_velocity(const Vector2& at) const {
  nvfx_scene_fields s{};
  if (scene_) nvfx_scene_sample(scene_, static_cast<float>(at.x), static_cast<float>(at.y), &s);
  return Vector2(s.u, s.v);
}

PackedFloat32Array NeuralVFXScene::get_field_grid(Field field, const Vector2& origin, const Vector2& step, const Vector2i& count) const {
  PackedFloat32Array out;
  if (!scene_ || count.x <= 0 || count.y <= 0) return out;
  out.resize(static_cast<int64_t>(count.x) * count.y);
  nvfx_scene_sample_grid(scene_, static_cast<nvfx_scene_field>(field), static_cast<float>(origin.x), static_cast<float>(origin.y), static_cast<float>(step.x),
                         static_cast<float>(step.y), count.x, count.y, out.ptrw(), static_cast<size_t>(count.x));
  return out;
}

float NeuralVFXScene::get_field_max(Field field, const Rect2& r) const {
  float mx = 0.f;
  if (scene_) {
    nvfx_scene_field_region(scene_, static_cast<nvfx_scene_field>(field), static_cast<float>(r.position.x), static_cast<float>(r.position.y),
                            static_cast<float>(r.position.x + r.size.x), static_cast<float>(r.position.y + r.size.y), &mx, nullptr);
  }
  return mx;
}

float NeuralVFXScene::get_field_mean(Field field, const Rect2& r) const {
  float mean = 0.f;
  if (scene_) {
    nvfx_scene_field_region(scene_, static_cast<nvfx_scene_field>(field), static_cast<float>(r.position.x), static_cast<float>(r.position.y),
                            static_cast<float>(r.position.x + r.size.x), static_cast<float>(r.position.y + r.size.y), nullptr, &mean);
  }
  return mean;
}

bool NeuralVFXScene::set_input(const String& name, float value) { return scene_ && nvfx_scene_set_input(scene_, utf8(name).get_data(), value) == NVFX_OK; }

float NeuralVFXScene::get_input(const String& name) const {
  float v = 0.f;
  if (scene_) nvfx_scene_get_input(scene_, utf8(name).get_data(), &v);
  return v;
}

PackedStringArray NeuralVFXScene::get_input_names() const {
  PackedStringArray out;
  for (int i = 0; scene_ && nvfx_scene_input_name(scene_, i); ++i) out.push_back(String::utf8(nvfx_scene_input_name(scene_, i)));
  return out;
}

bool NeuralVFXScene::trigger(const String& rule) { return scene_ && nvfx_scene_trigger(scene_, utf8(rule).get_data()) == NVFX_OK; }

int NeuralVFXScene::get_rule_count(const String& rule) const {
  int n = -1;
  if (scene_) nvfx_scene_rule_state(scene_, utf8(rule).get_data(), &n, nullptr);
  return n;
}

PackedStringArray NeuralVFXScene::get_rule_names() const {
  PackedStringArray out;
  for (int i = 0; scene_ && nvfx_scene_rule_name(scene_, i); ++i) out.push_back(String::utf8(nvfx_scene_rule_name(scene_, i)));
  return out;
}

PackedStringArray NeuralVFXScene::get_module_names() const {
  PackedStringArray out;
  for (int i = 0; scene_ && nvfx_scene_module_name(scene_, i); ++i) out.push_back(String::utf8(nvfx_scene_module_name(scene_, i)));
  return out;
}

Dictionary NeuralVFXScene::get_module_info(const String& module) const {
  Dictionary d;
  nvfx_scene_module_info m{};
  const CharString name = utf8(module);
  if (!scene_ || nvfx_scene_module_get_info(scene_, name.get_data(), &m) != NVFX_OK) return d;
  d["position"] = Vector2(m.x, m.y);
  d["width"] = m.width;
  d["started"] = std::isinf(m.started) ? -1.0 : static_cast<double>(m.started);
  d["active"] = m.active != 0;
  d["tiles"] = m.tiles;
  Dictionary controls;
  for (int c = 0; c < m.n_controls && c < 8; ++c) controls[String::utf8(nvfx_scene_module_control_name(scene_, name.get_data(), c))] = m.controls[c];
  d["controls"] = controls;
  return d;
}

bool NeuralVFXScene::move_module(const String& module, const Vector2& at) {
  return scene_ && nvfx_scene_module_place(scene_, utf8(module).get_data(), static_cast<float>(at.x), static_cast<float>(at.y)) == NVFX_OK;
}

bool NeuralVFXScene::set_module_control(const String& module, const String& control, float value) {
  return scene_ && nvfx_scene_module_set_control(scene_, utf8(module).get_data(), utf8(control).get_data(), value) == NVFX_OK;
}

void NeuralVFXScene::_ready() {
  if (Engine::get_singleton()->is_editor_hint()) {
    set_process(false);  // nothing plays in the editor
    return;
  }
  if (!tried_ && !scene_script_.is_empty()) load_scene();  // unless the game loaded it (or tried) before
  set_process(true);
}

void NeuralVFXScene::_process(double delta) {
  if (playing_) advance(delta * speed_);
}

void NeuralVFXScene::_bind_methods() {
  ClassDB::bind_method(D_METHOD("set_scene_script", "path"), &NeuralVFXScene::set_scene_script);
  ClassDB::bind_method(D_METHOD("get_scene_script"), &NeuralVFXScene::get_scene_script);
  ClassDB::bind_method(D_METHOD("set_effects_dir", "path"), &NeuralVFXScene::set_effects_dir);
  ClassDB::bind_method(D_METHOD("get_effects_dir"), &NeuralVFXScene::get_effects_dir);
  ClassDB::bind_method(D_METHOD("set_threads", "threads"), &NeuralVFXScene::set_threads);
  ClassDB::bind_method(D_METHOD("get_threads"), &NeuralVFXScene::get_threads);
  ClassDB::bind_method(D_METHOD("set_overlap", "on"), &NeuralVFXScene::set_overlap);
  ClassDB::bind_method(D_METHOD("get_overlap"), &NeuralVFXScene::get_overlap);
  ClassDB::bind_method(D_METHOD("set_output_size", "size"), &NeuralVFXScene::set_output_size);
  ClassDB::bind_method(D_METHOD("get_output_size"), &NeuralVFXScene::get_output_size);
  ClassDB::bind_method(D_METHOD("set_playing", "on"), &NeuralVFXScene::set_playing);
  ClassDB::bind_method(D_METHOD("is_playing"), &NeuralVFXScene::is_playing);
  ClassDB::bind_method(D_METHOD("set_looping", "on"), &NeuralVFXScene::set_looping);
  ClassDB::bind_method(D_METHOD("is_looping"), &NeuralVFXScene::is_looping);
  ClassDB::bind_method(D_METHOD("set_speed", "speed"), &NeuralVFXScene::set_speed);
  ClassDB::bind_method(D_METHOD("get_speed"), &NeuralVFXScene::get_speed);
  ClassDB::bind_method(D_METHOD("get_texture"), &NeuralVFXScene::get_texture);
  ClassDB::bind_method(D_METHOD("get_image"), &NeuralVFXScene::get_image);
  ADD_PROPERTY(PropertyInfo(Variant::STRING, "scene_script", PROPERTY_HINT_FILE, "*.nvfxs"), "set_scene_script", "get_scene_script");
  ADD_PROPERTY(PropertyInfo(Variant::STRING, "effects_dir", PROPERTY_HINT_DIR), "set_effects_dir", "get_effects_dir");
  ADD_PROPERTY(PropertyInfo(Variant::INT, "threads", PROPERTY_HINT_RANGE, "1,64"), "set_threads", "get_threads");
  ADD_PROPERTY(PropertyInfo(Variant::BOOL, "overlap"), "set_overlap", "get_overlap");
  ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "output_size"), "set_output_size", "get_output_size");
  ADD_PROPERTY(PropertyInfo(Variant::BOOL, "playing"), "set_playing", "is_playing");
  ADD_PROPERTY(PropertyInfo(Variant::BOOL, "looping"), "set_looping", "is_looping");
  ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed"), "set_speed", "get_speed");
  ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "ImageTexture", PROPERTY_USAGE_NONE), "", "get_texture");

  ClassDB::bind_method(D_METHOD("load_scene"), &NeuralVFXScene::load_scene);
  ClassDB::bind_method(D_METHOD("unload_scene"), &NeuralVFXScene::unload_scene);
  ClassDB::bind_method(D_METHOD("is_loaded"), &NeuralVFXScene::is_loaded);
  ClassDB::bind_method(D_METHOD("get_error"), &NeuralVFXScene::get_error);
  ClassDB::bind_method(D_METHOD("get_error_line"), &NeuralVFXScene::get_error_line);
  ClassDB::bind_method(D_METHOD("get_error_column"), &NeuralVFXScene::get_error_column);
  ClassDB::bind_method(D_METHOD("advance", "seconds"), &NeuralVFXScene::advance);
  ClassDB::bind_method(D_METHOD("seek", "seconds"), &NeuralVFXScene::seek);
  ClassDB::bind_method(D_METHOD("restart"), &NeuralVFXScene::restart);
  ClassDB::bind_method(D_METHOD("get_time"), &NeuralVFXScene::get_time);
  ClassDB::bind_method(D_METHOD("get_frame"), &NeuralVFXScene::get_frame);
  ClassDB::bind_method(D_METHOD("get_length"), &NeuralVFXScene::get_length);
  ClassDB::bind_method(D_METHOD("get_fps"), &NeuralVFXScene::get_fps);
  ClassDB::bind_method(D_METHOD("get_scene_size"), &NeuralVFXScene::get_scene_size);
  ClassDB::bind_method(D_METHOD("sample", "position"), &NeuralVFXScene::sample);
  ClassDB::bind_method(D_METHOD("get_heat", "position"), &NeuralVFXScene::get_heat);
  ClassDB::bind_method(D_METHOD("get_soot", "position"), &NeuralVFXScene::get_soot);
  ClassDB::bind_method(D_METHOD("get_velocity", "position"), &NeuralVFXScene::get_velocity);
  ClassDB::bind_method(D_METHOD("get_field_grid", "field", "origin", "step", "count"), &NeuralVFXScene::get_field_grid);
  ClassDB::bind_method(D_METHOD("get_field_max", "field", "rect"), &NeuralVFXScene::get_field_max);
  ClassDB::bind_method(D_METHOD("get_field_mean", "field", "rect"), &NeuralVFXScene::get_field_mean);
  ClassDB::bind_method(D_METHOD("set_input", "name", "value"), &NeuralVFXScene::set_input);
  ClassDB::bind_method(D_METHOD("get_input", "name"), &NeuralVFXScene::get_input);
  ClassDB::bind_method(D_METHOD("get_input_names"), &NeuralVFXScene::get_input_names);
  ClassDB::bind_method(D_METHOD("trigger", "rule"), &NeuralVFXScene::trigger);
  ClassDB::bind_method(D_METHOD("get_rule_count", "rule"), &NeuralVFXScene::get_rule_count);
  ClassDB::bind_method(D_METHOD("get_rule_names"), &NeuralVFXScene::get_rule_names);
  ClassDB::bind_method(D_METHOD("get_module_names"), &NeuralVFXScene::get_module_names);
  ClassDB::bind_method(D_METHOD("get_module_info", "module"), &NeuralVFXScene::get_module_info);
  ClassDB::bind_method(D_METHOD("move_module", "module", "position"), &NeuralVFXScene::move_module);
  ClassDB::bind_method(D_METHOD("set_module_control", "module", "control", "value"), &NeuralVFXScene::set_module_control);

  ADD_SIGNAL(MethodInfo("frame_drawn", PropertyInfo(Variant::INT, "frame")));
  ADD_SIGNAL(MethodInfo("finished"));
  BIND_ENUM_CONSTANT(FIELD_HEAT);
  BIND_ENUM_CONSTANT(FIELD_SOOT);
  BIND_ENUM_CONSTANT(FIELD_U);
  BIND_ENUM_CONSTANT(FIELD_V);
}

}  // namespace godot
