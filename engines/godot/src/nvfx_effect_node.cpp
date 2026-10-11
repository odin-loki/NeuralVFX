// NeuralVFXEffect (nvfx_nodes.hpp): one effect played through the C API (include/neuralfx/nvfx.h).
#include "nvfx_nodes.hpp"

#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {

NeuralVFXEffect::~NeuralVFXEffect() { unload(); }

void NeuralVFXEffect::unload() {
  if (instance_) nvfx_instance_free(instance_);
  if (effect_) nvfx_effect_free(effect_);
  instance_ = nullptr;
  effect_ = nullptr;
}

bool NeuralVFXEffect::load_effect() {
  unload();
  tried_ = true;
  error_ = String();
  const PackedByteArray bytes = FileAccess::get_file_as_bytes(effect_path_);
  nvfx_status s = bytes.is_empty() ? NVFX_ERROR_IO : nvfx_effect_load_memory(bytes.ptr(), static_cast<size_t>(bytes.size()), &effect_);
  if (s == NVFX_OK) s = nvfx_instance_create(effect_, size_, &instance_);
  if (s != NVFX_OK) {
    error_ = vformat("cannot load %s at %d px: %s", effect_path_, size_, String(nvfx_status_string(s)));
    UtilityFunctions::push_error("NeuralVFXEffect: ", error_);
    unload();
    return false;
  }
  apply_settings();
  pixels_.resize(static_cast<int64_t>(size_) * size_ * 4);
  image_ = Image::create_empty(size_, size_, false, Image::FORMAT_RGBA8);
  texture_ = ImageTexture::create_from_image(image_);
  render_at(time_);
  return true;
}

void NeuralVFXEffect::apply_settings() {
  if (!instance_) return;
  if (!controls_.is_empty()) nvfx_instance_set_controls(instance_, controls_.ptr(), static_cast<int>(controls_.size()));
  nvfx_instance_set_seed(instance_, static_cast<uint64_t>(seed_));
  if (drift_ >= 0.0) nvfx_instance_set_drift(instance_, static_cast<float>(drift_));
  nvfx_instance_set_colour(instance_, static_cast<float>(hue_), static_cast<float>(brightness_));
}

void NeuralVFXEffect::set_controls(const PackedFloat32Array& c) {
  controls_ = c;
  apply_settings();
}
void NeuralVFXEffect::set_seed(int64_t seed) {
  seed_ = seed;
  apply_settings();
}
void NeuralVFXEffect::set_drift(double seconds) {
  drift_ = seconds;
  apply_settings();
}
void NeuralVFXEffect::set_hue(double radians) {
  hue_ = radians;
  apply_settings();
}
void NeuralVFXEffect::set_brightness(double b) {
  brightness_ = b;
  apply_settings();
}

void NeuralVFXEffect::render_at(double seconds) {
  time_ = seconds;
  if (!instance_) return;
  nvfx_render(instance_, seconds, pixels_.ptrw(), static_cast<size_t>(size_) * 4);
  image_->set_data(size_, size_, false, Image::FORMAT_RGBA8, pixels_);
  texture_->update(image_);
}

PackedStringArray NeuralVFXEffect::get_control_names() const {
  PackedStringArray out;
  for (int i = 0; effect_ && nvfx_effect_control_name(effect_, i); ++i) out.push_back(String::utf8(nvfx_effect_control_name(effect_, i)));
  return out;
}

Dictionary NeuralVFXEffect::get_info() const {
  Dictionary d;
  nvfx_effect_info i{};
  if (!effect_ || nvfx_effect_get_info(effect_, &i) != NVFX_OK) return d;
  d["name"] = String::utf8(i.name);
  d["arch"] = i.arch;
  d["native_size"] = i.native_size;
  d["frames"] = i.frames;
  d["fps"] = i.fps;
  d["loops"] = i.loops != 0;
  d["controls"] = i.n_controls;
  d["variations"] = i.n_variations;
  d["stored_bytes"] = static_cast<int64_t>(i.stored_bytes);
  d["resident_bytes"] = static_cast<int64_t>(i.resident_bytes);
  return d;
}

void NeuralVFXEffect::_ready() {
  if (Engine::get_singleton()->is_editor_hint()) {
    set_process(false);
    return;
  }
  if (!tried_ && !effect_path_.is_empty()) load_effect();  // unless the game loaded it (or tried) before
  set_process(true);
}

void NeuralVFXEffect::_process(double delta) {
  if (playing_ && instance_) render_at(time_ + delta * speed_);
}

void NeuralVFXEffect::_bind_methods() {
  ClassDB::bind_method(D_METHOD("set_effect_path", "path"), &NeuralVFXEffect::set_effect_path);
  ClassDB::bind_method(D_METHOD("get_effect_path"), &NeuralVFXEffect::get_effect_path);
  ClassDB::bind_method(D_METHOD("set_size", "size"), &NeuralVFXEffect::set_size);
  ClassDB::bind_method(D_METHOD("get_size"), &NeuralVFXEffect::get_size);
  ClassDB::bind_method(D_METHOD("set_controls", "controls"), &NeuralVFXEffect::set_controls);
  ClassDB::bind_method(D_METHOD("get_controls"), &NeuralVFXEffect::get_controls);
  ClassDB::bind_method(D_METHOD("set_seed", "seed"), &NeuralVFXEffect::set_seed);
  ClassDB::bind_method(D_METHOD("get_seed"), &NeuralVFXEffect::get_seed);
  ClassDB::bind_method(D_METHOD("set_drift", "seconds"), &NeuralVFXEffect::set_drift);
  ClassDB::bind_method(D_METHOD("get_drift"), &NeuralVFXEffect::get_drift);
  ClassDB::bind_method(D_METHOD("set_hue", "radians"), &NeuralVFXEffect::set_hue);
  ClassDB::bind_method(D_METHOD("get_hue"), &NeuralVFXEffect::get_hue);
  ClassDB::bind_method(D_METHOD("set_brightness", "brightness"), &NeuralVFXEffect::set_brightness);
  ClassDB::bind_method(D_METHOD("get_brightness"), &NeuralVFXEffect::get_brightness);
  ClassDB::bind_method(D_METHOD("set_playing", "on"), &NeuralVFXEffect::set_playing);
  ClassDB::bind_method(D_METHOD("is_playing"), &NeuralVFXEffect::is_playing);
  ClassDB::bind_method(D_METHOD("set_speed", "speed"), &NeuralVFXEffect::set_speed);
  ClassDB::bind_method(D_METHOD("get_speed"), &NeuralVFXEffect::get_speed);
  ClassDB::bind_method(D_METHOD("set_time", "seconds"), &NeuralVFXEffect::set_time);
  ClassDB::bind_method(D_METHOD("get_time"), &NeuralVFXEffect::get_time);
  ClassDB::bind_method(D_METHOD("get_texture"), &NeuralVFXEffect::get_texture);
  ClassDB::bind_method(D_METHOD("get_image"), &NeuralVFXEffect::get_image);
  ADD_PROPERTY(PropertyInfo(Variant::STRING, "effect_path", PROPERTY_HINT_FILE, "*.nvfx"), "set_effect_path", "get_effect_path");
  ADD_PROPERTY(PropertyInfo(Variant::INT, "size", PROPERTY_HINT_RANGE, "16,1024,16"), "set_size", "get_size");
  ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "controls"), "set_controls", "get_controls");
  ADD_PROPERTY(PropertyInfo(Variant::INT, "seed"), "set_seed", "get_seed");
  ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "drift"), "set_drift", "get_drift");
  ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "hue"), "set_hue", "get_hue");
  ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "brightness"), "set_brightness", "get_brightness");
  ADD_PROPERTY(PropertyInfo(Variant::BOOL, "playing"), "set_playing", "is_playing");
  ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed"), "set_speed", "get_speed");
  ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time"), "set_time", "get_time");
  ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "ImageTexture", PROPERTY_USAGE_NONE), "", "get_texture");

  ClassDB::bind_method(D_METHOD("load_effect"), &NeuralVFXEffect::load_effect);
  ClassDB::bind_method(D_METHOD("is_loaded"), &NeuralVFXEffect::is_loaded);
  ClassDB::bind_method(D_METHOD("get_error"), &NeuralVFXEffect::get_error);
  ClassDB::bind_method(D_METHOD("render_at", "seconds"), &NeuralVFXEffect::render_at);
  ClassDB::bind_method(D_METHOD("get_control_names"), &NeuralVFXEffect::get_control_names);
  ClassDB::bind_method(D_METHOD("get_info"), &NeuralVFXEffect::get_info);
}

}  // namespace godot
