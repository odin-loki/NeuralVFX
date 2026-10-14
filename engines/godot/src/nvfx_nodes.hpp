// The NeuralVFX nodes for Godot 4 (docs/ENGINES.md §8), over the C API of libnvfx:
//   NeuralVFXScene   plays a scene script (docs/COMPOSE.md §4) into an ImageTexture; GDScript reads its fields (heat,
//                    soot, velocity) and drives it (inputs, rules, modules) while it plays;
//   NeuralVFXStage   a NeuralVFXScene whose scene is a stage of the effect library (docs/LIBRARY.md): it lists the
//                    catalogue's effects and spawns, moves, drives and stops them while it plays;
//   NeuralVFXEffect  plays one effect (an .nvfx file: a frame model or a rollout effect) into an ImageTexture.
// Files are read with FileAccess (res://, user:// or absolute paths), so the nodes work from an exported package too.
#pragma once

#include <neuralfx/nvfx.h>
#include <neuralfx/nvfx_library.h>
#include <neuralfx/nvfx_scene.h>

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/image_texture.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector2i.hpp>

namespace godot {

class NeuralVFXScene : public Node {
  GDCLASS(NeuralVFXScene, Node)

 public:
  enum Field { FIELD_HEAT = NVFX_FIELD_HEAT, FIELD_SOOT = NVFX_FIELD_SOOT, FIELD_U = NVFX_FIELD_U, FIELD_V = NVFX_FIELD_V };

  NeuralVFXScene() = default;
  ~NeuralVFXScene() override;

  // Properties: where the scene comes from and how it plays (a change takes effect at the next load_scene()).
  void set_scene_script(const String& path) { scene_script_ = path; }
  String get_scene_script() const { return scene_script_; }
  void set_effects_dir(const String& path) { effects_dir_ = path; }
  String get_effects_dir() const { return effects_dir_; }
  void set_threads(int n) { threads_ = n; }
  int get_threads() const { return threads_; }
  void set_overlap(bool on) { overlap_ = on; }
  bool get_overlap() const { return overlap_; }
  void set_output_size(const Vector2i& size) { output_size_ = size; }
  Vector2i get_output_size() const { return output_size_; }
  void set_playing(bool on) { playing_ = on; }
  bool is_playing() const { return playing_; }
  void set_looping(bool on) { looping_ = on; }
  bool is_looping() const { return looping_; }
  void set_speed(double s) { speed_ = s; }
  double get_speed() const { return speed_; }
  Ref<ImageTexture> get_texture() const { return texture_; }
  Ref<Image> get_image() const { return image_; }

  // The scene.
  bool load_scene();
  void unload_scene();
  bool is_loaded() const { return scene_ != nullptr; }
  String get_error() const { return error_; }
  int get_error_line() const { return error_line_; }
  int get_error_column() const { return error_column_; }
  void advance(double seconds);  // the clock forward, and the current frame into the texture
  void seek(double seconds);
  void restart();
  double get_time() const;
  int get_frame() const;
  double get_length() const;
  double get_fps() const;
  Vector2i get_scene_size() const;

  // Fields, for gameplay (world pixels, y down).
  Dictionary sample(const Vector2& at) const;  // heat, soot, velocity (pixels per second)
  float get_heat(const Vector2& at) const;
  float get_soot(const Vector2& at) const;
  Vector2 get_velocity(const Vector2& at) const;
  PackedFloat32Array get_field_grid(Field field, const Vector2& origin, const Vector2& step, const Vector2i& count) const;
  float get_field_max(Field field, const Rect2& rect) const;
  float get_field_mean(Field field, const Rect2& rect) const;

  // Driving the scene.
  bool set_input(const String& name, float value);
  float get_input(const String& name) const;
  PackedStringArray get_input_names() const;
  bool trigger(const String& rule);
  int get_rule_count(const String& rule) const;
  PackedStringArray get_rule_names() const;
  PackedStringArray get_module_names() const;
  Dictionary get_module_info(const String& module) const;
  bool move_module(const String& module, const Vector2& at);
  bool set_module_control(const String& module, const String& control, float value);

  void _ready() override;
  void _process(double delta) override;

 protected:
  static void _bind_methods();
  // What a subclass changes: where the scene comes from (the script, by default) and how it goes.
  virtual bool can_load() const { return !scene_script_.is_empty(); }
  virtual nvfx_scene* create_scene();  // nullptr after fail()
  virtual void release_scene();        // frees scene_
  virtual void restart_scene();        // back to time 0
  void fail(const String& message, int line = 0, int column = 0);
  nvfx_scene* scene_ = nullptr;

 private:
  void draw();
  String scene_script_, effects_dir_, error_;
  int error_line_ = 0, error_column_ = 0;
  int threads_ = 2;
  bool overlap_ = true, playing_ = true, looping_ = false, finished_ = false, tried_ = false;
  double speed_ = 1.0;
  Vector2i output_size_;
  int width_ = 0, height_ = 0, drawn_ = -1;
  PackedByteArray pixels_;
  Ref<Image> image_;
  Ref<ImageTexture> texture_;
};

class NeuralVFXStage : public NeuralVFXScene {
  GDCLASS(NeuralVFXStage, NeuralVFXScene)

 public:
  NeuralVFXStage() = default;
  ~NeuralVFXStage() override;

  // Properties (a change takes effect at the next load_scene()): the catalogue (a folder of .nvfxl files), its effects'
  // folder (effects_dir, the scene's), the picture and the ground, and the rooms: {"campfire": 4, "grenade": [8, true]}
  // (how many of each may play at once; with true, a spawn when all play replaces the oldest).
  void set_library_dir(const String& path) { library_dir_ = path; }
  String get_library_dir() const { return library_dir_; }
  void set_stage_size(const Vector2i& size) { stage_size_ = size; }
  Vector2i get_stage_size() const { return stage_size_; }
  void set_ground(double y) { ground_ = y; }
  double get_ground() const { return ground_; }
  void set_rooms(const Dictionary& rooms) { rooms_ = rooms; }
  Dictionary get_rooms() const { return rooms_; }

  // The catalogue (read by open_library(), or by load_scene()).
  bool open_library();
  PackedStringArray list_effects() const;
  PackedStringArray get_categories() const;
  Dictionary get_effect_info(const String& effect) const;  // category, description, models, elements, size, ...

  // Instances (handles: 0 is none).
  int64_t spawn(const String& effect, const Vector2& at, double scale = 1.0, double time = -1.0);
  bool set_param(int64_t fx, const String& param, double value);
  double get_param(int64_t fx, const String& param) const;
  bool move_fx(int64_t fx, const Vector2& at);
  bool stop(int64_t fx, double time = -1.0);
  Dictionary get_fx_info(int64_t fx) const;  // state ("waiting", "playing", "ended", "none"), effect, position, ...
  int free_rooms(const String& effect) const;
  bool set_camera(const Vector2& at);
  String get_stage_script() const;

 protected:
  static void _bind_methods();
  bool can_load() const override { return !library_dir_.is_empty(); }
  nvfx_scene* create_scene() override;
  void release_scene() override;
  void restart_scene() override;

 private:
  void close_library();
  nvfx_library* library_ = nullptr;
  nvfx_stage* stage_ = nullptr;
  String library_dir_;
  Vector2i stage_size_{960, 540};
  double ground_ = 0.0;
  Dictionary rooms_;
};

class NeuralVFXEffect : public Node {
  GDCLASS(NeuralVFXEffect, Node)

 public:
  NeuralVFXEffect() = default;
  ~NeuralVFXEffect() override;

  void set_effect_path(const String& path) { effect_path_ = path; }
  String get_effect_path() const { return effect_path_; }
  void set_size(int size) { size_ = size; }
  int get_size() const { return size_; }
  void set_controls(const PackedFloat32Array& c);
  PackedFloat32Array get_controls() const { return controls_; }
  void set_seed(int64_t seed);
  int64_t get_seed() const { return seed_; }
  void set_drift(double seconds);  // < 0: the effect's default
  double get_drift() const { return drift_; }
  void set_hue(double radians);
  double get_hue() const { return hue_; }
  void set_brightness(double b);
  double get_brightness() const { return brightness_; }
  void set_playing(bool on) { playing_ = on; }
  bool is_playing() const { return playing_; }
  void set_speed(double s) { speed_ = s; }
  double get_speed() const { return speed_; }
  void set_time(double t) { time_ = t; }
  double get_time() const { return time_; }
  Ref<ImageTexture> get_texture() const { return texture_; }
  Ref<Image> get_image() const { return image_; }

  bool load_effect();
  bool is_loaded() const { return instance_ != nullptr; }
  String get_error() const { return error_; }
  void render_at(double seconds);  // the frame at `seconds` into the texture (premultiplied alpha)
  PackedStringArray get_control_names() const;
  Dictionary get_info() const;

  void _ready() override;
  void _process(double delta) override;

 protected:
  static void _bind_methods();

 private:
  void apply_settings();
  void unload();
  nvfx_effect* effect_ = nullptr;
  nvfx_instance* instance_ = nullptr;
  String effect_path_, error_;
  int size_ = 128;
  PackedFloat32Array controls_;
  int64_t seed_ = 1;
  double drift_ = -1.0, hue_ = 0.0, brightness_ = 1.0, speed_ = 1.0, time_ = 0.0;
  bool playing_ = true, tried_ = false;
  PackedByteArray pixels_;
  Ref<Image> image_;
  Ref<ImageTexture> texture_;
};

}  // namespace godot

VARIANT_ENUM_CAST(NeuralVFXScene::Field);
