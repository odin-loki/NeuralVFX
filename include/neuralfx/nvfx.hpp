// NeuralVFX for C++ engines: a header-only C++23 wrapper over the C API (nvfx.h, nvfx_scene.h, nvfx_library.h,
// nvfx_manager.h, nvfx_pack.h; docs/INTEGRATION.md).
//
//   auto pack = nvfx::Pack::open("effects.nvfxp");                 // std::expected<nvfx::Pack, nvfx::Error>
//   if (!pack) log(pack.error().message);                          // the C API's message ("effects.nvfxp: truncated: ...")
//   auto lib = nvfx::Library::open_pack(*pack);
//   nvfx::StageRooms rooms = {{"campfire", 4, 0}, {"grenade", 8, 1}};
//   auto desc = nvfx::stage_desc();
//   desc.rooms = rooms.data();
//   desc.n_rooms = static_cast<int>(rooms.size());
//   auto stage = nvfx::Stage::create(*lib, desc);
//   auto fire = stage->spawn("campfire", 300, 470);
//   every frame: stage->scene().step(dt); stage->scene().render(rgba);   // std::span<std::uint8_t>
//
// - Handles (Effect, Instance, Scene, Library, Stage, Manager, Pack) own their C object and free it; they are move-only.
//   get() gives the C pointer, release() gives it up. SceneView is the non-owning view a Stage's scene is.
// - Errors are std::expected<T, nvfx::Error>: the status, the C API's message (with the file, line and column for
//   scripts and catalogues) and the position. No exceptions are thrown: the header builds with -fno-exceptions.
// - Buffers are std::span; their sizes are checked against the picture before the C call (NVFX_ERROR_ARGUMENT).
// - Nothing allocates where the C call does not (render, step, sample, tick, spawn, ...), except an Error on failure.
// - Names follow the C API: nvfx_stage_spawn(stage, ...) is stage.spawn(...), nvfx_scene_get_info is get_info().
// - Lifetimes are the C API's: a Stage needs its Library, a Manager the Effects it was given; a Pack may go first.
#pragma once

#include <neuralfx/nvfx.h>
#include <neuralfx/nvfx_library.h>
#include <neuralfx/nvfx_manager.h>
#include <neuralfx/nvfx_pack.h>
#include <neuralfx/nvfx_scene.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nvfx {

// --- errors ----------------------------------------------------------------------------------------------------------

struct Error {
  nvfx_status status = NVFX_ERROR_ARGUMENT;
  std::string message;   // the C API's message, or nvfx_status_string(status)
  int line = 0, column = 0;  // in a script or catalogue file, 1-based (0: not about a line)
};

template <class T = void>
using Result = std::expected<T, Error>;

inline const char* status_string(nvfx_status s) noexcept { return nvfx_status_string(s); }

// A NUL-terminated string argument: a const char* or a std::string, without copying.
class zstring {
 public:
  zstring(const char* s) noexcept : p_(s) {}  // NOLINT: implicit by design
  zstring(const std::string& s) noexcept : p_(s.c_str()) {}  // NOLINT
  const char* c_str() const noexcept { return p_; }

 private:
  const char* p_;
};

namespace detail {

inline std::unexpected<Error> fail(nvfx_status s) { return std::unexpected(Error{s, nvfx_status_string(s), 0, 0}); }
inline std::unexpected<Error> fail(nvfx_status s, const nvfx_scene_error& e) {
  return std::unexpected(Error{s, e.message[0] ? std::string(e.message) : std::string(nvfx_status_string(s)), e.line, e.column});
}
inline Result<> check(nvfx_status s) {
  if (s == NVFX_OK) return {};
  return fail(s);
}
template <class T>
Result<T> value_or_fail(nvfx_status s, const T& v) {
  if (s == NVFX_OK) return v;
  return fail(s);
}

// fn(name, file) for the C API's nvfx_scene_effect_fn callbacks. fn must not throw.
template <class F>
void effect_trampoline(void* user, const char* name, const char* file) {
  (*static_cast<F*>(user))(name, file);
}

// An owning handle: the C object, freed by Free.
template <class T, void (*Free)(T*)>
class Handle {
 public:
  Handle() noexcept = default;
  explicit Handle(T* p) noexcept : p_(p) {}
  Handle(Handle&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
  Handle& operator=(Handle&& o) noexcept {
    if (this != &o) reset(std::exchange(o.p_, nullptr));
    return *this;
  }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  ~Handle() { reset(); }
  T* get() const noexcept { return p_; }
  T* release() noexcept { return std::exchange(p_, nullptr); }
  void reset(T* p = nullptr) noexcept {
    if (p_) Free(p_);
    p_ = p;
  }
  explicit operator bool() const noexcept { return p_ != nullptr; }

 protected:
  T* p_ = nullptr;
};

// The bytes a picture of w x h RGBA8 pixels needs with rows `stride` apart (0: w * 4); 0 if the stride is too small.
inline std::size_t picture_bytes(int w, int h, std::size_t& stride) noexcept {
  const std::size_t row = static_cast<std::size_t>(w) * 4;
  if (stride == 0) stride = row;
  if (stride < row || w <= 0 || h <= 0) return 0;
  return stride * static_cast<std::size_t>(h - 1) + row;
}

}  // namespace detail

// --- version and the CPU -----------------------------------------------------------------------------------------------

struct Version {
  std::uint32_t version = 0;   // the library's NVFX_VERSION
  std::uint32_t features = 0;  // NVFX_FEATURE_* bits
  // The library is the one these headers describe: an engine checks this at start.
  bool matches_headers() const noexcept { return version == NVFX_VERSION; }
  bool has(std::uint32_t feature) const noexcept { return (features & feature) == feature; }
};

inline Version version() noexcept {
  Version v;
  v.version = nvfx_version(&v.features);
  return v;
}
inline nvfx_isa get_isa() noexcept { return nvfx_get_isa(); }
inline Result<> set_isa(nvfx_isa isa) { return detail::check(nvfx_set_isa(isa)); }

inline Result<nvfx_profile> init_profile(nvfx_profile_preset preset = NVFX_PROFILE_DEFAULT) {
  nvfx_profile p{};
  return detail::value_or_fail(nvfx_profile_init(&p, preset), p);
}
inline Result<> set_profile(const nvfx_profile& profile) { return detail::check(nvfx_set_profile(&profile)); }
inline Result<nvfx_profile> get_profile() {
  nvfx_profile p{};
  return detail::value_or_fail(nvfx_get_profile(&p), p);
}

class Pack;

// --- effects and instances (nvfx.h) ----------------------------------------------------------------------------------

class Effect : public detail::Handle<nvfx_effect, nvfx_effect_free> {
 public:
  using Handle::Handle;
  static Result<Effect> load(zstring path) {
    nvfx_effect* e = nullptr;
    if (const nvfx_status s = nvfx_effect_load(path.c_str(), &e); s != NVFX_OK) return detail::fail(s);
    return Effect(e);
  }
  static Result<Effect> load_memory(std::span<const std::uint8_t> data) {
    nvfx_effect* e = nullptr;
    if (const nvfx_status s = nvfx_effect_load_memory(data.data(), data.size(), &e); s != NVFX_OK) return detail::fail(s);
    return Effect(e);
  }
  Result<nvfx_effect_info> get_info() const {
    nvfx_effect_info i{};
    return detail::value_or_fail(nvfx_effect_get_info(p_, &i), i);
  }
  const char* control_name(int i) const noexcept { return nvfx_effect_control_name(p_, i); }  // nullptr past the last
  // The prior against drift (rollout effects): before sharing the effect and creating instances.
  Result<> attach_prior(zstring path) { return detail::check(nvfx_effect_attach_prior(p_, path.c_str())); }
  Result<> attach_prior_memory(std::span<const std::uint8_t> data) { return detail::check(nvfx_effect_attach_prior_memory(p_, data.data(), data.size())); }
  // The learned 2x upscaler (rollout effects, study P): before sharing the effect between threads.
  Result<> attach_upscaler(zstring path) { return detail::check(nvfx_effect_attach_upscaler(p_, path.c_str())); }
  Result<> attach_upscaler_memory(std::span<const std::uint8_t> data) {
    return detail::check(nvfx_effect_attach_upscaler_memory(p_, data.data(), data.size()));
  }
};

class Instance : public detail::Handle<nvfx_instance, nvfx_instance_free> {
 public:
  Instance() noexcept = default;
  Instance(nvfx_instance* p, int size) noexcept : Handle(p), size_(size) {}
  static Result<Instance> create(const Effect& effect, int size) {
    nvfx_instance* in = nullptr;
    if (const nvfx_status s = nvfx_instance_create(effect.get(), size, &in); s != NVFX_OK) return detail::fail(s);
    return Instance(in, size);
  }
  int size() const noexcept { return size_; }  // the picture's side in pixels
  std::size_t frame_bytes() const noexcept { return static_cast<std::size_t>(size_) * static_cast<std::size_t>(size_) * 4; }
  std::size_t scratch_bytes() const noexcept { return nvfx_instance_scratch_bytes(p_); }
  Result<> set_controls(std::span<const float> controls) {
    return detail::check(nvfx_instance_set_controls(p_, controls.data(), static_cast<int>(controls.size())));
  }
  Result<> set_seed(std::uint64_t seed) { return detail::check(nvfx_instance_set_seed(p_, seed)); }
  Result<> set_variation(int training_index) { return detail::check(nvfx_instance_set_variation(p_, training_index)); }
  Result<> set_drift(float seconds) { return detail::check(nvfx_instance_set_drift(p_, seconds)); }
  Result<> set_colour(float hue_radians, float brightness) { return detail::check(nvfx_instance_set_colour(p_, hue_radians, brightness)); }
  Result<> set_precision(nvfx_precision precision) { return detail::check(nvfx_instance_set_precision(p_, precision)); }
  Result<> set_prior(int every_frames, int t, float beta) { return detail::check(nvfx_instance_set_prior(p_, every_frames, t, beta)); }
  Result<> set_handoff(int frames, int fade) { return detail::check(nvfx_instance_set_handoff(p_, frames, fade)); }
  Result<> set_upscale(nvfx_upscale mode) { return detail::check(nvfx_instance_set_upscale(p_, mode)); }
  // The frame at `seconds` into `rgba` (size x size premultiplied RGBA8, rows `stride` bytes apart; 0: size * 4).
  Result<> render(double seconds, std::span<std::uint8_t> rgba, std::size_t stride = 0) {
    if (rgba.size() < detail::picture_bytes(size_, size_, stride) || stride == 0) return detail::fail(NVFX_ERROR_ARGUMENT);
    return detail::check(nvfx_render(p_, seconds, rgba.data(), stride));
  }
  // `frames` frames into a flipbook: frame f at rgba[f * frame_bytes()].
  Result<> bake(int frames, std::span<std::uint8_t> rgba) {
    if (frames < 1 || rgba.size() < frame_bytes() * static_cast<std::size_t>(frames)) return detail::fail(NVFX_ERROR_ARGUMENT);
    return detail::check(nvfx_bake(p_, frames, rgba.data()));
  }
  double macs_per_pixel() const noexcept { return nvfx_instance_macs_per_pixel(p_); }

 private:
  int size_ = 0;
};

// --- scenes (nvfx_scene.h) -------------------------------------------------------------------------------------------

inline nvfx_scene_desc scene_desc() noexcept {
  nvfx_scene_desc d;
  nvfx_scene_desc_init(&d);
  return d;
}

// A scene, not owned: a Stage's, or a Scene's view.
class SceneView {
 public:
  SceneView() noexcept = default;
  explicit SceneView(nvfx_scene* s) noexcept : s_(s) {}
  nvfx_scene* get() const noexcept { return s_; }
  explicit operator bool() const noexcept { return s_ != nullptr; }

  Result<nvfx_scene_info> get_info() const {
    nvfx_scene_info i{};
    return detail::value_or_fail(nvfx_scene_get_info(s_, &i), i);
  }
  // The clock. step returns the frames computed.
  Result<int> step(double dt_seconds) {
    int n = 0;
    return detail::value_or_fail(nvfx_scene_step(s_, dt_seconds, &n), n);
  }
  Result<> step_frames(int frames) { return detail::check(nvfx_scene_step_frames(s_, frames)); }
  Result<> seek(double seconds) { return detail::check(nvfx_scene_seek(s_, seconds)); }
  Result<> restart() { return detail::check(nvfx_scene_restart(s_)); }
  double time() const noexcept { return nvfx_scene_time(s_); }
  int frame() const noexcept { return nvfx_scene_frame(s_); }
  // The current frame into `rgba` (info.width x info.height RGBA8, rows `stride` bytes apart; 0: width * 4).
  Result<> render(std::span<std::uint8_t> rgba, std::size_t stride = 0) {
    nvfx_scene_info i{};
    if (const nvfx_status s = nvfx_scene_get_info(s_, &i); s != NVFX_OK) return detail::fail(s);
    if (rgba.size() < detail::picture_bytes(i.width, i.height, stride) || stride == 0) return detail::fail(NVFX_ERROR_ARGUMENT);
    return detail::check(nvfx_scene_render(s_, rgba.data(), stride));
  }
  // One of the engine layers (docs/ENGINES.md §11; the scene or stage must have been created with it in its `layers`)
  // into `pixels`: info.width x info.height pixels of nvfx_pixel_format_bytes(format) bytes, rows `stride` bytes apart
  // (0: width times the pixel size).
  Result<> render_layer(nvfx_scene_layer layer, nvfx_pixel_format format, std::span<std::byte> pixels, std::size_t stride = 0) {
    nvfx_scene_info i{};
    if (const nvfx_status s = nvfx_scene_get_info(s_, &i); s != NVFX_OK) return detail::fail(s);
    const std::size_t px = nvfx_pixel_format_bytes(format), row = static_cast<std::size_t>(i.width) * px;
    if (px == 0 || i.width <= 0 || i.height <= 0) return detail::fail(NVFX_ERROR_ARGUMENT);
    if (stride == 0) stride = row;
    if (stride < row || pixels.size() < stride * static_cast<std::size_t>(i.height - 1) + row) return detail::fail(NVFX_ERROR_ARGUMENT);
    return detail::check(nvfx_scene_render_layer(s_, layer, format, pixels.data(), stride));
  }
  // Fields, for gameplay.
  Result<nvfx_scene_fields> sample(float x, float y) const {
    nvfx_scene_fields f{};
    return detail::value_or_fail(nvfx_scene_sample(s_, x, y, &f), f);
  }
  // out[j * row_stride + i] for i < nx, j < ny (row_stride in floats; 0: nx).
  Result<> sample_grid(nvfx_scene_field field, float x0, float y0, float dx, float dy, int nx, int ny, std::span<float> out, std::size_t row_stride = 0) const {
    if (row_stride == 0) row_stride = static_cast<std::size_t>(nx > 0 ? nx : 0);
    if (nx < 1 || ny < 1 || row_stride < static_cast<std::size_t>(nx) || out.size() < row_stride * static_cast<std::size_t>(ny - 1) + static_cast<std::size_t>(nx))
      return detail::fail(NVFX_ERROR_ARGUMENT);
    return detail::check(nvfx_scene_sample_grid(s_, field, x0, y0, dx, dy, nx, ny, out.data(), row_stride));
  }
  struct Region {
    float max = 0.f, mean = 0.f;
  };
  Result<Region> field_region(nvfx_scene_field field, float x0, float y0, float x1, float y1) const {
    Region r;
    return detail::value_or_fail(nvfx_scene_field_region(s_, field, x0, y0, x1, y1, &r.max, &r.mean), r);
  }
  Result<std::array<float, 2>> camera() const {
    std::array<float, 2> c{};
    return detail::value_or_fail(nvfx_scene_camera(s_, &c[0], &c[1]), c);
  }
  // Inputs, rules, modules (names as the script writes them).
  Result<> set_input(zstring name, float value) { return detail::check(nvfx_scene_set_input(s_, name.c_str(), value)); }
  Result<float> get_input(zstring name) const {
    float v = 0.f;
    return detail::value_or_fail(nvfx_scene_get_input(s_, name.c_str(), &v), v);
  }
  const char* input_name(int i) const noexcept { return nvfx_scene_input_name(s_, i); }
  Result<> trigger(zstring rule) { return detail::check(nvfx_scene_trigger(s_, rule.c_str())); }
  struct RuleState {
    int count = 0;
    float last_time = 0.f;  // INFINITY: never fired
  };
  Result<RuleState> rule_state(zstring rule) const {
    RuleState r;
    return detail::value_or_fail(nvfx_scene_rule_state(s_, rule.c_str(), &r.count, &r.last_time), r);
  }
  const char* rule_name(int i) const noexcept { return nvfx_scene_rule_name(s_, i); }
  const char* module_name(int i) const noexcept { return nvfx_scene_module_name(s_, i); }
  Result<nvfx_scene_module_info> module_get_info(zstring module) const {
    nvfx_scene_module_info m{};
    return detail::value_or_fail(nvfx_scene_module_get_info(s_, module.c_str(), &m), m);
  }
  const char* module_control_name(zstring module, int i) const noexcept { return nvfx_scene_module_control_name(s_, module.c_str(), i); }
  Result<> module_place(zstring module, float x, float y) { return detail::check(nvfx_scene_module_place(s_, module.c_str(), x, y)); }
  Result<> module_set_control(zstring module, zstring control, float value) {
    return detail::check(nvfx_scene_module_set_control(s_, module.c_str(), control.c_str(), value));
  }

 protected:
  nvfx_scene* s_ = nullptr;
};

class Scene : public SceneView {
 public:
  Scene() noexcept = default;
  explicit Scene(nvfx_scene* s) noexcept : SceneView(s) {}
  Scene(Scene&& o) noexcept : SceneView(std::exchange(o.s_, nullptr)) {}
  Scene& operator=(Scene&& o) noexcept {
    if (this != &o) {
      reset();
      s_ = std::exchange(o.s_, nullptr);
    }
    return *this;
  }
  Scene(const Scene&) = delete;
  Scene& operator=(const Scene&) = delete;
  ~Scene() { reset(); }
  void reset() noexcept {
    if (s_) nvfx_scene_free(s_);
    s_ = nullptr;
  }
  nvfx_scene* release() noexcept { return std::exchange(s_, nullptr); }

  static Result<Scene> create(const nvfx_scene_desc& desc) {
    nvfx_scene* s = nullptr;
    nvfx_scene_error e{};
    if (const nvfx_status st = nvfx_scene_create(&desc, &s, &e); st != NVFX_OK) return detail::fail(st, e);
    return Scene(s);
  }
  // The script's effects from the pack (and desc.script_path may name a script of the pack).
  static Result<Scene> create_from_pack(const Pack& pack, const nvfx_scene_desc& desc);
  static Result<> check(zstring script, zstring source_name = "script") {
    nvfx_scene_error e{};
    if (const nvfx_status st = nvfx_scene_check(script.c_str(), source_name.c_str(), &e); st != NVFX_OK) return detail::fail(st, e);
    return {};
  }
  // fn(const char* name, const char* file) once per `effect NAME = "FILE"` of the script.
  template <class F>
  static Result<> list_effects(zstring script, zstring source_name, F&& fn) {
    nvfx_scene_error e{};
    auto f = std::forward<F>(fn);
    if (const nvfx_status st = nvfx_scene_list_effects(script.c_str(), source_name.c_str(), &detail::effect_trampoline<decltype(f)>, &f, &e); st != NVFX_OK)
      return detail::fail(st, e);
    return {};
  }
};

// --- the effect library (nvfx_library.h) -------------------------------------------------------------------------------

inline nvfx_library_desc library_desc() noexcept {
  nvfx_library_desc d;
  nvfx_library_desc_init(&d);
  return d;
}
inline nvfx_stage_desc stage_desc() noexcept {
  nvfx_stage_desc d;
  nvfx_stage_desc_init(&d);
  return d;
}
using StageRooms = std::vector<nvfx_stage_room>;

class Library : public detail::Handle<nvfx_library, nvfx_library_free> {
 public:
  using Handle::Handle;
  static Result<Library> open(const nvfx_library_desc& desc) {
    nvfx_library* l = nullptr;
    nvfx_scene_error e{};
    if (const nvfx_status st = nvfx_library_open(&desc, &l, &e); st != NVFX_OK) return detail::fail(st, e);
    return Library(l);
  }
  // The catalogue of a pack; its stages load their models from the pack (which may be freed first).
  static Result<Library> open_pack(const Pack& pack);
  int count() const noexcept { return nvfx_library_count(p_); }
  int find(zstring name) const noexcept { return nvfx_library_find(p_, name.c_str()); }  // -1: none
  Result<nvfx_library_effect> get_effect(int index) const {
    nvfx_library_effect e{};
    return detail::value_or_fail(nvfx_library_get_effect(p_, index, &e), e);
  }
  Result<nvfx_library_param> get_param(int effect, int param) const {
    nvfx_library_param q{};
    return detail::value_or_fail(nvfx_library_get_param(p_, effect, param, &q), q);
  }
  int category_count() const noexcept { return nvfx_library_category_count(p_); }
  const char* category(int i) const noexcept { return nvfx_library_category(p_, i); }
  // fn(const char* name, const char* file) once per learned effect the entry plays.
  template <class F>
  Result<> list_models(int effect, F&& fn) const {
    auto f = std::forward<F>(fn);
    return detail::check(nvfx_library_list_models(p_, effect, &detail::effect_trampoline<decltype(f)>, &f));
  }
};

class Stage : public detail::Handle<nvfx_stage, nvfx_stage_free> {
 public:
  using Handle::Handle;
  // The library must outlive the stage.
  static Result<Stage> create(const Library& library, const nvfx_stage_desc& desc) {
    nvfx_stage* s = nullptr;
    nvfx_scene_error e{};
    if (const nvfx_status st = nvfx_stage_create(library.get(), &desc, &s, &e); st != NVFX_OK) return detail::fail(st, e);
    return Stage(s);
  }
  // The stage's scene: step, render, sample (do not restart it or seek it backwards: clear() does that).
  SceneView scene() const noexcept { return SceneView(nvfx_stage_scene(p_)); }
  // The stage's scene script into `out` (NUL-terminated, cut to fit); returns the length the whole script needs.
  std::size_t script(std::span<char> out) const noexcept { return nvfx_stage_script(p_, out.data(), out.size()); }
  std::string script() const {  // allocates
    std::string s(nvfx_stage_script(p_, nullptr, 0), '\0');
    nvfx_stage_script(p_, s.data(), s.size() + 1);
    return s;
  }
  Result<nvfx_fx> spawn(zstring effect, float x, float y, float scale = 1.f, double at = -1.0) {
    nvfx_fx fx = 0;
    return detail::value_or_fail(nvfx_stage_spawn(p_, effect.c_str(), x, y, scale, at, &fx), fx);
  }
  Result<> set_param(nvfx_fx fx, zstring param, float value) { return detail::check(nvfx_stage_set_param(p_, fx, param.c_str(), value)); }
  Result<float> get_param(nvfx_fx fx, zstring param) const {
    float v = 0.f;
    return detail::value_or_fail(nvfx_stage_get_param(p_, fx, param.c_str(), &v), v);
  }
  Result<> move(nvfx_fx fx, float x, float y) { return detail::check(nvfx_stage_move(p_, fx, x, y)); }
  Result<> stop(nvfx_fx fx, double at = -1.0) { return detail::check(nvfx_stage_stop(p_, fx, at)); }
  Result<nvfx_stage_fx_info> get_fx(nvfx_fx fx) const {
    nvfx_stage_fx_info i{};
    return detail::value_or_fail(nvfx_stage_get_fx(p_, fx, &i), i);
  }
  int free_rooms(zstring effect) const noexcept { return nvfx_stage_free_rooms(p_, effect.c_str()); }  // -1: no room for it
  Result<> set_camera(float x, float y) { return detail::check(nvfx_stage_set_camera(p_, x, y)); }
  Result<> clear() { return detail::check(nvfx_stage_clear(p_)); }  // allocates (rebuilds the scene)
};

// --- the effect manager (nvfx_manager.h) -------------------------------------------------------------------------------

inline nvfx_manager_desc manager_desc() noexcept {
  nvfx_manager_desc d;
  nvfx_manager_desc_init(&d);
  return d;
}
inline nvfx_manager_model_desc manager_model_desc() noexcept {
  nvfx_manager_model_desc d;
  nvfx_manager_model_desc_init(&d);
  return d;
}
inline nvfx_spawn_desc spawn_desc() noexcept {
  nvfx_spawn_desc d;
  nvfx_spawn_desc_init(&d);
  return d;
}

struct Cost {
  float step_ms = 0.f, render_ms = 0.f;
};

// A manager's picture: the C struct, and its pixels as a span (empty before the first picture).
struct Picture : nvfx_manager_picture {
  std::span<const std::uint8_t> pixels() const noexcept {
    if (!rgba || size <= 0) return {};
    return {rgba, stride * static_cast<std::size_t>(size - 1) + static_cast<std::size_t>(size) * 4};
  }
  bool is_new() const noexcept { return (flags & NVFX_PICTURE_NEW) != 0; }
  bool finished() const noexcept { return (flags & NVFX_PICTURE_FINISHED) != 0; }
  // drawn at half size (nvfx_manager_desc.half_px, docs/MANAGER.md §2.10): lod_size is the drawing's size
  bool half() const noexcept { return (flags & NVFX_PICTURE_HALF) != 0; }
};

class Manager : public detail::Handle<nvfx_manager, nvfx_manager_free> {
 public:
  using Handle::Handle;
  static Result<Manager> create(const nvfx_manager_desc& desc) {
    nvfx_manager* m = nullptr;
    if (const nvfx_status st = nvfx_manager_create(&desc, &m); st != NVFX_OK) return detail::fail(st);
    return Manager(m);
  }
  // A rollout effect, which must outlive the manager; nullptr desc: the model defaults. Returns the model's index.
  Result<int> add_model(const Effect& effect, const nvfx_manager_model_desc* desc = nullptr) {
    int model = -1;
    return detail::value_or_fail(nvfx_manager_add_model(p_, effect.get(), desc, &model), model);
  }
  Result<> set_cost(int model, int size, float step_ms, float render_ms) { return detail::check(nvfx_manager_set_cost(p_, model, size, step_ms, render_ms)); }
  Result<Cost> get_cost(int model, int size) const {
    Cost c;
    return detail::value_or_fail(nvfx_manager_get_cost(p_, model, size, &c.step_ms, &c.render_ms), c);
  }
  // Measures this machine (do it once per model at start, docs/MANAGER.md §7.1).
  Result<> calibrate(int model, int frames = 24) { return detail::check(nvfx_manager_calibrate(p_, model, frames)); }
  // The pack's default costs of a model for every size it has them (none: nothing changes). Returns how many were set.
  Result<int> set_costs_from(const Pack& pack, int model, zstring name);
  Result<> set_camera(std::span<const float, 3> position, std::span<const float, 3> forward, std::span<const float, 3> up, float vertical_fov_radians,
                      float viewport_width_px, float viewport_height_px) {
    return detail::check(nvfx_manager_set_camera(p_, position.data(), forward.data(), up.data(), vertical_fov_radians, viewport_width_px, viewport_height_px));
  }
  Result<> set_budget(float budget_ms) { return detail::check(nvfx_manager_set_budget(p_, budget_ms)); }
  Result<nvfx_handle> spawn(const nvfx_spawn_desc& desc) {
    nvfx_handle h = 0;
    return detail::value_or_fail(nvfx_manager_spawn(p_, &desc, &h), h);
  }
  Result<> stop(nvfx_handle h) { return detail::check(nvfx_manager_stop(p_, h)); }
  Result<> set_controls(nvfx_handle h, std::span<const float> controls) {
    return detail::check(nvfx_manager_set_controls(p_, h, controls.data(), static_cast<int>(controls.size())));
  }
  Result<> set_position(nvfx_handle h, std::span<const float, 3> position, float world_size) {
    return detail::check(nvfx_manager_set_position(p_, h, position.data(), world_size));
  }
  Result<> set_screen_size(nvfx_handle h, float pixels, bool visible = true) { return detail::check(nvfx_manager_set_screen_size(p_, h, pixels, visible ? 1 : 0)); }
  Result<> set_priority(nvfx_handle h, float priority) { return detail::check(nvfx_manager_set_priority(p_, h, priority)); }
  Result<> set_lod_lock(nvfx_handle h, int size, int rate_div) { return detail::check(nvfx_manager_set_lod_lock(p_, h, size, rate_div)); }
  // A tick on the manager's threads; or tick_begin (returns the jobs), run_jobs(slot) on the engine's threads, tick_end.
  Result<> tick(double dt_seconds) { return detail::check(nvfx_manager_tick(p_, dt_seconds)); }
  Result<int> tick_begin(double dt_seconds) {
    int jobs = 0;
    return detail::value_or_fail(nvfx_manager_tick_begin(p_, dt_seconds, &jobs), jobs);
  }
  Result<> run_jobs(int slot) { return detail::check(nvfx_manager_run_jobs(p_, slot)); }
  Result<> tick_end() { return detail::check(nvfx_manager_tick_end(p_)); }
  // Valid until the next tick.
  Result<Picture> get_picture(nvfx_handle h) const {
    Picture pic{};
    return detail::value_or_fail(nvfx_manager_get_picture(p_, h, &pic), pic);
  }
  Result<nvfx_manager_stats> get_stats() const {
    nvfx_manager_stats s{};
    return detail::value_or_fail(nvfx_manager_get_stats(p_, &s), s);
  }
};

// --- effect packs (nvfx_pack.h) ---------------------------------------------------------------------------------------

inline nvfx_pack_desc pack_desc() noexcept {
  nvfx_pack_desc d;
  nvfx_pack_desc_init(&d);
  return d;
}

inline std::span<const std::uint8_t> bytes(const nvfx_pack_member& m) noexcept { return {m.data, m.bytes}; }

class Pack : public detail::Handle<nvfx_pack, nvfx_pack_free> {
 public:
  using Handle::Handle;
  static Result<Pack> open(const nvfx_pack_desc& desc) {
    nvfx_pack* p = nullptr;
    nvfx_scene_error e{};
    if (const nvfx_status st = nvfx_pack_open(&desc, &p, &e); st != NVFX_OK) return detail::fail(st, e);
    return Pack(p);
  }
  static Result<Pack> open(zstring path, int threads = 1) {
    nvfx_pack_desc d = pack_desc();
    d.path = path.c_str();
    d.threads = threads;
    return open(d);
  }
  // A pack in memory, read during the call only.
  static Result<Pack> open_memory(std::span<const std::uint8_t> data, int threads = 1, const char* name = nullptr) {
    nvfx_pack_desc d = pack_desc();
    d.data = data.data();
    d.bytes = data.size();
    d.threads = threads;
    d.name = name;
    return open(d);
  }
  Result<nvfx_pack_info> get_info() const {
    nvfx_pack_info i{};
    return detail::value_or_fail(nvfx_pack_get_info(p_, &i), i);
  }
  int find(zstring name) const noexcept { return nvfx_pack_find(p_, name.c_str()); }  // -1: none
  Result<nvfx_pack_member> get_member(int index) const {
    nvfx_pack_member m{};
    return detail::value_or_fail(nvfx_pack_get_member(p_, index, &m), m);
  }
  const char* entry_name(int i) const noexcept { return nvfx_pack_entry_name(p_, i); }
  // fn(const char* name, const char* file) once per model entry i needs.
  template <class F>
  Result<> entry_models(int i, F&& fn) const {
    auto f = std::forward<F>(fn);
    return detail::check(nvfx_pack_entry_models(p_, i, &detail::effect_trampoline<decltype(f)>, &f));
  }
  // A model as an Effect of its own (it does not need the pack afterwards).
  Result<Effect> load_effect(zstring name) const {
    nvfx_effect* e = nullptr;
    if (const nvfx_status st = nvfx_pack_load_effect(p_, name.c_str(), &e); st != NVFX_OK) return detail::fail(st);
    return Effect(e);
  }
  Result<Cost> model_cost(zstring model, int size) const {
    Cost c;
    return detail::value_or_fail(nvfx_pack_model_cost(p_, model.c_str(), size, &c.step_ms, &c.render_ms), c);
  }
};

inline Result<Scene> Scene::create_from_pack(const Pack& pack, const nvfx_scene_desc& desc) {
  nvfx_scene* s = nullptr;
  nvfx_scene_error e{};
  if (const nvfx_status st = nvfx_scene_create_from_pack(pack.get(), &desc, &s, &e); st != NVFX_OK) return detail::fail(st, e);
  return Scene(s);
}

inline Result<Library> Library::open_pack(const Pack& pack) {
  nvfx_library* l = nullptr;
  nvfx_scene_error e{};
  if (const nvfx_status st = nvfx_library_open_pack(pack.get(), &l, &e); st != NVFX_OK) return detail::fail(st, e);
  return Library(l);
}

inline Result<int> Manager::set_costs_from(const Pack& pack, int model, zstring name) {
  int n = 0;
  for (const int size : {32, 64, 128, 256}) {
    const auto c = pack.model_cost(name, size);
    if (!c) continue;
    if (auto r = set_cost(model, size, c->step_ms, c->render_ms); !r) return std::unexpected(std::move(r.error()));
    ++n;
  }
  return n;
}

}  // namespace nvfx
