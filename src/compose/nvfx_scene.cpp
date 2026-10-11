// The C API for scenes (include/neuralfx/nvfx_scene.h, docs/ENGINES.md §7): a scene script played by
// compose::script::Scene, with a clock in seconds, its picture as RGBA, its field bus for gameplay and the game's
// run-time settings. Exceptions never cross the C boundary: every entry point returns a status.
//
// The scene keeps the parsed script and its effects (read once), so that a restart rebuilds it without reading files.
// Every buffer a frame needs is made at creation; stepping, drawing, the field reads and the settings allocate nothing.
#include <neuralfx/nvfx_scene.h>

#include "nvfx_internal.hpp"
#include "script.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace sc = nfx::compose::script;
using nfx::rt::RolloutEffect;

namespace {

std::size_t zs(int v) { return static_cast<std::size_t>(v); }

// One axis of the bilinear resampling from the scene's size to the output's: per output pixel, the two source pixels
// and the weight of the second (pixel centres aligned, edges clamped).
struct Axis {
  std::vector<int> i0, i1;
  std::vector<float> f;
  void build(int src, int dst) {
    i0.resize(zs(dst));
    i1.resize(zs(dst));
    f.resize(zs(dst));
    const float k = static_cast<float>(src) / static_cast<float>(dst);
    for (int d = 0; d < dst; ++d) {
      const float s = std::clamp((static_cast<float>(d) + 0.5f) * k - 0.5f, 0.f, static_cast<float>(src - 1));
      const int a = std::min(static_cast<int>(s), src - 1);
      i0[zs(d)] = a;
      i1[zs(d)] = std::min(a + 1, src - 1);
      f[zs(d)] = s - static_cast<float>(a);
    }
  }
};

void set_error(nvfx_scene_error* err, int line, int col, std::string_view message) {
  if (!err) return;
  err->line = line;
  err->column = col;
  const std::size_t n = std::min(message.size(), sizeof(err->message) - 1);
  std::memcpy(err->message, message.data(), n);
  err->message[n] = '\0';
}

void clear_error(nvfx_scene_error* err) { set_error(err, 0, 0, ""); }

}  // namespace

struct nvfx_scene {
  sc::Script script;
  std::map<std::string, RolloutEffect, std::less<>> effects;  // by the file as the script writes it
  sc::Options opt;
  std::unique_ptr<sc::Scene> scene;
  int sw = 0, sh = 0, ow = 0, oh = 0;
  float fps = 30.f;
  int frame = 0;   // the clock's frame
  int drawn = -1;  // the frame in rgb
  double time = 0.0;
  std::vector<std::uint8_t> rgb;  // the scene's picture, sw x sh x 3
  Axis ax, ay;                    // resampling, when the output's size differs
  std::vector<std::string> modules, rules, inputs;
  std::vector<std::vector<std::string>> controls;  // per module: its effect's control names

  void build() {
    scene.reset();
    scene = std::make_unique<sc::Scene>(script, [this](const std::string& file) { return effects.at(file); }, opt);
    scene->advance(0);
    frame = 0;
    drawn = -1;
    time = 0.0;
  }
  int frame_at(double seconds) const {
    const double f = std::floor(seconds * static_cast<double>(fps) + 1e-6);
    return static_cast<int>(std::clamp(f, 0.0, static_cast<double>(std::numeric_limits<int>::max() / 2)));
  }
  void go(int target) {  // forwards
    if (target <= frame) return;
    scene->advance(target);
    frame = target;
  }
  void restart() {
    opt.inputs.clear();
    for (int i = 0; i < scene->inputs(); ++i) opt.inputs.emplace_back(scene->input_name(i), scene->input(i));
    build();
  }
  int module_index(const char* name) const {
    if (!name) return -1;
    for (std::size_t i = 0; i < modules.size(); ++i)
      if (modules[i] == name) return static_cast<int>(i);
    return -1;
  }
};

namespace {

// Runs f, turning exceptions into statuses (and messages, when `err` is given).
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
  } catch (const std::exception& e) {
    set_error(err, 0, 0, e.what());
    return NVFX_ERROR_ARGUMENT;
  } catch (...) {
    set_error(err, 0, 0, "unknown error");
    return NVFX_ERROR_ARGUMENT;
  }
}

std::string read_file(const std::filesystem::path& p, bool& ok) {
  std::ifstream f(p, std::ios::binary);
  ok = static_cast<bool>(f);
  if (!ok) return {};
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

float field_of(const nfx::compose::FieldBus::Sample& s, nvfx_scene_field f, float fps) {
  switch (f) {
    case NVFX_FIELD_HEAT: return s.heat;
    case NVFX_FIELD_SOOT: return s.soot;
    case NVFX_FIELD_U: return s.u * fps;
    case NVFX_FIELD_V: return s.v * fps;
  }
  return 0.f;
}

bool valid_field(nvfx_scene_field f) { return f == NVFX_FIELD_HEAT || f == NVFX_FIELD_SOOT || f == NVFX_FIELD_U || f == NVFX_FIELD_V; }

}  // namespace

extern "C" {

void nvfx_scene_desc_init(nvfx_scene_desc* d) {
  if (!d) return;
  *d = {};
  d->threads = 2;
  d->overlap = -1;
}

nvfx_status nvfx_scene_check(const char* script, const char* source_name, nvfx_scene_error* err) {
  clear_error(err);
  if (!script) return NVFX_ERROR_ARGUMENT;
  return guarded(err, [&] {
    sc::validate(sc::parse(script, source_name ? source_name : "script"));
    return NVFX_OK;
  });
}

nvfx_status nvfx_scene_list_effects(const char* script, const char* source_name, nvfx_scene_effect_fn fn, void* user, nvfx_scene_error* err) {
  clear_error(err);
  if (!script || !fn) return NVFX_ERROR_ARGUMENT;
  return guarded(err, [&] {
    const sc::Script s = sc::parse(script, source_name ? source_name : "script");
    for (const sc::Statement& st : s.statements)
      if (st.keyword == "effect") fn(user, st.name.c_str(), st.kind.c_str());
    return NVFX_OK;
  });
}

nvfx_status nvfx_scene_create(const nvfx_scene_desc* d, nvfx_scene** out, nvfx_scene_error* err) {
  clear_error(err);
  if (!d || !out) return NVFX_ERROR_ARGUMENT;
  *out = nullptr;
  if ((!d->script && !d->script_path) || d->threads < 0 || d->threads > 256 || d->width < 0 || d->height < 0 || d->n_effects < 0 || d->n_inputs < 0 ||
      (d->n_effects > 0 && !d->effects) || (d->n_inputs > 0 && !d->inputs)) {
    set_error(err, 0, 0, "invalid scene description");
    return NVFX_ERROR_ARGUMENT;
  }
  return guarded(err, [&]() -> nvfx_status {
    auto s = std::make_unique<nvfx_scene>();
    // the script: parsed and checked before any effect is loaded, so that its errors come first
    std::string text;
    std::string source = d->source_name ? d->source_name : "script";
    std::filesystem::path dir = d->effects_dir ? std::filesystem::path(d->effects_dir) : std::filesystem::path();
    if (d->script) {
      text = d->script;
    } else {
      const std::filesystem::path p(d->script_path);
      bool ok = false;
      text = read_file(p, ok);
      if (!ok) {
        set_error(err, 0, 0, std::format("cannot read {}", p.string()));
        return NVFX_ERROR_IO;
      }
      if (!d->source_name) source = p.filename().string();
      if (!d->effects_dir) dir = p.parent_path();
    }
    s->script = sc::parse(text, source);
    sc::validate(s->script);
    // the effects: passed in, or read from their files
    for (const sc::Statement& st : s->script.statements) {
      if (st.keyword != "effect" || s->effects.contains(st.kind)) continue;
      const std::string& file = st.kind;
      const std::string stem = std::filesystem::path(file).stem().string();
      const nvfx_effect* given = nullptr;
      for (int i = 0; i < d->n_effects; ++i) {
        const char* n = d->effects[i].name;
        if (n && (file == n || stem == n)) given = d->effects[i].effect;
      }
      const auto fail = [&](nvfx_status status, std::string_view why) {
        set_error(err, st.pos.line, st.pos.col, std::format("{}:{}:{}: cannot load effect '{}' ({}): {}", source, st.pos.line, st.pos.col, st.name, file, why));
        return status;
      };
      nvfx_effect* loaded = nullptr;
      if (!given) {
        const std::filesystem::path p = std::filesystem::path(file).is_absolute() ? std::filesystem::path(file) : dir / file;
        const nvfx_status ls = nvfx_effect_load(p.string().c_str(), &loaded);
        if (ls != NVFX_OK) return fail(ls, std::format("{}: {}", p.string(), nvfx_status_string(ls)));
        given = loaded;
      }
      const RolloutEffect* r = nfx::rt::rollout_of(given);
      if (r) s->effects.emplace(file, *r);
      if (loaded) nvfx_effect_free(loaded);
      if (!r) return fail(NVFX_ERROR_FORMAT, "not a rollout effect");
    }
    // the scene
    s->opt.threads = d->threads > 0 ? d->threads : 2;
    s->opt.isa = nfx::compose::best_isa();
    s->opt.overlap = s->opt.threads > 1 && d->overlap != 0;  // default (-1) and 1: on, given a second thread
    for (int i = 0; i < d->n_inputs; ++i) {
      if (!d->inputs[i].name) throw std::invalid_argument("an input without a name");
      s->opt.inputs.emplace_back(d->inputs[i].name, d->inputs[i].value);
    }
    s->build();
    s->opt.inputs.clear();
    s->sw = s->scene->width();
    s->sh = s->scene->height();
    s->ow = d->width > 0 ? d->width : s->sw;
    s->oh = d->height > 0 ? d->height : s->sh;
    if (s->ow > 16384 || s->oh > 16384) throw std::invalid_argument("the output is larger than 16384 pixels");
    s->fps = s->scene->fps();
    s->rgb.assign(zs(s->sw) * zs(s->sh) * 3, 0);
    if (s->ow != s->sw || s->oh != s->sh) {
      s->ax.build(s->sw, s->ow);
      s->ay.build(s->sh, s->oh);
    }
    for (int i = 0; i < s->scene->script_modules(); ++i) {
      const sc::Scene::ModuleInfo m = s->scene->module_info(i);
      s->modules.emplace_back(m.name);
      s->controls.emplace_back(m.control_names.begin(), m.control_names.end());
    }
    for (int i = 0; i < s->scene->rules(); ++i) s->rules.push_back(s->scene->rule_name(i));
    for (int i = 0; i < s->scene->inputs(); ++i) s->inputs.push_back(s->scene->input_name(i));
    *out = s.release();
    return NVFX_OK;
  });
}

void nvfx_scene_free(nvfx_scene* s) { delete s; }

nvfx_status nvfx_scene_get_info(const nvfx_scene* s, nvfx_scene_info* info) {
  if (!s || !info || !s->scene) return NVFX_ERROR_ARGUMENT;
  *info = {};
  const sc::Scene& S = *s->scene;
  info->width = s->ow;
  info->height = s->oh;
  info->scene_width = s->sw;
  info->scene_height = s->sh;
  info->fps = s->fps;
  info->frames = S.frames();
  info->length = S.length();
  info->ground = S.frame().ground_y;
  const auto& b = S.bus();
  info->bus_x = b.x0();
  info->bus_y = b.y0();
  info->bus_cell = b.cell();
  info->bus_nx = b.nx();
  info->bus_ny = b.ny();
  info->n_modules = static_cast<int>(s->modules.size());
  info->n_rules = static_cast<int>(s->rules.size());
  info->n_inputs = static_cast<int>(s->inputs.size());
  info->threads = s->opt.threads;
  info->overlap = s->opt.overlap ? 1 : 0;
  info->scratch_bytes = S.scratch_bytes();
  return NVFX_OK;
}

nvfx_status nvfx_scene_step(nvfx_scene* s, double dt, int* frames) {
  if (frames) *frames = 0;
  if (!s || !s->scene || !(dt >= 0.0) || !std::isfinite(dt)) return NVFX_ERROR_ARGUMENT;
  return guarded(nullptr, [&] {
    const int before = s->frame;
    s->time += dt;
    s->go(s->frame_at(s->time));
    if (frames) *frames = s->frame - before;
    return NVFX_OK;
  });
}

nvfx_status nvfx_scene_step_frames(nvfx_scene* s, int n) {
  if (!s || !s->scene || n < 0 || n > std::numeric_limits<int>::max() / 4 - s->frame) return NVFX_ERROR_ARGUMENT;
  return guarded(nullptr, [&] {
    s->go(s->frame + n);
    s->time = static_cast<double>(s->frame) / static_cast<double>(s->fps);
    return NVFX_OK;
  });
}

nvfx_status nvfx_scene_seek(nvfx_scene* s, double seconds) {
  if (!s || !s->scene || !(seconds >= 0.0) || !std::isfinite(seconds)) return NVFX_ERROR_ARGUMENT;
  return guarded(nullptr, [&] {
    const int target = s->frame_at(seconds);
    if (target < s->frame) s->restart();
    s->go(target);
    s->time = seconds;
    return NVFX_OK;
  });
}

nvfx_status nvfx_scene_restart(nvfx_scene* s) {
  if (!s || !s->scene) return NVFX_ERROR_ARGUMENT;
  return guarded(nullptr, [&] {
    s->restart();
    return NVFX_OK;
  });
}

double nvfx_scene_time(const nvfx_scene* s) { return s ? s->time : 0.0; }
int nvfx_scene_frame(const nvfx_scene* s) { return s ? s->frame : 0; }

nvfx_status nvfx_scene_render(nvfx_scene* s, uint8_t* rgba, size_t stride) {
  if (!s || !s->scene || !rgba || stride < zs(s->ow) * 4) return NVFX_ERROR_ARGUMENT;
  return guarded(nullptr, [&] {
    if (s->drawn != s->frame) {
      s->scene->render(s->frame, s->rgb);
      s->drawn = s->frame;
    }
    const std::uint8_t* src = s->rgb.data();
    const std::size_t sstride = zs(s->sw) * 3;
    if (s->ow == s->sw && s->oh == s->sh) {
      for (int y = 0; y < s->oh; ++y) {
        const std::uint8_t* a = src + zs(y) * sstride;
        std::uint8_t* o = rgba + zs(y) * stride;
        for (int x = 0; x < s->ow; ++x, a += 3, o += 4) {
          o[0] = a[0];
          o[1] = a[1];
          o[2] = a[2];
          o[3] = 255;
        }
      }
      return NVFX_OK;
    }
    for (int y = 0; y < s->oh; ++y) {
      const std::uint8_t* r0 = src + zs(s->ay.i0[zs(y)]) * sstride;
      const std::uint8_t* r1 = src + zs(s->ay.i1[zs(y)]) * sstride;
      const float fy = s->ay.f[zs(y)];
      std::uint8_t* o = rgba + zs(y) * stride;
      for (int x = 0; x < s->ow; ++x, o += 4) {
        const std::size_t c0 = zs(s->ax.i0[zs(x)]) * 3, c1 = zs(s->ax.i1[zs(x)]) * 3;
        const float fx = s->ax.f[zs(x)];
        for (std::size_t c = 0; c < 3; ++c) {
          const float top = static_cast<float>(r0[c0 + c]) + fx * (static_cast<float>(r0[c1 + c]) - static_cast<float>(r0[c0 + c]));
          const float bot = static_cast<float>(r1[c0 + c]) + fx * (static_cast<float>(r1[c1 + c]) - static_cast<float>(r1[c0 + c]));
          o[c] = static_cast<std::uint8_t>(std::clamp(top + fy * (bot - top) + 0.5f, 0.f, 255.f));
        }
        o[3] = 255;
      }
    }
    return NVFX_OK;
  });
}

nvfx_status nvfx_scene_sample(const nvfx_scene* s, float x, float y, nvfx_scene_fields* out) {
  if (!s || !s->scene || !out) return NVFX_ERROR_ARGUMENT;
  const auto v = s->scene->bus().at(x, y);
  out->u = v.u * s->fps;
  out->v = v.v * s->fps;
  out->heat = v.heat;
  out->soot = v.soot;
  return NVFX_OK;
}

nvfx_status nvfx_scene_sample_grid(const nvfx_scene* s, nvfx_scene_field field, float x0, float y0, float dx, float dy, int nx, int ny, float* out, size_t row_stride) {
  if (!s || !s->scene || !out || nx < 0 || ny < 0 || row_stride < zs(nx) || !valid_field(field)) return NVFX_ERROR_ARGUMENT;
  const auto& bus = s->scene->bus();
  for (int j = 0; j < ny; ++j) {
    const float y = y0 + static_cast<float>(j) * dy;
    for (int i = 0; i < nx; ++i) out[zs(j) * row_stride + zs(i)] = field_of(bus.at(x0 + static_cast<float>(i) * dx, y), field, s->fps);
  }
  return NVFX_OK;
}

nvfx_status nvfx_scene_field_region(const nvfx_scene* s, nvfx_scene_field field, float x0, float y0, float x1, float y1, float* max, float* mean) {
  if (!s || !s->scene || !valid_field(field)) return NVFX_ERROR_ARGUMENT;
  const auto& bus = s->scene->bus();
  const float c = bus.cell();
  // the cells whose centres bus.x0() + (i + 0.5) c fall in [x0, x1)
  const auto first = [&](float lo, float origin, int n) { return std::clamp(static_cast<int>(std::ceil((lo - origin) / c - 0.5f)), 0, n); };
  const int i0 = first(x0, bus.x0(), bus.nx()), i1 = first(x1, bus.x0(), bus.nx());
  const int j0 = first(y0, bus.y0(), bus.ny()), j1 = first(y1, bus.y0(), bus.ny());
  float best = 0.f;
  double sum = 0.0;
  int n = 0;
  for (int j = j0; j < j1; ++j) {
    for (int i = i0; i < i1; ++i) {
      const float v = field_of(bus.at(bus.x0() + (static_cast<float>(i) + 0.5f) * c, bus.y0() + (static_cast<float>(j) + 0.5f) * c), field, s->fps);
      best = n == 0 ? v : std::max(best, v);
      sum += static_cast<double>(v);
      ++n;
    }
  }
  if (max) *max = best;
  if (mean) *mean = n > 0 ? static_cast<float>(sum / n) : 0.f;
  return NVFX_OK;
}

nvfx_status nvfx_scene_set_input(nvfx_scene* s, const char* name, float value) {
  if (!s || !s->scene || !name) return NVFX_ERROR_ARGUMENT;
  const int i = s->scene->input_index(name);
  if (i < 0) return NVFX_ERROR_ARGUMENT;
  s->scene->set_input(i, value);
  return NVFX_OK;
}

nvfx_status nvfx_scene_get_input(const nvfx_scene* s, const char* name, float* value) {
  if (!s || !s->scene || !name || !value) return NVFX_ERROR_ARGUMENT;
  const int i = s->scene->input_index(name);
  if (i < 0) return NVFX_ERROR_ARGUMENT;
  *value = s->scene->input(i);
  return NVFX_OK;
}

const char* nvfx_scene_input_name(const nvfx_scene* s, int i) { return s && i >= 0 && zs(i) < s->inputs.size() ? s->inputs[zs(i)].c_str() : nullptr; }

nvfx_status nvfx_scene_trigger(nvfx_scene* s, const char* rule) {
  if (!s || !s->scene || !rule) return NVFX_ERROR_ARGUMENT;
  return s->scene->trigger(rule) ? NVFX_OK : NVFX_ERROR_ARGUMENT;
}

nvfx_status nvfx_scene_rule_state(const nvfx_scene* s, const char* rule, int* count, float* last_time) {
  if (!s || !s->scene || !rule) return NVFX_ERROR_ARGUMENT;
  const int n = s->scene->rule_count(rule);
  if (n < 0) return NVFX_ERROR_ARGUMENT;
  if (count) *count = n;
  if (last_time) *last_time = s->scene->rule_time(rule);
  return NVFX_OK;
}

const char* nvfx_scene_rule_name(const nvfx_scene* s, int i) { return s && i >= 0 && zs(i) < s->rules.size() ? s->rules[zs(i)].c_str() : nullptr; }

const char* nvfx_scene_module_name(const nvfx_scene* s, int i) { return s && i >= 0 && zs(i) < s->modules.size() ? s->modules[zs(i)].c_str() : nullptr; }

nvfx_status nvfx_scene_module_get_info(const nvfx_scene* s, const char* module, nvfx_scene_module_info* info) {
  if (!s || !s->scene || !info) return NVFX_ERROR_ARGUMENT;
  const int mi = s->module_index(module);
  if (mi < 0) return NVFX_ERROR_ARGUMENT;
  const sc::Scene::ModuleInfo m = s->scene->module_info(mi);
  *info = {};
  info->x = m.x;
  info->y = m.y;
  info->width = m.width;
  info->started = m.started;
  info->active = m.active ? 1 : 0;
  info->tiles = m.tiles;
  info->n_controls = static_cast<int>(m.controls.size());
  for (std::size_t c = 0; c < std::min(m.controls.size(), std::size(info->controls)); ++c) info->controls[c] = m.controls[c];
  return NVFX_OK;
}

const char* nvfx_scene_module_control_name(const nvfx_scene* s, const char* module, int i) {
  if (!s || !s->scene || i < 0) return nullptr;
  const int mi = s->module_index(module);
  if (mi < 0) return nullptr;
  const auto& names = s->controls[zs(mi)];
  return zs(i) < names.size() ? names[zs(i)].c_str() : nullptr;
}

nvfx_status nvfx_scene_module_place(nvfx_scene* s, const char* module, float x, float y) {
  if (!s || !s->scene || !module || !std::isfinite(x) || !std::isfinite(y)) return NVFX_ERROR_ARGUMENT;
  return s->scene->place(module, x, y) ? NVFX_OK : NVFX_ERROR_ARGUMENT;
}

nvfx_status nvfx_scene_module_set_control(nvfx_scene* s, const char* module, const char* control, float value) {
  if (!s || !s->scene || !module || !control || !std::isfinite(value)) return NVFX_ERROR_ARGUMENT;
  return s->scene->set_control(module, control, value) ? NVFX_OK : NVFX_ERROR_ARGUMENT;
}

}  // extern "C"
