// The C API (include/neuralfx/nvfx.h): effects, instances, time and variation logic, ISA dispatch, baking.
// Exceptions never cross the C boundary: every entry point returns a status.
#include <neuralfx/noise.hpp>
#include <neuralfx/nvfx.h>

#include "rt_common.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <new>
#include <numbers>
#include <spanstream>
#include <vector>

using nfx::rt::Effect;
using nfx::rt::FrameInput;
using nfx::rt::Renderer;

struct nvfx_effect {
  Effect e;
};

struct nvfx_instance {
  const nvfx_effect* effect = nullptr;
  int size = 0;
  std::unique_ptr<Renderer> renderer;
  std::vector<float> controls, cond, za, zb;
  std::uint64_t seed = 0;
  int variation = -1;
  float drift_seconds = 0.f;
  float hue = 0.f, brightness = 1.f;
  std::array<float, 9> colour{};
  bool apply_colour = false;
};

namespace {

std::atomic<int> g_forced_isa{NVFX_ISA_AUTO};

bool cpu_has(nvfx_isa isa) {
#if defined(__GNUC__) && defined(__x86_64__)
  __builtin_cpu_init();
  if (isa == NVFX_ISA_AVX512) return __builtin_cpu_supports("x86-64-v4");
  if (isa == NVFX_ISA_AVX2) return __builtin_cpu_supports("x86-64-v3");
#endif
  return isa == NVFX_ISA_BASELINE;
}

nvfx_isa resolved_isa() {
  const auto forced = static_cast<nvfx_isa>(g_forced_isa.load());
  if (forced != NVFX_ISA_AUTO) return forced;
  // AVX2 by default: AVX-512 measured as a tie on the benchmark machine (docs/PLAN.md §8) and lowers clocks on some CPUs.
  if (cpu_has(NVFX_ISA_AVX2)) return NVFX_ISA_AVX2;
  return NVFX_ISA_BASELINE;
}

std::size_t resident_bytes(const nfx::Model& m) {
  std::size_t n = m.raw_f16.size() * 2 + m.raw_u8.size() + m.raw_ranges.size() * 4;
  n += (m.basis.w.size() + m.basis.b.size()) * 4;
  for (const auto* group : {&m.layers, &m.films}) {
    for (const auto& d : *group) n += (d.w.size() + d.b.size()) * 4;
  }
  n += (m.z_mean.size() + m.z_std.size()) * 4;
  for (const auto& z : m.z_train) n += z.size() * 4;
  return n;
}

nvfx_status adopt(std::expected<nfx::Model, std::string>&& m, nvfx_effect** out) {
  if (!m) return m.error().find("cannot open") != std::string::npos ? NVFX_ERROR_IO : NVFX_ERROR_FORMAT;
  auto* e = new (std::nothrow) nvfx_effect;
  if (!e) return NVFX_ERROR_MEMORY;
  e->e.m = std::move(*m);
  if (e->e.m.raw_f16.empty() && e->e.m.raw_u8.empty()) e->e.m.pack_features();
  e->e.stored_bytes = e->e.m.storage_bytes();
  e->e.m.features.clear();  // the runtime reads the stored format only
  e->e.m.features.shrink_to_fit();
  e->e.resident_bytes = resident_bytes(e->e.m);
  *out = e;
  return NVFX_OK;
}

// A variation code for (seed, step): a random point between two training codes (stays among realistic variations),
// or a draw from the codes' Gaussian when no codes were kept.
void seeded_code(const nfx::Model& m, std::uint64_t seed, std::int64_t step, std::span<float> z) {
  const auto h = [&](std::uint32_t salt) {
    return nfx::hash_cell(static_cast<std::int32_t>(step), static_cast<std::int32_t>(step >> 31), static_cast<std::int32_t>(salt), seed);
  };
  if (!m.z_train.empty()) {
    const auto& a = m.z_train[h(1) % m.z_train.size()];
    const auto& b = m.z_train[h(2) % m.z_train.size()];
    const float u = static_cast<float>(h(3) >> 8) / 16777215.f;
    for (std::size_t j = 0; j < z.size(); ++j) z[j] = a[j] + u * (b[j] - a[j]);
    return;
  }
  for (std::size_t j = 0; j < z.size(); ++j) {  // Box-Muller from two hashes
    const float u1 = (static_cast<float>(h(10 + 2 * static_cast<std::uint32_t>(j)) >> 8) + 1.f) / 16777217.f;
    const float u2 = static_cast<float>(h(11 + 2 * static_cast<std::uint32_t>(j)) >> 8) / 16777215.f;
    const float g = std::sqrt(-2.f * std::log(u1)) * std::cos(2.f * std::numbers::pi_v<float> * u2);
    z[j] = m.z_mean[j] + m.z_std[j] * g;
  }
}

// Model time and condition vector for a moment in seconds. No allocation.
float prepare(nvfx_instance& in, double seconds, bool allow_drift) {
  const nfx::Model& m = in.effect->e.m;
  const nfx::Hyper& h = m.h;
  const double frames = seconds * static_cast<double>(m.fps);
  float t;
  if (h.loop) {
    const double u = frames / h.frames;
    t = static_cast<float>(u - std::floor(u));
  } else {
    t = static_cast<float>(std::clamp(frames / std::max(1, h.frames - 1), 0.0, 1.0));
  }
  std::copy(in.controls.begin(), in.controls.end(), in.cond.begin());
  const std::span z(in.cond.data() + h.n_controls, static_cast<std::size_t>(h.n_latent));
  if (h.n_latent > 0) {
    if (in.variation >= 0) {
      std::ranges::copy(m.z_train[static_cast<std::size_t>(in.variation)], z.begin());
    } else if (allow_drift && h.loop && in.drift_seconds > 0.f) {
      // Drift: a smooth path between seeded codes, one waypoint every drift_seconds, so the loop never repeats.
      const double u = seconds / in.drift_seconds;
      const auto step = static_cast<std::int64_t>(std::floor(u));
      const float f = static_cast<float>(u - std::floor(u)), s = f * f * (3.f - 2.f * f);
      seeded_code(m, in.seed, step, in.za);
      seeded_code(m, in.seed, step + 1, in.zb);
      for (std::size_t j = 0; j < z.size(); ++j) z[j] = in.za[j] + s * (in.zb[j] - in.za[j]);
    } else {
      seeded_code(m, in.seed, 0, z);
    }
  }
  return t;
}

}  // namespace

extern "C" {

const char* nvfx_status_string(nvfx_status s) {
  switch (s) {
    case NVFX_OK: return "ok";
    case NVFX_ERROR_ARGUMENT: return "invalid argument";
    case NVFX_ERROR_IO: return "file could not be read";
    case NVFX_ERROR_FORMAT: return "not a valid .nvfx file";
    case NVFX_ERROR_MEMORY: return "out of memory";
    case NVFX_ERROR_UNSUPPORTED: return "unsupported size or ISA";
  }
  return "unknown status";
}

nvfx_status nvfx_effect_load(const char* path, nvfx_effect** out) {
  if (!path || !out) return NVFX_ERROR_ARGUMENT;
  try {
    return adopt(nfx::load_model(std::filesystem::path(path)), out);
  } catch (const std::bad_alloc&) {
    return NVFX_ERROR_MEMORY;
  } catch (...) {
    return NVFX_ERROR_FORMAT;
  }
}

nvfx_status nvfx_effect_load_memory(const void* data, size_t bytes, nvfx_effect** out) {
  if (!data || !out) return NVFX_ERROR_ARGUMENT;
  try {
    std::ispanstream in(std::span(static_cast<const char*>(data), bytes));
    return adopt(nfx::load_model(in), out);
  } catch (const std::bad_alloc&) {
    return NVFX_ERROR_MEMORY;
  } catch (...) {
    return NVFX_ERROR_FORMAT;
  }
}

void nvfx_effect_free(nvfx_effect* e) { delete e; }

nvfx_status nvfx_effect_get_info(const nvfx_effect* e, nvfx_effect_info* info) {
  if (!e || !info) return NVFX_ERROR_ARGUMENT;
  const nfx::Model& m = e->e.m;
  *info = {};
  info->arch = static_cast<int>(m.h.arch);
  info->native_size = m.h.size;
  info->frames = m.h.frames;
  info->fps = m.fps;
  info->loops = m.h.loop ? 1 : 0;
  info->n_controls = m.h.n_controls;
  info->n_variations = m.h.n_latent > 0 ? static_cast<int>(m.z_train.size()) : 0;
  info->stored_bytes = e->e.stored_bytes;
  info->resident_bytes = e->e.resident_bytes;
  std::strncpy(info->name, m.effect.c_str(), sizeof(info->name) - 1);
  return NVFX_OK;
}

const char* nvfx_effect_control_name(const nvfx_effect* e, int i) {
  if (!e || i < 0 || static_cast<std::size_t>(i) >= e->e.m.control_names.size()) return nullptr;
  return e->e.m.control_names[static_cast<std::size_t>(i)].c_str();
}

nvfx_status nvfx_instance_create(const nvfx_effect* e, int size, nvfx_instance** out) {
  if (!e || !out) return NVFX_ERROR_ARGUMENT;
  const nfx::Hyper& h = e->e.m.h;
  if (h.arch == nfx::Arch::grid) {
    if (size < 16 || size > 1024 || size % 16) return NVFX_ERROR_UNSUPPORTED;
  } else if (size != h.size && size * 2 != h.size && size * 4 != h.size) {
    return NVFX_ERROR_UNSUPPORTED;
  }
  try {
    auto in = std::make_unique<nvfx_instance>();
    in->effect = e;
    in->size = size;
    in->controls.assign(static_cast<std::size_t>(h.n_controls), 0.5f);
    in->cond.assign(static_cast<std::size_t>(h.dims()), 0.f);
    in->za.assign(static_cast<std::size_t>(h.n_latent), 0.f);
    in->zb.assign(static_cast<std::size_t>(h.n_latent), 0.f);
    in->drift_seconds = h.loop ? 4.f * static_cast<float>(h.frames) / std::max(1.f, e->e.m.fps) : 0.f;
    switch (resolved_isa()) {
      case NVFX_ISA_AVX512: in->renderer = nfx::rt::isa_avx512::make_renderer(e->e, size); break;
      case NVFX_ISA_AVX2: in->renderer = nfx::rt::isa_avx2::make_renderer(e->e, size); break;
      default: in->renderer = nfx::rt::isa_base::make_renderer(e->e, size); break;
    }
    *out = in.release();
    return NVFX_OK;
  } catch (const std::bad_alloc&) {
    return NVFX_ERROR_MEMORY;
  } catch (...) {
    return NVFX_ERROR_ARGUMENT;
  }
}

void nvfx_instance_free(nvfx_instance* in) { delete in; }

size_t nvfx_instance_scratch_bytes(const nvfx_instance* in) {
  if (!in) return 0;
  return in->renderer->scratch_bytes() + 4 * (in->controls.size() + in->cond.size() + in->za.size() + in->zb.size());
}

nvfx_status nvfx_instance_set_controls(nvfx_instance* in, const float* controls, int count) {
  if (!in || (count > 0 && !controls) || count < 0) return NVFX_ERROR_ARGUMENT;
  for (int i = 0; i < count && static_cast<std::size_t>(i) < in->controls.size(); ++i) {
    in->controls[static_cast<std::size_t>(i)] = controls[i];
  }
  return NVFX_OK;
}

nvfx_status nvfx_instance_set_seed(nvfx_instance* in, uint64_t seed) {
  if (!in) return NVFX_ERROR_ARGUMENT;
  in->seed = seed;
  return NVFX_OK;
}

nvfx_status nvfx_instance_set_variation(nvfx_instance* in, int index) {
  if (!in) return NVFX_ERROR_ARGUMENT;
  const nfx::Model& m = in->effect->e.m;
  const int n = m.h.n_latent > 0 ? static_cast<int>(m.z_train.size()) : 0;
  if (index < -1 || index >= n) return NVFX_ERROR_ARGUMENT;
  in->variation = index;
  return NVFX_OK;
}

nvfx_status nvfx_instance_set_drift(nvfx_instance* in, float seconds) {
  if (!in || seconds < 0.f) return NVFX_ERROR_ARGUMENT;
  in->drift_seconds = seconds;
  return NVFX_OK;
}

nvfx_status nvfx_instance_set_colour(nvfx_instance* in, float hue, float brightness) {
  if (!in || brightness < 0.f) return NVFX_ERROR_ARGUMENT;
  // Rotation about the grey axis (Rodrigues), scaled by brightness. Linear, so exact on premultiplied colour.
  const float c = std::cos(hue), s = std::sin(hue), k = (1.f - c) / 3.f, r = std::numbers::inv_sqrt3_v<float> * s;
  in->colour = {c + k, k - r, k + r, k + r, c + k, k - r, k - r, k + r, c + k};
  for (float& v : in->colour) v *= brightness;
  in->apply_colour = hue != 0.f || brightness != 1.f;
  return NVFX_OK;
}

nvfx_status nvfx_render(nvfx_instance* in, double seconds, uint8_t* rgba, size_t stride) {
  if (!in || !rgba || stride < static_cast<size_t>(in->size) * 4 || !std::isfinite(seconds)) return NVFX_ERROR_ARGUMENT;
  FrameInput f;
  f.t = prepare(*in, seconds, true);
  f.c = in->cond;
  f.colour = in->colour;
  f.apply_colour = in->apply_colour;
  in->renderer->render(f, rgba, stride);
  return NVFX_OK;
}

nvfx_status nvfx_bake(nvfx_instance* in, int frames, uint8_t* rgba) {
  if (!in || !rgba || frames < 1) return NVFX_ERROR_ARGUMENT;
  const nfx::Model& m = in->effect->e.m;
  const double duration = (m.h.loop ? m.h.frames : std::max(1, m.h.frames - 1)) / static_cast<double>(m.fps);
  const std::size_t bytes = static_cast<std::size_t>(in->size) * in->size * 4;
  for (int f = 0; f < frames; ++f) {
    const double frac = m.h.loop ? static_cast<double>(f) / frames : (frames > 1 ? static_cast<double>(f) / (frames - 1) : 0.0);
    FrameInput fi;
    fi.t = prepare(*in, frac * duration, false);  // one variation for the whole bake
    fi.c = in->cond;
    fi.colour = in->colour;
    fi.apply_colour = in->apply_colour;
    in->renderer->render(fi, rgba + bytes * static_cast<std::size_t>(f), static_cast<std::size_t>(in->size) * 4);
  }
  return NVFX_OK;
}

nvfx_isa nvfx_get_isa(void) { return resolved_isa(); }

nvfx_status nvfx_set_isa(nvfx_isa isa) {
  if (isa != NVFX_ISA_AUTO && !cpu_has(isa)) return NVFX_ERROR_UNSUPPORTED;
  g_forced_isa.store(isa);
  return NVFX_OK;
}

double nvfx_instance_macs_per_pixel(const nvfx_instance* in) { return in ? in->renderer->macs_per_pixel() : 0.0; }

}  // extern "C"
