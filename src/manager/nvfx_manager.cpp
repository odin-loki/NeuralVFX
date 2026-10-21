// The C API of the effect manager (include/neuralfx/nvfx_manager.h). Exceptions never cross the C boundary.
#include <neuralfx/nvfx_manager.h>

#include "manager.hpp"

#include <cmath>
#include <new>

struct nvfx_manager {
  explicit nvfx_manager(const nvfx_manager_desc& d) : m(d) {}
  nfx::mgr::Manager m;
};

namespace {

bool valid(const nvfx_manager_desc& d) {
  const bool sizes = nfx::mgr::level_of_size(d.copy_size) >= 0 && nfx::mgr::level_of_size(d.max_size) >= 0;
  return d.max_instances >= 1 && d.max_instances <= 65535 && d.threads >= 0 && d.threads <= 64 && (d.threads > 0 || (d.job_slots >= 1 && d.job_slots <= 64)) &&
         d.budget_ms >= 0.f && std::isfinite(d.budget_ms) && d.burst >= 1.f && std::isfinite(d.burst) && d.interp >= NVFX_INTERP_DEFAULT &&
         d.interp <= NVFX_INTERP_MOTION && sizes && d.texel_scale > 0.f && std::isfinite(d.texel_scale) && d.rate_px[0] >= d.rate_px[1] &&
         d.rate_px[1] >= d.rate_px[2] && d.rate_px[2] >= 0.f && d.hysteresis >= 0.f && d.hysteresis < 0.9f && d.min_dwell >= 0.f && d.catchup_frames >= 1.f &&
         d.fade_frames >= 0 && d.park_after >= 0.f && d.cull_after >= 0.f && d.proxy_dwell >= 0.f && d.half_px >= 0.f && std::isfinite(d.half_px) &&
         d.half_share >= 0.f && std::isfinite(d.half_share) && (d.half_upscale == NVFX_UPSCALE_OFF || d.half_upscale == NVFX_UPSCALE_BILINEAR);
}

}  // namespace

extern "C" {

void nvfx_manager_desc_init(nvfx_manager_desc* d) {
  if (!d) return;
  *d = nvfx_manager_desc{};
  d->max_instances = 256;
  d->threads = 1;
  d->job_slots = 4;
  d->budget_ms = 4.f;
  d->burst = 1.5f;
  d->interp = NVFX_INTERP_DEFAULT;
  d->max_size = 256;
  d->texel_scale = 1.f;
  d->rate_px[0] = d->rate_px[1] = d->rate_px[2] = 0.f;  // 30 Hz wherever the budget allows (docs/MANAGER.md §6)
  d->hysteresis = 0.15f;
  d->min_dwell = 1.f;
  d->catchup_frames = 8.f;
  d->copy_size = 64;
  d->fade_frames = 8;
  d->park_after = 1.f;
  d->cull_after = 30.f;
  d->proxies = 1;
  d->proxy_dwell = 2.f;
  d->seed = 1;
  d->half_px = nfx::mgr::kDefaultHalfPx;  // docs/MANAGER.md §2.10, docs/REPORT.md §22
  d->half_share = 0.f;
  d->half_upscale = NVFX_UPSCALE_OFF;  // the engine's sampler draws the half-size picture larger (docs/MANAGER.md §2.10)
}

nvfx_status nvfx_manager_create(const nvfx_manager_desc* desc, nvfx_manager** out) {
  if (!out) return NVFX_ERROR_ARGUMENT;
  *out = nullptr;
  nvfx_manager_desc d;
  nvfx_manager_desc_init(&d);
  if (desc) d = *desc;
  if (!valid(d)) return NVFX_ERROR_ARGUMENT;
  try {
    *out = new nvfx_manager(d);
    return NVFX_OK;
  } catch (const std::bad_alloc&) {
    return NVFX_ERROR_MEMORY;
  } catch (...) {
    return NVFX_ERROR_ARGUMENT;
  }
}

void nvfx_manager_free(nvfx_manager* m) { delete m; }

void nvfx_manager_model_desc_init(nvfx_manager_model_desc* d) {
  if (!d) return;
  *d = nvfx_manager_model_desc{};
  d->max_sims = 16;
  for (int& s : d->slots) s = -1;
  d->history = 16;
  d->proxy_sims = 8;
  d->offset_sims = 8;
  d->proxy_steps = 3;
  d->proxy_size = 64;
  d->proxy_rate_div = 4;
  d->shard_seconds = 6.f;
  d->interp = NVFX_INTERP_DEFAULT;
  d->half_px = -1.f;  // the manager's thresholds
}

nvfx_status nvfx_manager_add_model(nvfx_manager* m, const nvfx_effect* effect, const nvfx_manager_model_desc* desc, int* model) {
  if (!m || !effect || !model) return NVFX_ERROR_ARGUMENT;
  nvfx_manager_model_desc d;
  nvfx_manager_model_desc_init(&d);
  if (desc) d = *desc;
  try {
    return m->m.add_model(effect, d, model);
  } catch (const std::bad_alloc&) {
    return NVFX_ERROR_MEMORY;
  } catch (...) {
    return NVFX_ERROR_ARGUMENT;
  }
}

nvfx_status nvfx_manager_set_cost(nvfx_manager* m, int model, int size, float step_ms, float render_ms) {
  return m ? m->m.set_cost(model, size, step_ms, render_ms) : NVFX_ERROR_ARGUMENT;
}

nvfx_status nvfx_manager_get_cost(const nvfx_manager* m, int model, int size, float* step_ms, float* render_ms) {
  return m ? m->m.get_cost(model, size, step_ms, render_ms) : NVFX_ERROR_ARGUMENT;
}

nvfx_status nvfx_manager_calibrate(nvfx_manager* m, int model, int frames) {
  if (!m) return NVFX_ERROR_ARGUMENT;
  try {
    return m->m.calibrate(model, frames);
  } catch (const std::bad_alloc&) {
    return NVFX_ERROR_MEMORY;
  } catch (...) {
    return NVFX_ERROR_ARGUMENT;
  }
}

nvfx_status nvfx_manager_set_camera(nvfx_manager* m, const float position[3], const float forward[3], const float up[3], float fov, float w, float h) {
  return m ? m->m.set_camera(position, forward, up, fov, w, h) : NVFX_ERROR_ARGUMENT;
}

nvfx_status nvfx_manager_set_budget(nvfx_manager* m, float ms) {
  if (!m || !(ms >= 0.f) || !std::isfinite(ms)) return NVFX_ERROR_ARGUMENT;
  m->m.set_budget(ms);
  return NVFX_OK;
}

void nvfx_spawn_desc_init(nvfx_spawn_desc* d) {
  if (!d) return;
  *d = nvfx_spawn_desc{};
  d->share = 1;
  d->allow_proxy = 1;
  d->priority = 1.f;
  d->world_size = 1.f;
  d->visible = 1;
}

nvfx_status nvfx_manager_spawn(nvfx_manager* m, const nvfx_spawn_desc* d, nvfx_handle* out) {
  if (!m || !d || !out) return NVFX_ERROR_ARGUMENT;
  *out = 0;
  return m->m.spawn(*d, out);
}

nvfx_status nvfx_manager_stop(nvfx_manager* m, nvfx_handle h) { return m ? m->m.stop(h) : NVFX_ERROR_ARGUMENT; }

nvfx_status nvfx_manager_set_controls(nvfx_manager* m, nvfx_handle h, const float* controls, int count) {
  return m ? m->m.set_controls(h, controls, count) : NVFX_ERROR_ARGUMENT;
}

nvfx_status nvfx_manager_set_position(nvfx_manager* m, nvfx_handle h, const float position[3], float world_size) {
  return m ? m->m.set_position(h, position, world_size) : NVFX_ERROR_ARGUMENT;
}

nvfx_status nvfx_manager_set_screen_size(nvfx_manager* m, nvfx_handle h, float pixels, int visible) {
  return m ? m->m.set_screen(h, pixels, visible) : NVFX_ERROR_ARGUMENT;
}

nvfx_status nvfx_manager_set_priority(nvfx_manager* m, nvfx_handle h, float priority) { return m ? m->m.set_priority(h, priority) : NVFX_ERROR_ARGUMENT; }

nvfx_status nvfx_manager_set_lod_lock(nvfx_manager* m, nvfx_handle h, int size, int rate_div) { return m ? m->m.set_lock(h, size, rate_div) : NVFX_ERROR_ARGUMENT; }

nvfx_status nvfx_manager_tick(nvfx_manager* m, double dt) {
  if (!m || m->m.job_slots() == 0) return NVFX_ERROR_ARGUMENT;
  return m->m.tick(dt);
}

nvfx_status nvfx_manager_tick_begin(nvfx_manager* m, double dt, int* jobs) { return m ? m->m.begin(dt, jobs) : NVFX_ERROR_ARGUMENT; }

nvfx_status nvfx_manager_run_jobs(nvfx_manager* m, int slot) {
  if (!m || slot < 0 || slot >= m->m.job_slots()) return NVFX_ERROR_ARGUMENT;
  m->m.run_jobs(slot);
  return NVFX_OK;
}

nvfx_status nvfx_manager_tick_end(nvfx_manager* m) { return m ? m->m.end() : NVFX_ERROR_ARGUMENT; }

nvfx_status nvfx_manager_get_picture(const nvfx_manager* m, nvfx_handle h, nvfx_manager_picture* out) {
  return m ? m->m.picture(h, out) : NVFX_ERROR_ARGUMENT;
}

nvfx_status nvfx_manager_get_stats(const nvfx_manager* m, nvfx_manager_stats* out) {
  if (!m || !out) return NVFX_ERROR_ARGUMENT;
  m->m.stats(out);
  return NVFX_OK;
}

}  // extern "C"
