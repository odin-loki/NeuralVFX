// The volume C API (include/neuralfx/nvfx_volume.h; study N, docs/STUDY_3D.md): 3D rollout effects for engines. A volume
// runs the reference code of src/core/rollout3d.cpp on buffers made at creation, on its own threads (every loop writes per
// cell or per slab, so the thread count never changes a value), with the AVX2 convolution where the CPU has it and the
// detail layer on vectors (rt_detail3.hpp: the reference's fields to the bit, on AVX-512 where the CPU has it, else
// AVX2); its pictures come from src/render3d. Shards as the 2D runtime's: each a fresh rollout from a start point,
// crossfaded in.
#include <neuralfx/nvfx_volume.h>
#include <neuralfx/render3d.hpp>
#include <neuralfx/rollout3d.hpp>

#include "nvfx_internal.hpp"
#include "rt_volume.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <new>
#include <spanstream>
#include <stdfloat>
#include <string>

namespace r3 = nfx::rollout3d;

struct nvfx_volume_effect {
  r3::Model m;
};

struct nvfx_volume {
  explicit nvfx_volume(const nvfx_volume_effect* effect, int threads)
      : e(effect), pool(threads), ws(effect->m), renderer(effect->m.h.fx(), effect->m.h.fy(), effect->m.h.fz(), nfx::render3d::optics_for(effect->m.effect)) {}
  struct Runner {
    r3::State s;
    int shard = -1;  // -1: nothing started
    int frame = 0;   // steps since the rollout began (after its warm-up)
  };
  const nvfx_volume_effect* e;
  nfx::Pool pool;
  r3::Workspace ws;
  std::unique_ptr<r3::DetailKernel> detail;  // ws.detail: the detail layer on vectors (none at the baseline ISA)
  Runner run[2];
  std::vector<float> out_t, out_d, out_flow;  // a crossfaded frame
  const float* cur_t = nullptr;
  const float* cur_d = nullptr;
  const float* cur_flow = nullptr;
  nfx::render3d::Renderer renderer;
  std::vector<float> controls;
  std::uint64_t seed = 1;
  int variation = -1;
  int shard_frames = 240, fade = 15;
  long frame = -1;  // the frame the fields show (-1: none yet)
  bool prepared = false;
  nfx::rt::Profile profile;
};

std::unique_ptr<r3::DetailKernel> nfx::rt::make_volume_detail(const r3::Model& m, const r3::Workspace& ws, nvfx_isa isa) {
  switch (isa) {
    case NVFX_ISA_AVX512: return d3_avx512::make_detail3(m, ws);
    case NVFX_ISA_AVX2: return d3_avx2::make_detail3(m, ws);
    default: return nullptr;
  }
}

namespace {

std::size_t zu(int v) { return static_cast<std::size_t>(v); }

std::uint64_t mix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

nvfx_status load_from(std::istream& in, nvfx_volume_effect** out) {
  if (!out) return NVFX_ERROR_ARGUMENT;
  *out = nullptr;
  auto m = r3::load_model(in);
  if (!m) return NVFX_ERROR_FORMAT;
  if (m->h.hidden > 128 || m->starts.empty()) return NVFX_ERROR_FORMAT;
  auto* e = new (std::nothrow) nvfx_volume_effect{std::move(*m)};
  if (!e) return NVFX_ERROR_MEMORY;
  *out = e;
  return NVFX_OK;
}

// The start point of shard q: the one set, else one of the three nearest the controls, picked by the seed and the shard.
int start_for(const nvfx_volume& v, int q) {
  const auto& starts = v.e->m.starts;
  if (v.variation >= 0) return v.variation;
  int best[3] = {-1, -1, -1};
  float bd[3] = {1e30f, 1e30f, 1e30f};
  for (int i = 0; i < static_cast<int>(starts.size()); ++i) {
    float d = 0.f;
    for (std::size_t k = 0; k < starts[zu(i)].controls.size(); ++k) {
      const float c = k < v.controls.size() ? v.controls[k] : 0.5f;
      d += (starts[zu(i)].controls[k] - c) * (starts[zu(i)].controls[k] - c);
    }
    for (int j = 0; j < 3; ++j) {
      if (d < bd[j]) {
        for (int l = 2; l > j; --l) {
          bd[l] = bd[l - 1];
          best[l] = best[l - 1];
        }
        bd[j] = d;
        best[j] = i;
        break;
      }
    }
  }
  const int n = std::min(3, static_cast<int>(starts.size()));
  return best[mix64(v.seed * 31ULL + static_cast<std::uint64_t>(q)) % static_cast<std::uint64_t>(n)];
}

std::uint64_t seed_for(const nvfx_volume& v, int q) { return q == 0 ? v.seed : mix64(v.seed ^ (static_cast<std::uint64_t>(q) * 0xD1B54A32D192ED03ULL)); }

// The first frame shard q shows (shard 0 from frame 0; the others fade in over `fade` frames before their turn).
long shard_begin(const nvfx_volume& v, int q) { return q == 0 ? 0 : static_cast<long>(q) * v.shard_frames - v.fade; }

// Brings a runner to shard q's frame `local`, starting it (in a slot not in `keep`) when needed.
void ensure(nvfx_volume& v, int q, long local, int keep_slot, int& slot_out) {
  int slot = -1;
  for (int k = 0; k < 2; ++k) {
    if (v.run[k].shard == q && v.run[k].frame <= local) slot = k;
  }
  if (slot < 0) {
    slot = keep_slot == 0 ? 1 : 0;
    nvfx_volume::Runner& r = v.run[slot];
    r3::start(v.e->m, start_for(v, q), v.controls, seed_for(v, q), r.s, v.ws, &v.pool);
    r.shard = q;
    r.frame = 0;
  }
  nvfx_volume::Runner& r = v.run[slot];
  const std::uint64_t sd = seed_for(v, q);
  while (r.frame < local) {
    r3::step(v.e->m, r.s, v.controls, sd, v.ws, &v.pool);
    ++r.frame;
  }
  slot_out = slot;
}

}  // namespace

extern "C" {

nvfx_status nvfx_volume_effect_load(const char* path, nvfx_volume_effect** out) {
  if (!path || !out) return NVFX_ERROR_ARGUMENT;
  try {
    std::ifstream in(path, std::ios::binary);
    if (!in) return NVFX_ERROR_IO;
    return load_from(in, out);
  } catch (const std::bad_alloc&) {
    return NVFX_ERROR_MEMORY;
  } catch (...) {
    return NVFX_ERROR_FORMAT;
  }
}

nvfx_status nvfx_volume_effect_load_memory(const void* data, std::size_t bytes, nvfx_volume_effect** out) {
  if (!data || !out) return NVFX_ERROR_ARGUMENT;
  try {
    std::ispanstream in(std::span(static_cast<const char*>(data), bytes));
    return load_from(in, out);
  } catch (const std::bad_alloc&) {
    return NVFX_ERROR_MEMORY;
  } catch (...) {
    return NVFX_ERROR_FORMAT;
  }
}

void nvfx_volume_effect_free(nvfx_volume_effect* effect) { delete effect; }

nvfx_status nvfx_volume_effect_get_info(const nvfx_volume_effect* effect, nvfx_volume_info* info) {
  if (!effect || !info) return NVFX_ERROR_ARGUMENT;
  const r3::Model& m = effect->m;
  *info = nvfx_volume_info{};
  info->coarse[0] = m.h.rx;
  info->coarse[1] = m.h.ry;
  info->coarse[2] = m.h.rz;
  info->size[0] = m.h.fx();
  info->size[1] = m.h.fy();
  info->size[2] = m.h.fz();
  info->extent[0] = 1.f;
  info->extent[1] = static_cast<float>(m.h.ry) / static_cast<float>(m.h.rx);
  info->extent[2] = static_cast<float>(m.h.rz) / static_cast<float>(m.h.rx);
  info->fps = m.fps;
  info->n_controls = m.h.n_controls;
  info->n_start_points = static_cast<int>(m.starts.size());
  info->stored_bytes = m.storage_bytes();
  std::size_t resident = sizeof(float) * m.step_w.size();
  for (const auto& s : m.starts) resident += sizeof(float) * (s.coarse.size() + s.controls.size());
  info->resident_bytes = resident;
  std::strncpy(info->name, m.effect.c_str(), sizeof(info->name) - 1);
  return NVFX_OK;
}

const char* nvfx_volume_effect_control_name(const nvfx_volume_effect* effect, int i) {
  if (!effect || i < 0 || zu(i) >= effect->m.control_names.size()) return nullptr;
  return effect->m.control_names[zu(i)].c_str();
}

nvfx_status nvfx_volume_create(const nvfx_volume_effect* effect, int threads, nvfx_volume** out) {
  if (!effect || !out || threads < 0 || threads > 256) return NVFX_ERROR_ARGUMENT;
  *out = nullptr;
  try {
    auto v = std::make_unique<nvfx_volume>(effect, threads);
    const r3::Model& m = effect->m;
    v->profile = nfx::rt::current_profile();
    if (nfx::rt::resolve_frame_isa(v->profile.isa) != NVFX_ISA_BASELINE) v->ws.conv = nfx::rt::volume_conv3_avx2;
    v->detail = nfx::rt::make_volume_detail(m, v->ws, nfx::rt::rollout_isa(nfx::rt::Precision::float32, v->profile.isa));
    v->ws.detail = v->detail.get();
    for (auto& r : v->run) r.s = r3::make_state(m);
    v->out_t.assign(m.h.fine_cells(), 0.f);
    v->out_d.assign(m.h.fine_cells(), 0.f);
    v->out_flow.assign(zu(m.h.cells()) * 3, 0.f);
    v->controls.assign(zu(m.h.n_controls), 0.5f);
    v->shard_frames = std::max(1, static_cast<int>(std::lround(8.0 * m.fps)));
    v->fade = std::max(1, static_cast<int>(std::lround(0.5 * m.fps)));
    *out = v.release();
    return NVFX_OK;
  } catch (const std::bad_alloc&) {
    return NVFX_ERROR_MEMORY;
  } catch (...) {
    return NVFX_ERROR_ARGUMENT;
  }
}

void nvfx_volume_free(nvfx_volume* volume) { delete volume; }

std::size_t nvfx_volume_scratch_bytes(const nvfx_volume* v) {
  if (!v) return 0;
  const auto vb = [](const std::vector<float>& x) { return x.size() * sizeof(float); };
  std::size_t n = v->renderer.bytes() + vb(v->out_t) + vb(v->out_d) + vb(v->out_flow);
  const r3::Workspace& w = v->ws;
  for (const auto* x : {&w.X, &w.y1, &w.y2, &w.d, &w.mid, &w.div, &w.tmp, &w.noise, &w.cond, &w.next, &w.ufine, &w.fa, &w.fb, &w.lo, &w.hi, &w.rr, &w.aa}) n += vb(*x);
  if (v->detail) n += v->detail->bytes();
  for (const auto& r : v->run) {
    for (const auto* x : {&r.s.coarse, &r.s.pressure, &r.s.flow, &r.s.fine_t, &r.s.fine_d}) n += vb(*x);
  }
  return n;
}

nvfx_status nvfx_volume_set_controls(nvfx_volume* v, const float* controls, int count) {
  if (!v || (count > 0 && !controls) || count < 0) return NVFX_ERROR_ARGUMENT;
  for (int k = 0; k < static_cast<int>(v->controls.size()); ++k) v->controls[zu(k)] = k < count ? std::clamp(controls[k], 0.f, 1.f) : 0.5f;
  return NVFX_OK;
}

nvfx_status nvfx_volume_set_seed(nvfx_volume* v, std::uint64_t seed) {
  if (!v) return NVFX_ERROR_ARGUMENT;
  v->seed = seed;
  for (auto& r : v->run) r.shard = -1;
  v->frame = -1;
  return NVFX_OK;
}

nvfx_status nvfx_volume_set_variation(nvfx_volume* v, int start_point) {
  if (!v || start_point < -1 || start_point >= static_cast<int>(v->e->m.starts.size())) return NVFX_ERROR_ARGUMENT;
  v->variation = start_point;
  for (auto& r : v->run) r.shard = -1;
  v->frame = -1;
  return NVFX_OK;
}

nvfx_status nvfx_volume_set_shard(nvfx_volume* v, float seconds) {
  if (!v || !(seconds >= 0.f) || seconds > 3600.f) return NVFX_ERROR_ARGUMENT;
  const int frames = static_cast<int>(std::lround(static_cast<double>(seconds) * v->e->m.fps));
  if (seconds > 0.f && frames <= 2 * v->fade) return NVFX_ERROR_ARGUMENT;
  v->shard_frames = seconds > 0.f ? frames : 0;
  for (auto& r : v->run) r.shard = -1;
  v->frame = -1;
  return NVFX_OK;
}

nvfx_status nvfx_volume_step_to(nvfx_volume* v, double time_seconds) {
  if (!v || !(time_seconds >= 0.0) || time_seconds > 1e7) return NVFX_ERROR_ARGUMENT;
  const long f = static_cast<long>(std::floor(time_seconds * static_cast<double>(v->e->m.fps) + 1e-6));
  if (f == v->frame) return NVFX_OK;
  v->prepared = false;
  int a = 0;
  if (v->shard_frames == 0) {
    ensure(*v, 0, f, -1, a);
    v->cur_t = v->run[a].s.fine_t.data();
    v->cur_d = v->run[a].s.fine_d.data();
    v->cur_flow = v->run[a].s.flow.data();
    v->frame = f;
    return NVFX_OK;
  }
  const int S = v->shard_frames, F = v->fade;
  const int q = static_cast<int>(f / S);
  ensure(*v, q, f - shard_begin(*v, q), -1, a);
  if (f < static_cast<long>(q + 1) * S - F) {
    v->cur_t = v->run[a].s.fine_t.data();
    v->cur_d = v->run[a].s.fine_d.data();
    v->cur_flow = v->run[a].s.flow.data();
  } else {  // the next shard fades in
    int b = 0;
    ensure(*v, q + 1, f - shard_begin(*v, q + 1), a, b);
    const float w = static_cast<float>(f - (static_cast<long>(q + 1) * S - F) + 1) / static_cast<float>(F + 1);
    const r3::State &A = v->run[a].s, &B = v->run[b].s;
    const r3::Hyper& h = v->e->m.h;
    const std::size_t slab = zu(h.fx()) * zu(h.fz());
    v->pool.run(h.fy(), [&](int y) {
      for (std::size_t i = zu(y) * slab; i < zu(y + 1) * slab; ++i) {
        v->out_t[i] = (1.f - w) * A.fine_t[i] + w * B.fine_t[i];
        v->out_d[i] = (1.f - w) * A.fine_d[i] + w * B.fine_d[i];
      }
    });
    for (std::size_t i = 0; i < v->out_flow.size(); ++i) v->out_flow[i] = (1.f - w) * A.flow[i] + w * B.flow[i];
    v->cur_t = v->out_t.data();
    v->cur_d = v->out_d.data();
    v->cur_flow = v->out_flow.data();
  }
  v->frame = f;
  return NVFX_OK;
}

nvfx_status nvfx_volume_read(nvfx_volume* v, nvfx_volume_field field, nvfx_volume_texel texel, void* out, std::size_t bytes) {
  if (!v || !out || v->frame < 0) return NVFX_ERROR_ARGUMENT;
  if (field < NVFX_VOLUME_TEMPERATURE || field > NVFX_VOLUME_VELOCITY || (texel != NVFX_VOLUME_FLOAT && texel != NVFX_VOLUME_HALF)) return NVFX_ERROR_ARGUMENT;
  const r3::Model& m = v->e->m;
  const r3::Hyper& h = m.h;
  const int FX = h.fx(), FY = h.fy(), FZ = h.fz(), comps = field == NVFX_VOLUME_VELOCITY ? 3 : 1;
  const std::size_t need = h.fine_cells() * zu(comps) * (texel == NVFX_VOLUME_FLOAT ? 4u : 2u);
  if (bytes < need) return NVFX_ERROR_ARGUMENT;
  const nfx::render3d::Optics& o = v->renderer.optics();
  const float kf = static_cast<float>(h.fine), vscale = m.fps / static_cast<float>(h.rx);
  v->pool.run(FZ, [&](int z) {
    for (int y = 0; y < FY; ++y) {
      for (int x = 0; x < FX; ++x) {
        const std::size_t src = (zu(y) * zu(FZ) + zu(z)) * zu(FX) + zu(x), dst = ((zu(z) * zu(FY) + zu(y)) * zu(FX) + zu(x)) * zu(comps);
        float val[3] = {0.f, 0.f, 0.f};
        if (field == NVFX_VOLUME_TEMPERATURE) {
          val[0] = v->cur_t[src];
        } else if (field == NVFX_VOLUME_SOOT) {
          val[0] = v->cur_d[src];
        } else if (field == NVFX_VOLUME_EXTINCTION) {
          const float T = std::max(0.f, v->cur_t[src]);
          const float t = std::clamp((T - o.flame0) / (o.flame1 - o.flame0), 0.f, 1.f);
          val[0] = o.soot_extinction * std::max(0.f, v->cur_d[src]) + o.flame_extinction * t * t * (3.f - 2.f * t);
        } else {  // the coarse flow, trilinear at the texel's centre, in world units per second
          const float cx = std::clamp((static_cast<float>(x) + 0.5f) / kf - 0.5f, 0.f, static_cast<float>(h.rx - 1));
          const float cy = std::clamp((static_cast<float>(y) + 0.5f) / kf - 0.5f, 0.f, static_cast<float>(h.ry - 1));
          const float cz = std::clamp((static_cast<float>(z) + 0.5f) / kf - 0.5f, 0.f, static_cast<float>(h.rz - 1));
          const int x0 = std::min(static_cast<int>(cx), h.rx - 2), y0 = std::min(static_cast<int>(cy), h.ry - 2), z0 = std::min(static_cast<int>(cz), h.rz - 2);
          const float fx = cx - static_cast<float>(x0), fy = cy - static_cast<float>(y0), fz = cz - static_cast<float>(z0);
          for (int c = 0; c < 3; ++c) {
            const auto at = [&](int dx, int dy, int dz) { return v->cur_flow[((zu(y0 + dy) * zu(h.rz) + zu(z0 + dz)) * zu(h.rx) + zu(x0 + dx)) * 3 + zu(c)]; };
            const float a00 = at(0, 0, 0) + fx * (at(1, 0, 0) - at(0, 0, 0)), a10 = at(0, 0, 1) + fx * (at(1, 0, 1) - at(0, 0, 1));
            const float a01 = at(0, 1, 0) + fx * (at(1, 1, 0) - at(0, 1, 0)), a11 = at(0, 1, 1) + fx * (at(1, 1, 1) - at(0, 1, 1));
            const float b0 = a00 + fz * (a10 - a00), b1 = a01 + fz * (a11 - a01);
            val[c] = (b0 + fy * (b1 - b0)) * vscale;
          }
        }
        for (int c = 0; c < comps; ++c) {
          if (texel == NVFX_VOLUME_FLOAT) {
            static_cast<float*>(out)[dst + zu(c)] = val[c];
          } else {
            static_cast<std::uint16_t*>(out)[dst + zu(c)] = std::bit_cast<std::uint16_t>(static_cast<std::float16_t>(val[c]));
          }
        }
      }
    }
  });
  return NVFX_OK;
}

nvfx_status nvfx_volume_camera_look_at(const nvfx_volume_look_at* d, nvfx_volume_camera* out) {
  if (!d || !out || !(d->fov_y_degrees > 0.f && d->fov_y_degrees < 179.f) || !(d->aspect > 0.f)) return NVFX_ERROR_ARGUMENT;
  const float dx = d->target[0] - d->eye[0], dy = d->target[1] - d->eye[1], dz = d->target[2] - d->eye[2];
  if (dx * dx + dy * dy + dz * dz < 1e-12f) return NVFX_ERROR_ARGUMENT;
  const float n = d->near_plane > 0.f ? d->near_plane : 0.02f, f = d->far_plane > 0.f ? d->far_plane : 50.f;
  if (!(f > n)) return NVFX_ERROR_ARGUMENT;
  const auto c = nfx::render3d::look_at({d->eye[0], d->eye[1], d->eye[2]}, {d->target[0], d->target[1], d->target[2]}, {d->up[0], d->up[1], d->up[2]},
                                        d->fov_y_degrees, d->aspect, n, f);
  std::memcpy(out->view, c.view.data(), sizeof(out->view));
  std::memcpy(out->proj, c.proj.data(), sizeof(out->proj));
  for (const float x : out->view) {
    if (!std::isfinite(x)) return NVFX_ERROR_ARGUMENT;
  }
  return NVFX_OK;
}

nvfx_status nvfx_volume_render(nvfx_volume* v, const nvfx_volume_camera* camera, int width, int height, std::uint8_t* rgba, std::size_t stride_bytes) {
  if (!v || !camera || !rgba || width < 1 || height < 1 || width > 8192 || height > 8192 || stride_bytes < zu(width) * 4 || v->frame < 0) {
    return NVFX_ERROR_ARGUMENT;
  }
  nfx::render3d::Camera cam;
  std::memcpy(cam.view.data(), camera->view, sizeof(camera->view));
  std::memcpy(cam.proj.data(), camera->proj, sizeof(camera->proj));
  if (!nfx::render3d::invertible(cam)) return NVFX_ERROR_ARGUMENT;
  const std::size_t n = v->e->m.h.fine_cells();
  if (!v->prepared) {
    v->renderer.prepare(std::span<const float>(v->cur_t, n), std::span<const float>(v->cur_d, n), &v->pool);
    v->prepared = true;
  }
  v->renderer.render(cam, width, height, rgba, stride_bytes, &v->pool);
  return NVFX_OK;
}

}  // extern "C"
