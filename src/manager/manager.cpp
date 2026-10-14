// The effect manager: planning, shard timelines advanced to keyframes, and the jobs (manager.hpp, docs/MANAGER.md).
#include "manager.hpp"

#include "../runtime/nvfx_internal.hpp"

#include <neuralfx/noise.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

namespace nfx::mgr {

namespace {

std::size_t z(int v) { return static_cast<std::size_t>(v); }

std::uint64_t mix64(std::uint64_t x) {  // splitmix64's finaliser
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Lean runners (rt_common.hpp): what a step or a picture uses up lives in the step scratch of the thread that runs the
// job, so a simulation holds only its state.
std::unique_ptr<rt::RolloutRunner> make_runner(const rt::RolloutEffect& e, int size) {
  switch (nvfx_get_isa()) {
    case NVFX_ISA_AVX512: return rt::isa_avx512::make_rollout(e, size, true);
    case NVFX_ISA_AVX2: return rt::isa_avx2::make_rollout(e, size, true);
    default: return rt::isa_base::make_rollout(e, size, true);
  }
}

// The planner's defaults (docs/MANAGER.md §6): one AVX2 core of the project's machine, milliseconds per simulation step
// and per keyframe picture at 32, 64, 128 and 256 px, and per composed picture.
constexpr std::array<float, kLevels> kStepMs{0.33f, 0.36f, 0.48f, 1.05f};
constexpr std::array<float, kLevels> kRenderMs{0.064f, 0.11f, 0.30f, 1.30f};
constexpr std::array<float, kLevels> kFadeMs{0.0007f, 0.0023f, 0.009f, 0.031f};
constexpr std::array<float, kLevels> kMotionMs{0.021f, 0.082f, 0.335f, 1.35f};
constexpr nvfx_interp kDefaultInterp = NVFX_INTERP_CROSSFADE;  // the quality study's choice (docs/MANAGER.md §5.3)

struct ShardPlan {
  int index = 0;
  std::uint64_t seed = 0;
  std::int64_t first = 0;
};

// The shard timeline of src/runtime/nvfx.cpp (pick_start, plan_shard), for a simulation with no replayed variation.
int pick_start(const Model& M, const Sim& s, std::int64_t shard) {
  const auto& m = M.re->m;
  std::array<int, 3> best{-1, -1, -1};
  std::array<float, 3> dist{1e30f, 1e30f, 1e30f};
  for (std::size_t i = 0; i < m.starts.size(); ++i) {
    float d = 0.f;
    for (std::size_t k = 0; k < m.starts[i].controls.size() && k < z(M.nc); ++k) {
      const float e = m.starts[i].controls[k] - s.controls[k];
      d += e * e;
    }
    for (std::size_t j = 0; j < best.size(); ++j) {
      if (d < dist[j]) {
        for (std::size_t q = best.size() - 1; q > j; --q) {
          best[q] = best[q - 1];
          dist[q] = dist[q - 1];
        }
        best[j] = static_cast<int>(i);
        dist[j] = d;
        break;
      }
    }
  }
  const int count = static_cast<int>(std::min<std::size_t>(3, m.starts.size()));
  const std::uint32_t h = nfx::hash_cell(static_cast<std::int32_t>(shard), static_cast<std::int32_t>(shard >> 31), 17, s.seed);
  return best[h % static_cast<std::uint32_t>(count)];
}

ShardPlan plan_shard(const Model& M, const Sim& s, std::int64_t k) {
  const auto& m = M.re->m;
  ShardPlan p;
  p.index = pick_start(M, s, k);
  p.seed = k == 0 ? s.seed : s.seed ^ (0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(k + 1));
  const std::int64_t pre = m.starts[z(p.index)].fine_t.empty() ? m.h.warmup : 0;
  p.first = (M.shard == 0 || k == 0) ? -pre : k * M.shard - kBlend - pre;
  return p;
}

// The smallest allowed level whose size holds px, else the largest.
int level_for(const Model& M, float px) {
  for (int L = M.min_level; L <= M.max_level; ++L) {
    if (M.slots_n[z(L)] > 0 && static_cast<float>(kSizes[z(L)]) >= px) return L;
  }
  return M.max_level;
}

int div_for(const std::array<float, 3>& t, float px) {
  if (px >= t[0]) return 1;
  if (px >= t[1]) return 2;
  if (px >= t[2]) return 3;
  return 4;
}

}  // namespace

int level_of_size(int size) {
  for (int L = 0; L < kLevels; ++L) {
    if (kSizes[z(L)] == size) return L;
  }
  return -1;
}

// --- set-up -------------------------------------------------------------------------------------------------------------

Manager::Manager(const nvfx_manager_desc& d) : desc_(d) {
  interp_ = d.interp == NVFX_INTERP_DEFAULT ? kDefaultInterp : d.interp;
  fade_ms_ = kFadeMs;
  motion_ms_ = kMotionMs;
  insts_.resize(z(d.max_instances));
  free_insts_.reserve(insts_.size());
  for (int i = d.max_instances - 1; i >= 0; --i) free_insts_.push_back(i);
  const std::size_t side = static_cast<std::size_t>(std::clamp(d.copy_size, 32, 256)), copy = side * side * 4;
  for (Inst& in : insts_) {
    in.buf.assign(copy, 0);
    in.snap.assign(copy, 0);
  }
  order_inst_.reserve(insts_.size());
  const int slots = d.threads > 0 ? d.threads : d.job_slots;
  work_.resize(z(slots));
  for (Work& w : work_) {
    w.scratch = std::make_unique<rt::RolloutScratch>();
    w.px.assign(copy, 0);
  }
  memory_ = insts_.size() * (2 * copy + sizeof(Inst)) + work_.size() * copy;
  for (int i = 1; i < d.threads; ++i) threads_.emplace_back([this, i] { worker(i); });
}

Manager::~Manager() {
  stop_.store(true);
  epoch_.fetch_add(1);
  epoch_.notify_all();
  for (auto& t : threads_) t.join();
}

void Manager::fit_work(const rt::RolloutRunner& r, int res) {
  for (Work& w : work_) {
    w.scratch->fit(r);
    if (w.uv.size() < z(res) * z(res) * 2) w.uv.assign(z(res) * z(res) * 2, 0.f);
  }
}

nvfx_status Manager::add_model(const nvfx_effect* fx, const nvfx_manager_model_desc& d, int* id) {
  const rt::RolloutEffect* re = rt::rollout_of(fx);
  if (!re) return NVFX_ERROR_UNSUPPORTED;
  if (d.max_sims < 1 || d.max_sims > 4096 || d.history < 0 || d.history > 256 || d.proxy_steps < 1 || d.proxy_steps > 16 || d.proxy_rate_div < 1 ||
      d.proxy_rate_div > kMaxDiv || !(d.shard_seconds > 0.f) || level_of_size(d.proxy_size) < 0 || d.proxy_sims < 0 || d.proxy_sims > 4096 ||
      d.offset_sims < 0 || d.offset_sims > 4096 || d.interp < NVFX_INTERP_DEFAULT || d.interp > NVFX_INTERP_MOTION) {
    return NVFX_ERROR_ARGUMENT;
  }
  const auto& m = re->m;
  const auto& h = m.h;
  if (h.n_controls > kMaxControls || m.starts.empty()) return NVFX_ERROR_UNSUPPORTED;
  Model M;
  M.fx = fx;
  M.re = re;
  M.res = h.res;
  M.nc = h.n_controls;
  M.fps = m.fps;
  M.loop = m.loop;
  M.frames = h.frames;
  M.advect = m.detail.advect;
  M.shard = M.loop ? std::max<std::int64_t>(4 * kBlend, std::lround(d.shard_seconds * m.fps)) : 0;
  float pre = 0.f;
  for (const auto& sp : m.starts) pre += sp.fine_t.empty() ? static_cast<float>(h.warmup) : 0.f;
  pre /= static_cast<float>(m.starts.size());
  M.step_factor = M.shard > 0 ? 1.f + (pre + static_cast<float>(kBlend)) / static_cast<float>(M.shard) : 1.f;
  M.render_factor = M.shard > 0 ? 1.f + static_cast<float>(kBlend) / static_cast<float>(M.shard) : 1.f;
  M.min_level = -1;
  for (int L = 0; L < kLevels; ++L) {
    const int size = kSizes[z(L)];
    const bool allowed = size <= desc_.max_size && size >= h.res && size % h.res == 0;
    const int def = std::max(1, d.max_sims >> (L == 0 ? 0 : L == 1 ? 1 : L == 2 ? 2 : 4));
    const int n = !allowed ? 0 : d.slots[L] < 0 ? def : d.slots[L];
    M.slots_n[z(L)] = std::min(n, d.max_sims);
    if (M.slots_n[z(L)] > 0) {
      if (M.min_level < 0) M.min_level = L;
      M.max_level = L;
    }
  }
  if (M.min_level < 0) return NVFX_ERROR_UNSUPPORTED;
  M.history = d.history >= 3 ? d.history : 0;
  M.proxy_steps = d.proxy_steps;
  M.proxy_level = std::clamp(level_of_size(d.proxy_size), M.min_level, M.max_level);
  while (M.proxy_level > M.min_level && M.slots_n[z(M.proxy_level)] == 0) --M.proxy_level;
  M.proxy_div = d.proxy_rate_div;
  M.interp = d.interp == NVFX_INTERP_DEFAULT ? interp_ : d.interp;
  std::size_t ncell = 1;
  for (int k = 0; k < M.nc && ncell <= 4096; ++k) ncell *= z(M.proxy_steps);
  if (ncell > 4096 || M.history == 0 || !M.loop) ncell = 0;  // no proxies: too many controls, no history, or a one-shot
  M.own_sims = d.max_sims;
  M.proxy_max = static_cast<int>(std::min<std::size_t>(ncell, z(d.proxy_sims)));
  M.step_ms = kStepMs;
  M.render_ms = kRenderMs;
  try {
    std::size_t bytes = 0;
    for (int L = 0; L < kLevels; ++L) {
      const int size = kSizes[z(L)];
      const std::size_t px = z(size) * z(size) * 4;
      const int reserved = L == M.proxy_level ? M.proxy_max : 0;
      M.free[z(L)].reserve(z(M.slots_n[z(L)]));
      M.proxy_free.reserve(z(M.proxy_max));
      for (int i = 0; i < M.slots_n[z(L)] + reserved; ++i) {
        auto S = std::make_unique<Slot>();
        S->level = L;
        S->size = size;
        S->proxy_pool = i >= M.slots_n[z(L)];
        for (Track& t : S->t) {
          t.r = make_runner(*re, size);
          fit_work(*t.r, h.res);
          t.r->use_scratch(work_[0].scratch.get());  // frees its own; each job binds its thread's
          bytes += t.r->scratch_bytes();
        }
        S->mix.assign(px, 0);
        S->out.assign(px, 0);
        for (auto& k : S->key) k.assign(px, 0);
        bytes += 4 * px + sizeof(Slot);
        (S->proxy_pool ? M.proxy_free : M.free[z(L)]).push_back(S.get());
        M.slots.push_back(std::move(S));
      }
    }
    const int nsims = M.own_sims + M.proxy_max;
    M.sims.resize(z(nsims));
    M.free_sims.reserve(z(M.own_sims));
    M.free_proxy.reserve(z(M.proxy_max));
    const std::size_t cells = z(h.res) * z(h.res) * 2;
    for (int i = M.own_sims - 1; i >= 0; --i) M.free_sims.push_back(i);
    for (int i = nsims - 1; i >= M.own_sims; --i) M.free_proxy.push_back(i);
    for (int i = 0; i < nsims; ++i) {
      Sim& s = M.sims[z(i)];
      s.index = i;
      s.model = static_cast<int>(models_.size());
      s.acc.assign(cells, 0.f);
      s.dab.assign(cells, 0.f);
      bytes += 2 * 4 * cells + sizeof(Sim);
    }
    const int nrings = M.history > 0 ? M.proxy_max + d.offset_sims : 0;
    for (int i = 0; i < nrings; ++i) {
      auto R = std::make_unique<Ring>();
      R->size = desc_.copy_size;
      R->cap = M.history;
      R->res = h.res;
      R->px.assign(z(R->cap) * z(R->size) * z(R->size) * 4, 0);
      R->d.assign(z(R->cap) * cells, 0);
      R->scale.assign(z(R->cap), 0.f);
      R->frame.assign(z(R->cap), 0);
      bytes += R->px.size() + R->d.size() + 12 * z(R->cap) + sizeof(Ring);
      M.rings.push_back(std::move(R));
    }
    M.free_rings.reserve(z(nrings));
    for (int i = nrings - 1; i >= M.proxy_max; --i) M.free_rings.push_back(M.rings[z(i)].get());
    M.proxy_sim.assign(ncell, -1);
    M.waiting.assign(ncell, 0);
    M.planned.assign(ncell, 0);
    std::size_t total = 0;
    for (const Model& o : models_) total += o.sims.size();
    total += M.sims.size();
    jobs_.reserve(total);
    order_.reserve(total);
    memory_ += bytes;
    models_.push_back(std::move(M));
  } catch (const std::bad_alloc&) {
    return NVFX_ERROR_MEMORY;
  }
  *id = static_cast<int>(models_.size()) - 1;
  return NVFX_OK;
}

nvfx_status Manager::set_cost(int model, int size, float step_ms, float render_ms) {
  const int L = level_of_size(size);
  if (model < 0 || model >= static_cast<int>(models_.size()) || L < 0 || !(step_ms >= 0.f) || !(render_ms >= 0.f)) return NVFX_ERROR_ARGUMENT;
  models_[z(model)].step_ms[z(L)] = step_ms;
  models_[z(model)].render_ms[z(L)] = render_ms;
  return NVFX_OK;
}

nvfx_status Manager::get_cost(int model, int size, float* step_ms, float* render_ms) const {
  const int L = level_of_size(size);
  if (model < 0 || model >= static_cast<int>(models_.size()) || L < 0) return NVFX_ERROR_ARGUMENT;
  if (step_ms) *step_ms = models_[z(model)].step_ms[z(L)];
  if (render_ms) *render_ms = models_[z(model)].render_ms[z(L)];
  return NVFX_OK;
}

nvfx_status Manager::calibrate(int model, int frames) {
  if (model < 0 || model >= static_cast<int>(models_.size()) || frames < 4 || in_tick_) return NVFX_ERROR_ARGUMENT;
  Model& M = models_[z(model)];
  try {
    const std::vector<float> ctl(z(M.nc), 0.5f);
    for (int L = M.min_level; L <= M.max_level; ++L) {
      if (M.slots_n[z(L)] == 0) continue;
      const int size = kSizes[z(L)];
      auto r = make_runner(*M.re, size);
      rt::RolloutScratch sc;
      sc.fit(*r);
      r->use_scratch(&sc);
      r->start(0, ctl, 12345);
      std::vector<std::uint8_t> a(z(size) * z(size) * 4), b(a.size()), o(a.size());
      rt::FrameInput fi;
      std::vector<double> st, rd, cf, mo;
      std::vector<float> uv(z(M.res) * z(M.res) * 2, 0.f);
      for (int i = 0; i < frames + 4; ++i) {
        const double t0 = now_ms();
        r->step(ctl, 12345);
        const double t1 = now_ms();
        r->render(fi, (i % 2) ? a.data() : b.data(), z(size) * 4);
        const double t2 = now_ms();
        const auto flow = r->flow();
        std::copy(flow.begin(), flow.end(), uv.begin());
        crossfade(a.data(), b.data(), size, 0.5f, o.data());
        const double t3 = now_ms();
        motion_blend(a.data(), b.data(), size, 0.5f, Motion{uv.data(), M.res, static_cast<float>(size) / static_cast<float>(M.res) * M.advect}, o.data());
        const double t4 = now_ms();
        if (i < 4) continue;  // warm caches
        st.push_back(t1 - t0);
        rd.push_back(t2 - t1);
        cf.push_back(t3 - t2);
        mo.push_back(t4 - t3);
      }
      const auto med = [](std::vector<double>& v) {
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
        return static_cast<float>(v[v.size() / 2]);
      };
      M.step_ms[z(L)] = med(st);
      M.render_ms[z(L)] = med(rd);
      fade_ms_[z(L)] = med(cf);
      motion_ms_[z(L)] = med(mo);
    }
  } catch (const std::bad_alloc&) {
    return NVFX_ERROR_MEMORY;
  }
  return NVFX_OK;
}

nvfx_status Manager::set_camera(const float* pos, const float* fwd, const float* up, float fov, float w, float h) {
  if (!pos || !fwd || !up || !(fov > 0.f && fov < 3.14f) || !(w > 0.f) || !(h > 0.f)) return NVFX_ERROR_ARGUMENT;
  const auto norm = [](std::array<float, 3> v) {
    const float n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return n > 0.f ? std::array<float, 3>{v[0] / n, v[1] / n, v[2] / n} : v;
  };
  const auto cross = [](const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return std::array<float, 3>{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
  };
  const std::array<float, 3> f = norm({fwd[0], fwd[1], fwd[2]});
  std::array<float, 3> r = norm(cross(f, {up[0], up[1], up[2]}));
  if (!(std::abs(r[0]) + std::abs(r[1]) + std::abs(r[2]) > 0.f)) return NVFX_ERROR_ARGUMENT;
  cam_pos_ = {pos[0], pos[1], pos[2]};
  cam_fwd_ = f;
  cam_right_ = r;
  cam_up_ = cross(r, f);
  focal_ = h / (2.f * std::tan(0.5f * fov));
  view_w_ = w;
  view_h_ = h;
  camera_ = true;
  return NVFX_OK;
}

// --- instances ------------------------------------------------------------------------------------------------------------

Inst* Manager::find(nvfx_handle h) { return const_cast<Inst*>(static_cast<const Manager*>(this)->find(h)); }

const Inst* Manager::find(nvfx_handle h) const {
  const std::uint32_t idx = (h & 0xFFFFu), gen = h >> 16;
  if (idx == 0 || idx > insts_.size()) return nullptr;
  const Inst& in = insts_[idx - 1];
  return in.used && in.alive && in.gen == gen ? &in : nullptr;
}

nvfx_status Manager::spawn(const nvfx_spawn_desc& d, nvfx_handle* out) {
  if (in_tick_ || d.model < 0 || d.model >= static_cast<int>(models_.size()) || d.n_controls < 0 || (d.n_controls > 0 && !d.controls) ||
      !(d.time_offset >= 0.f) || !(d.priority >= 0.f) || !std::isfinite(d.world_size) || !std::isfinite(d.screen_px)) {
    return NVFX_ERROR_ARGUMENT;
  }
  if (free_insts_.empty()) return NVFX_ERROR_MEMORY;
  const int idx = free_insts_.back();
  free_insts_.pop_back();
  Inst& in = insts_[z(idx)];
  const Model& M = models_[z(d.model)];
  std::vector<std::uint8_t> buf = std::move(in.buf), snap = std::move(in.snap);
  const std::uint16_t gen = static_cast<std::uint16_t>(in.gen == 0xFFFF ? 1 : in.gen + 1);
  in = Inst{};
  in.buf = std::move(buf);
  in.snap = std::move(snap);
  in.gen = gen;
  in.used = in.alive = true;
  in.model = d.model;
  in.nc = M.nc;
  in.controls.fill(0.5f);
  for (int k = 0; k < std::min(d.n_controls, M.nc); ++k) in.controls[z(k)] = d.controls[k];
  in.seed = d.seed;
  in.share = d.share != 0;
  in.allow_proxy = d.allow_proxy != 0;
  in.offset_s = d.time_offset;
  in.priority = d.priority;
  in.pos = {d.position[0], d.position[1], d.position[2]};
  in.world_size = d.world_size;
  in.screen_mode = d.screen_px > 0.f;
  in.screen_px = d.screen_px;
  in.screen_visible = d.visible != 0;
  *out = (static_cast<std::uint32_t>(gen) << 16) | static_cast<std::uint32_t>(idx + 1);
  return NVFX_OK;
}

nvfx_status Manager::stop(nvfx_handle h) {
  Inst* in = find(h);
  if (!in || in_tick_) return NVFX_ERROR_ARGUMENT;
  in->alive = false;  // detached and freed by the next tick
  return NVFX_OK;
}

nvfx_status Manager::set_controls(nvfx_handle h, const float* c, int n) {
  Inst* in = find(h);
  if (!in || in_tick_ || n < 0 || (n > 0 && !c)) return NVFX_ERROR_ARGUMENT;
  bool changed = false;
  for (int k = 0; k < std::min(n, in->nc); ++k) {
    changed |= std::memcmp(&in->controls[z(k)], &c[k], sizeof(float)) != 0;
    in->controls[z(k)] = c[k];
  }
  in->controls_dirty |= changed;
  return NVFX_OK;
}

nvfx_status Manager::set_position(nvfx_handle h, const float* p, float size) {
  Inst* in = find(h);
  if (!in || in_tick_ || !p || !std::isfinite(size)) return NVFX_ERROR_ARGUMENT;
  in->pos = {p[0], p[1], p[2]};
  in->world_size = size;
  in->screen_mode = false;
  return NVFX_OK;
}

nvfx_status Manager::set_screen(nvfx_handle h, float px, int visible) {
  Inst* in = find(h);
  if (!in || in_tick_ || !std::isfinite(px) || px < 0.f) return NVFX_ERROR_ARGUMENT;
  in->screen_mode = true;
  in->screen_px = px;
  in->screen_visible = visible != 0;
  return NVFX_OK;
}

nvfx_status Manager::set_priority(nvfx_handle h, float p) {
  Inst* in = find(h);
  if (!in || in_tick_ || !(p >= 0.f) || !std::isfinite(p)) return NVFX_ERROR_ARGUMENT;
  in->priority = p;
  return NVFX_OK;
}

nvfx_status Manager::set_lock(nvfx_handle h, int size, int div) {
  Inst* in = find(h);
  if (!in || in_tick_) return NVFX_ERROR_ARGUMENT;
  if (size == 0 && div == 0) {
    in->lock_level = -1;
    in->lock_div = 0;
    return NVFX_OK;
  }
  const Model& M = models_[z(in->model)];
  const int L = level_of_size(size);
  if (L < 0 || div < 1 || div > kMaxDiv || M.slots_n[z(L)] == 0) return NVFX_ERROR_ARGUMENT;
  in->lock_level = L;
  in->lock_div = div;
  return NVFX_OK;
}

// --- simulations ------------------------------------------------------------------------------------------------------------

int Manager::new_sim(Model& M, int model, bool proxy) {
  std::vector<int>& from = proxy ? M.free_proxy : M.free_sims;
  if (from.empty()) return -1;
  const int i = from.back();
  from.pop_back();
  Sim& s = M.sims[z(i)];
  std::vector<float> acc = std::move(s.acc), dab = std::move(s.dab);
  s = Sim{};
  s.acc = std::move(acc);
  s.dab = std::move(dab);
  std::fill(s.acc.begin(), s.acc.end(), 0.f);
  std::fill(s.dab.begin(), s.dab.end(), 0.f);
  s.used = true;
  s.model = model;
  s.index = i;
  s.serial = ++serial_;
  const std::uint64_t h = mix64(s.serial ^ mix64(desc_.seed));
  s.subframe = static_cast<double>(h % 1024) / 1024.0 / static_cast<double>(M.fps);
  return i;
}

void Manager::free_slot(Model& M, Slot*& slot) {
  if (!slot) return;
  slot->used = false;
  (slot->proxy_pool ? M.proxy_free : M.free[z(slot->level)]).push_back(slot);
  slot = nullptr;
}

void Manager::release_sim(Model& M, Sim& s) {
  free_slot(M, s.slot);
  free_slot(M, s.next_slot);
  free_slot(M, s.old_slot);
  const bool proxy = s.index >= M.own_sims;
  if (s.proxy_cell >= 0 && z(s.proxy_cell) < M.proxy_sim.size() && M.proxy_sim[z(s.proxy_cell)] == s.index) M.proxy_sim[z(s.proxy_cell)] = -1;
  if (s.ring) {
    s.ring->count = s.ring->head = 0;
    if (!proxy) M.free_rings.push_back(s.ring);
    s.ring = nullptr;
  }
  s.used = false;
  (proxy ? M.free_proxy : M.free_sims).push_back(s.index);
}

// No simulation free for a new instance: the one paused longest goes (culled; its viewers start afresh when seen again).
bool Manager::evict(Model& M) {
  Sim* worst = nullptr;
  for (Sim& s : M.sims) {
    if (s.used && s.index < M.own_sims && s.paused && (!worst || s.paused_s > worst->paused_s)) worst = &s;
  }
  if (!worst) return false;
  for (Inst& in : insts_) {
    if (!in.used || in.model != worst->model) continue;
    if (in.own_wait == worst->index) {
      in.own_wait = -1;
      worst->refs--;
    }
    if (in.sim == worst->index) {
      in.sim = -1;
      in.offset = 0;
      worst->refs--;
      in.culled = true;
      in.finished = !M.loop;
      in.pic = nullptr;
    }
  }
  release_sim(M, *worst);
  return true;
}

void Manager::attach(Inst& in, int model, int sim, std::int64_t offset) {
  in.model = model;
  in.sim = sim;
  in.offset = offset;
  in.buf_frame = -1;
  models_[z(model)].sims[z(sim)].refs++;
}

void Manager::detach(Inst& in) {
  Model& M = models_[z(in.model)];
  if (in.own_wait >= 0) {
    Sim& w = M.sims[z(in.own_wait)];
    if (--w.refs <= 0) release_sim(M, w);
    in.own_wait = -1;
  }
  if (in.sim < 0) return;
  Sim& s = M.sims[z(in.sim)];
  in.sim = -1;
  in.offset = 0;
  if (--s.refs <= 0 && s.proxy_cell < 0) release_sim(M, s);  // proxies linger (begin releases idle ones)
}

int Manager::find_shared(const Model& M, const Inst& in) const {
  for (const Sim& s : M.sims) {
    if (!s.used || !s.shareable || s.proxy_cell >= 0 || s.seed != in.seed || s.refs <= 0) continue;
    if (std::memcmp(s.controls.data(), in.controls.data(), z(M.nc) * sizeof(float)) != 0) continue;
    if (!M.loop && s.started) continue;  // a one-shot joins only a simulation that has not begun
    return s.index;
  }
  return -1;
}

int Manager::proxy_cell(const Model& M, const Inst& in) const {
  if (!desc_.proxies || !M.loop || M.history < 3 || M.proxy_sim.empty()) return -1;
  int cell = 0, mul = 1;
  for (int k = 0; k < M.nc; ++k) {
    const int q = std::clamp(static_cast<int>(std::floor(in.controls[z(k)] * static_cast<float>(M.proxy_steps))), 0, M.proxy_steps - 1);
    cell += q * mul;
    mul *= M.proxy_steps;
  }
  return cell;
}

// The proxy simulation of a cell: created (with a pre-roll that fills its ring) if it does not exist. -1 if none can be.
int Manager::ensure_proxy(int model, int cell) {
  Model& M = models_[z(model)];
  if (cell < 0 || z(cell) >= M.proxy_sim.size()) return -1;
  if (M.proxy_sim[z(cell)] >= 0) {
    M.sims[z(M.proxy_sim[z(cell)])].idle_s = 0.0;  // wanted: it does not linger out
    return M.proxy_sim[z(cell)];
  }
  if (M.proxy_free.empty()) {  // every proxy is in use: the idle one that has waited longest goes
    Sim* idle = nullptr;
    for (int k = M.own_sims; k < M.own_sims + M.proxy_max; ++k) {
      Sim& q = M.sims[z(k)];
      if (q.used && q.refs <= 0 && (!idle || q.idle_s > idle->idle_s)) idle = &q;
    }
    if (!idle) return -1;
    release_sim(M, *idle);
  }
  const int i = new_sim(M, model, true);
  if (i < 0) return -1;
  Sim& s = M.sims[z(i)];
  s.slot = M.proxy_free.back();
  M.proxy_free.pop_back();
  s.slot->used = true;
  for (Track& t : s.slot->t) t.shard = -1;
  s.ring = M.rings[z(i - M.own_sims)].get();
  s.ring->head = s.ring->count = 0;
  int c = cell;
  for (int k = 0; k < M.nc; ++k) {
    s.controls[z(k)] = (static_cast<float>(c % M.proxy_steps) + 0.5f) / static_cast<float>(M.proxy_steps);
    c /= M.proxy_steps;
  }
  s.seed = mix64(desc_.seed ^ (0xA24BAED4963EE407ULL * static_cast<std::uint64_t>(cell + 1)) ^ static_cast<std::uint64_t>(model));
  s.proxy_cell = cell;
  s.div = M.proxy_div;
  s.preroll = static_cast<std::int64_t>(M.history - 2) * M.proxy_div;
  M.proxy_sim[z(cell)] = i;
  return i;
}

// The cell of the existing (or planned) proxy whose controls are nearest those of `cell`'s centre; -1 if none.
int Manager::nearest_proxy(const Model& M, int cell) const {
  const auto centre = [&](int c, int k) {
    for (int q = 0; q < k; ++q) c /= M.proxy_steps;
    return c % M.proxy_steps;
  };
  int best = -1, dist = std::numeric_limits<int>::max();
  for (std::size_t q = 0; q < M.proxy_sim.size(); ++q) {
    if (M.proxy_sim[q] < 0 && !M.planned[q]) continue;
    int d = 0;
    for (int k = 0; k < M.nc; ++k) {
      const int e = centre(static_cast<int>(q), k) - centre(cell, k);
      d += e * e;
    }
    if (d < dist) {
      dist = d;
      best = static_cast<int>(q);
    }
  }
  return best;
}

// Whether a simulation's viewers may all go to a proxy now: looping, proxies on, every viewer allows it, is not locked,
// and has kept its source for proxy_dwell (or has not been shown yet).
bool Manager::proxy_ok(const Model& M, const Sim& s) const {
  if (!s.all_proxy || !M.loop || M.history < 3 || !desc_.proxies || s.first_viewer < 0 || M.proxy_sim.empty()) return false;
  for (int v = s.first_viewer; v >= 0; v = insts_[z(v)].next_viewer) {
    const Inst& in = insts_[z(v)];
    if (!(now_ - in.switched_at >= static_cast<double>(desc_.proxy_dwell) || !s.started) || proxy_cell(M, in) < 0) return false;
  }
  return true;
}

// A proxy's viewers play it at offsets of their own (from their seed) within its ring, so they do not move in step.
std::int64_t Manager::proxy_offset(const Model& M, const Inst& in, int p) const {
  const std::int64_t span = std::max<std::int64_t>(1, static_cast<std::int64_t>(M.history - 3) * M.proxy_div + 1);
  return static_cast<std::int64_t>(mix64(in.seed ^ mix64(static_cast<std::uint64_t>(p) + desc_.seed)) % static_cast<std::uint64_t>(span));
}

void Manager::start_fade(Inst& in) {
  if (!in.pic || desc_.fade_frames <= 0) return;
  resample(in.pic, in.pic_size, in.snap.data(), desc_.copy_size);
  in.fade = static_cast<float>(desc_.fade_frames) / models_[z(in.model)].fps;
}

// An instance leaves a shared simulation with new controls: a simulation of its own continues from a copy of the shared
// one's state at the same size (or a smaller one), so nothing restarts.
bool Manager::fork(Inst& in) {
  Model& M = models_[z(in.model)];
  Sim& src = M.sims[z(in.sim)];
  const int i = new_sim(M, in.model);
  if (i < 0) return false;
  Sim& s = M.sims[z(i)];
  s.controls = in.controls;
  s.seed = in.seed;
  s.shareable = in.share;
  if (src.slot && src.started) {
    Slot* to = nullptr;
    for (int L = src.slot->level; L >= M.min_level && !to; --L) {
      if (!M.free[z(L)].empty()) {
        to = M.free[z(L)].back();
        M.free[z(L)].pop_back();
      }
    }
    if (to) {
      to->used = true;
      const Slot& from = *src.slot;
      for (std::size_t k = 0; k < 2; ++k) {
        to->t[k].shard = from.t[k].shard;
        to->t[k].frame = from.t[k].frame;
        to->t[k].seed = from.t[k].seed;
        if (from.t[k].shard >= 0) to->t[k].r->transfer_from(*from.t[k].r);
      }
      for (std::size_t k = 0; k < 2; ++k) resample(from.key[k].data(), from.size, to->key[k].data(), to->size);
      s.slot = to;
      s.started = src.started;
      s.ready = src.ready;
      s.clock = src.clock;
      s.subframe = src.subframe;
      s.div = src.div;
      s.phase = src.phase;
      s.ka = src.ka;
      s.fa = src.fa;
      s.fb = src.fb;
      s.acc = src.acc;
      s.dab = src.dab;
      s.last_up = src.last_up;
    }
  }
  in.offset_s = 0.f;
  detach(in);
  attach(in, in.model, i, 0);
  return true;
}

// --- planning -----------------------------------------------------------------------------------------------------------------

void Manager::view(Inst& in) {
  float px = 0.f;
  bool vis = false;
  if (in.screen_mode) {
    px = in.screen_px;
    vis = in.screen_visible && px > 0.f;
  } else if (camera_) {
    const std::array<float, 3> r{in.pos[0] - cam_pos_[0], in.pos[1] - cam_pos_[1], in.pos[2] - cam_pos_[2]};
    const float d = r[0] * cam_fwd_[0] + r[1] * cam_fwd_[1] + r[2] * cam_fwd_[2];
    if (d > 1e-4f) {
      px = in.world_size * focal_ / d;
      const float sx = (r[0] * cam_right_[0] + r[1] * cam_right_[1] + r[2] * cam_right_[2]) * focal_ / d;
      const float sy = (r[0] * cam_up_[0] + r[1] * cam_up_[1] + r[2] * cam_up_[2]) * focal_ / d;
      vis = std::abs(sx) - 0.5f * px <= 0.5f * view_w_ && std::abs(sy) - 0.5f * px <= 0.5f * view_h_ && px > 0.f;
    }
  } else {
    px = in.world_size;  // no camera: world units are pixels
    vis = px > 0.f;
  }
  in.visible = vis;
  in.px = vis ? px : 0.f;
  in.importance = vis ? px * in.priority : 0.f;
}

void Manager::want(Model& M, Sim& s) {
  if (s.proxy_cell >= 0) {
    s.want_level = M.proxy_level;
    s.want_div = M.proxy_div;
    return;
  }
  if (s.locked) return;  // set by begin() from the locking viewer
  const float h = std::clamp(desc_.hysteresis, 0.f, 0.9f);
  const float px = s.px * desc_.texel_scale;
  const int cur = s.slot ? (s.next_slot ? s.next_slot->level : s.slot->level) : -1;
  if (cur < 0) {
    s.want_level = level_for(M, px);
  } else {
    const int up = level_for(M, px / (1.f + h)), down = level_for(M, px / (1.f - h));
    s.want_level = up > cur ? up : down < cur ? down : cur;
  }
  const std::array<float, 3> T = rate_px();
  if (!s.slot) {
    s.want_div = div_for(T, s.px);
  } else {
    const int faster = div_for(T, s.px / (1.f + h)), slower = div_for(T, s.px / (1.f - h));
    s.want_div = faster < s.div ? faster : slower > s.div ? slower : s.div;
  }
}

float Manager::compose_ms(const Model& M, int size) const {
  const int L = std::max(0, level_of_size(size));
  if (M.interp == NVFX_INTERP_HOLD) return 0.f;
  return M.interp == NVFX_INTERP_MOTION ? motion_ms_[z(L)] : fade_ms_[z(L)];
}

// Mean milliseconds per tick of a simulation at a level: its steps and keyframe pictures per second, and the pictures
// composed between keyframes, at the game's tick rate.
float Manager::cost(const Model& M, int level, int div) const {
  const float hz = static_cast<float>(hz_);
  const float steps = M.fps * M.step_factor / hz, pictures = M.fps / static_cast<float>(div) * M.render_factor / hz;
  const float between = div > 1 ? std::min(M.fps, hz) / hz * compose_ms(M, kSizes[z(level)]) : 0.f;
  return steps * M.step_ms[z(level)] + pictures * M.render_ms[z(level)] + between;
}

void Manager::plan() {
  order_.clear();
  float total = 0.f;
  for (Model& M : models_) std::fill(M.planned.begin(), M.planned.end(), std::uint8_t{0});
  for (Model& M : models_) {
    for (Sim& s : M.sims) {
      s.plan_proxy = false;
      if (!s.used || s.paused || (s.refs <= 0 && s.proxy_cell < 0)) {
        s.plan_level = s.slot ? s.slot->level : -1;
        s.plan_div = s.div;
        continue;
      }
      want(M, s);
      if (!s.slot) {
        s.plan_level = s.want_level;
        s.plan_div = s.want_div;
      } else {
        const int cur = s.next_slot ? s.next_slot->level : s.slot->level;
        s.plan_level = std::min(cur, s.want_level);  // down at once; up later, within the budget (below)
        s.plan_div = std::max(s.div, s.want_div);
      }
      if (s.locked) {
        s.plan_level = s.want_level;
        s.plan_div = s.want_div;
      }
      s.cost = cost(M, s.plan_level, s.plan_div);
      total += s.cost;
      if (s.proxy_cell >= 0 && z(s.proxy_cell) < M.planned.size()) M.planned[z(s.proxy_cell)] = 1;
      order_.push_back(&s);
    }
  }
  for (const Inst& in : insts_) {  // pictures composed for offset copies, proxies and fades
    if (in.used && in.alive && in.visible && (in.offset > 0 || in.proxied || in.fade > 0.f)) total += compose_ms(models_[z(in.model)], desc_.copy_size);
  }
  const float budget = desc_.budget_ms;
  float proxied_cut = 0.f;
  const auto by_importance = [](const Sim* a, const Sim* b) { return a->importance != b->importance ? a->importance < b->importance : a->serial < b->serial; };
  if (total > budget) {
    std::sort(order_.begin(), order_.end(), by_importance);
    for (Sim* sp : order_) {
      if (total <= budget) break;
      Sim& s = *sp;
      if (s.locked || s.proxy_cell >= 0) continue;
      Model& M = models_[z(s.model)];
      // At the proxies' size or smaller a lower rate saves little (the steps cost the same at every rate): such a
      // simulation goes to a proxy at once if it can (docs/MANAGER.md §6); larger ones lower their rate, then size.
      const bool small = s.plan_level <= M.proxy_level;
      while (total > budget) {
        float next = 0.f;
        const bool try_proxy = small && proxy_ok(M, s);
        if (s.plan_div < kMaxDiv && !try_proxy) {
          next = cost(M, s.plan_level, s.plan_div + 1);
          ++s.plan_div;
        } else if (s.plan_level > M.min_level && !try_proxy) {
          int L = s.plan_level - 1;
          while (L > M.min_level && M.slots_n[z(L)] == 0) --L;
          if (M.slots_n[z(L)] == 0) break;
          next = cost(M, L, s.plan_div);
          s.plan_level = L;
        } else {
          // a proxy, if every viewer allows it and has kept its source long enough
          if (!proxy_ok(M, s)) break;
          int cell = proxy_cell(M, insts_[z(s.first_viewer)]);
          next = static_cast<float>(s.viewers) * compose_ms(M, desc_.copy_size);
          if (M.proxy_sim[z(cell)] < 0 && !M.planned[z(cell)]) {
            // a proxy of its own cell if the budget has room for one more, else the nearest one there is
            const float c = cost(M, M.proxy_level, M.proxy_div);
            const int near = nearest_proxy(M, cell);
            if (near >= 0 && total - s.cost + next + c > budget) {
              cell = near;
            } else {
              next += c;
              M.planned[z(cell)] = 1;
            }
          }
          s.proxy_target = cell;
          s.plan_proxy = true;
          proxied_cut = std::max(proxied_cut, s.importance);
          total += next - s.cost;
          s.cost = next;
          break;
        }
        total += next - s.cost;
        s.cost = next;
      }
    }
  } else if (total < 0.85f * budget) {
    std::sort(order_.begin(), order_.end(), [&](const Sim* a, const Sim* b) { return by_importance(b, a); });
    const float room = 0.85f * budget;
    for (Sim* sp : order_) {
      Sim& s = *sp;
      if (s.locked || s.proxy_cell >= 0 || !s.slot || now_ - s.last_up < static_cast<double>(desc_.min_dwell)) continue;
      Model& M = models_[z(s.model)];
      while (true) {
        int L = s.plan_level, d = s.plan_div;
        if (L < s.want_level) {
          ++L;
        } else if (d > s.want_div) {
          --d;
        } else {
          break;
        }
        const float next = cost(M, L, d);
        if (total + next - s.cost > room) break;
        total += next - s.cost;
        s.cost = next;
        s.plan_level = L;
        s.plan_div = d;
        break;  // one rung per tick and simulation
      }
    }
    // a proxy for the cell whose instances are furthest from their own controls (the most of them shown from another
    // cell's proxy), if there is room for one more
    for (Model& M : models_) std::fill(M.waiting.begin(), M.waiting.end(), 0);
    int best_model = -1, best_cell = -1, best_n = 0;
    for (Inst& in : insts_) {
      if (!in.used || !in.alive || !in.proxied || !in.visible || in.sim < 0) continue;
      Model& M = models_[z(in.model)];
      const int want_cell = proxy_cell(M, in);
      if (want_cell < 0 || want_cell == M.sims[z(in.sim)].proxy_cell || M.proxy_sim[z(want_cell)] >= 0) continue;
      const int n = ++M.waiting[z(want_cell)];
      if (n > best_n) {
        best_n = n;
        best_model = in.model;
        best_cell = want_cell;
      }
    }
    if (best_model >= 0) {
      Model& M = models_[z(best_model)];
      const float c = cost(M, M.proxy_level, M.proxy_div);
      if (total + c <= room && !M.free_proxy.empty()) {
        split_model_ = best_model;
        split_cell_ = best_cell;
        total += c;
      }
    }
    // instances shown from proxies return to simulations of their own, most important first
    order_inst_.clear();
    for (Inst& in : insts_) {
      if (in.used && in.alive && in.proxied && in.visible && in.own_wait < 0 && now_ - in.switched_at >= static_cast<double>(desc_.proxy_dwell)) {
        order_inst_.push_back(&in);
      }
    }
    std::sort(order_inst_.begin(), order_inst_.end(), [](const Inst* a, const Inst* b) { return a->importance > b->importance; });
    for (Inst* ip : order_inst_) {
      Inst& in = *ip;
      Model& M = models_[z(in.model)];
      const float c = cost(M, std::min(level_for(M, in.px * desc_.texel_scale), M.proxy_level), kMaxDiv);
      if (total + c > room) break;
      int i = in.share ? find_shared(M, in) : -1;
      if (i < 0) {
        i = new_sim(M, in.model);
        if (i < 0) continue;
        Sim& s = M.sims[z(i)];
        s.controls = in.controls;
        s.seed = in.seed;
        s.shareable = in.share;
        s.plan_level = std::min(level_for(M, in.px * desc_.texel_scale), M.proxy_level);
        s.plan_div = kMaxDiv;
        total += c;
      }
      M.sims[z(i)].refs++;
      in.own_wait = i;
    }
  }
  // still over: proxies with the fewest viewers merge into the nearest other proxy (their viewers fade across)
  if (total > budget) {
    order_.clear();
    for (Model& M : models_) {
      for (Sim& s : M.sims) {
        if (s.used && s.proxy_cell >= 0 && s.ready && !s.paused) order_.push_back(&s);
      }
    }
    std::sort(order_.begin(), order_.end(), [&](const Sim* a, const Sim* b) {
      if (a->viewers != b->viewers) return a->viewers < b->viewers;
      return by_importance(a, b);
    });
    for (Sim* sp : order_) {
      if (total <= budget) break;
      Sim& s = *sp;
      Model& M = models_[z(s.model)];
      M.planned[z(s.proxy_cell)] = 0;
      const int near = nearest_proxy(M, s.proxy_cell);
      if (near < 0 || near == s.proxy_cell) {
        M.planned[z(s.proxy_cell)] = 1;
        continue;
      }
      s.proxy_target = near;
      s.plan_proxy = true;  // apply() moves its viewers
      total -= s.cost;
      s.cost = 0.f;
    }
  }
  // a proxied instance that is locked to a level gets a simulation of its own whatever the budget
  for (Inst& in : insts_) {
    if (!in.used || !in.alive || !in.proxied || in.lock_level < 0 || in.own_wait >= 0) continue;
    Model& M = models_[z(in.model)];
    int i = in.share ? find_shared(M, in) : -1;
    if (i < 0) {
      if (M.free_sims.empty() && !evict(M)) continue;
      i = new_sim(M, in.model);
      if (i < 0) continue;
      Sim& w = M.sims[z(i)];
      w.controls = in.controls;
      w.seed = in.seed;
      w.shareable = in.share;
      w.plan_level = in.lock_level;
      w.plan_div = in.lock_div;
    }
    M.sims[z(i)].refs++;
    in.own_wait = i;
  }
  plan_total_ = total;
  // new instances below this importance start as proxies while the budget is short
  proxy_cut_ = total <= 0.85f * budget ? 0.f : std::max(0.99f * proxy_cut_, proxied_cut);
}

// The plan carried out: proxies taken and left, slots for new simulations and level changes, update rates.
void Manager::apply() {
  for (Model& M : models_) {
    const int mi = static_cast<int>(&M - models_.data());
    for (Sim& s : M.sims) {
      if (!s.used || !s.plan_proxy) continue;
      const int p = ensure_proxy(mi, s.proxy_target);
      if (p < 0) continue;
      Sim& P = M.sims[z(p)];
      P.importance = std::max(P.importance, s.importance);  // its pre-roll is as urgent as what waits for it
      if (!P.ready) continue;  // the proxy is pre-rolling: this simulation plays on meanwhile
      for (int v = s.first_viewer; v >= 0;) {
        Inst& in = insts_[z(v)];
        v = in.next_viewer;
        start_fade(in);
        detach(in);  // releases s with its last viewer
        attach(in, mi, p, proxy_offset(M, in, p));
        in.proxied = true;
        in.switched_at = now_;
      }
      if (s.used && s.proxy_cell >= 0 && s.refs <= 0) release_sim(M, s);  // a proxy merged into another
    }
  }
  // a new proxy for a cell whose instances were shown from another's: once it has pre-rolled, they move to it
  if (split_model_ >= 0) {
    Model& M = models_[z(split_model_)];
    const int p = ensure_proxy(split_model_, split_cell_);
    if (p >= 0 && M.sims[z(p)].ready) {
      for (Inst& in : insts_) {
        if (!in.used || !in.alive || !in.proxied || in.model != split_model_ || in.sim < 0 || in.sim == p) continue;
        if (proxy_cell(M, in) != split_cell_ || now_ - in.switched_at < static_cast<double>(desc_.proxy_dwell)) continue;
        start_fade(in);
        detach(in);
        attach(in, split_model_, p, proxy_offset(M, in, p));
        in.switched_at = now_;
      }
    }
    split_model_ = split_cell_ = -1;
  }
  // instances whose own simulation has its first picture leave their proxy
  for (Inst& in : insts_) {
    if (!in.used || !in.alive || in.own_wait < 0) continue;
    Model& M = models_[z(in.model)];
    Sim& w = M.sims[z(in.own_wait)];
    if (!w.ready) continue;
    const int target = in.own_wait;
    start_fade(in);
    in.own_wait = -1;
    w.refs--;  // the waiting reference becomes a viewer's
    detach(in);
    attach(in, in.model, target, 0);
    in.proxied = false;
    in.switched_at = now_;
  }
  for (Model& M : models_) {
    for (Sim& s : M.sims) {
      if (!s.used) continue;
      if (s.proxy_cell >= 0) {  // proxies keep their level
        s.div = M.proxy_div;
        continue;
      }
      if (s.paused) {
        // parked: a paused simulation drops to the smallest size, freeing its slot
        if (s.slot && s.started && !s.next_slot && s.paused_s >= static_cast<double>(desc_.park_after) && s.slot->level > M.min_level &&
            !M.free[z(M.min_level)].empty()) {
          s.next_slot = M.free[z(M.min_level)].back();
          M.free[z(M.min_level)].pop_back();
          s.next_slot->used = true;
          s.parked = true;
        }
        continue;
      }
      if (s.refs <= 0 && s.proxy_cell < 0) continue;
      int L = std::max(s.plan_level, M.min_level);
      if (!s.slot) {
        for (int q = L; q >= M.min_level && !s.slot; --q) {
          if (!M.free[z(q)].empty()) {
            s.slot = M.free[z(q)].back();
            M.free[z(q)].pop_back();
            s.slot->used = true;
            for (Track& t : s.slot->t) t.shard = -1;
          }
        }
        s.div = s.plan_div;
        continue;
      }
      const int cur = s.next_slot ? s.next_slot->level : s.slot->level;
      if (L != cur && !s.next_slot) {
        Slot* to = nullptr;
        if (L > cur) {  // up: the largest free size up to L
          for (int q = L; q > cur && !to; --q) {
            if (!M.free[z(q)].empty()) to = M.free[z(q)].back(), M.free[z(q)].pop_back();
          }
          if (to) s.last_up = now_;
        } else {  // down: the largest free size from L down
          for (int q = L; q >= M.min_level && !to; --q) {
            if (!M.free[z(q)].empty()) to = M.free[z(q)].back(), M.free[z(q)].pop_back();
          }
        }
        if (to) {
          to->used = true;
          s.next_slot = to;
        }
      }
      if (s.plan_div < s.div) s.last_up = now_;
      s.div = s.plan_div;
      s.parked = false;
    }
  }
}

// Clocks, which updates are due, their cost, which run now, and the jobs.
void Manager::schedule(double dt) {
  jobs_.clear();
  float compose_only = 0.f;
  order_.clear();
  for (Model& M : models_) {
    const double fps = static_cast<double>(M.fps);
    for (Sim& s : M.sims) {
      s.run = s.compose = s.due = s.updated = s.held = s.composed = false;
      s.steps = s.renders = 0;
      s.transfer_job = s.next_slot != nullptr;
      if (!s.used || !s.slot) continue;
      if (s.paused) {
        if (s.transfer_job) jobs_.push_back(&s);
        continue;
      }
      if (s.ready) {
        s.clock += dt;
        auto f = static_cast<std::int64_t>(std::floor(s.clock * fps + 1e-9));
        if (f > s.fb) {  // behind (its updates waited): it holds its last keyframe and its clock waits too (keeping
          f = s.fb;      // its phase within a frame, which staggers the simulations' updates over the ticks)
          s.clock = static_cast<double>(s.fb) / fps + s.subframe;
        }
        const bool ended = !M.loop && M.frames > 0 && s.fb >= M.frames - 1;
        s.show = f;
        s.due = f >= s.fb && !ended;
      } else {
        s.due = true;
        s.show = s.fa;
      }
      int views = 0;  // viewers with a picture of their own to compose (a new frame for them, or a fade)
      for (int v = s.first_viewer; v >= 0; v = insts_[z(v)].next_viewer) {
        const Inst& in = insts_[z(v)];
        if (in.visible && (in.offset > 0 || in.proxied || in.fade > 0.f) && (in.fade > 0.f || s.show - in.offset != in.buf_frame)) ++views;
      }
      const int L = s.slot->level;
      const bool moved = s.show != s.shown;
      s.compose = moved || s.transfer_job || views > 0;
      float c = 0.f;
      if (s.due) {
        const float steps = s.ready ? static_cast<float>(s.div) * M.step_factor : std::min(desc_.catchup_frames, static_cast<float>(M.re->m.h.warmup + s.div));
        c = steps * M.step_ms[z(L)] + M.render_factor * M.render_ms[z(L)] * (s.started ? 1.f : 2.f);
      }
      const float comp = (moved && s.div > 1 ? compose_ms(M, kSizes[z(L)]) : 0.f) + static_cast<float>(views) * compose_ms(M, desc_.copy_size);
      s.cost = c + comp;
      s.compose_cost = comp;
      if (s.due) {
        order_.push_back(&s);
      } else if (s.compose) {
        compose_only += comp;
      }
    }
  }
  // admission: due updates in order of waiting, then importance, within burst x budget
  std::sort(order_.begin(), order_.end(), [](const Sim* a, const Sim* b) {
    const bool pa = a->proxy_cell >= 0 && !a->ready, pb = b->proxy_cell >= 0 && !b->ready;  // proxies' pre-rolls first:
    if (pa != pb) return pa;                                                                 // others wait for them
    if (a->deferred != b->deferred) return a->deferred > b->deferred;
    if (a->importance != b->importance) return a->importance > b->importance;
    return a->serial < b->serial;
  });
  const float allow = std::max(0.f, desc_.burst * desc_.budget_ms - compose_only);
  float used = 0.f;
  for (Sim* sp : order_) {
    Sim& s = *sp;
    if (used == 0.f || used + s.cost <= allow) {
      s.run = true;
      s.deferred = 0;
      used += s.cost;
    } else {
      s.deferred++;
      s.held = s.ready;
      stats_.skipped++;
      s.cost = s.compose_cost;  // it composes only
      compose_only += s.compose_cost;
    }
  }
  planned_ = used + compose_only;
  for (Model& M : models_) {
    for (Sim& s : M.sims) {
      if (s.used && s.slot && !s.paused && (s.run || s.compose || s.transfer_job)) jobs_.push_back(&s);
    }
  }
  std::sort(jobs_.begin(), jobs_.end(), [](const Sim* a, const Sim* b) { return a->cost != b->cost ? a->cost > b->cost : a->serial < b->serial; });
  n_jobs_ = static_cast<int>(jobs_.size());
}

nvfx_status Manager::begin(double dt, int* jobs) {
  if (in_tick_ || !std::isfinite(dt) || dt < 0.0) return NVFX_ERROR_ARGUMENT;
  in_tick_ = true;
  t_begin_ = now_ms();
  const double t0 = t_begin_;
  stats_ = nvfx_manager_stats{};
  dt_ = dt;
  now_ += dt;
  ++tick_;
  if (dt > 0.0) hz_ = 0.8 * hz_ + 0.2 * std::clamp(1.0 / dt, 5.0, 1000.0);
  // stopped instances leave
  for (std::size_t i = 0; i < insts_.size(); ++i) {
    Inst& in = insts_[i];
    if (in.used && !in.alive) {
      detach(in);
      in.used = false;
      free_insts_.push_back(static_cast<int>(i));
    }
  }
  for (Inst& in : insts_) {
    if (in.used) view(in);
  }
  // new controls: in place for a simulation of its own, a fork from a shared one, another proxy
  for (Inst& in : insts_) {
    if (!in.used || !in.controls_dirty) continue;
    Model& M = models_[z(in.model)];
    in.controls_dirty = false;
    if (in.own_wait >= 0) {
      Sim& w = M.sims[z(in.own_wait)];
      if (w.refs == 1) w.controls = in.controls;
    }
    if (in.sim < 0) continue;
    Sim& s = M.sims[z(in.sim)];
    if (in.proxied) {
      int cell = proxy_cell(M, in);
      if (cell >= 0 && M.proxy_sim[z(cell)] < 0 && plan_total_ + cost(M, M.proxy_level, M.proxy_div) > desc_.budget_ms) {
        if (const int near = nearest_proxy(M, cell); near >= 0) cell = near;  // no room for another proxy
      }
      if (cell >= 0 && cell != s.proxy_cell) {
        const int p = ensure_proxy(in.model, cell);
        if (p >= 0 && M.sims[z(p)].ready) {
          start_fade(in);
          detach(in);
          attach(in, in.model, p, proxy_offset(M, in, p));
          in.switched_at = now_;
        } else {
          in.controls_dirty = true;  // again next tick
        }
      }
    } else if (s.refs == 1) {
      s.controls = in.controls;
    } else if (!fork(in)) {
      in.controls_dirty = true;  // no simulation free: it stays in its group, and tries again next tick
    }
  }
  // new instances, most important first, join a simulation of the same effect, controls and seed; or get their own
  // while the plan (last tick's, plus what joins now) has room, or they matter more than what the planner last sent to
  // proxies; or else a proxy (of their own cell if there is room for one more, else the nearest there is)
  order_inst_.clear();
  for (Inst& in : insts_) {
    if (in.used && in.sim < 0 && !in.finished && (!in.culled || in.visible)) order_inst_.push_back(&in);
  }
  std::sort(order_inst_.begin(), order_inst_.end(), [&](const Inst* x, const Inst* y) {
    return x->importance != y->importance ? x->importance > y->importance : x < y;
  });
  float joined = plan_total_;
  for (Inst* ip : order_inst_) {
    Inst& in = *ip;
    Model& M = models_[z(in.model)];
    const int mi = in.model;
    in.culled = false;
    // an offset needs a ring (the effect's history) and a simulation to share: otherwise it is ignored
    const auto off = M.history > 0 && in.share ? static_cast<std::int64_t>(std::llround(static_cast<double>(in.offset_s) * static_cast<double>(M.fps))) : 0;
    int i = in.share ? find_shared(M, in) : -1;
    if (i >= 0) {
      attach(in, mi, i, off);
      continue;
    }
    const float own = in.visible ? cost(M, level_for(M, in.px * desc_.texel_scale), div_for(rate_px(), in.px)) : 0.f;
    const bool room = joined + own <= desc_.budget_ms || (proxy_cut_ > 0.f && in.importance > proxy_cut_) || !in.visible;
    int cell = in.allow_proxy && in.lock_level < 0 ? proxy_cell(M, in) : -1;
    if (cell >= 0 && (M.free_sims.empty() || !room)) {
      if (M.proxy_sim[z(cell)] < 0) {
        const float c = cost(M, M.proxy_level, M.proxy_div);
        const int near = nearest_proxy(M, cell);
        if (near >= 0 && joined + c > desc_.budget_ms) {
          cell = near;  // no room for another: the nearest there is
        } else {
          joined += c;
        }
      }
      const int p = ensure_proxy(mi, cell);
      if (p >= 0) {
        attach(in, mi, p, proxy_offset(M, in, p));
        in.proxied = true;
        in.switched_at = now_;
        continue;
      }
    }
    if (M.free_sims.empty() && !evict(M)) continue;  // none free: it waits
    i = new_sim(M, mi);
    if (i < 0) continue;
    Sim& s = M.sims[z(i)];
    s.controls = in.controls;
    s.seed = in.seed;
    s.shareable = in.share;
    attach(in, mi, i, off);
    joined += own;
  }
  // simulations: who views them, how large, whether they pause
  for (Model& M : models_) {
    for (Sim& s : M.sims) {
      s.viewers = s.visible = 0;
      s.first_viewer = -1;
      s.px = s.importance = 0.f;
      s.locked = false;
      s.all_proxy = true;
      s.max_offset = 0;
    }
  }
  for (std::size_t i = insts_.size(); i-- > 0;) {
    Inst& in = insts_[i];
    in.next_viewer = -1;
    if (!in.used) continue;
    if (in.fade > 0.f) in.fade = std::max(0.f, in.fade - static_cast<float>(dt));
    Model& M = models_[z(in.model)];
    for (const int si : {in.sim, in.own_wait}) {
      if (si < 0) continue;
      Sim& s = M.sims[z(si)];
      if (si == in.sim) {
        s.viewers++;
        in.next_viewer = s.first_viewer;
        s.first_viewer = static_cast<int>(i);
        s.max_offset = std::max(s.max_offset, in.offset);
        s.all_proxy = s.all_proxy && in.allow_proxy && in.lock_level < 0;
        if (in.lock_level >= 0 && !s.locked) {
          s.locked = true;
          s.want_level = in.lock_level;
          s.want_div = in.lock_div;
        }
      }
      if (in.visible) {
        s.visible++;
        s.px = std::max(s.px, in.px);
        s.importance = std::max(s.importance, in.importance);
      }
    }
    in.use_buf = in.offset > 0 || in.proxied || in.fade > 0.f;
  }
  for (Model& M : models_) {
    for (Sim& s : M.sims) {
      if (!s.used) continue;
      if (s.max_offset > 0 && !s.ring && !M.free_rings.empty()) {  // copies offset in time: a ring from the pool
        s.ring = M.free_rings.back();
        M.free_rings.pop_back();
        s.ring->head = s.ring->count = 0;
        if (!s.started) s.preroll = std::max(s.preroll, std::min<std::int64_t>(s.max_offset, static_cast<std::int64_t>(s.ring->cap - 2) * s.div) + s.div);
      }
      if (s.proxy_cell >= 0) {  // proxies without viewers linger, then go
        s.idle_s = s.refs > 0 ? 0.0 : s.idle_s + dt;
        if (s.refs <= 0 && s.idle_s >= static_cast<double>(desc_.proxy_dwell)) {
          release_sim(M, s);
          continue;
        }
        s.paused = s.refs > 0 && s.visible == 0 && s.ready;
        if (s.refs <= 0) s.importance = 0.f;
      } else {
        s.paused = s.visible == 0;
      }
      s.paused_s = s.paused ? s.paused_s + dt : 0.0;
      if (s.paused && desc_.cull_after > 0.f && s.paused_s >= static_cast<double>(desc_.cull_after) && s.proxy_cell < 0) {
        for (int v = s.first_viewer; v >= 0;) {  // culled: released; viewers start afresh when seen again (one-shots end)
          Inst& in = insts_[z(v)];
          v = in.next_viewer;
          detach(in);
          in.culled = true;
          in.finished = !M.loop;
          in.pic = nullptr;
        }
        if (s.used && s.refs <= 0) release_sim(M, s);
      }
    }
  }
  plan();
  apply();
  schedule(dt);
  stats_.plan_ms = static_cast<double>(plan_total_);
  stats_.planned_ms = static_cast<double>(planned_);
  next_job_.store(0);
  for (Work& w : work_) w.ms = 0.0;
  if (jobs) *jobs = n_jobs_;
  plan_wall_ = now_ms() - t0;
  return NVFX_OK;
}

void Manager::run_jobs(int slot) {
  Work& w = work_[z(slot)];
  for (int j; (j = next_job_.fetch_add(1)) < n_jobs_;) {
    const double t0 = now_ms();
    try {
      process(*jobs_[z(j)], w);
    } catch (...) {  // nothing in a job throws (every buffer exists); never across threads
    }
    w.ms += now_ms() - t0;
  }
}

nvfx_status Manager::tick(double dt) {
  int n = 0;
  if (const nvfx_status st = begin(dt, &n); st != NVFX_OK) return st;
  if (!threads_.empty() && n > 1) {
    active_.store(static_cast<int>(threads_.size()));
    epoch_.fetch_add(1);
    epoch_.notify_all();
    run_jobs(0);
    for (int a; (a = active_.load()) != 0;) active_.wait(a);
  } else {
    run_jobs(0);
  }
  return end();
}

void Manager::worker(int index) {
  std::uint32_t seen = epoch_.load();
  for (;;) {
    epoch_.wait(seen);
    const std::uint32_t e = epoch_.load();
    if (stop_.load()) return;
    if (e == seen) continue;
    seen = e;
    run_jobs(index);
    if (active_.fetch_sub(1) == 1) active_.notify_all();
  }
}

nvfx_status Manager::end() {
  if (!in_tick_) return NVFX_ERROR_ARGUMENT;
  const double t_end0 = now_ms();
  for (Model& M : models_) {
    for (Sim& s : M.sims) {
      free_slot(M, s.old_slot);
    }
  }
  nvfx_manager_stats& st = stats_;
  for (Inst& in : insts_) {
    if (!in.used || !in.alive) continue;
    st.instances++;
    const Model& M = models_[z(in.model)];
    const Sim* s = in.sim >= 0 ? &M.sims[z(in.sim)] : nullptr;
    const std::uint8_t* before = in.pic;
    int flags = 0;
    if (s && s->started && s->ready) {
      if (in.use_buf) {
        in.pic = in.buf.data();
        in.pic_size = desc_.copy_size;
        if (in.buf_interp) flags |= NVFX_PICTURE_INTERPOLATED;
        if (in.buf_new) flags |= NVFX_PICTURE_NEW;
      } else if (s->pic) {
        in.pic = s->pic;
        in.pic_size = s->pic_size;
        if (s->pic_interp) flags |= NVFX_PICTURE_INTERPOLATED;
        if (s->composed) flags |= NVFX_PICTURE_NEW;
      }
      in.pic_time = static_cast<double>(s->show - in.offset) / static_cast<double>(M.fps);
      if (s->refs > 1) flags |= NVFX_PICTURE_SHARED;
      if (s->held || s->paused) flags |= NVFX_PICTURE_HELD;
      if (!M.loop && M.frames > 0 && s->show - in.offset >= M.frames - 1) in.finished = true;
    }
    if (before != in.pic) flags |= NVFX_PICTURE_NEW;
    if (in.proxied) flags |= NVFX_PICTURE_PROXY;
    if (in.fade > 0.f) flags |= NVFX_PICTURE_FADING;
    if (in.finished) flags |= NVFX_PICTURE_FINISHED;
    in.pic_flags = flags;
    in.buf_new = false;  // (buf_interp stays: the picture is still the one composed)
    if (in.visible) st.visible++;
    if (in.finished) st.finished++;
    if (in.proxied) st.proxied++;
    if ((flags & NVFX_PICTURE_SHARED) && !in.proxied) st.shared_viewers++;
    if (in.visible && (flags & NVFX_PICTURE_INTERPOLATED) && (flags & NVFX_PICTURE_NEW)) st.interpolated++;
  }
  for (const Model& M : models_) {
    for (const Sim& s : M.sims) {
      if (!s.used) continue;
      st.sims++;
      if (s.proxy_cell >= 0) st.proxy_sims++;
      if (s.paused) {
        st.sims_paused++;
        continue;
      }
      if (s.slot) st.lod[s.slot->level][std::clamp(s.div, 1, kMaxDiv) - 1]++;
      st.updated += s.updated ? 1 : 0;
      st.steps += s.steps;
      st.renders += s.renders;
      if (s.held && s.visible > 0) st.held++;
    }
  }
  double jobs_ms = 0.0;
  for (const Work& w : work_) jobs_ms += w.ms;
  std::size_t mem = memory_;
  for (const Work& w : work_) mem += w.scratch->bytes() + w.uv.size() * 4;
  st.jobs_ms = jobs_ms;
  st.budget_ms = static_cast<double>(desc_.budget_ms);
  st.memory_bytes = mem;
  st.threads = desc_.threads > 0 ? desc_.threads : desc_.job_slots;
  st.plan_ms = static_cast<double>(plan_total_);
  st.planned_ms = static_cast<double>(planned_);
  const double t_end = now_ms();
  st.wall_ms = t_end - t_begin_;
  st.overhead_ms = plan_wall_ + (t_end - t_end0);
  in_tick_ = false;
  return NVFX_OK;
}

nvfx_status Manager::picture(nvfx_handle h, nvfx_manager_picture* out) const {
  const Inst* in = find(h);
  if (!in || !out) return NVFX_ERROR_ARGUMENT;
  *out = nvfx_manager_picture{};
  const Model& M = models_[z(in->model)];
  out->rgba = in->pic;
  out->size = in->pic ? in->pic_size : 0;
  out->stride = z(out->size) * 4;
  out->flags = in->pic_flags;
  out->effect_time = in->pic_time;
  out->visible = in->visible ? 1 : 0;
  out->screen_px = in->px;
  if (in->sim >= 0) {
    const Sim& s = M.sims[z(in->sim)];
    if (s.slot) out->lod_size = s.slot->size;
    out->lod_rate_hz = s.paused ? 0.f : M.fps / static_cast<float>(s.div);
  }
  return NVFX_OK;
}

// --- jobs --------------------------------------------------------------------------------------------------------------------

void Manager::process(Sim& s, Work& w) {
  Model& M = models_[z(s.model)];
  if (s.next_slot) transfer(M, s);
  if (s.run) update(M, s, w);
  if (s.started && s.ready && (s.compose || s.updated || s.shown != s.show)) compose_main(M, s, w);
  for (int v = s.first_viewer; v >= 0; v = insts_[z(v)].next_viewer) {
    Inst& in = insts_[z(v)];
    if (in.use_buf && s.ready && (in.visible || in.fade > 0.f)) compose_view(M, s, in, w);
  }
}

// A level change: the state moves to the new slot (the coarse state exactly, the fine fields resampled), and the two
// keyframes are resampled to its size.
void Manager::transfer(Model& M, Sim& s) {
  (void)M;
  Slot& from = *s.slot;
  Slot& to = *s.next_slot;
  for (std::size_t k = 0; k < 2; ++k) {
    to.t[k].shard = from.t[k].shard;
    to.t[k].frame = from.t[k].frame;
    to.t[k].seed = from.t[k].seed;
    if (from.t[k].shard >= 0) to.t[k].r->transfer_from(*from.t[k].r);
  }
  if (s.started) {
    for (std::size_t k = 0; k < 2; ++k) resample(from.key[k].data(), from.size, to.key[k].data(), to.size);
  }
  s.old_slot = &from;
  s.slot = &to;
  s.next_slot = nullptr;
  s.shown = -1;
}

// Shard timeline (src/runtime/nvfx.cpp's bring): track t to shard k at frame f, at most `limit` steps in all.
bool Manager::bring(Model& M, Sim& s, Track& t, bool shown, std::int64_t k, std::int64_t f, int limit, int& steps) {
  if (t.shard != k || f < t.frame) {
    const ShardPlan p = plan_shard(M, s, k);
    t.r->begin(p.index, p.seed);
    t.shard = k;
    t.frame = p.first;
    t.seed = p.seed;
  }
  const std::span<const float> ctl(s.controls.data(), z(M.nc));
  while (t.frame < f) {
    if (steps >= limit) return false;
    t.r->step(ctl, t.seed);
    ++t.frame;
    ++steps;
    ++s.steps;
    if (shown && t.frame > 0) {  // the displacement of the picture: the flow of the shard on screen
      const auto flow = t.r->flow();
      for (std::size_t i = 0; i < s.acc.size(); ++i) s.acc[i] += flow[i];
    }
  }
  return true;
}

// Both tracks to frame f as nvfx's rollout_frame brings them (the shard on screen, and the next one once it has begun).
bool Manager::advance(Model& M, Sim& s, Slot& L, std::int64_t f, int limit, int& steps) {
  if (!M.loop && M.frames > 0) f = std::min<std::int64_t>(f, M.frames - 1);
  f = std::max<std::int64_t>(f, 0);
  if (M.shard == 0) return bring(M, s, L.t[0], true, 0, f, limit, steps);
  const std::int64_t k = f / M.shard;
  Track& cur = L.t[1].shard == k ? L.t[1] : L.t[0];
  Track& next = &cur == &L.t[0] ? L.t[1] : L.t[0];
  if (!bring(M, s, cur, true, k, f, limit, steps)) return false;
  if (f >= plan_shard(M, s, k + 1).first) return bring(M, s, next, false, k + 1, f, limit, steps);
  return true;
}

// The picture of frame f from the tracks (as rollout_frame draws it: the shard on screen, crossfaded into the next).
void Manager::draw(Model& M, Sim& s, Slot& L, std::int64_t f, std::uint8_t* out) {
  (void)s;
  if (!M.loop && M.frames > 0) f = std::min<std::int64_t>(f, M.frames - 1);
  f = std::max<std::int64_t>(f, 0);
  const rt::FrameInput fi;
  const std::size_t row = z(L.size) * 4;
  if (M.shard == 0) {
    L.t[0].r->render(fi, out, row);
    return;
  }
  const std::int64_t k = f / M.shard, j = f - k * M.shard;
  Track& cur = L.t[1].shard == k ? L.t[1] : L.t[0];
  Track& next = &cur == &L.t[0] ? L.t[1] : L.t[0];
  cur.r->render(fi, out, row);
  if (j < M.shard - kBlend) return;
  next.r->render(fi, L.mix.data(), row);
  ++s.renders;
  const float w = (static_cast<float>(j - (M.shard - kBlend)) + 0.5f) / static_cast<float>(kBlend);
  const std::size_t n = row * z(L.size);
  for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<std::uint8_t>(static_cast<float>(out[i]) * (1.f - w) + static_cast<float>(L.mix[i]) * w + 0.5f);
}

void Manager::push_ring(Sim& s, const std::uint8_t* px, int size, std::int64_t frame, const float* d) {
  if (!s.ring) return;
  Ring& R = *s.ring;
  const int e = R.head;
  resample(px, size, R.px.data() + z(e) * z(R.size) * z(R.size) * 4, R.size);
  const std::size_t n = z(R.res) * z(R.res) * 2;
  float big = 0.f;
  for (std::size_t i = 0; i < n; ++i) big = std::max(big, std::abs(d[i]));
  const float scale = big > 0.f ? big / 127.f : 1.f;
  std::int8_t* q = R.d.data() + z(e) * n;
  for (std::size_t i = 0; i < n; ++i) q[i] = static_cast<std::int8_t>(std::lround(d[i] / scale));
  R.scale[z(e)] = scale;
  R.frame[z(e)] = frame;
  R.head = (R.head + 1) % R.cap;
  R.count = std::min(R.count + 1, R.cap);
}

// Keyframes up to the next one due: the first two (frame 0 and the first interval, or a pre-roll's worth), then one per
// update. A spawn's warm-up and a pre-roll spread over ticks (desc.catchup_frames steps each).
int Manager::update(Model& M, Sim& s, Work& w) {
  Slot& L = *s.slot;
  for (Track& t : L.t) t.r->use_scratch(w.scratch.get());
  const int limit = std::max(1, static_cast<int>(desc_.catchup_frames));
  int steps = 0;
  if (!s.started) {
    if (!advance(M, s, L, 0, limit, steps)) return steps;
    s.ka = 0;
    s.fa = s.fb = 0;
    draw(M, s, L, 0, L.key[0].data());
    ++s.renders;
    std::fill(s.acc.begin(), s.acc.end(), 0.f);
    std::fill(s.dab.begin(), s.dab.end(), 0.f);
    push_ring(s, L.key[0].data(), L.size, 0, s.dab.data());
    s.started = true;
    s.phase = 1 + static_cast<int>(mix64(s.serial ^ s.seed) % static_cast<std::uint64_t>(s.div));
    s.updated = true;
  }
  const bool one_shot = !M.loop && M.frames > 0;
  while (true) {
    std::int64_t target = s.fb == s.fa ? s.fa + s.phase : s.fb + s.div;
    if (s.ready) target = s.fb + static_cast<std::int64_t>(s.div) * std::max<std::int64_t>(1, (s.show + 1 - s.fb + s.div - 1) / s.div);
    if (one_shot) target = std::min<std::int64_t>(target, M.frames - 1);
    if (target <= s.fb) {  // a one-shot effect at its end
      if (!s.ready) {
        s.ready = true;
        s.clock = static_cast<double>(s.fa) / static_cast<double>(M.fps) + s.subframe;
      }
      break;
    }
    if (!advance(M, s, L, target, limit, steps)) break;
    if (s.fb > s.fa) {
      s.ka ^= 1;
      s.fa = s.fb;
    }
    draw(M, s, L, target, L.key[z(s.ka ^ 1)].data());
    ++s.renders;
    s.fb = target;
    std::copy(s.acc.begin(), s.acc.end(), s.dab.begin());
    std::fill(s.acc.begin(), s.acc.end(), 0.f);
    push_ring(s, L.key[z(s.ka ^ 1)].data(), L.size, target, s.dab.data());
    s.updated = true;
    if (s.ready) break;
    if (s.fb >= s.preroll) {
      s.ready = true;
      s.clock = static_cast<double>(s.fa) / static_cast<double>(M.fps) + s.subframe;
      s.show = s.fa;
      break;
    }
  }
  return steps;
}

void Manager::interpolate(Model& M, const std::uint8_t* a, const std::uint8_t* b, int size, float wt, const float* uv, std::uint8_t* out) const {
  switch (M.interp) {
    case NVFX_INTERP_MOTION:
      motion_blend(a, b, size, wt, Motion{uv, M.res, static_cast<float>(size) / static_cast<float>(M.res) * M.advect}, out);
      break;
    case NVFX_INTERP_CROSSFADE: crossfade(a, b, size, wt, out); break;
    default: std::memcpy(out, a, z(size) * z(size) * 4); break;
  }
}

// The picture of the simulation's display frame for its viewers without an offset: a keyframe, or between two.
void Manager::compose_main(Model& M, Sim& s, Work& w) {
  (void)w;
  Slot& L = *s.slot;
  const std::int64_t f = std::clamp(s.show, s.fa, s.fb);
  s.pic_size = L.size;
  s.pic_interp = false;
  s.composed = true;
  if (f >= s.fb && s.fb > s.fa) {
    s.pic = L.key[z(s.ka ^ 1)].data();
  } else if (f <= s.fa || s.fb <= s.fa || M.interp == NVFX_INTERP_HOLD) {
    s.pic = L.key[z(s.ka)].data();
  } else {
    const float wt = static_cast<float>(f - s.fa) / static_cast<float>(s.fb - s.fa);
    interpolate(M, L.key[z(s.ka)].data(), L.key[z(s.ka ^ 1)].data(), L.size, wt, s.dab.data(), L.out.data());
    s.pic = L.out.data();
    s.pic_interp = true;
  }
  s.shown = s.show;
}

// The ring's picture of frame f at copy size into out: 0 if the ring is empty, 1 a keyframe, 2 between two.
int Manager::ring_view(Model& M, Sim& s, std::int64_t f, std::uint8_t* out, Work& w) {
  if (!s.ring || s.ring->count == 0) return 0;
  const Ring& R = *s.ring;
  const std::size_t bytes = z(R.size) * z(R.size) * 4, n = z(R.res) * z(R.res) * 2;
  int k = -1;  // the newest entry at or before f
  for (int q = R.count - 1; q >= 0; --q) {
    if (R.frame[z(R.slot(q))] <= f) {
      k = q;
      break;
    }
  }
  if (k < 0 || k == R.count - 1) {
    std::memcpy(out, R.px.data() + z(R.slot(k < 0 ? 0 : k)) * bytes, bytes);
    return 1;
  }
  const int e1 = R.slot(k), e2 = R.slot(k + 1);
  const std::int64_t f1 = R.frame[z(e1)], f2 = R.frame[z(e2)];
  const float wt = static_cast<float>(f - f1) / static_cast<float>(f2 - f1);
  const std::uint8_t* a = R.px.data() + z(e1) * bytes;
  const std::uint8_t* b = R.px.data() + z(e2) * bytes;
  if (wt <= 0.f || M.interp == NVFX_INTERP_HOLD) {
    std::memcpy(out, a, bytes);
    return 1;
  }
  const std::int8_t* q = R.d.data() + z(e2) * n;
  const float scale = R.scale[z(e2)];
  for (std::size_t i = 0; i < n; ++i) w.uv[i] = static_cast<float>(q[i]) * scale;
  interpolate(M, a, b, R.size, wt, w.uv.data(), out);
  return 2;
}

// The picture of a viewer with its own buffer: a copy offset in time or a proxy (from the ring), or a fade from its last
// source (the snapshot) into its new one.
void Manager::compose_view(Model& M, Sim& s, Inst& in, Work& w) {
  const int C = desc_.copy_size;
  const std::int64_t f = s.show - in.offset;
  if (f == in.buf_frame && in.fade <= 0.f) return;  // the same frame as last time (a 60 Hz game: every other tick)
  bool interp = false;
  if (in.offset > 0 || in.proxied) {
    const int r = ring_view(M, s, f, in.buf.data(), w);
    if (r == 0) {
      if (!s.pic) return;
      resample(s.pic, s.pic_size, in.buf.data(), C);
    }
    interp = r == 2;
  } else {
    if (!s.pic) return;
    resample(s.pic, s.pic_size, in.buf.data(), C);
    interp = s.pic_interp;
  }
  if (in.fade > 0.f) {
    const float total = static_cast<float>(desc_.fade_frames) / M.fps;
    crossfade(in.snap.data(), in.buf.data(), C, 1.f - in.fade / total, in.buf.data());
  }
  in.buf_new = true;
  in.buf_interp = interp;
  in.buf_frame = in.fade > 0.f ? -1 : f;
}

}  // namespace nfx::mgr
