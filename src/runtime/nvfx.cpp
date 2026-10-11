// The C API (include/neuralfx/nvfx.h): effects, instances, time and variation logic, ISA dispatch, baking.
// Exceptions never cross the C boundary: every entry point returns a status.
#include <neuralfx/noise.hpp>
#include <neuralfx/nvfx.h>

#include "nvfx_internal.hpp"
#include "rt_common.hpp"
#include "rt_prior.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <new>
#include <numbers>
#include <spanstream>
#include <vector>

using nfx::rt::Effect;
using nfx::rt::FrameInput;
using nfx::rt::Renderer;
using nfx::rt::RolloutEffect;
using nfx::rt::RolloutRunner;

// Either kind of effect: a frame model (model.hpp) or a rollout effect (rollout.hpp), told apart by the file's magic.
struct nvfx_effect {
  Effect e;
  std::unique_ptr<RolloutEffect> roll;
  std::unique_ptr<nfx::rt::PriorNet> prior;  // rollout effects: the optional prior against drift (nvfx_effect_attach_prior)
};

struct nvfx_instance {
  const nvfx_effect* effect = nullptr;
  int size = 0;
  std::unique_ptr<Renderer> renderer;
  // Rollout effects: two runners (the shard on screen and the next one, rolled ahead for the crossfade) and a frame
  // for the blend. A runner's state belongs to `shard` at frame `frame` of the effect's timeline.
  struct Track {
    std::unique_ptr<RolloutRunner> r;
    std::int64_t shard = -1, frame = 0;
    std::uint64_t seed = 0;
  };
  Track a, b;  // a.r is non-null for rollout effects
  std::vector<std::uint8_t> mix;
  std::vector<float> controls, cond, za, zb;
  std::uint64_t seed = 0;
  int variation = -1;
  float drift_seconds = 0.f;
  float hue = 0.f, brightness = 1.f;
  std::array<float, 9> colour{};
  bool apply_colour = false;
  // The prior against drift (docs/DCM.md G2.13): its work buffers, its condition, and when and how strongly it acts.
  std::unique_ptr<nfx::rt::Prior> prior;
  std::vector<float> prior_cond;
  int prior_every = 16, prior_t = 100;
  float prior_beta = 1.f;
};

const RolloutEffect* nfx::rt::rollout_of(const nvfx_effect* e) { return e ? e->roll.get() : nullptr; }

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
  std::size_t n = m.raw_f16.size() * 2 + m.raw_u8.size() + m.raw_ranges.size() * 4 + m.raw_codebook.size() * 4;
  n += (m.basis.w.size() + m.basis.b.size()) * 4;
  for (const auto* group : {&m.layers, &m.films}) {
    for (const auto& d : *group) n += (d.w.size() + d.b.size()) * 4;
  }
  n += (m.z_mean.size() + m.z_std.size()) * 4;
  for (const auto& z : m.z_train) n += z.size() * 4;
  return n;
}

std::size_t rollout_resident(const nfx::rollout::Model& m) {
  std::size_t n = (m.step_w.size() + m.render_w.size()) * 4;
  for (const auto& s : m.starts) n += (s.controls.size() + s.coarse.size() + s.fine_t.size() + s.fine_d.size()) * 4;
  return n;
}

nvfx_status adopt_rollout(std::expected<nfx::rollout::Model, std::string>&& m, nvfx_effect** out) {
  if (!m) return m.error().find("cannot open") != std::string::npos ? NVFX_ERROR_IO : NVFX_ERROR_FORMAT;
  auto* e = new (std::nothrow) nvfx_effect;
  if (!e) return NVFX_ERROR_MEMORY;
  e->roll = std::make_unique<RolloutEffect>();
  e->roll->m = std::move(*m);
  e->roll->stored_bytes = e->roll->m.storage_bytes();
  e->roll->resident_bytes = rollout_resident(e->roll->m);
  *out = e;
  return NVFX_OK;
}

// Rollout timeline (the owner's design: inference sharded from start points). A looping effect is cut into shards of
// drift_seconds (default 6 s). Shard k is a fresh rollout from a start point chosen by seed, shard and the nearest
// controls, with a seed of its own; it is rolled ahead of its turn (its warm-up, if its start point has no fine
// fields) and crossfades in over the last kBlend frames of shard k - 1. So drift never builds up past one shard, and
// the frame shown depends only on the time (and the controls). drift_seconds = 0: one continuous rollout (it drifts
// after 20 s or so; seeks then step from the start). One-shot effects are a single shard and hold their last frame.
constexpr std::int64_t kBlend = 15, kCatchUp = 60;

std::int64_t shard_frames(const nvfx_instance& in) {
  const auto& m = in.effect->roll->m;
  if (!m.loop || in.drift_seconds <= 0.f) return 0;
  return std::max<std::int64_t>(4 * kBlend, static_cast<std::int64_t>(std::lround(in.drift_seconds * m.fps)));
}

// The start point for (seed, shard): one of the three whose controls are nearest the instance's.
int pick_start(const nvfx_instance& in, std::int64_t shard) {
  const auto& m = in.effect->roll->m;
  if (in.variation >= 0 && shard == 0) return in.variation;
  std::array<int, 3> best{-1, -1, -1};
  std::array<float, 3> dist{1e30f, 1e30f, 1e30f};
  for (std::size_t i = 0; i < m.starts.size(); ++i) {
    float d = 0.f;
    for (std::size_t k = 0; k < m.starts[i].controls.size() && k < in.controls.size(); ++k) {
      const float e = m.starts[i].controls[k] - in.controls[k];
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
  const std::uint32_t h = nfx::hash_cell(static_cast<std::int32_t>(shard), static_cast<std::int32_t>(shard >> 31), 17, in.seed);
  return best[h % static_cast<std::uint32_t>(count)];
}

struct ShardPlan {
  int index = 0;
  std::uint64_t seed = 0;
  std::int64_t first = 0;  // frame of the timeline at which the shard's rollout begins (before it is shown)
};

ShardPlan plan_shard(const nvfx_instance& in, std::int64_t k) {
  const auto& m = in.effect->roll->m;
  ShardPlan p;
  p.index = pick_start(in, k);
  const bool replay = in.variation >= 0 && k == 0;  // a replayed start point keeps its run's seed (it tracks that run)
  p.seed = replay ? m.starts[static_cast<std::size_t>(p.index)].seed
                  : (k == 0 ? in.seed : in.seed ^ (0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(k + 1)));
  const std::int64_t pre = m.starts[static_cast<std::size_t>(p.index)].fine_t.empty() ? m.h.warmup : 0;
  const std::int64_t L = shard_frames(in);
  p.first = (L == 0 || k == 0) ? -pre : k * L - kBlend - pre;
  return p;
}

// The prior against drift (docs/DCM.md G2.13), after the step that reached t.frame: in one continuous rollout only,
// every prior_every frames counted from the end of the start point's warm-up, as study G applied it (G2b). No allocation.
void apply_prior(nvfx_instance& in, nvfx_instance::Track& t) {
  if (!in.prior || in.prior_every <= 0 || t.frame <= 0 || t.frame % in.prior_every != 0 || shard_frames(in) != 0) return;
  const auto& m = in.effect->roll->m;
  nfx::rollout::condition(m, in.controls, t.r->time(), in.prior_cond);
  in.prior->apply(t.r->coarse_mut(), m.h.channels(), m.lo, m.hi, in.prior_t, in.prior_beta, in.prior_cond);
}

// Bring a track to shard k at frame f: continue it, or begin the shard afresh and step to f. No allocation.
void bring(nvfx_instance& in, nvfx_instance::Track& t, std::int64_t k, std::int64_t f) {
  if (t.shard != k || f < t.frame || f - t.frame > kCatchUp) {
    const ShardPlan p = plan_shard(in, k);
    t.r->begin(p.index, p.seed);
    t.shard = k;
    t.frame = p.first;
    t.seed = p.seed;
  }
  while (t.frame < f) {
    t.r->step(in.controls, t.seed);
    ++t.frame;
    apply_prior(in, t);
  }
}

// The frame at f (frames since the effect began) into rgba.
void rollout_frame(nvfx_instance& in, std::int64_t f, const FrameInput& fi, std::uint8_t* rgba, std::size_t stride) {
  const auto& m = in.effect->roll->m;
  if (!m.loop && m.h.frames > 0) f = std::min<std::int64_t>(f, m.h.frames - 1);  // one-shot: hold the last frame
  f = std::max<std::int64_t>(f, 0);
  const std::int64_t L = shard_frames(in);
  if (L == 0) {
    bring(in, in.a, 0, f);
    in.a.r->render(fi, rgba, stride);
    return;
  }
  const std::int64_t k = f / L, j = f - k * L;
  nvfx_instance::Track& cur = in.b.shard == k ? in.b : in.a;
  nvfx_instance::Track& next = &cur == &in.a ? in.b : in.a;
  bring(in, cur, k, f);
  if (f >= plan_shard(in, k + 1).first) bring(in, next, k + 1, f);  // rolled ahead of its turn
  cur.r->render(fi, rgba, stride);
  if (j < L - kBlend) return;
  // the crossfade: the next shard comes in over the last kBlend frames
  const std::size_t row = static_cast<std::size_t>(in.size) * 4;
  next.r->render(fi, in.mix.data(), row);
  const float w = (static_cast<float>(j - (L - kBlend)) + 0.5f) / static_cast<float>(kBlend);
  for (int y = 0; y < in.size; ++y) {
    std::uint8_t* d = rgba + stride * static_cast<std::size_t>(y);
    const std::uint8_t* s = in.mix.data() + row * static_cast<std::size_t>(y);
    for (std::size_t i = 0; i < row; ++i) d[i] = static_cast<std::uint8_t>(static_cast<float>(d[i]) * (1.f - w) + static_cast<float>(s[i]) * w + 0.5f);
  }
}

// The prior's work buffers for an instance, on the instance's ISA.
std::unique_ptr<nfx::rt::Prior> make_prior(const nfx::rt::PriorNet& n) {
  switch (resolved_isa()) {
    case NVFX_ISA_AVX512: return nfx::rt::isa_avx512::make_prior(n);
    case NVFX_ISA_AVX2: return nfx::rt::isa_avx2::make_prior(n);
    default: return nfx::rt::isa_base::make_prior(n);
  }
}

// A denoiser for a rollout effect: read, checked against the effect's grid and condition, and attached.
nvfx_status attach_prior(nvfx_effect* e, std::span<const char> bytes) {
  if (!e->roll) return NVFX_ERROR_ARGUMENT;
  if (e->prior) return NVFX_ERROR_ARGUMENT;  // instances may hold the attached one
  auto n = nfx::rt::parse_prior(bytes);
  if (!n) return NVFX_ERROR_FORMAT;
  const auto& h = e->roll->m.h;
  if (n->res != h.res || n->channels != nfx::rollout::kPhys || n->cond != h.cond()) return NVFX_ERROR_FORMAT;
  e->prior = std::make_unique<nfx::rt::PriorNet>(std::move(*n));
  e->roll->resident_bytes += e->prior->resident_bytes();
  return NVFX_OK;
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
  e->e.m.vq_codebook.clear();  // kept as raw_codebook
  e->e.m.vq_codebook.shrink_to_fit();
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
    case NVFX_ERROR_SCRIPT: return "error in the scene script";
  }
  return "unknown status";
}

nvfx_status nvfx_effect_load(const char* path, nvfx_effect** out) {
  if (!path || !out) return NVFX_ERROR_ARGUMENT;
  try {
    char magic[8] = {};
    {
      std::ifstream f(path, std::ios::binary);
      if (!f) return NVFX_ERROR_IO;
      f.read(magic, 8);
    }
    if (nfx::rollout::is_rollout_file(std::span<const char>(magic, 8))) return adopt_rollout(nfx::rollout::load_model(std::filesystem::path(path)), out);
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
    if (nfx::rollout::is_rollout_file(std::span(static_cast<const char*>(data), bytes))) return adopt_rollout(nfx::rollout::load_model(in), out);
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
  *info = {};
  if (e->roll) {
    const auto& m = e->roll->m;
    info->arch = 3;
    info->native_size = 128;
    info->frames = m.h.frames;
    info->fps = m.fps;
    info->loops = m.loop ? 1 : 0;
    info->n_controls = m.h.n_controls;
    info->n_variations = static_cast<int>(m.starts.size());
    info->stored_bytes = e->roll->stored_bytes;
    info->resident_bytes = e->roll->resident_bytes;
    std::strncpy(info->name, m.effect.c_str(), sizeof(info->name) - 1);
    return NVFX_OK;
  }
  const nfx::Model& m = e->e.m;
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
  if (!e || i < 0) return nullptr;
  const auto& names = e->roll ? e->roll->m.control_names : e->e.m.control_names;
  if (static_cast<std::size_t>(i) >= names.size()) return nullptr;
  return names[static_cast<std::size_t>(i)].c_str();
}

nvfx_status nvfx_instance_create(const nvfx_effect* e, int size, nvfx_instance** out) {
  if (!e || !out) return NVFX_ERROR_ARGUMENT;
  if (e->roll) {
    const auto& h = e->roll->m.h;
    if (size < h.res || size > 1024 || size % h.res) return NVFX_ERROR_UNSUPPORTED;
    try {
      auto in = std::make_unique<nvfx_instance>();
      in->effect = e;
      in->size = size;
      in->controls.assign(static_cast<std::size_t>(h.n_controls), 0.5f);
      for (auto* t : {&in->a, &in->b}) {
        switch (resolved_isa()) {
          case NVFX_ISA_AVX512: t->r = nfx::rt::isa_avx512::make_rollout(*e->roll, size); break;
          case NVFX_ISA_AVX2: t->r = nfx::rt::isa_avx2::make_rollout(*e->roll, size); break;
          default: t->r = nfx::rt::isa_base::make_rollout(*e->roll, size); break;
        }
      }
      in->mix.assign(static_cast<std::size_t>(size) * size * 4, 0);
      in->drift_seconds = e->roll->m.loop ? 6.f : 0.f;
      in->prior_cond.assign(static_cast<std::size_t>(h.cond()), 0.f);
      if (e->prior) in->prior = make_prior(*e->prior);
      *out = in.release();
      return NVFX_OK;
    } catch (const std::bad_alloc&) {
      return NVFX_ERROR_MEMORY;
    } catch (...) {
      return NVFX_ERROR_ARGUMENT;
    }
  }
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
  if (in->a.r) {
    return in->a.r->scratch_bytes() + in->b.r->scratch_bytes() + in->mix.size() + 4 * (in->controls.size() + in->prior_cond.size()) +
           (in->prior ? in->prior->scratch_bytes() : 0);
  }
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
  if (in->a.r) {
    if (index < -1 || index >= static_cast<int>(in->effect->roll->m.starts.size())) return NVFX_ERROR_ARGUMENT;
    if (index != in->variation) in->a.shard = in->b.shard = -1;  // restart from that start point at the next render
    in->variation = index;
    return NVFX_OK;
  }
  const nfx::Model& m = in->effect->e.m;
  const int n = m.h.n_latent > 0 ? static_cast<int>(m.z_train.size()) : 0;
  if (index < -1 || index >= n) return NVFX_ERROR_ARGUMENT;
  in->variation = index;
  return NVFX_OK;
}

nvfx_status nvfx_instance_set_drift(nvfx_instance* in, float seconds) {
  if (!in || seconds < 0.f) return NVFX_ERROR_ARGUMENT;
  if (in->a.r && seconds != in->drift_seconds) in->a.shard = in->b.shard = -1;  // rollout: a new shard length, a new timeline
  in->drift_seconds = seconds;
  return NVFX_OK;
}

nvfx_status nvfx_effect_attach_prior(nvfx_effect* e, const char* path) {
  if (!e || !path) return NVFX_ERROR_ARGUMENT;
  try {
    std::ifstream f(path, std::ios::binary);
    if (!f) return NVFX_ERROR_IO;
    const std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (f.bad()) return NVFX_ERROR_IO;
    return attach_prior(e, bytes);
  } catch (const std::bad_alloc&) {
    return NVFX_ERROR_MEMORY;
  } catch (...) {
    return NVFX_ERROR_FORMAT;
  }
}

nvfx_status nvfx_effect_attach_prior_memory(nvfx_effect* e, const void* data, size_t bytes) {
  if (!e || !data) return NVFX_ERROR_ARGUMENT;
  try {
    return attach_prior(e, std::span(static_cast<const char*>(data), bytes));
  } catch (const std::bad_alloc&) {
    return NVFX_ERROR_MEMORY;
  } catch (...) {
    return NVFX_ERROR_FORMAT;
  }
}

nvfx_status nvfx_instance_set_prior(nvfx_instance* in, int every_frames, int t, float beta) {
  if (!in || !in->a.r || every_frames < 0 || !(beta >= 0.f && beta <= 1.f)) return NVFX_ERROR_ARGUMENT;
  const nfx::rt::PriorNet* net = in->effect->prior.get();
  if (every_frames > 0 && (!net || t < 1 || t > net->timesteps)) return NVFX_ERROR_ARGUMENT;
  const bool was_on = in->prior && in->prior_every > 0, on = every_frames > 0;
  if (on && !in->prior) {  // the instance was created before the prior was attached: a set-up call allocates
    try {
      in->prior = make_prior(*net);
    } catch (const std::bad_alloc&) {
      return NVFX_ERROR_MEMORY;
    }
  }
  if (was_on != on || (on && (every_frames != in->prior_every || t != in->prior_t || beta != in->prior_beta))) {
    in->a.shard = in->b.shard = -1;  // another prior, another timeline: replay from the start at the next render
  }
  in->prior_every = every_frames;
  if (every_frames > 0) {
    in->prior_t = t;
    in->prior_beta = beta;
  }
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
  if (in->a.r) {
    const double frames = std::clamp(seconds * static_cast<double>(in->effect->roll->m.fps), -1e12, 1e12);
    FrameInput f;
    f.colour = in->colour;
    f.apply_colour = in->apply_colour;
    rollout_frame(*in, static_cast<std::int64_t>(std::floor(frames + 1e-6)), f, rgba, stride);
    return NVFX_OK;
  }
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
  if (in->a.r) {
    // `frames` consecutive frames from the start of the timeline; a looping effect's `blend` extra frames are
    // crossfaded into its first ones (as the simulation makes its clips loop), so the flipbook loops without a jump.
    // Baking happens at load time, so unlike nvfx_render it may allocate (one frame).
    const auto& m = in->effect->roll->m;
    const std::size_t bytes = static_cast<std::size_t>(in->size) * in->size * 4, stride = static_cast<std::size_t>(in->size) * 4;
    const int blend = m.loop ? std::min(16, frames / 2) : 0;
    FrameInput f;
    f.colour = in->colour;
    f.apply_colour = in->apply_colour;
    try {
      std::vector<std::uint8_t> extra(blend > 0 ? bytes : 0);
      for (int k = 0; k < frames + blend; ++k) {
        if (k < frames) {
          rollout_frame(*in, k, f, rgba + bytes * static_cast<std::size_t>(k), stride);
          continue;
        }
        const int e = k - frames;  // extra frame e flows into frame e: mix(extra, frame, (e + 0.5) / blend)
        rollout_frame(*in, k, f, extra.data(), stride);
        std::uint8_t* d = rgba + bytes * static_cast<std::size_t>(e);
        const float w = (static_cast<float>(e) + 0.5f) / static_cast<float>(blend);
        for (std::size_t i = 0; i < bytes; ++i) d[i] = static_cast<std::uint8_t>(static_cast<float>(extra[i]) * (1.f - w) + static_cast<float>(d[i]) * w + 0.5f);
      }
    } catch (const std::bad_alloc&) {
      return NVFX_ERROR_MEMORY;
    }
    return NVFX_OK;
  }
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

double nvfx_instance_macs_per_pixel(const nvfx_instance* in) {
  if (!in) return 0.0;
  return in->a.r ? in->a.r->macs_per_pixel() : in->renderer->macs_per_pixel();
}

}  // extern "C"
