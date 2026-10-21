// The detail layer's A/B probe (tools/nvfx_detail_bench.cpp; docs/REPORT.md §19): a small C interface to one build of
// the runtime's internals, built as a shared module. nvfx_detail_bench loads two of them into one process, one built
// from this tree and one from the commit before the change (tools/detail_bench/build_old.sh), and steps their runners
// in turn: the same models, sizes, ISAs, seeds and controls, so their times can be compared frame by frame and their
// frames checked bit for bit. Only the probe_* functions are exported (the libraries' symbols are made local when it is
// linked), so the two builds' internals do not meet.
//
// This file uses only what both builds have: the rollout runners of each ISA (rt_common.hpp), the .nvfx loader, the
// C API (nvfx.h) and the volume C API (nvfx_volume.h).
#include "rt_common.hpp"

#include <neuralfx/nvfx.h>
#include <neuralfx/nvfx_volume.h>
#include <neuralfx/rollout.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

#if defined(__GNUC__)
#define PROBE_API extern "C" __attribute__((visibility("default")))
#else
#define PROBE_API extern "C"
#endif

namespace rt = nfx::rt;

namespace {

struct Runner {
  std::unique_ptr<rt::RolloutRunner> r;
  rt::RolloutProfile prof;
  std::vector<std::uint8_t> rgba;
  int size = 0;
};

}  // namespace

PROBE_API int probe_abi() { return 1; }

// --- 2D: rollout runners ----------------------------------------------------------------------------------------------

PROBE_API void* probe_effect(const char* path) {
  try {
    auto m = nfx::rollout::load_model(path);
    if (!m) return nullptr;
    auto* e = new rt::RolloutEffect;
    e->m = std::move(*m);
    return e;
  } catch (...) {
    return nullptr;
  }
}
PROBE_API void probe_effect_free(void* e) { delete static_cast<rt::RolloutEffect*>(e); }

// A runner of the effect at `size` pixels on an ISA (nvfx_isa: 1 baseline, 2 AVX2, 3 AVX-512), float (int8 = 0) or
// with the stepper's second convolution in 8 bits (int8 = 1).
PROBE_API void* probe_runner(void* effect, int size, int isa, int int8) {
  try {
    const auto& e = *static_cast<const rt::RolloutEffect*>(effect);
    const rt::Precision p = int8 ? rt::Precision::int8 : rt::Precision::float32;
    auto* r = new Runner;
    switch (isa) {
      case NVFX_ISA_AVX512: r->r = rt::isa_avx512::make_rollout(e, size, false, p); break;
      case NVFX_ISA_AVX2: r->r = rt::isa_avx2::make_rollout(e, size, false, p); break;
      default: r->r = rt::isa_base::make_rollout(e, size, false, p); break;
    }
    r->size = size;
    r->rgba.assign(static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4, 0);
    r->r->set_profile(&r->prof);
    return r;
  } catch (...) {
    return nullptr;
  }
}
PROBE_API void probe_runner_free(void* r) { delete static_cast<Runner*>(r); }
PROBE_API void probe_start(void* r, int index, const float* c, int n, std::uint64_t seed) {
  static_cast<Runner*>(r)->r->start(index, std::span<const float>(c, static_cast<std::size_t>(n)), seed);
}
PROBE_API void probe_step(void* r, const float* c, int n, std::uint64_t seed) {
  static_cast<Runner*>(r)->r->step(std::span<const float>(c, static_cast<std::size_t>(n)), seed);
}
// The picture (RGBA8, size x size), into the runner's own buffer; returns it.
PROBE_API const std::uint8_t* probe_render(void* r) {
  auto* q = static_cast<Runner*>(r);
  rt::FrameInput in;
  q->r->render(in, q->rgba.data(), static_cast<std::size_t>(q->size) * 4);
  return q->rgba.data();
}
// 0: fine heat, 1: fine soot, 2: coarse state, 3: flow.
PROBE_API const float* probe_field(void* r, int which, std::size_t* n) {
  const rt::RolloutRunner& q = *static_cast<Runner*>(r)->r;
  const std::span<const float> s = which == 0 ? q.fine_heat() : which == 1 ? q.fine_soot() : which == 2 ? q.coarse() : q.flow();
  *n = s.size();
  return s.data();
}
// The thread CPU time (ms) of each stage so far: noise, conv, project, detail, shade cells, shade pixels.
PROBE_API void probe_stages(void* r, double* ms) {
  const auto& p = static_cast<Runner*>(r)->prof;
  for (std::size_t s = 0; s < p.ms.size(); ++s) ms[s] = p.ms[s];
}

// --- 2D: instances through the C API (shards, crossfades, warm-ups) ---------------------------------------------------

PROBE_API void* probe_api_effect(const char* path) {
  nvfx_effect* e = nullptr;
  return nvfx_effect_load(path, &e) == NVFX_OK ? e : nullptr;
}
PROBE_API void probe_api_effect_free(void* e) { nvfx_effect_free(static_cast<nvfx_effect*>(e)); }
// An instance at `size` on an ISA (0: the default), at a precision (nvfx_precision), with controls and a seed.
PROBE_API void* probe_api_instance(void* e, int size, int isa, int precision, const float* c, int n, std::uint64_t seed) {
  if (nvfx_set_isa(static_cast<nvfx_isa>(isa)) != NVFX_OK) return nullptr;
  nvfx_instance* in = nullptr;
  const nvfx_status s = nvfx_instance_create(static_cast<nvfx_effect*>(e), size, &in);
  nvfx_set_isa(NVFX_ISA_AUTO);
  if (s != NVFX_OK) return nullptr;
  nvfx_instance_set_precision(in, static_cast<nvfx_precision>(precision));
  nvfx_instance_set_controls(in, c, n);
  nvfx_instance_set_seed(in, seed);
  return in;
}
PROBE_API void probe_api_instance_free(void* in) { nvfx_instance_free(static_cast<nvfx_instance*>(in)); }
PROBE_API int probe_api_render(void* in, double t, std::uint8_t* rgba, std::size_t stride) {
  return nvfx_render(static_cast<nvfx_instance*>(in), t, rgba, stride) == NVFX_OK ? 0 : 1;
}

// --- 3D: volumes through the volume C API -----------------------------------------------------------------------------

PROBE_API void* probe_volume_effect(const char* path) {
  nvfx_volume_effect* e = nullptr;
  return nvfx_volume_effect_load(path, &e) == NVFX_OK ? e : nullptr;
}
PROBE_API void probe_volume_effect_free(void* e) { nvfx_volume_effect_free(static_cast<nvfx_volume_effect*>(e)); }
// A volume on `threads` threads on an ISA (0: the default), playing one continuous rollout (no shards) from a start
// point with controls and a seed; its fine size in size[3].
PROBE_API void* probe_volume(void* e, int threads, int isa, int start, const float* c, int n, std::uint64_t seed, int* size) {
  if (nvfx_set_isa(static_cast<nvfx_isa>(isa)) != NVFX_OK) return nullptr;
  nvfx_volume* v = nullptr;
  const nvfx_status s = nvfx_volume_create(static_cast<nvfx_volume_effect*>(e), threads, &v);
  nvfx_set_isa(NVFX_ISA_AUTO);
  if (s != NVFX_OK) return nullptr;
  nvfx_volume_set_controls(v, c, n);
  nvfx_volume_set_seed(v, seed);
  nvfx_volume_set_variation(v, start);
  nvfx_volume_set_shard(v, 0.f);
  nvfx_volume_info info{};
  nvfx_volume_effect_get_info(static_cast<nvfx_volume_effect*>(e), &info);
  for (int k = 0; k < 3; ++k) size[k] = info.size[k];
  return v;
}
PROBE_API void probe_volume_free(void* v) { nvfx_volume_free(static_cast<nvfx_volume*>(v)); }
PROBE_API int probe_volume_step_to(void* v, double t) { return nvfx_volume_step_to(static_cast<nvfx_volume*>(v), t) == NVFX_OK ? 0 : 1; }
// field: nvfx_volume_field (0 temperature, 1 soot); out: size[0] * size[1] * size[2] floats.
PROBE_API int probe_volume_read(void* v, int field, float* out, std::size_t floats) {
  return nvfx_volume_read(static_cast<nvfx_volume*>(v), static_cast<nvfx_volume_field>(field), NVFX_VOLUME_FLOAT, out, floats * 4) == NVFX_OK ? 0 : 1;
}
PROBE_API std::size_t probe_volume_scratch_bytes(void* v) { return nvfx_volume_scratch_bytes(static_cast<nvfx_volume*>(v)); }

// --- 3D: the detail layer alone, the reference against the runtime's (builds that have rt_volume.hpp) ----------------
// Two copies of one rollout step in lockstep: the coarse step on each (not timed), then rollout3d::detail_step (the
// reference, which every volume ran before the runtime had its own) on one and the volume runtime's detail layer on the
// other, each timed, and their fields compared bit for bit.
#if __has_include("rt_volume.hpp")
#include "rt_volume.hpp"

#include <neuralfx/parallel.hpp>
#include <neuralfx/rollout3d.hpp>

#include <time.h>

#include <chrono>

namespace {

namespace r3 = nfx::rollout3d;

struct Detail3Pair {
  r3::Model m;
  std::unique_ptr<nfx::Pool> pool;
  std::unique_ptr<r3::Workspace> wa, wb;
  std::unique_ptr<r3::DetailKernel> kernel;
  r3::State a, b;
  std::vector<float> c;
  std::uint64_t seed = 1;
};

}  // namespace

PROBE_API void* probe_d3_pair(const char* path, int isa, int threads, int start, const float* c, int n, std::uint64_t seed) {
  try {
    auto m = r3::load_model(path);
    if (!m) return nullptr;
    auto* p = new Detail3Pair;
    p->m = std::move(*m);  // as a volume loads it
    if (threads > 1) p->pool = std::make_unique<nfx::Pool>(threads);
    p->wa = std::make_unique<r3::Workspace>(p->m);
    p->wb = std::make_unique<r3::Workspace>(p->m);
    p->kernel = rt::make_volume_detail(p->m, *p->wb, static_cast<nvfx_isa>(isa));
    if (!p->kernel) {
      delete p;
      return nullptr;
    }
    p->c.assign(c, c + n);
    p->seed = seed;
    p->a = r3::make_state(p->m);
    p->b = r3::make_state(p->m);
    r3::start(p->m, start, p->c, seed, p->a, *p->wa, p->pool.get());  // both warm up with the reference
    r3::start(p->m, start, p->c, seed, p->b, *p->wa, p->pool.get());
    return p;
  } catch (...) {
    return nullptr;
  }
}
PROBE_API void probe_d3_pair_free(void* p) { delete static_cast<Detail3Pair*>(p); }
// One step of both; ms[0] the reference's detail layer, ms[1] the runtime's (thread CPU time on one thread, else wall
// time); kernel_first swaps the order.
// Returns the number of fine cells whose heat or soot differ.
PROBE_API long probe_d3_pair_step(void* q, int kernel_first, double* ms) {
  auto& p = *static_cast<Detail3Pair*>(q);
  nfx::Pool* pool = p.pool.get();
  for (r3::State* s : {&p.a, &p.b}) {
    r3::Workspace& ws = s == &p.a ? *p.wa : *p.wb;
    for (int k = 0; k < p.m.h.cond(); ++k) ws.cond[static_cast<std::size_t>(k)] = static_cast<std::size_t>(k) < p.c.size() ? p.c[static_cast<std::size_t>(k)] : 0.5f;
    r3::coarse_noise(p.m, p.seed, s->time, ws, ws.noise, pool);
    r3::coarse_step(p.m, s->coarse, ws.noise, ws.cond, s->pressure, ws.next, s->flow, ws, pool);
    s->coarse.swap(ws.next);
  }
  const auto now = [&] {  // one thread: its CPU time; more: wall time
    if (!pool) {
      timespec t{};
      clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
      return static_cast<double>(t.tv_sec) * 1e3 + static_cast<double>(t.tv_nsec) * 1e-6;
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
  };
  const auto run = [&](bool kernel) {
    const double t0 = now();
    if (kernel) p.kernel->step(p.m, p.b, p.seed, p.c, *p.wb, pool);
    else r3::detail_step(p.m, p.a, p.seed, p.c, *p.wa, pool);
    ms[kernel ? 1 : 0] = now() - t0;
  };
  run(kernel_first != 0);
  run(kernel_first == 0);
  long bad = 0;
  for (std::size_t i = 0; i < p.a.fine_t.size(); ++i) {
    bad += std::memcmp(&p.a.fine_t[i], &p.b.fine_t[i], 4) != 0 || std::memcmp(&p.a.fine_d[i], &p.b.fine_d[i], 4) != 0;
  }
  for (r3::State* s : {&p.a, &p.b}) {
    s->time += 1.f / p.m.fps;
    s->since_start += 1.f / p.m.fps;
  }
  return bad;
}
#endif
