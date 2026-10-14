// Composed effects (compose.hpp, docs/COMPOSE.md).
#include "compose.hpp"

#include <neuralfx/noise.hpp>
#include <neuralfx/nvfx.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numbers>
#include <stdexcept>

namespace nfx::compose {

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int ifloor(float v) { return static_cast<int>(std::floor(v)); }

float smooth01(float t) {
  t = std::clamp(t, 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}

[[gnu::always_inline]] inline std::uint32_t hash32(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// A hash of integer coordinates in [0, 1): unit(hx(x) ^ hy(y) ^ hz(z)). The key is made by axis, so a stage can make
// each part once per column or row.
std::uint32_t hx(int x) { return static_cast<std::uint32_t>(x) * 73856093U; }
std::uint32_t hy(int y) { return static_cast<std::uint32_t>(y) * 19349663U; }
std::uint32_t hz(int z) { return static_cast<std::uint32_t>(z) * 83492791U; }
[[gnu::always_inline]] inline float unit(std::uint32_t key) { return static_cast<float>(hash32(key) >> 8) * (1.f / 16777216.f); }

// 8-bit display values to linear light (the learned renderers' output), and linear to 8-bit display values.
struct Gamma {
  std::array<float, 256> to_linear{};
  std::array<std::uint8_t, 4096> to_display{};
  Gamma() {
    for (int i = 0; i < 256; ++i) to_linear[zs(i)] = std::pow(fl(i) / 255.f, 2.2f);
    for (int i = 0; i < 4096; ++i) to_display[zs(i)] = static_cast<std::uint8_t>(std::lround(255.f * std::pow(fl(i) / 4095.f, 1.f / 2.2f)));
  }
};
const Gamma& gamma() {
  static const Gamma g;
  return g;
}

// Bilinear sample of an interleaved grid with `ch` channels at continuous cell coordinates (cell centres at
// integers), clamped to the grid. The reference the faster helpers below reproduce to the bit; no stage calls it now.
[[maybe_unused]] float bilerp(const float* f, int nx, int ny, int ch, int c, float x, float y) {
  x = std::clamp(x, 0.f, fl(nx - 1));
  y = std::clamp(y, 0.f, fl(ny - 1));
  const int x0 = std::min(static_cast<int>(x), std::max(0, nx - 2)), y0 = std::min(static_cast<int>(y), std::max(0, ny - 2));
  const int x1 = std::min(x0 + 1, nx - 1), y1 = std::min(y0 + 1, ny - 1);
  const float fx = x - fl(x0), fy = y - fl(y0);
  const auto at = [&](int xx, int yy) { return f[(zs(yy) * zs(nx) + zs(xx)) * zs(ch) + zs(c)]; };
  return (1.f - fy) * ((1.f - fx) * at(x0, y0) + fx * at(x1, y0)) + fy * ((1.f - fx) * at(x0, y1) + fx * at(x1, y1));
}

// One axis of bilerp(): the two cells a continuous coordinate falls between (clamped to n cells) and the weight of the
// second. Computed exactly as bilerp() does, so samples built from it are the same to the last bit.
struct Taps {
  int i0, i1;
  float f;
};
Taps taps(float x, int n) {
  x = std::clamp(x, 0.f, fl(n - 1));
  const int i0 = std::min(static_cast<int>(x), std::max(0, n - 2));
  return {i0, std::min(i0 + 1, n - 1), x - fl(i0)};
}

// bilerp() of channels 0 .. N - 1 of a grid with `ch` interleaved channels, the weights computed once: the same values
// as N calls of bilerp().
template <int N>
[[gnu::always_inline]] inline std::array<float, N> bilerp_n(const float* f, int nx, int ny, int ch, float x, float y) {
  const Taps tx = taps(x, nx), ty = taps(y, ny);
  const float* a = f + (zs(ty.i0) * zs(nx) + zs(tx.i0)) * zs(ch);
  const float* b = f + (zs(ty.i0) * zs(nx) + zs(tx.i1)) * zs(ch);
  const float* c = f + (zs(ty.i1) * zs(nx) + zs(tx.i0)) * zs(ch);
  const float* d = f + (zs(ty.i1) * zs(nx) + zs(tx.i1)) * zs(ch);
  const float fx = tx.f, fy = ty.f;
  std::array<float, N> s;
  for (int k = 0; k < N; ++k) s[zs(k)] = (1.f - fy) * ((1.f - fx) * a[k] + fx * b[k]) + fy * ((1.f - fx) * c[k] + fx * d[k]);
  return s;
}

// exp(x) without branches or calls, so loops over pixels vectorise: the polynomial and range reduction of the Cephes
// library, within a few units in the last place of std::exp. x is clamped to [-87, 88] (no denormals, no overflow).
inline float exp_fast(float x) {
  x = std::clamp(x, -87.f, 88.f);
  const float n = (x * 1.44269504088896341f + 12582912.f) - 12582912.f;  // round to the nearest integer
  const float r = (x - n * 0.693359375f) - n * -2.12194440e-4f;            // x - n ln 2, in two parts
  float p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  return (p * r * r + r + 1.f) * std::bit_cast<float>((static_cast<std::int32_t>(n) + 127) * (1 << 23));  // times 2^n
}

// heat_colour() as a sum of ramps, one per stop: no branches or table, so loops over pixels vectorise. The same colours
// up to rounding.
inline void heat_rgb(float t, float& r, float& g, float& b) {
  t = std::clamp(t, 0.f, 1.4f);
  const float r1 = std::max(0.f, t - 0.25f), r2 = std::max(0.f, t - 0.5f), r3 = std::max(0.f, t - 0.75f), r4 = std::max(0.f, t - 1.f);
  r = 0.25f + 2.f * t - r1 - r2;
  g = 0.02f + 0.4f * t + 0.52f * r1 + 0.16f * r2 - 0.16f * r3 - 0.62f * r4;
  b = 0.04f * t + 0.08f * r1 + 0.32f * r2 + 0.76f * r3 - 0.2f * r4;
}

// The field shader on pixels [x0, x1) of one row (Module::shade): everything per pixel, in one loop without branches or
// calls, which GCC vectorises. The light and the shadow come as rows already resampled along x, to be blended.
struct ShadeSpan {
  const float* heat;                      // fine heat and soot of the row
  const float* soot;
  const float* soot_up;                   // soot two rows above and below (clamped to the tile)
  const float* soot_down;
  const float* slope_x;                   // soot slope along x
  const float* ramp;                      // (heat / heat_scale) ^ emission_power
  const float* shadow0;                   // two coarse rows of the shadow sum resampled along x, and the second's weight
  const float* shadow1;
  float shadow_f;
  const float* light0;                    // two rows of the light resampled along x (a plane of `size` per colour)
  const float* light1;
  float light_f;
  std::array<float, 3> flash;
  int size;
  float* out;                             // RGBA, the row's first pixel
};

// (The rows are restrict-qualified parameters: GCC then knows the output overlaps none of them.)
[[gnu::always_inline]] inline void shade_pixels(const float* __restrict fh, const float* __restrict fs, const float* __restrict up, const float* __restrict dn,
                                    const float* __restrict sx, const float* __restrict pw, const float* __restrict s0, const float* __restrict s1,
                                    const float* __restrict l0, const float* __restrict l1, float* __restrict o, const ShadeSpan& s, const ShaderSpec& sp,
                                    int x0, int x1) {
  const float* r0 = l0;
  const float* g0 = l0 + s.size;
  const float* b0 = l0 + 2 * s.size;
  const float* r1 = l1;
  const float* g1 = l1 + s.size;
  const float* b1 = l1 + 2 * s.size;
  const float sf = s.shadow_f, lf = s.light_f, fr = s.flash[0], fg = s.flash[1], fb = s.flash[2];
  const float density = sp.soot_density, shadow = sp.shadow, relief = sp.relief, sky = sp.sky, scene = sp.scene_light;
  const float inv_hs = 1.f / sp.heat_scale, emission = sp.emission, albedo = sp.soot_albedo;
  const float tr = sp.tint[0], tg = sp.tint[1], tb = sp.tint[2];
  for (int x = x0; x < x1; ++x) {
    const float T = std::max(0.f, fh[x]), D = std::max(0.f, fs[x]);
    const std::uint32_t keep = std::max(T, D) >= 1e-4f ? ~0u : 0u;  // empty pixels stay transparent
    const float a = 1.f - exp_fast(-density * D);
    const float sh = exp_fast(-shadow * ((1.f - sf) * s0[x] + sf * s1[x]));
    // the soot as a height field: its slope towards the moon (up and left) lights billows, away from it darkens them
    const float nx = -relief * sx[x], ny = -relief * (0.5f * (up[x] - dn[x])), inv = 1.f / std::sqrt(nx * nx + ny * ny + 1.f);
    const float lambert = std::max(0.f, (-0.45f * nx + 0.6f * ny + 0.66f) * inv);
    const float moon = sky * sh * (0.35f + 0.9f * lambert);
    const float under = 0.4f + 0.9f * std::max(0.f, (0.2f * nx - 0.7f * ny + 0.68f) * inv);  // facing down: lit by the fire below
    const float lr = (1.f - lf) * r0[x] + lf * r1[x] + fr, lg = (1.f - lf) * g0[x] + lf * g1[x] + fg, lb = (1.f - lf) * b0[x] + lf * b1[x] + fb;
    const float Lr = 0.7f * moon + scene * lr / (1.f + lr) * under;  // soft limit: hot gas inside its own glow
    const float Lg = 0.8f * moon + scene * lg / (1.f + lg) * under;
    const float Lb = 1.1f * moon + scene * lb / (1.f + lb) * under;
    float hr, hg, hb;
    heat_rgb(T * inv_hs, hr, hg, hb);
    const float e = emission * pw[x] * (1.f - 0.55f * a);
    const float c[4] = {a * albedo * tr * Lr + e * hr, a * albedo * tg * Lg + e * hg, a * albedo * tb * Lb + e * hb, a};
    for (int k = 0; k < 4; ++k) o[zs(x) * 4 + zs(k)] = std::bit_cast<float>(std::bit_cast<std::uint32_t>(c[k]) & keep);
  }
}

// Compiled twice: for the baseline ISA and, when the module's runner uses it, for AVX2 (wider vectors, and multiply-adds
// fused, so the two differ in the last bit or so, as the runtime's ISAs do).
[[gnu::noinline]] void shade_span_base(const ShadeSpan& s, const ShaderSpec& sp, int x0, int x1) {
  shade_pixels(s.heat, s.soot, s.soot_up, s.soot_down, s.slope_x, s.ramp, s.shadow0, s.shadow1, s.light0, s.light1, s.out, s, sp, x0, x1);
}

#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#define NFX_COMPOSE_AVX2 1
[[gnu::noinline, gnu::target("arch=x86-64-v3")]] void shade_span_avx2(const ShadeSpan& s, const ShaderSpec& sp, int x0, int x1) {
  shade_pixels(s.heat, s.soot, s.soot_up, s.soot_down, s.slope_x, s.ramp, s.shadow0, s.shadow1, s.light0, s.light1, s.out, s, sp, x0, x1);
}
#endif

void shade_span(const ShadeSpan& s, const ShaderSpec& sp, int x0, int x1, Isa isa) {
#if defined(NFX_COMPOSE_AVX2)
  if (isa != Isa::base) return shade_span_avx2(s, sp, x0, x1);
#endif
  (void)isa;
  shade_span_base(s, sp, x0, x1);
}

}  // namespace

// --- Pool --------------------------------------------------------------------------------------------------------------

Pool::Pool(int threads) {
  for (int i = 1; i < threads; ++i) workers_.emplace_back([this] { work(); });
}

Pool::~Pool() {
  stop_.store(true);
  wake();
  for (auto& t : workers_) t.join();
}

void Pool::wake() {
  posted_.fetch_add(1, std::memory_order_release);
  posted_.notify_all();
}

namespace {
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#endif
}
}  // namespace

void Pool::run_impl(int n, Fn fn, void* ctx) {
  if (n <= 0) return;
  if ((workers_.empty() && !shared_) || n == 1) {
    for (int i = 0; i < n; ++i) fn(ctx, i);
    return;
  }
  int s = -1;
  {
    std::lock_guard lk(slots_mu_);
    for (int q = 0; q < kJobs && s < 0; ++q)
      if (!slot_taken_[zs(q)]) s = q;
    if (s >= 0) slot_taken_[zs(s)] = true;
  }
  if (s < 0) {  // more callers than slots: alone
    for (int i = 0; i < n; ++i) fn(ctx, i);
    return;
  }
  Job& j = jobs_[zs(s)];
  while (j.users.load() != 0) cpu_relax();  // a late worker of the slot's last job has left it
  j.fn = fn;
  j.ctx = ctx;
  j.n.store(n);
  j.next.store(0);
  j.left.store(n);
  j.live.store(true);
  wake();
  for (int i; (i = j.next.fetch_add(1)) < n;) {
    fn(ctx, i);
    if (j.left.fetch_sub(1) == 1) j.left.notify_all();
  }
  // the others' tasks of this job: run other jobs' tasks meanwhile, one at a time, then sleep until the last is done
  for (;;) {
    if (j.left.load() == 0) break;
    if (take(s, true)) continue;
    int left = 0;
    for (int k = 0; k < 64 && (left = j.left.load()) != 0; ++k) cpu_relax();
    if (left != 0) j.left.wait(left);
  }
  j.live.store(false);
  std::lock_guard lk(slots_mu_);
  slot_taken_[zs(s)] = false;
}

// A worker (or a waiting caller) enters a live job only after counting itself among its users and seeing it still
// live; the owner reuses the slot only when no user is left. (Sequentially consistent atomics: if the owner saw no
// user after ending the job, a worker that comes later sees the job ended, or the next one fully set up.)
bool Pool::take(int skip, bool one) {
  for (int q = 0; q < kJobs; ++q) {
    if (q == skip) continue;
    Job& j = jobs_[zs(q)];
    if (!j.live.load() || j.next.load(std::memory_order_relaxed) >= j.n.load(std::memory_order_relaxed)) continue;
    j.users.fetch_add(1);
    if (!j.live.load()) {
      j.users.fetch_sub(1);
      continue;
    }
    const Fn fn = j.fn;
    void* ctx = j.ctx;
    const int n = j.n.load();
    bool ran = false;
    for (int i; (i = j.next.fetch_add(1)) < n;) {
      fn(ctx, i);
      ran = true;
      if (j.left.fetch_sub(1) == 1) j.left.notify_all();
      if (one) break;
    }
    j.users.fetch_sub(1);
    if (ran) return true;
  }
  return false;
}

void Pool::help_impl(bool (*done)(void*), void* ctx) {
  for (;;) {
    const std::uint32_t seen = posted_.load();
    if (done(ctx)) return;
    if (take(-1, true)) continue;
    if (done(ctx)) return;
    posted_.wait(seen);
  }
}

void Pool::work() {
  for (;;) {
    const std::uint32_t seen = posted_.load();
    if (stop_.load()) return;
    bool any = false;
    while (take(-1, false)) any = true;
    if (any) continue;
    for (int k = 0; k < 256 && posted_.load(std::memory_order_relaxed) == seen; ++k) cpu_relax();  // a new job often follows soon
    posted_.wait(seen);
  }
}

// --- modules -----------------------------------------------------------------------------------------------------------

Isa best_isa() {
  switch (nvfx_get_isa()) {
    case NVFX_ISA_AVX512: return Isa::avx512;
    case NVFX_ISA_AVX2: return Isa::avx2;
    default: return Isa::base;
  }
}

const char* isa_name(Isa isa) {
  switch (isa) {
    case Isa::avx512: return "avx512";
    case Isa::avx2: return "avx2";
    case Isa::base: return "baseline";
  }
  return "?";
}

std::unique_ptr<rt::RolloutRunner> make_runner(const rt::RolloutEffect& e, int size, Isa isa) {
  switch (isa) {
    case Isa::avx512: return rt::isa_avx512::make_rollout(e, size);
    case Isa::avx2: return rt::isa_avx2::make_rollout(e, size);
    case Isa::base: return rt::isa_base::make_rollout(e, size);
  }
  return nullptr;
}

StepScratch::StepScratch(int slots) : s_(zs(std::max(1, slots))), busy_(std::make_unique<std::atomic<bool>[]>(s_.size())) {}

void StepScratch::fit(const rt::RolloutRunner& r) {
  for (rt::RolloutScratch& s : s_) s.fit(r);
}

rt::RolloutScratch& StepScratch::take() {
  for (;;) {
    for (std::size_t i = 0; i < s_.size(); ++i) {
      bool free = false;
      if (!busy_[i].load(std::memory_order_relaxed) && busy_[i].compare_exchange_strong(free, true, std::memory_order_acquire)) return s_[i];
    }
    std::this_thread::yield();  // more threads step than there are slots: wait for one
  }
}

void StepScratch::give(rt::RolloutScratch& s) { busy_[zs(static_cast<int>(&s - s_.data()))].store(false, std::memory_order_release); }

std::size_t StepScratch::bytes() const {
  std::size_t n = 0;
  for (const rt::RolloutScratch& s : s_) n += s.bytes();
  return n;
}

Module::Module(std::string name, const rt::RolloutEffect& e, int size, Placement p, Isa isa)
    : at(p), name_(std::move(name)), e_(e), size_(size), isa_(isa), r_(make_runner(e, size, isa)) {
  img_.allocate(size, size);
  rgba8_.assign(zs(size) * zs(size) * 4, 0);
  shadow_.assign(zs(res()) * zs(res()), 0.f);
  shadow_x_.assign(zs(res()) * zs(size), 0.f);
  light_x_.assign(zs(size) * 6, 0.f);
  sx_.assign(zs(size) * 2, 0);
  lx_.assign(zs(size) * 2, 0);
  sfx_.assign(zs(size), 0.f);
  lfx_.assign(zs(size), 0.f);
  row_.assign(zs(size) * 2, 0.f);
  drawn_.assign(zs(size) * 2, 0);  // the image starts black
  const float k = fl(size) / fl(res());
  for (int x = 0; x < size; ++x) {  // the coarse columns (shadow) each pixel's x falls between: fixed for the tile
    const Taps t = taps((fl(x) + 0.5f) / k - 0.5f, res());
    sx_[zs(x)] = t.i0;
    sx_[zs(size + x)] = t.i1;
    sfx_[zs(x)] = t.f;
  }
  pushed_.assign(zs(res()) * zs(res()) * 2, 0.f);
  controls.assign(zs(e.m.h.n_controls), 0.5f);
}

void Module::share_scratch(StepScratch& shared) {
  shared.fit(*r_);
  rt::RolloutScratch& s = shared.take();
  r_->use_scratch(&s);  // frees the runner's own
  shared.give(s);
  shared_ = &shared;
}

void Module::start(int index, std::uint64_t run_seed) {
  seed = run_seed;
  rt::RolloutScratch* s = shared_ ? &shared_->take() : nullptr;
  if (s) r_->use_scratch(s);
  r_->start(index, controls, seed);
  if (s) shared_->give(*s);
  active = true;
  frames = 0;
}

void Module::start_empty(float seconds, std::uint64_t run_seed) {
  seed = run_seed;
  r_->begin(0, seed);
  std::ranges::fill(r_->coarse_mut(), 0.f);
  std::ranges::fill(r_->fine_heat_mut(), 0.f);
  std::ranges::fill(r_->fine_soot_mut(), 0.f);
  r_->adopt(seconds);
  active = true;
  frames = 0;
}

void hand_over(const Module& from, Module& to) {
  if (from.size() != to.size() || from.res() != to.res()) throw std::invalid_argument("hand_over: modules differ in size");
  auto a = from.runner().coarse();
  auto b = to.runner().coarse_mut();
  const int Ca = from.channels(), Cb = to.channels(), N = from.res() * from.res();
  std::ranges::fill(b, 0.f);  // the receiving model's own memory channels start from rest
  for (int i = 0; i < N; ++i) {
    for (int c = 0; c < rollout::kPhys; ++c) b[zs(i) * zs(Cb) + zs(c)] = a[zs(i) * zs(Ca) + zs(c)];
  }
  std::ranges::copy(from.runner().fine_heat(), to.runner().fine_heat_mut().begin());
  std::ranges::copy(from.runner().fine_soot(), to.runner().fine_soot_mut().begin());
  to.runner().adopt(from.runner().time());
}

void Module::take_over(const Module& from) {
  hand_over(from, *this);
  active = true;
  frames = 0;
}

void Module::step() {
  const auto t0 = std::chrono::steady_clock::now();
  rt::RolloutScratch* s = shared_ ? &shared_->take() : nullptr;
  if (s) r_->use_scratch(s);
  r_->step(controls, seed);
  if (s) shared_->give(*s);
  if (has_push_) {  // what was pushed in moved material for one step; take it out again
    auto co = r_->coarse_mut();
    const int C = channels(), N = res() * res();
    for (int i = 0; i < N; ++i) {
      co[zs(i) * zs(C)] -= pushed_[zs(i) * 2];
      co[zs(i) * zs(C) + 1] -= pushed_[zs(i) * 2 + 1];
    }
    std::ranges::fill(pushed_, 0.f);
    has_push_ = false;
  }
  ++frames;
  step_ms = ms_since(t0);
}

float Module::weight_px(float x, float y) const {
  const float k = fl(size_) / fl(res()), R = fl(res());
  const float cx = x / k - 0.5f, cy = y / k - 0.5f;  // cell coordinates (cell centres at integers)
  float w = 1.f;
  if (band[0] > 0) w *= 1.f - band_weight(cx, band[0]);
  if (band[1] > 0) w *= band_weight(cx - (R - fl(band[1])), band[1]);
  if (band[2] > 0) w *= 1.f - band_weight(cy, band[2]);
  if (band[3] > 0) w *= band_weight(cy - (R - fl(band[3])), band[3]);
  if (feather > 0.f) {
    const float S = fl(size_);
    if (band[0] == 0) w *= smooth01(x / feather);
    if (band[1] == 0) w *= smooth01((S - x) / feather);
    if (band[2] == 0) w *= smooth01(y / feather);
    if (band[3] == 0) w *= smooth01((S - y) / feather);
  }
  return w;
}

Module::WeightX Module::weight_x(float x) const {
  const float k = fl(size_) / fl(res()), R = fl(res());
  const float cx = x / k - 0.5f;
  WeightX w;
  if (band[0] > 0) w.band *= 1.f - band_weight(cx, band[0]);
  if (band[1] > 0) w.band *= band_weight(cx - (R - fl(band[1])), band[1]);
  if (feather > 0.f) {
    if (band[0] == 0) w.feather0 = smooth01(x / feather);
    if (band[1] == 0) w.feather1 = smooth01((fl(size_) - x) / feather);
  }
  return w;
}

Module::WeightY Module::weight_y(float y) const {
  const float k = fl(size_) / fl(res()), R = fl(res());
  const float cy = y / k - 0.5f;
  WeightY w;
  if (band[2] > 0) w.band0 = 1.f - band_weight(cy, band[2]);
  if (band[3] > 0) w.band1 = band_weight(cy - (R - fl(band[3])), band[3]);
  if (feather > 0.f) {
    if (band[2] == 0) w.feather0 = smooth01(y / feather);
    if (band[3] == 0) w.feather1 = smooth01((fl(size_) - y) / feather);
  }
  return w;
}

float Module::weight_cell(int cx, int cy) const {
  const float k = fl(size_) / fl(res());
  return weight_px((fl(cx) + 0.5f) * k, (fl(cy) + 0.5f) * k);
}

void Module::shade(const Light* light) {
  const auto t0 = std::chrono::steady_clock::now();
  const int S = size_;
  const Gamma& g = gamma();
  if (look == Look::learned) {
    r_->render(rt::FrameInput{}, rgba8_.data(), zs(S) * 4);
    for (std::size_t i = 0; i < rgba8_.size(); i += 4) {
      img_.px[i] = g.to_linear[rgba8_[i]];
      img_.px[i + 1] = g.to_linear[rgba8_[i + 1]];
      img_.px[i + 2] = g.to_linear[rgba8_[i + 2]];
      img_.px[i + 3] = fl(rgba8_[i + 3]) * (1.f / 255.f);
    }
    for (int y = 0; y < S; ++y) {
      drawn_[zs(2 * y)] = 0;
      drawn_[zs(2 * y + 1)] = S;
    }
    shade_ms = ms_since(t0);
    return;
  }
  // field shader
  const int R = res(), C = channels();
  auto coarse = r_->coarse();
  for (int y = 0; y < R; ++y) {  // soot between each cell and the sky (straight up, slightly left)
    for (int x = 0; x < R; ++x) {
      float s = 0.f;
      for (int st = 1; st <= 8; ++st) {
        const int xx = x - st / 3, yy = y + st;
        if (xx < 0 || yy >= R) break;
        s += coarse[(zs(yy) * zs(R) + zs(xx)) * zs(C) + 3];
      }
      shadow_[zs(y) * zs(R) + zs(x)] = s;
    }
  }
  const ShaderSpec& sp = spec;
  auto ft = r_->fine_heat();
  auto fd = r_->fine_soot();
  const float k = fl(S) / fl(R), inv_hs = 1.f / sp.heat_scale, p = sp.emission_power;
  for (int cy = 0; cy < R; ++cy) {  // every coarse row of the shadow sum resampled at the pixels' x
    const float* s = shadow_.data() + zs(cy) * zs(R);
    float* o = shadow_x_.data() + zs(cy) * zs(S);
    for (int x = 0; x < S; ++x) o[x] = (1.f - sfx_[zs(x)]) * s[sx_[zs(x)]] + sfx_[zs(x)] * s[sx_[zs(S + x)]];
  }
  // the light grid's columns at the pixels' x (the tile may have moved); its rows are resampled along x when first
  // needed, two at a time (rows of pixels go down the light grid in order). No light: rows of zeros.
  std::array<int, 2> held{-1, -1};
  if (light) {
    for (int x = 0; x < S; ++x) {
      const Taps t = taps((at.x + (fl(x) + 0.5f) * at.scale - light->x0()) / light->cell() - 0.5f, light->nx());
      lx_[zs(x)] = t.i0;
      lx_[zs(S + x)] = t.i1;
      lfx_[zs(x)] = t.f;
    }
  } else {
    std::ranges::fill(light_x_, 0.f);
  }
  const auto light_row = [&](int j, int other) {  // slot holding light row j, filled if needed (keeping row `other`)
    for (int q = 0; q < 2; ++q)
      if (held[zs(q)] == j) return light_x_.data() + zs(q) * zs(S) * 3;
    const int q = held[0] == other ? 1 : 0;
    const float* L = light->field().data() + zs(j) * zs(light->nx()) * 3;
    float* o = light_x_.data() + zs(q) * zs(S) * 3;
    for (int c = 0; c < 3; ++c) {
      for (int x = 0; x < S; ++x) o[zs(c * S + x)] = (1.f - lfx_[zs(x)]) * L[zs(lx_[zs(x)]) * 3 + zs(c)] + lfx_[zs(x)] * L[zs(lx_[zs(S + x)]) * 3 + zs(c)];
    }
    held[zs(q)] = j;
    return static_cast<float*>(o);
  };
  constexpr int kSpan = 16;  // pixels tested together for material: empty spans are only cleared
  float* slope = row_.data();
  float* ramp = row_.data() + S;
  for (int y = 0; y < S; ++y) {  // y up
    float* out = img_.row(S - 1 - y);
    const float* fh = ft.data() + zs(y) * zs(S);
    const float* fs = fd.data() + zs(y) * zs(S);
    int* drawn = drawn_.data() + zs(S - 1 - y) * 2;  // [lo, hi): what the row held before; outside it is still black
    const int lo = drawn[0], hi = drawn[1];
    drawn[0] = S;
    drawn[1] = 0;
    ShadeSpan sp_row{};
    bool ready = false;
    for (int x0 = 0; x0 < S;) {
      const auto empty = [&](int xa) {
        unsigned any = 0;  // (an unsigned "or", which GCC vectorises)
        for (int x = xa; x < std::min(S, xa + kSpan); ++x) any |= static_cast<unsigned>(fh[x] >= 1e-4f) | static_cast<unsigned>(fs[x] >= 1e-4f);
        return any == 0;
      };
      int x1 = std::min(S, x0 + kSpan);
      if (empty(x0)) {  // a run of empty spans: cleared where the row was not black already
        while (x1 < S && empty(x1)) x1 = std::min(S, x1 + kSpan);
        const int c0 = std::max(x0, lo), c1 = std::min(x1, hi);
        if (c0 < c1) std::fill(out + zs(c0) * 4, out + zs(c1) * 4, 0.f);
        x0 = x1;
        continue;
      }
      while (x1 < S && !empty(x1)) x1 = std::min(S, x1 + kSpan);  // a run of spans with material: shaded together
      if (!ready) {  // what the row needs: its rows of the shadow and the light, and its neighbours' soot
        const Taps sy = taps((fl(y) + 0.5f) / k - 0.5f, R);
        sp_row.shadow0 = shadow_x_.data() + zs(sy.i0) * zs(S);
        sp_row.shadow1 = shadow_x_.data() + zs(sy.i1) * zs(S);
        sp_row.shadow_f = sy.f;
        if (light) {
          const float wy = at.y + (fl(S - y) - 0.5f) * at.scale;
          const Taps ly = taps((wy - light->y0()) / light->cell() - 0.5f, light->ny());
          sp_row.light0 = light_row(ly.i0, ly.i1);
          sp_row.light1 = light_row(ly.i1, ly.i0);
          sp_row.light_f = ly.f;
          sp_row.flash = light->flash();
        } else {
          sp_row.light0 = sp_row.light1 = light_x_.data();
        }
        sp_row.heat = fh;
        sp_row.soot = fs;
        sp_row.soot_up = fd.data() + zs(std::min(y + 2, S - 1)) * zs(S);
        sp_row.soot_down = fd.data() + zs(std::max(y - 2, 0)) * zs(S);
        sp_row.slope_x = slope;
        sp_row.ramp = ramp;
        sp_row.size = S;
        sp_row.out = out;
        ready = true;
      }
      const int xa = std::clamp(x0, 2, std::max(2, S - 2)), xb = std::clamp(x1, xa, std::max(2, S - 2));  // [xa, xb): both neighbours inside
      for (int x = x0; x < std::min(xa, x1); ++x) slope[x] = 0.5f * (fs[std::min(x + 2, S - 1)] - fs[std::max(x - 2, 0)]);
      for (int x = xa; x < xb; ++x) slope[x] = 0.5f * (fs[x + 2] - fs[x - 2]);
      for (int x = std::max(xb, x0); x < x1; ++x) slope[x] = 0.5f * (fs[std::min(x + 2, S - 1)] - fs[std::max(x - 2, 0)]);
      if (p == 3.f) {  // the usual powers without std::pow
        for (int x = x0; x < x1; ++x) {
          const float t = std::max(0.f, fh[x]) * inv_hs;
          ramp[x] = t * t * t;
        }
      } else if (p == 2.f) {
        for (int x = x0; x < x1; ++x) {
          const float t = std::max(0.f, fh[x]) * inv_hs;
          ramp[x] = t * t;
        }
      } else {
        for (int x = x0; x < x1; ++x) ramp[x] = std::pow(std::max(0.f, fh[x]) * inv_hs, p);
      }
      shade_span(sp_row, sp, x0, x1, isa_);
      drawn[0] = std::min(drawn[0], x0);
      drawn[1] = x1;
      x0 = x1;
    }
  }
  shade_ms = ms_since(t0);
}

// --- couplings ---------------------------------------------------------------------------------------------------------

void blend_band(Module& a, Module& b, Side b_is, int cells) {
  const int R = a.res(), S = a.size(), k = S / R, Ca = a.channels(), Cb = b.channels();
  auto A = a.runner().coarse_mut();
  auto B = b.runner().coarse_mut();
  // index of the j-th band line, position i along it, in a and in b
  const auto ca = [&](int j, int i) { return b_is == Side::top ? zs(R - cells + j) * zs(R) + zs(i) : zs(i) * zs(R) + zs(R - cells + j); };
  const auto cb = [&](int j, int i) { return b_is == Side::top ? zs(j) * zs(R) + zs(i) : zs(i) * zs(R) + zs(j); };
  for (int j = 0; j < cells; ++j) {
    const float w = band_weight(fl(j), cells);
    for (int i = 0; i < R; ++i) {
      float* pa = A.data() + ca(j, i) * zs(Ca);
      float* pb = B.data() + cb(j, i) * zs(Cb);
      for (int c = 0; c < rollout::kPhys; ++c) pa[c] = pb[c] = w * pa[c] + (1.f - w) * pb[c];
    }
  }
  const int bp = cells * k;
  for (auto [FA, FB] : {std::pair{a.runner().fine_heat_mut(), b.runner().fine_heat_mut()}, std::pair{a.runner().fine_soot_mut(), b.runner().fine_soot_mut()}}) {
    if (b_is == Side::top) {  // band lines are rows: a's top rows, b's bottom rows
      for (int j = 0; j < bp; ++j) {
        const float w = band_weight((fl(j) + 0.5f) / fl(k) - 0.5f, cells);
        float* __restrict pa = FA.data() + zs(S - bp + j) * zs(S);
        float* __restrict pb = FB.data() + zs(j) * zs(S);
        for (int i = 0; i < S; ++i) pa[i] = pb[i] = w * pa[i] + (1.f - w) * pb[i];
      }
    } else {  // band lines are columns: walked row by row, so memory is read in order
      for (int i = 0; i < S; ++i) {
        float* __restrict pa = FA.data() + zs(i) * zs(S) + zs(S - bp);
        float* __restrict pb = FB.data() + zs(i) * zs(S);
        for (int j = 0; j < bp; ++j) {
          const float w = band_weight((fl(j) + 0.5f) / fl(k) - 0.5f, cells);
          pa[j] = pb[j] = w * pa[j] + (1.f - w) * pb[j];
        }
      }
    }
  }
}

void suppress(Module& m, int x0, int y0, int x1, int y1) {
  const int R = m.res(), S = m.size(), k = S / R, C = m.channels();
  x0 = std::clamp(x0, 0, R);
  x1 = std::clamp(x1, 0, R);
  y0 = std::clamp(y0, 0, R);
  y1 = std::clamp(y1, 0, R);
  auto c = m.runner().coarse_mut();
  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      c[(zs(y) * zs(R) + zs(x)) * zs(C) + 2] = 0.f;
      c[(zs(y) * zs(R) + zs(x)) * zs(C) + 3] = 0.f;
    }
  }
  auto ft = m.runner().fine_heat_mut();
  auto fd = m.runner().fine_soot_mut();
  for (int y = y0 * k; y < y1 * k; ++y) {
    for (int x = x0 * k; x < x1 * k; ++x) ft[zs(y) * zs(S) + zs(x)] = fd[zs(y) * zs(S) + zs(x)] = 0.f;
  }
}

FieldBus::FieldBus(float x0, float y0, int nx, int ny, float cell, int groups) : x0_(x0), y0_(y0), cell_(cell), nx_(nx), ny_(ny), groups_(groups) {
  all_.assign(zs(nx) * zs(ny) * 4, 0.f);
  layer_.assign(zs(groups) * all_.size(), 0.f);
  heat_.assign(zs(nx) * zs(ny), 0.f);
  soot_.assign(heat_.size(), 0.f);
  glow_.assign(all_.size(), 0.f);
  dirty_.assign(zs(groups) + 1, Rect{});
  cols_.assign(zs(nx), Column{});
}

void FieldBus::clear() {
  // only what publish() wrote since the last clear (the rest is still zero)
  const auto zero = [&](std::vector<float>& f, std::size_t offset, int ch, const Rect& r) {
    for (int j = r.j0; j <= r.j1; ++j) std::fill_n(f.begin() + static_cast<std::ptrdiff_t>(offset + (zs(j) * zs(nx_) + zs(r.i0)) * zs(ch)), zs(r.i1 - r.i0 + 1) * zs(ch), 0.f);
  };
  for (int g = 0; g < groups_; ++g) zero(layer_, zs(g) * all_.size(), 4, dirty_[zs(g)]);
  const Rect& r = dirty_[zs(groups_)];
  zero(all_, 0, 4, r);
  zero(heat_, 0, 1, r);
  zero(soot_, 0, 1, r);
  if (glowing_) zero(glow_, 0, 4, r);
  glowing_ = false;
  std::ranges::fill(dirty_, Rect{});
}

void FieldBus::publish(const Module& m) {
  if (!m.active || m.group < 0 || m.group >= groups_) return;
  const int R = m.res(), S = m.size(), C = m.channels();
  const float k = fl(S) / fl(R), sc = m.at.scale, span = fl(S) * sc;
  auto co = m.runner().coarse();
  const int i0 = std::max(0, ifloor((m.at.x - x0_) / cell_)), i1 = std::min(nx_ - 1, ifloor((m.at.x + span - x0_) / cell_) + 1);
  const int j0 = std::max(0, ifloor((m.at.y - y0_) / cell_)), j1 = std::min(ny_ - 1, ifloor((m.at.y + span - y0_) / cell_) + 1);
  if (i0 > i1 || j0 > j1) return;
  for (Rect* r : {&dirty_[zs(m.group)], &dirty_[zs(groups_)]}) {
    if (r->i0 > r->i1) {
      *r = {i0, i1, j0, j1};
    } else {
      *r = {std::min(r->i0, i0), std::max(r->i1, i1), std::min(r->j0, j0), std::max(r->j1, j1)};
    }
  }
  for (int i = i0; i <= i1; ++i) {  // what depends on the column alone: computed once
    Column& c = cols_[zs(i)];
    const float wx = x0_ + (fl(i) + 0.5f) * cell_;
    const float tx = (wx - m.at.x) / sc;
    c.inside = !(tx < 0.f || tx > fl(S));
    if (!c.inside) continue;
    c.w = m.weight_x(tx);
    const Taps t = taps(tx / k - 0.5f, R);
    c.x0 = t.i0;
    c.x1 = t.i1;
    c.fx = t.f;
  }
  float* L = layer_.data() + zs(m.group) * all_.size();
  const bool glows = m.glows;
  const std::array<float, 3> tint = m.glow;
  glowing_ = glowing_ || glows;
  for (int j = j0; j <= j1; ++j) {
    const float wy = y0_ + (fl(j) + 0.5f) * cell_;
    const float ty = fl(S) - (wy - m.at.y) / sc;  // tile pixels, y up
    if (ty < 0.f || ty > fl(S)) continue;
    const Module::WeightY wyf = m.weight_y(ty);
    const Taps t = taps(ty / k - 0.5f, R);
    const float fy = t.f;
    const float* r0 = co.data() + zs(t.i0) * zs(R) * zs(C);
    const float* r1 = co.data() + zs(t.i1) * zs(R) * zs(C);
    for (int i = i0; i <= i1; ++i) {
      const Column& c = cols_[zs(i)];
      if (!c.inside) continue;
      const float w = Module::weight(c.w, wyf) * m.opacity;  // weight_px(tx, ty) * opacity
      if (w <= 0.f) continue;
      const float fx = c.fx;
      const float* a = r0 + zs(c.x0) * zs(C);
      const float* b = r0 + zs(c.x1) * zs(C);
      const float* d = r1 + zs(c.x0) * zs(C);
      const float* e = r1 + zs(c.x1) * zs(C);
      float v[4];  // bilinear, as bilerp() computes it
      for (int ch = 0; ch < 4; ++ch) v[ch] = (1.f - fy) * ((1.f - fx) * a[ch] + fx * b[ch]) + fy * ((1.f - fx) * d[ch] + fx * e[ch]);
      const float s[4] = {v[0] * k * sc, -v[1] * k * sc, v[2], v[3]};
      const std::size_t q = (zs(j) * zs(nx_) + zs(i)) * 4;
      for (int ch = 0; ch < 4; ++ch) {
        all_[q + zs(ch)] += w * s[ch];
        L[q + zs(ch)] += w * s[ch];
      }
      heat_[q / 4] += w * s[2];
      soot_[q / 4] += w * s[3];
      if (glows) {
        const float h = w * s[2];
        glow_[q] += h;
        for (int ch = 0; ch < 3; ++ch) glow_[q + 1 + zs(ch)] += h * tint[zs(ch)];
      }
    }
  }
}

FieldBus::Sample FieldBus::sample(const float* f, float x, float y) const {
  const float gx = (x - x0_) / cell_ - 0.5f, gy = (y - y0_) / cell_ - 0.5f;
  if (gx < -1.f || gy < -1.f || gx > fl(nx_) || gy > fl(ny_)) return {};
  const auto s = bilerp_n<4>(f, nx_, ny_, 4, gx, gy);
  return {s[0], s[1], s[2], s[3]};
}

FieldBus::Sample FieldBus::at(float x, float y) const { return sample(all_.data(), x, y); }

FieldBus::Sample FieldBus::others(float x, float y, int group) const {
  Sample s = sample(all_.data(), x, y);
  if (group < 0 || group >= groups_) return s;
  const Sample own = sample(layer_.data() + zs(group) * all_.size(), x, y);  // zero outside, as all_ is
  s.u -= own.u;
  s.v -= own.v;
  s.heat -= own.heat;
  s.soot -= own.soot;
  return s;
}

void push(Module& m, const FieldBus& bus, float gain) {
  if (!m.active) return;
  const int R = m.res(), S = m.size(), C = m.channels();
  const float k = fl(S) / fl(R), sc = m.at.scale, to_cells = 1.f / (k * sc);
  auto co = m.runner().coarse_mut();
  for (int cy = 0; cy < R; ++cy) {
    const float wy = m.at.y + (fl(S) - (fl(cy) + 0.5f) * k) * sc;
    for (int cx = 0; cx < R; ++cx) {
      const float wx = m.at.x + (fl(cx) + 0.5f) * k * sc;
      const FieldBus::Sample o = bus.others(wx, wy, m.group);
      const std::size_t i = zs(cy) * zs(R) + zs(cx);
      float* c = co.data() + i * zs(C);
      const float du = gain * o.u * to_cells, dv = -gain * o.v * to_cells;
      c[0] += du;
      c[1] += dv;
      m.pushed_[i * 2] += du;
      m.pushed_[i * 2 + 1] += dv;
    }
  }
  m.has_push_ = true;
}

void apply(Module& m, const ForceField& f, float weight) {
  if (!m.active || weight <= 0.f) return;
  if (f.kind == ForceField::Kind::heat || f.kind == ForceField::Kind::cold) return apply_heat(m, f, weight);  // fields.cpp
  const int R = m.res(), S = m.size(), C = m.channels();
  const float k = fl(S) / fl(R), sc = m.at.scale, to_cells = 1.f / (k * sc);
  auto co = m.runner().coarse_mut();
  for (int cy = 0; cy < R; ++cy) {
    const float wy = m.at.y + (fl(S) - (fl(cy) + 0.5f) * k) * sc;
    for (int cx = 0; cx < R; ++cx) {
      const float wx = m.at.x + (fl(cx) + 0.5f) * k * sc;
      const std::size_t i = zs(cy) * zs(R) + zs(cx);
      float* c = co.data() + i * zs(C);
      float du = 0.f, dv = 0.f;  // world pixels per frame, y down
      switch (f.kind) {
        case ForceField::Kind::ceiling: {
          const float depth = smooth01((f.y - wy) / f.soft);  // 0 below the ceiling, 1 well above it
          c[1] *= 1.f - weight * f.damping * depth;           // permanent: damping does not build up
          continue;
        }
        case ForceField::Kind::vortex: {
          const float ex = wx - f.x, ey = wy - f.y, r2 = ex * ex + ey * ey, rr = f.radius * f.radius;
          const float s = weight * f.strength * std::sqrt(r2) / f.radius * std::exp(0.5f * (1.f - r2 / rr)) / (std::sqrt(r2) + 1e-3f);
          du = -s * ey;
          dv = s * ex;
          break;
        }
        case ForceField::Kind::wind:
          du = weight * f.u;
          dv = weight * f.v;
          break;
        default: {  // gust, ring, attract's swirl (fields.cpp)
          const auto uv = field_flow(f, wx, wy, weight);
          du = uv[0];
          dv = uv[1];
          break;
        }
      }
      const float a = du * to_cells, b = -dv * to_cells;
      c[0] += a;
      c[1] += b;
      m.pushed_[i * 2] += a;
      m.pushed_[i * 2 + 1] += b;
    }
  }
  if (f.kind != ForceField::Kind::ceiling) m.has_push_ = true;
}

void transfer(Module& from, std::span<Module* const> to, float fraction, int row0, float heat_gain, float soot_gain) {
  if (!from.active || fraction <= 0.f) return;
  const int R = from.res(), S = from.size(), C = from.channels();
  const float k = fl(S) / fl(R), sc = from.at.scale;
  auto co = from.runner().coarse_mut();
  auto ft = from.runner().fine_heat_mut();
  auto fd = from.runner().fine_soot_mut();
  row0 = std::clamp(row0, 0, R);
  // The targets a source cell can reach are those whose tile overlaps it (by a margin of a world pixel, for rounding):
  // only those are tried for the cell and its pixels, in their order, so the result is as if all were. Up to 64
  // targets are told apart by a bit each; with more, all are tried.
  const bool masked = to.size() <= 64;
  std::array<float*, 64> to_heat{}, to_soot{};
  if (masked) {
    for (std::size_t q = 0; q < to.size(); ++q) {
      to_heat[q] = to[q]->runner().fine_heat_mut().data();
      to_soot[q] = to[q]->runner().fine_soot_mut().data();
    }
  }
  // coarse: each source cell's amount, splatted bilinearly into the targets' cells at its world centre
  for (int cy = row0; cy < R; ++cy) {
    for (int cx = 0; cx < R; ++cx) {
      float* c = co.data() + (zs(cy) * zs(R) + zs(cx)) * zs(C);
      const float h = c[2] * fraction * heat_gain, d = c[3] * fraction * soot_gain;
      if (h <= 0.f && d <= 0.f) continue;
      const float wx = from.at.x + (fl(cx) + 0.5f) * k * sc, wy = from.at.y + (fl(S) - (fl(cy) + 0.5f) * k) * sc;
      const float area = (k * sc) * (k * sc);
      std::uint64_t near = 0;
      if (masked) {
        const float x0 = from.at.x + fl(cx) * k * sc - 1.f, x1 = from.at.x + fl(cx + 1) * k * sc + 1.f;
        const float y0 = from.at.y + (fl(S) - fl(cy + 1) * k) * sc - 1.f, y1 = from.at.y + (fl(S) - fl(cy) * k) * sc + 1.f;
        for (std::size_t q = 0; q < to.size(); ++q) {
          const Module* t = to[q];
          const float span = fl(t->size()) * t->at.scale;
          if (t->at.x <= x1 && t->at.x + span >= x0 && t->at.y <= y1 && t->at.y + span >= y0) near |= std::uint64_t{1} << q;
        }
      }
      const auto reaches = [&](std::size_t q) { return !masked || (near >> q & 1) != 0; };
      float placed = 0.f;
      for (std::size_t q = 0; q < to.size(); ++q) {
        Module* t = to[q];
        if (!reaches(q) || !t->active || t == &from) continue;
        const int tR = t->res(), tS = t->size(), tC = t->channels();
        const float tk = fl(tS) / fl(tR), tsc = t->at.scale;
        const float tx = (wx - t->at.x) / tsc, ty = fl(tS) - (wy - t->at.y) / tsc;
        if (tx < 0.f || ty < 0.f || tx >= fl(tS) || ty >= fl(tS)) continue;
        const float w = t->weight_px(tx, ty);
        if (w <= 0.f) continue;
        const float gx = std::clamp(tx / tk - 0.5f, 0.f, fl(tR - 1) - 1e-3f), gy = std::clamp(ty / tk - 0.5f, 0.f, fl(tR - 1) - 1e-3f);
        const int x0 = std::min(static_cast<int>(gx), tR - 2), y0 = std::min(static_cast<int>(gy), tR - 2);
        const float fx = gx - fl(x0), fy = gy - fl(y0);
        const float ratio = w * area / ((tk * tsc) * (tk * tsc));
        auto tc = t->runner().coarse_mut();
        const float ws[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
        const int dx[4] = {0, 1, 0, 1}, dy[4] = {0, 0, 1, 1};
        for (int n = 0; n < 4; ++n) {
          float* p = tc.data() + (zs(y0 + dy[n]) * zs(tR) + zs(x0 + dx[n])) * zs(tC);
          p[2] += ratio * ws[n] * h;
          p[3] += ratio * ws[n] * d;
        }
        placed += w;
      }
      const float keep = 1.f - fraction * std::min(1.f, placed);
      c[2] *= keep;
      c[3] *= keep;
      // fine pixels of the cell: deposited at their world position, nearest target pixel
      for (int py = cy * static_cast<int>(k); py < (cy + 1) * static_cast<int>(k); ++py) {
        for (int px = cx * static_cast<int>(k); px < (cx + 1) * static_cast<int>(k); ++px) {
          const std::size_t i = zs(py) * zs(S) + zs(px);
          const float fh = ft[i] * fraction * heat_gain, fdd = fd[i] * fraction * soot_gain;
          const float pwx = from.at.x + (fl(px) + 0.5f) * sc, pwy = from.at.y + (fl(S - py) - 0.5f) * sc;
          for (std::size_t q = 0; q < to.size(); ++q) {
            Module* t = to[q];
            if (!reaches(q) || !t->active || t == &from) continue;
            const int tS = t->size();
            const float tsc = t->at.scale;
            const float tx = (pwx - t->at.x) / tsc, ty = fl(tS) - (pwy - t->at.y) / tsc;
            if (tx < 0.f || ty < 0.f || tx >= fl(tS) || ty >= fl(tS)) continue;
            const float w = t->weight_px(tx, ty);
            if (w <= 0.f) continue;
            const std::size_t j = zs(static_cast<int>(ty)) * zs(tS) + zs(static_cast<int>(tx));
            const float ratio = w * (sc * sc) / (tsc * tsc);
            (masked ? to_heat[q] : t->runner().fine_heat_mut().data())[j] += ratio * fh;
            (masked ? to_soot[q] : t->runner().fine_soot_mut().data())[j] += ratio * fdd;
          }
          ft[i] *= keep;
          fd[i] *= keep;
        }
      }
    }
  }
}

// --- light -------------------------------------------------------------------------------------------------------------

std::array<float, 3> heat_colour(float t) {
  static constexpr float stops[][4] = {{0.f, 0.25f, 0.02f, 0.0f},   {0.25f, 0.75f, 0.12f, 0.01f}, {0.5f, 1.0f, 0.35f, 0.04f},
                                       {0.75f, 1.0f, 0.62f, 0.15f}, {1.0f, 1.0f, 0.85f, 0.45f},   {1.4f, 1.0f, 0.97f, 0.85f}};
  t = std::max(0.f, t);
  if (t >= 1.4f) return {1.f, 0.97f, 0.85f};
  int i = 0;
  while (t > stops[i + 1][0]) ++i;
  const float u = (t - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
  return {stops[i][1] + u * (stops[i + 1][1] - stops[i][1]), stops[i][2] + u * (stops[i + 1][2] - stops[i][2]), stops[i][3] + u * (stops[i + 1][3] - stops[i][3])};
}

Light::Light(const FieldBus& bus) : x0_(bus.x0()), y0_(bus.y0()), cell_(bus.cell()), nx_(bus.nx()), ny_(bus.ny()) {
  L_.assign(zs(nx_) * zs(ny_) * 3, 0.f);
  int nx = nx_, ny = ny_;
  for (int l = 0; l < 6; ++l) {
    Level v;
    v.nx = nx;
    v.ny = ny;
    v.a.assign(zs(nx) * zs(ny) * 3, 0.f);
    v.b.assign(v.a.size(), 0.f);
    // where the finest cells' centres fall in this level (cell centres at integers), as bilerp() clamps them
    const float f = static_cast<float>(1u << l);
    for (int x = 0; x < nx_; ++x) {
      const Taps t = taps((fl(x) + 0.5f) / f - 0.5f, nx);
      v.x0.push_back(t.i0);
      v.x1.push_back(t.i1);
      v.fx.push_back(t.f);
    }
    for (int y = 0; y < ny_; ++y) {
      const Taps t = taps((fl(y) + 0.5f) / f - 0.5f, ny);
      v.y0.push_back(t.i0);
      v.y1.push_back(t.i1);
      v.fy.push_back(t.f);
    }
    if (l > 0) v.up.assign(zs(ny) * zs(nx_) * 3, 0.f);
    levels_.push_back(std::move(v));
    nx = std::max(1, (nx + 1) / 2);
    ny = std::max(1, (ny + 1) / 2);
  }
}

namespace {

// One row of the [1 4 6 4 1] / 16 blur along x of an RGB row of n cells (clamped at the ends), summed in the order of
// the taps.
void blur_row(const float* __restrict src, float* __restrict dst, int n) {
  static constexpr float kw[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
  const auto edge = [&](int x) {
    for (int ch = 0; ch < 3; ++ch) {
      float s = 0.f;
      for (int t = -2; t <= 2; ++t) s += kw[t + 2] * src[zs(std::clamp(x + t, 0, n - 1)) * 3 + zs(ch)];
      dst[zs(x) * 3 + zs(ch)] = s;
    }
  };
  for (int x = 0; x < std::min(2, n); ++x) edge(x);
  for (int i = 6; i < 3 * n - 6; ++i) {  // interior cells, channels interleaved: the taps are 3 floats apart
    float s = 0.f;
    s += kw[0] * src[i - 6];
    s += kw[1] * src[i - 3];
    s += kw[2] * src[i];
    s += kw[3] * src[i + 3];
    s += kw[4] * src[i + 6];
    dst[i] = s;
  }
  for (int x = std::max(2, n - 2); x < n; ++x) edge(x);
}

}  // namespace

void Light::update(const FieldBus& bus, float gain, std::array<float, 3> flash, Pool& pool) {
  // Every value is computed as a plain pass over the levels would compute it, in the same order of operations, so the
  // light does not depend on the number of threads. Rows are spread over the pool in four passes.
  flash_ = flash;
  static constexpr float kw[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
  const int L = static_cast<int>(levels_.size());
  auto heat = bus.heat();
  auto glow = bus.glow();
  const bool glowing = bus.glowing();
  // 1. the finest level: each cell's heat emits; then the blur along x, row by row
  Level& l0 = levels_[0];
  const int chunk = 8;
  pool.run((ny_ + chunk - 1) / chunk, [&](int task) {
    for (int y = task * chunk; y < std::min(ny_, (task + 1) * chunk); ++y) {
      float* a = l0.a.data() + zs(y) * zs(nx_) * 3;
      for (int x = 0; x < nx_; ++x) {
        const std::size_t i = zs(y) * zs(nx_) + zs(x);
        if (glowing) {  // modules of their own colour (Module::glows): their heat in it, the rest in the fire colours
          const float* g = glow.data() + i * 4;
          const float h = std::max(0.f, heat[i] - g[0]), hg = std::max(0.f, g[0]);
          const auto c = heat_colour(h / 1.2f);
          const float e = gain * h * h;
          for (int ch = 0; ch < 3; ++ch) a[zs(x) * 3 + zs(ch)] = e * c[zs(ch)] + gain * hg * std::max(0.f, g[1 + ch]);
          continue;
        }
        const float h = std::max(0.f, heat[i]);
        const auto c = heat_colour(h / 1.2f);
        const float e = gain * h * h;
        for (int ch = 0; ch < 3; ++ch) a[zs(x) * 3 + zs(ch)] = e * c[zs(ch)];
      }
      blur_row(a, l0.b.data() + zs(y) * zs(nx_) * 3, nx_);
    }
  });
  // 2. the coarser levels (small): 2x2 boxes down from the unblurred level above (amounts add up: coarser levels carry
  // the total), then the blur along x
  for (int l = 1; l < L; ++l) {
    const Level& s = levels_[zs(l - 1)];
    Level& d = levels_[zs(l)];
    for (int y = 0; y < d.ny; ++y) {
      const float* r0 = s.a.data() + zs(std::min(2 * y, s.ny - 1)) * zs(s.nx) * 3;
      const float* r1 = s.a.data() + zs(std::min(2 * y + 1, s.ny - 1)) * zs(s.nx) * 3;
      float* o = d.a.data() + zs(y) * zs(d.nx) * 3;
      for (int x = 0; x < d.nx; ++x) {
        const std::size_t p = zs(std::min(2 * x, s.nx - 1)) * 3, q = zs(std::min(2 * x + 1, s.nx - 1)) * 3;
        for (int ch = 0; ch < 3; ++ch) o[zs(x) * 3 + zs(ch)] = 0.f + r0[p + zs(ch)] + r0[q + zs(ch)] + r1[p + zs(ch)] + r1[q + zs(ch)];
      }
    }
    for (int y = 0; y < d.ny; ++y) blur_row(d.a.data() + zs(y) * zs(d.nx) * 3, d.b.data() + zs(y) * zs(d.nx) * 3, d.nx);
  }
  // 3. the blur along y of every row of every level (b into a), and each coarser row resampled at the finest columns
  int rows = 0;
  for (const Level& v : levels_) rows += v.ny;
  pool.run(rows, [&](int task) {
    int l = 0, y = task;
    while (y >= levels_[zs(l)].ny) y -= levels_[zs(l++)].ny;
    Level& v = levels_[zs(l)];
    const std::size_t n = zs(v.nx) * 3;
    float* o = v.a.data() + zs(y) * n;
    const float* src[5];
    for (int t = -2; t <= 2; ++t) src[t + 2] = v.b.data() + zs(std::clamp(y + t, 0, v.ny - 1)) * n;
    for (std::size_t i = 0; i < n; ++i) {
      float s = 0.f;
      for (int t = 0; t < 5; ++t) s += kw[t] * src[t][i];
      o[i] = s;
    }
    if (l == 0) return;
    float* u = v.up.data() + zs(y) * zs(nx_) * 3;  // first half of the bilinear sample: along x
    for (int x = 0; x < nx_; ++x) {
      const float fx = v.fx[zs(x)];
      const float* a = o + zs(v.x0[zs(x)]) * 3;
      const float* b = o + zs(v.x1[zs(x)]) * 3;
      for (int ch = 0; ch < 3; ++ch) u[zs(x) * 3 + zs(ch)] = (1.f - fx) * a[ch] + fx * b[ch];
    }
  });
  // 4. light = sum over levels of the blurred amounts per area of that level's cell: a soft falloff with a long tail.
  // The finest level is sampled at its own centres (bilinear weights 0 and 1: the cells themselves).
  pool.run((ny_ + chunk - 1) / chunk, [&](int task) {
    const std::size_t n = zs(nx_) * 3;
    for (int y = task * chunk; y < std::min(ny_, (task + 1) * chunk); ++y) {
      float* __restrict o = L_.data() + zs(y) * n;
      const float* __restrict a = l0.a.data() + zs(y) * n;
      const float s0 = 0.6f;  // the finest level's scale (1 / 1 * 0.6, as below)
      for (std::size_t i = 0; i < n; ++i) o[i] = 0.f + s0 * a[i];
      for (int l = 1; l < L; ++l) {
        const Level& v = levels_[zs(l)];
        const float scale = 1.f / static_cast<float>(1u << (2 * l)) * 0.6f;  // per unit area of the level's cells
        const float fy = v.fy[zs(y)];
        const float* __restrict u0 = v.up.data() + zs(v.y0[zs(y)]) * n;  // second half of the bilinear sample: along y
        const float* __restrict u1 = v.up.data() + zs(v.y1[zs(y)]) * n;
        for (std::size_t i = 0; i < n; ++i) o[i] += scale * ((1.f - fy) * u0[i] + fy * u1[i]);
      }
    }
  });
}

std::array<float, 3> Light::at(float x, float y) const {
  const auto l = bilerp_n<3>(L_.data(), nx_, ny_, 3, (x - x0_) / cell_ - 0.5f, (y - y0_) / cell_ - 0.5f);
  return {l[0] + flash_[0], l[1] + flash_[1], l[2] + flash_[2]};
}

// --- particles ---------------------------------------------------------------------------------------------------------

Particles::Particles(int capacity) {
  for (auto* v : {&x_, &y_, &px_, &py_, &vx_, &vy_, &temp_, &size_, &life_, &age_, &cool_}) v->assign(zs(capacity), 0.f);
  kind_.assign(zs(capacity), Kind::ember);
  landed_.assign(zs(capacity), Landing{});
}

float Particles::uniform() {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return static_cast<float>(rng >> 40) * (1.f / 16777216.f);
}

bool Particles::spawn(Kind k, float x, float y, float vx, float vy, float temp, float size, float life) {
  if (n_ >= capacity()) return false;
  const std::size_t i = zs(n_++);
  kind_[i] = k;
  x_[i] = px_[i] = x;
  y_[i] = py_[i] = y;
  vx_[i] = vx;
  vy_[i] = vy;
  temp_[i] = temp;
  size_[i] = size;
  life_[i] = life;
  age_[i] = 0.f;
  cool_[i] = k == Kind::spark ? 4.f + 4.f * uniform() : k == Kind::ember ? 0.35f + 0.9f * uniform() : 1.2f;
  return true;
}

void Particles::kill(int i) {
  const int last = --n_;
  if (i == last) return;
  for (auto* v : {&x_, &y_, &px_, &py_, &vx_, &vy_, &temp_, &size_, &life_, &age_, &cool_}) (*v)[zs(i)] = (*v)[zs(last)];
  kind_[zs(i)] = kind_[zs(last)];
}

void Particles::update(float dt, const FieldBus* bus, float flow_gain, float ground_y) {
  n_landed_ = 0;
  for (int i = 0; i < n_;) {
    const std::size_t q = zs(i);
    age_[q] += dt;
    if (age_[q] > life_[q]) {
      kill(i);
      continue;
    }
    const Kind k = kind_[q];
    const float gravity = k == Kind::spark ? 380.f : k == Kind::ember ? 70.f : k == Kind::debris ? 900.f : 30.f;
    const float drag = k == Kind::spark ? 1.5f : k == Kind::ember ? 1.1f : k == Kind::debris ? 0.25f : 3.f;
    float fu = 0.f, fv = 0.f;
    if (bus) {
      const auto s = bus->at(x_[q], y_[q]);
      fu = s.u * 30.f * flow_gain;  // bus flow is per frame at 30 fps
      fv = s.v * 30.f * flow_gain;
    }
    vx_[q] += (fu - vx_[q]) * std::min(1.f, drag * dt);
    vy_[q] += (fv - vy_[q]) * std::min(1.f, drag * dt) + gravity * dt;
    if (k == Kind::ember || k == Kind::flake) {  // small turbulent kicks, so they do not settle on one streamline
      vx_[q] += (uniform() - 0.5f) * 240.f * dt;
      vy_[q] += (uniform() - 0.5f) * 240.f * dt;
    }
    px_[q] = x_[q];
    py_[q] = y_[q];
    x_[q] += vx_[q] * dt;
    y_[q] += vy_[q] * dt;
    temp_[q] *= std::exp(-cool_[q] * dt);
    if (y_[q] >= ground_y) {
      y_[q] = ground_y;
      if ((k == Kind::ember || k == Kind::debris) && vy_[q] > 30.f && temp_[q] > 0.25f && n_landed_ < static_cast<int>(landed_.size())) {
        landed_[zs(n_landed_++)] = {x_[q], temp_[q], k};
      }
      if (k == Kind::flake || k == Kind::spark) {
        kill(i);
        continue;
      }
      if (k == Kind::debris && vy_[q] > 120.f) {
        vy_[q] = -0.3f * vy_[q];
        vx_[q] *= 0.6f;
      } else {
        vy_[q] = 0.f;
        vx_[q] *= 0.5f;
      }
    }
    ++i;
  }
}

void Particles::copy_to(Particles& to) const {
  const auto n = static_cast<std::ptrdiff_t>(n_);
  for (auto [src, dst] : {std::pair{&x_, &to.x_}, std::pair{&y_, &to.y_}, std::pair{&px_, &to.px_}, std::pair{&py_, &to.py_}, std::pair{&temp_, &to.temp_},
                          std::pair{&size_, &to.size_}, std::pair{&life_, &to.life_}, std::pair{&age_, &to.age_}}) {
    std::copy(src->begin(), src->begin() + n, dst->begin());
  }
  std::copy(kind_.begin(), kind_.begin() + n, to.kind_.begin());
  to.n_ = n_;
}

void Particles::draw(Image4& screen, float cam_x, float cam_y) const {
  const int W = screen.w, H = screen.h;
  const auto add = [&](float x, float y, float r, float g, float b, float a) {  // bilinear splat
    const int x0 = ifloor(x - 0.5f), y0 = ifloor(y - 0.5f);
    const float fx = x - 0.5f - fl(x0), fy = y - 0.5f - fl(y0);
    const float w[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
    for (int q = 0; q < 4; ++q) {
      const int xx = x0 + (q & 1), yy = y0 + (q >> 1);
      if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
      float* p = screen.row(yy) + zs(xx) * 4;
      if (a > 0.f) {  // over (dark particles)
        const float k = 1.f - a * w[q];
        p[0] = p[0] * k + w[q] * r;
        p[1] = p[1] * k + w[q] * g;
        p[2] = p[2] * k + w[q] * b;
        p[3] = 1.f - (1.f - p[3]) * k;
      } else {  // add (glow)
        p[0] += w[q] * r;
        p[1] += w[q] * g;
        p[2] += w[q] * b;
      }
    }
  };
  for (int i = 0; i < n_; ++i) {
    const std::size_t q = zs(i);
    const float x = x_[q] - cam_x, y = y_[q] - cam_y, ox = px_[q] - cam_x, oy = py_[q] - cam_y;
    if (std::max(x, ox) < -4.f || std::min(x, ox) > fl(W) + 4.f || std::max(y, oy) < -4.f || std::min(y, oy) > fl(H) + 4.f) continue;
    const float fade = std::min(1.f, (life_[q] - age_[q]) * 4.f);
    const Kind k = kind_[q];
    if (k == Kind::spark || k == Kind::ember) {
      const float t = temp_[q];
      const auto c = heat_colour(t * 1.3f);
      const float bright = (k == Kind::spark ? 9.f : 3.2f) * t * t * fade * size_[q];
      const float len = std::hypot(x - ox, y - oy);
      const int n = std::max(1, static_cast<int>(std::ceil(len)));
      for (int s = 0; s < n; ++s) {
        const float u = (fl(s) + 0.5f) / fl(n);
        add(ox + u * (x - ox), oy + u * (y - oy), bright * c[0] / fl(n), bright * c[1] / fl(n), bright * c[2] / fl(n), 0.f);
      }
    } else {
      const float a = (k == Kind::debris ? 0.95f : 0.55f) * fade;
      const float glow = k == Kind::debris ? 1.5f * temp_[q] * temp_[q] : 0.f;
      const auto c = heat_colour(temp_[q]);
      const float base = k == Kind::debris ? 0.015f : 0.03f;
      const int r = std::max(1, static_cast<int>(size_[q]));
      for (int dy = -r / 2; dy <= r / 2; ++dy) {
        for (int dx = -r / 2; dx <= r / 2; ++dx) add(x + fl(dx), y + fl(dy), a * (base + glow * c[0]), a * (base + glow * c[1]), a * (base + glow * c[2]), a);
      }
    }
  }
}

// --- the frame ---------------------------------------------------------------------------------------------------------

float Shock::radius(float t) const {
  const float s = t - t0;
  if (s <= 0.f) return 0.f;
  return speed * decay * (1.f - std::exp(-s / decay)) + 120.f * s;  // fast at first, then sound-like
}

namespace {

// The taps bilerp() takes at continuous cell coordinate v on an axis of n cells (clamped as it clamps).
Frame::Tap tap(float v, int n) {
  v = std::clamp(v, 0.f, fl(n - 1));
  const int i0 = std::min(static_cast<int>(v), std::max(0, n - 2));
  const float f = v - fl(i0);
  return {i0, std::min(i0 + 1, n - 1), 1.f - f, f};
}

constexpr int kChunk = 8;    // screen rows per task
constexpr int kBlock = 256;  // pixels per block of a row, for scratch on the stack

// A pixel's four channels as one vector (GCC vector extensions, as in src/common/simd_kernels.hpp). Each lane does
// what the scalar code does to its channel, in the same order, so the results are the same to the bit.
typedef float px4 __attribute__((vector_size(16)));
typedef float px4u __attribute__((vector_size(16), aligned(4)));  // unaligned access
[[gnu::always_inline]] inline px4 load4(const float* p) { return *reinterpret_cast<const px4u*>(p); }
[[gnu::always_inline]] inline void store4(float* p, px4 v) { *reinterpret_cast<px4u*>(p) = v; }

// A bilinear lookup of four channels by taps, in bilerp()'s order of operations (to the bit).
[[gnu::always_inline]] inline px4 blend4(const float* a, const float* b, const float* c, const float* d, const Frame::Tap& x, const Frame::Tap& y) {
  return y.w0 * (x.w0 * load4(a) + x.w1 * load4(b)) + y.w1 * (x.w0 * load4(c) + x.w1 * load4(d));
}

double ms_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

Frame::Frame(int width, int height) : w_(width), h_(height), blocks_((width + kBlock - 1) / kBlock), avx2_(best_isa() != Isa::base) {
  screen_.allocate(width, height);
  tmp_.allocate(width, height);
  int w = (width + 1) / 2, h = (height + 1) / 2;
  for (int l = 0; l < 6; ++l) {
    Image4 a, b;
    a.allocate(w, h);
    b.allocate(w, h);
    mips_.push_back(std::move(a));
    mips_tmp_.push_back(std::move(b));
    w = std::max(1, (w + 1) / 2);
    h = std::max(1, (h + 1) / 2);
  }
  cols_.resize(zs(width));
  star_runs_.reserve(zs(width));
  fin_vig_.resize(zs(width));
  fin_key_.resize(zs(width));
  tile_cols_.resize(zs(kMaxTiles) * zs(width));
  seen_cols_.resize(zs(kMaxTiles));
  moved_.assign(zs(height) * zs(blocks_), {0, 0});
  // bloom's way up: level l of the mips read at half the coordinates of the next finer level (the screen for l = 0)
  for (std::size_t l = 0; l < mips_.size(); ++l) {
    const Image4& c = mips_[l];
    const int fw = l == 0 ? w_ : mips_[l - 1].w, fh = l == 0 ? h_ : mips_[l - 1].h;
    std::vector<Tap> tx(zs(fw)), ty(zs(fh));
    for (int x = 0; x < fw; ++x) tx[zs(x)] = tap((fl(x) + 0.5f) / 2.f - 0.5f, c.w);
    for (int y = 0; y < fh; ++y) ty[zs(y)] = tap((fl(y) + 0.5f) / 2.f - 0.5f, c.h);
    up_x_.push_back(std::move(tx));
    up_y_.push_back(std::move(ty));
  }
  reserve(16, 16, 16, 0);
}

void Frame::reserve(int modules, int shocks, int scorch, int particles) {
  tiles_.reserve(zs(modules));
  order_.reserve(zs(modules));
  groups_.reserve(zs(modules));
  shocks_.reserve(zs(shocks));
  scorch_.reserve(zs(scorch));
  if (parts_.capacity() < particles) parts_ = Particles(particles);
}

// --- background -------------------------------------------------------------------------------------------------------

namespace {
// The light at taps of its grid (a cell to a vector), as Light::at() computes it.
px4 light_at(const float* L, int lnx, const px4& flash4, const Frame::Tap& x, const Frame::Tap& y) {
  const float* r0 = L + zs(y.i0) * zs(lnx) * 4;
  const float* r1 = L + zs(y.i1) * zs(lnx) * 4;
  return blend4(r0 + zs(x.i0) * 4, r0 + zs(x.i1) * 4, r1 + zs(x.i0) * 4, r1 + zs(x.i1) * 4, x, y) + flash4;
}
}  // namespace

Frame::LightView Frame::light_view(const Light& light) {
  // The light is looked up as Light::at() does it, by taps (its columns once per screen column, its rows once per row)
  // in a copy with a fourth channel, so that a cell is one vector.
  const auto field = light.field();
  const int lnx = light.nx(), lny = light.ny();
  light4_.resize(zs(lnx) * zs(lny) * 4);  // the same size every frame: allocates on the first only
  for (std::size_t i = 0; i < zs(lnx) * zs(lny); ++i) {
    for (std::size_t c = 0; c < 3; ++c) light4_[i * 4 + c] = field[i * 3 + c];
    light4_[i * 4 + 3] = 0.f;
  }
  return {light4_.data(), lnx, lny, light.x0(), light.y0(), light.cell(), light.flash()};
}

void Frame::background_columns(const Params& P, const LightView& L) {
  const px4 flash4{L.flash[0], L.flash[1], L.flash[2], 0.f};
  for (int x = 0; x < w_; ++x) {  // per column: the hills' outline and their colour, the light's columns, hash keys
    Column& k = cols_[zs(x)];
    k.wx = P.cam_x + fl(x) + 0.5f;
    k.hill = P.ground_y - (18.f + 26.f * (0.5f + 0.5f * std::sin(k.wx * 0.0042f + 1.3f)) * (0.6f + 0.4f * std::sin(k.wx * 0.011f + 0.4f)));
    k.light = tap((k.wx - L.x0) / L.cell - 0.5f, L.nx);
    const px4 l = light_at(L.L4, L.nx, flash4, k.light, tap((k.hill - L.y0) / L.cell - 0.5f, L.ny));  // distant hills: a dark silhouette, faintly lit
    k.hill_colour = {0.006f + 0.05f * l[0], 0.007f + 0.05f * l[1], 0.012f + 0.05f * l[2], 1.f};
    k.star = hx(ifloor(k.wx / 3.f));
    k.tex = hx(ifloor(k.wx / 2.f));
  }
  // runs of columns with the same star key (about 3 columns each): a row's stars are found once per run
  star_runs_.clear();
  for (int x = 0; x < w_;) {
    int e = x + 1;
    while (e < w_ && cols_[zs(e)].star == cols_[zs(x)].star) ++e;
    star_runs_.push_back({x, e});
    x = e;
  }
}

Frame::Tap Frame::light_tap(int y, const Params& P, const LightView& L) const {
  const float wy = P.cam_y + fl(y) + 0.5f;
  // the sky at its own height; the ground from just above it (higher towards the viewer)
  const float ly = wy < P.ground_y ? wy : P.ground_y - 6.f - 30.f * std::clamp((wy - P.ground_y) / 160.f, 0.f, 1.f);
  return tap((ly - L.y0) / L.cell - 0.5f, L.ny);
}

void Frame::light_columns(const Params& P, const LightView& L, Pool& pool) {
  light_x_.resize(zs(L.ny) * zs(w_) * 4);  // the same size every frame: allocates on the first only
  int lo = L.ny, hi = -1;
  for (int y = 0; y < h_; ++y) {
    const Tap t = light_tap(y, P, L);
    lo = std::min(lo, t.i0);
    hi = std::max(hi, t.i1);
  }
  if (lo > hi) return;
  const int rows = hi - lo + 1;
  pool.run((rows + kChunk - 1) / kChunk, [&](int task) {
    for (int j = lo + task * kChunk; j < std::min(hi + 1, lo + (task + 1) * kChunk); ++j) {
      const float* r = L.L4 + zs(j) * zs(L.nx) * 4;
      float* o = light_x_.data() + zs(j) * zs(w_) * 4;
      for (int x = 0; x < w_; ++x) {
        const Tap& t = cols_[zs(x)].light;
        store4(o + zs(x) * 4, t.w0 * load4(r + zs(t.i0) * 4) + t.w1 * load4(r + zs(t.i1) * 4));
      }
    }
  });
}

void Frame::background_row(int y, const Params& P, const LightView& L, std::span<const std::array<float, 4>> scorch, StarCache& cache) {
  const px4 flash4{L.flash[0], L.flash[1], L.flash[2], 0.f};
  const auto star_run = [&](int x0, int x1, std::uint32_t star_y, px4 sky, float u, float wy_, float* row_, const auto& light_at_x_, float now_) {
    for (int x = x0; x < x1; ++x) {
      const Column& k = cols_[zs(x)];
      if (wy_ >= k.hill) continue;
      px4 v = sky;
      const float h = unit(k.star ^ star_y ^ hz(7));
      const float tw = 0.6f + 0.4f * std::sin(now_ * (2.f + 6.f * unit(k.star ^ star_y ^ hz(9))) + 20.f * h);
      const float s = (h - 0.9965f) / 0.0035f * 0.35f * tw * u;
      v += px4{s, s, 1.1f * s, 0.f};
      v += 0.08f * light_at_x_(x);
      v[3] = 1.f;
      store4(row_ + zs(x) * 4, v);
    }
  };
  float* row = screen_.row(y);
  const float wy = P.cam_y + fl(y) + 0.5f;
  const float ground = P.ground_y, now = P.time;
  // the light, bilinear as Light::at(): its rows resampled at the screen's columns (light_columns()), blended here
  const Tap ly = light_tap(y, P, L);
  const float* lx0 = light_x_.data() + zs(ly.i0) * zs(w_) * 4;
  const float* lx1 = light_x_.data() + zs(ly.i1) * zs(w_) * 4;
  const auto light_at_x = [&](int x) { return ly.w0 * load4(lx0 + zs(x) * 4) + ly.w1 * load4(lx1 + zs(x) * 4) + flash4; };
  if (wy < ground) {
    // the hills, and above them the sky: dark blue, lighter at the horizon, a few stars fixed in the world, and a
    // little glow where the light is
    const float u = std::clamp((ground - wy) / 900.f, 0.f, 1.f);
    const px4 sky{0.010f + 0.020f * (1.f - u), 0.013f + 0.024f * (1.f - u), 0.030f + 0.035f * (1.f - u), 1.f};
    const std::uint32_t star_y = hy(ifloor(wy / 3.f));
    for (int x = 0; x < w_; ++x) {  // the sky without stars
      const Column& k = cols_[zs(x)];
      float* p = row + zs(x) * 4;
      if (wy >= k.hill) {
        store4(p, load4(k.hill_colour.data()));
        continue;
      }
      px4 v = sky + 0.08f * light_at_x(x);
      v[3] = 1.f;
      store4(p, v);
    }
    // the stars: a pixel's star value depends on its run of columns and its band of rows (star_y), so it is found
    // once per run (and kept for the rows of the band); a star pixel is then drawn as a whole, as without the runs
    if (cache.star_y != star_y || !cache.valid) {
      cache.star_y = star_y;
      cache.valid = true;
      cache.n = 0;
      for (const auto& r : star_runs_) {
        if (unit(cols_[zs(r[0])].star ^ star_y ^ hz(7)) > 0.9965f && cache.n < static_cast<int>(cache.runs.size())) cache.runs[zs(cache.n++)] = r;
      }
      if (cache.n == static_cast<int>(cache.runs.size())) {  // (many stars: all of them, without the cache)
        cache.valid = false;
        cache.n = 0;
        for (const auto& r : star_runs_)
          if (unit(cols_[zs(r[0])].star ^ star_y ^ hz(7)) > 0.9965f) star_run(r[0], r[1], star_y, sky, u, wy, row, light_at_x, now);
      }
    }
    for (int q = 0; q < cache.n; ++q) star_run(cache.runs[zs(q)][0], cache.runs[zs(q)][1], star_y, sky, u, wy, row, light_at_x, now);
    return;
  }
  // ground: dark earth, lit by the scene's light from just above it, darker towards the viewer
  const float depth = std::clamp((wy - ground) / 160.f, 0.f, 1.f);
  const float tex_amp = 0.5f + 0.5f * depth, shade = 1.f - 0.55f * depth;
  const std::uint32_t tex_y = hy(ifloor(wy / 2.f)) ^ hz(3);
  const px4 earth{0.012f, 0.011f, 0.010f, 0.f}, tint{0.9f, 0.75f, 0.6f, 0.f};
  float tex = 0.f;
  for (int x = 0; x < w_; ++x) {
    const Column& k = cols_[zs(x)];
    if (x == 0 || k.tex != cols_[zs(x - 1)].tex) tex = 0.75f + 0.5f * unit(k.tex ^ tex_y) * tex_amp;  // per run of columns
    const float lit = shade * tex;
    const px4 l = light_at_x(x);
    px4 v = (earth + 0.35f * l / (1.f + 0.6f * l)) * lit * tint;
    v[3] = 1.f;
    store4(row + zs(x) * 4, v);
  }
  for (const auto& s : scorch) {  // scorch marks (x, y, radius, glow), over the columns each covers in this row
    const float dy = (wy - s[1]) / (0.28f * s[2]);
    if (!(dy * dy < 1.f)) continue;
    const float half = s[2] * std::sqrt(1.f - dy * dy);  // with a margin below: the test per pixel decides
    const int xa = std::max(0, ifloor(s[0] - half - P.cam_x) - 2), xb = std::min(w_, ifloor(s[0] + half - P.cam_x) + 3);
    for (int x = xa; x < xb; ++x) {
      const Column& k = cols_[zs(x)];
      const float dx = (k.wx - s[0]) / s[2];
      const float r2 = dx * dx + dy * dy;
      if (r2 >= 1.f) continue;
      float* p = row + zs(x) * 4;
      const float kk = 1.f - smooth01(r2);
      for (int c = 0; c < 3; ++c) p[c] *= 1.f - 0.85f * kk;
      const float n = 0.5f + 0.5f * value_noise(k.wx / 7.f, wy / 3.f, now * 0.4f, 5);  // smooth glowing patches
      const float g = s[3] * kk * kk * std::pow(n, 5.f) * 0.8f;
      const auto c = heat_colour(0.2f + 0.35f * n);
      p[0] += g * c[0];
      p[1] += g * c[1];
      p[2] += g * c[2];
    }
  }
}

void Frame::background(const Light& light, std::span<const std::array<float, 4>> scorch, Pool& pool) {
  moved_rows_ = bloom_pending_ = false;  // every pixel is written anew
  const Params P = params();
  const LightView L = light_view(light);
  background_columns(P, L);
  light_columns(P, L, pool);
  pool.run((h_ + kChunk - 1) / kChunk, [&](int task) {
    StarCache cache;
    for (int y = task * kChunk; y < std::min(h_, (task + 1) * kChunk); ++y) background_row(y, P, L, scorch, cache);
  });
}

// --- modules ----------------------------------------------------------------------------------------------------------

void Frame::take_tiles(std::span<Module* const> modules) {
  tiles_.clear();
  for (const Module* m : modules) {
    Tile t;
    t.img = &m->image();
    t.spans = m->drawn_spans().data();
    t.at = m->at;
    t.opacity = m->opacity;
    t.feather = m->feather;
    t.group = m->group;
    t.size = m->size();
    t.res = m->res();
    t.band = m->band;
    t.active = m->active;
    tiles_.push_back(t);
  }
}

void Frame::draw(std::span<Module* const> modules, Pool& pool) {
  settle(&pool);
  take_tiles(modules);
  compose(params(), nullptr, {}, pool);
}

void Frame::compose(const Params& P, const LightView* L, std::span<const std::array<float, 4>> scorch, Pool& pool) {
  // groups in order of first appearance; each drawn once
  order_.clear();
  groups_.clear();
  for (std::size_t i = 0; i < tiles_.size(); ++i) {
    const Tile& m = tiles_[i];
    if (!m.active || m.opacity <= 0.f) continue;
    bool seen = false;
    for (const Group& g : groups_) seen |= g.count > 0 && tiles_[zs(order_[zs(g.first)])].group == m.group;
    if (seen && m.group >= 0) continue;
    Group g;
    g.first = static_cast<int>(order_.size());
    if (m.group < 0) {
      order_.push_back(static_cast<int>(i));
      g.count = 1;
    } else {
      for (std::size_t j = 0; j < tiles_.size(); ++j) {
        const Tile& t = tiles_[j];
        if (t.active && t.opacity > 0.f && t.group == m.group && g.count < kMaxTiles) {
          order_.push_back(static_cast<int>(j));
          ++g.count;
        }
      }
    }
    groups_.push_back(g);
  }
  if (L) {
    moved_rows_ = bloom_pending_ = false;  // every pixel is written anew
    background_columns(P, *L);
    light_columns(P, *L, pool);
  }
  // Passes of whole groups, up to kMaxTiles tiles each (one pass unless the scene is large); the background goes
  // with the first. Every pixel gets the background, then each group over it in order, as stage by stage.
  std::size_t g0 = 0;
  bool first = true;
  while (g0 < groups_.size() || (first && L)) {
    std::size_t g1 = g0;
    int tiles = 0;
    while (g1 < groups_.size() && (g1 == g0 || tiles + groups_[g1].count <= kMaxTiles)) tiles += groups_[g1++].count;
    const int base = g1 > g0 ? groups_[g0].first : 0;  // batch position of a tile: its index in order_ minus base
    int rows0 = h_, rows1 = 0;
    for (std::size_t gi = g0; gi < g1; ++gi) {
      Group& g = groups_[gi];
      // screen rows covered by the group
      float top = 1e9f, bottom = -1e9f;
      for (int q = g.first; q < g.first + g.count; ++q) {
        const Tile& t = tiles_[zs(order_[zs(q)])];
        top = std::min(top, t.at.y - P.cam_y);
        bottom = std::max(bottom, t.at.y + fl(t.size) * t.at.scale - P.cam_y);
      }
      g.y0 = std::max(0, ifloor(top));
      g.y1 = std::min(h_, ifloor(bottom) + 1);
      if (g.y0 >= g.y1) continue;
      rows0 = std::min(rows0, g.y0);
      rows1 = std::max(rows1, g.y1);
      // Each tile's screen columns, once: its image columns, and the factors of its ownership weight that depend on x.
      // Module::weight_px() is a product of factors of x and of y; multiplied here in its order, the weight is the
      // same to the bit. Only the columns that see the image are kept (a run: the image column grows with the
      // screen's).
      for (int q = g.first; q < g.first + g.count; ++q) {
        const Tile& t = tiles_[zs(order_[zs(q)])];
        const float sc = t.at.scale, S = fl(t.size), R = fl(t.res), k = S / R;
        const int x0 = std::max(0, ifloor(t.at.x - P.cam_x)), x1 = std::min(w_, ifloor(t.at.x + S * sc - P.cam_x) + 1);
        int fc = x1, lc = x0 - 1;
        TileColumn* cols = tile_cols_.data() + zs(q - base) * zs(w_);
        for (int x = x0; x < x1; ++x) {
          const float ix = (P.cam_x + fl(x) + 0.5f - t.at.x) / sc - 0.5f;  // image column, continuous
          if (ix < -0.5f || ix > S - 0.5f) continue;
          fc = std::min(fc, x);
          lc = x;
          TileColumn& c = cols[zs(x)];
          const int ix0 = std::clamp(ifloor(ix), 0, t.size - 1);
          const float fx = std::clamp(ix - fl(ix0), 0.f, 1.f);
          c.t = {ix0, std::min(ix0 + 1, t.size - 1), 1.f - fx, fx};
          const float px = ix + 0.5f, cx = px / k - 0.5f;
          c.band = 1.f;
          if (t.band[0] > 0) c.band *= 1.f - band_weight(cx, t.band[0]);
          if (t.band[1] > 0) c.band *= band_weight(cx - (R - fl(t.band[1])), t.band[1]);
          c.feather_left = t.feather > 0.f && t.band[0] == 0 ? smooth01(px / t.feather) : 1.f;
          c.feather_right = t.feather > 0.f && t.band[1] == 0 ? smooth01((S - px) / t.feather) : 1.f;
          c.x_weight = c.band * c.feather_left * c.feather_right;
        }
        seen_cols_[zs(q - base)] = fc <= lc ? std::array<int, 2>{fc, lc + 1} : std::array<int, 2>{0, 0};
      }
    }
    const bool bg = first && L;
    if (bg) {
      rows0 = 0;
      rows1 = h_;
    }
    first = false;
    if (rows0 < rows1) {
      pool.run((rows1 - rows0 + kChunk - 1) / kChunk, [&](int task) {
        // A pixel whose four image pixels are all empty (every channel 0) adds exactly nothing, so each image row is
        // drawn only between its first and last pixel that is not empty. Rows found here are kept, two per tile: the
        // tiles are scaled up, so the next screen row mostly reads the same two image rows.
        struct Occupied {
          int row = -1, a = 0, b = 0;  // image columns [a, b) of image row `row`
        };
        std::array<std::array<Occupied, 2>, kMaxTiles> known{};
        const auto occupied = [&](int q, const Tile& t, int r) {
          Occupied& o = known[zs(q)][zs(r & 1)];
          if (o.row == r) return o;
          const float* p = t.img->row(r);
          const int w = t.size;
          const auto empty = [&](int x) {  // all four channels +0 (bits all 0; a -0 is drawn, which adds nothing either)
            std::uint64_t a = 0, b = 0;
            std::memcpy(&a, p + zs(x) * 4, 8);
            std::memcpy(&b, p + zs(x) * 4 + 2, 8);
            return (a | b) == 0;
          };
          // the scan starts from the span outside which the shader left every pixel +0
          const int lo = t.spans[2 * r], hi = t.spans[2 * r + 1];
          o = {r, lo, hi};
          while (o.a < hi && empty(o.a)) ++o.a;
          if (o.a >= hi) {  // all empty
            o.a = w;
            o.b = 0;
          }
          while (o.b > o.a && empty(o.b - 1)) --o.b;
          return o;
        };
        struct TileRow {
          Tap ty;
          float band_b = 1.f, band_t = 1.f, feather_b = 1.f, feather_t = 1.f;
          int xa = 0, xb = 0;  // screen columns drawn
        };
        std::array<TileRow, kMaxTiles> tr;
        StarCache stars;
        for (int y = rows0 + task * kChunk; y < std::min(rows1, rows0 + (task + 1) * kChunk); ++y) {
          if (bg) background_row(y, P, *L, scorch, stars);
          const float wy = P.cam_y + fl(y) + 0.5f;
          const float clip = smooth01((P.ground_y + 2.f - wy) / 3.f);  // the ground hides what is below it
          if (clip <= 0.f) continue;
          for (std::size_t gi = g0; gi < g1; ++gi) {
            const Group& g = groups_[gi];
            if (y < g.y0 || y >= g.y1) continue;
            int ux0 = w_, ux1 = 0;  // screen columns any tile draws in this row
            for (int q = g.first - base; q < g.first - base + g.count; ++q) {
              const Tile& t = tiles_[zs(order_[zs(q + base)])];
              TileRow& r = tr[zs(q)];
              r.xa = r.xb = 0;
              const float sc = t.at.scale, S = fl(t.size), R = fl(t.res), k = S / R;
              const float iy = (wy - t.at.y) / sc - 0.5f;  // image row (top to bottom), continuous
              if (iy < -0.5f || iy > S - 0.5f) continue;
              const int iy0 = std::clamp(ifloor(iy), 0, t.size - 1);
              const float fy = std::clamp(iy - fl(iy0), 0.f, 1.f);
              r.ty = {iy0, std::min(iy0 + 1, t.size - 1), 1.f - fy, fy};
              const Occupied o0 = occupied(q, t, r.ty.i0), o1 = occupied(q, t, r.ty.i1);
              const int a = std::min(o0.a, o1.a), b = std::max(o0.b, o1.b);
              if (a >= b) continue;
              // the screen columns whose image columns reach [a, b)
              const TileColumn* cols = tile_cols_.data() + zs(q) * zs(w_);
              const TileColumn* c0 = cols + seen_cols_[zs(q)][0];
              const TileColumn* c1 = cols + seen_cols_[zs(q)][1];
              const TileColumn* ca = std::partition_point(c0, c1, [&](const TileColumn& c) { return c.t.i1 < a; });
              const TileColumn* cb = std::partition_point(ca, c1, [&](const TileColumn& c) { return c.t.i0 < b; });
              r.xa = static_cast<int>(ca - cols);
              r.xb = static_cast<int>(cb - cols);
              if (r.xa >= r.xb) continue;
              ux0 = std::min(ux0, r.xa);
              ux1 = std::max(ux1, r.xb);
              // the factors of the ownership weight that depend on y (tile pixels, y up)
              const float py = S - (iy + 0.5f), cy = py / k - 0.5f;
              r.band_b = t.band[2] > 0 ? 1.f - band_weight(cy, t.band[2]) : 1.f;
              r.band_t = t.band[3] > 0 ? band_weight(cy - (R - fl(t.band[3])), t.band[3]) : 1.f;
              r.feather_b = t.feather > 0.f && t.band[2] == 0 ? smooth01(py / t.feather) : 1.f;
              r.feather_t = t.feather > 0.f && t.band[3] == 0 ? smooth01((S - py) / t.feather) : 1.f;
            }
            if (ux0 >= ux1) continue;
            float* acc = tmp_.row(y);
            std::fill(acc + zs(ux0) * 4, acc + zs(ux1) * 4, 0.f);
            bool any = false;
            for (int q = g.first - base; q < g.first - base + g.count; ++q) {
              const TileRow& r = tr[zs(q)];
              if (r.xa >= r.xb) continue;
              const Tile& t = tiles_[zs(order_[zs(q + base)])];
              const float opacity = t.opacity;
              const float* r0 = t.img->row(r.ty.i0);
              const float* r1 = t.img->row(r.ty.i1);
              const TileColumn* cols = tile_cols_.data() + zs(q) * zs(w_);
              // the weight, Module::weight_px() in its order of factors; where the factors of y are all 1, the product
              // is the factors of x' (a product with 1 is exact)
              const bool x_only = r.band_b == 1.f && r.band_t == 1.f && r.feather_b == 1.f && r.feather_t == 1.f;
              for (int x = r.xa; x < r.xb; ++x) {
                const TileColumn& c = cols[zs(x)];
                float w;
                if (x_only) {
                  w = c.x_weight;
                } else {
                  w = c.band * r.band_b;
                  w *= r.band_t;
                  w *= c.feather_left;
                  w *= c.feather_right;
                  w *= r.feather_b;
                  w *= r.feather_t;
                }
                w = w * opacity * clip;
                if (w <= 0.f) continue;
                float* o = acc + zs(x) * 4;
                store4(o, load4(o) + w * blend4(r0 + zs(c.t.i0) * 4, r0 + zs(c.t.i1) * 4, r1 + zs(c.t.i0) * 4, r1 + zs(c.t.i1) * 4, c.t, r.ty));
                any = true;
              }
            }
            if (!any) continue;
            float* s = screen_.row(y);
            for (int x = ux0; x < ux1; ++x) {  // over: colour only, the screen keeps its alpha
              const px4 o = load4(acc + zs(x) * 4);
              float* p = s + zs(x) * 4;
              px4 v = load4(p) * (1.f - std::clamp(o[3], 0.f, 1.f)) + o;
              v[3] = p[3];
              store4(p, v);
            }
          }
        }
      });
    }
    g0 = g1;
  }
}

void Frame::particles(const Particles& p) {
  settle(nullptr);
  p.draw(screen_, cam_x, cam_y);
}

// --- distortion -------------------------------------------------------------------------------------------------------

void Frame::distort(std::span<const Shock> shocks, const FieldBus& bus, Pool& pool) {
  const HeatView H{bus.heat().data(), bus.nx(), bus.ny(), bus.x0(), bus.y0(), bus.cell()};
  distort_impl(params(), shocks, H, pool);
}

void Frame::distort_impl(const Params& P, std::span<const Shock> shocks, const HeatView& H, Pool& pool) {
  settle(&pool);
  // The bus's heat is looked up as FieldBus::at() does it (nothing beyond a cell outside the bus), by taps: its columns
  // once per screen column, its rows once per row. The haze's terms of x alone are made once per column.
  const float* heat = H.heat;
  const int bnx = H.nx, bny = H.ny;
  const float now = P.time, haze_k = P.haze;
  for (int x = 0; x < w_; ++x) {
    Column& k = cols_[zs(x)];
    k.wx = P.cam_x + fl(x) + 0.5f;
    const float gx = (k.wx - H.x0) / H.cell - 0.5f;
    k.on_bus = !(gx < -1.f || gx > fl(bnx));
    k.bus = tap(gx, bnx);
    k.wobble = std::sin(k.wx * 0.045f + now * 1.7f);
    k.phase = k.wx * 0.07f - now * 8.f;
  }
  // The haze's heat lookup in two halves: the bus rows the screen reads, resampled at the screen's columns (once per
  // bus row), then blended per pixel. And the bus columns each block of pixels reads, for a test of a block's heat.
  heat_x_.resize(zs(bny) * zs(w_));  // the same size every frame: allocates on the first only
  block_bus_.resize(zs(blocks_));
  for (int b = 0; b < blocks_; ++b) {
    std::array<int, 2> r{bnx, -1};
    for (int x = b * kBlock; x < std::min(w_, (b + 1) * kBlock); ++x) {
      const Column& k = cols_[zs(x)];
      if (!k.on_bus) continue;
      r[0] = std::min(r[0], k.bus.i0);
      r[1] = std::max(r[1], k.bus.i1);
    }
    block_bus_[zs(b)] = r;
  }
  const auto bus_rows = [&](int y) {  // the bus rows the haze of screen row y reads (none: i0 > i1)
    const float gy = (P.cam_y + fl(y) + 0.5f + 22.f - H.y0) / H.cell - 0.5f;
    if (!(haze_k > 0.f) || gy < -1.f || gy > fl(bny)) return Tap{1, 0, 0.f, 0.f};
    return tap(gy, bny);
  };
  {
    int lo = bny, hi = -1;
    for (int y = 0; y < h_; ++y) {
      const Tap t = bus_rows(y);
      if (t.i0 > t.i1) continue;
      lo = std::min(lo, t.i0);
      hi = std::max(hi, t.i1);
    }
    if (lo <= hi) {
      pool.run((hi - lo + 1 + kChunk - 1) / kChunk, [&](int task) {
        for (int j = lo + task * kChunk; j < std::min(hi + 1, lo + (task + 1) * kChunk); ++j) {
          const float* r = heat + zs(j) * zs(bnx);
          float* o = heat_x_.data() + zs(j) * zs(w_);
          for (int x = 0; x < w_; ++x) {
            const Column& k = cols_[zs(x)];
            o[x] = k.on_bus ? k.bus.w0 * r[k.bus.i0] + k.bus.w1 * r[k.bus.i1] : 0.f;
          }
        }
      });
    }
  }
  // A pixel that does not move keeps its value, so only the pixels that move are computed (into tmp_, which holds a
  // block's span of them, [first, last] moved pixel) and copied back by the next stage that reads the screen.
  pool.run((h_ + kChunk - 1) / kChunk, [&](int task) {
    std::array<float, kBlock> dxs{}, dys{};
    for (int y = task * kChunk; y < std::min(h_, (task + 1) * kChunk); ++y) {
      float* out = tmp_.row(y);
      std::array<int, 2>* spans = moved_.data() + zs(y) * zs(blocks_);
      const float wy = P.cam_y + fl(y) + 0.5f;
      // hot air below shimmers what is seen through it
      const Tap by = bus_rows(y);
      const bool hazy_row = by.i0 <= by.i1;
      const float* hx0 = heat_x_.data() + zs(hazy_row ? by.i0 : 0) * zs(w_);
      const float* hx1 = heat_x_.data() + zs(hazy_row ? by.i1 : 0) * zs(w_);
      const float haze_y = wy * 0.09f + now * 11.f, wobble_y = 1.5f * std::sin(wy * 0.05f);
      for (int bx = 0, b = 0; bx < w_; bx += kBlock, ++b) {
        const int n = std::min(kBlock, w_ - bx);
        spans[b] = {0, 0};
        std::fill_n(dxs.begin(), n, 0.f);
        std::fill_n(dys.begin(), n, 0.f);
        // a block whose bus cells are all cool has no haze: a blend of corners all below 0.0099 stays below the
        // threshold, 0.01, rounding included
        bool hazy = hazy_row && block_bus_[zs(b)][0] <= block_bus_[zs(b)][1];
        if (hazy) {
          float most = -1e30f;
          for (int i = block_bus_[zs(b)][0]; i <= block_bus_[zs(b)][1]; ++i) most = std::max({most, heat[zs(by.i0) * zs(bnx) + zs(i)], heat[zs(by.i1) * zs(bnx) + zs(i)]});
          hazy = most >= 0.0099f;
        }
        bool moved = hazy;
        for (const Shock& s : shocks) {  // shock rings, over the columns each ring may cover in this row
          const float R = s.radius(now);
          if (R <= 0.f) continue;
          const float ey = wy - s.y, ay = std::fabs(ey);
          const float outer = R + s.width + 2.f, inner = R - s.width - 2.f;  // a margin: the test per pixel decides
          if (ay >= outer) continue;
          const float xo = std::sqrt(outer * outer - ey * ey), xi = inner > ay ? std::sqrt(inner * inner - ey * ey) : 0.f;
          const float centre = s.x - P.cam_x - 0.5f;  // screen column of the centre, continuous
          int sp2[2][2] = {{ifloor(centre - xo) - 1, ifloor(centre - xi) + 2}, {ifloor(centre + xi) - 1, ifloor(centre + xo) + 2}};
          if (sp2[1][0] <= sp2[0][1]) {  // the two spans meet: one
            sp2[0][1] = sp2[1][1];
            sp2[1][0] = sp2[1][1];
          }
          const float amp = s.amp * std::exp(-(now - s.t0) / (2.f * s.decay));
          for (const auto& sp : sp2) {
            const int xa = std::max(sp[0], bx), xb = std::min(sp[1], bx + n);
            for (int x = xa; x < xb; ++x) {
              const float ex = cols_[zs(x)].wx - s.x, r = std::sqrt(ex * ex + ey * ey) + 1e-3f;
              const float d = r - R;
              if (std::fabs(d) >= s.width) continue;
              const float a = amp * std::sin(std::numbers::pi_v<float> * d / s.width);
              dxs[zs(x - bx)] += a * ex / r;
              dys[zs(x - bx)] += a * ey / r;
              moved = true;
            }
          }
        }
        if (hazy) {
          for (int i = 0; i < n; ++i) {
            const Column& k = cols_[zs(bx + i)];
            if (!k.on_bus) continue;
            const float h = by.w0 * hx0[bx + i] + by.w1 * hx1[bx + i];  // blend() of the four cells, to the bit
            if (h > 0.01f) {
              const float a = haze_k * std::min(1.f, 1.6f * h);
              dxs[zs(i)] += a * 1.6f * std::sin(haze_y + 2.f * k.wobble);
              dys[zs(i)] += a * 1.1f * std::sin(k.phase + wobble_y);
            }
          }
        }
        if (!moved) continue;
        int a = n, e = 0;  // the block's first and last pixel that moves
        for (int i = 0; i < n; ++i) {
          if (dxs[zs(i)] != 0.f || dys[zs(i)] != 0.f) {
            a = std::min(a, i);
            e = i + 1;
          }
        }
        if (a >= e) continue;
        spans[b] = {bx + a, bx + e};
        const float* src_row = screen_.row(y);
        for (int i = a; i < e; ++i) {
          const int x = bx + i;
          const float dx = dxs[zs(i)], dy = dys[zs(i)];
          float* o = out + zs(x) * 4;
          if (dx == 0.f && dy == 0.f) {
            std::copy_n(src_row + zs(x) * 4, 4, o);
            continue;
          }
          const float sx = std::clamp(fl(x) + dx, 0.f, fl(w_ - 1)), sy = std::clamp(fl(y) + dy, 0.f, fl(h_ - 1));
          const int x0 = std::min(static_cast<int>(sx), w_ - 2), y0 = std::min(static_cast<int>(sy), h_ - 2);
          const float fx = sx - fl(x0), fy = sy - fl(y0);
          const float* pa = screen_.row(y0) + zs(x0) * 4;
          const float* pc = screen_.row(y0 + 1) + zs(x0) * 4;
          store4(o, (1.f - fy) * ((1.f - fx) * load4(pa) + fx * load4(pa + 4)) + fy * ((1.f - fx) * load4(pc) + fx * load4(pc + 4)));
        }
      }
    }
  });
  moved_rows_ = true;
}

void Frame::settle_row(int y) {
  const std::array<int, 2>* spans = moved_.data() + zs(y) * zs(blocks_);
  const float* src = tmp_.row(y);
  float* dst = screen_.row(y);
  for (int b = 0; b < blocks_; ++b) {
    if (spans[b][0] < spans[b][1]) std::copy(src + zs(spans[b][0]) * 4, src + zs(spans[b][1]) * 4, dst + zs(spans[b][0]) * 4);
  }
}

void Frame::settle(Pool* pool) {
  if (moved_rows_) {
    const auto rows = [&](int task) {
      for (int y = task * kChunk; y < std::min(h_, (task + 1) * kChunk); ++y) settle_row(y);
    };
    const int n = (h_ + kChunk - 1) / kChunk;
    if (pool) {
      pool->run(n, rows);
    } else {
      for (int t = 0; t < n; ++t) rows(t);
    }
    moved_rows_ = false;
  }
  if (bloom_pending_) {  // bloom's last pass, as finish() would do it on the way
    const Image4& m0 = mips_[0];
    const auto rows = [&](int task) {
      for (int y = task * kChunk; y < std::min(h_, (task + 1) * kChunk); ++y) {
        const Tap& ty = up_y_[0][zs(y)];
        const float* r0 = m0.row(ty.i0);
        const float* r1 = m0.row(ty.i1);
        const Tap* tx = up_x_[0].data();
        float* o = screen_.row(y);
        for (int x = 0; x < w_; ++x) {
          const Tap& t = tx[x];
          float* p = o + zs(x) * 4;
          store4(p, load4(p) + bloom_gain_ * blend4(r0 + zs(t.i0) * 4, r0 + zs(t.i1) * 4, r1 + zs(t.i0) * 4, r1 + zs(t.i1) * 4, t, ty) / bloom_div_);
        }
      }
    };
    const int n = (h_ + kChunk - 1) / kChunk;
    if (pool) {
      pool->run(n, rows);
    } else {
      for (int t = 0; t < n; ++t) rows(t);
    }
    bloom_pending_ = false;
  }
}

// --- tone mapping, and the row kernels --------------------------------------------------------------------------------

namespace {

// Tone mapping of n values into levels of the display table: ACES of the value times the exposure, vignette and fade,
// plus grain. The level is clamped before it is made an integer (the same level as clamping the integer for every value
// that can arrive here: finite, at most 1.003 * 4095 + 0.5, or NaN, which max(0, NaN) makes 0 as the integer's clamp
// did), so that the loop vectorises without integer min and max.
[[gnu::always_inline]] inline void tone(const float* p, const float* vig, const float* grain, float exposure, float fade, int* level, int n) {
  const auto aces = [](float v) { return std::clamp(v * (2.51f * v + 0.03f) / (v * (2.43f * v + 0.59f) + 0.14f), 0.f, 1.f); };
  for (int i = 0; i < n; ++i) {
    const float v = aces(p[i] * exposure * vig[i] * fade) + grain[i];
    level[i] = static_cast<int>(std::min(4095.f, std::max(0.f, v * 4095.f + 0.5f)));
  }
}

}  // namespace

void Frame::finish(std::span<std::uint8_t> rgb, Pool& pool) { finish_impl(params(), rgb, pool); }

// The row kernels of the picture stages that gain from wider vectors, compiled twice: for the baseline ISA and for AVX2
// without FMA. Both do the same IEEE operations on each value in the same order (only more values at once), so they give
// the same bits; the second is used when the runtime's ISA is AVX2 or better (best_isa()).
struct FrameKernels {
  struct Finish {
    const Frame::Params* P;
    std::uint8_t* rgb;
    std::uint32_t salt;
    bool settle, bloom;
    float gain, div;
  };
  // Wide: tone mapping two pixels to a vector of 8 (for AVX2; on the baseline ISA GCC splits such vectors badly, so
  // there it is a loop over values that GCC vectorises itself). The same operations on each value either way.
  template <bool Wide>
  [[gnu::always_inline]] static inline void finish_rows(Frame& F, const Finish& a, int y0, int y1) {
    typedef float f8 __attribute__((vector_size(32)));
    typedef float f8u __attribute__((vector_size(32), aligned(4)));
    typedef int i8 __attribute__((vector_size(32)));
    typedef float f4u __attribute__((vector_size(16), aligned(4)));
    const Gamma& g = gamma();
    const Image4& m0 = F.mips_[0];
    const float expo = a.P->exposure, fade_k = a.P->fade, gain = a.gain, div = a.div;
    const int w = F.w_, h = F.h_;
    alignas(32) std::array<float, kBlock + 4> vig, grain;  // per pixel of a block (padded: read four at a time)
    alignas(32) std::array<float, 4 * kBlock> vig_v, grain_v, sum;  // per value (not Wide); the screen plus bloom
    alignas(32) std::array<int, 4 * kBlock> level;  // display levels, all four channels
    const f8 zero{}, one = zero + 1.f, top = zero + 4095.f;
    for (int y = y0; y < y1; ++y) {
      if (a.settle) F.settle_row(y);
      const float* p = F.screen_.row(y);
      std::uint8_t* o = a.rgb + zs(y) * zs(w) * 3;
      const float ny = (fl(y) + 0.5f) / fl(h) - 0.5f, ny2 = ny * ny;
      const std::uint32_t key_y = hy(y) ^ a.salt;
      const Frame::Tap& ty = F.up_y_[0][zs(y)];
      const float* b0 = m0.row(ty.i0);
      const float* b1 = m0.row(ty.i1);
      for (int bx = 0; bx < w; bx += kBlock) {
        const int n = std::min(kBlock, w - bx);
        for (int i = 0; i < n; ++i) {
          vig[zs(i)] = 1.f - 0.45f * (F.fin_vig_[zs(bx + i)] + ny2);
          grain[zs(i)] = (unit(F.fin_key_[zs(bx + i)] ^ key_y) - 0.5f) * 0.006f;
        }
        const float* src = p + zs(bx) * 4;
        if (a.bloom) {
          const Frame::Tap* tx = F.up_x_[0].data() + bx;
          for (int i = 0; i < n; ++i) {
            const Frame::Tap& t = tx[i];
            store4(sum.data() + zs(i) * 4, load4(src + zs(i) * 4) + gain * blend4(b0 + zs(t.i0) * 4, b0 + zs(t.i1) * 4, b1 + zs(t.i0) * 4, b1 + zs(t.i1) * 4, t, ty) / div);
          }
          src = sum.data();
        }
        if constexpr (Wide) {
          // tone(), two pixels to a vector: ACES of the value times the exposure, vignette and fade, plus grain,
          // clamped as a float, then the level
          int i = 0;
          for (; i + 2 <= n; i += 2) {
            const f8 v = *reinterpret_cast<const f8u*>(src + zs(i) * 4);
            const f4u vq = *reinterpret_cast<const f4u*>(vig.data() + i), gq = *reinterpret_cast<const f4u*>(grain.data() + i);
            const f8 vv = __builtin_shufflevector(vq, vq, 0, 0, 0, 0, 1, 1, 1, 1), gg = __builtin_shufflevector(gq, gq, 0, 0, 0, 0, 1, 1, 1, 1);
            const f8 t = v * expo * vv * fade_k;
            f8 c = t * (2.51f * t + 0.03f) / (t * (2.43f * t + 0.59f) + 0.14f);
            c = c < zero ? zero : (one < c ? one : c);  // std::clamp(c, 0, 1)
            f8 x = (c + gg) * 4095.f + 0.5f;
            x = zero < x ? x : zero;  // std::max(0, x)
            x = x < top ? x : top;    // std::min(4095, x)
            *reinterpret_cast<i8*>(level.data() + zs(i) * 4) = __builtin_convertvector(x, i8);
          }
          for (; i < n; ++i) {
            const std::array<float, 4> v4{vig[zs(i)], vig[zs(i)], vig[zs(i)], vig[zs(i)]}, g4{grain[zs(i)], grain[zs(i)], grain[zs(i)], grain[zs(i)]};
            tone(src + zs(i) * 4, v4.data(), g4.data(), expo, fade_k, level.data() + zs(i) * 4, 4);
          }
        } else {
          for (int i = 0; i < n; ++i) {
            for (int c = 0; c < 4; ++c) {
              vig_v[zs(i) * 4 + zs(c)] = vig[zs(i)];
              grain_v[zs(i) * 4 + zs(c)] = grain[zs(i)];
            }
          }
          tone(src, vig_v.data(), grain_v.data(), expo, fade_k, level.data(), 4 * n);
        }
        std::uint8_t* ob = o + zs(bx) * 3;
        for (int q = 0; q < n; ++q) {
          for (int c = 0; c < 3; ++c) ob[zs(q) * 3 + zs(c)] = g.to_display[zs(level[zs(q) * 4 + zs(c)])];
        }
      }
    }
  }

  // --- bloom --------------------------------------------------------------------------------------------------------

  // Bright pass: mip 0 row y from screen rows 2y and 2y + 1 (copied back from the distortion first, if it moved them).
  [[gnu::always_inline]] static inline void bright_rows(Frame& F, float threshold, bool settle, int y0, int y1) {
    Image4& m0 = F.mips_[0];
    const int w = F.w_, h = F.h_;
    // a quarter of the bright part of a run of source pixels, by source row: k = 0.25 max(0, lum - threshold) /
    // max(lum, 1e-4), four pixels at a time (transposed to planes), each value as the scalar code computes it
    const auto bright = [&](const float* src, int n, float* k) {
      const px4 zero{}, eps{1e-4f, 1e-4f, 1e-4f, 1e-4f};
      int j = 0;
      for (; j + 4 <= n; j += 4) {
        const px4 a = load4(src + zs(j) * 4), b = load4(src + zs(j) * 4 + 4), c = load4(src + zs(j) * 4 + 8), d = load4(src + zs(j) * 4 + 12);
        const px4 t0 = __builtin_shufflevector(a, b, 0, 4, 1, 5), t1 = __builtin_shufflevector(c, d, 0, 4, 1, 5);
        const px4 t2 = __builtin_shufflevector(a, b, 2, 6, 3, 7), t3 = __builtin_shufflevector(c, d, 2, 6, 3, 7);
        const px4 R = __builtin_shufflevector(t0, t1, 0, 1, 4, 5), G = __builtin_shufflevector(t0, t1, 2, 3, 6, 7), B = __builtin_shufflevector(t2, t3, 0, 1, 4, 5);
        const px4 lum = 0.2126f * R + 0.7152f * G + 0.0722f * B;
        const px4 over = lum - threshold;
        const px4 num = zero < over ? over : zero;  // std::max(0, over)
        const px4 den = lum < eps ? eps : lum;      // std::max(lum, 1e-4)
        store4(k + j, 0.25f * (num / den));
      }
      for (; j < n; ++j) {
        const float* p = src + zs(j) * 4;
        const float lum = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
        k[j] = 0.25f * (std::max(0.f, lum - threshold) / std::max(lum, 1e-4f));
      }
    };
    for (int y = y0; y < y1; ++y) {
      if (settle) {
        F.settle_row(2 * y);
        if (2 * y + 1 < h) F.settle_row(2 * y + 1);
      }
      const float* r0 = F.screen_.row(std::min(2 * y, h - 1));
      const float* r1 = F.screen_.row(std::min(2 * y + 1, h - 1));
      float* o = m0.row(y);
      alignas(16) std::array<float, 2 * kBlock> k0, k1;
      for (int bx = 0; bx < m0.w; bx += kBlock) {
        const int n = std::min(kBlock, m0.w - bx), sx0 = 2 * bx, sn = std::min(2 * n, w - sx0);
        bright(r0 + zs(sx0) * 4, sn, k0.data());
        bright(r1 + zs(sx0) * 4, sn, k1.data());
        for (int i = 0; i < n; ++i) {
          const int ja = std::min(2 * i, sn - 1), jb = std::min(2 * i + 1, sn - 1);
          const float* a = r0 + zs(sx0 + ja) * 4;
          const float* b = r0 + zs(sx0 + jb) * 4;
          const float* c = r1 + zs(sx0 + ja) * 4;
          const float* d = r1 + zs(sx0 + jb) * 4;
          px4 v = k0[zs(ja)] * load4(a) + k0[zs(jb)] * load4(b) + k1[zs(ja)] * load4(c) + k1[zs(jb)] * load4(d);
          v[3] = 0.f;
          store4(o + zs(bx + i) * 4, v);
        }
      }
    }
  }
  // Level l from level l - 1: 2 x 2 boxes.
  [[gnu::always_inline]] static inline void down_rows(Frame& F, int l, int y0, int y1) {
    const Image4& s = F.mips_[zs(l - 1)];
    Image4& d = F.mips_[zs(l)];
    for (int y = y0; y < y1; ++y) {
      const float* r0 = s.row(std::min(2 * y, s.h - 1));
      const float* r1 = s.row(std::min(2 * y + 1, s.h - 1));
      float* o = d.row(y);
      for (int x = 0; x < d.w; ++x) {
        const std::size_t a = zs(std::min(2 * x, s.w - 1)) * 4, b = zs(std::min(2 * x + 1, s.w - 1)) * 4;
        store4(o + zs(x) * 4, 0.25f * (load4(r0 + a) + load4(r0 + b) + load4(r1 + a) + load4(r1 + b)));
      }
    }
  }
  static constexpr float kw[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
  // The [1 4 6 4 1] / 16 blur of level l across, into its scratch: the clamped ends, then the middle.
  [[gnu::always_inline]] static inline void blur_x_rows(Frame& F, int l, int y0, int y1) {
    const Image4& a = F.mips_[zs(l)];
    Image4& b = F.mips_tmp_[zs(l)];
    const int W = a.w;
    for (int y = y0; y < y1; ++y) {
      const float* __restrict s = a.row(y);
      float* __restrict o = b.row(y);
      const auto clamped = [&](int x) {
        for (int c = 0; c < 4; ++c) {
          float v = 0.f;
          for (int t = -2; t <= 2; ++t) v += kw[t + 2] * s[zs(std::clamp(x + t, 0, W - 1)) * 4 + zs(c)];
          o[zs(x) * 4 + zs(c)] = v;
        }
      };
      for (int x = 0; x < std::min(2, W); ++x) clamped(x);
      for (int x = std::max(2, W - 2); x < W; ++x) clamped(x);
      for (int i = 8; i < 4 * (W - 2); ++i) o[i] = kw[0] * s[i - 8] + kw[1] * s[i - 4] + kw[2] * s[i] + kw[3] * s[i + 4] + kw[4] * s[i + 8];
    }
  }
  // ... and down, from the scratch back into the level.
  [[gnu::always_inline]] static inline void blur_y_rows(Frame& F, int l, int y0, int y1) {
    Image4& a = F.mips_[zs(l)];
    const Image4& b = F.mips_tmp_[zs(l)];
    const int W = a.w;
    for (int y = y0; y < y1; ++y) {
      const float* r[5];
      for (int t = 0; t < 5; ++t) r[t] = b.row(std::clamp(y + t - 2, 0, a.h - 1));
      float* __restrict o = a.row(y);
      for (int i = 0; i < 4 * W; ++i) o[i] = kw[0] * r[0][i] + kw[1] * r[1][i] + kw[2] * r[2][i] + kw[3] * r[3][i] + kw[4] * r[4][i];
    }
  }
  // Level l - 1 plus level l, bilinear.
  [[gnu::always_inline]] static inline void up_rows(Frame& F, int l, int y0, int y1) {
    Image4& f = F.mips_[zs(l - 1)];
    const Image4& c = F.mips_[zs(l)];
    for (int y = y0; y < y1; ++y) {
      const Frame::Tap& ty = F.up_y_[zs(l)][zs(y)];
      const float* r0 = c.row(ty.i0);
      const float* r1 = c.row(ty.i1);
      const Frame::Tap* tx = F.up_x_[zs(l)].data();
      float* o = f.row(y);
      for (int x = 0; x < f.w; ++x) {
        const Frame::Tap& t = tx[x];
        float* p = o + zs(x) * 4;
        store4(p, load4(p) + 1.f * blend4(r0 + zs(t.i0) * 4, r0 + zs(t.i1) * 4, r1 + zs(t.i0) * 4, r1 + zs(t.i1) * 4, t, ty) / 1.f);
      }
    }
  }
};

namespace {

// Every kernel twice: for the baseline ISA and for AVX2 without FMA.
struct KernelSet {
  void (*finish)(Frame&, const FrameKernels::Finish&, int, int);
  void (*bright)(Frame&, float, bool, int, int);
  void (*down)(Frame&, int, int, int);
  void (*blur_x)(Frame&, int, int, int);
  void (*blur_y)(Frame&, int, int, int);
  void (*up)(Frame&, int, int, int);
};
#define NFX_FRAME_KERNELS(SUFFIX, WIDE, ...)                                                                                                         \
  __VA_ARGS__ void finish_##SUFFIX(Frame& F, const FrameKernels::Finish& a, int y0, int y1) { FrameKernels::finish_rows<WIDE>(F, a, y0, y1); } \
  __VA_ARGS__ void bright_##SUFFIX(Frame& F, float t, bool s, int y0, int y1) { FrameKernels::bright_rows(F, t, s, y0, y1); }            \
  __VA_ARGS__ void down_##SUFFIX(Frame& F, int l, int y0, int y1) { FrameKernels::down_rows(F, l, y0, y1); }                            \
  __VA_ARGS__ void blur_x_##SUFFIX(Frame& F, int l, int y0, int y1) { FrameKernels::blur_x_rows(F, l, y0, y1); }                        \
  __VA_ARGS__ void blur_y_##SUFFIX(Frame& F, int l, int y0, int y1) { FrameKernels::blur_y_rows(F, l, y0, y1); }                        \
  __VA_ARGS__ void up_##SUFFIX(Frame& F, int l, int y0, int y1) { FrameKernels::up_rows(F, l, y0, y1); }                                \
  const KernelSet kernels_##SUFFIX{finish_##SUFFIX, bright_##SUFFIX, down_##SUFFIX, blur_x_##SUFFIX, blur_y_##SUFFIX, up_##SUFFIX};
NFX_FRAME_KERNELS(base, false, [[gnu::noinline]])
#if defined(NFX_COMPOSE_AVX2)
NFX_FRAME_KERNELS(avx2, true, [[gnu::noinline, gnu::target("avx2")]])
#endif
#undef NFX_FRAME_KERNELS

const KernelSet& kernels(bool avx2) {
#if defined(NFX_COMPOSE_AVX2)
  if (avx2) return kernels_avx2;
#endif
  (void)avx2;
  return kernels_base;
}

}  // namespace

// --- bloom ------------------------------------------------------------------------------------------------------------

void Frame::bloom(float threshold, float strength, Pool& pool) { bloom_impl(threshold, strength, pool); }

void Frame::bloom_impl(float threshold, float strength, Pool& pool) {
  // Bright pass into mip 0 (half size), then down, blur, and up. Every pass works on all four channels; channel 3 of
  // the mips stays 0, so what it adds to the screen's alpha is 0. The distortion's moved pixels are copied back into
  // the screen on the way (the bright pass reads each screen row once), and the last pass, adding mip 0 to the
  // screen, is left to finish(), which reads the screen anyway. The rows are FrameKernels'.
  if (bloom_pending_) settle(&pool);  // (a second bloom without finish() between)
  const KernelSet& K = kernels(avx2_);
  const auto each_row = [&](const Image4& im, auto&& rows) {  // rows(y0, y1)
    if (zs(im.w) * zs(im.h) < 16384) {  // a small level: waking the workers would cost more than the work
      rows(0, im.h);
      return;
    }
    pool.run((im.h + kChunk - 1) / kChunk, [&](int task) { rows(task * kChunk, std::min(im.h, (task + 1) * kChunk)); });
  };
  const bool settle_rows = moved_rows_;
  each_row(mips_[0], [&](int y0, int y1) { K.bright(*this, threshold, settle_rows, y0, y1); });
  moved_rows_ = false;
  for (int l = 1; l < static_cast<int>(mips_.size()); ++l) each_row(mips_[zs(l)], [&](int y0, int y1) { K.down(*this, l, y0, y1); });
  for (int l = 0; l < static_cast<int>(mips_.size()); ++l) {
    each_row(mips_[zs(l)], [&](int y0, int y1) { K.blur_x(*this, l, y0, y1); });
    each_row(mips_[zs(l)], [&](int y0, int y1) { K.blur_y(*this, l, y0, y1); });
  }
  for (int l = static_cast<int>(mips_.size()) - 1; l > 0; --l) each_row(mips_[zs(l - 1)], [&](int y0, int y1) { K.up(*this, l, y0, y1); });
  // the last pass, screen += strength * mip 0 / levels, is finish()'s
  bloom_pending_ = true;
  bloom_gain_ = strength;
  bloom_div_ = static_cast<float>(mips_.size());
}


void Frame::finish_impl(const Params& P, std::span<std::uint8_t> rgb, Pool& pool) {
  const std::uint32_t salt = hz(static_cast<int>(P.time * 30.f));
  for (int x = 0; x < w_; ++x) {
    const float nx = ((fl(x) + 0.5f) / fl(w_) - 0.5f) * (fl(w_) / fl(h_));
    fin_vig_[zs(x)] = nx * nx;
    fin_key_[zs(x)] = hx(x);
  }
  // On the way: the distortion's moved pixels back into the screen, and bloom's last pass (the screen plus mip 0,
  // bilinear, times its gain over the levels: as bloom() would have added it, to the bit).
  const FrameKernels::Finish a{&P, rgb.data(), salt, moved_rows_, bloom_pending_, bloom_gain_, bloom_div_};
  const auto rows = kernels(avx2_).finish;
  pool.run((h_ + kChunk - 1) / kChunk, [&](int task) { rows(*this, a, task * kChunk, std::min(h_, (task + 1) * kChunk)); });
  moved_rows_ = bloom_pending_ = false;
}

// --- capture and render -----------------------------------------------------------------------------------------------

void Frame::capture(const Light& light, std::span<const std::array<float, 4>> scorch, std::span<Module* const> modules, const Particles& parts,
                    std::span<const Shock> shocks, const FieldBus& bus) {
  P_ = params();
  light_cap_ = light_view(light);
  scorch_.assign(scorch.begin(), scorch.end());
  take_tiles(modules);
  if (parts_.capacity() < parts.alive()) parts_ = Particles(parts.capacity());
  parts.copy_to(parts_);
  shocks_.assign(shocks.begin(), shocks.end());
  const auto heat = bus.heat();
  heat_.assign(heat.begin(), heat.end());  // the same size every frame: allocates on the first only
  heat_cap_ = {heat_.data(), bus.nx(), bus.ny(), bus.x0(), bus.y0(), bus.cell()};
  // render()'s buffers that depend on the grids' sizes, sized here (render() may run on another thread, and nothing it
  // does should allocate)
  light_x_.resize(zs(light.ny()) * zs(w_) * 4);
  heat_x_.resize(zs(bus.ny()) * zs(w_));
  block_bus_.resize(zs(blocks_));
}

void Frame::render(std::span<std::uint8_t> rgb, float bloom_threshold, float bloom_strength, Pool& pool, std::atomic<bool>* images_read) {
  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  compose(P_, &light_cap_, scorch_, pool);
  if (images_read) {
    images_read->store(true);
    pool.wake();
  }
  const auto t1 = Clock::now();
  parts_.draw(screen_, P_.cam_x, P_.cam_y);
  const auto t2 = Clock::now();
  distort_impl(P_, shocks_, heat_cap_, pool);
  const auto t3 = Clock::now();
  bloom_impl(bloom_threshold, bloom_strength, pool);
  const auto t4 = Clock::now();
  finish_impl(P_, rgb, pool);
  const auto t5 = Clock::now();
  render_ms_ = {ms_between(t0, t1), ms_between(t1, t2), ms_between(t2, t3), ms_between(t3, t4), ms_between(t4, t5)};
}

// --- the picture thread -----------------------------------------------------------------------------------------------

PictureThread::PictureThread(Frame& frame, Pool& pool) : frame_(frame), pool_(pool), thread_([this] { loop(); }) { pool_.share_with_callers(); }

PictureThread::~PictureThread() {
  wait();
  stop_.store(true);
  pool_.wake();
  thread_.join();
}

void PictureThread::start(std::span<std::uint8_t> rgb, float bloom_threshold, float bloom_strength) {
  rgb_ = rgb;
  threshold_ = bloom_threshold;
  strength_ = bloom_strength;
  images_read_.store(false);
  done_.store(false);
  go_.store(true);
  pool_.wake();
}

void PictureThread::loop() {
  for (;;) {
    pool_.help_until([&] { return go_.load() || stop_.load(); });  // a thread of the pool until there is a picture to draw
    if (!go_.load()) return;
    go_.store(false);
    frame_.render(rgb_, threshold_, strength_, pool_, &images_read_);
    done_.store(true);
    pool_.wake();
  }
}

void PictureThread::wait_images() {
  const auto t0 = std::chrono::steady_clock::now();
  pool_.help_until([&] { return images_read_.load(); });
  waited_images_ms = ms_between(t0, std::chrono::steady_clock::now());
}

void PictureThread::wait() {
  const auto t0 = std::chrono::steady_clock::now();
  pool_.help_until([&] { return done_.load(); });
  waited_ms = ms_between(t0, std::chrono::steady_clock::now());
}

}  // namespace nfx::compose
