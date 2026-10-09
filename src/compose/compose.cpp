// Composed effects (compose.hpp, docs/COMPOSE.md).
#include "compose.hpp"

#include <neuralfx/noise.hpp>
#include <neuralfx/nvfx.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
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

std::uint32_t hash32(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

float hash01(int x, int y, int z) {
  const auto h = hash32(static_cast<std::uint32_t>(x) * 73856093U ^ static_cast<std::uint32_t>(y) * 19349663U ^ static_cast<std::uint32_t>(z) * 83492791U);
  return static_cast<float>(h >> 8) * (1.f / 16777216.f);
}

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
// integers), clamped to the grid.
float bilerp(const float* f, int nx, int ny, int ch, int c, float x, float y) {
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
  {
    std::lock_guard lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : workers_) t.join();
}

void Pool::run_impl(int n, Fn fn, void* ctx) {
  if (n <= 0) return;
  if (workers_.empty() || n == 1) {
    for (int i = 0; i < n; ++i) fn(ctx, i);
    return;
  }
  {
    std::unique_lock lk(mu_);
    done_cv_.wait(lk, [&] { return busy_ == 0; });  // a late worker of the last job has left it
    job_ = fn;
    ctx_ = ctx;
    n_ = n;
    next_.store(0);
    left_.store(n);
    ++generation_;
  }
  cv_.notify_all();
  drain();
  std::unique_lock lk(mu_);
  done_cv_.wait(lk, [&] { return left_.load() == 0 && busy_ == 0; });
  job_ = nullptr;
}

void Pool::drain() {
  for (int i; (i = next_.fetch_add(1)) < n_;) {
    job_(ctx_, i);
    left_.fetch_sub(1);
  }
}

void Pool::work() {
  std::uint64_t seen = 0;
  for (;;) {
    std::unique_lock lk(mu_);
    cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
    if (stop_) return;
    seen = generation_;
    const Fn f = job_;
    void* ctx = ctx_;
    const int n = n_;
    ++busy_;
    lk.unlock();
    if (f) {
      for (int i; (i = next_.fetch_add(1)) < n;) {
        f(ctx, i);
        left_.fetch_sub(1);
      }
    }
    lk.lock();
    --busy_;
    done_cv_.notify_all();
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

void Module::start(int index, std::uint64_t run_seed) {
  seed = run_seed;
  r_->start(index, controls, seed);
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
  r_->step(controls, seed);
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
  // 1. the finest level: each cell's heat emits; then the blur along x, row by row
  Level& l0 = levels_[0];
  const int chunk = 8;
  pool.run((ny_ + chunk - 1) / chunk, [&](int task) {
    for (int y = task * chunk; y < std::min(ny_, (task + 1) * chunk); ++y) {
      float* a = l0.a.data() + zs(y) * zs(nx_) * 3;
      for (int x = 0; x < nx_; ++x) {
        const float h = std::max(0.f, heat[zs(y) * zs(nx_) + zs(x)]);
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

Frame::Frame(int width, int height) : w_(width), h_(height) {
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
}

void Frame::background(const Light& light, std::span<const std::array<float, 4>> scorch) {
  for (int y = 0; y < h_; ++y) {
    float* row = screen_.row(y);
    const float wy = cam_y + fl(y) + 0.5f;
    for (int x = 0; x < w_; ++x) {
      const float wx = cam_x + fl(x) + 0.5f;
      float* p = row + zs(x) * 4;
      const float hill = ground_y - (18.f + 26.f * (0.5f + 0.5f * std::sin(wx * 0.0042f + 1.3f)) * (0.6f + 0.4f * std::sin(wx * 0.011f + 0.4f)));
      if (wy >= hill && wy < ground_y) {  // distant hills: a dark silhouette, faintly lit
        const auto l = light.at(wx, hill);
        p[0] = 0.006f + 0.05f * l[0];
        p[1] = 0.007f + 0.05f * l[1];
        p[2] = 0.012f + 0.05f * l[2];
      } else if (wy < ground_y) {  // sky: dark blue, lighter at the horizon, a few stars fixed in the world
        const float u = std::clamp((ground_y - wy) / 900.f, 0.f, 1.f);
        p[0] = 0.010f + 0.020f * (1.f - u);
        p[1] = 0.013f + 0.024f * (1.f - u);
        p[2] = 0.030f + 0.035f * (1.f - u);
        const int sx = ifloor(wx / 3.f), sy = ifloor(wy / 3.f);
        const float h = hash01(sx, sy, 7);
        if (h > 0.9965f) {
          const float tw = 0.6f + 0.4f * std::sin(time * (2.f + 6.f * hash01(sx, sy, 9)) + 20.f * h);
          const float s = (h - 0.9965f) / 0.0035f * 0.35f * tw * u;
          p[0] += s;
          p[1] += s;
          p[2] += 1.1f * s;
        }
        const auto l = light.at(wx, wy);  // the sky glows a little where the light is
        p[0] += 0.08f * l[0];
        p[1] += 0.08f * l[1];
        p[2] += 0.08f * l[2];
      } else {  // ground: dark earth, lit by the scene's light from just above it, darker towards the viewer
        const float depth = std::clamp((wy - ground_y) / 160.f, 0.f, 1.f);
        const float tex = 0.75f + 0.5f * hash01(ifloor(wx / 2.f), ifloor(wy / 2.f), 3) * (0.5f + 0.5f * depth);
        const auto l = light.at(wx, ground_y - 6.f - 30.f * depth);
        const float lit = (1.f - 0.55f * depth) * tex;
        p[0] = (0.012f + 0.35f * l[0] / (1.f + 0.6f * l[0])) * lit * 0.9f;
        p[1] = (0.011f + 0.35f * l[1] / (1.f + 0.6f * l[1])) * lit * 0.75f;
        p[2] = (0.010f + 0.35f * l[2] / (1.f + 0.6f * l[2])) * lit * 0.6f;
        for (const auto& s : scorch) {  // scorch marks: x, y, radius, glow
          const float dx = (wx - s[0]) / s[2], dy = (wy - s[1]) / (0.28f * s[2]);
          const float r2 = dx * dx + dy * dy;
          if (r2 >= 1.f) continue;
          const float k = 1.f - smooth01(r2);
          for (int c = 0; c < 3; ++c) p[c] *= 1.f - 0.85f * k;
          const float n = 0.5f + 0.5f * value_noise(wx / 7.f, wy / 3.f, time * 0.4f, 5);  // smooth glowing patches
          const float g = s[3] * k * k * std::pow(n, 5.f) * 0.8f;
          const auto c = heat_colour(0.2f + 0.35f * n);
          p[0] += g * c[0];
          p[1] += g * c[1];
          p[2] += g * c[2];
        }
      }
      p[3] = 1.f;
    }
  }
}

void Frame::draw(std::span<Module* const> modules, Pool& pool) {
  // groups in order of first appearance; each drawn once
  std::array<int, 64> done{};
  int n_done = 0;
  std::array<Module*, 64> tiles{};
  for (Module* m : modules) {
    if (!m->active || m->opacity <= 0.f) continue;
    bool seen = false;
    for (int i = 0; i < n_done; ++i) seen |= done[zs(i)] == m->group;
    if (seen && m->group >= 0) continue;
    int n = 0;
    if (m->group < 0) {
      tiles[0] = m;
      n = 1;
    } else {
      done[zs(n_done++)] = m->group;
      for (Module* t : modules) {
        if (t->active && t->opacity > 0.f && t->group == m->group && n < 64) tiles[zs(n++)] = t;
      }
    }
    draw_group(std::span<Module* const>(tiles.data(), zs(n)), pool);
  }
}

void Frame::draw_group(std::span<Module* const> tiles, Pool& pool) {
  // screen rows covered by the group
  float top = 1e9f, bottom = -1e9f;
  for (const Module* t : tiles) {
    top = std::min(top, t->at.y - cam_y);
    bottom = std::max(bottom, t->at.y + fl(t->size()) * t->at.scale - cam_y);
  }
  const int y0 = std::max(0, ifloor(top)), y1 = std::min(h_, ifloor(bottom) + 1);
  if (y0 >= y1) return;
  const int rows = y1 - y0, chunk = 8;
  pool.run((rows + chunk - 1) / chunk, [&](int task) {
    for (int y = y0 + task * chunk; y < std::min(y1, y0 + (task + 1) * chunk); ++y) {
      float* acc = tmp_.row(y);
      std::fill_n(acc, zs(w_) * 4, 0.f);
      const float wy = cam_y + fl(y) + 0.5f;
      const float clip = smooth01((ground_y + 2.f - wy) / 3.f);  // the ground hides what is below it
      if (clip <= 0.f) continue;
      bool any = false;
      for (const Module* t : tiles) {
        const float sc = t->at.scale, S = fl(t->size());
        const float iy = (wy - t->at.y) / sc - 0.5f;  // image row (top to bottom), continuous
        if (iy < -0.5f || iy > S - 0.5f) continue;
        const int iy0 = std::clamp(ifloor(iy), 0, t->size() - 1), iy1 = std::min(iy0 + 1, t->size() - 1);
        const float fy = std::clamp(iy - fl(iy0), 0.f, 1.f);
        const float ty_up = S - (iy + 0.5f);
        const int x0 = std::max(0, ifloor(t->at.x - cam_x)), x1 = std::min(w_, ifloor(t->at.x + S * sc - cam_x) + 1);
        const float* r0 = t->image().row(iy0);
        const float* r1 = t->image().row(iy1);
        for (int x = x0; x < x1; ++x) {
          const float ix = (cam_x + fl(x) + 0.5f - t->at.x) / sc - 0.5f;
          if (ix < -0.5f || ix > S - 0.5f) continue;
          const int ix0 = std::clamp(ifloor(ix), 0, t->size() - 1), ix1 = std::min(ix0 + 1, t->size() - 1);
          const float fx = std::clamp(ix - fl(ix0), 0.f, 1.f);
          const float w = t->weight_px(ix + 0.5f, ty_up) * t->opacity * clip;
          if (w <= 0.f) continue;
          const float* a = r0 + zs(ix0) * 4;
          const float* b = r0 + zs(ix1) * 4;
          const float* c = r1 + zs(ix0) * 4;
          const float* d = r1 + zs(ix1) * 4;
          float* o = acc + zs(x) * 4;
          for (int ch = 0; ch < 4; ++ch) o[ch] += w * ((1.f - fy) * ((1.f - fx) * a[ch] + fx * b[ch]) + fy * ((1.f - fx) * c[ch] + fx * d[ch]));
          any = true;
        }
      }
      if (!any) continue;
      float* s = screen_.row(y);
      for (int x = 0; x < w_; ++x) {
        const float* o = acc + zs(x) * 4;
        float* p = s + zs(x) * 4;
        const float k = 1.f - std::clamp(o[3], 0.f, 1.f);
        p[0] = p[0] * k + o[0];
        p[1] = p[1] * k + o[1];
        p[2] = p[2] * k + o[2];
      }
    }
  });
}

void Frame::distort(std::span<const Shock> shocks, const FieldBus& bus, Pool& pool) {
  const int chunk = 8;
  pool.run((h_ + chunk - 1) / chunk, [&](int task) {
    for (int y = task * chunk; y < std::min(h_, (task + 1) * chunk); ++y) {
      const float* src_row = screen_.row(y);
      float* out = tmp_.row(y);
      const float wy = cam_y + fl(y) + 0.5f;
      for (int x = 0; x < w_; ++x) {
        const float wx = cam_x + fl(x) + 0.5f;
        float dx = 0.f, dy = 0.f;
        for (const Shock& s : shocks) {
          const float R = s.radius(time);
          if (R <= 0.f) continue;
          const float ex = wx - s.x, ey = wy - s.y, r = std::sqrt(ex * ex + ey * ey) + 1e-3f;
          const float d = r - R;
          if (std::fabs(d) >= s.width) continue;
          const float a = s.amp * std::exp(-(time - s.t0) / (2.f * s.decay)) * std::sin(std::numbers::pi_v<float> * d / s.width);
          dx += a * ex / r;
          dy += a * ey / r;
        }
        if (haze > 0.f) {
          const float h = bus.at(wx, wy + 22.f).heat;  // hot air below shimmers what is seen through it
          if (h > 0.01f) {
            const float a = haze * std::min(1.f, 1.6f * h);
            dx += a * 1.6f * std::sin(wy * 0.09f + time * 11.f + 2.f * std::sin(wx * 0.045f + time * 1.7f));
            dy += a * 1.1f * std::sin(wx * 0.07f - time * 8.f + 1.5f * std::sin(wy * 0.05f));
          }
        }
        float* o = out + zs(x) * 4;
        if (dx == 0.f && dy == 0.f) {
          std::copy_n(src_row + zs(x) * 4, 4, o);
          continue;
        }
        const float sx = std::clamp(fl(x) + dx, 0.f, fl(w_ - 1)), sy = std::clamp(fl(y) + dy, 0.f, fl(h_ - 1));
        const int x0 = std::min(static_cast<int>(sx), w_ - 2), y0 = std::min(static_cast<int>(sy), h_ - 2);
        const float fx = sx - fl(x0), fy = sy - fl(y0);
        const float* a = screen_.row(y0) + zs(x0) * 4;
        const float* c = screen_.row(y0 + 1) + zs(x0) * 4;
        for (int ch = 0; ch < 4; ++ch) o[ch] = (1.f - fy) * ((1.f - fx) * a[ch] + fx * a[4 + ch]) + fy * ((1.f - fx) * c[ch] + fx * c[4 + ch]);
      }
    }
  });
  std::swap(screen_, tmp_);
}

void Frame::bloom(float threshold, float strength, Pool& pool) {
  // bright pass into mip 0 (half size), then down, blur, and up
  Image4& m0 = mips_[0];
  pool.run(m0.h, [&](int y) {
    float* o = m0.row(y);
    for (int x = 0; x < m0.w; ++x) {
      float s[3] = {0, 0, 0};
      for (int q = 0; q < 4; ++q) {
        const int sx = std::min(2 * x + (q & 1), w_ - 1), sy = std::min(2 * y + (q >> 1), h_ - 1);
        const float* p = screen_.row(sy) + zs(sx) * 4;
        const float l = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
        const float k = std::max(0.f, l - threshold) / std::max(l, 1e-4f);
        for (int c = 0; c < 3; ++c) s[c] += 0.25f * k * p[c];
      }
      for (int c = 0; c < 3; ++c) o[zs(x) * 4 + zs(c)] = s[c];
    }
  });
  for (std::size_t l = 1; l < mips_.size(); ++l) {
    Image4& s = mips_[l - 1];
    Image4& d = mips_[l];
    pool.run(d.h, [&](int y) {
      float* o = d.row(y);
      for (int x = 0; x < d.w; ++x) {
        for (int c = 0; c < 3; ++c) {
          float sum = 0.f;
          for (int q = 0; q < 4; ++q) sum += s.row(std::min(2 * y + (q >> 1), s.h - 1))[zs(std::min(2 * x + (q & 1), s.w - 1)) * 4 + zs(c)];
          o[zs(x) * 4 + zs(c)] = 0.25f * sum;
        }
      }
    });
  }
  static constexpr float kw[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
  for (std::size_t l = 0; l < mips_.size(); ++l) {
    Image4& a = mips_[l];
    Image4& b = mips_tmp_[l];
    pool.run(a.h, [&](int y) {
      for (int x = 0; x < a.w; ++x) {
        for (int c = 0; c < 3; ++c) {
          float s = 0.f;
          for (int t = -2; t <= 2; ++t) s += kw[t + 2] * a.row(y)[zs(std::clamp(x + t, 0, a.w - 1)) * 4 + zs(c)];
          b.row(y)[zs(x) * 4 + zs(c)] = s;
        }
      }
    });
    pool.run(a.h, [&](int y) {
      for (int x = 0; x < a.w; ++x) {
        for (int c = 0; c < 3; ++c) {
          float s = 0.f;
          for (int t = -2; t <= 2; ++t) s += kw[t + 2] * b.row(std::clamp(y + t, 0, a.h - 1))[zs(x) * 4 + zs(c)];
          a.row(y)[zs(x) * 4 + zs(c)] = s;
        }
      }
    });
  }
  for (std::size_t l = mips_.size() - 1; l > 0; --l) {  // up: each level adds the coarser one, bilinear
    Image4& c = mips_[l];
    Image4& f = mips_[l - 1];
    pool.run(f.h, [&](int y) {
      for (int x = 0; x < f.w; ++x) {
        const float gx = (fl(x) + 0.5f) / 2.f - 0.5f, gy = (fl(y) + 0.5f) / 2.f - 0.5f;
        for (int ch = 0; ch < 3; ++ch) f.row(y)[zs(x) * 4 + zs(ch)] += bilerp(c.px.data(), c.w, c.h, 4, ch, gx, gy);
      }
    });
  }
  pool.run(h_, [&](int y) {
    float* p = screen_.row(y);
    for (int x = 0; x < w_; ++x) {
      const float gx = (fl(x) + 0.5f) / 2.f - 0.5f, gy = (fl(y) + 0.5f) / 2.f - 0.5f;
      for (int ch = 0; ch < 3; ++ch) p[zs(x) * 4 + zs(ch)] += strength * bilerp(m0.px.data(), m0.w, m0.h, 4, ch, gx, gy) / static_cast<float>(mips_.size());
    }
  });
}

void Frame::finish(std::span<std::uint8_t> rgb, Pool& pool) {
  const Gamma& g = gamma();
  const auto aces = [](float v) { return std::clamp(v * (2.51f * v + 0.03f) / (v * (2.43f * v + 0.59f) + 0.14f), 0.f, 1.f); };
  const int frame_salt = static_cast<int>(time * 30.f);
  pool.run(h_, [&](int y) {
    const float* p = screen_.row(y);
    std::uint8_t* o = rgb.data() + zs(y) * zs(w_) * 3;
    const float ny = (fl(y) + 0.5f) / fl(h_) - 0.5f;
    for (int x = 0; x < w_; ++x) {
      const float nx = ((fl(x) + 0.5f) / fl(w_) - 0.5f) * (fl(w_) / fl(h_));
      const float vig = 1.f - 0.45f * (nx * nx + ny * ny);
      const float grain = (hash01(x, y, frame_salt) - 0.5f) * 0.006f;
      for (int c = 0; c < 3; ++c) {
        const float v = aces(p[zs(x) * 4 + zs(c)] * exposure * vig * fade) + grain;
        o[zs(x) * 3 + zs(c)] = g.to_display[zs(std::clamp(static_cast<int>(v * 4095.f + 0.5f), 0, 4095))];
      }
    }
  });
}

}  // namespace nfx::compose
