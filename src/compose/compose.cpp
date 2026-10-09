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

std::uint32_t hash32(std::uint32_t x) {
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
float unit(std::uint32_t key) { return static_cast<float>(hash32(key) >> 8) * (1.f / 16777216.f); }

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
    : at(p), name_(std::move(name)), e_(e), size_(size), r_(make_runner(e, size, isa)) {
  img_.allocate(size, size);
  rgba8_.assign(zs(size) * zs(size) * 4, 0);
  shadow_.assign(zs(res()) * zs(res()), 0.f);
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
  const float k = fl(S) / fl(R), inv_hs = 1.f / sp.heat_scale;
  for (int y = 0; y < S; ++y) {  // y up
    float* out = img_.row(S - 1 - y);
    const float wy = at.y + (fl(S - y) - 0.5f) * at.scale;
    const float cy = (fl(y) + 0.5f) / k - 0.5f;
    for (int x = 0; x < S; ++x) {
      const std::size_t i = zs(y) * zs(S) + zs(x);
      const float T = std::max(0.f, ft[i]), D = std::max(0.f, fd[i]);
      float* o = out + zs(x) * 4;
      if (T < 1e-4f && D < 1e-4f) {
        o[0] = o[1] = o[2] = o[3] = 0.f;
        continue;
      }
      const float a = 1.f - std::exp(-sp.soot_density * D);
      const float sh = std::exp(-sp.shadow * bilerp(shadow_.data(), R, R, 1, 0, (fl(x) + 0.5f) / k - 0.5f, cy));
      // the soot as a height field: its slope towards the moon (up and left) lights billows, away from it darkens them
      const float dx = 0.5f * (fd[zs(y) * zs(S) + zs(std::min(x + 2, S - 1))] - fd[zs(y) * zs(S) + zs(std::max(x - 2, 0))]);
      const float dy = 0.5f * (fd[zs(std::min(y + 2, S - 1)) * zs(S) + zs(x)] - fd[zs(std::max(y - 2, 0)) * zs(S) + zs(x)]);
      const float nx = -sp.relief * dx, ny = -sp.relief * dy, inv = 1.f / std::sqrt(nx * nx + ny * ny + 1.f);
      const float lambert = std::max(0.f, (-0.45f * nx + 0.6f * ny + 0.66f) * inv);
      const float moon = sp.sky * sh * (0.35f + 0.9f * lambert);
      const float under = std::max(0.f, (0.2f * nx - 0.7f * ny + 0.68f) * inv);  // facing down: lit by the fire below
      std::array<float, 3> L{0.7f * moon, 0.8f * moon, 1.1f * moon};
      if (light) {
        const auto l = light->at(at.x + (fl(x) + 0.5f) * at.scale, wy);
        for (int c = 0; c < 3; ++c) L[zs(c)] += sp.scene_light * l[zs(c)] / (1.f + l[zs(c)]) * (0.4f + 0.9f * under);  // soft limit: hot gas inside its own glow
      }
      const float t = T * inv_hs;
      const auto hc = heat_colour(t);
      const float e = sp.emission * std::pow(t, sp.emission_power) * (1.f - 0.55f * a);
      for (int c = 0; c < 3; ++c) o[c] = a * sp.soot_albedo * sp.tint[zs(c)] * L[zs(c)] + e * hc[zs(c)];
      o[3] = a;
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
  const auto fa = [&](int j, int i) { return b_is == Side::top ? zs(S - bp + j) * zs(S) + zs(i) : zs(i) * zs(S) + zs(S - bp + j); };
  const auto fb = [&](int j, int i) { return b_is == Side::top ? zs(j) * zs(S) + zs(i) : zs(i) * zs(S) + zs(j); };
  for (auto [FA, FB] : {std::pair{a.runner().fine_heat_mut(), b.runner().fine_heat_mut()}, std::pair{a.runner().fine_soot_mut(), b.runner().fine_soot_mut()}}) {
    for (int j = 0; j < bp; ++j) {
      const float w = band_weight((fl(j) + 0.5f) / fl(k) - 0.5f, cells);
      for (int i = 0; i < S; ++i) {
        float& va = FA[fa(j, i)];
        float& vb = FB[fb(j, i)];
        va = vb = w * va + (1.f - w) * vb;
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
}

void FieldBus::clear() {
  std::ranges::fill(all_, 0.f);
  std::ranges::fill(layer_, 0.f);
  std::ranges::fill(heat_, 0.f);
  std::ranges::fill(soot_, 0.f);
}

void FieldBus::publish(const Module& m) {
  if (!m.active || m.group < 0 || m.group >= groups_) return;
  const int R = m.res(), S = m.size(), C = m.channels();
  const float k = fl(S) / fl(R), sc = m.at.scale, span = fl(S) * sc;
  auto co = m.runner().coarse();
  const int i0 = std::max(0, ifloor((m.at.x - x0_) / cell_)), i1 = std::min(nx_ - 1, ifloor((m.at.x + span - x0_) / cell_) + 1);
  const int j0 = std::max(0, ifloor((m.at.y - y0_) / cell_)), j1 = std::min(ny_ - 1, ifloor((m.at.y + span - y0_) / cell_) + 1);
  float* L = layer_.data() + zs(m.group) * all_.size();
  for (int j = j0; j <= j1; ++j) {
    const float wy = y0_ + (fl(j) + 0.5f) * cell_;
    const float ty = fl(S) - (wy - m.at.y) / sc;  // tile pixels, y up
    if (ty < 0.f || ty > fl(S)) continue;
    for (int i = i0; i <= i1; ++i) {
      const float wx = x0_ + (fl(i) + 0.5f) * cell_;
      const float tx = (wx - m.at.x) / sc;
      if (tx < 0.f || tx > fl(S)) continue;
      const float w = m.weight_px(tx, ty) * m.opacity;
      if (w <= 0.f) continue;
      const float cx = tx / k - 0.5f, cy = ty / k - 0.5f;
      const float s[4] = {bilerp(co.data(), R, R, C, 0, cx, cy) * k * sc, -bilerp(co.data(), R, R, C, 1, cx, cy) * k * sc,
                          bilerp(co.data(), R, R, C, 2, cx, cy), bilerp(co.data(), R, R, C, 3, cx, cy)};
      const std::size_t q = (zs(j) * zs(nx_) + zs(i)) * 4;
      for (int c = 0; c < 4; ++c) {
        all_[q + zs(c)] += w * s[c];
        L[q + zs(c)] += w * s[c];
      }
      heat_[q / 4] += w * s[2];
      soot_[q / 4] += w * s[3];
    }
  }
}

FieldBus::Sample FieldBus::sample(const std::vector<float>& f, float x, float y) const {
  const float gx = (x - x0_) / cell_ - 0.5f, gy = (y - y0_) / cell_ - 0.5f;
  if (gx < -1.f || gy < -1.f || gx > fl(nx_) || gy > fl(ny_)) return {};
  return {bilerp(f.data(), nx_, ny_, 4, 0, gx, gy), bilerp(f.data(), nx_, ny_, 4, 1, gx, gy), bilerp(f.data(), nx_, ny_, 4, 2, gx, gy),
          bilerp(f.data(), nx_, ny_, 4, 3, gx, gy)};
}

FieldBus::Sample FieldBus::at(float x, float y) const { return sample(all_, x, y); }

FieldBus::Sample FieldBus::others(float x, float y, int group) const {
  Sample s = sample(all_, x, y);
  if (group < 0 || group >= groups_) return s;
  const float gx = (x - x0_) / cell_ - 0.5f, gy = (y - y0_) / cell_ - 0.5f;
  if (gx < -1.f || gy < -1.f || gx > fl(nx_) || gy > fl(ny_)) return s;
  const float* L = layer_.data() + zs(group) * all_.size();
  s.u -= bilerp(L, nx_, ny_, 4, 0, gx, gy);
  s.v -= bilerp(L, nx_, ny_, 4, 1, gx, gy);
  s.heat -= bilerp(L, nx_, ny_, 4, 2, gx, gy);
  s.soot -= bilerp(L, nx_, ny_, 4, 3, gx, gy);
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
  // coarse: each source cell's amount, splatted bilinearly into the targets' cells at its world centre
  for (int cy = row0; cy < R; ++cy) {
    for (int cx = 0; cx < R; ++cx) {
      float* c = co.data() + (zs(cy) * zs(R) + zs(cx)) * zs(C);
      const float h = c[2] * fraction * heat_gain, d = c[3] * fraction * soot_gain;
      if (h <= 0.f && d <= 0.f) continue;
      const float wx = from.at.x + (fl(cx) + 0.5f) * k * sc, wy = from.at.y + (fl(S) - (fl(cy) + 0.5f) * k) * sc;
      const float area = (k * sc) * (k * sc);
      float placed = 0.f;
      for (Module* t : to) {
        if (!t->active || t == &from) continue;
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
        for (int q = 0; q < 4; ++q) {
          float* p = tc.data() + (zs(y0 + dy[q]) * zs(tR) + zs(x0 + dx[q])) * zs(tC);
          p[2] += ratio * ws[q] * h;
          p[3] += ratio * ws[q] * d;
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
          for (Module* t : to) {
            if (!t->active || t == &from) continue;
            const int tS = t->size();
            const float tsc = t->at.scale;
            const float tx = (pwx - t->at.x) / tsc, ty = fl(tS) - (pwy - t->at.y) / tsc;
            if (tx < 0.f || ty < 0.f || tx >= fl(tS) || ty >= fl(tS)) continue;
            const float w = t->weight_px(tx, ty);
            if (w <= 0.f) continue;
            const std::size_t j = zs(static_cast<int>(ty)) * zs(tS) + zs(static_cast<int>(tx));
            const float ratio = w * (sc * sc) / (tsc * tsc);
            t->runner().fine_heat_mut()[j] += ratio * fh;
            t->runner().fine_soot_mut()[j] += ratio * fdd;
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
    levels_.push_back(std::move(v));
    nx = std::max(1, (nx + 1) / 2);
    ny = std::max(1, (ny + 1) / 2);
  }
}

void Light::update(const FieldBus& bus, float gain, std::array<float, 3> flash) {
  flash_ = flash;
  auto heat = bus.heat();
  Level& l0 = levels_[0];
  for (std::size_t i = 0; i < heat.size(); ++i) {
    const float h = std::max(0.f, heat[i]);
    const auto c = heat_colour(h / 1.2f);
    const float e = gain * h * h;
    for (int ch = 0; ch < 3; ++ch) l0.a[i * 3 + zs(ch)] = e * c[zs(ch)];
  }
  for (std::size_t l = 1; l < levels_.size(); ++l) {  // 2x2 box down
    Level& s = levels_[l - 1];
    Level& d = levels_[l];
    for (int y = 0; y < d.ny; ++y) {
      for (int x = 0; x < d.nx; ++x) {
        for (int ch = 0; ch < 3; ++ch) {
          float sum = 0.f;
          for (int q = 0; q < 4; ++q) {
            const int sx = std::min(2 * x + (q & 1), s.nx - 1), sy = std::min(2 * y + (q >> 1), s.ny - 1);
            sum += s.a[(zs(sy) * zs(s.nx) + zs(sx)) * 3 + zs(ch)];
          }
          d.a[(zs(y) * zs(d.nx) + zs(x)) * 3 + zs(ch)] = sum;  // amounts add up: coarser levels carry the total
        }
      }
    }
  }
  for (Level& v : levels_) {  // separable [1 4 6 4 1] / 16 blur, a then b then a
    for (int pass = 0; pass < 2; ++pass) {
      const std::vector<float>& src = pass == 0 ? v.a : v.b;
      std::vector<float>& dst = pass == 0 ? v.b : v.a;
      for (int y = 0; y < v.ny; ++y) {
        for (int x = 0; x < v.nx; ++x) {
          for (int ch = 0; ch < 3; ++ch) {
            float s = 0.f;
            static constexpr float kw[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
            for (int t = -2; t <= 2; ++t) {
              const int xx = pass == 0 ? std::clamp(x + t, 0, v.nx - 1) : x, yy = pass == 1 ? std::clamp(y + t, 0, v.ny - 1) : y;
              s += kw[t + 2] * src[(zs(yy) * zs(v.nx) + zs(xx)) * 3 + zs(ch)];
            }
            dst[(zs(y) * zs(v.nx) + zs(x)) * 3 + zs(ch)] = s;
          }
        }
      }
    }
  }
  // light = sum over levels of the blurred amounts per area of that level's cell: a soft falloff with a long tail
  std::ranges::fill(L_, 0.f);
  for (std::size_t l = 0; l < levels_.size(); ++l) {
    const Level& v = levels_[l];
    const float scale = 1.f / static_cast<float>(1u << (2 * l)) * 0.6f;  // per unit area of the level's cells
    const float f = static_cast<float>(1u << l);
    for (int y = 0; y < ny_; ++y) {
      for (int x = 0; x < nx_; ++x) {
        const float gx = (fl(x) + 0.5f) / f - 0.5f, gy = (fl(y) + 0.5f) / f - 0.5f;
        for (int ch = 0; ch < 3; ++ch) L_[(zs(y) * zs(nx_) + zs(x)) * 3 + zs(ch)] += scale * bilerp(v.a.data(), v.nx, v.ny, 3, ch, gx, gy);
      }
    }
  }
}

std::array<float, 3> Light::at(float x, float y) const {
  const float gx = (x - x0_) / cell_ - 0.5f, gy = (y - y0_) / cell_ - 0.5f;
  return {bilerp(L_.data(), nx_, ny_, 3, 0, gx, gy) + flash_[0], bilerp(L_.data(), nx_, ny_, 3, 1, gx, gy) + flash_[1],
          bilerp(L_.data(), nx_, ny_, 3, 2, gx, gy) + flash_[2]};
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

namespace {

// The taps bilerp() takes at continuous cell coordinate v on an axis of n cells (clamped as it clamps).
Frame::Tap tap(float v, int n) {
  v = std::clamp(v, 0.f, fl(n - 1));
  const int i0 = std::min(static_cast<int>(v), std::max(0, n - 2));
  const float f = v - fl(i0);
  return {i0, std::min(i0 + 1, n - 1), 1.f - f, f};
}

// bilerp()'s blend of the corners a, b (first row) and c, d (second row), in its order of operations: a lookup by taps
// gives its result to the bit.
float blend(float a, float b, float c, float d, const Frame::Tap& x, const Frame::Tap& y) {
  return y.w0 * (x.w0 * a + x.w1 * b) + y.w1 * (x.w0 * c + x.w1 * d);
}

// Channel c of a grid of `ch` interleaved channels and nx columns, looked up by taps.
float lookup(const float* f, int nx, int ch, int c, const Frame::Tap& x, const Frame::Tap& y) {
  const float* r0 = f + zs(y.i0) * zs(nx) * zs(ch) + zs(c);
  const float* r1 = f + zs(y.i1) * zs(nx) * zs(ch) + zs(c);
  return blend(r0[zs(x.i0) * zs(ch)], r0[zs(x.i1) * zs(ch)], r1[zs(x.i0) * zs(ch)], r1[zs(x.i1) * zs(ch)], x, y);
}

constexpr int kChunk = 8;    // screen rows per task
constexpr int kBlock = 256;  // pixels per block of a row, for scratch on the stack

// A pixel's four channels as one vector (GCC vector extensions, as in src/common/simd_kernels.hpp). Each lane does
// what the scalar code does to its channel, in the same order, so the results are the same to the bit.
typedef float px4 __attribute__((vector_size(16)));
typedef float px4u __attribute__((vector_size(16), aligned(4)));  // unaligned access
px4 load4(const float* p) { return *reinterpret_cast<const px4u*>(p); }
void store4(float* p, px4 v) { *reinterpret_cast<px4u*>(p) = v; }

// blend() of four pixels.
px4 blend4(const float* a, const float* b, const float* c, const float* d, const Frame::Tap& x, const Frame::Tap& y) {
  return y.w0 * (x.w0 * load4(a) + x.w1 * load4(b)) + y.w1 * (x.w0 * load4(c) + x.w1 * load4(d));
}

}  // namespace

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
  cols_.resize(zs(width));
  tile_cols_.resize(zs(kMaxTiles) * zs(width));
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
}

void Frame::background(const Light& light, std::span<const std::array<float, 4>> scorch, Pool& pool) {
  // The light is looked up as Light::at() does it, by taps (its columns once per screen column, its rows once per row)
  // in a copy with a fourth channel, so that a cell is one vector.
  const auto field = light.field();
  const int lnx = light.nx(), lny = light.ny();
  light4_.resize(zs(lnx) * zs(lny) * 4);  // the same size every frame: allocates on the first only
  for (std::size_t i = 0; i < zs(lnx) * zs(lny); ++i) {
    for (std::size_t c = 0; c < 3; ++c) light4_[i * 4 + c] = field[i * 3 + c];
    light4_[i * 4 + 3] = 0.f;
  }
  const std::array<float, 3> flash = light.flash();
  const px4 flash4{flash[0], flash[1], flash[2], 0.f};
  const float* L = light4_.data();
  const auto light_at = [&](const Tap& x, const Tap& y) {
    const float* r0 = L + zs(y.i0) * zs(lnx) * 4;
    const float* r1 = L + zs(y.i1) * zs(lnx) * 4;
    return blend4(r0 + zs(x.i0) * 4, r0 + zs(x.i1) * 4, r1 + zs(x.i0) * 4, r1 + zs(x.i1) * 4, x, y) + flash4;
  };
  const auto light_rows = [&](float wy) { return tap((wy - light.y0()) / light.cell() - 0.5f, lny); };
  for (int x = 0; x < w_; ++x) {  // per column: the hills' outline and their colour, the light's columns, hash keys
    Column& k = cols_[zs(x)];
    k.wx = cam_x + fl(x) + 0.5f;
    k.hill = ground_y - (18.f + 26.f * (0.5f + 0.5f * std::sin(k.wx * 0.0042f + 1.3f)) * (0.6f + 0.4f * std::sin(k.wx * 0.011f + 0.4f)));
    k.light = tap((k.wx - light.x0()) / light.cell() - 0.5f, lnx);
    const px4 l = light_at(k.light, light_rows(k.hill));  // distant hills: a dark silhouette, faintly lit
    k.hill_colour = {0.006f + 0.05f * l[0], 0.007f + 0.05f * l[1], 0.012f + 0.05f * l[2], 1.f};
    k.star = hx(ifloor(k.wx / 3.f));
    k.tex = hx(ifloor(k.wx / 2.f));
  }
  pool.run((h_ + kChunk - 1) / kChunk, [&](int task) {
    for (int y = task * kChunk; y < std::min(h_, (task + 1) * kChunk); ++y) {
      float* row = screen_.row(y);
      const float wy = cam_y + fl(y) + 0.5f;
      if (wy < ground_y) {
        // the hills, and above them the sky: dark blue, lighter at the horizon, a few stars fixed in the world, and a
        // little glow where the light is
        const float u = std::clamp((ground_y - wy) / 900.f, 0.f, 1.f);
        const px4 sky{0.010f + 0.020f * (1.f - u), 0.013f + 0.024f * (1.f - u), 0.030f + 0.035f * (1.f - u), 1.f};
        const std::uint32_t star_y = hy(ifloor(wy / 3.f));
        const Tap ly = light_rows(wy);
        for (int x = 0; x < w_; ++x) {
          const Column& k = cols_[zs(x)];
          float* p = row + zs(x) * 4;
          if (wy >= k.hill) {
            store4(p, load4(k.hill_colour.data()));
            continue;
          }
          px4 v = sky;
          const float h = unit(k.star ^ star_y ^ hz(7));
          if (h > 0.9965f) {
            const float tw = 0.6f + 0.4f * std::sin(time * (2.f + 6.f * unit(k.star ^ star_y ^ hz(9))) + 20.f * h);
            const float s = (h - 0.9965f) / 0.0035f * 0.35f * tw * u;
            v += px4{s, s, 1.1f * s, 0.f};
          }
          v += 0.08f * light_at(k.light, ly);
          v[3] = 1.f;
          store4(p, v);
        }
        continue;
      }
      // ground: dark earth, lit by the scene's light from just above it, darker towards the viewer
      const float depth = std::clamp((wy - ground_y) / 160.f, 0.f, 1.f);
      const float tex_amp = 0.5f + 0.5f * depth, shade = 1.f - 0.55f * depth;
      const std::uint32_t tex_y = hy(ifloor(wy / 2.f)) ^ hz(3);
      const Tap ly = light_rows(ground_y - 6.f - 30.f * depth);
      const px4 earth{0.012f, 0.011f, 0.010f, 0.f}, tint{0.9f, 0.75f, 0.6f, 0.f};
      for (int x = 0; x < w_; ++x) {
        const Column& k = cols_[zs(x)];
        const float tex = 0.75f + 0.5f * unit(k.tex ^ tex_y) * tex_amp;
        const float lit = shade * tex;
        const px4 l = light_at(k.light, ly);
        px4 v = (earth + 0.35f * l / (1.f + 0.6f * l)) * lit * tint;
        v[3] = 1.f;
        store4(row + zs(x) * 4, v);
      }
      for (const auto& s : scorch) {  // scorch marks (x, y, radius, glow), over the columns each covers in this row
        const float dy = (wy - s[1]) / (0.28f * s[2]);
        if (!(dy * dy < 1.f)) continue;
        const float half = s[2] * std::sqrt(1.f - dy * dy);  // with a margin below: the test per pixel decides
        const int xa = std::max(0, ifloor(s[0] - half - cam_x) - 2), xb = std::min(w_, ifloor(s[0] + half - cam_x) + 3);
        for (int x = xa; x < xb; ++x) {
          const Column& k = cols_[zs(x)];
          const float dx = (k.wx - s[0]) / s[2];
          const float r2 = dx * dx + dy * dy;
          if (r2 >= 1.f) continue;
          float* p = row + zs(x) * 4;
          const float kk = 1.f - smooth01(r2);
          for (int c = 0; c < 3; ++c) p[c] *= 1.f - 0.85f * kk;
          const float n = 0.5f + 0.5f * value_noise(k.wx / 7.f, wy / 3.f, time * 0.4f, 5);  // smooth glowing patches
          const float g = s[3] * kk * kk * std::pow(n, 5.f) * 0.8f;
          const auto c = heat_colour(0.2f + 0.35f * n);
          p[0] += g * c[0];
          p[1] += g * c[1];
          p[2] += g * c[2];
        }
      }
    }
  });
}

void Frame::draw(std::span<Module* const> modules, Pool& pool) {
  // groups in order of first appearance; each drawn once
  std::array<int, kMaxTiles> done{};
  int n_done = 0;
  std::array<Module*, kMaxTiles> tiles{};
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
        if (t->active && t->opacity > 0.f && t->group == m->group && n < kMaxTiles) tiles[zs(n++)] = t;
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
  // Each tile's screen columns, once: its image columns, and the factors of its ownership weight that depend on x.
  // Module::weight_px() is a product of factors of x and of y; multiplied here in its order, the weight is the same
  // to the bit. Only the columns that see the image are kept (a run: the image column grows with the screen's).
  std::array<std::array<int, 2>, kMaxTiles> seen_cols{};
  for (std::size_t i = 0; i < tiles.size(); ++i) {
    const Module* t = tiles[i];
    const float sc = t->at.scale, S = fl(t->size()), R = fl(t->res()), k = S / R;
    const int x0 = std::max(0, ifloor(t->at.x - cam_x)), x1 = std::min(w_, ifloor(t->at.x + S * sc - cam_x) + 1);
    int first = x1, last = x0 - 1;
    TileColumn* cols = tile_cols_.data() + i * zs(w_);
    for (int x = x0; x < x1; ++x) {
      const float ix = (cam_x + fl(x) + 0.5f - t->at.x) / sc - 0.5f;  // image column, continuous
      if (ix < -0.5f || ix > S - 0.5f) continue;
      first = std::min(first, x);
      last = x;
      TileColumn& c = cols[zs(x)];
      const int ix0 = std::clamp(ifloor(ix), 0, t->size() - 1);
      const float fx = std::clamp(ix - fl(ix0), 0.f, 1.f);
      c.t = {ix0, std::min(ix0 + 1, t->size() - 1), 1.f - fx, fx};
      const float px = ix + 0.5f, cx = px / k - 0.5f;
      c.band = 1.f;
      if (t->band[0] > 0) c.band *= 1.f - band_weight(cx, t->band[0]);
      if (t->band[1] > 0) c.band *= band_weight(cx - (R - fl(t->band[1])), t->band[1]);
      c.feather_left = t->feather > 0.f && t->band[0] == 0 ? smooth01(px / t->feather) : 1.f;
      c.feather_right = t->feather > 0.f && t->band[1] == 0 ? smooth01((S - px) / t->feather) : 1.f;
    }
    seen_cols[i] = first <= last ? std::array<int, 2>{first, last + 1} : std::array<int, 2>{0, 0};
  }
  const int rows = y1 - y0;
  pool.run((rows + kChunk - 1) / kChunk, [&](int task) {
    // A pixel whose four image pixels are all empty (every channel 0) adds exactly nothing, so each image row is drawn
    // only between its first and last pixel that is not empty. Rows found here are kept, two per tile: the tiles are
    // scaled up, so the next screen row mostly reads the same two image rows.
    struct Occupied {
      int row = -1, a = 0, b = 0;  // image columns [a, b) of image row `row`
    };
    std::array<std::array<Occupied, 2>, kMaxTiles> known{};
    const auto occupied = [&](std::size_t i, int r) {
      Occupied& o = known[i][zs(r & 1)];
      if (o.row == r) return o;
      const float* p = tiles[i]->image().row(r);
      const int w = tiles[i]->size();
      const auto empty = [&](int x) {  // all four channels +0 (bits all 0; a -0 is drawn, which adds nothing either)
        std::uint64_t a = 0, b = 0;
        std::memcpy(&a, p + zs(x) * 4, 8);
        std::memcpy(&b, p + zs(x) * 4 + 2, 8);
        return (a | b) == 0;
      };
      o = {r, 0, w};
      while (o.a < w && empty(o.a)) ++o.a;
      if (o.a == w) o.b = 0;  // all empty
      while (o.b > o.a && empty(o.b - 1)) --o.b;
      return o;
    };
    struct TileRow {
      Tap ty;
      float band_b = 1.f, band_t = 1.f, feather_b = 1.f, feather_t = 1.f;
      int xa = 0, xb = 0;  // screen columns drawn
    };
    std::array<TileRow, kMaxTiles> tr;
    for (int y = y0 + task * kChunk; y < std::min(y1, y0 + (task + 1) * kChunk); ++y) {
      const float wy = cam_y + fl(y) + 0.5f;
      const float clip = smooth01((ground_y + 2.f - wy) / 3.f);  // the ground hides what is below it
      if (clip <= 0.f) continue;
      int ux0 = w_, ux1 = 0;  // screen columns any tile draws in this row
      for (std::size_t i = 0; i < tiles.size(); ++i) {
        const Module* t = tiles[i];
        TileRow& r = tr[i];
        r.xa = r.xb = 0;
        const float sc = t->at.scale, S = fl(t->size()), R = fl(t->res()), k = S / R;
        const float iy = (wy - t->at.y) / sc - 0.5f;  // image row (top to bottom), continuous
        if (iy < -0.5f || iy > S - 0.5f) continue;
        const int iy0 = std::clamp(ifloor(iy), 0, t->size() - 1);
        const float fy = std::clamp(iy - fl(iy0), 0.f, 1.f);
        r.ty = {iy0, std::min(iy0 + 1, t->size() - 1), 1.f - fy, fy};
        const Occupied o0 = occupied(i, r.ty.i0), o1 = occupied(i, r.ty.i1);
        const int a = std::min(o0.a, o1.a), b = std::max(o0.b, o1.b);
        if (a >= b) continue;
        // the screen columns whose image columns reach [a, b)
        const TileColumn* cols = tile_cols_.data() + i * zs(w_);
        const TileColumn* c0 = cols + seen_cols[i][0];
        const TileColumn* c1 = cols + seen_cols[i][1];
        const TileColumn* ca = std::partition_point(c0, c1, [&](const TileColumn& c) { return c.t.i1 < a; });
        const TileColumn* cb = std::partition_point(ca, c1, [&](const TileColumn& c) { return c.t.i0 < b; });
        r.xa = static_cast<int>(ca - cols);
        r.xb = static_cast<int>(cb - cols);
        if (r.xa >= r.xb) continue;
        ux0 = std::min(ux0, r.xa);
        ux1 = std::max(ux1, r.xb);
        // the factors of the ownership weight that depend on y (tile pixels, y up)
        const float py = S - (iy + 0.5f), cy = py / k - 0.5f;
        r.band_b = t->band[2] > 0 ? 1.f - band_weight(cy, t->band[2]) : 1.f;
        r.band_t = t->band[3] > 0 ? band_weight(cy - (R - fl(t->band[3])), t->band[3]) : 1.f;
        r.feather_b = t->feather > 0.f && t->band[2] == 0 ? smooth01(py / t->feather) : 1.f;
        r.feather_t = t->feather > 0.f && t->band[3] == 0 ? smooth01((S - py) / t->feather) : 1.f;
      }
      if (ux0 >= ux1) continue;
      float* acc = tmp_.row(y);
      std::fill(acc + zs(ux0) * 4, acc + zs(ux1) * 4, 0.f);
      bool any = false;
      for (std::size_t i = 0; i < tiles.size(); ++i) {
        const TileRow& r = tr[i];
        if (r.xa >= r.xb) continue;
        const float opacity = tiles[i]->opacity;
        const float* r0 = tiles[i]->image().row(r.ty.i0);
        const float* r1 = tiles[i]->image().row(r.ty.i1);
        const TileColumn* cols = tile_cols_.data() + i * zs(w_);
        for (int x = r.xa; x < r.xb; ++x) {
          const TileColumn& c = cols[zs(x)];
          float w = c.band * r.band_b;
          w *= r.band_t;
          w *= c.feather_left;
          w *= c.feather_right;
          w *= r.feather_b;
          w *= r.feather_t;
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
  });
}

void Frame::distort(std::span<const Shock> shocks, const FieldBus& bus, Pool& pool) {
  // The bus's heat is looked up as FieldBus::at() does it (nothing beyond a cell outside the bus), by taps: its columns
  // once per screen column, its rows once per row. The haze's terms of x alone are made once per column.
  const float* heat = bus.heat().data();
  const int bnx = bus.nx(), bny = bus.ny();
  for (int x = 0; x < w_; ++x) {
    Column& k = cols_[zs(x)];
    k.wx = cam_x + fl(x) + 0.5f;
    const float gx = (k.wx - bus.x0()) / bus.cell() - 0.5f;
    k.on_bus = !(gx < -1.f || gx > fl(bnx));
    k.bus = tap(gx, bnx);
    k.wobble = std::sin(k.wx * 0.045f + time * 1.7f);
    k.phase = k.wx * 0.07f - time * 8.f;
  }
  pool.run((h_ + kChunk - 1) / kChunk, [&](int task) {
    std::array<float, kBlock> dxs{}, dys{};
    for (int y = task * kChunk; y < std::min(h_, (task + 1) * kChunk); ++y) {
      const float* src_row = screen_.row(y);
      float* out = tmp_.row(y);
      const float wy = cam_y + fl(y) + 0.5f;
      // hot air below shimmers what is seen through it; a row whose bus rows are all cool has none
      const float gy = (wy + 22.f - bus.y0()) / bus.cell() - 0.5f;
      const Tap by = tap(gy, bny);
      bool hazy = haze > 0.f && !(gy < -1.f || gy > fl(bny));
      if (hazy) {
        float most = -1e30f;
        for (int i = 0; i < bnx; ++i) most = std::max({most, heat[zs(by.i0) * zs(bnx) + zs(i)], heat[zs(by.i1) * zs(bnx) + zs(i)]});
        hazy = most >= 0.0099f;  // a blend of corners all below this stays below the threshold, 0.01, rounding included
      }
      const float haze_y = wy * 0.09f + time * 11.f, wobble_y = 1.5f * std::sin(wy * 0.05f);
      for (int bx = 0; bx < w_; bx += kBlock) {
        const int n = std::min(kBlock, w_ - bx);
        std::fill_n(dxs.begin(), n, 0.f);
        std::fill_n(dys.begin(), n, 0.f);
        bool moved = hazy;
        for (const Shock& s : shocks) {  // shock rings, over the columns each ring may cover in this row
          const float R = s.radius(time);
          if (R <= 0.f) continue;
          const float ey = wy - s.y, ay = std::fabs(ey);
          const float outer = R + s.width + 2.f, inner = R - s.width - 2.f;  // a margin: the test per pixel decides
          if (ay >= outer) continue;
          const float xo = std::sqrt(outer * outer - ey * ey), xi = inner > ay ? std::sqrt(inner * inner - ey * ey) : 0.f;
          const float centre = s.x - cam_x - 0.5f;  // screen column of the centre, continuous
          int spans[2][2] = {{ifloor(centre - xo) - 1, ifloor(centre - xi) + 2}, {ifloor(centre + xi) - 1, ifloor(centre + xo) + 2}};
          if (spans[1][0] <= spans[0][1]) {  // the two spans meet: one
            spans[0][1] = spans[1][1];
            spans[1][0] = spans[1][1];
          }
          const float amp = s.amp * std::exp(-(time - s.t0) / (2.f * s.decay));
          for (const auto& sp : spans) {
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
            const float h = lookup(heat, bnx, 1, 0, k.bus, by);
            if (h > 0.01f) {
              const float a = haze * std::min(1.f, 1.6f * h);
              dxs[zs(i)] += a * 1.6f * std::sin(haze_y + 2.f * k.wobble);
              dys[zs(i)] += a * 1.1f * std::sin(k.phase + wobble_y);
            }
          }
        }
        if (!moved) {
          std::copy_n(src_row + zs(bx) * 4, zs(n) * 4, out + zs(bx) * 4);
          continue;
        }
        for (int i = 0; i < n; ++i) {
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
          const float* a = screen_.row(y0) + zs(x0) * 4;
          const float* c = screen_.row(y0 + 1) + zs(x0) * 4;
          store4(o, (1.f - fy) * ((1.f - fx) * load4(a) + fx * load4(a + 4)) + fy * ((1.f - fx) * load4(c) + fx * load4(c + 4)));
        }
      }
    }
  });
  std::swap(screen_, tmp_);
}

void Frame::bloom(float threshold, float strength, Pool& pool) {
  // Bright pass into mip 0 (half size), then down, blur, and up. Every pass works on all four channels; channel 3 of
  // the mips stays 0, so what it adds to the screen's alpha is 0.
  const auto each_row = [&](const Image4& im, auto&& f) {
    if (zs(im.w) * zs(im.h) < 16384) {  // a small level: waking the workers would cost more than the work
      for (int y = 0; y < im.h; ++y) f(y);
      return;
    }
    pool.run((im.h + kChunk - 1) / kChunk, [&](int task) {
      for (int y = task * kChunk; y < std::min(im.h, (task + 1) * kChunk); ++y) f(y);
    });
  };
  Image4& m0 = mips_[0];
  each_row(m0, [&](int y) {
    const float* r0 = screen_.row(std::min(2 * y, h_ - 1));
    const float* r1 = screen_.row(std::min(2 * y + 1, h_ - 1));
    float* o = m0.row(y);
    std::array<float, 2 * kBlock> k0, k1;  // a quarter of the bright part of each source pixel, by source row
    for (int bx = 0; bx < m0.w; bx += kBlock) {
      const int n = std::min(kBlock, m0.w - bx), sx0 = 2 * bx, sn = std::min(2 * n, w_ - sx0);
      for (int j = 0; j < sn; ++j) {
        const float* p = r0 + zs(sx0 + j) * 4;
        const float* q = r1 + zs(sx0 + j) * 4;
        const float lp = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2], lq = 0.2126f * q[0] + 0.7152f * q[1] + 0.0722f * q[2];
        k0[zs(j)] = 0.25f * (std::max(0.f, lp - threshold) / std::max(lp, 1e-4f));
        k1[zs(j)] = 0.25f * (std::max(0.f, lq - threshold) / std::max(lq, 1e-4f));
      }
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
  });
  for (std::size_t l = 1; l < mips_.size(); ++l) {
    const Image4& s = mips_[l - 1];
    Image4& d = mips_[l];
    each_row(d, [&](int y) {
      const float* r0 = s.row(std::min(2 * y, s.h - 1));
      const float* r1 = s.row(std::min(2 * y + 1, s.h - 1));
      float* o = d.row(y);
      for (int x = 0; x < d.w; ++x) {
        const std::size_t a = zs(std::min(2 * x, s.w - 1)) * 4, b = zs(std::min(2 * x + 1, s.w - 1)) * 4;
        store4(o + zs(x) * 4, 0.25f * (load4(r0 + a) + load4(r0 + b) + load4(r1 + a) + load4(r1 + b)));
      }
    });
  }
  static constexpr float kw[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
  for (std::size_t l = 0; l < mips_.size(); ++l) {
    Image4& a = mips_[l];
    Image4& b = mips_tmp_[l];
    const int W = a.w;
    each_row(a, [&](int y) {  // across, a into b: the clamped ends, then the middle
      const float* s = a.row(y);
      float* o = b.row(y);
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
    });
    each_row(a, [&](int y) {  // down, b into a
      const float* r[5];
      for (int t = 0; t < 5; ++t) r[t] = b.row(std::clamp(y + t - 2, 0, a.h - 1));
      float* o = a.row(y);
      for (int i = 0; i < 4 * W; ++i) o[i] = kw[0] * r[0][i] + kw[1] * r[1][i] + kw[2] * r[2][i] + kw[3] * r[3][i] + kw[4] * r[4][i];
    });
  }
  const auto add_up = [&](float* o, const Image4& c, int y, int fw, std::size_t l, float gain, float div) {  // o += gain * c / div
    const Tap& ty = up_y_[l][zs(y)];
    const float* r0 = c.row(ty.i0);
    const float* r1 = c.row(ty.i1);
    const Tap* tx = up_x_[l].data();
    for (int x = 0; x < fw; ++x) {
      const Tap& t = tx[x];
      float* p = o + zs(x) * 4;
      store4(p, load4(p) + gain * blend4(r0 + zs(t.i0) * 4, r0 + zs(t.i1) * 4, r1 + zs(t.i0) * 4, r1 + zs(t.i1) * 4, t, ty) / div);
    }
  };
  for (std::size_t l = mips_.size() - 1; l > 0; --l) {  // up: each level adds the coarser one, bilinear
    Image4& f = mips_[l - 1];
    each_row(f, [&](int y) { add_up(f.row(y), mips_[l], y, f.w, l, 1.f, 1.f); });
  }
  each_row(screen_, [&](int y) { add_up(screen_.row(y), m0, y, w_, 0, strength, static_cast<float>(mips_.size())); });
}

namespace {

// Tone mapping of n values into levels of the display table: ACES of the value times the exposure, vignette and fade,
// plus grain.
void tone(const float* p, const float* vig, const float* grain, float exposure, float fade, int* level, int n) {
  const auto aces = [](float v) { return std::clamp(v * (2.51f * v + 0.03f) / (v * (2.43f * v + 0.59f) + 0.14f), 0.f, 1.f); };
  for (int i = 0; i < n; ++i) {
    const float v = aces(p[i] * exposure * vig[i] * fade) + grain[i];
    level[i] = std::clamp(static_cast<int>(v * 4095.f + 0.5f), 0, 4095);
  }
}

}  // namespace

void Frame::finish(std::span<std::uint8_t> rgb, Pool& pool) {
  const Gamma& g = gamma();
  const std::uint32_t salt = hz(static_cast<int>(time * 30.f));
  for (int x = 0; x < w_; ++x) {
    const float nx = ((fl(x) + 0.5f) / fl(w_) - 0.5f) * (fl(w_) / fl(h_));
    cols_[zs(x)].vig = nx * nx;
    cols_[zs(x)].grain = hx(x);
  }
  pool.run((h_ + kChunk - 1) / kChunk, [&](int task) {
    std::array<float, 4 * kBlock> vig, grain;  // per value of a block of pixels (all four channels, for vectors)
    std::array<int, 4 * kBlock> level;
    for (int y = task * kChunk; y < std::min(h_, (task + 1) * kChunk); ++y) {
      const float* p = screen_.row(y);
      std::uint8_t* o = rgb.data() + zs(y) * zs(w_) * 3;
      const float ny = (fl(y) + 0.5f) / fl(h_) - 0.5f, ny2 = ny * ny;
      const std::uint32_t key_y = hy(y) ^ salt;
      for (int bx = 0; bx < w_; bx += kBlock) {
        const int n = std::min(kBlock, w_ - bx);
        for (int i = 0; i < n; ++i) {
          const Column& k = cols_[zs(bx + i)];
          const float v = 1.f - 0.45f * (k.vig + ny2), gr = (unit(k.grain ^ key_y) - 0.5f) * 0.006f;
          for (int c = 0; c < 4; ++c) {
            vig[zs(i) * 4 + zs(c)] = v;
            grain[zs(i) * 4 + zs(c)] = gr;
          }
        }
        tone(p + zs(bx) * 4, vig.data(), grain.data(), exposure, fade, level.data(), 4 * n);
        std::uint8_t* ob = o + zs(bx) * 3;
        for (int i = 0; i < n; ++i) {
          for (int c = 0; c < 3; ++c) ob[zs(i) * 3 + zs(c)] = g.to_display[zs(level[zs(i) * 4 + zs(c)])];
        }
      }
    }
  });
}

}  // namespace nfx::compose
