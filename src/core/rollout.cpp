// Rollout effects (include/neuralfx/rollout.hpp): layouts, the .nvfx format with magic NVFXROL1, and the float
// reference of every operation. The trainer differentiates exactly these operations and the runtime reproduces them;
// tests/test_rollout.cpp holds both to this file.
#include <neuralfx/binio.hpp>
#include <neuralfx/rollout.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <stdfloat>

namespace nfx::rollout {

namespace {

constexpr std::uint32_t kVersion = 1;

float round_f16(float v) { return static_cast<float>(static_cast<std::float16_t>(v)); }

void put_f16(std::ostream& o, std::span<const float> v) {
  for (const float x : v) bin::put(o, std::bit_cast<std::uint16_t>(static_cast<std::float16_t>(x)));
}
std::expected<void, std::string> get_f16(std::istream& i, std::span<float> v) {
  for (float& x : v) {
    auto r = bin::get<std::uint16_t>(i);
    if (!r) return std::unexpected(r.error());
    x = static_cast<float>(std::bit_cast<std::float16_t>(*r));
  }
  return {};
}

// 8-bit fields: a scale (fp16) then v / scale * 255 rounded.
float field_scale(std::span<const float> v) {
  float mx = 0.f;
  for (const float x : v) mx = std::max(mx, x);
  return round_f16(std::max(mx, 1e-6f));
}
void put_u8_field(std::ostream& o, std::span<const float> v) {
  const float s = field_scale(v);
  put_f16(o, std::array{s});
  for (const float x : v) bin::put(o, static_cast<std::uint8_t>(std::clamp(x / s, 0.f, 1.f) * 255.f + 0.5f));
}
std::expected<void, std::string> get_u8_field(std::istream& i, std::span<float> v) {
  std::array<float, 1> s{};
  if (auto r = get_f16(i, s); !r) return r;
  for (float& x : v) {
    auto r = bin::get<std::uint8_t>(i);
    if (!r) return std::unexpected(r.error());
    x = static_cast<float>(*r) / 255.f * s[0];
  }
  return {};
}
void round_u8_field(std::vector<float>& v) {
  const float s = field_scale(v);
  for (float& x : v) x = std::round(std::clamp(x / s, 0.f, 1.f) * 255.f) / 255.f * s;
}

std::size_t sz(int v) { return static_cast<std::size_t>(v); }
float fl(int v) { return static_cast<float>(v); }

// Bilinear sample of an n x n field at cell-centre coordinates, clamped to the grid.
float bilinear(const float* f, int n, float x, float y) {
  x = std::clamp(x, 0.f, fl(n - 1));
  y = std::clamp(y, 0.f, fl(n - 1));
  const int x0 = std::min(static_cast<int>(x), n - 2), y0 = std::min(static_cast<int>(y), n - 2);
  const float fx = x - fl(x0), fy = y - fl(y0);
  const float* a = f + sz(y0) * sz(n) + sz(x0);
  return (1.f - fy) * ((1.f - fx) * a[0] + fx * a[1]) + fy * ((1.f - fx) * a[n] + fx * a[n + 1]);
}

// The same on a strided multi-channel field [cell][channels].
float bilinear_ch(const float* f, int n, int channels, int c, float x, float y) {
  x = std::clamp(x, 0.f, fl(n - 1));
  y = std::clamp(y, 0.f, fl(n - 1));
  const int x0 = std::min(static_cast<int>(x), n - 2), y0 = std::min(static_cast<int>(y), n - 2);
  const float fx = x - fl(x0), fy = y - fl(y0);
  const auto at = [&](int xx, int yy) { return f[(sz(yy) * sz(n) + sz(xx)) * sz(channels) + sz(c)]; };
  return (1.f - fy) * ((1.f - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1.f - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
}

float smoothstep01(float t) {
  t = std::clamp(t, 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}

}  // namespace

StepLayout step_layout(const Hyper& h) {
  StepLayout L{};
  std::size_t o = 0;
  const auto take = [&o](std::size_t n) {
    const std::size_t at = o;
    o += n;
    return at;
  };
  L.w1 = take(9 * sz(h.inputs()) * sz(h.hidden));
  L.b1 = take(sz(h.hidden));
  L.w2 = take(9 * sz(h.hidden) * sz(h.hidden));
  L.b2 = take(sz(h.hidden));
  L.wo = take(sz(h.hidden) * sz(h.outputs()));
  L.bo = take(sz(h.outputs()));
  L.g1 = take(sz(h.cond()) * sz(h.hidden));
  L.e1 = take(sz(h.cond()) * sz(h.hidden));
  L.g2 = take(sz(h.cond()) * sz(h.hidden));
  L.e2 = take(sz(h.cond()) * sz(h.hidden));
  L.size = o;
  return L;
}

RenderLayout render_layout(const Hyper& h) {
  RenderLayout L{};
  const std::size_t H = sz(h.render_hidden);
  std::size_t o = 0;
  L.w1 = o;
  o += H * kRenderIn;
  L.b1 = o;
  o += H;
  L.w2 = o;
  o += H * H;
  L.b2 = o;
  o += H;
  L.wo = o;
  o += 4 * H;
  L.bo = o;
  o += 4;
  L.size = o;
  return L;
}

std::size_t Model::storage_bytes() const {
  std::size_t n = 64 + 16 * control_names.size() + 2 * (step_w.size() + render_w.size());
  for (const auto& s : starts) {
    n += 16 + 2 * (s.controls.size() + s.coarse.size());
    if (!s.fine_t.empty()) n += 4 + s.fine_t.size() + s.fine_d.size();
  }
  return n;
}

Model init_model(const Hyper& h, std::uint64_t seed) {
  Model m;
  m.h = h;
  const StepLayout L = step_layout(h);
  m.step_w.assign(L.size, 0.f);
  std::mt19937_64 rng(seed);
  const auto fill = [&rng](std::vector<float>& w, std::size_t off, std::size_t n, float sd) {
    std::normal_distribution<float> nd(0.f, sd);
    for (std::size_t i = 0; i < n; ++i) w[off + i] = nd(rng);
  };
  fill(m.step_w, L.w1, 9 * sz(h.inputs()) * sz(h.hidden), std::sqrt(2.f / (9.f * fl(h.inputs()))));
  fill(m.step_w, L.w2, 9 * sz(h.hidden) * sz(h.hidden), std::sqrt(2.f / (9.f * fl(h.hidden))));
  fill(m.step_w, L.wo, sz(h.hidden) * sz(h.outputs()), 0.01f);
  const RenderLayout R = render_layout(h);
  m.render_w.assign(R.size, 0.f);
  fill(m.render_w, R.w1, sz(h.render_hidden) * kRenderIn, std::sqrt(2.f / kRenderIn));
  fill(m.render_w, R.w2, sz(h.render_hidden) * sz(h.render_hidden), std::sqrt(2.f / fl(h.render_hidden)));
  fill(m.render_w, R.wo, 4 * sz(h.render_hidden), std::sqrt(1.f / fl(h.render_hidden)));
  for (int k = 0; k < kPhys; ++k) {
    m.lo[sz(k)] = k < 2 ? -1e9f : 0.f;
    m.hi[sz(k)] = 1e9f;
  }
  return m;
}

// --- format ----------------------------------------------------------------------------------------------------------

std::expected<void, std::string> save_model(std::ostream& o, const Model& m) {
  const Hyper& h = m.h;
  o.write(kMagic, sizeof(kMagic));
  bin::put(o, kVersion);
  for (const int v : {h.res, h.hidden, h.memory, h.jacobi, h.n_controls, h.n_age, h.frames, h.render_hidden, h.start_fine, h.warmup}) {
    bin::put(o, static_cast<std::int32_t>(v));
  }
  bin::put_str(o, m.effect, 32);
  bin::put(o, m.fps);
  bin::put(o, static_cast<std::uint8_t>(m.loop ? 1 : 0));
  for (int k = 0; k < h.n_controls; ++k) {
    bin::put_str(o, sz(k) < m.control_names.size() ? m.control_names[sz(k)] : std::string{}, 16);
  }
  const NoiseSpec& n = m.noise;
  const DetailSpec& d = m.detail;
  for (const float v : {n.curl_scale, n.curl_rate, n.flicker_freq, n.flicker_rate, d.contrast, d.kappa, d.edge0, d.edge1, d.swirl,
                        d.swirl_scale, d.swirl_rate, d.swirl_ramp, m.qscale, m.render_scale[0], m.render_scale[1]}) {
    bin::put(o, v);
  }
  bin::put(o, static_cast<std::int32_t>(n.flicker_octaves));
  bin::put(o, static_cast<std::int32_t>(d.swirl_control));
  for (const auto* a : {&m.scale, &m.lo, &m.hi}) bin::put_array(o, std::span<const float>(*a));
  put_f16(o, m.step_w);
  put_f16(o, m.render_w);
  bin::put(o, static_cast<std::uint32_t>(m.starts.size()));
  for (const StartPoint& s : m.starts) {
    bin::put(o, s.seed);
    bin::put(o, s.time);
    put_f16(o, s.controls);
    put_f16(o, s.coarse);
    bin::put(o, static_cast<std::uint8_t>(s.fine_t.empty() ? 0 : 1));
    if (!s.fine_t.empty()) {
      put_u8_field(o, s.fine_t);
      put_u8_field(o, s.fine_d);
    }
  }
  if (!o) return std::unexpected("nvfx: write failed");
  return {};
}

std::expected<void, std::string> save_model(const std::filesystem::path& path, const Model& m) {
  std::ofstream o(path, std::ios::binary);
  if (!o) return std::unexpected("nvfx: cannot open " + path.string() + " for writing");
  return save_model(o, m);
}

std::expected<Model, std::string> load_model(std::istream& i) {
  char magic[8];
  i.read(magic, 8);
  if (!i || std::memcmp(magic, kMagic, 8) != 0) return std::unexpected("not a rollout effect");
  auto version = bin::get<std::uint32_t>(i);
  if (!version || *version != kVersion) return std::unexpected("unsupported rollout version");
  Model m;
  Hyper& h = m.h;
  for (int* v : {&h.res, &h.hidden, &h.memory, &h.jacobi, &h.n_controls, &h.n_age, &h.frames, &h.render_hidden, &h.start_fine, &h.warmup}) {
    auto r = bin::get<std::int32_t>(i);
    if (!r) return std::unexpected(r.error());
    *v = *r;
  }
  if (h.res < 8 || h.res > 256 || h.hidden < 1 || h.hidden > 256 || h.memory < 0 || h.memory > 64 || h.jacobi < 0 || h.jacobi > 1000 ||
      h.n_controls < 0 || h.n_controls > 8 || h.n_age < 0 || h.n_age > 2 || h.frames < 0 || h.render_hidden < 1 ||
      h.render_hidden > 256 || h.start_fine < 0 || h.start_fine > 1024 || h.warmup < 0 || h.warmup > 10000) {
    return std::unexpected("rollout: hyperparameters out of range");
  }
  auto name = bin::get_str(i, 32);
  if (!name) return std::unexpected(name.error());
  m.effect = *name;
  auto fps = bin::get<float>(i);
  auto loop = bin::get<std::uint8_t>(i);
  if (!fps || !loop || !(*fps > 0.f)) return std::unexpected("rollout: bad header");
  m.fps = *fps;
  m.loop = *loop != 0;
  for (int k = 0; k < h.n_controls; ++k) {
    auto s = bin::get_str(i, 16);
    if (!s) return std::unexpected(s.error());
    m.control_names.push_back(*s);
  }
  NoiseSpec& n = m.noise;
  DetailSpec& d = m.detail;
  for (float* v : {&n.curl_scale, &n.curl_rate, &n.flicker_freq, &n.flicker_rate, &d.contrast, &d.kappa, &d.edge0, &d.edge1, &d.swirl,
                   &d.swirl_scale, &d.swirl_rate, &d.swirl_ramp, &m.qscale, &m.render_scale[0], &m.render_scale[1]}) {
    auto r = bin::get<float>(i);
    if (!r) return std::unexpected(r.error());
    *v = *r;
  }
  auto oct = bin::get<std::int32_t>(i);
  auto sc = bin::get<std::int32_t>(i);
  if (!oct || !sc || *oct < 1 || *oct > 8 || *sc < -1 || *sc >= h.n_controls) return std::unexpected("rollout: bad noise spec");
  n.flicker_octaves = *oct;
  d.swirl_control = *sc;
  if (!(d.swirl_scale > 0.f) || !(d.edge1 > d.edge0)) return std::unexpected("rollout: bad detail spec");
  for (auto* a : {&m.scale, &m.lo, &m.hi}) {
    if (auto r = bin::get_array(i, std::span<float>(*a)); !r) return std::unexpected(r.error());
  }
  m.step_w.resize(step_layout(h).size);
  m.render_w.resize(render_layout(h).size);
  if (auto r = get_f16(i, m.step_w); !r) return std::unexpected(r.error());
  if (auto r = get_f16(i, m.render_w); !r) return std::unexpected(r.error());
  auto count = bin::get<std::uint32_t>(i);
  if (!count || *count < 1 || *count > 4096) return std::unexpected("rollout: bad start point count");
  for (std::uint32_t k = 0; k < *count; ++k) {
    StartPoint s;
    auto seed = bin::get<std::uint64_t>(i);
    auto time = bin::get<float>(i);
    if (!seed || !time) return std::unexpected("truncated file");
    s.seed = *seed;
    s.time = *time;
    s.controls.resize(sz(h.n_controls));
    s.coarse.resize(sz(h.res) * sz(h.res) * kPhys);
    if (auto r = get_f16(i, s.controls); !r) return std::unexpected(r.error());
    if (auto r = get_f16(i, s.coarse); !r) return std::unexpected(r.error());
    auto has_fine = bin::get<std::uint8_t>(i);
    if (!has_fine) return std::unexpected(has_fine.error());
    if (*has_fine) {
      if (h.start_fine < 2) return std::unexpected("rollout: fine fields without a size");
      s.fine_t.resize(sz(h.start_fine) * sz(h.start_fine));
      s.fine_d.resize(s.fine_t.size());
      if (auto r = get_u8_field(i, s.fine_t); !r) return std::unexpected(r.error());
      if (auto r = get_u8_field(i, s.fine_d); !r) return std::unexpected(r.error());
    }
    m.starts.push_back(std::move(s));
  }
  return m;
}

std::expected<Model, std::string> load_model(const std::filesystem::path& path) {
  std::ifstream i(path, std::ios::binary);
  if (!i) return std::unexpected("nvfx: cannot open " + path.string());
  return load_model(i);
}

bool is_rollout_file(std::span<const char> first_bytes) {
  return first_bytes.size() >= 8 && std::memcmp(first_bytes.data(), kMagic, 8) == 0;
}

void quantise_like_storage(Model& m) {
  const auto r16 = [](std::vector<float>& v) { std::ranges::for_each(v, [](float& x) { x = round_f16(x); }); };
  r16(m.step_w);
  r16(m.render_w);
  for (StartPoint& s : m.starts) {
    r16(s.controls);
    r16(s.coarse);
    if (!s.fine_t.empty()) {
      round_u8_field(s.fine_t);
      round_u8_field(s.fine_d);
    }
  }
}

// --- reference evaluation ---------------------------------------------------------------------------------------------

void condition(const Model& m, std::span<const float> controls, float seconds, std::span<float> out) {
  for (int k = 0; k < m.h.n_controls; ++k) out[sz(k)] = sz(k) < controls.size() ? controls[sz(k)] : 0.5f;
  if (m.h.n_age > 0) out[sz(m.h.n_controls)] = std::exp(-seconds / 0.15f);
  if (m.h.n_age > 1) out[sz(m.h.n_controls) + 1] = std::exp(-seconds / 1.0f);
}

void coarse_noise(const Model& m, std::uint64_t seed, float seconds, std::span<float> out) {
  const int R = m.h.res;
  const float k = 128.f / fl(R), t = seconds + 0.5f / m.fps;
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const float X = (fl(x) + 0.5f) * k + 0.5f, Y = (fl(y) + 0.5f) * k + 0.5f;
      const std::size_t i = (sz(y) * sz(R) + sz(x)) * kNoise;
      out[i] = noise_curl(m.noise, seed, X, Y, t);
      out[i + 1] = noise_flicker(m.noise, seed, X, Y, t);
    }
  }
}

namespace {

void conv3(int R, const float* in, int ci, const float* W, const float* b, int co, float* out) {
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      float* o = out + (sz(y) * sz(R) + sz(x)) * sz(co);
      std::copy(b, b + co, o);
      for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= R) continue;
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = x + dx;
          if (xx < 0 || xx >= R) continue;
          const float* a = in + (sz(yy) * sz(R) + sz(xx)) * sz(ci);
          const float* w = W + sz((dy + 1) * 3 + (dx + 1)) * sz(ci) * sz(co);
          for (int c = 0; c < ci; ++c) {
            for (int k = 0; k < co; ++k) o[k] += w[sz(c) * sz(co) + sz(k)] * a[c];
          }
        }
      }
    }
  }
}

}  // namespace

void coarse_step(const Model& m, std::span<const float> in, std::span<const float> noise, std::span<const float> cond,
                 std::span<float> pressure, std::span<float> out, std::span<float> flow) {
  const Hyper& h = m.h;
  const int R = h.res, N = R * R, C = h.channels(), I = h.inputs(), H = h.hidden, O = h.outputs();
  const StepLayout L = step_layout(h);
  const float* w = m.step_w.data();
  std::vector<float> X(sz(N) * sz(I)), y1(sz(N) * sz(H)), y2(sz(N) * sz(H)), d(sz(N) * sz(O)), mid(sz(N) * sz(C));
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const std::size_t i = sz(y) * sz(R) + sz(x);
      float* xi = X.data() + i * sz(I);
      for (int k = 0; k < C; ++k) xi[k] = in[i * sz(C) + sz(k)] / (k < kPhys ? m.scale[sz(k)] : 1.f);
      xi[C] = noise[i * kNoise];
      xi[C + 1] = noise[i * kNoise + 1];
      xi[C + 2] = (fl(x) + 0.5f) / fl(R) * 2.f - 1.f;
      xi[C + 3] = (fl(y) + 0.5f) / fl(R) * 2.f - 1.f;
    }
  }
  const auto film_relu = [&](std::vector<float>& v, std::size_t G, std::size_t E) {
    std::vector<float> g(sz(H), 1.f), e(sz(H), 0.f);
    for (int k = 0; k < h.cond(); ++k) {
      for (int j = 0; j < H; ++j) {
        g[sz(j)] += w[G + sz(k) * sz(H) + sz(j)] * cond[sz(k)];
        e[sz(j)] += w[E + sz(k) * sz(H) + sz(j)] * cond[sz(k)];
      }
    }
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < H; ++j) {
        float& z = v[sz(i) * sz(H) + sz(j)];
        z = std::max(0.f, g[sz(j)] * z + e[sz(j)]);
      }
    }
  };
  conv3(R, X.data(), I, w + L.w1, w + L.b1, H, y1.data());
  film_relu(y1, L.g1, L.e1);
  conv3(R, y1.data(), H, w + L.w2, w + L.b2, H, y2.data());
  film_relu(y2, L.g2, L.e2);
  for (int i = 0; i < N; ++i) {
    float* o = d.data() + sz(i) * sz(O);
    for (int k = 0; k < O; ++k) {
      float s = w[L.bo + sz(k)];
      for (int j = 0; j < H; ++j) s += w[L.wo + sz(j) * sz(O) + sz(k)] * y2[sz(i) * sz(H) + sz(j)];
      o[k] = s;
    }
    for (int k = 0; k < kPhys; ++k) mid[sz(i) * sz(C) + sz(k)] = in[sz(i) * sz(C) + sz(k)] + m.scale[sz(k)] * o[k];
    for (int k = kPhys; k < C; ++k) mid[sz(i) * sz(C) + sz(k)] = std::tanh(in[sz(i) * sz(C) + sz(k)] + o[k]);
  }
  // Pressure projection: divergence (velocity replicated at the edges) plus the learned source, warm-started Jacobi
  // with p = 0 outside, then the pressure gradient is subtracted.
  const auto U = [&](int x, int y, int c) { return mid[(sz(std::clamp(y, 0, R - 1)) * sz(R) + sz(std::clamp(x, 0, R - 1))) * sz(C) + sz(c)]; };
  std::vector<float> div(sz(N)), tmp(sz(N));
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      div[sz(y) * sz(R) + sz(x)] = -0.5f * (U(x + 1, y, 0) - U(x - 1, y, 0) + U(x, y + 1, 1) - U(x, y - 1, 1)) +
                                   m.qscale * d[(sz(y) * sz(R) + sz(x)) * sz(O) + sz(C)];
    }
  }
  const auto P = [&](int x, int y) { return (x < 0 || y < 0 || x >= R || y >= R) ? 0.f : pressure[sz(y) * sz(R) + sz(x)]; };
  for (int it = 0; it < h.jacobi; ++it) {
    for (int y = 0; y < R; ++y) {
      for (int x = 0; x < R; ++x) tmp[sz(y) * sz(R) + sz(x)] = 0.25f * (div[sz(y) * sz(R) + sz(x)] + P(x - 1, y) + P(x + 1, y) + P(x, y - 1) + P(x, y + 1));
    }
    std::copy(tmp.begin(), tmp.end(), pressure.begin());
  }
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const std::size_t i = sz(y) * sz(R) + sz(x);
      mid[i * sz(C)] -= 0.5f * (P(x + 1, y) - P(x - 1, y));
      mid[i * sz(C) + 1] -= 0.5f * (P(x, y + 1) - P(x, y - 1));
      flow[i * 2] = mid[i * sz(C)];
      flow[i * 2 + 1] = mid[i * sz(C) + 1];
    }
  }
  // Semi-Lagrangian advection of every channel by the projected velocity, then the state is kept in range.
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const std::size_t i = sz(y) * sz(R) + sz(x);
      const float bx = fl(x) - mid[i * sz(C)], by = fl(y) - mid[i * sz(C) + 1];
      for (int c = 0; c < C; ++c) {
        float v = bilinear_ch(mid.data(), R, C, c, bx, by);
        if (c < kPhys) v = std::clamp(v, m.lo[sz(c)], m.hi[sz(c)]);
        out[i * sz(C) + sz(c)] = v;
      }
    }
  }
}

namespace {

// The sub-grid swirl: a stream function sampled on a lattice of spacing swirl_scale / 2 (in pixels of a 128-pixel
// frame), its curl by central differences, bilinear between lattice nodes. Returns velocity in pixels of a 128-pixel
// frame per frame for a unit amplitude.
struct Swirl {
  int n = 0;
  float spacing = 1;
  std::vector<float> u, v;
};

Swirl swirl_field(const Model& m, std::uint64_t seed, float t) {
  Swirl s;
  const DetailSpec& d = m.detail;
  s.spacing = 0.5f * d.swirl_scale;
  s.n = static_cast<int>(std::ceil(130.f / s.spacing)) + 3;
  const int n = s.n;
  std::vector<float> psi(sz(n) * sz(n));
  const std::uint64_t sseed = seed * 0x9E3779B97F4A7C15ULL + 5;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) psi[sz(j) * sz(n) + sz(i)] = value_noise(fl(i - 1) * 0.5f, fl(j - 1) * 0.5f, t * d.swirl_rate, sseed);
  }
  s.u.assign(psi.size(), 0.f);
  s.v.assign(psi.size(), 0.f);
  for (int j = 1; j < n - 1; ++j) {
    for (int i = 1; i < n - 1; ++i) {
      const std::size_t k = sz(j) * sz(n) + sz(i);
      s.u[k] = psi[k + sz(n)] - psi[k - sz(n)];  // d psi / d y in lattice units of 0.5 noise cells
      s.v[k] = psi[k - 1] - psi[k + 1];
    }
  }
  return s;
}

}  // namespace

void detail_step(const Model& m, State& s, std::uint64_t seed, std::span<const float> controls) {
  const Hyper& h = m.h;
  const DetailSpec& dt = m.detail;
  const int R = h.res, S = s.size, C = h.channels();
  const float k = fl(S) / fl(R), px128 = fl(S) / 128.f, t = s.time + 0.5f / m.fps;
  float amp = dt.swirl * px128;
  if (dt.swirl_control >= 0 && sz(dt.swirl_control) < controls.size()) amp *= 0.3f + controls[sz(dt.swirl_control)];
  if (dt.swirl_ramp > 0.f) amp *= std::min(1.f, s.since_start / dt.swirl_ramp);
  Swirl sw;
  if (amp > 0.f) sw = swirl_field(m, seed, t);
  std::vector<float> ux(sz(S) * sz(S)), vy(sz(S) * sz(S));
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      const float xc = (fl(x) + 0.5f) / k - 0.5f, yc = (fl(y) + 0.5f) / k - 0.5f;
      float u = bilinear_ch(s.flow.data(), R, 2, 0, xc, yc) * k, v = bilinear_ch(s.flow.data(), R, 2, 1, xc, yc) * k;
      if (amp > 0.f) {
        const float lx = ((fl(x) + 0.5f) / px128 + 0.5f) / sw.spacing + 1.f, ly = ((fl(y) + 0.5f) / px128 + 0.5f) / sw.spacing + 1.f;
        u += amp * bilinear(sw.u.data(), sw.n, lx, ly);
        v += amp * bilinear(sw.v.data(), sw.n, lx, ly);
      }
      ux[sz(y) * sz(S) + sz(x)] = u;
      vy[sz(y) * sz(S) + sz(x)] = v;
    }
  }
  // MacCormack advection of the fine fields (clamped to the forward step's stencil, as in the simulation).
  std::vector<float> fa(sz(S) * sz(S)), fb(sz(S) * sz(S)), fo(sz(S) * sz(S));
  for (std::vector<float>* q : {&s.fine_t, &s.fine_d}) {
    const std::vector<float>& Q = *q;
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) {
        const std::size_t i = sz(y) * sz(S) + sz(x);
        fa[i] = bilinear(Q.data(), S, fl(x) - ux[i], fl(y) - vy[i]);
      }
    }
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) {
        const std::size_t i = sz(y) * sz(S) + sz(x);
        fb[i] = bilinear(fa.data(), S, fl(x) + ux[i], fl(y) + vy[i]);
      }
    }
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) {
        const std::size_t i = sz(y) * sz(S) + sz(x);
        const float bx = std::clamp(fl(x) - ux[i], 0.f, fl(S - 1)), by = std::clamp(fl(y) - vy[i], 0.f, fl(S - 1));
        const int x0 = std::min(static_cast<int>(bx), S - 2), y0 = std::min(static_cast<int>(by), S - 2);
        const float* a = Q.data() + sz(y0) * sz(S) + sz(x0);
        const float lo = std::min({a[0], a[1], a[S], a[S + 1]}), hi = std::max({a[0], a[1], a[S], a[S + 1]});
        fo[i] = std::max(0.f, std::clamp(fa[i] + 0.5f * (Q[i] - fb[i]), lo, hi));
      }
    }
    q->swap(fo);
  }
  // Lock to the coarse state: where the fine field holds more than the coarse cell, scale it down; where it holds
  // less, add the difference, broken up by the flicker noise (new material arrives as tongues, not as a smear).
  const int kk = S / R;
  std::vector<float> B(sz(R) * sz(R)), rr(sz(R) * sz(R)), aa(sz(R) * sz(R));
  for (int ch = 0; ch < 2; ++ch) {
    std::vector<float>& Q = ch == 0 ? s.fine_t : s.fine_d;
    std::ranges::fill(B, 0.f);
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) B[sz(y / kk) * sz(R) + sz(x / kk)] += Q[sz(y) * sz(S) + sz(x)];
    }
    for (int i = 0; i < R * R; ++i) {
      const float b = B[sz(i)] / fl(kk * kk), target = s.coarse[sz(i) * sz(C) + 2 + sz(ch)];
      constexpr float eps = 1e-4f;
      rr[sz(i)] = std::min(1.f, (target + eps) / (b + eps));
      aa[sz(i)] = std::max(0.f, target - b);
    }
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) {
        const float xc = (fl(x) + 0.5f) / k - 0.5f, yc = (fl(y) + 0.5f) / k - 0.5f;
        const std::size_t i = sz(y) * sz(S) + sz(x);
        float add = bilinear(aa.data(), R, xc, yc);
        if (add > 0.f && dt.contrast > 0.f) {
          const float X = (fl(x) + 0.5f) / px128 + 0.5f, Y = (fl(y) + 0.5f) / px128 + 0.5f;
          const float phi = noise_flicker(m.noise, seed, X, Y, t);
          add *= (1.f - dt.contrast) + dt.contrast * dt.kappa * smoothstep01((phi - dt.edge0) / (dt.edge1 - dt.edge0));
        }
        Q[i] = Q[i] * bilinear(rr.data(), R, xc, yc) + add;
      }
    }
  }
}

void render_features(const Model& m, const State& s, int x, int y, std::span<float> out) {
  const int R = m.h.res, S = s.size, C = m.h.channels();
  const float k = fl(S) / fl(R), xc = (fl(x) + 0.5f) / k - 0.5f, yc = (fl(y) + 0.5f) / k - 0.5f;
  const float it = 1.f / m.render_scale[0], id = 1.f / m.render_scale[1];
  out[0] = s.fine_t[sz(y) * sz(S) + sz(x)] * it;
  out[1] = s.fine_d[sz(y) * sz(S) + sz(x)] * id;
  out[2] = bilinear_ch(s.coarse.data(), R, C, 2, xc, yc) * it;
  out[3] = bilinear_ch(s.coarse.data(), R, C, 3, xc, yc) * id;
  // Directional soot sums on the coarse grid (zero outside), interpolated like the fields.
  static constexpr std::array<std::array<int, 2>, kDirs> dirs{{{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}};
  const auto dirsum = [&](int cx, int cy, int j) {
    float sum = 0.f;
    for (int st = 1; st <= kDirSteps; ++st) {
      const int xx = cx + st * dirs[sz(j)][0], yy = cy + st * dirs[sz(j)][1];
      if (xx >= 0 && yy >= 0 && xx < R && yy < R) sum += s.coarse[(sz(yy) * sz(R) + sz(xx)) * sz(C) + 3];
    }
    return sum;
  };
  const float cx = std::clamp(xc, 0.f, fl(R - 1)), cy = std::clamp(yc, 0.f, fl(R - 1));
  const int x0 = std::min(static_cast<int>(cx), R - 2), y0 = std::min(static_cast<int>(cy), R - 2);
  const float fx = cx - fl(x0), fy = cy - fl(y0);
  for (int j = 0; j < kDirs; ++j) {
    const float a = (1.f - fx) * dirsum(x0, y0, j) + fx * dirsum(x0 + 1, y0, j);
    const float b = (1.f - fx) * dirsum(x0, y0 + 1, j) + fx * dirsum(x0 + 1, y0 + 1, j);
    out[4 + sz(j)] = ((1.f - fy) * a + fy * b) * id;
  }
}

std::array<float, 4> render_mlp(const Model& m, std::span<const float> f) {
  const RenderLayout L = render_layout(m.h);
  const int H = m.h.render_hidden;
  const float* w = m.render_w.data();
  std::vector<float> h1(sz(H)), h2(sz(H));
  for (int j = 0; j < H; ++j) {
    float s = w[L.b1 + sz(j)];
    for (int i = 0; i < kRenderIn; ++i) s += w[L.w1 + sz(j) * kRenderIn + sz(i)] * f[sz(i)];
    h1[sz(j)] = std::max(0.f, s);
  }
  for (int j = 0; j < H; ++j) {
    float s = w[L.b2 + sz(j)];
    for (int i = 0; i < H; ++i) s += w[L.w2 + sz(j) * sz(H) + sz(i)] * h1[sz(i)];
    h2[sz(j)] = std::max(0.f, s);
  }
  std::array<float, 4> o{};
  const float g = render_gate(f[0], f[1]);
  for (int c = 0; c < 4; ++c) {
    float s = w[L.bo + sz(c)];
    for (int i = 0; i < H; ++i) s += w[L.wo + sz(c) * sz(H) + sz(i)] * h2[sz(i)];
    o[sz(c)] = s * g;
  }
  return o;
}

void render(const Model& m, const State& s, std::span<float> rgba) {
  const int S = s.size;
  std::array<float, kRenderIn> f{};
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      render_features(m, s, x, y, f);
      const auto o = render_mlp(m, f);
      float* p = rgba.data() + (sz(S - 1 - y) * sz(S) + sz(x)) * 4;
      for (int c = 0; c < 4; ++c) p[c] = std::clamp(o[sz(c)], 0.f, 1.f);
    }
  }
}

State start(const Model& m, int index, int size, std::span<const float> controls, std::uint64_t seed) {
  const Hyper& h = m.h;
  const StartPoint& sp = m.starts.at(sz(index));
  const int R = h.res, C = h.channels();
  State s;
  s.res = R;
  s.size = size;
  s.time = sp.time;
  s.coarse.assign(sz(R) * sz(R) * sz(C), 0.f);
  for (int i = 0; i < R * R; ++i) {
    for (int c = 0; c < kPhys; ++c) s.coarse[sz(i) * sz(C) + sz(c)] = sp.coarse[sz(i) * kPhys + sz(c)];
  }
  s.pressure.assign(sz(R) * sz(R), 0.f);
  s.flow.assign(sz(R) * sz(R) * 2, 0.f);
  s.fine_t.assign(sz(size) * sz(size), 0.f);
  s.fine_d.assign(sz(size) * sz(size), 0.f);
  const bool fine = !sp.fine_t.empty();
  const int n = fine ? h.start_fine : R;
  const float k = fl(size) / fl(n);
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const float xs = (fl(x) + 0.5f) / k - 0.5f, ys = (fl(y) + 0.5f) / k - 0.5f;
      const std::size_t i = sz(y) * sz(size) + sz(x);
      s.fine_t[i] = fine ? bilinear(sp.fine_t.data(), n, xs, ys) : bilinear_ch(sp.coarse.data(), R, kPhys, 2, xs, ys);
      s.fine_d[i] = fine ? bilinear(sp.fine_d.data(), n, xs, ys) : bilinear_ch(sp.coarse.data(), R, kPhys, 3, xs, ys);
    }
  }
  if (!fine) {  // grow detail: material flows through and picks up filaments and flicker
    s.since_start = m.detail.swirl_ramp;
    for (int f = 0; f < h.warmup; ++f) step(m, s, controls, seed);
    s.since_start = m.detail.swirl_ramp;
  }
  return s;
}

void step(const Model& m, State& s, std::span<const float> controls, std::uint64_t seed) {
  const Hyper& h = m.h;
  std::vector<float> cond(sz(h.cond())), noise(sz(h.res) * sz(h.res) * kNoise), next(s.coarse.size());
  condition(m, controls, s.time, cond);
  coarse_noise(m, seed, s.time, noise);
  coarse_step(m, s.coarse, noise, cond, s.pressure, next, s.flow);
  s.coarse.swap(next);
  detail_step(m, s, seed, controls);
  s.time += 1.f / m.fps;
  s.since_start += 1.f / m.fps;
}

}  // namespace nfx::rollout
