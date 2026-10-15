// Study N's 3D rollout effect (include/neuralfx/rollout3d.hpp): the layout, the .nvfx3 format (magic NVFXVOL1) and the
// float reference of every operation. The trainer (src/train/rollout3d_train.cpp) differentiates exactly these coarse
// operations; the volume runtime (src/runtime/nvfx_volume.cpp) runs this code with its own convolution kernel and
// threads. Every loop writes per cell or per slab, so a pool never changes a value.
#include <neuralfx/binio.hpp>
#include <neuralfx/rollout3d.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <stdfloat>

namespace nfx::rollout3d {

namespace {

constexpr std::uint32_t kVersion = 1;

std::size_t zu(int v) { return static_cast<std::size_t>(v); }
float fl(int v) { return static_cast<float>(v); }
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

// 8-bit start states: per channel an fp16 (lo, hi), then one code per cell; a channel whose range starts at zero keeps
// its empty cells exactly empty.
struct Range {
  float lo, hi;
};
Range channel_range(std::span<const float> coarse, int c) {
  float mn = 1e30f, mx = -1e30f;
  for (std::size_t i = zu(c); i < coarse.size(); i += kPhys) {
    mn = std::min(mn, coarse[i]);
    mx = std::max(mx, coarse[i]);
  }
  const float lo = round_f16(mn);
  float hi = round_f16(std::max(mx, mn + 1e-6f));
  if (hi <= lo) hi = round_f16(lo + std::max(1e-3f, std::abs(lo) * 1e-3f));
  return {lo, hi};
}
std::uint8_t code8(float v, Range r) { return static_cast<std::uint8_t>(std::clamp(std::round((v - r.lo) / (r.hi - r.lo) * 255.f), 0.f, 255.f)); }
float value8(std::uint8_t q, Range r) { return q == 0 && r.lo == 0.f ? 0.f : r.lo + static_cast<float>(q) / 255.f * (r.hi - r.lo); }

float smoothstep01(float t) {
  t = std::clamp(t, 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}

// Trilinear sample of an nx x ny x nz field [cell][channels] at cell-centre coordinates, clamped to the grid.
float trilinear(const float* f, int nx, int ny, int nz, int channels, int c, float x, float y, float z) {
  x = std::clamp(x, 0.f, fl(nx - 1));
  y = std::clamp(y, 0.f, fl(ny - 1));
  z = std::clamp(z, 0.f, fl(nz - 1));
  const int x0 = std::min(static_cast<int>(x), nx - 2), y0 = std::min(static_cast<int>(y), ny - 2), z0 = std::min(static_cast<int>(z), nz - 2);
  const float fx = x - fl(x0), fy = y - fl(y0), fz = z - fl(z0);
  const std::size_t sx = zu(channels), sz = zu(nx) * sx, sy = zu(nz) * sz;
  const float* p = f + zu(y0) * sy + zu(z0) * sz + zu(x0) * sx + zu(c);
  const float a00 = p[0] + fx * (p[sx] - p[0]), a10 = p[sz] + fx * (p[sz + sx] - p[sz]);
  const float a01 = p[sy] + fx * (p[sy + sx] - p[sy]), a11 = p[sy + sz] + fx * (p[sy + sz + sx] - p[sy + sz]);
  const float b0 = a00 + fz * (a10 - a00), b1 = a01 + fz * (a11 - a01);
  return b0 + fy * (b1 - b0);
}

// Trilinear sample of a single-channel field with zero outside it (material leaves and nothing flows in), and the range
// of the eight values used.
float trilinear_zero(const float* f, int nx, int ny, int nz, float x, float y, float z, float* lo = nullptr, float* hi = nullptr) {
  x = std::clamp(x, -1.f, fl(nx));
  y = std::clamp(y, -1.f, fl(ny));
  z = std::clamp(z, -1.f, fl(nz));
  const float fx0 = std::floor(x), fy0 = std::floor(y), fz0 = std::floor(z);
  const int x0 = static_cast<int>(fx0), y0 = static_cast<int>(fy0), z0 = static_cast<int>(fz0);
  const float fx = x - fx0, fy = y - fy0, fz = z - fz0;
  float v[8];
  for (int c = 0; c < 8; ++c) {
    const int xx = x0 + (c & 1), zz = z0 + ((c >> 1) & 1), yy = y0 + ((c >> 2) & 1);
    v[c] = (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz) ? 0.f : f[(zu(yy) * zu(nz) + zu(zz)) * zu(nx) + zu(xx)];
  }
  if (lo) *lo = std::min({v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]});
  if (hi) *hi = std::max({v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]});
  const float a0 = v[0] + fx * (v[1] - v[0]), a1 = v[2] + fx * (v[3] - v[2]), a2 = v[4] + fx * (v[5] - v[4]), a3 = v[6] + fx * (v[7] - v[6]);
  const float b0 = a0 + fz * (a1 - a0), b1 = a2 + fz * (a3 - a2);
  return b0 + fy * (b1 - b0);
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
  L.w1 = take(kTaps * zu(h.inputs()) * zu(h.hidden));
  L.b1 = take(zu(h.hidden));
  L.w2 = take(kTaps * zu(h.hidden) * zu(h.hidden));
  L.b2 = take(zu(h.hidden));
  L.wo = take(zu(h.hidden) * zu(h.outputs()));
  L.bo = take(zu(h.outputs()));
  L.g1 = take(zu(h.cond()) * zu(h.hidden));
  L.e1 = take(zu(h.cond()) * zu(h.hidden));
  L.g2 = take(zu(h.cond()) * zu(h.hidden));
  L.e2 = take(zu(h.cond()) * zu(h.hidden));
  L.size = o;
  return L;
}

std::size_t Model::storage_bytes() const {
  std::size_t n = 64 + 16 * control_names.size() + 2 * step_w.size();
  for (const auto& s : starts) n += 16 + 2 * s.controls.size() + kPhys * 4 + s.coarse.size();
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
  fill(m.step_w, L.w1, kTaps * zu(h.inputs()) * zu(h.hidden), std::sqrt(2.f / (27.f * fl(h.inputs()))));
  fill(m.step_w, L.w2, kTaps * zu(h.hidden) * zu(h.hidden), std::sqrt(2.f / (27.f * fl(h.hidden))));
  fill(m.step_w, L.wo, zu(h.hidden) * zu(h.outputs()), 0.01f);
  for (int k = 0; k < kPhys; ++k) {
    m.lo[zu(k)] = k < 3 ? -1e9f : 0.f;
    m.hi[zu(k)] = 1e9f;
  }
  m.control_names = {"intensity", "wind", "turbulence"};
  m.control_names.resize(zu(h.n_controls));
  return m;
}

// --- format ----------------------------------------------------------------------------------------------------------

std::expected<void, std::string> save_model(std::ostream& o, const Model& m) {
  const Hyper& h = m.h;
  o.write(kMagic, sizeof(kMagic));
  bin::put(o, kVersion);
  for (const int v : {h.rx, h.ry, h.rz, h.fine, h.hidden, h.memory, h.jacobi, h.n_controls, h.warmup}) bin::put(o, static_cast<std::int32_t>(v));
  bin::put_str(o, m.effect, 32);
  bin::put(o, m.fps);
  for (int k = 0; k < h.n_controls; ++k) bin::put_str(o, zu(k) < m.control_names.size() ? m.control_names[zu(k)] : std::string{}, 16);
  const Forcing3& n = m.noise;
  const DetailSpec& d = m.detail;
  for (const float v : {n.curl_length, n.curl_rate, n.flicker_freq, n.flicker_rate, d.contrast, d.kappa, d.edge0, d.edge1, d.swirl, d.swirl_scale,
                        d.swirl_rate, d.swirl_ramp, d.grow, m.qscale}) {
    bin::put(o, v);
  }
  bin::put(o, static_cast<std::int32_t>(n.flicker_octaves));
  for (const auto* a : {&m.scale, &m.lo, &m.hi}) bin::put_array(o, std::span<const float>(*a));
  put_f16(o, m.step_w);
  bin::put(o, static_cast<std::uint32_t>(m.starts.size()));
  std::vector<std::uint8_t> codes(zu(h.cells()));
  for (const StartPoint& s : m.starts) {
    if (s.coarse.size() != zu(h.cells()) * kPhys || s.controls.size() != zu(h.n_controls)) return std::unexpected("rollout3d: a start point has the wrong size");
    bin::put(o, s.seed);
    bin::put(o, s.time);
    put_f16(o, s.controls);
    for (int c = 0; c < kPhys; ++c) {
      const Range r = channel_range(s.coarse, c);
      put_f16(o, std::array{r.lo, r.hi});
      for (int i = 0; i < h.cells(); ++i) codes[zu(i)] = code8(s.coarse[zu(i) * kPhys + zu(c)], r);
      o.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(codes.size()));
    }
  }
  if (!o) return std::unexpected("nvfx3: write failed");
  return {};
}

std::expected<void, std::string> save_model(const std::filesystem::path& path, const Model& m) {
  std::ofstream o(path, std::ios::binary);
  if (!o) return std::unexpected("nvfx3: cannot open " + path.string() + " for writing");
  return save_model(o, m);
}

std::expected<Model, std::string> load_model(std::istream& i) {
  char magic[8];
  i.read(magic, 8);
  if (!i || std::memcmp(magic, kMagic, 8) != 0) return std::unexpected("not a 3D rollout effect");
  auto version = bin::get<std::uint32_t>(i);
  if (!version || *version != kVersion) return std::unexpected("unsupported 3D rollout version");
  Model m;
  Hyper& h = m.h;
  for (int* v : {&h.rx, &h.ry, &h.rz, &h.fine, &h.hidden, &h.memory, &h.jacobi, &h.n_controls, &h.warmup}) {
    auto r = bin::get<std::int32_t>(i);
    if (!r) return std::unexpected(r.error());
    *v = *r;
  }
  if (h.rx < 4 || h.rx > 128 || h.ry < 4 || h.ry > 192 || h.rz < 4 || h.rz > 128 || h.fine < 1 || h.fine > 8 || h.hidden < 1 || h.hidden > 128 ||
      h.memory < 0 || h.memory > 32 || h.jacobi < 0 || h.jacobi > 1000 || h.n_controls < 0 || h.n_controls > 8 || h.warmup < 0 || h.warmup > 10000 ||
      h.fine_cells() > (std::size_t{1} << 26)) {
    return std::unexpected("rollout3d: hyperparameters out of range");
  }
  auto name = bin::get_str(i, 32);
  auto fps = bin::get<float>(i);
  if (!name || !fps || !(*fps > 0.f)) return std::unexpected("rollout3d: bad header");
  m.effect = *name;
  m.fps = *fps;
  for (int k = 0; k < h.n_controls; ++k) {
    auto s = bin::get_str(i, 16);
    if (!s) return std::unexpected(s.error());
    m.control_names.push_back(*s);
  }
  Forcing3& n = m.noise;
  DetailSpec& d = m.detail;
  for (float* v : {&n.curl_length, &n.curl_rate, &n.flicker_freq, &n.flicker_rate, &d.contrast, &d.kappa, &d.edge0, &d.edge1, &d.swirl, &d.swirl_scale,
                   &d.swirl_rate, &d.swirl_ramp, &d.grow, &m.qscale}) {
    auto r = bin::get<float>(i);
    if (!r) return std::unexpected(r.error());
    *v = *r;
  }
  auto oct = bin::get<std::int32_t>(i);
  if (!oct || *oct < 1 || *oct > 8) return std::unexpected("rollout3d: bad noise spec");
  n.flicker_octaves = *oct;
  if (!(n.curl_length > 1.f) || !(d.swirl_scale > 1.f) || !(d.edge1 > d.edge0) || !(d.grow >= 1.f)) return std::unexpected("rollout3d: bad constants");
  for (auto* a : {&m.scale, &m.lo, &m.hi}) {
    if (auto r = bin::get_array(i, std::span<float>(*a)); !r) return std::unexpected(r.error());
  }
  m.step_w.resize(step_layout(h).size);
  if (auto r = get_f16(i, m.step_w); !r) return std::unexpected(r.error());
  auto count = bin::get<std::uint32_t>(i);
  if (!count || *count < 1 || *count > 4096) return std::unexpected("rollout3d: bad start point count");
  std::vector<std::uint8_t> codes(zu(h.cells()));
  for (std::uint32_t k = 0; k < *count; ++k) {
    StartPoint s;
    auto seed = bin::get<std::uint64_t>(i);
    auto time = bin::get<float>(i);
    if (!seed || !time) return std::unexpected("truncated file");
    s.seed = *seed;
    s.time = *time;
    s.controls.resize(zu(h.n_controls));
    if (auto r = get_f16(i, s.controls); !r) return std::unexpected(r.error());
    s.coarse.resize(zu(h.cells()) * kPhys);
    for (int c = 0; c < kPhys; ++c) {
      std::array<float, 2> r{};
      if (auto e = get_f16(i, r); !e) return std::unexpected(e.error());
      i.read(reinterpret_cast<char*>(codes.data()), static_cast<std::streamsize>(codes.size()));
      if (!i) return std::unexpected("truncated file");
      for (int j = 0; j < h.cells(); ++j) s.coarse[zu(j) * kPhys + zu(c)] = value8(codes[zu(j)], {r[0], r[1]});
    }
    m.starts.push_back(std::move(s));
  }
  return m;
}

std::expected<Model, std::string> load_model(const std::filesystem::path& path) {
  std::ifstream i(path, std::ios::binary);
  if (!i) return std::unexpected("nvfx3: cannot open " + path.string());
  return load_model(i);
}

void quantise_like_storage(Model& m) {
  for (float& w : m.step_w) w = round_f16(w);
  for (StartPoint& s : m.starts) {
    for (float& c : s.controls) c = round_f16(c);
    for (int c = 0; c < kPhys; ++c) {
      const Range r = channel_range(s.coarse, c);
      for (std::size_t j = zu(c); j < s.coarse.size(); j += kPhys) s.coarse[j] = value8(code8(s.coarse[j], r), r);
    }
  }
}

// --- the reference ---------------------------------------------------------------------------------------------------

void conv3_reference(int rx, int ry, int rz, const float* in, int ci, const float* W, const float* b, int co, float* out, int y0, int y1) {
  for (int y = y0; y < y1; ++y) {
    for (int z = 0; z < rz; ++z) {
      for (int x = 0; x < rx; ++x) {
        float* o = out + ((zu(y) * zu(rz) + zu(z)) * zu(rx) + zu(x)) * zu(co);
        for (int k = 0; k < co; ++k) o[k] = b[k];
        for (int dy = -1; dy <= 1; ++dy) {
          const int yy = y + dy;
          if (yy < 0 || yy >= ry) continue;
          for (int dz = -1; dz <= 1; ++dz) {
            const int zz = z + dz;
            if (zz < 0 || zz >= rz) continue;
            for (int dx = -1; dx <= 1; ++dx) {
              const int xx = x + dx;
              if (xx < 0 || xx >= rx) continue;
              const float* a = in + ((zu(yy) * zu(rz) + zu(zz)) * zu(rx) + zu(xx)) * zu(ci);
              const float* w = W + zu(((dy + 1) * 3 + (dz + 1)) * 3 + (dx + 1)) * zu(ci) * zu(co);
              for (int c = 0; c < ci; ++c) {
                const float av = a[c];
                const float* wr = w + zu(c) * zu(co);
                for (int k = 0; k < co; ++k) o[k] += wr[k] * av;
              }
            }
          }
        }
      }
    }
  }
}

Workspace::Workspace(const Model& m) {
  const Hyper& h = m.h;
  const std::size_t N = zu(h.cells());
  X.assign(N * zu(h.inputs()), 0.f);
  y1.assign(N * zu(h.hidden), 0.f);
  y2.assign(N * zu(h.hidden), 0.f);
  d.assign(N * zu(h.outputs()), 0.f);
  mid.assign(N * zu(h.channels()), 0.f);
  div.assign(N, 0.f);
  tmp.assign(N, 0.f);
  noise.assign(N * kNoise, 0.f);
  cond.assign(zu(h.cond()), 0.f);
  next.assign(N * zu(h.channels()), 0.f);
  const std::size_t F = h.fine_cells();
  ufine.assign(F * 3, 0.f);
  fa.assign(F, 0.f);
  fb.assign(F, 0.f);
  lo.assign(F, 0.f);
  hi.assign(F, 0.f);
  B.assign(N, 0.f);
  rr.assign(N, 0.f);
  aa.assign(N, 0.f);
  const float H = 128.f * fl(h.ry) / fl(h.rx), D = 128.f * fl(h.rz) / fl(h.rx);
  curl.resize(m.noise, 128.f, H, D);
  Forcing3 sw = m.noise;
  sw.curl_length = m.detail.swirl_scale;
  sw.curl_rate = m.detail.swirl_rate;
  swirl.resize(sw, 128.f, H, D);
}

State make_state(const Model& m) {
  const Hyper& h = m.h;
  State s;
  s.coarse.assign(zu(h.cells()) * zu(h.channels()), 0.f);
  s.pressure.assign(zu(h.cells()), 0.f);
  s.flow.assign(zu(h.cells()) * 3, 0.f);
  s.fine_t.assign(h.fine_cells(), 0.f);
  s.fine_d.assign(h.fine_cells(), 0.f);
  return s;
}

void coarse_noise(const Model& m, std::uint64_t seed, float seconds, Workspace& ws, std::span<float> out, Pool* pool) {
  const Hyper& h = m.h;
  const float t = seconds + 0.5f / m.fps, k = 128.f / fl(h.rx);
  ws.curl.build(m.noise, seed, t, pool);
  parallel_for(pool, h.ry, [&](int y) {
    for (int z = 0; z < h.rz; ++z) {
      for (int x = 0; x < h.rx; ++x) {
        const float X = (fl(x) + 0.5f) * k, Y = (fl(y) + 0.5f) * k, Z = (fl(z) + 0.5f) * k;
        const auto c = ws.curl.sample(X, Y, Z);
        float* o = out.data() + ((zu(y) * zu(h.rz) + zu(z)) * zu(h.rx) + zu(x)) * kNoise;
        o[0] = c[0];
        o[1] = c[1];
        o[2] = c[2];
        o[3] = flicker3(m.noise, seed, X, Y, Z, t);
      }
    }
  });
}

void coarse_step(const Model& m, std::span<const float> in, std::span<const float> noise, std::span<const float> cond, std::span<float> pressure,
                 std::span<float> out, std::span<float> flow, Workspace& ws, Pool* pool) {
  const Hyper& h = m.h;
  const int rx = h.rx, ry = h.ry, rz = h.rz, C = h.channels(), I = h.inputs(), H = h.hidden, O = h.outputs();
  const StepLayout L = step_layout(h);
  const float* w = m.step_w.data();
  const auto at = [&](int x, int y, int z) { return (zu(y) * zu(rz) + zu(z)) * zu(rx) + zu(x); };
  // inputs
  parallel_for(pool, ry, [&](int y) {
    for (int z = 0; z < rz; ++z) {
      for (int x = 0; x < rx; ++x) {
        const std::size_t i = at(x, y, z);
        float* xi = ws.X.data() + i * zu(I);
        for (int k = 0; k < C; ++k) xi[k] = in[i * zu(C) + zu(k)] / (k < kPhys ? m.scale[zu(k)] : 1.f);
        for (int k = 0; k < kNoise; ++k) xi[C + k] = noise[i * kNoise + zu(k)];
        xi[C + kNoise] = (fl(x) + 0.5f) / fl(rx) * 2.f - 1.f;
        xi[C + kNoise + 1] = (fl(y) + 0.5f) / fl(ry) * 2.f - 1.f;
        xi[C + kNoise + 2] = (fl(z) + 0.5f) / fl(rz) * 2.f - 1.f;
      }
    }
  });
  // FiLM gains and shifts
  float g1[128], e1[128], g2[128], e2[128];
  for (int j = 0; j < H; ++j) {
    g1[j] = 1.f;
    e1[j] = 0.f;
    g2[j] = 1.f;
    e2[j] = 0.f;
    for (int k = 0; k < h.cond(); ++k) {
      const float c = cond[zu(k)];
      g1[j] += w[L.g1 + zu(k) * zu(H) + zu(j)] * c;
      e1[j] += w[L.e1 + zu(k) * zu(H) + zu(j)] * c;
      g2[j] += w[L.g2 + zu(k) * zu(H) + zu(j)] * c;
      e2[j] += w[L.e2 + zu(k) * zu(H) + zu(j)] * c;
    }
  }
  const std::size_t slab = zu(rx) * zu(rz);
  parallel_for(pool, ry, [&](int y) {
    ws.conv(rx, ry, rz, ws.X.data(), I, w + L.w1, w + L.b1, H, ws.y1.data(), y, y + 1);
    for (std::size_t i = zu(y) * slab; i < zu(y + 1) * slab; ++i) {
      float* v = ws.y1.data() + i * zu(H);
      for (int j = 0; j < H; ++j) v[j] = std::max(0.f, g1[j] * v[j] + e1[j]);
    }
  });
  parallel_for(pool, ry, [&](int y) {
    ws.conv(rx, ry, rz, ws.y1.data(), H, w + L.w2, w + L.b2, H, ws.y2.data(), y, y + 1);
    for (std::size_t i = zu(y) * slab; i < zu(y + 1) * slab; ++i) {
      float* v = ws.y2.data() + i * zu(H);
      for (int j = 0; j < H; ++j) v[j] = std::max(0.f, g2[j] * v[j] + e2[j]);
      float* o = ws.d.data() + i * zu(O);
      for (int k = 0; k < O; ++k) {
        float s = w[L.bo + zu(k)];
        for (int j = 0; j < H; ++j) s += w[L.wo + zu(j) * zu(O) + zu(k)] * v[j];
        o[k] = s;
      }
      for (int k = 0; k < kPhys; ++k) ws.mid[i * zu(C) + zu(k)] = in[i * zu(C) + zu(k)] + m.scale[zu(k)] * o[k];
      for (int k = kPhys; k < C; ++k) ws.mid[i * zu(C) + zu(k)] = std::tanh(in[i * zu(C) + zu(k)] + o[k]);
    }
  });
  // the projection: divergence (velocity replicated at the edges) plus the learned source; Jacobi, warm-started, with
  // p = 0 outside the sides and the top and a closed floor (the cell below the floor is the floor cell)
  const auto U = [&](int x, int y, int z, int c) {
    return ws.mid[at(std::clamp(x, 0, rx - 1), std::clamp(y, 0, ry - 1), std::clamp(z, 0, rz - 1)) * zu(C) + zu(c)];
  };
  parallel_for(pool, ry, [&](int y) {
    for (int z = 0; z < rz; ++z) {
      for (int x = 0; x < rx; ++x) {
        ws.div[at(x, y, z)] = -0.5f * (U(x + 1, y, z, 0) - U(x - 1, y, z, 0) + U(x, y + 1, z, 1) - U(x, y - 1, z, 1) + U(x, y, z + 1, 2) - U(x, y, z - 1, 2)) +
                              m.qscale * ws.d[at(x, y, z) * zu(O) + zu(C)];
      }
    }
  });
  const auto P = [&](int x, int y, int z) {
    if (x < 0 || z < 0 || x >= rx || z >= rz || y >= ry) return 0.f;
    return pressure[at(x, std::max(y, 0), z)];
  };
  for (int it = 0; it < h.jacobi; ++it) {
    parallel_for(pool, ry, [&](int y) {
      for (int z = 0; z < rz; ++z) {
        for (int x = 0; x < rx; ++x) {
          ws.tmp[at(x, y, z)] = (1.f / 6.f) * (ws.div[at(x, y, z)] + P(x - 1, y, z) + P(x + 1, y, z) + P(x, y - 1, z) + P(x, y + 1, z) + P(x, y, z - 1) + P(x, y, z + 1));
        }
      }
    });
    std::copy(ws.tmp.begin(), ws.tmp.end(), pressure.begin());
  }
  parallel_for(pool, ry, [&](int y) {
    for (int z = 0; z < rz; ++z) {
      for (int x = 0; x < rx; ++x) {
        const std::size_t i = at(x, y, z);
        ws.mid[i * zu(C)] -= 0.5f * (P(x + 1, y, z) - P(x - 1, y, z));
        ws.mid[i * zu(C) + 1] -= 0.5f * (P(x, y + 1, z) - P(x, y - 1, z));
        ws.mid[i * zu(C) + 2] -= 0.5f * (P(x, y, z + 1) - P(x, y, z - 1));
        for (int c = 0; c < 3; ++c) flow[i * 3 + zu(c)] = ws.mid[i * zu(C) + zu(c)];
      }
    }
  });
  // semi-Lagrangian advection of every channel by the projected velocity, then the state is kept in range
  parallel_for(pool, ry, [&](int y) {
    for (int z = 0; z < rz; ++z) {
      for (int x = 0; x < rx; ++x) {
        const std::size_t i = at(x, y, z);
        const float* s = ws.mid.data() + i * zu(C);
        const float bx = fl(x) - s[0], by = fl(y) - s[1], bz = fl(z) - s[2];
        for (int c = 0; c < C; ++c) {
          float v = trilinear(ws.mid.data(), rx, ry, rz, C, c, bx, by, bz);
          if (c < kPhys) v = std::clamp(v, m.lo[zu(c)], m.hi[zu(c)]);
          out[i * zu(C) + zu(c)] = v;
        }
      }
    }
  });
}

void detail_step(const Model& m, State& s, std::uint64_t seed, std::span<const float> controls, Workspace& ws, Pool* pool) {
  (void)controls;
  const Hyper& h = m.h;
  const DetailSpec& dt = m.detail;
  const int k = h.fine, FX = h.fx(), FY = h.fy(), FZ = h.fz(), rx = h.rx, ry = h.ry, rz = h.rz, C = h.channels();
  const float t = s.time + 0.5f / m.fps, u128 = 128.f / fl(FX);
  float amp = dt.swirl * fl(FX) / 64.f;
  if (dt.swirl_ramp > 0.f) amp *= std::min(1.f, s.since_start / dt.swirl_ramp);
  if (amp > 0.f) {
    Forcing3 sw = m.noise;
    sw.curl_length = dt.swirl_scale;
    sw.curl_rate = dt.swirl_rate;
    ws.swirl.build(sw, seed * 0x9E3779B97F4A7C15ULL + 5, t, pool);
  }
  const auto fi = [&](int x, int y, int z) { return (zu(y) * zu(FZ) + zu(z)) * zu(FX) + zu(x); };
  const float kf = fl(k);
  // the fine velocity: the coarse flow up-sampled (fine cells per frame), plus the swirl
  parallel_for(pool, FY, [&](int y) {
    for (int z = 0; z < FZ; ++z) {
      for (int x = 0; x < FX; ++x) {
        const float cx = (fl(x) + 0.5f) / kf - 0.5f, cy = (fl(y) + 0.5f) / kf - 0.5f, cz = (fl(z) + 0.5f) / kf - 0.5f;
        float* u = ws.ufine.data() + fi(x, y, z) * 3;
        for (int c = 0; c < 3; ++c) u[c] = kf * trilinear(s.flow.data(), rx, ry, rz, 3, c, cx, cy, cz);
        if (amp > 0.f) {
          const auto v = ws.swirl.sample((fl(x) + 0.5f) * u128, (fl(y) + 0.5f) * u128, (fl(z) + 0.5f) * u128);
          for (int c = 0; c < 3; ++c) u[c] += amp * v[zu(c)];
        }
      }
    }
  });
  // MacCormack advection of the fine fields, clamped to the forward sample's corners, zero outside
  for (std::vector<float>* q : {&s.fine_t, &s.fine_d}) {
    std::vector<float>& Q = *q;
    parallel_for(pool, FY, [&](int y) {
      for (int z = 0; z < FZ; ++z) {
        for (int x = 0; x < FX; ++x) {
          const std::size_t i = fi(x, y, z);
          const float* u = ws.ufine.data() + i * 3;
          ws.fa[i] = trilinear_zero(Q.data(), FX, FY, FZ, fl(x) - u[0], fl(y) - u[1], fl(z) - u[2], &ws.lo[i], &ws.hi[i]);
        }
      }
    });
    parallel_for(pool, FY, [&](int y) {
      for (int z = 0; z < FZ; ++z) {
        for (int x = 0; x < FX; ++x) {
          const std::size_t i = fi(x, y, z);
          const float* u = ws.ufine.data() + i * 3;
          ws.fb[i] = trilinear_zero(ws.fa.data(), FX, FY, FZ, fl(x) + u[0], fl(y) + u[1], fl(z) + u[2]);
        }
      }
    });
    parallel_for(pool, FY, [&](int y) {
      for (std::size_t i = zu(y) * zu(FX) * zu(FZ); i < zu(y + 1) * zu(FX) * zu(FZ); ++i) {
        Q[i] = std::max(0.f, std::clamp(ws.fa[i] + 0.5f * (Q[i] - ws.fb[i]), ws.lo[i], ws.hi[i]));
      }
    });
  }
  // the lock: where a block holds more than its coarse cell, scale it down; where less, scale it up (by at most
  // `grow`) and add the rest, broken up by the flicker noise
  const float inv = 1.f / (kf * kf * kf);
  for (int ch = 0; ch < 2; ++ch) {
    std::vector<float>& Q = ch == 0 ? s.fine_t : s.fine_d;
    parallel_for(pool, ry, [&](int cy) {
      for (int cz = 0; cz < rz; ++cz) {
        for (int cx = 0; cx < rx; ++cx) {
          float b = 0.f;
          for (int y = cy * k; y < (cy + 1) * k; ++y) {
            for (int z = cz * k; z < (cz + 1) * k; ++z) {
              for (int x = cx * k; x < (cx + 1) * k; ++x) b += Q[fi(x, y, z)];
            }
          }
          const std::size_t c = (zu(cy) * zu(rz) + zu(cz)) * zu(rx) + zu(cx);
          b *= inv;
          const float target = s.coarse[c * zu(C) + 3 + zu(ch)];
          constexpr float eps = 1e-4f;
          const float r = (target + eps) / (b + eps);
          ws.rr[c] = r <= 1.f ? r : std::min(r, dt.grow);
          ws.aa[c] = std::max(0.f, target - b * ws.rr[c]);
        }
      }
    });
    parallel_for(pool, FY, [&](int y) {
      for (int z = 0; z < FZ; ++z) {
        for (int x = 0; x < FX; ++x) {
          const float cx = (fl(x) + 0.5f) / kf - 0.5f, cy = (fl(y) + 0.5f) / kf - 0.5f, cz = (fl(z) + 0.5f) / kf - 0.5f;
          float add = trilinear(ws.aa.data(), rx, ry, rz, 1, 0, cx, cy, cz);
          if (add > 0.f && dt.contrast > 0.f) {
            const float phi = flicker3(m.noise, seed, (fl(x) + 0.5f) * u128, (fl(y) + 0.5f) * u128, (fl(z) + 0.5f) * u128, t);
            add *= (1.f - dt.contrast) + dt.contrast * dt.kappa * smoothstep01((phi - dt.edge0) / (dt.edge1 - dt.edge0));
          }
          float& q = Q[fi(x, y, z)];
          q = q * trilinear(ws.rr.data(), rx, ry, rz, 1, 0, cx, cy, cz) + add;
        }
      }
    });
  }
}

void step(const Model& m, State& s, std::span<const float> controls, std::uint64_t seed, Workspace& ws, Pool* pool) {
  const Hyper& h = m.h;
  for (int k = 0; k < h.cond(); ++k) ws.cond[zu(k)] = zu(k) < controls.size() ? controls[zu(k)] : 0.5f;
  coarse_noise(m, seed, s.time, ws, ws.noise, pool);
  coarse_step(m, s.coarse, ws.noise, ws.cond, s.pressure, ws.next, s.flow, ws, pool);
  s.coarse.swap(ws.next);
  detail_step(m, s, seed, controls, ws, pool);
  s.time += 1.f / m.fps;
  s.since_start += 1.f / m.fps;
}

void start_from(const Model& m, std::span<const float> coarse_phys, float time, std::span<const float> controls, std::uint64_t seed, State& s, Workspace& ws,
                Pool* pool, bool grow) {
  const Hyper& h = m.h;
  const int C = h.channels(), FX = h.fx(), FY = h.fy(), FZ = h.fz();
  if (s.coarse.size() != zu(h.cells()) * zu(C)) s = make_state(m);
  std::ranges::fill(s.coarse, 0.f);
  for (int i = 0; i < h.cells(); ++i) {
    for (int c = 0; c < kPhys; ++c) s.coarse[zu(i) * zu(C) + zu(c)] = coarse_phys[zu(i) * kPhys + zu(c)];
  }
  std::ranges::fill(s.pressure, 0.f);
  std::ranges::fill(s.flow, 0.f);
  s.time = time;
  s.since_start = m.detail.swirl_ramp;
  const float kf = fl(h.fine);
  parallel_for(pool, FY, [&](int y) {
    for (int z = 0; z < FZ; ++z) {
      for (int x = 0; x < FX; ++x) {
        const float cx = (fl(x) + 0.5f) / kf - 0.5f, cy = (fl(y) + 0.5f) / kf - 0.5f, cz = (fl(z) + 0.5f) / kf - 0.5f;
        const std::size_t i = (zu(y) * zu(FZ) + zu(z)) * zu(FX) + zu(x);
        s.fine_t[i] = std::max(0.f, trilinear(coarse_phys.data(), h.rx, h.ry, h.rz, kPhys, 3, cx, cy, cz));
        s.fine_d[i] = std::max(0.f, trilinear(coarse_phys.data(), h.rx, h.ry, h.rz, kPhys, 4, cx, cy, cz));
      }
    }
  });
  if (grow) {  // grow detail: material flows through and picks up filaments and flicker
    for (int f = 0; f < h.warmup; ++f) step(m, s, controls, seed, ws, pool);
    s.since_start = m.detail.swirl_ramp;
  }
}

void start(const Model& m, int index, std::span<const float> controls, std::uint64_t seed, State& s, Workspace& ws, Pool* pool) {
  const StartPoint& sp = m.starts.at(zu(index));
  start_from(m, sp.coarse, sp.time, controls, seed, s, ws, pool, true);
}

void block_average(const Hyper& h, std::span<const float> fine, std::span<float> coarse, int channel, int channels) {
  const int k = h.fine, FX = h.fx(), FZ = h.fz();
  const float inv = 1.f / static_cast<float>(k * k * k);
  for (int cy = 0; cy < h.ry; ++cy) {
    for (int cz = 0; cz < h.rz; ++cz) {
      for (int cx = 0; cx < h.rx; ++cx) {
        float b = 0.f;
        for (int y = cy * k; y < (cy + 1) * k; ++y) {
          for (int z = cz * k; z < (cz + 1) * k; ++z) {
            for (int x = cx * k; x < (cx + 1) * k; ++x) b += fine[(zu(y) * zu(FZ) + zu(z)) * zu(FX) + zu(x)];
          }
        }
        coarse[((zu(cy) * zu(h.rz) + zu(cz)) * zu(h.rx) + zu(cx)) * zu(channels) + zu(channel)] = b * inv;
      }
    }
  }
}

}  // namespace nfx::rollout3d
