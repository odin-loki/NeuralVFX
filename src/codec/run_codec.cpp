// The G3a run codec (include/neuralfx/codec/run_codec.hpp).
#include <neuralfx/codec/run_codec.hpp>
#include <neuralfx/codec/rcoder.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <ctime>
#include <format>
#include <limits>
#include <sstream>

namespace nfx::codec {

namespace {

using rollout::kPhys;

constexpr std::uint8_t kVersion = 1;
constexpr std::size_t kTrailer = 8;  // hash of the final reconstruction state, then a checksum of the header, every coded
                                     // integer and that hash

std::size_t sz(int v) { return static_cast<std::size_t>(v); }
double thread_seconds() {
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return static_cast<double>(t.tv_sec) + 1e-9 * static_cast<double>(t.tv_nsec);
}
float fl(int v) { return static_cast<float>(v); }

// --- 8-bit step codes: q = 2^((code - 160) / 16); code 0 means "off" -------------------------------------------------
std::uint8_t step_code(float q) {
  if (!(q > 0.f)) return 0;
  const long c = std::lround(16.0 * std::log2(static_cast<double>(q))) + 160;
  return static_cast<std::uint8_t>(std::clamp(c, 1L, 255L));
}
float step_value(std::uint8_t code) { return code == 0 ? 0.f : static_cast<float>(std::exp2((static_cast<double>(code) - 160.0) / 16.0)); }

// --- byte helpers ----------------------------------------------------------------------------------------------------
void put_le(std::vector<std::uint8_t>& o, std::uint64_t v, int bytes) {
  for (int i = 0; i < bytes; ++i) o.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}
void put_varint(std::vector<std::uint8_t>& o, std::uint64_t v) {
  while (v >= 0x80) {
    o.push_back(static_cast<std::uint8_t>(v | 0x80));
    v >>= 7;
  }
  o.push_back(static_cast<std::uint8_t>(v));
}

struct Reader {
  std::span<const std::uint8_t> b;
  std::size_t pos = 0;
  bool ok = true;
  std::uint64_t le(int bytes) {
    std::uint64_t v = 0;
    if (pos + sz(bytes) > b.size()) {
      ok = false;
      return 0;
    }
    for (int i = 0; i < bytes; ++i) v |= static_cast<std::uint64_t>(b[pos + sz(i)]) << (8 * i);
    pos += sz(bytes);
    return v;
  }
  std::uint64_t varint() {
    std::uint64_t v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
      if (pos >= b.size()) {
        ok = false;
        return 0;
      }
      const std::uint8_t c = b[pos++];
      v |= static_cast<std::uint64_t>(c & 0x7f) << shift;
      if (!(c & 0x80)) return v;
    }
    ok = false;
    return 0;
  }
};

constexpr std::uint32_t kFnvBasis = 2166136261u, kFnvPrime = 16777619u;
void fnv(std::uint32_t& h, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    h ^= (v >> (8 * i)) & 0xffu;
    h *= kFnvPrime;
  }
}

// --- the stored settings -----------------------------------------------------------------------------------------------
struct Codes {
  std::uint8_t q = 0, q_vel = 0, q_mat = 0, q_start = 0, q_fs = 0, qf = 0;
  std::uint8_t k = 0, kf = 0, start_fine = 0, fine_res = 0;  // sides stored / 4
  std::uint8_t round = 128;                                  // rounding offset * 256
  std::uint8_t flags = 0;  // 1: start from the nearest stored start; 2: correct flow; 4: fine residuals set coarse heat;
                           // 8: side context
};
constexpr unsigned kFlags = 15;

Codes to_codes(const Settings& s) {
  Codes c;
  c.q = step_code(s.q);
  c.q_vel = s.q_vel > 0.f ? step_code(s.q * s.q_vel) : 0;
  c.q_mat = s.q_mat > 0.f ? step_code(s.q * s.q_mat) : 0;
  c.round = static_cast<std::uint8_t>(std::clamp(std::lround(s.round * 256.f), 0L, 128L));
  c.q_start = step_code(s.q_start > 0.f ? s.q_start : s.q);
  c.q_fs = s.start_fine > 0 ? step_code(s.q_fine_start) : 0;
  c.qf = s.kf > 0 ? step_code(s.qf) : 0;
  c.k = static_cast<std::uint8_t>(std::clamp(s.k, 0, 255));
  c.kf = static_cast<std::uint8_t>(std::clamp(s.kf, 0, 255));
  c.start_fine = static_cast<std::uint8_t>(std::clamp(s.start_fine / 4, 0, 255));
  c.fine_res = static_cast<std::uint8_t>(std::clamp(s.fine_res / 4, 1, 255));
  c.flags = static_cast<std::uint8_t>((s.start_nearest ? 1 : 0) | (s.correct_flow ? 2 : 0) | (s.fine_sets_coarse ? 4 : 0) | (s.side_context ? 8 : 0));
  return c;
}

Settings from_codes(const Codes& c) {
  Settings s;
  s.k = c.k;
  s.q = step_value(c.q);
  s.q_vel = c.q_vel && c.q ? step_value(c.q_vel) / s.q : 0.f;
  s.q_mat = c.q_mat && c.q ? step_value(c.q_mat) / s.q : 0.f;
  s.round = static_cast<float>(c.round) / 256.f;
  s.fine_sets_coarse = (c.flags & 4) != 0;
  s.side_context = (c.flags & 8) != 0;
  s.q_start = step_value(c.q_start);
  s.start_nearest = (c.flags & 1) != 0;
  s.correct_flow = (c.flags & 2) != 0;
  s.start_fine = c.start_fine * 4;
  s.q_fine_start = step_value(c.q_fs);
  s.kf = c.kf;
  s.fine_res = c.fine_res * 4;
  s.qf = step_value(c.qf);
  return s;
}

// --- sampling ----------------------------------------------------------------------------------------------------------
// Bilinear sample of channel ch of an n x n grid with `channels` interleaved channels, edges clamped.
float sample(const float* f, int n, int channels, int ch, float x, float y) {
  x = std::clamp(x, 0.f, fl(n - 1));
  y = std::clamp(y, 0.f, fl(n - 1));
  const int x0 = std::min(static_cast<int>(x), n - 2 < 0 ? 0 : n - 2), y0 = std::min(static_cast<int>(y), n - 2 < 0 ? 0 : n - 2);
  const int x1 = std::min(x0 + 1, n - 1), y1 = std::min(y0 + 1, n - 1);
  const float fx = x - fl(x0), fy = y - fl(y0);
  const auto at = [&](int xx, int yy) { return f[(sz(yy) * sz(n) + sz(xx)) * sz(channels) + sz(ch)]; };
  return (1.f - fy) * ((1.f - fx) * at(x0, y0) + fx * at(x1, y0)) + fy * ((1.f - fx) * at(x0, y1) + fx * at(x1, y1));
}

// Upsample channel ch of an n x n grid to `out` (side m), cell centres aligned.
void upsample(const float* f, int n, int channels, int ch, int m, float* out) {
  const float k = fl(m) / fl(n);
  for (int y = 0; y < m; ++y) {
    for (int x = 0; x < m; ++x) out[sz(y) * sz(m) + sz(x)] = sample(f, n, channels, ch, (fl(x) + 0.5f) / k - 0.5f, (fl(y) + 0.5f) / k - 0.5f);
  }
}

// Box average of an m x m field to n x n (n divides m).
void box(const float* f, int m, int n, float* out) {
  const int k = m / n;
  const float a = 1.f / fl(k * k);
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      float s = 0.f;
      for (int dy = 0; dy < k; ++dy) {
        for (int dx = 0; dx < k; ++dx) s += f[sz(y * k + dy) * sz(m) + sz(x * k + dx)];
      }
      out[sz(y) * sz(n) + sz(x)] = s * a;
    }
  }
}

// sign(v) floor(|v| / step + round), clamped to what the coder takes.
std::int32_t quantise(float v, float step, float round) {
  const double a = std::floor(std::abs(static_cast<double>(v) / static_cast<double>(step)) + static_cast<double>(round));
  const double r = std::min(a, static_cast<double>(ResidualModel::kMaxAbs));
  return static_cast<std::int32_t>(v < 0.f ? -r : r);
}

std::vector<int> corrected_channels(const Settings& s) {
  std::vector<int> c;
  if (s.q_vel > 0.f) c.insert(c.end(), {0, 1});
  if (s.q_mat > 0.f) c.insert(c.end(), {2, 3});
  return c;
}

// --- the reconstruction loop, shared by the encoder and the decoder ------------------------------------------------------
struct Loop {
  const rollout::Model& m;
  Settings s;
  std::array<float, 3> controls;
  std::uint64_t seed;
  rollout::State st;
  std::vector<float> cond, noise, next, rgba;
  std::array<float, kPhys> cstep{};
  std::array<float, 2> fstep{};
  std::vector<int> chans;  // coarse channels corrected per cell (u, v if q_vel > 0; heat, soot if q_mat > 0)

  Loop(const rollout::Model& model, const Settings& settings, const std::array<float, 3>& ctl, std::uint64_t sd, float time0, int size)
      : m(model), s(settings), controls(ctl), seed(sd) {
    const int R = m.h.res, C = m.h.channels();
    st.res = R;
    st.size = size;
    st.time = time0;
    st.since_start = 0.f;
    st.coarse.assign(sz(R) * sz(R) * sz(C), 0.f);
    st.pressure.assign(sz(R) * sz(R), 0.f);
    st.flow.assign(sz(R) * sz(R) * 2, 0.f);
    st.fine_t.assign(sz(size) * sz(size), 0.f);
    st.fine_d.assign(sz(size) * sz(size), 0.f);
    cond.resize(sz(m.h.cond()));
    noise.resize(sz(R) * sz(R) * rollout::kNoise);
    next.resize(st.coarse.size());
    rgba.reserve(sz(m.h.res) * sz(m.h.res) * rollout::kDirs);
    chans = corrected_channels(s);
    for (int c = 0; c < kPhys; ++c) cstep[sz(c)] = s.q * (c < 2 ? s.q_vel : s.q_mat) * m.scale[sz(c)];
    for (int c = 0; c < 2; ++c) fstep[sz(c)] = s.qf * m.render_scale[sz(c)];
  }
  int res() const { return m.h.res; }
  int size() const { return st.size; }

  // The start: a prediction of the coarse start (the nearest stored start or zero), res * res * kPhys.
  std::vector<float> start_prediction() const {
    std::vector<float> p(sz(res()) * sz(res()) * kPhys, 0.f);
    if (s.start_nearest && !m.starts.empty()) {
      const auto& sp = m.starts[sz(nearest_start(m, controls))];
      std::copy(sp.coarse.begin(), sp.coarse.end(), p.begin());
    }
    return p;
  }
  std::array<float, kPhys> start_steps() const {
    std::array<float, kPhys> a{};
    for (int c = 0; c < kPhys; ++c) a[sz(c)] = s.q_start * m.scale[sz(c)];
    return a;
  }
  // Sets the coarse start from its prediction and the decoded integers r (res * res * kPhys).
  void set_coarse_start(const std::vector<float>& pred, std::span<const std::int32_t> r) {
    const int R = res(), C = m.h.channels();
    const auto a = start_steps();
    for (int i = 0; i < R * R; ++i) {
      for (int c = 0; c < kPhys; ++c) {
        float v = pred[sz(i) * kPhys + sz(c)] + static_cast<float>(r[sz(i) * kPhys + sz(c)]) * a[sz(c)];
        if (c >= 2) v = std::max(0.f, v);
        st.coarse[sz(i) * sz(C) + sz(c)] = v;
      }
    }
  }
  // The fine start's prediction at side n: the coarse heat and soot, upsampled. [n][n][2].
  std::vector<float> fine_start_prediction(int n) const {
    const int C = m.h.channels();
    std::vector<float> t(sz(n) * sz(n)), d(sz(n) * sz(n)), p(sz(n) * sz(n) * 2);
    upsample(st.coarse.data(), res(), C, 2, n, t.data());
    upsample(st.coarse.data(), res(), C, 3, n, d.data());
    for (std::size_t i = 0; i < t.size(); ++i) {
      p[i * 2] = t[i];
      p[i * 2 + 1] = d[i];
    }
    return p;
  }
  std::array<float, 2> fine_start_steps() const { return {s.q_fine_start * m.render_scale[0], s.q_fine_start * m.render_scale[1]}; }
  // The fine fields from a decoded fine start at side n ([n][n][2] values), or from the coarse state (n = 0).
  void set_fine_start(int n, const std::vector<float>& pred, std::span<const std::int32_t> r) {
    const int S = size();
    if (n == 0) {
      const auto p = fine_start_prediction(S);
      for (std::size_t i = 0; i < st.fine_t.size(); ++i) {
        st.fine_t[i] = p[i * 2];
        st.fine_d[i] = p[i * 2 + 1];
      }
      return;
    }
    const auto a = fine_start_steps();
    std::vector<float> f(sz(n) * sz(n) * 2);
    for (std::size_t i = 0; i < f.size(); ++i) f[i] = std::max(0.f, pred[i] + static_cast<float>(r[i]) * a[i % 2]);
    upsample(f.data(), n, 2, 0, S, st.fine_t.data());
    upsample(f.data(), n, 2, 1, S, st.fine_d.data());
  }

  // Side contexts: a class of the prediction (0 empty, then log2 steps of the material in channel units).
  static std::uint8_t level(float x) {
    if (!(x > 1e-6f)) return 0;
    const auto u = static_cast<std::uint32_t>(std::min(x * 16.f, 1e6f));
    return static_cast<std::uint8_t>(std::min(15, 1 + static_cast<int>(std::bit_width(u))));
  }
  std::vector<std::uint8_t> coarse_side(std::span<const float> coarse, int stride) const {
    std::vector<std::uint8_t> v;
    if (!s.side_context) return v;
    const int R = res();
    v.resize(sz(R) * sz(R));
    for (int i = 0; i < R * R; ++i) v[sz(i)] = level(coarse[sz(i) * sz(stride) + 2] / m.scale[2] + coarse[sz(i) * sz(stride) + 3] / m.scale[3]);
    return v;
  }
  // From n x n x 2 fine values (heat, soot).
  std::vector<std::uint8_t> fine_side(std::span<const float> f, int n) const {
    std::vector<std::uint8_t> v;
    if (!s.side_context) return v;
    v.resize(sz(n) * sz(n));
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = level(f[i * 2] / m.render_scale[0] + f[i * 2 + 1] / m.render_scale[1]);
    return v;
  }

  void predict() {
    rollout::condition(m, controls, st.time, cond);
    rollout::coarse_noise(m, seed, st.time, noise);
    rollout::coarse_step(m, st.coarse, noise, cond, st.pressure, next, st.flow);
    st.coarse.swap(next);
  }
  // r: res * res * chans.size() integers.
  void apply_coarse(std::span<const std::int32_t> r) {
    const int R = res(), C = m.h.channels(), n = static_cast<int>(chans.size());
    for (int i = 0; i < R * R; ++i) {
      for (int j = 0; j < n; ++j) {
        const int c = chans[sz(j)];
        const float d = static_cast<float>(r[sz(i) * sz(n) + sz(j)]) * cstep[sz(c)];
        float& v = st.coarse[sz(i) * sz(C) + sz(c)];
        v += d;
        if (c >= 2) v = std::max(0.f, v);
        else if (s.correct_flow) st.flow[sz(i) * 2 + sz(c)] += d;
      }
    }
  }
  void detail() { rollout::detail_step(m, st, seed, controls); }
  // The fine fields block-averaged to the residual's side: [n][n][2].
  std::vector<float> fine_blocks(int n) const {
    const int S = size();
    std::vector<float> t(sz(n) * sz(n)), d(sz(n) * sz(n)), p(sz(n) * sz(n) * 2);
    box(st.fine_t.data(), S, n, t.data());
    box(st.fine_d.data(), S, n, d.data());
    for (std::size_t i = 0; i < t.size(); ++i) {
      p[i * 2] = t[i];
      p[i * 2 + 1] = d[i];
    }
    return p;
  }
  void apply_fine(std::span<const std::int32_t> r) {
    const int S = size(), n = s.fine_res, k = S / n;
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) {
        const std::size_t j = (sz(y / k) * sz(n) + sz(x / k)) * 2, i = sz(y) * sz(S) + sz(x);
        st.fine_t[i] = std::max(0.f, st.fine_t[i] + static_cast<float>(r[j]) * fstep[0]);
        st.fine_d[i] = std::max(0.f, st.fine_d[i] + static_cast<float>(r[j + 1]) * fstep[1]);
      }
    }
    if (s.fine_sets_coarse) {  // the coarse heat and soot follow the corrected fine fields (their block means)
      const int R = res(), C = m.h.channels();
      std::vector<float> t(sz(R) * sz(R)), d(sz(R) * sz(R));
      box(st.fine_t.data(), S, R, t.data());
      box(st.fine_d.data(), S, R, d.data());
      for (int i = 0; i < R * R; ++i) {
        st.coarse[sz(i) * sz(C) + 2] = t[sz(i)];
        st.coarse[sz(i) * sz(C) + 3] = d[sz(i)];
      }
    }
  }
  void render(std::span<std::uint8_t> out) { render_u8(m, st, out, rgba); }
  void advance() {
    st.time += 1.f / m.fps;
    st.since_start += 1.f / m.fps;
  }
  std::uint32_t state_hash() const {
    std::uint32_t h = kFnvBasis;
    for (const auto* v : {&st.coarse, &st.fine_t, &st.fine_d, &st.pressure}) {
      for (const float x : *v) fnv(h, std::bit_cast<std::uint32_t>(x));
    }
    return h;
  }
  std::size_t memory_bytes() const {
    return (st.coarse.size() + st.pressure.size() + st.flow.size() + st.fine_t.size() + st.fine_d.size() + cond.size() + noise.size() + next.size() +
            rgba.size()) * sizeof(float);
  }
};

// rollout.cpp's bilinear sample of a strided field, operation for operation (the renderer's coarse inputs).
float bilinear_ch(const float* f, int n, int channels, int c, float x, float y) {
  x = std::clamp(x, 0.f, fl(n - 1));
  y = std::clamp(y, 0.f, fl(n - 1));
  const int x0 = std::min(static_cast<int>(x), n - 2), y0 = std::min(static_cast<int>(y), n - 2);
  const float fx = x - fl(x0), fy = y - fl(y0);
  const auto at = [&](int xx, int yy) { return f[(sz(yy) * sz(n) + sz(xx)) * sz(channels) + sz(c)]; };
  return (1.f - fy) * ((1.f - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1.f - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
}

bool coarse_frame(const Settings& s, int f) { return s.k > 0 && (s.q_vel > 0.f || s.q_mat > 0.f) && f % s.k == 0; }
bool fine_frame(const Settings& s, int f) { return s.kf > 0 && f % s.kf == 0; }

std::uint16_t header_check(std::span<const std::uint8_t> h) {
  std::uint32_t x = kFnvBasis;
  for (const std::uint8_t b : h) {
    x ^= b;
    x *= kFnvPrime;
  }
  return static_cast<std::uint16_t>(x ^ (x >> 16));
}

std::vector<std::uint8_t> header(const rollout::Model& m, const TrueRun& run, const Codes& c, const std::array<std::uint16_t, 3>& ctl) {
  std::vector<std::uint8_t> h = {'G', '3', kVersion};
  put_le(h, model_tag(m), 4);
  h.push_back(c.flags);
  for (const std::uint16_t v : ctl) put_le(h, v, 2);
  put_varint(h, run.seed);
  put_le(h, std::bit_cast<std::uint32_t>(run.time0), 4);
  put_le(h, static_cast<std::uint64_t>(run.frames), 2);
  put_le(h, static_cast<std::uint64_t>(run.size), 2);
  for (const std::uint8_t b : {c.k, c.kf, c.start_fine, c.fine_res, c.q, c.q_vel, c.q_mat, c.q_start, c.q_fs, c.qf, c.round}) h.push_back(b);
  put_le(h, header_check(h), 2);  // a damaged header is refused before the loop runs
  return h;
}

std::array<float, 3> controls_of(const std::array<std::uint16_t, 3>& q) {
  return {static_cast<float>(q[0]) / 65535.f, static_cast<float>(q[1]) / 65535.f, static_cast<float>(q[2]) / 65535.f};
}

}  // namespace

// --- public helpers --------------------------------------------------------------------------------------------------------

void render_u8(const rollout::Model& m, const rollout::State& s, std::span<std::uint8_t> out, std::vector<float>& dirs) {
  using rollout::kDirs;
  using rollout::kDirSteps;
  using rollout::kRenderIn;
  const int R = m.h.res, S = s.size, C = m.h.channels(), H = m.h.render_hidden;
  if (H > 64) throw std::invalid_argument("render_u8: renderer wider than 64");
  static constexpr std::array<std::array<int, 2>, kDirs> dir{{{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}};
  dirs.resize(sz(R) * sz(R) * kDirs);
  for (int cy = 0; cy < R; ++cy) {
    for (int cx = 0; cx < R; ++cx) {
      for (int j = 0; j < kDirs; ++j) {
        float sum = 0.f;
        for (int st = 1; st <= kDirSteps; ++st) {
          const int xx = cx + st * dir[sz(j)][0], yy = cy + st * dir[sz(j)][1];
          if (xx >= 0 && yy >= 0 && xx < R && yy < R) sum += s.coarse[(sz(yy) * sz(R) + sz(xx)) * sz(C) + 3];
        }
        dirs[(sz(cy) * sz(R) + sz(cx)) * kDirs + sz(j)] = sum;
      }
    }
  }
  const auto D = [&](int cx, int cy, int j) { return dirs[(sz(cy) * sz(R) + sz(cx)) * kDirs + sz(j)]; };
  const rollout::RenderLayout L = rollout::render_layout(m.h);
  const float* w = m.render_w.data();
  const float k = fl(S) / fl(R), it = 1.f / m.render_scale[0], id = 1.f / m.render_scale[1];
  std::array<float, kRenderIn> f{};
  std::array<float, 64> h1{}, h2{};
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      std::uint8_t* p = out.data() + (sz(S - 1 - y) * sz(S) + sz(x)) * 4;
      f[0] = s.fine_t[sz(y) * sz(S) + sz(x)] * it;
      f[1] = s.fine_d[sz(y) * sz(S) + sz(x)] * id;
      const float g = rollout::render_gate(f[0], f[1]);
      if (g == 0.f) {  // the reference multiplies the MLP's output by 0 here: every channel rounds to 0
        p[0] = p[1] = p[2] = p[3] = 0;
        continue;
      }
      const float xc = (fl(x) + 0.5f) / k - 0.5f, yc = (fl(y) + 0.5f) / k - 0.5f;
      f[2] = bilinear_ch(s.coarse.data(), R, C, 2, xc, yc) * it;
      f[3] = bilinear_ch(s.coarse.data(), R, C, 3, xc, yc) * id;
      const float cx = std::clamp(xc, 0.f, fl(R - 1)), cy = std::clamp(yc, 0.f, fl(R - 1));
      const int x0 = std::min(static_cast<int>(cx), R - 2), y0 = std::min(static_cast<int>(cy), R - 2);
      const float fx = cx - fl(x0), fy = cy - fl(y0);
      for (int j = 0; j < kDirs; ++j) {
        const float a = (1.f - fx) * D(x0, y0, j) + fx * D(x0 + 1, y0, j);
        const float b = (1.f - fx) * D(x0, y0 + 1, j) + fx * D(x0 + 1, y0 + 1, j);
        f[4 + sz(j)] = ((1.f - fy) * a + fy * b) * id;
      }
      for (int j = 0; j < H; ++j) {
        float acc = w[L.b1 + sz(j)];
        for (int i = 0; i < kRenderIn; ++i) acc += w[L.w1 + sz(j) * kRenderIn + sz(i)] * f[sz(i)];
        h1[sz(j)] = std::max(0.f, acc);
      }
      for (int j = 0; j < H; ++j) {
        float acc = w[L.b2 + sz(j)];
        for (int i = 0; i < H; ++i) acc += w[L.w2 + sz(j) * sz(H) + sz(i)] * h1[sz(i)];
        h2[sz(j)] = std::max(0.f, acc);
      }
      for (int c = 0; c < 4; ++c) {
        float acc = w[L.bo + sz(c)];
        for (int i = 0; i < H; ++i) acc += w[L.wo + sz(c) * sz(H) + sz(i)] * h2[sz(i)];
        p[c] = static_cast<std::uint8_t>(std::clamp(acc * g, 0.f, 1.f) * 255.f + 0.5f);
      }
    }
  }
}

Settings stored_settings(const Settings& s) {
  return from_codes(to_codes(s));
}

std::string describe(const Settings& in) {
  const Settings s = stored_settings(in);
  std::string d = std::format("k{}_q{:.4f}_v{:.2f}_m{:.2f}_s{:.4f}{}_f{}", s.k, s.q, s.q_vel, s.q_mat, s.q_start, s.start_nearest ? "n" : "z", s.start_fine);
  if (s.start_fine > 0) d += std::format("_fs{:.4f}", s.q_fine_start);
  if (s.kf > 0) d += std::format("_kf{}_r{}_qf{:.4f}", s.kf, s.fine_res, s.qf);
  if (s.kf > 0 && !s.fine_sets_coarse) d += "_nosync";
  if (s.side_context) d += "_side";
  if (!s.correct_flow) d += "_noflow";
  if (s.round != 0.5f) d += std::format("_o{:.3f}", s.round);
  return d;
}

std::uint32_t model_tag(const rollout::Model& m) {
  std::ostringstream os;
  if (!rollout::save_model(os, m)) return 0;
  const std::string b = os.str();
  std::uint32_t h = kFnvBasis;
  for (const char c : b) {
    h ^= static_cast<std::uint8_t>(c);
    h *= kFnvPrime;
  }
  return h;
}

int nearest_start(const rollout::Model& m, std::span<const float> ctl) {
  int best = 0;
  float bd = std::numeric_limits<float>::infinity();
  for (std::size_t k = 0; k < m.starts.size(); ++k) {
    float d = 0.f;
    for (std::size_t q = 0; q < 3 && q < m.starts[k].controls.size() && q < ctl.size(); ++q) {
      d += (m.starts[k].controls[q] - ctl[q]) * (m.starts[k].controls[q] - ctl[q]);
    }
    if (d < bd) {
      bd = d;
      best = static_cast<int>(k);
    }
  }
  return best;
}

// --- encoder -------------------------------------------------------------------------------------------------------------

Encoded encode(const rollout::Model& m, const TrueRun& run, const Settings& in, bool render) {
  const int R = m.h.res, S = run.size, F = run.frames;
  const std::size_t cn = sz(R) * sz(R) * kPhys, fn = sz(S) * sz(S);
  if (run.coarse.size() != sz(F + 1) * cn || run.fine_t.size() != sz(F + 1) * fn || run.fine_d.size() != run.fine_t.size()) {
    throw std::invalid_argument("codec: the run's arrays do not match its frames and size");
  }
  if (S % 4 != 0 || S > 1020 || F > 65535) throw std::invalid_argument("codec: unsupported size or length");
  const Codes codes = to_codes(in);
  const Settings s = from_codes(codes);
  if (s.kf > 0 && (s.fine_res <= 0 || S % s.fine_res != 0)) throw std::invalid_argument("codec: the fine residual's side must divide the size");
  if (s.start_fine > 0 && S % s.start_fine != 0) throw std::invalid_argument("codec: the fine start's side must divide the size");
  std::array<std::uint16_t, 3> qc{};
  for (std::size_t i = 0; i < 3; ++i) qc[i] = static_cast<std::uint16_t>(std::lround(std::clamp(run.controls[i], 0.f, 1.f) * 65535.f));

  Encoded e;
  e.stream = header(m, run, codes, qc);
  const std::size_t head = e.stream.size();
  std::uint32_t sum = kFnvBasis;
  for (const std::uint8_t b : e.stream) fnv(sum, b);
  ArithEncoder ac(e.stream);
  ResidualModel rm;
  const auto code = [&](PlaneKind kind, int h, int w, int c, const std::vector<std::int32_t>& r, const std::vector<std::uint8_t>& side) {
    rm.encode(ac, kind, h, w, c, r, side);
    for (const std::int32_t v : r) fnv(sum, static_cast<std::uint32_t>(v));
  };

  Loop L(m, s, controls_of(qc), run.seed, run.time0, S);
  // start: coarse
  {
    const auto pred = L.start_prediction();
    const auto a = L.start_steps();
    std::vector<std::int32_t> r(cn);
    for (std::size_t i = 0; i < cn; ++i) r[i] = quantise(run.coarse[i] - pred[i], a[i % kPhys], s.round);
    code(PlaneKind::coarse_start, R, R, kPhys, r, L.coarse_side(pred, kPhys));
    L.set_coarse_start(pred, r);
  }
  // start: fine
  {
    const int n = s.start_fine;
    std::vector<float> pred;
    std::vector<std::int32_t> r;
    if (n > 0) {
      pred = L.fine_start_prediction(n);
      std::vector<float> t(sz(n) * sz(n)), d(sz(n) * sz(n));
      box(run.fine_t.data(), S, n, t.data());
      box(run.fine_d.data(), S, n, d.data());
      const auto a = L.fine_start_steps();
      r.resize(sz(n) * sz(n) * 2);
      for (std::size_t i = 0; i < t.size(); ++i) {
        r[i * 2] = quantise(t[i] - pred[i * 2], a[0], s.round);
        r[i * 2 + 1] = quantise(d[i] - pred[i * 2 + 1], a[1], s.round);
      }
      code(PlaneKind::fine_start, n, n, 2, r, L.fine_side(pred, n));
    }
    L.set_fine_start(n, pred, r);
  }
  if (render) e.frames.resize(sz(F) * fn * 4);
  const int nc = static_cast<int>(L.chans.size());
  std::vector<std::int32_t> rc(sz(R) * sz(R) * sz(nc)), rf;
  for (int f = 1; f <= F; ++f) {
    L.predict();
    if (coarse_frame(s, f)) {
      const float* truth = run.coarse.data() + sz(f) * cn;
      const int C = m.h.channels();
      for (int i = 0; i < R * R; ++i) {
        for (int j = 0; j < nc; ++j) {
          const int c = L.chans[sz(j)];
          rc[sz(i) * sz(nc) + sz(j)] = quantise(truth[sz(i) * kPhys + sz(c)] - L.st.coarse[sz(i) * sz(C) + sz(c)], L.cstep[sz(c)], s.round);
        }
      }
      code(PlaneKind::coarse, R, R, nc, rc, L.coarse_side(L.st.coarse, m.h.channels()));
      L.apply_coarse(rc);
      ++e.coarse_planes;
    }
    L.detail();
    if (fine_frame(s, f)) {
      const int n = s.fine_res;
      const auto have = L.fine_blocks(n);
      std::vector<float> t(sz(n) * sz(n)), d(sz(n) * sz(n));
      box(run.fine_t.data() + sz(f) * fn, S, n, t.data());
      box(run.fine_d.data() + sz(f) * fn, S, n, d.data());
      rf.resize(sz(n) * sz(n) * 2);
      for (std::size_t i = 0; i < t.size(); ++i) {
        rf[i * 2] = quantise(t[i] - have[i * 2], L.fstep[0], s.round);
        rf[i * 2 + 1] = quantise(d[i] - have[i * 2 + 1], L.fstep[1], s.round);
      }
      code(PlaneKind::fine, n, n, 2, rf, L.fine_side(have, n));
      L.apply_fine(rf);
      ++e.fine_planes;
    }
    if (render) L.render(std::span(e.frames).subspan(sz(f - 1) * fn * 4, fn * 4));
    L.advance();
  }
  ac.flush();
  const std::uint32_t state = L.state_hash();
  fnv(sum, state);
  put_le(e.stream, state, 4);
  put_le(e.stream, sum, 4);
  e.header_bytes = head + kTrailer;
  e.coarse_start_bytes = rm.bits(PlaneKind::coarse_start) / 8.0;
  e.fine_start_bytes = rm.bits(PlaneKind::fine_start) / 8.0;
  e.coarse_bytes = rm.bits(PlaneKind::coarse) / 8.0;
  e.fine_bytes = rm.bits(PlaneKind::fine) / 8.0;
  return e;
}

// --- decoder -------------------------------------------------------------------------------------------------------------

std::expected<Decoded, std::string> decode(const rollout::Model& m, std::span<const std::uint8_t> stream,
                                           const std::function<void(int, std::span<const std::uint8_t>)>& on_frame) {
  Reader rd{stream};
  if (stream.size() < 3 + kTrailer || stream[0] != 'G' || stream[1] != '3') return std::unexpected("g3: not a run stream");
  if (stream[2] != kVersion) return std::unexpected("g3: unsupported version");
  rd.pos = 3;
  if (rd.le(4) != model_tag(m)) return std::unexpected("g3: the stream was made for another model");
  Codes c;
  c.flags = static_cast<std::uint8_t>(rd.le(1));
  std::array<std::uint16_t, 3> qc{};
  for (auto& v : qc) v = static_cast<std::uint16_t>(rd.le(2));
  Decoded out;
  out.seed = rd.varint();
  out.time0 = std::bit_cast<float>(static_cast<std::uint32_t>(rd.le(4)));
  out.frames = static_cast<int>(rd.le(2));
  out.size = static_cast<int>(rd.le(2));
  for (std::uint8_t* b : {&c.k, &c.kf, &c.start_fine, &c.fine_res, &c.q, &c.q_vel, &c.q_mat, &c.q_start, &c.q_fs, &c.qf, &c.round}) {
    *b = static_cast<std::uint8_t>(rd.le(1));
  }
  const std::size_t head_end = rd.pos;
  const auto check = static_cast<std::uint16_t>(rd.le(2));
  if (!rd.ok || rd.pos + kTrailer > stream.size()) return std::unexpected("g3: truncated header");
  if (check != header_check(stream.first(head_end))) return std::unexpected("g3: damaged header");
  const Settings s = from_codes(c);
  const int R = m.h.res, S = out.size, F = out.frames;
  if (S <= 0 || S > 1024 || S % 4 != 0 || !std::isfinite(out.time0) || (c.flags & ~kFlags) != 0 || c.round > 128) return std::unexpected("g3: corrupt header");
  if ((s.kf > 0 && (s.fine_res <= 0 || S % s.fine_res != 0 || c.qf == 0)) || (s.start_fine > 0 && (S % s.start_fine != 0 || c.q_fs == 0)) || c.q == 0 ||
      c.q_start == 0) {
    return std::unexpected("g3: corrupt settings");
  }
  out.settings = s;
  out.controls = controls_of(qc);

  // The integers are decoded as the loop needs them (their contexts may depend on its state); frames are kept until the
  // checksum of every coded integer has been checked, unless they are streamed to on_frame.
  const auto payload = stream.subspan(rd.pos, stream.size() - rd.pos - kTrailer);
  Reader tr{stream.subspan(stream.size() - kTrailer)};
  const auto want_state = static_cast<std::uint32_t>(tr.le(4)), want_sum = static_cast<std::uint32_t>(tr.le(4));
  ArithDecoder ad(payload);
  ResidualModel rm;
  std::uint32_t sum = kFnvBasis;
  for (std::size_t i = 0; i < rd.pos; ++i) fnv(sum, stream[i]);
  bool ok = true;
  std::vector<std::int32_t> r;
  const auto take = [&](PlaneKind kind, int h, int w, int ch, const std::vector<std::uint8_t>& side) {
    const double t0 = thread_seconds();
    r.resize(sz(h) * sz(w) * sz(ch));
    ok = ok && rm.decode(ad, kind, h, w, ch, r, side);
    out.entropy_seconds += thread_seconds() - t0;
    for (const std::int32_t v : r) fnv(sum, static_cast<std::uint32_t>(v));
    return std::span<const std::int32_t>(r);
  };
  const int corrected = static_cast<int>(corrected_channels(s).size());
  Loop L(m, s, out.controls, out.seed, out.time0, S);
  {
    const auto pred = L.start_prediction();
    L.set_coarse_start(pred, take(PlaneKind::coarse_start, R, R, kPhys, L.coarse_side(pred, kPhys)));
  }
  if (s.start_fine > 0) {
    const auto pred = L.fine_start_prediction(s.start_fine);
    L.set_fine_start(s.start_fine, pred, take(PlaneKind::fine_start, s.start_fine, s.start_fine, 2, L.fine_side(pred, s.start_fine)));
  } else {
    L.set_fine_start(0, {}, {});
  }
  const std::size_t fn = sz(S) * sz(S) * 4;
  std::vector<std::uint8_t> frame(fn);
  if (!on_frame) out.rgba.resize(sz(F) * fn);
  for (int f = 1; f <= F && ok; ++f) {
    L.predict();
    if (coarse_frame(s, f)) L.apply_coarse(take(PlaneKind::coarse, R, R, corrected, L.coarse_side(L.st.coarse, m.h.channels())));
    L.detail();
    if (fine_frame(s, f) && ok) {
      const auto have = L.fine_blocks(s.fine_res);
      L.apply_fine(take(PlaneKind::fine, s.fine_res, s.fine_res, 2, L.fine_side(have, s.fine_res)));
    }
    if (on_frame) {
      L.render(frame);
      on_frame(f - 1, frame);
    } else {
      L.render(std::span(out.rgba).subspan(sz(f - 1) * fn, fn));
    }
    L.advance();
  }
  if (on_frame) out.rgba.clear();
  fnv(sum, want_state);
  if (!ok) return std::unexpected("g3: damaged stream (the decoder ran past its end)");
  if (sum != want_sum) return std::unexpected("g3: damaged stream (checksum mismatch)");
  out.verified = L.state_hash() == want_state;
  out.working_bytes = r.capacity() * sizeof(std::int32_t) + rm.memory_bytes() + L.memory_bytes() + fn;
  return out;
}

}  // namespace nfx::codec
