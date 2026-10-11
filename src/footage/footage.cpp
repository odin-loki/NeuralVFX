// From footage to rollout effects (include/neuralfx/footage.hpp, docs/FOOTAGE.md): the inverse network, refinement
// through the effect's learned renderer, block-matched motion, assimilation through the stepper, and effect files with
// estimated start points. Plain float C++; the inverse network's training is threaded like the rollout renderer's.
#include <neuralfx/binio.hpp>
#include <neuralfx/footage.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <thread>

namespace nfx::footage {

namespace {

std::size_t sz(int v) { return static_cast<std::size_t>(v); }
float fl(int v) { return static_cast<float>(v); }

using Vec = std::vector<float>;

}  // namespace

Frame to_float(std::span<const std::uint8_t> rgba) {
  Frame f(rgba.size());
  for (std::size_t i = 0; i < rgba.size(); ++i) f[i] = static_cast<float>(rgba[i]) / 255.f;
  return f;
}

void to_u8(std::span<const float> rgba, std::span<std::uint8_t> out) {
  for (std::size_t i = 0; i < rgba.size(); ++i) out[i] = static_cast<std::uint8_t>(std::clamp(rgba[i], 0.f, 1.f) * 255.f + 0.5f);
}

// --- degradations ---------------------------------------------------------------------------------------------------

void degrade(std::span<std::uint8_t> rgba, int size, const Degradation& d, std::uint64_t seed) {
  const std::size_t n = sz(size) * sz(size);
  Vec f(n * 4);
  for (std::size_t i = 0; i < f.size(); ++i) f[i] = static_cast<float>(rgba[i]) / 255.f;
  if (d.blur > 0.f) {  // separable Gaussian, zero outside (transparent)
    const int r = std::max(1, static_cast<int>(std::ceil(3.f * d.blur)));
    Vec k(sz(2 * r + 1));
    float sum = 0.f;
    for (int i = -r; i <= r; ++i) sum += k[sz(i + r)] = std::exp(-0.5f * fl(i * i) / (d.blur * d.blur));
    for (float& v : k) v /= sum;
    Vec tmp(f.size(), 0.f);
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        for (int c = 0; c < 4; ++c) {
          float s = 0.f;
          for (int i = -r; i <= r; ++i) {
            const int xx = x + i;
            if (xx >= 0 && xx < size) s += k[sz(i + r)] * f[(sz(y) * sz(size) + sz(xx)) * 4 + sz(c)];
          }
          tmp[(sz(y) * sz(size) + sz(x)) * 4 + sz(c)] = s;
        }
      }
    }
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        for (int c = 0; c < 4; ++c) {
          float s = 0.f;
          for (int i = -r; i <= r; ++i) {
            const int yy = y + i;
            if (yy >= 0 && yy < size) s += k[sz(i + r)] * tmp[(sz(yy) * sz(size) + sz(x)) * 4 + sz(c)];
          }
          f[(sz(y) * sz(size) + sz(x)) * 4 + sz(c)] = s;
        }
      }
    }
  }
  if (d.noise > 0.f) {
    std::mt19937_64 rng(seed * 0x9E3779B97F4A7C15ULL + 11);
    std::normal_distribution<float> nd(0.f, d.noise);
    for (float& v : f) v += nd(rng);
  }
  for (std::size_t i = 0; i < n; ++i) {
    float* p = f.data() + i * 4;
    for (int c = 0; c < 4; ++c) p[c] = std::clamp(p[c], 0.f, 1.f);
    if (d.drop_alpha) p[3] = std::max({p[0], p[1], p[2]});
  }
  to_u8(f, rgba);
  if (d.drop_alpha) {  // as nvfx_ingest --alpha luma does on the 8-bit values
    for (std::size_t i = 0; i < n; ++i) rgba[i * 4 + 3] = std::max({rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]});
  }
}

// --- the inverse network -------------------------------------------------------------------------------------------

namespace {

// A two-output MLP with two ReLU hidden layers, weights input-major: [I][H], [H], [H][H], [H], [H][2], [2].
struct InvLayout {
  int I = 0, H = 0;
  std::size_t w1, b1, w2, b2, wo, bo, size;
};
InvLayout mlp_layout(int inputs, int hidden) {
  const std::size_t I = sz(inputs), H = sz(hidden);
  InvLayout L{};
  L.I = inputs;
  L.H = hidden;
  L.w1 = 0;
  L.b1 = L.w1 + I * H;
  L.w2 = L.b1 + H;
  L.b2 = L.w2 + H * H;
  L.wo = L.b2 + H;
  L.bo = L.wo + H * 2;
  L.size = L.bo + 2;
  return L;
}
InvLayout inv_layout(const Inverse& inv) { return mlp_layout(inv.inputs(), inv.spec.hidden); }

void mlp_init(std::vector<float>& w, const InvLayout& L, std::uint64_t seed) {
  w.assign(L.size, 0.f);
  std::mt19937_64 rng(seed);
  const auto fill = [&](std::size_t off, std::size_t n, float sd) {
    std::normal_distribution<float> nd(0.f, sd);
    for (std::size_t i = 0; i < n; ++i) w[off + i] = nd(rng);
  };
  fill(L.w1, sz(L.I) * sz(L.H), std::sqrt(2.f / fl(L.I)));
  fill(L.w2, sz(L.H) * sz(L.H), std::sqrt(2.f / fl(L.H)));
  fill(L.wo, sz(L.H) * 2, std::sqrt(1.f / fl(L.H)));
}

// Forward pass: h1, h2 (hidden activations after ReLU) and the two outputs.
void mlp_forward(const float* w, const InvLayout& L, const float* f, float* h1, float* h2, float* out) {
  const int I = L.I, H = L.H;
  std::copy(w + L.b1, w + L.b1 + H, h1);
  for (int i = 0; i < I; ++i) {
    const float x = f[i];
    if (x == 0.f) continue;
    const float* wi = w + L.w1 + sz(i) * sz(H);
    for (int j = 0; j < H; ++j) h1[j] += wi[j] * x;
  }
  for (int j = 0; j < H; ++j) h1[j] = std::max(0.f, h1[j]);
  std::copy(w + L.b2, w + L.b2 + H, h2);
  for (int i = 0; i < H; ++i) {
    const float x = h1[i];
    if (x == 0.f) continue;
    const float* wi = w + L.w2 + sz(i) * sz(H);
    for (int j = 0; j < H; ++j) h2[j] += wi[j] * x;
  }
  for (int j = 0; j < H; ++j) h2[j] = std::max(0.f, h2[j]);
  out[0] = w[L.bo];
  out[1] = w[L.bo + 1];
  for (int i = 0; i < H; ++i) {
    out[0] += w[L.wo + sz(i) * 2] * h2[i];
    out[1] += w[L.wo + sz(i) * 2 + 1] * h2[i];
  }
}

// Adds to g the weight gradient of one sample with output gradients (ge0, ge1); g1 and g2 are scratch of size H.
void mlp_backward(const float* w, const InvLayout& L, const float* f, const float* h1, const float* h2, float ge0, float ge1, float* g, float* g1, float* g2) {
  const int I = L.I, H = L.H;
  g[L.bo] += ge0;
  g[L.bo + 1] += ge1;
  for (int j = 0; j < H; ++j) {
    g[L.wo + sz(j) * 2] += ge0 * h2[j];
    g[L.wo + sz(j) * 2 + 1] += ge1 * h2[j];
    g2[j] = h2[j] > 0.f ? ge0 * w[L.wo + sz(j) * 2] + ge1 * w[L.wo + sz(j) * 2 + 1] : 0.f;
  }
  for (int j = 0; j < H; ++j) g1[j] = 0.f;
  for (int j = 0; j < H; ++j) g[L.b2 + sz(j)] += g2[j];
  for (int i2 = 0; i2 < H; ++i2) {
    const float a = h1[i2];
    if (a <= 0.f) continue;
    const float* wi = w + L.w2 + sz(i2) * sz(H);
    float* gi = g + L.w2 + sz(i2) * sz(H);
    float acc = 0.f;
    for (int j = 0; j < H; ++j) {
      gi[j] += a * g2[j];
      acc += wi[j] * g2[j];
    }
    g1[i2] = acc;
  }
  for (int j = 0; j < H; ++j) g[L.b1 + sz(j)] += g1[j];
  for (int k = 0; k < I; ++k) {
    const float a = f[k];
    if (a == 0.f) continue;
    float* gi = g + L.w1 + sz(k) * sz(H);
    for (int j = 0; j < H; ++j) gi[j] += a * g1[j];
  }
}

void inv_forward(const Inverse& inv, const InvLayout& L, const float* f, float* h1, float* h2, float* out) { mlp_forward(inv.w.data(), L, f, h1, h2, out); }

float sample_zero(const float* f, int n, int C, int c, float x, float y) {
  const float fx0 = std::floor(x), fy0 = std::floor(y);
  const int x0 = static_cast<int>(fx0), y0 = static_cast<int>(fy0);
  const float ax = x - fx0, ay = y - fy0;
  const auto at = [&](int xx, int yy) { return (xx < 0 || yy < 0 || xx >= n || yy >= n) ? 0.f : f[(sz(yy) * sz(n) + sz(xx)) * sz(C) + sz(c)]; };
  return (1.f - ay) * ((1.f - ax) * at(x0, y0) + ax * at(x0 + 1, y0)) + ay * ((1.f - ax) * at(x0, y0 + 1) + ax * at(x0 + 1, y0 + 1));
}

}  // namespace

std::size_t Inverse::weights() const { return inv_layout(*this).size; }

Inverse init_inverse(const InverseSpec& spec, std::uint64_t seed) {
  Inverse inv;
  inv.spec = spec;
  mlp_init(inv.w, inv_layout(inv), seed);
  return inv;
}

Motion init_motion(int hidden, std::uint64_t seed) {
  Motion mo;
  mo.hidden = hidden;
  mlp_init(mo.w, mlp_layout(Motion::kInputs, hidden), seed);
  return mo;
}

std::expected<void, std::string> save_inverse(std::ostream& o, const Inverse& inv) {
  o.write(kInverseMagic, 8);
  bin::put(o, std::uint32_t{inv.motion.empty() ? 1u : 2u});  // 2: a motion network follows
  for (const int v : {inv.spec.hidden, inv.spec.levels, inv.spec.alpha ? 1 : 0, inv.size}) bin::put(o, static_cast<std::int32_t>(v));
  bin::put_str(o, inv.effect, 32);
  bin::put(o, inv.scale[0]);
  bin::put(o, inv.scale[1]);
  bin::put(o, static_cast<std::uint32_t>(inv.w.size()));
  bin::put_array(o, std::span<const float>(inv.w));
  if (!inv.motion.empty()) {
    bin::put(o, static_cast<std::int32_t>(inv.motion.hidden));
    for (const float v : {inv.motion.in_scale[0], inv.motion.in_scale[1], inv.motion.out_scale}) bin::put(o, v);
    bin::put(o, static_cast<std::uint32_t>(inv.motion.w.size()));
    bin::put_array(o, std::span<const float>(inv.motion.w));
  }
  if (!o) return std::unexpected("inverse: write failed");
  return {};
}

std::expected<void, std::string> save_inverse(const std::filesystem::path& path, const Inverse& inv) {
  std::ofstream o(path, std::ios::binary);
  if (!o) return std::unexpected("inverse: cannot open " + path.string() + " for writing");
  return save_inverse(o, inv);
}

std::expected<Inverse, std::string> load_inverse(std::istream& i) {
  char magic[8];
  i.read(magic, 8);
  if (!i || std::memcmp(magic, kInverseMagic, 8) != 0) return std::unexpected("not an inverse network (NVFXINV1)");
  auto version = bin::get<std::uint32_t>(i);
  if (!version || *version < 1 || *version > 2) return std::unexpected("inverse: unsupported version");
  Inverse inv;
  std::array<int, 4> v{};
  for (int& x : v) {
    auto r = bin::get<std::int32_t>(i);
    if (!r) return std::unexpected(r.error());
    x = *r;
  }
  inv.spec.hidden = v[0];
  inv.spec.levels = v[1];
  inv.spec.alpha = v[2] != 0;
  inv.size = v[3];
  if (inv.spec.hidden < 1 || inv.spec.hidden > 512 || inv.spec.levels < 1 || inv.spec.levels > 8 || inv.size < 8 || inv.size > 4096) {
    return std::unexpected("inverse: bad header");
  }
  auto name = bin::get_str(i, 32);
  auto s0 = bin::get<float>(i);
  auto s1 = bin::get<float>(i);
  auto n = bin::get<std::uint32_t>(i);
  if (!name || !s0 || !s1 || !n) return std::unexpected("truncated file");
  inv.effect = *name;
  inv.scale = {*s0, *s1};
  if (*n != inv_layout(inv).size || !(inv.scale[0] > 0.f) || !(inv.scale[1] > 0.f)) return std::unexpected("inverse: bad weights");
  inv.w.resize(*n);
  if (auto r = bin::get_array(i, std::span<float>(inv.w)); !r) return std::unexpected(r.error());
  if (*version >= 2) {
    auto h = bin::get<std::int32_t>(i);
    auto a0 = bin::get<float>(i);
    auto a1 = bin::get<float>(i);
    auto os = bin::get<float>(i);
    auto mn = bin::get<std::uint32_t>(i);
    if (!h || !a0 || !a1 || !os || !mn) return std::unexpected("truncated file");
    if (*h < 1 || *h > 512 || !(*a0 > 0.f) || !(*a1 > 0.f) || !(*os > 0.f)) return std::unexpected("inverse: bad motion network");
    inv.motion.hidden = *h;
    inv.motion.in_scale = {*a0, *a1};
    inv.motion.out_scale = *os;
    if (*mn != mlp_layout(Motion::kInputs, *h).size) return std::unexpected("inverse: bad motion weights");
    inv.motion.w.resize(*mn);
    if (auto r = bin::get_array(i, std::span<float>(inv.motion.w)); !r) return std::unexpected(r.error());
  }
  return inv;
}

std::expected<Inverse, std::string> load_inverse(const std::filesystem::path& path) {
  std::ifstream i(path, std::ios::binary);
  if (!i) return std::unexpected("inverse: cannot open " + path.string());
  return load_inverse(i);
}

Features::Features(const Inverse& inv, std::span<const float> frame, int size)
    : size_(size), levels_(inv.spec.levels), channels_(inv.channels()) {
  pyramid_.resize(sz(levels_));
  Vec& l0 = pyramid_[0];
  l0.resize(sz(size) * sz(size) * sz(channels_));
  for (int y = 0; y < size; ++y) {
    const float* row = frame.data() + sz(size - 1 - y) * sz(size) * 4;  // field row y is frame row size - 1 - y
    for (int x = 0; x < size; ++x) {
      for (int c = 0; c < channels_; ++c) l0[(sz(y) * sz(size) + sz(x)) * sz(channels_) + sz(c)] = row[sz(x) * 4 + sz(c)];
    }
  }
  int n = size;
  for (int l = 1; l < levels_; ++l) {
    const int m = std::max(1, n / 2);
    const Vec& a = pyramid_[sz(l - 1)];
    Vec& b = pyramid_[sz(l)];
    b.assign(sz(m) * sz(m) * sz(channels_), 0.f);
    for (int y = 0; y < m; ++y) {
      for (int x = 0; x < m; ++x) {
        for (int c = 0; c < channels_; ++c) {
          float s = 0.f;
          for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
              const int xx = std::min(n - 1, 2 * x + dx), yy = std::min(n - 1, 2 * y + dy);
              s += a[(sz(yy) * sz(n) + sz(xx)) * sz(channels_) + sz(c)];
            }
          }
          b[(sz(y) * sz(m) + sz(x)) * sz(channels_) + sz(c)] = 0.25f * s;
        }
      }
    }
    n = m;
  }
}

void Features::at(int x, int y, std::span<float> out) const {
  std::size_t o = 0;
  int n = size_;
  float cell = 1.f;
  for (int l = 0; l < levels_; ++l) {
    const float lx = (fl(x) + 0.5f) / cell - 0.5f, ly = (fl(y) + 0.5f) / cell - 0.5f;
    const float* p = pyramid_[sz(l)].data();
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        for (int c = 0; c < channels_; ++c) out[o++] = sample_zero(p, n, channels_, c, lx + fl(dx), ly + fl(dy));
      }
    }
    n = std::max(1, n / 2);
    cell *= 2.f;
  }
  out[o++] = (fl(x) + 0.5f) / fl(size_) * 2.f - 1.f;
  out[o] = (fl(y) + 0.5f) / fl(size_) * 2.f - 1.f;
}

namespace {

// Field cells with nothing visible within `r` pixels (max over the matched channels below 1.5 / 255).
std::vector<std::uint8_t> empty_mask(std::span<const float> frame, int size, int channels, int r) {
  Vec v(sz(size) * sz(size));
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const float* p = frame.data() + (sz(size - 1 - y) * sz(size) + sz(x)) * 4;
      float m = 0.f;
      for (int c = 0; c < channels; ++c) m = std::max(m, p[c]);
      v[sz(y) * sz(size) + sz(x)] = m;
    }
  }
  Vec h(v.size());
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      float m = 0.f;
      for (int i = std::max(0, x - r); i <= std::min(size - 1, x + r); ++i) m = std::max(m, v[sz(y) * sz(size) + sz(i)]);
      h[sz(y) * sz(size) + sz(x)] = m;
    }
  }
  std::vector<std::uint8_t> e(v.size());
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      float m = 0.f;
      for (int i = std::max(0, y - r); i <= std::min(size - 1, y + r); ++i) m = std::max(m, h[sz(i) * sz(size) + sz(x)]);
      e[sz(y) * sz(size) + sz(x)] = m < 1.5f / 255.f ? 1 : 0;
    }
  }
  return e;
}

}  // namespace

Fields apply_inverse(const Inverse& inv, std::span<const float> frame, int size) {
  Fields out;
  out.size = size;
  out.heat.assign(sz(size) * sz(size), 0.f);
  out.soot.assign(sz(size) * sz(size), 0.f);
  const Features F(inv, frame, size);
  const InvLayout L = inv_layout(inv);
  const std::vector<std::uint8_t> empty = empty_mask(frame, size, inv.channels(), 3);
  Vec f(sz(inv.inputs())), h1(sz(inv.spec.hidden)), h2(sz(inv.spec.hidden));
  float o[2];
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const std::size_t i = sz(y) * sz(size) + sz(x);
      if (empty[i]) continue;
      F.at(x, y, f);
      inv_forward(inv, L, f.data(), h1.data(), h2.data(), o);
      out.heat[i] = std::max(0.f, o[0]) * inv.scale[0];
      out.soot[i] = std::max(0.f, o[1]) * inv.scale[1];
    }
  }
  return out;
}

namespace {

struct InvAdam {
  Vec m, v;
  int t = 0;
  explicit InvAdam(std::size_t n) : m(n, 0.f), v(n, 0.f) {}
  void step(Vec& w, const Vec& g, float lr) {
    ++t;
    constexpr float b1 = 0.9f, b2 = 0.999f;
    const float c1 = 1.f - std::pow(b1, static_cast<float>(t)), c2 = 1.f - std::pow(b2, static_cast<float>(t));
    for (std::size_t j = 0; j < w.size(); ++j) {
      m[j] = b1 * m[j] + (1.f - b1) * g[j];
      v[j] = b2 * v[j] + (1.f - b2) * g[j] * g[j];
      w[j] -= lr * (m[j] / c1) / (std::sqrt(v[j] / c2) + 1e-8f);
    }
  }
};

}  // namespace

double train_inverse(Inverse& inv, std::span<const InverseSample> samples, const InverseTrainOptions& o) {
  if (samples.empty()) throw std::invalid_argument("inverse: no samples");
  inv.size = samples[0].size;
  {  // output units: the 99.5th percentile of present heat and soot
    Vec t, d;
    for (const InverseSample& s : samples) {
      for (std::size_t i = 0; i < s.truth.heat.size(); i += 7) {
        if (s.truth.heat[i] > 1e-4f) t.push_back(s.truth.heat[i]);
        if (s.truth.soot[i] > 1e-5f) d.push_back(s.truth.soot[i]);
      }
    }
    const auto pct = [](Vec& v) {
      if (v.empty()) return 1.f;
      const auto k = static_cast<std::ptrdiff_t>(static_cast<double>(v.size() - 1) * 0.995);
      std::nth_element(v.begin(), v.begin() + k, v.end());
      return std::max(v[static_cast<std::size_t>(k)], 1e-4f);
    };
    inv.scale = {pct(t), pct(d)};
  }
  std::vector<Features> feats;
  feats.reserve(samples.size());
  for (const InverseSample& s : samples) feats.emplace_back(inv, s.frame, s.size);
  const InvLayout L = inv_layout(inv);
  const int I = inv.inputs(), H = inv.spec.hidden;
  const int threads = std::max(1, o.threads);
  InvAdam adam(L.size);
  std::vector<double> tail;
  for (int it = 0; it < o.iterations; ++it) {
    std::vector<Vec> grads(sz(threads), Vec(L.size, 0.f));
    std::vector<double> sse(sz(threads), 0.0);
    const auto work = [&](int t) {
      std::mt19937_64 rng(o.seed * 7919ULL + static_cast<std::uint64_t>(it) * 131ULL + static_cast<std::uint64_t>(t));
      Vec f(sz(I)), h1(sz(H)), h2(sz(H)), g1(sz(H)), g2(sz(H));
      float out[2];
      float* g = grads[sz(t)].data();
      const float* w = inv.w.data();
      for (int b = t; b < o.batch; b += threads) {
        const std::size_t si = rng() % samples.size();
        const InverseSample& s = samples[si];
        const int S = s.size;
        int x = static_cast<int>(rng() % static_cast<std::uint64_t>(S)), y = static_cast<int>(rng() % static_cast<std::uint64_t>(S));
        if (b & 1) {  // half the pixels where something is present or visible
          for (int tries = 0; tries < 8; ++tries) {
            const std::size_t i = sz(y) * sz(S) + sz(x);
            const float* p = s.frame.data() + (sz(S - 1 - y) * sz(S) + sz(x)) * 4;
            if (s.truth.heat[i] > 0.01f * inv.scale[0] || s.truth.soot[i] > 0.01f * inv.scale[1] || std::max({p[0], p[1], p[2], p[3]}) > 2.f / 255.f) break;
            x = static_cast<int>(rng() % static_cast<std::uint64_t>(S));
            y = static_cast<int>(rng() % static_cast<std::uint64_t>(S));
          }
        }
        feats[si].at(x, y, f);
        inv_forward(inv, L, f.data(), h1.data(), h2.data(), out);
        const std::size_t i = sz(y) * sz(S) + sz(x);
        const float e0 = out[0] - s.truth.heat[i] / inv.scale[0], e1 = out[1] - s.truth.soot[i] / inv.scale[1];
        sse[sz(t)] += static_cast<double>(e0 * e0 + e1 * e1);
        const float ge0 = 2.f * e0 / fl(o.batch), ge1 = 2.f * e1 / fl(o.batch);
        mlp_backward(w, L, f.data(), h1.data(), h2.data(), ge0, ge1, g, g1.data(), g2.data());
      }
    };
    if (threads == 1) {
      work(0);
    } else {
      std::vector<std::jthread> pool;
      for (int t = 0; t < threads; ++t) pool.emplace_back(work, t);
    }
    for (int t = 1; t < threads; ++t) {
      for (std::size_t j = 0; j < L.size; ++j) grads[0][j] += grads[sz(t)][j];
    }
    double total = 0;
    for (const double v : sse) total += v;
    const double loss = total / o.batch;
    const float lr = o.lr * (0.05f + 0.95f * 0.5f * (1.f + std::cos(3.14159265f * fl(it) / fl(std::max(1, o.iterations)))));
    adam.step(inv.w, grads[0], lr);
    if (it >= o.iterations - std::max(1, o.iterations / 20)) tail.push_back(loss);
    if (o.progress && (it % 500 == 0 || it == o.iterations - 1)) o.progress(it, loss);
  }
  double mean = 0;
  for (const double v : tail) mean += v / static_cast<double>(tail.size());
  return mean;
}

// --- the motion network --------------------------------------------------------------------------------------------

namespace {
constexpr float kFlowIn = 4.f;  // measured flow enters the motion network in pixels of a 4-pixel cell
}  // namespace

std::vector<float> motion_inputs(const Motion& mo, std::span<const Fields> three, const Flow& into_middle, const Flow& into_last, int res) {
  if (three.size() != 3) throw std::invalid_argument("footage: the motion network takes three frames");
  std::array<std::vector<float>, 3> c;
  for (std::size_t k = 0; k < 3; ++k) c[k] = coarse_fields(three[k], res);
  const auto known = [res](const Flow& f, int x, int y) { return f.res == res && f.known[sz(y) * sz(res) + sz(x)] != 0; };
  std::vector<float> out(sz(res) * sz(res) * Motion::kInputs, 0.f);
  for (int y = 0; y < res; ++y) {
    for (int x = 0; x < res; ++x) {
      float* o = out.data() + (sz(y) * sz(res) + sz(x)) * Motion::kInputs;
      std::size_t q = 0;
      for (std::size_t k = 0; k < 3; ++k) {
        for (int dy = -2; dy <= 2; ++dy) {
          for (int dx = -2; dx <= 2; ++dx) {
            const int xx = x + dx, yy = y + dy;
            const bool in = xx >= 0 && yy >= 0 && xx < res && yy < res;
            o[q++] = in ? c[k][(sz(yy) * sz(res) + sz(xx)) * 2] / mo.in_scale[0] : 0.f;
            o[q++] = in ? c[k][(sz(yy) * sz(res) + sz(xx)) * 2 + 1] / mo.in_scale[1] : 0.f;
          }
        }
      }
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = x + dx, yy = y + dy;
          if (xx >= 0 && yy >= 0 && xx < res && yy < res && known(into_last, xx, yy)) {
            o[q] = kFlowIn * into_last.uv[(sz(yy) * sz(res) + sz(xx)) * 2];
            o[q + 1] = kFlowIn * into_last.uv[(sz(yy) * sz(res) + sz(xx)) * 2 + 1];
            o[q + 2] = 1.f;
          }
          q += 3;
        }
      }
      if (known(into_middle, x, y)) {
        o[q] = kFlowIn * into_middle.uv[(sz(y) * sz(res) + sz(x)) * 2];
        o[q + 1] = kFlowIn * into_middle.uv[(sz(y) * sz(res) + sz(x)) * 2 + 1];
        o[q + 2] = 1.f;
      }
      q += 3;
      o[q++] = (fl(x) + 0.5f) / fl(res) * 2.f - 1.f;
      o[q++] = (fl(y) + 0.5f) / fl(res) * 2.f - 1.f;
    }
  }
  return out;
}

Flow apply_motion(const Motion& mo, std::span<const float> inputs, int res) {
  const InvLayout L = mlp_layout(Motion::kInputs, mo.hidden);
  Flow f;
  f.res = res;
  f.uv.assign(sz(res) * sz(res) * 2, 0.f);
  f.known.assign(sz(res) * sz(res), 1);
  Vec h1(sz(mo.hidden)), h2(sz(mo.hidden));
  float o[2];
  for (int c = 0; c < res * res; ++c) {
    mlp_forward(mo.w.data(), L, inputs.data() + sz(c) * Motion::kInputs, h1.data(), h2.data(), o);
    f.uv[sz(c) * 2] = o[0] * mo.out_scale;
    f.uv[sz(c) * 2 + 1] = o[1] * mo.out_scale;
  }
  return f;
}

double train_motion(Motion& mo, std::span<const MotionSample> samples, const InverseTrainOptions& o) {
  std::vector<std::pair<std::size_t, std::size_t>> rows;  // (sample, cell)
  double ss = 0;
  for (std::size_t s = 0; s < samples.size(); ++s) {
    const std::size_t n = samples[s].target.size() / 2;
    for (std::size_t c = 0; c < n; ++c) {
      rows.emplace_back(s, c);
      ss += static_cast<double>(samples[s].target[c * 2]) * samples[s].target[c * 2] + static_cast<double>(samples[s].target[c * 2 + 1]) * samples[s].target[c * 2 + 1];
    }
  }
  if (rows.empty()) throw std::invalid_argument("motion: no samples");
  mo.out_scale = static_cast<float>(std::max(1e-4, std::sqrt(ss / (2.0 * static_cast<double>(rows.size())))));
  if (mo.w.empty()) mo = init_motion(mo.hidden, o.seed);
  const InvLayout L = mlp_layout(Motion::kInputs, mo.hidden);
  const int H = mo.hidden, threads = std::max(1, o.threads);
  InvAdam adam(L.size);
  std::vector<double> tail;
  for (int it = 0; it < o.iterations; ++it) {
    std::vector<Vec> grads(sz(threads), Vec(L.size, 0.f));
    std::vector<double> sse(sz(threads), 0.0);
    const auto work = [&](int t) {
      std::mt19937_64 rng(o.seed * 104729ULL + static_cast<std::uint64_t>(it) * 131ULL + static_cast<std::uint64_t>(t));
      Vec h1(sz(H)), h2(sz(H)), g1(sz(H)), g2(sz(H));
      float out[2];
      for (int b = t; b < o.batch; b += threads) {
        const auto [si, c] = rows[rng() % rows.size()];
        const float* f = samples[si].inputs.data() + c * Motion::kInputs;
        mlp_forward(mo.w.data(), L, f, h1.data(), h2.data(), out);
        const float e0 = out[0] - samples[si].target[c * 2] / mo.out_scale, e1 = out[1] - samples[si].target[c * 2 + 1] / mo.out_scale;
        sse[sz(t)] += static_cast<double>(e0 * e0 + e1 * e1);
        mlp_backward(mo.w.data(), L, f, h1.data(), h2.data(), 2.f * e0 / fl(o.batch), 2.f * e1 / fl(o.batch), grads[sz(t)].data(), g1.data(), g2.data());
      }
    };
    if (threads == 1) {
      work(0);
    } else {
      std::vector<std::jthread> pool;
      for (int t = 0; t < threads; ++t) pool.emplace_back(work, t);
    }
    for (int t = 1; t < threads; ++t) {
      for (std::size_t j = 0; j < L.size; ++j) grads[0][j] += grads[sz(t)][j];
    }
    double total = 0;
    for (const double v : sse) total += v;
    const double loss = total / o.batch;
    const float lr = o.lr * (0.05f + 0.95f * 0.5f * (1.f + std::cos(3.14159265f * fl(it) / fl(std::max(1, o.iterations)))));
    adam.step(mo.w, grads[0], lr);
    if (it >= o.iterations - std::max(1, o.iterations / 20)) tail.push_back(loss);
    if (o.progress && (it % 500 == 0 || it == o.iterations - 1)) o.progress(it, loss);
  }
  double mean = 0;
  for (const double v : tail) mean += v / static_cast<double>(tail.size());
  return mean;
}

double field_psnr(std::span<const float> estimate, std::span<const float> truth, float scale) {
  double s = 0;
  for (std::size_t i = 0; i < truth.size(); ++i) {
    const double e = (static_cast<double>(estimate[i]) - static_cast<double>(truth[i])) / scale;
    s += e * e;
  }
  const double mse = s / static_cast<double>(std::max<std::size_t>(1, truth.size()));
  return mse <= 1e-10 ? 99.0 : std::min(99.0, -10.0 * std::log10(mse));
}

// --- refinement through the learned renderer -----------------------------------------------------------------------

namespace {

constexpr std::array<std::array<int, 2>, rollout::kDirs> kDirsXY{{{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}};

// What the renderer sees of a pixel, with the bilinear corners of its coarse cell (as rollout::render_features).
struct Interp {
  int x0, y0;
  float fx, fy;
};
Interp interp_at(int R, int k, int x, int y) {
  const float xc = (fl(x) + 0.5f) / fl(k) - 0.5f, yc = (fl(y) + 0.5f) / fl(k) - 0.5f;
  const float cx = std::clamp(xc, 0.f, fl(R - 1)), cy = std::clamp(yc, 0.f, fl(R - 1));
  const int x0 = std::min(static_cast<int>(cx), R - 2), y0 = std::min(static_cast<int>(cy), R - 2);
  return {x0, y0, cx - fl(x0), cy - fl(y0)};
}

struct Coarse {
  Vec heat, soot;                       // R * R
  std::array<Vec, rollout::kDirs> dir;  // directional soot sums per cell
};
Coarse coarse_of(const Fields& f, int R) {
  const int S = f.size, k = S / R;
  Coarse c;
  c.heat.assign(sz(R) * sz(R), 0.f);
  c.soot.assign(sz(R) * sz(R), 0.f);
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      const std::size_t o = sz(y / k) * sz(R) + sz(x / k), i = sz(y) * sz(S) + sz(x);
      c.heat[o] += f.heat[i];
      c.soot[o] += f.soot[i];
    }
  }
  const float a = 1.f / fl(k * k);
  for (std::size_t i = 0; i < c.heat.size(); ++i) {
    c.heat[i] *= a;
    c.soot[i] *= a;
  }
  for (int j = 0; j < rollout::kDirs; ++j) {
    Vec& d = c.dir[sz(j)];
    d.assign(sz(R) * sz(R), 0.f);
    for (int cy = 0; cy < R; ++cy) {
      for (int cx = 0; cx < R; ++cx) {
        float s = 0.f;
        for (int st = 1; st <= rollout::kDirSteps; ++st) {
          const int xx = cx + st * kDirsXY[sz(j)][0], yy = cy + st * kDirsXY[sz(j)][1];
          if (xx >= 0 && yy >= 0 && xx < R && yy < R) s += c.soot[sz(yy) * sz(R) + sz(xx)];
        }
        d[sz(cy) * sz(R) + sz(cx)] = s;
      }
    }
  }
  return c;
}

float lerp4(const Vec& v, int R, const Interp& p) {
  const float* a = v.data() + sz(p.y0) * sz(R) + sz(p.x0);
  return (1.f - p.fy) * ((1.f - p.fx) * a[0] + p.fx * a[1]) + p.fy * ((1.f - p.fx) * a[R] + p.fx * a[R + 1]);
}
void lerp4_back(Vec& g, int R, const Interp& p, float v) {
  float* a = g.data() + sz(p.y0) * sz(R) + sz(p.x0);
  a[0] += (1.f - p.fy) * (1.f - p.fx) * v;
  a[1] += (1.f - p.fy) * p.fx * v;
  a[R] += p.fy * (1.f - p.fx) * v;
  a[R + 1] += p.fy * p.fx * v;
}

void check_sizes(const rollout::Model& m, const Fields& f) {
  if (f.size % m.h.res != 0 || f.size < 2 * m.h.res) throw std::invalid_argument("footage: the frame size must be a multiple of the effect's grid (at least twice it)");
}

}  // namespace

void render_fields(const rollout::Model& m, const Fields& f, std::span<float> rgba) {
  check_sizes(m, f);
  const int R = m.h.res, C = m.h.channels();
  rollout::State s;
  s.res = R;
  s.size = f.size;
  const Coarse c = coarse_of(f, R);
  s.coarse.assign(sz(R) * sz(R) * sz(C), 0.f);
  for (int i = 0; i < R * R; ++i) {
    s.coarse[sz(i) * sz(C) + 2] = c.heat[sz(i)];
    s.coarse[sz(i) * sz(C) + 3] = c.soot[sz(i)];
  }
  s.fine_t = f.heat;
  s.fine_d = f.soot;
  rollout::render(m, s, rgba);
}

double refine_loss(const rollout::Model& m, std::span<const float> frame, const Fields& f, const Fields& start, const RefineOptions& o,
                   Fields* grad) {
  check_sizes(m, f);
  const int R = m.h.res, S = f.size, k = S / R, H = m.h.render_hidden, nch = o.alpha ? 4 : 3;
  const rollout::RenderLayout L = rollout::render_layout(m.h);
  const float* w = m.render_w.data();
  const float it = 1.f / m.render_scale[0], id = 1.f / m.render_scale[1];
  const Coarse c = coarse_of(f, R);
  Vec gch, gcs;
  std::array<Vec, rollout::kDirs> gdir;
  if (grad) {
    grad->size = S;
    grad->heat.assign(sz(S) * sz(S), 0.f);
    grad->soot.assign(sz(S) * sz(S), 0.f);
    gch.assign(sz(R) * sz(R), 0.f);
    gcs.assign(sz(R) * sz(R), 0.f);
    for (Vec& g : gdir) g.assign(sz(R) * sz(R), 0.f);
  }
  const double inv_n = 1.0 / (static_cast<double>(S) * S * nch), inv_p = 1.0 / (static_cast<double>(S) * S);
  double loss = 0;
  std::array<float, rollout::kRenderIn> fe{}, gf{};
  Vec h1(sz(H)), h2(sz(H)), g1(sz(H)), g2(sz(H));
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      const std::size_t i = sz(y) * sz(S) + sz(x);
      const Interp p = interp_at(R, k, x, y);
      fe[0] = f.heat[i] * it;
      fe[1] = f.soot[i] * id;
      fe[2] = lerp4(c.heat, R, p) * it;
      fe[3] = lerp4(c.soot, R, p) * id;
      for (int j = 0; j < rollout::kDirs; ++j) fe[4 + sz(j)] = lerp4(c.dir[sz(j)], R, p) * id;
      for (int j = 0; j < H; ++j) {
        float s = w[L.b1 + sz(j)];
        for (int q = 0; q < rollout::kRenderIn; ++q) s += w[L.w1 + sz(j) * rollout::kRenderIn + sz(q)] * fe[sz(q)];
        h1[sz(j)] = std::max(0.f, s);
      }
      for (int j = 0; j < H; ++j) {
        float s = w[L.b2 + sz(j)];
        for (int q = 0; q < H; ++q) s += w[L.w2 + sz(j) * sz(H) + sz(q)] * h1[sz(q)];
        h2[sz(j)] = std::max(0.f, s);
      }
      const float graw = 50.f * (std::max(0.f, fe[0]) + std::max(0.f, fe[1])), gate = std::min(1.f, graw);
      const float* tgt = frame.data() + (sz(S - 1 - y) * sz(S) + sz(x)) * 4;
      std::array<float, 4> so{}, go{};
      for (int ch = 0; ch < 4; ++ch) {
        float s = w[L.bo + sz(ch)];
        for (int q = 0; q < H; ++q) s += w[L.wo + sz(ch) * sz(H) + sz(q)] * h2[sz(q)];
        so[sz(ch)] = s;
        if (ch >= nch) continue;
        // squared error of the drawn (clamped) value, extended linearly outside [0, 1] so a pixel pushed past the
        // clamp still feels the way back: (c - t)^2 + 2 (c - t) (o - c), c = clamp(o)
        const float ov = s * gate, cl = std::clamp(ov, 0.f, 1.f), e = cl - tgt[ch];
        loss += (static_cast<double>(e) * e + 2.0 * e * (ov - cl)) * inv_n;
        go[sz(ch)] = 2.f * e * static_cast<float>(inv_n);
      }
      const float pt = (f.heat[i] - start.heat[i]) * it, pd = (f.soot[i] - start.soot[i]) * id;
      loss += static_cast<double>(o.prior) * (pt * pt + pd * pd) * inv_p;
      if (!grad) continue;
      grad->heat[i] += 2.f * o.prior * pt * it * static_cast<float>(inv_p);
      grad->soot[i] += 2.f * o.prior * pd * id * static_cast<float>(inv_p);
      float ggate = 0.f;
      std::fill(g2.begin(), g2.end(), 0.f);
      for (int ch = 0; ch < nch; ++ch) {
        ggate += go[sz(ch)] * so[sz(ch)];
        const float gs = go[sz(ch)] * gate;
        for (int q = 0; q < H; ++q) g2[sz(q)] += w[L.wo + sz(ch) * sz(H) + sz(q)] * gs;
      }
      std::fill(g1.begin(), g1.end(), 0.f);
      for (int j = 0; j < H; ++j) {
        if (h2[sz(j)] <= 0.f) continue;
        for (int q = 0; q < H; ++q) g1[sz(q)] += w[L.w2 + sz(j) * sz(H) + sz(q)] * g2[sz(j)];
      }
      gf.fill(0.f);
      for (int j = 0; j < H; ++j) {
        if (h1[sz(j)] <= 0.f) continue;
        for (int q = 0; q < rollout::kRenderIn; ++q) gf[sz(q)] += w[L.w1 + sz(j) * rollout::kRenderIn + sz(q)] * g1[sz(j)];
      }
      if (graw < 1.f) {
        if (fe[0] > 0.f) gf[0] += 50.f * ggate;
        if (fe[1] > 0.f) gf[1] += 50.f * ggate;
      }
      grad->heat[i] += gf[0] * it;
      grad->soot[i] += gf[1] * id;
      lerp4_back(gch, R, p, gf[2] * it);
      lerp4_back(gcs, R, p, gf[3] * id);
      for (int j = 0; j < rollout::kDirs; ++j) lerp4_back(gdir[sz(j)], R, p, gf[4 + sz(j)] * id);
    }
  }
  if (grad) {
    for (int j = 0; j < rollout::kDirs; ++j) {  // directional sums back to the coarse soot they summed
      for (int cy = 0; cy < R; ++cy) {
        for (int cx = 0; cx < R; ++cx) {
          const float g = gdir[sz(j)][sz(cy) * sz(R) + sz(cx)];
          if (g == 0.f) continue;
          for (int st = 1; st <= rollout::kDirSteps; ++st) {
            const int xx = cx + st * kDirsXY[sz(j)][0], yy = cy + st * kDirsXY[sz(j)][1];
            if (xx >= 0 && yy >= 0 && xx < R && yy < R) gcs[sz(yy) * sz(R) + sz(xx)] += g;
          }
        }
      }
    }
    const float a = 1.f / fl(k * k);
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) {
        const std::size_t i = sz(y) * sz(S) + sz(x), b = sz(y / k) * sz(R) + sz(x / k);
        grad->heat[i] += gch[b] * a;
        grad->soot[i] += gcs[b] * a;
      }
    }
  }
  return loss;
}

double refine_fields(const rollout::Model& m, std::span<const float> frame, Fields& f, const RefineOptions& o) {
  const Fields start = f;
  const std::size_t n = f.heat.size();
  // Adam in the renderer's input units (heat / scale, soot / scale), kept at or above zero.
  Vec mo(2 * n, 0.f), ve(2 * n, 0.f);
  Fields g;
  constexpr float b1 = 0.9f, b2 = 0.999f;
  const std::array<float, 2> sc = m.render_scale;
  for (int t = 1; t <= o.iterations; ++t) {
    refine_loss(m, frame, f, start, o, &g);
    const float c1 = 1.f - std::pow(b1, static_cast<float>(t)), c2 = 1.f - std::pow(b2, static_cast<float>(t));
    for (int ch = 0; ch < 2; ++ch) {
      Vec& v = ch == 0 ? f.heat : f.soot;
      const Vec& gv = ch == 0 ? g.heat : g.soot;
      for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = sz(ch) * n + i;
        const float gu = gv[i] * sc[sz(ch)];  // gradient in normalised units
        mo[j] = b1 * mo[j] + (1.f - b1) * gu;
        ve[j] = b2 * ve[j] + (1.f - b2) * gu * gu;
        const float u = v[i] / sc[sz(ch)] - o.lr * (mo[j] / c1) / (std::sqrt(ve[j] / c2) + 1e-12f);
        v[i] = std::max(0.f, u) * sc[sz(ch)];
      }
    }
  }
  RefineOptions plain = o;
  plain.prior = 0.f;
  return refine_loss(m, frame, f, start, plain, nullptr);
}

// --- coarse state -----------------------------------------------------------------------------------------------

std::vector<float> coarse_fields(const Fields& f, int res) {
  const int S = f.size, k = S / res;
  if (k * res != S) throw std::invalid_argument("footage: the grid must divide the frame size");
  std::vector<float> out(sz(res) * sz(res) * 2, 0.f);
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      const std::size_t o = (sz(y / k) * sz(res) + sz(x / k)) * 2, i = sz(y) * sz(S) + sz(x);
      out[o] += f.heat[i];
      out[o + 1] += f.soot[i];
    }
  }
  const float a = 1.f / fl(k * k);
  for (float& v : out) v *= a;
  return out;
}

Flow block_flow(const Fields& a, const Fields& b, int res, std::array<float, 2> scale, const FlowOptions& o) {
  const int S = a.size, k = S / res, r = o.radius, W = o.window;
  if (k * res != S || b.size != S) throw std::invalid_argument("footage: flow needs fields of one size, a multiple of the grid");
  // The earlier frame padded by the search radius plus the window (zero outside), heat and soot interleaved.
  const int P = r + W + 1, PS = S + 2 * P;
  Vec A(sz(PS) * sz(PS) * 2, 0.f);
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      const std::size_t i = sz(y) * sz(S) + sz(x), q = (sz(y + P) * sz(PS) + sz(x + P)) * 2;
      A[q] = a.heat[i] / scale[0];
      A[q + 1] = a.soot[i] / scale[1];
    }
  }
  Flow out;
  out.res = res;
  out.uv.assign(sz(res) * sz(res) * 2, 0.f);
  out.known.assign(sz(res) * sz(res), 0);
  const int side = 2 * r + 1;
  Vec cost(sz(side) * sz(side)), win;
  for (int cy = 0; cy < res; ++cy) {
    for (int cx = 0; cx < res; ++cx) {
      const float px = fl(cx * k) + 0.5f * fl(k - 1), py = fl(cy * k) + 0.5f * fl(k - 1);
      const int x0 = static_cast<int>(std::ceil(px - fl(W))), x1 = static_cast<int>(std::floor(px + fl(W)));
      const int y0 = static_cast<int>(std::ceil(py - fl(W))), y1 = static_cast<int>(std::floor(py + fl(W)));
      const int ww = x1 - x0 + 1, wh = y1 - y0 + 1;
      win.assign(sz(ww) * sz(wh) * 2, 0.f);  // the later frame's window (zero outside the frame)
      double mass = 0;
      for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
          if (x < 0 || y < 0 || x >= S || y >= S) continue;
          const std::size_t i = sz(y) * sz(S) + sz(x), q = (sz(y - y0) * sz(ww) + sz(x - x0)) * 2;
          win[q] = b.heat[i] / scale[0];
          win[q + 1] = b.soot[i] / scale[1];
          mass += static_cast<double>(win[q] + win[q + 1]);
        }
      }
      if (mass / (ww * wh) < static_cast<double>(o.min_mass)) continue;
      // b(x) = a(x - d): the material at x now was at x - d one frame ago
      for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
          float s = 0.f;
          for (int y = 0; y < wh; ++y) {
            const float* ar = A.data() + (sz(y0 + y - dy + P) * sz(PS) + sz(x0 - dx + P)) * 2;
            const float* br = win.data() + sz(y) * sz(ww) * 2;
            for (int q = 0; q < 2 * ww; ++q) {
              const float e = br[q] - ar[q];
              s += e * e;
            }
          }
          cost[sz(dy + r) * sz(side) + sz(dx + r)] = s;
        }
      }
      int best = (side * side) / 2;  // ties keep the smallest displacement found first from the centre
      for (int i = 0; i < side * side; ++i) {
        if (cost[sz(i)] < cost[sz(best)]) best = i;
      }
      const int bx = best % side - r, by = best / side - r;
      const auto C = [&](int dx, int dy) { return cost[sz(dy + r) * sz(side) + sz(dx + r)]; };
      const auto sub = [](float cm, float c0, float cp) {
        const float den = cm - 2.f * c0 + cp;
        return den > 1e-12f ? std::clamp(0.5f * (cm - cp) / den, -0.5f, 0.5f) : 0.f;
      };
      const float sx = std::abs(bx) < r ? sub(C(bx - 1, by), C(bx, by), C(bx + 1, by)) : 0.f;
      const float sy = std::abs(by) < r ? sub(C(bx, by - 1), C(bx, by), C(bx, by + 1)) : 0.f;
      const std::size_t c = sz(cy) * sz(res) + sz(cx);
      out.uv[c * 2] = (fl(bx) + sx) / fl(k);
      out.uv[c * 2 + 1] = (fl(by) + sy) / fl(k);
      out.known[c] = 1;
    }
  }
  return out;
}

Flow mean_flow(std::span<const Flow> flows) {
  Flow out;
  for (const Flow& f : flows) {
    if (f.res == 0) continue;
    if (out.res == 0) {
      out.res = f.res;
      out.uv.assign(f.uv.size(), 0.f);
      out.known.assign(f.known.size(), 0);
    }
  }
  if (out.res == 0) return out;
  std::vector<int> n(out.known.size(), 0);
  for (const Flow& f : flows) {
    if (f.res != out.res) continue;
    for (std::size_t c = 0; c < f.known.size(); ++c) {
      if (!f.known[c]) continue;
      out.uv[c * 2] += f.uv[c * 2];
      out.uv[c * 2 + 1] += f.uv[c * 2 + 1];
      ++n[c];
    }
  }
  for (std::size_t c = 0; c < n.size(); ++c) {
    if (n[c] == 0) continue;
    out.uv[c * 2] /= static_cast<float>(n[c]);
    out.uv[c * 2 + 1] /= static_cast<float>(n[c]);
    out.known[c] = 1;
  }
  return out;
}

void finish_flow(Flow& f, const FlowOptions& o) {
  const int R = f.res;
  if (R == 0) return;
  const auto idx = [R](int x, int y) { return sz(y) * sz(R) + sz(x); };
  if (o.fill && std::ranges::any_of(f.known, [](std::uint8_t v) { return v != 0; })) {
    Vec next(f.uv);
    for (int it = 0; it < 4 * R; ++it) {
      for (int y = 0; y < R; ++y) {
        for (int x = 0; x < R; ++x) {
          if (f.known[idx(x, y)]) continue;
          for (int c = 0; c < 2; ++c) {
            float s = 0.f;
            int n = 0;
            for (const auto& [dx, dy] : {std::pair{1, 0}, std::pair{-1, 0}, std::pair{0, 1}, std::pair{0, -1}}) {
              const int xx = x + dx, yy = y + dy;
              if (xx < 0 || yy < 0 || xx >= R || yy >= R) continue;
              s += f.uv[idx(xx, yy) * 2 + sz(c)];
              ++n;
            }
            next[idx(x, y) * 2 + sz(c)] = n ? s / fl(n) : 0.f;
          }
        }
      }
      f.uv = next;
    }
  }
  for (int pass = 0; pass < o.smooth; ++pass) {
    Vec next(f.uv.size());
    for (int y = 0; y < R; ++y) {
      for (int x = 0; x < R; ++x) {
        for (int c = 0; c < 2; ++c) {
          float s = 0.f;
          int n = 0;
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
              const int xx = x + dx, yy = y + dy;
              if (xx < 0 || yy < 0 || xx >= R || yy >= R) continue;
              s += f.uv[idx(xx, yy) * 2 + sz(c)];
              ++n;
            }
          }
          next[idx(x, y) * 2 + sz(c)] = s / fl(n);
        }
      }
    }
    f.uv = next;
  }
}

std::vector<float> assimilate(const rollout::Model& m, std::span<const Fields> fields, std::span<const Flow> flows, const AssimOptions& o) {
  if (fields.empty()) throw std::invalid_argument("footage: no frames to assimilate");
  const rollout::Hyper& h = m.h;
  const int R = h.res, N = R * R, C = h.channels(), K = static_cast<int>(fields.size());
  std::vector<float> controls = o.controls;
  controls.resize(sz(h.n_controls), 0.5f);
  const float dt = 1.f / m.fps;
  const auto frame_time = [&](int i) { return o.time - fl(K - 1 - i) * dt; };
  const auto measured = [&](int i) -> const Flow* {
    if (i < 0 || sz(i) >= flows.size() || flows[sz(i)].res != R) return nullptr;
    return &flows[sz(i)];
  };
  Vec S(sz(N) * sz(C), 0.f), next(S.size()), pressure(sz(N), 0.f), flowbuf(sz(N) * 2), noise(sz(N) * rollout::kNoise), cond(sz(h.cond()));
  const auto set_fields = [&](int i) {
    const std::vector<float> c = coarse_fields(fields[sz(i)], R);
    for (int j = 0; j < N; ++j) {
      S[sz(j) * sz(C) + 2] = std::clamp(c[sz(j) * 2], m.lo[2], m.hi[2]);
      S[sz(j) * sz(C) + 3] = std::clamp(c[sz(j) * 2 + 1], m.lo[3], m.hi[3]);
    }
  };
  if (!o.stepper) {  // no assimilation: the start frame's fields and the velocity measured into it
    set_fields(K - 1);
    const Flow* f = o.use_flow ? measured(K - 1) : nullptr;
    for (int j = 0; j < N; ++j) {
      S[sz(j) * sz(C)] = f ? std::clamp(f->uv[sz(j) * 2], m.lo[0], m.hi[0]) : 0.f;
      S[sz(j) * sz(C) + 1] = f ? std::clamp(f->uv[sz(j) * 2 + 1], m.lo[1], m.hi[1]) : 0.f;
    }
    std::vector<float> out(sz(N) * rollout::kPhys);
    for (int j = 0; j < N; ++j) {
      for (int c = 0; c < rollout::kPhys; ++c) out[sz(j) * rollout::kPhys + sz(c)] = S[sz(j) * sz(C) + sz(c)];
    }
    return out;
  }
  set_fields(0);
  if (o.use_flow) {
    const Flow* f0 = measured(0);
    if (!f0) f0 = measured(1);
    if (f0) {
      for (int j = 0; j < N; ++j) {
        S[sz(j) * sz(C)] = std::clamp(f0->uv[sz(j) * 2], m.lo[0], m.hi[0]);
        S[sz(j) * sz(C) + 1] = std::clamp(f0->uv[sz(j) * 2 + 1], m.lo[1], m.hi[1]);
      }
    }
  }
  for (int i = 1; i < K; ++i) {
    const float t = frame_time(i - 1);
    rollout::condition(m, controls, t, cond);
    rollout::coarse_noise(m, o.seed, t, noise);
    rollout::coarse_step(m, S, noise, cond, pressure, next, flowbuf);
    S.swap(next);
    set_fields(i);
    const Flow* f = measured(i);
    if (f && o.nudge_velocity > 0.f) {
      for (int j = 0; j < N; ++j) {
        if (!f->known[sz(j)]) continue;
        for (int c = 0; c < 2; ++c) {
          float& v = S[sz(j) * sz(C) + sz(c)];
          v = std::clamp((1.f - o.nudge_velocity) * v + o.nudge_velocity * f->uv[sz(j) * 2 + sz(c)], m.lo[sz(c)], m.hi[sz(c)]);
        }
      }
    }
  }
  std::vector<float> out(sz(N) * rollout::kPhys);
  for (int j = 0; j < N; ++j) {
    for (int c = 0; c < rollout::kPhys; ++c) out[sz(j) * rollout::kPhys + sz(c)] = S[sz(j) * sz(C) + sz(c)];
  }
  return out;
}

// --- the whole estimate -----------------------------------------------------------------------------------------------

rollout::StartPoint make_start(std::vector<float> coarse, const Fields& fine, std::span<const float> controls, std::uint64_t seed, float time) {
  rollout::StartPoint sp;
  sp.controls.assign(controls.begin(), controls.end());
  sp.seed = seed;
  sp.time = time;
  sp.coarse = std::move(coarse);
  sp.fine_t = fine.heat;
  sp.fine_d = fine.soot;
  return sp;
}

Estimate estimate_start(const rollout::Model& m, const Inverse& inv, std::span<const Frame> frames, int size, const EstimateOptions& o) {
  if (frames.empty()) throw std::invalid_argument("footage: no frames");
  const int K = std::clamp(o.context, 1, static_cast<int>(frames.size()));
  const std::span<const Frame> use = frames.subspan(frames.size() - sz(K));
  Estimate e;
  for (const Frame& f : use) e.fields.push_back(apply_inverse(inv, f, size));
  std::vector<Flow> raw(sz(K));
  for (int i = 1; i < K; ++i) raw[sz(i)] = block_flow(e.fields[sz(i - 1)], e.fields[sz(i)], m.h.res, inv.scale, o.flow_opt);
  e.flows.resize(sz(K));
  for (int i = 1; i < K; ++i) {
    const int first = std::max(1, i - std::max(1, o.flow_pairs) + 1);
    e.flows[sz(i)] = mean_flow(std::span<const Flow>(raw).subspan(sz(first), sz(i - first + 1)));
    finish_flow(e.flows[sz(i)], o.flow_opt);
    if (o.motion && !inv.motion.empty() && i >= 2) {  // the motion network where three frames are at hand
      const std::vector<float> in = motion_inputs(inv.motion, std::span<const Fields>(e.fields).subspan(sz(i - 2), 3), raw[sz(i - 1)], raw[sz(i)], m.h.res);
      e.flows[sz(i)] = apply_motion(inv.motion, in, m.h.res);
    }
  }
  if (o.refine) {
    RefineOptions ro = o.refine_opt;
    ro.alpha = inv.spec.alpha;
    refine_fields(m, use.back(), e.fields.back(), ro);
  }
  std::vector<float> controls = o.assim.controls;
  controls.resize(sz(m.h.n_controls), 0.5f);
  e.start = make_start(assimilate(m, e.fields, e.flows, o.assim), e.fields.back(), controls, o.assim.seed, o.assim.time);
  return e;
}

rollout::Model with_starts(const rollout::Model& m, std::span<const rollout::StartPoint> starts, int fine_size, bool keep_stored) {
  rollout::Model out = m;
  if (!keep_stored) out.starts.clear();
  const bool stored_fine = keep_stored && std::ranges::any_of(out.starts, [](const rollout::StartPoint& s) { return !s.fine_t.empty(); });
  const int F = stored_fine ? m.h.start_fine : fine_size;
  if (F < 2) throw std::invalid_argument("footage: fine fields need a size of at least 2");
  for (const rollout::StartPoint& s : starts) {
    rollout::StartPoint sp = s;
    sp.controls.resize(sz(m.h.n_controls), 0.5f);
    if (sp.coarse.size() != sz(m.h.res) * sz(m.h.res) * rollout::kPhys) throw std::invalid_argument("footage: start point of another grid");
    if (!s.fine_t.empty()) {
      const int S = static_cast<int>(std::lround(std::sqrt(static_cast<double>(s.fine_t.size()))));
      if (S * S != static_cast<int>(s.fine_t.size())) throw std::invalid_argument("footage: fine fields must be square");
      if (S != F) {
        sp.fine_t.assign(sz(F) * sz(F), 0.f);
        sp.fine_d.assign(sz(F) * sz(F), 0.f);
        if (S % F == 0) {  // box average
          const int k = S / F;
          const float a = 1.f / fl(k * k);
          for (int y = 0; y < S; ++y) {
            for (int x = 0; x < S; ++x) {
              const std::size_t o = sz(y / k) * sz(F) + sz(x / k), i = sz(y) * sz(S) + sz(x);
              sp.fine_t[o] += s.fine_t[i] * a;
              sp.fine_d[o] += s.fine_d[i] * a;
            }
          }
        } else {  // bilinear at the new cells' centres
          const float k = fl(S) / fl(F);
          for (int y = 0; y < F; ++y) {
            for (int x = 0; x < F; ++x) {
              const float xs = std::clamp((fl(x) + 0.5f) * k - 0.5f, 0.f, fl(S - 1)), ys = std::clamp((fl(y) + 0.5f) * k - 0.5f, 0.f, fl(S - 1));
              const int x0 = std::min(static_cast<int>(xs), S - 2), y0 = std::min(static_cast<int>(ys), S - 2);
              const float ax = xs - fl(x0), ay = ys - fl(y0);
              const auto bl = [&](const Vec& v) {
                const float* a = v.data() + sz(y0) * sz(S) + sz(x0);
                return (1.f - ay) * ((1.f - ax) * a[0] + ax * a[1]) + ay * ((1.f - ax) * a[S] + ax * a[S + 1]);
              };
              sp.fine_t[sz(y) * sz(F) + sz(x)] = bl(s.fine_t);
              sp.fine_d[sz(y) * sz(F) + sz(x)] = bl(s.fine_d);
            }
          }
        }
      }
    }
    out.starts.push_back(std::move(sp));
  }
  if (std::ranges::any_of(out.starts, [](const rollout::StartPoint& s) { return !s.fine_t.empty(); })) out.h.start_fine = F;
  if (out.starts.empty()) throw std::invalid_argument("footage: an effect needs at least one start point");
  return out;
}

}  // namespace nfx::footage
