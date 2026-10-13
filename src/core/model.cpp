#include <neuralfx/binio.hpp>
#include <neuralfx/model.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <format>
#include <fstream>
#include <limits>
#include <random>
#include <ranges>
#include <stdexcept>
#include <stdfloat>
#include <tuple>

namespace nfx {

std::string Hyper::describe() const {
  if (arch == Arch::multi) {
    std::string lv;
    for (const Level& l : levels) lv += std::format("{}G{} T{} C{}", lv.empty() ? "" : ", ", l.grid, l.grid_t, l.channels);
    return std::format("multi [{}] H{} L{} pe{}/{} D{}+z{} K{}", lv, hidden, layers, pe_xy, pe_t, n_controls, n_latent, bases);
  }
  const std::string cond = std::format("D{}+z{} K{} T{}", n_controls, n_latent, bases, grid_t);
  if (arch == Arch::grid) return std::format("grid G{} C{} H{} L{} {}", grid, channels, hidden, layers, cond);
  return std::format("conv l{} {}-{}-{}-4 {}", latent, c0, c1, c2, cond);
}

int Hyper::feature_channels() const {
  if (arch == Arch::grid) return channels;
  if (arch == Arch::conv) return c0;
  int n = 0;
  for (const Level& l : levels) n += l.channels;
  return n;
}

std::vector<Volume> volumes(const Hyper& h) {
  std::vector<Volume> v;
  const auto add = [&](int side, int slices, int channels) {
    Volume x{side, slices, channels, 0, 0, 0, 0};
    if (!v.empty()) {
      const Volume& p = v.back();
      const std::size_t planes = static_cast<std::size_t>(h.bases) * static_cast<std::size_t>(p.slices) * static_cast<std::size_t>(p.channels);
      x.value0 = p.value0 + planes * p.plane_values();
      x.plane0 = p.plane0 + planes;
      x.slice0 = p.slice0 + static_cast<std::size_t>(p.slices);
      x.mask0 = p.mask0 + static_cast<std::size_t>(p.slices) * p.plane_values();
    }
    v.push_back(x);
  };
  if (h.arch == Arch::multi) {
    for (const Level& l : h.levels) add(l.grid, l.grid_t, l.channels);
  } else {
    add(h.feature_side(), h.grid_t, h.feature_channels());
  }
  return v;
}

namespace {
std::size_t multi_storage(const Model& m);  // storage_bytes() of the multi family (file version 4), below
}  // namespace

std::size_t Model::feature_count() const { return features.size(); }

std::size_t Model::param_count() const {
  std::size_t n = features.size() + basis.params() + z_mean.size() + z_std.size();
  for (const auto& l : layers) n += l.params();
  for (const auto& f : films) n += f.params();
  for (const auto& z : z_train) n += z.size();
  return n;
}

std::size_t Model::storage_bytes() const {
  if (h.arch == Arch::multi) return multi_storage(*this);
  const std::size_t slices = static_cast<std::size_t>(h.bases) * h.grid_t * h.feature_channels();
  const std::size_t side2 = static_cast<std::size_t>(h.feature_side()) * h.feature_side();
  std::size_t feat = feature_bits < 16 ? slices * (packed_plane_bytes(side2, feature_bits) + 4) : features.size() * 2;
  if (per_plane()) {  // flags, a width byte per plane, the mask; per plane its range, fill (masked) and codes
    const bool mk = masked() || !raw_mask.empty();
    feat = 4 + plane_bits.size() + (mk ? static_cast<std::size_t>(h.grid_t) * packed_plane_bytes(side2, 1) : 0);
    const std::size_t C = static_cast<std::size_t>(h.feature_channels()), T = static_cast<std::size_t>(h.grid_t);
    for (std::size_t k = 0; k < plane_bits.size(); ++k) {
      feat += 4 + (mk ? 2 : 0) + packed_plane_bytes(plane_points(static_cast<int>((k / C) % T)), plane_bits[k]);
    }
  } else if (vq_groups() > 0) {
    const std::size_t idx_planes = static_cast<std::size_t>(h.bases) * h.grid_t * static_cast<std::size_t>(vq_groups());
    feat = 2 * vq_codebook.size() + idx_planes * packed_plane_bytes(side2, vq_bits);
  }
  return feat + (param_count() - features.size()) * 2;
}

std::size_t Model::plane_points(int t) const {
  const std::size_t side2 = static_cast<std::size_t>(h.feature_side()) * h.feature_side();
  if (!feature_mask.empty()) {
    const auto first = feature_mask.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(t) * side2);
    return static_cast<std::size_t>(std::count_if(first, first + static_cast<std::ptrdiff_t>(side2), [](std::uint8_t v) { return v != 0; }));
  }
  if (!raw_mask.empty()) {  // the runtime keeps the mask bit-packed only
    const std::size_t mb = packed_plane_bytes(side2, 1);
    std::size_t n = 0;
    for (std::size_t j = 0; j < mb; ++j) n += static_cast<std::size_t>(std::popcount(raw_mask[static_cast<std::size_t>(t) * mb + j]));
    return n;
  }
  return side2;
}

double Model::macs_per_pixel(int out) const {
  const double px = static_cast<double>(out) * out;
  if (h.arch == Arch::multi) {  // per frame: blend and time lerp of every level; per pixel: bilinear rows, the MLP
    double slice = 0;
    for (const Level& l : h.levels) slice += 2.0 * h.bases * l.channels * l.grid * l.grid / px;
    double mlp = 0;
    for (const auto& l : layers) mlp += static_cast<double>(l.in) * l.out;
    return slice + 2.0 * h.feature_channels() + mlp;
  }
  const double side = h.feature_side();
  const double slice = 2.0 * h.bases * h.feature_channels() * side * side / px;  // per-frame blend and time lerp
  if (h.arch == Arch::grid) {
    double mlp = 0;
    for (const auto& l : layers) mlp += static_cast<double>(l.in) * l.out;
    return slice + 2.0 * h.channels + mlp;  // 2C: the row-wise bilinear expansion
  }
  const int ch[4] = {h.c0, h.c1, h.c2, 4};
  double m = 0;
  int s = h.latent;
  for (int k = 0; k < 3; ++k) {
    s *= 2;
    m += 9.0 * ch[k] * ch[k + 1] * s * s;
  }
  return slice + m / px;
}

namespace {

void validate(const Hyper& h) {
  const auto bad = [](std::string_view what) { throw std::invalid_argument(std::format("model: {}", what)); };
  if (h.frames < 2 || h.bases < 1 || (h.grid_t < 2 && h.arch != Arch::multi) || h.n_controls < 0 || h.n_latent < 0 || h.dims() > 64) bad("bad shape");
  if (h.arch == Arch::grid) {
    if (h.grid < 2 || h.channels < 1 || h.hidden < 1 || h.layers < 1 || h.size < 4) bad("bad grid shape");
  } else if (h.arch == Arch::multi) {
    if (h.levels.empty() || h.levels.size() > 16 || h.hidden < 1 || h.layers < 1 || h.size < 4 || h.pe_xy < 0 || h.pe_xy > 12 ||
        h.pe_t < 0 || h.pe_t > 12) {
      bad("bad multi-level shape");
    }
    for (const Level& l : h.levels) {
      if (l.grid < 2 || l.grid > 1024 || l.grid_t < 1 || l.grid_t > 4096 || l.channels < 1 || l.channels > 256) {
        bad("bad level (side 2 to 1024, 1 to 4096 slices, 1 to 256 channels)");
      }
    }
  } else if (h.arch == Arch::conv) {
    if (h.latent < 2 || h.c0 < 1 || h.c1 < 1 || h.c2 < 1 || h.size != 8 * h.latent) bad("conv: size must be 8 * latent");
  } else {
    bad("unknown architecture");
  }
}

void he_init(Dense& d, std::mt19937_64& rng, float gain) {
  const float a = gain * std::sqrt(6.f / static_cast<float>(std::max(1, d.in)));
  std::uniform_real_distribution<float> u(-a, a);
  for (float& w : d.w) w = u(rng);
}

// Time slices bracketing t, and the weight of the second.
struct TimeLerp {
  int i0, i1;
  float w;
};
TimeLerp time_lerp(const Hyper& h, float t) {
  if (h.loop) {
    const float u = (t - std::floor(t)) * static_cast<float>(h.grid_t);
    const int i0 = std::min(static_cast<int>(u), h.grid_t - 1);
    return {i0, (i0 + 1) % h.grid_t, u - static_cast<float>(i0)};
  }
  const float u = std::clamp(t, 0.f, 1.f) * static_cast<float>(h.grid_t - 1);
  const int i0 = std::min(static_cast<int>(u), h.grid_t - 2);
  return {i0, i0 + 1, u - static_cast<float>(i0)};
}

std::vector<float> apply(const Dense& d, std::span<const float> x) {
  std::vector<float> y(d.b);
  for (int o = 0; o < d.out; ++o) {
    for (int i = 0; i < d.in; ++i) y[static_cast<std::size_t>(o)] += d.w[static_cast<std::size_t>(o) * d.in + i] * x[static_cast<std::size_t>(i)];
  }
  return y;
}

// The blended, time-sliced feature planes [C][side][side] for (t, c).
std::vector<float> slice(const Model& m, float t, std::span<const float> c) {
  const Hyper& h = m.h;
  const auto w = apply(m.basis, c);
  const std::size_t plane = static_cast<std::size_t>(h.feature_channels()) * h.feature_side() * h.feature_side();
  const auto [i0, i1, ft] = time_lerp(h, t);
  std::vector<float> s(plane, 0.f);
  for (int k = 0; k < h.bases; ++k) {
    const float* a = m.features.data() + (static_cast<std::size_t>(k) * h.grid_t + i0) * plane;
    const float* b = m.features.data() + (static_cast<std::size_t>(k) * h.grid_t + i1) * plane;
    for (std::size_t i = 0; i < plane; ++i) s[i] += w[static_cast<std::size_t>(k)] * (a[i] + ft * (b[i] - a[i]));
  }
  return s;
}

// The multi family's reference forward pass: every level blended and sliced at its own time slices, sampled bilinearly
// at the pixel (the grid family's coordinates at the level's side), the levels' features concatenated with the Fourier
// features, then the grid family's MLP.
void multi_render(const Model& m, float t, std::span<const float> c, int size, std::span<float> rgba) {
  const Hyper& h = m.h;
  const auto vols = volumes(h);
  const auto w = apply(m.basis, c);
  const auto film = apply(m.films[0], c);
  std::vector<std::vector<float>> sl;
  for (const Volume& v : vols) {
    int i0, i1;
    float ft;
    slice_lerp(v.slices, h.loop, t, i0, i1, ft);
    const std::size_t n = static_cast<std::size_t>(v.channels) * v.plane_values();
    std::vector<float> s(n, 0.f);
    for (int k = 0; k < h.bases; ++k) {
      const float* a = m.features.data() + v.value0 + (static_cast<std::size_t>(k) * v.slices + static_cast<std::size_t>(i0)) * n;
      const float* b = m.features.data() + v.value0 + (static_cast<std::size_t>(k) * v.slices + static_cast<std::size_t>(i1)) * n;
      for (std::size_t i = 0; i < n; ++i) s[i] += w[static_cast<std::size_t>(k)] * (a[i] + ft * (b[i] - a[i]));
    }
    sl.push_back(std::move(s));
  }
  std::vector<float> tf(static_cast<std::size_t>(2 * h.pe_t));
  time_features(h.pe_t, h.loop, t, tf.data());
  const int H = h.hidden;
  std::vector<float> f(static_cast<std::size_t>(h.mlp_in()));
  for (int py = 0; py < size; ++py) {
    for (int px = 0; px < size; ++px) {
      std::size_t at = 0;
      for (std::size_t l = 0; l < vols.size(); ++l) {
        const int G = vols[l].side;
        const float gx = std::clamp((static_cast<float>(px) + 0.5f) / static_cast<float>(size) * static_cast<float>(G) - 0.5f, 0.f, static_cast<float>(G - 1));
        const float gy = std::clamp((static_cast<float>(py) + 0.5f) / static_cast<float>(size) * static_cast<float>(G) - 0.5f, 0.f, static_cast<float>(G - 1));
        const int x0 = std::min(static_cast<int>(gx), G - 2), y0 = std::min(static_cast<int>(gy), G - 2);
        const float fx = gx - static_cast<float>(x0), fy = gy - static_cast<float>(y0);
        for (int ch = 0; ch < vols[l].channels; ++ch) {
          const float* p = sl[l].data() + static_cast<std::size_t>(ch) * G * G;
          const auto v = [&](int x, int y) { return p[static_cast<std::size_t>(y) * G + x]; };
          const float a = v(x0, y0) + fx * (v(x0 + 1, y0) - v(x0, y0));
          const float b = v(x0, y0 + 1) + fx * (v(x0 + 1, y0 + 1) - v(x0, y0 + 1));
          f[at++] = a + fy * (b - a);
        }
      }
      position_features(h.pe_xy, (static_cast<float>(px) + 0.5f) / static_cast<float>(size), (static_cast<float>(py) + 0.5f) / static_cast<float>(size),
                        f.data() + at);
      at += static_cast<std::size_t>(4 * h.pe_xy);
      std::ranges::copy(tf, f.begin() + static_cast<std::ptrdiff_t>(at));
      auto x = apply(m.layers[0], f);
      for (int o = 0; o < H; ++o) {
        const auto i = static_cast<std::size_t>(o);
        x[i] = std::max(0.f, (1.f + film[i]) * x[i] + film[static_cast<std::size_t>(H) + i]);
      }
      for (std::size_t l = 1; l < m.layers.size(); ++l) {
        x = apply(m.layers[l], x);
        if (l + 1 < m.layers.size()) std::ranges::for_each(x, [](float& v) { v = std::max(0.f, v); });
      }
      std::ranges::copy(x, rgba.begin() + static_cast<std::ptrdiff_t>((static_cast<std::size_t>(py) * size + px) * 4));
    }
  }
}

}  // namespace

Model init_model(const Hyper& h, std::uint64_t seed) {
  validate(h);
  Model m;
  m.h = h;
  std::mt19937_64 rng(seed);
  const int D = h.dims();
  if (h.arch == Arch::multi) {
    const Volume last = volumes(h).back();
    m.features.resize(last.value0 + static_cast<std::size_t>(h.bases) * last.slices * last.channels * last.plane_values());
  } else {
    const std::size_t plane = static_cast<std::size_t>(h.feature_channels()) * h.feature_side() * h.feature_side();
    m.features.resize(static_cast<std::size_t>(h.bases) * h.grid_t * plane);
  }
  std::uniform_real_distribution<float> fu(-0.1f, 0.1f);
  for (float& v : m.features) v = fu(rng);
  m.basis = Dense(D, h.bases);
  std::uniform_real_distribution<float> bu(-0.5f, 0.5f);
  for (float& w : m.basis.w) w = bu(rng) / std::sqrt(static_cast<float>(std::max(1, D)));
  std::ranges::fill(m.basis.b, 1.f);
  if (h.arch == Arch::grid || h.arch == Arch::multi) {
    int in = h.arch == Arch::grid ? h.channels : h.mlp_in();
    for (int l = 0; l < h.layers; ++l) {
      m.layers.emplace_back(in, h.hidden);
      he_init(m.layers.back(), rng, 1.f);
      in = h.hidden;
    }
    m.layers.emplace_back(in, 4);
    he_init(m.layers.back(), rng, 0.1f);
    m.films.emplace_back(D, 2 * h.hidden);
  } else {
    const int ch[4] = {h.c0, h.c1, h.c2, 4};
    for (int k = 0; k < 3; ++k) {
      m.layers.emplace_back(9 * ch[k], ch[k + 1]);
      he_init(m.layers.back(), rng, k == 2 ? 0.1f : 1.f);
    }
    m.films.emplace_back(D, 2 * h.c1);
    m.films.emplace_back(D, 2 * h.c2);
  }
  m.z_mean.assign(static_cast<std::size_t>(h.n_latent), 0.f);
  m.z_std.assign(static_cast<std::size_t>(h.n_latent), 0.f);
  return m;
}

float frame_time(const Hyper& h, int frame, int frames) {
  if (h.loop) return static_cast<float>(frame) / static_cast<float>(frames);
  return frames > 1 ? static_cast<float>(frame) / static_cast<float>(frames - 1) : 0.f;
}

std::vector<float> condition(const Model& m, std::span<const float> controls, std::span<const float> z) {
  std::vector<float> c(static_cast<std::size_t>(m.h.dims()), 0.f);
  for (std::size_t i = 0; i < std::min<std::size_t>(controls.size(), static_cast<std::size_t>(m.h.n_controls)); ++i) c[i] = controls[i];
  for (std::size_t i = 0; i < std::min<std::size_t>(z.size(), static_cast<std::size_t>(m.h.n_latent)); ++i) {
    c[static_cast<std::size_t>(m.h.n_controls) + i] = z[i];
  }
  return c;
}

void reference_render(const Model& m, float t, std::span<const float> c, int size, std::span<float> rgba) {
  const Hyper& h = m.h;
  if (rgba.size() < static_cast<std::size_t>(size) * size * 4) throw std::invalid_argument("reference_render: buffer too small");
  if (h.arch == Arch::multi) {
    multi_render(m, t, c, size, rgba);
    return;
  }
  const auto s = slice(m, t, c);
  if (h.arch == Arch::grid) {
    const auto film = apply(m.films[0], c);
    const int G = h.grid, C = h.channels, H = h.hidden;
    std::vector<float> f(static_cast<std::size_t>(C));
    for (int py = 0; py < size; ++py) {
      for (int px = 0; px < size; ++px) {
        const float gx = std::clamp((static_cast<float>(px) + 0.5f) / static_cast<float>(size) * static_cast<float>(G) - 0.5f, 0.f, static_cast<float>(G - 1));
        const float gy = std::clamp((static_cast<float>(py) + 0.5f) / static_cast<float>(size) * static_cast<float>(G) - 0.5f, 0.f, static_cast<float>(G - 1));
        const int x0 = std::min(static_cast<int>(gx), G - 2), y0 = std::min(static_cast<int>(gy), G - 2);
        const float fx = gx - static_cast<float>(x0), fy = gy - static_cast<float>(y0);
        for (int ch = 0; ch < C; ++ch) {
          const float* p = s.data() + static_cast<std::size_t>(ch) * G * G;
          const auto at = [&](int x, int y) { return p[static_cast<std::size_t>(y) * G + x]; };
          const float a = at(x0, y0) + fx * (at(x0 + 1, y0) - at(x0, y0));
          const float b = at(x0, y0 + 1) + fx * (at(x0 + 1, y0 + 1) - at(x0, y0 + 1));
          f[static_cast<std::size_t>(ch)] = a + fy * (b - a);
        }
        auto x = apply(m.layers[0], f);
        for (int o = 0; o < H; ++o) {
          const auto i = static_cast<std::size_t>(o);
          x[i] = std::max(0.f, (1.f + film[i]) * x[i] + film[static_cast<std::size_t>(H) + i]);
        }
        for (std::size_t l = 1; l < m.layers.size(); ++l) {
          x = apply(m.layers[l], x);
          if (l + 1 < m.layers.size()) std::ranges::for_each(x, [](float& v) { v = std::max(0.f, v); });
        }
        std::ranges::copy(x, rgba.begin() + static_cast<std::ptrdiff_t>((static_cast<std::size_t>(py) * size + px) * 4));
      }
    }
    return;
  }
  if (size != h.size) throw std::invalid_argument("reference_render: the conv family renders at its native size only");
  const int ch[4] = {h.c0, h.c1, h.c2, 4};
  std::vector<float> cur = s;
  int side = h.latent;
  for (int k = 0; k < 3; ++k) {
    const int n = 2 * side;
    const Dense& L = m.layers[static_cast<std::size_t>(k)];
    std::vector<float> film;
    if (k < 2) film = apply(m.films[static_cast<std::size_t>(k)], c);
    std::vector<float> next(static_cast<std::size_t>(ch[k + 1]) * n * n);
    for (int o = 0; o < ch[k + 1]; ++o) {
      for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
          float acc = L.b[static_cast<std::size_t>(o)];
          for (int i = 0; i < ch[k]; ++i) {
            for (int ky = 0; ky < 3; ++ky) {
              for (int kx = 0; kx < 3; ++kx) {
                const int yy = y + ky - 1, xx = x + kx - 1;
                if (yy < 0 || xx < 0 || yy >= n || xx >= n) continue;  // zero padding
                const float v = cur[(static_cast<std::size_t>(i) * side + yy / 2) * side + xx / 2];  // nearest upsample
                acc += L.w[((static_cast<std::size_t>(o) * ch[k] + i) * 3 + ky) * 3 + kx] * v;
              }
            }
          }
          if (k < 2) {
            const auto j = static_cast<std::size_t>(o);
            acc = std::max(0.f, (1.f + film[j]) * acc + film[static_cast<std::size_t>(ch[k + 1]) + j]);
          }
          next[(static_cast<std::size_t>(o) * n + y) * n + x] = acc;
        }
      }
    }
    cur = std::move(next);
    side = n;
  }
  const std::size_t px = static_cast<std::size_t>(size) * size;
  for (std::size_t i = 0; i < px; ++i) {
    for (std::size_t c4 = 0; c4 < 4; ++c4) rgba[i * 4 + c4] = cur[c4 * px + i];
  }
}

void to_rgba8(std::span<const float> rgba, std::span<std::uint8_t> out) {
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<std::uint8_t>(std::clamp(rgba[i], 0.f, 1.f) * 255.f + 0.5f);
}

// --- storage ------------------------------------------------------------------------------------------------------

namespace {

constexpr std::string_view kMagic = "NVFXMDL1";
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kVersionVq = 2;  // vector-quantised features (vq_bits, vq_dim after feature_bits)
constexpr std::uint32_t kVersionMixed = 3;  // bits per feature plane (a byte per plane before the planes)
constexpr std::uint32_t kVersionMulti = 4;  // the multi family (study F4)

float round_f16(float v) { return static_cast<float>(static_cast<std::float16_t>(v)); }

// N-bit features (2 to 8): each [C][side][side] channel plane of each slice gets its own fp16 min and max.
struct Plane8 {
  float lo, hi;
};
Plane8 minmax_range(std::span<const float> p) {
  const auto [mn, mx] = std::ranges::minmax(p);
  return {round_f16(mn), round_f16(std::max(mx, mn + 1e-6f))};
}
long qmax_of(int bits) { return (1L << bits) - 1; }
std::uint8_t q8(float v, Plane8 r, int bits = 8) {
  if (bits == 0) return 0;  // a plane of one value
  const long qm = qmax_of(bits);
  return static_cast<std::uint8_t>(std::clamp(std::lround((v - r.lo) / (r.hi - r.lo) * static_cast<float>(qm)), 0L, qm));
}
float dq8(unsigned q, Plane8 r, int bits = 8) {
  if (bits == 0) return r.lo;
  return r.lo + static_cast<float>(q) / static_cast<float>(qmax_of(bits)) * (r.hi - r.lo);
}

Plane8 plane_range(std::span<const float> p, int bits, bool trim) {
  const auto [lo, hi] = feature_plane_range(p, bits, trim);
  return {lo, hi};
}

// One plane's codes into `out` (packed_plane_bytes bytes: one per code at 8 bits, bit-packed below).
void put_plane_codes(std::span<const float> plane, Plane8 r, int bits, std::uint8_t* out) {
  if (bits == 8) {
    for (std::size_t j = 0; j < plane.size(); ++j) out[j] = q8(plane[j], r);
    return;
  }
  if (bits == 0) return;  // no codes
  std::fill_n(out, packed_plane_bytes(plane.size(), bits), std::uint8_t{0});
  for (std::size_t j = 0; j < plane.size(); ++j) put_packed_code(out, j, bits, q8(plane[j], r, bits));
}

std::size_t plane_size(const Hyper& h) { return static_cast<std::size_t>(h.feature_side()) * h.feature_side(); }

void put_f16(std::ostream& o, std::span<const float> v) {
  for (const float x : v) bin::put(o, std::bit_cast<std::uint16_t>(static_cast<std::float16_t>(x)));
}
std::expected<void, std::string> get_f16(std::istream& i, std::span<float> v) {
  for (float& x : v) {
    const auto r = bin::get<std::uint16_t>(i);
    if (!r) return std::unexpected(r.error());
    x = static_cast<float>(std::bit_cast<std::float16_t>(*r));
  }
  return {};
}

void put_dense(std::ostream& o, const Dense& d) {
  bin::put(o, static_cast<std::uint32_t>(d.in));
  bin::put(o, static_cast<std::uint32_t>(d.out));
  put_f16(o, d.w);
  put_f16(o, d.b);
}
std::expected<Dense, std::string> get_dense(std::istream& i) {
  const auto in = bin::get<std::uint32_t>(i), out = bin::get<std::uint32_t>(i);
  if (!in || !out || *in > 100000 || *out > 100000) return std::unexpected("nvfx: bad layer header");
  Dense d(static_cast<int>(*in), static_cast<int>(*out));
  if (auto r = get_f16(i, d.w); !r) return std::unexpected(r.error());
  if (auto r = get_f16(i, d.b); !r) return std::unexpected(r.error());
  return d;
}

}  // namespace

std::pair<float, float> feature_plane_range(std::span<const float> p, int bits, bool trim) {
  if (bits == 0) {
    double s = 0;
    for (const float v : p) s += v;
    const float mean = round_f16(p.empty() ? 0.f : static_cast<float>(s / static_cast<double>(p.size())));
    return {mean, mean};
  }
  const Plane8 full = minmax_range(p);
  if (!trim || p.size() < 8) return {full.lo, full.hi};
  std::vector<float> v(p.begin(), p.end());
  std::ranges::sort(v);
  const std::size_t n = v.size();
  Plane8 best = full;
  double best_err = std::numeric_limits<double>::infinity();
  for (const double t : {0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08}) {
    const auto k = static_cast<std::size_t>(t * static_cast<double>(n - 1));
    Plane8 r = t == 0.0 ? full : Plane8{round_f16(v[k]), round_f16(v[n - 1 - k])};
    if (!(r.hi > r.lo)) continue;
    double err = 0;
    for (const float x : v) {
      const double d = static_cast<double>(x) - static_cast<double>(dq8(q8(x, r, bits), r, bits));
      err += d * d;
    }
    if (err < best_err) {
      best_err = err;
      best = r;
    }
  }
  return {best.lo, best.hi};
}

namespace {

// Per-plane storage of one plane: the range of its stored values, the fill of the others, and the codes of the stored
// values in raster order. `active` empty: every value is stored. No stored values: the range is (0, 0).
struct PlaneCodes {
  float lo = 0, hi = 0, fill = 0;
  std::vector<unsigned> codes;
};
PlaneCodes plane_codes(std::span<const float> plane, int bits, bool trim, std::span<const std::uint8_t> active) {
  PlaneCodes pc;
  std::vector<float> on;
  double fs = 0;
  std::size_t fn = 0;
  if (active.empty()) {
    on.assign(plane.begin(), plane.end());
  } else {
    for (std::size_t j = 0; j < plane.size(); ++j) {
      if (active[j]) {
        on.push_back(plane[j]);
      } else {
        fs += plane[j];
        ++fn;
      }
    }
  }
  pc.fill = round_f16(fn > 0 ? static_cast<float>(fs / static_cast<double>(fn)) : 0.f);
  if (!on.empty()) std::tie(pc.lo, pc.hi) = feature_plane_range(on, bits, trim);
  pc.codes.reserve(on.size());
  for (const float v : on) pc.codes.push_back(q8(v, {pc.lo, pc.hi}, bits));
  return pc;
}

// Plane k's mask (its time slice's), or empty.
std::span<const std::uint8_t> plane_mask(const Model& m, std::size_t k) {
  if (m.feature_mask.empty()) return {};
  const std::size_t S2 = plane_size(m.h), C = static_cast<std::size_t>(m.h.feature_channels()), T = static_cast<std::size_t>(m.h.grid_t);
  return std::span(m.feature_mask).subspan(((k / C) % T) * S2, S2);
}

// Why a model's per-plane storage is invalid, or empty.
std::string per_plane_error(const Model& m) {
  const std::size_t S2 = plane_size(m.h);
  if (m.vq_bits > 0) return "per-plane widths and vector quantisation do not combine";
  if (m.plane_bits.size() * S2 != m.features.size()) return "one width per feature plane";
  if (std::ranges::any_of(m.plane_bits, [](std::uint8_t b) { return b > 8; })) return "widths 0 to 8";
  if (!m.feature_mask.empty()) {
    if (m.h.arch != Arch::grid) return "a mask needs the grid family";
    if (m.feature_mask.size() != static_cast<std::size_t>(m.h.grid_t) * S2) return "a mask of grid_t x side x side";
    if (std::ranges::any_of(m.feature_mask, [](std::uint8_t v) { return v > 1; })) return "mask values 0 or 1";
  }
  return {};
}

// Codes into a plane's bytes (one per code at 8 bits, bit-packed below, none at 0 bits).
void put_codes(const std::vector<unsigned>& codes, int bits, std::uint8_t* out) {
  if (bits == 8) {
    for (std::size_t j = 0; j < codes.size(); ++j) out[j] = static_cast<std::uint8_t>(codes[j]);
  } else if (bits > 0) {
    std::fill_n(out, packed_plane_bytes(codes.size(), bits), std::uint8_t{0});
    for (std::size_t j = 0; j < codes.size(); ++j) put_packed_code(out, j, bits, codes[j]);
  }
}

}  // namespace

void quantise_plane(std::span<float> plane, int bits, bool trim, std::span<const std::uint8_t> active) {
  if (bits > 8) {  // the fill only
    if (active.empty()) return;
    const PlaneCodes pc = plane_codes(plane, 8, trim, active);
    for (std::size_t j = 0; j < plane.size(); ++j) {
      if (!active[j]) plane[j] = pc.fill;
    }
    return;
  }
  const PlaneCodes pc = plane_codes(plane, bits, trim, active);
  std::size_t n = 0;
  for (std::size_t j = 0; j < plane.size(); ++j) plane[j] = active.empty() || active[j] ? dq8(pc.codes[n++], {pc.lo, pc.hi}, bits) : pc.fill;
}

std::vector<std::uint8_t> vq_assign(const Model& m) {
  const Hyper& h = m.h;
  const int G = m.vq_groups(), d = m.vq_dim, K = 1 << m.vq_bits, C = h.feature_channels();
  const std::size_t S2 = static_cast<std::size_t>(h.feature_side()) * h.feature_side();
  const std::size_t slices = static_cast<std::size_t>(h.bases) * h.grid_t;
  std::vector<std::uint8_t> idx(slices * static_cast<std::size_t>(G) * S2);
  std::vector<float> v(static_cast<std::size_t>(d));
  for (std::size_t sl = 0; sl < slices; ++sl) {
    const float* f = m.features.data() + sl * static_cast<std::size_t>(C) * S2;
    for (int g = 0; g < G; ++g) {
      const float* cb = m.vq_codebook.data() + static_cast<std::size_t>(g) * K * d;
      for (std::size_t j = 0; j < S2; ++j) {
        for (int c = 0; c < d; ++c) v[static_cast<std::size_t>(c)] = f[static_cast<std::size_t>(g * d + c) * S2 + j];
        int best = 0;
        float best_d = std::numeric_limits<float>::infinity();
        for (int k = 0; k < K; ++k) {
          float dist = 0;
          for (int c = 0; c < d; ++c) {
            const float e = v[static_cast<std::size_t>(c)] - cb[static_cast<std::size_t>(k) * d + c];
            dist += e * e;
          }
          if (dist < best_d) {
            best_d = dist;
            best = k;
          }
        }
        idx[(sl * static_cast<std::size_t>(G) + static_cast<std::size_t>(g)) * S2 + j] = static_cast<std::uint8_t>(best);
      }
    }
  }
  return idx;
}

namespace {

// The codebook in storage precision, ordered within each group by the codewords' projection on the group's first
// principal axis (power iteration from a fixed start): neighbouring grid points with similar codewords then get
// similar indices, which the lossless coder's numeric predictors exploit. The order changes no feature.
void vq_canonical(Model& m) {
  const int G = m.vq_groups(), d = m.vq_dim, K = 1 << m.vq_bits;
  for (float& c : m.vq_codebook) c = round_f16(c);
  for (int g = 0; g < G; ++g) {
    float* cb = m.vq_codebook.data() + static_cast<std::size_t>(g) * K * d;
    std::vector<double> mean(static_cast<std::size_t>(d), 0.0), axis(static_cast<std::size_t>(d), 1.0);
    for (int k = 0; k < K; ++k) {
      for (int c = 0; c < d; ++c) mean[static_cast<std::size_t>(c)] += static_cast<double>(cb[static_cast<std::size_t>(k) * d + c]) / K;
    }
    for (int it = 0; it < 30; ++it) {
      std::vector<double> next(static_cast<std::size_t>(d), 0.0);
      for (int k = 0; k < K; ++k) {
        double dot = 0;
        for (int c = 0; c < d; ++c) dot += (cb[static_cast<std::size_t>(k) * d + c] - mean[static_cast<std::size_t>(c)]) * axis[static_cast<std::size_t>(c)];
        for (int c = 0; c < d; ++c) next[static_cast<std::size_t>(c)] += dot * (cb[static_cast<std::size_t>(k) * d + c] - mean[static_cast<std::size_t>(c)]);
      }
      double norm = 0;
      for (const double x : next) norm += x * x;
      if (norm <= 0) break;
      for (int c = 0; c < d; ++c) axis[static_cast<std::size_t>(c)] = next[static_cast<std::size_t>(c)] / std::sqrt(norm);
    }
    std::vector<std::pair<double, int>> order;
    for (int k = 0; k < K; ++k) {
      double dot = 0;
      for (int c = 0; c < d; ++c) dot += cb[static_cast<std::size_t>(k) * d + c] * axis[static_cast<std::size_t>(c)];
      order.emplace_back(dot, k);
    }
    std::ranges::sort(order);
    std::vector<float> sorted(static_cast<std::size_t>(K) * d);
    for (int k = 0; k < K; ++k) {
      std::copy_n(cb + static_cast<std::size_t>(order[static_cast<std::size_t>(k)].second) * d, d, sorted.begin() + static_cast<std::ptrdiff_t>(k) * d);
    }
    std::ranges::copy(sorted, cb);
  }
}

bool vq_valid(const Model& m) {
  return m.vq_bits >= 2 && m.vq_bits <= 8 && m.vq_dim >= 1 && m.h.feature_channels() % m.vq_dim == 0 &&
         m.vq_codebook.size() == static_cast<std::size_t>(m.vq_groups()) * (std::size_t{1} << m.vq_bits) * static_cast<std::size_t>(m.vq_dim);
}

// Features replaced by their codewords.
void vq_apply(Model& m, std::span<const std::uint8_t> idx) {
  const Hyper& h = m.h;
  const int G = m.vq_groups(), d = m.vq_dim, K = 1 << m.vq_bits, C = h.feature_channels();
  const std::size_t S2 = static_cast<std::size_t>(h.feature_side()) * h.feature_side();
  const std::size_t slices = static_cast<std::size_t>(h.bases) * h.grid_t;
  for (std::size_t sl = 0; sl < slices; ++sl) {
    float* f = m.features.data() + sl * static_cast<std::size_t>(C) * S2;
    for (int g = 0; g < G; ++g) {
      const float* cb = m.vq_codebook.data() + static_cast<std::size_t>(g) * K * d;
      for (std::size_t j = 0; j < S2; ++j) {
        const std::size_t q = idx[(sl * static_cast<std::size_t>(G) + static_cast<std::size_t>(g)) * S2 + j];
        for (int c = 0; c < d; ++c) f[static_cast<std::size_t>(g * d + c) * S2 + j] = cb[q * static_cast<std::size_t>(d) + static_cast<std::size_t>(c)];
      }
    }
  }
}

}  // namespace

std::vector<std::uint8_t> mask_runs(std::span<const std::uint8_t> mask) {
  std::vector<std::uint8_t> out;
  const auto put = [&](std::size_t n) {
    do {
      const auto low = static_cast<std::uint8_t>(n & 0x7f);
      n >>= 7;
      out.push_back(static_cast<std::uint8_t>(low | (n ? 0x80 : 0)));
    } while (n);
  };
  std::size_t j = 0;
  std::uint8_t value = 0;
  while (j < mask.size()) {
    std::size_t n = 0;
    while (j + n < mask.size() && (mask[j + n] != 0) == (value != 0)) ++n;
    put(n);
    j += n;
    value ^= 1;
  }
  if (mask.empty()) put(0);
  return out;
}

std::size_t read_mask_runs(std::span<const std::uint8_t> runs, std::span<std::uint8_t> mask) {
  std::size_t at = 0, j = 0;
  std::uint8_t value = 0;
  bool first = true;
  do {
    std::size_t n = 0;
    for (int shift = 0;; shift += 7) {
      if (at >= runs.size() || shift > 56) return 0;
      const std::uint8_t b = runs[at++];
      n |= static_cast<std::size_t>(b & 0x7f) << shift;
      if (!(b & 0x80)) break;
    }
    if ((n == 0 && !first) || n > mask.size() - j) return 0;
    std::fill_n(mask.begin() + static_cast<std::ptrdiff_t>(j), n, value);
    j += n;
    value ^= 1;
    first = false;
  } while (j < mask.size());
  return at;
}

namespace {

// --- the multi family (file version 4) ---

// Every feature plane of a multi model: its first value, values, level, time slice and first mask point.
struct PlaneAt {
  std::size_t value0, n, mask0;
  int level, slice;
};
std::vector<PlaneAt> multi_planes(const Hyper& h) {
  std::vector<PlaneAt> v;
  const auto vols = volumes(h);
  for (std::size_t l = 0; l < vols.size(); ++l) {
    const Volume& vol = vols[l];
    const std::size_t n = vol.plane_values();
    for (int k = 0; k < h.bases; ++k) {
      for (int t = 0; t < vol.slices; ++t) {
        for (int c = 0; c < vol.channels; ++c) {
          const std::size_t local = (static_cast<std::size_t>(k) * vol.slices + static_cast<std::size_t>(t)) * vol.channels + static_cast<std::size_t>(c);
          v.push_back({vol.value0 + local * n, n, vol.mask0 + static_cast<std::size_t>(t) * n, static_cast<int>(l), t});
        }
      }
    }
  }
  return v;
}

int multi_bits(const Model& m, std::size_t plane) {
  return m.plane_bits.empty() ? std::min(m.feature_bits, 8) : m.plane_bits[plane];
}

// Why a multi model cannot be stored, or empty.
std::string multi_error(const Model& m) {
  const auto planes = multi_planes(m.h);
  if (!m.plane_bits.empty()) {
    if (m.plane_bits.size() != planes.size()) return "one width per feature plane";
    for (std::size_t k = 0; k < planes.size(); ++k) {
      if (m.plane_bits[k] < 1 || m.plane_bits[k] > 8) return "widths 1 to 8";
      if (k > 0 && planes[k].level == planes[k - 1].level && m.plane_bits[k] != m.plane_bits[k - 1]) return "one width per level";
    }
  } else if (m.feature_bits < 1) {
    return "feature bits";
  }
  if (!m.feature_mask.empty()) {
    const Volume last = volumes(m.h).back();
    if (m.feature_mask.size() != last.mask0 + static_cast<std::size_t>(last.slices) * last.plane_values()) return "a mask per level";
    if (std::ranges::any_of(m.feature_mask, [](std::uint8_t v) { return v > 1; })) return "mask values 0 or 1";
  }
  if (m.mlp_bits != 8 && m.mlp_bits != 16) return "MLP weights at 8 or 16 bits";
  if (m.vq_bits > 0) return "no vector quantisation";
  return {};
}

std::span<const std::uint8_t> multi_mask(const Model& m, const PlaneAt& p) {
  if (m.feature_mask.empty()) return {};
  return std::span(m.feature_mask).subspan(p.mask0, p.n);
}

// A layer's weights as stored at 8 bits: per output unit the fp16 scale of its largest |w| over 127, and signed codes.
float weight_scale8(const Dense& d, int o) {
  float mx = 0.f;
  for (int i = 0; i < d.in; ++i) mx = std::max(mx, std::abs(d.w[static_cast<std::size_t>(o) * d.in + i]));
  return round_f16(mx / 127.f);
}
std::int8_t weight_code8(float w, float scale) {
  if (!(scale > 0.f)) return 0;
  return static_cast<std::int8_t>(std::clamp(std::lround(w / scale), -127L, 127L));
}

std::size_t multi_storage(const Model& m) {
  const auto vols = volumes(m.h);
  const auto planes = multi_planes(m.h);
  const bool mk = !m.feature_mask.empty() || !m.raw_mask_at.empty();
  std::size_t n = 20 + 16 * vols.size();  // level count, Fourier frequencies, flags, mask bytes; per level its shape and width
  if (!m.raw_offsets.empty()) {  // the resident form (pack_features)
    n += m.raw_mask.size() + m.raw_u8.size() + planes.size() * (mk ? 6 : 4);
  } else {
    if (mk) {
      for (const Volume& v : vols) {
        for (int t = 0; t < v.slices; ++t) n += mask_runs(std::span(m.feature_mask).subspan(v.mask0 + static_cast<std::size_t>(t) * v.plane_values(), v.plane_values())).size();
      }
    }
    for (std::size_t k = 0; k < planes.size(); ++k) {
      const auto act = multi_mask(m, planes[k]);
      const std::size_t pts = act.empty() ? planes[k].n : static_cast<std::size_t>(std::ranges::count(act, std::uint8_t{1}));
      n += (mk ? 6 : 4) + packed_plane_bytes(pts, multi_bits(m, k));
    }
  }
  n += 2 * m.basis.params();
  for (const auto& d : m.layers) n += m.mlp_bits == 8 ? d.w.size() + 2 * static_cast<std::size_t>(d.out) + 2 * d.b.size() : 2 * d.params();
  for (const auto& d : m.films) n += 2 * d.params();
  n += 2 * (m.z_mean.size() + m.z_std.size());
  for (const auto& z : m.z_train) n += 2 * z.size();
  return n;
}

void multi_pack(Model& m) {
  const auto vols = volumes(m.h);
  const auto planes = multi_planes(m.h);
  if (m.masked()) {
    for (const Volume& v : vols) {
      for (int t = 0; t < v.slices; ++t) {
        m.raw_mask_at.push_back(static_cast<std::uint32_t>(m.raw_mask.size()));
        const auto r = mask_runs(std::span(m.feature_mask).subspan(v.mask0 + static_cast<std::size_t>(t) * v.plane_values(), v.plane_values()));
        m.raw_mask.insert(m.raw_mask.end(), r.begin(), r.end());
      }
    }
  }
  std::vector<PlaneCodes> pcs;
  std::size_t total = 0;
  for (std::size_t k = 0; k < planes.size(); ++k) {
    pcs.push_back(plane_codes(std::span(m.features.data() + planes[k].value0, planes[k].n), multi_bits(m, k), m.feature_trim, multi_mask(m, planes[k])));
    m.raw_offsets.push_back(static_cast<std::uint32_t>(total));
    total += packed_plane_bytes(pcs.back().codes.size(), multi_bits(m, k));
  }
  m.raw_u8.assign(total, 0);
  for (std::size_t k = 0; k < pcs.size(); ++k) {
    m.raw_ranges.push_back(pcs[k].lo);
    m.raw_ranges.push_back(pcs[k].hi);
    if (m.masked()) m.raw_fill.push_back(pcs[k].fill);
    put_codes(pcs[k].codes, multi_bits(m, k), m.raw_u8.data() + m.raw_offsets[k]);
  }
}

void multi_quantise(Model& m) {
  const auto planes = multi_planes(m.h);
  for (std::size_t k = 0; k < planes.size(); ++k) {
    quantise_plane(std::span(m.features.data() + planes[k].value0, planes[k].n), multi_bits(m, k), m.feature_trim, multi_mask(m, planes[k]));
  }
  if (m.mlp_bits == 8) {
    for (auto& d : m.layers) quantise_weights8(d);
  }
}

std::expected<void, std::string> save_multi(std::ostream& o, const Model& m) {
  if (const std::string e = multi_error(m); !e.empty()) return std::unexpected("nvfx: multi-level storage: " + e);
  const Hyper& h = m.h;
  const auto vols = volumes(h);
  const auto planes = multi_planes(h);
  const bool mk = m.masked();
  o.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  bin::put(o, kVersionMulti);
  bin::put(o, static_cast<std::uint32_t>(h.arch));
  for (const int v : {h.size, h.frames, static_cast<int>(h.loop), h.n_controls, h.n_latent, h.bases, h.grid_t, h.grid,
                      h.channels, h.hidden, h.layers, h.latent, h.c0, h.c1, h.c2}) {
    bin::put(o, static_cast<std::int32_t>(v));
  }
  std::vector<std::uint8_t> runs;  // every (level, slice)'s mask runs, one after another
  if (mk) {
    for (const Volume& v : vols) {
      for (int t = 0; t < v.slices; ++t) {
        const auto r = mask_runs(std::span(m.feature_mask).subspan(v.mask0 + static_cast<std::size_t>(t) * v.plane_values(), v.plane_values()));
        runs.insert(runs.end(), r.begin(), r.end());
      }
    }
  }
  bin::put(o, static_cast<std::uint32_t>(vols.size()));
  bin::put(o, static_cast<std::uint32_t>(h.pe_xy));
  bin::put(o, static_cast<std::uint32_t>(h.pe_t));
  bin::put(o, static_cast<std::uint32_t>((mk ? 1u : 0u) | (m.mlp_bits == 8 ? 2u : 0u)));
  bin::put(o, static_cast<std::uint32_t>(runs.size()));
  for (std::size_t l = 0; l < vols.size(); ++l) {
    std::size_t first = 0;
    while (first < planes.size() && planes[first].level != static_cast<int>(l)) ++first;
    for (const int v : {vols[l].side, vols[l].slices, vols[l].channels, multi_bits(m, first)}) bin::put(o, static_cast<std::uint32_t>(v));
  }
  bin::put_str(o, m.effect, 32);
  bin::put(o, m.fps);
  for (int k = 0; k < h.n_controls; ++k) {
    bin::put_str(o, static_cast<std::size_t>(k) < m.control_names.size() ? m.control_names[static_cast<std::size_t>(k)] : std::string{}, 16);
  }
  o.write(reinterpret_cast<const char*>(runs.data()), static_cast<std::streamsize>(runs.size()));
  std::vector<std::uint8_t> codes;
  for (std::size_t k = 0; k < planes.size(); ++k) {
    const int b = multi_bits(m, k);
    const PlaneCodes pc = plane_codes(std::span(m.features.data() + planes[k].value0, planes[k].n), b, m.feature_trim, multi_mask(m, planes[k]));
    put_f16(o, std::array{pc.lo, pc.hi});
    if (mk) put_f16(o, std::array{pc.fill});
    codes.assign(std::max<std::size_t>(1, packed_plane_bytes(pc.codes.size(), b)), 0);
    put_codes(pc.codes, b, codes.data());
    o.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(packed_plane_bytes(pc.codes.size(), b)));
  }
  put_dense(o, m.basis);
  bin::put(o, static_cast<std::uint32_t>(m.layers.size()));
  for (const auto& d : m.layers) {
    if (m.mlp_bits != 8) {
      put_dense(o, d);
      continue;
    }
    bin::put(o, static_cast<std::uint32_t>(d.in));
    bin::put(o, static_cast<std::uint32_t>(d.out));
    std::vector<float> sc(static_cast<std::size_t>(d.out));
    for (int u = 0; u < d.out; ++u) sc[static_cast<std::size_t>(u)] = weight_scale8(d, u);
    put_f16(o, sc);
    for (int u = 0; u < d.out; ++u) {
      for (int i = 0; i < d.in; ++i) {
        const std::int8_t q = weight_code8(d.w[static_cast<std::size_t>(u) * d.in + i], sc[static_cast<std::size_t>(u)]);
        o.put(static_cast<char>(static_cast<std::uint8_t>(q + 128)));  // excess 128: 1 to 255
      }
    }
    put_f16(o, d.b);
  }
  bin::put(o, static_cast<std::uint32_t>(m.films.size()));
  for (const auto& d : m.films) put_dense(o, d);
  put_f16(o, m.z_mean);
  put_f16(o, m.z_std);
  bin::put(o, static_cast<std::uint32_t>(m.z_train.size()));
  for (const auto& z : m.z_train) put_f16(o, z);
  if (!o) return std::unexpected("nvfx: write failed");
  return {};
}

std::expected<Model, std::string> load_multi(std::istream& i, Hyper h) {
  const auto nl = bin::get<std::uint32_t>(i), pxy = bin::get<std::uint32_t>(i), pt = bin::get<std::uint32_t>(i), flags = bin::get<std::uint32_t>(i);
  const auto mask_bytes = bin::get<std::uint32_t>(i);
  if (!nl || !pxy || !pt || !flags || !mask_bytes || *nl < 1 || *nl > 16 || *pxy > 12 || *pt > 12 || (*flags & ~3u) != 0 ||
      ((*flags & 1u) == 0 && *mask_bytes != 0) || *mask_bytes > (1u << 28)) {
    return std::unexpected("nvfx: bad multi-level header");
  }
  h.pe_xy = static_cast<int>(*pxy);
  h.pe_t = static_cast<int>(*pt);
  h.levels.clear();
  std::vector<int> level_bits;
  for (std::uint32_t l = 0; l < *nl; ++l) {
    std::array<std::uint32_t, 4> f{};
    for (auto& v : f) {
      const auto r = bin::get<std::uint32_t>(i);
      if (!r || *r > 4096) return std::unexpected("nvfx: bad level");
      v = *r;
    }
    if (f[3] < 1 || f[3] > 8) return std::unexpected("nvfx: bad level width");
    h.levels.push_back({static_cast<int>(f[0]), static_cast<int>(f[1]), static_cast<int>(f[2])});
    level_bits.push_back(static_cast<int>(f[3]));
  }
  Model m;
  try {
    m = init_model(h, 0);  // shapes and validation
  } catch (const std::exception& e) {
    return std::unexpected(std::format("nvfx: {}", e.what()));
  }
  const auto effect = bin::get_str(i, 32);
  const auto fps = bin::get<float>(i);
  if (!effect || !fps) return std::unexpected("nvfx: bad header");
  m.effect = *effect;
  m.fps = *fps;
  m.feature_bits = 8;
  m.mlp_bits = (*flags & 2u) ? 8 : 16;
  for (int k = 0; k < h.n_controls; ++k) {
    auto name = bin::get_str(i, 16);
    if (!name) return std::unexpected(name.error());
    m.control_names.push_back(*name);
  }
  const auto vols = volumes(h);
  const auto planes = multi_planes(h);
  m.plane_bits.resize(planes.size());
  for (std::size_t k = 0; k < planes.size(); ++k) m.plane_bits[k] = static_cast<std::uint8_t>(level_bits[static_cast<std::size_t>(planes[k].level)]);
  const bool mk = (*flags & 1u) != 0;
  if (mk) {  // the runs of every (level, slice), one after another, exactly mask_bytes of them
    m.feature_mask.assign(vols.back().mask0 + static_cast<std::size_t>(vols.back().slices) * vols.back().plane_values(), 0);
    std::vector<std::uint8_t> runs(*mask_bytes);
    i.read(reinterpret_cast<char*>(runs.data()), static_cast<std::streamsize>(runs.size()));
    if (!i) return std::unexpected("truncated file");
    std::size_t at = 0;
    for (const Volume& v : vols) {
      for (int t = 0; t < v.slices; ++t) {
        const std::size_t used = read_mask_runs(std::span(runs).subspan(at), std::span(m.feature_mask.data() + v.mask0 + static_cast<std::size_t>(t) * v.plane_values(), v.plane_values()));
        if (used == 0) return std::unexpected("nvfx: bad mask runs");
        at += used;
      }
    }
    if (at != runs.size()) return std::unexpected("nvfx: bad mask runs");
  }
  std::vector<std::uint8_t> codes;
  for (std::size_t k = 0; k < planes.size(); ++k) {
    const int b = m.plane_bits[k];
    const auto act = multi_mask(m, planes[k]);
    const std::size_t n = act.empty() ? planes[k].n : static_cast<std::size_t>(std::ranges::count(act, std::uint8_t{1}));
    std::array<float, 3> r{};
    if (auto e = get_f16(i, std::span(r).first(mk ? 3 : 2)); !e) return std::unexpected(e.error());
    codes.assign(packed_plane_bytes(n, b), 0);
    i.read(reinterpret_cast<char*>(codes.data()), static_cast<std::streamsize>(codes.size()));
    if (!i) return std::unexpected("truncated file");
    float* v = m.features.data() + planes[k].value0;
    for (std::size_t j = 0, q = 0; j < planes[k].n; ++j) {
      if (!act.empty() && !act[j]) {
        v[j] = r[2];
        continue;
      }
      v[j] = dq8(b == 8 ? codes[q] : packed_code(codes.data(), q, b), {r[0], r[1]}, b);
      ++q;
    }
  }
  const auto expect_dense = [&](Dense& slot, bool w8) -> std::expected<void, std::string> {
    if (!w8) {
      auto d = get_dense(i);
      if (!d) return std::unexpected(d.error());
      if (d->in != slot.in || d->out != slot.out) return std::unexpected("nvfx: layer shape does not match the header");
      slot = std::move(*d);
      return {};
    }
    const auto in = bin::get<std::uint32_t>(i), out = bin::get<std::uint32_t>(i);
    if (!in || !out || static_cast<int>(*in) != slot.in || static_cast<int>(*out) != slot.out) return std::unexpected("nvfx: layer shape does not match the header");
    std::vector<float> sc(static_cast<std::size_t>(slot.out));
    if (auto e = get_f16(i, sc); !e) return std::unexpected(e.error());
    std::vector<char> q(slot.w.size());
    i.read(q.data(), static_cast<std::streamsize>(q.size()));
    if (!i) return std::unexpected("truncated file");
    for (std::size_t j = 0; j < q.size(); ++j) {
      const int c = static_cast<int>(static_cast<std::uint8_t>(q[j])) - 128;  // excess 128
      if (c < -127) return std::unexpected("nvfx: bad 8-bit weight");
      slot.w[j] = static_cast<float>(c) * sc[j / static_cast<std::size_t>(slot.in)];
    }
    return get_f16(i, slot.b);
  };
  if (auto e = expect_dense(m.basis, false); !e) return std::unexpected(e.error());
  for (auto* group : {&m.layers, &m.films}) {
    const auto n = bin::get<std::uint32_t>(i);
    if (!n || *n != group->size()) return std::unexpected("nvfx: layer count does not match the header");
    for (auto& d : *group) {
      if (auto e = expect_dense(d, group == &m.layers && m.mlp_bits == 8); !e) return std::unexpected(e.error());
    }
  }
  if (auto e = get_f16(i, m.z_mean); !e) return std::unexpected(e.error());
  if (auto e = get_f16(i, m.z_std); !e) return std::unexpected(e.error());
  const auto nz = bin::get<std::uint32_t>(i);
  if (!nz || *nz > 1000000) return std::unexpected("nvfx: bad code count");
  m.z_train.assign(*nz, std::vector<float>(static_cast<std::size_t>(h.n_latent)));
  for (auto& z : m.z_train) {
    if (auto e = get_f16(i, z); !e) return std::unexpected(e.error());
  }
  m.pack_features();
  return m;
}

}  // namespace

void quantise_weights8(Dense& d) {
  for (int o = 0; o < d.out; ++o) {
    const float s = weight_scale8(d, o);
    for (int i = 0; i < d.in; ++i) {
      float& w = d.w[static_cast<std::size_t>(o) * d.in + i];
      w = static_cast<float>(weight_code8(w, s)) * s;
    }
  }
}

void Model::pack_features() {
  raw_f16.clear();
  raw_u8.clear();
  raw_ranges.clear();
  raw_codebook.clear();
  raw_offsets.clear();
  raw_mask.clear();
  raw_fill.clear();
  raw_mask_at.clear();
  if (h.arch == Arch::multi) {
    multi_pack(*this);
    return;
  }
  if (per_plane()) {
    const std::size_t p = plane_size(h);
    if (masked()) {
      const std::size_t mb = packed_plane_bytes(p, 1);
      raw_mask.assign(static_cast<std::size_t>(h.grid_t) * mb, 0);
      for (std::size_t t = 0; t < static_cast<std::size_t>(h.grid_t); ++t) {
        for (std::size_t j = 0; j < p; ++j) put_packed_code(raw_mask.data() + t * mb, j, 1, feature_mask[t * p + j]);
      }
    }
    std::vector<PlaneCodes> pcs;
    std::size_t total = 0;
    for (std::size_t off = 0, k = 0; off < features.size(); off += p, ++k) {
      pcs.push_back(plane_codes(std::span(features.data() + off, p), plane_bits[k], feature_trim, plane_mask(*this, k)));
      raw_offsets.push_back(static_cast<std::uint32_t>(total));
      total += packed_plane_bytes(pcs.back().codes.size(), plane_bits[k]);
    }
    raw_u8.assign(total, 0);
    for (std::size_t k = 0; k < pcs.size(); ++k) {
      raw_ranges.push_back(pcs[k].lo);
      raw_ranges.push_back(pcs[k].hi);
      if (masked()) raw_fill.push_back(pcs[k].fill);
      put_codes(pcs[k].codes, plane_bits[k], raw_u8.data() + raw_offsets[k]);
    }
    return;
  }
  if (vq_groups() > 0) {
    const std::size_t S2 = static_cast<std::size_t>(h.feature_side()) * h.feature_side(), pb = packed_plane_bytes(S2, vq_bits);
    const auto idx = vq_assign(*this);
    raw_u8.assign(idx.size() / S2 * pb, 0);
    for (std::size_t p = 0; p < idx.size() / S2; ++p) {
      for (std::size_t j = 0; j < S2; ++j) {
        if (vq_bits == 8) raw_u8[p * pb + j] = idx[p * S2 + j];
        else put_packed_code(raw_u8.data() + p * pb, j, vq_bits, idx[p * S2 + j]);
      }
    }
    raw_codebook = vq_codebook;
    return;
  }
  if (feature_bits < 16) {
    const std::size_t p = plane_size(h), pb = packed_plane_bytes(p, feature_bits);
    raw_u8.assign(features.size() / p * pb, 0);
    for (std::size_t off = 0, k = 0; off < features.size(); off += p, ++k) {
      const std::span plane(features.data() + off, p);
      const Plane8 r = plane_range(plane, feature_bits, feature_trim);
      raw_ranges.push_back(r.lo);
      raw_ranges.push_back(r.hi);
      put_plane_codes(plane, r, feature_bits, raw_u8.data() + k * pb);
    }
  } else {
    raw_f16.reserve(features.size());
    for (const float v : features) raw_f16.push_back(std::bit_cast<std::uint16_t>(static_cast<std::float16_t>(v)));
  }
}

void quantise_like_storage(Model& m) {
  if (m.h.arch == Arch::multi) {
    multi_quantise(m);  // features at their widths, the MLP's weights at mlp_bits (then fp16, below, changes nothing)
  } else if (m.per_plane()) {
    const std::size_t p = plane_size(m.h);
    for (std::size_t off = 0, k = 0; off < m.features.size() && k < m.plane_bits.size(); off += p, ++k) {
      quantise_plane(std::span(m.features.data() + off, p), m.plane_bits[k], m.feature_trim, plane_mask(m, k));
    }
  } else if (m.vq_groups() > 0) {
    vq_canonical(m);
    vq_apply(m, vq_assign(m));
  } else if (m.feature_bits < 16) {
    const std::size_t p = plane_size(m.h);
    for (std::size_t off = 0; off < m.features.size(); off += p) {
      const std::span plane(m.features.data() + off, p);
      const Plane8 r = plane_range(plane, m.feature_bits, m.feature_trim);
      for (float& v : plane) v = dq8(q8(v, r, m.feature_bits), r, m.feature_bits);
    }
  } else {
    for (float& v : m.features) v = round_f16(v);
  }
  const auto r16 = [](std::vector<float>& v) { std::ranges::for_each(v, [](float& x) { x = round_f16(x); }); };
  r16(m.basis.w);
  r16(m.basis.b);
  for (auto* group : {&m.layers, &m.films}) {
    for (auto& d : *group) {
      if (group == &m.layers && m.h.arch == Arch::multi && m.mlp_bits == 8) {
        r16(d.b);  // the weights are their 8-bit codes times an fp16 scale already
        continue;
      }
      r16(d.w);
      r16(d.b);
    }
  }
}

std::expected<void, std::string> save_model(const std::filesystem::path& path, const Model& m) {
  std::error_code ec;
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
  std::ofstream o(path, std::ios::binary);
  if (!o) return std::unexpected(std::format("nvfx: cannot write {}", path.string()));
  return save_model(o, m);
}

std::expected<void, std::string> save_model(std::ostream& o, const Model& m0) {
  if (m0.h.arch == Arch::multi) return save_multi(o, m0);
  const bool vq = m0.vq_bits > 0;
  if (vq && !vq_valid(m0)) return std::unexpected("nvfx: bad vector quantisation (bits 2 to 8, a codebook per channel group)");
  const bool mixed = m0.per_plane();
  if (mixed) {
    if (const std::string e = per_plane_error(m0); !e.empty()) return std::unexpected("nvfx: per-plane storage: " + e);
  } else if (!m0.feature_mask.empty()) {
    return std::unexpected("nvfx: a feature mask needs per-plane widths");
  }
  Model canon;
  if (vq) {  // the stored codebook order (vq_canonical)
    canon = m0;
    vq_canonical(canon);
  }
  const Model& m = vq ? canon : m0;
  const Hyper& h = m.h;
  o.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  bin::put(o, vq ? kVersionVq : mixed ? kVersionMixed : kVersion);
  bin::put(o, static_cast<std::uint32_t>(h.arch));
  for (const int v : {h.size, h.frames, static_cast<int>(h.loop), h.n_controls, h.n_latent, h.bases, h.grid_t, h.grid,
                      h.channels, h.hidden, h.layers, h.latent, h.c0, h.c1, h.c2}) {
    bin::put(o, static_cast<std::int32_t>(v));
  }
  bin::put_str(o, m.effect, 32);
  bin::put(o, m.fps);
  bin::put(o, static_cast<std::uint32_t>(m.feature_bits));
  if (vq) {
    bin::put(o, static_cast<std::uint32_t>(m.vq_bits));
    bin::put(o, static_cast<std::uint32_t>(m.vq_dim));
  }
  for (int k = 0; k < h.n_controls; ++k) {
    bin::put_str(o, static_cast<std::size_t>(k) < m.control_names.size() ? m.control_names[static_cast<std::size_t>(k)] : std::string{}, 16);
  }
  if (!valid_feature_bits(m.feature_bits)) return std::unexpected("nvfx: feature bits must be 2 to 8 or 16");
  if (mixed) {
    // Flags (bit 0: a mask), the width of every plane, the mask (per slice, bit-packed), then per plane its range
    // (fp16), its fill (fp16, masked) and the codes of its stored points (packed_plane_bytes at its width).
    const bool mk = m.masked();
    bin::put(o, static_cast<std::uint32_t>(mk ? 1 : 0));
    o.write(reinterpret_cast<const char*>(m.plane_bits.data()), static_cast<std::streamsize>(m.plane_bits.size()));
    const std::size_t p = plane_size(h);
    if (mk) {
      std::vector<std::uint8_t> packed(packed_plane_bytes(p, 1));
      for (std::size_t t = 0; t < static_cast<std::size_t>(h.grid_t); ++t) {
        std::ranges::fill(packed, std::uint8_t{0});
        for (std::size_t j = 0; j < p; ++j) put_packed_code(packed.data(), j, 1, m.feature_mask[t * p + j]);
        o.write(reinterpret_cast<const char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
      }
    }
    std::vector<std::uint8_t> codes(p);
    for (std::size_t off = 0, k = 0; off < m.features.size(); off += p, ++k) {
      const int b = m.plane_bits[k];
      const PlaneCodes pc = plane_codes(std::span(m.features.data() + off, p), b, m.feature_trim, plane_mask(m, k));
      put_f16(o, std::array{pc.lo, pc.hi});
      if (mk) put_f16(o, std::array{pc.fill});
      put_codes(pc.codes, b, codes.data());
      o.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(packed_plane_bytes(pc.codes.size(), b)));
    }
  } else if (vq) {
    // The codebooks (fp16), then the index planes [basis][slice][group][side][side].
    put_f16(o, m.vq_codebook);
    const auto idx = vq_assign(m);
    const std::size_t S2 = plane_size(h), pb = packed_plane_bytes(S2, m.vq_bits);
    std::vector<std::uint8_t> codes(pb);
    for (std::size_t p = 0; p < idx.size() / S2; ++p) {
      std::ranges::fill(codes, std::uint8_t{0});
      for (std::size_t j = 0; j < S2; ++j) {
        if (m.vq_bits == 8) codes[j] = idx[p * S2 + j];
        else put_packed_code(codes.data(), j, m.vq_bits, idx[p * S2 + j]);
      }
      o.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(codes.size()));
    }
  } else if (m.feature_bits < 16) {
    const std::size_t p = plane_size(h);
    std::vector<std::uint8_t> codes(packed_plane_bytes(p, m.feature_bits));
    for (std::size_t off = 0; off < m.features.size(); off += p) {
      const std::span plane(m.features.data() + off, p);
      const Plane8 r = plane_range(plane, m.feature_bits, m.feature_trim);
      put_f16(o, std::array{r.lo, r.hi});
      put_plane_codes(plane, r, m.feature_bits, codes.data());
      o.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(codes.size()));
    }
  } else {
    put_f16(o, m.features);
  }
  put_dense(o, m.basis);
  bin::put(o, static_cast<std::uint32_t>(m.layers.size()));
  for (const auto& d : m.layers) put_dense(o, d);
  bin::put(o, static_cast<std::uint32_t>(m.films.size()));
  for (const auto& d : m.films) put_dense(o, d);
  put_f16(o, m.z_mean);
  put_f16(o, m.z_std);
  bin::put(o, static_cast<std::uint32_t>(m.z_train.size()));
  for (const auto& z : m.z_train) put_f16(o, z);
  if (!o) return std::unexpected("nvfx: write failed");
  return {};
}

std::expected<Model, std::string> load_model(const std::filesystem::path& path) {
  std::ifstream i(path, std::ios::binary);
  if (!i) return std::unexpected(std::format("nvfx: cannot open {}", path.string()));
  auto m = load_model(i);
  if (!m) return std::unexpected(std::format("{} ({})", m.error(), path.string()));
  return m;
}

std::expected<Model, std::string> load_model(std::istream& i) {
  std::string magic(kMagic.size(), '\0');
  i.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!i || magic != kMagic) return std::unexpected("nvfx: not a model");
  const auto version = bin::get<std::uint32_t>(i);
  if (!version || (*version != kVersion && *version != kVersionVq && *version != kVersionMixed && *version != kVersionMulti)) {
    return std::unexpected("nvfx: unsupported version");
  }
  const auto arch = bin::get<std::uint32_t>(i);
  if (!arch) return std::unexpected(arch.error());
  Hyper h;
  h.arch = static_cast<Arch>(*arch);
  int* fields[] = {&h.size, &h.frames, nullptr, &h.n_controls, &h.n_latent, &h.bases, &h.grid_t, &h.grid,
                   &h.channels, &h.hidden, &h.layers, &h.latent, &h.c0, &h.c1, &h.c2};
  for (int* f : fields) {
    const auto v = bin::get<std::int32_t>(i);
    if (!v) return std::unexpected(v.error());
    if (f) *f = *v;
    else h.loop = *v != 0;
  }
  if ((h.arch == Arch::multi) != (*version == kVersionMulti)) return std::unexpected("nvfx: the multi family is file version 4");
  if (h.arch == Arch::multi) return load_multi(i, h);
  Model m;
  try {
    m = init_model(h, 0);  // shapes and validation
  } catch (const std::exception& e) {
    return std::unexpected(std::format("nvfx: {}", e.what()));
  }
  const auto effect = bin::get_str(i, 32);
  const auto fps = bin::get<float>(i);
  const auto bits = bin::get<std::uint32_t>(i);
  if (!effect || !fps || !bits || !valid_feature_bits(static_cast<int>(*bits))) return std::unexpected("nvfx: bad header");
  m.effect = *effect;
  m.fps = *fps;
  if (*version == kVersionVq) {
    const auto vb = bin::get<std::uint32_t>(i), vd = bin::get<std::uint32_t>(i);
    if (!vb || !vd || *vb < 2 || *vb > 8 || *vd < 1 || h.feature_channels() % static_cast<int>(*vd) != 0) return std::unexpected("nvfx: bad vector quantisation");
    m.vq_bits = static_cast<int>(*vb);
    m.vq_dim = static_cast<int>(*vd);
    m.vq_codebook.resize(static_cast<std::size_t>(m.vq_groups()) * (std::size_t{1} << m.vq_bits) * static_cast<std::size_t>(m.vq_dim));
  }
  for (int k = 0; k < h.n_controls; ++k) {
    auto name = bin::get_str(i, 16);
    if (!name) return std::unexpected(name.error());
    m.control_names.push_back(*name);
  }
  m.feature_bits = static_cast<int>(*bits);
  if (*version == kVersionMixed) {
    const std::size_t p = plane_size(h);
    const auto flags = bin::get<std::uint32_t>(i);
    if (!flags || (*flags & ~1u) != 0) return std::unexpected("nvfx: bad per-plane flags");
    const bool mk = (*flags & 1u) != 0;
    if (mk && h.arch != Arch::grid) return std::unexpected("nvfx: a feature mask needs the grid family");
    m.plane_bits.resize(m.features.size() / p);
    i.read(reinterpret_cast<char*>(m.plane_bits.data()), static_cast<std::streamsize>(m.plane_bits.size()));
    if (!i) return std::unexpected("truncated file");
    if (std::ranges::any_of(m.plane_bits, [](std::uint8_t b) { return b > 8; })) return std::unexpected("nvfx: bad plane bits");
    if (mk) {
      std::vector<std::uint8_t> packed(packed_plane_bytes(p, 1));
      m.feature_mask.resize(static_cast<std::size_t>(h.grid_t) * p);
      for (std::size_t t = 0; t < static_cast<std::size_t>(h.grid_t); ++t) {
        i.read(reinterpret_cast<char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
        if (!i) return std::unexpected("truncated file");
        for (std::size_t j = 0; j < p; ++j) m.feature_mask[t * p + j] = static_cast<std::uint8_t>(packed_code(packed.data(), j, 1));
      }
    }
    std::vector<std::uint8_t> codes(p);
    for (std::size_t off = 0, k = 0; off < m.features.size(); off += p, ++k) {
      const int b = m.plane_bits[k];
      const auto active = plane_mask(m, k);
      const std::size_t n = active.empty() ? p : static_cast<std::size_t>(std::ranges::count(active, std::uint8_t{1}));
      std::array<float, 3> r{};
      if (auto e = get_f16(i, std::span(r).first(mk ? 3 : 2)); !e) return std::unexpected(e.error());
      i.read(reinterpret_cast<char*>(codes.data()), static_cast<std::streamsize>(packed_plane_bytes(n, b)));
      if (!i) return std::unexpected("truncated file");
      float* v = m.features.data() + off;
      for (std::size_t j = 0, q = 0; j < p; ++j) {
        if (!active.empty() && !active[j]) {
          v[j] = r[2];
          continue;
        }
        v[j] = dq8(b == 8 ? codes[q] : b == 0 ? 0u : packed_code(codes.data(), q, b), {r[0], r[1]}, b);
        ++q;
      }
    }
  } else if (m.vq_bits > 0) {
    if (auto e = get_f16(i, m.vq_codebook); !e) return std::unexpected(e.error());
    const std::size_t S2 = plane_size(h), pb = packed_plane_bytes(S2, m.vq_bits);
    const std::size_t planes = static_cast<std::size_t>(h.bases) * h.grid_t * static_cast<std::size_t>(m.vq_groups());
    std::vector<std::uint8_t> idx(planes * S2), codes(pb);
    for (std::size_t p = 0; p < planes; ++p) {
      i.read(reinterpret_cast<char*>(codes.data()), static_cast<std::streamsize>(codes.size()));
      if (!i) return std::unexpected("truncated file");
      for (std::size_t j = 0; j < S2; ++j) idx[p * S2 + j] = static_cast<std::uint8_t>(m.vq_bits == 8 ? codes[j] : packed_code(codes.data(), j, m.vq_bits));
    }
    vq_apply(m, idx);
  } else if (m.feature_bits < 16) {
    const std::size_t p = plane_size(h);
    std::vector<std::uint8_t> codes(packed_plane_bytes(p, m.feature_bits));
    for (std::size_t off = 0; off < m.features.size(); off += p) {
      std::array<float, 2> r{};
      if (auto e = get_f16(i, r); !e) return std::unexpected(e.error());
      i.read(reinterpret_cast<char*>(codes.data()), static_cast<std::streamsize>(codes.size()));
      if (!i) return std::unexpected("truncated file");
      float* v = m.features.data() + off;
      for (std::size_t j = 0; j < p; ++j) {
        v[j] = dq8(m.feature_bits == 8 ? codes[j] : packed_code(codes.data(), j, m.feature_bits), {r[0], r[1]}, m.feature_bits);
      }
    }
  } else if (auto e = get_f16(i, m.features); !e) {
    return std::unexpected(e.error());
  }
  const auto expect_dense = [&](Dense& slot) -> std::expected<void, std::string> {
    auto d = get_dense(i);
    if (!d) return std::unexpected(d.error());
    if (d->in != slot.in || d->out != slot.out) return std::unexpected("nvfx: layer shape does not match the header");
    slot = std::move(*d);
    return {};
  };
  if (auto e = expect_dense(m.basis); !e) return std::unexpected(e.error());
  for (auto* group : {&m.layers, &m.films}) {
    const auto n = bin::get<std::uint32_t>(i);
    if (!n || *n != group->size()) return std::unexpected("nvfx: layer count does not match the header");
    for (auto& d : *group) {
      if (auto e = expect_dense(d); !e) return std::unexpected(e.error());
    }
  }
  if (auto e = get_f16(i, m.z_mean); !e) return std::unexpected(e.error());
  if (auto e = get_f16(i, m.z_std); !e) return std::unexpected(e.error());
  const auto nz = bin::get<std::uint32_t>(i);
  if (!nz || *nz > 1000000) return std::unexpected("nvfx: bad code count");
  m.z_train.assign(*nz, std::vector<float>(static_cast<std::size_t>(h.n_latent)));
  for (auto& z : m.z_train) {
    if (auto e = get_f16(i, z); !e) return std::unexpected(e.error());
  }
  m.pack_features();  // the runtime keeps the features in their storage format
  return m;
}

}  // namespace nfx
