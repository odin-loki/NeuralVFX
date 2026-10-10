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

namespace nfx {

std::string Hyper::describe() const {
  const std::string cond = std::format("D{}+z{} K{} T{}", n_controls, n_latent, bases, grid_t);
  if (arch == Arch::grid) return std::format("grid G{} C{} H{} L{} {}", grid, channels, hidden, layers, cond);
  return std::format("conv l{} {}-{}-{}-4 {}", latent, c0, c1, c2, cond);
}

std::size_t Model::feature_count() const { return features.size(); }

std::size_t Model::param_count() const {
  std::size_t n = features.size() + basis.params() + z_mean.size() + z_std.size();
  for (const auto& l : layers) n += l.params();
  for (const auto& f : films) n += f.params();
  for (const auto& z : z_train) n += z.size();
  return n;
}

std::size_t Model::storage_bytes() const {
  const std::size_t slices = static_cast<std::size_t>(h.bases) * h.grid_t * h.feature_channels();
  const std::size_t side2 = static_cast<std::size_t>(h.feature_side()) * h.feature_side();
  const std::size_t feat = feature_bits < 16 ? slices * (packed_plane_bytes(side2, feature_bits) + 4) : features.size() * 2;
  return feat + (param_count() - features.size()) * 2;
}

double Model::macs_per_pixel(int out) const {
  const double px = static_cast<double>(out) * out;
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
  if (h.frames < 2 || h.bases < 1 || h.grid_t < 2 || h.n_controls < 0 || h.n_latent < 0 || h.dims() > 64) bad("bad shape");
  if (h.arch == Arch::grid) {
    if (h.grid < 2 || h.channels < 1 || h.hidden < 1 || h.layers < 1 || h.size < 4) bad("bad grid shape");
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

}  // namespace

Model init_model(const Hyper& h, std::uint64_t seed) {
  validate(h);
  Model m;
  m.h = h;
  std::mt19937_64 rng(seed);
  const int D = h.dims();
  const std::size_t plane = static_cast<std::size_t>(h.feature_channels()) * h.feature_side() * h.feature_side();
  m.features.resize(static_cast<std::size_t>(h.bases) * h.grid_t * plane);
  std::uniform_real_distribution<float> fu(-0.1f, 0.1f);
  for (float& v : m.features) v = fu(rng);
  m.basis = Dense(D, h.bases);
  std::uniform_real_distribution<float> bu(-0.5f, 0.5f);
  for (float& w : m.basis.w) w = bu(rng) / std::sqrt(static_cast<float>(std::max(1, D)));
  std::ranges::fill(m.basis.b, 1.f);
  if (h.arch == Arch::grid) {
    int in = h.channels;
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
  const long qm = qmax_of(bits);
  return static_cast<std::uint8_t>(std::clamp(std::lround((v - r.lo) / (r.hi - r.lo) * static_cast<float>(qm)), 0L, qm));
}
float dq8(unsigned q, Plane8 r, int bits = 8) { return r.lo + static_cast<float>(q) / static_cast<float>(qmax_of(bits)) * (r.hi - r.lo); }

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

void Model::pack_features() {
  raw_f16.clear();
  raw_u8.clear();
  raw_ranges.clear();
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
  if (m.feature_bits < 16) {
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

std::expected<void, std::string> save_model(std::ostream& o, const Model& m) {
  const Hyper& h = m.h;
  o.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  bin::put(o, kVersion);
  bin::put(o, static_cast<std::uint32_t>(h.arch));
  for (const int v : {h.size, h.frames, static_cast<int>(h.loop), h.n_controls, h.n_latent, h.bases, h.grid_t, h.grid,
                      h.channels, h.hidden, h.layers, h.latent, h.c0, h.c1, h.c2}) {
    bin::put(o, static_cast<std::int32_t>(v));
  }
  bin::put_str(o, m.effect, 32);
  bin::put(o, m.fps);
  bin::put(o, static_cast<std::uint32_t>(m.feature_bits));
  for (int k = 0; k < h.n_controls; ++k) {
    bin::put_str(o, static_cast<std::size_t>(k) < m.control_names.size() ? m.control_names[static_cast<std::size_t>(k)] : std::string{}, 16);
  }
  if (!valid_feature_bits(m.feature_bits)) return std::unexpected("nvfx: feature bits must be 2 to 8 or 16");
  if (m.feature_bits < 16) {
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
  if (!version || *version != kVersion) return std::unexpected("nvfx: unsupported version");
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
  for (int k = 0; k < h.n_controls; ++k) {
    auto name = bin::get_str(i, 16);
    if (!name) return std::unexpected(name.error());
    m.control_names.push_back(*name);
  }
  m.feature_bits = static_cast<int>(*bits);
  if (m.feature_bits < 16) {
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
