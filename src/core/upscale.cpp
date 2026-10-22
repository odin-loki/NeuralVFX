// Study P's upscaler (include/neuralfx/upscale.hpp): its layout, the .nvfxup format and the float reference.
#include <neuralfx/binio.hpp>
#include <neuralfx/noise.hpp>
#include <neuralfx/upscale.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <spanstream>
#include <sstream>

namespace nfx::upscale {

namespace {

constexpr std::uint32_t kVersion = 1;
std::size_t sz(int v) { return static_cast<std::size_t>(v); }

// Pixel (x, y) of an n x n RGBA8 frame, channel c, over 255; zero outside the frame (the convolutions' padding).
float px(std::span<const std::uint8_t> lo, int n, int x, int y, int c) {
  if (x < 0 || y < 0 || x >= n || y >= n) return 0.f;
  return static_cast<float>(lo[(sz(y) * sz(n) + sz(x)) * 4 + sz(c)]) * (1.f / 255.f);
}

}  // namespace

Net::Layout Net::layout() const {
  Layout L{};
  std::size_t at = 0;
  const auto take = [&](std::size_t n) {
    const std::size_t a = at;
    at += n;
    return a;
  };
  L.w1 = take(9 * sz(kIn) * sz(c1));
  L.b1 = take(sz(c1));
  L.w2 = take(sz(k2) * sz(k2) * sz(c1) * sz(c2));
  L.b2 = take(sz(c2));
  L.wo = take(sz(c2) * kOut);
  L.bo = take(kOut);
  L.size = at;
  return L;
}

double Net::macs_per_output_pixel() const {
  return (9.0 * kIn * c1 + static_cast<double>(k2) * k2 * c1 * c2 + static_cast<double>(c2) * kOut) / 4.0;
}

Net init_net(int c1, int c2, int k2, std::uint64_t seed) {
  Net n;
  n.c1 = c1;
  n.c2 = c2;
  n.k2 = k2;
  const Net::Layout L = n.layout();
  n.w.assign(L.size, 0.f);
  std::mt19937_64 rng(seed);
  const auto fill = [&](std::size_t at, std::size_t count, double fan_in, double gain) {
    std::normal_distribution<float> d(0.f, static_cast<float>(gain * std::sqrt(2.0 / fan_in)));
    for (std::size_t i = 0; i < count; ++i) n.w[at + i] = d(rng);
  };
  fill(L.w1, L.b1 - L.w1, 9.0 * kIn, 1.0);
  fill(L.w2, L.b2 - L.w2, static_cast<double>(k2) * k2 * c1, 1.0);
  fill(L.wo, L.bo - L.wo, c2, 0.05);
  return n;
}

// --- format ------------------------------------------------------------------------------------------------------------

std::expected<void, std::string> save_net(std::ostream& o, const Net& n) {
  if (n.w.size() != n.layout().size) return std::unexpected("upscaler: weights do not match the layout");
  o.write(kMagic, 8);
  bin::put(o, kVersion);
  bin::put(o, static_cast<std::int32_t>(n.c1));
  bin::put(o, static_cast<std::int32_t>(n.c2));
  bin::put(o, static_cast<std::int32_t>(n.k2));
  bin::put(o, n.gain);
  bin::put(o, n.rate);
  bin::put_str(o, n.tag, 32);
  bin::put(o, static_cast<std::uint32_t>(n.w.size()));
  bin::put_array(o, std::span<const float>(n.w));
  if (!o) return std::unexpected("upscaler: write failed");
  return {};
}

std::expected<void, std::string> save_net(const std::filesystem::path& path, const Net& n) {
  std::ofstream o(path, std::ios::binary);
  if (!o) return std::unexpected("cannot open " + path.string());
  return save_net(o, n);
}

std::expected<Net, std::string> load_net(std::istream& i) {
  char magic[8] = {};
  i.read(magic, 8);
  if (!i || !std::equal(magic, magic + 8, kMagic)) return std::unexpected("not an upscaler file");
  const auto version = bin::get<std::uint32_t>(i);
  if (!version) return std::unexpected(version.error());
  if (*version != kVersion) return std::unexpected("upscaler: unknown version");
  Net n;
  const auto c1 = bin::get<std::int32_t>(i), c2 = bin::get<std::int32_t>(i), k2 = bin::get<std::int32_t>(i);
  const auto gain = bin::get<float>(i), rate = bin::get<float>(i);
  if (!c1 || !c2 || !k2 || !gain || !rate) return std::unexpected("truncated file");
  if (*c1 < 1 || *c1 > 64 || *c2 < 1 || *c2 > 64 || (*k2 != 1 && *k2 != 3)) return std::unexpected("upscaler: sizes out of range");
  if (!std::isfinite(*gain) || *gain < 0.f || !std::isfinite(*rate) || *rate <= 0.f || *rate > 240.f) return std::unexpected("upscaler: bad noise constants");
  n.c1 = *c1;
  n.c2 = *c2;
  n.k2 = *k2;
  n.gain = *gain;
  n.rate = *rate;
  auto tag = bin::get_str(i, 32);
  if (!tag) return std::unexpected(tag.error());
  n.tag = *tag;
  const auto count = bin::get<std::uint32_t>(i);
  if (!count) return std::unexpected(count.error());
  if (*count != n.layout().size) return std::unexpected("upscaler: weight count does not match its sizes");
  n.w.resize(*count);
  if (auto r = bin::get_array(i, std::span<float>(n.w)); !r) return std::unexpected(r.error());
  for (const float v : n.w) {
    if (!std::isfinite(v)) return std::unexpected("upscaler: weights not finite");
  }
  return n;
}

std::expected<Net, std::string> load_net(const std::filesystem::path& path) {
  std::ifstream i(path, std::ios::binary);
  if (!i) return std::unexpected("cannot open " + path.string());
  return load_net(i);
}

std::expected<Net, std::string> parse_net(std::span<const char> bytes) {
  std::ispanstream i(bytes);
  return load_net(i);
}

// --- reference ---------------------------------------------------------------------------------------------------------

void bilinear2x(std::span<const std::uint8_t> lo, int n, std::span<float> up) {
  const int N = 2 * n;
  for (int y = 0; y < N; ++y) {
    const int yi = y / 2, yj = std::clamp(y % 2 ? yi + 1 : yi - 1, 0, n - 1);
    for (int x = 0; x < N; ++x) {
      const int xi = x / 2, xj = std::clamp(x % 2 ? xi + 1 : xi - 1, 0, n - 1);
      for (int c = 0; c < 4; ++c) {
        const float a = 0.75f * px(lo, n, xi, yi, c) + 0.25f * px(lo, n, xj, yi, c);
        const float b = 0.75f * px(lo, n, xi, yj, c) + 0.25f * px(lo, n, xj, yj, c);
        up[(sz(y) * sz(N) + sz(x)) * 4 + sz(c)] = 0.75f * a + 0.25f * b;
      }
    }
  }
}

void network(const Net& net, std::span<const std::uint8_t> lo, int n, std::span<float> out) {
  const Net::Layout L = net.layout();
  const float* w = net.w.data();
  const int c1 = net.c1, c2 = net.c2, k2 = net.k2, r2 = k2 / 2;
  std::vector<float> h1(sz(n) * sz(n) * sz(c1));
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      float* h = h1.data() + (sz(y) * sz(n) + sz(x)) * sz(c1);
      for (int j = 0; j < c1; ++j) h[j] = w[L.b1 + sz(j)];
      for (int t = 0; t < 9; ++t) {
        const int dx = t % 3 - 1, dy = t / 3 - 1;
        for (int i = 0; i < kIn; ++i) {
          const float v = px(lo, n, x + dx, y + dy, i);
          const float* wt = w + L.w1 + (sz(t) * kIn + sz(i)) * sz(c1);
          for (int j = 0; j < c1; ++j) h[j] += wt[j] * v;
        }
      }
      for (int j = 0; j < c1; ++j) h[j] = std::max(0.f, h[j]);
    }
  }
  std::vector<float> h2(sz(c2));
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      for (int j = 0; j < c2; ++j) h2[sz(j)] = w[L.b2 + sz(j)];
      for (int t = 0; t < k2 * k2; ++t) {
        const int xx = x + t % k2 - r2, yy = y + t / k2 - r2;
        if (xx < 0 || yy < 0 || xx >= n || yy >= n) continue;
        const float* h = h1.data() + (sz(yy) * sz(n) + sz(xx)) * sz(c1);
        for (int i = 0; i < c1; ++i) {
          const float* wt = w + L.w2 + (sz(t) * sz(c1) + sz(i)) * sz(c2);
          for (int j = 0; j < c2; ++j) h2[sz(j)] += wt[j] * h[i];
        }
      }
      float* o = out.data() + (sz(y) * sz(n) + sz(x)) * kOut;
      for (int k = 0; k < kOut; ++k) o[k] = w[L.bo + sz(k)];
      for (int i = 0; i < c2; ++i) {
        const float v = std::max(0.f, h2[sz(i)]);
        const float* wt = w + L.wo + sz(i) * kOut;
        for (int k = 0; k < kOut; ++k) o[k] += wt[k] * v;
      }
      for (int k = kResid; k < kOut; ++k) o[k] = std::max(0.f, o[k]);
    }
  }
}

std::uint64_t noise_seed(std::uint64_t seed) { return seed * 0xD1B54A32D192ED03ULL + 0x5EED0F0123456789ULL; }

std::uint32_t noise_key(std::uint64_t seed, std::int64_t k) {
  const std::uint64_t s = noise_seed(seed);
  const std::uint32_t b = hash32(static_cast<std::uint32_t>(s) ^ hash32(static_cast<std::uint32_t>(s >> 32) + 0x9e3779b9U));
  return hash32(b ^ static_cast<std::uint32_t>(k) * 0xcb1ab31fU);
}

float white(std::uint32_t key, std::int32_t x, std::int32_t y) {
  const std::uint32_t row = hash32(key ^ static_cast<std::uint32_t>(y) * 0xd8163841U);
  return static_cast<float>(hash32(row ^ static_cast<std::uint32_t>(x) * 0x8da6b343U) >> 8) * (2.f / 16777215.f) - 1.f;
}

float noise_slice(std::uint64_t seed, std::int32_t x, std::int32_t y, std::int64_t k) {
  const std::uint32_t key = noise_key(seed, k);
  // the 3 x 3 sum as the runtime forms it: three sums along x, then their sum along y
  float sum = 0.f;
  for (int dy = -1; dy <= 1; ++dy) sum += white(key, x - 1, y + dy) + white(key, x, y + dy) + white(key, x + 1, y + dy);
  return (white(key, x, y) - sum * (1.f / 9.f)) * kNoiseNorm;
}

NoiseBlend noise_blend(double seconds, float rate) {
  NoiseBlend b;
  const double u = seconds * static_cast<double>(rate);
  const double f = std::floor(u);
  b.k = static_cast<std::int64_t>(f);
  const float t = static_cast<float>(u - f);
  b.s = t * t * (3.f - 2.f * t);
  b.norm = 1.f / std::sqrt((1.f - b.s) * (1.f - b.s) + b.s * b.s);
  return b;
}

void noise_frame(std::uint64_t seed, double seconds, float rate, int size, std::span<float> eta) {
  const NoiseBlend b = noise_blend(seconds, rate);
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const float a = noise_slice(seed, x, y, b.k), c = noise_slice(seed, x, y, b.k + 1);
      eta[sz(y) * sz(size) + sz(x)] = ((1.f - b.s) * a + b.s * c) * b.norm;
    }
  }
}

void upscale_float(const Net* net, std::span<const std::uint8_t> lo, int n, std::span<const float> eta, std::span<float> out) {
  const int N = 2 * n;
  bilinear2x(lo, n, out);
  if (!net) return;
  std::vector<float> o(sz(n) * sz(n) * kOut);
  network(*net, lo, n, o);
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      float* q = out.data() + (sz(y) * sz(N) + sz(x)) * 4;
      if (q[0] == 0.f && q[1] == 0.f && q[2] == 0.f && q[3] == 0.f) continue;  // empty stays empty
      const float* r = o.data() + (sz(y / 2) * sz(n) + sz(x / 2)) * kOut;
      const int s = (y % 2) * 2 + x % 2;
      const float e = eta.empty() ? 0.f : net->gain * eta[sz(y) * sz(N) + sz(x)];
      for (int c = 0; c < 4; ++c) q[c] += r[s * 4 + c] + r[kResid + c] * e;
    }
  }
}

void upscale(const Net* net, std::span<const std::uint8_t> lo, int n, std::uint64_t seed, double seconds, std::uint8_t* hi, std::size_t stride,
             const float* colour) {
  const int N = 2 * n;
  std::vector<float> out(sz(N) * sz(N) * 4), eta;
  if (net && net->gain > 0.f) {
    eta.resize(sz(N) * sz(N));
    noise_frame(seed, seconds, net->rate, N, eta);
  }
  upscale_float(net, lo, n, eta, out);
  for (int y = 0; y < N; ++y) {
    std::uint8_t* row = hi + stride * sz(y);
    for (int x = 0; x < N; ++x) {
      float c[4];
      for (int k = 0; k < 4; ++k) c[k] = out[(sz(y) * sz(N) + sz(x)) * 4 + sz(k)];
      if (colour) {
        const float r = c[0], g = c[1], b = c[2];
        c[0] = colour[0] * r + colour[1] * g + colour[2] * b;
        c[1] = colour[3] * r + colour[4] * g + colour[5] * b;
        c[2] = colour[6] * r + colour[7] * g + colour[8] * b;
      }
      for (int k = 0; k < 4; ++k) row[sz(x) * 4 + sz(k)] = static_cast<std::uint8_t>(static_cast<int>(std::clamp(c[k], 0.f, 1.f) * 255.f + 0.5f));
    }
  }
}

}  // namespace nfx::upscale
