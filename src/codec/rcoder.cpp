// The residual-plane coder (include/neuralfx/codec/rcoder.hpp): integer context mixing, lpaq-style.
#include <neuralfx/codec/rcoder.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>

namespace nfx::codec {

namespace {

constexpr int kBins = 40;     // decisions per value: zero, sign, 20 magnitude classes, 18 mantissa slots
constexpr int kInputs = ResidualModel::kModels + 1;  // the context models and a bias
constexpr int kActs = 4;      // mixer weight sets per decision: local activity classes
constexpr int kApmCtx = 3;
constexpr int kLimit = 255;   // counters adapt at 1 / (n + 1.5) until n reaches this
constexpr int kMixLr = 6;     // mixer learning rate: w += x * err * kMixLr / 2^14 (about 0.006 in logit units)
constexpr int kApmRate = 6;
constexpr std::size_t kKinds = static_cast<std::size_t>(PlaneKind::count);

// squash(d) = 4096 / (1 + e^-d / 256), integer interpolation of a 33-point table (lpaq1); stretch is its inverse.
int squash(int d) {
  static constexpr int t[33] = {1,    2,    3,    6,    10,   16,   27,   45,   73,   120,  194,
                                310,  488,  747,  1101, 1546, 2047, 2549, 2994, 3348, 3607, 3785,
                                3901, 3975, 4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094};
  if (d > 2047) return 4095;
  if (d < -2047) return 1;
  const int w = d & 127;
  const int i = (d >> 7) + 16;
  return (t[i] * (128 - w) + t[i + 1] * w + 64) >> 7;
}

struct Stretch {
  std::array<std::int16_t, 4096> t{};
  Stretch() {
    int pi = 0;
    for (int x = -2047; x <= 2047; ++x) {
      const int v = squash(x);
      for (int j = pi; j <= v; ++j) t[static_cast<std::size_t>(j)] = static_cast<std::int16_t>(x);
      pi = v + 1;
    }
    for (int j = pi; j < 4096; ++j) t[static_cast<std::size_t>(j)] = 2047;
  }
};
const Stretch kStretch;
int stretch(int p) { return kStretch.t[static_cast<std::size_t>(p)]; }

struct Recip {
  std::array<std::int32_t, kLimit + 1> t{};
  Recip() {
    for (int n = 0; n <= kLimit; ++n) t[static_cast<std::size_t>(n)] = static_cast<std::int32_t>(65536 * 2 / (2 * n + 3));
  }
};
const Recip kRecip;

constexpr std::uint32_t hash2(std::uint32_t a, std::uint32_t b) {
  std::uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u;
  h ^= h >> 15;
  h *= 0xC2B2AE3Du;
  h ^= h >> 13;
  return h;
}
template <class... T>
constexpr std::uint32_t hashv(std::uint32_t h, T... v) {
  ((h = hash2(h, static_cast<std::uint32_t>(v))), ...);
  return h;
}

int clampi(std::int32_t v, int m) { return static_cast<int>(std::clamp(v, -m, m)); }
std::uint32_t uabs(std::int32_t v) { return v < 0 ? static_cast<std::uint32_t>(-static_cast<std::int64_t>(v)) : static_cast<std::uint32_t>(v); }
// Magnitude class of a non-negative sum: 0, 1, 2-3, 4-7, 8-15, 16+.
int qa(std::uint32_t a) { return a == 0 ? 0 : std::min(5, static_cast<int>(std::bit_width(a))); }

}  // namespace

// --- arithmetic coder ----------------------------------------------------------------------------------------------

void ArithEncoder::encode(int bit, int p1) {
  const auto p = static_cast<std::uint32_t>(std::clamp(p1, 1, 4095));
  const std::uint32_t xmid = x1_ + ((x2_ - x1_) >> 12) * p;
  if (bit) x2_ = xmid;
  else x1_ = xmid + 1;
  while (((x1_ ^ x2_) & 0xff000000u) == 0) {
    out_.push_back(static_cast<std::uint8_t>(x2_ >> 24));
    x1_ <<= 8;
    x2_ = (x2_ << 8) | 255u;
  }
}

void ArithEncoder::flush() {
  for (int i = 0; i < 4; ++i) {
    out_.push_back(static_cast<std::uint8_t>(x1_ >> 24));
    x1_ <<= 8;
  }
}

ArithDecoder::ArithDecoder(std::span<const std::uint8_t> in) : in_(in) {
  for (int i = 0; i < 4; ++i) x_ = (x_ << 8) | next();
}

std::uint32_t ArithDecoder::next() {
  const std::size_t i = pos_++;
  return i < in_.size() ? in_[i] : 0u;
}

int ArithDecoder::decode(int p1) {
  const auto p = static_cast<std::uint32_t>(std::clamp(p1, 1, 4095));
  const std::uint32_t xmid = x1_ + ((x2_ - x1_) >> 12) * p;
  const int bit = x_ <= xmid ? 1 : 0;
  if (bit) x2_ = xmid;
  else x1_ = xmid + 1;
  while (((x1_ ^ x2_) & 0xff000000u) == 0) {
    x1_ <<= 8;
    x2_ = (x2_ << 8) | 255u;
    x_ = (x_ << 8) | next();
  }
  return bit;
}

// --- the model -----------------------------------------------------------------------------------------------------

struct ResidualModel::Ctx {
  std::array<std::uint32_t, kModels> h{};
  int act = 0;   // mixer weight set (local activity class)
  int apm = 0;   // APM context
};

ResidualModel::ResidualModel()
    : counters_(static_cast<std::size_t>(kModels) << kTableBits, (1u << 31) & ~1023u),
      weights_(kKinds * kBins * kActs * kInputs, (1 << 16) / 4),
      apm_(kKinds * kBins * kApmCtx * 33) {
  for (std::size_t c = 0; c < kKinds * kBins * kApmCtx; ++c) {
    for (int j = 0; j < 33; ++j) apm_[c * 33 + static_cast<std::size_t>(j)] = static_cast<std::uint16_t>(squash((j - 16) * 128) * 16);
  }
}

std::size_t ResidualModel::memory_bytes() const {
  std::size_t b = counters_.size() * sizeof(std::uint32_t) + weights_.size() * sizeof(std::int32_t) + apm_.size() * sizeof(std::uint16_t);
  for (const auto& p : prev_) b += p.capacity() * sizeof(std::int32_t);
  return b;
}

int ResidualModel::predict(const Ctx& ctx, int bin) {
  const auto k = static_cast<std::size_t>(kind_);
  for (int i = 0; i < kModels; ++i) {
    const std::uint32_t slot = hash2(ctx.h[static_cast<std::size_t>(i)], static_cast<std::uint32_t>(bin)) >> (32 - kTableBits);
    idx_[static_cast<std::size_t>(i)] = (static_cast<std::size_t>(i) << kTableBits) + slot;
    st_[static_cast<std::size_t>(i)] = stretch(static_cast<int>(counters_[idx_[static_cast<std::size_t>(i)]] >> 20));
  }
  st_[kModels] = 256;
  wset_ = ((k * kBins + static_cast<std::size_t>(bin)) * kActs + static_cast<std::size_t>(ctx.act)) * kInputs;
  std::int64_t dot = 0;
  for (std::size_t i = 0; i < kInputs; ++i) dot += static_cast<std::int64_t>(st_[i]) * weights_[wset_ + i];
  const int d = static_cast<int>(std::clamp<std::int64_t>(dot >> 16, -2047, 2047));
  p_mix_ = squash(d);
  const int s = stretch(p_mix_) + 2048;  // 1..4095
  const int lo = s >> 7, w = s & 127;
  apm_idx_ = ((k * kBins + static_cast<std::size_t>(bin)) * kApmCtx + static_cast<std::size_t>(ctx.apm)) * 33 + static_cast<std::size_t>(lo);
  apm_w_ = w;
  const int pa = (apm_[apm_idx_] * (128 - w) + apm_[apm_idx_ + 1] * w) >> 11;
  p_ = std::clamp((p_mix_ + 3 * pa + 2) >> 2, 1, 4095);
  return p_;
}

void ResidualModel::update(int y) {
  bits_[static_cast<std::size_t>(kind_)] -= std::log2(y ? p_ / 4096.0 : 1.0 - p_ / 4096.0);
  for (std::size_t i = 0; i < static_cast<std::size_t>(kModels); ++i) {
    std::uint32_t& c = counters_[idx_[i]];
    const auto n = static_cast<int>(c & 1023u);
    const auto p22 = static_cast<std::int64_t>(c >> 10);
    const std::int64_t target = y ? (1 << 22) - 1 : 0;
    const std::int64_t np = p22 + (((target - p22) * kRecip.t[static_cast<std::size_t>(n)]) >> 16);
    c = (static_cast<std::uint32_t>(std::clamp<std::int64_t>(np, 0, (1 << 22) - 1)) << 10) | static_cast<std::uint32_t>(std::min(n + 1, kLimit));
  }
  const int err = ((y << 12) - p_mix_) * kMixLr;
  for (std::size_t i = 0; i < kInputs; ++i) {
    std::int32_t& w = weights_[wset_ + i];
    w = static_cast<std::int32_t>(std::clamp<std::int64_t>(w + ((static_cast<std::int64_t>(st_[i]) * err) >> 14), -(1 << 22), 1 << 22));
  }
  const std::size_t a = apm_idx_ + (apm_w_ >= 64 ? 1 : 0);
  const int g = y ? 65535 : 0;
  apm_[a] = static_cast<std::uint16_t>(apm_[a] + ((g - apm_[a]) >> kApmRate));
}

template <class Bit>
std::int32_t ResidualModel::code_value(Bit& bit, const Ctx& ctx, std::int32_t v) {
  const int nz = bit(v != 0 ? 1 : 0, predict(ctx, 0));
  update(nz);
  if (!nz) return 0;
  const int neg = bit(v < 0 ? 1 : 0, predict(ctx, 1));
  update(neg);
  const std::uint32_t a = uabs(v);
  const int e = a ? static_cast<int>(std::bit_width(a)) - 1 : 0;
  int ee = 0;
  while (ee < kMaxExp) {
    const int more = bit(e > ee ? 1 : 0, predict(ctx, 2 + ee));
    update(more);
    if (!more) break;
    ++ee;
  }
  std::uint32_t m = 1;
  for (int j = ee - 1; j >= 0; --j) {
    const int bin = 2 + kMaxExp + std::min(ee, 8) * 2 - 2 + (j == ee - 1 ? 0 : 1);
    const int b = bit(static_cast<int>((a >> j) & 1u), predict(ctx, bin));
    update(b);
    m = (m << 1) | static_cast<std::uint32_t>(b);
  }
  const auto mv = static_cast<std::int32_t>(m);
  return neg ? -mv : mv;
}

template <class Bit>
void ResidualModel::code_plane(Bit& bit, PlaneKind kind, int h, int w, int c, std::int32_t* v, std::span<const std::uint8_t> side) {
  kind_ = kind;
  const auto k = static_cast<std::size_t>(kind);
  const bool has_prev = prev_dims_[k * 3] == h && prev_dims_[k * 3 + 1] == w && prev_dims_[k * 3 + 2] == c && !prev_[k].empty();
  const std::int32_t* P = has_prev ? prev_[k].data() : nullptr;
  const auto at = [&](const std::int32_t* f, int y, int x, int ch) -> std::int32_t {
    if (!f || y < 0 || x < 0 || x >= w) return 0;
    return f[(static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * static_cast<std::size_t>(c) + static_cast<std::size_t>(ch)];
  };
  const std::uint32_t kseed = static_cast<std::uint32_t>(kind) * 64u;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int ch = 0; ch < c; ++ch) {
        const std::int32_t L = at(v, y, x - 1, ch), A = at(v, y - 1, x, ch), AL = at(v, y - 1, x - 1, ch), AR = at(v, y - 1, x + 1, ch);
        const std::int32_t LL = at(v, y, x - 2, ch), AA = at(v, y - 2, x, ch);
        const std::int32_t S0 = ch > 0 ? at(v, y, x, ch - 1) : 0, S1 = ch > 1 ? at(v, y, x, ch - 2) : 0;
        std::int32_t Pc = 0;
        std::uint32_t Pa = 0;
        if (P) {
          Pc = at(P, y, x, ch);
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
              if (y + dy < h) Pa += uabs(at(P, y + dy, x + dx, ch));
            }
          }
        }
        const std::uint32_t near = uabs(L) + uabs(A) + uabs(AL) + uabs(AR);
        const auto base = kseed + static_cast<std::uint32_t>(ch) * 8u + 1u;
        Ctx ctx;
        ctx.h[0] = hashv(base);
        ctx.h[1] = hashv(base + 1000u, clampi(L, 3), clampi(A, 3));
        ctx.h[2] = hashv(base + 2000u, qa(near), clampi(Pc, 2), P != nullptr);
        ctx.h[3] = hashv(base + 3000u, clampi(Pc, 4), qa(Pa), P != nullptr);
        ctx.h[4] = hashv(base + 4000u, clampi(S0, 3), clampi(S1, 2), qa(uabs(L) + uabs(A)));
        ctx.h[5] = hashv(base + 5000u, y * 8 / h, x * 8 / w);
        ctx.h[6] = hashv(base + 6000u, clampi(L, 2), clampi(LL, 1), clampi(A, 2), clampi(AA, 1), clampi(AL, 1), clampi(AR, 1));
        const int sd = side.empty() ? 0 : std::min(15, static_cast<int>(side[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)]));
        ctx.h[7] = hashv(base + 7000u, sd, clampi(L, 2), clampi(A, 2));
        ctx.h[8] = hashv(base + 8000u, sd, qa(near), clampi(Pc, 2), clampi(S0, 1));
        ctx.act = std::min(kActs - 1, qa(uabs(L) + uabs(A) + uabs(Pc)));
        ctx.apm = std::min(kApmCtx - 1, qa(uabs(L) + uabs(A)));
        const std::size_t i = (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * static_cast<std::size_t>(c) + static_cast<std::size_t>(ch);
        v[i] = code_value(bit, ctx, v[i]);
      }
    }
  }
  const std::size_t n = static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * static_cast<std::size_t>(c);
  prev_[k].assign(v, v + n);
  prev_dims_[k * 3] = h;
  prev_dims_[k * 3 + 1] = w;
  prev_dims_[k * 3 + 2] = c;
}

void ResidualModel::encode(ArithEncoder& ac, PlaneKind kind, int h, int w, int c, std::span<const std::int32_t> v, std::span<const std::uint8_t> side) {
  std::vector<std::int32_t> tmp(v.begin(), v.end());
  for (std::int32_t& x : tmp) x = std::clamp(x, -kMaxAbs, kMaxAbs);
  auto bit = [&ac](int y, int p) {
    ac.encode(y, p);
    return y;
  };
  code_plane(bit, kind, h, w, c, tmp.data(), side);
}

bool ResidualModel::decode(ArithDecoder& ac, PlaneKind kind, int h, int w, int c, std::span<std::int32_t> v, std::span<const std::uint8_t> side) {
  std::ranges::fill(v, 0);
  auto bit = [&ac](int, int p) { return ac.decode(p); };
  code_plane(bit, kind, h, w, c, v.data(), side);
  return !ac.overrun();
}

}  // namespace nfx::codec
