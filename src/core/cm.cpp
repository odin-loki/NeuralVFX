// The context-mixing coder (include/neuralfx/cm.hpp): logistic tables, adaptive counters, the arithmetic coder, the
// bit model (hashed and direct context models, two mixers, two APMs), the tensor coder that turns a tensor's
// structure into contexts, and the containers that parse .nvfx files into tensors.
//
// Determinism: every quantity the decoder depends on is computed with integer arithmetic (shifts of negative values
// are arithmetic in C++20 and later), from the same inputs in the same order as the encoder. Floating point appears
// only in the encoder's cost report and in parsing a file to decide whether it is a model at all.
#include <neuralfx/cm.hpp>
#include <neuralfx/model.hpp>
#include <neuralfx/rollout.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>
#include <spanstream>
#include <stdexcept>

namespace nfx::cm {

std::size_t Shape::size() const {
  if (dims.empty()) return 0;
  std::size_t n = 1;
  for (const std::uint32_t d : dims) n *= d;
  return n;
}

std::size_t Shape::planes() const {
  const std::size_t inner = channels ? 3 : 2;
  std::size_t n = 1;
  for (std::size_t i = 0; i + inner < dims.size(); ++i) n *= dims[i];
  return n;
}

std::uint16_t f16_order(std::uint16_t b) {
  return static_cast<std::uint16_t>((b & 0x8000) ? 0x7fff - (b & 0x7fff) : 0x8000 + b);
}

std::uint16_t f16_unorder(std::uint16_t u) {
  return static_cast<std::uint16_t>(u >= 0x8000 ? u - 0x8000 : 0x8000 | (0x7fff - u));
}

const char* kind_name(Kind k) {
  static constexpr std::array<const char*, static_cast<std::size_t>(Kind::count)> names = {
      "header", "features", "feature ranges", "weights", "biases", "codes", "coarse states", "fine fields", "field scales",
      "bc3 blocks", "rgba pixels", "motion vectors"};
  const auto i = static_cast<std::size_t>(k);
  return i < names.size() ? names[i] : "?";
}

namespace {

using std::int64_t;
using std::size_t;
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;
using std::uint8_t;

constexpr int kKinds = static_cast<int>(Kind::count);

// --- logistic functions in fixed point ------------------------------------------------------------------------------

// squash(d) = 4096 / (1 + exp(-d / 256)), interpolated in a fixed table: the same integers everywhere.
constexpr int squash(int d) {
  constexpr int t[33] = {1,    2,    3,    6,    10,   16,   27,   45,   73,   120,  194,  310,  488,  747,  1101, 1546, 2047,
                                2549, 2994, 3348, 3607, 3785, 3901, 3975, 4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094};
  if (d > 2047) return 4095;
  if (d < -2047) return 1;
  const int w = d & 127;
  const int i = (d >> 7) + 16;
  return (t[i] * (128 - w) + t[i + 1] * w + 64) >> 7;
}

// stretch(p) = ln(p / (1 - p)) * 256, the inverse of squash, tabulated by inverting it.
struct StretchTable {
  std::array<std::int16_t, 4096> t{};
  constexpr StretchTable() {
    int pi = 0;
    for (int x = -2047; x <= 2047; ++x) {
      const int v = squash(x);
      for (int i = pi; i <= v; ++i) t[static_cast<size_t>(i)] = static_cast<std::int16_t>(x);
      pi = v + 1;
    }
    for (int i = pi; i < 4096; ++i) t[static_cast<size_t>(i)] = 2047;
  }
};
constexpr StretchTable kStretch;
inline int stretch(int p) { return kStretch.t[static_cast<size_t>(p)]; }

// --- adaptive counters ----------------------------------------------------------------------------------------------

// A probability (22 bits) and a count (10 bits) in one word. The probability moves towards each bit by 1 / (n + 1.5),
// so a new context learns fast and a well-seen one averages, up to a count limit that sets how far back it remembers.
constexpr uint32_t kFresh = 1u << 31;  // p = 1/2, n = 0

struct RateTable {
  std::array<int, 1024> r{};
  constexpr RateTable() {
    for (int n = 0; n < 1024; ++n) r[static_cast<size_t>(n)] = 131072 / (2 * n + 3);
  }
};
constexpr RateTable kRate;

inline void train(uint32_t& v, int bit, uint32_t limit) {
  const uint32_t n = v & 1023;
  const int64_t p = v >> 10;
  const int64_t target = bit ? (1 << 22) - 1 : 0;
  const int64_t np = p + (((target - p) * kRate.r[n]) >> 16);
  v = (static_cast<uint32_t>(np) << 10) | (n < limit ? n + 1 : n);
}

inline int counter_p12(uint32_t v) { return static_cast<int>(v >> 20); }

// --- hashing --------------------------------------------------------------------------------------------------------

constexpr uint32_t mix(uint32_t h, uint32_t v) {
  h = (h + v * 0x9E3779B1u + 0x7F4A7C15u) * 0x85EBCA77u;
  h ^= h >> 15;
  h *= 0xC2B2AE3Du;
  return h ^ (h >> 13);
}
template <class... T>
constexpr uint32_t mixes(uint32_t h, T... v) {
  ((h = mix(h, static_cast<uint32_t>(v))), ...);
  return h;
}

// --- arithmetic coder -----------------------------------------------------------------------------------------------

// Carry-less binary arithmetic coder over a 32-bit range (as in lpaq), with 16-bit probabilities.
class Encoder {
 public:
  static constexpr bool encoding = true;
  explicit Encoder(std::vector<uint8_t>& out) : out_(out) {}
  int code(int p16, int bit) {
    const uint32_t xmid = x1_ + static_cast<uint32_t>((static_cast<uint64_t>(x2_ - x1_) * static_cast<uint32_t>(p16)) >> 16);
    if (bit) {
      x2_ = xmid;
      cost += cost_of(p16);
    } else {
      x1_ = xmid + 1;
      cost += cost_of(65536 - p16);
    }
    while (((x1_ ^ x2_) & 0xff000000u) == 0) {
      out_.push_back(static_cast<uint8_t>(x2_ >> 24));
      x1_ <<= 8;
      x2_ = (x2_ << 8) | 255;
    }
    return bit;
  }
  // n bits of v at even odds, most significant first: the same as n calls of code(32768, bit), with the state kept in
  // registers.
  uint32_t plain(uint32_t v, int n) {
    uint32_t x1 = x1_, x2 = x2_;
    for (int b = n - 1; b >= 0; --b) {
      const uint32_t xmid = x1 + ((x2 - x1) >> 1);
      if ((v >> b) & 1u) x2 = xmid;
      else x1 = xmid + 1;
      while (((x1 ^ x2) & 0xff000000u) == 0) {
        out_.push_back(static_cast<uint8_t>(x2 >> 24));
        x1 <<= 8;
        x2 = (x2 << 8) | 255;
      }
    }
    x1_ = x1;
    x2_ = x2;
    cost += n;
    return v;
  }
  void flush() {
    for (int i = 0; i < 4; ++i) {
      out_.push_back(static_cast<uint8_t>(x1_ >> 24));
      x1_ <<= 8;
    }
  }
  double cost = 0;  // ideal code length so far, in bits (for reports only; never read by the model)
  bool overrun() const { return false; }

 private:
  static double cost_of(int p) {
    static const std::array<float, 4097> t = [] {
      std::array<float, 4097> c{};
      for (int i = 1; i <= 4096; ++i) c[static_cast<size_t>(i)] = static_cast<float>(-std::log2((static_cast<double>(i) - 0.5) / 4096.0));
      c[0] = c[1];
      return c;
    }();
    return t[static_cast<size_t>((p + 8) >> 4)];
  }
  std::vector<uint8_t>& out_;
  uint32_t x1_ = 0, x2_ = 0xffffffffu;
};

class Decoder {
 public:
  static constexpr bool encoding = false;
  explicit Decoder(std::span<const uint8_t> in) : in_(in) {
    for (int i = 0; i < 4; ++i) x_ = (x_ << 8) | next();
  }
  int code(int p16, int /*bit*/) {
    const uint32_t xmid = x1_ + static_cast<uint32_t>((static_cast<uint64_t>(x2_ - x1_) * static_cast<uint32_t>(p16)) >> 16);
    const int bit = x_ <= xmid ? 1 : 0;
    if (bit) x2_ = xmid;
    else x1_ = xmid + 1;
    while (((x1_ ^ x2_) & 0xff000000u) == 0) {
      x1_ <<= 8;
      x2_ = (x2_ << 8) | 255;
      x_ = (x_ << 8) | next();
    }
    return bit;
  }
  uint32_t plain(uint32_t /*v*/, int n) {
    uint32_t x1 = x1_, x2 = x2_, x = x_, r = 0;
    for (int b = 0; b < n; ++b) {
      const uint32_t xmid = x1 + ((x2 - x1) >> 1);
      const bool bit = x <= xmid;
      r = (r << 1) | (bit ? 1u : 0u);
      x2 = bit ? xmid : x2;
      x1 = bit ? x1 : xmid + 1;
      while (((x1 ^ x2) & 0xff000000u) == 0) {
        x1 <<= 8;
        x2 = (x2 << 8) | 255;
        x = (x << 8) | next();
      }
    }
    x1_ = x1;
    x2_ = x2;
    x_ = x;
    return r;
  }
  double cost = 0;
  // A valid stream is read exactly to its end (the decoder shifts in one byte for each byte the encoder shifted out,
  // after the four it starts with), so reading past it means the data is damaged: decoding stops there.
  bool overrun() const { return pos_ > in_.size(); }

 private:
  uint32_t next() {
    const uint32_t b = pos_ < in_.size() ? in_[pos_] : 0u;
    if (pos_ <= in_.size()) ++pos_;
    return b;
  }
  std::span<const uint8_t> in_;
  size_t pos_ = 0;
  uint32_t x1_ = 0, x2_ = 0xffffffffu, x_ = 0;
};

// --- the bit model ----------------------------------------------------------------------------------------------------

constexpr int kHashed = 14;               // context models looked up by hash, per byte
constexpr int kDirect = 3;                // context models indexed directly, per bit
constexpr int kInputs = kHashed + kDirect + 1;  // plus a bias input
constexpr int kBitIdx = 24;               // bit indices: 0..7 for 8-bit values, 8..23 for fp16 values
constexpr int kRel = 49;                  // buckets of the prediction's offset from the middle of the interval left
constexpr int kAct = 16;                  // local error levels
constexpr int kSig = 64;                  // buckets of the offset in units of the local error

// Hashed statistics live in slots of one cache line: a check word and the 15 counters of one nibble's binary tree.
struct alignas(64) Slot {
  uint32_t check;
  std::array<uint32_t, 15> c;
};

// Logistic mixing: 16-bit weights (14 fractional bits) per selected set, trained online on the coding error. Inputs
// and weights are 16-bit and the sums 32-bit, which compilers turn into vector multiply-adds.
constexpr int kLanes = 24;  // inputs padded to a multiple of 8
static_assert(kInputs <= kLanes);

class Mixer {
 public:
  Mixer(size_t sets, int lr, int init) : w_(sets * kLanes, static_cast<std::int16_t>(init)), lr_(lr) {}
  int dot(const std::int16_t* x, size_t set) {
    sel_ = set * kLanes;
    const std::int16_t* w = w_.data() + sel_;
    int s = 0;  // at most 24 * 2^11 * 2^15 < 2^31
    for (int i = 0; i < kLanes; ++i) s += x[i] * w[i];
    out_ = std::clamp(s >> 14, -2047, 2047);
    p_ = squash(out_);
    return out_;
  }
  // The step x * err / 2^16 is below 2^10 in size (|x| < 2^11, |err| < 2^12 * 6), so with the weights kept inside
  // +-(2^15 - 2^10) the sum never overflows 16 bits: multiply-high, add and min/max on 16-bit lanes.
  void update(const std::int16_t* x, int bit) {
    const auto err = static_cast<std::int16_t>(((bit << 12) - p_) * lr_);
    std::int16_t* w = w_.data() + sel_;
    for (int i = 0; i < kInputs; ++i) {  // the padding lanes are zero and stay so
      const auto step = static_cast<std::int16_t>((x[i] * err + 32768) >> 16);
      const auto v = static_cast<std::int16_t>(w[i] + step);
      w[i] = std::min<std::int16_t>(std::max<std::int16_t>(v, -kWeightMax), kWeightMax);
    }
  }

 private:
  static constexpr std::int16_t kWeightMax = 32767 - 1024;
  std::vector<std::int16_t> w_;
  size_t sel_ = 0;
  int out_ = 0, p_ = 2048, lr_;
};

// Adaptive probability map: a probability per context and per stretch(p) bucket (33, interpolated), trained towards
// the coded bit, which corrects the mixer's probability where it is systematically off.
class Apm {
 public:
  Apm(size_t contexts, int rate) : t_(contexts * 33), rate_(rate) {
    for (size_t i = 0; i < contexts; ++i) {
      for (int j = 0; j < 33; ++j) t_[i * 33 + static_cast<size_t>(j)] = static_cast<uint16_t>(squash((j - 16) * 128) * 16);
    }
  }
  int p(int st, size_t cx) {
    const int pos = st + 2048;
    idx_ = cx * 33 + static_cast<size_t>(pos >> 7);
    w_ = pos & 127;
    return (t_[idx_] * (128 - w_) + t_[idx_ + 1] * w_) >> 7;
  }
  void update(int bit) {
    const int target = bit ? 65535 : 0;
    t_[idx_] = static_cast<uint16_t>(t_[idx_] + (((target - t_[idx_]) * (128 - w_)) >> (rate_ + 7)));
    t_[idx_ + 1] = static_cast<uint16_t>(t_[idx_ + 1] + (((target - t_[idx_ + 1]) * w_) >> (rate_ + 7)));
  }

 private:
  std::vector<uint16_t> t_;
  size_t idx_ = 0;
  int w_ = 0, rate_;
};

// Selectors and direct contexts of one bit.
struct BitSel {
  size_t direct[kDirect];
  size_t mix1, mix2, apm1, apm2;
};

class Model {
 public:
  static constexpr size_t kMix1 = static_cast<size_t>(kKinds) * kBitIdx * 4;
  static constexpr size_t kMix2 = static_cast<size_t>(kKinds) * 3 * 256;
  static constexpr size_t kRelN = static_cast<size_t>(kKinds) * kBitIdx * kRel * kAct;
  static constexpr size_t kSigN = static_cast<size_t>(kKinds) * kBitIdx * kSig;

  explicit Model(size_t total_bytes)
      : bits_(std::clamp(static_cast<int>(std::bit_width(total_bytes)) + 1, 12, 20)),
        table_(size_t{1} << bits_),
        mask_((uint32_t{1} << bits_) - 1),
        m1_(kMix1, 6, 1 << 10),
        m2_(kMix2, 6, 1 << 10),
        a1_(kMix2, 7),
        a2_(static_cast<size_t>(kKinds) * kBitIdx * kRel, 7) {
    direct_[0].assign(kRelN, kFresh);
    direct_[1].assign(kSigN, kFresh);
    direct_[2].assign(kRelN, kFresh);
  }

  uint32_t next_serial() { return ++serial_; }

  // Start a byte: its hashed contexts.
  void byte(const uint32_t* ctx) {
    for (int i = 0; i < kHashed; ++i) ctx_[i] = ctx[i] + static_cast<uint32_t>(i) * 0x3C6EF372u;
    lookup(0);
    nib_ = 1;
    bits_in_nib_ = 0;
  }

  // P(next bit = 1) in 16 bits.
  int predict(const BitSel& s) {
    for (int i = 0; i < kHashed; ++i) {
      cnt_[i] = &slot_[i]->c[static_cast<size_t>(nib_ - 1)];
      x_[static_cast<size_t>(i)] = static_cast<std::int16_t>(stretch(counter_p12(*cnt_[i])));
    }
    for (int j = 0; j < kDirect; ++j) {
      dcnt_[j] = &direct_[j][s.direct[j]];
      x_[static_cast<size_t>(kHashed + j)] = static_cast<std::int16_t>(stretch(counter_p12(*dcnt_[j])));
    }
    x_[kInputs - 1] = 256;
    const int d1 = m1_.dot(x_.data(), s.mix1), d2 = m2_.dot(x_.data(), s.mix2);
    const int st = (d1 + d2) >> 1;
    const int pm = squash(st) * 16;
    const int p1 = a1_.p(st, s.apm1), p2 = a2_.p(st, s.apm2);
    return std::clamp((pm + p1 + 2 * p2) >> 2, 32, 65536 - 32);
  }

  void update(int bit) {
    for (int i = 0; i < kHashed; ++i) train(*cnt_[i], bit, 255);
    for (int j = 0; j < kDirect; ++j) train(*dcnt_[j], bit, 1023);
    m1_.update(x_.data(), bit);
    m2_.update(x_.data(), bit);
    a1_.update(bit);
    a2_.update(bit);
    nib_ = nib_ * 2 + bit;
    if (++bits_in_nib_ == 4) {
      lookup(static_cast<uint32_t>(nib_));
      nib_ = 1;
      bits_in_nib_ = 0;
    }
  }

 private:
  // The slots of every context for the next nibble: all addresses first (fetched together), then the lookups.
  void lookup(uint32_t nibble) {
    std::array<uint32_t, kHashed> h;
    for (int i = 0; i < kHashed; ++i) {
      h[static_cast<size_t>(i)] = mix(ctx_[static_cast<size_t>(i)], nibble);
#if defined(__GNUC__) || defined(__clang__)
      const uint32_t at = (h[static_cast<size_t>(i)] >> 7) & mask_;
      __builtin_prefetch(&table_[at]);
      __builtin_prefetch(&table_[at ^ 1u]);
#endif
    }
    for (int i = 0; i < kHashed; ++i) slot_[static_cast<size_t>(i)] = find(h[static_cast<size_t>(i)]);
  }

  // Two-way associative lookup; a miss takes over the less used of the two slots.
  Slot* find(uint32_t h) {
    const uint32_t check = h | 1u;
    const uint32_t i = (h >> 7) & mask_;
    Slot* a = &table_[i];
    if (a->check == check) return a;
    Slot* b = &table_[i ^ 1u];
    if (b->check == check) return b;
    Slot* r = (a->c[0] & 1023) <= (b->c[0] & 1023) ? a : b;
    r->check = check;
    r->c.fill(kFresh);
    return r;
  }

  int bits_;
  std::vector<Slot> table_;
  uint32_t mask_;
  std::array<std::vector<uint32_t>, kDirect> direct_;
  Mixer m1_, m2_;
  Apm a1_, a2_;
  std::array<uint32_t, kHashed> ctx_{};
  std::array<Slot*, kHashed> slot_{};
  std::array<uint32_t*, kHashed> cnt_{};
  std::array<uint32_t*, kDirect> dcnt_{};
  std::array<std::int16_t, kLanes> x_{};
  int nib_ = 1, bits_in_nib_ = 0;
  uint32_t serial_ = 0;
};

// --- numeric values ---------------------------------------------------------------------------------------------------

// fp16 bit pattern -> value * 2^24, exactly (infinities and NaNs as the largest finite value).
int64_t f16_lin(uint16_t b) {
  int e = (b >> 10) & 31;
  int64_t m = b & 1023;
  if (e == 31) {
    e = 30;
    m = 1023;
  }
  const int64_t mag = e == 0 ? m : (1024 + m) << (e - 1);
  return (b & 0x8000) ? -mag : mag;
}

// value * 2^24 -> the order code of the nearest fp16.
uint16_t lin_code(int64_t v) {
  const bool neg = v < 0;
  const uint64_t mag = neg ? static_cast<uint64_t>(-v) : static_cast<uint64_t>(v);
  uint32_t bits = 0;
  if (mag < 1024) {
    bits = static_cast<uint32_t>(mag);
  } else {
    int e = static_cast<int>(std::bit_width(mag)) - 10;
    const int shift = e - 1;
    uint64_t mm = shift > 0 ? (mag + (uint64_t{1} << (shift - 1))) >> shift : mag;
    if (mm >= 2048) {
      mm >>= 1;
      ++e;
    }
    if (e >= 31) {
      e = 30;
      mm = 2047;
    }
    bits = (static_cast<uint32_t>(e) << 10) | static_cast<uint32_t>(mm - 1024);
  }
  if (neg && bits) bits |= 0x8000u;
  return f16_order(static_cast<uint16_t>(bits));
}

constexpr int64_t kLinLimit = int64_t{1} << 42;  // plane maps beyond fp16's range are not used

// --- the tensor coder -------------------------------------------------------------------------------------------------

constexpr int kPred = 14;  // numeric predictors blended per value
constexpr int kLms = 12;   // inputs of the adaptive linear predictor

struct Geometry {
  uint32_t C = 1, X = 1, Y = 1;  // channels, columns, rows
  size_t plane = 1, planes = 1;  // values per plane, planes
  uint32_t D1 = 1, D2 = 1;       // sizes of the innermost two plane axes (1 when absent)
};

Geometry geometry(const Shape& s) {
  Geometry g;
  const auto& d = s.dims;
  const size_t n = d.size();
  size_t at = n;
  if (s.channels && at > 0) g.C = d[--at];
  if (at > 0) g.X = d[--at];
  if (at > 0) g.Y = d[--at];
  if (at > 0) g.D1 = d[at - 1];
  if (at > 1) g.D2 = d[at - 2];
  g.plane = static_cast<size_t>(g.C) * g.X * g.Y;
  g.planes = s.planes();
  return g;
}

// Maps an 8-bit neighbour from plane n into plane c's scale: lin_c = (A + lin_n * B) / D, all in 1/256 steps.
struct Remap {
  bool on = false;
  int64_t A = 0, B = 0, D = 1;
  int64_t operator()(int64_t lin) const {
    if (!on) return lin;
    return std::clamp<int64_t>((A + lin * B) / D, -2 * 65280, 3 * 65280);
  }
};

Remap make_remap(const Shape& s, size_t n, size_t c) {
  Remap r;
  if (s.width != 1 || s.lo.size() != s.planes() || s.hi.size() != s.planes()) return r;
  const int64_t lo_n = s.lo[n], hi_n = s.hi[n], lo_c = s.lo[c], hi_c = s.hi[c];
  for (const int64_t v : {lo_n, hi_n, lo_c, hi_c}) {
    if (v < -kLinLimit || v > kLinLimit) return r;
  }
  if (hi_c <= lo_c || hi_n < lo_n) return r;
  r.on = true;
  r.A = (lo_n - lo_c) * 65280;
  r.B = hi_n - lo_n;
  r.D = hi_c - lo_c;
  return r;
}

int ilog2(uint64_t v) { return v ? static_cast<int>(std::bit_width(v)) - 1 : -1; }

// --- LZ tokens (format 2) ---------------------------------------------------------------------------------------------

// LZ tokens in the manner of LZP: a hash of the kLzOrder values before a position names the last position that had the
// same values before it, and the value that followed there is offered. One flag, coded with an adaptive probability
// (by kind, by how long the match has held, and by whether the offered value is the numeric prediction), says whether
// it is the value; if so, none of its bits are coded. A match is followed while it holds, so a long repeat costs one
// cheap flag per value and no modelling at all. Within one tensor (or one segment of it), in coding order.
constexpr int kLzOrder = 4;
constexpr int kLzLen = 16;

struct LzState {
  std::vector<uint32_t> table;  // context hash -> position + 1 of the value that followed that context
  uint32_t mask = 0;
  size_t ptr = 0;  // position + 1 of the value the current match offers next (0: no match)
  int len = 0;     // values matched so far in the current match
  std::array<uint32_t, static_cast<size_t>(kKinds) * kLzLen * 2> flag;
  LzState() { flag.fill(kFresh); }
  void start(size_t n) {
    const int bits = std::clamp(static_cast<int>(std::bit_width(n)) + 1, 8, 16);  // at most 256 KB: stays in cache
    table.assign(size_t{1} << bits, 0);
    mask = (uint32_t{1} << bits) - 1;
    ptr = 0;
    len = 0;
  }
};

// The token at position i (codes of positions before i known). Returns true when the value was coded as a match (and
// then sets sym); otherwise the caller codes the value's bits.
template <class AC>
bool lz_token(LzState& z, AC& ac, const uint16_t* code, size_t i, uint32_t group, uint32_t pcode, uint32_t& sym) {
  uint32_t h = 0;
  const bool ctx = i >= kLzOrder;
  if (ctx) {
    h = mixes(0x2545F491u, code[i - 1], code[i - 2], code[i - 3], code[i - 4]) & z.mask;
    if (z.ptr == 0) {
      // a candidate only when its kLzOrder values before really are these (not a collision of the hash)
      const size_t c = z.table[h];
      if (c > kLzOrder && code[c - 2] == code[i - 1] && code[c - 3] == code[i - 2] && code[c - 4] == code[i - 3] && code[c - 5] == code[i - 4]) z.ptr = c;
      z.len = 0;
    }
  }
  bool hit = false;
  if (z.ptr != 0) {
    const uint32_t want = code[z.ptr - 1];
    uint32_t& c = z.flag[(static_cast<size_t>(group) * kLzLen + static_cast<size_t>(std::min(z.len, kLzLen - 1))) * 2 + (want == pcode ? 1 : 0)];
    const int p16 = std::clamp(static_cast<int>(c >> 16), 32, 65536 - 32);
    const int bit = ac.code(p16, AC::encoding ? (sym == want ? 1 : 0) : 0);
    train(c, bit, 255);
    if (bit) {
      sym = want;
      hit = true;
      ++z.ptr;
      ++z.len;
    } else {
      z.ptr = 0;
      z.len = 0;
    }
  }
  if (ctx) z.table[h] = static_cast<uint32_t>(i + 1);
  return hit;
}

// --- the light model (format 2) ---------------------------------------------------------------------------------------

// The fast literal coder: per bit, four directly indexed statistics (the three of the full model, relative to the
// numeric prediction, and the bits of the byte so far), one small mixer chosen by bit position and error level, and one
// APM. No hashed contexts. `groups` is 1 for a segment (one kind of tensor) or every kind for a whole stream.
class Light {
 public:
  static constexpr int kIn = 4, kLanes = 8;
  explicit Light(size_t groups)
      : groups_(groups),
        d0_(groups * kBitIdx * kRel * kAct, kFresh),
        d1_(groups * kBitIdx * kSig, kFresh),
        d2_(groups * kBitIdx * kRel * kAct, kFresh),
        d3_(groups * 3 * 256, kFresh),
        w_(groups * kBitIdx * 4 * kLanes, 0),
        apm_(groups * kBitIdx * kRel, 7) {
    for (size_t s = 0; s < w_.size(); s += kLanes) {
      for (int k = 0; k < kIn; ++k) w_[s + static_cast<size_t>(k)] = static_cast<std::int16_t>((1 << 14) / kIn);
    }
  }
  size_t groups() const { return groups_; }

  int predict(size_t i0, size_t i1, size_t i2, size_t i3, size_t mix, size_t apm) {
    c_[0] = &d0_[i0];
    c_[1] = &d1_[i1];
    c_[2] = &d2_[i2];
    c_[3] = &d3_[i3];
    for (int k = 0; k < kIn; ++k) x_[static_cast<size_t>(k)] = static_cast<std::int16_t>(stretch(counter_p12(*c_[static_cast<size_t>(k)])));
    x_[kIn] = 256;
    sel_ = mix * kLanes;
    const std::int16_t* w = w_.data() + sel_;
    int s = 0;
    for (int k = 0; k < kLanes; ++k) s += x_[static_cast<size_t>(k)] * w[k];
    const int st = std::clamp(s >> 14, -2047, 2047);
    p_ = squash(st);
    const int pa = apm_.p(st, apm);
    return std::clamp((p_ * 16 + 3 * pa) >> 2, 32, 65536 - 32);
  }

  void update(int bit) {
    for (int k = 0; k < kIn; ++k) train(*c_[static_cast<size_t>(k)], bit, 1023);
    const auto err = static_cast<std::int16_t>(((bit << 12) - p_) * 6);
    std::int16_t* w = w_.data() + sel_;
    for (int k = 0; k <= kIn; ++k) {
      const auto step = static_cast<std::int16_t>((x_[static_cast<size_t>(k)] * err + 32768) >> 16);
      w[k] = std::clamp<std::int16_t>(static_cast<std::int16_t>(w[k] + step), -kWeightMax, kWeightMax);
    }
    apm_.update(bit);
  }

 private:
  static constexpr std::int16_t kWeightMax = 32767 - 1024;
  size_t groups_;
  std::vector<uint32_t> d0_, d1_, d2_, d3_;
  std::vector<std::int16_t> w_;
  Apm apm_;
  std::array<uint32_t*, kIn> c_{};
  std::array<std::int16_t, kLanes> x_{};
  size_t sel_ = 0;
  int p_ = 2048;
};

template <class AC>
bool code_tensor(Model& model, AC& ac, const Shape& s, uint16_t* values, LzState* lz = nullptr) {
  const size_t n = s.size();
  if (n == 0) return true;
  const Geometry g = geometry(s);
  const int width = s.width;
  const bool f16 = width == 2;
  const uint32_t kind = static_cast<uint32_t>(s.kind);
  const bool bc3 = s.kind == Kind::bc3;
  const uint32_t serial = model.next_serial();  // the statistics of one tensor kept apart
  // Mixers, maps and direct contexts are shared by kinds of small fp16 tensors (they learn from few values).
  uint32_t group = kind;
  if (s.kind == Kind::biases || s.kind == Kind::codes || s.kind == Kind::scales) group = static_cast<uint32_t>(Kind::weights);
  const int err_shift = f16 ? 8 : 4;               // errors kept in code units (fp16) or 1/16 code units (8-bit)
  const int64_t code_max = f16 ? 65535 * 256 : 255 * 256;
  const int64_t lin_default = f16 ? 0 : 128 * 256;
  std::vector<uint16_t> code(n);                    // order codes (fp16) or values (8-bit) known so far
  std::vector<int64_t> lin(n);                      // values in their plane's units (8-bit: q * 256) or * 2^24 (fp16)
  // Errors of each predictor and of the blend, for the current and (when there are planes) the previous plane.
  const size_t ring_plane = g.plane * (kPred + 1);
  const size_t rings = g.D1 > 1 ? 2 : 1;
  std::vector<uint16_t> errs(rings * ring_plane, 0);
  const size_t CX = static_cast<size_t>(g.C) * g.X;
  const size_t stride1 = g.plane, stride2 = g.plane * g.D1;

  const int lms_shift = f16 ? 8 : 4;
  constexpr int64_t lms_mu = 512;  // NLMS step, 1/128
  std::array<int64_t, kLms> lw{};  // weights of the adaptive linear predictor, 16 fractional bits
  std::array<uint32_t, kHashed> ctx{};
  if (lz) lz->start(n);
  for (size_t p = 0; p < g.planes; ++p) {
    const uint32_t a1 = static_cast<uint32_t>(p % g.D1), a2 = static_cast<uint32_t>((p / g.D1) % g.D2);
    const bool has1 = a1 > 0, has2 = a2 > 0;
    const Remap r1 = has1 ? make_remap(s, p - 1, p) : Remap{};
    const Remap r2 = has2 ? make_remap(s, p - g.D1, p) : Remap{};
    uint16_t* ecur = errs.data() + (p % rings) * ring_plane;
    const uint16_t* eprv = errs.data() + ((p + 1) % rings) * ring_plane;  // read only when the plane before exists
    const size_t base = p * g.plane;
    for (uint32_t y = 0; y < g.Y; ++y) {
      for (uint32_t x = 0; x < g.X; ++x) {
        if (ac.overrun()) return false;
        uint32_t chan_hash = 0;
        for (uint32_t c = 0; c < g.C; ++c) {
          const size_t o = (static_cast<size_t>(y) * g.X + x) * g.C + c;
          const size_t i = base + o;
          // Neighbours (same channel): offsets within the plane, or npos.
          constexpr size_t npos = ~size_t{0};
          const size_t oW = x > 0 ? o - g.C : npos, oN = y > 0 ? o - CX : npos;
          const size_t oNW = (x > 0 && y > 0) ? o - CX - g.C : npos, oNE = (y > 0 && x + 1 < g.X) ? o - CX + g.C : npos;
          const size_t oWW = x > 1 ? o - 2 * g.C : npos, oNN = y > 1 ? o - 2 * CX : npos;
          const size_t oS = y + 1 < g.Y ? o + CX : npos, oE = x + 1 < g.X ? o + g.C : npos;
          const size_t oSW = (y + 1 < g.Y && x > 0) ? o + CX - g.C : npos;
          const auto L = [&](size_t off) { return lin[base + off]; };
          const auto L1 = [&](size_t off) { return r1(lin[base - stride1 + off]); };
          const bool hW = oW != npos, hN = oN != npos, hNW = oNW != npos, hNE = oNE != npos;
          int64_t base_pred = lin_default;
          if (hW) base_pred = L(oW);
          else if (hN) base_pred = L(oN);
          else if (has1) base_pred = L1(o);
          else if (c > 0) base_pred = L(o - 1);
          else if (o > 0) base_pred = L(o - 1);
          const int64_t W = hW ? L(oW) : base_pred, N = hN ? L(oN) : base_pred;
          const int64_t NW = hNW ? L(oNW) : (hN ? N : W), NE = hNE ? L(oNE) : N;
          std::array<int64_t, kPred - 1> pr;  // fixed predictors, in values; the last of kPred is the adaptive one
          pr[0] = W;
          pr[1] = N;
          pr[2] = W + N - NW;
          {
            const int64_t mn = std::min(W, N), mx = std::max(W, N);
            pr[3] = NW >= mx ? mn : NW <= mn ? mx : W + N - NW;  // median edge detector (LOCO-I)
          }
          pr[4] = (W + NE) / 2;
          const int64_t P1 = has1 ? L1(o) : base_pred;
          const int64_t P1S = has1 && oS != npos ? L1(oS) : P1, P1E = has1 && oE != npos ? L1(oE) : P1;
          pr[5] = P1;
          pr[6] = P1S;
          pr[7] = has1 && hW ? P1 + W - L1(oW) : P1;
          pr[8] = has1 && hW && oSW != npos ? P1S + W - L1(oSW) : P1S;
          pr[9] = (2 * P1 + P1S + P1E + 2) / 4;
          pr[10] = has2 ? r2(lin[i - stride2]) : P1;
          if (c > 0) {
            const int64_t prev = L(o - 1);
            pr[11] = hW ? W + prev - L(oW - 1) : prev;
            pr[12] = hN ? N + prev - L(oN - 1) : prev;
          } else {
            pr[11] = (W + N + 1) / 2;
            pr[12] = (3 * W + 3 * N - 2 * NW + 2) / 4;
          }
          // Predictions as codes * 256, blended by the inverse square of their recent errors nearby.
          std::array<int64_t, kPred> pc;
          for (int j = 0; j < kPred - 1; ++j) {
            pc[static_cast<size_t>(j)] = f16 ? static_cast<int64_t>(lin_code(pr[static_cast<size_t>(j)])) << 8
                                             : std::clamp<int64_t>(pr[static_cast<size_t>(j)], 0, code_max);
          }
          // Adaptive linear predictor (normalised LMS) on the neighbours, relative to the median predictor.
          const auto V = [&](int64_t l) { return f16 ? static_cast<int64_t>(lin_code(l)) << 8 : std::clamp<int64_t>(l, 0, code_max); };
          const int64_t ref = pc[3];
          std::array<int64_t, kLms> lx{};
          {
            const auto put = [&](int k, bool have, int64_t l) { lx[static_cast<size_t>(k)] = have ? (V(l) - ref) >> lms_shift : 0; };
            put(0, hW, W);
            put(1, hN, N);
            put(2, hNW, NW);
            put(3, hNE, NE);
            put(4, oWW != npos, oWW != npos ? L(oWW) : 0);
            put(5, oNN != npos, oNN != npos ? L(oNN) : 0);
            put(6, has1, P1);
            put(7, has1 && oS != npos, P1S);
            put(8, has1 && oE != npos, P1E);
            put(9, has1 && hW, has1 && hW ? L1(oW) : 0);
            put(10, has1 && hN, has1 && hN ? L1(oN) : 0);
            put(11, has2, pr[10]);
          }
          int64_t ldot = 0, lnorm = 64;
          for (int k = 0; k < kLms; ++k) {
            ldot += lw[static_cast<size_t>(k)] * lx[static_cast<size_t>(k)];
            lnorm += lx[static_cast<size_t>(k)] * lx[static_cast<size_t>(k)];
          }
          ldot >>= 16;
          pc[kPred - 1] = std::clamp<int64_t>(ref + (ldot << lms_shift), 0, code_max);
          uint64_t wsum = 0, best_e = ~uint64_t{0};
          int64_t acc = 0, best = pc[0];
          for (int j = 0; j < kPred; ++j) {
            const uint16_t* ej = ecur + static_cast<size_t>(j) * g.plane;
            uint64_t e = 1;
            if (hW) e += ej[oW];
            if (hN) e += ej[oN];
            if (hNW) e += ej[oNW];
            if (hNE) e += ej[oNE];
            if (has1) e += eprv[static_cast<size_t>(j) * g.plane + o];
            if (e < best_e) {
              best_e = e;
              best = pc[static_cast<size_t>(j)];
            }
            const uint64_t r = e < 65536 ? 65536u / static_cast<uint32_t>(e) : 0u;
            const uint64_t w = r * r;  // about 2^32 / e^2
            wsum += w;
            acc += static_cast<int64_t>(w) * pc[static_cast<size_t>(j)];
          }
          const int64_t pred = wsum ? acc / static_cast<int64_t>(wsum) : pc[3];
          // Local error level of the blend.
          uint64_t act = 0;
          int nact = 0;
          {
            const uint16_t* eb = ecur + static_cast<size_t>(kPred) * g.plane;
            if (hW) act += eb[oW], ++nact;
            if (hN) act += eb[oN], ++nact;
            if (hNW) act += eb[oNW], ++nact;
            if (hNE) act += eb[oNE], ++nact;
            if (has1) act += eprv[static_cast<size_t>(kPred) * g.plane + o], ++nact;
          }
          const uint64_t act4 = nact ? act * 4 / static_cast<uint64_t>(nact) : (f16 ? 8192u : 2048u);
          const int actb = std::min(kAct - 1, ilog2(act4) + 1);
          // 2^32 / (the local error scale): offsets from the prediction in units of the expected error.
          const int64_t sig_inv = (int64_t{1} << 32) / (f16 ? 16 * (static_cast<int64_t>(act4) + 1) : static_cast<int64_t>(act4) + 1);
          // Neighbour codes for the hashed contexts.
          const auto C0 = [&](size_t off) -> int { return off == npos ? -1 : code[base + off]; };
          const auto C1 = [&](size_t off) -> int {
            if (!has1 || off == npos) return -1;
            if (!r1.on) return code[base - stride1 + off];
            return static_cast<int>(std::clamp<int64_t>((r1(lin[base - stride1 + off]) + 128) >> 8, 0, 255));
          };
          const int cW = C0(oW), cN = C0(oN), cNE = C0(oNE), cNW = C0(oNW), cWW = C0(oWW), cNN = C0(oNN);
          const int cP1 = C1(o), cP1S = C1(oS);
          int cP2 = -1;
          if (has2) cP2 = r2.on ? static_cast<int>(std::clamp<int64_t>((r2(lin[i - stride2]) + 128) >> 8, 0, 255)) : code[i - stride2];
          const int cprev = c > 0 ? code[i - 1] : -1;
          const int pcode = static_cast<int>(std::clamp<int64_t>((pred + 128) >> 8, 0, code_max >> 8));
          int bflags = 0;
          if (bc3 && c >= 2) {
            const uint16_t* blk = code.data() + i - c;
            bflags = 1 + (blk[0] == blk[1]) + 2 * (blk[0] == 0) + 4 * (c >= 12 && blk[8] == blk[10] && blk[9] == blk[11]);
          }

          uint32_t sym = AC::encoding ? (f16 ? f16_order(values[i]) : values[i]) : 0u;
          const bool matched = lz != nullptr && lz_token(*lz, ac, code.data(), i, group, static_cast<uint32_t>(pcode), sym);
          for (int b = matched ? -1 : width - 1; b >= 0; --b) {
            const int role = f16 ? (b == 1 ? 1 : 2) : 0;
            const int hi = role == 2 ? static_cast<int>(sym >> 8) : 0;
            // A neighbour's code as seen by this byte: the value (8-bit), the high byte (fp16 high byte), or for the
            // low byte its low byte when its high byte matches, else below or above.
            const auto nb = [&](int v) -> uint32_t {
              if (v < 0) return 0x3ffu;
              if (role == 0) return static_cast<uint32_t>(v);
              if (role == 1) return static_cast<uint32_t>(v >> 8);
              const int h = v >> 8;
              return h == hi ? static_cast<uint32_t>(v & 255) : (h < hi ? 0x100u : 0x101u);
            };
            const uint32_t predq = role == 1 ? static_cast<uint32_t>(pcode >> 6) : nb(pcode);
            const uint32_t B = mixes(kind + 1, role, c, role == 2 ? hi + 1 : 0);
            const uint32_t nW = nb(cW), nN = nb(cN), nP1 = nb(cP1);
            ctx[0] = B;
            ctx[1] = mixes(B, 1, nW);
            ctx[2] = mixes(B, 2, nN);
            ctx[3] = mixes(B, 3, nP1);
            ctx[4] = mixes(B, 4, nW, nN);
            ctx[5] = mixes(B, 5, predq);
            ctx[6] = mixes(B, 6, predq >> 2, actb);
            ctx[7] = mixes(B, 7, nW >> 2, nN >> 2, nb(cNE) >> 2);
            ctx[8] = mixes(B, 8, serial, p);
            ctx[9] = mixes(B, 9, serial, p, y);
            ctx[10] = mixes(B, 10, serial, p, x);
            ctx[11] = g.C > 1 ? mixes(B, 11, chan_hash) : mixes(B, 11, nW, nb(cWW));
            ctx[12] = has1 ? mixes(B, 12, nP1, nb(cP1S), nW) : mixes(B, 12, nb(cNW), nW);
            ctx[13] = has2 ? mixes(B, 13, nb(cP2), nW) : mixes(B, 13, nN, nb(cNN));
            if (bc3) {
              ctx[5] = mixes(B, 5, nW, nN, nP1);
              ctx[6] = mixes(B, 6, bflags, nb(cprev));
              ctx[7] = mixes(B, 7, chan_hash, nW);
              ctx[9] = mixes(B, 9, chan_hash, nP1);
              ctx[10] = mixes(B, 10, nW, nN, nb(cNW));
            }
            model.byte(ctx.data());
            for (int k = 8 * b + 7; k >= 8 * b; --k) {
              const int kk = f16 ? 8 + k : k;
              const int64_t mid = static_cast<int64_t>(((sym >> (k + 1)) << (k + 1)) + (1u << k)) << 8;
              const int64_t d = pred - mid, db = best - mid;
              const int rr1 = static_cast<int>(std::clamp<int64_t>(d >> (k + 5), -24, 24));
              const int rr3 = static_cast<int>(std::clamp<int64_t>(db >> (k + 5), -24, 24));
              const int rr2 = static_cast<int>(std::clamp<int64_t>((d * sig_inv) >> 32, -32, 31));
              const size_t kb = static_cast<size_t>(group) * kBitIdx + static_cast<size_t>(kk);
              const int bits_done = 8 * b + 7 - k;
              const uint32_t c0 = (1u << bits_done) | ((sym >> (k + 1)) & ((1u << bits_done) - 1));
              BitSel bs{};
              bs.direct[0] = (kb * kRel + static_cast<size_t>(rr1 + 24)) * kAct + static_cast<size_t>(actb);
              bs.direct[1] = kb * kSig + static_cast<size_t>(rr2 + 32);
              bs.direct[2] = (kb * kRel + static_cast<size_t>(rr3 + 24)) * kAct + static_cast<size_t>(actb);
              bs.mix1 = kb * 4 + static_cast<size_t>(actb >> 2);
              bs.mix2 = (static_cast<size_t>(group) * 3 + static_cast<size_t>(role)) * 256 + c0;
              bs.apm1 = bs.mix2;
              bs.apm2 = kb * kRel + static_cast<size_t>(rr1 + 24);
              const int p16 = model.predict(bs);
              const int bit = ac.code(p16, static_cast<int>((sym >> k) & 1u));
              if constexpr (!AC::encoding) sym |= static_cast<uint32_t>(bit) << k;
              model.update(bit);
            }
          }
          if constexpr (!AC::encoding) values[i] = f16 ? f16_unorder(static_cast<uint16_t>(sym)) : static_cast<uint16_t>(sym);
          code[i] = static_cast<uint16_t>(sym);
          lin[i] = f16 ? f16_lin(f16_unorder(static_cast<uint16_t>(sym))) : static_cast<int64_t>(sym) << 8;
          // Errors of every predictor and of the blend at this position.
          const int64_t actual = static_cast<int64_t>(sym) << 8;
          for (int j = 0; j <= kPred; ++j) {
            const int64_t e = (j < kPred ? pc[static_cast<size_t>(j)] : pred) - actual;
            ecur[static_cast<size_t>(j) * g.plane + o] = static_cast<uint16_t>(std::min<int64_t>(65535, (e < 0 ? -e : e) >> err_shift));
          }
          chan_hash = mixes(chan_hash + 0x51u, sym);
          {
            const int64_t err = ((actual - ref) >> lms_shift) - ldot;
            const int64_t step = (err * lms_mu * 65536) / lnorm;
            for (int k = 0; k < kLms; ++k) {
              int64_t& w = lw[static_cast<size_t>(k)];
              w = std::clamp<int64_t>(w + ((step * lx[static_cast<size_t>(k)]) >> 16), -(int64_t{1} << 20), int64_t{1} << 20);
            }
          }
        }
      }
    }
  }
  return !ac.overrun();
}

// The light coder (format 2) on planes [p0, p1) of a tensor; `values` holds those planes only. Predictions as the full
// coder's but fewer (seven fixed predictors, no adaptive linear one), blended the same way; nothing before p0 is read,
// so a segment decodes on its own. Each value is first offered to the LZ tokens when `lz` is set.
constexpr int kFastPred = 7;
constexpr int kPlainShift = 3;  // a bit is coded plainly when the expected error is at least 8 times its weight

// 65536 / e without a division: exact below 4096, from the table at e / 64 above (e < 2^18), else 0.
struct RecipTable {
  std::array<uint32_t, 4096> t{};
  constexpr RecipTable() {
    t[0] = 65536;
    for (uint32_t e = 1; e < 4096; ++e) t[e] = 65536u / e;
  }
};
constexpr RecipTable kRecip;
inline uint64_t recip16(uint64_t e) {
  if (e < 4096) return kRecip.t[e];
  if (e < (uint64_t{1} << 18)) return kRecip.t[e >> 6] >> 6;
  return 0;
}

template <class AC>
bool code_tensor_fast(Light& model, AC& ac, const Shape& s, size_t p0, size_t p1, uint16_t* values, LzState* lz) {
  // Plain bits (see below) for the kinds whose low bits are noise: features and weights. Not for smooth fields (coarse
  // states), whose exact zeros the low bits still predict, nor for bytes without structure.
  const bool plain_ok = s.kind == Kind::features || s.kind == Kind::weights || s.kind == Kind::biases || s.kind == Kind::codes;
  const Geometry g = geometry(s);
  const size_t n = (p1 - p0) * g.plane;
  if (n == 0) return true;
  const int width = s.width;
  const bool f16 = width == 2;
  uint32_t group = static_cast<uint32_t>(s.kind);
  if (s.kind == Kind::biases || s.kind == Kind::codes || s.kind == Kind::scales) group = static_cast<uint32_t>(Kind::weights);
  const uint32_t slot = model.groups() == 1 ? 0u : group;  // where this group's statistics are
  const int err_shift = f16 ? 8 : 4;
  const int64_t code_max = f16 ? 65535 * 256 : 255 * 256;
  const int64_t lin_default = f16 ? 0 : 128 * 256;
  std::vector<uint16_t> code(n);
  std::vector<int64_t> lin(n);
  const size_t ring_plane = g.plane * (kFastPred + 1);
  const size_t rings = g.D1 > 1 ? 2 : 1;
  std::vector<uint16_t> errs(rings * ring_plane, 0);
  const size_t CX = static_cast<size_t>(g.C) * g.X;
  const size_t stride1 = g.plane, stride2 = g.plane * g.D1;
  if (lz) lz->start(n);
  for (size_t p = p0; p < p1; ++p) {
    const uint32_t a1 = static_cast<uint32_t>(p % g.D1), a2 = static_cast<uint32_t>((p / g.D1) % g.D2);
    const bool has1 = a1 > 0 && p > p0, has2 = a2 > 0 && p >= p0 + g.D1;
    const Remap r1 = has1 ? make_remap(s, p - 1, p) : Remap{};
    const Remap r2 = has2 ? make_remap(s, p - g.D1, p) : Remap{};
    uint16_t* ecur = errs.data() + (p % rings) * ring_plane;
    const uint16_t* eprv = errs.data() + ((p + 1) % rings) * ring_plane;
    const size_t base = (p - p0) * g.plane;
    for (uint32_t y = 0; y < g.Y; ++y) {
      for (uint32_t x = 0; x < g.X; ++x) {
        if (ac.overrun()) return false;
        for (uint32_t c = 0; c < g.C; ++c) {
          const size_t o = (static_cast<size_t>(y) * g.X + x) * g.C + c;
          const size_t i = base + o;
          constexpr size_t npos = ~size_t{0};
          const size_t oW = x > 0 ? o - g.C : npos, oN = y > 0 ? o - CX : npos;
          const size_t oNW = (x > 0 && y > 0) ? o - CX - g.C : npos, oNE = (y > 0 && x + 1 < g.X) ? o - CX + g.C : npos;
          const auto L = [&](size_t off) { return lin[base + off]; };
          const auto L1 = [&](size_t off) { return r1(lin[base - stride1 + off]); };
          const bool hW = oW != npos, hN = oN != npos, hNW = oNW != npos, hNE = oNE != npos;
          int64_t base_pred = lin_default;
          if (hW) base_pred = L(oW);
          else if (hN) base_pred = L(oN);
          else if (has1) base_pred = L1(o);
          else if (o > 0) base_pred = L(o - 1);
          const int64_t W = hW ? L(oW) : base_pred, N = hN ? L(oN) : base_pred;
          const int64_t NW = hNW ? L(oNW) : (hN ? N : W), NE = hNE ? L(oNE) : N;
          std::array<int64_t, kFastPred> pr;
          pr[0] = W;
          pr[1] = N;
          {
            const int64_t mn = std::min(W, N), mx = std::max(W, N);
            pr[2] = NW >= mx ? mn : NW <= mn ? mx : W + N - NW;
          }
          pr[3] = W + N - NW;
          const int64_t P1 = has1 ? L1(o) : base_pred;
          pr[4] = P1;
          pr[5] = has1 && hW ? P1 + W - L1(oW) : (has2 ? r2(lin[i - stride2]) : (W + NE) / 2);
          if (c > 0) {
            const int64_t prev = L(o - 1);
            pr[6] = hW ? W + prev - L(oW - 1) : prev;
          } else {
            pr[6] = (W + NE) / 2;
          }
          std::array<int64_t, kFastPred> pc;
          for (int j = 0; j < kFastPred; ++j) {
            pc[static_cast<size_t>(j)] = f16 ? static_cast<int64_t>(lin_code(pr[static_cast<size_t>(j)])) << 8
                                             : std::clamp<int64_t>(pr[static_cast<size_t>(j)], 0, code_max);
          }
          uint64_t wsum = 0, best_e = ~uint64_t{0};
          int64_t acc = 0, best = pc[0];
          for (int j = 0; j < kFastPred; ++j) {
            const uint16_t* ej = ecur + static_cast<size_t>(j) * g.plane;
            uint64_t e = 1;
            if (hW) e += ej[oW];
            if (hN) e += ej[oN];
            if (hNW) e += ej[oNW];
            if (hNE) e += ej[oNE];
            if (has1) e += eprv[static_cast<size_t>(j) * g.plane + o];
            if (e < best_e) {
              best_e = e;
              best = pc[static_cast<size_t>(j)];
            }
            const uint64_t r = recip16(e);
            const uint64_t w = r * r;
            wsum += w;
            acc += static_cast<int64_t>(w) * pc[static_cast<size_t>(j)];
          }
          const int64_t pred = wsum ? acc / static_cast<int64_t>(wsum) : pc[2];
          uint64_t act = 0;
          int nact = 0;
          {
            const uint16_t* eb = ecur + static_cast<size_t>(kFastPred) * g.plane;
            if (hW) act += eb[oW], ++nact;
            if (hN) act += eb[oN], ++nact;
            if (hNW) act += eb[oNW], ++nact;
            if (hNE) act += eb[oNE], ++nact;
            if (has1) act += eprv[static_cast<size_t>(kFastPred) * g.plane + o], ++nact;
          }
          const uint64_t act4 = nact ? act * 4 / static_cast<uint64_t>(nact) : (f16 ? 8192u : 2048u);
          const int actb = std::min(kAct - 1, ilog2(act4) + 1);
          const int64_t sig_inv = (int64_t{1} << 32) / (f16 ? 16 * (static_cast<int64_t>(act4) + 1) : static_cast<int64_t>(act4) + 1);
          const int pcode = static_cast<int>(std::clamp<int64_t>((pred + 128) >> 8, 0, code_max >> 8));

          uint32_t sym = AC::encoding ? (f16 ? f16_order(values[i]) : values[i]) : 0u;
          const bool matched = lz != nullptr && lz_token(*lz, ac, code.data(), i, group, static_cast<uint32_t>(pcode), sym);
          // Bits whose weight is far below the expected error are close to even odds: coded as such, without a model.
          // The expected error in code units is act4 / 64 (8-bit) or act4 / 4 (fp16); bit k weighs 2^k.
          const int64_t expected = f16 ? static_cast<int64_t>(act4 >> 2) : static_cast<int64_t>(act4 >> 6);
          int plain = 0;  // bits below this one are coded plainly
          if (plain_ok) {
            while (plain < 8 * width - 1 && (int64_t{1} << (plain + kPlainShift)) <= expected) ++plain;
          }
          if (!matched) {
            for (int k = 8 * width - 1; k >= 0; --k) {
              if (k < plain) {
                const int bit = ac.code(32768, static_cast<int>((sym >> k) & 1u));
                if constexpr (!AC::encoding) sym |= static_cast<uint32_t>(bit) << k;
                continue;
              }
              const int kk = f16 ? 8 + k : k;
              const int role = f16 ? (k >= 8 ? 1 : 2) : 0;
              const int64_t mid = static_cast<int64_t>(((sym >> (k + 1)) << (k + 1)) + (1u << k)) << 8;
              const int64_t d = pred - mid, db = best - mid;
              const int rr1 = static_cast<int>(std::clamp<int64_t>(d >> (k + 5), -24, 24));
              const int rr3 = static_cast<int>(std::clamp<int64_t>(db >> (k + 5), -24, 24));
              const int rr2 = static_cast<int>(std::clamp<int64_t>((d * sig_inv) >> 32, -32, 31));
              const size_t kb = static_cast<size_t>(slot) * kBitIdx + static_cast<size_t>(kk);
              const int b8 = k & 7, bits_done = 7 - b8;
              const uint32_t c0 = (1u << bits_done) | ((sym >> (k + 1)) & ((1u << bits_done) - 1));
              const int p16 = model.predict((kb * kRel + static_cast<size_t>(rr1 + 24)) * kAct + static_cast<size_t>(actb),
                                            kb * kSig + static_cast<size_t>(rr2 + 32),
                                            (kb * kRel + static_cast<size_t>(rr3 + 24)) * kAct + static_cast<size_t>(actb),
                                            (static_cast<size_t>(slot) * 3 + static_cast<size_t>(role)) * 256 + c0,
                                            kb * 4 + static_cast<size_t>(actb >> 2), kb * kRel + static_cast<size_t>(rr1 + 24));
              const int bit = ac.code(p16, static_cast<int>((sym >> k) & 1u));
              if constexpr (!AC::encoding) sym |= static_cast<uint32_t>(bit) << k;
              model.update(bit);
            }
          }
          if constexpr (!AC::encoding) values[i] = f16 ? f16_unorder(static_cast<uint16_t>(sym)) : static_cast<uint16_t>(sym);
          code[i] = static_cast<uint16_t>(sym);
          lin[i] = f16 ? f16_lin(f16_unorder(static_cast<uint16_t>(sym))) : static_cast<int64_t>(sym) << 8;
          const int64_t actual = static_cast<int64_t>(sym) << 8;
          for (int j = 0; j <= kFastPred; ++j) {
            const int64_t e = (j < kFastPred ? pc[static_cast<size_t>(j)] : pred) - actual;
            ecur[static_cast<size_t>(j) * g.plane + o] = static_cast<uint16_t>(std::min<int64_t>(65535, (e < 0 ? -e : e) >> err_shift));
          }
        }
      }
    }
  }
  return !ac.overrun();
}

// The fast coder (format 2), on planes [p0, p1) as code_tensor_fast. In the manner of LOCO-I (JPEG-LS): per value one
// of three cheap predictions in code units (the one with the smaller error at the left and upper neighbours: the
// median edge detector, and with a plane before, that plane's value and its value plus the local gradient), the
// residual binarised as a Golomb-Rice code whose parameter follows the residuals' running mean in an activity context,
// the unary part coded with one adaptive statistic per decision and the low bits plainly. A few coder steps per value
// and no mixing: many times faster than the light model, larger.
constexpr int kRiceBuckets = 18;  // activity contexts: bit length of the left and upper residuals' sum
constexpr int kRiceUnary = 16;    // unary decisions before the escape (the residual then follows in plain bits)

class Rice {
 public:
  explicit Rice(size_t groups) : groups_(groups), cont_(groups * kRiceBuckets * kRiceUnary, kFresh), A_(groups * kRiceBuckets, 4), N_(groups * kRiceBuckets, 1) {}
  size_t groups() const { return groups_; }
  uint32_t& cont(size_t g, int b, int j) { return cont_[(g * kRiceBuckets + static_cast<size_t>(b)) * kRiceUnary + static_cast<size_t>(j)]; }
  int k(size_t g, int b) const {
    const size_t i = g * kRiceBuckets + static_cast<size_t>(b);
    int k = 0;
    while (k < 16 && (static_cast<uint64_t>(N_[i]) << k) < A_[i]) ++k;
    return k;
  }
  void update(size_t g, int b, uint32_t u) {
    const size_t i = g * kRiceBuckets + static_cast<size_t>(b);
    A_[i] += u;
    if (++N_[i] == 64) {
      A_[i] = (A_[i] + 1) >> 1;
      N_[i] >>= 1;
    }
  }

 private:
  size_t groups_;
  std::vector<uint32_t> cont_;
  std::vector<uint64_t> A_;
  std::vector<uint32_t> N_;
};

template <class AC>
inline uint32_t plain_bits(AC& ac, uint32_t v, int n) {
  return ac.plain(v, n);
}

template <class AC>
bool code_tensor_rice(Rice& model, AC& ac, const Shape& s, size_t p0, size_t p1, uint16_t* values, LzState* lz) {
  const Geometry g = geometry(s);
  const size_t n = (p1 - p0) * g.plane;
  if (n == 0) return true;
  const bool f16 = s.width == 2;
  uint32_t group = static_cast<uint32_t>(s.kind);
  if (s.kind == Kind::biases || s.kind == Kind::codes || s.kind == Kind::scales) group = static_cast<uint32_t>(Kind::weights);
  const size_t slot = model.groups() == 1 ? 0u : group;
  const int32_t cmax = f16 ? 65535 : 255;
  const int ubits = f16 ? 17 : 9;  // bits of a zigzagged residual
  std::vector<uint16_t> code(n);
  std::vector<uint16_t> err(g.plane * 4, 0);  // per position: |error| of the three predictions and of the one used
  const size_t CX = static_cast<size_t>(g.C) * g.X;
  std::array<uint8_t, 256> lut{};  // 8-bit planes with maps: the plane before's codes in this plane's scale
  if (lz) lz->start(n);
  for (size_t p = p0; p < p1; ++p) {
    const bool has1 = p % g.D1 > 0 && p > p0;
    bool mapped = false;
    if (has1) {
      const Remap r1 = make_remap(s, p - 1, p);
      mapped = r1.on;
      if (mapped) {
        for (int q = 0; q < 256; ++q) lut[static_cast<size_t>(q)] = static_cast<uint8_t>(std::clamp<int64_t>((r1(int64_t{q} << 8) + 128) >> 8, 0, 255));
      }
    }
    const size_t base = (p - p0) * g.plane;
    const auto P = [&](size_t o) -> int32_t {
      const uint16_t v = code[base - g.plane + o];
      return mapped ? lut[v] : v;
    };
    for (uint32_t y = 0; y < g.Y; ++y) {
      if (ac.overrun()) return false;
      for (uint32_t x = 0; x < g.X; ++x) {
        for (uint32_t c = 0; c < g.C; ++c) {
          const size_t o = (static_cast<size_t>(y) * g.X + x) * g.C + c;
          const size_t i = base + o;
          const bool hW = x > 0, hN = y > 0;
          const size_t oW = o - g.C, oN = o - CX;
          int32_t fallback = f16 ? 0x8000 : 128;
          if (hW) fallback = code[i - g.C];
          else if (hN) fallback = code[i - CX];
          else if (has1) fallback = P(o);
          else if (o > 0) fallback = code[i - 1];
          const int32_t W = hW ? code[i - g.C] : fallback, N = hN ? code[i - CX] : fallback;
          const int32_t NW = hW && hN ? code[i - CX - g.C] : (hN ? N : W);
          std::array<int32_t, 3> pr;
          {
            const int32_t mn = std::min(W, N), mx = std::max(W, N);
            pr[0] = NW >= mx ? mn : NW <= mn ? mx : W + N - NW;
          }
          if (has1) {
            const int32_t P1 = P(o);
            pr[1] = P1;
            pr[2] = hW ? P1 + W - P(oW) : P1;
          } else if (c > 0) {
            const int32_t prev = code[i - 1];
            pr[1] = hW ? W + prev - code[i - g.C - 1] : prev;
            pr[2] = W + N - NW;
          } else {
            pr[1] = W + N - NW;
            pr[2] = (W + N + 1) >> 1;
          }
          int best = 0;
          uint32_t best_e = ~0u;
          for (int j = 0; j < 3; ++j) {
            pr[static_cast<size_t>(j)] = std::clamp(pr[static_cast<size_t>(j)], 0, cmax);
            uint32_t e = 0;
            if (hW) e += err[oW * 4 + static_cast<size_t>(j)];
            if (hN) e += err[oN * 4 + static_cast<size_t>(j)];
            if (e < best_e) {
              best_e = e;
              best = j;
            }
          }
          const int32_t pred = pr[static_cast<size_t>(best)];
          uint32_t act = 0;
          if (hW) act += err[oW * 4 + 3];
          if (hN) act += err[oN * 4 + 3];
          if (!hW && !hN) act = f16 ? 4096u : 16u;
          const int b = std::min(kRiceBuckets - 1, static_cast<int>(std::bit_width(act)));

          uint32_t sym = AC::encoding ? (f16 ? f16_order(values[i]) : values[i]) : 0u;
          const bool matched = lz != nullptr && lz_token(*lz, ac, code.data(), i, group, static_cast<uint32_t>(pred), sym);
          if (!matched) {
            const int k = model.k(slot, b);
            uint32_t u = 0;
            if constexpr (AC::encoding) {
              const int32_t r = static_cast<int32_t>(sym) - pred;
              u = r >= 0 ? static_cast<uint32_t>(2 * r) : static_cast<uint32_t>(-2 * r - 1);
            }
            const uint32_t q = u >> k;
            int j = 0;
            for (; j < kRiceUnary; ++j) {
              uint32_t& cn = model.cont(slot, b, j);
              const int p16 = std::clamp(static_cast<int>(cn >> 16), 32, 65536 - 32);
              const int more = ac.code(p16, AC::encoding ? (q > static_cast<uint32_t>(j) ? 1 : 0) : 0);
              train(cn, more, 255);
              if (!more) break;
            }
            if (j == kRiceUnary) {
              u = plain_bits(ac, u, ubits);  // escape: the whole residual
            } else {
              const uint32_t low = plain_bits(ac, u & ((1u << k) - 1u), k);
              if constexpr (!AC::encoding) u = (static_cast<uint32_t>(j) << k) | low;
            }
            model.update(slot, b, u);
            if constexpr (!AC::encoding) {
              const int32_t r = (u & 1u) ? -static_cast<int32_t>((u + 1) >> 1) : static_cast<int32_t>(u >> 1);
              const int32_t v = pred + r;
              if (v < 0 || v > cmax) return false;  // damaged data
              sym = static_cast<uint32_t>(v);
            }
          }
          if constexpr (!AC::encoding) values[i] = f16 ? f16_unorder(static_cast<uint16_t>(sym)) : static_cast<uint16_t>(sym);
          code[i] = static_cast<uint16_t>(sym);
          for (int j = 0; j < 3; ++j) err[o * 4 + static_cast<size_t>(j)] = static_cast<uint16_t>(std::min<int32_t>(65535, std::abs(static_cast<int32_t>(sym) - pr[static_cast<size_t>(j)])));
          err[o * 4 + 3] = static_cast<uint16_t>(std::min<int32_t>(65535, std::abs(static_cast<int32_t>(sym) - pred)));
        }
      }
    }
  }
  return !ac.overrun();
}

// --- containers -------------------------------------------------------------------------------------------------------

constexpr std::array<uint8_t, 4> kMagic = {'N', 'V', 'F', 'Z'};
constexpr uint8_t kFormat = 1;
constexpr size_t kHeader = 4 + 1 + 1 + 8 + 8;  // magic, format, mode, size, checksum
constexpr uint64_t kMaxSize = uint64_t{1} << 28;  // the largest file packed (256 MB; effect files are a few MB)
enum class Mode : uint8_t { bytes = 0, frame = 1, rollout = 2, tensors = 3 };

uint64_t checksum(std::span<const uint8_t> b) {  // FNV-1a, 64 bits
  uint64_t h = 0xcbf29ce484222325ull;
  for (const uint8_t v : b) h = (h ^ v) * 0x100000001b3ull;
  return h;
}

void put_le(std::vector<uint8_t>& o, uint64_t v, int bytes) {
  for (int i = 0; i < bytes; ++i) o.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
uint64_t get_le(const uint8_t* p, int bytes) {
  uint64_t v = 0;
  for (int i = 0; i < bytes; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
  return v;
}

// The coder's view of a file: bytes are coded in the order the parser asks for them, and in decoding written to
// where they belong, so the parser (shared by both directions) reads only bytes already coded.
template <class AC>
class Io {
 public:
  Io(Model& m, AC& ac, const uint8_t* in, uint8_t* out, size_t size) : m_(m), ac_(ac), in_(in), out_(out), size_(size) {
    if constexpr (AC::encoding) covered_.assign(size, false);
  }
  const uint8_t* data() const { return AC::encoding ? in_ : out_; }
  size_t size() const { return size_; }
  bool fits(size_t off, size_t n) const { return off <= size_ && n <= size_ - off; }
  uint64_t le(size_t off, int bytes) const { return get_le(data() + off, bytes); }

  // Plain bytes, in pieces of at most 1 MB (the working memory of a tensor grows with its size).
  bool raw(size_t off, size_t n) {
    if (!fits(off, n)) return false;
    constexpr size_t kPiece = size_t{1} << 20;
    for (size_t done = 0; done < n; done += kPiece) {
      const size_t len = std::min(kPiece, n - done);
      std::vector<size_t> at(len);
      for (size_t k = 0; k < len; ++k) at[k] = off + done + k;
      Shape s;
      s.kind = Kind::bytes;
      s.dims = {static_cast<uint32_t>(len)};
      if (!tensor(s, at)) return false;
    }
    return true;
  }

  // Values of `s` at byte offsets `at` (little-endian, s.width bytes each), in the order of `at`.
  bool tensor(const Shape& s, const std::vector<size_t>& at) {
    if (at.size() != s.size()) return false;
    for (const size_t a : at) {
      if (!fits(a, static_cast<size_t>(s.width))) return false;
    }
    std::vector<uint16_t> v(at.size());
    if constexpr (AC::encoding) {
      for (size_t k = 0; k < at.size(); ++k) {
        v[k] = static_cast<uint16_t>(get_le(in_ + at[k], s.width));
        for (int b = 0; b < s.width; ++b) {
          if (covered_[at[k] + static_cast<size_t>(b)]) overlap_ = true;
          covered_[at[k] + static_cast<size_t>(b)] = true;
        }
      }
    }
    const double before = ac_.cost;
    if (!code_tensor(m_, ac_, s, v.data())) return false;
    Part& part = parts[static_cast<size_t>(s.kind)];
    part.kind = s.kind;
    part.values += v.size();
    part.bytes += v.size() * static_cast<size_t>(s.width);
    part.coded_bytes += (ac_.cost - before) / 8.0;
    if constexpr (!AC::encoding) {
      for (size_t k = 0; k < at.size(); ++k) {
        for (int b = 0; b < s.width; ++b) out_[at[k] + static_cast<size_t>(b)] = static_cast<uint8_t>(v[k] >> (8 * b));
      }
    }
    return true;
  }

  // Encoding: every byte coded exactly once.
  bool complete() const { return !overlap_ && std::ranges::all_of(covered_, [](bool b) { return b; }); }

  std::array<Part, static_cast<size_t>(Kind::count)> parts{};

 private:
  Model& m_;
  AC& ac_;
  const uint8_t* in_;
  uint8_t* out_;
  size_t size_;
  std::vector<bool> covered_;
  bool overlap_ = false;
};

// The parser's view without coding: the bytes of each kind, in coding order.
class Collect {
 public:
  explicit Collect(std::span<const uint8_t> in) : in_(in) {}
  const uint8_t* data() const { return in_.data(); }
  size_t size() const { return in_.size(); }
  bool fits(size_t off, size_t n) const { return off <= in_.size() && n <= in_.size() - off; }
  uint64_t le(size_t off, int n) const { return get_le(in_.data() + off, n); }
  bool raw(size_t off, size_t n) {
    if (!fits(off, n)) return false;
    auto& b = out[static_cast<size_t>(Kind::bytes)];
    b.insert(b.end(), in_.begin() + static_cast<std::ptrdiff_t>(off), in_.begin() + static_cast<std::ptrdiff_t>(off + n));
    return true;
  }
  bool tensor(const Shape& s, const std::vector<size_t>& at) {
    auto& b = out[static_cast<size_t>(s.kind)];
    for (const size_t a : at) {
      if (!fits(a, static_cast<size_t>(s.width))) return false;
      for (int k = 0; k < s.width; ++k) b.push_back(in_[a + static_cast<size_t>(k)]);
    }
    return true;
  }
  std::array<std::vector<uint8_t>, static_cast<size_t>(Kind::count)> out;

 private:
  std::span<const uint8_t> in_;
};

// Offsets of a contiguous run of values.
std::vector<size_t> run(size_t off, size_t count, int width) {
  std::vector<size_t> at(count);
  for (size_t k = 0; k < count; ++k) at[k] = off + k * static_cast<size_t>(width);
  return at;
}

Shape shape(Kind k, int width, std::vector<uint32_t> dims, bool channels = false) {
  Shape s;
  s.kind = k;
  s.width = width;
  s.dims = std::move(dims);
  s.channels = channels;
  return s;
}

// A dense layer as saved: in, out (u32 each), then fp16 weights [out][in] and biases [out]. `kernel` > 1 codes the
// weights as [out][in / kernel * kernel] rows of whole kernels (conv layers store [out][in channel][3][3]).
template <class IO>
bool walk_dense(IO& io, size_t& pos) {
  if (!io.raw(pos, 8)) return false;
  const uint64_t in = io.le(pos, 4), out = io.le(pos + 4, 4);
  pos += 8;
  if (in > 100000 || out > 100000 || !io.fits(pos, 2 * (in * out + out))) return false;
  if (in * out > 0 && !io.tensor(shape(Kind::weights, 2, {static_cast<uint32_t>(out), static_cast<uint32_t>(in)}), run(pos, in * out, 2))) return false;
  pos += 2 * in * out;
  if (out > 0 && !io.tensor(shape(Kind::biases, 2, {static_cast<uint32_t>(out)}), run(pos, out, 2))) return false;
  pos += 2 * out;
  return true;
}

// A frame model (NVFXMDL1, model.cpp): header, features, dense layers, codes.
template <class IO>
bool walk_frame(IO& io, size_t& pos) {
  pos = 0;
  if (!io.raw(0, 76)) return false;
  const auto i32 = [&](size_t off) { return static_cast<std::int32_t>(io.le(off, 4)); };
  const std::uint32_t arch = static_cast<std::uint32_t>(io.le(12, 4));
  const int n_controls = i32(28), n_latent = i32(32), bases = i32(36), grid_t = i32(40), grid = i32(44), channels = i32(48);
  const int latent = i32(60), c0 = i32(64);
  if ((arch != 1 && arch != 2) || n_controls < 0 || n_controls > 64 || n_latent < 0 || n_latent > 64 || bases < 1 || grid_t < 1) return false;
  const int side = arch == 1 ? grid : latent, C = arch == 1 ? channels : c0;
  if (side < 1 || side > 4096 || C < 1 || C > 4096) return false;
  pos = 76;
  if (!io.raw(pos, 40 + 16 * static_cast<size_t>(n_controls))) return false;
  const uint64_t bits = io.le(pos + 36, 4);
  pos += 40 + 16 * static_cast<size_t>(n_controls);
  const size_t K = static_cast<size_t>(bases), T = static_cast<size_t>(grid_t), Cs = static_cast<size_t>(C), S = static_cast<size_t>(side);
  const size_t planes = K * T * Cs, plane = S * S;
  if (planes * plane > io.size()) return false;
  // Coding order [C][K][T][y][x]: the plane before is the previous time slice, two axes back the previous basis.
  const auto plane_index = [&](size_t c, size_t k, size_t t) { return (k * T + t) * Cs + c; };
  if (bits == 8) {
    if (!io.fits(pos, planes * (4 + plane))) return false;
    std::vector<size_t> at;
    at.reserve(planes * 2);
    for (size_t c = 0; c < Cs; ++c) {
      for (size_t k = 0; k < K; ++k) {
        for (size_t t = 0; t < T; ++t) {
          const size_t off = pos + plane_index(c, k, t) * (4 + plane);
          at.push_back(off);
          at.push_back(off + 2);
        }
      }
    }
    if (!io.tensor(shape(Kind::ranges, 2, {static_cast<uint32_t>(Cs), static_cast<uint32_t>(K), static_cast<uint32_t>(T), 2}, true), at)) return false;
    Shape fs = shape(Kind::features, 1, {static_cast<uint32_t>(Cs), static_cast<uint32_t>(K), static_cast<uint32_t>(T), static_cast<uint32_t>(S), static_cast<uint32_t>(S)});
    at.clear();
    at.reserve(planes * plane);
    for (size_t c = 0; c < Cs; ++c) {
      for (size_t k = 0; k < K; ++k) {
        for (size_t t = 0; t < T; ++t) {
          const size_t off = pos + plane_index(c, k, t) * (4 + plane);
          fs.lo.push_back(f16_lin(static_cast<uint16_t>(io.le(off, 2))));
          fs.hi.push_back(f16_lin(static_cast<uint16_t>(io.le(off + 2, 2))));
          for (size_t j = 0; j < plane; ++j) at.push_back(off + 4 + j);
        }
      }
    }
    if (!io.tensor(fs, at)) return false;
    pos += planes * (4 + plane);
  } else if (bits == 16) {
    if (!io.fits(pos, planes * plane * 2)) return false;
    std::vector<size_t> at;
    at.reserve(planes * plane);
    for (size_t c = 0; c < Cs; ++c) {
      for (size_t k = 0; k < K; ++k) {
        for (size_t t = 0; t < T; ++t) {
          for (size_t j = 0; j < plane; ++j) at.push_back(pos + 2 * (plane_index(c, k, t) * plane + j));
        }
      }
    }
    if (!io.tensor(shape(Kind::features, 2, {static_cast<uint32_t>(Cs), static_cast<uint32_t>(K), static_cast<uint32_t>(T), static_cast<uint32_t>(S), static_cast<uint32_t>(S)}), at)) return false;
    pos += planes * plane * 2;
  } else {
    return false;
  }
  if (!walk_dense(io, pos)) return false;  // basis
  for (int group = 0; group < 2; ++group) {  // layers, films
    if (!io.raw(pos, 4)) return false;
    const uint64_t count = io.le(pos, 4);
    pos += 4;
    if (count > 64) return false;
    for (uint64_t l = 0; l < count; ++l) {
      if (!walk_dense(io, pos)) return false;
    }
  }
  const size_t nl = static_cast<size_t>(n_latent);
  for (int stat = 0; stat < 2; ++stat) {  // z_mean, z_std
    if (nl > 0 && !io.tensor(shape(Kind::codes, 2, {static_cast<uint32_t>(nl)}), run(pos, nl, 2))) return false;
    pos += 2 * nl;
  }
  if (!io.raw(pos, 4)) return false;
  const uint64_t nz = io.le(pos, 4);
  pos += 4;
  if (nz > 1000000 || !io.fits(pos, 2 * nz * nl)) return false;
  if (nz * nl > 0 && !io.tensor(shape(Kind::codes, 2, {static_cast<uint32_t>(nz), static_cast<uint32_t>(nl)}), run(pos, nz * nl, 2))) return false;
  pos += 2 * nz * nl;
  return true;
}

// A rollout effect (NVFXROL1 versions 1 and 2, rollout.cpp): header, stepper and renderer weights, start points.
template <class IO>
bool walk_rollout(IO& io, size_t& pos) {
  pos = 0;
  if (!io.raw(0, 52)) return false;
  const auto i32 = [&](size_t off) { return static_cast<std::int32_t>(io.le(off, 4)); };
  const uint64_t version = io.le(8, 4);
  const int res = i32(12), hidden = i32(16), memory = i32(20), n_controls = i32(28), n_age = i32(32);
  const int render_hidden = i32(40), start_fine = i32(44);
  if (version < 1 || version > 2 || res < 2 || res > 256 || hidden < 1 || hidden > 256 || memory < 0 || memory > 64 ||
      n_controls < 0 || n_controls > 8 || n_age < 0 || n_age > 2 || render_hidden < 1 || render_hidden > 256 || start_fine < 0 ||
      start_fine > 1024) {
    return false;
  }
  pos = 52;
  const size_t rest = 32 + 4 + 1 + 16 * static_cast<size_t>(n_controls) + 60 + 8 + (version >= 2 ? 4 : 0) + 48;
  if (!io.raw(pos, rest)) return false;
  pos += rest;
  const uint32_t H = static_cast<uint32_t>(hidden), I = static_cast<uint32_t>(4 + memory + 2 + 2), O = static_cast<uint32_t>(4 + memory + 1);
  const uint32_t Cd = static_cast<uint32_t>(n_controls + n_age), RH = static_cast<uint32_t>(render_hidden), RI = static_cast<uint32_t>(rollout::kRenderIn);
  struct Seg {
    Kind kind;
    std::vector<uint32_t> dims;
  };
  // The stepper (step_layout) then the renderer (render_layout), as saved.
  const std::vector<Seg> segs = {
      {Kind::weights, {9, I, H}}, {Kind::biases, {H}},     {Kind::weights, {9, H, H}}, {Kind::biases, {H}},
      {Kind::weights, {H, O}},    {Kind::biases, {O}},     {Kind::weights, {Cd, H}},   {Kind::weights, {Cd, H}},
      {Kind::weights, {Cd, H}},   {Kind::weights, {Cd, H}}, {Kind::weights, {RH, RI}}, {Kind::biases, {RH}},
      {Kind::weights, {RH, RH}},  {Kind::biases, {RH}},    {Kind::weights, {4, RH}},   {Kind::biases, {4}}};
  for (const Seg& sg : segs) {
    Shape s = shape(sg.kind, 2, sg.dims);
    const size_t n = s.size();
    if (!io.fits(pos, 2 * n)) return false;
    if (n > 0 && !io.tensor(s, run(pos, n, 2))) return false;
    pos += 2 * n;
  }
  if (!io.raw(pos, 4)) return false;
  const uint64_t count = io.le(pos, 4);
  pos += 4;
  if (count < 1 || count > 4096) return false;
  // Start points: first the small headers in order (each start's layout depends on its has-fine flag), then all
  // coarse states as one tensor, then the fine fields.
  const size_t R = static_cast<size_t>(res), coarse = R * R * rollout::kPhys, SF = static_cast<size_t>(start_fine);
  std::vector<size_t> coarse_at, fine_at;
  for (uint64_t k = 0; k < count; ++k) {
    if (!io.raw(pos, 12 + 2 * static_cast<size_t>(n_controls))) return false;
    pos += 12 + 2 * static_cast<size_t>(n_controls);
    coarse_at.push_back(pos);
    pos += 2 * coarse;
    if (!io.raw(pos, 1)) return false;
    const bool fine = io.data()[pos] != 0;
    pos += 1;
    if (fine) {
      if (SF < 2) return false;
      fine_at.push_back(pos);
      pos += 2 * (2 + SF * SF);
    }
    if (pos > io.size()) return false;
  }
  std::vector<size_t> at;
  at.reserve(count * coarse);
  for (const size_t c : coarse_at) {
    for (size_t j = 0; j < coarse; ++j) at.push_back(c + 2 * j);
  }
  if (!io.tensor(shape(Kind::coarse, 2, {static_cast<uint32_t>(count), static_cast<uint32_t>(R), static_cast<uint32_t>(R), rollout::kPhys}, true), at)) return false;
  if (!fine_at.empty()) {
    const uint32_t nf = static_cast<uint32_t>(fine_at.size());
    at.clear();
    for (const size_t f : fine_at) {
      at.push_back(f);
      at.push_back(f + 2 + SF * SF);
    }
    if (!io.tensor(shape(Kind::scales, 2, {nf, 2}, true), at)) return false;
    // [start][field][y][x]: the plane before is the other field of the same start, two axes back the previous start.
    Shape fs = shape(Kind::fine, 1, {nf, 2, static_cast<uint32_t>(SF), static_cast<uint32_t>(SF)});
    at.clear();
    for (const size_t f : fine_at) {
      for (size_t field = 0; field < 2; ++field) {
        const size_t off = f + field * (2 + SF * SF);
        fs.lo.push_back(0);
        fs.hi.push_back(f16_lin(static_cast<uint16_t>(io.le(off, 2))));
        for (size_t j = 0; j < SF * SF; ++j) at.push_back(off + 2 + j);
      }
    }
    if (!io.tensor(fs, at)) return false;
  }
  return true;
}

// The serialised form of pack_tensors: count, then per tensor its shape and its values.
template <class IO>
bool walk_tensors(IO& io, size_t& pos) {
  pos = 0;
  if (!io.raw(0, 4)) return false;
  const uint64_t count = io.le(0, 4);
  pos = 4;
  if (count > 65536) return false;
  for (uint64_t t = 0; t < count; ++t) {
    if (!io.raw(pos, 5)) return false;
    Shape s;
    s.kind = static_cast<Kind>(io.data()[pos]);
    s.width = io.data()[pos + 1];
    s.channels = io.data()[pos + 2] != 0;
    const size_t nd = io.data()[pos + 3];
    const bool map = io.data()[pos + 4] != 0;
    pos += 5;
    if (static_cast<int>(s.kind) >= kKinds || (s.width != 1 && s.width != 2) || nd > 16 || (map && s.width != 1)) return false;
    if (!io.raw(pos, 4 * nd)) return false;
    size_t total = nd ? 1 : 0;
    for (size_t k = 0; k < nd; ++k) {
      s.dims.push_back(static_cast<uint32_t>(io.le(pos + 4 * k, 4)));
      total *= s.dims.back();
      if (total > io.size()) return false;
    }
    pos += 4 * nd;
    if (map) {
      const size_t np = s.planes();  // at most `total`, which is bounded by the file size
      if (total == 0 || np > total || !io.raw(pos, 16 * np)) return false;
      for (size_t k = 0; k < np; ++k) {
        s.lo.push_back(static_cast<int64_t>(io.le(pos + 16 * k, 8)));
        s.hi.push_back(static_cast<int64_t>(io.le(pos + 16 * k + 8, 8)));
      }
      pos += 16 * np;
    }
    if (!io.fits(pos, total * static_cast<size_t>(s.width))) return false;
    if (total > 0 && !io.tensor(s, run(pos, total, s.width))) return false;
    pos += total * static_cast<size_t>(s.width);
  }
  return true;
}

template <class IO>
bool walk(Mode mode, IO& io) {
  size_t pos = 0;
  bool ok = true;
  switch (mode) {
    case Mode::frame: ok = walk_frame(io, pos); break;
    case Mode::rollout: ok = walk_rollout(io, pos); break;
    case Mode::tensors: ok = walk_tensors(io, pos); break;
    case Mode::bytes: break;
  }
  if (!ok || pos > io.size()) return false;
  return io.raw(pos, io.size() - pos);  // anything after the parsed structure (or the whole file) as plain bytes
}

// --- format 2: LZ tokens, the light model, seekable segments --------------------------------------------------------

constexpr uint8_t kFormat2 = 2;
constexpr uint8_t kModelMask = 3, kFlagLz = 4, kFlagSeek = 8;  // flags: the literal model (0 full, 1 light, 2 fast), options

uint32_t checksum32(uint32_t h, uint8_t v) { return (h ^ v) * 16777619u; }  // FNV-1a, 32 bits
constexpr uint32_t kSum32 = 2166136261u;

// Planes [first, last) of each segment of a tensor, about `target` values each, as even as the units allow. With two
// plane axes or more, the unit is D1 planes (all time slices of a feature plane, both fields of a start point), so the
// plane before is in the segment except at its start; with one (coarse start states), a unit is one plane.
std::vector<std::pair<size_t, size_t>> chunks(const Shape& s, size_t target) {
  const Geometry g = geometry(s);
  const size_t unit = g.D1 > 1 && g.planes > g.D1 ? g.D1 : 1, units = std::max<size_t>(1, g.planes / unit);
  const size_t per = std::max<size_t>(1, target / std::max<size_t>(1, unit * g.plane));
  const size_t n = (units + per - 1) / per;
  std::vector<std::pair<size_t, size_t>> c;
  for (size_t k = 0; k < n; ++k) c.emplace_back(k * units / n * unit, (k + 1) * units / n * unit);
  c.back().second = g.planes;
  return c;
}

// LZ tokens only where values repeat exactly: fine fields (mostly empty), headers and the flipbook kinds. Features,
// weights and states almost never repeat four values in a row and the flag would only cost (measured: results/compression).
bool lz_kind(Kind k) {
  return k == Kind::fine || k == Kind::bytes || k == Kind::bc3 || k == Kind::rgba || k == Kind::flow;
}

// In seekable files, headers and the small tensors that later parsing reads (feature ranges, field scales) go to one
// stream, segment 0, decoded whenever anything is.
bool in_header_stream(Kind k) { return k == Kind::bytes || k == Kind::ranges || k == Kind::scales; }

struct Opt2 {
  int model = 1;  // 0 full, 1 light, 2 fast
  bool lz = true, seek = false;
  size_t segment = size_t{1} << 16;
};

// What a seekable file holds, segment by segment (filled while walking).
struct SegMeta {
  Kind kind = Kind::bytes;
  size_t tensor = 0, first = 0, last = 0, values = 0;
  Shape shape;
  std::vector<size_t> at;  // kept only for the selected segment
};

// The coder's view of a file in format 2 (as Io for format 1). Encoding writes either one stream or segments; decoding
// reads them, and with `select` set decodes segment 0 and that segment only (the others are left as zeros).
template <class AC>
class Io2 {
 public:
  static constexpr bool kEnc = AC::encoding;
  Io2(const Opt2& o, const uint8_t* in, uint8_t* out, size_t size) : o_(o), in_(in), out_(out), size_(size) {
    if constexpr (kEnc) covered_.assign(size, false);
    if (o_.model == 0) full_ = std::make_unique<Model>(size);
    else if (o_.model == 1) light_ = std::make_unique<Light>(static_cast<size_t>(kKinds));  // the stream's, or segment 0's
    else rice_ = std::make_unique<Rice>(static_cast<size_t>(kKinds));
  }
  const uint8_t* data() const { return kEnc ? in_ : out_; }
  size_t size() const { return size_; }
  bool fits(size_t off, size_t n) const { return off <= size_ && n <= size_ - off; }
  uint64_t le(size_t off, int bytes) const { return get_le(data() + off, bytes); }

  bool raw(size_t off, size_t n) {
    if (!fits(off, n)) return false;
    constexpr size_t kPiece = size_t{1} << 20;
    for (size_t done = 0; done < n; done += kPiece) {
      const size_t len = std::min(kPiece, n - done);
      std::vector<size_t> at(len);
      for (size_t k = 0; k < len; ++k) at[k] = off + done + k;
      Shape s;
      s.kind = Kind::bytes;
      s.dims = {static_cast<uint32_t>(len)};
      if (!tensor(s, at)) return false;
    }
    return true;
  }

  bool tensor(const Shape& s, const std::vector<size_t>& at) {
    if (at.size() != s.size()) return false;
    for (const size_t a : at) {
      if (!fits(a, static_cast<size_t>(s.width))) return false;
    }
    std::vector<uint16_t> v(at.size());
    if constexpr (kEnc) {
      for (size_t k = 0; k < at.size(); ++k) {
        v[k] = static_cast<uint16_t>(get_le(in_ + at[k], s.width));
        for (int b = 0; b < s.width; ++b) {
          if (covered_[at[k] + static_cast<size_t>(b)]) overlap_ = true;
          covered_[at[k] + static_cast<size_t>(b)] = true;
        }
      }
    }
    Part& part = parts[static_cast<size_t>(s.kind)];
    part.kind = s.kind;
    part.values += v.size();
    part.bytes += v.size() * static_cast<size_t>(s.width);
    const size_t planes = geometry(s).planes;
    if (!o_.seek) {
      AC& ac = *ac_;
      const double before = ac.cost;
      if (!code_stream(ac, s, planes, v.data())) return false;
      part.coded_bytes += (ac.cost - before) / 8.0;
      place(at, v, s.width, 0, v.size());
      ++tensors_;
      return true;
    }
    if (in_header_stream(s.kind)) {  // segment 0, one stream
      AC& ac = *ac_;
      const double before = ac.cost;
      if (!code_stream(ac, s, planes, v.data())) return false;
      part.coded_bytes += (ac.cost - before) / 8.0;
      for (size_t k = 0; k < v.size(); ++k) {
        for (int b = 0; b < s.width; ++b) header_sum_ = checksum32(header_sum_, static_cast<uint8_t>(v[k] >> (8 * b)));
      }
      place(at, v, s.width, 0, v.size());
      ++tensors_;
      return true;
    }
    const size_t plane = geometry(s).plane;
    for (const auto& [first, last] : chunks(s, o_.segment)) {
      const size_t seg = segs.size() + 1;  // segment 0 is the header stream
      SegMeta m;
      m.kind = s.kind;
      m.tensor = tensors_;
      m.first = first;
      m.last = last;
      m.values = (last - first) * plane;
      const size_t lo = first * plane, hi = last * plane;
      uint16_t* vals = v.data() + lo;
      if constexpr (kEnc) {
        std::vector<uint8_t> buf;
        Encoder ac(buf);
        if (!code_segment(ac, s, first, last, vals)) return false;
        ac.flush();
        part.coded_bytes += ac.cost / 8.0;
        seg_data.push_back(std::move(buf));
        seg_sums.push_back(sum_of(vals, hi - lo, s.width));
      } else {
        if (seg >= spans.size()) return false;
        const bool want = select == 0 || select == seg;
        if (want) {
          Decoder ac(spans[seg]);
          if (!code_segment(ac, s, first, last, vals)) return false;
          if (sum_of(vals, hi - lo, s.width) != sums[seg]) return false;
          place(at, v, s.width, lo, hi);
        }
        if (select == seg) {
          m.shape = s;
          m.at.assign(at.begin() + static_cast<std::ptrdiff_t>(lo), at.begin() + static_cast<std::ptrdiff_t>(hi));
          picked.assign(vals, vals + (hi - lo));
        }
      }
      segs.push_back(std::move(m));
    }
    ++tensors_;
    return true;
  }

  bool complete() const { return !overlap_ && std::ranges::all_of(covered_, [](bool b) { return b; }); }

  std::array<Part, static_cast<size_t>(Kind::count)> parts{};
  std::unique_ptr<AC> ac_;                // the single stream, or the header stream (segment 0)
  std::vector<SegMeta> segs;              // segments 1.. in order
  std::vector<std::vector<uint8_t>> seg_data;  // encoding: their streams
  std::vector<uint32_t> seg_sums;              // encoding: their checksums
  std::vector<std::span<const uint8_t>> spans;  // decoding: every segment's stream (0 = header)
  std::vector<uint32_t> sums;                   // decoding: every segment's checksum
  size_t select = 0;                            // decoding: 0 = every segment, else that one only
  std::vector<uint16_t> picked;                 // decoding: the selected segment's values
  uint32_t header_sum_ = kSum32;

 private:
  // A whole tensor in the single stream (or segment 0), with the stream's model.
  template <class A>
  bool code_stream(A& ac, const Shape& s, size_t planes, uint16_t* v) {
    LzState* lz = o_.lz && lz_kind(s.kind) ? &lz_ : nullptr;
    if (o_.model == 0) return code_tensor(*full_, ac, s, v, lz);
    if (o_.model == 1) return code_tensor_fast(*light_, ac, s, 0, planes, v, lz);
    return code_tensor_rice(*rice_, ac, s, 0, planes, v, lz);
  }
  // A segment: a fresh model for one kind, a fresh coder.
  template <class A>
  bool code_segment(A& ac, const Shape& s, size_t first, size_t last, uint16_t* v) {
    LzState lz;
    LzState* z = o_.lz && lz_kind(s.kind) ? &lz : nullptr;
    if (o_.model == 1) {
      Light m(1);
      return code_tensor_fast(m, ac, s, first, last, v, z);
    }
    Rice m(1);
    return code_tensor_rice(m, ac, s, first, last, v, z);
  }
  static uint32_t sum_of(const uint16_t* v, size_t n, int width) {
    uint32_t h = kSum32;
    for (size_t k = 0; k < n; ++k) {
      for (int b = 0; b < width; ++b) h = checksum32(h, static_cast<uint8_t>(v[k] >> (8 * b)));
    }
    return h;
  }
  void place(const std::vector<size_t>& at, const std::vector<uint16_t>& v, int width, size_t lo, size_t hi) {
    if constexpr (!kEnc) {
      for (size_t k = lo; k < hi; ++k) {
        for (int b = 0; b < width; ++b) out_[at[k] + static_cast<size_t>(b)] = static_cast<uint8_t>(v[k] >> (8 * b));
      }
    }
  }
  Opt2 o_;
  const uint8_t* in_;
  uint8_t* out_;
  size_t size_;
  std::vector<bool> covered_;
  bool overlap_ = false;
  std::unique_ptr<Light> light_;
  std::unique_ptr<Model> full_;
  std::unique_ptr<Rice> rice_;
  LzState lz_;
  size_t tensors_ = 0;
};

uint8_t flags_of(const Opt2& o) {
  return static_cast<uint8_t>(o.model | (o.lz ? kFlagLz : 0) | (o.seek ? kFlagSeek : 0));
}

Packed encode2(Mode mode, std::span<const uint8_t> file, Opt2 o) {
  if (o.seek && o.model == 0) o.model = 1;  // segments: not the full model (too large to restart per segment)
  Packed r;
  r.data.assign(kMagic.begin(), kMagic.end());
  r.data.push_back(kFormat2);
  r.data.push_back(static_cast<uint8_t>(mode));
  put_le(r.data, file.size(), 8);
  put_le(r.data, checksum(file), 8);
  r.data.push_back(flags_of(o));
  std::vector<uint8_t> main;
  Io2<Encoder> io(o, file.data(), nullptr, file.size());
  io.ac_ = std::make_unique<Encoder>(main);
  if (!walk(mode, io) || !io.complete()) {
    if (mode == Mode::bytes) throw std::logic_error("cm: plain coding failed");
    return encode2(Mode::bytes, file, o);
  }
  io.ac_->flush();
  if (o.seek) {
    put_le(r.data, std::min<size_t>(o.segment, 0xffffffffu), 4);
    put_le(r.data, io.seg_data.size() + 1, 4);
    put_le(r.data, main.size(), 4);
    put_le(r.data, io.header_sum_, 4);
    for (size_t k = 0; k < io.seg_data.size(); ++k) {
      put_le(r.data, io.seg_data[k].size(), 4);
      put_le(r.data, io.seg_sums[k], 4);
    }
    r.data.insert(r.data.end(), main.begin(), main.end());
    for (const auto& b : io.seg_data) r.data.insert(r.data.end(), b.begin(), b.end());
  } else {
    r.data.insert(r.data.end(), main.begin(), main.end());
  }
  for (const Part& p : io.parts) {
    if (p.values > 0) r.parts.push_back(p);
  }
  return r;
}

// Decodes a format-2 file: everything (select = 0), the header stream and segment `select`, or (kListOnly) the header
// stream alone, which is enough to list the segments.
constexpr size_t kListOnly = ~size_t{0};

struct Decoded2 {
  Mode mode = Mode::bytes;
  std::vector<uint8_t> file;
  std::vector<SegMeta> segs;
  std::vector<size_t> seg_bytes;  // packed bytes of each segment (0 = header stream)
  std::vector<uint16_t> picked;
};

std::expected<Decoded2, std::string> decode2(std::span<const uint8_t> packed, size_t select) {
  constexpr size_t kHeader2 = kHeader + 1;
  if (packed.size() < kHeader2) return std::unexpected("nvfz: truncated");
  const auto mode = static_cast<Mode>(packed[5]);
  if (packed[5] > static_cast<uint8_t>(Mode::tensors)) return std::unexpected("nvfz: unknown content");
  const uint64_t size = get_le(packed.data() + 6, 8), sum = get_le(packed.data() + 14, 8);
  if (size > kMaxSize) return std::unexpected("nvfz: implausible size");
  const uint8_t flags = packed[kHeader];
  if ((flags & ~(kModelMask | kFlagLz | kFlagSeek)) || (flags & kModelMask) == 3) return std::unexpected("nvfz: unknown options");
  Opt2 o;
  o.model = flags & kModelMask;
  o.lz = (flags & kFlagLz) != 0;
  o.seek = (flags & kFlagSeek) != 0;
  if (o.seek && o.model == 0) return std::unexpected("nvfz: unknown options");
  if (!o.seek && select != 0) return std::unexpected("nvfz: not seekable");
  Decoded2 d;
  d.mode = mode;
  d.file.assign(static_cast<size_t>(size), 0);
  if (o.seek && packed.size() >= kHeader2 + 4) o.segment = std::max<size_t>(1, static_cast<size_t>(get_le(packed.data() + kHeader2, 4)));
  Io2<Decoder> io(o, nullptr, d.file.data(), d.file.size());
  if (o.seek) {
    // the segment size, the number of segments, then each one's packed length and checksum
    constexpr size_t kIndex = kHeader2 + 8;
    if (packed.size() < kIndex) return std::unexpected("nvfz: truncated");
    const uint64_t segment = get_le(packed.data() + kHeader2, 4), count = get_le(packed.data() + kHeader2 + 4, 4);
    if (segment == 0 || count < 1 || count > (packed.size() - kIndex) / 8) return std::unexpected("nvfz: corrupt index");
    size_t at = kIndex + 8 * static_cast<size_t>(count);
    for (size_t k = 0; k < count; ++k) {
      const uint64_t len = get_le(packed.data() + kIndex + 8 * k, 4);
      if (len > packed.size() - at) return std::unexpected("nvfz: corrupt index");
      io.spans.push_back(packed.subspan(at, static_cast<size_t>(len)));
      io.sums.push_back(static_cast<uint32_t>(get_le(packed.data() + kIndex + 4 + 8 * k, 4)));
      d.seg_bytes.push_back(static_cast<size_t>(len));
      at += static_cast<size_t>(len);
    }
    if (select >= count && select != kListOnly) return std::unexpected("nvfz: no such segment");
    io.select = select;
    io.ac_ = std::make_unique<Decoder>(io.spans[0]);
  } else {
    io.ac_ = std::make_unique<Decoder>(packed.subspan(kHeader2));
  }
  if (!walk(mode, io)) return std::unexpected("nvfz: corrupt data");
  if (o.seek && (io.segs.size() + 1 != io.spans.size() || io.header_sum_ != io.sums[0])) return std::unexpected("nvfz: corrupt data");
  if (select == 0 && checksum(d.file) != sum) return std::unexpected("nvfz: checksum mismatch");
  if (select != 0 && select != kListOnly && select > io.segs.size()) return std::unexpected("nvfz: no such segment");
  d.segs = std::move(io.segs);
  d.picked = std::move(io.picked);
  return d;
}

Packed encode(Mode mode, std::span<const uint8_t> file) {
  Packed r;
  r.data.assign(kMagic.begin(), kMagic.end());
  r.data.push_back(kFormat);
  r.data.push_back(static_cast<uint8_t>(mode));
  put_le(r.data, file.size(), 8);
  put_le(r.data, checksum(file), 8);
  Model model(file.size());
  Encoder ac(r.data);
  Io<Encoder> io(model, ac, file.data(), nullptr, file.size());
  if (!walk(mode, io) || !io.complete()) {
    if (mode == Mode::bytes) throw std::logic_error("cm: plain coding failed");
    return encode(Mode::bytes, file);
  }
  ac.flush();
  for (const Part& p : io.parts) {
    if (p.values > 0) r.parts.push_back(p);
  }
  return r;
}

std::expected<std::pair<Mode, std::vector<uint8_t>>, std::string> decode(std::span<const uint8_t> packed) {
  if (packed.size() < kHeader || !std::equal(kMagic.begin(), kMagic.end(), packed.begin())) return std::unexpected("nvfz: not a packed file");
  if (packed[4] == kFormat2) {
    auto d = decode2(packed, 0);
    if (!d) return std::unexpected(d.error());
    return std::pair{d->mode, std::move(d->file)};
  }
  if (packed[4] != kFormat) return std::unexpected("nvfz: unsupported format");
  const auto mode = static_cast<Mode>(packed[5]);
  if (packed[5] > static_cast<uint8_t>(Mode::tensors)) return std::unexpected("nvfz: unknown content");
  const uint64_t size = get_le(packed.data() + 6, 8), sum = get_le(packed.data() + 14, 8);
  if (size > kMaxSize) return std::unexpected("nvfz: implausible size");
  std::vector<uint8_t> out(static_cast<size_t>(size));
  Model model(out.size());
  Decoder ac(packed.subspan(kHeader));
  Io<Decoder> io(model, ac, nullptr, out.data(), out.size());
  if (!walk(mode, io)) return std::unexpected("nvfz: corrupt data");
  if (checksum(out) != sum) return std::unexpected("nvfz: checksum mismatch");
  return std::pair{mode, std::move(out)};
}

}  // namespace

namespace {

// A frame model or a rollout effect that loads, else plain bytes.
Mode detect(std::span<const uint8_t> file) {
  if (file.size() < 8) return Mode::bytes;
  std::ispanstream in(std::span<const char>(reinterpret_cast<const char*>(file.data()), file.size()));
  if (std::memcmp(file.data(), "NVFXMDL1", 8) == 0) return load_model(in) ? Mode::frame : Mode::bytes;
  if (rollout::is_rollout_file(std::span<const char>(reinterpret_cast<const char*>(file.data()), 8))) {
    return rollout::load_model(in) ? Mode::rollout : Mode::bytes;
  }
  return Mode::bytes;
}

}  // namespace

Packed pack_model(std::span<const uint8_t> file) {
  if (file.size() > kMaxSize) throw std::invalid_argument("pack_model: files over 256 MB are not supported");
  return encode(detect(file), file);
}

namespace {
Opt2 opt2(const Options& o) {
  Opt2 r;
  r.model = static_cast<int>(o.literal);
  if (o.seekable && r.model == 0) r.model = 1;
  r.lz = o.lz;
  r.seek = o.seekable;
  r.segment = std::max<std::size_t>(1, o.segment);
  return r;
}
}  // namespace

Packed pack_model(std::span<const uint8_t> file, const Options& o) {
  if (file.size() > kMaxSize) throw std::invalid_argument("pack_model: files over 256 MB are not supported");
  return encode2(detect(file), file, opt2(o));
}

std::expected<std::vector<SliceInfo>, std::string> list_slices(std::span<const uint8_t> packed) {
  if (packed.size() < kHeader || !std::equal(kMagic.begin(), kMagic.end(), packed.begin())) return std::unexpected("nvfz: not a packed file");
  if (packed[4] != kFormat2) return std::unexpected("nvfz: not seekable");
  const auto d = decode2(packed, kListOnly);
  if (!d) return std::unexpected(d.error());
  std::vector<SliceInfo> r;
  for (std::size_t k = 0; k < d->segs.size(); ++k) {
    const SegMeta& m = d->segs[k];
    r.push_back({m.kind, m.values, d->seg_bytes[k + 1], m.tensor, m.first, m.last - m.first});
  }
  return r;
}

std::expected<Slice, std::string> unpack_slice(std::span<const uint8_t> packed, std::size_t index) {
  if (packed.size() < kHeader || !std::equal(kMagic.begin(), kMagic.end(), packed.begin())) return std::unexpected("nvfz: not a packed file");
  if (packed[4] != kFormat2) return std::unexpected("nvfz: not seekable");
  if (index + 1 == 0) return std::unexpected("nvfz: no such segment");
  auto d = decode2(packed, index + 1);
  if (!d) return std::unexpected(d.error());
  const SegMeta& m = d->segs[index];
  Slice s;
  s.tensor.shape = m.shape;
  s.tensor.values = std::move(d->picked);
  s.first_plane = m.first;
  s.at = m.at;
  return s;
}

std::vector<std::pair<Kind, std::vector<uint8_t>>> split_model(std::span<const uint8_t> file) {
  Collect c(file);
  if (!walk(detect(file), c)) {
    Collect plain(file);
    walk(Mode::bytes, plain);
    c = std::move(plain);
  }
  std::vector<std::pair<Kind, std::vector<uint8_t>>> parts;
  for (size_t k = 0; k < c.out.size(); ++k) {
    if (!c.out[k].empty()) parts.emplace_back(static_cast<Kind>(k), std::move(c.out[k]));
  }
  return parts;
}

std::expected<std::vector<uint8_t>, std::string> unpack_model(std::span<const uint8_t> packed) {
  auto r = decode(packed);
  if (!r) return std::unexpected(r.error());
  if (r->first == Mode::tensors) return std::unexpected("nvfz: holds tensors, not a file");
  return std::move(r->second);
}

namespace {
std::vector<uint8_t> serialise(std::span<const Tensor> tensors) {
  std::vector<uint8_t> b;
  put_le(b, tensors.size(), 4);
  for (const Tensor& t : tensors) {
    const Shape& s = t.shape;
    if ((s.width != 1 && s.width != 2) || s.dims.size() > 16 || t.values.size() != s.size() || static_cast<int>(s.kind) >= kKinds) {
      throw std::invalid_argument("pack_tensors: bad shape");
    }
    const bool map = s.width == 1 && !s.lo.empty();
    if (map && (s.lo.size() != s.planes() || s.hi.size() != s.planes())) throw std::invalid_argument("pack_tensors: one map per plane");
    b.push_back(static_cast<uint8_t>(s.kind));
    b.push_back(static_cast<uint8_t>(s.width));
    b.push_back(s.channels ? 1 : 0);
    b.push_back(static_cast<uint8_t>(s.dims.size()));
    b.push_back(map ? 1 : 0);
    for (const uint32_t d : s.dims) put_le(b, d, 4);
    if (map) {
      for (size_t k = 0; k < s.lo.size(); ++k) {
        put_le(b, static_cast<uint64_t>(s.lo[k]), 8);
        put_le(b, static_cast<uint64_t>(s.hi[k]), 8);
      }
    }
    for (const uint16_t v : t.values) {
      if (s.width == 1 && v > 255) throw std::invalid_argument("pack_tensors: 8-bit value out of range");
      put_le(b, v, s.width);
    }
  }
  if (b.size() > kMaxSize) throw std::invalid_argument("pack_tensors: more than 256 MB");
  return b;
}
}  // namespace

Packed pack_tensors(std::span<const Tensor> tensors) { return encode(Mode::tensors, serialise(tensors)); }

Packed pack_tensors(std::span<const Tensor> tensors, const Options& o) { return encode2(Mode::tensors, serialise(tensors), opt2(o)); }

std::expected<std::vector<Tensor>, std::string> unpack_tensors(std::span<const uint8_t> packed) {
  auto r = decode(packed);
  if (!r) return std::unexpected(r.error());
  if (r->first != Mode::tensors) return std::unexpected("nvfz: holds a file, not tensors");
  const std::vector<uint8_t>& b = r->second;
  std::vector<Tensor> out;
  size_t pos = 4;
  const uint64_t count = get_le(b.data(), 4);
  for (uint64_t t = 0; t < count; ++t) {
    Tensor x;
    Shape& s = x.shape;
    s.kind = static_cast<Kind>(b[pos]);
    s.width = b[pos + 1];
    s.channels = b[pos + 2] != 0;
    const size_t nd = b[pos + 3];
    const bool map = b[pos + 4] != 0;
    pos += 5;
    for (size_t k = 0; k < nd; ++k, pos += 4) s.dims.push_back(static_cast<uint32_t>(get_le(b.data() + pos, 4)));
    if (map) {
      for (size_t k = 0; k < s.planes(); ++k, pos += 16) {
        s.lo.push_back(static_cast<int64_t>(get_le(b.data() + pos, 8)));
        s.hi.push_back(static_cast<int64_t>(get_le(b.data() + pos + 8, 8)));
      }
    }
    x.values.resize(s.size());
    for (uint16_t& v : x.values) {
      v = static_cast<uint16_t>(get_le(b.data() + pos, s.width));
      pos += static_cast<size_t>(s.width);
    }
    out.push_back(std::move(x));
  }
  return out;
}

}  // namespace nfx::cm
