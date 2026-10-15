// Study P's upscaler (rt_upscale.hpp), compiled once per ISA (rt_upscale_<isa>.cpp) inside namespace NFX_NS with NFX_UW
// float lanes, the ISA switched on by #pragma GCC target after the standard headers, as rt_impl.hpp does.
//
// The operations and their order per value are the reference's (src/core/upscale.cpp): a convolution starts from its bias
// and adds tap by tap, input by input; the bilinear 2x blends along x, then along y; the noise's 3 x 3 sum adds three sums
// along x along y. Under contraction (CMakeLists.txt, neuralfx_simd) the AVX2 and AVX-512 builds fuse some multiply-adds,
// so a byte can differ from the reference's by one where a value falls on a rounding boundary (tests/test_upscale.cpp).
// Vectors run along x over the half-size pixels: one vector of the half-size row gives the even and the odd full-size
// pixels of two full-size rows, interleaved when the bytes are written. The noise slices are stored with their even and
// odd columns apart for the same reason.
#if !defined(NFX_NS) || !defined(NFX_UW)
#error "define NFX_NS and NFX_UW before including rt_upscale_impl.hpp"
#endif

#include "rt_aligned.hpp"
#include "rt_upscale.hpp"

#include <neuralfx/noise.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(NFX_ARCH) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#define NFX_PRAGMA(x) _Pragma(#x)
#define NFX_TARGET(a) NFX_PRAGMA(GCC target(a))
NFX_TARGET(NFX_ARCH)
#define NFX_PUSHED 1
#endif

namespace nfx::rt::NFX_NS {

namespace {

constexpr int kL = NFX_UW;  // lanes
typedef float vf __attribute__((vector_size(kL * 4)));
typedef float vfu __attribute__((vector_size(kL * 4), aligned(4)));
typedef std::int32_t vi __attribute__((vector_size(kL * 4)));
typedef std::uint32_t vu __attribute__((vector_size(kL * 4)));
typedef std::uint32_t vuu __attribute__((vector_size(kL * 4), aligned(4)));

inline std::size_t z(int v) { return static_cast<std::size_t>(v); }
inline vf ld(const float* p) { return *reinterpret_cast<const vfu*>(p); }
inline void st(float* p, vf v) { *reinterpret_cast<vfu*>(p) = v; }
inline vf bc(float x) { return x - vf{}; }
inline vf vmax(vf a, vf b) { return a > b ? a : b; }
inline vf vmin(vf a, vf b) { return a < b ? a : b; }
inline vu hash32v(vu x) {  // nfx::hash32 on lanes
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}
template <std::size_t... L>
inline vu iota_(std::index_sequence<L...>) {
  return vu{static_cast<std::uint32_t>(L)...};
}
inline vu lanes() { return iota_(std::make_index_sequence<kL>{}); }
// The even and the odd lanes of a and b, in order.
template <std::size_t... L>
inline vf evens(vf a, vf b, std::index_sequence<L...>) {
  return __builtin_shufflevector(a, b, static_cast<int>(2 * L)...);
}
template <std::size_t... L>
inline vf odds(vf a, vf b, std::index_sequence<L...>) {
  return __builtin_shufflevector(a, b, static_cast<int>(2 * L + 1)...);
}

// Half H of the lanes of a and b interleaved (a0, b0, a1, b1, ...).
template <int H, std::size_t... L>
inline vu zip_(vu a, vu b, std::index_sequence<L...>) {
  return __builtin_shufflevector(a, b, (H * kL / 2 + static_cast<int>(L / 2) + (L % 2 ? kL : 0))...);
}

class UpscalerImpl final : public Upscaler {
 public:
  UpscalerImpl(const upscale::Net* net, int n) : net_(net), n_(n) {
    if (n < kL || n % kL != 0 || n % 16 != 0 || n > 512) throw std::invalid_argument("upscaler: the half size must be a multiple of 16 up to 512");
    const int P = n + 2;
    plane_.assign(4 * z(P) * z(P), 0.f);
    blocks_ = n / kL;
    any_.assign(z(n) * z(blocks_), 0);
    live_.assign(z(blocks_), 0);
    for (auto& r : he_) r.assign(4 * z(n), 0.f);
    for (auto& r : ho_) r.assign(4 * z(n), 0.f);
    if (!net_) return;
    if (net_->w.size() != net_->layout().size) throw std::invalid_argument("upscaler: weights do not match the layout");
    L_ = net_->layout();
    c1_ = net_->c1;
    c2_ = net_->c2;
    k2_ = net_->k2;
    // the first layer's rows, padded by one half-size pixel on each side (zero), for k2 = 3: three of them
    for (auto& r : h1_) r.assign(z(c1_) * z(P), 0.f);
    h2_.assign(z(c2_) * kL, 0.f);
    out_.assign(upscale::kOut * z(n), 0.f);
    if (net_->gain > 0.f) {
      const std::size_t N = 2 * z(n);
      for (auto& s : slice_) s.assign(N * N, 0.f);  // [row][even columns | odd columns]
      wrow_.assign(3 * (N + 2 * kL), 0.f);
      hrow_.assign(3 * N, 0.f);
    }
  }

  void run(const std::uint8_t* lo, std::size_t lo_stride, std::uint64_t seed, double seconds, const FrameInput& in, std::uint8_t* hi,
           std::size_t hi_stride) override {
    const int n = n_, P = n + 2;
    // the half-size frame as planar floats with a zero border
    for (int y = 0; y < n; ++y) {
      const std::uint8_t* row = lo + lo_stride * z(y);
      for (int x = 0; x < n; x += kL) {
        vu px;
        std::memcpy(&px, row + 4 * z(x), sizeof px);
        for (int c = 0; c < 4; ++c) {
          const vu v = (px >> (8 * c)) & 0xffU;
          st(plane_.data() + (z(c) * z(P) + z(y + 1)) * z(P) + 1 + z(x), __builtin_convertvector(__builtin_bit_cast(vi, v), vf) * (1.f / 255.f));
        }
      }
    }
    // which blocks of each half-size row hold anything, with a pixel of margin on each side (the bilinear 2x and the
    // network of a block read no further for the pixels a block writes): an empty block of two full-size rows is written
    // as zeros, which is what it is, and its network is not run
    for (int y = 0; y < n; ++y) {
      const std::uint8_t* row = lo + lo_stride * z(y);
      for (int b = 0; b < blocks_; ++b) {
        const int x0 = std::max(0, b * kL - 1), x1 = std::min(n, (b + 1) * kL + 1);
        std::uint32_t acc = 0;
        for (int x = x0; x < x1; ++x) {
          std::uint32_t v;
          std::memcpy(&v, row + 4 * z(x), 4);
          acc |= v;
        }
        any_[z(y) * z(blocks_) + z(b)] = acc != 0;
      }
    }
    const bool noise = net_ && net_->gain > 0.f;
    upscale::NoiseBlend nb;
    if (noise) {
      nb = upscale::noise_blend(seconds, net_->rate);
      slices(seed, nb.k);
    }
    if (net_ && k2_ == 3) {
      std::fill(h1_[0].begin(), h1_[0].end(), 0.f);  // the row above the frame
      layer1(0, h1_[1].data());
    }
    for (int y = 0; y < n; ++y) {
      // the bilinear rows along x: y - 1, y, y + 1 (clamped), kept in a ring of three
      if (y == 0) {
        horizontal(0, 0);
        horizontal(std::min(1, n - 1), 1);
      } else if (y + 1 < n) {
        horizontal(y + 1, (y + 1) % 3);
      }
      const int rm = std::max(y - 1, 0) % 3, r0 = y % 3, rp = std::min(y + 1, n - 1) % 3;
      for (int b = 0; b < blocks_; ++b) {
        live_[z(b)] = any_[z(y) * z(blocks_) + z(b)] | (y > 0 ? any_[z(y - 1) * z(blocks_) + z(b)] : 0) |
                      (y + 1 < n ? any_[z(y + 1) * z(blocks_) + z(b)] : 0);
      }
      if (net_) network_row(y);
      for (int dy = 0; dy < 2; ++dy) {
        const int ro = dy ? rp : rm;
        write_row(y, dy, r0, ro, noise ? &nb : nullptr, in, hi + hi_stride * z(2 * y + dy));
      }
    }
  }

  std::size_t scratch_bytes() const override {
    std::size_t b = 4 * (plane_.size() + h2_.size() + out_.size() + wrow_.size() + hrow_.size()) + any_.size() + live_.size();
    for (const auto& r : he_) b += 4 * r.size();
    for (const auto& r : ho_) b += 4 * r.size();
    for (const auto& r : h1_) b += 4 * r.size();
    for (const auto& s : slice_) b += 4 * s.size();
    return b;
  }

 private:
  const float* plane(int c, int y) const { return plane_.data() + (z(c) * z(n_ + 2) + z(y + 1)) * z(n_ + 2) + 1; }

  // Row y of the bilinear 2x along x into slot s: even[x] = 0.75 L[x] + 0.25 L[x - 1], odd[x] = 0.75 L[x] + 0.25 L[x + 1],
  // clamped at the ends.
  void horizontal(int y, int s) {
    const int n = n_;
    for (int c = 0; c < 4; ++c) {
      const float* p = plane(c, y);
      float* e = he_[z(s)].data() + z(c) * z(n);
      float* o = ho_[z(s)].data() + z(c) * z(n);
      for (int x = 0; x < n; x += kL) {
        const vf m = ld(p + x);
        st(e + x, 0.75f * m + 0.25f * ld(p + x - 1));
        st(o + x, 0.75f * m + 0.25f * ld(p + x + 1));
      }
      e[0] = 0.75f * p[0] + 0.25f * p[0];
      o[n - 1] = 0.75f * p[n - 1] + 0.25f * p[n - 1];
    }
  }

  // The first layer on half-size row y (3 x 3, 4 -> c1, ReLU) into a padded row [c1][n + 2] (border zero).
  void layer1(int y, float* dst) {
    const float* w = net_->w.data();
    const int n = n_, P = n + 2;
    for (int x = 0; x < n; x += kL) {
      for (int j = 0; j < c1_; ++j) {
        vf a = bc(w[L_.b1 + z(j)]);
        for (int t = 0; t < 9; ++t) {
          const int dx = t % 3 - 1, dy = t / 3 - 1;
          for (int i = 0; i < upscale::kIn; ++i) a += w[L_.w1 + (z(t) * upscale::kIn + z(i)) * z(c1_) + z(j)] * ld(plane(i, y + dy) + x + dx);
        }
        st(dst + z(j) * z(P) + 1 + z(x), vmax(a, vf{}));
      }
    }
  }

  // The network's 20 outputs for half-size row y, planar: out_[k][x]. The widths of the shipped sizes are compiled in, so
  // that a block's activations stay in registers; any other runs the general loops below.
  void network_row(int y) {
    if (k2_ == 1) {
      if (c1_ == 8 && c2_ == 8) return fused<8, 8>(y);
      if (c1_ == 12 && c2_ == 8) return fused<12, 8>(y);
      if (c1_ == 16 && c2_ == 8) return fused<16, 8>(y);
      if (c1_ == 12 && c2_ == 12) return fused<12, 12>(y);
      if (c1_ == 16 && c2_ == 16) return fused<16, 16>(y);
    }
    general_row(y);
  }

  // The first layer, the second (1 x 1) and the output for a block of lanes, the activations in registers. Per value the
  // operations and their order are the general loops' (the reference's).
  template <int C1, int C2>
  void fused(int y) {
    const float* w = net_->w.data();
    const float* w1 = w + L_.w1;
    const float* b1 = w + L_.b1;
    const float* w2 = w + L_.w2;
    const float* b2 = w + L_.b2;
    const float* wo = w + L_.wo;
    const float* bo = w + L_.bo;
    const int n = n_;
    const float* rows[3][upscale::kIn];
    for (int dy = 0; dy < 3; ++dy) {
      for (int i = 0; i < upscale::kIn; ++i) rows[dy][i] = plane(i, y + dy - 1);
    }
    for (int x = 0; x < n; x += kL) {
      if (!live_[z(x / kL)]) continue;
      vf h1[C1];
#pragma GCC unroll 64
      for (int j = 0; j < C1; ++j) h1[j] = bc(b1[j]);
#pragma GCC unroll 9
      for (int t = 0; t < 9; ++t) {
        const int dx = t % 3 - 1, dy = t / 3;
#pragma GCC unroll 4
        for (int i = 0; i < upscale::kIn; ++i) {
          const vf v = ld(rows[dy][i] + x + dx);
          const float* wt = w1 + (t * upscale::kIn + i) * C1;
#pragma GCC unroll 64
          for (int j = 0; j < C1; ++j) h1[j] += wt[j] * v;
        }
      }
#pragma GCC unroll 64
      for (int i = 0; i < C1; ++i) h1[i] = vmax(h1[i], vf{});
      vf h2[C2];
#pragma GCC unroll 64
      for (int j = 0; j < C2; ++j) {  // one sum at a time, the inputs in registers
        vf a = bc(b2[j]);
#pragma GCC unroll 64
        for (int i = 0; i < C1; ++i) a += w2[i * C2 + j] * h1[i];
        h2[j] = vmax(a, vf{});
      }
#pragma GCC unroll 20
      for (int k = 0; k < upscale::kOut; ++k) {
        vf a = bc(bo[k]);
#pragma GCC unroll 64
        for (int i = 0; i < C2; ++i) a += wo[i * upscale::kOut + k] * h2[i];
        st(out_.data() + z(k) * z(n) + z(x), k >= upscale::kResid ? vmax(a, vf{}) : a);
      }
    }
  }

  // Any widths and kernels.
  void general_row(int y) {
    const float* w = net_->w.data();
    const int n = n_, P = n + 2;
    const float* rows[3] = {nullptr, nullptr, nullptr};
    if (k2_ == 1) {
      layer1(y, h1_[0].data());
      rows[1] = h1_[0].data();
    } else {  // the first layer's rows y - 1, y, y + 1 in slots 0, 1, 2 (run() put the zero row and row 0 in slots 0 and 1)
      if (y + 1 < n) layer1(y + 1, h1_[2].data());
      else std::fill(h1_[2].begin(), h1_[2].end(), 0.f);
      rows[0] = h1_[0].data();
      rows[1] = h1_[1].data();
      rows[2] = h1_[2].data();
    }
    const int r2 = k2_ / 2;
    for (int x = 0; x < n; x += kL) {
      if (!live_[z(x / kL)]) continue;
      for (int j = 0; j < c2_; ++j) {
        vf a = bc(w[L_.b2 + z(j)]);
        for (int t = 0; t < k2_ * k2_; ++t) {
          const int dx = t % k2_ - r2, dy = t / k2_ - r2;
          if (y + dy < 0 || y + dy >= n) continue;  // the reference skips taps outside the frame
          const float* h = rows[1 + dy] + 1 + x + dx;
          for (int i = 0; i < c1_; ++i) a += w[L_.w2 + (z(t) * z(c1_) + z(i)) * z(c2_) + z(j)] * ld(h + z(i) * z(P));
        }
        st(h2_.data() + z(j) * kL, vmax(a, vf{}));
      }
      for (int k = 0; k < upscale::kOut; ++k) {
        vf a = bc(w[L_.bo + z(k)]);
        for (int i = 0; i < c2_; ++i) a += w[L_.wo + z(i) * upscale::kOut + z(k)] * ld(h2_.data() + z(i) * kL);
        if (k >= upscale::kResid) a = vmax(a, vf{});
        st(out_.data() + z(k) * z(n) + z(x), a);
      }
    }
    if (k2_ == 3) {  // slide the rows: y becomes the row above, y + 1 the row (slot 2 is overwritten next time)
      std::swap(h1_[0], h1_[1]);
      std::swap(h1_[1], h1_[2]);
    }
  }

  // Noise slices k and k + 1 in slice_[0] and slice_[1] (kept from the last frame when they are the same).
  void slices(std::uint64_t seed, std::int64_t k) {
    const std::uint64_t key = upscale::noise_seed(seed);
    if (have_ && key == key_ && k == k_) return;
    if (have_ && key == key_ && k == k_ + 1) {
      std::swap(slice_[0], slice_[1]);
      make_slice(seed, k + 1, slice_[1].data());
    } else {
      make_slice(seed, k, slice_[0].data());
      make_slice(seed, k + 1, slice_[1].data());
    }
    have_ = true;
    key_ = key;
    k_ = k;
  }

  // White noise rows (columns -1 to 2n) and their sums along x, three rows in rings; each output row of the slice is the
  // white value minus the 3 x 3 mean, times kNoiseNorm, its even and odd columns apart.
  void make_slice(std::uint64_t seed, std::int64_t k, float* dst) {
    const int N = 2 * n_, Wr = N + 2 * kL;
    const std::uint32_t key = upscale::noise_key(seed, k);
    const auto white_row = [&](int Y, float* w) {  // w[X + kL] for X in [-kL, N + kL)
      const std::uint32_t row = hash32(key ^ static_cast<std::uint32_t>(Y) * 0xd8163841U);
      for (int X = -kL; X < N + kL; X += kL) {
        const vu xs = (lanes() + static_cast<std::uint32_t>(X)) * 0x8da6b343U;
        const vu h = hash32v(xs ^ row) >> 8;
        const vf v = __builtin_convertvector(__builtin_bit_cast(vi, h), vf) * (2.f / 16777215.f) - 1.f;
        st(w + X + kL, v);
      }
    };
    const auto sum_row = [&](const float* w, float* h) {
      for (int X = 0; X < N; X += kL) st(h + X, (ld(w + kL + X - 1) + ld(w + kL + X)) + ld(w + kL + X + 1));
    };
    float* W0 = wrow_.data();
    float* H0 = hrow_.data();
    const auto wr = [&](int Y) { return W0 + z(((Y % 3) + 3) % 3) * z(Wr); };
    const auto hr = [&](int Y) { return H0 + z(((Y % 3) + 3) % 3) * z(N); };
    for (int Y = -1; Y <= 0; ++Y) {
      white_row(Y, wr(Y));
      sum_row(wr(Y), hr(Y));
    }
    for (int Y = 0; Y < N; ++Y) {
      white_row(Y + 1, wr(Y + 1));
      sum_row(wr(Y + 1), hr(Y + 1));
      const float* w = wr(Y) + kL;
      const float *a = hr(Y - 1), *b = hr(Y), *c = hr(Y + 1);
      float* e = dst + z(Y) * z(N);
      float* o = e + z(n_);
      for (int X = 0; X < N; X += 2 * kL) {
        const vf v0 = (ld(w + X) - ((ld(a + X) + ld(b + X)) + ld(c + X)) * (1.f / 9.f)) * upscale::kNoiseNorm;
        const vf v1 = (ld(w + X + kL) - ((ld(a + X + kL) + ld(b + X + kL)) + ld(c + X + kL)) * (1.f / 9.f)) * upscale::kNoiseNorm;
        st(e + X / 2, evens(v0, v1, std::make_index_sequence<kL>{}));
        st(o + X / 2, odds(v0, v1, std::make_index_sequence<kL>{}));
      }
    }
  }

  // Full-size row 2 y + dy: the bilinear 2x (half-size row y, slot r0, blended with row y -/+ 1, slot ro), the residuals
  // and the micro-detail, the colour matrix, bytes.
  void write_row(int y, int dy, int r0, int ro, const upscale::NoiseBlend* nb, const FrameInput& in, std::uint8_t* dst) {
    const int n = n_, N = 2 * n;
    const float gain = net_ ? net_->gain : 0.f;
    const float* ea = nullptr;
    const float* eb = nullptr;
    if (nb) {
      ea = slice_[0].data() + z(2 * y + dy) * z(N);
      eb = slice_[1].data() + z(2 * y + dy) * z(N);
    }
    for (int x = 0; x < n; x += kL) {
      if (!live_[z(x / kL)]) {
        std::memset(dst + 8 * z(x), 0, 8 * kL);
        continue;
      }
      vu px[2]{};
      vi empty[2];
      for (int h = 0; h < 2; ++h) {  // even and odd full-size columns
        const auto& H0 = h ? ho_ : he_;
        vf v[4];
        vi zero = vi{} == vi{};
        for (int c = 0; c < 4; ++c) {
          const vf up = 0.75f * ld(H0[z(r0)].data() + z(c) * z(n) + z(x)) + 0.25f * ld(H0[z(ro)].data() + z(c) * z(n) + z(x));
          zero &= up == 0.f;
          v[c] = up;
        }
        empty[h] = zero;
        if (net_) {
          const int s = dy * 2 + h;
          vf e{};
          if (nb) {
            const std::size_t off = z(h) * z(n) + z(x);
            e = gain * (((1.f - nb->s) * ld(ea + off) + nb->s * ld(eb + off)) * nb->norm);
          }
          for (int c = 0; c < 4; ++c) {
            const vf r = ld(out_.data() + z(s * 4 + c) * z(n) + z(x));
            const vf a = ld(out_.data() + z(upscale::kResid + c) * z(n) + z(x));
            v[c] += r + a * e;
          }
        }
        if (in.apply_colour) {
          const auto& M = in.colour;
          const vf r = v[0], g = v[1], b = v[2];
          v[0] = M[0] * r + M[1] * g + M[2] * b;
          v[1] = M[3] * r + M[4] * g + M[5] * b;
          v[2] = M[6] * r + M[7] * g + M[8] * b;
        }
        vu q{};
        for (int c = 0; c < 4; ++c) {
          const vf u = vmin(vmax(v[c], vf{}), bc(1.f));
          q |= __builtin_bit_cast(vu, __builtin_convertvector(u * 255.f + 0.5f, vi)) << (8 * c);
        }
        px[h] = empty[h] ? vu{} : q;
      }
      const vu lo = zip_<0>(px[0], px[1], std::make_index_sequence<kL>{});
      const vu hi = zip_<1>(px[0], px[1], std::make_index_sequence<kL>{});
      std::memcpy(dst + 8 * z(x), &lo, sizeof lo);
      std::memcpy(dst + 8 * z(x) + sizeof lo, &hi, sizeof hi);
    }
  }

  const upscale::Net* net_;
  int n_;
  upscale::Net::Layout L_{};
  int c1_ = 0, c2_ = 0, k2_ = 1;
  AlignedFloats plane_;               // [4][n + 2][n + 2], zero border
  int blocks_ = 0;                    // blocks of lanes in a half-size row
  std::vector<std::uint8_t> any_, live_;  // a block (with a pixel of margin) holds anything: per row; this row's, with its neighbours
  std::array<AlignedFloats, 3> he_, ho_;  // bilinear rows along x, even and odd columns, [4][n] each
  std::array<AlignedFloats, 3> h1_;   // the first layer's rows, [c1][n + 2]
  AlignedFloats h2_, out_;            // a block of the second layer [c2][lanes]; the outputs of a row [20][n]
  std::array<AlignedFloats, 2> slice_;  // noise slices k and k + 1, [2n rows][n even | n odd]
  AlignedFloats wrow_, hrow_;           // the slices' white rows and sums along x (rings of three)
  bool have_ = false;
  std::uint64_t key_ = 0;
  std::int64_t k_ = 0;
};

}  // namespace

std::unique_ptr<Upscaler> make_upscaler(const upscale::Net* net, int n) { return std::make_unique<UpscalerImpl>(net, n); }

}  // namespace nfx::rt::NFX_NS

#if defined(NFX_PUSHED)
#pragma GCC pop_options
#undef NFX_PUSHED
#endif
