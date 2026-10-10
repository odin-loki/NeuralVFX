// The prior against drift (rt_prior.hpp), compiled once per ISA (rt_prior_<isa>.cpp) inside namespace NFX_NS, with
// the ISA switched on by #pragma GCC target after the standard headers, as rt_impl.hpp does.
//
// The forward pass of the denoiser (src/dcm/ddpm.cpp, forward) for inference only. Every value is computed with the
// reference's operations in the reference's order: a 3 x 3 convolution starts from the bias and adds tap by tap, input
// channel by input channel, one multiply-add each; FiLM is h (1 + gamma) + beta; SiLU is x / (1 + exp(-x)) with the
// same vectorised exponential; the residual is added after the convolution. Under contraction (CMakeLists.txt,
// neuralfx_simd) with FMA (AVX2, AVX-512) each multiply-add is fused exactly where the reference fuses it, so the
// floats are the reference's bit for bit; the baseline build has no FMA and rounds twice. What differs is
// bookkeeping only: padded buffers keep their zero border and are written inside, SiLU writes straight into them, the
// position channels are written once, and the output layer's weights are widened to 8 channels once, not per pass.
// Every buffer is allocated in the constructor.
#if !defined(NFX_NS)
#error "define NFX_NS before including rt_prior_impl.hpp"
#endif

#include "rt_prior.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <span>
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

inline std::size_t z(int v) { return static_cast<std::size_t>(v); }
inline float fl(int v) { return static_cast<float>(v); }

// --- the reference's kernels (src/dcm/ddpm_kernels.hpp), forward only ------------------------------------------------

typedef float v8 __attribute__((vector_size(32)));
typedef float v8u __attribute__((vector_size(32), aligned(4)));
typedef int v8i __attribute__((vector_size(32)));

inline v8 load8(const float* p) { return *reinterpret_cast<const v8u*>(p); }
inline void store8(float* p, v8 v) { *reinterpret_cast<v8u*>(p) = v; }
inline v8 splat(float a) { return v8{a, a, a, a, a, a, a, a}; }

// e^x (x clamped to [-87, 87]): x = n ln 2 + r, e^r by its Taylor polynomial of degree 7, 2^n through the exponent
// bits (the reference's exp8, operation for operation).
inline v8 exp8(v8 x) {
  const v8 lo = splat(-87.f), hi = splat(87.f);
  x = x < lo ? lo : x;
  x = x > hi ? hi : x;
  const v8 magic = splat(12582912.f);
  const v8 n = (x * splat(1.44269504088896341f) + magic) - magic;
  const v8 r = (x - n * splat(0.693145751953125f)) - n * splat(1.42860682030941723212e-6f);
  v8 p = splat(1.f / 5040.f);
  p = p * r + splat(1.f / 720.f);
  p = p * r + splat(1.f / 120.f);
  p = p * r + splat(1.f / 24.f);
  p = p * r + splat(1.f / 6.f);
  p = p * r + splat(0.5f);
  p = p * r + splat(1.f);
  p = p * r + splat(1.f);
  const v8i e = (__builtin_convertvector(n, v8i) + 127) << 23;
  v8 scale;
  std::memcpy(&scale, &e, sizeof(scale));
  return p * scale;
}

inline v8 load_n(const float* p, std::size_t n) {
  if (n == 8) return load8(p);
  v8 v{};
  for (std::size_t k = 0; k < n; ++k) v[k] = p[k];
  return v;
}
inline void store_n(float* p, v8 v, std::size_t n) {
  if (n == 8) return store8(p, v);
  for (std::size_t k = 0; k < n; ++k) p[k] = v[k];
}

// y = x * sigmoid(x) = x / (1 + e^-x), on blocks of eight (the tail through a zero-padded block). Each value is
// computed alone, so how a row is cut into blocks does not change it.
inline void silu_n(const float* x, float* y, std::size_t n) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const v8 v = load8(x + i);
    store8(y + i, v / (splat(1.f) + exp8(-v)));
  }
  if (i < n) {
    const v8 v = load_n(x + i, n - i);
    store_n(y + i, v / (splat(1.f) + exp8(-v)), n - i);
  }
}

// SiLU of an R^2 x C map written inside a padded (R + 2)^2 x C buffer whose border is zero and stays zero.
inline void silu_pad(int R, const float* in, int C, float* pad) {
  const std::size_t row = z(R) * z(C);
  for (int y = 0; y < R; ++y) silu_n(in + z(y) * row, pad + (z(y + 1) * z(R + 2) + 1) * z(C), row);
}

// NV output vectors (8 channels each) starting at k0, for cells (x, y) and (x + 1, y) when two is set: the
// reference's conv3_block.
template <int NV>
inline void conv3_block(int R, const float* __restrict ip, int ci, const float* __restrict W, const float* b, int co, int k0, int x, int y,
                        bool two, float* __restrict out) {
  const int P = R + 2;
  v8 a0[NV], a1[NV];
  for (int v = 0; v < NV; ++v) {
    a0[v] = b ? load8(b + k0 + 8 * v) : v8{};
    a1[v] = a0[v];
  }
  for (int tap = 0; tap < 9; ++tap) {
    const int dy = tap / 3, dx = tap % 3;
    const float* p0 = ip + (z(y + dy) * z(P) + z(x + dx)) * z(ci);
    const float* p1 = p0 + ci;
    const float* w = W + z(tap) * z(ci) * z(co) + z(k0);
    if (two) {
      for (int c = 0; c < ci; ++c) {
        const v8 s0 = splat(p0[c]), s1 = splat(p1[c]);
        const float* wr = w + z(c) * z(co);
        for (int v = 0; v < NV; ++v) {
          const v8 wv = load8(wr + 8 * v);
          a0[v] += s0 * wv;
          a1[v] += s1 * wv;
        }
      }
    } else {
      for (int c = 0; c < ci; ++c) {
        const v8 s0 = splat(p0[c]);
        const float* wr = w + z(c) * z(co);
        for (int v = 0; v < NV; ++v) a0[v] += s0 * load8(wr + 8 * v);
      }
    }
  }
  float* o0 = out + (z(y) * z(R) + z(x)) * z(co) + z(k0);
  for (int v = 0; v < NV; ++v) store8(o0 + 8 * v, a0[v]);
  if (two) {
    float* o1 = o0 + co;
    for (int v = 0; v < NV; ++v) store8(o1 + 8 * v, a1[v]);
  }
}

// out (R^2 x co) = b + conv3(in), ip the padded input ((R + 2)^2 x ci), co a multiple of 8, weights [tap][in][out].
inline void conv3_fwd(int R, const float* ip, int ci, const float* W, const float* b, int co, float* out) {
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; x += 2) {
      const bool two = x + 1 < R;
      int k0 = 0;
      for (; k0 + 32 <= co; k0 += 32) conv3_block<4>(R, ip, ci, W, b, co, k0, x, y, two, out);
      for (; k0 + 16 <= co; k0 += 16) conv3_block<2>(R, ip, ci, W, b, co, k0, x, y, two, out);
      for (; k0 < co; k0 += 8) conv3_block<1>(R, ip, ci, W, b, co, k0, x, y, two, out);
    }
  }
}

// out (n x co) = b + in W, W [ci][co]: the reference's conv1_fwd.
inline void conv1_fwd(int n, const float* in, int ci, const float* W, const float* b, int co, float* out) {
  for (int i = 0; i < n; ++i) {
    float* __restrict o = out + z(i) * z(co);
    for (int k = 0; k < co; ++k) o[k] = b[k];
    const float* a = in + z(i) * z(ci);
    for (int c = 0; c < ci; ++c) {
      const float av = a[c];
      const float* __restrict wr = W + z(c) * z(co);
      for (int k = 0; k < co; ++k) o[k] += wr[k] * av;
    }
  }
}

// R x R -> R/2 x R/2, mean of each 2 x 2 block.
inline void avgpool(int R, const float* in, int C, float* out) {
  const int r = R / 2;
  for (int y = 0; y < r; ++y) {
    for (int x = 0; x < r; ++x) {
      float* o = out + (z(y) * z(r) + z(x)) * z(C);
      const float* a = in + (z(2 * y) * z(R) + z(2 * x)) * z(C);
      const float* b = a + C;
      const float* c = a + z(R) * z(C);
      const float* d = c + C;
      for (int k = 0; k < C; ++k) o[k] = 0.25f * (a[k] + b[k] + c[k] + d[k]);
    }
  }
}

// r x r -> 2r x 2r nearest, added to out.
inline void upsample_add(int r, const float* in, int C, float* out) {
  const int R = 2 * r;
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const float* s = in + (z(y / 2) * z(r) + z(x / 2)) * z(C);
      float* o = out + (z(y) * z(R) + z(x)) * z(C);
      for (int k = 0; k < C; ++k) o[k] += s[k];
    }
  }
}

// --- the network ------------------------------------------------------------------------------------------------------

class PriorImpl final : public Prior {
 public:
  explicit PriorImpl(const PriorNet& n) : n_(n) {
    R0_ = n.res;
    R1_ = R0_ / 2;
    R2_ = R0_ / 4;
    const auto N = [](int R) { return z(R) * z(R); };
    const auto P = [](int R) { return z(R + 2) * z(R + 2); };
    const int I = n.inputs(), ch = n.channels;
    emb_.assign(z(n.embed()), 0.f);
    mh_.assign(z(n.film_hidden), 0.f);
    ma_.assign(z(n.film_hidden), 0.f);
    film_.assign(n.film_size, 0.f);
    xpad_.assign(P(R0_) * z(I), 0.f);
    for (int y = 0; y < R0_; ++y) {  // the position channels, written once
      for (int x = 0; x < R0_; ++x) {
        float* o = xpad_.data() + (z(y + 1) * z(R0_ + 2) + z(x + 1)) * z(I);
        o[ch] = (fl(x) + 0.5f) / fl(R0_) * 2.f - 1.f;
        o[ch + 1] = (fl(y) + 0.5f) / fl(R0_) * 2.f - 1.f;
      }
    }
    for (int l = 0; l < 3; ++l) {
      const int R = l == 0 ? R0_ : (l == 1 ? R1_ : R2_), C = l == 0 ? n.c0 : (l == 1 ? n.c1 : n.c2);
      pad_[z(l)].assign(P(R) * z(C), 0.f);
      act_[z(l)].assign(N(R) * z(C), 0.f);
      h_[z(l)].assign(N(R) * z(C), 0.f);
    }
    stem_.assign(N(R0_) * z(n.c0), 0.f);
    s0_.assign(N(R0_) * z(n.c0), 0.f);
    s1_.assign(N(R1_) * z(n.c1), 0.f);
    pool0_.assign(N(R1_) * z(n.c0), 0.f);
    pool1_.assign(N(R2_) * z(n.c1), 0.f);
    d1in_.assign(N(R1_) * z(n.c1), 0.f);
    d2in_.assign(N(R2_) * z(n.c2), 0.f);
    mm_.assign(N(R2_) * z(n.c2), 0.f);
    mout_.assign(N(R2_) * z(n.c2), 0.f);
    u2c_.assign(N(R2_) * z(n.c1), 0.f);
    u1in_.assign(N(R1_) * z(n.c1), 0.f);
    dec1_.assign(N(R1_) * z(n.c1), 0.f);
    u1c_.assign(N(R1_) * z(n.c0), 0.f);
    u0in_.assign(N(R0_) * z(n.c0), 0.f);
    dec0_.assign(N(R0_) * z(n.c0), 0.f);
    // The output layer widened to a multiple of 8 channels with zeros, as the reference's conv3_fwd widens it per call.
    c8_ = (ch + 7) / 8 * 8;
    out_w8_.assign(9 * z(n.c0) * z(c8_), 0.f);
    out_b8_.assign(z(c8_), 0.f);
    for (int tc = 0; tc < 9 * n.c0; ++tc) std::copy_n(n.w.data() + n.out_w + z(tc) * z(ch), ch, out_w8_.data() + z(tc) * z(c8_));
    std::copy_n(n.w.data() + n.out_b, ch, out_b8_.data());
    y8_.assign(N(R0_) * z(c8_), 0.f);
    const std::size_t state = N(R0_) * z(ch);
    x_.assign(state, 0.f);
    xt_.assign(state, 0.f);
    eps_.assign(state, 0.f);
  }

  void predict(std::span<const float> xt, int t, std::span<const float> cond, std::span<float> eps) override {
    forward(xt.data(), t, cond);
    const int ch = n_.channels;
    for (int i = 0; i < R0_ * R0_; ++i) std::copy_n(y8_.data() + z(i) * z(c8_), ch, eps.data() + z(i) * z(ch));
  }

  void step(std::span<float> x, int t, float beta, std::span<const float> cond) override {
    const double ab = prior_alpha_bar(n_.timesteps, t);
    const float sa = static_cast<float>(std::sqrt(ab));
    const float r = static_cast<float>(std::sqrt((1.0 - ab) / ab));
    for (std::size_t i = 0; i < x.size(); ++i) xt_[i] = sa * x[i];
    predict(xt_, t, cond, eps_);
    const float* e = eps_.data();
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = (1.f - beta) * x[i] + beta * (x[i] - r * e[i]);
    const std::size_t ch = z(n_.channels);
    for (std::size_t i = 0; i < x.size(); ++i) {
      const std::size_t q = i % ch;
      x[i] = std::clamp(x[i], n_.lo[q], n_.hi[q]);
    }
  }

  void apply(std::span<float> coarse, int stride, std::span<const float> phys_lo, std::span<const float> phys_hi, int t, float beta,
             std::span<const float> cond) override {
    const int ch = n_.channels, cells = R0_ * R0_;
    for (int i = 0; i < cells; ++i) {
      for (int k = 0; k < ch; ++k) x_[z(i) * z(ch) + z(k)] = coarse[z(i) * z(stride) + z(k)] / n_.scale[z(k)];
    }
    step(x_, t, beta, cond);
    for (int i = 0; i < cells; ++i) {
      for (int k = 0; k < ch; ++k) coarse[z(i) * z(stride) + z(k)] = x_[z(i) * z(ch) + z(k)] * n_.scale[z(k)];
    }
    const std::size_t clamped = std::min({z(ch), phys_lo.size(), phys_hi.size()});
    for (std::size_t i = 0; i < coarse.size(); ++i) {
      const std::size_t k = i % z(stride);
      if (k < clamped) coarse[i] = std::clamp(coarse[i], phys_lo[k], phys_hi[k]);
    }
  }

  std::size_t scratch_bytes() const override {
    std::size_t n = 0;
    for (const auto* v : {&emb_, &mh_, &ma_, &film_, &xpad_, &stem_, &s0_, &s1_, &pool0_, &pool1_, &d1in_, &d2in_, &mm_, &mout_, &u2c_, &u1in_, &dec1_,
                          &u1c_, &u0in_, &dec0_, &out_w8_, &out_b8_, &y8_, &x_, &xt_, &eps_}) {
      n += v->size();
    }
    for (int l = 0; l < 3; ++l) n += pad_[z(l)].size() + act_[z(l)].size() + h_[z(l)].size();
    return 4 * n;
  }

 private:
  // A residual block at level l: out = in + conv3(SiLU(FiLM(conv3(SiLU(in))))).
  void block(int k, int l, const float* in, float* out) {
    const PriorNet::Block& B = n_.blocks[z(k)];
    const float* w = n_.w.data();
    const int C = B.width, R = B.side;
    const std::size_t n = z(R) * z(R) * z(C);
    float* pad = pad_[z(l)].data();
    float* act = act_[z(l)].data();
    float* h = h_[z(l)].data();
    silu_pad(R, in, C, pad);
    conv3_fwd(R, pad, C, w + B.wa, w + B.ba, C, h);
    const float* gamma = film_.data() + B.film;
    const float* beta = gamma + C;
    for (std::size_t i = 0; i < n; i += z(C)) {
      for (int c = 0; c < C; ++c) act[i + z(c)] = h[i + z(c)] * (1.f + gamma[c]) + beta[c];
    }
    silu_pad(R, act, C, pad);
    conv3_fwd(R, pad, C, w + B.wb, w + B.bb, C, out);
    for (std::size_t i = 0; i < n; ++i) out[i] += in[i];
  }

  // y8_ = eps_hat(x, t), widened to c8_ channels.
  void forward(const float* x, int t, std::span<const float> cond) {
    const PriorNet& n = n_;
    const float* w = n.w.data();
    const int F = n.freqs, H = n.film_hidden, E = n.embed(), I = n.inputs(), ch = n.channels;
    for (int j = 0; j < F; ++j) {
      const double f = std::exp(-std::log(10000.0) * static_cast<double>(j) / static_cast<double>(F));
      const double a = static_cast<double>(t) * f;
      emb_[z(j)] = static_cast<float>(std::sin(a));
      emb_[z(F + j)] = static_cast<float>(std::cos(a));
    }
    for (int j = 0; j < n.cond; ++j) emb_[z(2 * F + j)] = z(j) < cond.size() ? cond[z(j)] : 0.f;
    conv1_fwd(1, emb_.data(), E, w + n.mlp1_w, w + n.mlp1_b, H, mh_.data());
    silu_n(mh_.data(), ma_.data(), z(H));
    conv1_fwd(1, ma_.data(), H, w + n.mlp2_w, w + n.mlp2_b, static_cast<int>(n.film_size), film_.data());
    for (int yy = 0; yy < R0_; ++yy) {
      for (int xx = 0; xx < R0_; ++xx) {
        const std::size_t i = z(yy) * z(R0_) + z(xx);
        float* o = xpad_.data() + (z(yy + 1) * z(R0_ + 2) + z(xx + 1)) * z(I);
        for (int q = 0; q < ch; ++q) o[q] = x[i * z(ch) + z(q)];
      }
    }
    const int R0 = R0_, R1 = R1_, R2 = R2_;
    // encoder
    conv3_fwd(R0, xpad_.data(), I, w + n.stem_w, w + n.stem_b, n.c0, stem_.data());
    block(0, 0, stem_.data(), s0_.data());
    avgpool(R0, s0_.data(), n.c0, pool0_.data());
    conv1_fwd(R1 * R1, pool0_.data(), n.c0, w + n.down1_w, w + n.down1_b, n.c1, d1in_.data());
    block(1, 1, d1in_.data(), s1_.data());
    avgpool(R1, s1_.data(), n.c1, pool1_.data());
    conv1_fwd(R2 * R2, pool1_.data(), n.c1, w + n.down2_w, w + n.down2_b, n.c2, d2in_.data());
    block(2, 2, d2in_.data(), mm_.data());
    block(3, 2, mm_.data(), mout_.data());
    // decoder
    conv1_fwd(R2 * R2, mout_.data(), n.c2, w + n.up2_w, w + n.up2_b, n.c1, u2c_.data());
    std::copy(s1_.begin(), s1_.end(), u1in_.begin());
    upsample_add(R2, u2c_.data(), n.c1, u1in_.data());
    block(4, 1, u1in_.data(), dec1_.data());
    conv1_fwd(R1 * R1, dec1_.data(), n.c1, w + n.up1_w, w + n.up1_b, n.c0, u1c_.data());
    std::copy(s0_.begin(), s0_.end(), u0in_.begin());
    upsample_add(R1, u1c_.data(), n.c0, u0in_.data());
    block(5, 0, u0in_.data(), dec0_.data());
    // output
    silu_pad(R0, dec0_.data(), n.c0, pad_[0].data());
    conv3_fwd(R0, pad_[0].data(), n.c0, out_w8_.data(), out_b8_.data(), c8_, y8_.data());
  }

  const PriorNet& n_;
  int R0_ = 0, R1_ = 0, R2_ = 0, c8_ = 8;
  std::vector<float> emb_, mh_, ma_, film_, xpad_;
  std::array<std::vector<float>, 3> pad_, act_, h_;  // per level: padded input of a convolution, activations, first convolution
  std::vector<float> stem_, s0_, s1_, pool0_, pool1_, d1in_, d2in_, mm_, mout_, u2c_, u1in_, dec1_, u1c_, u0in_, dec0_;
  std::vector<float> out_w8_, out_b8_, y8_;
  std::vector<float> x_, xt_, eps_;
};

}  // namespace

std::unique_ptr<Prior> make_prior(const PriorNet& net) { return std::make_unique<PriorImpl>(net); }

}  // namespace nfx::rt::NFX_NS

#if defined(NFX_PUSHED)
#pragma GCC pop_options
#undef NFX_PUSHED
#endif
