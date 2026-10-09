// Convolution kernels of the hand-written denoiser (src/dcm/ddpm.cpp), on [cell][channel] layouts.
//
// conv3_ref / conv3_back_ref are the plain patterns of src/train/rollout_train.cpp (copied here, not moved: that file
// belongs to the stepper's trainer). The tests use them as the reference for the faster kernels below, which do the
// same arithmetic on a zero-padded copy of the input so that the inner loops have no edge tests:
//   conv3_fwd    out = b + W * in                    (weights [tap][in][out], tap = (dy + 1) * 3 + (dx + 1))
//   conv3_wgrad  gW += in^T * g, gb += sum of g      (the weight gradient, accumulated over cells)
//   flip_transpose: the weights of the input gradient. gin = conv3_fwd(pad(g), flip_transpose(W)), because the
//                transpose of a 3 x 3 convolution with zero padding is the convolution with the taps mirrored and the
//                in and out channels swapped.
// Output channels are processed in blocks of eight floats (one AVX register) for two cells at a time, so a weight row
// is loaded once for two cells. Other output counts (the output layer's 4) are widened to a multiple of 8 with zeros.
// SiLU and its derivative use a vectorised exponential. The including file switches on AVX2 + FMA and multiply-add
// contraction.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace nfx::dcm::ddpm::kernels {

inline std::size_t sz(int v) { return static_cast<std::size_t>(v); }

typedef float v8 __attribute__((vector_size(32)));
typedef float v8u __attribute__((vector_size(32), aligned(4)));

inline v8 load8(const float* p) { return *reinterpret_cast<const v8u*>(p); }
inline void store8(float* p, v8 v) { *reinterpret_cast<v8u*>(p) = v; }
inline v8 splat(float a) { return v8{a, a, a, a, a, a, a, a}; }
inline float hsum(v8 v) {
  float s = 0;
  for (int k = 0; k < 8; ++k) s += v[k];
  return s;
}

inline float dot(const float* __restrict a, const float* __restrict b, int n) {
  v8 acc{};
  int i = 0;
  for (; i + 8 <= n; i += 8) acc += load8(a + i) * load8(b + i);
  float s = hsum(acc);
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

// --- SiLU ------------------------------------------------------------------------------------------------------

typedef int v8i __attribute__((vector_size(32)));

// e^x (x clamped to [-87, 87]) with a relative error below 3e-7: x = n ln 2 + r with |r| <= ln 2 / 2, e^r by its
// Taylor polynomial of degree 7, 2^n through the exponent bits. The same code for every lane, so the forward pass and
// its derivative agree exactly.
inline v8 exp8(v8 x) {
  const v8 lo = splat(-87.f), hi = splat(87.f);
  x = x < lo ? lo : x;
  x = x > hi ? hi : x;
  const v8 magic = splat(12582912.f);  // 1.5 * 2^23: adding and subtracting it rounds to the nearest integer
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

// Runs f on blocks of eight values (the tail through a zero-padded block).
template <class F>
inline void by8(std::size_t n, F&& f) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) f(i, 8);
  if (i < n) f(i, n - i);
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

// y = x * sigmoid(x)
inline void silu_n(const float* x, float* y, std::size_t n) {
  by8(n, [&](std::size_t i, std::size_t m) {
    const v8 v = load_n(x + i, m);
    store_n(y + i, v / (splat(1.f) + exp8(-v)), m);
  });
}
// g *= d silu / dx at x = s (1 + x (1 - s)), s = sigmoid(x)
inline void silu_grad_mul(const float* x, float* g, std::size_t n) {
  by8(n, [&](std::size_t i, std::size_t m) {
    const v8 v = load_n(x + i, m);
    const v8 s = splat(1.f) / (splat(1.f) + exp8(-v));
    store_n(g + i, load_n(g + i, m) * (s * (splat(1.f) + v * (splat(1.f) - s))), m);
  });
}

// --- the reference patterns (rollout_train.cpp) ---------------------------------------------------------------------

inline void conv3_ref(int R, const float* in, int ci, const float* W, const float* b, int co, float* out) {
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      float* __restrict o = out + (sz(y) * sz(R) + sz(x)) * sz(co);
      for (int k = 0; k < co; ++k) o[k] = b ? b[k] : 0.f;
      for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= R) continue;
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = x + dx;
          if (xx < 0 || xx >= R) continue;
          const float* a = in + (sz(yy) * sz(R) + sz(xx)) * sz(ci);
          const float* w = W + sz((dy + 1) * 3 + (dx + 1)) * sz(ci) * sz(co);
          for (int c = 0; c < ci; ++c) {
            const float av = a[c];
            const float* __restrict wr = w + sz(c) * sz(co);
            for (int k = 0; k < co; ++k) o[k] += wr[k] * av;
          }
        }
      }
    }
  }
}

inline void conv3_back_ref(int R, const float* in, int ci, const float* W, int co, const float* g, float* gin, float* gW, float* gb) {
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const float* go = g + (sz(y) * sz(R) + sz(x)) * sz(co);
      for (int k = 0; k < co; ++k) gb[k] += go[k];
      for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= R) continue;
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = x + dx;
          if (xx < 0 || xx >= R) continue;
          const float* a = in + (sz(yy) * sz(R) + sz(xx)) * sz(ci);
          float* ga = gin ? gin + (sz(yy) * sz(R) + sz(xx)) * sz(ci) : nullptr;
          const std::size_t t = sz((dy + 1) * 3 + (dx + 1)) * sz(ci) * sz(co);
          for (int c = 0; c < ci; ++c) {
            const float av = a[c];
            float* __restrict gw = gW + t + sz(c) * sz(co);
            for (int k = 0; k < co; ++k) gw[k] += av * go[k];
            if (ga) ga[c] += dot(W + t + sz(c) * sz(co), go, co);
          }
        }
      }
    }
  }
}

// --- padding and weight transforms -------------------------------------------------------------------------------

// (R + 2)^2 x C copy of in with a zero border.
inline void pad(int R, const float* in, int C, float* out) {
  const int P = R + 2;
  std::fill_n(out, sz(P) * sz(P) * sz(C), 0.f);
  for (int y = 0; y < R; ++y) std::memcpy(out + (sz(y + 1) * sz(P) + 1) * sz(C), in + sz(y) * sz(R) * sz(C), sizeof(float) * sz(R) * sz(C));
}

// Wt[tap][co][ci] = W[8 - tap][ci][co]: the weights of the input gradient.
inline void flip_transpose(const float* W, int ci, int co, float* Wt) {
  for (int tap = 0; tap < 9; ++tap) {
    const float* w = W + sz(8 - tap) * sz(ci) * sz(co);
    float* t = Wt + sz(tap) * sz(co) * sz(ci);
    for (int c = 0; c < ci; ++c) {
      for (int k = 0; k < co; ++k) t[sz(k) * sz(ci) + sz(c)] = w[sz(c) * sz(co) + sz(k)];
    }
  }
}

// --- forward ----------------------------------------------------------------------------------------------------

// NV output vectors (8 channels each) starting at k0, for cells (x, y) and (x + 1, y) when two is set.
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
    const float* p0 = ip + (sz(y + dy) * sz(P) + sz(x + dx)) * sz(ci);
    const float* p1 = p0 + ci;
    const float* w = W + sz(tap) * sz(ci) * sz(co) + sz(k0);
    if (two) {
      for (int c = 0; c < ci; ++c) {
        const v8 s0 = splat(p0[c]), s1 = splat(p1[c]);
        const float* wr = w + sz(c) * sz(co);
        for (int v = 0; v < NV; ++v) {
          const v8 wv = load8(wr + 8 * v);
          a0[v] += s0 * wv;
          a1[v] += s1 * wv;
        }
      }
    } else {
      for (int c = 0; c < ci; ++c) {
        const v8 s0 = splat(p0[c]);
        const float* wr = w + sz(c) * sz(co);
        for (int v = 0; v < NV; ++v) a0[v] += s0 * load8(wr + 8 * v);
      }
    }
  }
  float* o0 = out + (sz(y) * sz(R) + sz(x)) * sz(co) + sz(k0);
  for (int v = 0; v < NV; ++v) store8(o0 + 8 * v, a0[v]);
  if (two) {
    float* o1 = o0 + co;
    for (int v = 0; v < NV; ++v) store8(o1 + 8 * v, a1[v]);
  }
}

// out (R^2 x co) = b + conv3(in), with ip the padded input ((R + 2)^2 x ci). b may be null (no bias).
inline void conv3_fwd(int R, const float* ip, int ci, const float* W, const float* b, int co, float* out) {
  if (co % 8 == 0) {
    for (int y = 0; y < R; ++y) {
      for (int x = 0; x < R; x += 2) {
        const bool two = x + 1 < R;
        int k0 = 0;
        for (; k0 + 32 <= co; k0 += 32) conv3_block<4>(R, ip, ci, W, b, co, k0, x, y, two, out);
        for (; k0 + 16 <= co; k0 += 16) conv3_block<2>(R, ip, ci, W, b, co, k0, x, y, two, out);
        for (; k0 < co; k0 += 8) conv3_block<1>(R, ip, ci, W, b, co, k0, x, y, two, out);
      }
    }
    return;
  }
  // Other output counts (the output layer's 4): the weights and bias widened to a multiple of 8 with zeros.
  const int c8 = (co + 7) / 8 * 8;
  thread_local std::vector<float> wk, bk, ok;
  wk.assign(9 * sz(ci) * sz(c8), 0.f);
  bk.assign(sz(c8), 0.f);
  ok.resize(sz(R) * sz(R) * sz(c8));
  for (int tc = 0; tc < 9 * ci; ++tc) std::copy_n(W + sz(tc) * sz(co), co, wk.data() + sz(tc) * sz(c8));
  if (b) std::copy_n(b, co, bk.data());
  conv3_fwd(R, ip, ci, wk.data(), bk.data(), c8, ok.data());
  for (int i = 0; i < R * R; ++i) std::copy_n(ok.data() + sz(i) * sz(c8), co, out + sz(i) * sz(co));
}

// --- weight gradient ----------------------------------------------------------------------------------------------

template <int NV>
inline void wgrad_block(int R, const float* __restrict ip, int ci, const float* __restrict g, int co, int tap, int c, bool two, int k0,
                        float* __restrict gW) {
  const int P = R + 2, dy = tap / 3, dx = tap % 3;
  v8 a0[NV], a1[NV];
  for (int v = 0; v < NV; ++v) a0[v] = a1[v] = v8{};
  for (int y = 0; y < R; ++y) {
    const float* prow = ip + (sz(y + dy) * sz(P) + sz(dx)) * sz(ci) + sz(c);
    const float* grow = g + sz(y) * sz(R) * sz(co) + sz(k0);
    for (int x = 0; x < R; ++x) {
      const float* gp = grow + sz(x) * sz(co);
      const float* pp = prow + sz(x) * sz(ci);
      const v8 s0 = splat(pp[0]);
      if (two) {
        const v8 s1 = splat(pp[1]);
        for (int v = 0; v < NV; ++v) {
          const v8 gv = load8(gp + 8 * v);
          a0[v] += s0 * gv;
          a1[v] += s1 * gv;
        }
      } else {
        for (int v = 0; v < NV; ++v) a0[v] += s0 * load8(gp + 8 * v);
      }
    }
  }
  float* w0 = gW + (sz(tap) * sz(ci) + sz(c)) * sz(co) + sz(k0);
  for (int v = 0; v < NV; ++v) store8(w0 + 8 * v, load8(w0 + 8 * v) + a0[v]);
  if (two) {
    float* w1 = w0 + co;
    for (int v = 0; v < NV; ++v) store8(w1 + 8 * v, load8(w1 + 8 * v) + a1[v]);
  }
}

// gW (9 x ci x co) += in^T g per tap, gb (co) += sum over cells of g; ip padded, g R^2 x co.
inline void conv3_wgrad(int R, const float* ip, int ci, const float* g, int co, float* gW, float* gb) {
  for (int i = 0; i < R * R; ++i) {
    for (int k = 0; k < co; ++k) gb[k] += g[sz(i) * sz(co) + sz(k)];
  }
  if (co % 8 == 0) {
    for (int tap = 0; tap < 9; ++tap) {
      for (int c = 0; c < ci; c += 2) {
        const bool two = c + 1 < ci;
        int k0 = 0;
        for (; k0 + 32 <= co; k0 += 32) wgrad_block<4>(R, ip, ci, g, co, tap, c, two, k0, gW);
        for (; k0 + 16 <= co; k0 += 16) wgrad_block<2>(R, ip, ci, g, co, tap, c, two, k0, gW);
        for (; k0 < co; k0 += 8) wgrad_block<1>(R, ip, ci, g, co, tap, c, two, k0, gW);
      }
    }
    return;
  }
  // Other output counts: the gradient widened to a multiple of 8 with zeros.
  const int c8 = (co + 7) / 8 * 8;
  thread_local std::vector<float> gk, wk, bk;
  gk.assign(sz(R) * sz(R) * sz(c8), 0.f);
  wk.assign(9 * sz(ci) * sz(c8), 0.f);
  bk.assign(sz(c8), 0.f);
  for (int i = 0; i < R * R; ++i) std::copy_n(g + sz(i) * sz(co), co, gk.data() + sz(i) * sz(c8));
  conv3_wgrad(R, ip, ci, gk.data(), c8, wk.data(), bk.data());
  for (int tc = 0; tc < 9 * ci; ++tc) {
    for (int k = 0; k < co; ++k) gW[sz(tc) * sz(co) + sz(k)] += wk[sz(tc) * sz(c8) + sz(k)];
  }
}

// --- 1 x 1 convolutions ---------------------------------------------------------------------------------------------

// out (n x co) = b + in W, W [ci][co].
inline void conv1_fwd(int n, const float* in, int ci, const float* W, const float* b, int co, float* out) {
  for (int i = 0; i < n; ++i) {
    float* __restrict o = out + sz(i) * sz(co);
    for (int k = 0; k < co; ++k) o[k] = b[k];
    const float* a = in + sz(i) * sz(ci);
    for (int c = 0; c < ci; ++c) {
      const float av = a[c];
      const float* __restrict wr = W + sz(c) * sz(co);
      for (int k = 0; k < co; ++k) o[k] += wr[k] * av;
    }
  }
}

// gW += in^T g, gb += sum g, gin (n x ci, may be null) += g W^T.
inline void conv1_back(int n, const float* in, int ci, const float* W, int co, const float* g, float* gin, float* gW, float* gb) {
  for (int i = 0; i < n; ++i) {
    const float* go = g + sz(i) * sz(co);
    const float* a = in + sz(i) * sz(ci);
    for (int k = 0; k < co; ++k) gb[k] += go[k];
    for (int c = 0; c < ci; ++c) {
      const float av = a[c];
      float* __restrict gw = gW + sz(c) * sz(co);
      for (int k = 0; k < co; ++k) gw[k] += av * go[k];
      if (gin) gin[sz(i) * sz(ci) + sz(c)] += dot(W + sz(c) * sz(co), go, co);
    }
  }
}

}  // namespace nfx::dcm::ddpm::kernels
