// Forward and backward passes of one frame for both model families. The math mirrors reference_render() in
// src/core/model.cpp exactly; tests/test_train.cpp checks the forward pass against it and the gradients against
// finite differences.
#include "net.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

// AVX2 + FMA for the loops below, switched on after the standard headers (train() checks the CPU first).
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#pragma GCC push_options
#pragma GCC target("arch=x86-64-v3")
#define NFX_TRAIN_PUSHED 1
#endif

namespace nfx::train::detail {

namespace {

constexpr int kChunk = 256;  // pixels per SoA chunk (grid family)

typedef float v8 __attribute__((vector_size(32)));
typedef float v8u __attribute__((vector_size(32), aligned(4)));

inline float dot(const float* __restrict a, const float* __restrict b, int n) {
  v8 acc0{}, acc1{};
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 += *reinterpret_cast<const v8u*>(a + i) * *reinterpret_cast<const v8u*>(b + i);
    acc1 += *reinterpret_cast<const v8u*>(a + i + 8) * *reinterpret_cast<const v8u*>(b + i + 8);
  }
  for (; i + 8 <= n; i += 8) acc0 += *reinterpret_cast<const v8u*>(a + i) * *reinterpret_cast<const v8u*>(b + i);
  acc0 += acc1;
  float s = 0;
  for (int k = 0; k < 8; ++k) s += acc0[k];
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

inline float sum(const float* a, int n) {
  float s = 0;
  for (int i = 0; i < n; ++i) s += a[i];
  return s;
}

inline void axpy(float w, const float* __restrict x, float* __restrict y, int n) {
  for (int i = 0; i < n; ++i) y[i] += w * x[i];
}

std::vector<float> apply(const Dense& d, std::span<const float> x) {
  std::vector<float> y(d.b);
  for (int o = 0; o < d.out; ++o) y[static_cast<std::size_t>(o)] += dot(d.w.data() + static_cast<std::size_t>(o) * d.in, x.data(), d.in);
  return y;
}

// y[o][p] = b[o] + sum_i W[o][i] x[i][p] over n pixels (rows n floats apart).
void dense_fwd(const Dense& d, const float* __restrict x, float* __restrict y, int n) {
  for (int o = 0; o < d.out; ++o) {
    float* yo = y + static_cast<std::size_t>(o) * n;
    std::fill_n(yo, n, d.b[static_cast<std::size_t>(o)]);
    const float* w = d.w.data() + static_cast<std::size_t>(o) * d.in;
    for (int i = 0; i < d.in; ++i) axpy(w[i], x + static_cast<std::size_t>(i) * n, yo, n);
  }
}

// g.W += dy x^T, g.b += sum dy, dx = W^T dy (when dx is given).
void dense_bwd(const Dense& d, Dense& g, const float* x, const float* dy, float* dx, int n) {
  for (int o = 0; o < d.out; ++o) {
    const float* dyo = dy + static_cast<std::size_t>(o) * n;
    g.b[static_cast<std::size_t>(o)] += sum(dyo, n);
    float* gw = g.w.data() + static_cast<std::size_t>(o) * d.in;
    for (int i = 0; i < d.in; ++i) gw[i] += dot(dyo, x + static_cast<std::size_t>(i) * n, n);
  }
  if (!dx) return;
  std::fill_n(dx, static_cast<std::size_t>(d.in) * n, 0.f);
  for (int o = 0; o < d.out; ++o) {
    const float* w = d.w.data() + static_cast<std::size_t>(o) * d.in;
    const float* dyo = dy + static_cast<std::size_t>(o) * n;
    for (int i = 0; i < d.in; ++i) axpy(w[i], dyo, dx + static_cast<std::size_t>(i) * n, n);
  }
}

// 3x3 convolution, planar: in [ci][s+2][s+2] (zero border), W [co][ci][3][3], out [co][s][s].
void conv_fwd(const float* in, const Dense& L, float* out, int ci, int co, int s) {
  const int ws = s + 2;
  for (int o = 0; o < co; ++o) {
    float* oo = out + static_cast<std::size_t>(o) * s * s;
    std::fill_n(oo, static_cast<std::size_t>(s) * s, L.b[static_cast<std::size_t>(o)]);
    for (int i = 0; i < ci; ++i) {
      const float* ii = in + static_cast<std::size_t>(i) * ws * ws;
      for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
          const float w = L.w[((static_cast<std::size_t>(o) * ci + i) * 3 + ky) * 3 + kx];
          for (int y = 0; y < s; ++y) axpy(w, ii + static_cast<std::size_t>(y + ky) * ws + kx, oo + static_cast<std::size_t>(y) * s, s);
        }
      }
    }
  }
}

// Gradients of conv_fwd: g.W, g.b, and din [ci][s+2][s+2] (overwritten).
void conv_bwd(const float* in, const Dense& L, Dense& g, const float* dout, float* din, int ci, int co, int s) {
  const int ws = s + 2;
  std::fill_n(din, static_cast<std::size_t>(ci) * ws * ws, 0.f);
  for (int o = 0; o < co; ++o) {
    const float* dd = dout + static_cast<std::size_t>(o) * s * s;
    g.b[static_cast<std::size_t>(o)] += sum(dd, s * s);
    for (int i = 0; i < ci; ++i) {
      const float* ii = in + static_cast<std::size_t>(i) * ws * ws;
      float* di = din + static_cast<std::size_t>(i) * ws * ws;
      for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
          const std::size_t wi = ((static_cast<std::size_t>(o) * ci + i) * 3 + ky) * 3 + kx;
          const float w = L.w[wi];
          float acc = 0;
          for (int y = 0; y < s; ++y) {
            const float* drow = dd + static_cast<std::size_t>(y) * s;
            acc += dot(drow, ii + static_cast<std::size_t>(y + ky) * ws + kx, s);
            axpy(w, drow, di + static_cast<std::size_t>(y + ky) * ws + kx, s);
          }
          g.w[wi] += acc;
        }
      }
    }
  }
}

// Nearest 2x upsampling into a zero-bordered buffer: in [c][s][s] -> out [c][2s+2][2s+2].
void upsample_pad(const float* in, float* out, int c, int s) {
  const int os = 2 * s + 2;
  std::fill_n(out, static_cast<std::size_t>(c) * os * os, 0.f);
  for (int k = 0; k < c; ++k) {
    for (int y = 0; y < s; ++y) {
      const float* src = in + (static_cast<std::size_t>(k) * s + y) * s;
      float* d = out + (static_cast<std::size_t>(k) * os + 2 * y + 1) * os + 1;
      for (int x = 0; x < s; ++x) d[2 * x] = d[2 * x + 1] = src[x];
      std::memcpy(d + os, d, sizeof(float) * 2 * static_cast<std::size_t>(s));
    }
  }
}

// Gradient of upsample_pad: d [c][s][s] = sum of each 2x2 block of the padded gradient's interior.
void down_sum(const float* dpad, float* d, int c, int s) {
  const int os = 2 * s + 2;
  for (int k = 0; k < c; ++k) {
    for (int y = 0; y < s; ++y) {
      const float* r0 = dpad + (static_cast<std::size_t>(k) * os + 2 * y + 1) * os + 1;
      const float* r1 = r0 + os;
      float* o = d + (static_cast<std::size_t>(k) * s + y) * s;
      for (int x = 0; x < s; ++x) o[x] = r0[2 * x] + r0[2 * x + 1] + r1[2 * x] + r1[2 * x + 1];
    }
  }
}

}  // namespace

namespace {
std::size_t total_slices(const Hyper& h) {
  const Volume last = volumes(h).back();
  return last.slice0 + static_cast<std::size_t>(last.slices);
}
}  // namespace

Grads::Grads(const Model& m) : g(m), touched(total_slices(m.h), 0) { zero(); }

void Grads::zero() {
  std::ranges::fill(g.features, 0.f);
  std::ranges::fill(touched, 0);
  for (auto* d : {&g.basis}) {
    std::ranges::fill(d->w, 0.f);
    std::ranges::fill(d->b, 0.f);
  }
  for (auto* group : {&g.layers, &g.films}) {
    for (auto& d : *group) {
      std::ranges::fill(d.w, 0.f);
      std::ranges::fill(d.b, 0.f);
    }
  }
}

void Grads::add(const Grads& o) {
  const Hyper& h = g.h;
  for (const Volume& v : volumes(h)) {
    const std::size_t plane = static_cast<std::size_t>(v.channels) * v.plane_values();
    for (int s = 0; s < v.slices; ++s) {
      if (!o.touched[v.slice0 + static_cast<std::size_t>(s)]) continue;
      touched[v.slice0 + static_cast<std::size_t>(s)] = 1;
      for (int k = 0; k < h.bases; ++k) {
        const std::size_t off = v.value0 + (static_cast<std::size_t>(k) * v.slices + static_cast<std::size_t>(s)) * plane;
        for (std::size_t i = 0; i < plane; ++i) g.features[off + i] += o.g.features[off + i];
      }
    }
  }
  const auto add_dense = [](Dense& a, const Dense& b) {
    for (std::size_t i = 0; i < a.w.size(); ++i) a.w[i] += b.w[i];
    for (std::size_t i = 0; i < a.b.size(); ++i) a.b[i] += b.b[i];
  };
  add_dense(g.basis, o.g.basis);
  for (std::size_t l = 0; l < g.layers.size(); ++l) add_dense(g.layers[l], o.g.layers[l]);
  for (std::size_t l = 0; l < g.films.size(); ++l) add_dense(g.films[l], o.g.films[l]);
}

Net::Net(const Model& m) {
  const Hyper& h = m.h;
  if (h.arch == Arch::multi) {
    vols_ = volumes(h);
    std::size_t total = 0;
    for (const Volume& v : vols_) {
      loff_.push_back(total);
      total += static_cast<std::size_t>(v.channels) * v.plane_values();
    }
    slice_.resize(total);
    dslice_.resize(total);
    li0_.resize(vols_.size());
    li1_.resize(vols_.size());
    lft_.resize(vols_.size());
    tpe_.resize(static_cast<std::size_t>(2 * h.pe_t));
  } else {
    const std::size_t plane = static_cast<std::size_t>(h.feature_channels()) * h.feature_side() * h.feature_side();
    slice_.resize(plane);
    dslice_.resize(plane);
  }
  if (h.arch == Arch::grid || h.arch == Arch::multi) {
    const auto n = static_cast<std::size_t>(kChunk);
    const int inputs = h.arch == Arch::grid ? h.channels : h.mlp_in();
    const std::size_t widest = static_cast<std::size_t>(std::max({inputs, h.hidden, 4}));
    x0_.resize(static_cast<std::size_t>(inputs) * n);
    dx_.resize(static_cast<std::size_t>(inputs) * n);
    z1_.resize(static_cast<std::size_t>(h.hidden) * n);
    a1_.resize(static_cast<std::size_t>(h.hidden) * n);
    dz_.resize(widest * n);
    dh_.resize(widest * n);
    dy_.resize(widest * n);
    out_.resize(4 * n);
    hs_.assign(static_cast<std::size_t>(h.layers), std::vector<float>(static_cast<std::size_t>(h.hidden) * n));
    const std::size_t L = h.arch == Arch::multi ? vols_.size() : 1;  // bilinear corners per pixel and level
    cx_.resize(n * L);
    cy_.resize(n * L);
    cfx_.resize(n * L);
    cfy_.resize(n * L);
  } else {
    const std::size_t l = static_cast<std::size_t>(h.latent), s1 = 2 * l, s2 = 4 * l, s3 = 8 * l;
    const std::size_t c0 = static_cast<std::size_t>(h.c0), c1 = static_cast<std::size_t>(h.c1), c2 = static_cast<std::size_t>(h.c2);
    pad0_.resize(c0 * (s1 + 2) * (s1 + 2));
    y0_.resize(c1 * s1 * s1);
    h0_.resize(c1 * s1 * s1);
    pad1_.resize(c1 * (s2 + 2) * (s2 + 2));
    y1_.resize(c2 * s2 * s2);
    h1_.resize(c2 * s2 * s2);
    pad2_.resize(c2 * (s3 + 2) * (s3 + 2));
    out_.resize(4 * s3 * s3);
    dpad_.resize(std::max({c2 * (s3 + 2) * (s3 + 2), c1 * (s2 + 2) * (s2 + 2), c0 * (s1 + 2) * (s1 + 2)}));
    dy_.resize(std::max({4 * s3 * s3, c2 * s2 * s2, c1 * s1 * s1}));
    dh_.resize(std::max(c2 * s2 * s2, c1 * s1 * s1));
  }
}

void Net::begin_frame(const Model& m, float t, std::span<const float> c) {
  const Hyper& h = m.h;
  w_ = apply(m.basis, c);
  film_.clear();
  for (const auto& f : m.films) {
    const auto v = apply(f, c);
    film_.insert(film_.end(), v.begin(), v.end());
  }
  dfilm_.assign(film_.size(), 0.f);
  if (h.arch == Arch::multi) {
    std::ranges::fill(slice_, 0.f);
    std::ranges::fill(dslice_, 0.f);
    for (std::size_t l = 0; l < vols_.size(); ++l) {
      const Volume& v = vols_[l];
      slice_lerp(v.slices, h.loop, t, li0_[l], li1_[l], lft_[l]);
      const std::size_t plane = static_cast<std::size_t>(v.channels) * v.plane_values();
      float* s = slice_.data() + loff_[l];
      for (int k = 0; k < h.bases; ++k) {
        const float* a = m.features.data() + v.value0 + (static_cast<std::size_t>(k) * v.slices + static_cast<std::size_t>(li0_[l])) * plane;
        const float* b = m.features.data() + v.value0 + (static_cast<std::size_t>(k) * v.slices + static_cast<std::size_t>(li1_[l])) * plane;
        const float wk = w_[static_cast<std::size_t>(k)], ft = lft_[l];
        for (std::size_t i = 0; i < plane; ++i) s[i] += wk * (a[i] + ft * (b[i] - a[i]));
      }
    }
    time_features(h.pe_t, h.loop, t, tpe_.data());
    return;
  }
  if (h.loop) {
    const float u = (t - std::floor(t)) * static_cast<float>(h.grid_t);
    i0_ = std::min(static_cast<int>(u), h.grid_t - 1);
    i1_ = (i0_ + 1) % h.grid_t;
    ft_ = u - static_cast<float>(i0_);
  } else {
    const float u = std::clamp(t, 0.f, 1.f) * static_cast<float>(h.grid_t - 1);
    i0_ = std::min(static_cast<int>(u), h.grid_t - 2);
    i1_ = i0_ + 1;
    ft_ = u - static_cast<float>(i0_);
  }
  const std::size_t plane = slice_.size();
  std::ranges::fill(slice_, 0.f);
  std::ranges::fill(dslice_, 0.f);
  for (int k = 0; k < h.bases; ++k) {
    const float* a = m.features.data() + (static_cast<std::size_t>(k) * h.grid_t + i0_) * plane;
    const float* b = m.features.data() + (static_cast<std::size_t>(k) * h.grid_t + i1_) * plane;
    const float wk = w_[static_cast<std::size_t>(k)];
    for (std::size_t i = 0; i < plane; ++i) slice_[i] += wk * (a[i] + ft_ * (b[i] - a[i]));
  }
}

void Net::end_frame(const Model& m, Grads& g, std::span<const float> c, std::span<float> dc) {
  const Hyper& h = m.h;
  const std::size_t plane = slice_.size();
  const int D = h.dims();
  std::vector<float> dw(static_cast<std::size_t>(h.bases), 0.f);
  if (h.arch == Arch::multi) {
    for (std::size_t l = 0; l < vols_.size(); ++l) {
      const Volume& v = vols_[l];
      const std::size_t lp = static_cast<std::size_t>(v.channels) * v.plane_values();
      const std::size_t s0 = static_cast<std::size_t>(li0_[l]), s1 = static_cast<std::size_t>(li1_[l]);
      g.touched[v.slice0 + s0] = 1;
      g.touched[v.slice0 + s1] = 1;
      const float ft = lft_[l];
      const float* ds = dslice_.data() + loff_[l];
      for (int k = 0; k < h.bases; ++k) {
        const std::size_t o0 = v.value0 + (static_cast<std::size_t>(k) * v.slices + s0) * lp, o1 = v.value0 + (static_cast<std::size_t>(k) * v.slices + s1) * lp;
        const float wk = w_[static_cast<std::size_t>(k)];
        const float* a = m.features.data() + o0;
        const float* b = m.features.data() + o1;
        axpy(wk * (1.f - ft), ds, g.g.features.data() + o0, static_cast<int>(lp));
        axpy(wk * ft, ds, g.g.features.data() + o1, static_cast<int>(lp));
        float sum_k = 0;
        for (std::size_t i = 0; i < lp; ++i) sum_k += ds[i] * (a[i] + ft * (b[i] - a[i]));
        dw[static_cast<std::size_t>(k)] += sum_k;
      }
    }
  }
  for (int k = 0; k < h.bases && h.arch != Arch::multi; ++k) {
    if (k == 0) {
      g.touched[static_cast<std::size_t>(i0_)] = 1;
      g.touched[static_cast<std::size_t>(i1_)] = 1;
    }
    const std::size_t o0 = (static_cast<std::size_t>(k) * h.grid_t + i0_) * plane, o1 = (static_cast<std::size_t>(k) * h.grid_t + i1_) * plane;
    const float wk = w_[static_cast<std::size_t>(k)];
    const float* a = m.features.data() + o0;
    const float* b = m.features.data() + o1;
    axpy(wk * (1.f - ft_), dslice_.data(), g.g.features.data() + o0, static_cast<int>(plane));
    axpy(wk * ft_, dslice_.data(), g.g.features.data() + o1, static_cast<int>(plane));
    float s = 0;
    for (std::size_t i = 0; i < plane; ++i) s += dslice_[i] * (a[i] + ft_ * (b[i] - a[i]));
    dw[static_cast<std::size_t>(k)] = s;
  }
  const auto back = [&](const Dense& d, Dense& gd, std::span<const float> dy) {
    for (int o = 0; o < d.out; ++o) {
      const float v = dy[static_cast<std::size_t>(o)];
      gd.b[static_cast<std::size_t>(o)] += v;
      for (int j = 0; j < D; ++j) {
        gd.w[static_cast<std::size_t>(o) * D + j] += v * c[static_cast<std::size_t>(j)];
        if (!dc.empty()) dc[static_cast<std::size_t>(j)] += d.w[static_cast<std::size_t>(o) * D + j] * v;
      }
    }
  };
  back(m.basis, g.g.basis, dw);
  std::size_t off = 0;
  for (std::size_t f = 0; f < m.films.size(); ++f) {
    back(m.films[f], g.g.films[f], std::span(dfilm_).subspan(off, static_cast<std::size_t>(m.films[f].out)));
    off += static_cast<std::size_t>(m.films[f].out);
  }
}

bool Net::mlp_chunk(const Model& m, Grads* g, std::span<const int> pixels, std::size_t start, int n, std::span<const std::uint8_t> target,
                    float scale, std::span<float> out, double& sse) {
  const int H = m.h.hidden;
  const std::size_t L = m.layers.size();  // hidden layers + head
  // layer 0 with FiLM, then the hidden stack, then the head
  dense_fwd(m.layers[0], x0_.data(), z1_.data(), n);
  for (int o = 0; o < H; ++o) {
    const float gm = 1.f + film_[static_cast<std::size_t>(o)], bt = film_[static_cast<std::size_t>(H + o)];
    float* z = z1_.data() + static_cast<std::size_t>(o) * n;
    float* a = a1_.data() + static_cast<std::size_t>(o) * n;
    float* hh = hs_[0].data() + static_cast<std::size_t>(o) * n;
    for (int p = 0; p < n; ++p) {
      a[p] = gm * z[p] + bt;
      hh[p] = std::max(0.f, a[p]);
    }
  }
  for (std::size_t l = 1; l + 1 < L; ++l) {
    dense_fwd(m.layers[l], hs_[l - 1].data(), hs_[l].data(), n);
    for (float& v : std::span(hs_[l].data(), static_cast<std::size_t>(H) * n)) v = std::max(0.f, v);
  }
  dense_fwd(m.layers[L - 1], hs_[L - 2].data(), out_.data(), n);
  if (!out.empty()) {
    for (int j = 0; j < n; ++j) {
      for (int ch = 0; ch < 4; ++ch) out[static_cast<std::size_t>(pixels[start + static_cast<std::size_t>(j)]) * 4 + static_cast<std::size_t>(ch)] = out_[static_cast<std::size_t>(ch) * n + static_cast<std::size_t>(j)];
    }
  }
  if (target.empty()) return false;
  for (int ch = 0; ch < 4; ++ch) {
    for (int j = 0; j < n; ++j) {
      const std::size_t pix = static_cast<std::size_t>(pixels[start + static_cast<std::size_t>(j)]);
      const float d = out_[static_cast<std::size_t>(ch) * n + static_cast<std::size_t>(j)] - static_cast<float>(target[pix * 4 + static_cast<std::size_t>(ch)]) * (1.f / 255.f);
      sse += static_cast<double>(d) * d;
      dy_[static_cast<std::size_t>(ch) * n + static_cast<std::size_t>(j)] = 2.f * d * scale;
    }
  }
  if (!g) return false;
  // backward through the head and the hidden stack
  dense_bwd(m.layers[L - 1], g->g.layers[L - 1], hs_[L - 2].data(), dy_.data(), dh_.data(), n);
  for (std::size_t l = L - 2; l >= 1; --l) {
    const float* hh = hs_[l].data();
    for (std::size_t i = 0; i < static_cast<std::size_t>(H) * n; ++i) dh_[i] = hh[i] > 0.f ? dh_[i] : 0.f;
    dense_bwd(m.layers[l], g->g.layers[l], hs_[l - 1].data(), dh_.data(), dz_.data(), n);
    std::swap(dh_, dz_);
  }
  // layer 0: relu, FiLM, dense
  for (int o = 0; o < H; ++o) {
    const float gm = 1.f + film_[static_cast<std::size_t>(o)];
    const float* a = a1_.data() + static_cast<std::size_t>(o) * n;
    const float* z = z1_.data() + static_cast<std::size_t>(o) * n;
    float* d = dh_.data() + static_cast<std::size_t>(o) * n;
    float dg = 0, db = 0;
    for (int p = 0; p < n; ++p) {
      const float da = a[p] > 0.f ? d[p] : 0.f;
      dg += da * z[p];
      db += da;
      d[p] = da * gm;
    }
    dfilm_[static_cast<std::size_t>(o)] += dg;
    dfilm_[static_cast<std::size_t>(H + o)] += db;
  }
  dense_bwd(m.layers[0], g->g.layers[0], x0_.data(), dh_.data(), dx_.data(), n);
  return true;
}

double Net::grid_pixels(const Model& m, Grads* g, std::span<const int> pixels, int size, std::span<const std::uint8_t> target,
                        float scale, std::span<float> out) {
  const Hyper& h = m.h;
  const int G = h.grid, C = h.channels;
  const float gmax = static_cast<float>(G - 1);
  double sse = 0;
  for (std::size_t start = 0; start < pixels.size(); start += kChunk) {
    const int n = static_cast<int>(std::min<std::size_t>(kChunk, pixels.size() - start));
    for (int j = 0; j < n; ++j) {  // bilinear features, SoA [C][n]
      const int pix = pixels[start + static_cast<std::size_t>(j)];
      const int px = pix % size, py = pix / size;
      const float gx = std::clamp((static_cast<float>(px) + 0.5f) / static_cast<float>(size) * static_cast<float>(G) - 0.5f, 0.f, gmax);
      const float gy = std::clamp((static_cast<float>(py) + 0.5f) / static_cast<float>(size) * static_cast<float>(G) - 0.5f, 0.f, gmax);
      const int x0 = std::min(static_cast<int>(gx), G - 2), y0 = std::min(static_cast<int>(gy), G - 2);
      const float fx = gx - static_cast<float>(x0), fy = gy - static_cast<float>(y0);
      cx_[static_cast<std::size_t>(j)] = x0;
      cy_[static_cast<std::size_t>(j)] = y0;
      cfx_[static_cast<std::size_t>(j)] = fx;
      cfy_[static_cast<std::size_t>(j)] = fy;
      for (int ch = 0; ch < C; ++ch) {
        const float* p = slice_.data() + (static_cast<std::size_t>(ch) * G + y0) * G + x0;
        const float a = p[0] + fx * (p[1] - p[0]), b = p[G] + fx * (p[G + 1] - p[G]);
        x0_[static_cast<std::size_t>(ch) * n + static_cast<std::size_t>(j)] = a + fy * (b - a);
      }
    }
    if (!mlp_chunk(m, g, pixels, start, n, target, scale, out, sse)) continue;
    for (int j = 0; j < n; ++j) {  // scatter to the four bilinear corners
      const int x0 = cx_[static_cast<std::size_t>(j)], y0 = cy_[static_cast<std::size_t>(j)];
      const float fx = cfx_[static_cast<std::size_t>(j)], fy = cfy_[static_cast<std::size_t>(j)];
      const float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
      for (int ch = 0; ch < C; ++ch) {
        const float d = dx_[static_cast<std::size_t>(ch) * n + static_cast<std::size_t>(j)];
        float* p = dslice_.data() + (static_cast<std::size_t>(ch) * G + y0) * G + x0;
        p[0] += w00 * d;
        p[1] += w10 * d;
        p[G] += w01 * d;
        p[G + 1] += w11 * d;
      }
    }
  }
  return sse;
}

// The multi family: every level's features sampled as grid_pixels samples the grid's, one after another in x0_, then
// the Fourier features of the pixel and of the frame; the backward pass scatters each level's gradient to its corners.
double Net::multi_pixels(const Model& m, Grads* g, std::span<const int> pixels, int size, std::span<const std::uint8_t> target,
                         float scale, std::span<float> out) {
  const Hyper& h = m.h;
  const std::size_t NL = vols_.size();
  const int C = h.feature_channels();
  double sse = 0;
  std::array<float, 48> pf{};
  for (std::size_t start = 0; start < pixels.size(); start += kChunk) {
    const int n = static_cast<int>(std::min<std::size_t>(kChunk, pixels.size() - start));
    const auto N = static_cast<std::size_t>(n);
    for (int j = 0; j < n; ++j) {
      const int pix = pixels[start + static_cast<std::size_t>(j)];
      const int px = pix % size, py = pix / size;
      std::size_t ch0 = 0;
      for (std::size_t l = 0; l < NL; ++l) {
        const int G = vols_[l].side;
        const float gmax = static_cast<float>(G - 1);
        const float gx = std::clamp((static_cast<float>(px) + 0.5f) / static_cast<float>(size) * static_cast<float>(G) - 0.5f, 0.f, gmax);
        const float gy = std::clamp((static_cast<float>(py) + 0.5f) / static_cast<float>(size) * static_cast<float>(G) - 0.5f, 0.f, gmax);
        const int x0 = std::min(static_cast<int>(gx), G - 2), y0 = std::min(static_cast<int>(gy), G - 2);
        const float fx = gx - static_cast<float>(x0), fy = gy - static_cast<float>(y0);
        const std::size_t at = l * kChunk + static_cast<std::size_t>(j);
        cx_[at] = x0;
        cy_[at] = y0;
        cfx_[at] = fx;
        cfy_[at] = fy;
        const float* base = slice_.data() + loff_[l];
        for (int ch = 0; ch < vols_[l].channels; ++ch) {
          const float* p = base + (static_cast<std::size_t>(ch) * G + y0) * G + x0;
          const float a = p[0] + fx * (p[1] - p[0]), b = p[G] + fx * (p[G + 1] - p[G]);
          x0_[(ch0 + static_cast<std::size_t>(ch)) * N + static_cast<std::size_t>(j)] = a + fy * (b - a);
        }
        ch0 += static_cast<std::size_t>(vols_[l].channels);
      }
      if (h.pe_xy > 0) {
        position_features(h.pe_xy, (static_cast<float>(px) + 0.5f) / static_cast<float>(size), (static_cast<float>(py) + 0.5f) / static_cast<float>(size), pf.data());
        for (int k = 0; k < 4 * h.pe_xy; ++k) x0_[(static_cast<std::size_t>(C) + static_cast<std::size_t>(k)) * N + static_cast<std::size_t>(j)] = pf[static_cast<std::size_t>(k)];
      }
      for (int k = 0; k < 2 * h.pe_t; ++k) {
        x0_[(static_cast<std::size_t>(C + 4 * h.pe_xy) + static_cast<std::size_t>(k)) * N + static_cast<std::size_t>(j)] = tpe_[static_cast<std::size_t>(k)];
      }
    }
    if (!mlp_chunk(m, g, pixels, start, n, target, scale, out, sse)) continue;
    std::size_t ch0 = 0;
    for (std::size_t l = 0; l < NL; ++l) {  // scatter to the four bilinear corners of every level
      const int G = vols_[l].side;
      float* base = dslice_.data() + loff_[l];
      for (int j = 0; j < n; ++j) {
        const std::size_t at = l * kChunk + static_cast<std::size_t>(j);
        const int x0 = cx_[at], y0 = cy_[at];
        const float fx = cfx_[at], fy = cfy_[at];
        const float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
        for (int ch = 0; ch < vols_[l].channels; ++ch) {
          const float d = dx_[(ch0 + static_cast<std::size_t>(ch)) * N + static_cast<std::size_t>(j)];
          float* p = base + (static_cast<std::size_t>(ch) * G + y0) * G + x0;
          p[0] += w00 * d;
          p[1] += w10 * d;
          p[G] += w01 * d;
          p[G + 1] += w11 * d;
        }
      }
      ch0 += static_cast<std::size_t>(vols_[l].channels);
    }
  }
  return sse;
}

double Net::conv_frame(const Model& m, Grads* g, std::span<const std::uint8_t> target, float scale, std::span<float> out) {
  const Hyper& h = m.h;
  const int l = h.latent, s1 = 2 * l, s2 = 4 * l, s3 = 8 * l;
  const int c0 = h.c0, c1 = h.c1, c2 = h.c2;
  const auto film_relu = [&](const float* y, float* hh, int ch, int s, std::size_t off) {
    for (int c = 0; c < ch; ++c) {
      const float gm = 1.f + film_[off + static_cast<std::size_t>(c)], bt = film_[off + static_cast<std::size_t>(ch + c)];
      const std::size_t base = static_cast<std::size_t>(c) * s * s;
      for (std::size_t i = 0; i < static_cast<std::size_t>(s) * s; ++i) hh[base + i] = std::max(0.f, gm * y[base + i] + bt);
    }
  };
  upsample_pad(slice_.data(), pad0_.data(), c0, l);
  conv_fwd(pad0_.data(), m.layers[0], y0_.data(), c0, c1, s1);
  film_relu(y0_.data(), h0_.data(), c1, s1, 0);
  upsample_pad(h0_.data(), pad1_.data(), c1, s1);
  conv_fwd(pad1_.data(), m.layers[1], y1_.data(), c1, c2, s2);
  const std::size_t off2 = 2 * static_cast<std::size_t>(c1);
  film_relu(y1_.data(), h1_.data(), c2, s2, off2);
  upsample_pad(h1_.data(), pad2_.data(), c2, s2);
  conv_fwd(pad2_.data(), m.layers[2], out_.data(), c2, 4, s3);
  const std::size_t px = static_cast<std::size_t>(s3) * s3;
  if (!out.empty()) {
    for (std::size_t i = 0; i < px; ++i) {
      for (std::size_t c = 0; c < 4; ++c) out[i * 4 + c] = out_[c * px + i];
    }
  }
  if (target.empty()) return 0;
  double sse = 0;
  for (std::size_t c = 0; c < 4; ++c) {
    for (std::size_t i = 0; i < px; ++i) {
      const float d = out_[c * px + i] - static_cast<float>(target[i * 4 + c]) * (1.f / 255.f);
      sse += static_cast<double>(d) * d;
      dy_[c * px + i] = 2.f * d * scale;
    }
  }
  if (!g) return sse;
  // FiLM + relu backward: d (gradient wrt h) -> dy (gradient wrt the conv output), FiLM gradients.
  const auto film_relu_bwd = [&](const float* y, const float* hh, const float* d, float* dy, int ch, int s, std::size_t off) {
    for (int c = 0; c < ch; ++c) {
      const float gm = 1.f + film_[off + static_cast<std::size_t>(c)];
      const std::size_t base = static_cast<std::size_t>(c) * s * s;
      float dg = 0, db = 0;
      for (std::size_t i = 0; i < static_cast<std::size_t>(s) * s; ++i) {
        const float da = hh[base + i] > 0.f ? d[base + i] : 0.f;
        dg += da * y[base + i];
        db += da;
        dy[base + i] = da * gm;
      }
      dfilm_[off + static_cast<std::size_t>(c)] += dg;
      dfilm_[off + static_cast<std::size_t>(ch + c)] += db;
    }
  };
  conv_bwd(pad2_.data(), m.layers[2], g->g.layers[2], dy_.data(), dpad_.data(), c2, 4, s3);
  down_sum(dpad_.data(), dh_.data(), c2, s2);
  film_relu_bwd(y1_.data(), h1_.data(), dh_.data(), dy_.data(), c2, s2, off2);
  conv_bwd(pad1_.data(), m.layers[1], g->g.layers[1], dy_.data(), dpad_.data(), c1, c2, s2);
  down_sum(dpad_.data(), dh_.data(), c1, s1);
  film_relu_bwd(y0_.data(), h0_.data(), dh_.data(), dy_.data(), c1, s1, 0);
  conv_bwd(pad0_.data(), m.layers[0], g->g.layers[0], dy_.data(), dpad_.data(), c0, c1, s1);
  down_sum(dpad_.data(), dslice_.data(), c0, l);
  return sse;
}

double Net::step(const Model& m, Grads& g, float t, std::span<const float> c, std::span<const std::uint8_t> target,
                 std::span<const int> pixels, int size, float scale, std::span<float> dc) {
  begin_frame(m, t, c);
  const double sse = m.h.arch == Arch::grid    ? grid_pixels(m, &g, pixels, size, target, scale, {})
                     : m.h.arch == Arch::multi ? multi_pixels(m, &g, pixels, size, target, scale, {})
                                               : conv_frame(m, &g, target, scale, {});
  end_frame(m, g, c, dc);
  return sse;
}

void Net::render(const Model& m, float t, std::span<const float> c, int size, std::span<float> rgba) {
  begin_frame(m, t, c);
  if (m.h.arch == Arch::grid || m.h.arch == Arch::multi) {
    std::vector<int> all(static_cast<std::size_t>(size) * size);
    for (std::size_t i = 0; i < all.size(); ++i) all[i] = static_cast<int>(i);
    if (m.h.arch == Arch::grid) grid_pixels(m, nullptr, all, size, {}, 0.f, rgba);
    else multi_pixels(m, nullptr, all, size, {}, 0.f, rgba);
  } else {
    if (size != m.h.size) throw std::invalid_argument("render: the conv family renders at its native size only");
    conv_frame(m, nullptr, {}, 0.f, rgba);
  }
}

}  // namespace nfx::train::detail

#if defined(NFX_TRAIN_PUSHED)
#pragma GCC pop_options
#endif
