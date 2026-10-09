// The runtime renderers, compiled once per ISA (rt_<isa>.cpp) inside namespace NFX_NS, with the ISA switched on by
// #pragma GCC target after the standard headers (as in src/proto). Every buffer is allocated in the constructor;
// render() allocates nothing.
//
// Per frame, the work that does not depend on the pixel is done once: the basis blend and time slice of the
// features (decoded from fp16 or 8-bit storage), and FiLM folded into the first layer's (grid) or the first two
// convolutions' (conv) weights and biases. Per pixel only the network itself runs.
#if !defined(NFX_NS) || !defined(NFX_TILE) || !defined(NFX_VW)
#error "define NFX_NS, NFX_TILE and NFX_VW before including rt_impl.hpp"
#endif

#include "rt_common.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <stdfloat>
#include <type_traits>
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

#include "../common/simd_kernels.hpp"

namespace {

// y = W x + b for small condition vectors (scalar; per frame only).
void small_dense(const Dense& d, std::span<const float> x, std::span<float> y) {
  for (int o = 0; o < d.out; ++o) {
    float s = d.b[static_cast<std::size_t>(o)];
    for (int i = 0; i < d.in; ++i) s += d.w[static_cast<std::size_t>(o) * d.in + i] * x[static_cast<std::size_t>(i)];
    y[static_cast<std::size_t>(o)] = s;
  }
}

// Time slices bracketing t and the weight of the second (same rule as the trainer and the reference).
void time_lerp(const Hyper& h, float t, int& i0, int& i1, float& ft) {
  if (h.loop) {
    const float u = (t - std::floor(t)) * static_cast<float>(h.grid_t);
    i0 = std::min(static_cast<int>(u), h.grid_t - 1);
    i1 = (i0 + 1) % h.grid_t;
    ft = u - static_cast<float>(i0);
  } else {
    const float u = std::clamp(t, 0.f, 1.f) * static_cast<float>(h.grid_t - 1);
    i0 = std::min(static_cast<int>(u), h.grid_t - 2);
    i1 = i0 + 1;
    ft = u - static_cast<float>(i0);
  }
}

// slice += a * plane(k, i) for every channel plane of slice i of basis k, decoding the storage format.
void accumulate_slice(const Model& m, int k, int i, float a, std::span<float> slice) {
  const Hyper& h = m.h;
  const std::size_t side2 = static_cast<std::size_t>(h.feature_side()) * h.feature_side();
  const std::size_t planes = static_cast<std::size_t>(h.feature_channels());
  const std::size_t first_plane = (static_cast<std::size_t>(k) * h.grid_t + i) * planes;
  if (m.feature_bits == 8) {
    for (std::size_t p = 0; p < planes; ++p) {
      const float lo = m.raw_ranges[(first_plane + p) * 2], hi = m.raw_ranges[(first_plane + p) * 2 + 1];
      const float base = a * lo, step = a * (hi - lo) / 255.f;
      const std::uint8_t* q = m.raw_u8.data() + (first_plane + p) * side2;
      float* s = slice.data() + p * side2;
      for (std::size_t j = 0; j < side2; ++j) s[j] += base + step * static_cast<float>(q[j]);
    }
  } else {
    const std::uint16_t* q = m.raw_f16.data() + first_plane * side2;
    const std::size_t n = planes * side2;
    float* s = slice.data();
    for (std::size_t j = 0; j < n; ++j) s[j] += a * static_cast<float>(std::bit_cast<std::float16_t>(q[j]));
  }
}

void blend_slice(const Model& m, float t, std::span<const float> w, std::span<float> slice) {
  int i0, i1;
  float ft;
  time_lerp(m.h, t, i0, i1, ft);
  std::ranges::fill(slice, 0.f);
  for (int k = 0; k < m.h.bases; ++k) {
    const float wk = w[static_cast<std::size_t>(k)];
    accumulate_slice(m, k, i0, wk * (1.f - ft), slice);
    accumulate_slice(m, k, i1, wk * ft, slice);
  }
}

// One channel of a grid row expanded to full width: d[x] = lerp(r[xi[x]], r[xi[x] + 1], xf[x]). Kept out of line so its
// registers are its own (inlined into render() it reloaded every pointer from the stack per pixel).
[[gnu::noinline]] void expand_row(const float* __restrict r, const int* __restrict xi, const float* __restrict xf,
                                  float* __restrict d, int n) {
  for (int x = 0; x < n; ++x) {
    const float a = r[xi[x]], b = r[xi[x] + 1];
    d[x] = a + xf[x] * (b - a);
  }
}

// Premultiplied RGBA floats (planar rows r, g, b, a of n pixels) to RGBA8, with the optional colour matrix.
// Planar rows are processed in chunks of 16: the colour matrix and the clamp-and-round vectorise as plain loops; only
// the final interleave into RGBA bytes is per pixel.
inline void write_pixels(const float* r, const float* g, const float* b, const float* a, int n, const FrameInput& in,
                         std::uint8_t* out) {
  for (int k0 = 0; k0 < n; k0 += kB) {
    const int m = std::min(kB, n - k0);
    float q[4][kB];
    for (int k = 0; k < m; ++k) {
      q[0][k] = r[k0 + k];
      q[1][k] = g[k0 + k];
      q[2][k] = b[k0 + k];
      q[3][k] = a[k0 + k];
    }
    if (in.apply_colour) {
      const auto& M = in.colour;
      for (int k = 0; k < m; ++k) {
        const float cr = q[0][k], cg = q[1][k], cb = q[2][k];
        q[0][k] = M[0] * cr + M[1] * cg + M[2] * cb;
        q[1][k] = M[3] * cr + M[4] * cg + M[5] * cb;
        q[2][k] = M[6] * cr + M[7] * cg + M[8] * cb;
      }
    }
    std::uint8_t u[4][kB];
    for (int c = 0; c < 4; ++c) {
      for (int k = 0; k < m; ++k) {
        const float v = q[c][k] < 0.f ? 0.f : (q[c][k] > 1.f ? 1.f : q[c][k]);
        u[c][k] = static_cast<std::uint8_t>(static_cast<int>(v * 255.f + 0.5f));
      }
    }
    std::uint8_t* o = out + 4 * static_cast<std::size_t>(k0);
    for (int k = 0; k < m; ++k) {
      o[4 * k] = u[0][k];
      o[4 * k + 1] = u[1][k];
      o[4 * k + 2] = u[2][k];
      o[4 * k + 3] = u[3][k];
    }
  }
}

class GridRenderer final : public Renderer {
 public:
  GridRenderer(const Effect& e, int size) : e_(e), m_(e.m), S_(size) {
    const Hyper& h = m_.h;
    const int G = h.grid, C = h.channels, H = h.hidden;
    w_.resize(static_cast<std::size_t>(h.bases));
    film_.resize(static_cast<std::size_t>(2 * H));
    slice_.resize(static_cast<std::size_t>(C) * G * G);
    rowg_.resize(static_cast<std::size_t>(C) * G);
    rowf_.resize(static_cast<std::size_t>(C) * S_);
    x0_.resize(static_cast<std::size_t>(S_));
    fx_.resize(static_cast<std::size_t>(S_));
    l0_ = m_.layers[0];  // folded per frame
    const int widest = std::max(H, 4);
    buf_a_.resize(static_cast<std::size_t>(widest) * kB);
    buf_b_.resize(static_cast<std::size_t>(widest) * kB);
    for (int x = 0; x < S_; ++x) {
      const float g = std::clamp((static_cast<float>(x) + 0.5f) / static_cast<float>(S_) * static_cast<float>(G) - 0.5f, 0.f, static_cast<float>(G - 1));
      x0_[static_cast<std::size_t>(x)] = std::min(static_cast<int>(g), G - 2);
      fx_[static_cast<std::size_t>(x)] = g - static_cast<float>(x0_[static_cast<std::size_t>(x)]);
    }
  }

  void render(const FrameInput& in, std::uint8_t* rgba, std::size_t stride) override {
    const Hyper& h = m_.h;
    const int G = h.grid, C = h.channels, H = h.hidden;
    small_dense(m_.basis, in.c, w_);
    small_dense(m_.films[0], in.c, film_);
    // FiLM folded into layer 0: (1 + gamma) (W f + b) + beta.
    const Dense& L0 = m_.layers[0];
    for (int o = 0; o < H; ++o) {
      const float g = 1.f + film_[static_cast<std::size_t>(o)];
      for (int i = 0; i < C; ++i) l0_.w[static_cast<std::size_t>(o) * C + i] = g * L0.w[static_cast<std::size_t>(o) * C + i];
      l0_.b[static_cast<std::size_t>(o)] = g * L0.b[static_cast<std::size_t>(o)] + film_[static_cast<std::size_t>(H + o)];
    }
    blend_slice(m_, in.t, w_, slice_);
    const std::size_t nl = m_.layers.size();
    float out[4 * kB];
    for (int y = 0; y < S_; ++y) {
      const float gy = std::clamp((static_cast<float>(y) + 0.5f) / static_cast<float>(S_) * static_cast<float>(G) - 0.5f, 0.f, static_cast<float>(G - 1));
      const int y0 = std::min(static_cast<int>(gy), G - 2);
      const float fy = gy - static_cast<float>(y0);
      for (int c = 0; c < C; ++c) {  // lerp the two grid rows, then expand along x to full width
        const float* a = slice_.data() + (static_cast<std::size_t>(c) * G + y0) * G;
        float* r = rowg_.data() + static_cast<std::size_t>(c) * G;
        for (int gx = 0; gx < G; ++gx) r[gx] = a[gx] + fy * (a[gx + G] - a[gx]);
        expand_row(r, x0_.data(), fx_.data(), rowf_.data() + static_cast<std::size_t>(c) * S_, S_);
      }
      std::uint8_t* row = rgba + stride * static_cast<std::size_t>(y);
      for (int x0 = 0; x0 < S_; x0 += kB) {
        dense(l0_.w.data(), l0_.b.data(), rowf_.data() + x0, S_, buf_a_.data(), kB, C, H, true);
        float* src = buf_a_.data();
        float* dst = buf_b_.data();
        for (std::size_t l = 1; l + 1 < nl; ++l) {
          const Dense& L = m_.layers[l];
          dense(L.w.data(), L.b.data(), src, kB, dst, kB, L.in, L.out, true);
          std::swap(src, dst);
        }
        const Dense& head = m_.layers[nl - 1];
        dense(head.w.data(), head.b.data(), src, kB, out, kB, head.in, 4, false);
        write_pixels(out, out + kB, out + 2 * kB, out + 3 * kB, kB, in, row + 4 * static_cast<std::size_t>(x0));
      }
    }
  }

  std::size_t scratch_bytes() const override {
    return 4 * (w_.size() + film_.size() + slice_.size() + rowg_.size() + rowf_.size() + fx_.size() + l0_.w.size() +
                l0_.b.size() + buf_a_.size() + buf_b_.size() + x0_.size());
  }
  double macs_per_pixel() const override { return m_.macs_per_pixel(S_); }

 private:
  const Effect& e_;
  const Model& m_;
  int S_;
  std::vector<float> w_, film_, slice_, rowg_, rowf_, fx_, buf_a_, buf_b_;
  std::vector<int> x0_;
  Dense l0_;
};

class ConvRenderer final : public Renderer {
 public:
  ConvRenderer(const Effect& e, int size) : e_(e), m_(e.m), S_(size) {
    const Hyper& h = m_.h;
    const std::size_t l = static_cast<std::size_t>(h.latent), s1 = 2 * l, s2 = 4 * l, s3 = 8 * l;
    const std::size_t c0 = static_cast<std::size_t>(h.c0), c1 = static_cast<std::size_t>(h.c1), c2 = static_cast<std::size_t>(h.c2);
    w_.resize(static_cast<std::size_t>(h.bases));
    film0_.resize(2 * c1);
    film1_.resize(2 * c2);
    slice_.resize(c0 * l * l);
    pad0_.resize(c0 * (s1 + 2) * (s1 + 2));
    act0_.resize(c1 * s1 * s1);
    pad1_.resize(c1 * (s2 + 2) * (s2 + 2));
    act1_.resize(c2 * s2 * s2);
    pad2_.resize(c2 * (s3 + 2) * (s3 + 2));
    out_.resize(4 * s3 * s3);
    if (S_ != static_cast<int>(s3)) small_.resize(4 * static_cast<std::size_t>(S_) * S_);
    k0_ = m_.layers[0];
    k1_ = m_.layers[1];
  }

  void render(const FrameInput& in, std::uint8_t* rgba, std::size_t stride) override {
    const Hyper& h = m_.h;
    const int l = h.latent, s1 = 2 * l, s2 = 4 * l, s3 = 8 * l;
    small_dense(m_.basis, in.c, w_);
    small_dense(m_.films[0], in.c, film0_);
    small_dense(m_.films[1], in.c, film1_);
    fold(m_.layers[0], film0_, k0_);
    fold(m_.layers[1], film1_, k1_);
    blend_slice(m_, in.t, w_, slice_);
    upsample2_pad(slice_.data(), pad0_.data(), h.c0, l, l);
    conv3x3(pad0_.data(), k0_.w.data(), k0_.b.data(), act0_.data(), h.c0, h.c1, s1, s1, true);
    upsample2_pad(act0_.data(), pad1_.data(), h.c1, s1, s1);
    conv3x3(pad1_.data(), k1_.w.data(), k1_.b.data(), act1_.data(), h.c1, h.c2, s2, s2, true);
    upsample2_pad(act1_.data(), pad2_.data(), h.c2, s2, s2);
    const Dense& k2 = m_.layers[2];
    conv3x3(pad2_.data(), k2.w.data(), k2.b.data(), out_.data(), h.c2, 4, s3, s3, false);
    const float* src = out_.data();
    int n = s3;
    if (S_ != s3) {  // level of detail: box-filter the native frame down
      const int f = s3 / S_;
      const float inv = 1.f / static_cast<float>(f * f);
      for (int c = 0; c < 4; ++c) {
        for (int y = 0; y < S_; ++y) {
          for (int x = 0; x < S_; ++x) {
            float s = 0;
            for (int j = 0; j < f; ++j) {
              for (int i = 0; i < f; ++i) s += out_[(static_cast<std::size_t>(c) * s3 + y * f + j) * s3 + x * f + i];
            }
            small_[(static_cast<std::size_t>(c) * S_ + y) * S_ + x] = s * inv;
          }
        }
      }
      src = small_.data();
      n = S_;
    }
    const std::size_t plane = static_cast<std::size_t>(n) * n;
    for (int y = 0; y < n; ++y) {
      const std::size_t o = static_cast<std::size_t>(y) * n;
      write_pixels(src + o, src + plane + o, src + 2 * plane + o, src + 3 * plane + o, n, in, rgba + stride * static_cast<std::size_t>(y));
    }
  }

  std::size_t scratch_bytes() const override {
    std::size_t n = w_.size() + film0_.size() + film1_.size() + slice_.size() + pad0_.size() + act0_.size() + pad1_.size() +
                    act1_.size() + pad2_.size() + out_.size() + small_.size() + k0_.w.size() + k0_.b.size() + k1_.w.size() + k1_.b.size();
    return 4 * n;
  }
  double macs_per_pixel() const override { return m_.macs_per_pixel(m_.h.size) * static_cast<double>(m_.h.size) * m_.h.size / (static_cast<double>(S_) * S_); }

 private:
  // FiLM folded into a convolution: output channel o scaled by (1 + gamma_o), bias shifted by beta_o.
  static void fold(const Dense& k, std::span<const float> film, Dense& out) {
    const std::size_t per = static_cast<std::size_t>(k.in);
    for (int o = 0; o < k.out; ++o) {
      const float g = 1.f + film[static_cast<std::size_t>(o)];
      for (std::size_t i = 0; i < per; ++i) out.w[static_cast<std::size_t>(o) * per + i] = g * k.w[static_cast<std::size_t>(o) * per + i];
      out.b[static_cast<std::size_t>(o)] = g * k.b[static_cast<std::size_t>(o)] + film[static_cast<std::size_t>(k.out + o)];
    }
  }

  const Effect& e_;
  const Model& m_;
  int S_;
  std::vector<float> w_, film0_, film1_, slice_, pad0_, act0_, pad1_, act1_, pad2_, out_, small_;
  Dense k0_, k1_;
};

#include "rt_rollout.hpp"

}  // namespace

std::unique_ptr<Renderer> make_renderer(const Effect& e, int size) {
  if (e.m.h.arch == Arch::grid) return std::make_unique<GridRenderer>(e, size);
  return std::make_unique<ConvRenderer>(e, size);
}

std::unique_ptr<RolloutRunner> make_rollout(const RolloutEffect& e, int size) { return std::make_unique<Rollout>(e, size); }

}  // namespace nfx::rt::NFX_NS

#if defined(NFX_PUSHED)
#pragma GCC pop_options
#undef NFX_PUSHED
#endif
