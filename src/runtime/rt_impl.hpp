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

#include "rt_aligned.hpp"
#include "rt_common.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <immintrin.h>
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
  if (!m.plane_bits.empty()) {  // per-plane widths (0 to 8 bits), each plane found by its offset; optionally masked
    const std::size_t mb = packed_plane_bytes(side2, 1);
    const std::uint8_t* mask = m.raw_mask.empty() ? nullptr : m.raw_mask.data() + static_cast<std::size_t>(i) * mb;
    for (std::size_t p = 0; p < planes; ++p) {
      const int bits = m.plane_bits[first_plane + p];
      const float lo = m.raw_ranges[(first_plane + p) * 2], hi = m.raw_ranges[(first_plane + p) * 2 + 1];
      const float base = a * lo, step = bits > 0 ? a * (hi - lo) / static_cast<float>((1 << bits) - 1) : 0.f;
      const std::uint8_t* q = m.raw_u8.data() + m.raw_offsets[first_plane + p];
      float* s = slice.data() + p * side2;
      if (mask) {  // stored points in raster order; the others take the plane's fill
        const float fill = a * m.raw_fill[first_plane + p];
        std::size_t n = 0;
        for (std::size_t j = 0; j < side2; ++j) {
          if ((mask[j >> 3] >> (j & 7)) & 1u) {
            s[j] += base + step * static_cast<float>(bits == 0 ? 0u : bits == 8 ? q[n] : packed_code(q, n, bits));
            ++n;
          } else {
            s[j] += fill;
          }
        }
      } else if (bits == 0) {
        for (std::size_t j = 0; j < side2; ++j) s[j] += base;
      } else if (bits == 8) {
        for (std::size_t j = 0; j < side2; ++j) s[j] += base + step * static_cast<float>(q[j]);
      } else if (bits == 4) {
        for (std::size_t j = 0; j + 1 < side2; j += 2) {
          s[j] += base + step * static_cast<float>(q[j >> 1] & 15u);
          s[j + 1] += base + step * static_cast<float>(q[j >> 1] >> 4);
        }
        if (side2 & 1) s[side2 - 1] += base + step * static_cast<float>(q[side2 >> 1] & 15u);
      } else {
        for (std::size_t j = 0; j < side2; ++j) s[j] += base + step * static_cast<float>(packed_code(q, j, bits));
      }
    }
  } else if (m.vq_bits > 0) {  // vector-quantised: per channel group an index plane into the group's codebook
    const int G = m.vq_groups(), d = m.vq_dim, bits = m.vq_bits;
    const std::size_t pb = packed_plane_bytes(side2, bits), K = std::size_t{1} << bits;
    for (int g = 0; g < G; ++g) {
      const std::uint8_t* q = m.raw_u8.data() + ((static_cast<std::size_t>(k) * h.grid_t + i) * G + static_cast<std::size_t>(g)) * pb;
      const float* cb = m.raw_codebook.data() + static_cast<std::size_t>(g) * K * static_cast<std::size_t>(d);
      float* s = slice.data() + static_cast<std::size_t>(g * d) * side2;
      for (std::size_t j = 0; j < side2; ++j) {
        const float* w = cb + static_cast<std::size_t>(bits == 8 ? q[j] : packed_code(q, j, bits)) * static_cast<std::size_t>(d);
        for (int c = 0; c < d; ++c) s[static_cast<std::size_t>(c) * side2 + j] += a * w[c];
      }
    }
  } else if (m.feature_bits == 8) {
    for (std::size_t p = 0; p < planes; ++p) {
      const float lo = m.raw_ranges[(first_plane + p) * 2], hi = m.raw_ranges[(first_plane + p) * 2 + 1];
      const float base = a * lo, step = a * (hi - lo) / 255.f;
      const std::uint8_t* q = m.raw_u8.data() + (first_plane + p) * side2;
      float* s = slice.data() + p * side2;
      for (std::size_t j = 0; j < side2; ++j) s[j] += base + step * static_cast<float>(q[j]);
    }
  } else if (m.feature_bits < 8) {  // bit-packed codes (model.hpp), decoded as they are read
    const int bits = m.feature_bits;
    const std::size_t pb = packed_plane_bytes(side2, bits);
    for (std::size_t p = 0; p < planes; ++p) {
      const float lo = m.raw_ranges[(first_plane + p) * 2], hi = m.raw_ranges[(first_plane + p) * 2 + 1];
      const float base = a * lo, step = a * (hi - lo) / static_cast<float>((1 << bits) - 1);
      const std::uint8_t* q = m.raw_u8.data() + (first_plane + p) * pb;
      float* s = slice.data() + p * side2;
      if (bits == 4) {
        for (std::size_t j = 0; j + 1 < side2; j += 2) {
          s[j] += base + step * static_cast<float>(q[j >> 1] & 15u);
          s[j + 1] += base + step * static_cast<float>(q[j >> 1] >> 4);
        }
        if (side2 & 1) s[side2 - 1] += base + step * static_cast<float>(q[side2 >> 1] & 15u);
      } else {
        for (std::size_t j = 0; j < side2; ++j) s[j] += base + step * static_cast<float>(packed_code(q, j, bits));
      }
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

// slice += a * plane(k, i) for every channel plane of slice i of basis k of one level of the multi family (study F4),
// [C][side][side]. Masked levels decode their stored points along the slice's runs (model.hpp, mask_runs): runs of
// points that take the plane's fill alternate with runs of stored points, read in raster order.
void accumulate_level(const Model& m, const Volume& v, int k, int i, float a, float* slice) {
  const std::size_t side2 = v.plane_values(), C = static_cast<std::size_t>(v.channels);
  const std::size_t first = v.plane0 + (static_cast<std::size_t>(k) * static_cast<std::size_t>(v.slices) + static_cast<std::size_t>(i)) * C;
  const std::uint8_t* runs = m.raw_mask_at.empty() ? nullptr : m.raw_mask.data() + m.raw_mask_at[v.slice0 + static_cast<std::size_t>(i)];
  for (std::size_t p = 0; p < C; ++p) {
    const int bits = m.plane_bits[first + p];
    const float lo = m.raw_ranges[(first + p) * 2], hi = m.raw_ranges[(first + p) * 2 + 1];
    const float base = a * lo, step = a * (hi - lo) / static_cast<float>((1 << bits) - 1);
    const std::uint8_t* q = m.raw_u8.data() + m.raw_offsets[first + p];
    float* s = slice + p * side2;
    const auto code = [&](std::size_t n) { return static_cast<float>(bits == 8 ? q[n] : bits == 4 ? (q[n >> 1] >> ((n & 1) * 4)) & 15u : packed_code(q, n, bits)); };
    if (!runs) {
      for (std::size_t j = 0; j < side2; ++j) s[j] += base + step * code(j);
      continue;
    }
    const float fill = a * m.raw_fill[first + p];
    const std::uint8_t* r = runs;
    std::size_t j = 0, n = 0;
    bool stored = false;
    while (j < side2) {
      std::size_t len = 0;
      for (int shift = 0;; shift += 7) {  // the run's length (a varint; checked when the file was loaded)
        const std::uint8_t b = *r++;
        len |= static_cast<std::size_t>(b & 0x7f) << shift;
        if (!(b & 0x80)) break;
      }
      const std::size_t end = std::min(side2, j + len);
      if (stored) {
        for (; j < end; ++j) s[j] += base + step * code(n++);
      } else {
        for (; j < end; ++j) s[j] += fill;
      }
      stored = !stored;
    }
  }
}

// The multi family (study F4), float. Per frame: the basis weights and FiLM, FiLM folded into the first layer, the
// Fourier features of time folded into its bias, and every level blended and sliced. Per row: the y half of the
// Fourier features of position folded into the bias, every level's two grid rows blended and expanded along x into
// its channels' rows. The x half of the position features are inputs of their own, constant rows. Then the MLP per
// block, as the grid family's.
class MultiRenderer final : public Renderer {
 public:
  MultiRenderer(const Effect& e, int size) : m_(e.m), S_(size), vols_(volumes(e.m.h)) {
    const Hyper& h = m_.h;
    const int H = h.hidden;
    CT_ = h.feature_channels();
    in_ = CT_ + 2 * h.pe_xy;
    w_.resize(static_cast<std::size_t>(h.bases));
    film_.resize(static_cast<std::size_t>(2 * H));
    gain_.resize(static_cast<std::size_t>(H));
    bias0_.resize(static_cast<std::size_t>(H));
    rowb_.resize(static_cast<std::size_t>(H));
    tpe_.resize(static_cast<std::size_t>(2 * h.pe_t));
    std::size_t total = 0, widest = 0;
    for (const Volume& v : vols_) {
      loff_.push_back(total);
      total += static_cast<std::size_t>(v.channels) * v.plane_values();
      widest = std::max(widest, static_cast<std::size_t>(v.channels) * static_cast<std::size_t>(v.side));
    }
    slice_.resize(total);
    rowg_.resize(widest);
    rowf_.assign(static_cast<std::size_t>(in_) * static_cast<std::size_t>(S_), 0.f);
    xi_.resize(vols_.size() * static_cast<std::size_t>(S_));
    xf_.resize(vols_.size() * static_cast<std::size_t>(S_));
    for (std::size_t l = 0; l < vols_.size(); ++l) {
      const int G = vols_[l].side;
      for (int x = 0; x < S_; ++x) {
        const float g = std::clamp((static_cast<float>(x) + 0.5f) / static_cast<float>(S_) * static_cast<float>(G) - 0.5f, 0.f, static_cast<float>(G - 1));
        const std::size_t at = l * static_cast<std::size_t>(S_) + static_cast<std::size_t>(x);
        xi_[at] = std::min(static_cast<int>(g), G - 2);
        xf_[at] = g - static_cast<float>(xi_[at]);
      }
    }
    // The position features: x's (sin, cos per frequency) as constant input rows after the levels' channels; y's per row.
    ype_.resize(static_cast<std::size_t>(S_) * static_cast<std::size_t>(2 * h.pe_xy));
    std::array<float, 48> pf{};
    for (int x = 0; x < S_; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(S_);
      position_features(h.pe_xy, u, u, pf.data());
      for (int k = 0; k < h.pe_xy; ++k) {
        rowf_[static_cast<std::size_t>(CT_ + 2 * k) * static_cast<std::size_t>(S_) + static_cast<std::size_t>(x)] = pf[static_cast<std::size_t>(4 * k)];
        rowf_[static_cast<std::size_t>(CT_ + 2 * k + 1) * static_cast<std::size_t>(S_) + static_cast<std::size_t>(x)] = pf[static_cast<std::size_t>(4 * k + 1)];
        ype_[static_cast<std::size_t>(x) * static_cast<std::size_t>(2 * h.pe_xy) + static_cast<std::size_t>(2 * k)] = pf[static_cast<std::size_t>(4 * k + 2)];
        ype_[static_cast<std::size_t>(x) * static_cast<std::size_t>(2 * h.pe_xy) + static_cast<std::size_t>(2 * k + 1)] = pf[static_cast<std::size_t>(4 * k + 3)];
      }
    }
    l0_ = Dense(in_, H);
    const int widest_out = std::max(H, 4);
    buf_a_.resize(static_cast<std::size_t>(widest_out) * kB);
    buf_b_.resize(static_cast<std::size_t>(widest_out) * kB);
  }

  void render(const FrameInput& in, std::uint8_t* rgba, std::size_t stride) override {
    const Hyper& h = m_.h;
    const int H = h.hidden, NI = h.mlp_in();
    small_dense(m_.basis, in.c, w_);
    small_dense(m_.films[0], in.c, film_);
    time_features(h.pe_t, h.loop, in.t, tpe_.data());
    // FiLM folded into layer 0, (1 + gamma) (W f + b) + beta, its columns compacted to [levels' channels][x features];
    // the time features into the bias.
    const Dense& L0 = m_.layers[0];
    for (int o = 0; o < H; ++o) {
      const float g = 1.f + film_[static_cast<std::size_t>(o)];
      const float* wr = L0.w.data() + static_cast<std::size_t>(o) * static_cast<std::size_t>(NI);
      float* dst = l0_.w.data() + static_cast<std::size_t>(o) * static_cast<std::size_t>(in_);
      for (int i = 0; i < CT_; ++i) dst[i] = g * wr[i];
      for (int k = 0; k < h.pe_xy; ++k) {
        dst[CT_ + 2 * k] = g * wr[CT_ + 4 * k];
        dst[CT_ + 2 * k + 1] = g * wr[CT_ + 4 * k + 1];
      }
      float b = L0.b[static_cast<std::size_t>(o)];
      for (int k = 0; k < 2 * h.pe_t; ++k) b += wr[CT_ + 4 * h.pe_xy + k] * tpe_[static_cast<std::size_t>(k)];
      gain_[static_cast<std::size_t>(o)] = g;
      bias0_[static_cast<std::size_t>(o)] = g * b + film_[static_cast<std::size_t>(H + o)];
    }
    std::ranges::fill(slice_, 0.f);
    for (std::size_t l = 0; l < vols_.size(); ++l) {
      int i0, i1;
      float ft;
      slice_lerp(vols_[l].slices, h.loop, in.t, i0, i1, ft);
      for (int k = 0; k < h.bases; ++k) {
        const float wk = w_[static_cast<std::size_t>(k)];
        accumulate_level(m_, vols_[l], k, i0, wk * (1.f - ft), slice_.data() + loff_[l]);
        if (ft > 0.f) accumulate_level(m_, vols_[l], k, i1, wk * ft, slice_.data() + loff_[l]);
      }
    }
    const std::size_t nl = m_.layers.size();
    float out[4 * kB];
    for (int y = 0; y < S_; ++y) {
      for (int o = 0; o < H; ++o) {  // the y position features into the row's bias
        float b = bias0_[static_cast<std::size_t>(o)];
        const float* wr = L0.w.data() + static_cast<std::size_t>(o) * static_cast<std::size_t>(NI) + CT_;
        const float* py = ype_.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(2 * h.pe_xy);
        for (int k = 0; k < h.pe_xy; ++k) b += gain_[static_cast<std::size_t>(o)] * (wr[4 * k + 2] * py[2 * k] + wr[4 * k + 3] * py[2 * k + 1]);
        rowb_[static_cast<std::size_t>(o)] = b;
      }
      std::size_t ch0 = 0;
      for (std::size_t l = 0; l < vols_.size(); ++l) {
        const int G = vols_[l].side;
        const float gy = std::clamp((static_cast<float>(y) + 0.5f) / static_cast<float>(S_) * static_cast<float>(G) - 0.5f, 0.f, static_cast<float>(G - 1));
        const int y0 = std::min(static_cast<int>(gy), G - 2);
        const float fy = gy - static_cast<float>(y0);
        for (int c = 0; c < vols_[l].channels; ++c) {
          const float* a = slice_.data() + loff_[l] + (static_cast<std::size_t>(c) * static_cast<std::size_t>(G) + static_cast<std::size_t>(y0)) * static_cast<std::size_t>(G);
          float* r = rowg_.data() + static_cast<std::size_t>(c) * static_cast<std::size_t>(G);
          for (int gx = 0; gx < G; ++gx) r[gx] = a[gx] + fy * (a[gx + G] - a[gx]);
          expand_row(r, xi_.data() + l * static_cast<std::size_t>(S_), xf_.data() + l * static_cast<std::size_t>(S_),
                     rowf_.data() + (ch0 + static_cast<std::size_t>(c)) * static_cast<std::size_t>(S_), S_);
        }
        ch0 += static_cast<std::size_t>(vols_[l].channels);
      }
      std::uint8_t* row = rgba + stride * static_cast<std::size_t>(y);
      for (int x0 = 0; x0 < S_; x0 += kB) {
        dense(l0_.w.data(), rowb_.data(), rowf_.data() + x0, S_, buf_a_.data(), kB, in_, H, true);
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
    return 4 * (w_.size() + film_.size() + gain_.size() + bias0_.size() + rowb_.size() + tpe_.size() + slice_.size() + rowg_.size() +
                rowf_.size() + xi_.size() + xf_.size() + ype_.size() + l0_.w.size() + l0_.b.size() + buf_a_.size() + buf_b_.size()) +
           sizeof(std::size_t) * loff_.size();
  }
  double macs_per_pixel() const override { return m_.macs_per_pixel(S_); }

 private:
  const Model& m_;
  int S_, CT_ = 0, in_ = 0;
  std::vector<Volume> vols_;
  std::vector<std::size_t> loff_;
  std::vector<float> w_, film_, gain_, bias0_, rowb_, tpe_, slice_, rowg_, rowf_, xf_, ype_, buf_a_, buf_b_;
  std::vector<int> xi_;
  Dense l0_;
};

#include "rt_int8.hpp"
#include "rt_rollout.hpp"

}  // namespace

std::unique_ptr<Renderer> make_renderer(const Effect& e, int size, Precision p) {
  if (e.m.h.arch == Arch::grid) {
    if (p == Precision::int8) return std::make_unique<GridRendererQ>(e, size);
    return std::make_unique<GridRenderer>(e, size);
  }
  if (e.m.h.arch == Arch::multi) return std::make_unique<MultiRenderer>(e, size);  // float (no int8 path yet)
  return std::make_unique<ConvRenderer>(e, size);  // the conv family has no int8 path
}

std::unique_ptr<RolloutRunner> make_rollout(const RolloutEffect& e, int size) { return std::make_unique<Rollout>(e, size); }

}  // namespace nfx::rt::NFX_NS

#if defined(NFX_PUSHED)
#pragma GCC pop_options
#undef NFX_PUSHED
#endif
