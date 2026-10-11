// SIMD kernels shared by the Phase 0 prototypes and the runtime: dense layers on blocks of 16 pixels and planar
// 3x3 convolutions. Include inside a per-ISA namespace after `#pragma GCC target` (see src/runtime/rt_impl.hpp),
// with NFX_TILE (outputs per register tile) and NFX_VW (floats per native vector) defined.
// Vectors are GCC vector extensions of the native width; a block of 16 pixels is 16 / NFX_VW of them.
#if !defined(NFX_TILE) || !defined(NFX_VW)
#error "define NFX_TILE and NFX_VW before including simd_kernels.hpp"
#endif

constexpr int kB = 16;        // pixels per block
constexpr int kW = NFX_VW;    // floats per native vector: 4 (SSE2), 8 (AVX2), 16 (AVX-512)
constexpr int kV = kB / kW;   // native vectors per block
static_assert(kB % kW == 0);
typedef float vf __attribute__((vector_size(kW * 4)));
typedef float vfu __attribute__((vector_size(kW * 4), aligned(4)));  // unaligned access

inline vf load(const float* p) { return *reinterpret_cast<const vfu*>(p); }
inline void store(float* p, vf v) { *reinterpret_cast<vfu*>(p) = v; }
// x in every lane, written so that it compiles to one broadcast (x - 0 is x, -0 included). Filling the lanes one by one
// made GCC build 512-bit vectors from four 128-bit stores and one load, a store-forwarding stall each time (most of the
// AVX-512 build's dense layers' time with one vector per block).
inline vf splat(float x) { return x - vf{}; }
inline vf relu(vf v) { return v > 0.f ? v : vf{}; }

// out[o][k] = act(b[o] + sum_i W[o][i] * in[i][k]) for one block of 16 pixels; rows are `stride` floats apart.
// T outputs at a time share each input load; T * kV accumulators stay in registers.
template <int T>
inline void dense_tile(const float* W, const float* b, const float* in, int in_stride, float* out, int out_stride,
                       int nin, int o, bool act) {
  vf acc[T][kV];
  for (int t = 0; t < T; ++t) {
    for (int v = 0; v < kV; ++v) acc[t][v] = splat(b[o + t]);
  }
  for (int i = 0; i < nin; ++i) {
    const float* row = in + static_cast<std::ptrdiff_t>(i) * in_stride;
    vf x[kV];
    for (int v = 0; v < kV; ++v) x[v] = load(row + v * kW);
    for (int t = 0; t < T; ++t) {
      const float w = W[static_cast<std::ptrdiff_t>(o + t) * nin + i];
      for (int v = 0; v < kV; ++v) acc[t][v] += w * x[v];
    }
  }
  for (int t = 0; t < T; ++t) {
    for (int v = 0; v < kV; ++v) {
      store(out + static_cast<std::ptrdiff_t>(o + t) * out_stride + v * kW, act ? relu(acc[t][v]) : acc[t][v]);
    }
  }
}

inline void dense(const float* W, const float* b, const float* in, int in_stride, float* out, int out_stride, int nin,
                  int nout, bool act) {
  int o = 0;
  for (; o + NFX_TILE <= nout; o += NFX_TILE) dense_tile<NFX_TILE>(W, b, in, in_stride, out, out_stride, nin, o, act);
  for (; o + 2 <= nout; o += 2) dense_tile<2>(W, b, in, in_stride, out, out_stride, nin, o, act);
  for (; o < nout; ++o) dense_tile<1>(W, b, in, in_stride, out, out_stride, nin, o, act);
}

// 3x3 convolution, planar: in [ci][h+2][w+2] (zero border), W [co][ci][3][3], out [co][h][w]. Vectorised over 16
// columns at a time, with a scalar tail.
template <int T>
inline void conv_tile(const float* in, const float* W, const float* b, float* out, int ci, int h, int w, int o,
                      bool act) {
  const int ws = w + 2;
  const std::ptrdiff_t plane = static_cast<std::ptrdiff_t>(h + 2) * ws;
  for (int y = 0; y < h; ++y) {
    int x = 0;
    for (; x + kB <= w; x += kB) {
      vf acc[T][kV];
      for (int t = 0; t < T; ++t) {
        for (int v = 0; v < kV; ++v) acc[t][v] = splat(b[o + t]);
      }
      for (int c = 0; c < ci; ++c) {
        const float* src = in + c * plane + static_cast<std::ptrdiff_t>(y) * ws + x;
        for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
            vf p[kV];
            for (int v = 0; v < kV; ++v) p[v] = load(src + ky * ws + kx + v * kW);
            for (int t = 0; t < T; ++t) {
              const float wt = W[((static_cast<std::ptrdiff_t>(o + t) * ci + c) * 3 + ky) * 3 + kx];
              for (int v = 0; v < kV; ++v) acc[t][v] += wt * p[v];
            }
          }
        }
      }
      for (int t = 0; t < T; ++t) {
        for (int v = 0; v < kV; ++v) {
          store(out + (static_cast<std::ptrdiff_t>(o + t) * h + y) * w + x + v * kW, act ? relu(acc[t][v]) : acc[t][v]);
        }
      }
    }
    for (; x < w; ++x) {  // scalar tail for widths that are not a multiple of 16
      for (int t = 0; t < T; ++t) {
        float acc = b[o + t];
        for (int c = 0; c < ci; ++c) {
          const float* src = in + c * plane + static_cast<std::ptrdiff_t>(y) * ws + x;
          for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) acc += W[((static_cast<std::ptrdiff_t>(o + t) * ci + c) * 3 + ky) * 3 + kx] * src[ky * ws + kx];
          }
        }
        out[(static_cast<std::ptrdiff_t>(o + t) * h + y) * w + x] = act ? std::max(0.f, acc) : acc;
      }
    }
  }
}

inline void conv3x3(const float* in, const float* W, const float* b, float* out, int ci, int co, int h, int w,
                    bool act) {
  int o = 0;
  for (; o + NFX_TILE <= co; o += NFX_TILE) conv_tile<NFX_TILE>(in, W, b, out, ci, h, w, o, act);
  for (; o + 2 <= co; o += 2) conv_tile<2>(in, W, b, out, ci, h, w, o, act);
  for (; o < co; ++o) conv_tile<1>(in, W, b, out, ci, h, w, o, act);
}

// Nearest-neighbour 2x upsampling into a zero-bordered buffer: in [c][h][w] -> out [c][2h+2][2w+2].
inline void upsample2_pad(const float* in, float* out, int c, int h, int w) {
  const int ow = 2 * w + 2, oh = 2 * h + 2;
  std::memset(out, 0, sizeof(float) * static_cast<std::size_t>(c) * oh * ow);
  for (int k = 0; k < c; ++k) {
    for (int y = 0; y < h; ++y) {
      const float* s = in + (static_cast<std::ptrdiff_t>(k) * h + y) * w;
      float* d0 = out + (static_cast<std::ptrdiff_t>(k) * oh + 2 * y + 1) * ow + 1;
      for (int x = 0; x < w; ++x) d0[2 * x] = d0[2 * x + 1] = s[x];
      std::memcpy(d0 + ow, d0, sizeof(float) * 2 * w);
    }
  }
}

inline std::uint8_t to_u8(float v) { return static_cast<std::uint8_t>(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f); }

// RGBA8 for one block from SoA rows out[4][16].
inline void block_to_rgba(const float* out, std::uint8_t* rgba) {
  for (int k = 0; k < kB; ++k) {
    for (int ch = 0; ch < 4; ++ch) rgba[4 * k + ch] = to_u8(out[ch * kB + k]);
  }
}

