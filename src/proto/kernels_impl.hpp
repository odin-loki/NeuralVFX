// The hot loops and the Phase 0 candidates. This file is compiled once per ISA (kernels_<isa>.cpp), each time
// inside its own namespace NFX_NS, so one binary carries every variant and picks at run time. The ISA is switched on
// with `#pragma GCC target` after the standard headers, so inline library code (std::vector and friends) stays at the
// build's baseline ISA and the linker can never pick an AVX-512 copy of a shared inline function.
// Vectors are GCC vector extensions of the native width (NFX_VW floats); a block of 16 pixels is 16 / NFX_VW of them.
#if !defined(NFX_NS) || !defined(NFX_TILE) || !defined(NFX_VW)
#error "define NFX_NS, NFX_TILE and NFX_VW before including kernels_impl.hpp"
#endif

#include <neuralfx/proto.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numbers>
#include <random>
#include <stdexcept>
#include <vector>

#if defined(NFX_ARCH) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#define NFX_PRAGMA(x) _Pragma(#x)
#define NFX_TARGET(a) NFX_PRAGMA(GCC target(a))
NFX_TARGET(NFX_ARCH)
#define NFX_PUSHED 1
#endif

namespace nfx::proto::NFX_NS {

constexpr int kB = 16;        // pixels per block
constexpr int kW = NFX_VW;    // floats per native vector: 4 (SSE2), 8 (AVX2), 16 (AVX-512)
constexpr int kV = kB / kW;   // native vectors per block
static_assert(kB % kW == 0);
typedef float vf __attribute__((vector_size(kW * 4)));
typedef float vfu __attribute__((vector_size(kW * 4), aligned(4)));  // unaligned access

inline vf load(const float* p) { return *reinterpret_cast<const vfu*>(p); }
inline void store(float* p, vf v) { *reinterpret_cast<vfu*>(p) = v; }
inline vf splat(float x) {
  vf v;
  for (int k = 0; k < kW; ++k) v[k] = x;
  return v;
}
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

// 3x3 convolution, planar: in [ci][h+2][w+2] (zero border), W [co][ci][3][3], out [co][h][w]; w % 16 == 0.
template <int T>
inline void conv_tile(const float* in, const float* W, const float* b, float* out, int ci, int h, int w, int o,
                      bool act) {
  const int ws = w + 2;
  const std::ptrdiff_t plane = static_cast<std::ptrdiff_t>(h + 2) * ws;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; x += kB) {
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

// sin and cos of 2^f * pi * x for f < n, by angle doubling from one sin/cos (what a runtime would do).
inline void fourier(float x, int n, float* dst) {
  float s = std::sin(std::numbers::pi_v<float> * x), c = std::cos(std::numbers::pi_v<float> * x);
  for (int f = 0; f < n; ++f) {
    dst[2 * f] = s;
    dst[2 * f + 1] = c;
    const float s2 = 2.f * s * c, c2 = 1.f - 2.f * s * s;
    s = s2;
    c = c2;
  }
}

// --- weights ----------------------------------------------------------------------------------------------------

struct Rng {
  std::mt19937_64 g;
  explicit Rng(std::uint64_t seed) : g(seed) {}
  float uniform(float a) { return a * (2.f * static_cast<float>(g() >> 40) / static_cast<float>(1ull << 24) - 1.f); }
};

struct Layer {
  int nin = 0, nout = 0;
  std::vector<float> w, b;
  Layer() = default;
  Layer(int in, int out, Rng& r) : nin(in), nout(out), w(static_cast<std::size_t>(in) * out), b(out) {
    const float a = std::sqrt(6.f / static_cast<float>(in));  // He-uniform
    for (auto& x : w) x = r.uniform(a);
    for (auto& x : b) x = r.uniform(0.1f);
  }
  std::size_t params() const { return w.size() + b.size(); }
};

// Hidden stack shared by the MLP heads: layers[0] reads the encoding, the last writes RGBA.
struct Mlp {
  std::vector<Layer> layers;
  std::vector<float> buf_a, buf_b;  // [max width][16]
  Mlp(int nin, int hidden, int n_hidden, Rng& r) {
    int in = nin;
    for (int l = 0; l < n_hidden; ++l) {
      layers.emplace_back(in, hidden, r);
      in = hidden;
    }
    layers.emplace_back(in, 4, r);
    const int widest = std::max(nin, hidden);
    buf_a.assign(static_cast<std::size_t>(widest) * kB, 0.f);
    buf_b.assign(static_cast<std::size_t>(widest) * kB, 0.f);
  }
  std::size_t params() const {
    std::size_t n = 0;
    for (const auto& l : layers) n += l.params();
    return n;
  }
  // Runs layers [first, end) on a block whose input rows are `in_stride` apart; writes out[4][16].
  void run(std::size_t first, const float* in, int in_stride, float* out) {
    const float* src = in;
    int stride = in_stride;
    for (std::size_t l = first; l < layers.size(); ++l) {
      const Layer& L = layers[l];
      const bool last = l + 1 == layers.size();
      float* dst = last ? out : (src == buf_a.data() ? buf_b.data() : buf_a.data());
      dense(L.w.data(), L.b.data(), src, stride, dst, kB, L.nin, L.nout, !last);
      src = dst;
      stride = kB;
    }
  }
  double macs(std::size_t first) const {
    double m = 0;
    for (std::size_t l = first; l < layers.size(); ++l) m += static_cast<double>(layers[l].nin) * layers[l].nout;
    return m;
  }
};

inline void controls_vec(const Controls& c, float* v) {
  v[0] = c.intensity;
  v[1] = c.wind;
  v[2] = c.speed;
  v[3] = c.hue;
  v[4] = c.seed;
}

// --- candidates -----------------------------------------------------------------------------------------------------

// A: coordinate MLP, everything evaluated per pixel. Input = Fourier(x, y, t) + the 5 controls.
class MlpNaive final : public Model {
 public:
  explicit MlpNaive(const Spec& s) : s_(s), rng_(s.seed), nin_(6 * s.freqs + kControls), mlp_(nin_, s.hidden, s.layers, rng_),
                                     enc_(static_cast<std::size_t>(nin_) * kB), out_(4 * kB) {}
  void render(const Controls& c, std::uint8_t* rgba) override {
    const int n = s_.size, f = s_.freqs;
    float tenc[64], ctl[kControls], tmp[64];
    fourier(c.t * 2.f - 1.f, f, tenc);
    controls_vec(c, ctl);
    for (int y = 0; y < n; ++y) {
      const float fy = (static_cast<float>(y) + 0.5f) / static_cast<float>(n) * 2.f - 1.f;
      for (int x0 = 0; x0 < n; x0 += kB) {
        for (int k = 0; k < kB; ++k) {
          const float fx = (static_cast<float>(x0 + k) + 0.5f) / static_cast<float>(n) * 2.f - 1.f;
          int i = 0;
          fourier(fx, f, tmp);
          for (int j = 0; j < 2 * f; ++j) enc_[static_cast<std::size_t>(i++) * kB + k] = tmp[j];
          fourier(fy, f, tmp);
          for (int j = 0; j < 2 * f; ++j) enc_[static_cast<std::size_t>(i++) * kB + k] = tmp[j];
          for (int j = 0; j < 2 * f; ++j) enc_[static_cast<std::size_t>(i++) * kB + k] = tenc[j];
          for (int j = 0; j < kControls; ++j) enc_[static_cast<std::size_t>(i++) * kB + k] = ctl[j];
        }
        mlp_.run(0, enc_.data(), kB, out_.data());
        block_to_rgba(out_.data(), rgba + 4 * (static_cast<std::ptrdiff_t>(y) * n + x0));
      }
    }
  }
  std::size_t param_count() const override { return mlp_.params(); }
  double macs_per_pixel() const override { return mlp_.macs(0); }
  int size() const override { return s_.size; }

 private:
  Spec s_;
  Rng rng_;
  int nin_;
  Mlp mlp_;
  std::vector<float> enc_, out_;
};

// B: the same network with the first layer split by input group. W1 * [enc(x); enc(y); enc(t); ctl] =
// colT[x] + row[y] + frame: the x and y parts are tables of size * hidden computed once, the time and control part
// once per frame, so the first layer costs one add per hidden unit per pixel instead of nin multiply-adds.
class MlpSep final : public Model {
 public:
  explicit MlpSep(const Spec& s)
      : s_(s), rng_(s.seed), nin_(6 * s.freqs + kControls), mlp_(nin_, s.hidden, s.layers, rng_),
        colT_(static_cast<std::size_t>(s.hidden) * s.size), row_(static_cast<std::size_t>(s.size) * s.hidden),
        frame_(s.hidden), h1_(static_cast<std::size_t>(s.hidden) * kB), out_(4 * kB) {
    const Layer& L = mlp_.layers[0];
    const int n = s.size, f = s.freqs, H = s.hidden;
    float e[64];
    for (int p = 0; p < n; ++p) {
      fourier((static_cast<float>(p) + 0.5f) / static_cast<float>(n) * 2.f - 1.f, f, e);
      for (int o = 0; o < H; ++o) {
        float ax = 0.f, ay = 0.f;
        for (int j = 0; j < 2 * f; ++j) {
          ax += L.w[static_cast<std::size_t>(o) * nin_ + j] * e[j];
          ay += L.w[static_cast<std::size_t>(o) * nin_ + 2 * f + j] * e[j];
        }
        colT_[static_cast<std::size_t>(o) * n + p] = ax;
        row_[static_cast<std::size_t>(p) * H + o] = ay;
      }
    }
  }
  void render(const Controls& c, std::uint8_t* rgba) override {
    const Layer& L = mlp_.layers[0];
    const int n = s_.size, f = s_.freqs, H = s_.hidden;
    float in[64];
    fourier(c.t * 2.f - 1.f, f, in);
    controls_vec(c, in + 2 * f);
    for (int o = 0; o < H; ++o) {
      float a = L.b[o];
      for (int j = 0; j < 2 * f + kControls; ++j) a += L.w[static_cast<std::size_t>(o) * nin_ + 4 * f + j] * in[j];
      frame_[o] = a;
    }
    for (int y = 0; y < n; ++y) {
      const float* ry = row_.data() + static_cast<std::size_t>(y) * H;
      for (int x0 = 0; x0 < n; x0 += kB) {
        for (int o = 0; o < H; ++o) {
          const float add = ry[o] + frame_[o];
          for (int v = 0; v < kV; ++v) {
            store(h1_.data() + static_cast<std::size_t>(o) * kB + v * kW,
                  relu(load(colT_.data() + static_cast<std::size_t>(o) * n + x0 + v * kW) + add));
          }
        }
        mlp_.run(1, h1_.data(), kB, out_.data());
        block_to_rgba(out_.data(), rgba + 4 * (static_cast<std::ptrdiff_t>(y) * n + x0));
      }
    }
  }
  std::size_t param_count() const override { return mlp_.params(); }
  double macs_per_pixel() const override { return mlp_.macs(1) + s_.hidden * 0.5; }  // the add counts half
  int size() const override { return s_.size; }

 private:
  Spec s_;
  Rng rng_;
  int nin_;
  Mlp mlp_;
  std::vector<float> colT_, row_, frame_, h1_, out_;
};

// C: a dense learned feature volume [grid_t][C][grid][grid] sampled trilinearly, then a tiny MLP. The controls enter
// the first layer as a per-frame bias. Per frame: slice in t; per row: lerp in y then expand in x to full width.
class GridMlp final : public Model {
 public:
  explicit GridMlp(const Spec& s)
      : s_(s), rng_(s.seed), C_(s.channels), grid_(static_cast<std::size_t>(s.grid_t) * C_ * s.grid * s.grid),
        mlp_(C_ + kControls, s.hidden, s.layers, rng_), slice_(static_cast<std::size_t>(C_) * s.grid * s.grid),
        rowg_(static_cast<std::size_t>(C_) * s.grid), rowf_(static_cast<std::size_t>(C_) * s.size),
        x0_(s.size), fx_(s.size), out_(4 * kB) {
    for (auto& v : grid_) v = rng_.uniform(1.f);
    for (int x = 0; x < s.size; ++x) {
      const float g = std::clamp((static_cast<float>(x) + 0.5f) / static_cast<float>(s.size) * static_cast<float>(s.grid) - 0.5f, 0.f,
                                 static_cast<float>(s.grid - 1) - 1e-4f);
      x0_[x] = static_cast<int>(g);
      fx_[x] = g - static_cast<float>(x0_[x]);
    }
    // the first layer reads C features (from rowf_) plus the controls (folded into the first-layer bias per frame)
    first_ = mlp_.layers[0];
    Layer feat;
    feat.nin = C_;
    feat.nout = s.hidden;
    feat.w.resize(static_cast<std::size_t>(C_) * s.hidden);
    for (int o = 0; o < s.hidden; ++o) {
      for (int i = 0; i < C_; ++i) feat.w[static_cast<std::size_t>(o) * C_ + i] = first_.w[static_cast<std::size_t>(o) * (C_ + kControls) + i];
    }
    feat.b.assign(s.hidden, 0.f);
    mlp_.layers[0] = std::move(feat);
  }
  void render(const Controls& c, std::uint8_t* rgba) override {
    const int n = s_.size, G = s_.grid, H = s_.hidden;
    // controls -> first-layer bias
    float ctl[kControls];
    controls_vec(c, ctl);
    for (int o = 0; o < H; ++o) {
      float a = first_.b[o];
      for (int j = 0; j < kControls; ++j) a += first_.w[static_cast<std::size_t>(o) * (C_ + kControls) + C_ + j] * ctl[j];
      mlp_.layers[0].b[o] = a;
    }
    // time slice
    const float gt = std::min(c.t * static_cast<float>(s_.grid_t - 1), static_cast<float>(s_.grid_t - 1) - 1e-4f);
    const int t0 = static_cast<int>(gt);
    const float ft = gt - static_cast<float>(t0);
    const std::size_t vol = slice_.size();
    for (std::size_t i = 0; i < vol; ++i) {
      slice_[i] = grid_[t0 * vol + i] + ft * (grid_[(t0 + 1) * vol + i] - grid_[t0 * vol + i]);
    }
    for (int y = 0; y < n; ++y) {
      const float gy = std::clamp((static_cast<float>(y) + 0.5f) / static_cast<float>(n) * static_cast<float>(G) - 0.5f, 0.f,
                                  static_cast<float>(G - 1) - 1e-4f);
      const int y0 = static_cast<int>(gy);
      const float fy = gy - static_cast<float>(y0);
      for (int ch = 0; ch < C_; ++ch) {
        const float* a = slice_.data() + (static_cast<std::size_t>(ch) * G + y0) * G;
        float* r = rowg_.data() + static_cast<std::size_t>(ch) * G;
        for (int gx = 0; gx < G; ++gx) r[gx] = a[gx] + fy * (a[gx + G] - a[gx]);
        float* d = rowf_.data() + static_cast<std::size_t>(ch) * n;
        for (int x = 0; x < n; ++x) d[x] = r[x0_[x]] + fx_[x] * (r[x0_[x] + 1] - r[x0_[x]]);
      }
      for (int x0 = 0; x0 < n; x0 += kB) {
        mlp_.run(0, rowf_.data() + x0, n, out_.data());
        block_to_rgba(out_.data(), rgba + 4 * (static_cast<std::ptrdiff_t>(y) * n + x0));
      }
    }
  }
  std::size_t param_count() const override { return grid_.size() + mlp_.params() + static_cast<std::size_t>(kControls) * s_.hidden; }
  double macs_per_pixel() const override { return 2.0 * C_ + mlp_.macs(0); }
  int size() const override { return s_.size; }

 private:
  Spec s_;
  Rng rng_;
  int C_;
  std::vector<float> grid_;
  Mlp mlp_;
  Layer first_;
  std::vector<float> slice_, rowg_, rowf_;
  std::vector<int> x0_;
  std::vector<float> fx_, out_;
};

// D: multiresolution hash encoding over (x, y, t) in the style of instant neural graphics primitives, then a tiny
// MLP. Per pixel and level: 8 hashed corners, trilinear weights, F features each.
class HashMlp final : public Model {
 public:
  explicit HashMlp(const Spec& s)
      : s_(s), rng_(s.seed), L_(s.levels), F_(s.features), T_(1u << s.log2_table),
        table_(static_cast<std::size_t>(L_) * T_ * F_), mlp_(L_ * F_ + kControls, s.hidden, s.layers, rng_),
        enc_(static_cast<std::size_t>(L_) * F_ * kB), out_(4 * kB) {
    for (auto& v : table_) v = rng_.uniform(1e-1f);
    for (int l = 0; l < L_; ++l) {
      const double g = L_ > 1 ? std::pow(static_cast<double>(s.size) / 16.0, static_cast<double>(l) / (L_ - 1)) : 1.0;
      res_.push_back(static_cast<float>(16.0 * g));
      res_t_.push_back(static_cast<float>(4.0 * g));
    }
    first_ = mlp_.layers[0];
    Layer feat;
    feat.nin = L_ * F_;
    feat.nout = s.hidden;
    feat.w.resize(static_cast<std::size_t>(feat.nin) * s.hidden);
    for (int o = 0; o < s.hidden; ++o) {
      for (int i = 0; i < feat.nin; ++i) feat.w[static_cast<std::size_t>(o) * feat.nin + i] = first_.w[static_cast<std::size_t>(o) * first_.nin + i];
    }
    feat.b.assign(s.hidden, 0.f);
    mlp_.layers[0] = std::move(feat);
  }
  void render(const Controls& c, std::uint8_t* rgba) override {
    const int n = s_.size, H = s_.hidden, nf = L_ * F_;
    float ctl[kControls];
    controls_vec(c, ctl);
    for (int o = 0; o < H; ++o) {
      float a = first_.b[o];
      for (int j = 0; j < kControls; ++j) a += first_.w[static_cast<std::size_t>(o) * first_.nin + nf + j] * ctl[j];
      mlp_.layers[0].b[o] = a;
    }
    const std::uint32_t mask = T_ - 1;
    for (int y = 0; y < n; ++y) {
      const float py = (static_cast<float>(y) + 0.5f) / static_cast<float>(n);
      for (int x0 = 0; x0 < n; x0 += kB) {
        for (int l = 0; l < L_; ++l) {
          const float* tab = table_.data() + static_cast<std::size_t>(l) * T_ * F_;
          const float gt = c.t * res_t_[l];
          const auto it = static_cast<std::uint32_t>(gt);
          const float ft = gt - static_cast<float>(it);
          const float gy = py * res_[l];
          const auto iy = static_cast<std::uint32_t>(gy);
          const float fy = gy - static_cast<float>(iy);
          for (int k = 0; k < kB; ++k) {
            const float gx = (static_cast<float>(x0 + k) + 0.5f) / static_cast<float>(n) * res_[l];
            const auto ix = static_cast<std::uint32_t>(gx);
            const float fx = gx - static_cast<float>(ix);
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            for (int corner = 0; corner < 8; ++corner) {
              const std::uint32_t dx = corner & 1, dy = (corner >> 1) & 1, dt = corner >> 2;
              const float w = (dx ? fx : 1.f - fx) * (dy ? fy : 1.f - fy) * (dt ? ft : 1.f - ft);
              const std::uint32_t h = ((ix + dx) ^ ((iy + dy) * 2654435761u) ^ ((it + dt) * 805459861u)) & mask;
              for (int f = 0; f < F_; ++f) acc[f] += w * tab[static_cast<std::size_t>(h) * F_ + f];
            }
            for (int f = 0; f < F_; ++f) enc_[static_cast<std::size_t>(l * F_ + f) * kB + k] = acc[f];
          }
        }
        mlp_.run(0, enc_.data(), kB, out_.data());
        block_to_rgba(out_.data(), rgba + 4 * (static_cast<std::ptrdiff_t>(y) * n + x0));
      }
    }
  }
  std::size_t param_count() const override { return table_.size() + mlp_.params() + static_cast<std::size_t>(kControls) * s_.hidden; }
  double macs_per_pixel() const override { return 8.0 * L_ * (F_ + 2) + mlp_.macs(0); }  // weights + gathers
  int size() const override { return s_.size; }

 private:
  Spec s_;
  Rng rng_;
  int L_, F_;
  std::uint32_t T_;
  std::vector<float> table_;
  Mlp mlp_;
  Layer first_;
  std::vector<float> res_, res_t_, enc_, out_;
};

// E: a small convolutional decoder. A learned latent volume [grid_t][c0][latent][latent] is sliced in t and modulated
// per channel by the controls (scale and shift), then three (2x nearest upsample + 3x3 conv) stages reach full size.
class ConvDec final : public Model {
 public:
  explicit ConvDec(const Spec& s)
      : s_(s), rng_(s.seed), l_(s.latent),
        lat_(static_cast<std::size_t>(s.grid_t) * s.c0 * l_ * l_), cur_(static_cast<std::size_t>(s.c0) * l_ * l_) {
    if (s.size != 8 * s.latent) throw std::invalid_argument("conv_dec: size must be 8 * latent");
    for (auto& v : lat_) v = rng_.uniform(1.f);
    film_ = Layer(kControls, 2 * s.c0, rng_);
    const int ch[4] = {s.c0, s.c1, s.c2, 4};
    for (int k = 0; k < 3; ++k) conv_.emplace_back(9 * ch[k], ch[k + 1], rng_);  // weights [co][ci][3][3]
    int side = l_;
    for (int k = 0; k < 3; ++k) {
      pad_.emplace_back(static_cast<std::size_t>(ch[k]) * (2 * side + 2) * (2 * side + 2));
      side *= 2;
      act_.emplace_back(static_cast<std::size_t>(ch[k + 1]) * side * side);
    }
  }
  void render(const Controls& c, std::uint8_t* rgba) override {
    const int c0 = s_.c0, n = s_.size;
    float ctl[kControls];
    controls_vec(c, ctl);
    const float gt = std::min(c.t * static_cast<float>(s_.grid_t - 1), static_cast<float>(s_.grid_t - 1) - 1e-4f);
    const int t0 = static_cast<int>(gt);
    const float ft = gt - static_cast<float>(t0);
    const std::size_t vol = cur_.size(), plane = static_cast<std::size_t>(l_) * l_;
    for (int ch = 0; ch < c0; ++ch) {
      float sc = film_.b[ch], sh = film_.b[c0 + ch];
      for (int j = 0; j < kControls; ++j) {
        sc += film_.w[static_cast<std::size_t>(ch) * kControls + j] * ctl[j];
        sh += film_.w[static_cast<std::size_t>(c0 + ch) * kControls + j] * ctl[j];
      }
      for (std::size_t i = ch * plane; i < (ch + 1) * plane; ++i) {
        const float v = lat_[t0 * vol + i] + ft * (lat_[(t0 + 1) * vol + i] - lat_[t0 * vol + i]);
        cur_[i] = (1.f + sc) * v + sh;
      }
    }
    const int ch[4] = {s_.c0, s_.c1, s_.c2, 4};
    const float* src = cur_.data();
    int side = l_;
    for (int k = 0; k < 3; ++k) {
      upsample2_pad(src, pad_[k].data(), ch[k], side, side);
      side *= 2;
      conv3x3(pad_[k].data(), conv_[k].w.data(), conv_[k].b.data(), act_[k].data(), ch[k], ch[k + 1], side, side, k < 2);
      src = act_[k].data();
    }
    const std::size_t px = static_cast<std::size_t>(n) * n;
    for (std::size_t i = 0; i < px; ++i) {
      for (int k = 0; k < 4; ++k) rgba[4 * i + k] = to_u8(src[k * px + i]);
    }
  }
  std::size_t param_count() const override {
    std::size_t p = lat_.size() + film_.params();
    for (const auto& l : conv_) p += l.params();
    return p;
  }
  double macs_per_pixel() const override {
    double m = 0;
    int side = l_;
    const int ch[4] = {s_.c0, s_.c1, s_.c2, 4};
    for (int k = 0; k < 3; ++k) {
      side *= 2;
      m += 9.0 * ch[k] * ch[k + 1] * side * side;
    }
    return m / (static_cast<double>(s_.size) * s_.size);
  }
  int size() const override { return s_.size; }

 private:
  Spec s_;
  Rng rng_;
  int l_;
  std::vector<float> lat_;
  Layer film_;
  std::vector<float> cur_;
  std::vector<Layer> conv_;
  std::vector<std::vector<float>> pad_, act_;
};

inline std::unique_ptr<Model> make(const Spec& s) {
  if (s.size <= 0 || s.size % kB != 0) throw std::invalid_argument("size must be a positive multiple of 16");
  if (s.hidden <= 0 || s.layers < 1 || s.freqs < 1 || s.freqs > 8) throw std::invalid_argument("bad MLP shape");
  if (s.kind == "mlp_naive") return std::make_unique<MlpNaive>(s);
  if (s.kind == "mlp_sep") return std::make_unique<MlpSep>(s);
  if (s.kind == "grid_mlp") {
    if (s.grid < 2 || s.grid_t < 2 || s.channels < 1) throw std::invalid_argument("bad grid");
    return std::make_unique<GridMlp>(s);
  }
  if (s.kind == "hash_mlp") {
    if (s.levels < 1 || s.features < 1 || s.features > 4 || s.log2_table < 8 || s.log2_table > 24) {
      throw std::invalid_argument("bad hash grid");
    }
    return std::make_unique<HashMlp>(s);
  }
  if (s.kind == "conv_dec") {
    if (s.grid_t < 2 || s.c0 < 1 || s.c1 < 1 || s.c2 < 1) throw std::invalid_argument("bad conv decoder");
    return std::make_unique<ConvDec>(s);
  }
  throw std::invalid_argument("unknown kind: " + s.kind);
}

}  // namespace nfx::proto::NFX_NS

namespace nfx::proto::NFX_NS {
// The one out-of-line entry point of this ISA (dispatch.cpp).
std::unique_ptr<Model> create(const Spec& s) { return make(s); }
}  // namespace nfx::proto::NFX_NS

#if defined(NFX_PUSHED)
#pragma GCC pop_options
#endif
