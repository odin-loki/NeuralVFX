// Frame models of the grid family at int8 precision (nvfx_instance_set_precision): included by rt_impl.hpp inside the
// per-ISA namespace, after the target pragma and the float renderers. The float GridRenderer stays the reference; this
// one differs from it in three ways:
//   - the hidden layers (H -> H) multiply 8-bit activations by 8-bit weights and add in 32-bit integers. Weights have
//     one scale per output unit (its largest |w| maps to 127), activations one scale per pixel (its largest unit maps
//     to 255, or 127 on AVX2, below; they follow a ReLU, so none is negative). The sums are dequantised (the unit's
//     scale times the pixel's, plus the bias) and go through the ReLU in float. The first and the output layer stay in
//     float. One scale for all pixels lost 0.46 dB on study A's clips; one per pixel loses 0.008 dB, 0.018 dB at 7 bits
//     (docs/REPORT.md §7);
//   - the first layer is projected: it is linear in the features, which reach a pixel by bilinear interpolation, so it
//     is evaluated once per grid point and frame (FiLM folded in) and its H outputs are interpolated instead of the C
//     features. A pixel then pays H interpolations instead of C interpolations and C x H multiply-adds. Used when the
//     frame has at least as many pixels across as the grid has points, on AVX2 and AVX-512 (the interpolation
//     permutes a vector of grid values per vector of pixels); otherwise the first layer runs per pixel as in float;
//   - the integer products: AVX-512 VNNI (vpdpbusd: four u8 x s8 products added per 32-bit lane) where the CPU has it;
//     on AVX2, pmaddubsw (pairs of products added in 16 bits) with activations of 7 bits (0 to 127, so that no pair
//     saturates; 0.01 dB more lost on study A's clips, 10% faster than pairs of 16-bit values); elsewhere (SSE2,
//     AVX-512 without VNNI) pairs of 16-bit values (pmaddwd), which give VNNI's sums.
// Every buffer is allocated in the constructor; render() allocates nothing.

typedef std::int32_t qi __attribute__((vector_size(kW * 4)));
typedef std::int32_t qiu __attribute__((vector_size(kW * 4), aligned(4)));

inline qi load_q(const std::int32_t* p) { return *reinterpret_cast<const qiu*>(p); }
inline void store_q(std::int32_t* p, qi v) { *reinterpret_cast<qiu*>(p) = v; }

// One hidden layer in 8 bits: the weights of output unit o in words [o * words, (o + 1) * words), four 8-bit weights
// per word (quads, for VNNI and pmaddubsw) or two 16-bit ones (pairs, for pmaddwd), input i in byte or half i % 4
// (i % 2) of word i / 4 (i / 2); the scale of each output unit and its bias.
struct QLayer {
  int in = 0, out = 0, words = 0;
  std::vector<std::uint32_t> w;
  std::vector<float> scale, bias;
  // The last hidden layer feeds the output layer directly (fold_head): head [out][4] holds the output layer's weights
  // times each unit's scale, and bias the bias over the scale, so a unit costs a multiply less.
  std::vector<float> head;
};

// Fold unit o's scale s into the output layer: ReLU(s (sa acc + b / s)) = s ReLU(sa acc + b / s) for s > 0 (a unit with
// no weights keeps s = 1: its sum is 0).
inline void fold_head(QLayer& q, const Dense& head) {
  q.head.assign(static_cast<std::size_t>(q.out) * 4, 0.f);
  for (int o = 0; o < q.out; ++o) {
    float& s = q.scale[static_cast<std::size_t>(o)];
    if (!(s > 0.f)) s = 1.f;
    q.bias[static_cast<std::size_t>(o)] /= s;
    for (int t = 0; t < 4; ++t) {
      q.head[static_cast<std::size_t>(o) * 4 + static_cast<std::size_t>(t)] = head.w[static_cast<std::size_t>(t) * static_cast<std::size_t>(head.in) + static_cast<std::size_t>(o)] * s;
    }
  }
}

inline QLayer quantise_layer(const Dense& d, bool quads) {
  QLayer q;
  q.in = d.in;
  q.out = d.out;
  const int per = quads ? 4 : 2;
  q.words = (d.in + per - 1) / per;
  q.w.assign(static_cast<std::size_t>(q.out) * static_cast<std::size_t>(q.words), 0u);
  q.scale.assign(static_cast<std::size_t>(q.out), 0.f);
  q.bias = d.b;
  for (int o = 0; o < d.out; ++o) {
    const float* row = d.w.data() + static_cast<std::size_t>(o) * static_cast<std::size_t>(d.in);
    float mx = 0.f;
    for (int i = 0; i < d.in; ++i) mx = std::max(mx, std::abs(row[i]));
    const float s = mx / 127.f;
    q.scale[static_cast<std::size_t>(o)] = s;
    if (!(s > 0.f)) continue;  // a unit with no weights: its words stay 0
    for (int i = 0; i < d.in; ++i) {
      const long v = std::clamp(std::lround(row[i] / s), -127l, 127l);
      const std::uint32_t bits = static_cast<std::uint32_t>(v) & (quads ? 0xffu : 0xffffu);
      q.w[static_cast<std::size_t>(o) * static_cast<std::size_t>(q.words) + static_cast<std::size_t>(i / per)] |=
          bits << (quads ? 8 * (i % 4) : 16 * (i % 2));
    }
  }
  return q;
}

inline vf vmaxf(vf a, vf b) { return a > b ? a : b; }

// The largest of h [n][kB] per pixel into mx [kB] (none negative), with four running maxima so that the loop is not one
// chain of latencies.
inline void block_max(const float* h, int n, float* mx) {
  for (int v = 0; v < kV; ++v) {
    vf m[4] = {};
    int c = 0;
    for (; c + 4 <= n; c += 4) {
      for (int j = 0; j < 4; ++j) m[j] = vmaxf(m[j], load(h + (c + j) * kB + v * kW));
    }
    for (; c < n; ++c) m[0] = vmaxf(m[0], load(h + c * kB + v * kW));
    store(mx + v * kW, vmaxf(vmaxf(m[0], m[1]), vmaxf(m[2], m[3])));
  }
}

// The ReLU outputs of a block of kB pixels, h [n][kB] (none negative; mx [kB] the largest per pixel), to 8 bits with one
// scale per pixel: act [words][kB] (four values per 32-bit lane for quads, two 16-bit ones otherwise) and the scales
// sa [kB] (the pixel's largest value / 255; 0 for a pixel whose values are all 0).
template <int Per, int Levels>
inline void quantise_block_(const float* h, int n, const float* mxp, std::int32_t* act, float* sa) {
  const int whole = n / Per;
  constexpr float L = static_cast<float>(Levels);
  for (int v = 0; v < kV; ++v) {
    const vf mx = load(mxp + v * kW);
    const vf inv = mx > 0.f ? L / mx : vf{};
    store(sa + v * kW, mx * (1.f / L));
    const auto q = [&](int c) { return __builtin_convertvector(load(h + c * kB + v * kW) * inv + 0.5f, qi); };  // round (>= 0)
    for (int wd = 0; wd < whole; ++wd) {
      qi word = q(wd * Per);
      for (int k = 1; k < Per; ++k) word |= q(wd * Per + k) << (32 / Per * k);
      store_q(act + wd * kB + v * kW, word);
    }
    if (whole * Per < n) {  // a last word in part
      qi word{};
      for (int c = whole * Per; c < n; ++c) word |= q(c) << (32 / Per * (c - whole * Per));
      store_q(act + whole * kB + v * kW, word);
    }
  }
}
// Words of four 8-bit values (VNNI: 0 to 255; AVX2's pmaddubsw: 0 to 127, so that two products and their sum fit in
// 16 bits) or of two 16-bit ones (pmaddwd: 0 to 255).
enum class Pack { quads255, quads127, pairs255 };
inline void quantise_block(const float* h, int n, const float* mxp, Pack pack, std::int32_t* act, float* sa) {
  if (pack == Pack::quads255) quantise_block_<4, 255>(h, n, mxp, act, sa);
  else if (pack == Pack::quads127) quantise_block_<4, 127>(h, n, mxp, act, sa);
  else quantise_block_<2, 255>(h, n, mxp, act, sa);
}

// The integer layer, compiled twice: with this build's own instructions (pmaddubsw on AVX2, pmaddwd on SSE2 and
// AVX-512BW) and, in the AVX-512 build, with VNNI switched on for it alone (vpdpbusd: four u8 x s8 products per 32-bit
// lane), which the renderer uses only where the CPU has it.
namespace qmadd {
[[gnu::always_inline]] inline qi qdot(qi acc, qi a, std::uint32_t w) {
#if NFX_VW == 16 && defined(__AVX512BW__)
  return acc + reinterpret_cast<qi>(_mm512_madd_epi16(reinterpret_cast<__m512i>(a), _mm512_set1_epi32(static_cast<int>(w))));
#elif NFX_VW == 8 && defined(__AVX2__)
  // quads of 7-bit activations (Pack::quads127) times 8-bit weights: pairs of products added in 16 bits without
  // saturating (2 x 127 x 127 < 32768), then the two pairs in 32 bits
  const __m256i p = _mm256_maddubs_epi16(reinterpret_cast<__m256i>(a), _mm256_set1_epi32(static_cast<int>(w)));
  return acc + reinterpret_cast<qi>(_mm256_madd_epi16(p, _mm256_set1_epi16(1)));
#elif NFX_VW == 4 && defined(__SSE2__)
  return acc + reinterpret_cast<qi>(_mm_madd_epi16(reinterpret_cast<__m128i>(a), _mm_set1_epi32(static_cast<int>(w))));
#else
  for (int k = 0; k < kW; ++k) {  // no SIMD: the same sums, lane by lane
    const auto lo = [](std::uint32_t x) { return static_cast<std::int32_t>(static_cast<std::int16_t>(x & 0xffffu)); };
    const auto hi = [](std::uint32_t x) { return static_cast<std::int32_t>(static_cast<std::int16_t>(x >> 16)); };
    const auto u = static_cast<std::uint32_t>(a[k]);
    acc[k] += lo(u) * lo(w) + hi(u) * hi(w);
  }
  return acc;
#endif
}
}  // namespace qmadd
#define NFX_QNS qmadd
#include "rt_int8_layer.hpp"
#undef NFX_QNS

#if NFX_VW == 16 && defined(NFX_PUSHED)
#pragma GCC push_options
#pragma GCC target("avx512vnni")
namespace qvnni {
[[gnu::always_inline]] inline qi qdot(qi acc, qi a, std::uint32_t w) {
  return reinterpret_cast<qi>(_mm512_dpbusd_epi32(reinterpret_cast<__m512i>(acc), reinterpret_cast<__m512i>(a), _mm512_set1_epi32(static_cast<int>(w))));
}
}  // namespace qvnni
#define NFX_QNS qvnni
#include "rt_int8_layer.hpp"
#undef NFX_QNS
#pragma GCC pop_options
constexpr bool kVnniBuild = true;  // the AVX-512 build has the VNNI kernel (used where the CPU has VNNI)
#else
namespace qvnni = qmadd;  // (never used: vnni_ is false)
constexpr bool kVnniBuild = false;
#endif
// The output layer (RGBA) for one block: out [4][kB] = b + W h, with the weights by input (wt [n][4]: the four of input
// i together) and the biases b [4]. With one vector per block (AVX-512) each unit's sum is split over even and odd
// inputs, so that eight chains of multiply-adds run at once instead of four.
inline void head_block(const float* __restrict wt, const float* __restrict b, int n, const float* __restrict h, float* __restrict out) {
  constexpr int P = kV == 1 ? 2 : 1;
  vf acc[4][kV][P];
  for (int t = 0; t < 4; ++t) {
    for (int v = 0; v < kV; ++v) {
      acc[t][v][0] = splat(b[t]);
      for (int p = 1; p < P; ++p) acc[t][v][p] = vf{};
    }
  }
  int i = 0;
  for (; i + P <= n; i += P) {
    for (int p = 0; p < P; ++p) {
      vf x[kV];
      for (int v = 0; v < kV; ++v) x[v] = load(h + (i + p) * kB + v * kW);
      for (int t = 0; t < 4; ++t) {
        for (int v = 0; v < kV; ++v) acc[t][v][p] += wt[(i + p) * 4 + t] * x[v];
      }
    }
  }
  for (; i < n; ++i) {
    for (int t = 0; t < 4; ++t) {
      for (int v = 0; v < kV; ++v) acc[t][v][0] += wt[i * 4 + t] * load(h + i * kB + v * kW);
    }
  }
  for (int t = 0; t < 4; ++t) {
    for (int v = 0; v < kV; ++v) {
      vf a = acc[t][v][0];
      for (int p = 1; p < P; ++p) a += acc[t][v][p];
      store(out + t * kB + v * kW, a);
    }
  }
}

// Premultiplied RGBA floats of a block (planes r, g, b, a of kB) to RGBA8, with the optional colour matrix: as
// write_pixels(), on vectors, the four bytes of a pixel assembled in one 32-bit lane.
inline void write_block(const float* q, const FrameInput& in, std::uint8_t* out) {
  for (int v = 0; v < kV; ++v) {
    vf c[4];
    for (int ch = 0; ch < 4; ++ch) c[ch] = load(q + ch * kB + v * kW);
    if (in.apply_colour) {
      const auto& M = in.colour;
      const vf r = c[0], g = c[1], b = c[2];
      c[0] = M[0] * r + M[1] * g + M[2] * b;
      c[1] = M[3] * r + M[4] * g + M[5] * b;
      c[2] = M[6] * r + M[7] * g + M[8] * b;
    }
    qi px{};
    for (int ch = 0; ch < 4; ++ch) {
      const vf u = c[ch] < 0.f ? vf{} : (c[ch] > 1.f ? splat(1.f) : c[ch]);
      px |= __builtin_convertvector(u * 255.f + 0.5f, qi) << (8 * ch);
    }
    std::memcpy(out + 4 * v * kW, &px, sizeof px);
  }
}

// The values of grid row r at the pixels of a vector: lane k reads r[base + idx[k]].
inline vf take(const float* r, qi idx) {
#if NFX_VW == 16 && defined(__AVX512F__)
  return reinterpret_cast<vf>(_mm512_permutexvar_ps(reinterpret_cast<__m512i>(idx), reinterpret_cast<__m512>(load(r))));
#elif NFX_VW == 8 && defined(__AVX2__)
  return reinterpret_cast<vf>(_mm256_permutevar8x32_ps(reinterpret_cast<__m256>(load(r)), reinterpret_cast<__m256i>(idx)));
#else
  vf v;
  for (int k = 0; k < kW; ++k) v[k] = r[idx[k]];
  return v;
#endif
}

class GridRendererQ final : public Renderer {
 public:
  GridRendererQ(const Effect& e, int size) : m_(e.m), S_(size) {
    const Hyper& h = m_.h;
    const int G = h.grid, C = h.channels, H = h.hidden;
    if constexpr (kVnniBuild) {
      __builtin_cpu_init();
      vnni_ = __builtin_cpu_supports("avx512vnni");
    }
    pack_ = vnni_ ? Pack::quads255 : kW == 8 ? Pack::quads127 : Pack::pairs255;
    for (std::size_t l = 1; l + 1 < m_.layers.size(); ++l) ql_.push_back(quantise_layer(m_.layers[l], pack_ != Pack::pairs255));
    if (!ql_.empty()) fold_head(ql_.back(), m_.layers.back());
    w_.resize(static_cast<std::size_t>(h.bases));
    film_.resize(static_cast<std::size_t>(2 * H));
    slice_.resize(static_cast<std::size_t>(C) * G * G);
    l0_ = m_.layers[0];
    x0_.resize(static_cast<std::size_t>(S_));
    fx_.resize(static_cast<std::size_t>(S_));
    for (int x = 0; x < S_; ++x) {
      const float g = std::clamp((static_cast<float>(x) + 0.5f) / static_cast<float>(S_) * static_cast<float>(G) - 0.5f, 0.f, static_cast<float>(G - 1));
      x0_[static_cast<std::size_t>(x)] = std::min(static_cast<int>(g), G - 2);
      fx_[static_cast<std::size_t>(x)] = g - static_cast<float>(x0_[static_cast<std::size_t>(x)]);
    }
    // Projected when every vector of pixels reads at most kW consecutive grid points (true when S >= G).
    proj_ = kW >= 8 && S_ >= G;
    if (proj_) {
      base_.resize(static_cast<std::size_t>(S_ / kW));
      idx_.resize(static_cast<std::size_t>(S_));
      for (int x = 0; x < S_; x += kW) {
        const int b = x0_[static_cast<std::size_t>(x)];
        base_[static_cast<std::size_t>(x / kW)] = b;
        for (int k = 0; k < kW; ++k) {
          const int d = x0_[static_cast<std::size_t>(x + k)] - b;
          if (d < 0 || d >= kW) proj_ = false;
          idx_[static_cast<std::size_t>(x + k)] = d;
        }
      }
    }
    if (proj_) {
      Gp_ = G + kW;  // a row padded so that a vector loaded from any base stays inside it
      // padded for the row blend's last vector, which reads past the last row (values that are never used)
      proj_grid_.assign(static_cast<std::size_t>(H) * G * G + static_cast<std::size_t>(H + 1) * G + kW + 1, 0.f);
      rowg_.assign(static_cast<std::size_t>(H) * Gp_, 0.f);
      rowd_.assign(static_cast<std::size_t>(H) * Gp_, 0.f);
    } else {
      base_.clear();
      idx_.clear();
      rowg_.resize(static_cast<std::size_t>(C) * G);
      rowf_.resize(static_cast<std::size_t>(C) * S_);
    }
    const int widest = std::max(H, 4);
    buf_a_.resize(static_cast<std::size_t>(widest) * kB);
    buf_b_.resize(static_cast<std::size_t>(widest) * kB);
    act_.resize(static_cast<std::size_t>((H + 1) / 2) * kB);
    sa_.resize(kB);
    mx_.resize(kB);
    const Dense& head = m_.layers.back();
    head_wt_.resize(static_cast<std::size_t>(head.in) * 4);
    for (int i = 0; i < head.in; ++i) {
      for (int t = 0; t < 4; ++t) head_wt_[static_cast<std::size_t>(i) * 4 + static_cast<std::size_t>(t)] = head.w[static_cast<std::size_t>(t) * static_cast<std::size_t>(head.in) + static_cast<std::size_t>(i)];
    }
  }

  void render(const FrameInput& in, std::uint8_t* rgba, std::size_t stride) override {
    const Hyper& h = m_.h;
    const int G = h.grid, C = h.channels, H = h.hidden;
    small_dense(m_.basis, in.c, w_);
    small_dense(m_.films[0], in.c, film_);
    const Dense& L0 = m_.layers[0];  // FiLM folded into layer 0, as the float path
    for (int o = 0; o < H; ++o) {
      const float g = 1.f + film_[static_cast<std::size_t>(o)];
      for (int i = 0; i < C; ++i) l0_.w[static_cast<std::size_t>(o) * C + i] = g * L0.w[static_cast<std::size_t>(o) * C + i];
      l0_.b[static_cast<std::size_t>(o)] = g * L0.b[static_cast<std::size_t>(o)] + film_[static_cast<std::size_t>(H + o)];
    }
    blend_slice(m_, in.t, w_, slice_);
    if (proj_) project();
    const Dense& head = m_.layers.back();
    float out[4 * kB];
    for (int y = 0; y < S_; ++y) {
      const float gy = std::clamp((static_cast<float>(y) + 0.5f) / static_cast<float>(S_) * static_cast<float>(G) - 0.5f, 0.f, static_cast<float>(G - 1));
      const int y0 = std::min(static_cast<int>(gy), G - 2);
      const float fy = gy - static_cast<float>(y0);
      if (proj_) {  // the projected first layer's two grid rows blended, and the steps along x
        // Whole vectors (a row is a few of them: compiled loops spent more on their checks than on the work); values past
        // G are never read. The steps come from the grid rows, not from r (reading back a row just stored, shifted by
        // one, stalls).
        const std::size_t HG = static_cast<std::size_t>(H) * G;  // grid row y0 of every unit, then row y0 + 1
        for (int o = 0; o < H; ++o) {
          const float* a = proj_grid_.data() + static_cast<std::size_t>(y0) * HG + static_cast<std::size_t>(o) * G;
          float* r = rowg_.data() + static_cast<std::size_t>(o) * Gp_;
          float* d = rowd_.data() + static_cast<std::size_t>(o) * Gp_;
          for (int gx = 0; gx < G; gx += kW) {
            const vf a0 = load(a + gx), b0 = load(a + gx + 1);
            const vf r0 = a0 + fy * (load(a + gx + HG) - a0), r1 = b0 + fy * (load(a + gx + 1 + HG) - b0);
            store(r + gx, r0);
            store(d + gx, r1 - r0);
          }
        }
      } else {
        for (int c = 0; c < C; ++c) {  // as the float path: the features' two grid rows blended, then expanded along x
          const float* a = slice_.data() + (static_cast<std::size_t>(c) * G + y0) * G;
          float* r = rowg_.data() + static_cast<std::size_t>(c) * G;
          for (int gx = 0; gx < G; ++gx) r[gx] = a[gx] + fy * (a[gx + G] - a[gx]);
          expand_row(r, x0_.data(), fx_.data(), rowf_.data() + static_cast<std::size_t>(c) * S_, S_);
        }
      }
      std::uint8_t* row = rgba + stride * static_cast<std::size_t>(y);
      for (int x0 = 0; x0 < S_; x0 += kB) {
        if (proj_) {
          expand_block(x0, buf_a_.data(), mx_.data());
        } else {
          dense(l0_.w.data(), l0_.b.data(), rowf_.data() + x0, S_, buf_a_.data(), kB, C, H, true);
          if (!ql_.empty()) block_max(buf_a_.data(), H, mx_.data());
        }
        float* src = buf_a_.data();
        float* dst = buf_b_.data();
        for (std::size_t l = 0; l < ql_.size(); ++l) {
          const QLayer& L = ql_[l];
          if (l > 0) block_max(src, L.in, mx_.data());
          quantise_block(src, L.in, mx_.data(), pack_, act_.data(), sa_.data());
          if (l + 1 < ql_.size()) {
            if (vnni_) qvnni::qlayer<false>(L, act_.data(), sa_.data(), dst);
            else qmadd::qlayer<false>(L, act_.data(), sa_.data(), dst);
            std::swap(src, dst);
            continue;
          }
          for (int k = 0; k < 4; ++k) std::fill_n(out + k * kB, kB, head.b[static_cast<std::size_t>(k)]);  // the last: into the output layer
          if (vnni_) qvnni::qlayer<true>(L, act_.data(), sa_.data(), out);
          else qmadd::qlayer<true>(L, act_.data(), sa_.data(), out);
        }
        if (ql_.empty()) head_block(head_wt_.data(), head.b.data(), head.in, src, out);
        write_block(out, in, row + 4 * static_cast<std::size_t>(x0));
      }
    }
  }

  std::size_t scratch_bytes() const override {
    std::size_t n = 4 * (w_.size() + film_.size() + slice_.size() + proj_grid_.size() + rowg_.size() + rowd_.size() + rowf_.size() +
                         fx_.size() + x0_.size() + base_.size() + idx_.size() + l0_.w.size() + l0_.b.size() + buf_a_.size() +
                         buf_b_.size() + act_.size() + sa_.size() + mx_.size() + head_wt_.size());
    for (const QLayer& L : ql_) n += 4 * (L.w.size() + L.scale.size() + L.bias.size() + L.head.size());
    return n;
  }
  double macs_per_pixel() const override { return m_.macs_per_pixel(S_); }

 private:
  // The first layer at every grid point: proj_grid_[gy][o][gx] = b_o + sum_i w_oi slice[i][gy][gx]. By grid row, then
  // unit: a frame row reads two grid rows of every unit, side by side (by unit, they were 4 KB apart and fought over
  // the same cache sets).
  void project() {
    const Hyper& h = m_.h;
    const int C = h.channels, H = h.hidden, G = h.grid;
    const std::size_t n = static_cast<std::size_t>(G) * G;
    for (int gy = 0; gy < G; ++gy) {
      const float* s = slice_.data() + static_cast<std::size_t>(gy) * G;
      float* p = proj_grid_.data() + static_cast<std::size_t>(gy) * H * G;
      int gx = 0;
      for (; gx + kW <= G; gx += kW) {  // a vector of points: all C inputs of each unit in a register, the units side by side
        for (int o = 0; o < H; ++o) {
          const float* w = l0_.w.data() + static_cast<std::size_t>(o) * C;
          vf acc = splat(l0_.b[static_cast<std::size_t>(o)]);
          for (int i = 0; i < C; ++i) acc += w[i] * load(s + static_cast<std::size_t>(i) * n + gx);
          store(p + static_cast<std::size_t>(o) * G + gx, acc);
        }
      }
      for (; gx < G; ++gx) {
        for (int o = 0; o < H; ++o) {
          const float* w = l0_.w.data() + static_cast<std::size_t>(o) * C;
          float acc = l0_.b[static_cast<std::size_t>(o)];
          for (int i = 0; i < C; ++i) acc += w[i] * s[static_cast<std::size_t>(i) * n + gx];
          p[static_cast<std::size_t>(o) * G + gx] = acc;
        }
      }
    }
  }

  // The first layer's outputs (ReLU) for the block at x0 from the blended rows, h [H][kB], and their largest per pixel.
  void expand_block(int x0, float* h, float* mx) const {
    const int H = m_.h.hidden;
    for (int v = 0; v < kV; ++v) {
      const int x = x0 + v * kW;
      const int b = base_[static_cast<std::size_t>(x / kW)];
      const qi idx = load_q(idx_.data() + x);
      const vf f = load(fx_.data() + x);
      const float* r = rowg_.data() + b;
      const float* d = rowd_.data() + b;
      const auto one = [&](int o) {
        const vf y = relu(take(r + static_cast<std::size_t>(o) * Gp_, idx) + f * take(d + static_cast<std::size_t>(o) * Gp_, idx));
        store(h + o * kB + v * kW, y);
        return y;
      };
      vf m0{}, m1{}, m2{}, m3{};  // four running maxima (named: an array was kept on the stack)
      int o = 0;
      for (; o + 4 <= H; o += 4) {
        m0 = vmaxf(m0, one(o));
        m1 = vmaxf(m1, one(o + 1));
        m2 = vmaxf(m2, one(o + 2));
        m3 = vmaxf(m3, one(o + 3));
      }
      for (; o < H; ++o) m0 = vmaxf(m0, one(o));
      store(mx + v * kW, vmaxf(vmaxf(m0, m1), vmaxf(m2, m3)));
    }
  }

  const Model& m_;
  int S_, Gp_ = 0;
  bool vnni_ = false, proj_ = false;
  Pack pack_ = Pack::pairs255;
  std::vector<QLayer> ql_;
  std::vector<float> w_, film_, slice_, proj_grid_, rowg_, rowd_, rowf_, fx_, buf_a_, buf_b_, sa_, mx_, head_wt_;
  std::vector<int> x0_, base_;
  std::vector<std::int32_t> idx_;
  std::vector<std::int32_t> act_;
  Dense l0_;
};

// The multi family (study F4) at int8: as GridRendererQ, with the projected first layer per level. The first layer is
// linear in every level's features and in the Fourier features, so each level's part is evaluated once per grid point
// and frame (FiLM folded in) and interpolated at the pixel, the x Fourier features' part is a vector per column, and
// the y and time parts are the row's bias; a pixel pays H interpolations per level. Projected when every level has at
// most as many points across as the frame has pixels, on AVX2 and AVX-512; otherwise the first layer runs per pixel
// in float, as MultiRenderer's. The hidden layers in 8-bit integers as GridRendererQ's.
class MultiRendererQ final : public Renderer {
 public:
  MultiRendererQ(const Effect& e, int size) : m_(e.m), S_(size), vols_(volumes(e.m.h)) {
    const Hyper& h = m_.h;
    const int H = h.hidden;
    if constexpr (kVnniBuild) {
      __builtin_cpu_init();
      vnni_ = __builtin_cpu_supports("avx512vnni");
    }
    pack_ = vnni_ ? Pack::quads255 : kW == 8 ? Pack::quads127 : Pack::pairs255;
    for (std::size_t l = 1; l + 1 < m_.layers.size(); ++l) ql_.push_back(quantise_layer(m_.layers[l], pack_ != Pack::pairs255));
    if (!ql_.empty()) fold_head(ql_.back(), m_.layers.back());
    CT_ = h.feature_channels();
    in_ = CT_ + 2 * h.pe_xy;
    w_.resize(static_cast<std::size_t>(h.bases));
    film_.resize(static_cast<std::size_t>(2 * H));
    gain_.resize(static_cast<std::size_t>(H));
    bias0_.resize(static_cast<std::size_t>(H));
    rowb_.resize(static_cast<std::size_t>(H));
    tpe_.resize(static_cast<std::size_t>(2 * h.pe_t));
    l0_ = Dense(in_, H);
    std::size_t total = 0, widest = 0;
    proj_ = kW >= 8;
    for (const Volume& v : vols_) {
      loff_.push_back(total);
      total += static_cast<std::size_t>(v.channels) * v.plane_values();
      widest = std::max(widest, static_cast<std::size_t>(v.channels) * static_cast<std::size_t>(v.side));
      if (S_ < v.side) proj_ = false;
    }
    slice_.resize(total);
    const std::size_t L = vols_.size(), S = static_cast<std::size_t>(S_);
    xi_.resize(L * S);
    xf_.resize(L * S);
    for (std::size_t l = 0; l < L; ++l) {
      const int G = vols_[l].side;
      for (int x = 0; x < S_; ++x) {
        const float g = std::clamp((static_cast<float>(x) + 0.5f) / static_cast<float>(S_) * static_cast<float>(G) - 0.5f, 0.f, static_cast<float>(G - 1));
        const std::size_t at = l * S + static_cast<std::size_t>(x);
        xi_[at] = std::min(static_cast<int>(g), G - 2);
        xf_[at] = g - static_cast<float>(xi_[at]);
      }
    }
    if (proj_) {  // every vector of pixels reads at most kW consecutive points of each level
      base_.resize(L * (S / static_cast<std::size_t>(kW)));
      idx_.resize(L * S);
      for (std::size_t l = 0; l < L && proj_; ++l) {
        for (int x = 0; x < S_; x += kW) {
          const int b = xi_[l * S + static_cast<std::size_t>(x)];
          base_[l * (S / static_cast<std::size_t>(kW)) + static_cast<std::size_t>(x / kW)] = b;
          for (int k = 0; k < kW; ++k) {
            const int d = xi_[l * S + static_cast<std::size_t>(x + k)] - b;
            if (d < 0 || d >= kW) proj_ = false;
            idx_[l * S + static_cast<std::size_t>(x + k)] = d;
          }
        }
      }
    }
    ype_.resize(S * static_cast<std::size_t>(2 * h.pe_xy));
    std::vector<float> xpe(S * static_cast<std::size_t>(2 * h.pe_xy));  // [x][2 pe_xy]
    std::array<float, 48> pf{};
    for (int x = 0; x < S_; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(S_);
      position_features(h.pe_xy, u, u, pf.data());
      for (int k = 0; k < h.pe_xy; ++k) {
        const std::size_t at = static_cast<std::size_t>(x) * static_cast<std::size_t>(2 * h.pe_xy) + static_cast<std::size_t>(2 * k);
        xpe[at] = pf[static_cast<std::size_t>(4 * k)];
        xpe[at + 1] = pf[static_cast<std::size_t>(4 * k + 1)];
        ype_[at] = pf[static_cast<std::size_t>(4 * k + 2)];
        ype_[at + 1] = pf[static_cast<std::size_t>(4 * k + 3)];
      }
    }
    if (proj_) {
      for (const Volume& v : vols_) {
        const std::size_t G = static_cast<std::size_t>(v.side), Gp = G + static_cast<std::size_t>(kW), Hs = static_cast<std::size_t>(H);
        poff_.push_back(proj_grid_.size());
        proj_grid_.resize(proj_grid_.size() + Hs * G * G + (Hs + 1) * G + static_cast<std::size_t>(kW) + 1, 0.f);  // padded as GridRendererQ's
        roff_.push_back(rowg_.size());
        rowg_.resize(rowg_.size() + Hs * Gp, 0.f);
        rowd_.resize(rowg_.size(), 0.f);
      }
      xpe_ = std::move(xpe);
      ax_.assign(static_cast<std::size_t>(H) * S, 0.f);  // per frame: the x features' part of the first layer, [H][S]
    } else {
      rowgf_.resize(widest);
      rowf_.assign(static_cast<std::size_t>(in_) * S, 0.f);
      for (int x = 0; x < S_; ++x) {  // the x features as constant input rows
        for (int j = 0; j < 2 * h.pe_xy; ++j) {
          rowf_[static_cast<std::size_t>(CT_ + j) * S + static_cast<std::size_t>(x)] = xpe[static_cast<std::size_t>(x) * static_cast<std::size_t>(2 * h.pe_xy) + static_cast<std::size_t>(j)];
        }
      }
    }
    const int widest_out = std::max(H, 4);
    buf_a_.resize(static_cast<std::size_t>(widest_out) * kB);
    buf_b_.resize(static_cast<std::size_t>(widest_out) * kB);
    act_.resize(static_cast<std::size_t>((H + 1) / 2) * kB);
    sa_.resize(kB);
    mx_.resize(kB);
    const Dense& head = m_.layers.back();
    head_wt_.resize(static_cast<std::size_t>(head.in) * 4);
    for (int i = 0; i < head.in; ++i) {
      for (int t = 0; t < 4; ++t) head_wt_[static_cast<std::size_t>(i) * 4 + static_cast<std::size_t>(t)] = head.w[static_cast<std::size_t>(t) * static_cast<std::size_t>(head.in) + static_cast<std::size_t>(i)];
    }
  }

  void render(const FrameInput& in, std::uint8_t* rgba, std::size_t stride) override {
    const Hyper& h = m_.h;
    const int H = h.hidden, NI = h.mlp_in();
    const std::size_t S = static_cast<std::size_t>(S_);
    small_dense(m_.basis, in.c, w_);
    small_dense(m_.films[0], in.c, film_);
    time_features(h.pe_t, h.loop, in.t, tpe_.data());
    const Dense& L0 = m_.layers[0];  // FiLM folded in, columns compacted, time features in the bias: as MultiRenderer
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
    if (proj_) {
      project();
      for (int o = 0; o < H; ++o) {  // the x features' part per column
        float* a = ax_.data() + static_cast<std::size_t>(o) * S;
        const float* wr = l0_.w.data() + static_cast<std::size_t>(o) * static_cast<std::size_t>(in_) + CT_;
        for (std::size_t x = 0; x < S; ++x) {
          float s = 0.f;
          for (int j = 0; j < 2 * h.pe_xy; ++j) s += wr[j] * xpe_[x * static_cast<std::size_t>(2 * h.pe_xy) + static_cast<std::size_t>(j)];
          a[x] = s;
        }
      }
    }
    const Dense& head = m_.layers.back();
    float out[4 * kB];
    for (int y = 0; y < S_; ++y) {
      for (int o = 0; o < H; ++o) {  // the y features into the row's bias
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
        if (proj_) {  // the level's projected rows y0 and y0 + 1 blended, and the steps along x (as GridRendererQ)
          const std::size_t HG = static_cast<std::size_t>(H) * static_cast<std::size_t>(G), Gp = static_cast<std::size_t>(G + kW);
          for (int o = 0; o < H; ++o) {
            const float* a = proj_grid_.data() + poff_[l] + static_cast<std::size_t>(y0) * HG + static_cast<std::size_t>(o) * static_cast<std::size_t>(G);
            float* r = rowg_.data() + roff_[l] + static_cast<std::size_t>(o) * Gp;
            float* d = rowd_.data() + roff_[l] + static_cast<std::size_t>(o) * Gp;
            for (int gx = 0; gx < G; gx += kW) {
              const vf a0 = load(a + gx), b0 = load(a + gx + 1);
              const vf r0 = a0 + fy * (load(a + gx + HG) - a0), r1 = b0 + fy * (load(a + gx + 1 + HG) - b0);
              store(r + gx, r0);
              store(d + gx, r1 - r0);
            }
          }
        } else {
          for (int c = 0; c < vols_[l].channels; ++c) {
            const float* a = slice_.data() + loff_[l] + (static_cast<std::size_t>(c) * static_cast<std::size_t>(G) + static_cast<std::size_t>(y0)) * static_cast<std::size_t>(G);
            float* r = rowgf_.data() + static_cast<std::size_t>(c) * static_cast<std::size_t>(G);
            for (int gx = 0; gx < G; ++gx) r[gx] = a[gx] + fy * (a[gx + G] - a[gx]);
            expand_row(r, xi_.data() + l * S, xf_.data() + l * S, rowf_.data() + (ch0 + static_cast<std::size_t>(c)) * S, S_);
          }
        }
        ch0 += static_cast<std::size_t>(vols_[l].channels);
      }
      std::uint8_t* row = rgba + stride * static_cast<std::size_t>(y);
      for (int x0 = 0; x0 < S_; x0 += kB) {
        if (proj_) {
          expand_block(x0, buf_a_.data(), mx_.data());
        } else {
          dense(l0_.w.data(), rowb_.data(), rowf_.data() + x0, S_, buf_a_.data(), kB, in_, H, true);
          if (!ql_.empty()) block_max(buf_a_.data(), H, mx_.data());
        }
        float* src = buf_a_.data();
        float* dst = buf_b_.data();
        for (std::size_t l = 0; l < ql_.size(); ++l) {
          const QLayer& Lq = ql_[l];
          if (l > 0) block_max(src, Lq.in, mx_.data());
          quantise_block(src, Lq.in, mx_.data(), pack_, act_.data(), sa_.data());
          if (l + 1 < ql_.size()) {
            if (vnni_) qvnni::qlayer<false>(Lq, act_.data(), sa_.data(), dst);
            else qmadd::qlayer<false>(Lq, act_.data(), sa_.data(), dst);
            std::swap(src, dst);
            continue;
          }
          for (int k = 0; k < 4; ++k) std::fill_n(out + k * kB, kB, head.b[static_cast<std::size_t>(k)]);
          if (vnni_) qvnni::qlayer<true>(Lq, act_.data(), sa_.data(), out);
          else qmadd::qlayer<true>(Lq, act_.data(), sa_.data(), out);
        }
        if (ql_.empty()) head_block(head_wt_.data(), head.b.data(), head.in, src, out);
        write_block(out, in, row + 4 * static_cast<std::size_t>(x0));
      }
    }
  }

  std::size_t scratch_bytes() const override {
    std::size_t n = 4 * (w_.size() + film_.size() + gain_.size() + bias0_.size() + rowb_.size() + tpe_.size() + slice_.size() + proj_grid_.size() +
                         rowg_.size() + rowd_.size() + rowgf_.size() + rowf_.size() + xf_.size() + xi_.size() + base_.size() + idx_.size() +
                         ype_.size() + xpe_.size() + ax_.size() + l0_.w.size() + l0_.b.size() + buf_a_.size() + buf_b_.size() + act_.size() +
                         sa_.size() + mx_.size() + head_wt_.size());
    n += sizeof(std::size_t) * (loff_.size() + poff_.size() + roff_.size());
    for (const QLayer& L : ql_) n += 4 * (L.w.size() + L.scale.size() + L.bias.size() + L.head.size());
    return n;
  }
  double macs_per_pixel() const override { return m_.macs_per_pixel(S_); }

 private:
  // Each level's part of the first layer at its grid points: proj[gy][o][gx] = sum_c w_oc slice[c][gy][gx] (by grid row,
  // then unit, as GridRendererQ's; the bias is the row's).
  void project() {
    const int H = m_.h.hidden;
    std::size_t ch0 = 0;
    for (std::size_t l = 0; l < vols_.size(); ++l) {
      const int G = vols_[l].side, C = vols_[l].channels;
      const std::size_t n = static_cast<std::size_t>(G) * static_cast<std::size_t>(G);
      for (int gy = 0; gy < G; ++gy) {
        const float* s = slice_.data() + loff_[l] + static_cast<std::size_t>(gy) * static_cast<std::size_t>(G);
        float* p = proj_grid_.data() + poff_[l] + static_cast<std::size_t>(gy) * static_cast<std::size_t>(H) * static_cast<std::size_t>(G);
        int gx = 0;
        for (; gx + kW <= G; gx += kW) {
          for (int o = 0; o < H; ++o) {
            const float* w = l0_.w.data() + static_cast<std::size_t>(o) * static_cast<std::size_t>(in_) + ch0;
            vf acc{};
            for (int i = 0; i < C; ++i) acc += w[i] * load(s + static_cast<std::size_t>(i) * n + static_cast<std::size_t>(gx));
            store(p + static_cast<std::size_t>(o) * static_cast<std::size_t>(G) + static_cast<std::size_t>(gx), acc);
          }
        }
        for (; gx < G; ++gx) {
          for (int o = 0; o < H; ++o) {
            const float* w = l0_.w.data() + static_cast<std::size_t>(o) * static_cast<std::size_t>(in_) + ch0;
            float acc = 0.f;
            for (int i = 0; i < C; ++i) acc += w[i] * s[static_cast<std::size_t>(i) * n + static_cast<std::size_t>(gx)];
            p[static_cast<std::size_t>(o) * static_cast<std::size_t>(G) + static_cast<std::size_t>(gx)] = acc;
          }
        }
      }
      ch0 += static_cast<std::size_t>(C);
    }
  }

  // The first layer's outputs (ReLU) for the block at x0: the row's bias, the x features' part and every level's
  // interpolated projection; h [H][kB], and their largest per pixel.
  void expand_block(int x0, float* h, float* mx) const {
    switch (vols_.size()) {  // the levels' loop unrolled for the usual counts
      case 1: expand_block_<1>(x0, h, mx); break;
      case 2: expand_block_<2>(x0, h, mx); break;
      case 3: expand_block_<3>(x0, h, mx); break;
      default: expand_block_<0>(x0, h, mx); break;
    }
  }
  template <int NL>  // NL levels (0: any number, at most 16)
  void expand_block_(int x0, float* h, float* mx) const {
    const int H = m_.h.hidden;
    const std::size_t S = static_cast<std::size_t>(S_), L = NL > 0 ? static_cast<std::size_t>(NL) : vols_.size(), per = S / static_cast<std::size_t>(kW);
    const bool px = m_.h.pe_xy > 0;
    for (int v = 0; v < kV; ++v) {
      const int x = x0 + v * kW;
      std::array<qi, 16> idx;  // per level: the lanes' grid points, weights and rows (the same for every unit)
      std::array<vf, 16> f;
      std::array<const float*, 16> r, d;
      std::array<std::size_t, 16> gp;
      for (std::size_t l = 0; l < L; ++l) {
        gp[l] = static_cast<std::size_t>(vols_[l].side + kW);
        const std::size_t b = static_cast<std::size_t>(base_[l * per + static_cast<std::size_t>(x / kW)]);
        idx[l] = load_q(idx_.data() + l * S + static_cast<std::size_t>(x));
        f[l] = load(xf_.data() + l * S + static_cast<std::size_t>(x));
        r[l] = rowg_.data() + roff_[l] + b;
        d[l] = rowd_.data() + roff_[l] + b;
      }
      vf m0{}, m1{};
      for (int o = 0; o < H; ++o) {
        vf acc = splat(rowb_[static_cast<std::size_t>(o)]);
        if (px) acc += load(ax_.data() + static_cast<std::size_t>(o) * S + static_cast<std::size_t>(x));
        for (std::size_t l = 0; l < L; ++l) {
          const std::size_t at = static_cast<std::size_t>(o) * gp[l];
          acc += take(r[l] + at, idx[l]) + f[l] * take(d[l] + at, idx[l]);
        }
        const vf y = relu(acc);
        store(h + o * kB + v * kW, y);
        if (o & 1) m1 = vmaxf(m1, y);
        else m0 = vmaxf(m0, y);
      }
      store(mx + v * kW, vmaxf(m0, m1));
    }
  }

  const Model& m_;
  int S_, CT_ = 0, in_ = 0;
  bool vnni_ = false, proj_ = false;
  Pack pack_ = Pack::pairs255;
  std::vector<Volume> vols_;
  std::vector<QLayer> ql_;
  std::vector<std::size_t> loff_, poff_, roff_;
  std::vector<float> w_, film_, gain_, bias0_, rowb_, tpe_, slice_, proj_grid_, rowg_, rowd_, rowgf_, rowf_, xf_, ype_, xpe_, ax_, buf_a_, buf_b_, sa_, mx_,
      head_wt_;
  std::vector<int> xi_, base_;
  std::vector<std::int32_t> idx_;
  std::vector<std::int32_t> act_;
  Dense l0_;
};
