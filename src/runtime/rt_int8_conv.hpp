// The rollout stepper's 3x3 convolution in 8-bit integers (rt_rollout.hpp at Precision::int8; docs/REPORT.md §13).
// Included twice by rt_int8.hpp, into namespace NFX_QNS, each time after the qdot() of rt_int8_layer.hpp: the build's
// own instructions and, in the AVX-512 build, VNNI. No include guard on purpose.
//
// Input: words of 8-bit activation codes, planar with a border of the zero points' codes, act [words][R + 2][R + 2]
// (four channels per word for quads, two 16-bit ones for pairs). Every ISA gets the same codes and the same integer
// sums: 7-bit activations with 8-bit weights, or 8-bit activations with 7-bit weights, so that pmaddubsw's pairs never
// saturate (2 x 127 x 127 and 2 x 255 x 63 < 32768). Weights: w [out][9 taps][words], tap = 3 ky + kx. Output:
// out [co][R][R] = ReLU(scale[o] * (sum - corr[o]) + bias[o]) in float.
#if !defined(NFX_QNS)
#error "define NFX_QNS (and a qdot() in it) before including rt_int8_conv.hpp"
#endif

namespace NFX_QNS {

// T output channels and V vectors of cells (V * kW consecutive cells of row y from x).
template <int T, int V>
[[gnu::always_inline]] inline void qconv_tile(const QConv& L, const std::int32_t* act, int R, int y, int x, int o, float* out) {
  const int Pw = R + 2;
  const std::ptrdiff_t plane = static_cast<std::ptrdiff_t>(Pw) * Pw;
  qi acc[T][V] = {};
  const std::uint32_t* w = L.w.data() + static_cast<std::size_t>(o) * 9 * static_cast<std::size_t>(L.words);
  for (int wd = 0; wd < L.words; ++wd) {
    const std::int32_t* src = act + wd * plane + static_cast<std::ptrdiff_t>(y) * Pw + x;
    for (int tap = 0; tap < 9; ++tap) {
      const std::int32_t* p = src + (tap / 3) * Pw + tap % 3;
      qi a[V];
      for (int v = 0; v < V; ++v) a[v] = load_q(p + v * kW);
      for (int t = 0; t < T; ++t) {
        const std::uint32_t wt = w[(static_cast<std::size_t>(t) * 9 + static_cast<std::size_t>(tap)) * static_cast<std::size_t>(L.words) + static_cast<std::size_t>(wd)];
        for (int v = 0; v < V; ++v) acc[t][v] = qdot(acc[t][v], a[v], wt);
      }
    }
  }
  for (int t = 0; t < T; ++t) {
    const float s = L.scale[static_cast<std::size_t>(o + t)], b = L.bias[static_cast<std::size_t>(o + t)];
    const std::int32_t k = L.corr[static_cast<std::size_t>(o + t)];
    float* row = out + (static_cast<std::ptrdiff_t>(o + t) * R + y) * R + x;
    for (int v = 0; v < V; ++v) store(row + v * kW, relu(__builtin_convertvector(acc[t][v] - k, vf) * s + b));
  }
}

// One cell at a time (rows that are not a multiple of the vector width): the same sums, lane by lane.
inline void qconv_cell(const QConv& L, const std::int32_t* act, int R, int y, int x, int o, float* out) {
  const int Pw = R + 2;
  const std::ptrdiff_t plane = static_cast<std::ptrdiff_t>(Pw) * Pw;
  const std::uint32_t* w = L.w.data() + static_cast<std::size_t>(o) * 9 * static_cast<std::size_t>(L.words);
  std::int32_t acc = 0;
  for (int wd = 0; wd < L.words; ++wd) {
    for (int tap = 0; tap < 9; ++tap) {
      const auto a = static_cast<std::uint32_t>(act[wd * plane + static_cast<std::ptrdiff_t>(y + tap / 3) * Pw + x + tap % 3]);
      const std::uint32_t wt = w[static_cast<std::size_t>(tap) * static_cast<std::size_t>(L.words) + static_cast<std::size_t>(wd)];
      if (L.quads) {
        for (int k = 0; k < 4; ++k) acc += static_cast<std::int32_t>((a >> (8 * k)) & 0xffu) * static_cast<std::int32_t>(static_cast<std::int8_t>((wt >> (8 * k)) & 0xffu));
      } else {
        for (int k = 0; k < 2; ++k) acc += static_cast<std::int32_t>((a >> (16 * k)) & 0xffffu) * static_cast<std::int32_t>(static_cast<std::int16_t>((wt >> (16 * k)) & 0xffffu));
      }
    }
  }
  acc -= L.corr[static_cast<std::size_t>(o)];
  out[(static_cast<std::ptrdiff_t>(o) * R + y) * R + x] = std::max(0.f, static_cast<float>(acc) * L.scale[static_cast<std::size_t>(o)] + L.bias[static_cast<std::size_t>(o)]);
}

// The layer on an R x R grid.
template <int T, int V>
[[gnu::noinline]] void qconv(const QConv& L, const std::int32_t* act, int R, float* out) {
  for (int y = 0; y < R; ++y) {
    int x = 0;
    for (; x + V * kW <= R; x += V * kW) {
      int o = 0;
      for (; o + T <= L.out; o += T) qconv_tile<T, V>(L, act, R, y, x, o, out);
      for (; o < L.out; ++o) qconv_tile<1, V>(L, act, R, y, x, o, out);
    }
    for (; x + kW <= R; x += kW) {
      int o = 0;
      for (; o + T <= L.out; o += T) qconv_tile<T, 1>(L, act, R, y, x, o, out);
      for (; o < L.out; ++o) qconv_tile<1, 1>(L, act, R, y, x, o, out);
    }
    for (; x < R; ++x) {
      for (int o = 0; o < L.out; ++o) qconv_cell(L, act, R, y, x, o, out);
    }
  }
}

}  // namespace NFX_QNS
