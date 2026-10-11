// One hidden layer of rt_int8.hpp's renderer at int8, for one block of kB pixels. Included twice by rt_int8.hpp, into
// namespace NFX_QNS, each time after a qdot() of its own: the build's own instructions (namespace qmadd: pmaddubsw on
// AVX2, pmaddwd elsewhere) and, in the AVX-512 build with VNNI switched on for these functions alone, vpdpbusd
// (namespace qvnni). No include guard on purpose.
#if !defined(NFX_QNS)
#error "define NFX_QNS (and a qdot() in it) before including rt_int8_layer.hpp"
#endif

namespace NFX_QNS {

// T output units: out[o][k] = ReLU(scale_o * sa_k * sum + bias_o); or, for the last hidden layer (Head, fold_head()),
// their terms of the output layer added to out [4][kB].
template <int T, bool Head>
[[gnu::always_inline]] inline void qtile(const QLayer& L, const std::int32_t* act, const float* sa, float* out, int o) {
  qi acc[T][kV] = {};
  const std::uint32_t* w = L.w.data() + static_cast<std::size_t>(o) * static_cast<std::size_t>(L.words);
  for (int wd = 0; wd < L.words; ++wd) {
    qi a[kV];
    for (int v = 0; v < kV; ++v) a[v] = load_q(act + wd * kB + v * kW);
    for (int t = 0; t < T; ++t) {
      const std::uint32_t wt = w[t * L.words + wd];
      for (int v = 0; v < kV; ++v) acc[t][v] = qdot(acc[t][v], a[v], wt);
    }
  }
  if constexpr (Head) {  // straight into the output layer's sums, out [4][kB]
    vf hs[4][kV];
    for (int k = 0; k < 4; ++k) {
      for (int v = 0; v < kV; ++v) hs[k][v] = load(out + k * kB + v * kW);
    }
    for (int t = 0; t < T; ++t) {
      const float b = L.bias[static_cast<std::size_t>(o + t)];
      const float* hw = L.head.data() + static_cast<std::size_t>(o + t) * 4;
      for (int v = 0; v < kV; ++v) {
        const vf z = relu(__builtin_convertvector(acc[t][v], vf) * load(sa + v * kW) + b);
        for (int k = 0; k < 4; ++k) hs[k][v] += hw[k] * z;
      }
    }
    for (int k = 0; k < 4; ++k) {
      for (int v = 0; v < kV; ++v) store(out + k * kB + v * kW, hs[k][v]);
    }
  } else {
    for (int t = 0; t < T; ++t) {
      const float s = L.scale[static_cast<std::size_t>(o + t)], b = L.bias[static_cast<std::size_t>(o + t)];
      for (int v = 0; v < kV; ++v) {
        const vf y = __builtin_convertvector(acc[t][v], vf) * (s * load(sa + v * kW)) + b;
        store(out + (o + t) * kB + v * kW, relu(y));
      }
    }
  }
}

// The whole layer: act [words][kB] (quantise_block), sa [kB] the pixels' scales.
template <bool Head>
[[gnu::noinline]] void qlayer(const QLayer& L, const std::int32_t* act, const float* sa, float* out) {
  constexpr int T = kW == 16 ? 8 : kW == 8 ? 4 : 2;  // accumulators T * kV within the register file
  int o = 0;
  for (; o + T <= L.out; o += T) qtile<T, Head>(L, act, sa, out, o);
  for (; o < L.out; ++o) qtile<1, Head>(L, act, sa, out, o);
}

}  // namespace NFX_QNS
