// Rollout effects in the runtime: included by rt_impl.hpp inside the per-ISA namespace (after the target pragma), so
// the loops below are compiled once per ISA. The operations are those of the reference (src/core/rollout.cpp); what
// differs is bookkeeping:
//   - FiLM is folded into the stepper's convolution weights once per frame;
//   - the procedural noise on fixed points (coarse cells, swirl lattice) is value noise, which is a blend of two 2D
//     slices in its third coordinate; the slices are cached and recomputed only when time crosses a lattice step;
//   - the detail layer runs on blocks of kW pixels (one native vector). Bilinear upsampling (from the coarse grid, the
//     swirl lattice and the flicker noise lattices) is separable: each source row is expanded along x once per frame
//     and a fine row only blends two expanded rows. The flicker noise's two time slices are blended once per frame,
//     so a sample is one bilinear interpolation per octave. The MacCormack samples read heat and soot interleaved, so
//     the two corners of a row come with one 16-byte load per pixel, transposed into vectors in registers. The stages
//     run row by row a few rows apart (see detail_step), so their intermediate rows stay in cache;
//   - the renderer's first layer is linear in features that are themselves interpolated from the coarse grid, so its
//     coarse part is evaluated per coarse cell; the rest runs on blocks of 16 pixels with the shared dense kernel.
// Every buffer is allocated in the constructor.

using rollout::kDirSteps;
using rollout::kDirs;
using rollout::kNoise;
using rollout::kPhys;
using rollout::kRenderIn;

// --- lanes ----------------------------------------------------------------------------------------------------------
// The detail layer's kernels are written once, as templates on the lane type: vf for a block of kW pixels, float for
// the pixels left at the end of a row when the size is not a multiple of kW.

typedef std::int32_t vi __attribute__((vector_size(kW * 4)));
typedef float v4 __attribute__((vector_size(16)));
typedef float v4u __attribute__((vector_size(16), aligned(4)));

template <class V>
inline constexpr bool kOne = std::is_same_v<V, float>;
template <class V>
using IntOf = std::conditional_t<kOne<V>, std::int32_t, vi>;

template <class V>
inline V ld(const float* p) {
  if constexpr (kOne<V>) return *p;
  else return load(p);
}
template <class V>
inline void st(float* p, V v) {
  if constexpr (kOne<V>) *p = v;
  else store(p, v);
}
template <class V>
inline V bc(float x) {  // x in every lane (written so that it compiles to one broadcast)
  if constexpr (kOne<V>) return x;
  else return x - V{};
}
template <class V>
inline V vmin(V a, V b) {
  return a < b ? a : b;
}
template <class V>
inline V vmax(V a, V b) {
  return a > b ? a : b;
}
template <class V>
inline IntOf<V> to_int(V v) {  // truncation, for coordinates that are not negative
  if constexpr (kOne<V>) return static_cast<std::int32_t>(v);
  else return __builtin_convertvector(v, vi);
}
template <class V>
inline V to_float(IntOf<V> v) {
  if constexpr (kOne<V>) return static_cast<float>(v);
  else return __builtin_convertvector(v, vf);
}
template <class V>
inline V lane_index() {  // 0, 1, 2, ...
  V v{};
  if constexpr (!kOne<V>) {
    for (int k = 0; k < kW; ++k) v[k] = static_cast<float>(k);
  }
  return v;
}

// Within every group of four lanes, lane s takes element P[s] of the group: 0-3 from a, 4-7 from b.
template <int P0, int P1, int P2, int P3, std::size_t... L>
inline vf shuffle4_(vf a, vf b, std::index_sequence<L...>) {
  constexpr int P[4] = {P0, P1, P2, P3};
  return __builtin_shufflevector(a, b, (static_cast<int>(L / 4 * 4) + (P[L % 4] < 4 ? P[L % 4] : kW + P[L % 4] - 4))...);
}
template <int P0, int P1, int P2, int P3>
inline vf shuffle4(vf a, vf b) {
  return shuffle4_<P0, P1, P2, P3>(a, b, std::make_index_sequence<kW>{});
}

// Half H of the lanes of a and b interleaved (a0, b0, a1, b1, ...).
template <int H, std::size_t... L>
inline vf zip_(vf a, vf b, std::index_sequence<L...>) {
  return __builtin_shufflevector(a, b, (H * kW / 2 + static_cast<int>(L / 2) + (L % 2 ? kW : 0))...);
}

// p[2 j] = a[j], p[2 j + 1] = b[j].
template <class V>
inline void st_pairs(float* p, V a, V b) {
  if constexpr (kOne<V>) {
    p[0] = a;
    p[1] = b;
  } else {
    store(p, zip_<0>(a, b, std::make_index_sequence<kW>{}));
    store(p + kW, zip_<1>(a, b, std::make_index_sequence<kW>{}));
  }
}

inline v4 load4(const float* p) { return *reinterpret_cast<const v4u*>(p); }

// Lanes 4 g to 4 g + 3 hold the four floats at p + o[j + 4 g].
inline vf quad_lanes(const float* p, const std::int32_t* o, int j) {
#if NFX_VW == 4
  return load4(p + o[j]);
#elif NFX_VW == 8
  return __builtin_shufflevector(load4(p + o[j]), load4(p + o[j + 4]), 0, 1, 2, 3, 4, 5, 6, 7);
#else
  typedef float v8 __attribute__((vector_size(32)));
  const v8 a = __builtin_shufflevector(load4(p + o[j]), load4(p + o[j + 4]), 0, 1, 2, 3, 4, 5, 6, 7);
  const v8 b = __builtin_shufflevector(load4(p + o[j + 8]), load4(p + o[j + 12]), 0, 1, 2, 3, 4, 5, 6, 7);
  return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#endif
}

// q[c] = lane c of every group of four lanes of a0 to a3, in the order of the a's: a 4 x 4 transpose within each group.
inline void transpose4(vf a0, vf a1, vf a2, vf a3, vf (&q)[4]) {
  const vf t0 = shuffle4<0, 4, 1, 5>(a0, a1), t1 = shuffle4<2, 6, 3, 7>(a0, a1);
  const vf t2 = shuffle4<0, 4, 1, 5>(a2, a3), t3 = shuffle4<2, 6, 3, 7>(a2, a3);
  q[0] = shuffle4<0, 1, 4, 5>(t0, t2);
  q[1] = shuffle4<2, 3, 6, 7>(t0, t2);
  q[2] = shuffle4<0, 1, 4, 5>(t1, t3);
  q[3] = shuffle4<2, 3, 6, 7>(t1, t3);
}

// The two rows of bilinear stencils, for every lane j: a[c] = p[off[j] + c] and b[c] = p[off[j] + row + c], c = 0 to 3
// (one 16-byte load per lane and row, transposed in registers).
template <class V>
inline void quads2(const float* p, const std::int32_t* off, int row, V (&a)[4], V (&b)[4]) {
  if constexpr (kOne<V>) {
    for (int c = 0; c < 4; ++c) {
      a[c] = p[off[0] + c];
      b[c] = p[off[0] + row + c];
    }
  } else {
    const float* q = p + row;
    transpose4(quad_lanes(p, off, 0), quad_lanes(p, off, 1), quad_lanes(p, off, 2), quad_lanes(p, off, 3), a);
    transpose4(quad_lanes(q, off, 0), quad_lanes(q, off, 1), quad_lanes(q, off, 2), quad_lanes(q, off, 3), b);
  }
}

template <class V>
inline void st_int(std::int32_t* p, IntOf<V> v) {
  if constexpr (kOne<V>) *p = v;
  else std::memcpy(p, &v, sizeof v);
}

// tanh as in Cephes' tanhf, on lanes: x + x^3 P(x^2) for |x| < 0.625, else 1 - 2 / (exp(2 |x|) + 1) with the sign of x,
// and exp by reduction to [-ln 2 / 2, ln 2 / 2], a polynomial and a power of two built from its exponent bits. Within
// about 3e-7 of std::tanh (a few units in the last place), and no library call per value.
template <class V>
inline V vtanh(V x) {
  const V a = vmin<V>(vmax<V>(x, -x), bc<V>(9.f));  // |x|; past 9, tanh is 1 in float
  const V t = a + a;
  const IntOf<V> n = to_int<V>(t * 1.44269504088896341f + 0.5f);  // nearest integer to t / ln 2 (t is not negative)
  const V fn = to_float<V>(n);
  const V r = (t - fn * 0.693359375f) - fn * -2.12194440e-4f;  // t - n ln 2, in two parts
  V p = 1.9875691500e-4f * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  p = p * (r * r) + r + 1.f;
  const V e = p * __builtin_bit_cast(V, (n + 127) << 23);  // exp(t) = exp(r) 2^n
  const V big = 1.f - 2.f / (e + 1.f);
  const V z = x * x;
  V q = -5.70498872745e-3f * z + 2.06390887954e-2f;
  q = q * z - 5.37397155531e-2f;
  q = q * z + 1.33314422036e-1f;
  q = q * z - 3.33332819422e-1f;
  const V small = q * z * x + x;
  return a < 0.625f ? small : (x < 0.f ? -big : big);
}

// f.template operator()<vf>(x) for the whole blocks of a row of `width` pixels, f.template operator()<float>(x) for the
// pixels left over.
template <class F>
inline void each_block(int width, F&& f) {
  int x = 0;
  for (; x + kW <= width; x += kW) f.template operator()<vf>(x);
  for (; x < width; ++x) f.template operator()<float>(x);
}

// each_block() restricted to pixels [a, b): the same blocks (whole blocks of kW from 0, the pixels past the last whole
// block one by one), so a pixel goes through the same code as in each_block(). Returns the pixels covered, [lo, hi).
template <class F>
inline std::array<int, 2> each_block_in(int width, int a, int b, F&& f) {
  const int full = width / kW * kW;
  if (a >= b) return {0, 0};
  int x = a / kW * kW;
  const int lo = a < full ? x : a;
  for (; x < b && x + kW <= width; x += kW) f.template operator()<vf>(x);
  const int hi = b <= full ? x : b;
  for (x = std::max(x, a); x < b; ++x) f.template operator()<float>(x);
  return {lo, hi};
}

// --- noise and interpolation ------------------------------------------------------------------------------------------

// Value noise at fixed 2D points, as a function of the third coordinate z. value_noise(x, y, z) is
// lerp(slice(floor z), slice(floor z + 1), smooth5(frac z)); the two slices are kept per point.
class SliceNoise {
 public:
  void init(std::size_t points, std::uint64_t seed) {
    seed_ = seed;
    x_.assign(points, 0.f);
    y_.assign(points, 0.f);
    a_.assign(points, 0.f);
    b_.assign(points, 0.f);
    iz_ = std::numeric_limits<std::int64_t>::min();
  }
  void set(std::size_t i, float x, float y) {
    x_[i] = x;
    y_[i] = y;
    iz_ = std::numeric_limits<std::int64_t>::min();
  }
  // Refresh the slices if z moved to another lattice step.
  void at(float z) {
    const float fz = std::floor(z);
    const auto iz = static_cast<std::int64_t>(fz);
    tz_ = smooth5(z - fz);
    if (iz == iz_) return;
    if (iz == iz_ + 1) {
      a_.swap(b_);
      fill(b_, static_cast<std::int32_t>(iz + 1));
    } else {
      fill(a_, static_cast<std::int32_t>(iz));
      fill(b_, static_cast<std::int32_t>(iz + 1));
    }
    iz_ = iz;
  }
  float value(std::size_t i) const { return a_[i] + tz_ * (b_[i] - a_[i]); }
  std::size_t bytes() const { return 4 * (x_.size() + y_.size() + a_.size() + b_.size()); }

 private:
  void fill(std::vector<float>& s, std::int32_t iz) const {
    for (std::size_t i = 0; i < s.size(); ++i) {
      const float fx = std::floor(x_[i]), fy = std::floor(y_[i]);
      const auto ix = static_cast<std::int32_t>(fx), iy = static_cast<std::int32_t>(fy);
      const float tx = smooth5(x_[i] - fx), ty = smooth5(y_[i] - fy);
      const float c00 = cell_value(ix, iy, iz, seed_), c10 = cell_value(ix + 1, iy, iz, seed_);
      const float c01 = cell_value(ix, iy + 1, iz, seed_), c11 = cell_value(ix + 1, iy + 1, iz, seed_);
      const float a = c00 + tx * (c10 - c00), b = c01 + tx * (c11 - c01);
      s[i] = a + ty * (b - a);
    }
  }
  std::uint64_t seed_ = 0;
  std::vector<float> x_, y_, a_, b_;
  std::int64_t iz_ = 0;
  float tz_ = 0.f;
};

// Fractal value noise (fbm of value_noise) in a fixed square, as a function of time. Each octave's lattice corner values
// for the two time slices around z are cached and refreshed when z crosses a lattice step, and at() blends them for the
// current time: a sample is then the bilinear interpolation of one lattice with value_noise's smooth5 weights (the same
// corners and weights as value_noise, so the same values up to rounding).
class LatticeFbm {
 public:
  struct Octave {
    std::uint64_t seed = 0;
    int i0 = 0, n = 0;           // the lattice: n x n corners from (i0, i0)
    std::vector<float> a, b, c;  // corner values of the slices below and above z, and their blend at z
    std::int64_t iz = 0;
    float amp = 1.f;  // weight in the sum
  };

  void init(float freq, float rate, int octaves, std::uint64_t seed, float lo, float hi) {
    freq_ = freq;
    rate_ = rate;
    norm_ = 0.f;
    oct_.resize(static_cast<std::size_t>(octaves));
    float f = freq, amp = 1.f;
    for (int o = 0; o < octaves; ++o) {
      Octave& q = oct_[static_cast<std::size_t>(o)];
      q.seed = seed + static_cast<std::uint64_t>(o) * 0x9e3779b97f4a7c15ULL;
      q.i0 = static_cast<int>(std::floor(lo * f)) - 1;
      q.n = static_cast<int>(std::floor(hi * f)) - q.i0 + 3;
      q.a.assign(static_cast<std::size_t>(q.n) * static_cast<std::size_t>(q.n), 0.f);
      q.b.assign(q.a.size(), 0.f);
      q.c.assign(q.a.size(), 0.f);
      q.iz = std::numeric_limits<std::int64_t>::min();
      q.amp = amp;
      norm_ += amp;
      amp *= 0.5f;
      f *= 2.f;
    }
  }
  void at(float t) {
    float z = t * rate_;
    for (Octave& q : oct_) {
      const float fz = std::floor(z);
      const auto iz = static_cast<std::int64_t>(fz);
      if (iz != q.iz) {
        if (iz == q.iz + 1) {
          q.a.swap(q.b);
          fill(q, q.b, static_cast<std::int32_t>(iz + 1));
        } else {
          fill(q, q.a, static_cast<std::int32_t>(iz));
          fill(q, q.b, static_cast<std::int32_t>(iz + 1));
        }
        q.iz = iz;
      }
      const float tz = smooth5(z - fz);
      for (std::size_t i = 0; i < q.c.size(); ++i) q.c[i] = q.a[i] + tz * (q.b[i] - q.a[i]);
      z *= 2.f;
    }
  }
  const std::vector<Octave>& octaves() const { return oct_; }
  float freq() const { return freq_; }
  float norm() const { return norm_; }  // the sum of the octaves' weights
  std::size_t bytes() const {
    std::size_t n = 0;
    for (const Octave& q : oct_) n += 4 * (q.a.size() + q.b.size() + q.c.size());
    return n;
  }

 private:
  static void fill(const Octave& q, std::vector<float>& s, std::int32_t iz) {
    for (int j = 0; j < q.n; ++j) {
      for (int i = 0; i < q.n; ++i) s[static_cast<std::size_t>(j) * static_cast<std::size_t>(q.n) + static_cast<std::size_t>(i)] = cell_value(q.i0 + i, q.i0 + j, iz, q.seed);
    }
  }
  float freq_ = 1.f, rate_ = 1.f, norm_ = 1.f;
  std::vector<Octave> oct_;
};

// Interpolation weights from a grid to fine pixel centres along one axis: pixel p reads grid cells i[p] and i[p] + 1,
// with weight w[p] on the second.
struct Axis {
  std::vector<int> i;
  std::vector<float> w;
};

// Rows of a small grid (n columns of Q interleaved values; Q = 0: as many as init() says) expanded to the fine width
// along x, kept for two grid rows at a time. Fine rows interpolate between grid rows j and j + 1 and walk down the grid
// in order, so each grid row is expanded about once per frame and a fine row only blends two expanded rows.
template <int Q = 0>
class RowPair {
 public:
  // Planes of `stride` floats, of which the first `width` are written (the rest stay zero).
  void init(int width, int planes = Q, int stride = 0) {
    w_ = static_cast<std::size_t>(width);
    q_ = static_cast<std::size_t>(planes);
    stride_ = static_cast<std::size_t>(std::max(width, stride));
    for (auto& r : rows_) r.assign(q_ * stride_, 0.f);
    reset();
  }
  void reset() { key_ = {-1, -1}; }  // the grid changed
  // Rows j and j + 1 of the grid: Q planes of `stride` floats each.
  std::array<const float*, 2> get(const float* grid, int n, const Axis& ax, int j) {
    int s0 = slot(j), s1 = slot(j + 1);
    if (s0 < 0) {
      s0 = s1 == 0 ? 1 : 0;
      expand(grid, n, ax, j, s0);
    }
    if (s1 < 0) {
      s1 = 1 - s0;
      expand(grid, n, ax, j + 1, s1);
    }
    return {rows_[static_cast<std::size_t>(s0)].data(), rows_[static_cast<std::size_t>(s1)].data()};
  }
  std::size_t bytes() const { return 4 * (rows_[0].size() + rows_[1].size()); }

 private:
  int slot(int j) const { return key_[0] == j ? 0 : key_[1] == j ? 1 : -1; }
  void expand(const float* grid, int n, const Axis& ax, int j, int s) {
    const std::size_t nq = Q > 0 ? static_cast<std::size_t>(Q) : q_;
    const float* g = grid + static_cast<std::size_t>(j) * static_cast<std::size_t>(n) * nq;
    float* out = rows_[static_cast<std::size_t>(s)].data();
    for (std::size_t x = 0; x < w_; ++x) {
      const float* a = g + static_cast<std::size_t>(ax.i[x]) * nq;
      const float w = ax.w[x];
      for (std::size_t q = 0; q < nq; ++q) out[q * stride_ + x] = a[q] + w * (a[q + nq] - a[q]);
    }
    key_[static_cast<std::size_t>(s)] = j;
  }
  std::size_t w_ = 0, q_ = 0, stride_ = 0;
  std::array<std::vector<float>, 2> rows_;
  std::array<int, 2> key_{-1, -1};
};

class Rollout final : public RolloutRunner {
 public:
  Rollout(const RolloutEffect& e, int size) : m_(e.m), h_(e.m.h), S_(size) {
    using namespace rollout;
    R_ = h_.res;
    N_ = R_ * R_;
    C_ = h_.channels();
    I_ = h_.inputs();
    H_ = h_.hidden;
    O_ = h_.outputs();
    if (S_ % R_ != 0 || S_ < R_) throw std::invalid_argument("rollout: size must be a multiple of the coarse grid");
    const StepLayout L = step_layout(h_);
    L_ = L;
    w1_.resize(9 * z(I_) * z(H_));
    b1_.resize(z(H_));
    w2_.resize(9 * z(H_) * z(H_));
    b2_.resize(z(H_));
    cond_.assign(z(h_.cond()), 0.f);
    folded_cond_.assign(z(h_.cond()), std::numeric_limits<float>::quiet_NaN());
    X_.assign(z(R_ + 2) * z(R_ + 2) * z(I_), 0.f);  // planar, zero border (written inside only)
    h1p_.assign(z(R_ + 2) * z(R_ + 2) * z(H_), 0.f);
    h1_.resize(z(N_) * z(H_));
    h2_.resize(z(N_) * z(H_));
    d_.resize(z(N_) * z(O_));
    mid_.resize(z(N_) * z(C_));
    next_.resize(z(N_) * z(C_));
    coarse_.resize(z(N_) * z(C_));
    flow_.resize(z(N_) * 2);
    div_.resize(z(N_));
    p_.assign(z(R_ + 2) * z(R_ + 2), 0.f);  // pressure with a zero border (written inside only)
    tmp_.assign(p_.size(), 0.f);
    wot_.resize(z(O_) * z(H_));  // the output layer's weights [out][in], for the dense kernel
    for (int j = 0; j < H_; ++j) {
      for (int k = 0; k < O_; ++k) wot_[z(k) * z(H_) + z(j)] = m_.step_w[L.wo + z(j) * z(O_) + z(k)];
    }
    noise_.resize(z(N_) * kNoise);
    dirsum_.resize(z(N_) * kDirs);
    soot_.assign(z(R_ + 2 * kDirSteps) * z(R_ + 2 * kDirSteps), 0.f);  // zero border, written inside only
    fac_.resize(z(N_) * 4);
    any_.resize(z(R_));
    ext_.assign(z(S_) * 2, 0);
    qmin_.assign(z(S_), 0);
    qmax_.assign(z(S_), 0);
    act_.assign(z(S_) * 2, 0);
    nm_.assign(z(R_) * 2, 0);
    const std::size_t S2 = z(S_) * z(S_);
    ft_.resize(S2);
    fd_.resize(S2);
    rec_.resize(z(S_) * z(record_stride()));  // as many rows as the largest lag can need
    // Rings of padded rows (zero border columns, written inside only): enough slots for every padded row of the frame,
    // a power of two, and one more for a copy of slot 0 (the row above the last slot).
    slots_ = static_cast<int>(std::bit_ceil(z(S_ + 3)));
    td_.assign(z(slots_ + 1) * z(S_ + 3) * 2, 0.f);
    fg_.assign(td_.size(), 0.f);
    cs_t_.assign(z(S_), 0.f);
    cs_d_.assign(z(S_), 0.f);
    off_.assign(z(S_), 0);
    wx_.assign(z(S_), 0.f);
    wy_.assign(z(S_), 0.f);
    r1_.resize(z(h_.render_hidden) * kB);
    r2_.resize(z(h_.render_hidden) * kB);
    out_.resize(4 * kB);
    const RenderLayout RL = render_layout(h_);
    RL_ = RL;
    // Noise on fixed points: coarse cell centres (curl stream function; flicker octaves), the swirl lattice.
    curl_.init(z(N_), 0);
    flicker_.resize(z(m_.noise.flicker_octaves));
    for (auto& f : flicker_) f.init(z(N_), 0);
    sw_spacing_ = 0.5f * m_.detail.swirl_scale;
    sw_n_ = static_cast<int>(std::ceil(130.f / sw_spacing_)) + 3;
    swirl_.init(z(sw_n_) * z(sw_n_), 0);
    swl_.resize(z(sw_n_) * z(sw_n_) * 2);
    ax_ = axis(S_, R_, static_cast<float>(R_) / static_cast<float>(S_), -0.5f);
    // fine pixels whose lock reads coarse cell j (as the first or second cell of their interpolation): [cell_lo, cell_hi)
    cell_lo_.assign(z(R_), S_);
    cell_hi_.assign(z(R_), 0);
    for (int p = 0; p < S_; ++p) {
      for (const int j : {ax_.i[z(p)], ax_.i[z(p)] + 1}) {
        cell_lo_[z(j)] = std::min(cell_lo_[z(j)], p);
        cell_hi_[z(j)] = std::max(cell_hi_[z(j)], p + 1);
      }
    }
    sx_ = axis(S_, sw_n_, 128.f / static_cast<float>(S_) / sw_spacing_, 0.5f / sw_spacing_ + 1.f);
    const int Sb = (S_ + kB - 1) / kB * kB;  // a row padded to whole blocks of 16 pixels
    frow_.assign(2 * z(Sb), 0.f);
    g1_.resize(z(N_) * z(h_.render_hidden));
    render_rows_.init(S_, h_.render_hidden, Sb);
    flow_rows_.init(S_);
    swirl_rows_.init(S_);
    fac_rows_.init(S_);
    noise_rows_.resize(z(m_.noise.flicker_octaves));
    for (auto& r : noise_rows_) r.init(S_);
    reseed(0);  // sizes every noise cache now, so a later begin() with another seed allocates nothing
    // The flicker lattices along x and y (the same for both): fine pixel p is at X = (p + 0.5) / px128 + 0.5 in a
    // 128-pixel frame, octave o at X * freq * 2^o, as in noise_flicker.
    const float px128 = static_cast<float>(S_) / 128.f;
    const auto& oct = fine_flicker_.octaves();
    nax_.resize(oct.size());
    for (auto& a : nax_) {
      a.i.resize(z(S_));
      a.w.resize(z(S_));
    }
    for (int p = 0; p < S_; ++p) {
      float x = ((static_cast<float>(p) + 0.5f) / px128 + 0.5f) * fine_flicker_.freq();
      for (std::size_t o = 0; o < oct.size(); ++o) {
        const float fx = std::floor(x);
        nax_[o].i[z(p)] = static_cast<int>(fx) - oct[o].i0;
        nax_[o].w[z(p)] = smooth5(x - fx);
        x *= 2.f;
      }
    }
  }

  void start(int index, std::span<const float> controls, std::uint64_t seed) override {
    begin(index, seed);
    if (!m_.starts[z(index)].fine_t.empty()) return;
    for (int f = 0; f < h_.warmup; ++f) step(controls, seed);
    since_start_ = m_.detail.swirl_ramp;
  }

  void begin(int index, std::uint64_t seed) override {
    using namespace rollout;
    const StartPoint& sp = m_.starts[z(index)];
    computed_ = 0.0;
    dense_steps_ = 0;
    reseed(seed);
    time_ = sp.time;
    std::fill(coarse_.begin(), coarse_.end(), 0.f);
    for (int i = 0; i < N_; ++i) {
      for (int c = 0; c < kPhys; ++c) coarse_[z(i) * z(C_) + z(c)] = sp.coarse[z(i) * kPhys + z(c)];
    }
    std::fill(p_.begin(), p_.end(), 0.f);
    std::fill(flow_.begin(), flow_.end(), 0.f);
    const bool fine = !sp.fine_t.empty();
    const int n = fine ? h_.start_fine : R_;
    const float k = static_cast<float>(S_) / static_cast<float>(n);
    for (int y = 0; y < S_; ++y) {
      for (int x = 0; x < S_; ++x) {
        const float xs = (static_cast<float>(x) + 0.5f) / k - 0.5f, ys = (static_cast<float>(y) + 0.5f) / k - 0.5f;
        const std::size_t i = z(y) * z(S_) + z(x);
        ft_[i] = fine ? bilinear(sp.fine_t.data(), n, 1, 0, xs, ys) : bilinear(sp.coarse.data(), R_, kPhys, 2, xs, ys);
        fd_[i] = fine ? bilinear(sp.fine_d.data(), n, 1, 0, xs, ys) : bilinear(sp.coarse.data(), R_, kPhys, 3, xs, ys);
      }
    }
    since_start_ = fine ? 0.f : m_.detail.swirl_ramp;
  }

  void step(std::span<const float> controls, std::uint64_t seed) override {
    reseed(seed);
    rollout::condition(m_, controls, time_, cond_);
    fold();
    coarse_noise(time_ + 0.5f / m_.fps);
    coarse_step();
    detail_step(controls);
    time_ += 1.f / m_.fps;
    since_start_ += 1.f / m_.fps;
  }

  void render(const FrameInput& in, std::uint8_t* rgba, std::size_t stride) override {
    using namespace rollout;
    // directional soot sums on the coarse grid, planar [direction][cell], from the soot with a zero border as wide as
    // the sums are long (zero outside, as the reference)
    static constexpr int dirs[kDirs][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
    const int Ps = R_ + 2 * kDirSteps;
    for (int y = 0; y < R_; ++y) {
      for (int x = 0; x < R_; ++x) soot_[z(y + kDirSteps) * z(Ps) + z(x + kDirSteps)] = coarse_[(z(y) * z(R_) + z(x)) * z(C_) + 3];
    }
    for (int j = 0; j < kDirs; ++j) {
      for (int y = 0; y < R_; ++y) {
        float* __restrict o = dirsum_.data() + z(j) * z(N_) + z(y) * z(R_);
        std::fill_n(o, R_, 0.f);
        for (int st = 1; st <= kDirSteps; ++st) {
          const float* __restrict a = soot_.data() + z(y + kDirSteps + st * dirs[j][1]) * z(Ps) + z(kDirSteps + st * dirs[j][0]);
          for (int x = 0; x < R_; ++x) o[x] += a[x];
        }
      }
    }
    const int RH = h_.render_hidden;
    const float* w = m_.render_w.data();
    const float it = 1.f / m_.render_scale[0], id = 1.f / m_.render_scale[1];
    // The first layer is linear and the coarse features (heat, soot, directional sums) reach a pixel by bilinear
    // interpolation, so the layer's coarse part and its bias are evaluated per coarse cell and interpolated instead:
    // g1[cell][j]. Per pixel only the two fine features are left.
    for (int i = 0; i < N_; ++i) {
      std::array<float, 2 + kDirs> f{};
      f[0] = coarse_[z(i) * z(C_) + 2] * it;
      f[1] = coarse_[z(i) * z(C_) + 3] * id;
      for (int j = 0; j < kDirs; ++j) f[z(2 + j)] = dirsum_[z(j) * z(N_) + z(i)] * id;
      for (int j = 0; j < RH; ++j) {
        const float* wj = w + RL_.w1 + z(j) * kRenderIn + 2;
        float s = w[RL_.b1 + z(j)];
        for (std::size_t q = 0; q < f.size(); ++q) s += wj[q] * f[q];
        g1_[z(i) * z(RH) + z(j)] = s;
      }
    }
    render_rows_.reset();
    const std::size_t S = z(S_), Sb = frow_.size() / 2;  // a row, and a row padded to whole blocks
    float* ftn = frow_.data();  // the row's normalised fine features (zero past the end of the row)
    float* fdn = frow_.data() + Sb;
    float* h1 = r1_.data();
    for (int y = 0; y < S_; ++y) {
      const auto g = render_rows_.get(g1_.data(), R_, ax_, ax_.i[z(y)]);
      const float wy = ax_.w[z(y)];
      const float* tf = ft_.data() + z(y) * S;
      const float* df = fd_.data() + z(y) * S;
      for (std::size_t x = 0; x < S; ++x) {
        ftn[x] = tf[x] * it;
        fdn[x] = df[x] * id;
      }
      std::uint8_t* out_row = rgba + stride * z(S_ - 1 - y);
      for (int x0 = 0; x0 < S_; x0 += kB) {  // blocks of 16 pixels (the last one maybe in part)
        vf tv[kV], dv[kV];
        vi lit{};
        for (int v = 0; v < kV; ++v) {
          tv[v] = load(ftn + x0 + v * kW);
          dv[v] = load(fdn + x0 + v * kW);
          lit |= (tv[v] > 0.f) | (dv[v] > 0.f);
        }
        if (skip_) {  // no heat or soot in the block: the material gate is +0, so every byte is 0
          bool any = false;
          for (int l = 0; l < kW; ++l) any |= lit[l] != 0;
          if (!any) {
            std::memset(out_row + 4 * z(x0), 0, 4 * z(std::min(kB, S_ - x0)));
            continue;
          }
        }
        for (int j = 0; j < RH; ++j) {
          const float w0 = w[RL_.w1 + z(j) * kRenderIn], w1 = w[RL_.w1 + z(j) * kRenderIn + 1];
          const float* g0 = g[0] + z(j) * Sb + z(x0);
          const float* g1 = g[1] + z(j) * Sb + z(x0);
          for (int v = 0; v < kV; ++v) {
            const vf a = load(g0 + v * kW);
            store(h1 + j * kB + v * kW, relu(a + wy * (load(g1 + v * kW) - a) + w0 * tv[v] + w1 * dv[v]));
          }
        }
        render_layers(w, h1);
        for (int v = 0; v < kV; ++v) {  // the material gate (render_gate)
          const vf m = 50.f * (vmax<vf>(tv[v], vf{}) + vmax<vf>(dv[v], vf{}));
          const vf gate = vmin<vf>(m, bc<vf>(1.f));
          for (int ch = 0; ch < 4; ++ch) store(out_.data() + ch * kB + v * kW, load(out_.data() + ch * kB + v * kW) * gate);
        }
        rgba_block(out_.data(), std::min(kB, S_ - x0), in, out_row + 4 * z(x0));
      }
    }
  }

  // Premultiplied RGBA floats of a block (planes r, g, b, a of 16) to RGBA8 for its first n pixels, with the optional
  // colour matrix: as write_pixels, on vectors, the four bytes of a pixel assembled in one 32-bit lane.
  static void rgba_block(const float* q, int n, const FrameInput& in, std::uint8_t* out) {
    for (int v = 0; v < kV && v * kW < n; ++v) {
      vf c[4];
      for (int ch = 0; ch < 4; ++ch) c[ch] = load(q + ch * kB + v * kW);
      if (in.apply_colour) {
        const auto& M = in.colour;
        const vf r = c[0], g = c[1], b = c[2];
        c[0] = M[0] * r + M[1] * g + M[2] * b;
        c[1] = M[3] * r + M[4] * g + M[5] * b;
        c[2] = M[6] * r + M[7] * g + M[8] * b;
      }
      vi px{};
      for (int ch = 0; ch < 4; ++ch) {
        const vf u = vmin<vf>(vmax<vf>(c[ch], vf{}), bc<vf>(1.f));
        px |= __builtin_convertvector(u * 255.f + 0.5f, vi) << (8 * ch);
      }
      if (n - v * kW >= kW) std::memcpy(out + 4 * v * kW, &px, sizeof px);
      else std::memcpy(out + 4 * v * kW, &px, 4 * static_cast<std::size_t>(n - v * kW));
    }
  }

  // The renderer's second layer and output layer on one block of 16 pixels (from r1_ into out_), with the shared dense
  // kernel inlined: called once per block, its calls cost about as much as its arithmetic.
  [[gnu::flatten]] void render_layers(const float* w, const float* h1) {
    const int RH = h_.render_hidden;
    dense(w + RL_.w2, w + RL_.b2, h1, kB, r2_.data(), kB, RH, RH, true);
    dense(w + RL_.wo, w + RL_.bo, r2_.data(), kB, out_.data(), kB, RH, 4, false);
  }

  std::size_t scratch_bytes() const override {
    std::size_t n = 0;
    for (const auto* v : {&w1_, &b1_, &w2_, &b2_, &cond_, &folded_cond_, &X_, &h1p_, &h1_, &h2_, &d_, &mid_, &next_, &coarse_, &flow_, &div_, &p_,
                          &tmp_, &wot_, &noise_, &dirsum_, &soot_, &fac_, &ft_, &fd_, &rec_, &td_, &fg_,
                          &cs_t_, &cs_d_, &wx_, &wy_, &swl_, &g1_, &frow_, &r1_, &r2_, &out_}) {
      n += v->size() * 4;
    }
    n += any_.size() + 4 * off_.size() + 4 * (ext_.size() + act_.size() + nm_.size() + cell_lo_.size() + cell_hi_.size() + qmin_.size() + qmax_.size());
    n += curl_.bytes() + swirl_.bytes() + fine_flicker_.bytes();
    for (const auto& f : flicker_) n += f.bytes();
    n += flow_rows_.bytes() + swirl_rows_.bytes() + fac_rows_.bytes() + render_rows_.bytes();
    for (const auto& r : noise_rows_) n += r.bytes();
    return n;
  }

  double macs_per_pixel() const override {
    const double coarse = static_cast<double>(N_) * (9.0 * I_ * H_ + 9.0 * H_ * H_ + static_cast<double>(H_) * O_);
    const double renderer = kRenderIn * h_.render_hidden + h_.render_hidden * h_.render_hidden + 4.0 * h_.render_hidden;
    return coarse / (static_cast<double>(S_) * S_) + renderer;
  }

  float time() const override { return time_; }
  std::span<const float> coarse() const override { return coarse_; }
  std::span<const float> fine_heat() const override { return ft_; }
  std::span<const float> fine_soot() const override { return fd_; }
  int size() const override { return S_; }
  std::span<const float> flow() const override { return flow_; }
  std::span<float> coarse_mut() override { return coarse_; }
  std::span<float> fine_heat_mut() override { return ft_; }
  std::span<float> fine_soot_mut() override { return fd_; }
  void skip_empty(bool on) override { skip_ = on; }
  void adopt(float seconds) override {
    time_ = seconds;
    since_start_ = m_.detail.swirl_ramp;
    std::fill(p_.begin(), p_.end(), 0.f);
    std::fill(flow_.begin(), flow_.end(), 0.f);
  }

 private:
  static std::size_t z(int v) { return static_cast<std::size_t>(v); }

  // Bilinear at cell-centre coordinates on an n x n grid of `channels` interleaved values, clamped.
  static float bilinear(const float* f, int n, int channels, int c, float x, float y) {
    x = std::clamp(x, 0.f, static_cast<float>(n - 1));
    y = std::clamp(y, 0.f, static_cast<float>(n - 1));
    const int x0 = std::min(static_cast<int>(x), n - 2), y0 = std::min(static_cast<int>(y), n - 2);
    const float fx = x - static_cast<float>(x0), fy = y - static_cast<float>(y0);
    const auto at = [&](int xx, int yy) { return f[(z(yy) * z(n) + z(xx)) * z(channels) + z(c)]; };
    return (1.f - fy) * ((1.f - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1.f - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
  }

  // New seed: the noise caches belong to a seed (the positions do not change).
  void reseed(std::uint64_t seed) {
    if (seeded_ && seed == seed_) return;
    seeded_ = true;
    seed_ = seed;
    const rollout::NoiseSpec& ns = m_.noise;
    const float k = 128.f / static_cast<float>(R_);
    curl_.init(z(N_), seed * 0x2545F4914F6CDD1DULL + 77);
    for (int o = 0; o < ns.flicker_octaves; ++o) flicker_[z(o)].init(z(N_), seed + static_cast<std::uint64_t>(o) * 0x9e3779b97f4a7c15ULL);
    for (int y = 0; y < R_; ++y) {
      for (int x = 0; x < R_; ++x) {
        const float X = (static_cast<float>(x) + 0.5f) * k + 0.5f, Y = (static_cast<float>(y) + 0.5f) * k + 0.5f;
        const std::size_t i = z(y) * z(R_) + z(x);
        curl_.set(i, X * ns.curl_scale, Y * ns.curl_scale);
        float sx = X * ns.flicker_freq, sy = Y * ns.flicker_freq;
        for (int o = 0; o < ns.flicker_octaves; ++o) {
          flicker_[z(o)].set(i, sx, sy);
          sx *= 2.f;
          sy *= 2.f;
        }
      }
    }
    fine_flicker_.init(ns.flicker_freq, ns.flicker_rate, ns.flicker_octaves, seed, 0.f, 130.f);
    swirl_.init(z(sw_n_) * z(sw_n_), seed * 0x9E3779B97F4A7C15ULL + 5);
    for (int j = 0; j < sw_n_; ++j) {
      for (int i = 0; i < sw_n_; ++i) swirl_.set(z(j) * z(sw_n_) + z(i), static_cast<float>(i - 1) * 0.5f, static_cast<float>(j - 1) * 0.5f);
    }
  }

  void coarse_noise(float t) {
    const rollout::NoiseSpec& ns = m_.noise;
    curl_.at(t * ns.curl_rate);
    float zt = t * ns.flicker_rate;
    for (int o = 0; o < ns.flicker_octaves; ++o) {
      flicker_[z(o)].at(zt);
      zt *= 2.f;
    }
    for (int i = 0; i < N_; ++i) {
      noise_[z(i) * 2] = curl_.value(z(i));
      float sum = 0.f, amp = 1.f, norm = 0.f;
      for (int o = 0; o < ns.flicker_octaves; ++o) {
        sum += amp * flicker_[z(o)].value(z(i));
        norm += amp;
        amp *= 0.5f;
      }
      noise_[z(i) * 2 + 1] = sum / norm;
    }
  }

  // FiLM folded into the convolution weights: gamma * (W x + b) + beta = (gamma W) x + (gamma b + beta).
  void fold() {
    if (std::equal(cond_.begin(), cond_.end(), folded_cond_.begin())) return;
    const float* w = m_.step_w.data();
    const auto one = [&](std::size_t W, std::size_t Bb, std::size_t G, std::size_t E, int ci, std::vector<float>& wo, std::vector<float>& bo) {
      for (int j = 0; j < H_; ++j) {
        float g = 1.f, e = 0.f;
        for (int k = 0; k < h_.cond(); ++k) {
          g += w[G + z(k) * z(H_) + z(j)] * cond_[z(k)];
          e += w[E + z(k) * z(H_) + z(j)] * cond_[z(k)];
        }
        // stored [tap][in][out]; the planar kernel wants [out][in][tap]
        for (int tap = 0; tap < 9; ++tap) {
          for (int c = 0; c < ci; ++c) wo[(z(j) * z(ci) + z(c)) * 9 + z(tap)] = g * w[W + (z(tap) * z(ci) + z(c)) * z(H_) + z(j)];
        }
        bo[z(j)] = g * w[Bb + z(j)] + e;
      }
    };
    one(L_.w1, L_.b1, L_.g1, L_.e1, I_, w1_, b1_);
    one(L_.w2, L_.b2, L_.g2, L_.e2, H_, w2_, b2_);
    std::copy(cond_.begin(), cond_.end(), folded_cond_.begin());
  }

  void coarse_step() {
    using namespace rollout;
    const float* w = m_.step_w.data();
    // inputs, planar with a zero border, for the shared 3x3 kernel: [input][R + 2][R + 2]
    const int Pw = R_ + 2;
    for (int y = 0; y < R_; ++y) {
      for (int x = 0; x < R_; ++x) {
        const std::size_t i = z(y) * z(R_) + z(x), q = z(y + 1) * z(Pw) + z(x + 1), plane = z(Pw) * z(Pw);
        for (int k = 0; k < C_; ++k) X_[z(k) * plane + q] = coarse_[i * z(C_) + z(k)] / (k < kPhys ? m_.scale[z(k)] : 1.f);
        X_[z(C_) * plane + q] = noise_[i * 2];
        X_[z(C_ + 1) * plane + q] = noise_[i * 2 + 1];
        X_[z(C_ + 2) * plane + q] = (static_cast<float>(x) + 0.5f) / static_cast<float>(R_) * 2.f - 1.f;
        X_[z(C_ + 3) * plane + q] = (static_cast<float>(y) + 0.5f) / static_cast<float>(R_) * 2.f - 1.f;
      }
    }
    conv(X_.data(), w1_.data(), b1_.data(), h1_.data(), I_, H_);
    for (int c = 0; c < H_; ++c) {  // into a zero-bordered buffer for the second convolution
      for (int y = 0; y < R_; ++y) {
        std::copy_n(h1_.data() + (z(c) * z(R_) + z(y)) * z(R_), R_, h1p_.data() + (z(c) * z(Pw) + z(y + 1)) * z(Pw) + 1);
      }
    }
    conv(h1p_.data(), w2_.data(), b2_.data(), h2_.data(), H_, H_);
    // the 1x1 output layer, planar: d[k][cell], blocks of 16 cells with the dense kernel, then the cells left over
    int i0 = 0;
    for (; i0 + kB <= N_; i0 += kB) dense(wot_.data(), w + L_.bo, h2_.data() + i0, N_, d_.data() + i0, N_, H_, O_, false);
    for (; i0 < N_; ++i0) {
      for (int k = 0; k < O_; ++k) {
        float s = w[L_.bo + z(k)];
        for (int j = 0; j < H_; ++j) s += wot_[z(k) * z(H_) + z(j)] * h2_[z(j) * z(N_) + z(i0)];
        d_[z(k) * z(N_) + z(i0)] = s;
      }
    }
    for (int i = 0; i < N_; ++i) {
      for (int k = 0; k < kPhys; ++k) mid_[z(i) * z(C_) + z(k)] = coarse_[z(i) * z(C_) + z(k)] + m_.scale[z(k)] * d_[z(k) * z(N_) + z(i)];
      for (int k = kPhys; k < C_; ++k) d_[z(k) * z(N_) + z(i)] += coarse_[z(i) * z(C_) + z(k)];
    }
    for (int k = kPhys; k < C_; ++k) {  // the memory channels, bounded by tanh
      float* q = d_.data() + z(k) * z(N_);
      each_block(N_, [&]<class V>(int i) { st<V>(q + i, vtanh<V>(ld<V>(q + i))); });
      for (int i = 0; i < N_; ++i) mid_[z(i) * z(C_) + z(k)] = q[i];
    }
    const auto U = [&](int x, int y, int c) { return mid_[(z(std::clamp(y, 0, R_ - 1)) * z(R_) + z(std::clamp(x, 0, R_ - 1))) * z(C_) + z(c)]; };
    for (int y = 0; y < R_; ++y) {
      for (int x = 0; x < R_; ++x) {
        div_[z(y) * z(R_) + z(x)] = -0.5f * (U(x + 1, y, 0) - U(x - 1, y, 0) + U(x, y + 1, 1) - U(x, y - 1, 1)) + m_.qscale * d_[z(C_) * z(N_) + z(y) * z(R_) + z(x)];
      }
    }
    const int Pr = R_ + 2;  // pressure rows, with the zero border
    for (int it = 0; it < h_.jacobi; ++it) {
      for (int y = 0; y < R_; ++y) {
        const float* __restrict row = p_.data() + z(y + 1) * z(Pr) + 1;
        float* __restrict t = tmp_.data() + z(y + 1) * z(Pr) + 1;
        const float* __restrict dv = div_.data() + z(y) * z(R_);
        for (int x = 0; x < R_; ++x) t[x] = 0.25f * (dv[x] + (row[x - 1] + row[x + 1] + row[x - Pr] + row[x + Pr]));
      }
      p_.swap(tmp_);
    }
    const auto P = [&](int x, int y) { return p_[z(y + 1) * z(Pr) + z(x + 1)]; };  // zero outside
    for (int y = 0; y < R_; ++y) {
      for (int x = 0; x < R_; ++x) {
        const std::size_t i = z(y) * z(R_) + z(x);
        mid_[i * z(C_)] -= 0.5f * (P(x + 1, y) - P(x - 1, y));
        mid_[i * z(C_) + 1] -= 0.5f * (P(x, y + 1) - P(x, y - 1));
        flow_[i * 2] = mid_[i * z(C_)];
        flow_[i * 2 + 1] = mid_[i * z(C_) + 1];
      }
    }
    for (int y = 0; y < R_; ++y) {
      for (int x = 0; x < R_; ++x) {
        const std::size_t i = z(y) * z(R_) + z(x);
        const float px = std::clamp(static_cast<float>(x) - mid_[i * z(C_)], 0.f, static_cast<float>(R_ - 1));
        const float py = std::clamp(static_cast<float>(y) - mid_[i * z(C_) + 1], 0.f, static_cast<float>(R_ - 1));
        const int x0 = std::min(static_cast<int>(px), R_ - 2), y0 = std::min(static_cast<int>(py), R_ - 2);
        const float fx = px - static_cast<float>(x0), fy = py - static_cast<float>(y0);
        const float* a = mid_.data() + (z(y0) * z(R_) + z(x0)) * z(C_);
        const float* b = a + C_;
        const float* c = a + z(R_) * z(C_);
        const float* d = c + C_;
        float* o = next_.data() + i * z(C_);
        for (int k = 0; k < C_; ++k) {
          float v = (1.f - fy) * ((1.f - fx) * a[k] + fx * b[k]) + fy * ((1.f - fx) * c[k] + fx * d[k]);
          if (k < kPhys) v = std::clamp(v, m_.lo[z(k)], m_.hi[z(k)]);
          o[k] = v;
        }
      }
    }
    coarse_.swap(next_);
  }

  // The stepper's 3x3 convolutions (with ReLU) on the coarse grid: the shared kernel, with six output channels per
  // register tile on AVX2 (twelve accumulators of its sixteen registers) and the shared tile elsewhere.
  void conv(const float* in, const float* W, const float* b, float* out, int ci, int co) const {
    constexpr int T = kW == 8 ? 6 : NFX_TILE;
    int o = 0;
    for (; o + T <= co; o += T) conv_tile<T>(in, W, b, out, ci, R_, R_, o, true);
    for (; o + 2 <= co; o += 2) conv_tile<2>(in, W, b, out, ci, R_, R_, o, true);
    for (; o < co; ++o) conv_tile<1>(in, W, b, out, ci, R_, R_, o, true);
  }

  // Bilinear weights from the coarse grid (or the swirl lattice) to fine pixel centres: they depend on x or y alone.
  static Axis axis(int S, int n, float scale, float offset) {  // coordinate = (p + 0.5) * scale + offset, clamped
    Axis a;
    a.i.resize(z(S));
    a.w.resize(z(S));
    for (int p = 0; p < S; ++p) {
      const float c = std::clamp((static_cast<float>(p) + 0.5f) * scale + offset, 0.f, static_cast<float>(n - 1));
      const int i0 = std::min(static_cast<int>(c), n - 2);
      a.i[z(p)] = i0;
      a.w[z(p)] = c - static_cast<float>(i0);
    }
    return a;
  }

  // Where a bilinear sample of a ring of padded, interleaved rows (heat and soot of a pixel side by side, rows of Pw
  // pixels, the frame starting at (1, 1), padded row p in slot p & mask) at frame coordinates (px, py) reads: the offset
  // of its lower left corner and its weights. Coordinates are clamped to [-1, size], so that a sample outside the frame
  // reads the zero border. Kept apart from the sampling, so that each loop is short and the loads of one block overlap
  // the arithmetic of others.
  template <class V>
  static void stencil(V px, V py, int Pw, int mask, float edge, std::int32_t* off, float* wx, float* wy) {
    px = vmin<V>(vmax<V>(px, bc<V>(-1.f)), bc<V>(edge)) + 1.f;  // padded coordinates in [0, size + 1]
    py = vmin<V>(vmax<V>(py, bc<V>(-1.f)), bc<V>(edge)) + 1.f;
    const IntOf<V> xi = to_int<V>(px), yi = to_int<V>(py);
    st<V>(wx, px - to_float<V>(xi));
    st<V>(wy, py - to_float<V>(yi));
    st_int<V>(off, ((yi & mask) * Pw + xi) * 2);
  }

  // Slot of padded row p in a ring; after writing slot 0, mirror() copies it above the last slot, so that the row above
  // any slot is the next one in memory.
  float* ring_row(std::vector<float>& ring, int p) { return ring.data() + z(p & (ring_rows_ - 1)) * z(S_ + 3) * 2; }
  void mirror(std::vector<float>& ring, int p) {
    if ((p & (ring_rows_ - 1)) == 0) std::copy_n(ring.data(), z(S_ + 3) * 2, ring.data() + z(ring_rows_) * z(S_ + 3) * 2);
  }
  void zero_row(std::vector<float>& ring, int p) {
    std::fill_n(ring_row(ring, p), z(S_ + 3) * 2, 0.f);
    mirror(ring, p);
  }

  // Bilinear samples of both fields at stencils from stencil() (rows `row` floats apart), with the range of the four
  // corners when Range is set.
  template <class V, bool Range>
  static void sample2(const float* src, int row, const std::int32_t* off, V fx, V fy, V& a, V& b, V* lo, V* hi) {
    V r0[4], r1[4];  // a and b at the two corners of the lower row, then of the upper row
    quads2<V>(src, off, row, r0, r1);
    const V a0 = r0[0] + fx * (r0[2] - r0[0]), a1 = r1[0] + fx * (r1[2] - r1[0]);
    const V b0 = r0[1] + fx * (r0[3] - r0[1]), b1 = r1[1] + fx * (r1[3] - r1[1]);
    a = a0 + fy * (a1 - a0);
    b = b0 + fy * (b1 - b0);
    if constexpr (Range) {
      lo[0] = vmin<V>(vmin<V>(r0[0], r0[2]), vmin<V>(r1[0], r1[2]));
      hi[0] = vmax<V>(vmax<V>(r0[0], r0[2]), vmax<V>(r1[0], r1[2]));
      lo[1] = vmin<V>(vmin<V>(r0[1], r0[3]), vmin<V>(r1[1], r1[3]));
      hi[1] = vmax<V>(vmax<V>(r0[1], r0[3]), vmax<V>(r1[1], r1[3]));
    }
  }

  // The detail layer: MacCormack advection of heat and soot together, zero outside the frame (clamped to the forward
  // step's stencil, as in the simulation), then the lock to the coarse state. It runs in four stages, row by row, each
  // a few rows behind the stage whose output it reads, so that output is still in cache:
  //   copy     the fields, interleaved, into a ring of zero-bordered rows, ahead of the forward samples by as many rows
  //            as they reach up;
  //   forward  the fine velocity, the forward samples (also into a ring of interleaved, zero-bordered rows, for the
  //            round trip) and the range of their stencils;
  //   back     the round trip and the correction, behind the forward samples by as many rows as it reaches down, and
  //            column sums of the result; when a row of coarse cells is complete, its block sums and lock factors;
  //   lock     scale and new material, behind by the rows of coarse cells that a fine row interpolates.
  void detail_step(std::span<const float> controls) {
    using namespace rollout;
    const DetailSpec& dt = m_.detail;
    const float t = time_ + 0.5f / m_.fps, px128 = static_cast<float>(S_) / 128.f;
    float amp = dt.swirl * px128;
    if (dt.swirl_control >= 0 && z(dt.swirl_control) < controls.size()) amp *= 0.3f + controls[z(dt.swirl_control)];
    if (dt.swirl_ramp > 0.f) amp *= std::min(1.f, since_start_ / dt.swirl_ramp);
    amp_ = amp;
    if (amp > 0.f) swirl_lattice(t * dt.swirl_rate);
    if (dt.contrast > 0.f) fine_flicker_.at(t);
    flow_rows_.reset();
    swirl_rows_.reset();
    fac_rows_.reset();
    for (auto& r : noise_rows_) r.reset();
    std::fill(cs_t_.begin(), cs_t_.end(), 0.f);
    std::fill(cs_d_.begin(), cs_d_.end(), 0.f);
    // How far the samples reach up and down: from the range of the vertical velocity (interpolation stays within the
    // range of what it interpolates, and the margin covers rounding).
    const float k = static_cast<float>(S_) / static_cast<float>(R_), edge = static_cast<float>(S_);
    float flo = 0.f, fhi = 0.f, slo = 0.f, shi = 0.f;
    for (int i = 0; i < N_; ++i) {
      flo = std::min(flo, flow_[z(i) * 2 + 1]);
      fhi = std::max(fhi, flow_[z(i) * 2 + 1]);
    }
    if (amp > 0.f) {
      for (std::size_t q = 1; q < swl_.size(); q += 2) {
        slo = std::min(slo, swl_[q]);
        shi = std::max(shi, swl_[q]);
      }
    }
    const int lead = static_cast<int>(std::min(edge, std::max(0.f, -(k * flo + amp * slo)))) + 3;
    const int lag = static_cast<int>(std::min(edge, std::max(0.f, k * fhi + amp * shi))) + 3;
    lead_ = lead;
    lag_ = lag;
    // Spans only pay where much is empty: a step uses them when the last step that measured its spans computed at most
    // 90% of the pixels, and measures again every 8 steps (both paths give the same bits).
    spans_ = skip_ && (computed_ <= 0.9 || dense_steps_ >= 8);
    dense_steps_ = spans_ ? 0 : dense_steps_ + 1;
    if (spans_) {  // how far the samples reach left and right, as lead and lag up and down
      float ulo = 0.f, uhi = 0.f, sulo = 0.f, suhi = 0.f;
      for (int i = 0; i < N_; ++i) {
        ulo = std::min(ulo, flow_[z(i) * 2]);
        uhi = std::max(uhi, flow_[z(i) * 2]);
      }
      if (amp > 0.f) {
        for (std::size_t q = 0; q < swl_.size(); q += 2) {
          sulo = std::min(sulo, swl_[q]);
          suhi = std::max(suhi, swl_[q]);
        }
      }
      // a pixel at x samples x - u: it can see material in [first, last] only if x is in [first + ulo - 1, last + uhi + 1]
      reach_lo_ = static_cast<int>(std::min(edge, std::max(0.f, -(k * ulo + amp * sulo)))) + 3;
      reach_hi_ = static_cast<int>(std::min(edge, std::max(0.f, k * uhi + amp * suhi))) + 3;
    }
    ring_ = std::min(S_, lag + 1);  // a row's record lives from its forward samples until its round trip, lag rows later
    // The rings of padded rows hold every row from the oldest a sample still reads to the newest written ahead of it,
    // and the zero rows below and above the frame.
    ring_rows_ = std::min(slots_, static_cast<int>(std::bit_ceil(z(lead + lag + 2))));
    zero_row(td_, 0);
    zero_row(fg_, 0);
    win_next_ = qmin_head_ = qmin_tail_ = qmax_head_ = qmax_tail_ = 0;
    const int kk = S_ / R_;
    int copied = 0, back = 0, locked = 0;
    for (int r = 0; r < S_; ++r) {
      for (; copied < std::min(S_, r + lead); ++copied) {
        copy_row(copied);
        if (copied + 1 == S_) {
          zero_row(td_, S_ + 1);
          zero_row(td_, S_ + 2);
        }
      }
      forward_row(r, forward_span(r, copied));
      const bool last = r + 1 == S_;
      if (last) {
        zero_row(fg_, S_ + 1);
        zero_row(fg_, S_ + 2);
      }
      for (; back < S_ && (last || back + lag <= r + 1); ++back) {
        back_row(back);
        if ((back + 1) % kk == 0) lock_factors(back / kk);
      }
      for (; locked < S_ && (back == S_ || back / kk >= ax_.i[z(locked)] + 2); ++locked) lock_row(locked);
    }
    if (spans_) {
      long n = 0;
      for (int y = 0; y < S_; ++y) n += act_[z(y) * 2 + 1] - act_[z(y) * 2];
      computed_ = static_cast<double>(n) / (static_cast<double>(S_) * static_cast<double>(S_));
    }
  }

  // The sub-grid swirl on its lattice: the curl of the stream function by central differences, (u, v) interleaved.
  void swirl_lattice(float zt) {
    swirl_.at(zt);
    const int n = sw_n_;
    std::fill(swl_.begin(), swl_.end(), 0.f);
    for (int j = 1; j < n - 1; ++j) {
      for (int i = 1; i < n - 1; ++i) {
        const std::size_t q = z(j) * z(n) + z(i);
        swl_[2 * q] = swirl_.value(q + z(n)) - swirl_.value(q - z(n));
        swl_[2 * q + 1] = swirl_.value(q - 1) - swirl_.value(q + 1);
      }
    }
  }

  // Row y of the fields, interleaved, into the ring of zero-bordered rows.
  void copy_row(int y) {
    const std::size_t S = z(S_);
    float* o = ring_row(td_, y + 1) + 2;
    const float* a = ft_.data() + z(y) * S;
    const float* b = fd_.data() + z(y) * S;
    each_block(S_, [&]<class V>(int x) { st_pairs<V>(o + 2 * x, ld<V>(a + x), ld<V>(b + x)); });
    mirror(td_, y + 1);
    if (spans_) {  // the first and last pixel of the row that is not +0 in either field (bit patterns: -0 and NaN count)
      // Blocks of 8 pixels are tested at once (an "or" of their bits, which vectorises), then the pixel in the block.
      const auto bits = [&](int x) { return std::bit_cast<std::uint32_t>(a[x]) | std::bit_cast<std::uint32_t>(b[x]); };
      const auto any8 = [&](int x0) {
        std::uint32_t m = 0;
        for (int x = x0; x < std::min(S_, x0 + 8); ++x) m |= bits(x);
        return m != 0;
      };
      int first = S_, last = -1;
      for (int x0 = 0; x0 < S_ && first == S_; x0 += 8) {
        if (!any8(x0)) continue;
        for (int x = x0;; ++x) {
          if (bits(x)) {
            first = x;
            break;
          }
        }
      }
      for (int x0 = (S_ - 1) / 8 * 8; x0 >= 0 && last < 0 && first < S_; x0 -= 8) {
        if (!any8(x0)) continue;
        for (int x = std::min(S_, x0 + 8) - 1;; --x) {
          if (bits(x)) {
            last = x;
            break;
          }
        }
      }
      ext_[z(y) * 2] = first;
      ext_[z(y) * 2 + 1] = last;
    }
  }

  // The pixels of row y whose forward samples can see material (copied rows within the samples' vertical reach,
  // widened by their horizontal reach): [a, b), or every pixel when nothing is skipped.
  // The window [y - lag, y + lead] moves down one row per call, so its least first pixel and greatest last pixel are
  // kept in two monotone queues of rows (each row enters and leaves once).
  std::array<int, 2> forward_span(int y, int copied) {
    if (!spans_) return {0, S_};
    for (const int top = std::min(copied - 1, y + lead_); win_next_ <= top; ++win_next_) {
      const int r = win_next_;
      while (qmin_tail_ > qmin_head_ && ext_[z(qmin_[z(qmin_tail_ - 1)]) * 2] >= ext_[z(r) * 2]) --qmin_tail_;
      qmin_[z(qmin_tail_++)] = r;
      while (qmax_tail_ > qmax_head_ && ext_[z(qmax_[z(qmax_tail_ - 1)]) * 2 + 1] <= ext_[z(r) * 2 + 1]) --qmax_tail_;
      qmax_[z(qmax_tail_++)] = r;
    }
    while (qmin_head_ < qmin_tail_ && qmin_[z(qmin_head_)] < y - lag_) ++qmin_head_;
    while (qmax_head_ < qmax_tail_ && qmax_[z(qmax_head_)] < y - lag_) ++qmax_head_;
    if (qmin_head_ == qmin_tail_) return {0, 0};
    const int first = ext_[z(qmin_[z(qmin_head_)]) * 2], last = ext_[z(qmax_[z(qmax_head_)]) * 2 + 1];
    if (last < first) return {0, 0};
    // samples of pixel x lie in [x - uhi, x - ulo] (+1 for the stencil): x + reach_lo_ reaches first, x - reach_hi_ last
    return {std::max(0, first - reach_lo_), std::min(S_, last + reach_hi_ + 1)};
  }

  // Row y of the fine velocity (the coarse flow and the swirl, interpolated), the forward samples and their range.
  void forward_row(int y, std::array<int, 2> span) {
    const std::size_t S = z(S_);
    const int Pw = S_ + 3;
    const float k = static_cast<float>(S_) / static_cast<float>(R_), edge = static_cast<float>(S_), yf = static_cast<float>(y);
    const auto f = flow_rows_.get(flow_.data(), R_, ax_, ax_.i[z(y)]);
    const float wy0 = ax_.w[z(y)];
    const float *u0 = f[0], *v0 = f[0] + S, *u1 = f[1], *v1 = f[1] + S;
    const bool swirl = amp_ > 0.f;
    const float amp = amp_;
    const float *su0 = u0, *sv0 = u0, *su1 = u0, *sv1 = u0;
    float sy = 0.f;
    if (swirl) {
      const auto s = swirl_rows_.get(swl_.data(), sw_n_, sx_, sx_.i[z(y)]);
      sy = sx_.w[z(y)];
      su0 = s[0];
      sv0 = s[0] + S;
      su1 = s[1];
      sv1 = s[1] + S;
    }
    float* rec = row_record(y);
    const float* td = td_.data();
    float* o = ring_row(fg_, y + 1) + 2;
    const int mask = ring_rows_ - 1;
    std::int32_t* off = off_.data();
    float *wx = wx_.data(), *wy = wy_.data();
    each_block_in(S_, span[0], span[1], [&]<class V>(int x) {  // the velocity and the stencils
      const V a = ld<V>(u0 + x), b = ld<V>(v0 + x);
      V u = (a + wy0 * (ld<V>(u1 + x) - a)) * k, v = (b + wy0 * (ld<V>(v1 + x) - b)) * k;
      if (swirl) {
        const V c = ld<V>(su0 + x), d = ld<V>(sv0 + x);
        u += amp * (c + sy * (ld<V>(su1 + x) - c));
        v += amp * (d + sy * (ld<V>(sv1 + x) - d));
      }
      float* r = in_record<V>(rec, x);
      st<V>(r, u);
      st<V>(r + kW, v);
      stencil<V>(bc<V>(static_cast<float>(x)) + lane_index<V>() - u, bc<V>(yf) - v, Pw, mask, edge, off + x, wx + x, wy + x);
    });
    const auto done = each_block_in(S_, span[0], span[1], [&]<class V>(int x) {  // the samples and their range
      V sa, sb, lo[2], hi[2];
      sample2<V, true>(td, 2 * Pw, off + x, ld<V>(wx + x), ld<V>(wy + x), sa, sb, lo, hi);
      st_pairs<V>(o + 2 * x, sa, sb);
      float* r = in_record<V>(rec, x);
      st<V>(r + 2 * kW, sa);
      st<V>(r + 3 * kW, sb);
      st<V>(r + 4 * kW, lo[0]);
      st<V>(r + 5 * kW, hi[0]);
      st<V>(r + 6 * kW, lo[1]);
      st<V>(r + 7 * kW, hi[1]);
    });
    // Outside, every sample and its range is +0 (its stencil reads only +0): stored as such, and the round trip of
    // this row skips the same pixels.
    std::fill(o, o + 2 * done[0], 0.f);
    std::fill(o + 2 * std::max(done[0], done[1]), o + 2 * S, 0.f);
    act_[z(y) * 2] = done[0];
    act_[z(y) * 2 + 1] = done[1];
    mirror(fg_, y + 1);
  }

  // Row y of the round trip and the correction, and its column sums.
  void back_row(int y) {
    const std::size_t i = z(y) * z(S_);
    const int Pw = S_ + 3;
    const float edge = static_cast<float>(S_), yf = static_cast<float>(y);
    const float* rec = row_record(y);
    const float* fg = fg_.data();
    const int mask = ring_rows_ - 1;
    float *ft = ft_.data() + i, *fd = fd_.data() + i, *cst = cs_t_.data(), *csd = cs_d_.data();
    std::int32_t* off = off_.data();
    float *wx = wx_.data(), *wy = wy_.data();
    const int xa = act_[z(y) * 2], xb = act_[z(y) * 2 + 1];
    each_block_in(S_, xa, xb, [&]<class V>(int x) {
      const float* r = in_record<V>(rec, x);
      stencil<V>(bc<V>(static_cast<float>(x)) + lane_index<V>() + ld<V>(r), bc<V>(yf) + ld<V>(r + kW), Pw, mask, edge, off + x, wx + x, wy + x);
    });
    // Outside, the forward samples' range is [+0, +0], so the corrected value is +0 whatever the round trip gives;
    // the column sums add nothing.
    std::fill(ft, ft + xa, 0.f);
    std::fill(fd, fd + xa, 0.f);
    std::fill(ft + std::max(xa, xb), ft + S_, 0.f);
    std::fill(fd + std::max(xa, xb), fd + S_, 0.f);
    each_block_in(S_, xa, xb, [&]<class V>(int x) {
      V a, b;
      sample2<V, false>(fg, 2 * Pw, off + x, ld<V>(wx + x), ld<V>(wy + x), a, b, nullptr, nullptr);
      const float* r = in_record<V>(rec, x);
      const V tn = vmax<V>(vmin<V>(vmax<V>(ld<V>(r + 2 * kW) + 0.5f * (ld<V>(ft + x) - a), ld<V>(r + 4 * kW)), ld<V>(r + 5 * kW)), V{});
      const V dn = vmax<V>(vmin<V>(vmax<V>(ld<V>(r + 3 * kW) + 0.5f * (ld<V>(fd + x) - b), ld<V>(r + 6 * kW)), ld<V>(r + 7 * kW)), V{});
      st<V>(ft + x, tn);
      st<V>(fd + x, dn);
      st<V>(cst + x, ld<V>(cst + x) + tn);
      st<V>(csd + x, ld<V>(csd + x) + dn);
    });
  }

  // The record of fine row y in the ring that carries it from the forward samples to the round trip: in blocks of kW
  // pixels, kRec planes of kW floats each (velocity u, v; forward samples of heat and soot; their ranges: heat low,
  // high, soot low, high), so that one pointer reaches all of a block.
  float* row_record(int y) { return rec_.data() + z(y % ring_) * z(record_stride()); }
  int record_stride() const { return (S_ + kW - 1) / kW * kW * kRec; }  // whole blocks, the last one maybe in part
  template <class V, class T>
  static T* in_record(T* row, int x) {
    if constexpr (kOne<V>) return row + (x / kW) * (kW * kRec) + x % kW;
    else return row + x * kRec;
  }

  // Row c of coarse cells is complete: its block sums (from the column sums, which start again) and the lock's factors
  // (as the reference): where the fine field holds more than the coarse cell, scale it down; where it holds less, scale
  // it up to `grow` and add the rest as new material.
  void lock_factors(int c) {
    const rollout::DetailSpec& dt = m_.detail;
    const int kk = S_ / R_;
    const float inv = 1.f / static_cast<float>(kk * kk);
    constexpr float eps = 1e-4f;
    any_[z(c)] = 0;
    nm_[z(c) * 2] = R_;  // the cells with new material: [first, last]
    nm_[z(c) * 2 + 1] = -1;
    for (int x = 0; x < R_; ++x) {
      float st = 0.f, sd = 0.f;
      for (int p = x * kk; p < (x + 1) * kk; ++p) {
        st += cs_t_[z(p)];
        sd += cs_d_[z(p)];
      }
      const std::size_t i = z(c) * z(R_) + z(x);
      const float bt = st * inv, tt = coarse_[i * z(C_) + 2];
      const float bd = sd * inv, td = coarse_[i * z(C_) + 3];
      const float rt = (tt + eps) / (bt + eps), rd = (td + eps) / (bd + eps);
      float* f = fac_.data() + i * 4;  // new heat, heat scale, new soot, soot scale
      f[1] = rt <= 1.f ? rt : std::min(rt, dt.grow);
      f[0] = std::max(0.f, tt - bt * f[1]);
      f[3] = rd <= 1.f ? rd : std::min(rd, dt.grow);
      f[2] = std::max(0.f, td - bd * f[3]);
      if (f[0] > 0.f || f[2] > 0.f) {
        any_[z(c)] = 1;
        nm_[z(c) * 2] = std::min(nm_[z(c) * 2], x);
        nm_[z(c) * 2 + 1] = std::max(nm_[z(c) * 2 + 1], x);
      }
    }
    std::fill(cs_t_.begin(), cs_t_.end(), 0.f);
    std::fill(cs_d_.begin(), cs_d_.end(), 0.f);
  }

  // Row y of the lock, heat and soot together: the factors interpolated from the coarse cells, new material broken up by
  // the flicker noise (tongues, not a smear). The noise is evaluated only on rows that receive new material.
  void lock_row(int y) {
    const rollout::DetailSpec& dt = m_.detail;
    const std::size_t S = z(S_);
    const int c = ax_.i[z(y)];
    const float wy = ax_.w[z(y)];
    const auto f = fac_rows_.get(fac_.data(), R_, ax_, c);
    const float *at0 = f[0], *rt0 = f[0] + S, *ad0 = f[0] + 2 * S, *rd0 = f[0] + 3 * S;
    const float *at1 = f[1], *rt1 = f[1] + S, *ad1 = f[1] + 2 * S, *rd1 = f[1] + 3 * S;
    const bool noisy = dt.contrast > 0.f && (any_[z(c)] || any_[z(c + 1)]);
    const auto& oct = fine_flicker_.octaves();
    const std::size_t no = oct.size();
    std::array<const float*, 8> n0{}, n1{};
    std::array<float, 8> ty{}, amp{};
    if (noisy) {
      for (std::size_t o = 0; o < no; ++o) {
        const Axis& a = nax_[o];
        const auto r = noise_rows_[o].get(oct[o].c.data(), oct[o].n, a, a.i[z(y)]);
        n0[o] = r[0];
        n1[o] = r[1];
        ty[o] = a.w[z(y)];
        amp[o] = oct[o].amp;
      }
    }
    // the contrast curve's argument (phi - edge0) / (edge1 - edge0) as one multiply-add of the octaves' weighted sum
    const float span = 1.f / (dt.edge1 - dt.edge0), scale = span / fine_flicker_.norm(), shift = -dt.edge0 * span;
    const float keep = 1.f - dt.contrast, ck = dt.contrast * dt.kappa;
    float* qt = ft_.data() + z(y) * S;
    float* qd = fd_.data() + z(y) * S;
    // Where the fields are +0 after the round trip and no new material is interpolated, the lock gives +0 * scale + 0:
    // +0 again. So it runs on the round trip's pixels and on those that read a coarse cell with new material.
    int xa = 0, xb = S_;
    if (spans_) {
      xa = act_[z(y) * 2];
      xb = act_[z(y) * 2 + 1];
      if (xa >= xb) xa = S_, xb = 0;
      for (const int cc : {c, c + 1}) {
        if (nm_[z(cc) * 2 + 1] >= nm_[z(cc) * 2]) {
          xa = std::min(xa, cell_lo_[z(nm_[z(cc) * 2])]);
          xb = std::max(xb, cell_hi_[z(nm_[z(cc) * 2 + 1])]);
        }
      }
    }
    each_block_in(S_, xa, xb, [&]<class V>(int x) {
      V at = ld<V>(at0 + x), ad = ld<V>(ad0 + x), rt = ld<V>(rt0 + x), rd = ld<V>(rd0 + x);
      at = at + wy * (ld<V>(at1 + x) - at);
      rt = rt + wy * (ld<V>(rt1 + x) - rt);
      ad = ad + wy * (ld<V>(ad1 + x) - ad);
      rd = rd + wy * (ld<V>(rd1 + x) - rd);
      if (noisy) {
        V sum{};
        for (std::size_t o = 0; o < no; ++o) {
          const V a = ld<V>(n0[o] + x);
          sum += amp[o] * (a + ty[o] * (ld<V>(n1[o] + x) - a));
        }
        V s = sum * scale + shift;
        s = vmin<V>(vmax<V>(s, V{}), bc<V>(1.f));
        const V g = keep + ck * (s * s * (3.f - 2.f * s));
        at *= g;
        ad *= g;
      }
      st<V>(qt + x, ld<V>(qt + x) * rt + at);
      st<V>(qd + x, ld<V>(qd + x) * rd + ad);
    });
  }

  const rollout::Model& m_;
  const rollout::Hyper& h_;
  int S_, R_ = 0, N_ = 0, C_ = 0, I_ = 0, H_ = 0, O_ = 0;
  rollout::StepLayout L_{};
  rollout::RenderLayout RL_{};
  std::vector<float> w1_, b1_, w2_, b2_, cond_, folded_cond_;
  std::vector<float> X_, h1p_, h1_, h2_, d_, mid_, next_, coarse_, flow_, div_, p_, tmp_, wot_, noise_, dirsum_, soot_;
  std::vector<float> ft_, fd_;  // the fine fields
  std::vector<float> td_, fg_;  // rings of padded rows: the fine fields and the forward samples, interleaved
  int slots_ = 1, ring_rows_ = 1;  // slots of the rings, and those in use in this frame (powers of two)
  static constexpr int kRec = 8;                            // planes of a row record (see row_record)
  std::vector<float> rec_;                                  // the ring of row records
  int ring_ = 1;                                            // rows in the ring in this frame
  std::vector<float> cs_t_, cs_d_;                          // column sums of the row of coarse cells being summed
  std::vector<std::int32_t> off_;                           // one row of sample stencils: offsets of the lower left
  std::vector<float> wx_, wy_;                              // corners, and the weights
  std::vector<float> fac_;                                  // the lock's factors per coarse cell: add, scale (heat, soot)
  std::vector<std::uint8_t> any_;                           // a row of coarse cells receives new material
  // Study H (H2): skipping what is +0. Per fine row, the first and last pixel that is not +0 before the step (ext_) and
  // the pixels the forward samples and the round trip computed (act_, [a, b)); per coarse row, the first and last cell
  // with new material (nm_); per coarse cell, the fine pixels whose lock reads it ([cell_lo_, cell_hi_)).
  std::vector<int> ext_, act_, nm_, cell_lo_, cell_hi_;
  std::vector<int> qmin_, qmax_;  // the rows of the forward samples' window, as monotone queues (see forward_span)
  int win_next_ = 0, qmin_head_ = 0, qmin_tail_ = 0, qmax_head_ = 0, qmax_tail_ = 0;
  bool skip_ = true, spans_ = false;  // skipping allowed; used in this step
  double computed_ = 0.0;            // the fraction of pixels the last step with spans computed
  int dense_steps_ = 0;              // steps since then
  int lead_ = 0, lag_ = 0, reach_lo_ = 0, reach_hi_ = 0;
  std::vector<float> swl_;                                  // the swirl lattice's velocity, (u, v) interleaved
  std::vector<float> r1_, r2_, out_;                        // the renderer's activations for one block
  Axis ax_, sx_;                     // fine pixel -> coarse cell, fine pixel -> swirl lattice
  std::vector<Axis> nax_;            // fine pixel -> flicker lattice, per octave (smooth5 weights)
  RowPair<2> flow_rows_, swirl_rows_;  // expanded rows of the flow and of the swirl
  RowPair<4> fac_rows_;                // expanded rows of the lock's factors
  std::vector<RowPair<1>> noise_rows_;  // expanded rows of the flicker lattices, per octave
  RowPair<> render_rows_;              // expanded rows of the renderer's first layer (its coarse part)
  std::vector<float> g1_;              // the renderer's first layer per coarse cell: bias and coarse part [cell][unit]
  std::vector<float> frow_;            // one row of the renderer's fine features, planar [feature][padded size]
  SliceNoise curl_, swirl_;
  LatticeFbm fine_flicker_;
  std::vector<SliceNoise> flicker_;
  float sw_spacing_ = 1.f;
  float amp_ = 0.f;  // the swirl's amplitude in this frame
  int sw_n_ = 0;
  std::uint64_t seed_ = 0;
  bool seeded_ = false;
  float time_ = 0.f, since_start_ = 0.f;
};
