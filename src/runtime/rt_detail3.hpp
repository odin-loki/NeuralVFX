// The volume runtime's detail layer (study N's 3D rollout effects, docs/STUDY_3D.md; rollout3d::detail_step is the
// reference): the same values to the bit, on vectors of the native width, with the work that does not depend on the cell
// done once per frame. Included once per ISA (rt_detail3_<isa>.cpp) inside namespace NFX_NS, with NFX_VW floats per
// vector and NFX_ARCH the target. Those files are compiled without contraction (-ffp-contract=off, CMakeLists.txt):
// every multiply and add rounds on its own, as in the reference (baseline ISA, no FMA), so a lane computes exactly what
// the reference computes for its cell.
//
// What differs from the reference is bookkeeping only:
//   - trilinear samples from per-axis tables are separable: a coarse or lattice row is interpolated along x once per
//     frame (the reference's first lerp, the same operands), and a cell blends those rows along z and y (or y and z) as
//     the reference does;
//   - the fine fields are copied, heat and soot interleaved, into a zero-bordered box, so a MacCormack sample needs no
//     bounds checks: each lane fetches the two x-corners of both fields of one (y, z) row with one 16-byte load, four
//     rows per sample, transposed in registers (as the 2D runtime's detail layer, rt_rollout.hpp). Positions are
//     clamped to [-1, size] and floored as in trilinear_zero(), which gives the same corners and weights as the
//     reference's fast path wherever that one applies;
//   - the hashed noise lattices of the sub-grid swirl (CurlField3::build) and of the flicker (FlickerLattice::build) are
//     cached per seed and time level (they change every 10 to 20 frames). The swirl's value noise is cached as its two
//     time slices (value_noise4's last lerp is in time), its corners hashed once per lattice corner rather than once per
//     node and octant;
//   - the MacCormack's two fields share their stencils, and the block sums of the lock are taken in the reference's
//     order as the corrected rows are written.
// No allocation after construction. Threads (the step's Pool) split rows, slabs or lattice rows; every task writes its
// own cells, so the thread count never changes a value.
#if !defined(NFX_NS) || !defined(NFX_VW)
#error "define NFX_NS and NFX_VW before including rt_detail3.hpp"
#endif

#include <neuralfx/noise.hpp>
#include <neuralfx/noise4.hpp>
#include <neuralfx/parallel.hpp>
#include <neuralfx/rollout3d.hpp>

#include "rt_aligned.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(NFX_ARCH) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#define NFX_D3_PRAGMA(x) _Pragma(#x)
#define NFX_D3_TARGET(a) NFX_D3_PRAGMA(GCC target(a))
NFX_D3_TARGET(NFX_ARCH)
#define NFX_D3_PUSHED 1
#endif

namespace nfx::rt::NFX_NS {

namespace {

namespace r3 = rollout3d;

constexpr int kW = NFX_VW;  // floats per vector
typedef float vf __attribute__((vector_size(kW * 4)));
typedef float vfu __attribute__((vector_size(kW * 4), aligned(4)));
typedef std::int32_t vi __attribute__((vector_size(kW * 4)));
typedef float v4 __attribute__((vector_size(16)));
typedef float v4u __attribute__((vector_size(16), aligned(4)));

std::size_t zu(int v) { return static_cast<std::size_t>(v); }
float fl(int v) { return static_cast<float>(v); }

// --- lanes: V is vf (kW cells) or float (the cells left at the end of a row) -------------------------------------------

template <class V>
inline constexpr bool kOne = std::is_same_v<V, float>;
template <class V>
using IntOf = std::conditional_t<kOne<V>, std::int32_t, vi>;

template <class V>
inline V ld(const float* p) {
  if constexpr (kOne<V>) return *p;
  else return *reinterpret_cast<const vfu*>(p);
}
template <class V>
inline void st(float* p, V v) {
  if constexpr (kOne<V>) *p = v;
  else *reinterpret_cast<vfu*>(p) = v;
}
template <class V>
inline V bc(float x) {  // x in every lane (one broadcast)
  if constexpr (kOne<V>) return x;
  else return x - V{};
}
template <class V>
inline V lane_index() {  // 0, 1, 2, ...
  V v{};
  if constexpr (!kOne<V>) {
    for (int k = 0; k < kW; ++k) v[k] = static_cast<float>(k);
  }
  return v;
}
// std::min and std::max, lane by lane, with their ties and NaNs: min(a, b) is b only when b < a, max(a, b) is b only
// when a < b. std::clamp(v, lo, hi) is min(max(v, lo), hi) (libstdc++).
template <class V>
inline V smin(V a, V b) {
  return b < a ? b : a;
}
template <class V>
inline V smax(V a, V b) {
  return a < b ? b : a;
}
template <class V>
inline V sclamp(V v, V lo, V hi) {
  return smin<V>(smax<V>(v, lo), hi);
}
template <class V>
inline V vfloor(V x) {
  if constexpr (kOne<V>) {
    return std::floor(x);
  } else {
#if NFX_VW == 16
    return reinterpret_cast<vf>(_mm512_roundscale_ps(reinterpret_cast<__m512>(x), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
#elif NFX_VW == 8
    return reinterpret_cast<vf>(_mm256_round_ps(reinterpret_cast<__m256>(x), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
#else
#error "rt_detail3.hpp: NFX_VW must be 8 or 16"
#endif
  }
}
template <class V>
inline IntOf<V> to_int(V v) {  // truncation (of floored values: exact)
  if constexpr (kOne<V>) return static_cast<std::int32_t>(v);
  else return __builtin_convertvector(v, vi);
}
template <class V>
inline bool any(IntOf<V> m) {
  if constexpr (kOne<V>) {
    return m != 0;
  } else {
    bool a = false;
    for (int k = 0; k < kW; ++k) a |= m[k] != 0;
    return a;
  }
}
template <class V>
inline IntOf<V> mask(bool b) {  // a comparison's "true" (all bits) or "false" for scalar lanes
  if constexpr (kOne<V>) return b ? -1 : 0;
  else return b ? vi{} - 1 : vi{};
}
template <class V>
inline V pick(IntOf<V> m, V a, V b) {  // m ? a : b
  return m ? a : b;
}
template <class V>
inline IntOf<V> ipick(IntOf<V> m, IntOf<V> a, IntOf<V> b) {  // m ? a : b, on integer lanes
  return m ? a : b;
}
template <class V>
inline IntOf<V> lt(V a, V b) {
  if constexpr (kOne<V>) return a < b ? -1 : 0;
  else return a < b;
}
template <class V>
inline IntOf<V> le(V a, V b) {
  if constexpr (kOne<V>) return a <= b ? -1 : 0;
  else return a <= b;
}
template <class V>
inline IntOf<V> isnan_(V a) {
  if constexpr (kOne<V>) return a != a ? -1 : 0;
  else return a != a;
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
inline v4 load4(const float* p) { return *reinterpret_cast<const v4u*>(p); }
// Lanes 4 g to 4 g + 3 hold the four floats at p + o[j + 4 g].
inline vf quad_lanes(const float* p, const std::int32_t* o, int j) {
#if NFX_VW == 8
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
// Half H of the lanes of a and b interleaved (a0, b0, a1, b1, ...), and its inverse.
template <int H, std::size_t... L>
inline vf zip_(vf a, vf b, std::index_sequence<L...>) {
  return __builtin_shufflevector(a, b, (H * kW / 2 + static_cast<int>(L / 2) + (L % 2 ? kW : 0))...);
}
template <int Odd, std::size_t... L>
inline vf unzip_(vf a, vf b, std::index_sequence<L...>) {
  return __builtin_shufflevector(a, b, (2 * static_cast<int>(L) + Odd)...);
}
// p[2 j] = a[j], p[2 j + 1] = b[j], and back.
template <class V>
inline void st_pairs(float* p, V a, V b) {
  if constexpr (kOne<V>) {
    p[0] = a;
    p[1] = b;
  } else {
    *reinterpret_cast<vfu*>(p) = zip_<0>(a, b, std::make_index_sequence<kW>{});
    *reinterpret_cast<vfu*>(p + kW) = zip_<1>(a, b, std::make_index_sequence<kW>{});
  }
}
template <class V>
inline void ld_pairs(const float* p, V& a, V& b) {
  if constexpr (kOne<V>) {
    a = p[0];
    b = p[1];
  } else {
    const vf x = *reinterpret_cast<const vfu*>(p), y = *reinterpret_cast<const vfu*>(p + kW);
    a = unzip_<0>(x, y, std::make_index_sequence<kW>{});
    b = unzip_<1>(x, y, std::make_index_sequence<kW>{});
  }
}

// f.template operator()<vf>(x) for the whole vectors of a row of `width` cells, f.template operator()<float>(x) for the
// cells left over.
template <class F>
inline void each_vector(int width, F&& f) {
  int x = 0;
  for (; x + kW <= width; x += kW) f.template operator()<vf>(x);
  for (; x < width; ++x) f.template operator()<float>(x);
}

// out[x] = p[0] + w[x] (p[s] - p[0]) with p = row + idx[x] s, for x in [0, n): a grid row's lerp at the fine cells' x
// (idx, w: an axis table; s: the grid's stride along x), the first lerp of the reference's trilinear interpolations.
// Scalar: hardware gathers were no faster on one CPU (Sapphire Rapids) and slower on another (Cascade Lake).
inline void expand_row(const float* row, const std::int32_t* idx, const float* w, int s, float* out, int n) {
  for (int x = 0; x < n; ++x) {
    const float* p = row + static_cast<std::ptrdiff_t>(idx[x]) * s;
    out[x] = p[0] + w[x] * (p[s] - p[0]);
  }
}

// The eight corners of a trilinear sample of both fields of a zero-bordered box (heat and soot interleaved; x rows
// 2 floats per cell, z rows `rz` floats apart, y planes `ry`), for lanes whose lower corner is at float offset off[j]:
// t[c] and d[c] are the corners in trilinear_zero()'s order (c = x + 2 z + 4 y).
template <class V>
inline void corners(const float* p, const std::int32_t* off, std::size_t rz, std::size_t ry, V (&t)[8], V (&d)[8]) {
  const std::size_t row[4] = {0, rz, ry, ry + rz};
  for (int r = 0; r < 4; ++r) {
    const float* q = p + row[r];
    if constexpr (kOne<V>) {
      t[2 * r] = q[off[0]];
      d[2 * r] = q[off[0] + 1];
      t[2 * r + 1] = q[off[0] + 2];
      d[2 * r + 1] = q[off[0] + 3];
    } else {
      vf c[4];
      transpose4(quad_lanes(q, off, 0), quad_lanes(q, off, 1), quad_lanes(q, off, 2), quad_lanes(q, off, 3), c);
      t[2 * r] = c[0];
      d[2 * r] = c[1];
      t[2 * r + 1] = c[2];
      d[2 * r + 1] = c[3];
    }
  }
}

// trilinear_zero()'s interpolation of the corners v (x, then z, then y).
template <class V>
inline V trilerp(const V (&v)[8], V fx, V fy, V fz) {
  const V a0 = v[0] + fx * (v[1] - v[0]), a1 = v[2] + fx * (v[3] - v[2]), a2 = v[4] + fx * (v[5] - v[4]), a3 = v[6] + fx * (v[7] - v[6]);
  const V b0 = a0 + fz * (a1 - a0), b1 = a2 + fz * (a3 - a2);
  return b0 + fy * (b1 - b0);
}
// std::min({v...}) and std::max({v...}): the first least (greatest) of the eight, in order.
template <class V>
inline void range8(const V (&v)[8], V& lo, V& hi) {
  lo = v[0];
  hi = v[0];
  for (int c = 1; c < 8; ++c) {
    lo = smin<V>(lo, v[c]);
    hi = smax<V>(hi, v[c]);
  }
}

// A sample position (cells) of a box of n cells per axis: clamped to [-1, n] and floored, as trilinear_zero() does,
// into the box's index (2 cells of zero border below) and the weight. A NaN position reads only the border, as the
// reference's (its corners are all outside).
template <class V>
inline IntOf<V> axis_cell(V p, float n, V& w) {
  const V c = sclamp<V>(p, bc<V>(-1.f), bc<V>(n));
  const V f = vfloor<V>(c);
  w = c - f;
  const IntOf<V> i = to_int<V>(f) + 2;
  return ipick<V>(isnan_<V>(c), IntOf<V>{}, i);
}

// --- the kernel -------------------------------------------------------------------------------------------------------

class Detail3 final : public r3::DetailKernel {
 public:
  Detail3(const r3::Model& m, const r3::Workspace& ws) : h_(m.h) {
    const r3::Hyper& h = m.h;
    FX_ = h.fx();
    FY_ = h.fy();
    FZ_ = h.fz();
    PX_ = FX_ + 4;
    PY_ = FY_ + 4;
    PZ_ = FZ_ + 4;
    N_ = h.cells();
    const std::size_t box = zu(PX_) * zu(PY_) * zu(PZ_) * 2;
    box_.assign(box, 0.f);  // the fields, interleaved, zero-bordered (the border is never written)
    fwd_.assign(box, 0.f);  // the forward samples, likewise
    cells_.assign(zu(N_) * 2, 0.f);  // block sums
    flow_rows_.assign(zu(h.ry) * zu(h.rz) * 3 * zu(FX_), 0.f);
    lock_rows_.assign(zu(h.ry) * zu(h.rz) * 4 * zu(FX_), 0.f);
    // the swirl's lattice (CurlField3's nodes), its time slices' cache, the potential, the curl, its x-expanded rows
    snx_ = ws.swirl.nx();
    sny_ = ws.swirl.ny();
    snz_ = ws.swirl.nz();
    const std::size_t nodes = zu(snx_) * zu(sny_) * zu(snz_);
    for (int a = 0; a < 3; ++a) {
      const int n = a == 0 ? snx_ : a == 1 ? sny_ : snz_;
      corner_lo_[a] = static_cast<int>(std::floor(static_cast<float>(-1) * 0.5f));
      corner_n_[a] = static_cast<int>(std::floor(static_cast<float>(n - 2) * 0.5f)) + 2 - corner_lo_[a];
    }
    corners_.assign(3 * zu(corner_n_[0]) * zu(corner_n_[1]) * zu(corner_n_[2]), 0.f);
    for (auto& s : slices_) {
      s.v.assign(nodes * 3, 0.f);
      s.used = 0;
    }
    pot_.assign(nodes * 3, 0.f);
    curl_.assign(nodes * 3, 0.f);
    swirl_rows_.assign(zu(sny_) * zu(snz_) * 3 * zu(FX_), 0.f);
    // the flicker's lattices (FlickerLattice's), their levels' cache and x-expanded rows
    octaves_ = ws.flicker.octaves();
    std::size_t lat = 0, rows = 0;
    for (int o = 0; o < octaves_; ++o) {
      Oct& q = oct_[zu(o)];
      for (int a = 0; a < 3; ++a) q.ax[a] = ws.flicker.axis(o, a);
      q.at = lat;
      q.rows_at = rows;
      lat += zu(q.ax[0].n) * zu(q.ax[1].n) * zu(q.ax[2].n);
      rows += 2 * zu(q.ax[1].n) * zu(q.ax[2].n) * zu(FX_);
    }
    for (auto& l : levels_) {
      l.v.assign(lat, 0.f);
      l.used = 0;
    }
    flick_rows_.assign(rows, 0.f);
    // per-chunk scratch: one fine y's swirl rows blended along y, or flicker rows blended along y
    std::size_t blend = 0;
    for (int o = 0; o < octaves_; ++o) blend += 2 * zu(oct_[zu(o)].ax[2].n) * zu(FX_);
    chunk_floats_ = std::max(zu(snz_) * 3 * zu(FX_), blend);
    chunks_ = std::min(kChunks, FY_);
    chunk_.assign(zu(chunks_) * chunk_floats_, 0.f);
  }

  std::size_t bytes() const override {
    std::size_t n = box_.size() + fwd_.size() + cells_.size() + flow_rows_.size() + lock_rows_.size() + corners_.size() + pot_.size() + curl_.size() +
                    swirl_rows_.size() + flick_rows_.size() + chunk_.size();
    for (const auto& s : slices_) n += s.v.size();
    for (const auto& l : levels_) n += l.v.size();
    return 4 * n;
  }

  void step(const r3::Model& m, r3::State& s, std::uint64_t seed, std::span<const float> controls, r3::Workspace& ws, Pool* pool) override {
    (void)controls;
    const r3::Hyper& h = m.h;
    const r3::DetailSpec& dt = m.detail;
    const float t = s.time + 0.5f / m.fps;
    float amp = dt.swirl * fl(FX_) / 64.f;
    if (dt.swirl_ramp > 0.f) amp *= std::min(1.f, s.since_start / dt.swirl_ramp);
    const bool swirl = amp > 0.f, flicker = dt.contrast > 0.f;
    if (swirl) swirl_lattice(m, seed * 0x9E3779B97F4A7C15ULL + 5, t, ws, pool);
    if (flicker) flicker_lattice(m, seed, t, pool);
    // the coarse flow's rows along x, and the fields into the box
    expand_coarse(s.flow.data(), 3, ws, flow_rows_.data(), pool);
    parallel_for(pool, FY_, [&](int y) { copy_plane(s, y); });
    // the fine velocity and the forward samples (fine rows split into chunks: each has its own blended swirl rows)
    parallel_for(pool, chunks_, [&](int c) {
      for (int y = chunk_begin(c); y < chunk_begin(c + 1); ++y) forward_plane(m, ws, y, amp, swirl, chunk_.data() + zu(c) * chunk_floats_);
    });
    // the round trip, the correction and, slab by slab, the block sums and the lock's factors
    parallel_for(pool, h.ry, [&](int cy) { back_slab(m, s, ws, cy); });
    expand_coarse(ws.rr.data(), 2, ws, lock_rows_.data(), pool, ws.aa.data());
    // the lock, with the flicker's breakup of new material
    parallel_for(pool, chunks_, [&](int c) {
      for (int y = chunk_begin(c); y < chunk_begin(c + 1); ++y) lock_plane(m, s, ws, y, flicker, chunk_.data() + zu(c) * chunk_floats_);
    });
  }

 private:
  static constexpr int kChunks = 32;
  static constexpr int kSlots = 4;  // cached time levels (two per seed in use: crossfading shards use two seeds)
  int chunk_begin(int c) const { return static_cast<int>(static_cast<long>(FY_) * c / chunks_); }

  // --- the noise lattices -------------------------------------------------------------------------------------------

  struct Slot {
    AlignedFloats v;
    std::uint64_t seed = 0;
    std::int64_t level = 0;
    std::uint64_t used = 0;  // 0: empty, else the step that last used it
  };
  template <std::size_t K>
  Slot& slot_for(std::array<Slot, K>& slots, std::uint64_t seed, std::int64_t level, const Slot* keep, bool& fresh) {
    for (Slot& s : slots) {
      if (s.used && s.seed == seed && s.level == level) {
        s.used = clock_;
        fresh = false;
        return s;
      }
    }
    Slot* best = nullptr;
    for (Slot& s : slots) {
      if (&s == keep) continue;
      if (!best || s.used < best->used) best = &s;
    }
    best->seed = seed;
    best->level = level;
    best->used = clock_;
    fresh = true;
    return *best;
  }

  // The swirl (CurlField3::build for this seed and time, then sampled per fine cell): the potential's three
  // components are value_noise4 at the lattice nodes, the lerp in time of two slices; each slice is cached per seed and
  // time level, from corner values hashed once per corner.
  void swirl_lattice(const r3::Model& m, std::uint64_t sseed, float t, const r3::Workspace& ws, Pool* pool) {
    ++clock_;
    const std::uint64_t base = sseed * 0x2545F4914F6CDD1DULL + 77;
    const float w = t * m.detail.swirl_rate;  // CurlField3::build's tw (curl_rate = swirl_rate)
    const float fw = std::floor(w);
    const auto iw = static_cast<std::int32_t>(fw);
    const float tw = smooth5(w - fw);
    bool fresh = false;
    Slot& s0 = slot_for(slices_, base, iw, nullptr, fresh);
    if (fresh) slice(base, iw, s0.v.data(), pool);
    Slot& s1 = slot_for(slices_, base, static_cast<std::int64_t>(iw) + 1, &s0, fresh);
    if (fresh) slice(base, iw + 1, s1.v.data(), pool);
    const int nx = snx_, ny = sny_, nz = snz_;
    const float* a0 = s0.v.data();
    const float* a1 = s1.v.data();
    const auto idx = [&](int i, int j, int k) { return (zu(j) * zu(nz) + zu(k)) * zu(nx) + zu(i); };
    parallel_for(pool, ny, [&](int j) {
      for (std::size_t q = idx(0, j, 0) * 3; q < idx(0, j + 1, 0) * 3; ++q) pot_[q] = a0[q] + tw * (a1[q] - a0[q]);  // value_noise4's lerp in w
    });
    parallel_for(pool, ny, [&](int j) {
      for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
          float* c = curl_.data() + idx(i, j, k) * 3;
          if (i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1) {
            c[0] = c[1] = c[2] = 0.f;
            continue;
          }
          const auto d = [&](int di, int dj, int dk, int comp) {
            return pot_[idx(i + di, j + dj, k + dk) * 3 + zu(comp)] - pot_[idx(i - di, j - dj, k - dk) * 3 + zu(comp)];
          };
          c[0] = d(0, 1, 0, 2) - d(0, 0, 1, 1);
          c[1] = d(0, 0, 1, 0) - d(1, 0, 0, 2);
          c[2] = d(1, 0, 0, 1) - d(0, 1, 0, 0);
        }
      }
    });
    // rows of the curl along x at the fine cells' x (CurlField3::at's first lerp): [j][k][component][x]
    const r3::Workspace::Axis& ax = ws.swirl_axis[0];
    parallel_for(pool, ny, [&](int j) {
      for (int k = 0; k < nz; ++k) {
        float* o = swirl_rows_.data() + (zu(j) * zu(nz) + zu(k)) * 3 * zu(FX_);
        const float* row = curl_.data() + idx(0, j, k) * 3;
        for (int c = 0; c < 3; ++c) expand_row(row + c, ax.i.data(), ax.f.data(), 3, o + zu(c) * zu(FX_), FX_);
      }
    });
  }

  // value_noise4's slice at time level `level` (its x, y and z lerps, before the lerp in w) at every node, for the three
  // components (seeds base + c * golden ratio): [node][component].
  void slice(std::uint64_t base, std::int32_t level, float* out, Pool* pool) {
    const int cx = corner_n_[0], cy = corner_n_[1], cz = corner_n_[2];
    const std::size_t per = zu(cx) * zu(cy) * zu(cz);
    parallel_for(pool, 3 * cy, [&](int cj) {  // the corner values: [component][z][y][x]
      const int c = cj / cy, j = cj % cy;
      const std::uint64_t seed = base + static_cast<std::uint64_t>(c) * 0x9E3779B97F4A7C15ULL;
      for (int k = 0; k < cz; ++k) {
        float* row = corners_.data() + zu(c) * per + (zu(k) * zu(cy) + zu(j)) * zu(cx);
        for (int i = 0; i < cx; ++i) {
          row[i] = static_cast<float>(hash_cell4(corner_lo_[0] + i, corner_lo_[1] + j, corner_lo_[2] + k, level, seed) >> 8) * (2.f / 16777215.f) - 1.f;
        }
      }
    });
    const int nx = snx_, ny = sny_, nz = snz_;
    const auto lerp = [](float a, float b, float t) { return a + t * (b - a); };
    parallel_for(pool, ny, [&](int j) {
      const float qy = static_cast<float>(j - 1) * 0.5f, fy = std::floor(qy);
      const int iy = static_cast<int>(fy) - corner_lo_[1];
      const float ty = smooth5(qy - fy);
      for (int k = 0; k < nz; ++k) {
        const float qz = static_cast<float>(k - 1) * 0.5f, fz = std::floor(qz);
        const int iz = static_cast<int>(fz) - corner_lo_[2];
        const float tz = smooth5(qz - fz);
        for (int i = 0; i < nx; ++i) {
          const float qx = static_cast<float>(i - 1) * 0.5f, fx = std::floor(qx);
          const int ix = static_cast<int>(fx) - corner_lo_[0];
          const float tx = smooth5(qx - fx);
          for (int c = 0; c < 3; ++c) {
            const float* v = corners_.data() + zu(c) * per;
            float sz[2];
            for (int dk = 0; dk < 2; ++dk) {
              float sy[2];
              for (int dj = 0; dj < 2; ++dj) {
                const float* r = v + (zu(iz + dk) * zu(cy) + zu(iy + dj)) * zu(cx) + zu(ix);
                sy[dj] = lerp(r[0], r[1], tx);
              }
              sz[dk] = lerp(sy[0], sy[1], ty);
            }
            out[((zu(j) * zu(nz) + zu(k)) * zu(nx) + zu(i)) * 3 + zu(c)] = lerp(sz[0], sz[1], tz);
          }
        }
      }
    });
  }

  // The flicker's lattice values (FlickerLattice::build) for the time levels around t, cached per seed and level, and
  // their rows along x at the fine cells' x (FlickerLattice::at's first lerp): [octave][level][y][z][x].
  void flicker_lattice(const r3::Model& m, std::uint64_t seed, float t, Pool* pool) {
    ++clock_;
    const float w = t * m.noise.flicker_rate, fw = std::floor(w);
    const auto iw = static_cast<std::int32_t>(fw);
    ftw_ = smooth5(w - fw);
    const Slot* lv[2];
    for (int l = 0; l < 2; ++l) {
      bool fresh = false;
      Slot& s = slot_for(levels_, seed, static_cast<std::int64_t>(iw) + l, l ? lv[0] : nullptr, fresh);
      lv[l] = &s;
      if (!fresh) continue;
      for (int o = 0; o < octaves_; ++o) {
        const Oct& q = oct_[zu(o)];
        const std::uint64_t so = seed + static_cast<std::uint64_t>(o) * 0x9e3779b97f4a7c15ULL;
        const int nx = q.ax[0].n, ny = q.ax[1].n, nz = q.ax[2].n;
        float* v = s.v.data() + q.at;
        parallel_for(pool, ny, [&](int y) {
          for (int z = 0; z < nz; ++z) {
            float* row = v + (zu(y) * zu(nz) + zu(z)) * zu(nx);
            for (int x = 0; x < nx; ++x) {
              row[x] = static_cast<float>(hash_cell4(q.ax[0].lo + x, q.ax[1].lo + y, q.ax[2].lo + z, iw + l, so) >> 8) * (2.f / 16777215.f) - 1.f;
            }
          }
        });
      }
    }
    // rows along x, per octave, level, lattice y and z
    int total = 0;
    for (int o = 0; o < octaves_; ++o) total += 2 * oct_[zu(o)].ax[1].n;
    parallel_for(pool, total, [&](int r) {
      int o = 0;
      while (r >= 2 * oct_[zu(o)].ax[1].n) r -= 2 * oct_[zu(o++)].ax[1].n;
      const Oct& q = oct_[zu(o)];
      const int l = r / q.ax[1].n, y = r % q.ax[1].n, nx = q.ax[0].n, nz = q.ax[2].n;
      const float* v = lv[l]->v.data() + q.at;
      const auto& ax = q.ax[0];
      for (int z = 0; z < nz; ++z) {
        const float* row = v + (zu(y) * zu(nz) + zu(z)) * zu(nx);
        float* o2 = flick_rows_.data() + q.rows_at + ((zu(l) * zu(q.ax[1].n) + zu(y)) * zu(nz) + zu(z)) * zu(FX_);
        expand_row(row, ax.i, ax.t, 1, o2, FX_);
      }
    });
  }

  // --- the fields ---------------------------------------------------------------------------------------------------

  // Rows along x of a coarse field [cell][channels] at the fine cells' x (CoarseTap's first lerp), every channel of
  // `a` then of `b` (channels each): [y][z][channel][x].
  void expand_coarse(const float* a, int channels, const r3::Workspace& ws, float* out, Pool* pool, const float* b = nullptr) const {
    const r3::Hyper& h = h_;
    const r3::Workspace::Axis& ax = ws.coarse_axis[0];
    const int planes = b ? 2 * channels : channels;
    parallel_for(pool, h.ry, [&](int j) {
      for (int k = 0; k < h.rz; ++k) {
        float* o = out + (zu(j) * zu(h.rz) + zu(k)) * zu(planes) * zu(FX_);
        for (int pl = 0; pl < planes; ++pl) {
          const float* f = pl < channels ? a : b;
          const int c = pl % channels;
          const float* row = f + (zu(j) * zu(h.rz) + zu(k)) * zu(h.rx) * zu(channels) + zu(c);
          expand_row(row, ax.i.data(), ax.f.data(), channels, o + zu(pl) * zu(FX_), FX_);
        }
      }
    });
  }

  float* box_row(AlignedFloats& b, int y, int z) {
    return b.data() + ((zu(y + 2) * zu(PZ_) + zu(z + 2)) * zu(PX_) + 2) * 2;
  }

  void copy_plane(const r3::State& s, int y) {
    for (int z = 0; z < FZ_; ++z) {
      const std::size_t i = (zu(y) * zu(FZ_) + zu(z)) * zu(FX_);
      const float* a = s.fine_t.data() + i;
      const float* b = s.fine_d.data() + i;
      float* o = box_row(box_, y, z);
      each_vector(FX_, [&]<class V>(int x) { st_pairs<V>(o + 2 * x, ld<V>(a + x), ld<V>(b + x)); });
    }
  }

  // The box offsets of the lower corners of samples at (px, py, pz), and their weights.
  template <class V>
  void stencil(V px, V py, V pz, std::int32_t* off, V& wx, V& wy, V& wz) const {
    const IntOf<V> ix = axis_cell<V>(px, fl(FX_), wx), iy = axis_cell<V>(py, fl(FY_), wy), iz = axis_cell<V>(pz, fl(FZ_), wz);
    const IntOf<V> o = ((iy * PZ_ + iz) * PX_ + ix) * 2;
    if constexpr (kOne<V>) off[0] = o;
    else std::memcpy(off, &o, sizeof o);
  }

  // Fine plane y: the velocity (the coarse flow up-sampled, in fine cells per frame, plus the swirl), the forward samples
  // of both fields from the box and their corners' range. The velocity goes to ws.ufine ([cell row][component][x]), the
  // samples to the forward box, the ranges to ws.lo, ws.hi (heat) and ws.fa, ws.fb (soot).
  void forward_plane(const r3::Model& m, r3::Workspace& ws, int y, float amp, bool swirl, float* scratch) {
    const r3::Hyper& h = m.h;
    const float kf = fl(h.fine), yf = fl(y);
    const int j = ws.coarse_axis[1].i[zu(y)];
    const float fy = ws.coarse_axis[1].f[zu(y)];
    const std::size_t rowc = 3 * zu(FX_);
    if (swirl) {  // the swirl's rows blended along y for this plane (CurlField3::at's second lerp): [z lattice][component][x]
      const int sj = ws.swirl_axis[1].i[zu(y)];
      const float sfy = ws.swirl_axis[1].f[zu(y)];
      for (int k = 0; k < snz_; ++k) {
        const float* r0 = swirl_rows_.data() + (zu(sj) * zu(snz_) + zu(k)) * rowc;
        const float* r1 = swirl_rows_.data() + (zu(sj + 1) * zu(snz_) + zu(k)) * rowc;
        float* o = scratch + zu(k) * rowc;
        each_vector(3 * FX_, [&]<class V>(int x) {
          const V a = ld<V>(r0 + x);
          st<V>(o + x, a + sfy * (ld<V>(r1 + x) - a));
        });
      }
    }
    const std::size_t RZ = zu(PX_) * 2, RY = zu(PZ_) * RZ;
    for (int z = 0; z < FZ_; ++z) {
      const int k = ws.coarse_axis[2].i[zu(z)];
      const float fz = ws.coarse_axis[2].f[zu(z)], zf = fl(z);
      const float* e00 = flow_rows_.data() + (zu(j) * zu(h.rz) + zu(k)) * rowc;  // rows (j, k), (j, k + 1), (j + 1, k), (j + 1, k + 1)
      const float* e10 = e00 + rowc;
      const float* e01 = e00 + zu(h.rz) * rowc;
      const float* e11 = e01 + rowc;
      const float *s0 = nullptr, *s1 = nullptr;
      float sfz = 0.f;
      if (swirl) {
        const int sk = ws.swirl_axis[2].i[zu(z)];
        sfz = ws.swirl_axis[2].f[zu(z)];
        s0 = scratch + zu(sk) * rowc;
        s1 = s0 + rowc;
      }
      const std::size_t cell = (zu(y) * zu(FZ_) + zu(z)) * zu(FX_);
      float* u = ws.ufine.data() + cell * 3;
      float* fwd = box_row(fwd_, y, z);
      each_vector(FX_, [&]<class V>(int x) {
        V uc[3];
        for (int c = 0; c < 3; ++c) {
          const std::size_t q = zu(c) * zu(FX_) + zu(x);
          const V a00 = ld<V>(e00 + q), a10 = ld<V>(e10 + q), a01 = ld<V>(e01 + q), a11 = ld<V>(e11 + q);
          const V b0 = a00 + fz * (a10 - a00), b1 = a01 + fz * (a11 - a01);
          uc[c] = kf * (b0 + fy * (b1 - b0));
          if (swirl) {
            const V y0 = ld<V>(s0 + q), y1 = ld<V>(s1 + q);
            uc[c] += amp * (y0 + sfz * (y1 - y0));
          }
          st<V>(u + q, uc[c]);
        }
        alignas(64) std::int32_t off[kW];
        V wx, wy, wz;
        stencil<V>(bc<V>(fl(x)) + lane_index<V>() - uc[0], bc<V>(yf) - uc[1], bc<V>(zf) - uc[2], off, wx, wy, wz);
        V vt[8], vd[8];
        corners<V>(box_.data(), off, RZ, RY, vt, vd);
        V lot, hit, lod, hid;
        range8<V>(vt, lot, hit);
        range8<V>(vd, lod, hid);
        st_pairs<V>(fwd + 2 * x, trilerp<V>(vt, wx, wy, wz), trilerp<V>(vd, wx, wy, wz));
        st<V>(ws.lo.data() + cell + zu(x), lot);
        st<V>(ws.hi.data() + cell + zu(x), hit);
        st<V>(ws.fa.data() + cell + zu(x), lod);
        st<V>(ws.fb.data() + cell + zu(x), hid);
      });
    }
  }

  // Coarse slab cy: the round trip and the correction of its fine planes, then its blocks' sums (in the reference's
  // order: y, z, x) and the lock's factors (ws.rr, ws.aa).
  void back_slab(const r3::Model& m, r3::State& s, r3::Workspace& ws, int cy) {
    const r3::Hyper& h = m.h;
    const int k = h.fine;
    const std::size_t RZ = zu(PX_) * 2, RY = zu(PZ_) * RZ;
    const std::size_t plane = zu(h.rz) * zu(h.rx);
    float* bs = cells_.data() + zu(cy) * plane * 2;  // the slab's block sums: heat [z][x], then soot
    std::fill_n(bs, plane * 2, 0.f);
    const bool lanes = k == 4 && FX_ % (4 * kW) == 0;  // block sums on vectors, a block per lane
    for (int y = cy * k; y < (cy + 1) * k; ++y) {
      const float yf = fl(y);
      for (int z = 0; z < FZ_; ++z) {
        const float zf = fl(z);
        const std::size_t cell = (zu(y) * zu(FZ_) + zu(z)) * zu(FX_);
        const float* u = ws.ufine.data() + cell * 3;
        const float* fwd = box_row(fwd_, y, z);
        float* qt = s.fine_t.data() + cell;
        float* qd = s.fine_d.data() + cell;
        each_vector(FX_, [&]<class V>(int x) {
          const V lot = ld<V>(ws.lo.data() + cell + zu(x)), hit = ld<V>(ws.hi.data() + cell + zu(x));
          const V lod = ld<V>(ws.fa.data() + cell + zu(x)), hid = ld<V>(ws.fb.data() + cell + zu(x));
          const IntOf<V> et = le<V>(hit, V{}), ed = le<V>(hid, V{});  // all corners empty: the round trip is not needed
          V bt{}, bd{};
          if (any<V>(~et) || any<V>(~ed)) {
            alignas(64) std::int32_t off[kW];
            V wx, wy, wz;
            stencil<V>(bc<V>(fl(x)) + lane_index<V>() + ld<V>(u + x), bc<V>(yf) + ld<V>(u + zu(FX_) + zu(x)), bc<V>(zf) + ld<V>(u + 2 * zu(FX_) + zu(x)), off,
                       wx, wy, wz);
            V vt[8], vd[8];
            corners<V>(fwd_.data(), off, RZ, RY, vt, vd);
            bt = pick<V>(et, V{}, trilerp<V>(vt, wx, wy, wz));
            bd = pick<V>(ed, V{}, trilerp<V>(vd, wx, wy, wz));
          }
          V at, ad;
          ld_pairs<V>(fwd + 2 * x, at, ad);
          st<V>(qt + x, smax<V>(V{}, sclamp<V>(at + 0.5f * (ld<V>(qt + x) - bt), lot, hit)));
          st<V>(qd + x, smax<V>(V{}, sclamp<V>(ad + 0.5f * (ld<V>(qd + x) - bd), lod, hid)));
        });
        // each block's sum adds its cells in the reference's order: y, z, then x
        float* b = bs + zu(z / k) * zu(h.rx);
        if (lanes) {
          for (int x = 0; x < FX_; x += 4 * kW) {
            for (int ch = 0; ch < 2; ++ch) {
              const float* q = (ch ? qd : qt) + x;
              const vf q0 = ld<vf>(q), q1 = ld<vf>(q + kW), q2 = ld<vf>(q + 2 * kW), q3 = ld<vf>(q + 3 * kW);
              const vf e0 = unzip_<0>(q0, q1, std::make_index_sequence<kW>{}), o0 = unzip_<1>(q0, q1, std::make_index_sequence<kW>{});
              const vf e1 = unzip_<0>(q2, q3, std::make_index_sequence<kW>{}), o1 = unzip_<1>(q2, q3, std::make_index_sequence<kW>{});
              float* acc = b + zu(ch) * plane + zu(x / 4);
              vf a = ld<vf>(acc);
              a += unzip_<0>(e0, e1, std::make_index_sequence<kW>{});  // cells 4 j
              a += unzip_<0>(o0, o1, std::make_index_sequence<kW>{});  // 4 j + 1
              a += unzip_<1>(e0, e1, std::make_index_sequence<kW>{});  // 4 j + 2
              a += unzip_<1>(o0, o1, std::make_index_sequence<kW>{});  // 4 j + 3
              st<vf>(acc, a);
            }
          }
        } else {
          for (int x = 0; x < FX_; ++x) {
            b[zu(x / k)] += qt[x];
            b[plane + zu(x / k)] += qd[x];
          }
        }
      }
    }
    const float kf = fl(k), inv = 1.f / (kf * kf * kf);
    const int C = h.channels();
    for (int cz = 0; cz < h.rz; ++cz) {
      for (int cx = 0; cx < h.rx; ++cx) {
        const std::size_t c = (zu(cy) * zu(h.rz) + zu(cz)) * zu(h.rx) + zu(cx);
        for (int ch = 0; ch < 2; ++ch) {
          const float bb = bs[zu(ch) * plane + zu(cz) * zu(h.rx) + zu(cx)] * inv, target = s.coarse[c * zu(C) + 3 + zu(ch)];
          constexpr float eps = 1e-4f;
          const float r = (target + eps) / (bb + eps);
          const float rv = r <= 1.f ? r : std::min(r, m.detail.grow);
          ws.rr[2 * c + zu(ch)] = rv;
          ws.aa[2 * c + zu(ch)] = std::max(0.f, target - bb * rv);
        }
      }
    }
  }

  // Fine plane y of the lock: scale (heat and soot), plus new material broken up by the flicker where there is any.
  void lock_plane(const r3::Model& m, r3::State& s, const r3::Workspace& ws, int y, bool flicker, float* scratch) {
    const r3::Hyper& h = m.h;
    const r3::DetailSpec& dt = m.detail;
    const int j = ws.coarse_axis[1].i[zu(y)];
    const float fy = ws.coarse_axis[1].f[zu(y)];
    const std::size_t rowc = 4 * zu(FX_);
    const float keep = 1.f - dt.contrast, ck = dt.contrast * dt.kappa, e0 = dt.edge0, span = dt.edge1 - dt.edge0;
    bool blended = false;  // this plane's flicker rows blended along y (made when first needed)
    for (int z = 0; z < FZ_; ++z) {
      const int k = ws.coarse_axis[2].i[zu(z)];
      const float fz = ws.coarse_axis[2].f[zu(z)];
      const float* e00 = lock_rows_.data() + (zu(j) * zu(h.rz) + zu(k)) * rowc;
      const float* e10 = e00 + rowc;
      const float* e01 = e00 + zu(h.rz) * rowc;
      const float* e11 = e01 + rowc;
      const std::size_t cell = (zu(y) * zu(FZ_) + zu(z)) * zu(FX_);
      float* qt = s.fine_t.data() + cell;
      float* qd = s.fine_d.data() + cell;
      each_vector(FX_, [&]<class V>(int x) {
        V tap[4];  // heat's scale, soot's scale, new heat, new soot
        for (int c = 0; c < 4; ++c) {
          const std::size_t q = zu(c) * zu(FX_) + zu(x);
          const V a00 = ld<V>(e00 + q), a10 = ld<V>(e10 + q), a01 = ld<V>(e01 + q), a11 = ld<V>(e11 + q);
          const V b0 = a00 + fz * (a10 - a00), b1 = a01 + fz * (a11 - a01);
          tap[c] = b0 + fy * (b1 - b0);
        }
        V at = tap[2], ad = tap[3];
        const IntOf<V> nm = lt<V>(V{}, at) | lt<V>(V{}, ad);
        if (flicker && any<V>(nm)) {
          if (!blended) {
            blend_flicker(y, scratch);
            blended = true;
          }
          const V phi = flicker_at<V>(z, x, scratch);
          const V g = keep + ck * smoothstep<V>((phi - e0) / span);
          at = pick<V>(nm, at * g, at);
          ad = pick<V>(nm, ad * g, ad);
        }
        st<V>(qt + x, ld<V>(qt + x) * tap[0] + at);
        st<V>(qd + x, ld<V>(qd + x) * tap[1] + ad);
      });
    }
  }

  template <class V>
  static V smoothstep(V t) {  // smoothstep01
    t = sclamp<V>(t, V{}, bc<V>(1.f));
    return t * t * (3.f - 2.f * t);
  }

  // The flicker's rows along x, blended along y for fine plane y (FlickerLattice::at's second lerp): per octave
  // [level][z lattice][x].
  void blend_flicker(int y, float* out) const {
    for (int o = 0; o < octaves_; ++o) {
      const Oct& q = oct_[zu(o)];
      const int ny = q.ax[1].n, nz = q.ax[2].n, iy = q.ax[1].i[zu(y)];
      const float ty = q.ax[1].t[zu(y)];
      for (int l = 0; l < 2; ++l) {
        const float* r0 = flick_rows_.data() + q.rows_at + (zu(l) * zu(ny) + zu(iy)) * zu(nz) * zu(FX_);
        const float* r1 = r0 + zu(nz) * zu(FX_);
        float* d = out + zu(l) * zu(nz) * zu(FX_);
        each_vector(nz * FX_, [&]<class V>(int x) {
          const V a = ld<V>(r0 + x);
          st<V>(d + x, a + ty * (ld<V>(r1 + x) - a));
        });
      }
      out += 2 * zu(nz) * zu(FX_);
    }
  }

  // FlickerLattice::at(x, y, z) for the cells of a vector, from blend_flicker()'s rows: the lerps in z and w, the
  // octaves' weighted sum, its normalisation.
  template <class V>
  V flicker_at(int z, int x, const float* rows) const {
    V sum{};
    float amp = 1.f, norm = 0.f;
    for (int o = 0; o < octaves_; ++o) {
      const Oct& q = oct_[zu(o)];
      const int nz = q.ax[2].n, iz = q.ax[2].i[zu(z)];
      const float tz = q.ax[2].t[zu(z)];
      V sw[2];
      for (int l = 0; l < 2; ++l) {
        const float* r = rows + (zu(l) * zu(nz) + zu(iz)) * zu(FX_) + zu(x);
        const V a = ld<V>(r);
        sw[l] = a + tz * (ld<V>(r + FX_) - a);
      }
      sum += amp * (sw[0] + ftw_ * (sw[1] - sw[0]));
      norm += amp;
      amp *= 0.5f;
      rows += 2 * zu(nz) * zu(FX_);
    }
    return sum / norm;
  }

  struct Oct {
    FlickerLattice::AxisView ax[3];
    std::size_t at = 0, rows_at = 0;  // where its values start in a level, and its rows in flick_rows_
  };

  const r3::Hyper h_;
  int FX_ = 0, FY_ = 0, FZ_ = 0, PX_ = 0, PY_ = 0, PZ_ = 0, N_ = 0;
  AlignedFloats box_, fwd_, cells_, flow_rows_, lock_rows_;
  int snx_ = 0, sny_ = 0, snz_ = 0;
  int corner_lo_[3] = {}, corner_n_[3] = {};
  AlignedFloats corners_, pot_, curl_, swirl_rows_;
  std::array<Slot, kSlots> slices_;
  int octaves_ = 0;
  std::array<Oct, 8> oct_{};
  std::array<Slot, kSlots> levels_;
  AlignedFloats flick_rows_;
  float ftw_ = 0.f;
  std::uint64_t clock_ = 0;
  int chunks_ = 1;
  std::size_t chunk_floats_ = 0;
  AlignedFloats chunk_;
};

}  // namespace

std::unique_ptr<rollout3d::DetailKernel> make_detail3(const rollout3d::Model& m, const rollout3d::Workspace& ws) { return std::make_unique<Detail3>(m, ws); }

}  // namespace nfx::rt::NFX_NS

#if defined(NFX_D3_PUSHED)
#pragma GCC pop_options
#undef NFX_D3_PUSHED
#endif
