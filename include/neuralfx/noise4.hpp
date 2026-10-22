// Seeded 4D value noise and the 3D forcing of study N (docs/STUDY_3D.md), shared by the 3D simulator (sim3d.hpp) and
// the 3D rollout model (rollout3d.hpp), so the model is driven by exactly the noise that forces the simulation.
//
// Positions are in units of a 128-unit width (X across, Y up, Z in depth; the box is [0, 128] x [0, 128 h / w] x
// [0, 128]), time in seconds. Deterministic across platforms (integer hashing; float maths only in the interpolation).
#pragma once

#include <neuralfx/noise.hpp>
#include <neuralfx/parallel.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace nfx {

inline std::uint32_t hash_cell4(std::int32_t x, std::int32_t y, std::int32_t z, std::int32_t w, std::uint64_t seed) {
  return hash32(hash_cell(x, y, z, seed) ^ static_cast<std::uint32_t>(w) * 0x9e3779b1U);
}

// Value noise in [-1, 1], smooth (C2) in all four coordinates.
inline float value_noise4(float x, float y, float z, float w, std::uint64_t seed) {
  const float fx = std::floor(x), fy = std::floor(y), fz = std::floor(z), fw = std::floor(w);
  const auto ix = static_cast<std::int32_t>(fx), iy = static_cast<std::int32_t>(fy), iz = static_cast<std::int32_t>(fz),
             iw = static_cast<std::int32_t>(fw);
  const float tx = smooth5(x - fx), ty = smooth5(y - fy), tz = smooth5(z - fz), tw = smooth5(w - fw);
  const auto lerp = [](float a, float b, float t) { return a + t * (b - a); };
  float sw[2];
  for (int l = 0; l < 2; ++l) {
    float sz[2];
    for (int k = 0; k < 2; ++k) {
      float sy[2];
      for (int j = 0; j < 2; ++j) {
        const auto v = [&](int i) {
          return static_cast<float>(hash_cell4(ix + i, iy + j, iz + k, iw + l, seed) >> 8) * (2.f / 16777215.f) - 1.f;
        };
        sy[j] = lerp(v(0), v(1), tx);
      }
      sz[k] = lerp(sy[0], sy[1], ty);
    }
    sw[l] = lerp(sz[0], sz[1], tz);
  }
  return lerp(sw[0], sw[1], tw);
}

// Fractal sum of `octaves` layers of 4D value noise, normalised to about [-1, 1].
inline float fbm4(float x, float y, float z, float w, std::uint64_t seed, int octaves) {
  float sum = 0.f, amp = 1.f, norm = 0.f;
  for (int o = 0; o < octaves; ++o) {
    sum += amp * value_noise4(x, y, z, w, seed + static_cast<std::uint64_t>(o) * 0x9e3779b97f4a7c15ULL);
    norm += amp;
    amp *= 0.5f;
    x *= 2.f;
    y *= 2.f;
    z *= 2.f;
  }
  return sum / norm;
}

// The forcing's constants (lengths in units of a 128-unit width).
struct Forcing3 {
  float curl_length = 24.f;  // the curl noise's vector potential varies on this length
  float curl_rate = 0.8f;    // and changes this fast (noise units per second)
  float flicker_freq = 0.10f, flicker_rate = 2.6f;
  int flicker_octaves = 3;
};

// The source flicker: 4D fractal noise at a position and time (the fire's burner and the detail layer's breakup).
inline float flicker3(const Forcing3& f, std::uint64_t seed, float X, float Y, float Z, float t) {
  return fbm4(X * f.flicker_freq, Y * f.flicker_freq, Z * f.flicker_freq, t * f.flicker_rate, seed, f.flicker_octaves);
}

// The curl noise: the curl of a vector potential A = (A0, A1, A2), each 4D value noise of (X, Y, Z) / curl_length and
// curl_rate t with a seed of its own. A is sampled on a lattice of spacing curl_length / 2, its curl taken by central
// differences there (in noise units, so a unit potential gives a curl of order one), and the curl interpolated
// trilinearly between nodes. Divergence-free up to the interpolation. Sized once (resize allocates); build() and
// sample() do not allocate.
class CurlField3 {
 public:
  void resize(const Forcing3& f, float width_units, float height_units, float depth_units) {
    length_ = f.curl_length;
    nx_ = static_cast<int>(std::ceil(2.f * width_units / length_)) + 4;
    ny_ = static_cast<int>(std::ceil(2.f * height_units / length_)) + 4;
    nz_ = static_cast<int>(std::ceil(2.f * depth_units / length_)) + 4;
    a_.assign(static_cast<std::size_t>(nx_) * ny_ * nz_ * 3, 0.f);
    c_.assign(a_.size(), 0.f);
  }
  bool sized() const { return nx_ > 0; }

  // The field at time t for a run's seed (the potential's seeds follow from it, as the 2D curl noise's does).
  void build(const Forcing3& f, std::uint64_t seed, float t, Pool* pool = nullptr) {
    const std::uint64_t base = seed * 0x2545F4914F6CDD1DULL + 77;
    const float tw = t * f.curl_rate;
    parallel_for(pool, ny_, [&](int j) {
      for (int k = 0; k < nz_; ++k) {
        for (int i = 0; i < nx_; ++i) {
          float* a = a_.data() + idx(i, j, k) * 3;
          const float qx = static_cast<float>(i - 1) * 0.5f, qy = static_cast<float>(j - 1) * 0.5f, qz = static_cast<float>(k - 1) * 0.5f;
          for (int c = 0; c < 3; ++c) a[c] = value_noise4(qx, qy, qz, tw, base + static_cast<std::uint64_t>(c) * 0x9E3779B97F4A7C15ULL);
        }
      }
    });
    parallel_for(pool, ny_, [&](int j) {
      for (int k = 0; k < nz_; ++k) {
        for (int i = 0; i < nx_; ++i) {
          float* c = c_.data() + idx(i, j, k) * 3;
          if (i == 0 || j == 0 || k == 0 || i == nx_ - 1 || j == ny_ - 1 || k == nz_ - 1) {
            c[0] = c[1] = c[2] = 0.f;
            continue;
          }
          const auto d = [&](int di, int dj, int dk, int comp) {
            return a_[idx(i + di, j + dj, k + dk) * 3 + static_cast<std::size_t>(comp)] - a_[idx(i - di, j - dj, k - dk) * 3 + static_cast<std::size_t>(comp)];
          };
          // curl A = (dA2/dy - dA1/dz, dA0/dz - dA2/dx, dA1/dx - dA0/dy); a node difference spans one noise unit
          c[0] = d(0, 1, 0, 2) - d(0, 0, 1, 1);
          c[1] = d(0, 0, 1, 0) - d(1, 0, 0, 2);
          c[2] = d(1, 0, 0, 1) - d(0, 1, 0, 0);
        }
      }
    });
  }

  // The curl at a position (128-unit coordinates).
  std::array<float, 3> sample(float X, float Y, float Z) const {
    int i, j, k;
    float fx, fy, fz;
    coord(X, 0, i, fx);
    coord(Y, 1, j, fy);
    coord(Z, 2, k, fz);
    return at(i, j, k, fx, fy, fz);
  }
  // sample() in two parts, so a caller on a regular grid can tabulate the first per axis: the lattice cell and the
  // fraction of a coordinate along an axis (0 x, 1 y, 2 z), then the interpolated curl.
  void coord(float X, int axis, int& i, float& f) const {
    const int n = axis == 0 ? nx_ : axis == 1 ? ny_ : nz_;
    const float l = std::clamp(2.f * X / length_ + 1.f, 0.f, static_cast<float>(n - 1) - 1e-3f);
    i = static_cast<int>(l);
    f = l - static_cast<float>(i);
  }
  std::array<float, 3> at(int i, int j, int k, float fx, float fy, float fz) const {
    std::array<float, 3> out{};
    const std::size_t sx = 3, sz = static_cast<std::size_t>(nx_) * 3, sy = static_cast<std::size_t>(nz_) * sz;
    const float* base = c_.data() + idx(i, j, k) * 3;
    for (int c = 0; c < 3; ++c) {
      const float* p = base + c;
      const float x00 = p[0] + fx * (p[sx] - p[0]), x10 = p[sy] + fx * (p[sy + sx] - p[sy]);
      const float x01 = p[sz] + fx * (p[sz + sx] - p[sz]), x11 = p[sy + sz] + fx * (p[sy + sz + sx] - p[sy + sz]);
      const float y0 = x00 + fy * (x10 - x00), y1 = x01 + fy * (x11 - x01);
      out[static_cast<std::size_t>(c)] = y0 + fz * (y1 - y0);
    }
    return out;
  }
  // The lattice's nodes along x, y and z (node (i, j, k) at 128-unit coordinates ((i, j, k) - 1) curl_length / 2), for a
  // caller that builds and samples the same lattice itself (the volume runtime's detail layer, src/runtime/rt_detail3.hpp).
  int nx() const { return nx_; }
  int ny() const { return ny_; }
  int nz() const { return nz_; }

 private:
  std::size_t idx(int i, int j, int k) const { return (static_cast<std::size_t>(j) * static_cast<std::size_t>(nz_) + static_cast<std::size_t>(k)) * static_cast<std::size_t>(nx_) + static_cast<std::size_t>(i); }
  float length_ = 24.f;
  int nx_ = 0, ny_ = 0, nz_ = 0;
  std::vector<float> a_, c_;
};

// The flicker (flicker3) at the centres of a regular grid of cells, `unit` 128-unit coordinates apart, for one time: the
// noise's lattice values are hashed once per frame for every octave (a few hundred thousand instead of 48 hashes per
// cell), then each cell is interpolated exactly as value_noise4 and fbm4 interpolate, so at(x, y, z) equals
// flicker3(f, seed, (x + 0.5) unit, (y + 0.5) unit, (z + 0.5) unit, t) to the bit (tested). Sized once (resize
// allocates); build() and at() do not allocate.
class FlickerLattice {
 public:
  void resize(const Forcing3& f, int nx, int ny, int nz, float unit) {
    octaves_ = f.flicker_octaves;
    oct_.assign(static_cast<std::size_t>(octaves_), {});
    for (int o = 0; o < octaves_; ++o) {
      Octave& q = oct_[static_cast<std::size_t>(o)];
      for (int a = 0; a < 3; ++a) {
        const int n = a == 0 ? nx : a == 1 ? ny : nz;
        auto& ax = q.axis[static_cast<std::size_t>(a)];
        ax.i.resize(static_cast<std::size_t>(n));
        ax.t.resize(static_cast<std::size_t>(n));
        int lo = 1 << 30, hi = -(1 << 30);
        for (int c = 0; c < n; ++c) {
          float v = (static_cast<float>(c) + 0.5f) * unit * f.flicker_freq;
          for (int k = 0; k < o; ++k) v *= 2.f;
          const float fv = std::floor(v);
          ax.i[static_cast<std::size_t>(c)] = static_cast<std::int32_t>(fv);
          ax.t[static_cast<std::size_t>(c)] = smooth5(v - fv);
          lo = std::min(lo, static_cast<int>(fv));
          hi = std::max(hi, static_cast<int>(fv) + 1);
        }
        ax.lo = lo;
        ax.n = hi - lo + 1;
        for (auto& i : ax.i) i -= lo;
      }
      q.v.assign(2 * static_cast<std::size_t>(q.axis[0].n) * static_cast<std::size_t>(q.axis[1].n) * static_cast<std::size_t>(q.axis[2].n), 0.f);
    }
  }
  void build(const Forcing3& f, std::uint64_t seed, float t, Pool* pool = nullptr) {
    const float w = t * f.flicker_rate, fw = std::floor(w);
    const auto iw = static_cast<std::int32_t>(fw);
    tw_ = smooth5(w - fw);
    for (int o = 0; o < octaves_; ++o) {
      Octave& q = oct_[static_cast<std::size_t>(o)];
      const std::uint64_t so = seed + static_cast<std::uint64_t>(o) * 0x9e3779b97f4a7c15ULL;
      const int nx = q.axis[0].n, ny = q.axis[1].n, nz = q.axis[2].n;
      parallel_for(pool, 2 * ny, [&](int ly) {
        const int l = ly / ny, y = ly % ny;
        for (int z = 0; z < nz; ++z) {
          float* row = q.v.data() + ((static_cast<std::size_t>(l) * static_cast<std::size_t>(ny) + static_cast<std::size_t>(y)) * static_cast<std::size_t>(nz) + static_cast<std::size_t>(z)) * static_cast<std::size_t>(nx);
          for (int x = 0; x < nx; ++x) {
            row[x] = static_cast<float>(hash_cell4(q.axis[0].lo + x, q.axis[1].lo + y, q.axis[2].lo + z, iw + l, so) >> 8) * (2.f / 16777215.f) - 1.f;
          }
        }
      });
    }
  }
  float at(int x, int y, int z) const {
    float sum = 0.f, amp = 1.f, norm = 0.f;
    const auto lerp = [](float a, float b, float t) { return a + t * (b - a); };
    for (const Octave& q : oct_) {
      const std::size_t ix = static_cast<std::size_t>(q.axis[0].i[static_cast<std::size_t>(x)]), iy = static_cast<std::size_t>(q.axis[1].i[static_cast<std::size_t>(y)]),
                        iz = static_cast<std::size_t>(q.axis[2].i[static_cast<std::size_t>(z)]);
      const float tx = q.axis[0].t[static_cast<std::size_t>(x)], ty = q.axis[1].t[static_cast<std::size_t>(y)], tz = q.axis[2].t[static_cast<std::size_t>(z)];
      const std::size_t nx = static_cast<std::size_t>(q.axis[0].n), ny = static_cast<std::size_t>(q.axis[1].n), nz = static_cast<std::size_t>(q.axis[2].n);
      float sw[2];
      for (std::size_t l = 0; l < 2; ++l) {
        float szv[2];
        for (std::size_t k = 0; k < 2; ++k) {
          float sy[2];
          for (std::size_t j = 0; j < 2; ++j) {
            const float* v = q.v.data() + ((l * ny + iy + j) * nz + iz + k) * nx + ix;
            sy[j] = lerp(v[0], v[1], tx);
          }
          szv[k] = lerp(sy[0], sy[1], ty);
        }
        sw[l] = lerp(szv[0], szv[1], tz);
      }
      sum += amp * lerp(sw[0], sw[1], tw_);
      norm += amp;
      amp *= 0.5f;
    }
    return sum / norm;
  }
  // The lattices' layout, for a caller that builds and interpolates them itself as build() and at() do (the volume
  // runtime's detail layer, src/runtime/rt_detail3.hpp): per octave and axis, the lattice cell of each grid cell (from
  // the lattice's first, lo) and its smooth5 weight, and the lattice's size n.
  struct AxisView {
    const std::int32_t* i = nullptr;
    const float* t = nullptr;
    int lo = 0, n = 0;
  };
  int octaves() const { return octaves_; }
  AxisView axis(int octave, int a) const {
    const auto& ax = oct_[static_cast<std::size_t>(octave)].axis[static_cast<std::size_t>(a)];
    return {ax.i.data(), ax.t.data(), ax.lo, ax.n};
  }

 private:
  struct Axis {
    std::vector<std::int32_t> i;
    std::vector<float> t;
    int lo = 0, n = 0;
  };
  struct Octave {
    std::array<Axis, 3> axis;
    std::vector<float> v;  // [w level][y][z][x]
  };
  int octaves_ = 0;
  float tw_ = 0.f;
  std::vector<Octave> oct_;
};

}  // namespace nfx
