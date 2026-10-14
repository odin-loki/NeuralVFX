// Interpolation and resampling of premultiplied RGBA8 pictures for the effect manager (interp.hpp).
#include "interp.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define NFX_MGR_X86 1
#endif

namespace nfx::mgr {

namespace {

std::size_t z(int v) { return static_cast<std::size_t>(v); }

std::atomic<bool> g_portable{false};

// Per column of the picture: the displacement grid's cell and weight along x; per row the same along the state's rows
// (from the bottom). The displacement of a picture row is blended from two grid rows, then along x.
struct GridAxis {
  int i = 0;
  float w = 0.f;
};
GridAxis grid_axis(float c, int res) {
  c = std::clamp(c, 0.f, static_cast<float>(res - 1));
  const int i = std::min(static_cast<int>(c), res - 2);
  return {i, c - static_cast<float>(i)};
}

// The displacement of picture row y in pixels (dx right, dy down) at every column.
void displacement_row(const Motion& m, int size, int y, const GridAxis* cols, float* dx, float* dy) {
  const float scale = static_cast<float>(m.res) / static_cast<float>(size);
  const GridAxis r = grid_axis((static_cast<float>(size - 1 - y) + 0.5f) * scale - 0.5f, m.res);
  const float* g0 = m.uv + z(r.i) * z(m.res) * 2;
  const float* g1 = g0 + z(m.res) * 2;
  for (int x = 0; x < size; ++x) {
    const GridAxis c = cols[x];
    const float* p0 = g0 + z(c.i) * 2;
    const float* p1 = g1 + z(c.i) * 2;
    const float u0 = p0[0] + c.w * (p0[2] - p0[0]), u1 = p1[0] + c.w * (p1[2] - p1[0]);
    const float v0 = p0[1] + c.w * (p0[3] - p0[1]), v1 = p1[1] + c.w * (p1[3] - p1[1]);
    dx[x] = (u0 + r.w * (u1 - u0)) * m.pixels_per_cell;
    dy[x] = -(v0 + r.w * (v1 - v0)) * m.pixels_per_cell;  // the state's v is up, the picture's rows go down
  }
}

// Bilinear sample of channel-interleaved RGBA8 at pixel coordinates (pixel i's centre at i), clamped to the picture.
inline void bilinear4(const std::uint8_t* p, int size, float x, float y, float* out) {
  const float hi = static_cast<float>(size - 1);
  x = std::clamp(x, 0.f, hi);
  y = std::clamp(y, 0.f, hi);
  const int x0 = std::min(static_cast<int>(x), size - 2), y0 = std::min(static_cast<int>(y), size - 2);
  const float fx = x - static_cast<float>(x0), fy = y - static_cast<float>(y0);
  const std::uint8_t* q = p + (z(y0) * z(size) + z(x0)) * 4;
  const std::uint8_t* r = q + z(size) * 4;
  for (int c = 0; c < 4; ++c) {
    const float c00 = q[c], c10 = q[4 + c], c01 = r[c], c11 = r[4 + c];
    const float top = c00 + fx * (c10 - c00), bot = c01 + fx * (c11 - c01);
    out[c] = top + fy * (bot - top);
  }
}

void motion_portable(const std::uint8_t* a, const std::uint8_t* b, int size, float w, const Motion& m, std::uint8_t* out, const GridAxis* cols,
                     float* dx, float* dy) {
  const float iw = 1.f - w;
  for (int y = 0; y < size; ++y) {
    displacement_row(m, size, y, cols, dx, dy);
    std::uint8_t* o = out + z(y) * z(size) * 4;
    const float yf = static_cast<float>(y);
    for (int x = 0; x < size; ++x) {
      const float xf = static_cast<float>(x);
      float va[4], vb[4];
      bilinear4(a, size, xf - w * dx[x], yf - w * dy[x], va);
      bilinear4(b, size, xf + iw * dx[x], yf + iw * dy[x], vb);
      for (int c = 0; c < 4; ++c) o[4 * x + c] = static_cast<std::uint8_t>(static_cast<int>(va[c] * iw + vb[c] * w + 0.5f));
    }
  }
}

#if defined(NFX_MGR_X86)
[[gnu::target("avx2,fma")]] inline __m256 channel(__m256i v, int shift) {
  return _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v, shift), _mm256_set1_epi32(0xff)));
}

// Eight bilinear samples of RGBA8 at (x, y) (clamped), as four planes of floats.
[[gnu::target("avx2,fma")]] inline void gather_bilinear(const std::uint8_t* p, int size, __m256 x, __m256 y, __m256 (&c)[4]) {
  const __m256 hi = _mm256_set1_ps(static_cast<float>(size - 1));
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_setzero_ps()), hi);
  y = _mm256_min_ps(_mm256_max_ps(y, _mm256_setzero_ps()), hi);
  const __m256i lim = _mm256_set1_epi32(size - 2);
  const __m256i x0 = _mm256_min_epi32(_mm256_cvttps_epi32(x), lim), y0 = _mm256_min_epi32(_mm256_cvttps_epi32(y), lim);
  const __m256 fx = _mm256_sub_ps(x, _mm256_cvtepi32_ps(x0)), fy = _mm256_sub_ps(y, _mm256_cvtepi32_ps(y0));
  const __m256i idx = _mm256_add_epi32(_mm256_mullo_epi32(y0, _mm256_set1_epi32(size)), x0);
  const int* base = reinterpret_cast<const int*>(p);
  const __m256i q00 = _mm256_i32gather_epi32(base, idx, 4);
  const __m256i q10 = _mm256_i32gather_epi32(base + 1, idx, 4);
  const __m256i q01 = _mm256_i32gather_epi32(base + size, idx, 4);
  const __m256i q11 = _mm256_i32gather_epi32(base + size + 1, idx, 4);
  for (int k = 0; k < 4; ++k) {
    const __m256 c00 = channel(q00, 8 * k), c10 = channel(q10, 8 * k), c01 = channel(q01, 8 * k), c11 = channel(q11, 8 * k);
    const __m256 top = _mm256_fmadd_ps(fx, _mm256_sub_ps(c10, c00), c00);
    const __m256 bot = _mm256_fmadd_ps(fx, _mm256_sub_ps(c11, c01), c01);
    c[k] = _mm256_fmadd_ps(fy, _mm256_sub_ps(bot, top), top);
  }
}

[[gnu::target("avx2,fma")]] void motion_avx2(const std::uint8_t* a, const std::uint8_t* b, int size, float w, const Motion& m, std::uint8_t* out,
                                              const GridAxis* cols, float* dx, float* dy) {
  const __m256 vw = _mm256_set1_ps(w), viw = _mm256_set1_ps(1.f - w), half = _mm256_set1_ps(0.5f);
  const __m256 lane = _mm256_setr_ps(0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f);
  for (int y = 0; y < size; ++y) {
    displacement_row(m, size, y, cols, dx, dy);
    const __m256 yf = _mm256_set1_ps(static_cast<float>(y));
    std::uint8_t* o = out + z(y) * z(size) * 4;
    for (int x = 0; x < size; x += 8) {
      const __m256 xf = _mm256_add_ps(_mm256_set1_ps(static_cast<float>(x)), lane);
      const __m256 ddx = _mm256_loadu_ps(dx + x), ddy = _mm256_loadu_ps(dy + x);
      __m256 ca[4], cb[4];
      gather_bilinear(a, size, _mm256_fnmadd_ps(vw, ddx, xf), _mm256_fnmadd_ps(vw, ddy, yf), ca);
      gather_bilinear(b, size, _mm256_fmadd_ps(viw, ddx, xf), _mm256_fmadd_ps(viw, ddy, yf), cb);
      __m256i px = _mm256_setzero_si256();
      for (int k = 0; k < 4; ++k) {
        const __m256 v = _mm256_add_ps(_mm256_fmadd_ps(ca[k], viw, _mm256_mul_ps(cb[k], vw)), half);
        px = _mm256_or_si256(px, _mm256_slli_epi32(_mm256_cvttps_epi32(v), 8 * k));
      }
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + 4 * z(x)), px);
    }
  }
}

bool cpu_avx2() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#endif

}  // namespace

void force_portable_kernels(bool on) { g_portable.store(on); }

bool using_avx2_kernels() {
#if defined(NFX_MGR_X86)
  static const bool has = cpu_avx2();
  return has && !g_portable.load();
#else
  return false;
#endif
}

void crossfade(const std::uint8_t* a, const std::uint8_t* b, int size, float w, std::uint8_t* out) {
  const int k = std::clamp(static_cast<int>(std::lround(w * 256.f)), 0, 256), ik = 256 - k;
  const std::size_t n = z(size) * z(size) * 4;
  for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<std::uint8_t>((a[i] * ik + b[i] * k + 128) >> 8);
}

void motion_blend(const std::uint8_t* a, const std::uint8_t* b, int size, float w, const Motion& m, std::uint8_t* out) {
  constexpr int kMax = 1024;
  std::array<GridAxis, kMax> cols;
  std::array<float, kMax> dx, dy;
  const int n = std::min(size, kMax);
  const float scale = static_cast<float>(m.res) / static_cast<float>(size);
  for (int x = 0; x < n; ++x) cols[z(x)] = grid_axis((static_cast<float>(x) + 0.5f) * scale - 0.5f, m.res);
#if defined(NFX_MGR_X86)
  if (using_avx2_kernels() && size % 8 == 0 && size <= kMax) {
    motion_avx2(a, b, size, w, m, out, cols.data(), dx.data(), dy.data());
    return;
  }
#endif
  motion_portable(a, b, n, w, m, out, cols.data(), dx.data(), dy.data());
}

void resample(const std::uint8_t* src, int from, std::uint8_t* dst, int to) {
  if (from == to) {
    std::memcpy(dst, src, z(from) * z(from) * 4);
    return;
  }
  if (from > to && from % to == 0) {
    const int f = from / to, n = f * f;
    for (int y = 0; y < to; ++y) {
      for (int x = 0; x < to; ++x) {
        int s[4] = {0, 0, 0, 0};
        for (int j = 0; j < f; ++j) {
          const std::uint8_t* q = src + (z(y * f + j) * z(from) + z(x * f)) * 4;
          for (int i = 0; i < f; ++i) {
            for (int c = 0; c < 4; ++c) s[c] += q[4 * i + c];
          }
        }
        std::uint8_t* o = dst + (z(y) * z(to) + z(x)) * 4;
        for (int c = 0; c < 4; ++c) o[c] = static_cast<std::uint8_t>((s[c] + n / 2) / n);
      }
    }
    return;
  }
  const float k = static_cast<float>(from) / static_cast<float>(to);
  for (int y = 0; y < to; ++y) {
    for (int x = 0; x < to; ++x) {
      float v[4];
      bilinear4(src, from, (static_cast<float>(x) + 0.5f) * k - 0.5f, (static_cast<float>(y) + 0.5f) * k - 0.5f, v);
      std::uint8_t* o = dst + (z(y) * z(to) + z(x)) * 4;
      for (int c = 0; c < 4; ++c) o[c] = static_cast<std::uint8_t>(static_cast<int>(v[c] + 0.5f));
    }
  }
}

}  // namespace nfx::mgr
