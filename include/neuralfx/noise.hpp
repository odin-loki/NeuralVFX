// Seeded 3D value noise and fractal sums. Deterministic across platforms (integer hashing, float maths only in the
// interpolation), header-only so the simulation and the runtime share it.
#pragma once

#include <cmath>
#include <cstdint>

namespace nfx {

inline std::uint32_t hash32(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

inline std::uint32_t hash_cell(std::int32_t x, std::int32_t y, std::int32_t z, std::uint64_t seed) {
  std::uint32_t h = hash32(static_cast<std::uint32_t>(seed) ^ hash32(static_cast<std::uint32_t>(seed >> 32) + 0x9e3779b9U));
  h = hash32(h ^ static_cast<std::uint32_t>(x) * 0x8da6b343U);
  h = hash32(h ^ static_cast<std::uint32_t>(y) * 0xd8163841U);
  h = hash32(h ^ static_cast<std::uint32_t>(z) * 0xcb1ab31fU);
  return h;
}

// Uniform in [-1, 1].
inline float cell_value(std::int32_t x, std::int32_t y, std::int32_t z, std::uint64_t seed) {
  return static_cast<float>(hash_cell(x, y, z, seed) >> 8) * (2.f / 16777215.f) - 1.f;
}

inline float smooth5(float t) { return t * t * t * (t * (t * 6.f - 15.f) + 10.f); }

// Value noise in [-1, 1], smooth (C2) in all three coordinates.
inline float value_noise(float x, float y, float z, std::uint64_t seed) {
  const float fx = std::floor(x), fy = std::floor(y), fz = std::floor(z);
  const auto ix = static_cast<std::int32_t>(fx), iy = static_cast<std::int32_t>(fy), iz = static_cast<std::int32_t>(fz);
  const float tx = smooth5(x - fx), ty = smooth5(y - fy), tz = smooth5(z - fz);
  float c[2][2][2];
  for (int k = 0; k < 2; ++k) {
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) c[k][j][i] = cell_value(ix + i, iy + j, iz + k, seed);
    }
  }
  const auto lerp = [](float a, float b, float t) { return a + t * (b - a); };
  const float a0 = lerp(lerp(c[0][0][0], c[0][0][1], tx), lerp(c[0][1][0], c[0][1][1], tx), ty);
  const float a1 = lerp(lerp(c[1][0][0], c[1][0][1], tx), lerp(c[1][1][0], c[1][1][1], tx), ty);
  return lerp(a0, a1, tz);
}

// Fractal sum of `octaves` value-noise layers, normalised to about [-1, 1].
inline float fbm(float x, float y, float z, std::uint64_t seed, int octaves) {
  float sum = 0.f, amp = 1.f, norm = 0.f;
  for (int o = 0; o < octaves; ++o) {
    sum += amp * value_noise(x, y, z, seed + static_cast<std::uint64_t>(o) * 0x9e3779b97f4a7c15ULL);
    norm += amp;
    amp *= 0.5f;
    x *= 2.f;
    y *= 2.f;
    z *= 2.f;
  }
  return sum / norm;
}

}  // namespace nfx
