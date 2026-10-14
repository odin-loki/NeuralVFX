// Pictures between a rollout effect's updates (docs/MANAGER.md §3): the effect manager renders keyframes and draws the
// frames between them by temporal interpolation, and resamples pictures between its levels of detail. Premultiplied
// RGBA8, rows top to bottom, tightly packed (size * 4 bytes per row). Nothing here allocates.
#pragma once

#include <cstddef>
#include <cstdint>

namespace nfx::mgr {

// Plain crossfade: out = a + w (b - a), with w rounded to 1/256 and the sum rounded to the nearest level (integers, the
// same bits on every path).
void crossfade(const std::uint8_t* a, const std::uint8_t* b, int size, float w, std::uint8_t* out);

// The displacement field between two keyframes, as the effect's state holds its flow: res x res cells of (u, v), rows
// from the bottom (v up), in coarse cells. `pixels_per_cell` turns it into pixels of the picture (the picture's size /
// res, times the detail layer's advection gain).
struct Motion {
  const float* uv = nullptr;
  int res = 0;
  float pixels_per_cell = 0.f;
};

// Motion-compensated interpolation along the displacement d of the keyframes' interval: a pixel x of the frame at
// fraction w comes from a at x - w d(x) and from b at x + (1 - w) d(x) (bilinear samples, edges clamped), blended by w.
void motion_blend(const std::uint8_t* a, const std::uint8_t* b, int size, float w, const Motion& m, std::uint8_t* out);

// A picture of side `from` resampled to side `to`: block averages for whole factors down, bilinear at pixel centres
// otherwise (edges clamped), a copy at the same size.
void resample(const std::uint8_t* src, int from, std::uint8_t* dst, int to);

// Which kernels run: AVX2 where the CPU has it, else the portable ones (tests compare the two).
void force_portable_kernels(bool on);
bool using_avx2_kernels();

}  // namespace nfx::mgr
