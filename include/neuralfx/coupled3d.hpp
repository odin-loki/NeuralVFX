// The coupled demo's rule (docs/STUDY_R.md §6.1; `nvfx_3d video --kind coupled`): two learned volumes stacked through
// the volume C API, the smoke's box on the fire's. The fire's soot leaving the top of its box (the soot times the upward
// velocity over its top few layers, smoothed over about a second, against its running mean) sets the smoke's intensity;
// the two fields are drawn as one volume. A demo of coupling through the public API: the boxes exchange no heat or
// momentum (the solver's Stack, sim3d.hpp, does that for the simulation). Here so that the tool and the tests run the
// same arithmetic.
#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace nfx::coupled3d {

// The soot leaving the top of a volume: the sum over its top `layers` rows of soot times the upward velocity (0 where
// the gas sinks). Fields in the volume API's layout (texel (x, y, z) at (z * ny + y) * nx + x; the velocity three per
// texel, world units per second), summed in that order.
inline double top_soot_flux(std::span<const float> soot, std::span<const float> velocity, int nx, int ny, int nz, int layers) {
  const auto zu = [](int i) { return static_cast<std::size_t>(i); };
  if (nx < 1 || ny < 1 || nz < 1 || layers < 1 || layers > ny) throw std::invalid_argument("coupled3d: bad sizes");
  const std::size_t n = zu(nx) * zu(ny) * zu(nz);
  if (soot.size() < n || velocity.size() < 3 * n) throw std::invalid_argument("coupled3d: fields too small");
  double flux = 0;
  for (int z = 0; z < nz; ++z) {
    for (int y = ny - layers; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const std::size_t i = (zu(z) * zu(ny) + zu(y)) * zu(nx) + zu(x);
        flux += static_cast<double>(soot[i]) * std::max(0.f, velocity[3 * i + 1]);
      }
    }
  }
  return flux;
}

// The smoke's intensity from the fire's flux, frame by frame: the flux smoothed over about a second (a first-order
// filter with the frame rate's time constant) against its running mean since the start, times the base intensity,
// clamped to [0.05, 1]. For the first `settle` seconds the smoke plays at the base (the mean has nothing to compare
// with yet).
struct FluxDrive {
  double smoothed = 0, mean = 0;
  long frames = 0;
  float intensity = 0.f;
  float update(double flux, float base, double fps, double time, double settle = 1.0) {
    smoothed = frames == 0 ? flux : smoothed + (flux - smoothed) / fps;
    mean += (flux - mean) / static_cast<double>(frames + 1);
    ++frames;
    intensity = time >= settle ? static_cast<float>(std::clamp(static_cast<double>(base) * smoothed / std::max(1e-9, mean), 0.05, 1.0)) : base;
    return intensity;
  }
};

// One volume's field from the API's layout (z slowest) into the renderer's ((y * nz + z) * nx + x, render3d.hpp),
// written at `dst` (nx * ny * nz values): stacking two boxes in y is writing the lower at dst and the upper at
// dst + the lower's cells.
inline void rows_to_renderer(std::span<const float> src, int nx, int ny, int nz, float* dst) {
  const auto zu = [](int i) { return static_cast<std::size_t>(i); };
  if (src.size() < zu(nx) * zu(ny) * zu(nz)) throw std::invalid_argument("coupled3d: field too small");
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) dst[(zu(y) * zu(nz) + zu(z)) * zu(nx) + zu(x)] = src[(zu(z) * zu(ny) + zu(y)) * zu(nx) + zu(x)];
    }
  }
}

}  // namespace nfx::coupled3d
