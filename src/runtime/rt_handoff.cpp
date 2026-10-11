// The simulator's look for a rollout effect's first frames (rt_handoff.hpp). The drawing repeats sim::Fluid::render
// operation for operation (src/sim/sim.cpp), on the runtime's fine fields with the solver's zero border, so the bytes
// are the simulator's (tests: Handoff.SimLookIsTheSimulatorsRenderer). Baseline ISA, no allocation.
#include "rt_handoff.hpp"

#include <algorithm>
#include <cmath>

namespace nfx::rt {

namespace {

struct Rgb {
  float r, g, b;
};

float clamp01(float x) { return std::clamp(x, 0.f, 1.f); }
float fl(int v) { return static_cast<float>(v); }

Rgb fire_ramp(float t) {
  return {clamp01((t - 0.10f) * 2.2f), clamp01((t - 0.36f) * 1.8f) * 0.92f, clamp01((t - 0.85f) * 2.0f) * 0.85f};
}

std::uint8_t q8(float v) { return static_cast<std::uint8_t>(clamp01(v) * 255.f + 0.5f); }

// A field of n x n cells (rows from the bottom) seen as the solver's (n + 2) x (n + 2) grid with a zero border:
// sim::Field::sample.
class Bordered {
 public:
  Bordered(const float* q, int n) : q_(q), n_(n) {}
  float at(int x, int y) const {
    return x < 1 || y < 1 || x > n_ || y > n_ ? 0.f : q_[static_cast<std::size_t>(y - 1) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(x - 1)];
  }
  float sample(float x, float y) const {
    const float lim = fl(n_) + 0.5f;
    x = std::clamp(x, 0.5f, lim);
    y = std::clamp(y, 0.5f, lim);
    const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
    const float fx = x - fl(x0), fy = y - fl(y0);
    const float a = at(x0, y0) + fx * (at(x0 + 1, y0) - at(x0, y0));
    const float b = at(x0, y0 + 1) + fx * (at(x0 + 1, y0 + 1) - at(x0, y0 + 1));
    return a + fy * (b - a);
  }

 private:
  const float* q_;
  int n_;
};

}  // namespace

SimLook sim_look_for(std::string_view name) {
  if (name == "fire") return SimLook::fire;
  if (name == "smoke") return SimLook::smoke;
  if (name == "explosion") return SimLook::explosion;
  return SimLook::none;
}

void draw_sim_look(SimLook look, std::span<const float> heat, std::span<const float> soot, int size, std::uint8_t* rgba,
                   std::size_t stride, const float* colour) {
  const std::size_t S2 = static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
  if (look == SimLook::none || size < 1 || heat.size() < S2 || soot.size() < S2) return;
  const Bordered temp(heat.data(), size), dens(soot.data(), size);
  const int S = size;
  const float n = fl(size), sc = n / fl(S);
  constexpr float lx = -0.45f, ly = 0.89f;  // towards the light: up and to the left
  for (int py = 0; py < S; ++py) {
    std::uint8_t* row = rgba + stride * static_cast<std::size_t>(py);
    for (int px = 0; px < S; ++px) {
      const float gx = (fl(px) + 0.5f) * sc + 0.5f;
      const float gy = n - (fl(py) + 0.5f) * sc + 0.5f;
      const float T = temp.sample(gx, gy), D = dens.sample(gx, gy);
      Rgb c{};
      float a = 0.f;
      // Where there is no soot the exponentials give exactly 1 (no cover) and the light falls on nothing, so they are
      // skipped: the bytes are the same.
      if (look == SimLook::fire) {
        const Rgb e = fire_ramp(T);
        const float as = D == 0.f ? 0.f : 1.f - std::exp(-2.5f * D), keep = 1.f - 0.5f * as;
        c = {e.r * keep + 0.06f * as, e.g * keep + 0.05f * as, e.b * keep + 0.05f * as};
        a = clamp01(as + 0.25f * std::max(e.r, e.g));
      } else {
        float light = 1.f, as = 0.f;
        if (D != 0.f) {
          float tau = 0.f;  // optical depth towards the light
          for (int k = 1; k <= 12; ++k) tau += dens.sample(gx + lx * 2.f * fl(k) * sc, gy + ly * 2.f * fl(k) * sc);
          light = std::exp(-0.55f * tau);
          as = 1.f - std::exp(-3.f * D);
        }
        if (look == SimLook::smoke) {
          const float g = 0.80f * (0.32f + 0.68f * light);
          c = {g * as, 0.98f * g * as, 0.95f * g * as};
          a = as;
        } else {
          const Rgb e = fire_ramp(0.75f * T);
          const float g = 0.42f * (0.30f + 0.70f * light), keep = 1.f - 0.6f * as;
          c = {e.r * keep + g * as, e.g * keep + 0.95f * g * as, e.b * keep + 0.90f * g * as};
          a = clamp01(as + 0.2f * std::max(e.r, e.g));
        }
      }
      if (colour) {
        const float* M = colour;
        c = {M[0] * c.r + M[1] * c.g + M[2] * c.b, M[3] * c.r + M[4] * c.g + M[5] * c.b, M[6] * c.r + M[7] * c.g + M[8] * c.b};
      }
      std::uint8_t* o = row + 4 * static_cast<std::size_t>(px);
      o[0] = q8(c.r);
      o[1] = q8(c.g);
      o[2] = q8(c.b);
      o[3] = q8(a);  // rgb may exceed alpha: emission adds light
    }
  }
}

float handoff_weight(std::int64_t f, int frames, int fade) {
  if (f < frames) return 0.f;
  if (f >= static_cast<std::int64_t>(frames) + fade) return 1.f;
  return static_cast<float>(f - frames + 1) / static_cast<float>(fade + 1);
}

void blend_handoff(std::uint8_t* rgba, std::size_t stride, const std::uint8_t* sim, std::size_t sim_stride, int size, float w) {
  const std::size_t row = static_cast<std::size_t>(size) * 4;
  for (int y = 0; y < size; ++y) {
    std::uint8_t* d = rgba + stride * static_cast<std::size_t>(y);
    const std::uint8_t* s = sim + sim_stride * static_cast<std::size_t>(y);
    for (std::size_t i = 0; i < row; ++i) d[i] = static_cast<std::uint8_t>(static_cast<float>(s[i]) * (1.f - w) + static_cast<float>(d[i]) * w + 0.5f);
  }
}

}  // namespace nfx::rt
