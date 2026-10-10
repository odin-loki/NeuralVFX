// More field effects for composed scenes (compose.hpp, docs/COMPOSE.md §5): gusting wind, vortex rings, attractors,
// heat sources and extinguishers. apply() in compose.cpp calls field_flow() for the fields that push and apply_heat()
// for heat and cold; pull() moves material for attract. Nothing here allocates.
#include "compose.hpp"

#include <neuralfx/noise.hpp>

#include <algorithm>
#include <cmath>

namespace nfx::compose {

namespace {

// The vortex profile of apply(): `strength` at radius `radius`, zero at the centre, falling off outside; returned as
// the factor s of the flow (-s ey, s ex) around a centre at offset (ex, ey).
float swirl_factor(float ex, float ey, float radius, float strength) {
  const float r2 = ex * ex + ey * ey, rr = radius * radius, r = std::sqrt(r2);
  return strength * r / radius * std::exp(0.5f * (1.f - r2 / rr)) / (r + 1e-3f);
}

// How far attract moves material at distance r from its centre: strength at the radius, zero at the centre, falling
// off outside; never past the centre.
float pull_distance(float r, const ForceField& f, float weight) {
  const float R = std::max(f.radius, 1e-3f);
  const float d = weight * f.strength * (r / R) * std::exp(0.5f * (1.f - r * r / (R * R)));
  return std::min(d, r);
}

// Bilinear splat of `amount` at continuous grid coordinates (gx, gy) (cell centres at integers) into an n x n grid,
// clamped to the grid so nothing is lost.
void splat(float* grid, int n, float gx, float gy, float amount) {
  gx = std::clamp(gx, 0.f, static_cast<float>(n - 1));
  gy = std::clamp(gy, 0.f, static_cast<float>(n - 1));
  const int x0 = std::min(static_cast<int>(gx), n - 2 < 0 ? 0 : n - 2), y0 = std::min(static_cast<int>(gy), n - 2 < 0 ? 0 : n - 2);
  const int x1 = std::min(x0 + 1, n - 1), y1 = std::min(y0 + 1, n - 1);
  const float fx = gx - static_cast<float>(x0), fy = gy - static_cast<float>(y0);
  grid[zs(y0) * zs(n) + zs(x0)] += (1.f - fx) * (1.f - fy) * amount;
  grid[zs(y0) * zs(n) + zs(x1)] += fx * (1.f - fy) * amount;
  grid[zs(y1) * zs(n) + zs(x0)] += (1.f - fx) * fy * amount;
  grid[zs(y1) * zs(n) + zs(x1)] += fx * fy * amount;
}

}  // namespace

std::array<float, 2> field_flow(const ForceField& f, float wx, float wy, float weight) {
  switch (f.kind) {
    case ForceField::Kind::gust: {
      // the gusts' pattern travels with the wind (bus units: pixels per frame at 30 frames a second) and changes
      // `rate` times a second
      const float travel = 30.f * f.time, cell = std::max(f.soft, 1.f);
      const float nx = (wx - f.u * travel) / cell, ny = (wy - f.v * travel) / cell, nz = f.rate * f.time;
      const float along = fbm(nx, ny, nz, f.seed, 2), across = fbm(nx + 17.3f, ny - 9.1f, nz, f.seed + 1, 2);
      const float g = std::max(0.f, 1.f + 2.f * f.amount * along);  // fbm is about +-0.5: amount is the relative swing
      const float c = 2.f * f.amount * across / 3.f;
      return {weight * (f.u * g - f.v * c), weight * (f.v * g + f.u * c)};
    }
    case ForceField::Kind::ring: {
      float dx = f.u, dy = f.v;
      const float len = std::sqrt(dx * dx + dy * dy);
      if (len > 0.f) {
        dx /= len;
        dy /= len;
      } else {
        dx = 1.f;
        dy = 0.f;
      }
      const float px = -dy * f.radius, py = dx * f.radius;  // the cores: centre +- radius across the direction
      const float core = std::max(f.soft, 1.f);
      float du = 0.f, dv = 0.f;
      for (const float sign : {1.f, -1.f}) {
        const float ex = wx - (f.x + sign * px), ey = wy - (f.y + sign * py);
        const float s = sign * weight * swirl_factor(ex, ey, core, f.strength);
        du += -s * ey;
        dv += s * ex;
      }
      return {du, dv};
    }
    case ForceField::Kind::attract: {
      if (f.damping == 0.f) return {0.f, 0.f};
      const float ex = wx - f.x, ey = wy - f.y;
      const float s = weight * swirl_factor(ex, ey, std::max(f.radius, 1e-3f), f.damping);
      return {-s * ey, s * ex};
    }
    default: return {0.f, 0.f};
  }
}

void apply_heat(Module& m, const ForceField& f, float weight) {
  if (!m.active || weight <= 0.f) return;
  const int R = m.res(), S = m.size(), C = m.channels();
  const float k = fl(S) / fl(R), sc = m.at.scale, R2 = std::max(f.radius, 1e-3f) * std::max(f.radius, 1e-3f);
  const bool heat = f.kind == ForceField::Kind::heat;
  const auto falloff = [&](float wx, float wy) {
    const float ex = wx - f.x, ey = wy - f.y;
    return std::exp(-(ex * ex + ey * ey) / R2);
  };
  // what one point gets: heat added, or heat taken and a part of it turned into soot
  const auto change = [&](float& h, float& d, float g) {
    if (heat) {
      h += weight * f.strength * g;
    } else {
      const float take = std::clamp(weight * f.strength * g, 0.f, 1.f) * std::max(0.f, h);
      h -= take;
      d += f.damping * take;
    }
  };
  auto co = m.runner().coarse_mut();
  for (int cy = 0; cy < R; ++cy) {
    const float wy = m.at.y + (fl(S) - (fl(cy) + 0.5f) * k) * sc;
    for (int cx = 0; cx < R; ++cx) {
      const float wx = m.at.x + (fl(cx) + 0.5f) * k * sc;
      const float g = falloff(wx, wy);
      if (g < 1e-6f) continue;
      float* c = co.data() + (zs(cy) * zs(R) + zs(cx)) * zs(C);
      change(c[2], c[3], g);
    }
  }
  auto ft = m.runner().fine_heat_mut();
  auto fd = m.runner().fine_soot_mut();
  for (int py = 0; py < S; ++py) {
    const float wy = m.at.y + (fl(S - py) - 0.5f) * sc;
    for (int px = 0; px < S; ++px) {
      const float wx = m.at.x + (fl(px) + 0.5f) * sc;
      const float g = falloff(wx, wy);
      if (g < 1e-6f) continue;
      const std::size_t i = zs(py) * zs(S) + zs(px);
      change(ft[i], fd[i], g);
    }
  }
}

std::size_t pull_scratch(const Module& m) { return 2 * zs(m.res()) * zs(m.res()) + 2 * zs(m.size()) * zs(m.size()); }

void pull(Module& m, const ForceField& f, float weight, std::span<float> scratch) {
  if (!m.active || weight <= 0.f || f.strength == 0.f || scratch.size() < pull_scratch(m)) return;
  const int R = m.res(), S = m.size(), C = m.channels();
  const float k = fl(S) / fl(R), sc = m.at.scale;
  float* ch = scratch.data();
  float* cd = ch + zs(R) * zs(R);
  float* fh = cd + zs(R) * zs(R);
  float* fd = fh + zs(S) * zs(S);
  std::fill(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(pull_scratch(m)), 0.f);
  // where a point at world (wx, wy) goes: towards the centre by pull_distance()
  const auto moved = [&](float wx, float wy, float& ox, float& oy) {
    const float ex = wx - f.x, ey = wy - f.y, r = std::sqrt(ex * ex + ey * ey);
    if (r < 1e-4f) {
      ox = wx;
      oy = wy;
      return;
    }
    const float d = pull_distance(r, f, weight) / r;
    ox = wx - d * ex;
    oy = wy - d * ey;
  };
  auto co = m.runner().coarse_mut();
  for (int cy = 0; cy < R; ++cy) {
    const float wy = m.at.y + (fl(S) - (fl(cy) + 0.5f) * k) * sc;
    for (int cx = 0; cx < R; ++cx) {
      const float wx = m.at.x + (fl(cx) + 0.5f) * k * sc;
      const float* c = co.data() + (zs(cy) * zs(R) + zs(cx)) * zs(C);
      if (c[2] == 0.f && c[3] == 0.f) continue;
      float ox, oy;
      moved(wx, wy, ox, oy);
      const float gx = (ox - m.at.x) / (k * sc) - 0.5f, gy = (fl(S) - (oy - m.at.y) / sc) / k - 0.5f;
      splat(ch, R, gx, gy, c[2]);
      splat(cd, R, gx, gy, c[3]);
    }
  }
  for (int i = 0; i < R * R; ++i) {
    co[zs(i) * zs(C) + 2] = ch[i];
    co[zs(i) * zs(C) + 3] = cd[i];
  }
  auto ft = m.runner().fine_heat_mut();
  auto fs = m.runner().fine_soot_mut();
  for (int py = 0; py < S; ++py) {
    const float wy = m.at.y + (fl(S - py) - 0.5f) * sc;
    for (int px = 0; px < S; ++px) {
      const std::size_t i = zs(py) * zs(S) + zs(px);
      if (ft[i] == 0.f && fs[i] == 0.f) continue;
      const float wx = m.at.x + (fl(px) + 0.5f) * sc;
      float ox, oy;
      moved(wx, wy, ox, oy);
      const float gx = (ox - m.at.x) / sc - 0.5f, gy = fl(S) - (oy - m.at.y) / sc - 0.5f;
      splat(fh, S, gx, gy, ft[i]);
      splat(fd, S, gx, gy, fs[i]);
    }
  }
  std::copy(fh, fh + zs(S) * zs(S), ft.begin());
  std::copy(fd, fd + zs(S) * zs(S), fs.begin());
}

}  // namespace nfx::compose
