// Trails: material left along a moving source's path in tiles that stay in the world (compose.hpp, docs/COMPOSE.md
// §5.3). Nothing here allocates after the Trail is made.
#include "compose.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nfx::compose {

namespace {

int ifloor(float v) { return static_cast<int>(std::floor(v)); }

// Copy the cells `src` shares with `dst` from src into dst: the coarse physical channels and the fine fields. `dir`
// says where src lies: 0 to dst's left, 1 to its right, 2 below it, 3 above it. Cells and fine rows count from the
// bottom (y up), as the runner keeps them.
void seed_band(Module& dst, const Module& src, int dir, int cells) {
  const int R = dst.res(), S = dst.size(), k = S / R, Cd = dst.channels(), Cs = src.channels();
  int x0 = 0, y0 = 0, w = R, h = R, ox = 0, oy = 0;  // dst's cells [x0, x0 + w) x [y0, y0 + h) are src's shifted by (ox, oy)
  switch (dir) {
    case 0:
      w = cells;
      ox = R - cells;
      break;
    case 1:
      x0 = R - cells;
      w = cells;
      ox = -(R - cells);
      break;
    case 2:
      h = cells;
      oy = R - cells;
      break;
    default:
      y0 = R - cells;
      h = cells;
      oy = -(R - cells);
      break;
  }
  auto D = dst.runner().coarse_mut();
  auto C = src.runner().coarse();
  for (int y = y0; y < y0 + h; ++y) {
    for (int x = x0; x < x0 + w; ++x) {
      float* d = D.data() + (zs(y) * zs(R) + zs(x)) * zs(Cd);
      const float* s = C.data() + (zs(y + oy) * zs(R) + zs(x + ox)) * zs(Cs);
      for (int c = 0; c < rollout::kPhys; ++c) d[c] = s[c];
    }
  }
  for (auto [F, G] : {std::pair{dst.runner().fine_heat_mut(), src.runner().fine_heat()}, std::pair{dst.runner().fine_soot_mut(), src.runner().fine_soot()}}) {
    for (int y = y0 * k; y < (y0 + h) * k; ++y) {
      const float* s = G.data() + zs(y + oy * k) * zs(S) + zs(ox * k);
      float* d = F.data() + zs(y) * zs(S);
      for (int x = x0 * k; x < (x0 + w) * k; ++x) d[x] = s[x];
    }
  }
}

}  // namespace

void deposit(Module& m, float x, float y, float radius, float heat, float soot, float u, float v, float pull) {
  if (!m.active) return;
  const int R = m.res(), S = m.size(), C = m.channels();
  const float k = fl(S) / fl(R), sc = m.at.scale, r = std::max(radius, 0.5f), inv_r2 = 1.f / (r * r), reach = 3.f * r;
  const float to_cells = 1.f / (k * sc);
  const float tu = u * to_cells, tv = -v * to_cells;  // cells per frame, y up
  heat = std::max(0.f, heat);
  soot = std::max(0.f, soot);
  pull = std::clamp(pull, 0.f, 1.f);
  // coarse cells: centre of (cx, cy) at world (at.x + (cx + 0.5) k sc, at.y + (S - (cy + 0.5) k) sc)
  const int cx0 = std::max(0, ifloor((x - reach - m.at.x) / (k * sc) - 0.5f)), cx1 = std::min(R - 1, ifloor((x + reach - m.at.x) / (k * sc) - 0.5f) + 1);
  const int cy0 = std::max(0, ifloor((fl(S) - (y + reach - m.at.y) / sc) / k - 0.5f)), cy1 = std::min(R - 1, ifloor((fl(S) - (y - reach - m.at.y) / sc) / k - 0.5f) + 1);
  if (cx0 > cx1 || cy0 > cy1) return;
  auto co = m.runner().coarse_mut();
  for (int cy = cy0; cy <= cy1; ++cy) {
    const float wy = m.at.y + (fl(S) - (fl(cy) + 0.5f) * k) * sc, ey = wy - y;
    for (int cx = cx0; cx <= cx1; ++cx) {
      const float wx = m.at.x + (fl(cx) + 0.5f) * k * sc, ex = wx - x;
      const float g = std::exp(-(ex * ex + ey * ey) * inv_r2);
      if (g < 1e-5f) continue;
      float* c = co.data() + (zs(cy) * zs(R) + zs(cx)) * zs(C);
      c[2] += heat * g;
      c[3] += soot * g;
      if (pull > 0.f) {
        c[0] += pull * g * (tu - c[0]);
        c[1] += pull * g * (tv - c[1]);
      }
    }
  }
  // fine pixels: centre of (px, py) (rows from the bottom) at world (at.x + (px + 0.5) sc, at.y + (S - py - 0.5) sc)
  const int px0 = std::max(0, ifloor((x - reach - m.at.x) / sc - 0.5f)), px1 = std::min(S - 1, ifloor((x + reach - m.at.x) / sc - 0.5f) + 1);
  const int py0 = std::max(0, ifloor(fl(S) - (y + reach - m.at.y) / sc - 0.5f)), py1 = std::min(S - 1, ifloor(fl(S) - (y - reach - m.at.y) / sc - 0.5f) + 1);
  auto ft = m.runner().fine_heat_mut();
  auto fd = m.runner().fine_soot_mut();
  for (int py = py0; py <= py1; ++py) {
    const float ey = m.at.y + (fl(S - py) - 0.5f) * sc - y;
    float* th = ft.data() + zs(py) * zs(S);
    float* ts = fd.data() + zs(py) * zs(S);
    for (int px = px0; px <= px1; ++px) {
      const float ex = m.at.x + (fl(px) + 0.5f) * sc - x;
      const float g = std::exp(-(ex * ex + ey * ey) * inv_r2);
      if (g < 1e-5f) continue;
      th[px] += heat * g;
      ts[px] += soot * g;
    }
  }
}

Trail::Trail(std::span<Module* const> tiles, const TrailSpec& spec, int group, std::uint64_t seed)
    : tiles_(tiles.begin(), tiles.end()), cells_(tiles.size()), spec_(spec), seed_(seed) {
  if (tiles_.empty() || tiles_.size() > 64) throw std::invalid_argument("a trail has 1 to 64 tiles");
  const Module& m0 = *tiles_[0];
  const int R = m0.res();
  for (const Module* m : tiles_)
    if (m->size() != m0.size() || m->res() != R) throw std::invalid_argument("a trail's tiles are of one effect and one size");
  if (spec_.band < 0 || 2 * spec_.band >= R) throw std::invalid_argument("a trail's band is less than half its effect's grid");
  if (!(spec_.width > 0.f)) throw std::invalid_argument("a trail's tiles have a width");
  stride_ = spec_.width * fl(R - spec_.band) / fl(R);
  const float scale = spec_.width / fl(m0.size());
  for (Module* m : tiles_) {
    m->active = false;
    m->group = group;
    m->at.scale = scale;
    m->band = {};
    m->feather = spec_.feather;
  }
}

int Trail::find(int i, int j) const {
  for (std::size_t q = 0; q < cells_.size(); ++q)
    if (cells_[q].on && cells_[q].i == i && cells_[q].j == j) return static_cast<int>(q);
  return -1;
}

int Trail::active() const {
  int n = 0;
  for (const Cell& c : cells_) n += c.on ? 1 : 0;
  return n;
}

void Trail::off(int k) {
  cells_[zs(k)].on = false;
  tiles_[zs(k)]->active = false;
}

void Trail::clear() {
  for (std::size_t q = 0; q < cells_.size(); ++q) off(static_cast<int>(q));
  bands();
  had_ = on_ = false;
  vx_ = vy_ = 0.f;
}

int Trail::take(int i, int j, float t, std::uint64_t need, bool recycle) {
  int k = -1;
  for (std::size_t q = 0; q < cells_.size() && k < 0; ++q)
    if (!cells_[q].on) k = static_cast<int>(q);
  if (k < 0 && !recycle) return -1;
  if (k < 0) {  // every tile plays: the one fed longest ago that this frame's path does not need
    float oldest = std::numeric_limits<float>::infinity();
    for (std::size_t q = 0; q < cells_.size(); ++q) {
      if ((need >> q) & 1u) continue;
      if (cells_[q].fed < oldest) {
        oldest = cells_[q].fed;
        k = static_cast<int>(q);
      }
    }
    if (k < 0) return -1;
    off(k);
    ++recycled_;
  }
  cells_[zs(k)] = {i, j, t, true};
  Module& m = *tiles_[zs(k)];
  m.at.x = spec_.x0 + fl(i) * stride_;
  m.at.y = spec_.y0 + fl(j) * stride_;
  m.start_empty(spec_.age, seed_ + 7919u * ++starts_);
  if (spec_.band > 0) {  // the cells it shares with tiles already playing: theirs, so the first blend keeps them
    static constexpr int di[4] = {-1, 1, 0, 0}, dj[4] = {0, 0, 1, -1};
    for (int dir = 0; dir < 4; ++dir) {
      const int q = find(i + di[dir], j + dj[dir]);
      if (q >= 0 && q != k) seed_band(m, *tiles_[zs(q)], dir, spec_.band);
    }
  }
  return k;
}

float Trail::edge(int k, int dir) const {
  const Module& m = *tiles_[zs(k)];
  const int R = m.res(), C = m.channels();
  auto co = m.runner().coarse();
  float most = 0.f;
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < R; ++b) {  // dir: 0 left, 1 right, 2 below, 3 above (cells y up)
      const int x = dir == 0 ? a : dir == 1 ? R - 1 - a : b, y = dir == 2 ? a : dir == 3 ? R - 1 - a : b;
      const float* c = co.data() + (zs(y) * zs(R) + zs(x)) * zs(C);
      most = std::max(most, std::max(c[2], c[3]));
    }
  }
  return most;
}

void Trail::bands() {
  const int B = spec_.band;
  for (std::size_t q = 0; q < cells_.size(); ++q) {
    if (!cells_[q].on) continue;
    const Cell& c = cells_[q];
    tiles_[q]->band = {find(c.i - 1, c.j) >= 0 ? B : 0, find(c.i + 1, c.j) >= 0 ? B : 0, find(c.i, c.j + 1) >= 0 ? B : 0, find(c.i, c.j - 1) >= 0 ? B : 0};
  }
}

void Trail::move(float x, float y, bool on, float radius, float t, float fps) {
  const float dx = x - x_, dy = y - y_;
  const bool path = on && had_ && std::sqrt(dx * dx + dy * dy) <= spec_.jump;
  if (path) {  // the source's velocity, smoothed
    const float a = 1.f - std::exp(-1.f / (0.15f * fps));
    vx_ += a * (dx * fps - vx_);
    vy_ += a * (dy * fps - vy_);
  } else {
    vx_ = vy_ = 0.f;
  }
  px_ = path ? x_ : x;
  py_ = path ? y_ : y;
  x_ = x;
  y_ = y;
  had_ = on_ = on;
  bool changed = false;
  const float life = spec_.hold + 5.f * spec_.fade;  // faded to under 1%: off
  for (std::size_t q = 0; q < cells_.size(); ++q) {
    if (cells_[q].on && t - cells_[q].fed > life) {
      off(static_cast<int>(q));
      changed = true;
    }
  }
  if (on) {
    // the tiles the footprint touches along the path (sampled every half stride), switched on where they are not
    const float reach = 2.f * std::max(radius, 0.5f), W = spec_.width, s = stride_;
    const float len = std::sqrt((x - px_) * (x - px_) + (y - py_) * (y - py_));
    const int n = std::max(1, static_cast<int>(std::ceil(len / (0.5f * s))));
    std::uint64_t need = 0;
    for (int p = 0; p <= n; ++p) {
      const float u = static_cast<float>(p) / static_cast<float>(n);
      const float sx = px_ + u * (x - px_), sy = py_ + u * (y - py_);
      // tile i covers [x0 + i s, x0 + i s + W): those that meet [sx - reach, sx + reach]
      const int i0 = ifloor((sx - reach - spec_.x0 - W) / s) + 1, i1 = ifloor((sx + reach - spec_.x0) / s);
      const int j0 = std::max(spec_.row_lo, ifloor((sy - reach - spec_.y0 - W) / s) + 1), j1 = std::min(spec_.row_hi, ifloor((sy + reach - spec_.y0) / s));
      for (int j = j0; j <= j1; ++j) {
        for (int i = i0; i <= i1; ++i) {
          int k = find(i, j);
          if (k < 0) {
            k = take(i, j, t, need);
            changed = true;
          }
          if (k < 0) continue;
          need |= std::uint64_t{1} << k;
          cells_[zs(k)].fed = t;
        }
      }
    }
  }
  if (spec_.spread > 0.f) {  // material at an outer edge: the neighbour there, if a tile is free
    static constexpr int di[4] = {-1, 1, 0, 0}, dj[4] = {0, 0, 1, -1};
    const int n = static_cast<int>(cells_.size());
    for (int q = 0; q < n; ++q) {
      if (!cells_[zs(q)].on) continue;
      for (int dir = 0; dir < 4; ++dir) {
        const int i = cells_[zs(q)].i + di[dir], j = cells_[zs(q)].j + dj[dir];
        if (j < spec_.row_lo || j > spec_.row_hi || find(i, j) >= 0 || edge(q, dir) <= spec_.spread) continue;
        if (take(i, j, cells_[zs(q)].fed, 0, false) < 0) break;
        changed = true;
      }
    }
  }
  if (changed) bands();
}

void Trail::couple(const Deposit& d, float t, float fps) {
  const int B = spec_.band;
  if (B > 0) {  // shared bands, as a domain's tiles share theirs
    for (std::size_t q = 0; q < cells_.size(); ++q) {
      if (!cells_[q].on) continue;
      const Cell& c = cells_[q];
      if (const int r = find(c.i + 1, c.j); r >= 0) blend_band(*tiles_[q], *tiles_[zs(r)], Side::right, B);
      if (const int a = find(c.i, c.j - 1); a >= 0) blend_band(*tiles_[q], *tiles_[zs(a)], Side::top, B);
    }
  }
  if (spec_.has_quiet) {  // a learned source of its own, kept only where a tile below fills the cells
    for (std::size_t q = 0; q < cells_.size(); ++q)
      if (cells_[q].on && find(cells_[q].i, cells_[q].j + 1) < 0) suppress(*tiles_[q], spec_.quiet[0], spec_.quiet[1], spec_.quiet[2], spec_.quiet[3]);
  }
  if (on_ && (d.heat > 0.f || d.soot > 0.f || d.kick != 0.f || d.u != 0.f || d.v != 0.f)) {
    const float r = std::max(d.radius, 0.5f), reach = 3.f * r;
    const float len = std::sqrt((x_ - px_) * (x_ - px_) + (y_ - py_) * (y_ - py_));
    const int n = std::max(1, static_cast<int>(std::ceil(len / (0.5f * r))));
    const float tu = d.kick * vx_ / fps + d.u, tv = d.kick * vy_ / fps + d.v;
    const float inv_n = 1.f / static_cast<float>(n);
    for (int p = 0; p < n; ++p) {
      const float u = (static_cast<float>(p) + 0.5f) * inv_n;
      const float sx = px_ + u * (x_ - px_), sy = py_ + u * (y_ - py_);
      for (std::size_t q = 0; q < cells_.size(); ++q) {
        if (!cells_[q].on) continue;
        const Module& m = *tiles_[q];
        const float span = spec_.width;
        if (sx + reach < m.at.x || sx - reach > m.at.x + span || sy + reach < m.at.y || sy - reach > m.at.y + span) continue;
        deposit(*tiles_[q], sx, sy, r, d.heat * inv_n, d.soot * inv_n, tu, tv, inv_n);
      }
    }
  }
  const float keep = std::exp(-1.f / (std::max(spec_.fade, 1e-3f) * fps));
  for (std::size_t q = 0; q < cells_.size(); ++q) {  // tiles the source left: their material fades
    if (!cells_[q].on || t - cells_[q].fed <= spec_.hold) continue;
    Module& m = *tiles_[q];
    auto co = m.runner().coarse_mut();
    const int C = m.channels(), N = m.res() * m.res();
    for (int i = 0; i < N; ++i) {
      co[zs(i) * zs(C) + 2] *= keep;
      co[zs(i) * zs(C) + 3] *= keep;
    }
    for (float& v : m.runner().fine_heat_mut()) v *= keep;
    for (float& v : m.runner().fine_soot_mut()) v *= keep;
  }
}

}  // namespace nfx::compose
