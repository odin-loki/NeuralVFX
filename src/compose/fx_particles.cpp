// The procedural elements of the library's third round, as particles (compose.hpp, docs/COMPOSE.md §4.10): hail that
// bounces, spatter (coloured drops: blood, oil) that flies and stains, shards that spin and glint as they turn, and
// streaks of light whose radius, brightness and colour change along them. None is learned: they move by the simple rules
// below and are drawn by the code below, into the picture and into the engine layers (docs/ENGINES.md §11), from the
// scene's particle pool, so nothing is allocated. Their random choices (a bounce, how a shard lies) come from a hash of
// the particle's own state, never from the scene's generator. (Water's droplets, ripples and bubbles are not here.)
#include "compose.hpp"

#include <neuralfx/noise.hpp>

#include <algorithm>
#include <bit>
#include <cmath>

namespace nfx::compose {

namespace {

using Rgb = std::array<float, 3>;

int ifl(float v) { return static_cast<int>(std::floor(v)); }

// A number in [0, 1) from a key and a counter.
float h01(std::uint32_t key, std::uint32_t k) { return static_cast<float>(hash32(key + 0x9e3779b9U * (k + 1U)) >> 8) * (1.f / 16777216.f); }
std::uint32_t key_of(float a, float b) { return hash32(std::bit_cast<std::uint32_t>(a) ^ hash32(std::bit_cast<std::uint32_t>(b) + 0x632be5abU)); }

// Coverage of a pixel whose centre is d from the middle of a line (or a disc's centre) that reaches r either side: an
// edge a pixel wide.
float edge(float r, float d) { return std::clamp(r + 0.5f - d, 0.f, 1.f); }

// Where the pixels go: the picture (Particles::draw), or the engine layers (Particles::draw_layers). `cover` lays a
// colour over what is there with coverage a, and adds `light`; `add` adds light only. In the layers, as weather's: the
// colour as the picture's, the albedo (the colour, at most 1: a reflectance), the emission (the light), a flat normal
// facing the viewer and the optical depth -ln(1 - a) (0 exactly where the coverage is 0).
struct Picture {
  Image4& s;
  void cover(std::size_t at, float a, const Rgb& col, const Rgb& light) const {
    float* p = s.px.data() + at;
    const float k = 1.f - a;
    for (std::size_t c = 0; c < 3; ++c) p[c] = p[c] * k + a * col[c] + light[c];
    p[3] = 1.f - (1.f - p[3]) * k;
  }
  void add(std::size_t at, const Rgb& light) const {
    float* p = s.px.data() + at;
    for (std::size_t c = 0; c < 3; ++c) p[c] += light[c];
  }
};

struct Layers {
  Image4& C;
  Image4& A;
  Image4& E;
  Image4& N;
  std::span<std::array<int, 2>> spans;
  void mark(std::size_t at) const {
    const std::size_t i = at / 4, W = zs(C.w);
    const int x = static_cast<int>(i % W), y = static_cast<int>(i / W);
    std::array<int, 2>& sp = spans[zs(y)];
    sp[0] = std::min(sp[0], x);
    sp[1] = std::max(sp[1], x + 1);
  }
  void cover(std::size_t at, float a, const Rgb& col, const Rgb& light) const {
    mark(at);
    const float k = 1.f - a;
    float* c = C.px.data() + at;
    float* al = A.px.data() + at;
    float* e = E.px.data() + at;
    float* n = N.px.data() + at;
    for (std::size_t ch = 0; ch < 3; ++ch) {
      c[ch] = c[ch] * k + a * col[ch] + light[ch];
      al[ch] = al[ch] * k + a * std::min(col[ch], 1.f);
      e[ch] = e[ch] * k + light[ch];
    }
    c[3] = 1.f - (1.f - c[3]) * k;
    al[3] = c[3];
    n[0] *= k;
    n[1] *= k;
    n[2] = n[2] * k + a;
    n[3] += k > 0.f ? -std::log(k) : 6.f;
  }
  void add(std::size_t at, const Rgb& light) const {
    mark(at);
    float* c = C.px.data() + at;
    float* e = E.px.data() + at;
    for (std::size_t ch = 0; ch < 3; ++ch) {
      c[ch] += light[ch];
      e[ch] += light[ch];
    }
  }
};

std::size_t pixel(int x, int y, int W) { return (zs(y) * zs(W) + zs(x)) * 4; }

// The pixels within r of the segment from (ax, ay) to (bx, by) (a disc when they are the same): f(pixel, coverage).
template <class F>
void capsule(float ax, float ay, float bx, float by, float r, int W, int H, F&& f) {
  const float m = r + 1.f;
  const int x0 = std::max(0, ifl(std::min(ax, bx) - m)), x1 = std::min(W - 1, ifl(std::max(ax, bx) + m));
  const int y0 = std::max(0, ifl(std::min(ay, by) - m)), y1 = std::min(H - 1, ifl(std::max(ay, by) + m));
  const float dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy, inv = l2 > 1e-9f ? 1.f / l2 : 0.f;
  for (int yy = y0; yy <= y1; ++yy) {
    const float qy = fl(yy) + 0.5f - ay;
    for (int xx = x0; xx <= x1; ++xx) {
      const float qx = fl(xx) + 0.5f - ax;
      const float u = std::clamp((qx * dx + qy * dy) * inv, 0.f, 1.f);
      const float ex = qx - u * dx, ey = qy - u * dy;
      const float c = edge(r, std::sqrt(ex * ex + ey * ey));
      if (c > 0.f) f(pixel(xx, yy, W), c);
    }
  }
}

// A filled ellipse of radii rx, ry about (cx, cy), its edge a pixel wide. The distance to the edge is taken to first
// order (|q - 1| / |grad q|, q the ellipse's normalised radius).
template <class F>
void ellipse(float cx, float cy, float rx, float ry, int W, int H, F&& f) {
  rx = std::max(rx, 0.5f);
  ry = std::max(ry, 0.35f);
  const float m = 1.f;
  const int x0 = std::max(0, ifl(cx - rx - m)), x1 = std::min(W - 1, ifl(cx + rx + m));
  const int y0 = std::max(0, ifl(cy - ry - m)), y1 = std::min(H - 1, ifl(cy + ry + m));
  const float irx = 1.f / rx, iry = 1.f / ry;
  for (int yy = y0; yy <= y1; ++yy) {
    const float dy = fl(yy) + 0.5f - cy;
    for (int xx = x0; xx <= x1; ++xx) {
      const float dx = fl(xx) + 0.5f - cx;
      const float q = std::sqrt(dx * dx * irx * irx + dy * dy * iry * iry);
      const float gx = dx * irx * irx, gy = dy * iry * iry, g = std::sqrt(gx * gx + gy * gy);
      const float d = g > 1e-6f ? (q - 1.f) * q / g : -std::min(rx, ry);  // signed: negative inside
      const float c = std::clamp(0.5f - d, 0.f, 1.f);
      if (c > 0.f) f(pixel(xx, yy, W), c);
    }
  }
}

// A rhombus centred on (cx, cy), `hl` long either side along `angle` and `hw` across: a splinter.
template <class F>
void rhombus(float cx, float cy, float hl, float hw, float angle, int W, int H, F&& f) {
  const float cs = std::cos(angle), sn = std::sin(angle), m = hl + 1.f;
  const float across = hl * hw / std::sqrt(hl * hl + hw * hw);  // from the centre to an edge, square to it
  const int x0 = std::max(0, ifl(cx - m)), x1 = std::min(W - 1, ifl(cx + m)), y0 = std::max(0, ifl(cy - m)), y1 = std::min(H - 1, ifl(cy + m));
  for (int yy = y0; yy <= y1; ++yy) {
    const float dy = fl(yy) + 0.5f - cy;
    for (int xx = x0; xx <= x1; ++xx) {
      const float dx = fl(xx) + 0.5f - cx;
      const float u = dx * cs + dy * sn, v = -dx * sn + dy * cs;
      const float e = std::abs(u) / hl + std::abs(v) / hw;
      const float c = std::clamp((1.f - e) * across + 0.5f, 0.f, 1.f);
      if (c > 0.f) f(pixel(xx, yy, W), c);
    }
  }
}

// What a particle is, for drawing (screen coordinates).
struct View {
  Kind kind = Kind::hail;
  float x = 0, y = 0, px = 0, py = 0, temp = 0, size = 0, life = 0, age = 0, cool = 0, aux = 0, wa = 0, wu = 0, wv = 0, wf = 0;
  Rgb col{};
};

Rgb scaled(const Rgb& c, float k) { return {c[0] * k, c[1] * k, c[2] * k}; }

template <class P>
void draw_view(const View& v, const P& paint, int W, int H) {
  const Rgb none{};
  switch (v.kind) {
    case Kind::hail:
    case Kind::spatter: {
      if (v.kind == Kind::spatter && v.wf >= 1.f) {  // a stain: it spreads at once, lies, and fades at the end of its life
        const float r = v.cool * (0.55f + 0.45f * std::min(1.f, v.age / 0.12f));
        const float a = v.wa * std::clamp((v.life - v.age) / (0.4f * v.life + 1e-4f), 0.f, 1.f);
        ellipse(v.x, v.y, r, 0.35f * r, W, H, [&](std::size_t at, float c) { paint.cover(at, a * c, v.col, none); });
        return;
      }
      // a disc blurred along the last half step, as a camera's shutter blurs it (its coverage spread over the blur)
      const float r = v.size, fade = std::min(1.f, (v.life - v.age) * 3.f);
      const float tx = v.x + 0.5f * (v.px - v.x), ty = v.y + 0.5f * (v.py - v.y);
      const float len = std::hypot(v.x - tx, v.y - ty), blur = std::min(1.f, (2.f * r + 1.f) / (2.f * r + 1.f + len));
      const float a = v.wa * fade * blur;
      capsule(tx, ty, v.x, v.y, r, W, H, [&](std::size_t at, float c) { paint.cover(at, a * c, v.col, none); });
      return;
    }
    case Kind::shard: {  // a splinter that glints as it turns
      const float hl = 0.5f * v.size, hw = std::max(0.55f, 0.16f * v.size), fade = std::min(1.f, (v.life - v.age) * 2.5f);
      const float glint = v.wv * std::pow(std::abs(std::cos(2.f * v.temp + 0.7f)), 10.f) * fade;
      const Rgb lc{0.5f + 0.5f * v.col[0], 0.5f + 0.5f * v.col[1], 0.5f + 0.5f * v.col[2]};
      rhombus(v.x, v.y, hl, hw, v.temp, W, H, [&](std::size_t at, float c) { paint.cover(at, v.wa * fade * c, v.col, scaled(lc, glint * c)); });
      return;
    }
    case Kind::streak: {  // as a glow (draw_glow, compose.cpp), its radius, brightness and colour changing along it
      const float x0 = v.px, y0 = v.py, x1 = v.x, y1 = v.y, r0 = std::max(v.size, 0.3f), r1 = std::max(v.wv, 0.3f);
      const Rgb c0 = v.col, c1{v.aux, v.wa, v.wf};
      const int caps = static_cast<int>(v.cool);
      const float rm = std::max(r0, r1);
      const int xa = std::max(0, ifl(std::min(x0, x1) - rm)), xb = std::min(W - 1, ifl(std::max(x0, x1) + rm));
      const int ya = std::max(0, ifl(std::min(y0, y1) - rm)), yb = std::min(H - 1, ifl(std::max(y0, y1) + rm));
      if (!(v.temp > 0.f || v.wu > 0.f)) return;
      const float dx = x1 - x0, dy = y1 - y0, len2 = dx * dx + dy * dy, inv = len2 > 1e-12f ? 1.f / len2 : 0.f;
      for (int yy = ya; yy <= yb; ++yy) {
        const float qy = fl(yy) + 0.5f - y0;
        for (int xx = xa; xx <= xb; ++xx) {
          const float qx = fl(xx) + 0.5f - x0;
          float u = (qx * dx + qy * dy) * inv;
          if (len2 > 1e-12f) {
            if (u < 0.f) {
              if (!(caps & 1)) continue;
              u = 0.f;
            } else if (u >= 1.f) {
              if (!(caps & 2)) continue;
              u = 1.f;
            }
          } else {
            u = 0.f;
          }
          const float r = r0 + (r1 - r0) * u, ex = qx - u * dx, ey = qy - u * dy, d2 = (ex * ex + ey * ey) / (r * r);
          if (d2 >= 1.f) continue;
          const float b = (v.temp + (v.wu - v.temp) * u) * (1.f - d2) * (1.f - d2);
          if (!(b > 0.f)) continue;
          paint.add(pixel(xx, yy, W), {b * (c0[0] + (c1[0] - c0[0]) * u), b * (c0[1] + (c1[1] - c0[1]) * u), b * (c0[2] + (c1[2] - c0[2]) * u)});
        }
      }
      return;
    }
    default: return;
  }
}

}  // namespace

// --- spawning ----------------------------------------------------------------------------------------------------------
//
// The arrays a kind uses (compose.hpp has the general meaning of each):
//   hail    wu, wv: drift; wf: flow; aux: floor; cool: bounces left; temp: 0 falling, 1 bouncing, 2 lying; wa
//   spatter temp: gravity; aux: floor; wu: stain seconds; wf: 1 lying; cool: the stain's radius; wa
//   shard   size: length; temp: angle; cool: spin; aux: floor; wu: gravity; wv: shine; wf: 1 lying; wa
//   streak  (px, py) to (x, y); size, wv: radii; temp, wu: brightnesses; r, g, b and aux, wa, wf: colours; cool: caps

bool Particles::spawn_hail(float x, float y, float du, float dv, float radius, std::array<float, 3> colour, float alpha, float flow, float floor, int bounces,
                           float life) {
  if (n_ >= capacity()) return false;
  const std::size_t i = zs(n_++);
  kind_[i] = Kind::hail;
  x_[i] = px_[i] = x;
  y_[i] = py_[i] = y;
  vx_[i] = wu_[i] = du;
  vy_[i] = wv_[i] = dv;
  size_[i] = std::max(radius, 0.4f);
  life_[i] = life;
  age_[i] = 0.f;
  wf_[i] = flow;
  aux_[i] = floor;
  cool_[i] = static_cast<float>(std::max(0, bounces));
  temp_[i] = 0.f;
  r_[i] = colour[0];
  g_[i] = colour[1];
  b_[i] = colour[2];
  wa_[i] = alpha;
  return true;
}

bool Particles::spawn_spatter(float x, float y, float vx, float vy, float radius, std::array<float, 3> colour, float alpha, float gravity, float floor,
                              float stain, float life) {
  if (n_ >= capacity()) return false;
  const std::size_t i = zs(n_++);
  kind_[i] = Kind::spatter;
  x_[i] = px_[i] = x;
  y_[i] = py_[i] = y;
  vx_[i] = vx;
  vy_[i] = vy;
  size_[i] = std::max(radius, 0.25f);
  life_[i] = life;
  age_[i] = 0.f;
  temp_[i] = gravity;
  aux_[i] = floor;
  wu_[i] = stain;
  wv_[i] = 0.f;
  wf_[i] = 0.f;
  cool_[i] = 0.f;
  r_[i] = colour[0];
  g_[i] = colour[1];
  b_[i] = colour[2];
  wa_[i] = alpha;
  return true;
}

bool Particles::spawn_shard(float x, float y, float vx, float vy, float length, float angle, float spin, std::array<float, 3> colour, float alpha, float gravity,
                            float floor, float shine, float life) {
  if (n_ >= capacity()) return false;
  const std::size_t i = zs(n_++);
  kind_[i] = Kind::shard;
  x_[i] = px_[i] = x;
  y_[i] = py_[i] = y;
  vx_[i] = vx;
  vy_[i] = vy;
  size_[i] = std::max(length, 1.f);
  life_[i] = life;
  age_[i] = 0.f;
  temp_[i] = angle;
  cool_[i] = spin;
  aux_[i] = floor;
  wu_[i] = gravity;
  wv_[i] = shine;
  wf_[i] = 0.f;
  r_[i] = colour[0];
  g_[i] = colour[1];
  b_[i] = colour[2];
  wa_[i] = alpha;
  return true;
}

bool Particles::spawn_streak(float x0, float y0, float x1, float y1, float r0, float r1, float b0, float b1, std::array<float, 3> c0, std::array<float, 3> c1,
                             int caps, float life) {
  if (n_ >= capacity()) return false;
  const std::size_t i = zs(n_++);
  kind_[i] = Kind::streak;
  px_[i] = x0;
  py_[i] = y0;
  x_[i] = x1;
  y_[i] = y1;
  vx_[i] = vy_[i] = 0.f;
  size_[i] = std::max(r0, 0.3f);
  wv_[i] = std::max(r1, 0.3f);
  temp_[i] = b0;
  wu_[i] = b1;
  r_[i] = c0[0];
  g_[i] = c0[1];
  b_[i] = c0[2];
  aux_[i] = c1[0];
  wa_[i] = c1[1];
  wf_[i] = c1[2];
  cool_[i] = static_cast<float>(caps & 3);
  life_[i] = life;
  age_[i] = 0.f;
  return true;
}

// --- moving ------------------------------------------------------------------------------------------------------------

bool Particles::update_fx(int i, float dt, const FieldBus* bus, float flow_gain, float ground_y) {
  const std::size_t q = zs(i);
  const auto flow = [&](float k, float& fu, float& fv) {  // the bus's flow where it is, world pixels a second
    fu = fv = 0.f;
    if (!bus || k == 0.f) return;
    const auto s = bus->at(x_[q], y_[q]);
    fu = s.u * 30.f * flow_gain * k;
    fv = s.v * 30.f * flow_gain * k;
  };
  const auto step = [&] {
    px_[q] = x_[q];
    py_[q] = y_[q];
    x_[q] += vx_[q] * dt;
    y_[q] += vy_[q] * dt;
  };
  switch (kind_[q]) {
    case Kind::spatter: {
      if (wf_[q] >= 1.f) return true;  // a stain, lying
      vy_[q] += temp_[q] * dt;
      step();
      const float floor = std::min(aux_[q], ground_y);
      if (!(y_[q] >= floor) || !(vy_[q] > 0.f)) return true;
      if (!(wu_[q] > 0.f)) {  // no stain: gone
        kill(i);
        return false;
      }
      // where it crossed the floor in this step, it lies as a stain
      const float f = std::clamp((floor - py_[q]) / std::max(1e-6f, y_[q] - py_[q]), 0.f, 1.f);
      x_[q] = px_[q] = px_[q] + f * (x_[q] - px_[q]);
      y_[q] = py_[q] = floor;
      vx_[q] = vy_[q] = 0.f;
      cool_[q] = 2.6f * size_[q] + 1.f;
      wf_[q] = 1.f;
      age_[q] = 0.f;
      life_[q] = wu_[q];
      return true;
    }
    case Kind::hail: {
      if (temp_[q] >= 2.f) return true;  // lying where it settled
      if (temp_[q] < 1.f) {              // falling as rain does: drawn to its drift and the bus's flow
        float fu, fv;
        flow(wf_[q], fu, fv);
        const float drag = std::min(1.f, 6.f * dt);
        vx_[q] += (wu_[q] + fu - vx_[q]) * drag;
        vy_[q] += (wv_[q] + fv - vy_[q]) * drag;
      } else {  // thrown up by a bounce
        vy_[q] += 900.f * dt;
        vx_[q] *= std::exp(-1.2f * dt);
      }
      step();
      const float floor = std::min(aux_[q], ground_y);
      if (!(y_[q] >= floor)) return true;
      y_[q] = floor;
      const std::uint32_t key = key_of(x_[q], age_[q]);
      if (cool_[q] >= 1.f && vy_[q] > 80.f) {  // it bounces, less high each time, and a little sideways
        const float up = vy_[q] * (0.2f + 0.2f * h01(key, 0));
        vx_[q] = 0.5f * vx_[q] + (h01(key, 1) - 0.5f) * 0.8f * up;
        vy_[q] = -up;
        cool_[q] -= 1.f;
        temp_[q] = 1.f;
      } else {  // it lies a moment, then is gone
        vx_[q] = vy_[q] = 0.f;
        temp_[q] = 2.f;
        life_[q] = std::min(life_[q], age_[q] + 0.3f + 0.8f * h01(key, 2));
      }
      return true;
    }
    case Kind::shard: {
      if (wf_[q] >= 1.f) return true;  // lying
      temp_[q] += cool_[q] * dt;
      vy_[q] += wu_[q] * dt;
      const float k = std::exp(-0.35f * dt);
      vx_[q] *= k;
      vy_[q] *= k;
      step();
      const float floor = std::min(aux_[q], ground_y);
      if (!(y_[q] >= floor) || !(vy_[q] > 0.f)) return true;
      y_[q] = floor;
      if (vy_[q] > 90.f) {  // it bounces
        vy_[q] *= -0.25f;
        vx_[q] *= 0.55f;
        cool_[q] *= 0.5f;
      } else {  // it lies nearly flat, a moment, and is gone
        const std::uint32_t key = key_of(x_[q], age_[q]);
        vx_[q] = vy_[q] = 0.f;
        cool_[q] = 0.f;
        wf_[q] = 1.f;
        temp_[q] = (h01(key, 0) - 0.5f) * 0.4f;
        life_[q] = std::min(life_[q], age_[q] + 0.8f + 1.2f * h01(key, 1));
      }
      return true;
    }
    default: return true;  // streaks stay where they were made, for their life
  }
}

// --- drawing -----------------------------------------------------------------------------------------------------------

namespace {
template <class Arr>
View view_of(Kind k, std::size_t q, float cam_x, float cam_y, const Arr& a) {
  View v;
  v.kind = k;
  v.x = a.x[q] - cam_x;
  v.y = a.y[q] - cam_y;
  v.px = a.px[q] - cam_x;
  v.py = a.py[q] - cam_y;
  v.temp = a.temp[q];
  v.size = a.size[q];
  v.life = a.life[q];
  v.age = a.age[q];
  v.cool = a.cool[q];
  v.aux = a.aux[q];
  v.wa = a.wa[q];
  v.wu = a.wu[q];
  v.wv = a.wv[q];
  v.wf = a.wf[q];
  v.col = {a.r[q], a.g[q], a.b[q]};
  return v;
}
struct Arrays {
  const std::vector<float>&x, &y, &px, &py, &temp, &size, &life, &age, &cool, &aux, &wa, &wu, &wv, &wf, &r, &g, &b;
};
bool on_screen(const View& v) { return std::abs(v.x) < 1e6f && std::abs(v.y) < 1e6f && std::abs(v.px) < 1e6f && std::abs(v.py) < 1e6f && v.size < 1e5f; }
}  // namespace

void Particles::draw_fx(std::size_t q, Image4& screen, float cam_x, float cam_y) const {
  const Arrays a{x_, y_, px_, py_, temp_, size_, life_, age_, cool_, aux_, wa_, wu_, wv_, wf_, r_, g_, b_};
  const View v = view_of(kind_[q], q, cam_x, cam_y, a);
  if (on_screen(v)) draw_view(v, Picture{screen}, screen.w, screen.h);
}

void Particles::draw_fx_layers(std::size_t q, Image4& C, Image4& A, Image4& E, Image4& N, float cam_x, float cam_y, std::span<std::array<int, 2>> spans) const {
  const Arrays a{x_, y_, px_, py_, temp_, size_, life_, age_, cool_, aux_, wa_, wu_, wv_, wf_, r_, g_, b_};
  const View v = view_of(kind_[q], q, cam_x, cam_y, a);
  if (on_screen(v)) draw_view(v, Layers{C, A, E, N, spans}, C.w, C.h);
}

}  // namespace nfx::compose
