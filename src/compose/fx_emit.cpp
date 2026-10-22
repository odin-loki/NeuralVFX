// The emitters of the procedural elements added for the library (fx.hpp, docs/COMPOSE.md §4.10). Each frame, each makes
// particles of the kinds of fx_particles.cpp (hail falling), or lines of light remade every frame (glows and streaks, which
// live for that frame): an aurora's curtains, a rainbow's bands, meteors, a beam, a shield and its hits, glints. Shapes
// and timings come from a hash of the emitter's place in the script, its seed and the frame (or the time slot of a meteor
// or a hit), never from the scene's generator.
#include "fx.hpp"

#include <neuralfx/noise.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>

namespace nfx::compose::fx {

namespace {

using Rgb = std::array<float, 3>;
constexpr float kPi = 3.14159265f, kTau = 6.2831853f;

constexpr std::string_view kNames[] = {"hail", "aurora", "rainbow", "meteors", "beam", "shield", "glints"};

constexpr std::string_view kHail[] = {"in", "size", "density", "wind", "radius", "colour", "brightness", "flow", "seed", "speed"};
constexpr std::string_view kAurora[] = {"in", "size", "colour", "top", "brightness", "rate", "jitter", "seed"};
constexpr std::string_view kRainbow[] = {"at", "radius", "width", "brightness", "squash", "count"};
constexpr std::string_view kMeteors[] = {"in", "size", "rate", "speed", "direction", "length", "width", "colour", "brightness", "seed"};
constexpr std::string_view kBeam[] = {"along", "width", "colour", "brightness", "jitter", "rate", "flare", "fade", "seed"};
constexpr std::string_view kShield[] = {"at", "radius", "squash", "width", "colour", "brightness", "rate", "hit", "angle", "seed"};
constexpr std::string_view kGlints[] = {"at", "spread", "in", "size", "count", "rate", "radius", "colour", "brightness", "fade", "speed", "direction", "seed"};

constexpr std::string_view kNeedIn[] = {"in", "size"};
constexpr std::string_view kNeedAtRadius[] = {"at", "radius"};
constexpr std::string_view kNeedAlong[] = {"along"};

struct KeySlot {
  std::string_view key;
  V first;
  int n;
};
constexpr KeySlot kSlots[] = {
    {"at", v_x, 2},          {"in", v_x, 2},           {"along", v_x, 4},       {"size", v_w, 2},        {"colour", v_r, 3},     {"top", v_tr, 3},
    {"wind", v_wu, 2},       {"direction", v_dx, 2},   {"rate", v_rate, 1},     {"count", v_count, 1},   {"spread", v_spread, 1}, {"speed", v_speed, 1},
    {"radius", v_radius, 1},  {"width", v_width, 1},   {"brightness", v_bright, 1}, {"jitter", v_jitter, 1}, {"squash", v_squash, 1},
    {"density", v_density, 1},
    {"flow", v_flow, 1},     {"length", v_length, 1},  {"flare", v_flare, 1},   {"fade", v_fade, 1},     {"hit", v_hit, 1},      {"angle", v_angle, 1},
    {"seed", v_seed, 1}};

// A random number generator from a hash: [0, 1).
struct Rng {
  std::uint32_t h;
  float operator()() {
    h = hash32(h + 0x9e3779b9U);
    return static_cast<float>(h >> 8) * (1.f / 16777216.f);
  }
};

std::uint64_t seed_of(const Emit& e) { return static_cast<std::uint64_t>(std::max(0.f, e.get(v_seed, 1.f))); }
std::int32_t frame_of(float t, float fps) { return static_cast<std::int32_t>(std::lround(t * fps)); }
std::int32_t i32(std::int64_t v) { return static_cast<std::int32_t>(v); }
float smooth01(float v) {
  v = std::clamp(v, 0.f, 1.f);
  return v * v * (3.f - 2.f * v);
}
Rgb scaled(const Rgb& c, float k) { return {c[0] * k, c[1] * k, c[2] * k}; }
Rgb mix(const Rgb& a, const Rgb& b, float w) { return {a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w}; }
Rgb colour(const Emit& e, Rgb d) { return {e.get(v_r, d[0]), e.get(v_g, d[1]), e.get(v_b, d[2])}; }
// How many to make this frame for `per` a frame on average: its whole part, and one more with the chance of the rest.
int count_of(float per, Rng& rnd, int most) {
  per = std::max(0.f, per);
  const int whole = static_cast<int>(std::min(per, static_cast<float>(most)));
  return std::min(most, whole + (rnd() < per - static_cast<float>(whole) ? 1 : 0));
}

// --- sky ---------------------------------------------------------------------------------------------------------------

// Hail, as the weather of §4.6: the rectangle filled at once when it starts, then fed from its top edge and from upwind
// of it; pellets fall at `speed` (560) world pixels a second and bounce once or twice on its bottom edge (or the ground).
void hail(const Emit& e, Particles& P, float t, float fps, bool fresh, std::uint32_t salt) {
  Rng rnd{hash_cell(frame_of(t, fps), 83 + static_cast<std::int32_t>(salt), 4, seed_of(e))};
  const float x0 = e.get(v_x, 0.f), y0 = e.get(v_y, 0.f), W = std::max(1.f, e.get(v_w, 100.f)), H = std::max(1.f, e.get(v_h, 100.f));
  const float density = std::max(0.f, e.get(v_density, 1.2f)), wu = e.get(v_wu, 0.f), wv = e.get(v_wv, 0.f), size = std::max(0.3f, e.get(v_radius, 1.3f));
  const float flow = e.get(v_flow, 0.1f);
  const Rgb col = scaled(colour(e, {0.86f, 0.9f, 0.95f}), e.get(v_bright, 1.f));
  const float vy = std::max(5.f, e.get(v_speed, 560.f) + wv), shift = wu * H / vy, xa = x0 - std::max(0.f, shift), span = W + std::abs(shift);
  const float live = density * span * H / 10000.f;
  const int n = count_of(fresh ? live : live * vy / H / fps, rnd, 4096);
  const float floor = y0 + H;
  for (int i = 0; i < n; ++i) {
    const float x = xa + rnd() * span, y = fresh ? y0 + rnd() * H : y0 - rnd() * vy / fps;
    const float s = size * (0.7f + 0.6f * rnd()), fall = vy * (0.85f + 0.3f * rnd()), tone = 0.85f + 0.3f * rnd();
    const int bounces = rnd() < 0.6f ? 2 : 1;
    P.spawn_hail(x, y, wu, fall, s, scaled(col, tone), 0.9f, flow, floor, bounces, (floor - y) / fall * 1.3f + 3.f);
  }
}

// An aurora: curtains of rays over the rectangle `in`, `size`, their lower edge wandering across its lower part, their
// folds drifting (`jitter`) and their brightness shimmering along them, at `rate` times the natural pace; bright in
// `colour` at the lower edge, fading upwards into `top`.
void aurora(const Emit& e, Particles& P, float t, float fps) {
  const std::uint64_t seed = seed_of(e);
  const float x = e.get(v_x, 0.f), y = e.get(v_y, 0.f), W = std::max(10.f, e.get(v_w, 600.f)), H = std::max(10.f, e.get(v_h, 300.f));
  const Rgb base = colour(e, {0.25f, 1.f, 0.55f}), top{e.get(v_tr, 0.65f), e.get(v_tg, 0.3f), e.get(v_tb, 1.f)}, mid = mix(base, top, 0.45f);
  const float B = e.get(v_bright, 0.5f), pace = e.get(v_rate, 1.f), fold = e.get(v_jitter, 1.f), life = 1.5f / fps;
  if (!(B > 0.f)) return;
  const float sp = std::max(3.f, W / 170.f), r = 1.3f * sp;
  const int n = std::clamp(static_cast<int>(W / sp), 2, 400);
  for (int i = 0; i < n; ++i) {
    const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(n);
    const float X = x + u * W + 0.06f * W * fold * value_noise(3.f * u, 0.05f * pace * t, 0.5f, seed);
    const float yb = y + H * (0.8f + 0.12f * value_noise(2.2f * u + 7.f, 0.07f * pace * t, 1.5f, seed));
    const float h = H * (0.5f + 0.22f * value_noise(5.f * u + 3.f, 0.1f * pace * t, 2.5f, seed));
    const float shimmer = 0.5f + 0.5f * value_noise(16.f * u - 0.5f * pace * t, 0.4f * pace * t, 3.5f, seed);
    const float b = B * smooth01(u / 0.12f) * smooth01((1.f - u) / 0.12f) * (0.3f + 0.7f * shimmer * shimmer);
    if (!(b > 0.f)) continue;
    P.spawn_streak(X, yb + 0.05f * h, X, yb - 0.1f * h, r, r, 0.f, b, base, base, 0, life);
    P.spawn_streak(X, yb - 0.1f * h, X, yb - 0.45f * h, r, r, b, 0.45f * b, base, mid, 0, life);
    P.spawn_streak(X, yb - 0.45f * h, X, yb - h, r, 1.3f * r, 0.45f * b, 0.f, mid, top, 0, life);
  }
}

// A rainbow: seven bands of an arc about `at` (red outside at `radius`, `width` across all seven), fading into the
// ground at its feet; `count 2` adds the fainter second bow outside it, its colours reversed.
void rainbow(const Emit& e, Particles& P, float fps) {
  static constexpr Rgb kBands[7] = {{1.f, 0.16f, 0.1f}, {1.f, 0.5f, 0.08f}, {1.f, 0.92f, 0.2f}, {0.3f, 1.f, 0.3f},
                                    {0.2f, 0.55f, 1.f}, {0.3f, 0.25f, 0.95f}, {0.6f, 0.25f, 0.9f}};
  const float cx = e.get(v_x, 0.f), cy = e.get(v_y, 0.f), R = std::max(10.f, e.get(v_radius, 300.f)), w = std::max(2.f, e.get(v_width, 0.1f * R));
  const float B = e.get(v_bright, 0.35f), sq = e.get(v_squash, 1.f), life = 1.5f / fps;
  const int bows = std::clamp(static_cast<int>(e.get(v_count, 1.f)), 1, 2);
  if (!(B > 0.f)) return;
  for (int bow = 0; bow < bows; ++bow) {
    const float Rb = bow == 0 ? R : 1.3f * R, wb = bow == 0 ? w : 1.35f * w, Bb = bow == 0 ? B : 0.32f * B;
    const int segs = std::clamp(static_cast<int>(kPi * Rb / 12.f), 24, 128);
    for (int k = 0; k < 7; ++k) {
      const Rgb& c = kBands[bow == 0 ? k : 6 - k];
      const float rr = Rb - (static_cast<float>(k) + 0.5f) * wb / 7.f, rad = 0.75f * wb / 7.f + 0.6f;
      for (int s = 0; s < segs; ++s) {
        const float a0 = kPi * static_cast<float>(s) / static_cast<float>(segs), a1 = kPi * static_cast<float>(s + 1) / static_cast<float>(segs);
        const float b = Bb * smooth01(std::sin(0.5f * (a0 + a1)) / 0.3f) * (k == 6 ? 0.8f : 1.f);
        P.spawn_glow(cx + rr * std::cos(a0), cy - rr * sq * std::sin(a0), cx + rr * std::cos(a1), cy - rr * sq * std::sin(a1), rad, c, b, life, false, false);
      }
    }
  }
}

// Meteors (shooting stars): `rate` a second, each starting in the rectangle `in`, `size` at a time of its own slot, and
// crossing the sky along `direction` at about `speed` for under a second, its tail `length` long behind it.
void meteors(const Emit& e, Particles& P, float t, float fps, std::uint32_t salt) {
  const std::uint64_t seed = seed_of(e);
  const float rate = std::max(1e-3f, e.get(v_rate, 0.5f)), speed = e.get(v_speed, 650.f), dir = std::atan2(e.get(v_dy, 0.45f), e.get(v_dx, 1.f));
  const float x = e.get(v_x, 0.f), y = e.get(v_y, 0.f), W = e.get(v_w, 400.f), H = e.get(v_h, 100.f), len = e.get(v_length, 140.f);
  const float w = std::max(0.4f, e.get(v_width, 1.6f)), B = e.get(v_bright, 2.5f), life = 1.5f / fps;
  const Rgb col = colour(e, {0.9f, 0.95f, 1.f}), tail{col[0] * 0.9f, col[1] * 0.85f, col[2] * 0.8f};
  const auto i1 = static_cast<std::int64_t>(std::floor(t * rate));
  const std::int64_t i0 = std::max(static_cast<std::int64_t>(std::floor((t - 0.9f) * rate)) - 1, i1 - 64);
  for (std::int64_t i = i0; i <= i1; ++i) {
    Rng rnd{hash_cell(i32(i), 89 + static_cast<std::int32_t>(salt), 5, seed)};
    const float start = (static_cast<float>(i) + rnd()) / rate, L = 0.35f + 0.5f * rnd();
    const float sx = x + rnd() * W, sy = y + rnd() * H, ang = dir + (rnd() - 0.5f) * 0.3f, v = speed * (0.8f + 0.4f * rnd()), bb = B * (0.5f + 0.5f * rnd());
    if (!(t >= start && t < start + L)) continue;
    const float a = t - start, env = std::pow(std::sin(kPi * a / L), 0.6f), c = std::cos(ang), s = std::sin(ang);
    const float hx = sx + v * c * a, hy = sy + v * s * a, tl = std::min(len, v * a + 2.f);
    P.spawn_streak(hx - c * tl, hy - s * tl, hx, hy, 0.35f * w, w, 0.f, bb * env, tail, col, 2, life);
    P.spawn_glow(hx, hy, hx, hy, 2.6f * w, col, 0.35f * bb * env, life, true, true);
  }
}

// --- combat and magic --------------------------------------------------------------------------------------------------

// A beam along a line: a white-hot core and a wide glow in `colour`, their width wobbling by `jitter` in a pattern that
// runs along it at `rate`, its brightness going to `fade` of itself at the end, a flare of `flare` where it hits.
void beam(const Emit& e, Particles& P, float t, float fps) {
  const std::uint64_t seed = seed_of(e);
  const float x0 = e.get(v_x, 0.f), y0 = e.get(v_y, 0.f), x1 = e.get(v_x1, 0.f), y1 = e.get(v_y1, 0.f);
  const float width = std::max(0.3f, e.get(v_width, 2.5f)), B = e.get(v_bright, 2.5f), jitter = e.get(v_jitter, 0.35f), run = e.get(v_rate, 8.f);
  const float flare = std::max(0.f, e.get(v_flare, 5.f * width)), fade = e.get(v_fade, 1.f), life = 1.5f / fps;
  const Rgb col = colour(e, {1.f, 0.3f, 0.25f}), core = mix(col, {1.f, 1.f, 1.f}, 0.65f);
  const float dx = x1 - x0, dy = y1 - y0, L = std::sqrt(dx * dx + dy * dy);
  if (!(B > 0.f) || !(L >= 1.f)) return;
  const int n = std::clamp(static_cast<int>(std::ceil(L / 12.f)), 1, 64);
  const float flick = 0.9f + 0.1f * value_noise(22.f * t, 0.5f, 9.5f, seed);
  const auto wk = [&](int k) { return std::max(0.2f, 1.f + jitter * value_noise(0.45f * static_cast<float>(k) - run * t, 1.5f, 0.5f, seed)); };
  for (int k = 0; k < n; ++k) {
    const float u0 = static_cast<float>(k) / static_cast<float>(n), u1 = static_cast<float>(k + 1) / static_cast<float>(n);
    const float f0 = (1.f + (fade - 1.f) * u0) * flick, f1 = (1.f + (fade - 1.f) * u1) * flick, w0 = wk(k), w1 = wk(k + 1);
    const int caps = (k == 0 ? 1 : 0) | (k == n - 1 ? 2 : 0);
    const float ax = x0 + u0 * dx, ay = y0 + u0 * dy, bx = x0 + u1 * dx, by = y0 + u1 * dy;
    P.spawn_streak(ax, ay, bx, by, 3.2f * width * w0, 3.2f * width * w1, 0.28f * B * f0, 0.28f * B * f1, col, col, caps, life);
    P.spawn_streak(ax, ay, bx, by, 0.6f * width * w0, 0.6f * width * w1, B * f0, B * f1, core, core, caps, life);
  }
  P.spawn_glow(x0, y0, x0, y0, 2.4f * width, core, 0.7f * B * flick, life, true, true);
  if (flare > 0.f) {  // where it hits: a disc and four spikes turning slowly
    const float fb = B * std::max(0.f, fade) * flick, pulse = 0.8f + 0.2f * value_noise(25.f * t, 4.5f, 1.5f, seed);
    P.spawn_glow(x1, y1, x1, y1, flare * pulse, mix(col, {1.f, 1.f, 1.f}, 0.4f), 0.8f * fb, life, true, true);
    for (int k = 0; k < 4; ++k) {
      const float a = 0.7f * t + kPi * 0.5f * static_cast<float>(k) + 0.3f * value_noise(3.f * t, static_cast<float>(k), 2.5f, seed);
      P.spawn_streak(x1, y1, x1 + 2.2f * flare * pulse * std::cos(a), y1 + 2.2f * flare * pulse * std::sin(a), 0.25f * flare, 0.3f, 0.9f * fb, 0.f, core, col, 1, life);
    }
  }
}

// A shield: a bubble (an ellipse `squash` as high as wide) with a bright shimmering rim, fainter rings inside it (its
// thickness seen edge on) and a sheen at its upper left; where it is hit, a flash, the rim lit and ripples running out
// over it. Hits: `rate` a second at random places, and one the script places: `hit` its age in seconds (nothing when
// below 0 or over 0.7 s), `angle` where (degrees, counter-clockwise from the right).
void shield(const Emit& e, Particles& P, float t, float fps, std::uint32_t salt) {
  constexpr float kHit = 0.7f;
  const std::uint64_t seed = seed_of(e);
  const float cx = e.get(v_x, 0.f), cy = e.get(v_y, 0.f), R = std::max(4.f, e.get(v_radius, 80.f)), sq = e.get(v_squash, 1.f);
  const float w = std::max(0.4f, e.get(v_width, 2.2f)), B = e.get(v_bright, 1.4f), rate = std::max(0.f, e.get(v_rate, 0.f)), life = 1.5f / fps;
  const Rgb col = colour(e, {0.4f, 0.75f, 1.f}), light = mix(col, {1.f, 1.f, 1.f}, 0.5f);
  if (!(B > 0.f)) return;
  std::array<std::array<float, 2>, 6> hits{};  // angle, age
  int nh = 0;
  if (e.has(v_hit) && e.v[v_hit] >= 0.f && e.v[v_hit] < kHit) hits[zs(nh++)] = {e.get(v_angle, 200.f) * kPi / 180.f, e.v[v_hit]};
  if (rate > 0.f) {
    const auto i1 = static_cast<std::int64_t>(std::floor(t * rate));
    for (std::int64_t i = std::max(static_cast<std::int64_t>(std::floor((t - kHit) * rate)) - 1, i1 - 16); i <= i1 && nh < 6; ++i) {
      Rng rnd{hash_cell(i32(i), 97 + static_cast<std::int32_t>(salt), 6, seed)};
      const float start = (static_cast<float>(i) + rnd()) / rate, ang = kTau * rnd();
      if (t >= start && t < start + kHit) hits[zs(nh++)] = {ang, t - start};
    }
  }
  const auto boost = [&](float th) {  // how much brighter the rim is near the hits
    float b = 0.f;
    for (int k = 0; k < nh; ++k) {
      const float env = (1.f - hits[zs(k)][1] / kHit) * (1.f - hits[zs(k)][1] / kHit);
      const float d = std::remainder(th - hits[zs(k)][0], kTau);
      b += 2.5f * env * std::exp(-(d / 0.45f) * (d / 0.45f));
    }
    return b;
  };
  const auto at = [&](float th, float r) { return std::array<float, 2>{cx + r * std::cos(th), cy - r * sq * std::sin(th)}; };
  const int n = std::clamp(static_cast<int>(kTau * R / 10.f), 24, 160);
  for (int i = 0; i < n; ++i) {
    const float th0 = kTau * static_cast<float>(i) / static_cast<float>(n), th1 = kTau * static_cast<float>(i + 1) / static_cast<float>(n), m = 0.5f * (th0 + th1);
    const float bo = boost(m), b = B * (0.7f + 0.3f * value_noise(2.5f * m, 0.4f * t, 0.5f, seed)) * (1.f + bo);
    const auto p0 = at(th0, R), p1 = at(th1, R), q0 = at(th0, 0.93f * R), q1 = at(th1, 0.93f * R), s0 = at(th0, 0.8f * R), s1 = at(th1, 0.8f * R);
    P.spawn_glow(p0[0], p0[1], p1[0], p1[1], w, bo > 0.3f ? light : col, b, life, false, false);
    P.spawn_glow(q0[0], q0[1], q1[0], q1[1], 2.5f * w, col, 0.3f * B * (1.f + 0.5f * bo), life, false, false);
    P.spawn_glow(s0[0], s0[1], s1[0], s1[1], 5.f * w, col, 0.1f * B, life, false, false);
  }
  for (int i = 0; i < 12; ++i) {  // the sheen: an arc at the upper left
    const float a0 = 2.2f + 0.55f * static_cast<float>(i) / 12.f, a1 = 2.2f + 0.55f * static_cast<float>(i + 1) / 12.f;
    const auto p0 = at(a0, 0.86f * R), p1 = at(a1, 0.86f * R);
    P.spawn_glow(p0[0], p0[1], p1[0], p1[1], 1.3f * w, light, 0.5f * B * std::sin(kPi * (static_cast<float>(i) + 0.5f) / 12.f), life, false, false);
  }
  for (int k = 0; k < nh; ++k) {
    const float phi = hits[zs(k)][0], a = hits[zs(k)][1], env = (1.f - a / kHit) * (1.f - a / kHit);
    const auto h = at(phi, R);
    P.spawn_glow(h[0], h[1], h[0], h[1], 0.22f * R * (1.f - 0.4f * a / kHit), light, 2.2f * B * std::exp(-a / 0.1f), life, true, true);
    for (const float ring : {1.f, 0.55f}) {  // ripples running out from the hit, over the shield only
      const float rr = ring * 1.3f * R * (1.f - std::exp(-a / 0.16f));
      if (rr < 2.f) continue;
      const int m = std::clamp(static_cast<int>(kTau * rr / 9.f), 12, 96);
      for (int j = 0; j < m; ++j) {
        const float b0 = kTau * static_cast<float>(j) / static_cast<float>(m), b1 = kTau * static_cast<float>(j + 1) / static_cast<float>(m);
        const float ax = h[0] + rr * std::cos(b0), ay = h[1] + rr * std::sin(b0), bx = h[0] + rr * std::cos(b1), by = h[1] + rr * std::sin(b1);
        const float mx = (0.5f * (ax + bx) - cx) / R, my = (0.5f * (ay + by) - cy) / (R * sq);
        if (mx * mx + my * my > 0.97f) continue;
        P.spawn_glow(ax, ay, bx, by, 1.3f * w, light, 1.1f * B * env * (ring == 1.f ? 1.f : 0.6f), life, false, false);
      }
    }
  }
}

// Glints: `count` places that twinkle in turn, `rate` times a second each, lit for `fade` of the time (1: a slow pulse),
// at a new place each time (within `spread` of `at`, or in the rectangle `in`, `size`), drifting at `speed` along
// `direction` while lit: a four-pointed star of `radius` with fainter diagonals. `rate 0`: lit all the time, in place (a
// script drives their brightness and size: a coin's glint now and then).
void glints(const Emit& e, Particles& P, float t, float fps, std::uint32_t salt) {
  const std::uint64_t seed = seed_of(e);
  const bool rect = e.has(v_w);
  const float x = e.get(v_x, 0.f), y = e.get(v_y, 0.f), W = e.get(v_w, 0.f), H = e.get(v_h, 0.f), spread = std::max(0.f, e.get(v_spread, 0.f));
  const int count = std::clamp(static_cast<int>(e.get(v_count, 12.f)), 1, 256);
  const float rate = std::max(0.f, e.get(v_rate, 0.7f)), len = std::max(0.5f, e.get(v_radius, 7.f)), B = e.get(v_bright, 2.2f);
  const float lit = std::clamp(e.get(v_fade, 0.24f), 0.02f, 1.f), speed = e.get(v_speed, 0.f), life = 1.5f / fps;
  const float ddx = e.get(v_dx, 0.f), ddy = e.get(v_dy, -1.f), dl = std::max(1e-6f, std::hypot(ddx, ddy)), ux = ddx / dl, uy = ddy / dl;
  const Rgb col = colour(e, {1.f, 0.97f, 0.9f});
  if (!(B > 0.f)) return;
  for (int i = 0; i < count; ++i) {
    const float phase = static_cast<float>(hash_cell(i, 0, 101 + static_cast<std::int32_t>(salt), seed) >> 8) * (1.f / 16777216.f);
    const float cyc = rate * t + phase, c = std::floor(cyc), u = rate > 0.f ? cyc - c : 0.5f;  // rate 0: lit all the time, in place
    float env = rate <= 0.f ? 1.f : lit >= 1.f ? std::sin(kPi * u) : std::max(0.f, 1.f - std::abs(u - 0.5f) / (0.5f * lit));
    env *= env;
    if (env <= 0.01f) continue;
    Rng rnd{hash_cell(i, static_cast<std::int32_t>(c), 103 + static_cast<std::int32_t>(salt), seed)};
    float px, py;
    if (rect) {
      px = x + rnd() * W;
      py = y + rnd() * H;
    } else {
      const float a = kTau * rnd(), d = std::sqrt(rnd()) * spread;
      px = x + d * std::cos(a);
      py = y + d * std::sin(a);
    }
    const float drift = rate > 0.f ? speed * u / rate : 0.f, L = len * (0.6f + 0.8f * rnd()) * (0.5f + 0.5f * env), b = B * env;
    px += ux * drift;
    py += uy * drift;
    P.spawn_glow(px, py, px, py, 0.18f * L + 0.5f, col, b, life, true, true);
    for (int k = 0; k < 8; ++k) {
      const bool main = k < 4;
      const float a = kPi * 0.5f * static_cast<float>(k) + (main ? 0.f : kPi * 0.25f), l = main ? L : 0.45f * L;
      P.spawn_streak(px, py, px + l * std::cos(a), py + l * std::sin(a), 0.12f * l + 0.3f, 0.3f, (main ? 0.8f : 0.4f) * b, 0.f, col, col, 1, life);
    }
  }
}

}  // namespace

bool emitter_of(std::string_view word, Emitter& out) {
  for (std::size_t i = 0; i < std::size(kNames); ++i) {
    if (kNames[i] == word) {
      out = static_cast<Emitter>(i);
      return true;
    }
  }
  return false;
}

std::span<const std::string_view> keys(Emitter e) {
  switch (e) {
    case Emitter::hail: return kHail;
    case Emitter::aurora: return kAurora;
    case Emitter::rainbow: return kRainbow;
    case Emitter::meteors: return kMeteors;
    case Emitter::beam: return kBeam;
    case Emitter::shield: return kShield;
    case Emitter::glints: return kGlints;
  }
  return {};
}

std::span<const std::string_view> needs(Emitter e) {
  switch (e) {
    case Emitter::hail:
    case Emitter::aurora:
    case Emitter::meteors: return kNeedIn;
    case Emitter::rainbow:
    case Emitter::shield: return kNeedAtRadius;
    case Emitter::beam: return kNeedAlong;
    case Emitter::glints: return {};
  }
  return {};
}

bool slot_of(std::string_view key, Slot& out) {
  for (const KeySlot& k : kSlots) {
    if (k.key == key) {
      out = {k.first, k.n};
      return true;
    }
  }
  return false;
}

void emit(const Emit& e, Particles& P, float t, float fps, bool fresh, std::uint32_t salt) {
  switch (e.kind) {
    case Emitter::hail: return hail(e, P, t, fps, fresh, salt);
    case Emitter::aurora: return aurora(e, P, t, fps);
    case Emitter::rainbow: return rainbow(e, P, fps);
    case Emitter::meteors: return meteors(e, P, t, fps, salt);
    case Emitter::beam: return beam(e, P, t, fps);
    case Emitter::shield: return shield(e, P, t, fps, salt);
    case Emitter::glints: return glints(e, P, t, fps, salt);
  }
}

void burst(const Burst& b, Particles& P, float ground) {
  const float aim = std::atan2(b.dy, b.dx), spread = b.cone * kPi / 180.f, floor = std::isnan(b.floor) ? ground : b.floor;
  const bool own = !std::isnan(b.colour[0]);
  for (int i = 0; i < b.spatter; ++i) {  // each value drawn in its own statement: the order of the draws is fixed
    const float a = aim + (P.uniform() - 0.5f) * spread;
    const float s = b.speed * (0.3f + 0.9f * P.uniform());
    const float d = P.uniform() * b.radius;
    const float da = P.uniform() * kTau;
    const float r = (std::isnan(b.grain) ? 1.6f : b.grain) * (0.6f + 0.8f * P.uniform());
    const float k = 0.7f + 0.6f * P.uniform();
    P.spawn_spatter(b.x + d * std::cos(da), b.y + d * std::sin(da), s * std::cos(a), s * std::sin(a), r, own ? b.colour : Rgb{0.2f, 0.008f, 0.012f}, 0.88f,
                    900.f, floor, b.stain * k, 8.f);
  }
  for (int i = 0; i < b.shards; ++i) {
    const float a = aim + (P.uniform() - 0.5f) * spread;
    const float s = b.speed * (0.35f + 0.85f * P.uniform());
    const float len = (std::isnan(b.grain) ? 9.f : b.grain) * (0.6f + 0.8f * P.uniform());
    const float ang = P.uniform() * kTau;
    const float spin = (P.uniform() - 0.5f) * 36.f;
    const float life = 4.f + P.uniform();
    P.spawn_shard(b.x, b.y, s * std::cos(a), s * std::sin(a), len, ang, spin, own ? b.colour : Rgb{0.75f, 0.9f, 1.f}, 0.85f, 900.f, floor, own ? 0.5f : 1.6f,
                  life);  // ice glints brightly, other splinters a little
  }
}

}  // namespace nfx::compose::fx
