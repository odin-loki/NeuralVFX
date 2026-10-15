// Water for scenes (water.hpp, docs/EFFECTS_WATER.md): the seeded simulation, its picture and how the picture is drawn.
#include "water.hpp"

#include <neuralfx/noise.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nfx::compose {

namespace {

constexpr float kPi = 3.14159265f;
constexpr float kDeg = 0.017453293f;

int wfloor(float v) { return static_cast<int>(std::floor(v)); }
float clamp01(float v) { return std::clamp(v, 0.f, 1.f); }
float smoothstep01(float v) {
  v = clamp01(v);
  return v * v * (3.f - 2.f * v);
}

// The counter-based random numbers: a hash of the seed and three counters, uniform in [0, 1).
float rnd(std::uint64_t seed, std::int32_t a, std::int32_t b, std::int32_t c) {
  return static_cast<float>(hash_cell(a, b, c, seed) >> 8) * (1.f / 16777216.f);
}

// A count with a fractional expectation: its whole part, and one more with the chance of its fraction.
int count_of(float expected, float u) {
  const float e = std::max(0.f, expected);
  const float f = std::floor(e);
  return static_cast<int>(f) + (u < e - f ? 1 : 0);
}

}  // namespace

const char* water_kind_name(WaterKind k) {
  switch (k) {
    case WaterKind::pool: return "pool";
    case WaterKind::sheet: return "sheet";
    case WaterKind::jet: return "jet";
    case WaterKind::drips: return "drips";
    case WaterKind::bubbles: return "bubbles";
    case WaterKind::shaft: return "shaft";
  }
  return "?";
}

// --- the simulation ----------------------------------------------------------------------------------------------------

Water::Water(std::span<const WaterLayout> layout, int drops, int crowns, int rings) : layout_(layout.begin(), layout.end()), spec_(layout.size()) {
  const int E = elements();
  node0_.assign(zs(E), 0);
  strand0_.assign(zs(E), 0);
  cell0_.assign(zs(E), 0);
  foam0_.assign(zs(E), 0);
  item0_.assign(zs(E), 0);
  next_item_.assign(zs(E), 0);
  int nodes = 0, strands = 0, cells = 0, foam = 0, bubbles = 0, drips = 0, streaks = 0;
  for (int i = 0; i < E; ++i) {
    WaterLayout& L = layout_[zs(i)];
    if (L.onto >= 0 && (L.onto >= E || layout_[zs(L.onto)].kind != WaterKind::pool)) throw std::invalid_argument("water: 'onto' names a pool");
    node0_[zs(i)] = nodes;
    strand0_[zs(i)] = strands;
    cell0_[zs(i)] = cells;
    foam0_[zs(i)] = foam;
    switch (L.kind) {
      case WaterKind::sheet:
      case WaterKind::jet:
        L.strands = std::clamp(L.strands, 1, 256);
        L.nodes = std::clamp(L.nodes, 8, 512);
        nodes += L.strands * L.nodes;
        strands += L.strands;
        break;
      case WaterKind::pool:
        L.cells = std::clamp(L.cells, 8, 4096);
        L.count = std::clamp(L.count, 1, 8192);
        cells += L.cells;
        foam += L.count;
        break;
      case WaterKind::drips:
        L.count = std::clamp(L.count, 1, 8192);
        item0_[zs(i)] = drips;
        drips += L.count;
        break;
      case WaterKind::bubbles:
        L.count = std::clamp(L.count, 1, 8192);
        item0_[zs(i)] = bubbles;
        bubbles += L.count;
        break;
      case WaterKind::shaft:
        L.count = std::clamp(L.count, 1, 65536);
        item0_[zs(i)] = streaks;
        streaks += L.count;
        break;
    }
  }
  nodes_.assign(zs(nodes), Node{});
  head_.assign(zs(strands), 0);
  count_.assign(zs(strands), 0);
  h_.assign(zs(cells), 0.f);
  hv_.assign(zs(cells), 0.f);
  foam_.assign(zs(foam), Foam{});
  bubbles_.assign(zs(bubbles), Bubble{});
  drips_.assign(zs(drips), Drip{});
  streaks_.assign(zs(streaks), Streak{});
  drops_.assign(zs(std::max(1, drops)), Drop{});
  rings_.assign(zs(std::max(1, rings)), Ring{});
  crowns_.assign(zs(std::max(1, crowns)), Crown{});
  crown_nodes_.assign(crowns_.size() * kCrownStrands * kCrownNodes, Node{});
  crown_head_.assign(crowns_.size() * kCrownStrands, 0);
  crown_count_.assign(crowns_.size() * kCrownStrands, 0);
  bins_.assign(zs(E) + crowns_.size(), Bins{});
}

float Water::surface_y(int pool, float x) const {
  const WaterSpec& s = spec_[zs(pool)];
  const WaterLayout& L = layout_[zs(pool)];
  const float span = s.x1 - s.x0;
  if (!(std::abs(span) > 1e-3f)) return s.y0;
  const float u = std::clamp((x - s.x0) / span, 0.f, 1.f);
  const float c = u * fl(L.cells) - 0.5f;
  const int c0 = std::clamp(wfloor(c), 0, L.cells - 1), c1 = std::min(c0 + 1, L.cells - 1);
  const float f = clamp01(c - fl(c0));
  const float* h = h_.data() + cell0_[zs(pool)];
  float y = s.y0 + u * (s.y1 - s.y0) - (h[c0] + f * (h[c1] - h[c0]));
  if (s.swell != 0.f) y -= s.swell * std::sin(2.f * kPi * (x / std::max(1.f, s.wavelength) - time_ / std::max(0.05f, s.period)));
  return y;
}

float Water::land_y(int pool, float x, float floor) const {
  if (pool >= 0 && spec_[zs(pool)].on) {
    const WaterSpec& s = spec_[zs(pool)];
    if (x >= std::min(s.x0, s.x1) && x <= std::max(s.x0, s.x1)) return surface_y(pool, x);
  }
  return floor;
}

void Water::spawn_drop(const Drop& d) {
  Drop& o = drops_[zs(next_drop_)];
  next_drop_ = (next_drop_ + 1) % static_cast<int>(drops_.size());  // full: the oldest makes room
  o = d;
  o.alive = 1;
}

void Water::add_ring(int pool, float x, float size, float t) {
  if (pool < 0 || !(spec_[zs(pool)].surface > 0.f)) return;
  Ring& r = rings_[zs(next_ring_)];
  next_ring_ = (next_ring_ + 1) % static_cast<int>(rings_.size());
  r = {x, 0.f, t, size, 0.6f + 0.02f * size, pool};
}

void Water::add_foam(int pool, float x, float vx, float size, float life) {
  const WaterLayout& L = layout_[zs(pool)];
  Foam* f = foam_.data() + foam0_[zs(pool)];
  int& next = next_item_[zs(pool)];
  Foam& o = f[next];
  next = (next + 1) % L.count;  // full: the oldest makes room
  o = {x, vx, size, 0.f, life, 1};
}

void Water::stir(int pool, float x, float amount, float width, int frame) {
  if (pool < 0 || pool >= elements() || layout_[zs(pool)].kind != WaterKind::pool) return;
  wave(pool, x, amount, width);
  const std::uint64_t seed = layout_[zs(pool)].seed;
  const auto u = [&](int k) { return rnd(seed, frame, wfloor(8.f * x), 8100 + k); };
  if (u(0) < std::min(1.f, std::abs(amount) / 60.f)) add_foam(pool, x + (u(1) - 0.5f) * width, (u(2) - 0.5f) * 30.f, 1.5f + 0.2f * width * u(3), 2.f + 2.f * u(4));
}

void Water::wave(int pool, float x, float amount, float width) {
  if (pool < 0 || pool >= elements() || layout_[zs(pool)].kind != WaterKind::pool) return;
  const WaterSpec& s = spec_[zs(pool)];
  const WaterLayout& L = layout_[zs(pool)];
  const float span = s.x1 - s.x0;
  if (!(std::abs(span) > 1e-3f)) return;
  const float dx = std::abs(span) / fl(L.cells);
  const float c = (x - s.x0) / span * fl(L.cells) - 0.5f, w = std::max(0.6f, 0.5f * width / dx);
  const int lo = std::max(0, wfloor(c - 3.f * w)), hi = std::min(L.cells - 1, wfloor(c + 3.f * w) + 1);
  float* hv = hv_.data() + cell0_[zs(pool)];
  // down at x and up to its sides (a Mexican hat, its sum nothing): water pushed aside, not lost, so that what keeps
  // landing in one place stirs the surface without digging a hole in it
  float sum = 0.f;
  for (int j = lo; j <= hi; ++j) {
    const float e = (fl(j) - c) / w;
    sum += (1.f - 2.f * e * e) * std::exp(-e * e);
  }
  const float mean = sum / fl(hi - lo + 1);
  for (int j = lo; j <= hi; ++j) {
    const float e = (fl(j) - c) / w;
    hv[j] -= amount * ((1.f - 2.f * e * e) * std::exp(-e * e) - mean);
  }
}

// Something lands on a pool (or the ground) at x: the pool is pushed down where it lands, a ring spreads where its
// surface is seen, foam is left where a lot lands at speed, a little water is thrown back up; its mist is counted.
void Water::land(int source, int pool, float x, float y, float mass, float speed, float r, int frame, std::uint32_t salt) {
  const std::uint64_t seed = source >= 0 && source < elements() ? layout_[zs(source)].seed : 0x51ce5u;
  const auto u = [&](int k) { return rnd(seed, frame, static_cast<std::int32_t>(salt), 7000 + k); };
  const bool on_pool = pool >= 0 && spec_[zs(pool)].on && x >= std::min(spec_[zs(pool)].x0, spec_[zs(pool)].x1) &&
                       x <= std::max(spec_[zs(pool)].x0, spec_[zs(pool)].x1);
  const float hit = mass * std::sqrt(std::max(0.f, speed));
  if (on_pool) {
    wave(pool, x, std::min(400.f, 6.f * hit), std::max(3.f, 2.5f * r));
    if (u(0) < std::min(1.f, 0.3f + 2.f * mass)) add_ring(pool, x, std::clamp(6.f + 3.f * r + 0.6f * hit, 6.f, 60.f), time_);
    if (hit > 4.f && u(1) < std::min(1.f, 0.05f * hit)) add_foam(pool, x, (u(2) - 0.5f) * 60.f, 1.5f + 2.5f * u(3), 1.2f + 2.5f * u(4));
  }
  if (speed > 160.f && u(5) < std::min(0.6f, 0.4f * mass + 0.05f)) {  // a little water thrown back up
    Drop d;
    d.x = x;
    d.y = y - 1.f;
    d.vx = (u(6) - 0.5f) * 0.35f * speed;
    d.vy = -(0.12f + 0.2f * u(7)) * speed;
    d.r = 0.6f + 0.7f * u(8);
    d.life = 1.2f;
    d.g = 700.f;
    d.floor = y + 2.f;
    d.pool = pool;
    d.source = -1;
    d.white = 1;
    d.bounced = 1;
    spawn_drop(d);
  }
  if (source >= 0) {
    Bins& b = bins_[zs(source)];
    const int k = std::clamp(wfloor((x - b.lo) / std::max(1e-3f, b.hi - b.lo) * fl(kBins)), 0, kBins - 1);
    b.mass[zs(k)] += mass;
  }
}

void Water::step(float t, float dt, int frame) {
  time_ = t;
  dt_ = dt;
  for (Bins& b : bins_) b.mass.fill(0.f);
  const int E = elements();
  for (int i = 0; i < E; ++i)
    if (layout_[zs(i)].kind == WaterKind::pool) step_pool(i, t, dt, frame);
  for (int i = 0; i < E; ++i) {
    switch (layout_[zs(i)].kind) {
      case WaterKind::sheet:
      case WaterKind::jet: step_strands(i, t, dt, frame); break;
      case WaterKind::drips: step_drips(i, t, dt, frame); break;
      case WaterKind::bubbles: step_bubbles(i, t, dt, frame); break;
      case WaterKind::shaft: step_shaft(i, t, dt, frame); break;
      case WaterKind::pool: break;
    }
  }
  step_crowns(t, dt, frame);
  step_drops(dt, frame);
  feed_mist();
}

// A damped wave equation on the pool's cells, h_tt = c^2 h_xx - 2 k h_t, its ends reflecting; rain on it; its foam drifts.
void Water::step_pool(int i, float t, float dt, int frame) {
  const WaterSpec& s = spec_[zs(i)];
  const WaterLayout& L = layout_[zs(i)];
  const int N = L.cells;
  float* h = h_.data() + cell0_[zs(i)];
  float* hv = hv_.data() + cell0_[zs(i)];
  const float len = std::max(1.f, std::hypot(s.x1 - s.x0, s.y1 - s.y0)), dx = len / fl(N);
  const float c = s.speed > 0.f ? s.speed : 140.f, k = std::max(0.f, s.damping);
  const int sub = std::clamp(static_cast<int>(std::ceil(c * dt / (0.45f * dx))), 1, 32);
  const float dts = dt / fl(sub), cc = c * c / (dx * dx), cap = std::max(2.f, 0.35f * std::max(s.depth, s.surface));
  constexpr float kLevel = 4.f;  // (2 per second)^2: the surface settles back to its level (long waves do not linger)
  for (int q = 0; q < sub; ++q) {
    for (int j = 0; j < N; ++j) {
      const float l = h[std::max(0, j - 1)] + h[std::min(N - 1, j + 1)] - 2.f * h[j];
      hv[j] += dts * (cc * l - 2.f * k * hv[j] - kLevel * h[j]);
    }
    for (int j = 0; j < N; ++j) h[j] = std::clamp(h[j] + dts * hv[j], -cap, cap);
  }
  if (s.on && s.rate > 0.f) {  // rain on the pool: drops at random places, rings and a fleck thrown up
    const int n = count_of(s.rate * len / 100.f * dt, rnd(L.seed, frame, 0, 9001));
    for (int q = 0; q < std::min(n, 256); ++q) {
      const float u = rnd(L.seed, frame, q, 9002), x = s.x0 + u * (s.x1 - s.x0);
      wave(i, x, 25.f + 25.f * rnd(L.seed, frame, q, 9003), 2.f * dx);
      add_ring(i, x, 5.f + 7.f * rnd(L.seed, frame, q, 9004), t);
      if (rnd(L.seed, frame, q, 9005) < 0.35f) {
        Drop d;
        d.x = x;
        d.y = surface_y(i, x) - 1.f;
        d.vx = (rnd(L.seed, frame, q, 9006) - 0.5f) * 40.f;
        d.vy = -(40.f + 50.f * rnd(L.seed, frame, q, 9007));
        d.r = 0.6f;
        d.life = 0.6f;
        d.g = 700.f;
        d.floor = d.y + 2.f;
        d.pool = i;
        d.white = 1;
        d.bounced = 1;
        spawn_drop(d);
      }
    }
  }
  Foam* f = foam_.data() + foam0_[zs(i)];
  const float lo = std::min(s.x0, s.x1), hi = std::max(s.x0, s.x1), slow = std::exp(-1.5f * dt);
  for (int q = 0; q < L.count; ++q) {
    Foam& o = f[q];
    if (!o.alive) continue;
    o.x += (s.drift + o.vx) * dt;
    o.vx *= slow;
    o.age += dt;
    if (o.age >= o.life || o.x < lo || o.x > hi) o.alive = 0;
  }
}

// Sheets and jets: each strand emits a node every frame at its source; nodes fall under their weight; past a time of
// their own they break into drops (and throw spray); where they land, they push the pool down, ring, foam, and feed the
// mist.
void Water::step_strands(int i, float t, float dt, int frame) {
  const WaterSpec& s = spec_[zs(i)];
  const WaterLayout& L = layout_[zs(i)];
  const bool jet = L.kind == WaterKind::jet;
  const int S = L.strands, M = L.nodes;
  const float g = s.gravity, breakup = std::max(0.05f, s.breakup);
  // where it lands, for its mist: under the lip, or around the nozzle as far as its throw reaches
  Bins& b = bins_[zs(i)];
  if (jet) {
    const float reach = std::max(30.f, s.speed * s.speed / std::max(1.f, g) * 0.6f);
    b.lo = s.x0 - reach;
    b.hi = s.x0 + reach;
  } else {
    const float w = std::abs(s.x1 - s.x0);
    b.lo = std::min(s.x0, s.x1) - 0.3f * w - 20.f + std::min(0.f, s.vx) * 0.5f;
    b.hi = std::max(s.x0, s.x1) + 0.3f * w + 20.f + std::max(0.f, s.vx) * 0.5f;
  }
  b.y = L.onto >= 0 ? spec_[zs(L.onto)].y0 : s.floor;
  const float drag = 1.f - std::exp(-1.2f * dt);
  const float gap = jet ? 2.f : std::hypot(s.x1 - s.x0, s.y1 - s.y0) / fl(S);  // between neighbouring strands
  for (int st = 0; st < S; ++st) {
    Node* ring = nodes_.data() + node0_[zs(i)] + st * M;
    int& head = head_[zs(strand0_[zs(i)] + st)];
    int& count = count_[zs(strand0_[zs(i)] + st)];
    for (int q = 0; q < count; ++q) {  // move
      Node& n = ring[(head + q) % M];
      if (!n.alive) continue;
      n.vy += g * dt;
      if (n.broken) {
        n.vx += drag * (s.wu - n.vx);
        n.vy += drag * s.wv;
      }
      n.x += n.vx * dt;
      n.y += n.vy * dt;
      n.age += dt;
      if (!n.broken && n.age >= n.life) {  // it breaks: into a drop, or into spray and mist only; spray off it
        n.broken = 1;
        const std::uint32_t id = static_cast<std::uint32_t>(st * 1031 + (head + q) % M);
        const auto ub = [&](int k) { return rnd(L.seed, frame, static_cast<std::int32_t>(id), k) - 0.5f; };
        n.vx += ub(105) * 80.f;
        n.vy *= 1.f + 0.2f * ub(107);
        // off the lattice it was emitted on (a strand's place, a frame's step): a node a strand and a frame wide
        n.x += ub(108) * gap;
        n.y += ub(109) * std::abs(n.vy) * dt;
        if (rnd(L.seed, frame, static_cast<std::int32_t>(id), 106) < 0.55f) n.alive = 0;
        if (rnd(L.seed, frame, static_cast<std::int32_t>(id), 101) < 0.45f * std::min(1.5f, n.w)) {
          Drop d;
          d.x = n.x;
          d.y = n.y;
          d.vx = n.vx + (rnd(L.seed, frame, static_cast<std::int32_t>(id), 102) - 0.5f) * 90.f;
          d.vy = 0.85f * n.vy - 40.f * rnd(L.seed, frame, static_cast<std::int32_t>(id), 103);
          d.r = 0.6f + 0.8f * rnd(L.seed, frame, static_cast<std::int32_t>(id), 104);
          d.life = 1.6f;
          d.g = g;
          d.wu = s.wu;
          d.floor = s.floor;
          d.pool = L.onto;
          d.source = i;
          d.white = 1;
          spawn_drop(d);
        }
      }
      if (n.vy > 0.f) {
        const float ly = land_y(L.onto, n.x, s.floor);
        if (n.y >= ly) {
          land(i, L.onto, n.x, ly, n.w, std::hypot(n.vx, n.vy), 2.f, frame, static_cast<std::uint32_t>(st * 4099 + (head + q) % M));
          n.alive = 0;
        }
      }
    }
    while (count > 0 && !ring[head].alive) {  // the oldest that are gone
      head = (head + 1) % M;
      --count;
    }
    if (!(s.on && s.flow > 0.f)) continue;
    // emit
    Node n;
    n.alive = 1;
    n.born = frame * 8;
    const auto u = [&](int k) { return rnd(L.seed, frame, st, k); };
    if (jet) {
      // the aim wanders smoothly by `spread` (a stream that sways), and each node a little about it
      const float a = (s.angle + (fl(st) - 0.5f * fl(S - 1)) * s.fan + s.spread * value_noise(3.f * t, 0.71f * fl(st), 4.5f, L.seed) +
                       (u(1) - 0.5f) * 0.4f * s.spread) * kDeg;
      const float V = s.speed * (1.f + s.pulse * value_noise(0.9f * t, 0.37f * fl(st), 2.5f, L.seed)) * (0.97f + 0.06f * u(2));
      n.x = s.x0 + (u(3) - 0.5f) * 1.5f;
      n.y = s.y0;
      n.vx = V * std::sin(a);
      n.vy = -V * std::cos(a);
      n.w = s.flow * (0.75f + 0.25f * u(4));
      const float apex = std::max(0.3f, -n.vy / std::max(1.f, g));  // (thrown down: a trickle that holds a while)
      n.life = apex * (0.7f + 0.5f * u(5)) / breakup;
      n.v0 = V;
    } else {
      // where along the lip and how fast: each strand wanders smoothly in time (the sheet ripples), each node a little
      const float k = (fl(st) + 0.5f + 0.4f * value_noise(0.9f * fl(st), 1.1f * t, 6.5f, L.seed)) / fl(S);
      const float sp = std::hypot(s.vx, s.vy) + 20.f;
      n.x = s.x0 + k * (s.x1 - s.x0);
      n.y = s.y0 + k * (s.y1 - s.y0);
      n.vx = s.vx + sp * (0.1f * value_noise(0.83f * fl(st), 1.7f * t, 7.5f, L.seed) + 0.02f * (u(2) - 0.5f));
      n.vy = s.vy + sp * (0.08f * value_noise(0.61f * fl(st), 1.3f * t, 8.5f, L.seed) + 0.02f * (u(3) - 0.5f));
      // streaks: each strand's thickness swells and thins with smooth noise over its index and time
      const float streak = 0.3f + 0.7f * smoothstep01(0.5f + 0.9f * value_noise(0.43f * fl(st), 1.3f * t, 0.5f, L.seed));
      n.w = s.flow * streak;
      // when it breaks: smooth over the strands and time (the sheet tears into tongues, not into blocks), a little each
      const float hold = smoothstep01(0.5f + 0.8f * value_noise(0.35f * fl(st), 0.9f * t, 9.5f, L.seed));
      n.life = 0.75f * (0.5f + 1.f * hold) * (0.9f + 0.2f * u(4)) * (0.6f + 0.6f * std::min(1.5f, n.w)) / breakup;
      n.v0 = std::max(30.f, std::hypot(n.vx, n.vy));
    }
    if (count == M) {  // full: the oldest node makes room
      head = (head + 1) % M;
      --count;
    }
    ring[(head + count) % M] = n;
    ++count;
  }
}

// Drips: drops swell at random points along the line, let go after a while and fall (as drops).
void Water::step_drips(int i, float t, float dt, int frame) {
  (void)t;
  const WaterSpec& s = spec_[zs(i)];
  const WaterLayout& L = layout_[zs(i)];
  Drip* d = drips_.data() + item0_[zs(i)];
  Bins& b = bins_[zs(i)];
  b.lo = std::min(s.x0, s.x1) - 20.f;
  b.hi = std::max(s.x0, s.x1) + 20.f;
  b.y = L.onto >= 0 ? spec_[zs(L.onto)].y0 : s.floor;
  for (int q = 0; q < L.count; ++q) {
    Drip& o = d[q];
    if (!o.alive) continue;
    o.age += dt;
    if (o.age >= o.form) {
      Drop p;
      p.x = o.x;
      p.y = o.y + o.r;
      p.r = o.r;
      p.life = 6.f;
      p.g = s.gravity;
      p.wu = s.wu;
      p.floor = s.floor;
      p.pool = L.onto;
      p.source = i;
      spawn_drop(p);
      o.alive = 0;
    }
  }
  if (!s.on) return;
  const int n = count_of(s.rate * dt, rnd(L.seed, frame, 0, 201));
  for (int q = 0; q < std::min(n, 64); ++q) {
    Drip& o = d[next_item_[zs(i)]];
    next_item_[zs(i)] = (next_item_[zs(i)] + 1) % L.count;
    const float u = rnd(L.seed, frame, q, 202);
    o = {s.x0 + u * (s.x1 - s.x0), s.y0 + u * (s.y1 - s.y0), s.radius * (0.7f + 0.5f * rnd(L.seed, frame, q, 203)), 0.f,
         0.4f + 1.f * rnd(L.seed, frame, q, 204), 1};
  }
}

// Bubbles: rising from the bottom of their region, wobbling, popping at the pool's surface (or the region's top).
void Water::step_bubbles(int i, float t, float dt, int frame) {
  const WaterSpec& s = spec_[zs(i)];
  const WaterLayout& L = layout_[zs(i)];
  Bubble* p = bubbles_.data() + item0_[zs(i)];
  const float rise = s.speed > 0.f ? s.speed : 45.f;
  for (int q = 0; q < L.count; ++q) {
    Bubble& o = p[q];
    if (!o.alive) continue;
    o.age += dt;
    o.y -= o.vy * dt;
    o.r *= 1.f + 0.06f * dt;  // a little larger as it rises
    const float x = o.x + o.wobble * std::sin(o.phase + 5.f * o.age);
    const float top = L.onto >= 0 && spec_[zs(L.onto)].on ? land_y(L.onto, x, s.y0) : s.y0;
    if (o.y - o.r <= top) {  // pops
      if (L.onto >= 0 && spec_[zs(L.onto)].on) {
        wave(L.onto, x, 8.f + 6.f * o.r, 2.f * o.r);
        add_ring(L.onto, x, 3.f + 2.5f * o.r, t);
        if (rnd(L.seed, frame, q, 302) < 0.5f) {
          Drop d;
          d.x = x;
          d.y = top - 1.f;
          d.vx = (rnd(L.seed, frame, q, 303) - 0.5f) * 30.f;
          d.vy = -(30.f + 40.f * rnd(L.seed, frame, q, 304));
          d.r = 0.5f;
          d.life = 0.5f;
          d.g = s.gravity;
          d.floor = top + 2.f;
          d.pool = L.onto;
          d.white = 1;
          d.bounced = 1;
          spawn_drop(d);
        }
      }
      o.alive = 0;
    }
  }
  if (!s.on) return;
  const int n = count_of(s.rate * dt, rnd(L.seed, frame, 0, 301));
  for (int q = 0; q < std::min(n, 64); ++q) {
    Bubble& o = p[next_item_[zs(i)]];
    next_item_[zs(i)] = (next_item_[zs(i)] + 1) % L.count;
    const float r = s.radius * (0.3f + 0.7f * rnd(L.seed, frame, q, 305));
    o = {s.x0 + rnd(L.seed, frame, q, 306) * s.x1, s.y0 + s.y1 - 2.f * rnd(L.seed, frame, q, 307), r,
         rise * (0.7f + 0.6f * rnd(L.seed, frame, q, 308)) * (0.6f + 0.4f * r / std::max(0.1f, s.radius)), 0.f,
         2.f * kPi * rnd(L.seed, frame, q, 309), 0.5f + 1.5f * rnd(L.seed, frame, q, 310), 1};
  }
}

namespace {

// A rain shaft's density across its width (x: 0 to 1 of it) at time t: thicker shafts and thinner gaps that swell and
// thin, soft at its edges.
float shaft_profile(const WaterSpec& s, std::uint64_t seed, float x, float t) {
  const float edge = smoothstep01(x / 0.3f) * smoothstep01((1.f - x) / 0.3f);
  const float n = value_noise(x * std::max(1.f, s.x1) / 90.f, 0.25f * t, 3.5f, seed);
  return edge * ((1.f - s.gust) + s.gust * smoothstep01(0.5f + 1.2f * n));
}

// Its wind now: the wind, swinging by `sway` with smooth noise.
float shaft_wind(const WaterSpec& s, std::uint64_t seed, float t) { return s.wu + s.sway * value_noise(0.35f * t, 0.5f, 7.5f, seed); }

}  // namespace

// A rain shaft: streaks falling through the region, more where its profile is dense, slanted by its wind; at the bottom
// they feed the mist.
void Water::step_shaft(int i, float t, float dt, int frame) {
  const WaterSpec& s = spec_[zs(i)];
  const WaterLayout& L = layout_[zs(i)];
  Streak* p = streaks_.data() + item0_[zs(i)];
  const float W = std::max(1.f, s.x1), H = std::max(1.f, s.y1), V = s.speed > 0.f ? s.speed : 520.f;
  Bins& b = bins_[zs(i)];
  b.lo = s.x0 - 0.5f * W;
  b.hi = s.x0 + 1.5f * W;
  b.y = s.y0 + H;
  for (int q = 0; q < L.count; ++q) {
    Streak& o = p[q];
    if (!o.alive) continue;
    o.x += o.vx * dt;
    o.y += o.vy * dt;
    if (o.y >= o.bottom) {
      land(i, -1, o.x, o.bottom, 0.04f, o.vy, 0.5f, frame, static_cast<std::uint32_t>(q));
      o.alive = 0;
    }
  }
  if (!s.on) return;
  const float wind = shaft_wind(s, L.seed, t);
  const float live = std::max(0.f, s.density) * W * H / 10000.f;
  const int n = count_of(live * V * dt / H, rnd(L.seed, frame, 0, 401));
  for (int q = 0; q < std::min(n, 4096); ++q) {
    for (int tries = 0; tries < 4; ++tries) {  // where: as dense as the profile
      const float u = rnd(L.seed, frame, q, 410 + tries);
      if (rnd(L.seed, frame, q, 420 + tries) > shaft_profile(s, L.seed, u, t)) continue;
      Streak& o = p[next_item_[zs(i)]];
      next_item_[zs(i)] = (next_item_[zs(i)] + 1) % L.count;
      const float v = V * (0.85f + 0.3f * rnd(L.seed, frame, q, 430));
      o = {s.x0 + u * W, s.y0 + rnd(L.seed, frame, q, 431) * v * dt, wind, v, v * 0.05f * (0.7f + 0.6f * rnd(L.seed, frame, q, 432)),
           0.16f + 0.2f * rnd(L.seed, frame, q, 433), s.y0 + H, 1};
      break;
    }
  }
}

void Water::splash(float x, float y, float power, int onto, Module* mist, float mist_amount, float mist_heat, std::uint64_t seed, float t, int frame) {
  const int ci = next_crown_;
  next_crown_ = (next_crown_ + 1) % static_cast<int>(crowns_.size());  // all in use: the oldest makes room
  power = std::clamp(power, 0.05f, 3.f);
  Crown& c = crowns_[zs(ci)];
  c = {x, y, t, power, mist_amount, mist_heat, onto, mist, seed, 1};
  for (int j = 0; j < kCrownStrands; ++j) {
    crown_head_[zs(ci * kCrownStrands + j)] = 0;
    crown_count_[zs(ci * kCrownStrands + j)] = 0;
  }
  const int E = elements();
  Bins& b = bins_[zs(E + ci)];
  b.lo = x - 60.f - 140.f * power;
  b.hi = x + 60.f + 140.f * power;
  b.y = y;
  b.mass.fill(0.f);
  const auto u = [&](int k, int salt) { return rnd(seed, frame, k, salt); };
  // spray with the crown (its walls are strands, thrown in step_crowns): drops up and out from the crater's rim, both
  // ways, steep and fast or flat and slow
  const int n = std::clamp(static_cast<int>(16.f + 70.f * power), 8, 300);
  const float rim = 4.f + 10.f * power;
  for (int k = 0; k < n; ++k) {
    const float side = (k & 1) ? 1.f : -1.f;
    const float th = (10.f + 50.f * u(k, 1)) * kDeg;
    const float sp = (150.f + 260.f * power) * (0.55f + 0.6f * u(k, 2)) * (1.15f - 0.35f * th / (60.f * kDeg));
    Drop d;
    d.x = x + side * rim * (0.5f + 0.9f * u(k, 3));
    d.y = y - 1.f;
    d.vx = side * sp * std::sin(th);
    d.vy = -sp * std::cos(th);
    d.r = (0.6f + 1.f * u(k, 4)) * (0.6f + 0.5f * std::min(power, 1.5f));
    d.life = 4.f;
    d.g = 700.f;
    d.floor = y;
    d.pool = onto;
    d.source = E + ci;
    d.white = u(k, 5) < 0.55f ? 1 : 0;
    spawn_drop(d);
  }
  if (onto >= 0 && onto < E && layout_[zs(onto)].kind == WaterKind::pool) {  // the crater, its rings and foam
    wave(onto, x, 160.f + 320.f * power, 10.f + 16.f * power);
    add_ring(onto, x, 30.f + 50.f * power, t);
    add_ring(onto, x, 15.f + 25.f * power, t + 0.15f);
    const int foam = 6 + static_cast<int>(10.f * power);
    for (int k = 0; k < foam; ++k) add_foam(onto, x + (u(k, 6) - 0.5f) * 30.f * power, (u(k, 7) - 0.5f) * 140.f * power, 2.f + 4.f * u(k, 8), 2.f + 3.f * u(k, 9));
  }
  if (mist) {  // the burst of spray at once
    const float m = 0.15f * power * mist_amount;
    deposit(*mist, x, y - 8.f - 12.f * power, 10.f + 14.f * power, mist_heat * m, m, 0.f, -1.2f, 0.4f);
    mist_given_ += m;
  }
}

// Splashes: the crown's walls, strands thrown up and out from the crater's rim for its first tenth of a second (sheets
// of water that break into drops from their tops down), then the jet that rises from the crater, a column that breaks
// up at its top.
void Water::step_crowns(float t, float dt, int frame) {
  const int E = elements();
  constexpr int kWallSub = 4;  // wall nodes a frame (back-dated within it): walls are fed for a few frames only
  for (std::size_t ci = 0; ci < crowns_.size(); ++ci) {
    Crown& c = crowns_[ci];
    if (!c.alive) continue;
    const float age = t - c.t0, p = std::min(c.power, 1.5f), rim = 4.f + 10.f * c.power;
    int live = 0;
    for (int j = 0; j < kCrownStrands; ++j) {
      const std::size_t g = ci * kCrownStrands + zs(j);
      Node* ring = crown_nodes_.data() + g * kCrownNodes;
      int& head = crown_head_[g];
      int& count = crown_count_[g];
      for (int q = 0; q < count; ++q) {
        Node& n = ring[(head + q) % kCrownNodes];
        if (!n.alive) continue;
        n.vy += 700.f * dt;
        n.x += n.vx * dt;
        n.y += n.vy * dt;
        n.age += dt;
        if (!n.broken && n.age >= n.life) {
          n.broken = 1;
          n.vx += (rnd(c.seed, frame, j * 97 + (head + q) % kCrownNodes, 505) - 0.5f) * 40.f;
        }
        if (n.vy > 0.f) {
          const float ly = land_y(c.onto, n.x, c.y);
          if (n.y >= ly) {
            land(E + static_cast<int>(ci), c.onto, n.x, ly, n.w, std::hypot(n.vx, n.vy), 2.f, frame,
                 static_cast<std::uint32_t>(5000 + j * kCrownNodes + (head + q) % kCrownNodes));
            n.alive = 0;
          }
        }
      }
      while (count > 0 && !ring[head].alive) {
        head = (head + 1) % kCrownNodes;
        --count;
      }
      live += count;
      const bool wall = j < 2 * kCrownWalls;
      const float t0 = wall ? 0.f : 0.2f, t1 = wall ? 0.12f : 0.45f;
      if (age < t0 || age >= t1) continue;
      const int subs = wall ? kWallSub : 1;
      for (int k = 0; k < subs && count < kCrownNodes; ++k) {
        const auto u = [&](int salt) { return rnd(c.seed, frame, j * 8 + k, salt); };
        const float back = wall ? dt * fl(subs - 1 - k) / fl(subs) : 0.f;  // how long before now it left
        const float a = std::clamp((age - back - t0) / (t1 - t0), 0.f, 1.f);
        Node n;
        n.alive = 1;
        n.born = frame * 8 + k;
        if (wall) {
          const float side = j < kCrownWalls ? -1.f : 1.f;
          const int w = j % kCrownWalls;
          // each wall its own angle and throw for the whole splash (a smooth wall), each node a little about them
          const auto uw = [&](int salt) { return rnd(c.seed, 0, j, salt); };
          const float th = (24.f + 12.f * fl(w) + 6.f * (uw(1) - 0.5f) + 12.f * a + 1.f * (u(1) - 0.5f)) * kDeg;  // from upright, outwards: opens
          const float sp = (150.f + 250.f * p) * (1.f - 0.12f * fl(w)) * (1.f - 0.45f * a) * (0.92f + 0.16f * uw(2)) * (0.99f + 0.02f * u(2));
          n.x = c.x + side * rim * (1.f + 0.3f * fl(w)) * (1.f + 0.5f * a);  // the crater widens as it is fed
          n.y = c.y - 1.f;
          n.vx = side * sp * std::sin(th);
          n.vy = -sp * std::cos(th);
          n.w = (0.35f + 0.25f * p) * (1.f - 0.15f * fl(w));
          n.life = (0.18f + 0.06f * u(3)) * (0.7f + 0.4f * p) + 0.1f * a;  // it breaks from its rim down
        } else {
          n.x = c.x + (u(1) - 0.5f) * 2.f;
          n.y = c.y;
          n.vx = (u(2) - 0.5f) * 20.f;
          n.vy = -(120.f + 200.f * p) * (1.f - 0.6f * a) * (0.92f + 0.08f * u(3));
          n.w = 0.45f + 0.35f * p;
          n.life = -n.vy / 700.f * (0.8f + 0.3f * u(4));
        }
        n.v0 = std::max(30.f, std::hypot(n.vx, n.vy));
        n.vy += 700.f * back;
        n.x += n.vx * back;
        n.y += n.vy * back;
        n.age = back;
        ring[(head + count) % kCrownNodes] = n;
        ++count;
        ++live;
      }
    }
    if (age > 4.f && live == 0) c.alive = 0;
  }
}

// Drops and spray: they fall under their weight, drift with the wind; on a pool they land; on the ground small ones
// bounce once.
void Water::step_drops(float dt, int frame) {
  for (std::size_t q = 0; q < drops_.size(); ++q) {
    Drop& d = drops_[q];
    if (!d.alive) continue;
    d.vy += d.g * dt;
    d.vx += (1.f - std::exp(-0.8f * dt)) * (d.wu - d.vx);
    d.x += d.vx * dt;
    d.y += d.vy * dt;
    d.age += dt;
    if (d.age >= d.life) {
      d.alive = 0;
      continue;
    }
    if (d.vy <= 0.f) continue;
    const float ly = land_y(d.pool, d.x, d.floor);
    if (d.y < ly) continue;
    const bool on_pool = d.pool >= 0 && ly != d.floor;
    if (!on_pool && !d.bounced && d.r < 2.5f) {  // bounces once off the ground
      d.y = ly - 0.1f;
      d.vy *= -0.3f;
      d.vx *= 0.6f;
      d.bounced = 1;
      continue;
    }
    land(d.source, d.pool, d.x, ly, 0.05f * d.r * d.r, std::hypot(d.vx, d.vy), d.r, frame, static_cast<std::uint32_t>(q + 20000));
    d.alive = 0;
  }
}

// What landed this frame, as mist into each source's mist module: a puff per bin along x.
void Water::feed_mist() {
  const int E = elements();
  for (std::size_t si = 0; si < bins_.size(); ++si) {
    Module* m = nullptr;
    float amount = 0.f, heat = 0.f;
    if (static_cast<int>(si) < E) {
      m = layout_[si].mist;
      amount = spec_[si].mist_amount;
      heat = spec_[si].mist_heat;
    } else {
      const Crown& c = crowns_[si - zs(E)];
      if (!c.alive) continue;
      m = c.mist;
      amount = c.mist_amount;
      heat = c.mist_heat;
    }
    if (!m || !m->active || !(amount > 0.f)) continue;
    const Bins& b = bins_[si];
    const float w = (b.hi - b.lo) / fl(kBins);
    for (int k = 0; k < kBins; ++k) {
      const float mass = b.mass[zs(k)];
      if (!(mass > 0.f)) continue;
      const float q = 0.02f * amount * mass;
      deposit(*m, b.lo + (fl(k) + 0.5f) * w, b.y - 6.f, std::max(8.f, 0.6f * w), heat * q, q, 0.f, -0.8f, 0.2f);
      mist_given_ += q;
    }
  }
}

int Water::drops_alive() const {
  int n = 0;
  for (const Drop& d : drops_) n += d.alive;
  return n;
}

float Water::pool_height(int pool, float x) const {
  if (pool < 0 || pool >= elements() || layout_[zs(pool)].kind != WaterKind::pool) return 0.f;
  const WaterSpec& s = spec_[zs(pool)];
  if (x < std::min(s.x0, s.x1) || x > std::max(s.x0, s.x1)) return 0.f;
  const float span = s.x1 - s.x0;
  return s.y0 + (x - s.x0) / span * (s.y1 - s.y0) - surface_y(pool, x);
}

Water::Counts Water::counts() const {
  Counts c;
  for (const Node& n : nodes_) c.nodes += n.alive;
  for (const Node& n : crown_nodes_) c.nodes += n.alive;
  c.drops = drops_alive();
  for (const Foam& f : foam_) c.foam += f.alive;
  for (const Ring& r : rings_) c.rings += r.pool >= 0 && time_ - r.t0 < r.life && time_ >= r.t0;
  for (const Bubble& b : bubbles_) c.bubbles += b.alive;
  for (const Drip& d : drips_) c.drips += d.alive;
  for (const Streak& s : streaks_) c.streaks += s.alive;
  for (const Crown& k : crowns_) c.crowns += k.alive;
  return c;
}

// --- the picture -------------------------------------------------------------------------------------------------------

void Water::reserve(WaterPicture& p, int screen_h) const {
  int prims = 0, pools = 0, cells = 0, foam = 0, shafts = 0;
  for (const WaterLayout& L : layout_) {
    switch (L.kind) {
      case WaterKind::sheet: prims += 2 * L.strands * L.nodes; break;  // its nodes, and the sheet between them
      case WaterKind::jet: prims += L.strands * L.nodes; break;
      case WaterKind::pool:
        ++pools;
        cells += L.cells;
        foam += L.count;
        break;
      case WaterKind::drips: prims += 2 * L.count; break;
      case WaterKind::bubbles: prims += 2 * L.count; break;
      case WaterKind::shaft:
        prims += L.count;
        ++shafts;
        break;
    }
  }
  prims += static_cast<int>(drops_.size() + crown_nodes_.size() + crowns_.size() * kCrownWalls * kCrownNodes);
  p.reserve(prims, pools, cells, foam, static_cast<int>(rings_.size()), shafts, screen_h);
}

namespace {

WaterPicture::Prim capsule(float ax, float ay, float bx, float by, float r, float alpha, const std::array<float, 3>& c, float glint, float sky, int group = -1) {
  WaterPicture::Prim p;
  p.ax = ax;
  p.ay = ay;
  p.bx = bx;
  p.by = by;
  p.r = r;
  p.alpha = alpha;
  p.red = c[0];
  p.green = c[1];
  p.blue = c[2];
  p.glint = glint;
  p.sky = sky;
  p.light = sky;
  p.group = group;
  p.x0 = wfloor(std::min(ax, bx) - r - 1.f);
  p.x1 = wfloor(std::max(ax, bx) + r + 1.f);
  p.y0 = wfloor(std::min(ay, by) - r - 1.f);
  p.y1 = wfloor(std::max(ay, by) + r + 1.f);
  return p;
}

std::array<float, 3> whiter(const std::array<float, 3>& c, float k) { return {c[0] + k * (1.f - c[0]), c[1] + k * (1.f - c[1]), c[2] + k * (1.f - c[2])}; }

// A quad between four corners in order (a piece of a sheet): false when it is not convex or has next to no area.
bool quad(WaterPicture::Prim& p, float ax, float ay, float bx, float by, float cx, float cy, float dx, float dy) {
  const float xs[4] = {ax, bx, cx, dx}, ys[4] = {ay, by, cy, dy};
  float area = 0.f;
  int pos = 0, neg = 0;
  for (int i = 0; i < 4; ++i) {
    const int j = (i + 1) & 3, k = (i + 2) & 3;
    area += xs[i] * ys[j] - xs[j] * ys[i];
    const float turn = (xs[j] - xs[i]) * (ys[k] - ys[j]) - (ys[j] - ys[i]) * (xs[k] - xs[j]);
    pos += turn > 0.f;
    neg += turn < 0.f;
  }
  if (std::abs(area) < 0.5f || (pos > 0 && neg > 0)) return false;
  p.ring = WaterPicture::kQuad;
  p.ax = ax;
  p.ay = ay;
  p.bx = bx;
  p.by = by;
  p.cx = cx;
  p.cy = cy;
  p.dx = dx;
  p.dy = dy;
  p.r = area > 0.f ? 1.f : -1.f;  // its winding
  p.x0 = wfloor(std::min(std::min(ax, bx), std::min(cx, dx)) - 1.f);
  p.x1 = wfloor(std::max(std::max(ax, bx), std::max(cx, dx)) + 1.f);
  p.y0 = wfloor(std::min(std::min(ay, by), std::min(cy, dy)) - 1.f);
  p.y1 = wfloor(std::max(std::max(ay, by), std::max(cy, dy)) + 1.f);
  return true;
}

}  // namespace

void Water::picture(WaterPicture& out, float cam_x, float cam_y, int screen_w, int screen_h) const {
  out.clear(screen_h);
  out.cam_x = cam_x;
  out.cam_y = cam_y;
  const float W = fl(screen_w), H = fl(screen_h), t = time_;
  const auto visible = [&](const WaterPicture::Prim& p) { return p.x1 >= 0 && p.x0 < screen_w && p.y1 >= 0 && p.y0 < screen_h; };
  const auto add = [&](const WaterPicture::Prim& p) {
    if (visible(p) && out.prims_.size() < out.prims_.capacity()) out.prims_.push_back(p);
  };
  const int E = elements();
  // pools: their surfaces, foam and rings
  for (int i = 0; i < E; ++i) {
    const WaterLayout& L = layout_[zs(i)];
    const WaterSpec& s = spec_[zs(i)];
    if (L.kind != WaterKind::pool || !s.on) continue;
    WaterPicture::Pool p;
    p.x0 = std::min(s.x0, s.x1) - cam_x;
    p.x1 = std::max(s.x0, s.x1) - cam_x;
    p.y = 0.5f * (s.y0 + s.y1) - cam_y;
    p.depth = std::max(0.f, s.depth);
    p.surface = std::max(0.f, s.surface);
    p.clarity = clamp01(s.clarity);
    p.brightness = s.brightness;
    p.colour = s.colour;
    p.time = t;
    p.cells = L.cells;
    p.first = static_cast<int>(out.heights_.size());
    float top = 1e9f;
    for (int c = 0; c < L.cells; ++c) {
      const float wx = s.x0 + (fl(c) + 0.5f) / fl(L.cells) * (s.x1 - s.x0);
      const float y = surface_y(i, wx) - cam_y;
      out.heights_.push_back(y);
      top = std::min(top, y);
    }
    if (s.x1 < s.x0) std::reverse(out.heights_.begin() + p.first, out.heights_.end());  // left to right
    p.foam = static_cast<int>(out.foam_.size());
    const Foam* f = foam_.data() + foam0_[zs(i)];
    for (int q = 0; q < L.count; ++q) {
      const Foam& o = f[q];
      if (!o.alive) continue;
      const float a = 0.6f * smoothstep01((o.life - o.age) / 0.8f) * std::min(1.f, o.age * 8.f + 0.3f);
      out.foam_.push_back({o.x - cam_x, o.size, a});
    }
    p.foam_n = static_cast<int>(out.foam_.size()) - p.foam;
    p.ring = static_cast<int>(out.rings_.size());
    for (const Ring& r : rings_) {
      if (r.pool != i || t < r.t0 || t - r.t0 >= r.life) continue;
      const float k = (t - r.t0) / r.life;
      out.rings_.push_back({r.x - cam_x, r.size * (0.15f + 0.85f * std::sqrt(k)), 0.8f * (1.f - k) * (1.f - k)});
    }
    p.ring_n = static_cast<int>(out.rings_.size()) - p.ring;
    p.y0 = std::max(0, wfloor(top - 3.f));
    p.y1 = std::min(screen_h - 1, wfloor(p.y + p.depth + 1.f));
    if (p.y0 <= p.y1 && p.x1 >= 0.f && p.x0 < W) out.pools_.push_back(p);
  }
  // rain shafts: their veils
  for (int i = 0; i < E; ++i) {
    const WaterLayout& L = layout_[zs(i)];
    const WaterSpec& s = spec_[zs(i)];
    if (L.kind != WaterKind::shaft || !s.on) continue;
    WaterPicture::Shaft sh;
    const float V = s.speed > 0.f ? s.speed : 520.f;
    sh.x0 = s.x0 - cam_x;
    sh.y0 = s.y0 - cam_y;
    sh.w = std::max(1.f, s.x1);
    sh.h = std::max(1.f, s.y1);
    sh.slant = shaft_wind(s, L.seed, t) / V;
    sh.veil = 0.3f * (1.f - clamp01(s.clarity)) * std::min(1.f, std::max(0.f, s.density) / 30.f);
    sh.brightness = s.brightness;
    sh.colour = s.colour;
    sh.first = static_cast<int>(out.profile_.size());
    for (int k = 0; k < WaterPicture::kProfile; ++k) out.profile_.push_back(shaft_profile(s, L.seed, (fl(k) + 0.5f) / fl(WaterPicture::kProfile), t));
    if (sh.x0 + sh.w + std::abs(sh.slant) * sh.h >= 0.f && sh.x0 - std::abs(sh.slant) * sh.h < W && sh.y0 < H && sh.y0 + sh.h >= 0.f) out.shafts_.push_back(sh);
  }
  // primitives, in this order: shafts' streaks, strands, crowns' jets, drips, bubbles, drops
  int group = 0;
  for (int i = 0; i < E; ++i) {
    const WaterLayout& L = layout_[zs(i)];
    const WaterSpec& s = spec_[zs(i)];
    if (L.kind != WaterKind::shaft) continue;
    const Streak* p = streaks_.data() + item0_[zs(i)];
    const std::array<float, 3> c = whiter(s.colour, 0.15f);
    for (int q = 0; q < L.count; ++q) {
      const Streak& o = p[q];
      if (!o.alive) continue;
      const float k = o.len / std::max(1.f, o.vy);
      add(capsule(o.x - o.vx * k - cam_x, o.y - o.len - cam_y, o.x - cam_x, o.y - cam_y, 0.55f, o.alpha, c, 0.1f, s.brightness));
    }
  }
  // Strands: the pieces of one (and of a sheet, all its strands and the sheet between them) are one group, drawn as
  // their union; what has broken off them is drawn after, drop by drop.
  const auto newest = [](const Node* ring, int head, int count, int M, int q) -> const Node& { return ring[(head + count - 1 - q) % M]; };
  const auto whole = [](const Node& n) { return n.alive && !n.broken; };
  // a node's radius: `thin`: as it speeds up it thins (what flows through stays the same), as it slows it swells
  const auto radius = [](const Node& n, float base_r, bool thin) {
    const float k = thin ? std::clamp(std::sqrt(n.v0 / std::max(1.f, std::hypot(n.vx, n.vy))), 0.4f, 1.3f) : 1.f;
    return base_r * (0.4f + 0.6f * std::min(1.f, n.w)) * k;
  };
  // a sheet's colour as it falls: glassy at first, whiter as air gets into it
  const auto aerated = [](const std::array<float, 3>& col, float age) { return whiter(col, std::clamp(0.08f + 1.1f * age, 0.f, 0.7f)); };
  const auto strand_line = [&](const Node* ring, int head, int count, int M, float base_r, const std::array<float, 3>& col, float bright, bool thin,
                               bool airy, int grp) {
    for (int q = 0; q < count; ++q) {  // newest first: from the source down the strand
      const Node& n = newest(ring, head, count, M, q);
      if (!whole(n)) continue;
      const float r = radius(n, base_r, thin), a = std::min(0.92f, 0.35f + 0.5f * std::min(1.5f, n.w));
      const std::array<float, 3> c = airy ? aerated(col, n.age) : col;
      // joined to the next older (further down the strand) unless they have drifted apart (a stream torn by its sway)
      // or it has overtaken it (faster water passing slower water: the strand would fold back on itself)
      const Node* o = q + 1 < count ? &newest(ring, head, count, M, q + 1) : nullptr;
      if (o && (!whole(*o) || std::hypot(n.x - o->x, n.y - o->y) > 3.f * std::max(std::hypot(n.vx, n.vy), std::hypot(o->vx, o->vy)) * dt_ + 3.f ||
                (n.x - o->x) * n.vx + (n.y - o->y) * n.vy > 0.f))
        o = nullptr;
      if (o) {
        add(capsule(n.x - cam_x, n.y - cam_y, o->x - cam_x, o->y - cam_y, 0.5f * (r + radius(*o, base_r, thin)), a, c, 0.7f, bright, grp));
      } else {
        add(capsule(n.x - cam_x, n.y - cam_y, n.x - cam_x, n.y - cam_y, r, a, c, 0.7f, bright, grp));
      }
    }
  };
  // `k`: how long a streak each leaves (seconds of its motion): white water falling (long), drops (short)
  const auto broken_bits = [&](const Node* ring, int head, int count, int M, float base_r, const std::array<float, 3>& col, float bright,
                               float k = 0.012f, float alpha = 0.7f) {
    const std::array<float, 3> spray = whiter(col, 0.5f);
    for (int q = 0; q < count; ++q) {
      const Node& n = newest(ring, head, count, M, q);
      if (!n.alive || !n.broken) continue;
      const float r = std::max(0.6f, (0.25f + 0.25f * std::min(1.5f, n.w)) * base_r);
      add(capsule(n.x - n.vx * k - cam_x, n.y - n.vy * k - cam_y, n.x - cam_x, n.y - cam_y, r, alpha, spray, 0.8f, bright));
    }
  };
  // the water between two strands' nodes that left together (a, b: the newer pair; their next older below them), unless
  // they have torn apart
  const auto sheet_between = [&](const Node* ra, int ha, int ca, const Node* rb, int hb, int cb, int M, float tear, float alpha_k,
                                 const std::array<float, 3>& col, bool airy, bool fade, float lift, float bright, int grp) {
    const int n = std::min(ca, cb);
    for (int q = 0; q + 1 < n; ++q) {
      const Node &a0 = newest(ra, ha, ca, M, q), &a1 = newest(ra, ha, ca, M, q + 1);
      const Node &b0 = newest(rb, hb, cb, M, q), &b1 = newest(rb, hb, cb, M, q + 1);
      if (!whole(a0) || !whole(a1) || !whole(b0) || !whole(b1) || a0.born != b0.born || a1.born != b1.born) continue;
      if (std::hypot(a0.x - b0.x, a0.y - b0.y) > tear || std::hypot(a1.x - b1.x, a1.y - b1.y) > tear) continue;
      const float w = 0.25f * (std::min(1.5f, a0.w) + std::min(1.5f, a1.w) + std::min(1.5f, b0.w) + std::min(1.5f, b1.w));
      const float age = 0.25f * (a0.age + a1.age + b0.age + b1.age);
      // it thins to its streaks before it breaks: holes open softly
      const float keep = fade ? 1.f - smoothstep01((std::max(std::max(a0.age / a0.life, a1.age / a1.life), std::max(b0.age / b0.life, b1.age / b1.life)) - 0.35f) / 0.5f) : 1.f;
      if (keep <= 0.f) continue;
      const std::array<float, 3> c = airy ? aerated(col, age) : col;
      const auto put_quad = [&](WaterPicture::Prim& p) {
        p.alpha = keep * alpha_k * (0.55f + 0.45f * std::min(1.f, w));
        p.red = c[0];
        p.green = c[1];
        p.blue = c[2];
        p.glint = 0.6f;
        p.sky = p.light = bright;
        p.ny = lift;
        p.group = grp;
        add(p);
      };
      const float ax = a0.x - cam_x, ay = a0.y - cam_y, bx = a1.x - cam_x, by = a1.y - cam_y;
      const float cx = b1.x - cam_x, cy = b1.y - cam_y, dx = b0.x - cam_x, dy = b0.y - cam_y;
      WaterPicture::Prim p;
      if (quad(p, ax, ay, bx, by, cx, cy, dx, dy)) {
        put_quad(p);
      } else {  // folded: as two triangles
        if (quad(p, ax, ay, bx, by, cx, cy, cx, cy)) put_quad(p);
        if (quad(p, ax, ay, cx, cy, dx, dy, dx, dy)) put_quad(p);
      }
    }
  };
  for (int i = 0; i < E; ++i) {
    const WaterLayout& L = layout_[zs(i)];
    const WaterSpec& s = spec_[zs(i)];
    if (L.kind == WaterKind::jet) {  // each jet a strand
      const float base_r = 1.5f + 2.5f * std::min(1.5f, s.flow);
      for (int st = 0; st < L.strands; ++st) {
        const int g = strand0_[zs(i)] + st;
        const Node* ring = nodes_.data() + node0_[zs(i)] + st * L.nodes;
        strand_line(ring, head_[zs(g)], count_[zs(g)], L.nodes, base_r, s.colour, s.brightness, true, false, group++);
        broken_bits(ring, head_[zs(g)], count_[zs(g)], L.nodes, base_r, s.colour, s.brightness);
      }
    } else if (L.kind == WaterKind::sheet) {  // a sheet: streaks (its strands) in the water between them
      const float gap = std::max(1.f, std::hypot(s.x1 - s.x0, s.y1 - s.y0)) / fl(L.strands);
      const float line_r = 0.3f * gap + 0.4f, drop_r = 0.75f * gap;
      const int grp = group++;
      for (int st = 0; st < L.strands; ++st) {
        const int g = strand0_[zs(i)] + st;
        const Node* ring = nodes_.data() + node0_[zs(i)] + st * L.nodes;
        strand_line(ring, head_[zs(g)], count_[zs(g)], L.nodes, line_r, s.colour, s.brightness, false, true, grp);
        if (st + 1 < L.strands)
          sheet_between(ring, head_[zs(g)], count_[zs(g)], ring + L.nodes, head_[zs(g + 1)], count_[zs(g + 1)], L.nodes, 3.f * gap, 0.55f,
                        s.colour, true, true, 0.15f, s.brightness, grp);
      }
      for (int st = 0; st < L.strands; ++st) {
        const int g = strand0_[zs(i)] + st;
        broken_bits(nodes_.data() + node0_[zs(i)] + st * L.nodes, head_[zs(g)], count_[zs(g)], L.nodes, drop_r, s.colour, s.brightness, 0.03f, 0.5f);
      }
    }
  }
  const std::array<float, 3> clear{0.62f, 0.8f, 0.95f};
  for (std::size_t ci = 0; ci < crowns_.size(); ++ci) {  // splashes: the crown (its walls and the water between them), the jet
    const Crown& c = crowns_[ci];
    if (!c.alive) continue;
    const float p = std::min(1.5f, c.power);
    const std::array<float, 3> col = whiter(c.onto >= 0 ? spec_[zs(c.onto)].colour : clear, 0.3f);
    const std::size_t g0 = ci * kCrownStrands;
    const auto ring_of = [&](int j) { return crown_nodes_.data() + (g0 + zs(j)) * kCrownNodes; };
    const auto head_of = [&](int j) { return crown_head_[g0 + zs(j)]; };
    const auto count_of_ = [&](int j) { return crown_count_[g0 + zs(j)]; };
    const float wall_r = 1.2f + 1.4f * p;
    const int grp = group++;
    for (int j = 0; j < 2 * kCrownWalls; ++j) strand_line(ring_of(j), head_of(j), count_of_(j), kCrownNodes, wall_r, col, 1.f, false, false, grp);
    for (int w = 0; w < kCrownWalls; ++w)  // the crown's near face, between a left wall and its right twin
      sheet_between(ring_of(w), head_of(w), count_of_(w), ring_of(kCrownWalls + w), head_of(kCrownWalls + w), count_of_(kCrownWalls + w),
                    kCrownNodes, 1e9f, 0.13f, whiter(col, 0.4f), false, true, 0.1f, 1.f, grp);
    const int jet = 2 * kCrownWalls;
    strand_line(ring_of(jet), head_of(jet), count_of_(jet), kCrownNodes, 1.2f + 1.6f * p, col, 1.f, true, false, group++);
    for (int j = 0; j < kCrownStrands; ++j) broken_bits(ring_of(j), head_of(j), count_of_(j), kCrownNodes, j == jet ? 1.2f + 1.6f * p : wall_r * 1.5f, col, 1.f);
  }
  for (int i = 0; i < E; ++i) {  // drips forming: a swelling drop hanging from its point
    const WaterLayout& L = layout_[zs(i)];
    const WaterSpec& s = spec_[zs(i)];
    if (L.kind != WaterKind::drips) continue;
    const Drip* d = drips_.data() + item0_[zs(i)];
    for (int q = 0; q < L.count; ++q) {
      const Drip& o = d[q];
      if (!o.alive) continue;
      const float k = std::sqrt(clamp01(o.age / o.form)), r = std::max(0.4f, o.r * k);
      add(capsule(o.x - cam_x, o.y - cam_y, o.x - cam_x, o.y + r - cam_y, 0.45f * r, 0.6f, s.colour, 0.6f, s.brightness));
      add(capsule(o.x - cam_x, o.y + r - cam_y, o.x - cam_x, o.y + r - cam_y, r, 0.7f, s.colour, 1.f, s.brightness));
    }
  }
  for (int i = 0; i < E; ++i) {  // bubbles: a bright rim and a glint
    const WaterLayout& L = layout_[zs(i)];
    const WaterSpec& s = spec_[zs(i)];
    if (L.kind != WaterKind::bubbles) continue;
    const Bubble* b = bubbles_.data() + item0_[zs(i)];
    const std::array<float, 3> rim = whiter(s.colour, 0.6f), white{1.f, 1.f, 1.f};
    for (int q = 0; q < L.count; ++q) {
      const Bubble& o = b[q];
      if (!o.alive) continue;
      const float x = o.x + o.wobble * std::sin(o.phase + 5.f * o.age) - cam_x, y = o.y - cam_y;
      WaterPicture::Prim p = capsule(x, y, x, y, std::max(0.45f, 0.18f * o.r), 0.75f, rim, 0.3f, s.brightness);
      p.ring = WaterPicture::kRing;
      p.bx = o.r;
      p.by = 1.f;
      p.x0 = wfloor(x - o.r - 2.f);
      p.x1 = wfloor(x + o.r + 2.f);
      p.y0 = wfloor(y - o.r - 2.f);
      p.y1 = wfloor(y + o.r + 2.f);
      add(p);
      add(capsule(x - 0.35f * o.r, y - 0.35f * o.r, x - 0.35f * o.r, y - 0.35f * o.r, std::max(0.4f, 0.2f * o.r), 0.8f, white, 0.f, s.brightness));
    }
  }
  for (const Drop& d : drops_) {  // drops and spray, streaked along their motion
    if (!d.alive) continue;
    const std::array<float, 3> c = d.white ? std::array<float, 3>{0.9f, 0.95f, 1.f} : clear;
    const float k = 0.014f, a = 0.8f * std::min(1.f, (d.life - d.age) * 4.f);
    add(capsule(d.x - d.vx * k - cam_x, d.y - d.vy * k - cam_y, d.x - cam_x, d.y - cam_y, std::max(0.5f, 0.75f * d.r), a, c, 1.f, 1.f));
  }
  out.bin();
}

// --- drawing -----------------------------------------------------------------------------------------------------------

void WaterPicture::reserve(int prims, int pools, int cells, int foam, int rings, int shafts, int rows) {
  prims_.reserve(zs(std::max(1, prims)));
  pools_.reserve(zs(std::max(1, pools)));
  heights_.reserve(zs(std::max(1, cells)));
  foam_.reserve(zs(std::max(1, foam)));
  rings_.reserve(zs(std::max(1, rings)));
  shafts_.reserve(zs(std::max(1, shafts)));
  profile_.reserve(zs(std::max(1, shafts)) * kProfile);
  bin_start_.reserve(zs(rows / kBand + 2));
  bin_items_.reserve(zs(std::max(1, prims)) * 8);
  rows_ = rows;
}

void WaterPicture::clear(int rows) {
  prims_.clear();
  pools_.clear();
  heights_.clear();
  foam_.clear();
  rings_.clear();
  shafts_.clear();
  profile_.clear();
  bin_items_.clear();
  rows_ = rows;
}

void WaterPicture::bin() {
  const int bands = rows_ / kBand + 1;
  bin_start_.assign(zs(bands + 1), 0);
  const std::size_t cap = bin_items_.capacity();
  for (const Prim& p : prims_) {
    const int b0 = std::clamp(p.y0 / kBand, 0, bands - 1), b1 = std::clamp(p.y1 / kBand, 0, bands - 1);
    for (int b = b0; b <= b1; ++b) ++bin_start_[zs(b + 1)];
  }
  for (int b = 0; b < bands; ++b) bin_start_[zs(b + 1)] += bin_start_[zs(b)];
  const std::size_t total = std::min(cap, zs(bin_start_[zs(bands)]));
  bin_items_.resize(total);
  // fill (a cursor per band: bin_start_ shifted)
  std::vector<int>& cur = bin_start_;  // fill using the starts, then restore them
  for (int i = 0; i < static_cast<int>(prims_.size()); ++i) {
    const Prim& p = prims_[zs(i)];
    const int b0 = std::clamp(p.y0 / kBand, 0, bands - 1), b1 = std::clamp(p.y1 / kBand, 0, bands - 1);
    for (int b = b0; b <= b1; ++b) {
      const int at = cur[zs(b)]++;
      if (zs(at) < total) bin_items_[zs(at)] = i;
    }
  }
  for (int b = bands; b > 0; --b) cur[zs(b)] = cur[zs(b - 1)];  // each cursor ended at the next band's start
  cur[0] = 0;
}

namespace {

// The light of the scene at a world point (bilinear in the light's grid, as the field shader samples it), and the flash.
std::array<float, 3> light_at(const WaterLight& L, float wx, float wy) {
  if (!L.L4 || L.nx <= 0 || L.ny <= 0) return L.flash;
  const float gx = std::clamp((wx - L.x0) / L.cell - 0.5f, 0.f, fl(L.nx - 1)), gy = std::clamp((wy - L.y0) / L.cell - 0.5f, 0.f, fl(L.ny - 1));
  const int x0 = std::min(wfloor(gx), L.nx - 1), y0 = std::min(wfloor(gy), L.ny - 1);
  const int x1 = std::min(x0 + 1, L.nx - 1), y1 = std::min(y0 + 1, L.ny - 1);
  const float fx = gx - fl(x0), fy = gy - fl(y0);
  std::array<float, 3> o{};
  for (int c = 0; c < 3; ++c) {
    const auto at = [&](int x, int y) { return L.L4[(zs(y) * zs(L.nx) + zs(x)) * 4 + zs(c)]; };
    const float a = at(x0, y0) + fx * (at(x1, y0) - at(x0, y0)), b = at(x0, y1) + fx * (at(x1, y1) - at(x0, y1));
    o[zs(c)] = a + fy * (b - a) + L.flash[zs(c)];
  }
  return o;
}

// One water pixel: its coverage, normal (x right, y up, z towards the viewer: unit) and what it reflects, lit by the night
// sky (from above and to the left, as the field shader's moon) and the scene's light, with a highlight; laid over the
// screen (colour only) and the layers.
struct Px {
  float cov = 0, nx = 0, ny = 0, nz = 1, r = 1, g = 1, b = 1, glint = 0, sky = 1, light = 1;
};

void lay(const Px& p, const std::array<float, 3>& L, float* s, float* C, float* A, float* E, float* N) {
  const float cov = std::min(p.cov, 0.995f);
  if (!(cov >= 1e-3f)) return;
  const float lambert = std::max(0.f, -0.45f * p.nx + 0.6f * p.ny + 0.66f * p.nz);
  const float moon = p.sky * (0.35f + 0.9f * lambert);
  const float lr = p.light * L[0] / (1.f + L[0]), lg = p.light * L[1] / (1.f + L[1]), lb = p.light * L[2] / (1.f + L[2]);
  // the highlight: the half vector between the sky's direction and the viewer's
  constexpr float hx = -0.2556f, hy = 0.3408f, hz = 0.9047f;
  const float nh = std::max(0.f, p.nx * hx + p.ny * hy + p.nz * hz);
  const float n2 = nh * nh, n4 = n2 * n2, n8 = n4 * n4, n16 = n8 * n8;
  const float spec = p.glint * n16 * n8 * (0.8f * p.sky + 0.6f * (lr + lg + lb) * (1.f / 3.f));
  const float cr = cov * (p.r * (0.7f * moon + lr) + spec), cg = cov * (p.g * (0.8f * moon + lg) + spec), cb = cov * (p.b * (1.1f * moon + lb) + spec);
  const float k = 1.f - cov;
  s[0] = s[0] * k + cr;
  s[1] = s[1] * k + cg;
  s[2] = s[2] * k + cb;
  if (!C) return;
  C[0] = C[0] * k + cr;
  C[1] = C[1] * k + cg;
  C[2] = C[2] * k + cb;
  C[3] = 1.f - (1.f - C[3]) * k;
  A[0] = A[0] * k + cov * std::min(1.f, p.r);
  A[1] = A[1] * k + cov * std::min(1.f, p.g);
  A[2] = A[2] * k + cov * std::min(1.f, p.b);
  A[3] = C[3];
  for (int c = 0; c < 3; ++c) E[c] *= k;
  N[0] = N[0] * k + cov * p.nx;
  N[1] = N[1] * k + cov * p.ny;
  N[2] = N[2] * k + cov * p.nz;
  N[3] += -std::log(k);
}

}  // namespace

std::array<int, 2> WaterPicture::draw_row(int y, float* screen, float* colour, float* albedo, float* emission, float* normal, int width,
                                          const WaterLight& light) const {
  std::array<int, 2> span{width, 0};
  const float py = fl(y) + 0.5f, wy = py + cam_y;
  const bool layers = colour != nullptr;
  const auto put = [&](int x, const Px& p, const std::array<float, 3>& L) {
    const std::size_t i = zs(x) * 4;
    lay(p, L, screen + i, layers ? colour + i : nullptr, layers ? albedo + i : nullptr, layers ? emission + i : nullptr, layers ? normal + i : nullptr);
    span[0] = std::min(span[0], x);
    span[1] = std::max(span[1], x + 1);
  };
  // pools: the body under the surface, the surface (seen from above when it has a depth of its own) with its rings, foam
  for (const Pool& p : pools_) {
    if (y < p.y0 || y > p.y1) continue;
    const int xa = std::max(0, wfloor(p.x0)), xb = std::min(width - 1, wfloor(p.x1));
    const float span_x = std::max(1e-3f, p.x1 - p.x0), bottom = p.y + p.depth;
    const float* hs = heights_.data() + p.first;
    std::array<float, 3> L = light_at(light, cam_x + 0.5f * (p.x0 + p.x1), wy);
    for (int x = xa; x <= xb; ++x) {
      const float px = fl(x) + 0.5f;
      if ((x & 15) == 0) L = light_at(light, px + cam_x, wy);
      const float c = (px - p.x0) / span_x * fl(p.cells) - 0.5f;
      const int c0 = std::clamp(wfloor(c), 0, p.cells - 1), c1 = std::min(c0 + 1, p.cells - 1);
      const float f = clamp01(c - fl(c0));
      const float top = hs[c0] + f * (hs[c1] - hs[c0]);
      const float slope = (hs[c1] - hs[std::max(0, c0 - 1)]) / std::max(1.f, 2.f * span_x / fl(p.cells));  // screen y per x (down positive)
      if (py < top - 1.f || py > bottom + 0.5f) continue;
      const float edge = clamp01(py - top + 0.5f) * clamp01(bottom + 0.5f - py);
      Px q;
      q.sky = q.light = p.brightness;
      const float band = top + p.surface;
      if (py < band) {  // the surface seen from above: dark, mirroring the night sky (more of it far off), glinting
        const float n = std::sqrt(slope * slope * 4.f + 1.f + 0.12f);
        const float near = clamp01((py - top) / std::max(1.f, p.surface)), mirror = 0.2f - 0.13f * near;
        q.nx = slope * 2.f / n;
        q.ny = 1.f / n;
        q.nz = std::sqrt(0.12f) / n;
        // the sheen: broken streaks of the sky's light along the surface, shimmering as the waves pass and drifting with
        // its current
        const float wx = px + cam_x;
        const float sheen = clamp01((value_noise(0.06f * wx - 0.4f * p.time, 0.45f * (py - top), 0.7f * p.time, 91) - 0.3f) * 2.5f) *
                            (1.f - 0.6f * near) * (0.6f + 4.f * std::abs(slope));
        const float lit = mirror + std::min(0.6f, 0.45f * sheen);
        q.r = p.colour[0] * lit;
        q.g = p.colour[1] * lit;
        q.b = p.colour[2] * lit;
        q.cov = edge * (0.75f + 0.2f * (1.f - p.clarity));
        q.glint = 1.2f;
      } else {  // the body (seen through its side): darker and denser deeper down
        const float d = clamp01((py - band) / std::max(1.f, bottom - band));
        const float n = std::sqrt(slope * slope * 0.09f + 1.f);
        q.nx = 0.3f * slope / n;
        q.ny = 0.f;
        q.nz = 1.f / n;
        q.r = p.colour[0] * 0.12f * (1.f - 0.6f * d);
        q.g = p.colour[1] * 0.12f * (1.f - 0.5f * d);
        q.b = p.colour[2] * 0.12f * (1.f - 0.4f * d);
        q.cov = edge * (0.2f + (1.f - p.clarity) * (0.5f + 0.3f * d)) * smoothstep01((bottom - py) / 6.f);
        q.glint = 0.2f;
      }
      // the bright line where the surface meets the eye: crests that face the sky catch more
      const float line = clamp01(1.f - std::abs(py - top - 0.5f));
      if (line > 0.f) {
        q.glint += line * (0.8f + 2.5f * std::max(0.f, -slope));
        q.cov = std::max(q.cov, 0.7f * line);
      }
      put(x, q, L);
    }
    // rings on the surface seen from above: ellipses as flat as the surface is deep
    if (p.surface > 0.f) {
      for (int k = 0; k < p.ring_n; ++k) {
        const auto& r = rings_[zs(p.ring + k)];
        const float c = (r[0] - p.x0) / span_x * fl(p.cells) - 0.5f;
        const int c0 = std::clamp(wfloor(c), 0, p.cells - 1);
        const float cy = hs[c0] + 0.5f * p.surface, ry = std::min(0.5f * p.surface, 0.28f * r[1]);
        if (std::abs(py - cy) > ry + 1.5f) continue;
        const float ex = r[1], sq = ry / std::max(1e-3f, ex);
        const int xa2 = std::max(0, wfloor(r[0] - ex - 2.f)), xb2 = std::min(width - 1, wfloor(r[0] + ex + 2.f));
        for (int x = xa2; x <= xb2; ++x) {
          const float dx = fl(x) + 0.5f - r[0], dy = py - cy;
          const float e = std::sqrt(dx * dx + (dy / sq) * (dy / sq));
          const float g = std::sqrt(dx * dx + (dy / (sq * sq)) * (dy / (sq * sq))) / std::max(1e-3f, e);
          const float d = std::abs(e - ex) / std::max(1e-3f, g);
          const float cv = r[2] * clamp01(1.f - d);
          if (cv <= 0.f) continue;
          Px q;
          q.cov = cv;
          q.ny = 0.6f;
          q.nz = 0.8f;
          q.r = 0.7f + 0.3f * p.colour[0];
          q.g = 0.7f + 0.3f * p.colour[1];
          q.b = 0.7f + 0.3f * p.colour[2];
          q.glint = 1.f;
          q.sky = q.light = p.brightness;
          put(x, q, L);
        }
      }
    }
    for (int k = 0; k < p.foam_n; ++k) {  // foam on the surface
      const auto& f = foam_[zs(p.foam + k)];
      const float c = (f[0] - p.x0) / span_x * fl(p.cells) - 0.5f;
      const int c0 = std::clamp(wfloor(c), 0, p.cells - 1);
      const float cy = hs[c0] + 0.1f * f[1] + 0.3f * p.surface, ry = std::max(0.7f, 0.18f * f[1] + 0.04f * p.surface);
      if (std::abs(py - cy) > ry + 1.f) continue;
      const int xa2 = std::max(0, wfloor(f[0] - f[1] - 1.f)), xb2 = std::min(width - 1, wfloor(f[0] + f[1] + 1.f));
      for (int x = xa2; x <= xb2; ++x) {
        const float dx = (fl(x) + 0.5f - f[0]) / f[1], dy = (py - cy) / ry;
        const float e = dx * dx + dy * dy;
        if (e >= 1.f) continue;
        Px q;
        q.cov = f[2] * smoothstep01((1.f - e) * 2.f) * (0.55f + 0.45f * value_noise(0.45f * (fl(x) + cam_x), 1.7f * fl(k), 0.5f, 77));  // patchy
        q.nx = 0.4f * dx;
        q.ny = 0.6f;
        q.nz = std::sqrt(std::max(0.f, 1.f - q.nx * q.nx - 0.36f));
        q.r = q.g = q.b = 0.97f;
        q.glint = 0.2f;
        q.sky = q.light = p.brightness;
        put(x, q, L);
      }
    }
  }
  // rain shafts: a veil as dense as the profile, slanted by the wind
  for (const Shaft& sh : shafts_) {
    if (py < sh.y0 || py >= sh.y0 + sh.h) continue;
    const float dy = py - sh.y0, shift = sh.slant * dy, fade = smoothstep01(dy / (0.3f * sh.h));
    const int xa = std::max(0, wfloor(sh.x0 + shift)), xb = std::min(width - 1, wfloor(sh.x0 + shift + sh.w));
    const std::array<float, 3> L = light_at(light, cam_x + sh.x0 + shift + 0.5f * sh.w, wy);
    for (int x = xa; x <= xb; ++x) {
      const float u = (fl(x) + 0.5f - sh.x0 - shift) / sh.w * fl(kProfile) - 0.5f;
      const int k0 = std::clamp(wfloor(u), 0, kProfile - 1), k1 = std::min(k0 + 1, kProfile - 1);
      const float f = clamp01(u - fl(k0));
      const float d = profile_[zs(sh.first + k0)] + f * (profile_[zs(sh.first + k1)] - profile_[zs(sh.first + k0)]);
      Px q;
      q.cov = sh.veil * d * fade;
      q.r = sh.colour[0];
      q.g = sh.colour[1];
      q.b = sh.colour[2];
      q.sky = q.light = sh.brightness;
      put(x, q, L);
    }
  }
  // primitives of this row's band, in order; the pieces of one strand as one (their union)
  if (prims_.empty() || bin_start_.empty()) return span;
  const int band = std::clamp(y / kBand, 0, static_cast<int>(bin_start_.size()) - 2);
  const int i0 = bin_start_[zs(band)], i1 = std::min(bin_start_[zs(band + 1)], static_cast<int>(bin_items_.size()));
  constexpr int kSpan = 192;
  std::array<float, kSpan> best{};
  std::array<Px, kSpan> who{};
  const auto eval = [&](const Prim& p, float px, Px& q) {  // coverage and normal of p at pixel centre (px, py)
    float cov, vx, vy;
    if (p.ring == kQuad) {  // flat, with its own normal: the inside distance to its nearest edge
      const float xs[4] = {p.ax, p.bx, p.cx, p.dx}, ys[4] = {p.ay, p.by, p.cy, p.dy};
      float d = 1e9f;
      for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) & 3;
        const float ex = xs[j] - xs[i], ey = ys[j] - ys[i], l = std::sqrt(ex * ex + ey * ey);
        if (l > 1e-4f) d = std::min(d, p.r * (ex * (py - ys[i]) - ey * (px - xs[i])) / l);
      }
      q.cov = p.alpha * clamp01(d + 1.f);  // half a pixel fuller: neighbouring quads meet without a seam
      q.nx = p.nx;
      q.ny = p.ny;
      q.nz = std::sqrt(std::max(0.f, 1.f - p.nx * p.nx - p.ny * p.ny));
      q.r = p.red;
      q.g = p.green;
      q.b = p.blue;
      q.glint = p.glint;
      q.sky = p.sky;
      q.light = p.light;
      return;
    }
    if (p.ring == kRing) {
      const float dx = px - p.ax, dy = py - p.ay, sq = std::max(1e-3f, p.by);
      const float e = std::sqrt(dx * dx + (dy / sq) * (dy / sq));
      cov = p.alpha * clamp01(p.r + 0.5f - std::abs(e - p.bx));
      const float inv = 1.f / std::max(1e-3f, p.bx);
      vx = std::clamp(dx * inv, -1.f, 1.f);
      vy = std::clamp(dy * inv, -1.f, 1.f);
    } else {
      const float dx = p.bx - p.ax, dy = p.by - p.ay, l2 = dx * dx + dy * dy;
      const float t = l2 > 0.f ? clamp01(((px - p.ax) * dx + (py - p.ay) * dy) / l2) : 0.f;
      const float ex = px - (p.ax + t * dx), ey = py - (p.ay + t * dy);
      const float d = std::sqrt(ex * ex + ey * ey);
      cov = p.alpha * clamp01(p.r + 0.5f - d);
      const float inv = 1.f / std::max(0.5f, p.r);
      vx = std::clamp(ex * inv, -1.f, 1.f);
      vy = std::clamp(ey * inv, -1.f, 1.f);
    }
    const float l2 = vx * vx + vy * vy;
    if (l2 > 1.f) {
      const float s = 1.f / std::sqrt(l2);
      vx *= s;
      vy *= s;
    }
    q.cov = cov;
    q.nx = vx;
    q.ny = -vy;  // screen y points down; the normal's points up
    q.nz = std::sqrt(std::max(0.f, 1.f - vx * vx - vy * vy));
    q.r = p.red;
    q.g = p.green;
    q.b = p.blue;
    q.glint = p.glint;
    q.sky = p.sky;
    q.light = p.light;
  };
  for (int it = i0; it < i1;) {
    const Prim& p0 = prims_[zs(bin_items_[zs(it)])];
    if (p0.group < 0) {  // alone
      ++it;
      if (y < p0.y0 || y > p0.y1) continue;
      const int xa = std::max(0, p0.x0), xb = std::min(width - 1, p0.x1);
      if (xa > xb) continue;
      const std::array<float, 3> L = light_at(light, cam_x + 0.5f * (p0.ax + p0.bx), cam_y + 0.5f * (p0.ay + p0.by));
      for (int x = xa; x <= xb; ++x) {
        Px q;
        eval(p0, fl(x) + 0.5f, q);
        if (q.cov > 0.f) put(x, q, L);
      }
      continue;
    }
    // a strand: its pieces in this band, the most covering at each pixel
    int end = it;
    int xa = width, xb = -1;
    while (end < i1 && prims_[zs(bin_items_[zs(end)])].group == p0.group) {
      const Prim& p = prims_[zs(bin_items_[zs(end)])];
      if (y >= p.y0 && y <= p.y1) {
        xa = std::min(xa, std::max(0, p.x0));
        xb = std::max(xb, std::min(width - 1, p.x1));
      }
      ++end;
    }
    for (int c0 = xa; c0 <= xb; c0 += kSpan) {
      const int c1 = std::min(xb, c0 + kSpan - 1);
      std::fill(best.begin(), best.begin() + (c1 - c0 + 1), 0.f);
      for (int k = it; k < end; ++k) {
        const Prim& p = prims_[zs(bin_items_[zs(k)])];
        if (y < p.y0 || y > p.y1) continue;
        const int pa = std::max(c0, p.x0), pb = std::min(c1, p.x1);
        for (int x = pa; x <= pb; ++x) {
          Px q;
          eval(p, fl(x) + 0.5f, q);
          if (q.cov > best[zs(x - c0)]) {
            best[zs(x - c0)] = q.cov;
            who[zs(x - c0)] = q;
          }
        }
      }
      const std::array<float, 3> L = light_at(light, cam_x + 0.5f * fl(c0 + c1), wy);
      for (int x = c0; x <= c1; ++x)
        if (best[zs(x - c0)] > 0.f) put(x, who[zs(x - c0)], L);
    }
    it = end;
  }
  return span;
}

}  // namespace nfx::compose
