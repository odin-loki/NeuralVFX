#include <neuralfx/noise.hpp>
#include <neuralfx/sim.hpp>

#include <algorithm>
#include <cmath>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace nfx::sim {

std::string_view effect_name(Effect e) {
  switch (e) {
    case Effect::fire: return "fire";
    case Effect::smoke: return "smoke";
    case Effect::explosion: return "explosion";
    case Effect::steam: return "steam";
    case Effect::magic: return "magic";
  }
  std::unreachable();
}

bool parse_effect(std::string_view text, Effect& out) {
  const auto it = std::ranges::find(kAllEffects, text, effect_name);
  if (it == kAllEffects.end()) return false;
  out = *it;
  return true;
}

bool effect_loops(Effect e) { return e != Effect::explosion; }

std::array<std::string_view, kControls> control_names(Effect e) {
  if (e == Effect::magic) return {"intensity", "spin", "turbulence"};
  return kControlNames;
}

float flicker_rate(Effect e) {
  switch (e) {
    case Effect::fire: return 2.6f;
    case Effect::smoke: return 1.4f;
    case Effect::explosion: return 1.4f;
    case Effect::steam: return 2.0f;   // puffs from a vent
    case Effect::magic: return 1.8f;   // a shimmer
  }
  std::unreachable();
}

float Field::sample(float x, float y) const {
  const float lim = static_cast<float>(n_) + 0.5f;
  x = std::clamp(x, 0.5f, lim);
  y = std::clamp(y, 0.5f, lim);
  const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
  const float fx = x - static_cast<float>(x0), fy = y - static_cast<float>(y0);
  const Field& f = *this;
  const float a = f[x0, y0] + fx * (f[x0 + 1, y0] - f[x0, y0]);
  const float b = f[x0, y0 + 1] + fx * (f[x0 + 1, y0 + 1] - f[x0, y0 + 1]);
  return a + fy * (b - a);
}

namespace {

float clamp01(float x) { return std::clamp(x, 0.f, 1.f); }
float smoothstep(float a, float b, float x) {
  const float t = clamp01((x - a) / (b - a));
  return t * t * (3.f - 2.f * t);
}
float f(int i) { return static_cast<float>(i); }

// Per-effect constants, in solver cells and seconds for a 128-cell grid (scaled by n / 128 where they are lengths).
struct Tune {
  float buoyancy, soot_weight, cooling, soot_decay, wind_speed, wind_drag, confinement, curl_force, vel_damp;
};

Tune tune(const Params& p) {
  const float turb = p.turbulence;
  switch (p.effect) {
    case Effect::fire: return {150.f, 8.f, 1.25f, 0.7f, 90.f, 2.5f, 0.6f + 3.5f * turb, 10.f + 60.f * turb, 0.15f};
    case Effect::smoke: return {150.f, 0.6f, 0.30f, 0.30f, 70.f, 2.0f, 0.4f + 2.5f * turb, 6.f + 40.f * turb, 0.10f};
    case Effect::explosion: return {110.f, 3.f, 0.75f, 0.30f, 60.f, 1.5f, 0.6f + 3.0f * turb, 8.f + 50.f * turb, 0.9f};
    // vapour: a fast jet that slows, little buoyancy, gone within a second or two, bent by the wind
    case Effect::steam: return {70.f, 0.2f, 0.9f, 0.65f, 75.f, 2.5f, 0.4f + 2.6f * turb, 10.f + 55.f * turb, 0.35f};
    // no buoyancy and no wind: the spin force (add_forces) and a sink in the middle (project) drive it
    case Effect::magic: return {0.f, 0.f, 1.3f, 1.0f, 0.f, 0.f, 1.0f + 4.0f * turb, 10.f + 70.f * turb, 1.2f};
  }
  std::unreachable();
}

// magic: a ring of energy around the middle of the frame, spun by a force that peaks at the ring (radius in cells).
float magic_radius(const Params& p, float n) { return (0.19f + 0.09f * p.intensity) * n; }
float magic_spin_speed(const Params& p, float s) { return (40.f + 120.f * p.wind) * s; }  // cells per second at the ring

}  // namespace

float curl_potential(const Params& p, float X, float Y, float time) {
  return value_noise(X / 14.f, Y / 14.f, time * 0.8f, p.seed * 0x2545F4914F6CDD1DULL + 77);  // as in add_forces
}

float source_flicker(const Params& p, float X, float Y, float time) {
  return fbm(X * 0.10f, Y * 0.10f, time * flicker_rate(p.effect), p.seed, 3);  // as in add_sources
}

State Fluid::state() const {
  State s;
  s.n = n_;
  s.frame = frame_;
  s.time = time_;
  const auto inside = [this](const Field& q) {
    std::vector<float> o(static_cast<std::size_t>(n_) * n_);
    for (int y = 1; y <= n_; ++y) {
      for (int x = 1; x <= n_; ++x) o[static_cast<std::size_t>(y - 1) * n_ + (x - 1)] = q[x, y];
    }
    return o;
  };
  s.u = inside(u_);
  s.v = inside(v_);
  s.temp = inside(temp_);
  s.soot = inside(soot_);
  s.pressure = inside(pressure_);
  return s;
}

void Fluid::set_state(const State& s) {
  const auto nn = static_cast<std::size_t>(n_) * n_;
  if (s.n != n_ || s.u.size() != nn || s.v.size() != nn || s.temp.size() != nn || s.soot.size() != nn || s.pressure.size() != nn) {
    throw std::invalid_argument("sim: state does not match the solver resolution");
  }
  const auto put = [this](Field& q, const std::vector<float>& from) {
    for (int y = 1; y <= n_; ++y) {
      for (int x = 1; x <= n_; ++x) q[x, y] = from[static_cast<std::size_t>(y - 1) * n_ + (x - 1)];
    }
  };
  put(u_, s.u);
  put(v_, s.v);
  put(temp_, s.temp);
  put(soot_, s.soot);
  put(pressure_, s.pressure);
  velocity_border();
  frame_ = s.frame;
  time_ = s.time;
}

void Fluid::velocity_border() {  // as project() leaves it: the neighbour's value
  for (int k = 1; k <= n_; ++k) {
    for (Field* q : {&u_, &v_}) {
      (*q)[0, k] = (*q)[1, k];
      (*q)[n_ + 1, k] = (*q)[n_, k];
      (*q)[k, 0] = (*q)[k, 1];
      (*q)[k, n_ + 1] = (*q)[k, n_];
    }
  }
}

void Fluid::push(std::span<const float> du, std::span<const float> dv) {
  const auto nn = static_cast<std::size_t>(n_) * n_;
  if (du.size() != nn || dv.size() != nn) throw std::invalid_argument("sim: push field does not match the solver resolution");
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) {
      const std::size_t i = static_cast<std::size_t>(y - 1) * n_ + static_cast<std::size_t>(x - 1);
      u_[x, y] += du[i];
      v_[x, y] += dv[i];
    }
  }
  velocity_border();
}

void Fluid::add_material(std::span<const float> dtemp, std::span<const float> dsoot) {
  const auto nn = static_cast<std::size_t>(n_) * n_;
  if (dtemp.size() != nn || dsoot.size() != nn) throw std::invalid_argument("sim: material field does not match the solver resolution");
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) {
      const std::size_t i = static_cast<std::size_t>(y - 1) * n_ + static_cast<std::size_t>(x - 1);
      temp_[x, y] = std::max(0.f, temp_[x, y] + dtemp[i]);
      soot_[x, y] = std::max(0.f, soot_[x, y] + dsoot[i]);
    }
  }
}

Fluid::Fluid(const Params& p) : p_(p), n_(p.sim_res > 0 ? p.sim_res : p.size) {
  if (p.size < 8 || p.size > 1024 || n_ < 8 || n_ > 1024) throw std::invalid_argument("sim: size out of range");
  if (p.substeps < 1 || p.fps <= 0.f) throw std::invalid_argument("sim: bad substeps or fps");
  for (Field* fld : {&u_, &v_, &temp_, &soot_, &tmp_a_, &tmp_b_, &tmp_c_, &pressure_, &div_, &curl_}) *fld = Field(n_);
}

void Fluid::step_frame() {
  const float dt = 1.f / (p_.fps * f(p_.substeps));
  for (int s = 0; s < p_.substeps; ++s) step(dt);
  ++frame_;
}

void Fluid::step(float dt) {
  add_sources(dt);
  add_forces(dt);
  project();
  advect_velocity(dt);
  project();
  advect_scalar(temp_, dt);
  advect_scalar(soot_, dt);
  const Tune t = tune(p_);
  const float cool = std::exp(-t.cooling * dt), decay = std::exp(-t.soot_decay * dt);
  std::ranges::for_each(temp_.values(), [cool](float& x) { x *= cool; });
  std::ranges::for_each(soot_.values(), [decay](float& x) { x *= decay; });
  time_ += dt;
}

void Fluid::add_sources(float dt) {
  const float n = f(n_), s = n / 128.f, I = p_.intensity;
  const std::uint64_t seed = p_.seed;
  if (p_.effect == Effect::explosion) {
    if (time_ > 0.f) return;  // one burst at the start
    const float cx = 0.5f * n, cy = 0.32f * n, r = (0.10f + 0.07f * I) * n;
    for (int y = 1; y <= n_; ++y) {
      for (int x = 1; x <= n_; ++x) {
        const float dx = f(x) - cx, dy = f(y) - cy;
        const float d = std::sqrt(dx * dx + dy * dy);
        const float jag = 0.25f * fbm(f(x) * 0.09f / s, f(y) * 0.09f / s, 0.f, seed, 3);
        const float fall = 1.f - smoothstep(0.5f * r, r * (1.f + jag), d);
        if (fall <= 0.f) continue;
        const float m = 0.6f + 0.4f * fbm(f(x) * 0.15f / s, f(y) * 0.15f / s, 1.f, seed + 3, 3);
        temp_[x, y] = std::max(temp_[x, y], (2.2f + 0.6f * I) * fall * m);
        soot_[x, y] = std::max(soot_[x, y], (1.2f + 0.6f * I) * fall * m);
        v_[x, y] += 20.f * s * fall;  // the expansion itself comes from the divergence source in project()
      }
    }
    return;
  }
  if (p_.effect == Effect::steam || p_.effect == Effect::magic) {
    add_new_sources(dt);
    return;
  }
  const bool fire = p_.effect == Effect::fire;
  const float cx = 0.5f * n, cy = (fire ? 0.10f : 0.09f) * n;
  const float hw = (fire ? 0.10f + 0.08f * I : 0.09f + 0.07f * I) * n, hh = 0.035f * n;
  const float tt = time_ * (fire ? 2.6f : 1.4f);
  const int y_lo = std::max(1, static_cast<int>(cy - 2.f * hh)), y_hi = std::min(n_, static_cast<int>(cy + 2.f * hh) + 1);
  const int x_lo = std::max(1, static_cast<int>(cx - 1.5f * hw)), x_hi = std::min(n_, static_cast<int>(cx + 1.5f * hw) + 1);
  for (int y = y_lo; y <= y_hi; ++y) {
    for (int x = x_lo; x <= x_hi; ++x) {
      const float ex = (f(x) - cx) / hw, ey = (f(y) - cy) / hh;
      const float shape = 1.f - smoothstep(0.6f, 1.f, std::sqrt(ex * ex + ey * ey));
      if (shape <= 0.f) continue;
      const float k = shape * smoothstep(-0.25f, 0.6f, fbm(f(x) * 0.10f / s, f(y) * 0.10f / s, tt, seed, 3));
      if (fire) {
        temp_[x, y] = std::max(temp_[x, y], (0.95f + 0.55f * I) * k);
        soot_[x, y] += dt * (0.35f + 0.25f * I) * k;
        v_[x, y] = std::max(v_[x, y], (22.f + 14.f * I) * s * k);
      } else {
        temp_[x, y] = std::max(temp_[x, y], (0.55f + 0.30f * I) * k);
        soot_[x, y] = std::max(soot_[x, y], (0.7f + 0.5f * I) * k);
        v_[x, y] = std::max(v_[x, y], (45.f + 15.f * I) * s * k);
      }
    }
  }
}

// steam: a narrow vent at the bottom blowing a jet of vapour. magic: a ring of energy (heat) and a little dark mist
// (soot) around the middle. Both are broken up by the flicker noise, as the fire's source is.
void Fluid::add_new_sources(float dt) {
  (void)dt;
  const float n = f(n_), s = n / 128.f, I = p_.intensity;
  const float tt = time_ * flicker_rate(p_.effect);
  const auto flicker = [&](int x, int y) { return smoothstep(-0.25f, 0.6f, fbm(f(x) * 0.10f / s, f(y) * 0.10f / s, tt, p_.seed, 3)); };
  if (p_.effect == Effect::steam) {
    const float cx = 0.5f * n, cy = 0.06f * n, hw = (0.045f + 0.035f * I) * n, hh = 0.03f * n;
    const int y_lo = std::max(1, static_cast<int>(cy - 2.f * hh)), y_hi = std::min(n_, static_cast<int>(cy + 2.f * hh) + 1);
    const int x_lo = std::max(1, static_cast<int>(cx - 1.5f * hw)), x_hi = std::min(n_, static_cast<int>(cx + 1.5f * hw) + 1);
    for (int y = y_lo; y <= y_hi; ++y) {
      for (int x = x_lo; x <= x_hi; ++x) {
        const float ex = (f(x) - cx) / hw, ey = (f(y) - cy) / hh;
        const float shape = 1.f - smoothstep(0.6f, 1.f, std::sqrt(ex * ex + ey * ey));
        if (shape <= 0.f) continue;
        const float k = shape * flicker(x, y);
        temp_[x, y] = std::max(temp_[x, y], (0.35f + 0.25f * I) * k);
        soot_[x, y] = std::max(soot_[x, y], (1.2f + 0.8f * I) * k);
        v_[x, y] = std::max(v_[x, y], (85.f + 75.f * I) * s * k);
      }
    }
    return;
  }
  const float c = 0.5f * (n + 1.f), R = magic_radius(p_, n), w = 0.035f * n;
  const int lo = std::max(1, static_cast<int>(c - R - 2.f * w)), hi = std::min(n_, static_cast<int>(c + R + 2.f * w) + 1);
  for (int y = lo; y <= hi; ++y) {
    for (int x = lo; x <= hi; ++x) {
      const float dx = f(x) - c, dy = f(y) - c;
      const float shape = 1.f - smoothstep(0.5f, 1.f, std::abs(std::sqrt(dx * dx + dy * dy) - R) / w);
      if (shape <= 0.f) continue;
      const float k = shape * flicker(x, y);
      temp_[x, y] = std::max(temp_[x, y], (0.8f + 0.5f * I) * k);
      soot_[x, y] = std::max(soot_[x, y], (0.25f + 0.15f * I) * k);
    }
  }
}

void Fluid::add_forces(float dt) {
  const Tune t = tune(p_);
  const float s = f(n_) / 128.f;
  const float wind = (2.f * p_.wind - 1.f) * t.wind_speed * s;
  const float wdrag = 1.f - std::exp(-t.wind_drag * dt), damp = std::exp(-t.vel_damp * dt);
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) curl_[x, y] = 0.5f * ((v_[x + 1, y] - v_[x - 1, y]) - (u_[x, y + 1] - u_[x, y - 1]));
  }
  const float L = 14.f * s;  // curl-noise length scale in cells
  const float amp = t.curl_force * s;
  const float tt = time_ * 0.8f;
  const std::uint64_t cseed = p_.seed * 0x2545F4914F6CDD1DULL + 77;
  const bool spin = p_.effect == Effect::magic;
  const float sc = 0.5f * (f(n_) + 1.f), sR = magic_radius(p_, f(n_)), sU = magic_spin_speed(p_, s);
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) {
      float fx = 0.f;
      float fy = t.buoyancy * s * temp_[x, y] - t.soot_weight * s * soot_[x, y];  // hot rises, soot sinks
      if (x > 1 && x < n_ && y > 1 && y < n_) {                                     // vorticity confinement
        const float gx = 0.5f * (std::fabs(curl_[x + 1, y]) - std::fabs(curl_[x - 1, y]));
        const float gy = 0.5f * (std::fabs(curl_[x, y + 1]) - std::fabs(curl_[x, y - 1]));
        const float len = std::sqrt(gx * gx + gy * gy) + 1e-5f;
        fx += t.confinement * (gy / len) * curl_[x, y];
        fy -= t.confinement * (gx / len) * curl_[x, y];
      }
      // curl noise (divergence-free), only where there is material to stir
      const float mat = std::min(1.f, 2.f * (temp_[x, y] + soot_[x, y]));
      if (mat > 0.01f) {
        const float px = f(x) / L, py = f(y) / L, e = 0.25f;
        const float dpsi_dy = (value_noise(px, py + e, tt, cseed) - value_noise(px, py - e, tt, cseed)) / (2.f * e);
        const float dpsi_dx = (value_noise(px + e, py, tt, cseed) - value_noise(px - e, py, tt, cseed)) / (2.f * e);
        fx += mat * amp * dpsi_dy;
        fy -= mat * amp * dpsi_dx;
      }
      if (spin) {  // magic: a swirl around the middle, strongest at the ring; the damping sets its speed
        const float dx = f(x) - sc, dy = f(y) - sc, r = std::sqrt(dx * dx + dy * dy) / sR;
        const float a = t.vel_damp * sU * std::exp(0.5f * (1.f - r * r)) / sR;  // (r / R) exp(...) along the tangent
        fx -= a * dy;
        fy += a * dx;
      }
      u_[x, y] = (u_[x, y] + dt * fx) * damp;
      v_[x, y] = (v_[x, y] + dt * fy) * damp;
      u_[x, y] += wdrag * (wind - u_[x, y]) * mat;  // air drags material towards the wind speed
    }
  }
}

void Fluid::project() {
  const int n = n_;
  for (int y = 1; y <= n; ++y) {
    for (int x = 1; x <= n; ++x) div_[x, y] = -0.5f * (u_[x + 1, y] - u_[x - 1, y] + v_[x, y + 1] - v_[x, y - 1]);
  }
  // Combustion: hot gas of an explosion expands. The solve then targets a divergence of `rate` (1/s) there instead of
  // zero, which is what pushes the fireball outwards (an injected radial velocity would be projected away).
  if (p_.effect == Effect::explosion) {
    const float rate = (10.f + 10.f * p_.intensity) * std::exp(-time_ / 0.15f);
    if (rate > 0.05f) {
      for (int y = 1; y <= n; ++y) {
        for (int x = 1; x <= n; ++x) div_[x, y] += rate * smoothstep(0.4f, 1.2f, temp_[x, y]);
      }
    }
  }
  // magic: a sink in the middle draws the swirl inwards, so the ring's arms spiral into the centre.
  if (p_.effect == Effect::magic) {
    const float c = 0.5f * (f(n) + 1.f), a = 0.07f * f(n);
    for (int y = 1; y <= n; ++y) {
      for (int x = 1; x <= n; ++x) {
        const float dx = f(x) - c, dy = f(y) - c;
        div_[x, y] -= 25.f * (1.f - smoothstep(0.3f * a, a, std::sqrt(dx * dx + dy * dy)));
      }
    }
  }
  // Open boundaries: p = 0 on the border, which is never written. Warm-started from the last solve.
  constexpr float w = 1.7f;
  for (int it = 0; it < p_.pressure_iters; ++it) {
    for (const int colour : {0, 1}) {
      for (int y = 1; y <= n; ++y) {
        for (int x = 1 + ((y + colour) & 1); x <= n; x += 2) {
          const float g = 0.25f * (div_[x, y] + pressure_[x - 1, y] + pressure_[x + 1, y] + pressure_[x, y - 1] + pressure_[x, y + 1]);
          pressure_[x, y] += w * (g - pressure_[x, y]);
        }
      }
    }
  }
  for (int y = 1; y <= n; ++y) {
    for (int x = 1; x <= n; ++x) {
      u_[x, y] -= 0.5f * (pressure_[x + 1, y] - pressure_[x - 1, y]);
      v_[x, y] -= 0.5f * (pressure_[x, y + 1] - pressure_[x, y - 1]);
    }
  }
  for (int k = 1; k <= n; ++k) {  // velocity border: copy the neighbour (outflow)
    for (Field* q : {&u_, &v_}) {
      (*q)[0, k] = (*q)[1, k];
      (*q)[n + 1, k] = (*q)[n, k];
      (*q)[k, 0] = (*q)[k, 1];
      (*q)[k, n + 1] = (*q)[k, n];
    }
  }
}

void Fluid::advect_velocity(float dt) {
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) {
      const float bx = f(x) - dt * u_[x, y], by = f(y) - dt * v_[x, y];
      tmp_a_[x, y] = u_.sample(bx, by);
      tmp_b_[x, y] = v_.sample(bx, by);
    }
  }
  std::swap(u_, tmp_a_);
  std::swap(v_, tmp_b_);
}

void Fluid::advect_scalar(Field& q, float dt) {
  // MacCormack: a forward step, a backward step, half the round-trip error added back, clamped to the stencil of the
  // forward step so no new extrema appear.
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) tmp_a_[x, y] = q.sample(f(x) - dt * u_[x, y], f(y) - dt * v_[x, y]);
  }
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) tmp_b_[x, y] = tmp_a_.sample(f(x) + dt * u_[x, y], f(y) + dt * v_[x, y]);
  }
  const float lim = f(n_) + 0.5f;
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) {
      const float bx = std::clamp(f(x) - dt * u_[x, y], 0.5f, lim), by = std::clamp(f(y) - dt * v_[x, y], 0.5f, lim);
      const int x0 = static_cast<int>(bx), y0 = static_cast<int>(by);
      const auto [lo, hi] = std::minmax({q[x0, y0], q[x0 + 1, y0], q[x0, y0 + 1], q[x0 + 1, y0 + 1]});
      tmp_c_[x, y] = std::max(0.f, std::clamp(tmp_a_[x, y] + 0.5f * (q[x, y] - tmp_b_[x, y]), lo, hi));
    }
  }
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) q[x, y] = tmp_c_[x, y];
  }
}

namespace {

struct Rgb {
  float r, g, b;
};

Rgb fire_ramp(float t) {
  return {clamp01((t - 0.10f) * 2.2f), clamp01((t - 0.36f) * 1.8f) * 0.92f, clamp01((t - 0.85f) * 2.0f) * 0.85f};
}

// magic's light: violet where it is faint, through blue and cyan to white where it is strongest.
Rgb magic_ramp(float t) {
  static constexpr std::array<float, 5> at{0.f, 0.25f, 0.55f, 0.85f, 1.2f};
  static constexpr std::array<Rgb, 5> c{{{0.f, 0.f, 0.f}, {0.30f, 0.05f, 0.55f}, {0.25f, 0.35f, 0.95f}, {0.35f, 0.85f, 1.f}, {0.9f, 1.f, 1.f}}};
  if (t <= 0.f) return c[0];
  for (std::size_t k = 1; k < at.size(); ++k) {
    if (t < at[k]) {
      const float w = (t - at[k - 1]) / (at[k] - at[k - 1]);
      return {c[k - 1].r + w * (c[k].r - c[k - 1].r), c[k - 1].g + w * (c[k].g - c[k - 1].g), c[k - 1].b + w * (c[k].b - c[k - 1].b)};
    }
  }
  return c[4];
}

std::uint8_t q8(float v) { return static_cast<std::uint8_t>(clamp01(v) * 255.f + 0.5f); }

}  // namespace

void Fluid::render(std::span<std::uint8_t> out) const {
  const int S = p_.size;
  if (out.size() < static_cast<std::size_t>(S) * S * 4) throw std::invalid_argument("sim: render buffer too small");
  const float n = f(n_), sc = n / f(S);
  constexpr float lx = -0.45f, ly = 0.89f;  // towards the light: up and to the left
  for (int py = 0; py < S; ++py) {
    for (int px = 0; px < S; ++px) {
      const float gx = (f(px) + 0.5f) * sc + 0.5f;
      const float gy = n - (f(py) + 0.5f) * sc + 0.5f;
      const float T = temp_.sample(gx, gy), D = soot_.sample(gx, gy);
      Rgb c{};
      float a = 0.f;
      if (p_.effect == Effect::fire) {
        const Rgb e = fire_ramp(T);
        const float as = 1.f - std::exp(-2.5f * D), keep = 1.f - 0.5f * as;
        c = {e.r * keep + 0.06f * as, e.g * keep + 0.05f * as, e.b * keep + 0.05f * as};
        a = clamp01(as + 0.25f * std::max(e.r, e.g));
      } else if (p_.effect == Effect::magic) {  // light, over a thin dark violet mist
        const Rgb e = magic_ramp(T);
        const float as = 1.f - std::exp(-2.f * D), keep = 1.f - 0.4f * as;
        c = {e.r * keep + 0.08f * as, e.g * keep + 0.03f * as, e.b * keep + 0.14f * as};
        a = clamp01(as + 0.3f * std::max({e.r, e.g, e.b}));
      } else {
        float tau = 0.f;  // optical depth towards the light
        for (int k = 1; k <= 12; ++k) tau += soot_.sample(gx + lx * 2.f * f(k) * sc, gy + ly * 2.f * f(k) * sc);
        const float light = std::exp(-0.55f * tau);
        const float as = 1.f - std::exp(-3.f * D);
        if (p_.effect == Effect::smoke) {
          const float g = 0.80f * (0.32f + 0.68f * light);
          c = {g * as, 0.98f * g * as, 0.95f * g * as};
          a = as;
        } else if (p_.effect == Effect::steam) {  // white vapour, softly shadowed, a little blue in its shade
          const float g = 0.97f * (0.55f + 0.45f * std::exp(-0.11f * tau));  // vapour shades itself less than smoke
          const float vs = 1.f - std::exp(-2.f * D);
          c = {0.95f * g * vs, 0.98f * g * vs, g * vs};
          a = vs;
        } else {
          const Rgb e = fire_ramp(0.75f * T);
          const float g = 0.42f * (0.30f + 0.70f * light), keep = 1.f - 0.6f * as;
          c = {e.r * keep + g * as, e.g * keep + 0.95f * g * as, e.b * keep + 0.90f * g * as};
          a = clamp01(as + 0.2f * std::max(e.r, e.g));
        }
      }
      auto o = out.subspan((static_cast<std::size_t>(py) * S + px) * 4, 4);
      o[0] = q8(c.r);
      o[1] = q8(c.g);
      o[2] = q8(c.b);
      o[3] = q8(a);  // rgb may exceed alpha: emission adds light
    }
  }
}

void stamp(Clip& clip, const Params& p) {
  clip.effect = effect_name(p.effect);
  clip.source = "sim";
  clip.fps = p.fps;
  clip.seed = p.seed;
  clip.n_controls = kControls;
  clip.controls = {};
  clip.controls[0] = p.intensity;
  clip.controls[1] = p.wind;
  clip.controls[2] = p.turbulence;
}

Params params_of(const Clip& clip) {
  Params p;
  if (!parse_effect(clip.effect, p.effect)) throw std::invalid_argument("params_of: not a simulated effect");
  p.intensity = clip.controls[0];
  p.wind = clip.controls[1];
  p.turbulence = clip.controls[2];
  p.seed = clip.seed;
  p.size = clip.size;
  p.frames = clip.frames;
  p.fps = clip.fps;
  return p;
}

Clip simulate(const Params& p) {
  if (p.frames < 1) throw std::invalid_argument("sim: frames must be positive");
  Fluid fluid(p);
  const bool loops = effect_loops(p.effect);
  const int blend = loops ? std::clamp(p.loop_blend, 0, p.frames / 2) : 0;
  if (loops) {
    // Long enough for the plume to reach a statistically steady state, so the loop crossfade joins similar frames.
    const int warmup = p.warmup >= 0 ? p.warmup : (p.effect == Effect::smoke ? 180 : 90);
    for (int i = 0; i < warmup; ++i) fluid.step_frame();
  }
  Clip raw;
  raw.allocate(p.size, p.frames + blend);
  for (int i = 0; i < raw.frames; ++i) {
    fluid.step_frame();
    fluid.render(raw.frame(i));
  }
  Clip clip;
  stamp(clip, p);
  clip.allocate(p.size, p.frames);
  clip.loop = loops;
  // out[i] = mix(raw[frames + i], raw[i], (i + 0.5) / blend) for i < blend: the end flows into the start.
  for (int i = 0; i < p.frames; ++i) {
    const auto o = clip.frame(i);
    const auto a = raw.frame(i);
    if (i >= blend) {
      std::ranges::copy(a, o.begin());
      continue;
    }
    const auto e = raw.frame(p.frames + i);
    const float w = (f(i) + 0.5f) / f(blend);
    for (const auto [k, out] : std::views::enumerate(o)) {
      out = static_cast<std::uint8_t>(static_cast<float>(e[static_cast<std::size_t>(k)]) * (1.f - w) +
                                      static_cast<float>(a[static_cast<std::size_t>(k)]) * w + 0.5f);
    }
  }
  return clip;
}

}  // namespace nfx::sim
