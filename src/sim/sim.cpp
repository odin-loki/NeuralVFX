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
    case Effect::jet: return "jet";
    case Effect::orb: return "orb";
    case Effect::vortex: return "vortex";
    case Effect::fog: return "fog";
    case Effect::dust: return "dust";
    case Effect::cloud: return "cloud";
    case Effect::wildfire: return "wildfire";
    case Effect::coals: return "coals";
    case Effect::mushroom: return "mushroom";
    case Effect::launch: return "launch";
    case Effect::puffs: return "puffs";
    case Effect::ink: return "ink";
    case Effect::waterfall: return "waterfall";
    case Effect::fountain: return "fountain";
    case Effect::splash: return "splash";
    case Effect::rainshaft: return "rainshaft";
    case Effect::lava: return "lava";
    case Effect::eruption: return "eruption";
    case Effect::avalanche: return "avalanche";
    case Effect::candle: return "candle";
    case Effect::smokering: return "smokering";
    case Effect::heavygas: return "heavygas";
    case Effect::crownfire: return "crownfire";
    case Effect::windowfire: return "windowfire";
    case Effect::dustexplosion: return "dustexplosion";
  }
  std::unreachable();
}

bool parse_effect(std::string_view text, Effect& out) {
  const auto it = std::ranges::find(kAllEffects, text, effect_name);
  if (it == kAllEffects.end()) return false;
  out = *it;
  return true;
}

bool effect_loops(Effect e) { return e != Effect::explosion && e != Effect::dust && e != Effect::mushroom &&
                                   e != Effect::splash && e != Effect::avalanche && e != Effect::smokering &&
                                   e != Effect::dustexplosion; }

int one_shot_frames(Effect e) {
  if (effect_loops(e)) return 0;
  if (e == Effect::avalanche) return 150;  // docs/EFFECTS_EARTH.md
  if (e == Effect::smokering) return 120;  // 4 s: a smoke ring's rise, roll and break-up (docs/EFFECTS_AIR.md)
  if (e == Effect::dustexplosion) return 120;  // 4 s: the burn through the cloud, the fireball, the pall (docs/EFFECTS_FIRE2.md)
  return e == Effect::mushroom ? 150 : 90;
}

std::array<std::string_view, kControls> control_names(Effect e) {
  if (e == Effect::magic) return {"intensity", "spin", "turbulence"};
  if (e == Effect::jet) return {"intensity", "aim", "turbulence"};
  if (e == Effect::orb) return {"intensity", "pulse", "turbulence"};
  if (e == Effect::vortex) return {"intensity", "sway", "turbulence"};
  if (e == Effect::puffs) return {"intensity", "rate", "turbulence"};  // docs/EFFECTS_FLOW.md
  if (e == Effect::ink) return {"intensity", "current", "turbulence"};
  if (e == Effect::fountain) return {"intensity", "aim", "turbulence"};
  if (e == Effect::lava) return {"intensity", "flow", "turbulence"};  // docs/EFFECTS_EARTH.md
  if (e == Effect::avalanche) return {"intensity", "direction", "turbulence"};
  if (e == Effect::candle) return {"intensity", "draught", "turbulence"};  // docs/EFFECTS_AIR.md
  if (e == Effect::heavygas) return {"intensity", "flow", "turbulence"};
  if (e == Effect::dustexplosion) return {"intensity", "ignition", "turbulence"};  // docs/EFFECTS_FIRE2.md
  return kControlNames;
}

float flicker_rate(Effect e) {
  switch (e) {
    case Effect::fire: return 2.6f;
    case Effect::smoke: return 1.4f;
    case Effect::explosion: return 1.4f;
    case Effect::steam: return 2.0f;   // puffs from a vent
    case Effect::magic: return 1.8f;   // a shimmer
    case Effect::jet: return 3.0f;     // a roaring burner
    case Effect::orb: return 2.2f;     // a crackle
    case Effect::vortex: return 1.5f;  // gusts of dust
    case Effect::fog: return 1.0f;     // a slow, uneven release
    case Effect::dust: return 1.4f;    // the burst's shape, as the explosion's
    case Effect::cloud: return 0.6f;   // slow billows
    case Effect::wildfire: return 2.2f;  // a front of flickering flames
    case Effect::coals: return 0.8f;     // a glow that crawls
    case Effect::mushroom: return 1.4f;  // the burst's shape, as the explosion's
    case Effect::launch: return 3.0f;  // a roaring engine, as the jet's burner
    case Effect::puffs: return 4.0f;   // the pulses' timing (puff_gate)
    case Effect::ink: return 0.8f;     // a slow, uneven release
    case Effect::waterfall: return 1.2f;  // the streaks across the lip
    case Effect::fountain: return 2.4f;   // a fluttering jet
    case Effect::splash: return 1.4f;     // the crown's shape, as the burst of dust
    case Effect::rainshaft: return 0.7f;  // slow gusts: the shafts thin and thicken
    case Effect::lava: return 0.6f;       // a slow crawl of the surface (docs/EFFECTS_EARTH.md)
    case Effect::eruption: return 2.0f;   // a roaring vent
    case Effect::avalanche: return 1.4f;  // the slab's shape, as the burst of dust
    case Effect::candle: return 1.2f;     // a flame that breathes and gutters
    case Effect::smokering: return 1.4f;  // the puff's lumps, as the explosion's
    case Effect::heavygas: return 0.8f;   // a slow, uneven release, as ink's
    case Effect::crownfire: return 2.2f;      // crowns torching, as the wildfire's front (docs/EFFECTS_FIRE2.md)
    case Effect::windowfire: return 2.6f;     // flames, as fire's
    case Effect::dustexplosion: return 1.4f;  // the cloud's lumps, as the burst of dust
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
    // jet, orb and vortex: their own forces are in add_energy_forces (docs/EFFECTS_ENERGY.md); no wind
    case Effect::jet: return {38.f, 3.f, 1.1f, 0.4f, 0.f, 0.f, 0.8f + 3.2f * turb, 12.f + 55.f * turb, 0.45f};
    case Effect::orb: return {0.f, 0.f, 1.0f, 1.2f, 0.f, 0.f, 1.5f + 3.5f * turb, 20.f + 60.f * turb, 1.0f};
    case Effect::vortex: return {115.f, 2.f, 0.4f, 0.75f, 0.f, 0.f, 0.3f + 0.5f * turb, 4.f + 8.f * turb, 0.35f};
    // fog, dust and cloud (docs/EFFECTS_ATMOSPHERE.md) carry no heat: their material is the solver's soot, and its
    // weight moves them. fog: heavy, thins slowly, drifts with a light wind. dust: very heavy, so the burst falls back
    // and spreads along the ground. cloud: light (its thick parts rise a little), slow and well damped. Their sources,
    // the ground under fog and dust and dust's burst are in add_atmos_sources, add_atmos_forces, add_atmos_divergence.
    case Effect::fog: return {0.f, 20.f, 1.f, 0.12f, 20.f, 1.5f, 0.2f + 1.2f * turb, 3.f + 14.f * turb, 0.4f};
    case Effect::dust: return {0.f, 150.f, 1.f, 0.25f, 50.f, 1.5f, 1.5f + 4.0f * turb, 20.f + 80.f * turb, 0.6f};
    case Effect::cloud: return {0.f, -20.f, 1.f, 0.10f, 40.f, 1.0f, 0.2f + 0.8f * turb, 3.f + 16.f * turb, 1.3f};
    // wildfire, coals and mushroom (docs/EFFECTS_BLAZE.md); their sources, the coals' still bed, the mushroom's burst,
    // ceiling and sheared wind are in add_blaze_sources, add_blaze_forces and add_blaze_divergence. wildfire: fire's
    // buoyancy, its flames shorter at low intensity, smoke from what burns. coals: little rise, short flames that cool
    // fast, thin smoke. mushroom: a hot cloud that cools slowly into smoke; its wind is its own (sheared).
    case Effect::wildfire:
      return {115.f, 6.f, 3.2f - 1.0f * p.intensity, 0.3f, 70.f, 2.0f, 0.6f + 3.0f * turb, 10.f + 55.f * turb, 0.2f};
    case Effect::coals: return {90.f, 2.f, 4.5f, 3.0f, 45.f, 1.8f, 0.3f + 1.5f * turb, 5.f + 30.f * turb, 0.6f};
    case Effect::mushroom: return {45.f, 2.f, 0.45f, 0.12f, 0.f, 0.f, 0.3f + 1.0f * turb, 5.f + 25.f * turb, 0.5f};
    // launch, puffs and ink (docs/EFFECTS_FLOW.md); their own sources and forces are in add_flow_sources and
    // add_flow_forces. launch: hot exhaust (little buoyancy, so it reaches the ground) that turns into light smoke and
    // steam, which rises; the wind drifts it. puffs: warm smoke, no wind (the second control is the rate; a fixed
    // drift instead). ink: no heat, dye barely lighter than the water, slow and damped; the current is its own force.
    case Effect::launch: return {20.f, -8.f, 1.6f, 0.15f, 30.f, 1.0f, 0.4f + 1.6f * turb, 8.f + 30.f * turb, 0.3f};
    case Effect::puffs: return {40.f, 0.5f, 0.6f, 0.22f, 0.f, 0.f, 0.2f + 1.2f * turb, 5.f + 25.f * turb, 0.4f};
    case Effect::ink: return {0.f, -3.f, 1.f, 0.05f, 0.f, 0.f, 0.05f + 0.35f * turb, 3.f + 8.f * turb, 0.6f};
    // waterfall, fountain, splash and rainshaft (docs/EFFECTS_WATER.md): water is the solver's soot, heavy, so it falls;
    // spray and mist are its heat, light, so they rise and fade (the cooling is the mist's). Their wind blows the mist
    // and the rain, not the falling water, so it is in add_water_forces with how their water falls and their pools.
    // Water does not roll up in the air as gas does: little confinement and curl noise, and still air keeps little
    // momentum of its own (vel_damp), so its sheets and arcs stay whole; the mist churns by its own force.
    case Effect::waterfall: return {120.f, 0.f, 0.6f, 0.05f, 0.f, 0.f, 0.1f + 0.5f * turb, 2.f + 12.f * turb, 1.0f};
    case Effect::fountain: return {20.f, 30.f, 0.6f, 0.05f, 0.f, 0.f, 0.05f + 0.3f * turb, 2.f + 10.f * turb, 1.6f};
    case Effect::splash: return {10.f, 60.f, 0.6f, 0.05f, 0.f, 0.f, 0.1f + 0.4f * turb, 2.f + 10.f * turb, 1.6f};
    case Effect::rainshaft: return {25.f, 0.f, 0.6f, 0.10f, 0.f, 0.f, 0.1f + 0.5f * turb, 3.f + 12.f * turb, 0.5f};
    // lava, eruption and avalanche (docs/EFFECTS_EARTH.md); their sources, flows, ground and drawing are in
    // add_earth_sources, add_earth_forces and earth_decay. lava: heavy rock (soot) whose heat is its glow, barely stirred,
    // very viscous. eruption: hot ash that rises while it is hot and is heavy once it cools; its wind is its own
    // (sheared). avalanche: snow (soot, no heat), heavy, lifted by its own speed; its slope is its own force.
    case Effect::lava: return {0.f, 0.f, 0.3f, 0.05f, 0.f, 0.f, 0.05f + 0.25f * turb, 1.f + 6.f * turb, 2.0f};
    case Effect::eruption: return {70.f, 12.f, 0.6f, 0.08f, 0.f, 0.f, 0.3f + 1.2f * turb, 4.f + 20.f * turb, 0.4f};
    case Effect::avalanche: return {0.f, 25.f, 1.f, 0.12f, 0.f, 0.f, 0.4f + 1.2f * turb, 10.f + 40.f * turb, 0.6f};
    // candle, smokering and heavygas (docs/EFFECTS_AIR.md); their sources, the candle's gusty draught, heavy gas's floor and
    // its draught are in add_air_sources and add_air_forces. candle: a small flame whose heat cools fast (its visible
    // length), a warm plume above it, calm air. smokering: warm smoke, little buoyancy, its own vortices; the wind
    // drifts it. heavygas: no heat, heavy, thins slowly.
    case Effect::candle: return {200.f, 0.f, 0.3f, 0.8f, 0.f, 0.f, 0.15f + 0.9f * turb, 2.f + 22.f * turb, 0.3f};
    case Effect::smokering: return {30.f, 0.f, 0.8f, 0.25f, 0.f, 0.f, 0.2f + 1.0f * turb, 4.f + 26.f * turb, 0.35f};
    case Effect::heavygas: return {0.f, 18.f, 1.f, 0.08f, 0.f, 0.f, 0.2f + 1.2f * turb, 3.f + 14.f * turb, 0.2f};
    // crownfire, windowfire and dustexplosion (docs/EFFECTS_FIRE2.md); their sources, winds, the window fire's ceiling and
    // the dust explosion's burning are in add_fire2_sources, add_fire2_forces, add_fire2_divergence and fire2_decay.
    // crownfire: hot flames over the crowns that cool into a tall column of thick smoke; its wind is its own (sheared,
    // slowed in the canopy). windowfire: flames out of a room that cool into black smoke. dustexplosion: dust (soot)
    // that burns where it is hot; the burnt cloud rises, cools and its dust settles.
    case Effect::crownfire: return {110.f, 3.f, 0.3f, 0.08f, 0.f, 0.f, 0.5f + 2.5f * turb, 8.f + 45.f * turb, 0.2f};
    case Effect::windowfire: return {110.f, 2.f, 0.5f, 0.2f, 0.f, 0.f, 0.4f + 1.6f * turb, 8.f + 40.f * turb, 0.2f};
    case Effect::dustexplosion: return {35.f, 5.f, 1.3f, 0.08f, 0.f, 0.f, 0.2f + 0.6f * turb, 4.f + 20.f * turb, 0.6f};
  }
  std::unreachable();
}

// fog, dust and cloud (docs/EFFECTS_ATMOSPHERE.md)
bool atmos(Effect e) { return e == Effect::fog || e == Effect::dust || e == Effect::cloud; }
bool on_ground(Effect e) { return e == Effect::fog || e == Effect::dust; }  // lie on a ground (add_atmos_forces)
// waterfall, fountain, splash and rainshaft (docs/EFFECTS_WATER.md): they fall onto a pool or the ground
bool water(Effect e) { return e == Effect::waterfall || e == Effect::fountain || e == Effect::splash || e == Effect::rainshaft; }
// lava, eruption and avalanche (docs/EFFECTS_EARTH.md): they lie on the ground (project)
bool earth(Effect e) { return e == Effect::lava || e == Effect::eruption || e == Effect::avalanche; }

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
  energy_decay(dt);  // jet, orb and vortex
  blaze_decay(dt);   // wildfire and mushroom
  flow_decay(dt);    // launch
  water_decay(dt);   // waterfall, fountain, splash and rainshaft
  earth_decay(dt);   // avalanche
  air_decay(dt);     // candle, smokering and heavygas
  fire2_decay(dt);   // crownfire, windowfire and dustexplosion
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
  if (add_energy_sources(dt)) return;  // jet, orb and vortex
  if (add_water_sources(dt)) return;   // waterfall, fountain, splash and rainshaft
  if (atmos(p_.effect)) {
    add_atmos_sources(dt);
    return;
  }
  if (add_blaze_sources(dt)) return;  // wildfire, coals and mushroom
  if (add_flow_sources(dt)) return;  // launch, puffs and ink
  if (add_earth_sources(dt)) return;  // lava, eruption and avalanche
  if (add_air_sources(dt)) return;   // candle, smokering and heavygas
  if (add_fire2_sources(dt)) return;  // crownfire, windowfire and dustexplosion
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
  add_energy_forces(dt);  // jet, orb and vortex
  if (atmos(p_.effect)) add_atmos_forces(dt);
  add_blaze_forces(dt);  // wildfire, coals and mushroom
  add_flow_forces(dt);  // launch, puffs and ink
  add_water_forces(dt);  // waterfall, fountain, splash and rainshaft
  add_earth_forces(dt);  // lava, eruption and avalanche
  add_air_forces(dt);    // candle, smokering and heavygas
  add_fire2_forces(dt);  // crownfire, windowfire and dustexplosion
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
  add_energy_divergence();  // jet, orb and vortex
  if (p_.effect == Effect::dust) add_atmos_divergence();
  add_blaze_divergence();  // the mushroom's burst
  add_earth_divergence();  // the avalanche's cloud (docs/EFFECTS_EARTH.md)
  add_fire2_divergence();  // the dust explosion's burning gas (docs/EFFECTS_FIRE2.md)
  // Open boundaries: p = 0 on the border, which is never written. Warm-started from the last solve.
  constexpr float w = 1.7f;
  // the mushroom and launch stand on the ground too (docs/EFFECTS_BLAZE.md, docs/EFFECTS_FLOW.md), and water falls onto it
  const bool ground = on_ground(p_.effect) || p_.effect == Effect::mushroom || p_.effect == Effect::launch ||
                    water(p_.effect) || earth(p_.effect) || p_.effect == Effect::heavygas ||  // docs/EFFECTS_EARTH.md, docs/EFFECTS_AIR.md
                    p_.effect == Effect::crownfire || p_.effect == Effect::dustexplosion;     // docs/EFFECTS_FIRE2.md
  const bool ceiling = p_.effect == Effect::windowfire;  // the eave over the window fire (docs/EFFECTS_FIRE2.md)
  for (int it = 0; it < p_.pressure_iters; ++it) {
    for (const int colour : {0, 1}) {
      for (int y = 1; y <= n; ++y) {
        for (int x = 1 + ((y + colour) & 1); x <= n; x += 2) {
          const float g = 0.25f * (div_[x, y] + pressure_[x - 1, y] + pressure_[x + 1, y] + pressure_[x, y - 1] + pressure_[x, y + 1]);
          pressure_[x, y] += w * (g - pressure_[x, y]);
        }
      }
      if (ground) {  // fog, dust and the water effects: the bottom border is closed (no flow through the ground)
        for (int x = 1; x <= n; ++x) pressure_[x, 0] = pressure_[x, 1];
      }
      if (ceiling) {  // and the top border under a ceiling
        for (int x = 1; x <= n; ++x) pressure_[x, n + 1] = pressure_[x, n];
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
  if (render_energy(out)) return;  // jet, orb and vortex
  if (render_flow(out)) return;    // launch, puffs and ink
  if (render_water(out)) return;   // waterfall, fountain, splash and rainshaft
  if (render_air(out)) return;     // candle, smokering and heavygas
  if (render_fire2(out)) return;   // crownfire, windowfire and dustexplosion
  if (atmos(p_.effect)) {
    render_atmos(out);
    return;
  }
  if (render_blaze(out)) return;  // wildfire, coals and mushroom
  if (render_earth(out)) return;  // lava, eruption and avalanche
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

// --- jet, orb and vortex (docs/EFFECTS_ENERGY.md) ---------------------------------------------------------------------
//
// Three presets kept apart from the other effects' code: their sources, forces, decay and drawing. Every force is a
// function of the state where it acts, the position, the controls and the run's two noise fields (curl_potential,
// source_flicker), which learned dynamics see as inputs; nothing follows a clock of its own.

namespace {

bool energy_effect(Effect e) { return e == Effect::jet || e == Effect::orb || e == Effect::vortex; }

// jet: a nozzle at the left edge, aimed from 20 degrees below to 30 degrees above horizontal (the second control).
struct JetNozzle {
  float x, y, c, s;  // position (cells), direction (cos, sin)
};
JetNozzle jet_nozzle(const Params& p, float n) {
  const float a = (-20.f + 50.f * p.wind) * 0.017453293f;
  return {0.05f * n, 0.40f * n, std::cos(a), std::sin(a)};
}

// orb: a ball of plasma in the middle of the frame (radius in cells), breathing with the curl noise where it is.
float orb_radius(const Params& p, float n) { return (0.16f + 0.10f * p.intensity) * n; }
float orb_beat(const Params& p, float psi) { return std::tanh((1.f + 4.f * p.wind) * psi); }  // pulse: sharper beats

// vortex: a column of warm air rising from the ground in the middle of the frame; half-width of its base (cells).
float vortex_base(const Params& p, float n) { return (0.035f + 0.03f * p.intensity) * n; }

}  // namespace

bool Fluid::add_energy_sources(float dt) {
  if (!energy_effect(p_.effect)) return false;
  const float n = f(n_), s = n / 128.f, I = p_.intensity;
  const auto flicker = [&](int x, int y) {
    return smoothstep(-0.25f, 0.6f, source_flicker(p_, f(x) / s, f(y) / s, time_));
  };
  if (p_.effect == Effect::jet) {
    const JetNozzle g = jet_nozzle(p_, n);
    const float hl = 0.045f * n, hw = (0.032f + 0.016f * I) * n, speed = (85.f + 65.f * I) * s;
    const float ext = hl + hw + 2.f;
    const int x_lo = std::max(1, static_cast<int>(g.x - ext)), x_hi = std::min(n_, static_cast<int>(g.x + ext) + 1);
    const int y_lo = std::max(1, static_cast<int>(g.y - ext)), y_hi = std::min(n_, static_cast<int>(g.y + ext) + 1);
    for (int y = y_lo; y <= y_hi; ++y) {
      for (int x = x_lo; x <= x_hi; ++x) {
        const float dx = f(x) - g.x, dy = f(y) - g.y;
        const float ea = (dx * g.c + dy * g.s) / hl, ec = (dy * g.c - dx * g.s) / hw;
        const float shape = 1.f - smoothstep(0.6f, 1.f, std::sqrt(ea * ea + ec * ec));
        if (shape <= 0.f) continue;
        const float k = shape * (0.35f + 0.65f * flicker(x, y));
        temp_[x, y] = std::max(temp_[x, y], (1.5f + 0.7f * I) * k);
        soot_[x, y] += dt * (0.4f + 0.3f * I) * k;
        u_[x, y] += k * (speed * g.c - u_[x, y]);
        v_[x, y] += k * (speed * g.s - v_[x, y]);
      }
    }
    return true;
  }
  if (p_.effect == Effect::orb) {
    const float c = 0.5f * (n + 1.f), R = orb_radius(p_, n);
    const int lo = std::max(1, static_cast<int>(c - R)), hi = std::min(n_, static_cast<int>(c + R) + 1);
    for (int y = lo; y <= hi; ++y) {
      for (int x = lo; x <= hi; ++x) {
        const float dx = f(x) - c, dy = f(y) - c;
        const float r = std::sqrt(dx * dx + dy * dy) / R, shape = 1.f - smoothstep(0.5f, 0.9f, r);
        if (shape <= 0.f) continue;
        const float beat = orb_beat(p_, curl_potential(p_, f(x) / s, f(y) / s, time_));
        const float e = (1.1f + 0.5f * I) * (1.f - 0.35f * r * r) * (1.f + 0.4f * p_.wind * beat);
        const float k = shape * (0.4f + 0.6f * flicker(x, y));
        temp_[x, y] = std::max(temp_[x, y], e * k);
        soot_[x, y] = std::max(soot_[x, y], 0.35f * e * k);
      }
    }
    return true;
  }
  // vortex: warm air from a patch of hot ground, and dust lifted where the air rushes in along the ground
  const float x0 = 0.5f * (n + 1.f), wb = vortex_base(p_, n), g0 = 0.06f * n;
  const int y_hi = std::min(n_, static_cast<int>(g0) + 1);
  for (int y = 1; y <= y_hi; ++y) {
    for (int x = 1; x <= n_; ++x) {
      const float ax = std::abs(f(x) - x0), ground = 1.f - smoothstep(0.4f * g0, g0, f(y));
      const float warm = (1.f - smoothstep(0.5f * wb, wb, ax)) * ground;
      if (warm > 0.f) temp_[x, y] = std::max(temp_[x, y], (0.5f + 0.3f * I) * warm);
      const float skirt = (1.f - smoothstep(1.2f * wb, 3.5f * wb, ax)) * ground;
      if (skirt <= 0.f) continue;
      const float rush = std::min(1.f, std::abs(u_[x, y]) / (12.f * s) + warm);
      soot_[x, y] += dt * (14.f + 10.f * I) * skirt * rush * (0.3f + 0.7f * flicker(x, y));
    }
  }
  return true;
}

void Fluid::add_energy_forces(float dt) {
  if (!energy_effect(p_.effect)) return;
  const float n = f(n_), s = n / 128.f;
  if (p_.effect == Effect::orb) {  // hot plasma is drawn to the middle; the air around it is still
    const float c = 0.5f * (n + 1.f), R = orb_radius(p_, n), G = 140.f * s;
    for (int y = 1; y <= n_; ++y) {
      for (int x = 1; x <= n_; ++x) {
        const float dx = f(x) - c, dy = f(y) - c, r = std::sqrt(dx * dx + dy * dy) + 1e-3f;
        const float pull = G * temp_[x, y] * smoothstep(0.2f * R, 0.6f * R, r) / r;
        const float hold = std::exp(-4.f * dt * smoothstep(1.05f * R, 1.6f * R, r));
        u_[x, y] = (u_[x, y] - dt * pull * dx) * hold;
        v_[x, y] = (v_[x, y] - dt * pull * dy) * hold;
      }
    }
    return;
  }
  if (p_.effect == Effect::vortex) {
    // Gusts: the air is dragged towards a wind that swings with the curl noise where it is (sway: how strongly), so
    // the column leans and wanders. Above the ground the column is stirred more the higher it gets (its own vorticity
    // confinement and curl noise, on top of tune()'s small amounts): smooth at its foot, eddying at its top.
    const float G = (8.f + 42.f * p_.wind) * s, drag = 1.f - std::exp(-1.5f * dt), turb = p_.turbulence;
    const float L = 14.f * s, e = 0.25f, tt = time_ * 0.8f;
    const std::uint64_t cseed = p_.seed * 0x2545F4914F6CDD1DULL + 77;  // as curl_potential
    for (int y = 1; y <= n_; ++y) {
      const float h = smoothstep(0.1f * n, 0.85f * n, f(y));
      for (int x = 1; x <= n_; ++x) {
        u_[x, y] += drag * (G * curl_potential(p_, f(x) / s, f(y) / s, time_) - u_[x, y]);
        const float w = std::min(1.f, 4.f * temp_[x, y] + 2.f * soot_[x, y]);
        if (w <= 0.01f || h <= 0.f || x == 1 || x == n_ || y == 1 || y == n_) continue;
        const float gx = 0.5f * (std::fabs(curl_[x + 1, y]) - std::fabs(curl_[x - 1, y]));
        const float gy = 0.5f * (std::fabs(curl_[x, y + 1]) - std::fabs(curl_[x, y - 1]));
        const float len = std::sqrt(gx * gx + gy * gy) + 1e-5f, conf = h * (0.5f + 3.f * turb);
        u_[x, y] += dt * conf * (gy / len) * curl_[x, y];
        v_[x, y] -= dt * conf * (gx / len) * curl_[x, y];
        const float px = f(x) / L, py = f(y) / L, amp = h * w * (8.f + 50.f * turb) * s;
        u_[x, y] += dt * amp * (value_noise(px, py + e, tt, cseed) - value_noise(px, py - e, tt, cseed)) / (2.f * e);
        v_[x, y] -= dt * amp * (value_noise(px + e, py, tt, cseed) - value_noise(px - e, py, tt, cseed)) / (2.f * e);
      }
    }
  }
}

void Fluid::add_energy_divergence() {
  if (p_.effect == Effect::vortex) {
    // The warm column draws air in at its foot and spreads as it rises (more at higher intensity): a funnel.
    const float n = f(n_);
    for (int y = 1; y <= n_; ++y) {
      const float up = 3.f * (0.6f + 0.8f * p_.intensity) * clamp01(f(y) / n);
      const float foot = 6.f * (1.f - smoothstep(0.03f * n, 0.15f * n, f(y)));
      for (int x = 1; x <= n_; ++x) div_[x, y] += (up - foot) * std::min(1.f, 2.f * temp_[x, y]);
    }
    return;
  }
  if (p_.effect != Effect::orb) return;
  // the orb breathes: a source of divergence in its core, driven by the curl noise there (pulse: how strongly)
  const float n = f(n_), s = n / 128.f, c = 0.5f * (n + 1.f), R = orb_radius(p_, n), D = 1.f + 5.f * p_.wind;
  const int lo = std::max(1, static_cast<int>(c - 0.5f * R)), hi = std::min(n_, static_cast<int>(c + 0.5f * R) + 1);
  for (int y = lo; y <= hi; ++y) {
    for (int x = lo; x <= hi; ++x) {
      const float dx = f(x) - c, dy = f(y) - c;
      const float core = 1.f - smoothstep(0.2f * R, 0.45f * R, std::sqrt(dx * dx + dy * dy));
      if (core <= 0.f) continue;
      div_[x, y] += D * core * orb_beat(p_, curl_potential(p_, f(x) / s, f(y) / s, time_));
    }
  }
}

void Fluid::energy_decay(float dt) {
  if (p_.effect == Effect::jet) {  // the burning gas turns to smoke as it burns
    for (int y = 1; y <= n_; ++y) {
      for (int x = 1; x <= n_; ++x) soot_[x, y] += dt * 0.7f * temp_[x, y];
    }
    return;
  }
  if (p_.effect == Effect::vortex) {  // dust spreads sideways as it rises: the funnel widens towards the top
    const float n = f(n_), s = n / 128.f;
    for (int y = 1; y <= n_; ++y) {  // explicit diffusion along rows, 25 (y / n)^2 cells^2 per second
      const float h = f(y) / n, k = std::min(0.45f, 25.f * s * s * h * h * dt);
      if (k <= 0.f) continue;
      for (int x = 1; x <= n_; ++x) tmp_a_[x, y] = soot_[x - 1, y] + soot_[x + 1, y] - 2.f * soot_[x, y];
      for (int x = 1; x <= n_; ++x) soot_[x, y] = std::max(0.f, soot_[x, y] + k * tmp_a_[x, y]);
    }
    return;
  }
  if (p_.effect != Effect::orb) return;
  // the orb's soft boundary: plasma thrown out of it fades within a few frames
  const float n = f(n_), c = 0.5f * (n + 1.f), R = orb_radius(p_, n);
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) {
      const float dx = f(x) - c, dy = f(y) - c;
      const float out = smoothstep(0.95f * R, 1.5f * R, std::sqrt(dx * dx + dy * dy));
      if (out <= 0.f) continue;
      const float k = std::exp(-3.5f * dt * out);
      temp_[x, y] *= k;
      soot_[x, y] *= k;
    }
  }
}

namespace {

// A colour ramp through stops (t, colour); black below the first.
template <std::size_t N>
Rgb ramp(const std::array<float, N>& at, const std::array<Rgb, N>& c, float t) {
  if (t <= at[0]) return c[0];
  for (std::size_t k = 1; k < N; ++k) {
    if (t < at[k]) {
      const float w = (t - at[k - 1]) / (at[k] - at[k - 1]);
      return {c[k - 1].r + w * (c[k].r - c[k - 1].r), c[k - 1].g + w * (c[k].g - c[k - 1].g), c[k - 1].b + w * (c[k].b - c[k - 1].b)};
    }
  }
  return c[N - 1];
}

// the jet's flame: dark red where it is cooling, through orange and yellow to white, bluish white at its hottest
Rgb jet_ramp(float t) {
  static constexpr std::array<float, 6> at{0.f, 0.2f, 0.45f, 0.8f, 1.25f, 1.8f};
  static constexpr std::array<Rgb, 6> c{{{0.f, 0.f, 0.f}, {0.45f, 0.05f, 0.f}, {1.f, 0.42f, 0.04f}, {1.f, 0.82f, 0.3f}, {1.f, 0.97f, 0.88f}, {0.78f, 0.9f, 1.f}}};
  return ramp(at, c, t);
}

// the orb's light: deep blue where it is faint, through cyan to white
Rgb orb_ramp(float t) {
  static constexpr std::array<float, 5> at{0.f, 0.15f, 0.35f, 0.65f, 1.f};
  static constexpr std::array<Rgb, 5> c{{{0.f, 0.f, 0.f}, {0.03f, 0.14f, 0.42f}, {0.10f, 0.58f, 0.95f}, {0.55f, 0.97f, 1.f}, {0.95f, 1.f, 1.f}}};
  return ramp(at, c, t);
}

}  // namespace

bool Fluid::render_energy(std::span<std::uint8_t> out) const {
  if (!energy_effect(p_.effect)) return false;
  const int S = p_.size;
  const float n = f(n_), sc = n / f(S);
  constexpr float lx = -0.45f, ly = 0.89f;  // towards the light: up and to the left
  for (int py = 0; py < S; ++py) {
    for (int px = 0; px < S; ++px) {
      const float gx = (f(px) + 0.5f) * sc + 0.5f;
      const float gy = n - (f(py) + 0.5f) * sc + 0.5f;
      const float T = temp_.sample(gx, gy), D = soot_.sample(gx, gy);
      Rgb c{};
      float a = 0.f;
      if (p_.effect == Effect::orb) {  // light, with a faint blue haze where it has been
        const Rgb e = orb_ramp(T);
        const float h = 1.f - std::exp(-1.5f * D);
        c = {e.r + 0.02f * h, e.g + 0.12f * h, e.b + 0.30f * h};
        a = clamp01(0.35f * std::max({e.r, e.g, e.b}) + 0.15f * h);
      } else {
        float tau = 0.f;  // optical depth towards the light
        for (int k = 1; k <= 12; ++k) tau += soot_.sample(gx + lx * 2.f * f(k) * sc, gy + ly * 2.f * f(k) * sc);
        const float light = std::exp(-0.5f * tau);
        if (p_.effect == Effect::jet) {  // flame over dark smoke
          const Rgb e = jet_ramp(T);
          const float as = 1.f - std::exp(-2.2f * D), keep = 1.f - 0.55f * as, g = 0.22f * (0.35f + 0.65f * light);
          c = {e.r * keep + g * as, e.g * keep + 0.95f * g * as, e.b * keep + 0.9f * g * as};
          a = clamp01(as + 0.25f * std::max(e.r, e.g));
        } else {  // vortex: brown-grey dust, lit from above and shaded by itself
          const float as = 1.f - std::exp(-2.5f * D), g = 0.32f + 0.68f * light;
          c = {0.70f * g * as, 0.60f * g * as, 0.48f * g * as};
          a = as;
        }
      }
      auto o = out.subspan((static_cast<std::size_t>(py) * S + px) * 4, 4);
      o[0] = q8(c.r);
      o[1] = q8(c.g);
      o[2] = q8(c.b);
      o[3] = q8(a);
    }
  }
  return true;
}

// --- fog, dust and cloud (docs/EFFECTS_ATMOSPHERE.md) -------------------------------------------------------------------
//
// None of them has heat: their material is the solver's soot, and they give no light. Fog and dust lie on a ground: a
// band of still air three cells high at the bottom of the frame (add_atmos_forces) over a closed bottom border (project),
// so what sinks onto it spreads sideways. Cloud has no ground and no source to be seen.

namespace {

constexpr float kGround = 3.f;  // the top of the ground, in cells of a 128-cell frame

}  // namespace

// fog: heavy gas released a little above the ground, poured downwards. dust: one burst at the start, a mound of dust
// over the point of impact. cloud: inside the cloud's extent (wide, flatter below than above, its edge moved by the
// flicker noise) the density is drawn gently towards a target that the flicker noise breaks into puffs; outside it,
// what the flow carries out thins away. All are broken up by the flicker noise (source_flicker), as fire's source is.
void Fluid::add_atmos_sources(float dt) {
  const float n = f(n_), s = n / 128.f, I = p_.intensity;
  const float tt = time_ * flicker_rate(p_.effect);
  const auto flicker = [&](int x, int y) { return fbm(f(x) * 0.10f / s, f(y) * 0.10f / s, tt, p_.seed, 3); };
  if (p_.effect == Effect::fog) {
    const float cx = 0.5f * n, cy = 0.2f * n, hw = (0.10f + 0.08f * I) * n, hh = 0.07f * n;
    const float level = 0.5f + 0.5f * I, pour = (4.f + 8.f * I) * s;
    const int y_lo = std::max(1, static_cast<int>(cy - 2.f * hh)), y_hi = std::min(n_, static_cast<int>(cy + 2.f * hh) + 1);
    const int x_lo = std::max(1, static_cast<int>(cx - 1.5f * hw)), x_hi = std::min(n_, static_cast<int>(cx + 1.5f * hw) + 1);
    for (int y = y_lo; y <= y_hi; ++y) {
      for (int x = x_lo; x <= x_hi; ++x) {
        const float ex = (f(x) - cx) / hw, ey = (f(y) - cy) / hh;
        const float shape = 1.f - smoothstep(0.6f, 1.f, std::sqrt(ex * ex + ey * ey));
        if (shape <= 0.f) continue;
        const float k = shape * smoothstep(-0.3f, 0.5f, flicker(x, y));
        soot_[x, y] = std::max(soot_[x, y], level * k);
        v_[x, y] = std::min(v_[x, y], -pour * k);
      }
    }
    return;
  }
  if (p_.effect == Effect::dust) {
    if (time_ > 0.f) return;  // one burst at the start, as the explosion's; add_atmos_divergence throws it out
    const float cx = 0.5f * n, g = kGround * s, a = (0.06f + 0.03f * I) * n, b = (0.05f + 0.02f * I) * n;
    const float level = 1.f + 0.6f * I;
    const int y_lo = std::max(1, static_cast<int>(g)), y_hi = std::min(n_, static_cast<int>(g + 1.5f * b) + 1);
    const int x_lo = std::max(1, static_cast<int>(cx - 1.5f * a)), x_hi = std::min(n_, static_cast<int>(cx + 1.5f * a) + 1);
    for (int y = y_lo; y <= y_hi; ++y) {
      for (int x = x_lo; x <= x_hi; ++x) {
        if (f(y) < g) continue;
        const float ex = (f(x) - cx) / a, ey = (f(y) - g) / b, fl = flicker(x, y);
        const float fall = 1.f - smoothstep(0.5f, 1.f + 0.4f * fl, std::sqrt(ex * ex + ey * ey));
        if (fall <= 0.f) continue;
        soot_[x, y] = std::max(soot_[x, y], level * fall * (0.7f + 0.3f * smoothstep(-0.6f, 0.6f, fl)));
      }
    }
    return;
  }
  const float cx = 0.5f * n, cy = 0.5f * n, a = (0.26f + 0.14f * I) * n, b = (0.13f + 0.07f * I) * n;
  const float level = 0.6f + 0.6f * I, rate = 1.f - std::exp(-0.4f * dt), out = 1.f - std::exp(-1.5f * dt);
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) {
      const float ex = (f(x) - cx) / a, ey = (f(y) - cy) / b * (f(y) < cy ? 1.6f : 1.f), fl = flicker(x, y);
      const float inside = 1.f - smoothstep(0.55f, 1.f, std::sqrt(ex * ex + ey * ey) - 0.25f * fl);
      soot_[x, y] -= out * (1.f - inside) * soot_[x, y];
      if (inside <= 0.f) continue;
      const float target = level * inside * (0.25f + 0.75f * smoothstep(-0.2f, 0.5f, fl));
      soot_[x, y] += rate * std::max(0.f, target - soot_[x, y]);
    }
  }
}

// The ground under fog and dust: the air in it is still, and what comes down onto it stops (the closed bottom border
// of the pressure solve does the rest). It acts like the other forces, before the pressure solve.
void Fluid::add_atmos_forces(float dt) {
  (void)dt;
  if (!on_ground(p_.effect)) return;
  const float s = f(n_) / 128.f, g = kGround * s, top = g + 4.f * s;
  const int y_hi = std::min(n_, static_cast<int>(top) + 1);
  for (int y = 1; y <= y_hi; ++y) {
    const float hold = 1.f - smoothstep(g - 2.f * s, g + 2.f * s, f(y)), stop = 1.f - smoothstep(g, top, f(y));
    for (int x = 1; x <= n_; ++x) {
      u_[x, y] *= 1.f - hold;
      v_[x, y] *= 1.f - (v_[x, y] < 0.f ? std::max(hold, stop) : hold);
    }
  }
}

// dust: the burst throws the dust out where it is thick, for its first moments (the explosion's hot gas expands in the
// same way). The dust is heavy, so it falls back and runs out along the ground.
void Fluid::add_atmos_divergence() {
  const float rate = (22.f + 12.f * p_.intensity) * std::exp(-time_ / 0.1f);
  if (rate <= 0.05f) return;
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) div_[x, y] += rate * smoothstep(0.15f, 0.6f, soot_[x, y]);
  }
}

// Reflected light only, no emission: lit from above and to the left, shaded by the material towards the light.
// fog: pale, soft mist, barely shaded; dust: brown-grey and deeply self-shadowed; cloud: white with grey-blue shade.
void Fluid::render_atmos(std::span<std::uint8_t> out) const {
  const int S = p_.size;
  const float n = f(n_), sc = n / f(S);
  constexpr float lx = -0.45f, ly = 0.89f;
  struct Look {
    float density, shade;  // cover 1 - exp(-density D); light exp(-shade tau)
    Rgb lit, dark;         // the colour in full light and in full shade (times the cover)
  };
  const Look look = p_.effect == Effect::fog    ? Look{1.0f, 0.08f, {0.86f, 0.88f, 0.90f}, {0.56f, 0.59f, 0.64f}}
                    : p_.effect == Effect::dust ? Look{2.0f, 0.45f, {0.62f, 0.54f, 0.43f}, {0.19f, 0.17f, 0.14f}}
                                                : Look{2.6f, 0.45f, {0.97f, 0.97f, 0.96f}, {0.42f, 0.45f, 0.52f}};
  for (int py = 0; py < S; ++py) {
    for (int px = 0; px < S; ++px) {
      const float gx = (f(px) + 0.5f) * sc + 0.5f;
      const float gy = n - (f(py) + 0.5f) * sc + 0.5f;
      float tau = 0.f;  // optical depth towards the light
      for (int k = 1; k <= 12; ++k) tau += soot_.sample(gx + lx * 2.f * f(k) * sc, gy + ly * 2.f * f(k) * sc);
      const float light = std::exp(-look.shade * tau), cover = 1.f - std::exp(-look.density * soot_.sample(gx, gy));
      auto o = out.subspan((static_cast<std::size_t>(py) * S + px) * 4, 4);
      o[0] = q8((look.dark.r + (look.lit.r - look.dark.r) * light) * cover);
      o[1] = q8((look.dark.g + (look.lit.g - look.dark.g) * light) * cover);
      o[2] = q8((look.dark.b + (look.lit.b - look.dark.b) * light) * cover);
      o[3] = q8(cover);
    }
  }
}

// --- wildfire, coals and mushroom (docs/EFFECTS_BLAZE.md) ---------------------------------------------------------------
//
// Three presets kept apart from the other effects' code: their sources, forces, divergence, decay and drawing. As for the
// energy effects, every force is a function of the state where it acts, the position, the controls and the run's two
// noise fields (curl_potential, source_flicker); the mushroom, which plays once, also follows the time since its start,
// but only through the two decays a learned one-shot sees as inputs (exp(-t / 0.15 s) and exp(-t / 1 s)).

namespace {

bool blaze_effect(Effect e) { return e == Effect::wildfire || e == Effect::coals || e == Effect::mushroom; }

// wildfire: a burning front along the bottom of the frame, 84% of its width (the ends taper), its base 6% up.
constexpr float kFrontHalfWidth = 0.42f, kFrontBase = 0.06f;

// coals: a bed along the ground, 90% of the frame wide (its ends taper), its top 10 to 18% up and undulating with the
// flicker noise where it is (cells).
constexpr float kBedHalfWidth = 0.45f;
float bed_top(const Params& p, float n, float fl) { return (0.12f + 0.04f * p.intensity) * n * (0.8f + 0.35f * fl); }

// mushroom: the burst's centre and radius (cells), and the height from which the air is stable (the cap stops rising
// there and spreads).
float mushroom_radius(const Params& p, float n) { return (0.075f + 0.04f * p.intensity) * n; }
float mushroom_ceiling(const Params& p, float n) { return (0.55f + 0.1f * p.intensity) * n; }

}  // namespace

bool Fluid::add_blaze_sources(float dt) {
  if (!blaze_effect(p_.effect)) return false;
  const float n = f(n_), s = n / 128.f, I = p_.intensity;
  const auto flick = [&](int x, int y) { return source_flicker(p_, f(x) / s, f(y) / s, time_); };
  const auto psi = [&](int x, int y) { return curl_potential(p_, f(x) / s, f(y) / s, time_); };
  if (p_.effect == Effect::wildfire) {
    // The whole width burns: tongues where the flicker is high, in stronger and weaker stretches that move along the
    // front with the curl noise, so tongues merge and part.
    const float cx = 0.5f * n, hw = kFrontHalfWidth * n, cy = kFrontBase * n, hh = (0.03f + 0.02f * I) * n;
    const float heat = 1.0f + 0.6f * I, lift = (16.f + 18.f * I) * s;
    const int y_lo = std::max(1, static_cast<int>(cy - hh)), y_hi = std::min(n_, static_cast<int>(cy + hh) + 1);
    for (int y = y_lo; y <= y_hi; ++y) {
      for (int x = 1; x <= n_; ++x) {
        const float along = 1.f - smoothstep(0.78f, 1.f, std::abs(f(x) - cx) / hw);
        const float shape = along * (1.f - smoothstep(0.5f, 1.f, std::abs(f(y) - cy) / hh));
        if (shape <= 0.f) continue;
        const float k = shape * (0.35f + 0.65f * smoothstep(-0.25f, 0.6f, flick(x, y))) * (0.6f + 0.4f * psi(x, y));
        if (k <= 0.f) continue;
        temp_[x, y] = std::max(temp_[x, y], heat * k);
        soot_[x, y] += dt * (0.3f + 0.3f * I) * k;
        v_[x, y] = std::max(v_[x, y], lift * k);
      }
    }
    return true;
  }
  if (p_.effect == Effect::coals) {
    // The bed glows in a pattern the flicker noise draws (hot cracks and dull patches that crawl) and that brightens
    // and dims with the curl noise there (the glow pulses); dark char fills it. Where it glows most, short flames lick
    // up from its top.
    const float cx = 0.5f * n, hw = kBedHalfWidth * n, rate = 1.f - std::exp(-5.f * dt);
    const float level = 0.9f + 0.7f * I, lick_heat = 0.8f + 0.5f * I, lick_lift = (14.f + 20.f * I) * s;
    const int y_hi = std::min(n_, static_cast<int>(0.2f * n) + 1);
    for (int y = 1; y <= y_hi; ++y) {
      for (int x = 1; x <= n_; ++x) {
        const float along = 1.f - smoothstep(0.7f, 1.f, std::abs(f(x) - cx) / hw);
        if (along <= 0.f) continue;
        const float fl = flick(x, y), top = bed_top(p_, n, fl);
        const float inside = along * (1.f - smoothstep(0.75f * top, top, f(y)));
        if (inside > 0.f) {
          const float glow = (0.08f + 0.92f * smoothstep(-0.1f, 0.6f, fl)) * (1.f + 0.4f * psi(x, y));
          temp_[x, y] += inside * rate * (level * glow - temp_[x, y]);
          soot_[x, y] += inside * rate * std::max(0.f, 0.9f - soot_[x, y]);
        }
        const float lick = along * smoothstep(0.35f - 0.25f * I, 0.75f - 0.15f * I, fl) * smoothstep(0.7f * top, top, f(y)) *
                           (1.f - smoothstep(top, top + 3.f * s, f(y)));
        if (lick <= 0.f) continue;
        temp_[x, y] = std::max(temp_[x, y], lick_heat * lick);
        soot_[x, y] += dt * 0.1f * lick;
        v_[x, y] = std::max(v_[x, y], lick_lift * lick);
      }
    }
    return true;
  }
  // mushroom: one burst at the start, a fireball on the ground (its edge and lumps from the flicker noise); then the
  // base keeps glowing and throwing up dust, which the draught under the rising fireball draws up as its stem.
  const float cx = 0.5f * n, r = mushroom_radius(p_, n), by = 0.1f * n;
  if (time_ <= 0.f) {
    const int y_hi = std::min(n_, static_cast<int>(by + 1.4f * r) + 1);
    const int x_lo = std::max(1, static_cast<int>(cx - 1.4f * r)), x_hi = std::min(n_, static_cast<int>(cx + 1.4f * r) + 1);
    for (int y = 1; y <= y_hi; ++y) {
      for (int x = x_lo; x <= x_hi; ++x) {
        const float dx = f(x) - cx, dy = f(y) - by, fl = flick(x, y);
        const float fall = 1.f - smoothstep(0.5f * r, r * (1.f + 0.3f * fl), std::sqrt(dx * dx + dy * dy));
        if (fall <= 0.f) continue;
        const float m = 0.75f + 0.25f * fl;
        temp_[x, y] = std::max(temp_[x, y], (2.2f + 0.8f * I) * fall * m);
        soot_[x, y] = std::max(soot_[x, y], (0.3f + 0.2f * I) * fall * m);
        v_[x, y] += 15.f * s * fall;
      }
    }
    return true;
  }
  // the base's heat fades with the second age input's decay (1 s); the dust it throws up, and the draught that lifts it
  // into the stem, keep on at a third of their first strength
  const float hot = std::exp(-time_ / 1.f), lift = 0.3f + 0.7f * hot, ws = (0.06f + 0.03f * I) * n, hs = 0.08f * n;
  const int y_hi = std::min(n_, static_cast<int>(hs) + 1);
  const int x_lo = std::max(1, static_cast<int>(cx - ws)), x_hi = std::min(n_, static_cast<int>(cx + ws) + 1);
  for (int y = 1; y <= y_hi; ++y) {
    for (int x = x_lo; x <= x_hi; ++x) {
      const float shape = (1.f - smoothstep(0.4f * ws, ws, std::abs(f(x) - cx))) * (1.f - smoothstep(0.5f * hs, hs, f(y)));
      const float k = shape * (0.5f + 0.5f * smoothstep(-0.3f, 0.5f, flick(x, y)));
      if (k <= 0.f) continue;
      temp_[x, y] = std::max(temp_[x, y], (1.4f + 0.5f * I) * (0.12f + 0.88f * hot) * k);
      soot_[x, y] += dt * (3.f + 1.f * I) * lift * k;
      v_[x, y] = std::max(v_[x, y], (25.f + 20.f * I) * s * lift * k);
    }
  }
  return true;
}

void Fluid::add_blaze_forces(float dt) {
  if (!blaze_effect(p_.effect)) return;
  const float n = f(n_), s = n / 128.f;
  if (p_.effect == Effect::coals) {  // the bed's air is still: its glow stays where it is, only the flames rise
    const float cx = 0.5f * n, hw = kBedHalfWidth * n;
    const int y_hi = std::min(n_, static_cast<int>(0.2f * n) + 1);
    for (int y = 1; y <= y_hi; ++y) {
      for (int x = 1; x <= n_; ++x) {
        const float along = 1.f - smoothstep(0.7f, 1.f, std::abs(f(x) - cx) / hw);
        if (along <= 0.f) continue;
        const float top = bed_top(p_, n, source_flicker(p_, f(x) / s, f(y) / s, time_));
        const float hold = along * (1.f - smoothstep(0.6f * top, 0.9f * top, f(y)));
        u_[x, y] *= 1.f - hold;
        v_[x, y] *= 1.f - hold;
      }
    }
    return;
  }
  if (p_.effect != Effect::mushroom) return;
  // The wind leans the cloud and shears it: the air drags it towards a wind that grows with height. Above the ceiling
  // the air is stable: hot gas there is pushed back down, so the cap stops rising and spreads.
  const float W = (2.f * p_.wind - 1.f) * 45.f * s, drag = 1.f - std::exp(-1.2f * dt), ceil = mushroom_ceiling(p_, n);
  const float stable = 3.f * tune(p_).buoyancy * s;
  for (int y = 1; y <= n_; ++y) {
    const float shear = W * f(y) / n, above = smoothstep(ceil, ceil + 0.15f * n, f(y));
    for (int x = 1; x <= n_; ++x) {
      const float mat = std::min(1.f, 2.f * (temp_[x, y] + soot_[x, y]));
      u_[x, y] += drag * (shear - u_[x, y]) * mat;
      v_[x, y] -= dt * stable * above * temp_[x, y];
    }
  }
}

// mushroom: the burst's hot gas expands for its first moments, as the explosion's does.
void Fluid::add_blaze_divergence() {
  if (p_.effect != Effect::mushroom) return;
  const float rate = (6.f + 6.f * p_.intensity) * std::exp(-time_ / 0.15f);
  if (rate <= 0.05f) return;
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) div_[x, y] += rate * smoothstep(0.4f, 1.2f, temp_[x, y]);
  }
}

// wildfire and mushroom: the burning gas turns to smoke as it cools.
void Fluid::blaze_decay(float dt) {
  if (p_.effect != Effect::wildfire && p_.effect != Effect::mushroom) return;
  const float k = dt * (p_.effect == Effect::wildfire ? 1.5f : 0.45f);
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) soot_[x, y] += k * temp_[x, y];
  }
}

namespace {

// coals' glow: deep red where it is dull, through red and orange to yellow where it is hottest
Rgb coal_ramp(float t) {
  static constexpr std::array<float, 6> at{0.f, 0.12f, 0.32f, 0.58f, 0.9f, 1.3f};
  static constexpr std::array<Rgb, 6> c{{{0.f, 0.f, 0.f}, {0.28f, 0.02f, 0.f}, {0.78f, 0.12f, 0.01f}, {1.f, 0.42f, 0.05f}, {1.f, 0.72f, 0.22f}, {1.f, 0.92f, 0.62f}}};
  return ramp(at, c, t);
}

}  // namespace

// Emitted light by heat over smoke shaded towards the light (up and to the left). wildfire: fire's colours over grey
// smoke; coals: a glow from deep red to yellow over dark char, whose thin smoke is pale; mushroom: the explosion's
// colours over brown-grey smoke.
bool Fluid::render_blaze(std::span<std::uint8_t> out) const {
  if (!blaze_effect(p_.effect)) return false;
  const int S = p_.size;
  const float n = f(n_), sc = n / f(S);
  constexpr float lx = -0.45f, ly = 0.89f;
  for (int py = 0; py < S; ++py) {
    for (int px = 0; px < S; ++px) {
      const float gx = (f(px) + 0.5f) * sc + 0.5f;
      const float gy = n - (f(py) + 0.5f) * sc + 0.5f;
      const float T = temp_.sample(gx, gy), D = soot_.sample(gx, gy);
      float tau = 0.f;  // optical depth towards the light
      for (int k = 1; k <= 12; ++k) tau += soot_.sample(gx + lx * 2.f * f(k) * sc, gy + ly * 2.f * f(k) * sc);
      const float light = std::exp(-0.5f * tau);
      Rgb c{};
      float a = 0.f;
      if (p_.effect == Effect::coals) {
        const Rgb e = coal_ramp(T);
        const float as = 1.f - std::exp(-2.5f * D), dense = smoothstep(0.15f, 0.6f, D);
        const float g = (0.32f - 0.24f * dense) * (0.45f + 0.55f * light), keep = 1.f - 0.3f * as;
        c = {e.r * keep + g * as, e.g * keep + 0.96f * g * as, e.b * keep + 0.92f * g * as};
        a = clamp01(as + 0.3f * std::max(e.r, e.g));
      } else {
        const bool wild = p_.effect == Effect::wildfire;
        const Rgb e = fire_ramp(wild ? T : 0.75f * T);
        const float as = 1.f - std::exp(-(wild ? 2.2f : 3.f) * D), keep = 1.f - (wild ? 0.55f : 0.6f) * as;
        const float g = (wild ? 0.3f : 0.55f) * (0.3f + 0.7f * light);
        const Rgb tint = wild ? Rgb{1.f, 0.96f, 0.92f} : Rgb{1.f, 0.9f, 0.78f};
        c = {e.r * keep + tint.r * g * as, e.g * keep + tint.g * g * as, e.b * keep + tint.b * g * as};
        a = clamp01(as + 0.25f * std::max(e.r, e.g));
      }
      auto o = out.subspan((static_cast<std::size_t>(py) * S + px) * 4, 4);
      o[0] = q8(c.r);
      o[1] = q8(c.g);
      o[2] = q8(c.b);
      o[3] = q8(a);
    }
  }
  return true;
}

// --- launch, puffs and ink (docs/EFFECTS_FLOW.md) ------------------------------------------------------------------------
//
// Three presets kept apart from the other effects' code: their sources, forces, decay and drawing. As in the energy and
// atmosphere blocks, every force is a function of the state where it acts, the position, the controls and the run's two
// noise fields (curl_potential, source_flicker), which learned dynamics see as inputs; nothing follows a clock of its own.
// launch lies on the atmosphere effects' ground (kGround, the closed bottom border); puffs and ink have none.

namespace {

bool flow_effect(Effect e) { return e == Effect::launch || e == Effect::puffs || e == Effect::ink; }

// Frames simulated before a clip is recorded (simulate): launch's billows take about 4 s to reach the frame's edges,
// ink's dye as long to spread; the effects before them warm up for 90 frames.
int flow_warmup(Effect e) { return flow_effect(e) ? 150 : 90; }

// launch: the nozzle of an engine above the ground, blowing straight down at it.
struct LaunchNozzle {
  float x, y, hw;  // centre and half-width (cells)
};
LaunchNozzle launch_nozzle(const Params& p, float n) { return {0.5f * (n + 1.f), 0.56f * n, (0.035f + 0.02f * p.intensity) * n}; }

// puffs: the mouth of a chimney at the bottom of the frame, a little left of the middle (its trail drifts right). The
// pulses are timed by the flicker noise at the middle of the mouth (a coarse cell's centre, where a learned stepper
// sees exactly that value): a puff whenever its size |f| rises above a level, which the rate control lowers (from
// about 0.5 pulses a second to 2.6, irregularly; docs/EFFECTS_FLOW.md §2).
constexpr float kPuffX = 46.5f, kPuffY = 6.5f;  // the middle of the mouth, in cells of a 128-cell frame
float puff_gate(const Params& p, float time) {
  const float level = 0.40f - 0.26f * p.wind;  // rate: lower, more pulses per second
  return smoothstep(level, level + 0.05f, std::abs(source_flicker(p, kPuffX, kPuffY, time)));
}

// ink: a slow release of dye a quarter of the way up the middle of the frame, aimed by the curl noise where it is (it
// wanders); a coarse cell's centre, as puffs' mouth.
constexpr float kInkX = 62.5f, kInkY = 34.5f;

}  // namespace

bool Fluid::add_flow_sources(float dt) {
  if (!flow_effect(p_.effect)) return false;
  const float n = f(n_), s = n / 128.f, I = p_.intensity;
  const auto flicker = [&](int x, int y) { return source_flicker(p_, f(x) / s, f(y) / s, time_); };
  if (p_.effect == Effect::launch) {  // hot gas blown down at 100 to 160 cells per second, broken up by the flicker
    const LaunchNozzle g = launch_nozzle(p_, n);
    const float hl = 0.04f * n, speed = (100.f + 60.f * I) * s;
    const int x_lo = std::max(1, static_cast<int>(g.x - g.hw - 2.f)), x_hi = std::min(n_, static_cast<int>(g.x + g.hw + 2.f) + 1);
    const int y_lo = std::max(1, static_cast<int>(g.y - hl - 2.f)), y_hi = std::min(n_, static_cast<int>(g.y + hl + 2.f) + 1);
    for (int y = y_lo; y <= y_hi; ++y) {
      for (int x = x_lo; x <= x_hi; ++x) {
        const float ex = (f(x) - g.x) / g.hw, ey = (f(y) - g.y) / hl;
        const float shape = 1.f - smoothstep(0.6f, 1.f, std::sqrt(ex * ex + ey * ey));
        if (shape <= 0.f) continue;
        const float k = shape * (0.45f + 0.55f * smoothstep(-0.3f, 0.5f, flicker(x, y)));
        temp_[x, y] = std::max(temp_[x, y], (1.6f + 0.6f * I) * k);
        soot_[x, y] += dt * (0.3f + 0.3f * I) * k;
        u_[x, y] += k * (0.f - u_[x, y]);
        v_[x, y] += k * (-speed - v_[x, y]);
      }
    }
    return true;
  }
  if (p_.effect == Effect::puffs) {  // warm smoke kicked up out of the mouth while the gate is open
    const float g = puff_gate(p_, time_);
    if (g <= 0.f) return true;
    const float cx = kPuffX * s, cy = kPuffY * s, hw = (0.05f + 0.03f * I) * n, hh = 0.035f * n;
    const float kick = (60.f + 40.f * I) * s;
    const int y_lo = std::max(1, static_cast<int>(cy - 2.f * hh)), y_hi = std::min(n_, static_cast<int>(cy + 2.f * hh) + 1);
    const int x_lo = std::max(1, static_cast<int>(cx - 1.5f * hw)), x_hi = std::min(n_, static_cast<int>(cx + 1.5f * hw) + 1);
    for (int y = y_lo; y <= y_hi; ++y) {
      for (int x = x_lo; x <= x_hi; ++x) {
        const float ex = (f(x) - cx) / hw, ey = (f(y) - cy) / hh;
        const float shape = 1.f - smoothstep(0.6f, 1.f, std::sqrt(ex * ex + ey * ey));
        if (shape <= 0.f) continue;
        const float k = g * shape;
        temp_[x, y] = std::max(temp_[x, y], (0.5f + 0.3f * I) * k);
        soot_[x, y] = std::max(soot_[x, y], (1.2f + 0.8f * I) * g * k);
        v_[x, y] = std::max(v_[x, y], kick * k);
      }
    }
    return true;
  }
  // ink: dye released from a small round source, pushed out slowly in a direction the curl noise there turns
  const float cx = kInkX * s, cy = kInkY * s, r = (0.045f + 0.035f * I) * n;
  const float psi = curl_potential(p_, kInkX, kInkY, time_), a = 1.5707963f + 1.1f * psi;
  const float speed = (12.f + 12.f * I) * s, ux = speed * std::cos(a), uy = speed * std::sin(a);
  const int lo_x = std::max(1, static_cast<int>(cx - 1.5f * r)), hi_x = std::min(n_, static_cast<int>(cx + 1.5f * r) + 1);
  const int lo_y = std::max(1, static_cast<int>(cy - 1.5f * r)), hi_y = std::min(n_, static_cast<int>(cy + 1.5f * r) + 1);
  for (int y = lo_y; y <= hi_y; ++y) {
    for (int x = lo_x; x <= hi_x; ++x) {
      const float dx = f(x) - cx, dy = f(y) - cy;
      const float shape = 1.f - smoothstep(0.5f, 1.f, std::sqrt(dx * dx + dy * dy) / r);
      if (shape <= 0.f) continue;
      const float k = shape * smoothstep(-0.4f, 0.4f, flicker(x, y));
      soot_[x, y] = std::max(soot_[x, y], (1.0f + 0.8f * I) * k);
      u_[x, y] += k * (ux - u_[x, y]);
      v_[x, y] += k * (uy - v_[x, y]);
    }
  }
  return true;
}

// launch: the ground the exhaust hits (as fog's and dust's, add_atmos_forces). puffs: a light, fixed breeze to the right
// that carries the smoke (as the wind of tune() does the other effects').
// ink: the water drifts as a whole with the current (left or right), and is stirred everywhere, not only where the dye
// is.
void Fluid::add_flow_forces(float dt) {
  if (!flow_effect(p_.effect)) return;
  const float n = f(n_), s = n / 128.f;
  if (p_.effect == Effect::launch) {
    const float g = kGround * s, top = g + 4.f * s;
    const int y_hi = std::min(n_, static_cast<int>(top) + 1);
    for (int y = 1; y <= y_hi; ++y) {
      const float hold = 1.f - smoothstep(g - 2.f * s, g + 2.f * s, f(y)), stop = 1.f - smoothstep(g, top, f(y));
      for (int x = 1; x <= n_; ++x) {
        u_[x, y] *= 1.f - hold;
        v_[x, y] *= 1.f - (v_[x, y] < 0.f ? std::max(hold, stop) : hold);
      }
    }
    return;
  }
  if (p_.effect == Effect::puffs) {  // a light breeze to the right, carrying the smoke
    const float drift = 14.f * s, drag = 1.f - std::exp(-1.0f * dt);
    for (int y = 1; y <= n_; ++y) {
      for (int x = 1; x <= n_; ++x) u_[x, y] += drag * (drift - u_[x, y]) * std::min(1.f, 2.f * (temp_[x, y] + soot_[x, y]));
    }
    return;
  }
  const float current = (2.f * p_.wind - 1.f) * 22.f * s, drag = 1.f - std::exp(-0.7f * dt);
  const float L = 14.f * s, e = 0.25f, tt = time_ * 0.8f, amp = (2.f + 6.f * p_.turbulence) * s;
  const std::uint64_t cseed = p_.seed * 0x2545F4914F6CDD1DULL + 77;  // as curl_potential
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) {
      u_[x, y] += drag * (current - u_[x, y]);
      const float px = f(x) / L, py = f(y) / L;
      u_[x, y] += dt * amp * (value_noise(px, py + e, tt, cseed) - value_noise(px, py - e, tt, cseed)) / (2.f * e);
      v_[x, y] -= dt * amp * (value_noise(px + e, py, tt, cseed) - value_noise(px - e, py, tt, cseed)) / (2.f * e);
    }
  }
}

// launch: the exhaust turns to smoke and steam as it cools (as the jet's burning gas does), grey-white clouds.
void Fluid::flow_decay(float dt) {
  if (p_.effect != Effect::launch) return;
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) soot_[x, y] += dt * 2.2f * temp_[x, y];
  }
}

bool Fluid::render_flow(std::span<std::uint8_t> out) const {
  if (!flow_effect(p_.effect)) return false;
  const int S = p_.size;
  const float n = f(n_), sc = n / f(S);
  constexpr float lx = -0.45f, ly = 0.89f;  // towards the light: up and to the left
  for (int py = 0; py < S; ++py) {
    for (int px = 0; px < S; ++px) {
      const float gx = (f(px) + 0.5f) * sc + 0.5f;
      const float gy = n - (f(py) + 0.5f) * sc + 0.5f;
      const float T = temp_.sample(gx, gy), D = soot_.sample(gx, gy);
      float tau = 0.f;  // optical depth towards the light
      for (int k = 1; k <= 12; ++k) tau += soot_.sample(gx + lx * 2.f * f(k) * sc, gy + ly * 2.f * f(k) * sc);
      Rgb c{};
      float a = 0.f;
      if (p_.effect == Effect::launch) {  // the exhaust's light over grey-white clouds
        const Rgb e = jet_ramp(1.1f * T);
        const float as = 1.f - std::exp(-2.2f * D), keep = 1.f - 0.5f * as, g = 0.97f * (0.6f + 0.4f * std::exp(-0.2f * tau));
        c = {e.r * keep + g * as, e.g * keep + 0.97f * g * as, e.b * keep + 0.95f * g * as};
        a = clamp01(as + 0.25f * std::max(e.r, e.g));
      } else if (p_.effect == Effect::puffs) {  // grey smoke, lit from above and shaded by itself
        const float as = 1.f - std::exp(-2.5f * D), g = 0.86f * (0.4f + 0.6f * std::exp(-0.35f * tau));
        c = {g * as, 0.98f * g * as, 0.95f * g * as};
        a = as;
      } else {  // ink: dark, barely shaded, thin where it is thin
        const float as = 1.f - std::exp(-1.6f * D), g = 0.6f + 0.4f * std::exp(-0.2f * tau);
        c = {0.10f * g * as, 0.11f * g * as, 0.16f * g * as};
        a = as;
      }
      auto o = out.subspan((static_cast<std::size_t>(py) * S + px) * 4, 4);
      o[0] = q8(c.r);
      o[1] = q8(c.g);
      o[2] = q8(c.b);
      o[3] = q8(a);
    }
  }
  return true;
}

// --- waterfall, fountain, splash and rainshaft (docs/EFFECTS_WATER.md) ------------------------------------------------
//
// The solver has no liquid with a free surface, so water is made of the channels it has. Water is the solver's soot:
// heavy, so it falls under its weight (as dust does), and without heat. Spray and mist are its heat: light, so they
// rise (as steam's vapour does) and fade (the cooling is the mist's). Water breaks into spray where it falls fast and
// turns into foam and mist where it lands: on a pool (waterfall, fountain, splash) or the ground (rainshaft) at the
// bottom of the frame, over a closed bottom border as fog's and dust's ground. Every force is a function of the state
// where it acts, the position, the controls and the run's two noise fields (curl_potential, source_flicker); the
// splash's crown and its jet also of its age, as the explosion's expansion is.

namespace {

constexpr float kPool = 7.f;  // the pool's surface (the ground under the rain), in cells of a 128-cell frame

// The fountain's nozzle: on the pool in the middle of the frame, aimed from 12 degrees left of upright to 12 degrees
// right (the second control, aim).
struct Nozzle {
  float x, y, sx, cy;  // position (cells), direction (sine and cosine of the angle from upright)
};
Nozzle fountain_nozzle(const Params& p, float n) {
  const float a = (-12.f + 24.f * p.wind) * 0.017453293f;
  return {0.5f * (n + 1.f), (kPool + 2.f) * n / 128.f, std::sin(a), std::cos(a)};
}
// How fast the fountain's water leaves the nozzle (cells per second; s = cells per cell of a 128-cell frame), and the
// pull that slows it (cells per second squared): its arc peaks 45 to 95 cells above the nozzle (intensity: height).
constexpr float kFountainGravity = 70.f;
float fountain_speed(const Params& p, float s) { return (80.f + 35.f * p.intensity) * s; }

}  // namespace

// waterfall: the lip, a band across the top of the frame where the water comes over the ledge and pours down, broken
// into streaks by the flicker noise. fountain: a jet of water from the nozzle. splash: the crown, held up for half a
// second, then a jet of water thrown up from the middle of the crater. rainshaft: rain from the base of the cloud,
// across the top of the frame, thick where the flicker noise is high: shafts that thin and thicken.
bool Fluid::add_water_sources(float dt) {
  (void)dt;
  if (!water(p_.effect)) return false;
  const float n = f(n_), s = n / 128.f, I = p_.intensity, c = 0.5f * (n + 1.f);
  const auto flicker = [&](int x, int y) { return source_flicker(p_, f(x) / s, f(y) / s, time_); };
  const auto box = [&](float x0, float x1, float y0, float y1, auto&& fn) {
    const int xl = std::max(1, static_cast<int>(x0)), xh = std::min(n_, static_cast<int>(x1) + 1);
    const int yl = std::max(1, static_cast<int>(y0)), yh = std::min(n_, static_cast<int>(y1) + 1);
    for (int y = yl; y <= yh; ++y) {
      for (int x = xl; x <= xh; ++x) fn(x, y);
    }
  };
  if (p_.effect == Effect::waterfall) {
    const float cy = 0.90f * n, hw = (0.07f + 0.09f * I) * n, hh = 0.025f * n;
    const float level = 0.9f + 0.5f * I, pour = (60.f + 30.f * I) * s;
    box(c - 1.2f * hw, c + 1.2f * hw, cy - hh, cy + hh, [&](int x, int y) {
      const float shape = (1.f - smoothstep(0.75f, 1.f, std::abs(f(x) - c) / hw)) * (1.f - smoothstep(0.5f, 1.f, std::abs(f(y) - cy) / hh));
      if (shape <= 0.f) return;
      const float k = shape * (0.15f + 0.85f * smoothstep(-0.3f, 0.35f, flicker(x, y)));
      soot_[x, y] = std::max(soot_[x, y], level * k);
      v_[x, y] = std::min(v_[x, y], -pour * k);
      u_[x, y] *= 1.f - shape;
    });
    return true;
  }
  if (p_.effect == Effect::fountain) {
    const Nozzle z = fountain_nozzle(p_, n);
    const float hl = 0.035f * n, hw = (0.03f + 0.015f * I) * n, speed = fountain_speed(p_, s), ext = hl + hw + 2.f;
    box(z.x - ext, z.x + ext, z.y - ext, z.y + ext, [&](int x, int y) {
      const float dx = f(x) - z.x, dy = f(y) - z.y;
      const float ea = (dx * z.sx + dy * z.cy) / hl, ec = (dx * z.cy - dy * z.sx) / hw;
      const float shape = 1.f - smoothstep(0.6f, 1.f, std::sqrt(ea * ea + ec * ec));
      if (shape <= 0.f) return;
      const float k = shape * (0.7f + 0.3f * smoothstep(-0.4f, 0.5f, flicker(x, y)));
      soot_[x, y] = std::max(soot_[x, y], (1.0f + 0.3f * I) * k);
      u_[x, y] += k * (speed * z.sx - u_[x, y]);
      v_[x, y] += k * (speed * z.cy - v_[x, y]);
    });
    return true;
  }
  if (p_.effect == Effect::splash) {
    // The crown: for its first half second the impact holds up two walls of water (a ring seen from the side) leaning
    // out from the crater's rim, growing to their height, thinning towards their ragged top, where the flicker noise
    // breaks them into fingers and drops; their water moves up and out. Then they are let go and fall back.
    const float g = kPool * s, t = time_;
    const float grow = smoothstep(0.f, 0.4f, t), hold = 1.f - smoothstep(0.42f, 0.5f, t);
    if (hold > 0.f) {
      const float r0 = (0.05f + 0.03f * I) * n, H = (0.22f + 0.28f * I) * n * (0.25f + 0.75f * grow);
      const float w = (0.04f + 0.02f * I) * n, lean = 0.5f, speed = (40.f + 40.f * I) * s * (1.f - grow);
      const float level = 1.2f + 0.4f * I;
      box(c - r0 - H * lean - 2.f * w, c + r0 + H * lean + 2.f * w, g, g + 1.2f * H, [&](int x, int y) {
        const float h = (f(y) - g) / H, fl = flicker(x, y), side = f(x) < c ? -1.f : 1.f;
        const float top = 1.f - smoothstep(0.6f + 0.3f * fl, 1.f + 0.3f * fl, h);  // a ragged top
        const float off = std::abs(std::abs(f(x) - c) - (r0 + lean * (f(y) - g))) / (w * (1.f - 0.5f * clamp01(h)));
        const float wall = (1.f - smoothstep(0.5f, 1.f, off)) * top * hold;
        if (wall <= 0.f) return;
        const float m = level * wall * (0.35f + 0.65f * smoothstep(-0.3f, 0.3f, fl));  // broken into fingers and drops
        soot_[x, y] = std::max(soot_[x, y], m);
        temp_[x, y] = std::max(temp_[x, y], 0.4f * m * clamp01(h));  // spray, more towards its top
        u_[x, y] += wall * (side * lean * speed - u_[x, y]);
        v_[x, y] += wall * (speed - v_[x, y]);
      });
    }
    // then the jet from the middle of the crater (a Worthington jet), from 0.45 to 0.85 seconds
    const float on = smoothstep(0.42f, 0.5f, t) * (1.f - smoothstep(0.75f, 0.85f, t));
    if (on <= 0.f) return true;
    const float hw = (0.02f + 0.012f * I) * n, hh = 0.03f * n, cy = g + hh, speed = (55.f + 45.f * I) * s;
    box(c - hw, c + hw, cy - hh, cy + hh, [&](int x, int y) {
      const float ex = (f(x) - c) / hw, ey = (f(y) - cy) / hh;
      const float shape = 1.f - smoothstep(0.5f, 1.f, std::sqrt(ex * ex + ey * ey));
      if (shape <= 0.f) return;
      const float k = on * shape * (0.6f + 0.4f * smoothstep(-0.4f, 0.5f, flicker(x, y)));
      soot_[x, y] = std::max(soot_[x, y], (0.9f + 0.4f * I) * k);
      v_[x, y] = std::max(v_[x, y], speed * k);
    });
    return true;
  }
  // rainshaft
  const float cy = 0.93f * n, hw = (0.12f + 0.14f * I) * n, hh = 0.04f * n;
  const float level = 0.40f + 0.50f * I, fall = (70.f + 30.f * I) * s;
  box(c - 1.2f * hw, c + 1.2f * hw, cy - hh, cy + hh, [&](int x, int y) {
    const float shape = (1.f - smoothstep(0.7f, 1.f, std::abs(f(x) - c) / hw)) * (1.f - smoothstep(0.5f, 1.f, std::abs(f(y) - cy) / hh));
    if (shape <= 0.f) return;
    const float k = shape * (0.1f + 0.9f * smoothstep(-0.2f, 0.4f, flicker(x, y)));  // shafts, thick and thin
    soot_[x, y] = std::max(soot_[x, y], level * k);
    v_[x, y] = std::min(v_[x, y], -fall * k);
  });
  return true;
}

// Falling water (waterfall, rainshaft) falls at its terminal speed: its velocity is drawn towards straight down at that
// speed (the rain's towards the wind as well), so a curtain stays a curtain and its streaks are carried down whole. The
// wind blows the mist and spray (waterfall, splash) or the rain and its mist (rainshaft), not the falling water; the
// rain's wind also swings with the curl noise where it is (gusts, more with turbulence), so the shaft sways. The pool
// (the ground under the rain): what comes down onto it stops and spreads sideways (the closed bottom border of the
// pressure solve does the rest), as on fog's and dust's ground; over the waterfall's pool the air moves with the wind.
void Fluid::add_water_forces(float dt) {
  if (!water(p_.effect)) return;
  const float n = f(n_), s = n / 128.f, g = kPool * s, top = g + 5.f * s, I = p_.intensity;
  const bool rain = p_.effect == Effect::rainshaft, falls = rain || p_.effect == Effect::waterfall;
  const float fall = rain ? (80.f + 30.f * I) * s : (85.f + 30.f * I) * s;  // terminal speed, cells per second
  const float relax = falls ? 1.f - std::exp(-(rain ? 6.f : 14.f) * dt) : 0.f;
  const float rough = p_.effect == Effect::waterfall ? 1.f - std::exp(-4.f * dt) : 0.f;  // the air over the pool moves with the wind
  const float drift = p_.effect == Effect::fountain ? 0.f : (2.f * p_.wind - 1.f) * (rain ? 40.f : 60.f) * s;
  const float drag = p_.effect == Effect::fountain ? 0.f : 1.f - std::exp(-(rain ? 1.5f : 4.f) * dt);
  const float sway = rain ? (6.f + 30.f * p_.turbulence) * s : 0.f;
  // water thrown up (fountain, splash) rises against its weight and, once it falls, falls at its terminal speed; the
  // bare air around it keeps little momentum of its own, so it does not carry the water round in eddies
  const bool thrown = p_.effect == Effect::fountain || p_.effect == Effect::splash;
  const float drop = (p_.effect == Effect::fountain ? 55.f + 20.f * I : 60.f + 20.f * I) * s, settle = thrown ? 1.f - std::exp(-3.f * dt) : 0.f;
  const float calm = thrown ? 1.f - std::exp(-(p_.effect == Effect::splash ? 4.f : 1.5f) * dt) : 0.f;
  // the fountain's water follows its arc: rising, its speed is drawn towards what a drop thrown from the nozzle has at
  // that height; past the top it falls, spreading to either side of where the arc peaks
  const bool arc = p_.effect == Effect::fountain;
  const Nozzle z = fountain_nozzle(p_, n);
  const float V = fountain_speed(p_, s), G = kFountainGravity * s, vx = V * z.sx, vy = V * z.cy;
  const float peak_x = z.x + vx * vy / G, spread = (12.f + 12.f * p_.turbulence) * s, follow = arc ? 1.f - std::exp(-6.f * dt) : 0.f;
  const float scatter = (8.f + 24.f * p_.turbulence) * s;  // the falling spray is scattered by the curl noise
  // the mist churns: curl noise of its own (on top of tune()'s, which stirs all material alike), more with turbulence
  const float churn = rain ? 0.f : (10.f + 40.f * p_.turbulence) * s, L = 14.f * s, e = 0.25f, tt = time_ * 0.8f;
  const std::uint64_t cseed = p_.seed * 0x2545F4914F6CDD1DULL + 77;  // as curl_potential
  for (int y = 1; y <= n_; ++y) {
    const float hold = 1.f - smoothstep(g - 2.f * s, g + s, f(y)), stop = 1.f - smoothstep(g, top, f(y));
    const float above = smoothstep(g + 2.f * s, g + 10.f * s, f(y));  // away from the pool, where water still falls
    const float near = rough * (1.f - smoothstep(g + 4.f * s, g + 14.f * s, f(y)));
    for (int x = 1; x <= n_; ++x) {
      const float T = temp_[x, y], D = soot_[x, y];
      const float wind = drift + (sway > 0.f ? sway * curl_potential(p_, f(x) / s, f(y) / s, time_) : 0.f);
      if (relax > 0.f && D > 0.f) {
        const float w = relax * std::min(1.f, 4.f * D) * above;
        u_[x, y] += w * ((rain ? wind : 0.f) - u_[x, y]);
        v_[x, y] += w * (-fall - v_[x, y]);
      }
      if (follow > 0.f && D + T > 0.f) {  // rising (b = 1) or falling (b = 0), blended around the top of the arc; the spray
                                          // falls with the water
        const float b = smoothstep(-0.3f * drop, 0.3f * drop, v_[x, y]);
        const float w = follow * std::min(1.f, 4.f * D + 2.f * (1.f - b) * T) * above;
        const float fu = 0.3f * vx + spread * std::clamp((f(x) - peak_x) / (0.06f * n), -1.f, 1.f) + scatter * curl_potential(p_, f(x) / s, f(y) / s, time_);
        const float ru = vx, rv = std::sqrt(std::max(0.f, vy * vy - 2.f * G * (f(y) - z.y)));
        u_[x, y] += w * (fu + b * (ru - fu) - u_[x, y]);
        v_[x, y] += w * (-drop + b * (rv + drop) - v_[x, y]);
      } else if (settle > 0.f && D > 0.f && v_[x, y] < 0.f) {
        v_[x, y] += settle * std::min(1.f, 4.f * D) * above * (-drop - v_[x, y]);
      }
      if (calm > 0.f) {
        const float bare = calm * (1.f - std::min(1.f, 4.f * (D + T)));
        u_[x, y] *= 1.f - bare;
        v_[x, y] *= 1.f - bare;
      }
      const float m = rain ? std::min(1.f, 3.f * T) : std::min(1.f, 3.f * T) * (1.f - std::min(1.f, 0.7f * D));
      if (drag > 0.f && m > 0.f) u_[x, y] += drag * m * (wind - u_[x, y]);
      if (churn > 0.f && m > 0.f) {
        const float px = f(x) / L, py = f(y) / L, a = dt * churn * m;
        u_[x, y] += a * (value_noise(px, py + e, tt, cseed) - value_noise(px, py - e, tt, cseed)) / (2.f * e);
        v_[x, y] -= a * (value_noise(px + e, py, tt, cseed) - value_noise(px - e, py, tt, cseed)) / (2.f * e);
      }
      if (near > 0.f) u_[x, y] += near * (drift - u_[x, y]);
      if (f(y) <= top && v_[x, y] < 0.f) v_[x, y] *= 1.f - std::max(hold, stop);
    }
  }
}

// Water breaks up into spray as it falls: the waterfall's curtain more the further down it is (and with turbulence) and
// where it is thrown sideways, the fountain's and the splash's water at the top of its arc and as it falls back. The
// spray is heat, so it whitens the water it is in. Water that reaches the pool or the ground turns into foam and mist.
void Fluid::water_decay(float dt) {
  if (!water(p_.effect)) return;
  const float n = f(n_), s = n / 128.f, g = kPool * s;
  struct Turn {
    float spray, v0, v1;   // spray per second per unit of water, from falling speed v0 to v1 (cells per second)
    float land, reach, mist;  // how fast water near the pool turns over (per second), within how many cells of it, and
                              // the mist it gives per unit
  };
  const float turb = p_.turbulence;
  const Turn t = p_.effect == Effect::waterfall ? Turn{0.25f + 0.5f * turb, 0.f, 0.f, 12.f, 8.f, 1.2f}
                 : p_.effect == Effect::fountain ? Turn{0.f, 0.f, 20.f, 4.f, 6.f, 0.8f}
                 : p_.effect == Effect::splash   ? Turn{0.5f, 20.f, 90.f, 3.f, 6.f, 1.0f}
                                                 : Turn{0.f, 0.f, 0.f, 6.f, 6.f, 0.4f};
  const bool curtain = p_.effect == Effect::waterfall;
  for (int y = 1; y <= n_; ++y) {
    const float plunge = 1.f - smoothstep(g, g + t.reach * s, f(y));
    const float turn = plunge > 0.f ? 1.f - std::exp(-t.land * dt * plunge) : 0.f;
    const float low = curtain ? smoothstep(0.85f * n, 0.25f * n, f(y)) : 0.f;  // how far down the curtain
    for (int x = 1; x <= n_; ++x) {
      const float D = soot_[x, y];
      if (D <= 0.f) continue;
      if (turn > 0.f) {
        soot_[x, y] -= turn * D;
        temp_[x, y] += t.mist * turn * D;
      }
      if ((p_.effect == Effect::fountain || p_.effect == Effect::splash) && v_[x, y] < 15.f * s) {  // top of the arc, falling
        const float c = D * (1.f - std::exp(-dt * 1.5f * smoothstep(15.f * s, -15.f * s, v_[x, y])));
        soot_[x, y] -= c;
        temp_[x, y] += c;
      }
      if (curtain) {  // the curtain breaks up: water turns into spray lower down, and where it is thrown sideways
        const float c = D * (1.f - std::exp(-dt * (t.spray * low + 1.5f * smoothstep(8.f * s, 40.f * s, std::abs(u_[x, y])))));
        soot_[x, y] -= c;
        temp_[x, y] += c;
      } else if (t.spray > 0.f && v_[x, y] < 0.f) {  // the splash's falling water sprays as well
        temp_[x, y] += dt * t.spray * D * smoothstep(t.v0 * s, t.v1 * s, -v_[x, y]);
      }
    }
  }
}

// Reflected light only, no emission, lit from above and to the left. Water is translucent, blue-white, lit where the
// light reaches it and bluer in its own shade; spray in it makes it white and opaque (foam); mist is soft and white.
// The rain is grey-blue and its mist pale grey.
bool Fluid::render_water(std::span<std::uint8_t> out) const {
  if (!water(p_.effect)) return false;
  const int S = p_.size;
  const float n = f(n_), sc = n / f(S);
  constexpr float lx = -0.45f, ly = 0.89f;
  struct Look {
    float density, clear, shading;  // water's cover 1 - exp(-density D); its alpha where clear; light exp(-shading tau)
    Rgb lit, shade, foam;           // the water's colour in full light and in its shade, and the foam's
    float mist_density;             // mist's cover 1 - exp(-mist_density T)
    Rgb mist;
  };
  const Look look = p_.effect == Effect::rainshaft
                        ? Look{2.2f, 0.65f, 0.10f, {0.70f, 0.76f, 0.84f}, {0.40f, 0.47f, 0.58f}, {0.82f, 0.86f, 0.90f}, 2.0f, {0.76f, 0.80f, 0.85f}}
                        : Look{1.6f, 0.70f, 0.10f, {0.92f, 0.97f, 1.00f}, {0.30f, 0.50f, 0.66f}, {0.97f, 0.99f, 1.00f}, 2.5f, {0.92f, 0.95f, 0.98f}};
  for (int py = 0; py < S; ++py) {
    for (int px = 0; px < S; ++px) {
      const float gx = (f(px) + 0.5f) * sc + 0.5f;
      const float gy = n - (f(py) + 0.5f) * sc + 0.5f;
      const float T = temp_.sample(gx, gy), D = soot_.sample(gx, gy);
      float tau = 0.f;  // optical depth of the water towards the light
      for (int k = 1; k <= 12; ++k) tau += soot_.sample(gx + lx * 2.f * f(k) * sc, gy + ly * 2.f * f(k) * sc);
      const float light = std::exp(-look.shading * tau);
      const float cw = 1.f - std::exp(-look.density * D), froth = 1.f - std::exp(-4.f * T);
      const float aw = cw * (look.clear + (1.f - look.clear) * froth);
      const float b = (0.3f + 0.7f * cw) * (0.75f + 0.25f * light);  // thick water is bright, thin water shows its tint
      const Rgb w{look.shade.r + (look.lit.r - look.shade.r) * b, look.shade.g + (look.lit.g - look.shade.g) * b,
                  look.shade.b + (look.lit.b - look.shade.b) * b};
      const Rgb cwc{w.r + (look.foam.r - w.r) * froth, w.g + (look.foam.g - w.g) * froth, w.b + (look.foam.b - w.b) * froth};
      const float am = 0.85f * (1.f - std::exp(-look.mist_density * T)) * (1.f - aw), gm = 0.8f + 0.2f * light;
      auto o = out.subspan((static_cast<std::size_t>(py) * S + px) * 4, 4);
      o[0] = q8(cwc.r * aw + look.mist.r * gm * am);
      o[1] = q8(cwc.g * aw + look.mist.g * gm * am);
      o[2] = q8(cwc.b * aw + look.mist.b * gm * am);
      o[3] = q8(aw + am);
    }
  }
  return true;
}

// --- lava, eruption and avalanche (docs/EFFECTS_EARTH.md) ------------------------------------------------------------
//
// Three presets kept apart from the other effects' code: their sources, forces, decay and drawing. As in the other
// blocks, every force is a function of the state where it acts, the position, the controls and the run's two noise
// fields (curl_potential, source_flicker), which learned dynamics see as inputs; the avalanche, which plays once, also
// follows the time since its start, but only through the two decays a learned one-shot sees (exp(-t / 0.15 s) and
// exp(-t / 1 s)). Nothing follows a clock of its own. All three lie on the ground: the closed bottom border of the
// pressure solve (project).
//   lava: molten rock is the solver's soot (heavy, very viscous) and its glow is its heat; the crust is rock that has
//         cooled. A band of lava along the ground, kept full by what wells up from below, flows along the slope (the
//         second control, `flow`) or pools where the ground is level; its glow is drawn to a pattern of plates and
//         seams that the curl noise moves and the flow carries; bubbles of hot lava well up where the curl noise is
//         high (more with turbulence).
//   eruption: hot ash (heat and soot) blown up from a vent; it rises while it is hot and spreads into an umbrella under
//         a stable layer; the ash is heavy, so what has cooled falls back down the column's sides and runs out along
//         the ground. A sheared wind leans it.
//   avalanche: snow is soot without heat. A slab breaks loose on the upslope side in its first moments, runs down the
//         slope (the second control, `direction`: left, level or right), picks up snow from the ground as it goes and
//         is lifted into a billowing cloud by its own speed; where the slope runs out it slows and settles.

namespace {

// lava: the band's surface (cells): 13 to 20% of the frame high (intensity), deeper where it pools (a level slope, and
// downhill along a slope), lumpy where the flicker noise is.
constexpr float kLavaHalfWidth = 0.48f;
float lava_slope(const Params& p) { return 2.f * p.wind - 1.f; }  // -1: down to the left, 0 level, 1: down to the right
// Per column of the band (X in cells of a 128-cell frame): its surface (cells of an n-cell grid) and how strongly a
// bubble wells up there. Both follow the flicker noise along a row at the band's height (a coarse cell's centre, 18.5),
// smoothed over a coarse cell either side, so its lumps and bubbles are about a coarse cell wide or more.
struct LavaColumn {
  float top, bubble;
};
LavaColumn lava_column(const Params& p, float n, float X, float time) {
  constexpr float row = 18.5f;
  const float fl = 0.25f * source_flicker(p, X - 4.f, row, time) + 0.5f * source_flicker(p, X, row, time) +
                   0.25f * source_flicker(p, X + 4.f, row, time);
  const float s = lava_slope(p), along = std::clamp((X - 64.f) / (kLavaHalfWidth * 128.f), -1.f, 1.f);
  const float top = (0.13f + 0.07f * p.intensity) * n * (1.1f - 0.25f * std::abs(s) + 0.25f * s * along) * (0.9f + 0.3f * fl);
  const float level = 0.42f - 0.22f * p.turbulence;
  return {top, smoothstep(level, level + 0.1f, fl)};
}

// eruption: the vent at the bottom of the middle of the frame, and the height of the stable layer (the umbrella)
float eruption_ceiling(const Params& p, float n) { return (0.55f + 0.2f * p.intensity) * n; }

// avalanche: the slope (-1: running left, 1: right) and where the slab breaks loose (cells), upslope of the middle
float avalanche_slope(const Params& p) { return 2.f * p.wind - 1.f; }
float avalanche_release(const Params& p, float n) { return 0.5f * n - 0.32f * n * avalanche_slope(p); }
// How fast the slope carries snow at x (cells of a 128-cell frame per second, signed): 25 to 55 (intensity) where the
// slope is steep (most of the control's range: the speed saturates), running out over 45 to 80% of the way from the
// release.
float avalanche_flow(const Params& p, float n, float x) {
  const float sl = avalanche_slope(p), run = (x - avalanche_release(p, n)) * (sl >= 0.f ? 1.f : -1.f) / n;
  return (25.f + 30.f * p.intensity) * std::tanh(3.f * sl) * (1.f - smoothstep(0.45f, 0.8f, run));
}

}  // namespace

bool Fluid::add_earth_sources(float dt) {
  if (!earth(p_.effect)) return false;
  const float n = f(n_), s = n / 128.f, I = p_.intensity;
  const auto flick = [&](int x, int y) { return source_flicker(p_, f(x) / s, f(y) / s, time_); };
  const auto psi = [&](int x, int y) { return curl_potential(p_, f(x) / s, f(y) / s, time_); };
  if (p_.effect == Effect::lava) {
    // The band is kept full of rock (drawn towards a density of 1), and its heat is drawn slowly towards a pattern: hot
    // seams between dark plates of crust, where the curl noise is near its middle, brighter at intensity; the bottom of
    // the band stays a little hotter. The draw is slow, so the flow carries the pattern before it fades.
    const float cx = 0.5f * n, hw = kLavaHalfWidth * n, rate_d = 1.f - std::exp(-3.f * dt), rate_t = 1.f - std::exp(-1.2f * dt);
    const float hot = 0.8f + 0.6f * I;
    const int y_hi = std::min(n_, static_cast<int>(0.3f * n) + 1);
    for (int x = 1; x <= n_; ++x) {
      const float along = 1.f - smoothstep(0.85f, 1.f, std::abs(f(x) - cx) / hw);
      if (along <= 0.f) continue;
      const float top = lava_column(p_, n, f(x) / s, time_).top;
      for (int y = 1; y <= y_hi; ++y) {
        const float inside = along * (1.f - smoothstep(0.8f * top, top, f(y)));
        if (inside <= 0.f) continue;
        soot_[x, y] += inside * rate_d * (1.f - soot_[x, y]);
        const float seam = 1.f - smoothstep(0.05f, 0.22f, std::abs(flick(x, y)));
        const float glow = hot * (0.03f + 0.97f * seam * (0.55f + 0.45f * smoothstep(-0.6f, 0.6f, psi(x, y))));
        temp_[x, y] += inside * rate_t * (glow - temp_[x, y]);
      }
    }
    return true;
  }
  if (p_.effect == Effect::eruption) {
    // hot ash blown up out of the vent, broken up by the flicker noise: a hot core, and a rim of denser, cooler ash that
    // the blast throws up but its heat cannot carry (it falls back down the column's sides)
    const float cx = 0.5f * (n + 1.f), cy = 0.05f * n, hw = (0.06f + 0.03f * I) * n, hh = 0.04f * n;
    const float speed = (50.f + 25.f * I) * s, heat = 1.5f + 0.5f * I, ash = 2.0f + 1.0f * I;
    const int y_lo = std::max(1, static_cast<int>(cy - 2.f * hh)), y_hi = std::min(n_, static_cast<int>(cy + 2.f * hh) + 1);
    const int x_lo = std::max(1, static_cast<int>(cx - 1.5f * hw)), x_hi = std::min(n_, static_cast<int>(cx + 1.5f * hw) + 1);
    for (int y = y_lo; y <= y_hi; ++y) {
      for (int x = x_lo; x <= x_hi; ++x) {
        const float ex = (f(x) - cx) / hw, ey = (f(y) - cy) / hh;
        const float shape = 1.f - smoothstep(0.6f, 1.f, std::sqrt(ex * ex + ey * ey));
        if (shape <= 0.f) continue;
        const float k = shape * (0.45f + 0.55f * smoothstep(-0.3f, 0.5f, flick(x, y))), rim = smoothstep(0.35f, 0.8f, std::abs(ex));
        temp_[x, y] = std::max(temp_[x, y], heat * k * (1.f - 0.75f * rim));
        soot_[x, y] = std::max(soot_[x, y], ash * k * (1.f + 0.3f * rim));
        v_[x, y] = std::max(v_[x, y], speed * k);
      }
    }
    return true;
  }
  // avalanche: the slab breaks loose over its first moments (the first age input's decay, 0.15 s), a mound of snow on
  // the ground upslope of the middle, its edge from the flicker noise; it starts moving down the slope at once
  const float release = std::exp(-time_ / 0.15f) / 0.15f;
  if (release < 1e-3f) return true;
  const float cx = avalanche_release(p_, n), a = (0.10f + 0.06f * I) * n, b = (0.08f + 0.05f * I) * n;
  const float level = 2.f + 1.f * I, slide = 35.f * s * avalanche_slope(p_);
  const int y_hi = std::min(n_, static_cast<int>(1.6f * b) + 1);
  const int x_lo = std::max(1, static_cast<int>(cx - 1.6f * a)), x_hi = std::min(n_, static_cast<int>(cx + 1.6f * a) + 1);
  for (int y = 1; y <= y_hi; ++y) {
    for (int x = x_lo; x <= x_hi; ++x) {
      const float ex = (f(x) - cx) / a, ey = f(y) / b, fl = flick(x, y);
      const float shape = 1.f - smoothstep(0.5f, 1.f + 0.35f * fl, std::sqrt(ex * ex + ey * ey));
      if (shape <= 0.f) continue;
      const float k = dt * release * shape;
      soot_[x, y] += k * level * (0.75f + 0.25f * smoothstep(-0.5f, 0.5f, fl));
      u_[x, y] += std::min(1.f, 6.f * k) * (slide - u_[x, y]);
    }
  }
  return true;
}

void Fluid::add_earth_forces(float dt) {
  if (!earth(p_.effect)) return;
  const float n = f(n_), s = n / 128.f, I = p_.intensity;
  if (p_.effect == Effect::lava) {
    // The lava flows along the slope: its velocity is drawn (strongly: it is viscous) towards a flow that is still at
    // the ground and fastest at its surface, 6 to 20 cells per second at the steepest slope (intensity); it does not
    // move up and down, except where a bubble of hot lava wells up (the curl noise above a level that turbulence
    // lowers), which lifts the surface into a glowing dome that falls back as the noise moves on.
    const float slope = lava_slope(p_), U = slope * (6.f + 14.f * I) * s, visc = 1.f - std::exp(-4.f * dt);
    const float lift = (10.f + 12.f * p_.turbulence) * s, fall = 15.f * s;
    const int y_hi = std::min(n_, static_cast<int>(0.35f * n) + 1);
    for (int x = 1; x <= n_; ++x) {
      const auto [top, welling] = lava_column(p_, n, f(x) / s, time_);
      const float bubble = welling * (1.f - smoothstep(0.75f, 0.95f, std::abs(f(x) - 0.5f * n) / (kLavaHalfWidth * n)));  // in the band
      for (int y = 1; y <= y_hi; ++y) {
        const float D = soot_[x, y];
        if (D <= 0.01f) continue;
        const float h = std::clamp(f(y) / top, 0.f, 1.f), m = std::min(1.f, 2.f * D);
        u_[x, y] += m * visc * (U * h * (2.f - h) - u_[x, y]);
        const float dome = smoothstep(0.4f * top, 0.8f * top, f(y)) * (1.f - smoothstep(top, 1.25f * top, f(y)));
        const float above = smoothstep(top, 1.2f * top, f(y));  // lifted rock falls back (it is heavy)
        v_[x, y] += m * visc * (bubble * lift * dome - fall * above - v_[x, y]);
        if (bubble > 0.f) temp_[x, y] += dt * 3.f * bubble * dome * std::max(0.f, (0.8f + 0.6f * I) * 1.2f - temp_[x, y]);
      }
    }
    return;
  }
  if (p_.effect == Effect::eruption) {
    // The wind leans the column and shears it: the air drags the ash towards a wind that grows with height. Above the
    // stable layer hot ash is pushed back down, so the column stops rising and spreads into an umbrella. On the ground
    // the air is still, so what falls back runs out sideways.
    const float W = (2.f * p_.wind - 1.f) * 40.f * s, drag = 1.f - std::exp(-1.0f * dt), ceil = eruption_ceiling(p_, n);
    const float stable = 4.f * tune(p_).buoyancy * s, still = 1.f - std::exp(-3.f * dt), spread = 45.f * s, cx = 0.5f * (n + 1.f);
    const float collapse = (20.f + 30.f * I) * s;
    for (int y = 1; y <= n_; ++y) {
      const float shear = W * f(y) / n, above = smoothstep(ceil, ceil + 0.12f * n, f(y));
      const float level = smoothstep(ceil - 0.1f * n, ceil, f(y));  // at the stable layer the umbrella runs out sideways
      const float low = 1.f - smoothstep(0.3f * ceil, 0.6f * ceil, f(y));
      for (int x = 1; x <= n_; ++x) {
        const float mat = std::min(1.f, 2.f * (temp_[x, y] + soot_[x, y]));
        u_[x, y] += drag * (shear - u_[x, y]) * mat;
        u_[x, y] += dt * spread * level * mat * std::tanh((f(x) - cx) / (0.05f * n));
        v_[x, y] -= dt * stable * above * temp_[x, y];
        v_[x, y] *= 1.f - still * above;  // the stable layer damps vertical motion: the ash spreads out sideways
        // low down, ash that carries too little heat for its weight (the rim, and what has cooled) collapses: it falls
        // back down the column's sides and runs out along the ground
        const float T = temp_[x, y], D = soot_[x, y];
        if (D > 0.05f && low > 0.f) v_[x, y] -= dt * collapse * low * D * (1.f - smoothstep(0.25f, 0.45f, T / D));
      }
    }
    earth_ground();
    return;
  }
  // avalanche: snow on the slope is pulled down it (the slope runs out past 45 to 80% of the way from the release), and
  // where it moves fast it is held up in the air (turbulent suspension), so the cloud billows up as it runs; the ground
  // holds what settles.
  const float pull = 1.f - std::exp(-2.f * dt), carry = 0.85f * tune(p_).soot_weight * s;
  for (int x = 1; x <= n_; ++x) {
    const float U = avalanche_flow(p_, n, f(x)) * s;
    for (int y = 1; y <= n_; ++y) {
      const float D = soot_[x, y];
      if (D <= 0.f) continue;
      u_[x, y] += pull * std::min(1.f, 2.f * D) * (U - u_[x, y]) * smoothstep(0.f, 5.f * s, std::abs(U));
      v_[x, y] += dt * carry * D * smoothstep(10.f * s, 45.f * s, std::abs(u_[x, y]));
    }
  }
  (void)I;
  earth_ground();
}

// avalanche: the running cloud takes in air (entrainment): where it moves fast it swells, so it grows and billows up as
// it goes; once it slows it stops growing.
void Fluid::add_earth_divergence() {
  if (p_.effect != Effect::avalanche) return;
  const float n = f(n_), rate = (0.9f + 1.2f * p_.intensity) * (0.3f + 0.7f * std::exp(-time_ / 1.f));
  if (rate < 0.01f) return;
  const int y_hi = std::min(n_, static_cast<int>(0.5f * n) + 1);
  for (int x = 1; x <= n_; ++x) {
    const float run = std::abs(avalanche_flow(p_, n, f(x))) / 55.f;
    if (run <= 0.f) continue;
    for (int y = 1; y <= y_hi; ++y) {
      const float D = soot_[x, y];
      if (D > 0.02f) div_[x, y] += rate * run * smoothstep(0.05f, 0.5f, D) * (1.f - smoothstep(0.25f * n, 0.5f * n, f(y)));
    }
  }
}

// The ground under the eruption and the avalanche (as fog's and dust's, add_atmos_forces): the air in it is still, and
// what comes down onto it stops (the closed bottom border does the rest).
void Fluid::earth_ground() {
  const float s = f(n_) / 128.f, g = 3.f * s, top = g + 4.f * s;
  const int y_hi = std::min(n_, static_cast<int>(top) + 1);
  for (int y = 1; y <= y_hi; ++y) {
    const float hold = 1.f - smoothstep(g - 2.f * s, g + 2.f * s, f(y)), stop = 1.f - smoothstep(g, top, f(y));
    for (int x = 1; x <= n_; ++x) {
      u_[x, y] *= 1.f - hold;
      v_[x, y] *= 1.f - (v_[x, y] < 0.f ? std::max(hold, stop) : hold);
    }
  }
}

// All three: where material leaves the frame (at its sides and top), the outermost cells take their inner neighbours'
// values (material flows out evenly). The scalar advection's correction reads a stale border there, which would leave a
// line of material, or of nothing, along the border. avalanche: the snow picks up more snow from the ground where it runs fast (more while it
// is young: the second age input's decay, 1 s), and a cloud that has slowed settles out.
void Fluid::earth_decay(float dt) {
  if (!earth(p_.effect)) return;
  for (Field* q : {&temp_, &soot_}) {
    for (int k = 1; k <= n_; ++k) {
      (*q)[1, k] = (*q)[2, k];
      (*q)[n_, k] = (*q)[n_ - 1, k];
      (*q)[k, n_] = (*q)[k, n_ - 1];
    }
  }
  if (p_.effect != Effect::avalanche) return;
  const float n = f(n_), s = n / 128.f, I = p_.intensity, young = 0.3f + 0.7f * std::exp(-time_ / 1.f);
  for (int y = 1; y <= n_; ++y) {
    const float low = 1.f - smoothstep(0.03f * n, 0.08f * n, f(y)), reach = 1.f - smoothstep(0.15f * n, 0.4f * n, f(y));
    for (int x = 1; x <= n_; ++x) {
      const float D = soot_[x, y];
      if (D <= 0.f) continue;
      const float speed = std::abs(u_[x, y]) / s;
      if (reach > 0.f) {
        const float pick = dt * (1.5f + 1.5f * I) * young * reach * smoothstep(15.f, 60.f, speed) * smoothstep(0.05f, 0.4f, D);
        soot_[x, y] += pick * std::max(0.f, 1.5f - D);
      }
      soot_[x, y] -= dt * (0.45f + 0.45f * low) * (1.f - smoothstep(5.f, 25.f, speed)) * soot_[x, y];  // a slow cloud settles out
    }
  }
}

namespace {

// lava's glow: deep red where it is dull, through red and orange to yellow and white-yellow where it is hottest
Rgb lava_ramp(float t) {
  static constexpr std::array<float, 6> at{0.f, 0.15f, 0.4f, 0.7f, 1.0f, 1.4f};
  static constexpr std::array<Rgb, 6> c{{{0.f, 0.f, 0.f}, {0.32f, 0.02f, 0.f}, {0.85f, 0.16f, 0.02f}, {1.f, 0.48f, 0.06f}, {1.f, 0.78f, 0.28f}, {1.f, 0.95f, 0.7f}}};
  return ramp(at, c, t);
}

}  // namespace

// lava: its glow over dark rock, the crust faintly lit from above. eruption: dark grey-brown ash, deeply self-shadowed,
// glowing red to orange where it is hot (at the vent). avalanche: white snow, its shade grey-blue; no light of its own.
bool Fluid::render_earth(std::span<std::uint8_t> out) const {
  if (!earth(p_.effect)) return false;
  const int S = p_.size;
  const float n = f(n_), sc = n / f(S);
  constexpr float lx = -0.45f, ly = 0.89f;  // towards the light: up and to the left
  for (int py = 0; py < S; ++py) {
    for (int px = 0; px < S; ++px) {
      const float gx = (f(px) + 0.5f) * sc + 0.5f;
      const float gy = n - (f(py) + 0.5f) * sc + 0.5f;
      const float T = temp_.sample(gx, gy), D = soot_.sample(gx, gy);
      float tau = 0.f;  // optical depth towards the light
      for (int k = 1; k <= 12; ++k) tau += soot_.sample(gx + lx * 2.f * f(k) * sc, gy + ly * 2.f * f(k) * sc);
      Rgb c{};
      float a = 0.f;
      if (p_.effect == Effect::lava) {
        const Rgb e = lava_ramp(T);
        const float as = 1.f - std::exp(-3.f * D), g = 0.10f * (0.5f + 0.5f * std::exp(-0.15f * tau));
        c = {e.r + g * as, e.g + 0.92f * g * as, e.b + 0.85f * g * as};
        a = clamp01(as + 0.3f * std::max(e.r, e.g));
      } else if (p_.effect == Effect::eruption) {  // only the hottest ash glows (near the vent)
        const Rgb e = fire_ramp(1.3f * std::max(0.f, T - 0.85f));
        const float as = 1.f - std::exp(-3.f * D), keep = 1.f - 0.45f * as, light = std::exp(-0.2f * tau);
        const Rgb g{0.09f + 0.42f * light, 0.085f + 0.39f * light, 0.08f + 0.35f * light};
        c = {e.r * keep + g.r * as, e.g * keep + g.g * as, e.b * keep + g.b * as};
        a = clamp01(as + 0.25f * std::max(e.r, e.g));
      } else {
        const float as = 1.f - std::exp(-2.2f * D), light = std::exp(-0.25f * tau);
        c = {(0.50f + 0.49f * light) * as, (0.55f + 0.44f * light) * as, (0.66f + 0.34f * light) * as};
        a = as;
      }
      auto o = out.subspan((static_cast<std::size_t>(py) * S + px) * 4, 4);
      o[0] = q8(c.r);
      o[1] = q8(c.g);
      o[2] = q8(c.b);
      o[3] = q8(a);
    }
  }
  return true;
}

// --- candle, smokering and heavygas (docs/EFFECTS_AIR.md) ---------------------------------------------------------------
//
// Three presets kept apart from the other effects' code: their sources, forces, decay and drawing. As in the other blocks,
// every force is a function of the state where it acts, the position, the controls and the run's two noise fields
// (curl_potential, source_flicker), which learned dynamics see as inputs; nothing follows a clock of its own. The smoke
// ring, which plays once, is made at its first step (as the explosion's burst) and also follows the time since its start,
// but only through the two decays a learned one-shot sees as inputs (exp(-t / 0.15 s), exp(-t / 1 s)), as the mushroom.

namespace {

bool air_effect(Effect e) { return e == Effect::candle || e == Effect::smokering || e == Effect::heavygas; }

// candle: the wick in the middle of the frame, 12% up (cells of a 128-cell frame); the flame burns above it.
constexpr float kWickX = 64.5f, kWickY = 16.f;

// smokering: the opening the ring is puffed from, in the middle near the bottom of the frame.
constexpr float kOpeningY = 14.f;

// heavygas: the floor (cells of a 128-cell frame): a mound in the middle, 12 cells up, sloping down on either side into
// a hollow 4 cells up, a quarter of the frame wide, with a lip rising to 12 cells at the frame's edges; the gas is
// released over the mound, flows down into the hollows and pools there.
float gas_floor(float X) {
  const float d = std::abs(X - 64.5f);
  return 4.f + 8.f * smoothstep(28.f, 16.f, d) + 8.f * smoothstep(46.f, 60.f, d);
}
constexpr float kGasX = 64.5f, kGasY = 26.5f;  // the release, over the mound (its height a coarse cell's centre)

}  // namespace

bool Fluid::add_air_sources(float dt) {
  if (!air_effect(p_.effect)) return false;
  const float n = f(n_), s = n / 128.f, I = p_.intensity;
  const auto flicker = [&](int x, int y) { return source_flicker(p_, f(x) / s, f(y) / s, time_); };
  const auto box = [&](float x0, float x1, float y0, float y1, auto&& fn) {
    const int xl = std::max(1, static_cast<int>(x0)), xh = std::min(n_, static_cast<int>(x1) + 1);
    const int yl = std::max(1, static_cast<int>(y0)), yh = std::min(n_, static_cast<int>(y1) + 1);
    for (int y = yl; y <= yh; ++y) {
      for (int x = xl; x <= xh; ++x) fn(x, y);
    }
  };
  if (p_.effect == Effect::candle) {
    // The flame: a teardrop of hot gas standing on the wick (rounded below, pointed above), its heat and an upward flow
    // drawn in quickly, so it is fed and steady. It leans with the draught, in gusts that follow the curl noise where it
    // is, and its height breathes with the flicker noise (more with turbulence). Above its tip the gas burns out fast
    // (air_decay) and goes on rising as a warm, invisible plume that carries the flame's soot up as a thin wisp.
    const float cx = kWickX * s, y0 = kWickY * s, W = (4.f + 3.5f * I) * s, H = (20.f + 16.f * I) * s;
    const float level = 1.6f + 0.25f * I, breathe = 0.05f + 0.3f * p_.turbulence, lift = (16.f + 10.f * I) * s;
    const float d = 2.f * p_.wind - 1.f, rate = 1.f - std::exp(-12.f * dt);
    box(cx - W - 0.8f * H, cx + W + 0.8f * H, y0 - 0.2f * H, y0 + 1.4f * H, [&](int x, int y) {
      const float lean = 0.5f * d * (1.f + 0.5f * curl_potential(p_, f(x) / s, f(y) / s, time_));
      const float h = (f(y) - y0) / (H * (1.f + breathe * (flicker(x, y) - 0.15f)));
      if (h <= -0.1f || h >= 1.f) return;
      const float hc = std::max(h, 0.f);
      const float w = W * (h < 0.35f ? std::sqrt(std::max(0.f, 1.f - (0.35f - h) * (0.35f - h) / (0.45f * 0.45f)))
                                     : std::pow(1.f - (h - 0.35f) / 0.65f, 1.2f));
      if (w <= 0.f) return;
      const float k = 1.f - smoothstep(0.55f, 1.f, std::abs(f(x) - cx - lean * (f(y) - y0)) / w);
      if (k <= 0.f) return;
      const float r = rate * k;
      temp_[x, y] += r * (level * k * (0.85f + 0.15f * hc) - temp_[x, y]);
      v_[x, y] += r * (lift - v_[x, y]);
      u_[x, y] += r * (lean * lift - u_[x, y]);
    });
    return true;
  }
  if (p_.effect == Effect::smokering) {
    const float cx = 0.5f * (n + 1.f), oy = kOpeningY * s;
    if (time_ <= 0.f) {
      // The ring, just puffed: two vortices turning in opposite senses (a ring seen from the side), so the air between
      // them moves up and carries them with it, each core full of smoke (warm, lumpy with the flicker noise) out to
      // where it turns as a whole, so the smoke rolls as a broad blob rather than winding into thin spirals.
      const float R = (11.f + 5.f * I) * s, a = 0.7f * R, yc = oy + 0.7f * R, U = (24.f + 16.f * I) * s;
      const float G = 2.f * 3.14159265f * 2.f * R * U, level = 1.0f + 0.4f * I;
      for (int y = 1; y <= n_; ++y) {
        for (int x = 1; x <= n_; ++x) {
          float du = 0.f, dv = 0.f, m = 0.f;
          for (const float side : {-1.f, 1.f}) {  // left core turns anticlockwise, right core clockwise
            const float dx = f(x) - (cx + side * R), dy = f(y) - yc, r2 = dx * dx + dy * dy + 1e-6f;
            const float w = -side * G / (2.f * 3.14159265f * r2) * (1.f - std::exp(-r2 / (a * a)));
            du -= w * dy;
            dv += w * dx;
            m += std::exp(-r2 / (0.7f * 0.7f * a * a));
          }
          u_[x, y] += du;
          v_[x, y] += dv;
          if (m < 0.01f) continue;
          const float d = level * std::min(1.f, 1.3f * m) * (0.7f + 0.3f * smoothstep(-0.5f, 0.5f, flicker(x, y)));
          soot_[x, y] = std::max(soot_[x, y], d);
          temp_[x, y] = std::max(temp_[x, y], 0.3f * d);
        }
      }
      return true;
    }
    // For a moment after the puff a little smoke trails out of the opening behind the ring (the faster age decay).
    const float trail = std::exp(-time_ / 0.15f);
    if (trail < 0.02f) return true;
    const float hw = (5.f + 3.f * I) * s, hh = 3.f * s, up = (25.f + 15.f * I) * s;
    box(cx - hw, cx + hw, oy - hh, oy + hh, [&](int x, int y) {
      const float ex = (f(x) - cx) / hw, ey = (f(y) - oy) / hh;
      const float shape = 1.f - smoothstep(0.5f, 1.f, std::sqrt(ex * ex + ey * ey));
      if (shape <= 0.f) return;
      const float k = trail * shape;
      soot_[x, y] = std::max(soot_[x, y], 0.5f * k);
      temp_[x, y] = std::max(temp_[x, y], 0.15f * k);
      v_[x, y] = std::max(v_[x, y], up * k);
    });
    return true;
  }
  // heavygas: poured out over the mound, in the direction of the flow (left at 0, straight down at 0.5, right at 1), at
  // a rate set by the intensity that surges and slackens with the curl noise at the release (a spill is uneven), broken
  // up by the flicker noise
  const float cx = kGasX * s, cy = kGasY * s, hw = (8.f + 6.f * I) * s, hh = 7.f * s;
  const float surge = 0.2f + 0.8f * smoothstep(-0.35f, 0.35f, curl_potential(p_, kGasX, kGasY, time_));
  const float level = (1.2f + 0.8f * I) * surge, dir = 2.f * p_.wind - 1.f, rate = 1.f - std::exp(-6.f * dt);
  const float pu = dir * (14.f + 18.f * I) * s, pv = -(10.f + 8.f * I) * s;
  box(cx - 1.5f * hw, cx + 1.5f * hw, cy - 1.5f * hh, cy + 1.5f * hh, [&](int x, int y) {
    const float ex = (f(x) - cx) / hw, ey = (f(y) - cy) / hh;
    const float shape = 1.f - smoothstep(0.6f, 1.f, std::sqrt(ex * ex + ey * ey));
    if (shape <= 0.f) return;
    const float k = rate * shape * (0.3f + 0.7f * smoothstep(-0.3f, 0.5f, flicker(x, y)));
    soot_[x, y] += k * std::max(0.f, level - soot_[x, y]);
    u_[x, y] += k * (pu - u_[x, y]);
    v_[x, y] += k * (pv - v_[x, y]);
  });
  return true;
}

// candle: the draught blows the flame and its plume sideways, in gusts that the curl noise where it blows makes (more
// flicker the stronger it blows). smokering: the wind, which moves the clear air (the ring's vortices are its own), and
// the curl noise that breaks it up as it ages. heavygas: its floor (the air in it is still and what comes down onto it
// stops, as fog's ground but following the hollows) and a draught along it in the flow's direction.
void Fluid::add_air_forces(float dt) {
  if (!air_effect(p_.effect)) return;
  const float n = f(n_), s = n / 128.f;
  if (p_.effect == Effect::candle) {
    const float d = 2.f * p_.wind - 1.f, W = d * 40.f * s, drag = 1.f - std::exp(-3.f * dt);
    const float stir = 18.f * std::abs(d) * s, L = 14.f * s, e = 0.25f, tt = time_ * 0.8f;
    const std::uint64_t cseed = p_.seed * 0x2545F4914F6CDD1DULL + 77;  // as curl_potential
    for (int y = 1; y <= n_; ++y) {
      for (int x = 1; x <= n_; ++x) {
        const float mat = std::min(1.f, 2.f * (temp_[x, y] + soot_[x, y]));
        if (mat <= 0.01f) continue;
        const float px = f(x) / L, py = f(y) / L, psi = value_noise(px, py, tt, cseed);
        u_[x, y] += drag * mat * (W * (1.f + 0.6f * psi) - u_[x, y]);
        if (stir > 0.f) {
          u_[x, y] += dt * stir * mat * (value_noise(px, py + e, tt, cseed) - value_noise(px, py - e, tt, cseed)) / (2.f * e);
          v_[x, y] -= dt * stir * mat * (value_noise(px + e, py, tt, cseed) - value_noise(px - e, py, tt, cseed)) / (2.f * e);
        }
      }
    }
    return;
  }
  if (p_.effect == Effect::smokering) {
    // The wind moves the air, and the air carries the ring: the clear air is drawn towards the wind's speed, the smoky
    // cores (whose turning is the ring) much less. As the ring slows it breaks up: the curl noise stirs its smoke harder
    // as it ages (the slower age decay, exp(-t / 1 s)), more with turbulence.
    const float W = (2.f * p_.wind - 1.f) * 18.f * s, drag = 1.f - std::exp(-0.8f * dt);
    const float amp = (10.f + 40.f * p_.turbulence) * s * (1.f - std::exp(-time_ / 1.f)), L = 14.f * s, e = 0.25f, tt = time_ * 0.8f;
    const std::uint64_t cseed = p_.seed * 0x2545F4914F6CDD1DULL + 77;  // as curl_potential
    for (int y = 1; y <= n_; ++y) {
      for (int x = 1; x <= n_; ++x) {
        const float mat = std::min(1.f, 2.f * (temp_[x, y] + soot_[x, y]));
        u_[x, y] += drag * (1.f - 0.8f * mat) * (W - u_[x, y]);
        if (mat <= 0.01f) continue;
        const float px = f(x) / L, py = f(y) / L;
        u_[x, y] += dt * amp * mat * (value_noise(px, py + e, tt, cseed) - value_noise(px, py - e, tt, cseed)) / (2.f * e);
        v_[x, y] -= dt * amp * mat * (value_noise(px + e, py, tt, cseed) - value_noise(px - e, py, tt, cseed)) / (2.f * e);
      }
    }
    return;
  }
  if (p_.effect != Effect::heavygas) return;
  const float dir = 2.f * p_.wind - 1.f, draught = dir * 16.f * s, drag = 1.f - std::exp(-0.8f * dt);
  for (int x = 1; x <= n_; ++x) {
    const float g = gas_floor(f(x) / s) * s, top = g + 4.f * s, low = g + 25.f * s;
    for (int y = 1; y <= n_; ++y) {
      const float Y = f(y);
      if (Y > low) break;
      const float hold = 1.f - smoothstep(g - 2.f * s, g + 2.f * s, Y), stop = 1.f - smoothstep(g, top, Y);
      u_[x, y] *= 1.f - hold;
      v_[x, y] *= 1.f - (v_[x, y] < 0.f ? std::max(hold, stop) : hold);
      u_[x, y] += drag * (1.f - smoothstep(g + 10.f * s, low, Y)) * std::min(1.f, 2.f * soot_[x, y]) * (draught - u_[x, y]);
    }
  }
}

// candle: the flame's gas burns out fast (its heat above a fifth falls at up to 9 per second, on top of tune()'s slow
// cooling of the warm plume) and leaves soot as it burns: the wisp, more in a draught or with turbulence, when it gutters.
// smokering: its smoke (and warmth) spreads a little, as smoke in still air does (4 cells squared per second), so the
// ring's cores stay broad blobs rather than winding into thin spirals.
// heavygas: what the flow has carried into the floor is gone.
void Fluid::air_decay(float dt) {
  if (p_.effect == Effect::smokering) {
    const float k = 4.f * dt * (f(n_) / 128.f) * (f(n_) / 128.f);
    for (Field* q : {&soot_, &temp_}) {
      Field& a = *q;
      for (int y = 1; y <= n_; ++y) {
        for (int x = 1; x <= n_; ++x) {
          const float l = a[std::max(1, x - 1), y] + a[std::min(n_, x + 1), y] + a[x, std::max(1, y - 1)] + a[x, std::min(n_, y + 1)] - 4.f * a[x, y];
          tmp_a_[x, y] = a[x, y] + k * l;
        }
      }
      for (int y = 1; y <= n_; ++y) {
        for (int x = 1; x <= n_; ++x) a[x, y] = tmp_a_[x, y];
      }
    }
    return;
  }
  if (p_.effect == Effect::candle) {
    const float soot = 0.07f + 0.15f * p_.turbulence + 0.15f * std::abs(2.f * p_.wind - 1.f);
    for (int y = 1; y <= n_; ++y) {
      for (int x = 1; x <= n_; ++x) {
        const float T = temp_[x, y];
        if (T <= 0.2f) continue;
        const float burnt = T * (1.f - std::exp(-9.f * dt * smoothstep(0.2f, 0.45f, T)));
        temp_[x, y] = T - burnt;
        soot_[x, y] += soot * burnt;
      }
    }
    return;
  }
  if (p_.effect != Effect::heavygas) return;
  (void)dt;
  const float s = f(n_) / 128.f;
  for (int x = 1; x <= n_; ++x) {
    const float g = gas_floor(f(x) / s) * s;
    for (int y = 1; y <= n_ && f(y) < g - 2.f * s; ++y) soot_[x, y] = 0.f;
  }
}

namespace {

// the candle's light: dull orange-red at its edge, through orange and yellow to a pale yellow-white core
Rgb candle_ramp(float t) {
  static constexpr std::array<float, 6> at{0.3f, 0.38f, 0.48f, 0.64f, 0.82f, 1.05f};
  static constexpr std::array<Rgb, 6> c{{{0.f, 0.f, 0.f}, {0.40f, 0.10f, 0.01f}, {0.90f, 0.38f, 0.06f}, {1.f, 0.68f, 0.20f}, {1.f, 0.87f, 0.48f}, {1.f, 0.97f, 0.82f}}};
  return ramp(at, c, t);
}

}  // namespace

// candle: emitted light over its thin grey smoke; smokering: pale grey smoke, lit from above and to the left and shaded
// by itself; heavygas: a cold white vapour, barely shaded, faintly blue in its depth. The floor is the scene's to draw.
bool Fluid::render_air(std::span<std::uint8_t> out) const {
  if (!air_effect(p_.effect)) return false;
  const int S = p_.size;
  const float n = f(n_), sc = n / f(S);
  constexpr float lx = -0.45f, ly = 0.89f;  // towards the light: up and to the left
  for (int py = 0; py < S; ++py) {
    for (int px = 0; px < S; ++px) {
      const float gx = (f(px) + 0.5f) * sc + 0.5f;
      const float gy = n - (f(py) + 0.5f) * sc + 0.5f;
      const float T = temp_.sample(gx, gy), D = soot_.sample(gx, gy);
      float tau = 0.f;  // optical depth towards the light
      for (int k = 1; k <= 12; ++k) tau += soot_.sample(gx + lx * 2.f * f(k) * sc, gy + ly * 2.f * f(k) * sc);
      Rgb c{};
      float a = 0.f;
      if (p_.effect == Effect::candle) {
        const Rgb e = candle_ramp(T);
        const float as = 1.f - std::exp(-2.5f * D), g = 0.5f * (0.5f + 0.5f * std::exp(-0.3f * tau)), keep = 1.f - 0.4f * as;
        c = {e.r * keep + g * as, e.g * keep + 0.98f * g * as, e.b * keep + 0.96f * g * as};
        a = clamp01(as + 0.3f * std::max(e.r, e.g));
      } else if (p_.effect == Effect::smokering) {
        const float as = 1.f - std::exp(-2.2f * D), g = 0.84f * (0.42f + 0.58f * std::exp(-0.3f * tau));
        c = {0.97f * g * as, 0.97f * g * as, g * as};
        a = as;
      } else {
        const float as = 1.f - std::exp(-1.8f * D), g = 0.56f + 0.42f * std::exp(-0.1f * tau);
        c = {(0.9f + 0.06f * g) * g * as, (0.94f + 0.04f * g) * g * as, g * as};
        a = as;
      }
      auto o = out.subspan((static_cast<std::size_t>(py) * S + px) * 4, 4);
      o[0] = q8(c.r);
      o[1] = q8(c.g);
      o[2] = q8(c.b);
      o[3] = q8(a);
    }
  }
  return true;
}

// --- crownfire, windowfire and dustexplosion (docs/EFFECTS_FIRE2.md) ----------------------------------------------------
//
// Three presets kept apart from the other effects' code: their sources, forces, decay and drawing. As in the other blocks,
// every force is a function of the state where it acts, the position, the controls and the run's two noise fields
// (curl_potential, source_flicker), which learned dynamics see as inputs; nothing follows a clock of its own. The dust
// explosion, which plays once, is made at its first step (its cloud and the spark that lights it, as the explosion's
// burst) and then burns by its own state: where its dust is hot it burns, giving heat and expanding, and the heat spreads
// into the dust beside it, so the flame runs through the cloud. It also follows the time since its start, but only
// through the slower of the two decays a learned one-shot sees as inputs (exp(-t / 1 s)): the cloud can burn for its
// first one to two seconds.

namespace {

bool fire2_effect(Effect e) { return e == Effect::crownfire || e == Effect::windowfire || e == Effect::dustexplosion; }

// crownfire: the canopy (cells of a 128-cell frame): the crowns from 22 cells up to their tops, a tree every 16 cells,
// whose tops are 40 to 46 cells up (a crown's middle is its highest and densest part); the trunks under them and the
// ground (the closed bottom border) are not drawn.
constexpr float kCrownBase = 22.f;
float crown_shape(float X) { return 0.5f + 0.5f * std::cos(2.f * 3.14159265f * (X - 8.5f) / 16.f); }  // 1 at a crown's middle
float crown_top(float X) { return 40.f + 6.f * crown_shape(X); }
// The fire's extent along the canopy: its burning zone, shifted upwind (the head of the fire is its downwind edge), and
// where the embers it throws ahead light spot fires in the crowns (both sides in still air, a short way; downwind, further
// the harder the wind blows). d = 2 wind - 1.
struct CrownZone {
  float c, hw;
};
CrownZone crown_zone(const Params& p) { return {64.5f - 8.f * (2.f * p.wind - 1.f), 30.f + 10.f * p.intensity}; }
// How strongly the crowns burn at X (cells of a 128-cell frame), before the torching that the flicker noise draws: the
// zone, its downwind half burning harder in a wind (the head).
float crown_burn(const Params& p, float X) {
  const auto [c, hw] = crown_zone(p);
  const float d = 2.f * p.wind - 1.f, a = (X - c) / hw;
  return (1.f - smoothstep(0.7f, 1.f, std::abs(a))) * (0.75f + 0.25f * d * std::clamp(a, -1.f, 1.f));
}
// Where spot fires can catch (0 to 1 at X): beyond either edge of the zone, out to 4 cells in still air, up to 36 cells
// downwind in the strongest wind and none upwind.
float crown_spots(const Params& p, float X) {
  const auto [c, hw] = crown_zone(p);
  const float d = 2.f * p.wind - 1.f, side = X >= c ? 1.f : -1.f, edge = c + side * (hw + 2.f);
  const float reach = 4.f + 32.f * std::max(0.f, side * d) - 4.f * std::max(0.f, -side * d), out = side * (X - edge);
  if (reach <= 0.5f || out <= -2.f) return 0.f;
  return smoothstep(-2.f, 1.f, out) * (1.f - smoothstep(0.7f * reach, reach, out));
}

// windowfire: the window (cells of a 128-cell frame, the facade seen front-on): an opening 36 cells wide and 32 tall, its
// sill 12 cells up; the burning room's hot gas flows out through its upper part, above the neutral plane, and the
// fresh air in through the lower part. The eave over it is the top of the frame (a closed border).
constexpr float kWinX0 = 46.5f, kWinX1 = 82.5f, kWinSill = 12.f, kWinHead = 44.f;
float window_neutral(const Params& p) { return 30.f - 6.f * p.intensity; }  // a bigger fire pushes it lower

// dustexplosion: the cloud (cells of a 128-cell frame): an oval on the ground in the middle of the frame, and where the
// spark lights it (the second control: its left end at 0, its middle at 0.5, its right end at 1).
struct DustCloud {
  float cx, cy, a, b;
};
DustCloud dust_cloud(const Params& p) {
  const float a = 30.f + 14.f * p.intensity, b = 14.f + 6.f * p.intensity;
  return {64.5f, 5.f + b, a, b};
}
// The burn: where the dust is hotter than its ignition (0.35 to 0.7) it burns at up to 80 times its amount a second,
// while the cloud can burn (the slow age decay from 0.4 down to 0.12: fully for its first 0.9 s, not at all after 2.1 s)
// and until its air is used up (the burning stops as the heat nears its cap, 1.25 to 1.6: the cloud has more dust than
// its air can burn, so most of it is left as the pall).
float dust_burn_rate(float T, float D, float can, float cap) {
  if (D <= 0.f || T <= 0.35f) return 0.f;
  return 80.f * can * D * smoothstep(0.35f, 0.7f, T) * (1.f - smoothstep(cap - 0.4f, cap, T));
}
// The heat at which the burning stops (its air used up): 1.25 to 1.6, richer and leaner where the curl noise is (psi).
float dust_cap(float psi) { return 1.25f + 0.35f * smoothstep(-0.5f, 0.5f, psi); }
float dust_can_burn(float time) { return smoothstep(0.12f, 0.4f, std::exp(-time / 1.f)); }
constexpr float kDustHeat = 4.f;  // heat per unit of dust burnt

}  // namespace

bool Fluid::add_fire2_sources(float dt) {
  if (!fire2_effect(p_.effect)) return false;
  const float n = f(n_), s = n / 128.f, I = p_.intensity;
  const auto flicker = [&](int x, int y) { return source_flicker(p_, f(x) / s, f(y) / s, time_); };
  if (p_.effect == Effect::crownfire) {
    // The crowns burn through their depth: flames where the flicker noise is high (crowns torching), in stronger and
    // weaker stretches that move along the canopy with the curl noise; the head burns hardest. Beyond the zone, where the
    // flicker is high, spot fires catch in the crowns the embers land in, and die down as it falls.
    const float heat = 1.5f + 0.4f * I, lift = (24.f + 16.f * I) * s, smoke = 1.2f + 0.6f * I;
    const int y_lo = std::max(1, static_cast<int>(kCrownBase * s)), y_hi = std::min(n_, static_cast<int>(46.f * s) + 1);
    for (int x = 1; x <= n_; ++x) {
      const float X = f(x) / s, burn = crown_burn(p_, X), spots = crown_spots(p_, X);
      if (burn <= 0.f && spots <= 0.f) continue;
      const float top = crown_top(X), crown = 0.6f + 0.4f * crown_shape(X);
      for (int y = y_lo; y <= y_hi; ++y) {
        const float Y = f(y) / s, in = crown * smoothstep(kCrownBase, kCrownBase + 4.f, Y) * (1.f - smoothstep(top - 5.f, top, Y));
        if (in <= 0.f) continue;
        const float fl = flicker(x, y);
        const float k = in * (burn * (0.5f + 0.5f * smoothstep(-0.2f, 0.6f, fl)) * (0.75f + 0.25f * curl_potential(p_, X, Y, time_)) +
                              0.7f * spots * smoothstep(0.3f, 0.6f, fl));
        if (k <= 0.f) continue;
        temp_[x, y] = std::max(temp_[x, y], heat * k);
        soot_[x, y] += dt * smoke * k;
        v_[x, y] = std::max(v_[x, y], lift * k);
      }
    }
    return true;
  }
  if (p_.effect == Effect::windowfire) {
    // Above the neutral plane the room's hot, smoky gas pours out (drawn in quickly, so the flames are fed and steady),
    // rising faster towards the head of the window and broken up by the flicker noise; below it the opening is the
    // burning room seen through the inflowing air: still, glowing red to orange as the flicker noise draws it. The
    // gas burns once it is out of the window (fire2_decay).
    const float neutral = window_neutral(p_) * s, x0 = kWinX0 * s, x1 = kWinX1 * s, sill = kWinSill * s, head = kWinHead * s;
    const float level = 1.3f + 0.5f * I, lift = (18.f + 14.f * I) * s, smoke = 0.3f + 0.2f * I, glow = 0.75f + 0.25f * I;
    const float rate = 1.f - std::exp(-10.f * dt), slow = 1.f - std::exp(-4.f * dt);
    const int xl = std::max(1, static_cast<int>(x0)), xh = std::min(n_, static_cast<int>(x1) + 1);
    const int yl = std::max(1, static_cast<int>(sill - 3.f * s)), yh = std::min(n_, static_cast<int>(head) + 1);
    for (int y = yl; y <= yh; ++y) {
      const float Y = f(y), vert = smoothstep(sill - 0.5f, sill + 1.5f * s, Y) * (1.f - smoothstep(head - 1.5f * s, head + 0.5f, Y));
      for (int x = xl; x <= xh; ++x) {
        const float X = f(x), jamb = smoothstep(x0 - 0.5f, x0 + 1.5f * s, X) * (1.f - smoothstep(x1 - 1.5f * s, x1 + 0.5f, X));
        const float in = vert * jamb;
        if (Y < sill + 1.5f * s && jamb > 0.f) {  // the wall under the sill: still air, so nothing runs down it
          u_[x, y] *= 1.f - jamb;
          v_[x, y] *= 1.f - jamb;
        }
        if (in <= 0.f) continue;
        const float fl = smoothstep(-0.3f, 0.5f, flicker(x, y));
        if (Y >= neutral) {
          const float up = std::clamp((Y - neutral) / std::max(1.f, head - neutral), 0.f, 1.f), r = rate * in;
          const float to = std::max(level * (0.5f + 0.5f * fl) * smoothstep(0.f, 0.4f, up), glow * (0.35f + 0.65f * fl));
          temp_[x, y] += r * (to - temp_[x, y]);
          v_[x, y] += r * (lift * (0.4f + 0.6f * up) - v_[x, y]);
          soot_[x, y] += dt * smoke * in * (0.5f + 0.5f * fl) * smoothstep(0.f, 0.4f, up);
        } else {
          const float hold = in * (1.f - smoothstep(neutral - 3.f * s, neutral, Y));
          temp_[x, y] += slow * in * (glow * (0.35f + 0.65f * fl) - temp_[x, y]);
          u_[x, y] *= 1.f - std::max(hold, in);
          v_[x, y] *= 1.f - std::max(hold, in);
        }
      }
    }
    return true;
  }
  // dustexplosion: at its first step, the cloud (lumpy, its edge and its lumps from the flicker noise), still, and the
  // spark that lights it, low in the cloud at its left end, its middle or its right end (the second control)
  if (time_ > 0.f) return true;
  const auto [cx, cy, a, b] = dust_cloud(p_);
  const float level = 0.7f + 0.5f * I, ix = cx + (2.f * p_.wind - 1.f) * 0.75f * a, iy = cy - 0.25f * b, ir = 4.5f;
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) {
      const float X = f(x) / s, Y = f(y) / s, fl = flicker(x, y);
      const float ex = (X - cx) / a, ey = (Y - cy) / b;
      const float fall = 1.f - smoothstep(0.55f, 1.f + 0.3f * fl, std::sqrt(ex * ex + ey * ey));
      if (fall > 0.f) soot_[x, y] = std::max(soot_[x, y], level * fall * (0.7f + 0.3f * smoothstep(-0.5f, 0.5f, fl)));
      const float r = std::sqrt((X - ix) * (X - ix) + (Y - iy) * (Y - iy));
      if (r < ir) temp_[x, y] = std::max(temp_[x, y], 1.4f * (1.f - smoothstep(0.4f * ir, ir, r)));
    }
  }
  return true;
}

// crownfire: the wind, slowed in the canopy and stronger with height above it, drags the flames and the column (it leans
// and bends the column over). windowfire: the eave (upward flow stops under it, so the smoke spreads out sideways under
// it as a ceiling jet) and a wind along the facade. dustexplosion: none of its own (its burning expands it, project()).
void Fluid::add_fire2_forces(float dt) {
  if (!fire2_effect(p_.effect)) return;
  const float n = f(n_), s = n / 128.f, d = 2.f * p_.wind - 1.f;
  if (p_.effect == Effect::crownfire) {
    const float drag = 1.f - std::exp(-2.f * dt);
    for (int y = 1; y <= n_; ++y) {
      const float W = d * (16.f + 50.f * smoothstep(40.f, 110.f, f(y) / s)) * s;
      for (int x = 1; x <= n_; ++x) {
        const float mat = std::min(1.f, 2.f * (temp_[x, y] + soot_[x, y]));
        u_[x, y] += drag * (W - u_[x, y]) * mat;
      }
    }
    return;
  }
  if (p_.effect == Effect::windowfire) {
    const float W = d * 22.f * s, drag = 1.f - std::exp(-1.2f * dt), under = 5.f * s;
    for (int y = 1; y <= n_; ++y) {
      const float stop = smoothstep(n - under, n, f(y));
      for (int x = 1; x <= n_; ++x) {
        const float mat = std::min(1.f, 2.f * (temp_[x, y] + soot_[x, y]));
        u_[x, y] += drag * (W * (0.4f + 0.6f * f(y) / n) - u_[x, y]) * mat;
        if (v_[x, y] > 0.f) v_[x, y] *= 1.f - stop;
      }
    }
    return;
  }
  (void)dt;
  (void)n;
}

// dustexplosion: the burning gas expands, at a rate set by how fast its dust burns, which pushes the cloud out ahead of
// the flame and swells the fireball.
void Fluid::add_fire2_divergence() {
  if (p_.effect != Effect::dustexplosion) return;
  const float can = dust_can_burn(time_), s = f(n_) / 128.f;
  if (can <= 0.f) return;
  for (int y = 1; y <= n_; ++y) {
    for (int x = 1; x <= n_; ++x) {
      const float cap = dust_cap(curl_potential(p_, f(x) / s, f(y) / s, time_));
      div_[x, y] += 2.5f * dust_burn_rate(temp_[x, y], soot_[x, y], can, cap);
    }
  }
}

// crownfire and windowfire: the flames' gas burns out (its heat above 0.4 falls at up to 1.7 and 1.6 a second, on top of
// tune()'s slow cooling of the warm column) and leaves smoke as it burns. dustexplosion: its dust burns where it is hot (the burn
// above), and the heat spreads into the dust beside it (the flame's front runs through the cloud), not into clear air.
// All three: where material leaves the frame at its sides (and top), the outermost cells take their inner neighbours'
// values, as the earth effects' (earth_decay).
void Fluid::fire2_decay(float dt) {
  if (!fire2_effect(p_.effect)) return;
  if (p_.effect == Effect::dustexplosion) {
    const float can = dust_can_burn(time_);
    if (can > 0.f) {
      // the heat spreads at 40 cells squared a second (four passes of 10, each stable), only where there is dust
      const float s = f(n_) / 128.f, k = 10.f * dt * s * s;
      for (int y = 1; y <= n_; ++y) {  // how fast it spreads: faster and slower where the flicker noise is (the front wrinkles)
        for (int x = 1; x <= n_; ++x) {
          tmp_b_[x, y] = soot_[x, y] <= 0.02f ? 0.f : smoothstep(0.02f, 0.2f, soot_[x, y]) *
                                                         (0.5f + smoothstep(-0.4f, 0.4f, source_flicker(p_, f(x) / s, f(y) / s, time_)));
        }
      }
      for (int pass = 0; pass < 4; ++pass) {
        for (int y = 1; y <= n_; ++y) {
          for (int x = 1; x <= n_; ++x) {
            const auto& T = temp_;
            if (tmp_b_[x, y] <= 0.f) {
              tmp_a_[x, y] = T[x, y];
              continue;
            }
            const int xl = std::max(1, x - 1), xr = std::min(n_, x + 1), yd = std::max(1, y - 1), yu = std::min(n_, y + 1);
            const float l = (4.f * (T[xl, y] + T[xr, y] + T[x, yd] + T[x, yu]) + T[xl, yd] + T[xr, yd] + T[xl, yu] + T[xr, yu] - 20.f * T[x, y]) / 6.f;
            tmp_a_[x, y] = T[x, y] + k * tmp_b_[x, y] * l;
          }
        }
        for (int y = 1; y <= n_; ++y) {
          for (int x = 1; x <= n_; ++x) temp_[x, y] = tmp_a_[x, y];
        }
      }
      for (int y = 1; y <= n_; ++y) {
        for (int x = 1; x <= n_; ++x) {
          const float T = temp_[x, y], D = soot_[x, y], cap = dust_cap(curl_potential(p_, f(x) / s, f(y) / s, time_));
          const float burnt = std::min({D, dt * dust_burn_rate(T, D, can, cap), std::max(0.f, cap - T) / kDustHeat});
          temp_[x, y] = T + kDustHeat * burnt;
          soot_[x, y] = D - 0.1f * burnt;  // most of what burns leaves ash and smoke
        }
      }
    }
  } else {
    const bool crown = p_.effect == Effect::crownfire;
    const float rate = crown ? 1.7f : 1.6f, smoke = crown ? 0.6f : 0.8f, s = f(n_) / 128.f;
    const float room = crown ? 0.f : kWinHead * s, x0 = kWinX0 * s, x1 = kWinX1 * s;  // the gas in the window burns once out of it
    for (int y = 1; y <= n_; ++y) {
      for (int x = 1; x <= n_; ++x) {
        const float T = temp_[x, y];
        if (T <= 0.4f || (f(y) < room && f(x) > x0 - 1.f && f(x) < x1 + 1.f)) continue;
        const float burnt = T * (1.f - std::exp(-rate * dt * smoothstep(0.4f, 0.7f, T)));
        temp_[x, y] = T - burnt;
        soot_[x, y] += smoke * burnt;
      }
    }
  }
  const bool top = p_.effect != Effect::windowfire;
  for (Field* q : {&temp_, &soot_}) {
    for (int k = 1; k <= n_; ++k) {
      (*q)[1, k] = (*q)[2, k];
      (*q)[n_, k] = (*q)[n_ - 1, k];
      if (top) (*q)[k, n_] = (*q)[k, n_ - 1];
    }
  }
}

namespace {

// the flames of the crown fire and the window fire: invisible below a heat of 0.4 (the warm column), then dull red,
// orange, yellow and a pale yellow core
Rgb flame_ramp(float t) {
  static constexpr std::array<float, 6> at{0.4f, 0.48f, 0.62f, 0.85f, 1.1f, 1.45f};
  static constexpr std::array<Rgb, 6> c{{{0.f, 0.f, 0.f}, {0.45f, 0.06f, 0.f}, {0.92f, 0.30f, 0.03f}, {1.f, 0.58f, 0.10f}, {1.f, 0.82f, 0.35f}, {1.f, 0.95f, 0.72f}}};
  return ramp(at, c, t);
}
// the dust flame: a deep brown-red at its edge, through a brown orange to orange and yellow where it is hottest
Rgb dust_flame_ramp(float t) {
  static constexpr std::array<float, 6> at{0.3f, 0.45f, 0.7f, 1.0f, 1.3f, 1.65f};
  static constexpr std::array<Rgb, 6> c{{{0.f, 0.f, 0.f}, {0.35f, 0.08f, 0.02f}, {0.75f, 0.24f, 0.04f}, {1.f, 0.42f, 0.08f}, {1.f, 0.56f, 0.16f}, {1.f, 0.72f, 0.32f}}};
  return ramp(at, c, t);
}

}  // namespace

// Emitted light by heat over smoke or dust shaded towards the light (up and to the left). crownfire: its flames over a
// thick grey-brown smoke; windowfire: its flames and the room's glow over black smoke; dustexplosion: a brown-orange
// fireball over tan-brown dust. The trees, the facade and the ground are the scene's to draw.
bool Fluid::render_fire2(std::span<std::uint8_t> out) const {
  if (!fire2_effect(p_.effect)) return false;
  const int S = p_.size;
  const float n = f(n_), sc = n / f(S);
  constexpr float lx = -0.45f, ly = 0.89f;
  for (int py = 0; py < S; ++py) {
    for (int px = 0; px < S; ++px) {
      const float gx = (f(px) + 0.5f) * sc + 0.5f;
      const float gy = n - (f(py) + 0.5f) * sc + 0.5f;
      const float T = temp_.sample(gx, gy), D = soot_.sample(gx, gy);
      float tau = 0.f;  // optical depth towards the light
      for (int k = 1; k <= 12; ++k) tau += soot_.sample(gx + lx * 2.f * f(k) * sc, gy + ly * 2.f * f(k) * sc);
      const float light = std::exp(-0.45f * tau);
      Rgb c{};
      float a = 0.f;
      if (p_.effect == Effect::dustexplosion) {
        const Rgb e = dust_flame_ramp(T);
        const float as = 1.f - std::exp(-3.2f * D), keep = 1.f - 0.6f * as;
        const float lit = std::exp(-0.25f * tau), g = 0.72f * (0.45f + 0.55f * lit) * (1.f - 0.6f * smoothstep(0.4f, 1.3f, T));  // hot dust is charred dark
        c = {e.r * keep + 0.86f * g * as, e.g * keep + 0.72f * g * as, e.b * keep + 0.52f * g * as};
        a = clamp01(as + 0.3f * std::max(e.r, e.g));
      } else {
        const bool crown = p_.effect == Effect::crownfire;
        const Rgb e = flame_ramp(T);
        const float as = 1.f - std::exp(-(crown ? 2.2f : 2.8f) * D), keep = 1.f - 0.5f * as;
        const float g = (crown ? 0.55f : 0.13f) * (0.3f + 0.7f * light);
        const Rgb tint = crown ? Rgb{1.f, 0.94f, 0.86f} : Rgb{1.f, 0.97f, 0.94f};
        c = {e.r * keep + tint.r * g * as, e.g * keep + tint.g * g * as, e.b * keep + tint.b * g * as};
        a = clamp01(as + 0.3f * std::max(e.r, e.g));
      }
      auto o = out.subspan((static_cast<std::size_t>(py) * S + px) * 4, 4);
      o[0] = q8(c.r);
      o[1] = q8(c.g);
      o[2] = q8(c.b);
      o[3] = q8(a);
    }
  }
  return true;
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
    // Long enough for the plume to reach a statistically steady state, so the loop crossfade joins similar frames
    // (fog takes about 5 s to cover the ground; cloud as long to grow from nothing; the water effects' mist as long).
    const int warmup = p.warmup >= 0 ? p.warmup
                       : (p.effect == Effect::smoke                                                    ? 180
                          : atmos(p.effect) || water(p.effect) || earth(p.effect) ||
                                    p.effect == Effect::heavygas || p.effect == Effect::crownfire ||
                                    p.effect == Effect::windowfire ? 150  // heavy gas pools as fog does; the fires' columns and ceiling layer build up
                                                                 : flow_warmup(p.effect));
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
