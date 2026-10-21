// The weather director in 2D (namespace wx: docs/WEATHER.md): a small weather state that moves between regimes and drives the layers
// that exist (the rain, snow and hail emitters, the fog and cloud models, the storm layers, the wind every module feels,
// the sky's light and the lightning's flashes).
//
//   - The state: cloud cover, rain and snow rates, hail, mean wind and gustiness, visibility, temperature, the rate of
//     lightning, the load of dust (and the fog's extinction, from which, with the rain's, the snow's and the dust's, the
//     visibility follows).
//   - Regimes: clear, overcast, rain, storm, snow, fog and dust storm. Each is a target state, varied by the seed each
//     time it is asked for, with weather noise around it.
//   - Transitions: from the state it is in to the one asked for, over a time, each quantity in a window of its own so
//     that the order real weather keeps holds: clouds build before rain, the gust front (the wind, its gusts and the drop
//     in temperature) arrives before the downpour, lightning and hail come with the heaviest rain; and things clear in
//     reverse (the lightning stops first, then the rain, then the wind eases, then the clouds break).
//   - Weather noise: gusts and rain cells are seeded Ornstein-Uhlenbeck processes, stepped exactly (Gillespie 1996): the
//     gusts with the Kaimal integral time scale of the surface wind (IEC 61400-1 at 10 m: L_u / U), smoothed over about
//     a second as gusts are measured (3-second means); the rain cells log-normal, minutes long.
//   - Relations from the literature instead of guesses (cited in docs/WEATHER.md §3): Koschmieder's visibility, the
//     Marshall-Palmer drop sizes, the fall speeds of Atlas, Srivastava and Sekhon, Locatelli and Hobbs's snowflakes, the
//     Beaufort scale, Kasten and Czeplak's daylight under cloud.
//
// Deterministic: every random number is a hash of the seed, the step and a channel (or of the seed, the request and a
// channel), never a shared generator, so a scene draws the same random numbers with a director or without one. Nothing
// here allocates.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace nfx::compose::wx {

enum class Regime : std::uint8_t { clear, overcast, rain, storm, snow, fog, dust };
inline constexpr int kRegimes = 7;
std::string_view regime_name(Regime r);
bool regime_of(std::string_view word, Regime& out);  // false: not a regime's name
std::span<const std::string_view> regime_names();     // in order: clear, overcast, rain, storm, snow, fog, dust

// --- established relations (docs/WEATHER.md §3) -------------------------------------------------------------------------

// Koschmieder (1924): a black object at distance V against the horizon sky shows the contrast exp(-sigma V); the eye's
// threshold of 2 % gives V = ln(50) / sigma = 3.912 / sigma.
inline constexpr float kKoschmieder = 3.912f;
float visibility_m(float extinction_per_km);   // 3912 / sigma (sigma in km^-1), metres
float extinction_per_km(float visibility_m);   // its inverse

// Marshall and Palmer (1948): N(D) = N0 exp(-Lambda D), N0 = 8000 m^-3 mm^-1, Lambda = 4.1 R^-0.21 mm^-1 (R in mm/h).
inline constexpr float kMpN0 = 8000.f;
inline constexpr float kDropMin = 0.5f, kDropMax = 6.f;  // the drops drawn, mm (smaller ones are drizzle; larger ones break up)
float mp_slope(float rain_mm_h);                                            // Lambda, mm^-1
float mp_number(float rain_mm_h, float d0 = kDropMin, float d1 = kDropMax); // drops a cubic metre with D in [d0, d1]
float mp_mean_diameter(float rain_mm_h, float d0 = kDropMin, float d1 = kDropMax);  // their mean diameter, mm
// A diameter in [d0, d1] (mm) drawn from the distribution by its inverse, for u in [0, 1).
float mp_sample(float rain_mm_h, float u, float d0 = kDropMin, float d1 = kDropMax);
// The rain's extinction (km^-1): drops much larger than the wavelength remove twice their cross-section (the extinction
// paradox), so sigma = 2 * integral of (pi / 4) D^2 N(D) dD = pi N0 / Lambda^3 = 0.365 R^0.63 km^-1.
float rain_extinction(float rain_mm_h);

// Atlas, Srivastava and Sekhon (1973): a raindrop's fall speed v = 9.65 - 10.3 exp(-0.6 D) m/s (D in mm, near sea
// level), the fit used between 0.6 and 5.8 mm (0 below 0.03 mm, where it turns negative).
float drop_speed(float d_mm);
float mp_mean_speed(float rain_mm_h, float d0 = kDropMin, float d1 = kDropMax);  // number-weighted mean over [d0, d1], m/s
// The rain rate the two give together, 6 pi 10^-4 integral of D^3 v(D) N(D) dD over all sizes (mm/h): a check of their
// consistency (docs/WEATHER.md §3.3), not used to draw.
float mp_atlas_rate(float rain_mm_h);

// Locatelli and Hobbs (1974), aggregates of unrimed radiating assemblages of dendrites (2 to 10 mm): V = 0.8 D^0.16 m/s,
// M = 0.073 D^1.4 mg (D the maximum dimension, mm). The flakes' sizes are an assumption: exponential, slope 0.4 mm^-1,
// over their 2 to 10 mm (mean 4.2 mm).
inline constexpr float kFlakeMin = 2.f, kFlakeMax = 10.f, kFlakeSlope = 0.4f;
float flake_speed(float d_mm);
float flake_mass_mg(float d_mm);
float flake_sample(float u);                // a maximum dimension (mm) for u in [0, 1)
float flake_mean_speed();                   // number-weighted, m/s
float snow_number(float snow_mm_h);         // flakes a cubic metre: the mass flux S / 3600 kg m^-2 s^-1 over <M V>
// Snow's extinction (km^-1): twice the flakes' cross-section, a flake covering 0.3 of the circle around it (an
// assumption: aggregates are open; Rasmussen and colleagues (1999) find the visibility at a given rate varying by a factor
// of 3 to 10 with the snow). 1 mm/h gives about 800 m, inside the US weather service's "moderate snow" (0.5 to 1 km).
float snow_extinction(float snow_mm_h);

// Hail: a sphere of ice (900 kg m^-3) in air (1.2 kg m^-3) at the drag coefficient of a rough sphere at high Reynolds
// number (0.6): v = sqrt(4 g D rho_ice / (3 C_d rho_air)). 10 mm: 12.8 m/s.
float hail_speed(float d_mm);

// The Beaufort scale (WMO): v = 0.836 B^(3/2) m/s at 10 m, and what the wind does on land at each force.
float beaufort(float wind_ms);         // the force, continuous (0 to 12 and beyond)
float beaufort_speed(float force);     // m/s
std::string_view beaufort_name(int force);         // "calm", "light air", ... "hurricane"
std::string_view beaufort_on_land(int force);      // "smoke rises vertically", ...
// The Beaufort effects the layers show: smoke and flames leaning (a learned module's `wind` control, 0.5 still air, its
// full lean at a gale, force 8, "generally impedes progress"), leaves in motion (0 up to force 2, 1 from force 4: "leaves
// and small twigs in constant motion" is force 3), dust and loose paper raised (0 up to force 3.5, 1 from 5: "raises dust
// and loose paper" is force 4).
float lean(float wind_ms, float direction);
float leaves(float wind_ms);
float lift(float wind_ms);

// Kasten and Czeplak (1980): the global irradiance under N oktas of cloud is G(0) (1 - 0.75 (N/8)^3.4), cloud = N/8.
float daylight(float cloud);

// Gusts: the integral length scale of the longitudinal wind at 10 m, Kaimal's spectrum as IEC 61400-1 gives it,
// L_u = 8.1 x 0.7 z = 56.7 m; its integral time scale is L_u / U (1 to 30 s).
inline constexpr float kGustLength = 56.7f;
float gust_time(float mean_ms);

// Rain and snow: the share that falls as snow, a ramp from all rain at 2 C to all snow at 0 C (a simple partition, not
// a fit to observations).
float snow_share(float temp_c);

// Lightning: a flash has 1 to 5 return strokes (one stroke 17 % of the time; 3 on average), 40 to 100 ms apart (Rakov and
// Uman 2003: typically 3 to 5, about 60 ms), each a flash of light decaying over 45 ms.
inline constexpr int kMaxStrokes = 5;

// --- the state -----------------------------------------------------------------------------------------------------------

// What a script reads as NAME.member (and an engine with nvfx_scene_get_input(scene, "NAME.member")). The first twelve are
// the state; the rest follow from it.
enum Member : int {
  m_cloud,       // cloud cover, 0 to 1 (oktas / 8)
  m_rain,        // rain rate, mm/h
  m_snow,        // snow rate, mm/h of melted water
  m_hail,        // hail, 0 to 1 (a share of the strongest hail)
  m_wind,        // the wind now (mean and gust), m/s at 10 m
  m_mean,        // the mean wind, m/s
  m_gust,        // gustiness: the turbulence intensity, the gusts' standard deviation over the mean wind
  m_visibility,  // Koschmieder's visibility, m
  m_temp,        // temperature, C
  m_lightning,   // flashes a minute
  m_dust,        // dust load: 1 is a severe dust storm's (WMO: visibility under 200 m from dust alone)
  m_fog,         // the fog's extinction, km^-1
  m_extinction,  // everything's: clear air, fog, rain, snow and dust, km^-1
  m_u,           // the wind now along x, m/s (its direction's sign)
  m_beaufort,    // the wind's Beaufort force now, continuous
  m_lean,        // lean(): a learned module's `wind` control (0.5: still air)
  m_leaves,      // leaves(): 0 to 1
  m_lift,        // lift(): 0 to 1
  m_light,       // daylight(cloud): Kasten and Czeplak
  m_flash,       // the lightning's light now, 0 (none) to about 1.5
  m_strike,      // flashes so far (a whole number)
  m_bolt,        // where the last flash struck, -1 to 1 across the scene (for placing its bolt)
  m_since,       // seconds since the last flash began (infinity before the first)
  m_regime,      // the regime it is heading for (0 clear ... 6 dust)
  m_progress,    // how far through its transition, 0 to 1 (1: there)
  kMembers
};
std::span<const std::string_view> member_names();
int member_of(std::string_view word);  // -1: none

// A regime's target and its weather noise.
struct Target {
  float cloud = 0, precip = 0, hail = 0, mean = 0, gust = 0, temp = 0, lightning = 0, fog = 0, dust = 0;
  float cell_time = 240.f;  // the rain cells' e-folding time, s
  float cell_spread = 0.f;  // their log-normal spread (the standard deviation of ln R)
};
const Target& target(Regime r);
inline constexpr float kClearAir = 3.912f / 40.f;  // clear air's extinction, km^-1 (a 40 km day)

class Director {
 public:
  // Starts in `start`'s target (the seed's variant of it), its noise already at its steady spread. `direction`: the
  // wind blows along +x (1) or -x (-1).
  explicit Director(std::uint64_t seed = 1, Regime start = Regime::clear, float direction = 1.f);

  // Head for `to` over `seconds` (at least 1), from the state it is in now, so nothing jumps. Asking for the regime it
  // is already heading for does nothing.
  void request(Regime to, float seconds);
  // One step of dt seconds (a frame): the transition, the noise, the lightning, then every member.
  void step(float dt);

  Regime regime() const { return to_regime_; }
  float progress() const { return p_; }
  float time() const { return time_; }
  int requests() const { return requests_; }
  float value(int m) const { return v_[static_cast<std::size_t>(m)]; }
  const std::array<float, kMembers>& values() const { return v_; }
  // The state without its noise (the transition's blend of the targets), for tests: cloud, precip, hail, mean, gust,
  // temp, lightning, fog, dust.
  const Target& base() const { return base_; }

 private:
  float normal(std::uint32_t channel) const;   // N(0, 1) for this step
  float uniform(std::uint32_t channel) const;  // [0, 1) for this step
  Target variant(Regime r, int request) const;
  void blend();
  void derive();

  std::uint64_t seed_;
  float dir_;
  Regime to_regime_;
  Target from_{}, to_{}, base_{};
  float p_ = 1.f, over_ = 1.f, t0_ = 0.f, time_ = 0.f;
  std::int64_t steps_ = 0;
  int requests_ = 0;
  // the noise: gusts (an OU and its 1-second smoothing), rain cells, cloud, temperature, fog
  float xg_ = 0, yg_ = 0, xr_ = 0, xc_ = 0, xt_ = 0, xf_ = 0;
  // lightning: the last flash (its start, its strokes' times and peaks)
  int strikes_ = 0, strokes_ = 0;
  float flash_t0_ = 0.f;
  std::array<float, kMaxStrokes> stroke_t_{}, stroke_a_{};
  float bolt_ = 0.f;
  std::array<float, kMembers> v_{};
};

}  // namespace nfx::compose::wx
