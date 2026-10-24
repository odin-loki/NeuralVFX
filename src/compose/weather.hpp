// The weather director in 2D (namespace wx: docs/WEATHER.md): a small weather state that moves between regimes and drives the layers
// that exist (the rain, snow and hail emitters, the fog and cloud models, the storm layers, the wind every module feels,
// the sky's light and the lightning's flashes).
//
//   - The state: cloud cover, rain and snow rates, hail, mean wind and gustiness, visibility, temperature, the rate of
//     lightning, the load of dust (and the fog's extinction, from which, with the rain's, the snow's and the dust's, the
//     visibility follows), the relative humidity (the dew point and the cloud base follow) and the water standing on
//     the ground (the rain fills it, Penman's evaporation dries it: the ground's wetness and its puddles follow).
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
//   - The storm moves across the scene (Taylor's hypothesis, docs/WEATHER.md §3.13): the state is the weather at the
//     camera, and the mean wind carries it along x, so the distance it has carried the air past the camera places the
//     pattern in the scene, and a transition's front (where the new weather begins) is that distance since the request.
//   - The cloud deck is a layer (docs/WEATHER.md §3.15): the cover is a deck once it is more than half the sky (aviation's
//     ceiling), standing at the cloud base; its underside is seen out to the visibility, where it meets the haze (its
//     foot), and fades below by Koschmieder along the slant path.
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
// and loose paper" is force 4), trees swaying (0 up to force 4, "small branches are moved"; 1 from force 7, "whole trees
// in motion"; force 5 "small trees in leaf begin to sway" and 6 "large branches in motion" lie between). The sway is
// the amplitude a foliage layer bends by (a hook: nothing draws trees yet); its direction is the wind's, `u`'s sign.
float lean(float wind_ms, float direction);
float leaves(float wind_ms);
float lift(float wind_ms);
float trees(float wind_ms);

// Kasten and Czeplak (1980): the global irradiance under N oktas of cloud is G(0) (1 - 0.75 (N/8)^3.4), cloud = N/8.
float daylight(float cloud);

// Gusts: the integral length scale of the longitudinal wind at 10 m, Kaimal's spectrum as IEC 61400-1 gives it,
// L_u = 8.1 x 0.7 z = 56.7 m; its integral time scale is L_u / U (1 to 30 s).
inline constexpr float kGustLength = 56.7f;
float gust_time(float mean_ms);

// Rain and snow: the share that falls as snow, a ramp from all rain at 2 C to all snow at 0 C (a simple partition, not
// a fit to observations).
float snow_share(float temp_c);

// Moist air (the cloud deck's first relation: where its base is). The saturation vapour pressure over water by the Magnus
// formula with Alduchov and Eskridge's (1996) coefficients, e_s = 6.1094 exp(17.625 T / (T + 243.04)) hPa (within 0.4 %
// of the measured values from -40 to 50 C); the dew point its inverse at the humidity's share of it; the cloud base
// (the lifting condensation level) 125 m for each degree the dew point is below the air (Espy's rule: a parcel cools at
// 9.8 K/km and its dew point falls at about 1.8 K/km, so the two meet at 125 m/K; Lawrence 2005).
float saturation_vapour_hpa(float temp_c);
float dew_point(float temp_c, float humidity);     // humidity 0 to 1 (relative); the dew point never above the air's temperature
float cloud_base_m(float temp_c, float dew_point_c);  // 125 (T - Td), at least 0: saturated air is cloud at the ground (fog)
// The cloud deck as a layer (its second relation, docs/WEATHER.md §3.15: when the cover is a layer, and where that layer
// stands in the picture). A ceiling, in aviation's sense (ICAO Annex 3; FMH-1 chapter 9), is the base of the lowest layer
// covering more than half the sky, broken (5 to 7 oktas) or overcast (8); up to scattered (3 to 4 oktas) the clouds are
// puffs with sky between them and no layer. So the deck is the cover's share as a layer, a smoothstep from 4 oktas (none:
// the puffs alone) to 6 (a layer; the ramp's width a choice), standing at `cloud_base_m`. Its underside is a horizontal
// plane at that height: at horizontal distance x it stands atan(base / x) above the horizon, along a slant path
// base / sin(elevation), and by Koschmieder (§3.1) its contrast against the horizon sky is exp(-sigma base / sin(elevation)),
// 2 % where that path is the visibility V. So the deck is seen out to the visibility, where its underside stands
// atan(base / V) above the horizon (the deck's foot; the atan, the elevation at the visibility's horizontal distance,
// rather than the 2 % threshold's asin(base / V), which they agree with to a degree below 20 degrees and which pins at
// 90 with an infinite slope once the base is beyond the visibility), and the haze takes it below that: fair-weather
// cumulus seen 40 km (1200 m: 1.7 degrees) reach the horizon, a storm's low deck in its rain is lost below about 20,
// fog's is at the ground (base 0), a dust storm's base beyond its visibility is not seen at all.
inline constexpr float kDeckScattered = 0.5f, kDeckBroken = 0.75f;  // 4 and 6 oktas: the cover becomes a layer between them
float deck_cover(float cloud);                          // the cover as a layer, 0 (scattered puffs) to 1 (a deck)
float deck_foot_deg(float base_m, float visibility_m);  // atan(base / V): the underside at the visibility, degrees above the horizon
float deck_contrast(float base_m, float visibility_m, float elevation_deg);  // the underside's contrast against the haze, exp(-3.912 base / (V sin(elevation))): 0 at or below the horizon, 1 when the base is at the ground

// Wet ground (its first relation: how much water stands on the ground and how fast it dries). The rain fills a film of
// water up to the surface's depression storage (what exceeds it runs off; an impervious surface holds 1.3 to 2.5 mm,
// ASCE 1992: 1.5 mm here); the film evaporates by Dalton's law with Penman's (1948) wind function, the aerodynamic term
// of his open-water evaporation, E_a = 0.2625 (1 + 0.537 U_2)(e_s(T) - e) mm/day (his 0.35 (1 + u/100) mm/day with u in
// miles a day and e in mmHg, converted), U_2 the wind at 2 m in m/s and e_s - e the vapour pressure deficit in hPa.
// Saturated air dries nothing: fog keeps the ground wet. The ground darkens when a film fills its roughness (Lekner and
// Dorf 1988): 0.1 mm here (a choice).
inline constexpr float kDepressionStorage = 1.5f;  // mm
inline constexpr float kWetFilm = 0.1f;             // mm: the film that darkens the ground
inline constexpr float kWind2m = 0.723f;            // the 2 m wind over the 10 m wind: a log profile over grass, z0 = 3 cm
float evaporation_mm_h(float temp_c, float humidity, float wind_2m_ms);
float wind_2m(float wind_10m_ms);  // kWind2m times it

// The sun and its shafts (their first relation: where the sun stands, how much of it reaches the ground, and whether
// its shafts show). Where it stands follows from the solar hour alone, at a latitude of 45 degrees north on the equinox
// (a choice: nothing in the state says the place or the date): sin(elevation) = cos(latitude) cos(H), H = 15 (hour - 12)
// degrees (the standard geometry: Iqbal 1983), and across the picture the sun stands at sin(H), -1 (rising, on the left
// of a camera facing south) to 1 (setting, on the right). How much of it reaches the ground: Kasten and Czeplak's (1980)
// diffuse fraction, D/G = 0.3 + 0.7 (N/8)^2, so the direct beam is 0.7 (1 - cloud^2) of the global, times the daylight
// under the cloud and Beer-Lambert's transmittance exp(-sigma L / sin(elevation)) of the weather's extinction along the
// slant path through the layer it fills (L a choice: 300 m, a radiation fog's depth). The shafts (crepuscular rays)
// are lit air beside shadowed air: they need the cloud's shadows and the gaps' beams (4 c (1 - c): most in a half-broken
// deck, none under a clear or an overcast sky), the beam through the weather, and air that scatters enough of it to be
// seen, 1 - exp(-sigma D) over the depth of lit air along the line of sight (D a choice: 500 m), so clear air shows
// hardly any and haze, mist or dust show them.
inline constexpr float kLatitude = 45.f;     // degrees north: a choice
inline constexpr float kWeatherLayer = 0.3f;  // km: the layer the weather's extinction fills along the beam's slant path
inline constexpr float kShaftDepth = 0.5f;    // km: the depth of lit air along the line of sight through a shaft
float sun_elevation_deg(float hour);   // degrees above the horizon (negative at night); the hour wraps at 24
float sun_across(float hour);          // -1 to 1 across the picture: sin(H)
float sun_share(float cloud);          // the direct beam's share of the global irradiance, 0.7 (1 - cloud^2) (Kasten and Czeplak)
float beam_transmittance(float extinction_per_km, float elevation_deg);  // exp(-sigma kWeatherLayer / sin(elevation)); 0 at or below the horizon
float shaft_visibility(float cloud, float extinction_per_km, float elevation_deg);  // 0 to 1

// Taylor's hypothesis (1938): a pattern the mean wind U carries past a fixed point changes little while it passes, so
// what the point sees over a time t is the pattern's length U t along the wind. The director's state is the weather at
// the camera; the wind carries it along x, so a rain cell that lasts its e-folding time at the camera is that long
// along the wind (a storm's, 90 s at 11 m/s: about 1 km), and a transition over T seconds is a band U T wide crossing
// the scene. It holds where the gusts are small against the mean (a turbulence intensity well under 1: the regimes'
// are 0.1 to 0.4).
float taylor_length_m(float mean_ms, float seconds);  // U t

// Lightning: a flash has 1 to 5 return strokes (one stroke 17 % of the time; 3 on average), 40 to 100 ms apart (Rakov and
// Uman 2003: typically 3 to 5, about 60 ms), each a flash of light decaying over 45 ms.
inline constexpr int kMaxStrokes = 5;

// --- the state -----------------------------------------------------------------------------------------------------------

// What a script reads as NAME.member (and an engine with nvfx_scene_get_input(scene, "NAME.member")). The first twelve are
// the state (with the humidity and the water on the ground); the rest follow from it.
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
  m_trees,       // trees(): the trees' sway, 0 (up to force 4) to 1 (from force 7, "whole trees in motion")
  m_light,       // daylight(cloud): Kasten and Czeplak
  m_flash,       // the lightning's light now, 0 (none) to about 1.5
  m_strike,      // flashes so far (a whole number)
  m_bolt,        // where the last flash struck, -1 to 1 across the scene (for placing its bolt)
  m_since,       // seconds since the last flash began (infinity before the first)
  m_regime,      // the regime it is heading for (0 clear ... 6 dust)
  m_progress,    // how far through its transition, 0 to 1 (1: there)
  m_humidity,    // relative humidity at the ground, 0 to 1 (1 in fog)
  m_dewpoint,    // dew_point(temp, humidity), C
  m_base,        // cloud_base_m(temp, dewpoint): the cloud deck's base, m above the ground (0 in fog)
  m_water,       // the water standing on the ground, mm (0 to kDepressionStorage: the rain fills it, Penman dries it)
  m_wet,         // the ground's wetness, 0 dry to 1 (the film fills its roughness and it darkens)
  m_puddle,      // the share of the depression storage filled: puddles, 0 to 1
  m_evaporation, // evaporation_mm_h(temp, humidity, wind_2m(wind)): what dries the ground now, mm/h
  m_carried,     // the air the mean wind has carried past the camera since the start, m along +x (Taylor's coordinate:
                 // the weather now at x metres along +x is the camera's when this was `carried - x`)
  m_front,       // where the transition's front (the first of the weather it heads for) is, m along +x from the camera:
                 // at the camera at the request, then downwind with the mean wind (`carried` since the request)
  m_hour,        // the solar hour, 0 to 24 (set by the script's `hour`, default 12: noon)
  m_elevation,   // sun_elevation_deg(hour): the sun's elevation, degrees (negative at night)
  m_sunx,        // sun_across(hour): where the sun stands across the picture, -1 (left, rising) to 1 (right, setting)
  m_sun,         // the direct sunlight on the ground, a share of a clear day's daylight: sun_share(cloud) x daylight(cloud) x beam_transmittance (0 under full cover, at night)
  m_shaft,       // shaft_visibility(cloud, extinction, elevation): the sun's shafts, 0 (none) to 1
  m_deck,        // deck_cover(cloud): how much of the sky is a deck (0 puffs, 1 overcast)
  m_foot,        // deck_foot_deg(base, visibility): the deck underside's foot, degrees above the horizon
  kMembers
};
std::span<const std::string_view> member_names();
int member_of(std::string_view word);  // -1: none

// A regime's target and its weather noise.
struct Target {
  float cloud = 0, precip = 0, hail = 0, mean = 0, gust = 0, temp = 0, lightning = 0, fog = 0, dust = 0;
  float cell_time = 240.f;  // the rain cells' e-folding time, s
  float cell_spread = 0.f;  // their log-normal spread (the standard deviation of ln R)
  float humidity = 0.5f;    // relative humidity at the ground, 0 to 1
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
  // The solar hour (0 to 24, wrapping; default 12, noon): where the sun stands, and so the sun's members, at once. A
  // script sets it each frame from its `hour` expression; the weather's noise and transitions do not move it.
  void set_hour(float hour);
  float hour() const { return hour_; }

  Regime regime() const { return to_regime_; }
  float progress() const { return p_; }
  float time() const { return time_; }
  int requests() const { return requests_; }
  float value(int m) const { return v_[static_cast<std::size_t>(m)]; }
  const std::array<float, kMembers>& values() const { return v_; }
  // The state without its noise (the transition's blend of the targets), for tests: cloud, precip, hail, mean, gust,
  // temp, lightning, fog, dust, humidity.
  const Target& base() const { return base_; }
  const Target& heading() const { return to_; }  // the target it heads for: its regime's, as this request varied it

 private:
  float normal(std::uint32_t channel) const;   // N(0, 1) for this step
  float uniform(std::uint32_t channel) const;  // [0, 1) for this step
  Target variant(Regime r, int request) const;
  void blend();
  void derive();
  void sun();  // the sun's members from the hour, the cloud and the extinction

  std::uint64_t seed_;
  float dir_;
  float hour_ = 12.f;
  Regime to_regime_;
  Target from_{}, to_{}, base_{};
  float p_ = 1.f, over_ = 1.f, t0_ = 0.f, time_ = 0.f, dt_ = 0.f;
  std::int64_t steps_ = 0;
  int requests_ = 0;
  // The noise: gusts, rain cells, cloud, temperature and fog, each an OU process x and its smoothing y (a second or a
  // few: gusts are measured as 3-second means), so the state is smooth from frame to frame.
  struct Noise {
    float x = 0, y = 0;
  };
  std::array<Noise, 5> noise_{};
  float noise_time(int i) const;  // its OU's time scale now, s
  float dust_ = 0;                // the dust load, following what the gusts raise over a few seconds
  float water_ = 0;               // the water on the ground, mm
  float carried_ = 0;             // the air carried past the camera, m along +x (the mean wind's integral, signed)
  float front0_ = 0;              // `carried_` at the last request: the front has gone `carried_ - front0_` since
  float noise(int i) const;       // its smoothed value, at unit variance
  // lightning: the last flash (its start, its strokes' times and peaks)
  int strikes_ = 0, strokes_ = 0;
  float flash_t0_ = 0.f;
  std::array<float, kMaxStrokes> stroke_t_{}, stroke_a_{};
  float bolt_ = 0.f;
  std::array<float, kMembers> v_{};
};

}  // namespace nfx::compose::wx
