// The weather director in 2D (weather.hpp, docs/WEATHER.md).
#include "weather.hpp"

#include <neuralfx/noise.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>

namespace nfx::compose::wx {

namespace {

constexpr std::string_view kRegimeNames[] = {"clear", "overcast", "rain", "storm", "snow", "fog", "dust"};
constexpr std::string_view kMemberNames[] = {"cloud", "rain",     "snow",  "hail",     "wind",    "mean",  "gust",   "visibility", "temp",
                                             "lightning", "dust", "fog",   "extinction", "u",     "beaufort", "lean", "leaves", "lift",
                                             "trees", "light", "flash",    "strike", "bolt",     "since",   "regime", "progress",
                                             "humidity", "dewpoint", "base",   "water",    "wet",     "puddle", "evaporation",
                                             "carried", "front", "hour", "elevation", "sunx", "sun", "shaft", "deck", "foot"};
static_assert(std::size(kMemberNames) == static_cast<std::size_t>(kMembers));

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kPi = 3.14159265358979f;

float smooth01(float v) {
  v = std::clamp(v, 0.f, 1.f);
  return v * v * (3.f - 2.f * v);
}

// The regimes' targets (docs/WEATHER.md §2): each a choice inside what is observed for such weather, not a fit. The
// humidity sets the dew point and so the cloud deck's base (§3.11): clear 1200 m, overcast 560 m, rain and snow about
// 200 m, a storm 330 m, fog 0 (saturated: cloud at the ground), a dust storm's dry air 3.8 km.
//                                cloud precip hail mean gust  temp lightning fog    dust  cell_time spread humidity
constexpr Target kTargets[] = {
    {0.10f, 0.f, 0.f, 3.f, 0.20f, 22.f, 0.f, 0.f, 0.f, 300.f, 0.f, 0.55f},            // clear: a few clouds, a light breeze (force 2)
    {0.95f, 0.f, 0.f, 5.f, 0.18f, 17.f, 0.f, 0.f, 0.f, 300.f, 0.f, 0.75f},            // overcast: a gentle breeze (force 3)
    {1.00f, 6.f, 0.f, 7.f, 0.22f, 13.f, 0.f, 0.f, 0.f, 300.f, 0.35f, 0.90f},          // rain: moderate (2.5 to 7.6 mm/h)
    {1.00f, 40.f, 0.6f, 11.f, 0.40f, 16.f, 20.f, 0.f, 0.f, 90.f, 0.5f, 0.85f},        // storm: heavy rain, force 6 gusting to 8, lightning
    {1.00f, 1.5f, 0.f, 4.f, 0.20f, -4.f, 0.f, 0.f, 0.f, 400.f, 0.3f, 0.90f},          // snow: moderate
    {1.00f, 0.f, 0.f, 1.f, 0.10f, 8.f, 0.f, 3.912f / 0.15f - kClearAir, 0.f, 300.f, 0.f, 1.f},  // fog: 150 m, light air, saturated
    {0.30f, 0.f, 0.f, 15.f, 0.30f, 34.f, 0.f, 0.f, 0.6f, 300.f, 0.f, 0.15f},          // dust storm: about 330 m, dry
};
static_assert(std::size(kTargets) == static_cast<std::size_t>(kRegimes));

// The windows of a transition (docs/WEATHER.md §2.2): where in it (0 to 1) each quantity moves, rising and falling. Rising
// they follow the order clouds, gust front (wind, gusts, temperature), rain, lightning and hail; falling the reverse.
struct Window {
  float up0, up1, down0, down1;
};
constexpr Window kCloud{0.f, 0.45f, 0.55f, 1.f}, kPrecip{0.45f, 0.8f, 0.15f, 0.55f}, kHail{0.7f, 0.9f, 0.f, 0.25f},
    kWind{0.3f, 0.5f, 0.35f, 0.8f}, kTemp{0.3f, 0.55f, 0.3f, 0.55f}, kLightning{0.65f, 0.95f, 0.f, 0.3f}, kFog{0.2f, 0.9f, 0.f, 0.6f},
    kDust{0.4f, 0.8f, 0.15f, 0.6f};

float along(float from, float to, float p, const Window& w) {
  const bool up = to > from;
  const float a = up ? w.up0 : w.down0, b = up ? w.up1 : w.down1;
  return from + (to - from) * smooth01((p - a) / (b - a));
}

// The snowflakes' sizes (exponential, slope kFlakeSlope, on [kFlakeMin, kFlakeMax]): their moments, by Simpson's rule.
struct FlakeMoments {
  float n = 0, speed = 0, mass_speed = 0, area = 0;  // <V> (m/s), <M V> (kg m/s), <pi D^2 / 4> (m^2)
};
FlakeMoments flake_moments() {
  constexpr int kSteps = 400;
  double w = 0, v = 0, mv = 0, a = 0;
  const double h = static_cast<double>(kFlakeMax - kFlakeMin) / kSteps;
  for (int i = 0; i <= kSteps; ++i) {
    const double d = static_cast<double>(kFlakeMin) + h * i;
    const double k = (i == 0 || i == kSteps ? 1.0 : i % 2 ? 4.0 : 2.0) * std::exp(-static_cast<double>(kFlakeSlope) * d);
    const double speed = 0.8 * std::pow(d, 0.16), mass = 0.073e-6 * std::pow(d, 1.4);
    w += k;
    v += k * speed;
    mv += k * mass * speed;
    a += k * 0.25 * 3.14159265358979 * d * d * 1e-6;
  }
  return {1.f, static_cast<float>(v / w), static_cast<float>(mv / w), static_cast<float>(a / w)};
}
const FlakeMoments& flakes() {
  static const FlakeMoments m = flake_moments();
  return m;
}

}  // namespace

std::string_view regime_name(Regime r) { return kRegimeNames[static_cast<std::size_t>(r)]; }
bool regime_of(std::string_view word, Regime& out) {
  for (std::size_t i = 0; i < std::size(kRegimeNames); ++i) {
    if (kRegimeNames[i] == word) {
      out = static_cast<Regime>(i);
      return true;
    }
  }
  return false;
}
std::span<const std::string_view> regime_names() { return kRegimeNames; }
std::span<const std::string_view> member_names() { return kMemberNames; }
int member_of(std::string_view word) {
  for (std::size_t i = 0; i < std::size(kMemberNames); ++i)
    if (kMemberNames[i] == word) return static_cast<int>(i);
  return -1;
}
const Target& target(Regime r) { return kTargets[static_cast<std::size_t>(r)]; }

// --- relations ---------------------------------------------------------------------------------------------------------

float visibility_m(float sigma) { return sigma > 0.f ? 1000.f * kKoschmieder / sigma : kInf; }
float extinction_per_km(float v) { return v > 0.f ? 1000.f * kKoschmieder / v : kInf; }

float mp_slope(float R) { return 4.1f * std::pow(std::max(R, 1e-3f), -0.21f); }
float mp_number(float R, float d0, float d1) {
  if (!(R > 0.f)) return 0.f;
  const float L = mp_slope(R);
  return kMpN0 / L * (std::exp(-L * d0) - std::exp(-L * d1));
}
float mp_mean_diameter(float R, float d0, float d1) {
  const float L = mp_slope(R), e0 = std::exp(-L * d0), e1 = std::exp(-L * d1);
  return ((d0 + 1.f / L) * e0 - (d1 + 1.f / L) * e1) / (e0 - e1);
}
float mp_sample(float R, float u, float d0, float d1) {
  const float L = mp_slope(R);
  return std::min(d1, d0 - std::log1p(-u * (1.f - std::exp(-L * (d1 - d0)))) / L);
}
float rain_extinction(float R) {
  if (!(R > 0.f)) return 0.f;
  const float L = mp_slope(R);
  return kPi * kMpN0 * 1e-3f / (L * L * L);  // pi N0 / Lambda^3: m^-3 mm^2 = 1e-6 m^-1 = 1e-3 km^-1
}

float drop_speed(float d) { return std::max(0.f, 9.65f - 10.3f * std::exp(-0.6f * d)); }
float mp_mean_speed(float R, float d0, float d1) {
  const float L = mp_slope(R), k = L + 0.6f;
  const float m = L / k * (std::exp(-k * d0) - std::exp(-k * d1)) / (std::exp(-L * d0) - std::exp(-L * d1));  // <exp(-0.6 D)>
  return 9.65f - 10.3f * m;
}
float mp_atlas_rate(float R) {
  if (!(R > 0.f)) return 0.f;
  const float L = mp_slope(R), k = L + 0.6f;
  // 6 pi 10^-4 N0 integral of D^3 (9.65 - 10.3 exp(-0.6 D)) exp(-Lambda D) dD, with integral of D^3 exp(-a D) = 6 / a^4
  return 6.f * kPi * 1e-4f * kMpN0 * (9.65f * 6.f / (L * L * L * L) - 10.3f * 6.f / (k * k * k * k));
}

float flake_speed(float d) { return 0.8f * std::pow(std::max(d, 0.f), 0.16f); }
float flake_mass_mg(float d) { return 0.073f * std::pow(std::max(d, 0.f), 1.4f); }
float flake_sample(float u) { return std::min(kFlakeMax, kFlakeMin - std::log1p(-u * (1.f - std::exp(-kFlakeSlope * (kFlakeMax - kFlakeMin)))) / kFlakeSlope); }
float flake_mean_speed() { return flakes().speed; }
float snow_number(float S) { return S > 0.f ? S / 3600.f / flakes().mass_speed : 0.f; }
float snow_extinction(float S) { return 2.f * 0.3f * flakes().area * snow_number(S) * 1000.f; }

float hail_speed(float d) { return std::sqrt(4.f * 9.81f * std::max(d, 0.f) * 1e-3f * 900.f / (3.f * 0.6f * 1.2f)); }

float beaufort(float w) { return std::pow(std::max(w, 0.f) / 0.836f, 2.f / 3.f); }
float beaufort_speed(float B) { return 0.836f * std::pow(std::max(B, 0.f), 1.5f); }
std::string_view beaufort_name(int B) {
  static constexpr std::string_view n[] = {"calm",       "light air", "light breeze", "gentle breeze", "moderate breeze", "fresh breeze", "strong breeze",
                                           "near gale", "gale",      "strong gale",  "storm",         "violent storm",   "hurricane"};
  return n[static_cast<std::size_t>(std::clamp(B, 0, 12))];
}
std::string_view beaufort_on_land(int B) {
  static constexpr std::string_view d[] = {
      "smoke rises vertically",
      "direction of wind shown by smoke drift, but not by wind vanes",
      "wind felt on face; leaves rustle; ordinary vanes moved by wind",
      "leaves and small twigs in constant motion; wind extends light flag",
      "raises dust and loose paper; small branches are moved",
      "small trees in leaf begin to sway; crested wavelets form on inland waters",
      "large branches in motion; whistling heard in telegraph wires; umbrellas used with difficulty",
      "whole trees in motion; inconvenience felt when walking against the wind",
      "breaks twigs off trees; generally impedes progress",
      "slight structural damage occurs (chimney pots and slates removed)",
      "trees uprooted; considerable structural damage occurs",
      "very rarely experienced; accompanied by widespread damage",
      "devastation"};
  return d[static_cast<std::size_t>(std::clamp(B, 0, 12))];
}
float lean(float w, float dir) { return 0.5f + 0.5f * dir * std::min(1.f, std::max(w, 0.f) / beaufort_speed(8.f)); }
float leaves(float w) { return smooth01((beaufort(w) - 2.f) / 2.f); }
float lift(float w) { return smooth01((beaufort(w) - 3.5f) / 1.5f); }
float trees(float w) { return smooth01((beaufort(w) - 4.f) / 3.f); }

float daylight(float cloud) { return 1.f - 0.75f * std::pow(std::clamp(cloud, 0.f, 1.f), 3.4f); }
float gust_time(float U) { return std::clamp(kGustLength / std::max(U, 0.5f), 1.f, 30.f); }
float snow_share(float T) { return std::clamp((2.f - T) / 2.f, 0.f, 1.f); }

// Magnus with Alduchov and Eskridge's coefficients; the dew point by its inverse: g = ln(h) + a T / (b + T), Td = b g / (a - g).
float saturation_vapour_hpa(float T) { return 6.1094f * std::exp(17.625f * T / (T + 243.04f)); }
float dew_point(float T, float h) {
  if (h >= 1.f) return T;  // saturated, to the bit
  h = std::max(h, 1e-4f);
  const float g = std::log(h) + 17.625f * T / (T + 243.04f);
  return std::min(T, 243.04f * g / (17.625f - g));
}
float cloud_base_m(float T, float Td) { return std::max(0.f, 125.f * (T - Td)); }

// The deck (docs/WEATHER.md §3.15): a layer from broken cover (a ceiling, ICAO: more than half the sky), its underside seen
// out to the visibility, where it stands atan(base / V) above the horizon, and fading into the haze below by Koschmieder
// along the slant path base / sin(elevation).
float deck_cover(float cloud) { return smooth01((std::clamp(cloud, 0.f, 1.f) - kDeckScattered) / (kDeckBroken - kDeckScattered)); }
float deck_foot_deg(float base_m, float visibility_m) {
  if (!(base_m > 0.f)) return 0.f;
  if (!(visibility_m > 0.f)) return 90.f;
  return std::atan(base_m / visibility_m) * (180.f / kPi);  // smooth where the base passes the visibility (asin would pin at 90)
}
float deck_contrast(float base_m, float visibility_m, float elevation_deg) {
  if (!(elevation_deg > 0.f)) return 0.f;
  if (!(base_m > 0.f)) return 1.f;
  if (!(visibility_m > 0.f)) return 0.f;
  return std::exp(-kKoschmieder * base_m / (visibility_m * std::sin(std::min(elevation_deg, 90.f) * (kPi / 180.f))));
}

// Penman's drying power of the air, E_a = 0.2625 (1 + 0.537 U_2) (e_s - e) mm/day (e in hPa), in mm/h.
float evaporation_mm_h(float T, float h, float u2) {
  const float deficit = saturation_vapour_hpa(T) * (1.f - std::clamp(h, 0.f, 1.f));
  return 0.2625f / 24.f * (1.f + 0.537f * std::max(u2, 0.f)) * deficit;
}
float wind_2m(float u10) { return kWind2m * u10; }

float taylor_length_m(float U, float t) { return std::max(U, 0.f) * std::max(t, 0.f); }

// The sun: the hour angle H = 15 (hour - 12) degrees; on the equinox at latitude phi, sin(elevation) = cos(phi) cos(H)
// and the sun's east-west component is -sin(H), so across a picture facing south it stands at sin(H).
namespace {
float hour_angle_rad(float hour) {
  const float h = hour - 24.f * std::floor(hour / 24.f);  // wraps
  return (h - 12.f) * (15.f * kPi / 180.f);
}
}  // namespace
float sun_elevation_deg(float hour) {
  const float s = std::cos(kLatitude * kPi / 180.f) * std::cos(hour_angle_rad(hour));
  return std::asin(std::clamp(s, -1.f, 1.f)) * (180.f / kPi);
}
float sun_across(float hour) { return std::sin(hour_angle_rad(hour)); }
float sun_share(float cloud) {
  const float c = std::clamp(cloud, 0.f, 1.f);
  return 0.7f * (1.f - c * c);  // 1 - D/G, Kasten and Czeplak (1980): D/G = 0.3 + 0.7 (N/8)^2
}
float beam_transmittance(float sigma, float elevation_deg) {
  if (!(elevation_deg > 0.f)) return 0.f;
  return std::exp(-std::max(sigma, 0.f) * kWeatherLayer / std::sin(elevation_deg * (kPi / 180.f)));
}
float shaft_visibility(float cloud, float sigma, float elevation_deg) {
  const float c = std::clamp(cloud, 0.f, 1.f);
  const float lit = 1.f - std::exp(-std::max(sigma, 0.f) * kShaftDepth);  // what the lit air scatters toward the eye
  return std::clamp(4.f * c * (1.f - c) * beam_transmittance(sigma, elevation_deg) * lit, 0.f, 1.f);
}

// --- the director --------------------------------------------------------------------------------------------------------

namespace {
// Channels of the step's random numbers.
enum : std::uint32_t { c_flash, c_strokes, c_count, c_gap, c_peak, c_bolt, c_noise = 20 };  // a noise: c_noise + 2 i, + 1
enum : int { n_gust, n_rain, n_cloud, n_temp, n_fog };
constexpr float kSmooth[] = {1.f, 3.f, 5.f, 5.f, 3.f};  // the noises' smoothing, s (gusts: as 3-second means)
constexpr float kStroke = 0.045f;  // a stroke's light decays over this, s
constexpr float kDustTime = 5.f;   // raised dust follows the gusts over this, s (a choice: it mixes, it settles)
}  // namespace

float Director::noise_time(int i) const {
  switch (i) {
    case n_gust: return gust_time(base_.mean);
    case n_rain: return base_.cell_time;
    case n_cloud: return 90.f;
    case n_temp: return 120.f;
    default: return 40.f;
  }
}
// An OU process of unit variance smoothed over s seconds has the variance tau / (tau + s): brought back to 1.
float Director::noise(int i) const {
  const float tau = noise_time(i);
  return noise_[static_cast<std::size_t>(i)].y / std::sqrt(tau / (tau + kSmooth[i]));
}

float Director::uniform(std::uint32_t ch) const {
  const std::uint32_t h = hash_cell(static_cast<std::int32_t>(steps_ & 0x7fffffff), static_cast<std::int32_t>(ch), 41, seed_);
  return (static_cast<float>(h >> 8) + 0.5f) * (1.f / 16777216.f);
}
float Director::normal(std::uint32_t ch) const {
  const float u1 = uniform(2 * ch + 100), u2 = uniform(2 * ch + 101);
  return std::sqrt(-2.f * std::log(u1)) * std::cos(2.f * kPi * u2);
}

Target Director::variant(Regime r, int k) const {
  const auto g = [&](std::int32_t ch) {  // N(0, 1) for this request
    const std::uint32_t a = hash_cell(k, ch, 77, seed_), b = hash_cell(k, ch, 78, seed_);
    const float u1 = (static_cast<float>(a >> 8) + 0.5f) * (1.f / 16777216.f), u2 = static_cast<float>(b >> 8) * (1.f / 16777216.f);
    return std::sqrt(-2.f * std::log(u1)) * std::cos(2.f * kPi * u2);
  };
  Target t = target(r);
  t.cloud = std::clamp(t.cloud + 0.04f * g(0), 0.f, 1.f);
  t.precip *= std::exp(0.25f * g(1));
  t.hail = std::min(1.f, t.hail * std::exp(0.2f * g(2)));
  t.mean *= std::exp(0.15f * g(3));
  t.gust *= std::exp(0.1f * g(4));
  t.temp += 1.5f * g(5);
  t.lightning *= std::exp(0.3f * g(6));
  t.fog *= std::exp(0.25f * g(7));
  t.dust *= std::exp(0.2f * g(8));
  t.humidity = std::clamp(t.humidity + 0.08f * (1.f - t.humidity) * g(9), 0.f, 1.f);  // saturated air (fog) stays saturated
  return t;
}

Director::Director(std::uint64_t seed, Regime start, float direction) : seed_(seed), dir_(direction < 0.f ? -1.f : 1.f), to_regime_(start) {
  to_ = from_ = base_ = variant(start, 0);
  // the noise at its steady spread (each smoothing correlated with its process as it is in their steady state)
  for (int i = 0; i < static_cast<int>(noise_.size()); ++i) {
    Noise& n = noise_[static_cast<std::size_t>(i)];
    const float tau = noise_time(i), c = tau / (tau + kSmooth[i]);
    n.x = normal(c_noise + 2 * static_cast<std::uint32_t>(i));
    n.y = c * n.x + std::sqrt(c - c * c) * normal(c_noise + 2 * static_cast<std::uint32_t>(i) + 1);
  }
  derive();
}

void Director::request(Regime to, float seconds) {
  if (to == to_regime_) return;
  from_ = base_;
  to_regime_ = to;
  to_ = variant(to, ++requests_);
  over_ = std::max(1.f, seconds);
  t0_ = time_;
  p_ = 0.f;
  front0_ = carried_;  // the new weather's front is at the camera; the wind takes it downwind from here
}

void Director::blend() {
  const float p = p_;
  base_.cloud = along(from_.cloud, to_.cloud, p, kCloud);
  base_.precip = along(from_.precip, to_.precip, p, kPrecip);
  base_.hail = along(from_.hail, to_.hail, p, kHail);
  base_.mean = along(from_.mean, to_.mean, p, kWind);
  base_.gust = along(from_.gust, to_.gust, p, kWind);
  base_.temp = along(from_.temp, to_.temp, p, kTemp);
  base_.lightning = along(from_.lightning, to_.lightning, p, kLightning);
  base_.fog = along(from_.fog, to_.fog, p, kFog);
  base_.dust = along(from_.dust, to_.dust, p, kDust);
  base_.humidity = along(from_.humidity, to_.humidity, p, kCloud);  // the air moistens as the clouds build, dries as they break
  // the rain cells change with the rain
  const Window& w = kPrecip;
  const bool up = to_.precip > from_.precip;
  const float k = smooth01((p - (up ? w.up0 : w.down0)) / ((up ? w.up1 : w.down1) - (up ? w.up0 : w.down0)));
  base_.cell_time = from_.cell_time + (to_.cell_time - from_.cell_time) * k;
  base_.cell_spread = from_.cell_spread + (to_.cell_spread - from_.cell_spread) * k;
}

void Director::step(float dt) {
  if (!(dt > 0.f)) return;
  ++steps_;
  dt_ = dt;
  time_ += dt;
  if (p_ < 1.f) {
    p_ = std::min(1.f, (time_ - t0_) / over_);
    if (p_ < 1.f) blend();
    else base_ = to_;  // there, to the bit
  }
  // Taylor's hypothesis: the mean wind of the moment carries the weather along x (docs/WEATHER.md §3.13)
  carried_ += dir_ * base_.mean * dt;
  // the noise: each an Ornstein-Uhlenbeck process stepped exactly (x' = a x + sqrt(1 - a^2) n, a = exp(-dt / tau)), then
  // its smoothing (y' = b y + (1 - b) x, b = exp(-dt / s))
  for (int i = 0; i < static_cast<int>(noise_.size()); ++i) {
    Noise& n = noise_[static_cast<std::size_t>(i)];
    const float a = std::exp(-dt / noise_time(i)), b = std::exp(-dt / kSmooth[i]);
    n.x = a * n.x + std::sqrt(std::max(0.f, 1.f - a * a)) * normal(c_noise + 2 * static_cast<std::uint32_t>(i));
    n.y = b * n.y + (1.f - b) * n.x;
  }
  derive();
  // lightning: a Poisson process at the rate of the moment; a new flash takes over from the last
  const float rate = v_[m_lightning] / 60.f;
  if (rate > 0.f && uniform(c_flash) < 1.f - std::exp(-rate * dt)) {
    ++strikes_;
    flash_t0_ = time_;
    const auto u = [&](std::uint32_t ch, int i) {
      const std::uint32_t h = hash_cell(strikes_, static_cast<std::int32_t>(ch), i, seed_);
      return static_cast<float>(h >> 8) * (1.f / 16777216.f);
    };
    strokes_ = u(c_strokes, 0) < 0.17f ? 1 : 2 + std::min(3, static_cast<int>(4.f * u(c_count, 0)));
    float at = 0.f;
    for (int i = 0; i < strokes_; ++i) {
      if (i > 0) at += 0.04f * std::pow(2.5f, u(c_gap, i));  // 40 to 100 ms
      stroke_t_[static_cast<std::size_t>(i)] = at;
      stroke_a_[static_cast<std::size_t>(i)] = i == 0 ? 1.f : 0.5f + 0.4f * u(c_peak, i);
    }
    bolt_ = 2.f * u(c_bolt, 0) - 1.f;
  }
  float flash = 0.f;
  if (strikes_ > 0) {
    const float since = time_ - flash_t0_;
    for (int i = 0; i < strokes_; ++i) {
      const float s = since - stroke_t_[static_cast<std::size_t>(i)];
      if (s >= 0.f && s < 20.f * kStroke) flash += stroke_a_[static_cast<std::size_t>(i)] * std::exp(-s / kStroke);
    }
  }
  v_[m_flash] = std::min(1.5f, flash);
  v_[m_strike] = static_cast<float>(strikes_);
  v_[m_bolt] = bolt_;
  v_[m_since] = strikes_ > 0 ? time_ - flash_t0_ : kInf;
}

void Director::derive() {
  const Target& B = base_;
  const float g = noise(n_gust);
  const float wind = std::max(0.f, B.mean * (1.f + B.gust * g));
  const float s = B.cell_spread, r = std::exp(s * noise(n_rain) - 0.5f * s * s);
  const float temp = B.temp + 0.3f * noise(n_temp);
  const float precip = B.precip * r, fs = snow_share(temp);
  const float cloud = std::clamp(B.cloud + 0.6f * B.cloud * (1.f - B.cloud) * noise(n_cloud), 0.f, 1.f);
  const float gust_up = std::max(0.f, 1.f + B.gust * g);
  v_[m_cloud] = cloud;
  v_[m_rain] = precip * (1.f - fs);
  v_[m_snow] = precip * fs;
  v_[m_hail] = std::min(1.f, B.hail * smooth01((r - 0.9f) / 0.6f));  // in the heaviest cells
  v_[m_wind] = wind;
  v_[m_mean] = B.mean;
  v_[m_gust] = B.gust;
  v_[m_temp] = temp;
  v_[m_lightning] = B.lightning * r * r * std::exp(-s * s);  // with the heaviest rain (its mean kept: E[r^2] = exp(s^2))
  // dust: raised as the wind's cube (as sand's flux goes as u*^3, Bagnold 1941), mixed over a few seconds
  const float raised = B.dust * std::min(3.f, gust_up * gust_up * gust_up / (1.f + 3.f * B.gust * B.gust));
  dust_ = steps_ == 0 ? raised : dust_ + (raised - dust_) * (1.f - std::exp(-dt_ / kDustTime));
  v_[m_dust] = dust_;
  v_[m_fog] = B.fog * std::exp(0.2f * noise(n_fog) - 0.02f);
  v_[m_extinction] = kClearAir + v_[m_fog] + rain_extinction(v_[m_rain]) + snow_extinction(v_[m_snow]) + v_[m_dust] * (kKoschmieder / 0.2f);
  v_[m_visibility] = visibility_m(v_[m_extinction]);
  v_[m_u] = dir_ * wind;
  v_[m_beaufort] = beaufort(wind);
  v_[m_lean] = lean(wind, dir_);
  v_[m_leaves] = leaves(wind);
  v_[m_lift] = lift(wind);
  v_[m_trees] = trees(wind);
  v_[m_light] = daylight(cloud);
  v_[m_regime] = static_cast<float>(static_cast<int>(to_regime_));
  v_[m_progress] = p_;
  // moist air: the humidity is the blend's (no noise of its own); the dew point follows the temperature's noise at that
  // humidity, so the cloud base, 125 m a degree of the spread, hardly moves with it
  v_[m_humidity] = B.humidity;
  v_[m_dewpoint] = dew_point(temp, B.humidity);
  v_[m_base] = cloud_base_m(temp, v_[m_dewpoint]);
  // the deck (§3.15): the cover as a layer at that base (the cloud's noise carries through), its underside seen out to
  // the visibility, where it meets the haze
  v_[m_deck] = deck_cover(cloud);
  v_[m_foot] = deck_foot_deg(v_[m_base], v_[m_visibility]);
  // wet ground: the rain fills the film up to the depression storage (the excess runs off), the air dries it by Penman's
  // wind function (nothing in saturated air: fog keeps it wet). Snow does not wet it (nothing lies yet). It starts as
  // long such weather leaves it: full where the rain beats the drying, dry elsewhere.
  const float evap = evaporation_mm_h(temp, B.humidity, wind_2m(wind));
  if (steps_ == 0) water_ = v_[m_rain] > evap ? kDepressionStorage : 0.f;
  else water_ = std::clamp(water_ + (v_[m_rain] - evap) * dt_ / 3600.f, 0.f, kDepressionStorage);
  v_[m_water] = water_;
  v_[m_wet] = smooth01(water_ / kWetFilm);
  v_[m_puddle] = water_ / kDepressionStorage;
  v_[m_evaporation] = evap;
  // the storm across the scene: where the wind has carried the air, and the front of the weather it heads for with it
  v_[m_carried] = carried_;
  v_[m_front] = carried_ - front0_;
  sun();
  if (steps_ == 0) {  // before the first step: no flash yet
    v_[m_flash] = 0.f;
    v_[m_strike] = 0.f;
    v_[m_bolt] = 0.f;
    v_[m_since] = kInf;
  }
}

// The sun and its shafts (docs/WEATHER.md §3.14): where it stands from the hour; the direct beam's share of the daylight
// by Kasten and Czeplak's diffuse fraction, through the weather along its slant path; the shafts where the broken deck's
// beams light hazy air. Nothing here has noise of its own: the cloud's and the extinction's carry through.
void Director::sun() {
  const float elevation = sun_elevation_deg(hour_), sigma = v_[m_extinction], cloud = v_[m_cloud];
  v_[m_hour] = hour_;
  v_[m_elevation] = elevation;
  v_[m_sunx] = sun_across(hour_);
  v_[m_sun] = sun_share(cloud) * daylight(cloud) * beam_transmittance(sigma, elevation);
  v_[m_shaft] = shaft_visibility(cloud, sigma, elevation);
}

void Director::set_hour(float hour) {
  hour_ = std::isfinite(hour) ? hour : 12.f;
  sun();
}

}  // namespace nfx::compose::wx
