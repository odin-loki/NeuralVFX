// The weather director in 2D (src/compose/weather.hpp, docs/WEATHER.md): the relations it is built on, against their
// papers and against measurements; its noise (gusts and rain cells) against the processes it says it is; determinism; no
// jumps in its state; each regime reaching its targets; transitions in the order real weather keeps; the script's weather
// statement, its members and the engine's reading of them through the scene API; the physical rain, snow and hail; the
// day's sky; the scene's random numbers kept; the same frames on any number of threads; and no allocation per frame
// (this program replaces operator new to count them, as test_upscale.cpp does).
#include "compose_scene.hpp"
#include "library.hpp"
#include "script.hpp"
#include "weather.hpp"

#include <neuralfx/nvfx_library.h>
#include <neuralfx/nvfx_scene.h>
#include <neuralfx/rollout.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>
#include <functional>
#include <limits>
#include <new>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::atomic<long> g_allocations{0};
std::atomic<bool> g_counting{false};
}  // namespace

void* operator new(std::size_t n) {
  if (g_counting.load(std::memory_order_relaxed)) g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) {
  if (g_counting.load(std::memory_order_relaxed)) g_allocations.fetch_add(1, std::memory_order_relaxed);
  const std::size_t al = static_cast<std::size_t>(a);
  if (void* p = std::aligned_alloc(al, (n + al - 1) / al * al + (n ? 0 : al))) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n, std::align_val_t a) { return operator new(n, a); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

namespace cm = nfx::compose;
namespace sc = nfx::compose::script;
namespace wx = nfx::compose::wx;
using wx::Regime;

constexpr float kFps = 30.f, kDt = 1.f / kFps;
constexpr std::uint64_t kSeed = 28500200;  // the weather director's seeds: 28,500,000 to 28,599,999

double simpson(const std::function<double(double)>& f, double a, double b, int n = 2000) {
  const double h = (b - a) / n;
  double s = f(a) + f(b);
  for (int i = 1; i < n; ++i) s += (i % 2 ? 4.0 : 2.0) * f(a + h * i);
  return s * h / 3.0;
}

double mean(const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); }
double stddev(const std::vector<double>& v) {
  const double m = mean(v);
  double s = 0;
  for (const double x : v) s += (x - m) * (x - m);
  return std::sqrt(s / static_cast<double>(v.size()));
}
double autocorrelation(const std::vector<double>& v, std::size_t lag) {
  const double m = mean(v), s = stddev(v);
  double c = 0;
  for (std::size_t i = 0; i + lag < v.size(); ++i) c += (v[i] - m) * (v[i + lag] - m);
  return c / static_cast<double>(v.size() - lag) / (s * s);
}

// Steps a director for `seconds`, calling f after each step.
void run(wx::Director& d, float seconds, const std::function<void(const wx::Director&)>& f = {}, float dt = kDt) {
  const int n = static_cast<int>(std::lround(seconds / dt));
  for (int i = 0; i < n; ++i) {
    d.step(dt);
    if (f) f(d);
  }
}

sc::Scene scene_of(const std::string& text, int threads = 2, bool overlap = false) {
  sc::Options o;
  o.threads = threads;
  o.overlap = overlap;
  return sc::Scene(sc::parse(text, "t"), [](const std::string& f) { return cm::testing::tiny_effect(f == "other" ? 4 : 3); }, o);
}

std::string error_of(const std::string& text) {
  try {
    sc::Scene s = scene_of(text);
  } catch (const sc::Error& e) {
    return e.what();
  }
  return "no error";
}

// A scene with every layer the director drives (the tiny effect stands for the clouds and the smoke).
constexpr const char* kWeatherScene = R"(
scene size 200 x 120, fps 30, length 5 s, ground 115
particles capacity 20000
effect tiny = "tiny"
weather sky, seed 28500200, regime clear:
  storm at 0.5, over 2
  snow at 3.5, over 1
module m = tiny, size 32, width 64, at (100, 110), start 1, seed 28500202, wind sky.lean, opacity sky.cloud
emit rain, in (0, 0), size 200 x 110, rate sky.rain, scale 300, wind (sky.u * 30, 0), wash 0.01 * sky.rain, seed 28500204
emit snow, in (0, 0), size 200 x 110, rate sky.snow, scale 300, wind (sky.u * 30, 0), seed 28500206
emit hail, in (0, 0), size 200 x 110, density 2 * sky.hail, speed 600, seed 28500208
emit leaves, in (0, 0), size 200 x 110, density 0.5 * sky.leaves, wind (sky.u * 10, 10), seed 28500210
emit arc, along (100 + 80 * sky.bolt, 0) to (100, 110), rate 0.001, seed 28500212 + 2 * sky.strike, brightness 3 * sky.flash
field inside = lamp, at (100 + 80 * sky.bolt, 20), radius 60, strength 2 * sky.flash
field breeze = wind, velocity (0.02 * sky.u, 0), on m, particles
light flash (sky.flash, sky.flash, sky.flash)
frame daylight sky.light, overcast sky.cloud, visibility sky.visibility, wet sky.wet, puddle sky.puddle, deck sky.deck, foot sky.foot
)";

// --- the relations ------------------------------------------------------------------------------------------------------

TEST(WeatherRelations, KoschmiedersVisibility) {
  EXPECT_FLOAT_EQ(wx::visibility_m(3.912f), 1000.f) << "an extinction of 3.912 km^-1 is a visibility of 1 km";
  EXPECT_FLOAT_EQ(wx::visibility_m(wx::extinction_per_km(150.f)), 150.f);
  EXPECT_NEAR(std::exp(-wx::kKoschmieder), 0.02, 1e-4) << "the threshold of contrast is 2 %";
  wx::Director d(kSeed, Regime::fog);
  d.request(Regime::rain, 20.f);
  run(d, 60.f, [](const wx::Director& w) {
    EXPECT_NEAR(w.value(wx::m_visibility) * w.value(wx::m_extinction), 3912.f, 0.01f);
    const float sum = wx::kClearAir + w.value(wx::m_fog) + wx::rain_extinction(w.value(wx::m_rain)) + wx::snow_extinction(w.value(wx::m_snow)) +
                      w.value(wx::m_dust) * 3.912f / 0.2f;
    EXPECT_NEAR(w.value(wx::m_extinction), sum, 1e-4f * sum) << "everything's extinction adds up";
  });
}

TEST(WeatherRelations, MarshallPalmerDropSizes) {
  EXPECT_FLOAT_EQ(wx::mp_slope(1.f), 4.1f);
  for (const float R : {0.5f, 2.f, 10.f, 30.f, 100.f}) {
    const double L = 4.1 * std::pow(static_cast<double>(R), -0.21);
    EXPECT_NEAR(wx::mp_slope(R), L, 1e-5 * L);
    const auto N = [&](double D) { return 8000.0 * std::exp(-L * D); };
    const double n = simpson(N, 0.5, 6.0), dm = simpson([&](double D) { return D * N(D); }, 0.5, 6.0) / n;
    EXPECT_NEAR(wx::mp_number(R), n, 1e-4 * n) << "drops a cubic metre from 0.5 to 6 mm at " << R << " mm/h";
    EXPECT_NEAR(wx::mp_mean_diameter(R), dm, 1e-4 * dm);
    // geometric optics: twice the cross-section (the extinction paradox), m^-3 mm^2 -> km^-1
    const double sigma = simpson([&](double D) { return 2.0 * 0.25 * 3.14159265358979 * D * D * N(D); }, 0.0, 30.0) * 1e-3;
    EXPECT_NEAR(wx::rain_extinction(R), sigma, 2e-3 * sigma);
    // sampled by the inverse distribution: the mean and the share below the mean as the distribution has them
    std::uint32_t h = 7;
    double sum = 0, below = 0;
    const int K = 100000;
    for (int i = 0; i < K; ++i) {
      h = h * 1664525u + 1013904223u;
      const float D = wx::mp_sample(R, static_cast<float>(h >> 8) * (1.f / 16777216.f));
      ASSERT_GE(D, 0.5f);
      ASSERT_LE(D, 6.f);
      sum += D;
      below += D < dm ? 1.0 : 0.0;
    }
    EXPECT_NEAR(sum / K, dm, 0.005 * dm);
    EXPECT_NEAR(below / K, simpson(N, 0.5, dm) / n, 0.01);
  }
  // the rain's own visibility: heavy rain (40 mm/h) about a kilometre, light rain (1 mm/h) about 10 km
  EXPECT_NEAR(wx::visibility_m(wx::rain_extinction(1.f)), 10700.f, 200.f);
  EXPECT_NEAR(wx::visibility_m(wx::rain_extinction(40.f)), 1000.f, 60.f);
}

TEST(WeatherRelations, AtlasFallSpeedsMatchGunnAndKinzer) {
  // Gunn and Kinzer (1949), terminal velocities measured in still air at sea level, m/s
  const float D[] = {1.f, 2.f, 3.f, 4.f, 5.f}, measured[] = {4.03f, 6.49f, 8.06f, 8.83f, 9.09f};
  for (int i = 0; i < 5; ++i) {
    const float v = 9.65f - 10.3f * std::exp(-0.6f * D[i]);
    EXPECT_FLOAT_EQ(wx::drop_speed(D[i]), v);
    EXPECT_NEAR(wx::drop_speed(D[i]), measured[i], 0.025f * measured[i]) << D[i] << " mm";
  }
  EXPECT_EQ(wx::drop_speed(0.f), 0.f) << "the fit turns negative below 0.03 mm: it stops at 0";
  for (const float R : {1.f, 10.f, 50.f}) {
    const double L = wx::mp_slope(R);
    const double n = simpson([&](double d) { return std::exp(-L * d); }, 0.5, 6.0);
    const double v = simpson([&](double d) { return std::exp(-L * d) * wx::drop_speed(static_cast<float>(d)); }, 0.5, 6.0) / n;
    EXPECT_NEAR(wx::mp_mean_speed(R), v, 1e-3 * v);
  }
}

TEST(WeatherRelations, MarshallPalmerAndAtlasTogetherGiveBackTheRate) {
  // the rain they imply, (pi / 6) integral of D^3 v(D) N(D) dD: within a fifth of the rate asked for, from drizzle to a
  // cloudburst (docs/WEATHER.md §3.3 has the numbers: 18 % over at 1 mm/h, 5 % at 100)
  for (const float R : {1.f, 3.f, 10.f, 30.f, 100.f}) {
    const double L = wx::mp_slope(R);
    const double numeric = 6.0 * 3.14159265358979e-4 *
                           simpson([&](double d) { return d * d * d * wx::drop_speed(static_cast<float>(d)) * 8000.0 * std::exp(-L * d); }, 0.0, 40.0, 8000);
    EXPECT_NEAR(wx::mp_atlas_rate(R), numeric, 2e-3 * numeric) << R;
    EXPECT_GT(wx::mp_atlas_rate(R), R);
    EXPECT_LT(wx::mp_atlas_rate(R), 1.2f * R);
  }
}

TEST(WeatherRelations, TheBeaufortScale) {
  // the WMO's bands (m/s at 10 m) for forces 0 to 12; the formula v = 0.836 B^1.5 puts each force inside its own band
  const float lo[] = {0.f, 0.3f, 1.6f, 3.4f, 5.5f, 8.f, 10.8f, 13.9f, 17.2f, 20.8f, 24.5f, 28.5f, 32.7f};
  const float hi[] = {0.2f, 1.5f, 3.3f, 5.4f, 7.9f, 10.7f, 13.8f, 17.1f, 20.7f, 24.4f, 28.4f, 32.6f, 1e9f};
  for (int b = 0; b <= 12; ++b) {
    const float v = wx::beaufort_speed(static_cast<float>(b));
    EXPECT_GE(v, lo[b]) << "force " << b;
    EXPECT_LE(v, hi[b]) << "force " << b;
    EXPECT_NEAR(wx::beaufort(v), static_cast<float>(b), 1e-4f);
    EXPECT_FALSE(wx::beaufort_on_land(b).empty());
  }
  EXPECT_EQ(wx::beaufort_on_land(0), "smoke rises vertically");
  EXPECT_EQ(wx::beaufort_on_land(4), "raises dust and loose paper; small branches are moved");
  // what the layers show at each force
  EXPECT_EQ(wx::lean(0.f, 1.f), 0.5f) << "force 0: smoke rises vertically (a learned module's wind control at still air)";
  EXPECT_GT(wx::lean(wx::beaufort_speed(1.f), 1.f), 0.5f) << "force 1: shown by smoke drift";
  EXPECT_LT(wx::lean(wx::beaufort_speed(1.f), -1.f), 0.5f) << "and the other way when it blows the other way";
  EXPECT_EQ(wx::lean(wx::beaufort_speed(8.f), 1.f), 1.f) << "a gale leans it all the way";
  EXPECT_EQ(wx::leaves(wx::beaufort_speed(2.f)), 0.f);
  EXPECT_GT(wx::leaves(wx::beaufort_speed(3.f)), 0.3f) << "force 3: leaves and small twigs in constant motion";
  EXPECT_EQ(wx::leaves(wx::beaufort_speed(4.f)), 1.f);
  EXPECT_EQ(wx::lift(wx::beaufort_speed(3.5f)), 0.f);
  EXPECT_GT(wx::lift(wx::beaufort_speed(4.f)), 0.1f) << "force 4: raises dust and loose paper";
  EXPECT_EQ(wx::lift(wx::beaufort_speed(5.f)), 1.f);
  // trees: still through a moderate breeze, swaying from force 5, all of them in motion by force 7, and no further
  EXPECT_EQ(wx::trees(0.f), 0.f);
  EXPECT_EQ(wx::trees(wx::beaufort_speed(4.f)), 0.f) << "force 4: small branches are moved, the trees stand";
  EXPECT_GT(wx::trees(wx::beaufort_speed(5.f)), 0.2f) << "force 5: small trees in leaf begin to sway";
  EXPECT_LT(wx::trees(wx::beaufort_speed(5.f)), 0.5f);
  EXPECT_GT(wx::trees(wx::beaufort_speed(6.f)), 0.5f) << "force 6: large branches in motion";
  EXPECT_LT(wx::trees(wx::beaufort_speed(6.f)), 1.f);
  EXPECT_EQ(wx::trees(wx::beaufort_speed(7.f)), 1.f) << "force 7: whole trees in motion";
  EXPECT_EQ(wx::trees(wx::beaufort_speed(10.f)), 1.f) << "a storm sways them no further (nothing uproots them)";
  for (float b = 0.f; b < 9.f; b += 0.25f)
    EXPECT_LE(wx::trees(wx::beaufort_speed(b)), wx::trees(wx::beaufort_speed(b + 0.25f))) << "more wind, more sway";
  // the director carries it with the wind now (gusts included), a script reads it as `sky.trees`
  EXPECT_EQ(wx::member_of("trees"), wx::m_trees);
  EXPECT_EQ(wx::member_names()[static_cast<std::size_t>(wx::m_trees)], "trees");
  wx::Director clear(kSeed, Regime::clear), storm(kSeed, Regime::storm);
  float clear_most = 0.f, storm_most = 0.f;
  run(clear, 30.f, [&](const wx::Director& s) {
    EXPECT_EQ(s.value(wx::m_trees), wx::trees(s.value(wx::m_wind)));
    clear_most = std::max(clear_most, s.value(wx::m_trees));
  });
  run(storm, 30.f, [&](const wx::Director& s) {
    EXPECT_EQ(s.value(wx::m_trees), wx::trees(s.value(wx::m_wind)));
    storm_most = std::max(storm_most, s.value(wx::m_trees));
  });
  EXPECT_EQ(clear_most, 0.f) << "a light breeze (force 2) moves no tree";
  EXPECT_GT(storm_most, 0.5f) << "a storm's force 6 gusting to 8 has the trees in motion";
}

TEST(WeatherRelations, KastenAndCzeplaksDaylight) {
  EXPECT_FLOAT_EQ(wx::daylight(0.f), 1.f);
  EXPECT_FLOAT_EQ(wx::daylight(1.f), 0.25f) << "eight oktas: a quarter of the clear sky's global irradiance";
  EXPECT_NEAR(wx::daylight(0.5f), 1.f - 0.75f * std::pow(0.5f, 3.4f), 1e-6f);
  for (float c = 0.f; c < 1.f; c += 0.05f) EXPECT_GE(wx::daylight(c), wx::daylight(c + 0.05f)) << "more cloud, less light";
}

TEST(WeatherRelations, LocatelliAndHobbsSnowflakes) {
  for (const float D : {2.f, 5.f, 10.f}) {
    EXPECT_FLOAT_EQ(wx::flake_speed(D), 0.8f * std::pow(D, 0.16f));
    EXPECT_FLOAT_EQ(wx::flake_mass_mg(D), 0.073f * std::pow(D, 1.4f));
    EXPECT_LT(wx::flake_speed(D), 1.2f) << "unrimed aggregates fall at about 1 m/s whatever their size";
  }
  // the flakes' sizes: exponential on 2 to 10 mm; their number at a rate carries that rate's mass flux
  const auto pdf = [](double d) { return std::exp(-0.4 * d); };
  const double z = simpson(pdf, 2.0, 10.0);
  const double mv = simpson([&](double d) { return pdf(d) * 0.073e-6 * std::pow(d, 1.4) * 0.8 * std::pow(d, 0.16); }, 2.0, 10.0) / z;
  for (const float S : {0.5f, 1.f, 3.f}) EXPECT_NEAR(wx::snow_number(S) * mv * 3600.0, S, 1e-3 * S) << "kg m^-2 per hour of melted water";
  EXPECT_NEAR(wx::flake_mean_speed(), simpson([&](double d) { return pdf(d) * 0.8 * std::pow(d, 0.16); }, 2.0, 10.0) / z, 1e-4);
  // the US weather service's classes by visibility: moderate snow 0.5 to 1 km (FMH-1); 1 mm/h of melted water is moderate
  const float v1 = wx::visibility_m(wx::snow_extinction(1.f) + wx::kClearAir);
  EXPECT_GT(v1, 500.f);
  EXPECT_LT(v1, 1000.f);
}

TEST(WeatherRelations, HailFallsByItsDrag) {
  EXPECT_NEAR(wx::hail_speed(10.f), 12.8f, 0.1f);
  EXPECT_NEAR(wx::hail_speed(40.f) / wx::hail_speed(10.f), 2.f, 1e-4f) << "v grows as the square root of the size";
}

TEST(WeatherRelations, DewPointAndCloudBase) {
  // the saturation vapour pressure over water against the measured values (Sonntag 1990; hPa) from -20 to 40 C
  const float T[] = {-20.f, -10.f, 0.f, 10.f, 20.f, 30.f, 40.f}, es[] = {1.2553f, 2.8639f, 6.1121f, 12.282f, 23.388f, 42.455f, 73.813f};
  for (int i = 0; i < 7; ++i) EXPECT_NEAR(wx::saturation_vapour_hpa(T[i]), es[i], 0.004f * es[i]) << T[i] << " C";
  // the dew point: the temperature at which the air's vapour would saturate; saturated air's is the air's own
  EXPECT_NEAR(wx::dew_point(20.f, 0.5f), 9.3f, 0.1f) << "20 C at 50 %: the textbook 9.3 C";
  EXPECT_FLOAT_EQ(wx::dew_point(8.f, 1.f), 8.f);
  for (const float t : {-5.f, 10.f, 25.f})
    for (const float h : {0.2f, 0.5f, 0.8f, 0.95f}) {
      const float td = wx::dew_point(t, h);
      EXPECT_LT(td, t);
      EXPECT_NEAR(wx::saturation_vapour_hpa(td) / wx::saturation_vapour_hpa(t), h, 1e-3f) << "e_s(Td) / e_s(T) is the humidity";
    }
  // Espy's rule, 125 m a degree of the dew point spread; against Lawrence's (2005) h = (20 + T / 5)(100 - RH) m for
  // RH above 50 %, the two agree to about 15 %
  EXPECT_FLOAT_EQ(wx::cloud_base_m(20.f, 9.3f), 125.f * 10.7f);
  EXPECT_FLOAT_EQ(wx::cloud_base_m(8.f, 8.f), 0.f) << "saturated: cloud at the ground";
  for (const float h : {0.5f, 0.7f, 0.9f}) {
    const float espy = wx::cloud_base_m(20.f, wx::dew_point(20.f, h)), lawrence = (20.f + 20.f / 5.f) * (100.f - 100.f * h);
    EXPECT_NEAR(espy, lawrence, 0.15f * lawrence) << "RH " << h;
  }
}

TEST(WeatherRelations, PenmansDryingPowerOfTheAir) {
  // Penman's (1948) aerodynamic term, E_a = 0.35 (1 + u / 100)(e_s - e) mm/day with u in miles a day and e in mmHg: in
  // m/s (53.69 miles a day) and hPa (1.3332 a mmHg), 0.2625 (1 + 0.537 U_2) D mm/day; the director gives it in mm/h
  const float es20 = wx::saturation_vapour_hpa(20.f);
  EXPECT_NEAR(wx::evaporation_mm_h(20.f, 0.5f, 2.f) * 24.f, 0.35f / 1.3332f * (1.f + 2.f * 53.69f / 100.f) * 0.5f * es20, 0.01f)
      << "20 C, 50 %, 2 m/s: about 6.3 mm a day";
  EXPECT_NEAR(wx::evaporation_mm_h(20.f, 0.5f, 2.f), 0.263f, 0.003f) << "mm/h: a 1 mm puddle lasts about 4 hours";
  EXPECT_FLOAT_EQ(wx::evaporation_mm_h(8.f, 1.f, 1.f), 0.f) << "saturated air dries nothing";
  EXPECT_NEAR(wx::evaporation_mm_h(20.f, 0.5f, 4.f) / wx::evaporation_mm_h(20.f, 0.5f, 0.f), 1.f + 0.537f * 4.f, 1e-3f) << "linear in the wind";
  EXPECT_NEAR(wx::evaporation_mm_h(20.f, 0.25f, 2.f) / wx::evaporation_mm_h(20.f, 0.5f, 2.f), 1.5f, 1e-3f) << "linear in the deficit";
  EXPECT_GT(wx::evaporation_mm_h(34.f, 0.15f, 2.f), 4.f * wx::evaporation_mm_h(13.f, 0.9f, 2.f)) << "hot dry air dries far faster than a rainy day's";
  // the depression storage of an impervious surface, 1.3 to 2.5 mm (ASCE 1992); the 2 m wind by the log profile
  EXPECT_GE(wx::kDepressionStorage, 1.3f);
  EXPECT_LE(wx::kDepressionStorage, 2.5f);
  EXPECT_LT(wx::kWetFilm, wx::kDepressionStorage) << "the ground darkens long before it puddles";
  EXPECT_NEAR(wx::kWind2m, std::log(2.f / 0.03f) / std::log(10.f / 0.03f), 1e-3f);
  EXPECT_FLOAT_EQ(wx::wind_2m(10.f), 10.f * wx::kWind2m);
}

TEST(WeatherRelations, TheSunAndItsShafts) {
  // where the sun stands, from the hour alone, at 45 N on the equinox: sin h = cos(latitude) cos(H), H = 15 (hour - 12)
  EXPECT_NEAR(wx::sun_elevation_deg(12.f), 90.f - wx::kLatitude, 1e-3f) << "noon: the sun as high as the colatitude";
  EXPECT_NEAR(wx::sun_elevation_deg(6.f), 0.f, 1e-3f) << "it rises at 6";
  EXPECT_NEAR(wx::sun_elevation_deg(18.f), 0.f, 1e-3f) << "and sets at 18";
  EXPECT_NEAR(wx::sun_elevation_deg(0.f), wx::kLatitude - 90.f, 1e-3f) << "midnight: as far below";
  EXPECT_NEAR(wx::sun_elevation_deg(9.f), wx::sun_elevation_deg(15.f), 1e-4f) << "symmetric about noon";
  EXPECT_NEAR(wx::sun_elevation_deg(9.f), std::asin(std::cos(45.f * 3.14159265f / 180.f) * std::cos(45.f * 3.14159265f / 180.f)) * 180.f / 3.14159265f, 1e-3f);
  EXPECT_NEAR(wx::sun_elevation_deg(36.f), wx::sun_elevation_deg(12.f), 1e-4f) << "the hour wraps at 24";
  EXPECT_NEAR(wx::sun_across(6.f), -1.f, 1e-4f) << "rising on the left (a camera facing south)";
  EXPECT_NEAR(wx::sun_across(12.f), 0.f, 1e-4f);
  EXPECT_NEAR(wx::sun_across(18.f), 1.f, 1e-4f) << "setting on the right";
  EXPECT_LT(wx::sun_across(9.f), wx::sun_across(15.f));
  // Kasten and Czeplak (1980): the diffuse fraction D/G = 0.3 + 0.7 (N/8)^2, so the direct beam is 0.7 (1 - c^2) of
  // the global: 70 % of a clear sky's, none of an overcast's
  EXPECT_FLOAT_EQ(wx::sun_share(0.f), 0.7f);
  EXPECT_FLOAT_EQ(wx::sun_share(1.f), 0.f);
  EXPECT_FLOAT_EQ(wx::sun_share(0.5f), 1.f - (0.3f + 0.7f * 0.25f));
  for (float c = 0.f; c < 1.f; c += 0.1f) EXPECT_GT(wx::sun_share(c), wx::sun_share(c + 0.1f)) << "more cloud, less sun";
  // Beer-Lambert along the slant path through the layer the weather fills: exp(-sigma L / sin h)
  EXPECT_NEAR(wx::beam_transmittance(2.f, 90.f), std::exp(-2.f * wx::kWeatherLayer), 1e-6f);
  EXPECT_NEAR(wx::beam_transmittance(2.f, 30.f), std::pow(wx::beam_transmittance(2.f, 90.f), 2.f), 1e-5f) << "at 30 degrees the path is twice as long";
  EXPECT_EQ(wx::beam_transmittance(0.1f, 0.f), 0.f) << "none at the horizon";
  EXPECT_EQ(wx::beam_transmittance(0.1f, -20.f), 0.f) << "nor at night";
  EXPECT_GT(wx::beam_transmittance(wx::kClearAir, 45.f), 0.95f) << "clear air hardly dims it";
  EXPECT_LT(wx::beam_transmittance(wx::target(Regime::fog).fog, 45.f), 1e-3f) << "fog hides the sun";
  EXPECT_LT(wx::beam_transmittance(0.6f * 3.912f / 0.2f, 45.f), 0.01f) << "so does a dust storm's load";
  // the shafts: lit air beside shadowed air, so none under a clear sky or an overcast one, most in a half-broken deck;
  // the beam through the weather; and air that scatters enough of it to be seen, so clear air shows hardly any, haze
  // shows them, and air thick enough to hide the sun hides them too
  EXPECT_EQ(wx::shaft_visibility(0.f, 2.f, 45.f), 0.f) << "nothing shadows";
  EXPECT_EQ(wx::shaft_visibility(1.f, 2.f, 45.f), 0.f) << "nothing is lit";
  EXPECT_GT(wx::shaft_visibility(0.5f, 2.f, 45.f), wx::shaft_visibility(0.2f, 2.f, 45.f));
  EXPECT_GT(wx::shaft_visibility(0.5f, 2.f, 45.f), wx::shaft_visibility(0.8f, 2.f, 45.f));
  for (float c = 0.f; c <= 1.f; c += 0.1f) EXPECT_LT(wx::shaft_visibility(c, wx::kClearAir, 45.f), 0.05f) << "clear air at cover " << c;
  EXPECT_GT(wx::shaft_visibility(0.5f, 2.f, 45.f), 0.25f) << "haze (a 2 km day) under a half-broken deck: about 0.27";
  EXPECT_LT(wx::shaft_visibility(0.5f, 30.f, 45.f), wx::shaft_visibility(0.5f, 2.f, 45.f)) << "fog hides the beam";
  EXPECT_NEAR(wx::shaft_visibility(0.5f, 2.f, 90.f), 1.f * wx::beam_transmittance(2.f, 90.f) * (1.f - std::exp(-2.f * wx::kShaftDepth)), 1e-6f);
  EXPECT_EQ(wx::shaft_visibility(0.5f, 2.f, -5.f), 0.f) << "none at night";
}

TEST(WeatherRelations, TheCloudDeck) {
  EXPECT_FLOAT_EQ(wx::deck_cover(0.f), 0.f);
  EXPECT_FLOAT_EQ(wx::deck_cover(0.5f), 0.f) << "four oktas: scattered, no deck";
  EXPECT_GT(wx::deck_cover(0.625f), 0.f) << "five oktas: broken cover begins a deck";
  EXPECT_FLOAT_EQ(wx::deck_cover(1.f), 1.f);
  for (float c = 0.f; c < 1.f; c += 0.05f) EXPECT_LE(wx::deck_cover(c), wx::deck_cover(c + 0.05f) + 1e-6f);
  wx::Director d(28'500'200, wx::Regime::overcast);
  for (int i = 0; i < 600; ++i) d.step(1.f / 30.f);
  EXPECT_GT(d.value(wx::m_deck), 0.9f);
  EXPECT_GT(d.value(wx::m_base), 400.f);
}

TEST(WeatherRelations, TheDeckFootHidesInDust) {
  EXPECT_FLOAT_EQ(wx::deck_foot_deg(0.f, 40'000.f), 0.f) << "fog: the base is at the ground";
  EXPECT_NEAR(wx::deck_foot_deg(1200.f, 40'000.f), std::atan(1200.f / 40'000.f) * 180.f / 3.14159265f, 0.02f);
  EXPECT_GT(wx::deck_foot_deg(3800.f, 330.f), 80.f) << "a dust storm's haze hides the high base even overhead";
  EXPECT_FLOAT_EQ(wx::deck_foot_deg(1200.f, 0.f), 90.f) << "no visibility at all: nothing of it";
  wx::Director d(28'500'200, wx::Regime::dust);
  for (int i = 0; i < 900; ++i) d.step(1.f / 30.f);
  EXPECT_GT(d.value(wx::m_foot), 75.f);
  EXPECT_LE(d.value(wx::m_foot), 90.f);
}

TEST(WeatherRelations, TheDeckUndersideFadesIntoTheHazeByKoschmieder) {
  // The deck's underside is a plane at the base: seen at elevation theta it is base / sin(theta) away along the slant
  // path, and Koschmieder's contrast against the horizon sky there is exp(-sigma x), sigma = 3.912 / V. So the contrast
  // is 2 % where the slant path is the visibility, more above (a shorter path), less toward the horizon, and none at it.
  const float base = 560.f, V = 4000.f, sigma = wx::kKoschmieder / V;
  for (const float e : {5.f, 10.f, 30.f, 60.f, 90.f}) {
    const float slant = base / std::sin(e * 3.14159265f / 180.f);
    EXPECT_NEAR(wx::deck_contrast(base, V, e), std::exp(-sigma * slant), 1e-6f) << e << " degrees";
  }
  EXPECT_NEAR(wx::deck_contrast(base, V, 90.f), std::exp(-wx::kKoschmieder * base / V), 1e-6f) << "overhead: the base's own path";
  EXPECT_GT(wx::deck_contrast(base, V, 60.f), wx::deck_contrast(base, V, 20.f)) << "clearer overhead than toward the horizon";
  EXPECT_EQ(wx::deck_contrast(base, V, 0.f), 0.f) << "at the horizon the path is endless";
  EXPECT_EQ(wx::deck_contrast(base, V, -10.f), 0.f);
  EXPECT_EQ(wx::deck_contrast(0.f, V, 30.f), 1.f) << "a base at the ground (fog): the deck is here";
  EXPECT_EQ(wx::deck_contrast(base, 0.f, 30.f), 0.f) << "no visibility: nothing of it";
  // the foot is where the underside meets the haze: its contrast there is at most the 2 % threshold (the foot stands at
  // the visibility's horizontal distance, whose slant path is a little longer than V), and it reaches the threshold as
  // the deck is high and far (a fair-weather deck's foot at the horizon); the deck shows above its foot
  for (const auto& [b, v] : {std::pair{1200.f, 40000.f}, std::pair{560.f, 40000.f}, std::pair{200.f, 3200.f}, std::pair{315.f, 1000.f}, std::pair{175.f, 550.f}}) {
    const float foot = wx::deck_foot_deg(b, v);
    EXPECT_NEAR(foot, std::atan(b / v) * 180.f / 3.14159265f, 1e-4f);
    EXPECT_LE(wx::deck_contrast(b, v, foot), 0.02f + 1e-6f) << b << " m seen " << v << " m: below the foot the haze has it";
    EXPECT_GT(wx::deck_contrast(b, v, 2.f * foot), wx::deck_contrast(b, v, foot)) << "and above it the deck shows";
    EXPECT_NEAR(wx::deck_contrast(b, v, foot), 0.02f * std::exp(-wx::kKoschmieder * (std::sqrt(1.f + (b / v) * (b / v)) - 1.f)), 1e-4f) << "the slant path at the foot is sqrt(1 + (base / V)^2) of V";
  }
  EXPECT_NEAR(wx::deck_contrast(1200.f, 40000.f, wx::deck_foot_deg(1200.f, 40000.f)), 0.02f, 2e-4f) << "a high deck far away: the 2 % threshold at its foot";
  // the regimes' feet (§3.15): fair-weather cumulus at the horizon, an overcast deck lower still (nearer the ground, as
  // far away), a low deck under rain a few degrees up, a storm's and snow's lost below about 20 degrees in their haze
  EXPECT_NEAR(wx::deck_foot_deg(1200.f, 40000.f), 1.7f, 0.1f);
  EXPECT_LT(wx::deck_foot_deg(560.f, 40000.f), wx::deck_foot_deg(1200.f, 40000.f));
  EXPECT_NEAR(wx::deck_foot_deg(200.f, 3200.f), 3.6f, 0.1f);
  EXPECT_GT(wx::deck_foot_deg(315.f, 1000.f), 15.f);
  EXPECT_LT(wx::deck_foot_deg(315.f, 1000.f), 25.f);
  // it is smooth where the base passes the visibility (a dust storm's), where the 2 % threshold's asin would pin at 90
  float last = 0.f;
  for (float r = 0.f; r < 4.f; r += 0.01f) {
    const float foot = wx::deck_foot_deg(r * 330.f, 330.f);
    EXPECT_GE(foot, last);
    EXPECT_LT(foot - last, 0.6f) << "a hundredth of the ratio moves it under 0.6 degrees everywhere";
    last = foot;
  }
  EXPECT_LT(last, 90.f);
  EXPECT_GT(last, 75.f);
  // the deck's ramp: between the WMO's scattered (4 oktas) and broken (6) amounts, as the constants say
  EXPECT_FLOAT_EQ(wx::kDeckScattered, 4.f / 8.f);
  EXPECT_FLOAT_EQ(wx::kDeckBroken, 6.f / 8.f);
  EXPECT_FLOAT_EQ(wx::deck_cover(0.625f), 0.5f) << "five oktas, the ICAO's 'more than half the sky': half a layer";
  EXPECT_FLOAT_EQ(wx::deck_cover(wx::kDeckBroken), 1.f);
}

// --- the noise ----------------------------------------------------------------------------------------------------------

TEST(WeatherNoise, GustsAreAnOrnsteinUhlenbeckProcessAtTheKaimalTimeScale) {
  wx::Director d(kSeed, Regime::storm);
  std::vector<double> w;
  const float T = 3000.f;
  run(d, T, [&](const wx::Director& s) { w.push_back(s.value(wx::m_wind)); });
  const double U = d.value(wx::m_mean), I = d.value(wx::m_gust);
  EXPECT_NEAR(mean(w), U, 0.03 * U) << "the gusts average out to the mean wind";
  EXPECT_NEAR(stddev(w) / U, I, 0.12 * I) << "their spread is the gustiness";
  // the OU (time scale tau = L_u / U) smoothed over a second: its autocorrelation at tau is
  // (tau e^-1 - 1 e^(-tau / 1)) / (tau - 1)
  const double tau = wx::gust_time(static_cast<float>(U));
  EXPECT_NEAR(tau, 56.7 / U, 1e-3 * tau);
  const double expected = (tau * std::exp(-1.0) - std::exp(-tau)) / (tau - 1.0);
  const double acf = autocorrelation(w, static_cast<std::size_t>(std::lround(tau * kFps)));
  EXPECT_NEAR(acf, expected, 0.1) << "at the integral time scale";
  EXPECT_LT(std::abs(autocorrelation(w, static_cast<std::size_t>(std::lround(8.0 * tau * kFps)))), 0.12) << "forgotten after several";
  EXPECT_GT(autocorrelation(w, 3), 0.99) << "smooth from frame to frame";
}

TEST(WeatherNoise, RainCellsAreLogNormalOverMinutes) {
  wx::Director d(kSeed + 2, Regime::storm);
  std::vector<double> lr;
  run(d, 30000.f, [&](const wx::Director& s) { lr.push_back(std::log(s.value(wx::m_rain) / s.base().precip)); });
  const double spread = d.base().cell_spread, tau = d.base().cell_time;
  EXPECT_NEAR(mean(lr), -0.5 * spread * spread, 0.25 * spread) << "the cells keep the mean rate";
  EXPECT_NEAR(stddev(lr), spread, 0.15 * spread);
  EXPECT_NEAR(autocorrelation(lr, static_cast<std::size_t>(tau * kFps)), std::exp(-1.0), 0.12) << "e-folding over the cells' time";
}

TEST(WeatherNoise, LightningComesWithTheHeaviestRain) {
  wx::Director d(kSeed + 4, Regime::storm);
  double expected = 0, rain_all = 0, rain_flash = 0;
  int frames = 0, flashes = 0, last = 0;
  run(d, 3000.f, [&](const wx::Director& s) {
    expected += s.value(wx::m_lightning) / 60.0 * kDt;
    rain_all += s.value(wx::m_rain);
    ++frames;
    const int strikes = static_cast<int>(s.value(wx::m_strike));
    if (strikes != last) {
      ++flashes;
      rain_flash += s.value(wx::m_rain);
      EXPECT_EQ(s.value(wx::m_since), 0.f) << "a flash begins at its frame";
      EXPECT_EQ(s.value(wx::m_flash), 1.f) << "its first stroke at full light";
      EXPECT_GE(s.value(wx::m_bolt), -1.f);
      EXPECT_LE(s.value(wx::m_bolt), 1.f);
      last = strikes;
    }
  });
  EXPECT_NEAR(flashes, expected, 4.0 * std::sqrt(expected)) << "a Poisson process at the rate of the moment";
  EXPECT_GT(flashes, 300);
  EXPECT_GT(rain_flash / flashes, 1.2 * rain_all / frames) << "the flashes fall in the heavier rain";
}

TEST(WeatherNoise, HailOnlyInTheHeaviestCellsAndSnowOnlyInTheCold) {
  wx::Director d(kSeed + 6, Regime::storm);
  int hail = 0;
  run(d, 2000.f, [&](const wx::Director& s) {
    if (s.value(wx::m_hail) > 0.f) {
      ++hail;
      EXPECT_GT(s.value(wx::m_rain), 0.9f * s.base().precip) << "hail falls in cells heavier than the storm's mean";
    }
  });
  EXPECT_GT(hail, 1000);
  wx::Director snow(kSeed + 8, Regime::snow);
  run(snow, 600.f, [&](const wx::Director& s) {
    EXPECT_EQ(s.value(wx::m_rain), s.value(wx::m_temp) < 0.f ? 0.f : s.value(wx::m_rain)) << "below 0 C it all falls as snow";
    EXPECT_GT(s.value(wx::m_snow), 0.f);
  });
  wx::Director rain(kSeed + 10, Regime::rain);
  run(rain, 600.f, [&](const wx::Director& s) { EXPECT_EQ(s.value(wx::m_snow), 0.f) << "above 2 C it all falls as rain"; });
  EXPECT_FLOAT_EQ(wx::snow_share(2.f), 0.f);
  EXPECT_FLOAT_EQ(wx::snow_share(0.f), 1.f);
  EXPECT_FLOAT_EQ(wx::snow_share(1.f), 0.5f);
}

// --- the director -------------------------------------------------------------------------------------------------------

std::vector<std::array<float, wx::kMembers>> history(std::uint64_t seed, float dt = kDt) {
  wx::Director d(seed, Regime::clear);
  std::vector<std::array<float, wx::kMembers>> out;
  const auto rec = [&](const wx::Director& s) { out.push_back(s.values()); };
  const Regime path[] = {Regime::overcast, Regime::rain, Regime::storm, Regime::snow, Regime::fog, Regime::dust, Regime::clear};
  for (const Regime r : path) {
    d.request(r, 20.f);
    run(d, 30.f, rec, dt);
  }
  d.request(Regime::storm, 30.f);  // interrupted halfway, twice
  run(d, 15.f, rec, dt);
  d.request(Regime::fog, 30.f);
  run(d, 15.f, rec, dt);
  d.request(Regime::clear, 10.f);
  run(d, 20.f, rec, dt);
  return out;
}

TEST(WeatherDirector, ItIsDeterministic) {
  const auto a = history(kSeed), b = history(kSeed), c = history(kSeed + 2);
  ASSERT_EQ(a.size(), b.size());
  bool same = true, differ = false;
  for (std::size_t i = 0; i < a.size(); ++i)
    for (int k = 0; k < wx::kMembers; ++k) {
      const auto ku = static_cast<std::size_t>(k);
      if (std::memcmp(&a[i][ku], &b[i][ku], sizeof(float)) != 0) same = false;
      if (a[i][ku] != c[i][ku]) differ = true;
    }
  EXPECT_TRUE(same) << "the same seed and requests give the same state, to the bit";
  EXPECT_TRUE(differ) << "another seed, another weather";
}

TEST(WeatherDirector, NothingJumps) {
  // The state's members (not the lightning's light, which flashes, nor its count, place and time): the most each moves
  // in a frame, through every transition and two interrupted ones, is small, and halves when the frames do (a jump would
  // not: it is the same size however fine the steps).
  const int members[] = {wx::m_cloud, wx::m_rain,      wx::m_snow, wx::m_hail, wx::m_wind,       wx::m_mean,  wx::m_gust,     wx::m_temp,     wx::m_lightning,
                         wx::m_dust,  wx::m_fog,       wx::m_extinction, wx::m_light, wx::m_lean, wx::m_humidity, wx::m_dewpoint, wx::m_base,
                         wx::m_water, wx::m_wet,       wx::m_puddle, wx::m_evaporation, wx::m_carried, wx::m_sun, wx::m_shaft, wx::m_deck, wx::m_foot};
  // (the extinction's bound: rain turning to snow as the cold front passes hides much more, per mm/h, in a second; the
  // water on the ground moves by the rain's mm/h over 3600 a second, the evaporation with the gusts; the air carried
  // past the camera by the mean wind's metres in a frame. The front is not here: it is at the camera again at each
  // request, as the progress is 0 again)
  // The sun and its shafts follow the cloud's noise and the extinction (the dust's, as the gusts raise it). The deck
  // follows the cloud (its ramp over two oktas is four times as steep as the cover); its foot, atan(base / V), follows
  // the visibility, and rain turning to snow is again the fastest (measured 9.5 degrees a frame; the asin of the 2 %
  // threshold would move 29, pinned at 90 with an infinite slope once the base passes the visibility).
  const float bound[] = {0.02f, 2.5f, 0.8f, 0.05f, 0.6f, 0.4f, 0.01f, 0.4f, 3.f, 0.05f, 3.f, 4.f, 0.03f, 0.02f, 0.01f, 0.5f, 60.f, 0.005f, 0.08f, 0.0035f, 0.3f, 1.f, 0.05f, 0.05f, 0.08f, 12.f};
  std::array<double, std::size(members)> worst30{}, worst60{};
  for (const std::uint64_t seed : {kSeed, kSeed + 2, kSeed + 4}) {
    for (const float dt : {kDt, kDt / 2.f}) {
      const auto h = history(seed, dt);
      auto& worst = dt == kDt ? worst30 : worst60;
      for (std::size_t i = 1; i < h.size(); ++i)
        for (std::size_t j = 0; j < std::size(members); ++j) {
          const auto k = static_cast<std::size_t>(members[j]);
          worst[j] = std::max(worst[j], static_cast<double>(std::abs(h[i][k] - h[i - 1][k])));
        }
    }
  }
  for (std::size_t j = 0; j < std::size(members); ++j) {
    const std::string name(wx::member_names()[static_cast<std::size_t>(members[j])]);
    EXPECT_LT(worst30[j], bound[j]) << name << ": the most it moves in a frame at 30 fps";
    EXPECT_LT(worst60[j], 0.75 * worst30[j]) << name << ": at 60 fps it moves about half as far a step (" << worst60[j] << " against " << worst30[j] << ")";
  }
}

TEST(WeatherDirector, EachRegimeReachesItsTargets) {
  for (int r = 0; r < wx::kRegimes; ++r) {
    const auto regime = static_cast<Regime>(r);
    std::vector<double> cloud, rain, snow, wind, temp, lightning, fog, dust, vis;
    for (const std::uint64_t seed : {kSeed, kSeed + 2, kSeed + 4, kSeed + 6}) {
      wx::Director d(seed, regime == Regime::clear ? Regime::storm : Regime::clear);
      d.request(regime, 10.f);
      run(d, 10.f);
      EXPECT_EQ(d.progress(), 1.f);
      const wx::Target& want = d.heading();
      const wx::Target& got = d.base();
      EXPECT_EQ(got.cloud, want.cloud);
      EXPECT_EQ(got.precip, want.precip);
      EXPECT_EQ(got.mean, want.mean);
      EXPECT_EQ(got.temp, want.temp);
      EXPECT_EQ(got.lightning, want.lightning);
      EXPECT_EQ(got.fog, want.fog);
      EXPECT_EQ(got.dust, want.dust);
      // the request's target: the regime's, varied by the seed within the variability it says
      const wx::Target& base = wx::target(regime);
      EXPECT_NEAR(want.cloud, base.cloud, 0.17f);
      EXPECT_LE(want.precip, base.precip * std::exp(1.1f));
      EXPECT_GE(want.precip, base.precip * std::exp(-1.1f));
      EXPECT_NEAR(want.temp, base.temp, 6.f);
      run(d, 3600.f, [&](const wx::Director& s) {
        cloud.push_back(s.value(wx::m_cloud) - want.cloud);
        rain.push_back(want.precip > 0.f ? (s.value(wx::m_rain) + s.value(wx::m_snow)) / want.precip : 0.0);
        wind.push_back(s.value(wx::m_wind) / want.mean);
        temp.push_back(s.value(wx::m_temp) - want.temp);
        lightning.push_back(want.lightning > 0.f ? s.value(wx::m_lightning) / want.lightning : 0.0);
        fog.push_back(want.fog > 0.f ? s.value(wx::m_fog) / want.fog : 0.0);
        dust.push_back(want.dust > 0.f ? s.value(wx::m_dust) / want.dust : 0.0);
        vis.push_back(s.value(wx::m_visibility));
        // moist air: the humidity is the target's; the dew point and the cloud base follow it and the temperature
        EXPECT_EQ(s.value(wx::m_humidity), want.humidity);
        EXPECT_LE(s.value(wx::m_dewpoint), s.value(wx::m_temp));
        EXPECT_EQ(s.value(wx::m_base), wx::cloud_base_m(s.value(wx::m_temp), s.value(wx::m_dewpoint)));
        if (regime == Regime::fog) {
          EXPECT_EQ(s.value(wx::m_humidity), 1.f) << "fog is saturated air";
          EXPECT_EQ(s.value(wx::m_dewpoint), s.value(wx::m_temp));
          EXPECT_EQ(s.value(wx::m_base), 0.f) << "cloud at the ground";
        }
        if (regime == Regime::clear) {
          EXPECT_GT(s.value(wx::m_base), 700.f) << "fair-weather cumulus stand high";
          EXPECT_LT(s.value(wx::m_base), 2000.f);
        }
        if (regime == Regime::rain || regime == Regime::snow) {
          EXPECT_GT(s.value(wx::m_base), 50.f);
          EXPECT_LT(s.value(wx::m_base), 500.f) << "a low deck under steady precipitation";
        }
        if (regime == Regime::dust) {
          EXPECT_GT(s.value(wx::m_base), 2500.f) << "dry air: a high base";
        }
        // the sun: at noon (no hour set), the direct beam's share of the daylight through the weather; little under
        // the cover of rain, storm, snow and fog (1, less the seed's few gaps: Kasten and Czeplak give 7 oktas 8 %)
        EXPECT_EQ(s.value(wx::m_hour), 12.f);
        EXPECT_EQ(s.value(wx::m_sun), wx::sun_share(s.value(wx::m_cloud)) * wx::daylight(s.value(wx::m_cloud)) *
                                          wx::beam_transmittance(s.value(wx::m_extinction), s.value(wx::m_elevation)));
        if (wx::target(regime).cloud >= 1.f) {
          EXPECT_LT(s.value(wx::m_sun), 0.08f) << "under the deck";
          EXPECT_LT(s.value(wx::m_shaft), 0.15f);
        }
        if (regime == Regime::clear) {
          EXPECT_GT(s.value(wx::m_sun), 0.4f) << "a clear day's sun";
        }
      });
    }
    const std::string name(wx::regime_name(regime));
    EXPECT_NEAR(mean(cloud), 0.0, 0.035) << name;
    EXPECT_NEAR(mean(wind), 1.0, 0.05) << name;
    EXPECT_NEAR(mean(temp), 0.0, 0.2) << name;
    if (wx::target(regime).precip > 0.f) {
      EXPECT_NEAR(mean(rain), 1.0, 0.2) << name << ": the rate, on average over its cells";
    }
    if (wx::target(regime).lightning > 0.f) {
      EXPECT_NEAR(mean(lightning), 1.0, 0.3) << name;
    }
    if (wx::target(regime).fog > 0.f) {
      EXPECT_NEAR(mean(fog), 1.0, 0.05) << name;
    }
    if (wx::target(regime).dust > 0.f) {
      EXPECT_NEAR(mean(dust), 1.0, 0.1) << name;
    }
    std::vector<double> v = vis;
    std::ranges::sort(v);
    const double median = v[v.size() / 2];
    if (regime == Regime::clear || regime == Regime::overcast) {
      EXPECT_NEAR(median, 40000.0, 1.0) << name << ": clear air";
    }
    if (regime == Regime::fog) {
      EXPECT_NEAR(median, 150.0, 30.0) << name << ": fog, under 1 km (dense: 150 m)";
    }
    if (regime == Regime::dust) {
      EXPECT_LT(median, 1000.0) << name << ": a dust storm, under 1 km (WMO)";
    }
    if (regime == Regime::storm) {
      EXPECT_LT(median, 1500.0) << name;
    }
    if (regime == Regime::rain) {
      EXPECT_GT(median, 2000.0) << name << ": moderate rain hides little";
    }
  }
}

// When each of the base's quantities has made half its change (seconds after the request).
std::array<float, 10> halfway(Regime from, Regime to, float over) {
  wx::Director d(kSeed, from);
  const wx::Target a = d.base();
  d.request(to, over);
  const wx::Target b = d.heading();
  std::array<float, 10> at;
  at.fill(std::numeric_limits<float>::infinity());
  run(d, over, [&](const wx::Director& s) {
    const wx::Target& x = s.base();
    const float va[] = {a.cloud, a.precip, a.hail, a.mean, a.gust, a.temp, a.lightning, a.fog, a.dust, a.humidity};
    const float vb[] = {b.cloud, b.precip, b.hail, b.mean, b.gust, b.temp, b.lightning, b.fog, b.dust, b.humidity};
    const float vx[] = {x.cloud, x.precip, x.hail, x.mean, x.gust, x.temp, x.lightning, x.fog, x.dust, x.humidity};
    for (std::size_t k = 0; k < 10; ++k)
      if (std::isinf(at[k]) && va[k] != vb[k] && std::abs(vx[k] - va[k]) >= 0.5f * std::abs(vb[k] - va[k])) at[k] = s.time();
  });
  return at;
}
enum : std::size_t { h_cloud, h_precip, h_hail, h_mean, h_gust, h_temp, h_lightning, h_fog, h_dust, h_humidity };

TEST(WeatherDirector, TransitionsKeepTheOrderOfRealWeather) {
  // building: the clouds, then the gust front (the wind, its gusts and the drop in temperature), then the downpour, and
  // the lightning and the hail with the heaviest rain
  const auto up = halfway(Regime::clear, Regime::storm, 60.f);
  EXPECT_LT(up[h_cloud], up[h_mean]) << "clouds build before the wind comes";
  EXPECT_LT(up[h_mean], up[h_precip]) << "the gust front arrives before the downpour";
  EXPECT_LE(up[h_temp], up[h_precip]) << "the cold air with it";
  EXPECT_LT(up[h_precip], up[h_lightning]) << "lightning comes with the heaviest rain";
  EXPECT_LT(up[h_precip], up[h_hail]);
  EXPECT_EQ(up[h_humidity], up[h_cloud]) << "the air moistens as the clouds build: the cloud base comes down before the rain";
  EXPECT_LT(up[h_humidity], up[h_precip]);
  // and the gust front is all in before the rain has begun in earnest
  {
    wx::Director d(kSeed, Regime::clear);
    d.request(Regime::storm, 60.f);
    float rain_at_front = -1.f;
    run(d, 60.f, [&](const wx::Director& s) {
      if (rain_at_front < 0.f && s.base().mean >= 0.9f * s.heading().mean + 0.1f * wx::target(Regime::clear).mean) rain_at_front = s.base().precip / s.heading().precip;
    });
    EXPECT_GE(rain_at_front, 0.f);
    EXPECT_LT(rain_at_front, 0.2f) << "the wind at 90 % of its rise while the rain is under a fifth of its";
  }
  // clearing: in reverse
  const auto down = halfway(Regime::storm, Regime::clear, 60.f);
  EXPECT_LT(down[h_lightning], down[h_precip]) << "the lightning stops first";
  EXPECT_LT(down[h_hail], down[h_precip]);
  EXPECT_LT(down[h_precip], down[h_mean]) << "then the rain, then the wind eases";
  EXPECT_LT(down[h_mean], down[h_cloud]) << "and last the clouds break";
  // a haboob: the wind before the dust it raises; a cold front's snow: the cold before the snow
  const auto haboob = halfway(Regime::clear, Regime::dust, 60.f);
  EXPECT_LT(haboob[h_mean], haboob[h_dust]);
  const auto front = halfway(Regime::clear, Regime::snow, 60.f);
  EXPECT_LT(front[h_cloud], front[h_temp]);
  EXPECT_LT(front[h_temp], front[h_precip]);
  // fog lifts before the sky clears, and its base rises with the clouds as the air dries
  const auto lift = halfway(Regime::fog, Regime::clear, 60.f);
  EXPECT_LT(lift[h_fog], lift[h_cloud]);
  EXPECT_LT(lift[h_fog], lift[h_humidity]);
  {
    wx::Director d(kSeed, Regime::fog);
    d.request(Regime::clear, 60.f);
    float last = 0.f;
    run(d, 60.f, [&](const wx::Director& s) {
      EXPECT_GE(s.value(wx::m_base), last - 10.f) << "the base rises as the fog lifts into low cloud and the sky clears (the temperature's noise jitters it a few metres)";
      last = s.value(wx::m_base);
    });
    EXPECT_GT(last, 700.f);
  }
}

TEST(WeatherDirector, TheGroundWetsInTheRainAndDriesByPenman) {
  // a clear day: the ground is dry and the air would dry it (Penman's E_a at the frame's temperature, humidity and wind)
  wx::Director d(kSeed, Regime::clear);
  EXPECT_EQ(d.value(wx::m_water), 0.f);
  EXPECT_EQ(d.value(wx::m_wet), 0.f);
  EXPECT_EQ(d.value(wx::m_puddle), 0.f);
  float evap_clear = 0.f;
  run(d, 60.f, [&](const wx::Director& s) {
    EXPECT_EQ(s.value(wx::m_evaporation), wx::evaporation_mm_h(s.value(wx::m_temp), s.value(wx::m_humidity), wx::wind_2m(s.value(wx::m_wind))));
    EXPECT_EQ(s.value(wx::m_water), 0.f) << "nothing to dry";
    evap_clear += s.value(wx::m_evaporation) * kDt / 60.f;
  });
  EXPECT_GT(evap_clear, 0.1f) << "a clear day's drying power, mm/h";
  EXPECT_LT(evap_clear, 1.f);
  // the rain comes: the water on the ground rises by the rain less the drying, the ground darkens when a film of
  // kWetFilm has fallen, and the water never exceeds the depression storage (the excess runs off): puddles
  d.request(Regime::rain, 10.f);
  run(d, 10.f);
  float water = d.value(wx::m_water), darkened_at = -1.f;
  double fallen = water, dried = 0.0;  // what fell and what dried since, mm (the first frames' rain is in the start)
  run(d, 3600.f, [&](const wx::Director& s) {
    EXPECT_GE(s.value(wx::m_water), water) << "it only rises while the rain beats the drying";
    EXPECT_LE(s.value(wx::m_water), wx::kDepressionStorage);
    EXPECT_EQ(s.value(wx::m_puddle), s.value(wx::m_water) / wx::kDepressionStorage);
    EXPECT_EQ(s.value(wx::m_snow), 0.f);
    fallen += s.value(wx::m_rain) * kDt / 3600.0;
    dried += s.value(wx::m_evaporation) * kDt / 3600.0;
    if (s.value(wx::m_water) < wx::kDepressionStorage) {
      EXPECT_NEAR(s.value(wx::m_water), fallen - dried, 1e-2 * fallen + 1e-3) << "the rain less the drying";
    }
    if (darkened_at < 0.f && s.value(wx::m_wet) >= 0.5f) darkened_at = s.time();
    water = s.value(wx::m_water);
  });
  EXPECT_GT(fallen, 2.0 * wx::kDepressionStorage) << "an hour of moderate rain is several times what the ground holds";
  EXPECT_EQ(water, wx::kDepressionStorage) << "full: the rest ran off";
  EXPECT_EQ(d.value(wx::m_puddle), 1.f);
  EXPECT_EQ(d.value(wx::m_wet), 1.f);
  EXPECT_GT(darkened_at, 70.f) << "the ground darkens once half the film has fallen: at 6 mm/h, 0.05 mm is 30 s of rain";
  EXPECT_LT(darkened_at, 160.f);
  // it clears: the water only falls, by the evaporation the state says, and a puddle outlasts the hour (Penman's 0.26 mm/h
  // on a mild day: 1.5 mm is an afternoon)
  d.request(Regime::clear, 10.f);
  run(d, 10.f);
  const float water0 = water = d.value(wx::m_water);
  EXPECT_GT(water0, 0.99f * wx::kDepressionStorage);
  dried = 0.0;
  run(d, 3600.f, [&](const wx::Director& s) {
    EXPECT_EQ(s.value(wx::m_rain), 0.f);
    EXPECT_LE(s.value(wx::m_water), water) << "it only dries";
    dried += s.value(wx::m_evaporation) * kDt / 3600.0;
    water = s.value(wx::m_water);
  });
  EXPECT_NEAR(d.value(wx::m_water), water0 - dried, 2e-3) << "down by what evaporated";
  EXPECT_GT(dried, 0.1);
  EXPECT_LT(dried, 1.0);
  EXPECT_GT(d.value(wx::m_water), 0.5f) << "the puddles outlast the hour";
  EXPECT_EQ(d.value(wx::m_wet), 1.f) << "and the ground is still dark";
  // fog keeps the ground wet: saturated air dries nothing
  wx::Director f(kSeed + 2, Regime::rain);
  EXPECT_EQ(f.value(wx::m_water), wx::kDepressionStorage) << "it starts as long rain leaves it";
  f.request(Regime::fog, 20.f);
  run(f, 1200.f);
  const float in_fog = f.value(wx::m_water);
  EXPECT_GT(in_fog, 0.9f * wx::kDepressionStorage);
  run(f, 2400.f, [&](const wx::Director& s) {
    EXPECT_EQ(s.value(wx::m_evaporation), 0.f);
    EXPECT_EQ(s.value(wx::m_water), in_fog) << "to the bit";
  });
  // a dust storm's hot dry wind dries fastest; a rainy day's air hardly at all; snow does not wet the ground
  wx::Director dust(kSeed + 4, Regime::dust), rain(kSeed + 4, Regime::rain), snow(kSeed + 4, Regime::snow);
  for (wx::Director* s : {&dust, &rain, &snow}) s->step(kDt);
  EXPECT_GT(dust.value(wx::m_evaporation), 1.5f) << "mm/h (about 2.3 for this seed: 55 mm a day)";
  EXPECT_LT(rain.value(wx::m_evaporation), 0.2f);
  EXPECT_GT(dust.value(wx::m_evaporation), 10.f * rain.value(wx::m_evaporation));
  EXPECT_EQ(dust.value(wx::m_water), 0.f);
  EXPECT_EQ(snow.value(wx::m_water), 0.f) << "snow lies (nothing lies yet): it does not wet the ground";
  // the member is read by its name, as a script does
  EXPECT_EQ(wx::member_of("wet"), wx::m_wet);
  EXPECT_EQ(wx::member_of("puddle"), wx::m_puddle);
  EXPECT_EQ(wx::member_of("evaporation"), wx::m_evaporation);
}

TEST(WeatherDirector, TheStormMovesAcrossTheSceneWithTheWind) {
  // Taylor's hypothesis (docs/WEATHER.md §3.13): the weather at the camera is a pattern the mean wind carries along x.
  // `carried` is the air it has carried past the camera since the start (m along +x, the mean wind's integral), and
  // `front` where a transition's leading edge is: at the camera at the request, then downwind at the mean wind of the
  // moment. So a storm comes in from upwind and crosses the scene at the wind's speed, and a layer x metres downwind
  // sees it begin x / U seconds after the camera does.
  EXPECT_FLOAT_EQ(wx::taylor_length_m(11.f, 90.f), 990.f) << "a storm's rain cell, 90 s at 11 m/s, is about a kilometre along the wind";
  EXPECT_FLOAT_EQ(wx::taylor_length_m(10.f, wx::gust_time(10.f)), wx::kGustLength) << "the gusts' time scale is their length scale over the wind";
  wx::Director d(kSeed, Regime::clear), back(kSeed, Regime::clear, -1.f);
  EXPECT_EQ(d.value(wx::m_carried), 0.f);
  EXPECT_EQ(d.value(wx::m_front), 0.f);
  float last = 0.f;
  double sum = 0;
  run(d, 60.f, [&](const wx::Director& s) {
    EXPECT_NEAR(s.value(wx::m_carried) - last, s.value(wx::m_mean) * kDt, 1e-4f) << "each frame, the mean wind's metres";
    last = s.value(wx::m_carried);
    sum += s.value(wx::m_mean) * kDt;
  });
  EXPECT_NEAR(d.value(wx::m_carried), sum, 1e-3 * sum);
  EXPECT_GT(d.value(wx::m_carried), 100.f) << "a minute of a light breeze: about 180 m of air past the camera";
  EXPECT_LT(d.value(wx::m_carried), 400.f);
  EXPECT_EQ(d.value(wx::m_front), d.value(wx::m_carried)) << "no request yet: the start's weather has been coming since the start";
  run(back, 60.f);
  EXPECT_EQ(back.value(wx::m_carried), -d.value(wx::m_carried)) << "the wind the other way carries it the other way, to the bit";
  EXPECT_EQ(back.value(wx::m_front), -d.value(wx::m_front));
  // a storm requested: its front leaves the camera and goes downwind at the mean wind of the moment (the clear day's
  // breeze until the gust front, the storm's wind after), where the wind has carried the air since the request
  const float carried0 = d.value(wx::m_carried), u_clear = d.base().mean;
  d.request(Regime::storm, 60.f);
  const float u_storm = d.heading().mean;
  float front = 0.f, reached_200 = -1.f;
  double travelled = 0;
  run(d, 60.f, [&](const wx::Director& s) {
    EXPECT_GE(s.value(wx::m_front), front) << "it only goes on downwind";
    EXPECT_NEAR(s.value(wx::m_front), s.value(wx::m_carried) - carried0, 0.01f) << "the air carried since the request";
    front = s.value(wx::m_front);
    travelled += s.base().mean * kDt;
    if (reached_200 < 0.f && front >= 200.f) reached_200 = s.time() - 60.f;
  });
  EXPECT_NEAR(front, travelled, 1e-3 * travelled);
  EXPECT_GT(front, u_clear * 60.f) << "further than the breeze would have taken it: the storm's wind came";
  EXPECT_LT(front, u_storm * 60.f) << "less than the storm's own wind would have: it came part way through";
  // Taylor: a layer 200 m downwind sees the storm begin when the front reaches it, 200 / U after the camera did
  EXPECT_GT(reached_200, 200.f / u_storm) << "s after the request";
  EXPECT_LT(reached_200, 200.f / u_clear);
  // a new request puts a new front at the camera; the air carried past it is kept
  d.request(Regime::clear, 30.f);
  d.step(kDt);
  EXPECT_LT(d.value(wx::m_front), 1.f) << "the clearing's front, just off the camera";
  EXPECT_GT(d.value(wx::m_front), 0.f);
  EXPECT_NEAR(d.value(wx::m_carried), carried0 + front + d.base().mean * kDt, 0.01f);
  // read by name, as a script does (sky.front)
  EXPECT_EQ(wx::member_of("carried"), wx::m_carried);
  EXPECT_EQ(wx::member_of("front"), wx::m_front);
  EXPECT_EQ(wx::member_names()[static_cast<std::size_t>(wx::m_front)], "front");
}

TEST(WeatherDirector, TheSunStandsByTheHourAndShowsThroughBrokenCloud) {
  // a clear day at noon (the default hour): the sun high in the middle of the sky, its direct light most of the daylight
  // (Kasten and Czeplak's 70 % through a few clouds and clear air), hardly a shaft (clear air scatters too little)
  wx::Director d(kSeed, Regime::clear);
  EXPECT_EQ(d.hour(), 12.f);
  EXPECT_EQ(d.value(wx::m_hour), 12.f);
  EXPECT_NEAR(d.value(wx::m_elevation), 45.f, 1e-3f);
  EXPECT_NEAR(d.value(wx::m_sunx), 0.f, 1e-4f);
  double sun_clear = 0, shaft_clear = 0;
  int n = 0;
  run(d, 60.f, [&](const wx::Director& s) {
    EXPECT_EQ(s.value(wx::m_sun), wx::sun_share(s.value(wx::m_cloud)) * wx::daylight(s.value(wx::m_cloud)) *
                                      wx::beam_transmittance(s.value(wx::m_extinction), s.value(wx::m_elevation)))
        << "the direct beam's share of the daylight, through the weather";
    EXPECT_EQ(s.value(wx::m_shaft), wx::shaft_visibility(s.value(wx::m_cloud), s.value(wx::m_extinction), s.value(wx::m_elevation)));
    sun_clear += s.value(wx::m_sun);
    shaft_clear += s.value(wx::m_shaft);
    ++n;
  });
  sun_clear /= n;
  shaft_clear /= n;
  EXPECT_GT(sun_clear, 0.55) << "a clear day's sun";
  EXPECT_LT(sun_clear, 0.7);
  EXPECT_LT(shaft_clear, 0.05) << "no shafts in clear air";
  // the hour moves the sun and nothing else: morning on the left and lower, evening on the right, night none at all
  const float cloud_before = d.value(wx::m_cloud), rain_before = d.value(wx::m_rain);
  d.set_hour(9.f);
  EXPECT_EQ(d.value(wx::m_hour), 9.f);
  EXPECT_NEAR(d.value(wx::m_elevation), 30.f, 0.01f) << "9 in the morning at 45 N on the equinox";
  EXPECT_LT(d.value(wx::m_sunx), -0.5f) << "on the left";
  EXPECT_EQ(d.value(wx::m_cloud), cloud_before);
  EXPECT_EQ(d.value(wx::m_rain), rain_before);
  const float sun9 = d.value(wx::m_sun);
  d.set_hour(12.f);
  EXPECT_GT(d.value(wx::m_sun), sun9) << "a longer path through the air in the morning";
  d.set_hour(16.f);
  EXPECT_GT(d.value(wx::m_sunx), 0.5f) << "the afternoon: on the right";
  d.set_hour(21.f);
  EXPECT_LT(d.value(wx::m_elevation), 0.f) << "night";
  EXPECT_EQ(d.value(wx::m_sun), 0.f);
  EXPECT_EQ(d.value(wx::m_shaft), 0.f);
  d.set_hour(30.f);
  EXPECT_NEAR(d.value(wx::m_elevation), 0.f, 1e-3f) << "the hour wraps: 30 is 6, sunrise";
  d.set_hour(std::numeric_limits<float>::quiet_NaN());
  EXPECT_EQ(d.hour(), 12.f) << "no hour: noon";
  run(d, 1.f);
  EXPECT_EQ(d.value(wx::m_hour), 12.f) << "a step keeps the hour";
  // fog comes in under a building deck: the sun dims as the clouds close and the mist thickens, and on the way, while
  // the deck is broken and the air already hazy, its shafts show; in the fog there is no sun and no shaft
  d.request(Regime::fog, 60.f);
  float most_shaft = 0.f, shaft_at = 0.f, sun_last = d.value(wx::m_sun);
  run(d, 60.f, [&](const wx::Director& s) {
    if (s.value(wx::m_shaft) > most_shaft) {
      most_shaft = s.value(wx::m_shaft);
      shaft_at = s.progress();
    }
    EXPECT_LE(s.value(wx::m_sun), sun_last + 0.03f) << "the sun only dims as the fog comes in (the cloud's noise jitters it)";
    sun_last = s.value(wx::m_sun);
  });
  EXPECT_GT(most_shaft, 0.1f) << "shafts through the mist under the broken deck";
  EXPECT_GT(shaft_at, 0.1f);
  EXPECT_LT(shaft_at, 0.5f) << "while the deck is still broken";
  EXPECT_LT(d.value(wx::m_sun), 1e-4f) << "in the fog";
  EXPECT_LT(d.value(wx::m_shaft), 1e-3f);
  // under an overcast or a raining sky there is next to no direct sun; a dust storm hides it in its load, and shows more
  // shafts than a clear day as the sun breaks through its thinner moments
  const auto averages = [](Regime r) {
    wx::Director w(kSeed + 2, r);
    double sun = 0, shaft = 0;
    int k = 0;
    run(w, 1800.f, [&](const wx::Director& s) {
      EXPECT_GE(s.value(wx::m_sun), 0.f);
      EXPECT_LE(s.value(wx::m_sun), 1.f);
      EXPECT_GE(s.value(wx::m_shaft), 0.f);
      EXPECT_LE(s.value(wx::m_shaft), 1.f);
      sun += s.value(wx::m_sun);
      shaft += s.value(wx::m_shaft);
      ++k;
    });
    return std::pair{sun / k, shaft / k};
  };
  const auto [sun_overcast, shaft_overcast] = averages(Regime::overcast);
  EXPECT_LT(sun_overcast, 0.06) << "overcast: Kasten and Czeplak's 7 % of a darkened sky";
  const auto [sun_rain, shaft_rain] = averages(Regime::rain);
  EXPECT_LT(sun_rain, 0.03);
  EXPECT_LT(shaft_rain, 0.1);
  const auto [sun_dust, shaft_dust] = averages(Regime::dust);
  EXPECT_LT(sun_dust, 0.5 * sun_clear) << "a haboob darkens the day";
  EXPECT_GT(shaft_dust, shaft_clear) << "and the sun's shafts stand in its dust";
  // read by name, as a script does (sky.sun)
  EXPECT_EQ(wx::member_of("hour"), wx::m_hour);
  EXPECT_EQ(wx::member_of("elevation"), wx::m_elevation);
  EXPECT_EQ(wx::member_of("sunx"), wx::m_sunx);
  EXPECT_EQ(wx::member_of("sun"), wx::m_sun);
  EXPECT_EQ(wx::member_of("shaft"), wx::m_shaft);
}

TEST(WeatherDirector, TheDeckFormsBeforeTheRainAndItsFootFollowsTheHaze) {
  // The deck (docs/WEATHER.md §3.15): the cover as a layer at the base, `deck`, and where its underside meets the haze,
  // `foot`. A clear day: fair-weather puffs under four oktas, no layer, and the high base seen 40 km stands at the horizon.
  wx::Director d(kSeed, Regime::clear);
  double deck_clear = 0;
  int n = 0;
  run(d, 60.f, [&](const wx::Director& s) {
    EXPECT_EQ(s.value(wx::m_deck), wx::deck_cover(s.value(wx::m_cloud)));
    EXPECT_EQ(s.value(wx::m_foot), wx::deck_foot_deg(s.value(wx::m_base), s.value(wx::m_visibility)));
    EXPECT_LT(s.value(wx::m_deck), 0.5f) << "scattered puffs, no deck";
    EXPECT_GT(s.value(wx::m_foot), 0.5f) << "fair-weather cumulus at 1200 m, seen 40 km: 1.7 degrees";
    EXPECT_LT(s.value(wx::m_foot), 3.f);
    deck_clear += s.value(wx::m_deck);
    ++n;
  });
  EXPECT_LT(deck_clear / n, 0.05);
  // a storm comes in: the deck is a layer as the clouds build, before the downpour (§2.2's order); in the storm the deck
  // is solid and its foot high, the low base lost in the rain's haze below about 20 degrees
  d.request(Regime::storm, 60.f);
  float deck_at = -1.f, rain_at = -1.f;
  run(d, 60.f, [&](const wx::Director& s) {
    if (deck_at < 0.f && s.value(wx::m_deck) >= 0.5f) deck_at = s.time();
    if (rain_at < 0.f && s.base().precip >= 0.5f * s.heading().precip) rain_at = s.time();
  });
  EXPECT_GT(deck_at, 60.f);
  EXPECT_LT(deck_at, rain_at) << "the deck forms before the rain";
  double foot_storm = 0;
  n = 0;
  run(d, 600.f, [&](const wx::Director& s) {
    EXPECT_EQ(s.value(wx::m_deck), 1.f) << "a solid deck";
    EXPECT_GT(s.value(wx::m_foot), 5.f);
    foot_storm += s.value(wx::m_foot);
    ++n;
  });
  foot_storm /= n;
  EXPECT_GT(foot_storm, 15.0) << "the storm's low deck in its rain: lost below about 20 degrees";
  EXPECT_LT(foot_storm, 35.0);
  // it clears: the rain stops and the haze thins, so the foot comes down toward the horizon while the deck still stands;
  // the deck breaks last, as the clouds do (§2.2)
  d.request(Regime::clear, 60.f);
  float rain_gone = -1.f, deck_broken = -1.f;
  run(d, 60.f, [&](const wx::Director& s) {
    if (rain_gone < 0.f && s.base().precip < 1.f) rain_gone = s.time();
    if (deck_broken < 0.f && s.value(wx::m_deck) < 0.5f) deck_broken = s.time();
  });
  EXPECT_GT(rain_gone, 0.f);
  EXPECT_GT(deck_broken, rain_gone) << "the rain is gone before the deck breaks";
  EXPECT_LT(d.value(wx::m_deck), 0.05f);
  EXPECT_LT(d.value(wx::m_foot), 3.f) << "and the foot is back at the horizon";
  // fog comes in under a lowering deck: as the air moistens the base comes down and as the fog thickens the visibility
  // falls, so on the way the foot lifts to several degrees (a base of a few hundred metres under a kilometre or two of
  // haze) before the deck reaches the ground, where the fog is the cloud: base 0, foot 0, a deck of 1
  wx::Director f(kSeed + 2, Regime::clear);
  f.request(Regime::fog, 60.f);
  float most = 0.f, at = 0.f;
  run(f, 60.f, [&](const wx::Director& s) {
    if (s.value(wx::m_foot) > most) {
      most = s.value(wx::m_foot);
      at = s.progress();
    }
  });
  EXPECT_GT(most, 4.f) << "the deck's foot lifts as it comes down into the thickening mist";
  EXPECT_GT(at, 0.15f);
  EXPECT_LT(at, 0.5f) << "before the fog is in";
  EXPECT_EQ(f.value(wx::m_base), 0.f) << "saturated: cloud at the ground";
  EXPECT_EQ(f.value(wx::m_foot), 0.f);
  EXPECT_EQ(f.value(wx::m_deck), 1.f);
  // a dust storm: a few clouds, no deck, and a base beyond the visibility (the load swings with the gusts' cube, §2.3):
  // whatever cloud there is, its underside is lost in the dust but for the sky more than halfway up
  wx::Director dust(kSeed + 4, Regime::dust);
  run(dust, 60.f, [](const wx::Director& s) {
    EXPECT_LT(s.value(wx::m_deck), 0.5f);
    EXPECT_GT(s.value(wx::m_foot), 45.f);
  });
  // read by name, as a script does (sky.deck, sky.foot)
  EXPECT_EQ(wx::member_of("deck"), wx::m_deck);
  EXPECT_EQ(wx::member_of("foot"), wx::m_foot);
  EXPECT_EQ(wx::member_names()[static_cast<std::size_t>(wx::m_foot)], "foot");
}

TEST(WeatherDirector, ARequestStartsFromWhereItIs) {
  wx::Director d(kSeed, Regime::clear);
  d.request(Regime::storm, 20.f);
  run(d, 12.f);
  const wx::Target before = d.base();
  EXPECT_LT(d.progress(), 1.f);
  d.request(Regime::fog, 20.f);
  EXPECT_EQ(d.progress(), 0.f);
  EXPECT_EQ(d.base().cloud, before.cloud) << "the new transition starts from the state as it was";
  EXPECT_EQ(d.base().precip, before.precip);
  d.step(kDt);
  EXPECT_NEAR(d.base().precip, before.precip, 0.01f * before.precip + 1e-3f);
  EXPECT_NEAR(d.base().mean, before.mean, 0.01f);
  const int n = d.requests();
  d.request(Regime::fog, 5.f);
  EXPECT_EQ(d.requests(), n) << "asking again for where it is heading changes nothing";
  EXPECT_EQ(wx::regime_name(d.regime()), "fog");
}

// --- the script ---------------------------------------------------------------------------------------------------------

TEST(WeatherScript, ParsesPrintsAndChecks) {
  const sc::Script s = sc::parse(kWeatherScene, "t");
  const std::string printed = sc::print(s);
  EXPECT_EQ(sc::print(sc::parse(printed, "t")), printed) << "the canonical form reads back the same";
  EXPECT_NE(printed.find("weather sky, seed 28500200, regime clear:\n  storm at 0.5, over 2\n  snow at 3.5, over 1\n"), std::string::npos) << printed;
  EXPECT_NO_THROW(sc::validate(sc::parse("scene size 64 x 64\nweather w, regime storm\nweather v, regime fog: snow at 2; rain at 4\n", "t")))
      << "a regime alone, or a schedule on one line";
  const std::string head = "scene size 64 x 64, length 4 s, ground 60\n";
  const auto err = [&](const std::string& body) { return error_of(head + body); };
  EXPECT_EQ(err("weather sky, regime clear:\n  strom at 3\n"), "t:3:3: 'strom' is not a regime (clear, overcast, rain, storm, snow, fog, dust) (did you mean 'storm'?)");
  EXPECT_NE(err("weather sky, regime strom\n").find("unknown name 'strom': a regime is clear, overcast, rain, storm, snow, fog or dust (did you mean 'storm'?)"),
            std::string::npos);
  EXPECT_NE(err("weather sky, regime clear:\n  rain at 3\n  storm at 2\n").find("in the order of their times"), std::string::npos);
  EXPECT_NE(err("weather sky, regime clear:\n  rain\n").find("say when the weather heads for rain"), std::string::npos);
  EXPECT_NE(err("input r = 0\nweather sky, regime r:\n  rain at 2\n").find("must be a constant"), std::string::npos);
  EXPECT_NE(err("weather sky, regime 9\n").find("a regime is clear, overcast"), std::string::npos);
  EXPECT_NE(err("weather sky, direction 2\n").find("direction is 1"), std::string::npos);
  EXPECT_NE(err("weather sky\nlet a = sky.rainn\n").find("a weather has no 'rainn' (did you mean 'rain'?)"), std::string::npos);
  EXPECT_NE(err("weather sky\nlet a = sky\nframe exposure a\n").find("a weather is not a value: use sky.rain"), std::string::npos);
  EXPECT_NE(err("weather sky\nmodule sky = tiny, size 32\n").find("already the name of a weather"), std::string::npos);
  EXPECT_NE(err("weather sky, regime clear:\n").find("schedule has no lines"), std::string::npos);
  EXPECT_NE(err("weather sky, seed 28500001\n").find("cannot be held exactly"), std::string::npos) << "an odd seed above 2^24";
  EXPECT_EQ(err("input h = 9\nweather sky, hour h + t / 60\n"), "no error") << "the sun's hour: an expression, re-evaluated each frame";
  EXPECT_EQ(sc::print(sc::parse("weather sky, regime overcast, hour 15.5\n", "t")), "weather sky, regime overcast, hour 15.5\n");
  EXPECT_NE(err("emit rain, in (0, 0), size 10 x 10, shutter 0.01\nemit snow, in (0, 0), size 10 x 10, shutter 0.01\n").find("'shutter' does not apply to emit snow"),
            std::string::npos);
  EXPECT_NE(err("emit ash, in (0, 0), size 10 x 10, rate 2\n").find("'rate' does not apply to emit ash"), std::string::npos);
  // the new keys reserve no word: a module may still be named after them
  EXPECT_EQ(err("effect tiny = \"tiny\"\nmodule depth = tiny, size 32\nmodule shutter = tiny, size 32\nlet weather = 1\nlet storm = 2\nlet rain = 3\n"), "no error");
}

TEST(WeatherScript, TheSceneAndTheEngineReadTheWeather) {
  sc::Scene scene = scene_of(kWeatherScene);
  std::vector<std::uint8_t> rgb(200 * 120 * 3);
  float most_rain = 0.f;
  for (int f = 0; f < 100; ++f) {
    scene.render(f, rgb);
    float v = 0.f;
    ASSERT_TRUE(scene.weather_value("sky.rain", v));
    EXPECT_EQ(v, scene.director(0).value(wx::m_rain));
    most_rain = std::max(most_rain, v);
    ASSERT_TRUE(scene.weather_value("sky.cloud", v));
    EXPECT_EQ(scene.module("m")->opacity, v) << "a module's setting follows the weather";
    ASSERT_TRUE(scene.weather_value("sky.front", v));
    EXPECT_EQ(v, scene.director(0).value(wx::m_front));
  }
  EXPECT_GT(most_rain, 10.f) << "the storm's rain";
  float v = 0.f;
  EXPECT_FALSE(scene.weather_value("sky.rian", v));
  EXPECT_FALSE(scene.weather_value("cloudy.rain", v));
  EXPECT_FALSE(scene.weather_value("rain", v));
  EXPECT_EQ(scene.weathers(), 1);
  // the same through the C API, driven by an input as an engine drives it, with a rule the flashes fire
  std::string tiny;
  {
    std::ostringstream os;
    ASSERT_TRUE(nfx::rollout::save_model(os, cm::testing::tiny_effect(3).m));
    tiny = os.str();
  }
  nvfx_effect* et = nullptr;
  ASSERT_EQ(nvfx_effect_load_memory(tiny.data(), tiny.size(), &et), NVFX_OK);
  const nvfx_scene_effect list[] = {{"tiny", et}};
  const std::string script = "scene size 160 x 90, fps 30, length 60 s, ground 85\neffect tiny = \"tiny\"\ninput regime = 0\ninput change = 4\ninput hour = 9\n"
                             "weather sky, seed 28500220, regime regime, over change, hour hour\n"
                             "module m = tiny, size 32, width 64, at (80, 80), start 1, seed 28500222, wind sky.lean\n"
                             "when sky.since == 0 as thunder, repeat:\n";
  nvfx_scene_desc d;
  nvfx_scene_desc_init(&d);
  d.script = script.c_str();
  d.effects = list;
  d.n_effects = 1;
  nvfx_scene* s = nullptr;
  nvfx_scene_error err;
  ASSERT_EQ(nvfx_scene_create(&d, &s, &err), NVFX_OK) << err.message;
  float value = -1.f;
  ASSERT_EQ(nvfx_scene_get_input(s, "sky.regime", &value), NVFX_OK);
  EXPECT_EQ(value, 0.f);
  ASSERT_EQ(nvfx_scene_get_input(s, "sky.visibility", &value), NVFX_OK);
  EXPECT_NEAR(value, 40000.f, 1.f) << "a clear day";
  EXPECT_EQ(nvfx_scene_get_input(s, "sky.nothing", &value), NVFX_ERROR_ARGUMENT);
  // the sun: the game's hour (an input the script hands to the director's `hour`) says where it stands
  ASSERT_EQ(nvfx_scene_get_input(s, "sky.hour", &value), NVFX_OK);
  EXPECT_EQ(value, 9.f);
  ASSERT_EQ(nvfx_scene_get_input(s, "sky.elevation", &value), NVFX_OK);
  EXPECT_NEAR(value, 30.f, 0.01f) << "9 in the morning";
  ASSERT_EQ(nvfx_scene_get_input(s, "sky.sunx", &value), NVFX_OK);
  EXPECT_LT(value, -0.5f) << "on the left";
  ASSERT_EQ(nvfx_scene_get_input(s, "sky.sun", &value), NVFX_OK);
  EXPECT_GT(value, 0.4f) << "a clear morning's sun";
  ASSERT_EQ(nvfx_scene_set_input(s, "hour", 20.f), NVFX_OK);  // evening
  ASSERT_EQ(nvfx_scene_set_input(s, "regime", 3.f), NVFX_OK);  // a storm, in 4 s
  ASSERT_EQ(nvfx_scene_step(s, 20.0, nullptr), NVFX_OK);
  ASSERT_EQ(nvfx_scene_get_input(s, "sky.hour", &value), NVFX_OK);
  EXPECT_EQ(value, 20.f);
  ASSERT_EQ(nvfx_scene_get_input(s, "sky.elevation", &value), NVFX_OK);
  EXPECT_LT(value, 0.f) << "the sun has set";
  ASSERT_EQ(nvfx_scene_get_input(s, "sky.sun", &value), NVFX_OK);
  EXPECT_EQ(value, 0.f);
  ASSERT_EQ(nvfx_scene_get_input(s, "sky.regime", &value), NVFX_OK);
  EXPECT_EQ(value, 3.f);
  ASSERT_EQ(nvfx_scene_get_input(s, "sky.rain", &value), NVFX_OK);
  EXPECT_GT(value, 10.f);
  ASSERT_EQ(nvfx_scene_get_input(s, "sky.progress", &value), NVFX_OK);
  EXPECT_EQ(value, 1.f);
  float strikes = 0.f;
  ASSERT_EQ(nvfx_scene_get_input(s, "sky.strike", &strikes), NVFX_OK);
  int thunder = 0;
  ASSERT_EQ(nvfx_scene_rule_state(s, "thunder", &thunder, nullptr), NVFX_OK);
  EXPECT_GT(strikes, 0.f);
  EXPECT_EQ(static_cast<float>(thunder), strikes) << "a rule on sky.since tells the game of every flash (for its thunder)";
  nvfx_scene_free(s);
  nvfx_effect_free(et);
}

// Live rain particles inside a region: their count, mean fall speed (world pixels a second) and mean streak.
struct Drops {
  int n = 0;
  double fall = 0, streak = 0, streak_error = 0;
};
Drops drops_of(sc::Scene& scene, cm::Kind kind, float x0, float y0, float x1, float y1, float shutter) {
  Drops d;
  cm::Particles& P = scene.particles();
  int i = 0;
  P.each([&](float x, float y, float&, float& vy) {
    if (P.kind(i) == kind && x >= x0 && x < x1 && y >= y0 && y < y1) {
      ++d.n;
      d.fall += vy;
      if (kind == cm::Kind::rain && P.streak(i) > 0.f) {
        d.streak += P.streak(i);
        d.streak_error = std::max(d.streak_error, std::abs(static_cast<double>(P.streak(i)) - vy * shutter) / (vy * shutter));
      }
    }
    ++i;
  });
  if (d.n) {
    d.fall /= d.n;
    d.streak /= d.n;
  }
  return d;
}

TEST(WeatherScript, ThePhysicalRainHoldsItsDistribution) {
  // 600 x 400 world pixels at 600 a metre: 0.667 m^2 of a 0.5 m slab
  for (const float R : {2.f, 20.f, 60.f}) {
    sc::Scene scene = scene_of(std::format("scene size 600 x 500, fps 30, length 3 s, ground 2000\nparticles capacity 60000\n"
                                           "emit rain, in (0, 0), size 600 x 400, rate {}, scale 600, depth 0.5, seed 28500230\n", R));
    std::vector<std::uint8_t> rgb(600 * 500 * 3);
    double n = 0, fall = 0, streak = 0, worst = 0;
    for (int f = 0; f < 60; ++f) {
      scene.render(f, rgb);
      if (f < 20) continue;  // the region filled at once, then fed from its top
      const Drops d = drops_of(scene, cm::Kind::rain, 0.f, 100.f, 600.f, 400.f, 0.5f / 30.f);
      n += d.n;
      fall += d.fall;
      streak += d.streak;
      worst = std::max(worst, d.streak_error);
    }
    n /= 40;
    fall /= 40;
    streak /= 40;
    const double want = wx::mp_number(R) * 0.5 * (600.0 * 300.0) / (600.0 * 600.0);
    EXPECT_NEAR(n, want, 0.08 * want + 3.0) << R << " mm/h: as many drops as Marshall-Palmer holds in the slab";
    EXPECT_NEAR(fall, wx::mp_mean_speed(R) * 600.0, 0.03 * wx::mp_mean_speed(R) * 600.0) << "falling at Atlas's speeds, sized by Marshall-Palmer";
    EXPECT_NEAR(streak, fall * 0.5 / 30.0, 0.02 * fall * 0.5 / 30.0) << "each streak as long as it falls in half a frame";
    EXPECT_LT(worst, 0.01) << "drop by drop";
  }
}

TEST(WeatherScript, StreaksAndCountsFollowTheRate) {
  const auto at = [](float R, const char* extra) {
    sc::Scene scene = scene_of(std::format("scene size 300 x 300, fps 30, length 2 s, ground 2000\nparticles capacity 60000\n"
                                           "emit rain, in (0, 0), size 300 x 300, rate {}, scale 600{}, seed 28500232\n", R, extra));
    std::vector<std::uint8_t> rgb(300 * 300 * 3);
    for (int f = 0; f < 30; ++f) scene.render(f, rgb);
    return drops_of(scene, cm::Kind::rain, 0.f, 0.f, 300.f, 300.f, 0.5f / 30.f);
  };
  const Drops light = at(2.f, ""), heavy = at(50.f, ""), long_shutter = at(50.f, ", shutter 0.04");
  EXPECT_GT(heavy.n, 3 * light.n) << "heavier rain: more drops";
  EXPECT_GT(heavy.streak, 1.1 * light.streak) << "and bigger, faster ones: longer streaks";
  EXPECT_NEAR(long_shutter.streak / heavy.streak, 0.04 / (0.5 / 30.0), 0.05 * 0.04 / (0.5 / 30.0)) << "streaks as long as the shutter is open";
  const Drops dense = at(50.f, ", depth 1");
  EXPECT_NEAR(static_cast<double>(dense.n) / heavy.n, 2.0, 0.2) << "a slab twice as deep, twice the drops";
  const Drops given = at(50.f, ", density 3");
  EXPECT_NEAR(given.n, 3.0 * 9.0, 8.0) << "a density given is the density drawn";
}

TEST(WeatherScript, PhysicalSnowAndHail) {
  sc::Scene scene = scene_of("scene size 600 x 500, fps 30, length 6 s, ground 2000\nparticles capacity 60000\n"
                             "emit snow, in (0, 0), size 600 x 400, rate 1.5, scale 300, depth 0.5, seed 28500234\n"
                             "emit hail, in (0, 0), size 600 x 400, density 1, speed 900, seed 28500236\n");
  std::vector<std::uint8_t> rgb(600 * 500 * 3);
  double n = 0, fall = 0, hail_fall = 0;
  int frames = 0;
  for (int f = 0; f < 120; ++f) {
    scene.render(f, rgb);
    if (f < 60) continue;
    const Drops s = drops_of(scene, cm::Kind::snow, 0.f, 100.f, 600.f, 400.f, 0.f);
    const Drops h = drops_of(scene, cm::Kind::hail, 0.f, 0.f, 600.f, 300.f, 0.f);
    n += s.n;
    fall += s.fall;
    hail_fall += h.fall;
    ++frames;
  }
  n /= frames;
  fall /= frames;
  hail_fall /= frames;
  const double want = wx::snow_number(1.5f) * 0.5 * (600.0 * 300.0) / (300.0 * 300.0);
  EXPECT_NEAR(n, want, 0.1 * want) << "as many flakes as carry 1.5 mm/h of melted water";
  EXPECT_NEAR(fall, wx::flake_mean_speed() * 300.0, 0.05 * wx::flake_mean_speed() * 300.0) << "at Locatelli and Hobbs's speeds";
  EXPECT_NEAR(hail_fall, 900.0, 90.0) << "hail at the speed it is given";
}

TEST(WeatherScript, ItKeepsTheScenesRandomNumbers) {
  const auto run_scene = [](const std::string& extra) {
    sc::Scene scene = scene_of("scene size 200 x 120, fps 30, length 3 s, ground 115\nparticles capacity 20000\neffect tiny = \"tiny\"\n"
                               "module f = tiny, size 32, width 64, at (100, 110), start 1, seed 28500240\n"
                               "emit sparks, along (20, 100) to (180, 100), count 3\nemit embers, on f\n" +
                               extra);
    std::vector<std::uint8_t> rgb(200 * 120 * 3);
    for (int f = 0; f < 80; ++f) scene.render(f, rgb);
    return std::pair{scene.particles().rng, rgb};
  };
  const auto [rng0, rgb0] = run_scene("");
  const std::string director = "weather sky, seed 28500242, regime storm\n";
  const auto [rng1, rgb1] = run_scene(director);
  EXPECT_EQ(rng0, rng1);
  EXPECT_EQ(rgb0, rgb1) << "a director that drives nothing changes nothing";
  const auto [rng2, rgb2] = run_scene(director + std::string(R"(
emit rain, in (0, 0), size 200 x 110, rate sky.rain, scale 300, wind (sky.u * 30, 0), seed 28500244
emit hail, in (0, 0), size 200 x 110, density 2 * sky.hail, speed 600, seed 28500246
emit leaves, in (0, 0), size 200 x 110, density sky.leaves, seed 28500248
emit arc, along (100 + 80 * sky.bolt, 0) to (100, 110), rate 0.001, seed 28500250 + 2 * sky.strike, brightness 3 * sky.flash
light flash (sky.flash, sky.flash, sky.flash)
frame daylight sky.light, overcast sky.cloud, visibility sky.visibility
)"));
  EXPECT_EQ(rng0, rng2) << "the director and the layers it drives draw none of the scene's random numbers";
  EXPECT_NE(rgb0, rgb2);
}

TEST(WeatherScript, TheSameFramesOnAnyNumberOfThreads) {
  std::vector<std::vector<std::uint8_t>> runs;
  for (const auto& [threads, overlap] : {std::pair{1, false}, std::pair{3, false}, std::pair{2, true}}) {
    sc::Scene scene = scene_of(kWeatherScene, threads, overlap);
    std::vector<std::uint8_t> rgb(200 * 120 * 3), all;
    for (int f = 0; f < scene.frames(); f += 1) {
      scene.render(f, rgb);
      if (f % 10 == 0) all.insert(all.end(), rgb.begin(), rgb.end());
    }
    runs.push_back(std::move(all));
  }
  EXPECT_EQ(runs[0], runs[1]);
  EXPECT_EQ(runs[0], runs[2]);
}

// --- the library ----------------------------------------------------------------------------------------------------------

constexpr const char* kWeatherCatalogue = R"(
effect tiny = "tiny.nvfx"
entry squall, category weather, size 200 x 120, anchor base, loop, show 1, about "a director in an entry"
param regime = 0, range (0, 6), about "where it heads"
param change = 2, range (1, 60), about "how long it takes"
weather sky, seed 28500260, regime regime, over change
module cloud = tiny, size 32, width 64 * scale, at (X, Y - 40), start 1, seed 28500262, wind sky.lean, opacity sky.cloud
emit rain, in (X - 100, Y - 120), size 200 x 120, rate sky.rain, scale 300 * scale, seed 28500264
)";

TEST(WeatherLibrary, AnEntryCarriesItsOwnWeather) {
  namespace lib = nfx::library;
  const std::vector<std::pair<std::string, std::string>> files = {{"w.nvfxl", kWeatherCatalogue}};
  const lib::Catalogue c = lib::parse(files);
  lib::StageSpec spec;
  spec.width = 240;
  spec.height = 160;
  spec.ground = 150.f;
  spec.rooms.push_back({c.index("squall"), 2});
  const lib::Stage stage = lib::build_stage(c, spec);
  ASSERT_EQ(stage.slots.size(), 2u);
  const std::string text = stage.text();
  const std::string a = stage.slots[0].prefix + "sky", b = stage.slots[1].prefix + "sky";
  EXPECT_NE(text.find("weather " + a + ", seed 28500260, regime " + stage.slots[0].params[0] + ", over " + stage.slots[0].params[1]), std::string::npos) << text;
  EXPECT_NE(text.find(a + ".rain"), std::string::npos) << "its members renamed with its room";
  sc::Scene scene(stage.script, [](const std::string&) { return cm::testing::tiny_effect(3); }, {});
  std::vector<std::uint8_t> rgb(240 * 160 * 3);
  scene.render(0, rgb);
  scene.set_input(scene.input_index(stage.slots[0].params[0]), 3.f);  // the game asks the first room for a storm
  for (int f = 1; f < 150; ++f) scene.render(f, rgb);
  float rain_a = -1.f, rain_b = -1.f;
  ASSERT_TRUE(scene.weather_value(a + ".rain", rain_a));
  ASSERT_TRUE(scene.weather_value(b + ".rain", rain_b));
  EXPECT_GT(rain_a, 10.f) << "the first room's weather turned to a storm";
  EXPECT_EQ(rain_b, 0.f) << "the second room's stays clear";
  // through the C API: a parameter the game sets
  std::string bytes;
  {
    std::ostringstream os;
    ASSERT_TRUE(nfx::rollout::save_model(os, cm::testing::tiny_effect(3).m));
    bytes = os.str();
  }
  nvfx_effect* et = nullptr;
  ASSERT_EQ(nvfx_effect_load_memory(bytes.data(), bytes.size(), &et), NVFX_OK);
  const nvfx_library_file file{"w.nvfxl", kWeatherCatalogue};
  nvfx_library_desc ld;
  nvfx_library_desc_init(&ld);
  ld.files = &file;
  ld.n_files = 1;
  nvfx_library* l = nullptr;
  nvfx_scene_error err;
  ASSERT_EQ(nvfx_library_open(&ld, &l, &err), NVFX_OK) << err.message;
  const nvfx_stage_room rooms[] = {{"squall", 1, 0}};
  const nvfx_scene_effect fx{"tiny.nvfx", et};
  nvfx_stage_desc sd;
  nvfx_stage_desc_init(&sd);
  sd.rooms = rooms;
  sd.n_rooms = 1;
  sd.effects = &fx;
  sd.n_effects = 1;
  sd.width = 240;
  sd.height = 160;
  nvfx_stage* st = nullptr;
  ASSERT_EQ(nvfx_stage_create(l, &sd, &st, &err), NVFX_OK) << err.message;
  nvfx_fx h = 0;
  ASSERT_EQ(nvfx_stage_spawn(st, "squall", 120.f, 150.f, 1.f, -1.0, &h), NVFX_OK);
  ASSERT_EQ(nvfx_stage_set_param(st, h, "regime", 3.f), NVFX_OK);
  nvfx_scene* sc_ = nvfx_stage_scene(st);
  ASSERT_EQ(nvfx_scene_step(sc_, 5.0, nullptr), NVFX_OK);
  float rain = 0.f;
  ASSERT_EQ(nvfx_scene_get_input(sc_, (stage.slots[0].prefix + "sky.rain").c_str(), &rain), NVFX_OK) << "the room's weather, read by its name";
  EXPECT_GT(rain, 10.f);
  nvfx_stage_free(st);
  nvfx_library_free(l);
  nvfx_effect_free(et);
  // an entry's weather has no schedule
  const std::string bad = "effect tiny = \"tiny.nvfx\"\nentry s, category weather, size 64 x 64, anchor base, loop, about \"x\"\nweather sky, regime clear:\n  storm at 2\n";
  const std::vector<std::pair<std::string, std::string>> bad_files = {{"b.nvfxl", bad}};
  try {
    (void)lib::parse(bad_files);
    ADD_FAILURE() << "a schedule in an entry";
  } catch (const sc::Error& e) {
    EXPECT_NE(std::string(e.what()).find("an entry's weather has no schedule"), std::string::npos) << e.what();
  }
}

// --- the day ------------------------------------------------------------------------------------------------------------

TEST(WeatherDay, TheNightIsAsItWas) {
  const std::string base = "scene size 160 x 120, fps 30, length 1 s, ground 100\nlight gain 0.14, flash (0.1, 0.1, 0.1)\n";
  const auto frame_of = [](const std::string& text) {
    sc::Scene scene = scene_of(text);
    std::vector<std::uint8_t> rgb(160 * 120 * 3);
    for (int f = 0; f < 5; ++f) scene.render(f, rgb);
    return rgb;
  };
  EXPECT_EQ(frame_of(base), frame_of(base + "frame daylight 0, overcast 0, visibility infinity\n")) << "by default, the night of every scene before the day, to the bit";
  EXPECT_NE(frame_of(base), frame_of(base + "frame daylight 1\n"));
}

TEST(WeatherDay, TheFrameCarriesTheWetGroundAndTheCloudDeck) {
  // the director's ground and ceiling as keys of `frame` (docs/COMPOSE.md §4.9.1): `wet`, `puddle`, `deck`, `foot`, each
  // an expression re-evaluated every frame as `daylight` is, each 0 by default
  const std::string head = "scene size 160 x 120, fps 30, length 2 s, ground 100\n";
  // they parse, print and are checked
  const std::string printed = sc::print(sc::parse(head + "frame daylight 1, wet 0.5, puddle 0.25, deck 1, foot 3.5\n", "t"));
  EXPECT_NE(printed.find("wet 0.5, puddle 0.25, deck 1, foot 3.5"), std::string::npos) << printed;
  EXPECT_EQ(sc::print(sc::parse(printed, "t")), printed) << "the canonical form reads back the same";
  EXPECT_NE(error_of(head + "frame wett 1\n").find("'wett' is not a property of frame (did you mean 'wet'?)"), std::string::npos);
  EXPECT_NE(error_of(head + "frame puddel 1\n").find("'puddel' is not a property of frame (did you mean 'puddle'?)"), std::string::npos);
  EXPECT_NE(error_of(head + "frame deck 1, deck 0\n").find("'deck' is given twice"), std::string::npos);
  EXPECT_EQ(error_of(head + "effect tiny = \"tiny\"\nmodule deck = tiny, size 32\nmodule foot = tiny, size 32\nlet wet = 1\nlet puddle = 2\n"), "no error")
      << "the keys reserve no word";
  // by default, and at 0, every scene before stays as it was, to the bit
  const auto frame_of = [](const std::string& text) {
    sc::Scene scene = scene_of(text);
    std::vector<std::uint8_t> rgb(160 * 120 * 3);
    for (int f = 0; f < 5; ++f) scene.render(f, rgb);
    return rgb;
  };
  const std::string night = head + "light gain 0.14, flash (0.1, 0.1, 0.1)\n";
  EXPECT_EQ(frame_of(night), frame_of(night + "frame wet 0, puddle 0, deck 0, foot 0\n"));
  EXPECT_EQ(frame_of(night + "frame daylight 1"), frame_of(night + "frame daylight 1, wet 0, puddle 0, deck 0, foot 0\n"));
  {
    sc::Scene scene = scene_of(night);
    std::vector<std::uint8_t> rgb(160 * 120 * 3);
    scene.render(0, rgb);
    EXPECT_EQ(scene.frame().wet, 0.f);
    EXPECT_EQ(scene.frame().puddle, 0.f);
    EXPECT_EQ(scene.frame().deck, 0.f);
    EXPECT_EQ(scene.frame().foot, 0.f);
  }
  // constants and expressions of t reach the frame, within their ranges (shares 0 to 1, the foot 0 to 90 degrees)
  {
    sc::Scene scene = scene_of(head + "frame wet 0.5, puddle 2, deck -1, foot t * 100\n");
    std::vector<std::uint8_t> rgb(160 * 120 * 3);
    scene.render(0, rgb);
    EXPECT_FLOAT_EQ(scene.frame().wet, 0.5f);
    EXPECT_EQ(scene.frame().puddle, 1.f) << "a share, held to 1";
    EXPECT_EQ(scene.frame().deck, 0.f) << "a share, held to 0";
    EXPECT_EQ(scene.frame().foot, 0.f);
    scene.render(6, rgb);
    EXPECT_NEAR(scene.frame().foot, 20.f, 1e-3f) << "re-evaluated every frame";
    scene.render(30, rgb);
    EXPECT_EQ(scene.frame().foot, 90.f) << "100 degrees at t = 1, held to 90";
  }
  // wired to the director, the frame follows its state frame by frame: a storm wets the ground and forms a deck
  {
    sc::Scene scene = scene_of("scene size 160 x 120, fps 30, length 8 s, ground 100\nweather sky, seed 28500280, regime storm\n"
                               "frame wet sky.wet, puddle sky.puddle, deck sky.deck, foot sky.foot\n");
    std::vector<std::uint8_t> rgb(160 * 120 * 3);
    float most_wet = 0.f, most_deck = 0.f;
    for (int f = 0; f < scene.frames(); ++f) {
      scene.render(f, rgb);
      const wx::Director& d = scene.director(0);
      EXPECT_EQ(scene.frame().wet, d.value(wx::m_wet)) << "frame " << f;
      EXPECT_EQ(scene.frame().puddle, d.value(wx::m_puddle)) << "frame " << f;
      EXPECT_EQ(scene.frame().deck, d.value(wx::m_deck)) << "frame " << f;
      EXPECT_EQ(scene.frame().foot, d.value(wx::m_foot)) << "frame " << f;
      most_wet = std::max(most_wet, scene.frame().wet);
      most_deck = std::max(most_deck, scene.frame().deck);
    }
    EXPECT_GT(most_wet, 0.5f) << "eight seconds of a storm's rain darken the ground (half dark at 4.5 s, WEATHER.md §3.12)";
    EXPECT_GT(most_deck, 0.5f) << "a storm's sky is a deck";
  }
}

TEST(WeatherDay, TheHillsFadeIntoTheHazeAsKoschmiederSays) {
  // the hills' contrast against the horizon sky, C = (sky - hills) / sky, is exp(-3.912 x / V) of what it is in clear air
  // for hills x = 2 km away: the ratio between two visibilities is exp(-3.912 x (1 / V1 - 1 / V2))
  const auto contrast = [](float V) {
    sc::Scene scene = scene_of(std::format("scene size 400 x 300, fps 30, length 1 s, ground 250\nlight gain 0\nframe daylight 1, bloom 0, visibility {}\n", V));
    std::vector<std::uint8_t> rgb(400 * 300 * 3);
    scene.render(0, rgb);
    const cm::Image4& s = scene.frame().screen();
    // a column where the hills stand highest, and the sky just above them
    double best = 0, sky = 0, hills = 0;
    for (int x = 0; x < 400; ++x) {
      int top = 250;
      while (top > 200 && s.row(top - 1)[static_cast<std::size_t>(x) * 4 + 2] != s.row(200)[static_cast<std::size_t>(x) * 4 + 2]) --top;
      if (250 - top > best) {
        best = 250 - top;
        hills = s.row(248)[static_cast<std::size_t>(x) * 4 + 2];
        sky = s.row(top - 3)[static_cast<std::size_t>(x) * 4 + 2];
      }
    }
    return (sky - hills) / sky;
  };
  const double c40 = contrast(40000.f), c4 = contrast(4000.f), c2 = contrast(2000.f);
  EXPECT_GT(c40, 0.3);
  EXPECT_NEAR(c4 / c40, std::exp(-3.912 * 2000.0 * (1.0 / 4000.0 - 1.0 / 40000.0)), 0.03);
  EXPECT_NEAR(c2 / c40, std::exp(-3.912 * 2000.0 * (1.0 / 2000.0 - 1.0 / 40000.0)), 0.03);
}

// --- no allocation per frame --------------------------------------------------------------------------------------------

TEST(WeatherAllocation, NoneAFrame) {
  wx::Director d(kSeed, Regime::clear);
  g_allocations = 0;
  g_counting = true;
  for (int i = 0; i < 3000; ++i) {
    if (i == 100) d.request(Regime::storm, 20.f);
    if (i == 1500) d.request(Regime::clear, 20.f);
    d.step(kDt);
  }
  g_counting = false;
  EXPECT_EQ(g_allocations.load(), 0) << "the director";
  for (const bool overlap : {false, true}) {
    sc::Scene scene = scene_of(kWeatherScene, 2, overlap);
    std::vector<std::uint8_t> rgb(200 * 120 * 3);
    scene.render(0, rgb);  // warm-up outside the count
    g_allocations = 0;
    g_counting = true;
    for (int f = 1; f < scene.frames(); ++f) scene.render(f, rgb);
    float v = 0.f;
    (void)scene.weather_value("sky.rain", v);
    g_counting = false;
    EXPECT_EQ(g_allocations.load(), 0) << "a scripted scene with the director and every layer it drives" << (overlap ? ", overlapped" : "");
  }
}

}  // namespace
