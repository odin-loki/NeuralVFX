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
frame daylight sky.light, overcast sky.cloud, visibility sky.visibility
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
  const int members[] = {wx::m_cloud, wx::m_rain,      wx::m_snow, wx::m_hail, wx::m_wind,       wx::m_mean,  wx::m_gust,
                         wx::m_temp,  wx::m_lightning, wx::m_dust, wx::m_fog,  wx::m_extinction, wx::m_light, wx::m_lean};
  // (the extinction's bound: rain turning to snow as the cold front passes hides much more, per mm/h, in a second)
  const float bound[] = {0.02f, 2.5f, 0.8f, 0.05f, 0.6f, 0.4f, 0.01f, 0.4f, 3.f, 0.05f, 3.f, 4.f, 0.03f, 0.02f};
  std::array<double, 14> worst30{}, worst60{};
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
std::array<float, 9> halfway(Regime from, Regime to, float over) {
  wx::Director d(kSeed, from);
  const wx::Target a = d.base();
  d.request(to, over);
  const wx::Target b = d.heading();
  std::array<float, 9> at;
  at.fill(std::numeric_limits<float>::infinity());
  run(d, over, [&](const wx::Director& s) {
    const wx::Target& x = s.base();
    const float va[] = {a.cloud, a.precip, a.hail, a.mean, a.gust, a.temp, a.lightning, a.fog, a.dust};
    const float vb[] = {b.cloud, b.precip, b.hail, b.mean, b.gust, b.temp, b.lightning, b.fog, b.dust};
    const float vx[] = {x.cloud, x.precip, x.hail, x.mean, x.gust, x.temp, x.lightning, x.fog, x.dust};
    for (std::size_t k = 0; k < 9; ++k)
      if (std::isinf(at[k]) && va[k] != vb[k] && std::abs(vx[k] - va[k]) >= 0.5f * std::abs(vb[k] - va[k])) at[k] = s.time();
  });
  return at;
}
enum : std::size_t { h_cloud, h_precip, h_hail, h_mean, h_gust, h_temp, h_lightning, h_fog, h_dust };

TEST(WeatherDirector, TransitionsKeepTheOrderOfRealWeather) {
  // building: the clouds, then the gust front (the wind, its gusts and the drop in temperature), then the downpour, and
  // the lightning and the hail with the heaviest rain
  const auto up = halfway(Regime::clear, Regime::storm, 60.f);
  EXPECT_LT(up[h_cloud], up[h_mean]) << "clouds build before the wind comes";
  EXPECT_LT(up[h_mean], up[h_precip]) << "the gust front arrives before the downpour";
  EXPECT_LE(up[h_temp], up[h_precip]) << "the cold air with it";
  EXPECT_LT(up[h_precip], up[h_lightning]) << "lightning comes with the heaviest rain";
  EXPECT_LT(up[h_precip], up[h_hail]);
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
  // fog lifts before the sky clears
  const auto lift = halfway(Regime::fog, Regime::clear, 60.f);
  EXPECT_LT(lift[h_fog], lift[h_cloud]);
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
  const std::string script = "scene size 160 x 90, fps 30, length 60 s, ground 85\neffect tiny = \"tiny\"\ninput regime = 0\ninput change = 4\n"
                             "weather sky, seed 28500220, regime regime, over change\n"
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
  ASSERT_EQ(nvfx_scene_set_input(s, "regime", 3.f), NVFX_OK);  // a storm, in 4 s
  ASSERT_EQ(nvfx_scene_step(s, 20.0, nullptr), NVFX_OK);
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
