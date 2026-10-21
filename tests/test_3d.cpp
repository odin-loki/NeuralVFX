// Study N (docs/STUDY_3D.md): the 3D simulator (deterministic on any number of threads; the plume rises), the volume
// renderer (Beer-Lambert through a uniform slab, the blackbody colour of a thick isothermal slab, the same pictures on any
// number of threads), the 3D rollout model (its format, its reference on any number of threads), the trainer (its
// forward pass is the reference's; its gradients match finite differences through time, with the profile and activity
// losses), and the volume C API (parity with the reference: exact at the baseline ISA, within float rounding with AVX2;
// the same fields and pictures on any number of threads; shards; texel formats; errors; its detail layer on vectors gives
// the reference's fields to the bit on every ISA). Study N2 (§14): ambient air at
// the open sides and top of the coarse advection (what flows in, the reference against the trainer, gradients through
// it with inflow at every side and the top, study N's clamped files unchanged), controls that change during a run (the
// simulator's, the runs' format, gradients with them) and the long training stage (resumes to the bit). Study R
// (docs/STUDY_R.md): the validated smoke's solver (the flux form keeps what it carries; the buoyancy flux is the same up
// the plume; the same fields on any number of threads), warm smoke's optics, runs that keep their effect.
#include <neuralfx/nvfx_volume.h>
#include <neuralfx/render3d.hpp>
#include <neuralfx/rollout3d.hpp>
#include <neuralfx/rollout3d_train.hpp>
#include <neuralfx/sim3d.hpp>

#include <gtest/gtest.h>

#include "rt_volume.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <random>
#include <sstream>
#include <stdexcept>
#include <stdfloat>
#include <vector>

namespace r3 = nfx::rollout3d;
namespace rd = nfx::render3d;

namespace {

std::size_t zu(int v) { return static_cast<std::size_t>(v); }
float fl(int v) { return static_cast<float>(v); }

// --- simulator --------------------------------------------------------------------------------------------------------

nfx::sim3d::Params small_fire(int threads) {
  nfx::sim3d::Params p;
  p.width = 16;
  p.height = 24;
  p.intensity = 0.7f;
  p.wind = 0.6f;
  p.turbulence = 0.6f;
  p.seed = 9'000'990;
  p.threads = threads;
  return p;
}

TEST(Sim3d, SameFieldsOnAnyNumberOfThreads) {
  nfx::sim3d::Fluid a(small_fire(1)), b(small_fire(3));
  for (int f = 0; f < 6; ++f) {
    a.step_frame();
    b.step_frame();
  }
  const auto sa = a.state(), sb = b.state();
  EXPECT_EQ(sa.u, sb.u);
  EXPECT_EQ(sa.v, sb.v);
  EXPECT_EQ(sa.w, sb.w);
  EXPECT_EQ(sa.temp, sb.temp);
  EXPECT_EQ(sa.soot, sb.soot);
  EXPECT_EQ(sa.pressure, sb.pressure);
  double heat = 0;
  for (const float t : sa.temp) heat += t;
  EXPECT_GT(heat, 0.0);
}

TEST(Sim3d, ThePlumeRises) {
  nfx::sim3d::Params p = small_fire(2);
  p.width = 24;
  p.height = 36;
  p.wind = 0.5f;
  nfx::sim3d::Fluid f(p);
  const auto centroid = [&] {
    const auto s = f.state();
    double h = 0, hy = 0;
    for (int y = 0; y < s.ny; ++y) {
      for (int i = 0; i < s.nx * s.nz; ++i) {
        const double t = s.temp[zu(y * s.nx * s.nz + i)];
        h += t;
        hy += t * (y + 0.5);
      }
    }
    return hy / std::max(1e-9, h);
  };
  for (int k = 0; k < 8; ++k) f.step_frame();
  const double early = centroid();
  for (int k = 0; k < 30; ++k) f.step_frame();
  const double late = centroid();
  EXPECT_GT(late, std::max(early + 1.0, 1.5 * early)) << "the heat should rise: centroid " << early << " -> " << late;
  const auto s = f.state();
  double vt = 0, t = 0;
  for (std::size_t i = 0; i < s.v.size(); ++i) {
    vt += s.v[i] * s.temp[i];
    t += s.temp[i];
  }
  EXPECT_GT(vt / t, 1.0) << "hot gas moves up";
  for (const float v : s.temp) ASSERT_TRUE(std::isfinite(v));
}

TEST(Sim3d, StateRoundTrips) {
  nfx::sim3d::Fluid a(small_fire(1));
  for (int f = 0; f < 4; ++f) a.step_frame();
  nfx::sim3d::Fluid b(small_fire(2));
  b.set_state(a.state());
  a.step_frame();
  b.step_frame();
  EXPECT_EQ(a.state().temp, b.state().temp);
  EXPECT_EQ(a.state().v, b.state().v);
}

// --- study R (docs/STUDY_R.md): the validated smoke ---------------------------------------------------------------

nfx::sim3d::Params small_smoke(int threads, int scalars = -1) {
  nfx::sim3d::Params p;
  p.effect = nfx::sim3d::Effect::smoke_r;
  p.width = 32;
  p.height = 48;
  p.intensity = 0.5f;
  p.wind = 0.5f;
  p.turbulence = 0.5f;
  p.seed = 29'099'100;
  p.threads = threads;
  p.scalars = scalars;
  return p;
}

TEST(Sim3d, SmokeRSameFieldsOnAnyNumberOfThreads) {
  nfx::sim3d::Params p1 = small_smoke(1), p3 = small_smoke(3);
  p1.width = p3.width = 16;
  p1.height = p3.height = 24;
  nfx::sim3d::Fluid a(p1), b(p3);
  for (int f = 0; f < 6; ++f) {
    a.step_frame();
    b.step_frame();
  }
  EXPECT_EQ(a.state().temp, b.state().temp);
  EXPECT_EQ(a.state().soot, b.state().soot);
  EXPECT_EQ(a.state().v, b.state().v);
  // and a state restored into another solver goes on the same way (the flux form's sweep order follows the frame)
  nfx::sim3d::Fluid c(p1);
  c.set_state(a.state());
  a.step_frame();
  c.step_frame();
  EXPECT_EQ(a.state().temp, c.state().temp);
  EXPECT_EQ(a.state().soot, c.state().soot);
}

// The flux form neither makes nor loses soot: with the burner far below and nothing reaching the box's sides or top in
// a few frames, a blob of soot carried by a swirling, rising flow keeps its total to float rounding (the MacCormack form
// changes it by about 1e-4 here).
TEST(Sim3d, FluxFormKeepsTheSootItCarries) {
  const auto total_after = [](int scalars) {
    nfx::sim3d::Params p = small_smoke(2, scalars);
    nfx::sim3d::Fluid f(p);
    nfx::sim3d::State s = f.state();
    double before = 0;
    for (int y = 0; y < s.ny; ++y) {
      for (int z = 0; z < s.nz; ++z) {
        for (int x = 0; x < s.nx; ++x) {
          const std::size_t i = (zu(y) * zu(s.nz) + zu(z)) * zu(s.nx) + zu(x);
          const float dx = fl(x) - 15.5f, dy = fl(y) - 30.f, dz = fl(z) - 15.5f;
          s.soot[i] = std::exp(-(dx * dx + dy * dy + dz * dz) / 18.f);
          s.u[i] = 6.f * dz / 8.f;  // a swirl about the vertical axis and a slow rise
          s.w[i] = -6.f * dx / 8.f;
          s.v[i] = 3.f;
          if (y >= 14) before += s.soot[i];
        }
      }
    }
    f.set_state(s);
    for (int k = 0; k < 3; ++k) f.step_frame();
    const auto t = f.state();
    double after = 0;
    for (int y = 14; y < t.ny; ++y) {
      for (int i = 0; i < t.nx * t.nz; ++i) after += t.soot[zu(y * t.nx * t.nz + i)];
    }
    return after / before;
  };
  const double flux = total_after(1), mac = total_after(0);
  EXPECT_NEAR(flux, 1.0, 2e-6) << "the flux form keeps the soot (MacCormack: " << mac << ")";
}

// Study R's physics in small: in still air the smoke's buoyancy flux through a horizontal plane is the same at every
// height of the plume (Morton, Taylor and Turner), to within 10% from 12 to 40 of the box's 48 cells.
TEST(Sim3d, SmokeRKeepsItsBuoyancyFluxUpThePlume) {
  nfx::sim3d::Fluid f(small_smoke(3));
  const int nx = f.nx(), ny = f.ny(), nz = f.nz();
  const std::size_t n = zu(nx) * zu(ny) * zu(nz);
  std::vector<float> T(n), D(n), U(3 * n);
  std::vector<double> F(zu(ny), 0.0);
  const auto k = f.buoyancy_coefficients();
  for (int fr = 0; fr < 300; ++fr) {
    f.step_frame();
    if (fr < 120) continue;
    f.fields(T, D);
    f.velocity(U);
    for (std::size_t i = 0; i < n; ++i) F[i / (zu(nx) * zu(nz))] += U[3 * i + 1] * (k[0] * T[i] - k[1] * D[i]);
  }
  double lo = 1e300, hi = 0;
  for (int y = 12; y <= 40; ++y) {
    lo = std::min(lo, F[zu(y)]);
    hi = std::max(hi, F[zu(y)]);
  }
  EXPECT_GT(lo, 0.0);
  EXPECT_LT(hi / lo, 1.10) << "the buoyancy flux should not change with height: " << lo << " to " << hi;
}

// --- study R2 (docs/STUDY_R2.md): the momentum audit and the velocity in flux form -------------------------------

nfx::sim3d::Params small_smoke_r2(int threads) {
  nfx::sim3d::Params p = small_smoke(threads);
  p.effect = nfx::sim3d::Effect::smoke_r2;
  p.seed = 34'099'100;
  return p;
}

// Sums of a state's v over rows y0..y1.
double rows_v(const nfx::sim3d::State& s, int y0, int y1) {
  double t = 0;
  for (int y = y0; y <= y1; ++y) {
    for (int i = 0; i < s.nx * s.nz; ++i) t += s.v[zu(y * s.nx * s.nz + i)];
  }
  return t;
}

TEST(Sim3d, SmokeR2SameFieldsOnAnyNumberOfThreads) {
  nfx::sim3d::Params p1 = small_smoke_r2(1), p3 = small_smoke_r2(3);
  p1.width = p3.width = 16;
  p1.height = p3.height = 24;
  nfx::sim3d::Fluid a(p1), b(p3);
  for (int f = 0; f < 6; ++f) {
    a.step_frame();
    b.step_frame();
  }
  EXPECT_EQ(a.state().u, b.state().u);
  EXPECT_EQ(a.state().v, b.state().v);
  EXPECT_EQ(a.state().w, b.state().w);
  EXPECT_EQ(a.state().soot, b.state().soot);
  nfx::sim3d::Fluid c(p1);  // a restored state goes on the same way
  c.set_state(a.state());
  a.step_frame();
  c.step_frame();
  EXPECT_EQ(a.state().v, c.state().v);
  EXPECT_EQ(a.state().temp, c.state().temp);
  // and it is not smoke_r: the velocity is carried another way
  nfx::sim3d::Params pr = p1;
  pr.effect = nfx::sim3d::Effect::smoke_r;
  nfx::sim3d::Fluid r(pr);
  nfx::sim3d::Fluid r2(p1);
  for (int f = 0; f < 3; ++f) {
    r.step_frame();
    r2.step_frame();
  }
  EXPECT_NE(r.state().v, r2.state().v);
  nfx::sim3d::Effect e{};
  EXPECT_TRUE(nfx::sim3d::parse_effect("smoke_r2", e));
  EXPECT_EQ(e, nfx::sim3d::Effect::smoke_r2);
  EXPECT_EQ(nfx::sim3d::effect_name(e), "smoke_r2");
}

// The audit only reads: every effect's fields are the same with it as without it, and its terms add up to the slab's
// change of vertical momentum.
TEST(Sim3d, TheMomentumAuditOnlyReadsAndAddsUp) {
  for (const auto effect : {nfx::sim3d::Effect::fire, nfx::sim3d::Effect::smoke, nfx::sim3d::Effect::smoke_r, nfx::sim3d::Effect::smoke_r2}) {
    nfx::sim3d::Params p = small_smoke(2);
    p.effect = effect;
    p.width = 16;
    p.height = 24;
    nfx::sim3d::Fluid plain(p), audited(p);
    for (int f = 0; f < 20; ++f) {
      plain.step_frame();
      audited.step_frame();
    }
    nfx::sim3d::MomentumAudit au;
    au.y_lo = 1;
    au.y_hi = 14;
    const double before = rows_v(audited.state(), au.y_lo, au.y_hi);
    audited.set_audit(&au);
    for (int f = 0; f < 6; ++f) {
      plain.step_frame();
      audited.step_frame();
    }
    audited.set_audit(nullptr);
    const auto a = plain.state(), b = audited.state();
    EXPECT_EQ(a.u, b.u);
    EXPECT_EQ(a.v, b.v);
    EXPECT_EQ(a.temp, b.temp);
    EXPECT_EQ(a.soot, b.soot);
    const double change = rows_v(b, au.y_lo, au.y_hi) - before;
    const double terms = au.sources + au.buoyancy + au.confinement + au.curl + au.damping + au.advection + au.projection;
    const double scale = std::abs(au.buoyancy) + std::abs(au.advection) + std::abs(au.projection) + 1e-6;
    EXPECT_NEAR(terms, change, 1e-4 * scale) << nfx::sim3d::effect_name(effect);
    EXPECT_NEAR(au.seconds, 6.0 / 30.0, 1e-5);
    EXPECT_GT(au.buoyancy, 0.0) << "warm gas in the slab is pushed up: " << nfx::sim3d::effect_name(effect);
    if (effect == nfx::sim3d::Effect::smoke) {
      EXPECT_NE(au.damping, 0.0) << "study N's smoke damps its velocity";
    }
    if (nfx::sim3d::plume_smoke(effect)) {
      EXPECT_EQ(au.damping, 0.0);
    }
  }
}

// The flux form conserves momentum: the advection's change of the slab's vertical momentum is what its faces carried in
// and out (to float rounding), where the semi-Lagrangian step's is not tied to any face.
TEST(Sim3d, FluxFormVelocityConservesMomentum) {
  nfx::sim3d::Params p = small_smoke_r2(2);
  nfx::sim3d::Fluid f(p);
  nfx::sim3d::State s = f.state();
  for (int y = 0; y < s.ny; ++y) {
    for (int z = 0; z < s.nz; ++z) {
      for (int x = 0; x < s.nx; ++x) {
        const std::size_t i = (zu(y) * zu(s.nz) + zu(z)) * zu(s.nx) + zu(x);
        const float dx = fl(x) - 15.5f, dy = fl(y) - 26.f, dz = fl(z) - 15.5f;
        const float g = std::exp(-(dx * dx + dy * dy + dz * dz) / 30.f);
        s.v[i] = 40.f * g;  // a rising, swirling jet in the middle of the box
        s.u[i] = 8.f * dz / 6.f * g;
        s.w[i] = -8.f * dx / 6.f * g;
      }
    }
  }
  f.set_state(s);
  nfx::sim3d::MomentumAudit au;
  au.y_lo = 14;
  au.y_hi = 30;
  f.set_audit(&au);
  for (int k = 0; k < 3; ++k) f.step_frame();
  const double faces = au.flux_bottom - au.flux_top + au.flux_sides;
  ASSERT_GT(std::abs(au.flux_top) + std::abs(au.flux_bottom), 1.0) << "momentum crosses the slab's ends";
  EXPECT_NEAR(au.advection, faces, 2e-5 * (std::abs(au.flux_top) + std::abs(au.flux_bottom))) << "the advection is its faces' fluxes";
}

TEST(Render3d, WarmSmokeDoesNotGlow) {
  const int n = 12;
  rd::Renderer warm(n, n, n, rd::optics_for("smoke3d")), cold(n, n, n, rd::optics_for("smoke3d"));
  std::vector<float> T(zu(n * n * n), 0.6f), T0(T.size(), 0.f), D(T.size(), 0.f);
  for (std::size_t i = 0; i < D.size(); ++i) D[i] = i % 7 == 0 ? 0.4f : 0.05f;
  warm.prepare(T, D);
  cold.prepare(T0, D);
  const auto cam = rd::look_at({0.f, 0.5f, 2.f}, {0.f, 0.5f, 0.f}, {0.f, 1.f, 0.f}, 40.f, 1.f);
  std::vector<std::uint8_t> a(32 * 32 * 4), b(a.size());
  warm.render(cam, 32, 32, a.data(), 32 * 4);
  cold.render(cam, 32, 32, b.data(), 32 * 4);
  EXPECT_EQ(a, b) << "warm smoke's temperature adds no light";
  EXPECT_GT(*std::max_element(a.begin(), a.end()), 10) << "but the soot scatters the key light";
  // the fire's optics are unchanged for every other effect name
  const rd::Optics f = rd::optics_for("fire3d"), d{};
  EXPECT_EQ(f.kelvin0, d.kelvin0);
  EXPECT_EQ(f.kelvin_per_unit, d.kelvin_per_unit);
  EXPECT_EQ(f.flame_extinction, d.flame_extinction);
}

TEST(Rollout3d, RunsKeepTheirEffect) {
  nfx::sim3d::Params p = small_smoke(1);
  p.width = 16;
  p.height = 24;
  r3::Hyper h;
  h.rx = 4;
  h.ry = 6;
  h.rz = 4;
  const r3::Run r = r3::record_run(p, 3, h);
  const auto f = std::filesystem::temp_directory_path() / "nvfx_test_smoke_r.n3run";
  ASSERT_TRUE(r3::save_run(f, r));
  const auto back = r3::load_run(f);
  ASSERT_TRUE(back);
  EXPECT_EQ(back->p.effect, nfx::sim3d::Effect::smoke_r);
  EXPECT_EQ(back->coarse, r.coarse);
  std::filesystem::remove(f);
  nfx::sim3d::Effect e{};
  EXPECT_TRUE(nfx::sim3d::parse_effect("smoke_r", e));
  EXPECT_EQ(e, nfx::sim3d::Effect::smoke_r);
}

TEST(Noise4, CurlFieldIsNearlyDivergenceFree) {
  nfx::Forcing3 f;
  nfx::CurlField3 c;
  c.resize(f, 128.f, 192.f, 128.f);
  c.build(f, 7, 1.3f);
  double div = 0, mag = 0;
  const float e = 0.5f;
  for (int k = 0; k < 200; ++k) {
    const float X = 10.f + 0.53f * fl(k), Y = 20.f + 0.71f * fl(k), Z = 100.f - 0.37f * fl(k);
    const auto px = c.sample(X + e, Y, Z), mx = c.sample(X - e, Y, Z), py = c.sample(X, Y + e, Z), my = c.sample(X, Y - e, Z), pz = c.sample(X, Y, Z + e),
               mz = c.sample(X, Y, Z - e);
    div += std::abs((px[0] - mx[0]) + (py[1] - my[1]) + (pz[2] - mz[2]));
    mag += std::abs(px[0] - mx[0]) + std::abs(py[1] - my[1]) + std::abs(pz[2] - mz[2]);
  }
  EXPECT_LT(div, 0.35 * mag) << "the curl noise should be mostly divergence-free";
  EXPECT_GT(mag, 0.0);
  for (int k = 0; k < 100; ++k) {
    const float v = nfx::value_noise4(0.37f * fl(k), 0.11f * fl(k), -0.23f * fl(k), 0.05f * fl(k), 99);
    ASSERT_GE(v, -1.f);
    ASSERT_LE(v, 1.f);
  }
}

TEST(Noise4, FlickerLatticeIsTheFlickerToTheBit) {
  const nfx::Forcing3 f = nfx::sim3d::forcing(nfx::sim3d::Effect::fire);
  nfx::FlickerLattice lat;
  const int nx = 12, ny = 18, nz = 10;
  const float unit = 128.f / 12.f;
  lat.resize(f, nx, ny, nz, unit);
  nfx::Pool pool(2);
  for (const float t : {0.f, 1.37f, 12.9f}) {
    lat.build(f, 9'000'123, t, &pool);
    for (int y = 0; y < ny; ++y) {
      for (int z = 0; z < nz; ++z) {
        for (int x = 0; x < nx; ++x) {
          const float want = nfx::flicker3(f, 9'000'123, (fl(x) + 0.5f) * unit, (fl(y) + 0.5f) * unit, (fl(z) + 0.5f) * unit, t);
          ASSERT_EQ(std::bit_cast<std::uint32_t>(lat.at(x, y, z)), std::bit_cast<std::uint32_t>(want)) << x << " " << y << " " << z << " t " << t;
        }
      }
    }
  }
}

// --- renderer --------------------------------------------------------------------------------------------------------

rd::Optics bare_optics() {
  rd::Optics o;
  o.albedo = 0.f;
  o.ambient_rgb = {0.f, 0.f, 0.f};
  o.light_rgb = {0.f, 0.f, 0.f};
  o.fire_light = 0.f;
  o.kelvin0 = 300.f;  // cold soot: its glow (a blackbody at 300 K) is far below float precision
  return o;
}

TEST(Render3d, BeerLambertThroughAUniformSlab) {
  rd::Optics o = bare_optics();
  const int n = 8;
  rd::Renderer r(n, n, n, o);
  const std::size_t cells = zu(n * n * n);
  std::vector<float> T(cells, 0.f), D(cells, 0.1f);
  r.prepare(T, D);
  // looking straight through the box (one world unit deep) along -z
  const auto cam = rd::orthographic({0.f, 0.5f, 5.f}, {0.f, 0.5f, 0.f}, {0.f, 1.f, 0.f}, 0.25f, 0.25f);
  std::vector<float> px(16 * 16 * 4);
  r.render_linear(cam, 16, 16, px.data());
  const float expect = 1.f - std::exp(-o.soot_extinction * 0.1f * 1.f);
  for (int i = 0; i < 16 * 16; ++i) {
    ASSERT_NEAR(px[zu(i) * 4 + 3], expect, 1e-4f) << "pixel " << i;
    ASSERT_LT(px[zu(i) * 4], 1e-12f);  // no light: absorption only (cold soot glows by less than 1e-20)
  }
  // twice the soot: the optical depth doubles
  std::ranges::fill(D, 0.2f);
  r.prepare(T, D);
  r.render_linear(cam, 16, 16, px.data());
  EXPECT_NEAR(px[3], 1.f - std::exp(-o.soot_extinction * 0.2f), 1e-4f);
  // an empty volume is exactly transparent
  std::ranges::fill(D, 0.f);
  r.prepare(T, D);
  r.render_linear(cam, 16, 16, px.data());
  for (const float v : px) ASSERT_EQ(v, 0.f);
}

TEST(Render3d, AThickIsothermalSlabShowsTheBlackbodyColour) {
  rd::Optics o = bare_optics();
  o.flame_extinction = 400.f;
  o.flame0 = 0.f;
  o.flame1 = 0.01f;
  const int n = 8;
  rd::Renderer r(n, n, n, o);
  const std::size_t cells = zu(n * n * n);
  const float kelvin = 1500.f, T0 = (kelvin - o.kelvin0) / o.kelvin_per_unit;
  std::vector<float> T(cells, T0), D(cells, 0.f);
  r.prepare(T, D);
  const auto cam = rd::orthographic({0.f, 0.5f, 5.f}, {0.f, 0.5f, 0.f}, {0.f, 1.f, 0.f}, 0.25f, 0.25f);
  std::vector<float> px(8 * 8 * 4);
  r.render_linear(cam, 8, 8, px.data());
  const auto bb = rd::blackbody_rgb(kelvin);
  for (int c = 0; c < 3; ++c) EXPECT_NEAR(px[zu(c)], o.emission * bb[zu(c)], 2e-4f * o.emission * bb[0] + 1e-7f) << "channel " << c;
  EXPECT_NEAR(px[3], 1.f, 1e-6f);
  // a blackbody is red when cool, close to white at 6500 K
  const auto cool = rd::blackbody_rgb(1200.f), white = rd::blackbody_rgb(6500.f);
  EXPECT_GT(cool[0], cool[1]);
  EXPECT_GT(cool[1], cool[2]);
  EXPECT_LT(std::max({white[0], white[1], white[2]}) / std::min({white[0], white[1], white[2]}), 1.15f);
  // luminance at the reference temperature is 1
  const auto ref = rd::blackbody_rgb(1800.f);
  EXPECT_NEAR(0.2126f * ref[0] + 0.7152f * ref[1] + 0.0722f * ref[2], 1.f, 0.02f);
}

std::vector<float> blob(int nx, int ny, int nz, float cx, float cy, float cz, float r, float amp, std::uint64_t seed) {
  std::vector<float> v(zu(nx * ny * nz));
  std::mt19937 rng(static_cast<unsigned>(seed));
  std::uniform_real_distribution<float> u(0.7f, 1.f);
  for (int y = 0; y < ny; ++y) {
    for (int z = 0; z < nz; ++z) {
      for (int x = 0; x < nx; ++x) {
        const float d = std::hypot(fl(x) - cx, fl(y) - cy, fl(z) - cz) / r;
        v[(zu(y) * zu(nz) + zu(z)) * zu(nx) + zu(x)] = d < 1.f ? amp * (1.f - d * d) * u(rng) : 0.f;
      }
    }
  }
  return v;
}

TEST(Render3d, SamePicturesOnAnyNumberOfThreads) {
  rd::Renderer r1(16, 24, 16), r3(16, 24, 16);
  const auto T = blob(16, 24, 16, 8, 6, 8, 6, 1.3f, 1), D = blob(16, 24, 16, 7, 12, 9, 7, 0.2f, 2);
  nfx::Pool pool(3);
  r1.prepare(T, D);
  r3.prepare(T, D, &pool);
  const auto cam = rd::look_at({1.5f, 1.f, 1.8f}, {0.f, 0.6f, 0.f}, {0.f, 1.f, 0.f}, 45.f, 1.25f);
  std::vector<std::uint8_t> a(50 * 40 * 4), b(a.size());
  r1.render(cam, 50, 40, a.data(), 200);
  r3.render(cam, 50, 40, b.data(), 200, &pool);
  EXPECT_EQ(a, b);
  int lit = 0;
  for (std::size_t i = 0; i < a.size(); i += 4) lit += a[i + 3] > 0;
  EXPECT_GT(lit, 100) << "the blob is in view";
  EXPECT_FALSE(rd::invertible(rd::Camera{}));
  EXPECT_TRUE(rd::invertible(cam));
}

// --- the 3D rollout model -----------------------------------------------------------------------------------------------

// vbase: the mean upward flow (0.25: up everywhere; near 0, air also comes down through the top)
std::vector<float> smooth_state(const r3::Hyper& h, float phase, std::uint64_t seed, float vbase = 0.25f) {
  std::vector<float> c(zu(h.cells()) * r3::kPhys);
  std::mt19937 rng(static_cast<unsigned>(seed));
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  const float a = u(rng), b = u(rng);
  for (int y = 0; y < h.ry; ++y) {
    for (int z = 0; z < h.rz; ++z) {
      for (int x = 0; x < h.rx; ++x) {
        float* s = c.data() + ((zu(y) * zu(h.rz) + zu(z)) * zu(h.rx) + zu(x)) * r3::kPhys;
        const float X = (fl(x) + 0.5f) / fl(h.rx), Y = (fl(y) + 0.5f) / fl(h.ry), Z = (fl(z) + 0.5f) / fl(h.rz);
        s[0] = 0.12f * std::sin(6.f * Y + phase + a) * std::cos(3.f * Z);
        s[1] = vbase + 0.15f * std::sin(5.f * X + 2.f * phase + b);
        s[2] = 0.1f * std::cos(4.f * X + 3.f * Y - phase);
        s[3] = std::max(0.f, 1.2f * std::exp(-8.f * ((X - 0.5f) * (X - 0.5f) + (Z - 0.5f) * (Z - 0.5f))) * (1.f - 0.6f * Y) + 0.1f * std::sin(7.f * Y + phase));
        s[4] = std::max(0.f, 0.15f * Y * std::exp(-5.f * ((X - 0.45f) * (X - 0.45f) + (Z - 0.55f) * (Z - 0.55f))) + 0.02f * std::cos(5.f * X - phase));
      }
    }
  }
  return c;
}

// dims: the coarse grid (x, y, z) and the fine cells per coarse cell
r3::Model tiny_model(std::uint64_t seed = 3, int hidden = 8, r3::Edges edges = r3::Edges::ambient, std::array<int, 4> dims = {4, 6, 5, 2}) {
  r3::Hyper h;
  h.rx = dims[0];
  h.ry = dims[1];
  h.rz = dims[2];
  h.fine = dims[3];
  h.hidden = hidden;
  h.memory = 2;
  h.jacobi = 6;
  h.warmup = 3;
  r3::Model m = r3::init_model(h, seed);
  m.edges = edges;
  m.effect = "tiny";
  m.scale = {0.08f, 0.15f, 0.08f, 0.4f, 0.06f};
  m.lo = {-0.6f, -0.6f, -0.6f, 0.f, 0.f};
  m.hi = {0.6f, 0.9f, 0.6f, 2.f, 0.5f};
  const r3::StepLayout L = r3::step_layout(h);
  for (std::size_t i = L.wo; i < L.wo + zu(h.hidden) * zu(h.outputs()); ++i) m.step_w[i] *= 15.f;
  for (int k = 0; k < 2; ++k) {
    r3::StartPoint s;
    s.controls = {0.3f + 0.4f * fl(k), 0.5f, 0.6f};
    s.seed = 9'000'500 + static_cast<std::uint64_t>(k);
    s.time = 2.f + fl(k);
    s.coarse = smooth_state(h, fl(k), 11 + static_cast<std::uint64_t>(k));
    m.starts.push_back(std::move(s));
  }
  return m;
}

TEST(Rollout3d, FinerCoarseStateMatchesTheDetailGrid) {
  // study hres (STUDY_3D_HRES.md): 32 x 48 x 32 coarse with fine 2 still fills 64 x 96 x 64
  r3::Hyper h;
  h.rx = 32;
  h.ry = 48;
  h.rz = 32;
  h.fine = 2;
  EXPECT_EQ(h.fx(), 64);
  EXPECT_EQ(h.fy(), 96);
  EXPECT_EQ(h.fz(), 64);
  EXPECT_EQ(h.cells(), 32 * 48 * 32);
}

TEST(Rollout3d, SaveLoadRoundTripsAsStorageRounds) {
  // study N's files (clamped edges) are version 1, as they were; ambient edges make version 2
  for (const r3::Edges e : {r3::Edges::clamp, r3::Edges::ambient}) {
    const r3::Model m = tiny_model(3, 8, e);
    std::stringstream ss;
    ASSERT_TRUE(r3::save_model(ss, m));
    const std::string bytes = ss.str();
    std::uint32_t version = 0;
    std::memcpy(&version, bytes.data() + 8, 4);
    EXPECT_EQ(version, e == r3::Edges::clamp ? 1u : 2u);
    auto back = r3::load_model(ss);
    ASSERT_TRUE(back) << back.error();
    EXPECT_EQ(back->edges, e);
    std::string bad = bytes;
    if (e == r3::Edges::ambient) {
      bad[12] = 7;  // no such edge mode
      std::stringstream sb(bad);
      EXPECT_FALSE(r3::load_model(sb));
    }
  }
  r3::Model m = tiny_model();
  m.detail.swirl = 0.7f;
  m.noise.flicker_rate = 1.9f;
  std::stringstream ss;
  ASSERT_TRUE(r3::save_model(ss, m));
  auto back = r3::load_model(ss);
  ASSERT_TRUE(back) << back.error();
  r3::quantise_like_storage(m);
  EXPECT_EQ(back->step_w, m.step_w);
  ASSERT_EQ(back->starts.size(), m.starts.size());
  for (std::size_t k = 0; k < m.starts.size(); ++k) {
    EXPECT_EQ(back->starts[k].coarse, m.starts[k].coarse);
    EXPECT_EQ(back->starts[k].controls, m.starts[k].controls);
    EXPECT_EQ(back->starts[k].seed, m.starts[k].seed);
  }
  EXPECT_EQ(back->detail.swirl, 0.7f);
  EXPECT_EQ(back->noise.flicker_rate, 1.9f);
  EXPECT_EQ(back->h.rz, 5);
  EXPECT_EQ(back->scale, m.scale);
  std::stringstream bad("NVFXVOL1garbage");
  EXPECT_FALSE(r3::load_model(bad));
}

TEST(Rollout3d, ReferenceIsTheSameOnAnyNumberOfThreads) {
  const r3::Model m = tiny_model();
  r3::Workspace w1(m), w3(m);
  r3::State a = r3::make_state(m), b = r3::make_state(m);
  nfx::Pool pool(3);
  const std::vector<float> c{0.6f, 0.4f, 0.7f};
  r3::start(m, 1, c, 77, a, w1, nullptr);
  r3::start(m, 1, c, 77, b, w3, &pool);
  for (int f = 0; f < 4; ++f) {
    r3::step(m, a, c, 77, w1, nullptr);
    r3::step(m, b, c, 77, w3, &pool);
  }
  EXPECT_EQ(a.coarse, b.coarse);
  EXPECT_EQ(a.fine_t, b.fine_t);
  EXPECT_EQ(a.fine_d, b.fine_d);
  double heat = 0;
  for (const float v : a.fine_t) {
    ASSERT_TRUE(std::isfinite(v));
    ASSERT_GE(v, 0.f);
    heat += v;
  }
  EXPECT_GT(heat, 0.0);
}

TEST(Rollout3d, DetailLockFollowsTheCoarseState) {
  // with no swirl and no breakup, the lock scales and adds material so that the fine fields' total follows the coarse
  // state's (not exactly: the factors are interpolated between blocks)
  r3::Model m = tiny_model();
  m.detail.swirl = 0.f;
  m.detail.contrast = 0.f;
  const std::vector<float> c{0.5f, 0.5f, 0.5f};
  r3::Workspace ws(m);
  r3::State s = r3::make_state(m);
  r3::start_from(m, m.starts[0].coarse, 1.f, c, 5, s, ws, nullptr, false);
  for (int f = 0; f < 3; ++f) r3::step(m, s, c, 5, ws, nullptr);
  double fine = 0, coarse = 0;
  for (const float v : s.fine_t) fine += v;
  for (int i = 0; i < m.h.cells(); ++i) coarse += s.coarse[zu(i) * zu(m.h.channels()) + 3];
  coarse *= m.h.fine * m.h.fine * m.h.fine;
  EXPECT_NEAR(fine / coarse, 1.0, 0.15) << fine << " against " << coarse;
}

// --- the trainer ---------------------------------------------------------------------------------------------------------

r3::Run tiny_run(const r3::Hyper& h, int frames, float vbase = 0.25f, bool changing = false) {
  r3::Run r;
  r.p.intensity = 0.6f;
  r.p.wind = 0.4f;
  r.p.turbulence = 0.7f;
  r.p.seed = 9'000'777;
  r.frames = frames;
  for (int i = 0; i < frames; ++i) {
    const auto c = smooth_state(h, 0.3f * fl(i), 21, vbase);
    for (const float v : c) r.coarse.push_back(static_cast<std::float16_t>(v));
    if (changing) {  // the wind sweeps, the intensity and turbulence wander
      r.schedule.push_back(0.6f + 0.3f * std::sin(0.7f * fl(i)));
      r.schedule.push_back(std::clamp(0.1f * fl(i), 0.f, 1.f));
      r.schedule.push_back(0.7f - 0.4f * std::cos(0.5f * fl(i)));
    }
  }
  return r;
}

// What flows in at a side: with no learned change (the output layer zero), a uniform state moving at half a cell per
// frame towards +x and up. Ambient edges: the cells at x = 0 take in half a cell of still, empty air (heat, soot, memory
// and velocity halve); clamped edges (study N): they keep their own. The floor is closed either way (it clamps), the far
// side and the interior are untouched.
TEST(Rollout3d, AmbientAirFlowsInAtTheOpenSides) {
  for (const r3::Edges e : {r3::Edges::clamp, r3::Edges::ambient}) {
    r3::Model m = tiny_model(3, 8, e);
    const r3::StepLayout L = r3::step_layout(m.h);
    std::fill(m.step_w.begin() + static_cast<std::ptrdiff_t>(L.wo), m.step_w.begin() + static_cast<std::ptrdiff_t>(L.bo + zu(m.h.outputs())), 0.f);
    const int N = m.h.cells(), C = m.h.channels();
    std::vector<float> in(zu(N) * zu(C)), out(in.size()), flow(zu(N) * 3), p(zu(N), 0.f), noise(zu(N) * r3::kNoise, 0.f);
    for (int j = 0; j < N; ++j) {
      float* s = in.data() + zu(j) * zu(C);
      s[0] = 0.5f;   // u: towards +x
      s[1] = 0.5f;   // v: up
      s[2] = 0.f;
      s[3] = 1.f;    // heat
      s[4] = 0.25f;  // soot
      for (int k = r3::kPhys; k < C; ++k) s[k] = 0.8f;  // memory (tanh(0.8) after the step's tanh)
    }
    r3::Workspace ws(m);
    r3::coarse_step(m, in, noise, std::vector<float>{0.5f, 0.5f, 0.5f}, p, out, flow, ws);
    const float mem = std::tanh(0.8f);
    for (int y = 0; y < m.h.ry; ++y) {
      for (int z = 0; z < m.h.rz; ++z) {
        for (int x = 0; x < m.h.rx; ++x) {
          const float* o = out.data() + ((zu(y) * zu(m.h.rz) + zu(z)) * zu(m.h.rx) + zu(x)) * zu(C);
          const float k = e == r3::Edges::ambient && x == 0 ? 0.5f : 1.f;  // the floor (y = 0) clamps: nothing comes from below it
          ASSERT_NEAR(o[3], 1.f * k, 1e-6f) << "heat at " << x << " " << y << " " << z;
          ASSERT_NEAR(o[4], 0.25f * k, 1e-6f);
          ASSERT_NEAR(o[0], 0.5f * k, 1e-6f) << "ambient air is still";
          ASSERT_NEAR(o[r3::kPhys], mem * k, 1e-6f);
        }
      }
    }
  }
}

// Down through the top and in at every side: the same with the flow reversed (towards -x, -z and down) at the far
// sides and the top (the floor now outflow).
TEST(Rollout3d, AmbientAirFlowsInThroughTheTopAndTheFarSides) {
  r3::Model m = tiny_model(3, 8, r3::Edges::ambient);
  const r3::StepLayout L = r3::step_layout(m.h);
  std::fill(m.step_w.begin() + static_cast<std::ptrdiff_t>(L.wo), m.step_w.begin() + static_cast<std::ptrdiff_t>(L.bo + zu(m.h.outputs())), 0.f);
  const int N = m.h.cells(), C = m.h.channels();
  std::vector<float> in(zu(N) * zu(C), 0.f), out(in.size()), flow(zu(N) * 3), p(zu(N), 0.f), noise(zu(N) * r3::kNoise, 0.f);
  for (int j = 0; j < N; ++j) {
    float* s = in.data() + zu(j) * zu(C);
    s[0] = -0.25f;
    s[1] = -0.25f;
    s[2] = -0.25f;
    s[3] = 1.f;
  }
  r3::Workspace ws(m);
  r3::coarse_step(m, in, noise, std::vector<float>{0.5f, 0.5f, 0.5f}, p, out, flow, ws);
  for (int y = 0; y < m.h.ry; ++y) {
    for (int z = 0; z < m.h.rz; ++z) {
      for (int x = 0; x < m.h.rx; ++x) {
        const float* o = out.data() + ((zu(y) * zu(m.h.rz) + zu(z)) * zu(m.h.rx) + zu(x)) * zu(C);
        const float kx = x == m.h.rx - 1 ? 0.75f : 1.f, ky = y == m.h.ry - 1 ? 0.75f : 1.f, kz = z == m.h.rz - 1 ? 0.75f : 1.f;
        ASSERT_NEAR(o[3], kx * ky * kz, 1e-5f) << x << " " << y << " " << z;
      }
    }
  }
}

void forward_matches(r3::Edges edges, float vbase) {
  const r3::Model m = tiny_model(3, 8, edges);
  const auto r = tiny_run(m.h, 6, vbase);
  const int N = m.h.cells(), C = m.h.channels();
  std::vector<float> in(zu(N) * zu(C), 0.f), out(in.size()), flow(zu(N) * 3), p(zu(N), 0.f), phys(zu(N) * r3::kPhys);
  r.state(2, phys.size(), phys);
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < r3::kPhys; ++k) in[zu(j) * zu(C) + zu(k)] = phys[zu(j) * r3::kPhys + zu(k)];
  }
  r3::Workspace ws(m);
  std::vector<float> noise(zu(N) * r3::kNoise);
  r3::coarse_noise(m, r.p.seed, r.time(2), ws, noise);
  r3::coarse_step(m, in, noise, r.controls(), p, out, flow, ws);
  r.state(3, phys.size(), phys);
  double ref = 0;
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < r3::kPhys; ++k) {
      const double e = out[zu(j) * zu(C) + zu(k)] - phys[zu(j) * r3::kPhys + zu(k)];
      ref += e * e / (static_cast<double>(m.scale[zu(k)]) * m.scale[zu(k)]);
    }
  }
  ref /= N * r3::kPhys;
  const double tr = r3::window_loss(m, r, 2, 1, 0, 0.f, 0.f, 1, nullptr);
  EXPECT_NEAR(tr, ref, 1e-4 * ref + 1e-9);
}

TEST(Rollout3d, TrainerForwardMatchesTheReference) {
  forward_matches(r3::Edges::ambient, 0.25f);
  forward_matches(r3::Edges::ambient, 0.02f);  // air also comes down through the top
  forward_matches(r3::Edges::clamp, 0.25f);    // study N's edges
  forward_matches(r3::Edges::clamp, 0.02f);
}

void gradient_check(int unroll, float profile, float activity, r3::Edges edges = r3::Edges::ambient, float vbase = 0.25f, bool changing = false,
                    r3::ExtraLosses extra = {}) {
  const r3::Model m0 = tiny_model(11 + static_cast<std::uint64_t>(unroll), 4, edges);
  r3::Model m = m0;
  const auto r = tiny_run(m.h, 10, vbase, changing);
  std::vector<float> grad(m.step_w.size(), 0.f);
  const double l0 = r3::window_loss(m, r, 1, unroll, 0, 0.f, profile, 5, &grad, activity, extra);
  ASSERT_GT(l0, 0.0);
  if (extra.spread > 0.f || extra.edge > 0.f) {  // the extra losses are there, and they are not all of it
    const double plain = r3::window_loss(m, r, 1, unroll, 0, 0.f, profile, 5, nullptr, activity);
    EXPECT_GT(l0, plain * (1.0 + 1e-4)) << "the extra losses add to the loss";
  }
  const auto eval = [&] { return r3::window_loss(m, r, 1, unroll, 0, 0.f, profile, 5, nullptr, activity, extra); };
  // as study D's check (tests/test_rollout.cpp): central and one-sided differences at several steps; a parameter passes
  // when one of them matches within 3%; scattered estimates mark a kink (ReLU, clamp, trilinear) and are skipped
  const auto diffs = [&](float& slot) {
    const float keep = slot;
    const double base = eval();
    std::vector<double> out;
    for (const float eps : {1e-2f, 3e-3f, 1e-3f}) {
      slot = keep + eps;
      const double lp = eval();
      slot = keep - eps;
      const double lm = eval();
      slot = keep;
      out.push_back((lp - lm) / (2.0 * static_cast<double>(eps)));
      if (eps < 5e-3f) {
        out.push_back((lp - base) / static_cast<double>(eps));
        out.push_back((base - lm) / static_cast<double>(eps));
      }
    }
    return out;
  };
  const auto near = [](double a, double b) { return std::abs(a - b) <= 3e-2 * std::max(std::abs(a), std::abs(b)) + 2e-5; };
  int checked = 0, bad = 0, live = 0, kinks = 0;
  for (std::size_t i = 0; i < m.step_w.size(); i += std::max<std::size_t>(1, m.step_w.size() / 200)) {
    const std::vector<double> n = diffs(m.step_w[i]);
    const double ana = grad[i];
    if (std::ranges::any_of(n, [&](double v) { return near(v, ana); })) {
      ++checked;
      live += std::abs(n[0]) > 2e-5;
      continue;
    }
    const auto [lo, hi] = std::ranges::minmax(n);
    if (hi - lo > 0.1 * std::max(std::abs(lo), std::abs(hi)) + 1e-6) {
      ++kinks;
      continue;
    }
    ++checked;
    if (++bad <= 6) ADD_FAILURE() << std::format("param {}: analytic {} central {} {} {}", i, ana, n[0], n[1], n[4]);
  }
  EXPECT_GT(checked, 100);
  EXPECT_LE(kinks, (checked + kinks) / 10) << "too many parameters at kinks";
  EXPECT_GT(live, checked / 2) << "most gradients are zero: the check would be vacuous";
}

TEST(Rollout3d, GradientsMatchFiniteDifferencesOneStep) { gradient_check(1, 0.f, 0.f); }
TEST(Rollout3d, GradientsMatchFiniteDifferencesThroughTime) { gradient_check(4, 0.f, 0.f); }
TEST(Rollout3d, GradientsMatchFiniteDifferencesWithProfilesAndActivity) { gradient_check(3, 1.f, 50.f); }
// air flowing in through every side and the top (ambient edges), study N's clamped edges, and controls that change
TEST(Rollout3d, GradientsMatchFiniteDifferencesWithInflowEverywhere) { gradient_check(3, 1.f, 50.f, r3::Edges::ambient, 0.02f); }
TEST(Rollout3d, GradientsMatchFiniteDifferencesWithClampedEdges) { gradient_check(3, 1.f, 50.f, r3::Edges::clamp, 0.02f); }
TEST(Rollout3d, GradientsMatchFiniteDifferencesWithChangingControls) { gradient_check(3, 1.f, 50.f, r3::Edges::ambient, 0.02f, true); }
// study R2's losses: the plume's spread at each height and the heat and soot at the open sides and top
TEST(Rollout3d, GradientsMatchFiniteDifferencesWithTheSpreadLoss) { gradient_check(3, 1.f, 0.f, r3::Edges::ambient, 0.25f, false, {4.f, 0.f}); }
TEST(Rollout3d, GradientsMatchFiniteDifferencesWithTheEdgeLoss) { gradient_check(3, 0.f, 0.f, r3::Edges::ambient, 0.02f, false, {0.f, 20.f}); }
TEST(Rollout3d, GradientsMatchFiniteDifferencesWithEveryLoss) { gradient_check(3, 1.f, 50.f, r3::Edges::ambient, 0.02f, true, {4.f, 20.f}); }

// The spread of a row: two equal lumps two cells apart in x spread 1 cell^2 about their centroid, wherever the pair sits.
TEST(Rollout3d, RowSpreadIsAboutTheRowsOwnCentroid) {
  r3::Hyper h;
  h.rx = 6;
  h.ry = 3;
  h.rz = 5;
  const int C = 5;
  std::vector<float> st(zu(h.cells()) * C, 0.f);
  const auto at = [&](int x, int y, int z) { return ((zu(y) * zu(h.rz) + zu(z)) * zu(h.rx) + zu(x)) * C + 4; };
  st[at(1, 0, 2)] = 0.5f;
  st[at(3, 0, 2)] = 0.5f;
  st[at(2, 1, 0)] = 0.2f;
  st[at(4, 1, 4)] = 0.2f;  // two lumps 2 apart in x and 4 in z: spread (1 + 4) = 5
  const auto w = r3::row_spread(h, st, C, 4);
  ASSERT_EQ(w.size(), 3u);
  EXPECT_NEAR(w[0], 1.0, 1e-9);
  EXPECT_NEAR(w[1], 5.0, 1e-6);
  EXPECT_EQ(w[2], 0.0);
}

TEST(Rollout3d, RunsAndCheckpointsRoundTrip) {
  const r3::Model m = tiny_model();
  {  // a run whose controls change keeps its schedule
    r3::Run rs = tiny_run(m.h, 5, 0.25f, true);
    const auto dir = std::filesystem::temp_directory_path() / std::format("nvfx3d_sched_{}", ::testing::UnitTest::GetInstance()->random_seed());
    std::filesystem::create_directories(dir);
    ASSERT_TRUE(r3::save_run(dir / "s.n3run", rs));
    auto back = r3::load_run(dir / "s.n3run");
    ASSERT_TRUE(back) << back.error();
    EXPECT_EQ(back->schedule, rs.schedule);
    EXPECT_EQ(back->controls_at(3), rs.controls_at(3));
    EXPECT_NE(back->controls_at(0), back->controls_at(4));
    std::filesystem::remove_all(dir);
  }
  r3::Run r = tiny_run(m.h, 5);
  r.p.width = 8;
  r.p.height = 12;
  const auto dir = std::filesystem::temp_directory_path() / std::format("nvfx3d_test_{}", ::testing::UnitTest::GetInstance()->random_seed());
  std::filesystem::create_directories(dir);
  ASSERT_TRUE(r3::save_run(dir / "a.n3run", r));
  auto back = r3::load_run(dir / "a.n3run");
  ASSERT_TRUE(back) << back.error();
  EXPECT_EQ(back->frames, 5);
  EXPECT_EQ(back->p.seed, r.p.seed);
  EXPECT_EQ(back->coarse.size(), r.coarse.size());
  EXPECT_TRUE(std::ranges::equal(back->coarse, r.coarse, [](std::float16_t a, std::float16_t b) { return std::bit_cast<std::uint16_t>(a) == std::bit_cast<std::uint16_t>(b); }));
  r3::TrainState s;
  s.iteration = 12;
  s.adam_t = 12;
  s.m1.assign(m.step_w.size(), 0.25f);
  s.m2.assign(m.step_w.size(), 0.5f);
  s.curve = {{0, 1.5}, {10, 0.75}};
  ASSERT_TRUE(r3::save_checkpoint(dir / "c.ckpt", m, s));
  r3::Model m2 = tiny_model(99);
  r3::TrainState s2;
  ASSERT_TRUE(r3::load_checkpoint(dir / "c.ckpt", m2, s2));
  EXPECT_EQ(m2.step_w, m.step_w);
  EXPECT_EQ(s2.iteration, 12);
  EXPECT_EQ(s2.m2, s.m2);
  EXPECT_EQ(s2.curve, s.curve);
  std::filesystem::remove_all(dir);
}

TEST(Rollout3d, TrainingResumesAsIfItNeverStopped) {
  r3::Model a = tiny_model(5, 8);
  const std::vector<r3::Run> runs{tiny_run(a.h, 14), tiny_run(a.h, 14)};
  r3::set_normalisation(a, runs);
  r3::Model b = a;
  r3::TrainOptions o;
  o.iterations = 4;
  o.finetune = 2;
  o.activity_stage = 2;
  o.batch = 3;
  o.max_unroll = 3;
  o.burn_max = 2;
  o.threads = 2;
  r3::TrainState sa, sb;
  r3::train_stepper(a, runs, o, sa);
  // b: stop at 5 (a checkpoint), then resume on another thread count
  r3::Model ck;
  r3::TrainState cs;
  {
    r3::TrainOptions ob = o;
    ob.checkpoint_every = 5;
    bool stopped = false;
    r3::Model run_b = b;
    r3::TrainState st;
    try {
      r3::train_stepper(run_b, runs, ob, st, [&](const r3::Model& mm, const r3::TrainState& ss) {
        if (!stopped && ss.iteration == 5) {
          ck = mm;
          cs = ss;
          stopped = true;
          throw std::runtime_error("stop");
        }
      });
    } catch (const std::runtime_error&) {
    }
    ASSERT_TRUE(stopped);
  }
  r3::TrainOptions oc = o;
  oc.threads = 1;
  r3::train_stepper(ck, runs, oc, cs);
  EXPECT_EQ(ck.step_w, a.step_w);
  EXPECT_EQ(cs.iteration, sa.iteration);
}

// Study N2's schedule: runs of two lengths (windows drawn by frames), one with controls that change, and a fourth stage
// of long burn-ins; stopped in stage 4 and resumed on another thread count, it ends as a run that never stopped. A
// snapshot is taken at every stage's end.
TEST(Rollout3d, TrainingWithALongStageResumesAsIfItNeverStopped) {
  r3::Model a = tiny_model(6, 8);
  const std::vector<r3::Run> runs{tiny_run(a.h, 14), tiny_run(a.h, 22, 0.1f, true), tiny_run(a.h, 9)};
  r3::set_normalisation(a, runs);
  r3::Model b = a;
  r3::TrainOptions o;
  o.iterations = 3;
  o.finetune = 2;
  o.activity_stage = 2;
  o.long_stage = 3;
  o.burn_long_min = 6;
  o.burn_long_max = 12;
  o.batch = 3;
  o.max_unroll = 3;
  o.burn_max = 2;
  o.threads = 2;
  EXPECT_EQ(r3::train_stage(o, 6), 3);
  EXPECT_EQ(r3::train_stage(o, 7), 4);
  EXPECT_EQ(r3::stage_end(o, 4), 10);
  r3::TrainState sa;
  std::vector<int> snaps;
  r3::train_stepper(a, runs, o, sa, [&](const r3::Model&, const r3::TrainState& ss) { snaps.push_back(ss.iteration); });
  EXPECT_EQ(snaps, (std::vector<int>{3, 5, 7, 10}));
  r3::Model ck;
  r3::TrainState cs;
  {
    r3::TrainOptions ob = o;
    ob.stop_at = 8;
    r3::train_stepper(b, runs, ob, cs);
    ck = b;
  }
  EXPECT_EQ(cs.iteration, 8);
  r3::TrainOptions oc = o;
  oc.threads = 1;
  r3::train_stepper(ck, runs, oc, cs);
  EXPECT_EQ(ck.step_w, a.step_w);
  EXPECT_EQ(cs.iteration, 10);
  for (const float w : a.step_w) ASSERT_TRUE(std::isfinite(w));
}

// The simulator's controls changed during a run (study N2's training runs): a schedule that holds the controls gives the
// fixed run to the bit, one that moves the wind does not.
TEST(Sim3d, ASteadyScheduleIsTheFixedRun) {
  r3::Hyper h;
  h.rx = 4;
  h.ry = 6;
  h.rz = 4;
  h.fine = 2;
  nfx::sim3d::Params p = small_fire(2);
  p.width = 8;
  p.height = 12;
  const int frames = 5;
  const r3::Run fixed = r3::record_run(p, frames, h);
  std::vector<float> hold, sweep;
  for (int i = 0; i < frames; ++i) {
    for (const float c : {p.intensity, p.wind, p.turbulence}) hold.push_back(c);
    for (const float c : {p.intensity, 0.1f * fl(i), p.turbulence}) sweep.push_back(c);
  }
  const r3::Run held = r3::record_run(p, frames, h, hold), swept = r3::record_run(p, frames, h, sweep);
  const auto bits = [](const r3::Run& r) {
    std::vector<std::uint16_t> v;
    for (const auto x : r.coarse) v.push_back(std::bit_cast<std::uint16_t>(x));
    return v;
  };
  EXPECT_EQ(bits(fixed), bits(held));
  EXPECT_NE(bits(fixed), bits(swept));
  EXPECT_EQ(swept.controls_at(3)[1], 0.3f);
  EXPECT_THROW(r3::record_run(p, frames, h, std::vector<float>(3, 0.5f)), std::invalid_argument);
}

// --- the volume C API ------------------------------------------------------------------------------------------------

struct Loaded {
  nvfx_volume_effect* e = nullptr;
  explicit Loaded(const r3::Model& m) {
    std::stringstream ss;
    EXPECT_TRUE(r3::save_model(ss, m));
    const std::string bytes = ss.str();
    EXPECT_EQ(nvfx_volume_effect_load_memory(bytes.data(), bytes.size(), &e), NVFX_OK);
  }
  ~Loaded() { nvfx_volume_effect_free(e); }
};

// The reference's fine heat after `frames` steps from start point `index` (continuous, as the runtime with shard 0).
std::vector<float> reference_heat(const r3::Model& stored, int index, const std::vector<float>& controls, std::uint64_t seed, int frames, std::vector<float>* soot) {
  r3::Model m = stored;
  r3::quantise_like_storage(m);
  r3::Workspace ws(m);
  r3::State s = r3::make_state(m);
  r3::start(m, index, controls, seed, s, ws, nullptr);
  for (int f = 0; f < frames; ++f) r3::step(m, s, controls, seed, ws, nullptr);
  // to the API's layout: x, then y, then z
  const int FX = m.h.fx(), FY = m.h.fy(), FZ = m.h.fz();
  std::vector<float> out(s.fine_t.size());
  if (soot) soot->resize(out.size());
  for (int z = 0; z < FZ; ++z) {
    for (int y = 0; y < FY; ++y) {
      for (int x = 0; x < FX; ++x) {
        out[(zu(z) * zu(FY) + zu(y)) * zu(FX) + zu(x)] = s.fine_t[(zu(y) * zu(FZ) + zu(z)) * zu(FX) + zu(x)];
        if (soot) (*soot)[(zu(z) * zu(FY) + zu(y)) * zu(FX) + zu(x)] = s.fine_d[(zu(y) * zu(FZ) + zu(z)) * zu(FX) + zu(x)];
      }
    }
  }
  return out;
}

std::vector<float> play(const nvfx_volume_effect* e, int threads, nvfx_isa isa, int frames, std::uint64_t seed, int variation, float shard,
                        std::vector<std::uint8_t>* picture = nullptr) {
  EXPECT_EQ(nvfx_set_isa(isa), NVFX_OK);
  nvfx_volume* v = nullptr;
  EXPECT_EQ(nvfx_volume_create(e, threads, &v), NVFX_OK);
  nvfx_set_isa(NVFX_ISA_AUTO);
  const float c[3] = {0.6f, 0.4f, 0.7f};
  nvfx_volume_set_controls(v, c, 3);
  nvfx_volume_set_seed(v, seed);
  EXPECT_EQ(nvfx_volume_set_variation(v, variation), NVFX_OK);
  EXPECT_EQ(nvfx_volume_set_shard(v, shard), NVFX_OK);
  nvfx_volume_info info{};
  nvfx_volume_effect_get_info(e, &info);
  EXPECT_EQ(nvfx_volume_step_to(v, (frames + 0.5) / info.fps), NVFX_OK);
  std::vector<float> t(zu(info.size[0] * info.size[1] * info.size[2]));
  EXPECT_EQ(nvfx_volume_read(v, NVFX_VOLUME_TEMPERATURE, NVFX_VOLUME_FLOAT, t.data(), t.size() * 4), NVFX_OK);
  if (picture) {
    nvfx_volume_look_at la{{1.4f, 1.1f, 1.6f}, {0.f, 0.7f, 0.f}, {0.f, 1.f, 0.f}, 50.f, 1.5f, 0.f, 0.f};
    nvfx_volume_camera cam{};
    EXPECT_EQ(nvfx_volume_camera_look_at(&la, &cam), NVFX_OK);
    picture->assign(48 * 32 * 4, 0);
    EXPECT_EQ(nvfx_volume_render(v, &cam, 48, 32, picture->data(), 48 * 4), NVFX_OK);
  }
  nvfx_volume_free(v);
  return t;
}

TEST(VolumeApi, MatchesTheReferenceExactlyAtTheBaselineIsa) {
  for (const r3::Edges e : {r3::Edges::ambient, r3::Edges::clamp}) {  // study N2's edges and study N's
    const r3::Model m = tiny_model(3, 8, e);
    Loaded l(m);
    const std::vector<float> c{0.6f, 0.4f, 0.7f};
    std::vector<float> soot;
    const auto ref = reference_heat(m, 1, c, 42, 5, &soot);
    const auto got = play(l.e, 1, NVFX_ISA_BASELINE, 5, 42, 1, 0.f);
    EXPECT_EQ(got, ref);
  }
  // the two edges play differently
  const auto amb = play(Loaded(tiny_model(3, 8, r3::Edges::ambient)).e, 1, NVFX_ISA_BASELINE, 5, 42, 1, 0.f);
  const auto clp = play(Loaded(tiny_model(3, 8, r3::Edges::clamp)).e, 1, NVFX_ISA_BASELINE, 5, 42, 1, 0.f);
  EXPECT_NE(amb, clp);
}

TEST(VolumeApi, MatchesTheReferenceWithAvx2WithinRounding) {
  if (nvfx_set_isa(NVFX_ISA_AVX2) != NVFX_OK) GTEST_SKIP() << "no AVX2";
  nvfx_set_isa(NVFX_ISA_AUTO);
  const r3::Model m = tiny_model();
  Loaded l(m);
  const std::vector<float> c{0.6f, 0.4f, 0.7f};
  const auto ref = reference_heat(m, 1, c, 42, 5, nullptr);
  const auto got = play(l.e, 2, NVFX_ISA_AVX2, 5, 42, 1, 0.f);
  ASSERT_EQ(got.size(), ref.size());
  double worst = 0, peak = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    worst = std::max(worst, static_cast<double>(std::abs(got[i] - ref[i])));
    peak = std::max(peak, static_cast<double>(ref[i]));
  }
  EXPECT_LT(worst, 1e-3 * peak) << "worst difference " << worst << " of a peak of " << peak;
}

TEST(VolumeApi, SameFieldsAndPicturesOnAnyNumberOfThreads) {
  const r3::Model m = tiny_model();
  Loaded l(m);
  std::vector<std::uint8_t> p1, p3;
  // shards of 1.5 s (45 frames) with a 15-frame fade: frame 35 shows two rollouts crossfaded
  const auto a = play(l.e, 1, NVFX_ISA_AUTO, 35, 7, -1, 1.5f, &p1);
  const auto b = play(l.e, 3, NVFX_ISA_AUTO, 35, 7, -1, 1.5f, &p3);
  EXPECT_EQ(a, b);
  EXPECT_EQ(p1, p3);
  int lit = 0;
  for (std::size_t i = 3; i < p1.size(); i += 4) lit += p1[i] > 0;
  EXPECT_GT(lit, 50);
  for (const float v : a) ASSERT_TRUE(std::isfinite(v));
}

TEST(VolumeApi, StepsAreAFunctionOfTime) {
  const r3::Model m = tiny_model();
  Loaded l(m);
  nvfx_volume* v = nullptr;
  ASSERT_EQ(nvfx_volume_create(l.e, 2, &v), NVFX_OK);
  nvfx_volume_set_seed(v, 3);
  ASSERT_EQ(nvfx_volume_set_shard(v, 1.5f), NVFX_OK);
  nvfx_volume_info info{};
  nvfx_volume_effect_get_info(l.e, &info);
  const std::size_t n = zu(info.size[0] * info.size[1] * info.size[2]);
  std::vector<float> a(n), b(n), vel(3 * n);
  std::vector<std::uint16_t> half(n);
  ASSERT_EQ(nvfx_volume_step_to(v, 1.4), NVFX_OK);  // across a shard change
  ASSERT_EQ(nvfx_volume_read(v, NVFX_VOLUME_SOOT, NVFX_VOLUME_FLOAT, a.data(), n * 4), NVFX_OK);
  ASSERT_EQ(nvfx_volume_read(v, NVFX_VOLUME_SOOT, NVFX_VOLUME_HALF, half.data(), n * 2), NVFX_OK);
  for (std::size_t i = 0; i < n; ++i) ASSERT_EQ(half[i], std::bit_cast<std::uint16_t>(static_cast<std::float16_t>(a[i])));
  ASSERT_EQ(nvfx_volume_step_to(v, 0.2), NVFX_OK);  // back in time: restarts
  ASSERT_EQ(nvfx_volume_step_to(v, 1.4), NVFX_OK);
  ASSERT_EQ(nvfx_volume_read(v, NVFX_VOLUME_SOOT, NVFX_VOLUME_FLOAT, b.data(), n * 4), NVFX_OK);
  EXPECT_EQ(a, b);
  ASSERT_EQ(nvfx_volume_read(v, NVFX_VOLUME_VELOCITY, NVFX_VOLUME_FLOAT, vel.data(), vel.size() * 4), NVFX_OK);
  ASSERT_EQ(nvfx_volume_read(v, NVFX_VOLUME_EXTINCTION, NVFX_VOLUME_FLOAT, b.data(), n * 4), NVFX_OK);
  for (std::size_t i = 0; i < n; ++i) ASSERT_GE(b[i], 0.f);
  // errors
  EXPECT_EQ(nvfx_volume_read(v, NVFX_VOLUME_VELOCITY, NVFX_VOLUME_FLOAT, vel.data(), n * 4), NVFX_ERROR_ARGUMENT);  // too small
  EXPECT_EQ(nvfx_volume_step_to(v, -1.0), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_volume_set_variation(v, 5), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_volume_set_shard(v, -2.f), NVFX_ERROR_ARGUMENT);
  nvfx_volume_camera bad{};
  std::vector<std::uint8_t> px(16 * 16 * 4);
  EXPECT_EQ(nvfx_volume_render(v, &bad, 16, 16, px.data(), 64), NVFX_ERROR_ARGUMENT);
  nvfx_volume_free(v);
  nvfx_volume_effect* e = nullptr;
  EXPECT_EQ(nvfx_volume_effect_load("/nonexistent.nvfx3", &e), NVFX_ERROR_IO);
  EXPECT_EQ(nvfx_volume_effect_load_memory("NVFXROL1", 8, &e), NVFX_ERROR_FORMAT);
  EXPECT_STREQ(nvfx_volume_effect_control_name(l.e, 1), "wind");
  EXPECT_EQ(nvfx_volume_effect_control_name(l.e, 3), nullptr);
  EXPECT_EQ(info.coarse[2], 5);
  EXPECT_FLOAT_EQ(info.extent[1], 1.5f);
  uint32_t features = 0;
  nvfx_version(&features);
  EXPECT_TRUE(features & NVFX_FEATURE_VOLUME);
}

// The volume runtime's detail layer on vectors (src/runtime/rt_detail3.hpp) gives rollout3d::detail_step's fields to the
// bit, on every vector ISA the CPU has: through the warm-up and step by step, on 1 and 3 threads, with the swirl and the
// breakup on or off, with either edge mode, on fine rows of whole vectors, of vectors and a remainder, and shorter than
// a vector; from a state the reference made; and for two runs of different seeds stepped in turn through one kernel and
// workspace (as a volume's crossfading shards: its noise caches are per seed).
TEST(VolumeDetail, VectorsGiveTheReferenceToTheBit) {
  struct Case {
    std::array<int, 4> dims;
    float swirl, contrast;
    r3::Edges edges;
  };
  const Case cases[] = {{{4, 6, 5, 2}, 0.5f, 1.f, r3::Edges::ambient}, {{4, 6, 5, 4}, 0.5f, 1.f, r3::Edges::ambient},
                        {{5, 6, 4, 4}, 1.5f, 0.5f, r3::Edges::clamp},  {{4, 6, 5, 4}, 0.f, 1.f, r3::Edges::ambient},
                        {{4, 5, 4, 4}, 0.5f, 0.f, r3::Edges::ambient}};
  const std::vector<float> c{0.6f, 0.4f, 0.7f};
  const auto bits = [](const r3::State& s) {
    std::vector<std::uint32_t> v;
    for (const auto* f : {&s.fine_t, &s.fine_d, &s.coarse}) {
      for (const float x : *f) v.push_back(std::bit_cast<std::uint32_t>(x));
    }
    return v;
  };
  int isas = 0;
  for (const nvfx_isa isa : {NVFX_ISA_AVX2, NVFX_ISA_AVX512}) {
    if (nvfx_set_isa(isa) != NVFX_OK) continue;  // not on this CPU
    nvfx_set_isa(NVFX_ISA_AUTO);
    ++isas;
    for (const Case& k : cases) {
      r3::Model m = tiny_model(3, 8, k.edges, k.dims);
      m.detail.swirl = k.swirl;
      m.detail.contrast = k.contrast;
      for (const int threads : {1, 3}) {
        nfx::Pool pool(threads);
        nfx::Pool* p = threads > 1 ? &pool : nullptr;
        r3::Workspace wa(m), wb(m);
        const auto kernel = nfx::rt::make_volume_detail(m, wb, isa);
        ASSERT_TRUE(kernel);
        wb.detail = kernel.get();
        r3::State a = r3::make_state(m), b = r3::make_state(m);
        r3::start(m, 1, c, 42, a, wa, p);
        r3::start(m, 1, c, 42, b, wb, p);
        ASSERT_EQ(bits(a), bits(b)) << "after the warm-up, isa " << isa << " dims " << k.dims[0] << k.dims[3] << " threads " << threads;
        for (int f = 0; f < 8; ++f) {
          r3::step(m, a, c, 42, wa, p);
          r3::step(m, b, c, 42, wb, p);
          ASSERT_EQ(bits(a), bits(b)) << "frame " << f << ", isa " << isa << " dims " << k.dims[0] << k.dims[3] << " threads " << threads;
        }
        // from the reference's state, and two seeds in turn
        r3::State a2 = r3::make_state(m), b2 = r3::make_state(m);
        r3::start(m, 0, c, 43, a2, wa, p);
        b = a;
        b2 = a2;
        for (int f = 0; f < 4; ++f) {
          r3::step(m, a, c, 42, wa, p);
          r3::step(m, a2, c, 43, wa, p);
          r3::step(m, b, c, 42, wb, p);
          r3::step(m, b2, c, 43, wb, p);
          ASSERT_EQ(bits(a), bits(b)) << "seed 42, frame " << f;
          ASSERT_EQ(bits(a2), bits(b2)) << "seed 43, frame " << f;
        }
        EXPECT_NE(bits(a), bits(a2));
      }
    }
  }
  if (isas == 0) GTEST_SKIP() << "no AVX2";
}

}  // namespace
