// Study R §6.1 (docs/STUDY_R.md): the coupled demo's rule (coupled3d.hpp, shared with `nvfx_3d video --kind coupled`).
// The soot flux off the top of a volume reads the right cells in the volume API's layout (z slowest), counts only the
// gas that rises, and refuses fields too small for its sizes; the flux drive plays at the base until the settle time,
// then follows the smoothed flux against its running mean (above the mean brightens the smoke, below dims it), clamped
// to [0.05, 1]; a field moved into the renderer's layout (y slowest) lands at the renderer's index.
#include <neuralfx/coupled3d.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace c3 = nfx::coupled3d;

namespace {

std::size_t api_index(int x, int y, int z, int nx, int ny) {
  return (static_cast<std::size_t>(z) * static_cast<std::size_t>(ny) + static_cast<std::size_t>(y)) * static_cast<std::size_t>(nx) + static_cast<std::size_t>(x);
}

}  // namespace

TEST(Coupled3d, TopSootFluxReadsTheTopLayersInTheApisLayout) {
  const int nx = 3, ny = 5, nz = 2;
  const std::size_t n = static_cast<std::size_t>(nx * ny * nz);
  std::vector<float> soot(n, 0.f), vel(3 * n, 0.f);
  // Soot and an upward velocity everywhere: the flux over the top `layers` rows is the count of those cells times
  // soot times velocity, whatever the layout, so the layout is tested by the cells that differ.
  for (std::size_t i = 0; i < n; ++i) {
    soot[i] = 2.f;
    vel[3 * i + 1] = 0.5f;
  }
  EXPECT_DOUBLE_EQ(c3::top_soot_flux(soot, vel, nx, ny, nz, 1), 1.0 * nx * nz);
  EXPECT_DOUBLE_EQ(c3::top_soot_flux(soot, vel, nx, ny, nz, 2), 1.0 * nx * nz * 2);
  EXPECT_DOUBLE_EQ(c3::top_soot_flux(soot, vel, nx, ny, nz, ny), 1.0 * n);

  // One cell on the top row stands out: it is counted at layers = 1 wherever it sits in x and z, and a cell one row
  // down is not. If the layout were read y slowest (the renderer's) the top row would land on other texels.
  std::fill(soot.begin(), soot.end(), 0.f);
  soot[api_index(2, ny - 1, 1, nx, ny)] = 4.f;  // top row, far corner
  soot[api_index(0, ny - 2, 0, nx, ny)] = 100.f;  // one row down: not in the top layer
  EXPECT_DOUBLE_EQ(c3::top_soot_flux(soot, vel, nx, ny, nz, 1), 4.0 * 0.5);
  EXPECT_DOUBLE_EQ(c3::top_soot_flux(soot, vel, nx, ny, nz, 2), 4.0 * 0.5 + 100.0 * 0.5);
}

TEST(Coupled3d, TopSootFluxCountsOnlyTheGasThatRises) {
  const int nx = 2, ny = 2, nz = 2;
  const std::size_t n = static_cast<std::size_t>(nx * ny * nz);
  std::vector<float> soot(n, 1.f), vel(3 * n, 0.f);
  // The top row: one cell rises at 3, one sinks at -5, one is still, one rises at 1. Sideways velocity (x, z
  // components) is not flux through the top and is ignored however large.
  vel[3 * api_index(0, 1, 0, nx, ny) + 1] = 3.f;
  vel[3 * api_index(1, 1, 0, nx, ny) + 1] = -5.f;
  vel[3 * api_index(0, 1, 1, nx, ny) + 1] = 0.f;
  vel[3 * api_index(1, 1, 1, nx, ny) + 1] = 1.f;
  for (std::size_t i = 0; i < n; ++i) {
    vel[3 * i + 0] = 50.f;
    vel[3 * i + 2] = -50.f;
  }
  EXPECT_DOUBLE_EQ(c3::top_soot_flux(soot, vel, nx, ny, nz, 1), 4.0);
  // Everything sinking: no flux, not a negative one.
  for (std::size_t i = 0; i < n; ++i) vel[3 * i + 1] = -1.f;
  EXPECT_DOUBLE_EQ(c3::top_soot_flux(soot, vel, nx, ny, nz, ny), 0.0);
}

TEST(Coupled3d, TopSootFluxRefusesBadSizesAndShortFields) {
  const int nx = 2, ny = 3, nz = 2;
  const std::size_t n = static_cast<std::size_t>(nx * ny * nz);
  std::vector<float> soot(n, 1.f), vel(3 * n, 1.f);
  EXPECT_NO_THROW(c3::top_soot_flux(soot, vel, nx, ny, nz, 1));
  EXPECT_NO_THROW(c3::top_soot_flux(soot, vel, nx, ny, nz, ny));
  EXPECT_THROW(c3::top_soot_flux(soot, vel, nx, ny, nz, 0), std::invalid_argument);
  EXPECT_THROW(c3::top_soot_flux(soot, vel, nx, ny, nz, ny + 1), std::invalid_argument);
  EXPECT_THROW(c3::top_soot_flux(soot, vel, 0, ny, nz, 1), std::invalid_argument);
  EXPECT_THROW(c3::top_soot_flux(soot, vel, nx, 0, nz, 1), std::invalid_argument);
  EXPECT_THROW(c3::top_soot_flux(soot, vel, nx, ny, 0, 1), std::invalid_argument);
  std::vector<float> short_soot(n - 1, 1.f), short_vel(3 * n - 1, 1.f);
  EXPECT_THROW(c3::top_soot_flux(short_soot, vel, nx, ny, nz, 1), std::invalid_argument);
  EXPECT_THROW(c3::top_soot_flux(soot, short_vel, nx, ny, nz, 1), std::invalid_argument);
  // Larger fields than the sizes are fine: only the first nx * ny * nz texels are read.
  std::vector<float> long_soot(n + 7, 1.f), long_vel(3 * n + 7, 1.f);
  EXPECT_DOUBLE_EQ(c3::top_soot_flux(long_soot, long_vel, nx, ny, nz, 1), 1.0 * nx * nz);
}

TEST(Coupled3d, FluxDrivePlaysTheBaseUntilItSettles) {
  c3::FluxDrive drive;
  const double fps = 30.0;
  const float base = 0.6f;
  // Before `settle` seconds the intensity is the base whatever the flux does, while the statistics still accumulate.
  for (int f = 0; f < 30; ++f) {
    const double t = f / fps;
    const double flux = (f % 2 == 0) ? 10.0 : 0.0;
    EXPECT_FLOAT_EQ(drive.update(flux, base, fps, t), base) << "frame " << f;
    EXPECT_FLOAT_EQ(drive.intensity, base);
  }
  EXPECT_EQ(drive.frames, 30);
  EXPECT_NEAR(drive.mean, 5.0, 1e-12);  // the running mean of alternating 10 and 0 over 30 frames
  EXPECT_GT(drive.smoothed, 0.0);
  // At the settle time it starts following the flux: a steady flux equal to the mean gives the base again.
  c3::FluxDrive steady;
  for (int f = 0; f <= 60; ++f) steady.update(3.0, base, fps, f / fps);
  EXPECT_NEAR(steady.smoothed, 3.0, 1e-12);
  EXPECT_NEAR(steady.mean, 3.0, 1e-12);
  EXPECT_FLOAT_EQ(steady.intensity, base);
  // A custom settle time: with settle = 0 the first frame already follows (smoothed == mean == flux, so the base).
  c3::FluxDrive quick;
  EXPECT_FLOAT_EQ(quick.update(7.0, 0.4f, fps, 0.0, 0.0), 0.4f);
}

TEST(Coupled3d, FluxDriveFollowsTheSmoothedFluxAgainstItsMean) {
  const double fps = 30.0;
  const float base = 0.5f;
  // The first frame seeds the smoothed value; afterwards it is a first-order filter with the frame rate's time
  // constant, and the mean is the running average. Check the recurrence by hand for a few frames.
  c3::FluxDrive drive;
  drive.update(2.0, base, fps, 0.0);
  EXPECT_DOUBLE_EQ(drive.smoothed, 2.0);
  EXPECT_DOUBLE_EQ(drive.mean, 2.0);
  drive.update(5.0, base, fps, 1 / fps);
  EXPECT_DOUBLE_EQ(drive.smoothed, 2.0 + (5.0 - 2.0) / fps);
  EXPECT_DOUBLE_EQ(drive.mean, 3.5);
  drive.update(5.0, base, fps, 2 / fps);
  const double s2 = (2.0 + 3.0 / fps) + (5.0 - (2.0 + 3.0 / fps)) / fps;
  EXPECT_DOUBLE_EQ(drive.smoothed, s2);
  EXPECT_DOUBLE_EQ(drive.mean, 4.0);
  EXPECT_EQ(drive.frames, 3);

  // A fire that flares after a quiet start brightens its smoke above the base; one that dies down dims it. Two
  // seconds at flux 1 (the mean settles near 1), then two seconds at flux 4: the smoothed flux climbs well above the
  // mean, which rises more slowly, so the intensity rises above the base. Then two seconds at 0: it falls below.
  c3::FluxDrive d;
  int f = 0;
  for (; f < 60; ++f) d.update(1.0, base, fps, f / fps);
  EXPECT_FLOAT_EQ(d.intensity, base);
  float peak = 0.f;
  for (; f < 120; ++f) peak = std::max(peak, d.update(4.0, base, fps, f / fps));
  EXPECT_GT(d.intensity, base);
  EXPECT_GT(peak, base);
  EXPECT_LE(peak, 1.0f);
  // The exact value at the end of the flare: base * smoothed / mean.
  EXPECT_FLOAT_EQ(d.intensity, static_cast<float>(base * d.smoothed / d.mean));
  for (; f < 180; ++f) d.update(0.0, base, fps, f / fps);
  EXPECT_LT(d.intensity, base);
  EXPECT_GE(d.intensity, 0.05f);
  EXPECT_FLOAT_EQ(d.intensity, static_cast<float>(std::max(0.05, base * d.smoothed / d.mean)));
}

TEST(Coupled3d, FluxDriveClampsToItsRange) {
  const double fps = 30.0;
  // A long quiet run (mean near 0) then a huge step: the ratio would blow up, the intensity stops at 1.
  c3::FluxDrive up;
  int f = 0;
  for (; f < 300; ++f) up.update(1e-3, 0.6f, fps, f / fps);
  for (; f < 360; ++f) up.update(1e6, 0.6f, fps, f / fps);
  EXPECT_FLOAT_EQ(up.intensity, 1.0f);
  // A fire that goes out: the smoothed flux decays towards 0 against a mean that stays up, the intensity stops at 0.05.
  c3::FluxDrive down;
  f = 0;
  for (; f < 60; ++f) down.update(10.0, 0.6f, fps, f / fps);
  for (; f < 600; ++f) down.update(0.0, 0.6f, fps, f / fps);
  EXPECT_FLOAT_EQ(down.intensity, 0.05f);
  // A zero flux from the start: the mean is 0, guarded by the 1e-9 floor, and the ratio is 0, so the floor again
  // (not a NaN) once it settles.
  c3::FluxDrive zero;
  for (f = 0; f < 60; ++f) zero.update(0.0, 0.6f, fps, f / fps);
  EXPECT_FLOAT_EQ(zero.intensity, 0.05f);
  // A base out of range is clamped too once following; before settling it is returned as given.
  c3::FluxDrive hot;
  EXPECT_FLOAT_EQ(hot.update(1.0, 1.5f, fps, 0.0), 1.5f);
  EXPECT_FLOAT_EQ(hot.update(1.0, 1.5f, fps, 2.0), 1.0f);
}

TEST(Coupled3d, RowsToRendererMovesEachTexelToTheRenderersIndex) {
  const int nx = 3, ny = 4, nz = 2;
  const std::size_t n = static_cast<std::size_t>(nx * ny * nz);
  std::vector<float> src(n), dst(n, -1.f);
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) src[api_index(x, y, z, nx, ny)] = static_cast<float>(100 * x + 10 * y + z);
    }
  }
  c3::rows_to_renderer(src, nx, ny, nz, dst.data());
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const std::size_t r = (static_cast<std::size_t>(y) * static_cast<std::size_t>(nz) + static_cast<std::size_t>(z)) * static_cast<std::size_t>(nx) + static_cast<std::size_t>(x);
        EXPECT_FLOAT_EQ(dst[r], static_cast<float>(100 * x + 10 * y + z)) << "x " << x << " y " << y << " z " << z;
      }
    }
  }
  // Stacking in y: the upper box written after the lower's cells keeps its own layout and sits above the lower's rows.
  std::vector<float> stacked(2 * n, 0.f);
  c3::rows_to_renderer(src, nx, ny, nz, stacked.data());
  c3::rows_to_renderer(src, nx, ny, nz, stacked.data() + n);
  for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(stacked[n + i], stacked[i]);
  std::vector<float> short_src(n - 1, 0.f);
  EXPECT_THROW(c3::rows_to_renderer(short_src, nx, ny, nz, dst.data()), std::invalid_argument);
}
