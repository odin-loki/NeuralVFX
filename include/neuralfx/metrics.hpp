// Quality metrics on premultiplied RGBA clips (docs/PLAN.md §6), and paired bootstrap intervals.
// All pixel values are taken as v / 255 in [0, 1]; PSNR uses a peak of 1.
#pragma once

#include <neuralfx/clip.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace nfx::metrics {

inline constexpr double kPsnrCap = 99.0;  // reported for identical inputs

double mse(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b);
double psnr_from_mse(double mse);
// Mean SSIM over the four channels (Gaussian window 11, sigma 1.5, the usual constants).
double ssim(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b, int size);

struct ClipScores {
  double psnr = 0;          // from the mean MSE over all frames
  double ssim = 0;          // mean over frames
  double temporal_psnr = 0; // PSNR of frame-to-frame differences (motion error); wraps for looping clips
  double flicker = 0;       // second temporal difference energy, test / reference (1 = same, > 1 = more jitter)
  std::vector<double> frame_psnr, frame_ssim;
};

// Scores of `test` against `ref` (same size and frame count). `frames` restricts scoring to those frame indices
// (for example the held-out odd frames); empty = all.
ClipScores score(const Clip& ref, const Clip& test, std::span<const int> frames = {});

// Distribution statistics of a clip, for comparing generated variations with real clips when pixels cannot match
// (different seeds): where it covers, how bright, how fine its detail, how fast it changes.
struct ClipStats {
  std::vector<double> coverage;   // mean alpha per frame
  std::vector<double> emission;   // mean max(0, rgb - alpha) per frame (light added)
  std::vector<double> spectrum;   // radially averaged log power of luminance (bins 1..size/2), mean over frames
  std::vector<float> mean_frame;  // mean premultiplied RGBA over time
  double motion = 0;              // mean absolute frame-to-frame difference
};
ClipStats stats(const Clip& clip);

struct StatDistance {
  double coverage_l1 = 0;    // mean |coverage curve difference|
  double emission_l1 = 0;
  double spectrum_l1 = 0;    // mean |log power difference| over radial bins
  double mean_frame_psnr = 0;
  double motion_ratio = 0;   // test motion / reference motion
};
StatDistance distance(const ClipStats& ref, const ClipStats& test);

// Paired bootstrap of mean(a - b): resample pairs with replacement. 95% percentile interval.
struct Interval {
  double mean = 0, lo = 0, hi = 0;
  bool covers_zero() const { return lo <= 0 && hi >= 0; }
};
Interval paired_bootstrap(std::span<const double> a, std::span<const double> b, int resamples = 10000,
                          std::uint64_t seed = 1);
Interval bootstrap_mean(std::span<const double> x, int resamples = 10000, std::uint64_t seed = 1);

}  // namespace nfx::metrics
