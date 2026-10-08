#include <neuralfx/metrics.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <numeric>
#include <random>
#include <ranges>
#include <stdexcept>

namespace nfx::metrics {

namespace {

constexpr double kInv255 = 1.0 / 255.0;

std::vector<int> all_frames(int n) { return std::views::iota(0, n) | std::ranges::to<std::vector>(); }

// Separable Gaussian blur of a size x size plane (clamped edges).
void blur(std::span<const double> in, std::span<double> out, std::span<double> tmp, int size, std::span<const double> k) {
  const int r = static_cast<int>(k.size() / 2);
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      double s = 0;
      for (int i = -r; i <= r; ++i) s += k[static_cast<std::size_t>(i + r)] * in[static_cast<std::size_t>(y * size + std::clamp(x + i, 0, size - 1))];
      tmp[static_cast<std::size_t>(y * size + x)] = s;
    }
  }
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      double s = 0;
      for (int i = -r; i <= r; ++i) s += k[static_cast<std::size_t>(i + r)] * tmp[static_cast<std::size_t>(std::clamp(y + i, 0, size - 1) * size + x)];
      out[static_cast<std::size_t>(y * size + x)] = s;
    }
  }
}

std::vector<double> gaussian(int radius, double sigma) {
  std::vector<double> k(static_cast<std::size_t>(2 * radius + 1));
  for (int i = -radius; i <= radius; ++i) k[static_cast<std::size_t>(i + radius)] = std::exp(-0.5 * i * i / (sigma * sigma));
  const double s = std::accumulate(k.begin(), k.end(), 0.0);
  for (double& v : k) v /= s;
  return k;
}

// In-place radix-2 FFT (n a power of two).
void fft(std::span<std::complex<double>> a) {
  const std::size_t n = a.size();
  for (std::size_t i = 1, j = 0; i < n; ++i) {
    std::size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (std::size_t len = 2; len <= n; len <<= 1) {
    const double ang = -2 * std::numbers::pi / static_cast<double>(len);
    const std::complex<double> wl(std::cos(ang), std::sin(ang));
    for (std::size_t i = 0; i < n; i += len) {
      std::complex<double> w(1);
      for (std::size_t j = 0; j < len / 2; ++j) {
        const auto u = a[i + j], v = a[i + j + len / 2] * w;
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
        w *= wl;
      }
    }
  }
}

}  // namespace

double mse(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
  if (a.size() != b.size() || a.empty()) throw std::invalid_argument("mse: size mismatch");
  double s = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (static_cast<double>(a[i]) - static_cast<double>(b[i])) * kInv255;
    s += d * d;
  }
  return s / static_cast<double>(a.size());
}

double psnr_from_mse(double m) { return m <= 0 ? kPsnrCap : std::min(kPsnrCap, -10.0 * std::log10(m)); }

double ssim(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b, int size) {
  const std::size_t n = static_cast<std::size_t>(size) * size;
  if (a.size() != n * 4 || b.size() != n * 4) throw std::invalid_argument("ssim: size mismatch");
  static const std::vector<double> k = gaussian(5, 1.5);
  constexpr double c1 = 0.01 * 0.01, c2 = 0.03 * 0.03;
  std::vector<double> x(n), y(n), xx(n), yy(n), xy(n), mx(n), my(n), sxx(n), syy(n), sxy(n), tmp(n);
  double total = 0;
  for (int c = 0; c < 4; ++c) {
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = a[i * 4 + static_cast<std::size_t>(c)] * kInv255;
      y[i] = b[i * 4 + static_cast<std::size_t>(c)] * kInv255;
      xx[i] = x[i] * x[i];
      yy[i] = y[i] * y[i];
      xy[i] = x[i] * y[i];
    }
    blur(x, mx, tmp, size, k);
    blur(y, my, tmp, size, k);
    blur(xx, sxx, tmp, size, k);
    blur(yy, syy, tmp, size, k);
    blur(xy, sxy, tmp, size, k);
    double s = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const double vx = sxx[i] - mx[i] * mx[i], vy = syy[i] - my[i] * my[i], cov = sxy[i] - mx[i] * my[i];
      s += ((2 * mx[i] * my[i] + c1) * (2 * cov + c2)) / ((mx[i] * mx[i] + my[i] * my[i] + c1) * (vx + vy + c2));
    }
    total += s / static_cast<double>(n);
  }
  return total / 4;
}

ClipScores score(const Clip& ref, const Clip& test, std::span<const int> frames_in) {
  if (ref.size != test.size || ref.frames != test.frames) throw std::invalid_argument("score: clips differ in shape");
  const std::vector<int> frames = frames_in.empty() ? all_frames(ref.frames) : std::vector<int>(frames_in.begin(), frames_in.end());
  ClipScores s;
  double mse_sum = 0;
  for (const int f : frames) {
    const double m = mse(ref.frame(f), test.frame(f));
    mse_sum += m;
    s.frame_psnr.push_back(psnr_from_mse(m));
    s.frame_ssim.push_back(ssim(ref.frame(f), test.frame(f), ref.size));
  }
  s.psnr = psnr_from_mse(mse_sum / static_cast<double>(frames.size()));
  s.ssim = std::ranges::fold_left(s.frame_ssim, 0.0, std::plus{}) / static_cast<double>(frames.size());
  // Temporal differences over consecutive frame pairs (wrapping for looping clips).
  const int pairs = ref.loop ? ref.frames : ref.frames - 1;
  double tm = 0, e_ref = 0, e_test = 0;
  const std::size_t nb = ref.frame_bytes();
  for (int f = 0; f < pairs; ++f) {
    const int g = (f + 1) % ref.frames, h = (f + 2) % ref.frames;
    const auto r0 = ref.frame(f), r1 = ref.frame(g), t0 = test.frame(f), t1 = test.frame(g);
    const bool second = ref.loop || f + 2 < ref.frames;
    const auto r2 = ref.frame(h), t2 = test.frame(h);
    for (std::size_t i = 0; i < nb; ++i) {
      const double dr = (double(r1[i]) - double(r0[i])) * kInv255, dt = (double(t1[i]) - double(t0[i])) * kInv255;
      tm += (dr - dt) * (dr - dt);
      if (second) {
        const double ar = (double(r2[i]) - 2.0 * r1[i] + r0[i]) * kInv255, at = (double(t2[i]) - 2.0 * t1[i] + t0[i]) * kInv255;
        e_ref += ar * ar;
        e_test += at * at;
      }
    }
  }
  s.temporal_psnr = pairs > 0 ? psnr_from_mse(tm / (static_cast<double>(pairs) * static_cast<double>(nb))) : kPsnrCap;
  s.flicker = e_ref > 0 ? e_test / e_ref : (e_test > 0 ? 1e9 : 1.0);
  return s;
}

ClipStats stats(const Clip& clip) {
  ClipStats s;
  const int n = clip.size;
  const std::size_t px = clip.pixels();
  s.mean_frame.assign(px * 4, 0.f);
  const bool pow2 = (n & (n - 1)) == 0;
  const int bins = n / 2;
  std::vector<double> spec(static_cast<std::size_t>(bins), 0.0), count(static_cast<std::size_t>(bins), 0.0);
  std::vector<std::complex<double>> plane(px), line(static_cast<std::size_t>(n));
  double motion = 0;
  for (int f = 0; f < clip.frames; ++f) {
    const auto fr = clip.frame(f);
    double cov = 0, emi = 0;
    for (std::size_t i = 0; i < px; ++i) {
      const double a = fr[i * 4 + 3] * kInv255;
      cov += a;
      for (int c = 0; c < 3; ++c) emi += std::max(0.0, fr[i * 4 + static_cast<std::size_t>(c)] * kInv255 - a) / 3.0;
      for (int c = 0; c < 4; ++c) s.mean_frame[i * 4 + static_cast<std::size_t>(c)] += static_cast<float>(fr[i * 4 + static_cast<std::size_t>(c)]);
      plane[i] = 0.2126 * fr[i * 4] + 0.7152 * fr[i * 4 + 1] + 0.0722 * fr[i * 4 + 2];
    }
    s.coverage.push_back(cov / static_cast<double>(px));
    s.emission.push_back(emi / static_cast<double>(px));
    if (f > 0) {
      const auto pr = clip.frame(f - 1);
      double d = 0;
      for (std::size_t i = 0; i < fr.size(); ++i) d += std::abs(double(fr[i]) - double(pr[i]));
      motion += d * kInv255 / static_cast<double>(fr.size());
    }
    if (!pow2) continue;
    for (int y = 0; y < n; ++y) {  // 2D FFT: rows then columns
      std::span row(plane.data() + static_cast<std::size_t>(y) * n, static_cast<std::size_t>(n));
      fft(row);
    }
    for (int x = 0; x < n; ++x) {
      for (int y = 0; y < n; ++y) line[static_cast<std::size_t>(y)] = plane[static_cast<std::size_t>(y) * n + x];
      fft(line);
      for (int y = 0; y < n; ++y) plane[static_cast<std::size_t>(y) * n + x] = line[static_cast<std::size_t>(y)];
    }
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) {
        const int ky = y <= n / 2 ? y : y - n, kx = x <= n / 2 ? x : x - n;
        const int r = static_cast<int>(std::lround(std::sqrt(double(kx * kx + ky * ky))));
        if (r < 1 || r > bins) continue;
        spec[static_cast<std::size_t>(r - 1)] += std::norm(plane[static_cast<std::size_t>(y) * n + x]);
        count[static_cast<std::size_t>(r - 1)] += 1;
      }
    }
  }
  for (float& v : s.mean_frame) v /= static_cast<float>(clip.frames);
  if (pow2) {
    for (int b = 0; b < bins; ++b) {
      const auto i = static_cast<std::size_t>(b);
      s.spectrum.push_back(std::log10(1e-9 + spec[i] / std::max(1.0, count[i]) / clip.frames));
    }
  }
  s.motion = clip.frames > 1 ? motion / (clip.frames - 1) : 0;
  return s;
}

StatDistance distance(const ClipStats& ref, const ClipStats& test) {
  StatDistance d;
  const auto l1 = [](const std::vector<double>& a, const std::vector<double>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    double s = 0;
    for (std::size_t i = 0; i < n; ++i) s += std::abs(a[i] - b[i]);
    return n ? s / static_cast<double>(n) : 0.0;
  };
  d.coverage_l1 = l1(ref.coverage, test.coverage);
  d.emission_l1 = l1(ref.emission, test.emission);
  d.spectrum_l1 = l1(ref.spectrum, test.spectrum);
  double m = 0;
  for (std::size_t i = 0; i < std::min(ref.mean_frame.size(), test.mean_frame.size()); ++i) {
    const double e = (ref.mean_frame[i] - test.mean_frame[i]) * kInv255;
    m += e * e;
  }
  d.mean_frame_psnr = psnr_from_mse(m / static_cast<double>(std::max<std::size_t>(1, ref.mean_frame.size())));
  d.motion_ratio = ref.motion > 0 ? test.motion / ref.motion : 0;
  return d;
}

Interval bootstrap_mean(std::span<const double> x, int resamples, std::uint64_t seed) {
  if (x.empty()) throw std::invalid_argument("bootstrap: no data");
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::size_t> pick(0, x.size() - 1);
  std::vector<double> means(static_cast<std::size_t>(resamples));
  for (double& m : means) {
    double s = 0;
    for (std::size_t i = 0; i < x.size(); ++i) s += x[pick(rng)];
    m = s / static_cast<double>(x.size());
  }
  std::ranges::sort(means);
  Interval r;
  r.mean = std::ranges::fold_left(x, 0.0, std::plus{}) / static_cast<double>(x.size());
  r.lo = means[static_cast<std::size_t>(0.025 * (resamples - 1))];
  r.hi = means[static_cast<std::size_t>(0.975 * (resamples - 1))];
  return r;
}

Interval paired_bootstrap(std::span<const double> a, std::span<const double> b, int resamples, std::uint64_t seed) {
  if (a.size() != b.size()) throw std::invalid_argument("paired bootstrap: sizes differ");
  const auto d = std::views::zip(a, b) | std::views::transform([](auto p) { return std::get<0>(p) - std::get<1>(p); }) |
                 std::ranges::to<std::vector>();
  return bootstrap_mean(d, resamples, seed);
}

}  // namespace nfx::metrics
