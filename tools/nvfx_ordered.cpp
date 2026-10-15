// nvfx_ordered: study M (docs/REPORT.md §14): model what's ordered, leave random to random.
//
// A rollout effect's frames are an ordered part (what the effect looks like on average at its controls: the ensemble
// mean over seeds, the time-averaged fields and picture) and a random part (the fluctuations around it: their
// amplitude, their spectrum, how fast they decorrelate). The pooled statistics of study D2 (docs/REPORT.md §6.9) put
// v2's average picture 6.5 to 8 dB further from a real run's than another real run's, and its detail spectrum 4 to 7
// times as far. This tool takes both apart on training settings, with study L's oracle swaps (§12.1), and then measures
// candidates by a rule fixed before validation.
//
//   nvfx_ordered diag  --effect E [--model F] [--settings 8] [--reps 4] [--tag NAME]
//       training settings (the controls of the effect's salt-1 training runs 0 .. settings - 1): per setting `reps` real
//       runs A (the reference) and `reps` real runs B. Methods applied to the B runs, each pooled over the runs:
//         real_B            B's own frames (the floor)
//         rend_true         the learned renderer on B's true fields (the renderer alone)
//         detail_sim        B's true coarse states moving the detail layer from B's true fine fields, the simulation's
//                           renderer (the detail layer alone)
//         detail_lr         the same, the learned renderer
//         neural_true_sim   the model's stepper and detail layer from B's true state (one continuous rollout, B's
//                           seed), the simulation's renderer (the stepper and the detail layer)
//         neural_true_lr    the same, the learned renderer
//         neural_one_lr     one continuous rollout from one of the three stored start points nearest the controls, a
//                           seed of its own (start points: the stepper from where it actually starts)
//         neural_shards     the runtime as shipped (shards from the start points), a seed of its own
//       For each: the five statistics of study D2 against the pooled A runs; the ordered part at the coarse level (the
//       time- and ensemble-mean coarse heat and soot against A's: relative error, total, centroid, spread, mean
//       velocity); the random part (the RMS of the frames about their mean frame, the spectra of the mean frame and of
//       the fluctuations, the fluctuations' temporal correlation at 1 to 16 frames).
//
//   nvfx_ordered cost --effect E --current F [--cand NAME=F,...] [--sizes 64,128,256] [--reps 7]
//                     [--precision float|int8|default] [--ftz] [--out CSV]
//       thread CPU time per frame through nvfx_render, the models interleaved (the cost of §14.6).
//
//   nvfx_ordered info --model F
//       a model file's constants: the detail layer's, the stepper's advection and forcing, its start points, a mean state.
//
// Options: --root DATA (default $NEURALVFX_DATA or /root/nvfx-data; this study's files are under DATA/m_ordered),
// --results DIR (results/experiments/ordered), --threads N (1).
//
// Study M2 (docs/REPORT.md §14.8): --study m2 keeps every file under DATA/m2 instead, caches the plain statistics with
// their mean frame in single precision, and gives `val` and `test` study M2's splits (probe2, val2, test2: fresh seeds
// in 8,500,000 to 8,999,999) and its rule (non-inferiority margins fixed before validation; `val --split val2`, `test`).
//
// Study M3 (docs/REPORT.md §14.9): --study m3 keeps every file under DATA/m3 and gives the mean state for the explosion and
// dust a rule planned for power. `val --split probe3 [--null]` is the probe (training settings, 8 runs pooled; --null adds
// the current model played with fresh seeds, the null) and writes each measure's paired spread, the margins the null gives
// (m3_margins.csv) and the power at a number of settings by resampling the probe's settings (m3_power.csv); `val --split
// val3` and `test` (once) use study G's and study B's settings extended to the number of settings the probe fixed, with
// seeds in 8,300,000 to 8,499,999 and 8,700,000 to 8,899,999.
#include "args.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/dcm/mixer.hpp>
#include <neuralfx/image_io.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <functional>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <print>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <xmmintrin.h>

namespace fs = std::filesystem;
using namespace nfx;

namespace {

constexpr int kSize = 128, kRes = 32;
std::size_t sz(int v) { return static_cast<std::size_t>(v); }
std::string ename(sim::Effect e) { return std::string(sim::effect_name(e)); }
double seconds_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

struct Ctx {
  fs::path data, results;
  int threads = 1;
  bool m2 = false;  // study M2 (--study m2): files under DATA/m2, its splits and its rule
  bool m3 = false;  // study M3 (--study m3): files under DATA/m3, its splits and its rule
  fs::path dir() const { return data / (m3 ? "m3" : m2 ? "m2" : "m_ordered"); }
};

template <class F>
void parallel(int threads, int n, F&& f) {
  std::atomic<int> next{0};
  std::vector<std::jthread> pool;
  for (int t = 0; t < std::max(1, threads); ++t) {
    pool.emplace_back([&] {
      for (int i; (i = next++) < n;) f(i);
    });
  }
}

sim::Effect effect_of(const std::string& s) {
  sim::Effect e{};
  if (!sim::parse_effect(s, e)) throw std::invalid_argument("unknown effect " + s);
  return e;
}

rollout::Model load_or_throw(const fs::path& p) {
  auto m = rollout::load_model(p);
  if (!m) throw std::runtime_error(std::format("{}: {}", p.string(), m.error()));
  return std::move(*m);
}

std::vector<std::string> split_list(const std::string& s, char sep) {
  std::vector<std::string> v;
  std::size_t a = 0;
  while (a <= s.size()) {
    const std::size_t b = std::min(s.find(sep, a), s.size());
    if (b > a) v.push_back(s.substr(a, b - a));
    a = b + 1;
  }
  return v;
}

void write_rows(const fs::path& path, const std::string& header, const std::vector<std::string>& rows) {
  fs::create_directories(path.parent_path());
  std::ofstream o(path);
  o << header << "\n";
  for (const auto& r : rows) o << r << "\n";
}

// --- the runtime and the reference ---------------------------------------------------------------------------------------

struct Runtime {
  nvfx_effect* e = nullptr;
  explicit Runtime(const rollout::Model& m) {
    std::ostringstream os;
    if (auto r = rollout::save_model(os, m); !r) throw std::runtime_error(r.error());
    const std::string b = os.str();
    if (nvfx_effect_load_memory(b.data(), b.size(), &e) != NVFX_OK) throw std::runtime_error("runtime load failed");
  }
  ~Runtime() { nvfx_effect_free(e); }
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
};

// `frames` frames of the runtime at controls, at float (the frozen tests' precision; study K's int8 ties it): a stored
// start point (variation >= 0, which keeps its run's seed; one continuous rollout) or a seed (shards, as shipped).
Clip runtime_clip(nvfx_effect* e, std::span<const float> controls, int variation, std::uint64_t seed, int frames, bool shards = true) {
  nvfx_instance* in = nullptr;
  if (nvfx_instance_create(e, kSize, &in) != NVFX_OK) throw std::runtime_error("instance");
  if (nvfx_instance_set_precision(in, NVFX_PRECISION_FLOAT) != NVFX_OK) throw std::runtime_error("precision");
  nvfx_instance_set_controls(in, controls.data(), static_cast<int>(controls.size()));
  if (!shards) nvfx_instance_set_drift(in, 0.f);
  if (variation >= 0) nvfx_instance_set_variation(in, variation);
  else nvfx_instance_set_seed(in, seed);
  Clip c;
  c.allocate(kSize, frames);
  c.fps = 30.f;
  for (int f = 0; f < frames; ++f) nvfx_render(in, f / 30.0, c.frame(f).data(), kSize * 4);
  nvfx_instance_free(in);
  return c;
}

// The simulation's own renderer on heat and soot fields at 128 x 128.
void render_fields(const sim::Params& p, const std::vector<float>& T, const std::vector<float>& D, std::span<std::uint8_t> out) {
  sim::Params q = p;
  q.sim_res = kSize;
  q.size = kSize;
  sim::Fluid f(q);
  sim::State s;
  s.n = kSize;
  s.u.assign(T.size(), 0.f);
  s.v.assign(T.size(), 0.f);
  s.pressure.assign(T.size(), 0.f);
  s.temp = T;
  s.soot = D;
  f.set_state(s);
  f.render(out);
}

void to_bytes(const std::vector<float>& rgba, std::span<std::uint8_t> out) {
  for (std::size_t q = 0; q < rgba.size(); ++q) out[q] = static_cast<std::uint8_t>(std::clamp(rgba[q], 0.f, 1.f) * 255.f + 0.5f);
}

// --- settings and seeds -----------------------------------------------------------------------------------------------------

using Setting = std::array<float, 3>;
sim::Params setting_params(sim::Effect e, const Setting& s, std::uint64_t seed) {
  sim::Params q;
  q.effect = e;
  q.intensity = s[0];
  q.wind = s[1];
  q.turbulence = s[2];
  q.seed = seed;
  q.size = kSize;
  return q;
}
sim::Params run_params(sim::Effect e, std::uint64_t salt, std::uint64_t index) {
  rollout::SimRecipe r = rollout::recipe_for(e);
  r.salt = salt;
  return rollout::recipe_run(r, index);
}
// Training settings: the controls of the effect's salt-1 recipe runs (the runs the stepper was trained on).
std::vector<Setting> training_settings(sim::Effect e, int n, int first = 0) {
  std::vector<Setting> v;
  for (int i = first; i < first + n; ++i) {
    const sim::Params p = run_params(e, 1, static_cast<std::uint64_t>(i));
    v.push_back({p.intensity, p.wind, p.turbulence});
  }
  return v;
}
int frames_of(sim::Effect e) { return sim::effect_loops(e) ? 300 : 89; }
int warm_of(sim::Effect e) { return sim::effect_loops(e) ? 150 : 1; }

// --- what a clip holds: the statistics of study D2, the random part, the ordered coarse fields -------------------------------

void fft(std::span<std::complex<double>> a) {  // radix 2, in place (as metrics.cpp)
  const std::size_t n = a.size();
  for (std::size_t i = 1, j = 0; i < n; ++i) {
    std::size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (std::size_t len = 2; len <= n; len <<= 1) {
    const double ang = -2.0 * 3.14159265358979323846 / static_cast<double>(len);
    const std::complex<double> wl(std::cos(ang), std::sin(ang));
    for (std::size_t i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (std::size_t j = 0; j < len / 2; ++j) {
        const std::complex<double> u = a[i + j], v = a[i + j + len / 2] * w;
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
        w *= wl;
      }
    }
  }
}
// Radially averaged power of a luminance plane (bins 1 .. n / 2), summed into spec with the counts (as metrics::stats).
void radial_power(std::vector<std::complex<double>>& plane, int n, std::vector<double>& spec, std::vector<double>& count) {
  std::vector<std::complex<double>> line(sz(n));
  for (int y = 0; y < n; ++y) fft(std::span(plane.data() + sz(y) * sz(n), sz(n)));
  for (int x = 0; x < n; ++x) {
    for (int y = 0; y < n; ++y) line[sz(y)] = plane[sz(y) * sz(n) + sz(x)];
    fft(line);
    for (int y = 0; y < n; ++y) plane[sz(y) * sz(n) + sz(x)] = line[sz(y)];
  }
  const int bins = n / 2;
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      const int ky = y <= n / 2 ? y : y - n, kx = x <= n / 2 ? x : x - n;
      const int r = static_cast<int>(std::lround(std::sqrt(double(kx * kx + ky * ky))));
      if (r < 1 || r > bins) continue;
      spec[sz(r - 1)] += std::norm(plane[sz(y) * sz(n) + sz(x)]);
      count[sz(r - 1)] += 1;
    }
  }
}

constexpr std::array<int, 5> kLags{1, 2, 4, 8, 16};
// The random part of a clip: the frames about the clip's own mean frame.
struct Fluct {
  double rms = 0;                   // RMS over frames, pixels and RGBA of frame - mean frame (values in [0, 1])
  double lum_rms = 0;               // the same for the luminance
  std::vector<double> spec_fluct;   // log10 radial power of the luminance fluctuation, mean over frames (as metrics)
  std::vector<double> spec_mean;    // log10 radial power of the mean frame's luminance
  std::array<double, kLags.size()> acf{};  // correlation of the luminance fluctuation with itself kLags frames later
  // per band of radial bins (1-4, 5-16, 17-32, 33-64): the fluctuation's correlation with itself 1 and 4 frames later
  // (from the frames' Fourier coefficients: how fast each scale decorrelates); empty in records made before it
  std::vector<double> band_acf1, band_acf4;
};
constexpr std::array<int, 5> kBandEdges{1, 5, 17, 33, 65};
Fluct fluct_stats(const Clip& c) {
  Fluct f;
  const int n = c.size, T = c.frames;
  const std::size_t px = c.pixels();
  std::vector<double> mean(px * 4, 0.0), lmean(px, 0.0);
  const auto lum = [&](std::span<const std::uint8_t> fr, std::size_t i) { return 0.2126 * fr[i * 4] + 0.7152 * fr[i * 4 + 1] + 0.0722 * fr[i * 4 + 2]; };
  for (int t = 0; t < T; ++t) {
    const auto fr = c.frame(t);
    for (std::size_t i = 0; i < px * 4; ++i) mean[i] += fr[i];
    for (std::size_t i = 0; i < px; ++i) lmean[i] += lum(fr, i);
  }
  for (double& v : mean) v /= T;
  for (double& v : lmean) v /= T;
  const int bins = n / 2;
  std::vector<double> sf(sz(bins), 0.0), cf(sz(bins), 0.0), sm(sz(bins), 0.0), cm(sz(bins), 0.0);
  std::vector<std::complex<double>> plane(px);
  std::vector<std::vector<float>> lf(sz(T), std::vector<float>(px));  // the luminance fluctuation of every frame
  // the band of each Fourier coefficient (-1: none), and the last 5 frames' coefficients
  std::vector<int> bandof(px, -1);
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      const int ky = y <= n / 2 ? y : y - n, kx = x <= n / 2 ? x : x - n;
      const int r = static_cast<int>(std::lround(std::sqrt(double(kx * kx + ky * ky))));
      for (std::size_t b = 0; b + 1 < kBandEdges.size(); ++b) {
        if (r >= kBandEdges[b] && r < kBandEdges[b + 1]) bandof[sz(y) * sz(n) + sz(x)] = static_cast<int>(b);
      }
    }
  }
  constexpr std::size_t NB = kBandEdges.size() - 1;
  std::vector<std::vector<std::complex<double>>> hist(5, std::vector<std::complex<double>>(px));
  std::array<double, NB> c1{}, p1a{}, p1b{}, c4{}, p4a{}, p4b{};
  double e2 = 0, l2 = 0;
  for (int t = 0; t < T; ++t) {
    const auto fr = c.frame(t);
    for (std::size_t i = 0; i < px * 4; ++i) {
      const double d = (fr[i] - mean[i]) / 255.0;
      e2 += d * d;
    }
    for (std::size_t i = 0; i < px; ++i) {
      const double d = lum(fr, i) - lmean[i];
      lf[sz(t)][i] = static_cast<float>(d);
      l2 += d * d / (255.0 * 255.0);
      plane[i] = d;
    }
    radial_power(plane, n, sf, cf);  // plane now holds the coefficients
    hist[sz(t % 5)] = plane;
    for (const int lag : {1, 4}) {
      if (t < lag) continue;
      const auto& a = hist[sz(t % 5)];
      const auto& b = hist[sz((t - lag) % 5)];
      auto& cc = lag == 1 ? c1 : c4;
      auto& pa = lag == 1 ? p1a : p4a;
      auto& pb = lag == 1 ? p1b : p4b;
      for (std::size_t i = 0; i < px; ++i) {
        if (bandof[i] < 0) continue;
        const std::size_t q = sz(bandof[i]);
        cc[q] += (a[i] * std::conj(b[i])).real();
        pa[q] += std::norm(a[i]);
        pb[q] += std::norm(b[i]);
      }
    }
  }
  for (std::size_t q = 0; q < NB; ++q) {
    f.band_acf1.push_back(p1a[q] > 0 && p1b[q] > 0 ? c1[q] / std::sqrt(p1a[q] * p1b[q]) : 0.0);
    f.band_acf4.push_back(p4a[q] > 0 && p4b[q] > 0 ? c4[q] / std::sqrt(p4a[q] * p4b[q]) : 0.0);
  }
  for (std::size_t i = 0; i < px; ++i) plane[i] = lmean[i];
  radial_power(plane, n, sm, cm);
  f.rms = std::sqrt(e2 / (static_cast<double>(T) * static_cast<double>(px) * 4.0));
  f.lum_rms = std::sqrt(l2 / (static_cast<double>(T) * static_cast<double>(px)));
  for (int b = 0; b < bins; ++b) {
    f.spec_fluct.push_back(std::log10(1e-9 + sf[sz(b)] / std::max(1.0, cf[sz(b)]) / T));
    f.spec_mean.push_back(std::log10(1e-9 + sm[sz(b)] / std::max(1.0, cm[sz(b)])));
  }
  for (std::size_t k = 0; k < kLags.size(); ++k) {
    double num = 0, den = 0;
    for (int t = 0; t + kLags[k] < T; ++t) {
      for (std::size_t i = 0; i < px; ++i) num += static_cast<double>(lf[sz(t)][i]) * lf[sz(t + kLags[k])][i];
    }
    for (int t = 0; t < T; ++t) {
      for (std::size_t i = 0; i < px; ++i) den += static_cast<double>(lf[sz(t)][i]) * lf[sz(t)][i];
    }
    f.acf[k] = den > 0 ? (num / static_cast<double>(T - kLags[k])) / (den / T) : 0.0;
  }
  return f;
}

// A run of one method: its statistics, its random part, and (when it has a coarse state) the time-mean coarse fields
// (kRes^2 cells x u, v, heat, soot; empty otherwise).
struct Rec {
  metrics::ClipStats st;
  Fluct fl;
  std::vector<double> cm;
};

void put_vec(std::ostream& o, const std::vector<double>& v) {
  const auto n = static_cast<std::uint32_t>(v.size());
  o.write(reinterpret_cast<const char*>(&n), 4);
  o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(double)));
}
bool get_vec(std::istream& i, std::vector<double>& v) {
  std::uint32_t n = 0;
  i.read(reinterpret_cast<char*>(&n), 4);
  if (!i || n > 1000000) return false;
  v.resize(n);
  i.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(double)));
  return static_cast<bool>(i);
}
void put_stats(std::ostream& o, const metrics::ClipStats& s) {
  put_vec(o, s.coverage);
  put_vec(o, s.emission);
  put_vec(o, s.spectrum);
  put_vec(o, std::vector<double>(s.mean_frame.begin(), s.mean_frame.end()));
  o.write(reinterpret_cast<const char*>(&s.motion), sizeof(double));
}
bool get_stats(std::istream& i, metrics::ClipStats& s) {
  std::vector<double> mf;
  if (!get_vec(i, s.coverage) || !get_vec(i, s.emission) || !get_vec(i, s.spectrum) || !get_vec(i, mf)) return false;
  s.mean_frame.assign(mf.begin(), mf.end());
  i.read(reinterpret_cast<char*>(&s.motion), sizeof(double));
  return static_cast<bool>(i);
}
void save_rec(const fs::path& p, const Rec& r) {
  fs::create_directories(p.parent_path());
  const fs::path tmp = p.string() + ".tmp";
  {
    std::ofstream o(tmp, std::ios::binary);
    put_stats(o, r.st);
    const std::vector<double> head{r.fl.rms, r.fl.lum_rms};
    put_vec(o, head);
    put_vec(o, r.fl.spec_fluct);
    put_vec(o, r.fl.spec_mean);
    put_vec(o, std::vector<double>(r.fl.acf.begin(), r.fl.acf.end()));
    put_vec(o, r.cm);
    put_vec(o, r.fl.band_acf1);
    put_vec(o, r.fl.band_acf4);
  }
  fs::rename(tmp, p);
}
bool load_rec(const fs::path& p, Rec& r) {
  std::ifstream i(p, std::ios::binary);
  if (!i) return false;
  std::vector<double> head, acf;
  if (!get_stats(i, r.st) || !get_vec(i, head) || head.size() != 2 || !get_vec(i, r.fl.spec_fluct) || !get_vec(i, r.fl.spec_mean) || !get_vec(i, acf) ||
      acf.size() != kLags.size() || !get_vec(i, r.cm)) {
    return false;
  }
  r.fl.rms = head[0];
  r.fl.lum_rms = head[1];
  std::ranges::copy(acf, r.fl.acf.begin());
  if (!get_vec(i, r.fl.band_acf1) || !get_vec(i, r.fl.band_acf4)) {  // a record from before the band correlations
    r.fl.band_acf1.clear();
    r.fl.band_acf4.clear();
  }
  return true;
}
Rec make_rec(const Clip& c, std::vector<double> cm = {}) {
  Rec r;
  r.st = metrics::stats(c);
  r.fl = fluct_stats(c);
  r.cm = std::move(cm);
  return r;
}

// The statistics of several runs pooled (study D2's protocol): per-frame curves, log spectrum, mean frame and motion
// averaged over the runs; the random part and the coarse means likewise.
metrics::ClipStats pool(const std::vector<metrics::ClipStats>& v) {
  metrics::ClipStats p = v.at(0);
  const double n = static_cast<double>(v.size());
  for (std::size_t k = 1; k < v.size(); ++k) {
    for (std::size_t i = 0; i < p.coverage.size(); ++i) p.coverage[i] += v[k].coverage[i];
    for (std::size_t i = 0; i < p.emission.size(); ++i) p.emission[i] += v[k].emission[i];
    for (std::size_t i = 0; i < p.spectrum.size(); ++i) p.spectrum[i] += v[k].spectrum[i];
    for (std::size_t i = 0; i < p.mean_frame.size(); ++i) p.mean_frame[i] += v[k].mean_frame[i];
    p.motion += v[k].motion;
  }
  for (double& x : p.coverage) x /= n;
  for (double& x : p.emission) x /= n;
  for (double& x : p.spectrum) x /= n;
  for (float& x : p.mean_frame) x /= static_cast<float>(n);
  p.motion /= n;
  return p;
}
Rec pool_recs(const std::vector<Rec>& v) {
  Rec p;
  std::vector<metrics::ClipStats> s;
  for (const auto& r : v) s.push_back(r.st);
  p.st = pool(s);
  const double n = static_cast<double>(v.size());
  p.fl = v.at(0).fl;
  p.cm = v.at(0).cm;
  for (std::size_t k = 1; k < v.size(); ++k) {
    p.fl.rms += v[k].fl.rms;
    p.fl.lum_rms += v[k].fl.lum_rms;
    for (std::size_t i = 0; i < p.fl.spec_fluct.size(); ++i) p.fl.spec_fluct[i] += v[k].fl.spec_fluct[i];
    for (std::size_t i = 0; i < p.fl.spec_mean.size(); ++i) p.fl.spec_mean[i] += v[k].fl.spec_mean[i];
    for (std::size_t i = 0; i < kLags.size(); ++i) p.fl.acf[i] += v[k].fl.acf[i];
    for (std::size_t i = 0; i < p.fl.band_acf1.size() && i < v[k].fl.band_acf1.size(); ++i) p.fl.band_acf1[i] += v[k].fl.band_acf1[i];
    for (std::size_t i = 0; i < p.fl.band_acf4.size() && i < v[k].fl.band_acf4.size(); ++i) p.fl.band_acf4[i] += v[k].fl.band_acf4[i];
    for (std::size_t i = 0; i < p.cm.size() && i < v[k].cm.size(); ++i) p.cm[i] += v[k].cm[i];
  }
  p.fl.rms /= n;
  p.fl.lum_rms /= n;
  for (double& x : p.fl.spec_fluct) x /= n;
  for (double& x : p.fl.spec_mean) x /= n;
  for (double& x : p.fl.acf) x /= n;
  for (double& x : p.fl.band_acf1) x /= n;
  for (double& x : p.fl.band_acf4) x /= n;
  for (double& x : p.cm) x /= n;
  return p;
}

double l1(const std::vector<double>& a, const std::vector<double>& b) {
  const std::size_t n = std::min(a.size(), b.size());
  double s = 0;
  for (std::size_t i = 0; i < n; ++i) s += std::abs(a[i] - b[i]);
  return n ? s / static_cast<double>(n) : 0.0;
}
// Mean signed difference over a range of radial bins [b0, b1) (1-based bins as metrics: bin b is index b - 1).
double band(const std::vector<double>& a, const std::vector<double>& b, int b0, int b1) {
  double s = 0;
  int n = 0;
  for (int k = b0; k < b1 && sz(k - 1) < std::min(a.size(), b.size()); ++k, ++n) s += a[sz(k - 1)] - b[sz(k - 1)];
  return n ? s / n : 0.0;
}

// Moments of a time-mean coarse channel: total per cell, centroid (frame fractions, y up) and spread (standard deviation).
struct Moments {
  double total = 0, cx = 0, cy = 0, sx = 0, sy = 0;
};
Moments moments(const std::vector<double>& cm, int c) {
  Moments m;
  double sx = 0, sy = 0, sxx = 0, syy = 0;
  for (int y = 0; y < kRes; ++y) {
    for (int x = 0; x < kRes; ++x) {
      const double h = std::max(0.0, cm[(sz(y) * kRes + sz(x)) * rollout::kPhys + sz(c)]);
      const double px = (x + 0.5) / kRes, py = (y + 0.5) / kRes;
      m.total += h;
      sx += h * px;
      sy += h * py;
      sxx += h * px * px;
      syy += h * py * py;
    }
  }
  if (m.total > 0) {
    m.cx = sx / m.total;
    m.cy = sy / m.total;
    m.sx = std::sqrt(std::max(0.0, sxx / m.total - m.cx * m.cx));
    m.sy = std::sqrt(std::max(0.0, syy / m.total - m.cy * m.cy));
  }
  m.total /= kRes * kRes;
  return m;
}
// |a - b|_2 / |b|_2 over the cells of channel c.
double rel_err(const std::vector<double>& a, const std::vector<double>& b, int c) {
  double e = 0, t = 0;
  for (int i = 0; i < kRes * kRes; ++i) {
    const double x = a[sz(i) * rollout::kPhys + sz(c)], y = b[sz(i) * rollout::kPhys + sz(c)];
    e += (x - y) * (x - y);
    t += y * y;
  }
  return t > 0 ? std::sqrt(e / t) : 0;
}
// Mean of channel c over the cells where the reference's time-mean heat or soot is above 2% of its peak (the effect).
double mean_where(const std::vector<double>& cm, const std::vector<double>& ref, int c) {
  double mx = 0;
  for (int i = 0; i < kRes * kRes; ++i) mx = std::max({mx, ref[sz(i) * rollout::kPhys + 2], ref[sz(i) * rollout::kPhys + 3]});
  double s = 0;
  int n = 0;
  for (int i = 0; i < kRes * kRes; ++i) {
    if (std::max(ref[sz(i) * rollout::kPhys + 2], ref[sz(i) * rollout::kPhys + 3]) > 0.02 * mx) {
      s += cm[sz(i) * rollout::kPhys + sz(c)];
      ++n;
    }
  }
  return n ? s / n : 0;
}

// --- real runs -------------------------------------------------------------------------------------------------------------

// A real run at a setting: warmed up, then frames_of(e) frames; its record (with the coarse means), cached under
// DATA/m_ordered/real.
Rec real_rec(const Ctx& c, sim::Effect e, const Setting& s, std::uint64_t seed) {
  const fs::path p = c.dir() / "real" / std::format("{}_{}.rec", ename(e), seed);
  Rec r;
  if (load_rec(p, r)) return r;
  const sim::Params q = setting_params(e, s, seed);
  sim::Fluid f(q);
  for (int i = 0; i < warm_of(e); ++i) f.step_frame();
  Clip cl;
  cl.allocate(kSize, frames_of(e));
  cl.fps = 30.f;
  std::vector<double> cm(sz(kRes) * kRes * rollout::kPhys, 0.0);
  std::vector<float> co(cm.size());
  for (int i = 0; i < cl.frames; ++i) {
    f.step_frame();
    f.render(cl.frame(i));
    rollout::coarse_from_sim(f.state(), kRes, q.fps, co);
    for (std::size_t j = 0; j < cm.size(); ++j) cm[j] += static_cast<double>(co[j]) / cl.frames;
  }
  r = make_rec(cl, std::move(cm));
  save_rec(p, r);
  return r;
}

// --- diag: the ordered and the random parts, taken apart (training settings) -------------------------------------------------

const std::vector<std::string> kDiagMethods = {"real_B",         "rend_true",      "detail_sim",    "detail_lr",
                                               "neural_true_sim", "neural_true_lr", "neural_one_lr", "neural_shards"};

// The three stored start points nearest the controls, nearest first (the runtime's pick_start chooses among them).
std::vector<int> nearest_starts(const rollout::Model& M, std::span<const float> ctl) {
  std::vector<std::pair<float, int>> d;
  for (std::size_t i = 0; i < M.starts.size(); ++i) {
    float s = 0;
    for (std::size_t k = 0; k < 3 && k < M.starts[i].controls.size(); ++k) s += (M.starts[i].controls[k] - ctl[k]) * (M.starts[i].controls[k] - ctl[k]);
    d.emplace_back(s, static_cast<int>(i));
  }
  std::ranges::sort(d);
  std::vector<int> v;
  for (std::size_t i = 0; i < std::min<std::size_t>(3, d.size()); ++i) v.push_back(d[i].second);
  return v;
}

// Every method on one B run; records cached under DATA/m_ordered/diag/<tag>/<effect>.
std::map<std::string, Rec> diag_run(const Ctx& c, sim::Effect e, const rollout::Model& M, const std::string& tag, const Setting& s, std::uint64_t seed_b,
                                    std::uint64_t seed_play, int k) {
  const fs::path dir = c.dir() / "diag" / tag / ename(e);
  std::map<std::string, Rec> out;
  bool all = true;
  for (const auto& m : kDiagMethods) {
    Rec r;
    if (load_rec(dir / std::format("{}_{}.rec", m, seed_b), r)) out[m] = std::move(r);
    else all = false;
  }
  if (all) return out;
  out.clear();
  const int F = frames_of(e), W = warm_of(e), C = M.h.channels();
  const std::vector<float> ctl(s.begin(), s.end());
  const sim::Params pb = setting_params(e, s, seed_b);
  const std::size_t NC = sz(kRes) * kRes * rollout::kPhys;
  std::map<std::string, Clip> clips;
  std::map<std::string, std::vector<double>> cms;
  for (const auto& m : kDiagMethods) {
    clips[m].allocate(kSize, F);
    clips[m].fps = 30.f;
  }
  std::vector<float> rgba(sz(kSize) * kSize * 4);
  // B: the real run, its coarse states, the learned renderer on its true fields
  sim::Fluid fb(pb);
  for (int i = 0; i < W; ++i) fb.step_frame();
  const sim::State st0 = fb.state();
  std::vector<std::vector<float>> coarse(sz(F), std::vector<float>(NC));
  std::vector<double> cm_b(NC, 0.0);
  for (int i = 0; i < F; ++i) {
    fb.step_frame();
    fb.render(clips["real_B"].frame(i));
    const sim::State st = fb.state();
    rollout::coarse_from_sim(st, kRes, pb.fps, coarse[sz(i)]);
    for (std::size_t j = 0; j < NC; ++j) cm_b[j] += static_cast<double>(coarse[sz(i)][j]) / F;
    rollout::State rs;
    rs.res = kRes;
    rs.size = kSize;
    rs.coarse.assign(sz(kRes) * kRes * sz(C), 0.f);
    for (int q = 0; q < kRes * kRes; ++q) {
      for (int ch = 0; ch < rollout::kPhys; ++ch) rs.coarse[sz(q) * sz(C) + sz(ch)] = coarse[sz(i)][sz(q) * rollout::kPhys + sz(ch)];
    }
    rs.fine_t = st.temp;
    rs.fine_d = st.soot;
    rollout::render(M, rs, rgba);
    to_bytes(rgba, clips["rend_true"].frame(i));
  }
  cms["real_B"] = cm_b;
  // the detail layer on B's true coarse states (the oracle stepper)
  {
    rollout::State rs;
    rs.res = kRes;
    rs.size = kSize;
    rs.time = st0.time;
    rs.coarse.assign(sz(kRes) * kRes * sz(C), 0.f);
    rs.flow.assign(sz(kRes) * kRes * 2, 0.f);
    rs.pressure.assign(sz(kRes) * kRes, 0.f);
    rs.fine_t = st0.temp;
    rs.fine_d = st0.soot;
    rs.since_start = 1.f;
    for (int i = 0; i < F; ++i) {
      for (int j = 0; j < kRes * kRes; ++j) {
        for (int q = 0; q < rollout::kPhys; ++q) rs.coarse[sz(j) * sz(C) + sz(q)] = coarse[sz(i)][sz(j) * rollout::kPhys + sz(q)];
        rs.flow[sz(j) * 2] = coarse[sz(i)][sz(j) * rollout::kPhys];
        rs.flow[sz(j) * 2 + 1] = coarse[sz(i)][sz(j) * rollout::kPhys + 1];
      }
      rollout::detail_step(M, rs, pb.seed, ctl);
      rs.time += 1.f / M.fps;
      rs.since_start += 1.f / M.fps;
      render_fields(pb, rs.fine_t, rs.fine_d, clips["detail_sim"].frame(i));
      rollout::render(M, rs, rgba);
      to_bytes(rgba, clips["detail_lr"].frame(i));
    }
  }
  const auto rollout_into = [&](rollout::State rs, std::uint64_t seed, const std::string& sim_name, const std::string& lr_name) {
    std::vector<double> cm(NC, 0.0);
    for (int i = 0; i < F; ++i) {
      rollout::step(M, rs, ctl, seed);
      for (int j = 0; j < kRes * kRes; ++j) {
        for (int q = 0; q < rollout::kPhys; ++q) cm[sz(j) * rollout::kPhys + sz(q)] += static_cast<double>(rs.coarse[sz(j) * sz(C) + sz(q)]) / F;
      }
      if (!sim_name.empty()) render_fields(pb, rs.fine_t, rs.fine_d, clips[sim_name].frame(i));
      rollout::render(M, rs, rgba);
      to_bytes(rgba, clips[lr_name].frame(i));
    }
    cms[lr_name] = cm;
    if (!sim_name.empty()) cms[sim_name] = cm;
  };
  {  // the model from B's true state (its fine fields at full size), B's seed
    rollout::Model mt = M;
    mt.h.start_fine = kSize;
    rollout::StartPoint sp;
    sp.controls = ctl;
    sp.seed = pb.seed;
    sp.time = st0.time;
    sp.coarse.resize(NC);
    rollout::coarse_from_sim(st0, kRes, pb.fps, sp.coarse);
    sp.fine_t = st0.temp;
    sp.fine_d = st0.soot;
    mt.starts = {sp};
    rollout_into(rollout::start(mt, 0, kSize, ctl, pb.seed), pb.seed, "neural_true_sim", "neural_true_lr");
  }
  {  // one continuous rollout from a stored start point (one of the three nearest, by k), a seed of its own
    const auto near = nearest_starts(M, ctl);
    const int idx = near[sz(k) % near.size()];
    rollout_into(rollout::start(M, idx, kSize, ctl, seed_play + 50), seed_play + 50, "", "neural_one_lr");
  }
  {  // the runtime as shipped
    Runtime rt(M);
    clips["neural_shards"] = runtime_clip(rt.e, ctl, -1, seed_play, F);
  }
  for (const auto& m : kDiagMethods) {
    Rec r = make_rec(clips[m], cms.count(m) ? cms[m] : std::vector<double>{});
    save_rec(dir / std::format("{}_{}.rec", m, seed_b), r);
    out[m] = std::move(r);
  }
  return out;
}

// The tables of a diagnosis: study D2's statistics, the random part and the ordered coarse part of each method against the
// pooled A runs; rows to csv.
void diag_tables(const std::string& en, const std::string& tag, int reps, const std::vector<Rec>& ref, const std::vector<std::map<std::string, Rec>>& pooled,
                 const std::vector<std::string>& methods, const std::vector<std::string>& coarse_methods, const fs::path& csv) {
  const int n_set = static_cast<int>(ref.size());
  std::vector<std::string> rows;
  std::println("\n{} ({}): study D2's statistics against {} real runs pooled per setting, mean over {} training settings", en, tag, reps, n_set);
  std::println("{:<18}{:>10}{:>10}{:>9}{:>11}{:>11}{:>9}", "method", "spectrum", "|ln mot|", "motion", "coverage", "emission", "PSNR");
  for (const auto& m : methods) {
    double sp = 0, lm = 0, mo = 0, cv = 0, em = 0, ps = 0;
    for (int si = 0; si < n_set; ++si) {
      const auto d = metrics::distance(ref[sz(si)].st, pooled[sz(si)].at(m).st);
      sp += d.spectrum_l1 / n_set;
      lm += std::abs(std::log(std::max(1e-3, d.motion_ratio))) / n_set;
      mo += d.motion_ratio / n_set;
      cv += d.coverage_l1 / n_set;
      em += d.emission_l1 / n_set;
      ps += d.mean_frame_psnr / n_set;
      rows.push_back(std::format("{},{},stats,{},{:.5f},{:.5f},{:.6f},{:.6f},{:.4f}", en, m, si, d.spectrum_l1, d.motion_ratio, d.coverage_l1, d.emission_l1,
                                 d.mean_frame_psnr));
    }
    std::println("{:<18}{:>10.4f}{:>10.4f}{:>9.3f}{:>11.5f}{:>11.5f}{:>9.2f}", m, sp, lm, mo, cv, em, ps);
  }
  // the spectrum taken apart: the mean frame's and the fluctuations', and bands of the whole
  std::println("\nthe random part against A's (mean over settings): fluctuation RMS ratio (RGBA, luminance); spectrum distance of the whole, of the "
               "mean frame and of the fluctuations; the whole's log10 power difference in bands of radial bins; the fluctuations' correlation at "
               "1, 2, 4, 8, 16 frames");
  std::println("{:<18}{:>7}{:>7}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}  acf", "method", "rms", "lrms", "whole", "mean", "fluct", "b1-4", "b5-16", "b17-32",
               "b33-64");
  {
    std::array<double, kLags.size()> acf{};
    for (int si = 0; si < n_set; ++si) {
      for (std::size_t q = 0; q < kLags.size(); ++q) acf[q] += ref[sz(si)].fl.acf[q] / n_set;
    }
    std::println("{:<18}{:>7}{:>7}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}  {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}", "real_A", "1", "1", "0", "0", "0", "0", "0", "0", "0",
                 acf[0], acf[1], acf[2], acf[3], acf[4]);
  }
  for (const auto& m : methods) {
    double rr = 0, lr = 0, sw = 0, smn = 0, sfl = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0;
    std::array<double, kLags.size()> acf{};
    for (int si = 0; si < n_set; ++si) {
      const Rec& x = pooled[sz(si)].at(m);
      const Rec& r = ref[sz(si)];
      rr += x.fl.rms / r.fl.rms / n_set;
      lr += x.fl.lum_rms / r.fl.lum_rms / n_set;
      sw += l1(x.st.spectrum, r.st.spectrum) / n_set;
      smn += l1(x.fl.spec_mean, r.fl.spec_mean) / n_set;
      sfl += l1(x.fl.spec_fluct, r.fl.spec_fluct) / n_set;
      b1 += band(x.st.spectrum, r.st.spectrum, 1, 5) / n_set;
      b2 += band(x.st.spectrum, r.st.spectrum, 5, 17) / n_set;
      b3 += band(x.st.spectrum, r.st.spectrum, 17, 33) / n_set;
      b4 += band(x.st.spectrum, r.st.spectrum, 33, 65) / n_set;
      for (std::size_t q = 0; q < kLags.size(); ++q) acf[q] += x.fl.acf[q] / n_set;
      rows.push_back(std::format("{},{},random,{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}", en, m, si,
                                 x.fl.rms / r.fl.rms, x.fl.lum_rms / r.fl.lum_rms, l1(x.st.spectrum, r.st.spectrum), l1(x.fl.spec_mean, r.fl.spec_mean),
                                 l1(x.fl.spec_fluct, r.fl.spec_fluct), band(x.st.spectrum, r.st.spectrum, 1, 5), band(x.st.spectrum, r.st.spectrum, 5, 17),
                                 band(x.st.spectrum, r.st.spectrum, 17, 33), band(x.st.spectrum, r.st.spectrum, 33, 65), x.fl.acf[0], x.fl.acf[1], x.fl.acf[2],
                                 x.fl.acf[3], x.fl.acf[4]));
    }
    std::println("{:<18}{:>7.3f}{:>7.3f}{:>8.4f}{:>8.4f}{:>8.4f}{:>+8.3f}{:>+8.3f}{:>+8.3f}{:>+8.3f}  {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}", m, rr, lr, sw, smn, sfl,
                 b1, b2, b3, b4, acf[0], acf[1], acf[2], acf[3], acf[4]);
  }
  // how fast each scale decorrelates
  if (!ref[0].fl.band_acf1.empty()) {
    std::println("\nthe fluctuations' correlation with themselves 1 frame / 4 frames later, by band of radial bins (mean over settings)");
    std::println("{:<18}{:>14}{:>14}{:>14}{:>14}", "method", "b1-4", "b5-16", "b17-32", "b33-64");
    const auto line = [&](const std::string& name, const std::function<const Fluct&(int)>& get) {
      std::array<double, 4> a1{}, a4{};
      for (int si = 0; si < n_set; ++si) {
        const Fluct& f = get(si);
        for (std::size_t q = 0; q < 4 && q < f.band_acf1.size(); ++q) {
          a1[q] += f.band_acf1[q] / n_set;
          a4[q] += f.band_acf4[q] / n_set;
        }
      }
      std::println("{:<18}{:>7.3f}/{:<6.3f}{:>7.3f}/{:<6.3f}{:>7.3f}/{:<6.3f}{:>7.3f}/{:<6.3f}", name, a1[0], a4[0], a1[1], a4[1], a1[2], a4[2], a1[3], a4[3]);
    };
    line("real_A", [&](int si) -> const Fluct& { return ref[sz(si)].fl; });
    for (const auto& m : methods) line(m, [&](int si) -> const Fluct& { return pooled[sz(si)].at(m).fl; });
  }
  // the ordered part at the coarse level
  std::println("\nthe ordered part, coarse (time and ensemble mean over the runs; mean over settings): relative error of the mean heat and soot "
               "against A's; total, centroid height, centroid x, vertical and horizontal spread of the mean heat (h) and soot (s); mean u and v "
               "over the effect's cells (coarse cells per frame)");
  std::println("{:<16}{:>7}{:>7} | {:>7}{:>7}{:>7}{:>7}{:>7} | {:>7}{:>7}{:>7}{:>7}{:>7} | {:>7}{:>7}", "method", "err h", "err s", "h tot", "h cy", "h cx",
               "h sy", "h sx", "s tot", "s cy", "s cx", "s sy", "s sx", "u", "v");
  const auto coarse_line = [&](const std::string& name, const std::function<const std::vector<double>&(int)>& get) {
    double eh = 0, es = 0, uu = 0, vv = 0;
    std::array<Moments, 2> mm{};
    for (int si = 0; si < n_set; ++si) {
      const auto& cm = get(si);
      const auto& rc = ref[sz(si)].cm;
      eh += rel_err(cm, rc, 2) / n_set;
      es += rel_err(cm, rc, 3) / n_set;
      uu += mean_where(cm, rc, 0) / n_set;
      vv += mean_where(cm, rc, 1) / n_set;
      for (int q = 0; q < 2; ++q) {
        const Moments x = moments(cm, 2 + q);
        mm[sz(q)].total += x.total / n_set;
        mm[sz(q)].cy += x.cy / n_set;
        mm[sz(q)].cx += x.cx / n_set;
        mm[sz(q)].sy += x.sy / n_set;
        mm[sz(q)].sx += x.sx / n_set;
      }
      const Moments h = moments(cm, 2), s2 = moments(cm, 3);
      rows.push_back(std::format("{},{},coarse,{},{:.4f},{:.4f},{:.5f},{:.4f},{:.4f},{:.4f},{:.4f},{:.5f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}", en, name, si,
                                 rel_err(cm, rc, 2), rel_err(cm, rc, 3), h.total, h.cy, h.cx, h.sy, h.sx, s2.total, s2.cy, s2.cx, s2.sy, s2.sx,
                                 mean_where(cm, rc, 0), mean_where(cm, rc, 1)));
    }
    std::println("{:<16}{:>7.3f}{:>7.3f} | {:>7.4f}{:>7.3f}{:>7.3f}{:>7.3f}{:>7.3f} | {:>7.4f}{:>7.3f}{:>7.3f}{:>7.3f}{:>7.3f} | {:>+7.3f}{:>+7.3f}", name, eh, es,
                 mm[0].total, mm[0].cy, mm[0].cx, mm[0].sy, mm[0].sx, mm[1].total, mm[1].cy, mm[1].cx, mm[1].sy, mm[1].sx, uu, vv);
  };
  coarse_line("real_A", [&](int si) -> const std::vector<double>& { return ref[sz(si)].cm; });
  for (const auto& m : coarse_methods) {
    coarse_line(m, [&](int si) -> const std::vector<double>& { return pooled[sz(si)].at(m).cm; });
  }
  write_rows(csv, "effect,method,part,setting,values...", rows);
  std::println("\nrows: {}", csv.string());
}

void step_diag(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const fs::path model_path = a.str("model", (c.data / "v2" / "models" / std::format("{}.nvfx", ename(e))).string());
  const rollout::Model M = load_or_throw(model_path);
  const std::string tag = a.str("tag", "v2");
  const int n_set = a.i("settings", 8), reps = a.i("reps", 4), first = a.i("first", 0);
  const auto settings = training_settings(e, n_set, first);
  // Seeds (training settings; this study's own): real A runs 7,100,000 + 100 s + k, real B runs 7,100,050 + 100 s + k,
  // the model's plays 7,200,000 + 100 s + k.
  const std::uint64_t base_a = 7'100'000, base_b = 7'100'050, base_play = 7'200'000;
  const auto seed = [](std::uint64_t base, int si, int k) { return base + 100 * static_cast<std::uint64_t>(si) + static_cast<std::uint64_t>(k); };
  const auto t0 = std::chrono::steady_clock::now();
  parallel(c.threads, n_set * reps, [&](int j) {
    const int si = j / reps, k = j % reps;
    (void)real_rec(c, e, settings[sz(si)], seed(base_a, si, k));
  });
  std::println("diag: {} real A runs ready ({:.0f} s)", ename(e), seconds_since(t0));
  std::fflush(stdout);
  std::vector<std::vector<std::map<std::string, Rec>>> runs(sz(n_set), std::vector<std::map<std::string, Rec>>(sz(reps)));
  parallel(c.threads, n_set * reps, [&](int j) {
    const int si = j / reps, k = j % reps;
    runs[sz(si)][sz(k)] = diag_run(c, e, M, tag, settings[sz(si)], seed(base_b, si, k), seed(base_play, si, k), k);
    std::println("diag: {} setting {} run {} done ({:.0f} s)", ename(e), si, k, seconds_since(t0));
    std::fflush(stdout);
  });
  // pooled per setting
  std::vector<Rec> ref(sz(n_set));
  std::vector<std::map<std::string, Rec>> pooled(sz(n_set));
  for (int si = 0; si < n_set; ++si) {
    std::vector<Rec> v;
    for (int k = 0; k < reps; ++k) v.push_back(real_rec(c, e, settings[sz(si)], seed(base_a, si, k)));
    ref[sz(si)] = pool_recs(v);
    for (const auto& m : kDiagMethods) {
      std::vector<Rec> w;
      for (int k = 0; k < reps; ++k) w.push_back(runs[sz(si)][sz(k)].at(m));
      pooled[sz(si)][m] = pool_recs(w);
    }
  }
  diag_tables(ename(e), tag, reps, ref, pooled, kDiagMethods, {"real_B", "neural_true_lr", "neural_one_lr"},
              c.results / std::format("m_diag_{}_{}.csv", ename(e), tag));
}

// --- dvar: detail-layer variants on the true coarse states (training settings) ------------------------------------------------

float contrast_kappa(const rollout::Model& m) {  // as rollout::calibrate_detail
  double mean = 0;
  constexpr int n = 20000;
  for (int i = 0; i < n; ++i) {
    const float phi = rollout::noise_flicker(m.noise, 9, static_cast<float>(i % 137) * 0.7f, static_cast<float>(i / 137) * 0.9f, 0.3f);
    const float t = std::clamp((phi - m.detail.edge0) / (m.detail.edge1 - m.detail.edge0), 0.f, 1.f);
    mean += static_cast<double>(t * t * (3.f - 2.f * t));
  }
  return static_cast<float>(n / std::max(1e-6, mean));
}

// A variant: "name:k=v;k=v" with the detail layer's file constants (contrast, edge0, edge1, swirl, swirl_scale, swirl_rate,
// grow, advect, soften, passes) and study M's coarse forcing (force); kappa follows the edges as calibrate_detail sets it.
struct Variant {
  std::string name, spec;
  rollout::Model m;
};
Variant make_variant(const rollout::Model& base, const std::string& spec) {
  Variant v;
  v.spec = spec;
  const std::size_t colon = spec.find(':');
  v.name = spec.substr(0, colon);
  v.m = base;
  bool edges = false;
  if (colon != std::string::npos) {
    for (const auto& kv : split_list(spec.substr(colon + 1), ';')) {
      const std::size_t eq = kv.find('=');
      if (eq == std::string::npos) throw std::invalid_argument("variant: " + kv);
      const std::string key = kv.substr(0, eq);
      const float val = std::stof(kv.substr(eq + 1));
      rollout::DetailSpec& d = v.m.detail;
      if (key == "contrast") d.contrast = val;
      else if (key == "edge0") d.edge0 = val, edges = true;
      else if (key == "edge1") d.edge1 = val, edges = true;
      else if (key == "swirl") d.swirl = val;
      else if (key == "swirl_scale") d.swirl_scale = val;
      else if (key == "swirl_rate") d.swirl_rate = val;
      else if (key == "grow") d.grow = val;
      else if (key == "advect") d.advect = val;
      else if (key == "soften") d.soften = val;
      else if (key == "passes") d.passes = static_cast<int>(val);
      else if (key == "force") v.m.h.force = val;  // study M's coarse forcing (Hyper::force)
      else throw std::invalid_argument("variant: unknown key " + key);
    }
  }
  if (edges) v.m.detail.kappa = contrast_kappa(v.m);
  return v;
}
std::string short_hash(const std::string& s) {  // FNV-1a, 8 hex digits: a variant's cache files carry its constants
  std::uint32_t h = 2166136261u;
  for (const char ch : s) h = (h ^ static_cast<std::uint8_t>(ch)) * 16777619u;
  return std::format("{:08x}", h);
}

// Each variant's detail layer on one B run's true coarse states (drawn by the simulation's renderer: dv_sim_<name>) and,
// with --play, the runtime as shipped with the variant's constants (dv_play_<name>, the diagnosis's play seed).
std::map<std::string, Rec> dvar_run(const Ctx& c, sim::Effect e, const std::string& tag, const std::vector<Variant>& vars, bool play, const Setting& s,
                                    std::uint64_t seed_b, std::uint64_t seed_play) {
  const fs::path dir = c.dir() / "diag" / tag / ename(e);
  std::map<std::string, Rec> out;
  const auto file = [&](const std::string& kind, const Variant& v) { return dir / std::format("dv_{}_{}_{}_{}.rec", kind, v.name, short_hash(v.spec), seed_b); };
  bool all = true;
  for (const auto& v : vars) {
    for (const std::string kind : {"sim", "play"}) {
      if (kind == "play" && !play) continue;
      Rec r;
      if (load_rec(file(kind, v), r)) out["dv_" + kind + "_" + v.name] = std::move(r);
      else all = false;
    }
  }
  if (all) return out;
  const int F = frames_of(e), W = warm_of(e);
  const std::vector<float> ctl(s.begin(), s.end());
  const sim::Params pb = setting_params(e, s, seed_b);
  const std::size_t NC = sz(kRes) * kRes * rollout::kPhys;
  sim::Fluid fb(pb);
  for (int i = 0; i < W; ++i) fb.step_frame();
  const sim::State st0 = fb.state();
  std::vector<std::vector<float>> coarse(sz(F), std::vector<float>(NC));
  for (int i = 0; i < F; ++i) {
    fb.step_frame();
    rollout::coarse_from_sim(fb.state(), kRes, pb.fps, coarse[sz(i)]);
  }
  for (const auto& v : vars) {
    const std::string key = "dv_sim_" + v.name;
    if (!out.count(key)) {
      const int C = v.m.h.channels();
      rollout::State rs;
      rs.res = kRes;
      rs.size = kSize;
      rs.time = st0.time;
      rs.coarse.assign(sz(kRes) * kRes * sz(C), 0.f);
      rs.flow.assign(sz(kRes) * kRes * 2, 0.f);
      rs.pressure.assign(sz(kRes) * kRes, 0.f);
      rs.fine_t = st0.temp;
      rs.fine_d = st0.soot;
      rs.since_start = 1.f;
      Clip cl;
      cl.allocate(kSize, F);
      cl.fps = 30.f;
      for (int i = 0; i < F; ++i) {
        for (int j = 0; j < kRes * kRes; ++j) {
          for (int q = 0; q < rollout::kPhys; ++q) rs.coarse[sz(j) * sz(C) + sz(q)] = coarse[sz(i)][sz(j) * rollout::kPhys + sz(q)];
          rs.flow[sz(j) * 2] = coarse[sz(i)][sz(j) * rollout::kPhys];
          rs.flow[sz(j) * 2 + 1] = coarse[sz(i)][sz(j) * rollout::kPhys + 1];
        }
        rollout::detail_step(v.m, rs, pb.seed, ctl);
        rs.time += 1.f / v.m.fps;
        rs.since_start += 1.f / v.m.fps;
        render_fields(pb, rs.fine_t, rs.fine_d, cl.frame(i));
      }
      Rec r = make_rec(cl);
      save_rec(file("sim", v), r);
      out[key] = std::move(r);
    }
    if (play && !out.count("dv_play_" + v.name)) {
      Runtime rt(v.m);
      Rec r = make_rec(runtime_clip(rt.e, ctl, -1, seed_play, F));
      save_rec(file("play", v), r);
      out["dv_play_" + v.name] = std::move(r);
    }
  }
  return out;
}

void step_dvar(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const fs::path model_path = a.str("model", (c.data / "v2" / "models" / std::format("{}.nvfx", ename(e))).string());
  const rollout::Model M = load_or_throw(model_path);
  const std::string tag = a.str("tag", "v2");
  const int n_set = a.i("settings", 8), reps = a.i("reps", 4);
  const bool play = a.flag("play");
  const auto settings = training_settings(e, n_set);
  std::vector<Variant> vars;
  for (const auto& v : split_list(a.need("variants"), '/')) vars.push_back(make_variant(M, v));
  const std::uint64_t base_a = 7'100'000, base_b = 7'100'050, base_play = 7'200'000;  // as diag
  const auto seed = [](std::uint64_t base, int si, int k) { return base + 100 * static_cast<std::uint64_t>(si) + static_cast<std::uint64_t>(k); };
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::vector<std::map<std::string, Rec>>> runs(sz(n_set), std::vector<std::map<std::string, Rec>>(sz(reps)));
  parallel(c.threads, n_set * reps, [&](int j) {
    const int si = j / reps, k = j % reps;
    auto base = diag_run(c, e, M, tag, settings[sz(si)], seed(base_b, si, k), seed(base_play, si, k), k);  // cached by diag
    auto var = dvar_run(c, e, tag, vars, play, settings[sz(si)], seed(base_b, si, k), seed(base_play, si, k));
    for (auto& [name, r] : var) base[name] = std::move(r);
    runs[sz(si)][sz(k)] = std::move(base);
    std::println("dvar: {} setting {} run {} done ({:.0f} s)", ename(e), si, k, seconds_since(t0));
    std::fflush(stdout);
  });
  std::vector<std::string> methods = {"real_B", "detail_sim", "neural_shards"};
  for (const auto& v : vars) {
    methods.push_back("dv_sim_" + v.name);
    if (play) methods.push_back("dv_play_" + v.name);
  }
  std::vector<Rec> ref(sz(n_set));
  std::vector<std::map<std::string, Rec>> pooled(sz(n_set));
  for (int si = 0; si < n_set; ++si) {
    std::vector<Rec> v;
    for (int k = 0; k < reps; ++k) v.push_back(real_rec(c, e, settings[sz(si)], seed(base_a, si, k)));
    ref[sz(si)] = pool_recs(v);
    for (const auto& m : methods) {
      std::vector<Rec> w;
      for (int k = 0; k < reps; ++k) w.push_back(runs[sz(si)][sz(k)].at(m));
      pooled[sz(si)][m] = pool_recs(w);
    }
  }
  for (const auto& v : vars) std::println("variant {}: {}", v.name, v.spec);
  diag_tables(ename(e), tag, reps, ref, pooled, methods, {}, c.results / std::format("m_dvar_{}_{}_{}.csv", ename(e), tag, a.str("out-tag", "dvar")));
}

// --- fitmean: the ordered part as a function of the controls (training runs) -------------------------------------------------

// The recipe's training runs (salt 1), recorded once at 32 cells and cached under DATA/m_ordered/runs (as nvfx_swirl's).
std::vector<rollout::Run> training_runs(const Ctx& c, sim::Effect e, int threads) {
  const rollout::SimRecipe r = rollout::recipe_for(e);
  std::vector<rollout::Run> runs(sz(r.runs));
  const std::size_t per = sz(kRes) * kRes * rollout::kPhys;
  parallel(threads, r.runs, [&](int j) {
    const fs::path p = c.dir() / "runs" / ename(e) / std::format("run_{}.bin", j);
    rollout::Run& run = runs[sz(j)];
    run.p = run_params(e, 1, static_cast<std::uint64_t>(j));
    run.frames = r.frames;
    run.coarse.resize(sz(r.frames) * per);
    if (std::ifstream in(p, std::ios::binary); in) {
      in.read(reinterpret_cast<char*>(run.coarse.data()), static_cast<std::streamsize>(run.coarse.size() * sizeof(float)));
      if (in) return;
    }
    run = rollout::record_run(run.p, r.frames, kRes);
    fs::create_directories(p.parent_path());
    const fs::path tmp = p.string() + ".tmp";
    {
      std::ofstream o(tmp, std::ios::binary);
      o.write(reinterpret_cast<const char*>(run.coarse.data()), static_cast<std::streamsize>(run.coarse.size() * sizeof(float)));
    }
    fs::rename(tmp, p);
  });
  return runs;
}

// Quadratic features of the controls: 1, I, W, T, I^2, W^2, T^2, IW, IT, WT.
constexpr int kFeat = 10;
std::array<double, kFeat> features(std::span<const float> c) {
  const double i = c[0], w = c[1], t = c[2];
  return {1.0, i, w, t, i * i, w * w, t * t, i * w, i * t, w * t};
}
// Solve (A + lambda D) x = b for a small symmetric system by Gaussian elimination (D: the identity except the constant term).
std::vector<double> solve_ridge(std::vector<double> A, std::vector<double> b, int n, int cols, double lambda) {
  for (int k = 1; k < n; ++k) A[sz(k) * sz(n) + sz(k)] += lambda;
  for (int k = 0; k < n; ++k) {
    int piv = k;
    for (int r = k + 1; r < n; ++r) {
      if (std::abs(A[sz(r) * sz(n) + sz(k)]) > std::abs(A[sz(piv) * sz(n) + sz(k)])) piv = r;
    }
    for (int q = 0; q < n; ++q) std::swap(A[sz(k) * sz(n) + sz(q)], A[sz(piv) * sz(n) + sz(q)]);
    for (int q = 0; q < cols; ++q) std::swap(b[sz(k) * sz(cols) + sz(q)], b[sz(piv) * sz(cols) + sz(q)]);
    for (int r = 0; r < n; ++r) {
      if (r == k) continue;
      const double f = A[sz(r) * sz(n) + sz(k)] / A[sz(k) * sz(n) + sz(k)];
      if (f == 0.0) continue;
      for (int q = k; q < n; ++q) A[sz(r) * sz(n) + sz(q)] -= f * A[sz(k) * sz(n) + sz(q)];
      for (int q = 0; q < cols; ++q) b[sz(r) * sz(cols) + sz(q)] -= f * b[sz(k) * sz(cols) + sz(q)];
    }
  }
  for (int k = 0; k < n; ++k) {
    for (int q = 0; q < cols; ++q) b[sz(k) * sz(cols) + sz(q)] /= A[sz(k) * sz(n) + sz(k)];
  }
  return b;
}

// The ordered state of each training run: looping effects its time mean over states [from, frames), one-shot effects its
// first state (where their start points are taken); averaged over blocks of `block` cells.
std::vector<std::vector<double>> run_means(const std::vector<rollout::Run>& runs, bool once, int from, int block) {
  const int G = kRes / block;
  std::vector<std::vector<double>> out;
  for (const auto& r : runs) {
    std::vector<double> m(sz(G) * sz(G) * rollout::kPhys, 0.0);
    const int a = once ? 0 : from, b = once ? 1 : r.frames;
    for (int s = a; s < b; ++s) {
      const float* st = r.coarse.data() + sz(s) * sz(kRes) * kRes * rollout::kPhys;
      for (int y = 0; y < kRes; ++y) {
        for (int x = 0; x < kRes; ++x) {
          for (int q = 0; q < rollout::kPhys; ++q) {
            m[((sz(y / block) * sz(G)) + sz(x / block)) * rollout::kPhys + sz(q)] += static_cast<double>(st[(sz(y) * kRes + sz(x)) * rollout::kPhys + sz(q)]) / (block * block * (b - a));
          }
        }
      }
    }
    out.push_back(std::move(m));
  }
  return out;
}

void step_fitmean(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto runs = training_runs(c, e, c.threads);
  std::println("fitmean: {} {} runs ready ({:.0f} s)", ename(e), runs.size(), seconds_since(t0));
  const bool once = !sim::effect_loops(e);
  const int block = a.i("block", 2), from = a.i("from", 89), G = kRes / block, cols = G * G * rollout::kPhys;
  const auto Y = run_means(runs, once, from, block);
  const int n = static_cast<int>(runs.size());
  // ridge regression on the quadratic features, lambda chosen by 5-fold cross-validation
  const auto fit = [&](const std::vector<int>& use, double lambda) {
    std::vector<double> A(sz(kFeat) * kFeat, 0.0), B(sz(kFeat) * sz(cols), 0.0);
    for (const int i : use) {
      const auto f = features(runs[sz(i)].controls());
      for (int p = 0; p < kFeat; ++p) {
        for (int q = 0; q < kFeat; ++q) A[sz(p) * kFeat + sz(q)] += f[sz(p)] * f[sz(q)];
        for (int k = 0; k < cols; ++k) B[sz(p) * sz(cols) + sz(k)] += f[sz(p)] * Y[sz(i)][sz(k)];
      }
    }
    return solve_ridge(A, B, kFeat, cols, lambda * static_cast<double>(use.size()));
  };
  const auto predict = [&](const std::vector<double>& beta, std::span<const float> ctl, int k) {
    const auto f = features(ctl);
    double v = 0;
    for (int p = 0; p < kFeat; ++p) v += f[sz(p)] * beta[sz(p) * sz(cols) + sz(k)];
    return v;
  };
  // per channel: the cross-validated residual of the fit and of the plain mean over runs, relative to the runs' own spread
  double best_lambda = 0, best_err = 1e30;
  for (const double lambda : {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}) {
    double err = 0;
    for (int fold = 0; fold < 5; ++fold) {
      std::vector<int> tr;
      for (int i = 0; i < n; ++i) {
        if (i % 5 != fold) tr.push_back(i);
      }
      const auto beta = fit(tr, lambda);
      for (int i = fold; i < n; i += 5) {
        for (int k = 0; k < cols; ++k) {
          const double d = predict(beta, runs[sz(i)].controls(), k) - Y[sz(i)][sz(k)];
          err += d * d;
        }
      }
    }
    std::println("fitmean: lambda {:.0e}: cross-validated squared error {:.5g}", lambda, err);
    if (err < best_err) {
      best_err = err;
      best_lambda = lambda;
    }
  }
  const auto beta = fit([&] {
    std::vector<int> v(sz(n));
    std::iota(v.begin(), v.end(), 0);
    return v;
  }(), best_lambda);
  // how much of the runs' variation over the control space the fit explains, per channel (cross-validated at best lambda)
  std::array<double, rollout::kPhys> res{}, tot{};
  {
    std::vector<double> mean(sz(cols), 0.0);
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < cols; ++k) mean[sz(k)] += Y[sz(i)][sz(k)] / n;
    }
    for (int fold = 0; fold < 5; ++fold) {
      std::vector<int> tr;
      for (int i = 0; i < n; ++i) {
        if (i % 5 != fold) tr.push_back(i);
      }
      const auto b2 = fit(tr, best_lambda);
      for (int i = fold; i < n; i += 5) {
        for (int k = 0; k < cols; ++k) {
          const double d = predict(b2, runs[sz(i)].controls(), k) - Y[sz(i)][sz(k)], d0 = mean[sz(k)] - Y[sz(i)][sz(k)];
          res[sz(k % rollout::kPhys)] += d * d;
          tot[sz(k % rollout::kPhys)] += d0 * d0;
        }
      }
    }
  }
  std::println("fitmean: {} lambda {:.0e}; share of the runs' ordered variation explained (cross-validated): u {:.3f} v {:.3f} heat {:.3f} soot {:.3f}", ename(e),
               best_lambda, 1 - res[0] / tot[0], 1 - res[1] / tot[1], 1 - res[2] / tot[2], 1 - res[3] / tot[3]);
  const fs::path out = c.dir() / "meanstate" / std::format("{}_b{}.txt", ename(e), block);
  fs::create_directories(out.parent_path());
  std::ofstream o(out);
  o << std::format("{} {} {} {} {}\n", ename(e), block, kFeat, cols, best_lambda);
  for (const double v : beta) o << std::format("{:.9g}\n", v);
  std::println("fitmean: -> {}", out.string());
}

// A fitted mean state (fitmean's file), as a function of the controls at 32 cells (bilinear from its blocks).
struct MeanState {
  int block = 2, cols = 0;
  std::vector<double> beta;
  bool empty() const { return beta.empty(); }
  // kRes^2 x kPhys at controls ctl
  std::vector<float> at(std::span<const float> ctl) const {
    const int G = kRes / block;
    const auto f = features(ctl);
    std::vector<double> g(sz(cols), 0.0);
    for (int k = 0; k < cols; ++k) {
      for (int p = 0; p < kFeat; ++p) g[sz(k)] += f[sz(p)] * beta[sz(p) * sz(cols) + sz(k)];
    }
    std::vector<float> out(sz(kRes) * kRes * rollout::kPhys);
    for (int y = 0; y < kRes; ++y) {
      for (int x = 0; x < kRes; ++x) {
        const float xs = std::clamp((static_cast<float>(x) + 0.5f) / static_cast<float>(block) - 0.5f, 0.f, static_cast<float>(G - 1));
        const float ys = std::clamp((static_cast<float>(y) + 0.5f) / static_cast<float>(block) - 0.5f, 0.f, static_cast<float>(G - 1));
        const int x0 = std::min(static_cast<int>(xs), G - 2), y0 = std::min(static_cast<int>(ys), G - 2);
        const double fx = xs - static_cast<float>(x0), fy = ys - static_cast<float>(y0);
        for (int q = 0; q < rollout::kPhys; ++q) {
          const auto v = [&](int xx, int yy) { return g[(sz(yy) * sz(G) + sz(xx)) * rollout::kPhys + sz(q)]; };
          out[(sz(y) * kRes + sz(x)) * rollout::kPhys + sz(q)] =
              static_cast<float>((1 - fy) * ((1 - fx) * v(x0, y0) + fx * v(x0 + 1, y0)) + fy * ((1 - fx) * v(x0, y0 + 1) + fx * v(x0 + 1, y0 + 1)));
        }
      }
    }
    return out;
  }
};
MeanState load_meanstate(const fs::path& p) {
  MeanState m;
  std::ifstream i(p);
  if (!i) throw std::runtime_error("cannot read " + p.string());
  std::string name;
  int feat = 0;
  double lambda = 0;
  i >> name >> m.block >> feat >> m.cols >> lambda;
  if (feat != kFeat) throw std::runtime_error("meanstate: wrong feature count");
  m.beta.resize(sz(kFeat) * sz(m.cols));
  for (double& v : m.beta) i >> v;
  if (!i) throw std::runtime_error("meanstate: truncated " + p.string());
  return m;
}
// A start state moved to other controls by the mean state: s + mu(to) - mu(from), heat and soot kept within the model's range.
void shift_start(const rollout::Model& M, const MeanState& ms, std::vector<float>& coarse, std::span<const float> from, std::span<const float> to) {
  const auto a = ms.at(to), b = ms.at(from);
  for (std::size_t i = 0; i < coarse.size(); ++i) {
    const int q = static_cast<int>(i % rollout::kPhys);
    coarse[i] = std::clamp(coarse[i] + a[i] - b[i], M.lo[sz(q)], M.hi[sz(q)]);
  }
}

// --- setmean: a model file with a fitted mean state (study M's candidate files) --------------------------------------------------

// A model file with other constants: "k=v;k=v" (make_variant's keys).
void step_set(const tools::Args& a) {
  const Variant v = make_variant(load_or_throw(a.need("model")), "set:" + a.need("values"));
  if (auto w = rollout::save_model(a.need("out"), v.m); !w) throw std::runtime_error(w.error());
  std::println("set: {} with {} -> {}", a.need("model"), a.need("values"), a.need("out"));
}

void step_setmean(const tools::Args& a) {
  rollout::Model m = load_or_throw(a.need("model"));
  const MeanState ms = load_meanstate(a.need("meanstate"));
  if (m.h.res != kRes || m.h.n_controls != 3) throw std::invalid_argument("setmean: a 32-cell model with three controls");
  m.mean.block = ms.block;
  m.mean.beta.assign(ms.beta.begin(), ms.beta.end());
  rollout::quantise_like_storage(m);  // as stored (fp16)
  if (auto w = rollout::save_model(a.need("out"), m); !w) throw std::runtime_error(w.error());
  std::println("setmean: {} with the mean state of {} (blocks of {} cells, {} coefficients) -> {} ({} bytes stored)", a.need("model"), a.need("meanstate"), ms.block,
               m.mean.beta.size(), a.need("out"), m.storage_bytes());
}

// --- fitbias: a correction of the stepper's ordered bias (training runs' controls) ------------------------------------------

// The model's own mean coarse state at controls: coarse-only reference rollouts (the coarse step does not read the fine
// fields) from the start points the runtime would use (the three nearest, by rep; moved by the mean state if the model has
// one), `frames` frames, the mean over frames [skip, frames), averaged over blocks of `block` cells as fitmean's.
std::vector<double> model_mean(const rollout::Model& M, std::span<const float> ctl, int reps, int frames, int skip, int block, std::uint64_t seed0) {
  const int G = kRes / block, C = M.h.channels(), N = kRes * kRes;
  std::vector<double> out(sz(G) * sz(G) * rollout::kPhys, 0.0);
  const auto near = nearest_starts(M, ctl);
  std::vector<float> S(sz(N) * sz(C)), next(S.size()), pressure(sz(N)), flow(sz(N) * 2), cond(sz(M.h.cond())), noise(sz(N) * rollout::kNoise),
      c0(sz(N) * rollout::kPhys);
  for (int r = 0; r < reps; ++r) {
    const int idx = near[sz(r) % near.size()];
    rollout::start_coarse(M, idx, ctl, c0);
    std::ranges::fill(S, 0.f);
    for (int i = 0; i < N; ++i) {
      for (int q = 0; q < rollout::kPhys; ++q) S[sz(i) * sz(C) + sz(q)] = c0[sz(i) * rollout::kPhys + sz(q)];
    }
    std::ranges::fill(pressure, 0.f);
    float t = M.starts[sz(idx)].time;
    const std::uint64_t seed = seed0 + static_cast<std::uint64_t>(r);
    for (int f = 0; f < frames; ++f) {
      rollout::condition(M, ctl, t, cond);
      rollout::coarse_noise(M, seed, t, noise);
      rollout::coarse_step(M, S, noise, cond, pressure, next, flow);
      S.swap(next);
      t += 1.f / M.fps;
      if (f < skip) continue;
      for (int y = 0; y < kRes; ++y) {
        for (int x = 0; x < kRes; ++x) {
          for (int q = 0; q < rollout::kPhys; ++q) {
            out[(sz(y / block) * sz(G) + sz(x / block)) * rollout::kPhys + sz(q)] +=
                static_cast<double>(S[(sz(y) * kRes + sz(x)) * sz(C) + sz(q)]) / (block * block * (frames - skip) * reps);
          }
        }
      }
    }
  }
  return out;
}

void step_fitbias(const Ctx& c, sim::Effect e, const tools::Args& a) {
  rollout::Model M = load_or_throw(a.need("model"));
  const MeanState target = load_meanstate(a.need("meanstate"));
  const int n_runs = a.i("runs", 80), reps = a.i("reps", 2), frames = a.i("frames", 180), skip = a.i("skip", 30), iters = a.i("iters", 4);
  const int block = target.block, G = kRes / block, cols = G * G * rollout::kPhys;
  if (!sim::effect_loops(e)) throw std::invalid_argument("fitbias: looping effects (a one-shot's mean changes with its age)");
  const auto runs = training_runs(c, e, c.threads);  // their controls (cached runs)
  std::vector<std::vector<float>> ctls;
  for (int j = 0; j < n_runs && j < static_cast<int>(runs.size()); ++j) ctls.push_back(runs[sz(j)].controls());
  // the quadratic fit of the model's means over the controls (ridge, the target's lambda)
  const auto fit = [&](const std::vector<std::vector<double>>& Y) {
    std::vector<double> A(sz(kFeat) * kFeat, 0.0), B(sz(kFeat) * sz(cols), 0.0);
    for (std::size_t i = 0; i < Y.size(); ++i) {
      const auto f = features(ctls[i]);
      for (int p = 0; p < kFeat; ++p) {
        for (int q = 0; q < kFeat; ++q) A[sz(p) * kFeat + sz(q)] += f[sz(p)] * f[sz(q)];
        for (int k = 0; k < cols; ++k) B[sz(p) * sz(cols) + sz(k)] += f[sz(p)] * Y[i][sz(k)];
      }
    }
    return solve_ridge(A, B, kFeat, cols, 1e-4 * static_cast<double>(Y.size()));
  };
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<double> bias(sz(kFeat) * sz(cols), 0.0), prev_bias, prev_model;
  if (!M.bias.empty()) {
    if (M.bias.block != block) throw std::invalid_argument("fitbias: the model's bias has other blocks than the target");
    bias.assign(M.bias.beta.begin(), M.bias.beta.end());
  }
  std::array<double, rollout::kPhys> gain{};  // per channel: change of the model's mean per unit of bias (secant)
  const std::array<double, rollout::kPhys> gain0{a.f("gain-u", 20.f), a.f("gain-v", 20.f), a.f("gain-heat", 20.f), a.f("gain-soot", 40.f)};
  for (int it = 0; it <= iters; ++it) {
    M.bias.block = block;
    M.bias.beta.assign(bias.begin(), bias.end());
    std::vector<std::vector<double>> Y(ctls.size());
    parallel(c.threads, static_cast<int>(ctls.size()), [&](int j) {
      Y[sz(j)] = model_mean(M, ctls[sz(j)], reps, frames, skip, block, 7'500'000 + 10 * static_cast<std::uint64_t>(j));
    });
    const auto beta_model = fit(Y);
    // the error of the model's mean against the target, over the training controls, per channel
    std::array<double, rollout::kPhys> err{}, tot{};
    for (std::size_t j = 0; j < ctls.size(); ++j) {
      const auto f = features(ctls[j]);
      for (int k = 0; k < cols; ++k) {
        double tv = 0, mv = 0;
        for (int p = 0; p < kFeat; ++p) {
          tv += f[sz(p)] * target.beta[sz(p) * sz(cols) + sz(k)];
          mv += f[sz(p)] * beta_model[sz(p) * sz(cols) + sz(k)];
        }
        err[sz(k % rollout::kPhys)] += (mv - tv) * (mv - tv);
        tot[sz(k % rollout::kPhys)] += tv * tv;
      }
    }
    std::println("fitbias: {} iteration {}: relative error of the model's mean against the training runs': u {:.4f} v {:.4f} heat {:.4f} soot {:.4f} ({:.0f} s)",
                 ename(e), it, std::sqrt(err[0] / tot[0]), std::sqrt(err[1] / tot[1]), std::sqrt(err[2] / tot[2]), std::sqrt(err[3] / tot[3]), seconds_since(t0));
    std::fflush(stdout);
    if (it == iters) break;
    if (!prev_model.empty()) {  // secant: how much the model's mean moved per unit of bias added, per channel
      for (int q = 0; q < rollout::kPhys; ++q) {
        double num = 0, den = 0;
        for (std::size_t k = static_cast<std::size_t>(q); k < bias.size(); k += rollout::kPhys) {
          const double db = bias[k] - prev_bias[k], dm = beta_model[k] - prev_model[k];
          num += dm * db;
          den += db * db;
        }
        gain[sz(q)] = den > 0 && num > 0 ? std::clamp(num / den, 0.25 * gain0[sz(q)], 4.0 * gain0[sz(q)]) : gain0[sz(q)];
      }
    } else {
      gain = gain0;
    }
    prev_bias = bias;
    prev_model = beta_model;
    const double damp = a.f("damp", 0.7f);
    for (std::size_t k = 0; k < bias.size(); ++k) bias[k] += damp * (target.beta[k] - beta_model[k]) / gain[k % rollout::kPhys];
    std::println("fitbias: gains u {:.1f} v {:.1f} heat {:.1f} soot {:.1f}", gain[0], gain[1], gain[2], gain[3]);
  }
  rollout::quantise_like_storage(M);
  if (auto w = rollout::save_model(a.need("out"), M); !w) throw std::runtime_error(w.error());
  std::println("fitbias: -> {}", a.need("out"));
}

// --- train: the stepper fine-tuned with the ensemble-mean loss (training runs) ----------------------------------------------

void step_train(const Ctx& c, sim::Effect e, const tools::Args& a) {
  rollout::Model m = load_or_throw(a.need("model"));
  const fs::path out = a.need("out");
  const auto t0 = std::chrono::steady_clock::now();
  const auto runs = training_runs(c, e, c.threads);
  std::println("train: {} {} runs ready ({:.0f} s)", ename(e), runs.size(), seconds_since(t0));
  rollout::StepperOptions so = rollout::recipe_for(e).stepper;
  so.iterations = 0;
  so.finetune = 0;
  so.activity_stage = 0;
  so.mean_stage = a.i("mean-stage", 600);
  so.mean_weight = a.f("mean-weight", 4.f);
  so.mean_burn_min = a.i("mean-burn-min", 0);
  so.mean_burn_max = a.i("mean-burn-max", 90);
  so.mean_pool = a.i("mean-pool", 4);
  so.activity = a.f("activity", 100.f);
  so.max_unroll = a.i("unroll", 16);
  so.lr_finetune = a.f("lr-ft", 3e-4f);
  so.seed = a.u64("seed", 41);
  so.keep_normalisation = true;
  so.floor_gradient = a.flag("floor-gradient");  // study L's vortex was trained with it (REPORT §12.1)
  so.threads = c.threads;
  so.log_every = 25;
  so.progress = [&](int it, int unroll, double loss) {
    std::println("  {} it {:5d} unroll {:2d} loss {:.5f} ({:.0f} s)", ename(e), it, unroll, loss, seconds_since(t0));
    std::fflush(stdout);
  };
  so.checkpoint_every = a.i("checkpoint-every", 100);
  so.checkpoint = [&](int done) {
    rollout::Model ck = m;
    rollout::quantise_like_storage(ck);
    if (auto w = rollout::save_model(out.string() + std::format(".it{}", done), ck); !w) throw std::runtime_error(w.error());
  };
  const rollout::StepperResult r = rollout::train_stepper(m, runs, so);
  rollout::quantise_like_storage(m);
  if (auto w = rollout::save_model(out, m); !w) throw std::runtime_error(w.error());
  std::println("train: {} -> {}: loss {:.5f} in {:.0f} s", ename(e), out.string(), r.final_loss, r.seconds);
}

// --- sstart: start points moved to the controls by the mean state (training settings) ------------------------------------------

// The diagnosis's neural_one_lr (one continuous reference rollout from one of the three nearest start points, the same index
// and seed), with the start state moved to the setting's controls by a fitted mean state: ss_one_lr.
void step_sstart(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const fs::path model_path = a.str("model", (c.data / "v2" / "models" / std::format("{}.nvfx", ename(e))).string());
  const rollout::Model M = load_or_throw(model_path);
  const MeanState ms = load_meanstate(a.need("meanstate"));
  const std::string tag = a.str("tag", "v2"), name = a.str("name", "ss");
  const int n_set = a.i("settings", 8), reps = a.i("reps", 4);
  const auto settings = training_settings(e, n_set);
  const std::uint64_t base_a = 7'100'000, base_b = 7'100'050, base_play = 7'200'000;  // as diag
  const auto seed = [](std::uint64_t base, int si, int k) { return base + 100 * static_cast<std::uint64_t>(si) + static_cast<std::uint64_t>(k); };
  const fs::path dir = c.dir() / "diag" / tag / ename(e);
  const std::string method = name + "_one_lr", hash = short_hash(a.need("meanstate"));
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::vector<std::map<std::string, Rec>>> runs(sz(n_set), std::vector<std::map<std::string, Rec>>(sz(reps)));
  parallel(c.threads, n_set * reps, [&](int j) {
    const int si = j / reps, k = j % reps;
    auto base = diag_run(c, e, M, tag, settings[sz(si)], seed(base_b, si, k), seed(base_play, si, k), k);  // cached by diag
    const fs::path f = dir / std::format("{}_{}_{}.rec", method, hash, seed(base_b, si, k));
    Rec r;
    if (!load_rec(f, r)) {
      const std::vector<float> ctl(settings[sz(si)].begin(), settings[sz(si)].end());
      const auto near = nearest_starts(M, ctl);
      const int idx = near[sz(k) % near.size()];
      rollout::Model mm = M;
      shift_start(M, ms, mm.starts[sz(idx)].coarse, mm.starts[sz(idx)].controls, ctl);
      const std::uint64_t sd = seed(base_play, si, k) + 50;
      rollout::State rs = rollout::start(mm, idx, kSize, ctl, sd);
      const int F = frames_of(e), C = M.h.channels();
      const std::size_t NC = sz(kRes) * kRes * rollout::kPhys;
      std::vector<double> cm(NC, 0.0);
      std::vector<float> rgba(sz(kSize) * kSize * 4);
      Clip cl;
      cl.allocate(kSize, F);
      cl.fps = 30.f;
      for (int i = 0; i < F; ++i) {
        rollout::step(mm, rs, ctl, sd);
        for (int q = 0; q < kRes * kRes; ++q) {
          for (int ch = 0; ch < rollout::kPhys; ++ch) cm[sz(q) * rollout::kPhys + sz(ch)] += static_cast<double>(rs.coarse[sz(q) * sz(C) + sz(ch)]) / F;
        }
        rollout::render(mm, rs, rgba);
        to_bytes(rgba, cl.frame(i));
      }
      r = make_rec(cl, std::move(cm));
      save_rec(f, r);
    }
    base[method] = std::move(r);
    runs[sz(si)][sz(k)] = std::move(base);
    std::println("sstart: {} setting {} run {} done ({:.0f} s)", ename(e), si, k, seconds_since(t0));
    std::fflush(stdout);
  });
  const std::vector<std::string> methods = {"real_B", "neural_true_lr", "neural_one_lr", method, "neural_shards"};
  std::vector<Rec> ref(sz(n_set));
  std::vector<std::map<std::string, Rec>> pooled(sz(n_set));
  for (int si = 0; si < n_set; ++si) {
    std::vector<Rec> v;
    for (int k = 0; k < reps; ++k) v.push_back(real_rec(c, e, settings[sz(si)], seed(base_a, si, k)));
    ref[sz(si)] = pool_recs(v);
    for (const auto& m : methods) {
      std::vector<Rec> w;
      for (int k = 0; k < reps; ++k) w.push_back(runs[sz(si)][sz(k)].at(m));
      pooled[sz(si)][m] = pool_recs(w);
    }
  }
  diag_tables(ename(e), tag, reps, ref, pooled, methods, {"real_B", "neural_true_lr", "neural_one_lr", method},
              c.results / std::format("m_sstart_{}_{}_{}.csv", ename(e), tag, name));
}

// --- where: where the average picture differs (cached records only) ----------------------------------------------------------

// Per method, from the diagnosis's cached records (pooled per setting): the mean frame's error against the A runs' by band of
// rows (16 bands of 8 rows, top to bottom; RMS over RGBA and the signed difference of alpha and of luminance), the share of
// the mean frame's squared error in its smooth part (the error blurred by 9 x 9 boxes: an error the size of a coarse cell or
// more) and in the rest, and the mean frame's and the fluctuations' log power difference by band. With --figure, a sheet:
// per setting the A runs' mean frame, then each method's.
void step_where(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const std::string tag = a.str("tag", "v2");
  const int n_set = a.i("settings", 8), reps = a.i("reps", 4);
  const auto settings = training_settings(e, n_set);
  const std::vector<std::string> methods = split_list(a.str("methods", "real_B,rend_true,detail_sim,neural_true_sim,neural_true_lr,neural_one_lr,neural_shards"), ',');
  const std::uint64_t base_a = 7'100'000, base_b = 7'100'050;
  const auto seed = [](std::uint64_t base, int si, int k) { return base + 100 * static_cast<std::uint64_t>(si) + static_cast<std::uint64_t>(k); };
  const fs::path dir = c.dir() / "diag" / tag / ename(e);
  std::vector<Rec> ref(sz(n_set));
  std::vector<std::map<std::string, Rec>> pooled(sz(n_set));
  for (int si = 0; si < n_set; ++si) {
    std::vector<Rec> v;
    for (int k = 0; k < reps; ++k) v.push_back(real_rec(c, e, settings[sz(si)], seed(base_a, si, k)));
    ref[sz(si)] = pool_recs(v);
    for (const auto& m : methods) {
      std::vector<Rec> w;
      for (int k = 0; k < reps; ++k) {
        Rec r;
        if (!load_rec(dir / std::format("{}_{}.rec", m, seed(base_b, si, k)), r)) {
          throw std::runtime_error(std::format("where: no record {} for setting {} run {} (run diag first)", m, si, k));
        }
        w.push_back(std::move(r));
      }
      pooled[sz(si)][m] = pool_recs(w);
    }
  }
  constexpr int kBands = 16, kRows = kSize / kBands;
  std::println("{} ({}): the mean frame's error against the A runs', mean over {} settings", ename(e), tag, n_set);
  std::println("RMS error (x1000) by band of rows, top to bottom; then the smooth share of the squared error");
  for (const auto& m : methods) {
    std::array<double, kBands> rms{};
    double smooth = 0, total = 0;
    std::array<double, kBands> da{}, dl{};
    for (int si = 0; si < n_set; ++si) {
      const auto& x = pooled[sz(si)][m].st.mean_frame;
      const auto& r = ref[sz(si)].st.mean_frame;
      std::vector<double> err(sz(kSize) * kSize * 4);
      for (std::size_t i = 0; i < err.size(); ++i) err[i] = (static_cast<double>(x[i]) - r[i]) / 255.0;
      for (int y = 0; y < kSize; ++y) {
        for (int xx = 0; xx < kSize; ++xx) {
          const std::size_t p = (sz(y) * kSize + sz(xx)) * 4;
          double s2 = 0;
          for (int q = 0; q < 4; ++q) s2 += err[p + sz(q)] * err[p + sz(q)];
          rms[sz(y / kRows)] += s2 / (4.0 * kRows * kSize * n_set);
          da[sz(y / kRows)] += err[p + 3] / (static_cast<double>(kRows) * kSize * n_set);
          dl[sz(y / kRows)] += (0.2126 * err[p] + 0.7152 * err[p + 1] + 0.0722 * err[p + 2]) / (static_cast<double>(kRows) * kSize * n_set);
        }
      }
      // the smooth part: a 9 x 9 box blur of the error (zero outside)
      for (int q = 0; q < 4; ++q) {
        for (int y = 0; y < kSize; ++y) {
          for (int xx = 0; xx < kSize; ++xx) {
            double s = 0;
            for (int dy = -4; dy <= 4; ++dy) {
              for (int dx = -4; dx <= 4; ++dx) {
                const int yy = y + dy, x2 = xx + dx;
                if (yy >= 0 && yy < kSize && x2 >= 0 && x2 < kSize) s += err[(sz(yy) * kSize + sz(x2)) * 4 + sz(q)];
              }
            }
            s /= 81.0;
            const double e0 = err[(sz(y) * kSize + sz(xx)) * 4 + sz(q)];
            smooth += s * s;
            total += e0 * e0;
          }
        }
      }
    }
    std::string line = std::format("{:<16}", m);
    for (const double v : rms) line += std::format("{:6.2f}", 1000.0 * std::sqrt(v));
    std::println("{}  smooth {:.2f}", line, total > 0 ? smooth / total : 0.0);
    std::string la = std::format("{:<16}", "  alpha x1000"), ll = std::format("{:<16}", "  lum x1000");
    for (int b = 0; b < kBands; ++b) {
      la += std::format("{:+6.1f}", 1000.0 * da[sz(b)]);
      ll += std::format("{:+6.1f}", 1000.0 * dl[sz(b)]);
    }
    std::println("{}\n{}", la, ll);
  }
  std::println("\nlog10 power difference against the A runs', by band of radial bins: mean frame | fluctuations");
  std::println("{:<16}{:>8}{:>8}{:>8}{:>8} |{:>8}{:>8}{:>8}{:>8}", "method", "b1-4", "b5-16", "b17-32", "b33-64", "b1-4", "b5-16", "b17-32", "b33-64");
  for (const auto& m : methods) {
    std::array<double, 8> v{};
    for (int si = 0; si < n_set; ++si) {
      const Rec& x = pooled[sz(si)][m];
      const Rec& r = ref[sz(si)];
      for (int b = 0; b < 4; ++b) {
        v[sz(b)] += band(x.fl.spec_mean, r.fl.spec_mean, kBandEdges[sz(b)], kBandEdges[sz(b) + 1]) / n_set;
        v[sz(4 + b)] += band(x.fl.spec_fluct, r.fl.spec_fluct, kBandEdges[sz(b)], kBandEdges[sz(b) + 1]) / n_set;
      }
    }
    std::println("{:<16}{:>+8.3f}{:>+8.3f}{:>+8.3f}{:>+8.3f} |{:>+8.3f}{:>+8.3f}{:>+8.3f}{:>+8.3f}", m, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
  }
  if (a.has("figure")) {
    std::vector<Clip> rows;
    const auto clip_of = [&](const std::vector<float>& mf) {
      Clip cl;
      cl.allocate(kSize, 1);
      auto f = cl.frame(0);
      for (std::size_t i = 0; i < mf.size(); ++i) f[i] = static_cast<std::uint8_t>(std::clamp(mf[i], 0.f, 255.f) + 0.5f);
      return cl;
    };
    for (int si = 0; si < n_set; ++si) {
      Clip row;
      row.allocate(kSize, static_cast<int>(methods.size()) + 1);
      std::ranges::copy(clip_of(ref[sz(si)].st.mean_frame).frame(0), row.frame(0).begin());
      for (std::size_t q = 0; q < methods.size(); ++q) std::ranges::copy(clip_of(pooled[sz(si)][methods[q]].st.mean_frame).frame(0), row.frame(static_cast<int>(q) + 1).begin());
      rows.push_back(std::move(row));
    }
    std::vector<const Clip*> rp;
    for (const auto& r : rows) rp.push_back(&r);
    if (auto w = write_png(a.str("figure"), comparison_sheet(rp, static_cast<int>(methods.size()) + 1, Background::black, 1)); !w) throw std::runtime_error(w.error());
    std::println("figure: {}", a.str("figure"));
  }
}

// --- validation and test: the current model against candidates -------------------------------------------------------------

bool off_grid(const Setting& s) {
  const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
  return off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f});
}
// Study B's ten held-out settings (the test) and study G's ten validation settings (docs/DCM.md §4), as nvfx_d2. Study M3
// draws more the same way: the first n of the same streams, the validation settings at least 0.05 from every one of the
// n_test test settings (with n = n_test = 10, studies B's and G's own).
std::vector<Setting> test_settings(int n = 10) {
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < sz(n)) {
    const Setting s{u(rng), u(rng), u(rng)};
    if (off_grid(s)) v.push_back(s);
  }
  return v;
}
std::vector<Setting> validation_settings(int n = 10, int n_test = 10) {
  const auto test = test_settings(n_test);
  std::mt19937_64 rng(2027);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < sz(n)) {
    const Setting s{u(rng), u(rng), u(rng)};
    const bool far = std::ranges::all_of(test, [&](const Setting& t) {
      return std::sqrt((s[0] - t[0]) * (s[0] - t[0]) + (s[1] - t[1]) * (s[1] - t[1]) + (s[2] - t[2]) * (s[2] - t[2])) >= 0.05f;
    });
    if (off_grid(s) && far) v.push_back(s);
  }
  return v;
}

// Seeds. Per split, real run k of setting s has seed real + 100 s + k (k < 8), the floor's runs real + 50 + 100 s + k, the
// models' plays model + 100 s + k; the test's tracking runs track + 100 s + k (k < 4), each simulated from an empty grid
// (looping effects: 100 frames, then followed for 240; the explosion: 1 frame, then its 88).
//   probe  development only (training settings: the controls of salt-1 training runs 0 to 9; 4 runs pooled): real
//          7,300,000, plays 7,400,000, tracking 7,450,000 (2 runs per setting).
//   val    study G's validation settings, with study D2's seeds (real 4,300,000, plays 4,400,000), so that v2's numbers
//          are §6.9.2's; tracking: the studies' salt-3 validation runs 0 to 15.
//   test   study B's held-out settings, seeds no study has used: real 8,000,000, plays 8,100,000, tracking 8,200,000.
// Study M2 (docs/REPORT.md §14.8; its own range, 8,500,000 to 8,999,999; tracking runs k < 8 per setting, each from an empty
// grid as the test's):
//   probe2 development only (training settings, salt-1 runs 0 to 9; 4 runs pooled): real 8,900,000, plays 8,930,000,
//          tracking 8,960,000 (2 runs per setting).
//   val2   study G's validation settings: real 8,500,000, plays 8,600,000, tracking 8,650,000.
//   test2  study B's held-out settings: real 8,700,000, plays 8,800,000, tracking 8,850,000.
// Study M3 (docs/REPORT.md §14.9; the blocks 8,300,000 to 8,499,999 and 8,700,000 to 8,899,999 of study M's range; 8 runs
// pooled; tracking runs k < 2 per setting, each from an empty grid as the test's):
//   probe3 development only (training settings: the controls of salt-1 training runs 0 to kM3ProbeSettings - 1): real
//          8,300,000, plays 8,310,000, the null (the current model with fresh seeds) 8,320,000, tracking 8,330,000.
//   val3   study G's validation settings extended to kM3Settings (validation_settings(kM3Settings, kM3Settings)): real
//          8,350,000, plays 8,400,000, tracking 8,450,000.
//   test3  study B's held-out settings extended to kM3Settings (test_settings(kM3Settings)): real 8,700,000, plays
//          8,800,000, tracking 8,850,000 (study M2's test block, never used).
constexpr int kM3ProbeSettings = 40;
constexpr int kM3Settings = 0;  // fixed by the probe (REPORT §14.9.2) before validation; 0: not yet
struct Split {
  std::string name;
  std::vector<Setting> settings;
  int reps = 8, track_reps = 4;
  std::uint64_t real = 0, model = 0, track = 0, null = 0;  // null: the plays of the current model with fresh seeds (probe3)
  int track_salt = 0, track_runs = 0;  // > 0: the tracking runs are recipe runs 0 .. track_runs - 1 of this salt instead
};
Split split_of(const std::string& name, sim::Effect e) {
  Split s;
  s.name = name;
  if (name == "probe") {
    s.settings = training_settings(e, 10);
    s.reps = 4;
    s.track_reps = 2;
    s.real = 7'300'000;
    s.model = 7'400'000;
    s.track = 7'450'000;
  } else if (name == "val") {
    s.settings = validation_settings();
    s.real = 4'300'000;
    s.model = 4'400'000;
    s.track_salt = 3;
    s.track_runs = 16;
  } else if (name == "test") {
    s.settings = test_settings();
    s.real = 8'000'000;
    s.model = 8'100'000;
    s.track = 8'200'000;
  } else if (name == "probe2") {
    s.settings = training_settings(e, 10);
    s.reps = 4;
    s.track_reps = 2;
    s.real = 8'900'000;
    s.model = 8'930'000;
    s.track = 8'960'000;
  } else if (name == "val2") {
    s.settings = validation_settings();
    s.track_reps = 8;
    s.real = 8'500'000;
    s.model = 8'600'000;
    s.track = 8'650'000;
  } else if (name == "test2") {
    s.settings = test_settings();
    s.track_reps = 8;
    s.real = 8'700'000;
    s.model = 8'800'000;
    s.track = 8'850'000;
  } else if (name == "probe3") {
    s.settings = training_settings(e, kM3ProbeSettings);
    s.track_reps = 2;
    s.real = 8'300'000;
    s.model = 8'310'000;
    s.null = 8'320'000;
    s.track = 8'330'000;
  } else if (name == "val3" || name == "test3") {
    if (kM3Settings <= 0) throw std::invalid_argument("study M3: the probe has not fixed the number of settings (kM3Settings)");
    const bool val = name == "val3";
    s.settings = val ? validation_settings(kM3Settings, kM3Settings) : test_settings(kM3Settings);
    s.track_reps = 2;
    s.real = val ? 8'350'000 : 8'700'000;
    s.model = val ? 8'400'000 : 8'800'000;
    s.track = val ? 8'450'000 : 8'850'000;
  } else {
    throw std::invalid_argument("unknown split " + name);
  }
  return s;
}

struct Entry {
  std::string name;
  fs::path path;
  rollout::Model m;
  std::string sha;
  std::uint64_t seed_base = 0;  // > 0: its plays take this seed base instead of the split's (study M3's null)
};

std::string file_sha256(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + p.string());
  std::stringstream ss;
  ss << f.rdbuf();
  return dcm::sha256_hex(ss.str());
}

// "name=path,name=path": the entries to score; the first is the current model.
std::vector<Entry> read_entries(const std::string& current, const std::string& cands) {
  std::vector<Entry> v;
  v.push_back({"current", current, load_or_throw(current), file_sha256(current)});
  for (const auto& item : split_list(cands, ',')) {
    const auto eq = item.find('=');
    if (eq == std::string::npos) throw std::invalid_argument("--cand: name=path");
    const fs::path p = item.substr(eq + 1);
    v.push_back({item.substr(0, eq), p, load_or_throw(p), file_sha256(p)});
  }
  return v;
}

// Plain statistics of a run, cached (real runs under vt/real, the models' plays under vt/plays/<first 16 of the file's
// SHA-256>), so every step resumes and a model is never played twice with a seed. Study M2's caches (.fst) hold the mean
// frame as the floats it is (half the disk; the same numbers).
bool get_stats_f(std::istream& i, metrics::ClipStats& s) {
  std::uint32_t n = 0;
  if (!get_vec(i, s.coverage) || !get_vec(i, s.emission) || !get_vec(i, s.spectrum)) return false;
  i.read(reinterpret_cast<char*>(&n), 4);
  if (!i || n > 1000000) return false;
  s.mean_frame.resize(n);
  i.read(reinterpret_cast<char*>(s.mean_frame.data()), static_cast<std::streamsize>(n * sizeof(float)));
  i.read(reinterpret_cast<char*>(&s.motion), sizeof(double));
  return static_cast<bool>(i);
}
void put_stats_f(std::ostream& o, const metrics::ClipStats& s) {
  put_vec(o, s.coverage);
  put_vec(o, s.emission);
  put_vec(o, s.spectrum);
  const auto n = static_cast<std::uint32_t>(s.mean_frame.size());
  o.write(reinterpret_cast<const char*>(&n), 4);
  o.write(reinterpret_cast<const char*>(s.mean_frame.data()), static_cast<std::streamsize>(s.mean_frame.size() * sizeof(float)));
  o.write(reinterpret_cast<const char*>(&s.motion), sizeof(double));
}
bool load_stats(const fs::path& p, metrics::ClipStats& s) {
  std::ifstream i(p, std::ios::binary);
  return i && (p.extension() == ".fst" ? get_stats_f(i, s) : get_stats(i, s));
}
void save_stats(const fs::path& p, const metrics::ClipStats& s) {
  fs::create_directories(p.parent_path());
  const fs::path tmp = p.string() + ".tmp";
  {
    std::ofstream o(tmp, std::ios::binary);
    if (p.extension() == ".fst") put_stats_f(o, s);
    else put_stats(o, s);
  }
  fs::rename(tmp, p);
}
std::string stats_ext(const Ctx& c) { return c.m2 || c.m3 ? "fst" : "stats"; }
metrics::ClipStats real_stats(const Ctx& c, sim::Effect e, const Setting& s, std::uint64_t seed) {
  const fs::path p = c.dir() / "vt" / "real" / std::format("{}_{}.{}", ename(e), seed, stats_ext(c));
  metrics::ClipStats st;
  if (load_stats(p, st)) return st;
  sim::Fluid f(setting_params(e, s, seed));
  for (int i = 0; i < warm_of(e); ++i) f.step_frame();
  Clip cl;
  cl.allocate(kSize, frames_of(e));
  cl.fps = 30.f;
  for (int i = 0; i < cl.frames; ++i) {
    f.step_frame();
    f.render(cl.frame(i));
  }
  st = metrics::stats(cl);
  save_stats(p, st);
  return st;
}
metrics::ClipStats play_stats(const Ctx& c, sim::Effect e, const Entry& m, nvfx_effect* fx, const Setting& s, std::uint64_t seed) {
  const fs::path p = c.dir() / "vt" / "plays" / m.sha.substr(0, 16) / std::format("{}_{}.{}", ename(e), seed, stats_ext(c));
  metrics::ClipStats st;
  if (load_stats(p, st)) return st;
  const std::vector<float> ctl(s.begin(), s.end());
  st = metrics::stats(runtime_clip(fx, ctl, -1, seed, frames_of(e)));
  save_stats(p, st);
  return st;
}

double apsnr(std::span<const std::uint8_t> r, std::span<const std::uint8_t> t) {  // study D's active PSNR of one frame
  double sum = 0, n = 0;
  for (std::size_t i = 0; i < r.size(); i += 4) {
    if (std::max({r[i], r[i + 1], r[i + 2], r[i + 3]}) <= 4 && std::max({t[i], t[i + 1], t[i + 2], t[i + 3]}) <= 4) continue;
    n += 4;
    for (std::size_t c = 0; c < 4; ++c) {
      const double d = (static_cast<double>(r[i + c]) - static_cast<double>(t[i + c])) / 255.0;
      sum += d * d;
    }
  }
  return n > 0 ? metrics::psnr_from_mse(sum / n) : metrics::kPsnrCap;
}

constexpr std::array kTrackH = {1, 8, 30, 60};
constexpr int kTrack1s = 2;  // index of 30 frames (1 s)
const std::array<std::string, 5> kStatNames = {"spectrum_l1", "abs_ln_motion_ratio", "coverage_l1", "emission_l1", "mean_frame_psnr"};

struct Scores {
  std::vector<std::array<double, kTrackH.size()>> track;  // per tracking unit: mean over its runs
  std::vector<metrics::StatDistance> d, d1;               // per setting: pooled plays against pooled real; first against first
};
double stat_value(const metrics::StatDistance& x, int k) {
  switch (k) {
    case 0: return x.spectrum_l1;
    case 1: return std::abs(std::log(std::max(1e-3, x.motion_ratio)));
    case 2: return x.coverage_l1;
    case 3: return x.emission_l1;
    default: return x.mean_frame_psnr;
  }
}

// Tracking (the runtime from the true state, one continuous rollout, as study D's d-eval) and the endless statistics (the
// runtime as shipped, pooled over runs) of every entry on a split; the last Scores is the floor (real, other seeds).
std::vector<Scores> evaluate(const Ctx& c, sim::Effect e, const Split& sp, const std::vector<Entry>& models) {
  const bool once = !sim::effect_loops(e);
  const int n = static_cast<int>(sp.settings.size()), KE = once ? 88 : 60, warm_t = once ? 1 : 100;
  const int nt = sp.track_salt > 0 ? sp.track_runs : n, per = sp.track_salt > 0 ? 1 : sp.track_reps;
  std::vector<Scores> out(models.size());
  for (auto& s : out) {
    s.track.assign(sz(nt), {});
    s.d.resize(sz(n));
    s.d1.resize(sz(n));
  }
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::vector<std::vector<std::array<double, kTrackH.size()>>>> tr(models.size(),
                                                                               std::vector<std::vector<std::array<double, kTrackH.size()>>>(sz(nt)));
  std::mutex mu;
  parallel(c.threads, nt * per, [&](int j) {
    const int si = j / per, k = j % per;
    const sim::Params p = sp.track_salt > 0 ? run_params(e, static_cast<std::uint64_t>(sp.track_salt), static_cast<std::uint64_t>(si))
                                            : setting_params(e, sp.settings[sz(si)], sp.track + 100 * static_cast<std::uint64_t>(si) + static_cast<std::uint64_t>(k));
    const std::vector<float> ctl{p.intensity, p.wind, p.turbulence};
    sim::Fluid truth(p);
    for (int i = 0; i < warm_t; ++i) truth.step_frame();
    const sim::State st = truth.state();
    Clip ref;
    ref.allocate(kSize, KE);
    for (int i = 0; i < KE; ++i) {
      truth.step_frame();
      truth.render(ref.frame(i));
    }
    rollout::StartPoint s0;
    s0.controls = ctl;
    s0.seed = p.seed;
    s0.time = st.time;
    s0.coarse.resize(sz(kRes) * kRes * rollout::kPhys);
    rollout::coarse_from_sim(st, kRes, p.fps, s0.coarse);
    s0.fine_t = st.temp;
    s0.fine_d = st.soot;
    for (std::size_t mi = 0; mi < models.size(); ++mi) {
      if (models[mi].seed_base > 0) continue;  // the null: tracking takes the run's own seed, so it is the current model's
      rollout::Model mt = models[mi].m;
      mt.h.start_fine = kSize;
      mt.starts = {s0};
      Runtime fx(mt);
      const Clip nc = runtime_clip(fx.e, ctl, 0, 0, KE + 1, false);  // frame 0: the start
      std::array<double, kTrackH.size()> a{};
      for (std::size_t h = 0; h < kTrackH.size(); ++h) a[h] = apsnr(ref.frame(kTrackH[h] - 1), nc.frame(kTrackH[h]));
      std::lock_guard lk(mu);
      tr[mi][sz(si)].push_back(a);
    }
    std::println("{}: {} tracking {} run {} done ({:.0f} s)", sp.name, ename(e), si, k, seconds_since(t0));
    std::fflush(stdout);
  });
  for (std::size_t mi = 0; mi < models.size(); ++mi) {
    for (int si = 0; si < nt; ++si) {
      for (std::size_t h = 0; h < kTrackH.size(); ++h) {
        double s1 = 0;
        for (const auto& v : tr[mi][sz(si)]) s1 += v[h] / static_cast<double>(tr[mi][sz(si)].size());
        out[mi].track[sz(si)][h] = s1;
      }
    }
    if (models[mi].seed_base > 0) out[mi].track = out[0].track;
  }
  const auto seed = [&](std::uint64_t base, int si, int k) { return base + 100 * static_cast<std::uint64_t>(si) + static_cast<std::uint64_t>(k); };
  parallel(c.threads, n * sp.reps * 2, [&](int j) {
    const int si = j % n, k = (j / n) % sp.reps, o = j / (n * sp.reps);
    (void)real_stats(c, e, sp.settings[sz(si)], seed(sp.real + (o ? 50 : 0), si, k));
  });
  std::println("{}: {} real runs ready ({:.0f} s)", sp.name, ename(e), seconds_since(t0));
  std::vector<metrics::ClipStats> ref(sz(n)), other(sz(n)), ref1(sz(n));
  for (int si = 0; si < n; ++si) {
    for (int o = 0; o < 2; ++o) {
      std::vector<metrics::ClipStats> v;
      for (int k = 0; k < sp.reps; ++k) v.push_back(real_stats(c, e, sp.settings[sz(si)], seed(sp.real + (o ? 50 : 0), si, k)));
      (o ? other : ref)[sz(si)] = pool(v);
      if (!o) ref1[sz(si)] = v[0];
    }
  }
  for (std::size_t mi = 0; mi < models.size(); ++mi) {
    Runtime rt(models[mi].m);
    std::vector<metrics::ClipStats> runs(sz(n * sp.reps));
    parallel(c.threads, n * sp.reps, [&](int j) {
      const int si = j / sp.reps, k = j % sp.reps;
      runs[sz(j)] = play_stats(c, e, models[mi], rt.e, sp.settings[sz(si)], seed(models[mi].seed_base > 0 ? models[mi].seed_base : sp.model, si, k));
    });
    for (int si = 0; si < n; ++si) {
      const metrics::ClipStats pl = pool(std::vector<metrics::ClipStats>(runs.begin() + si * sp.reps, runs.begin() + (si + 1) * sp.reps));
      out[mi].d[sz(si)] = metrics::distance(ref[sz(si)], pl);
      out[mi].d1[sz(si)] = metrics::distance(ref1[sz(si)], runs[sz(si * sp.reps)]);
    }
    std::println("{}: {} {} played ({:.0f} s)", sp.name, ename(e), models[mi].name, seconds_since(t0));
    std::fflush(stdout);
  }
  Scores floor;
  for (int si = 0; si < n; ++si) floor.d.push_back(metrics::distance(ref[sz(si)], other[sz(si)]));
  out.push_back(floor);
  return out;
}

// The statistics in the order the comparisons use: tracking at 1 s, then the five endless ones (kStatNames).
const std::array<std::string, 6> kCmpNames = {"track_1s", "spectrum_l1", "abs_ln_motion_ratio", "coverage_l1", "emission_l1", "mean_frame_psnr"};
std::vector<double> cmp_stat(const Scores& s, int k) {  // tracking: per tracking unit; the rest: per setting
  std::vector<double> v;
  const std::size_t n = k == 0 ? s.track.size() : s.d.size();
  for (std::size_t i = 0; i < n; ++i) v.push_back(k == 0 ? s.track[i][kTrack1s] : stat_value(s.d[i], k - 1));
  return v;
}

void merge_rows(const fs::path& path, const std::string& header, const std::vector<std::string>& rows, const std::string& key) {
  std::vector<std::string> keep;
  if (std::ifstream in(path); in) {
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      if (!line.empty() && line.substr(0, line.find(',')) != key) keep.push_back(line);
    }
  }
  for (const auto& r : rows) keep.push_back(r);
  write_rows(path, header, keep);
}

// Means per model and every candidate minus the current model, paired (tracking over its units, the rest over the
// settings), 95% bootstrap with 10,000 resamples (metrics::paired_bootstrap, seed 2026). Rows to <prefix>_*.csv.
void report_split(const Ctx& c, sim::Effect e, const Split& sp, const std::vector<Entry>& models, const std::vector<Scores>& sc, const std::string& prefix) {
  const std::string en = ename(e);
  std::vector<std::string> track_rows, stat_rows, cmp_rows;
  const std::size_t n = sp.settings.size();
  std::println("\n{} {}: means over the tracking runs and the {} settings ({} runs pooled per setting)", sp.name, en, n, sp.reps);
  std::println("{:<22}{:>8}{:>8}{:>8}{:>8} |{:>9}{:>9}{:>9}{:>10}{:>10}{:>8}", "model", "f1", "f8", "1s", "2s", "spectrum", "|lnmot|", "motion", "cover", "emiss",
               "PSNR");
  for (std::size_t mi = 0; mi <= models.size(); ++mi) {
    const bool fl0 = mi == models.size();
    const std::string name = fl0 ? "real_other_seeds" : models[mi].name;
    std::array<double, kTrackH.size()> t{};
    std::array<double, 6> st{};
    const std::size_t nt = fl0 ? 0 : sc[mi].track.size();
    for (std::size_t si = 0; si < nt; ++si) {
      std::string row = std::format("{},{},{}", en, name, si);
      for (std::size_t h = 0; h < kTrackH.size(); ++h) {
        t[h] += sc[mi].track[si][h] / static_cast<double>(nt);
        row += std::format(",{:.3f}", sc[mi].track[si][h]);
      }
      track_rows.push_back(row);
    }
    for (std::size_t si = 0; si < n; ++si) {
      for (int k = 0; k < 5; ++k) st[sz(k)] += stat_value(sc[mi].d[si], k) / static_cast<double>(n);
      st[5] += sc[mi].d[si].motion_ratio / static_cast<double>(n);
      for (int pooled = 1; pooled >= (fl0 ? 1 : 0); --pooled) {
        const auto& x = pooled ? sc[mi].d[si] : sc[mi].d1[si];
        stat_rows.push_back(std::format("{},{},{},{},{:.5f},{:.5f},{:.5f},{:.6f},{:.6f},{:.4f}", en, name, pooled ? std::format("pooled{}", sp.reps) : "one", si,
                                        x.spectrum_l1, x.motion_ratio, std::abs(std::log(std::max(1e-3, x.motion_ratio))), x.coverage_l1, x.emission_l1,
                                        x.mean_frame_psnr));
      }
    }
    std::println("{:<22}{:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f} |{:>9.4f}{:>9.4f}{:>9.3f}{:>10.5f}{:>10.5f}{:>8.2f}", fl0 ? "real, other seeds" : name, t[0], t[1], t[2],
                 t[3], st[0], st[1], st[5], st[2], st[3], st[4]);
  }
  for (std::size_t mi = 1; mi < models.size(); ++mi) {
    std::string line = std::format("{} {} {} - current:", sp.name, en, models[mi].name);
    for (int k = 0; k < 6; ++k) {
      const auto a = cmp_stat(sc[mi], k), b = cmp_stat(sc[0], k);
      const auto iv = metrics::paired_bootstrap(a, b, 10000, 2026);
      line += std::format(" {} {:+.4f} [{:+.4f}, {:+.4f}];", kCmpNames[sz(k)], iv.mean, iv.lo, iv.hi);
      cmp_rows.push_back(std::format("{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{}", en, models[mi].name, kCmpNames[sz(k)], iv.mean, iv.lo, iv.hi,
                                     std::accumulate(a.begin(), a.end(), 0.0) / static_cast<double>(a.size()),
                                     std::accumulate(b.begin(), b.end(), 0.0) / static_cast<double>(b.size()), models[mi].sha));
    }
    std::println("{}", line);
  }
  merge_rows(c.results / std::format("{}_track.csv", prefix), "effect,model,unit,track_f1,track_f8,track_f30,track_f60", track_rows, en);
  merge_rows(c.results / std::format("{}_stats.csv", prefix),
             "effect,model,runs,setting,spectrum_l1,motion_ratio,abs_ln_motion_ratio,coverage_l1,emission_l1,mean_frame_psnr", stat_rows, en);
  merge_rows(c.results / std::format("{}_compare.csv", prefix), "effect,model,statistic,mean_difference,lo,hi,candidate_mean,current_mean,sha256", cmp_rows, en);
}

// The rule (docs/REPORT.md §14.3; committed before validation). Candidate minus current model, paired (tracking at 1 s
// over the tracking units, the endless statistics over the 10 settings, 8 runs pooled per setting), 95% bootstrap with
// 10,000 resamples (metrics::paired_bootstrap, seed 2026). A candidate meets the rule when
//   1. the mean-frame PSNR is better (interval above zero) and the detail spectrum distance is better (interval below
//      zero), and
//   2. it is not worse on tracking at 1 s (no interval below zero), |ln motion ratio|, coverage distance or emission
//      distance (no interval above zero).
// On validation, among the candidates that meet it, the one with the largest mean-frame gain is chosen; an effect with none
// stops there. The test runs once, on the chosen candidate frozen by SHA-256.
struct RuleM {
  std::array<metrics::Interval, 6> iv{};  // kCmpNames' order
  bool better = false, none_worse = false;
  bool pass() const { return better && none_worse; }
};
RuleM apply_rule_m(const Scores& cand, const Scores& cur) {
  RuleM r;
  for (int k = 0; k < 6; ++k) r.iv[sz(k)] = metrics::paired_bootstrap(cmp_stat(cand, k), cmp_stat(cur, k), 10000, 2026);
  // kCmpNames: track_1s, spectrum_l1, abs_ln_motion_ratio, coverage_l1, emission_l1, mean_frame_psnr
  r.better = r.iv[5].lo > 0 && r.iv[1].hi < 0;
  r.none_worse = !(r.iv[0].hi < 0) && !(r.iv[2].lo > 0) && !(r.iv[3].lo > 0) && !(r.iv[4].lo > 0);
  return r;
}
void decide(const Ctx& c, sim::Effect e, const Split& sp, const std::vector<Entry>& models, const std::vector<Scores>& sc, const std::string& prefix) {
  const std::string en = ename(e);
  std::vector<std::string> rows;
  int chosen = -1;
  double best = -1e9;
  for (std::size_t mi = 1; mi < models.size(); ++mi) {
    const RuleM r = apply_rule_m(sc[mi], sc[0]);
    std::println("{} {} {}: better on both targets {}, worse on none {} -> {}", sp.name, en, models[mi].name, r.better ? "yes" : "no", r.none_worse ? "yes" : "no",
                 r.pass() ? "MEETS the rule" : "does not meet it");
    rows.push_back(std::format("{},{},{},{},{},{:.4f},{:.5f},{}", en, models[mi].name, r.better ? 1 : 0, r.none_worse ? 1 : 0, r.pass() ? 1 : 0, r.iv[5].mean,
                               r.iv[1].mean, models[mi].sha));
    if (r.pass() && r.iv[5].mean > best) {
      best = r.iv[5].mean;
      chosen = static_cast<int>(mi);
    }
  }
  rows.push_back(std::format("{},chosen,,,,,,{}", en, chosen > 0 ? models[sz(chosen)].name : "none"));
  merge_rows(c.results / std::format("{}_choice.csv", prefix), "effect,model,better,none_worse,meets_rule,mean_frame_gain_db,spectrum_change,sha256", rows, en);
  std::println("{} {}: {}", sp.name, en, chosen > 0 ? "chosen " + models[sz(chosen)].name : std::string("none meets the rule"));
}

// Study M2's rule (docs/REPORT.md §14.8; committed before any of its numbers). Candidate minus current model, paired, 95%
// bootstrap with 10,000 resamples (metrics::paired_bootstrap, seed 2026), as study M's. A candidate meets it when
//   1. it is better on its targets, the parts it changes on purpose: the mean-frame PSNR (interval above zero) for every
//      candidate (the ordered part: the fine-tune, the mean state), and the detail spectrum distance (interval below zero)
//      when it changes the sub-grid swirl's scale (the random part's scale); and
//   2. it is non-inferior on every other measure, by margins fixed in advance (the interval's worse end within them):
//      tracking at 1 s at least -0.20 dB; the spectrum distance, when it is not a target, at most +0.02; |ln motion
//      ratio| at most +0.02; coverage and emission distances at most +10% of the current model's own mean distance on
//      the split, and never less than 0.0001 (amended after the probe, before validation: an effect that adds almost no
//      light has an emission distance of about 10^-5, and 10% of that is the size of its noise).
// The strict reading (no interval entirely on the worse side of zero; study M's) is reported beside and does not decide.
// On validation, among the candidates that meet it, the one with the largest mean-frame gain is chosen.
constexpr double kM2Track = -0.20, kM2Spectrum = 0.02, kM2Motion = 0.02, kM2Share = 0.10, kM2Least = 1e-4;
struct RuleM2 {
  std::array<metrics::Interval, 6> iv{};  // kCmpNames' order
  bool spectrum_target = false;           // the spectrum is a target (the swirl's scale changed)
  bool better = false, none_worse = false;
  std::array<bool, 6> within{};           // per measure: a target, or within its margin
  std::array<bool, 6> strict_worse{};     // per measure: the interval entirely on the worse side of zero
  double margin_cov = 0, margin_emi = 0;
  bool pass() const { return better && none_worse; }
};
RuleM2 apply_rule_m2(const Scores& cand, const Scores& cur, bool spectrum_target) {
  RuleM2 r;
  r.spectrum_target = spectrum_target;
  for (int k = 0; k < 6; ++k) r.iv[sz(k)] = metrics::paired_bootstrap(cmp_stat(cand, k), cmp_stat(cur, k), 10000, 2026);
  const auto mean_of = [](const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); };
  r.margin_cov = std::max(kM2Least, kM2Share * mean_of(cmp_stat(cur, 3)));
  r.margin_emi = std::max(kM2Least, kM2Share * mean_of(cmp_stat(cur, 4)));
  // kCmpNames: track_1s, spectrum_l1, abs_ln_motion_ratio, coverage_l1, emission_l1, mean_frame_psnr
  const bool psnr_better = r.iv[5].lo > 0, spec_better = r.iv[1].hi < 0;
  r.better = psnr_better && (!spectrum_target || spec_better);
  r.within[0] = r.iv[0].lo >= kM2Track;
  r.within[1] = spectrum_target || r.iv[1].hi <= kM2Spectrum;  // a target: in `better`
  r.within[2] = r.iv[2].hi <= kM2Motion;
  r.within[3] = r.iv[3].hi <= r.margin_cov;
  r.within[4] = r.iv[4].hi <= r.margin_emi;
  r.within[5] = true;  // a target (in `better`)
  r.none_worse = std::ranges::all_of(r.within, [](bool b) { return b; });
  r.strict_worse[0] = r.iv[0].hi < 0;
  for (int k = 1; k < 5; ++k) r.strict_worse[sz(k)] = r.iv[sz(k)].lo > 0;
  r.strict_worse[5] = r.iv[5].hi < 0;
  return r;
}
std::string rule_m2_row(const std::string& en, const Entry& m, const RuleM2& r) {
  std::string within, strict;
  for (int k = 0; k < 6; ++k) {
    if (!r.within[sz(k)]) within += (within.empty() ? "" : ";") + kCmpNames[sz(k)];
    if (r.strict_worse[sz(k)]) strict += (strict.empty() ? "" : ";") + kCmpNames[sz(k)];
  }
  return std::format("{},{},{},{},{},{},{:.4f},{:.5f},{:.6f},{:.6f},{},{},{}", en, m.name, r.spectrum_target ? "mean_frame;spectrum_l1" : "mean_frame", r.better ? 1 : 0,
                     r.none_worse ? 1 : 0, r.pass() ? 1 : 0, r.iv[5].mean, r.iv[1].mean, r.margin_cov, r.margin_emi, within.empty() ? "none" : within,
                     strict.empty() ? "none" : strict, m.sha);
}
const std::string kRuleM2Header =
    "effect,model,targets,better,none_worse,meets_rule,mean_frame_gain_db,spectrum_change,coverage_margin,emission_margin,outside_margin,strict_worse,sha256";
bool targets_spectrum(const Entry& cand, const Entry& cur) { return cand.m.detail.swirl_scale != cur.m.detail.swirl_scale; }
void print_rule_m2(const std::string& split, const std::string& en, const Entry& m, const RuleM2& r) {
  std::println("{} {} {} (targets: {}): better on its targets {}, within every margin {} -> {}", split, en, m.name,
               r.spectrum_target ? "the mean frame and the spectrum" : "the mean frame", r.better ? "yes" : "no", r.none_worse ? "yes" : "no",
               r.pass() ? "MEETS the rule" : "does not meet it");
  for (int k = 0; k < 6; ++k) {
    std::println("    {:<20} {:+.5f} [{:+.5f}, {:+.5f}]{}{}", kCmpNames[sz(k)], r.iv[sz(k)].mean, r.iv[sz(k)].lo, r.iv[sz(k)].hi,
                 r.within[sz(k)] ? "" : "  OUTSIDE its margin", r.strict_worse[sz(k)] ? "  (strictly worse)" : "");
  }
}
void decide_m2(const Ctx& c, sim::Effect e, const Split& sp, const std::vector<Entry>& models, const std::vector<Scores>& sc, const std::string& prefix) {
  const std::string en = ename(e);
  std::vector<std::string> rows;
  int chosen = -1;
  double best = -1e9;
  for (std::size_t mi = 1; mi < models.size(); ++mi) {
    const RuleM2 r = apply_rule_m2(sc[mi], sc[0], targets_spectrum(models[mi], models[0]));
    print_rule_m2(sp.name, en, models[mi], r);
    rows.push_back(rule_m2_row(en, models[mi], r));
    if (r.pass() && r.iv[5].mean > best) {
      best = r.iv[5].mean;
      chosen = static_cast<int>(mi);
    }
  }
  rows.push_back(std::format("{},chosen,,,,,,,,,,,{}", en, chosen > 0 ? models[sz(chosen)].name : "none"));
  merge_rows(c.results / std::format("{}_choice.csv", prefix), kRuleM2Header, rows, en);
  std::println("{} {}: {}", sp.name, en, chosen > 0 ? "chosen " + models[sz(chosen)].name : std::string("none meets the rule"));
}

// Study M3's rule (docs/REPORT.md §14.9; its numbers fixed by the probe and committed before validation). Candidate minus
// current model, paired, 95% bootstrap with 10,000 resamples (metrics::paired_bootstrap, seed 2026), as studies M's and
// M2's. A candidate (the mean state) meets it when
//   1. its mean-frame PSNR is better (the interval above zero), and
//   2. every other measure is within its margin (the interval's worse end): tracking at 1 s at least margin.track; the
//      detail spectrum distance, |ln motion ratio|, coverage and emission distances at most their margins.
// Each margin is the larger of study K's at the measure's scale (as M2 used them) and the null's RMS change at a setting
// (the current model against itself with fresh seeds, 8 runs pooled, over the probe's settings); §14.9.2 says why. The
// strict reading (no interval entirely on the worse side of zero) is reported beside and does not decide.
struct MarginsM3 {
  double track = 0, spectrum = 0, motion = 0, coverage = 0, emission = 0;
  bool set() const { return spectrum > 0; }
  double of(int k) const { return std::array{track, spectrum, motion, coverage, emission, 0.0}[sz(k)]; }  // kCmpNames' order
};
constexpr double kM3Track = -0.20, kM3Spectrum = 0.02, kM3Motion = 0.02, kM3Share = 0.10, kM3Least = 1e-4;
// The margins of the rule, per effect: m3_margins.csv from the probe's null (§14.9.2), copied here before validation.
MarginsM3 margins_m3(sim::Effect e) {
  const std::string en = ename(e);
  if (en == "explosion") return {};
  if (en == "dust") return {};
  return {};
}
double mean_of(const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); }
std::vector<double> diff_of(const Scores& a, const Scores& b, int k) {
  auto x = cmp_stat(a, k);
  const auto y = cmp_stat(b, k);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] -= y[i];
  return x;
}
double rms_of(const std::vector<double>& v) {
  double s = 0;
  for (const double x : v) s += x * x;
  return std::sqrt(s / static_cast<double>(v.size()));
}
double sd_of(const std::vector<double>& v) {
  const double m = mean_of(v);
  double s = 0;
  for (const double x : v) s += (x - m) * (x - m);
  return std::sqrt(s / static_cast<double>(v.size() - 1));
}
// Study K's margins at the measures' scales, as study M2 used them (§14.8.1): tracking -0.20 dB at 1 s, spectrum and motion
// +0.02, coverage and emission +10% of the current model's mean distance (here on the probe), at least 0.0001.
MarginsM3 margins_k(const Scores& cur) {
  return {kM3Track, kM3Spectrum, kM3Motion, std::max(kM3Least, kM3Share * mean_of(cmp_stat(cur, 3))), std::max(kM3Least, kM3Share * mean_of(cmp_stat(cur, 4)))};
}
// The margins the null gives: each the larger of study K's and the null's RMS change over the probe's settings (tracking
// has no null: a tracked run takes its own seed, so the current model plays the same frames; it keeps study K's).
MarginsM3 margins_from_null(const Scores& cur, const Scores& null) {
  const MarginsM3 k = margins_k(cur);
  return {k.track, std::max(k.spectrum, rms_of(diff_of(null, cur, 1))), std::max(k.motion, rms_of(diff_of(null, cur, 2))),
          std::max(k.coverage, rms_of(diff_of(null, cur, 3))), std::max(k.emission, rms_of(diff_of(null, cur, 4)))};
}

struct RuleM3 {
  std::array<metrics::Interval, 6> iv{};  // kCmpNames' order
  MarginsM3 mg;
  bool better = false, none_worse = false;
  std::array<bool, 6> within{}, strict_worse{};
  bool pass() const { return better && none_worse; }
};
RuleM3 apply_rule_m3(const Scores& cand, const Scores& cur, const MarginsM3& mg) {
  RuleM3 r;
  r.mg = mg;
  for (int k = 0; k < 6; ++k) r.iv[sz(k)] = metrics::paired_bootstrap(cmp_stat(cand, k), cmp_stat(cur, k), 10000, 2026);
  r.better = r.iv[5].lo > 0;
  r.within[0] = r.iv[0].lo >= mg.track;
  for (int k = 1; k < 5; ++k) r.within[sz(k)] = r.iv[sz(k)].hi <= mg.of(k);
  r.within[5] = true;  // the target (in `better`)
  r.none_worse = std::ranges::all_of(r.within, [](bool b) { return b; });
  r.strict_worse[0] = r.iv[0].hi < 0;
  for (int k = 1; k < 5; ++k) r.strict_worse[sz(k)] = r.iv[sz(k)].lo > 0;
  r.strict_worse[5] = r.iv[5].hi < 0;
  return r;
}
const std::string kRuleM3Header =
    "effect,model,better,none_worse,meets_rule,mean_frame_gain_db,mean_frame_lo,spectrum_change,spectrum_hi,motion_hi,coverage_hi,emission_hi,track_lo,"
    "margin_spectrum,margin_motion,margin_coverage,margin_emission,margin_track,outside_margin,strict_worse,sha256";
std::string rule_m3_row(const std::string& en, const Entry& m, const RuleM3& r) {
  std::string within, strict;
  for (int k = 0; k < 6; ++k) {
    if (!r.within[sz(k)]) within += (within.empty() ? "" : ";") + kCmpNames[sz(k)];
    if (r.strict_worse[sz(k)]) strict += (strict.empty() ? "" : ";") + kCmpNames[sz(k)];
  }
  return std::format("{},{},{},{},{},{:.4f},{:.4f},{:.5f},{:.5f},{:.5f},{:.6f},{:.7f},{:.4f},{:.5f},{:.5f},{:.6f},{:.7f},{:.2f},{},{},{}", en, m.name, r.better ? 1 : 0,
                     r.none_worse ? 1 : 0, r.pass() ? 1 : 0, r.iv[5].mean, r.iv[5].lo, r.iv[1].mean, r.iv[1].hi, r.iv[2].hi, r.iv[3].hi, r.iv[4].hi, r.iv[0].lo,
                     r.mg.spectrum, r.mg.motion, r.mg.coverage, r.mg.emission, r.mg.track, within.empty() ? "none" : within, strict.empty() ? "none" : strict, m.sha);
}
void print_rule_m3(const std::string& split, const std::string& en, const Entry& m, const RuleM3& r) {
  std::println("{} {} {} (target: the mean frame): better on its target {}, within every margin {} -> {}", split, en, m.name, r.better ? "yes" : "no",
               r.none_worse ? "yes" : "no", r.pass() ? "MEETS the rule" : "does not meet it");
  for (int k = 0; k < 6; ++k) {
    std::println("    {:<20} {:+.5f} [{:+.5f}, {:+.5f}]{}{}{}", kCmpNames[sz(k)], r.iv[sz(k)].mean, r.iv[sz(k)].lo, r.iv[sz(k)].hi,
                 k == 5 ? "  (target)" : std::format("  margin {:+.5f}", r.mg.of(k)), r.within[sz(k)] ? "" : "  OUTSIDE its margin",
                 r.strict_worse[sz(k)] ? "  (strictly worse)" : "");
  }
}
void decide_m3(const Ctx& c, sim::Effect e, const Split& sp, const std::vector<Entry>& models, const std::vector<Scores>& sc, const std::string& prefix) {
  const std::string en = ename(e);
  const MarginsM3 mg = margins_m3(e);
  if (!mg.set()) throw std::runtime_error("study M3: the margins of " + en + " are not fixed (margins_m3)");
  std::vector<std::string> rows;
  int chosen = -1;
  double best = -1e9;
  for (std::size_t mi = 1; mi < models.size(); ++mi) {
    const RuleM3 r = apply_rule_m3(sc[mi], sc[0], mg);
    print_rule_m3(sp.name, en, models[mi], r);
    rows.push_back(rule_m3_row(en, models[mi], r));
    if (r.pass() && r.iv[5].mean > best) {
      best = r.iv[5].mean;
      chosen = static_cast<int>(mi);
    }
  }
  rows.push_back(std::format("{},chosen,,,,,,,,,,,,,,,,,,,{}", en, chosen > 0 ? models[sz(chosen)].name : "none"));
  merge_rows(c.results / std::format("{}_choice.csv", prefix), kRuleM3Header, rows, en);
  std::println("{} {}: {}", sp.name, en, chosen > 0 ? "chosen " + models[sz(chosen)].name : std::string("none meets the rule"));
}

// The probe's power analysis (§14.9.2). For each candidate (and the null) and measure: the paired differences over the
// probe's settings (mean, SD, RMS, the SD's upper 80% confidence bound by Wilson and Hilferty's chi-square), the number of
// settings at which a measure truly unchanged on average with that spread passes its margin with probability 0.9 (normal
// form: ((1.96 + 1.2816) sd / margin)^2; the target: to clear zero at the probe's mean), and the probability of passing at
// N settings by resampling the probe's settings: 4,000 sets of N drawn with replacement, jointly over the measures, each
// set's 95% interval in the bootstrap's normal form (mean +- 1.96 sd / sqrt(N), sd over N), with each measure's
// differences centred on zero (unchanged), centred on its margin (a loss of the margin's size), or as measured (the
// target), and the rule as a whole (every other measure unchanged, the target as measured).
constexpr double kZ80 = 0.8416212335729143;                            // the standard normal's 80% quantile
constexpr double kZPower = 1.959963984540054 + 1.2815515655446004;     // a 95% interval's end, and 90% power
double sd_upper80(double sd, int n) {  // the upper end of the SD's one-sided 80% interval (chi-square's 20% quantile)
  const double nu = n - 1, q = nu * std::pow(1.0 - 2.0 / (9.0 * nu) - kZ80 * std::sqrt(2.0 / (9.0 * nu)), 3.0);
  return sd * std::sqrt(nu / q);
}
const std::array kM3PowerN = {10, 20, 30, 40, 50, 60, 70, 80, 100, 120};

void probe_m3(const Ctx& c, sim::Effect e, const Split& sp, const std::vector<Entry>& models, const std::vector<Scores>& sc) {
  const std::string en = ename(e);
  const int P = static_cast<int>(sp.settings.size());
  int null_i = -1;
  for (std::size_t mi = 1; mi < models.size(); ++mi) {
    if (models[mi].seed_base > 0) null_i = static_cast<int>(mi);
  }
  const Scores& floor = sc.back();
  MarginsM3 mg = margins_m3(e);
  if (null_i > 0) {
    const Scores& nl = sc[sz(null_i)];
    mg = margins_from_null(sc[0], nl);
    const MarginsM3 mk = margins_k(sc[0]);
    std::vector<std::string> rows;
    std::println("\nprobe3 {}: the margins the null gives (each the larger of study K's and the null's RMS change at a setting, {} settings)", en, P);
    std::println("{:<20}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}", "measure", "K's", "null RMS", "null mean", "null SD", "floor", "current", "margin");
    for (int k = 0; k < 5; ++k) {
      const auto d = diff_of(nl, sc[0], k);
      const double fl = k == 0 ? 0.0 : mean_of(cmp_stat(floor, k));
      std::println("{:<20}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}", kCmpNames[sz(k)], mk.of(k), rms_of(d), mean_of(d), sd_of(d), fl,
                   mean_of(cmp_stat(sc[0], k)), mg.of(k));
      rows.push_back(std::format("{},{},{},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f}", en, kCmpNames[sz(k)], P, mk.of(k), rms_of(d), mean_of(d), sd_of(d), fl,
                                 mean_of(cmp_stat(sc[0], k)), mg.of(k)));
    }
    merge_rows(c.results / "m3_margins.csv", "effect,measure,settings,k_margin,null_rms,null_mean,null_sd,floor_mean,current_mean,margin", rows, en);
  }
  if (!mg.set()) throw std::runtime_error("probe3: no null (--null) and no margins fixed for " + en);
  std::vector<std::string> rows;
  for (std::size_t mi = 1; mi < models.size(); ++mi) {
    std::array<std::vector<double>, 6> d;
    std::array<double, 6> mean{}, sd{}, sdu{};
    for (int k = 0; k < 6; ++k) {
      d[sz(k)] = diff_of(sc[mi], sc[0], k);
      mean[sz(k)] = mean_of(d[sz(k)]);
      sd[sz(k)] = sd_of(d[sz(k)]);
      sdu[sz(k)] = sd_upper80(sd[sz(k)], P);
    }
    if (static_cast<int>(mi) != null_i) print_rule_m3("probe3", en, models[mi], apply_rule_m3(sc[mi], sc[0], mg));
    std::println("\nprobe3 {} {} - current: power by resampling the probe's {} settings (pass: unchanged / a loss of the margin's size; the target as measured)", en,
                 models[mi].name, P);
    std::println("{:<20}{:>10}{:>10}{:>10}{:>10}{:>8}{:>8} | pass at N settings: {}", "measure", "mean", "SD", "SD up80", "margin", "N90", "N90up",
                 [] {
                   std::string s;
                   for (const int N : kM3PowerN) s += std::format("{:>6}", N);
                   return s;
                 }());
    std::array<std::array<double, kM3PowerN.size()>, 6> pu{}, pl{};
    std::array<double, kM3PowerN.size()> prule{}, pothers{};
    std::mt19937_64 rng(2026 + mi);
    std::uniform_int_distribution<int> pick(0, P - 1);
    for (std::size_t ni = 0; ni < kM3PowerN.size(); ++ni) {
      const int N = kM3PowerN[ni];
      constexpr int R = 4000;
      std::vector<int> idx(sz(N));
      for (int r = 0; r < R; ++r) {
        for (int& i : idx) i = pick(rng);
        bool others = true, target = false;
        for (int k = 0; k < 6; ++k) {
          double s1 = 0, s2 = 0;
          for (const int i : idx) {
            const double x = d[sz(k)][sz(i)] - mean[sz(k)];
            s1 += x;
            s2 += x * x;
          }
          const double m = s1 / N, h = 1.959963984540054 * std::sqrt(std::max(0.0, s2 / N - m * m) / N);
          if (k == 5) {  // the target, as measured
            target = m + mean[5] - h > 0;
            pu[5][ni] += target ? 1.0 / R : 0.0;
            continue;
          }
          const double mk = mg.of(k);
          const bool u = k == 0 ? m - h >= mk : m + h <= mk;  // unchanged: centred on zero
          const bool l = k == 0 ? m - h >= 0 : m + h <= 0;    // a loss of the margin's size: centred on the margin
          pu[sz(k)][ni] += u ? 1.0 / R : 0.0;
          pl[sz(k)][ni] += l ? 1.0 / R : 0.0;
          others = others && u;
        }
        pothers[ni] += others ? 1.0 / R : 0.0;
        prule[ni] += others && target ? 1.0 / R : 0.0;
      }
    }
    for (int k = 0; k < 6; ++k) {
      const double scale = k == 5 ? mean[5] : std::abs(mg.of(k));
      const double n90 = scale > 0 ? std::pow(kZPower * sd[sz(k)] / scale, 2) : 1e9, n90u = scale > 0 ? std::pow(kZPower * sdu[sz(k)] / scale, 2) : 1e9;
      std::string line = std::format("{:<20}{:>+10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>8.0f}{:>8.0f} |                     ", kCmpNames[sz(k)], mean[sz(k)], sd[sz(k)],
                                     sdu[sz(k)], k == 5 ? 0.0 : mg.of(k), std::ceil(n90), std::ceil(n90u));
      for (std::size_t ni = 0; ni < kM3PowerN.size(); ++ni) {
        line += std::format("{:>6.2f}", pu[sz(k)][ni]);
        rows.push_back(std::format("{},{},{},{},{},{:.7f},{:.7f},{:.7f},{:.7f},{:.1f},{:.1f},{:.4f},{}", en, models[mi].name, kCmpNames[sz(k)], P, kM3PowerN[ni],
                                   mean[sz(k)], sd[sz(k)], sdu[sz(k)], k == 5 ? 0.0 : mg.of(k), n90, n90u, pu[sz(k)][ni],
                                   k == 5 ? std::string("") : std::format("{:.4f}", pl[sz(k)][ni])));
      }
      std::println("{}", line);
    }
    std::string l1 = std::format("{:<84}", "every other measure unchanged"), l2 = std::format("{:<84}", "the rule (others unchanged, target as measured)");
    for (std::size_t ni = 0; ni < kM3PowerN.size(); ++ni) {
      l1 += std::format("{:>6.2f}", pothers[ni]);
      l2 += std::format("{:>6.2f}", prule[ni]);
      rows.push_back(std::format("{},{},others_unchanged,{},{},,,,,,,{:.4f},", en, models[mi].name, P, kM3PowerN[ni], pothers[ni]));
      rows.push_back(std::format("{},{},rule,{},{},,,,,,,{:.4f},", en, models[mi].name, P, kM3PowerN[ni], prule[ni]));
    }
    std::println("{}\n{}", l1, l2);
  }
  merge_rows(c.results / "m3_power.csv", "effect,model,measure,probe_settings,n_settings,mean,sd,sd_up80,margin,n90,n90_up80,p_pass,p_pass_if_loss", rows, en);
}

void step_val(const Ctx& c, sim::Effect e, const tools::Args& a) {
  auto models = read_entries(a.need("current"), a.str("cand", ""));
  const Split sp = split_of(a.str("split", c.m3 ? "probe3" : c.m2 ? "probe2" : "probe"), e);
  if (sp.name == "test" || sp.name == "test2" || sp.name == "test3") throw std::invalid_argument("val: the test runs once, through `test`");
  if (c.m2 != (sp.name == "probe2" || sp.name == "val2")) throw std::invalid_argument("val: study M2's splits are probe2 and val2, with --study m2");
  if (c.m3 != (sp.name == "probe3" || sp.name == "val3")) throw std::invalid_argument("val: study M3's splits are probe3 and val3, with --study m3");
  if (a.flag("null")) {  // study M3's null: the current model played with fresh seeds (the probe only)
    if (sp.name != "probe3") throw std::invalid_argument("val: --null is the probe's (probe3)");
    Entry n = models[0];
    n.name = "null";
    n.seed_base = sp.null;
    models.push_back(std::move(n));
  }
  for (const auto& m : models) {
    std::println("{}: {} {} {} sha256 {}{}", sp.name, ename(e), m.name, m.path.string(), m.sha,
                 m.seed_base > 0 ? std::format(" (its plays from seed {})", m.seed_base) : std::string());
  }
  const auto sc = evaluate(c, e, sp, models);
  const std::string prefix = a.str("prefix", std::format("m_{}", sp.name));
  report_split(c, e, sp, models, sc, prefix);
  if (sp.name == "val") decide(c, e, sp, models, sc, prefix);
  if (sp.name == "val2") decide_m2(c, e, sp, models, sc, prefix);
  if (sp.name == "val3") decide_m3(c, e, sp, models, sc, prefix);
  if (sp.name == "probe3") probe_m3(c, e, sp, models, sc);
  if (sp.name == "probe2") {  // the rule's reading on training settings, for development only (not a decision)
    for (std::size_t mi = 1; mi < models.size(); ++mi) print_rule_m2(sp.name, ename(e), models[mi], apply_rule_m2(sc[mi], sc[0], targets_spectrum(models[mi], models[0])));
  }
}

// The test, once per effect: the current model and the chosen candidate, whose SHA-256 must be in --frozen (with the
// effect's name) before it runs; a marker under DATA/m_ordered keeps it from running twice.
void step_test(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const fs::path done = c.dir() / std::format("test_{}.done", ename(e));
  if (fs::exists(done)) throw std::runtime_error("the test of " + ename(e) + " has run (it runs once)");
  const auto models = read_entries(a.need("current"), a.need("cand"));
  if (models.size() != 2) throw std::invalid_argument("test: one candidate");
  {
    std::ifstream in(a.need("frozen"));
    std::string line;
    bool ok = false;
    while (std::getline(in, line)) {
      if (line.find(models[1].sha) != std::string::npos && line.rfind(ename(e) + ",", 0) == 0) ok = true;
    }
    if (!ok) throw std::runtime_error("test: the candidate's SHA-256 is not frozen for " + ename(e) + " in " + a.need("frozen"));
  }
  const MarginsM3 mg3 = c.m3 ? margins_m3(e) : MarginsM3{};
  if (c.m3 && !mg3.set()) throw std::runtime_error("test: study M3's margins of " + ename(e) + " are not fixed");
  const Split sp = split_of(c.m3 ? "test3" : c.m2 ? "test2" : "test", e);  // throws before the marker if the settings are not fixed
  std::ofstream(done) << "started " << models[1].sha << "\n";
  for (const auto& m : models) std::println("test: {} {} {} sha256 {}", ename(e), m.name, m.path.string(), m.sha);
  const auto sc = evaluate(c, e, sp, models);
  if (c.m3) {  // study M3: its rule, once
    report_split(c, e, sp, models, sc, "m3_test");
    const RuleM3 r = apply_rule_m3(sc[1], sc[0], mg3);
    print_rule_m3("test3", ename(e), models[1], r);
    std::println("test3 {} {}: {}", ename(e), models[1].name, r.pass() ? "MEETS the rule: replace" : "does not meet it: keep the current model");
    merge_rows(c.results / "m3_test_decision.csv", kRuleM3Header, {rule_m3_row(ename(e), models[1], r)}, ename(e));
    std::ofstream(done) << "done " << models[1].sha << " " << (r.pass() ? "met" : "not met") << "\n";
    return;
  }
  if (c.m2) {  // study M2: its rule, once
    report_split(c, e, sp, models, sc, "m2_test");
    const RuleM2 r = apply_rule_m2(sc[1], sc[0], targets_spectrum(models[1], models[0]));
    print_rule_m2("test2", ename(e), models[1], r);
    std::println("test2 {} {}: {}", ename(e), models[1].name, r.pass() ? "MEETS the rule: replace" : "does not meet it: keep the current model");
    merge_rows(c.results / "m2_test_decision.csv", kRuleM2Header, {rule_m2_row(ename(e), models[1], r)}, ename(e));
    std::ofstream(done) << "done " << models[1].sha << " " << (r.pass() ? "met" : "not met") << "\n";
    return;
  }
  report_split(c, e, sp, models, sc, "m_test");
  const RuleM r = apply_rule_m(sc[1], sc[0]);
  std::println("test {} {}: better on both targets {}, worse on none {} -> {}", ename(e), models[1].name, r.better ? "yes" : "no", r.none_worse ? "yes" : "no",
               r.pass() ? "MEETS the rule: replace" : "does not meet it: keep the current model");
  merge_rows(c.results / "m_test_decision.csv", "effect,model,better,none_worse,meets_rule,sha256",
             {std::format("{},{},{},{},{},{}", ename(e), models[1].name, r.better ? 1 : 0, r.none_worse ? 1 : 0, r.pass() ? 1 : 0, models[1].sha)}, ename(e));
  std::ofstream(done) << "done " << models[1].sha << " " << (r.pass() ? "met" : "not met") << "\n";
}

// --- cost: thread CPU time per frame through nvfx_render, models interleaved -------------------------------------------

double thread_ms() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) * 1e3 + static_cast<double>(ts.tv_nsec) * 1e-6;
}
// As study L's (nvfx_swirl cost), with the precision chosen: per repetition and size, each model in turn plays 200 frames
// (one-shots: their 88) after the first at controls (0.6, 0.5, 0.6) and the median over frames is kept; reported per size: the median over
// repetitions, its range, and the median of each model's paired difference from the first (the current model).
// --precision float|int8|default (the runtime's: int8 where the CPU has VNNI); --ftz flushes subnormal floats to zero in
// this thread (FTZ and DAZ), as many engines run.
void step_cost(sim::Effect e, const tools::Args& a) {
  const auto models = read_entries(a.need("current"), a.str("cand", ""));
  const int reps = a.i("reps", 7), frames = sim::effect_loops(e) ? 200 : 88;  // a one-shot holds its last frame after 89
  const std::string prec = a.str("precision", "float");
  const nvfx_precision pr = prec == "float" ? NVFX_PRECISION_FLOAT : prec == "int8" ? NVFX_PRECISION_INT8 : NVFX_PRECISION_DEFAULT;
  if (prec != "float" && prec != "int8" && prec != "default") throw std::invalid_argument("--precision float, int8 or default");
  const bool ftz = a.flag("ftz");
  if (ftz) _mm_setcsr(_mm_getcsr() | 0x8040);
  std::vector<int> sizes;
  for (const auto& t : split_list(a.str("sizes", "64,128,256"), ',')) sizes.push_back(std::stoi(t));
  std::vector<std::unique_ptr<Runtime>> rts;
  for (const auto& m : models) rts.push_back(std::make_unique<Runtime>(m.m));
  std::vector<std::string> rows;
  for (const int size : sizes) {
    std::vector<std::vector<double>> med(models.size());
    std::vector<std::uint8_t> buf(sz(size) * sz(size) * 4);
    for (int r = 0; r < reps; ++r) {
      for (std::size_t k = 0; k < models.size(); ++k) {
        nvfx_instance* in = nullptr;
        if (nvfx_instance_create(rts[k]->e, size, &in) != NVFX_OK) throw std::runtime_error("instance");
        if (nvfx_instance_set_precision(in, pr) != NVFX_OK) throw std::runtime_error("precision");
        const float ctl[3] = {0.6f, 0.5f, 0.6f};
        nvfx_instance_set_controls(in, ctl, 3);
        nvfx_instance_set_seed(in, 77 + static_cast<std::uint64_t>(r));
        nvfx_render(in, 0.0, buf.data(), sz(size) * 4);
        std::vector<double> ms;
        for (int f = 1; f <= frames; ++f) {
          const double t0 = thread_ms();
          nvfx_render(in, f / 30.0, buf.data(), sz(size) * 4);
          ms.push_back(thread_ms() - t0);
        }
        nvfx_instance_free(in);
        std::ranges::sort(ms);
        med[k].push_back(ms[ms.size() / 2]);
      }
    }
    const auto median = [](std::vector<double> v) {
      std::ranges::sort(v);
      return v[v.size() / 2];
    };
    for (std::size_t k = 0; k < models.size(); ++k) {
      std::vector<double> d;
      for (int r = 0; r < reps; ++r) d.push_back(med[k][sz(r)] - med[0][sz(r)]);
      const auto [lo, hi] = std::ranges::minmax(med[k]);
      std::println("cost {} {} at {} px, {}{}: {:.3f} ms (range {:.3f} to {:.3f}), minus current {:+.3f} ms (median of {} paired repetitions)", ename(e),
                   models[k].name, size, prec, ftz ? ", FTZ" : "", median(med[k]), lo, hi, median(d), reps);
      rows.push_back(std::format("{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:+.4f},{},{}", ename(e), models[k].name, size, prec, ftz ? 1 : 0, median(med[k]), lo, hi,
                                 median(d), reps, models[k].sha));
    }
  }
  if (a.has("out")) {  // appended: one session's rounds in order
    const bool fresh = !fs::exists(a.str("out"));
    std::ofstream o(a.str("out"), std::ios::app);
    if (fresh) o << "effect,model,size,precision,ftz,median_ms,min_ms,max_ms,minus_current_ms,reps,sha256\n";
    for (const auto& r : rows) o << r << "\n";
  }
}

// --- info: a model file's constants ------------------------------------------------------------------------------------------

void step_info(const tools::Args& a) {
  const fs::path p = a.need("model");
  const rollout::Model m = load_or_throw(p);
  const rollout::DetailSpec& d = m.detail;
  std::println("{}: {} ({} bytes, sha256 {}), {} controls, {} age inputs, loop {}", p.string(), m.effect, fs::file_size(p), file_sha256(p), m.h.n_controls,
               m.h.n_age, m.loop ? "yes" : "no");
  std::println("  stepper: res {}, hidden {}, memory {}, advection {}, force {}; start points {} (fine fields {} px, coarse bits {})", m.h.res, m.h.hidden,
               m.h.memory, m.h.advection, m.h.force, m.starts.size(), m.h.start_fine, m.start_bits);
  std::println("  detail: contrast {}, kappa {}, edges {} {}, swirl {}, swirl_scale {}, swirl_rate {}, swirl_ramp {}, swirl_control {}, grow {}, advect {}, "
               "soften {}, passes {}",
               d.contrast, d.kappa, d.edge0, d.edge1, d.swirl, d.swirl_scale, d.swirl_rate, d.swirl_ramp, d.swirl_control, d.grow, d.advect, d.soften, d.passes);
  std::println("  mean state: {}; bias correction: {}", m.mean.empty() ? "none" : std::format("blocks of {} cells, {} coefficients", m.mean.block, m.mean.beta.size()),
               m.bias.empty() ? "none" : std::format("blocks of {} cells", m.bias.block));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    tools::Args a(argc, argv, {"play", "ftz", "floor-gradient", "null"});
    if (a.positional().empty()) {
      std::println("nvfx_ordered diag --effect E [--model F] [--settings 8] [--reps 4] [--root DIR] [--results DIR] [--threads N]");
      return 2;
    }
    Ctx c;
    const char* env = std::getenv("NEURALVFX_DATA");
    c.data = a.str("root", env ? env : "/root/nvfx-data");
    c.results = a.str("results", "results/experiments/ordered");
    c.threads = a.i("threads", 1);
    const std::string study = a.str("study", "m");
    if (study != "m" && study != "m2" && study != "m3") throw std::invalid_argument("--study m, m2 or m3");
    c.m2 = study == "m2";
    c.m3 = study == "m3";
    const std::string step = a.positional()[0];
    if (step == "setmean" || step == "set" || step == "info") {
      if (step == "set") step_set(a);
      else if (step == "info") step_info(a);
      else step_setmean(a);
      return 0;
    }
    const sim::Effect e = effect_of(a.need("effect"));
    if (step == "diag") step_diag(c, e, a);
    else if (step == "val") step_val(c, e, a);
    else if (step == "test") step_test(c, e, a);
    else if (step == "dvar") step_dvar(c, e, a);
    else if (step == "where") step_where(c, e, a);
    else if (step == "fitmean") step_fitmean(c, e, a);
    else if (step == "sstart") step_sstart(c, e, a);
    else if (step == "fitbias") step_fitbias(c, e, a);
    else if (step == "train") step_train(c, e, a);
    else if (step == "cost") step_cost(e, a);
    else throw std::invalid_argument("unknown step " + step);
    a.warn_unused();
  } catch (const std::exception& e) {
    std::println(stderr, "nvfx_ordered: {}", e.what());
    return 1;
  }
  return 0;
}
