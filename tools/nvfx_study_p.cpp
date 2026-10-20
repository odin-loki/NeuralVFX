// nvfx_study_p: study P (docs/STUDY_P.md, docs/REPORT.md §17): large rollout effects drawn at half size and upscaled.
//
// Three arms per effect at a target size S (256 and 512 px), all through the runtime as it ships on this CPU (its default
// precision and ISA):
//   N  an instance created at S (native);
//   B  an instance at S / 2, upscaled by the bilinear 2x (what an engine's GPU sampler gives for free);
//   L  the same S / 2 instance, upscaled by the learned 2x with seeded micro-detail (include/neuralfx/upscale.hpp).
// The S / 2 instance's frames are upscaled by the runtime's own upscaler (rt_upscale.hpp) with the instance's seed and the
// frame's time, which is exactly what an instance with nvfx_instance_set_upscale draws (`check` verifies it).
//
//   nvfx_study_p pin                      the six effects' files as the library ships them now, copied to DATA/p_upscale/models
//                                         and frozen by SHA-256 (results/experiments/upscale/p_models.csv)
//   nvfx_study_p info [--size 128]        every shipped effect's coverage (mean alpha) at 8 training settings: how much of
//                                         the frame each fills (the choice of the six)
//   nvfx_study_p check                    the simulator's dynamics do not depend on its output size; the coarse state of an
//                                         instance does not depend on its size (the macro motion is shared); an instance
//                                         that upscales draws the S / 2 instance's frames upscaled
//   nvfx_study_p tiles --sizes 256,512    training tiles (training settings and seeds) to DATA/p_upscale/tiles
//   nvfx_study_p train --effects LIST --out NAME [--sizes 256,512] [--c1 8 --c2 8 --k2 1] [--steps 4000] [--batch 32]
//                      [--lr 3e-3] [--amp-from 2000] [--seed 1]
//                                         trains an upscaler on the tiles of the effects (shared: all six) to
//                                         DATA/p_upscale/nets/NAME.nvfxup
//   nvfx_study_p dev --nets A,B,... [--effects] [--sizes] [--settings 4] [--reps 1] [--gains g1,g2] [--rates r1,r2]
//                                         development only (training settings 16 on, other seeds): (a), the high band and
//                                         motion against N for each net, gain and rate
//   nvfx_study_p calib --net NAME --gains ... --rates ...   the micro-detail's gain and rate on the dev split, by the
//                                         procedure of docs/STUDY_P.md §3 (training data only), written into NAME
//   nvfx_study_p eval --split val|test [--null] [--cands NAME=FILE,...] [--effects] [--sizes] [--settings N] [--reps 4]
//                                         every measure per setting: (a) against N on the same seeds, (b) against the
//                                         simulator's frames at S (tracking at 1 s, the endless statistics, the high band);
//                                         rows to results/experiments/upscale/p_eval_<split>.csv (resumes by setting)
//   nvfx_study_p margins                  the null probe's margins (N with fresh seeds against N, validation):
//                                         results/experiments/upscale/p_margins.csv
//   nvfx_study_p decide --split val|test  the rule (docs/STUDY_P.md §4): paired bootstrap, 95%, 10,000 resamples
//   nvfx_study_p cost [--net FILE] [--sizes 256,512] [--rounds 30] [--frames 20] [--core -1]
//                                         thread CPU time per frame, N, the S / 2 instance alone (B on a GPU), B and L
//                                         interleaved within one process: results/experiments/upscale/p_cost.csv
//   nvfx_study_p video --effect E --net FILE --out F.mp4 [--size 512] [--seconds 10]
//                                         N, B and L side by side
//
// Splits and seeds (study P owns 3,600,000 to 3,999,999). Real run k of setting s has seed real + 100 s + k, plays
// play + 100 s + k (the same seed for N and the S / 2 instance: B and L are paired with N), the null's plays null + 100 s +
// k, tracking runs track + 100 s + k (from an empty grid: looping effects 100 frames, one-shots 1, then followed for 1 s).
//   train  the controls of salt-1 training runs 0 to 15: the upscaler's tiles, plays 3,600,000
//   dev    salt-1 training runs 16 to 23 (development only): plays 3,650,000
//   val    study G's validation settings, extended as study M3's (validation_settings(n, 30)): real 3,700,000, plays
//          3,710,000, the null 3,720,000, tracking 3,730,000
//   test   study B's held-out settings, extended as study M3's (test_settings(n)): real 3,800,000, plays 3,810,000,
//          tracking 3,830,000
//   cost and videos: 3,900,000 on
// The real frames at S are the teacher's: the simulation on its 128-cell grid (the resolution every model was trained and
// measured at), drawn by its own renderer at S px.
//
// Options: --root DATA (default $NEURALVFX_DATA or /root/nvfx-data; this study's files are under DATA/p_upscale),
// --results DIR (results/experiments/upscale), --threads N (2).
#include "args.hpp"
#include "label_font.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>
#include <neuralfx/upscale.hpp>

#include "nvfx_internal.hpp"
#include "rt_common.hpp"
#include "rt_upscale.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <print>
#include <random>
#include <sched.h>
#include <set>
#include <sstream>
#include <stdfloat>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace nfx;

namespace {

constexpr int kRes = 32, kTeacher = 128;
constexpr double kFps = 30.0;
std::size_t sz(int v) { return static_cast<std::size_t>(v); }
std::string ename(sim::Effect e) { return std::string(sim::effect_name(e)); }
double seconds_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

double thread_ms() {
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return static_cast<double>(t.tv_sec) * 1e3 + static_cast<double>(t.tv_nsec) * 1e-6;
}

void pin(int core) {
  if (core < 0) return;
  cpu_set_t one;
  CPU_ZERO(&one);
  CPU_SET(core, &one);
  sched_setaffinity(0, sizeof(one), &one);
}

double median(std::vector<double> v) {
  if (v.empty()) return std::nan("");
  std::ranges::sort(v);
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
double mean(const std::vector<double>& v) { return v.empty() ? std::nan("") : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); }

template <class F>
void parallel(int threads, int n, F&& f) {
  std::atomic<int> next{0};
  std::vector<std::jthread> pool;
  for (int t = 0; t < std::max(1, std::min(threads, n)); ++t) {
    pool.emplace_back([&] {
      for (int i; (i = next++) < n;) f(i);
    });
  }
}

std::vector<int> ints(const std::string& s) {
  std::vector<int> v;
  for (const float f : tools::parse_floats(s)) v.push_back(static_cast<int>(f));
  return v;
}
std::vector<std::string> words(const std::string& s) {
  std::vector<std::string> v;
  std::size_t a = 0;
  while (a < s.size()) {
    const std::size_t b = std::min(s.find(',', a), s.size());
    if (b > a) v.push_back(s.substr(a, b - a));
    a = b + 1;
  }
  return v;
}

// --- the effects, their files, settings and seeds -----------------------------------------------------------------------

const std::array kStudy = {sim::Effect::explosion, sim::Effect::mushroom, sim::Effect::wildfire, sim::Effect::launch, sim::Effect::smoke, sim::Effect::waterfall};

std::vector<sim::Effect> effects_of(const std::string& list) {
  if (list.empty() || list == "all") return {kStudy.begin(), kStudy.end()};
  std::vector<sim::Effect> v;
  for (const auto& w : words(list)) {
    sim::Effect e{};
    if (!sim::parse_effect(w, e)) throw std::invalid_argument("unknown effect " + w);
    v.push_back(e);
  }
  return v;
}
int effect_index(sim::Effect e) {
  for (std::size_t i = 0; i < kStudy.size(); ++i) {
    if (kStudy[i] == e) return static_cast<int>(i);
  }
  return -1;
}

struct Ctx {
  fs::path data, results;
  int threads = 2;
  fs::path dir() const { return data / "p_upscale"; }
  fs::path model(sim::Effect e) const { return dir() / "models" / std::format("{}.nvfx", ename(e)); }
  fs::path net(const std::string& name) const {
    if (name.find('/') != std::string::npos || name.ends_with(".nvfxup")) return fs::path(name);
    return dir() / "nets" / std::format("{}.nvfxup", name);
  }
};

using Setting = std::array<float, 3>;
bool off_grid(const Setting& s) {
  const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
  return off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f});
}
// Study B's held-out settings and study G's validation settings, extended as study M3's (tools/nvfx_ordered.cpp).
std::vector<Setting> test_settings(int n) {
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < sz(n)) {
    const Setting s{u(rng), u(rng), u(rng)};
    if (off_grid(s)) v.push_back(s);
  }
  return v;
}
std::vector<Setting> validation_settings(int n, int n_test = 30) {
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
std::vector<Setting> training_settings(sim::Effect e, int n, int first) {
  rollout::SimRecipe r = rollout::recipe_for(e);
  r.salt = 1;
  std::vector<Setting> v;
  for (int i = first; i < first + n; ++i) {
    const sim::Params p = rollout::recipe_run(r, static_cast<std::uint64_t>(i));
    v.push_back({p.intensity, p.wind, p.turbulence});
  }
  return v;
}
int frames_of(sim::Effect e) { return sim::effect_loops(e) ? 300 : sim::one_shot_frames(e) - 1; }
int warm_of(sim::Effect e) { return sim::effect_loops(e) ? 150 : 1; }
int track_warm(sim::Effect e) { return sim::effect_loops(e) ? 100 : 1; }

// The teacher at a setting, drawn at `size` px from its 128-cell grid.
sim::Params setting_params(sim::Effect e, const Setting& s, std::uint64_t seed, int size) {
  sim::Params q;
  q.effect = e;
  q.intensity = s[0];
  q.wind = s[1];
  q.turbulence = s[2];
  q.seed = seed;
  q.size = size;
  q.sim_res = kTeacher;
  return q;
}

struct Split {
  std::string name;
  std::vector<Setting> settings;
  std::uint64_t real = 0, play = 0, null = 0, track = 0;
};
Split split_of(const std::string& name, sim::Effect e, int n) {
  Split s;
  s.name = name;
  if (name == "train") {
    s.settings = training_settings(e, n > 0 ? n : 16, 0);
    s.play = 3'600'000;
  } else if (name == "dev") {
    s.settings = training_settings(e, n > 0 ? n : 8, 16);
    s.play = 3'650'000;
    s.real = 3'660'000;
    s.track = 3'670'000;
  } else if (name == "val") {
    s.settings = validation_settings(n > 0 ? n : 10);
    s.real = 3'700'000;
    s.play = 3'710'000;
    s.null = 3'720'000;
    s.track = 3'730'000;
  } else if (name == "test") {
    s.settings = test_settings(n > 0 ? n : 10);
    s.real = 3'800'000;
    s.play = 3'810'000;
    s.track = 3'830'000;
  } else {
    throw std::invalid_argument("--split train, dev, val or test");
  }
  return s;
}
std::uint64_t seed_of(std::uint64_t base, int si, int k) { return base + 100 * static_cast<std::uint64_t>(si) + static_cast<std::uint64_t>(k); }

// --- the runtime ------------------------------------------------------------------------------------------------------

struct Loaded {
  nvfx_effect* e = nullptr;
  explicit Loaded(const fs::path& p) {
    if (nvfx_effect_load(p.string().c_str(), &e) != NVFX_OK) throw std::runtime_error("cannot load " + p.string());
  }
  explicit Loaded(const rollout::Model& m) {
    std::ostringstream os;
    if (auto r = rollout::save_model(os, m); !r) throw std::runtime_error(r.error());
    const std::string b = os.str();
    if (nvfx_effect_load_memory(b.data(), b.size(), &e) != NVFX_OK) throw std::runtime_error("runtime load failed");
  }
  ~Loaded() { nvfx_effect_free(e); }
  Loaded(const Loaded&) = delete;
  Loaded& operator=(const Loaded&) = delete;
};

struct Instance {
  nvfx_instance* p = nullptr;
  int size = 0;
  // variation >= 0: that start point with its run's seed, one continuous rollout (tracking); else a seed, shards.
  Instance(nvfx_effect* e, int s, const Setting& ctl, std::uint64_t seed, int variation = -1) : size(s) {
    if (nvfx_instance_create(e, s, &p) != NVFX_OK) throw std::runtime_error("instance");
    nvfx_instance_set_controls(p, ctl.data(), 3);
    if (variation >= 0) {
      nvfx_instance_set_drift(p, 0.f);
      nvfx_instance_set_variation(p, variation);
    } else {
      nvfx_instance_set_seed(p, seed);
    }
  }
  ~Instance() { nvfx_instance_free(p); }
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;
  void frame(int f, std::span<std::uint8_t> out) { nvfx_render(p, static_cast<double>(f) / kFps, out.data(), sz(size) * 4); }
};

// The runtime's upscaler for an S / 2 instance (the ISA of its runners, as nvfx_instance_set_upscale makes it).
std::unique_ptr<rt::Upscaler> runtime_upscaler(const upscale::Net* net, int n, nvfx_isa isa) {
  switch (isa) {
    case NVFX_ISA_AVX512: return rt::isa_avx512::make_upscaler(net, n);
    case NVFX_ISA_AVX2: return rt::isa_avx2::make_upscaler(net, n);
    default: return rt::isa_base::make_upscaler(net, n);
  }
}

upscale::Net load_net(const fs::path& p) {
  auto n = upscale::load_net(p);
  if (!n) throw std::runtime_error(p.string() + ": " + n.error());
  return std::move(*n);
}

// --- the statistics of metrics::stats, accumulated frame by frame ----------------------------------------------------------

// A 2D FFT of a real n x n plane (n a power of two), in doubles: along y for every x at once (the inner loops run along x
// and vectorise), a transpose, and again. The power at (i, j) of the result is the plane's at (j, i) or (i, j): the
// radial bins do not care which.
class Fft2 {
 public:
  explicit Fft2(int n) : n_(n), re_(sz(n) * sz(n)), im_(sz(n) * sz(n)), tr_(sz(n) * sz(n)), ti_(sz(n) * sz(n)), rev_(sz(n)), cw_(sz(n / 2)), sw_(sz(n / 2)) {
    int bits = 0;
    while ((1 << bits) < n) ++bits;
    for (int i = 0; i < n; ++i) {
      int r = 0;
      for (int b = 0; b < bits; ++b) r |= ((i >> b) & 1) << (bits - 1 - b);
      rev_[sz(i)] = r;
    }
    for (int j = 0; j < n / 2; ++j) {
      const double a = -2.0 * 3.14159265358979323846 * j / n;
      cw_[sz(j)] = std::cos(a);
      sw_[sz(j)] = std::sin(a);
    }
  }
  std::vector<double>& re() { return re_; }
  // In place on re() (imaginary part zero); the power of each coefficient into pw (n * n).
  void power(std::vector<double>& pw) {
    std::ranges::fill(im_, 0.0);
    pass(re_, im_);
    transpose(re_, tr_);
    transpose(im_, ti_);
    pass(tr_, ti_);
    for (std::size_t i = 0; i < pw.size(); ++i) pw[i] = tr_[i] * tr_[i] + ti_[i] * ti_[i];
  }

 private:
  void transpose(const std::vector<double>& a, std::vector<double>& b) const {
    const int n = n_, B = 16;
    for (int y0 = 0; y0 < n; y0 += B) {
      for (int x0 = 0; x0 < n; x0 += B) {
        for (int y = y0; y < std::min(n, y0 + B); ++y) {
          for (int x = x0; x < std::min(n, x0 + B); ++x) b[sz(x) * sz(n) + sz(y)] = a[sz(y) * sz(n) + sz(x)];
        }
      }
    }
  }
  [[gnu::target("avx2,fma")]] void pass(std::vector<double>& re, std::vector<double>& im) {
    const std::size_t n = sz(n_);
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t j = sz(rev_[i]);
      if (i < j) {
        std::swap_ranges(re.begin() + static_cast<std::ptrdiff_t>(i * n), re.begin() + static_cast<std::ptrdiff_t>(i * n + n), re.begin() + static_cast<std::ptrdiff_t>(j * n));
        std::swap_ranges(im.begin() + static_cast<std::ptrdiff_t>(i * n), im.begin() + static_cast<std::ptrdiff_t>(i * n + n), im.begin() + static_cast<std::ptrdiff_t>(j * n));
      }
    }
    for (std::size_t len = 2; len <= n; len <<= 1) {
      const std::size_t half = len / 2, step = n / len;
      for (std::size_t i = 0; i < n; i += len) {
        for (std::size_t j = 0; j < half; ++j) {
          const double wr = cw_[j * step], wi = sw_[j * step];
          double* __restrict ar = re.data() + (i + j) * n;
          double* __restrict ai = im.data() + (i + j) * n;
          double* __restrict br = re.data() + (i + j + half) * n;
          double* __restrict bi = im.data() + (i + j + half) * n;
          for (std::size_t x = 0; x < n; ++x) {
            const double vr = br[x] * wr - bi[x] * wi, vi = br[x] * wi + bi[x] * wr;
            const double ur = ar[x], ui = ai[x];
            ar[x] = ur + vr;
            ai[x] = ui + vi;
            br[x] = ur - vr;
            bi[x] = ui - vi;
          }
        }
      }
    }
  }
  int n_;
  std::vector<double> re_, im_, tr_, ti_;
  std::vector<int> rev_;
  std::vector<double> cw_, sw_;
};

// metrics::stats without the clip: coverage and emission per frame, the radially averaged log luminance power (with the
// same normalisation: power summed per bin over frames, divided by the bin's count over frames and by the frames), the
// mean frame and the motion.
class StatAcc {
 public:
  explicit StatAcc(int n) : n_(n), fft_(n), pw_(sz(n) * sz(n)), spec_(sz(n / 2), 0.0), mean_(sz(n) * sz(n) * 4, 0.f), prev_(sz(n) * sz(n) * 4, 0), bin_(sz(n) * sz(n), -1) {
    count_.assign(sz(n / 2), 0.0);
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) {
        const int ky = y <= n / 2 ? y : y - n, kx = x <= n / 2 ? x : x - n;
        const int r = static_cast<int>(std::lround(std::sqrt(double(kx * kx + ky * ky))));
        if (r < 1 || r > n / 2) continue;
        bin_[sz(y) * sz(n) + sz(x)] = r - 1;
        count_[sz(r - 1)] += 1;
      }
    }
  }
  void add(std::span<const std::uint8_t> fr) {
    const std::size_t px = sz(n_) * sz(n_);
    double cov = 0, emi = 0;
    auto& plane = fft_.re();
    for (std::size_t i = 0; i < px; ++i) {
      const double a = fr[i * 4 + 3] / 255.0;
      cov += a;
      for (int c = 0; c < 3; ++c) emi += std::max(0.0, fr[i * 4 + sz(c)] / 255.0 - a) / 3.0;
      for (int c = 0; c < 4; ++c) mean_[i * 4 + sz(c)] += static_cast<float>(fr[i * 4 + sz(c)]);
      plane[i] = 0.2126 * fr[i * 4] + 0.7152 * fr[i * 4 + 1] + 0.0722 * fr[i * 4 + 2];
    }
    coverage_.push_back(cov / static_cast<double>(px));
    emission_.push_back(emi / static_cast<double>(px));
    if (frames_ > 0) {
      double d = 0;
      for (std::size_t i = 0; i < fr.size(); ++i) d += std::abs(double(fr[i]) - double(prev_[i]));
      motion_ += d / 255.0 / static_cast<double>(fr.size());
    }
    std::ranges::copy(fr, prev_.begin());
    fft_.power(pw_);
    for (std::size_t i = 0; i < px; ++i) {
      if (bin_[i] >= 0) spec_[sz(bin_[i])] += pw_[i];
    }
    ++frames_;
  }
  metrics::ClipStats finish() const {
    metrics::ClipStats s;
    s.coverage = coverage_;
    s.emission = emission_;
    s.mean_frame = mean_;
    for (float& v : s.mean_frame) v /= static_cast<float>(frames_);
    for (std::size_t b = 0; b < spec_.size(); ++b) s.spectrum.push_back(std::log10(1e-9 + spec_[b] / std::max(1.0, count_[b] * frames_) / frames_));
    s.motion = frames_ > 1 ? motion_ / (frames_ - 1) : 0;
    return s;
  }

 private:
  int n_, frames_ = 0;
  Fft2 fft_;
  std::vector<double> pw_, spec_, count_, coverage_, emission_;
  std::vector<float> mean_;
  std::vector<std::uint8_t> prev_;
  std::vector<int> bin_;
  double motion_ = 0;
};

// The statistics of several runs pooled (study D2's protocol, as tools/nvfx_ordered.cpp's pool).
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

// The high band: radial bins above the S / 2 frame's Nyquist frequency (bins S / 4 + 1 to S / 2, 1-based), which an S / 2
// instance cannot draw. Mean |a - b| of the log power there, and the mean of a - b (the log ratio of the band's power).
double hf_l1(const std::vector<double>& a, const std::vector<double>& b) {
  const std::size_t n = std::min(a.size(), b.size()), lo = n / 2;
  double s = 0;
  for (std::size_t i = lo; i < n; ++i) s += std::abs(a[i] - b[i]);
  return n > lo ? s / static_cast<double>(n - lo) : 0.0;
}
double hf_log_ratio(const std::vector<double>& a, const std::vector<double>& b) {
  const std::size_t n = std::min(a.size(), b.size()), lo = n / 2;
  double s = 0;
  for (std::size_t i = lo; i < n; ++i) s += a[i] - b[i];
  return n > lo ? s / static_cast<double>(n - lo) : 0.0;
}

// Cached statistics (DATA/p_upscale/cache): the mean frame in half precision (4 MB a pool at 512 px otherwise).
void put_stats(std::ostream& o, const metrics::ClipStats& s) {
  const auto vec = [&](const std::vector<double>& v) {
    const auto n = static_cast<std::uint32_t>(v.size());
    o.write(reinterpret_cast<const char*>(&n), 4);
    o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * 8));
  };
  vec(s.coverage);
  vec(s.emission);
  vec(s.spectrum);
  const auto n = static_cast<std::uint32_t>(s.mean_frame.size());
  o.write(reinterpret_cast<const char*>(&n), 4);
  std::vector<std::uint16_t> h(s.mean_frame.size());
  for (std::size_t i = 0; i < h.size(); ++i) h[i] = std::bit_cast<std::uint16_t>(static_cast<std::float16_t>(s.mean_frame[i]));
  o.write(reinterpret_cast<const char*>(h.data()), static_cast<std::streamsize>(h.size() * 2));
  o.write(reinterpret_cast<const char*>(&s.motion), 8);
}
bool get_stats(const fs::path& p, metrics::ClipStats& s) {
  std::ifstream i(p, std::ios::binary);
  if (!i) return false;
  const auto vec = [&](std::vector<double>& v) {
    std::uint32_t n = 0;
    i.read(reinterpret_cast<char*>(&n), 4);
    v.resize(n);
    i.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 8));
  };
  vec(s.coverage);
  vec(s.emission);
  vec(s.spectrum);
  std::uint32_t n = 0;
  i.read(reinterpret_cast<char*>(&n), 4);
  std::vector<std::uint16_t> h(n);
  i.read(reinterpret_cast<char*>(h.data()), static_cast<std::streamsize>(h.size() * 2));
  s.mean_frame.resize(n);
  for (std::size_t k = 0; k < h.size(); ++k) s.mean_frame[k] = static_cast<float>(std::bit_cast<std::float16_t>(h[k]));
  i.read(reinterpret_cast<char*>(&s.motion), 8);
  return static_cast<bool>(i);
}
// The mean frame as the cache keeps it, so that a pool used fresh and the same pool read back give the same numbers.
void round_mean_frame(metrics::ClipStats& s) {
  for (float& v : s.mean_frame) v = static_cast<float>(static_cast<std::float16_t>(v));
}

// Active PSNR accumulated over frames (study D's: pixels visible in either frame), and SSIM every 10th frame.
struct PairAcc {
  double sum = 0, count = 0;
  std::vector<double> ssim;
  void add(std::span<const std::uint8_t> r, std::span<const std::uint8_t> t, int size, bool with_ssim) {
    for (std::size_t i = 0; i < r.size(); i += 4) {
      if (std::max({r[i], r[i + 1], r[i + 2], r[i + 3]}) <= 4 && std::max({t[i], t[i + 1], t[i + 2], t[i + 3]}) <= 4) continue;
      count += 4;
      for (std::size_t c = 0; c < 4; ++c) {
        const double d = (static_cast<double>(r[i + c]) - static_cast<double>(t[i + c])) / 255.0;
        sum += d * d;
      }
    }
    if (with_ssim) ssim.push_back(metrics::ssim(r, t, size));
  }
  double psnr() const { return count > 0 ? metrics::psnr_from_mse(sum / count) : metrics::kPsnrCap; }
  double mean_ssim() const { return mean(ssim); }
};

std::string o_str(const PairAcc& l) { return l.count > 0 ? std::format(", L vs N {:.2f} dB", l.psnr()) : std::string(); }

double apsnr(std::span<const std::uint8_t> r, std::span<const std::uint8_t> t) {
  PairAcc a;
  a.add(r, t, 0, false);
  return a.psnr();
}

// --- arms -----------------------------------------------------------------------------------------------------------------

// An upscaling arm: a name and a net (nullptr: the bilinear 2x, arm B).
struct Arm {
  std::string name;
  std::shared_ptr<const upscale::Net> net;
};
// Arms built one by one (an initializer list of one Arm trips a false -Warray-bounds in GCC 14).
std::vector<Arm> arms_list(std::shared_ptr<const upscale::Net> l = nullptr, const std::string& lname = "L") {
  std::vector<Arm> v;
  v.push_back(Arm{"B", nullptr});
  if (l) v.push_back(Arm{lname, std::move(l)});
  return v;
}
std::vector<Arm> arms_of(const Ctx& c, const std::string& cands) {
  std::vector<Arm> v = arms_list();
  for (const auto& w : words(cands)) {
    const auto eq = w.find('=');
    const std::string name = eq == std::string::npos ? w : w.substr(0, eq), file = eq == std::string::npos ? w : w.substr(eq + 1);
    v.push_back({"L_" + name, std::make_shared<upscale::Net>(load_net(c.net(file)))});
  }
  return v;
}
// Per-effect candidates: a file name with {effect} is the effect's own net.
std::vector<Arm> arms_for(const Ctx& c, const std::string& cands, sim::Effect e) {
  std::string s = cands;
  for (std::size_t p; (p = s.find("{effect}")) != std::string::npos;) s.replace(p, 8, ename(e));
  return arms_of(c, s);
}

// One play: N at S and the S / 2 instance with the same seed and controls, frame by frame; each arm's frame from the S / 2
// frame; f(frame index, N's frame, the arms' frames).
template <class F>
void play_pair(nvfx_effect* fx, int S, const Setting& ctl, std::uint64_t seed, int frames, const std::vector<Arm>& arms, int variation, F&& f) {
  Instance N(fx, S, ctl, seed, variation), H(fx, S / 2, ctl, seed, variation);
  const nvfx_isa isa = rt::rollout_instance_isa(H.p);
  std::vector<std::unique_ptr<rt::Upscaler>> ups;
  for (const Arm& a : arms) ups.push_back(runtime_upscaler(a.net.get(), S / 2, isa));
  std::vector<std::uint8_t> n(sz(S) * sz(S) * 4), h(sz(S / 2) * sz(S / 2) * 4);
  std::vector<std::vector<std::uint8_t>> out(arms.size(), std::vector<std::uint8_t>(sz(S) * sz(S) * 4));
  const rt::FrameInput plain;
  // tracking (variation >= 0) replays the start point with its run's seed: the instance's own seed is 0 then
  const std::uint64_t useed = variation >= 0 ? 0 : seed;
  for (int i = 0; i < frames; ++i) {
    N.frame(i, n);
    H.frame(i, h);
    for (std::size_t a = 0; a < arms.size(); ++a) ups[a]->run(h.data(), sz(S / 2) * 4, useed, static_cast<double>(i) / kFps, plain, out[a].data(), sz(S) * 4);
    f(i, std::span<const std::uint8_t>(n), out);
  }
}

// --- check ----------------------------------------------------------------------------------------------------------------

int step_check(const Ctx& c) {
  bool ok = true;
  // 1. the simulator's dynamics do not depend on the size it draws at (its grid is sim_res)
  for (const sim::Effect e : {sim::Effect::cloud, sim::Effect::explosion}) {
    const Setting s{0.6f, 0.4f, 0.55f};
    sim::Fluid a(setting_params(e, s, 3'900'001, 128)), b(setting_params(e, s, 3'900'001, 512));
    for (int i = 0; i < 40; ++i) {
      a.step_frame();
      b.step_frame();
    }
    const sim::State sa = a.state(), sb = b.state();
    const bool same = sa.temp == sb.temp && sa.soot == sb.soot && sa.u == sb.u && sa.v == sb.v;
    std::println("simulator {}: state after 40 frames drawn at 128 and at 512 px {}", ename(e), same ? "the same to the bit" : "DIFFERS");
    ok &= same;
  }
  // 2. an instance's coarse state does not depend on its size: runners at 512 and 256 px, the same start and seed
  for (const sim::Effect e : kStudy) {
    rt::RolloutEffect fx;
    auto m = rollout::load_model(c.model(e));
    if (!m) throw std::runtime_error(m.error());
    fx.m = std::move(*m);
    const nvfx_isa isa = rt::rollout_isa(rt::default_rollout_precision());
    const auto make = [&](int size) {
      switch (isa) {
        case NVFX_ISA_AVX512: return rt::isa_avx512::make_rollout(fx, size, false, rt::default_rollout_precision(), rt::kInt8Convs);
        case NVFX_ISA_AVX2: return rt::isa_avx2::make_rollout(fx, size, false, rt::default_rollout_precision(), rt::kInt8Convs);
        default: return rt::isa_base::make_rollout(fx, size, false, rt::default_rollout_precision(), rt::kInt8Convs);
      }
    };
    auto a = make(512), b = make(256);
    const std::array<float, 3> ctl{0.6f, 0.5f, 0.6f};
    a->start(0, ctl, 3'900'002);
    b->start(0, ctl, 3'900'002);
    bool same = true;
    for (int f = 0; f < 120 && same; ++f) {
      a->step(ctl, 3'900'002);
      b->step(ctl, 3'900'002);
      same = std::ranges::equal(a->coarse(), b->coarse());
    }
    std::println("{}: coarse state at 512 and 256 px over 120 frames {}", ename(e), same ? "the same to the bit" : "DIFFERS");
    ok &= same;
  }
  // 3. an instance that upscales draws what the study's arms draw
  {
    const sim::Effect e = sim::Effect::smoke;
    const Loaded fx(c.model(e));
    upscale::Net net = upscale::init_net(8, 8, 1, 7);
    net.gain = 1.f;
    for (std::size_t i = net.layout().wo; i < net.w.size(); ++i) net.w[i] += 0.05f;  // some residual and micro-detail
    std::ostringstream os;
    (void)upscale::save_net(os, net);
    const std::string bytes = os.str();
    Loaded fx2(c.model(e));
    if (nvfx_effect_attach_upscaler_memory(fx2.e, bytes.data(), bytes.size()) != NVFX_OK) throw std::runtime_error("attach");
    const Setting ctl{0.6f, 0.5f, 0.6f};
    Instance I(fx2.e, 256, ctl, 3'900'003);
    if (nvfx_instance_set_upscale(I.p, NVFX_UPSCALE_LEARNED) != NVFX_OK) throw std::runtime_error("set_upscale");
    std::vector<Arm> arms;
    arms.push_back(Arm{"L", std::make_shared<upscale::Net>(net)});
    std::vector<std::uint8_t> got(256 * 256 * 4);
    bool same = true;
    play_pair(fx.e, 256, ctl, 3'900'003, 200, arms, -1, [&](int f, std::span<const std::uint8_t>, const std::vector<std::vector<std::uint8_t>>& out) {
      I.frame(f, got);
      same &= got == out[0];
    });
    std::println("an instance with nvfx_instance_set_upscale against the S / 2 instance and the upscaler, 200 frames: {}", same ? "the same to the bit" : "DIFFERS");
    ok &= same;
  }
  return ok ? 0 : 1;
}

// --- pin and info ---------------------------------------------------------------------------------------------------------

std::string file_sha(const fs::path& p) {
  std::ifstream i(p, std::ios::binary);
  const std::vector<char> b((std::istreambuf_iterator<char>(i)), std::istreambuf_iterator<char>());
  return pack::hex(pack::sha256(std::span(reinterpret_cast<const std::uint8_t*>(b.data()), b.size())));
}

int step_pin(const Ctx& c) {
  fs::create_directories(c.dir() / "models");
  fs::create_directories(c.results);
  std::ofstream o(c.results / "p_models.csv");
  o << "effect,source,bytes,sha256\n";
  for (const sim::Effect e : kStudy) {
    const fs::path src = fs::canonical(c.data / "library" / "models" / std::format("{}.nvfx", ename(e)));
    const fs::path dst = c.model(e);
    if (!fs::exists(dst)) fs::copy_file(src, dst);
    const std::string a = file_sha(src), b = file_sha(dst);
    if (a != b) throw std::runtime_error(std::format("{}: the pinned copy differs from {}", ename(e), src.string()));
    o << std::format("{},{},{},{}\n", ename(e), fs::relative(src, c.data).string(), fs::file_size(dst), b);
    std::println("{}: {} ({} bytes) {}", ename(e), fs::relative(src, c.data).string(), fs::file_size(dst), b.substr(0, 16));
  }
  return 0;
}

int step_info(const tools::Args& a, const Ctx& c) {
  const int size = a.i("size", 128);
  std::vector<std::pair<double, std::string>> rows;
  std::mutex mu;
  const std::vector<sim::Effect> all(sim::kAllEffects.begin(), sim::kAllEffects.end());
  parallel(c.threads, static_cast<int>(all.size()), [&](int k) {
    const sim::Effect e = all[sz(k)];
    const Loaded fx(c.data / "library" / "models" / std::format("{}.nvfx", ename(e)));
    const auto set = training_settings(e, 8, 0);
    double cov = 0, cov90 = 0;
    for (std::size_t si = 0; si < set.size(); ++si) {
      Instance I(fx.e, size, set[si], 3'900'100 + si);
      std::vector<std::uint8_t> fr(sz(size) * sz(size) * 4);
      const int F = frames_of(e);
      double s = 0, any = 0;
      for (int f = 0; f < F; ++f) {
        I.frame(f, fr);
        double t = 0, v = 0;
        for (std::size_t i = 0; i < fr.size(); i += 4) {
          t += fr[i + 3] / 255.0;
          v += std::max({fr[i], fr[i + 1], fr[i + 2], fr[i + 3]}) > 4 ? 1.0 : 0.0;
        }
        s += t / static_cast<double>(fr.size() / 4);
        any += v / static_cast<double>(fr.size() / 4);
      }
      cov += s / F / static_cast<double>(set.size());
      cov90 += any / F / static_cast<double>(set.size());
    }
    std::lock_guard lk(mu);
    rows.emplace_back(cov90, std::format("{:>10} mean alpha {:.3f}, pixels drawn {:.3f}", ename(e), cov, cov90));
  });
  std::ranges::sort(rows, std::greater{});
  std::println("every shipped effect at 8 training settings, {} px (pixels drawn: any channel above 4 of 255):", size);
  for (const auto& r : rows) std::println("{}", r.second);
  return 0;
}

// --- tiles ----------------------------------------------------------------------------------------------------------------

constexpr int kM = 24, kC = 3, kT = kM + 2 * kC;  // a tile: 24 x 24 half-size pixels (48 x 48 full), 3 pixels of context
struct Tile {
  std::int32_t x0 = 0, y0 = 0, n = 0, effect = 0;
  std::vector<std::uint8_t> lo, hi;  // [kT][kT][4] from (x0 - kC, y0 - kC), zero outside the frame; [2 kM][2 kM][4]
};
constexpr char kTileMagic[8] = {'P', 'T', 'I', 'L', 'E', 'S', '0', '1'};

void save_tiles(const fs::path& p, const std::vector<Tile>& v) {
  std::ofstream o(p, std::ios::binary);
  o.write(kTileMagic, 8);
  const auto n = static_cast<std::uint32_t>(v.size());
  o.write(reinterpret_cast<const char*>(&n), 4);
  for (const Tile& t : v) {
    o.write(reinterpret_cast<const char*>(&t.x0), 16);
    o.write(reinterpret_cast<const char*>(t.lo.data()), static_cast<std::streamsize>(t.lo.size()));
    o.write(reinterpret_cast<const char*>(t.hi.data()), static_cast<std::streamsize>(t.hi.size()));
  }
}
std::vector<Tile> load_tiles(const fs::path& p) {
  std::ifstream i(p, std::ios::binary);
  char m[8] = {};
  i.read(m, 8);
  if (!i || !std::equal(m, m + 8, kTileMagic)) throw std::runtime_error("not a tile file: " + p.string());
  std::uint32_t n = 0;
  i.read(reinterpret_cast<char*>(&n), 4);
  std::vector<Tile> v(n);
  for (Tile& t : v) {
    i.read(reinterpret_cast<char*>(&t.x0), 16);
    t.lo.resize(sz(kT) * kT * 4);
    t.hi.resize(sz(2 * kM) * 2 * kM * 4);
    i.read(reinterpret_cast<char*>(t.lo.data()), static_cast<std::streamsize>(t.lo.size()));
    i.read(reinterpret_cast<char*>(t.hi.data()), static_cast<std::streamsize>(t.hi.size()));
  }
  if (!i) throw std::runtime_error("truncated tile file: " + p.string());
  return v;
}
fs::path tile_file(const Ctx& c, sim::Effect e, int S) { return c.dir() / "tiles" / std::format("{}_{}.tiles", ename(e), S); }

int step_tiles(const tools::Args& a, const Ctx& c) {
  const auto sizes = ints(a.str("sizes", "256,512"));
  const int per = a.i("per-frame", 8), nset = a.i("settings", 16), every = a.i("every", 15);
  fs::create_directories(c.dir() / "tiles");
  for (const sim::Effect e : effects_of(a.str("effects", "all"))) {
    const Loaded fx(c.model(e));
    const Split sp = split_of("train", e, nset);
    for (const int S : sizes) {
      const fs::path out = tile_file(c, e, S);
      if (fs::exists(out)) {
        std::println("{} {}: tiles exist", ename(e), S);
        continue;
      }
      const auto t0 = std::chrono::steady_clock::now();
      std::vector<std::vector<Tile>> per_set(sp.settings.size());
      const int F = frames_of(e), step = sim::effect_loops(e) ? every : std::max(1, F / 20);
      const int n = S / 2;
      parallel(c.threads, static_cast<int>(sp.settings.size()), [&](int si) {
        const std::uint64_t seed = seed_of(sp.play, si, 0);
        Instance N(fx.e, S, sp.settings[sz(si)], seed), H(fx.e, n, sp.settings[sz(si)], seed);
        std::vector<std::uint8_t> nf(sz(S) * sz(S) * 4), hf(sz(n) * sz(n) * 4);
        std::mt19937_64 rng(seed);
        for (int f = 0; f < F; ++f) {
          N.frame(f, nf);
          H.frame(f, hf);
          if (f % step != step / 2) continue;
          for (int k = 0; k < per; ++k) {
            Tile t;
            t.n = n;
            t.effect = effect_index(e);
            for (int tries = 0; tries < 30; ++tries) {
              t.x0 = static_cast<std::int32_t>(rng() % sz(n - kM + 1));
              t.y0 = static_cast<std::int32_t>(rng() % sz(n - kM + 1));
              int mx = 0;
              for (int y = 0; y < kM; ++y) {
                for (int x = 0; x < kM * 4; ++x) mx = std::max<int>(mx, hf[(sz(t.y0 + y) * sz(n) + sz(t.x0)) * 4 + sz(x)]);
              }
              if (mx > 8) break;
            }
            t.lo.assign(sz(kT) * kT * 4, 0);
            for (int y = 0; y < kT; ++y) {
              for (int x = 0; x < kT; ++x) {
                const int X = t.x0 - kC + x, Y = t.y0 - kC + y;
                if (X < 0 || Y < 0 || X >= n || Y >= n) continue;
                for (int ch = 0; ch < 4; ++ch) t.lo[(sz(y) * kT + sz(x)) * 4 + sz(ch)] = hf[(sz(Y) * sz(n) + sz(X)) * 4 + sz(ch)];
              }
            }
            t.hi.assign(sz(2 * kM) * 2 * kM * 4, 0);
            for (int y = 0; y < 2 * kM; ++y) {
              std::memcpy(t.hi.data() + sz(y) * 2 * kM * 4, nf.data() + (sz(2 * t.y0 + y) * sz(S) + sz(2 * t.x0)) * 4, 2 * kM * 4);
            }
            per_set[sz(si)].push_back(std::move(t));
          }
        }
      });
      std::vector<Tile> all;
      for (auto& v : per_set) std::ranges::move(v, std::back_inserter(all));
      save_tiles(out, all);
      std::println("{} {}: {} tiles ({:.0f} s)", ename(e), S, all.size(), seconds_since(t0));
      std::fflush(stdout);
    }
  }
  return 0;
}

// --- training -------------------------------------------------------------------------------------------------------------

// Forward and backward of the upscaler on one tile (the reference's operations: src/core/upscale.cpp), with zero padding of
// the convolutions at the frame's edges and the bilinear 2x clamped there. The loss: squared error of up + r against N's
// pixels where up is not empty and either is visible; after `amp` is set, the amplitudes' squared error against the RMS of
// the high-passed remaining error over each 2 x 2 block (interior blocks), times amp_w.
struct Work {
  std::vector<float> x, h1, h1p, h2, h2p, o, up, d_o, d_h2, d_h1;
};
struct TileLoss {
  double sse = 0, n = 0, amp = 0, an = 0;
  double hf_abs = 0, hf_log = 0, hn = 0;  // |ln| and ln of the tile's high-pass energy against N's, per channel with energy
};

TileLoss tile_pass(const upscale::Net& net, const Tile& t, bool amp, float amp_w, std::vector<float>* grad, Work& w, float hf_w = 0.f) {
  const auto L = net.layout();
  const int c1 = net.c1, c2 = net.c2, k2 = net.k2, r2 = k2 / 2;
  const int A = kM + 2 * r2;  // the first layer over [-r2, kM + r2)^2
  const float* W = net.w.data();
  w.x.assign(sz(kT) * kT * 4, 0.f);
  for (std::size_t i = 0; i < w.x.size(); ++i) w.x[i] = static_cast<float>(t.lo[i]) * (1.f / 255.f);
  const auto inframe = [&](int tx, int ty) {
    const int X = t.x0 + tx, Y = t.y0 + ty;
    return X >= 0 && Y >= 0 && X < t.n && Y < t.n;
  };
  const auto X = [&](int tx, int ty, int ch) { return w.x[(sz(ty + kC) * kT + sz(tx + kC)) * 4 + sz(ch)]; };
  // layer 1
  w.h1.assign(sz(A) * sz(A) * sz(c1), 0.f);
  w.h1p.assign(w.h1.size(), 0.f);
  for (int ay = 0; ay < A; ++ay) {
    for (int ax = 0; ax < A; ++ax) {
      const int tx = ax - r2, ty = ay - r2;
      if (!inframe(tx, ty)) continue;  // outside the frame the second layer sees nothing
      float* h = w.h1p.data() + (sz(ay) * sz(A) + sz(ax)) * sz(c1);
      for (int j = 0; j < c1; ++j) h[j] = W[L.b1 + sz(j)];
      for (int tp = 0; tp < 9; ++tp) {
        const int dx = tp % 3 - 1, dy = tp / 3 - 1;
        for (int i = 0; i < 4; ++i) {
          const float v = X(tx + dx, ty + dy, i);
          const float* wt = W + L.w1 + (sz(tp) * 4 + sz(i)) * sz(c1);
          for (int j = 0; j < c1; ++j) h[j] += wt[j] * v;
        }
      }
      float* hr = w.h1.data() + (sz(ay) * sz(A) + sz(ax)) * sz(c1);
      for (int j = 0; j < c1; ++j) hr[j] = std::max(0.f, h[j]);
    }
  }
  // layer 2 and the output
  w.h2.assign(sz(kM) * kM * sz(c2), 0.f);
  w.h2p.assign(w.h2.size(), 0.f);
  w.o.assign(sz(kM) * kM * upscale::kOut, 0.f);
  for (int y = 0; y < kM; ++y) {
    for (int x = 0; x < kM; ++x) {
      float* h = w.h2p.data() + (sz(y) * kM + sz(x)) * sz(c2);
      for (int j = 0; j < c2; ++j) h[j] = W[L.b2 + sz(j)];
      for (int tp = 0; tp < k2 * k2; ++tp) {
        const int ax = x + tp % k2, ay = y + tp / k2;  // (x + dx + r2, y + dy + r2)
        const float* hh = w.h1.data() + (sz(ay) * sz(A) + sz(ax)) * sz(c1);
        for (int i = 0; i < c1; ++i) {
          const float* wt = W + L.w2 + (sz(tp) * sz(c1) + sz(i)) * sz(c2);
          for (int j = 0; j < c2; ++j) h[j] += wt[j] * hh[i];
        }
      }
      float* hr = w.h2.data() + (sz(y) * kM + sz(x)) * sz(c2);
      float* o = w.o.data() + (sz(y) * kM + sz(x)) * upscale::kOut;
      for (int k = 0; k < upscale::kOut; ++k) o[k] = W[L.bo + sz(k)];
      for (int i = 0; i < c2; ++i) {
        hr[i] = std::max(0.f, h[i]);
        const float* wt = W + L.wo + sz(i) * upscale::kOut;
        for (int k = 0; k < upscale::kOut; ++k) o[k] += wt[k] * hr[i];
      }
    }
  }
  // the bilinear 2x (clamped at the frame's edges) and the error
  const int N = 2 * kM;
  w.up.assign(sz(N) * N * 4, 0.f);
  const auto B = [&](int tx, int ty, int ch) {
    const int Xc = std::clamp(t.x0 + tx, 0, t.n - 1) - t.x0, Yc = std::clamp(t.y0 + ty, 0, t.n - 1) - t.y0;
    return X(Xc, Yc, ch);
  };
  w.d_o.assign(w.o.size(), 0.f);
  TileLoss tl;
  std::vector<float> err(sz(N) * N * 4, 0.f);
  std::vector<std::uint8_t> live(sz(N) * N, 0);
  for (int y = 0; y < N; ++y) {
    const int yi = y / 2, yj = y % 2 ? yi + 1 : yi - 1;
    for (int x = 0; x < N; ++x) {
      const int xi = x / 2, xj = x % 2 ? xi + 1 : xi - 1;
      float* u = w.up.data() + (sz(y) * N + sz(x)) * 4;
      bool any = false;
      for (int ch = 0; ch < 4; ++ch) {
        const float aa = 0.75f * B(xi, yi, ch) + 0.25f * B(xj, yi, ch);
        const float bb = 0.75f * B(xi, yj, ch) + 0.25f * B(xj, yj, ch);
        u[ch] = 0.75f * aa + 0.25f * bb;
        any |= u[ch] != 0.f;
      }
      const std::uint8_t* tg = t.hi.data() + (sz(y) * N + sz(x)) * 4;
      if (!any) {  // stays empty: its error is N's pixel
        for (int ch = 0; ch < 4; ++ch) err[(sz(y) * N + sz(x)) * 4 + sz(ch)] = static_cast<float>(tg[ch]) / 255.f;
        continue;
      }
      live[sz(y) * N + sz(x)] = 1;
      const float* o = w.o.data() + (sz(yi) * kM + sz(xi)) * upscale::kOut;
      const int s = (y % 2) * 2 + x % 2;
      bool vis = false;
      for (int ch = 0; ch < 4; ++ch) vis |= tg[ch] > 4 || u[ch] * 255.f > 4.f;
      for (int ch = 0; ch < 4; ++ch) {
        const float e = u[ch] + o[s * 4 + ch] - static_cast<float>(tg[ch]) / 255.f;
        err[(sz(y) * N + sz(x)) * 4 + sz(ch)] = -e;
        if (!vis) continue;
        tl.sse += static_cast<double>(e) * e;
        tl.n += 1;
        w.d_o[(sz(yi) * kM + sz(xi)) * upscale::kOut + sz(s * 4 + ch)] += 2.f * e;
      }
    }
  }
  // the high-pass energy (each pixel minus its 3 x 3 mean, inside the tile) of the output against N's, per channel; with
  // hf_w, a loss on its log ratio (weighted by the tile's pixels in play), whose gradient reaches the live pixels' residuals
  {
    std::vector<float> yh(sz(N) * N * 4, 0.f);  // the output before clamping (empty pixels zero)
    for (int y = 0; y < N; ++y) {
      for (int x = 0; x < N; ++x) {
        if (!live[sz(y) * N + sz(x)]) continue;
        const float* o = w.o.data() + (sz(y / 2) * kM + sz(x / 2)) * upscale::kOut;
        const int s = (y % 2) * 2 + x % 2;
        for (int ch = 0; ch < 4; ++ch) yh[(sz(y) * N + sz(x)) * 4 + sz(ch)] = w.up[(sz(y) * N + sz(x)) * 4 + sz(ch)] + o[s * 4 + ch];
      }
    }
    const auto hp = [&](const auto& img, int x, int y, int ch) {
      float m = 0.f;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) m += img(x + dx, y + dy, ch);
      }
      return img(x, y, ch) - m / 9.f;
    };
    const auto out_px = [&](int x, int y, int ch) { return yh[(sz(y) * N + sz(x)) * 4 + sz(ch)]; };
    const auto tgt_px = [&](int x, int y, int ch) { return static_cast<float>(t.hi[(sz(y) * N + sz(x)) * 4 + sz(ch)]) / 255.f; };
    std::vector<float> h(sz(N) * N, 0.f);
    for (int ch = 0; ch < 4; ++ch) {
      double eo = 0, en = 0;
      for (int y = 1; y + 1 < N; ++y) {
        for (int x = 1; x + 1 < N; ++x) {
          const float a = hp(out_px, x, y, ch), b = hp(tgt_px, x, y, ch);
          h[sz(y) * N + sz(x)] = a;
          eo += static_cast<double>(a) * a;
          en += static_cast<double>(b) * b;
        }
      }
      constexpr double kE = 1e-4;  // about one level of 255 over a few pixels: below it a channel has no detail to match
      if (en < kE) continue;
      const double lr = std::log((eo + kE) / (en + kE));
      tl.hf_abs += std::abs(lr);
      tl.hf_log += lr;
      tl.hn += 1;
      if (hf_w <= 0.f || !grad) continue;
      // d/dyh(q) of hf_w * pixels * lr^2: hf_w * pixels * 2 lr / (eo + kE) * 2 (h(q) - sum of h over q's neighbourhood / 9)
      const double k = hf_w * (tl.n > 0 ? 1.0 : 0.0) * static_cast<double>(N * N) * 2.0 * lr / (eo + kE) * 2.0;
      for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
          if (!live[sz(y) * N + sz(x)]) continue;
          float g = (y >= 1 && x >= 1 && y + 1 < N && x + 1 < N) ? h[sz(y) * N + sz(x)] : 0.f, m = 0.f;
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
              const int xx = x + dx, yy = y + dy;
              if (xx >= 1 && yy >= 1 && xx + 1 < N && yy + 1 < N) m += h[sz(yy) * N + sz(xx)];
            }
          }
          g -= m / 9.f;
          w.d_o[(sz(y / 2) * kM + sz(x / 2)) * upscale::kOut + sz(((y % 2) * 2 + x % 2) * 4 + ch)] += static_cast<float>(k * g);
        }
      }
    }
  }
  // the amplitudes: RMS of the high-passed error over each interior 2 x 2 block
  if (amp) {
    for (int yi = 1; yi + 1 < kM; ++yi) {
      for (int xi = 1; xi + 1 < kM; ++xi) {
        for (int ch = 0; ch < 4; ++ch) {
          double e2 = 0;
          int live_n = 0;
          for (int s = 0; s < 4; ++s) {
            const int x = 2 * xi + s % 2, y = 2 * yi + s / 2;
            live_n += live[sz(y) * N + sz(x)];
            float m = 0.f;
            for (int dy = -1; dy <= 1; ++dy) {
              for (int dx = -1; dx <= 1; ++dx) m += err[(sz(y + dy) * N + sz(x + dx)) * 4 + sz(ch)];
            }
            const float hp = err[(sz(y) * N + sz(x)) * 4 + sz(ch)] - m / 9.f;
            e2 += static_cast<double>(hp) * hp;
          }
          if (live_n == 0) continue;
          const float target = static_cast<float>(std::sqrt(e2 / 4.0));
          const float a = w.o[(sz(yi) * kM + sz(xi)) * upscale::kOut + upscale::kResid + sz(ch)];
          const float pre = a;  // the ReLU: the forward keeps the pre-activation here; its gradient is zero where it is negative
          const float av = std::max(0.f, pre);
          tl.amp += static_cast<double>(av - target) * (av - target);
          tl.an += 1;
          if (pre > 0.f) w.d_o[(sz(yi) * kM + sz(xi)) * upscale::kOut + upscale::kResid + sz(ch)] += 2.f * amp_w * (av - target);
        }
      }
    }
  }
  if (!grad) return tl;
  // backward: the output layer, layer 2, layer 1
  std::vector<float>& G = *grad;
  w.d_h2.assign(w.h2.size(), 0.f);
  for (int p = 0; p < kM * kM; ++p) {
    const float* d = w.d_o.data() + sz(p) * upscale::kOut;
    const float* h = w.h2.data() + sz(p) * sz(c2);
    float* dh = w.d_h2.data() + sz(p) * sz(c2);
    for (int k = 0; k < upscale::kOut; ++k) {
      if (d[k] == 0.f) continue;
      G[L.bo + sz(k)] += d[k];
      for (int i = 0; i < c2; ++i) {
        G[L.wo + sz(i) * upscale::kOut + sz(k)] += d[k] * h[i];
        dh[i] += d[k] * W[L.wo + sz(i) * upscale::kOut + sz(k)];
      }
    }
    for (int i = 0; i < c2; ++i) {
      if (w.h2p[sz(p) * sz(c2) + sz(i)] <= 0.f) dh[i] = 0.f;
    }
  }
  w.d_h1.assign(w.h1.size(), 0.f);
  for (int y = 0; y < kM; ++y) {
    for (int x = 0; x < kM; ++x) {
      const float* dh = w.d_h2.data() + (sz(y) * kM + sz(x)) * sz(c2);
      for (int j = 0; j < c2; ++j) G[L.b2 + sz(j)] += dh[j];
      for (int tp = 0; tp < k2 * k2; ++tp) {
        const int ax = x + tp % k2, ay = y + tp / k2;
        const float* hh = w.h1.data() + (sz(ay) * sz(A) + sz(ax)) * sz(c1);
        float* d1 = w.d_h1.data() + (sz(ay) * sz(A) + sz(ax)) * sz(c1);
        for (int i = 0; i < c1; ++i) {
          const float* wt = W + L.w2 + (sz(tp) * sz(c1) + sz(i)) * sz(c2);
          float* gt = G.data() + L.w2 + (sz(tp) * sz(c1) + sz(i)) * sz(c2);
          float s = 0.f;
          for (int j = 0; j < c2; ++j) {
            gt[j] += dh[j] * hh[i];
            s += dh[j] * wt[j];
          }
          d1[i] += s;
        }
      }
    }
  }
  for (int ay = 0; ay < A; ++ay) {
    for (int ax = 0; ax < A; ++ax) {
      const int tx = ax - r2, ty = ay - r2;
      if (!inframe(tx, ty)) continue;
      const float* d1 = w.d_h1.data() + (sz(ay) * sz(A) + sz(ax)) * sz(c1);
      const float* hp = w.h1p.data() + (sz(ay) * sz(A) + sz(ax)) * sz(c1);
      for (int j = 0; j < c1; ++j) {
        if (hp[j] <= 0.f || d1[j] == 0.f) continue;
        G[L.b1 + sz(j)] += d1[j];
        for (int tp = 0; tp < 9; ++tp) {
          const int dx = tp % 3 - 1, dy = tp / 3 - 1;
          for (int i = 0; i < 4; ++i) G[L.w1 + (sz(tp) * 4 + sz(i)) * sz(c1) + sz(j)] += d1[j] * X(tx + dx, ty + dy, i);
        }
      }
    }
  }
  return tl;
}

int step_train(const tools::Args& a, const Ctx& c) {
  const auto effects = effects_of(a.str("effects", "all"));
  const auto sizes = ints(a.str("sizes", "256,512"));
  const std::string name = a.need("out");
  const int c1 = a.i("c1", 8), c2 = a.i("c2", 8), k2 = a.i("k2", 1), steps = a.i("steps", 4000), batch = a.i("batch", 32);
  const int amp_from = a.i("amp-from", steps / 2);
  const float lr0 = a.f("lr", 3e-3f), amp_w = a.f("amp-weight", 1.f), hf_w = a.f("hf-weight", 0.f);
  const std::uint64_t seed = a.u64("seed", 1);
  // the tiles, grouped by effect so that each effect is drawn equally often
  std::vector<std::vector<Tile>> sets;
  std::size_t total = 0;
  for (const sim::Effect e : effects) {
    std::vector<Tile> v;
    for (const int S : sizes) std::ranges::move(load_tiles(tile_file(c, e, S)), std::back_inserter(v));
    total += v.size();
    sets.push_back(std::move(v));
  }
  // hold out every 16th tile of each effect for a loss to watch (training data still; not a validation)
  std::vector<std::vector<const Tile*>> train(sets.size()), held(sets.size());
  for (std::size_t s = 0; s < sets.size(); ++s) {
    for (std::size_t i = 0; i < sets[s].size(); ++i) (i % 16 == 15 ? held : train)[s].push_back(&sets[s][i]);
  }
  upscale::Net net = upscale::init_net(c1, c2, k2, seed);
  net.tag = effects.size() == kStudy.size() ? "shared" : ename(effects[0]);
  net.gain = 1.f;
  net.rate = 8.f;
  const std::size_t P = net.w.size();
  std::vector<double> m1(P, 0.0), m2(P, 0.0);
  std::mt19937_64 rng(seed * 7919);
  const int T = std::max(1, c.threads);
  std::vector<std::vector<float>> grads(sz(T), std::vector<float>(P));
  std::vector<Work> works(sz(T));
  std::println("train {}: {} tiles from {} effect(s) at {}, c1 {} c2 {} k2 {} ({} weights, {:.0f} multiply-adds per full-size pixel), {} steps of {}",
               name, total, effects.size(), a.str("sizes", "256,512"), c1, c2, k2, P, net.macs_per_output_pixel(), steps, batch);
  const auto evaluate = [&](bool amp) {
    TileLoss sum;
    for (const auto& h : held) {
      for (const Tile* t : h) {
        const TileLoss l = tile_pass(net, *t, amp, amp_w, nullptr, works[0]);
        sum.sse += l.sse;
        sum.n += l.n;
        sum.amp += l.amp;
        sum.an += l.an;
        sum.hf_abs += l.hf_abs;
        sum.hf_log += l.hf_log;
        sum.hn += l.hn;
      }
    }
    return sum;
  };
  const auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < steps; ++it) {
    const bool amp = it >= amp_from;
    std::vector<const Tile*> pick(sz(batch));
    for (int b = 0; b < batch; ++b) {
      const auto& set = train[sz(b) % train.size()];
      pick[sz(b)] = set[rng() % set.size()];
    }
    for (auto& g : grads) std::ranges::fill(g, 0.f);
    std::vector<TileLoss> losses(sz(batch));
    parallel(T, T, [&](int th) {
      for (int b = th; b < batch; b += T) losses[sz(b)] = tile_pass(net, *pick[sz(b)], amp, amp_w, &grads[sz(th)], works[sz(th)], hf_w);
    });
    double n = 0;
    for (const auto& l : losses) n += l.n;
    const double scale = 1.0 / std::max(1.0, n);
    // Adam with a cosine schedule (and the amplitudes' rows only start learning at amp_from)
    const double lr = lr0 * (0.02 + 0.98 * 0.5 * (1.0 + std::cos(3.14159265358979 * it / steps)));
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const double c1t = 1.0 - std::pow(b1, it + 1), c2t = 1.0 - std::pow(b2, it + 1);
    for (std::size_t i = 0; i < P; ++i) {
      double g = 0;
      for (const auto& gr : grads) g += gr[i];
      g *= scale;
      m1[i] = b1 * m1[i] + (1 - b1) * g;
      m2[i] = b2 * m2[i] + (1 - b2) * g * g;
      net.w[i] -= static_cast<float>(lr * (m1[i] / c1t) / (std::sqrt(m2[i] / c2t) + eps));
    }
    if (it % 250 == 0 || it + 1 == steps) {
      const TileLoss h = evaluate(amp);
      std::println("  step {:>5}: held-out tiles {:.2f} dB (mse {:.3e}), high band |ln E/E_N| {:.3f} (ln {:+.3f}){} ({:.0f} s)", it,
                   metrics::psnr_from_mse(h.sse / std::max(1.0, h.n)), h.sse / std::max(1.0, h.n), h.hf_abs / std::max(1.0, h.hn), h.hf_log / std::max(1.0, h.hn),
                   amp ? std::format(", amplitude rms error {:.4f}", std::sqrt(h.amp / std::max(1.0, h.an))) : "", seconds_since(t0));
      std::fflush(stdout);
    }
  }
  // bilinear alone on the held-out tiles, for comparison
  {
    upscale::Net zero = net;
    std::ranges::fill(zero.w, 0.f);
    TileLoss s;
    for (const auto& h : held) {
      for (const Tile* t : h) {
        const TileLoss l = tile_pass(zero, *t, false, 0.f, nullptr, works[0]);
        s.sse += l.sse;
        s.n += l.n;
        s.hf_abs += l.hf_abs;
        s.hf_log += l.hf_log;
        s.hn += l.hn;
      }
    }
    std::println("  bilinear on the held-out tiles: {:.2f} dB, high band |ln E/E_N| {:.3f} (ln {:+.3f})", metrics::psnr_from_mse(s.sse / std::max(1.0, s.n)),
                 s.hf_abs / std::max(1.0, s.hn), s.hf_log / std::max(1.0, s.hn));
  }
  fs::create_directories(c.dir() / "nets");
  if (auto r = upscale::save_net(c.net(name), net); !r) throw std::runtime_error(r.error());
  std::println("wrote {}", c.net(name).string());
  return 0;
}

// Development: nets on the held-out tiles (every 16th) of some effects and sizes: PSNR against N and the high band's energy.
int step_tile_eval(const tools::Args& a, const Ctx& c) {
  const auto sizes = ints(a.str("sizes", "256,512"));
  for (const sim::Effect e : effects_of(a.str("effects", "all"))) {
    for (const int S : sizes) {
      const auto tiles = load_tiles(tile_file(c, e, S));
      std::vector<std::pair<std::string, upscale::Net>> nets;
      upscale::Net zero = upscale::init_net(8, 8, 1, 1);
      std::ranges::fill(zero.w, 0.f);
      nets.emplace_back("bilinear", zero);
      for (const auto& n : words(a.str("nets", ""))) nets.emplace_back(n, load_net(c.net(n)));
      Work w;
      std::string line = std::format("{:>9} {}:", ename(e), S);
      for (const auto& [name, net] : nets) {
        TileLoss s;
        for (std::size_t i = 15; i < tiles.size(); i += 16) {
          const TileLoss l = tile_pass(net, tiles[i], false, 0.f, nullptr, w);
          s.sse += l.sse;
          s.n += l.n;
          s.hf_log += l.hf_log;
          s.hn += l.hn;
        }
        line += std::format("  {} {:.2f} dB (hf ln {:+.2f})", name, metrics::psnr_from_mse(s.sse / std::max(1.0, s.n)), s.hf_log / std::max(1.0, s.hn));
      }
      std::println("{}", line);
    }
  }
  return 0;
}

// --- evaluation -------------------------------------------------------------------------------------------------------------

// A real run at S: the teacher (128 cells) warmed up, then frames_of(e) frames drawn at S; its statistics.
metrics::ClipStats real_stats(sim::Effect e, const Setting& s, std::uint64_t seed, int S) {
  sim::Fluid f(setting_params(e, s, seed, S));
  for (int i = 0; i < warm_of(e); ++i) f.step_frame();
  StatAcc acc(S);
  std::vector<std::uint8_t> fr(sz(S) * sz(S) * 4);
  for (int i = 0; i < frames_of(e); ++i) {
    f.step_frame();
    f.render(fr);
    acc.add(fr);
  }
  return acc.finish();
}
// The real runs of a setting pooled, cached under DATA/p_upscale/cache (the mean frame in half precision, whether fresh or
// read back).
metrics::ClipStats real_pool(const Ctx& c, sim::Effect e, const Split& sp, int si, int reps, int S, int threads) {
  const fs::path p = c.dir() / "cache" / std::format("real_{}_{}_{}_{}_{}.st", sp.name, ename(e), S, si, reps);
  metrics::ClipStats st;
  if (get_stats(p, st)) return st;
  std::vector<metrics::ClipStats> v(sz(reps));
  parallel(threads, reps, [&](int k) { v[sz(k)] = real_stats(e, sp.settings[sz(si)], seed_of(sp.real, si, k), S); });
  st = pool(v);
  round_mean_frame(st);
  fs::create_directories(p.parent_path());
  {
    std::ofstream o(p.string() + ".tmp", std::ios::binary);
    put_stats(o, st);
  }
  fs::rename(p.string() + ".tmp", p);
  return st;
}

// Tracking from the true state (study D's protocol, as study M3's): a run from an empty grid, its state after the warm-up as
// the effect's only start point (its fine fields at the teacher's 128 cells, spread by the runtime to the instance's size),
// played with the run's seed as one continuous rollout; active PSNR at 1 s against the teacher drawn at S. Per arm: N, then
// each upscaling arm.
std::vector<double> track_run(const rollout::Model& M, sim::Effect e, const Setting& s, std::uint64_t seed, int S, const std::vector<Arm>& arms) {
  const sim::Params p = setting_params(e, s, seed, S);
  sim::Fluid truth(p);
  for (int i = 0; i < track_warm(e); ++i) truth.step_frame();
  const sim::State st = truth.state();
  std::vector<std::uint8_t> ref(sz(S) * sz(S) * 4);
  for (int i = 0; i < 30; ++i) truth.step_frame();
  truth.render(ref);
  rollout::StartPoint s0;
  s0.controls = {p.intensity, p.wind, p.turbulence};
  s0.seed = p.seed;
  s0.time = st.time;
  s0.coarse.resize(sz(kRes) * kRes * rollout::kPhys);
  rollout::coarse_from_sim(st, kRes, p.fps, s0.coarse);
  s0.fine_t = st.temp;
  s0.fine_d = st.soot;
  rollout::Model mt = M;
  mt.h.start_fine = kTeacher;
  mt.starts = {s0};
  const Loaded fx(mt);
  std::vector<double> out(1 + arms.size());
  play_pair(fx.e, S, s, 0, 31, arms, 0, [&](int f, std::span<const std::uint8_t> n, const std::vector<std::vector<std::uint8_t>>& o) {
    if (f != 30) return;
    out[0] = apsnr(ref, n);
    for (std::size_t a = 0; a < o.size(); ++a) out[a + 1] = apsnr(ref, o[a]);
  });
  return out;
}

// One endless play of every arm: the statistics of N and of each arm, and each arm's (a) against N.
struct PlayOut {
  metrics::ClipStats n;
  std::vector<metrics::ClipStats> arm;
  std::vector<PairAcc> pa;
};
PlayOut endless_play(nvfx_effect* fx, sim::Effect e, const Setting& s, std::uint64_t seed, int S, const std::vector<Arm>& arms) {
  StatAcc an(S);
  std::vector<StatAcc> aa;
  for (std::size_t i = 0; i < arms.size(); ++i) aa.emplace_back(S);
  PlayOut po;
  po.pa.resize(arms.size());
  play_pair(fx, S, s, seed, frames_of(e), arms, -1, [&](int f, std::span<const std::uint8_t> n, const std::vector<std::vector<std::uint8_t>>& o) {
    an.add(n);
    for (std::size_t a = 0; a < o.size(); ++a) {
      aa[a].add(o[a]);
      po.pa[a].add(n, o[a], S, f % 10 == 0);
    }
  });
  po.n = an.finish();
  for (auto& x : aa) po.arm.push_back(x.finish());
  return po;
}
metrics::ClipStats native_play(nvfx_effect* fx, sim::Effect e, const Setting& s, std::uint64_t seed, int S) {
  Instance N(fx, S, s, seed);
  StatAcc acc(S);
  std::vector<std::uint8_t> fr(sz(S) * sz(S) * 4);
  for (int i = 0; i < frames_of(e); ++i) {
    N.frame(i, fr);
    acc.add(fr);
  }
  return acc.finish();
}

// The (b) measures of a pooled play against the pooled real runs.
struct BMeasures {
  double spec, motion, cov, emi, mfp, hf;
};
BMeasures b_measures(const metrics::ClipStats& real, const metrics::ClipStats& play) {
  const metrics::StatDistance d = metrics::distance(real, play);
  return {d.spectrum_l1, std::abs(std::log(std::max(1e-3, d.motion_ratio))), d.coverage_l1, d.emission_l1, d.mean_frame_psnr, hf_l1(play.spectrum, real.spectrum)};
}

// CSV rows: effect,size,setting,arm,measure,value
struct Rows {
  std::vector<std::string> v;
  void add(sim::Effect e, int S, int si, const std::string& arm, const std::string& measure, double value) {
    v.push_back(std::format("{},{},{},{},{},{:.6f}", ename(e), S, si, arm, measure, value));
  }
};
const char* kHeader = "effect,size,setting,arm,measure,value";
void append_rows(const fs::path& path, const std::vector<std::string>& rows) {
  if (path.has_parent_path()) fs::create_directories(path.parent_path());
  const bool fresh = !fs::exists(path) || fs::file_size(path) == 0;
  std::ofstream o(path, std::ios::app);
  if (fresh) o << kHeader << "\n";
  for (const auto& r : rows) o << r << "\n";
}
struct Row {
  std::string effect, arm, measure;
  int size = 0, setting = 0;
  double value = 0;
};
std::vector<Row> read_rows(const fs::path& path) {
  std::vector<Row> v;
  std::ifstream i(path);
  std::string line;
  std::getline(i, line);
  while (std::getline(i, line)) {
    std::stringstream ss(line);
    Row r;
    std::string f;
    std::getline(ss, r.effect, ',');
    std::getline(ss, f, ',');
    r.size = std::stoi(f);
    std::getline(ss, f, ',');
    r.setting = std::stoi(f);
    std::getline(ss, r.arm, ',');
    std::getline(ss, r.measure, ',');
    std::getline(ss, f, ',');
    r.value = std::stod(f);
    v.push_back(r);
  }
  return v;
}

// Every measure of one setting: tracking (track runs), the endless plays (reps), the real pool; rows for N, the null, B and
// each L arm.
void eval_setting(const Ctx& c, sim::Effect e, const rollout::Model& M, nvfx_effect* fx, const Split& sp, int si, int S, int reps, int track_reps, bool null,
                  const std::vector<Arm>& arms, Rows& rows) {
  const Setting& s = sp.settings[sz(si)];
  const int T = c.threads;
  // tracking
  std::vector<std::vector<double>> tr(sz(track_reps));
  parallel(T, track_reps, [&](int k) { tr[sz(k)] = track_run(M, e, s, seed_of(sp.track, si, k), S, arms); });
  for (std::size_t a = 0; a <= arms.size(); ++a) {
    double m = 0;
    for (const auto& t : tr) m += t[a] / static_cast<double>(tr.size());
    rows.add(e, S, si, a == 0 ? "N" : arms[a - 1].name, "track1s", m);
  }
  // the endless plays and the null
  std::vector<PlayOut> po(sz(reps));
  std::vector<metrics::ClipStats> nul(null ? sz(reps) : 0);
  parallel(T, reps * (null ? 2 : 1), [&](int j) {
    if (j < reps) po[sz(j)] = endless_play(fx, e, s, seed_of(sp.play, si, j), S, arms);
    else nul[sz(j - reps)] = native_play(fx, e, s, seed_of(sp.null, si, j - reps), S);
  });
  const metrics::ClipStats real = real_pool(c, e, sp, si, reps, S, T);
  std::vector<metrics::ClipStats> vn;
  for (const auto& p : po) vn.push_back(p.n);
  metrics::ClipStats pn = pool(vn);
  round_mean_frame(pn);
  const auto put_b = [&](const std::string& arm, const metrics::ClipStats& p) {
    const BMeasures b = b_measures(real, p);
    rows.add(e, S, si, arm, "spectrum_l1", b.spec);
    rows.add(e, S, si, arm, "abs_ln_motion_ratio", b.motion);
    rows.add(e, S, si, arm, "coverage_l1", b.cov);
    rows.add(e, S, si, arm, "emission_l1", b.emi);
    rows.add(e, S, si, arm, "mean_frame_psnr", b.mfp);
    rows.add(e, S, si, arm, "hf_l1_real", b.hf);
    rows.add(e, S, si, arm, "hf_log_ratio_real", hf_log_ratio(p.spectrum, real.spectrum));
  };
  put_b("N", pn);
  if (null) {
    metrics::ClipStats pz = pool(nul);
    round_mean_frame(pz);
    put_b("Nnull", pz);
    rows.add(e, S, si, "Nnull", "hf_l1_vsN", hf_l1(pz.spectrum, pn.spectrum));
  }
  for (std::size_t a = 0; a < arms.size(); ++a) {
    std::vector<metrics::ClipStats> va;
    for (const auto& p : po) va.push_back(p.arm[a]);
    metrics::ClipStats pa = pool(va);
    round_mean_frame(pa);
    put_b(arms[a].name, pa);
    double ps = 0, ss = 0;
    for (const auto& p : po) {
      ps += p.pa[a].psnr() / static_cast<double>(po.size());
      ss += p.pa[a].mean_ssim() / static_cast<double>(po.size());
    }
    rows.add(e, S, si, arms[a].name, "apsnr_vsN", ps);
    rows.add(e, S, si, arms[a].name, "ssim_vsN", ss);
    rows.add(e, S, si, arms[a].name, "hf_l1_vsN", hf_l1(pa.spectrum, pn.spectrum));
    rows.add(e, S, si, arms[a].name, "hf_log_ratio_vsN", hf_log_ratio(pa.spectrum, pn.spectrum));
    rows.add(e, S, si, arms[a].name, "abs_ln_motion_vsN", std::abs(std::log(std::max(1e-6, pa.motion) / std::max(1e-6, pn.motion))));
  }
}

int step_eval(const tools::Args& a, const Ctx& c) {
  const std::string split = a.need("split");
  const auto sizes = ints(a.str("sizes", "256,512"));
  const int nset = a.i("settings", split == "dev" ? 4 : 10), reps = a.i("reps", split == "dev" ? 2 : 4), track_reps = a.i("track-reps", 2);
  const bool null = a.flag("null");
  if (null && split != "val") throw std::invalid_argument("--null: the null probe runs on validation");
  const std::string cands = a.str("cands", "");
  const fs::path out = a.has("out") ? fs::path(a.str("out")) : c.results / std::format("p_eval_{}.csv", split);
  if (split == "test") {  // once: a marker under DATA/p_upscale
    const fs::path mark = c.dir() / std::format("test_ran_{}", a.str("tag", "rule"));
    if (fs::exists(mark) && !a.flag("resume")) throw std::runtime_error("the test has run already (" + mark.string() + "); --resume only continues an interrupted run");
    std::ofstream(mark) << "started\n";
  }
  // settings already done (every arm asked for present)
  std::set<std::string> done;
  if (fs::exists(out)) {
    std::map<std::string, std::set<std::string>> arms_at;
    for (const Row& r : read_rows(out)) arms_at[std::format("{},{},{}", r.effect, r.size, r.setting)].insert(r.arm);
    for (const auto& [k, v] : arms_at) {
      bool all = v.count("N") && v.count("B") && (!null || v.count("Nnull"));
      for (const auto& w : words(cands)) all &= v.count("L_" + w.substr(0, w.find('='))) > 0;
      if (all) done.insert(k);
    }
  }
  const auto t0 = std::chrono::steady_clock::now();
  for (const sim::Effect e : effects_of(a.str("effects", "all"))) {
    const Split sp = split_of(split, e, nset);
    auto m = rollout::load_model(c.model(e));
    if (!m) throw std::runtime_error(m.error());
    const rollout::Model M = std::move(*m);
    const Loaded fx(c.model(e));
    const auto arms = arms_for(c, cands, e);
    for (const int S : sizes) {
      for (int si = 0; si < static_cast<int>(sp.settings.size()); ++si) {
        if (done.count(std::format("{},{},{}", ename(e), S, si))) continue;
        Rows rows;
        eval_setting(c, e, M, fx.e, sp, si, S, reps, track_reps, null, arms, rows);
        append_rows(out, rows.v);
        std::println("{} {} {} setting {}: done ({:.0f} s)", split, ename(e), S, si, seconds_since(t0));
        std::fflush(stdout);
      }
    }
  }
  if (split == "test") std::ofstream(c.dir() / std::format("test_ran_{}", a.str("tag", "rule"))) << "finished\n";
  return 0;
}

// --- dev and calibration (training settings only) -------------------------------------------------------------------------

// (a), the high band against N and motion against N on the dev split, for each net at each gain and rate.
struct DevScore {
  double apsnr = 0, ssim = 0, hf = 0, hf_ratio = 0, motion = 0;
  int n = 0;
};
std::map<std::string, DevScore> dev_scores(const Ctx& c, const std::vector<sim::Effect>& effects, const std::vector<int>& sizes, int nset, int reps,
                                           const std::vector<Arm>& arms, std::map<std::string, std::map<std::string, DevScore>>* per = nullptr) {
  std::map<std::string, DevScore> out;
  std::mutex mu;
  for (const sim::Effect e : effects) {
    const Loaded fx(c.model(e));
    const Split sp = split_of("dev", e, nset);
    for (const int S : sizes) {
      std::vector<std::pair<int, int>> jobs;
      for (int si = 0; si < nset; ++si) {
        for (int k = 0; k < reps; ++k) jobs.emplace_back(si, k);
      }
      std::vector<PlayOut> po(jobs.size());
      parallel(c.threads, static_cast<int>(jobs.size()), [&](int j) {
        po[sz(j)] = endless_play(fx.e, e, sp.settings[sz(jobs[sz(j)].first)], seed_of(sp.play, jobs[sz(j)].first, jobs[sz(j)].second), S, arms);
      });
      for (std::size_t a = 0; a < arms.size(); ++a) {
        DevScore d;
        for (const auto& p : po) {
          d.apsnr += p.pa[a].psnr();
          d.ssim += p.pa[a].mean_ssim();
          d.hf += hf_l1(p.arm[a].spectrum, p.n.spectrum);
          d.hf_ratio += hf_log_ratio(p.arm[a].spectrum, p.n.spectrum);
          d.motion += std::log(std::max(1e-6, p.arm[a].motion) / std::max(1e-6, p.n.motion));
          d.n += 1;
        }
        std::lock_guard lk(mu);
        auto& o = out[arms[a].name];
        o.apsnr += d.apsnr;
        o.ssim += d.ssim;
        o.hf += d.hf;
        o.hf_ratio += d.hf_ratio;
        o.motion += d.motion;
        o.n += d.n;
        if (per) {
          d.apsnr /= d.n;
          d.ssim /= d.n;
          d.hf /= d.n;
          d.hf_ratio /= d.n;
          d.motion /= d.n;
          (*per)[std::format("{} {}", ename(e), S)][arms[a].name] = d;
        }
      }
    }
  }
  for (auto& [k, d] : out) {
    d.apsnr /= d.n;
    d.ssim /= d.n;
    d.hf /= d.n;
    d.hf_ratio /= d.n;
    d.motion /= d.n;
  }
  return out;
}

// The arms of --nets at each --gains and --rates (gain 0: the residual alone).
std::vector<Arm> dev_arms(const Ctx& c, const std::string& nets, const std::string& gains, const std::string& rates) {
  std::vector<Arm> arms = arms_list();
  const auto gs = tools::parse_floats(gains.empty() ? "-1" : gains), rs = tools::parse_floats(rates.empty() ? "-1" : rates);
  for (const auto& name : words(nets)) {
    const upscale::Net base = load_net(c.net(name));
    for (const float g : gs) {
      for (const float r : rs) {
        upscale::Net n = base;
        if (g >= 0.f) n.gain = g;
        if (r > 0.f) n.rate = r;
        arms.push_back({std::format("{} g{:.2f} r{:.0f}", name, n.gain, n.rate), std::make_shared<upscale::Net>(n)});
      }
    }
  }
  return arms;
}

int step_dev(const tools::Args& a, const Ctx& c) {
  const auto arms = dev_arms(c, a.need("nets"), a.str("gains"), a.str("rates"));
  std::map<std::string, std::map<std::string, DevScore>> per;
  const auto t0 = std::chrono::steady_clock::now();
  const auto sc = dev_scores(c, effects_of(a.str("effects", "all")), ints(a.str("sizes", "256,512")), a.i("settings", 4), a.i("reps", 1), arms, &per);
  if (a.flag("per")) {
    for (const auto& [k, m] : per) {
      for (const auto& [arm, d] : m) {
        std::println("{:>16} {:<28} apsnr {:6.2f} ssim {:.4f} hf_l1 {:.3f} hf_log_ratio {:+.3f} ln_motion {:+.4f}", k, arm, d.apsnr, d.ssim, d.hf, d.hf_ratio, d.motion);
      }
    }
  }
  std::println("dev (training settings 16 on, other seeds; {:.0f} s): against N on the same seeds", seconds_since(t0));
  for (const Arm& arm : arms) {
    const DevScore& d = sc.at(arm.name);
    std::println("{:<30} apsnr {:6.2f} ssim {:.4f} hf_l1 {:.3f} hf_log_ratio {:+.3f} ln_motion {:+.4f}", arm.name, d.apsnr, d.ssim, d.hf, d.hf_ratio, d.motion);
  }
  return 0;
}

// The micro-detail's gain and rate (docs/STUDY_P.md §3), on the dev split (training settings 16 on, other seeds): among the
// gains, the largest whose PSNR and SSIM against N each keep at least half of the residual's own gain over B (the net at gain
// 0; means over the effects and sizes; a measure on which the residual gains nothing allows no micro-detail); then among the
// rates, the one whose motion is closest to N's (|mean ln motion ratio|).
int step_calib(const tools::Args& a, const Ctx& c) {
  const std::string name = a.need("net");
  const auto effects = effects_of(a.str("effects", "all"));
  const auto sizes = ints(a.str("sizes", "256,512"));
  const int nset = a.i("settings", 2), reps = a.i("reps", 1);
  const std::string gains = a.str("gains", "0,0.1,0.2,0.3,0.4,0.5,0.75,1"), rates = a.str("rates", "4,8,15,30");
  const float rate0 = a.f("rate", 8.f);
  std::vector<Arm> arms = dev_arms(c, name, gains, std::format("{}", rate0));
  const auto sg = dev_scores(c, effects, sizes, nset, reps, arms);
  const DevScore& b = sg.at("B");
  const DevScore* g0 = nullptr;
  for (const Arm& arm : arms) {
    if (arm.net && arm.net->gain == 0.f) g0 = &sg.at(arm.name);
  }
  if (!g0) throw std::invalid_argument("--gains must include 0");
  std::println("{} on {} effect(s), dev split ({} settings x {} plays): B apsnr {:.3f} ssim {:.4f} hf_l1 {:.3f} ln_motion {:+.4f}", name, effects.size(), nset, reps, b.apsnr,
               b.ssim, b.hf, b.motion);
  float gain = 0.f;
  for (const Arm& arm : arms) {
    if (!arm.net) continue;
    const DevScore& d = sg.at(arm.name);
    const bool keeps = d.apsnr - b.apsnr >= 0.5 * (g0->apsnr - b.apsnr) && d.ssim - b.ssim >= 0.5 * (g0->ssim - b.ssim) && g0->apsnr > b.apsnr && g0->ssim > b.ssim;
    std::println("{:<30} apsnr {:+.3f} ssim {:+.5f} (against B) hf_l1 {:.3f} hf_log_ratio {:+.3f} ln_motion {:+.4f}{}", arm.name, d.apsnr - b.apsnr, d.ssim - b.ssim, d.hf,
                 d.hf_ratio, d.motion, keeps ? "" : "  (keeps less than half of the residual's gain)");
    if (keeps) gain = std::max(gain, arm.net->gain);
  }
  float rate = rate0;
  if (gain > 0.f) {
    arms = dev_arms(c, name, std::format("{}", gain), rates);
    const auto sr = dev_scores(c, effects, sizes, nset, reps, arms);
    double best = 1e30;
    for (const Arm& arm : arms) {
      if (!arm.net) continue;
      const DevScore& d = sr.at(arm.name);
      std::println("{:<30} apsnr {:+.3f} ssim {:+.5f} hf_l1 {:.3f} ln_motion {:+.4f}", arm.name, d.apsnr - b.apsnr, d.ssim - b.ssim, d.hf, d.motion);
      if (std::abs(d.motion) < best) {
        best = std::abs(d.motion);
        rate = arm.net->rate;
      }
    }
  }
  upscale::Net n = load_net(c.net(name));
  n.gain = gain;
  n.rate = rate;
  const std::string out = a.str("out", name);
  if (auto r = upscale::save_net(c.net(out), n); !r) throw std::runtime_error(r.error());
  std::println("calibrated {}: gain {:.2f}, rate {:.0f} per second -> {}", name, gain, rate, c.net(out).string());
  return 0;
}

// Development: where N and the S / 2 instance differ. For one play per effect: active PSNR of B (and L, with --net) against N,
// of N box-averaged to S / 2 and brought back by the bilinear 2x (an ideal S / 2 frame) against N, and of the S / 2 instance
// against N box-averaged to S / 2; raw RGBA frames (N, B, L) at two times when --png DIR is given.
int step_anatomy(const tools::Args& a, const Ctx& c) {
  const auto sizes = ints(a.str("sizes", "256,512"));
  const std::string nets = a.str("net", "");
  for (const sim::Effect e : effects_of(a.str("effects", "all"))) {
    const Loaded fx(c.model(e));
    const Split sp = split_of("dev", e, 1);
    const std::vector<Arm> arms = arms_list(nets.empty() ? nullptr : std::make_shared<upscale::Net>(load_net(c.net(nets))));
    for (const int S : sizes) {
      const int n = S / 2;
      PairAcc b, ideal, half, l;
      std::vector<std::uint8_t> dn(sz(n) * sz(n) * 4), hn(sz(n) * sz(n) * 4), un(sz(S) * sz(S) * 4);
      std::vector<float> uf(sz(S) * sz(S) * 4);
      Instance H(fx.e, n, sp.settings[0], seed_of(sp.play, 0, 0));
      play_pair(fx.e, S, sp.settings[0], seed_of(sp.play, 0, 0), frames_of(e), arms, -1, [&](int f, std::span<const std::uint8_t> nf, const std::vector<std::vector<std::uint8_t>>& o) {
        H.frame(f, hn);
        for (int y = 0; y < n; ++y) {
          for (int x = 0; x < n; ++x) {
            for (int ch = 0; ch < 4; ++ch) {
              int q = 0;
              for (int j = 0; j < 2; ++j) {
                for (int i = 0; i < 2; ++i) q += nf[(sz(2 * y + j) * sz(S) + sz(2 * x + i)) * 4 + sz(ch)];
              }
              dn[(sz(y) * sz(n) + sz(x)) * 4 + sz(ch)] = static_cast<std::uint8_t>((q + 2) / 4);
            }
          }
        }
        upscale::bilinear2x(dn, n, uf);
        for (std::size_t i = 0; i < uf.size(); ++i) un[i] = static_cast<std::uint8_t>(static_cast<int>(std::clamp(uf[i], 0.f, 1.f) * 255.f + 0.5f));
        b.add(nf, o[0], S, false);
        if (o.size() > 1) l.add(nf, o[1], S, false);
        ideal.add(nf, un, S, false);
        half.add(dn, hn, n, false);
        if (a.has("png") && (f == 45 || f == frames_of(e) / 2)) {
          const fs::path dir = a.str("png");
          fs::create_directories(dir);
          std::ofstream(dir / std::format("{}_{}_f{}_N.rgba", ename(e), S, f), std::ios::binary).write(reinterpret_cast<const char*>(nf.data()), static_cast<std::streamsize>(nf.size()));
          for (std::size_t k = 0; k < o.size(); ++k) {
            std::ofstream(dir / std::format("{}_{}_f{}_{}.rgba", ename(e), S, f, arms[k].name), std::ios::binary)
                .write(reinterpret_cast<const char*>(o[k].data()), static_cast<std::streamsize>(o[k].size()));
          }
        }
      });
      std::println("{:>9} {}: B vs N {:.2f} dB{}, ideal half (N averaged, bilinear) vs N {:.2f} dB, the S/2 instance vs N averaged {:.2f} dB", ename(e), S, b.psnr(),
                   o_str(l), ideal.psnr(), half.psnr());
    }
  }
  return 0;
}

// --- margins and the rule -----------------------------------------------------------------------------------------------------

// The (b) measures: name, the direction that is better (+1 higher, -1 lower), study K's margin (a floor; "rel" ones are a
// fraction of N's own distance).
struct Measure {
  const char* name;
  int better;
  double k_margin;
  bool relative;
};
constexpr std::array kBMeasures = {Measure{"track1s", +1, 0.20, false},          Measure{"spectrum_l1", -1, 0.02, false},
                                   Measure{"abs_ln_motion_ratio", -1, 0.02, false}, Measure{"coverage_l1", -1, 0.10, true},
                                   Measure{"emission_l1", -1, 0.10, true},        Measure{"mean_frame_psnr", +1, 0.10, false},
                                   Measure{"hf_l1_real", -1, 0.02, false}};

using Table = std::map<std::string, std::map<int, double>>;  // key "effect,size,arm,measure" -> setting -> value
Table table_of(const std::vector<Row>& rows) {
  Table t;
  for (const Row& r : rows) t[std::format("{},{},{},{}", r.effect, r.size, r.arm, r.measure)][r.setting] = r.value;
  return t;
}
std::vector<double> column(const Table& t, const std::string& key, const std::vector<int>& settings) {
  std::vector<double> v;
  const auto it = t.find(key);
  if (it == t.end()) return v;
  for (const int s : settings) {
    const auto j = it->second.find(s);
    if (j == it->second.end()) return {};
    v.push_back(j->second);
  }
  return v;
}

int step_margins(const tools::Args& a, const Ctx& c) {
  const fs::path in = a.has("in") ? fs::path(a.str("in")) : c.results / "p_eval_val.csv";
  const Table t = table_of(read_rows(in));
  const auto sizes = ints(a.str("sizes", "256,512"));
  std::ofstream o(c.results / "p_margins.csv");
  o << "effect,size,measure,k_margin,null_rms,null_mean,n_value,margin,settings\n";
  std::println("the null (N with fresh seeds minus N, pooled plays, over the validation settings) and the margins:");
  for (const sim::Effect e : kStudy) {
    for (const int S : sizes) {
      for (const Measure& m : kBMeasures) {
        std::vector<int> set;
        const auto it = t.find(std::format("{},{},N,{}", ename(e), S, m.name));
        if (it == t.end()) continue;
        for (const auto& [s, v] : it->second) set.push_back(s);
        const auto n = column(t, std::format("{},{},N,{}", ename(e), S, m.name), set);
        double k = m.k_margin, rms = 0, mu = 0, nv = mean(n);
        if (m.relative) k = std::max(0.0001, m.k_margin * nv);
        if (std::string(m.name) != "track1s") {
          const auto z = column(t, std::format("{},{},Nnull,{}", ename(e), S, m.name), set);
          if (z.size() != n.size()) throw std::runtime_error("the null is missing for " + std::string(m.name));
          for (std::size_t i = 0; i < n.size(); ++i) {
            rms += (z[i] - n[i]) * (z[i] - n[i]) / static_cast<double>(n.size());
            mu += (z[i] - n[i]) / static_cast<double>(n.size());
          }
          rms = std::sqrt(rms);
        }
        const double margin = std::max(k, rms);
        o << std::format("{},{},{},{:.6f},{:.6f},{:+.6f},{:.6f},{:.6f},{}\n", ename(e), S, m.name, k, rms, mu, nv, margin, set.size());
        std::println("{:>9} {} {:<20} K {:.4f} null rms {:.4f} (mean {:+.4f}) N {:.4f} -> margin {:.4f}", ename(e), S, m.name, k, rms, mu, nv, margin);
      }
    }
  }
  return 0;
}

std::map<std::string, double> read_margins(const fs::path& p) {
  std::map<std::string, double> m;
  std::ifstream i(p);
  std::string line;
  std::getline(i, line);
  while (std::getline(i, line)) {
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string x;
    while (std::getline(ss, x, ',')) f.push_back(x);
    if (f.size() >= 8) m[std::format("{},{},{}", f[0], f[1], f[2])] = std::stod(f[7]);
  }
  return m;
}

std::string fmt_iv(const metrics::Interval& iv, int prec) {
  return std::format("{:+.{}f} [{:+.{}f}, {:+.{}f}]", iv.mean, prec, iv.lo, prec, iv.hi, prec);
}

// The rule (docs/STUDY_P.md §4) for each L arm: per effect and size, L - B on (a) and the high band against N (better: the
// interval clear of zero on the better side), L - N on each (b) measure (the interval's worse end within its margin).
int step_decide(const tools::Args& a, const Ctx& c) {
  const std::string split = a.need("split");
  const fs::path in = c.results / std::format("p_eval_{}.csv", split);
  const Table t = table_of(read_rows(in));
  const auto margins = read_margins(c.results / "p_margins.csv");
  const auto sizes = ints(a.str("sizes", "256,512"));
  std::set<std::string> larms;
  for (const auto& [k, v] : t) {
    const auto f = words(k);
    if (f.size() == 4 && f[2].starts_with("L_")) larms.insert(f[2]);
  }
  std::ofstream o(c.results / std::format("p_decide_{}.csv", split));
  o << "arm,effect,size,comparison,measure,mean,lo,hi,margin,verdict\n";
  for (const std::string& arm : larms) {
    for (const int S : sizes) {
      int a_pass = 0, b_pass = 0, both = 0;
      std::println("\n{} at {} px ({}):", arm, S, split);
      for (const sim::Effect e : kStudy) {
        std::vector<int> set;
        const auto it = t.find(std::format("{},{},{},apsnr_vsN", ename(e), S, arm));
        if (it == t.end()) continue;
        for (const auto& [s, v] : it->second) set.push_back(s);
        bool a_ok = true, b_ok = true;
        std::string line = std::format("  {:>9} (n={}):", ename(e), set.size());
        for (const auto& [measure, better] : std::vector<std::pair<std::string, int>>{{"apsnr_vsN", +1}, {"ssim_vsN", +1}, {"hf_l1_vsN", -1}}) {
          const auto l = column(t, std::format("{},{},{},{}", ename(e), S, arm, measure), set);
          const auto b = column(t, std::format("{},{},B,{}", ename(e), S, measure), set);
          const auto iv = metrics::paired_bootstrap(l, b, 10000, 2026);
          const bool ok = better > 0 ? iv.lo > 0 : iv.hi < 0;
          a_ok &= ok;
          const int prec = measure == "ssim_vsN" ? 4 : 3;
          line += std::format(" {} {}{}", measure, fmt_iv(iv, prec), ok ? "" : " (no)");
          o << std::format("{},{},{},L-B,{},{:.6f},{:.6f},{:.6f},,{}\n", arm, ename(e), S, measure, iv.mean, iv.lo, iv.hi, ok ? "better" : "not better");
        }
        line += " |";
        for (const Measure& m : kBMeasures) {
          const auto l = column(t, std::format("{},{},{},{}", ename(e), S, arm, m.name), set);
          const auto n = column(t, std::format("{},{},N,{}", ename(e), S, m.name), set);
          if (l.empty() || n.size() != l.size()) continue;
          const auto iv = metrics::paired_bootstrap(l, n, 10000, 2026);
          const auto mk = margins.find(std::format("{},{},{}", ename(e), S, m.name));
          if (mk == margins.end()) throw std::runtime_error("no margin for " + std::string(m.name));
          const double mg = mk->second;
          const bool ok = m.better > 0 ? iv.lo >= -mg : iv.hi <= mg;
          b_ok &= ok;
          line += std::format(" {} {}{}", m.name, fmt_iv(iv, 4), ok ? "" : std::format(" (past {:.4f})", mg));
          o << std::format("{},{},{},L-N,{},{:.6f},{:.6f},{:.6f},{:.6f},{}\n", arm, ename(e), S, m.name, iv.mean, iv.lo, iv.hi, mg, ok ? "within" : "past the margin");
        }
        a_pass += a_ok;
        b_pass += b_ok;
        both += a_ok && b_ok;
        std::println("{}\n      (a) and the high band beat B: {}; (b) within the margins of N: {}", line, a_ok ? "yes" : "NO", b_ok ? "yes" : "NO");
      }
      std::println("  {}: beats B on (a) and the high band for {} of 6, within N's margins on (b) for {} of 6", arm, a_pass, b_pass);
    }
  }
  // Beside the rule, not deciding: B against N on (b), with the same margins (is drawing at half size and letting the GPU's
  // bilinear sampler upscale as realistic as native?).
  for (const int S : sizes) {
    std::println("\nB against N on (b) at {} px ({}; beside the rule, not deciding):", S, split);
    int within = 0;
    for (const sim::Effect e : kStudy) {
      std::vector<int> set;
      const auto it = t.find(std::format("{},{},B,track1s", ename(e), S));
      if (it == t.end()) continue;
      for (const auto& [st, v] : it->second) set.push_back(st);
      bool ok = true;
      std::string line = std::format("  {:>9}:", ename(e));
      for (const Measure& m : kBMeasures) {
        const auto b = column(t, std::format("{},{},B,{}", ename(e), S, m.name), set);
        const auto n = column(t, std::format("{},{},N,{}", ename(e), S, m.name), set);
        if (b.empty() || b.size() != n.size()) continue;
        const auto iv = metrics::paired_bootstrap(b, n, 10000, 2026);
        const double mg = margins.at(std::format("{},{},{}", ename(e), S, m.name));
        const bool w = m.better > 0 ? iv.lo >= -mg : iv.hi <= mg;
        ok &= w;
        line += std::format(" {} {}{}", m.name, fmt_iv(iv, 4), w ? "" : " (past)");
        o << std::format("B,{},{},B-N,{},{:.6f},{:.6f},{:.6f},{:.6f},{}\n", ename(e), S, m.name, iv.mean, iv.lo, iv.hi, mg, w ? "within" : "past the margin");
      }
      within += ok;
      std::println("{}", line);
    }
    std::println("  B is within N's margins on (b) for {} of 6 at {} px", within, S);
  }
  return 0;
}

// Validation's choice (docs/STUDY_P.md §4.3) and the number of test settings (§4.4).
//   The choice: the per-effect upscalers (arm L_effect) replace the shared one (L_shared) only if, at 512 px, they are better
//   than it on (a)'s PSNR against N (L_effect - L_shared, interval above zero) for at least 4 of the 6 effects, and on none of
//   those worse (interval entirely on the worse side) on (a)'s SSIM or the high band against N. Otherwise the shared one.
//   The number of settings: for the chosen arm at 512 px, each effect and each measure of the rule, the paired spread over
//   the validation settings (its SD, taken at the upper end of its one-sided 80% interval) and its mean give the settings
//   at which the rule's interval would clear its bound with probability 0.9 if the test's mean were validation's:
//   ((1.96 + 1.28) SD / distance)^2, the distance being the mean's from zero (L - B) or from the worse end of the margin
//   (L - N). The test takes the smallest multiple of 10 that covers every finite one, at least 10, at most 20 (the time a
//   test takes here); measures whose validation mean is on the wrong side have no such number and are listed.
int step_choose(const tools::Args& a, const Ctx& c) {
  const Table t = table_of(read_rows(c.results / "p_eval_val.csv"));
  const auto margins = read_margins(c.results / "p_margins.csv");
  const int S = a.i("size", 512);
  const auto settings_of = [&](const std::string& key) {
    std::vector<int> v;
    const auto it = t.find(key);
    if (it != t.end()) {
      for (const auto& [s, x] : it->second) v.push_back(s);
    }
    return v;
  };
  int better = 0;
  std::ofstream o(c.results / "p_choice.csv");
  o << "effect,size,measure,mean,lo,hi\n";
  for (const int size : {256, 512}) {
    std::println("per-effect against shared at {} px (L_effect - L_shared, validation):", size);
    for (const sim::Effect e : kStudy) {
      const auto set = settings_of(std::format("{},{},L_shared,apsnr_vsN", ename(e), size));
      if (set.empty()) continue;
      std::string line = std::format("  {:>9}:", ename(e));
      bool psnr_better = false, worse = false;
      for (const auto& [m, sign] : std::vector<std::pair<std::string, int>>{{"apsnr_vsN", 1}, {"ssim_vsN", 1}, {"hf_l1_vsN", -1}}) {
        const auto x = column(t, std::format("{},{},L_effect,{}", ename(e), size, m), set);
        const auto y = column(t, std::format("{},{},L_shared,{}", ename(e), size, m), set);
        if (x.size() != y.size() || x.empty()) throw std::runtime_error("validation lacks L_effect or L_shared");
        const auto iv = metrics::paired_bootstrap(x, y, 10000, 2026);
        o << std::format("{},{},{},{:.6f},{:.6f},{:.6f}\n", ename(e), size, m, iv.mean, iv.lo, iv.hi);
        line += std::format(" {} {}", m, fmt_iv(iv, m == "ssim_vsN" ? 4 : 3));
        if (m == "apsnr_vsN") psnr_better = iv.lo > 0;
        else if (sign > 0 ? iv.hi < 0 : iv.lo > 0) worse = true;
      }
      if (size == S && psnr_better && !worse) ++better;
      std::println("{}", line);
    }
  }
  const std::string chosen = better >= 4 ? "L_effect" : "L_shared";
  std::println("at {} px the per-effect upscalers are better on PSNR (and not worse) for {} of 6 effects: the choice is {}", S, better, chosen);
  // the number of test settings
  constexpr double kZ = 1.96 + 1.2816;
  const double up80 = 1.293;  // sqrt(9 / chi2_{0.2, 9}): the SD's one-sided 80% bound at 10 settings
  std::ofstream pw(c.results / "p_power.csv");
  pw << "arm,effect,size,comparison,measure,mean,sd,sd_up80,distance,settings_for_0.9\n";
  int need = 10;
  std::vector<std::string> none;
  for (const sim::Effect e : kStudy) {
    const auto set = settings_of(std::format("{},{},{},apsnr_vsN", ename(e), S, chosen));
    const auto spread = [&](const std::vector<double>& x, const std::vector<double>& y) {
      double m = 0, v = 0;
      for (std::size_t i = 0; i < x.size(); ++i) m += (x[i] - y[i]) / static_cast<double>(x.size());
      for (std::size_t i = 0; i < x.size(); ++i) v += (x[i] - y[i] - m) * (x[i] - y[i] - m) / static_cast<double>(x.size() - 1);
      return std::pair{m, std::sqrt(v)};
    };
    const auto row = [&](const std::string& cmp, const std::string& m, double mean, double sd, double dist) {
      const double n = dist > 0 ? std::pow(kZ * sd * up80 / dist, 2) : std::numeric_limits<double>::infinity();
      pw << std::format("{},{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.1f}\n", chosen, ename(e), S, cmp, m, mean, sd, sd * up80, dist, n);
      if (std::isfinite(n)) need = std::max(need, static_cast<int>(std::ceil(n)));
      else none.push_back(std::format("{} {} {} (mean {:+.4f})", ename(e), cmp, m, mean));
    };
    for (const auto& [m, sign] : std::vector<std::pair<std::string, int>>{{"apsnr_vsN", 1}, {"ssim_vsN", 1}, {"hf_l1_vsN", -1}}) {
      const auto [mean, sd] = spread(column(t, std::format("{},{},{},{}", ename(e), S, chosen, m), set), column(t, std::format("{},{},B,{}", ename(e), S, m), set));
      row("L-B", m, mean, sd, sign * mean);
    }
    for (const Measure& m : kBMeasures) {
      const auto l = column(t, std::format("{},{},{},{}", ename(e), S, chosen, m.name), set);
      const auto n = column(t, std::format("{},{},N,{}", ename(e), S, m.name), set);
      if (l.empty() || l.size() != n.size()) continue;
      const auto [mean, sd] = spread(l, n);
      const double mg = margins.at(std::format("{},{},{}", ename(e), S, m.name));
      row("L-N", m.name, mean, sd, mg + m.better * mean);  // the margin minus the mean's step towards the worse side
    }
  }
  const int n_test = std::min(20, (need + 9) / 10 * 10);
  std::println("settings for the test: {} (the largest finite need {}; capped at 20)", n_test, need);
  for (const auto& s : none) std::println("  no number of settings gives power where validation's mean is on the wrong side: {}", s);
  std::ofstream(c.results / "p_choice.txt") << std::format("chosen {}\nsettings {}\nlargest_need {}\n", chosen, n_test, need);
  return 0;
}

// --- cost -----------------------------------------------------------------------------------------------------------------------

// Thread CPU time per frame through nvfx_render: N (an instance at S), H (the S / 2 instance alone: what B costs when the
// engine's GPU does the bilinear 2x), B (the instance with NVFX_UPSCALE_BILINEAR) and L (NVFX_UPSCALE_LEARNED), each
// playing its own timeline, the arms in turn within every round (their order rotated), `frames` frames an arm per round.
int step_cost(const tools::Args& a, const Ctx& c) {
  const auto sizes = ints(a.str("sizes", "256,512"));
  const int rounds = a.i("rounds", 30), frames = a.i("frames", 20);
  pin(a.i("core", -1));
  const std::string net_file = a.need("net");
  const fs::path out = a.has("out") ? fs::path(a.str("out")) : c.results / "p_cost.csv";
  const bool fresh = !fs::exists(out);
  std::ofstream o(out, std::ios::app);
  if (fresh) o << "label,effect,size,round,arm,ms_per_frame\n";
  const std::string label = a.str("label", "busy");
  const std::array<std::string, 4> names = {"N", "H", "B", "L"};
  for (const sim::Effect e : effects_of(a.str("effects", "all"))) {
    std::string file = net_file;
    for (std::size_t p; (p = file.find("{effect}")) != std::string::npos;) file.replace(p, 8, ename(e));
    const Loaded fx(c.model(e));
    if (nvfx_effect_attach_upscaler(fx.e, c.net(file).string().c_str()) != NVFX_OK) throw std::runtime_error("cannot attach " + c.net(file).string());
    for (const int S : sizes) {
      const Setting ctl{0.6f, 0.5f, 0.6f};
      std::array<std::unique_ptr<Instance>, 4> in;
      in[0] = std::make_unique<Instance>(fx.e, S, ctl, 3'900'200);
      in[1] = std::make_unique<Instance>(fx.e, S / 2, ctl, 3'900'200);
      in[2] = std::make_unique<Instance>(fx.e, S, ctl, 3'900'200);
      in[3] = std::make_unique<Instance>(fx.e, S, ctl, 3'900'200);
      if (nvfx_instance_set_upscale(in[2]->p, NVFX_UPSCALE_BILINEAR) != NVFX_OK || nvfx_instance_set_upscale(in[3]->p, NVFX_UPSCALE_LEARNED) != NVFX_OK) {
        throw std::runtime_error("set_upscale");
      }
      std::vector<std::uint8_t> buf(sz(S) * sz(S) * 4);
      std::array<int, 4> at{};
      const int F = frames_of(e);  // loop the timeline over 10 s, or a one-shot's run (its restart is part of its cost)
      for (int k = 0; k < 4; ++k) {  // warm up: a shard rolled ahead
        for (int f = 0; f < 20; ++f) in[sz(k)]->frame(at[sz(k)]++, buf);
      }
      std::array<std::vector<double>, 4> ms;
      for (int r = 0; r < rounds; ++r) {
        for (int j = 0; j < 4; ++j) {
          const int k = (j + r) % 4;
          Instance& I = *in[sz(k)];
          const double t0 = thread_ms();
          for (int f = 0; f < frames; ++f) {
            I.frame(at[sz(k)] % F, buf);
            ++at[sz(k)];
          }
          const double v = (thread_ms() - t0) / frames;
          ms[sz(k)].push_back(v);
          o << std::format("{},{},{},{},{},{:.5f}\n", label, ename(e), S, r, names[sz(k)], v);
        }
      }
      o.flush();
      const auto ratio = [&](int k) {
        std::vector<double> q;
        for (int r = 0; r < rounds; ++r) q.push_back(ms[sz(k)][sz(r)] / ms[0][sz(r)]);
        const auto iv = metrics::bootstrap_mean(q, 10000, 2026);
        return std::format("{:.3f} (median {:.3f}) [{:.3f}, {:.3f}]", iv.mean, median(q), iv.lo, iv.hi);
      };
      std::println("{:>9} {}: N {:.3f} ms, H/N {}, B/N {}, L/N {}", ename(e), S, median(ms[0]), ratio(1), ratio(2), ratio(3));
      std::fflush(stdout);
    }
  }
  return 0;
}

// Development: the runtime's upscaler alone on a frame of an S / 2 play, thread CPU time per call: bilinear, the net without
// micro-detail, the net with it (slices moving at its rate).
int step_bench(const tools::Args& a, const Ctx& c) {
  const int n = a.i("half", 256), reps = a.i("reps", 200);
  const nvfx_isa isa = rt::rollout_isa(rt::default_rollout_precision());
  const Loaded fx(c.model(sim::Effect::smoke));
  Instance H(fx.e, n, {0.6f, 0.5f, 0.6f}, 3'900'400);
  std::vector<std::uint8_t> h(sz(n) * sz(n) * 4), out(sz(2 * n) * sz(2 * n) * 4);
  H.frame(90, h);
  for (const auto& name : words(a.str("nets", ""))) {
    upscale::Net net = load_net(c.net(name));
    for (const float g : {-1.f, 0.f, 1.f}) {
      upscale::Net nn = net;
      if (g >= 0.f) nn.gain = g;
      auto u = runtime_upscaler(g < 0.f ? nullptr : &nn, n, isa);
      const rt::FrameInput fi;
      std::vector<double> t;
      for (int r = 0; r < reps; ++r) {
        const double t0 = thread_ms();
        u->run(h.data(), sz(n) * 4, 1, r / kFps, fi, out.data(), sz(2 * n) * 4);
        t.push_back(thread_ms() - t0);
      }
      std::println("{} {}: {:.3f} ms per call (median of {})", name, g < 0.f ? "bilinear" : std::format("gain {}", g), median(t), reps);
    }
  }
  return 0;
}

// --- video ----------------------------------------------------------------------------------------------------------------------

int step_video(const tools::Args& a, const Ctx& c) {
  sim::Effect e{};
  if (!sim::parse_effect(a.need("effect"), e)) throw std::invalid_argument("unknown effect");
  const int S = a.i("size", 512), crf = a.i("crf", 24);
  const double secs = a.f("seconds", sim::effect_loops(e) ? 10.f : static_cast<float>(frames_of(e)) / 30.f);
  const auto ctlv = tools::parse_floats(a.str("controls", "0.7,0.5,0.6"));
  const Setting ctl{ctlv.at(0), ctlv.at(1), ctlv.at(2)};
  const std::uint64_t seed = a.u64("seed", 3'900'300);
  const std::string out = a.need("out");
  const upscale::Net net = load_net(c.net(a.need("net")));
  const Loaded fx(c.model(e));
  const std::vector<Arm> arms = arms_list(std::make_shared<upscale::Net>(net));
  // a second row: a crop of each panel (--crop X,Y,SIDE in the panel's pixels) shown twice as large, nearest-neighbour
  const auto cropv = ints(a.str("crop", std::format("{},{},{}", S / 2 - S / 6, S / 2 - S / 6, S / 3)));
  const int cx = cropv.at(0), cy = cropv.at(1), cs = cropv.at(2), zoom = 2, zs = cs * zoom;
  if (cx < 0 || cy < 0 || cx + cs > S || cy + cs > S || 3 * zs + 4 * 12 > 3 * S + 4 * 12) throw std::invalid_argument("--crop out of the panel");
  const int gap = 12, title_h = 40, label_h = 30, W = 3 * S + 4 * gap, row2 = title_h + label_h + S + gap + label_h, Hh = row2 + zs + gap;
  std::vector<std::uint8_t> rgb(sz(W) * sz(Hh) * 3);
  const auto fill = [&](int x0, int y0, int w, int h, std::array<std::uint8_t, 3> col) {
    for (int y = std::max(0, y0); y < std::min(Hh, y0 + h); ++y) {
      for (int x = std::max(0, x0); x < std::min(W, x0 + w); ++x) std::memcpy(&rgb[(sz(y) * sz(W) + sz(x)) * 3], col.data(), 3);
    }
  };
  const auto text = [&](const std::string& s, int x, int y, int scale, std::array<std::uint8_t, 3> col) {
    tools::draw_text(s, x, y, scale, [&](int px, int py) {
      if (px >= 0 && py >= 0 && px < W && py < Hh) std::memcpy(&rgb[(sz(py) * sz(W) + sz(px)) * 3], col.data(), 3);
    });
  };
  const auto blit = [&](std::span<const std::uint8_t> fr, int x0, int y0) {  // over a dark gradient
    for (int y = 0; y < S; ++y) {
      const float g = static_cast<float>(y) / static_cast<float>(S);
      const float bg[3] = {14.f + 18.f * g, 16.f + 22.f * g, 26.f + 30.f * g};
      for (int x = 0; x < S; ++x) {
        const std::uint8_t* p = fr.data() + (sz(y) * sz(S) + sz(x)) * 4;
        const float al = static_cast<float>(p[3]) / 255.f;
        for (int ch = 0; ch < 3; ++ch) {
          const float v = static_cast<float>(p[ch]) + bg[ch] * (1.f - al);
          rgb[(sz(y0 + y) * sz(W) + sz(x0 + x)) * 3 + sz(ch)] = static_cast<std::uint8_t>(std::min(255.f, v + 0.5f));
        }
      }
    }
  };
  const int frames = static_cast<int>(std::lround(secs * kFps));
  const std::string cmd = std::format("ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgb24 -s {}x{} -r 30 -i - -an -c:v libx264 -preset slow -crf {} -threads 2 "
                                      "-pix_fmt yuv420p -movflags +faststart \"{}\"",
                                      W, Hh, crf, out);
  std::FILE* video = popen(cmd.c_str(), "w");
  if (!video) throw std::runtime_error("cannot start ffmpeg");
  const std::string title = a.str("title", std::format("{} at {} px: native, half size + bilinear, half size + learned 2x", ename(e), S));
  play_pair(fx.e, S, ctl, seed, frames, arms, -1, [&](int, std::span<const std::uint8_t> n, const std::vector<std::vector<std::uint8_t>>& o) {
    fill(0, 0, W, Hh, {8, 8, 10});
    text(title, gap, 12, 2, {255, 255, 255});
    const std::array<std::string, 3> labels = {std::format("N: NATIVE {}", S), std::format("B: {} + BILINEAR", S / 2), std::format("L: {} + LEARNED 2X", S / 2)};
    for (int k = 0; k < 3; ++k) {
      const int x = gap + k * (S + gap);
      text(labels[sz(k)], x, title_h + 6, 2, {230, 230, 235});
      text(std::format("{} PX CROP, 2X", cs), x + (S - zs) / 2, row2 - label_h + 8, 2, {170, 170, 176});
      const std::span<const std::uint8_t> fr = k == 0 ? n : std::span<const std::uint8_t>(o[sz(k - 1)]);
      blit(fr, x, title_h + label_h);
      // the crop, outlined on the panel and enlarged below it
      for (int t = 0; t < cs; ++t) {
        for (const auto& [px, py] : std::array<std::pair<int, int>, 4>{{{cx + t, cy}, {cx + t, cy + cs - 1}, {cx, cy + t}, {cx + cs - 1, cy + t}}}) {
          std::uint8_t* q = &rgb[(sz(title_h + label_h + py) * sz(W) + sz(x + px)) * 3];
          q[0] = 255;
          q[1] = 220;
          q[2] = 120;
        }
      }
      const int x2 = x + (S - zs) / 2;
      for (int y = 0; y < zs; ++y) {
        for (int xx = 0; xx < zs; ++xx) {
          const std::uint8_t* src = &rgb[(sz(title_h + label_h + cy + y / zoom) * sz(W) + sz(x + cx + xx / zoom)) * 3];
          std::uint8_t* dst = &rgb[(sz(row2 + y) * sz(W) + sz(x2 + xx)) * 3];
          if (y / zoom == 0 || y / zoom == cs - 1 || xx / zoom == 0 || xx / zoom == cs - 1) {
            const std::uint8_t* in = &rgb[(sz(title_h + label_h + cy + std::clamp(y / zoom, 1, cs - 2)) * sz(W) + sz(x + cx + std::clamp(xx / zoom, 1, cs - 2))) * 3];
            std::memcpy(dst, in, 3);  // not the outline
          } else {
            std::memcpy(dst, src, 3);
          }
        }
      }
    }
    if (std::fwrite(rgb.data(), 1, rgb.size(), video) != rgb.size()) throw std::runtime_error("ffmpeg stopped reading");
  });
  if (pclose(video) != 0) throw std::runtime_error("ffmpeg failed");
  std::println("wrote {} ({} frames, {} bytes)", out, frames, fs::file_size(out));
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"null", "per", "resume", "help"});
  if (a.flag("help") || a.positional().empty()) {
    std::println("nvfx_study_p pin|info|check|tiles|train|dev|calib|eval|margins|decide|cost|video ... (see the source's header)");
    return 0;
  }
  Ctx c;
  c.data = a.str("root", data_root().string());
  c.results = a.str("results", "results/experiments/upscale");
  c.threads = a.i("threads", 2);
  const std::string cmd = a.positional()[0];
  int rc = 0;
  if (cmd == "pin") rc = step_pin(c);
  else if (cmd == "info") rc = step_info(a, c);
  else if (cmd == "check") rc = step_check(c);
  else if (cmd == "tiles") rc = step_tiles(a, c);
  else if (cmd == "train") rc = step_train(a, c);
  else if (cmd == "dev") rc = step_dev(a, c);
  else if (cmd == "calib") rc = step_calib(a, c);
  else if (cmd == "eval") rc = step_eval(a, c);
  else if (cmd == "margins") rc = step_margins(a, c);
  else if (cmd == "decide") rc = step_decide(a, c);
  else if (cmd == "choose") rc = step_choose(a, c);
  else if (cmd == "cost") rc = step_cost(a, c);
  else if (cmd == "anatomy") rc = step_anatomy(a, c);
  else if (cmd == "bench") rc = step_bench(a, c);
  else if (cmd == "tile-eval") rc = step_tile_eval(a, c);
  else if (cmd == "video") rc = step_video(a, c);
  else throw std::invalid_argument("unknown step " + cmd);
  a.warn_unused();
  return rc;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_study_p: {}", e.what());
  return 1;
}
