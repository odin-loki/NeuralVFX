// Study N: a small 3D demo (docs/STUDY_3D.md). The 3D simulator's pilot, the training runs, the 3D rollout model's
// training, validation and the test (once), demo videos and timing. Every step writes under --root (default
// $NEURALVFX_DATA/n_3d, else /root/nvfx-data/n_3d) and resumes from what it has written. `--study n2` selects study
// N2's seeds, held-out settings, data root (n2_3d) and edges (ambient air at the open sides; STUDY_3D.md §14).
//
//   nvfx_3d pilot  --widths 48,64,80 --seconds 12 --seeds 2       cost per frame, plume rise and puffing at each grid
//   nvfx_3d look   --width 64 --frames 150 --out DIR               stills of the simulation from the four cameras
//   nvfx_3d record --runs 40 --frames 240 [--from K]               training runs (seeds 9,000,000 + i; N2: 9,200,000 + i)
//   nvfx_3d train  --name base [--iterations 1500 --finetune 600 --activity 300 --batch 8 --unroll 12]
//   nvfx_3d finish --ckpt base[_stage2] --out NAME                 a model file with start points
//   nvfx_3d truth  --split val|test                                 the tracking runs' true fields (simulated once)
//   nvfx_3d track  --split S --method NAME [--model F]              tracking from the true state: a model, coarse, hold
//   nvfx_3d real   --split S                                        endless statistics of real runs (and the floor)
//   nvfx_3d play   --split S --method NAME [--model F]              endless statistics of a model's plays, or coarse
//   nvfx_3d compare --split S --a NAME --b NAME                     paired differences with 95% bootstrap intervals
//   nvfx_3d freeze --model F                                        the model the test runs on (results/3d/frozen.csv)
//   nvfx_3d video  --kind orbit|wind|real-vs-learned --out F.mp4    demo videos (videos/README.md)
//   nvfx_3d timing --threads 1,2,4                                  ms per step and per picture (results/3d/timing.csv)
//   nvfx_3d edge   --controls I,W,T                                 the heat at the box's sides as the model plays (the runaway)
//   nvfx_3d edge   --winds 0.05,0.25,0.5,0.75,0.95 --seconds 30 --out F.csv [--sim]   study N2's edge check (§14)
//   nvfx_3d edgecheck --csv F.csv                                   its verdict per model, wind and side
//   nvfx_3d reedge --model F --edges clamp|ambient --out G          a copy of a model file with other edges
//   nvfx_3d choose --split val --cands C2,C3,C4 --edge F.csv        study N2's choice among candidates (§14)
#include "args.hpp"
#include "label_font.hpp"

#include <neuralfx/image_io.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/nvfx_volume.h>
#include <neuralfx/render3d.hpp>
#include <neuralfx/rollout3d.hpp>
#include <neuralfx/rollout3d_train.hpp>
#include <neuralfx/sim3d.hpp>

#include "sha256.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <print>
#include <set>
#include <sstream>
#include <stdfloat>
#include <string>
#include <vector>

namespace nfx::rt {
void volume_conv3_avx2(int rx, int ry, int rz, const float* in, int ci, const float* W, const float* b, int co, float* out, int y0, int y1);
}  // namespace nfx::rt

namespace fs = std::filesystem;
namespace r3 = nfx::rollout3d;
namespace rd = nfx::render3d;
using nfx::tools::Args;

namespace {

std::size_t zu(int v) { return static_cast<std::size_t>(v); }
float fl(int v) { return static_cast<float>(v); }

double cpu_now() {
  timespec ts{};
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

bool g_n2 = false;  // --study n2 (STUDY_3D.md §14)

fs::path data_root(const Args& a) {
  if (a.has("root")) return a.str("root");
  const char* dir = g_n2 ? "n2_3d" : "n_3d";
  if (const char* e = std::getenv("NEURALVFX_DATA")) return fs::path(e) / dir;
  return fs::path("/root/nvfx-data") / dir;
}

void log_line(const fs::path& file, const std::string& line) {
  fs::create_directories(file.parent_path());
  std::ofstream o(file, std::ios::app);
  o << line << "\n";
  std::println("{}", line);
  std::fflush(stdout);
}

// --- the study's constants (docs/STUDY_3D.md §5) -------------------------------------------------------------------

constexpr std::uint64_t kTrainBase = 9'000'000;  // training runs: kTrainBase + i
constexpr std::uint64_t kValTrack = 9'400'000, kValReal = 9'410'000, kValPlay = 9'420'000;
constexpr std::uint64_t kTestTrack = 9'700'000, kTestReal = 9'710'000, kTestPlay = 9'720'000;
// study N2 (§14): its own blocks inside study N's range
constexpr std::uint64_t kN2Train = 9'200'000;
constexpr std::uint64_t kN2ValTrack = 9'500'000, kN2ValReal = 9'510'000, kN2ValPlay = 9'520'000;
constexpr std::uint64_t kN2TestTrack = 9'800'000, kN2TestReal = 9'810'000, kN2TestPlay = 9'820'000;
constexpr std::uint64_t kN2Edge = 900;  // the edge check: the plays' (or the real runs') base + 900 + the wind's index
constexpr std::uint64_t kFloor = 50;  // the floor's real runs: the real base + 50 + 100 s + k
constexpr int kWidth = 64, kHeight = 96;
constexpr int kTrackWarm = 100;  // frames simulated from an empty grid before a tracked state
constexpr std::array<int, 3> kHorizons{8, 30, 60};
constexpr int kEndlessWarm = 150, kEndlessFrames = 240;
constexpr int kPic = 128;  // the four cameras' pictures for the measures
constexpr float kPeakT = 1.5f, kPeakD = 0.2f;

r3::Hyper study_hyper() {
  r3::Hyper h;
  h.rx = kWidth / 4;
  h.ry = kHeight / 4;
  h.rz = kWidth / 4;
  h.fine = 4;
  h.hidden = 16;
  h.memory = 4;
  h.jacobi = 40;
  h.n_controls = 3;
  h.warmup = 24;
  return h;
}

std::uint64_t mix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float u01(std::uint64_t h) { return static_cast<float>(h >> 40) / 16777216.f; }

std::array<float, 3> train_controls(std::uint64_t seed) {
  const std::uint64_t h = mix64(seed * 3 + 1);
  return {u01(h), u01(mix64(h + 1)), u01(mix64(h + 2))};
}

float smooth01(float t) {
  t = std::clamp(t, 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}

// Study N2's training runs (STUDY_3D.md §14), seed 9,200,000 + i, by i % 3: 0, a wind at an extreme (towards -x, 0 to
// 0.15, when i / 3 is even; towards +x, 0.85 to 1, when odd), the other controls random; 1, controls that change during
// the run (n2_schedule); 2, random controls held, as study N's runs but longer.
int n2_kind(int i) { return i % 3; }
std::array<float, 3> n2_controls(int i, std::uint64_t seed) {
  std::array<float, 3> c = train_controls(seed);
  if (n2_kind(i) == 0) {
    const float u = u01(mix64(seed * 7 + 5));
    c[1] = (i / 3) % 2 == 0 ? 0.15f * u : 0.85f + 0.15f * u;
  }
  return c;
}
// Controls that change: segments of 2 to 4 s, each with new targets reached by a smooth 1 s ramp from the last ones;
// the wind's target is at an extreme (0 to 0.15 or 0.85 to 1) six times in ten, else anywhere; intensity and turbulence
// anywhere. Per frame (Run::schedule).
std::vector<float> n2_schedule(std::uint64_t seed, int frames, float fps) {
  std::vector<float> out(zu(frames) * 3);
  std::uint64_t h = mix64(seed ^ 0x5CEDULL);
  const auto next = [&] {
    h = mix64(h);
    return u01(h);
  };
  const auto target = [&] {
    std::array<float, 3> c{next(), next(), next()};
    const float pw = next(), u = next();
    if (pw < 0.3f) c[1] = 0.15f * u;
    else if (pw < 0.6f) c[1] = 0.85f + 0.15f * u;
    return c;
  };
  std::array<float, 3> from = target(), to = from;
  float seg_start = 0.f, seg_len = 2.f + 2.f * next();
  for (int f = 0; f < frames; ++f) {
    const float t = fl(f) / fps;
    while (t >= seg_start + seg_len) {
      seg_start += seg_len;
      seg_len = 2.f + 2.f * next();
      from = to;
      to = target();
    }
    const float r = smooth01(t - seg_start);
    for (int k = 0; k < 3; ++k) out[zu(f) * 3 + zu(k)] = from[zu(k)] + (to[zu(k)] - from[zu(k)]) * r;
  }
  return out;
}

// Smoke tests only (--max-settings, --max-runs): fewer settings and runs than the protocol's.
int g_max_settings = 1 << 20, g_max_runs = 1 << 20;

std::vector<std::array<float, 3>> settings_all(const std::string& split);
std::vector<std::array<float, 3>> settings(const std::string& split) {
  auto s = settings_all(split);
  if (static_cast<int>(s.size()) > g_max_settings) s.resize(zu(g_max_settings));
  return s;
}
std::vector<std::array<float, 3>> settings_all(const std::string& split) {
  // probe: for choices made before validation (the detail layer's constants), at the controls of training runs 0 to 3
  // with seeds of its own (9,300,000 + 100 s + k); never used for the rule
  if (split == "probe") return {train_controls(kTrainBase), train_controls(kTrainBase + 1), train_controls(kTrainBase + 2), train_controls(kTrainBase + 3)};
  if (g_n2) {  // study N2 (§14): new settings, a strong wind towards -x and towards +x in each split
    if (split == "val") return {{0.60f, 0.08f, 0.50f}, {0.40f, 0.92f, 0.65f}, {0.75f, 0.30f, 0.35f}, {0.50f, 0.50f, 0.80f}, {0.30f, 0.65f, 0.25f}};
    if (split == "test") {
      return {{0.70f, 0.05f, 0.40f}, {0.55f, 0.95f, 0.55f}, {0.45f, 0.20f, 0.70f}, {0.80f, 0.50f, 0.30f}, {0.35f, 0.75f, 0.45f}, {0.65f, 0.40f, 0.90f}};
    }
    throw std::invalid_argument("split must be val or test");
  }
  if (split == "val") return {{0.35f, 0.30f, 0.40f}, {0.65f, 0.70f, 0.60f}, {0.50f, 0.15f, 0.75f}, {0.80f, 0.55f, 0.25f}};
  if (split == "test") {
    return {{0.30f, 0.60f, 0.55f}, {0.45f, 0.25f, 0.30f}, {0.55f, 0.80f, 0.70f}, {0.70f, 0.40f, 0.85f}, {0.85f, 0.65f, 0.45f}, {0.60f, 0.50f, 0.20f}};
  }
  throw std::invalid_argument("split must be val or test");
}
int track_runs(const std::string& split) { return std::min(split == "probe" ? 2 : 3, g_max_runs); }
int endless_runs(const std::string& split) { return std::min(split == "test" ? 3 : 2, g_max_runs); }
std::uint64_t base_of(const std::string& split, const std::string& what) {
  if (split == "probe") return 9'300'000 + (what == "track" ? 0 : what == "real" ? 10'000 : 20'000);
  const bool t = split == "test";
  if (g_n2) {
    if (what == "track") return t ? kN2TestTrack : kN2ValTrack;
    if (what == "real") return t ? kN2TestReal : kN2ValReal;
    return t ? kN2TestPlay : kN2ValPlay;
  }
  if (what == "track") return t ? kTestTrack : kValTrack;
  if (what == "real") return t ? kTestReal : kValReal;
  return t ? kTestPlay : kValPlay;
}

nfx::sim3d::Params fine_params(const std::array<float, 3>& c, std::uint64_t seed, int threads, int width = kWidth, int height = kHeight) {
  nfx::sim3d::Params p;
  p.width = width;
  p.height = height;
  p.intensity = c[0];
  p.wind = c[1];
  p.turbulence = c[2];
  p.seed = seed;
  p.threads = threads;
  return p;
}

// The four cameras of the study: around the effect at 0, 90, 180 and 270 degrees, a little above the flame's middle.
rd::Camera study_camera(int k, float aspect, float height_world) {
  const float az = static_cast<float>(k) * 0.5f * std::numbers::pi_v<float>;
  const float r = 2.2f, ty = 0.42f * height_world;
  const std::array<float, 3> eye{r * std::sin(az), ty + 0.35f, r * std::cos(az)};
  return rd::look_at(eye, {0.f, ty, 0.f}, {0.f, 1.f, 0.f}, 42.f, aspect);
}

struct Fields {
  std::vector<float> t, d;  // fine (y, z, x) layout
};

// The faster convolution where the CPU has AVX2 (as the volume runtime picks it).
void fast_conv(r3::Workspace& ws) {
  if (nvfx_get_isa() != NVFX_ISA_BASELINE) ws.conv = nfx::rt::volume_conv3_avx2;
}

// --- measures -----------------------------------------------------------------------------------------------------

// Active PSNR (cells where either field exceeds 1% of the peak) of a field against the truth.
double field_psnr(const std::vector<float>& truth, const std::vector<float>& test, float peak) {
  double s = 0;
  std::size_t n = 0;
  for (std::size_t i = 0; i < truth.size(); ++i) {
    if (truth[i] <= 0.01f * peak && test[i] <= 0.01f * peak) continue;
    const double e = (static_cast<double>(test[i]) - truth[i]) / peak;
    s += e * e;
    ++n;
  }
  return n ? std::min(nfx::metrics::kPsnrCap, -10.0 * std::log10(std::max(1e-12, s / static_cast<double>(n)))) : nfx::metrics::kPsnrCap;
}

// Active PSNR of two RGBA8 pictures (pixels where either has a channel above 4 of 255, as metrics::score).
double picture_psnr(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  double s = 0, n = 0;
  for (std::size_t i = 0; i < a.size(); i += 4) {
    const bool on = std::max({a[i], a[i + 1], a[i + 2], a[i + 3]}) > 4 || std::max({b[i], b[i + 1], b[i + 2], b[i + 3]}) > 4;
    if (!on) continue;
    for (std::size_t c = 0; c < 4; ++c) {
      const double e = (double(a[i + c]) - double(b[i + c])) / 255.0;
      s += e * e;
    }
    n += 4;
  }
  return n > 0 ? nfx::metrics::psnr_from_mse(s / n) : nfx::metrics::kPsnrCap;
}

void pictures(rd::Renderer& r, nfx::Pool& pool, const Fields& f, std::array<std::vector<std::uint8_t>, 4>& out) {
  r.prepare(f.t, f.d, &pool);
  for (int k = 0; k < 4; ++k) {
    out[zu(k)].resize(zu(kPic) * zu(kPic) * 4);
    r.render(study_camera(k, 1.f, fl(kHeight) / fl(kWidth)), kPic, kPic, out[zu(k)].data(), zu(kPic) * 4, &pool);
  }
}

// The 3D detail spectrum: radially averaged power of a field over the cube of the lowest kWidth slabs (where the flame
// and the lower plume are), bins |k| = 1 .. kWidth / 2.
void fft(std::span<std::complex<double>> a) {
  const std::size_t n = a.size();
  for (std::size_t i = 1, j = 0; i < n; ++i) {
    std::size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (std::size_t len = 2; len <= n; len <<= 1) {
    const double ang = -2.0 * std::numbers::pi / static_cast<double>(len);
    const std::complex<double> wl(std::cos(ang), std::sin(ang));
    for (std::size_t i = 0; i < n; i += len) {
      std::complex<double> w(1.0);
      for (std::size_t j = 0; j < len / 2; ++j) {
        const auto u = a[i + j], v = a[i + j + len / 2] * w;
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
        w *= wl;
      }
    }
  }
}
void spectrum3d(const std::vector<float>& f, std::vector<double>& acc) {
  const int n = kWidth;
  acc.resize(zu(n / 2), 0.0);
  std::vector<std::complex<double>> c(zu(n) * zu(n) * zu(n)), line(zu(n));
  for (int y = 0; y < n; ++y) {
    for (int z = 0; z < n; ++z) {
      for (int x = 0; x < n; ++x) c[(zu(y) * zu(n) + zu(z)) * zu(n) + zu(x)] = f[(zu(y) * zu(kWidth) + zu(z)) * zu(kWidth) + zu(x)];
    }
  }
  const auto axis = [&](auto index) {
    for (int a = 0; a < n; ++a) {
      for (int b = 0; b < n; ++b) {
        for (int i = 0; i < n; ++i) line[zu(i)] = c[index(a, b, i)];
        fft(line);
        for (int i = 0; i < n; ++i) c[index(a, b, i)] = line[zu(i)];
      }
    }
  };
  axis([&](int a, int b, int i) { return (zu(a) * zu(n) + zu(b)) * zu(n) + zu(i); });
  axis([&](int a, int b, int i) { return (zu(a) * zu(n) + zu(i)) * zu(n) + zu(b); });
  axis([&](int a, int b, int i) { return (zu(i) * zu(n) + zu(a)) * zu(n) + zu(b); });
  std::vector<double> cnt(acc.size(), 0.0), pw(acc.size(), 0.0);
  for (int y = 0; y < n; ++y) {
    for (int z = 0; z < n; ++z) {
      for (int x = 0; x < n; ++x) {
        const int ky = y <= n / 2 ? y : y - n, kz = z <= n / 2 ? z : z - n, kx = x <= n / 2 ? x : x - n;
        const int r = static_cast<int>(std::lround(std::sqrt(double(kx * kx + ky * ky + kz * kz))));
        if (r < 1 || r > n / 2) continue;
        pw[zu(r - 1)] += std::norm(c[(zu(y) * zu(n) + zu(z)) * zu(n) + zu(x)]);
        cnt[zu(r - 1)] += 1;
      }
    }
  }
  for (std::size_t b = 0; b < acc.size(); ++b) acc[b] += pw[b] / std::max(1.0, cnt[b]);
}

// One run's endless statistics: per camera the coverage and emission curves, the motion and the mean picture; the 3D
// spectra of heat and soot (power summed over the frames sampled).
struct Unit {
  int frames = 0;
  std::array<std::vector<double>, 4> coverage, emission;
  std::array<double, 4> motion{};
  std::array<std::vector<float>, 4> mean_frame;
  std::vector<double> spec_t, spec_d;
  int spec_frames = 0;
};

class UnitBuilder {
 public:
  explicit UnitBuilder(int frames) {
    u.frames = frames;
    for (int k = 0; k < 4; ++k) u.mean_frame[zu(k)].assign(zu(kPic) * zu(kPic) * 4, 0.f);
  }
  void add(const std::array<std::vector<std::uint8_t>, 4>& pics, const Fields& f, int frame) {
    for (int k = 0; k < 4; ++k) {
      const auto& p = pics[zu(k)];
      double cov = 0, emi = 0;
      for (std::size_t i = 0; i < p.size(); i += 4) {
        const double a = p[i + 3] / 255.0;
        cov += a;
        for (std::size_t c = 0; c < 3; ++c) emi += std::max(0.0, p[i + c] / 255.0 - a) / 3.0;
        for (std::size_t c = 0; c < 4; ++c) u.mean_frame[zu(k)][i + c] += static_cast<float>(p[i + c]) / static_cast<float>(u.frames);
      }
      const double px = static_cast<double>(p.size() / 4);
      u.coverage[zu(k)].push_back(cov / px);
      u.emission[zu(k)].push_back(emi / px);
      if (!prev[zu(k)].empty()) {
        double d = 0;
        for (std::size_t i = 0; i < p.size(); ++i) d += std::abs(double(p[i]) - double(prev[zu(k)][i]));
        u.motion[zu(k)] += d / 255.0 / static_cast<double>(p.size()) / static_cast<double>(u.frames - 1);
      }
      prev[zu(k)] = p;
    }
    if (frame % 4 == 0) {
      spectrum3d(f.t, u.spec_t);
      spectrum3d(f.d, u.spec_d);
      ++u.spec_frames;
    }
  }
  Unit u;

 private:
  std::array<std::vector<std::uint8_t>, 4> prev;
};

void save_unit(const fs::path& p, const Unit& u) {
  fs::create_directories(p.parent_path());
  const fs::path tmp = p.string() + ".tmp";
  {
    std::ofstream o(tmp, std::ios::binary);
    const auto put = [&](const auto& v) { o.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
    const auto putv = [&](const auto& v) {
      const std::uint64_t n = v.size();
      put(n);
      o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(n * sizeof(v[0])));
    };
    put(u.frames);
    put(u.spec_frames);
    for (int k = 0; k < 4; ++k) {
      putv(u.coverage[zu(k)]);
      putv(u.emission[zu(k)]);
      put(u.motion[zu(k)]);
      putv(u.mean_frame[zu(k)]);
    }
    putv(u.spec_t);
    putv(u.spec_d);
  }
  fs::rename(tmp, p);
}

bool load_unit(const fs::path& p, Unit& u) {
  std::ifstream i(p, std::ios::binary);
  if (!i) return false;
  const auto get = [&](auto& v) { i.read(reinterpret_cast<char*>(&v), sizeof(v)); };
  const auto getv = [&](auto& v) {
    std::uint64_t n = 0;
    get(n);
    v.resize(n);
    i.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(v[0])));
  };
  get(u.frames);
  get(u.spec_frames);
  for (int k = 0; k < 4; ++k) {
    getv(u.coverage[zu(k)]);
    getv(u.emission[zu(k)]);
    get(u.motion[zu(k)]);
    getv(u.mean_frame[zu(k)]);
  }
  getv(u.spec_t);
  getv(u.spec_d);
  return static_cast<bool>(i);
}

// Statistics of several runs pooled (study D2): curves, mean pictures, motion and spectra averaged over the runs.
Unit pool_units(const std::vector<Unit>& us) {
  Unit p = us.at(0);
  const double n = static_cast<double>(us.size());
  for (int k = 0; k < 4; ++k) {
    for (std::size_t j = 0; j < p.coverage[zu(k)].size(); ++j) {
      double c = 0, e = 0;
      for (const Unit& u : us) {
        c += u.coverage[zu(k)][j];
        e += u.emission[zu(k)][j];
      }
      p.coverage[zu(k)][j] = c / n;
      p.emission[zu(k)][j] = e / n;
    }
    double m = 0;
    for (const Unit& u : us) m += u.motion[zu(k)];
    p.motion[zu(k)] = m / n;
    for (std::size_t j = 0; j < p.mean_frame[zu(k)].size(); ++j) {
      double s = 0;
      for (const Unit& u : us) s += u.mean_frame[zu(k)][j];
      p.mean_frame[zu(k)][j] = static_cast<float>(s / n);
    }
  }
  for (std::size_t b = 0; b < p.spec_t.size(); ++b) {
    double t = 0, d = 0;
    for (const Unit& u : us) {
      t += u.spec_t[b] / std::max(1, u.spec_frames);
      d += u.spec_d[b] / std::max(1, u.spec_frames);
    }
    p.spec_t[b] = t / n;
    p.spec_d[b] = d / n;
  }
  p.spec_frames = 1;
  return p;
}

// The five endless statistics of a pooled unit against the real runs' (docs/STUDY_3D.md §5).
struct Endless {
  double spectrum = 0, motion = 0, coverage = 0, emission = 0, mean_psnr = 0, motion_ratio = 0;
};
Endless distance(const Unit& real, const Unit& test) {
  Endless e;
  double lr = 0, mr = 0;
  const auto l1 = [](const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) s += std::abs(a[i] - b[i]);
    return n ? s / static_cast<double>(n) : 0.0;
  };
  for (int k = 0; k < 4; ++k) {
    e.coverage += l1(real.coverage[zu(k)], test.coverage[zu(k)]) / 4;
    e.emission += l1(real.emission[zu(k)], test.emission[zu(k)]) / 4;
    double m = 0;
    for (std::size_t i = 0; i < real.mean_frame[zu(k)].size(); ++i) {
      const double d = (real.mean_frame[zu(k)][i] - test.mean_frame[zu(k)][i]) / 255.0;
      m += d * d;
    }
    e.mean_psnr += nfx::metrics::psnr_from_mse(m / static_cast<double>(real.mean_frame[zu(k)].size())) / 4;
    lr += real.motion[zu(k)];
    mr += test.motion[zu(k)];
  }
  e.motion_ratio = lr > 0 ? mr / lr : 0;
  e.motion = std::abs(std::log(std::max(1e-9, e.motion_ratio)));
  double s = 0;
  for (std::size_t b = 0; b < real.spec_t.size(); ++b) {
    s += std::abs(std::log10(1e-12 + test.spec_t[b] / std::max(1, test.spec_frames)) - std::log10(1e-12 + real.spec_t[b] / std::max(1, real.spec_frames)));
    s += std::abs(std::log10(1e-12 + test.spec_d[b] / std::max(1, test.spec_frames)) - std::log10(1e-12 + real.spec_d[b] / std::max(1, real.spec_frames)));
  }
  e.spectrum = s / (2.0 * static_cast<double>(real.spec_t.size()));
  return e;
}

// --- recording and training --------------------------------------------------------------------------------------

std::vector<r3::Run> load_runs(const fs::path& dir) {
  std::vector<fs::path> files;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.path().extension() == ".n3run") files.push_back(e.path());
  }
  std::ranges::sort(files);
  std::vector<r3::Run> runs;
  for (const auto& f : files) {
    auto r = r3::load_run(f);
    if (!r) throw std::runtime_error(r.error());
    runs.push_back(std::move(*r));
  }
  return runs;
}

int record(const Args& a) {
  const fs::path root = data_root(a), dir = root / "runs" / a.str("runs-dir", "train");
  fs::create_directories(dir);
  const int runs = a.i("runs", 40), frames = a.i("frames", g_n2 ? 540 : 240), threads = a.i("threads", 3);
  const std::uint64_t base = a.u64("seed-base", g_n2 ? kN2Train : kTrainBase);  // fire 9,000,000; smoke 9,100,000; N2 9,200,000
  nfx::sim3d::Effect effect = nfx::sim3d::Effect::fire;
  if (!nfx::sim3d::parse_effect(a.str("effect", "fire"), effect)) throw std::invalid_argument("--effect fire|smoke");
  const r3::Hyper h = study_hyper();
  for (int i = a.i("from", 0); i < runs; ++i) {  // --from: a second job records the later runs
    const fs::path f = dir / std::format("run_{:04}.n3run", i);
    if (fs::exists(f)) continue;
    const std::uint64_t seed = base + static_cast<std::uint64_t>(i);
    const std::vector<float> sched = g_n2 && n2_kind(i) == 1 ? n2_schedule(seed, frames, 30.f) : std::vector<float>{};
    const auto c = g_n2 ? (sched.empty() ? n2_controls(i, seed) : std::array<float, 3>{sched[0], sched[1], sched[2]}) : train_controls(seed);
    const double t0 = now(), c0 = cpu_now();
    auto p = fine_params(c, seed, threads);
    p.effect = effect;
    const r3::Run r = r3::record_run(p, frames, h, sched);
    if (auto e = r3::save_run(f, r); !e) throw std::runtime_error(e.error());
    std::string kind;
    if (g_n2) {
      kind = n2_kind(i) == 0 ? " (wind at an extreme)" : n2_kind(i) == 2 ? " (random, held)" : "";
      if (!sched.empty()) {
        float wlo = 1.f, whi = 0.f;
        for (std::size_t k = 1; k < sched.size(); k += 3) {
          wlo = std::min(wlo, sched[k]);
          whi = std::max(whi, sched[k]);
        }
        kind = std::format(" (changing: wind {:.2f} to {:.2f})", wlo, whi);
      }
    }
    log_line(root / "logs" / ("record_" + a.str("runs-dir", "train") + ".log"), std::format("run {} seed {} controls {:.3f} {:.3f} {:.3f}{}, {} frames: {:.1f} s wall, {:.1f} s CPU", i, seed,
                                                       c[0], c[1], c[2], kind, frames, now() - t0, cpu_now() - c0));
  }
  return 0;
}

// The training runs: root/runs/<runs-dir>, and the runs in --also (a directory; study N2 also trains on study N's runs).
std::vector<r3::Run> training_runs(const Args& a, const fs::path& root) {
  auto runs = load_runs(root / "runs" / a.str("runs-dir", "train"));
  if (a.has("also")) {
    auto more = load_runs(a.str("also"));
    for (auto& r : more) runs.push_back(std::move(r));
  }
  return runs;
}

r3::Model fresh_model(nfx::sim3d::Effect e = nfx::sim3d::Effect::fire) {
  r3::Model m = r3::init_model(study_hyper(), g_n2 ? 9'200'777 : 9'000'777);
  m.edges = g_n2 ? r3::Edges::ambient : r3::Edges::clamp;  // study N clamped; N2 brings ambient air in (§14)
  m.effect = e == nfx::sim3d::Effect::fire ? "fire3d" : "smoke3d";
  m.fps = 30.f;
  m.noise = nfx::sim3d::forcing(e);
  m.control_names = {"intensity", "wind", "turbulence"};
  return m;
}

r3::TrainOptions train_options(const Args& a) {
  r3::TrainOptions o;
  o.iterations = a.i("iterations", 1500);
  o.finetune = a.i("finetune", 600);
  o.activity_stage = a.i("activity", 300);
  o.long_stage = a.i("long", 0);
  o.burn_long_min = a.i("burn-long-min", 48);
  o.burn_long_max = a.i("burn-long-max", 240);
  o.batch = a.i("batch", 8);
  o.max_unroll = a.i("unroll", 12);
  o.burn_max = a.i("burn", 24);
  o.lr = a.f("lr", 2e-3f);
  o.lr_finetune = a.f("lr-ft", 7e-4f);
  o.threads = a.i("threads", 3);
  o.seed = a.u64("seed", 1);
  o.log_every = a.i("log-every", 10);
  o.checkpoint_every = a.i("checkpoint-every", 20);
  o.stop_at = a.i("stop-at", 0);
  return o;
}

int train(const Args& a) {
  const fs::path root = data_root(a);
  const std::string name = a.str("name", "base");
  const fs::path ck = root / "ckpt" / (name + ".ckpt"), log = root / "logs" / ("train_" + name + ".log");
  fs::create_directories(ck.parent_path());
  const auto runs = training_runs(a, root);
  if (runs.empty()) throw std::runtime_error("no training runs: nvfx_3d record first");
  r3::Model m = fresh_model(runs.at(0).p.effect);
  r3::set_normalisation(m, runs);
  r3::TrainState s;
  if (fs::exists(ck)) {
    if (auto e = r3::load_checkpoint(ck, m, s); !e) throw std::runtime_error(e.error());
    log_line(log, std::format("resuming {} at iteration {}", name, s.iteration));
  } else {
    log_line(log, std::format("training {} on {} runs; scales {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}", name, runs.size(), m.scale[0], m.scale[1], m.scale[2], m.scale[3],
                              m.scale[4]));
  }
  r3::TrainOptions o = train_options(a);
  const int stage2_end = o.iterations + o.finetune;
  const double c0 = cpu_now();
  log_line(log, std::format("schedule: {} + {} + {} + {} iterations (stage 4 burn {} to {}), batch {}, unroll {}, threads {}", o.iterations, o.finetune,
                            o.activity_stage, o.long_stage, o.burn_long_min, o.burn_long_max, o.batch, o.max_unroll, o.threads));
  o.progress = [&](int it, int unroll, double loss, double seconds) {
    log_line(log, std::format("it {} unroll {} loss {:.6f} ({:.0f} s, {:.0f} CPU s)", it, unroll, loss, seconds, cpu_now() - c0));
  };
  r3::train_stepper(m, runs, o, s, [&](const r3::Model& mm, const r3::TrainState& ss) {
    if (auto e = r3::save_checkpoint(ck, mm, ss); !e) throw std::runtime_error(e.error());
    if (!g_n2 && ss.iteration == stage2_end) {  // candidate A of the validation: the stepper after stages 1 and 2
      if (auto e = r3::save_checkpoint(root / "ckpt" / (name + "_stage2.ckpt"), mm, ss); !e) throw std::runtime_error(e.error());
    }
    for (int k = 1; g_n2 && k <= 4; ++k) {  // study N2: a snapshot at the end of every stage
      if (ss.iteration == r3::stage_end(o, k) && (k == 1 || r3::stage_end(o, k) != r3::stage_end(o, k - 1))) {
        if (auto e = r3::save_checkpoint(root / "ckpt" / std::format("{}_stage{}.ckpt", name, k), mm, ss); !e) throw std::runtime_error(e.error());
        log_line(log, std::format("snapshot {}_stage{} at iteration {}", name, k, ss.iteration));
      }
    }
  });
  log_line(log, std::format("done at iteration {}", s.iteration));
  return 0;
}

int finish(const Args& a) {
  const fs::path root = data_root(a);
  const std::string ckname = a.need("ckpt"), out = a.need("out");
  const auto runs = training_runs(a, root);
  r3::Model m = fresh_model(runs.at(0).p.effect);
  r3::set_normalisation(m, runs);
  r3::TrainState s;
  if (auto e = r3::load_checkpoint(root / "ckpt" / (ckname + ".ckpt"), m, s); !e) throw std::runtime_error(e.error());
  r3::choose_starts(m, runs, a.i("starts", 8), a.i("frame", 150));
  m.detail.swirl = a.f("swirl", m.detail.swirl);
  m.detail.contrast = a.f("contrast", m.detail.contrast);
  m.detail.grow = a.f("grow", m.detail.grow);
  const fs::path f = root / "models" / (out + ".nvfx3");
  fs::create_directories(f.parent_path());
  if (auto e = r3::save_model(f, m); !e) throw std::runtime_error(e.error());
  std::println("wrote {} ({} bytes stored, {} start points, iteration {})", f.string(), m.storage_bytes(), m.starts.size(), s.iteration);
  return 0;
}

r3::Model load_study_model(const fs::path& f) {
  auto m = r3::load_model(f);
  if (!m) throw std::runtime_error(f.string() + ": " + m.error());
  return std::move(*m);
}

std::string sha_of(const fs::path& f) {
  std::ifstream i(f, std::ios::binary);
  const std::vector<std::uint8_t> b((std::istreambuf_iterator<char>(i)), std::istreambuf_iterator<char>());
  return nfx::pack::hex(nfx::pack::sha256(b));
}

// The test runs once, on a model frozen before it (results/3d/frozen.csv).
void require_frozen(const Args& a, const std::string& split, const fs::path& model) {
  if (split != "test") return;
  const fs::path frozen = a.str("frozen", "results/3d/frozen.csv");
  std::ifstream i(frozen);
  std::string line;
  const std::string sha = sha_of(model);
  while (std::getline(i, line)) {
    if (line.find(sha) != std::string::npos) return;
  }
  throw std::runtime_error("the test runs only on the frozen model (" + frozen.string() + "); " + model.string() + " is " + sha);
}

// --- the coarse simulation with the same detail layer (study D0's baseline) -----------------------------------------

struct CoarseSim {
  std::unique_ptr<nfx::sim3d::Fluid> f;
  r3::State s;
  void load(const r3::Model& m, const std::vector<float>& phys_frame_units, float time, const std::array<float, 3>& c, std::uint64_t seed, int threads) {
    const r3::Hyper& h = m.h;
    f = std::make_unique<nfx::sim3d::Fluid>(fine_params(c, seed, threads, h.rx, h.ry));
    nfx::sim3d::State st;
    st.nx = h.rx;
    st.ny = h.ry;
    st.nz = h.rz;
    const std::size_t n = zu(h.cells());
    for (auto* v : {&st.u, &st.v, &st.w, &st.temp, &st.soot, &st.pressure}) v->assign(n, 0.f);
    for (std::size_t i = 0; i < n; ++i) {  // coarse cells per frame -> per second
      st.u[i] = phys_frame_units[i * r3::kPhys] * m.fps;
      st.v[i] = phys_frame_units[i * r3::kPhys + 1] * m.fps;
      st.w[i] = phys_frame_units[i * r3::kPhys + 2] * m.fps;
      st.temp[i] = phys_frame_units[i * r3::kPhys + 3];
      st.soot[i] = phys_frame_units[i * r3::kPhys + 4];
    }
    st.time = time;
    st.frame = static_cast<int>(std::lround(time * m.fps));
    f->set_state(st);
  }
  // one frame of the coarse simulation, then the detail layer driven by it
  void step(const r3::Model& m, std::uint64_t seed, const std::array<float, 3>& c, r3::Workspace& ws, nfx::Pool& pool) {
    f->step_frame();
    const auto st = f->state();
    const int C = m.h.channels();
    for (int i = 0; i < m.h.cells(); ++i) {
      const std::size_t j = zu(i);
      s.coarse[j * zu(C)] = st.u[j] / m.fps;
      s.coarse[j * zu(C) + 1] = st.v[j] / m.fps;
      s.coarse[j * zu(C) + 2] = st.w[j] / m.fps;
      s.coarse[j * zu(C) + 3] = st.temp[j];
      s.coarse[j * zu(C) + 4] = st.soot[j];
      for (int k = 0; k < 3; ++k) s.flow[j * 3 + zu(k)] = s.coarse[j * zu(C) + zu(k)];
    }
    r3::detail_step(m, s, seed, std::vector<float>(c.begin(), c.end()), ws, &pool);
    s.time += 1.f / m.fps;
    s.since_start += 1.f / m.fps;
  }
};

std::vector<float> coarse_of(const r3::Hyper& h, const nfx::sim3d::Fluid& f) {
  std::vector<float> c(zu(h.cells()) * r3::kPhys);
  r3::coarse_from_sim(f, h, c);
  return c;
}

// --- tracking ----------------------------------------------------------------------------------------------------------

// A tracked run's truth: the coarse state and the fine fields at the start, and the fine fields at each horizon.
struct Truth {
  std::array<float, 3> controls{};
  std::uint64_t seed = 0;
  float time = 0;
  std::vector<float> coarse;
  Fields start;
  std::array<Fields, 3> at;
};

void save_truth(const fs::path& p, const Truth& t) {
  fs::create_directories(p.parent_path());
  const fs::path tmp = p.string() + ".tmp";
  {
    std::ofstream o(tmp, std::ios::binary);
    const auto put = [&](const auto& v) { o.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
    const auto putv = [&](const std::vector<float>& v) {
      const std::uint64_t n = v.size();
      put(n);
      for (const float x : v) put(std::bit_cast<std::uint16_t>(static_cast<std::float16_t>(x)));
    };
    put(t.controls);
    put(t.seed);
    put(t.time);
    const std::uint64_t nc = t.coarse.size();
    put(nc);
    o.write(reinterpret_cast<const char*>(t.coarse.data()), static_cast<std::streamsize>(nc * 4));
    putv(t.start.t);
    putv(t.start.d);
    for (const auto& f : t.at) {
      putv(f.t);
      putv(f.d);
    }
  }
  fs::rename(tmp, p);
}

bool load_truth(const fs::path& p, Truth& t) {
  std::ifstream i(p, std::ios::binary);
  if (!i) return false;
  const auto get = [&](auto& v) { i.read(reinterpret_cast<char*>(&v), sizeof(v)); };
  const auto getv = [&](std::vector<float>& v) {
    std::uint64_t n = 0;
    get(n);
    std::vector<std::uint16_t> h(n);
    i.read(reinterpret_cast<char*>(h.data()), static_cast<std::streamsize>(n * 2));
    v.resize(n);
    for (std::size_t k = 0; k < n; ++k) v[k] = static_cast<float>(std::bit_cast<std::float16_t>(h[k]));
  };
  get(t.controls);
  get(t.seed);
  get(t.time);
  std::uint64_t nc = 0;
  get(nc);
  t.coarse.resize(nc);
  i.read(reinterpret_cast<char*>(t.coarse.data()), static_cast<std::streamsize>(nc * 4));
  getv(t.start.t);
  getv(t.start.d);
  for (auto& f : t.at) {
    getv(f.t);
    getv(f.d);
  }
  return static_cast<bool>(i);
}

int truth(const Args& a) {
  const fs::path root = data_root(a);
  const std::string split = a.need("split");
  const int threads = a.i("threads", 3);
  const r3::Hyper h = study_hyper();
  const auto sets = settings(split);
  for (int s = 0; s < static_cast<int>(sets.size()); ++s) {
    for (int k = 0; k < track_runs(split); ++k) {
      const fs::path f = root / "eval" / split / "truth" / std::format("s{}_k{}.bin", s, k);
      if (fs::exists(f)) continue;
      const double t0 = now();
      Truth t;
      t.controls = sets[zu(s)];
      t.seed = base_of(split, "track") + 100 * static_cast<std::uint64_t>(s) + static_cast<std::uint64_t>(k);
      nfx::sim3d::Fluid sim(fine_params(t.controls, t.seed, threads));
      for (int fr = 0; fr < kTrackWarm; ++fr) sim.step_frame();
      t.time = sim.time();
      t.coarse = coarse_of(h, sim);
      const std::size_t n = h.fine_cells();
      t.start.t.resize(n);
      t.start.d.resize(n);
      sim.fields(t.start.t, t.start.d);
      int done = 0;
      for (std::size_t j = 0; j < kHorizons.size(); ++j) {
        while (done < kHorizons[j]) {
          sim.step_frame();
          ++done;
        }
        t.at[j].t.resize(n);
        t.at[j].d.resize(n);
        sim.fields(t.at[j].t, t.at[j].d);
      }
      save_truth(f, t);
      log_line(root / "logs" / ("truth_" + split + ".log"), std::format("{} setting {} run {} seed {}: {:.0f} s", split, s, k, t.seed, now() - t0));
    }
  }
  return 0;
}

int track(const Args& a) {
  const fs::path root = data_root(a);
  const std::string split = a.need("split"), method = a.need("method");
  const int threads = a.i("threads", 3);
  nfx::Pool pool(threads);
  // the model: the learned stepper (a model file), or the detail layer's constants for coarse and hold
  const fs::path model_file = a.has("model") ? fs::path(a.str("model")) : root / "models" / "chosen.nvfx3";
  r3::Model m = load_study_model(model_file);
  r3::quantise_like_storage(m);
  const bool learned = method != "coarse" && method != "hold";
  if (learned) require_frozen(a, split, model_file);
  r3::Workspace ws(m);
  fast_conv(ws);
  rd::Renderer rend(m.h.fx(), m.h.fy(), m.h.fz());
  const auto sets = settings(split);
  const fs::path out = root / "eval" / split / "track" / (method + ".csv");
  fs::create_directories(out.parent_path());
  std::set<std::string> have;
  {
    std::ifstream i(out);
    std::string line;
    while (std::getline(i, line)) {
      std::istringstream ss(line);
      std::string s, k;
      std::getline(ss, s, ',');
      std::getline(ss, k, ',');
      have.insert(s + "," + k);
    }
  }
  std::array<std::vector<std::uint8_t>, 4> pt, pm;
  for (int s = 0; s < static_cast<int>(sets.size()); ++s) {
    for (int k = 0; k < track_runs(split); ++k) {
      if (have.count(std::format("{},{}", s, k))) continue;
      Truth t;
      if (!load_truth(root / "eval" / split / "truth" / std::format("s{}_k{}.bin", s, k), t)) throw std::runtime_error("missing truth: nvfx_3d truth first");
      const double t0 = now();
      const std::vector<float> cv(t.controls.begin(), t.controls.end());
      r3::State st = r3::make_state(m);
      CoarseSim cs;
      if (learned) {
        r3::start_from(m, t.coarse, t.time, cv, t.seed, st, ws, &pool, false);
        st.fine_t = t.start.t;
        st.fine_d = t.start.d;
      } else if (method == "coarse") {
        cs.s = r3::make_state(m);
        r3::start_from(m, t.coarse, t.time, cv, t.seed, cs.s, ws, &pool, false);
        cs.s.fine_t = t.start.t;
        cs.s.fine_d = t.start.d;
        cs.load(m, t.coarse, t.time, t.controls, t.seed, threads);
      }
      std::string row = std::format("{},{}", s, k);
      int done = 0;
      for (std::size_t j = 0; j < kHorizons.size(); ++j) {
        while (done < kHorizons[j]) {
          if (learned) r3::step(m, st, cv, t.seed, ws, &pool);
          else if (method == "coarse") cs.step(m, t.seed, t.controls, ws, pool);
          ++done;
        }
        Fields test;
        if (learned) test = {st.fine_t, st.fine_d};
        else if (method == "coarse") test = {cs.s.fine_t, cs.s.fine_d};
        else test = t.start;
        const double ft = field_psnr(t.at[j].t, test.t, kPeakT), fd = field_psnr(t.at[j].d, test.d, kPeakD);
        pictures(rend, pool, t.at[j], pt);
        pictures(rend, pool, test, pm);
        double pic = 0;
        std::string pics;
        for (int c = 0; c < 4; ++c) {
          const double v = picture_psnr(pt[zu(c)], pm[zu(c)]);
          pic += v / 4;
          pics += std::format(",{:.3f}", v);
        }
        row += std::format(",{},{:.3f},{:.3f},{:.3f}{}", kHorizons[j], ft, fd, pic, pics);
      }
      log_line(out, row);
      std::println(stderr, "  ({:.0f} s)", now() - t0);
    }
  }
  return 0;
}

// --- endless statistics ------------------------------------------------------------------------------------------

int nearest_start(const r3::Model& m, const std::array<float, 3>& c) {
  int best = 0;
  float bd = 1e30f;
  for (int i = 0; i < static_cast<int>(m.starts.size()); ++i) {
    float d = 0;
    for (int k = 0; k < 3; ++k) d += (m.starts[zu(i)].controls[zu(k)] - c[zu(k)]) * (m.starts[zu(i)].controls[zu(k)] - c[zu(k)]);
    if (d < bd) {
      bd = d;
      best = i;
    }
  }
  return best;
}

int real(const Args& a) {
  const fs::path root = data_root(a);
  const std::string split = a.need("split");
  const int threads = a.i("threads", 3);
  nfx::Pool pool(threads);
  rd::Renderer rend(kWidth, kHeight, kWidth);
  const auto sets = settings(split);
  const bool floor = split == "test" || a.flag("floor");
  for (int s = 0; s < static_cast<int>(sets.size()); ++s) {
    for (int which = 0; which < (floor ? 2 : 1); ++which) {
      for (int k = 0; k < endless_runs(split); ++k) {
        const std::string label = which == 0 ? "real" : "floor";
        const fs::path f = root / "eval" / split / "endless" / label / std::format("s{}_k{}.unit", s, k);
        if (fs::exists(f)) continue;
        const double t0 = now();
        const std::uint64_t seed = base_of(split, "real") + (which ? kFloor : 0) + 100 * static_cast<std::uint64_t>(s) + static_cast<std::uint64_t>(k);
        nfx::sim3d::Fluid sim(fine_params(sets[zu(s)], seed, threads));
        for (int fr = 0; fr < kEndlessWarm; ++fr) sim.step_frame();
        UnitBuilder ub(kEndlessFrames);
        const std::size_t n = zu(sim.nx()) * zu(sim.ny()) * zu(sim.nz());
        Fields fld{std::vector<float>(n), std::vector<float>(n)};
        std::array<std::vector<std::uint8_t>, 4> pics;
        for (int fr = 0; fr < kEndlessFrames; ++fr) {
          sim.step_frame();
          sim.fields(fld.t, fld.d);
          pictures(rend, pool, fld, pics);
          ub.add(pics, fld, fr);
        }
        save_unit(f, ub.u);
        log_line(root / "logs" / ("real_" + split + ".log"), std::format("{} {} setting {} run {} seed {}: {:.0f} s", split, label, s, k, seed, now() - t0));
      }
    }
  }
  return 0;
}

int play(const Args& a) {
  const fs::path root = data_root(a);
  const std::string split = a.need("split"), method = a.need("method");
  const int threads = a.i("threads", 3);
  nfx::Pool pool(threads);
  const fs::path model_file = a.has("model") ? fs::path(a.str("model")) : root / "models" / "chosen.nvfx3";
  r3::Model m = load_study_model(model_file);
  r3::quantise_like_storage(m);
  const bool learned = method != "coarse";
  if (learned) require_frozen(a, split, model_file);
  r3::Workspace ws(m);
  fast_conv(ws);
  rd::Renderer rend(m.h.fx(), m.h.fy(), m.h.fz());
  const auto sets = settings(split);
  for (int s = 0; s < static_cast<int>(sets.size()); ++s) {
    for (int k = 0; k < endless_runs(split); ++k) {
      const fs::path f = root / "eval" / split / "endless" / method / std::format("s{}_k{}.unit", s, k);
      if (fs::exists(f)) continue;
      const double t0 = now();
      const std::uint64_t seed = base_of(split, "play") + 100 * static_cast<std::uint64_t>(s) + static_cast<std::uint64_t>(k);
      const auto c = sets[zu(s)];
      const std::vector<float> cv(c.begin(), c.end());
      const int si = nearest_start(m, c);
      r3::State st = r3::make_state(m);
      CoarseSim cs;
      if (learned) {
        r3::start(m, si, cv, seed, st, ws, &pool);
      } else {  // the coarse simulation from the same start point, its fine fields grown by the same warm-up
        cs.s = r3::make_state(m);
        r3::start_from(m, m.starts[zu(si)].coarse, m.starts[zu(si)].time, cv, seed, cs.s, ws, &pool, false);
        cs.load(m, m.starts[zu(si)].coarse, m.starts[zu(si)].time, c, seed, threads);
        for (int fr = 0; fr < m.h.warmup; ++fr) cs.step(m, seed, c, ws, pool);
        cs.s.since_start = m.detail.swirl_ramp;
      }
      UnitBuilder ub(kEndlessFrames);
      std::array<std::vector<std::uint8_t>, 4> pics;
      for (int fr = 0; fr < kEndlessFrames; ++fr) {
        Fields fld;
        if (learned) {
          r3::step(m, st, cv, seed, ws, &pool);
          fld = {st.fine_t, st.fine_d};
        } else {
          cs.step(m, seed, c, ws, pool);
          fld = {cs.s.fine_t, cs.s.fine_d};
        }
        pictures(rend, pool, fld, pics);
        ub.add(pics, fld, fr);
      }
      save_unit(f, ub.u);
      log_line(root / "logs" / ("play_" + split + ".log"), std::format("{} {} setting {} run {} seed {} start {}: {:.0f} s", split, method, s, k, seed, si, now() - t0));
    }
  }
  return 0;
}

// --- comparisons ---------------------------------------------------------------------------------------------------

struct TrackRow {
  int s = 0, k = 0;
  std::array<std::array<double, 3>, 3> v{};  // horizon x (field T, field soot, picture mean)
};
std::map<std::pair<int, int>, TrackRow> read_track(const fs::path& f) {
  std::map<std::pair<int, int>, TrackRow> out;
  std::ifstream i(f);
  std::string line;
  while (std::getline(i, line)) {
    std::vector<std::string> t;
    std::istringstream ss(line);
    std::string x;
    while (std::getline(ss, x, ',')) t.push_back(x);
    if (t.size() < 2 + 3 * 8) continue;
    TrackRow r;
    r.s = std::stoi(t[0]);
    r.k = std::stoi(t[1]);
    for (int j = 0; j < 3; ++j) {
      const std::size_t b = 2 + zu(j) * 8;
      r.v[zu(j)] = {std::stod(t[b + 1]), std::stod(t[b + 2]), std::stod(t[b + 3])};
    }
    out[{r.s, r.k}] = r;
  }
  return out;
}

// judged: false for a statistic reported for context only (the motion ratio, whose ideal is 1, not more or less)
std::string interval(const nfx::metrics::Interval& iv, int digits, bool higher_better, bool judged = true) {
  const bool tie = iv.covers_zero();
  const bool better = higher_better ? iv.lo > 0 : iv.hi < 0;
  const std::string core = std::format("{:+.{}f} [{:+.{}f}, {:+.{}f}]", iv.mean, digits, iv.lo, digits, iv.hi, digits);
  if (!judged) return core + " (context)";
  return core + (tie ? " (tie)" : better ? " (better)" : " (worse)");
}

int compare(const Args& a) {
  const fs::path root = data_root(a);
  const std::string split = a.need("split"), A = a.need("a"), B = a.need("b");
  const fs::path results = a.str("results", "results/3d");
  fs::create_directories(results);
  const std::string tag = std::format("{}_{}_vs_{}", split, A, B);
  std::ofstream md(results / ("compare_" + tag + ".md"));
  const auto both = [&](const std::string& s) {
    std::println("{}", s);
    md << s << "\n";
  };
  both(std::format("# {} minus {} ({}; paired, 95% bootstrap, 10,000 resamples, seed 2026)\n", A, B, split));
  const auto ta = read_track(root / "eval" / split / "track" / (A + ".csv")), tb = read_track(root / "eval" / split / "track" / (B + ".csv"));
  std::ofstream csv(results / ("compare_" + tag + ".csv"));
  csv << "part,measure,horizon,mean_a,mean_b,diff,lo,hi\n";
  if (!ta.empty() && !tb.empty()) {
    both("Tracking from the true state (active PSNR, dB; paired over runs):\n");
    both("| measure | horizon | " + A + " | " + B + " | difference |");
    both("|---|---:|---:|---:|---|");
    const char* names[3] = {"field: temperature", "field: soot", "picture (4 cameras)"};
    for (int q = 0; q < 3; ++q) {
      for (int j = 0; j < 3; ++j) {
        std::vector<double> va, vb;
        for (const auto& [key, r] : ta) {
          const auto it = tb.find(key);
          if (it == tb.end()) continue;
          va.push_back(r.v[zu(j)][zu(q)]);
          vb.push_back(it->second.v[zu(j)][zu(q)]);
        }
        if (va.empty()) continue;
        const auto iv = nfx::metrics::paired_bootstrap(va, vb, 10000, 2026);
        double ma = 0, mb = 0;
        for (std::size_t i = 0; i < va.size(); ++i) {
          ma += va[i] / static_cast<double>(va.size());
          mb += vb[i] / static_cast<double>(vb.size());
        }
        const std::string hz = kHorizons[zu(j)] == 8 ? "8 frames" : kHorizons[zu(j)] == 30 ? "1 s" : "2 s";
        both(std::format("| {} | {} | {:.2f} | {:.2f} | {} |", names[q], hz, ma, mb, interval(iv, 2, true)));
        csv << std::format("track,{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n", names[q], kHorizons[zu(j)], ma, mb, iv.mean, iv.lo, iv.hi);
      }
    }
    both(std::format("\n({} runs)\n", ta.size()));
  }
  const auto sets = settings(split);
  std::vector<Endless> ea, eb, ef;
  bool have_floor = true;
  for (int s = 0; s < static_cast<int>(sets.size()); ++s) {
    const auto pooled = [&](const std::string& who) -> std::optional<Unit> {
      std::vector<Unit> us;
      for (int k = 0; k < endless_runs(split); ++k) {
        Unit u;
        if (!load_unit(root / "eval" / split / "endless" / who / std::format("s{}_k{}.unit", s, k), u)) return std::nullopt;
        us.push_back(std::move(u));
      }
      return pool_units(us);
    };
    const auto re = pooled("real"), ua = pooled(A), ub = pooled(B), uf = pooled("floor");
    if (!re || !ua || !ub) {
      ea.clear();
      break;
    }
    ea.push_back(distance(*re, *ua));
    eb.push_back(distance(*re, *ub));
    if (uf) ef.push_back(distance(*re, *uf));
    else have_floor = false;
  }
  if (!ea.empty()) {
    const bool show_floor = have_floor && !ef.empty();
    both(std::format("Endless statistics against real runs ({} settings, {} runs pooled per setting; lower is better but for PSNR and the motion ratio, "
                     "whose ideal is 1):\n",
                     ea.size(), endless_runs(split)));
    both("| statistic | " + A + " | " + B + (show_floor ? " | floor (other real runs)" : "") + " | difference |");
    both(std::string("|---|---:|---:|") + (show_floor ? "---:|" : "") + "---|");
    const auto stat = [&](const char* name, auto get, bool higher_better, int digits, bool judged = true) {
      std::vector<double> va, vb;
      double fm = 0;
      for (std::size_t i = 0; i < ea.size(); ++i) {
        va.push_back(get(ea[i]));
        vb.push_back(get(eb[i]));
        if (show_floor) fm += get(ef[i]) / static_cast<double>(ef.size());
      }
      const auto iv = nfx::metrics::paired_bootstrap(va, vb, 10000, 2026);
      double ma = 0, mb = 0;
      for (std::size_t i = 0; i < va.size(); ++i) {
        ma += va[i] / static_cast<double>(va.size());
        mb += vb[i] / static_cast<double>(vb.size());
      }
      const std::string flo = show_floor ? std::format(" | {:.{}f}", fm, digits) : "";
      both(std::format("| {} | {:.{}f} | {:.{}f}{} | {} |", name, ma, digits, mb, digits, flo, interval(iv, digits, higher_better, judged)));
      csv << std::format("endless,{},,{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n", name, ma, mb, iv.mean, iv.lo, iv.hi);
    };
    stat("3D spectrum distance", [](const Endless& e) { return e.spectrum; }, false, 3);
    stat("\\|ln motion ratio\\|", [](const Endless& e) { return e.motion; }, false, 3);
    stat("motion ratio", [](const Endless& e) { return e.motion_ratio; }, true, 3, false);
    stat("coverage distance", [](const Endless& e) { return e.coverage; }, false, 4);
    stat("emission distance", [](const Endless& e) { return e.emission; }, false, 4);
    stat("mean picture PSNR (dB)", [](const Endless& e) { return e.mean_psnr; }, true, 2);
    // each setting's statistics (context for the pooled means above; nothing here is judged)
    both("\nPer setting (" + A + " / " + B + "):\n");
    both("| setting (intensity, wind, turbulence) | motion ratio | coverage distance | emission distance | 3D spectrum distance | mean picture PSNR (dB) |");
    both("|---|---|---|---|---|---|");
    for (std::size_t i = 0; i < ea.size(); ++i) {
      const auto& c = sets[i];
      both(std::format("| {:.2f}, {:.2f}, {:.2f} | {:.2f} / {:.2f} | {:.4f} / {:.4f} | {:.4f} / {:.4f} | {:.3f} / {:.3f} | {:.2f} / {:.2f} |", c[0], c[1], c[2],
                       ea[i].motion_ratio, eb[i].motion_ratio, ea[i].coverage, eb[i].coverage, ea[i].emission, eb[i].emission, ea[i].spectrum,
                       eb[i].spectrum, ea[i].mean_psnr, eb[i].mean_psnr));
    }
  }
  return 0;
}

int freeze(const Args& a) {
  const fs::path model = a.need("model");
  const fs::path frozen = a.str("frozen", "results/3d/frozen.csv");
  fs::create_directories(frozen.parent_path());
  const bool fresh = !fs::exists(frozen);
  std::ofstream o(frozen, std::ios::app);
  if (fresh) o << "model,sha256,bytes\n";
  o << std::format("{},{},{}\n", model.filename().string(), sha_of(model), fs::file_size(model));
  std::println("froze {} as {}", model.string(), sha_of(model));
  return 0;
}

// --- videos ----------------------------------------------------------------------------------------------------------

struct Canvas {
  int w = 0, h = 0;
  std::vector<std::uint8_t> rgb;
  void fill(int x0, int y0, int fw, int fh, std::array<std::uint8_t, 3> c) {
    for (int y = std::max(0, y0); y < std::min(h, y0 + fh); ++y) {
      for (int x = std::max(0, x0); x < std::min(w, x0 + fw); ++x) {
        std::uint8_t* d = rgb.data() + (zu(y) * zu(w) + zu(x)) * 3;
        d[0] = c[0];
        d[1] = c[1];
        d[2] = c[2];
      }
    }
  }
  void text(std::string_view t, int x, int y, int scale, std::array<std::uint8_t, 3> c) {
    nfx::tools::draw_text(t, x + 1, y + 1, scale, [&](int px, int py) {  // a shadow, then the text
      if (px >= 0 && py >= 0 && px < w && py < h) std::fill_n(rgb.data() + (zu(py) * zu(w) + zu(px)) * 3, 3, std::uint8_t{0});
    });
    nfx::tools::draw_text(t, x, y, scale, [&](int px, int py) {
      if (px < 0 || py < 0 || px >= w || py >= h) return;
      std::uint8_t* d = rgb.data() + (zu(py) * zu(w) + zu(px)) * 3;
      d[0] = c[0];
      d[1] = c[1];
      d[2] = c[2];
    });
  }
};

struct View {
  std::array<float, 3> eye, target, up;
  float fov = 40.f;
  rd::Camera camera(float aspect) const { return rd::look_at(eye, target, up, fov, aspect); }
};

// The fire's light in a picture: the mean light it adds beyond its cover (for the glow on the ground).
float picture_glow(const std::vector<std::uint8_t>& rgba) {
  double e = 0;
  for (std::size_t i = 0; i < rgba.size(); i += 4) {
    const double a = rgba[i + 3];
    e += std::max(0.0, (rgba[i] + rgba[i + 1] + rgba[i + 2]) / 3.0 - a) / 255.0;
  }
  return static_cast<float>(e / static_cast<double>(rgba.size() / 4));
}

// The effect (premultiplied RGBA, w x h) over a dark sky and a ground grid (y = 0) lit by the fire, into the canvas at
// (x0, y0). The sky, the ground and its light are the tool's (procedural), not the model's.
void composite(Canvas& cv, int x0, int y0, int w, int h, const std::vector<std::uint8_t>& rgba, const View& v, float glow) {
  const auto norm = [](std::array<float, 3> a) {
    const float l = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    return std::array<float, 3>{a[0] / l, a[1] / l, a[2] / l};
  };
  const auto cross = [](std::array<float, 3> a, std::array<float, 3> b) {
    return std::array<float, 3>{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
  };
  const auto f = norm({v.target[0] - v.eye[0], v.target[1] - v.eye[1], v.target[2] - v.eye[2]});
  const auto r = norm(cross(f, v.up));
  const auto u = cross(r, f);
  const float th = std::tan(0.5f * v.fov * std::numbers::pi_v<float> / 180.f), aspect = fl(w) / fl(h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float nx = ((fl(x) + 0.5f) / fl(w) * 2.f - 1.f) * th * aspect, ny = (1.f - (fl(y) + 0.5f) / fl(h) * 2.f) * th;
      const auto d = norm({f[0] + nx * r[0] + ny * u[0], f[1] + nx * r[1] + ny * u[1], f[2] + nx * r[2] + ny * u[2]});
      float bg[3];
      const float sky = std::clamp(d[1] * 2.f, 0.f, 1.f);
      for (int c = 0; c < 3; ++c) bg[c] = (std::array{26.f, 26.f, 31.f}[zu(c)] * (1.f - sky) + std::array{7.f, 9.f, 17.f}[zu(c)] * sky);
      if (d[1] < -1e-4f && v.eye[1] > 0.f) {
        const float t = -v.eye[1] / d[1], px = v.eye[0] + t * d[0], pz = v.eye[2] + t * d[2];
        const float gx = std::abs(px / 0.25f - std::round(px / 0.25f)), gz = std::abs(pz / 0.25f - std::round(pz / 0.25f));
        const float wline = std::min(0.5f, 0.012f * (1.f + t));
        const float line = std::max(1.f - std::clamp(gx / wline, 0.f, 1.f), 1.f - std::clamp(gz / wline, 0.f, 1.f));
        const float fade = std::exp(-0.12f * t);
        const float rr = px * px + pz * pz, light = glow * 260.f / (1.f + 12.f * rr);
        const std::array<float, 3> ground{26.f, 24.f, 23.f}, warm{1.f, 0.55f, 0.25f};
        for (int c = 0; c < 3; ++c) {
          const float g = (ground[zu(c)] + 26.f * line) * fade + light * warm[zu(c)] * (0.7f + 0.3f * line);
          bg[c] = g + (bg[c] - g) * (1.f - fade) * 0.6f;
        }
      }
      const std::uint8_t* s = rgba.data() + (zu(y) * zu(w) + zu(x)) * 4;
      const float a = s[3] / 255.f;
      std::uint8_t* o = cv.rgb.data() + (zu(y0 + y) * zu(cv.w) + zu(x0 + x)) * 3;
      for (int c = 0; c < 3; ++c) o[c] = static_cast<std::uint8_t>(std::clamp(s[c] + bg[c] * (1.f - a), 0.f, 255.f));
    }
  }
}

struct Video {
  std::FILE* pipe = nullptr;
  Video(const std::string& out, int w, int h, int crf) {
    const std::string cmd = std::format("ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgb24 -s {}x{} -r 30 -i - -an -c:v libx264 -preset slow -crf {} "
                                        "-threads 2 -pix_fmt yuv420p -movflags +faststart \"{}\"",
                                        w, h, crf, out);
    pipe = popen(cmd.c_str(), "w");
    if (!pipe) throw std::runtime_error("cannot start ffmpeg");
  }
  void put(const Canvas& c) {
    if (std::fwrite(c.rgb.data(), 1, c.rgb.size(), pipe) != c.rgb.size()) throw std::runtime_error("ffmpeg stopped reading");
  }
  void close() {
    if (pclose(pipe) != 0) throw std::runtime_error("ffmpeg failed");
    pipe = nullptr;
  }
};

struct VolumeHandle {
  nvfx_volume_effect* e = nullptr;
  nvfx_volume* v = nullptr;
  VolumeHandle(const fs::path& f, int threads) {
    if (nvfx_volume_effect_load(f.string().c_str(), &e) != NVFX_OK) throw std::runtime_error("cannot load " + f.string());
    if (nvfx_volume_create(e, threads, &v) != NVFX_OK) throw std::runtime_error("cannot create a volume");
  }
  ~VolumeHandle() {
    nvfx_volume_free(v);
    nvfx_volume_effect_free(e);
  }
  void draw(const View& view, int w, int h, std::vector<std::uint8_t>& rgba) {
    rgba.resize(zu(w) * zu(h) * 4);
    nvfx_volume_camera cam{};
    const auto c = view.camera(fl(w) / fl(h));
    std::memcpy(cam.view, c.view.data(), sizeof(cam.view));
    std::memcpy(cam.proj, c.proj.data(), sizeof(cam.proj));
    if (nvfx_volume_render(v, &cam, w, h, rgba.data(), zu(w) * 4) != NVFX_OK) throw std::runtime_error("render failed");
  }
};

int video(const Args& a) {
  const fs::path root = data_root(a);
  const std::string kind = a.need("kind"), out = a.need("out");
  const fs::path model = a.has("model") ? fs::path(a.str("model")) : root / "models" / "chosen.nvfx3";
  const int W = a.i("width", 1280), H = a.i("height", 720), threads = a.i("threads", 3), crf = a.i("crf", 24);
  const float seconds = a.f("seconds", 12.f);
  const int frames = static_cast<int>(std::lround(seconds * 30.f));
  const auto cs = nfx::tools::parse_floats(a.str("controls", "0.7,0.5,0.6"));
  std::array<float, 3> c{cs.at(0), cs.at(1), cs.at(2)};
  const std::uint64_t seed = a.u64("seed", 9'900'000);
  const std::string title = a.str("title", "LEARNED 3D FIRE");  // the orbit's caption
  // the volume's shard length in seconds (the runtime's default 8; 0: one continuous rollout), and for --kind wind a
  // wind held instead of the sweep (study N2's figure of the edges)
  const float shard = a.f("shard", 8.f);
  const bool held = a.has("wind-fixed");
  const float wind_fixed = a.f("wind-fixed", 0.5f);
  Canvas cv{W, H, std::vector<std::uint8_t>(zu(W) * zu(H) * 3)};
  Video vid(out, W, H, crf);
  std::vector<std::uint8_t> pic, pic2;
  float glow1 = 0, glow2 = 0;
  const double t0 = now();
  if (kind == "orbit") {
    VolumeHandle vol(model, threads);
    nvfx_volume_set_controls(vol.v, c.data(), 3);
    nvfx_volume_set_seed(vol.v, seed);
    if (nvfx_volume_set_shard(vol.v, shard) != NVFX_OK) throw std::invalid_argument("bad --shard");
    for (int f = 0; f < frames; ++f) {
      const float t = static_cast<float>(f) / 30.f, az = 2.f * std::numbers::pi_v<float> * t / seconds;
      const View view{{2.1f * std::sin(az), 0.8f + 0.2f * std::sin(0.5f * az), 2.1f * std::cos(az)}, {0.f, 0.62f, 0.f}, {0.f, 1.f, 0.f}, 40.f};
      if (nvfx_volume_step_to(vol.v, t) != NVFX_OK) throw std::runtime_error("step failed");
      vol.draw(view, W, H, pic);
      glow1 = 0.7f * glow1 + 0.3f * picture_glow(pic);
      composite(cv, 0, 0, W, H, pic, view, glow1);
      cv.text(title, 24, 20, 3, {255, 255, 255});
      cv.text("A 16X24X16 LEARNED STATE DRIVES 64X96X64 SEEDED DETAIL. PLAYED AND RAY-MARCHED ON THE CPU.", 24, 52, 1, {190, 190, 196});
      cv.text(std::format("INTENSITY {:.2f}  WIND {:.2f}  TURBULENCE {:.2f}", c[0], c[1], c[2]), 24, H - 30, 2, {200, 200, 206});
      vid.put(cv);
      if (f % 60 == 0) std::println("  frame {} ({:.0f} s)", f, now() - t0);
    }
  } else if (kind == "wind") {
    VolumeHandle vol(model, threads);
    nvfx_volume_set_seed(vol.v, seed);
    if (nvfx_volume_set_shard(vol.v, shard) != NVFX_OK) throw std::invalid_argument("bad --shard");
    const int pw = W / 2;
    for (int f = 0; f < frames; ++f) {
      const float t = static_cast<float>(f) / 30.f, u = t / seconds;
      // wind: still, full towards -x, full towards +x, still again (or held)
      const float w = held ? wind_fixed
                           : u < 0.15f ? 0.5f : u < 0.4f ? 0.5f - 0.5f * (u - 0.15f) / 0.25f : u < 0.75f ? (u - 0.4f) / 0.35f : 1.f - 0.5f * (u - 0.75f) / 0.25f;
      c[1] = std::clamp(w, 0.f, 1.f);
      nvfx_volume_set_controls(vol.v, c.data(), 3);
      if (nvfx_volume_step_to(vol.v, t) != NVFX_OK) throw std::runtime_error("step failed");
      const View side{{0.f, 0.75f, 2.3f}, {0.f, 0.62f, 0.f}, {0.f, 1.f, 0.f}, 42.f};
      const View top{{0.f, 3.0f, 0.02f}, {0.f, 0.f, 0.f}, {0.f, 0.f, -1.f}, 40.f};
      vol.draw(side, pw, H, pic);
      glow1 = 0.7f * glow1 + 0.3f * picture_glow(pic);
      composite(cv, 0, 0, pw, H, pic, side, glow1);
      vol.draw(top, pw, H, pic2);
      glow2 = 0.7f * glow2 + 0.3f * picture_glow(pic2);
      composite(cv, pw, 0, pw, H, pic2, top, glow2);
      cv.fill(pw - 1, 0, 2, H, {0, 0, 0});
      cv.text("FROM THE SIDE", 24, 20, 2, {255, 255, 255});
      cv.text("FROM ABOVE", pw + 24, 20, 2, {255, 255, 255});
      cv.text(held ? a.str("caption", "LEARNED 3D FIRE: THE WIND HELD") : a.str("caption", "LEARNED 3D FIRE: THE WIND CONTROL SWEPT LIVE"), 24, H - 64, 2, {230, 230, 236});
      if (held) {
        const std::string ts = std::format("{:.1f} S", t);
        cv.text(ts, W - 24 - nfx::tools::text_width(ts, 3), 20, 3, {255, 225, 150});
      }
      const int bx = 24, by = H - 36, bw = W - 220;
      cv.fill(bx, by, bw, 16, {40, 40, 46});
      const int mid = bx + bw / 2, end = bx + static_cast<int>(std::lround(c[1] * static_cast<float>(bw)));
      cv.fill(std::min(mid, end), by + 2, std::max(2, std::abs(end - mid)), 12, {255, 170, 70});
      cv.fill(mid, by - 3, 1, 22, {150, 150, 156});
      cv.text(std::format("WIND {:.2f}", c[1]), bx + bw + 16, by, 2, {255, 225, 150});
      vid.put(cv);
      if (f % 60 == 0) std::println("  frame {} ({:.0f} s)", f, now() - t0);
    }
  } else if (kind == "real-vs-learned") {
    // a held-out run (a demo seed, at a setting no training run had): the simulation from an empty grid for 100 frames,
    // then the simulation and the learned model from that same state with the same noise seed, side by side
    r3::Model m = load_study_model(model);
    r3::quantise_like_storage(m);
    r3::Workspace ws(m);
    fast_conv(ws);
    nfx::Pool pool(threads);
    nfx::sim3d::Fluid sim(fine_params(c, seed, threads));
    for (int f = 0; f < kTrackWarm; ++f) sim.step_frame();
    r3::State st = r3::make_state(m);
    const std::vector<float> cv3(c.begin(), c.end());
    r3::start_from(m, coarse_of(m.h, sim), sim.time(), cv3, seed, st, ws, &pool, false);
    sim.fields(st.fine_t, st.fine_d);
    rd::Renderer rs(kWidth, kHeight, kWidth), rl(m.h.fx(), m.h.fy(), m.h.fz());
    const int pw = W / 2;
    std::vector<float> T(m.h.fine_cells()), D(m.h.fine_cells());
    for (int f = 0; f < frames; ++f) {
      if (f > 0) {
        sim.step_frame();
        r3::step(m, st, cv3, seed, ws, &pool);
      }
      const float t = static_cast<float>(f) / 30.f, az = 0.5f * std::numbers::pi_v<float> * t / seconds - 0.25f * std::numbers::pi_v<float>;
      const View view{{2.2f * std::sin(az), 0.85f, 2.2f * std::cos(az)}, {0.f, 0.62f, 0.f}, {0.f, 1.f, 0.f}, 44.f};
      const auto cam = view.camera(fl(pw) / fl(H));
      sim.fields(T, D);
      rs.prepare(T, D, &pool);
      pic.resize(zu(pw) * zu(H) * 4);
      rs.render(cam, pw, H, pic.data(), zu(pw) * 4, &pool);
      glow1 = 0.7f * glow1 + 0.3f * picture_glow(pic);
      composite(cv, 0, 0, pw, H, pic, view, glow1);
      rl.prepare(st.fine_t, st.fine_d, &pool);
      pic2.resize(pic.size());
      rl.render(cam, pw, H, pic2.data(), zu(pw) * 4, &pool);
      glow2 = 0.7f * glow2 + 0.3f * picture_glow(pic2);
      composite(cv, pw, 0, pw, H, pic2, view, glow2);
      cv.fill(pw - 1, 0, 2, H, {0, 0, 0});
      cv.text("SIMULATION", 24, 20, 3, {255, 255, 255});
      cv.text("64X96X64 CELLS", 24, 52, 2, {190, 190, 196});
      cv.text("LEARNED", pw + 24, 20, 3, {255, 255, 255});
      cv.text("16X24X16 LEARNED STATE, SEEDED DETAIL", pw + 24, 52, 2, {190, 190, 196});
      cv.text("SAME START STATE, SAME NOISE SEED. BOTH ARE CHAOTIC: THEY PART AFTER A SECOND OR TWO.", 24, H - 52, 2, {230, 230, 236});
      cv.text(std::format("HELD-OUT RUN, SEED {}; INTENSITY {:.2f} WIND {:.2f} TURBULENCE {:.2f}", seed, c[0], c[1], c[2]), 24, H - 26, 1, {170, 170, 176});
      const std::string ts = std::format("{:.1f} S", t);
      cv.text(ts, W - 24 - nfx::tools::text_width(ts, 3), 20, 3, {255, 225, 150});
      vid.put(cv);
      if (f % 60 == 0) std::println("  frame {} ({:.0f} s)", f, now() - t0);
    }
  } else {
    throw std::invalid_argument("--kind orbit | wind | real-vs-learned");
  }
  vid.close();
  std::println("wrote {} ({:.0f} s)", out, now() - t0);
  return 0;
}

// --- timing ----------------------------------------------------------------------------------------------------------

double median(std::vector<double> v) {
  std::ranges::sort(v);
  return v.empty() ? 0.0 : v[v.size() / 2];
}

int timing(const Args& a) {
  const fs::path root = data_root(a);
  const fs::path model = a.has("model") ? fs::path(a.str("model")) : root / "models" / "chosen.nvfx3";
  const int frames = a.i("frames", 60), reps = a.i("reps", 5);
  const fs::path out = a.str("out", "results/3d/timing.csv");
  fs::create_directories(out.parent_path());
  const bool fresh = !fs::exists(out);
  std::ofstream csv(out, std::ios::app);
  if (fresh) csv << "what,threads,ms_median,ms_min,ms_max,n,load\n";
  const auto load = [] {
    std::ifstream i("/proc/loadavg");
    double l = 0;
    i >> l;
    return l;
  };
  const auto row = [&](const std::string& what, int threads, std::vector<double> ms) {
    const double md = median(ms), mn = *std::ranges::min_element(ms), mx = *std::ranges::max_element(ms);
    const std::string r = std::format("{},{},{:.2f},{:.2f},{:.2f},{},{:.1f}", what, threads, md, mn, mx, ms.size(), load());
    csv << r << "\n";
    csv.flush();
    std::println("{}", r);
  };
  for (const float tf : nfx::tools::parse_floats(a.str("threads", "1,2,4"))) {
    const int threads = static_cast<int>(tf);
    VolumeHandle vol(model, threads);
    const std::array<float, 3> c{0.7f, 0.5f, 0.6f};
    nvfx_volume_set_controls(vol.v, c.data(), 3);
    nvfx_volume_set_seed(vol.v, 9'990'000);
    nvfx_volume_set_shard(vol.v, 0.f);
    std::vector<double> start, step, p256, p512;
    for (int r = 0; r < reps; ++r) {
      nvfx_volume_set_seed(vol.v, 9'990'000 + static_cast<std::uint64_t>(r));
      double t0 = now();
      nvfx_volume_step_to(vol.v, 0.0);  // a start: the start point and the fine fields' warm-up
      start.push_back(1000.0 * (now() - t0));
      for (int f = 1; f <= frames; ++f) {
        t0 = now();
        nvfx_volume_step_to(vol.v, f / 30.0);
        step.push_back(1000.0 * (now() - t0));
      }
      const View view{{1.6f, 0.9f, 1.4f}, {0.f, 0.62f, 0.f}, {0.f, 1.f, 0.f}, 42.f};
      std::vector<std::uint8_t> pic;
      for (int k = 0; k < 4; ++k) {
        t0 = now();
        vol.draw(view, 256, 256, pic);
        p256.push_back(1000.0 * (now() - t0));
        t0 = now();
        vol.draw(view, 512, 512, pic);
        p512.push_back(1000.0 * (now() - t0));
      }
    }
    row("start (warm-up of 24 frames)", threads, start);
    row("step", threads, step);
    row("picture 256x256", threads, p256);
    row("picture 512x512", threads, p512);
  }
  // where a step's time goes (the reference as the runtime runs it, on 1 thread): the noise inputs, the coarse step, the
  // detail layer, the renderer's preparation and one 256 x 256 picture
  {
    r3::Model m = load_study_model(model);
    r3::quantise_like_storage(m);
    r3::Workspace ws(m);
    fast_conv(ws);
    r3::State st = r3::make_state(m);
    const std::vector<float> c{0.7f, 0.5f, 0.6f};
    r3::start(m, 0, c, 9'990'200, st, ws, nullptr);
    rd::Renderer rend(m.h.fx(), m.h.fy(), m.h.fz());
    std::vector<double> tn, tc, td, tp, tr;
    std::vector<std::uint8_t> pic(256 * 256 * 4);
    for (int f = 0; f < frames; ++f) {
      double t0 = now();
      r3::coarse_noise(m, 9'990'200, st.time, ws, ws.noise);
      tn.push_back(1000.0 * (now() - t0));
      t0 = now();
      r3::coarse_step(m, st.coarse, ws.noise, c, st.pressure, ws.next, st.flow, ws);
      st.coarse.swap(ws.next);
      tc.push_back(1000.0 * (now() - t0));
      t0 = now();
      r3::detail_step(m, st, 9'990'200, c, ws);
      st.time += 1.f / m.fps;
      st.since_start += 1.f / m.fps;
      td.push_back(1000.0 * (now() - t0));
      t0 = now();
      rend.prepare(st.fine_t, st.fine_d);
      tp.push_back(1000.0 * (now() - t0));
      t0 = now();
      rend.render(study_camera(0, 1.f, 1.5f), 256, 256, pic.data(), 256 * 4);
      tr.push_back(1000.0 * (now() - t0));
    }
    row("stage: noise inputs (curl lattice, flicker)", 1, tn);
    row("stage: coarse step (convolutions, projection, advection)", 1, tc);
    row("stage: detail layer (64x96x64)", 1, td);
    row("stage: renderer preparation", 1, tp);
    row("stage: one 256x256 picture", 1, tr);
  }
  // the simulation itself, for scale: a frame at 64 x 96 x 64 and at the coarse 16 x 24 x 16
  for (const int w : {kWidth, kWidth / 4}) {
    nfx::sim3d::Fluid f(fine_params({0.7f, 0.5f, 0.6f}, 9'990'100, 1, w, w * 3 / 2));
    for (int k = 0; k < 30; ++k) f.step_frame();
    std::vector<double> ms;
    for (int k = 0; k < (w == kWidth ? 20 : 60); ++k) {
      const double t0 = now();
      f.step_frame();
      ms.push_back(1000.0 * (now() - t0));
    }
    row(std::format("simulation {}x{}x{}", w, w * 3 / 2, w), 1, ms);
  }
  return 0;
}

// --- the runaway's diagnosis (STUDY_3D.md §11) and study N2's edge check (§14) -------------------------------------

// The coarse heat in each side slab of the box (x = 0, x = max, z = 0, z = max) and in its top layer, in the whole box,
// and the hottest cell of the sides and top: f is [cell][channels], heat in channel `ch`.
struct SideHeat {
  std::array<double, 5> side{};  // x = 0, x = max, z = 0, z = max, top
  double all = 0, hot = 0;
};
SideHeat side_heat(const r3::Hyper& h, const float* f, int channels, int ch) {
  SideHeat s;
  for (int y = 0; y < h.ry; ++y) {
    for (int z = 0; z < h.rz; ++z) {
      for (int x = 0; x < h.rx; ++x) {
        const double t = f[((zu(y) * zu(h.rz) + zu(z)) * zu(h.rx) + zu(x)) * zu(channels) + zu(ch)];
        s.all += t;
        const bool on[5] = {x == 0, x == h.rx - 1, z == 0, z == h.rz - 1, y == h.ry - 1};
        bool any = false;
        for (int k = 0; k < 5; ++k) {
          if (on[k]) s.side[zu(k)] += t;
          any = any || on[k];
        }
        if (any) s.hot = std::max(s.hot, t);
      }
    }
  }
  return s;
}

// One frame of the coarse state alone, as r3::step() makes it (the detail layer never feeds back into it).
void coarse_only_step(const r3::Model& m, r3::State& st, const std::vector<float>& cv, std::uint64_t seed, r3::Workspace& ws, nfx::Pool& pool) {
  for (int k = 0; k < m.h.cond(); ++k) ws.cond[zu(k)] = zu(k) < cv.size() ? cv[zu(k)] : 0.5f;
  r3::coarse_noise(m, seed, st.time, ws, ws.noise, &pool);
  r3::coarse_step(m, st.coarse, ws.noise, ws.cond, st.pressure, ws.next, st.flow, ws, &pool);
  st.coarse.swap(ws.next);
  st.time += 1.f / m.fps;
  st.since_start += 1.f / m.fps;
}

// The model played in one rollout from the start point nearest the controls (its warm-up, then --seconds), or with
// --sim the simulation itself (64 x 96 x 64 from an empty grid, block averages, from 5 s on: a start point is 5 s into
// its run), every 15 frames. Study N's form (one wind, 8 s, rows to stdout) is the default; study N2's check takes
// --winds, --seconds 30 and --out (rows appended: label, wind, seconds, x0, xmax, z0, zmax, top, all, hottest).
int edge(const Args& a) {
  const fs::path root = data_root(a);
  const bool sim = a.flag("sim");
  const fs::path model = a.has("model") ? fs::path(a.str("model")) : root / "models" / "chosen.nvfx3";
  const auto cs = nfx::tools::parse_floats(a.str("controls", "0.7,0.05,0.5"));
  std::vector<float> winds = a.has("winds") ? nfx::tools::parse_floats(a.str("winds")) : std::vector<float>{cs.at(1)};
  const float seconds = a.f("seconds", 8.f);
  const int frames = static_cast<int>(std::lround(seconds * 30.f));
  const int threads = a.i("threads", 3);
  nfx::Pool pool(threads);
  // seeds: --seed for every wind (study N's default: 9,900,004), else study N2's: the base + 900 + the wind's index
  const std::uint64_t base = a.u64("seed-base", (sim ? kN2ValReal : kN2ValPlay) + kN2Edge);
  const std::string label = a.str("label", sim ? "simulation" : model.stem().string());
  std::ofstream csv;
  if (a.has("out")) {
    const fs::path out = a.str("out");
    fs::create_directories(out.parent_path());
    const bool fresh = !fs::exists(out);
    csv.open(out, std::ios::app);
    if (fresh) csv << "label,wind,seconds,x0,xmax,z0,zmax,top,all,hottest\n";
  }
  if (!a.has("out")) std::println("frame,heat x=0,heat x=max,heat z=0,heat z=max,heat in all,hottest side cell");
  const auto emit = [&](float w, int f, const SideHeat& s) {
    if (csv.is_open()) {
      csv << std::format("{},{:.2f},{:.2f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.3f},{:.4f}\n", label, w, f / 30.0, s.side[0], s.side[1], s.side[2], s.side[3],
                         s.side[4], s.all, s.hot);
      csv.flush();
    } else {  // study N's columns (no top)
      std::println("{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.4f}", f, s.side[0], s.side[1], s.side[2], s.side[3], s.all, s.hot);
    }
  };
  const r3::Hyper h = study_hyper();
  for (std::size_t wi = 0; wi < winds.size(); ++wi) {
    const std::array<float, 3> c{cs.at(0), winds[wi], cs.at(2)};
    const std::uint64_t seed = a.has("seed") ? a.u64("seed", 0) : g_n2 ? base + wi : 9'900'004;
    const double t0 = now();
    SideHeat last;
    if (sim) {
      auto fp = fine_params(c, seed, threads);
      if (!nfx::sim3d::parse_effect(a.str("effect", "fire"), fp.effect)) throw std::invalid_argument("--effect fire|smoke");
      nfx::sim3d::Fluid f(fp);
      for (int fr = 0; fr < kEndlessWarm; ++fr) f.step_frame();
      for (int fr = 0; fr <= frames; ++fr) {
        if (fr > 0) f.step_frame();
        if (fr % 15 != 0) continue;
        const auto cf = coarse_of(h, f);
        last = side_heat(h, cf.data(), r3::kPhys, 3);
        emit(c[1], fr, last);
      }
    } else {
      r3::Model m = load_study_model(model);
      r3::quantise_like_storage(m);
      if (a.has("edges")) m.edges = a.str("edges") == "clamp" ? r3::Edges::clamp : r3::Edges::ambient;
      r3::Workspace ws(m);
      fast_conv(ws);
      const std::vector<float> cv(c.begin(), c.end());
      r3::State st = r3::make_state(m);
      const r3::StartPoint& sp = m.starts[zu(nearest_start(m, c))];
      r3::start_from(m, sp.coarse, sp.time, cv, seed, st, ws, &pool, false);
      for (int fr = 0; fr < m.h.warmup; ++fr) coarse_only_step(m, st, cv, seed, ws, pool);  // as r3::start's warm-up
      for (int fr = 0; fr <= frames; ++fr) {
        if (fr > 0) coarse_only_step(m, st, cv, seed, ws, pool);
        if (fr % 15 != 0) continue;
        last = side_heat(m.h, st.coarse.data(), m.h.channels(), 3);
        emit(c[1], fr, last);
      }
    }
    std::println(stderr, "{} wind {:.2f} seed {}: at {:.0f} s the sides hold {:.3f} {:.3f} {:.3f} {:.3f}, the top {:.3f}, the box {:.2f} ({:.0f} s)", label, c[1], seed,
                 seconds, last.side[0], last.side[1], last.side[2], last.side[3], last.side[4], last.all, now() - t0);
  }
  return 0;
}

// Study N2's edge check (STUDY_3D.md §14) on the rows `edge --out` wrote: for each model, wind and side (x = 0, x = max,
// z = 0, z = max, the top), the side's heat runs away if its mean over 20 to 30 s is more than twice its mean over 5 to
// 15 s plus 0.5 (it grows), or its largest value over 5 to 30 s is more than twice the simulation's largest on that
// side over its own 5 to 30 s plus 1 (it holds far more than the teacher ever does). A model passes if nothing runs
// away at any wind.
struct EdgeVerdict {
  bool pass = true;
  std::vector<std::string> rows;  // markdown
};
std::map<std::string, EdgeVerdict> edge_verdicts(const fs::path& file) {
  struct Row {
    std::string label;
    float wind;
    double t;
    std::array<double, 5> s;
  };
  std::vector<Row> rows;
  std::ifstream i(file);
  if (!i) throw std::runtime_error("cannot open " + file.string());
  std::string line;
  std::getline(i, line);
  while (std::getline(i, line)) {
    std::vector<std::string> t;
    std::istringstream ss(line);
    std::string x;
    while (std::getline(ss, x, ',')) t.push_back(x);
    if (t.size() < 10) continue;
    rows.push_back({t[0], std::stof(t[1]), std::stod(t[2]), {std::stod(t[3]), std::stod(t[4]), std::stod(t[5]), std::stod(t[6]), std::stod(t[7])}});
  }
  const auto stats = [&](const std::string& label, float w, int k, double lo, double hi, bool mean) {
    double s = 0, mx = 0;
    int n = 0;
    for (const Row& r : rows) {
      if (r.label != label || std::abs(r.wind - w) > 1e-3f || r.t < lo - 1e-6 || r.t > hi + 1e-6) continue;
      s += r.s[zu(k)];
      mx = std::max(mx, r.s[zu(k)]);
      ++n;
    }
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    return mean ? s / n : mx;
  };
  std::map<std::string, EdgeVerdict> out;
  std::set<std::pair<std::string, float>> seen;
  const char* names[5] = {"x = 0", "x = max", "z = 0", "z = max", "top"};
  for (const Row& r : rows) {
    if (r.label == "simulation" || !seen.insert({r.label, r.wind}).second) continue;
    EdgeVerdict& v = out[r.label];
    std::string cells = std::format("| {} | {:.2f} |", r.label, r.wind);
    std::string worst;
    for (int k = 0; k < 5; ++k) {
      const double m1 = stats(r.label, r.wind, k, 5, 15, true), m2 = stats(r.label, r.wind, k, 20, 30, true), mx = stats(r.label, r.wind, k, 5, 30, false);
      const double sim = stats("simulation", r.wind, k, 5, 30, false);
      if (std::isnan(m1) || std::isnan(m2) || std::isnan(sim)) throw std::runtime_error(std::format("edge check: missing rows for {} at wind {:.2f}", r.label, r.wind));
      const bool grows = m2 > 2 * m1 + 0.5, holds = mx > 2 * sim + 1.0;
      if (grows || holds) {
        v.pass = false;
        worst += std::format("{}{} {}", worst.empty() ? "" : "; ", names[k], grows && holds ? "grows and holds too much" : grows ? "grows" : "holds too much");
      }
      cells += std::format(" {:.2f} / {:.2f} ({:.2f}; {:.2f}) |", m1, m2, mx, sim);
    }
    v.rows.push_back(cells + " " + (worst.empty() ? "no runaway" : "**runs away**: " + worst) + " |");
  }
  return out;
}

int edgecheck(const Args& a) {
  const auto v = edge_verdicts(a.need("csv"));
  std::println("Side heat, mean over 5 to 15 s / mean over 20 to 30 s (largest over 5 to 30 s; the simulation's largest):\n");
  std::println("| model | wind | x = 0 | x = max | z = 0 | z = max | top | verdict |");
  std::println("|---|---:|---|---|---|---|---|---|");
  for (const auto& [label, e] : v) {
    for (const auto& r : e.rows) std::println("{}", r);
  }
  std::println("");
  for (const auto& [label, e] : v) std::println("{}: {}", label, e.pass ? "passes the edge check" : "FAILS the edge check");
  return 0;
}

// A copy of a model file with other edges (study N2's reference "study N's model with the edge fix alone").
int reedge(const Args& a) {
  r3::Model m = load_study_model(a.need("model"));
  const std::string e = a.need("edges");
  if (e != "clamp" && e != "ambient") throw std::invalid_argument("--edges clamp|ambient");
  m.edges = e == "clamp" ? r3::Edges::clamp : r3::Edges::ambient;
  const fs::path out = a.need("out");
  if (auto r = r3::save_model(out, m); !r) throw std::runtime_error(r.error());
  std::println("wrote {} ({} edges)", out.string(), e);
  return 0;
}

// Study N2's choice among candidates on validation (STUDY_3D.md §14): (1) only candidates that pass the edge check;
// (2) tracking, non-inferiority: the mean picture PSNR at 1 s and the mean temperature field PSNR at 1 s each within
// 0.5 dB of the best eligible candidate's; (3) endless play, improvement: of those, the lowest mean rank over the five
// endless statistics (means over the settings; ties share ranks), ties broken by the picture PSNR at 1 s.
int choose(const Args& a) {
  const fs::path root = data_root(a);
  const std::string split = a.str("split", "val");
  std::vector<std::string> cands;
  {
    std::istringstream ss(a.need("cands"));
    std::string x;
    while (std::getline(ss, x, ',')) cands.push_back(x);
  }
  const auto ev = edge_verdicts(a.need("edge"));
  const auto sets = settings(split);
  struct Score {
    double pic = 0, temp = 0;
    std::array<double, 5> endless{};  // spectrum, |ln motion|, coverage, emission (lower better), mean picture (higher)
    bool edge = false;
    double rank = 0;
  };
  std::map<std::string, Score> sc;
  for (const auto& c : cands) {
    Score s;
    const auto it = ev.find(c);
    s.edge = it != ev.end() && it->second.pass;
    const auto tr = read_track(root / "eval" / split / "track" / (c + ".csv"));
    if (tr.empty()) throw std::runtime_error("no tracking rows for " + c);
    for (const auto& [key, r] : tr) {
      s.pic += r.v[1][2] / static_cast<double>(tr.size());
      s.temp += r.v[1][0] / static_cast<double>(tr.size());
    }
    for (int si = 0; si < static_cast<int>(sets.size()); ++si) {
      const auto pooled = [&](const std::string& who) {
        std::vector<Unit> us;
        for (int k = 0; k < endless_runs(split); ++k) {
          Unit u;
          if (!load_unit(root / "eval" / split / "endless" / who / std::format("s{}_k{}.unit", si, k), u)) throw std::runtime_error("missing plays of " + who);
          us.push_back(std::move(u));
        }
        return pool_units(us);
      };
      const Endless e = distance(pooled("real"), pooled(c));
      const double n = static_cast<double>(sets.size());
      s.endless[0] += e.spectrum / n;
      s.endless[1] += e.motion / n;
      s.endless[2] += e.coverage / n;
      s.endless[3] += e.emission / n;
      s.endless[4] += e.mean_psnr / n;
    }
    sc[c] = s;
  }
  std::println("| candidate | edge check | picture 1 s (dB) | temperature 1 s (dB) | 3D spectrum | \\|ln motion\\| | coverage | emission | mean picture (dB) |");
  std::println("|---|---|---:|---:|---:|---:|---:|---:|---:|");
  for (const auto& c : cands) {
    const Score& s = sc[c];
    std::println("| {} | {} | {:.2f} | {:.2f} | {:.3f} | {:.3f} | {:.4f} | {:.4f} | {:.2f} |", c, s.edge ? "passes" : "fails", s.pic, s.temp, s.endless[0], s.endless[1],
                 s.endless[2], s.endless[3], s.endless[4]);
  }
  std::vector<std::string> pool;
  for (const auto& c : cands) {
    if (sc[c].edge) pool.push_back(c);
  }
  if (pool.empty()) {
    std::println("\nNo candidate passes the edge check: the rule chooses among all of them, and the chosen one fails the edge check.");
    pool = cands;
  }
  double best_pic = -1e9, best_temp = -1e9;
  for (const auto& c : pool) {
    best_pic = std::max(best_pic, sc[c].pic);
    best_temp = std::max(best_temp, sc[c].temp);
  }
  std::vector<std::string> kept;
  for (const auto& c : pool) {
    if (sc[c].pic >= best_pic - 0.5 && sc[c].temp >= best_temp - 0.5) kept.push_back(c);
  }
  std::println("\nTracking within 0.5 dB of the best at 1 s (picture {:.2f}, temperature {:.2f} dB): {}", best_pic, best_temp, [&] {
    std::string t;
    for (const auto& c : kept) t += (t.empty() ? "" : ", ") + c;
    return t;
  }());
  for (int k = 0; k < 5; ++k) {
    for (const auto& c : kept) {
      double r = 1;
      int ties = 0;
      for (const auto& d : kept) {
        if (d == c) continue;
        const double x = sc[c].endless[zu(k)], y = sc[d].endless[zu(k)];
        const bool better = k == 4 ? y > x : y < x;
        if (better) r += 1;
        else if (x == y) ++ties;
      }
      sc[c].rank += (r + 0.5 * ties) / 5.0;
    }
  }
  std::string chosen;
  for (const auto& c : kept) {
    std::println("{}: mean rank over the endless statistics {:.2f}", c, sc[c].rank);
    if (chosen.empty() || sc[c].rank < sc[chosen].rank - 1e-12 || (std::abs(sc[c].rank - sc[chosen].rank) <= 1e-12 && sc[c].pic > sc[chosen].pic)) chosen = c;
  }
  std::println("\nchosen: {}", chosen);
  return 0;
}

// --- pilot and look ------------------------------------------------------------------------------------------------

void put_over(nfx::Image& img, const std::vector<std::uint8_t>& rgba, int w, int h, int x0, int y0) {
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::uint8_t* s = rgba.data() + (zu(y) * zu(w) + zu(x)) * 4;
      std::uint8_t* d = img.pixel(x0 + x, y0 + y);
      const float al = s[3] / 255.f;
      const float bg = 14.f + 10.f * static_cast<float>(y) / static_cast<float>(h);
      for (int c = 0; c < 3; ++c) d[c] = static_cast<std::uint8_t>(std::clamp(s[c] + bg * (1.f - al), 0.f, 255.f));
      d[3] = 255;
    }
  }
}

int pilot(const Args& a) {
  std::vector<int> widths;
  for (const float w : nfx::tools::parse_floats(a.str("widths", "48,64"))) widths.push_back(static_cast<int>(w));
  const float seconds = a.f("seconds", 12.f);
  const int seeds = a.i("seeds", 2), threads = a.i("threads", 2);
  const std::array<float, 3> c{a.f("intensity", 0.6f), a.f("wind", 0.5f), a.f("turbulence", 0.5f)};
  const fs::path root = data_root(a) / "pilot";
  fs::create_directories(root);
  std::ofstream out(root / "pilot.csv", std::ios::app);
  std::println("width,height,seed,ms_per_frame_wall,cpu_s_per_frame,threads,plume_top_2s,plume_top_6s,heat_centroid,puff_hz_first,puff_hz_second,puff_hz_all,peak_power_ratio");
  for (const int w : widths) {
    for (int k = 0; k < seeds; ++k) {
      const auto p = fine_params(c, 9'000'900 + static_cast<std::uint64_t>(k), threads, w, w * 3 / 2);  // pilot seeds (study N's range)
      nfx::sim3d::Fluid f(p);
      const int frames = static_cast<int>(seconds * p.fps);
      const std::size_t n = zu(f.nx()) * zu(f.ny()) * zu(f.nz());
      std::vector<float> T(n), D(n);
      std::vector<double> flame;  // the visible flame's volume per frame
      double top2 = 0, top6 = 0, centroid = 0;
      int centroid_n = 0;
      const double c0 = cpu_now(), w0 = now();
      for (int fr = 0; fr < frames; ++fr) {
        f.step_frame();
        f.fields(T, D);
        std::vector<double> slab(zu(f.ny()), 0.0), heat(zu(f.ny()), 0.0);
        for (int y = 0; y < f.ny(); ++y) {
          for (std::size_t i = zu(y) * zu(f.nz()) * zu(f.nx()); i < zu(y + 1) * zu(f.nz()) * zu(f.nx()); ++i) {
            slab[zu(y)] += D[i];
            heat[zu(y)] += T[i];
          }
        }
        const double mx = *std::ranges::max_element(slab);
        int top = 0;
        for (int y = 0; y < f.ny(); ++y) {
          if (slab[zu(y)] > 0.02 * mx) top = y;
        }
        const double top_rel = static_cast<double>(top + 1) / f.ny();
        if (fr == 2 * 30 - 1) top2 = top_rel;
        if (fr == 6 * 30 - 1) top6 = top_rel;
        double b = 0;
        for (const float t : T) b += t > 0.5f ? 1.0 : 0.0;
        flame.push_back(b);
        if (fr >= 90) {
          double hs = 0, hy = 0;
          for (int y = 0; y < f.ny(); ++y) {
            hs += heat[zu(y)];
            hy += heat[zu(y)] * (y + 0.5);
          }
          if (hs > 0) {
            centroid += hy / hs / f.ny();
            ++centroid_n;
          }
        }
      }
      const double cpu = (cpu_now() - c0) / frames, wall = (now() - w0) / frames;
      // the puffing frequency: the peak between 0.5 and 6 Hz of the flame volume's spectrum (Welch: 4 s Hann windows
      // overlapping by half, after a 3 s warm-up), on the first and second halves and on the whole (how steady it is)
      const auto peak = [&](int from, int to, double* ratio) {
        const int win = std::min(120, to - from);
        std::vector<double> power;
        for (double hz = 0.5; hz <= 6.0 + 1e-9; hz += 0.05) power.push_back(0.0);
        for (int s0 = from; s0 + win <= to; s0 += win / 2) {
          double mean = 0;
          for (int i = s0; i < s0 + win; ++i) mean += flame[zu(i)];
          mean /= win;
          std::size_t bin = 0;
          for (double hz = 0.5; hz <= 6.0 + 1e-9; hz += 0.05, ++bin) {
            double re = 0, im = 0;
            for (int i = s0; i < s0 + win; ++i) {
              const double x = flame[zu(i)] - mean, ph = 2.0 * std::numbers::pi * hz * (i - s0) / p.fps;
              const double wv = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * (i - s0) / (win - 1));
              re += wv * x * std::cos(ph);
              im += wv * x * std::sin(ph);
            }
            power[bin] += re * re + im * im;
          }
        }
        std::size_t best = 0;
        double total = 0;
        for (std::size_t j = 0; j < power.size(); ++j) {
          total += power[j];
          if (power[j] > power[best]) best = j;
        }
        if (ratio) *ratio = power[best] / (total / static_cast<double>(power.size()));
        return 0.5 + 0.05 * static_cast<double>(best);
      };
      const int start = 90, mid = (start + frames) / 2;
      double ratio = 0;
      const double f1 = peak(start, mid, nullptr), f2 = peak(mid, frames, nullptr), fa = peak(start, frames, &ratio);
      const std::string row = std::format("{},{},{},{:.1f},{:.3f},{},{:.3f},{:.3f},{:.3f},{:.2f},{:.2f},{:.2f},{:.1f}", f.nx(), f.ny(), p.seed, 1000.0 * wall, cpu,
                                          threads, top2, top6, centroid / std::max(1, centroid_n), f1, f2, fa, ratio);
      std::println("{}", row);
      std::fflush(stdout);
      out << row << "\n";
      out.flush();
    }
  }
  return 0;
}

int look(const Args& a) {
  const std::array<float, 3> c{a.f("intensity", 0.6f), a.f("wind", 0.5f), a.f("turbulence", 0.5f)};
  const int width = a.i("width", 64);
  auto p = fine_params(c, a.u64("seed", 9'000'950), a.i("threads", 2), width, width * 3 / 2);
  if (a.str("effect", "fire") == "smoke") p.effect = nfx::sim3d::Effect::smoke;
  const int frames = a.i("frames", 150), every = a.i("every", 30), px = a.i("size", 192);
  const fs::path out = a.has("out") ? fs::path(a.str("out")) : data_root(a) / "look";
  fs::create_directories(out);
  nfx::sim3d::Fluid f(p);
  rd::Renderer r(f.nx(), f.ny(), f.nz());
  nfx::Pool pool(p.threads);
  const std::size_t n = zu(f.nx()) * zu(f.ny()) * zu(f.nz());
  std::vector<float> T(n), D(n);
  std::vector<std::uint8_t> pic(zu(px) * zu(px) * 4);
  const int shots = frames / every;
  nfx::Image sheet;
  sheet.allocate(px * 4, px * shots);
  for (int fr = 1; fr <= frames; ++fr) {
    f.step_frame();
    if (fr % every) continue;
    f.fields(T, D);
    r.prepare(T, D, &pool);
    for (int k = 0; k < 4; ++k) {
      r.render(study_camera(k, 1.f, fl(f.ny()) / fl(f.nx())), px, px, pic.data(), zu(px) * 4, &pool);
      put_over(sheet, pic, px, px, k * px, (fr / every - 1) * px);
    }
  }
  const fs::path file = out / std::format("look_w{}_s{}.png", p.width, p.seed);
  if (auto e = nfx::write_png(file, sheet); !e) throw std::runtime_error(e.error());
  std::println("wrote {}", file.string());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a(argc, argv, {"floor", "sim"});
    if (a.positional().empty()) {
      std::println(stderr, "usage: nvfx_3d pilot|look|record|train|finish|truth|track|real|play|compare|freeze|video|timing [options] (see tools/nvfx_3d.cpp)");
      return 2;
    }
    const std::string cmd = a.positional()[0];
    g_max_settings = a.i("max-settings", g_max_settings);
    g_max_runs = a.i("max-runs", g_max_runs);
    g_n2 = a.str("study", "n") == "n2";
    int rc = 2;
    if (cmd == "pilot") rc = pilot(a);
    else if (cmd == "look") rc = look(a);
    else if (cmd == "record") rc = record(a);
    else if (cmd == "train") rc = train(a);
    else if (cmd == "finish") rc = finish(a);
    else if (cmd == "truth") rc = truth(a);
    else if (cmd == "track") rc = track(a);
    else if (cmd == "real") rc = real(a);
    else if (cmd == "play") rc = play(a);
    else if (cmd == "compare") rc = compare(a);
    else if (cmd == "freeze") rc = freeze(a);
    else if (cmd == "video") rc = video(a);
    else if (cmd == "timing") rc = timing(a);
    else if (cmd == "edge") rc = edge(a);
    else if (cmd == "edgecheck") rc = edgecheck(a);
    else if (cmd == "reedge") rc = reedge(a);
    else if (cmd == "choose") rc = choose(a);
    else std::println(stderr, "unknown command {}", cmd);
    a.warn_unused();
    return rc;
  } catch (const std::exception& e) {
    std::println(stderr, "error: {}", e.what());
    return 1;
  }
}
