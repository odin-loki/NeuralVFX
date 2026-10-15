// Study N: a small 3D demo (docs/STUDY_3D.md). The 3D simulator's pilot, the training runs, the 3D rollout model's
// training, validation and the test (once), demo videos and timing. Every step writes under --root (default
// $NEURALVFX_DATA/n_3d, else /root/nvfx-data/n_3d) and resumes from what it has written.
//
//   nvfx_3d pilot  --widths 48,64,80 --seconds 12 --seeds 2       cost per frame, plume rise and puffing at each grid
//   nvfx_3d look   --width 64 --frames 150 --out DIR               stills of the simulation from the four cameras
//   nvfx_3d record --runs 40 --frames 240                          training runs (seeds 9,000,000 + i)
//   nvfx_3d train  --name base [--iterations 1500 --finetune 600 --activity 300 --batch 8 --unroll 12]
//   nvfx_3d finish --ckpt base[_stage2] --out NAME                 a model file with start points
//   nvfx_3d truth  --split val|test                                 the tracking runs' true fields (simulated once)
//   nvfx_3d track  --split S --method NAME [--model F]              tracking from the true state: a model, coarse, hold
//   nvfx_3d real   --split S                                        endless statistics of real runs (and the floor)
//   nvfx_3d play   --split S --method NAME [--model F]              endless statistics of a model's plays, or coarse
//   nvfx_3d compare --split S --a NAME --b NAME                     paired differences with 95% bootstrap intervals
//   nvfx_3d freeze --model F                                        the model the test runs on (results/3d/frozen.csv)
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
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
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

fs::path data_root(const Args& a) {
  if (a.has("root")) return a.str("root");
  if (const char* e = std::getenv("NEURALVFX_DATA")) return fs::path(e) / "n_3d";
  return "/root/nvfx-data/n_3d";
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

std::vector<std::array<float, 3>> settings(const std::string& split) {
  if (split == "val") return {{0.35f, 0.30f, 0.40f}, {0.65f, 0.70f, 0.60f}, {0.50f, 0.15f, 0.75f}, {0.80f, 0.55f, 0.25f}};
  if (split == "test") {
    return {{0.30f, 0.60f, 0.55f}, {0.45f, 0.25f, 0.30f}, {0.55f, 0.80f, 0.70f}, {0.70f, 0.40f, 0.85f}, {0.85f, 0.65f, 0.45f}, {0.60f, 0.50f, 0.20f}};
  }
  throw std::invalid_argument("split must be val or test");
}
int track_runs(const std::string&) { return 3; }
int endless_runs(const std::string& split) { return split == "test" ? 3 : 2; }
std::uint64_t base_of(const std::string& split, const std::string& what) {
  const bool t = split == "test";
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
  const fs::path root = data_root(a), dir = root / "runs" / "train";
  fs::create_directories(dir);
  const int runs = a.i("runs", 40), frames = a.i("frames", 240), threads = a.i("threads", 3);
  const r3::Hyper h = study_hyper();
  for (int i = 0; i < runs; ++i) {
    const fs::path f = dir / std::format("run_{:04}.n3run", i);
    if (fs::exists(f)) continue;
    const std::uint64_t seed = kTrainBase + static_cast<std::uint64_t>(i);
    const auto c = train_controls(seed);
    const double t0 = now(), c0 = cpu_now();
    const r3::Run r = r3::record_run(fine_params(c, seed, threads), frames, h);
    if (auto e = r3::save_run(f, r); !e) throw std::runtime_error(e.error());
    log_line(root / "logs" / "record.log", std::format("run {} seed {} controls {:.3f} {:.3f} {:.3f}: {:.1f} s wall, {:.1f} s CPU", i, seed, c[0], c[1], c[2],
                                                       now() - t0, cpu_now() - c0));
  }
  return 0;
}

r3::Model fresh_model() {
  r3::Model m = r3::init_model(study_hyper(), 9'000'777);
  m.effect = "fire3d";
  m.fps = 30.f;
  m.noise = nfx::sim3d::forcing(nfx::sim3d::Effect::fire);
  m.control_names = {"intensity", "wind", "turbulence"};
  return m;
}

r3::TrainOptions train_options(const Args& a) {
  r3::TrainOptions o;
  o.iterations = a.i("iterations", 1500);
  o.finetune = a.i("finetune", 600);
  o.activity_stage = a.i("activity", 300);
  o.batch = a.i("batch", 8);
  o.max_unroll = a.i("unroll", 12);
  o.burn_max = a.i("burn", 24);
  o.lr = a.f("lr", 2e-3f);
  o.lr_finetune = a.f("lr-ft", 7e-4f);
  o.threads = a.i("threads", 3);
  o.seed = a.u64("seed", 1);
  o.log_every = a.i("log-every", 10);
  o.checkpoint_every = a.i("checkpoint-every", 20);
  return o;
}

int train(const Args& a) {
  const fs::path root = data_root(a);
  const std::string name = a.str("name", "base");
  const fs::path ck = root / "ckpt" / (name + ".ckpt"), log = root / "logs" / ("train_" + name + ".log");
  fs::create_directories(ck.parent_path());
  const auto runs = load_runs(root / "runs" / "train");
  if (runs.empty()) throw std::runtime_error("no training runs: nvfx_3d record first");
  r3::Model m = fresh_model();
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
  o.progress = [&](int it, int unroll, double loss, double seconds) {
    log_line(log, std::format("it {} unroll {} loss {:.6f} ({:.0f} s, {:.0f} CPU s)", it, unroll, loss, seconds, cpu_now() - c0));
  };
  r3::train_stepper(m, runs, o, s, [&](const r3::Model& mm, const r3::TrainState& ss) {
    if (auto e = r3::save_checkpoint(ck, mm, ss); !e) throw std::runtime_error(e.error());
    if (ss.iteration == stage2_end) {  // candidate A of the validation: the stepper after stages 1 and 2
      if (auto e = r3::save_checkpoint(root / "ckpt" / (name + "_stage2.ckpt"), mm, ss); !e) throw std::runtime_error(e.error());
    }
  });
  log_line(log, std::format("done at iteration {}", s.iteration));
  return 0;
}

int finish(const Args& a) {
  const fs::path root = data_root(a);
  const std::string ckname = a.need("ckpt"), out = a.need("out");
  const auto runs = load_runs(root / "runs" / "train");
  r3::Model m = fresh_model();
  r3::set_normalisation(m, runs);
  r3::TrainState s;
  if (auto e = r3::load_checkpoint(root / "ckpt" / (ckname + ".ckpt"), m, s); !e) throw std::runtime_error(e.error());
  r3::choose_starts(m, runs, a.i("starts", 8), a.i("frame", 150));
  m.detail.swirl = a.f("swirl", m.detail.swirl);
  m.detail.contrast = a.f("contrast", m.detail.contrast);
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

std::string interval(const nfx::metrics::Interval& iv, int digits, bool higher_better) {
  const bool tie = iv.covers_zero();
  const bool better = higher_better ? iv.lo > 0 : iv.hi < 0;
  const std::string core = std::format("{:+.{}f} [{:+.{}f}, {:+.{}f}]", iv.mean, digits, iv.lo, digits, iv.hi, digits);
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
    const auto stat = [&](const char* name, auto get, bool higher_better, int digits) {
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
      both(std::format("| {} | {:.{}f} | {:.{}f}{} | {} |", name, ma, digits, mb, digits, flo, interval(iv, digits, higher_better)));
      csv << std::format("endless,{},,{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n", name, ma, mb, iv.mean, iv.lo, iv.hi);
    };
    stat("3D spectrum distance", [](const Endless& e) { return e.spectrum; }, false, 3);
    stat("|ln motion ratio|", [](const Endless& e) { return e.motion; }, false, 3);
    stat("motion ratio", [](const Endless& e) { return e.motion_ratio; }, true, 3);
    stat("coverage distance", [](const Endless& e) { return e.coverage; }, false, 4);
    stat("emission distance", [](const Endless& e) { return e.emission; }, false, 4);
    stat("mean picture PSNR (dB)", [](const Endless& e) { return e.mean_psnr; }, true, 2);
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
    const Args a(argc, argv, {"floor"});
    if (a.positional().empty()) {
      std::println(stderr, "usage: nvfx_3d pilot|look|record|train|finish|truth|track|real|play|compare|freeze [options] (see tools/nvfx_3d.cpp)");
      return 2;
    }
    const std::string cmd = a.positional()[0];
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
    else std::println(stderr, "unknown command {}", cmd);
    a.warn_unused();
    return rc;
  } catch (const std::exception& e) {
    std::println(stderr, "error: {}", e.what());
    return 1;
  }
}
