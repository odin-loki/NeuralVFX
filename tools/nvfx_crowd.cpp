// nvfx_crowd: a battlefield of rollout effects through the effect manager (docs/MANAGER.md §7).
//
//   nvfx_crowd [--instances 160] [--seconds 20] [--fps 60] [--budget 4] [--threads 1] [--cores 2,3]
//              [--interp default|hold|crossfade|motion] [--proxies 1] [--max-size 256] [--rates 96,48,24]
//              [--no-share] [--unlimited] [--calibrate] [--max-sims 16] [--proxy-sims 8] [--seed 1] [--root DATA]
//              [--precision float|int8] [--scene crowd|close] [--slots S32,S64,S128,S256] [--half-px PX] [--half-share F]
//              [--half-upscale bilinear|off]
//              [--video FILE.mp4] [--sheet DIR] [--csv FILE] [--summary FILE] [--label NAME]
//   nvfx_crowd --ab RUNS.csv [--out SUMMARY.csv]
//
// The scene: fires, smoke columns, steam vents and rows of torches (identical copies, offset in time) spread over a
// field of 80 x 160 m; a camera 2 m up flies through it at 2 m/s, yawing +-25 degrees, so effects come near, shrink,
// leave the screen and come back. Explosions go off ahead of the camera at 1.5 a second and are stopped when they
// finish; every half second a fire dies and another starts somewhere ahead. Each tick: the camera, the game's calls,
// nvfx_manager_tick (timed: wall clock, and the process's CPU time, which counts every thread), the pictures.
// --scene close (docs/MANAGER.md §7.2): the same effects large on screen, where drawing at half size pays: 32 fires, smoke
// columns and steam vents (53, 29 and 18%) between 2 and 40 m ahead and 7 m either side, a camera 2 m up walking at
// 0.6 m/s and yawing +-20 degrees every 10 s, an explosion 8 to 20 m ahead every 1.5 s, a fire that dies and starts
// elsewhere every 2 s; most effects are 100 to 600 px on screen.
// --half-px, --half-share, --half-upscale: drawn at half size where it pays (docs/MANAGER.md §2.10; default the manager's).
// --unlimited: no budget (every simulation at the level its size asks for); with --rates 0,0,0 every simulation
// updates at 30 Hz and with --no-share no instance shares: what playing every copy with nvfx.h alone costs.
// --video composites every other tick (30 fps) at 1280 x 720 and pipes it to ffmpeg; --sheet writes the composited
// view and a sheet of every visible instance's picture (largest first) at a quarter, half and three quarters of the run.
// --ab: the A/B summary of runs written with --summary whose labels are CONFIG.ARM (tools/manager/time_halfsize.sh): per
// configuration and arm the median over the rounds (the k-th run of a label is round k) with a 95% bootstrap interval of
// the median, and each arm against the arm `off`, round by round (the ratio and the difference of the CPU per tick).
#include "args.hpp"

#include <neuralfx/image_io.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/nvfx_manager.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <numbers>
#include <print>
#include <random>
#include <sched.h>
#include <sstream>
#include <string>
#include <time.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::size_t sz(int v) { return static_cast<std::size_t>(v); }

double cpu_ms() {
  timespec t{};
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
  return static_cast<double>(t.tv_sec) * 1e3 + static_cast<double>(t.tv_nsec) * 1e-6;
}

double peak_rss_mb() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmHWM:", 0) == 0) return std::stod(line.substr(6)) / 1024.0;
  }
  return 0.0;
}

enum Kind { kFire = 0, kSmoke, kSteam, kExplosion, kKinds };
constexpr std::array<const char*, kKinds> kNames{"fire", "smoke", "steam", "explosion"};
constexpr std::array<float, kKinds> kWorld{2.5f, 6.f, 2.f, 6.f};  // metres

struct Thing {
  nvfx_handle h = 0;
  nvfx_instance* in = nullptr;  // --unmanaged: its own instance
  float offset = 0.f;
  int serial = 0;
  int kind = 0;
  bool torch = false;
  std::array<float, 3> pos{};
  float size = 1.f;
  double born = 0.0;
};

struct Camera {
  std::array<float, 3> pos{}, fwd{}, up{0.f, 1.f, 0.f}, right{};
};
// The crowd's fly-through (2 m/s, +-25 degrees every 12 s), or the close scene's walk (0.6 m/s, +-20 degrees every 10 s).
Camera camera_at(double t, bool close = false) {
  Camera c;
  const float period = close ? 10.f : 12.f, swing = close ? 20.f : 25.f, speed = close ? 0.6f : 2.f;
  const float yaw = swing * std::numbers::pi_v<float> / 180.f * std::sin(2.f * std::numbers::pi_v<float> * static_cast<float>(t) / period);
  c.pos = {0.f, 2.f, -5.f + speed * static_cast<float>(t)};
  c.fwd = {std::sin(yaw), -0.05f, std::cos(yaw)};
  const float n = std::sqrt(c.fwd[0] * c.fwd[0] + c.fwd[1] * c.fwd[1] + c.fwd[2] * c.fwd[2]);
  for (float& v : c.fwd) v /= n;
  c.right = {c.fwd[2], 0.f, -c.fwd[0]};
  const float r = std::sqrt(c.right[0] * c.right[0] + c.right[2] * c.right[2]);
  c.right[0] /= r;
  c.right[2] /= r;
  c.up = {c.right[1] * c.fwd[2] - c.right[2] * c.fwd[1], c.right[2] * c.fwd[0] - c.right[0] * c.fwd[2], c.right[0] * c.fwd[1] - c.right[1] * c.fwd[0]};
  return c;
}

constexpr float kFov = 1.0472f;  // 60 degrees vertically
constexpr int kW = 1280, kH = 720;

// The scene's sprites composited over a dusk sky and dark ground, far to near, premultiplied.
void composite(const nvfx_manager* m, const std::vector<Thing>& things, const Camera& cam, nfx::Image& img) {
  const float focal = static_cast<float>(kH) / (2.f * std::tan(0.5f * kFov));
  for (int y = 0; y < kH; ++y) {
    const float t = static_cast<float>(y) / static_cast<float>(kH);
    const std::array<int, 3> c = t < 0.5f ? std::array<int, 3>{static_cast<int>(20 + 30 * t), static_cast<int>(24 + 30 * t), static_cast<int>(40 + 40 * t)}
                                          : std::array<int, 3>{static_cast<int>(26 - 12 * (t - 0.5f)), static_cast<int>(24 - 12 * (t - 0.5f)), static_cast<int>(22 - 10 * (t - 0.5f))};
    for (int x = 0; x < kW; ++x) {
      std::uint8_t* p = img.pixel(x, y);
      p[0] = static_cast<std::uint8_t>(c[0]);
      p[1] = static_cast<std::uint8_t>(c[1]);
      p[2] = static_cast<std::uint8_t>(c[2]);
      p[3] = 255;
    }
  }
  struct Draw {
    float d, sx, sy, px;
    const Thing* t;
  };
  std::vector<Draw> list;
  for (const Thing& th : things) {
    const std::array<float, 3> r{th.pos[0] - cam.pos[0], th.pos[1] - cam.pos[1], th.pos[2] - cam.pos[2]};
    const float d = r[0] * cam.fwd[0] + r[1] * cam.fwd[1] + r[2] * cam.fwd[2];
    if (d < 0.3f) continue;
    const float sx = 0.5f * kW + (r[0] * cam.right[0] + r[1] * cam.right[1] + r[2] * cam.right[2]) * focal / d;
    const float sy = 0.5f * kH - (r[0] * cam.up[0] + r[1] * cam.up[1] + r[2] * cam.up[2]) * focal / d;
    list.push_back({d, sx, sy, th.size * focal / d, &th});
  }
  std::ranges::sort(list, [](const Draw& a, const Draw& b) { return a.d > b.d; });
  for (const Draw& dr : list) {
    nvfx_manager_picture p;
    if (nvfx_manager_get_picture(m, dr.t->h, &p) != NVFX_OK || !p.rgba || !p.visible) continue;
    const int x0 = static_cast<int>(std::floor(dr.sx - 0.5f * dr.px)), y0 = static_cast<int>(std::floor(dr.sy - 0.5f * dr.px));
    const int n = std::max(1, static_cast<int>(dr.px));
    const float k = static_cast<float>(p.size) / static_cast<float>(n);
    for (int j = std::max(0, -y0); j < n && y0 + j < kH; ++j) {
      const float fy = std::clamp((static_cast<float>(j) + 0.5f) * k - 0.5f, 0.f, static_cast<float>(p.size - 1));
      const int y = std::min(static_cast<int>(fy), p.size - 2);
      const float wy = fy - static_cast<float>(y);
      for (int i = std::max(0, -x0); i < n && x0 + i < kW; ++i) {  // bilinear, as a GPU draws a sprite
        const float fx = std::clamp((static_cast<float>(i) + 0.5f) * k - 0.5f, 0.f, static_cast<float>(p.size - 1));
        const int x = std::min(static_cast<int>(fx), p.size - 2);
        const float wx = fx - static_cast<float>(x);
        const std::uint8_t* q = p.rgba + sz(y) * p.stride + sz(x) * 4;
        const std::uint8_t* r = q + p.stride;
        float s[4];
        for (int c = 0; c < 4; ++c) {
          const float top = q[c] + wx * static_cast<float>(q[4 + c] - q[c]), bot = r[c] + wx * static_cast<float>(r[4 + c] - r[c]);
          s[c] = top + wy * (bot - top);
        }
        if (s[3] < 0.5f && s[0] < 0.5f && s[1] < 0.5f && s[2] < 0.5f) continue;
        std::uint8_t* d = img.pixel(x0 + i, y0 + j);
        for (int c = 0; c < 3; ++c) d[c] = static_cast<std::uint8_t>(std::min(255.f, s[c] + static_cast<float>(d[c]) * (255.f - s[3]) / 255.f));
      }
    }
  }
}

// Every visible instance's picture, largest on screen first, each in a 64-pixel cell (nearest), over black.
nfx::Image instance_sheet(const nvfx_manager* m, const std::vector<Thing>& things) {
  struct Cell {
    float px;
    nvfx_manager_picture p;
  };
  std::vector<Cell> cells;
  for (const Thing& th : things) {
    nvfx_manager_picture p;
    if (nvfx_manager_get_picture(m, th.h, &p) == NVFX_OK && p.rgba && p.visible) cells.push_back({p.screen_px, p});
  }
  std::ranges::sort(cells, [](const Cell& a, const Cell& b) { return a.px > b.px; });
  const int cols = 16, cell = 64, rows = std::max(1, static_cast<int>((cells.size() + cols - 1) / cols));
  nfx::Image img;
  img.allocate(cols * cell, rows * cell);
  for (std::size_t k = 0; k < cells.size(); ++k) {
    const auto& p = cells[k].p;
    const int cx = static_cast<int>(k % cols) * cell, cy = static_cast<int>(k / cols) * cell;
    for (int j = 0; j < cell; ++j) {
      for (int i = 0; i < cell; ++i) {
        const std::uint8_t* s = p.rgba + sz(j * p.size / cell) * p.stride + sz(i * p.size / cell) * 4;
        std::uint8_t* d = img.pixel(cx + i, cy + j);
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
        d[3] = 255;
      }
    }
    // the level as a bar along the cell's bottom: width = size / 256, colour = rate (green 30 Hz ... red 7.5 Hz)
    const int w = cell * p.lod_size / 256;
    const int rate = static_cast<int>(std::lround(30.f / std::max(1.f, p.lod_rate_hz)));
    const std::array<std::uint8_t, 3> col = (p.flags & NVFX_PICTURE_PROXY) ? std::array<std::uint8_t, 3>{80, 140, 255}
                                            : rate <= 1                     ? std::array<std::uint8_t, 3>{60, 220, 60}
                                            : rate == 2                     ? std::array<std::uint8_t, 3>{200, 220, 60}
                                            : rate == 3                     ? std::array<std::uint8_t, 3>{240, 150, 40}
                                                                            : std::array<std::uint8_t, 3>{240, 50, 40};
    for (int j = cell - 3; j < cell; ++j) {
      for (int i = 0; i < w; ++i) {
        std::uint8_t* d = img.pixel(cx + i, cy + j);
        d[0] = col[0];
        d[1] = col[1];
        d[2] = col[2];
      }
    }
  }
  return img;
}

double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::ranges::sort(v);
  return v[std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size() - 1) + 0.5))];
}

double median_of(std::vector<double> v) {
  if (v.empty()) return std::nan("");
  std::ranges::sort(v);
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// The median and its 95% percentile bootstrap interval (the rounds resampled with replacement, 10,000 times, seed 2026).
struct MedianIv {
  double median = 0, lo = 0, hi = 0;
};
MedianIv median_interval(const std::vector<double>& v) {
  MedianIv r{median_of(v), std::nan(""), std::nan("")};
  if (v.size() < 2) return r;
  std::mt19937_64 rng(2026);
  std::uniform_int_distribution<std::size_t> pick(0, v.size() - 1);
  std::vector<double> meds(10000), x(v.size());
  for (double& m : meds) {
    for (double& e : x) e = v[pick(rng)];
    m = median_of(x);
  }
  std::ranges::sort(meds);
  r.lo = meds[249];
  r.hi = meds[9749];
  return r;
}

// --ab: the A/B summary of runs whose labels are CONFIG.ARM (tools/manager/time_halfsize.sh).
int ab_summary(const fs::path& in, const fs::path& out) {
  std::ifstream f(in);
  if (!f) throw std::runtime_error("cannot read " + in.string());
  std::string line;
  std::getline(f, line);
  std::vector<std::string> head;
  {
    std::stringstream ss(line);
    for (std::string c; std::getline(ss, c, ',');) head.push_back(c);
  }
  const auto col = [&](const std::string& name) {
    const auto it = std::ranges::find(head, name);
    if (it == head.end()) throw std::runtime_error("no column " + name + " in " + in.string());
    return static_cast<std::size_t>(it - head.begin());
  };
  const std::vector<std::string> figures{"cpu_mean", "cpu_median", "cpu_p99", "wall_median", "plan_mean", "sims", "proxied", "half_sims"};
  // config -> arm -> figure -> one value per round (in run order)
  std::map<std::string, std::map<std::string, std::map<std::string, std::vector<double>>>> v;
  std::vector<std::string> configs;
  while (std::getline(f, line)) {
    std::vector<std::string> c;
    std::stringstream ss(line);
    for (std::string x; std::getline(ss, x, ',');) c.push_back(x);
    if (c.size() != head.size()) continue;
    const std::string& label = c[0];
    const std::size_t dot = label.rfind('.');
    if (dot == std::string::npos) continue;
    const std::string config = label.substr(0, dot), arm = label.substr(dot + 1);
    if (!v.count(config)) configs.push_back(config);
    for (const auto& fig : figures) v[config][arm][fig].push_back(std::stod(c[col(fig)]));
  }
  std::ofstream o(out);
  o << "config,arm,figure,rounds,median,lo,hi,vs_off,ratio_median,ratio_lo,ratio_hi,diff_median,diff_lo,diff_hi\n";
  for (const auto& config : configs) {
    const auto& arms = v[config];
    std::println("{}:", config);
    for (const auto& [arm, figs] : arms) {
      std::string text = std::format("  {:<4}", arm);
      for (const auto& fig : figures) {
        const auto& x = figs.at(fig);
        const MedianIv m = median_interval(x);
        o << std::format("{},{},{},{},{:.4f},{:.4f},{:.4f},,,,,,,\n", config, arm, fig, x.size(), m.median, m.lo, m.hi);
        if (fig == "cpu_mean" || fig == "cpu_p99" || fig == "plan_mean" || fig == "sims" || fig == "proxied" || fig == "half_sims") {
          text += std::format(" {} {:.2f} [{:.2f}, {:.2f}]", fig, m.median, m.lo, m.hi);
        }
      }
      std::println("{}", text);
    }
    if (!arms.count("off")) continue;
    for (const auto& [arm, figs] : arms) {
      if (arm == "off") continue;
      for (const std::string fig : {"cpu_mean", "cpu_median", "cpu_p99"}) {
        const auto& x = figs.at(fig);
        const auto& y = arms.at("off").at(fig);
        std::vector<double> ratio, diff;
        for (std::size_t r = 0; r < std::min(x.size(), y.size()); ++r) {
          ratio.push_back(x[r] / y[r]);
          diff.push_back(x[r] - y[r]);
        }
        const MedianIv q = median_interval(ratio), dd = median_interval(diff);
        o << std::format("{},{},{},{},,,,off,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n", config, arm, fig, ratio.size(), q.median, q.lo, q.hi, dd.median, dd.lo,
                         dd.hi);
        std::println("  {} / off, {}: ratio {:.3f} [{:.3f}, {:.3f}], difference {:+.3f} ms [{:+.3f}, {:+.3f}] ({} rounds)", arm, fig, q.median, q.lo, q.hi, dd.median, dd.lo,
                     dd.hi, ratio.size());
      }
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  const nfx::tools::Args a(argc, argv, {"help", "no-share", "unlimited", "calibrate"});
  // --unmanaged SIZE: no manager; every instance its own nvfx_instance at SIZE px, updated at 30 Hz (staggered over the
  // ticks) while on screen, as an engine would play them with nvfx.h alone
  const int unmanaged = a.i("unmanaged", 0);
  if (a.flag("help")) {
    std::println("nvfx_crowd [--instances 160] [--seconds 20] [--fps 60] [--budget 4] [--threads 1] [--cores 2,3] [--interp default] [--proxies 1] "
                 "[--max-size 256] [--rates 96,48,24] [--no-share] [--unlimited] [--calibrate] [--precision float|int8] [--seed 1] [--scene crowd|close] "
                 "[--half-px PX] [--half-share F] [--half-upscale bilinear|off] [--video F.mp4] [--sheet DIR] [--csv F] [--summary F] [--label L]\n"
                 "nvfx_crowd --ab RUNS.csv [--out SUMMARY.csv]");
    return 0;
  }
  if (a.has("ab")) return ab_summary(a.str("ab"), a.str("out", "ab_summary.csv"));
  const std::string scene = a.str("scene", "crowd");
  if (scene != "crowd" && scene != "close") throw std::invalid_argument("--scene crowd or close");
  const bool close = scene == "close";
  const int count = a.i("instances", close ? 32 : 160), fps = a.i("fps", 60), threads = a.i("threads", 1);
  const double seconds = a.f("seconds", 20.f);
  const fs::path root = a.str("root", std::getenv("NEURALVFX_DATA") ? std::getenv("NEURALVFX_DATA") : "/root/nvfx-data");
  if (a.has("cores")) {
    cpu_set_t set;
    CPU_ZERO(&set);
    const std::string list = a.str("cores");
    for (std::size_t p = 0; p < list.size();) {
      const std::size_t q = std::min(list.find(',', p), list.size());
      CPU_SET(std::stoi(list.substr(p, q - p)), &set);
      p = q + 1;
    }
    if (sched_setaffinity(0, sizeof(set), &set) != 0) std::println(stderr, "nvfx_crowd: could not pin to cores {}", list);
  }
  const std::array<fs::path, kKinds> files{root / "v2/models/fire.nvfx", root / "v2/models/smoke.nvfx", root / "fx/models/d/steam.nvfx", root / "v2/models/explosion.nvfx"};
  std::array<nvfx_effect*, kKinds> fx{};
  for (int k = 0; k < kKinds; ++k) {
    if (nvfx_effect_load(files[sz(k)].c_str(), &fx[sz(k)]) != NVFX_OK) throw std::runtime_error("cannot load " + files[sz(k)].string());
  }
  nvfx_manager_desc d;
  nvfx_manager_desc_init(&d);
  d.max_instances = std::max(256, count + 64);
  d.threads = threads;
  d.budget_ms = a.flag("unlimited") ? 1e9f : a.f("budget", 4.f);
  d.max_size = a.i("max-size", 256);
  d.proxies = a.i("proxies", 1);
  d.seed = a.u64("seed", 1);
  // drawn at half size where it pays (docs/MANAGER.md §2.10): the manager's default unless given
  if (a.has("half-px")) d.half_px = a.f("half-px", 0.f);
  if (a.has("half-share")) d.half_share = a.f("half-share", 0.f);
  const std::string hu = a.str("half-upscale", "bilinear");
  if (hu != "bilinear" && hu != "off") throw std::invalid_argument("--half-upscale bilinear or off");
  d.half_upscale = hu == "off" ? NVFX_UPSCALE_OFF : NVFX_UPSCALE_BILINEAR;
  const std::string im = a.str("interp", "default");
  d.interp = im == "hold" ? NVFX_INTERP_HOLD : im == "crossfade" ? NVFX_INTERP_CROSSFADE : im == "motion" ? NVFX_INTERP_MOTION : NVFX_INTERP_DEFAULT;
  if (a.has("rates")) {
    const std::string r = a.str("rates");
    std::size_t p = 0;
    for (int k = 0; k < 3; ++k) {
      const std::size_t q = std::min(r.find(',', p), r.size());
      d.rate_px[k] = std::stof(r.substr(p, q - p));
      p = q + 1;
    }
  }
  // --precision: the simulations' (and --unmanaged instances') precision (study K, docs/REPORT.md §13); default the runtime's.
  const std::string pn = a.str("precision", "");
  if (!pn.empty() && pn != "float" && pn != "int8") throw std::invalid_argument("--precision float or int8");
  const nvfx_precision precision = pn.empty() ? NVFX_PRECISION_DEFAULT : pn == "int8" ? NVFX_PRECISION_INT8 : NVFX_PRECISION_FLOAT;
  nvfx_manager* m = nullptr;
  if (nvfx_manager_create(&d, &m) != NVFX_OK) throw std::runtime_error("cannot create the manager");
  std::array<int, kKinds> model{};
  for (int k = 0; k < kKinds; ++k) {
    nvfx_manager_model_desc md;
    nvfx_manager_model_desc_init(&md);
    md.max_sims = a.i("max-sims", 16);
    md.proxy_sims = a.i("proxy-sims", 8);
    if (a.has("slots")) {  // simulations at once per size, 32 to 256 px (-1: the default share of max_sims)
      const auto s = a.str("slots");
      std::size_t p = 0;
      for (int L = 0; L < 4; ++L) {
        const std::size_t q = std::min(s.find(',', p), s.size());
        md.slots[L] = std::stoi(s.substr(p, q - p));
        p = q + 1;
      }
    }
    md.precision = precision;
    if (nvfx_manager_add_model(m, fx[sz(k)], &md, &model[sz(k)]) != NVFX_OK) throw std::runtime_error("cannot add a model");
    if (a.flag("calibrate")) nvfx_manager_calibrate(m, model[sz(k)], 24);
  }
  std::mt19937_64 rng(a.u64("seed", 1));
  std::uniform_real_distribution<float> u01(0.f, 1.f);
  std::vector<Thing> things;
  const bool share = !a.flag("no-share");
  const auto spawn = [&](int kind, std::array<float, 3> pos, std::array<float, 3> c, std::uint64_t seed, float offset, bool torch, double now) {
    nvfx_spawn_desc s;
    nvfx_spawn_desc_init(&s);
    s.model = model[sz(kind)];
    s.controls = c.data();
    s.n_controls = 3;
    s.seed = seed;
    s.time_offset = offset;
    s.share = share ? 1 : 0;
    s.priority = kind == kExplosion ? 2.f : 1.f;
    Thing t;
    t.kind = kind;
    t.torch = torch;
    t.size = torch ? 1.f : kWorld[sz(kind)];
    t.pos = {pos[0], 0.5f * t.size, pos[2]};
    std::copy(t.pos.begin(), t.pos.end(), s.position);
    s.world_size = t.size;
    t.born = now;
    t.offset = offset;
    static int serial = 0;
    t.serial = serial++;
    if (unmanaged > 0) {
      if (nvfx_instance_create(fx[sz(kind)], unmanaged, &t.in) != NVFX_OK) return;
      nvfx_instance_set_precision(t.in, precision);
      nvfx_instance_set_controls(t.in, c.data(), 3);
      nvfx_instance_set_seed(t.in, seed);
      things.push_back(t);
      return;
    }
    if (nvfx_manager_spawn(m, &s, &t.h) == NVFX_OK) things.push_back(t);
  };
  const auto stop = [&](Thing& t) {
    if (t.in) nvfx_instance_free(t.in);
    else nvfx_manager_stop(m, t.h);
  };
  const auto random_controls = [&] { return std::array<float, 3>{0.15f + 0.7f * u01(rng), 0.15f + 0.7f * u01(rng), 0.15f + 0.7f * u01(rng)}; };
  // the field: 15% torches in rows of 8 identical copies; the rest fires, smoke and steam (the close scene: no torches,
  // 2 to 40 m ahead and 7 m either side)
  const int torches = close ? 0 : count * 15 / 100;
  for (int i = 0; i < torches; ++i) {
    const int row = i / 8;
    const std::array<float, 3> c{0.35f, 0.5f, 0.5f};
    spawn(kFire, {-12.f + 24.f * static_cast<float>(row % 2), 0.f, 15.f + 25.f * static_cast<float>(row) + 3.f * static_cast<float>(i % 8)}, c,
          1000 + static_cast<std::uint64_t>(row), 0.37f * static_cast<float>(i % 8), true, 0.0);
  }
  for (int i = torches; i < count; ++i) {
    const float r = u01(rng);
    const int kind = r < 0.53f ? kFire : r < 0.82f ? kSmoke : kSteam;
    if (close) {
      const float x = -7.f + 14.f * u01(rng), zz = 2.f + 38.f * u01(rng);
      spawn(kind, {x, 0.f, zz}, random_controls(), rng(), 0.f, false, 0.0);
    } else {
      spawn(kind, {-40.f + 80.f * u01(rng), 0.f, 5.f + 155.f * u01(rng)}, random_controls(), rng(), 0.f, false, 0.0);
    }
  }
  std::vector<std::uint8_t> upx(sz(std::max(unmanaged, 1)) * sz(std::max(unmanaged, 1)) * 4);
  std::FILE* video = nullptr;
  nfx::Image canvas;
  canvas.allocate(kW, kH);
  if (a.has("video") && unmanaged <= 0) {
    const fs::path vp = a.str("video");
    fs::create_directories(vp.parent_path());
    const std::string cmd = std::format("ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgba -s {}x{} -r 30 -i - -c:v libx264 -pix_fmt yuv420p -crf 20 '{}'", kW, kH, vp.string());
    video = popen(cmd.c_str(), "w");
    if (!video) throw std::runtime_error("cannot start ffmpeg");
  }
  const fs::path sheet_dir = a.str("sheet", "");
  std::ofstream csv;
  if (a.has("csv")) {
    csv.open(a.str("csv"));
    csv << "tick,wall_ms,cpu_ms,jobs_ms,planned_ms,plan_ms,instances,visible,sims,paused,proxy_sims,proxied,shared,updated,skipped,held,interpolated,steps,renders\n";
  }
  const int ticks = static_cast<int>(seconds * fps);
  std::vector<double> wall, cpu, jobs, planned, plan, overhead;
  std::array<std::array<double, 4>, 4> lod{};
  double visible = 0, proxied = 0, sharedv = 0, sims = 0, paused = 0, skipped = 0, held = 0, interp = 0, steps = 0, inst = 0, halves = 0;
  std::size_t memory = 0;
  double next_boom = 0.0, next_swap = close ? 2.0 : 0.5;
  std::exponential_distribution<double> boom_gap(close ? 1.0 / 1.5 : 1.5);
  for (int t = 0; t < ticks; ++t) {
    const double now = static_cast<double>(t) / fps;
    const Camera cam = camera_at(now, close);
    nvfx_manager_set_camera(m, cam.pos.data(), cam.fwd.data(), cam.up.data(), kFov, static_cast<float>(kW), static_cast<float>(kH));
    // the game: explosions ahead, finished ones stopped, fires that die and start elsewhere
    while (next_boom <= now) {
      if (close) {
        const float x = cam.pos[0] - 6.f + 12.f * u01(rng), zz = cam.pos[2] + 8.f + 12.f * u01(rng);
        spawn(kExplosion, {x, 0.f, zz}, random_controls(), rng(), 0.f, false, now);
      } else {
        spawn(kExplosion, {cam.pos[0] - 25.f + 50.f * u01(rng), 0.f, cam.pos[2] + 10.f + 70.f * u01(rng)}, random_controls(), rng(), 0.f, false, now);
      }
      next_boom += boom_gap(rng);
    }
    if (now >= next_swap) {
      next_swap += close ? 2.0 : 0.5;
      for (int tries = 0; tries < 8; ++tries) {
        const std::size_t k = rng() % things.size();
        if (things[k].torch || things[k].kind == kExplosion) continue;
        const int kind = things[k].kind;
        stop(things[k]);
        things.erase(things.begin() + static_cast<std::ptrdiff_t>(k));
        if (close) {
          const float x = -7.f + 14.f * u01(rng), zz = cam.pos[2] + 6.f + 29.f * u01(rng);
          spawn(kind, {x, 0.f, zz}, random_controls(), rng(), 0.f, false, now);
        } else {
          spawn(kind, {-40.f + 80.f * u01(rng), 0.f, cam.pos[2] + 10.f + 150.f * u01(rng)}, random_controls(), rng(), 0.f, false, now);
        }
        break;
      }
    }
    for (std::size_t k = 0; k < things.size();) {
      nvfx_manager_picture p{};
      if (!things[k].in) nvfx_manager_get_picture(m, things[k].h, &p);
      if (things[k].kind == kExplosion && ((p.flags & NVFX_PICTURE_FINISHED) || now - things[k].born > 3.5)) {
        stop(things[k]);
        things.erase(things.begin() + static_cast<std::ptrdiff_t>(k));
        continue;
      }
      ++k;
    }
    const double c0 = cpu_ms();
    const auto w0 = std::chrono::steady_clock::now();
    nvfx_manager_stats st{};
    if (unmanaged > 0) {
      const float focal = static_cast<float>(kH) / (2.f * std::tan(0.5f * kFov));
      for (Thing& th : things) {
        const std::array<float, 3> r{th.pos[0] - cam.pos[0], th.pos[1] - cam.pos[1], th.pos[2] - cam.pos[2]};
        const float dd = r[0] * cam.fwd[0] + r[1] * cam.fwd[1] + r[2] * cam.fwd[2];
        if (dd <= 1e-4f) continue;
        const float px = th.size * focal / dd;
        const float sx = (r[0] * cam.right[0] + r[1] * cam.right[1] + r[2] * cam.right[2]) * focal / dd;
        const float sy = (r[0] * cam.up[0] + r[1] * cam.up[1] + r[2] * cam.up[2]) * focal / dd;
        if (std::abs(sx) - 0.5f * px > 0.5f * kW || std::abs(sy) - 0.5f * px > 0.5f * kH) continue;
        st.visible++;
        const long before = (static_cast<long>(t) * 30 + th.serial) / fps, after = (static_cast<long>(t + 1) * 30 + th.serial) / fps;
        if (after == before) continue;
        nvfx_render(th.in, now - th.born + static_cast<double>(th.offset), upx.data(), sz(unmanaged) * 4);
        st.renders++;
      }
      st.instances = static_cast<int>(things.size());
      st.memory_bytes = 0;
      for (const Thing& th : things) st.memory_bytes += nvfx_instance_scratch_bytes(th.in);
    } else {
      if (nvfx_manager_tick(m, 1.0 / fps) != NVFX_OK) throw std::runtime_error("tick");
    }
    const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    const double cpu_tick = cpu_ms() - c0;
    if (unmanaged <= 0) nvfx_manager_get_stats(m, &st);
    const bool warm = t >= fps;  // the first second (every simulation warming up) is not counted
    if (warm) {
      wall.push_back(wall_ms);
      cpu.push_back(cpu_tick);
      jobs.push_back(st.jobs_ms);
      planned.push_back(st.planned_ms);
      plan.push_back(st.plan_ms);
      overhead.push_back(st.overhead_ms);
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) lod[sz(i)][sz(j)] += st.lod[i][j];
      }
      visible += st.visible;
      proxied += st.proxied;
      sharedv += st.shared_viewers;
      sims += st.sims;
      paused += st.sims_paused;
      skipped += st.skipped;
      held += st.held;
      interp += st.interpolated;
      steps += st.steps;
      inst += st.instances;
      halves += st.half;
    }
    memory = std::max(memory, st.memory_bytes);
    if (csv.is_open()) {
      csv << std::format("{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{},{},{},{},{},{},{},{},{},{}\n", t, wall_ms, cpu_tick, st.jobs_ms, st.planned_ms, st.plan_ms, st.instances,
                         st.visible, st.sims, st.sims_paused, st.proxy_sims, st.proxied, st.shared_viewers, st.updated, st.skipped, st.held, st.interpolated, st.steps, st.renders);
    }
    const bool snap = !sheet_dir.empty() && unmanaged <= 0 && (t == ticks / 4 || t == ticks / 2 || t == 3 * ticks / 4);
    if ((video && t % std::max(1, fps / 30) == 0) || snap) composite(m, things, cam, canvas);
    if (video && t % std::max(1, fps / 30) == 0) std::fwrite(canvas.rgba.data(), 1, canvas.rgba.size(), video);
    if (snap) {
      fs::create_directories(sheet_dir);
      const std::string label = a.str("label", "crowd");
      (void)nfx::write_png(sheet_dir / std::format("{}_view_{:.0f}s.png", label, now), canvas);
      (void)nfx::write_png(sheet_dir / std::format("{}_sheet_{:.0f}s.png", label, now), instance_sheet(m, things));
    }
  }
  if (video) pclose(video);
  const double n = static_cast<double>(wall.size());
  const double mean_plan = std::ranges::fold_left(plan, 0.0, std::plus{}) / n;
  std::string lods;
  for (int i = 0; i < 4; ++i) {
    lods += std::format("{}px", 32 << i);
    for (int j = 0; j < 4; ++j) lods += std::format(" {:.1f}", lod[sz(i)][sz(j)] / n);
    lods += i < 3 ? "; " : "";
  }
  const std::string label = a.str("label", "crowd");
  std::println(
      "{}: {:.0f} instances ({:.0f} visible), {:.0f} s at {} Hz, budget {} ms, {} thread(s): tick wall median {:.2f} ms, p99 {:.2f}; CPU per tick median "
      "{:.2f}, p99 {:.2f}, mean {:.2f} ms ({:.0f}% of the budget); jobs {:.2f} ms; planned {:.2f}; plan {:.2f}; planning on the calling thread {:.3f} ms",
      label, inst / n, visible / n, seconds, fps, d.budget_ms > 1e8f ? std::string("unlimited") : std::format("{:.1f}", d.budget_ms), threads, percentile(wall, 0.5),
      percentile(wall, 0.99), percentile(cpu, 0.5), percentile(cpu, 0.99), std::ranges::fold_left(cpu, 0.0, std::plus{}) / n,
      d.budget_ms > 1e8f ? 0.0 : 100.0 * (std::ranges::fold_left(cpu, 0.0, std::plus{}) / n) / d.budget_ms, std::ranges::fold_left(jobs, 0.0, std::plus{}) / n,
      std::ranges::fold_left(planned, 0.0, std::plus{}) / n, mean_plan, std::ranges::fold_left(overhead, 0.0, std::plus{}) / n);
  std::println("{}: simulations {:.1f} ({:.1f} paused), proxied instances {:.1f}, sharing {:.1f}; per second: {:.0f} steps, {:.1f} updates skipped, {:.1f} held, "
               "{:.0f} pictures interpolated; levels (sims at 30/15/10/7.5 Hz, by the size drawn at): {}; drawn at half size {:.2f}; memory {:.1f} MB "
               "(manager), peak RSS {:.1f} MB",
               label, sims / n, paused / n, proxied / n, sharedv / n, steps / n * fps, skipped / n * fps, held / n * fps, interp / n * fps, lods, halves / n,
               static_cast<double>(memory) / 1048576.0, peak_rss_mb());
  if (a.has("summary")) {  // one CSV line per run, for tools/manager/time_crowd.sh
    const fs::path sp = a.str("summary");
    const bool fresh = !fs::exists(sp);
    std::ofstream o(sp, std::ios::app);
    if (fresh) {
      o << "label,instances,visible,seconds,fps,budget_ms,threads,wall_median,wall_p99,cpu_median,cpu_p99,cpu_mean,jobs_mean,planned_mean,plan_mean,sims,paused,"
           "proxied,sharing,steps_per_s,skipped_per_s,held_per_s,interpolated_per_s,overhead_mean";
      for (int i = 0; i < 4; ++i) {
        for (const char* r : {"30", "15", "10", "7.5"}) o << std::format(",lod{}_{}", 32 << i, r);
      }
      o << ",memory_mb,rss_mb,half_sims\n";
    }
    o << std::format("{},{:.1f},{:.1f},{:.0f},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.2f},{:.2f},{:.2f},{:.2f},{:.0f},{:.2f},{:.2f},{:.0f}", label,
                     inst / n, visible / n, seconds, fps, d.budget_ms > 1e8f ? -1.f : d.budget_ms, threads, percentile(wall, 0.5), percentile(wall, 0.99),
                     percentile(cpu, 0.5), percentile(cpu, 0.99), std::ranges::fold_left(cpu, 0.0, std::plus{}) / n, std::ranges::fold_left(jobs, 0.0, std::plus{}) / n,
                     std::ranges::fold_left(planned, 0.0, std::plus{}) / n, mean_plan, sims / n, paused / n, proxied / n, sharedv / n, steps / n * fps,
                     skipped / n * fps, held / n * fps, interp / n * fps);
    o << std::format(",{:.4f}", std::ranges::fold_left(overhead, 0.0, std::plus{}) / n);
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) o << std::format(",{:.2f}", lod[sz(i)][sz(j)] / n);
    }
    o << std::format(",{:.1f},{:.1f},{:.2f}\n", static_cast<double>(memory) / 1048576.0, peak_rss_mb(), halves / n);
  }
  for (Thing& th : things) {
    if (th.in) nvfx_instance_free(th.in);
  }
  nvfx_manager_free(m);
  for (auto* f : fx) nvfx_effect_free(f);
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_crowd: {}", e.what());
  return 2;
}
