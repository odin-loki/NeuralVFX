// Study N: a small 3D demo (docs/STUDY_3D.md). The 3D simulator's pilot, recording training runs, training the 3D
// rollout model, validation and the test (once), demo videos and timing.
//
//   nvfx_3d pilot  --widths 48,64 --seconds 12 --seeds 2       cost per frame, plume rise and puffing at each grid
//   nvfx_3d look   --width 64 --frames 150 --out DIR            stills of the simulation from four cameras
//
// Data, runs, checkpoints and logs go under --root (default $NEURALVFX_DATA/n_3d, else /root/nvfx-data/n_3d).
#include "args.hpp"

#include <neuralfx/image_io.hpp>
#include <neuralfx/render3d.hpp>
#include <neuralfx/sim3d.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <numbers>
#include <print>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using nfx::tools::Args;

namespace {

std::size_t zu(int v) { return static_cast<std::size_t>(v); }
float fl(int v) { return static_cast<float>(v); }

double thread_cpu() {
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

// The four cameras of the study: around the effect at 0, 90, 180 and 270 degrees, a little above the flame's middle.
nfx::render3d::Camera study_camera(int k, float aspect, float height_world) {
  const float az = static_cast<float>(k) * 0.5f * std::numbers::pi_v<float>;
  const float r = 2.2f, ty = 0.42f * height_world;
  const std::array<float, 3> eye{r * std::sin(az), ty + 0.35f, r * std::cos(az)};
  return nfx::render3d::look_at(eye, {0.f, ty, 0.f}, {0.f, 1.f, 0.f}, 42.f, aspect);
}

// A premultiplied RGBA picture over a dark background, into an image at (x0, y0).
void put_over(nfx::Image& img, const std::vector<std::uint8_t>& rgba, int w, int h, int x0, int y0) {
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::uint8_t* s = rgba.data() + (zu(y) * zu(w) + zu(x)) * 4;
      std::uint8_t* d = img.pixel(x0 + x, y0 + y);
      const float a = s[3] / 255.f;
      const float bg = 14.f + 10.f * static_cast<float>(y) / static_cast<float>(h);
      for (int c = 0; c < 3; ++c) d[c] = static_cast<std::uint8_t>(std::clamp(s[c] + bg * (1.f - a), 0.f, 255.f));
      d[3] = 255;
    }
  }
}

// --- pilot ----------------------------------------------------------------------------------------------------------

int pilot(const Args& a) {
  std::vector<int> widths;
  for (const float w : nfx::tools::parse_floats(a.str("widths", "48,64"))) widths.push_back(static_cast<int>(w));
  const float seconds = a.f("seconds", 12.f);
  const int seeds = a.i("seeds", 2), threads = a.i("threads", 2);
  const float I = a.f("intensity", 0.6f), W = a.f("wind", 0.5f), Tu = a.f("turbulence", 0.5f);
  const fs::path root = data_root(a) / "pilot";
  fs::create_directories(root);
  const std::string csv = (root / "pilot.csv").string();
  std::ofstream out(csv, std::ios::app);
  std::println("width,height,seed,ms_per_frame_wall,cpu_s_per_frame,threads,plume_top_2s,plume_top_6s,heat_centroid,puff_hz_first,puff_hz_second,puff_hz_all,peak_power_ratio");
  for (const int w : widths) {
    for (int k = 0; k < seeds; ++k) {
      nfx::sim3d::Params p;
      p.width = w;
      p.height = w * 3 / 2;
      p.intensity = I;
      p.wind = W;
      p.turbulence = Tu;
      p.seed = 9'000'900 + static_cast<std::uint64_t>(k);  // pilot seeds (study N's range)
      p.threads = threads;
      nfx::sim3d::Fluid f(p);
      const int frames = static_cast<int>(seconds * p.fps);
      const std::size_t n = zu(f.nx()) * zu(f.ny()) * zu(f.nz());
      std::vector<float> T(n), D(n);
      std::vector<double> band;  // heat in a band around the flames' tips, per frame
      double top2 = 0, top6 = 0, centroid = 0;
      int centroid_n = 0;
      const double c0 = thread_cpu(), w0 = now();
      for (int fr = 0; fr < frames; ++fr) {
        f.step_frame();
        f.fields(T, D);
        // the plume's top: the highest slab whose soot is above 2% of the largest slab's
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
        if (a.flag("verbose") && (fr + 1) % 15 == 0) {
          std::vector<float> uvw(3 * n);
          f.velocity(uvw);
          float vmax = 0, tmax = 0, dmax = 0;
          double vmean = 0, tsum = 0;
          for (std::size_t i = 0; i < n; ++i) {
            vmax = std::max(vmax, uvw[3 * i + 1]);
            tmax = std::max(tmax, T[i]);
            dmax = std::max(dmax, D[i]);
            vmean += static_cast<double>(uvw[3 * i + 1]) * T[i];
            tsum += T[i];
          }
          std::println(stderr, "  t {:.1f} s: top {:.2f}, v max {:.1f} cells/s (heat-weighted {:.1f}), T max {:.2f}, soot max {:.3f}", static_cast<float>(fr + 1) / p.fps, top_rel, vmax,
                       vmean / std::max(1e-9, tsum), tmax, dmax);
        }
        if (fr == 2 * 30 - 1) top2 = top_rel;
        if (fr == 6 * 30 - 1) top6 = top_rel;
        // the visible flame's volume (cells hotter than 0.5, about 1400 K in the renderer): its spectrum gives the puffing
        double b = 0;
        for (const float t : T) b += t > 0.5f ? 1.0 : 0.0;
        band.push_back(b);
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
      const double cpu = (thread_cpu() - c0) / frames, wall = (now() - w0) / frames;
      // the puffing frequency: the peak between 0.5 and 6 Hz of the flame volume's spectrum (Welch: 4 s Hann windows
      // overlapping by half, after a 3 s warm-up), on the first and second halves and on the whole (how steady it is)
      const auto peak = [&](int from, int to, double* ratio) {
        const int win = std::min(120, to - from);
        std::vector<double> power;
        for (double hz = 0.5; hz <= 6.0 + 1e-9; hz += 0.05) power.push_back(0.0);
        int segments = 0;
        for (int s0 = from; s0 + win <= to; s0 += win / 2) {
          double mean = 0;
          for (int i = s0; i < s0 + win; ++i) mean += band[zu(i)];
          mean /= win;
          std::size_t bin = 0;
          for (double hz = 0.5; hz <= 6.0 + 1e-9; hz += 0.05, ++bin) {
            double re = 0, im = 0;
            for (int i = s0; i < s0 + win; ++i) {
              const double x = band[zu(i)] - mean, ph = 2.0 * std::numbers::pi * hz * (i - s0) / p.fps;
              const double wv = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * (i - s0) / (win - 1));
              re += wv * x * std::cos(ph);
              im += wv * x * std::sin(ph);
            }
            power[bin] += re * re + im * im;
          }
          ++segments;
        }
        std::size_t best = 0;
        double total = 0;
        for (std::size_t j = 0; j < power.size(); ++j) {
          total += power[j];
          if (power[j] > power[best]) best = j;
        }
        if (ratio) *ratio = power[best] / (total / static_cast<double>(power.size()));
        (void)segments;
        return 0.5 + 0.05 * static_cast<double>(best);
      };
      const int start = 90, mid = (start + frames) / 2;
      double ratio = 0;
      const double f1 = peak(start, mid, nullptr), f2 = peak(mid, frames, nullptr), fa = peak(start, frames, &ratio);
      const std::string row = std::format("{},{},{},{:.1f},{:.3f},{},{:.3f},{:.3f},{:.3f},{:.2f},{:.2f},{:.2f},{:.1f}", f.nx(), f.ny(), p.seed, 1000.0 * wall, cpu,
                                          threads, top2, top6, centroid / std::max(1, centroid_n), f1, f2, fa, ratio);
      std::println("{}", row);
      out << row << "\n";
      out.flush();
    }
  }
  return 0;
}

// --- look -------------------------------------------------------------------------------------------------------------

int look(const Args& a) {
  nfx::sim3d::Params p;
  p.width = a.i("width", 64);
  p.height = p.width * 3 / 2;
  p.intensity = a.f("intensity", 0.6f);
  p.wind = a.f("wind", 0.5f);
  p.turbulence = a.f("turbulence", 0.5f);
  p.seed = a.u64("seed", 9'000'950);
  p.threads = a.i("threads", 2);
  if (a.str("effect", "fire") == "smoke") p.effect = nfx::sim3d::Effect::smoke;
  const int frames = a.i("frames", 150), every = a.i("every", 30), px = a.i("size", 192);
  const fs::path out = a.has("out") ? fs::path(a.str("out")) : data_root(a) / "look";
  fs::create_directories(out);
  nfx::render3d::Optics o;
  o.exposure = a.f("exposure", o.exposure);
  o.emission = a.f("emission", o.emission);
  o.soot_extinction = a.f("soot", o.soot_extinction);
  o.flame_extinction = a.f("flame", o.flame_extinction);
  o.kelvin_per_unit = a.f("kpu", o.kelvin_per_unit);
  o.kelvin0 = a.f("k0", o.kelvin0);
  nfx::sim3d::Fluid f(p);
  nfx::render3d::Renderer r(f.nx(), f.ny(), f.nz(), o);
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
    std::println("frame {}", fr);
  }
  const fs::path file = out / std::format("look_w{}_s{}.png", p.width, p.seed);
  if (auto e = nfx::write_png(file, sheet); !e) {
    std::println(stderr, "{}", e.error());
    return 1;
  }
  std::println("wrote {}", file.string());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a(argc, argv, {"verbose"});
    if (a.positional().empty()) {
      std::println(stderr, "usage: nvfx_3d pilot|look [options] (see the header of tools/nvfx_3d.cpp)");
      return 2;
    }
    const std::string cmd = a.positional()[0];
    int rc = 2;
    if (cmd == "pilot") rc = pilot(a);
    else if (cmd == "look") rc = look(a);
    else std::println(stderr, "unknown command {}", cmd);
    a.warn_unused();
    return rc;
  } catch (const std::exception& e) {
    std::println(stderr, "error: {}", e.what());
    return 1;
  }
}
