// nvfx_scene: how many neural effects fit in a game frame on one core (docs/REPORT.md §6).
//
//   nvfx_scene effect.nvfx [--near 8] [--far 16] [--rate 30] [--fps 60] [--seconds 10] [--core 3] [--float]
//
// A scene with `near` instances at 128 px and `far` instances at 64 px (level of detail), each with its own seed.
// Every instance is evaluated `rate` times per second, its updates spread evenly over the game's frames (a game at
// `fps` frames per second updates rate / fps of the instances each frame). Reports the cost per game frame on one
// pinned core: mean, 99th percentile and worst, plus the share of a frame's time. Frame models run at the runtime's
// default precision (int8 for the grid family); --float runs the float network.
#include "../tools/args.hpp"

#include <neuralfx/nvfx.h>

#include <algorithm>
#include <chrono>
#include <print>
#include <sched.h>
#include <vector>

int main(int argc, char** argv) try {
  const nfx::tools::Args a(argc, argv, {"help", "float"});
  if (a.flag("help") || a.positional().empty()) {
    std::println("nvfx_scene effect.nvfx [--near 8] [--far 16] [--rate 30] [--fps 60] [--seconds 10] [--core 3] [--float]");
    return 0;
  }
  const int near = a.i("near", 8), far = a.i("far", 16), rate = a.i("rate", 30), fps = a.i("fps", 60), core = a.i("core", 3);
  const double seconds = a.f("seconds", 10.f);
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  sched_setaffinity(0, sizeof(set), &set);

  nvfx_effect* fx = nullptr;
  if (nvfx_effect_load(a.positional()[0].c_str(), &fx) != NVFX_OK) throw std::runtime_error("cannot load the effect");
  nvfx_effect_info info{};
  nvfx_effect_get_info(fx, &info);
  struct Inst {
    nvfx_instance* in;
    int size;
    std::vector<std::uint8_t> pixels;
  };
  std::vector<Inst> scene;
  for (int i = 0; i < near + far; ++i) {
    const int size = i < near ? 128 : 64;
    Inst s{nullptr, size, std::vector<std::uint8_t>(static_cast<std::size_t>(size) * size * 4)};
    if (nvfx_instance_create(fx, size, &s.in) != NVFX_OK) throw std::runtime_error("cannot create an instance");
    if (a.flag("float")) nvfx_instance_set_precision(s.in, NVFX_PRECISION_FLOAT);
    nvfx_instance_set_seed(s.in, static_cast<std::uint64_t>(i) * 7919u + 1u);
    const float controls[3] = {0.3f + 0.05f * static_cast<float>(i % 8), 0.5f, 0.5f};
    nvfx_instance_set_controls(s.in, controls, info.n_controls);
    scene.push_back(std::move(s));
  }
  // Each game frame updates the instances whose turn it is: instance i at frames where (frame * rate + i) crosses a
  // multiple of fps, i.e. rate updates per second per instance, staggered across frames.
  const int frames = static_cast<int>(seconds * fps);
  std::vector<double> frame_ms;
  long renders = 0;
  for (int f = 0; f < frames; ++f) {
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < scene.size(); ++i) {
      const long before = (static_cast<long>(f) * rate + static_cast<long>(i)) / fps;
      const long after = (static_cast<long>(f + 1) * rate + static_cast<long>(i)) / fps;
      if (after == before) continue;
      nvfx_render(scene[i].in, f / static_cast<double>(fps), scene[i].pixels.data(), static_cast<std::size_t>(scene[i].size) * 4);
      ++renders;
    }
    frame_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::vector<double> sorted = frame_ms;
  std::ranges::sort(sorted);
  double mean = 0;
  for (const double v : frame_ms) mean += v / static_cast<double>(frame_ms.size());
  const double budget = 1000.0 / fps;
  std::println("{} near (128 px) + {} far (64 px) instances at {} Hz in a {} fps game: {:.2f} renders per frame; one core per game "
               "frame: mean {:.3f} ms, p99 {:.3f} ms, worst {:.3f} ms ({:.0f}% of a {:.1f} ms frame)",
               near, far, rate, fps, static_cast<double>(renders) / frames, mean, sorted[sorted.size() * 99 / 100], sorted.back(),
               100.0 * mean / budget, budget);
  for (auto& s : scene) nvfx_instance_free(s.in);
  nvfx_effect_free(fx);
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_scene: {}", e.what());
  return 2;
}
