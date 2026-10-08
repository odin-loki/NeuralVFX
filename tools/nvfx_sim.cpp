// nvfx_sim: simulate one effect clip and write it (.nfxclip), a contact sheet (PNG) and/or a video (mp4 via ffmpeg).
//
//   nvfx_sim --effect fire|smoke|explosion [--intensity 0.5] [--wind 0.5] [--turbulence 0.5] [--seed 1]
//            [--size 128] [--frames 64] [--sim-res 0] [--out clip.nfxclip] [--sheet sheet.png] [--video v.mp4]
//            [--bg black|grey|checker] [--bench]
//
// --bench reports the solver's cost per output frame (medians over the clip) as well.
#include "args.hpp"

#include <neuralfx/image_io.hpp>
#include <neuralfx/sim.hpp>

#include <algorithm>
#include <chrono>
#include <print>
#include <vector>

using namespace nfx;

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"bench", "help"});
  if (a.flag("help")) {
    std::println("nvfx_sim --effect fire|smoke|explosion [--intensity --wind --turbulence --seed --size --frames "
                 "--sim-res --substeps --pressure-iters --out --sheet --video --bg --bench]");
    return 0;
  }
  sim::Params p;
  if (!sim::parse_effect(a.str("effect", "fire"), p.effect)) throw std::invalid_argument("--effect: fire, smoke or explosion");
  p.intensity = a.f("intensity", p.intensity);
  p.wind = a.f("wind", p.wind);
  p.turbulence = a.f("turbulence", p.turbulence);
  p.seed = a.u64("seed", p.seed);
  p.size = a.i("size", p.size);
  p.frames = a.i("frames", p.frames);
  p.sim_res = a.i("sim-res", p.sim_res);
  p.substeps = a.i("substeps", p.substeps);
  p.pressure_iters = a.i("pressure-iters", p.pressure_iters);
  Background bg{};
  if (!parse_background(a.str("bg", p.effect == sim::Effect::fire ? "black" : "grey"), bg)) {
    throw std::invalid_argument("--bg: black, grey or checker");
  }

  if (a.flag("bench")) {
    sim::Fluid fluid(p);
    std::vector<std::uint8_t> frame(static_cast<std::size_t>(p.size) * p.size * 4);
    std::vector<double> step_ms, render_ms;
    using clock = std::chrono::steady_clock;
    const int warm = std::max(p.warmup, 60);
    for (int i = 0; i < warm + p.frames; ++i) {
      const auto t0 = clock::now();
      fluid.step_frame();
      const auto t1 = clock::now();
      fluid.render(frame);
      const auto t2 = clock::now();
      if (i >= warm) {
        step_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        render_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
      }
    }
    std::ranges::sort(step_ms);
    std::ranges::sort(render_ms);
    std::println("{} {}x{} (solver {} cells, {} substeps, {} pressure iterations): step {:.3f} ms, render {:.3f} ms "
                 "per output frame (medians of {})",
                 sim::effect_name(p.effect), p.size, p.size, p.sim_res > 0 ? p.sim_res : p.size, p.substeps,
                 p.pressure_iters, step_ms[step_ms.size() / 2], render_ms[render_ms.size() / 2], p.frames);
  }

  const auto t0 = std::chrono::steady_clock::now();
  const Clip clip = sim::simulate(p);
  std::println(stderr, "simulated {} ({} frames) in {:.2f} s", clip.effect, clip.frames,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  const auto report = [](const std::expected<void, std::string>& r) {
    if (!r) std::println(stderr, "warning: {}", r.error());
  };
  if (a.has("out")) report(write_clip(a.str("out"), clip));
  if (a.has("sheet")) report(write_png(a.str("sheet"), contact_sheet(clip, 8, std::max(1, clip.frames / 16), bg, 1)));
  if (a.has("video")) {
    const Clip* clips[] = {&clip};
    report(write_comparison_video(a.str("video"), clips, bg, 2, 2));
  }
  a.warn_unused();
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_sim: {}", e.what());
  return 2;
}
