// The int8 path of the frame models in nvfx_experiment (experiment_int8.hpp): quality against the float path, and cost.
#include "experiment_int8.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/nvfx.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <format>
#include <fstream>
#include <memory>
#include <print>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace nfx::study_int8 {
namespace {

namespace fs = std::filesystem;

constexpr int kSize = 128, kFrames = 64;

struct EffectFree {
  void operator()(nvfx_effect* e) const { nvfx_effect_free(e); }
};
using EffectPtr = std::unique_ptr<nvfx_effect, EffectFree>;

EffectPtr load(const fs::path& p) {
  nvfx_effect* e = nullptr;
  if (nvfx_effect_load(p.c_str(), &e) != NVFX_OK) throw std::runtime_error("cannot load " + p.string());
  return EffectPtr(e);
}

// What a frame is rendered with: the precision and the ISA (AUTO: the runtime's choice; int8 takes AVX-512 VNNI where
// the CPU has it).
struct Setting {
  nvfx_precision precision;
  nvfx_isa isa;
};

// A clip through the runtime, as nvfx_experiment's runtime_clip(): controls, then a training variation (>= 0) or a
// seed, drift off.
Clip render(nvfx_effect* e, Setting s, std::span<const float> controls, int variation, std::uint64_t seed) {
  if (nvfx_set_isa(s.isa) != NVFX_OK) throw std::runtime_error("ISA not available");
  nvfx_instance* in = nullptr;
  const nvfx_status st = nvfx_instance_create(e, kSize, &in);
  nvfx_set_isa(NVFX_ISA_AUTO);
  if (st != NVFX_OK) throw std::runtime_error("instance failed");
  nvfx_instance_set_precision(in, s.precision);
  nvfx_instance_set_controls(in, controls.data(), static_cast<int>(controls.size()));
  nvfx_instance_set_drift(in, 0.f);
  if (variation >= 0) nvfx_instance_set_variation(in, variation);
  else nvfx_instance_set_seed(in, seed);
  nvfx_effect_info info{};
  nvfx_effect_get_info(e, &info);
  Clip clip;
  clip.allocate(kSize, kFrames);
  clip.loop = info.loops != 0;
  clip.fps = info.fps;
  for (int f = 0; f < kFrames; ++f) nvfx_render(in, f / static_cast<double>(info.fps), clip.frame(f).data(), kSize * 4);
  nvfx_instance_free(in);
  return clip;
}

struct Diff {
  int max = 0;
  double mean = 0;
};
Diff diff(const Clip& a, const Clip& b) {
  Diff d;
  long sum = 0, n = 0;
  for (int f = 0; f < a.frames; ++f) {
    const auto x = a.frame(f), y = b.frame(f);
    for (std::size_t i = 0; i < x.size(); ++i) {
      const int e = std::abs(static_cast<int>(x[i]) - static_cast<int>(y[i]));
      d.max = std::max(d.max, e);
      sum += e;
      ++n;
    }
  }
  d.mean = n ? static_cast<double>(sum) / static_cast<double>(n) : 0.0;
  return d;
}

struct Case {
  std::string study, model, clip;
  fs::path model_path, clip_path;
  std::vector<float> controls;
  int variation = -1;
};

std::vector<Case> cases(const fs::path& data) {
  std::vector<Case> v;
  const fs::path models = data / "models", clips = data / "clips";
  // A: one model per clip; grid_m at 8 bits for all 12 clips (the rule), the other grid sizes for the first of each effect
  for (const char* e : {"fire", "smoke", "explosion"}) {
    for (int k = 0; k < 4; ++k) {
      const std::string clip = std::format("{}_{}", e, k);
      for (const char* cfg : {"grid_m8", "grid_s8", "grid_l8", "grid_mt8", "grid_m16"}) {
        const fs::path m = models / "a" / std::format("{}_{}.nvfx", clip, cfg);
        if (fs::exists(m)) v.push_back({"a", cfg, clip, m, clips / "a" / (clip + ".nfxclip"), {}, 0});
      }
    }
  }
  // B: the control models on the 10 held-out settings per effect (controls from the clip names, rounded to 0.01: the
  // same for both precisions)
  if (fs::exists(clips / "b")) {
    std::vector<fs::path> tests;
    for (const auto& f : fs::directory_iterator(clips / "b")) {
      if (f.path().stem().string().find("_test_") != std::string::npos) tests.push_back(f.path());
    }
    std::ranges::sort(tests);
    for (const fs::path& t : tests) {
      const std::string stem = t.stem().string();
      const std::string effect = stem.substr(0, stem.find("_test_"));
      std::vector<float> ctl;
      std::size_t at = stem.find("_test_") + 6;
      for (int k = 0; k < 3; ++k) {
        const std::size_t end = stem.find('_', at);
        ctl.push_back(std::stof(stem.substr(at, end - at)));
        at = end + 1;
      }
      for (const char* cfg : {"grid_k8", "grid_k16"}) {
        const fs::path m = models / "b" / std::format("{}_{}.nvfx", effect, cfg);
        if (fs::exists(m)) v.push_back({"b", cfg, stem, m, t, ctl, -1});
      }
    }
  }
  // C: the variation models replaying their 24 training seeds (reconstruction)
  for (const char* e : {"fire", "smoke", "explosion"}) {
    for (int k = 0; k < 24; ++k) {
      const std::string clip = std::format("{}_seed{}", e, 1000 + k);
      for (const char* cfg : {"variation_k8", "variation_k24"}) {
        const fs::path m = models / "c" / std::format("{}_{}.nvfx", e, cfg);
        if (fs::exists(m)) v.push_back({"c", cfg, clip, m, clips / "c" / (clip + ".nfxclip"), {}, k});
      }
    }
  }
  return v;
}

double cpu_ms() {
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return static_cast<double>(t.tv_sec) * 1e3 + static_cast<double>(t.tv_nsec) * 1e-6;
}

}  // namespace

void step_quality(const Ctx& c) {
  fs::create_directories(c.results);
  const fs::path out = c.results / "int8_quality.csv";
  std::ofstream csv(out);
  // int8: the runtime's choice (AVX-512 VNNI where the CPU has it, 8-bit activations); int8_avx2: AVX2 forced (7-bit
  // activations, pmaddubsw), what a CPU without AVX-512 gets
  csv << "study,model,clip,float_active_psnr,int8_active_psnr,change_db,float_psnr,int8_psnr,float_ssim,int8_ssim,max_diff,mean_abs_diff,"
         "int8_avx2_active_psnr,change_avx2_db,max_diff_avx2,mean_abs_diff_avx2,int8_max_diff_across_isas\n";
  struct Pair {
    std::vector<double> f, q;
    int max_diff = 0, isa_diff = 0;
    double mean_diff = 0;
  };
  std::vector<std::pair<std::string, Pair>> sets;
  const auto set = [&](const std::string& k) -> Pair& {
    for (auto& [name, p] : sets)
      if (name == k) return p;
    sets.emplace_back(k, Pair{});
    return sets.back().second;
  };
  std::vector<nvfx_isa> isas;  // every ISA the CPU has, for the int8 frames' agreement
  for (const nvfx_isa i : {NVFX_ISA_BASELINE, NVFX_ISA_AVX2, NVFX_ISA_AVX512})
    if (nvfx_set_isa(i) == NVFX_OK) isas.push_back(i);
  nvfx_set_isa(NVFX_ISA_AUTO);
  const bool avx2 = std::ranges::find(isas, NVFX_ISA_AVX2) != isas.end();
  const auto all = cases(c.data);
  if (all.empty()) throw std::runtime_error("no models under " + (c.data / "models").string());
  std::string last;
  EffectPtr e;
  for (const Case& k : all) {
    auto ref = read_clip(k.clip_path);
    if (!ref) throw std::runtime_error(ref.error());
    if (k.model_path.string() != last) {
      e = load(k.model_path);
      last = k.model_path.string();
    }
    const Clip fl = render(e.get(), {NVFX_PRECISION_FLOAT, NVFX_ISA_AUTO}, k.controls, k.variation, 0);
    const Clip q8 = render(e.get(), {NVFX_PRECISION_INT8, NVFX_ISA_AUTO}, k.controls, k.variation, 0);
    const Clip q2 = avx2 ? render(e.get(), {NVFX_PRECISION_INT8, NVFX_ISA_AVX2}, k.controls, k.variation, 0) : q8;
    int across = 0;
    for (const nvfx_isa i : isas) across = std::max(across, diff(q8, render(e.get(), {NVFX_PRECISION_INT8, i}, k.controls, k.variation, 0)).max);
    const auto sf = metrics::score(*ref, fl), sq = metrics::score(*ref, q8);
    const double a2 = metrics::active_psnr(*ref, q2);
    const Diff d = diff(fl, q8), d2 = diff(fl, q2);
    csv << std::format("{},{},{},{:.4f},{:.4f},{:+.4f},{:.4f},{:.4f},{:.5f},{:.5f},{},{:.4f},{:.4f},{:+.4f},{},{:.4f},{}\n", k.study, k.model, k.clip,
                       sf.active_psnr, sq.active_psnr, sq.active_psnr - sf.active_psnr, sf.psnr, sq.psnr, sf.ssim, sq.ssim, d.max, d.mean, a2,
                       a2 - sf.active_psnr, d2.max, d2.mean, across);
    csv.flush();
    const auto add = [&](const std::string& name, double q, const Diff& dd) {
      Pair& p = set(name);
      p.f.push_back(sf.active_psnr);
      p.q.push_back(q);
      p.max_diff = std::max(p.max_diff, dd.max);
      p.isa_diff = std::max(p.isa_diff, across);
      p.mean_diff += dd.mean;
    };
    add(k.study + ":" + k.model, sq.active_psnr, d);
    if (avx2) add(k.study + ":" + k.model + " avx2", a2, d2);
  }
  std::ofstream sum(c.results / "int8_summary.csv");
  sum << "set,clips,float_active_psnr,int8_active_psnr,change_db,lo,hi,max_diff,mean_abs_diff,int8_max_diff_across_isas,rule\n";
  for (const auto& [name, p] : sets) {
    const auto iv = metrics::paired_bootstrap(p.q, p.f);
    double mf = 0, mq = 0;
    for (std::size_t i = 0; i < p.f.size(); ++i) {
      mf += p.f[i];
      mq += p.q[i];
    }
    const double n = static_cast<double>(p.f.size());
    // the rule (A's grid_m only, at the runtime's choice and with AVX2 forced): mean change within -0.05 dB and the
    // interval not entirely below it
    const std::string rule = name.starts_with("a:grid_m8") ? (iv.mean >= -0.05 && iv.hi >= -0.05 ? "met: int8 can be the default" : "not met: int8 opt-in") : "";
    sum << std::format("{},{},{:.4f},{:.4f},{:+.4f},{:+.4f},{:+.4f},{},{:.4f},{},{}\n", name, p.f.size(), mf / n, mq / n, iv.mean, iv.lo, iv.hi, p.max_diff,
                       p.mean_diff / n, p.isa_diff, rule);
    std::println("{:18} {:3} clips: active PSNR {:.3f} -> {:.3f} dB, change {:+.4f} [{:+.4f}, {:+.4f}], max diff {}, mean {:.4f}, ISAs agree to {} {}", name,
                 p.f.size(), mf / n, mq / n, iv.mean, iv.lo, iv.hi, p.max_diff, p.mean_diff / n, p.isa_diff, rule);
  }
}

void step_timing(const Ctx& c) {
  fs::create_directories(c.results);
  cpu_set_t old, one;
  sched_getaffinity(0, sizeof(old), &old);
  CPU_ZERO(&one);
  CPU_SET(c.core, &one);
  sched_setaffinity(0, sizeof(one), &one);
  struct Model {
    std::string name;
    fs::path path;
  };
  std::vector<Model> models;
  for (const char* cfg : {"grid_s8", "grid_m8", "grid_l8", "grid_mt8"}) models.push_back({std::string(cfg).substr(0, std::string(cfg).size() - 1), c.data / "models" / "a" / std::format("fire_0_{}.nvfx", cfg)});
  models.push_back({"control_k8", c.data / "models" / "b" / "fire_grid_k8.nvfx"});
  models.push_back({"variation_k8", c.data / "models" / "c" / "fire_variation_k8.nvfx"});
  std::ofstream csv(c.results / "int8_timing.csv");
  csv << "model,size,isa,precision,cpu_ms,wall_ms,runs,note\n";
  const struct {
    nvfx_isa isa;
    const char* name;
  } isas[] = {{NVFX_ISA_AVX2, "avx2"}, {NVFX_ISA_AVX512, "avx512"}, {NVFX_ISA_BASELINE, "baseline"}};
  for (const Model& m : models) {
    if (!fs::exists(m.path)) continue;
    EffectPtr e = load(m.path);
    for (const int size : {64, 128, 256}) {
      for (const auto& isa : isas) {
        if (isa.isa == NVFX_ISA_BASELINE && size != 128) continue;
        for (const nvfx_precision p : {NVFX_PRECISION_FLOAT, NVFX_PRECISION_INT8}) {
          if (nvfx_set_isa(isa.isa) != NVFX_OK) continue;
          nvfx_instance* in = nullptr;
          const nvfx_status st = nvfx_instance_create(e.get(), size, &in);
          nvfx_set_isa(NVFX_ISA_AUTO);
          if (st != NVFX_OK) continue;
          nvfx_instance_set_precision(in, p);
          std::vector<std::uint8_t> buf(static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4);
          double best = 1e30, wall_best = 1e30;
          for (int r = 0; r < c.runs; ++r) {
            std::vector<double> cpu, wall;
            for (int f = 0; f < 200; ++f) {
              const double t0 = cpu_ms();
              timespec w0{}, w1{};
              clock_gettime(CLOCK_MONOTONIC, &w0);
              nvfx_render(in, f / 30.0, buf.data(), static_cast<std::size_t>(size) * 4);
              clock_gettime(CLOCK_MONOTONIC, &w1);
              if (f >= 20) {
                cpu.push_back(cpu_ms() - t0);
                wall.push_back(static_cast<double>(w1.tv_sec - w0.tv_sec) * 1e3 + static_cast<double>(w1.tv_nsec - w0.tv_nsec) * 1e-6);
              }
            }
            std::ranges::sort(cpu);
            std::ranges::sort(wall);
            best = std::min(best, cpu[cpu.size() / 2]);
            wall_best = std::min(wall_best, wall[wall.size() / 2]);
          }
          nvfx_instance_free(in);
          const char* pn = p == NVFX_PRECISION_FLOAT ? "float" : "int8";
          csv << std::format("{},{},{},{},{:.4f},{:.4f},{},{}\n", m.name, size, isa.name, pn, best, wall_best, c.runs,
                             isa.isa == NVFX_ISA_AVX512 && p == NVFX_PRECISION_INT8 ? "int8: VNNI where the CPU has it" : "");
          csv.flush();
          std::println("{:13} {:4} {:9} {:6} {:.4f} ms (wall {:.4f})", m.name, size, isa.name, pn, best, wall_best);
        }
      }
    }
  }
  sched_setaffinity(0, sizeof(old), &old);
}

}  // namespace nfx::study_int8
