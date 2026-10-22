// nvfx_lod: what the effect manager's levels of detail cost in quality (docs/MANAGER.md §5).
//
//   nvfx_lod val  [--effects fire,smoke,explosion,steam,magic] [--threads 3] [--frames 300]
//   nvfx_lod test [...]                       once, after the rule in docs/MANAGER.md §5 chose the default on val
//   nvfx_lod sheet [--effect fire] [--setting 0] [--size 64] [--div 4]   a contact sheet of the in-between frames
//
// Every run plays one instance through the manager with its level locked, 1/30 s per tick, and keeps the frames it
// shows. Against the full-rate run at the same size (the same run: the keyframes are its frames to the bit):
//   rate   sizes 32, 64 and 128 px, updates every 2, 3 or 4 frames (15, 10, 7.5 Hz), drawn between keyframes by holding,
//          crossfading or motion-compensated interpolation;
// against the full-rate run at 128 px, brought to the size shown (block averages):
//   res    32 and 64 px at 30 Hz (and up to 128 px, bilinear, against 128 px: a small texture drawn large);
//   proxy  64 px at 7.5 Hz, controls snapped to proxy cells (3 per control) and another seed: a proxy's picture;
//   floor  64 px at 30 Hz with another seed at the same controls: what any other run would show.
// Metrics: active PSNR per frame (mean over the frames drawn between keyframes for `rate`, over every frame otherwise),
// active PSNR of the whole clip, and the endless statistics (detail spectrum L1, motion ratio, mean-frame PSNR).
// Settings: validation, study G's ten (docs/DCM.md §4); test, study B's ten held-out ones; fresh seeds (val
// 7,300,000 + 100 s + e, the studies' salt-3 split; test 7,700,000 + 100 s + e). Paired bootstrap over settings, 10,000
// resamples. Writes results/manager/lod_<split>.csv and lod_<split>_summary.md; figures to the data root.
#include "args.hpp"

#include "../src/manager/interp.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/image_io.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/nvfx_manager.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <print>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
using namespace nfx;

namespace {

std::size_t sz(int v) { return static_cast<std::size_t>(v); }

using Setting = std::array<float, 3>;
bool off_grid(const Setting& s) {
  const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
  return off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f});
}
// Study B's ten held-out settings (the test) and study G's ten validation settings, as tools/nvfx_d2.cpp makes them.
std::vector<Setting> test_settings() {
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < 10) {
    const Setting s{u(rng), u(rng), u(rng)};
    if (off_grid(s)) v.push_back(s);
  }
  return v;
}
std::vector<Setting> validation_settings() {
  const auto test = test_settings();
  std::mt19937_64 rng(2027);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < 10) {
    const Setting s{u(rng), u(rng), u(rng)};
    const bool far = std::ranges::all_of(test, [&](const Setting& t) {
      return std::sqrt((s[0] - t[0]) * (s[0] - t[0]) + (s[1] - t[1]) * (s[1] - t[1]) + (s[2] - t[2]) * (s[2] - t[2])) >= 0.05f;
    });
    if (off_grid(s) && far) v.push_back(s);
  }
  return v;
}

struct EffectFile {
  std::string name;
  fs::path path;
};
std::vector<EffectFile> effect_files(const fs::path& root, const std::string& list) {
  const std::map<std::string, fs::path> where = {{"fire", root / "v2/models/fire.nvfx"},
                                                 {"smoke", root / "v2/models/smoke.nvfx"},
                                                 {"explosion", root / "v2/models/explosion.nvfx"},
                                                 {"steam", root / "fx/models/d/steam.nvfx"},
                                                 {"magic", root / "fx/models/d/magic.nvfx"}};
  std::vector<EffectFile> v;
  for (std::size_t a = 0; a <= list.size();) {
    const std::size_t b = std::min(list.find(',', a), list.size());
    const std::string n = list.substr(a, b - a);
    if (!where.contains(n)) throw std::invalid_argument("unknown effect " + n);
    v.push_back({n, where.at(n)});
    a = b + 1;
  }
  return v;
}

const char* interp_name(nvfx_interp m) {
  switch (m) {
    case NVFX_INTERP_HOLD: return "hold";
    case NVFX_INTERP_CROSSFADE: return "crossfade";
    case NVFX_INTERP_MOTION: return "motion";
    default: return "default";
  }
}

// One instance through the manager at a locked level; the frames it shows, by frame number, and which were keyframes.
struct Played {
  Clip clip;
  std::vector<std::uint8_t> key;  // 1: the frame is a keyframe (shown as rendered)
};
Played play(const nvfx_effect* fx, int size, int div, nvfx_interp interp, const Setting& c, std::uint64_t seed, int frames) {
  nvfx_manager_desc d;
  nvfx_manager_desc_init(&d);
  d.threads = 1;
  d.max_instances = 4;
  d.budget_ms = 1e6f;
  d.interp = interp;
  d.proxies = 0;
  d.catchup_frames = 64.f;
  nvfx_manager* m = nullptr;
  if (nvfx_manager_create(&d, &m) != NVFX_OK) throw std::runtime_error("manager");
  nvfx_manager_model_desc md;
  nvfx_manager_model_desc_init(&md);
  md.max_sims = 2;
  md.history = 0;
  int model = -1;
  if (nvfx_manager_add_model(m, fx, &md, &model) != NVFX_OK) throw std::runtime_error("model");
  nvfx_spawn_desc s;
  nvfx_spawn_desc_init(&s);
  s.model = model;
  s.controls = c.data();
  s.n_controls = 3;
  s.seed = seed;
  s.screen_px = static_cast<float>(size);
  s.share = 0;
  nvfx_handle h = 0;
  if (nvfx_manager_spawn(m, &s, &h) != NVFX_OK || nvfx_manager_set_lod_lock(m, h, size, div) != NVFX_OK) throw std::runtime_error("spawn");
  Played p;
  p.clip.allocate(size, frames);
  p.clip.fps = 30.f;
  p.key.assign(sz(frames), 0);
  std::vector<std::uint8_t> have(sz(frames), 0);
  int got = 0;
  for (int t = 0; got < frames && t < frames + 200; ++t) {
    if (nvfx_manager_tick(m, 1.0 / 30.0) != NVFX_OK) throw std::runtime_error("tick");
    nvfx_manager_picture pic;
    nvfx_manager_get_picture(m, h, &pic);
    if (!pic.rgba) continue;
    const long f = std::lround(pic.effect_time * 30.0);
    if (f < 0 || f >= frames || have[sz(static_cast<int>(f))]) continue;
    if (pic.size != size) throw std::runtime_error("size");
    std::copy(pic.rgba, pic.rgba + p.clip.frame_bytes(), p.clip.frame(static_cast<int>(f)).begin());
    have[sz(static_cast<int>(f))] = 1;
    p.key[sz(static_cast<int>(f))] = (pic.flags & NVFX_PICTURE_INTERPOLATED) ? 0 : 1;
    ++got;
  }
  nvfx_manager_free(m);
  if (got < frames) throw std::runtime_error("frames missing");
  return p;
}

Clip resized(const Clip& c, int size) {
  Clip o;
  o.allocate(size, c.frames);
  o.fps = c.fps;
  for (int f = 0; f < c.frames; ++f) mgr::resample(c.frame(f).data(), c.size, o.frame(f).data(), size);
  return o;
}

double apsnr(std::span<const std::uint8_t> r, std::span<const std::uint8_t> t) {  // as tools/experiment_d.cpp
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

struct Row {
  std::string effect, family, interp;
  int setting = 0, size = 0, div = 1;
  double apsnr_frames = 0, apsnr_clip = 0, spectrum = 0, motion = 0, mean_frame = 0, coverage = 0;
  int compared = 0;
};

// test against ref: per-frame active PSNR over the frames where `use` is set (all if empty), and the clip's statistics.
Row compare(const Clip& ref, const metrics::ClipStats& ref_stats, const Clip& test, const std::vector<std::uint8_t>& use) {
  Row r;
  double s = 0;
  int n = 0;
  for (int f = 0; f < ref.frames; ++f) {
    if (!use.empty() && !use[sz(f)]) continue;
    s += apsnr(ref.frame(f), test.frame(f));
    ++n;
  }
  r.apsnr_frames = n > 0 ? s / n : metrics::kPsnrCap;
  r.compared = n;
  r.apsnr_clip = metrics::active_psnr(ref, test);
  const metrics::StatDistance d = metrics::distance(ref_stats, metrics::stats(test));
  r.spectrum = d.spectrum_l1;
  r.motion = d.motion_ratio;
  r.mean_frame = d.mean_frame_psnr;
  r.coverage = d.coverage_l1;
  return r;
}

struct Split {
  std::vector<Setting> settings;
  std::uint64_t base = 0;
};
Split split_of(const std::string& name) {
  if (name == "val") return {validation_settings(), 7'300'000};
  if (name == "test") return {test_settings(), 7'700'000};
  throw std::invalid_argument("split: val or test");
}

constexpr std::array<nvfx_interp, 3> kInterps{NVFX_INTERP_HOLD, NVFX_INTERP_CROSSFADE, NVFX_INTERP_MOTION};

std::vector<Row> measure(const EffectFile& ef, const nvfx_effect* fx, int e, int si, const Setting& c, std::uint64_t seed, int frames) {
  std::vector<Row> rows;
  const auto add = [&](Row r, const char* family, int size, int div, const char* interp) {
    r.effect = ef.name;
    r.family = family;
    r.setting = si;
    r.size = size;
    r.div = div;
    r.interp = interp;
    rows.push_back(r);
  };
  std::map<int, Played> full;
  for (const int size : {32, 64, 128}) full[size] = play(fx, size, 1, NVFX_INTERP_CROSSFADE, c, seed, frames);
  const Clip& ref128 = full[128].clip;
  // rate: against the full-rate run at the same size, on the frames between keyframes
  for (const int size : {32, 64, 128}) {
    const Clip& ref = full[size].clip;
    const metrics::ClipStats rs = metrics::stats(ref);
    for (int div = 2; div <= 4; ++div) {
      // which frames are keyframes (the same whatever draws between them): from the crossfade's run
      const Played cf = play(fx, size, div, NVFX_INTERP_CROSSFADE, c, seed, frames);
      std::vector<std::uint8_t> between(sz(frames));
      for (int f = 0; f < frames; ++f) {
        between[sz(f)] = cf.key[sz(f)] ? 0 : 1;
        if (cf.key[sz(f)] && std::memcmp(cf.clip.frame(f).data(), ref.frame(f).data(), ref.frame_bytes()) != 0) {
          throw std::runtime_error(std::format("{} setting {}: keyframe {} at {} px / {} differs from the full-rate run", ef.name, si, f, size, div));
        }
      }
      for (const nvfx_interp m : kInterps) {
        if (m == NVFX_INTERP_CROSSFADE) {
          add(compare(ref, rs, cf.clip, between), "rate", size, div, interp_name(m));
          continue;
        }
        add(compare(ref, rs, play(fx, size, div, m, c, seed, frames).clip, between), "rate", size, div, interp_name(m));
      }
    }
  }
  // res: against 128 px at the size shown, and drawn at 128 px
  for (const int size : {32, 64}) {
    const Clip down = resized(ref128, size);
    add(compare(down, metrics::stats(down), full[size].clip, {}), "res", size, 1, "none");
    const Clip up = resized(full[size].clip, 128);
    add(compare(ref128, metrics::stats(ref128), up, {}), "res_up", size, 1, "none");
  }
  // proxy and floor at 64 px, against 128 px brought to 64
  {
    const Clip down = resized(ref128, 64);
    const metrics::ClipStats ds = metrics::stats(down);
    Setting snapped = c;
    for (float& x : snapped) x = (std::clamp(std::floor(x * 3.f), 0.f, 2.f) + 0.5f) / 3.f;
    for (const nvfx_interp m : {NVFX_INTERP_CROSSFADE, NVFX_INTERP_MOTION}) {
      const Played p = play(fx, 64, 4, m, snapped, seed + 70, frames);
      add(compare(down, ds, p.clip, {}), "proxy", 64, 4, interp_name(m));
    }
    const Played other = play(fx, 64, 1, NVFX_INTERP_CROSSFADE, c, seed + 50, frames);
    add(compare(down, ds, other.clip, {}), "floor", 64, 1, "none");
  }
  (void)e;
  return rows;
}

std::string fmt_interval(const metrics::Interval& i, int digits = 2) {
  const bool tie = i.covers_zero();
  return std::format("{:+.{}f} [{:+.{}f}, {:+.{}f}]{}", i.mean, digits, i.lo, digits, i.hi, digits, tie ? " (tie)" : "");
}

void summarise(const std::vector<Row>& rows, const std::string& split, const std::vector<EffectFile>& effects, const fs::path& out) {
  std::ofstream md(out);
  md << "# Effect manager: quality of the levels of detail (" << split << ")\n\n";
  md << "Written by `nvfx_lod " << split << "` (docs/MANAGER.md §5). Every number is a mean over the effects' ten settings; "
        "intervals are paired bootstrap 95% intervals over settings (10,000 resamples). Active PSNR per frame in dB.\n\n";
  // per cell: mean active PSNR per frame (between keyframes) and statistics, pooled over effects
  const auto find = [&](const std::string& e, int s, const char* fam, int size, int div, const std::string& interp) -> const Row* {
    for (const Row& r : rows) {
      if (r.effect == e && r.setting == s && r.family == fam && r.size == size && r.div == div && r.interp == interp) return &r;
    }
    return nullptr;
  };
  std::vector<std::pair<std::string, int>> pairs;
  for (const Row& r : rows) {
    if (r.family == "res" && r.size == 32) pairs.emplace_back(r.effect, r.setting);
  }
  md << "## Rate: frames between keyframes against the full-rate run at the same size\n\n";
  md << "| size | rate | interpolation | active PSNR per frame | spectrum L1 | motion ratio | mean-frame PSNR |\n|---:|---:|---|---:|---:|---:|---:|\n";
  for (const int size : {32, 64, 128}) {
    for (int div = 2; div <= 4; ++div) {
      for (const nvfx_interp m : kInterps) {
        std::vector<double> a, sp, mo, mf;
        for (const auto& [e, s] : pairs) {
          if (const Row* r = find(e, s, "rate", size, div, interp_name(m))) {
            a.push_back(r->apsnr_frames);
            sp.push_back(r->spectrum);
            mo.push_back(r->motion);
            mf.push_back(r->mean_frame);
          }
        }
        if (a.empty()) continue;
        const auto mean = [](const std::vector<double>& v) { return std::ranges::fold_left(v, 0.0, std::plus{}) / static_cast<double>(v.size()); };
        md << std::format("| {} | {:.1f} Hz | {} | {:.2f} | {:.3f} | {:.3f} | {:.2f} |\n", size, 30.0 / div, interp_name(m), mean(a), mean(sp), mean(mo), mean(mf));
      }
    }
  }
  // the rule's comparison: per (effect, setting), the mean over the six cells (64 and 128 px; 15, 10, 7.5 Hz)
  const auto cell_mean = [&](const std::string& e, int s, const std::string& interp, int what) {
    double v = 0;
    int n = 0;
    for (const int size : {64, 128}) {
      for (int div = 2; div <= 4; ++div) {
        if (const Row* r = find(e, s, "rate", size, div, interp)) {
          v += what == 0 ? r->apsnr_frames : what == 1 ? std::abs(std::log(r->motion)) : r->spectrum;
          ++n;
        }
      }
    }
    return n > 0 ? v / n : 0.0;
  };
  md << "\n## The comparisons of the rule (docs/MANAGER.md §5): mean over 64 and 128 px at 15, 10 and 7.5 Hz\n\n";
  md << "| comparison | effect | active PSNR per frame | motion error abs(ln ratio) | spectrum L1 |\n|---|---|---|---|---|\n";
  for (const auto& [x, y] : {std::pair<std::string, std::string>{"motion", "crossfade"}, {"crossfade", "hold"}, {"motion", "hold"}}) {
    std::vector<std::string> names;
    for (const auto& ef : effects) names.push_back(ef.name);
    names.push_back("pooled");
    for (const auto& name : names) {
      std::array<std::vector<double>, 3> a, b;
      for (const auto& [e, s] : pairs) {
        if (name != "pooled" && e != name) continue;
        for (int w = 0; w < 3; ++w) {
          a[sz(w)].push_back(cell_mean(e, s, x, w));
          b[sz(w)].push_back(cell_mean(e, s, y, w));
        }
      }
      if (a[0].empty()) continue;
      md << std::format("| {} - {} | {} | {} | {} | {} |\n", x, y, name, fmt_interval(metrics::paired_bootstrap(a[0], b[0], 10000, 1)),
                        fmt_interval(metrics::paired_bootstrap(a[1], b[1], 10000, 2), 3), fmt_interval(metrics::paired_bootstrap(a[2], b[2], 10000, 3), 3));
    }
  }
  md << "\n## Size: 32 and 64 px at 30 Hz against 128 px\n\n| size | shown at | active PSNR per frame | spectrum L1 | motion ratio | mean-frame PSNR |\n|---:|---|---:|---:|---:|---:|\n";
  for (const int size : {32, 64}) {
    for (const char* fam : {"res", "res_up"}) {
      std::vector<double> a, sp, mo, mf;
      for (const auto& [e, s] : pairs) {
        if (const Row* r = find(e, s, fam, size, 1, "none")) {
          a.push_back(r->apsnr_frames);
          sp.push_back(r->spectrum);
          mo.push_back(r->motion);
          mf.push_back(r->mean_frame);
        }
      }
      const auto mean = [](const std::vector<double>& v) { return std::ranges::fold_left(v, 0.0, std::plus{}) / static_cast<double>(v.size()); };
      md << std::format("| {} | {} | {:.2f} | {:.3f} | {:.3f} | {:.2f} |\n", size, std::string(fam) == "res" ? std::format("{} px (128 px averaged down)", size) : "128 px (bilinear up)",
                        mean(a), mean(sp), mean(mo), mean(mf));
    }
  }
  md << "\n## Proxies: 64 px, 7.5 Hz, snapped controls, another seed; against 128 px averaged to 64 (the floor: another seed at the same controls, 64 px, 30 Hz)\n\n";
  md << "| effect | method | spectrum L1 | motion ratio | mean-frame PSNR | proxy - floor: spectrum L1 | abs(ln motion) | mean-frame PSNR |\n|---|---|---:|---:|---:|---|---|---|\n";
  for (const auto& ef : effects) {
    for (const char* m : {"crossfade", "motion"}) {
      std::array<std::vector<double>, 3> p, f;
      for (const auto& [e, s] : pairs) {
        if (e != ef.name) continue;
        const Row* rp = find(e, s, "proxy", 64, 4, m);
        const Row* rf = find(e, s, "floor", 64, 1, "none");
        if (!rp || !rf) continue;
        p[0].push_back(rp->spectrum);
        p[1].push_back(std::abs(std::log(rp->motion)));
        p[2].push_back(rp->mean_frame);
        f[0].push_back(rf->spectrum);
        f[1].push_back(std::abs(std::log(rf->motion)));
        f[2].push_back(rf->mean_frame);
      }
      if (p[0].empty()) continue;
      const auto mean = [](const std::vector<double>& v) { return std::ranges::fold_left(v, 0.0, std::plus{}) / static_cast<double>(v.size()); };
      md << std::format("| {} | proxy, {} | {:.3f} | {:.3f} | {:.2f} | {} | {} | {} |\n", ef.name, m, mean(p[0]), std::exp(mean(p[1])), mean(p[2]),
                        fmt_interval(metrics::paired_bootstrap(p[0], f[0], 10000, 4), 3), fmt_interval(metrics::paired_bootstrap(p[1], f[1], 10000, 5), 3),
                        fmt_interval(metrics::paired_bootstrap(p[2], f[2], 10000, 6)));
      if (std::string(m) == "motion") {
        md << std::format("| {} | floor | {:.3f} | {:.3f} | {:.2f} | | | |\n", ef.name, mean(f[0]), std::exp(mean(f[1])), mean(f[2]));
      }
    }
  }
}

int run_split(const tools::Args& a, const std::string& split) {
  const fs::path root = a.str("root", std::getenv("NEURALVFX_DATA") ? std::getenv("NEURALVFX_DATA") : "/root/nvfx-data");
  const fs::path results = a.str("results", "results/manager");
  const auto effects = effect_files(root, a.str("effects", "fire,smoke,explosion,steam,magic"));
  const int threads = a.i("threads", 3), frames_loop = a.i("frames", 300);
  const Split sp = split_of(split);
  std::vector<nvfx_effect*> fx(effects.size(), nullptr);
  for (std::size_t e = 0; e < effects.size(); ++e) {
    if (nvfx_effect_load(effects[e].path.c_str(), &fx[e]) != NVFX_OK) throw std::runtime_error("cannot load " + effects[e].path.string());
  }
  struct Job {
    int e, s;
  };
  std::vector<Job> jobs;
  for (int e = 0; e < static_cast<int>(effects.size()); ++e) {
    for (int s = 0; s < static_cast<int>(sp.settings.size()); ++s) jobs.push_back({e, s});
  }
  std::vector<Row> rows;
  std::mutex mu;
  std::atomic<int> next{0}, done{0};
  {
    std::vector<std::jthread> pool;
    for (int t = 0; t < threads; ++t) {
      pool.emplace_back([&] {
        for (int j; (j = next++) < static_cast<int>(jobs.size());) {
          const Job job = jobs[sz(j)];
          nvfx_effect_info info{};
          nvfx_effect_get_info(fx[sz(job.e)], &info);
          const int frames = info.loops ? frames_loop : info.frames;
          const std::uint64_t seed = sp.base + 100 * static_cast<std::uint64_t>(job.s) + static_cast<std::uint64_t>(job.e);
          auto r = measure(effects[sz(job.e)], fx[sz(job.e)], job.e, job.s, sp.settings[sz(job.s)], seed, frames);
          std::lock_guard lk(mu);
          rows.insert(rows.end(), r.begin(), r.end());
          std::println("nvfx_lod {}: {} setting {} done ({} of {})", split, effects[sz(job.e)].name, job.s, ++done, jobs.size());
        }
      });
    }
  }
  std::ranges::sort(rows, [](const Row& x, const Row& y) {
    return std::tie(x.effect, x.setting, x.family, x.size, x.div, x.interp) < std::tie(y.effect, y.setting, y.family, y.size, y.div, y.interp);
  });
  fs::create_directories(results);
  {
    std::ofstream o(results / std::format("lod_{}.csv", split));
    o << "effect,setting,family,size,div,interp,apsnr_frames,apsnr_clip,spectrum_l1,motion_ratio,mean_frame_psnr,coverage_l1,frames_compared\n";
    for (const Row& r : rows) {
      o << std::format("{},{},{},{},{},{},{:.4f},{:.4f},{:.5f},{:.5f},{:.4f},{:.5f},{}\n", r.effect, r.setting, r.family, r.size, r.div, r.interp, r.apsnr_frames,
                       r.apsnr_clip, r.spectrum, r.motion, r.mean_frame, r.coverage, r.compared);
    }
  }
  summarise(rows, split, effects, results / std::format("lod_{}_summary.md", split));
  for (auto* f : fx) nvfx_effect_free(f);
  std::println("nvfx_lod {}: {} rows; {}", split, rows.size(), (results / std::format("lod_{}_summary.md", split)).string());
  return 0;
}

// A contact sheet of consecutive frames: the full-rate run at 128 px brought to the size, the full-rate run at the size,
// and the size at the rate held, crossfaded and motion-compensated.
int sheet(const tools::Args& a) {
  const fs::path root = a.str("root", std::getenv("NEURALVFX_DATA") ? std::getenv("NEURALVFX_DATA") : "/root/nvfx-data");
  const auto effects = effect_files(root, a.str("effect", "fire"));
  const int si = a.i("setting", 0), size = a.i("size", 64), div = a.i("div", 4), first = a.i("first", 60), cols = a.i("frames", 9);
  nvfx_effect* fx = nullptr;
  if (nvfx_effect_load(effects[0].path.c_str(), &fx) != NVFX_OK) throw std::runtime_error("cannot load");
  const Setting c = validation_settings()[sz(si)];
  const std::uint64_t seed = 7'300'000 + 100 * static_cast<std::uint64_t>(si);
  const int frames = first + cols;
  std::vector<Clip> rows;
  rows.push_back(resized(play(fx, 128, 1, NVFX_INTERP_CROSSFADE, c, seed, frames).clip, size));
  rows.push_back(play(fx, size, 1, NVFX_INTERP_CROSSFADE, c, seed, frames).clip);
  for (const nvfx_interp m : kInterps) rows.push_back(play(fx, size, div, m, c, seed, frames).clip);
  std::vector<Clip> cut;
  for (const Clip& r : rows) cut.push_back(slice_clip(r, first, cols));
  std::vector<const Clip*> ptr;
  for (const Clip& r : cut) ptr.push_back(&r);
  const fs::path out = root / "scale" / "figures" / std::format("lod_{}_{}px_div{}.png", effects[0].name, size, div);
  fs::create_directories(out.parent_path());
  const Image img = comparison_sheet(ptr, cols, Background::black, std::max(1, 128 / size));
  if (auto w = write_png(out, img); !w) throw std::runtime_error(w.error());
  std::println("rows: 128 px averaged to {0}; {0} px at 30 Hz; {0} px at {1:.1f} Hz held, crossfaded, motion-compensated; frames {2} to {3}: {4}", size,
               30.0 / div, first, first + cols - 1, out.string());
  nvfx_effect_free(fx);
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"help"});
  if (a.flag("help") || a.positional().empty()) {
    std::println("nvfx_lod val|test|sheet [--effects fire,smoke,explosion,steam,magic] [--threads 3] [--frames 300] [--root DATA] [--results DIR]");
    return 0;
  }
  const std::string cmd = a.positional()[0];
  if (cmd == "val" || cmd == "test") return run_split(a, cmd);
  if (cmd == "sheet") return sheet(a);
  throw std::invalid_argument("unknown command " + cmd);
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_lod: {}", e.what());
  return 2;
}
