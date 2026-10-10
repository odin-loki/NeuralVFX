// nvfx_prior: the prior against drift in the runtime (docs/DCM.md G2.13): its parity with study G's reference and its
// cost per frame.
//
//   nvfx_prior parity [--setting 0] [--frames 192] [--isa avx2|avx512|baseline] [--model F] [--prior F]
//       Study G's test run of the prior (G2.6): study B's held-out setting 1 (or 2), model seed 2,970,000 + setting, the
//       start point nearest the controls, 128 px, N = 16, t = 100, beta = 1. The reference (rollout::step and
//       dcm::ddpm::prior_step, applied as tools/experiment_g.cpp applies them) against the runtime (the runner and the
//       prior that nvfx_render uses), with and without the prior. Every 16 frames: the runtime's prior step on its own
//       state against the reference's prior step on the same state (floats that differ, largest difference), and how
//       far the two rollouts have drifted apart (coarse state in network units, pixels). The same frames through the
//       C API, when its start point for this seed is the nearest one (it picks among the three nearest by seed).
//   nvfx_prior time [--setting 0] [--size 128] [--seconds 60] [--repeats 5] [--core 3] [--isa avx2] [--every 16]
//       One instance through the C API: one continuous rollout (drift 0) with the prior and without, and the default
//       6 s shards. Thread CPU time of every frame on one pinned core, the least over the repeats for each frame (the
//       run is the same in every repeat); mean, median, p99 and max per frame, the frames where the prior acts apart,
//       and one prior pass alone (least of 200). Provisional unless the load average stays below 1.5.
// Defaults: --model NEURALVFX_DATA/experiments/models/d/fire.nvfx, --prior NEURALVFX_DATA/g/diff/fire.ddpm
// (NEURALVFX_DATA defaults to ~/nvfx-data).
#include "args.hpp"

#include "rt_common.hpp"
#include "rt_prior.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/dcm/ddpm.hpp>
#include <neuralfx/noise.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <print>
#include <random>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

using namespace nfx;
namespace dd = nfx::dcm::ddpm;
namespace fs = std::filesystem;

namespace {

constexpr int kSize = 128, kEvery = 16, kT = 100;
constexpr float kBeta = 1.f;
using Setting = std::array<float, 3>;

// Study B's held-out settings (tools/experiment_g.cpp, b_test_settings).
std::vector<Setting> b_test_settings() {
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < 10) {
    const Setting s{u(rng), u(rng), u(rng)};
    const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
    if (off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f})) v.push_back(s);
  }
  return v;
}

int nearest_start(const rollout::Model& m, std::span<const float> ctl) {
  int best = 0;
  float bd = 1e30f;
  for (std::size_t k = 0; k < m.starts.size(); ++k) {
    float d = 0;
    for (std::size_t q = 0; q < 3; ++q) d += (m.starts[k].controls[q] - ctl[q]) * (m.starts[k].controls[q] - ctl[q]);
    if (d < bd) {
      bd = d;
      best = static_cast<int>(k);
    }
  }
  return best;
}

// The start point the C API plays first for a seed (src/runtime/nvfx.cpp, pick_start for shard 0): one of the three
// nearest the controls, chosen by the seed.
int api_start(const rollout::Model& m, std::span<const float> ctl, std::uint64_t seed) {
  std::array<int, 3> best{-1, -1, -1};
  std::array<float, 3> dist{1e30f, 1e30f, 1e30f};
  for (std::size_t i = 0; i < m.starts.size(); ++i) {
    float d = 0.f;
    for (std::size_t k = 0; k < m.starts[i].controls.size() && k < ctl.size(); ++k) {
      const float e = m.starts[i].controls[k] - ctl[k];
      d += e * e;
    }
    for (std::size_t j = 0; j < best.size(); ++j) {
      if (d < dist[j]) {
        for (std::size_t q = best.size() - 1; q > j; --q) {
          best[q] = best[q - 1];
          dist[q] = dist[q - 1];
        }
        best[j] = static_cast<int>(i);
        dist[j] = d;
        break;
      }
    }
  }
  const int count = static_cast<int>(std::min<std::size_t>(3, m.starts.size()));
  return best[hash_cell(0, 0, 17, seed) % static_cast<std::uint32_t>(count)];
}

// The reference prior on a coarse state, as tools/experiment_g.cpp's long_run applies it (state_to_x, prior_step,
// x_to_state).
void reference_prior(const dd::Denoiser& d, const rollout::Model& m, std::span<const float> ctl, float time, std::span<float> coarse) {
  std::vector<float> x(static_cast<std::size_t>(d.cfg.res * d.cfg.res * d.cfg.channels)), cond(static_cast<std::size_t>(m.h.cond()));
  dd::to_network(d, coarse, m.h.channels(), x);
  rollout::condition(m, ctl, time, cond);
  dd::prior_step(d, x, kT, kBeta, cond);
  dd::to_physical(d, x, m.h.channels(), coarse);
  const int C = m.h.channels();
  for (std::size_t i = 0; i < coarse.size(); ++i) {
    const auto k = i % static_cast<std::size_t>(C);
    if (k < rollout::kPhys) coarse[i] = std::clamp(coarse[i], m.lo[k], m.hi[k]);
  }
}

std::vector<std::uint8_t> reference_pixels(const rollout::Model& m, const rollout::State& s) {
  std::vector<float> rgba(static_cast<std::size_t>(kSize) * kSize * 4);
  rollout::render(m, s, rgba);
  std::vector<std::uint8_t> out(rgba.size());
  for (std::size_t i = 0; i < rgba.size(); ++i) out[i] = static_cast<std::uint8_t>(rgba[i] * 255.f + 0.5f);
  return out;
}

struct Isa {
  nvfx_isa id = NVFX_ISA_AVX2;
  std::string name = "avx2";
};
Isa parse_isa(const std::string& s) {
  if (s == "avx2") return {NVFX_ISA_AVX2, s};
  if (s == "avx512") return {NVFX_ISA_AVX512, s};
  if (s == "baseline") return {NVFX_ISA_BASELINE, s};
  throw std::invalid_argument("--isa is avx2, avx512 or baseline");
}
std::unique_ptr<rt::RolloutRunner> make_runner(const Isa& isa, const rt::RolloutEffect& e) {
  if (isa.id == NVFX_ISA_AVX512) return rt::isa_avx512::make_rollout(e, kSize);
  if (isa.id == NVFX_ISA_BASELINE) return rt::isa_base::make_rollout(e, kSize);
  return rt::isa_avx2::make_rollout(e, kSize);
}
std::unique_ptr<rt::Prior> make_prior(const Isa& isa, const rt::PriorNet& n) {
  if (isa.id == NVFX_ISA_AVX512) return rt::isa_avx512::make_prior(n);
  if (isa.id == NVFX_ISA_BASELINE) return rt::isa_base::make_prior(n);
  return rt::isa_avx2::make_prior(n);
}

std::string read_file(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + p.string());
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

double load_average() {
  std::ifstream f("/proc/loadavg");
  double v = 99;
  f >> v;
  return v;
}

double thread_ms() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) * 1e3 + static_cast<double>(ts.tv_nsec) * 1e-6;
}

struct Diff {
  std::size_t differ = 0;  // floats whose bits differ
  double max = 0;          // largest absolute difference
};
Diff diff(std::span<const float> a, std::span<const float> b, int stride = 1, std::span<const float> unit = {}) {
  Diff d;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i])) ++d.differ;
    const std::size_t k = i % static_cast<std::size_t>(stride);
    const double u = k < unit.size() ? static_cast<double>(unit[k]) : 1.0;
    d.max = std::max(d.max, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])) / u);
  }
  return d;
}

struct PixelDiff {
  int max = 0;
  double mean = 0, off2 = 0;  // mean absolute difference (levels of 255); share of values more than 2 levels apart
};
PixelDiff pixel_diff(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
  PixelDiff p;
  double sum = 0, off = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const int d = std::abs(int(a[i]) - int(b[i]));
    p.max = std::max(p.max, d);
    sum += d;
    off += d > 2;
  }
  p.mean = sum / static_cast<double>(a.size());
  p.off2 = off / static_cast<double>(a.size());
  return p;
}

int parity(const tools::Args& a, const rollout::Model& m, const dd::Denoiser& d, const std::string& prior_bytes) {
  const int setting = a.i("setting", 0), frames = a.i("frames", 192);
  const Isa isa = parse_isa(a.str("isa", "avx2"));
  if (nvfx_set_isa(isa.id) != NVFX_OK) throw std::runtime_error("this CPU cannot run " + isa.name);
  const Setting s = b_test_settings().at(static_cast<std::size_t>(setting));
  const std::vector<float> ctl(s.begin(), s.end());
  const std::uint64_t seed = 2970000 + static_cast<std::uint64_t>(setting);
  const int idx = nearest_start(m, ctl);
  auto net = rt::parse_prior(prior_bytes);
  if (!net) throw std::runtime_error(net.error());
  rt::RolloutEffect re;
  re.m = m;
  const auto prior = make_prior(isa, *net);
  std::println("nvfx_prior parity: setting {} ({:.3f} / {:.3f} / {:.3f}), seed {}, start point {}, {} frames, runtime {}; prior every {} frames, t = {}, "
               "beta = {}", setting + 1, s[0], s[1], s[2], seed, idx, frames, isa.name, kEvery, kT, kBeta);
  // The C API: its start point for this seed (one of the three nearest), and whether it is the nearest.
  nvfx_effect* fx = nullptr;
  nvfx_instance* inst = nullptr;
  {
    std::ostringstream os;
    if (auto r = rollout::save_model(os, m); !r) throw std::runtime_error(r.error());
    const std::string b = os.str();
    if (nvfx_effect_load_memory(b.data(), b.size(), &fx) != NVFX_OK) throw std::runtime_error("runtime load failed");
    if (nvfx_effect_attach_prior_memory(fx, prior_bytes.data(), prior_bytes.size()) != NVFX_OK) throw std::runtime_error("attach failed");
    if (nvfx_instance_create(fx, kSize, &inst) != NVFX_OK) throw std::runtime_error("instance");
    nvfx_instance_set_controls(inst, ctl.data(), 3);
    nvfx_instance_set_seed(inst, seed);
    nvfx_instance_set_drift(inst, 0.f);
  }
  std::vector<std::uint8_t> api(static_cast<std::size_t>(kSize) * kSize * 4), rt_px(api.size());
  rt::FrameInput fi;
  const int aidx = api_start(m, ctl, seed);
  std::vector<float> cond(static_cast<std::size_t>(m.h.cond()));
  for (const bool with : {true, false}) {
    nvfx_instance_set_prior(inst, with ? kEvery : 0, kT, kBeta);
    rollout::State ref = rollout::start(m, idx, kSize, ctl, seed);
    const auto run = make_runner(isa, re);
    run->start(idx, ctl, seed);
    // The C API's own run (from its start point) against the same runner and prior driven by hand: equal bit for bit
    // when nvfx_render applies the prior where study G did.
    const auto twin = make_runner(isa, re);
    twin->start(aidx, ctl, seed);
    std::println("\n{} the prior. The C API starts from start point {} for this seed; its frames are compared with a runner started there.",
                 with ? "With" : "Without", aidx);
    std::println("frame | prior step on the runtime's state: floats differing / largest (network units) | rollouts apart: coarse, largest (network units) "
                 "| pixels: largest, mean, share > 2 levels | C API = runner and prior by hand");
    for (int f = 1; f <= frames; ++f) {
      rollout::step(m, ref, ctl, seed);
      run->step(ctl, seed);
      twin->step(ctl, seed);
      std::string step_text = "-";
      if (with && f % kEvery == 0) {
        rollout::condition(m, ctl, twin->time(), cond);
        prior->apply(twin->coarse_mut(), m.h.channels(), m.lo, m.hi, kT, kBeta, cond);
        // the same state through both priors
        const auto own = run->coarse();
        std::vector<float> theirs(own.begin(), own.end());
        reference_prior(d, m, ctl, run->time(), theirs);
        rollout::condition(m, ctl, run->time(), cond);
        prior->apply(run->coarse_mut(), m.h.channels(), m.lo, m.hi, kT, kBeta, cond);
        const Diff pd = diff(run->coarse(), theirs, m.h.channels(), d.scale);
        step_text = std::format("{} / {:.3g}", pd.differ, pd.max);
        reference_prior(d, m, ctl, ref.time, ref.coarse);  // the reference rollout's own prior
      }
      if (f % kEvery == 0 || f == frames) {
        const Diff cd = diff(run->coarse(), ref.coarse, m.h.channels(), d.scale);
        run->render(fi, rt_px.data(), kSize * 4);
        const PixelDiff pdx = pixel_diff(rt_px, reference_pixels(m, ref));
        nvfx_render(inst, f / static_cast<double>(m.fps), api.data(), kSize * 4);
        twin->render(fi, rt_px.data(), kSize * 4);
        std::println("{:5} | {} | {:.3g} | {}, {:.3f}, {:.4f} | {}", f, step_text, cd.max, pdx.max, pdx.mean, pdx.off2, api == rt_px ? "yes" : "NO");
      }
    }
  }
  nvfx_instance_free(inst);
  nvfx_effect_free(fx);
  nvfx_set_isa(NVFX_ISA_AUTO);
  return 0;
}

struct Stats {
  double mean = 0, median = 0, p99 = 0, max = 0, prior_mean = 0, other_mean = 0;
  int prior_frames = 0;
};
Stats stats(const std::vector<double>& ms, int every) {
  Stats s;
  std::vector<double> v(ms.begin() + 1, ms.end());  // frame 0 (the start point and its warm-up) apart
  s.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
  double pm = 0, om = 0;
  int other = 0;
  for (std::size_t f = 1; f < ms.size(); ++f) {
    if (every > 0 && f % static_cast<std::size_t>(every) == 0) {
      pm += ms[f];
      ++s.prior_frames;
    } else {
      om += ms[f];
      ++other;
    }
  }
  s.prior_mean = s.prior_frames ? pm / s.prior_frames : 0;
  s.other_mean = other ? om / other : 0;
  std::ranges::sort(v);
  s.median = v[v.size() / 2];
  s.p99 = v[std::min(v.size() - 1, static_cast<std::size_t>(std::ceil(0.99 * static_cast<double>(v.size()))) - 1)];
  s.max = v.back();
  return s;
}

int timing(const tools::Args& a, const rollout::Model& m, const std::string& model_bytes, const std::string& prior_bytes) {
  const int setting = a.i("setting", 0), size = a.i("size", kSize), repeats = a.i("repeats", 5), core = a.i("core", 3), every = a.i("every", kEvery);
  const double seconds = a.f("seconds", 60.f);
  const Isa isa = parse_isa(a.str("isa", "avx2"));
  if (nvfx_set_isa(isa.id) != NVFX_OK) throw std::runtime_error("this CPU cannot run " + isa.name);
  cpu_set_t one;
  CPU_ZERO(&one);
  CPU_SET(core, &one);
  if (sched_setaffinity(0, sizeof(one), &one) != 0) throw std::runtime_error("cannot pin to the core");
  const double before = load_average();
  const Setting s = b_test_settings().at(static_cast<std::size_t>(setting));
  nvfx_effect* fx = nullptr;
  if (nvfx_effect_load_memory(model_bytes.data(), model_bytes.size(), &fx) != NVFX_OK) throw std::runtime_error("runtime load failed");
  if (nvfx_effect_attach_prior_memory(fx, prior_bytes.data(), prior_bytes.size()) != NVFX_OK) throw std::runtime_error("attach failed");
  const int frames = static_cast<int>(std::lround(seconds * static_cast<double>(m.fps))) + 1;
  std::vector<std::uint8_t> px(static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4);
  std::println("nvfx_prior time: {} x {} px, {} frames ({:.0f} s), setting {}, seed {}, {} repeats, core {}, runtime {}", size, size, frames, seconds,
               setting + 1, 2970000 + setting, repeats, core, isa.name);
  struct Variant {
    const char* name;
    float drift;
    int every;
  };
  std::size_t scratch_prior = 0, scratch_plain = 0;
  for (const Variant v : {Variant{"continuous with the prior", 0.f, every}, Variant{"continuous without the prior", 0.f, 0}, Variant{"6 s shards (default)", 6.f, 0}}) {
    std::vector<double> best(static_cast<std::size_t>(frames), 1e30);
    for (int r = 0; r < repeats; ++r) {
      nvfx_instance* in = nullptr;
      if (nvfx_instance_create(fx, size, &in) != NVFX_OK) throw std::runtime_error("instance");
      nvfx_instance_set_controls(in, s.data(), 3);
      nvfx_instance_set_seed(in, 2970000 + static_cast<std::uint64_t>(setting));
      nvfx_instance_set_drift(in, v.drift);
      nvfx_instance_set_prior(in, v.every, kT, kBeta);
      if (v.every > 0) scratch_prior = nvfx_instance_scratch_bytes(in);
      for (int f = 0; f < frames; ++f) {
        const double c0 = thread_ms();
        nvfx_render(in, f / static_cast<double>(m.fps), px.data(), static_cast<std::size_t>(size) * 4);
        best[static_cast<std::size_t>(f)] = std::min(best[static_cast<std::size_t>(f)], thread_ms() - c0);
      }
      nvfx_instance_free(in);
    }
    const Stats st = stats(best, v.every);
    std::println("{:30}: frame 0 (start, warm-up) {:.2f} ms; frames 1 to {}: mean {:.3f} ms, median {:.3f}, p99 {:.3f}, max {:.3f}", v.name, best[0], frames - 1,
                 st.mean, st.median, st.p99, st.max);
    if (v.every > 0) {
      std::println("{:30}  the {} frames where the prior acts: mean {:.3f} ms; the others: mean {:.3f} ms", "", st.prior_frames, st.prior_mean, st.other_mean);
    }
  }
  {
    nvfx_instance* in = nullptr;
    nvfx_instance_create(fx, size, &in);
    nvfx_instance_set_prior(in, 0, kT, kBeta);
    scratch_plain = nvfx_instance_scratch_bytes(in);
    nvfx_instance_free(in);
  }
  // one pass alone, through the runtime's prior on a real state
  auto net = rt::parse_prior(prior_bytes);
  const auto prior = make_prior(isa, *net);
  rt::RolloutEffect re;
  re.m = m;
  const auto run = make_runner(isa, re);
  const std::vector<float> ctl(s.begin(), s.end());
  run->start(nearest_start(m, ctl), ctl, 2970000 + static_cast<std::uint64_t>(setting));
  std::vector<float> state(run->coarse().begin(), run->coarse().end()), cond(static_cast<std::size_t>(m.h.cond()));
  rollout::condition(m, ctl, run->time(), cond);
  std::vector<double> pass;
  for (int i = 0; i < 210; ++i) {
    std::vector<float>& x = state;
    std::ranges::copy(run->coarse(), x.begin());
    const double c0 = thread_ms();
    prior->apply(x, m.h.channels(), m.lo, m.hi, kT, kBeta, cond);
    if (i >= 10) pass.push_back(thread_ms() - c0);
  }
  std::ranges::sort(pass);
  const double after = load_average();
  std::println("one prior pass ({:.1f} M multiply-adds): least {:.3f} ms, median {:.3f} ms of {} ({:.1f} GMAC/s at the least)", net->macs() / 1e6, pass.front(),
               pass[pass.size() / 2], pass.size(), net->macs() / pass.front() / 1e6);
  std::println("instance scratch: {:.2f} MB with the prior's buffers, {:.2f} MB without; prior weights {:.2f} MB resident (file {:.2f} MB)",
               static_cast<double>(scratch_prior) / 1e6, static_cast<double>(scratch_plain) / 1e6, static_cast<double>(net->resident_bytes()) / 1e6,
               static_cast<double>(net->file_bytes) / 1e6);
  std::println("load average {:.2f} before, {:.2f} after: {}", before, after,
               before < 1.5 && after < 1.5 ? "quiet" : "busy machine: provisional (thread CPU time, least of the repeats)");
  nvfx_effect_free(fx);
  nvfx_set_isa(NVFX_ISA_AUTO);
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"help"});
  if (a.flag("help") || a.positional().empty()) {
    std::println("nvfx_prior parity|time [--setting 0] [--model fire.nvfx] [--prior fire.ddpm] [--isa avx2]  (options: see the source's header)");
    return 0;
  }
  const fs::path model_path = a.has("model") ? fs::path(a.str("model")) : data_root() / "experiments" / "models" / "d" / "fire.nvfx";
  const fs::path prior_path = a.has("prior") ? fs::path(a.str("prior")) : data_root() / "g" / "diff" / "fire.ddpm";
  auto m = rollout::load_model(model_path);
  if (!m) throw std::runtime_error(model_path.string() + ": " + m.error());
  auto d = dd::load(prior_path);
  if (!d) throw std::runtime_error(prior_path.string() + ": " + d.error());
  const std::string prior_bytes = read_file(prior_path);
  std::println("denoiser {} (version {}), {} parameters", prior_path.string(), dd::version(*d).substr(0, 16), d->parameters());
  int rc = 0;
  const std::string& cmd = a.positional()[0];
  if (cmd == "parity") {
    rc = parity(a, *m, *d, prior_bytes);
  } else if (cmd == "time") {
    rc = timing(a, *m, read_file(model_path), prior_bytes);
  } else {
    throw std::invalid_argument("unknown command (nvfx_prior --help)");
  }
  a.warn_unused();
  return rc;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_prior: {}", e.what());
  return 2;
}
