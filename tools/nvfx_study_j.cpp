// nvfx_study_j: study J (docs/FOOTAGE.md). Start points estimated from frames, measured where the truth is known: the
// stand-in footage is the simulator's own frames (and the learned renderer's), with the true states beside them.
//
//   nvfx_study_j inverse       train the inverse networks, per effect with alpha and without     -> j_inverse.csv
//   nvfx_study_j motion        train the motion network inside each inverse file                -> j_motion.csv
//   nvfx_study_j val           validation (salt-3 runs): oracles, the estimator's choices, every footage condition
//                                                                                                 -> j_val.csv
//   nvfx_study_j val-endless   endless statistics at validation settings                        -> j_val_endless.csv
//   nvfx_study_j test          the test, once (salt-2 runs, study B's held-out settings)        -> j_test.csv,
//                                                                                                    j_test_endless.csv
//   nvfx_study_j report        tables with paired bootstrap intervals                           -> j_summary.md
//
// Options: --models DIR (fire.nvfx, smoke.nvfx, explosion.nvfx; default <data root>/j/models), --data DIR (default
// <data root>/j: inverse networks, scratch), --results DIR (results/experiments), --effects fire,smoke, --threads 1,
// --runs N (per effect; default 12 for val, 16 for test), --quick (2 runs, short).
#include "args.hpp"

#include <neuralfx/footage.hpp>
#include <neuralfx/image_io.hpp>
#include <neuralfx/ingest.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <print>
#include <random>
#include <set>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;
using namespace nfx;

namespace {

constexpr int kSize = 128, kRes = 32, kCtx = 8, kFine = 64;
const std::vector<int> kHorizons = {0, 1, 8, 30, 60};
constexpr int kLast = 60;

std::size_t sz(int v) { return static_cast<std::size_t>(v); }

struct Ctx {
  fs::path models, data, results;
  std::string effects;
  int threads = 1;
  int runs = 0;
  bool quick = false;
};

bool wanted(const Ctx& c, sim::Effect e) {
  if (c.effects.empty()) return true;
  const std::string name(sim::effect_name(e));
  std::size_t a = 0;
  while (a <= c.effects.size()) {
    const std::size_t b = std::min(c.effects.find(',', a), c.effects.size());
    if (c.effects.substr(a, b - a) == name) return true;
    a = b + 1;
  }
  return false;
}

std::string ename(sim::Effect e) { return std::string(sim::effect_name(e)); }
int start_frame(sim::Effect e) { return e == sim::Effect::explosion ? 8 : 100; }

rollout::Model load_effect(const Ctx& c, sim::Effect e) {
  auto m = rollout::load_model(c.models / (ename(e) + ".nvfx"));
  if (!m) throw std::runtime_error(m.error());
  return std::move(*m);
}

footage::Inverse load_inv(const Ctx& c, sim::Effect e, bool alpha) {
  auto inv = footage::load_inverse(c.data / "inverse" / std::format("{}_{}.nvfxinv", ename(e), alpha ? "rgba" : "rgb"));
  if (!inv) throw std::runtime_error(inv.error() + " (run nvfx_study_j inverse first)");
  return std::move(*inv);
}

double apsnr(std::span<const std::uint8_t> r, std::span<const std::uint8_t> t) {  // as study D
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

void write_csv(const fs::path& path, const std::string& header, const std::vector<std::string>& rows) {
  fs::create_directories(path.parent_path());
  std::ofstream o(path);
  o << header << "\n";
  for (const auto& r : rows) o << r << "\n";
}

// --- the runtime --------------------------------------------------------------------------------------------------

struct Effect {
  nvfx_effect* e = nullptr;
  explicit Effect(const rollout::Model& m) {
    std::ostringstream os;
    if (auto r = rollout::save_model(os, m); !r) throw std::runtime_error(r.error());
    const std::string b = os.str();
    if (nvfx_effect_load_memory(b.data(), b.size(), &e) != NVFX_OK) throw std::runtime_error("runtime load failed");
  }
  ~Effect() { nvfx_effect_free(e); }
  Effect(const Effect&) = delete;
  Effect& operator=(const Effect&) = delete;
};

// As study D: controls, then a start point (variation >= 0 keeps its run's seed) or a seed; shards or one rollout.
Clip runtime_clip(nvfx_effect* e, std::span<const float> controls, int variation, std::uint64_t seed, int frames, bool shards) {
  nvfx_instance* in = nullptr;
  if (nvfx_instance_create(e, kSize, &in) != NVFX_OK) throw std::runtime_error("instance");
  nvfx_instance_set_controls(in, controls.data(), static_cast<int>(controls.size()));
  if (!shards) nvfx_instance_set_drift(in, 0.f);
  if (variation >= 0) nvfx_instance_set_variation(in, variation);
  else nvfx_instance_set_seed(in, seed);
  Clip c;
  c.allocate(kSize, frames);
  c.fps = 30.f;
  for (int f = 0; f < frames; ++f) nvfx_render(in, f / 30.0, c.frame(f).data(), kSize * 4);
  nvfx_instance_free(in);
  return c;
}

// One continuous rollout from a single start point (its seed and time), as the tool would store it.
Clip track(const rollout::Model& M, const rollout::StartPoint& sp, std::span<const float> controls, int frames) {
  const rollout::Model m = footage::with_starts(M, std::span(&sp, 1), kFine, false);
  Effect fx(m);
  return runtime_clip(fx.e, controls, 0, 0, frames, false);
}

// --- held-out runs with their truth ------------------------------------------------------------------------------

struct Truth {
  sim::Params p;
  float time = 0;                                  // state time at the start frame
  std::vector<std::vector<std::uint8_t>> ctx;      // kCtx clean frames, oldest first; the last is the start frame
  std::vector<footage::Fields> fields;             // true fine fields at those frames
  std::vector<std::vector<float>> coarse;          // true coarse states (res^2 * kPhys) at those frames
  Clip ref;                                        // clean frames 1 .. kLast steps after the start
};

footage::Fields fields_of(const sim::State& st) {
  footage::Fields f;
  f.size = st.n;
  f.heat = st.temp;
  f.soot = st.soot;
  return f;
}

Truth simulate_run(const sim::Params& p, int t0) {
  Truth t;
  t.p = p;
  sim::Fluid f(p);
  std::vector<std::uint8_t> rgba(sz(kSize) * kSize * 4);
  for (int i = 1; i <= t0; ++i) {
    f.step_frame();
    if (i > t0 - kCtx) {
      f.render(rgba);
      t.ctx.push_back(rgba);
      const sim::State st = f.state();
      t.fields.push_back(fields_of(st));
      std::vector<float> c(sz(kRes) * kRes * rollout::kPhys);
      rollout::coarse_from_sim(st, kRes, p.fps, c);
      t.coarse.push_back(std::move(c));
      if (i == t0) t.time = st.time;
    }
  }
  t.ref.allocate(kSize, kLast);
  t.ref.fps = 30.f;
  for (int i = 0; i < kLast; ++i) {
    f.step_frame();
    f.render(t.ref.frame(i));
  }
  return t;
}

// --- stand-in footage --------------------------------------------------------------------------------------------

enum class Cond { clean_rgba, clean_rgb, blur_noise_rgb, h264_rgb, learned_rgba };
const std::vector<Cond> kConds = {Cond::clean_rgba, Cond::clean_rgb, Cond::blur_noise_rgb, Cond::h264_rgb, Cond::learned_rgba};
std::string cname(Cond c) {
  switch (c) {
    case Cond::clean_rgba: return "clean_rgba";
    case Cond::clean_rgb: return "clean_rgb";
    case Cond::blur_noise_rgb: return "blur_noise_rgb";
    case Cond::h264_rgb: return "h264_rgb";
    case Cond::learned_rgba: return "learned_rgba";
  }
  return "?";
}
bool has_alpha(Cond c) { return c == Cond::clean_rgba || c == Cond::learned_rgba; }

// Frames through H.264 (libx264, 4:2:0, CRF 23) as colour over black, read back through the ingest path (alpha: luma).
std::vector<std::vector<std::uint8_t>> through_h264(const std::vector<std::vector<std::uint8_t>>& frames, const fs::path& path) {
  fs::create_directories(path.parent_path());
  const std::string cmd = std::format(
      "ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgb24 -s {}x{} -r 30 -i - -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \"{}\"", kSize,
      kSize, path.string());
  FILE* p = popen(cmd.c_str(), "w");
  if (!p) throw std::runtime_error("cannot start ffmpeg");
  std::vector<std::uint8_t> rgb(sz(kSize) * kSize * 3);
  for (const auto& f : frames) {
    for (std::size_t i = 0; i < sz(kSize) * kSize; ++i) std::copy_n(f.begin() + static_cast<std::ptrdiff_t>(i * 4), 3, rgb.begin() + static_cast<std::ptrdiff_t>(i * 3));
    if (std::fwrite(rgb.data(), 1, rgb.size(), p) != rgb.size()) throw std::runtime_error("ffmpeg pipe");
  }
  if (pclose(p) != 0) throw std::runtime_error("ffmpeg failed");
  ingest::Options o;
  o.size = kSize;
  o.fps = 30.f;
  o.alpha = ingest::AlphaMode::luma;
  o.max_frames = static_cast<int>(frames.size());
  auto clip = ingest::from_video(path, o);
  fs::remove(path);
  if (!clip) throw std::runtime_error(clip.error());
  if (clip->frames != static_cast<int>(frames.size())) throw std::runtime_error("h264: frame count changed");
  std::vector<std::vector<std::uint8_t>> out;
  for (int i = 0; i < clip->frames; ++i) out.emplace_back(clip->frame(i).begin(), clip->frame(i).end());
  return out;
}

std::vector<footage::Frame> footage_of(const Truth& t, Cond c, const rollout::Model& M, const fs::path& scratch, std::uint64_t seed) {
  std::vector<std::vector<std::uint8_t>> frames = t.ctx;
  switch (c) {
    case Cond::clean_rgba: break;
    case Cond::clean_rgb:
      for (auto& f : frames) footage::degrade(f, kSize, {0.f, 0.f, true}, 0);
      break;
    case Cond::blur_noise_rgb:
      for (std::size_t i = 0; i < frames.size(); ++i) footage::degrade(frames[i], kSize, {1.f, 2.f / 255.f, true}, seed * 31 + i);
      break;
    case Cond::h264_rgb: frames = through_h264(frames, scratch / std::format("h264_{}.mp4", seed)); break;
    case Cond::learned_rgba:
      for (std::size_t i = 0; i < frames.size(); ++i) {
        std::vector<float> rgba(sz(kSize) * kSize * 4);
        footage::render_fields(M, t.fields[i], rgba);
        footage::to_u8(rgba, frames[i]);
      }
      break;
  }
  std::vector<footage::Frame> out;
  for (const auto& f : frames) out.push_back(footage::to_float(f));
  return out;
}

// --- estimator configurations -----------------------------------------------------------------------------------

enum class Vel { zero, flow, assim, truth };
struct Config {
  std::string name;
  int K = 8;
  bool refine = true;
  Vel vel = Vel::assim;
  float beta = 0.5f;
  int pairs = 1;
  bool fields_truth = false;
  bool controls_given = true;
  bool run_seed = false;
  bool motion = false;  // the motion network's velocity in place of block matching where three frames are at hand
};

// What every configuration of one run and condition shares: the inverse's fields of the context frames, the start
// frame's fields refined, and the raw block-matched flows between consecutive frames (estimated and true fields).
struct Shared {
  std::vector<footage::Fields> est;
  footage::Fields refined;
  std::vector<footage::Flow> raw, raw_true;
  std::vector<footage::Flow> motion;  // the motion network's velocity at each frame (from the third on)
  double refine_s = 0, inverse_s = 0;
};

footage::FlowOptions flow_options() { return {}; }

std::vector<footage::Flow> true_flows(const rollout::Model& M, const Truth& t) {
  std::vector<footage::Flow> raw(t.fields.size());
  for (std::size_t i = 1; i < t.fields.size(); ++i) raw[i] = footage::block_flow(t.fields[i - 1], t.fields[i], kRes, M.render_scale, flow_options());
  return raw;
}

Shared prepare(const rollout::Model& M, const footage::Inverse& inv, const std::vector<footage::Frame>& frames, const std::vector<footage::Flow>* raw_true,
               bool with_refine) {
  Shared s;
  auto t0 = std::chrono::steady_clock::now();
  for (const auto& f : frames) s.est.push_back(footage::apply_inverse(inv, f, kSize));
  s.inverse_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  s.raw.resize(frames.size());
  for (std::size_t i = 1; i < frames.size(); ++i) s.raw[i] = footage::block_flow(s.est[i - 1], s.est[i], kRes, inv.scale, flow_options());
  s.motion.resize(frames.size());
  if (!inv.motion.empty()) {
    for (std::size_t i = 2; i < frames.size(); ++i) {
      s.motion[i] = footage::apply_motion(inv.motion, footage::motion_inputs(inv.motion, std::span<const footage::Fields>(s.est).subspan(i - 2, 3), s.raw[i - 1], s.raw[i], kRes), kRes);
    }
  }
  if (raw_true) s.raw_true = *raw_true;
  if (with_refine) {
    t0 = std::chrono::steady_clock::now();
    s.refined = s.est.back();
    footage::RefineOptions ro;
    ro.alpha = inv.spec.alpha;
    footage::refine_fields(M, frames.back(), s.refined, ro);
    s.refine_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  return s;
}

// The start point of a configuration (seed and time of the run, controls of the run: tracking uses them for every
// method), and the fine fields it estimated at frame size.
std::pair<rollout::StartPoint, footage::Fields> config_start(const rollout::Model& M, const Config& cf, const Truth& t, const Shared& s) {
  const int K = std::clamp(cf.K, 1, kCtx);
  std::vector<footage::Fields> fields;
  for (int i = kCtx - K; i < kCtx; ++i) fields.push_back(cf.fields_truth ? t.fields[sz(i)] : s.est[sz(i)]);
  if (!cf.fields_truth && cf.refine) fields.back() = s.refined;
  const std::vector<footage::Flow>& raw = cf.fields_truth ? s.raw_true : s.raw;
  std::vector<footage::Flow> flows(sz(K));
  for (int i = 1; i < K; ++i) {
    const int g = kCtx - K + i, first = std::max(kCtx - K + 1, g - std::max(1, cf.pairs) + 1);
    flows[sz(i)] = footage::mean_flow(std::span<const footage::Flow>(raw).subspan(sz(first), sz(g - first + 1)));
    footage::finish_flow(flows[sz(i)], flow_options());
    if (cf.motion && i >= 2 && !cf.fields_truth && s.motion[sz(g)].res == kRes) flows[sz(i)] = s.motion[sz(g)];
  }
  const std::vector<float> run_controls{t.p.intensity, t.p.wind, t.p.turbulence};
  std::vector<float> coarse(sz(kRes) * kRes * rollout::kPhys, 0.f);
  const std::vector<float> cf2 = footage::coarse_fields(fields.back(), kRes);
  for (int j = 0; j < kRes * kRes; ++j) {
    coarse[sz(j) * 4 + 2] = cf2[sz(j) * 2];
    coarse[sz(j) * 4 + 3] = cf2[sz(j) * 2 + 1];
  }
  switch (cf.vel) {
    case Vel::zero: break;
    case Vel::truth:
      for (int j = 0; j < kRes * kRes; ++j) {
        coarse[sz(j) * 4] = t.coarse.back()[sz(j) * 4];
        coarse[sz(j) * 4 + 1] = t.coarse.back()[sz(j) * 4 + 1];
      }
      break;
    case Vel::flow:
      if (K >= 2) {
        for (int j = 0; j < kRes * kRes; ++j) {
          coarse[sz(j) * 4] = flows.back().uv[sz(j) * 2];
          coarse[sz(j) * 4 + 1] = flows.back().uv[sz(j) * 2 + 1];
        }
      }
      break;
    case Vel::assim: {
      footage::AssimOptions ao;
      ao.use_flow = true;
      ao.nudge_velocity = cf.beta;
      ao.controls = cf.controls_given ? run_controls : std::vector<float>{0.5f, 0.5f, 0.5f};
      ao.time = t.time;
      ao.seed = cf.run_seed ? t.p.seed : 1;
      coarse = footage::assimilate(M, fields, flows, ao);
      break;
    }
  }
  return {footage::make_start(std::move(coarse), fields.back(), run_controls, t.p.seed, t.time), fields.back()};
}

// The stored start point nearest the run's controls, as the runtime picks it, replayed with the run's seed. Looping
// effects: timed so the noise lines up with the run (fire's warm-up included); one-shot effects keep their age, and
// `offset` frames are skipped so the ages line up.
std::pair<rollout::StartPoint, int> stored_start(const rollout::Model& M, const Truth& t) {
  const std::vector<float> c{t.p.intensity, t.p.wind, t.p.turbulence};
  std::size_t best = 0;
  float bd = 1e30f;
  for (std::size_t i = 0; i < M.starts.size(); ++i) {
    float d = 0;
    for (std::size_t k = 0; k < c.size(); ++k) d += (M.starts[i].controls[k] - c[k]) * (M.starts[i].controls[k] - c[k]);
    if (d < bd) {
      bd = d;
      best = i;
    }
  }
  rollout::StartPoint sp = M.starts[best];
  sp.seed = t.p.seed;
  int offset = 0;
  if (M.loop) {
    sp.time = t.time - (sp.fine_t.empty() ? static_cast<float>(M.h.warmup) / M.fps : 0.f);
  } else {
    offset = static_cast<int>(std::lround((t.time - sp.time) * M.fps));
  }
  return {sp, std::max(0, offset)};
}

struct StateErr {
  double heat = 0, soot = 0, vel = 0, heat_rms = 0, soot_rms = 0, vel_rms = 0, fine_heat = NAN, fine_soot = NAN;
};
StateErr state_error(const rollout::Model& M, const std::vector<float>& est, const std::vector<float>& truth, const footage::Fields* fine,
                     const footage::Fields* fine_truth) {
  StateErr e;
  const double n = kRes * kRes;
  for (int j = 0; j < kRes * kRes; ++j) {
    const float* a = est.data() + sz(j) * 4;
    const float* b = truth.data() + sz(j) * 4;
    e.vel += ((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])) / (2 * n);
    e.heat += (a[2] - b[2]) * (a[2] - b[2]) / n;
    e.soot += (a[3] - b[3]) * (a[3] - b[3]) / n;
    e.vel_rms += (b[0] * b[0] + b[1] * b[1]) / (2 * n);
    e.heat_rms += b[2] * b[2] / n;
    e.soot_rms += b[3] * b[3] / n;
  }
  for (double* v : {&e.heat, &e.soot, &e.vel, &e.heat_rms, &e.soot_rms, &e.vel_rms}) *v = std::sqrt(*v);
  if (fine && fine_truth) {
    e.fine_heat = footage::field_psnr(fine->heat, fine_truth->heat, M.render_scale[0]);
    e.fine_soot = footage::field_psnr(fine->soot, fine_truth->soot, M.render_scale[1]);
  }
  return e;
}

std::string track_cells(const Clip& clip, const Truth& t, int offset) {
  std::string s;
  for (const int h : kHorizons) {
    const auto ref = h == 0 ? std::span<const std::uint8_t>(t.ctx.back()) : t.ref.frame(h - 1);
    s += std::format(",{:.3f}", apsnr(ref, clip.frame(h + offset)));
  }
  return s;
}

std::string err_cells(const StateErr& e) {
  const auto f = [](double v) { return std::isfinite(v) ? std::format("{:.3f}", v) : std::string(); };
  return std::format(",{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{},{}", e.heat, e.soot, e.vel, e.heat_rms, e.soot_rms, e.vel_rms, f(e.fine_heat), f(e.fine_soot));
}

const std::string kTrackHeader =
    "split,effect,run,condition,method,h0,h1,h8,h30,h60,heat_rmse,soot_rmse,vel_rmse,heat_rms,soot_rms,vel_rms,fine_heat_psnr,fine_soot_psnr";

// The validation grid: oracles (true fields or velocity) and the estimator's choices. The test runs `true`,
// `stored_nearest` and the chosen configuration (named `estimated`).
std::vector<Config> val_configs() {
  std::vector<Config> v;
  const auto add = [&](std::string name, int K, bool refine, Vel vel, float beta = 0.5f, int pairs = 1) {
    Config c;
    c.name = std::move(name);
    c.K = K;
    c.refine = refine;
    c.vel = vel;
    c.beta = beta;
    c.pairs = pairs;
    v.push_back(c);
    return &v.back();
  };
  add("k1_zero", 1, true, Vel::zero);
  add("k1_zero_noref", 1, false, Vel::zero);
  add("k2_flow", 2, true, Vel::flow);
  add("k2_flow_noref", 2, false, Vel::flow);
  add("k4_flowmean", 4, true, Vel::flow, 0.f, 3);
  add("k8_flowmean", 8, true, Vel::flow, 0.f, 7);
  for (const int K : {4, 8}) {
    for (const float b : {0.f, 0.5f, 1.f}) add(std::format("k{}_assim_b{}", K, b == 0.5f ? std::string("05") : std::format("{}", static_cast<int>(b))), K, true, Vel::assim, b);
  }
  add("k3_motion", 3, true, Vel::flow)->motion = true;
  for (const float b : {0.5f, 1.f}) add(std::format("k8_assim_motion_b{}", b == 0.5f ? std::string("05") : std::string("1")), 8, true, Vel::assim, b)->motion = true;
  add("k4_assim_motion_b05", 4, true, Vel::assim, 0.5f)->motion = true;
  add("k8_assim_b05_noref", 8, false, Vel::assim, 0.5f);
  add("k8_assim_b05_p2", 8, true, Vel::assim, 0.5f, 2);
  add("k8_assim_b05_ctl05", 8, true, Vel::assim, 0.5f)->controls_given = false;
  add("k8_assim_b05_runseed", 8, true, Vel::assim, 0.5f)->run_seed = true;
  add("oracle_vel", 1, true, Vel::truth);
  add("oracle_fields_zero", 1, false, Vel::zero)->fields_truth = true;
  add("oracle_fields_flow", 2, false, Vel::flow)->fields_truth = true;
  add("oracle_fields_assim", 8, false, Vel::assim, 0.5f)->fields_truth = true;
  add("true", 1, false, Vel::truth)->fields_truth = true;
  return v;
}

// --- inverse ------------------------------------------------------------------------------------------------------

void step_inverse(const Ctx& c) {
  std::vector<std::string> rows;
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e)) continue;
    for (const bool alpha : {true, false}) {
      const auto t0 = std::chrono::steady_clock::now();
      footage::SampleOptions so;
      so.runs = c.quick ? 6 : 64;
      so.per_run = 6;
      so.salt = 1;
      so.drop_alpha = !alpha;
      so.threads = c.threads;
      const auto samples = footage::simulate_samples(e, so);
      const double sim_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      footage::InverseSpec spec;
      spec.alpha = alpha;
      footage::Inverse inv = footage::init_inverse(spec, 11);
      inv.effect = ename(e);
      footage::InverseTrainOptions to;
      to.iterations = c.quick ? 300 : 12000;
      to.threads = c.threads;
      to.progress = [&](int it, double loss) { std::println("  {} {} inverse {:5d} loss {:.5f}", ename(e), alpha ? "rgba" : "rgb", it, loss); };
      const double loss = footage::train_inverse(inv, samples, to);
      const double train_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() - sim_s;
      fs::create_directories(c.data / "inverse");
      if (auto w = footage::save_inverse(c.data / "inverse" / std::format("{}_{}.nvfxinv", ename(e), alpha ? "rgba" : "rgb"), inv); !w) throw std::runtime_error(w.error());
      // held out: validation runs (salt 3), clean frames
      footage::SampleOptions ho = so;
      ho.runs = c.quick ? 2 : 8;
      ho.per_run = 4;
      ho.salt = 3;
      ho.degrade_share = 0.f;
      const auto held = footage::simulate_samples(e, ho);
      const rollout::Model M = load_effect(c, e);
      double ph = 0, ps = 0;
      for (const auto& s : held) {
        const footage::Fields f = footage::apply_inverse(inv, s.frame, s.size);
        ph += footage::field_psnr(f.heat, s.truth.heat, M.render_scale[0]) / static_cast<double>(held.size());
        ps += footage::field_psnr(f.soot, s.truth.soot, M.render_scale[1]) / static_cast<double>(held.size());
      }
      std::println("inverse {} {}: {} samples, loss {:.5f}, held-out fine PSNR heat {:.2f} soot {:.2f} dB, {:.0f} s simulate, {:.0f} s train", ename(e),
                   alpha ? "rgba" : "rgb", samples.size(), loss, ph, ps, sim_s, train_s);
      std::fflush(stdout);
      rows.push_back(std::format("{},{},{},{},{:.6f},{:.3f},{:.3f},{:.4f},{:.4f},{},{:.0f},{:.0f}", ename(e), alpha ? "rgba" : "rgb", samples.size(), to.iterations, loss, ph, ps,
                                 inv.scale[0], inv.scale[1], inv.weights(), sim_s, train_s));
    }
  }
  write_csv(c.results / "j_inverse.csv", "effect,mode,samples,iterations,loss,heldout_heat_psnr,heldout_soot_psnr,scale_heat,scale_soot,weights,simulate_s,train_s", rows);
}

// The motion network of each inverse file: trained on the training runs (salt 1) through that inverse network, checked
// on validation runs (salt 3) against block matching.
void step_motion(const Ctx& c) {
  std::vector<std::string> rows;
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e)) continue;
    for (const bool alpha : {true, false}) {
      const auto t0 = std::chrono::steady_clock::now();
      footage::Inverse inv = load_inv(c, e, alpha);
      inv.motion = footage::Motion{};
      inv.motion.in_scale = inv.scale;
      footage::SampleOptions so;
      so.runs = c.quick ? 6 : 64;
      so.per_run = 6;
      so.salt = 1;
      so.threads = c.threads;
      const auto samples = footage::simulate_motion_samples(e, inv, so);
      const double sim_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      footage::InverseTrainOptions to;
      to.iterations = c.quick ? 300 : 12000;
      to.threads = c.threads;
      to.progress = [&](int it, double loss) { std::println("  {} {} motion {:5d} loss {:.5f}", ename(e), alpha ? "rgba" : "rgb", it, loss); };
      const double loss = footage::train_motion(inv.motion, samples, to);
      const double train_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() - sim_s;
      if (auto w = footage::save_inverse(c.data / "inverse" / std::format("{}_{}.nvfxinv", ename(e), alpha ? "rgba" : "rgb"), inv); !w) throw std::runtime_error(w.error());
      // held out (salt 3, clean frames): velocity RMSE of the motion network and of block matching, all cells
      footage::SampleOptions ho = so;
      ho.runs = c.quick ? 2 : 8;
      ho.per_run = 4;
      ho.salt = 3;
      ho.degrade_share = 0.f;
      const auto held = footage::simulate_motion_samples(e, inv, ho, kRes * kRes);
      double em = 0, ef = 0, rms = 0, n = 0;
      for (const auto& s : held) {
        const footage::Flow mv = footage::apply_motion(inv.motion, s.inputs, kRes);
        footage::Flow bf;  // block matching into the last frame, as the inputs carry it (3 x 3 centre tap), filled
        bf.res = kRes;
        bf.uv.assign(sz(kRes) * kRes * 2, 0.f);
        bf.known.assign(sz(kRes) * kRes, 0);
        for (int q = 0; q < kRes * kRes; ++q) {
          const float* in = s.inputs.data() + sz(q) * footage::Motion::kInputs + 150 + 4 * 3;
          if (in[2] > 0.5f) {
            bf.uv[sz(q) * 2] = in[0] / 4.f;
            bf.uv[sz(q) * 2 + 1] = in[1] / 4.f;
            bf.known[sz(q)] = 1;
          }
        }
        footage::finish_flow(bf, flow_options());
        for (int q = 0; q < kRes * kRes; ++q) {
          for (int k = 0; k < 2; ++k) {
            const double t = s.target[sz(q) * 2 + sz(k)];
            em += (mv.uv[sz(q) * 2 + sz(k)] - t) * (mv.uv[sz(q) * 2 + sz(k)] - t);
            ef += (bf.uv[sz(q) * 2 + sz(k)] - t) * (bf.uv[sz(q) * 2 + sz(k)] - t);
            rms += t * t;
            n += 1;
          }
        }
      }
      em = std::sqrt(em / n);
      ef = std::sqrt(ef / n);
      rms = std::sqrt(rms / n);
      std::println("motion {} {}: {} samples, loss {:.4f}, held-out velocity RMSE {:.4f} (block matching {:.4f}; true RMS {:.4f}), {:.0f} s simulate, {:.0f} s train", ename(e),
                   alpha ? "rgba" : "rgb", samples.size(), loss, em, ef, rms, sim_s, train_s);
      std::fflush(stdout);
      rows.push_back(std::format("{},{},{},{},{:.5f},{:.5f},{:.5f},{:.5f},{:.0f},{:.0f}", ename(e), alpha ? "rgba" : "rgb", samples.size(), to.iterations, loss, em, ef, rms, sim_s, train_s));
    }
  }
  write_csv(c.results / std::format("j_motion{}.csv", c.effects.empty() ? "" : "_" + c.effects),
            "effect,mode,samples,iterations,loss,heldout_motion_rmse,heldout_blockmatch_rmse,heldout_true_rms,simulate_s,train_s", rows);
}

// --- tracking: validation grid and the test -------------------------------------------------------------------

struct Job {
  sim::Effect e;
  int run;
};

void run_tracking(const Ctx& c, const std::string& split) {
  const bool test = split == "test";
  const std::uint64_t salt = test ? 2 : 3;
  const int runs = c.runs > 0 ? c.runs : (c.quick ? 2 : (test ? 16 : 12));
  std::vector<Config> configs;
  if (test) {  // the choice made on validation (docs/FOOTAGE.md §3)
    Config est;
    est.name = "estimated";
    est.K = 8;
    est.refine = true;
    est.vel = Vel::assim;
    est.beta = 0.5f;
    configs.push_back(est);
    Config tr;
    tr.name = "true";
    tr.K = 1;
    tr.refine = false;
    tr.vel = Vel::truth;
    tr.fields_truth = true;
    configs.push_back(tr);
  } else {
    configs = val_configs();
  }
  std::map<sim::Effect, rollout::Model> models;
  std::map<std::pair<sim::Effect, bool>, footage::Inverse> invs;
  std::vector<Job> jobs;
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e)) continue;
    models[e] = load_effect(c, e);
    invs[{e, true}] = load_inv(c, e, true);
    invs[{e, false}] = load_inv(c, e, false);
    for (int r = 0; r < runs; ++r) jobs.push_back({e, r});
  }
  std::vector<std::vector<std::string>> rows(jobs.size());
  std::atomic<int> next{0};
  std::mutex log;
  const auto work = [&] {
    for (int j; (j = next++) < static_cast<int>(jobs.size());) {
      const auto t0 = std::chrono::steady_clock::now();
      const Job job = jobs[sz(j)];
      const rollout::Model& M = models.at(job.e);
      rollout::SimRecipe rc = rollout::recipe_for(job.e);
      rc.salt = salt;
      sim::Params p = rollout::recipe_run(rc, static_cast<std::uint64_t>(job.run));
      p.size = kSize;
      const Truth t = simulate_run(p, start_frame(job.e));
      const std::vector<float> controls{p.intensity, p.wind, p.turbulence};
      const std::string en = ename(job.e);
      auto& out = rows[sz(j)];
      {  // today: the nearest stored start point
        const auto [sp, offset] = stored_start(M, t);
        const Clip clip = track(M, sp, controls, kLast + 1 + offset);
        out.push_back(std::format("{},{},{},any,stored_nearest{}{}", split, en, job.run, track_cells(clip, t, offset),
                                  err_cells(state_error(M, sp.coarse, t.coarse.back(), nullptr, nullptr))));
      }
      bool oracles_done = false;
      const std::vector<footage::Flow> raw_true = true_flows(M, t);
      for (const Cond cond : kConds) {
        const footage::Inverse& inv = invs.at({job.e, has_alpha(cond)});
        const auto frames = footage_of(t, cond, M, c.data / "scratch", (salt * 1000 + static_cast<std::uint64_t>(job.run)) * 16 + static_cast<std::uint64_t>(cond));
        const Shared s = prepare(M, inv, frames, &raw_true, true);
        for (const Config& cf : configs) {
          const bool oracle_fields = cf.fields_truth;
          if (oracle_fields && oracles_done) continue;  // true fields do not depend on the footage
          auto [sp, fine] = config_start(M, cf, t, s);
          const Clip clip = track(M, sp, controls, kLast + 1);
          out.push_back(std::format("{},{},{},{},{}{}{}", split, en, job.run, oracle_fields ? "any" : cname(cond), cf.name, track_cells(clip, t, 0),
                                    err_cells(state_error(M, sp.coarse, t.coarse.back(), &fine, &t.fields.back()))));
        }
        oracles_done = true;
        if (test && cond == Cond::clean_rgba) {  // the tool's own entry point gives the same start point
          footage::EstimateOptions eo;
          eo.context = 8;
          eo.assim.controls = controls;
          eo.assim.time = t.time;
          const footage::Estimate est = footage::estimate_start(M, inv, frames, kSize, eo);
          auto [sp, fine] = config_start(M, configs[0], t, s);
          double d = 0;
          for (std::size_t q = 0; q < sp.coarse.size(); ++q) d = std::max(d, static_cast<double>(std::abs(sp.coarse[q] - est.start.coarse[q])));
          if (d > 1e-5) std::println(stderr, "warning: the tool's estimate differs from the study's by {}", d);
        }
      }
      std::lock_guard lk(log);
      std::println("{} {} run {:2d}: {:.0f} s", split, en, job.run, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      std::fflush(stdout);
    }
  };
  {
    std::vector<std::jthread> pool;
    for (int t = 0; t < std::max(1, c.threads); ++t) pool.emplace_back(work);
  }
  std::vector<std::string> all;
  for (auto& r : rows) all.insert(all.end(), r.begin(), r.end());
  write_csv(c.results / std::format("j_{}{}.csv", split, c.effects.empty() ? "" : "_" + c.effects), kTrackHeader, all);
}

// --- endless statistics -----------------------------------------------------------------------------------------

std::vector<std::array<float, 3>> b_test_settings() {  // study B's held-out settings (as study D)
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<std::array<float, 3>> v;
  while (v.size() < 10) {
    const std::array<float, 3> s{u(rng), u(rng), u(rng)};
    const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
    if (off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f})) v.push_back(s);
  }
  return v;
}

void run_endless(const Ctx& c, const std::string& split) {
  const bool test = split == "test";
  std::vector<std::array<float, 3>> settings;
  if (test) {
    settings = b_test_settings();
  } else {
    for (int i = 0; i < 6; ++i) {
      rollout::SimRecipe rc;
      rc.salt = 3;
      const sim::Params p = rollout::recipe_run(rc, 100 + static_cast<std::uint64_t>(i));
      settings.push_back({p.intensity, p.wind, p.turbulence});
    }
  }
  if (c.quick) settings.resize(2);
  // seeds: real reference, real other seed (the floor), footage, play
  const std::uint64_t s_ref = test ? 900000 : 978000, s_other = test ? 910000 : 979000, s_foot = test ? 960000 : 970000, s_play = test ? 920000 : 975000;
  struct Job2 {
    sim::Effect e;
    int si;
  };
  std::vector<Job2> jobs;
  std::map<sim::Effect, rollout::Model> models;
  std::map<std::pair<sim::Effect, bool>, footage::Inverse> invs;
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e)) continue;
    models[e] = load_effect(c, e);
    invs[{e, true}] = load_inv(c, e, true);
    invs[{e, false}] = load_inv(c, e, false);
    for (int si = 0; si < static_cast<int>(settings.size()); ++si) jobs.push_back({e, si});
  }
  std::vector<std::vector<std::string>> rows(jobs.size());
  std::atomic<int> next{0};
  std::mutex log;
  const auto work = [&] {
    for (int j; (j = next++) < static_cast<int>(jobs.size());) {
      const auto t0 = std::chrono::steady_clock::now();
      const auto [e, si] = jobs[sz(j)];
      const bool ex = e == sim::Effect::explosion;
      const rollout::Model& M = models.at(e);
      const auto& s = settings[sz(si)];
      const std::vector<float> ctl{s[0], s[1], s[2]};
      const int F = ex ? 89 : (c.quick ? 90 : 300), warm = ex ? 1 : 150;
      const auto params = [&](std::uint64_t seed) {
        sim::Params p;
        p.effect = e;
        p.intensity = s[0];
        p.wind = s[1];
        p.turbulence = s[2];
        p.seed = seed;
        p.size = kSize;
        return p;
      };
      const auto real_run = [&](std::uint64_t seed) {
        sim::Fluid f(params(seed));
        for (int i = 0; i < warm; ++i) f.step_frame();
        Clip cl;
        cl.allocate(kSize, F);
        cl.fps = 30.f;
        for (int i = 0; i < F; ++i) {
          f.step_frame();
          f.render(cl.frame(i));
        }
        return cl;
      };
      // ages (frames since the effect began) of each clip's frame 0: the window compared is the ages all clips cover
      const Clip ref = real_run(s_ref + static_cast<std::uint64_t>(si)), other = real_run(s_other + static_cast<std::uint64_t>(si));
      const Clip stored = runtime_clip(Effect(M).e, ctl, -1, s_play + static_cast<std::uint64_t>(si), F, true);
      const std::vector<int> at = ex ? std::vector<int>{start_frame(e)} : std::vector<int>{150, 180, 210};
      std::map<std::string, Clip> est_clips;
      std::vector<Truth> truths;
      for (const int f0 : at) truths.push_back(simulate_run(params(s_foot + static_cast<std::uint64_t>(si)), f0));
      for (const Cond cond : {Cond::clean_rgba, Cond::h264_rgb}) {
        std::vector<rollout::StartPoint> starts;
        for (std::size_t q = 0; q < at.size(); ++q) {
          const int f0 = at[q];
          const Truth& t = truths[q];
          const auto frames = footage_of(t, cond, M, c.data / "scratch", (s_foot + static_cast<std::uint64_t>(si)) * 64 + static_cast<std::uint64_t>(f0) * 4 + static_cast<std::uint64_t>(cond));
          footage::EstimateOptions eo;
          eo.context = 8;
          eo.assim.controls = ctl;
          eo.assim.time = t.time;
          footage::Estimate est = footage::estimate_start(M, invs.at({e, has_alpha(cond)}), frames, kSize, eo);
          est.start.seed = 4242 + static_cast<std::uint64_t>(starts.size());
          starts.push_back(std::move(est.start));
        }
        const rollout::Model me = footage::with_starts(M, starts, kFine, false);
        est_clips[std::format("estimated_{}", cname(cond))] = runtime_clip(Effect(me).e, ctl, -1, s_play + static_cast<std::uint64_t>(si), F, true);
      }
      // explosions: real frame i is age warm + 1 + i; the stored start's frame f is its age + f; the estimate's t0 + f
      const auto window = [&](const Clip& cl, int age0) {
        if (!ex) return cl;
        const int lo = start_frame(e) + 1, hi = 89;
        return slice_clip(cl, lo - age0, hi - lo + 1);
      };
      int stored_age = 1;
      {
        std::size_t best = 0;
        float bd = 1e30f;
        for (std::size_t i = 0; i < M.starts.size(); ++i) {
          float d = 0;
          for (std::size_t k = 0; k < 3; ++k) d += (M.starts[i].controls[k] - ctl[k]) * (M.starts[i].controls[k] - ctl[k]);
          if (d < bd) {
            bd = d;
            best = i;
          }
        }
        stored_age = static_cast<int>(std::lround(M.starts[best].time * M.fps));
      }
      const metrics::ClipStats rs = metrics::stats(window(ref, warm + 1));
      std::map<std::string, metrics::StatDistance> d;
      d["real_other_seed"] = metrics::distance(rs, metrics::stats(window(other, warm + 1)));
      d["stored"] = metrics::distance(rs, metrics::stats(window(stored, stored_age)));
      for (const auto& [name, cl] : est_clips) d[name] = metrics::distance(rs, metrics::stats(window(cl, start_frame(e))));
      for (const auto& [name, x] : d) {
        rows[sz(j)].push_back(std::format("{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.3f}", split, ename(e), si, name, x.spectrum_l1, x.motion_ratio, x.coverage_l1,
                                          x.emission_l1, x.mean_frame_psnr));
      }
      std::lock_guard lk(log);
      std::println("{} endless {} setting {}: {:.0f} s", split, ename(e), si, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      std::fflush(stdout);
    }
  };
  {
    std::vector<std::jthread> pool;
    for (int t = 0; t < std::max(1, c.threads); ++t) pool.emplace_back(work);
  }
  std::vector<std::string> all;
  for (auto& r : rows) all.insert(all.end(), r.begin(), r.end());
  write_csv(c.results / std::format("j_{}_endless{}.csv", split, c.effects.empty() ? "" : "_" + c.effects),
            "split,effect,setting,method,spectrum_l1,motion_ratio,coverage_l1,emission_l1,mean_frame_psnr", all);
}

// --- report -------------------------------------------------------------------------------------------------------

std::vector<std::vector<std::string>> read_csv(const fs::path& p) {
  std::vector<std::vector<std::string>> rows;
  std::ifstream in(p);
  std::string line;
  bool header = true;
  while (std::getline(in, line)) {
    if (header) {
      header = false;
      continue;
    }
    if (line.empty()) continue;
    std::vector<std::string> cells;
    std::size_t a = 0;
    while (true) {
      const std::size_t b = line.find(',', a);
      cells.push_back(line.substr(a, b == std::string::npos ? std::string::npos : b - a));
      if (b == std::string::npos) break;
      a = b + 1;
    }
    rows.push_back(std::move(cells));
  }
  return rows;
}

double num(const std::string& s) { return s.empty() ? NAN : std::stod(s); }

std::string iv(const metrics::Interval& v, int prec = 2) {
  return std::format("{:+.{}f} [{:+.{}f}, {:+.{}f}]{}", v.mean, prec, v.lo, prec, v.hi, prec, v.covers_zero() ? " (tie)" : "");
}

// Tracking tables of one split: mean per method and condition, and paired differences.
void report_tracking(const fs::path& csv, std::ostream& md, const std::string& title, const std::vector<std::string>& order, bool full) {
  if (!fs::exists(csv)) return;
  // effect -> (condition, method) -> run -> values (h0..h60, then state columns)
  std::map<std::string, std::map<std::pair<std::string, std::string>, std::map<int, std::vector<double>>>> data;
  for (const auto& r : read_csv(csv)) {
    std::vector<double> v;
    for (std::size_t k = 5; k < r.size(); ++k) v.push_back(num(r[k]));
    data[r[1]][{r[3], r[4]}][std::stoi(r[2])] = v;
  }
  md << "\n### " << title << "\n\n";
  const std::vector<std::string> conds = {"any", "clean_rgba", "clean_rgb", "blur_noise_rgb", "h264_rgb", "learned_rgba"};
  for (const auto& [effect, methods] : data) {
    md << "**" << effect << "** (active PSNR in dB by frames after the start; state errors at the start: coarse RMSE, fine PSNR in dB)\n\n"
       << "| condition | method | runs | 0 | 1 | 8 | 30 | 60 | heat RMSE | soot RMSE | velocity RMSE | fine heat | fine soot |\n"
       << "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    std::vector<std::pair<std::string, std::string>> keys;
    for (const auto& cnd : conds) {
      for (const auto& m : order) {
        if (methods.contains({cnd, m})) keys.push_back({cnd, m});
      }
      if (full) {
        for (const auto& [k, v] : methods) {
          if (k.first == cnd && std::ranges::find(order, k.second) == order.end()) keys.push_back(k);
        }
      }
    }
    for (const auto& k : keys) {
      const auto& runs = methods.at(k);
      std::vector<double> mean(13, 0.0);
      std::vector<int> n(13, 0);
      for (const auto& [run, v] : runs) {
        for (std::size_t q = 0; q < std::min<std::size_t>(13, v.size()); ++q) {
          if (q >= 5 && q <= 10 && (q == 8 || q == 9 || q == 10)) continue;  // the truth's RMS columns
          if (std::isfinite(v[q])) {
            mean[q] += v[q];
            ++n[q];
          }
        }
      }
      const auto cell = [&](std::size_t q, int prec) { return n[q] ? std::format("{:.{}f}", mean[q] / n[q], prec) : std::string("-"); };
      md << std::format("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n", k.first, k.second, runs.size(), cell(0, 2), cell(1, 2), cell(2, 2),
                        cell(3, 2), cell(4, 2), cell(5, 4), cell(6, 4), cell(7, 4), cell(11, 2), cell(12, 2));
    }
    // paired differences against the nearest stored start and the true start, at every horizon
    md << "\nPaired over runs, active PSNR difference in dB (95% bootstrap, 10,000 resamples):\n\n| condition | comparison | 1 | 8 | 30 | 60 |\n|---|---|---|---|---|---|\n";
    const auto pair_row = [&](const std::string& cnd, const std::string& a, const std::string& ca, const std::string& b, const std::string& cb) {
      if (!methods.contains({ca, a}) || !methods.contains({cb, b})) return;
      md << "| " << cnd << " | " << a << " - " << b << " |";
      for (const std::size_t q : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4}}) {
        std::vector<double> x, y;
        for (const auto& [run, v] : methods.at({ca, a})) {
          const auto it = methods.at({cb, b}).find(run);
          if (it == methods.at({cb, b}).end() || q >= v.size() || q >= it->second.size()) continue;
          x.push_back(v[q]);
          y.push_back(it->second[q]);
        }
        md << " " << (x.size() >= 2 ? iv(metrics::paired_bootstrap(x, y)) : std::string("-")) << " |";
      }
      md << "\n";
    };
    for (const auto& cnd : conds) {
      if (cnd == "any") continue;
      for (const auto& m : order) {
        if (m == "stored_nearest" || m == "true" || m.starts_with("oracle")) continue;
        if (!methods.contains({cnd, m})) continue;
        pair_row(cnd, m, cnd, "stored_nearest", "any");
        pair_row(cnd, m, cnd, "true", "any");
      }
    }
    md << "\n";
  }
}

void report_endless(const fs::path& csv, std::ostream& md, const std::string& title) {
  if (!fs::exists(csv)) return;
  std::map<std::string, std::map<std::string, std::map<int, std::array<double, 5>>>> data;
  for (const auto& r : read_csv(csv)) data[r[1]][r[3]][std::stoi(r[2])] = {num(r[4]), num(r[5]), num(r[6]), num(r[7]), num(r[8])};
  md << "\n### " << title << "\n\n";
  const std::vector<std::string> stat_names = {"spectrum L1", "motion ratio", "coverage L1", "emission L1", "mean-frame PSNR"};
  for (const auto& [effect, methods] : data) {
    md << "**" << effect << "**\n\n| method | settings | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |\n|---|---:|---:|---:|---:|---:|---:|\n";
    for (const std::string m : {"real_other_seed", "stored", "estimated_clean_rgba", "estimated_h264_rgb"}) {
      if (!methods.contains(m)) continue;
      std::array<double, 5> s{};
      for (const auto& [si, v] : methods.at(m)) {
        for (std::size_t k = 0; k < 5; ++k) s[k] += v[k] / static_cast<double>(methods.at(m).size());
      }
      md << std::format("| {} | {} | {:.3f} | {:.2f} | {:.4f} | {:.4f} | {:.2f} |\n", m, methods.at(m).size(), s[0], s[1], s[2], s[3], s[4]);
    }
    md << "\nPaired over settings, estimated minus stored (motion: |log ratio| difference, lower is better for every statistic except mean-frame PSNR):\n\n"
       << "| estimated from | spectrum L1 | motion |log ratio| | coverage L1 | emission L1 | mean-frame PSNR |\n|---|---|---|---|---|---|\n";
    for (const std::string m : {"estimated_clean_rgba", "estimated_h264_rgb"}) {
      if (!methods.contains(m) || !methods.contains("stored")) continue;
      md << "| " << m.substr(10) << " |";
      for (std::size_t k = 0; k < 5; ++k) {
        std::vector<double> x, y;
        for (const auto& [si, v] : methods.at(m)) {
          const auto it = methods.at("stored").find(si);
          if (it == methods.at("stored").end()) continue;
          x.push_back(k == 1 ? std::abs(std::log(v[k])) : v[k]);
          y.push_back(k == 1 ? std::abs(std::log(it->second[k])) : it->second[k]);
        }
        md << " " << (x.size() >= 2 ? iv(metrics::paired_bootstrap(x, y), k == 4 ? 2 : 3) : std::string("-")) << " |";
      }
      md << "\n";
    }
    md << "\n";
  }
}

void step_report(const Ctx& c) {
  std::ostringstream md;
  md << "# Study J: start points estimated from frames (generated by `nvfx_study_j report`)\n\nSee docs/FOOTAGE.md for the protocol. Active PSNR "
        "against the real run's clean frames; `0` is the start frame itself redrawn by the effect.\n";
  report_tracking(c.results / "j_test.csv", md, "Test: tracking from the start frame (salt-2 runs)", {"true", "estimated", "stored_nearest"}, false);
  report_endless(c.results / "j_test_endless.csv", md, "Test: endless statistics at study B's held-out settings");
  std::vector<std::string> order = {"true", "stored_nearest"};
  for (const auto& cf : val_configs()) {
    if (cf.name != "true") order.push_back(cf.name);
  }
  report_tracking(c.results / "j_val.csv", md, "Validation: tracking (salt-3 runs), every configuration", order, true);
  report_endless(c.results / "j_val_endless.csv", md, "Validation: endless statistics (6 settings of salt-3 runs)");
  std::ofstream(c.results / "j_summary.md") << md.str();
  std::print("{}", md.str());
}

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"quick", "help"});
  if (a.flag("help") || a.positional().empty()) {
    std::println("nvfx_study_j inverse|motion|val|val-endless|test|test-endless|report [--models DIR] [--data DIR] [--results DIR] [--effects fire,smoke] [--threads 1] [--runs N] [--quick]");
    return 0;
  }
  Ctx c;
  c.data = a.has("data") ? fs::path(a.str("data")) : data_root() / "j";
  c.models = a.has("models") ? fs::path(a.str("models")) : c.data / "models";
  c.results = a.str("results", "results/experiments");
  c.effects = a.str("effects");
  c.threads = a.i("threads", 1);
  c.runs = a.i("runs", 0);
  c.quick = a.flag("quick");
  if (c.quick) c.results /= "quick";
  const std::string step = a.positional()[0];
  const auto t0 = std::chrono::steady_clock::now();
  if (step == "inverse") step_inverse(c);
  else if (step == "motion") step_motion(c);
  else if (step == "val") run_tracking(c, "val");
  else if (step == "val-endless") run_endless(c, "val");
  else if (step == "test") run_tracking(c, "test");
  else if (step == "test-endless") run_endless(c, "test");
  else if (step == "report") step_report(c);
  else throw std::invalid_argument("unknown step " + step);
  std::println("{} finished in {:.1f} min", step, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 60.0);
  a.warn_unused();
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_study_j: {}", e.what());
  return 2;
}
