// Study D (docs/REPORT.md §8): start points and learned dynamics. Part of nvfx_experiment.
//
//   d-chaos   how fast runs from the same start point drift apart, and how much the noise seed decides
//   d-train   per effect: record runs, train the stepper, the renderer, calibrate the detail layer, keep start points
//   d-eval    held-out runs and settings: tracking from true start points, statistics of endless runs through the
//             runtime against real runs, flipbook libraries, the B control models and a coarse simulation
//   d-timing  milliseconds per frame through the runtime (quiet machine)
#include "experiment_d.hpp"

#include <neuralfx/flipbook.hpp>
#include <neuralfx/image_io.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/model.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <map>
#include <print>
#include <random>
#include <sched.h>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;
using namespace nfx;

namespace nfx::study_d {

namespace {

constexpr int kSize = 128, kRes = 32;

std::string ename(sim::Effect e) { return std::string(sim::effect_name(e)); }

// The library recipe per effect (rollout::recipe_for), scaled down for --quick, with the experiment's thread count.
std::vector<rollout::SimRecipe> recipes(const Ctx& c) {
  std::vector<rollout::SimRecipe> v;
  for (const auto e : sim::kEffects) {
    rollout::SimRecipe r = rollout::recipe_for(e);
    r.threads = c.threads;
    r.stepper.iterations = c.iters(2500);
    r.stepper.finetune = c.iters(1500);
    r.stepper.threads = c.threads;
    r.renderer.iterations = c.iters(2500);
    r.renderer.threads = c.threads;
    if (c.quick) {
      r.runs = std::max(8, r.runs / 10);
      r.frames = std::min(r.frames, 80);
      r.start_frame = std::min(r.start_frame, 60);
      r.stepper.burn_max = 8;
      r.render_runs = 8;
    }
    v.push_back(r);
  }
  return v;
}

// Training runs use salt 1; held-out runs salt 2 (other seeds and settings).
constexpr std::uint64_t kTestSalt = 2;
sim::Params run_params(sim::Effect e, std::uint64_t index, std::uint64_t salt) {
  rollout::SimRecipe r;
  r.effect = e;
  r.salt = salt;
  return rollout::recipe_run(r, index);
}

double apsnr(std::span<const std::uint8_t> r, std::span<const std::uint8_t> t) {
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

}  // namespace

// --- d-chaos ----------------------------------------------------------------------------------------------------------

void step_chaos(const Ctx& c) {
  const std::vector<int> report = {1, 2, 4, 8, 15, 30, 60, 90, 120, 180, 240};
  std::vector<std::string> rows;
  for (const auto e : sim::kEffects) {
    const bool ex = e == sim::Effect::explosion;
    const int N = ex ? 90 : 240, warm = ex ? 1 : 100, seeds = c.quick ? 1 : 4;
    // curves: perturbed start (1e-3, 1e-1 of the velocity RMS) with the same seed; same start with another seed;
    // another start with the same seed
    const std::vector<std::string> names = {"start+1e-3_same_seed", "start+1e-1_same_seed", "same_start_other_seed", "other_start_same_seed"};
    std::vector<std::vector<double>> curves(names.size(), std::vector<double>(static_cast<std::size_t>(N), 0.0));
    for (int k = 0; k < seeds; ++k) {
      sim::Params p = run_params(e, static_cast<std::uint64_t>(k), 77);
      sim::Fluid a(p);
      for (int i = 0; i < warm; ++i) a.step_frame();
      const sim::State s0 = a.state();
      std::vector<std::vector<std::uint8_t>> ref(static_cast<std::size_t>(N), std::vector<std::uint8_t>(kSize * kSize * 4));
      {
        sim::Fluid b(p);
        b.set_state(s0);
        for (int i = 0; i < N; ++i) {
          b.step_frame();
          b.render(ref[static_cast<std::size_t>(i)]);
        }
      }
      std::vector<std::uint8_t> fb(kSize * kSize * 4);
      const auto run_from = [&](const sim::Params& q, const sim::State& s, std::size_t curve) {
        sim::Fluid b(q);
        b.set_state(s);
        for (int i = 0; i < N; ++i) {
          b.step_frame();
          b.render(fb);
          curves[curve][static_cast<std::size_t>(i)] += apsnr(ref[static_cast<std::size_t>(i)], fb) / seeds;
        }
      };
      for (std::size_t r = 0; r < 2; ++r) {
        sim::State s = s0;
        double rms = 0;
        for (const float x : s.u) rms += static_cast<double>(x) * x;
        for (const float x : s.v) rms += static_cast<double>(x) * x;
        rms = std::sqrt(rms / (2.0 * static_cast<double>(s.u.size())));
        std::mt19937 rng(static_cast<unsigned>(7 + k));
        std::normal_distribution<float> nd(0.f, 1.f);
        const float amp = static_cast<float>((r == 0 ? 1e-3 : 1e-1) * rms);
        for (float& x : s.u) x += amp * nd(rng);
        for (float& x : s.v) x += amp * nd(rng);
        run_from(p, s, r);
      }
      sim::Params q = p;
      q.seed = p.seed + 999;
      run_from(q, s0, 2);
      sim::Fluid w(q);
      for (int i = 0; i < warm; ++i) w.step_frame();
      sim::State s1 = w.state();
      s1.time = s0.time;
      s1.frame = s0.frame;
      run_from(p, s1, 3);
    }
    for (std::size_t r = 0; r < names.size(); ++r) {
      std::string row = std::format("{},{}", ename(e), names[r]);
      for (const int f : report) row += f <= N ? std::format(",{:.2f}", curves[r][static_cast<std::size_t>(f - 1)]) : ",";
      rows.push_back(row);
    }
    std::println("d-chaos: {} done", ename(e));
  }
  std::string header = "effect,case";
  for (const int f : report) header += std::format(",f{}", f);
  write_csv(c.results / "d_chaos.csv", header, rows);
}

// --- d-train ----------------------------------------------------------------------------------------------------------

fs::path model_path(const Ctx& c, sim::Effect e) { return c.data / "models" / "d" / std::format("{}.nvfx", ename(e)); }

void step_train(const Ctx& c) {
  std::vector<std::string> rows;
  for (const rollout::SimRecipe& r : recipes(c)) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<rollout::Run> runs = rollout::record_runs(r);
    const double sim_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    rollout::Model m = rollout::recipe_model(r);
    rollout::StepperOptions so = r.stepper;
    so.progress = [&](int it, int unroll, double loss) { std::println("  {} stepper {:5d} unroll {:2d} loss {:.5f}", m.effect, it, unroll, loss); };
    const rollout::StepperResult sr = rollout::train_stepper(m, runs, so);
    const rollout::FinishResult fin = rollout::finish_model(m, r, runs);
    fs::create_directories(model_path(c, r.effect).parent_path());
    if (auto w = rollout::save_model(model_path(c, r.effect), m); !w) throw std::runtime_error(w.error());
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double footage_min = static_cast<double>(r.runs) * r.frames / 30.0 / 60.0;
    std::println("d-train: {} {} runs x {} frames ({:.1f} min of simulation), stepper loss {:.4f} in {:.0f} s, renderer {:.2f} dB, detail contrast {} swirl {} (score {:.3f}), {} start points, {:.1f} KB, total {:.0f} s",
                 m.effect, r.runs, r.frames, footage_min, sr.final_loss, sr.seconds, fin.render_psnr, m.detail.contrast, m.detail.swirl, fin.detail_score,
                 m.starts.size(), static_cast<double>(m.storage_bytes()) / 1024.0, total);
    std::fflush(stdout);
    rows.push_back(std::format("{},{},{},{:.2f},{:.0f},{:.5f},{:.0f},{:.2f},{},{},{:.3f},{},{},{},{:.0f}", m.effect, r.runs, r.frames, footage_min, sim_seconds,
                               sr.final_loss, sr.seconds, fin.render_psnr, m.detail.contrast, m.detail.swirl, fin.detail_score, m.starts.size(), r.start_fine,
                               m.storage_bytes(), total));
  }
  write_csv(c.results / "d_train.csv",
            "effect,runs,frames,sim_minutes,record_s,stepper_loss,stepper_s,renderer_psnr,contrast,swirl,detail_score,starts,start_fine,stored_bytes,total_s",
            rows);
}

// Redo the cheap stages on trained models (the stepper is kept): renderer, detail layer, start points.
void step_finish(const Ctx& c) {
  std::vector<std::string> rows;
  for (const rollout::SimRecipe& r : recipes(c)) {
    auto loaded = rollout::load_model(model_path(c, r.effect));
    if (!loaded) throw std::runtime_error(loaded.error());
    rollout::Model m = std::move(*loaded);
    m.h.start_fine = r.start_fine;
    const std::vector<rollout::Run> runs = rollout::record_runs(r);
    const rollout::FinishResult fin = rollout::finish_model(m, r, runs);
    if (auto w = rollout::save_model(model_path(c, r.effect), m); !w) throw std::runtime_error(w.error());
    std::println("d-finish: {} renderer {:.2f} dB, detail contrast {} swirl {} (score {:.3f}), {} start points, {:.1f} KB", m.effect, fin.render_psnr,
                 m.detail.contrast, m.detail.swirl, fin.detail_score, m.starts.size(), static_cast<double>(m.storage_bytes()) / 1024.0);
    std::fflush(stdout);
    rows.push_back(std::format("{},{:.2f},{},{},{:.3f},{},{}", m.effect, fin.render_psnr, m.detail.contrast, m.detail.swirl, fin.detail_score, m.starts.size(), m.storage_bytes()));
  }
  write_csv(c.results / "d_finish.csv", "effect,renderer_psnr,contrast,swirl,detail_score,starts,stored_bytes", rows);
}

}  // namespace nfx::study_d

namespace nfx::study_d {

namespace {

// The held-out control settings of study B (the same generator as nvfx_experiment's b_test_settings).
std::vector<std::array<float, 3>> b_test_settings() {
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
std::vector<std::array<float, 3>> b_train_settings() {
  std::vector<std::array<float, 3>> v;
  for (const float i : {0.f, 0.5f, 1.f}) {
    for (const float w : {0.f, 0.25f, 0.5f, 0.75f, 1.f}) {
      for (const float t : {0.f, 0.5f, 1.f}) v.push_back({i, w, t});
    }
  }
  return v;
}

struct Effect {
  nvfx_effect* e = nullptr;
  explicit Effect(const fs::path& p) {
    if (nvfx_effect_load(p.string().c_str(), &e) != NVFX_OK) throw std::runtime_error("cannot load " + p.string());
  }
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

// `frames` frames from the runtime: controls, then a start point (variation >= 0, which keeps its run's seed) or a seed.
Clip runtime_clip(nvfx_effect* e, std::span<const float> controls, int variation, std::uint64_t seed, int frames, int first = 0) {
  nvfx_instance* in = nullptr;
  if (nvfx_instance_create(e, kSize, &in) != NVFX_OK) throw std::runtime_error("instance");
  nvfx_instance_set_controls(in, controls.data(), static_cast<int>(controls.size()));
  nvfx_instance_set_drift(in, 0.f);
  if (variation >= 0) nvfx_instance_set_variation(in, variation);
  else nvfx_instance_set_seed(in, seed);
  Clip c;
  c.allocate(kSize, frames);
  c.fps = 30.f;
  for (int f = 0; f < frames; ++f) nvfx_render(in, (first + f) / 30.0, c.frame(f).data(), kSize * 4);
  nvfx_instance_free(in);
  return c;
}

// The simulation's own renderer on given heat and soot fields at 128 x 128 (what a traditional upres would show).
void render_fields(const sim::Params& p, const std::vector<float>& T, const std::vector<float>& D, std::span<std::uint8_t> out) {
  sim::Params q = p;
  q.sim_res = kSize;
  q.size = kSize;
  sim::Fluid f(q);
  sim::State s;
  s.n = kSize;
  s.u.assign(T.size(), 0.f);
  s.v.assign(T.size(), 0.f);
  s.pressure.assign(T.size(), 0.f);
  s.temp = T;
  s.soot = D;
  f.set_state(s);
  f.render(out);
}

// The traditional cheap alternative: the simulation on a 32-cell grid, with or without the same detail layer
// (fine fields carried by its flow, locked to its coarse fields), drawn by the simulation's own renderer.
struct CoarseSim {
  sim::Params p;
  sim::Fluid fluid;
  rollout::State st;
  const rollout::Model& m;
  CoarseSim(const rollout::Model& model, const sim::Params& params, const sim::State& coarse_start, const std::vector<float>& fine_t,
            const std::vector<float>& fine_d)
      : p(params), fluid([&] {
          sim::Params q = params;
          q.sim_res = kRes;
          return q;
        }()),
        m(model) {
    fluid.set_state(coarse_start);
    st.res = kRes;
    st.size = kSize;
    st.time = coarse_start.time;
    st.since_start = 0.f;
    st.coarse.assign(static_cast<std::size_t>(kRes) * kRes * static_cast<std::size_t>(m.h.channels()), 0.f);
    st.flow.assign(static_cast<std::size_t>(kRes) * kRes * 2, 0.f);
    st.pressure.assign(static_cast<std::size_t>(kRes) * kRes, 0.f);
    st.fine_t = fine_t;
    st.fine_d = fine_d;
  }
  void step(std::span<std::uint8_t> with_detail, std::span<std::uint8_t> plain) {
    fluid.step_frame();
    const sim::State x = fluid.state();
    const int C = m.h.channels();
    for (int i = 0; i < kRes * kRes; ++i) {
      const auto j = static_cast<std::size_t>(i);
      st.flow[j * 2] = x.u[j] / p.fps;
      st.flow[j * 2 + 1] = x.v[j] / p.fps;
      st.coarse[j * static_cast<std::size_t>(C) + 2] = x.temp[j];
      st.coarse[j * static_cast<std::size_t>(C) + 3] = x.soot[j];
    }
    const std::vector<float> controls{p.intensity, p.wind, p.turbulence};
    rollout::detail_step(m, st, p.seed, controls);
    st.time += 1.f / p.fps;
    st.since_start += 1.f / p.fps;
    render_fields(p, st.fine_t, st.fine_d, with_detail);
    fluid.render(plain);
  }
};

sim::State coarse_of(const sim::State& fine) {
  sim::State c;
  c.n = kRes;
  c.frame = fine.frame;
  c.time = fine.time;
  const int k = fine.n / kRes;
  const auto n = static_cast<std::size_t>(kRes) * kRes;
  c.u.assign(n, 0.f);
  c.v.assign(n, 0.f);
  c.temp.assign(n, 0.f);
  c.soot.assign(n, 0.f);
  c.pressure.assign(n, 0.f);
  const float a = 1.f / static_cast<float>(k * k);
  for (int y = 0; y < fine.n; ++y) {
    for (int x = 0; x < fine.n; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * static_cast<std::size_t>(fine.n) + static_cast<std::size_t>(x);
      const std::size_t o = static_cast<std::size_t>(y / k) * kRes + static_cast<std::size_t>(x / k);
      c.u[o] += fine.u[i] * a / static_cast<float>(k);  // coarse cells per second
      c.v[o] += fine.v[i] * a / static_cast<float>(k);
      c.temp[o] += fine.temp[i] * a;
      c.soot[o] += fine.soot[i] * a;
    }
  }
  return c;
}

// Frames `cols` of a clip as a new clip (an empty clip gives black frames), for comparison sheets.
Clip pick_frames(const Clip& src, const std::vector<int>& cols) {
  Clip o;
  o.allocate(kSize, static_cast<int>(cols.size()));
  if (src.frames == 0) return o;
  for (std::size_t k = 0; k < cols.size(); ++k) {
    const auto from = src.frame(cols[k]);
    const auto to = o.frame(static_cast<int>(k));
    for (std::size_t i = 0; i < to.size(); ++i) to[i] = from[i];
  }
  return o;
}

const std::vector<int> kHorizons = {1, 4, 8, 16, 30, 60, 120, 240};

}  // namespace

// --- d-eval -----------------------------------------------------------------------------------------------------------

void step_eval(const Ctx& c) {
  std::vector<std::string> track_rows, stat_rows, long_rows;
  const fs::path fig = c.quick ? c.data / "media" / "d" : fs::path("docs/figures");
  fs::create_directories(fig);
  fs::create_directories(c.data / "media" / "d");
  for (const auto e : sim::kEffects) {
    const bool ex = e == sim::Effect::explosion;
    auto loaded = rollout::load_model(model_path(c, e));
    if (!loaded) throw std::runtime_error(loaded.error());
    const rollout::Model M = std::move(*loaded);
    const std::string en = ename(e);
    // ---- tracking from true start points (held-out runs, their own seeds)
    const int n_runs = c.quick ? 2 : 8, KE = ex ? 89 : (c.quick ? 60 : 240), warm = ex ? 1 : 100;
    const std::vector<std::string> methods = {"neural", "neural_dynamics_true_renderer", "coarse_sim_detail", "coarse_sim", "frozen"};
    std::vector<std::vector<std::vector<double>>> curves(methods.size(), std::vector<std::vector<double>>(static_cast<std::size_t>(n_runs)));
    std::vector<std::vector<double>> renderer_only(static_cast<std::size_t>(n_runs));
    std::vector<std::array<Clip, 4>> shown(1);
    Clip shown_ref;
    std::atomic<int> next{0};
    {
      std::vector<std::jthread> pool;
      for (int t = 0; t < c.threads; ++t) {
        pool.emplace_back([&] {
          for (int r; (r = next++) < n_runs;) {
            const sim::Params p = run_params(e, static_cast<std::uint64_t>(r), kTestSalt);
            sim::Fluid truth(p);
            for (int i = 0; i < warm; ++i) truth.step_frame();
            const sim::State st = truth.state();
            // the true start point, with its fine fields, as the only start point of a copy of the model
            rollout::Model mt = M;
            mt.h.start_fine = kSize;
            rollout::StartPoint sp;
            sp.controls = {p.intensity, p.wind, p.turbulence};
            sp.seed = p.seed;
            sp.time = st.time;
            sp.coarse.resize(static_cast<std::size_t>(kRes) * kRes * rollout::kPhys);
            rollout::coarse_from_sim(st, kRes, p.fps, sp.coarse);
            sp.fine_t = st.temp;
            sp.fine_d = st.soot;
            mt.starts = {sp};
            Clip ref;
            ref.allocate(kSize, KE);
            ref.fps = 30.f;
            std::vector<std::uint8_t> learned(kSize * kSize * 4);
            for (int i = 0; i < KE; ++i) {
              truth.step_frame();
              truth.render(ref.frame(i));
              if (i < 30) {  // the renderer alone, on the true fields
                const sim::State s2 = truth.state();
                rollout::State rs;
                rs.res = kRes;
                rs.size = kSize;
                rs.coarse.assign(static_cast<std::size_t>(kRes) * kRes * static_cast<std::size_t>(M.h.channels()), 0.f);
                std::vector<float> cc(static_cast<std::size_t>(kRes) * kRes * rollout::kPhys);
                rollout::coarse_from_sim(s2, kRes, p.fps, cc);
                for (int j = 0; j < kRes * kRes; ++j) {
                  for (int k = 0; k < rollout::kPhys; ++k) {
                    rs.coarse[static_cast<std::size_t>(j * M.h.channels() + k)] = cc[static_cast<std::size_t>(j * rollout::kPhys + k)];
                  }
                }
                rs.fine_t = s2.temp;
                rs.fine_d = s2.soot;
                std::vector<float> rgba(kSize * kSize * 4);
                rollout::render(M, rs, rgba);
                for (std::size_t q = 0; q < rgba.size(); ++q) learned[q] = static_cast<std::uint8_t>(rgba[q] * 255.f + 0.5f);
                renderer_only[static_cast<std::size_t>(r)].push_back(apsnr(ref.frame(i), learned));
              }
            }
            Effect fx(mt);
            const Clip neural = runtime_clip(fx.e, sp.controls, 0, 0, KE + 1);  // frame 0 is the start point itself
            Clip dyn;  // the same dynamics (reference implementation), drawn by the simulation's renderer
            dyn.allocate(kSize, KE);
            {
              rollout::State rs = rollout::start(mt, 0, kSize, sp.controls, sp.seed);
              for (int i = 0; i < KE; ++i) {
                rollout::step(mt, rs, sp.controls, sp.seed);
                render_fields(p, rs.fine_t, rs.fine_d, dyn.frame(i));
              }
            }
            Clip cs_detail, cs_plain;
            cs_detail.allocate(kSize, KE);
            cs_plain.allocate(kSize, KE);
            CoarseSim cs(M, p, coarse_of(st), st.temp, st.soot);
            for (int i = 0; i < KE; ++i) cs.step(cs_detail.frame(i), cs_plain.frame(i));
            std::vector<std::uint8_t> frozen(kSize * kSize * 4);
            {
              sim::Fluid f0(p);
              f0.set_state(st);
              f0.render(frozen);
            }
            for (int i = 0; i < KE; ++i) {
              curves[0][static_cast<std::size_t>(r)].push_back(apsnr(ref.frame(i), neural.frame(i + 1)));
              curves[1][static_cast<std::size_t>(r)].push_back(apsnr(ref.frame(i), dyn.frame(i)));
              curves[2][static_cast<std::size_t>(r)].push_back(apsnr(ref.frame(i), cs_detail.frame(i)));
              curves[3][static_cast<std::size_t>(r)].push_back(apsnr(ref.frame(i), cs_plain.frame(i)));
              curves[4][static_cast<std::size_t>(r)].push_back(apsnr(ref.frame(i), frozen));
            }
            if (r == 0) {
              shown_ref = ref;
              shown[0] = {slice_clip(neural, 1, KE), cs_detail, cs_plain, ref};
            }
          }
        });
      }
    }
    for (std::size_t mth = 0; mth < methods.size(); ++mth) {
      for (int r = 0; r < n_runs; ++r) {
        std::string row = std::format("{},{},{}", en, r, methods[mth]);
        for (const int hz : kHorizons) row += hz <= KE ? std::format(",{:.3f}", curves[mth][static_cast<std::size_t>(r)][static_cast<std::size_t>(hz - 1)]) : ",";
        track_rows.push_back(row);
      }
    }
    for (int r = 0; r < n_runs; ++r) {
      double mean = 0;
      for (const double v : renderer_only[static_cast<std::size_t>(r)]) mean += v / static_cast<double>(renderer_only[static_cast<std::size_t>(r)].size());
      std::string row = std::format("{},{},renderer_on_true_fields", en, r);
      for (const int hz : kHorizons) row += hz <= 30 ? std::format(",{:.3f}", renderer_only[static_cast<std::size_t>(r)][static_cast<std::size_t>(hz - 1)]) : std::format(",{:.3f}", mean);
      track_rows.push_back(row);
    }
    {  // figure: truth, neural, coarse sim + detail, coarse sim at frames 1, 8, 30, 60, (120, 240)
      std::vector<int> cols = {0, 7, 29, 59};
      if (KE >= 240) cols = {0, 7, 29, 59, 119, 239};
      if (ex) cols = {0, 7, 29, 59, 88};
      const auto pick = [&](const Clip& src) { return pick_frames(src, cols); };
      const Clip a = pick(shown_ref), b = pick(shown[0][0]), cc = pick(shown[0][1]), d = pick(shown[0][2]);
      std::vector<const Clip*> rows = {&a, &b, &cc, &d};
      if (auto w = write_png(fig / std::format("d_{}_track.png", en), comparison_sheet(rows, static_cast<int>(cols.size()), Background::black)); !w) {
        std::println("figure: {}", w.error());
      }
    }
    std::println("d-eval: {} tracking done", en);
    // ---- endless runs at held-out settings, new seeds: statistics against real runs
    const auto settings = b_test_settings();
    const int n_set = c.quick ? 2 : static_cast<int>(settings.size());
    const int F = ex ? 89 : (c.quick ? 90 : 300), real_warm = ex ? 1 : 150;
    std::unique_ptr<Effect> grid;
    if (fs::exists(c.data / "models" / "b" / std::format("{}_grid_k8.nvfx", en))) grid = std::make_unique<Effect>(c.data / "models" / "b" / std::format("{}_grid_k8.nvfx", en));
    Effect fx(model_path(c, e));
    struct SetResult {
      std::map<std::string, metrics::StatDistance> d;
      Clip real, neural, grid, flip;
    };
    std::vector<SetResult> results(static_cast<std::size_t>(n_set));
    next = 0;
    {
      std::vector<std::jthread> pool;
      for (int t = 0; t < c.threads; ++t) {
        pool.emplace_back([&] {
          for (int si; (si = next++) < n_set;) {
            const auto& s = settings[static_cast<std::size_t>(si)];
            const auto real_run = [&](std::uint64_t seed) {
              sim::Params p;
              p.effect = e;
              p.intensity = s[0];
              p.wind = s[1];
              p.turbulence = s[2];
              p.seed = seed;
              p.size = kSize;
              sim::Fluid f(p);
              for (int i = 0; i < real_warm; ++i) f.step_frame();
              Clip cl;
              cl.allocate(kSize, F);
              cl.fps = 30.f;
              for (int i = 0; i < F; ++i) {
                f.step_frame();
                f.render(cl.frame(i));
              }
              return std::make_pair(cl, f.state());
            };
            SetResult& res = results[static_cast<std::size_t>(si)];
            const auto [ra, sa] = real_run(900000 + static_cast<std::uint64_t>(si));
            const auto [rb, sb] = real_run(910000 + static_cast<std::uint64_t>(si));
            (void)sa;
            (void)sb;
            res.real = ra;
            const metrics::ClipStats ref = metrics::stats(ra);
            res.d["real_other_seed"] = metrics::distance(ref, metrics::stats(rb));
            const std::vector<float> ctl{s[0], s[1], s[2]};
            res.neural = runtime_clip(fx.e, ctl, -1, 920000 + static_cast<std::uint64_t>(si), F);
            res.d["neural"] = metrics::distance(ref, metrics::stats(res.neural));
            if (grid) {
              res.grid = runtime_clip(grid->e, ctl, -1, 1, F);
              res.d["grid_k8"] = metrics::distance(ref, metrics::stats(res.grid));
            }
            // the nearest training clip of study B's flipbook library, looped
            const auto train = b_train_settings();
            std::size_t best = 0;
            float bd = 1e9f;
            for (std::size_t k = 0; k < train.size(); ++k) {
              float dd = 0;
              for (int q = 0; q < 3; ++q) dd += (train[k][static_cast<std::size_t>(q)] - s[static_cast<std::size_t>(q)]) * (train[k][static_cast<std::size_t>(q)] - s[static_cast<std::size_t>(q)]);
              if (dd < bd) {
                bd = dd;
                best = k;
              }
            }
            const auto& tb = train[best];
            const fs::path lib = c.data / "clips" / "b" / std::format("{}_train_{:.2f}_{:.2f}_{:.2f}.nfxclip", en, tb[0], tb[1], tb[2]);
            if (auto clip = read_clip(lib)) {
              Clip looped;
              looped.allocate(kSize, F);
              for (int i = 0; i < F; ++i) {
                const int k = ex ? std::min(i, clip->frames - 1) : i % clip->frames;
                const auto from = clip->frame(k);
                const auto to = looped.frame(i);
                for (std::size_t q = 0; q < to.size(); ++q) to[q] = from[q];
              }
              res.flip = looped;
              res.d["flipbook_nearest"] = metrics::distance(ref, metrics::stats(looped));
            }
            {  // coarse simulation + detail, grown from its own warm-up
              sim::Params p;
              p.effect = e;
              p.intensity = s[0];
              p.wind = s[1];
              p.turbulence = s[2];
              p.seed = 930000 + static_cast<std::uint64_t>(si);
              p.size = kSize;
              sim::Params q = p;
              q.sim_res = kRes;
              sim::Fluid w(q);
              for (int i = 0; i < real_warm; ++i) w.step_frame();
              const sim::State cst = w.state();
              std::vector<float> ft(static_cast<std::size_t>(kSize) * kSize), fd(ft.size());
              for (int y = 0; y < kSize; ++y) {
                for (int x = 0; x < kSize; ++x) {
                  const auto i = static_cast<std::size_t>(y) * kSize + static_cast<std::size_t>(x);
                  const auto o = static_cast<std::size_t>(y / (kSize / kRes)) * kRes + static_cast<std::size_t>(x / (kSize / kRes));
                  ft[i] = cst.temp[o];
                  fd[i] = cst.soot[o];
                }
              }
              CoarseSim cs(M, p, cst, ft, fd);
              Clip cd, cp;
              cd.allocate(kSize, F);
              cp.allocate(kSize, F);
              for (int i = 0; i < F; ++i) cs.step(cd.frame(i), cp.frame(i));
              res.d["coarse_sim_detail"] = metrics::distance(ref, metrics::stats(cd));
              res.d["coarse_sim"] = metrics::distance(ref, metrics::stats(cp));
            }
          }
        });
      }
    }
    for (int si = 0; si < n_set; ++si) {
      for (const auto& [name, d] : results[static_cast<std::size_t>(si)].d) {
        stat_rows.push_back(std::format("{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.3f}", en, si, name, d.spectrum_l1, d.motion_ratio, d.coverage_l1, d.emission_l1, d.mean_frame_psnr));
      }
    }
    {  // figure: real (held-out setting), neural (new seed), the B control model, the nearest flipbook, across the run
      std::vector<int> cols = {0, 29, 59, 119, 179, 239, 299};
      if (ex || F < 300) cols = {0, 14, 29, 44, 59, std::min(74, F - 1), F - 1};
      const auto pick = [&](const Clip& src) { return pick_frames(src, cols); };
      const SetResult& r0 = results[0];
      const Clip a = pick(r0.real), b = pick(r0.neural), cc = pick(r0.grid), d = pick(r0.flip);
      std::vector<const Clip*> rows = {&a, &b, &cc, &d};
      if (auto w = write_png(fig / std::format("d_{}_endless.png", en), comparison_sheet(rows, static_cast<int>(cols.size()), Background::black)); !w) {
        std::println("figure: {}", w.error());
      }
      std::vector<const Clip*> vids = {&r0.real, &r0.neural};
      (void)write_comparison_video(c.data / "media" / "d" / std::format("{}_endless.mp4", en), vids, Background::black, 2, 1);
    }
    std::println("d-eval: {} endless runs done", en);
    // ---- long run: 60 s of the neural effect at two held-out settings, statistics per 10 s window against real 10 s
    if (!ex) {
      const int windows = c.quick ? 2 : 6, W = 300;
      for (int si = 0; si < std::min(2, n_set); ++si) {
        const auto& s = settings[static_cast<std::size_t>(si)];
        const std::vector<float> ctl{s[0], s[1], s[2]};
        const metrics::ClipStats ref = metrics::stats(results[static_cast<std::size_t>(si)].real);
        const Clip longrun = runtime_clip(fx.e, ctl, -1, 940000 + static_cast<std::uint64_t>(si), W * windows);
        for (int w = 0; w < windows; ++w) {
          const metrics::StatDistance d = metrics::distance(ref, metrics::stats(slice_clip(longrun, w * W, W)));
          long_rows.push_back(std::format("{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.3f}", en, si, w, d.spectrum_l1, d.motion_ratio, d.coverage_l1, d.emission_l1, d.mean_frame_psnr));
        }
      }
    }
  }
  std::string th = "effect,run,method";
  for (const int hz : kHorizons) th += std::format(",f{}", hz);
  write_csv(c.results / "d_track.csv", th, track_rows);
  write_csv(c.results / "d_stats.csv", "effect,setting,method,spectrum_l1,motion_ratio,coverage_l1,emission_l1,mean_frame_psnr", stat_rows);
  write_csv(c.results / "d_long.csv", "effect,setting,window,spectrum_l1,motion_ratio,coverage_l1,emission_l1,mean_frame_psnr", long_rows);
}

// --- d-timing ---------------------------------------------------------------------------------------------------------

void step_timing(const Ctx& c) {
  cpu_set_t old, one;
  sched_getaffinity(0, sizeof(old), &old);
  CPU_ZERO(&one);
  CPU_SET(3, &one);
  sched_setaffinity(0, sizeof(one), &one);
  std::vector<std::string> rows;
  const auto median = [](std::vector<double> v) {
    std::ranges::sort(v);
    return std::make_pair(v[v.size() / 2], v[v.size() * 9 / 10]);
  };
  for (const auto e : sim::kEffects) {
    Effect fx(model_path(c, e));
    nvfx_effect_info info{};
    nvfx_effect_get_info(fx.e, &info);
    for (const nvfx_isa isa : {NVFX_ISA_AVX2, NVFX_ISA_BASELINE, NVFX_ISA_AVX512}) {
      if (nvfx_set_isa(isa) != NVFX_OK) continue;
      for (const int size : {64, 128, 256}) {
        if (isa != NVFX_ISA_AVX2 && size != 128) continue;
        nvfx_instance* in = nullptr;
        nvfx_instance_create(fx.e, size, &in);
        const float ctl[3] = {0.6f, 0.5f, 0.6f};
        nvfx_instance_set_controls(in, ctl, 3);
        std::vector<std::uint8_t> buf(static_cast<std::size_t>(size) * size * 4);
        nvfx_render(in, 0.0, buf.data(), static_cast<std::size_t>(size) * 4);
        std::vector<double> ms;
        const int n = e == sim::Effect::explosion ? 80 : 200;
        for (int f = 1; f <= n; ++f) {
          const auto t0 = std::chrono::steady_clock::now();
          nvfx_render(in, f / 30.0, buf.data(), static_cast<std::size_t>(size) * 4);
          ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        // a seek: a restart from a start point (warm-up included) plus stepping into the shard
        const auto t0 = std::chrono::steady_clock::now();
        nvfx_render(in, 0.0, buf.data(), static_cast<std::size_t>(size) * 4);
        const double restart = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const auto [med, p90] = median(ms);
        const char* isa_name = isa == NVFX_ISA_AVX2 ? "avx2" : isa == NVFX_ISA_AVX512 ? "avx512" : "baseline";
        rows.push_back(std::format("rollout_{},{},{},{:.4f},{:.4f},{:.2f},{:.0f},{:.1f},{:.1f},{:.1f}", ename(e), isa_name, size, med, p90, restart,
                                   nvfx_instance_macs_per_pixel(in), static_cast<double>(info.stored_bytes) / 1024.0,
                                   static_cast<double>(info.resident_bytes) / 1024.0, static_cast<double>(nvfx_instance_scratch_bytes(in)) / 1024.0));
        nvfx_instance_free(in);
      }
    }
    nvfx_set_isa(NVFX_ISA_AUTO);
    // the coarse simulation (32 cells) alone, for one 128 x 128 frame (solver step plus its renderer)
    sim::Params p = run_params(e, 0, 5);
    p.sim_res = kRes;
    sim::Fluid f(p);
    std::vector<std::uint8_t> buf(kSize * kSize * 4);
    for (int i = 0; i < 30; ++i) f.step_frame();
    std::vector<double> ms;
    for (int i = 0; i < 100; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      f.step_frame();
      f.render(buf);
      ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    const auto [med, p90] = median(ms);
    rows.push_back(std::format("coarse_sim_{},scalar,128,{:.4f},{:.4f},,,,,", ename(e), med, p90));
    std::println("d-timing: {} done", ename(e));
  }
  sched_setaffinity(0, sizeof(old), &old);
  write_csv(c.results / "d_timing.csv", "model,isa,size,median_ms,p90_ms,restart_ms,macs_px,stored_kb,resident_kb,scratch_kb", rows);
}


// --- report -----------------------------------------------------------------------------------------------------------

namespace {

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

double num(const std::string& s) { return s.empty() ? std::nan("") : std::stod(s); }

std::string iv(const metrics::Interval& v, int prec = 2) {
  return std::format("{:+.{}f} [{:+.{}f}, {:+.{}f}]{}", v.mean, prec, v.lo, prec, v.hi, prec, v.covers_zero() ? " (tie)" : "");
}

}  // namespace

void report(const Ctx& c, std::ostream& md) {
  if (!fs::exists(c.results / "d_track.csv") && !fs::exists(c.results / "d_chaos.csv")) return;
  md << "\n## D: start points and learned dynamics\n\n";
  if (fs::exists(c.results / "d_chaos.csv")) {
    md << "### Chaos horizon (the simulation itself)\n\nActive PSNR against a run of the simulation from a stored start point, by frames after it "
          "(30 per second), mean over runs. Perturbations are relative to the velocity RMS.\n\n| effect | case | 1 | 4 | 8 | 15 | 30 | 60 | 120 | 240 |\n"
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& r : read_csv(c.results / "d_chaos.csv")) {
      md << "| " << r[0] << " | " << r[1] << " |";
      for (const int col : {2, 4, 5, 6, 7, 8, 10, 12}) md << (static_cast<std::size_t>(col) < r.size() && !r[static_cast<std::size_t>(col)].empty() ? " " + r[static_cast<std::size_t>(col)] : " -") << " |";
      md << "\n";
    }
  }
  std::map<std::string, std::vector<std::string>> finish;
  if (fs::exists(c.results / "d_finish.csv")) {
    for (auto& r : read_csv(c.results / "d_finish.csv")) finish[r[0]] = r;
  }
  if (fs::exists(c.results / "d_train.csv")) {
    md << "\n### Training\n\n| effect | runs x frames | minutes of simulation | stepper loss | stepper s | renderer dB | contrast | swirl | start points | KB stored |\n"
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& r : read_csv(c.results / "d_train.csv")) {
      const auto f = finish.find(r[0]);
      const bool fin = f != finish.end();
      md << std::format("| {} | {} x {} | {} | {} | {} | {} | {} | {} | {} | {:.1f} |\n", r[0], r[1], r[2], r[3], r[5], r[6], fin ? f->second[1] : r[7],
                        fin ? f->second[2] : r[8], fin ? f->second[3] : r[9], fin ? f->second[5] : r[11], num(fin ? f->second[6] : r[13]) / 1024.0);
    }
    if (!finish.empty()) md << "\nRenderer, detail constants and start points from `d-finish` (retrained on the trained steppers).\n";
  }
  if (fs::exists(c.results / "d_track.csv")) {
    md << "\n### Tracking a held-out run from its true start point\n\nActive PSNR against the true run, mean over held-out runs (other seeds and "
          "settings), by frames after the start point. `neural`: the rollout effect through the runtime, started from the true state (coarse and "
          "fine) with the run's noise seed. `coarse_sim_detail`: the simulation on the same 32-cell grid with the same detail layer, drawn by the "
          "simulation's own renderer. `neural_dynamics_true_renderer`: the neural dynamics drawn by that renderer too (separates dynamics from "
          "rendering). `renderer_on_true_fields`: the learned renderer on the true fields (frames 1-30, then their mean).\n\n";
    std::map<std::string, std::map<std::string, std::vector<std::vector<double>>>> data;  // effect -> method -> runs x horizons
    for (const auto& r : read_csv(c.results / "d_track.csv")) {
      std::vector<double> v;
      for (std::size_t k = 3; k < r.size(); ++k) v.push_back(num(r[k]));
      data[r[0]][r[2]].push_back(v);
    }
    for (const auto& [effect, methods] : data) {
      md << "**" << effect << "**\n\n| method | 1 | 4 | 8 | 16 | 30 | 60 | 120 | 240 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n";
      for (const std::string name : {"neural", "neural_dynamics_true_renderer", "coarse_sim_detail", "coarse_sim", "frozen", "renderer_on_true_fields"}) {
        const auto it = methods.find(name);
        if (it == methods.end()) continue;
        md << "| " << name << " |";
        for (std::size_t k = 0; k < kHorizons.size(); ++k) {
          double sum = 0;
          int n = 0;
          for (const auto& run : it->second) {
            if (k < run.size() && std::isfinite(run[k])) {
              sum += run[k];
              ++n;
            }
          }
          md << (n ? std::format(" {:.2f}", sum / n) : std::string(" -")) << " |";
        }
        md << "\n";
      }
      for (const std::string lhs : {"neural", "neural_dynamics_true_renderer"}) {
        if (!methods.contains(lhs) || !methods.contains("coarse_sim_detail")) continue;
        md << "\n" << lhs << " - coarse_sim_detail, paired over runs:";
        for (const std::size_t k : {std::size_t{0}, std::size_t{2}, std::size_t{4}, std::size_t{5}, std::size_t{7}}) {
          std::vector<double> a, b;
          for (std::size_t r = 0; r < methods.at(lhs).size(); ++r) {
            const auto& x = methods.at(lhs)[r];
            const auto& y = methods.at("coarse_sim_detail")[r];
            if (k < x.size() && k < y.size() && std::isfinite(x[k]) && std::isfinite(y[k])) {
              a.push_back(x[k]);
              b.push_back(y[k]);
            }
          }
          if (a.size() >= 2) md << std::format(" frame {}: {};", kHorizons[k], iv(metrics::paired_bootstrap(a, b)));
        }
        md << "\n";
      }
      md << "\n";
    }
  }
  if (fs::exists(c.results / "d_stats.csv")) {
    md << "### Endless runs at held-out settings\n\nThe ten held-out settings of study B, a new seed each, 10 s per run (explosions: their 3 s). "
          "Distances of each method's frame statistics to a real run of the simulation at that setting; `real_other_seed` is a second real run "
          "with another seed, the floor. Lower is better except motion ratio (1 is right) and mean-frame PSNR (higher is closer).\n\n";
    std::map<std::string, std::map<std::string, std::vector<std::array<double, 5>>>> data;
    for (const auto& r : read_csv(c.results / "d_stats.csv")) data[r[0]][r[2]].push_back({num(r[3]), num(r[4]), num(r[5]), num(r[6]), num(r[7])});
    for (const auto& [effect, methods] : data) {
      md << "**" << effect << "**\n\n| method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |\n|---|---:|---:|---:|---:|---:|\n";
      for (const std::string name : {"real_other_seed", "neural", "grid_k8", "flipbook_nearest", "coarse_sim_detail", "coarse_sim"}) {
        const auto it = methods.find(name);
        if (it == methods.end()) continue;
        std::array<double, 5> m{};
        for (const auto& v : it->second) {
          for (std::size_t k = 0; k < 5; ++k) m[k] += v[k] / static_cast<double>(it->second.size());
        }
        md << std::format("| {} | {:.3f} | {:.2f} | {:.4f} | {:.4f} | {:.2f} |\n", name, m[0], m[1], m[2], m[3], m[4]);
      }
      const auto paired = [&](const std::string& a, const std::string& b, std::size_t k) -> std::string {
        if (!methods.contains(a) || !methods.contains(b) || methods.at(a).size() != methods.at(b).size()) return "-";
        std::vector<double> x, y;
        for (std::size_t i = 0; i < methods.at(a).size(); ++i) {
          x.push_back(methods.at(a)[i][k]);
          y.push_back(methods.at(b)[i][k]);
        }
        return iv(metrics::paired_bootstrap(x, y), 3);
      };
      md << "\nSpectrum distance, paired over settings: neural - grid_k8 " << paired("neural", "grid_k8", 0) << "; neural - flipbook_nearest "
         << paired("neural", "flipbook_nearest", 0) << "; neural - coarse_sim_detail " << paired("neural", "coarse_sim_detail", 0)
         << "; neural - real_other_seed " << paired("neural", "real_other_seed", 0) << ".\n\n";
    }
  }
  if (fs::exists(c.results / "d_long.csv")) {
    md << "### One minute without a restart\n\nStatistics of each 10 s window of a 60 s neural run against a real 10 s run at the same setting (two "
          "held-out settings): if the dynamics drifted, the distances would grow window by window.\n\n| effect | setting | window | spectrum L1 | "
          "motion ratio | coverage L1 | mean-frame PSNR |\n|---|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& r : read_csv(c.results / "d_long.csv")) md << std::format("| {} | {} | {} | {} | {} | {} | {} |\n", r[0], r[1], r[2], r[3], r[4], r[5], r[7]);
  }
  if (fs::exists(c.results / "d_timing.csv")) {
    md << "\n### Cost\n\nMedian (90th percentile) ms per frame through `nvfx_render` on one pinned core; `restart` is a seek (a restart from a "
          "start point, including any warm-up). The coarse simulation row is the 32-cell solver plus its renderer at 128 px (scalar code).\n\n"
          "| model | ISA | size | ms | restart ms | MAC/px | KB stored / resident | KB scratch |\n|---|---|---:|---:|---:|---:|---|---:|\n";
    for (const auto& r : read_csv(c.results / "d_timing.csv")) {
      md << std::format("| {} | {} | {} | {} ({}) | {} | {} | {} | {} |\n", r[0], r[1], r[2], r[3], r[4], r[5].empty() ? "-" : r[5], r[6].empty() ? "-" : r[6],
                        r[7].empty() ? std::string("-") : r[7] + " / " + r[8], r[9].empty() ? "-" : r[9]);
    }
  }
}

}  // namespace nfx::study_d
