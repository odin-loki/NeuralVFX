// Study I (docs/COMPOSE.md §9): couplings in training. Part of nvfx_experiment.
//
// Study D's rollout effects (v1) were each trained alone; in a composed scene a push, a force field, a transfer or a
// hand-over gives a model states and flows it never saw. Here v1's stepper is fine-tuned on a mix of its own training
// runs and runs of the simulation with the same operations applied between frames (forced runs), and, for smoke, runs
// that continue an explosion's state (hand-over runs). Only the stepper changes; renderer, detail constants, start
// points and normalisation stay v1's.
//
//   i-data    per effect: plain runs (study D's recipe, salt 1), forced runs (salt 1, random couplings), hand-over runs
//             (smoke, from salt-1 explosions); saved under DATA/runs
//   i-probe   seconds per fine-tuning iteration (to size the runs)
//   i-train   one candidate: v1's stepper fine-tuned (--share S --lr L --iters N --checkpoint K --batch B), a model saved
//             at every checkpoint under DATA/cand
//   i-val     validation (salt-3 runs, study G's 10 validation settings, validation forcing seeds): v1 and every
//             candidate; the choice of v2c by the rule written in §9, copied to DATA/v2c
//   i-test    once: v1, v2c and the plain control (v2p) on study B's held-out settings and study D's held-out runs with
//             new seeds and held-out forcing seeds; the decisions by the rule of §9
#include "experiment_i.hpp"

#include "compose.hpp"

#include <neuralfx/image_io.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>
#include <neuralfx/train.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <map>
#include <print>
#include <random>
#include <set>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;
using namespace nfx;

namespace nfx::study_i {

namespace {

constexpr int kSize = 128, kRes = 32;
constexpr std::uint64_t kTrainSalt = 1, kTestSalt = 2, kValSalt = 3;

std::size_t sz(int v) { return static_cast<std::size_t>(v); }
std::string ename(sim::Effect e) { return std::string(sim::effect_name(e)); }

bool wanted(const Ctx& c, sim::Effect e) {
  if (c.effects.empty()) return true;
  const std::string name = ename(e);
  std::size_t a = 0;
  while (a <= c.effects.size()) {
    const std::size_t b = std::min(c.effects.find(',', a), c.effects.size());
    if (c.effects.substr(a, b - a) == name) return true;
    a = b + 1;
  }
  return false;
}

double seconds_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

// Rows keyed by their first cells (the effect, and the model when given): rows of other effects already in the file are
// kept, so per-effect invocations merge.
void merge_csv(const fs::path& path, const std::string& header, const std::vector<std::string>& rows, const std::set<std::string>& effects) {
  std::vector<std::string> keep;
  if (std::ifstream in(path); in) {
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      if (!line.empty() && !effects.contains(line.substr(0, line.find(',')))) keep.push_back(line);
    }
  }
  fs::create_directories(path.parent_path());
  std::ofstream o(path);
  o << header << "\n";
  for (const auto& r : keep) o << r << "\n";
  for (const auto& r : rows) o << r << "\n";
}

// --- the study's data sizes per effect -----------------------------------------------------------------------------

struct Sizes {
  int plain = 96, forced = 96, handover = 0, frames = 240, handover_frames = 120, first = 15;
};
Sizes sizes(const Ctx& c, sim::Effect e) {
  Sizes s;
  if (e == sim::Effect::smoke) {
    s.forced = 64;
    s.handover = 64;
  }
  if (e == sim::Effect::explosion) {
    s.plain = 144;
    s.forced = 144;
    s.frames = 90;
    s.first = 0;
  }
  if (c.quick) {
    s.plain = 6;
    s.forced = 6;
    s.handover = s.handover ? 4 : 0;
    s.frames = std::min(s.frames, 60);
    s.handover_frames = 40;
  }
  return s;
}

rollout::SimRecipe recipe(sim::Effect e, std::uint64_t salt) {
  rollout::SimRecipe r = rollout::recipe_for(e);
  r.salt = salt;
  return r;
}

// --- runs on disk ------------------------------------------------------------------------------------------------------

template <class T>
void put(std::ostream& o, const T& v) {
  o.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <class T>
void get(std::istream& i, T& v) {
  i.read(reinterpret_cast<char*>(&v), sizeof(T));
}
template <class T>
void put_vec(std::ostream& o, const std::vector<T>& v) {
  put(o, static_cast<std::uint64_t>(v.size()));
  o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}
template <class T>
void get_vec(std::istream& i, std::vector<T>& v) {
  std::uint64_t n = 0;
  get(i, n);
  v.resize(n);
  i.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
}

void save_runs(const fs::path& path, const std::vector<rollout::Run>& runs) {
  fs::create_directories(path.parent_path());
  const fs::path tmp = path.string() + ".tmp";
  {
    std::ofstream o(tmp, std::ios::binary);
    o.write("NVFXIRN1", 8);
    put(o, static_cast<std::uint64_t>(runs.size()));
    for (const rollout::Run& r : runs) {
      put(o, static_cast<std::int32_t>(r.p.effect));
      put(o, r.p.intensity);
      put(o, r.p.wind);
      put(o, r.p.turbulence);
      put(o, r.p.seed);
      put(o, r.p.size);
      put(o, r.p.sim_res);
      put(o, r.p.fps);
      put(o, r.frames);
      put(o, r.t0);
      put_vec(o, r.coarse);
      put_vec(o, r.forcing_at);
      put_vec(o, r.forcing);
    }
    if (!o) throw std::runtime_error("cannot write " + tmp.string());
  }
  fs::rename(tmp, path);
}

std::vector<rollout::Run> load_runs(const fs::path& path) {
  std::ifstream i(path, std::ios::binary);
  char magic[8];
  i.read(magic, 8);
  if (!i || std::memcmp(magic, "NVFXIRN1", 8) != 0) throw std::runtime_error("not a run file: " + path.string() + " (run i-data first)");
  std::uint64_t n = 0;
  get(i, n);
  std::vector<rollout::Run> runs(n);
  for (rollout::Run& r : runs) {
    std::int32_t e = 0;
    get(i, e);
    r.p.effect = static_cast<sim::Effect>(e);
    get(i, r.p.intensity);
    get(i, r.p.wind);
    get(i, r.p.turbulence);
    get(i, r.p.seed);
    get(i, r.p.size);
    get(i, r.p.sim_res);
    get(i, r.p.fps);
    get(i, r.frames);
    get(i, r.t0);
    get_vec(i, r.coarse);
    get_vec(i, r.forcing_at);
    get_vec(i, r.forcing);
  }
  if (!i) throw std::runtime_error("truncated run file: " + path.string());
  return runs;
}

fs::path runs_path(const Ctx& c, sim::Effect e, const std::string& kind) { return c.data / "runs" / std::format("{}_{}.bin", ename(e), kind); }

// In parallel over `n` items on c.threads threads.
template <class F>
void parallel(const Ctx& c, int n, F&& f) {
  std::atomic<int> next{0};
  std::vector<std::jthread> pool;
  for (int t = 0; t < std::max(1, c.threads); ++t) {
    pool.emplace_back([&] {
      for (int i; (i = next++) < n;) f(i);
    });
  }
}

rollout::Model load_or_throw(const fs::path& p) {
  auto m = rollout::load_model(p);
  if (!m) throw std::runtime_error(std::format("{}: {}", p.string(), m.error()));
  return std::move(*m);
}

}  // namespace

// --- i-data -----------------------------------------------------------------------------------------------------------

void step_data(const Ctx& c) {
  std::vector<std::string> rows;
  std::set<std::string> done;
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e)) continue;
    const Sizes s = sizes(c, e);
    const rollout::SimRecipe r = recipe(e, kTrainSalt);
    const auto t0 = std::chrono::steady_clock::now();
    // plain: study D's training runs 0..plain-1 (exactly as record_runs records them); files already there are kept
    const bool kept = fs::exists(runs_path(c, e, "plain"));
    if (!kept) {
      std::vector<rollout::Run> plain(sz(s.plain));
      parallel(c, s.plain, [&](int i) { plain[sz(i)] = rollout::record_run(rollout::recipe_run(r, static_cast<std::uint64_t>(i)), s.frames, kRes); });
      save_runs(runs_path(c, e, "plain"), plain);
    }
    const double t_plain = seconds_since(t0);
    // forced: new training runs (indices 200..), each with its own random couplings
    const auto t1 = std::chrono::steady_clock::now();
    std::vector<rollout::Run> forced(sz(s.forced));
    parallel(c, s.forced, [&](int i) {
      const sim::Params p = rollout::recipe_run(r, 200 + static_cast<std::uint64_t>(i));
      const rollout::ForcingSpec spec = rollout::random_forcing(e, s.frames, 1000 + static_cast<std::uint64_t>(i), s.first, s.frames - 10);
      forced[sz(i)] = rollout::record_forced_run(p, spec, s.frames, kRes);
    });
    std::size_t on = 0;
    for (const auto& run : forced) on += static_cast<std::size_t>(std::ranges::count_if(run.forcing_at, [](int k) { return k >= 0; }));
    save_runs(runs_path(c, e, "forced"), forced);
    const double t_forced = seconds_since(t1);
    {  // how often the forced states leave v1's range (the stepper's output is clamped to it)
      const rollout::Model v1 = load_or_throw(c.v1 / std::format("{}.nvfx", ename(e)));
      std::array<std::size_t, rollout::kPhys> out{};
      std::size_t n = 0;
      for (const auto& run : forced) {
        for (std::size_t j = 0; j < run.coarse.size(); j += rollout::kPhys) {
          for (int k = 0; k < rollout::kPhys; ++k) out[sz(k)] += run.coarse[j + sz(k)] < v1.lo[sz(k)] || run.coarse[j + sz(k)] > v1.hi[sz(k)];
          ++n;
        }
      }
      std::println("i-data: {} forced cell values outside v1's range: u {:.2e} v {:.2e} heat {:.2e} soot {:.2e}", ename(e), static_cast<double>(out[0]) / static_cast<double>(n),
                   static_cast<double>(out[1]) / static_cast<double>(n), static_cast<double>(out[2]) / static_cast<double>(n), static_cast<double>(out[3]) / static_cast<double>(n));
    }
    rows.push_back(std::format("{},plain,{},{},0,{}", ename(e), s.plain, s.frames, kept ? std::string() : std::format("{:.0f}", t_plain)));
    rows.push_back(std::format("{},forced,{},{},{:.3f},{:.0f}", ename(e), s.forced, s.frames,
                               static_cast<double>(on) / std::max(1.0, static_cast<double>(s.forced) * s.frames), t_forced));
    std::println("i-data: {} plain {} x {} ({:.0f} s), forced {} ({:.0f}% of frames forced, {:.0f} s)", ename(e), s.plain, s.frames, t_plain, s.forced,
                 100.0 * static_cast<double>(on) / std::max(1.0, static_cast<double>(s.forced) * s.frames), t_forced);
    if (s.handover > 0) {  // explosions of the training salt handed to the smoke simulation after 0.8 to 3 s
      const auto t2 = std::chrono::steady_clock::now();
      const rollout::SimRecipe rx = recipe(sim::Effect::explosion, kTrainSalt);
      std::vector<rollout::Run> hand(sz(s.handover));
      parallel(c, s.handover, [&](int i) {
        const sim::Params from = rollout::recipe_run(rx, 300 + static_cast<std::uint64_t>(i));
        const sim::Params p = rollout::recipe_run(r, 300 + static_cast<std::uint64_t>(i));
        const int before = 24 + static_cast<int>((static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ULL >> 40) % 67);
        hand[sz(i)] = rollout::record_handover_run(from, before, p, s.handover_frames, kRes);
      });
      save_runs(runs_path(c, e, "handover"), hand);
      const double t_hand = seconds_since(t2);
      rows.push_back(std::format("{},handover,{},{},0,{:.0f}", ename(e), s.handover, s.handover_frames, t_hand));
      std::println("i-data: {} hand-over {} x {} ({:.0f} s)", ename(e), s.handover, s.handover_frames, t_hand);
    }
    done.insert(ename(e));
    std::fflush(stdout);
  }
  merge_csv(c.results / "i_data.csv", "effect,kind,runs,frames,forced_share,seconds", rows, done);
}

namespace {

// Plain runs first, then the coupled runs (forced; for smoke also hand-over).
std::vector<rollout::Run> load_training_runs(const Ctx& c, sim::Effect e, int& n_plain) {
  std::vector<rollout::Run> runs = load_runs(runs_path(c, e, "plain"));
  n_plain = static_cast<int>(runs.size());
  for (auto& r : load_runs(runs_path(c, e, "forced"))) runs.push_back(std::move(r));
  if (fs::exists(runs_path(c, e, "handover"))) {
    for (auto& r : load_runs(runs_path(c, e, "handover"))) runs.push_back(std::move(r));
  }
  return runs;
}

rollout::StepperOptions finetune_options(const Ctx& c, sim::Effect e, int n_plain) {
  rollout::StepperOptions so = rollout::recipe_for(e).stepper;  // stage 3 of the recipe: burn-in, profiles, activity
  so.iterations = 0;
  so.finetune = 0;
  so.activity_stage = c.iters;
  so.lr_finetune = c.lr;
  so.keep_normalisation = true;
  so.plain_runs = n_plain;
  so.coupled_share = c.share;
  so.batch = c.batch;
  so.threads = c.threads;
  so.seed = 11;
  so.log_every = 25;
  so.stop_after = c.stop;
  if (c.quick) {
    so.burn_max = 8;
    so.max_unroll = 8;
  }
  return so;
}

std::string candidate_tag(const Ctx& c) { return c.tag.empty() ? std::format("s{:.2f}_lr{:.0e}", c.share, c.lr) : c.tag; }

}  // namespace

// --- i-probe ----------------------------------------------------------------------------------------------------------

void step_probe(const Ctx& c) {
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e)) continue;
    int n_plain = 0;
    const std::vector<rollout::Run> runs = load_training_runs(c, e, n_plain);
    rollout::Model m = load_or_throw(c.v1 / std::format("{}.nvfx", ename(e)));
    for (const float share : {0.f, 1.f}) {
      Ctx cc = c;
      cc.share = share;
      cc.iters = c.quick ? 3 : 6;
      rollout::StepperOptions so = finetune_options(cc, e, n_plain);
      rollout::Model mm = m;
      const auto t0 = std::chrono::steady_clock::now();
      rollout::train_stepper(mm, runs, so);
      std::println("i-probe: {} share {} batch {} threads {}: {:.2f} s per iteration", ename(e), share, so.batch, so.threads, seconds_since(t0) / cc.iters);
      std::fflush(stdout);
    }
  }
}

// --- i-train ----------------------------------------------------------------------------------------------------------

void step_train(const Ctx& c) {
  if (!train::cpu_supported()) throw std::runtime_error("training needs AVX2 + FMA");
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e)) continue;
    int n_plain = 0;
    const std::vector<rollout::Run> runs = load_training_runs(c, e, n_plain);
    rollout::Model m = load_or_throw(c.v1 / std::format("{}.nvfx", ename(e)));
    const std::vector<float> keep_render = m.render_w;
    const auto keep_starts = m.starts.size();
    const std::string tag = candidate_tag(c);
    rollout::StepperOptions so = finetune_options(c, e, n_plain);
    const fs::path dir = c.data / "cand";
    fs::create_directories(dir);
    std::ofstream curve(c.data / "logs" / std::format("train_{}_{}.csv", ename(e), tag));
    so.progress = [&](int it, int unroll, double loss) {
      std::println("  {} {} {:5d} unroll {:2d} loss {:.5f}", ename(e), tag, it, unroll, loss);
      std::fflush(stdout);
      curve << it << "," << loss << "\n";
      curve.flush();
    };
    so.checkpoint_every = c.checkpoint;
    so.checkpoint = [&](int done) {
      const fs::path out = dir / std::format("{}_{}_it{}.nvfx", ename(e), tag, done);
      if (auto w = rollout::save_model(out, m); !w) throw std::runtime_error(w.error());
      std::println("  saved {}", out.string());
      std::fflush(stdout);
    };
    std::println("i-train: {} {}: {} plain + {} coupled runs, share {}, lr {}, {} iterations, batch {}", ename(e), tag, n_plain, runs.size() - sz(n_plain),
                 c.share, c.lr, c.iters, c.batch);
    const rollout::StepperResult sr = rollout::train_stepper(m, runs, so);
    if (m.render_w != keep_render || m.starts.size() != keep_starts) throw std::logic_error("only the stepper may change");
    std::println("i-train: {} {} loss {:.5f} in {:.0f} s", ename(e), tag, sr.final_loss, sr.seconds);
    const fs::path log = c.results / "i_train.csv";
    std::vector<std::string> keep;
    if (std::ifstream in(log); in) {
      std::string line;
      std::getline(in, line);
      while (std::getline(in, line)) {
        if (!line.empty() && line.rfind(std::format("{},{},", ename(e), tag), 0) != 0) keep.push_back(line);
      }
    }
    const std::string done = c.stop > 0 ? std::format("{} of {}", c.stop, c.iters) : std::to_string(c.iters);  // a control stops early
    keep.push_back(std::format("{},{},{},{},{},{},{},{},{:.0f}", ename(e), tag, c.share, c.lr, done, c.batch, c.threads,
                               c.stop > 0 ? std::string() : std::format("{:.5f}", sr.final_loss), sr.seconds));
    std::ofstream o(log);
    o << "effect,candidate,share,lr,iterations,batch,threads,final_loss,seconds\n";
    for (const auto& r : keep) o << r << "\n";
  }
}

// --- evaluation ---------------------------------------------------------------------------------------------------------

namespace {

// Active PSNR as study D's (experiment_d.cpp): pixels where either frame shows something.
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

using Setting = std::array<float, 3>;
bool off_grid(const Setting& s) {
  const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
  return off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f});
}
// Study B's ten held-out settings (the test) and study G's ten validation settings (docs/DCM.md §4).
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

sim::Params at_setting(sim::Effect e, const Setting& s, std::uint64_t seed) {
  sim::Params p;
  p.effect = e;
  p.intensity = s[0];
  p.wind = s[1];
  p.turbulence = s[2];
  p.seed = seed;
  p.size = kSize;
  return p;
}

// One tracking case: a run of the simulation from `warm` frames (or an explosion handed to it), tracked for `frames`
// frames, with couplings from a forcing seed (0: none).
struct TrackCase {
  std::string kind;  // forced, handover, plain
  int index = 0;
  sim::Params p;
  int warm = 100, frames = 60;
  std::uint64_t forcing = 0;
  bool handover = false;
  sim::Params from;
  int before = 0;
};

struct Truth {
  sim::State start;
  rollout::ForcingSpec spec;
  std::vector<std::vector<std::uint8_t>> frames;
};

Truth make_truth(const TrackCase& tc) {
  Truth t;
  sim::Params p = tc.p;
  std::unique_ptr<sim::Fluid> f;
  if (tc.handover) {
    sim::Fluid a(tc.from);
    for (int i = 0; i < tc.before; ++i) a.step_frame();
    f = std::make_unique<sim::Fluid>(p);
    f->set_state(a.state());
  } else {
    f = std::make_unique<sim::Fluid>(p);
    for (int i = 0; i < tc.warm; ++i) f->step_frame();
  }
  t.start = f->state();
  if (tc.forcing) t.spec = rollout::random_forcing(p.effect, tc.frames, tc.forcing, 0, 15);
  rollout::SimForcing sf;
  t.frames.assign(sz(tc.frames), std::vector<std::uint8_t>(sz(kSize) * kSize * 4));
  for (int i = 0; i < tc.frames; ++i) {
    const bool on = t.spec.active(i);
    if (on) {
      rollout::forcing_fields(t.spec, p, kSize, i, sf);
      rollout::apply_forcing(*f, sf);
    }
    f->step_frame();
    if (on) rollout::unpush(*f, sf);
    f->render(t.frames[sz(i)]);
  }
  return t;
}

// The model from the true state (coarse and fine, as stored: through save and load) with the run's seed, given the same
// couplings as the truth, through the runtime's runner (as compose drives it); active PSNR per frame.
std::vector<double> track(const rollout::Model& M, const TrackCase& tc, const Truth& t) {
  rollout::Model mt = M;
  mt.h.start_fine = kSize;
  rollout::StartPoint sp;
  sp.controls = {tc.p.intensity, tc.p.wind, tc.p.turbulence};
  sp.seed = tc.p.seed;
  sp.time = t.start.time;
  sp.coarse.resize(sz(kRes) * kRes * rollout::kPhys);
  rollout::coarse_from_sim(t.start, kRes, tc.p.fps, sp.coarse);
  sp.fine_t = t.start.temp;
  sp.fine_d = t.start.soot;
  mt.starts = {sp};
  std::stringstream bytes;
  if (auto w = rollout::save_model(bytes, mt); !w) throw std::runtime_error(w.error());
  auto loaded = rollout::load_model(bytes);
  if (!loaded) throw std::runtime_error(loaded.error());
  rt::RolloutEffect re;
  re.m = std::move(*loaded);
  auto run = compose::make_runner(re, kSize, compose::best_isa());
  const std::vector<float> controls = sp.controls;
  run->begin(0, sp.seed);
  if (tc.handover) run->adopt(re.m.starts[0].time);  // as compose::hand_over: pressure and flow dropped, swirl at full strength
  const int C = re.m.h.channels();
  std::vector<std::uint8_t> rgba(sz(kSize) * kSize * 4);
  std::vector<float> cf(sz(kRes) * kRes * rollout::kForce);
  rollout::SimForcing sf;
  std::vector<double> out;
  for (int i = 0; i < tc.frames; ++i) {
    const bool on = t.spec.active(i);
    if (on) {
      rollout::forcing_fields(t.spec, tc.p, kSize, i, sf);
      rollout::coarse_forcing(sf, kRes, tc.p.fps, cf);
      rollout::apply_forcing(run->coarse_mut(), C, cf);
      if (sf.material) {  // the fine fields take the material at full resolution, as compose's transfer does
        auto ft = run->fine_heat_mut();
        auto fd = run->fine_soot_mut();
        for (std::size_t j = 0; j < ft.size(); ++j) {
          ft[j] = std::max(0.f, ft[j] * sf.mk[j] + sf.ah[j]);
          fd[j] = std::max(0.f, fd[j] * sf.mk[j] + sf.as[j]);
        }
      }
    }
    run->step(controls, sp.seed);
    if (on) rollout::remove_push(run->coarse_mut(), C, cf);
    run->render(rt::FrameInput{}, rgba.data(), sz(kSize) * 4);
    out.push_back(apsnr(t.frames[sz(i)], rgba));
  }
  return out;
}

// The models compared, by name.
struct Named {
  std::string name;
  rollout::Model m;
  fs::path file;
};

// Tracking results: [model][case] -> PSNR per frame.
using TrackResults = std::vector<std::vector<std::vector<double>>>;

TrackResults run_tracking(const Ctx& c, const std::vector<Named>& models, const std::vector<TrackCase>& cases) {
  TrackResults res(models.size(), std::vector<std::vector<double>>(cases.size()));
  parallel(c, static_cast<int>(cases.size()), [&](int k) {
    const Truth t = make_truth(cases[sz(k)]);
    for (std::size_t m = 0; m < models.size(); ++m) res[m][sz(k)] = track(models[m].m, cases[sz(k)], t);
  });
  return res;
}

// --- endless statistics (study D's protocol, tools/experiment_d.cpp) ---------------------------------------------------

struct RuntimeEffect {
  nvfx_effect* e = nullptr;
  explicit RuntimeEffect(const fs::path& p) {
    if (nvfx_effect_load(p.string().c_str(), &e) != NVFX_OK) throw std::runtime_error("cannot load " + p.string());
  }
  ~RuntimeEffect() { nvfx_effect_free(e); }
  RuntimeEffect(const RuntimeEffect&) = delete;
  RuntimeEffect& operator=(const RuntimeEffect&) = delete;
};

// `frames` frames of the runtime at controls with a seed, played as shipped (shards from start points).
Clip runtime_clip(nvfx_effect* e, std::span<const float> controls, std::uint64_t seed, int frames) {
  nvfx_instance* in = nullptr;
  if (nvfx_instance_create(e, kSize, &in) != NVFX_OK) throw std::runtime_error("instance");
  nvfx_instance_set_controls(in, controls.data(), static_cast<int>(controls.size()));
  nvfx_instance_set_seed(in, seed);
  Clip c;
  c.allocate(kSize, frames);
  c.fps = 30.f;
  for (int f = 0; f < frames; ++f) nvfx_render(in, f / 30.0, c.frame(f).data(), kSize * 4);
  nvfx_instance_free(in);
  return c;
}

Clip real_clip(sim::Effect e, const Setting& s, std::uint64_t seed, int warm, int frames) {
  sim::Fluid f(at_setting(e, s, seed));
  for (int i = 0; i < warm; ++i) f.step_frame();
  Clip cl;
  cl.allocate(kSize, frames);
  cl.fps = 30.f;
  for (int i = 0; i < frames; ++i) {
    f.step_frame();
    f.render(cl.frame(i));
  }
  return cl;
}

// [model][setting] distances; also the second real run (the floor) as the last row.
std::vector<std::vector<metrics::StatDistance>> run_endless(const Ctx& c, sim::Effect e, const std::vector<Named>& models, const std::vector<Setting>& settings,
                                                           std::uint64_t real_seed, std::uint64_t other_seed, std::uint64_t model_seed) {
  const bool ex = e == sim::Effect::explosion;
  const int F = ex ? 89 : (c.quick ? 90 : 300), warm = ex ? 1 : 150;
  std::vector<std::vector<metrics::StatDistance>> out(models.size() + 1, std::vector<metrics::StatDistance>(settings.size()));
  std::vector<std::unique_ptr<RuntimeEffect>> fx;
  for (const Named& m : models) fx.push_back(std::make_unique<RuntimeEffect>(m.file));
  parallel(c, static_cast<int>(settings.size()), [&](int si) {
    const Setting& s = settings[sz(si)];
    const metrics::ClipStats ref = metrics::stats(real_clip(e, s, real_seed + static_cast<std::uint64_t>(si), warm, F));
    const std::vector<float> ctl{s[0], s[1], s[2]};
    for (std::size_t m = 0; m < models.size(); ++m) {
      out[m][sz(si)] = metrics::distance(ref, metrics::stats(runtime_clip(fx[m]->e, ctl, model_seed + static_cast<std::uint64_t>(si), F)));
    }
    if (other_seed) out[models.size()][sz(si)] = metrics::distance(ref, metrics::stats(real_clip(e, s, other_seed + static_cast<std::uint64_t>(si), warm, F)));
  });
  return out;
}

double mean(const std::vector<double>& v) {
  double s = 0;
  for (const double x : v) s += x;
  return v.empty() ? 0.0 : s / static_cast<double>(v.size());
}

const std::vector<int> kTrackHorizons = {1, 8, 30, 60};
const std::vector<int> kDHorizons = {1, 4, 8, 16, 30, 60, 120, 240};

// forced: B's or the validation settings with couplings; unforced (a diagnostic, outside the rule): the same runs
// without them, which shows what the couplings cost a model.
std::vector<TrackCase> forced_cases(sim::Effect e, const std::vector<Setting>& settings, std::uint64_t seed0, std::uint64_t forcing0, bool quick,
                                    bool couplings = true) {
  std::vector<TrackCase> v;
  const bool ex = e == sim::Effect::explosion;
  for (std::size_t si = 0; si < settings.size(); ++si) {
    for (int k = 0; k < (quick ? 1 : 2); ++k) {
      TrackCase tc;
      tc.kind = couplings ? "forced" : "unforced";
      tc.index = static_cast<int>(v.size());
      tc.p = at_setting(e, settings[si], seed0 + 10 * si + static_cast<std::uint64_t>(k));
      tc.warm = ex ? 1 : 100;
      tc.frames = 60;
      tc.forcing = couplings ? forcing0 + 10 * si + static_cast<std::uint64_t>(k) : 0;
      v.push_back(tc);
    }
  }
  return v;
}

// Explosions (of the given salt's recipe runs) handed to the smoke simulation at a setting after 0.8 to 3 s.
std::vector<TrackCase> handover_cases(const std::vector<Setting>& settings, std::uint64_t salt, std::uint64_t seed0, std::uint64_t mix, bool quick) {
  std::vector<TrackCase> v;
  const rollout::SimRecipe rx = recipe(sim::Effect::explosion, salt);
  for (std::size_t si = 0; si < settings.size(); ++si) {
    for (int k = 0; k < (quick ? 1 : 2); ++k) {
      TrackCase tc;
      tc.kind = "handover";
      tc.index = static_cast<int>(v.size());
      tc.handover = true;
      tc.from = rollout::recipe_run(rx, 400 + static_cast<std::uint64_t>(tc.index));
      tc.before = 24 + static_cast<int>(((static_cast<std::uint64_t>(tc.index) + mix) * 0x9E3779B97F4A7C15ULL >> 40) % 67);
      tc.p = at_setting(sim::Effect::smoke, settings[si], seed0 + 10 * si + static_cast<std::uint64_t>(k));
      tc.p.sim_res = kSize;
      tc.frames = 60;
      v.push_back(tc);
    }
  }
  return v;
}

std::vector<TrackCase> plain_cases(sim::Effect e, std::uint64_t salt, int n, int frames) {
  std::vector<TrackCase> v;
  const bool ex = e == sim::Effect::explosion;
  const rollout::SimRecipe r = recipe(e, salt);
  for (int i = 0; i < n; ++i) {
    TrackCase tc;
    tc.kind = "plain";
    tc.index = i;
    tc.p = rollout::recipe_run(r, static_cast<std::uint64_t>(i));
    tc.warm = ex ? 1 : 100;
    tc.frames = ex ? std::min(frames, 89) : frames;
    v.push_back(tc);
  }
  return v;
}

std::string track_row(const std::string& effect, const std::string& model, const TrackCase& tc, const std::vector<double>& psnr, const std::vector<int>& hz) {
  std::string row = std::format("{},{},{},{}", effect, model, tc.kind, tc.index);
  for (const int h : hz) row += h <= static_cast<int>(psnr.size()) ? std::format(",{:.3f}", psnr[sz(h - 1)]) : ",";
  return row;
}

std::string stat_row(const std::string& effect, const std::string& model, int si, const metrics::StatDistance& d) {
  return std::format("{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.3f}", effect, model, si, d.spectrum_l1, d.motion_ratio, d.coverage_l1, d.emission_l1, d.mean_frame_psnr);
}

std::vector<Named> candidates(const Ctx& c, sim::Effect e) {
  std::vector<Named> v;
  v.push_back({"v1", load_or_throw(c.v1 / std::format("{}.nvfx", ename(e))), c.v1 / std::format("{}.nvfx", ename(e))});
  std::vector<fs::path> files;
  if (fs::exists(c.data / "cand")) {
    for (const auto& f : fs::directory_iterator(c.data / "cand")) {
      const std::string n = f.path().filename().string();
      if (n.starts_with(ename(e) + "_") && f.path().extension() == ".nvfx") files.push_back(f.path());
    }
  }
  std::ranges::sort(files);
  for (const auto& f : files) v.push_back({f.stem().string().substr(ename(e).size() + 1), load_or_throw(f), f});
  return v;
}

}  // namespace

// --- i-val ------------------------------------------------------------------------------------------------------------

void step_val(const Ctx& c) {
  std::vector<std::string> track_rows, stat_rows, choice_rows;
  std::set<std::string> done;
  const auto vs = validation_settings();
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e)) continue;
    const std::string en = ename(e);
    const std::vector<Named> models = candidates(c, e);
    std::println("i-val: {}: {} models", en, models.size());
    std::vector<TrackCase> cases = forced_cases(e, vs, 830000, 870000, c.quick);
    if (e == sim::Effect::smoke) {
      for (TrackCase tc : handover_cases(vs, kValSalt, 860000, 7, c.quick)) cases.push_back(tc);
    }
    for (TrackCase tc : plain_cases(e, kValSalt, c.quick ? 2 : 16, 60)) cases.push_back(tc);
    for (TrackCase tc : forced_cases(e, vs, 830000, 870000, c.quick, false)) cases.push_back(tc);
    const auto t0 = std::chrono::steady_clock::now();
    const TrackResults tr = run_tracking(c, models, cases);
    std::println("i-val: {} tracking in {:.0f} s", en, seconds_since(t0));
    const auto t1 = std::chrono::steady_clock::now();
    const auto st = run_endless(c, e, models, vs, 840000, 0, 850000);
    std::println("i-val: {} endless runs in {:.0f} s", en, seconds_since(t1));
    for (std::size_t m = 0; m < models.size(); ++m) {
      for (std::size_t k = 0; k < cases.size(); ++k) track_rows.push_back(track_row(en, models[m].name, cases[k], tr[m][k], kTrackHorizons));
      for (std::size_t si = 0; si < vs.size(); ++si) stat_rows.push_back(stat_row(en, models[m].name, static_cast<int>(si), st[m][si]));
    }
    // The choice (docs/COMPOSE.md §9): the coupled candidate (share > 0) with the best mean of the coupled tracking
    // PSNR at 8 and 30 frames (smoke: forced and hand-over cases alike), among those within the guards against v1 on
    // validation: plain tracking (mean over 1, 8, 30, 60 frames) at most 0.1 dB lower, endless spectrum distance at
    // most 0.01 higher, mean |log motion ratio| at most 0.03 higher, coverage distance at most 0.005 higher,
    // mean-frame PSNR at most 0.3 dB lower. If none is within the guards, the best coupled score is taken anyway.
    struct Summary {
      double coupled = 0, plain = 0, spectrum = 0, motion = 0, coverage = 0, mfp = 0;
    };
    std::vector<Summary> sum(models.size());
    for (std::size_t m = 0; m < models.size(); ++m) {
      std::vector<double> cp, pl;
      for (std::size_t k = 0; k < cases.size(); ++k) {
        const auto& v = tr[m][k];
        if (cases[k].kind == "plain") pl.push_back((v[0] + v[7] + v[29] + v[59]) / 4.0);
        if (cases[k].kind == "forced" || cases[k].kind == "handover") cp.push_back((v[7] + v[29]) / 2.0);
      }
      sum[m].coupled = mean(cp);
      sum[m].plain = mean(pl);
      std::vector<double> sp, mo, co, mf;
      for (const auto& d : st[m]) {
        sp.push_back(d.spectrum_l1);
        mo.push_back(std::abs(std::log(std::max(1e-3, d.motion_ratio))));
        co.push_back(d.coverage_l1);
        mf.push_back(d.mean_frame_psnr);
      }
      sum[m] = {sum[m].coupled, sum[m].plain, mean(sp), mean(mo), mean(co), mean(mf)};
    }
    int best = -1, best_any = -1;
    for (std::size_t m = 1; m < models.size(); ++m) {
      if (models[m].name.starts_with("s0.00")) continue;  // the plain control is not a coupled candidate
      const Summary& a = sum[m];
      const Summary& b = sum[0];
      const bool guards = a.plain >= b.plain - 0.1 && a.spectrum <= b.spectrum + 0.01 && a.motion <= b.motion + 0.03 && a.coverage <= b.coverage + 0.005 &&
                          a.mfp >= b.mfp - 0.3;
      if (best_any < 0 || a.coupled > sum[sz(best_any)].coupled) best_any = static_cast<int>(m);
      if (guards && (best < 0 || a.coupled > sum[sz(best)].coupled)) best = static_cast<int>(m);
      choice_rows.push_back(std::format("{},{},{:.3f},{:.3f},{:.4f},{:.4f},{:.4f},{:.3f},{}", en, models[m].name, a.coupled, a.plain, a.spectrum, a.motion,
                                        a.coverage, a.mfp, guards ? 1 : 0));
    }
    choice_rows.push_back(std::format("{},v1,{:.3f},{:.3f},{:.4f},{:.4f},{:.4f},{:.3f},1", en, sum[0].coupled, sum[0].plain, sum[0].spectrum, sum[0].motion,
                                      sum[0].coverage, sum[0].mfp));
    const int chosen = best >= 0 ? best : best_any;
    if (chosen > 0) {
      fs::create_directories(c.data / "v2c");
      fs::copy_file(models[sz(chosen)].file, c.data / "v2c" / std::format("{}.nvfx", en), fs::copy_options::overwrite_existing);
      choice_rows.push_back(std::format("{},chosen:{},{:.3f},,,,,,{}", en, models[sz(chosen)].name, sum[sz(chosen)].coupled, best >= 0 ? "within_guards" : "best_without_guards"));
      std::println("i-val: {} chose {} (coupled {:.3f} dB against v1 {:.3f})", en, models[sz(chosen)].name, sum[sz(chosen)].coupled, sum[0].coupled);
    }
    done.insert(en);
    std::fflush(stdout);
  }
  std::string th = "effect,model,kind,case";
  for (const int h : kTrackHorizons) th += std::format(",f{}", h);
  merge_csv(c.results / "i_val_track.csv", th, track_rows, done);
  merge_csv(c.results / "i_val_stats.csv", "effect,model,setting,spectrum_l1,motion_ratio,coverage_l1,emission_l1,mean_frame_psnr", stat_rows, done);
  merge_csv(c.results / "i_val_choice.csv", "effect,model,coupled_8_30,plain_1_8_30_60,spectrum_l1,abs_log_motion,coverage_l1,mean_frame_psnr,within_guards",
            choice_rows, done);
}

// --- i-test -----------------------------------------------------------------------------------------------------------

namespace {

std::string iv(const metrics::Interval& v, int prec = 2) {
  return std::format("{:+.{}f} [{:+.{}f}, {:+.{}f}]", v.mean, prec, v.lo, prec, v.hi, prec);
}

}  // namespace

void step_test(const Ctx& c) {
  std::vector<std::string> track_rows, stat_rows, decision_rows, compare_rows;
  std::set<std::string> done;
  const auto ts = test_settings();
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e)) continue;
    const std::string en = ename(e);
    const fs::path v2c = c.data / "v2c" / std::format("{}.nvfx", en), v2p = c.data / "v2p" / std::format("{}.nvfx", en);
    if (!fs::exists(v2c)) throw std::runtime_error("no v2c for " + en + " (run i-val first)");
    std::vector<Named> models;
    models.push_back({"v1", load_or_throw(c.v1 / std::format("{}.nvfx", en)), c.v1 / std::format("{}.nvfx", en)});
    models.push_back({"v2c", load_or_throw(v2c), v2c});
    if (fs::exists(v2p)) models.push_back({"v2p", load_or_throw(v2p), v2p});
    const bool ex = e == sim::Effect::explosion;
    // the held-out cases: B's ten settings with new seeds and held-out forcing seeds; D's held-out runs (salt 2)
    std::vector<TrackCase> cases = forced_cases(e, ts, 960000, 970000, c.quick);
    if (e == sim::Effect::smoke) {
      for (TrackCase tc : handover_cases(ts, kTestSalt, 980000, 101, c.quick)) cases.push_back(tc);
    }
    for (TrackCase tc : plain_cases(e, kTestSalt, c.quick ? 2 : 16, c.quick ? 60 : (ex ? 89 : 240))) cases.push_back(tc);
    for (TrackCase tc : forced_cases(e, ts, 960000, 970000, c.quick, false)) cases.push_back(tc);
    const auto t0 = std::chrono::steady_clock::now();
    const TrackResults tr = run_tracking(c, models, cases);
    std::println("i-test: {} tracking in {:.0f} s", en, seconds_since(t0));
    const auto t1 = std::chrono::steady_clock::now();
    const auto st = run_endless(c, e, models, ts, 900000, 910000, 920000);  // study D's seeds
    std::println("i-test: {} endless runs in {:.0f} s", en, seconds_since(t1));
    for (std::size_t m = 0; m < models.size(); ++m) {
      for (std::size_t k = 0; k < cases.size(); ++k) track_rows.push_back(track_row(en, models[m].name, cases[k], tr[m][k], kDHorizons));
      for (std::size_t si = 0; si < ts.size(); ++si) stat_rows.push_back(stat_row(en, models[m].name, static_cast<int>(si), st[m][si]));
    }
    for (std::size_t si = 0; si < ts.size(); ++si) stat_rows.push_back(stat_row(en, "real_other_seed", static_cast<int>(si), st[models.size()][si]));
    // paired differences, model minus v1, at each horizon and statistic
    const auto paired_track = [&](std::size_t m, const std::string& kind, int frame) {
      std::vector<double> a, b;
      for (std::size_t k = 0; k < cases.size(); ++k) {
        if (cases[k].kind != kind || static_cast<int>(tr[m][k].size()) < frame) continue;
        a.push_back(tr[m][k][sz(frame - 1)]);
        b.push_back(tr[0][k][sz(frame - 1)]);
      }
      return metrics::paired_bootstrap(a, b);
    };
    // statistics, oriented so that positive is better for the model: spectrum, |log motion|, coverage lower is
    // better; mean-frame PSNR higher is better
    const auto paired_stat = [&](std::size_t m, int which) {
      std::vector<double> a, b;
      const auto val = [&](const metrics::StatDistance& d) {
        switch (which) {
          case 0: return -d.spectrum_l1;
          case 1: return -std::abs(std::log(std::max(1e-3, d.motion_ratio)));
          case 2: return -d.coverage_l1;
          default: return d.mean_frame_psnr;
        }
      };
      for (std::size_t si = 0; si < ts.size(); ++si) {
        a.push_back(val(st[m][si]));
        b.push_back(val(st[0][si]));
      }
      return metrics::paired_bootstrap(a, b);
    };
    const std::vector<std::string> stat_names = {"spectrum_l1(-)", "abs_log_motion(-)", "coverage_l1(-)", "mean_frame_psnr"};
    std::vector<std::string> kinds = {"forced"};
    if (e == sim::Effect::smoke) kinds.push_back("handover");
    kinds.push_back("unforced");  // a diagnostic, outside the rule
    for (std::size_t m = 1; m < models.size(); ++m) {
      const std::string mn = models[m].name;
      for (const std::string& kind : kinds) {
        for (const int h : kTrackHorizons) {
          const auto d = paired_track(m, kind, h);
          compare_rows.push_back(std::format("{},{},{},f{},{:.3f},{:.3f},{:.3f}", en, mn, kind, h, d.mean, d.lo, d.hi));
        }
      }
      int plain_frames = 0;
      for (const TrackCase& tc : cases) plain_frames = tc.kind == "plain" ? tc.frames : plain_frames;
      for (const int h : kDHorizons) {
        if (h > plain_frames) continue;
        const auto d = paired_track(m, "plain", h);
        compare_rows.push_back(std::format("{},{},plain,f{},{:.3f},{:.3f},{:.3f}", en, mn, h, d.mean, d.lo, d.hi));
      }
      for (int k = 0; k < 4; ++k) {
        const auto d = paired_stat(m, k);
        compare_rows.push_back(std::format("{},{},endless,{},{:.4f},{:.4f},{:.4f}", en, mn, stat_names[sz(k)], d.mean, d.lo, d.hi));
      }
    }
    // The rule (docs/COMPOSE.md §9), for v2c against v1:
    //   coupled: the forced tracking (smoke: the forced or the hand-over tracking) is better at 8 and at 30 frames,
    //            both intervals above zero; for smoke the other coupled test is not worse at 8 or 30 frames;
    //   plain:   plain tracking at 1, 8, 30 and 60 frames has no interval entirely below zero;
    //   endless: spectrum distance, |log motion ratio|, coverage distance and mean-frame PSNR have no interval
    //            entirely on the worse side.
    const auto better = [&](const std::string& kind) {
      return paired_track(1, kind, 8).lo > 0 && paired_track(1, kind, 30).lo > 0;
    };
    const auto not_worse = [&](const std::string& kind) { return paired_track(1, kind, 8).hi >= 0 && paired_track(1, kind, 30).hi >= 0; };
    bool coupled = better("forced");
    if (e == sim::Effect::smoke) coupled = (better("forced") && not_worse("handover")) || (better("handover") && not_worse("forced"));
    bool plain = true;
    for (const int h : kTrackHorizons) plain = plain && paired_track(1, "plain", h).hi >= 0;
    bool endless = true;
    for (int k = 0; k < 4; ++k) endless = endless && paired_stat(1, k).hi >= 0;
    const bool keep = coupled && plain && endless;
    std::string detail;
    for (const std::string& kind : kinds) detail += std::format("{} f8 {} f30 {}; ", kind, iv(paired_track(1, kind, 8)), iv(paired_track(1, kind, 30)));
    decision_rows.push_back(std::format("{},{},{},{},{}", en, coupled ? 1 : 0, plain ? 1 : 0, endless ? 1 : 0, keep ? "keep_v2c" : "keep_v1"));
    std::println("i-test: {}: coupled {} plain {} endless {} -> {} ({})", en, coupled, plain, endless, keep ? "v2c kept" : "v1 kept", detail);
    done.insert(en);
    std::fflush(stdout);
  }
  std::string th = "effect,model,kind,case";
  for (const int h : kDHorizons) th += std::format(",f{}", h);
  merge_csv(c.results / "i_test_track.csv", th, track_rows, done);
  merge_csv(c.results / "i_test_stats.csv", "effect,model,setting,spectrum_l1,motion_ratio,coverage_l1,emission_l1,mean_frame_psnr", stat_rows, done);
  merge_csv(c.results / "i_test_compare.csv", "effect,model,test,measure,mean_diff,lo,hi", compare_rows, done);
  merge_csv(c.results / "i_decisions.csv", "effect,coupled_better,plain_not_worse,endless_not_worse,decision", decision_rows, done);
}

}  // namespace nfx::study_i
