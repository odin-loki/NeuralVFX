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
#include "rt_handoff.hpp"

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
#include <ctime>
#include <format>
#include <fstream>
#include <iterator>
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
  std::string kind;  // forced, handover, plain (round 2 also: strong)
  int index = 0;
  sim::Params p;
  int warm = 100, frames = 60;
  std::uint64_t forcing = 0;
  bool strong = false;  // round 2: the couplings from scene_forcing (strong pushes) instead of random_forcing
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
  if (tc.forcing) t.spec = tc.strong ? rollout::scene_forcing(p.effect, tc.frames, tc.forcing, 0, 30) : rollout::random_forcing(p.effect, tc.frames, tc.forcing, 0, 15);
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

// Light and cover of a frame (metrics::stats' emission and coverage): mean max(0, rgb - alpha) and mean alpha.
std::array<double, 2> light_cover(std::span<const std::uint8_t> f) {
  double e = 0, a = 0;
  for (std::size_t i = 0; i < f.size(); i += 4) {
    const double al = f[i + 3] / 255.0;
    a += al;
    for (std::size_t k = 0; k < 3; ++k) e += std::max(0.0, f[i + k] / 255.0 - al) / 3.0;
  }
  const double n = static_cast<double>(f.size() / 4);
  return {e / n, a / n};
}

// The model from the true state (coarse and fine, as stored: through save and load) with the run's seed, given the same
// couplings as the truth, through the runtime's runner (as compose drives it); active PSNR per frame. With `light`, also
// each frame's light and cover (round 2: does the fire keep burning).
std::vector<double> track(const rollout::Model& M, const TrackCase& tc, const Truth& t, std::vector<std::array<double, 2>>* light = nullptr) {
  rollout::Model mt = M;
  mt.h.start_fine = kSize;
  mt.start_bits = 16;  // the true state as the start point (round 2: v2's fire stores its own start states at 6 bits)
  mt.start_dither = false;
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
    if (light) light->push_back(light_cover(rgba));
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

// `frames` frames of the runtime at controls with a seed, played as shipped (shards from start points). Round 2: with
// a hand-off (frames, fade) the first frames are drawn in the simulator's look.
Clip runtime_clip(nvfx_effect* e, std::span<const float> controls, std::uint64_t seed, int frames, std::array<int, 2> handoff = {0, 0}) {
  nvfx_instance* in = nullptr;
  if (nvfx_instance_create(e, kSize, &in) != NVFX_OK) throw std::runtime_error("instance");
  if (nvfx_instance_set_handoff(in, handoff[0], handoff[1]) != NVFX_OK) throw std::runtime_error("hand-off");
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
std::vector<TrackCase> handover_cases(const std::vector<Setting>& settings, std::uint64_t salt, std::uint64_t seed0, std::uint64_t mix, bool quick,
                                      std::uint64_t from0 = 400) {
  std::vector<TrackCase> v;
  const rollout::SimRecipe rx = recipe(sim::Effect::explosion, salt);
  for (std::size_t si = 0; si < settings.size(); ++si) {
    for (int k = 0; k < (quick ? 1 : 2); ++k) {
      TrackCase tc;
      tc.kind = "handover";
      tc.index = static_cast<int>(v.size());
      tc.handover = true;
      tc.from = rollout::recipe_run(rx, from0 + static_cast<std::uint64_t>(tc.index));
      tc.before = 24 + static_cast<int>(((static_cast<std::uint64_t>(tc.index) + mix) * 0x9E3779B97F4A7C15ULL >> 40) % 67);
      tc.p = at_setting(sim::Effect::smoke, settings[si], seed0 + 10 * si + static_cast<std::uint64_t>(k));
      tc.p.sim_res = kSize;
      tc.frames = 60;
      v.push_back(tc);
    }
  }
  return v;
}

std::vector<TrackCase> plain_cases(sim::Effect e, std::uint64_t salt, int n, int frames, std::uint64_t first = 0) {
  std::vector<TrackCase> v;
  const bool ex = e == sim::Effect::explosion;
  const rollout::SimRecipe r = recipe(e, salt);
  for (int i = 0; i < n; ++i) {
    TrackCase tc;
    tc.kind = "plain";
    tc.index = i;
    tc.p = rollout::recipe_run(r, first + static_cast<std::uint64_t>(i));
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


// ======================================================================================================================
// Round 2 (docs/COMPOSE.md §10): fire and smoke sturdier under couplings, and the explosion's first second.
// ======================================================================================================================

namespace {

// Strong (scene-like) forced runs per effect: salt 1, run indices 600 on, scene_forcing seeds 5000 on.
int strong_runs(const Ctx& c, sim::Effect e) { return c.quick ? 4 : (e == sim::Effect::fire ? 64 : 48); }
fs::path strong_path(const Ctx& c, sim::Effect e) { return c.data2 / "runs" / std::format("{}_strong.bin", ename(e)); }
bool round2_effect(sim::Effect e) { return e != sim::Effect::explosion; }

// Round 2's seeds, all new (docs/COMPOSE.md §10.4): validation, and the test.
struct Seeds2 {
  std::uint64_t salt, forced, forcing, strong, strong_forcing, handover, handover_from, handover_mix, plain_first, real, other, model;
};
Seeds2 seeds2(bool test) {
  if (test) return {kTestSalt, 1960000, 1970000, 1940000, 1950000, 1980000, 600, 211, 200, 1900000, 1910000, 1920000};
  return {kValSalt, 1430000, 1470000, 1440000, 1480000, 1460000, 500, 13, 100, 1540000, 0, 1550000};
}

constexpr int kSurvival = 120;  // frames of fire's strong cases: survival is scored over frames 31 to 120

// Validation or test cases of an effect: forced (round 1's kind of couplings), strong (scene_forcing), hand-over
// (smoke), plain; on the test also the forced cases without their couplings (a diagnostic).
std::vector<TrackCase> cases2(const Ctx& c, sim::Effect e, const std::vector<Setting>& settings, const Seeds2& sd, bool test) {
  std::vector<TrackCase> v = forced_cases(e, settings, sd.forced, sd.forcing, c.quick);
  for (TrackCase tc : forced_cases(e, settings, sd.strong, sd.strong_forcing, c.quick)) {
    tc.kind = "strong";
    tc.strong = true;
    if (e == sim::Effect::fire && !c.quick) tc.frames = kSurvival;
    v.push_back(tc);
  }
  if (e == sim::Effect::smoke) {
    for (TrackCase tc : handover_cases(settings, sd.salt, sd.handover, sd.handover_mix, c.quick, sd.handover_from)) v.push_back(tc);
  }
  for (TrackCase tc : plain_cases(e, sd.salt, c.quick ? 2 : 16, test && !c.quick ? 240 : 60, sd.plain_first)) v.push_back(tc);
  if (test) {
    for (TrackCase tc : forced_cases(e, settings, sd.forced, sd.forcing, c.quick, false)) v.push_back(tc);
  }
  return v;
}

using Light = std::vector<std::array<double, 2>>;  // light and cover per frame
struct Tracked {
  std::vector<double> psnr;
  Light light;  // strong cases
};
using Tracks = std::vector<std::vector<Tracked>>;  // [model][case]

void run_tracking2(const Ctx& c, const std::vector<Named>& models, const std::vector<TrackCase>& cases, Tracks& res, std::vector<Light>& truth) {
  res.assign(models.size(), std::vector<Tracked>(cases.size()));
  truth.assign(cases.size(), {});
  parallel(c, static_cast<int>(cases.size()), [&](int k) {
    const Truth t = make_truth(cases[sz(k)]);
    if (cases[sz(k)].strong) {
      for (const auto& f : t.frames) truth[sz(k)].push_back(light_cover(f));
    }
    for (std::size_t m = 0; m < models.size(); ++m) {
      Tracked& r = res[m][sz(k)];
      r.psnr = track(models[m].m, cases[sz(k)], t, cases[sz(k)].strong ? &r.light : nullptr);
    }
  });
}

// Does the fire keep burning like the simulator under strong pushes (docs/COMPOSE.md §10.3): over frames 31 to 120,
// the mean of |ln| of the model's light over the truth's and of its cover over the truth's (each + 0.001); lower is
// better. And the share of the truth's light the model keeps over the last second (frames 91 to 120).
double survival_score(const Light& model, const Light& truth) {
  double s = 0;
  int n = 0;
  for (std::size_t i = 30; i < model.size() && i < truth.size(); ++i) {
    s += 0.5 * (std::abs(std::log((model[i][0] + 1e-3) / (truth[i][0] + 1e-3))) + std::abs(std::log((model[i][1] + 1e-3) / (truth[i][1] + 1e-3))));
    ++n;
  }
  return n > 0 ? s / n : 0.0;
}
double burning_share(const Light& model, const Light& truth) {
  double a = 0, b = 0;
  for (std::size_t i = model.size() >= 30 ? model.size() - 30 : 0; i < model.size() && i < truth.size(); ++i) {
    a += model[i][0];
    b += truth[i][0];
  }
  return b > 0 ? a / b : 1.0;
}

// The baseline (v2's file) and the candidates of round 2. Fire's candidates are written in v2's form: v2's fire (its
// start states at 6 bits) with the candidate's stepper, so only the stepper differs from the baseline.
std::vector<Named> models2(const Ctx& c, sim::Effect e, const std::vector<fs::path>& extra) {
  const std::string en = ename(e);
  const fs::path base = c.v2 / std::format("{}.nvfx", en);
  std::vector<Named> v;
  v.push_back({"v2", load_or_throw(base), base});
  if (e == sim::Effect::fire) {  // writing v2's fire again gives the same bytes (so the swap changes the stepper only)
    std::ifstream in(base, std::ios::binary);
    const std::string file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::ostringstream os;
    if (auto w = rollout::save_model(os, v[0].m); !w) throw std::runtime_error(w.error());
    if (os.str() != file) throw std::runtime_error("v2's fire does not round-trip through load and save");
  }
  for (const fs::path& f : extra) {
    std::string name = f.stem().string();
    if (name.starts_with(en + "_")) name = name.substr(en.size() + 1);
    rollout::Model m = load_or_throw(f);
    fs::path file = f;
    if (e == sim::Effect::fire) {
      rollout::Model w = v[0].m;
      if (w.step_w.size() != m.step_w.size()) throw std::runtime_error("stepper shapes differ: " + f.string());
      w.step_w = m.step_w;
      file = c.data2 / "v2form" / f.filename();
      fs::create_directories(file.parent_path());
      if (auto r = rollout::save_model(file, w); !r) throw std::runtime_error(r.error());
      m = load_or_throw(file);
    }
    v.push_back({name, std::move(m), file});
  }
  return v;
}

std::vector<fs::path> candidate_files(const Ctx& c, sim::Effect e) {
  std::vector<fs::path> files;
  if (fs::exists(c.data2 / "cand")) {
    for (const auto& f : fs::directory_iterator(c.data2 / "cand")) {
      if (f.path().filename().string().starts_with(ename(e) + "_") && f.path().extension() == ".nvfx") files.push_back(f.path());
    }
  }
  std::ranges::sort(files);
  return files;
}

// The rule of round 2 (docs/COMPOSE.md §10.5) for model m against the baseline (model 0), on validation or the test.
struct Verdict {
  bool coupled = false, plain = false, endless = false, survival = true;
  double score = 0;  // the coupled score: mean PSNR at 8 and 30 frames over the coupled cases (the choice)
  std::vector<std::string> rows;  // effect,model,test,measure,mean_diff,lo,hi
  std::string detail;
  bool keep() const { return coupled && plain && endless && survival; }
};

Verdict judge(sim::Effect e, const std::vector<Named>& models, std::size_t m, const std::vector<TrackCase>& cases, const Tracks& tr,
              const std::vector<Light>& truth, const std::vector<std::vector<metrics::StatDistance>>& st) {
  const std::string en = ename(e), mn = models[m].name;
  const bool smoke = e == sim::Effect::smoke, fire = e == sim::Effect::fire;
  Verdict v;
  using Kinds = std::vector<std::string>;
  const Kinds forced{"forced", "strong"}, hand{"handover"};
  const auto paired = [&](const Kinds& kinds, int frame) {
    std::vector<double> a, b;
    for (std::size_t k = 0; k < cases.size(); ++k) {
      if (std::ranges::find(kinds, cases[k].kind) == kinds.end() || static_cast<int>(tr[m][k].psnr.size()) < frame) continue;
      a.push_back(tr[m][k].psnr[sz(frame - 1)]);
      b.push_back(tr[0][k].psnr[sz(frame - 1)]);
    }
    return a.empty() ? metrics::Interval{} : metrics::paired_bootstrap(a, b);
  };
  const auto has = [&](const std::string& kind) { return std::ranges::any_of(cases, [&](const TrackCase& tc) { return tc.kind == kind; }); };
  const auto row = [&](const std::string& test, const std::string& measure, const metrics::Interval& d) {
    v.rows.push_back(std::format("{},{},{},{},{:.4f},{:.4f},{:.4f}", en, mn, test, measure, d.mean, d.lo, d.hi));
  };
  // 1. coupled tracking
  const auto better = [&](const Kinds& k) { return paired(k, 8).lo > 0 && paired(k, 30).lo > 0; };
  const auto not_worse = [&](const Kinds& k) { return paired(k, 8).hi >= 0 && paired(k, 30).hi >= 0; };
  v.coupled = smoke ? (better(forced) && not_worse(hand)) || (better(hand) && not_worse(forced)) : better(forced);
  std::vector<std::pair<std::string, Kinds>> tests{{"forced+strong", forced}, {"forced", {"forced"}}, {"strong", {"strong"}}};
  if (smoke) tests.push_back({"handover", hand});
  if (has("unforced")) tests.push_back({"unforced", {"unforced"}});
  for (const auto& [name, kinds] : tests) {
    for (const int h : {1, 8, 30, 60, 120}) {
      const auto d = paired(kinds, h);
      if (d.mean != 0 || d.lo != 0 || d.hi != 0) row(name, std::format("f{}", h), d);
    }
  }
  for (const auto& [name, kinds] : tests) {
    if (name == "forced+strong" || name == "handover") v.detail += std::format("{} f8 {} f30 {}; ", name, iv(paired(kinds, 8)), iv(paired(kinds, 30)));
  }
  // 2. plain tracking
  v.plain = true;
  for (const int h : kTrackHorizons) v.plain = v.plain && paired({"plain"}, h).hi >= 0;
  for (const int h : kDHorizons) {
    const auto d = paired({"plain"}, h);
    if (d.mean != 0 || d.lo != 0 || d.hi != 0) row("plain", std::format("f{}", h), d);
  }
  v.detail += std::format("plain f1 {} f8 {} f30 {} f60 {}; ", iv(paired({"plain"}, 1)), iv(paired({"plain"}, 8)), iv(paired({"plain"}, 30)), iv(paired({"plain"}, 60)));
  // 3. endless statistics, oriented so that positive is better
  const std::vector<std::string> names = {"spectrum_l1(-)", "abs_log_motion(-)", "coverage_l1(-)", "mean_frame_psnr"};
  v.endless = true;
  for (int k = 0; k < 4; ++k) {
    std::vector<double> a, b;
    const auto val = [&](const metrics::StatDistance& d) {
      switch (k) {
        case 0: return -d.spectrum_l1;
        case 1: return -std::abs(std::log(std::max(1e-3, d.motion_ratio)));
        case 2: return -d.coverage_l1;
        default: return d.mean_frame_psnr;
      }
    };
    for (std::size_t si = 0; si < st[m].size(); ++si) {
      a.push_back(val(st[m][si]));
      b.push_back(val(st[0][si]));
    }
    const auto d = metrics::paired_bootstrap(a, b);
    row("endless", names[sz(k)], d);
    v.endless = v.endless && d.hi >= 0;
    v.detail += std::format("{} {}; ", names[sz(k)], iv(d, 4));
  }
  // 4. fire: survival under strong pushes (oriented: minus the score)
  if (fire) {
    std::vector<double> a, b, ba, bb;
    for (std::size_t k = 0; k < cases.size(); ++k) {
      if (!cases[k].strong || tr[m][k].light.size() < 60) continue;
      a.push_back(-survival_score(tr[m][k].light, truth[k]));
      b.push_back(-survival_score(tr[0][k].light, truth[k]));
      ba.push_back(burning_share(tr[m][k].light, truth[k]));
      bb.push_back(burning_share(tr[0][k].light, truth[k]));
    }
    if (!a.empty()) {
      const auto d = metrics::paired_bootstrap(a, b);
      row("survival", "score(-)", d);
      row("survival", "burning_share", metrics::paired_bootstrap(ba, bb));
      v.survival = d.hi >= 0;
      v.detail += std::format("survival {}", iv(d, 3));
    }
  }
  // the coupled score
  std::vector<double> sc;
  for (std::size_t k = 0; k < cases.size(); ++k) {
    const std::string& kind = cases[k].kind;
    if (kind == "forced" || kind == "strong" || (smoke && kind == "handover")) sc.push_back((tr[m][k].psnr[7] + tr[m][k].psnr[29]) / 2.0);
  }
  v.score = mean(sc);
  return v;
}

std::string track_header() {
  std::string th = "effect,model,kind,case";
  for (const int h : kDHorizons) th += std::format(",f{}", h);
  return th;
}

void track_rows2(const std::string& en, const std::vector<Named>& models, const std::vector<TrackCase>& cases, const Tracks& tr, const std::vector<Light>& truth,
                 std::vector<std::string>& rows, std::vector<std::string>& surv) {
  for (std::size_t m = 0; m < models.size(); ++m) {
    for (std::size_t k = 0; k < cases.size(); ++k) {
      rows.push_back(track_row(en, models[m].name, cases[k], tr[m][k].psnr, kDHorizons));
      if (!cases[k].strong || tr[m][k].light.empty()) continue;
      const Light& l = tr[m][k].light;
      const std::size_t last = l.size() - 1;
      surv.push_back(std::format("{},{},{},{:.4f},{:.4f},{:.5f},{:.5f},{:.5f},{:.5f}", en, models[m].name, cases[k].index, survival_score(l, truth[k]),
                                 burning_share(l, truth[k]), l[last][0], truth[k][last][0], l[last][1], truth[k][last][1]));
    }
  }
}

constexpr const char* kSurvHeader = "effect,model,case,survival_score,burning_share,light_last,light_last_truth,cover_last,cover_last_truth";

}  // namespace

// --- i2-data -----------------------------------------------------------------------------------------------------------

void step_data2(const Ctx& c) {
  std::vector<std::string> rows;
  std::set<std::string> done;
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e) || !round2_effect(e)) continue;
    const std::string en = ename(e);
    const int n = strong_runs(c, e), frames = c.quick ? 60 : 240;
    const rollout::SimRecipe r = recipe(e, kTrainSalt);
    const auto t0 = std::chrono::steady_clock::now();
    if (fs::exists(strong_path(c, e))) {
      std::println("i2-data: {} strong runs already recorded", en);
      continue;
    }
    std::vector<rollout::Run> runs(sz(n));
    parallel(c, n, [&](int i) {
      const sim::Params p = rollout::recipe_run(r, 600 + static_cast<std::uint64_t>(i));
      const rollout::ForcingSpec spec = rollout::scene_forcing(e, frames, 5000 + static_cast<std::uint64_t>(i), 15, frames - 60);
      runs[sz(i)] = rollout::record_forced_run(p, spec, frames, kRes);
    });
    save_runs(strong_path(c, e), runs);
    const double secs = seconds_since(t0);
    std::size_t on = 0, out = 0, cells = 0;
    std::array<std::size_t, rollout::kPhys> by{};
    const rollout::Model v1 = load_or_throw(c.v1 / std::format("{}.nvfx", en));
    for (const auto& run : runs) {
      on += static_cast<std::size_t>(std::ranges::count_if(run.forcing_at, [](int k) { return k >= 0; }));
      for (std::size_t j = 0; j < run.coarse.size(); j += rollout::kPhys) {
        for (int k = 0; k < rollout::kPhys; ++k) {
          const bool o = run.coarse[j + sz(k)] < v1.lo[sz(k)] || run.coarse[j + sz(k)] > v1.hi[sz(k)];
          out += o;
          by[sz(k)] += o;
        }
        ++cells;
      }
    }
    std::println("i2-data: {} cell values outside v1's range by channel: u {:.2e} v {:.2e} heat {:.2e} soot {:.2e} (range u {:.3f} v {:.3f} heat {:.3f} soot {:.3f})", en,
                 static_cast<double>(by[0]) / static_cast<double>(cells), static_cast<double>(by[1]) / static_cast<double>(cells),
                 static_cast<double>(by[2]) / static_cast<double>(cells), static_cast<double>(by[3]) / static_cast<double>(cells), v1.hi[0], v1.hi[1], v1.hi[2], v1.hi[3]);
    const double share = static_cast<double>(on) / std::max(1.0, static_cast<double>(n) * frames);
    rows.push_back(std::format("{},strong,{},{},{:.3f},{:.2e},{:.0f}", en, n, frames, share, static_cast<double>(out) / static_cast<double>(std::max<std::size_t>(1, cells)), secs));
    std::println("i2-data: {} strong {} x {} ({:.0f}% of frames forced, {:.2e} of cell values outside v1's range, {:.0f} s)", en, n, frames, 100.0 * share,
                 static_cast<double>(out) / static_cast<double>(std::max<std::size_t>(1, cells)), secs);
    done.insert(en);
    std::fflush(stdout);
  }
  if (!rows.empty()) merge_csv(c.results / "i2_data.csv", "effect,kind,runs,frames,forced_share,outside_v1_range,seconds", rows, done);
}

// --- i2-probe ----------------------------------------------------------------------------------------------------------

void step_probe2(const Ctx& c) {
  std::vector<std::string> rows;
  std::set<std::string> done;
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e) || !round2_effect(e)) continue;
    const std::string en = ename(e);
    // (1) v1 under strong and under random couplings, on training-salt runs (indices 900 on): tracking and survival
    std::vector<Named> models;
    models.push_back({"v1", load_or_throw(c.v1 / std::format("{}.nvfx", en)), c.v1 / std::format("{}.nvfx", en)});
    if (fs::exists(c.data / "v2c" / std::format("{}.nvfx", en))) {
      models.push_back({"round1_v2c", load_or_throw(c.data / "v2c" / std::format("{}.nvfx", en)), c.data / "v2c" / std::format("{}.nvfx", en)});
    }
    std::vector<TrackCase> cases;
    const rollout::SimRecipe r = recipe(e, kTrainSalt);
    const int n = c.quick ? 2 : 8;
    for (int i = 0; i < 2 * n; ++i) {
      TrackCase tc;
      tc.kind = i < n ? "strong" : "forced";
      tc.strong = i < n;
      tc.index = i % n;
      tc.p = rollout::recipe_run(r, 900 + static_cast<std::uint64_t>(i % n));
      tc.frames = c.quick ? 60 : kSurvival;
      tc.forcing = 9000 + static_cast<std::uint64_t>(i % n);
      cases.push_back(tc);
    }
    Tracks tr;
    std::vector<Light> truth;
    run_tracking2(c, models, cases, tr, truth);
    for (std::size_t m = 0; m < models.size(); ++m) {
      for (std::size_t k = 0; k < cases.size(); ++k) {
        const auto& t = tr[m][k];
        std::string extra = ",,";
        if (cases[k].strong) extra = std::format(",{:.4f},{:.4f}", survival_score(t.light, truth[k]), burning_share(t.light, truth[k]));
        rows.push_back(std::format("{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f}{}", en, models[m].name, cases[k].kind, cases[k].index, t.psnr[0], t.psnr[7], t.psnr[29],
                                   t.psnr[59], extra));
      }
    }
    for (std::size_t m = 0; m < models.size(); ++m) {
      std::vector<double> s, b, p8, p30;
      for (std::size_t k = 0; k < cases.size(); ++k) {
        if (!cases[k].strong) continue;
        s.push_back(survival_score(tr[m][k].light, truth[k]));
        b.push_back(burning_share(tr[m][k].light, truth[k]));
        p8.push_back(tr[m][k].psnr[7]);
        p30.push_back(tr[m][k].psnr[29]);
      }
      std::println("i2-probe: {} {} strong: survival score {:.3f}, burning share {:.2f}, PSNR at 8 frames {:.2f}, 30 frames {:.2f}", en, models[m].name, mean(s), mean(b),
                   mean(p8), mean(p30));
    }
    // (2) the anchor's scale, in the loss's units: v1's first step from true states against the truth, and the first
    // steps of round 1's candidate against v1's
    if (models.size() > 1) {
      const std::vector<rollout::Run> plain = load_runs(runs_path(c, e, "plain"));
      std::mt19937_64 rng(5);
      std::vector<double> data, dist;
      for (int w = 0; w < 64; ++w) {
        const rollout::Run& run = plain[rng() % plain.size()];
        const int first = static_cast<int>(rng() % static_cast<std::uint64_t>(run.frames - 2));
        const double l1 = rollout::window_loss(models[0].m, run, first, 1, 0, 0.f, 0.f, 1, nullptr);
        const double l2 = rollout::window_loss(models[1].m, run, first, 1, 0, 0.f, 0.f, 1, nullptr);
        const double l3 = rollout::window_loss(models[1].m, run, first, 1, 0, 0.f, 0.f, 1, nullptr, 0.f, &models[0].m, 1.f);
        data.push_back(l1);
        dist.push_back(l3 - l2);
      }
      std::println("i2-probe: {} first step from true states: v1's squared error {:.5f}; round 1's candidate's distance from v1 {:.6f} (loss units)", en, mean(data),
                   mean(dist));
      rows.push_back(std::format("{},anchor_scale,v1_first_step_error,,{:.6f},,,,,", en, mean(data)));
      rows.push_back(std::format("{},anchor_scale,round1_distance_from_v1,,{:.6f},,,,,", en, mean(dist)));
    }
    done.insert(en);
    std::fflush(stdout);
  }
  merge_csv(c.results / "i2_probe.csv", "effect,model,kind,case,f1,f8,f30,f60,survival_score,burning_share", rows, done);
}

// --- i2-train ----------------------------------------------------------------------------------------------------------

void step_train2(const Ctx& c) {
  if (!train::cpu_supported()) throw std::runtime_error("training needs AVX2 + FMA");
  if (c.tag.empty()) throw std::runtime_error("i2-train needs --tag");
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e) || !round2_effect(e)) continue;
    const std::string en = ename(e);
    int n_plain = 0;
    std::vector<rollout::Run> runs = load_training_runs(c, e, n_plain);  // round 1's plain, forced and hand-over runs
    std::size_t n_strong = 0;
    if (c.strong) {
      for (auto& r : load_runs(strong_path(c, e))) {
        runs.push_back(std::move(r));
        ++n_strong;
      }
    }
    rollout::Model m = load_or_throw(c.v1 / std::format("{}.nvfx", en));
    const rollout::Model anchor = m;
    const std::vector<float> keep_render = m.render_w;
    rollout::StepperOptions so = finetune_options(c, e, n_plain);
    so.anchor_model = &anchor;
    so.anchor = c.anchor;
    so.aim_couplings = c.aim;
    const fs::path dir = c.data2 / "cand";
    fs::create_directories(dir);
    fs::create_directories(c.data2 / "logs");
    std::ofstream curve(c.data2 / "logs" / std::format("train_{}_{}.csv", en, c.tag));
    so.progress = [&](int it, int unroll, double loss) {
      std::println("  {} {} {:5d} unroll {:2d} loss {:.5f}", en, c.tag, it, unroll, loss);
      std::fflush(stdout);
      curve << it << "," << loss << "\n";
      curve.flush();
    };
    so.checkpoint_every = c.checkpoint;
    so.checkpoint = [&](int done) {
      const fs::path out = dir / std::format("{}_{}_it{}.nvfx", en, c.tag, done);
      if (auto w = rollout::save_model(out, m); !w) throw std::runtime_error(w.error());
      std::println("  saved {}", out.string());
      std::fflush(stdout);
    };
    std::println("i2-train: {} {}: {} plain + {} coupled runs ({} strong), share {}, lr {}, anchor {}, aim {}, {} iterations, batch {}", en, c.tag, n_plain,
                 runs.size() - sz(n_plain), n_strong, c.share, c.lr, c.anchor, c.aim, c.iters, c.batch);
    const rollout::StepperResult sr = rollout::train_stepper(m, runs, so);
    if (m.render_w != keep_render) throw std::logic_error("only the stepper may change");
    std::println("i2-train: {} {} loss {:.5f} in {:.0f} s", en, c.tag, sr.final_loss, sr.seconds);
    const fs::path log = c.results / "i2_train.csv";
    std::vector<std::string> keep;
    if (std::ifstream in(log); in) {
      std::string line;
      std::getline(in, line);
      while (std::getline(in, line)) {
        if (!line.empty() && line.rfind(std::format("{},{},", en, c.tag), 0) != 0) keep.push_back(line);
      }
    }
    keep.push_back(std::format("{},{},{},{},{},{},{},{},{},{},{:.5f},{:.0f}", en, c.tag, c.share, c.lr, c.anchor, c.aim ? 1 : 0, c.strong ? 1 : 0, c.iters, c.batch,
                               c.threads, sr.final_loss, sr.seconds));
    std::ofstream o(log);
    o << "effect,candidate,share,lr,anchor,aim,strong,iterations,batch,threads,final_loss,seconds\n";
    for (const auto& r : keep) o << r << "\n";
  }
}

// --- i2-val ------------------------------------------------------------------------------------------------------------

void step_val2(const Ctx& c) {
  std::vector<std::string> track_rows, stat_rows, surv_rows, choice_rows, compare_rows;
  std::set<std::string> done;
  const auto vs = validation_settings();
  const Seeds2 sd = seeds2(false);
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e) || !round2_effect(e)) continue;
    const std::string en = ename(e);
    const std::vector<Named> models = models2(c, e, candidate_files(c, e));
    std::println("i2-val: {}: {} models", en, models.size());
    const std::vector<TrackCase> cases = cases2(c, e, vs, sd, false);
    const auto t0 = std::chrono::steady_clock::now();
    Tracks tr;
    std::vector<Light> truth;
    run_tracking2(c, models, cases, tr, truth);
    std::println("i2-val: {} tracking ({} cases) in {:.0f} s", en, cases.size(), seconds_since(t0));
    const auto t1 = std::chrono::steady_clock::now();
    const auto st = run_endless(c, e, models, vs, sd.real, 0, sd.model);
    std::println("i2-val: {} endless runs in {:.0f} s", en, seconds_since(t1));
    track_rows2(en, models, cases, tr, truth, track_rows, surv_rows);
    for (std::size_t m = 0; m < models.size(); ++m) {
      for (std::size_t si = 0; si < vs.size(); ++si) stat_rows.push_back(stat_row(en, models[m].name, static_cast<int>(si), st[m][si]));
    }
    // The choice (docs/COMPOSE.md §10.5): among the candidates that meet every part of the rule on validation, the best
    // coupled score; none: the effect stops here (no test).
    int best = -1;
    double best_score = 0;
    for (std::size_t m = 1; m < models.size(); ++m) {
      const Verdict v = judge(e, models, m, cases, tr, truth, st);
      for (const auto& r : v.rows) compare_rows.push_back(r);
      choice_rows.push_back(std::format("{},{},{:.3f},{},{},{},{},{}", en, models[m].name, v.score, v.coupled ? 1 : 0, v.plain ? 1 : 0, v.endless ? 1 : 0,
                                        v.survival ? 1 : 0, v.keep() ? 1 : 0));
      std::println("i2-val: {} {}: coupled {} plain {} endless {} survival {} score {:.3f} ({})", en, models[m].name, v.coupled, v.plain, v.endless, v.survival,
                   v.score, v.detail);
      if (v.keep() && (best < 0 || v.score > best_score)) {
        best = static_cast<int>(m);
        best_score = v.score;
      }
    }
    {
      std::vector<double> sc;
      for (std::size_t k = 0; k < cases.size(); ++k) {
        const std::string& kind = cases[k].kind;
        if (kind == "forced" || kind == "strong" || (e == sim::Effect::smoke && kind == "handover")) sc.push_back((tr[0][k].psnr[7] + tr[0][k].psnr[29]) / 2.0);
      }
      choice_rows.push_back(std::format("{},v2,{:.3f},,,,,", en, mean(sc)));
    }
    const fs::path chosen = c.data2 / "chosen" / std::format("{}.nvfx", en);
    fs::create_directories(chosen.parent_path());
    fs::remove(chosen);
    if (best > 0) {
      fs::copy_file(models[sz(best)].file, chosen, fs::copy_options::overwrite_existing);
      choice_rows.push_back(std::format("{},chosen:{},{:.3f},,,,,", en, models[sz(best)].name, best_score));
      std::println("i2-val: {} chose {} (coupled score {:.3f})", en, models[sz(best)].name, best_score);
    } else {
      choice_rows.push_back(std::format("{},chosen:none,,,,,,", en));
      std::println("i2-val: {}: no candidate meets the rule on validation; the effect stops here", en);
    }
    done.insert(en);
    std::fflush(stdout);
  }
  merge_csv(c.results / "i2_val_track.csv", track_header(), track_rows, done);
  merge_csv(c.results / "i2_val_stats.csv", "effect,model,setting,spectrum_l1,motion_ratio,coverage_l1,emission_l1,mean_frame_psnr", stat_rows, done);
  merge_csv(c.results / "i2_val_survival.csv", kSurvHeader, surv_rows, done);
  merge_csv(c.results / "i2_val_compare.csv", "effect,model,test,measure,mean_diff,lo,hi", compare_rows, done);
  merge_csv(c.results / "i2_val_choice.csv", "effect,model,coupled_score,coupled,plain,endless,survival,meets_rule", choice_rows, done);
}

// --- i2-test -----------------------------------------------------------------------------------------------------------

void step_test2(const Ctx& c) {
  std::vector<std::string> track_rows, stat_rows, surv_rows, compare_rows, decision_rows;
  std::set<std::string> done;
  const auto ts = test_settings();
  const Seeds2 sd = seeds2(true);
  for (const auto e : sim::kEffects) {
    if (!wanted(c, e) || !round2_effect(e)) continue;
    const std::string en = ename(e);
    const fs::path chosen = c.data2 / "chosen" / std::format("{}.nvfx", en);
    done.insert(en);
    if (!fs::exists(chosen)) {
      decision_rows.push_back(std::format("{},none,,,,,keep_v2 (stopped at validation)", en));
      std::println("i2-test: {}: nothing passed validation; no test", en);
      continue;
    }
    std::vector<Named> models = models2(c, e, {});
    {
      Named n{"chosen", load_or_throw(chosen), chosen};
      models.push_back(std::move(n));
    }
    const std::vector<TrackCase> cases = cases2(c, e, ts, sd, true);
    const auto t0 = std::chrono::steady_clock::now();
    Tracks tr;
    std::vector<Light> truth;
    run_tracking2(c, models, cases, tr, truth);
    std::println("i2-test: {} tracking ({} cases) in {:.0f} s", en, cases.size(), seconds_since(t0));
    const auto t1 = std::chrono::steady_clock::now();
    const auto st = run_endless(c, e, models, ts, sd.real, sd.other, sd.model);
    std::println("i2-test: {} endless runs in {:.0f} s", en, seconds_since(t1));
    track_rows2(en, models, cases, tr, truth, track_rows, surv_rows);
    for (std::size_t m = 0; m < models.size(); ++m) {
      for (std::size_t si = 0; si < ts.size(); ++si) stat_rows.push_back(stat_row(en, models[m].name, static_cast<int>(si), st[m][si]));
    }
    for (std::size_t si = 0; si < ts.size(); ++si) stat_rows.push_back(stat_row(en, "real_other_seed", static_cast<int>(si), st[models.size()][si]));
    const Verdict v = judge(e, models, 1, cases, tr, truth, st);
    for (const auto& r : v.rows) compare_rows.push_back(r);
    decision_rows.push_back(std::format("{},{},{},{},{},{},{}", en, v.coupled ? 1 : 0, v.plain ? 1 : 0, v.endless ? 1 : 0, e == sim::Effect::fire ? (v.survival ? "1" : "0") : "",
                                        v.keep() ? 1 : 0, v.keep() ? "keep_round2" : "keep_v2"));
    std::println("i2-test: {}: coupled {} plain {} endless {} survival {} -> {} ({})", en, v.coupled, v.plain, v.endless, v.survival, v.keep() ? "round 2 kept" : "v2 kept",
                 v.detail);
    std::fflush(stdout);
  }
  merge_csv(c.results / "i2_test_track.csv", track_header(), track_rows, done);
  merge_csv(c.results / "i2_test_stats.csv", "effect,model,setting,spectrum_l1,motion_ratio,coverage_l1,emission_l1,mean_frame_psnr", stat_rows, done);
  merge_csv(c.results / "i2_test_survival.csv", kSurvHeader, surv_rows, done);
  merge_csv(c.results / "i2_test_compare.csv", "effect,model,test,measure,mean_diff,lo,hi", compare_rows, done);
  merge_csv(c.results / "i2_decisions.csv", "effect,coupled_better,plain_not_worse,endless_not_worse,survival_not_worse,kept,decision", decision_rows, done);
}

// --- i2-handoff: the explosion's first second ----------------------------------------------------------------------------

namespace {

struct Handoff {
  int frames = 0, fade = 0;
  std::string name() const { return std::format("N{}_M{}", frames, fade); }
};

std::vector<Handoff> handoff_grid() {
  std::vector<Handoff> v{{0, 0}};
  for (const int n : {4, 8, 15, 22, 30, 45}) {
    for (const int f : {0, 8, 15, 30}) v.push_back({n, f});
  }
  return v;
}

constexpr int kExplosionFrames = 89;

// One explosion tracked from its true state through the runtime's runner: the learned frames and the simulator's look
// of the same fields, each config's blend scored against the truth. Returns [config] -> PSNR per frame (frames 1 to 89).
std::vector<std::vector<double>> track_handoff(const rollout::Model& M, const sim::Params& p, const std::vector<Handoff>& cfg, std::vector<std::uint8_t>* first_bytes,
                                               std::string* saved) {
  sim::Fluid truth(p);
  truth.step_frame();  // warm 1 (REPORT §6.4, explosions)
  const sim::State st = truth.state();
  rollout::Model mt = M;
  mt.h.start_fine = kSize;
  mt.start_bits = 16;
  mt.start_dither = false;
  rollout::StartPoint sp;
  sp.controls = {p.intensity, p.wind, p.turbulence};
  sp.seed = p.seed;
  sp.time = st.time;
  sp.coarse.resize(sz(kRes) * kRes * rollout::kPhys);
  rollout::coarse_from_sim(st, kRes, p.fps, sp.coarse);
  sp.fine_t = st.temp;
  sp.fine_d = st.soot;
  mt.starts = {sp};
  std::stringstream bytes;
  if (auto w = rollout::save_model(bytes, mt); !w) throw std::runtime_error(w.error());
  if (saved) *saved = bytes.str();
  auto loaded = rollout::load_model(bytes);
  if (!loaded) throw std::runtime_error(loaded.error());
  rt::RolloutEffect re;
  re.m = std::move(*loaded);
  auto run = compose::make_runner(re, kSize, compose::best_isa());
  run->begin(0, sp.seed);
  const std::size_t row = sz(kSize) * 4, n = row * sz(kSize);
  std::vector<std::uint8_t> ref(n), learned(n), look(n), mix(n);
  std::vector<std::vector<double>> out(cfg.size());
  for (int i = 0; i < kExplosionFrames; ++i) {
    truth.step_frame();
    truth.render(ref);
    run->step(sp.controls, sp.seed);
    run->render(rt::FrameInput{}, learned.data(), row);
    rt::draw_sim_look(rt::SimLook::explosion, run->fine_heat(), run->fine_soot(), kSize, look.data(), row);
    for (std::size_t k = 0; k < cfg.size(); ++k) {
      mix = learned;
      rt::blend_handoff(mix.data(), row, look.data(), row, kSize, rt::handoff_weight(i + 1, cfg[k].frames, cfg[k].fade));
      out[k].push_back(apsnr(ref, mix));
      if (first_bytes && k + 1 == cfg.size()) first_bytes->insert(first_bytes->end(), mix.begin(), mix.end());
    }
  }
  return out;
}

double mean_range(const std::vector<double>& v, int a, int b) {  // frames a..b (1-based)
  double s = 0;
  int n = 0;
  for (int f = a; f <= b && f <= static_cast<int>(v.size()); ++f, ++n) s += v[sz(f - 1)];
  return n > 0 ? s / n : 0.0;
}

double thread_seconds() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

}  // namespace

void step_handoff(const Ctx& c) {
  const bool test = c.split == "test";
  if (!test && c.split != "val") throw std::runtime_error("--split val or test");
  const sim::Effect e = sim::Effect::explosion;
  const std::string en = ename(e);
  const fs::path file = c.v2 / "explosion.nvfx";
  const rollout::Model M = load_or_throw(file);
  const auto settings = test ? test_settings() : validation_settings();
  // seeds (docs/COMPOSE.md §10.4): tracked runs, real runs, the floor's other real runs, the runtime's instances
  const std::uint64_t track0 = test ? 1720000 : 1620000, real0 = test ? 1740000 : 1640000, other0 = test ? 1745000 : 0, model0 = test ? 1750000 : 1650000;
  std::vector<Handoff> cfg = handoff_grid();
  const fs::path choice_file = c.data2 / "handoff_choice.txt";
  if (test) {
    std::ifstream in(choice_file);
    Handoff h;
    if (!(in >> h.frames >> h.fade)) throw std::runtime_error("no hand-off was chosen on validation (" + choice_file.string() + ")");
    cfg = {{0, 0}, h};
  }
  // tracking: two seeds per setting
  const int per = c.quick ? 1 : 2;
  const int ncase = static_cast<int>(settings.size()) * per;
  std::vector<std::vector<std::vector<double>>> tr(sz(ncase));
  std::vector<std::uint8_t> blended0;
  std::string saved0;
  const auto t0 = std::chrono::steady_clock::now();
  parallel(c, ncase, [&](int k) {
    const sim::Params p = at_setting(e, settings[sz(k / per)], track0 + 10 * static_cast<std::uint64_t>(k / per) + static_cast<std::uint64_t>(k % per));
    tr[sz(k)] = track_handoff(M, p, cfg, k == 0 && test ? &blended0 : nullptr, k == 0 && test ? &saved0 : nullptr);
  });
  std::println("i2-handoff {}: tracking ({} cases, {} configs) in {:.0f} s", c.split, ncase, cfg.size(), seconds_since(t0));
  // on the test, the runtime's own path (nvfx_instance_set_handoff) must give the bytes scored here
  if (test) {
    nvfx_effect* fx = nullptr;
    if (nvfx_effect_load_memory(saved0.data(), saved0.size(), &fx) != NVFX_OK) throw std::runtime_error("load");
    nvfx_instance* in = nullptr;
    if (nvfx_instance_create(fx, kSize, &in) != NVFX_OK) throw std::runtime_error("instance");
    const sim::Params p = at_setting(e, settings[0], track0);
    const float ctl[3] = {p.intensity, p.wind, p.turbulence};
    nvfx_instance_set_controls(in, ctl, 3);
    nvfx_instance_set_variation(in, 0);
    nvfx_instance_set_handoff(in, cfg[1].frames, cfg[1].fade);
    std::vector<std::uint8_t> f(sz(kSize) * kSize * 4);
    std::size_t differ = 0;
    for (int i = 0; i < kExplosionFrames; ++i) {
      nvfx_render(in, (i + 1) / 30.0, f.data(), sz(kSize) * 4);
      for (std::size_t j = 0; j < f.size(); ++j) differ += f[j] != blended0[sz(i) * f.size() + j];
    }
    nvfx_instance_free(in);
    nvfx_effect_free(fx);
    std::println("i2-handoff test: the C API with the chosen hand-off against the frames scored: {} bytes differ in {} frames", differ, kExplosionFrames);
    if (differ != 0) throw std::runtime_error("the runtime's hand-off differs from the frames scored");
  }
  // endless statistics: one 89-frame play per setting, as study I's test
  const auto t1 = std::chrono::steady_clock::now();
  std::vector<std::vector<metrics::StatDistance>> st(cfg.size() + 1, std::vector<metrics::StatDistance>(settings.size()));
  {
    RuntimeEffect fx(file);
    parallel(c, static_cast<int>(settings.size()), [&](int si) {
      const Setting& s = settings[sz(si)];
      const metrics::ClipStats ref = metrics::stats(real_clip(e, s, real0 + static_cast<std::uint64_t>(si), 1, kExplosionFrames));
      const std::vector<float> ctl{s[0], s[1], s[2]};
      for (std::size_t k = 0; k < cfg.size(); ++k) {
        st[k][sz(si)] = metrics::distance(ref, metrics::stats(runtime_clip(fx.e, ctl, model0 + static_cast<std::uint64_t>(si), kExplosionFrames, {cfg[k].frames, cfg[k].fade})));
      }
      if (other0) st[cfg.size()][sz(si)] = metrics::distance(ref, metrics::stats(real_clip(e, s, other0 + static_cast<std::uint64_t>(si), 1, kExplosionFrames)));
    });
  }
  std::println("i2-handoff {}: endless runs in {:.0f} s", c.split, seconds_since(t1));
  // per config: intervals against no hand-off
  std::vector<std::string> rows, case_rows, stat_rows;
  const std::vector<std::string> names = {"spectrum_l1(-)", "abs_log_motion(-)", "coverage_l1(-)", "mean_frame_psnr"};
  struct Score {
    double gain = 0;
    bool eligible = false;
  };
  std::vector<Score> score(cfg.size());
  for (std::size_t k = 0; k < cfg.size(); ++k) {
    std::vector<double> a1, b1, a2, b2;
    for (int q = 0; q < ncase; ++q) {
      a1.push_back(mean_range(tr[sz(q)][k], 1, 30));
      b1.push_back(mean_range(tr[sz(q)][0], 1, 30));
      a2.push_back(mean_range(tr[sz(q)][k], 31, kExplosionFrames));
      b2.push_back(mean_range(tr[sz(q)][0], 31, kExplosionFrames));
      case_rows.push_back(std::format("{},{},{},{},{:.4f},{:.4f}", cfg[k].name(), cfg[k].frames, cfg[k].fade, q, a1.back(), a2.back()));
    }
    for (std::size_t si = 0; si < settings.size(); ++si) stat_rows.push_back(stat_row(en, cfg[k].name(), static_cast<int>(si), st[k][si]));
    const auto first = metrics::paired_bootstrap(a1, b1), rest = metrics::paired_bootstrap(a2, b2);
    std::string row = std::format("{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}", cfg[k].name(), cfg[k].frames, cfg[k].fade, mean(a1), first.mean, first.lo,
                                  first.hi, mean(a2), rest.mean, rest.lo, rest.hi);
    bool endless = true;
    for (int q = 0; q < 4; ++q) {
      std::vector<double> a, b;
      const auto val = [&](const metrics::StatDistance& d) {
        switch (q) {
          case 0: return -d.spectrum_l1;
          case 1: return -std::abs(std::log(std::max(1e-3, d.motion_ratio)));
          case 2: return -d.coverage_l1;
          default: return d.mean_frame_psnr;
        }
      };
      for (std::size_t si = 0; si < settings.size(); ++si) {
        a.push_back(val(st[k][si]));
        b.push_back(val(st[0][si]));
      }
      const auto d = metrics::paired_bootstrap(a, b);
      endless = endless && d.hi >= 0;
      row += std::format(",{:.4f},{:.4f},{:.4f}", d.mean, d.lo, d.hi);
    }
    score[k] = {first.mean, k > 0 && endless && rest.hi >= 0};
    const bool kept = k > 0 && first.lo > 0 && rest.hi >= 0 && endless;
    row += std::format(",{},{}", score[k].eligible ? 1 : 0, test ? (kept ? "kept" : (k > 0 ? "not_kept" : "")) : "");
    rows.push_back(row);
    std::println("i2-handoff {} {}: first second {:.2f} dB, gain {} ; frames 31-89 gain {} ; endless not worse {}{}", c.split, cfg[k].name(), mean(a1), iv(first), iv(rest),
                 endless, test && k > 0 ? (kept ? " -> KEPT" : " -> not kept") : "");
  }
  for (std::size_t si = 0; si < settings.size() && other0; ++si) stat_rows.push_back(stat_row(en, "real_other_seed", static_cast<int>(si), st[cfg.size()][si]));
  const std::string tag = test ? "test" : "val";
  {
    std::ofstream o(c.results / std::format("i2_handoff_{}.csv", tag));
    o << "config,frames,fade,first_second_psnr,first_gain,first_lo,first_hi,frames_31_89_psnr,rest_gain,rest_lo,rest_hi,spectrum_l1(-),lo,hi,abs_log_motion(-),lo,hi,"
         "coverage_l1(-),lo,hi,mean_frame_psnr,lo,hi,eligible,decision\n";
    for (const auto& r : rows) o << r << "\n";
  }
  {
    std::ofstream o(c.results / std::format("i2_handoff_{}_track.csv", tag));
    o << "config,frames,fade,case,first_second_psnr,frames_31_89_psnr\n";
    for (const auto& r : case_rows) o << r << "\n";
  }
  {
    std::ofstream o(c.results / std::format("i2_handoff_{}_stats.csv", tag));
    o << "effect,config,setting,spectrum_l1,motion_ratio,coverage_l1,emission_l1,mean_frame_psnr\n";
    for (const auto& r : stat_rows) o << r << "\n";
  }
  if (!test) {  // the choice (§10.6): the best first-second gain among the eligible; within 0.05 dB of it, the fewest frames
    int best = -1;
    for (std::size_t k = 1; k < cfg.size(); ++k) {
      if (score[k].eligible && (best < 0 || score[k].gain > score[sz(best)].gain)) best = static_cast<int>(k);
    }
    if (best < 0) {
      std::println("i2-handoff val: no configuration keeps the endless statistics and frames 31-89: stopped at validation");
      fs::remove(choice_file);
      return;
    }
    int pick = best;
    for (std::size_t k = 1; k < cfg.size(); ++k) {
      if (!score[k].eligible || score[k].gain < score[sz(best)].gain - 0.05) continue;
      const int span = cfg[k].frames + cfg[k].fade, ps = cfg[sz(pick)].frames + cfg[sz(pick)].fade;
      if (span < ps || (span == ps && cfg[k].frames < cfg[sz(pick)].frames)) pick = static_cast<int>(k);
    }
    fs::create_directories(c.data2);
    std::ofstream(choice_file) << cfg[sz(pick)].frames << " " << cfg[sz(pick)].fade << "\n";
    std::println("i2-handoff val: best {} ({:+.2f} dB); chosen {} ({:+.2f} dB)", cfg[sz(best)].name(), score[sz(best)].gain, cfg[sz(pick)].name(), score[sz(pick)].gain);
  }
}

// --- i2-cost -----------------------------------------------------------------------------------------------------------

void step_cost2(const Ctx& c) {
  const fs::path file = c.v2 / "explosion.nvfx";
  RuntimeEffect fx(file);
  const rollout::Model M = load_or_throw(file);
  std::vector<std::string> rows;
  const int reps = c.quick ? 3 : 15;
  for (const int size : {64, 128, 256}) {
    double best[2] = {1e30, 1e30}, look_best = 1e30;
    for (int with = 0; with < 2; ++with) {
      nvfx_instance* in = nullptr;
      if (nvfx_instance_create(fx.e, size, &in) != NVFX_OK) throw std::runtime_error("instance");
      const float ctl[3] = {0.6f, 0.5f, 0.5f};
      nvfx_instance_set_controls(in, ctl, 3);
      nvfx_instance_set_seed(in, 77);
      nvfx_instance_set_handoff(in, with ? 30 : 0, 0);
      std::vector<std::uint8_t> f(sz(size) * sz(size) * 4);
      for (int r = 0; r < reps; ++r) {
        nvfx_render(in, 0.0, f.data(), sz(size) * 4);  // restart, untimed
        const double t = thread_seconds();
        for (int k = 1; k <= 30; ++k) nvfx_render(in, k / 30.0, f.data(), sz(size) * 4);
        best[with] = std::min(best[with], (thread_seconds() - t) / 30.0);
      }
      nvfx_instance_free(in);
    }
    {  // the drawing alone, on the runner's fields at frame 15
      rt::RolloutEffect re;
      re.m = M;
      auto run = compose::make_runner(re, size, compose::best_isa());
      const std::vector<float> ctl{0.6f, 0.5f, 0.5f};
      run->start(0, ctl, 77);
      for (int k = 0; k < 15; ++k) run->step(ctl, 77);
      std::vector<std::uint8_t> f(sz(size) * sz(size) * 4);
      for (int r = 0; r < reps; ++r) {
        const double t = thread_seconds();
        for (int k = 0; k < 10; ++k) rt::draw_sim_look(rt::SimLook::explosion, run->fine_heat(), run->fine_soot(), size, f.data(), sz(size) * 4);
        look_best = std::min(look_best, (thread_seconds() - t) / 10.0);
      }
    }
    rows.push_back(std::format("explosion,{},{:.4f},{:.4f},{:.4f},{:.4f}", size, 1e3 * best[0], 1e3 * best[1], 1e3 * (best[1] - best[0]), 1e3 * look_best));
    std::println("i2-cost: {} px: {:.3f} ms per frame in the first second without the hand-off, {:.3f} ms with it (+{:.3f}); the simulator's look alone {:.3f} ms", size,
                 1e3 * best[0], 1e3 * best[1], 1e3 * (best[1] - best[0]), 1e3 * look_best);
  }
  std::ofstream o(c.results / "i2_handoff_cost.csv");
  o << "effect,size,learned_ms,with_handoff_ms,extra_ms,sim_look_ms\n";
  for (const auto& r : rows) o << r << "\n";
}

}  // namespace nfx::study_i
