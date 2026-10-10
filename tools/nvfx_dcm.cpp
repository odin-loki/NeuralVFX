// nvfx_dcm: diffusion-context mixing (docs/DCM.md), the owner's mixer and its search (ported from CameraDetector).
//
//   nvfx_dcm selftest [--threads 2] [--seed 5]   a small synthetic search under both objectives (classifying: ROC-AUC;
//                                                generating: Laplace bits), printing what it chose
//   nvfx_dcm version FILE                        the SHA-256 version of a serialised mixer (nvfx-paq-mixer v1 or
//                                                nvfx-value-mixer v1), after checking that it parses
//
// DCM-fine (docs/DCM.md G1), per effect (--effect fire|smoke|explosion, --threads 2, data under NEURALVFX_DATA/g/fine):
//   nvfx_dcm record --effect E          pixel rows of 48 training (salt 1) and 16 validation (salt 3) runs, and windows
//   nvfx_dcm experts --effect E         the spec (bins, regional clusters); each expert alone on the validation rows
//   nvfx_dcm search-fine --effect E [--pilot] [--family none|hand|hand+macro] [--domain linear|log] [--seed 0]
//                                       the pilot (default mixer against the v1 lock), or the nested mixer search
//   nvfx_dcm train-fine --effect E      own-rollout pass, tau and relock on validation settings, re-ranking, release
//   nvfx_dcm eval-fine --effect E       G1c and the test (once), v1 against DCM-fine through the reference
//   nvfx_dcm bench-experts --effect E   microseconds per pixel of each expert group, context and the mixer
//   nvfx_dcm fine-summary               paired-bootstrap tables of the CSVs
// The coarse-state denoiser (include/neuralfx/dcm/ddpm.hpp, study G stage S5; data under NEURALVFX_DATA/g/diff):
//   nvfx_dcm ddpm-train [--effect fire] [--steps 12000] [--batch 32] [--threads 2] ...
//                         records the training runs (salt 1) and validation runs (salt 3) once, as study D's d-train
//                         does, keeps every second coarse state after the warm-up, and trains the denoiser; logs the
//                         EMA loss at t = 50, 200, 500, 800 on validation states (results/experiments/g_diff_train.csv)
//   nvfx_dcm ddpm-sample [--effect fire] [--controls 0.5,0.5,0.5] [--n 8] [--steps 25] [--t0 400]
//                         fresh DDIM samples (and SDEdits of validation states with --t0) against real states: channel
//                         means, spreads and time per sample
//   nvfx_dcm ddpm-time [--effect fire] [--core 3]
//                         one denoiser pass on one pinned core (median, p90) and the load average around it
//   nvfx_dcm contexts [--effect fire] [--fit 1500] [--threads 2]
//                         G2a: denoiser contexts (t = 400 and 600, levels 16 x 16 and 8 x 8, PCA-16, k-means per 8 x 8
//                         region) and plain coarse-statistics contexts, fitted on training states; their mutual
//                         information with hand-made contexts on validation states (results/experiments/g_diff_nmi.csv)
//                         and a table of every validation region's contexts (data directory)
#include "args.hpp"
#include "fine_study.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/dcm/compact.hpp>
#include <neuralfx/dcm/ddpm.hpp>
#include <neuralfx/dcm/search.hpp>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <numeric>
#include <thread>
#include <atomic>
#include <print>
#include <random>
#include <sched.h>
#include <sstream>
#include <string>

using namespace nfx;

namespace {

// Rows of a training pool (site 0, folds 0-2) and four test sites. Inputs: "good" (column 0) and noise; contexts
// "shuffle" (random), "regime" (2 values: it flips the label's evidence, or selects the target's slope) and "spare"
// (random). With value = true the target is a real number: y = (1.5 or −0.5 by regime) · good + Laplace noise whose
// scale grows with the scale feature u.
dcm::SearchProblem synthetic(int per, unsigned seed, bool value) {
  dcm::SearchProblem p;
  p.input_names = {"good", "noise_a", "noise_b", "junk"};
  p.groups = {{"good", {0}}, {"noise", {1, 2}}, {"junk", {3}}};
  p.context_names = {"shuffle", "regime", "spare"};
  p.context_sizes = {3, 2, 4};
  p.site_names = {"pool", "site_a", "site_b", "site_c", "site_d"};
  p.holdout = {false, true, true, true, true};
  p.delta = 0.05;
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd;
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::exponential_distribution<double> ex(1.0);
  std::bernoulli_distribution half(0.5), positive(0.4);
  for (int s = 0; s < 5; ++s) {
    const int n = s == 0 ? 3 * per : per;
    for (int i = 0; i < n; ++i) {
      const int regime = half(rng) ? 1 : 0;
      if (value) {
        const double good = nd(rng), u = uni(rng);
        const double noise = (0.1 + 0.3 * u) * ex(rng) * (half(rng) ? 1.0 : -1.0);
        p.x.push_back({good, nd(rng), nd(rng), nd(rng)});
        p.z.push_back({u});
        p.y.push_back((regime ? 1.5 : -0.5) * good + noise);
      } else {
        const int y = positive(rng) ? 1 : 0;
        const double evidence = (y ? 1.2 : -1.2) * (regime ? -1.0 : 1.0);
        p.x.push_back({evidence + nd(rng), nd(rng), nd(rng), nd(rng)});
        p.y.push_back(y);
      }
      p.contexts.push_back({static_cast<int>(rng() % 3), regime, static_cast<int>(rng() % 4)});
      p.site.push_back(s);
      p.fold.push_back(s == 0 ? i % 3 : -1);
    }
  }
  return p;
}

int selftest(const tools::Args& a) {
  bool ok = true;
  for (const auto objective : {dcm::Objective::auc, dcm::Objective::laplace_bits}) {
    const bool value = dcm::value_objective(objective);
    const auto p = synthetic(150, 1, value);
    dcm::SearchOptions o;
    o.objective = objective;
    o.configs = 16;
    o.refine_rounds = 2;
    o.threads = a.i("threads", 2);
    o.seed = a.u64("seed", 5);
    o.space.max_mixer_contexts = 1;  // the default mixer then uses "shuffle": the search has to find "regime"
    if (value) o.space.lr1 = o.space.lr2 = {0.01, 0.03, 0.1};
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = dcm::run_search(p, o);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::println("objective {}: {} candidates, {} trainings, {:.2f} s", dcm::objective_name(objective), r.candidates,
                 r.trainings, sec);
    const auto unit = value ? "held-out bits per row" : "ROC-AUC";
    const auto shown = [&](double score) { return value ? -score : score; };
    std::println("  global best   {}  ({} {:.4f}, optimistic)", dcm::describe(p, o.space, r.global_best.config, objective),
                 unit, shown(r.global_best.score));
    for (const auto& c : r.nested_choices) {
      std::println("  nested {:7}  {}  (inner {:.4f} over {} sites)", p.site_names[static_cast<std::size_t>(c.site)],
                   dcm::describe(p, o.space, c.config, objective), shown(c.inner_auc), c.inner_sites);
    }
    std::println("  rfonly best   {}  ({:.4f} over the pool's folds)", dcm::describe(p, o.space, r.rfonly_best.config, objective),
                 shown(r.rfonly_best.score));
    const auto scores = value ? dcm::per_site_loss(p, objective, r.nested, r.nested_b) : dcm::per_site_auc(p, r.nested);
    std::print("  nested {} per test site:", unit);
    for (std::size_t s = 1; s < scores.size(); ++s) std::print(" {:.4f}", scores[s]);
    std::println("");
    std::println("  released mixer: version {} ({} bytes of text)", r.global_version, r.global_mixer.size());
    const bool found = std::ranges::find(r.global_best.config.mixer_contexts, 1) != r.global_best.config.mixer_contexts.end();
    std::println("  the informative context \"regime\" {}", found ? "was chosen" : "was NOT chosen");
    ok = ok && found;
  }
  std::println("selftest {}", ok ? "passed" : "FAILED");
  return ok ? 0 : 1;
}

int version(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();
  // parse it first, so a damaged or foreign file is refused rather than given a version
  if (text.starts_with("nvfx-paq-mixer v1\n")) {
    const dcm::CompactMixer<double> m(text);
    std::println("{}  paq-mixer  {} inputs, {} first-layer mixers  {}", m.version(), m.inputs(), m.first_layer(), path);
  } else if (text.starts_with("nvfx-value-mixer v1\n")) {
    const dcm::CompactValueNet<double> m(text);
    std::println("{}  value-mixer  {} inputs, {} first-layer mixers, {} scale features  {}", m.version(), m.inputs(),
                 m.first_layer(), m.scale_features(), path);
  } else {
    throw std::runtime_error(path + " is not a serialised mixer (nvfx-paq-mixer v1 or nvfx-value-mixer v1)");
  }
  return 0;
}

// --- the coarse-state denoiser ---------------------------------------------------------------------------------

namespace fs = std::filesystem;
namespace dd = nfx::dcm::ddpm;

struct DiffPaths {
  sim::Effect effect = sim::Effect::fire;
  std::string name;
  fs::path dir, dmodel, train_states, val_states, denoiser;
};

DiffPaths diff_paths(const tools::Args& a) {
  DiffPaths p;
  if (!sim::parse_effect(a.str("effect", "fire"), p.effect)) throw std::invalid_argument("unknown --effect");
  p.name = std::string(sim::effect_name(p.effect));
  p.dir = a.has("dir") ? fs::path(a.str("dir")) : data_root() / "g" / "diff";
  p.dmodel = a.has("dmodel") ? fs::path(a.str("dmodel")) : data_root() / "experiments" / "models" / "d" / (p.name + ".nvfx");
  p.train_states = p.dir / (p.name + "_train.states");
  p.val_states = p.dir / (p.name + "_val.states");
  p.denoiser = a.has("model") ? fs::path(a.str("model")) : p.dir / (p.name + ".ddpm");
  return p;
}

rollout::Model load_dmodel(const DiffPaths& p) {
  auto m = rollout::load_model(p.dmodel);
  if (!m) throw std::runtime_error(p.dmodel.string() + ": " + m.error());
  return std::move(*m);
}

// The training runs of study D's recipe (salt 1) or the validation runs (salt 3), recorded once and kept on disk.
dd::Dataset states(const tools::Args& a, const DiffPaths& p, const rollout::Model& m, bool validation) {
  const fs::path file = validation ? p.val_states : p.train_states;
  if (fs::exists(file)) {
    auto d = dd::load_dataset(file);
    if (!d) throw std::runtime_error(d.error());
    return std::move(*d);
  }
  rollout::SimRecipe r = rollout::recipe_for(p.effect);
  r.salt = validation ? 3 : 1;
  r.runs = validation ? a.i("val-runs", 16) : a.i("runs", r.runs);
  r.threads = a.i("threads", 2);
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<rollout::Run> runs = rollout::record_runs(r);
  const int first = a.i("first", 30), every = a.i("every", 2);
  dd::Dataset d = dd::states_from_runs(m, runs, first, every);
  std::println("{} states: {} runs (salt {}) x {} frames, from frame {} every {}: {} states in {:.0f} s", validation ? "validation" : "training",
               r.runs, r.salt, r.frames, first, every, d.count, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  if (auto w = dd::save_dataset(file, d); !w) throw std::runtime_error(w.error());
  return d;
}

void channel_summary(std::string_view label, const std::vector<float>& x, int channels) {
  std::vector<double> mean(static_cast<std::size_t>(channels), 0.0), sq(static_cast<std::size_t>(channels), 0.0);
  const std::size_t n = x.size() / static_cast<std::size_t>(channels);
  for (std::size_t i = 0; i < x.size(); ++i) {
    mean[i % static_cast<std::size_t>(channels)] += x[i];
    sq[i % static_cast<std::size_t>(channels)] += static_cast<double>(x[i]) * x[i];
  }
  std::print("  {:<28}", label);
  for (int k = 0; k < channels; ++k) {
    const double m = mean[static_cast<std::size_t>(k)] / static_cast<double>(n);
    std::print("  ch{} mean {:+.3f} sd {:.3f}", k, m, std::sqrt(std::max(0.0, sq[static_cast<std::size_t>(k)] / static_cast<double>(n) - m * m)));
  }
  std::println("");
}

int ddpm_train(const tools::Args& a) {
  const DiffPaths p = diff_paths(a);
  const rollout::Model m = load_dmodel(p);
  const dd::Dataset train = states(a, p, m, false), val = states(a, p, m, true);
  channel_summary("training states", train.x, rollout::kPhys);
  channel_summary("validation states", val.x, rollout::kPhys);
  dd::Config c;
  c.cond = m.h.cond();
  dd::Denoiser d = dd::init_denoiser(c, a.u64("init-seed", 1));
  d.scale.assign(m.scale.begin(), m.scale.end());
  dd::set_range(d, train);
  // validation states for the loss curve: spread over the validation runs
  dd::Dataset eval;
  eval.values = val.values;
  eval.conds = val.conds;
  const int want = a.i("eval-count", 256);
  for (int i = 0; i < want && val.count > 0; ++i) {
    const int s = static_cast<int>(static_cast<long long>(i) * val.count / want);
    const auto x = val.state(s), cnd = val.condition(s);
    eval.x.insert(eval.x.end(), x.begin(), x.end());
    eval.cond.insert(eval.cond.end(), cnd.begin(), cnd.end());
    ++eval.count;
  }
  dd::TrainOptions o;
  o.steps = a.i("steps", 12000);
  o.batch = a.i("batch", 32);
  o.lr = a.f("lr", 5e-4f);
  o.warmup = a.i("warmup", 500);
  o.clip = a.f("clip", 1.f);
  o.threads = a.i("threads", 2);
  o.seed = a.u64("seed", 1);
  o.log_every = a.i("log-every", 500);
  o.eval_count = eval.count;
  const fs::path curve_csv = a.str("curve", "results/experiments/g_diff_train.csv");
  std::vector<std::string> rows;
  o.progress = [&](const dd::TrainLog& l, const dd::Denoiser& ema) {
    std::print("step {:6d}  loss {:.4f}  lr {:.2e}  {:.3f} s/step", l.step, l.loss, l.lr, l.seconds / l.step);
    for (std::size_t k = 0; k < l.eval.size(); ++k) std::print("  t={} {:.4f}", o.eval_t[k], l.eval[k]);
    std::println("");
    std::fflush(stdout);
    std::string row = std::format("{},{},{:.5f},{:.3e},{:.1f}", p.name, l.step, l.loss, l.lr, l.seconds);
    for (const double v : l.eval) row += std::format(",{:.5f}", v);
    rows.push_back(row);
    if (auto w = dd::save(fs::path(p.denoiser.string() + ".ckpt"), ema); !w) std::println(stderr, "checkpoint: {}", w.error());
  };
  std::println("denoiser: {} parameters, {:.1f} M multiply-adds per pass; {} steps of batch {} on {} threads", d.parameters(),
               dd::forward_macs(c) / 1e6, o.steps, o.batch, o.threads);
  const dd::TrainResult res = dd::train(d, train, o, &eval);
  if (auto w = dd::save(p.denoiser, d); !w) throw std::runtime_error(w.error());
  std::println("trained in {:.0f} s ({:.3f} s/step); saved {} version {}", res.seconds, res.seconds / o.steps, p.denoiser.string(), dd::version(d));
  if (!a.flag("no-csv")) {
    fs::create_directories(curve_csv.parent_path());
    std::ofstream csv(curve_csv);
    csv << "effect,step,train_loss,lr,seconds";
    for (const int t : o.eval_t) csv << ",val_loss_t" << t;
    csv << "\n";
    for (const auto& r : rows) csv << r << "\n";
  }
  return 0;
}

int ddpm_sample(const tools::Args& a) {
  const DiffPaths p = diff_paths(a);
  auto loaded = dd::load(p.denoiser);
  if (!loaded) throw std::runtime_error(loaded.error());
  const dd::Denoiser d = std::move(*loaded);
  const rollout::Model m = load_dmodel(p);
  std::vector<float> ctl = tools::parse_floats(a.str("controls", "0.5,0.5,0.5"));
  std::vector<float> cond(static_cast<std::size_t>(m.h.cond()));
  rollout::condition(m, ctl, 5.f, cond);
  const int n = a.i("n", 8), steps = a.i("steps", 25);
  const std::size_t vals = static_cast<std::size_t>(d.cfg.res * d.cfg.res * d.cfg.channels);
  std::vector<float> all, one(vals);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i) {
    dd::sample(d, cond, steps, a.u64("seed", 1) + static_cast<std::uint64_t>(i), one);
    all.insert(all.end(), one.begin(), one.end());
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n;
  std::println("denoiser {} ({}), controls {}: {} DDIM samples of {} steps, {:.1f} ms each", p.denoiser.string(), dd::version(d).substr(0, 12),
               a.str("controls", "0.5,0.5,0.5"), n, steps, ms);
  channel_summary("fresh samples", all, d.cfg.channels);
  // real states nearest the controls, from the validation runs
  auto val = dd::load_dataset(p.val_states);
  if (val) {
    std::vector<std::pair<float, int>> near;
    for (int i = 0; i < val->count; ++i) {
      float dist = 0;
      for (std::size_t k = 0; k < ctl.size() && k < 3; ++k) dist += (val->condition(i)[k] - ctl[k]) * (val->condition(i)[k] - ctl[k]);
      near.emplace_back(dist, i);
    }
    std::ranges::sort(near);
    std::vector<float> real, edited;
    const int t0e = a.i("t0", 400);
    for (int i = 0; i < std::min<int>(n * 10, static_cast<int>(near.size())); i += 10) {
      const auto x = val->state(near[static_cast<std::size_t>(i)].second);
      real.insert(real.end(), x.begin(), x.end());
      dd::sdedit(d, x, t0e, steps, cond, 99 + static_cast<std::uint64_t>(i), one);
      edited.insert(edited.end(), one.begin(), one.end());
    }
    channel_summary("real states, nearest controls", real, d.cfg.channels);
    channel_summary(std::format("SDEdit of those from t = {}", t0e), edited, d.cfg.channels);
  }
  return 0;
}

// One denoiser pass on one pinned core: median and 90th percentile of 200 passes, with the load average before and
// after (the rules count it as measured only below 1.5).
int ddpm_time(const tools::Args& a) {
  const DiffPaths p = diff_paths(a);
  auto loaded = dd::load(p.denoiser);
  if (!loaded) throw std::runtime_error(loaded.error());
  const dd::Denoiser d = std::move(*loaded);
  const auto load = [] {
    std::ifstream f("/proc/loadavg");
    double v = 99;
    f >> v;
    return v;
  };
  const double before = load();
  cpu_set_t one;
  CPU_ZERO(&one);
  CPU_SET(a.i("core", 3), &one);
  sched_setaffinity(0, sizeof(one), &one);
  std::vector<float> x(static_cast<std::size_t>(d.cfg.res * d.cfg.res * d.cfg.channels)), eps(x.size());
  dd::gaussian(5, x);
  const std::vector<float> cond(static_cast<std::size_t>(d.cfg.cond), 0.5f);
  for (int i = 0; i < 10; ++i) dd::predict_eps(d, x, 50, cond, eps);
  std::vector<double> ms;
  for (int i = 0; i < a.i("n", 200); ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    dd::predict_eps(d, x, 50, cond, eps);
    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::ranges::sort(ms);
  const double after = load();
  std::println("one pass: median {:.3f} ms, p90 {:.3f} ms ({:.1f} M multiply-adds, {:.1f} GMAC/s); load {:.2f} before, {:.2f} after: {}", ms[ms.size() / 2],
               ms[ms.size() * 9 / 10], dd::forward_macs(d.cfg) / 1e6, dd::forward_macs(d.cfg) / ms[ms.size() / 2] / 1e6, before, after,
               before < 1.5 && after < 1.5 ? "quiet, measured" : "busy, an upper bound (unmeasured by the rules)");
  return 0;
}

// Entropies and mutual information (bits) of two labellings, from their contingency counts.
struct Info {
  double ha = 0, hb = 0, mi = 0;
};
Info information(std::span<const int> a, std::span<const int> b) {
  std::map<int, double> ca, cb;
  std::map<std::pair<int, int>, double> cab;
  for (std::size_t i = 0; i < a.size(); ++i) {
    ca[a[i]] += 1;
    cb[b[i]] += 1;
    cab[{a[i], b[i]}] += 1;
  }
  const double n = static_cast<double>(a.size());
  Info r;
  for (const auto& [k, v] : ca) r.ha -= v / n * std::log2(v / n);
  for (const auto& [k, v] : cb) r.hb -= v / n * std::log2(v / n);
  for (const auto& [k, v] : cab) r.mi += v / n * std::log2((v / n) / ((ca[k.first] / n) * (cb[k.second] / n)));
  return r;
}

int contexts(const tools::Args& a) {
  const DiffPaths p = diff_paths(a);
  auto loaded = dd::load(p.denoiser);
  if (!loaded) throw std::runtime_error(loaded.error());
  const dd::Denoiser d = std::move(*loaded);
  const rollout::Model m = load_dmodel(p);
  const dd::Dataset train = states(a, p, m, false), val = states(a, p, m, true);
  const int threads = a.i("threads", 2), nfit = std::min(a.i("fit", 1500), train.count);
  std::vector<int> fit_states, val_states;
  for (int i = 0; i < nfit; ++i) fit_states.push_back(static_cast<int>(static_cast<long long>(i) * train.count / nfit));
  const int nval = std::min(a.i("val", val.count), val.count);
  for (int i = 0; i < nval; ++i) val_states.push_back(static_cast<int>(static_cast<long long>(i) * val.count / nval));
  const int R = d.cfg.res, C = d.cfg.channels;
  const auto t0 = std::chrono::steady_clock::now();
  dd::ContextSpec sd, sp, sd2;
  sp.diffusion = false;
  sd2.seed = sd.seed + 1;  // another k-means seed: the stability of the clusters
  const dd::ContextModel cd = dd::fit_contexts(&d, sd, train, fit_states, R, C, threads);
  const dd::ContextModel cp = dd::fit_contexts(nullptr, sp, train, fit_states, R, C, threads);
  const dd::ContextModel cd2 = dd::fit_contexts(&d, sd2, train, fit_states, R, C, threads);
  const dd::HandMade hm = dd::fit_handmade(train, fit_states, R, C);
  std::println("fitted on {} training states in {:.0f} s (PCA-16 keeps {:.1f}% of the denoiser features' variance, {:.1f}% of the plain ones')", nfit,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
               100.0 * std::accumulate(cd.pca.variance.begin(), cd.pca.variance.end(), 0.0) / static_cast<double>(cd.mean.size()),
               100.0 * std::accumulate(cp.pca.variance.begin(), cp.pca.variance.end(), 0.0) / static_cast<double>(cp.mean.size()));
  // labels of every validation region
  const std::size_t NR = static_cast<std::size_t>(dd::kRegions * dd::kRegions), nk = sd.ks.size();
  std::vector<std::vector<int>> dl(nk), pl(nk), dl2(nk);
  std::vector<std::vector<std::vector<int>>> per_d(static_cast<std::size_t>(nval)), per_p(per_d), per_d2(per_d);
  std::vector<std::vector<std::array<int, dd::HandMade::kKinds>>> per_h(static_cast<std::size_t>(nval));
  const std::vector<float> unit(static_cast<std::size_t>(C), 1.f);
  {
    std::atomic<int> next{0};
    std::vector<std::jthread> pool;
    for (int t = 0; t < threads; ++t) {
      pool.emplace_back([&] {
        for (int i; (i = next++) < nval;) {
          const int s = val_states[static_cast<std::size_t>(i)];
          const auto x = val.state(s), cnd = val.condition(s);
          per_d[static_cast<std::size_t>(i)] = dd::context_planes(&d, cd, x, C, unit, cnd, R);
          per_d2[static_cast<std::size_t>(i)] = dd::context_planes(&d, cd2, x, C, unit, cnd, R);
          per_p[static_cast<std::size_t>(i)] = dd::context_planes(nullptr, cp, x, C, unit, cnd, R);
          per_h[static_cast<std::size_t>(i)] = dd::handmade_contexts(hm, x, cnd.subspan(0, 3), R, C);
        }
      });
    }
  }
  std::array<std::vector<int>, dd::HandMade::kKinds + 1> hv;  // heat, height, flow, controls, joint
  std::ofstream table(p.dir / (p.name + "_contexts.csv"));
  table << "state,region";
  for (const int k : sd.ks) table << ",diffusion_k" << k;
  for (const int k : sp.ks) table << ",plain_k" << k;
  table << ",heat,height,flow,controls\n";
  for (int i = 0; i < nval; ++i) {
    for (std::size_t r = 0; r < NR; ++r) {
      table << val_states[static_cast<std::size_t>(i)] << "," << r;
      for (std::size_t k = 0; k < nk; ++k) {
        dl[k].push_back(per_d[static_cast<std::size_t>(i)][k][r]);
        dl2[k].push_back(per_d2[static_cast<std::size_t>(i)][k][r]);
        pl[k].push_back(per_p[static_cast<std::size_t>(i)][k][r]);
        table << "," << per_d[static_cast<std::size_t>(i)][k][r];
      }
      for (std::size_t k = 0; k < nk; ++k) table << "," << per_p[static_cast<std::size_t>(i)][k][r];
      const auto& h = per_h[static_cast<std::size_t>(i)][r];
      for (int q = 0; q < dd::HandMade::kKinds; ++q) {
        hv[static_cast<std::size_t>(q)].push_back(h[static_cast<std::size_t>(q)]);
        table << "," << h[static_cast<std::size_t>(q)];
      }
      hv[dd::HandMade::kKinds].push_back(((h[0] * 4 + h[1]) * 3 + h[2]) * 8 + h[3]);
      table << "\n";
    }
  }
  // mutual information
  std::vector<std::string> rows;
  const auto add = [&](std::string_view what, int k, std::string_view versus, std::span<const int> x, std::span<const int> y) {
    const Info inf = information(x, y);
    const double nmi = dcm::normalized_mutual_info(x, y);
    rows.push_back(std::format("{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}", p.name, what, k, versus, nmi, inf.ha > 0 ? inf.mi / inf.ha : 0.0, inf.mi,
                               inf.ha, inf.hb));
    std::println("  {:9} K={:2}  vs {:16} NMI {:.3f}  explained share {:.3f}  (I {:.3f} bits, H {:.3f} / {:.3f})", what, k, versus, nmi,
                 inf.ha > 0 ? inf.mi / inf.ha : 0.0, inf.mi, inf.ha, inf.hb);
  };
  const std::array<std::string, dd::HandMade::kKinds + 1> hnames = {"heat", "height", "flow", "controls", "hand_made_joint"};
  for (std::size_t k = 0; k < nk; ++k) {
    const int K = sd.ks[k];
    for (std::size_t q = 0; q < hnames.size(); ++q) add("diffusion", K, hnames[q], dl[k], hv[q]);
    add("diffusion", K, std::format("plain_k{}", K), dl[k], pl[k]);
    add("diffusion", K, "diffusion_seed2", dl[k], dl2[k]);
    for (std::size_t q = 0; q < hnames.size(); ++q) add("plain", K, hnames[q], pl[k], hv[q]);
    std::println("  diffusion K={:2}: ARI between two k-means seeds {:.3f}", K, dcm::adjusted_rand_index(dl[k], dl2[k]));
  }
  const fs::path out = a.str("nmi", "results/experiments/g_diff_nmi.csv");
  fs::create_directories(out.parent_path());
  std::ofstream csv(out);
  csv << "effect,contexts,k,versus,nmi,explained_share,mi_bits,h_contexts_bits,h_versus_bits\n";
  for (const auto& r : rows) csv << r << "\n";
  std::println("contexts: {} validation states x {} regions; wrote {} and {}", nval, NR, (p.dir / (p.name + "_contexts.csv")).string(), out.string());
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"help", "quick", "pilot", "no-csv"});
  const auto& pos = a.positional();
  if (a.flag("help") || pos.empty()) {
    std::println("nvfx_dcm selftest [--threads 2] [--seed 5] | version FILE | record|experts|search-fine [--pilot]|train-fine|eval-fine|"
                 "bench-experts --effect fire|smoke|explosion [--threads 2] [--quick] | fine-summary | ddpm-train | ddpm-sample | ddpm-time | "
                 "contexts  (options: see the source's header)");
    return 0;
  }
  int rc = 0;
  const auto fine_ctx = [&] {
    fine_study::Ctx c;
    c.data = a.has("data") ? std::filesystem::path(a.str("data")) : data_root() / "g" / "fine";
    c.models = a.has("models") ? std::filesystem::path(a.str("models")) : data_root() / "experiments" / "models" / "d";
    c.results = a.str("results", "results/experiments");
    c.figures = a.str("figures", "docs/figures");
    c.threads = a.i("threads", 2);
    c.quick = a.flag("quick");
    if (c.quick) {
      c.data /= "quick";
      c.results /= "quick";
      c.figures = c.data / "figures";
    }
    c.seed = a.u64("seed", 0);
    c.family = a.str("family", "hand+macro");
    c.domain = a.str("domain", "linear");
    c.configs = a.i("configs", 200);
    c.refine = a.i("refine", 2);
    c.max_rows = a.i("max-rows", 0);
    c.budget_ms = a.f("budget", 1.f);
    return c;
  };
  const auto effect = [&] {
    sim::Effect e{};
    if (!sim::parse_effect(a.need("effect"), e)) throw std::invalid_argument("--effect fire|smoke|explosion");
    return e;
  };
  if (pos[0] == "selftest") {
    rc = selftest(a);
  } else if (pos[0] == "version" && pos.size() == 2) {
    rc = version(pos[1]);
  } else if (pos[0] == "record") {
    fine_study::record(fine_ctx(), effect());
  } else if (pos[0] == "experts") {
    fine_study::experts(fine_ctx(), effect());
  } else if (pos[0] == "search-fine") {
    if (a.flag("pilot")) {
      fine_study::pilot(fine_ctx(), effect());
    } else {
      fine_study::search(fine_ctx(), effect());
    }
  } else if (pos[0] == "train-fine") {
    fine_study::train(fine_ctx(), effect());
  } else if (pos[0] == "eval-fine") {
    fine_study::eval(fine_ctx(), effect());
  } else if (pos[0] == "bench-experts") {
    fine_study::bench(fine_ctx(), effect());
  } else if (pos[0] == "probe-gen") {
    fine_study::probe(fine_ctx(), effect(), a.str("mixer"), a.f("tau", 0.f), a.i("relock", 0), a.i("frames", 90));
  } else if (pos[0] == "fine-summary") {
    fine_study::summary(fine_ctx());
  } else if (pos[0] == "ddpm-train") {
    rc = ddpm_train(a);
  } else if (pos[0] == "ddpm-sample") {
    rc = ddpm_sample(a);
  } else if (pos[0] == "contexts") {
    rc = contexts(a);
  } else if (pos[0] == "ddpm-time") {
    rc = ddpm_time(a);
  } else {
    throw std::invalid_argument("unknown command (nvfx_dcm --help)");
  }
  a.warn_unused();
  return rc;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_dcm: {}", e.what());
  return 2;
}
