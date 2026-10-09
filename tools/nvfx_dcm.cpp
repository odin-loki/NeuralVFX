// nvfx_dcm: diffusion-context mixing (docs/DCM.md), the owner's mixer and its search (ported from CameraDetector).
//
//   nvfx_dcm selftest [--threads 2] [--seed 5]   a small synthetic search under both objectives (classifying: ROC-AUC;
//                                                generating: Laplace bits), printing what it chose
//   nvfx_dcm version FILE                        the SHA-256 version of a serialised mixer (nvfx-paq-mixer v1 or
//                                                nvfx-value-mixer v1), after checking that it parses
//
// Later stages (docs/DCM.md §5) add the subcommands that search and release mixers on effect data.
#include "args.hpp"

#include <neuralfx/dcm/compact.hpp>
#include <neuralfx/dcm/search.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <print>
#include <random>
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

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"help"});
  const auto& pos = a.positional();
  if (a.flag("help") || pos.empty()) {
    std::println("nvfx_dcm selftest [--threads 2] [--seed 5] | version FILE");
    return 0;
  }
  int rc = 0;
  if (pos[0] == "selftest") {
    rc = selftest(a);
  } else if (pos[0] == "version" && pos.size() == 2) {
    rc = version(pos[1]);
  } else {
    throw std::invalid_argument("unknown command (nvfx_dcm selftest | version FILE)");
  }
  a.warn_unused();
  return rc;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_dcm: {}", e.what());
  return 2;
}
