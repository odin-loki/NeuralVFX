// The mixer search of diffusion-context mixing (DCM), ported from the owner's CameraDetector, cabinlab/src/diffusion
// (Phase 12, search.hpp). Fine experts are the inputs of a PAQ-style mixer (mixer.hpp); macro contexts (clusters of a
// denoiser's features, hand-made regimes) select its weight sets. The search chooses which input groups and contexts
// the mixer uses and how it learns: a seeded random sample of configurations plus a short hill-climbing refinement
// around the best (flip one group, add, drop or swap one context, step one learning setting). Every configuration is
// trained the same way: seeded shuffled passes over the training rows, the learning rates halved after each pass, then
// frozen.
//
// Rows belong to sites: a training pool that is always trained on and test sites that may be held out (in
// CameraDetector, cameras; here, for example, bins of an effect's controls). Three protocols:
//   nested  for each test site c, the configuration with the best mean per-site score of an inner leave-one-site-out
//           over the other test sites c' (train on everything except c and c', score c') is trained on everything
//           except c and predicts c. Site c's targets never reach anything that scores c: every training checks that
//           no site it scores is among the sites it trained on, and throws std::logic_error otherwise.
//   global  one search scored by leave-one-site-out over every test site; c is predicted by a model trained without
//           c, but c's targets helped choose the configuration, so its scores are optimistic (a reference only). The
//           best configuration trained on every row and frozen gives the version hash of a deployable mixer.
//   rfonly  one search scored by cross-validation over the training pool's folds only (no test site's targets); the
//           chosen configuration, trained on the whole pool and frozen, scores every test site.
//
// Objectives. The original classifies (the ROC-AUC of a MixerNet's probabilities, per site, plain or stratified), and
// that path is kept exactly: the same inputs and seed give the same choices and the same numbers. NeuralVFX adds the
// value domain: under Objective::mse or Objective::laplace_bits every configuration is a ValueNet, the label is a real
// target, and a site's score is its mean squared error or its mean code length in bits (y quantised with the
// problem's delta). Lower is better there, so the search maximises the negated loss; every reported score (ScoredConfig,
// SiteChoice) is then that negated mean.
#pragma once

#include <neuralfx/dcm/mixer.hpp>

#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nfx::dcm {

enum class Objective { auc, mse, laplace_bits };

[[nodiscard]] std::string_view objective_name(Objective objective);
// mse and laplace_bits: ValueNet configurations and real targets.
[[nodiscard]] constexpr bool value_objective(Objective objective) noexcept { return objective != Objective::auc; }

// ROC-AUC of scores against 0/1 labels (ties count half); NaN when one class is missing. Ported from CameraDetector's
// probe.cpp (src/dcm/roc.cpp).
[[nodiscard]] double roc_auc(std::span<const float> scores, std::span<const int> y);

// Input columns switched on or off together by the search (one expert, or one family of experts).
struct SearchGroup {
  std::string name;
  std::vector<int> columns;  // input columns, 0 <= column < inputs
};

// The data of a mixer search, one row per example.
struct SearchProblem {
  std::vector<std::vector<double>> x;      // rows x inputs, already standardised (without targets)
  std::vector<std::string> input_names;    // one per input column
  std::vector<SearchGroup> groups;         // at least one
  std::vector<std::vector<int>> contexts;  // rows x context columns, 0 <= value < context_sizes[column]
  std::vector<std::string> context_names;  // one per context column
  std::vector<int> context_sizes;          // cardinality of each context column
  std::vector<double> y;                   // auc: 1 = positive, 0 = negative; value objectives: the target value
  std::vector<int> site;                   // index into site_names
  std::vector<std::string> site_names;
  std::vector<bool> holdout;               // per site: a test site (may be held out); false: the training pool
  std::vector<int> fold;                   // per row: cross-validation fold (>= 0) of a training-pool row, else -1
  // Optional, per row (>= 0), auc only: unlabelled strata. A site's ROC-AUC is then the size-weighted mean of the AUCs
  // of its strata that hold both classes (pairs are compared only within a stratum). Empty: one stratum per site.
  std::vector<int> stratum;
  // Value objectives only: per row, the scale net's features (the same count on every row; a constant 1 is added by
  // the net). Empty: none, so the scale depends on its context alone.
  std::vector<std::vector<double>> z;
  // laplace_bits only: the width of the bin a target is coded in (its quantisation step), > 0.
  double delta = 1.0;
};

// The grids the learning settings are drawn from (a configuration stores indices into them).
struct SearchSpace {
  std::vector<double> lr1 = {0.002, 0.005, 0.01, 0.02, 0.05};  // first-layer learning rates
  std::vector<double> lr2 = {0.002, 0.005, 0.01, 0.02, 0.05};  // final-mixer learning rates
  std::vector<int> epochs = {1, 2, 3, 4, 6, 8};                // shuffled passes over the training rows
  std::vector<double> apm_weight = {0.0, 0.15, 0.3, 0.5};      // auc: 0 means no APM
  int max_mixer_contexts = 3;  // context-selected first-layer mixers at most (besides the one without context)
  // Value objectives only.
  std::vector<ValueLoss> loss = {ValueLoss::squared, ValueLoss::laplace};  // the training loss
  std::vector<double> avm_weight = {0.0, 0.15, 0.3, 0.5};                  // 0: no AVM
};

// One mixer configuration.
struct SearchConfig {
  std::vector<int> groups;          // active groups, ascending, at least one
  std::vector<int> mixer_contexts;  // context columns, ascending, one first-layer mixer each; a first-layer mixer
                                    // without context is always present
  int apm_context = -1;             // context column of the APM (auc) or of the AVM (value objectives); -1: one table
                                    // for every row (and always -1 when the map's weight is 0)
  int lr1 = 0, lr2 = 0, epochs = 0, apm_weight = 0;  // indices into the SearchSpace grids (apm_weight: auc only)
  // Value objectives only (left at these defaults under auc):
  int loss = 0;                     // index into SearchSpace::loss
  int avm_weight = 0;               // index into SearchSpace::avm_weight
  int scale_context = -1;           // context column of the scale net; -1: one weight set for every row
};

// A configuration with its search score (a mean of per-site or per-fold scores: ROC-AUCs, or negated losses).
struct ScoredConfig {
  SearchConfig config;
  double score = 0.0;
};

// One training, reported to SearchOptions::on_train: which sites the model was trained on and which it scored.
struct TrainRecord {
  std::string_view stage;          // "nested", "global" or "rfonly"
  std::vector<int> excluded;       // sites held out of the training (ascending)
  int excluded_fold = -1;          // training-pool fold held out (rfonly cross-validation), else -1
  std::vector<int> trained_sites;  // sites with at least one training row (ascending), from the rows actually used
  std::vector<int> scored_sites;   // sites the model produced scores for (ascending)
};

// The inference cost of configurations (cost-aware search). Shared components (an encoder, a set of feature planes,
// a denoiser's prefix, ...) cost `ms` each; a group or a context column needs a set of components, and a configuration
// pays each component it needs once (its groups, its mixer contexts, its APM or AVM context when that map is on, and
// its scale context), plus base_ms. Without components every configuration costs base_ms and the search behaves as
// without costs.
struct SearchCost {
  std::vector<std::string> components;              // names
  std::vector<double> ms;                           // per component: milliseconds on one core, >= 0
  std::vector<std::vector<int>> group_components;   // per problem group: indices into components
  std::vector<std::vector<int>> context_components; // per context column: indices into components
  double base_ms = 0.0;                             // paid by every configuration
  // Configurations that cost more are never sampled, evaluated or chosen (the default mixer included).
  double budget_ms = std::numeric_limits<double>::infinity();
  // Selection maximises (mean score) − penalty · ms; the reported scores are then those penalised values.
  double penalty = 0.0;
  [[nodiscard]] bool active() const noexcept { return !components.empty(); }
};

// Milliseconds of a (canonical) configuration under `cost`.
[[nodiscard]] double config_cost(const SearchCost& cost, const SearchConfig& config);

struct SearchOptions {
  int configs = 200;        // random configurations (the first is default_config)
  int refine_rounds = 2;    // hill-climbing rounds around the best configuration
  int top = 10;             // global configurations reported
  std::uint64_t seed = 0;   // configuration sampling and the training shuffles
  int threads = 1;          // configuration evaluations in parallel
  bool nested = true, global = true, rfonly = true;  // protocols to run
  Objective objective = Objective::auc;
  // nested: also predict every row (the training pool and every test site) with each held-out site's final mixer.
  // That mixer never saw the site's targets, so its predictions are the site's teacher for distillation
  // (SearchResult::nested_teachers); the nested scores themselves do not change.
  bool nested_teachers = false;
  SearchSpace space;
  SearchCost cost;          // inference costs, a budget and a cost penalty (inactive by default)
  // Called after every training (serialised between threads): instrumentation for tests and audits.
  std::function<void(const TrainRecord&)> on_train;
};

// The chosen configuration of one held-out site under the nested protocol.
struct SiteChoice {
  int site = 0;
  SearchConfig config;
  double inner_auc = 0.0;  // the mean score of the inner leave-one-site-out over the other test sites (a ROC-AUC, or
                           // a negated loss under value objectives)
  int inner_sites = 0;     // test sites with a defined score in that mean
};

struct SearchResult {
  // Per row: the protocol's prediction (auc: the logit, rounded to float; value objectives: the predicted value μ);
  // NaN for rows it does not score (the training pool). Empty when the protocol was not run.
  std::vector<double> nested, global, rfonly;
  // Value objectives: the Laplace scale b of each prediction above (NaN where unscored); empty under auc.
  std::vector<double> nested_b, global_b, rfonly_b;
  std::vector<SiteChoice> nested_choices;  // one per test site, in site order
  // With SearchOptions::nested_teachers: per entry of nested_choices, the prediction (as in `nested`) of every row by
  // the mixer that predicts that site (trained without the site's targets). On the site's own rows it equals `nested`.
  std::vector<std::vector<double>> nested_teachers;
  std::vector<ScoredConfig> global_top;    // best first (optimistic: chosen with every test site's targets)
  ScoredConfig global_best;                // the released configuration (global_top.front() unless scores tie)
  std::string global_version;              // version of the global best trained on every row
  std::string global_mixer;                // its serialisation (MixerNet or ValueNet)
  ScoredConfig rfonly_best;                // score: mean over the training pool's folds
  std::string rfonly_version, rfonly_mixer;
  int candidates = 0;                      // distinct random configurations
  std::int64_t trainings = 0;              // mixers trained in all
  double nested_seconds = 0.0, global_seconds = 0.0, rfonly_seconds = 0.0;
};

// Throws std::invalid_argument when the problem's sizes, columns, contexts, sites, targets, folds or (for value
// objectives) scale features and delta do not agree. Under auc every target is 0 or 1; otherwise finite.
void validate(const SearchProblem& problem, Objective objective = Objective::auc);

// The MixerNet spec of a configuration (first-layer mixers: the one without context, then one per mixer context).
[[nodiscard]] MixerNetSpec mixer_spec(const SearchProblem& problem, const SearchSpace& space, const SearchConfig& config);

// The ValueNet spec of a configuration, before the training rows set its data-dependent parts (the AVM's range and the
// starting scale, see train_value_config).
[[nodiscard]] ValueNetSpec value_spec(const SearchProblem& problem, const SearchSpace& space, const SearchConfig& config);

// Readable one-line description, e.g. "groups cypha+cue | mixers -,light,c8 | apm light 0.3 | lr 0.02/0.01 | ep 4";
// under value objectives "groups ... | mixers ... | avm light 0.3 | scale - | loss laplace | lr ... | ep ...".
[[nodiscard]] std::string describe(const SearchProblem& problem, const SearchSpace& space, const SearchConfig& config,
                                   Objective objective = Objective::auc);

// The default mixer on this problem: every group, the first max_mixer_contexts context columns as first-layer
// mixers, the APM (or AVM) selected by the first context column, and the grid values nearest to lr 0.02 / 0.01,
// 4 epochs, map weight 0.3; under value objectives also no scale context and the loss that matches the objective
// (squared for mse, laplace for laplace_bits) when the grid has it.
[[nodiscard]] SearchConfig default_config(const SearchProblem& problem, const SearchSpace& space,
                                          Objective objective = Objective::auc);

// The configurations one step from `config` (flip one group; add, drop or swap one mixer context; change the map's
// context; step one learning setting; under value objectives also change the scale context and step the loss and the
// AVM weight), canonical and distinct, in a fixed order.
[[nodiscard]] std::vector<SearchConfig> neighbours(const SearchProblem& problem, const SearchSpace& space,
                                                   const SearchConfig& config, Objective objective = Objective::auc);

// Trains a configuration on the given rows (seeded shuffled passes, learning rates halved after each, then frozen).
// The searches train on the rows of each site in turn, in site order, each site's rows in row order.
[[nodiscard]] MixerNet train_config(const SearchProblem& problem, const SearchSpace& space, const SearchConfig& config,
                                    std::span<const std::size_t> rows, std::uint64_t seed);

// The same for value objectives. From the training rows alone it also sets the AVM's range (the targets' minimum and
// maximum) and the starting scale (log of their mean absolute deviation).
[[nodiscard]] ValueNet train_value_config(const SearchProblem& problem, const SearchSpace& space,
                                          const SearchConfig& config, std::span<const std::size_t> rows,
                                          std::uint64_t seed);

// The nested, global and rfonly searches selected in `options` (see the header comment).
[[nodiscard]] SearchResult run_search(const SearchProblem& problem, const SearchOptions& options);

// ROC-AUC of `scores` (one per row, NaN where unscored) within each site (stratified when the problem has strata);
// NaN for a site without scored rows of both classes.
[[nodiscard]] std::vector<double> per_site_auc(const SearchProblem& problem, std::span<const double> scores);

// The mean loss (squared error, or bits under laplace_bits) of predictions mu (and scales b, laplace_bits only) within
// each site; NaN where mu is NaN, and for a site without scored rows.
[[nodiscard]] std::vector<double> per_site_loss(const SearchProblem& problem, Objective objective,
                                                std::span<const double> mu, std::span<const double> b);

}  // namespace nfx::dcm
