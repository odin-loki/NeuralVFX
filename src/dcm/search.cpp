// The DCM mixer search (include/neuralfx/dcm/search.hpp), ported from the owner's CameraDetector,
// cabinlab/src/diffusion/search.cpp. The auc path is the original's, draw for draw and training for training; the
// value-domain path (ValueNet configurations scored by squared error or code length) is new here and branches off it
// only where a ValueNet differs from a MixerNet.
#include <neuralfx/dcm/search.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <format>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>

namespace nfx::dcm {

namespace {

constexpr double kAnneal = 500.0;  // as MixerNetSpec's defaults
constexpr double kApmRate = 0.02;
constexpr double kAvmRate = 0.02;    // the value domain's map and scale net learn at the APM's rate
constexpr double kScaleRate = 0.02;

double nan_value() { return std::numeric_limits<double>::quiet_NaN(); }

// a > b, NaN counting as minus infinity (a configuration without a defined score never wins).
bool better(double a, double b) {
  if (std::isnan(a)) return false;
  return std::isnan(b) || a > b;
}

bool contains(const std::vector<int>& v, int x) { return std::ranges::find(v, x) != v.end(); }

// ROC-AUC of the scores of `rows` (scores[k] belongs to rows[k]); with strata, the size-weighted mean of the AUCs of
// the strata that hold both classes. NaN when none does.
double rows_auc(const SearchProblem& p, std::span<const float> scores, std::span<const std::size_t> rows) {
  std::map<int, std::pair<std::vector<float>, std::vector<int>>> by;
  for (std::size_t k = 0; k < rows.size(); ++k) {
    auto& [s, y] = by[p.stratum.empty() ? 0 : p.stratum[rows[k]]];
    s.push_back(scores[k]);
    y.push_back(static_cast<int>(p.y[rows[k]]));
  }
  double sum = 0.0, weight = 0.0;
  for (const auto& [stratum, sy] : by) {
    const double a = roc_auc(sy.first, sy.second);
    if (std::isnan(a)) continue;
    sum += a * static_cast<double>(sy.first.size());
    weight += static_cast<double>(sy.first.size());
  }
  return weight > 0.0 ? sum / weight : nan_value();
}

// The loss of one row's prediction under a value objective: squared error, or bits.
double row_loss(const SearchProblem& p, Objective objective, std::size_t row, double mu, double b) {
  const double y = p.y[row];
  if (objective == Objective::mse) return (y - mu) * (y - mu);
  return laplace_bits(y, mu, b, p.delta);
}

// Runs fn(i) for i in [0, n) on up to `threads` threads; the first exception stops the work and is rethrown.
void parallel_for(std::size_t n, int threads, const std::function<void(std::size_t)>& fn) {
  std::atomic<std::size_t> next{0};
  std::exception_ptr error;
  std::mutex m;
  const auto work = [&] {
    for (std::size_t i = next.fetch_add(1); i < n; i = next.fetch_add(1)) {
      try {
        fn(i);
      } catch (...) {
        const std::lock_guard lock(m);
        if (!error) error = std::current_exception();
        next = n;
      }
    }
  };
  {
    std::vector<std::jthread> pool;
    const auto t = static_cast<std::size_t>(std::max(1, threads));
    for (std::size_t k = 1; k < std::min(t, n); ++k) pool.emplace_back(work);
    work();
  }
  if (error) std::rethrow_exception(error);
}

void check_config(const SearchProblem& p, const SearchSpace& s, const SearchConfig& c, bool value) {
  const auto in = [](int i, std::size_t n) { return i >= 0 && static_cast<std::size_t>(i) < n; };
  if (c.groups.empty()) throw std::invalid_argument("dcm: search config without an active group");
  for (const int g : c.groups) {
    if (!in(g, p.groups.size())) throw std::invalid_argument(std::format("dcm: search config group {}", g));
  }
  for (const int j : c.mixer_contexts) {
    if (!in(j, p.context_sizes.size())) throw std::invalid_argument(std::format("dcm: search config context {}", j));
  }
  if (c.apm_context != -1 && !in(c.apm_context, p.context_sizes.size())) {
    throw std::invalid_argument(std::format("dcm: search config APM context {}", c.apm_context));
  }
  if (!in(c.lr1, s.lr1.size()) || !in(c.lr2, s.lr2.size()) || !in(c.epochs, s.epochs.size())) {
    throw std::invalid_argument("dcm: search config index outside the search space");
  }
  if (!value && !in(c.apm_weight, s.apm_weight.size())) {
    throw std::invalid_argument("dcm: search config index outside the search space");
  }
  if (value) {
    if (!in(c.avm_weight, s.avm_weight.size()) || !in(c.loss, s.loss.size())) {
      throw std::invalid_argument("dcm: search config index outside the search space");
    }
    if (c.scale_context != -1 && !in(c.scale_context, p.context_sizes.size())) {
      throw std::invalid_argument(std::format("dcm: search config scale context {}", c.scale_context));
    }
  }
}

void canonicalise(SearchConfig& c, const SearchSpace& s, bool value) {
  std::ranges::sort(c.groups);
  c.groups.erase(std::unique(c.groups.begin(), c.groups.end()), c.groups.end());
  std::ranges::sort(c.mixer_contexts);
  c.mixer_contexts.erase(std::unique(c.mixer_contexts.begin(), c.mixer_contexts.end()), c.mixer_contexts.end());
  if (!value) {
    if (s.apm_weight.at(static_cast<std::size_t>(c.apm_weight)) <= 0.0) c.apm_context = -1;
    c.loss = 0;  // the value-domain fields mean nothing here
    c.avm_weight = 0;
    c.scale_context = -1;
  } else {
    if (s.avm_weight.at(static_cast<std::size_t>(c.avm_weight)) <= 0.0) c.apm_context = -1;
    c.apm_weight = 0;  // the APM's grid means nothing here
  }
}

std::string key_of(const SearchConfig& c) {
  std::string k = "g";
  for (const int g : c.groups) k += std::format("{},", g);
  k += "|m";
  for (const int j : c.mixer_contexts) k += std::format("{},", j);
  return k + std::format("|a{}|{},{},{},{}|v{},{},{}", c.apm_context, c.lr1, c.lr2, c.epochs, c.apm_weight, c.loss,
                         c.avm_weight, c.scale_context);
}

std::size_t nearest(const auto& grid, double value) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < grid.size(); ++i) {
    if (std::abs(static_cast<double>(grid[i]) - value) < std::abs(static_cast<double>(grid[best]) - value)) best = i;
  }
  return best;
}

// `n` distinct configurations: the default mixer first, then seeded random draws. The number of active groups (1 to
// all) and of mixer contexts (0 to max_mixer_contexts) is drawn uniformly first, then which ones, so small
// configurations are as likely as large ones however many groups there are; the map's context and the learning
// settings are uniform (under value objectives the AVM weight, the loss and the scale context are drawn after them, in
// place of the APM weight).
// With `admissible`, configurations it rejects (over the cost budget) are skipped, the default mixer included, and up
// to 1000 n draws are made.
std::vector<SearchConfig> sample_configs(const SearchProblem& p, const SearchSpace& s, int n, std::uint64_t seed,
                                         Objective objective,
                                         const std::function<bool(const SearchConfig&)>& admissible = {}) {
  const bool value = value_objective(objective);
  std::vector<SearchConfig> out;
  std::set<std::string> seen;
  const auto add = [&](SearchConfig c) {
    canonicalise(c, s, value);
    if (admissible && !admissible(c)) return;
    if (seen.insert(key_of(c)).second) out.push_back(std::move(c));
  };
  add(default_config(p, s, objective));
  std::mt19937_64 rng(seed);
  const auto pick = [&](std::size_t size) { return std::uniform_int_distribution<int>(0, static_cast<int>(size) - 1)(rng); };
  const auto subset = [&](std::size_t of, std::size_t count) {
    std::vector<int> all(of);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(count);
    return all;
  };
  const std::size_t n_ctx = p.context_sizes.size();
  const auto max_ctx = std::min(n_ctx, static_cast<std::size_t>(std::max(0, s.max_mixer_contexts)));
  const int attempts = (admissible ? 1000 : 50) * n;
  for (int attempt = 0; static_cast<int>(out.size()) < n && attempt < attempts; ++attempt) {
    SearchConfig c;
    c.groups = subset(p.groups.size(), static_cast<std::size_t>(1 + pick(p.groups.size())));
    c.mixer_contexts = subset(n_ctx, static_cast<std::size_t>(pick(max_ctx + 1)));
    c.apm_context = pick(n_ctx + 1) - 1;
    c.lr1 = pick(s.lr1.size());
    c.lr2 = pick(s.lr2.size());
    c.epochs = pick(s.epochs.size());
    if (!value) {
      c.apm_weight = pick(s.apm_weight.size());
    } else {
      c.avm_weight = pick(s.avm_weight.size());
      c.loss = pick(s.loss.size());
      c.scale_context = pick(n_ctx + 1) - 1;
    }
    add(std::move(c));
  }
  return out;
}

// A configuration compiled for training: its input columns, its contexts and (value objectives) its scale features
// gathered per row.
struct View {
  bool value = false;
  std::size_t inputs = 0, slots = 0, nz = 0;
  std::vector<double> x;  // rows x inputs
  std::vector<int> ctx;   // rows x slots: the first-layer mixers, the final mixer, the APM (or the AVM and the scale net)
  std::vector<double> z;  // rows x nz
  MixerNetSpec spec;
  ValueNetSpec vspec;
  int epochs = 1;
  [[nodiscard]] std::span<const double> row(std::size_t i) const { return {x.data() + i * inputs, inputs}; }
  [[nodiscard]] std::span<const int> contexts(std::size_t i) const { return {ctx.data() + i * slots, slots}; }
  [[nodiscard]] std::span<const double> zrow(std::size_t i) const { return {z.data() + i * nz, nz}; }
};

View compile(const SearchProblem& p, const SearchSpace& s, const SearchConfig& c, Objective objective) {
  View v;
  v.value = value_objective(objective);
  if (v.value) {
    v.vspec = value_spec(p, s, c);
  } else {
    v.spec = mixer_spec(p, s, c);
  }
  v.epochs = s.epochs[static_cast<std::size_t>(c.epochs)];
  std::vector<int> cols;
  for (const int g : c.groups) {
    const auto& gc = p.groups[static_cast<std::size_t>(g)].columns;
    cols.insert(cols.end(), gc.begin(), gc.end());
  }
  std::ranges::sort(cols);
  cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
  const std::size_t n = p.y.size();
  const std::size_t layer1 = c.mixer_contexts.size() + 1;
  v.inputs = cols.size();
  v.slots = v.value ? layer1 + 3 : layer1 + 2;
  v.x.resize(n * v.inputs);
  v.ctx.assign(n * v.slots, 0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < cols.size(); ++k) v.x[i * v.inputs + k] = p.x[i][static_cast<std::size_t>(cols[k])];
    int* ctx = v.ctx.data() + i * v.slots;
    for (std::size_t m = 0; m < c.mixer_contexts.size(); ++m) {
      ctx[m + 1] = p.contexts[i][static_cast<std::size_t>(c.mixer_contexts[m])];
    }
    if (!v.value) {
      if (c.apm_context >= 0) ctx[v.slots - 1] = p.contexts[i][static_cast<std::size_t>(c.apm_context)];
    } else {
      if (c.apm_context >= 0) ctx[layer1 + 1] = p.contexts[i][static_cast<std::size_t>(c.apm_context)];
      if (c.scale_context >= 0) ctx[layer1 + 2] = p.contexts[i][static_cast<std::size_t>(c.scale_context)];
    }
  }
  if (v.value && !p.z.empty()) {
    v.nz = p.z.front().size();
    v.z.resize(n * v.nz);
    for (std::size_t i = 0; i < n; ++i) std::ranges::copy(p.z[i], v.z.begin() + static_cast<std::ptrdiff_t>(i * v.nz));
  }
  return v;
}

// Seeded shuffled passes, the learning rates halved after each, then frozen.
MixerNet train_view(const View& v, std::span<const double> y, std::vector<std::size_t> rows, std::uint64_t seed) {
  MixerNet net(static_cast<int>(v.inputs), v.spec);
  std::mt19937_64 rng(seed);
  for (int e = 0; e < v.epochs; ++e) {
    std::shuffle(rows.begin(), rows.end(), rng);
    for (const auto i : rows) {
      (void)net.predict(v.row(i), v.contexts(i));
      net.update(static_cast<int>(y[i]));
    }
    net.scale_lr(0.5);
  }
  net.freeze();
  return net;
}

// The value-domain counterpart. The AVM's range and the starting scale come from the training rows' targets only.
ValueNet train_value_view(const View& v, std::span<const double> y, std::vector<std::size_t> rows, std::uint64_t seed) {
  ValueNetSpec spec = v.vspec;
  if (!rows.empty()) {
    double lo = std::numeric_limits<double>::infinity(), hi = -lo, mean = 0.0;
    for (const auto i : rows) {
      lo = std::min(lo, y[i]);
      hi = std::max(hi, y[i]);
      mean += y[i];
    }
    mean /= static_cast<double>(rows.size());
    double dev = 0.0;
    for (const auto i : rows) dev += std::abs(y[i] - mean);
    dev /= static_cast<double>(rows.size());
    spec.avm_lo = hi > lo ? lo : lo - 0.5;
    spec.avm_hi = hi > lo ? hi : lo + 0.5;
    spec.log_b_init = std::clamp(std::log(std::max(dev, 1e-6)), spec.log_b_min, spec.log_b_max);
  }
  ValueNet net(static_cast<int>(v.inputs), static_cast<int>(v.nz), spec);
  std::mt19937_64 rng(seed);
  for (int e = 0; e < v.epochs; ++e) {
    std::shuffle(rows.begin(), rows.end(), rng);
    for (const auto i : rows) {
      (void)net.predict(v.row(i), v.contexts(i), v.zrow(i));
      net.update(y[i]);
    }
    net.scale_lr(0.5);
  }
  net.freeze();
  return net;
}

enum class Stage { Nested, Global, RfOnly };

std::string_view stage_name(Stage s) {
  switch (s) {
    case Stage::Nested: return "nested";
    case Stage::Global: return "global";
    case Stage::RfOnly: return "rfonly";
  }
  return "?";
}

// What a training leaves out: whole sites, one fold of the training pool, or every test site.
struct Exclusion {
  std::vector<int> sites;  // ascending
  int fold = -1;
  bool pool_only = false;
};

struct Fit {
  std::vector<float> logits;   // auc: of the scored rows, in order
  std::vector<double> mu, b;   // value objectives: of the scored rows, in order
  std::optional<MixerNet> net;
  std::optional<ValueNet> vnet;
  [[nodiscard]] std::string version() const { return net ? net->version() : vnet->version(); }
  [[nodiscard]] std::string serialise() const { return net ? net->serialise() : vnet->serialise(); }
};

// Search scores of one configuration under one protocol.
struct Entry {
  SearchConfig config;
  std::size_t order = 0;       // creation order (ties go to the earlier configuration)
  std::vector<double> scores;  // nested: S x S, scores[a * S + b] = test site a by the model trained without a and b;
                               // global: S, test site a by the model without a; rfonly: one per training-pool fold
  std::vector<char> done;      // per job
};

// The shared machinery: training rows under an exclusion (with the leak check), the jobs of each protocol.
class Engine {
 public:
  Engine(const SearchProblem& p, const SearchOptions& o) : p_(p), o_(o), site_rows_(p.site_names.size()) {
    for (std::size_t i = 0; i < p.y.size(); ++i) site_rows_[static_cast<std::size_t>(p.site[i])].push_back(i);
    for (std::size_t s = 0; s < p.site_names.size(); ++s) {
      if (p.holdout[s] && !site_rows_[s].empty()) holdout_.push_back(static_cast<int>(s));
    }
    std::map<int, std::vector<std::size_t>> by_fold;
    for (std::size_t i = 0; i < p.y.size(); ++i) {
      if (!p.holdout[static_cast<std::size_t>(p.site[i])] && p.fold[i] >= 0) by_fold[p.fold[i]].push_back(i);
    }
    for (auto& [f, rows] : by_fold) {
      folds_.push_back(f);
      fold_rows_.push_back(std::move(rows));
    }
    for (std::size_t a = 0; a < holdout_.size(); ++a) {
      for (std::size_t b = a + 1; b < holdout_.size(); ++b) pairs_.emplace_back(a, b);
    }
  }

  [[nodiscard]] const SearchProblem& problem() const noexcept { return p_; }
  [[nodiscard]] const SearchOptions& options() const noexcept { return o_; }
  [[nodiscard]] const std::vector<int>& holdout() const noexcept { return holdout_; }
  [[nodiscard]] std::size_t folds() const noexcept { return folds_.size(); }
  [[nodiscard]] const std::vector<std::size_t>& rows_of_site(int s) const { return site_rows_[static_cast<std::size_t>(s)]; }
  [[nodiscard]] std::int64_t trainings() const noexcept { return trainings_.load(); }

  // Pair index of test-site positions a != b.
  [[nodiscard]] int pair_index(std::size_t a, std::size_t b) const {
    if (a > b) std::swap(a, b);
    const std::size_t s = holdout_.size();
    return static_cast<int>(a * (2 * s - a - 1) / 2 + (b - a - 1));
  }

  [[nodiscard]] std::size_t jobs(Stage stage) const {
    switch (stage) {
      case Stage::Nested: return pairs_.size();
      case Stage::Global: return holdout_.size();
      case Stage::RfOnly: return folds_.size();
    }
    return 0;
  }

  [[nodiscard]] std::size_t score_slots(Stage stage) const {
    return stage == Stage::Nested ? holdout_.size() * holdout_.size() : jobs(stage);
  }

  // Trains `v` on every row the exclusion leaves in and returns its predictions of `score`. The training rows are taken
  // site by site in site order, each site's in row order. The leak check: no scored row may come from a site the model
  // was trained on (site protocols) or lie outside the held-out fold.
  Fit fit_score(const View& v, Stage stage, const Exclusion& ex, std::span<const std::size_t> score) {
    const std::size_t n_sites = p_.site_names.size();
    std::vector<std::size_t> rows;
    std::vector<char> trained(n_sites, 0), scored(n_sites, 0);
    for (std::size_t s = 0; s < n_sites; ++s) {
      if (contains(ex.sites, static_cast<int>(s)) || (ex.pool_only && p_.holdout[s])) continue;
      for (const auto i : site_rows_[s]) {
        if (ex.fold >= 0 && p_.fold[i] == ex.fold) continue;
        rows.push_back(i);
        trained[s] = 1;
      }
    }
    for (const auto i : score) {
      const auto s = static_cast<std::size_t>(p_.site[i]);
      scored[s] = 1;
      const bool honest = ex.fold >= 0 ? (p_.fold[i] == ex.fold && !p_.holdout[s]) : !trained[s];
      if (!honest) {
        throw std::logic_error(std::format("dcm: search leak: a {} model trained on site {} scores it",
                                           stage_name(stage), p_.site_names[s]));
      }
    }
    Fit fit;
    if (!v.value) {
      fit.net.emplace(train_view(v, p_.y, rows, o_.seed));
      ++trainings_;
      fit.logits.reserve(score.size());
      for (const auto i : score) fit.logits.push_back(static_cast<float>(stretch(fit.net->predict(v.row(i), v.contexts(i)))));
    } else {
      fit.vnet.emplace(train_value_view(v, p_.y, rows, o_.seed));
      ++trainings_;
      fit.mu.reserve(score.size());
      fit.b.reserve(score.size());
      for (const auto i : score) {
        const auto pr = fit.vnet->predict(v.row(i), v.contexts(i), v.zrow(i));
        fit.mu.push_back(pr.mu);
        fit.b.push_back(pr.b);
      }
    }
    if (o_.on_train) {
      TrainRecord r{stage_name(stage), ex.sites, ex.fold, {}, {}};
      for (std::size_t s = 0; s < n_sites; ++s) {
        if (trained[s]) r.trained_sites.push_back(static_cast<int>(s));
        if (scored[s]) r.scored_sites.push_back(static_cast<int>(s));
      }
      const std::lock_guard lock(record_mutex_);
      o_.on_train(r);
    }
    return fit;
  }

  // Runs one job of a protocol for a compiled configuration and stores its per-site scores in `e`.
  void run_job(const View& v, Stage stage, std::size_t job, Entry& e) {
    switch (stage) {
      case Stage::Nested: {
        const auto [a, b] = pairs_[job];
        const auto& ra = rows_of_site(holdout_[a]);
        const auto& rb = rows_of_site(holdout_[b]);
        std::vector<std::size_t> rows = ra;
        rows.insert(rows.end(), rb.begin(), rb.end());
        const auto fit = fit_score(v, stage, {{holdout_[a], holdout_[b]}}, rows);
        const std::size_t n_hold = holdout_.size();
        e.scores[a * n_hold + b] = score_of(fit, 0, ra);
        e.scores[b * n_hold + a] = score_of(fit, ra.size(), rb);
        break;
      }
      case Stage::Global: {
        const auto& ra = rows_of_site(holdout_[job]);
        e.scores[job] = score_of(fit_score(v, stage, {{holdout_[job]}}, ra), 0, ra);
        break;
      }
      case Stage::RfOnly: {
        const auto& rf = fold_rows_[job];
        e.scores[job] = score_of(fit_score(v, stage, {{}, folds_[job], true}, rf), 0, rf);
        break;
      }
    }
    e.done[job] = 1;
  }

 private:
  // The score of the predictions fit made for `rows` (starting at `offset` of its scored rows): the ROC-AUC under auc,
  // the negated mean loss under value objectives.
  [[nodiscard]] double score_of(const Fit& fit, std::size_t offset, std::span<const std::size_t> rows) const {
    if (fit.net) return rows_auc(p_, std::span<const float>(fit.logits).subspan(offset, rows.size()), rows);
    if (rows.empty()) return nan_value();
    double sum = 0.0;
    for (std::size_t k = 0; k < rows.size(); ++k) sum += row_loss(p_, o_.objective, rows[k], fit.mu[offset + k], fit.b[offset + k]);
    return -(sum / static_cast<double>(rows.size()));
  }

  const SearchProblem& p_;
  const SearchOptions& o_;
  std::vector<std::vector<std::size_t>> site_rows_;
  std::vector<int> holdout_;  // test sites with rows, ascending
  std::vector<int> folds_;    // distinct training-pool folds, ascending
  std::vector<std::vector<std::size_t>> fold_rows_;
  std::vector<std::pair<std::size_t, std::size_t>> pairs_;
  std::atomic<std::int64_t> trainings_{0};
  std::mutex record_mutex_;
};

// One protocol's search: a cache of evaluated configurations and its objectives (one per held-out test site for
// nested, a single one otherwise), each a mean of per-site scores over its jobs.
class ProtocolSearch {
 public:
  ProtocolSearch(Engine& engine, Stage stage) : e_(engine), stage_(stage) {}

  [[nodiscard]] std::size_t objectives() const { return stage_ == Stage::Nested ? e_.holdout().size() : 1; }

  // The jobs objective `o` needs.
  [[nodiscard]] std::vector<int> jobs_of(std::size_t o) const {
    std::vector<int> out;
    if (stage_ == Stage::Nested) {
      for (std::size_t b = 0; b < e_.holdout().size(); ++b) {
        if (b != o) out.push_back(e_.pair_index(o, b));
      }
    } else {
      for (std::size_t j = 0; j < e_.jobs(stage_); ++j) out.push_back(static_cast<int>(j));
    }
    return out;
  }

  // Objective `o` of an entry: nested, the mean over the other test sites b of the score of b by the model trained
  // without o and b; otherwise the mean of every job's score. NaN when no score is defined. `count`: scores averaged.
  [[nodiscard]] double value(const Entry& en, std::size_t o, int* count = nullptr) const {
    double sum = 0.0;
    int n = 0;
    const std::size_t n_hold = e_.holdout().size();
    const auto add = [&](double a) {
      if (!std::isnan(a)) {
        sum += a;
        ++n;
      }
    };
    if (stage_ == Stage::Nested) {
      for (std::size_t b = 0; b < n_hold; ++b) {
        if (b != o) add(en.scores[b * n_hold + o]);
      }
    } else {
      for (const double a : en.scores) add(a);
    }
    if (count) *count = n;
    return n ? sum / n : nan_value();
  }

  // The selection score: value(), minus penalty x cost when the options set a cost penalty.
  [[nodiscard]] double score(const Entry& en, std::size_t o) const {
    const auto& cost = e_.options().cost;
    const double v = value(en, o);
    return cost.penalty > 0.0 ? v - cost.penalty * config_cost(cost, en.config) : v;
  }

  // Within the cost budget (always, without a budget).
  [[nodiscard]] bool admissible(const SearchConfig& c) const {
    const auto& cost = e_.options().cost;
    return !std::isfinite(cost.budget_ms) || config_cost(cost, c) <= cost.budget_ms + 1e-12;
  }

  Entry& entry(const SearchConfig& c) {
    const auto [it, created] = cache_.try_emplace(key_of(c));
    if (created) {
      it->second.config = c;
      it->second.order = cache_.size();
      it->second.scores.assign(e_.score_slots(stage_), nan_value());
      it->second.done.assign(e_.jobs(stage_), 0);
    }
    return it->second;
  }

  // Runs the requested jobs that are not done yet, configurations in parallel.
  void evaluate(const std::vector<std::pair<SearchConfig, std::vector<int>>>& requests) {
    std::map<Entry*, std::set<int>> want;
    for (const auto& [c, jobs] : requests) {
      auto& en = entry(c);
      for (const int j : jobs) {
        if (!en.done[static_cast<std::size_t>(j)]) want[&en].insert(j);
      }
    }
    std::vector<std::pair<Entry*, std::vector<int>>> tasks;
    for (const auto& [en, jobs] : want) tasks.emplace_back(en, std::vector<int>(jobs.begin(), jobs.end()));
    std::ranges::sort(tasks, {}, [](const auto& t) { return t.first->order; });
    parallel_for(tasks.size(), e_.options().threads, [&](std::size_t t) {
      auto& [en, jobs] = tasks[t];
      const auto v = compile(e_.problem(), e_.options().space, en->config, e_.options().objective);
      for (const int j : jobs) e_.run_job(v, stage_, static_cast<std::size_t>(j), *en);
    });
  }

  // Random candidates, then hill-climbing per objective. Returns the best configuration of each objective.
  std::vector<ScoredConfig> run(const std::vector<SearchConfig>& candidates) {
    const std::size_t n_obj = objectives();
    std::vector<int> all_jobs;
    for (std::size_t j = 0; j < e_.jobs(stage_); ++j) all_jobs.push_back(static_cast<int>(j));
    std::vector<std::pair<SearchConfig, std::vector<int>>> requests;
    for (const auto& c : candidates) requests.emplace_back(c, all_jobs);
    evaluate(requests);
    std::vector<ScoredConfig> best(n_obj, {candidates.front(), nan_value()});
    for (std::size_t o = 0; o < n_obj; ++o) {
      for (const auto& c : candidates) {
        const double v = score(entry(c), o);
        if (better(v, best[o].score)) best[o] = {c, v};
      }
    }
    const auto& space = e_.options().space;
    for (int round = 0; round < e_.options().refine_rounds; ++round) {
      std::vector<std::vector<SearchConfig>> nbrs(n_obj);
      requests.clear();
      for (std::size_t o = 0; o < n_obj; ++o) {
        nbrs[o] = neighbours(e_.problem(), space, best[o].config, e_.options().objective);
        std::erase_if(nbrs[o], [&](const SearchConfig& nb) { return !admissible(nb); });
        const auto jobs = jobs_of(o);
        for (const auto& nb : nbrs[o]) requests.emplace_back(nb, jobs);
      }
      evaluate(requests);
      bool moved = false;
      for (std::size_t o = 0; o < n_obj; ++o) {
        for (const auto& nb : nbrs[o]) {
          const double v = score(entry(nb), o);
          if (better(v, best[o].score)) {
            best[o] = {nb, v};
            moved = true;
          }
        }
      }
      if (!moved) break;
    }
    return best;
  }

  // Fully evaluated configurations by decreasing score (ties: the earlier one first), at most `n`.
  [[nodiscard]] std::vector<ScoredConfig> top(std::size_t n) const {
    std::vector<const Entry*> done;
    for (const auto& [k, en] : cache_) {
      if (std::ranges::all_of(en.done, [](char d) { return d != 0; })) done.push_back(&en);
    }
    std::ranges::sort(done, [&](const Entry* a, const Entry* b) {
      const double va = score(*a, 0), vb = score(*b, 0);
      if (better(va, vb)) return true;
      if (better(vb, va)) return false;
      return a->order < b->order;
    });
    std::vector<ScoredConfig> out;
    for (std::size_t i = 0; i < std::min(n, done.size()); ++i) out.push_back({done[i]->config, score(*done[i], 0)});
    return out;
  }

 private:
  Engine& e_;
  Stage stage_;
  std::map<std::string, Entry> cache_;
};

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

std::string_view objective_name(Objective objective) {
  switch (objective) {
    case Objective::auc: return "auc";
    case Objective::mse: return "mse";
    case Objective::laplace_bits: return "laplace_bits";
  }
  return "?";
}

void validate(const SearchProblem& p, Objective objective) {
  const std::size_t n = p.y.size(), k = p.input_names.size(), n_ctx = p.context_sizes.size();
  if (n == 0) throw std::invalid_argument("dcm: search problem without rows");
  if (p.x.size() != n || p.contexts.size() != n || p.site.size() != n || p.fold.size() != n) {
    throw std::invalid_argument("dcm: search problem needs one input row, context row, site and fold per row");
  }
  if (k == 0) throw std::invalid_argument("dcm: search problem without inputs");
  if (p.groups.empty()) throw std::invalid_argument("dcm: search problem without input groups");
  for (const auto& g : p.groups) {
    if (g.columns.empty()) throw std::invalid_argument(std::format("dcm: search group '{}' has no column", g.name));
    for (const int c : g.columns) {
      if (c < 0 || static_cast<std::size_t>(c) >= k) {
        throw std::invalid_argument(std::format("dcm: search group '{}' names column {} of {}", g.name, c, k));
      }
    }
  }
  if (p.context_names.size() != n_ctx) throw std::invalid_argument("dcm: search problem: one name per context");
  for (const int s : p.context_sizes) {
    if (s < 1) throw std::invalid_argument("dcm: search context with cardinality below 1");
  }
  if (p.holdout.size() != p.site_names.size()) throw std::invalid_argument("dcm: search problem: holdout per site");
  if (!p.stratum.empty() && (p.stratum.size() != n || std::ranges::any_of(p.stratum, [](int s) { return s < 0; }))) {
    throw std::invalid_argument("dcm: search strata: none, or one non-negative stratum per row");
  }
  const bool value = value_objective(objective);
  if (value) {
    if (!p.z.empty()) {
      if (p.z.size() != n) throw std::invalid_argument("dcm: search scale features: none, or one row per row");
      for (const auto& zr : p.z) {
        if (zr.size() != p.z.front().size()) throw std::invalid_argument("dcm: search scale features differ in count");
        if (!std::ranges::all_of(zr, [](double v) { return std::isfinite(v); })) {
          throw std::invalid_argument("dcm: search scale features are not finite");
        }
      }
    }
    if (objective == Objective::laplace_bits && !(p.delta > 0.0 && std::isfinite(p.delta))) {
      throw std::invalid_argument("dcm: search problem: laplace_bits needs a finite delta > 0");
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (p.x[i].size() != k) throw std::invalid_argument(std::format("dcm: search input row {} has {} values", i, p.x[i].size()));
    for (const double v : p.x[i]) {
      if (!std::isfinite(v)) throw std::invalid_argument(std::format("dcm: search input row {} is not finite", i));
    }
    if (p.contexts[i].size() != n_ctx) throw std::invalid_argument(std::format("dcm: search context row {} size", i));
    for (std::size_t j = 0; j < n_ctx; ++j) {
      if (p.contexts[i][j] < 0 || p.contexts[i][j] >= p.context_sizes[j]) {
        throw std::invalid_argument(std::format("dcm: search context {} of row {} is {} (cardinality {})",
                                                p.context_names[j], i, p.contexts[i][j], p.context_sizes[j]));
      }
    }
    if (p.site[i] < 0 || static_cast<std::size_t>(p.site[i]) >= p.site_names.size()) {
      throw std::invalid_argument(std::format("dcm: search row {} has site {}", i, p.site[i]));
    }
    if (value ? !std::isfinite(p.y[i]) : (p.y[i] != 0.0 && p.y[i] != 1.0)) {
      throw std::invalid_argument(std::format("dcm: search target of row {}", i));
    }
    if (p.fold[i] < -1 || (p.fold[i] >= 0 && p.holdout[static_cast<std::size_t>(p.site[i])])) {
      throw std::invalid_argument(std::format("dcm: search fold of row {} (test-site rows take -1)", i));
    }
  }
}

double config_cost(const SearchCost& cost, const SearchConfig& config) {
  if (!cost.active()) return cost.base_ms;
  std::vector<char> used(cost.components.size(), 0);
  const auto mark = [&](const std::vector<std::vector<int>>& lists, int i) {
    if (i < 0 || static_cast<std::size_t>(i) >= lists.size()) {
      throw std::invalid_argument(std::format("dcm: config_cost: no component list {}", i));
    }
    for (const int c : lists[static_cast<std::size_t>(i)]) used.at(static_cast<std::size_t>(c)) = 1;
  };
  for (const int g : config.groups) mark(cost.group_components, g);
  for (const int j : config.mixer_contexts) mark(cost.context_components, j);
  if (config.apm_context >= 0) mark(cost.context_components, config.apm_context);
  if (config.scale_context >= 0) mark(cost.context_components, config.scale_context);
  double ms = cost.base_ms;
  for (std::size_t c = 0; c < used.size(); ++c) {
    if (used[c]) ms += cost.ms.at(c);
  }
  return ms;
}

MixerNetSpec mixer_spec(const SearchProblem& problem, const SearchSpace& space, const SearchConfig& config) {
  check_config(problem, space, config, false);
  MixerNetSpec s;
  s.context_sizes = {1};
  for (const int j : config.mixer_contexts) s.context_sizes.push_back(problem.context_sizes[static_cast<std::size_t>(j)]);
  s.final_contexts = 1;
  s.apm_contexts = config.apm_context < 0 ? 1 : problem.context_sizes[static_cast<std::size_t>(config.apm_context)];
  s.lr1 = space.lr1[static_cast<std::size_t>(config.lr1)];
  s.lr2 = space.lr2[static_cast<std::size_t>(config.lr2)];
  s.anneal = kAnneal;
  s.apm_rate = kApmRate;
  s.apm_weight = space.apm_weight[static_cast<std::size_t>(config.apm_weight)];
  return s;
}

ValueNetSpec value_spec(const SearchProblem& problem, const SearchSpace& space, const SearchConfig& config) {
  check_config(problem, space, config, true);
  const auto size_of = [&](int column) { return column < 0 ? 1 : problem.context_sizes[static_cast<std::size_t>(column)]; };
  ValueNetSpec s;
  s.context_sizes = {1};
  for (const int j : config.mixer_contexts) s.context_sizes.push_back(size_of(j));
  s.final_contexts = 1;
  s.avm_contexts = size_of(config.apm_context);
  s.scale_contexts = size_of(config.scale_context);
  s.loss = space.loss[static_cast<std::size_t>(config.loss)];
  s.lr1 = space.lr1[static_cast<std::size_t>(config.lr1)];
  s.lr2 = space.lr2[static_cast<std::size_t>(config.lr2)];
  s.anneal = kAnneal;
  s.avm_rate = kAvmRate;
  s.avm_weight = space.avm_weight[static_cast<std::size_t>(config.avm_weight)];
  s.scale_lr = kScaleRate;
  return s;
}

std::string describe(const SearchProblem& problem, const SearchSpace& space, const SearchConfig& config,
                     Objective objective) {
  const bool value = value_objective(objective);
  check_config(problem, space, config, value);
  std::string groups, mixers = "-";
  for (const int g : config.groups) groups += (groups.empty() ? "" : "+") + problem.groups[static_cast<std::size_t>(g)].name;
  for (const int j : config.mixer_contexts) mixers += "," + problem.context_names[static_cast<std::size_t>(j)];
  const auto name_of = [&](int column) {
    return column < 0 ? std::string("-") : problem.context_names[static_cast<std::size_t>(column)];
  };
  const double w = value ? space.avm_weight[static_cast<std::size_t>(config.avm_weight)]
                         : space.apm_weight[static_cast<std::size_t>(config.apm_weight)];
  std::string map = "off";
  if (w > 0.0) map = std::format("{} {:g}", name_of(config.apm_context), w);
  const double lr1 = space.lr1[static_cast<std::size_t>(config.lr1)], lr2 = space.lr2[static_cast<std::size_t>(config.lr2)];
  const int epochs = space.epochs[static_cast<std::size_t>(config.epochs)];
  if (!value) return std::format("groups {} | mixers {} | apm {} | lr {:g}/{:g} | ep {}", groups, mixers, map, lr1, lr2, epochs);
  return std::format("groups {} | mixers {} | avm {} | scale {} | loss {} | lr {:g}/{:g} | ep {}", groups, mixers, map,
                     name_of(config.scale_context), loss_name(space.loss[static_cast<std::size_t>(config.loss)]), lr1,
                     lr2, epochs);
}

SearchConfig default_config(const SearchProblem& problem, const SearchSpace& space, Objective objective) {
  const bool value = value_objective(objective);
  if (space.lr1.empty() || space.lr2.empty() || space.epochs.empty() || (!value && space.apm_weight.empty()) ||
      (value && (space.avm_weight.empty() || space.loss.empty()))) {
    throw std::invalid_argument("dcm: search space with an empty grid");
  }
  SearchConfig c;
  for (int g = 0; g < static_cast<int>(problem.groups.size()); ++g) c.groups.push_back(g);
  for (int j = 0; j < std::min<int>(space.max_mixer_contexts, static_cast<int>(problem.context_sizes.size())); ++j) {
    c.mixer_contexts.push_back(j);
  }
  c.apm_context = problem.context_sizes.empty() ? -1 : 0;
  c.lr1 = static_cast<int>(nearest(space.lr1, 0.02));
  c.lr2 = static_cast<int>(nearest(space.lr2, 0.01));
  c.epochs = static_cast<int>(nearest(space.epochs, 4.0));
  if (!value) {
    c.apm_weight = static_cast<int>(nearest(space.apm_weight, 0.3));
  } else {
    c.avm_weight = static_cast<int>(nearest(space.avm_weight, 0.3));
    const ValueLoss want = objective == Objective::mse ? ValueLoss::squared : ValueLoss::laplace;
    const auto it = std::ranges::find(space.loss, want);
    c.loss = it == space.loss.end() ? 0 : static_cast<int>(it - space.loss.begin());
    c.scale_context = -1;
  }
  canonicalise(c, space, value);
  return c;
}

std::vector<SearchConfig> neighbours(const SearchProblem& problem, const SearchSpace& space, const SearchConfig& config,
                                     Objective objective) {
  const bool value = value_objective(objective);
  check_config(problem, space, config, value);
  std::vector<SearchConfig> out;
  SearchConfig self = config;
  canonicalise(self, space, value);
  std::set<std::string> seen{key_of(self)};
  const auto push = [&](SearchConfig c) {
    canonicalise(c, space, value);
    if (seen.insert(key_of(c)).second) out.push_back(std::move(c));
  };
  const auto toggle = [](std::vector<int>& v, int x) {
    if (const auto it = std::ranges::find(v, x); it != v.end()) {
      v.erase(it);
    } else {
      v.push_back(x);
    }
  };
  const int n_ctx = static_cast<int>(problem.context_sizes.size());
  for (int g = 0; g < static_cast<int>(problem.groups.size()); ++g) {
    SearchConfig c = self;
    toggle(c.groups, g);
    if (!c.groups.empty()) push(std::move(c));
  }
  for (int j = 0; j < n_ctx; ++j) {
    SearchConfig c = self;
    toggle(c.mixer_contexts, j);
    if (static_cast<int>(c.mixer_contexts.size()) <= space.max_mixer_contexts) push(std::move(c));
  }
  for (const int j : self.mixer_contexts) {
    for (int j2 = 0; j2 < n_ctx; ++j2) {
      if (contains(self.mixer_contexts, j2)) continue;
      SearchConfig c = self;
      *std::ranges::find(c.mixer_contexts, j) = j2;
      push(std::move(c));
    }
  }
  const double map_weight = value ? space.avm_weight[static_cast<std::size_t>(self.avm_weight)]
                                  : space.apm_weight[static_cast<std::size_t>(self.apm_weight)];
  if (map_weight > 0.0) {
    for (int a = -1; a < n_ctx; ++a) {
      SearchConfig c = self;
      c.apm_context = a;
      push(std::move(c));
    }
  }
  if (value) {
    for (int a = -1; a < n_ctx; ++a) {
      SearchConfig c = self;
      c.scale_context = a;
      push(std::move(c));
    }
  }
  const auto step = [&](int SearchConfig::*field, std::size_t size) {
    for (const int d : {-1, 1}) {
      SearchConfig c = self;
      c.*field += d;
      if (c.*field >= 0 && static_cast<std::size_t>(c.*field) < size) push(std::move(c));
    }
  };
  step(&SearchConfig::lr1, space.lr1.size());
  step(&SearchConfig::lr2, space.lr2.size());
  step(&SearchConfig::epochs, space.epochs.size());
  if (!value) {
    step(&SearchConfig::apm_weight, space.apm_weight.size());
  } else {
    step(&SearchConfig::avm_weight, space.avm_weight.size());
    step(&SearchConfig::loss, space.loss.size());
  }
  return out;
}

MixerNet train_config(const SearchProblem& problem, const SearchSpace& space, const SearchConfig& config,
                      std::span<const std::size_t> rows, std::uint64_t seed) {
  return train_view(compile(problem, space, config, Objective::auc), problem.y, {rows.begin(), rows.end()}, seed);
}

ValueNet train_value_config(const SearchProblem& problem, const SearchSpace& space, const SearchConfig& config,
                            std::span<const std::size_t> rows, std::uint64_t seed) {
  return train_value_view(compile(problem, space, config, Objective::mse), problem.y, {rows.begin(), rows.end()}, seed);
}

SearchResult run_search(const SearchProblem& problem, const SearchOptions& options) {
  validate(problem, options.objective);
  if (options.configs < 1 || options.threads < 1 || options.top < 1 || options.refine_rounds < 0) {
    throw std::invalid_argument("dcm: search needs configs, threads and top >= 1 and refine_rounds >= 0");
  }
  Engine engine(problem, options);
  const auto& hold = engine.holdout();
  if (options.nested && hold.size() < 2) throw std::invalid_argument("dcm: the nested search needs two test sites");
  if (options.global && hold.empty()) throw std::invalid_argument("dcm: the global search needs a test site");
  if (options.rfonly && engine.folds() < 2) {
    throw std::invalid_argument("dcm: the rfonly search needs two folds in the training pool");
  }
  const auto& cost = options.cost;
  if (cost.active()) {
    const auto bad = [](std::string_view what) {
      throw std::invalid_argument(std::format("dcm: search cost: {}", what));
    };
    if (cost.ms.size() != cost.components.size()) bad("one cost per component");
    for (const double ms : cost.ms) {
      if (!(ms >= 0.0) || !std::isfinite(ms)) bad("component costs are finite and >= 0");
    }
    if (cost.group_components.size() != problem.groups.size()) bad("one component list per group");
    if (cost.context_components.size() != problem.context_sizes.size()) bad("one component list per context column");
    for (const auto* lists : {&cost.group_components, &cost.context_components}) {
      for (const auto& l : *lists) {
        for (const int c : l) {
          if (c < 0 || static_cast<std::size_t>(c) >= cost.components.size()) bad("component index out of range");
        }
      }
    }
  }
  if (!(cost.penalty >= 0.0) || std::isnan(cost.budget_ms)) {
    throw std::invalid_argument("dcm: search cost: penalty >= 0 and a budget that is a number");
  }
  std::function<bool(const SearchConfig&)> admissible;
  if (std::isfinite(cost.budget_ms)) {
    admissible = [&](const SearchConfig& c) { return config_cost(cost, c) <= cost.budget_ms + 1e-12; };
  }
  const auto candidates = sample_configs(problem, options.space, options.configs, options.seed, options.objective, admissible);
  if (candidates.empty()) throw std::invalid_argument("dcm: no configuration within the search's cost budget");
  SearchResult r;
  r.candidates = static_cast<int>(candidates.size());
  const std::size_t n = problem.y.size();
  const auto& space = options.space;
  const bool value = value_objective(options.objective);
  // Writes a fit's predictions of `rows` into a protocol's per-row results.
  const auto store = [&](const Fit& fit, std::span<const std::size_t> rows, std::vector<double>& out, std::vector<double>& out_b) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (value) {
        out[rows[i]] = fit.mu[i];
        out_b[rows[i]] = fit.b[i];
      } else {
        out[rows[i]] = fit.logits[i];
      }
    }
  };
  const auto start = [&](std::vector<double>& out, std::vector<double>& out_b) {
    out.assign(n, nan_value());
    if (value) out_b.assign(n, nan_value());
  };

  if (options.nested) {
    const auto t0 = std::chrono::steady_clock::now();
    ProtocolSearch search(engine, Stage::Nested);
    const auto best = search.run(candidates);
    start(r.nested, r.nested_b);
    for (std::size_t o = 0; o < hold.size(); ++o) {
      int count = 0;
      (void)search.value(search.entry(best[o].config), o, &count);
      r.nested_choices.push_back({hold[o], best[o].config, best[o].score, count});
    }
    if (options.nested_teachers) r.nested_teachers.assign(hold.size(), std::vector<double>(n, nan_value()));
    parallel_for(hold.size(), options.threads, [&](std::size_t o) {
      const auto& rows = engine.rows_of_site(hold[o]);
      const auto v = compile(problem, space, best[o].config, options.objective);
      auto fit = engine.fit_score(v, Stage::Nested, {{hold[o]}}, rows);
      store(fit, rows, r.nested, r.nested_b);
      if (options.nested_teachers) {
        // the same frozen mixer on every row, rounded as fit_score rounds its logits (equal on the site's own rows)
        auto& teacher = r.nested_teachers[o];
        for (std::size_t i = 0; i < n; ++i) {
          teacher[i] = value ? fit.vnet->predict(v.row(i), v.contexts(i), v.zrow(i)).mu
                             : static_cast<float>(stretch(fit.net->predict(v.row(i), v.contexts(i))));
        }
      }
    });
    r.nested_seconds = seconds_since(t0);
  }

  if (options.global) {
    const auto t0 = std::chrono::steady_clock::now();
    ProtocolSearch search(engine, Stage::Global);
    const auto best = search.run(candidates).front();
    r.global_best = best;
    r.global_top = search.top(static_cast<std::size_t>(options.top));
    start(r.global, r.global_b);
    const auto v = compile(problem, space, best.config, options.objective);
    parallel_for(hold.size() + 1, options.threads, [&](std::size_t o) {
      if (o == hold.size()) {  // every row: the deployable frozen mixer
        const auto fit = engine.fit_score(v, Stage::Global, {}, {});
        r.global_version = fit.version();
        r.global_mixer = fit.serialise();
        return;
      }
      const auto& rows = engine.rows_of_site(hold[o]);
      store(engine.fit_score(v, Stage::Global, {{hold[o]}}, rows), rows, r.global, r.global_b);
    });
    r.global_seconds = seconds_since(t0);
  }

  if (options.rfonly) {
    const auto t0 = std::chrono::steady_clock::now();
    ProtocolSearch search(engine, Stage::RfOnly);
    r.rfonly_best = search.run(candidates).front();
    start(r.rfonly, r.rfonly_b);
    std::vector<std::size_t> rows;
    for (const int s : hold) {
      const auto& sr = engine.rows_of_site(s);
      rows.insert(rows.end(), sr.begin(), sr.end());
    }
    const auto fit = engine.fit_score(compile(problem, space, r.rfonly_best.config, options.objective), Stage::RfOnly,
                                      {{}, -1, true}, rows);
    store(fit, rows, r.rfonly, r.rfonly_b);
    r.rfonly_version = fit.version();
    r.rfonly_mixer = fit.serialise();
    r.rfonly_seconds = seconds_since(t0);
  }
  r.trainings = engine.trainings();
  return r;
}

std::vector<double> per_site_auc(const SearchProblem& problem, std::span<const double> scores) {
  if (scores.size() != problem.y.size()) throw std::invalid_argument("dcm: per_site_auc: one score per row");
  std::vector<std::vector<float>> s(problem.site_names.size());
  std::vector<std::vector<std::size_t>> rows(problem.site_names.size());
  for (std::size_t i = 0; i < scores.size(); ++i) {
    if (std::isnan(scores[i])) continue;
    const auto site = static_cast<std::size_t>(problem.site.at(i));
    s.at(site).push_back(static_cast<float>(scores[i]));
    rows[site].push_back(i);
  }
  std::vector<double> out;
  for (std::size_t k = 0; k < s.size(); ++k) out.push_back(rows_auc(problem, s[k], rows[k]));
  return out;
}

std::vector<double> per_site_loss(const SearchProblem& problem, Objective objective, std::span<const double> mu,
                                  std::span<const double> b) {
  if (!value_objective(objective)) throw std::invalid_argument("dcm: per_site_loss needs a value objective");
  if (mu.size() != problem.y.size() || (objective == Objective::laplace_bits && b.size() != mu.size())) {
    throw std::invalid_argument("dcm: per_site_loss: one prediction (and scale) per row");
  }
  std::vector<double> sum(problem.site_names.size(), 0.0);
  std::vector<std::size_t> count(problem.site_names.size(), 0);
  for (std::size_t i = 0; i < mu.size(); ++i) {
    if (std::isnan(mu[i])) continue;
    const auto site = static_cast<std::size_t>(problem.site.at(i));
    sum.at(site) += row_loss(problem, objective, i, mu[i], b.empty() ? 1.0 : b[i]);
    ++count[site];
  }
  std::vector<double> out;
  for (std::size_t k = 0; k < sum.size(); ++k) out.push_back(count[k] ? sum[k] / static_cast<double>(count[k]) : nan_value());
  return out;
}

}  // namespace nfx::dcm
