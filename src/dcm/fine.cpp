// DCM-fine (include/neuralfx/dcm/fine.hpp): the reference of the mixer in place of the rollout detail layer's lock.
// compute_frame repeats rollout::detail_step operation for operation (so its lock output L is v1's fine field) and
// keeps every intermediate the experts are made of.
#include <neuralfx/dcm/fine.hpp>
#include <neuralfx/noise.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <limits>
#include <mutex>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>

namespace nfx::dcm::fine {

namespace {

std::size_t sz(int v) { return static_cast<std::size_t>(v); }
float fl(int v) { return static_cast<float>(v); }

// --- the reference's interpolation helpers (src/core/rollout.cpp), repeated so the arithmetic is the same -----------

float bilinear(const float* f, int n, float x, float y) {
  x = std::clamp(x, 0.f, fl(n - 1));
  y = std::clamp(y, 0.f, fl(n - 1));
  const int x0 = std::min(static_cast<int>(x), n - 2), y0 = std::min(static_cast<int>(y), n - 2);
  const float fx = x - fl(x0), fy = y - fl(y0);
  const float* a = f + sz(y0) * sz(n) + sz(x0);
  return (1.f - fy) * ((1.f - fx) * a[0] + fx * a[1]) + fy * ((1.f - fx) * a[n] + fx * a[n + 1]);
}

float bilinear_ch(const float* f, int n, int channels, int c, float x, float y) {
  x = std::clamp(x, 0.f, fl(n - 1));
  y = std::clamp(y, 0.f, fl(n - 1));
  const int x0 = std::min(static_cast<int>(x), n - 2), y0 = std::min(static_cast<int>(y), n - 2);
  const float fx = x - fl(x0), fy = y - fl(y0);
  const auto at = [&](int xx, int yy) { return f[(sz(yy) * sz(n) + sz(xx)) * sz(channels) + sz(c)]; };
  return (1.f - fy) * ((1.f - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1.f - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
}

float bilinear_zero(const float* f, int n, float x, float y, float* lo = nullptr, float* hi = nullptr) {
  x = std::clamp(x, -1.f, fl(n));
  y = std::clamp(y, -1.f, fl(n));
  const float fx0 = std::floor(x), fy0 = std::floor(y);
  const int x0 = static_cast<int>(fx0), y0 = static_cast<int>(fy0);
  const float fx = x - fx0, fy = y - fy0;
  const auto at = [&](int xx, int yy) { return (xx < 0 || yy < 0 || xx >= n || yy >= n) ? 0.f : f[sz(yy) * sz(n) + sz(xx)]; };
  const float a = at(x0, y0), b = at(x0 + 1, y0), c = at(x0, y0 + 1), d = at(x0 + 1, y0 + 1);
  if (lo) *lo = std::min(std::min(a, b), std::min(c, d));
  if (hi) *hi = std::max(std::max(a, b), std::max(c, d));
  return (1.f - fy) * ((1.f - fx) * a + fx * b) + fy * ((1.f - fx) * c + fx * d);
}

float smoothstep01(float t) {
  t = std::clamp(t, 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}

struct Swirl {
  int n = 0;
  float spacing = 1;
  std::vector<float> u, v;
};

Swirl swirl_field(const rollout::Model& m, std::uint64_t seed, float t) {
  Swirl s;
  const rollout::DetailSpec& d = m.detail;
  s.spacing = 0.5f * d.swirl_scale;
  s.n = static_cast<int>(std::ceil(130.f / s.spacing)) + 3;
  const int n = s.n;
  std::vector<float> psi(sz(n) * sz(n));
  const std::uint64_t sseed = seed * 0x9E3779B97F4A7C15ULL + 5;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) psi[sz(j) * sz(n) + sz(i)] = value_noise(fl(i - 1) * 0.5f, fl(j - 1) * 0.5f, t * d.swirl_rate, sseed);
  }
  s.u.assign(psi.size(), 0.f);
  s.v.assign(psi.size(), 0.f);
  for (int j = 1; j < n - 1; ++j) {
    for (int i = 1; i < n - 1; ++i) {
      const std::size_t k = sz(j) * sz(n) + sz(i);
      s.u[k] = psi[k + sz(n)] - psi[k - sz(n)];
      s.v[k] = psi[k - 1] - psi[k + 1];
    }
  }
  return s;
}

// sign(v) log1p(|v| / s)
double slog(double v, double s) { return v >= 0.0 ? std::log1p(v / s) : -std::log1p(-v / s); }

// log of the mass Laplace(mu, b) gives [lo, hi] (lo < hi), without cancellation or underflow.
double log_mass(double lo, double hi, double mu, double b) {
  constexpr double kFloor = -690.7755278982137;  // ln(1e-300), as laplace_bits
  double lm = 0.0;
  if (lo >= mu || hi <= mu) {
    const double near = lo >= mu ? lo - mu : mu - hi;
    lm = -std::numbers::ln2 - near / b + std::log(-std::expm1(-(hi - lo) / b));
  } else {
    const double mass = -0.5 * (std::expm1(-(mu - lo) / b) + std::expm1(-(hi - mu) / b));
    lm = mass > 0.0 ? std::log(mass) : kFloor;
  }
  return std::max(lm, kFloor);
}

constexpr std::uint64_t kGrainSalt = 0xD1B54A32D192ED03ULL;

// The grain fbm's distribution: 257 quantiles of fbm over many points, so a value maps to a uniform u and then to a
// Laplace quantile. Fixed per octave count (deterministic sample).
const std::vector<float>& grain_quantiles(int octaves) {
  static std::mutex m;
  static std::vector<std::vector<float>> tables(9);
  const std::lock_guard lock(m);
  auto& t = tables.at(sz(std::clamp(octaves, 1, 8)));
  if (t.empty()) {
    std::vector<float> v;
    v.reserve(200000);
    for (int i = 0; i < 200000; ++i) {
      const float x = fl(i % 499) * 0.37f, y = fl((i / 499) % 401) * 0.41f, z = fl(i / (499 * 401)) * 0.53f + 0.11f * fl(i % 7);
      v.push_back(fbm(x, y, z, 12345, octaves));
    }
    std::ranges::sort(v);
    for (int k = 0; k <= 256; ++k) t.push_back(v[std::min(v.size() - 1, sz(k) * (v.size() - 1) / 256)]);
  }
  return t;
}

}  // namespace

// --- names -----------------------------------------------------------------------------------------------------------

std::string_view expert_name(int e) {
  static constexpr std::array<std::string_view, kExperts> n{"A",     "A_sl",   "A-A_sl", "L",        "rA",   "a_up", "a_phi1", "a_phi2",
                                                            "a_phi3", "C_up", "bres",   "lap", "grad", "prev", "bias"};
  return e >= 0 && e < kExperts ? n[sz(e)] : "?";
}

const std::vector<ExpertGroup>& expert_groups() {
  static const std::vector<ExpertGroup> g{{"adv", {kAdv, kAdvSl, kAdvDiff}},
                                          {"lock", {kLock, kLockScaled, kNew}},
                                          {"noise", {kNoise1, kNoise2, kNoise3}},
                                          {"coarse", {kCoarse, kBlockRes}},
                                          {"shape", {kLaplace, kGrad}},
                                          {"prev", {kPrev}},
                                          {"bias", {kBias}}};
  return g;
}

std::string_view context_name(int c) {
  static constexpr std::array<std::string_view, kContexts> n{"lvl", "ratio", "flow", "height", "ctrl", "age", "channel", "macro4", "macro8", "extra"};
  return c >= 0 && c < kContexts ? n[sz(c)] : "?";
}

int context_size(const Spec& s, int c) {
  switch (c) {
    case kLvl: return 6;
    case kRatio: return 4;
    case kFlow: return 6;
    case kHeight: return 4;
    case kCtrl: return 9;
    case kAge: return s.one_shot ? 4 : 1;
    case kChannel: return 2;
    case kMacro4: return std::max<int>(1, static_cast<int>(s.macro4.size() / kMacroStats));
    case kMacro8: return std::max<int>(1, static_cast<int>(s.macro8.size() / kMacroStats));
    case kExtra: return std::max(1, s.extra_size);
    default: throw std::out_of_range("dcm fine: context");
  }
}

std::string_view family_name(Family f) {
  switch (f) {
    case Family::none: return "none";
    case Family::hand: return "hand";
    case Family::hand_macro: return "hand+macro";
  }
  return "?";
}

std::vector<int> family_contexts(Family f, bool one_shot) {
  if (f == Family::none) return {};
  std::vector<int> c{kLvl, kRatio, kFlow, kHeight, kCtrl};
  if (one_shot) c.push_back(kAge);
  c.push_back(kChannel);
  if (f == Family::hand_macro) {
    c.push_back(kMacro4);
    c.push_back(kMacro8);
  }
  return c;
}

std::string_view domain_name(Domain d) { return d == Domain::linear ? "linear" : "log"; }

// --- domains, bits ---------------------------------------------------------------------------------------------------

double normalise_value(const Spec& s, int q, double v, double c_up) {
  if (s.domain == Domain::linear) return v / (std::max(0.0, c_up) + static_cast<double>(s.eps[sz(q)]));
  return slog(v, static_cast<double>(s.s[sz(q)]));
}

double value_of(const Spec& s, int q, double mu, double c_up) {
  if (s.domain == Domain::linear) return mu * (std::max(0.0, c_up) + static_cast<double>(s.eps[sz(q)]));
  const double sq = static_cast<double>(s.s[sz(q)]);
  return mu >= 0.0 ? sq * std::expm1(mu) : -sq * std::expm1(-mu);
}

double bin_width(const Spec& s, int q, double v, double c_up) {
  const double h = static_cast<double>(s.s[sz(q)]) / 512.0;
  return normalise_value(s, q, v + h, c_up) - normalise_value(s, q, v - h, c_up);
}

double value_bits(const Spec& s, int q, double v, double mu, double b, double c_up) {
  if (!(b > 0.0)) throw std::invalid_argument("dcm fine: value_bits needs b > 0");
  const double h = static_cast<double>(s.s[sz(q)]) / 512.0;
  const double lo = normalise_value(s, q, v - h, c_up), hi = normalise_value(s, q, v + h, c_up);
  return -log_mass(lo, hi, mu, b) / std::numbers::ln2;
}

void normalise(const Spec& s, const Row& r, std::span<double> x, std::span<double> z) {
  const int q = r.channel;
  const double c = r.e[kCoarse];
  for (int e = 0; e < kRawExperts; ++e) {
    double v = r.e[sz(e)];
    if (e == kGrad) v /= static_cast<double>(s.s[sz(q)]);
    x[sz(e)] = normalise_value(s, q, v, c);
  }
  x[kBias] = 1.0;
  z[0] = normalise_value(s, q, std::abs(static_cast<double>(r.e[kAdvDiff])), c);
  z[1] = normalise_value(s, q, static_cast<double>(r.grad), c);
  z[2] = normalise_value(s, q, c, c);
  z[3] = normalise_value(s, q, static_cast<double>(r.e[kNew]), c);
}

// --- spec and contexts -----------------------------------------------------------------------------------------------

namespace {

int nearest_centroid(const std::vector<double>& c, std::span<const double> v) {
  const int k = static_cast<int>(c.size() / kMacroStats);
  int best = 0;
  double bd = std::numeric_limits<double>::infinity();
  for (int j = 0; j < k; ++j) {
    double d = 0.0;
    for (int i = 0; i < kMacroStats; ++i) {
      const double t = v[sz(i)] - c[sz(j) * kMacroStats + sz(i)];
      d += t * t;
    }
    if (d < bd) {
      bd = d;
      best = j;
    }
  }
  return best;
}

template <std::size_t N>
int bin_of(const std::array<float, N>& edges, float v) {
  int b = 0;
  for (const float e : edges) b += v >= e ? 1 : 0;
  return b;
}

float quantile(std::vector<float> v, double q) {
  if (v.empty()) return 0.f;
  const auto k = static_cast<std::size_t>(std::clamp(q, 0.0, 1.0) * static_cast<double>(v.size() - 1));
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
  return v[k];
}

}  // namespace

Spec fit_spec(const rollout::Model& m, std::span<const Row> rows, Domain domain, int size, std::uint64_t seed) {
  if (rows.empty()) throw std::invalid_argument("dcm fine: fit_spec needs rows");
  Spec s;
  s.domain = domain;
  s.one_shot = !m.loop;
  s.size = size;
  for (int q = 0; q < 2; ++q) {
    s.s[sz(q)] = std::max(1e-6f, m.render_scale[sz(q)]);
    s.eps[sz(q)] = 0.02f * s.s[sz(q)];
  }
  for (int q = 0; q < 2; ++q) {
    std::vector<float> lv, ra;
    for (const Row& r : rows) {
      if (r.channel != q) continue;
      const float c = r.e[kCoarse] / s.s[sz(q)];
      if (c < s.empty) continue;
      lv.push_back(c);
      ra.push_back(r.e[kAdv] / (std::max(0.f, r.e[kCoarse]) + s.eps[sz(q)]));
    }
    for (int k = 0; k < 4; ++k) s.lvl[sz(q)][sz(k)] = quantile(lv, 0.2 * (k + 1));
    for (int k = 0; k < 3; ++k) s.ratio[sz(q)][sz(k)] = quantile(ra, 0.25 * (k + 1));
  }
  {
    std::vector<float> sp;
    for (const Row& r : rows) sp.push_back(r.speed);
    s.speed = {quantile(sp, 1.0 / 3.0), quantile(sp, 2.0 / 3.0)};
  }
  // regional statistics: standardised, then k-means on a deterministic subsample
  std::array<double, kMacroStats> mean{}, var{};
  for (const Row& r : rows) {
    for (int i = 0; i < kMacroStats; ++i) mean[sz(i)] += static_cast<double>(r.macro[sz(i)]);
  }
  for (auto& v : mean) v /= static_cast<double>(rows.size());
  for (const Row& r : rows) {
    for (int i = 0; i < kMacroStats; ++i) var[sz(i)] += std::pow(static_cast<double>(r.macro[sz(i)]) - mean[sz(i)], 2.0);
  }
  for (int i = 0; i < kMacroStats; ++i) {
    s.macro_mean[sz(i)] = static_cast<float>(mean[sz(i)]);
    s.macro_sd[sz(i)] = static_cast<float>(std::max(1e-9, std::sqrt(var[sz(i)] / static_cast<double>(rows.size()))));
  }
  const std::size_t stride = std::max<std::size_t>(1, rows.size() / 20000);
  std::vector<double> x;
  for (std::size_t i = 0; i < rows.size(); i += stride) {
    for (int k = 0; k < kMacroStats; ++k) x.push_back((static_cast<double>(rows[i].macro[sz(k)]) - mean[sz(k)]) / static_cast<double>(s.macro_sd[sz(k)]));
  }
  for (const int K : {4, 8}) {
    KMeansOptions o;
    o.k = K;
    o.restarts = 4;
    o.seed = seed;
    const KMeans km = fit_kmeans(x, kMacroStats, o);
    (K == 4 ? s.macro4 : s.macro8) = km.centroids;
  }
  return s;
}

std::array<int, kContexts> contexts_of(const Spec& s, const Row& r, std::span<const float> controls, int size) {
  std::array<int, kContexts> c{};
  const int q = r.channel;
  const float sq = s.s[sz(q)];
  const float lv = r.e[kCoarse] / sq;
  c[kLvl] = lv < s.empty ? 0 : 1 + bin_of(s.lvl[sz(q)], lv);
  c[kRatio] = bin_of(s.ratio[sz(q)], r.e[kAdv] / (std::max(0.f, r.e[kCoarse]) + s.eps[sz(q)]));
  c[kFlow] = bin_of(s.speed, r.speed) * 2 + (r.vort >= 0.f ? 1 : 0);
  c[kHeight] = std::clamp(r.y * 4 / std::max(1, size), 0, 3);
  const auto bin3 = [](float v) { return std::clamp(static_cast<int>(v * 3.f), 0, 2); };
  const float ci = controls.size() > 0 ? controls[0] : 0.5f, ct = controls.size() > 2 ? controls[2] : 0.5f;
  c[kCtrl] = bin3(ci) * 3 + bin3(ct);
  c[kAge] = s.one_shot ? bin_of(s.age, r.time) : 0;
  c[kChannel] = q;
  std::array<double, kMacroStats> v{};
  for (int i = 0; i < kMacroStats; ++i) v[sz(i)] = (static_cast<double>(r.macro[sz(i)]) - static_cast<double>(s.macro_mean[sz(i)])) / static_cast<double>(s.macro_sd[sz(i)]);
  c[kMacro4] = s.macro4.empty() ? 0 : nearest_centroid(s.macro4, v);
  c[kMacro8] = s.macro8.empty() ? 0 : nearest_centroid(s.macro8, v);
  c[kExtra] = std::min<int>(r.extra, std::max(1, s.extra_size) - 1);
  return c;
}

// --- the mixer -------------------------------------------------------------------------------------------------------

std::string describe(const Config& c) {
  std::string in, mx = "-";
  for (const int e : c.experts) in += (in.empty() ? "" : "+") + std::string(expert_name(e));
  for (const int k : c.mixer_contexts) mx += "," + std::string(context_name(k));
  const auto nm = [](int k) { return k < 0 ? std::string("-") : std::string(context_name(k)); };
  std::string avm = "off";
  if (c.spec.avm_weight > 0.0) avm = std::format("{} {:g}", nm(c.avm_context), c.spec.avm_weight);
  return std::format("inputs {} | mixers {} | avm {} | scale {} | loss {} | lr {:g}/{:g} | ep {}", in, mx, avm, nm(c.scale_context),
                     loss_name(c.spec.loss), c.spec.lr1, c.spec.lr2, c.epochs);
}

ValuePrediction Mixer::predict(std::span<const double> x, std::span<const int> ctx, std::span<const double> z) const {
  if (!net) throw std::logic_error("dcm fine: the mixer has no net");
  std::array<double, kExperts> xs{};
  std::array<int, 16> cs{};
  const std::size_t n = config.experts.size(), m = config.mixer_contexts.size() + 1;
  for (std::size_t k = 0; k < n; ++k) xs[k] = x[sz(config.experts[k])];
  for (std::size_t j = 1; j < m; ++j) cs[j] = ctx[sz(config.mixer_contexts[j - 1])];
  cs[m] = 0;
  cs[m + 1] = config.avm_context >= 0 ? ctx[sz(config.avm_context)] : 0;
  cs[m + 2] = config.scale_context >= 0 ? ctx[sz(config.scale_context)] : 0;
  return net->predict(std::span<const double>(xs.data(), n), std::span<const int>(cs.data(), m + 3), z);
}

Mixer make_mixer(const Spec& s, Config c, const ValueNet& net) {
  Mixer m;
  m.spec = s;
  const double lr1 = c.spec.lr1, lr2 = c.spec.lr2;  // as configured (training halves the net's own after each pass)
  c.spec = net.spec();
  c.spec.lr1 = lr1;
  c.spec.lr2 = lr2;
  m.config = std::move(c);
  m.text = net.serialise();
  m.net.emplace(m.text);
  return m;
}

Mixer v1_mixer(const Spec& s, std::span<const int> contexts) {
  Config c;
  for (int e = 0; e < kExperts; ++e) c.experts.push_back(e);
  c.mixer_contexts.assign(contexts.begin(), contexts.end());
  c.spec.context_sizes = {1};
  for (const int k : contexts) c.spec.context_sizes.push_back(context_size(s, k));
  c.spec.loss = ValueLoss::laplace;
  ValueNet net(kExperts, kScaleFeatures, c.spec);
  net.set_rule(kLock);
  net.freeze();
  return make_mixer(s, std::move(c), net);
}

namespace {

void put_array(std::ostream& o, std::string_view key, auto const& a) {
  o << key;
  for (const auto v : a) o << ' ' << std::format("{:.9g}", static_cast<double>(v));
  o << '\n';
}

std::vector<double> get_values(std::istream& in, std::string_view key) {
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error(std::format("dcm fine: mixer file ends before '{}'", key));
  std::istringstream ls(line);
  std::string k;
  ls >> k;
  if (k != key) throw std::runtime_error(std::format("dcm fine: mixer file: expected '{}', got '{}'", key, k));
  std::vector<double> v;
  for (double d; ls >> d;) v.push_back(d);
  return v;
}

}  // namespace

void save_mixer(const std::filesystem::path& path, const Mixer& m) {
  std::ofstream o(path);
  if (!o) throw std::runtime_error("dcm fine: cannot write " + path.string());
  const Spec& s = m.spec;
  o << "nvfx-dcm-fine v1\n";
  put_array(o, "domain", std::array{s.domain == Domain::linear ? 0 : 1, s.one_shot ? 1 : 0, s.size, s.grain_octaves, s.extra_size});
  put_array(o, "scale", s.s);
  put_array(o, "eps", s.eps);
  put_array(o, "lvl", std::array{s.lvl[0][0], s.lvl[0][1], s.lvl[0][2], s.lvl[0][3], s.lvl[1][0], s.lvl[1][1], s.lvl[1][2], s.lvl[1][3], s.empty});
  put_array(o, "ratio", std::array{s.ratio[0][0], s.ratio[0][1], s.ratio[0][2], s.ratio[1][0], s.ratio[1][1], s.ratio[1][2]});
  put_array(o, "speed", s.speed);
  put_array(o, "age", s.age);
  put_array(o, "macro_mean", s.macro_mean);
  put_array(o, "macro_sd", s.macro_sd);
  put_array(o, "macro4", s.macro4);
  put_array(o, "macro8", s.macro8);
  put_array(o, "grain", std::array{s.grain_freq, s.grain_rate});
  put_array(o, "experts", m.config.experts);
  put_array(o, "mixers", m.config.mixer_contexts);
  put_array(o, "maps", std::array{m.config.avm_context, m.config.scale_context, m.config.epochs});
  put_array(o, "learn", std::array{m.config.spec.loss == ValueLoss::laplace ? 1.0 : 0.0, m.config.spec.lr1, m.config.spec.lr2, m.config.spec.avm_weight,
                                   m.config.spec.scale_lr});
  o << "net\n" << m.text;
}

Mixer load_mixer(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("dcm fine: cannot read " + path.string());
  std::string line;
  std::getline(in, line);
  if (line != "nvfx-dcm-fine v1") throw std::runtime_error(path.string() + " is not a DCM-fine mixer");
  Mixer m;
  Spec& s = m.spec;
  const auto d = get_values(in, "domain");
  if (d.size() != 5) throw std::runtime_error("dcm fine: mixer file: domain line");
  s.domain = d[0] == 0 ? Domain::linear : Domain::log;
  s.one_shot = d[1] != 0;
  s.size = static_cast<int>(d[2]);
  s.grain_octaves = static_cast<int>(d[3]);
  s.extra_size = static_cast<int>(d[4]);
  const auto fill = [&](std::string_view key, auto& a) {
    const auto v = get_values(in, key);
    if (v.size() != a.size()) throw std::runtime_error(std::format("dcm fine: mixer file: {} values", key));
    for (std::size_t i = 0; i < v.size(); ++i) a[i] = static_cast<std::remove_reference_t<decltype(a[0])>>(v[i]);
  };
  fill("scale", s.s);
  fill("eps", s.eps);
  {
    const auto v = get_values(in, "lvl");
    if (v.size() != 9) throw std::runtime_error("dcm fine: mixer file: lvl values");
    for (int q = 0; q < 2; ++q) {
      for (int k = 0; k < 4; ++k) s.lvl[sz(q)][sz(k)] = static_cast<float>(v[sz(q * 4 + k)]);
    }
    s.empty = static_cast<float>(v[8]);
  }
  {
    const auto v = get_values(in, "ratio");
    if (v.size() != 6) throw std::runtime_error("dcm fine: mixer file: ratio values");
    for (int q = 0; q < 2; ++q) {
      for (int k = 0; k < 3; ++k) s.ratio[sz(q)][sz(k)] = static_cast<float>(v[sz(q * 3 + k)]);
    }
  }
  fill("speed", s.speed);
  fill("age", s.age);
  fill("macro_mean", s.macro_mean);
  fill("macro_sd", s.macro_sd);
  s.macro4 = get_values(in, "macro4");
  s.macro8 = get_values(in, "macro8");
  {
    const auto v = get_values(in, "grain");
    if (v.size() != 2) throw std::runtime_error("dcm fine: mixer file: grain values");
    s.grain_freq = static_cast<float>(v[0]);
    s.grain_rate = static_cast<float>(v[1]);
  }
  for (const double e : get_values(in, "experts")) m.config.experts.push_back(static_cast<int>(e));
  for (const double e : get_values(in, "mixers")) m.config.mixer_contexts.push_back(static_cast<int>(e));
  const auto mp = get_values(in, "maps");
  if (mp.size() != 3) throw std::runtime_error("dcm fine: mixer file: maps values");
  m.config.avm_context = static_cast<int>(mp[0]);
  m.config.scale_context = static_cast<int>(mp[1]);
  m.config.epochs = static_cast<int>(mp[2]);
  m.config.spec.context_sizes = {1};
  for (const int k : m.config.mixer_contexts) m.config.spec.context_sizes.push_back(context_size(m.spec, k));
  std::getline(in, line);
  if (line.starts_with("learn ")) {  // the learning settings (absent from the earliest files: describe() then shows defaults)
    std::istringstream ls(line.substr(6));
    double loss = 0;
    ls >> loss >> m.config.spec.lr1 >> m.config.spec.lr2 >> m.config.spec.avm_weight >> m.config.spec.scale_lr;
    if (!ls) throw std::runtime_error("dcm fine: mixer file: learn values");
    m.config.spec.loss = loss != 0.0 ? ValueLoss::laplace : ValueLoss::squared;
    std::getline(in, line);
  }
  if (line != "net") throw std::runtime_error("dcm fine: mixer file: no net");
  std::ostringstream rest;
  rest << in.rdbuf();
  m.text = rest.str();
  m.net.emplace(m.text);
  if (m.net->inputs() != static_cast<int>(m.config.experts.size())) throw std::runtime_error("dcm fine: mixer file: inputs");
  return m;
}

// --- frames ----------------------------------------------------------------------------------------------------------

void compute_frame(const rollout::Model& m, const rollout::State& s, std::uint64_t seed, std::span<const float> controls, Frame& f,
                   const CellContext& extra) {
  const rollout::Hyper& h = m.h;
  const rollout::DetailSpec& dt = m.detail;
  const int R = h.res, S = s.size, C = h.channels();
  const float k = fl(S) / fl(R), px128 = fl(S) / 128.f, t = s.time + 0.5f / m.fps;
  const std::size_t S2 = sz(S) * sz(S), N = sz(R) * sz(R);
  f.S = S;
  f.R = R;
  for (auto* a : {&f.prev, &f.A, &f.Asl, &f.L, &f.rA, &f.aup, &f.n1, &f.n2, &f.n3, &f.cup, &f.bres, &f.lap, &f.grad}) {
    for (auto& v : *a) v.assign(S2, 0.f);
  }
  f.ux.assign(S2, 0.f);
  f.vy.assign(S2, 0.f);
  f.speed.assign(S2, 0.f);
  f.vort.assign(S2, 0.f);
  f.macro.assign(N * kMacroStats, 0.f);
  f.extra.assign(N, 0);
  // fine velocity: the coarse flow (and the swirl), as rollout::detail_step
  float amp = dt.swirl * px128;
  if (dt.swirl_control >= 0 && sz(dt.swirl_control) < controls.size()) amp *= 0.3f + controls[sz(dt.swirl_control)];
  if (dt.swirl_ramp > 0.f) amp *= std::min(1.f, s.since_start / dt.swirl_ramp);
  Swirl sw;
  if (amp > 0.f) sw = swirl_field(m, seed, t);
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      const float xc = (fl(x) + 0.5f) / k - 0.5f, yc = (fl(y) + 0.5f) / k - 0.5f;
      float u = bilinear_ch(s.flow.data(), R, 2, 0, xc, yc) * k, v = bilinear_ch(s.flow.data(), R, 2, 1, xc, yc) * k;
      if (amp > 0.f) {
        const float lx = ((fl(x) + 0.5f) / px128 + 0.5f) / sw.spacing + 1.f, ly = ((fl(y) + 0.5f) / px128 + 0.5f) / sw.spacing + 1.f;
        u += amp * bilinear(sw.u.data(), sw.n, lx, ly);
        v += amp * bilinear(sw.v.data(), sw.n, lx, ly);
      }
      const std::size_t i = sz(y) * sz(S) + sz(x);
      f.ux[i] = u;
      f.vy[i] = v;
      f.speed[i] = std::sqrt(u * u + v * v) / px128;
    }
  }
  // MacCormack advection (clamped to the forward step's stencil), zero outside the frame
  std::vector<float> fb(S2), lo(S2), hi(S2);
  for (int q = 0; q < 2; ++q) {
    const std::vector<float>& Q = q == 0 ? s.fine_t : s.fine_d;
    f.prev[sz(q)] = Q;
    std::vector<float>& fa = f.Asl[sz(q)];
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) {
        const std::size_t i = sz(y) * sz(S) + sz(x);
        fa[i] = bilinear_zero(Q.data(), S, fl(x) - f.ux[i], fl(y) - f.vy[i], &lo[i], &hi[i]);
      }
    }
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) {
        const std::size_t i = sz(y) * sz(S) + sz(x);
        fb[i] = bilinear_zero(fa.data(), S, fl(x) + f.ux[i], fl(y) + f.vy[i]);
      }
    }
    std::vector<float>& A = f.A[sz(q)];
    for (std::size_t i = 0; i < S2; ++i) A[i] = std::max(0.f, std::clamp(fa[i] + 0.5f * (Q[i] - fb[i]), lo[i], hi[i]));
  }
  // the lock, keeping its parts, and the experts built on it
  const int kk = S / R;
  std::vector<float> B(N), rr(N), aa(N), res(N);
  for (int ch = 0; ch < 2; ++ch) {
    const std::vector<float>& Q = f.A[sz(ch)];
    std::ranges::fill(B, 0.f);
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) B[sz(y / kk) * sz(R) + sz(x / kk)] += Q[sz(y) * sz(S) + sz(x)];
    }
    for (int i = 0; i < R * R; ++i) {
      const float b = B[sz(i)] / fl(kk * kk), target = s.coarse[sz(i) * sz(C) + 2 + sz(ch)];
      constexpr float eps = 1e-4f;
      const float r = (target + eps) / (b + eps);
      rr[sz(i)] = r <= 1.f ? r : std::min(r, dt.grow);
      aa[sz(i)] = std::max(0.f, target - b * rr[sz(i)]);
      res[sz(i)] = target - b;
    }
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) {
        const float xc = (fl(x) + 0.5f) / k - 0.5f, yc = (fl(y) + 0.5f) / k - 0.5f;
        const std::size_t i = sz(y) * sz(S) + sz(x);
        float add = bilinear(aa.data(), R, xc, yc);
        f.aup[sz(ch)][i] = add;
        if (add > 0.f) {
          // the flicker fbm octave by octave, summed as fbm() sums them (so phi is noise_flicker's value)
          const float X = (fl(x) + 0.5f) / px128 + 0.5f, Y = (fl(y) + 0.5f) / px128 + 0.5f;
          float sum = 0.f, a = 1.f, norm = 0.f, nx = X * m.noise.flicker_freq, ny = Y * m.noise.flicker_freq, nz = t * m.noise.flicker_rate;
          std::array<float, 3> oct{};
          for (int o = 0; o < m.noise.flicker_octaves; ++o) {
            const float v = value_noise(nx, ny, nz, seed + static_cast<std::uint64_t>(o) * 0x9e3779b97f4a7c15ULL);
            if (o < 3) oct[sz(o)] = v;
            sum += a * v;
            norm += a;
            a *= 0.5f;
            nx *= 2.f;
            ny *= 2.f;
            nz *= 2.f;
          }
          f.n1[sz(ch)][i] = add * oct[0];
          f.n2[sz(ch)][i] = add * oct[1];
          f.n3[sz(ch)][i] = add * oct[2];
          if (dt.contrast > 0.f) {
            const float phi = sum / norm;
            add *= (1.f - dt.contrast) + dt.contrast * dt.kappa * smoothstep01((phi - dt.edge0) / (dt.edge1 - dt.edge0));
          }
        }
        const float ra = Q[i] * bilinear(rr.data(), R, xc, yc);
        f.rA[sz(ch)][i] = ra;
        f.L[sz(ch)][i] = ra + add;
        f.cup[sz(ch)][i] = bilinear_ch(s.coarse.data(), R, C, 2 + ch, xc, yc);
        f.bres[sz(ch)][i] = bilinear(res.data(), R, xc, yc);
      }
    }
    // shape of the advected field: 5-point laplacian and the central-difference gradient, zero outside
    const auto at = [&](int x, int y) { return (x < 0 || y < 0 || x >= S || y >= S) ? 0.f : Q[sz(y) * sz(S) + sz(x)]; };
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) {
        const std::size_t i = sz(y) * sz(S) + sz(x);
        const float c0 = Q[i], l = at(x - 1, y), r = at(x + 1, y), d = at(x, y - 1), u = at(x, y + 1);
        f.lap[sz(ch)][i] = l + r + d + u - 4.f * c0;
        const float gx = 0.5f * (r - l), gy = 0.5f * (u - d);
        f.grad[sz(ch)][i] = std::sqrt(gx * gx + gy * gy);
      }
    }
  }
  // coarse vorticity (cells per frame per cell), upsampled; regional statistics per coarse cell
  std::vector<float> w(N), spd(N);
  const auto U = [&](int x, int y, int c) { return s.flow[(sz(std::clamp(y, 0, R - 1)) * sz(R) + sz(std::clamp(x, 0, R - 1))) * 2 + sz(c)]; };
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      w[sz(y) * sz(R) + sz(x)] = 0.5f * (U(x + 1, y, 1) - U(x - 1, y, 1)) - 0.5f * (U(x, y + 1, 0) - U(x, y - 1, 0));
      spd[sz(y) * sz(R) + sz(x)] = std::sqrt(U(x, y, 0) * U(x, y, 0) + U(x, y, 1) * U(x, y, 1));
    }
  }
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      const float xc = (fl(x) + 0.5f) / k - 0.5f, yc = (fl(y) + 0.5f) / k - 0.5f;
      f.vort[sz(y) * sz(S) + sz(x)] = bilinear(w.data(), R, xc, yc);
    }
  }
  for (int cy = 0; cy < R; ++cy) {
    for (int cx = 0; cx < R; ++cx) {
      std::array<double, kMacroStats> sum{};
      int n = 0;
      for (int y = std::max(0, cy - 4); y < std::min(R, cy + 4); ++y) {
        for (int x = std::max(0, cx - 4); x < std::min(R, cx + 4); ++x) {
          const std::size_t j = sz(y) * sz(R) + sz(x);
          sum[0] += static_cast<double>(s.coarse[j * sz(C) + 2]);
          sum[1] += static_cast<double>(s.coarse[j * sz(C) + 3]);
          sum[2] += static_cast<double>(spd[j]);
          sum[3] += static_cast<double>(std::abs(w[j]));
          ++n;
        }
      }
      for (int i = 0; i < kMacroStats; ++i) f.macro[(sz(cy) * sz(R) + sz(cx)) * kMacroStats + sz(i)] = static_cast<float>(sum[sz(i)] / n);
    }
  }
  if (extra) extra(s, f.extra);
}

Row row_of(const Frame& f, const rollout::State& s, int x, int y, int q) {
  Row r;
  const std::size_t i = sz(y) * sz(f.S) + sz(x), qq = sz(q);
  r.x = static_cast<std::int16_t>(x);
  r.y = static_cast<std::int16_t>(y);
  r.channel = static_cast<std::int8_t>(q);
  const float A = f.A[qq][i];
  r.e[kAdv] = A;
  r.e[kAdvSl] = f.Asl[qq][i];
  r.e[kAdvDiff] = A - f.Asl[qq][i];
  r.e[kLock] = f.L[qq][i];
  r.e[kLockScaled] = f.rA[qq][i];
  r.e[kNew] = f.aup[qq][i];
  r.e[kNoise1] = f.n1[qq][i];
  r.e[kNoise2] = f.n2[qq][i];
  r.e[kNoise3] = f.n3[qq][i];
  r.e[kCoarse] = f.cup[qq][i];
  r.e[kBlockRes] = f.bres[qq][i];
  r.e[kLaplace] = f.lap[qq][i];
  r.e[kGrad] = f.grad[qq][i] * A;
  r.e[kPrev] = f.prev[qq][i];
  r.speed = f.speed[i];
  r.vort = f.vort[i];
  const int kk = f.S / f.R;
  const std::size_t cell = sz(y / kk) * sz(f.R) + sz(x / kk);
  for (int k = 0; k < kMacroStats; ++k) r.macro[sz(k)] = f.macro[cell * kMacroStats + sz(k)];
  r.extra = f.extra.empty() ? 0 : f.extra[cell];
  r.time = s.time;
  r.grad = f.grad[qq][i];
  return r;
}

std::array<int, kContexts> frame_contexts(const Spec& sp, const Frame& f, const rollout::State& s, std::span<const float> controls,
                                          int x, int y, int q) {
  return contexts_of(sp, row_of(f, s, x, y, q), controls, f.S);
}

bool skip_pixel(const Spec& sp, const Frame& f, std::size_t i, int q) {
  const std::size_t qq = sz(q);
  const float thr = kSkip * sp.s[qq];
  return f.A[qq][i] < thr && f.Asl[qq][i] < thr && f.aup[qq][i] < thr && f.prev[qq][i] < thr && f.cup[qq][i] < thr && f.L[qq][i] < thr;
}

bool skip_row(const Spec& sp, const Row& r) {
  const float thr = kSkip * sp.s[sz(r.channel)];
  return r.e[kAdv] < thr && r.e[kAdvSl] < thr && r.e[kNew] < thr && r.e[kPrev] < thr && r.e[kCoarse] < thr && r.e[kLock] < thr;
}

// --- generation ------------------------------------------------------------------------------------------------------

float grain(const Spec& sp, std::uint64_t seed, float X, float Y, float t) {
  const float g = fbm(X * sp.grain_freq, Y * sp.grain_freq, t * sp.grain_rate, seed * kGrainSalt + 0x5EEDULL, sp.grain_octaves);
  const std::vector<float>& qt = grain_quantiles(sp.grain_octaves);
  // the uniform level of g: interpolate between the quantiles around it
  const auto it = std::ranges::upper_bound(qt, g);
  double u = 0.0;
  if (it == qt.begin()) {
    u = 0.5 / 256.0;
  } else if (it == qt.end()) {
    u = 1.0 - 0.5 / 256.0;
  } else {
    const auto j = static_cast<std::size_t>(it - qt.begin());
    const float a = qt[j - 1], b = qt[j];
    const double fr = b > a ? static_cast<double>(g - a) / static_cast<double>(b - a) : 0.5;
    u = (static_cast<double>(j - 1) + fr) / 256.0;
  }
  u = std::clamp(u, 0.5 / 256.0, 1.0 - 0.5 / 256.0);
  return static_cast<float>(u < 0.5 ? std::log(2.0 * u) : -std::log(2.0 * (1.0 - u)));
}

void detail_step(const rollout::Model& m, const Mixer& mix, rollout::State& s, std::uint64_t seed, std::span<const float> controls,
                 const GenOptions& g, Frame& f) {
  compute_frame(m, s, seed, controls, f, g.extra);
  const Spec& sp = mix.spec;
  const int S = s.size, R = m.h.res, C = m.h.channels();
  const float px128 = fl(S) / 128.f, t = s.time + 0.5f / m.fps, k = fl(S) / fl(R);
  const std::size_t S2 = sz(S) * sz(S);
  std::array<std::vector<float>, 2> out;
  std::array<double, kExperts> x{};
  std::array<double, kScaleFeatures> z{};
  std::vector<float> gr;
  if (g.tau > 0.0) {
    gr.resize(S2);
    for (int y = 0; y < S; ++y) {
      for (int xx = 0; xx < S; ++xx) {
        gr[sz(y) * sz(S) + sz(xx)] = grain(sp, seed, (fl(xx) + 0.5f) / px128 + 0.5f, (fl(y) + 0.5f) / px128 + 0.5f, t);
      }
    }
  }
  for (int q = 0; q < 2; ++q) {
    out[sz(q)].assign(S2, 0.f);
    for (int y = 0; y < S; ++y) {
      for (int xx = 0; xx < S; ++xx) {
        const std::size_t i = sz(y) * sz(S) + sz(xx);
        if (skip_pixel(sp, f, i, q)) {
          out[sz(q)][i] = f.L[sz(q)][i];  // v1's value: nothing to see here
          continue;
        }
        const Row r = row_of(f, s, xx, y, q);
        const auto ctx = contexts_of(sp, r, controls, S);
        normalise(sp, r, x, z);
        const ValuePrediction p = mix.predict(x, ctx, z);
        double vn = p.mu;
        if (g.tau > 0.0) vn += g.tau * p.b * static_cast<double>(gr[i]);
        // a local maximum principle with slack: never more than twice the largest of the values the pixel is built from
        // (v1's lock output, the advected and previous values, the coarse value); it only binds on a runaway sample
        const double cap = 2.0 * std::max({r.e[kLock], r.e[kAdv], r.e[kAdvSl], r.e[kPrev], r.e[kCoarse]});
        out[sz(q)][i] = static_cast<float>(std::clamp(value_of(sp, q, vn, static_cast<double>(r.e[kCoarse])), 0.0, cap));
      }
    }
    if (g.relock > 0) {  // block means back to the coarse state: factors per block, bilinear between blocks
      const int kk = S / R;
      const rollout::DetailSpec& dt = m.detail;
      std::vector<float> B(sz(R) * sz(R), 0.f), rr(B.size()), aa(B.size());
      std::vector<float>& Q = out[sz(q)];
      for (int y = 0; y < S; ++y) {
        for (int xx = 0; xx < S; ++xx) B[sz(y / kk) * sz(R) + sz(xx / kk)] += Q[sz(y) * sz(S) + sz(xx)];
      }
      for (std::size_t i = 0; i < B.size(); ++i) {
        const float b = B[i] / fl(kk * kk), target = s.coarse[i * sz(C) + 2 + sz(q)];
        constexpr float eps = 1e-4f;
        const float r = (target + eps) / (b + eps);
        rr[i] = g.relock == 1 || r <= 1.f ? r : std::min(r, dt.grow);
        aa[i] = std::max(0.f, target - b * rr[i]);
      }
      for (int y = 0; y < S; ++y) {
        for (int xx = 0; xx < S; ++xx) {
          const float xc = (fl(xx) + 0.5f) / k - 0.5f, yc = (fl(y) + 0.5f) / k - 0.5f;
          float add = bilinear(aa.data(), R, xc, yc);
          if (g.relock == 2 && add > 0.f && dt.contrast > 0.f) {
            const float X = (fl(xx) + 0.5f) / px128 + 0.5f, Y = (fl(y) + 0.5f) / px128 + 0.5f;
            const float phi = rollout::noise_flicker(m.noise, seed, X, Y, t);
            add *= (1.f - dt.contrast) + dt.contrast * dt.kappa * smoothstep01((phi - dt.edge0) / (dt.edge1 - dt.edge0));
          }
          float& v = Q[sz(y) * sz(S) + sz(xx)];
          v = v * bilinear(rr.data(), R, xc, yc) + add;
        }
      }
    }
  }
  s.fine_t.swap(out[0]);
  s.fine_d.swap(out[1]);
}

void step(const rollout::Model& m, const Mixer& mix, rollout::State& s, std::span<const float> controls, std::uint64_t seed,
          const GenOptions& g, Frame& f) {
  const rollout::Hyper& h = m.h;
  std::vector<float> cond(sz(h.cond())), noise(sz(h.res) * sz(h.res) * rollout::kNoise), next(s.coarse.size());
  rollout::condition(m, controls, s.time, cond);
  rollout::coarse_noise(m, seed, s.time, noise);
  rollout::coarse_step(m, s.coarse, noise, cond, s.pressure, next, s.flow);
  s.coarse.swap(next);
  detail_step(m, mix, s, seed, controls, g, f);
  s.time += 1.f / m.fps;
  s.since_start += 1.f / m.fps;
}

rollout::State start(const rollout::Model& m, const Mixer& mix, int index, int size, std::span<const float> controls,
                     std::uint64_t seed, const GenOptions& g, int warmup, bool use_fine) {
  const rollout::Hyper& h = m.h;
  const rollout::StartPoint& sp = m.starts.at(sz(index));
  const int R = h.res, C = h.channels();
  rollout::State s;
  s.res = R;
  s.size = size;
  s.time = sp.time;
  s.coarse.assign(sz(R) * sz(R) * sz(C), 0.f);
  for (int i = 0; i < R * R; ++i) {
    for (int c = 0; c < rollout::kPhys; ++c) s.coarse[sz(i) * sz(C) + sz(c)] = sp.coarse[sz(i) * rollout::kPhys + sz(c)];
  }
  s.pressure.assign(sz(R) * sz(R), 0.f);
  s.flow.assign(sz(R) * sz(R) * 2, 0.f);
  s.fine_t.assign(sz(size) * sz(size), 0.f);
  s.fine_d.assign(sz(size) * sz(size), 0.f);
  const bool fine = use_fine && !sp.fine_t.empty();
  const int n = fine ? h.start_fine : R;
  const float k = fl(size) / fl(n);
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const float xs = (fl(x) + 0.5f) / k - 0.5f, ys = (fl(y) + 0.5f) / k - 0.5f;
      const std::size_t i = sz(y) * sz(size) + sz(x);
      s.fine_t[i] = fine ? bilinear(sp.fine_t.data(), n, xs, ys) : bilinear_ch(sp.coarse.data(), R, rollout::kPhys, 2, xs, ys);
      s.fine_d[i] = fine ? bilinear(sp.fine_d.data(), n, xs, ys) : bilinear_ch(sp.coarse.data(), R, rollout::kPhys, 3, xs, ys);
    }
  }
  if (!fine) {
    s.since_start = m.detail.swirl_ramp;
    Frame f;
    const int frames = warmup < 0 ? h.warmup : warmup;
    for (int i = 0; i < frames; ++i) step(m, mix, s, controls, seed, g, f);
    s.since_start = m.detail.swirl_ramp;
  }
  return s;
}

void render(const rollout::Model& m, const rollout::State& s, std::span<float> rgba) {
  using rollout::kDirs;
  using rollout::kDirSteps;
  using rollout::kRenderIn;
  const int R = m.h.res, S = s.size, C = m.h.channels(), H = m.h.render_hidden;
  const float k = fl(S) / fl(R), it = 1.f / m.render_scale[0], id = 1.f / m.render_scale[1];
  static constexpr std::array<std::array<int, 2>, kDirs> dirs{{{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}};
  std::vector<float> ds(sz(R) * sz(R) * kDirs);
  for (int cy = 0; cy < R; ++cy) {
    for (int cx = 0; cx < R; ++cx) {
      for (int j = 0; j < kDirs; ++j) {
        float sum = 0.f;
        for (int st = 1; st <= kDirSteps; ++st) {
          const int xx = cx + st * dirs[sz(j)][0], yy = cy + st * dirs[sz(j)][1];
          if (xx >= 0 && yy >= 0 && xx < R && yy < R) sum += s.coarse[(sz(yy) * sz(R) + sz(xx)) * sz(C) + 3];
        }
        ds[(sz(cy) * sz(R) + sz(cx)) * kDirs + sz(j)] = sum;
      }
    }
  }
  const rollout::RenderLayout L = rollout::render_layout(m.h);
  const float* w = m.render_w.data();
  std::vector<float> h1(sz(H)), h2(sz(H));
  std::array<float, kRenderIn> f{};
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      const float xc = (fl(x) + 0.5f) / k - 0.5f, yc = (fl(y) + 0.5f) / k - 0.5f;
      f[0] = s.fine_t[sz(y) * sz(S) + sz(x)] * it;
      f[1] = s.fine_d[sz(y) * sz(S) + sz(x)] * id;
      f[2] = bilinear_ch(s.coarse.data(), R, C, 2, xc, yc) * it;
      f[3] = bilinear_ch(s.coarse.data(), R, C, 3, xc, yc) * id;
      const float cx = std::clamp(xc, 0.f, fl(R - 1)), cy = std::clamp(yc, 0.f, fl(R - 1));
      const int x0 = std::min(static_cast<int>(cx), R - 2), y0 = std::min(static_cast<int>(cy), R - 2);
      const float fx = cx - fl(x0), fy = cy - fl(y0);
      const float* d00 = ds.data() + (sz(y0) * sz(R) + sz(x0)) * kDirs;
      const float* d10 = d00 + kDirs;
      const float* d01 = d00 + sz(R) * kDirs;
      const float* d11 = d01 + kDirs;
      for (int j = 0; j < kDirs; ++j) {
        const float a = (1.f - fx) * d00[j] + fx * d10[j];
        const float b = (1.f - fx) * d01[j] + fx * d11[j];
        f[4 + sz(j)] = ((1.f - fy) * a + fy * b) * id;
      }
      for (int j = 0; j < H; ++j) {
        float v = w[L.b1 + sz(j)];
        for (int i = 0; i < kRenderIn; ++i) v += w[L.w1 + sz(j) * kRenderIn + sz(i)] * f[sz(i)];
        h1[sz(j)] = std::max(0.f, v);
      }
      for (int j = 0; j < H; ++j) {
        float v = w[L.b2 + sz(j)];
        for (int i = 0; i < H; ++i) v += w[L.w2 + sz(j) * sz(H) + sz(i)] * h1[sz(i)];
        h2[sz(j)] = std::max(0.f, v);
      }
      const float g = rollout::render_gate(f[0], f[1]);
      float* px = rgba.data() + (sz(S - 1 - y) * sz(S) + sz(x)) * 4;
      for (int c = 0; c < 4; ++c) {
        float v = w[L.bo + sz(c)];
        for (int i = 0; i < H; ++i) v += w[L.wo + sz(c) * sz(H) + sz(i)] * h2[sz(i)];
        px[c] = std::clamp(v * g, 0.f, 1.f);
      }
    }
  }
}

}  // namespace nfx::dcm::fine
