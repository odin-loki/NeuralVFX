// Study G's extras (docs/DCM.md §10): G4a's renderers and renderer mixer, G5a's shard critic, G5b's update mixer. See
// include/neuralfx/dcm/extras.hpp.
#include <neuralfx/dcm/extras.hpp>

#include <neuralfx/dcm/fine.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/noise.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace nfx::dcm::extras {

namespace {

std::size_t sz(int v) { return static_cast<std::size_t>(v); }
float fl(int v) { return static_cast<float>(v); }

// A plane of a rollout state's coarse grid, bilinear at continuous cell coordinates (cell centres at integers),
// clamped to the grid.
float coarse_bilinear(const std::vector<float>& f, int n, int ch, int c, float x, float y) {
  x = std::clamp(x, 0.f, fl(n - 1));
  y = std::clamp(y, 0.f, fl(n - 1));
  const int x0 = std::min(static_cast<int>(x), n - 2), y0 = std::min(static_cast<int>(y), n - 2);
  const float fx = x - fl(x0), fy = y - fl(y0);
  const auto at = [&](int xx, int yy) { return f[(sz(yy) * sz(n) + sz(xx)) * sz(ch) + sz(c)]; };
  return (1.f - fy) * ((1.f - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1.f - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
}

// compose.cpp's heat_rgb: a blackbody-like ramp as a sum of ramps.
void heat_rgb(float t, float& r, float& g, float& b) {
  t = std::clamp(t, 0.f, 1.4f);
  const float r1 = std::max(0.f, t - 0.25f), r2 = std::max(0.f, t - 0.5f), r3 = std::max(0.f, t - 0.75f), r4 = std::max(0.f, t - 1.f);
  r = 0.25f + 2.f * t - r1 - r2;
  g = 0.02f + 0.4f * t + 0.52f * r1 + 0.16f * r2 - 0.16f * r3 - 0.62f * r4;
  b = 0.04f * t + 0.08f * r1 + 0.32f * r2 + 0.76f * r3 - 0.2f * r4;
}

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

}  // namespace

// --- G4a ----------------------------------------------------------------------------------------------------------

void field_shader(const rollout::Model& m, const rollout::State& s, const ShaderLook& sp, std::span<float> rgba) {
  const int R = m.h.res, C = m.h.channels(), S = s.size;
  if (rgba.size() < sz(S) * sz(S) * 4) throw std::invalid_argument("field_shader: buffer too small");
  std::vector<float> shadow(sz(R) * sz(R), 0.f);
  for (int y = 0; y < R; ++y) {  // soot between each cell and the sky (straight up, slightly left), as Module::shade
    for (int x = 0; x < R; ++x) {
      float acc = 0.f;
      for (int st = 1; st <= 8; ++st) {
        const int xx = x - st / 3, yy = y + st;
        if (xx < 0 || yy >= R) break;
        acc += s.coarse[(sz(yy) * sz(R) + sz(xx)) * sz(C) + 3];
      }
      shadow[sz(y) * sz(R) + sz(x)] = acc;
    }
  }
  const float k = fl(S) / fl(R), inv_hs = 1.f / sp.heat_scale;
  const auto& fh = s.fine_t;
  const auto& fs = s.fine_d;
  for (int y = 0; y < S; ++y) {  // y up
    const float* up = fs.data() + sz(std::min(y + 2, S - 1)) * sz(S);
    const float* dn = fs.data() + sz(std::max(y - 2, 0)) * sz(S);
    const float* row = fs.data() + sz(y) * sz(S);
    for (int x = 0; x < S; ++x) {
      float* o = rgba.data() + (sz(S - 1 - y) * sz(S) + sz(x)) * 4;
      const std::size_t i = sz(y) * sz(S) + sz(x);
      const float T = std::max(0.f, fh[i]), D = std::max(0.f, fs[i]);
      if (std::max(T, D) < 1e-4f) {  // empty pixels stay transparent
        o[0] = o[1] = o[2] = o[3] = 0.f;
        continue;
      }
      const float a = 1.f - std::exp(-sp.soot_density * D);
      const float shv = coarse_bilinear(shadow, R, 1, 0, (fl(x) + 0.5f) / k - 0.5f, (fl(y) + 0.5f) / k - 0.5f);
      const float sh = std::exp(-sp.shadow * shv);
      const float slope = 0.5f * (row[std::min(x + 2, S - 1)] - row[std::max(x - 2, 0)]);
      const float nx = -sp.relief * slope, ny = -sp.relief * (0.5f * (up[x] - dn[x])), inv = 1.f / std::sqrt(nx * nx + ny * ny + 1.f);
      const float lambert = std::max(0.f, (-0.45f * nx + 0.6f * ny + 0.66f) * inv);
      const float moon = sp.sky * sh * (0.35f + 0.9f * lambert);
      const float Lr = 0.7f * moon, Lg = 0.8f * moon, Lb = 1.1f * moon;  // no scene light: the night sky only
      float hr = 0, hg = 0, hb = 0;
      heat_rgb(T * inv_hs, hr, hg, hb);
      const float e = sp.emission * std::pow(T * inv_hs, sp.emission_power) * (1.f - 0.55f * a);
      const float c[3] = {a * sp.soot_albedo * sp.tint[0] * Lr + e * hr, a * sp.soot_albedo * sp.tint[1] * Lg + e * hg,
                          a * sp.soot_albedo * sp.tint[2] * Lb + e * hb};
      for (int q = 0; q < 3; ++q) o[q] = std::pow(std::clamp(c[q], 0.f, 1.f), 1.f / 2.2f);
      o[3] = std::clamp(a, 0.f, 1.f);
    }
  }
}

SimRenderer::SimRenderer(sim::Effect effect, int size) : size_(size) {
  sim::Params p;
  p.effect = effect;
  p.size = size;
  p.sim_res = size;
  fluid_ = std::make_unique<sim::Fluid>(p);
  const std::size_t n = sz(size) * sz(size);
  st_.n = size;
  st_.u.assign(n, 0.f);
  st_.v.assign(n, 0.f);
  st_.pressure.assign(n, 0.f);
  st_.temp.assign(n, 0.f);
  st_.soot.assign(n, 0.f);
  out_.resize(n * 4);
}
SimRenderer::~SimRenderer() = default;

void SimRenderer::render(const rollout::State& s, std::span<float> rgba) {
  if (s.size != size_) throw std::invalid_argument("SimRenderer: size mismatch");
  std::ranges::copy(s.fine_t, st_.temp.begin());
  std::ranges::copy(s.fine_d, st_.soot.begin());
  fluid_->set_state(st_);
  fluid_->render(out_);
  for (std::size_t i = 0; i < out_.size(); ++i) rgba[i] = static_cast<float>(out_[i]) * (1.f / 255.f);
}

std::string_view render_expert_name(int e) {
  static constexpr std::array<std::string_view, kRenderExperts> n{"learned", "sim", "shader", "other1", "other2"};
  return e >= 0 && e < kRenderExperts ? n[sz(e)] : "?";
}

int age_bin(float f) {
  if (f < 2.5f) return 0;
  if (f < 5.5f) return 1;
  if (f < 11.5f) return 2;
  if (f < 23.5f) return 3;
  return 4;
}
int level_bin(float v) {
  if (v < 0.02f) return 0;
  if (v < 0.2f) return 1;
  if (v < 1.f) return 2;
  return 3;
}

std::string RenderMixConfig::name() const {
  std::string s;
  for (const int e : experts) s += (s.empty() ? "" : "+") + std::string(render_expert_name(e));
  return s;
}

namespace {
ValueNetSpec render_spec(double lr, double anneal) {
  ValueNetSpec sp;
  sp.context_sizes = {4 * kAgeBins, 4 * kLevelBins, 4 * kLevelBins};
  sp.final_contexts = 4;
  sp.avm_contexts = 0;
  sp.scale_contexts = 1;
  sp.loss = ValueLoss::squared;
  sp.lr1 = lr;
  sp.lr2 = lr / 2;
  sp.anneal = anneal;
  sp.eps = 1.0;  // the inputs are colours in [0, 1]: a step size of plain LMS for dim pixels, normalised for bright ones
  sp.limit = 4.0;
  return sp;
}
}  // namespace

RenderMixer::RenderMixer(RenderMixConfig cfg, double lr, double anneal)
    : cfg_(std::move(cfg)), net_(static_cast<int>(cfg_.experts.size()), 0, render_spec(lr, anneal)) {
  if (cfg_.experts.empty() || cfg_.experts[0] != kLearned) throw std::invalid_argument("RenderMixer: the learned renderer comes first");
  net_.set_rule(0);
  x_.assign(cfg_.experts.size(), 0.0);
}

double RenderMixer::predict(std::span<const float> experts, int channel, int age, int heat, int soot) {
  for (std::size_t k = 0; k < cfg_.experts.size(); ++k) x_[k] = experts[sz(cfg_.experts[k])];
  const std::array<int, 6> ctx{channel * kAgeBins + age, channel * kLevelBins + heat, channel * kLevelBins + soot, channel, 0, 0};
  return std::clamp(net_.predict(x_, ctx, {}).mu, 0.0, 1.0);
}

std::vector<double> RenderMixer::mean_weights(int channel) const {
  const std::size_t n = cfg_.experts.size();
  std::vector<double> w(n, 0.0);
  const auto fw = net_.final_mixer().weights(channel);
  for (int k = 0; k < net_.first_layer(); ++k) {
    const ValueMixer& mx = net_.mixer(k);
    const int per = mx.contexts() / 4;
    for (int b = 0; b < per; ++b) {
      const auto ww = mx.weights(channel * per + b);
      for (std::size_t i = 0; i < n; ++i) w[i] += fw[sz(k)] * ww[i] / per;
    }
  }
  return w;
}

void render_experts(const rollout::Model& m, const rollout::State& s, SimRenderer& sim_r, const ShaderLook& look,
                    const rollout::Model* other1, const rollout::Model* other2, RenderPlanes& out, bool all) {
  const std::size_t n = sz(s.size) * sz(s.size) * 4;
  for (auto& p : out.p) p.assign(n, 0.f);
  fine::render(m, s, out.p[kLearned]);
  sim_r.render(s, out.p[kSim]);
  if (!all) return;
  field_shader(m, s, look, out.p[kShader]);
  const auto other = [&](const rollout::Model* o, std::vector<float>& plane) {
    if (!o) return;
    if (o->h.channels() != m.h.channels() || o->h.res != m.h.res) throw std::invalid_argument("render_experts: models differ in their coarse state");
    fine::render(*o, s, plane);
  };
  other(other1, out.p[kOther1]);
  other(other2, out.p[kOther2]);
}

void RenderMixer::freeze() {
  net_.freeze(true);
  text_ = net_.serialise();
  compact_ = std::make_shared<const CompactValueNet<float>>(text_);
}

RenderMixer RenderMixer::load(RenderMixConfig cfg, const std::string& text) {
  RenderMixer r(std::move(cfg));
  r.net_.freeze(true);
  r.text_ = text;
  r.compact_ = std::make_shared<const CompactValueNet<float>>(text);
  if (r.compact_->inputs() != static_cast<int>(r.cfg_.experts.size())) throw std::invalid_argument("RenderMixer::load: inputs differ from the config");
  return r;
}

void render_mixed(const RenderMixer& mix, const rollout::Model& m, const rollout::State& s, const RenderPlanes& planes, std::span<float> rgba) {
  const CompactValueNet<float>* compact = mix.compact();
  if (!compact) throw std::logic_error("render_mixed: freeze the mixer first");
  const int S = s.size;
  const float it = 1.f / m.render_scale[0], id = 1.f / m.render_scale[1];
  const int age = age_bin(s.since_start * m.fps);
  const auto& ex = mix.config().experts;
  std::vector<double> x(ex.size(), 0.0);
  for (int y = 0; y < S; ++y) {
    for (int xx = 0; xx < S; ++xx) {
      const std::size_t fi = sz(y) * sz(S) + sz(xx), pi = (sz(S - 1 - y) * sz(S) + sz(xx)) * 4;
      const int heat = level_bin(s.fine_t[fi] * it), soot = level_bin(s.fine_d[fi] * id);
      for (int c = 0; c < 4; ++c) {
        for (std::size_t k = 0; k < ex.size(); ++k) x[k] = planes.p[sz(ex[k])][pi + sz(c)];
        const std::array<int, 6> ctx{c * kAgeBins + age, c * kLevelBins + heat, c * kLevelBins + soot, c, 0, 0};
        rgba[pi + sz(c)] = static_cast<float>(std::clamp(compact->predict(x, ctx, {}).mu, 0.0, 1.0));
      }
    }
  }
}

// --- G5a ----------------------------------------------------------------------------------------------------------

std::string_view critic_feature_name(int i) {
  static constexpr std::array<std::string_view, kCriticFeatures> n{
      "log_coverage", "log_emission", "log_motion", "band_1_2", "band_3_4", "band_5_8", "band_9_16", "band_17_32",
      "band_33_64", "centroid_x", "centroid_y", "spread_x", "spread_y", "coverage_trend", "motion_trend", "log_opacity"};
  return i >= 0 && i < kCriticFeatures ? n[sz(i)] : "?";
}

std::array<double, kCriticFeatures> critic_features(const Clip& w) {
  std::array<double, kCriticFeatures> f{};
  const metrics::ClipStats st = metrics::stats(w);
  const int F = w.frames, S = w.size;
  const double cov = std::accumulate(st.coverage.begin(), st.coverage.end(), 0.0) / std::max(1, F);
  const double emi = std::accumulate(st.emission.begin(), st.emission.end(), 0.0) / std::max(1, F);
  f[0] = std::log(cov + 1e-4);
  f[1] = std::log(emi + 1e-4);
  f[2] = std::log(st.motion + 1e-5);
  const int bins = static_cast<int>(st.spectrum.size());
  for (int b = 0, lo = 1; b < 6; ++b, lo *= 2) {  // radii [1, 2], [3, 4], [5, 8], ...
    const int a = b == 0 ? 1 : lo + 1, z = b == 0 ? 2 : 2 * lo;
    double acc = 0;
    int n = 0;
    for (int r = a; r <= z && r <= bins; ++r, ++n) acc += st.spectrum[sz(r - 1)];
    f[sz(3 + b)] = n ? acc / n : 0.0;
  }
  double sx = 0, sy = 0, sxx = 0, syy = 0, sa = 0, op = 0, opn = 0;
  for (int t = 0; t < F; ++t) {
    const auto fr = w.frame(t);
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) {
        const double a = fr[(sz(y) * sz(S) + sz(x)) * 4 + 3] / 255.0;
        if (a <= 0) continue;
        const double X = (x + 0.5) / S, Y = 1.0 - (y + 0.5) / S;
        sa += a;
        sx += a * X;
        sy += a * Y;
        sxx += a * X * X;
        syy += a * Y * Y;
        if (a > 4.0 / 255.0) {
          op += a;
          opn += 1;
        }
      }
    }
  }
  if (sa > 0) {
    const double mx = sx / sa, my = sy / sa;
    f[9] = mx;
    f[10] = my;
    f[11] = std::sqrt(std::max(0.0, sxx / sa - mx * mx));
    f[12] = std::sqrt(std::max(0.0, syy / sa - my * my));
  } else {
    f[9] = f[10] = 0.5;
  }
  const int third = std::max(1, F / 3);
  double c0 = 0, c1 = 0;
  for (int t = 0; t < third; ++t) {
    c0 += st.coverage[sz(t)] / third;
    c1 += st.coverage[sz(F - 1 - t)] / third;
  }
  f[13] = std::log((c1 + 1e-4) / (c0 + 1e-4));
  double m0 = 0, m1 = 0;
  int n0 = 0, n1 = 0;
  for (int t = 1; t < F; ++t) {
    const auto a = w.frame(t), b = w.frame(t - 1);
    double d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) d += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    d /= 255.0 * static_cast<double>(a.size());
    if (t < F / 2) {
      m0 += d;
      ++n0;
    } else {
      m1 += d;
      ++n1;
    }
  }
  f[14] = std::log((m1 / std::max(1, n1) + 1e-5) / (m0 / std::max(1, n0) + 1e-5));
  f[15] = std::log((opn > 0 ? op / opn : 0.0) + 1e-4);
  return f;
}

namespace {
std::array<double, 6> terms(std::span<const float> c) {
  const double i = c[0], w = c[1], t = c[2];
  return {1.0, i, w, t, (w - 0.5) * (w - 0.5), i * t};
}
int tercile(float v) { return v < 1.f / 3.f ? 0 : (v < 2.f / 3.f ? 1 : 2); }
MixerNetSpec critic_spec() {
  MixerNetSpec sp;
  sp.context_sizes = {1, 3, 3};
  sp.final_contexts = 1;
  sp.apm_contexts = 1;
  sp.lr1 = 0.004;
  sp.lr2 = 0.004;
  sp.anneal = 500.0;
  sp.apm_rate = 0.02;
  sp.apm_weight = 0.5;
  return sp;
}
}  // namespace

ShardCritic::ShardCritic() : net_(std::make_shared<MixerNet>(kCriticFeatures + 1, critic_spec())) {}

void ShardCritic::fit(std::span<const CriticSample> fit_set, std::span<const CriticSample> train, int passes, std::uint64_t seed) {
  // per feature and class: least squares of the feature on the control terms (normal equations, a small ridge)
  for (int j = 0; j < kCriticFeatures; ++j) {
    for (int cls = 0; cls < 2; ++cls) {
      std::array<std::array<double, kTerms + 1>, kTerms> A{};
      int n = 0;
      for (const CriticSample& s : fit_set) {
        if (s.real != (cls == 1)) continue;
        const auto t = terms(s.controls);
        for (int a = 0; a < kTerms; ++a) {
          for (int b = 0; b < kTerms; ++b) A[sz(a)][sz(b)] += t[sz(a)] * t[sz(b)];
          A[sz(a)][kTerms] += t[sz(a)] * s.f[sz(j)];
        }
        ++n;
      }
      if (n < kTerms + 2) throw std::invalid_argument("ShardCritic: too few windows of a class");
      for (int a = 0; a < kTerms; ++a) A[sz(a)][sz(a)] += 1e-6 * n;
      for (int col = 0; col < kTerms; ++col) {  // Gauss-Jordan with partial pivoting
        int piv = col;
        for (int r = col + 1; r < kTerms; ++r) {
          if (std::abs(A[sz(r)][sz(col)]) > std::abs(A[sz(piv)][sz(col)])) piv = r;
        }
        std::swap(A[sz(col)], A[sz(piv)]);
        const double d = A[sz(col)][sz(col)];
        for (int c2 = col; c2 <= kTerms; ++c2) A[sz(col)][sz(c2)] /= d;
        for (int r = 0; r < kTerms; ++r) {
          if (r == col) continue;
          const double fct = A[sz(r)][sz(col)];
          for (int c2 = col; c2 <= kTerms; ++c2) A[sz(r)][sz(c2)] -= fct * A[sz(col)][sz(c2)];
        }
      }
      Gauss& g = g_[sz(j)][sz(cls)];
      for (int a = 0; a < kTerms; ++a) g.beta[sz(a)] = A[sz(a)][kTerms];
      double ss = 0;
      for (const CriticSample& s : fit_set) {
        if (s.real != (cls == 1)) continue;
        const auto t = terms(s.controls);
        double mu = 0;
        for (int a = 0; a < kTerms; ++a) mu += g.beta[sz(a)] * t[sz(a)];
        ss += (s.f[sz(j)] - mu) * (s.f[sz(j)] - mu);
      }
      g.sigma = std::max(1e-4, std::sqrt(ss / n));
    }
  }
  // the mixer, on other windows: shuffled passes, rates halved after each, then frozen
  net_ = std::make_shared<MixerNet>(kCriticFeatures + 1, critic_spec());
  std::vector<std::size_t> order(train.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::mt19937_64 rng(seed);
  for (int p = 0; p < passes; ++p) {
    std::shuffle(order.begin(), order.end(), rng);
    for (const std::size_t i : order) {
      (void)p_real(train[i].f, train[i].controls);
      net_->update(train[i].real ? 1 : 0);
    }
    net_->scale_lr(0.5);
  }
  net_->freeze(true);
  mixer_text_ = net_->serialise();
  compact_ = std::make_shared<const CompactMixer<double>>(mixer_text_);
}

std::array<double, kCriticFeatures> ShardCritic::experts(const std::array<double, kCriticFeatures>& f, std::span<const float> controls) const {
  std::array<double, kCriticFeatures> e{};
  const auto t = terms(controls);
  for (int j = 0; j < kCriticFeatures; ++j) {
    double ll[2];
    for (int cls = 0; cls < 2; ++cls) {
      const Gauss& g = g_[sz(j)][sz(cls)];
      double mu = 0;
      for (int a = 0; a < kTerms; ++a) mu += g.beta[sz(a)] * t[sz(a)];
      const double z = (f[sz(j)] - mu) / g.sigma;
      ll[cls] = -0.5 * z * z - std::log(g.sigma);
    }
    e[sz(j)] = std::clamp(ll[1] - ll[0], -8.0, 8.0);
  }
  return e;
}

double ShardCritic::p_real(const std::array<double, kCriticFeatures>& f, std::span<const float> controls) {
  const auto e = experts(f, controls);
  std::array<double, kCriticFeatures + 1> x{};
  std::copy(e.begin(), e.end(), x.begin());
  x.back() = 1.0;
  const std::array<int, 5> ctx{0, tercile(controls[0]), tercile(controls[2]), 0, 0};
  return compact_ ? compact_->predict(x, ctx) : net_->predict(x, ctx);
}

ShardCritic ShardCritic::load(const std::string& text) {
  ShardCritic c;
  std::size_t at = text.find('\n');
  if (text.substr(0, at) != "nvfx-shard-critic v1") throw std::runtime_error("ShardCritic::load: not a critic");
  for (int j = 0; j < kCriticFeatures; ++j) {
    for (int cls = 0; cls < 2; ++cls) {
      const std::size_t e = text.find('\n', at + 1);
      std::istringstream line(text.substr(at + 1, e - at - 1));
      std::string name;
      int k = -1;
      line >> name >> k;
      Gauss& g = c.g_[sz(j)][sz(cls)];
      for (double& b : g.beta) line >> b;
      line >> g.sigma;
      if (!line || name != critic_feature_name(j) || k != cls) throw std::runtime_error("ShardCritic::load: malformed expert line");
      at = e;
    }
  }
  c.mixer_text_ = text.substr(at + 1);
  c.compact_ = std::make_shared<const CompactMixer<double>>(c.mixer_text_);
  return c;
}

std::string ShardCritic::serialise() const {
  std::string s = "nvfx-shard-critic v1\n";
  for (int j = 0; j < kCriticFeatures; ++j) {
    for (int cls = 0; cls < 2; ++cls) {
      const Gauss& g = g_[sz(j)][sz(cls)];
      s += std::format("{} {}", critic_feature_name(j), cls);
      for (const double b : g.beta) s += std::format(" {:.17g}", b);
      s += std::format(" {:.17g}\n", g.sigma);
    }
  }
  return s + (compact_ ? mixer_text_ : net_->serialise());
}
std::string ShardCritic::version() const { return sha256_hex(serialise()); }

int runtime_start(const rollout::Model& m, std::span<const float> controls, std::uint64_t seed, std::int64_t shard) {
  std::array<int, 3> best{-1, -1, -1};
  std::array<float, 3> dist{1e30f, 1e30f, 1e30f};
  for (std::size_t i = 0; i < m.starts.size(); ++i) {
    float d = 0.f;
    for (std::size_t k = 0; k < m.starts[i].controls.size() && k < controls.size(); ++k) {
      const float e = m.starts[i].controls[k] - controls[k];
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
  const std::uint32_t h = hash_cell(static_cast<std::int32_t>(shard), static_cast<std::int32_t>(shard >> 31), 17, seed);
  return best[h % static_cast<std::uint32_t>(count)];
}

std::uint64_t runtime_shard_seed(std::uint64_t seed, std::int64_t shard) {
  return shard == 0 ? seed : seed ^ (0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(shard + 1));
}

Candidate shard_candidate(const rollout::Model& m, std::span<const float> controls, std::uint64_t instance_seed, std::int64_t shard, int j) {
  if (j == 0) return {runtime_start(m, controls, instance_seed, shard), runtime_shard_seed(instance_seed, shard)};
  const std::uint64_t s = splitmix64(runtime_shard_seed(instance_seed, shard) + static_cast<std::uint64_t>(j));
  return {runtime_start(m, controls, s, shard), s};
}

// --- G5b ----------------------------------------------------------------------------------------------------------

CoarseSolver::CoarseSolver(sim::Effect effect, std::span<const float> controls, std::uint64_t seed, int res, float fps) : res_(res), fps_(fps) {
  sim::Params p;
  p.effect = effect;
  p.intensity = controls[0];
  p.wind = controls[1];
  p.turbulence = controls[2];
  p.seed = seed;
  p.size = 128;
  p.sim_res = res;
  p.fps = fps;
  fluid_ = std::make_unique<sim::Fluid>(p);
  const std::size_t n = sz(res) * sz(res);
  st_.n = res;
  st_.u.assign(n, 0.f);
  st_.v.assign(n, 0.f);
  st_.temp.assign(n, 0.f);
  st_.soot.assign(n, 0.f);
  st_.pressure.assign(n, 0.f);
}
CoarseSolver::~CoarseSolver() = default;

void CoarseSolver::step(const rollout::Model& m, const rollout::State& s, std::span<float> out) {
  const int C = m.h.channels();
  if (m.h.res != res_) throw std::invalid_argument("CoarseSolver: resolution mismatch");
  for (int i = 0; i < res_ * res_; ++i) {
    const float* x = s.coarse.data() + sz(i) * sz(C);
    st_.u[sz(i)] = x[0] * fps_;
    st_.v[sz(i)] = x[1] * fps_;
    st_.temp[sz(i)] = x[2];
    st_.soot[sz(i)] = x[3];
  }
  st_.time = s.time;
  st_.frame = static_cast<int>(std::lround(s.time * fps_));
  fluid_->set_state(st_);
  fluid_->step_frame();
  const sim::State nx = fluid_->state();
  st_.pressure = nx.pressure;
  for (int i = 0; i < res_ * res_; ++i) {
    float* o = out.data() + sz(i) * rollout::kPhys;
    o[0] = nx.u[sz(i)] / fps_;
    o[1] = nx.v[sz(i)] / fps_;
    o[2] = nx.temp[sz(i)];
    o[3] = nx.soot[sz(i)];
  }
}

namespace {
ValueNetSpec update_spec(double lr, double anneal) {
  ValueNetSpec sp;
  sp.context_sizes = {4 * kHeatBins, 4 * kBands, 4 * kAges};
  sp.final_contexts = 4;
  sp.avm_contexts = 0;
  sp.scale_contexts = 1;
  sp.loss = ValueLoss::squared;
  sp.lr1 = lr;
  sp.lr2 = lr / 2;
  sp.anneal = anneal;
  sp.eps = 1e-4;
  sp.limit = 1e3;
  return sp;
}
}  // namespace

UpdateMixer::UpdateMixer(double lr, double anneal) : net_(3, 0, update_spec(lr, anneal)) { net_.set_rule(0); }

UpdateMixer UpdateMixer::blend(double a) {
  UpdateMixer u;
  const std::array<double, 3> w{1.0 - a, a, 0.0};
  u.net_.set_weights(w);
  return u;
}

double UpdateMixer::predict(double dn, double ds, int channel, int heat, int band, int age) {
  const std::array<double, 3> x{dn, ds, 1.0};
  const std::array<int, 6> ctx{channel * kHeatBins + heat, channel * kBands + band, channel * kAges + age, channel, 0, 0};
  return compact_ ? compact_->predict(x, ctx, {}).mu : net_.predict(x, ctx, {}).mu;
}

void UpdateMixer::freeze() {
  net_.freeze(true);
  text_ = net_.serialise();
  compact_ = std::make_shared<const CompactValueNet<double>>(text_);
}
void UpdateMixer::unfreeze() {
  if (!text_.empty() && text_ != net_.serialise()) throw std::logic_error("UpdateMixer::unfreeze: a loaded mixer cannot learn");
  net_.freeze(false);
  compact_.reset();
  text_.clear();
}
UpdateMixer UpdateMixer::load(const std::string& text) {
  UpdateMixer u;
  u.net_.freeze(true);
  u.text_ = text;
  u.compact_ = std::make_shared<const CompactValueNet<double>>(text);
  if (u.compact_->inputs() != 3) throw std::invalid_argument("UpdateMixer::load: not an update mixer");
  return u;
}

std::array<double, 3> UpdateMixer::mean_weights(int channel) const {
  std::array<double, 3> w{};
  const auto fw = net_.final_mixer().weights(channel);
  for (int k = 0; k < net_.first_layer(); ++k) {
    const ValueMixer& mx = net_.mixer(k);
    const int per = mx.contexts() / 4;
    for (int b = 0; b < per; ++b) {
      const auto ww = mx.weights(channel * per + b);
      for (std::size_t i = 0; i < 3; ++i) w[i] += fw[sz(k)] * ww[i] / per;
    }
  }
  return w;
}

CellCtx cell_context(const rollout::Model& m, const rollout::State& s, int cell) {
  const int R = m.h.res, C = m.h.channels();
  const float h = s.coarse[sz(cell) * sz(C) + 2] / m.scale[2];
  CellCtx c;
  c.heat = h < 0.02f ? 0 : (h < 0.3f ? 1 : (h < 1.f ? 2 : 3));
  c.band = std::min(kBands - 1, (cell / R) * kBands / R);
  return c;
}

int update_age(float f) { return f <= 8.5f ? 0 : (f <= 30.5f ? 1 : (f <= 60.5f ? 2 : 3)); }

void mixed_step(const rollout::Model& m, rollout::State& s, std::span<const float> controls, std::uint64_t seed, CoarseSolver& solver,
                UpdateMixer& mix, std::vector<UpdateRow>* rows, bool detail_layer) {
  const rollout::Hyper& h = m.h;
  const int R = h.res, C = h.channels();
  std::vector<float> cond(sz(h.cond())), noise(sz(R) * sz(R) * rollout::kNoise), next(s.coarse.size()),
      solv(sz(R) * sz(R) * rollout::kPhys);
  rollout::condition(m, controls, s.time, cond);
  rollout::coarse_noise(m, seed, s.time, noise);
  rollout::coarse_step(m, s.coarse, noise, cond, s.pressure, next, s.flow);
  solver.step(m, s, solv);
  const int age = update_age(s.since_start * m.fps);
  for (int i = 0; i < R * R; ++i) {
    const CellCtx cx = cell_context(m, s, i);
    for (int k = 0; k < rollout::kPhys; ++k) {
      const std::size_t j = sz(i) * sz(C) + sz(k);
      const double x = s.coarse[j];
      const double dn = static_cast<double>(next[j]) - x, ds = static_cast<double>(solv[sz(i) * rollout::kPhys + sz(k)]) - x;
      const double y = mix.predict(dn, ds, k, cx.heat, cx.band, age);
      const float v = std::clamp(static_cast<float>(x + y), m.lo[sz(k)], m.hi[sz(k)]);
      if (rows) {
        rows->push_back({static_cast<float>(dn), static_cast<float>(ds), static_cast<float>(x), static_cast<std::uint8_t>(k),
                         static_cast<std::uint8_t>(cx.heat), static_cast<std::uint8_t>(cx.band), static_cast<std::uint8_t>(age)});
      }
      if (k < 2) s.flow[sz(i) * 2 + sz(k)] += v - next[j];
      next[j] = v;
    }
  }
  s.coarse.swap(next);
  if (detail_layer) rollout::detail_step(m, s, seed, controls);
  s.time += 1.f / m.fps;
  s.since_start += 1.f / m.fps;
}

}  // namespace nfx::dcm::extras
