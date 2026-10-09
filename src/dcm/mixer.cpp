// PAQ-style context mixing (include/neuralfx/dcm/mixer.hpp). Mixer, APM, MixerNet and sha256_hex are ported from the
// owner's CameraDetector, cabinlab/src/diffusion/mixer.cpp, with the same arithmetic in the same order; the value
// domain (ValueMixer, AVM, ValueNet, laplace_bits) is new here.
#include <neuralfx/dcm/mixer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <stdexcept>

namespace nfx::dcm {

double stretch(double p) {
  p = std::clamp(p, 1e-6, 1.0 - 1e-6);
  return std::log(p / (1.0 - p));
}

double squash(double x) {
  x = std::clamp(x, -40.0, 40.0);
  return 1.0 / (1.0 + std::exp(-x));
}

namespace {

void append_values(std::string& out, std::span<const double> v) {
  char buf[32];
  for (const double d : v) {
    std::snprintf(buf, sizeof(buf), "%.9e,", d);
    out += buf;
  }
  out += '\n';
}

}  // namespace

// --- Mixer ----------------------------------------------------------------------------------------------------------

Mixer::Mixer(int n_inputs, int n_contexts, double lr, double init) : n_(n_inputs), k_(n_contexts), lr_(lr) {
  if (n_inputs <= 0 || n_contexts <= 0) throw std::invalid_argument("dcm: mixer needs inputs and contexts");
  w_.assign(static_cast<std::size_t>(n_) * static_cast<std::size_t>(k_), init);
  uses_.assign(static_cast<std::size_t>(k_), 0);
  x_.assign(static_cast<std::size_t>(n_), 0.0);
}

double Mixer::mix(std::span<const double> x, int context) {
  if (static_cast<int>(x.size()) != n_) throw std::invalid_argument("dcm: mixer input size");
  if (context < 0 || context >= k_) throw std::out_of_range(std::format("dcm: mixer context {} of {}", context, k_));
  ctx_ = context;
  std::copy(x.begin(), x.end(), x_.begin());
  const double* w = w_.data() + static_cast<std::size_t>(ctx_) * static_cast<std::size_t>(n_);
  double dot = 0.0;
  for (int i = 0; i < n_; ++i) dot += w[i] * x_[static_cast<std::size_t>(i)];
  last_ = dot;
  return dot;
}

void Mixer::mix_all(std::span<Mixer> mixers, std::span<const double> x, std::span<const int> contexts,
                    std::span<double> out, bool remember_inputs) {
  const std::size_t m = mixers.size(), n = x.size();
  if (contexts.size() != m || out.size() != m) throw std::invalid_argument("dcm: mix_all needs a context and an output per mixer");
  for (std::size_t k = 0; k < m; ++k) {
    Mixer& mx = mixers[k];
    if (static_cast<int>(n) != mx.n_) throw std::invalid_argument("dcm: mixer input size");
    if (contexts[k] < 0 || contexts[k] >= mx.k_) {
      throw std::out_of_range(std::format("dcm: mixer context {} of {}", contexts[k], mx.k_));
    }
    mx.ctx_ = contexts[k];
    if (remember_inputs) std::copy(x.begin(), x.end(), mx.x_.begin());
  }
  const double* xp = x.data();
  const auto weights_of = [&](std::size_t j) { return mixers[j].w_.data() + static_cast<std::size_t>(mixers[j].ctx_) * n; };
  const auto finish = [&](std::size_t j, double dot) {
    mixers[j].last_ = dot;
    out[j] = dot;
  };
  std::size_t k = 0;
  for (; k + 4 <= m; k += 4) {  // four independent sums, each in input order: the same bits as mix()
    const double *w0 = weights_of(k), *w1 = weights_of(k + 1), *w2 = weights_of(k + 2), *w3 = weights_of(k + 3);
    double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double xi = xp[i];
      d0 += w0[i] * xi;
      d1 += w1[i] * xi;
      d2 += w2[i] * xi;
      d3 += w3[i] * xi;
    }
    finish(k, d0);
    finish(k + 1, d1);
    finish(k + 2, d2);
    finish(k + 3, d3);
  }
  for (; k + 2 <= m; k += 2) {
    const double *w0 = weights_of(k), *w1 = weights_of(k + 1);
    double d0 = 0.0, d1 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      d0 += w0[i] * xp[i];
      d1 += w1[i] * xp[i];
    }
    finish(k, d0);
    finish(k + 1, d1);
  }
  for (; k < m; ++k) {
    const double* w0 = weights_of(k);
    double d0 = 0.0;
    for (std::size_t i = 0; i < n; ++i) d0 += w0[i] * xp[i];
    finish(k, d0);
  }
}

void Mixer::update_all(std::span<Mixer> mixers, std::span<const double> x, int y) {
  for (Mixer& m : mixers) {
    if (m.frozen_) continue;
    if (static_cast<int>(x.size()) != m.n_) throw std::invalid_argument("dcm: mixer input size");
    const double err = static_cast<double>(y) - squash(m.last_);
    auto& uses = m.uses_[static_cast<std::size_t>(m.ctx_)];
    const double lr = m.lr_ / (1.0 + static_cast<double>(uses) / m.anneal_);
    ++uses;
    double* w = m.w_.data() + static_cast<std::size_t>(m.ctx_) * static_cast<std::size_t>(m.n_);
    for (int i = 0; i < m.n_; ++i) w[i] += lr * err * x[static_cast<std::size_t>(i)];
  }
}

void Mixer::update(int y) {
  if (frozen_) return;
  const double err = static_cast<double>(y) - squash(last_);
  auto& uses = uses_[static_cast<std::size_t>(ctx_)];
  const double lr = lr_ / (1.0 + static_cast<double>(uses) / anneal_);
  ++uses;
  double* w = w_.data() + static_cast<std::size_t>(ctx_) * static_cast<std::size_t>(n_);
  for (int i = 0; i < n_; ++i) w[i] += lr * err * x_[static_cast<std::size_t>(i)];
}

std::span<const double> Mixer::weights(int context) const {
  return {w_.data() + static_cast<std::size_t>(context) * static_cast<std::size_t>(n_), static_cast<std::size_t>(n_)};
}

std::span<double> Mixer::weights(int context) {
  return {w_.data() + static_cast<std::size_t>(context) * static_cast<std::size_t>(n_), static_cast<std::size_t>(n_)};
}

void Mixer::serialise(std::string& out) const {
  out += std::format("mixer {} {}\n", n_, k_);
  append_values(out, w_);
}

// --- APM ------------------------------------------------------------------------------------------------------------

APM::APM(int n_contexts, double rate) : k_(std::max(1, n_contexts)), rate_(rate) {
  t_.resize(static_cast<std::size_t>(k_) * 33);
  for (int c = 0; c < k_; ++c) {
    for (int j = 0; j < 33; ++j) t_[static_cast<std::size_t>(c * 33 + j)] = squash((j - 16) / 2.0);
  }
}

double APM::refine(double p, int context) {
  context = std::clamp(context, 0, k_ - 1);
  const double s = std::clamp(stretch(p), -7.999, 7.999);
  const double pos = (s + 8.0) * 2.0;
  const auto j = static_cast<std::size_t>(pos);
  frac_ = pos - static_cast<double>(j);
  lo_ = static_cast<std::size_t>(context) * 33 + j;
  return t_[lo_] * (1.0 - frac_) + t_[lo_ + 1] * frac_;
}

void APM::update(int y) {
  if (frozen_) return;
  const double target = static_cast<double>(y);
  t_[lo_] += rate_ * (1.0 - frac_) * (target - t_[lo_]);
  t_[lo_ + 1] += rate_ * frac_ * (target - t_[lo_ + 1]);
}

void APM::serialise(std::string& out) const {
  out += std::format("apm {}\n", k_);
  append_values(out, t_);
}

// --- MixerNet -------------------------------------------------------------------------------------------------------

MixerNet::MixerNet(int n_inputs, const MixerNetSpec& spec)
    : spec_(spec),
      final_(std::max<int>(1, static_cast<int>(spec.context_sizes.size())), std::max(1, spec.final_contexts), spec.lr2,
             1.0 / std::max<double>(1.0, static_cast<double>(spec.context_sizes.size()))),
      apm_(std::max(1, spec.apm_contexts), spec.apm_rate) {
  if (spec.context_sizes.empty()) throw std::invalid_argument("dcm: MixerNet needs at least one first-layer mixer");
  for (const int k : spec.context_sizes) {
    layer1_.emplace_back(n_inputs, std::max(1, k), spec.lr1, 1.0 / n_inputs);
    layer1_.back().set_anneal(spec.anneal);
  }
  final_.set_anneal(spec.anneal);
  h_.assign(layer1_.size(), 0.0);
  x_.assign(static_cast<std::size_t>(n_inputs), 0.0);  // as each Mixer's own copy starts: an update before a predict reads zeros
}

double MixerNet::predict(std::span<const double> x, std::span<const int> contexts) {
  const std::size_t need = layer1_.size() + 2;
  if (contexts.size() != need) throw std::invalid_argument(std::format("dcm: MixerNet needs {} contexts", need));
  x_.assign(x.begin(), x.end());
  Mixer::mix_all(layer1_, x_, contexts.first(layer1_.size()), h_, false);
  for (double& h : h_) h = std::clamp(h, -16.0, 16.0);
  const double z = final_.mix(h_, contexts[layer1_.size()]);
  p_mix_ = squash(z);
  if (spec_.apm_contexts > 0 && spec_.apm_weight > 0.0) {
    const double pa = apm_.refine(p_mix_, contexts[layer1_.size() + 1]);
    p_out_ = (1.0 - spec_.apm_weight) * p_mix_ + spec_.apm_weight * pa;
  } else {
    p_out_ = p_mix_;
  }
  return p_out_;
}

void MixerNet::update(int y) {
  if (frozen_) return;
  Mixer::update_all(layer1_, x_, y);
  final_.update(y);
  if (spec_.apm_contexts > 0 && spec_.apm_weight > 0.0) apm_.update(y);
}

void MixerNet::freeze(bool on) noexcept {
  frozen_ = on;
  for (auto& m : layer1_) m.freeze(on);
  final_.freeze(on);
  apm_.freeze(on);
}

void MixerNet::scale_lr(double factor) noexcept {
  spec_.lr1 *= factor;
  spec_.lr2 *= factor;
  for (auto& m : layer1_) m.set_lr(spec_.lr1);
  final_.set_lr(spec_.lr2);
}

void MixerNet::restart_learning(const MixerNetSpec& spec) noexcept {
  spec_.lr1 = spec.lr1;
  spec_.lr2 = spec.lr2;
  for (auto& m : layer1_) {
    m.set_lr(spec_.lr1);
    m.reset_counts();
  }
  final_.set_lr(spec_.lr2);
  final_.reset_counts();
}

std::string MixerNet::serialise() const {
  std::string out = "nvfx-paq-mixer v1\n";
  for (const auto& m : layer1_) m.serialise(out);
  final_.serialise(out);
  if (spec_.apm_contexts > 0 && spec_.apm_weight > 0.0) {
    out += std::format("apm_weight {:.6f}\n", spec_.apm_weight);
    apm_.serialise(out);
  }
  return out;
}

std::string MixerNet::version() const { return sha256_hex(serialise()); }

// --- SHA-256 (FIPS 180-4) -------------------------------------------------------------------------------------------

std::string sha256_hex(std::string_view bytes) {
  static constexpr std::array<std::uint32_t, 64> k = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
      0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
      0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
      0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
      0xc67178f2};
  std::array<std::uint32_t, 8> h = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  const auto rotr = [](std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  std::string msg(bytes);
  const std::uint64_t bit_len = static_cast<std::uint64_t>(bytes.size()) * 8;
  msg += static_cast<char>(0x80);
  while (msg.size() % 64 != 56) msg += static_cast<char>(0);
  for (int i = 7; i >= 0; --i) msg += static_cast<char>((bit_len >> (8 * i)) & 0xff);
  for (std::size_t chunk = 0; chunk < msg.size(); chunk += 64) {
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16; ++i) {
      const auto* p = reinterpret_cast<const unsigned char*>(msg.data() + chunk + 4 * i);
      w[i] = (std::uint32_t{p[0]} << 24) | (std::uint32_t{p[1]} << 16) | (std::uint32_t{p[2]} << 8) | std::uint32_t{p[3]};
    }
    for (std::size_t i = 16; i < 64; ++i) {
      const auto s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const auto s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto a = h;
    for (std::size_t i = 0; i < 64; ++i) {
      const auto S1 = rotr(a[4], 6) ^ rotr(a[4], 11) ^ rotr(a[4], 25);
      const auto ch = (a[4] & a[5]) ^ (~a[4] & a[6]);
      const auto t1 = a[7] + S1 + ch + k[i] + w[i];
      const auto S0 = rotr(a[0], 2) ^ rotr(a[0], 13) ^ rotr(a[0], 22);
      const auto mj = (a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]);
      const auto t2 = S0 + mj;
      a = {t1 + t2, a[0], a[1], a[2], a[3] + t1, a[4], a[5], a[6]};
    }
    for (std::size_t i = 0; i < 8; ++i) h[i] += a[i];
  }
  std::string out;
  for (const auto v : h) out += std::format("{:08x}", v);
  return out;
}

// --- the value domain -----------------------------------------------------------------------------------------------

std::string_view loss_name(ValueLoss loss) {
  switch (loss) {
    case ValueLoss::squared: return "squared";
    case ValueLoss::laplace: return "laplace";
  }
  return "?";
}

double laplace_bits(double y, double mu, double b, double delta) {
  if (!(b > 0.0) || !(delta > 0.0)) throw std::invalid_argument("dcm: laplace_bits needs b > 0 and delta > 0");
  constexpr double kFloor = -690.7755278982137;  // ln(1e-300)
  const double lo = y - delta / 2.0, hi = y + delta / 2.0;
  double log_mass = 0.0;
  if (lo >= mu || hi <= mu) {
    // the bin lies on one side of the centre: mass = ½ e^(−near / b) (1 − e^(−delta / b)), near = distance to its
    // nearer edge, in logs so that far tails do not underflow
    const double near = lo >= mu ? lo - mu : mu - hi;
    log_mass = -std::numbers::ln2 - near / b + std::log(-std::expm1(-delta / b));
  } else {
    // the bin holds the centre: mass = 1 − ½ e^(−(mu − lo) / b) − ½ e^(−(hi − mu) / b), without cancellation
    const double mass = -0.5 * (std::expm1(-(mu - lo) / b) + std::expm1(-(hi - mu) / b));
    log_mass = mass > 0.0 ? std::log(mass) : kFloor;
  }
  return -std::max(log_mass, kFloor) / std::numbers::ln2;
}

// --- ValueMixer -----------------------------------------------------------------------------------------------------

ValueMixer::ValueMixer(int n_inputs, int n_contexts, double lr, double init, double eps)
    : n_(n_inputs), k_(n_contexts), lr_(lr), eps_(eps) {
  if (n_inputs <= 0 || n_contexts <= 0) throw std::invalid_argument("dcm: value mixer needs inputs and contexts");
  if (!(eps > 0.0)) throw std::invalid_argument("dcm: value mixer needs eps > 0");
  w_.assign(static_cast<std::size_t>(n_) * static_cast<std::size_t>(k_), init);
  uses_.assign(static_cast<std::size_t>(k_), 0);
  x_.assign(static_cast<std::size_t>(n_), 0.0);
}

double ValueMixer::mix(std::span<const double> x, int context) {
  if (static_cast<int>(x.size()) != n_) throw std::invalid_argument("dcm: value mixer input size");
  if (context < 0 || context >= k_) throw std::out_of_range(std::format("dcm: mixer context {} of {}", context, k_));
  ctx_ = context;
  std::copy(x.begin(), x.end(), x_.begin());
  const double* w = w_.data() + static_cast<std::size_t>(ctx_) * static_cast<std::size_t>(n_);
  double dot = 0.0;
  for (int i = 0; i < n_; ++i) dot += w[i] * x_[static_cast<std::size_t>(i)];
  last_ = dot;
  return dot;
}

void ValueMixer::update(double g) {
  if (frozen_) return;
  double norm = eps_;
  for (const double v : x_) norm += v * v;
  auto& uses = uses_[static_cast<std::size_t>(ctx_)];
  const double lr = lr_ / (1.0 + static_cast<double>(uses) / anneal_);
  ++uses;
  const double step = lr * g / norm;
  double* w = w_.data() + static_cast<std::size_t>(ctx_) * static_cast<std::size_t>(n_);
  for (int i = 0; i < n_; ++i) w[i] += step * x_[static_cast<std::size_t>(i)];
}

std::span<const double> ValueMixer::weights(int context) const {
  if (context < 0 || context >= k_) throw std::out_of_range(std::format("dcm: mixer context {} of {}", context, k_));
  return {w_.data() + static_cast<std::size_t>(context) * static_cast<std::size_t>(n_), static_cast<std::size_t>(n_)};
}

std::span<double> ValueMixer::weights(int context) {
  if (context < 0 || context >= k_) throw std::out_of_range(std::format("dcm: mixer context {} of {}", context, k_));
  return {w_.data() + static_cast<std::size_t>(context) * static_cast<std::size_t>(n_), static_cast<std::size_t>(n_)};
}

void ValueMixer::serialise(std::string& out) const {
  out += std::format("mixer {} {}\n", n_, k_);
  append_values(out, w_);
}

// --- AVM ------------------------------------------------------------------------------------------------------------

AVM::AVM(int n_contexts, double lo, double hi, double rate) : k_(std::max(1, n_contexts)), lo_(lo), hi_(hi), rate_(rate) {
  if (!(hi > lo) || !std::isfinite(lo) || !std::isfinite(hi)) throw std::invalid_argument("dcm: AVM needs a finite range lo < hi");
  d_.assign(static_cast<std::size_t>(k_) * kKnots, 0.0);
}

double AVM::offset(double v, int context) {
  context = std::clamp(context, 0, k_ - 1);
  v_ = v;
  double pos = (v - lo_) / (hi_ - lo_) * static_cast<double>(kKnots - 1);
  pos = std::isnan(pos) ? 0.0 : std::clamp(pos, 0.0, static_cast<double>(kKnots - 1) - 0.001);
  const auto j = static_cast<std::size_t>(pos);
  frac_ = pos - static_cast<double>(j);
  at_ = static_cast<std::size_t>(context) * kKnots + j;
  return d_[at_] * (1.0 - frac_) + d_[at_ + 1] * frac_;
}

void AVM::update(double y) {
  if (frozen_) return;
  const double target = y - v_;
  d_[at_] += rate_ * (1.0 - frac_) * (target - d_[at_]);
  d_[at_ + 1] += rate_ * frac_ * (target - d_[at_ + 1]);
}

void AVM::serialise(std::string& out) const {
  out += std::format("avm {} {:.9e} {:.9e}\n", k_, lo_, hi_);
  append_values(out, d_);
}

// --- ValueNet -------------------------------------------------------------------------------------------------------

ValueNet::ValueNet(int n_inputs, int n_scale_features, const ValueNetSpec& spec)
    : spec_(spec),
      n_(n_inputs),
      nz_(n_scale_features),
      final_(std::max<int>(1, static_cast<int>(spec.context_sizes.size())), std::max(1, spec.final_contexts), spec.lr2,
             1.0 / std::max<double>(1.0, static_cast<double>(spec.context_sizes.size())), spec.eps),
      avm_(std::max(1, spec.avm_contexts), spec.avm_lo, spec.avm_hi, spec.avm_rate),
      scale_(1 + std::max(0, n_scale_features), std::max(1, spec.scale_contexts), spec.scale_lr, 0.0, spec.eps) {
  if (spec.context_sizes.empty()) throw std::invalid_argument("dcm: ValueNet needs at least one first-layer mixer");
  if (n_scale_features < 0) throw std::invalid_argument("dcm: ValueNet scale features must be >= 0");
  if (!(spec.limit > 0.0)) throw std::invalid_argument("dcm: ValueNet needs limit > 0");
  if (!(spec.log_b_min < spec.log_b_max)) throw std::invalid_argument("dcm: ValueNet needs log_b_min < log_b_max");
  for (const int k : spec.context_sizes) {
    layer1_.emplace_back(n_inputs, std::max(1, k), spec.lr1, 1.0 / n_inputs, spec.eps);
    layer1_.back().set_anneal(spec.anneal);
  }
  final_.set_anneal(spec.anneal);
  scale_.set_anneal(spec.anneal);
  const double log_b = std::clamp(spec.log_b_init, spec.log_b_min, spec.log_b_max);
  for (int c = 0; c < scale_.contexts(); ++c) scale_.weights(c)[0] = log_b;
  h_.assign(layer1_.size(), 0.0);
  zz_.assign(static_cast<std::size_t>(1 + nz_), 0.0);
  zz_[0] = 1.0;
  out_ = {0.0, std::exp(log_b)};
}

double ValueNet::gradient(double y, double v) const noexcept {
  if (spec_.loss == ValueLoss::squared) return y - v;
  const double sign = y > v ? 1.0 : (y < v ? -1.0 : 0.0);
  return sign / out_.b;
}

ValuePrediction ValueNet::predict(std::span<const double> x, std::span<const int> contexts, std::span<const double> z) {
  const std::size_t m = layer1_.size();
  if (contexts.size() != m + 3) throw std::invalid_argument(std::format("dcm: ValueNet needs {} contexts", m + 3));
  if (z.size() != static_cast<std::size_t>(nz_)) {
    throw std::invalid_argument(std::format("dcm: ValueNet needs {} scale features, got {}", nz_, z.size()));
  }
  for (std::size_t k = 0; k < m; ++k) h_[k] = std::clamp(layer1_[k].mix(x, contexts[k]), -spec_.limit, spec_.limit);
  mu_mix_ = std::clamp(final_.mix(h_, contexts[m]), -spec_.limit, spec_.limit);
  // (1 − w) μ + w (μ + offset), written as μ + w · offset: the same value, and exactly μ while the offsets are 0
  const double mu = avm_on() ? mu_mix_ + spec_.avm_weight * avm_.offset(mu_mix_, contexts[m + 1]) : mu_mix_;
  std::copy(z.begin(), z.end(), zz_.begin() + 1);
  const double log_b = std::clamp(scale_.mix(zz_, contexts[m + 2]), spec_.log_b_min, spec_.log_b_max);
  out_ = {mu, std::exp(log_b)};
  return out_;
}

void ValueNet::update(double y) {
  if (frozen_) return;
  for (auto& m : layer1_) m.update(gradient(y, m.last()));
  final_.update(gradient(y, final_.last()));
  if (avm_on()) avm_.update(y);
  // d(Laplace loss) / d(log b) = 1 − |y − μ| / b: the scale net steps against it
  scale_.update(std::clamp(std::abs(y - out_.mu) / out_.b - 1.0, -1.0, 16.0));
}

void ValueNet::freeze(bool on) noexcept {
  frozen_ = on;
  for (auto& m : layer1_) m.freeze(on);
  final_.freeze(on);
  avm_.freeze(on);
  scale_.freeze(on);
}

void ValueNet::scale_lr(double factor) noexcept {
  spec_.lr1 *= factor;
  spec_.lr2 *= factor;
  for (auto& m : layer1_) m.set_lr(spec_.lr1);
  final_.set_lr(spec_.lr2);
}

void ValueNet::restart_learning(const ValueNetSpec& spec) noexcept {
  spec_.lr1 = spec.lr1;
  spec_.lr2 = spec.lr2;
  spec_.scale_lr = spec.scale_lr;
  for (auto& m : layer1_) {
    m.set_lr(spec_.lr1);
    m.reset_counts();
  }
  final_.set_lr(spec_.lr2);
  final_.reset_counts();
  scale_.set_lr(spec_.scale_lr);
  scale_.reset_counts();
}

void ValueNet::set_weights(std::span<const double> w) {
  if (w.size() != static_cast<std::size_t>(n_)) throw std::invalid_argument("dcm: ValueNet::set_weights needs one weight per input");
  for (auto& m : layer1_) {
    for (int c = 0; c < m.contexts(); ++c) std::ranges::copy(w, m.weights(c).begin());
  }
  for (int c = 0; c < final_.contexts(); ++c) {
    auto f = final_.weights(c);
    std::ranges::fill(f, 0.0);
    f[0] = 1.0;
  }
}

void ValueNet::set_rule(int input) {
  if (input < 0 || input >= n_) throw std::out_of_range(std::format("dcm: ValueNet::set_rule input {} of {}", input, n_));
  std::vector<double> w(static_cast<std::size_t>(n_), 0.0);
  w[static_cast<std::size_t>(input)] = 1.0;
  set_weights(w);
}

double ValueNet::bits(double y, double delta) const { return laplace_bits(y, out_.mu, out_.b, delta); }

std::string ValueNet::serialise() const {
  std::string out = "nvfx-value-mixer v1\n";
  out += std::format("limit {:.9e}\n", spec_.limit);
  for (const auto& m : layer1_) m.serialise(out);
  final_.serialise(out);
  if (avm_on()) {
    out += std::format("avm_weight {:.9e}\n", spec_.avm_weight);
    avm_.serialise(out);
  }
  out += std::format("scale {} {} {:.9e} {:.9e}\n", scale_.inputs(), scale_.contexts(), spec_.log_b_min, spec_.log_b_max);
  std::string weights;
  scale_.serialise(weights);
  out += weights.substr(weights.find('\n') + 1);  // the values line only: the scale line names the shape
  return out;
}

std::string ValueNet::version() const { return sha256_hex(serialise()); }

}  // namespace nfx::dcm
