// Context mixing in the style of the PAQ compressors (Mahoney), ported from the owner's CameraDetector (Phase 12,
// cabinlab/src/diffusion: diffusion-context mixing, DCM). Experts' scores enter as stretched probabilities (log-odds),
// a mixer forms a weighted sum with a weight set chosen by a context and squashes it back, and after each decision the
// chosen weights move along the gradient of coding cost: w += lr · (y − p) · x. PAQ8 stacks several such mixers, each
// selecting its weights by a different context, under a final mixer, then refines the probability with an adaptive
// probability map (APM, also called SSE). The arithmetic is the original's, operation for operation, so a version (the
// SHA-256 of a serialisation) means the same thing in both projects; only the serialisation's first line differs.
//
// Here the mixer serves generating and coding effects (docs/DCM.md): many cheap experts (the advected field, the
// detail layer's own rule, the upsampled coarse value, noise octaves, ...) are mixed under contexts that describe the
// regime (heat level, height, flow, controls, clusters of a small denoiser's features). Generation predicts a value
// and its uncertainty rather than a probability, so this header adds the value-domain version of the same network:
// ValueMixer and ValueNet (normalised-LMS updates, a squared or Laplace loss, an adaptive value map in place of the
// APM, and a scale net for the Laplace scale). A predicted distribution is also a code length (ValueNet::bits).
//
// A released mixer must not change, so a net can be frozen: update() then does nothing, and version() is the SHA-256
// of its serialised weights.
#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nfx::dcm {

// ln(p / (1 − p)) with p clamped to [1e-6, 1 − 1e-6].
[[nodiscard]] double stretch(double p);
// 1 / (1 + e^−x).
[[nodiscard]] double squash(double x);

// One mixer layer: n inputs, one weight set per context value.
class Mixer {
 public:
  // lr: learning rate; init: initial value of every weight (1 / n averages the inputs' log-odds).
  Mixer(int n_inputs, int n_contexts, double lr, double init);
  // Mixed log-odds of x under the weight set of `context` (0 <= context < n_contexts). Remembers x and the result for
  // update().
  double mix(std::span<const double> x, int context);
  // mix() of the same inputs through several mixers, mixers[k] under contexts[k], into out[k]. The dot products are
  // interleaved four at a time, each still summed in input order, so every result and remembered state is bit-identical
  // to mix()'s; only the independent sums overlap in the pipeline (MixerNet's first layer, the inner loop of training).
  // With remember_inputs = false the mixers do not copy x: their next update must be update_all() with the same x.
  static void mix_all(std::span<Mixer> mixers, std::span<const double> x, std::span<const int> contexts,
                      std::span<double> out, bool remember_inputs = true);
  // update() of every mixer after mix_all(), reading the inputs from x (the inputs given to mix_all) instead of each
  // mixer's own copy: the same arithmetic, so the same weights bit for bit.
  static void update_all(std::span<Mixer> mixers, std::span<const double> x, int y);
  // Moves the last used weight set: w += lr_eff · (y − squash(last)) · x, lr_eff = lr / (1 + uses / anneal) (the first
  // decisions of a context learn fastest, as PAQ's adaptive rates do). Does nothing when frozen.
  void update(int y);
  void set_lr(double lr) noexcept { lr_ = lr; }
  void set_anneal(double anneal) noexcept { anneal_ = anneal; }
  void freeze(bool on = true) noexcept { frozen_ = on; }
  // Forgets how often each context was used, so learning starts again at the full rate (online adaptation).
  void reset_counts() noexcept { std::fill(uses_.begin(), uses_.end(), 0u); }
  [[nodiscard]] bool frozen() const noexcept { return frozen_; }
  [[nodiscard]] int inputs() const noexcept { return n_; }
  [[nodiscard]] int contexts() const noexcept { return k_; }
  [[nodiscard]] std::span<const double> weights(int context) const;
  [[nodiscard]] std::span<double> weights(int context);
  void serialise(std::string& out) const;

 private:
  int n_, k_;
  double lr_, anneal_ = 500.0;
  bool frozen_ = false;
  std::vector<double> w_;
  std::vector<std::uint32_t> uses_;
  std::vector<double> x_;
  int ctx_ = 0;
  double last_ = 0.0;
};

// Adaptive probability map: per context, 33 buckets over stretch(p) in [−8, 8]; the output interpolates the two
// nearest buckets, and update() moves both toward the label at `rate`. Initialised to the identity.
class APM {
 public:
  APM(int n_contexts, double rate);
  double refine(double p, int context);
  void update(int y);
  void freeze(bool on = true) noexcept { frozen_ = on; }
  void serialise(std::string& out) const;

 private:
  int k_;
  double rate_;
  bool frozen_ = false;
  std::vector<double> t_;  // probabilities, 33 per context
  std::size_t lo_ = 0;
  double frac_ = 0.0;
};

struct MixerNetSpec {
  std::vector<int> context_sizes;  // one first-layer mixer per entry, selecting its weights by that context
  int final_contexts = 1;          // contexts of the final mixer
  int apm_contexts = 1;            // contexts of the APM (0: no APM)
  double lr1 = 0.02, lr2 = 0.01;   // first-layer and final learning rates
  double anneal = 500.0;
  double apm_rate = 0.02;
  double apm_weight = 0.5;         // output = (1 − apm_weight) · p + apm_weight · APM(p)
};

// PAQ8-style two-layer network. contexts passed to predict(): one per first-layer mixer, then the final mixer's, then
// the APM's.
class MixerNet {
 public:
  MixerNet(int n_inputs, const MixerNetSpec& spec);
  // Probability of the positive class. Remembers the state for update().
  double predict(std::span<const double> x, std::span<const int> contexts);
  void update(int y);
  void freeze(bool on = true) noexcept;
  [[nodiscard]] bool frozen() const noexcept { return frozen_; }
  // Scales every learning rate (training schedules).
  void scale_lr(double factor) noexcept;
  // Restores the spec's learning rates and forgets context counts: online adaptation from trained weights.
  void restart_learning(const MixerNetSpec& spec) noexcept;
  [[nodiscard]] int first_layer() const noexcept { return static_cast<int>(layer1_.size()); }
  [[nodiscard]] const Mixer& mixer(int i) const { return layer1_.at(static_cast<std::size_t>(i)); }
  [[nodiscard]] const Mixer& final_mixer() const noexcept { return final_; }
  // Canonical text of every weight and table value (fixed precision); first line "nvfx-paq-mixer v1".
  [[nodiscard]] std::string serialise() const;
  // SHA-256 (hex) of serialise(): the version stamp of a frozen mixer.
  [[nodiscard]] std::string version() const;

 private:
  MixerNetSpec spec_;
  std::vector<Mixer> layer1_;
  Mixer final_;
  APM apm_;
  std::vector<double> h_;
  std::vector<double> x_;  // the inputs of the last predict(), shared by the first-layer mixers' updates
  double p_mix_ = 0.5, p_out_ = 0.5;
  bool frozen_ = false;
};

// SHA-256 of bytes, as 64 lowercase hex digits.
[[nodiscard]] std::string sha256_hex(std::string_view bytes);

// --- the value domain (new in NeuralVFX) -----------------------------------------------------------------------------
//
// The same structure predicts a real value μ and the scale b of a Laplace distribution around it, instead of a
// probability. The mixers are linear (no squash) and learn by normalised least mean squares, which makes the step size
// independent of the inputs' scale: w += lr_eff · g · x / (eps + |x|²), with g = y − v for the squared loss and
// g = sign(y − v) / b for the Laplace loss (v: the mixer's own output). An adaptive value map (AVM) refines μ as the
// APM refines p, and a scale net predicts log b from caller-given features, trained by the Laplace loss's gradient.

enum class ValueLoss { squared, laplace };

[[nodiscard]] std::string_view loss_name(ValueLoss loss);

// −log2 of the probability mass that Laplace(mu, b) gives the bin [y − delta / 2, y + delta / 2]: the code length in
// bits of y quantised with step delta. Finite for every finite input (the mass is floored at 1e-300).
[[nodiscard]] double laplace_bits(double y, double mu, double b, double delta);

// One linear mixer layer for values: n inputs, one weight set per context value, normalised-LMS learning.
class ValueMixer {
 public:
  // lr: learning rate (0 < lr < 2 keeps normalised LMS stable); init: initial value of every weight; eps: the
  // regulariser added to |x|².
  ValueMixer(int n_inputs, int n_contexts, double lr, double init, double eps = 1e-3);
  // w · x under the weight set of `context`. Remembers x and the result for update().
  double mix(std::span<const double> x, int context);
  // Moves the last used weight set: w += lr_eff · g · x / (eps + |x|²), lr_eff = lr / (1 + uses / anneal). g is the
  // negative gradient of the loss with respect to this mixer's output (y − last() for the squared loss). Does nothing
  // when frozen.
  void update(double g);
  [[nodiscard]] double last() const noexcept { return last_; }
  void set_lr(double lr) noexcept { lr_ = lr; }
  void set_anneal(double anneal) noexcept { anneal_ = anneal; }
  void freeze(bool on = true) noexcept { frozen_ = on; }
  void reset_counts() noexcept { std::fill(uses_.begin(), uses_.end(), 0u); }
  [[nodiscard]] bool frozen() const noexcept { return frozen_; }
  [[nodiscard]] int inputs() const noexcept { return n_; }
  [[nodiscard]] int contexts() const noexcept { return k_; }
  [[nodiscard]] std::span<const double> weights(int context) const;
  [[nodiscard]] std::span<double> weights(int context);
  void serialise(std::string& out) const;

 private:
  int n_, k_;
  double lr_, eps_, anneal_ = 500.0;
  bool frozen_ = false;
  std::vector<double> w_;
  std::vector<std::uint32_t> uses_;
  std::vector<double> x_;
  int ctx_ = 0;
  double last_ = 0.0;
};

// Adaptive value map: per context, 33 knots evenly spaced over a fixed input range [lo, hi], each holding an offset
// from the identity (all 0 at the start, so the map starts as the identity exactly). The map of v is v plus the
// offset interpolated between the two nearest knots; outside the range the end knot's offset applies. update(y) moves
// both knots' offsets toward the residual y − v at `rate`, each weighted by its share of the interpolation (as the APM
// moves its buckets toward the label).
class AVM {
 public:
  static constexpr int kKnots = 33;
  AVM(int n_contexts, double lo, double hi, double rate);
  // The interpolated offset at v under `context` (clamped to the table's contexts). Remembers the lookup for update().
  double offset(double v, int context);
  // v + offset(v, context).
  double refine(double v, int context) { return v + offset(v, context); }
  void update(double y);
  void freeze(bool on = true) noexcept { frozen_ = on; }
  [[nodiscard]] double lo() const noexcept { return lo_; }
  [[nodiscard]] double hi() const noexcept { return hi_; }
  void serialise(std::string& out) const;

 private:
  int k_;
  double lo_, hi_, rate_;
  bool frozen_ = false;
  std::vector<double> d_;  // offsets, 33 per context
  std::size_t at_ = 0;
  double frac_ = 0.0, v_ = 0.0;
};

struct ValueNetSpec {
  std::vector<int> context_sizes;  // one first-layer mixer per entry, selecting its weights by that context
  int final_contexts = 1;          // contexts of the final mixer
  int avm_contexts = 1;            // contexts of the AVM (0: no AVM)
  int scale_contexts = 1;          // contexts of the scale net
  ValueLoss loss = ValueLoss::squared;
  double lr1 = 0.02, lr2 = 0.01;   // first-layer and final learning rates
  double anneal = 500.0;
  double eps = 1e-3;               // the normalised-LMS regulariser
  double limit = 1e6;              // first-layer outputs and the mixed value are clamped to [−limit, limit]
  double avm_lo = -4.0, avm_hi = 4.0;  // the AVM's fixed input range
  double avm_rate = 0.02;
  double avm_weight = 0.0;         // output = (1 − avm_weight) · μ + avm_weight · AVM(μ)
  double scale_lr = 0.02;          // the scale net's learning rate
  double log_b_init = 0.0;         // the scale net's starting log b, in every context
  double log_b_min = -12.0, log_b_max = 12.0;  // log b is clamped to this range
};

struct ValuePrediction {
  double mu = 0.0;  // the predicted value
  double b = 1.0;   // the scale of the Laplace distribution around it (> 0)
};

// The value-domain PAQ8-style network: first-layer value mixers (each selecting its weights by a context), a final
// value mixer over their outputs, the AVM, and the scale net log b = U[ctx] · (1, z) (z: a few scale features given by
// the caller; the constant 1 is the bias). contexts passed to predict(): one per first-layer mixer, then the final
// mixer's, the AVM's and the scale net's.
//
// update(y): every first-layer mixer and the final mixer learn from their own output's error (as in MixerNet), the
// AVM moves toward the residual of the mixed value, and the scale net steps along the Laplace loss's gradient in log b,
// |y − μ| / b − 1 (clamped to [−1, 16]).
class ValueNet {
 public:
  ValueNet(int n_inputs, int n_scale_features, const ValueNetSpec& spec);
  // Remembers the state for update() and bits().
  ValuePrediction predict(std::span<const double> x, std::span<const int> contexts, std::span<const double> z);
  void update(double y);
  void freeze(bool on = true) noexcept;
  [[nodiscard]] bool frozen() const noexcept { return frozen_; }
  // Scales the first-layer and final learning rates (training schedules).
  void scale_lr(double factor) noexcept;
  // Restores the spec's learning rates and forgets context counts: online adaptation from trained weights.
  void restart_learning(const ValueNetSpec& spec) noexcept;
  // Every weight set of every first-layer mixer becomes w, and the final mixer takes the first first-layer mixer
  // alone (weight 1 on it, 0 on the others): the net then predicts w · x. With w one-hot (weight 1 on one input, 0
  // elsewhere) and the AVM untouched, it predicts that input exactly: it starts as a copy of a hand-made rule.
  void set_weights(std::span<const double> w);
  // set_weights() with weight 1 on `input` and 0 elsewhere.
  void set_rule(int input);
  // Code length of y (quantised with step delta) under the last prediction.
  [[nodiscard]] double bits(double y, double delta) const;
  [[nodiscard]] const ValuePrediction& last() const noexcept { return out_; }
  [[nodiscard]] int inputs() const noexcept { return n_; }
  [[nodiscard]] int scale_features() const noexcept { return nz_; }
  [[nodiscard]] int first_layer() const noexcept { return static_cast<int>(layer1_.size()); }
  [[nodiscard]] const ValueMixer& mixer(int i) const { return layer1_.at(static_cast<std::size_t>(i)); }
  [[nodiscard]] const ValueMixer& final_mixer() const noexcept { return final_; }
  [[nodiscard]] const ValueMixer& scale_mixer() const noexcept { return scale_; }
  [[nodiscard]] const ValueNetSpec& spec() const noexcept { return spec_; }
  // Canonical text of every weight, knot and setting inference needs (fixed precision); first line
  // "nvfx-value-mixer v1".
  [[nodiscard]] std::string serialise() const;
  // SHA-256 (hex) of serialise().
  [[nodiscard]] std::string version() const;

 private:
  [[nodiscard]] bool avm_on() const noexcept { return spec_.avm_contexts > 0 && spec_.avm_weight > 0.0; }
  [[nodiscard]] double gradient(double y, double v) const noexcept;

  ValueNetSpec spec_;
  int n_, nz_;
  std::vector<ValueMixer> layer1_;
  ValueMixer final_;
  AVM avm_;
  ValueMixer scale_;
  std::vector<double> h_, zz_;
  double mu_mix_ = 0.0;
  ValuePrediction out_;
  bool frozen_ = false;
};

}  // namespace nfx::dcm
