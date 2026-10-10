// Study G's extras (docs/DCM.md §10, stage S8): three small uses of the owner's PAQ-style mixer (mixer.hpp) around a
// rollout effect (rollout.hpp), each on the reference implementation (plain float C++); the runtime is not touched.
//
//   G4a  one renderer: a value-domain mixer (ValueNet) over several renderers of the same fine fields (the effect's
//        learned renderer, the other effects' learned renderers, the simulator's own renderer and the compositor's
//        field shader), per pixel and colour channel, under contexts of age, heat and soot.
//   G5a  shard critic: a frozen binary mixer (MixerNet) that tells a short window of real frames from the model's own,
//        from frame statistics; at a shard's start the runtime would roll a few candidate seeds ahead and play the one
//        the critic finds most real.
//   G5b  a mixer over two updates of the coarse state: the learned stepper's and a cheap solver's (the simulation on
//        the 32-cell grid, with the same forcing seed), per cell and channel.
#pragma once

#include <neuralfx/clip.hpp>
#include <neuralfx/dcm/compact.hpp>
#include <neuralfx/dcm/mixer.hpp>
#include <neuralfx/rollout.hpp>
#include <neuralfx/sim.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nfx::dcm::extras {

// --- G4a: renderers of the same fields ----------------------------------------------------------------------------

// The compositor's field shader (src/compose/compose.cpp, Module::shade): heat emits along a blackbody-like ramp, soot
// absorbs, is shadowed towards the sky and lit as a height field. Without a scene light (no fires, no flash), as a
// module would be drawn with nothing around it. Defaults: the fireball's "cloud" look (examples/scenes/fireball.nvfxs).
struct ShaderLook {
  float heat_scale = 1.6f, emission = 1.6f, emission_power = 3.f, soot_density = 3.f, soot_albedo = 0.3f;
  float sky = 0.18f, shadow = 0.3f, relief = 4.f;
  std::array<float, 3> tint{1.f, 0.93f, 0.86f};
};

// The field shader on a rollout state: premultiplied RGBA [size][size][4], rows top to bottom, in display units: the
// shader's linear colour clamped to [0, 1] and raised to 1 / 2.2 (the compositor's gamma), alpha as it is.
void field_shader(const rollout::Model& m, const rollout::State& s, const ShaderLook& look, std::span<float> rgba);

// The simulator's own renderer (sim::Fluid::render) on a rollout state's fine fields, for `effect`, as RGBA floats
// v / 255 of its 8-bit output (rows top to bottom). Keeps a solver of the fields' size to draw with.
class SimRenderer {
 public:
  SimRenderer(sim::Effect effect, int size);
  ~SimRenderer();
  SimRenderer(const SimRenderer&) = delete;
  SimRenderer& operator=(const SimRenderer&) = delete;
  void render(const rollout::State& s, std::span<float> rgba);

 private:
  int size_;
  std::unique_ptr<sim::Fluid> fluid_;
  sim::State st_;
  std::vector<std::uint8_t> out_;
};

// The renderer mixer's experts, per pixel: one RGBA plane each.
enum RenderExpert : int { kLearned, kSim, kShader, kOther1, kOther2, kRenderExperts };
[[nodiscard]] std::string_view render_expert_name(int e);

// Contexts of the renderer mixer, per pixel (the channel is added per row): age bin (frames since the start point),
// heat bin and soot bin of the pixel's fine fields normalised by the effect's renderer scale.
inline constexpr int kAgeBins = 5, kLevelBins = 4;
[[nodiscard]] int age_bin(float frames_since_start);
[[nodiscard]] int level_bin(float normalised);

struct RenderMixConfig {
  std::vector<int> experts;  // which RenderExpert inputs the mixer sees (plus a bias of 1); kLearned first
  [[nodiscard]] std::string name() const;
};

// One mixer for every effect: a ValueNet whose inputs are the chosen experts' values of one channel and a bias, its
// first-layer mixers selected by channel x age, channel x heat and channel x soot, the final mixer by channel; squared
// loss, no AVM. It starts as the learned renderer (weight 1 on it).
class RenderMixer {
 public:
  explicit RenderMixer(RenderMixConfig cfg, double lr = 0.02);
  // A frozen mixer from its serialisation (ValueNet::serialise of a released one).
  static RenderMixer load(RenderMixConfig cfg, const std::string& text);
  // The mixed value of channel c (0..3) from the experts' values of that channel (all kRenderExperts of them; unused
  // ones ignored), clamped to [0, 1]. Remembers the state for update().
  double predict(std::span<const float> experts, int channel, int age, int heat, int soot);
  void update(double target) { net_.update(target); }
  void scale_lr(double f) { net_.scale_lr(f); }
  // Freezes the net and builds its inference copy (float weights), which render_mixed() uses.
  void freeze();
  [[nodiscard]] std::string version() const { return compact_ ? sha256_hex(text_) : net_.version(); }
  [[nodiscard]] std::string serialise() const { return compact_ ? text_ : net_.serialise(); }
  [[nodiscard]] const RenderMixConfig& config() const { return cfg_; }
  [[nodiscard]] const CompactValueNet<float>* compact() const { return compact_.get(); }
  [[nodiscard]] const ValueNet& net() const { return net_; }
  // Mean weight on each chosen expert (and the bias, last) over the contexts, through the final mixer: a summary of
  // what the mixer draws with, per channel.
  [[nodiscard]] std::vector<double> mean_weights(int channel) const;

 private:
  RenderMixConfig cfg_;
  ValueNet net_;
  std::vector<double> x_;
  std::string text_;
  std::shared_ptr<const CompactValueNet<float>> compact_;
};

// Every expert plane of a rollout state (each [size][size][4] floats, rows top to bottom). `others` are the other
// effects' models whose learned renderers are kOther1 and kOther2 (null: zeros).
struct RenderPlanes {
  std::array<std::vector<float>, kRenderExperts> p;
};
void render_experts(const rollout::Model& m, const rollout::State& s, SimRenderer& sim_r, const ShaderLook& look,
                    const rollout::Model* other1, const rollout::Model* other2, RenderPlanes& out, bool all = true);
// The mixed frame from the expert planes, through the frozen mixer's inference copy: RGBA floats in [0, 1].
void render_mixed(const RenderMixer& mix, const rollout::Model& m, const rollout::State& s, const RenderPlanes& planes,
                  std::span<float> rgba);

// --- G5a: the shard critic ------------------------------------------------------------------------------------------

// Statistics of a short window of frames (premultiplied RGBA8, 128 px or any power of two) that the critic reads.
inline constexpr int kCriticFeatures = 16;
[[nodiscard]] std::array<double, kCriticFeatures> critic_features(const Clip& window);
[[nodiscard]] std::string_view critic_feature_name(int i);

// One window for fitting: its features, its controls and whether it is real.
struct CriticSample {
  std::array<double, kCriticFeatures> f{};
  std::array<float, 3> controls{};
  bool real = false;
};

// The critic: per feature, a log-likelihood ratio of a real window against a model window, each class a Gaussian
// whose mean is linear in a few terms of the controls (1, i, w, t, (w - 0.5)^2, i t) with one spread per class; these
// experts, clamped to [-8, 8], and a bias enter a PAQ8 MixerNet (first-layer mixers: none, intensity tercile,
// turbulence tercile), which is trained on other windows and frozen. p_real() is its probability that a window is real.
class ShardCritic {
 public:
  ShardCritic();
  // Fits the experts on `fit`, then trains the mixer on `train` (passes shuffled with `seed`, rates halved after each
  // pass) and freezes it.
  void fit(std::span<const CriticSample> fit, std::span<const CriticSample> train, int passes, std::uint64_t seed);
  // A fitted critic from serialise()'s text.
  static ShardCritic load(const std::string& text);
  [[nodiscard]] double p_real(const std::array<double, kCriticFeatures>& f, std::span<const float> controls);
  // The experts' log-likelihood ratios (inputs of the mixer, without the bias).
  [[nodiscard]] std::array<double, kCriticFeatures> experts(const std::array<double, kCriticFeatures>& f, std::span<const float> controls) const;
  // The SHA-256 of the experts' parameters and the frozen mixer's serialisation.
  [[nodiscard]] std::string version() const;
  [[nodiscard]] std::string serialise() const;

 private:
  static constexpr int kTerms = 6;
  struct Gauss {
    std::array<double, kTerms> beta{};
    double sigma = 1.0;
  };
  std::array<std::array<Gauss, 2>, kCriticFeatures> g_{};  // [feature][model, real]
  std::shared_ptr<MixerNet> net_;                          // while training
  std::shared_ptr<const CompactMixer<double>> compact_;   // fitted: the frozen mixer's inference copy
  std::string mixer_text_;
};

// The start point the runtime plays for (seed, shard) at these controls: one of the three nearest, by a hash of the
// shard and the seed (src/runtime/nvfx.cpp pick_start, without a fixed variation), and the shard's own seed
// (plan_shard: the instance seed for shard 0, then the seed mixed with the shard number).
[[nodiscard]] int runtime_start(const rollout::Model& m, std::span<const float> controls, std::uint64_t seed, std::int64_t shard);
[[nodiscard]] std::uint64_t runtime_shard_seed(std::uint64_t seed, std::int64_t shard);
// Candidate j of shard k: j = 0 is the runtime's own choice; others take the seed splitmix64(seed_0 + j) and pick their
// start point with it as the runtime would.
struct Candidate {
  int start = 0;
  std::uint64_t seed = 0;
};
[[nodiscard]] Candidate shard_candidate(const rollout::Model& m, std::span<const float> controls, std::uint64_t instance_seed,
                                        std::int64_t shard, int j);

// --- G5b: the stepper's update and a cheap solver's update ----------------------------------------------------------

// The simulation on the coarse grid (sim::Fluid at res x res with the run's controls and seed), stepped one frame from a
// rollout state's physical channels. Keeps its own pressure from step to step (warm start).
class CoarseSolver {
 public:
  CoarseSolver(sim::Effect effect, std::span<const float> controls, std::uint64_t seed, int res, float fps);
  ~CoarseSolver();
  CoarseSolver(const CoarseSolver&) = delete;
  CoarseSolver& operator=(const CoarseSolver&) = delete;
  // From the state's coarse physical channels at time s.time: the solver's next physical channels (res * res * kPhys,
  // the stepper's units: velocity in coarse cells per frame).
  void step(const rollout::Model& m, const rollout::State& s, std::span<float> out);

 private:
  int res_;
  float fps_;
  std::unique_ptr<sim::Fluid> fluid_;
  sim::State st_;
};

// Contexts of the update mixer per cell (the channel is added per row): heat level (empty, then three bins), height
// band (4) and age (frames since the start point: 1-8, 9-30, 31-60, 61+).
inline constexpr int kHeatBins = 4, kBands = 4, kAges = 4;

// x' = x + mix(dN, dS, 1) per cell and physical channel, dN = stepper's next - x, dS = solver's next - x. A ValueNet
// (squared loss, no AVM; first-layer mixers by channel x heat, channel x height, channel x age; final by channel) that
// starts as the stepper alone (weight 1 on dN). A fixed blend (1 - a) dN + a dS is the same net with those weights.
class UpdateMixer {
 public:
  explicit UpdateMixer(double lr = 0.02);
  static UpdateMixer blend(double a);
  // A frozen mixer from its serialisation.
  static UpdateMixer load(const std::string& text);
  double predict(double dn, double ds, int channel, int heat, int band, int age);
  void update(double target) { net_.update(target); }
  void scale_lr(double f) { net_.scale_lr(f); }
  // Frozen, it predicts through its inference copy (CompactValueNet, the same arithmetic); unfrozen, it learns again.
  void freeze();
  void unfreeze();
  [[nodiscard]] std::string version() const { return sha256_hex(serialise()); }
  [[nodiscard]] std::string serialise() const { return compact_ ? text_ : net_.serialise(); }
  // Mean weights (dN, dS, bias) of a channel over the contexts, through the final mixer.
  [[nodiscard]] std::array<double, 3> mean_weights(int channel) const;

 private:
  ValueNet net_;
  std::string text_;
  std::shared_ptr<const CompactValueNet<double>> compact_;
};

// The contexts of a cell from the state before the step.
struct CellCtx {
  int heat = 0, band = 0;
};
[[nodiscard]] CellCtx cell_context(const rollout::Model& m, const rollout::State& s, int cell);
[[nodiscard]] int update_age(float frames_since_start);

// One frame of the mixed dynamics: the stepper's coarse step and the solver's, mixed per cell and physical channel
// (memory channels from the stepper; the detail layer's flow shifted by the mixer's change of velocity), then v1's
// detail layer. With the mixer at its start (weight 1 on dN) this is rollout::step exactly. When `rows` is given, the
// (dN, dS, contexts) of every cell and channel are appended for training, in cell-major order.
struct UpdateRow {
  float dn = 0, ds = 0, x = 0;
  std::uint8_t channel = 0, heat = 0, band = 0, age = 0;
};
// detail_layer = false skips the detail layer (training rollouts of the coarse state only).
void mixed_step(const rollout::Model& m, rollout::State& s, std::span<const float> controls, std::uint64_t seed, CoarseSolver& solver,
                UpdateMixer& mix, std::vector<UpdateRow>* rows = nullptr, bool detail_layer = true);

}  // namespace nfx::dcm::extras
