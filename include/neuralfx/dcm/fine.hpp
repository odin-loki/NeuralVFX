// DCM-fine (docs/DCM.md, design G1): the owner's context mixer in place of the rollout detail layer's lock.
//
// Each frame, the coarse step and the MacCormack advection of rollout::detail_step stay as they are. After them, for
// each fine pixel and channel q (heat, soot), a ValueNet (mixer.hpp) mixes cheap experts: planar predictions of the
// next fine value built from buffers the reference already has (the advected field, the semi-Lagrangian value, the v1
// lock and its parts, the new material shaped by each flicker octave, the upsampled coarse value, the block residual,
// two shape terms, the previous frame and a bias). Small integer contexts (heat level, A / C ratio, flow, height,
// controls, age, channel, clusters of regional coarse statistics) select the mixers' weight sets. The net predicts a
// value mu and a Laplace scale b in a normalised domain; generation draws v = max(0, mu + tau b xi) with xi a coherent
// "grain" noise mapped to Laplace quantiles, then may lock block means back to the coarse state.
//
// With the v1 weights (weight 1 on the lock L, 0 elsewhere, AVM off) and tau = 0 the step is v1's detail_step (a test
// holds it to 1e-6). Everything here is plain float/double C++: the reference. The runtime is not touched (stage S6).
#pragma once

#include <neuralfx/dcm/compact.hpp>
#include <neuralfx/dcm/kmeans.hpp>
#include <neuralfx/dcm/mixer.hpp>
#include <neuralfx/rollout.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nfx::dcm::fine {

// --- experts and contexts ----------------------------------------------------------------------------------------------

// Experts in value units (fine heat or soot), per pixel and channel. The bias is 1 in the normalised domain.
enum Expert : int {
  kAdv,         // A: the MacCormack-advected field (what v1 carries)
  kAdvSl,       // A_sl: the semi-Lagrangian value (MacCormack's forward step)
  kAdvDiff,     // A - A_sl
  kLock,        // L: v1's lock output, r_up A + g(phi) a_up
  kLockScaled,  // r_up A
  kNew,         // a_up: new material (before the flicker shaping)
  kNoise1,      // a_up phi_1: new material shaped by each flicker octave
  kNoise2,
  kNoise3,
  kCoarse,      // C_up: the bilinear coarse value
  kBlockRes,    // (C - block mean of A)_up
  kLaplace,     // laplacian of A
  kGrad,        // |grad A| A / s_q
  kPrev,        // the previous frame's value at the pixel (before advection)
  kBias,        // 1 (normalised domain)
  kExperts
};
inline constexpr int kRawExperts = kBias;  // experts stored per row (the bias is not)
[[nodiscard]] std::string_view expert_name(int e);

// Groups of experts switched together by the search.
struct ExpertGroup {
  std::string_view name;
  std::vector<int> experts;
};
[[nodiscard]] const std::vector<ExpertGroup>& expert_groups();  // adv, lock, noise, coarse, shape, prev, bias

enum Context : int {
  kLvl,      // C_up in 6 log bins, 0 = empty
  kRatio,    // A / C_up in 4 bins
  kFlow,     // |u| in 3 bins x vorticity sign
  kHeight,   // 4 bands
  kCtrl,     // intensity x turbulence, 3 x 3
  kAge,      // seconds since the effect began, 4 bins (one-shot effects only)
  kChannel,  // heat, soot
  kMacro4,   // k-means (K = 4) of regional coarse statistics: mean heat, soot, speed, |vorticity|
  kMacro8,   // the same with K = 8
  kExtra,    // a hook for later macro contexts (stage S5: clusters of a denoiser's features, macro_ddpm): a per coarse
             // cell value in [0, Spec::extra_size) supplied by a callback (CellContext); 0 everywhere until then
  kContexts
};
[[nodiscard]] std::string_view context_name(int c);
// Context families compared by the search (docs/DCM.md G1).
enum class Family { none, hand, hand_macro };
[[nodiscard]] std::string_view family_name(Family f);
[[nodiscard]] std::vector<int> family_contexts(Family f, bool one_shot);

// The normalised domain of inputs and targets: linear (v / (C_up + eps_q)) or log (sign(v) log1p(|v| / s_q)).
enum class Domain { linear, log };
[[nodiscard]] std::string_view domain_name(Domain d);

inline constexpr int kScaleFeatures = 4;  // z = (|A - A_sl|, |grad A|, C_up, a_up), normalised
inline constexpr int kMacroStats = 4;     // regional mean heat, soot, speed, |vorticity| (8 x 8 coarse cells)

// One sampled pixel and channel of one frame: raw experts and the raw features its contexts come from.
struct Row {
  std::int32_t run = 0;    // index into the table's runs
  std::int16_t frame = 0;  // the frame predicted (run state index)
  std::int16_t x = 0, y = 0;
  std::int8_t channel = 0;
  std::int8_t empty = 0;   // sampled as an empty pixel (target and lock below the activity threshold)
  std::array<float, kRawExperts> e{};
  float speed = 0;         // fine flow speed, pixels of a 128-pixel frame per frame
  float vort = 0;          // coarse vorticity (upsampled), cells per frame per cell
  std::array<float, kMacroStats> macro{};
  std::uint8_t extra = 0;  // the kExtra context of the pixel's coarse cell
  float time = 0;          // seconds since the effect began
  float grad = 0;          // |grad A|, value per pixel
  float target = 0;        // the true next fine value
};

// --- the spec: everything fixed before a mixer is trained -----------------------------------------------------------

struct Spec {
  Domain domain = Domain::linear;
  bool one_shot = false;
  int size = 128;                                // fine pixels per side the bins were fitted at
  std::array<float, 2> s{1, 1};                  // channel scale s_q (the renderer's input scale)
  std::array<float, 2> eps{0.02f, 0.02f};        // linear domain: C_up + eps_q
  std::array<std::array<float, 4>, 2> lvl{};     // edges between lvl bins 1..5, in units of s_q (bin 0: below empty)
  float empty = 1e-3f;                           // lvl 0 below empty * s_q
  std::array<std::array<float, 3>, 2> ratio{};   // edges of A / (C_up + eps_q)
  std::array<float, 2> speed{0.5f, 1.5f};        // edges of the flow speed bins
  std::array<float, 3> age{0.2f, 0.6f, 1.5f};    // edges of the age bins, seconds
  std::array<float, kMacroStats> macro_mean{}, macro_sd{1, 1, 1, 1};
  std::vector<double> macro4, macro8;            // centroids (K x 4) on standardised statistics
  int extra_size = 1;                            // values of the kExtra context (1: the hook is unused)
  // Grain noise for generation: fbm of value noise at this frequency (per pixel of a 128-pixel frame), rate (per
  // second) and octaves, with a seed stream of its own.
  float grain_freq = 0.25f, grain_rate = 2.f;
  int grain_octaves = 2;
};

[[nodiscard]] int context_size(const Spec& s, int c);

// The bins and clusters from training rows (sizes in the rows' frame), for a model and a domain.
Spec fit_spec(const rollout::Model& m, std::span<const Row> rows, Domain domain, int size, std::uint64_t seed = 0);

// Context values of a row (every Context; kAge is 0 for looping effects). `controls`: the run's.
std::array<int, kContexts> contexts_of(const Spec& s, const Row& r, std::span<const float> controls, int size);
// Normalised experts (every Expert, the bias last), the normalised target and the scale features of a row.
void normalise(const Spec& s, const Row& r, std::span<double> x, std::span<double> z);
[[nodiscard]] double normalise_value(const Spec& s, int channel, double v, double c_up);
[[nodiscard]] double value_of(const Spec& s, int channel, double mu, double c_up);  // the inverse, in value units
// Code length (bits) of the true value v under Laplace(mu, b) in the normalised domain, with the bin
// [v - s_q / 512, v + s_q / 512] in value units (delta = s_q / 256): every reported bits figure uses this.
[[nodiscard]] double value_bits(const Spec& s, int channel, double v, double mu, double b, double c_up);
// The bin width in the normalised domain of value v (what value_bits integrates over).
[[nodiscard]] double bin_width(const Spec& s, int channel, double v, double c_up);

// --- the mixer --------------------------------------------------------------------------------------------------------

// A mixer configuration in experts and contexts (independent of a search problem's column numbering).
struct Config {
  std::vector<int> experts;         // inputs, ascending Expert values
  std::vector<int> mixer_contexts;  // one context-selected first-layer mixer each (a mixer without context is always there)
  int avm_context = -1;             // -1: one table (or no AVM when the weight is 0)
  int scale_context = -1;
  ValueNetSpec spec;                // learning settings, AVM weight, loss (context sizes follow from the contexts)
  int epochs = 4;
};
[[nodiscard]] std::string describe(const Config& c);

// A released mixer: the spec, the configuration and the frozen net as its serialisation (whose SHA-256 is the
// version), run by the compact inference copy (no learning state, so one mixer serves many threads).
struct Mixer {
  Spec spec;
  Config config;
  std::string text;                                // ValueNet::serialise() of the frozen net
  std::optional<CompactValueNet<double>> net;
  // Predicts (mu, b) in the normalised domain. x: every expert normalised (normalise()), ctx: every context.
  [[nodiscard]] ValuePrediction predict(std::span<const double> x, std::span<const int> ctx, std::span<const double> z) const;
  [[nodiscard]] std::string version() const { return net ? net->version() : std::string(); }
};
// Wraps a frozen net (its learning settings recorded in config.spec).
Mixer make_mixer(const Spec& s, Config c, const ValueNet& net);

// The v1 rule as a mixer: every expert as input, weight 1 on the lock, AVM off, no contexts beyond `contexts`.
Mixer v1_mixer(const Spec& s, std::span<const int> contexts = {});

// Save and load a mixer: the spec, the configuration and the net's serialisation (text, versioned by its SHA-256).
void save_mixer(const std::filesystem::path& path, const Mixer& m);
Mixer load_mixer(const std::filesystem::path& path);

// --- frames -----------------------------------------------------------------------------------------------------------

// Every buffer of one detail step at size S: v1's (velocity, advection, lock) and the experts' (planar, per channel).
struct Frame {
  int S = 0, R = 0;
  std::array<std::vector<float>, 2> prev, A, Asl, L, rA, aup, n1, n2, n3, cup, bres, lap, grad;
  std::vector<float> ux, vy;     // fine velocity, pixels per frame
  std::vector<float> speed;      // |u|, pixels of a 128-pixel frame per frame
  std::vector<float> vort;       // upsampled coarse vorticity
  std::vector<float> macro;      // per coarse cell: kMacroStats raw statistics
  std::vector<std::uint8_t> extra;  // per coarse cell: the kExtra context (0 unless a CellContext fills it)
};

// The hook of kExtra: fills one value per coarse cell from the stepped coarse state (rows from the bottom).
using CellContext = std::function<void(const rollout::State& s, std::span<std::uint8_t> per_cell)>;

// v1's detail step on s (coarse and flow already stepped; fine fields from the frame before), keeping every
// intermediate. Does not change s. frame.L equals what rollout::detail_step would leave in the fine fields.
void compute_frame(const rollout::Model& m, const rollout::State& s, std::uint64_t seed, std::span<const float> controls,
                   Frame& f, const CellContext& extra = {});

// A Row of pixel (x, y), channel q of a computed frame (target left 0).
Row row_of(const Frame& f, const rollout::State& s, int x, int y, int q);

// Every context of pixel (x, y), channel q.
std::array<int, kContexts> frame_contexts(const Spec& sp, const Frame& f, const rollout::State& s, std::span<const float> controls,
                                          int x, int y, int q);

// Pixels the mixer does not predict: the advected, semi-Lagrangian, new, previous, coarse and v1 values are all below
// kSkip s_q (invisible: the renderer's gate is below 1%). They keep v1's value. skip_row is the same test on a Row.
inline constexpr float kSkip = 2e-4f;
[[nodiscard]] bool skip_pixel(const Spec& sp, const Frame& f, std::size_t i, int q);
[[nodiscard]] bool skip_row(const Spec& sp, const Row& r);

struct GenOptions {
  double tau = 0.0;     // grain amplitude in Laplace scales (0: the mean)
  // Block means back to the coarse state after sampling. 0: off; 1: exact (each block scaled to the coarse value,
  // without bound); 2: v1's lock on the mixer's output (growth limited by DetailSpec::grow, the rest arrives as new
  // material shaped by the flicker noise, as rollout::detail_step does it).
  int relock = 0;
  CellContext extra;    // the kExtra hook (empty: 0)
};

// Grain noise xi at a pixel: a Laplace(0, 1) quantile of the grain fbm (its own seed stream), for frame time t.
[[nodiscard]] float grain(const Spec& sp, std::uint64_t seed, float X, float Y, float t);

// DCM-fine's detail step (in place of rollout::detail_step): v1's advection, then the mixer per pixel and channel.
void detail_step(const rollout::Model& m, const Mixer& mix, rollout::State& s, std::uint64_t seed, std::span<const float> controls,
                 const GenOptions& g, Frame& f);
// One frame: the coarse stepper (as rollout::step), then DCM-fine's detail step.
void step(const rollout::Model& m, const Mixer& mix, rollout::State& s, std::span<const float> controls, std::uint64_t seed,
          const GenOptions& g, Frame& f);
// rollout::start with DCM-fine's warm-up (when the start point has no fine fields): `warmup` frames (-1: the model's).
// use_fine = false ignores stored fine fields (a cold start from the coarse state, as fire without them).
rollout::State start(const rollout::Model& m, const Mixer& mix, int index, int size, std::span<const float> controls,
                     std::uint64_t seed, const GenOptions& g, int warmup = -1, bool use_fine = true);

// rollout::render without allocations per pixel: the directional soot sums once per coarse cell, the MLP in the same
// order of operations, so the frame is bit-identical to rollout::render's (a test holds it). Premultiplied RGBA floats
// [size][size][4], rows top to bottom.
void render(const rollout::Model& m, const rollout::State& s, std::span<float> rgba);

}  // namespace nfx::dcm::fine
