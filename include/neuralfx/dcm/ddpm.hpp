// A small denoising diffusion model on the coarse state of rollout effects: the "macro" half of diffusion-context
// mixing (docs/DCM.md, study G, stage S5: designs G2a, G2b and G2c).
//
// The owner's CameraDetector trains its denoiser with LibTorch (cabinlab/src/diffusion/denoiser.cpp, train.cpp). This
// project uses no LibTorch, so the network here is written out by hand, forward and backward (tests/test_dcm_ddpm.cpp
// checks the gradients against finite differences). What is kept from the original: epsilon prediction, the cosine
// schedule (cosine_alpha_bar, the same formula and beta clip), a zero-initialised output layer and residual blocks that
// start as the identity, Adam with a linear warm-up then cosine decay to 10%, gradient clipping, an EMA copy of the
// weights (0.99 for the first 500 steps, then 0.999), and features read at a fixed noise level with a fixed noise image
// per level and seed, so features are bit-identical between runs.
//
// The network (Config defaults): an epsilon-prediction UNet on the 32 x 32 coarse state, 4 physical channels divided by
// the stepper's channel scales (rollout::Model::scale), plus two position channels.
//   level 0, 32 x 32, 32 channels: stem conv3, residual block e0                     -> skip s0
//   level 1, 16 x 16, 64 channels: 2 x 2 average pool, 1 x 1 conv, block e1           -> skip s1
//   level 2,  8 x  8, 64 channels: 2 x 2 average pool, 1 x 1 conv, blocks m0 and m1
//   up to level 1: 1 x 1 conv, nearest 2 x upsampling, + s1, block d1
//   up to level 0: 1 x 1 conv, nearest 2 x upsampling, + s0, block d0
//   output: SiLU, conv3 to 4 channels (zero-initialised): the predicted noise
// Two residual blocks per level. A block is x + conv3(SiLU(FiLM(conv3(SiLU(x))))); its second convolution starts at
// zero. FiLM (h * (1 + gamma) + beta per channel) comes from one two-layer MLP over the sinusoidal embedding of the
// noise level t and the condition (the effect's controls, then its age features for one-shot effects). No attention and
// no GroupNorm: the zero-initialised residual outputs, gradient clipping and the warm-up keep training stable instead.
//
// Layouts are [cell][channel], rows from the bottom, as everywhere in rollout.hpp. Everything is deterministic for a
// seed, independent of the thread count: the noise is drawn from counter-based generators, and per-sample gradients are
// summed in a fixed order.
#pragma once

#include <neuralfx/dcm/kmeans.hpp>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nfx::dcm::ddpm {

struct Config {
  int res = 32;          // cells per side, divisible by 4 (levels res, res / 2, res / 4)
  int channels = 4;      // modelled channels (u, v, heat, soot)
  int c0 = 32, c1 = 64, c2 = 64;  // widths of the three levels
  int cond = 3;          // condition values: controls, then age features
  int freqs = 16;        // frequencies of the noise level's sinusoidal embedding (a sine and a cosine each)
  int film_hidden = 64;  // hidden width of the FiLM MLP
  int timesteps = 1000;  // T
  [[nodiscard]] int inputs() const noexcept { return channels + 2; }  // the state, then x and y in [-1, 1]
  [[nodiscard]] int embed() const noexcept { return 2 * freqs + cond; }
  // Throws std::invalid_argument when a size is out of range.
  void validate() const;
};

// Residual blocks, in the order of the forward pass.
inline constexpr int kBlocks = 6;  // e0, e1, m0, m1, d1, d0
inline constexpr std::array<const char*, kBlocks> kBlockNames = {"e0", "e1", "m0", "m1", "d1", "d0"};

// Offsets of the weights in one float array. 3 x 3 convolutions are [tap][in][out] (tap = (dy + 1) * 3 + (dx + 1)),
// 1 x 1 convolutions and the MLP's layers [in][out], each followed by its bias.
struct BlockLayout {
  int level = 0, width = 0;
  std::size_t wa = 0, ba = 0, wb = 0, bb = 0;  // first and second convolution
  std::size_t film = 0;                        // gamma at film .. film + width - 1 of the MLP output, beta after it
};
struct Layout {
  std::size_t stem_w = 0, stem_b = 0;
  std::array<BlockLayout, kBlocks> blocks{};
  std::size_t down1_w = 0, down1_b = 0, down2_w = 0, down2_b = 0;  // level 0 -> 1, 1 -> 2
  std::size_t up2_w = 0, up2_b = 0, up1_w = 0, up1_b = 0;          // level 2 -> 1, 1 -> 0
  std::size_t out_w = 0, out_b = 0;
  std::size_t mlp1_w = 0, mlp1_b = 0, mlp2_w = 0, mlp2_b = 0;
  std::size_t film_size = 0;  // outputs of the MLP (2 x width per block)
  std::size_t size = 0;       // weights in all
};
[[nodiscard]] Layout layout(const Config& c);
// Multiply-adds of one forward pass (convolutions, MLP).
[[nodiscard]] double forward_macs(const Config& c);

// alpha_bar(t) for t = 0..T of the cosine schedule (Nichol and Dhariwal 2021, s = 0.008), with the owner's beta clip
// (1 - alpha_bar(t) / alpha_bar(t - 1) at most 0.999). alpha_bar(0) = 1.
[[nodiscard]] std::vector<double> cosine_alpha_bar(int timesteps);

struct Denoiser {
  Config cfg;
  std::vector<float> scale;   // per channel: physical units of one network unit (the stepper's m.scale)
  std::vector<float> lo, hi;  // per channel: range of the normalised training states (samples are kept inside it)
  std::vector<float> w;       // layout(cfg).size weights (after training: the EMA copy)
  [[nodiscard]] std::size_t parameters() const noexcept { return w.size(); }
};

// Fresh weights: He-initialised convolutions and MLP, the second convolution of every block and the output layer zero,
// the FiLM layer small. scale 1, range [-10, 10].
[[nodiscard]] Denoiser init_denoiser(const Config& c, std::uint64_t seed);

// Binary serialisation ("NVFXDDPM", version 1, little-endian: config, scale, range, weights as float32) and its
// SHA-256, the version stamp of a released denoiser.
[[nodiscard]] std::string serialise(const Denoiser& d);
[[nodiscard]] std::expected<Denoiser, std::string> parse(std::string_view bytes);
[[nodiscard]] std::string version(const Denoiser& d);
std::expected<void, std::string> save(const std::filesystem::path& path, const Denoiser& d);
[[nodiscard]] std::expected<Denoiser, std::string> load(const std::filesystem::path& path);

// Standard normal values from a counter-based generator (splitmix64 and Box-Muller): the same numbers on every
// platform with the same libm.
void gaussian(std::uint64_t seed, std::span<float> out);

// --- the network ---------------------------------------------------------------------------------------------------

// The predicted noise eps_hat(x_t, t) of one state x_t (res^2 x channels, network units), t in [1, T].
void predict_eps(const Denoiser& d, std::span<const float> xt, int t, std::span<const float> cond, std::span<float> eps);

// Tweedie's one-step denoise x0_hat = (x_t - sqrt(1 - alpha_bar) * eps_hat) / sqrt(alpha_bar).
void tweedie(const Denoiser& d, std::span<const float> xt, int t, std::span<const float> cond, std::span<float> x0_hat);

// The prior against drift (G2b): x is read as the clean signal of x_t = sqrt(alpha_bar) * x at level t, and moved by
// beta towards Tweedie's denoise of it: x <- (1 - beta) * x + beta * x0_hat, x0_hat = x - sqrt((1 - ab) / ab) * eps_hat.
// No noise is added. The result is kept inside [lo, hi].
void prior_step(const Denoiser& d, std::span<float> x, int t, float beta, std::span<const float> cond);

// Deterministic DDIM (eta = 0) with stride T / steps: from x at level t_start down to x0 (t_start, t_start - stride, ...,
// then 0), written back into x. x0_hat is kept inside [lo, hi] at every step.
void ddim(const Denoiser& d, std::span<float> x, int t_start, int steps, std::span<const float> cond);
// Where a fresh sample starts: T - T / steps + 1, the first point of DDIM's grid (Song et al. 2021). Not T itself: there
// the cosine schedule's clipped alpha_bar is about 2e-9, so the first x0 estimate is pure error (samples started at T
// came out several times too hot and too spread).
[[nodiscard]] int sample_start(const Config& c, int steps);
// A fresh sample: gaussian(seed) taken as x at sample_start, then ddim (`steps` passes).
void sample(const Denoiser& d, std::span<const float> cond, int steps, std::uint64_t seed, std::span<float> out);
// SDEdit (Meng et al. 2022): x0 noised to level t0 (rounded to the grid) with gaussian(seed), then ddim under `cond`.
void sdedit(const Denoiser& d, std::span<const float> x0, int t0, int steps, std::span<const float> cond, std::uint64_t seed,
            std::span<float> out);
// Forward passes of ddim from t_start (the cost of a sample or an SDEdit).
[[nodiscard]] int ddim_passes(const Config& c, int t_start, int steps);

// Feature maps of the denoiser at a fixed noise level: x_t = sqrt(ab) x0 + sqrt(1 - ab) fixed_noise(t, seed).
struct Features {
  int r1 = 0, r2 = 0, c1 = 0, c2 = 0;  // sides and widths of levels 1 and 2
  std::vector<float> enc1;  // r1^2 x c1: the output of block e1 (16 x 16)
  std::vector<float> mid;   // r2^2 x c2: the output of block m1 (8 x 8)
  std::vector<float> dec1;  // r1^2 x c1: the output of block d1 (16 x 16)
};
void fixed_noise(const Config& c, int t, std::uint64_t seed, std::span<float> out);
[[nodiscard]] Features features_at(const Denoiser& d, std::span<const float> x0, int t, std::span<const float> cond,
                                   std::uint64_t noise_seed);

// The training loss of one example, mean over cells and channels of (eps_hat - eps)^2 with
// x_t = sqrt(ab) x0 + sqrt(1 - ab) eps, under the weights w (layout(c).size). When grad is not empty the gradient of
// that loss times grad_scale is added to it. Exposed for the gradient tests.
double example_loss(const Config& c, std::span<const float> w, std::span<const float> x0, int t, std::span<const float> eps,
                    std::span<const float> cond, std::span<float> grad, float grad_scale = 1.f);

// Exposed for the tests: the largest absolute difference between the fast 3 x 3 kernels (forward, weight and bias
// gradient, input gradient through the mirrored weights) and the plain patterns copied from rollout_train.cpp, on
// random data of the given shape; also 1000 times the largest relative error of SiLU and its derivative (vectorised
// exponential) on [-30, 30].
[[nodiscard]] double kernel_max_error(int res, int ci, int co, std::uint64_t seed);

// --- training ------------------------------------------------------------------------------------------------------

// Normalised states and their conditions.
struct Dataset {
  int count = 0, values = 0, conds = 0;  // states; values and condition values per state
  std::vector<float> x, cond;
  [[nodiscard]] std::span<const float> state(int i) const { return std::span(x).subspan(static_cast<std::size_t>(i) * static_cast<std::size_t>(values), static_cast<std::size_t>(values)); }
  [[nodiscard]] std::span<const float> condition(int i) const { return std::span(cond).subspan(static_cast<std::size_t>(i) * static_cast<std::size_t>(conds), static_cast<std::size_t>(conds)); }
};

struct TrainLog {
  int step = 0;
  double loss = 0;           // mean training loss since the last log
  double lr = 0;
  double seconds = 0;
  std::vector<double> eval;  // EMA weights: loss at each TrainOptions::eval_t on the evaluation set (fixed noise)
};

struct TrainOptions {
  int steps = 12000;
  int batch = 32;
  float lr = 5e-4f;
  int warmup = 500;          // linear warm-up steps, then cosine decay to 10% of lr
  float clip = 1.f;          // gradient norm clip
  float ema = 0.999f;        // (0.99 for the first 500 steps)
  int threads = 2;
  std::uint64_t seed = 1;
  int log_every = 500;
  std::vector<int> eval_t = {50, 200, 500, 800};
  int eval_count = 256;      // states of the evaluation set scored at each log
  std::uint64_t eval_seed = 77;
  std::function<void(const TrainLog&, const Denoiser& ema)> progress;
  // The training state (step, weights, EMA, Adam moments, elapsed seconds) is written here at every log when set, and
  // training resumes from it when the file exists at the start: a resumed run gives the same weights, bit for bit, as
  // one that was never stopped (the noise of a step depends only on the seed and the step).
  std::filesystem::path state_path;
  int stop_after = 0;  // > 0: end after this step, as an interruption would (for the resume test)
};

struct TrainResult {
  std::vector<TrainLog> curve;
  double seconds = 0;
};

// Trains d.w on the states (d.scale, d.lo, d.hi must be set) and leaves the EMA weights in d.w. `eval` (may be null)
// is scored at every log.
TrainResult train(Denoiser& d, const Dataset& data, const TrainOptions& o, const Dataset* eval = nullptr);

// Mean loss at level t over the first `count` states of `data` (all when count <= 0), noise fixed by (seed, t, state).
[[nodiscard]] double eval_loss(const Denoiser& d, const Dataset& data, int t, std::uint64_t seed, int count, int threads);

// --- coarse states of rollout effects ---------------------------------------------------------------------------------

// Physical coarse values (res^2 x kPhys, or res^2 x channels() of a rollout State) to network units and back.
void to_network(const Denoiser& d, std::span<const float> phys, int stride, std::span<float> x);
void to_physical(const Denoiser& d, std::span<const float> x, int stride, std::span<float> phys);

// States first, first + every, ... of each run, divided by m.scale; the condition is rollout::condition at the state's
// time (controls, then age).
[[nodiscard]] Dataset states_from_runs(const rollout::Model& m, std::span<const rollout::Run> runs, int first, int every);
// Per-channel minimum and maximum of a data set, widened by 10% of the span (the range samples are kept in).
void set_range(Denoiser& d, const Dataset& data);
// A data set on disk ("NVFXGST1", little-endian counts, then the values and conditions as float32), so the simulation
// runs once.
std::expected<void, std::string> save_dataset(const std::filesystem::path& path, const Dataset& data);
[[nodiscard]] std::expected<Dataset, std::string> load_dataset(const std::filesystem::path& path);

// --- contexts (G2a) ---------------------------------------------------------------------------------------------------

// Contexts are per region: the frame is cut into kRegions x kRegions regions (4 x 4 coarse cells, 16 x 16 pixels of a
// 128-pixel frame). A region's features are either the denoiser's (the mean over the region of enc1, mid and dec1 at
// each noise level of the spec) or plain coarse statistics (the region's normalised cell values, no diffusion). They
// are standardised, reduced by PCA and clustered by k-means (kmeans.hpp), fitted on training states.
inline constexpr int kRegions = 8;

struct ContextSpec {
  bool diffusion = true;            // false: plain coarse statistics
  std::vector<int> ts = {400, 600};
  std::uint64_t noise_seed = 2027;
  int pca = 16;
  std::vector<int> ks = {4, 8, 16};
  std::uint64_t seed = 3;           // k-means starts
};

struct ContextModel {
  ContextSpec spec;
  std::vector<double> mean, sd;  // standardisation of the region features
  Pca pca;
  std::vector<KMeans> km;        // one per spec.ks
};

// Region features of one state (x in network units): kRegions^2 rows. d may be null for plain statistics.
[[nodiscard]] std::vector<double> region_features(const Denoiser* d, const ContextSpec& s, std::span<const float> x,
                                                  std::span<const float> cond, int res, int channels);
// Fits the standardisation, PCA and k-means on the given states (rows of every region of each).
[[nodiscard]] ContextModel fit_contexts(const Denoiser* d, const ContextSpec& s, const Dataset& data, std::span<const int> states,
                                        int res, int channels, int threads);
// The context planes of one coarse state in physical units (res^2 x stride values, the first `channels` physical):
// for each K of the spec, kRegions^2 cluster ids, regions in rows from the bottom. scale: the network units.
[[nodiscard]] std::vector<std::vector<int>> context_planes(const Denoiser* d, const ContextModel& cm, std::span<const float> coarse,
                                                          int stride, std::span<const float> scale, std::span<const float> cond,
                                                          int res);

// Hand-made coarse contexts per region, for the mutual-information check: heat level (0: empty, then terciles of the
// rest), height band (4 bands of two region rows), flow speed (terciles) and a control bin (each control below or
// above one half: 8 values). The thresholds are fitted on training states.
struct HandMade {
  double heat_empty = 0, heat_t1 = 0, heat_t2 = 0, flow_t1 = 0, flow_t2 = 0;
  static constexpr int kKinds = 4;
  static constexpr std::array<const char*, kKinds> kNames = {"heat", "height", "flow", "controls"};
};
[[nodiscard]] HandMade fit_handmade(const Dataset& data, std::span<const int> states, int res, int channels);
// kRegions^2 rows of HandMade::kKinds values (x in network units, controls the effect's first 3 condition values).
[[nodiscard]] std::vector<std::array<int, HandMade::kKinds>> handmade_contexts(const HandMade& h, std::span<const float> x,
                                                                               std::span<const float> controls, int res, int channels);

}  // namespace nfx::dcm::ddpm
