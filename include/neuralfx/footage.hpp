// From footage to rollout effects (docs/FOOTAGE.md): estimate the state of an effect from a few frames, so a rollout
// effect (rollout.hpp) can continue from it.
//
// A rollout effect starts from stored states of the simulation: a coarse grid of velocity, heat and soot, plus fine heat
// and soot. Footage has none of these, only pictures. The path here, all on the CPU:
//   1. Fine heat and soot from each frame: a small inverse network (a per-pixel MLP over a pyramid of 3 x 3
//      neighbourhoods of the frame) trained on the simulator's own (frame, fields) pairs, with mild blur and noise.
//      Optionally the start frame's fields are then refined by gradient descent through the effect's own learned
//      renderer, so that the effect redraws the footage frame.
//   2. Coarse heat and soot: block averages of the fine fields (as the simulation's states are averaged).
//   3. Coarse velocity from motion: block matching between the estimated fields of consecutive frames, then
//      assimilation through the effect's own stepper: it is run over the context frames, with its heat and soot
//      replaced by the estimates after every step and its velocity relaxed towards the measured motion, so the flow it
//      ends with is one its own dynamics agree with.
// What comes out is a start point (rollout::StartPoint) and an effect file with it. Validated on simulated renders
// only: the stand-in footage is the simulator's own frames (and the learned renderer's), with known true states.
//
// Conventions: a frame is float premultiplied RGBA in [0, 1], [size][size][4], rows top to bottom (a Clip frame / 255).
// A field is [size][size], rows from the bottom (y up), as the simulation's and the rollout's fields; pixel (x, row r)
// of a frame is field cell (x, size - 1 - r).
#pragma once

#include <neuralfx/rollout.hpp>
#include <neuralfx/sim.hpp>

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

namespace nfx::footage {

using Frame = std::vector<float>;  // size * size * 4

// Fine heat and soot of one frame, rows from the bottom.
struct Fields {
  int size = 0;
  std::vector<float> heat, soot;
};

// 8-bit frames to floats and back.
Frame to_float(std::span<const std::uint8_t> rgba);
void to_u8(std::span<const float> rgba, std::span<std::uint8_t> out);

// --- stand-in capture: what a camera and a codec do to frames ---------------------------------------------------------

struct Degradation {
  float blur = 0.f;         // Gaussian sigma in pixels (0: none), on premultiplied RGBA
  float noise = 0.f;        // Gaussian noise sigma in 8-bit steps / 255 units (e.g. 2 / 255)
  bool drop_alpha = false;  // footage without alpha: colour over black, alpha = max(r, g, b) (nvfx_ingest --alpha luma)
};
// In place on an 8-bit premultiplied RGBA frame; noise is deterministic for a seed; the result is rounded to 8 bits.
void degrade(std::span<std::uint8_t> rgba, int size, const Degradation& d, std::uint64_t seed);

// --- the inverse network: frame -> fine heat and soot --------------------------------------------------------------

inline constexpr char kInverseMagic[8] = {'N', 'V', 'F', 'X', 'I', 'N', 'V', '1'};

struct InverseSpec {
  int hidden = 32;
  int levels = 5;     // pyramid levels: neighbourhoods of 1, 2, 4, 8, 16 pixels (about 24 pixels of context)
  bool alpha = true;  // false: colour only (footage without alpha; the alpha channel is ignored)
};

// The motion network: the coarse velocity at a frame from that frame and the two before it. A per-cell MLP on the coarse
// grid over the cell's 5 x 5 neighbourhood of coarse heat and soot in the three frames, the block-matched flow into the
// frame (3 x 3 cells, with whether each was measured) and into the frame before (the cell), and the cell's position.
// Trained on the simulation's runs with the inverse network's fields, so it learns what block matching cannot see:
// the flow where there is no material, the plume's speed from its heat.
struct Motion {
  static constexpr int kInputs = 3 * 2 * 25 + 3 * 9 + 3 + 2;
  int hidden = 48;
  std::array<float, 2> in_scale{1, 1};  // coarse heat and soot units of the inputs
  float out_scale = 1.f;                // velocity units of the outputs (coarse cells per frame)
  std::vector<float> w;                 // input-major like the inverse network's; empty: no motion network
  bool empty() const { return w.empty(); }
};

struct Inverse {
  InverseSpec spec;
  std::string effect;
  int size = 128;                       // frame size it was trained at
  std::array<float, 2> scale{1, 1};     // output units of heat and soot
  std::vector<float> w;                 // [inputs][hidden], [hidden], [hidden][hidden], [hidden], [hidden][2], [2] (input-major)
  Motion motion;                        // optional (file version 2)
  int channels() const { return spec.alpha ? 4 : 3; }
  int inputs() const { return spec.levels * 9 * channels() + 2; }
  std::size_t weights() const;
};

Inverse init_inverse(const InverseSpec& spec, std::uint64_t seed);
std::expected<void, std::string> save_inverse(const std::filesystem::path& path, const Inverse& inv);
std::expected<void, std::string> save_inverse(std::ostream& out, const Inverse& inv);
std::expected<Inverse, std::string> load_inverse(const std::filesystem::path& path);
std::expected<Inverse, std::string> load_inverse(std::istream& in);

// The input features of every pixel of a frame: inputs() values per field cell, cells rows from the bottom.
// A pyramid of box-filtered copies of the frame (each level half the size), sampled bilinearly (zero outside) at the
// 3 x 3 neighbourhood of the pixel on every level, then the pixel's position (x and y in [-1, 1]).
class Features {
 public:
  Features(const Inverse& inv, std::span<const float> frame, int size);
  void at(int x, int y, std::span<float> out) const;  // field cell (x, y), y up
  int size() const { return size_; }

 private:
  int size_, levels_, channels_;
  std::vector<std::vector<float>> pyramid_;  // per level: [n][n][channels], rows from the bottom
};

// Fine heat and soot of one frame. Pixels with nothing visible within about 3 pixels are set to zero.
Fields apply_inverse(const Inverse& inv, std::span<const float> frame, int size);

struct InverseSample {
  int size = 0;
  Frame frame;      // the frame as the footage would show it (degraded if so)
  Fields truth;     // the true fine fields
};

struct InverseTrainOptions {
  int iterations = 6000;
  int batch = 1024;  // pixels per iteration, half of them where something is visible or present
  float lr = 2e-3f;
  int threads = 1;
  std::uint64_t seed = 3;
  std::function<void(int iteration, double loss)> progress;
};
// Sets inv.scale from the samples (99.5th percentile of present heat and soot) and trains inv.w. Returns the mean loss
// (squared error in output units) over the last 5% of iterations.
double train_inverse(Inverse& inv, std::span<const InverseSample> samples, const InverseTrainOptions& o);

// PSNR of an estimated field against the truth with the peak at `scale` (the effect's typical value), all pixels.
double field_psnr(std::span<const float> estimate, std::span<const float> truth, float scale);

// --- refinement through the effect's learned renderer ------------------------------------------------------------

struct RefineOptions {
  int iterations = 100;
  float lr = 0.02f;      // Adam step, in units of the renderer's input scale
  float prior = 0.05f;   // weight of staying near the starting fields (normalised squared distance per pixel)
  bool alpha = true;     // match alpha too (false: footage without alpha)
};
// The effect's renderer on fine fields (coarse heat and soot are their block averages): what the rollout would draw.
void render_fields(const rollout::Model& m, const Fields& f, std::span<float> rgba);
// Gradient descent on the fine fields so that the effect's renderer draws `frame`, starting from (and kept near) f.
// Returns the final mean squared error of the matched channels.
double refine_fields(const rollout::Model& m, std::span<const float> frame, Fields& f, const RefineOptions& o);
// The loss refine_fields minimises and its gradient with respect to the fields (in field units), for the tests.
double refine_loss(const rollout::Model& m, std::span<const float> frame, const Fields& f, const Fields& start, const RefineOptions& o,
                   Fields* grad);

// --- coarse state -------------------------------------------------------------------------------------------------

// Block averages of fine fields on the res x res grid: res * res * 2 (heat, soot).
std::vector<float> coarse_fields(const Fields& f, int res);

// Motion between the fields of two consecutive frames by block matching, on the coarse grid: displacement per frame in
// coarse cells (res * res * 2: u, v, y up), and a mask of cells where it was measured (enough material in the window).
struct FlowOptions {
  int radius = 6;         // search radius in pixels of the frame
  int window = 6;         // half-size of the matching window in pixels (13 x 13)
  float min_mass = 0.02f; // a window needs this much normalised material on average to be measured
  int smooth = 1;         // 3 x 3 box passes after filling
  bool fill = true;       // unmeasured cells: filled by diffusion from measured ones (false: zero)
};
struct Flow {
  int res = 0;
  std::vector<float> uv;            // res * res * 2
  std::vector<std::uint8_t> known;  // res * res
};
// As measured: unmeasured cells are zero and not `known`.
Flow block_flow(const Fields& a, const Fields& b, int res, std::array<float, 2> scale, const FlowOptions& o);
// The mean of measured flows, per cell over the flows that measured it.
Flow mean_flow(std::span<const Flow> flows);
// Unmeasured cells filled (by diffusion from measured ones, or zero) and the field smoothed; `known` is kept.
void finish_flow(Flow& f, const FlowOptions& o);

// --- the motion network (see Motion above) -----------------------------------------------------------------------

Motion init_motion(int hidden, std::uint64_t seed);
// Inputs of every coarse cell (res * res * Motion::kInputs) from three consecutive frames' fields (oldest first) and the
// raw block-matched flows between them (into the middle frame, into the last), as block_flow returns them.
std::vector<float> motion_inputs(const Motion& mo, std::span<const Fields> three, const Flow& into_middle, const Flow& into_last, int res);
// The velocity of every cell (every cell `known`).
Flow apply_motion(const Motion& mo, std::span<const float> inputs, int res);
struct MotionSample {
  std::vector<float> inputs;  // cells * Motion::kInputs
  std::vector<float> target;  // cells * 2: the true coarse velocity
};
// Sets mo.out_scale (RMS of the targets) and trains mo.w; returns the mean loss over the last 5% (output units).
double train_motion(Motion& mo, std::span<const MotionSample> samples, const InverseTrainOptions& o);


// Assimilation through the stepper. Fields of the context frames (oldest first; the last is the start frame) and the
// flow before each of them (flows[i]: from frame i - 1 to frame i; flows[0] may be empty: the next one is used).
struct AssimOptions {
  bool stepper = true;         // false: no assimilation; velocity is the measurement into the start frame (or zero)
  bool use_flow = true;        // start velocity from the measured flow (false: zero)
  float nudge_velocity = 0.5f; // after each step, velocity <- (1 - n) * stepper's + n * measured (where measured)
  std::vector<float> controls; // the controls during the footage (empty: 0.5 each)
  float time = 0.f;            // seconds since the effect began at the start frame (one-shot effects: its age)
  std::uint64_t seed = 1;      // noise seed of the assimilation (footage has none; this is not the run's seed)
};
// The coarse state at the start frame (res * res * kPhys): heat and soot are the start frame's block averages, velocity
// the stepper's after running through the context.
std::vector<float> assimilate(const rollout::Model& m, std::span<const Fields> fields, std::span<const Flow> flows, const AssimOptions& o);

// --- the whole estimate -----------------------------------------------------------------------------------------------

struct EstimateOptions {
  int context = 8;  // frames used, ending at the start frame
  bool refine = true;
  RefineOptions refine_opt;
  FlowOptions flow_opt;
  int flow_pairs = 1;  // frame pairs averaged into the measured flow before each step (1: the last pair only)
  bool motion = true;  // the motion network measures velocity where it has three frames (if the inverse file has one)
  AssimOptions assim;
};
struct Estimate {
  rollout::StartPoint start;       // controls, seed, time, coarse state; fine fields at frame size (h.start_fine set
                                   // by with_starts)
  std::vector<Fields> fields;      // per context frame, as estimated (the last refined if asked)
  std::vector<Flow> flows;         // per context frame (the first empty)
};
// `frames`: consecutive footage frames, oldest first; the last is the start frame. The inverse must match the footage
// (with alpha or not).
Estimate estimate_start(const rollout::Model& m, const Inverse& inv, std::span<const Frame> frames, int size, const EstimateOptions& o);

// A start point from a coarse state and fine fields at frame size.
rollout::StartPoint make_start(std::vector<float> coarse, const Fields& fine, std::span<const float> controls, std::uint64_t seed, float time);

// --- training samples and stand-in footage from the built-in simulation (src/footage/footage_sim.cpp) -----------------

struct SampleOptions {
  int runs = 48;               // training runs of the effect's recipe (rollout::recipe_run with `salt`)
  int per_run = 6;             // frames per run, at random times (looping effects: from frame 20; explosions: any)
  std::uint64_t salt = 1;      // 1: the runs the effects were trained on
  float degrade_share = 0.5f;  // share of frames given random blur (sigma up to 1.2 px) and noise (up to 3 / 255)
  bool drop_alpha = false;     // footage without alpha (the inverse ignores alpha then; kept for the record)
  int size = 128;
  int threads = 1;
};
std::vector<InverseSample> simulate_samples(sim::Effect e, const SampleOptions& o);
// Samples for the motion network: three consecutive frames per sample (degraded alike), through the inverse network,
// with the true coarse velocity of the last; `cells` cells per sample, half where there is material.
std::vector<MotionSample> simulate_motion_samples(sim::Effect e, const Inverse& inv, const SampleOptions& o, int cells = 256);

// A copy of the effect whose start points are `starts` (after the stored ones when keep_stored). Fine fields are kept
// at fine_size (box-averaged from frame size); when stored start points keep fine fields, their size is used instead.
rollout::Model with_starts(const rollout::Model& m, std::span<const rollout::StartPoint> starts, int fine_size, bool keep_stored);

}  // namespace nfx::footage
