// Training rollout effects (rollout.hpp) from simulation runs, on the CPU, with hand-written gradients.
//
//   1. record_run: simulate a run and keep its coarse state every frame (velocity, heat, soot averaged over blocks).
//   2. train_stepper: backpropagation through time over windows of up to `max_unroll` frames from true states, with
//      noise added to the inputs (so the stepper learns to recover from its own errors); then fine-tuning on windows
//      that start from the stepper's own rollout, with a loss on where the heat and soot are (row and column profiles),
//      which stays meaningful after the chaos horizon where frame-by-frame errors do not.
//   3. train_renderer: a per-pixel MLP from true fields to the simulation's frames.
//   4. calibrate_detail: the detail layer's few constants by matching frame statistics (detail spectrum, motion) on
//      training runs.
//   5. choose_starts: start points spread over the control space.
// Nothing here is evaluated on the runs it trained on; nvfx_experiment holds out seeds and settings.
#pragma once

#include <neuralfx/rollout.hpp>
#include <neuralfx/sim.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <span>
#include <vector>

namespace nfx::rollout {

struct Run {
  sim::Params p;               // re-simulating p reproduces the run exactly
  int frames = 0;              // states 0..frames-1: state i is after i + 1 simulated frames (time (i + 1) / fps)
  std::vector<float> coarse;   // frames * res * res * kPhys
  std::vector<float> controls() const { return {p.intensity, p.wind, p.turbulence}; }
};

// Block averages of a simulation state on a res x res grid; velocity converted to coarse cells per frame.
void coarse_from_sim(const sim::State& st, int res, float fps, std::span<float> out);
Run record_run(const sim::Params& p, int frames, int res);

struct StepperOptions {
  int iterations = 3000;   // stage 1: windows from true states, unroll growing to max_unroll over the first half
  int finetune = 1500;     // stage 2: half the windows start after up to burn_max frames of the stepper's own rollout
  int batch = 16;
  int max_unroll = 16;
  int burn_max = 48;
  float sigma = 0.03f;     // input noise, in channel scales
  float profile = 1.f;     // weight of the profile loss in stage 2
  float lr = 2e-3f, lr_finetune = 7e-4f;
  float clip = 1.f;        // gradient norm clip
  int threads = 0;
  std::uint64_t seed = 1;
  int log_every = 250;
  std::function<void(int iteration, int unroll, double loss)> progress;
};

struct StepperResult {
  double final_loss = 0;  // mean over the last 5% of stage 2 (or stage 1 when there is no stage 2)
  double seconds = 0;
  std::vector<std::pair<int, double>> curve;
};

// Sets m.scale, m.lo, m.hi from the runs, then trains m.step_w. m.h, m.noise and m.fps must be set.
StepperResult train_stepper(Model& m, std::span<const Run> runs, const StepperOptions& o);

// One frame for the renderer: true fine fields (size^2), the coarse state (res^2 * kPhys) and the simulation's frame.
struct RenderSample {
  int size = 0;
  std::vector<float> fine_t, fine_d, coarse;
  std::vector<std::uint8_t> rgba;  // premultiplied RGBA8, rows top to bottom
};
RenderSample render_sample(const sim::Fluid& f, int res);

struct RendererOptions {
  int iterations = 3000;
  int batch_frames = 8;
  int pixels = 4096;
  float lr = 3e-3f;
  int threads = 0;
  std::uint64_t seed = 2;
  std::function<void(int iteration, double loss)> progress;
};
// Sets m.render_scale and trains m.render_w. Returns PSNR on the samples' sampled pixels.
double train_renderer(Model& m, std::span<const RenderSample> samples, const RendererOptions& o);

// Grid search over the contrast and swirl constants on `runs` (training runs): rollouts of `frames` frames from a true
// state at frame `from` (with its fine fields), scored by detail-spectrum distance plus |log motion ratio| against the
// true frames. Writes the best into m.detail and returns its score.
double calibrate_detail(Model& m, std::span<const sim::Params> runs, int from, int frames);

// `count` start points from the runs: spread over the control space (farthest-point order), at frame `frame` (looping
// effects: a steady frame; one-shot effects: the first). Fine fields are kept when m.h.start_fine > 0.
void choose_starts(Model& m, std::span<const Run> runs, int count, int frame);

// --- the whole recipe for an effect of the built-in simulation (nvfx_train --rollout, nvfx_experiment d-train) -------

struct SimRecipe {
  sim::Effect effect = sim::Effect::fire;
  int runs = 160;          // training runs: controls uniform in [0, 1]^3, each with its own seed
  int frames = 240;        // frames recorded per run, from an empty grid (so the start-up is learned too)
  int start_frame = 150;   // where start points are taken (a steady frame; one-shot effects: the first)
  int start_fine = 0;      // fine fields kept with each start point (0: grown at instance start)
  int starts = 8;
  int render_runs = 100;   // runs that give two renderer samples each
  std::uint64_t salt = 1;  // which runs (held-out runs use another salt)
  int threads = 0;
  StepperOptions stepper;
  RendererOptions renderer;
  std::function<void(const std::string&)> log;
};
SimRecipe recipe_for(sim::Effect e);  // fire as above; smoke keeps 64-pixel fine fields; explosion: 240 runs x 90 frames, 16 starts
sim::Params recipe_run(const SimRecipe& r, std::uint64_t index);
std::vector<Run> record_runs(const SimRecipe& r);
Model recipe_model(const SimRecipe& r);  // hyperparameters, noise and detail settings for the effect, fresh weights

// Stages 3 to 5 on a model whose stepper is trained: renderer, detail calibration, start points; then quantised as
// stored.
struct FinishResult {
  double render_psnr = 0, detail_score = 0;
};
FinishResult finish_model(Model& m, const SimRecipe& r, std::span<const Run> runs);

// Exposed for tests: the loss of one window and its gradient (added to grad, which has step_layout size).
double window_loss(const Model& m, const Run& run, int first, int unroll, int burn, float sigma, float profile,
                   std::uint64_t noise_seed, std::vector<float>* grad);

}  // namespace nfx::rollout
