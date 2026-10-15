// Training study N's 3D rollout effect (rollout3d.hpp) from 3D simulation runs (sim3d.hpp), on the CPU, with hand-written
// gradients checked against finite differences (tests/test_3d.cpp). Study D's recipe with one more dimension:
//   1. record_run: simulate a run and keep its coarse state every frame (block averages of velocity, heat and soot);
//   2. train_stepper: backpropagation through time over windows from true states, with noise on the inputs; then
//      windows that start after the stepper's own rollout, with a loss on where the heat and soot are (their profiles
//      along each axis); then a stage that also matches how much the state changes from frame to frame;
//   3. choose_starts: start points spread over the control space.
// Training resumes from checkpoints (the weights, Adam's moments and the iteration), with the same result as a run that
// never stopped: every iteration's windows and noise follow from the seed and the iteration's index alone, and the
// gradient is summed over the batch in a fixed order, so the thread count does not change it either.
#pragma once

#include <neuralfx/rollout3d.hpp>
#include <neuralfx/sim3d.hpp>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <span>
#include <stdfloat>
#include <string>
#include <vector>

namespace nfx::rollout3d {

struct Run {
  sim3d::Params p;
  int frames = 0;                     // state i is after i + 1 simulated frames (time (i + 1) / fps)
  std::vector<std::float16_t> coarse;  // frames * cells * kPhys
  std::vector<float> controls() const { return {p.intensity, p.wind, p.turbulence}; }
  float time(int state) const { return static_cast<float>(state + 1) / p.fps; }
  void state(int i, std::size_t per, std::span<float> out) const {  // per = cells * kPhys
    for (std::size_t j = 0; j < per; ++j) out[j] = static_cast<float>(coarse[static_cast<std::size_t>(i) * per + j]);
  }
};

// Block averages of the simulation's fields on the model's coarse grid (the simulation's grid must be the fine grid):
// velocity in coarse cells per frame. out: cells * kPhys.
void coarse_from_sim(const sim3d::Fluid& f, const Hyper& h, std::span<float> out);
Run record_run(const sim3d::Params& p, int frames, const Hyper& h);
std::expected<void, std::string> save_run(const std::filesystem::path& path, const Run& r);
std::expected<Run, std::string> load_run(const std::filesystem::path& path);

// Channel scales (RMS over the runs) and the range the state is kept in (1.5 x the largest magnitude seen).
void set_normalisation(Model& m, std::span<const Run> runs);

struct TrainOptions {
  int iterations = 1500;     // stage 1: windows from true states, the unroll growing to max_unroll over its first half
  int finetune = 600;        // stage 2: half the windows start after up to burn_max frames of the stepper's own rollout
  int activity_stage = 300;  // stage 3: as stage 2, plus matching how much the state changes per frame
  int batch = 8;
  int max_unroll = 12;
  int burn_max = 24;
  float sigma = 0.03f;       // input noise, in channel scales
  float profile = 1.f;       // weight of the profile loss in stages 2 and 3
  float activity = 100.f;    // weight of the activity loss in stage 3
  float lr = 2e-3f, lr_finetune = 7e-4f;
  float clip = 1.f;
  int threads = 0;
  std::uint64_t seed = 1;
  int log_every = 25;
  int checkpoint_every = 50;
  std::function<void(int iteration, int unroll, double loss, double seconds)> progress;
};

// Adam's state and how far training has gone: what a checkpoint holds besides the weights.
struct TrainState {
  int iteration = 0;
  int adam_t = 0;
  std::vector<float> m1, m2;
  std::vector<std::pair<int, double>> curve;
};
std::expected<void, std::string> save_checkpoint(const std::filesystem::path& path, const Model& m, const TrainState& s);
std::expected<void, std::string> load_checkpoint(const std::filesystem::path& path, Model& m, TrainState& s);

// Trains m.step_w from s.iteration to the end of the schedule; calls checkpoint(model, state) every
// checkpoint_every iterations and at the end. m.scale, m.lo and m.hi must be set (set_normalisation).
void train_stepper(Model& m, std::span<const Run> runs, const TrainOptions& o, TrainState& s,
                   const std::function<void(const Model&, const TrainState&)>& checkpoint = {});

// The loss of one window (windows start at state `first`, after `burn` frames of the stepper's own rollout without a
// gradient) and its gradient, added to grad (step_layout size). Exposed for the tests.
double window_loss(const Model& m, const Run& run, int first, int unroll, int burn, float sigma, float profile, std::uint64_t noise_seed,
                   std::vector<float>* grad, float activity = 0.f);

// `count` start points from the runs: spread over the control space (farthest-point order), at state `frame`.
void choose_starts(Model& m, std::span<const Run> runs, int count, int frame);

}  // namespace nfx::rollout3d
