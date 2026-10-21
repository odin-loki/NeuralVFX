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
  // Controls that change during the run (study N2): frames * 3, the controls of the frame that made state i. Empty: the
  // run held p's controls throughout.
  std::vector<float> schedule;
  std::vector<float> controls() const { return {p.intensity, p.wind, p.turbulence}; }
  std::vector<float> controls_at(int state) const {
    if (schedule.empty()) return controls();
    const std::size_t i = static_cast<std::size_t>(state) * 3;
    return {schedule[i], schedule[i + 1], schedule[i + 2]};
  }
  float time(int state) const { return static_cast<float>(state + 1) / p.fps; }
  void state(int i, std::size_t per, std::span<float> out) const {  // per = cells * kPhys
    for (std::size_t j = 0; j < per; ++j) out[j] = static_cast<float>(coarse[static_cast<std::size_t>(i) * per + j]);
  }
};

// Block averages of the simulation's fields on the model's coarse grid (the simulation's grid must be the fine grid):
// velocity in coarse cells per frame. out: cells * kPhys.
void coarse_from_sim(const sim3d::Fluid& f, const Hyper& h, std::span<float> out);
// schedule: empty (p's controls throughout) or frames * 3, the controls of each frame (Run::schedule).
Run record_run(const sim3d::Params& p, int frames, const Hyper& h, std::span<const float> schedule = {});
std::expected<void, std::string> save_run(const std::filesystem::path& path, const Run& r);
std::expected<Run, std::string> load_run(const std::filesystem::path& path);

// Channel scales (RMS over the runs) and the range the state is kept in (1.5 x the largest magnitude seen).
void set_normalisation(Model& m, std::span<const Run> runs);

struct TrainOptions {
  int iterations = 1500;     // stage 1: windows from true states, the unroll growing to max_unroll over its first half
  int finetune = 600;        // stage 2: half the windows start after up to burn_max frames of the stepper's own rollout
  int activity_stage = 300;  // stage 3: as stage 2, plus matching how much the state changes per frame
  int long_stage = 0;        // stage 4 (study N2): as stage 3, every window after burn_long_min to burn_long_max frames
                             // of the stepper's own rollout (from runs long enough to hold it)
  int batch = 8;
  int max_unroll = 12;
  int burn_max = 24;
  int burn_long_min = 48, burn_long_max = 240;
  float sigma = 0.03f;       // input noise, in channel scales
  float profile = 1.f;       // weight of the profile loss in stages 2 and 3
  float activity = 100.f;    // weight of the activity loss in stage 3
  // Study R2 (docs/STUDY_R2.md §2): two more losses in stages 2 to 4, off at 0 (every earlier study's training is
  // unchanged). spread: the plume's width at each height, the model's against the truth's (heat and soot; each row's
  // mass-weighted spread about its own centroid, so a narrow plume that meanders is not punished for being elsewhere,
  // and a broad haze is); edge: heat and soot at the box's four open sides beyond the truth's (not the top, which the
  // plume leaves by).
  float spread = 0.f;
  float edge = 0.f;
  float lr = 2e-3f, lr_finetune = 7e-4f;
  float clip = 1.f;
  int threads = 0;
  std::uint64_t seed = 1;
  int log_every = 25;
  int checkpoint_every = 50;  // and at the end of every stage
  int stop_at = 0;            // stop once this many iterations are done (0: run the whole schedule)
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

// Trains m.step_w from s.iteration to the end of the schedule (or to stop_at); calls checkpoint(model, state) every
// checkpoint_every iterations, at the end of each stage and where it stops. m.scale, m.lo and m.hi must be set (set_normalisation).
// Windows are drawn from the runs in proportion to their frames. A window steps with the controls of each frame
// (Run::controls_at).
void train_stepper(Model& m, std::span<const Run> runs, const TrainOptions& o, TrainState& s,
                   const std::function<void(const Model&, const TrainState&)>& checkpoint = {});

// Study R2's losses for a window (TrainOptions::spread and ::edge).
struct ExtraLosses {
  float spread = 0.f, edge = 0.f;
};

// The loss of one window (windows start at state `first`, after `burn` frames of the stepper's own rollout without a
// gradient) and its gradient, added to grad (step_layout size). Exposed for the tests.
double window_loss(const Model& m, const Run& run, int first, int unroll, int burn, float sigma, float profile, std::uint64_t noise_seed,
                   std::vector<float>* grad, float activity = 0.f, const ExtraLosses& extra = {});

// Study R2: the mass-weighted horizontal spread of channel `ch` about each row's own centroid, per row of the coarse
// grid (cells^2; 0 where the row holds nothing), of a state laid out [cell][channels].
std::vector<double> row_spread(const Hyper& h, std::span<const float> state, int channels, int ch);

// The stage (1 to 4) an iteration belongs to, and the iteration that ends stage k (1 to 4).
int train_stage(const TrainOptions& o, int iteration);
int stage_end(const TrainOptions& o, int stage);

// `count` start points from the runs: spread over the control space (farthest-point order, by the controls at state
// `frame`), at state `frame`.
void choose_starts(Model& m, std::span<const Run> runs, int count, int frame);

}  // namespace nfx::rollout3d
