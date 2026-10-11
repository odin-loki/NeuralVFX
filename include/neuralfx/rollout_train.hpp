// Training rollout effects (rollout.hpp) from simulation runs, on the CPU, with hand-written gradients.
//
//   1. record_run: simulate a run and keep its coarse state every frame (velocity, heat, soot averaged over blocks).
//   2. train_stepper: backpropagation through time over windows of up to `max_unroll` frames from true states, with
//      noise added to the inputs (so the stepper learns to recover from its own errors); then fine-tuning on windows
//      that start from the stepper's own rollout, with a loss on where the heat and soot are (row and column profiles),
//      which stays meaningful after the chaos horizon where frame-by-frame errors do not; then a third stage that also
//      matches how much the state changes from frame to frame (squared error alone settles on a smooth average in
//      time, and the effect loses its flicker).
//   3. train_renderer: a per-pixel MLP from true fields to the simulation's frames.
//   4. calibrate_detail: the detail layer's few constants by matching frame statistics (detail spectrum, motion, light,
//      cover) of endless runs from the start points against real runs, at the controls of training runs.
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

// Outside operations on the coarse state before a step, per coarse cell (couplings, docs/COMPOSE.md §9):
//   0, 1  push u, v: added before the step and taken out after it (compose::push, vortex and wind fields)
//   2, 3  force u, v: added and kept (a lasting change of the flow)
//   4     v multiplier, kept (the ceiling field's damping)
//   5     material multiplier on heat and soot (transfer out, suppress)
//   6, 7  heat and soot added (transfer in)
// Velocities in coarse cells per frame. Applied in the order material, v multiplier, force and push.
inline constexpr int kForce = 8;

struct Run {
  sim::Params p;               // re-simulating p reproduces the run exactly (plain runs; forced runs: with their spec)
  int frames = 0;              // states 0..frames-1: state i is after i + 1 simulated frames (time t0 + (i + 1) / fps)
  std::vector<float> coarse;   // frames * res * res * kPhys
  float t0 = 0.f;              // seconds before the step that produced state 0 began (hand-over runs: the explosion's time)
  // Forced runs: forcing_at[i] >= 0 is the slot in `forcing` (res * res * kForce each) of the operations applied
  // before the step that produced state i; -1 (or an empty forcing_at): none.
  std::vector<int> forcing_at;
  std::vector<float> forcing;
  std::vector<float> controls() const { return {p.intensity, p.wind, p.turbulence}; }
  // The operations before the step that produced `state` (per = res * res * kForce), or null.
  const float* forcing_for(int state, std::size_t per) const {
    if (state < 0 || static_cast<std::size_t>(state) >= forcing_at.size() || forcing_at[static_cast<std::size_t>(state)] < 0) return nullptr;
    return forcing.data() + static_cast<std::size_t>(forcing_at[static_cast<std::size_t>(state)]) * per;
  }
};

// Block averages of a simulation state on a res x res grid; velocity converted to coarse cells per frame.
void coarse_from_sim(const sim::State& st, int res, float fps, std::span<float> out);
Run record_run(const sim::Params& p, int frames, int res);

// Apply the operations f (res * res * kForce) to a coarse state of `channels` channels per cell before a step, and
// take the push out again after it. Heat and soot stay at or above zero.
void apply_forcing(std::span<float> coarse, int channels, std::span<const float> f);
void remove_push(std::span<float> coarse, int channels, std::span<const float> f);

// --- couplings in training: forced runs and hand-over runs (docs/COMPOSE.md §9) -----------------------------------

// One outside operation over a span of frames. Positions are in domain units ([0, 1] across the square, y up).
struct Coupling {
  enum class Kind : std::uint8_t { push, force, ceiling, add, remove };
  enum class Shape : std::uint8_t { gust, vortex, wave };  // velocity fields (push, force)
  Kind kind = Kind::push;
  Shape shape = Shape::gust;
  int onset = 0, duration = 1;  // applied before the steps that produce states onset .. onset + duration - 1
  float x = 0.5f, y = 0.5f;     // centre (gust, vortex, add; remove: a disc) at onset
  float dx = 0, dy = 0;         // drift of the centre per frame
  float radius = 0.2f;
  float angle = 0;              // direction of a gust or a wave
  float across = 0, wavelength = 1.f, speed = 0;  // wave: modulated along `across` with this length (domain units),
                                                 // moving by `speed` wavelengths per frame
  float amp = 0;     // push, force: coarse cells per frame at the peak (vortex: at the radius); ceiling: fraction of v
                     // removed per frame at full depth; add: heat per frame at the centre; remove: fraction per frame
  float amp2 = 0;    // add: soot per frame at the centre
  bool band = false;                  // remove: everything above `height` instead of a disc (a fire's smoke leaving
                                      // through its tile top)
  float height = 0.7f, soft = 0.25f;  // ceiling and band: full strength from height + soft up, none below height
  float envelope(int state) const;    // 0 outside the span; ramps in and out over a few frames
};

struct ForcingSpec {
  std::vector<Coupling> events;
  bool active(int state) const;
};

// A random spec for a run of `frames` frames of effect e: a few events of every kind at random onsets (from `first`
// to `last`), durations, places and amplitudes, covering what the fireball scene does (docs/COMPOSE.md §9).
// Deterministic for a seed.
ForcingSpec random_forcing(sim::Effect e, int frames, std::uint64_t seed, int first, int last);
// Strong pushes as scenes give them (study I round 2, docs/COMPOSE.md §10), made of the same events: always a gale (a
// broad push that varies slowly across the domain and drifts like gusts, mostly sideways, for 2 to 5 s; fire up to
// 0.6 cells of the 32-cell grid per frame, smoke 0.4), often a vortex ring rolling through (two opposite vortices
// travelling together; up to 0.7, as the wall of examples/scenes/firewall.nvfxs feels), sometimes a
// short blast (a broad gust, as the fireball's shock on the wreck) and material leaving through the top (the
// fireball's transfer to the sky). The gale starts between `first` and `first + 10`, the rest by `last`.
// Deterministic for a seed.
ForcingSpec scene_forcing(sim::Effect e, int frames, std::uint64_t seed, int first, int last);

// The spec's fields at a state index on an n x n solver grid (rows from the bottom), in solver units: push and force
// in solver cells per second, multipliers, heat and soot. Fields that are not used are left at their neutral value.
struct SimForcing {
  int n = 0;
  std::vector<float> pu, pv, fu, fv, vk, mk, ah, as;
  bool push = false, force = false, material = false;
};
void forcing_fields(const ForcingSpec& spec, const sim::Params& p, int n, int state, SimForcing& out);
// Apply to the simulation before a frame (material, v multiplier, force, push); unpush after it.
void apply_forcing(sim::Fluid& f, const SimForcing& s);
void unpush(sim::Fluid& f, const SimForcing& s);
// The coarse version (res * res * kForce), averaged and scaled exactly as coarse_from_sim.
void coarse_forcing(const SimForcing& s, int res, float fps, std::span<float> out);

// A forced run: the simulation with the spec's operations, recorded with its coarse forcing.
Run record_forced_run(const sim::Params& p, const ForcingSpec& spec, int frames, int res);
// A hand-over run: an explosion (params `from`) simulated for `before` frames, its state set into a simulation with
// params p (the smoke), which is recorded: state 0 is the handed-over state, state i is i frames of p later.
Run record_handover_run(const sim::Params& from, int before, const sim::Params& p, int frames, int res);

struct StepperOptions {
  int iterations = 3000;   // stage 1: windows from true states, unroll growing to max_unroll over the first half
  int finetune = 1500;     // stage 2: half the windows start after up to burn_max frames of the stepper's own rollout
  int batch = 16;
  int max_unroll = 16;
  int burn_max = 48;
  float sigma = 0.03f;     // input noise, in channel scales
  float profile = 1.f;     // weight of the profile loss in stages 2 and 3
  int activity_stage = 0;  // stage 3: windows as in stage 2, plus matching how much the state changes per frame
                           // (recipe_for: 800)
  float activity = 1.f;    // weight of that loss (per channel: mean squared frame-to-frame change, model against truth;
                           // recipe_for: 100)
  float lr = 2e-3f, lr_finetune = 7e-4f;
  float clip = 1.f;        // gradient norm clip
  bool keep_normalisation = false;  // continue training a model: keep its channel scales and range
  // Couplings in training (docs/COMPOSE.md §9): runs from index `plain_runs` on are forced or hand-over runs, and a
  // window is taken from them with probability `coupled_share`. plain_runs < 0: every run is equally likely (as before).
  int plain_runs = -1;
  float coupled_share = 0.f;
  // Round 2 (docs/COMPOSE.md §10): a window from a plain run that starts at a true state (no burn-in) adds
  // anchor * (the first step's squared difference from anchor_model's first step on the same input), in the loss's
  // channel units, so plain one-step predictions stay close to the model fine-tuning started from; and with
  // aim_couplings a window from a forced run is placed so that a coupling acts inside it (up to 16 tries).
  const Model* anchor_model = nullptr;
  float anchor = 0.f;
  bool aim_couplings = false;
  int checkpoint_every = 0;                    // call checkpoint(iterations done) this often (0: never)
  int stop_after = 0;                          // stop after this many iterations, on the full schedule (0: run it all)
  std::function<void(int done)> checkpoint;
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

// Grid search over the contrast, swirl and growth constants, as the effect is used: from the start point nearest each
// run's controls with a seed of its own, `skip` frames of transition, then `frames` frames scored against a real run at
// those controls (another seed, warmed up `warm` frames) by detail-spectrum distance plus the absolute log ratios of
// motion, mean light (emission) and mean cover. Needs start points. Writes the best into m.detail; returns its score.
double calibrate_detail(Model& m, std::span<const sim::Params> runs, int warm, int skip, int frames);

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
SimRecipe recipe_for(sim::Effect e);  // fire, steam, magic as above; smoke keeps 64-pixel fine fields; explosion: 240 runs x 90 frames, 16 starts
sim::Params recipe_run(const SimRecipe& r, std::uint64_t index);
std::vector<Run> record_runs(const SimRecipe& r);
Model recipe_model(const SimRecipe& r);  // hyperparameters, noise and detail settings for the effect, fresh weights

// Stages 3 to 5 on a model whose stepper is trained: renderer, detail calibration, start points; then quantised as
// stored.
struct FinishResult {
  double render_psnr = 0, detail_score = 0;
};
FinishResult finish_model(Model& m, const SimRecipe& r, std::span<const Run> runs);

// Exposed for tests: the loss of one window and its gradient (added to grad, which has step_layout size). With an
// anchor model and weight (StepperOptions::anchor) and no burn-in, the first step's difference from the anchor's is
// added.
double window_loss(const Model& m, const Run& run, int first, int unroll, int burn, float sigma, float profile,
                   std::uint64_t noise_seed, std::vector<float>* grad, float activity = 0.f, const Model* anchor = nullptr,
                   float anchor_weight = 0.f);
// Whether the steps of a window (states first + 1 .. first + unroll after `first`) include a coupling of a forced run.
bool window_has_coupling(const Run& run, int first, int unroll);

}  // namespace nfx::rollout
