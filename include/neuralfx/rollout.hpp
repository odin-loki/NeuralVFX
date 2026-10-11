// Rollout effects: learned dynamics from stored start points (docs/REPORT.md §6).
//
// The frame models of model.hpp store an effect's frames (compressed into features) and reproduce them. A rollout
// effect stores almost nothing that changes over time. It keeps:
//   - start points: a few states of the simulation (a coarse grid of velocity, heat and soot), saved from training runs;
//   - a stepper: a small network that moves a coarse state forward by one frame. Advection and a pressure projection
//     are built in; the network supplies the forces and the sub-grid closure, conditioned on the controls and driven by
//     procedural noise (the same kind of noise that forces the simulation);
//   - a detail layer: full-resolution heat and soot carried by the learned flow (stretching by the flow makes the fine
//     filaments), locked to the coarse state, with new material broken up by the flicker noise and a small sub-grid
//     swirl;
//   - a renderer: a per-pixel MLP from the fields to premultiplied RGBA.
// The system is chaotic: after about a second the start point no longer decides the picture, the noise does. So the
// frames in between are not stored and not reproduced exactly; they only have to look right. A new seed gives a new
// run that never repeats.
//
// Everything here is plain float C++ (the reference). The trainer (rollout_train.hpp) differentiates the same
// operations and the runtime (src/runtime/rt_rollout.hpp) runs a faster version; tests check both against this one.
#pragma once

#include <neuralfx/noise.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

namespace nfx::rollout {

inline constexpr int kPhys = 4;     // coarse physical channels: u, v (coarse cells per frame), heat, soot
inline constexpr int kNoise = 2;    // noise inputs of the stepper: curl stream function, flicker
inline constexpr int kCoords = 2;   // position inputs of the stepper (the source is somewhere)
inline constexpr int kDirs = 8;     // directional soot sums seen by the renderer (light and shadow)
inline constexpr int kDirSteps = 6; // coarse cells summed per direction
inline constexpr int kRenderIn = 4 + kDirs;  // fine heat, fine soot, coarse heat, coarse soot, directional sums

// The procedural noise behind the dynamics. Positions X, Y are in cells of a 128-cell frame (Y up), time in seconds.
// With the same seed these equal sim::curl_potential and sim::source_flicker (tests check it).
struct NoiseSpec {
  float curl_scale = 1.f / 14.f, curl_rate = 0.8f;
  float flicker_freq = 0.10f, flicker_rate = 2.6f;
  int flicker_octaves = 3;
};
inline float noise_curl(const NoiseSpec& n, std::uint64_t seed, float X, float Y, float t) {
  return value_noise(X * n.curl_scale, Y * n.curl_scale, t * n.curl_rate, seed * 0x2545F4914F6CDD1DULL + 77);
}
inline float noise_flicker(const NoiseSpec& n, std::uint64_t seed, float X, float Y, float t) {
  return fbm(X * n.flicker_freq, Y * n.flicker_freq, t * n.flicker_rate, seed, n.flicker_octaves);
}

// The detail layer's free constants, calibrated on training runs by matching the statistics of real frames.
struct DetailSpec {
  float contrast = 1.f;      // 0: new material arrives smooth; 1: fully broken up by the flicker noise
  float kappa = 3.f;         // 1 / mean of the contrast curve, so the contrast keeps the amount of new material
  float edge0 = -0.25f, edge1 = 0.6f;  // the contrast curve: smoothstep(edge0, edge1, flicker)
  float swirl = 0.8f;        // sub-grid swirl speed, pixels of a 128-pixel frame per frame
  float swirl_scale = 6.f;   // swirl length, pixels of a 128-pixel frame
  float swirl_rate = 1.5f;   // swirl change per second
  float swirl_ramp = 0.5f;   // seconds over which the swirl fades in after a start point (keeps the start's look)
  int swirl_control = -1;    // a control that scales the swirl by (0.3 + value), e.g. turbulence; -1: none
  float grow = 1.f;          // the lock may scale a block's existing fine structure up by this much (peaks stay
                             // peaks); only the rest arrives as new material (1: all increases arrive as new material)
  // Study D2 (docs/REPORT.md §6.9; file version 4, written only when one of them differs from its default):
  float advect = 1.f;         // the fine fields move with this multiple of the coarse flow (the swirl is added as is)
  float soften = 0.f;         // each frame, before advection, the fine fields move this fraction of the way to the mean
                              // of their four neighbours (zero outside the frame): the fine-scale diffusion of the
                              // simulation, which the detail layer otherwise lacks
  bool d2() const { return advect != 1.f || soften != 0.f; }
};

// Study D2's softening of a fine field (DetailSpec::soften): q + s (mean of the four neighbours - q), zero outside.
void soften_field(std::span<float> q, int size, float s);

struct Hyper {
  int res = 32;            // coarse cells per side
  int hidden = 24;         // stepper width
  int memory = 4;          // hidden state channels carried from frame to frame (bounded by tanh)
  int jacobi = 60;         // pressure iterations per frame (warm-started from the last frame)
  int n_controls = 3;
  int n_age = 0;           // one-shot effects: exp(-t / 0.15 s), exp(-t / 1 s) join the controls in the FiLM input
  int frames = 0;          // one-shot effects: frames until it holds its last frame (0: endless)
  int render_hidden = 16;  // renderer MLP width (two hidden layers)
  int start_fine = 0;      // side of the fine fields kept with each start point (0: grown from the coarse state)
  int warmup = 30;         // frames run at instance start when a start point has no fine fields

  int channels() const { return kPhys + memory; }
  int inputs() const { return channels() + kNoise + kCoords; }
  int outputs() const { return channels() + 1; }  // state change, then a divergence source (expansion)
  int cond() const { return n_controls + n_age; }
};

// Offsets of the stepper's weights in one float array. Convolution weights are [tap][in][out], tap = (dy+1)*3+(dx+1).
struct StepLayout {
  std::size_t w1, b1, w2, b2, wo, bo, g1, e1, g2, e2, size;
};
StepLayout step_layout(const Hyper& h);

// Offsets of the renderer's weights: two hidden layers and the output, each [out][in] then a bias.
struct RenderLayout {
  std::size_t w1, b1, w2, b2, wo, bo, size;
};
RenderLayout render_layout(const Hyper& h);

struct StartPoint {
  std::vector<float> controls;        // the run's controls
  std::uint64_t seed = 0;             // the run's noise seed (replaying it with this seed tracks the run)
  float time = 0;                     // seconds since the run began (noise phase, age)
  std::vector<float> coarse;          // res * res * kPhys, rows from the bottom
  std::vector<float> fine_t, fine_d;  // start_fine^2 each (rows from the bottom), or empty
};

struct Model {
  Hyper h;
  NoiseSpec noise;
  DetailSpec detail;
  std::string effect;
  float fps = 30.f;
  bool loop = true;
  std::vector<std::string> control_names;
  std::array<float, kPhys> scale{1, 1, 1, 1};  // typical size of each physical channel (network input/output units)
  std::array<float, kPhys> lo{}, hi{};         // the state is kept inside the range seen in training (with margin)
  float qscale = 0.05f;                        // divergence source units
  std::array<float, 2> render_scale{1, 1};     // input normalisation of fine and coarse heat / soot for the renderer
  std::vector<float> step_w;                   // step_layout(h).size
  std::vector<float> render_w;                 // render_layout(h).size
  std::vector<StartPoint> starts;
  // Storage of the coarse start states (results/compression, study F2, design G3c): 16 = fp16 (file versions 1
  // and 2); 2 to 8 = codes per channel plane with an fp16 (lo, hi) per start and channel, bit-packed below 8 (file
  // version 3). With start_dither, the codes are dithered subtractively by the start's seed hashed per cell
  // (start_dither_value): stored q = round((v + d - lo) / step), restored v = lo + q step - d, so the error is uniform
  // and unrelated to the field; a zero code of a channel whose range starts at zero restores exactly zero.
  int start_bits = 16;
  bool start_dither = false;

  std::size_t storage_bytes() const;  // as saved: weights fp16, coarse states at start_bits, fine fields 8-bit
};

// The dither of a quantised start state at cell i (of res * res), channel c, for a start with this seed: in [-0.5, 0.5)
// steps, from the same integer hash as the effect's noise.
float start_dither_value(std::uint64_t seed, int i, int c);

// Fresh weights (He initialisation, small output layer), no start points.
Model init_model(const Hyper& h, std::uint64_t seed);

// .nvfx files with the magic NVFXROL1 (the runtime tells the two kinds apart by the magic).
inline constexpr char kMagic[8] = {'N', 'V', 'F', 'X', 'R', 'O', 'L', '1'};
std::expected<void, std::string> save_model(const std::filesystem::path& path, const Model& m);
std::expected<void, std::string> save_model(std::ostream& out, const Model& m);
std::expected<Model, std::string> load_model(const std::filesystem::path& path);
std::expected<Model, std::string> load_model(std::istream& in);
bool is_rollout_file(std::span<const char> first_bytes);
// Weights and start states rounded as saving would round them, so evaluation sees what ships.
void quantise_like_storage(Model& m);

// --- reference evaluation ---------------------------------------------------------------------------------------------

// The condition vector (controls, then age features) at a time since the effect began.
void condition(const Model& m, std::span<const float> controls, float seconds, std::span<float> out);

// The stepper's noise inputs on the coarse grid for the frame that starts at `seconds`: [cell][2].
void coarse_noise(const Model& m, std::uint64_t seed, float seconds, std::span<float> out);

// A running effect: coarse state, pressure, fine fields. All sizes fixed at construction.
struct State {
  int res = 0, size = 0;
  float time = 0;          // seconds since the effect began
  float since_start = 0;   // seconds since the start point
  std::vector<float> coarse;    // res * res * channels
  std::vector<float> pressure;  // res * res
  std::vector<float> flow;      // res * res * 2: the projected velocity of the last step (moves the fine fields)
  std::vector<float> fine_t, fine_d;  // size * size
};

// Start an effect at `size` pixels from start point `index` (time and coarse state from it; fine fields from it or
// grown from the coarse state by h.warmup frames with `seed`).
State start(const Model& m, int index, int size, std::span<const float> controls, std::uint64_t seed);
// One frame: the stepper on the coarse grid, then the detail layer.
void step(const Model& m, State& s, std::span<const float> controls, std::uint64_t seed);
// Premultiplied RGBA floats [size][size][4], rows top to bottom.
void render(const Model& m, const State& s, std::span<float> rgba);

// Pieces of step(), exposed for the trainer and the tests.
// Coarse: in (res*res*channels), pressure in/out, out (res*res*channels), flow (res*res*2).
void coarse_step(const Model& m, std::span<const float> in, std::span<const float> noise, std::span<const float> cond,
                 std::span<float> pressure, std::span<float> out, std::span<float> flow);
// Detail: advect the fine fields by the flow (plus swirl), lock their coarse averages to the coarse heat and soot.
void detail_step(const Model& m, State& s, std::uint64_t seed, std::span<const float> controls);
// Renderer input features for one pixel (kRenderIn values), from fine fields and the coarse state.
void render_features(const Model& m, const State& s, int x, int y, std::span<float> out);
// The renderer MLP on one feature vector, times the material gate.
std::array<float, 4> render_mlp(const Model& m, std::span<const float> features);
// Where there is no heat and no soot nothing is drawn: the MLP's output is scaled by this gate on the normalised fine
// fields (features 0 and 1), so empty pixels are exactly transparent instead of carrying the network's bias as haze.
inline float render_gate(float fine_heat, float fine_soot) { return std::min(1.f, 50.f * (std::max(0.f, fine_heat) + std::max(0.f, fine_soot))); }

}  // namespace nfx::rollout
