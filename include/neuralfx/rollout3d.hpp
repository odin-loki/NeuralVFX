// Study N's 3D rollout effect (docs/STUDY_3D.md §4): study D's model (rollout.hpp) with one more dimension.
//
// A 3D rollout effect keeps:
//   - start points: a few coarse states of the 3D simulation (sim3d.hpp), velocity, heat and soot averaged over blocks,
//     saved from training runs at 8 bits per value;
//   - a stepper: two learned 3 x 3 x 3 convolutions with FiLM on the controls, fed the state, the simulation's own
//     seeded noise (the curl force and the flicker, noise4.hpp) and the position; it supplies the change of every
//     channel and a divergence source. Advection (semi-Lagrangian, trilinear; ambient air flows in at the open sides
//     and top, see Edges) and a pressure projection (Jacobi, warm-started from the last frame; closed floor, open top
//     and sides) are built in;
//   - a detail layer: fine fields of heat and soot (4 fine cells per coarse cell along each axis) carried by the
//     up-sampled coarse flow plus a seeded sub-grid swirl, locked to the coarse state block by block; material the
//     coarse state adds arrives broken up by the flicker noise. What is ordered (the plume, its lean, its puffing) is
//     learned; what is random at the scales the coarse grid cannot hold is seeded, not learned (study M's principle).
// The picture comes from the volume renderer (render3d.hpp) on the fine fields: there is no learned renderer.
//
// The coarse grid has rx x ry x rz cells (x across, y up, z in depth), indexed (y * rz + z) * rx + x, channels last
// ([cell][channel]); the fine fields likewise at (rx, ry, rz) * fine. Everything here is plain float C++ (the
// reference): the trainer (rollout3d_train.hpp) differentiates the same operations and the volume runtime
// (nvfx_volume.h) runs this code with its own convolution kernel and threads; tests hold both to it.
#pragma once

#include <neuralfx/noise4.hpp>
#include <neuralfx/parallel.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

namespace nfx::rollout3d {

inline constexpr int kPhys = 5;    // coarse physical channels: u, v, w (coarse cells per frame), heat, soot
inline constexpr int kNoise = 4;   // noise inputs: the curl force (3 components) and the flicker
inline constexpr int kCoords = 3;  // position inputs
inline constexpr int kTaps = 27;   // 3 x 3 x 3

struct Hyper {
  int rx = 16, ry = 24, rz = 16;  // coarse cells
  int fine = 4;                   // fine cells per coarse cell along each axis
  int hidden = 16;                // stepper width
  int memory = 4;                 // hidden state channels carried from frame to frame (bounded by tanh)
  int jacobi = 40;                // pressure iterations per frame (warm-started)
  int n_controls = 3;
  int warmup = 24;                // frames run at instance start to grow the fine fields from the coarse state

  int channels() const { return kPhys + memory; }
  int inputs() const { return channels() + kNoise + kCoords; }
  int outputs() const { return channels() + 1; }  // the state's change, then a divergence source
  int cond() const { return n_controls; }
  int cells() const { return rx * ry * rz; }
  int fx() const { return rx * fine; }
  int fy() const { return ry * fine; }
  int fz() const { return rz * fine; }
  std::size_t fine_cells() const { return static_cast<std::size_t>(fx()) * static_cast<std::size_t>(fy()) * static_cast<std::size_t>(fz()); }
};

// Weights in one float array. Convolution weights are [tap][in][out], tap = ((dy + 1) * 3 + (dz + 1)) * 3 + (dx + 1).
struct StepLayout {
  std::size_t w1, b1, w2, b2, wo, bo, g1, e1, g2, e2, size;
};
StepLayout step_layout(const Hyper& h);

// The detail layer's constants (lengths in units of a 128-unit width; speeds in fine cells per frame at 64 fine cells
// across, scaled with the fine grid).
struct DetailSpec {
  float contrast = 1.f;            // 0: new material arrives smooth; 1: fully broken up by the flicker noise
  float kappa = 3.f;               // 1 / mean of the contrast curve, so the contrast keeps the amount of new material
  float edge0 = -0.25f, edge1 = 0.6f;  // the contrast curve: smoothstep(edge0, edge1, flicker)
  float swirl = 0.5f;              // the sub-grid swirl's speed
  float swirl_scale = 12.f;        // its length (the curl noise's, noise4.hpp)
  float swirl_rate = 1.5f;         // its change per second
  float swirl_ramp = 0.5f;         // seconds over which the swirl fades in after a start point
  float grow = 1.5f;               // the lock may scale a block's existing structure up by this much
};

// What the coarse advection samples where a back-traced position leaves the grid through an open side or the top
// (the floor is closed: there it always clamps). Study N's models clamped to the edge cell, so a side where air flows
// in kept and fed its own heat (docs/STUDY_3D.md §11); study N2's bring in ambient air: still, with no heat, soot or
// memory (§14). Files of version 1 are study N's (clamp); version 2 stores the mode.
enum class Edges : std::uint32_t { clamp = 0, ambient = 1 };

struct StartPoint {
  std::vector<float> controls;
  std::uint64_t seed = 0;
  float time = 0;
  std::vector<float> coarse;  // cells * kPhys
};

struct Model {
  Hyper h;
  Forcing3 noise;
  DetailSpec detail;
  std::string effect;
  float fps = 30.f;
  std::vector<std::string> control_names;
  std::array<float, kPhys> scale{1, 1, 1, 1, 1};  // typical size of each physical channel (network units)
  std::array<float, kPhys> lo{}, hi{};            // the state is kept inside the range seen in training
  float qscale = 0.05f;                           // divergence source units
  Edges edges = Edges::ambient;                   // the coarse advection's open sides and top
  std::vector<float> step_w;
  std::vector<StartPoint> starts;
  std::size_t storage_bytes() const;  // as saved: weights fp16, start states 8-bit
};

Model init_model(const Hyper& h, std::uint64_t seed);

// .nvfx3 files: magic NVFXVOL1. Version 1 (study N): clamped edges; version 2 adds the edge mode. A model with clamped
// edges is saved as version 1, so study N's files load and save as they were.
inline constexpr char kMagic[8] = {'N', 'V', 'F', 'X', 'V', 'O', 'L', '1'};
std::expected<void, std::string> save_model(const std::filesystem::path& path, const Model& m);
std::expected<void, std::string> save_model(std::ostream& out, const Model& m);
std::expected<Model, std::string> load_model(const std::filesystem::path& path);
std::expected<Model, std::string> load_model(std::istream& in);
// Weights and start states rounded as saving rounds them, so evaluation sees what ships.
void quantise_like_storage(Model& m);

// --- the reference --------------------------------------------------------------------------------------------------

// A 3 x 3 x 3 convolution over output slabs y0 <= y < y1: in [cell][ci], W [tap][ci][co], b [co] -> out [cell][co]
// (zero outside the grid). The runtime may pass its own (the same sums in another order).
using Conv3Fn = void (*)(int rx, int ry, int rz, const float* in, int ci, const float* W, const float* b, int co, float* out, int y0, int y1);
void conv3_reference(int rx, int ry, int rz, const float* in, int ci, const float* W, const float* b, int co, float* out, int y0, int y1);

struct Workspace;
struct State;

// A detail layer the runtime may pass in place of detail_step (step() and start() call it when Workspace::detail is
// set): the reference's values to the bit, faster (src/runtime/rt_detail3.hpp, on vectors per ISA). Sized for one model
// when made; step() allocates nothing.
class DetailKernel {
 public:
  virtual ~DetailKernel() = default;
  virtual void step(const Model& m, State& s, std::uint64_t seed, std::span<const float> controls, Workspace& ws, Pool* pool) = 0;
  virtual std::size_t bytes() const = 0;  // its own buffers
};

// Scratch for one runner: every buffer a step needs, sized once (no allocation per frame).
struct Workspace {
  explicit Workspace(const Model& m);
  std::vector<float> X, y1, y2, d, mid, div, tmp, noise, cond, next;
  std::vector<float> ufine;            // fine velocity [fine cell][3]
  std::vector<float> fa, fb, lo, hi;   // MacCormack
  std::vector<float> rr, aa;           // the lock, per coarse cell: heat's and soot's factors and new material
  CurlField3 curl, swirl;
  FlickerLattice flicker;              // the flicker at the fine cells' centres, for the breakup of new material
  struct Axis {                        // where a fine cell's centre falls on a coarser grid, per axis
    std::vector<int> i;
    std::vector<float> f;
  };
  std::array<Axis, 3> coarse_axis, swirl_axis;
  Conv3Fn conv = conv3_reference;
  DetailKernel* detail = nullptr;      // the runtime's own detail layer (not owned), or the reference's detail_step
};

// A running effect.
struct State {
  float time = 0;         // seconds since the effect began
  float since_start = 0;  // seconds since the start point
  std::vector<float> coarse;    // cells * channels
  std::vector<float> pressure;  // cells
  std::vector<float> flow;      // cells * 3: the projected velocity of the last step (moves the fine fields)
  std::vector<float> fine_t, fine_d;
};
State make_state(const Model& m);

// The stepper's noise inputs on the coarse grid for the frame that starts at `seconds`: [cell][kNoise].
void coarse_noise(const Model& m, std::uint64_t seed, float seconds, Workspace& ws, std::span<float> out, Pool* pool = nullptr);
// One coarse step: in (cells * C), noise, cond -> out (cells * C); pressure warm start in, solution out; flow.
void coarse_step(const Model& m, std::span<const float> in, std::span<const float> noise, std::span<const float> cond, std::span<float> pressure,
                 std::span<float> out, std::span<float> flow, Workspace& ws, Pool* pool = nullptr);
// The detail layer: carry the fine fields by the flow (plus the swirl) and lock them to the coarse heat and soot.
void detail_step(const Model& m, State& s, std::uint64_t seed, std::span<const float> controls, Workspace& ws, Pool* pool = nullptr);

// Start from start point `index` (or from a given coarse state: kPhys channels per cell) and grow the fine fields.
void start(const Model& m, int index, std::span<const float> controls, std::uint64_t seed, State& s, Workspace& ws, Pool* pool = nullptr);
void start_from(const Model& m, std::span<const float> coarse_phys, float time, std::span<const float> controls, std::uint64_t seed, State& s,
                Workspace& ws, Pool* pool = nullptr, bool grow = true);
// One frame: the stepper, then the detail layer (Workspace::detail's, if set, else detail_step).
void step(const Model& m, State& s, std::span<const float> controls, std::uint64_t seed, Workspace& ws, Pool* pool = nullptr);

// Block averages of fine fields (fx * fy * fz) to the coarse grid, and the fine fields' trilinear up-sampling.
void block_average(const Hyper& h, std::span<const float> fine, std::span<float> coarse, int channel, int channels);

}  // namespace nfx::rollout3d
