// Procedural ground truth: a 2D stable-fluids solver with temperature and soot density, rendered to premultiplied
// RGBA clips (docs/PLAN.md §5.3). Its controls are the ones the neural models learn, so every clip is labelled and
// test clips can use seeds and control settings that training never saw.
//
// Solver: collocated grid with a one-cell border; semi-Lagrangian advection with a clamped MacCormack correction
// for the scalars; buoyancy from temperature against soot weight; vorticity confinement; a seeded curl-noise force;
// pressure projection by red-black Gauss-Seidel with over-relaxation; open (p = 0) boundaries.
#pragma once

#include <neuralfx/clip.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace nfx::sim {

enum class Effect { fire, smoke, explosion };
inline constexpr std::array kEffects{Effect::fire, Effect::smoke, Effect::explosion};

std::string_view effect_name(Effect e);
bool parse_effect(std::string_view text, Effect& out);
bool effect_loops(Effect e);  // fire and smoke loop; an explosion plays once

// The learned controls, all normalised to [0, 1].
inline constexpr int kControls = 3;
inline constexpr std::array<std::string_view, kControls> kControlNames{"intensity", "wind", "turbulence"};

struct Params {
  Effect effect = Effect::fire;
  float intensity = 0.5f;   // source size and strength
  float wind = 0.5f;        // lateral wind: 0 = full left, 0.5 = none, 1 = full right
  float turbulence = 0.5f;  // vorticity confinement and curl-noise strength
  std::uint64_t seed = 1;   // source noise and turbulence field
  int size = 128;           // output pixels (square)
  int frames = 64;
  float fps = 30.f;
  int sim_res = 0;          // solver cells per side; 0 = size
  int substeps = 2;         // solver steps per output frame
  int warmup = -1;          // frames simulated before recording (looping effects); -1 = effect default
  int loop_blend = 16;      // crossfade length that makes looping clips seamless
  int pressure_iters = 30;
};

// A scalar field on an (n + 2) x (n + 2) grid: cells 1..n inside, a one-cell border. f[x, y], y up.
class Field {
 public:
  explicit Field(int n = 0) : n_(n), v_(static_cast<std::size_t>(n + 2) * (n + 2), 0.f) {}
  float& operator[](int x, int y) { return v_[static_cast<std::size_t>(y) * (n_ + 2) + x]; }
  float operator[](int x, int y) const { return v_[static_cast<std::size_t>(y) * (n_ + 2) + x]; }
  // Bilinear sample at a continuous cell position, clamped to the grid.
  float sample(float x, float y) const;
  std::span<float> values() { return v_; }
  int n() const { return n_; }

 private:
  int n_;
  std::vector<float> v_;
};

// The solver's fields without their border: n * n values each, row-major from the bottom row (y up). Velocities are
// in solver cells per second. Start points for learned dynamics are made from these (src/core/rollout.cpp).
struct State {
  int n = 0;
  int frame = 0;
  float time = 0.f;
  std::vector<float> u, v, temp, soot, pressure;
};

// The run's stochastic forcing, at a position in cells of a 128-cell frame (X, Y in [0, 128], Y up) and a time in
// seconds: the curl-noise stream function behind add_forces and the flicker of the source in add_sources. Learned
// dynamics take these as inputs, so a new seed gives new detail with the right statistics.
float curl_potential(const Params& p, float X, float Y, float time);
float source_flicker(const Params& p, float X, float Y, float time);

// The simulator state, for clip generation and for live use (viewer, cost measurement).
class Fluid {
 public:
  explicit Fluid(const Params& p);
  void step_frame();                                // one output frame: `substeps` solver steps
  void render(std::span<std::uint8_t> rgba) const;  // p.size * p.size * 4, premultiplied
  int frame_index() const { return frame_; }
  const Params& params() const { return p_; }
  State state() const;
  void set_state(const State& s);  // s.n must equal the solver resolution

  // Couplings from outside (docs/COMPOSE.md §9), between frames. Fields are n * n values without the border, rows from
  // the bottom (as State).
  //   push: adds a velocity field (solver cells per second). push(du, dv), step_frame(), push(-du, -dv) is what
  //         compose::push does to a learned effect: the flow moves material for one frame and does not build up; a
  //         push that is not taken out again is a lasting force.
  //   add_material: adds heat and soot, clamped at zero (negative amounts remove material).
  void push(std::span<const float> du, std::span<const float> dv);
  void add_material(std::span<const float> dtemp, std::span<const float> dsoot);

 private:
  void step(float dt);
  void add_sources(float dt);
  void add_forces(float dt);
  void project();
  void advect_velocity(float dt);
  void advect_scalar(Field& q, float dt);
  void velocity_border();

  Params p_;
  int n_;
  int frame_ = 0;
  float time_ = 0.f;
  Field u_, v_, temp_, soot_, tmp_a_, tmp_b_, tmp_c_, pressure_, div_, curl_;
};

// A clip of p.frames frames. Looping effects are warmed up, then made seamless by crossfading `loop_blend` extra
// frames into the start. Deterministic for equal parameters.
Clip simulate(const Params& p);

// Copy the controls of p into a clip's metadata.
void stamp(Clip& clip, const Params& p);

// Parameters from a clip's metadata (the inverse of stamp), for re-simulating a clip.
Params params_of(const Clip& clip);

}  // namespace nfx::sim
