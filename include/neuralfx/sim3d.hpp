// Study N's 3D simulator (docs/STUDY_3D.md): the 3D analogue of the 2D stable-fluids solver of sim.hpp, for a small
// demo of a learned 3D effect. It is our own solver, tuned by eye, NOT a validated fire code: the accurate 3D teachers
// are the PC plan's job (TODO.md). What was checked is listed in docs/STUDY_3D.md §2 (the plume rises; it puffs at a
// steady rate).
//
// Solver: collocated grid with a one-cell border; semi-Lagrangian advection (trilinear) for the velocity and a clamped
// MacCormack correction for the scalars; buoyancy from temperature against soot weight; vorticity confinement; a seeded
// 3D curl-noise force (noise4.hpp, shared with the learned model); combustion expansion (a divergence source where the
// gas is hot); pressure projection by red-black Gauss-Seidel with over-relaxation, warm-started. Boundaries: a closed
// floor (no flow through it, slip along it; Neumann pressure), open top and sides (p = 0, outflow).
//
// Axes: x across, y up, z in depth. A field of nx * ny * nz values is indexed (y * nz + z) * nx + x. Velocities are in
// solver cells per second. Deterministic: the same fields on any number of threads (Params::threads).
#pragma once

#include <neuralfx/noise4.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace nfx::sim3d {

// fire; smoke: study N's demo preset (warm gas that cools and soot that fades, every velocity damped; kept as it was
// so study N's smoke runs can be repeated); smoke_r: study R's smoke, the same source and controls without those sinks,
// checked against plume theory (docs/STUDY_R.md §1).
enum class Effect { fire, smoke, smoke_r };
std::string_view effect_name(Effect e);
bool parse_effect(std::string_view text, Effect& out);

inline constexpr int kControls = 3;
inline constexpr std::array<std::string_view, kControls> kControlNames{"intensity", "wind", "turbulence"};

struct Params {
  Effect effect = Effect::fire;
  float intensity = 0.5f;   // burner size and strength
  float wind = 0.5f;        // wind along x: 0 = full towards -x, 0.5 = none, 1 = full towards +x
  float turbulence = 0.5f;  // vorticity confinement and curl-noise strength
  std::uint64_t seed = 1;   // the source flicker and the curl noise
  int width = 64;           // cells across (x) and in depth (z)
  int height = 96;          // cells up (y)
  float fps = 30.f;
  int substeps = 2;         // solver steps per frame
  int pressure_iters = 30;  // red-black Gauss-Seidel sweeps per projection (warm-started; one projection per step)
  int threads = 1;          // the same fields on any number
  // The box's width in world units (x and z; lengths and speeds are set for 128 units across, so a wider span at the
  // same cells per unit is a larger box around the same source) and the burner's position across x as a fraction of
  // the span. Study R's plume measurement (docs/STUDY_R.md) uses larger boxes; the defaults are every earlier study's.
  float span = 128.f;
  float source_x = 0.5f;
  // How heat and soot are carried: -1 the effect's own (fire and smoke: semi-Lagrangian with the clamped MacCormack
  // correction; smoke_r: in flux form), 0 MacCormack, 1 flux form (conservative: what leaves a cell enters its
  // neighbour, so the solver neither makes nor loses heat or soot; a second-order upwind scheme with the monotonised
  // central limiter, one sweep per axis in alternating order, substeps where a sweep's Courant number would pass 0.95).
  int scalars = -1;
  std::array<float, 3> controls() const { return {intensity, wind, turbulence}; }
};

// The noise constants of an effect (the flicker's rate differs, as in 2D).
Forcing3 forcing(Effect e);

// The solver's fields without their border: nx * ny * nz values each, indexed (y * nz + z) * nx + x.
struct State {
  int nx = 0, ny = 0, nz = 0;
  int frame = 0;
  float time = 0.f;
  std::vector<float> u, v, w, temp, soot, pressure;
};

class Fluid {
 public:
  explicit Fluid(const Params& p);
  ~Fluid();
  Fluid(const Fluid&) = delete;
  Fluid& operator=(const Fluid&) = delete;

  void step_frame();  // one output frame: `substeps` solver steps
  // Change the controls from the next frame on (study N2's training runs whose controls change during the run).
  void set_controls(float intensity, float wind, float turbulence);
  int frame_index() const { return frame_; }
  float time() const { return time_; }
  const Params& params() const { return p_; }
  int nx() const { return nx_; }
  int ny() const { return ny_; }
  int nz() const { return nz_; }

  State state() const;
  void set_state(const State& s);  // sizes must match
  // Temperature and soot (nx * ny * nz each), and the velocity [cell][3] in cells per second.
  void fields(std::span<float> temp, std::span<float> soot) const;
  void velocity(std::span<float> uvw) const;
  // The vertical force per unit of heat and of soot (cells per second squared): the buoyancy is
  // heat_coeff * temperature - soot_coeff * soot (study R's plume measurement: the buoyancy flux).
  std::array<float, 2> buoyancy_coefficients() const;

 private:
  struct Grid {
    int nx = 0, ny = 0, nz = 0;
    std::size_t stride_z = 0, stride_y = 0;
    std::vector<float> v;
    void init(int x, int y, int z);
    float& at(int x, int y, int z) { return v[static_cast<std::size_t>(y) * stride_y + static_cast<std::size_t>(z) * stride_z + static_cast<std::size_t>(x)]; }
    float at(int x, int y, int z) const { return v[static_cast<std::size_t>(y) * stride_y + static_cast<std::size_t>(z) * stride_z + static_cast<std::size_t>(x)]; }
    float sample(float x, float y, float z) const;  // padded coordinates (cell centres at 1..n), clamped
  };
  void step(float dt);
  void add_sources(float dt);
  void add_forces(float dt);
  void project();
  void advect_velocity(float dt);
  void advect_scalar(Grid& q, float dt);
  void advect_scalar_flux(Grid& q, float dt);  // the flux form (Params::scalars)
  void velocity_border();
  template <class F>
  void slabs(F&& fn);  // fn(y) for y = 1..ny on the pool

  Params p_;
  int nx_, ny_, nz_;
  float s_;  // cells per unit of a 128-unit width
  int frame_ = 0;
  float time_ = 0.f;
  int steps_ = 0;  // the solver step under way, frame * substeps + substep (the flux form's sweep order alternates with it)
  Forcing3 forcing_;
  CurlField3 curl_;
  Grid u_, v_, w_, temp_, soot_, pressure_, div_, ta_, tb_, tc_, wx_, wy_, wz_, wn_;
  std::unique_ptr<Pool> pool_;
};

}  // namespace nfx::sim3d
