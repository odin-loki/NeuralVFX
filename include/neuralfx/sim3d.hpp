// Study N's 3D simulator (docs/STUDY_3D.md): the 3D analogue of the 2D stable-fluids solver of sim.hpp, for a small
// demo of a learned 3D effect. It is our own solver, tuned by eye, NOT a validated fire code: the accurate 3D teachers
// are the PC plan's job (TODO.md). What was checked is listed in docs/STUDY_3D.md §2 (the plume rises; it puffs at a
// steady rate).
//
// Study R (docs/STUDY_R.md) added a validated smoke, Effect::smoke_r: no cooling, fading or damping, its heat and soot
// carried in flux form (conservative), checked against Morton, Taylor and Turner's plume theory; and larger boxes
// (Params::span) for that check. The fire and study N's smoke are unchanged.
//
// Study R2 (docs/STUDY_R2.md) found where smoke_r's plume loses a fifth of the momentum its buoyancy gives (a budget of
// the vertical momentum per operator, MomentumAudit) and added Effect::smoke_r2: smoke_r with its velocity carried in
// flux form too (Params::velocity 2), which conserves momentum as the scalars' flux form conserves heat and soot. The
// fire, study N's smoke and smoke_r are unchanged to the bit.
//
// Solver: collocated grid with a one-cell border; semi-Lagrangian advection (trilinear) for the velocity and a clamped
// MacCormack correction for the scalars (or flux form, Params::scalars); buoyancy from temperature against soot weight; vorticity confinement; a seeded
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
// checked against plume theory (docs/STUDY_R.md §1); smoke_r2: study R2's, smoke_r with its velocity in flux form
// (docs/STUDY_R2.md §1).
enum class Effect { fire, smoke, smoke_r, smoke_r2 };
// The smoke presets of studies R and R2 (no sinks; heat and soot in flux form by default).
inline bool plume_smoke(Effect e) { return e == Effect::smoke_r || e == Effect::smoke_r2; }
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
  // How the velocity is carried: -1 the effect's own (smoke_r2: flux form; semi-Lagrangian for every other effect), 0
  // semi-Lagrangian, 1 with the clamped MacCormack correction (less numerical dissipation; study R's measurement of the
  // momentum budget), 2 flux form (study R2: each component carried as the scalars' flux form carries heat, by the
  // velocity after the forces, frozen for the step; what leaves a cell enters its neighbour, so momentum is conserved).
  int velocity = -1;
  // How vorticity confinement pushes: -1 the effect's own (smoke_r2: momentum-neutral; every other effect: as study N
  // wrote it), 0 as study N wrote it (epsilon N x omega), 1 momentum-neutral (study R2: the same force less its
  // horizontal Gaussian average of width confinement_sigma world units, each cell's force spread over its plane with
  // weights that sum to one, so that it sharpens the vortices it sees and adds no net momentum to any plane: in a plume
  // the plain force pushes down on the whole, docs/STUDY_R2.md §1.2), 2 off (for the diagnosis).
  int confinement = -1;
  float confinement_sigma = 8.f;
  std::array<float, 3> controls() const { return {intensity, wind, turbulence}; }
};

// The noise constants of an effect (the flicker's rate differs, as in 2D).
Forcing3 forcing(Effect e);

// Study R2 (docs/STUDY_R2.md §1): the budget of the vertical momentum in a slab of rows, per operator. Each field is the
// sum over the steps of what that operator added to the sum of v over the slab's cells (cells / s, summed over cells:
// divide by `seconds` for a rate in cells^4 / s^2, the units of a plane's momentum flux per cell of height). The
// buoyancy, the confinement and the curl noise are the forces' own terms (dt times each force, before any damping);
// `damping` is what the velocity damping took. The solver fills it when Fluid::set_audit has given it one; nothing the
// solver computes changes. The projection's term is the pressure's: it telescopes to the planes at the slab's ends.
struct MomentumAudit {
  int y_lo = 0, y_hi = -1;  // the slab: cell rows y_lo..y_hi (0-based, as State's), inclusive
  double sources = 0, buoyancy = 0, confinement = 0, curl = 0, damping = 0, advection = 0, projection = 0;
  double seconds = 0;  // the time the sums cover
  // the vertical momentum carried in through the slab's bottom face and out through its top by the advection, as the
  // scheme computes it where it is in flux form (velocity 2: the faces' fluxes; otherwise left at 0), and through the
  // box's four sides between the slab's rows (same units as the sums above)
  double flux_bottom = 0, flux_top = 0, flux_sides = 0;
};

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
  // Study R2: fill `a` (not owned; nullptr stops) with the vertical momentum budget of its slab from the next step on.
  void set_audit(MomentumAudit* a) { audit_ = a; }
  // The projection's pressure without its border (nx * ny * nz, indexed as the fields), in cells^2 / s^2: the solver's
  // pressure (velocity times cells) over the step's dt, so its gradient is an acceleration.
  void pressure(std::span<float> out) const;

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
  void advect_velocity_maccormack(float dt);  // Params::velocity 1
  void advect_velocity_flux(float dt);        // Params::velocity 2
  // The flux form's sweeps of q carried by the velocity cu, cv, cw (one per axis, alternating order; substeps by the
  // Courant number). Positive: q kept at or above 0 (the scalars); a velocity component instead refreshes its border
  // (outflow, the floor reflecting v when reflect_floor) after every substep. faces: where the audit's face fluxes go.
  template <bool Positive>
  void flux_sweeps(Grid& q, const Grid& cu, const Grid& cv, const Grid& cw, float dt, bool reflect_floor, bool audit_faces);
  void component_border(Grid& g, bool reflect_floor);
  double slab_sum(const Grid& g) const;  // the audit's slab: the sum of g over its cells (fixed order)
  int velocity_mode() const;
  int confinement_mode() const;
  void neutral_confinement(float conf);  // Params::confinement 1: the force into fcx_, fcy_, fcz_
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
  Grid fcx_, fcy_, fcz_, fct_;     // the momentum-neutral confinement's force and its scratch (Params::confinement 1)
  std::vector<float> gx_, gz_;     // its spreading weights along x and z: [cell][tap], normalised per source cell
  int gr_ = 0;                     // their radius in cells
  std::unique_ptr<Pool> pool_;
  MomentumAudit* audit_ = nullptr;
  std::vector<std::array<double, 4>> audit_rows_;  // per padded row: the forces' terms (buoyancy, confinement, curl, damping)
};

}  // namespace nfx::sim3d
