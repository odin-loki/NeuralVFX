// Composed effects: several rollout effects, and simple modules that are not learned (particles, light, distortion),
// running on one another through their fields (docs/COMPOSE.md). A prototype over the runtime's internals
// (rt_common.hpp); not part of the C API yet.
//
// The pieces:
//   - Module: one rollout effect placed in the world (a square tile), stepped with its controls and drawn either by its
//     learned renderer or by the field shader (light from heat, soot that absorbs and is lit by the scene).
//   - Couplings between modules, applied between steps (they read and write the runners' states):
//       blend_band  two tiles of one domain share a band of cells: a domain larger than any trained one;
//       hand_over   a module continues from another's state (an explosion's cloud becomes the smoke model's);
//       push        a module's flow is pushed by the others' (a blast bends a fire);
//       transfer    material moves from one module into others (a fire's smoke joins the sky above it);
//       suppress    material is removed from a region (a source that should not burn here).
//   - FieldBus: every module's velocity, heat and soot resampled into world space, by group, so a module can read the
//     others' fields (push, triggers, light, distortion, particles).
//   - Light, particles, distortion, bloom and tone mapping for the final picture (Frame).
// Buffers are allocated at construction; stepping, coupling and drawing allocate nothing.
#pragma once

#include "rt_common.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace nfx::compose {

inline float fl(int v) { return static_cast<float>(v); }
inline std::size_t zs(int v) { return static_cast<std::size_t>(v); }

// --- threads -----------------------------------------------------------------------------------------------------------

// A fixed set of worker threads. run(n, f) calls f(0) .. f(n - 1) spread over the workers and the calling thread, and
// returns when all are done. One thread: everything runs on the caller. Allocates nothing per run (f is referred to,
// not copied).
//
// Several threads may run jobs at once (a frame's state on one thread while the previous frame's picture is drawn on
// another): the workers take tasks from every job, and a caller whose own tasks are all taken runs other jobs' tasks,
// one at a time, while it waits for its own. Tasks of one job must not depend on each other; which thread runs a task
// changes nothing.
class Pool {
 public:
  explicit Pool(int threads);  // threads - 1 workers
  ~Pool();
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;
  int threads() const { return static_cast<int>(workers_.size()) + 1; }
  template <class F>
  void run(int n, F&& f) {
    run_impl(n, [](void* ctx, int i) { (*static_cast<std::remove_reference_t<F>*>(ctx))(i); }, const_cast<void*>(static_cast<const void*>(&f)));
  }
  // Runs other jobs' tasks until done() holds: for a thread that waits for another thread's work. done() is polled
  // between tasks and after wake(); whoever makes it true calls wake().
  template <class P>
  void help_until(P&& done) {
    help_impl([](void* ctx) { return static_cast<bool>((*static_cast<std::remove_reference_t<P>*>(ctx))()); }, const_cast<void*>(static_cast<const void*>(&done)));
  }
  void wake();
  // Post jobs even without workers, so that other threads waiting in help_until() run their tasks (a pool of one thread
  // shared with a PictureThread: two threads in all).
  void share_with_callers() { shared_ = true; }

 private:
  using Fn = void (*)(void*, int);
  struct alignas(64) Job {
    Fn fn = nullptr;
    void* ctx = nullptr;
    std::atomic<int> n{0}, next{0}, left{0};
    std::atomic<int> users{0};  // threads inside the job: its slot is reused when none is
    std::atomic<bool> live{false};
  };
  static constexpr int kJobs = 8;  // jobs at once (more callers run their tasks alone)
  void run_impl(int n, Fn fn, void* ctx);
  void help_impl(bool (*done)(void*), void* ctx);
  bool take(int skip, bool one);  // tasks of a live job other than `skip`: one, or all that can be had; false if none
  void work();
  std::vector<std::thread> workers_;
  std::array<Job, kJobs> jobs_;
  std::mutex slots_mu_;
  std::array<bool, kJobs> slot_taken_{};
  std::atomic<std::uint32_t> posted_{0};  // changes whenever there may be new work (or a reason to look again)
  std::atomic<bool> stop_{false};
  bool shared_ = false;
};

// --- images ------------------------------------------------------------------------------------------------------------

// Linear light, premultiplied RGBA floats, rows top to bottom.
struct Image4 {
  int w = 0, h = 0;
  std::vector<float> px;
  void allocate(int width, int height) {
    w = width;
    h = height;
    px.assign(zs(w) * zs(h) * 4, 0.f);
  }
  float* row(int y) { return px.data() + zs(y) * zs(w) * 4; }
  const float* row(int y) const { return px.data() + zs(y) * zs(w) * 4; }
};

// --- modules -----------------------------------------------------------------------------------------------------------

enum class Isa { base, avx2, avx512 };
// The ISA the runtime gives a rollout runner at a precision (nvfx_set_isa's, else AVX2; at int8 the AVX-512 build where
// the CPU has VNNI: study K, docs/REPORT.md §13).
Isa best_isa(rt::Precision p = rt::kRolloutPrecision);
const char* isa_name(Isa isa);
std::unique_ptr<rt::RolloutRunner> make_runner(const rt::RolloutEffect& e, int size, Isa isa, rt::Precision p = rt::kRolloutPrecision);

// The working memory of the modules' steps (rt::RolloutScratch: the detail layer's rings and row records, the coarse
// step's activations), shared: a module that shares it takes a scratch for each step (or start) and gives it back,
// so a scene holds one per thread that steps at once instead of one per module (docs/COMPOSE.md §7.3). Taking and
// giving back are lock-free and allocate nothing; which scratch a step takes changes nothing in its result.
class StepScratch {
 public:
  explicit StepScratch(int slots);  // slots: the threads that may step modules at the same time
  StepScratch(const StepScratch&) = delete;
  StepScratch& operator=(const StepScratch&) = delete;
  void fit(const rt::RolloutRunner& r);  // room for r in every slot (a set-up call: allocates)
  rt::RolloutScratch& take();            // a free slot (if every one is taken, waits for one)
  void give(rt::RolloutScratch& s);
  int slots() const { return static_cast<int>(s_.size()); }
  std::size_t bytes() const;

 private:
  std::vector<rt::RolloutScratch> s_;
  std::unique_ptr<std::atomic<bool>[]> busy_;
};

// Where a module's tile is in the world: its top-left corner and world pixels per tile pixel. World y points down.
struct Placement {
  float x = 0, y = 0, scale = 1;
};

enum class Look { learned, shader };

// The field shader: emission from heat (a blackbody-like ramp), soot that absorbs, shadowed towards the sky and lit by
// the scene's light (Light). Units are the simulation's heat and soot.
struct ShaderSpec {
  float heat_scale = 1.f;     // heat that maps to the top of the colour ramp (white-yellow)
  float emission = 4.f;       // emitted light at heat_scale (linear units; bloom and tone mapping follow)
  float emission_power = 2.f; // emission grows as (heat / heat_scale) ^ power
  float soot_density = 2.5f;  // alpha = 1 - exp(-density * soot)
  float soot_albedo = 0.55f;
  float sky = 0.06f;          // ambient light from the night sky on lit soot
  float shadow = 0.35f;       // self-shadow: soot towards the sky dims the ambient by exp(-shadow * sum)
  float scene_light = 1.f;    // how much of the scene's light (fires, flash) lights the soot
  float relief = 6.f;         // soot read as a height field for lighting (0: flat)
  std::array<float, 3> tint{1.f, 1.f, 1.f};  // soot colour
  // The emitted colours turned around the grey axis by this many degrees (a recolour, not new dynamics: 120 turns the
  // fire ramp's reds green, 240 blue to violet). 0 draws them as they are, by exactly the code without it.
  float hue = 0.f;
};

// The 3 x 3 matrix that turns a colour `degrees` around the grey axis (rows: output red, green, blue).
std::array<float, 9> hue_matrix(float degrees);

class Light;

class Module {
 public:
  // `precision`: the runner's (rt::Precision::int8: study K's int8 path, docs/REPORT.md §13).
  Module(std::string name, const rt::RolloutEffect& e, int size, Placement at, Isa isa, rt::Precision precision = rt::kRolloutPrecision);

  const std::string& name() const { return name_; }
  const rt::RolloutEffect& effect() const { return e_; }
  rt::RolloutRunner& runner() { return *r_; }
  const rt::RolloutRunner& runner() const { return *r_; }
  int size() const { return size_; }
  int res() const { return e_.m.h.res; }
  int channels() const { return e_.m.h.channels(); }
  float cell_px() const { return at.scale * fl(size_) / fl(res()); }  // world pixels per coarse cell

  // Velocity added for the next step only (push, force fields): added to the state before the step and taken out
  // after it, so it moves material without building up momentum. Coarse cells, [res * res][2].
  std::span<float> pushed() { return pushed_; }

  void start(int index, std::uint64_t run_seed);
  void start_empty(float seconds, std::uint64_t run_seed);  // nothing in the domain, at an age (receives material)
  // From now on, step (and start) with working memory taken from `shared` (made room for this module) instead of the
  // runner's own, which is freed. A set-up call; `shared` must outlive the module's steps.
  void share_scratch(StepScratch& shared);
  void take_over(const Module& from);                         // the other's physical state and fine fields
  // Shards, so that a looping effect plays for good (docs/COMPOSE.md §4.3): one continuous rollout wanders off within
  // tens of seconds (a flame dies, smoke breaks up), so every `period` frames of its life a second runner starts with
  // a new seed from the start point nearest the module's controls, `fade` frames before the change, and steps beside
  // it while the module's state is blended into it (every coarse channel and the fine fields, the last frame taking
  // the second's whole); then the second runner is the module's. `offset` frames shift the changes (instances apart).
  // A shard whose heat and soot fall below a sixth of the most it had (a flame gone out) is replaced at once.
  // A set-up call (it makes the second runner); a module started empty or taken over plays on without. 0: off.
  void use_shards(int period, int fade, int offset);
  int shards() const { return shard_; }  // shard changes since the module started
  std::size_t shard_bytes() const;       // the second runner's own memory
  void step();
  void shade(const Light* light);  // into image(): learned renderer or field shader, tile space
  const Image4& image() const { return img_; }
  // Per row of image(), [lo, hi): every pixel outside is +0 in every channel (written by shade()). [size][2]
  std::span<const int> drawn_spans() const { return drawn_; }

  // Engine layers (docs/ENGINES.md §11): with use_layers(true) (a set-up call: allocates), shade() also writes, beside
  // image(), emission(): the light each pixel gives (rgb, premultiplied as image() is) and its optical depth (a: 0 where
  // image()'s alpha is 0), and normals(): its normal times its alpha (x right, y up, z towards the viewer; a 0), from the
  // fine soot's slopes as the field shader's relief reads them. albedo() says what the pixel reflects, unlit: `rgb` times
  // image()'s alpha, or (`from_rgb`) image()'s colour itself. The field shader's looks give their soot colour and their
  // emission exactly; a learned look's picture is split by its model: light-giving models (fire, explosion, magic, jet,
  // orb, wildfire, coals, mushroom) give all of it as emission and reflect nothing, the others (smoke, steam, fog, dust,
  // cloud, vortex) reflect all of it (its learned shading kept, as baked light) and give none. image() is the same with
  // layers or without.
  void use_layers(bool on);
  bool layers() const { return layers_; }
  const Image4& emission() const { return emit_; }
  const Image4& normals() const { return norm_; }
  struct Albedo {
    std::array<float, 3> rgb{};
    bool from_rgb = false;
  };
  Albedo albedo() const;

  // Ownership weight of a tile pixel (x right, y up, tile pixels) or coarse cell for tiling groups: 1 inside, shared
  // with the neighbour across a band (the weights of the tiles of a group sum to 1 everywhere).
  float weight_px(float x, float y) const;
  float weight_cell(int cx, int cy) const;
  // weight_px() in two halves, for loops over rows and columns: the factors that depend on x and those that depend on
  // y. weight() multiplies them in weight_px()'s order, so the value is the same to the last bit.
  struct WeightX {
    float band = 1.f, feather0 = 1.f, feather1 = 1.f;
  };
  struct WeightY {
    float band0 = 1.f, band1 = 1.f, feather0 = 1.f, feather1 = 1.f;
  };
  WeightX weight_x(float x) const;
  WeightY weight_y(float y) const;
  static float weight(const WeightX& a, const WeightY& b) { return a.band * b.band0 * b.band1 * a.feather0 * a.feather1 * b.feather0 * b.feather1; }

  bool active = false;
  float opacity = 1.f;
  Look look = Look::learned;
  ShaderSpec spec;
  std::vector<float> controls;
  std::uint64_t seed = 1;
  Placement at;
  int group = -1;                 // tiles of one domain share a group (drawn and published as one)
  std::array<int, 4> band{};      // band cells shared with a neighbour: left, right, bottom, top (0: an outer edge)
  float feather = 0.f;            // outer edges fade over this many tile pixels when drawn (hides a domain's walls)
  // A light of its own colour: its heat lights the scene in `glow` (times heat squared and the light's gain, as the fires'
  // light) instead of in the fire colours (a magic portal's violet light). Its heat is on the bus as any module's.
  bool glows = false;
  std::array<float, 3> glow{};
  double step_ms = 0, shade_ms = 0;
  int frames = 0;                 // frames stepped since start

 private:
  std::string name_;
  const rt::RolloutEffect& e_;
  int size_;
  Isa isa_;
  rt::Precision prec_;
  std::unique_ptr<rt::RolloutRunner> r_;
  std::unique_ptr<rt::RolloutRunner> next_;  // shards: the next one, rolled ahead
  int shard_period_ = 0, shard_fade_ = 0, shard_offset_ = 0, next_frames_ = -1, shard_ = 0, start_index_ = 0, shard_start_ = 0;
  float shard_peak_ = 0.f;  // the most heat and soot the shard has had (a shard far below it is replaced early)
  bool shardable_ = false;
  std::uint64_t next_seed_ = 0;
  StepScratch* shared_ = nullptr;
  Image4 img_;
  std::vector<std::uint8_t> rgba8_;
  std::vector<float> shadow_;  // coarse: soot summed towards the sky
  // Field shader scratch. Light and shadow are bilinear in their grids, so each grid row is resampled along x once
  // (the first half of a bilinear sample) and every pixel row only blends two such rows (the second half).
  std::vector<float> shadow_x_;      // [res][size]: shadow_ resampled at the pixels' x
  std::vector<float> light_x_;       // [2][3][size]: two rows of the light grid resampled at the pixels' x, a plane per colour
  std::vector<int> sx_, lx_;         // [2][size]: the two grid columns each pixel's x falls between (shadow, light)
  std::vector<float> sfx_, lfx_;     // [size]: weight of the second column
  std::vector<float> row_;           // [2][size]: per row of pixels: soot slope along x, emission ramp
  std::vector<int> drawn_;           // [size][2]: per row of the image, the pixels that may not be zero (the rest are)
  std::vector<float> pushed_;
  bool has_push_ = false;
  bool layers_ = false;
  Image4 emit_, norm_;  // the engine layers (use_layers)
  void shade_layers();
  friend void push(Module&, const struct FieldBus&, float);
  friend void apply(Module&, const struct ForceField&, float);
};

// Band weight of the lower (or left) tile at band cell j of a band of `cells` (j from the band's start): 1 for most of
// the band, then a ramp to 0 at its far end. The upper (or right) tile takes 1 minus this.
inline float band_weight(float j, int cells) { return std::clamp((fl(cells) - 0.5f - j) / 3.5f, 0.f, 1.f); }

// --- couplings ---------------------------------------------------------------------------------------------------------

// Tiles a (below or left) and b (above or right) of one domain: same size, scale and model resolution, overlapping by
// `cells` coarse cells. Both take the weighted mean in the band (coarse physical channels and fine fields).
enum class Side { right, top };
void blend_band(Module& a, Module& b, Side b_is, int cells);
void hand_over(const Module& from, Module& to);
// Remove material (heat and soot, coarse and fine) in coarse cells [x0, x1) x [y0, y1) (y up).
void suppress(Module& m, int x0, int y0, int x1, int y1);

struct FieldBus {
 public:
  // A world rectangle at `cell` world pixels per bus cell, with room for `groups` groups.
  FieldBus(float x0, float y0, int nx, int ny, float cell, int groups);
  void clear();
  // Add a module's fields (velocity in world pixels per frame, y down; heat; soot), weighted by its ownership.
  void publish(const Module& m);
  struct Sample {
    float u = 0, v = 0, heat = 0, soot = 0;
  };
  Sample at(float x, float y) const;                     // all groups
  Sample others(float x, float y, int group) const;      // all groups but one
  int nx() const { return nx_; }
  int ny() const { return ny_; }
  float cell() const { return cell_; }
  float x0() const { return x0_; }
  float y0() const { return y0_; }
  std::span<const float> heat() const { return heat_; }  // all groups, [ny][nx]
  std::span<const float> soot() const { return soot_; }
  // The heat of modules that glow in their own colour (Module::glows), [ny][nx][4]: their heat, then their heat times
  // their colour. glowing(): some module that glows published since the last clear (otherwise glow() is all zero).
  std::span<const float> glow() const { return glow_; }
  bool glowing() const { return glowing_; }

 private:
  Sample sample(const float* f, float x, float y) const;
  float x0_, y0_, cell_;
  int nx_, ny_, groups_;
  std::vector<float> all_;    // [ny][nx][4]: u, v, heat, soot
  std::vector<float> layer_;  // [group][ny][nx][4]
  std::vector<float> heat_, soot_;
  struct Rect {  // bus cells [i0, i1] x [j0, j1] written since the last clear (empty: i0 > i1)
    int i0 = 0, i1 = -1, j0 = 0, j1 = -1;
  };
  std::vector<Rect> dirty_;  // [group], then all groups
  std::vector<float> glow_;  // [ny][nx][4]
  bool glowing_ = false;
  struct Column {            // publish(): what a column of bus cells takes from the module's tile
    Module::WeightX w;       // its factors of the ownership weight
    int x0 = 0, x1 = 0;      // the coarse columns it falls between
    float fx = 0.f;          // and the weight of the second
    bool inside = false;     // within the tile
  };
  std::vector<Column> cols_;
};

// Push: the others' flow (world pixels per frame from the bus), times `gain`, moves the module's material for its next
// step (added to its velocity before the step, taken out after it).
void push(Module& m, const FieldBus& bus, float gain);

// Field effects: forces placed in the world by the script, acting on any module they cover.
//   ceiling  damps vertical motion above world y `y` (over `soft` pixels): a rising cloud stalls and spreads, as a
//            cloud does at the height where it stops being buoyant;
//   vortex   swirls around (x, y) with `strength` world pixels per frame at radius `radius` (positive: clockwise on
//            screen), for one step at a time like a push;
//   wind     a uniform flow (`u`, `v` world pixels per frame) for one step at a time.
// More field effects (fields.cpp, docs/COMPOSE.md §5):
//   gust     the wind (`u`, `v`) made gusty: its speed varies by `amount` (0.5: +-50%) with smooth noise of `soft`
//            world pixels that travels with the wind and changes `rate` times a second (at `time`, from `seed`), with
//            a crosswind wobble of a third of that;
//   ring     a vortex ring seen from the side: two opposite vortices `radius` either side of (x, y), each with a core
//            of `soft` pixels and `strength` pixels per frame, blowing along (`u`, `v`) (a direction) through the
//            centre; it tears through whatever it crosses;
//   attract  material within about `radius` of (x, y) is drawn `strength` world pixels per frame towards it
//            (negative: pushed away), moved directly (pull()) because the pressure projection would remove a radial
//            flow; `damping` adds a swirl around the point (a push, pixels per frame);
//   heat     a heat source: adds `strength` heat per frame at (x, y), falling off over `radius` (a Gaussian) (a
//            burning object; the fire models turn the heat into flames);
//   cold     an extinguisher: removes a fraction `strength` of the heat per frame at (x, y), falling off over
//            `radius`, and turns a fraction `damping` of what it removes into soot (steam and smoke).
//   lamp     a light at (x, y) of `strength` and `radius` in `colour` (a Lamp for Light::update): acts on no module.
struct ForceField {
  enum class Kind { ceiling, vortex, wind, gust, ring, attract, heat, cold, lamp } kind = Kind::ceiling;
  float x = 0, y = 0, radius = 100.f, strength = 0.f, soft = 80.f, u = 0, v = 0;
  float damping = 0.25f;  // ceiling: fraction of vertical velocity removed per frame at full depth
  float amount = 0.5f, rate = 0.5f, time = 0.f;  // gust
  std::uint64_t seed = 1;                        // gust
  std::array<float, 3> colour{1.f, 0.6f, 0.3f};  // lamp
};
void apply(Module& m, const ForceField& f, float weight = 1.f);
// The flow of the fields that push (gust, ring, attract's swirl) at world (wx, wy), world pixels per frame, y down,
// times `weight` (fields.cpp). Zero for the others.
std::array<float, 2> field_flow(const ForceField& f, float wx, float wy, float weight);
// heat and cold: change the module's heat (and soot) in place, coarse and fine (fields.cpp; apply() calls it).
void apply_heat(Module& m, const ForceField& f, float weight);
// Rain washing smoke out (fields.cpp): within the world rectangle [x0, x1) x [y0, y1), the module's soot (coarse and fine)
// is multiplied by `soot`, its heat by `heat`.
void wash(Module& m, float x0, float y0, float x1, float y1, float soot, float heat);
// attract: move the module's material (heat and soot, coarse and fine) towards (f.x, f.y) by up to f.strength world
// pixels, conserving the amount (each cell and pixel is splatted bilinearly at its new place). `scratch` holds at least
// pull_scratch(m) floats. apply() does only the swirl; this does the pull.
std::size_t pull_scratch(const Module& m);
void pull(Module& m, const ForceField& f, float weight, std::span<float> scratch);
// Move `fraction` of the material in from's coarse rows [row0, res) (y up; all rows: row0 = 0) into the modules of
// `to` that cover it (by their ownership weights; fine fields follow). Material that lands outside them stays. What
// arrives is scaled by the gains (one model's thin soot can be another's smoke; 1 conserves the amount).
void transfer(Module& from, std::span<Module* const> to, float fraction, int row0 = 0, float heat_gain = 1.f, float soot_gain = 1.f);

// --- trails ------------------------------------------------------------------------------------------------------------

// A trail (docs/COMPOSE.md §5.3): a source that moves through the world (a rocket, a vehicle, a carried torch) leaves
// material in a learned effect's state along its path, and that material stays where it was left. The effect plays in
// tiles that lie on a grid fixed in the world: tile (i, j) has its top-left corner at (x0 + i s, y0 + j s), s the stride
// (the tile's width less the band of cells it shares with its neighbours). Each frame the source's footprint along the
// segment from its last place to its new one switches on the tiles it touches (a tile is started with nothing in it, at
// an age; the cells it shares with tiles already playing are copied from them), and deposits heat and soot there (as
// `heat` fields and `transfer` do, coarse and fine fields alike) with a velocity of its own (a kick along or against the
// source's motion, as `push` gives material a flow). Tiles that share a band are blended every frame as a domain's are
// (blend_band), and drawn and published as one group by ownership weights. A tile the source has not fed for `hold`
// seconds fades (its material falls by e every `fade` seconds) and is switched off once it has faded; when every tile is
// in use and another is needed, the one fed longest ago is taken. Nothing allocates after construction.
struct TrailSpec {
  float width = 256.f;          // world pixels a tile covers
  int band = 8;                 // cells two neighbouring tiles share
  float x0 = 0.f, y0 = 0.f;     // the grid's origin: tile (0, 0)'s top-left corner
  int row_lo = -(1 << 20), row_hi = 1 << 20;  // the rows of tiles it may use (j, downwards)
  float age = 3.f;              // the age a tile starts empty at (seconds since its effect began)
  float hold = 1.f;             // seconds a tile keeps its material after the source last fed it
  float fade = 1.5f;            // then: e-folding time of its material, seconds
  float jump = 512.f;           // a move longer than this in one frame is a jump: nothing is left along it
  bool has_quiet = false;       // cells [quiet[0], quiet[2]) x [quiet[1], quiet[3]) (y up) cleared every frame in tiles
  std::array<int, 4> quiet{};   // with no tile below them (a model's learned source, as `suppress` removes it)
  float feather = 0.f;          // tile pixels: outer edges fade when drawn
  // A tile whose material reaches an outer edge (heat or soot above `spread` in its outer two cells there) gets the
  // neighbour on that side, from the tiles not playing (none is taken from elsewhere for this), fed when it was: smoke
  // that rises or drifts out of the tiles the source touched is not cut off. 0: off.
  float spread = 0.05f;
};

// What the source leaves in one frame, spread along its path: `heat` and `soot` at the centre of a Gaussian of `radius`
// world pixels, and a velocity (world pixels per frame, y down) the material there takes on: `kick` times the source's
// own velocity plus (u, v).
struct Deposit {
  float heat = 0.f, soot = 0.f, radius = 12.f, kick = 0.f, u = 0.f, v = 0.f;
};

class Trail {
 public:
  // `tiles`: modules of one effect and size (the trail places them, switches them on and off, and sets their bands);
  // they take `group`. At most 64.
  Trail(std::span<Module* const> tiles, const TrailSpec& spec, int group, std::uint64_t seed);
  // Before the frame's step: the source is at (x, y); `on` false: it leaves nothing (and its next place starts a new
  // path). Switches on the tiles its footprint (of `radius`) touches along its path, and off those that have faded.
  void move(float x, float y, bool on, float radius, float t, float fps);
  // After the step: shared bands blended, quiet cells cleared, the deposit left along the path, faded tiles dimmed.
  void couple(const Deposit& d, float t, float fps);
  void clear();                               // every tile off, no path
  float x() const { return x_; }
  float y() const { return y_; }
  float vx() const { return vx_; }            // the source's velocity, smoothed over 0.15 s (world pixels per second)
  float vy() const { return vy_; }
  int active() const;                         // tiles playing
  int recycled() const { return recycled_; }  // tiles taken from where they were (the pool ran out) since construction
  const TrailSpec& spec() const { return spec_; }
  float stride() const { return stride_; }
  std::span<Module* const> tiles() const { return tiles_; }

 private:
  struct Cell {
    int i = 0, j = 0;
    float fed = 0.f;     // the last time the source left something in it
    bool on = false;
  };
  int find(int i, int j) const;                 // the tile at (i, j), or -1
  // Switch a tile on at (i, j), fed at t; with `recycle`, the one fed longest ago if none is free (need: tiles that must
  // not be taken). -1: none.
  int take(int i, int j, float t, std::uint64_t need, bool recycle = true);
  float edge(int k, int dir) const;             // the most heat or soot in tile k's outer two cells on a side
  void bands();                                 // every tile's shared sides from its neighbours
  void off(int k);
  std::vector<Module*> tiles_;
  std::vector<Cell> cells_;
  TrailSpec spec_;
  float stride_ = 0.f;
  std::uint64_t seed_ = 1, starts_ = 0;
  float x_ = 0.f, y_ = 0.f, px_ = 0.f, py_ = 0.f, vx_ = 0.f, vy_ = 0.f;
  bool on_ = false, had_ = false;  // depositing this frame; the last frame's place is a path's start
  int recycled_ = 0;
};

// Heat and soot added at world (x, y) over a Gaussian of `radius` world pixels (coarse cells and fine pixels alike),
// times `amount`; and the velocity there drawn towards (u, v) (world pixels per frame, y down) by up to `pull` (0 to 1,
// times the Gaussian). Only within three radii of the point.
void deposit(Module& m, float x, float y, float radius, float heat, float soot, float u, float v, float pull);

// --- light -------------------------------------------------------------------------------------------------------------

// A light placed in the world that is not on the bus (procedural, not learned): it emits like hot cells of the bus do,
// `strength` (a fire's heat, about 1) at its centre falling off as a Gaussian of `radius` world pixels, in `colour`.
// It lights the ground and the smoke; it is not heat (no haze, nothing for rules or gameplay to read).
struct Lamp {
  float x = 0, y = 0, radius = 40.f, strength = 1.f;
  std::array<float, 3> colour{1.f, 0.6f, 0.3f};
};

// Light in the world from everything hot on the bus: each bus cell's heat emits a blackbody-like colour, spread by a
// pyramid of blurs (a wide, soft falloff), plus a global flash. Sampled by the field shader and the ground.
class Light {
 public:
  explicit Light(const FieldBus& bus);
  // Rows are spread over the pool; the light is the same on any number of threads. Lamps add their emission to the
  // bus's before the blurs (none: the light of the bus alone, by exactly the code without them).
  void update(const FieldBus& bus, float gain, std::array<float, 3> flash, Pool& pool, std::span<const Lamp> lamps = {});
  std::array<float, 3> at(float x, float y) const;  // world position
  // The grid at() samples: field() is [ny][nx][3] cells of `cell` world pixels from (x0, y0); at() adds flash().
  std::span<const float> field() const { return L_; }
  float x0() const { return x0_; }
  float y0() const { return y0_; }
  float cell() const { return cell_; }
  int nx() const { return nx_; }
  int ny() const { return ny_; }
  std::array<float, 3> flash() const { return flash_; }

 private:
  float x0_, y0_, cell_;
  int nx_, ny_;
  std::vector<float> L_;  // [ny][nx][3]
  struct Level {
    int nx = 0, ny = 0;
    std::vector<float> a, b;
    // Sampling this level at the centres of the finest cells (bilinear): for each column and each row of the finest
    // grid, the two cells of this level it falls between and the weight of the second.
    std::vector<int> x0, x1, y0, y1;
    std::vector<float> fx, fy;
    std::vector<float> up;  // [ny][nx_ of the finest level][3]: each row resampled at the finest columns
  };
  std::vector<Level> levels_;
  std::array<float, 3> flash_{};
};

// Colour of hot gas at heat h / heat_scale = t (0: dark red, 1: yellow-white), unit brightness.
std::array<float, 3> heat_colour(float t);

// --- particles ---------------------------------------------------------------------------------------------------------

// rain, snow, ash and leaf: weather (procedural, not learned; docs/COMPOSE.md §4.6).
enum class Kind : std::uint8_t { spark, ember, debris, flake, glow, rain, snow, ash, leaf };
inline bool weather(Kind k) { return k >= Kind::rain; }

// Particles in world pixels and seconds, a fixed capacity (spawns beyond it are dropped). Embers and sparks glow by
// temperature and are drawn additively with motion blur; debris and flakes are dark and drawn over. Glows (procedural
// elements: lightning, arcs, rings, flares; not learned) are soft lines of light of their own colour that do not move:
// pieces of a bolt or of a ring, or a disc where both ends meet, drawn additively (bloom spreads them).
class Particles {
 public:
  explicit Particles(int capacity);
  int alive() const { return n_; }
  int capacity() const { return static_cast<int>(x_.size()); }
  // One particle; returns false when full.
  bool spawn(Kind k, float x, float y, float vx, float vy, float temp, float size, float life);
  // A glow from (x0, y0) to (x1, y1): `radius` world pixels either side of the line, `brightness` times `colour` at its
  // core, for `life` seconds. Pieces of one line join without overlapping: only the first piece draws the round cap at
  // its start (`cap_start`) and only the last the cap at its end (`cap_end`); a disc is a piece of length 0 with both.
  bool spawn_glow(float x0, float y0, float x1, float y1, float radius, std::array<float, 3> colour, float brightness, float life,
                  bool cap_start, bool cap_end);
  // A weather particle (procedural): rain, snow, ash or a leaf (or a petal) at (x, y), drifting at (du, dv) world pixels a
  // second (its fall and the wind), carried by `flow` of the bus's flow, fluttering at `freq` radians a second from
  // `phase`, of `size` (world pixels) and `colour`, drawn with coverage `alpha`. `aux`: snow melts where the bus's heat
  // is above it (0: never); ash glows at that temperature, cooling. It lives `life` seconds or until it reaches the
  // ground. Its random numbers are the caller's: the generator rng is not used.
  bool spawn_weather(Kind k, float x, float y, float du, float dv, float size, float life, float phase, float freq, std::array<float, 3> colour,
                     float alpha, float flow, float aux);
  struct Landing {
    float x = 0, temp = 0;
    Kind kind = Kind::ember;
  };
  // Gravity, drag towards the bus flow (others' flow seen by all particles), ground at ground_y (bounces or settles);
  // embers cool. Landings of hot particles are recorded (cleared each update).
  void update(float dt, const FieldBus* bus, float flow_gain, float ground_y);
  std::span<const Landing> landings() const { return std::span<const Landing>(landed_).first(zs(n_landed_)); }
  // Every live particle's position (world pixels) and velocity (world pixels per second, which f may change), for
  // fields that act on particles.
  template <class F>
  void each(F&& f) {
    for (int i = 0; i < n_; ++i) f(x_[zs(i)], y_[zs(i)], vx_[zs(i)], vy_[zs(i)]);
  }
  // Draw into a screen image (camera offset: world minus screen).
  void draw(Image4& screen, float cam_x, float cam_y) const;
  // The same particles into the engine layers (Frame::use_layers): what draw() adds (sparks, embers, glows) is light,
  // added to the colour and the emission; what it lays over (debris, flakes, weather) covers what is under it in every
  // layer, with its colour lit as draw() lights it, its albedo, its glow as emission, a flat normal and its optical depth.
  // `spans` (one per row) grows to the columns drawn in each row.
  void draw_layers(Image4& colour, Image4& albedo, Image4& emission, Image4& normal, float cam_x, float cam_y,
                   std::span<std::array<int, 2>> spans) const;
  // The live particles into `to`, as far as draw() needs them (`to` holds at least alive() particles).
  void copy_to(Particles& to) const;
  std::uint64_t rng = 0x9E3779B97F4A7C15ULL;
  float uniform();  // [0, 1)

 private:
  void kill(int i);
  // The pixels a weather particle covers on a screen of W x H: f(index of the pixel's first channel, coverage).
  template <class F>
  void weather_pixels(std::size_t q, float cam_x, float cam_y, int W, int H, F&& f) const;
  int n_ = 0;
  // cool_ is a glow's caps (1: start, 2: end); temp_ its brightness; size_ its radius; r_, g_, b_ its colour.
  // Weather: temp_ its flutter's phase, cool_ its frequency, (wu_, wv_) its drift, aux_ its melting heat or glow, wa_ its
  // coverage, wf_ how much it follows the bus's flow.
  std::vector<float> x_, y_, px_, py_, vx_, vy_, temp_, size_, life_, age_, cool_, r_, g_, b_, wu_, wv_, aux_, wa_, wf_;
  std::vector<Kind> kind_;
  std::vector<Landing> landed_;
  int n_landed_ = 0;
};

// --- the frame ---------------------------------------------------------------------------------------------------------

struct Shock {
  float x = 0, y = 0, t0 = 0, speed = 900.f, decay = 0.35f, amp = 6.f, width = 26.f;
  float radius(float t) const;  // world pixels at scene time t
};

// The final picture: background (sky, stars, ground lit by the scene's light), groups of tiles and single modules,
// particles, distortion (shock rings, heat haze), bloom, tone mapping. Screen-sized buffers, allocated once.
//
// Two ways to draw it, the same to the bit:
//   - the stages one at a time, on the live scene: background(), draw(), particles(), distort(), bloom(), finish(),
//     each reading the settings below when it runs;
//   - capture() then render(): capture() copies what the picture needs (the settings below, the light, the scorch
//     marks, the modules' places, opacity and groups, the particles, the shock fronts and the bus's heat), and render()
//     draws it, the background and the modules in one pass over the screen. Between the two, the scene may move on
//     (the next frame's script, step, couplings, bus, light and particles), on another thread: only the modules'
//     images are read late, so a module must not be shaded again before render() has drawn them (`images_read`).
// Some work moves to a later stage on the way (the distortion is copied back into the screen by bloom() or finish(),
// and bloom's last pass, adding it to the screen, is done by finish()), so screen() is only the finished picture
// before bloom's last pass.
class Frame {
 public:
  Frame(int width, int height);
  int width() const { return w_; }
  int height() const { return h_; }

  float cam_x = 0, cam_y = 0;  // world position of the screen's top-left corner (shake included)
  float ground_y = 600.f;      // world y of the ground line
  float exposure = 1.f;
  float haze = 1.f;            // heat haze strength
  float fade = 1.f;            // overall fade (0: black)
  float time = 0;

  void background(const Light& light, std::span<const std::array<float, 4>> scorch, Pool& pool);  // scorch: x, y, radius, glow
  // Draw modules (each group once, by ownership weights; single modules as they are) over the screen.
  void draw(std::span<Module* const> modules, Pool& pool);
  void particles(const Particles& p);
  void distort(std::span<const Shock> shocks, const FieldBus& bus, Pool& pool);
  void bloom(float threshold, float strength, Pool& pool);
  // Tone map, vignette and grain into RGB8 (width * height * 3).
  void finish(std::span<std::uint8_t> rgb, Pool& pool);
  const Image4& screen() const { return screen_; }

  // See above. `modules` in drawing order, as draw() takes them. render() sets *images_read (and wakes the pool's
  // waiting threads) once the modules' images have been drawn.
  void capture(const Light& light, std::span<const std::array<float, 4>> scorch, std::span<Module* const> modules, const Particles& parts,
               std::span<const Shock> shocks, const FieldBus& bus);
  void render(std::span<std::uint8_t> rgb, float bloom_threshold, float bloom_strength, Pool& pool, std::atomic<bool>* images_read = nullptr);
  // Room for captures (capture() allocates only when a count grows beyond what it held before).
  void reserve(int modules, int shocks, int scorch, int particles);
  // The row kernels for the baseline ISA or AVX2 (by default, AVX2 when the runtime's ISA is AVX2 or better): the same
  // frames either way.
  void use_avx2(bool on) { avx2_ = on; }
  // Engine layers (docs/ENGINES.md §11): with use_layers(true) (a set-up call: allocates four screen-sized images),
  // render() also draws the modules (whose layers are on: Module::use_layers) and the particles over a transparent
  // screen into layer(): kColour, the premultiplied colour as the picture lights it (alpha: coverage); kAlbedo, the
  // premultiplied colour it reflects, unlit (the same alpha); kEmission, the light it gives (rgb; a 0); kNormal, its
  // normal times its alpha (x right, y up, z towards the viewer) and its optical depth (a, added through the layers). Linear,
  // the scene's units, before exposure, bloom and tone mapping; no sky, ground, distortion or bloom. The picture is the
  // same with layers or without. Only render() draws them (not the stage-by-stage calls).
  enum Layer { kColour, kAlbedo, kEmission, kNormal, kLayers };
  void use_layers(bool on);
  bool layers() const { return layers_on_; }
  const Image4& layer(Layer l) const { return layers_[static_cast<std::size_t>(l)]; }
  // The exposure and fade of the last render() (what the layers are drawn before).
  float layer_exposure() const { return P_.exposure; }
  float layer_fade() const { return P_.fade; }
  // The stages of the last render(), wall ms.
  enum RenderStage { kCompose, kParticles, kDistort, kBloom, kFinish, kRenderStages };
  const std::array<double, kRenderStages>& render_ms() const { return render_ms_; }

  // One axis of a bilinear lookup: the two cells read and their weights. Taken once per column or row, it leaves a
  // blend per pixel.
  struct Tap {
    int i0 = 0, i1 = 0;
    float w0 = 1.f, w1 = 0.f;
  };

 private:
  struct Params {  // the settings above, as a stage reads them
    float cam_x, cam_y, ground_y, exposure, haze, fade, time;
  };
  Params params() const { return {cam_x, cam_y, ground_y, exposure, haze, fade, time}; }
  struct LightView {  // the light field with a fourth channel (0), a cell to a vector, and where it lies
    const float* L4 = nullptr;
    int nx = 0, ny = 0;
    float x0 = 0, y0 = 0, cell = 1;
    std::array<float, 3> flash{};
  };
  struct HeatView {  // the bus's heat, all groups, and where it lies
    const float* heat = nullptr;
    int nx = 0, ny = 0;
    float x0 = 0, y0 = 0, cell = 1;
  };
  struct Tile {  // a module as draw() reads it
    const Image4* img = nullptr;
    const int* spans = nullptr;  // Module::drawn_spans()
    const Image4* emission = nullptr;  // its layers (Module::use_layers), or none
    const Image4* normals = nullptr;
    Module::Albedo albedo;
    Placement at;
    float opacity = 0, feather = 0;
    int group = -1, size = 0, res = 0;
    std::array<int, 4> band{};
    bool active = false;
  };
  struct Group {  // tiles [first, first + count) of order_, and the screen rows they cover
    int first = 0, count = 0, y0 = 0, y1 = 0;
  };
  // What a stage needs of a screen column, worked out once per frame instead of once per pixel.
  struct Column {
    float wx = 0;                 // world x of the column's centre
    float hill = 0;               // background: world y of the hills' outline
    std::array<float, 4> hill_colour{};
    Tap light;                    // background: the light field's columns
    std::uint32_t star = 0, tex = 0;  // hash keys of the stars and the ground
    Tap bus;                      // distort: the bus's columns
    bool on_bus = false;
    float wobble = 0, phase = 0;  // distort: the haze's terms that depend on x alone
  };
  // A tile's screen column: its image columns and the factors of its ownership weight that depend on x.
  struct TileColumn {
    Tap t;
    float band = 1.f, feather_left = 1.f, feather_right = 1.f;
    float x_weight = 1.f;  // band * feather_left * feather_right: the weight where the factors of y are all 1
  };
  static constexpr int kMaxTiles = 64;  // tiles of a group, and of the groups drawn in one pass

  LightView light_view(const Light& light);  // fills light4_
  void background_columns(const Params& P, const LightView& L);
  Tap light_tap(int y, const Params& P, const LightView& L) const;  // the light's rows the background reads at screen row y
  void light_columns(const Params& P, const LightView& L, Pool& pool);  // fills light_x_ (after background_columns())
  struct StarCache {  // the runs of columns with a star in a band of sky rows (per task: rows come in order)
    std::uint32_t star_y = 0;
    bool valid = false;
    int n = 0;
    std::array<std::array<int, 2>, 32> runs;
  };
  void background_row(int y, const Params& P, const LightView& L, std::span<const std::array<float, 4>> scorch, StarCache& cache);
  void take_tiles(std::span<Module* const> modules);  // into tiles_
  // The background (if L) and the groups of tiles_, row by row in one pass (in passes of up to kMaxTiles tiles).
  void compose(const Params& P, const LightView* L, std::span<const std::array<float, 4>> scorch, Pool& pool);
  void distort_impl(const Params& P, std::span<const Shock> shocks, const HeatView& H, Pool& pool);
  void bloom_impl(float threshold, float strength, Pool& pool);
  void finish_impl(const Params& P, std::span<std::uint8_t> rgb, Pool& pool);
  friend struct FrameKernels;  // the row kernels (compose.cpp)
  void settle_row(int y);    // the distortion's moved pixels of row y back into the screen
  void settle(Pool* pool);   // every row's, and bloom's last pass, if pending (for stages that do not do them on the way)
  int w_, h_, blocks_;
  bool avx2_;  // the row kernels compiled for AVX2 (the same bits as the baseline's)
  Image4 screen_, tmp_;
  std::vector<Image4> mips_, mips_tmp_;
  std::vector<Column> cols_;                         // [w]
  std::vector<TileColumn> tile_cols_;                // [kMaxTiles][w]
  std::vector<std::array<int, 2>> seen_cols_;        // [kMaxTiles]: the screen columns that see each tile's image
  std::vector<std::vector<Tap>> up_x_, up_y_;        // bloom: each level read from the next coarser (0: the screen)
  std::vector<float> light4_;  // the light field with a fourth channel (0), a cell to a vector; sized on the first frame
  // The first half of the background's bilinear light lookup, done once per row of the light grid instead of once per
  // pixel: rows of light4_ resampled at the screen columns, [light ny][w][4] (the rows the screen reads).
  std::vector<float> light_x_;
  std::vector<std::array<int, 2>> star_runs_;  // runs of columns [x0, x1) with the same star key
  std::vector<float> fin_vig_;                 // finish(): per column, the vignette's term of x
  std::vector<std::uint32_t> fin_key_;         // and the grain's hash key of x
  // The same for distort()'s lookup of the bus's heat, [bus ny][w]; and per block of kBlock pixels, the bus columns
  // its pixels read, [lo, hi] (lo > hi: none).
  std::vector<float> heat_x_;
  std::vector<std::array<int, 2>> block_bus_;
  std::vector<Tile> tiles_;    // the modules to draw, in order
  std::vector<int> order_;     // tiles_ by group, in drawing order
  std::vector<Group> groups_;
  // distort(): per row and block of kBlock pixels, the pixels [a, b) it moved (written to tmp_; the rest stay as they
  // are), copied back into the screen by the next stage that reads it (pending: moved_rows_).
  std::vector<std::array<int, 2>> moved_;
  bool moved_rows_ = false;
  bool layers_on_ = false;
  std::array<Image4, kLayers> layers_;     // use_layers()
  std::array<Image4, 3> layer_acc_;        // per task of compose(), a row: a group's albedo, emission, normals by weight
  std::vector<std::array<int, 2>> layer_span_;  // per row: the columns [a, b) the layers were drawn in (cleared there next)
  bool bloom_pending_ = false;  // bloom's last pass: the screen += bloom_gain_ * mip 0 (bilinear) / bloom_div_
  float bloom_gain_ = 0.f, bloom_div_ = 1.f;
  // capture()
  Params P_{};
  LightView light_cap_;
  HeatView heat_cap_;
  std::vector<float> heat_;
  std::vector<std::array<float, 4>> scorch_;
  std::vector<Shock> shocks_;
  Particles parts_{0};
  std::array<double, kRenderStages> render_ms_{};
};

// Draws captured frames (Frame::capture, Frame::render) on a thread of its own, so that the caller can compute the next
// frame's state meanwhile. Both use the same pool: while one waits, it runs the other's tasks.
//
//   per frame f: state(f); wait_images(); shade(f); wait(); [use the picture of f - 1]; frame.capture(...); start(rgb)
class PictureThread {
 public:
  PictureThread(Frame& frame, Pool& pool);
  ~PictureThread();
  PictureThread(const PictureThread&) = delete;
  PictureThread& operator=(const PictureThread&) = delete;
  // Draws the frame captured last into rgb; returns at once. The previous picture must be done (wait()).
  void start(std::span<std::uint8_t> rgb, float bloom_threshold, float bloom_strength);
  void wait_images();  // until the modules' images of the picture being drawn have been read: they may be shaded again
  void wait();         // until the picture is done
  // Wall ms the caller spent in the last wait_images() and wait().
  double waited_images_ms = 0, waited_ms = 0;

 private:
  void loop();
  Frame& frame_;
  Pool& pool_;
  std::span<std::uint8_t> rgb_;
  float threshold_ = 1.f, strength_ = 1.f;
  std::atomic<bool> images_read_{true}, done_{true}, go_{false}, stop_{false};
  std::thread thread_;
};

}  // namespace nfx::compose
