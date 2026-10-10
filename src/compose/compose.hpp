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
class Pool {
 public:
  explicit Pool(int threads);
  ~Pool();
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;
  int threads() const { return static_cast<int>(workers_.size()) + 1; }
  template <class F>
  void run(int n, F&& f) {
    run_impl(n, [](void* ctx, int i) { (*static_cast<std::remove_reference_t<F>*>(ctx))(i); }, const_cast<void*>(static_cast<const void*>(&f)));
  }

 private:
  using Fn = void (*)(void*, int);
  void run_impl(int n, Fn fn, void* ctx);
  void work();
  void drain();
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  Fn job_ = nullptr;
  void* ctx_ = nullptr;
  int n_ = 0;
  std::atomic<int> next_{0}, left_{0};
  std::uint64_t generation_ = 0;
  int busy_ = 0;
  bool stop_ = false;
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
Isa best_isa();
const char* isa_name(Isa isa);
std::unique_ptr<rt::RolloutRunner> make_runner(const rt::RolloutEffect& e, int size, Isa isa);

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
};

class Light;

class Module {
 public:
  Module(std::string name, const rt::RolloutEffect& e, int size, Placement at, Isa isa);

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
  void take_over(const Module& from);                         // the other's physical state and fine fields
  void step();
  void shade(const Light* light);  // into image(): learned renderer or field shader, tile space
  const Image4& image() const { return img_; }

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
  double step_ms = 0, shade_ms = 0;
  int frames = 0;                 // frames stepped since start

 private:
  std::string name_;
  const rt::RolloutEffect& e_;
  int size_;
  Isa isa_;
  std::unique_ptr<rt::RolloutRunner> r_;
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
struct ForceField {
  enum class Kind { ceiling, vortex, wind } kind = Kind::ceiling;
  float x = 0, y = 0, radius = 100.f, strength = 0.f, soft = 80.f, u = 0, v = 0;
  float damping = 0.25f;  // ceiling: fraction of vertical velocity removed per frame at full depth
};
void apply(Module& m, const ForceField& f, float weight = 1.f);
// Move `fraction` of the material in from's coarse rows [row0, res) (y up; all rows: row0 = 0) into the modules of
// `to` that cover it (by their ownership weights; fine fields follow). Material that lands outside them stays. What
// arrives is scaled by the gains (one model's thin soot can be another's smoke; 1 conserves the amount).
void transfer(Module& from, std::span<Module* const> to, float fraction, int row0 = 0, float heat_gain = 1.f, float soot_gain = 1.f);

// --- light -------------------------------------------------------------------------------------------------------------

// Light in the world from everything hot on the bus: each bus cell's heat emits a blackbody-like colour, spread by a
// pyramid of blurs (a wide, soft falloff), plus a global flash. Sampled by the field shader and the ground.
class Light {
 public:
  explicit Light(const FieldBus& bus);
  // Rows are spread over the pool; the light is the same on any number of threads.
  void update(const FieldBus& bus, float gain, std::array<float, 3> flash, Pool& pool);
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

enum class Kind : std::uint8_t { spark, ember, debris, flake };

// Particles in world pixels and seconds, a fixed capacity (spawns beyond it are dropped). Embers and sparks glow by
// temperature and are drawn additively with motion blur; debris and flakes are dark and drawn over.
class Particles {
 public:
  explicit Particles(int capacity);
  int alive() const { return n_; }
  int capacity() const { return static_cast<int>(x_.size()); }
  // One particle; returns false when full.
  bool spawn(Kind k, float x, float y, float vx, float vy, float temp, float size, float life);
  struct Landing {
    float x = 0, temp = 0;
    Kind kind = Kind::ember;
  };
  // Gravity, drag towards the bus flow (others' flow seen by all particles), ground at ground_y (bounces or settles);
  // embers cool. Landings of hot particles are recorded (cleared each update).
  void update(float dt, const FieldBus* bus, float flow_gain, float ground_y);
  std::span<const Landing> landings() const { return std::span<const Landing>(landed_).first(zs(n_landed_)); }
  // Draw into a screen image (camera offset: world minus screen).
  void draw(Image4& screen, float cam_x, float cam_y) const;
  std::uint64_t rng = 0x9E3779B97F4A7C15ULL;
  float uniform();  // [0, 1)

 private:
  void kill(int i);
  int n_ = 0;
  std::vector<float> x_, y_, px_, py_, vx_, vy_, temp_, size_, life_, age_, cool_;
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
  void particles(const Particles& p) { p.draw(screen_, cam_x, cam_y); }
  void distort(std::span<const Shock> shocks, const FieldBus& bus, Pool& pool);
  void bloom(float threshold, float strength, Pool& pool);
  // Tone map, vignette and grain into RGB8 (width * height * 3).
  void finish(std::span<std::uint8_t> rgb, Pool& pool);
  const Image4& screen() const { return screen_; }

  // One axis of a bilinear lookup: the two cells read and their weights. Taken once per column or row, it leaves a
  // blend per pixel.
  struct Tap {
    int i0 = 0, i1 = 0;
    float w0 = 1.f, w1 = 0.f;
  };

 private:
  // What a stage needs of a screen column, worked out once per frame instead of once per pixel.
  struct Column {
    float wx = 0;                 // world x of the column's centre
    float hill = 0;               // background: world y of the hills' outline
    std::array<float, 4> hill_colour{};
    Tap light;                    // background: the light field's columns
    std::uint32_t star = 0, tex = 0, grain = 0;  // hash keys of the stars, the ground and the grain
    Tap bus;                      // distort: the bus's columns
    bool on_bus = false;
    float wobble = 0, phase = 0;  // distort: the haze's terms that depend on x alone
    float vig = 0;                // finish: the vignette's x term
  };
  // A tile's screen column: its image columns and the factors of its ownership weight that depend on x.
  struct TileColumn {
    Tap t;
    float band = 1.f, feather_left = 1.f, feather_right = 1.f;
  };
  static constexpr int kMaxTiles = 64;  // tiles of a group drawn together

  void draw_group(std::span<Module* const> tiles, Pool& pool);
  int w_, h_;
  Image4 screen_, tmp_;
  std::vector<Image4> mips_, mips_tmp_;
  std::vector<Column> cols_;                         // [w]
  std::vector<TileColumn> tile_cols_;                // [kMaxTiles][w]
  std::vector<std::vector<Tap>> up_x_, up_y_;        // bloom: each level read from the next coarser (0: the screen)
  std::vector<float> light4_;  // the light field with a fourth channel (0), a cell to a vector; sized on the first frame
};

}  // namespace nfx::compose
