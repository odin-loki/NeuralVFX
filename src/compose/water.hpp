// Water for scenes (docs/EFFECTS_WATER.md): a deterministic water system. Its motion is a seeded simulation, its look is
// drawn, and its mist is a learned effect fed through the existing couplings:
//   - pools: a water surface along a line, a 1D height field (a damped wave equation) excited by what lands on it, with
//     foam carried by its current, ripple rings where drops land (when its surface is seen from above) and rain;
//   - sheets: water pouring over a lip, falling as strands (ribbons of nodes) that break into drops and spray;
//   - jets: water thrown from a nozzle (fountains, hydrants, geysers), strands that rise, arc over and break up;
//   - drips: drops that swell at points along a line, let go and fall;
//   - bubbles: rising, wobbling, popping at a pool's surface;
//   - shafts: a rain shaft seen from afar, streaks in a curtain whose density swells and thins in gusts;
//   - splashes (an action): a crown (two walls and the faint face between them) and drops thrown up where something hits
//     the water, a jet from the crater, foam, waves.
// What lands gives its mist to a module (a learned steam or fog effect, WaterLayout::mist) through deposit().
//
// Determinism: every random number is a hash of the element's seed, the frame and a counter (a particle's index, a
// strand's), never a shared generator, and the simulation runs on the scene's own thread in a fixed order, so the same
// script gives the same frames on any number of threads. Capacities are fixed when the scene is made (what is spawned
// beyond them is dropped, the oldest first where it says so); stepping, the picture's capture and its drawing allocate
// nothing.
#pragma once

#include "compose.hpp"
#include "water_picture.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace nfx::compose {

enum class WaterKind : std::uint8_t { pool, sheet, jet, drips, bubbles, shaft };
const char* water_kind_name(WaterKind k);

// One element's settings for a frame (world pixels, y down; seconds). What each kind reads is said beside it.
struct WaterSpec {
  bool on = true;                        // emits only while on; a pool is drawn only while on
  float x0 = 0, y0 = 0, x1 = 100, y1 = 0;  // pool, sheet, drips: the line from (x0, y0) to (x1, y1); jet: the nozzle at
                                           // (x0, y0); bubbles, shaft: the region's top-left corner (x0, y0), size x1 by y1
  float flow = 0.6f;          // sheet, jet: how much water (0 to about 1.5): how thick its strands are
  float vx = 0, vy = 40;      // sheet: the water's velocity at the lip (world pixels a second)
  float speed = 0;            // jet: its throw; shaft: the rain's fall; bubbles: their rise; pool: the waves' speed (0: defaults)
  float angle = 0;            // jet: degrees from upright (positive: to the right)
  float spread = 3;           // jet: degrees of jitter of its throw
  float fan = 0;              // jet: degrees between neighbouring jets
  float pulse = 0;            // jet: how much its throw swings (0 to 1), with smooth noise
  float breakup = 1;          // sheet, jet: how soon strands break into drops (2: twice as soon)
  float gravity = 700;        // world pixels a second squared: everything that falls
  float wu = 0, wv = 0;       // wind on drops and spray (shaft: on the rain; it also slants it), world pixels a second
  float rate = 1;             // drips and bubbles: per second; pool: rain drops a second per 100 pixels of it
  float radius = 2;           // drips: a drop's; bubbles: the largest's
  float depth = 60;           // pool: how deep its body reaches below its line
  float damping = 1.2f;       // pool: how fast its waves die (per second)
  float drift = 0;            // pool: its current (foam rides it), world pixels a second
  float surface = 0;          // pool: its surface seen from above, this tall (0: edge-on); ripples are drawn on it
  float swell = 0, wavelength = 120, period = 3;  // pool: steady waves of this height, length and period
  float density = 20;         // shaft: streaks at once per 100 x 100 pixels
  float sway = 0;             // shaft: how far its wind swings (world pixels a second)
  float gust = 0.6f;          // shaft: how much its density swells and thins along it and over time (0 to 1)
  float floor = 600;          // where what falls lands when it is not over its pool (world y)
  float mist_amount = 1;      // mist given to the mist module per unit of water landing
  float mist_heat = 0.2f;     // and warmth with it, a fraction of the mist (steam rises with it; fog takes none: 0)
  float brightness = 1;       // how much light it takes (sky and scene)
  float clarity = 0.5f;       // pool, shaft: how see-through (0: opaque, 1: clear)
  std::array<float, 3> colour{0.62f, 0.8f, 0.95f};  // what the water reflects
};

// An element as the scene is made: its kind and what is fixed for good (capacities, counts, links).
struct WaterLayout {
  WaterKind kind = WaterKind::pool;
  int strands = 24;    // sheet: strands across the lip; jet: jets
  int nodes = 72;      // sheet, jet: nodes in flight per strand (frames: a node is emitted every frame)
  int cells = 96;      // pool: its height field
  int count = 256;     // drips: drops forming at once; bubbles: at once; shaft: streaks at once; pool: foam
  int onto = -1;       // sheet, jet, drips, bubbles: the pool (an element index) it lands on or pops at; -1: none
  Module* mist = nullptr;  // the module its landing water feeds with mist (a learned steam or fog effect); none
  std::uint64_t seed = 1;
};

class Water {
 public:
  // `layout`: the elements, in order. `drops`: drops and spray at once in the whole scene; `crowns`: splashes at once;
  // `rings`: ripple rings at once.
  Water(std::span<const WaterLayout> layout, int drops, int crowns, int rings);
  int elements() const { return static_cast<int>(layout_.size()); }
  const WaterLayout& layout(int i) const { return layout_[zs(i)]; }
  WaterSpec& spec(int i) { return spec_[zs(i)]; }
  const WaterSpec& spec(int i) const { return spec_[zs(i)]; }

  // One frame (dt seconds, frame number `frame`): emit, move, land, feed the mist. Pools first, then what lands on them.
  void step(float t, float dt, int frame);
  // A splash at (x, y) where something hits: a crown of `power` (about 0.2 to 1.5), on pool `onto` (-1: on the ground at
  // y), its mist into `mist` (none: nullptr). Takes the oldest splash's place when they are all in use.
  void splash(float x, float y, float power, int onto, Module* mist, float mist_amount, float mist_heat, std::uint64_t seed, float t, int frame);
  // Waves on a pool at x: its surface pushed down by `amount` (world pixels a second) over `width` pixels (a wake when
  // done every frame along a path).
  void wave(int pool, float x, float amount, float width);
  // The `wave` action: waves as wave(), and where it pushes hard, foam left on the surface (a wake's); frame `frame`.
  void stir(int pool, float x, float amount, float width, int frame);
  // The picture of the water now, in screen pixels (camera at cam).
  void picture(WaterPicture& out, float cam_x, float cam_y, int screen_w, int screen_h) const;
  void reserve(WaterPicture& p, int screen_w, int screen_h) const;  // room in a picture for everything this water may draw

  // For tests and tools: live drops, and a pool's surface height at x (world pixels, up positive; 0 off it).
  int drops_alive() const;
  float pool_height(int pool, float x) const;
  float mist_given() const { return mist_given_; }  // mist given to modules since the start (soot units)
  struct Counts {
    int nodes = 0, drops = 0, foam = 0, rings = 0, bubbles = 0, drips = 0, streaks = 0, crowns = 0;
  };
  Counts counts() const;

 private:
  struct Node {  // a piece of a strand: emitted at its source each frame, falling, breaking into a drop
    float x = 0, y = 0, vx = 0, vy = 0, w = 0, age = 0, life = 0;  // life: when it breaks (seconds after its birth)
    float v0 = 1;                                                 // its speed when it left: it thins as it speeds up
    std::int32_t born = 0;                                        // when it left (frame and sub-step): pairs it across strands
    std::uint8_t alive = 0, broken = 0;
  };
  struct Drop {  // a drop or spray
    float x = 0, y = 0, vx = 0, vy = 0, r = 1, age = 0, life = 2;
    float g = 700, wu = 0, floor = 600;  // its pull, the wind it drifts towards, and where it lands off its pool
    int source = -1;          // what it feeds when it lands (an element, or crowns at elements() + c; -1: nothing)
    int pool = -1;            // the pool it lands on
    std::uint8_t bounced = 0, alive = 0, white = 0;
  };
  struct Ring {
    float x = 0, y = 0, t0 = 0, size = 10, life = 1;
    int pool = -1;
  };
  struct Foam {
    float x = 0, vx = 0, size = 3, age = 0, life = 3;
    std::uint8_t alive = 0;
  };
  struct Bubble {
    float x = 0, y = 0, r = 2, vy = 40, age = 0, phase = 0, wobble = 2;
    std::uint8_t alive = 0;
  };
  struct Drip {
    float x = 0, y = 0, r = 2, age = 0, form = 1;
    std::uint8_t alive = 0;
  };
  struct Streak {
    float x = 0, y = 0, vx = 0, vy = 500, len = 20, alpha = 0.3f, bottom = 0;
    std::uint8_t alive = 0;
  };
  struct Crown {  // a splash: its crater's jet, and what it feeds
    float x = 0, y = 0, t0 = 0, power = 1, mist_amount = 1, mist_heat = 0.2f;
    int onto = -1;
    Module* mist = nullptr;
    std::uint64_t seed = 1;
    std::uint8_t alive = 0;
  };
  struct Bins {  // what lands, per element (and per crown), to feed its mist: kBins bins along x
    float lo = 0, hi = 1, y = 0;
    std::array<float, 8> mass{};
  };
  static constexpr int kBins = 8;
  // A splash's strands: the crown's two walls (kCrownWalls a side), thrown for its first 0.12 s, then the jet
  // from the crater.
  static constexpr int kCrownWalls = 1, kCrownStrands = 2 * kCrownWalls + 1, kCrownNodes = 32;

  float land_y(int pool, float x, float floor) const;  // where something over x lands
  float surface_y(int pool, float x) const;            // a pool's surface at x (world y), its line off it
  void land(int source, int pool, float x, float y, float mass, float speed, float r, int frame, std::uint32_t salt);
  void spawn_drop(const Drop& d);
  void add_ring(int pool, float x, float size, float t);
  void add_foam(int pool, float x, float vx, float size, float life);
  void step_pool(int i, float t, float dt, int frame);
  void step_strands(int i, float t, float dt, int frame);
  void step_drips(int i, float t, float dt, int frame);
  void step_bubbles(int i, float t, float dt, int frame);
  void step_shaft(int i, float t, float dt, int frame);
  void step_crowns(float t, float dt, int frame);
  void step_drops(float dt, int frame);
  void feed_mist();

  std::vector<WaterLayout> layout_;
  std::vector<WaterSpec> spec_;
  // per element, offsets into the shared arrays
  std::vector<int> node0_, strand0_, cell0_, foam0_, item0_;
  std::vector<Node> nodes_;               // [element strands][nodes]: rings, oldest first from head_
  std::vector<int> head_, count_;         // per strand
  std::vector<float> h_, hv_;             // pools' heights and their rates (world pixels, up positive)
  std::vector<Foam> foam_;
  std::vector<Bubble> bubbles_;
  std::vector<Drip> drips_;
  std::vector<Streak> streaks_;
  std::vector<int> next_item_;            // per element: where its next bubble, drip or streak goes (round robin)
  std::vector<Drop> drops_;
  int next_drop_ = 0;
  std::vector<Ring> rings_;
  int next_ring_ = 0;
  std::vector<Crown> crowns_;
  std::vector<Node> crown_nodes_;         // [crowns][kCrownStrands][kCrownNodes]
  std::vector<int> crown_head_, crown_count_;  // [crowns][kCrownStrands]
  int next_crown_ = 0;
  std::vector<Bins> bins_;                // [elements + crowns]
  float time_ = 0, dt_ = 1.f / 30.f;  // the last step's time and length
  float mist_given_ = 0;
};

}  // namespace nfx::compose
