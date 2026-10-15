// The picture of a scene's water (water.hpp, docs/EFFECTS_WATER.md) as the frame draws it: what the simulation looked
// like when the frame was captured, in screen pixels, drawn a row at a time. Kept apart from water.hpp so that the frame
// (compose.hpp) can hold one.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace nfx::compose {

// The scene's light as the water samples it (Frame passes its own).
struct WaterLight {
  const float* L4 = nullptr;  // [ny][nx][4]
  int nx = 0, ny = 0;
  float x0 = 0, y0 = 0, cell = 1;
  std::array<float, 3> flash{};
};

// What the picture draws, captured from the simulation (Water::picture) so that it can be drawn on another thread while
// the simulation moves on; drawn a screen row at a time (draw_row), rows in any order and on any thread.
class WaterPicture {
 public:
  // Primitives: a capsule (a segment with a radius, shaded as a cylinder or a sphere), a ring (an ellipse's outline), or
  // a quad (a piece of a sheet of water between four nodes, convex, shaded flat with its own normal).
  enum Shape : std::uint8_t { kCapsule = 0, kRing = 1, kQuad = 2 };
  struct Prim {
    float ax = 0, ay = 0, bx = 0, by = 0, r = 1;  // screen pixels; a ring: centre (ax, ay), radius bx, squash by
    float cx = 0, cy = 0, dx = 0, dy = 0;         // a quad: its corners a, b, c, d in order
    float nx = 0, ny = 0;                         // a quad: its normal's x and y (up positive; z towards the viewer)
    float alpha = 1;                              // coverage at its core
    float red = 1, green = 1, blue = 1;           // what it reflects
    float glint = 0.5f;                           // specular
    float sky = 1, light = 1;                     // how much sky and scene light it takes
    std::uint8_t ring = kCapsule;                 // its shape
    int group = -1;                               // pieces of one strand (consecutive): drawn as one, by their union
    int y0 = 0, y1 = 0, x0 = 0, x1 = 0;           // the screen pixels it may touch (inclusive)
  };
  struct Pool {
    float x0 = 0, x1 = 0, y = 0, depth = 0, surface = 0, clarity = 0.5f, brightness = 1, time = 0;
    std::array<float, 3> colour{};
    int first = 0, cells = 0;             // its heights (screen y of the surface per cell, up is smaller) in heights_
    int foam = 0, foam_n = 0;             // its foam (x, size, alpha) in foam_
    int ring = 0, ring_n = 0;             // its rings (x, radius, alpha) in rings_
    int y0 = 0, y1 = 0;                   // screen rows
  };
  struct Shaft {
    float x0 = 0, y0 = 0, w = 0, h = 0, slant = 0, veil = 0, brightness = 1;  // slant: x shift per pixel down
    std::array<float, 3> colour{};
    int first = 0;  // its density profile (kProfile values across its width at its top) in profile_
  };
  static constexpr int kProfile = 64;
  static constexpr int kBand = 16;  // rows per bin

  void reserve(int prims, int pools, int cells, int foam, int rings, int shafts, int rows);
  void clear(int rows);
  bool empty() const { return prims_.empty() && pools_.empty() && shafts_.empty(); }
  // Sorts the primitives into bins of kBand rows (after they are all added).
  void bin();
  // Draws screen row y over `screen` (premultiplied RGBA, colour only: its alpha is kept) and, when given, over the four
  // engine layers' rows (colour, albedo, emission, normal: Frame::Layer). Returns the columns [a, b) it drew in.
  std::array<int, 2> draw_row(int y, float* screen, float* colour, float* albedo, float* emission, float* normal, int width,
                              const WaterLight& light) const;

  std::vector<Prim> prims_;
  std::vector<Pool> pools_;
  std::vector<float> heights_;               // screen y of each pool cell's surface
  std::vector<std::array<float, 3>> foam_;   // x, half-size, alpha
  std::vector<std::array<float, 3>> rings_;  // x, radius, alpha
  std::vector<Shaft> shafts_;
  std::vector<float> profile_;
  std::vector<int> bin_start_, bin_items_;   // per band, [start, start of the next) in bin_items_
  int rows_ = 0;
  float cam_x = 0, cam_y = 0;                // world position of the screen's corner (the light is sampled in the world)
};

}  // namespace nfx::compose
