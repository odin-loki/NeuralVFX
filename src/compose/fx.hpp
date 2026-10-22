// The emitters of the procedural elements added for the library (docs/COMPOSE.md §4.10): what `emit hail`, `aurora`,
// `rainbow`, `meteors`, `beam`, `shield` and `glints` make each frame, and the spatter and shards of a `burst`. None is
// learned: they make particles of the kinds of fx_particles.cpp (and glows) in the scene's pool, so nothing is allocated.
// Their random numbers come from a hash of the emitter's place in the script, its seed and the frame (as the weather's
// and the glows' do), so a scene draws the same random numbers with them or without them; a burst's spatter and shards
// draw from the scene's generator after its sparks, as the sparks do. (Water's droplets, ripples and bubbles are not
// here: they belong to the water system.)
#pragma once

#include "compose.hpp"

#include <cmath>
#include <span>
#include <string_view>

namespace nfx::compose::fx {

enum class Emitter : std::uint8_t { hail, aurora, rainbow, meteors, beam, shield, glints };

// The emitter an `emit` word names (false: not one of these).
bool emitter_of(std::string_view word, Emitter& out);
// The keys each takes, besides `from`, `until` and `if` (which every emitter takes), and the keys it needs.
std::span<const std::string_view> keys(Emitter e);
std::span<const std::string_view> needs(Emitter e);

// The values an emitter is given. A key fills one, or more: `at` and `in` (x, y), `along` (x, y, x1, y1), `size`
// (w, h), `colour` (r, g, b), `top` (tr, tg, tb), `wind` (wu, wv), `direction` (dx, dy).
enum V : int {
  v_x, v_y, v_x1, v_y1, v_w, v_h, v_r, v_g, v_b, v_tr, v_tg, v_tb, v_wu, v_wv, v_dx, v_dy,
  v_rate, v_count, v_spread, v_speed, v_radius, v_width, v_bright, v_jitter, v_squash, v_density, v_flow, v_length, v_flare,
  v_fade, v_hit, v_angle, v_seed, kValues
};
struct Slot {
  V first = v_x;
  int n = 1;
};
// Where a key's values go (false: not a key of these emitters).
bool slot_of(std::string_view key, Slot& out);

struct Emit {
  Emitter kind = Emitter::hail;
  std::array<float, kValues> v{};  // NaN: not given
  bool has(V k) const { return !std::isnan(v[static_cast<std::size_t>(k)]); }
  float get(V k, float d) const { return has(k) ? v[static_cast<std::size_t>(k)] : d; }
};

// What it makes this frame, at scene time t. `fresh`: it starts (or starts again) this frame, so the weather-like ones
// fill their region at once; `salt`: the emitter's place in the script, mixed into its hash.
void emit(const Emit& e, Particles& P, float t, float fps, bool fresh, std::uint32_t salt);

// A burst's spatter and shards (after its embers, debris and sparks): thrown from within `radius` of (x, y) at about
// `speed` within `cone` degrees of (dx, dy); `colour` (NaN: dark red for spatter, ice for shards), `grain` (a drop's
// radius or a shard's length; NaN: 1.6 or 9); spatter lies as a stain `stain` seconds where it lands on `floor` (NaN: the
// ground), shards bounce and lie there.
struct Burst {
  float x = 0, y = 0, radius = 0, speed = 0, dx = 0, dy = -1, cone = 360;
  int spatter = 0, shards = 0;
  std::array<float, 3> colour{};
  float grain = 0, stain = 0, floor = 0;
};
void burst(const Burst& b, Particles& P, float ground);

}  // namespace nfx::compose::fx
