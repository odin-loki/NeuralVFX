// A small scene with water of every kind (docs/EFFECTS_WATER.md), for the tests (test_water.cpp) and the allocation
// check (alloc_test.cpp): a pool with rain, swell and a current, churned by what lands on it from the bus, a sheet, jets,
// drips, bubbles, a rain shaft and spray thrown off the bus's soot, a splash with its mist, a wake made every frame, and
// a module of the tests' tiny effect taking the mist (its soot is what the spray and the churn read).
#pragma once

namespace nfx::compose::testing {

inline constexpr const char* kWaterScene = R"(
scene size 200 x 120, fps 30, length 2 s, ground 110
keyframes 0.5, 1.5
effect tiny = "tiny"
look gas = shader, heat_scale 1, emission 0, soot_density 2, sky 1, relief 0.5
module mist = tiny, size 32, width 80, at (100, 112), look gas, controls (0, 0.5, 0.5), start 0, seed 2
water lake = pool, along (10, 100) to (190, 100), depth 14, surface 8, cells 64, rate 2, swell 1, drift 10, churn 1, seed 3
water falls = sheet, along (60, 10) to (90, 10), flow 0.8, velocity (10, 30), strands 10, capacity 48, onto lake, mist mist,
              seed 4
water fount = jet, at (150, 100), speed 200, jets 3, fan 10, pulse 0.3, onto lake, mist mist, mist_heat 0.4, seed 5
water leak = drips, along (20, 20) to (40, 20), rate 4, radius 2, onto lake, seed 6
water fizz = bubbles, in (120, 102), size 40 x 10, rate 8, radius 2, onto lake, seed 7
water storm = shaft, in (0, 0), size 60 x 100, density 30, wind (30, 0), sway 20, mist mist, seed 8
water flecks = spray, in (60, 30), size 80 x 70, density 400, threshold 0.02, radius 1.5, onto lake, seed 12
at 0.3: splash at (110, 100), power 0.8, onto lake, mist mist, seed 9
at 1.2: splash at (40, 100), power 1.2, onto lake, seed 10
every frame: wave lake at (30 + 60 * t, 100), amount 30, width 8
)";

// The same kinds as library entries (docs/LIBRARY.md), for a stage: spawned, moved, set, stopped and spawned again.
inline constexpr const char* kWaterCatalogue = R"(
effect tiny = "tiny.nvfx"
look gas = shader, heat_scale 1, emission 0, soot_density 2, sky 1, relief 0.5
entry falls, category water, size 160 x 96, anchor base, loop, about "a waterfall into a pool, with mist"
param flow = 0.8, range (0.3, 1.4), about "how much water"
module mist = tiny, size 32, width 64 * scale, at (X, Y), look gas, controls (0, 0.5, 0.5), seed 2
water lake = pool, along (X - 78 * scale, Y - 10 * scale) to (X + 78 * scale, Y - 10 * scale), depth 10 * scale,
             surface 10 * scale, cells 48, rate 1, seed 3
water sheet = sheet, along (X - 20 * scale, Y - 90 * scale) to (X + 20 * scale, Y - 90 * scale), flow flow,
              velocity (0, 30 * scale), strands 8, capacity 40, onto lake, mist mist, seed 4
water fount = jet, at (X + 50 * scale, Y - 10 * scale), speed 160 * scale, jets 2, fan 12, onto lake, seed 5
water leak = drips, along (X - 70 * scale, Y - 80 * scale) to (X - 50 * scale, Y - 80 * scale), rate 3, onto lake, seed 6
water fizz = bubbles, in (X - 40 * scale, Y - 9 * scale), size 40 * scale x 8 * scale, rate 6, onto lake, seed 7
every frame: wave lake at (X - 60 * scale + 40 * t * scale, Y - 10 * scale), amount 20, width 6
entry plunge, category water, size 120 x 80, anchor base, once 2, show 0.2, about "a splash"
param power = 1, range (0.4, 1.6), about "how big"
water lake = pool, along (X - 58 * scale, Y - 8 * scale) to (X + 58 * scale, Y - 8 * scale), depth 8 * scale,
             surface 8 * scale, cells 48, seed 11
water storm = shaft, in (X - 50 * scale, Y - 80 * scale), size 100 * scale x 72 * scale, density 20, seed 12
at 0: splash at (X, Y - 8 * scale), power power, onto lake, seed 13
)";

}  // namespace nfx::compose::testing
