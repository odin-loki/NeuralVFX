// A small composed scene for the tests (test_compose.cpp, alloc_test.cpp): an untrained rollout model, two tiles of
// one domain coupled by blend_band, a third effect pushed by them and feeding them through transfer, a force field,
// particles, light, distortion and bloom. Everything the fireball scene does, at a size that runs in milliseconds.
#pragma once

#include "compose.hpp"

#include <neuralfx/rollout.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace nfx::compose::testing {

inline rt::RolloutEffect tiny_effect(std::uint64_t seed = 3) {
  rollout::Hyper h;
  h.res = 16;
  h.hidden = 8;
  h.memory = 2;
  h.jacobi = 10;
  h.render_hidden = 6;
  h.warmup = 4;
  rt::RolloutEffect e;
  e.m = rollout::init_model(h, seed);
  e.m.effect = "tiny";
  e.m.control_names = {"intensity", "wind", "turbulence"};
  e.m.scale = {0.2f, 0.2f, 0.4f, 0.3f};
  e.m.lo = {-2.f, -2.f, 0.f, 0.f};
  e.m.hi = {2.f, 2.f, 3.f, 3.f};
  e.m.detail.swirl_control = 2;
  for (int k = 0; k < 2; ++k) {
    rollout::StartPoint sp;
    sp.controls = {0.3f + 0.4f * static_cast<float>(k), 0.5f, 0.5f};
    sp.seed = 10 + static_cast<std::uint64_t>(k);
    sp.coarse.resize(static_cast<std::size_t>(h.res) * static_cast<std::size_t>(h.res) * rollout::kPhys);
    for (int y = 0; y < h.res; ++y) {
      for (int x = 0; x < h.res; ++x) {
        float* c = sp.coarse.data() + (static_cast<std::size_t>(y) * static_cast<std::size_t>(h.res) + static_cast<std::size_t>(x)) * rollout::kPhys;
        c[0] = 0.1f * std::sin(0.5f * static_cast<float>(x + k));
        c[1] = 0.2f + 0.1f * std::cos(0.4f * static_cast<float>(y));
        c[2] = std::max(0.f, 1.2f - 0.08f * static_cast<float>(std::abs(x - 8) + y));
        c[3] = std::max(0.f, 0.7f - 0.05f * static_cast<float>(std::abs(x - 7) + std::abs(y - 6)));
      }
    }
    e.m.starts.push_back(sp);
  }
  return e;
}

struct MiniScene {
  rt::RolloutEffect e = tiny_effect();
  std::vector<std::unique_ptr<Module>> mods;
  FieldBus bus{-16.f, -16.f, 24, 16, 8.f, 4};
  Light light{bus};
  Particles parts{512};
  Frame frame{160, 90};
  Pool pool;
  std::vector<Module*> all;
  std::vector<Shock> shocks;

  explicit MiniScene(int threads) : pool(threads) {
    for (int i = 0; i < 2; ++i) {  // two tiles side by side, 64 world pixels each, overlapping by 4 cells (16 px)
      mods.push_back(std::make_unique<Module>("tile" + std::to_string(i), e, 32, Placement{16.f + 48.f * static_cast<float>(i), 10.f, 2.f}, best_isa()));
      Module& m = *mods.back();
      m.group = 0;
      m.band = {i == 1 ? 4 : 0, i == 0 ? 4 : 0, 0, 0};
      m.look = Look::shader;
      m.start(i, 7 + static_cast<std::uint64_t>(i));
    }
    mods.push_back(std::make_unique<Module>("fire", e, 32, Placement{50.f, 40.f, 1.f}, best_isa()));
    Module& f = *mods.back();
    f.group = 1;
    f.feather = 4.f;
    f.start(1, 99);
    for (const auto& m : mods) all.push_back(m.get());
    shocks.push_back({80.f, 50.f, 0.1f, 600.f, 0.2f, 3.f, 8.f});
    frame.ground_y = 80.f;
  }

  // One frame of the whole pipeline into rgb (160 * 90 * 3).
  void step(int f, std::span<std::uint8_t> rgb) {
    const float t = static_cast<float>(f) / 30.f;
    frame.time = t;
    pool.run(static_cast<int>(all.size()), [&](int i) { all[static_cast<std::size_t>(i)]->step(); });
    blend_band(*all[0], *all[1], Side::right, 4);
    Module* sky[2] = {all[0], all[1]};
    transfer(*all[2], sky, 0.2f, 12, 0.f, 2.f);
    bus.clear();
    for (Module* m : all) bus.publish(*m);
    push(*all[2], bus, 0.2f);
    const ForceField ceiling{ForceField::Kind::ceiling, 0.f, 20.f, 0.f, 0.f, 10.f, 0.f, 0.f, 0.3f};
    apply(*all[0], ceiling);
    const ForceField swirl{ForceField::Kind::vortex, 60.f, 40.f, 20.f, 1.f};
    apply(*all[1], swirl);
    light.update(bus, 0.2f, {0.1f, 0.05f, 0.f}, pool);
    if (f % 5 == 0) parts.spawn(Kind::ember, 60.f, 70.f, 20.f, -80.f, 1.f, 1.f, 2.f);
    parts.update(1.f / 30.f, &bus, 1.f, frame.ground_y);
    pool.run(static_cast<int>(all.size()), [&](int i) { all[static_cast<std::size_t>(i)]->shade(&light); });
    const std::array<float, 4> scorch[1] = {{80.f, 82.f, 20.f, 0.5f}};
    frame.background(light, scorch, pool);
    frame.draw(all, pool);
    frame.particles(parts);
    frame.distort(shocks, bus, pool);
    frame.bloom(0.8f, 1.f, pool);
    frame.finish(rgb, pool);
  }
};

// A scene script that uses every statement, value shape, field and action (test_script.cpp, alloc_test.cpp), for the
// effect "tiny" (tiny_effect(3)) and "other" (tiny_effect(4)).
inline constexpr const char* kEverything = R"(
scene size 160 x 90, fps 30, length 1.5 s, ground 80     # comments are dropped by the printer
bus at (-16, -16), size 192 x 128, cell 8
particles capacity 512, flow 1
keyframes 0.5, 1
effect tiny = "tiny", swirl 1.2, swirl_scale 7, contrast 1
effect other = "other"
look gas = shader, heat_scale 1.2, emission 2, tint (1, 0.9, 0.8)
look smoke = like gas, soot_density 3
let ring_x = -20 + 120 * t / length
let big = (1 + 2) * -3 - -(4 / 2) + 2 * (3 - 1)
module pair = tiny, tiles 2 x 1, band 4, size 32, width 64, at (80, 74), look gas to smoke by smooth(t - 0.5)
module sky = other, over pair, look smoke
module fire = tiny, size 32, at (66, 72), sink 0.1, feather 0.125, start 1, seed 99,
              intensity 0.3 + 0.2 * smooth(t / 0.5), opacity if(t > 2, 0.5, 1)
module spare = tiny, size 32, width 24, start 0, seed 5, waiting, controls (0.4, 0.5, 0.6)
field cap = ceiling, level 20, soft 10, damping 0.3, on pair, weight smooth(t), if t > 0.1
field swirl = vortex, at (60, 40), radius 20, strength 1, on pair, particles
field breeze = wind, velocity (0.5, 0), on fire, from 0.2, until 1
field gale = gust, velocity (1, 0), amount 0.6, scale 40, rate 0.7, seed 3, on pair, fire, particles
field punch = ring, at (ring_x, 50), direction (1, -0.2), radius 12, core 6, strength 2, on pair
field drain = attract, at (80, 40), radius 25, strength 1.5, swirl 0.5, on pair
field torch = heat, at (40, 60), radius 10, strength 0.05, on fire
field hose = cold, at (100, 50), radius 15, strength 0.2, steam 0.5, on pair, weight 1 - smooth(t)
emit sparks, along (10, 78) to (60, 78), from 0, until 0.5, count 4
emit embers, on fire, tries 2, chance 0.6, spread 10, if fire.active
emit flakes, in (0, 0), size 160 x 60, tries 4, chance 0.5, soot 0.01
light gain 0.2, flash (0.1, 0.05, 0)
frame exposure 1.1, fade smooth(t / 0.2), haze 1, bloom 1, bloom_threshold 0.8
camera x (rand - 0.5) * 2, y 0
at 0.1 as boom, repeat, at most 1:
  start pair, from 0, in (1, 0), seed 7
  shock at (80, 50), speed 600, decay 0.2, amp 3, width 8
  scorch at (80, 82), radius 20, glow 0.5 * (1 + exp(-(t - boom)))
  burst at (80, 60), radius 5, embers 20, debris 4, speed 100
when shock 1 reaches fire as hit:
  start fire, from 0, seed 12, at (66, 72)
when shock 1 reaches (150, 50):
  stop fire
at 0.7 as later: hand_over pair -> sky, seed 40; suppress sky, tiles (0, 0), cells (0, 0) to (4, 4)
when heat(40, 60) > 0.01 and not spare.active as lit:
  wake spare at (40, ground)
when ember lands where temp > 0.3 and abs(x - 80) < 200 and not near(spare, x, 10), at most 2:
  scorch at (x, ground + 2), radius 5
when debris lands, repeat:
  stop spare
every frame:
  transfer fire -> pair, sky, top 4, fraction 0.2, heat 0, soot 2
  push fire, gain 0.2, if t > 0.05
  stop spare, if t > 1.4
)";


}  // namespace nfx::compose::testing
