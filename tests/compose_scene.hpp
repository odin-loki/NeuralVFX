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

}  // namespace nfx::compose::testing
