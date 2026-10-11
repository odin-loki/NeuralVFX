// Composed effects (src/compose, docs/COMPOSE.md): the couplings keep what they promise, and a scene renders the same
// on any number of threads.
#include "compose_scene.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <numeric>
#include <thread>

namespace {

using namespace nfx;
using namespace nfx::compose;
using nfx::compose::testing::MiniScene;
using nfx::compose::testing::tiny_effect;

std::size_t z(int v) { return static_cast<std::size_t>(v); }

TEST(Compose, PoolRunsEveryTaskOnce) {
  Pool pool(4);
  for (int round = 0; round < 50; ++round) {
    std::vector<std::atomic<int>> hits(257);
    pool.run(257, [&](int i) { hits[z(i)].fetch_add(1); });
    for (const auto& h : hits) ASSERT_EQ(h.load(), 1);
  }
  Pool one(1);
  int n = 0;
  one.run(10, [&](int) { ++n; });
  EXPECT_EQ(n, 10);
}

// Jobs posted from several threads at once (a frame's state and the last frame's picture): every task runs once, and
// a thread that waits for another's work runs pool tasks meanwhile.
TEST(Compose, PoolRunsJobsFromSeveralThreadsAtOnce) {
  for (const int threads : {1, 2, 4}) {
    Pool pool(threads);
    constexpr int kCallers = 3, kRounds = 40;
    std::vector<std::vector<std::atomic<int>>> hits(kCallers);
    for (auto& h : hits) h = std::vector<std::atomic<int>>(kRounds * 64);
    std::atomic<int> finished{0};
    std::vector<std::thread> callers;
    for (int c = 0; c < kCallers; ++c) {
      callers.emplace_back([&, c] {
        for (int r = 0; r < kRounds; ++r) {
          const int n = 1 + (r * 7 + c * 13) % 64;
          pool.run(n, [&](int i) {
            volatile float x = 0.f;  // a little work, so that jobs overlap
            for (int k = 0; k < 200 * (i % 5); ++k) x = x + 1.f;
            hits[z(c)][z(r * 64 + i)].fetch_add(1);
          });
        }
        finished.fetch_add(1);
        pool.wake();
      });
    }
    pool.help_until([&] { return finished.load() == kCallers; });
    for (auto& t : callers) t.join();
    for (int c = 0; c < kCallers; ++c) {
      for (int r = 0; r < kRounds; ++r) {
        const int n = 1 + (r * 7 + c * 13) % 64;
        for (int i = 0; i < 64; ++i) ASSERT_EQ(hits[z(c)][z(r * 64 + i)].load(), i < n ? 1 : 0) << threads << " threads, caller " << c << " round " << r;
      }
    }
  }
}

TEST(Compose, TilesOfAGroupShareOwnershipEverywhere) {
  const rt::RolloutEffect e = tiny_effect();
  // a 2 x 2 tiling: 32-pixel tiles (16 cells) at scale 2, overlapping by a band of 4 cells (16 world pixels)
  std::vector<std::unique_ptr<Module>> t;
  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 2; ++c) {
      t.push_back(std::make_unique<Module>("t", e, 32, Placement{48.f * static_cast<float>(c), -48.f * static_cast<float>(r), 2.f}, Isa::base));
      t.back()->band = {c == 1 ? 4 : 0, c == 0 ? 4 : 0, r == 1 ? 4 : 0, r == 0 ? 4 : 0};
    }
  }
  for (float wy = -47.f; wy < 63.f; wy += 3.7f) {
    for (float wx = 1.f; wx < 111.f; wx += 3.3f) {
      float sum = 0.f;
      for (const auto& m : t) {
        const float tx = (wx - m->at.x) / m->at.scale, ty = 32.f - (wy - m->at.y) / m->at.scale;
        if (tx >= 0.f && ty >= 0.f && tx <= 32.f && ty <= 32.f) sum += m->weight_px(tx, ty);
      }
      ASSERT_NEAR(sum, 1.f, 1e-5f) << "at " << wx << ", " << wy;
    }
  }
}

TEST(Compose, BlendBandMakesNeighboursAgree) {
  const rt::RolloutEffect e = tiny_effect();
  for (const Side side : {Side::right, Side::top}) {
    Module a("a", e, 32, {}, Isa::base), b("b", e, 32, {}, Isa::base);
    a.start(0, 1);
    b.start(1, 2);
    for (int i = 0; i < 3; ++i) {
      a.step();
      b.step();
    }
    blend_band(a, b, side, 4);
    const int R = 16, Ca = a.channels(), S = 32;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < R; ++i) {
        const std::size_t ia = side == Side::top ? z((R - 4 + j) * R + i) : z(i * R + R - 4 + j);
        const std::size_t ib = side == Side::top ? z(j * R + i) : z(i * R + j);
        for (int c = 0; c < rollout::kPhys; ++c) ASSERT_EQ(a.runner().coarse()[ia * z(Ca) + z(c)], b.runner().coarse()[ib * z(Ca) + z(c)]);
      }
    }
    for (int j = 0; j < 8; ++j) {
      for (int i = 0; i < S; ++i) {
        const std::size_t fa = side == Side::top ? z((S - 8 + j) * S + i) : z(i * S + S - 8 + j);
        const std::size_t fb = side == Side::top ? z(j * S + i) : z(i * S + j);
        ASSERT_EQ(a.runner().fine_heat()[fa], b.runner().fine_heat()[fb]);
        ASSERT_EQ(a.runner().fine_soot()[fa], b.runner().fine_soot()[fb]);
      }
    }
  }
}

TEST(Compose, HandOverCarriesThePhysicalState) {
  const rt::RolloutEffect e1 = tiny_effect(3), e2 = tiny_effect(4);
  Module a("a", e1, 32, {}, Isa::base), b("b", e2, 32, {}, Isa::base);
  a.start(1, 5);
  for (int i = 0; i < 4; ++i) a.step();
  b.take_over(a);
  const int R = 16, C = a.channels();
  for (int i = 0; i < R * R; ++i) {
    for (int c = 0; c < C; ++c) {
      const float want = c < rollout::kPhys ? a.runner().coarse()[z(i) * z(C) + z(c)] : 0.f;  // memory starts from rest
      ASSERT_EQ(b.runner().coarse()[z(i) * z(C) + z(c)], want);
    }
  }
  EXPECT_TRUE(std::ranges::equal(a.runner().fine_heat(), b.runner().fine_heat()));
  EXPECT_TRUE(std::ranges::equal(a.runner().fine_soot(), b.runner().fine_soot()));
  EXPECT_EQ(a.runner().time(), b.runner().time());
  EXPECT_TRUE(b.active);
}

TEST(Compose, BusCarriesVelocityInWorldPixels) {
  const rt::RolloutEffect e = tiny_effect();
  Module m("m", e, 32, Placement{0.f, 0.f, 2.f}, Isa::base);  // 64 world pixels, 4 world pixels per coarse cell... x2
  m.start_empty(0.f, 1);
  m.group = 0;
  auto co = m.runner().coarse_mut();
  for (int i = 0; i < 16 * 16; ++i) {
    co[z(i) * z(m.channels())] = 0.5f;       // cells per frame, x right
    co[z(i) * z(m.channels()) + 1] = 0.25f;  // cells per frame, y up
    co[z(i) * z(m.channels()) + 2] = 0.7f;
  }
  FieldBus bus(0.f, 0.f, 8, 8, 8.f, 2);
  bus.publish(m);
  const auto s = bus.at(32.f, 32.f);
  const float cell = 64.f / 16.f;  // world pixels per coarse cell
  EXPECT_NEAR(s.u, 0.5f * cell, 1e-4f);
  EXPECT_NEAR(s.v, -0.25f * cell, 1e-4f);  // world y points down
  EXPECT_NEAR(s.heat, 0.7f, 1e-5f);
  const auto o = bus.others(32.f, 32.f, 0);  // without its own group, nothing is left
  EXPECT_NEAR(o.u, 0.f, 1e-5f);
  EXPECT_NEAR(o.heat, 0.f, 1e-5f);
}

// A module that glows (docs/EFFECTS.md: the magic portal) lights the scene in its own colour; its heat stays on the bus.
TEST(Compose, GlowingModuleLightsInItsOwnColour) {
  const rt::RolloutEffect e = tiny_effect();
  const auto light_of = [&](bool glows) {
    Module m("m", e, 32, Placement{0.f, 0.f, 2.f}, Isa::base);
    m.start_empty(0.f, 1);
    m.group = 0;
    m.glows = glows;
    m.glow = {0.3f, 0.1f, 0.9f};
    auto co = m.runner().coarse_mut();
    for (int i = 0; i < 16 * 16; ++i) co[z(i) * z(m.channels()) + 2] = 0.7f;
    FieldBus bus(0.f, 0.f, 8, 8, 8.f, 2);
    bus.publish(m);
    EXPECT_EQ(bus.glowing(), glows);
    EXPECT_NEAR(bus.at(32.f, 32.f).heat, 0.7f, 1e-5f);  // on the bus either way
    Light light(bus);
    Pool pool(1);
    light.update(bus, 0.14f, {0.f, 0.f, 0.f}, pool);
    const auto at = light.at(32.f, 32.f);
    bus.clear();
    EXPECT_FALSE(bus.glowing());
    EXPECT_TRUE(std::ranges::all_of(bus.glow(), [](float v) { return v == 0.f; }));
    return at;
  };
  const auto fire = light_of(false), magic = light_of(true);
  EXPECT_GT(fire[0], fire[2]);    // fire light: red over blue
  EXPECT_GT(magic[2], magic[0]);  // its own: blue over red
  EXPECT_GT(magic[2], 2.f * magic[1]);
  // the same amount of light: heat squared times the gain, in the module's colour against the fire ramp's
  EXPECT_NEAR(magic[0] / magic[2], 0.3f / 0.9f, 1e-3f);
}

TEST(Compose, PushMovesForOneStepOnly) {
  const rt::RolloutEffect e = tiny_effect();
  Module src("src", e, 32, Placement{0.f, 0.f, 2.f}, Isa::base), m("m", e, 32, Placement{0.f, 0.f, 2.f}, Isa::base);
  src.start_empty(0.f, 1);
  src.group = 0;
  m.start(0, 2);
  m.group = 1;
  for (int i = 0; i < 16 * 16; ++i) src.runner().coarse_mut()[z(i) * z(src.channels())] = 1.f;
  FieldBus bus(0.f, 0.f, 8, 8, 8.f, 2);
  bus.publish(src);
  const std::vector<float> before(m.runner().coarse().begin(), m.runner().coarse().end());
  push(m, bus, 0.5f);
  const int C = m.channels();
  EXPECT_NEAR(m.runner().coarse()[z(5 * 16 + 5) * z(C)] - before[z(5 * 16 + 5) * z(C)], 0.5f, 1e-4f);  // half the others' flow
  m.step();
  for (const float p : m.pushed()) ASSERT_EQ(p, 0.f);
  // without a push the same step gives the same velocity up to what the push moved
  Module ref("ref", e, 32, Placement{0.f, 0.f, 2.f}, Isa::base);
  ref.start(0, 2);
  ref.step();
  double diff = 0, mag = 0;
  for (int i = 0; i < 16 * 16; ++i) {
    diff += std::fabs(m.runner().coarse()[z(i) * z(C)] - ref.runner().coarse()[z(i) * z(C)]);
    mag += 0.5;
  }
  EXPECT_LT(diff / mag, 0.5) << "the push should not stay in the velocity";
}

TEST(Compose, TransferConservesMaterial) {
  const rt::RolloutEffect e = tiny_effect();
  Module big("big", e, 32, Placement{0.f, 0.f, 4.f}, Isa::base), small("small", e, 32, Placement{40.f, 30.f, 1.f}, Isa::base);
  big.start_empty(0.f, 1);
  small.start(1, 2);
  const auto amount = [](const Module& m, int c) {
    const float cell = m.cell_px();
    double s = 0;
    for (std::size_t i = 0; i < m.runner().coarse().size(); i += z(m.channels())) s += m.runner().coarse()[i + z(c)];
    return s * static_cast<double>(cell) * static_cast<double>(cell);
  };
  const double h0 = amount(big, 2) + amount(small, 2), d0 = amount(big, 3) + amount(small, 3);
  Module* to[1] = {&big};
  transfer(small, to, 1.f);
  EXPECT_NEAR(amount(small, 2), 0.0, 1e-6 * h0);
  EXPECT_NEAR(amount(big, 2) + amount(small, 2), h0, 1e-4 * h0);
  EXPECT_NEAR(amount(big, 3) + amount(small, 3), d0, 1e-4 * d0);
}

TEST(Compose, SuppressClearsItsRegion) {
  const rt::RolloutEffect e = tiny_effect();
  Module m("m", e, 32, {}, Isa::base);
  m.start(1, 3);
  suppress(m, 2, 0, 10, 5);
  const int C = m.channels();
  for (int y = 0; y < 5; ++y) {
    for (int x = 2; x < 10; ++x) {
      ASSERT_EQ(m.runner().coarse()[z(y * 16 + x) * z(C) + 2], 0.f);
      ASSERT_EQ(m.runner().coarse()[z(y * 16 + x) * z(C) + 3], 0.f);
    }
  }
  for (int y = 0; y < 10; ++y) {
    for (int x = 4; x < 20; ++x) ASSERT_EQ(m.runner().fine_soot()[z(y * 32 + x)], 0.f);
  }
}

TEST(Compose, CeilingStopsRisingAboveIt) {
  const rt::RolloutEffect e = tiny_effect();
  Module m("m", e, 32, Placement{0.f, 0.f, 2.f}, Isa::base);  // world y 0 (top) to 64 (bottom)
  m.start_empty(0.f, 1);
  const int C = m.channels();
  for (int i = 0; i < 256; ++i) m.runner().coarse_mut()[z(i) * z(C) + 1] = 1.f;
  const ForceField ceiling{ForceField::Kind::ceiling, 0.f, 32.f, 0.f, 0.f, 4.f, 0.f, 0.f, 1.f};
  apply(m, ceiling);
  EXPECT_NEAR(m.runner().coarse()[z(15 * 16 + 3) * z(C) + 1], 0.f, 1e-6f);  // top row: above the ceiling
  EXPECT_NEAR(m.runner().coarse()[z(0 * 16 + 3) * z(C) + 1], 1.f, 1e-6f);   // bottom row: below it
}

TEST(Compose, ShaderLeavesEmptyPixelsTransparent) {
  const rt::RolloutEffect e = tiny_effect();
  Module m("m", e, 32, {}, Isa::base);
  m.start_empty(0.f, 1);
  m.look = Look::shader;
  m.shade(nullptr);
  for (const float v : m.image().px) ASSERT_EQ(v, 0.f);
  const auto c = heat_colour(1.f);
  EXPECT_GT(c[0], c[2]);  // hot gas is warm
}

TEST(Compose, HotEmbersLandAndAreReported) {
  Particles p(8);
  ASSERT_TRUE(p.spawn(Kind::ember, 10.f, 0.f, 0.f, 200.f, 1.f, 1.f, 5.f));
  bool landed = false;
  for (int i = 0; i < 60 && !landed; ++i) {
    p.update(1.f / 30.f, nullptr, 0.f, 50.f);
    landed = !p.landings().empty();
  }
  EXPECT_TRUE(landed);
  for (int i = 0; i < 8; ++i) p.spawn(Kind::flake, 0, 0, 0, 0, 0, 1, 1);
  EXPECT_EQ(p.alive(), 8);  // capacity holds; extra spawns are dropped
}

TEST(Compose, ScenesRenderTheSameOnAnyNumberOfThreads) {
  MiniScene one(1), four(4);
  std::vector<std::uint8_t> a(160 * 90 * 3), b(a.size());
  for (int f = 0; f < 12; ++f) {
    one.step(f, a);
    four.step(f, b);
  }
  EXPECT_EQ(a, b);
  long sum = std::accumulate(a.begin(), a.end(), 0L);
  EXPECT_GT(sum, 0L);
}

// The picture captured and rendered in one go, at once or on the picture thread while the next frame's state is
// computed, is the stage-by-stage picture to the bit, on any number of threads.
TEST(Compose, CapturedAndOverlappedPicturesAreTheStagesPictures) {
  constexpr int kFrames = 14;
  std::vector<std::vector<std::uint8_t>> ref(kFrames, std::vector<std::uint8_t>(160 * 90 * 3));
  {
    MiniScene s(1);
    for (int f = 0; f < kFrames; ++f) s.step(f, ref[z(f)]);
  }
  for (const int threads : {1, 3}) {
    MiniScene s(threads);
    std::vector<std::uint8_t> rgb(ref[0].size());
    for (int f = 0; f < kFrames; ++f) {
      s.step_captured(f, rgb);
      ASSERT_EQ(rgb, ref[z(f)]) << "captured, " << threads << " threads, frame " << f;
    }
  }
  for (const int threads : {1, 2, 3}) {  // the scene's pool, plus the picture thread
    MiniScene s(threads);
    PictureThread picture(s.frame, s.pool);
    std::array<std::vector<std::uint8_t>, 2> rgb{ref[0], ref[0]};
    s.state(0);
    s.shade();
    for (int f = 0; f < kFrames; ++f) {
      s.capture();
      picture.start(rgb[z(f % 2)], 0.8f, 1.f);
      s.state(f + 1);
      picture.wait_images();
      s.shade();
      picture.wait();
      ASSERT_EQ(rgb[z(f % 2)], ref[z(f)]) << "overlapped, " << threads << " threads, frame " << f;
    }
  }
}

// The picture's row kernels for the baseline ISA and for AVX2 (without FMA) give the same bits.
TEST(Compose, RowKernelsGiveTheSameBitsOnEveryIsa) {
#if defined(__x86_64__)
  if (!__builtin_cpu_supports("avx2")) GTEST_SKIP() << "no AVX2";
#endif
  MiniScene a(2), b(2);
  a.frame.use_avx2(false);
  b.frame.use_avx2(true);
  std::vector<std::uint8_t> ra(160 * 90 * 3), rb(ra.size());
  for (int f = 0; f < 14; ++f) {
    a.step_captured(f, ra);
    b.step_captured(f, rb);
    ASSERT_EQ(ra, rb) << "frame " << f;
  }
}

}  // namespace

// --- study H (H2): skipping the empty parts of the fine fields changes nothing ----------------------------------------

namespace {

// A rollout effect whose material sits in a small blob of an otherwise empty grid, with a strong wind and swirl, so
// that material moves several pixels per step and most of every row is empty.
rt::RolloutEffect sparse_effect(int res, std::uint64_t seed) {
  rollout::Hyper h;
  h.res = res;
  h.hidden = 8;
  h.memory = 2;
  h.jacobi = 10;
  h.render_hidden = 6;
  h.warmup = 2;
  h.start_fine = 2 * res;
  rt::RolloutEffect e;
  e.m = rollout::init_model(h, seed);
  // no learned change to the state: the blob is only carried by the wind and swirl (and the lock keeps it together)
  std::ranges::fill(e.m.step_w, 0.f);
  e.m.effect = "sparse";
  e.m.control_names = {"intensity", "wind", "turbulence"};
  e.m.scale = {0.2f, 0.2f, 0.4f, 0.3f};
  e.m.lo = {-3.f, -3.f, 0.f, 0.f};
  e.m.hi = {3.f, 3.f, 3.f, 3.f};
  e.m.detail.swirl_control = 2;
  e.m.detail.swirl = 1.5f;
  for (int k = 0; k < 2; ++k) {
    rollout::StartPoint sp;
    sp.controls = {0.6f, 0.5f, 0.9f};
    sp.seed = 20 + static_cast<std::uint64_t>(k);
    sp.coarse.assign(static_cast<std::size_t>(res) * static_cast<std::size_t>(res) * rollout::kPhys, 0.f);
    for (int y = 0; y < res; ++y) {
      for (int x = 0; x < res; ++x) {
        float* c = sp.coarse.data() + (static_cast<std::size_t>(y) * static_cast<std::size_t>(res) + static_cast<std::size_t>(x)) * rollout::kPhys;
        c[0] = 1.2f + 0.3f * std::sin(0.5f * static_cast<float>(y));  // wind to the right, about a cell per step
        c[1] = 0.6f * std::cos(0.4f * static_cast<float>(x + k));
        const float d = std::hypot(static_cast<float>(x - 3 - k), static_cast<float>(y - res / 2));
        c[2] = std::max(0.f, 1.5f - 0.6f * d);
        c[3] = std::max(0.f, 0.9f - 0.5f * d);
      }
    }
    if (k == 1) {  // a start with fine fields: a blob too
      const int F = h.start_fine;
      for (int i = 0; i < F * F; ++i) {
        const float d = std::hypot(static_cast<float>(i % F - F / 4), static_cast<float>(i / F - F / 2));
        sp.fine_t.push_back(std::max(0.f, 0.8f - 0.2f * d));
        sp.fine_d.push_back(std::max(0.f, 0.5f - 0.15f * d));
      }
    }
    e.m.starts.push_back(sp);
  }
  rollout::quantise_like_storage(e.m);
  return e;
}

bool same_bits(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

TEST(Compose, SkippingEmptyFieldsIsBitExact) {
  std::vector<Isa> isas{Isa::base};
#if defined(__x86_64__)
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) isas.push_back(Isa::avx2);
  if (__builtin_cpu_supports("avx512f")) isas.push_back(Isa::avx512);
#endif
  int skipped_rows = 0;
  for (const auto& [res, size] : {std::pair{12, 36}, std::pair{12, 60}, std::pair{16, 64}, std::pair{16, 16}}) {
    const rt::RolloutEffect e = sparse_effect(res, 5);
    const rt::RolloutEffect dense = tiny_effect();  // fields that fill the tile: nothing to skip
    for (const rt::RolloutEffect* fx : {&e, &dense}) {
      if (fx == &dense && size % 16 != 0) continue;
      for (const Isa isa : isas) {
        for (const int start : {0, 1}) {
          auto a = make_runner(*fx, size, isa), b = make_runner(*fx, size, isa);
          b->skip_empty(false);
          const auto& controls = fx->m.starts[z(start)].controls;
          a->start(start, controls, 7);
          b->start(start, controls, 7);
          std::vector<std::uint8_t> ra(z(size) * z(size) * 4), rb(ra.size());
          for (int f = 0; f < 24; ++f) {
            if (f == 9 || f == 17) {  // written from outside between steps, as couplings do: a band cleared, a spot added
              for (rt::RolloutRunner* r : {a.get(), b.get()}) {
                auto ft = r->fine_heat_mut();
                auto fd = r->fine_soot_mut();
                for (int y = 0; y < size; ++y) {
                  for (int x = 0; x < size; ++x) {
                    const std::size_t i = z(y) * z(size) + z(x);
                    if (y < size / 3) ft[i] = fd[i] = 0.f;
                    if (std::abs(x - size + 4) < 3 && std::abs(y - 5) < 3) ft[i] += 0.7f;
                  }
                }
              }
            }
            a->step(controls, 7);
            b->step(controls, 7);
            ASSERT_TRUE(same_bits(a->fine_heat(), b->fine_heat())) << isa_name(isa) << " size " << size << " start " << start << " frame " << f;
            ASSERT_TRUE(same_bits(a->fine_soot(), b->fine_soot())) << isa_name(isa) << " size " << size << " start " << start << " frame " << f;
            ASSERT_TRUE(same_bits(a->coarse(), b->coarse()));
            a->render(rt::FrameInput{}, ra.data(), z(size) * 4);
            b->render(rt::FrameInput{}, rb.data(), z(size) * 4);
            ASSERT_EQ(ra, rb) << isa_name(isa) << " size " << size << " start " << start << " frame " << f;
            if (fx == &e) {  // the sparse effect really has empty rows to skip
              for (int y = 0; y < size; ++y) {
                bool any = false;
                for (int x = 0; x < size; ++x) any |= a->fine_heat()[z(y) * z(size) + z(x)] != 0.f;
                skipped_rows += any ? 0 : 1;
              }
            }
          }
        }
      }
    }
  }
  EXPECT_GT(skipped_rows, 100);
}

// Runners that share step()'s working memory (rt::RolloutScratch) give the bits of runners with their own, whatever
// the scratch holds when a step begins: the other runners' leftovers (other effects, grids and sizes, stepped in turn,
// with and without skipping) or garbage written over it before every step.
TEST(Compose, SharedStepScratchIsBitExact) {
  std::vector<Isa> isas{Isa::base};
#if defined(__x86_64__)
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) isas.push_back(Isa::avx2);
  if (__builtin_cpu_supports("avx512f")) isas.push_back(Isa::avx512);
#endif
  const rt::RolloutEffect sparse12 = sparse_effect(12, 5), dense = tiny_effect(), other = tiny_effect(4);
  struct Use {
    const rt::RolloutEffect* e;
    int size, start;
    bool skip;
  };
  const Use uses[] = {{&sparse12, 60, 0, true}, {&dense, 32, 1, true}, {&sparse12, 36, 1, false}, {&other, 64, 0, true}, {&dense, 16, 0, true}};
  for (const Isa isa : isas) {
    std::vector<std::unique_ptr<rt::RolloutRunner>> own, shared;
    rt::RolloutScratch scratch;
    for (const Use& u : uses) {
      own.push_back(make_runner(*u.e, u.size, isa));
      shared.push_back(make_runner(*u.e, u.size, isa));
      own.back()->skip_empty(u.skip);
      shared.back()->skip_empty(u.skip);
      scratch.fit(*shared.back());
    }
    rt::RolloutScratch small;  // not fit for any runner
    EXPECT_THROW(shared[0]->use_scratch(&small), std::invalid_argument);
    for (auto& r : shared) r->use_scratch(&scratch);
    EXPECT_LT(shared[0]->scratch_bytes(), own[0]->scratch_bytes());
    std::uint64_t rng = 12345;
    for (std::size_t k = 0; k < own.size(); ++k) {
      const auto& controls = uses[k].e->m.starts[z(uses[k].start)].controls;
      own[k]->start(uses[k].start, controls, 7 + k);
      shared[k]->start(uses[k].start, controls, 7 + k);
    }
    for (int f = 0; f < 16; ++f) {
      for (std::size_t k = 0; k < own.size(); ++k) {
        const auto& controls = uses[k].e->m.starts[z(uses[k].start)].controls;
        if (f % 4 == 1) {  // garbage over the scratch: NaN, huge values of either sign, or noise; small integers
          const int kind = (f / 4 + static_cast<int>(k)) % 4;
          for (float& v : scratch.floats) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            const float noise = static_cast<float>(static_cast<std::int32_t>(rng >> 32)) * 1e-6f;
            v = kind == 0 ? std::nanf("") : kind == 1 ? 3e38f : kind == 2 ? -3e38f : noise;
          }
          for (std::int32_t& v : scratch.ints) v = static_cast<std::int32_t>(f % 3);
        }
        own[k]->step(controls, 7 + k);
        shared[k]->step(controls, 7 + k);
        const int S = uses[k].size;
        ASSERT_TRUE(same_bits(own[k]->coarse(), shared[k]->coarse())) << isa_name(isa) << " runner " << k << " frame " << f;
        ASSERT_TRUE(same_bits(own[k]->fine_heat(), shared[k]->fine_heat())) << isa_name(isa) << " runner " << k << " frame " << f;
        ASSERT_TRUE(same_bits(own[k]->fine_soot(), shared[k]->fine_soot())) << isa_name(isa) << " runner " << k << " frame " << f;
        std::vector<std::uint8_t> a(z(S) * z(S) * 4), b(a.size());
        own[k]->render(rt::FrameInput{}, a.data(), z(S) * 4);
        shared[k]->render(rt::FrameInput{}, b.data(), z(S) * 4);
        ASSERT_EQ(a, b) << isa_name(isa) << " runner " << k << " frame " << f;
      }
    }
    for (auto& r : shared) r->use_scratch(nullptr);  // back to their own
    EXPECT_EQ(shared[0]->scratch_bytes(), own[0]->scratch_bytes());
  }
}

// A scene whose modules share one step scratch per thread renders the frames of one whose modules have their own, on
// any number of threads.
TEST(Compose, ScenesWithSharedStepScratchRenderTheSame) {
  MiniScene ref(1), one(1, true), four(4, true);
  EXPECT_EQ(four.shared->slots(), 4);
  std::vector<std::uint8_t> a(160 * 90 * 3), b(a.size()), c(a.size());
  for (int f = 0; f < 12; ++f) {
    ref.step(f, a);
    one.step(f, b);
    four.step(f, c);
    ASSERT_EQ(a, b) << "frame " << f;
    ASSERT_EQ(a, c) << "frame " << f;
  }
  std::size_t own = 0, shared = one.shared->bytes();  // three modules: three sets of working memory, or one
  for (const Module* m : ref.all) own += m->runner().scratch_bytes();
  for (const Module* m : one.all) shared += m->runner().scratch_bytes();
  EXPECT_LT(shared, own);
}
