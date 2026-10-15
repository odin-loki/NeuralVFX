// Rollout effects: the noise matches the simulation's forcing, the format round-trips, the trainer's forward pass is
// the reference's, its gradients (through time, burn-in and the profile loss) match finite differences, and the
// reference rollout stays finite.
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <random>
#include <sstream>

using namespace nfx;
using namespace nfx::rollout;

namespace {

Hyper tiny_hyper() {
  Hyper h;
  h.res = 8;
  h.hidden = 6;
  h.memory = 2;
  h.jacobi = 7;
  h.render_hidden = 5;
  h.warmup = 3;
  return h;
}

// A synthetic run on the tiny grid: smooth fields that move, so every operation is exercised.
Run tiny_run(const Hyper& h, int frames) {
  Run r;
  r.p.seed = 7;
  r.p.intensity = 0.3f;
  r.p.wind = 0.6f;
  r.p.turbulence = 0.8f;
  r.frames = frames;
  const int R = h.res;
  r.coarse.resize(static_cast<std::size_t>(frames) * R * R * kPhys);
  for (int f = 0; f < frames; ++f) {
    for (int y = 0; y < R; ++y) {
      for (int x = 0; x < R; ++x) {
        float* c = r.coarse.data() + ((static_cast<std::size_t>(f) * R + y) * R + x) * kPhys;
        const float t = 0.3f * static_cast<float>(f);
        // a steady drift keeps backtraces inside cells (bilinear interpolation has kinks at whole-cell positions)
        c[0] = 0.4f + 0.08f * std::sin(0.7f * static_cast<float>(x) + t);
        c[1] = 0.45f + 0.08f * std::cos(0.6f * static_cast<float>(y) - 0.5f * t);
        c[2] = 0.6f + 0.5f * std::sin(0.9f * static_cast<float>(x + y) + t);
        c[3] = 0.3f + 0.25f * std::cos(0.8f * static_cast<float>(x) - 0.4f * static_cast<float>(y) + t);
      }
    }
  }
  return r;
}

Model tiny_model(std::uint64_t seed = 3) {
  Model m = init_model(tiny_hyper(), seed);
  // Larger output weights than a fresh model, so the stepper's contribution is not negligible in the checks.
  std::mt19937_64 rng(seed + 1);
  std::normal_distribution<float> nd(0.f, 0.15f);
  for (float& w : m.step_w) w += nd(rng);
  const StepLayout L = step_layout(m.h);  // a modest output layer: velocity changes stay well below a cell
  for (std::size_t i = L.wo; i < L.bo + static_cast<std::size_t>(m.h.outputs()); ++i) m.step_w[i] *= 0.3f;
  m.scale = {0.4f, 0.4f, 0.5f, 0.3f};
  m.lo = {-5.f, -5.f, 0.f, 0.f};
  m.hi = {5.f, 5.f, 5.f, 5.f};
  m.effect = "tiny";
  m.control_names = {"intensity", "wind", "turbulence"};
  return m;
}

}  // namespace

TEST(Rollout, NoiseIsTheSimulationsForcing) {
  sim::Params p;
  p.seed = 4242;
  NoiseSpec n;
  for (const auto e : sim::kAllEffects) {
    p.effect = e;
    n.flicker_rate = sim::flicker_rate(e);
    const Model m = recipe_model(recipe_for(e));  // what the recipe gives the model
    EXPECT_EQ(m.noise.flicker_rate, n.flicker_rate);
    const auto names = sim::control_names(e);
    EXPECT_EQ(m.control_names, std::vector<std::string>(names.begin(), names.end()));
    for (int i = 0; i < 50; ++i) {
      const float X = 2.7f * static_cast<float>(i), Y = 1.9f * static_cast<float>(i % 13), t = 0.13f * static_cast<float>(i);
      // equal up to rounding (X * (1 / 14) against X / 14)
      EXPECT_NEAR(noise_curl(n, p.seed, X, Y, t), sim::curl_potential(p, X, Y, t), 1e-5f);
      EXPECT_NEAR(noise_flicker(n, p.seed, X, Y, t), sim::source_flicker(p, X, Y, t), 1e-5f);
    }
  }
}


// Effects that play once take the explosion's branch of study D's recipe; fog and cloud keep fine fields with their start
// points as smoke does (docs/EFFECTS_ATMOSPHERE.md).
TEST(Rollout, OneShotEffectsTakeTheExplosionsRecipe) {
  const SimRecipe ex = recipe_for(sim::Effect::explosion), dust = recipe_for(sim::Effect::dust);
  EXPECT_EQ(dust.runs, ex.runs);
  EXPECT_EQ(dust.frames, ex.frames);
  EXPECT_EQ(dust.start_frame, 0);
  EXPECT_EQ(dust.start_fine, ex.start_fine);
  EXPECT_EQ(dust.starts, ex.starts);
  EXPECT_EQ(dust.stepper.burn_max, ex.stepper.burn_max);
  const Model m = recipe_model(dust);
  EXPECT_FALSE(m.loop);
  EXPECT_EQ(m.h.frames, ex.frames);
  EXPECT_EQ(m.h.n_age, 2);
  for (const sim::Effect e : {sim::Effect::fog, sim::Effect::cloud}) {
    const SimRecipe r = recipe_for(e), fire = recipe_for(sim::Effect::fire);
    EXPECT_EQ(r.start_fine, recipe_for(sim::Effect::smoke).start_fine);
    EXPECT_EQ(r.runs, fire.runs);
    EXPECT_EQ(r.frames, fire.frames);
    EXPECT_EQ(r.starts, fire.starts);
    EXPECT_TRUE(recipe_model(r).loop);
  }
  // The mushroom (docs/EFFECTS_BLAZE.md) takes the same branch at its own length (5 s); wildfire and coals take fire's
  // recipe as it is.
  const SimRecipe mush = recipe_for(sim::Effect::mushroom);
  EXPECT_EQ(mush.runs, ex.runs);
  EXPECT_EQ(mush.frames, 150);
  EXPECT_EQ(mush.frames, sim::one_shot_frames(sim::Effect::mushroom));
  EXPECT_EQ(mush.start_frame, 0);
  EXPECT_EQ(mush.start_fine, ex.start_fine);
  EXPECT_EQ(mush.starts, ex.starts);
  EXPECT_EQ(mush.stepper.burn_max, ex.stepper.burn_max);
  EXPECT_EQ(ex.frames, 90);
  const Model mm = recipe_model(mush);
  EXPECT_FALSE(mm.loop);
  EXPECT_EQ(mm.h.frames, 150);
  EXPECT_EQ(mm.h.n_age, 2);
  for (const sim::Effect e : {sim::Effect::wildfire, sim::Effect::coals}) {
    const SimRecipe r = recipe_for(e), fire = recipe_for(sim::Effect::fire);
    EXPECT_EQ(r.start_fine, fire.start_fine);
    EXPECT_EQ(r.runs, fire.runs);
    EXPECT_EQ(r.frames, fire.frames);
    EXPECT_EQ(r.starts, fire.starts);
    EXPECT_TRUE(recipe_model(r).loop);
  }
}
// launch, puffs and ink keep fine fields with their start points as smoke does (docs/EFFECTS_FLOW.md); otherwise they
// take fire's recipe.
TEST(Rollout, FlowEffectsKeepFineFieldsAsSmokeDoes) {
  const SimRecipe fire = recipe_for(sim::Effect::fire), smoke = recipe_for(sim::Effect::smoke);
  for (const sim::Effect e : {sim::Effect::launch, sim::Effect::puffs, sim::Effect::ink}) {
    const SimRecipe r = recipe_for(e);
    EXPECT_EQ(r.start_fine, smoke.start_fine) << sim::effect_name(e);
    EXPECT_EQ(r.runs, fire.runs);
    EXPECT_EQ(r.frames, fire.frames);
    EXPECT_EQ(r.start_frame, fire.start_frame);
    EXPECT_EQ(r.starts, fire.starts);
    EXPECT_EQ(r.stepper.activity_stage, fire.stepper.activity_stage);
    const Model m = recipe_model(r);
    EXPECT_TRUE(m.loop);
    EXPECT_EQ(m.h.n_age, 0);
    EXPECT_EQ(m.noise.flicker_rate, sim::flicker_rate(e));
  }
  EXPECT_EQ(recipe_model(recipe_for(sim::Effect::puffs)).control_names[1], "rate");
  EXPECT_EQ(recipe_model(recipe_for(sim::Effect::ink)).control_names[1], "current");
}

// Lava and the eruption linger (a crust that crawls, an umbrella of ash), so they keep fine fields with their start
// points as smoke does; the avalanche plays once and takes the explosion's branch at its own length
// (docs/EFFECTS_EARTH.md).
TEST(Rollout, EarthEffectsKeepFineFieldsAndTheAvalanchePlaysOnce) {
  const SimRecipe fire = recipe_for(sim::Effect::fire), smoke = recipe_for(sim::Effect::smoke), ex = recipe_for(sim::Effect::explosion);
  for (const sim::Effect e : {sim::Effect::lava, sim::Effect::eruption}) {
    const SimRecipe r = recipe_for(e);
    EXPECT_EQ(r.start_fine, smoke.start_fine) << sim::effect_name(e);
    EXPECT_EQ(r.runs, fire.runs);
    EXPECT_EQ(r.frames, fire.frames);
    EXPECT_EQ(r.start_frame, fire.start_frame);
    EXPECT_EQ(r.starts, fire.starts);
    EXPECT_EQ(r.stepper.activity_stage, fire.stepper.activity_stage);
    const Model m = recipe_model(r);
    EXPECT_TRUE(m.loop);
    EXPECT_EQ(m.h.n_age, 0);
    EXPECT_EQ(m.noise.flicker_rate, sim::flicker_rate(e));
  }
  const SimRecipe av = recipe_for(sim::Effect::avalanche);
  EXPECT_EQ(av.runs, ex.runs);
  EXPECT_EQ(av.frames, sim::one_shot_frames(sim::Effect::avalanche));
  EXPECT_EQ(av.frames, 150);
  EXPECT_EQ(av.start_frame, 0);
  EXPECT_EQ(av.start_fine, ex.start_fine);
  EXPECT_EQ(av.starts, ex.starts);
  EXPECT_EQ(av.stepper.burn_max, ex.stepper.burn_max);
  const Model m = recipe_model(av);
  EXPECT_FALSE(m.loop);
  EXPECT_EQ(m.h.frames, 150);
  EXPECT_EQ(m.h.n_age, 2);
  EXPECT_EQ(recipe_model(recipe_for(sim::Effect::lava)).control_names[1], "flow");
  EXPECT_EQ(m.control_names[1], "direction");
  EXPECT_EQ(recipe_model(recipe_for(sim::Effect::eruption)).control_names[1], "wind");
}

TEST(Rollout, CoarseFromSimAveragesBlocks) {
  sim::State st;
  st.n = 16;
  const std::size_t nn = 256;
  st.u.assign(nn, 60.f);  // 60 cells per second
  st.v.assign(nn, -30.f);
  st.temp.resize(nn);
  st.soot.resize(nn);
  for (std::size_t i = 0; i < nn; ++i) {
    st.temp[i] = static_cast<float>(i % 16);  // x
    st.soot[i] = static_cast<float>(i / 16);  // y
  }
  std::vector<float> c(4 * 4 * kPhys);
  coarse_from_sim(st, 4, 30.f, c);
  EXPECT_FLOAT_EQ(c[0], 60.f / 4.f / 30.f);   // coarse cells per frame
  EXPECT_FLOAT_EQ(c[1], -30.f / 4.f / 30.f);
  EXPECT_FLOAT_EQ(c[2], 1.5f);                // mean of x = 0..3
  EXPECT_FLOAT_EQ(c[(4 * 2 + 1) * kPhys + 2], 5.5f);  // block (1, 2): x = 4..7
  EXPECT_FLOAT_EQ(c[(4 * 2 + 1) * kPhys + 3], 9.5f);  // y = 8..11
}

TEST(Rollout, SaveLoadRoundTrips) {
  Model m = tiny_model();
  m.h.start_fine = 4;
  const auto r = tiny_run(m.h, 4);
  for (int k = 0; k < 2; ++k) {
    StartPoint sp;
    sp.controls = r.controls();
    sp.seed = 99 + static_cast<std::uint64_t>(k);
    sp.time = 1.5f;
    sp.coarse.assign(r.coarse.begin(), r.coarse.begin() + 8 * 8 * kPhys);
    if (k == 1) {
      sp.fine_t.assign(16, 0.25f);
      sp.fine_d.assign(16, 0.75f);
      sp.fine_t[3] = 1.f;
    }
    m.starts.push_back(sp);
  }
  quantise_like_storage(m);
  std::stringstream ss;
  ASSERT_TRUE(save_model(ss, m));
  const std::string bytes = ss.str();
  EXPECT_TRUE(is_rollout_file(std::span(bytes.data(), bytes.size())));
  EXPECT_NEAR(static_cast<double>(bytes.size()), static_cast<double>(m.storage_bytes()), 0.2 * static_cast<double>(bytes.size()));
  auto back = load_model(ss);
  ASSERT_TRUE(back) << back.error();
  EXPECT_EQ(back->h.res, m.h.res);
  EXPECT_EQ(back->control_names, m.control_names);
  EXPECT_EQ(back->step_w, m.step_w);
  EXPECT_EQ(back->render_w, m.render_w);
  ASSERT_EQ(back->starts.size(), 2u);
  EXPECT_EQ(back->starts[0].coarse, m.starts[0].coarse);
  EXPECT_TRUE(back->starts[0].fine_t.empty());
  EXPECT_EQ(back->starts[1].fine_t, m.starts[1].fine_t);
  EXPECT_EQ(back->starts[1].seed, 100u);
  EXPECT_EQ(back->detail.swirl_control, m.detail.swirl_control);
  std::stringstream bad("NVFXMDL1 not this kind");
  EXPECT_FALSE(load_model(bad));
}

TEST(Rollout, QuantisedStartStatesRoundTrip) {
  // Version 3 files: coarse start states at 2 to 8 bits per channel plane, with and without the seed's dither. What
  // loads equals quantise_like_storage; the error stays within a step (half a step undithered); empty heat stays
  // empty; files without quantised starts keep the version 2 layout.
  Model m = tiny_model();
  const auto r = tiny_run(m.h, 4);
  for (int k = 0; k < 2; ++k) {
    StartPoint sp;
    sp.controls = r.controls();
    sp.seed = 7 + static_cast<std::uint64_t>(k);
    sp.time = 1.f;
    sp.coarse.assign(r.coarse.begin() + k * 8 * 8 * kPhys, r.coarse.begin() + (k + 1) * 8 * 8 * kPhys);
    for (int i = 0; i < 8; ++i) sp.coarse[static_cast<std::size_t>(i) * kPhys + 2] = 0.f;  // some empty heat
    m.starts.push_back(sp);
  }
  {
    std::stringstream ss;
    ASSERT_TRUE(save_model(ss, m));
    EXPECT_EQ(ss.str()[8], 2);  // version 2 layout when start states are fp16
  }
  for (const int bits : {8, 6, 4, 3}) {
    for (const bool dither : {false, true}) {
      Model q = m;
      q.start_bits = bits;
      q.start_dither = dither;
      std::stringstream ss;
      ASSERT_TRUE(save_model(ss, q));
      const std::string bytes = ss.str();
      EXPECT_EQ(bytes[8], 3);
      EXPECT_NEAR(static_cast<double>(bytes.size()), static_cast<double>(q.storage_bytes()), 0.2 * static_cast<double>(bytes.size()));
      auto back = load_model(ss);
      ASSERT_TRUE(back) << back.error();
      EXPECT_EQ(back->start_bits, bits);
      EXPECT_EQ(back->start_dither, dither);
      Model ref = q;
      quantise_like_storage(ref);
      for (std::size_t k = 0; k < 2; ++k) {
        ASSERT_EQ(back->starts[k].coarse, ref.starts[k].coarse) << bits << " " << dither;
        for (int c = 0; c < kPhys; ++c) {
          float mn = 1e9f, mx = -1e9f;
          for (std::size_t i = static_cast<std::size_t>(c); i < m.starts[k].coarse.size(); i += kPhys) {
            mn = std::min(mn, m.starts[k].coarse[i]);
            mx = std::max(mx, m.starts[k].coarse[i]);
          }
          const float step = (mx - mn) / static_cast<float>((1 << bits) - 1);
          for (std::size_t i = static_cast<std::size_t>(c); i < m.starts[k].coarse.size(); i += kPhys) {
            EXPECT_LE(std::abs(back->starts[k].coarse[i] - m.starts[k].coarse[i]), (dither ? 1.01f : 0.51f) * step + 2e-3f * std::abs(mx) + 1e-6f);
          }
        }
        for (int i = 0; i < 8; ++i) EXPECT_EQ(back->starts[k].coarse[static_cast<std::size_t>(i) * kPhys + 2], 0.f);
      }
    }
  }
}

TEST(Rollout, StudyD2ConstantsRoundTripInVersion4) {
  // Study D2: DetailSpec::advect and soften. At their defaults the file keeps its old layout (2, or 3
  // with quantised starts) byte for byte; otherwise it is version 4, with fp16 or quantised start states; out-of-range
  // values are refused.
  Model m = tiny_model();
  const auto r = tiny_run(m.h, 4);
  StartPoint sp;
  sp.controls = r.controls();
  sp.seed = 5;
  sp.time = 1.f;
  sp.coarse.assign(r.coarse.begin(), r.coarse.begin() + 8 * 8 * kPhys);
  m.starts.push_back(sp);
  quantise_like_storage(m);
  for (const int bits : {16, 6}) {
    Model q = m;
    q.start_bits = bits;
    q.start_dither = bits < 16;
    std::stringstream old_ss;
    ASSERT_TRUE(save_model(old_ss, q));
    EXPECT_EQ(old_ss.str()[8], bits < 16 ? 3 : 2);
    if (bits == 16) {  // a file without the new constants writes back to the same bytes (fp16 starts round-trip exactly)
      auto again = load_model(old_ss);
      ASSERT_TRUE(again) << again.error();
      std::stringstream again_ss;
      ASSERT_TRUE(save_model(again_ss, *again));
      EXPECT_EQ(again_ss.str(), old_ss.str());
    }
    q.detail.advect = 1.25f;
    q.detail.soften = 0.125f;
    std::stringstream ss;
    ASSERT_TRUE(save_model(ss, q));
    EXPECT_EQ(ss.str()[8], 4);
    auto back = load_model(ss);
    ASSERT_TRUE(back) << back.error();
    EXPECT_EQ(back->detail.advect, 1.25f);
    EXPECT_EQ(back->detail.soften, 0.125f);
    EXPECT_EQ(back->start_bits, bits);
    EXPECT_EQ(back->start_dither, bits < 16);
    Model ref = q;
    quantise_like_storage(ref);
    EXPECT_EQ(back->starts[0].coarse, ref.starts[0].coarse);
    EXPECT_EQ(back->step_w, q.step_w);
  }
  for (const auto& [advect, soften] : {std::pair{0.f, 0.f}, {1.f, 1.5f}, {1.f, -0.1f}}) {
    Model bad = m;
    bad.detail.advect = advect;
    bad.detail.soften = soften;
    std::stringstream ss;
    EXPECT_FALSE(save_model(ss, bad));
  }
  Model soft = m;  // soften alone also needs version 4
  soft.detail.soften = 0.25f;
  std::stringstream ss;
  ASSERT_TRUE(save_model(ss, soft));
  EXPECT_EQ(ss.str()[8], 4);
}

TEST(Rollout, StudyLAdvectionRoundTripsInVersion5) {
  // Study L: Hyper::advection. At 0 the file keeps its old layout; otherwise it is version 5 (with the version 4 fields),
  // and other values are refused.
  Model m = tiny_model();
  const auto r = tiny_run(m.h, 4);
  StartPoint sp;
  sp.controls = r.controls();
  sp.seed = 5;
  sp.time = 1.f;
  sp.coarse.assign(r.coarse.begin(), r.coarse.begin() + 8 * 8 * kPhys);
  m.starts.push_back(sp);
  quantise_like_storage(m);
  std::stringstream old_ss;
  ASSERT_TRUE(save_model(old_ss, m));
  EXPECT_EQ(old_ss.str()[8], 2);
  for (const int bits : {16, 6}) {
    Model q = m;
    q.h.advection = 1;
    q.start_bits = bits;
    q.start_dither = bits < 16;
    q.detail.soften = bits < 16 ? 0.25f : 0.f;
    std::stringstream ss;
    ASSERT_TRUE(save_model(ss, q));
    EXPECT_EQ(ss.str()[8], 5);
    auto back = load_model(ss);
    ASSERT_TRUE(back) << back.error();
    EXPECT_EQ(back->h.advection, 1);
    EXPECT_EQ(back->detail.soften, q.detail.soften);
    EXPECT_EQ(back->start_bits, bits);
    EXPECT_EQ(back->step_w, q.step_w);
    std::stringstream again;
    ASSERT_TRUE(save_model(again, *back));
    if (bits == 16) {
      EXPECT_EQ(again.str(), ss.str());
    }
  }
  Model bad = m;
  bad.h.advection = 2;
  std::stringstream ss;
  EXPECT_FALSE(save_model(ss, bad));
  Model passes = m;  // DetailSpec::passes alone also needs version 5
  passes.detail.passes = 3;
  std::stringstream ps;
  ASSERT_TRUE(save_model(ps, passes));
  EXPECT_EQ(ps.str()[8], 5);
  auto pb = load_model(ps);
  ASSERT_TRUE(pb) << pb.error();
  EXPECT_EQ(pb->detail.passes, 3);
  EXPECT_EQ(pb->h.advection, 0);
  for (const int bad_passes : {0, 9}) {
    Model q = m;
    q.detail.passes = bad_passes;
    std::stringstream qs;
    EXPECT_FALSE(save_model(qs, q));
  }
}

TEST(Rollout, DetailPassesSplitTheFramesFlow) {
  // Study L: with passes = 2 the detail layer moves the fields by half the flow and locks them, twice. A blob carried by
  // two pixels a frame towards a coarse state that holds it two pixels on ends there in one pass or in two, by another
  // path (the two results differ).
  Model m = tiny_model();
  m.detail.contrast = 0.f;
  m.detail.swirl = 0.f;
  m.detail.passes = 2;
  State s;
  s.res = 8;
  s.size = 32;
  const std::size_t C = static_cast<std::size_t>(m.h.channels());
  s.coarse.assign(8 * 8 * C, 0.f);
  s.flow.assign(8 * 8 * 2, 0.f);
  for (std::size_t i = 0; i < 64; ++i) s.flow[i * 2] = 0.5f;  // half a coarse cell a frame: 2 pixels
  const auto blob = [](float cx) {
    std::vector<float> f(32 * 32);
    for (int y = 0; y < 32; ++y) {
      for (int x = 0; x < 32; ++x) {
        const float dx = static_cast<float>(x) - cx, dy = static_cast<float>(y) - 16.f;
        f[static_cast<std::size_t>(y) * 32 + static_cast<std::size_t>(x)] = std::exp(-(dx * dx + dy * dy) / 8.f);
      }
    }
    return f;
  };
  s.fine_t = blob(12.f);
  s.fine_d.assign(32 * 32, 0.f);
  const std::vector<float> target = blob(14.f);
  for (std::size_t c = 0; c < 64; ++c) {  // the coarse heat: block averages of the blob two pixels on
    float sum = 0.f;
    for (std::size_t j = 0; j < 16; ++j) sum += target[((c / 8) * 4 + j / 4) * 32 + (c % 8) * 4 + j % 4];
    s.coarse[c * C + 2] = sum / 16.f;
  }
  const auto centre = [](const State& st) {
    double sx = 0, sw = 0;
    for (std::size_t i = 0; i < st.fine_t.size(); ++i) {
      sx += static_cast<double>(i % 32) * st.fine_t[i];
      sw += st.fine_t[i];
    }
    return sx / sw;
  };
  State one = s;
  Model m1 = m;
  m1.detail.passes = 1;
  const std::vector<float> ctl{0.5f, 0.5f, 0.5f};
  detail_step(m, s, 1, ctl);
  detail_step(m1, one, 1, ctl);
  EXPECT_NEAR(centre(s), 14.0, 0.3);
  EXPECT_NEAR(centre(one), 14.0, 0.3);
  EXPECT_NE(s.fine_t, one.fine_t);
}

TEST(Rollout, MacCormackAdvectionKeepsARotatingRingSharper) {
  // A ring of heat in a solid rotation of 1.5 cells a frame at its radius: after 8 steps the semi-Lagrangian step has
  // smeared it more than MacCormack, whose values stay inside the range of what they were read from.
  const int R = 32, C = 4;
  std::vector<float> s(static_cast<std::size_t>(R) * R * C, 0.f);
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const float dx = static_cast<float>(x) - 15.5f, dy = static_cast<float>(y) - 15.5f, rr = std::sqrt(dx * dx + dy * dy);
      float* c = s.data() + (static_cast<std::size_t>(y) * R + static_cast<std::size_t>(x)) * C;
      c[0] = -dy * 1.5f / 8.f;
      c[1] = dx * 1.5f / 8.f;
      c[2] = std::exp(-(rr - 8.f) * (rr - 8.f) / 2.f);
    }
  }
  std::vector<float> sl = s, mc = s, tmp(s.size());
  const auto peak = [&](const std::vector<float>& v) {
    float p = 0.f;
    for (std::size_t i = 2; i < v.size(); i += C) p = std::max(p, v[i]);
    return p;
  };
  for (int it = 0; it < 8; ++it) {
    advect_maccormack(mc, R, C, tmp);
    for (std::size_t i = 2; i < mc.size(); i += C) mc[i] = tmp[i];  // the flow is held fixed
    for (int y = 0; y < R; ++y) {
      for (int x = 0; x < R; ++x) {
        const std::size_t i = (static_cast<std::size_t>(y) * R + static_cast<std::size_t>(x)) * C;
        const float px = std::clamp(static_cast<float>(x) - sl[i], 0.f, 31.f), py = std::clamp(static_cast<float>(y) - sl[i + 1], 0.f, 31.f);
        const int x0 = std::min(static_cast<int>(px), 30), y0 = std::min(static_cast<int>(py), 30);
        const float fx = px - static_cast<float>(x0), fy = py - static_cast<float>(y0);
        const auto at = [&](int xx, int yy) { return sl[(static_cast<std::size_t>(yy) * R + static_cast<std::size_t>(xx)) * C + 2]; };
        tmp[i + 2] = (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1 - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
      }
    }
    for (std::size_t i = 2; i < sl.size(); i += C) sl[i] = tmp[i];
  }
  EXPECT_GT(peak(mc), peak(sl) + 0.1f);
  EXPECT_LE(peak(mc), 1.f + 1e-5f);
}

TEST(Rollout, SoftenMovesTowardsTheNeighboursMean) {
  // q + s (mean of the four neighbours - q), zero outside: a single spike keeps 1 - s and gives s / 4 to each
  // neighbour; a uniform field is unchanged inside and loses s / 4 per missing neighbour at the edges.
  const int S = 8;
  std::vector<float> q(S * S, 0.f);
  q[3 * S + 4] = 1.f;
  soften_field(q, S, 0.4f);
  EXPECT_FLOAT_EQ(q[3 * S + 4], 0.6f);
  for (const int i : {3 * S + 3, 3 * S + 5, 2 * S + 4, 4 * S + 4}) EXPECT_FLOAT_EQ(q[static_cast<std::size_t>(i)], 0.1f);
  EXPECT_FLOAT_EQ(q[2 * S + 3], 0.f);
  std::vector<float> u(S * S, 1.f);
  soften_field(u, S, 0.4f);
  EXPECT_FLOAT_EQ(u[3 * S + 4], 1.f);
  EXPECT_FLOAT_EQ(u[3 * S], 0.9f);  // left edge
  EXPECT_FLOAT_EQ(u[0], 0.8f);      // corner
}

TEST(Rollout, AdvectionGainScalesHowFarDetailMoves) {
  // As DetailLayerCarriesFieldsWithTheFlow, with advect = 2: the blob moves two pixels for a quarter cell of flow.
  Model m = tiny_model();
  m.detail.swirl = 0.f;
  m.detail.contrast = 0.f;
  m.detail.advect = 2.f;
  const int R = m.h.res, S = 32, C = m.h.channels();
  State s;
  s.res = R;
  s.size = S;
  s.coarse.assign(static_cast<std::size_t>(R) * R * C, 0.f);
  s.flow.assign(static_cast<std::size_t>(R) * R * 2, 0.f);
  for (int i = 0; i < R * R; ++i) s.flow[static_cast<std::size_t>(i) * 2] = 0.25f;
  s.fine_t.assign(static_cast<std::size_t>(S) * S, 0.f);
  s.fine_d.assign(s.fine_t.size(), 0.f);
  for (int y = 12; y < 20; ++y) {
    for (int x = 8; x < 16; ++x) s.fine_t[static_cast<std::size_t>(y) * S + x] = 1.f;
  }
  std::vector<float> moved(s.fine_t.size(), 0.f);
  for (int y = 12; y < 20; ++y) {
    for (int x = 10; x < 18; ++x) moved[static_cast<std::size_t>(y) * S + x] = 1.f;
  }
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) s.coarse[(static_cast<std::size_t>(y / 4) * R + x / 4) * C + 2] += moved[static_cast<std::size_t>(y) * S + x] / 16.f;
  }
  detail_step(m, s, 1, {});
  double mass = 0, err = 0;
  for (std::size_t i = 0; i < moved.size(); ++i) {
    mass += s.fine_t[i];
    err += std::abs(s.fine_t[i] - moved[i]);
  }
  EXPECT_NEAR(mass, 64.0, 1.0);
  EXPECT_LT(err, 2.0);
}

TEST(Rollout, TrainerForwardMatchesTheReference) {
  const Model m = tiny_model();
  const auto r = tiny_run(m.h, 6);
  const int R = m.h.res, N = R * R, C = m.h.channels();
  // Reference: one coarse_step from state 2, scored against state 3.
  std::vector<float> in(static_cast<std::size_t>(N) * C, 0.f), out(in.size()), flow(static_cast<std::size_t>(N) * 2), p(static_cast<std::size_t>(N), 0.f);
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < kPhys; ++k) in[static_cast<std::size_t>(j) * C + k] = r.coarse[(2 * static_cast<std::size_t>(N) + j) * kPhys + k];
  }
  std::vector<float> cond(static_cast<std::size_t>(m.h.cond())), noise(static_cast<std::size_t>(N) * kNoise);
  condition(m, r.controls(), 3.f / r.p.fps, cond);
  coarse_noise(m, r.p.seed, 3.f / r.p.fps, noise);
  coarse_step(m, in, noise, cond, p, out, flow);
  double ref = 0;
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < kPhys; ++k) {
      const double e = out[static_cast<std::size_t>(j) * C + k] - r.coarse[(3 * static_cast<std::size_t>(N) + j) * kPhys + k];
      ref += e * e / (static_cast<double>(m.scale[k]) * m.scale[k]);
    }
  }
  ref /= N * kPhys;
  const double tr = window_loss(m, r, 2, 1, 0, 0.f, 0.f, 1, nullptr);
  EXPECT_NEAR(tr, ref, 1e-5 * ref + 1e-9);
}

TEST(Rollout, TrainerForwardMatchesTheReferenceWithMacCormack) {
  // Study L's advection (Hyper::advection = 1): the trainer's step is the reference's, and it is not the
  // semi-Lagrangian one.
  Model m = tiny_model();
  m.h.advection = 1;
  const auto r = tiny_run(m.h, 6);
  const int R = m.h.res, N = R * R, C = m.h.channels();
  std::vector<float> in(static_cast<std::size_t>(N) * C, 0.f), out(in.size()), flow(static_cast<std::size_t>(N) * 2), p(static_cast<std::size_t>(N), 0.f);
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < kPhys; ++k) in[static_cast<std::size_t>(j) * C + k] = r.coarse[(2 * static_cast<std::size_t>(N) + j) * kPhys + k];
  }
  std::vector<float> cond(static_cast<std::size_t>(m.h.cond())), noise(static_cast<std::size_t>(N) * kNoise);
  condition(m, r.controls(), 3.f / r.p.fps, cond);
  coarse_noise(m, r.p.seed, 3.f / r.p.fps, noise);
  coarse_step(m, in, noise, cond, p, out, flow);
  double ref = 0;
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < kPhys; ++k) {
      const double e = out[static_cast<std::size_t>(j) * C + k] - r.coarse[(3 * static_cast<std::size_t>(N) + j) * kPhys + k];
      ref += e * e / (static_cast<double>(m.scale[k]) * m.scale[k]);
    }
  }
  ref /= N * kPhys;
  const double tr = window_loss(m, r, 2, 1, 0, 0.f, 0.f, 1, nullptr);
  EXPECT_NEAR(tr, ref, 1e-5 * ref + 1e-9);
  Model sl = m;
  sl.h.advection = 0;
  EXPECT_GT(std::abs(window_loss(sl, r, 2, 1, 0, 0.f, 0.f, 1, nullptr) - tr), 1e-4 * tr);
}

namespace {

// Couplings on the tiny run (docs/COMPOSE.md §9): every operation before two steps in three, small enough that the
// backtraces stay inside their cells.
Run tiny_forced_run(const Hyper& h, int frames) {
  Run r = tiny_run(h, frames);
  const int N = h.res * h.res;
  r.forcing_at.assign(static_cast<std::size_t>(frames), -1);
  for (int i = 0; i < frames; ++i) {
    if (i % 3 == 0) continue;
    r.forcing_at[static_cast<std::size_t>(i)] = static_cast<int>(r.forcing.size() / (static_cast<std::size_t>(N) * kForce));
    for (int j = 0; j < N; ++j) {
      const float a = 0.3f * static_cast<float>(i) + 0.7f * static_cast<float>(j);
      r.forcing.insert(r.forcing.end(), {0.03f * std::sin(a), 0.025f * std::cos(a), 0.012f * std::sin(1.3f * a), -0.01f * std::cos(0.7f * a),
                                         0.9f + 0.08f * std::sin(a), 0.85f + 0.1f * std::cos(a), 0.03f * (1.f + std::sin(a)), 0.02f * (1.f + std::cos(a))});
    }
  }
  return r;
}

void rollout_gradient_check(int unroll, int burn, float profile, float activity = 0.f, bool forced = false, const Model* anchor = nullptr,
                            float anchor_weight = 0.f, int advection = 0) {
  Model m = tiny_model(11 + static_cast<std::uint64_t>(unroll + burn));
  m.h.advection = advection;
  const auto r = forced ? tiny_forced_run(m.h, 12) : tiny_run(m.h, 12);
  std::vector<float> grad(m.step_w.size(), 0.f);
  const double l0 = window_loss(m, r, 1, unroll, burn, 0.f, profile, 5, &grad, activity, anchor, anchor_weight);
  ASSERT_GT(l0, 0.0);
  const auto eval = [&] { return window_loss(m, r, 1, unroll, burn, 0.f, profile, 5, nullptr, activity, anchor, anchor_weight); };
  // Central differences at three step sizes, and one-sided differences at the two smaller ones. A parameter passes
  // when one of them matches within 3%: the loss is summed in float, and ReLUs, clamps and bilinear interpolation make
  // it only piecewise smooth, so a kink just beside the current point spoils the central differences while the
  // difference on the far side of it stays exact. A wrong gradient matches none. Scattered estimates without a match
  // mark a kink and are skipped (at most 10%).
  const auto diffs = [&](float& slot) {
    const float keep = slot;
    const double base = eval();
    std::vector<double> out;
    for (const float eps : {1e-2f, 3e-3f, 1e-3f}) {
      slot = keep + eps;
      const double lp = eval();
      slot = keep - eps;
      const double lm = eval();
      slot = keep;
      out.push_back((lp - lm) / (2.0 * static_cast<double>(eps)));
      if (eps < 5e-3f) {
        out.push_back((lp - base) / static_cast<double>(eps));
        out.push_back((base - lm) / static_cast<double>(eps));
      }
    }
    return out;
  };
  const auto near = [](double a, double b) { return std::abs(a - b) <= 3e-2 * std::max(std::abs(a), std::abs(b)) + 2e-5; };
  int checked = 0, bad = 0, live = 0, kinks = 0;
  for (std::size_t i = 0; i < m.step_w.size(); i += std::max<std::size_t>(1, m.step_w.size() / 250)) {
    const std::vector<double> n = diffs(m.step_w[i]);
    const double ana = grad[i];
    if (std::ranges::any_of(n, [&](double v) { return near(v, ana); })) {
      ++checked;
      live += std::abs(n[0]) > 2e-5;
      continue;
    }
    const auto [lo, hi] = std::ranges::minmax(n);
    if (hi - lo > 0.1 * std::max(std::abs(lo), std::abs(hi)) + 1e-6) {
      ++kinks;
      continue;
    }
    ++checked;
    if (++bad <= 6) ADD_FAILURE() << std::format("param {}: analytic {} central {} {} {}", i, ana, n[0], n[1], n[4]);
  }
  EXPECT_GT(checked, 100);
  EXPECT_LE(kinks, (checked + kinks) / 10) << "too many parameters at kinks";
  EXPECT_GT(live, checked / 2) << "most gradients are zero: the check would be vacuous";
}

}  // namespace

TEST(Rollout, StudyMMeanLossGradientsMatchFiniteDifferences) {
  // mean_window_loss: two rollouts with their own seeds, the product of their pooled differences from the run; its
  // gradient through both rollouts against finite differences (the same acceptance as the window loss's). Without a
  // burn-in: the burn-in is not differentiated (as the window loss's), so finite differences would see it.
  Model m = tiny_model(31);
  const auto r = tiny_run(m.h, 14);
  std::vector<float> grad(m.step_w.size(), 0.f);
  const auto eval = [&](std::vector<float>* g) { return mean_window_loss(m, r, 1, 0, 5, 2, 101, 202, g, 1.f); };
  const double l0 = eval(&grad);
  EXPECT_NE(l0, 0.0);
  int checked = 0, bad = 0, live = 0, kinks = 0;
  const auto near = [](double a, double b) { return std::abs(a - b) <= 3e-2 * std::max(std::abs(a), std::abs(b)) + 2e-5; };
  for (std::size_t i = 0; i < m.step_w.size(); i += std::max<std::size_t>(1, m.step_w.size() / 250)) {
    float& slot = m.step_w[i];
    const float keep = slot;
    const double base = eval(nullptr);
    std::vector<double> n;
    for (const float eps : {1e-2f, 3e-3f, 1e-3f}) {
      slot = keep + eps;
      const double lp = eval(nullptr);
      slot = keep - eps;
      const double lm = eval(nullptr);
      slot = keep;
      n.push_back((lp - lm) / (2.0 * static_cast<double>(eps)));
      if (eps < 5e-3f) {
        n.push_back((lp - base) / static_cast<double>(eps));
        n.push_back((base - lm) / static_cast<double>(eps));
      }
    }
    const double ana = grad[i];
    if (std::ranges::any_of(n, [&](double v) { return near(v, ana); })) {
      ++checked;
      live += std::abs(n[0]) > 2e-5;
      continue;
    }
    const auto [lo, hi] = std::ranges::minmax(n);
    if (hi - lo > 0.1 * std::max(std::abs(lo), std::abs(hi)) + 1e-6) {
      ++kinks;
      continue;
    }
    ++checked;
    if (++bad <= 6) ADD_FAILURE() << std::format("param {}: analytic {} central {} {} {}", i, ana, n[0], n[1], n[4]);
  }
  EXPECT_GT(checked, 100);
  EXPECT_LE(kinks, (checked + kinks) / 10) << "too many parameters at kinks";
  EXPECT_GT(live, checked / 2) << "most gradients are zero: the check would be vacuous";
  // the seeds matter (each rollout is driven by its own noise), the same seeds give the same loss, and a burn-in moves
  // the window
  EXPECT_EQ(eval(nullptr), mean_window_loss(m, r, 1, 0, 5, 2, 101, 202, nullptr, 1.f));
  EXPECT_NE(eval(nullptr), mean_window_loss(m, r, 1, 0, 5, 2, 101, 303, nullptr, 1.f));
  EXPECT_NE(mean_window_loss(m, r, 1, 3, 5, 2, 101, 202, nullptr, 1.f), mean_window_loss(m, r, 4, 0, 5, 2, 101, 202, nullptr, 1.f));
}

TEST(Rollout, GradientsMatchFiniteDifferencesOneStep) { rollout_gradient_check(1, 0, 0.f); }
TEST(Rollout, GradientsMatchFiniteDifferencesThroughTime) { rollout_gradient_check(4, 0, 0.f); }
// The burn-in frames of a fine-tuning window are a stop-gradient by design (the window starts from wherever the
// stepper's own rollout went), so finite differences, which would re-run them, check the profile loss without them.
TEST(Rollout, GradientsMatchFiniteDifferencesWithProfiles) { rollout_gradient_check(3, 0, 1.f); }
TEST(Rollout, GradientsMatchFiniteDifferencesWithActivity) { rollout_gradient_check(4, 0, 0.5f, 50.f); }
// Through the couplings: the push is added and taken out, forces and material added, v and material multiplied.
TEST(Rollout, GradientsMatchFiniteDifferencesWithCouplings) { rollout_gradient_check(4, 0, 0.5f, 50.f, true); }
// Study L's MacCormack advection: through its round trip, its clamp and both samples' positions.
TEST(Rollout, GradientsMatchFiniteDifferencesWithMacCormackOneStep) { rollout_gradient_check(1, 0, 0.f, 0.f, false, nullptr, 0.f, 1); }
TEST(Rollout, GradientsMatchFiniteDifferencesWithMacCormackThroughTime) { rollout_gradient_check(4, 0, 0.5f, 50.f, false, nullptr, 0.f, 1); }

// Round 2 (docs/COMPOSE.md §10): the anchor to another model's first step, alone and with couplings.
TEST(Rollout, GradientsMatchFiniteDifferencesWithTheAnchor) {
  const Model a = tiny_model(99);
  rollout_gradient_check(3, 0, 0.5f, 20.f, false, &a, 5.f);
}
TEST(Rollout, GradientsMatchFiniteDifferencesWithTheAnchorAndCouplings) {
  const Model a = tiny_model(98);
  rollout_gradient_check(2, 0, 0.f, 0.f, true, &a, 3.f);
}

TEST(Rollout, AnchorAddsTheFirstStepsDistanceFromATrueState) {
  const Model m = tiny_model(), other = tiny_model(42);
  const auto r = tiny_run(m.h, 12);
  const double plain = window_loss(m, r, 1, 3, 0, 0.f, 0.5f, 5, nullptr);
  EXPECT_EQ(window_loss(m, r, 1, 3, 0, 0.f, 0.5f, 5, nullptr, 0.f, &m, 7.f), plain);  // anchored to itself: no distance
  const double d1 = window_loss(m, r, 1, 3, 0, 0.f, 0.5f, 5, nullptr, 0.f, &other, 1.f) - plain;
  const double d4 = window_loss(m, r, 1, 3, 0, 0.f, 0.5f, 5, nullptr, 0.f, &other, 4.f) - plain;
  EXPECT_GT(d1, 0.0);
  EXPECT_NEAR(d4, 4.0 * d1, 1e-6 * std::abs(plain) + 1e-9);
  // after a burn-in the window does not start at a true state: no anchor
  EXPECT_EQ(window_loss(m, r, 1, 3, 2, 0.f, 0.5f, 5, nullptr, 0.f, &other, 4.f), window_loss(m, r, 1, 3, 2, 0.f, 0.5f, 5, nullptr));
}

TEST(Rollout, WindowsKnowWhereCouplingsAct) {
  const Hyper h = tiny_hyper();
  const auto forced = tiny_forced_run(h, 12), plain = tiny_run(h, 12);
  // forced_run leaves out every third state (0, 3, 6, 9): steps that produce states 3 and 4 include state 4's coupling
  EXPECT_TRUE(window_has_coupling(forced, 2, 2));
  EXPECT_FALSE(window_has_coupling(forced, 2, 1));
  EXPECT_FALSE(window_has_coupling(plain, 0, 10));
}

TEST(Rollout, SceneForcingIsDeterministicAndStrong) {
  int rings = 0, removes = 0;
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    for (const auto e : {sim::Effect::fire, sim::Effect::smoke}) {
      const ForcingSpec a = scene_forcing(e, 240, seed, 15, 180), b = scene_forcing(e, 240, seed, 15, 180);
      ASSERT_EQ(a.events.size(), b.events.size());
      ASSERT_GE(a.events.size(), 1u);
      const Coupling& gale = a.events[0];
      EXPECT_EQ(gale.kind, Coupling::Kind::push);
      EXPECT_EQ(gale.shape, Coupling::Shape::wave);
      EXPECT_GE(gale.onset, 15);
      EXPECT_LE(gale.onset, 25);
      EXPECT_LE(gale.amp, e == sim::Effect::fire ? 0.6f : 0.4f);
      for (std::size_t k = 0; k < a.events.size(); ++k) {
        const Coupling& c = a.events[k];
        EXPECT_EQ(c.onset, b.events[k].onset);
        EXPECT_EQ(c.amp, b.events[k].amp);
        EXPECT_LE(c.onset + c.duration, 240);
        if (c.shape == Coupling::Shape::vortex) {
          EXPECT_LE(std::abs(c.amp), 0.7f);
          ++rings;
        }
        removes += c.kind == Coupling::Kind::remove;
      }
    }
  }
  EXPECT_GT(rings, 300);  // two vortices per ring, in about 60% of 400 specs
  EXPECT_GT(removes, 50);
}

TEST(Rollout, CouplingsChangeTheWindowAndBurnInFollowsThem) {
  const Model m = tiny_model();
  const auto plain = tiny_run(m.h, 12), forced = tiny_forced_run(m.h, 12);
  EXPECT_NE(window_loss(m, plain, 1, 3, 0, 0.f, 0.f, 5, nullptr), window_loss(m, forced, 1, 3, 0, 0.f, 0.f, 5, nullptr));
  const double a = window_loss(m, plain, 1, 2, 3, 0.f, 0.f, 5, nullptr), b = window_loss(m, forced, 1, 2, 3, 0.f, 0.f, 5, nullptr);
  EXPECT_TRUE(std::isfinite(a) && std::isfinite(b));
  EXPECT_NE(a, b);
  // a forced run whose slots are all empty is the plain run
  rollout::Run none = forced;
  std::ranges::fill(none.forcing_at, -1);
  EXPECT_EQ(window_loss(m, plain, 1, 3, 2, 0.f, 0.5f, 5, nullptr, 20.f), window_loss(m, none, 1, 3, 2, 0.f, 0.5f, 5, nullptr, 20.f));
}

TEST(Rollout, ApplyForcingAndRemovePushMirrorCompose) {
  const int C = 6;
  std::vector<float> s = {0.5f, -0.2f, 0.8f, 0.3f, 0.1f, 0.2f};
  const std::vector<float> f = {0.25f, 0.125f, 0.5f, -0.25f, 0.5f, 0.75f, 0.125f, 0.0625f};
  apply_forcing(s, C, f);
  EXPECT_FLOAT_EQ(s[0], 0.5f + 0.5f + 0.25f);
  EXPECT_FLOAT_EQ(s[1], -0.2f * 0.5f - 0.25f + 0.125f);
  EXPECT_FLOAT_EQ(s[2], 0.8f * 0.75f + 0.125f);
  EXPECT_FLOAT_EQ(s[3], 0.3f * 0.75f + 0.0625f);
  EXPECT_EQ(s[4], 0.1f);  // memory channels are not touched
  remove_push(s, C, f);
  EXPECT_FLOAT_EQ(s[0], 0.5f + 0.5f);  // the force stays, the push goes
  EXPECT_FLOAT_EQ(s[1], -0.2f * 0.5f - 0.25f);
  std::vector<float> t = {0.f, 0.f, 0.1f, 0.1f};
  apply_forcing(t, 4, std::vector<float>{0, 0, 0, 0, 1, 1, -0.5f, 0});
  EXPECT_EQ(t[2], 0.f);  // heat stays at or above zero
}

TEST(Rollout, ForcedRunWithoutEventsIsThePlainRun) {
  sim::Params p;
  p.effect = sim::Effect::fire;
  p.size = 32;
  p.pressure_iters = 10;
  p.seed = 9;
  const rollout::Run a = record_run(p, 12, 8), b = record_forced_run(p, ForcingSpec{}, 12, 8);
  EXPECT_EQ(a.coarse, b.coarse);
  EXPECT_TRUE(std::ranges::all_of(b.forcing_at, [](int k) { return k < 0; }));
  // with couplings: deterministic, different from the plain run, the slots where the spec is active
  const ForcingSpec spec = random_forcing(sim::Effect::fire, 12, 77, 2, 6);
  const rollout::Run c = record_forced_run(p, spec, 12, 8), d = record_forced_run(p, spec, 12, 8);
  EXPECT_EQ(c.coarse, d.coarse);
  EXPECT_EQ(c.forcing, d.forcing);
  EXPECT_NE(c.coarse, a.coarse);
  for (int i = 0; i < 12; ++i) EXPECT_EQ(c.forcing_at[static_cast<std::size_t>(i)] >= 0, spec.active(i)) << i;
}

TEST(Rollout, CoarseForcingIsAveragedLikeTheState) {
  ForcingSpec spec;
  Coupling g;
  g.kind = Coupling::Kind::push;
  g.shape = Coupling::Shape::vortex;
  g.duration = 5;
  g.amp = 0.4f;
  g.radius = 0.2f;
  spec.events.push_back(g);
  Coupling a;
  a.kind = Coupling::Kind::add;
  a.duration = 5;
  a.amp = 0.05f;
  a.amp2 = 0.02f;
  spec.events.push_back(a);
  sim::Params p;
  SimForcing sf;
  forcing_fields(spec, p, 64, 2, sf);
  std::vector<float> cf(16 * 16 * kForce), cs(16 * 16 * kPhys);
  coarse_forcing(sf, 16, p.fps, cf);
  sim::State st;
  st.n = 64;
  st.u = sf.pu;
  st.v = sf.pv;
  st.temp = sf.ah;
  st.soot = sf.as;
  coarse_from_sim(st, 16, p.fps, cs);
  float peak = 0.f;
  for (int i = 0; i < 16 * 16; ++i) {
    for (int k = 0; k < 2; ++k) EXPECT_EQ(cf[static_cast<std::size_t>(i) * kForce + k], cs[static_cast<std::size_t>(i) * kPhys + k]);
    EXPECT_EQ(cf[static_cast<std::size_t>(i) * kForce + 6], cs[static_cast<std::size_t>(i) * kPhys + 2]);
    EXPECT_EQ(cf[static_cast<std::size_t>(i) * kForce + 7], cs[static_cast<std::size_t>(i) * kPhys + 3]);
    EXPECT_FLOAT_EQ(cf[static_cast<std::size_t>(i) * kForce + 4], 1.f);  // no ceiling
    peak = std::max(peak, std::hypot(cf[static_cast<std::size_t>(i) * kForce], cf[static_cast<std::size_t>(i) * kForce + 1]));
  }
  // the vortex's peak speed is `amp` cells of a 32-cell grid per frame: 0.2 cells of this 16-cell grid
  EXPECT_NEAR(peak, 0.4f * 16.f / 32.f, 0.03f);
}

TEST(Rollout, RandomForcingIsDeterministicAndCoversEveryKind) {
  std::array<int, 5> kinds{};
  for (std::uint64_t seed = 1; seed <= 300; ++seed) {
    const ForcingSpec a = random_forcing(sim::Effect::smoke, 240, seed, 15, 230), b = random_forcing(sim::Effect::smoke, 240, seed, 15, 230);
    ASSERT_EQ(a.events.size(), b.events.size());
    ASSERT_GE(a.events.size(), 1u);
    for (std::size_t k = 0; k < a.events.size(); ++k) {
      const Coupling& c = a.events[k];
      EXPECT_EQ(c.onset, b.events[k].onset);
      EXPECT_EQ(c.amp, b.events[k].amp);
      EXPECT_GE(c.onset, 15);
      EXPECT_LE(c.onset + c.duration, 240);
      ++kinds[static_cast<std::size_t>(c.kind)];
    }
  }
  for (const int n : kinds) EXPECT_GT(n, 30);
}

TEST(Rollout, HandoverRunStartsFromTheExplosionsState) {
  sim::Params ex, sm;
  ex.effect = sim::Effect::explosion;
  ex.size = 32;
  ex.pressure_iters = 10;
  sm = ex;
  sm.effect = sim::Effect::smoke;
  sm.seed = 3;
  const rollout::Run r = record_handover_run(ex, 6, sm, 5, 8);
  sim::Fluid f(ex);
  for (int i = 0; i < 6; ++i) f.step_frame();
  std::vector<float> c(8 * 8 * kPhys);
  coarse_from_sim(f.state(), 8, ex.fps, c);
  EXPECT_EQ(std::vector<float>(r.coarse.begin(), r.coarse.begin() + static_cast<std::ptrdiff_t>(c.size())), c);
  EXPECT_EQ(r.p.effect, sim::Effect::smoke);
  EXPECT_FLOAT_EQ(r.t0 + 1.f / ex.fps, 6.f / ex.fps);  // state 0 at the explosion's time
  // and then the smoke simulation continues from it
  sim::Params q = sm;
  q.sim_res = 32;
  sim::Fluid g(q);
  g.set_state(f.state());
  g.step_frame();
  coarse_from_sim(g.state(), 8, ex.fps, c);
  EXPECT_EQ(std::vector<float>(r.coarse.begin() + static_cast<std::ptrdiff_t>(c.size()), r.coarse.begin() + static_cast<std::ptrdiff_t>(2 * c.size())), c);
}

TEST(Rollout, BurnInStartsFromTheStepperRollout) {
  const Model m = tiny_model();
  const auto r = tiny_run(m.h, 12);
  // Same window after burn-in, against a window from true states: different starts, so different losses, both finite.
  const double a = window_loss(m, r, 1, 2, 3, 0.f, 0.f, 5, nullptr), b = window_loss(m, r, 4, 2, 0, 0.f, 0.f, 5, nullptr);
  EXPECT_TRUE(std::isfinite(a) && std::isfinite(b));
  EXPECT_NE(a, b);
  std::vector<float> g(m.step_w.size(), 0.f);
  EXPECT_DOUBLE_EQ(window_loss(m, r, 1, 2, 3, 0.f, 0.f, 5, &g), a);
}

TEST(Rollout, ReferenceRolloutStaysFiniteAndRenders) {
  Model m = tiny_model();
  const auto r = tiny_run(m.h, 4);
  StartPoint sp;
  sp.controls = r.controls();
  sp.seed = 5;
  sp.time = 2.f;
  sp.coarse.assign(r.coarse.begin(), r.coarse.begin() + 8 * 8 * kPhys);
  m.starts.push_back(sp);
  m.render_scale = {1.f, 1.f};
  for (const int size : {16, 32}) {
    State s = start(m, 0, size, sp.controls, 77);
    EXPECT_NEAR(s.time, 2.f + static_cast<float>(m.h.warmup) / m.fps, 1e-5f);
    for (int f = 0; f < 20; ++f) step(m, s, sp.controls, 77);
    for (const float v : s.coarse) ASSERT_TRUE(std::isfinite(v));
    for (const float v : s.fine_t) ASSERT_TRUE(std::isfinite(v) && v >= 0.f);
    for (const float v : s.fine_d) ASSERT_TRUE(std::isfinite(v) && v >= 0.f);
    std::vector<float> rgba(static_cast<std::size_t>(size) * size * 4);
    render(m, s, rgba);
    for (const float v : rgba) ASSERT_TRUE(v >= 0.f && v <= 1.f);
  }
}

TEST(Rollout, DetailLayerCarriesFieldsWithTheFlow) {
  // No swirl, no contrast, a uniform flow of one coarse cell per frame to the right, a coarse state that matches the
  // fine field everywhere: a blob moves right by size / res pixels per frame and keeps its mass.
  Model m = tiny_model();
  m.detail.swirl = 0.f;
  m.detail.contrast = 0.f;
  const int R = m.h.res, S = 32, C = m.h.channels();
  State s;
  s.res = R;
  s.size = S;
  s.coarse.assign(static_cast<std::size_t>(R) * R * C, 0.f);
  s.flow.assign(static_cast<std::size_t>(R) * R * 2, 0.f);
  for (int i = 0; i < R * R; ++i) s.flow[static_cast<std::size_t>(i) * 2] = 0.25f;  // a quarter cell = 1 pixel
  s.fine_t.assign(static_cast<std::size_t>(S) * S, 0.f);
  s.fine_d.assign(s.fine_t.size(), 0.f);
  for (int y = 12; y < 20; ++y) {
    for (int x = 8; x < 16; ++x) s.fine_t[static_cast<std::size_t>(y) * S + x] = 1.f;
  }
  // coarse target: the block averages after the move
  std::vector<float> moved(s.fine_t.size(), 0.f);
  for (int y = 12; y < 20; ++y) {
    for (int x = 9; x < 17; ++x) moved[static_cast<std::size_t>(y) * S + x] = 1.f;
  }
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) s.coarse[(static_cast<std::size_t>(y / 4) * R + x / 4) * C + 2] += moved[static_cast<std::size_t>(y) * S + x] / 16.f;
  }
  detail_step(m, s, 1, {});
  double mass = 0, err = 0;
  for (std::size_t i = 0; i < moved.size(); ++i) {
    mass += s.fine_t[i];
    err += std::abs(s.fine_t[i] - moved[i]);
  }
  EXPECT_NEAR(mass, 64.0, 1.0);
  EXPECT_LT(err, 2.0);  // the edge columns are within a pixel's interpolation of exact
}

// --- the runtime (C API) on rollout effects ----------------------------------------------------------------------------

#include <neuralfx/nvfx.h>

namespace {

// A small rollout effect with two start points: one grown from its coarse state, one with fine fields.
Model runtime_model(int res = 16) {
  Hyper h;
  h.res = res;
  h.hidden = 8;
  h.memory = 2;
  h.jacobi = 10;
  h.render_hidden = 6;
  h.start_fine = 16;
  h.warmup = 3;
  Model m = init_model(h, 21);
  std::mt19937_64 rng(22);
  std::normal_distribution<float> nd(0.f, 0.1f);
  for (float& w : m.step_w) w += nd(rng);
  for (float& w : m.render_w) w = 0.5f * w + 0.05f;
  m.effect = "test";
  m.control_names = {"intensity", "wind", "turbulence"};
  m.scale = {0.2f, 0.2f, 0.4f, 0.3f};
  m.lo = {-2.f, -2.f, 0.f, 0.f};
  m.hi = {2.f, 2.f, 3.f, 3.f};
  m.render_scale = {1.f, 1.f};
  m.detail.swirl_control = 2;
  m.detail.swirl = 0.6f;
  const int R = h.res;
  for (int k = 0; k < 2; ++k) {
    StartPoint sp;
    sp.controls = {0.3f + 0.4f * static_cast<float>(k), 0.5f, 0.6f};
    sp.seed = 1000 + static_cast<std::uint64_t>(k);
    sp.time = 2.f;
    sp.coarse.resize(static_cast<std::size_t>(R) * R * kPhys);
    for (int y = 0; y < R; ++y) {
      for (int x = 0; x < R; ++x) {
        float* c = sp.coarse.data() + (static_cast<std::size_t>(y) * R + x) * kPhys;
        c[0] = 0.1f * std::sin(0.5f * static_cast<float>(x + k));
        c[1] = 0.2f + 0.1f * std::cos(0.4f * static_cast<float>(y));
        c[2] = std::max(0.f, 0.8f - 0.05f * static_cast<float>(std::abs(x - 8) + y));
        c[3] = std::max(0.f, 0.5f - 0.04f * static_cast<float>(std::abs(x - 7) + std::abs(y - 6)));
      }
    }
    if (k == 1) {
      sp.fine_t.resize(256);
      sp.fine_d.resize(256);
      for (int i = 0; i < 256; ++i) {
        sp.fine_t[static_cast<std::size_t>(i)] = std::max(0.f, 0.9f - 0.03f * static_cast<float>(std::abs(i % 16 - 8) + i / 16));
        sp.fine_d[static_cast<std::size_t>(i)] = 0.3f + 0.2f * std::sin(0.7f * static_cast<float>(i));
      }
    }
    m.starts.push_back(sp);
  }
  quantise_like_storage(m);
  return m;
}

struct Fx {
  nvfx_effect* e = nullptr;
  explicit Fx(const Model& m) {
    std::ostringstream os;
    EXPECT_TRUE(save_model(os, m));
    const std::string b = os.str();
    EXPECT_EQ(nvfx_effect_load_memory(b.data(), b.size(), &e), NVFX_OK);
  }
  ~Fx() { nvfx_effect_free(e); }
  Fx(const Fx&) = delete;
  Fx& operator=(const Fx&) = delete;
};

std::vector<std::uint8_t> reference_frame(const Model& m, int start_index, int size, int frames, Int8 q = Int8::off) {
  State s = rollout::start(m, start_index, size, m.starts[static_cast<std::size_t>(start_index)].controls, m.starts[static_cast<std::size_t>(start_index)].seed, q);
  for (int f = 0; f < frames; ++f) rollout::step(m, s, m.starts[static_cast<std::size_t>(start_index)].controls, m.starts[static_cast<std::size_t>(start_index)].seed, q);
  std::vector<float> rgba(static_cast<std::size_t>(size) * size * 4);
  rollout::render(m, s, rgba);
  std::vector<std::uint8_t> out(rgba.size());
  for (std::size_t i = 0; i < rgba.size(); ++i) out[i] = static_cast<std::uint8_t>(rgba[i] * 255.f + 0.5f);
  return out;
}

}  // namespace

namespace {

// Study M: a mean state of modest size on a model's grid (blocks of `block` cells), from a seed.
void add_mean_state(Model& m, int block, std::uint64_t seed) {
  const int G = m.h.res / block;
  m.mean.block = block;
  m.mean.beta.resize(static_cast<std::size_t>(kMeanFeatures) * G * G * kPhys);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.f, 0.08f);
  for (float& b : m.mean.beta) b = nd(rng);
}

}  // namespace

TEST(Rollout, StudyMMeanStateRoundTripsInVersion6) {
  // Study M: Model::mean. Without one the file keeps its old layout; with one it is version 6 (with the version 4 and 5
  // fields), it round-trips at fp16, and a mean state that does not fit the grid is refused.
  Model m = tiny_model();
  const auto r = tiny_run(m.h, 4);
  StartPoint sp;
  sp.controls = r.controls();
  sp.seed = 5;
  sp.time = 1.f;
  sp.coarse.assign(r.coarse.begin(), r.coarse.begin() + 8 * 8 * kPhys);
  m.starts.push_back(sp);
  quantise_like_storage(m);
  std::stringstream old_ss;
  ASSERT_TRUE(save_model(old_ss, m));
  EXPECT_EQ(old_ss.str()[8], 2);
  Model q = m;
  add_mean_state(q, 2, 9);
  q.detail.passes = 2;
  q.detail.soften = 0.1f;
  quantise_like_storage(q);
  std::stringstream ss;
  ASSERT_TRUE(save_model(ss, q));
  EXPECT_EQ(ss.str()[8], 6);
  EXPECT_EQ(ss.str().size(), old_ss.str().size() + 8 + 8 + 8 + 4 + 2 * q.mean.beta.size() + 4 + 4);  // the version 3, 4, 5 and 6 fields
  auto back = load_model(ss);
  ASSERT_TRUE(back) << back.error();
  EXPECT_EQ(back->mean.block, 2);
  EXPECT_EQ(back->mean.beta, q.mean.beta);
  EXPECT_EQ(back->detail.passes, 2);
  EXPECT_EQ(back->detail.soften, q.detail.soften);
  std::stringstream again;
  ASSERT_TRUE(save_model(again, *back));
  EXPECT_EQ(again.str(), ss.str());
  Model qb = m;  // a bias correction alone
  qb.bias = q.mean;
  std::stringstream bss;
  ASSERT_TRUE(save_model(bss, qb));
  EXPECT_EQ(bss.str()[8], 6);
  auto bback = load_model(bss);
  ASSERT_TRUE(bback) << bback.error();
  EXPECT_TRUE(bback->mean.empty());
  EXPECT_EQ(bback->bias.block, 2);
  EXPECT_EQ(bback->bias.beta, q.mean.beta);
  Model bad = q;
  bad.mean.block = 3;  // 8 cells are not whole blocks of 3
  std::stringstream bs;
  EXPECT_FALSE(save_model(bs, bad));
  bad = q;
  bad.mean.beta.pop_back();
  std::stringstream bs2;
  EXPECT_FALSE(save_model(bs2, bad));
}

TEST(Rollout, StudyMMeanStateMovesStartsToTheControls) {
  // A start point begun at its own controls is itself, to the bit; at other controls it is moved by mu(to) - mu(from),
  // heat and soot kept within the range; the mean state is the quadratic in the controls at every block's centre.
  Model m = runtime_model(16);
  const std::size_t n = 16 * 16 * kPhys;
  const std::vector<float> other{0.9f, 0.1f, 0.2f};
  std::vector<float> out(n);
  start_coarse(m, 0, other, out);  // no mean state: the start point's own
  EXPECT_EQ(out, m.starts[0].coarse);
  add_mean_state(m, 1, 4);
  for (const int index : {0, 1}) {
    start_coarse(m, index, m.starts[static_cast<std::size_t>(index)].controls, out);
    EXPECT_EQ(out, m.starts[static_cast<std::size_t>(index)].coarse);
  }
  std::vector<float> to(n), from(n);
  mean_state(m, other, to);
  mean_state(m, m.starts[1].controls, from);
  start_coarse(m, 1, other, out);
  int moved = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = i % kPhys;
    EXPECT_EQ(out[i], std::clamp(m.starts[1].coarse[i] + (to[i] - from[i]), m.lo[c], m.hi[c]));
    moved += out[i] != m.starts[1].coarse[i];
  }
  EXPECT_GT(moved, static_cast<int>(n) / 2);
  // blocks of one cell: the polynomial itself
  const float i0 = other[0], w0 = other[1], t0 = other[2];
  const std::array<float, kMeanFeatures> f{1.f, i0, w0, t0, i0 * i0, w0 * w0, t0 * t0, i0 * w0, i0 * t0, w0 * t0};
  for (const std::size_t k : {std::size_t{0}, std::size_t{37}, n - 1}) {
    float v = 0.f;
    for (int p = 0; p < kMeanFeatures; ++p) v += f[static_cast<std::size_t>(p)] * m.mean.beta[static_cast<std::size_t>(p) * n + k];
    EXPECT_NEAR(to[k], v, 1e-6f);
  }
  // rollout::start begins there (and grows fine fields from the moved state)
  const State s = rollout::start(m, 1, 32, other, 3);
  for (int i = 0; i < 16 * 16; ++i) {
    for (int c = 0; c < kPhys; ++c) EXPECT_EQ(s.coarse[static_cast<std::size_t>(i * m.h.channels() + c)], out[static_cast<std::size_t>(i * kPhys + c)]);
  }
}

TEST(Rollout, StudyMForceRoundTripsAndStirsWhereThereIsMaterial) {
  // Hyper::force alone also needs version 6; the forcing adds the curl of the noise's stream function to the velocity
  // where there is heat or soot and nowhere else, scaled with turbulence.
  Model m = runtime_model(16);
  m.h.force = 0.05f;
  std::stringstream ss;
  ASSERT_TRUE(save_model(ss, m));
  EXPECT_EQ(ss.str()[8], 6);
  auto back = load_model(ss);
  ASSERT_TRUE(back) << back.error();
  EXPECT_EQ(back->h.force, 0.05f);
  EXPECT_TRUE(back->mean.empty());
  Model bad = m;
  bad.h.force = -1.f;
  std::stringstream bs;
  EXPECT_FALSE(save_model(bs, bad));
  const int R = 16, C = m.h.channels();
  std::vector<float> noise(static_cast<std::size_t>(R) * R * kNoise);
  coarse_noise(m, 7, 1.f, noise);
  std::vector<float> state(static_cast<std::size_t>(R) * R * static_cast<std::size_t>(C), 0.f);
  for (int i = 0; i < R * R / 2; ++i) state[static_cast<std::size_t>(i * C + 2)] = 0.6f;  // heat in the lower half only
  std::vector<float> lo_t(3), hi_t(3);
  condition(m, std::vector<float>{0.5f, 0.5f, 0.1f}, 1.f, lo_t);
  condition(m, std::vector<float>{0.5f, 0.5f, 0.9f}, 1.f, hi_t);
  auto a = state, b = state;
  coarse_force(m, noise, lo_t, a);
  coarse_force(m, noise, hi_t, b);
  double moved = 0, ma = 0, mb = 0;
  for (int i = 0; i < R * R; ++i) {
    for (int c = 0; c < 2; ++c) {
      const float da = a[static_cast<std::size_t>(i * C + c)], db = b[static_cast<std::size_t>(i * C + c)];
      if (i >= R * R / 2) {
        EXPECT_EQ(da, 0.f);
        EXPECT_EQ(db, 0.f);
      } else {
        moved += da != 0.f;
        ma += std::abs(da);
        mb += std::abs(db);
      }
    }
    EXPECT_EQ(a[static_cast<std::size_t>(i * C + 2)], state[static_cast<std::size_t>(i * C + 2)]);
  }
  EXPECT_GT(moved, R * R / 2);
  EXPECT_NEAR(mb / ma, (1. + 6. * 0.9) / (1. + 6. * 0.1), 1e-3);
}

TEST(RolloutRuntime, InfoAndControls) {
  const Model m = runtime_model();
  Fx fx(m);
  nvfx_effect_info info{};
  ASSERT_EQ(nvfx_effect_get_info(fx.e, &info), NVFX_OK);
  EXPECT_EQ(info.arch, 3);
  EXPECT_EQ(info.n_variations, 2);
  EXPECT_EQ(info.n_controls, 3);
  EXPECT_STREQ(nvfx_effect_control_name(fx.e, 2), "turbulence");
  nvfx_instance* in = nullptr;
  EXPECT_EQ(nvfx_instance_create(fx.e, 24, &in), NVFX_ERROR_UNSUPPORTED);  // not a multiple of the coarse grid
  ASSERT_EQ(nvfx_instance_create(fx.e, 32, &in), NVFX_OK);
  EXPECT_GT(nvfx_instance_scratch_bytes(in), 0u);
  EXPECT_GT(nvfx_instance_macs_per_pixel(in), 0.0);
  EXPECT_EQ(nvfx_instance_set_variation(in, 2), NVFX_ERROR_ARGUMENT);
  EXPECT_EQ(nvfx_instance_set_variation(in, 1), NVFX_OK);
  nvfx_instance_free(in);
}

TEST(RolloutRuntime, MatchesTheReferenceOnEveryIsa) {
  const Model m = runtime_model();
  Fx fx(m);
  for (const nvfx_isa isa : {NVFX_ISA_BASELINE, NVFX_ISA_AVX2, NVFX_ISA_AVX512}) {
    if (nvfx_set_isa(isa) != NVFX_OK) continue;
    for (const int start_index : {0, 1}) {
      for (const int size : {32, 64}) {
        nvfx_instance* in = nullptr;
        ASSERT_EQ(nvfx_instance_create(fx.e, size, &in), NVFX_OK);
        ASSERT_EQ(nvfx_instance_set_precision(in, NVFX_PRECISION_FLOAT), NVFX_OK);  // the float path (int8: below)
        nvfx_instance_set_controls(in, m.starts[static_cast<std::size_t>(start_index)].controls.data(), 3);
        ASSERT_EQ(nvfx_instance_set_variation(in, start_index), NVFX_OK);
        std::vector<std::uint8_t> rt(static_cast<std::size_t>(size) * size * 4);
        for (const int frames : {0, 4}) {
          ASSERT_EQ(nvfx_render(in, frames / 30.0, rt.data(), static_cast<std::size_t>(size) * 4), NVFX_OK);
          const auto ref = reference_frame(m, start_index, size, frames);
          int worst = 0;
          std::size_t off = 0;
          for (std::size_t i = 0; i < ref.size(); ++i) {
            const int d = std::abs(int(ref[i]) - int(rt[i]));
            worst = std::max(worst, d);
            off += d > 1;
          }
          EXPECT_LE(worst, 3) << "isa " << isa << " start " << start_index << " size " << size << " frame " << frames;
          EXPECT_LE(off, ref.size() / 100) << "isa " << isa << " start " << start_index << " size " << size << " frame " << frames;
        }
        nvfx_instance_free(in);
      }
    }
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
}

TEST(RolloutRuntime, MatchesTheReferenceWithStudyD2sConstants) {
  // The advection gain and the softening (DetailSpec::advect, soften) in the runtime, as in the reference.
  Model m = runtime_model();
  m.detail.advect = 1.4f;
  m.detail.soften = 0.3f;
  m.detail.swirl = 1.2f;
  Fx fx(m);
  for (const nvfx_isa isa : {NVFX_ISA_BASELINE, NVFX_ISA_AVX2, NVFX_ISA_AVX512}) {
    if (nvfx_set_isa(isa) != NVFX_OK) continue;
    for (const int start_index : {0, 1}) {
      const int size = 64;
      nvfx_instance* in = nullptr;
      ASSERT_EQ(nvfx_instance_create(fx.e, size, &in), NVFX_OK);
      ASSERT_EQ(nvfx_instance_set_precision(in, NVFX_PRECISION_FLOAT), NVFX_OK);
      nvfx_instance_set_controls(in, m.starts[static_cast<std::size_t>(start_index)].controls.data(), 3);
      ASSERT_EQ(nvfx_instance_set_variation(in, start_index), NVFX_OK);
      std::vector<std::uint8_t> rt(static_cast<std::size_t>(size) * size * 4);
      for (const int frames : {4, 12}) {
        ASSERT_EQ(nvfx_render(in, frames / 30.0, rt.data(), static_cast<std::size_t>(size) * 4), NVFX_OK);
        const auto ref = reference_frame(m, start_index, size, frames);
        int worst = 0;
        std::size_t off = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
          const int d = std::abs(int(ref[i]) - int(rt[i]));
          worst = std::max(worst, d);
          off += d > 1;
        }
        EXPECT_LE(worst, 3) << "isa " << isa << " start " << start_index << " frame " << frames;
        EXPECT_LE(off, ref.size() / 100) << "isa " << isa << " start " << start_index << " frame " << frames;
      }
      nvfx_instance_free(in);
    }
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
  // and they change the frames: the same instance without them differs
  Model plain = runtime_model();
  plain.detail.swirl = 1.2f;
  EXPECT_NE(reference_frame(m, 1, 64, 12), reference_frame(plain, 1, 64, 12));
}

TEST(RolloutRuntime, MatchesTheReferenceWithStudyLsAdvection) {
  // MacCormack advection of the coarse state (Hyper::advection = 1) and the detail layer in passes (DetailSpec::passes,
  // with the swirl and the advection gain, which each pass shares) in the runtime, as in the reference, on every ISA.
  Model m = runtime_model();
  m.h.advection = 1;
  m.detail.passes = 2;
  m.detail.swirl = 1.2f;
  m.detail.advect = 1.3f;
  Fx fx(m);
  for (const nvfx_isa isa : {NVFX_ISA_BASELINE, NVFX_ISA_AVX2, NVFX_ISA_AVX512}) {
    if (nvfx_set_isa(isa) != NVFX_OK) continue;
    for (const int start_index : {0, 1}) {
      const int size = 64;
      nvfx_instance* in = nullptr;
      ASSERT_EQ(nvfx_instance_create(fx.e, size, &in), NVFX_OK);
      ASSERT_EQ(nvfx_instance_set_precision(in, NVFX_PRECISION_FLOAT), NVFX_OK);  // the float reference (int8: below)
      nvfx_instance_set_controls(in, m.starts[static_cast<std::size_t>(start_index)].controls.data(), 3);
      ASSERT_EQ(nvfx_instance_set_variation(in, start_index), NVFX_OK);
      std::vector<std::uint8_t> rt(static_cast<std::size_t>(size) * size * 4);
      for (const int frames : {4, 12}) {
        ASSERT_EQ(nvfx_render(in, frames / 30.0, rt.data(), static_cast<std::size_t>(size) * 4), NVFX_OK);
        const auto ref = reference_frame(m, start_index, size, frames);
        int worst = 0;
        std::size_t off = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
          const int d = std::abs(int(ref[i]) - int(rt[i]));
          worst = std::max(worst, d);
          off += d > 1;
        }
        EXPECT_LE(worst, 3) << "isa " << isa << " start " << start_index << " frame " << frames;
        EXPECT_LE(off, ref.size() / 100) << "isa " << isa << " start " << start_index << " frame " << frames;
      }
      nvfx_instance_free(in);
    }
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
  Model one_pass = m;
  one_pass.detail.passes = 1;
  EXPECT_NE(reference_frame(m, 1, 64, 12), reference_frame(one_pass, 1, 64, 12));
  Model sl = m;
  sl.h.advection = 0;
  EXPECT_NE(reference_frame(m, 1, 64, 12), reference_frame(sl, 1, 64, 12));
}

TEST(RolloutRuntime, MatchesTheReferenceWithStudyMsMeanState) {
  // A mean state (Model::mean): the instance plays a start point at controls other than its own, so the start is moved to
  // them, as rollout::start moves it, on every ISA; without the mean state the frames differ.
  Model m = runtime_model();
  add_mean_state(m, 2, 6);
  quantise_like_storage(m);
  const std::vector<float> other{0.85f, 0.25f, 0.4f};
  Fx fx(m);
  const auto reference = [&](const Model& mm, int start_index, int size, int frames) {
    const std::uint64_t seed = mm.starts[static_cast<std::size_t>(start_index)].seed;
    State s = rollout::start(mm, start_index, size, other, seed);
    for (int f = 0; f < frames; ++f) rollout::step(mm, s, other, seed);
    std::vector<float> rgba(static_cast<std::size_t>(size) * size * 4);
    rollout::render(mm, s, rgba);
    std::vector<std::uint8_t> out(rgba.size());
    for (std::size_t i = 0; i < rgba.size(); ++i) out[i] = static_cast<std::uint8_t>(rgba[i] * 255.f + 0.5f);
    return out;
  };
  for (const nvfx_isa isa : {NVFX_ISA_BASELINE, NVFX_ISA_AVX2, NVFX_ISA_AVX512}) {
    if (nvfx_set_isa(isa) != NVFX_OK) continue;
    for (const int start_index : {0, 1}) {
      const int size = 64;
      nvfx_instance* in = nullptr;
      ASSERT_EQ(nvfx_instance_create(fx.e, size, &in), NVFX_OK);
      ASSERT_EQ(nvfx_instance_set_precision(in, NVFX_PRECISION_FLOAT), NVFX_OK);
      nvfx_instance_set_controls(in, other.data(), 3);
      ASSERT_EQ(nvfx_instance_set_variation(in, start_index), NVFX_OK);
      std::vector<std::uint8_t> rt(static_cast<std::size_t>(size) * size * 4);
      for (const int frames : {0, 4, 12}) {
        ASSERT_EQ(nvfx_render(in, frames / 30.0, rt.data(), static_cast<std::size_t>(size) * 4), NVFX_OK);
        const auto ref = reference(m, start_index, size, frames);
        int worst = 0;
        std::size_t off = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
          const int d = std::abs(int(ref[i]) - int(rt[i]));
          worst = std::max(worst, d);
          off += d > 1;
        }
        EXPECT_LE(worst, 3) << "isa " << isa << " start " << start_index << " frame " << frames;
        EXPECT_LE(off, ref.size() / 100) << "isa " << isa << " start " << start_index << " frame " << frames;
      }
      nvfx_instance_free(in);
    }
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
  Model plain = m;
  plain.mean = {};
  EXPECT_NE(reference(m, 1, 64, 4), reference(plain, 1, 64, 4));
  EXPECT_NE(reference(m, 0, 64, 4), reference(plain, 0, 64, 4));
}

TEST(RolloutRuntime, MatchesTheReferenceWithStudyMsForce) {
  // The coarse stochastic forcing (Hyper::force), a mean state and a bias correction, in the runtime as in the
  // reference, on every ISA.
  Model m = runtime_model();
  m.h.force = 0.2f;
  add_mean_state(m, 2, 8);
  {
    Model b = runtime_model();
    add_mean_state(b, 4, 12);
    m.bias = b.mean;
    for (float& v : m.bias.beta) v *= 0.1f;
  }
  quantise_like_storage(m);
  Fx fx(m);
  for (const nvfx_isa isa : {NVFX_ISA_BASELINE, NVFX_ISA_AVX2, NVFX_ISA_AVX512}) {
    if (nvfx_set_isa(isa) != NVFX_OK) continue;
    for (const int start_index : {0, 1}) {
      const int size = 64;
      nvfx_instance* in = nullptr;
      ASSERT_EQ(nvfx_instance_create(fx.e, size, &in), NVFX_OK);
      ASSERT_EQ(nvfx_instance_set_precision(in, NVFX_PRECISION_FLOAT), NVFX_OK);
      nvfx_instance_set_controls(in, m.starts[static_cast<std::size_t>(start_index)].controls.data(), 3);
      ASSERT_EQ(nvfx_instance_set_variation(in, start_index), NVFX_OK);
      std::vector<std::uint8_t> rt(static_cast<std::size_t>(size) * size * 4);
      for (const int frames : {4, 12}) {
        ASSERT_EQ(nvfx_render(in, frames / 30.0, rt.data(), static_cast<std::size_t>(size) * 4), NVFX_OK);
        const auto ref = reference_frame(m, start_index, size, frames);
        int worst = 0;
        std::size_t off = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
          const int d = std::abs(int(ref[i]) - int(rt[i]));
          worst = std::max(worst, d);
          off += d > 1;
        }
        EXPECT_LE(worst, 3) << "isa " << isa << " start " << start_index << " frame " << frames;
        EXPECT_LE(off, ref.size() / 100) << "isa " << isa << " start " << start_index << " frame " << frames;
      }
      nvfx_instance_free(in);
    }
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
  Model calm = m;
  calm.h.force = 0.f;
  EXPECT_NE(reference_frame(m, 1, 64, 12), reference_frame(calm, 1, 64, 12));
  Model unbiased = m;
  unbiased.bias = {};
  EXPECT_NE(reference_frame(m, 1, 64, 12), reference_frame(unbiased, 1, 64, 12));
}

TEST(RolloutRuntime, ShardsMakeEveryFrameAFunctionOfTime) {
  const Model m = runtime_model();
  Fx fx(m);
  const int size = 32;
  std::vector<std::uint8_t> a(static_cast<std::size_t>(size) * size * 4), b(a.size()), c(a.size());
  nvfx_instance *played = nullptr, *fresh = nullptr;
  ASSERT_EQ(nvfx_instance_create(fx.e, size, &played), NVFX_OK);
  ASSERT_EQ(nvfx_instance_create(fx.e, size, &fresh), NVFX_OK);
  for (nvfx_instance* in : {played, fresh}) {
    nvfx_instance_set_seed(in, 99);
    nvfx_instance_set_drift(in, 2.f);  // 60-frame shards: frames 45-59 crossfade into the second shard
  }
  for (int f = 0; f <= 130; ++f) {
    nvfx_render(played, f / 30.0, a.data(), size * 4);
    if (f == 50 || f == 75 || f == 130) {
      nvfx_render(fresh, f / 30.0, b.data(), size * 4);  // a seek straight to the frame
      EXPECT_EQ(a, b) << "frame " << f;
    }
  }
  nvfx_render(fresh, 44 / 30.0, b.data(), size * 4);  // the last frame of the first shard alone
  nvfx_render(fresh, 52 / 30.0, c.data(), size * 4);  // inside the crossfade
  EXPECT_NE(b, c);
  nvfx_instance_free(played);
  nvfx_instance_free(fresh);
}

TEST(RolloutRuntime, SeeksAreDeterministicWithinAShard) {
  const Model m = runtime_model();
  Fx fx(m);
  const int size = 32;
  std::vector<std::uint8_t> a(static_cast<std::size_t>(size) * size * 4), b(a.size());
  nvfx_instance *played = nullptr, *fresh = nullptr;
  ASSERT_EQ(nvfx_instance_create(fx.e, size, &played), NVFX_OK);
  ASSERT_EQ(nvfx_instance_create(fx.e, size, &fresh), NVFX_OK);
  for (nvfx_instance* in : {played, fresh}) nvfx_instance_set_seed(in, 4242);
  for (int f = 0; f <= 40; ++f) nvfx_render(played, f / 30.0, a.data(), size * 4);  // played frame by frame
  nvfx_render(fresh, 40 / 30.0, b.data(), size * 4);                                // one seek
  EXPECT_EQ(a, b);
  nvfx_render(played, 10 / 30.0, a.data(), size * 4);  // backwards: restarts from the start point
  nvfx_render(fresh, 10 / 30.0, b.data(), size * 4);
  EXPECT_EQ(a, b);
  nvfx_instance_set_seed(fresh, 7);  // another seed: another run
  nvfx_render(fresh, 0.0, b.data(), size * 4);
  nvfx_render(fresh, 30 / 30.0, b.data(), size * 4);
  nvfx_render(played, 30 / 30.0, a.data(), size * 4);
  EXPECT_NE(a, b);
  nvfx_instance_free(played);
  nvfx_instance_free(fresh);
}

TEST(RolloutRuntime, OneShotsHoldTheirLastFrameAndBakesLoop) {
  Model m = runtime_model();
  m.loop = false;
  m.h.frames = 12;
  m.h.n_age = 2;
  m.step_w = init_model(m.h, 5).step_w;  // the FiLM input grew by the age features
  {
    Fx fx(m);
    const int size = 32;
    nvfx_instance* in = nullptr;
    ASSERT_EQ(nvfx_instance_create(fx.e, size, &in), NVFX_OK);
    std::vector<std::uint8_t> a(static_cast<std::size_t>(size) * size * 4), b(a.size());
    nvfx_render(in, 11 / 30.0, a.data(), size * 4);
    nvfx_render(in, 5.0, b.data(), size * 4);
    EXPECT_EQ(a, b);
    nvfx_instance_free(in);
  }
  const Model loop = runtime_model();
  Fx fx(loop);
  const int size = 32, frames = 20;
  nvfx_instance* in = nullptr;
  ASSERT_EQ(nvfx_instance_create(fx.e, size, &in), NVFX_OK);
  std::vector<std::uint8_t> baked(static_cast<std::size_t>(size) * size * 4 * frames), one(static_cast<std::size_t>(size) * size * 4);
  ASSERT_EQ(nvfx_bake(in, frames, baked.data()), NVFX_OK);
  // frames after the crossfade are the plain rollout
  nvfx_render(in, 15 / 30.0, one.data(), size * 4);
  EXPECT_TRUE(std::equal(one.begin(), one.end(), baked.begin() + static_cast<std::ptrdiff_t>(one.size() * 15)));
  nvfx_instance_free(in);
}

// --- study K: the int8 path of rollout effects (docs/REPORT.md §13) -----------------------------------------------------

#include "nvfx_internal.hpp"

namespace {

struct Diff {
  int worst = 0;
  std::size_t off = 0;  // channels more than one level away
  double mean = 0;
};
Diff diff(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  Diff d;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const int x = std::abs(int(a[i]) - int(b[i]));
    d.worst = std::max(d.worst, x);
    d.off += x > 1;
    d.mean += x;
  }
  d.mean /= static_cast<double>(a.size());
  return d;
}

// An instance of start point `start_index` at int8 with study K's candidate `convs`, frame `frames` of its timeline.
std::vector<std::uint8_t> int8_frame(nvfx_effect* e, const rollout::Model& m, int start_index, int size, int frames, nfx::rt::Int8Convs convs) {
  nvfx_instance* in = nullptr;
  EXPECT_EQ(nvfx_instance_create(e, size, &in), NVFX_OK);
  EXPECT_EQ(nfx::rt::set_rollout_int8_convs(in, convs), NVFX_OK);
  EXPECT_EQ(nvfx_instance_set_precision(in, NVFX_PRECISION_INT8), NVFX_OK);
  nvfx_instance_set_controls(in, m.starts[static_cast<std::size_t>(start_index)].controls.data(), 3);
  EXPECT_EQ(nvfx_instance_set_variation(in, start_index), NVFX_OK);
  std::vector<std::uint8_t> rt(static_cast<std::size_t>(size) * size * 4);
  EXPECT_EQ(nvfx_render(in, frames / 30.0, rt.data(), static_cast<std::size_t>(size) * 4), NVFX_OK);
  nvfx_instance_free(in);
  return rt;
}

}  // namespace

// The int8 path against its scalar reference (rollout::Int8: the same quantisation in plain C++), on every ISA and for
// both of study K's candidates. Every kernel computes the same codes and integer sums, so the ISAs' frames are held to
// one another with the float path's tolerance (no channel more than 3 levels apart, at most 1% more than one). Against
// the reference, which applies FiLM after the convolution where the runtime folds it into the weights, a last-bit
// difference can move a code by one on a rounding boundary, which shows in a few channels (5 levels in 2 channels of
// 16,384 once here); so there: at most 8 levels, at most 1% of channels more than one level away, and less than 0.05
// of a level on average. Grids of 16 and 32 cells (the AVX-512 kernel's two vectors per row), sizes 32 and 64, a start
// point grown from its coarse state and one with fine fields, frames 0 (with the warm-up), 4 and 12.
TEST(RolloutRuntime, Int8MatchesItsReferenceOnEveryIsa) {
  int compared = 0, isas = 0;
  for (const int res : {16, 32}) {
    const rollout::Model m = runtime_model(res);
    Fx fx(m);
    for (const auto convs : {nfx::rt::Int8Convs::second, nfx::rt::Int8Convs::both}) {
      const Int8 q = convs == nfx::rt::Int8Convs::both ? Int8::both : Int8::second;
      for (const int start_index : {0, 1}) {
        for (const int size : {32, 64}) {
          for (const int frames : {0, 4, 12}) {
            const auto ref = reference_frame(m, start_index, size, frames, q);
            std::vector<std::uint8_t> first;
            for (const nvfx_isa isa : {NVFX_ISA_BASELINE, NVFX_ISA_AVX2, NVFX_ISA_AVX512}) {
              if (nvfx_set_isa(isa) != NVFX_OK) continue;
              const auto rt = int8_frame(fx.e, m, start_index, size, frames, convs);
              const std::string at = std::format("res {} convs {} isa {} start {} size {} frame {}", res, static_cast<int>(convs), static_cast<int>(isa), start_index, size, frames);
              const Diff d = diff(rt, ref);
              EXPECT_LE(d.worst, 8) << at;
              EXPECT_LE(d.off, rt.size() / 100) << at;
              EXPECT_LT(d.mean, 0.05) << at;
              if (first.empty()) {
                first = rt;
              } else {
                const Diff x = diff(rt, first);
                EXPECT_LE(x.worst, 3) << at << " (against the baseline ISA)";
                EXPECT_LE(x.off, rt.size() / 100) << at << " (against the baseline ISA)";
                ++isas;
              }
              ++compared;
            }
          }
        }
      }
    }
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
  EXPECT_GE(compared, 48);
  EXPECT_GE(isas, 24);
}

// The int8 reference is not the float one, but close to it on the test model (whose weights are larger than a trained
// model's): the quantisation's effect after 12 frames, every channel within a few levels on average.
TEST(RolloutRuntime, Int8ReferenceStaysCloseToFloat) {
  const rollout::Model m = runtime_model(32);
  for (const Int8 q : {Int8::second, Int8::both}) {
    for (const int start_index : {0, 1}) {
      const auto fl = reference_frame(m, start_index, 64, 12);
      const auto q8 = reference_frame(m, start_index, 64, 12, q);
      const Diff d = diff(fl, q8);
      EXPECT_GT(d.worst, 0) << static_cast<int>(q);
      EXPECT_LT(d.mean, 2.0) << static_cast<int>(q) << " start " << start_index;
    }
  }
}

// nvfx_instance_set_precision on rollout instances: DEFAULT is int8 where the CPU has AVX-512 VNNI and float elsewhere
// (the frames of INT8 or FLOAT to the bit, study K's rule and its cost condition), INT8
// the int8 path (the candidate Precision::int8 means), FLOAT the float path, and back to INT8 gives the int8 frames
// again; a change starts a new timeline. Two int8 instances give the same frames, seeks included.
TEST(RolloutRuntime, PrecisionIsASettingOfRolloutInstances) {
  const rollout::Model m = runtime_model(32);
  Fx fx(m);
  const int size = 64;
  std::vector<std::uint8_t> a(static_cast<std::size_t>(size) * size * 4), b(a.size()), c(a.size());
  nvfx_instance *def = nullptr, *fl = nullptr, *q = nullptr, *q2 = nullptr;
  for (nvfx_instance** in : {&def, &fl, &q, &q2}) {
    ASSERT_EQ(nvfx_instance_create(fx.e, size, in), NVFX_OK);
    nvfx_instance_set_seed(*in, 31);
  }
  ASSERT_EQ(nvfx_instance_set_precision(fl, NVFX_PRECISION_FLOAT), NVFX_OK);
  ASSERT_EQ(nvfx_instance_set_precision(q, NVFX_PRECISION_INT8), NVFX_OK);
  ASSERT_EQ(nvfx_instance_set_precision(q2, NVFX_PRECISION_INT8), NVFX_OK);
  nvfx_render(def, 20 / 30.0, a.data(), size * 4);
  nvfx_render(q, 20 / 30.0, c.data(), size * 4);
  nvfx_render(fl, 20 / 30.0, b.data(), size * 4);
  EXPECT_NE(b, c);
  // the default for rollout effects: int8 where the CPU has AVX-512 VNNI, float elsewhere (study K's cost condition)
  const bool int8_default = nfx::rt::default_rollout_precision() == nfx::rt::Precision::int8;
  EXPECT_EQ(a, int8_default ? c : b);
  for (int f = 0; f <= 20; ++f) nvfx_render(q2, f / 30.0, b.data(), size * 4);  // played frame by frame
  EXPECT_EQ(b, c);
  EXPECT_GT(nvfx_instance_scratch_bytes(q), 0u);
  ASSERT_EQ(nvfx_instance_set_precision(q, NVFX_PRECISION_FLOAT), NVFX_OK);  // float, and back
  nvfx_render(q, 20 / 30.0, b.data(), size * 4);
  nvfx_render(fl, 20 / 30.0, c.data(), size * 4);
  EXPECT_EQ(b, c);
  ASSERT_EQ(nvfx_instance_set_precision(q, NVFX_PRECISION_DEFAULT), NVFX_OK);
  nvfx_render(q, 20 / 30.0, c.data(), size * 4);
  EXPECT_EQ(a, c);
  EXPECT_EQ(nfx::rt::set_rollout_int8_convs(nullptr, nfx::rt::Int8Convs::both), NVFX_ERROR_ARGUMENT);
  for (nvfx_instance* in : {def, fl, q, q2}) nvfx_instance_free(in);
}

namespace {

// Forces an ISA (nvfx_set_isa) for the instances created while it lives, then gives the choice back to the runtime.
struct ForcedIsa {
  bool ok;
  explicit ForcedIsa(nvfx_isa isa) : ok(nvfx_set_isa(isa) == NVFX_OK) {}
  ~ForcedIsa() { nvfx_set_isa(NVFX_ISA_AUTO); }
  ForcedIsa(const ForcedIsa&) = delete;
  ForcedIsa& operator=(const ForcedIsa&) = delete;
};

// The builds this CPU runs, from nvfx_version's features.
struct Cpu {
  bool avx2 = false, avx512 = false, vnni = false;
  Cpu() {
    std::uint32_t f = 0;
    nvfx_version(&f);
    avx2 = (f & NVFX_FEATURE_AVX2) != 0;
    avx512 = (f & NVFX_FEATURE_AVX512) != 0;
    vnni = (f & NVFX_FEATURE_VNNI) != 0;
  }
  nvfx_isa best() const { return avx2 ? NVFX_ISA_AVX2 : NVFX_ISA_BASELINE; }
  nvfx_isa float_isa() const { return avx512 ? NVFX_ISA_AVX512 : best(); }        // docs/REPORT.md §13.10
  nvfx_isa int8_isa() const { return avx512 && vnni ? NVFX_ISA_AVX512 : best(); }  // study K
};

}  // namespace

// With no ISA forced, float rollout instances take the AVX-512 build where the CPU has AVX-512 (VNNI is not needed),
// else AVX2, else the baseline (docs/REPORT.md §13.10): the ISA the runtime gives a float runner (scenes, the
// library's stages and the effect manager take theirs from it) and the one a new instance reports, at FLOAT and at
// DEFAULT where that is float. Int8 keeps study K's rule (AVX-512 only with VNNI), a forced ISA wins at either
// precision, and frame models keep theirs (nvfx_get_isa: AVX2, else the baseline).
TEST(RolloutRuntime, FloatTakesTheAvx512BuildWhereTheCpuHasIt) {
  const Cpu cpu;
  ASSERT_EQ(nvfx_set_isa(NVFX_ISA_AUTO), NVFX_OK);
  EXPECT_EQ(nfx::rt::rollout_isa(nfx::rt::Precision::float32), cpu.float_isa());
  EXPECT_EQ(nfx::rt::rollout_isa(nfx::rt::Precision::int8), cpu.int8_isa());
  EXPECT_EQ(nvfx_get_isa(), cpu.best());
  const rollout::Model m = runtime_model(32);
  Fx fx(m);
  nvfx_instance* in = nullptr;
  ASSERT_EQ(nvfx_instance_create(fx.e, 64, &in), NVFX_OK);
  const bool int8_default = nfx::rt::default_rollout_precision() == nfx::rt::Precision::int8;
  EXPECT_EQ(nfx::rt::rollout_instance_isa(in), int8_default ? cpu.int8_isa() : cpu.float_isa()) << "DEFAULT";
  ASSERT_EQ(nvfx_instance_set_precision(in, NVFX_PRECISION_FLOAT), NVFX_OK);
  EXPECT_EQ(nfx::rt::rollout_instance_isa(in), cpu.float_isa()) << "FLOAT";
  ASSERT_EQ(nvfx_instance_set_precision(in, NVFX_PRECISION_INT8), NVFX_OK);
  EXPECT_EQ(nfx::rt::rollout_instance_isa(in), cpu.int8_isa()) << "INT8";
  nvfx_instance_free(in);
  int forced = 0;
  for (const nvfx_isa isa : {NVFX_ISA_BASELINE, NVFX_ISA_AVX2, NVFX_ISA_AVX512}) {
    const ForcedIsa pin(isa);
    if (!pin.ok) continue;
    EXPECT_EQ(nfx::rt::rollout_isa(nfx::rt::Precision::float32), isa);
    EXPECT_EQ(nfx::rt::rollout_isa(nfx::rt::Precision::int8), isa);
    ASSERT_EQ(nvfx_instance_create(fx.e, 64, &in), NVFX_OK);
    for (const nvfx_precision p : {NVFX_PRECISION_DEFAULT, NVFX_PRECISION_FLOAT, NVFX_PRECISION_INT8}) {
      ASSERT_EQ(nvfx_instance_set_precision(in, p), NVFX_OK);
      EXPECT_EQ(nfx::rt::rollout_instance_isa(in), isa) << "forced " << static_cast<int>(isa) << ", precision " << static_cast<int>(p);
    }
    nvfx_instance_free(in);
    ++forced;
  }
  EXPECT_GE(forced, 1);
  EXPECT_EQ(nfx::rt::rollout_instance_isa(nullptr), NVFX_ISA_AUTO);
}

// A float instance with no ISA forced, on the build it takes (AVX-512 where the CPU has it, docs/REPORT.md §13.10),
// against the float reference with the tolerances of MatchesTheReferenceOnEveryIsa (no channel more than 3 levels off,
// at most 1% more than one level), and to the bit the frames of an instance with that build forced. Grids of 16 and 32
// cells (the AVX-512 kernels' one and two vectors per row), sizes 32 and 64, a start point grown from its coarse state
// and one with fine fields, frames 0 (with the warm-up), 4 and 12.
TEST(RolloutRuntime, DefaultFloatMatchesTheReference) {
  const Cpu cpu;
  ASSERT_EQ(nvfx_set_isa(NVFX_ISA_AUTO), NVFX_OK);
  const auto float_frame = [](nvfx_effect* e, const rollout::Model& m, int start_index, int size, int frames, nvfx_isa isa) {
    const ForcedIsa pin(isa);
    EXPECT_TRUE(pin.ok);
    nvfx_instance* in = nullptr;
    EXPECT_EQ(nvfx_instance_create(e, size, &in), NVFX_OK);
    std::vector<std::uint8_t> rt(static_cast<std::size_t>(size) * size * 4);
    if (!in) return rt;
    EXPECT_EQ(nvfx_instance_set_precision(in, NVFX_PRECISION_FLOAT), NVFX_OK);
    nvfx_instance_set_controls(in, m.starts[static_cast<std::size_t>(start_index)].controls.data(), 3);
    EXPECT_EQ(nvfx_instance_set_variation(in, start_index), NVFX_OK);
    EXPECT_EQ(nvfx_render(in, frames / 30.0, rt.data(), static_cast<std::size_t>(size) * 4), NVFX_OK);
    nvfx_instance_free(in);
    return rt;
  };
  int compared = 0;
  for (const int res : {16, 32}) {
    const rollout::Model m = runtime_model(res);
    Fx fx(m);
    for (const int start_index : {0, 1}) {
      for (const int size : {32, 64}) {
        for (const int frames : {0, 4, 12}) {
          const auto rt = float_frame(fx.e, m, start_index, size, frames, NVFX_ISA_AUTO);
          const auto ref = reference_frame(m, start_index, size, frames);
          const std::string at = std::format("res {} start {} size {} frame {}", res, start_index, size, frames);
          const Diff d = diff(rt, ref);
          EXPECT_LE(d.worst, 3) << at;
          EXPECT_LE(d.off, rt.size() / 100) << at;
          EXPECT_EQ(rt, float_frame(fx.e, m, start_index, size, frames, cpu.float_isa())) << at << ": not the build it takes";
          ++compared;
        }
      }
    }
  }
  EXPECT_EQ(compared, 24);
}

// Study L's options at int8 (study K): MacCormack advection and the detail layer in two passes (the new magic has
// both) with the stepper's second convolution in 8 bits, against the int8 reference on every ISA, with the tolerances
// of Int8MatchesItsReferenceOnEveryIsa.
TEST(RolloutRuntime, Int8MatchesItsReferenceWithStudyLsOptions) {
  rollout::Model m = runtime_model(32);
  m.h.advection = 1;
  m.detail.passes = 2;
  m.detail.swirl = 1.2f;
  m.detail.advect = 1.3f;
  Fx fx(m);
  int compared = 0;
  for (const int start_index : {0, 1}) {
    for (const int frames : {4, 12}) {
      const auto ref = reference_frame(m, start_index, 64, frames, Int8::second);
      for (const nvfx_isa isa : {NVFX_ISA_BASELINE, NVFX_ISA_AVX2, NVFX_ISA_AVX512}) {
        if (nvfx_set_isa(isa) != NVFX_OK) continue;
        const auto rt = int8_frame(fx.e, m, start_index, 64, frames, nfx::rt::Int8Convs::second);
        const Diff d = diff(rt, ref);
        const std::string at = std::format("isa {} start {} frame {}", static_cast<int>(isa), start_index, frames);
        EXPECT_LE(d.worst, 8) << at;
        EXPECT_LE(d.off, rt.size() / 100) << at;
        EXPECT_LT(d.mean, 0.05) << at;
        ++compared;
      }
    }
  }
  nvfx_set_isa(NVFX_ISA_AUTO);
  EXPECT_GE(compared, 8);
}
