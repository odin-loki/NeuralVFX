// Study J's estimation path (include/neuralfx/footage.hpp): degradations as ingest makes them, the inverse network's
// format and features, its training on a simple map, refinement gradients against finite differences, block-matched
// motion, assimilation through the stepper, and effect files with estimated start points that the runtime plays.
#include <neuralfx/footage.hpp>
#include <neuralfx/nvfx.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

using namespace nfx;
using namespace nfx::footage;

namespace {

std::size_t sz(int v) { return static_cast<std::size_t>(v); }
float fl(int v) { return static_cast<float>(v); }

// A small rollout effect (grid 8, frames of 32) with one start point and random weights.
rollout::Model small_model(std::uint64_t seed = 5) {
  rollout::Hyper h;
  h.res = 8;
  h.hidden = 6;
  h.memory = 2;
  h.jacobi = 6;
  h.render_hidden = 5;
  h.warmup = 2;
  rollout::Model m = rollout::init_model(h, seed);
  std::mt19937_64 rng(seed + 1);
  std::normal_distribution<float> nd(0.f, 0.1f);
  for (float& w : m.step_w) w += nd(rng);
  // A renderer that draws: the first hidden layer passes fine and coarse heat and soot and one directional sum, the
  // second passes them on, and the output mixes them (red and green from heat, alpha from soot), with a little noise.
  const rollout::RenderLayout L = rollout::render_layout(m.h);
  const int H = m.h.render_hidden;
  std::ranges::fill(m.render_w, 0.f);
  for (int j = 0; j < H; ++j) {
    m.render_w[L.w1 + sz(j) * rollout::kRenderIn + sz(j)] = 1.f;
    m.render_w[L.w2 + sz(j) * sz(H) + sz(j)] = 1.f;
  }
  const float mix[4][5] = {{0.5f, 0.05f, 0.1f, 0.f, 0.f}, {0.3f, 0.2f, 0.f, 0.05f, 0.f}, {0.05f, 0.1f, 0.f, 0.f, 0.05f}, {0.f, 0.6f, 0.f, 0.1f, 0.02f}};
  for (int c = 0; c < 4; ++c) {
    for (int j = 0; j < H; ++j) m.render_w[L.wo + sz(c) * sz(H) + sz(j)] = mix[c][j];
  }
  std::normal_distribution<float> small(0.f, 0.03f);
  for (float& w : m.render_w) w += small(rng);
  m.effect = "fire";
  m.control_names = {"intensity", "wind", "turbulence"};
  m.scale = {0.2f, 0.2f, 0.4f, 0.3f};
  m.lo = {-2.f, -2.f, 0.f, 0.f};
  m.hi = {2.f, 2.f, 3.f, 3.f};
  m.render_scale = {0.8f, 0.6f};
  rollout::StartPoint sp;
  sp.controls = {0.5f, 0.5f, 0.5f};
  sp.seed = 9;
  sp.time = 1.f;
  sp.coarse.assign(sz(h.res) * sz(h.res) * rollout::kPhys, 0.1f);
  m.starts.push_back(sp);
  return m;
}

// Smooth fields with structure at every scale the tests need (a few blobs), rows from the bottom.
Fields blobs(int S, float dx = 0.f, float dy = 0.f, float amp = 1.f) {
  Fields f;
  f.size = S;
  f.heat.assign(sz(S) * sz(S), 0.f);
  f.soot.assign(sz(S) * sz(S), 0.f);
  const float c[4][3] = {{0.3f, 0.35f, 0.08f}, {0.6f, 0.55f, 0.1f}, {0.45f, 0.7f, 0.06f}, {0.7f, 0.3f, 0.07f}};
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      float h = 0.f, d = 0.f;
      for (int k = 0; k < 4; ++k) {
        const float ex = (fl(x) - dx) / fl(S) - c[k][0], ey = (fl(y) - dy) / fl(S) - c[k][1];
        const float g = std::exp(-(ex * ex + ey * ey) / (2.f * c[k][2] * c[k][2]));
        h += (k % 2 ? 0.8f : 0.5f) * g;
        d += (k % 2 ? 0.3f : 0.6f) * g;
      }
      f.heat[sz(y) * sz(S) + sz(x)] = amp * h;
      f.soot[sz(y) * sz(S) + sz(x)] = amp * d;
    }
  }
  return f;
}

}  // namespace

TEST(Footage, DegradeDropsAlphaAsIngestAndNoiseIsDeterministic) {
  const int S = 16;
  std::vector<std::uint8_t> a(sz(S) * S * 4);
  for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<std::uint8_t>((i * 37) % 200);
  std::vector<std::uint8_t> b = a;
  degrade(b, S, {0.f, 0.f, true}, 1);
  for (std::size_t i = 0; i < b.size(); i += 4) {
    EXPECT_EQ(b[i], a[i]);
    EXPECT_EQ(b[i + 3], std::max({a[i], a[i + 1], a[i + 2]}));
  }
  std::vector<std::uint8_t> c1 = a, c2 = a, c3 = a;
  degrade(c1, S, {1.f, 2.f / 255.f, false}, 7);
  degrade(c2, S, {1.f, 2.f / 255.f, false}, 7);
  degrade(c3, S, {1.f, 2.f / 255.f, false}, 8);
  EXPECT_EQ(c1, c2);
  EXPECT_NE(c1, c3);
}

TEST(Footage, InverseRoundTripsAndFeaturesSeeTheRightPixel) {
  InverseSpec spec;
  spec.hidden = 7;
  spec.levels = 3;
  Inverse inv = init_inverse(spec, 3);
  inv.effect = "smoke";
  inv.size = 16;
  inv.scale = {0.7f, 1.3f};
  std::stringstream ss;
  ASSERT_TRUE(save_inverse(ss, inv));
  auto back = load_inverse(ss);
  ASSERT_TRUE(back) << back.error();
  EXPECT_EQ(back->w, inv.w);
  EXPECT_EQ(back->effect, "smoke");
  EXPECT_EQ(back->scale, inv.scale);
  EXPECT_EQ(back->inputs(), 3 * 9 * 4 + 2);
  // One lit pixel at frame row 2 (top), column 5: field cell (5, 16 - 1 - 2).
  const int S = 16;
  Frame fr(sz(S) * S * 4, 0.f);
  fr[(sz(2) * S + 5) * 4 + 1] = 0.75f;
  const Features F(inv, fr, S);
  std::vector<float> f(sz(inv.inputs()));
  F.at(5, 13, f);
  EXPECT_FLOAT_EQ(f[4 * 4 + 1], 0.75f);  // level 0, centre tap (dy = 0, dx = 0 is tap 4), green
  EXPECT_FLOAT_EQ(f[4 * 4 + 0], 0.f);
  F.at(6, 13, f);
  EXPECT_FLOAT_EQ(f[3 * 4 + 1], 0.75f);  // the left neighbour tap
  EXPECT_NEAR(f[f.size() - 2], (6.5f / 16.f) * 2.f - 1.f, 1e-6f);
  EXPECT_NEAR(f[f.size() - 1], (13.5f / 16.f) * 2.f - 1.f, 1e-6f);
  // Without alpha the alpha channel is not seen.
  spec.alpha = false;
  Inverse rgb = init_inverse(spec, 3);
  EXPECT_EQ(rgb.inputs(), 3 * 9 * 3 + 2);
}

TEST(Footage, InverseLearnsASimpleRenderer) {
  // frames: alpha = 1 - exp(-2 soot), red = heat / 2: the inverse network should learn to read both back.
  const int S = 32;
  std::vector<InverseSample> samples;
  for (int k = 0; k < 6; ++k) {
    InverseSample s;
    s.size = S;
    s.truth = blobs(S, fl(3 * k) - 6.f, fl(2 * k) - 4.f, 0.6f + 0.15f * fl(k));
    s.frame.assign(sz(S) * S * 4, 0.f);
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) {
        const std::size_t i = sz(y) * sz(S) + sz(x);
        float* p = s.frame.data() + (sz(S - 1 - y) * sz(S) + sz(x)) * 4;
        p[0] = std::min(1.f, 0.5f * s.truth.heat[i]);
        p[3] = 1.f - std::exp(-2.f * s.truth.soot[i]);
      }
    }
    samples.push_back(std::move(s));
  }
  InverseSpec spec;
  spec.hidden = 16;
  spec.levels = 2;
  Inverse inv = init_inverse(spec, 4);
  InverseTrainOptions o;
  o.iterations = 600;
  o.batch = 256;
  o.lr = 5e-3f;
  train_inverse(inv, std::span(samples).subspan(0, 5), o);
  const Fields f = apply_inverse(inv, samples[5].frame, S);
  EXPECT_GT(field_psnr(f.heat, samples[5].truth.heat, inv.scale[0]), 25.0);
  EXPECT_GT(field_psnr(f.soot, samples[5].truth.soot, inv.scale[1]), 25.0);
}

TEST(Footage, RefineGradientMatchesFiniteDifferences) {
  const rollout::Model m = small_model();
  const int S = 16;
  const Fields start = blobs(S, 0.f, 0.f, 0.7f);
  Fields f = blobs(S, 1.5f, -1.f, 0.8f);
  Frame target(sz(S) * S * 4);
  render_fields(m, start, target);
  for (float& v : target) v = std::clamp(v * 0.9f + 0.02f, 0.f, 1.f);
  for (const bool alpha : {true, false}) {
    RefineOptions o;
    o.alpha = alpha;
    o.prior = 0.3f;
    Fields g;
    refine_loss(m, target, f, start, o, &g);
    std::mt19937 rng(5);
    int checked = 0, bad = 0;
    for (int t = 0; t < 40; ++t) {
      const int ch = t % 2;
      std::vector<float>& v = ch == 0 ? f.heat : f.soot;
      std::size_t i = rng() % (sz(S) * S);
      while (v[i] < 0.05f) i = rng() % (sz(S) * S);  // away from the gate's kink at zero
      const float e = 1e-3f * (ch == 0 ? m.render_scale[0] : m.render_scale[1]), keep = v[i];
      v[i] = keep + e;
      const double lp = refine_loss(m, target, f, start, o, nullptr);
      v[i] = keep + 2.f * e;
      const double lp2 = refine_loss(m, target, f, start, o, nullptr);
      v[i] = keep - e;
      const double lm = refine_loss(m, target, f, start, o, nullptr);
      v[i] = keep;
      const double l0 = refine_loss(m, target, f, start, o, nullptr);
      const double fd = (lp - lm) / (2.0 * e);
      const bool kink = std::abs((lp2 - lp) - (lp - l0)) > 0.2 * std::abs(lp - l0) + 1e-12 || std::abs((lp - l0) - (l0 - lm)) > 0.2 * std::abs(lp - l0) + 1e-12;
      if (kink) continue;  // a ReLU or clamp kink inside the step
      const double an = (ch == 0 ? g.heat : g.soot)[i];
      ++checked;
      if (std::abs(fd - an) > 0.03 * std::max(std::abs(fd), std::abs(an)) + 1e-7) ++bad;
    }
    EXPECT_GE(checked, 20);
    EXPECT_LE(bad, 1) << "alpha " << alpha;
  }
}

TEST(Footage, RefineMovesTheFieldsTowardsWhatTheRendererDrew) {
  const rollout::Model m = small_model();
  const int S = 16;
  const Fields truth = blobs(S, 0.f, 0.f, 0.7f);
  Frame target(sz(S) * S * 4), drawn(target.size());
  render_fields(m, truth, target);
  Fields f = blobs(S, 1.f, 1.f, 0.4f);
  render_fields(m, f, drawn);
  double before = 0;
  for (std::size_t i = 0; i < target.size(); ++i) before += (drawn[i] - target[i]) * (drawn[i] - target[i]) / static_cast<double>(target.size());
  RefineOptions o;
  o.iterations = 80;
  o.prior = 0.f;
  const double after = refine_fields(m, target, f, o);
  EXPECT_LT(after, 0.5 * before);
  for (const float v : f.heat) EXPECT_GE(v, 0.f);
}

TEST(Footage, BlockFlowFindsAShift) {
  const int S = 64, R = 16;
  const Fields a = blobs(S, 0.f, 0.f), b = blobs(S, 2.f, -1.f);
  FlowOptions o;
  o.min_mass = 0.05f;
  const Flow f = block_flow(a, b, R, {1.f, 1.f}, o);
  int known = 0;
  double eu = 0, ev = 0;
  for (int c = 0; c < R * R; ++c) {
    if (!f.known[sz(c)]) {
      EXPECT_EQ(f.uv[sz(c) * 2], 0.f);
      continue;
    }
    ++known;
    eu += std::abs(f.uv[sz(c) * 2] - 2.f / 4.f);
    ev += std::abs(f.uv[sz(c) * 2 + 1] + 1.f / 4.f);
  }
  ASSERT_GT(known, 20);
  EXPECT_LT(eu / known, 0.05);  // coarse cells per frame (4 pixels per cell)
  EXPECT_LT(ev / known, 0.05);
  Flow g = f;
  finish_flow(g, o);
  EXPECT_EQ(g.known, f.known);
  for (int c = 0; c < R * R; ++c) EXPECT_TRUE(std::isfinite(g.uv[sz(c) * 2]));
  const Flow mean = mean_flow(std::vector<Flow>{f, f});
  EXPECT_EQ(mean.uv, f.uv);
}

TEST(Footage, AssimilationKeepsWhatWasSeen) {
  const rollout::Model m = small_model();
  const int S = 32, R = m.h.res;
  std::vector<Fields> fields = {blobs(S, 0.f, 0.f), blobs(S, 1.f, 0.f), blobs(S, 2.f, 0.f)};
  std::vector<Flow> flows(3);
  for (int i = 1; i < 3; ++i) {
    flows[sz(i)] = block_flow(fields[sz(i - 1)], fields[sz(i)], R, m.render_scale, {});
    finish_flow(flows[sz(i)], {});
  }
  AssimOptions o;
  o.time = 2.f;
  const std::vector<float> a = assimilate(m, fields, flows, o), b = assimilate(m, fields, flows, o);
  EXPECT_EQ(a, b);
  const std::vector<float> c = coarse_fields(fields.back(), R);
  for (int j = 0; j < R * R; ++j) {
    EXPECT_FLOAT_EQ(a[sz(j) * 4 + 2], c[sz(j) * 2]);
    EXPECT_FLOAT_EQ(a[sz(j) * 4 + 3], c[sz(j) * 2 + 1]);
  }
  // One frame and no flow: no motion to measure, velocity zero. With the flow into that frame: its velocity.
  const std::vector<float> one = assimilate(m, std::span(fields).subspan(2), std::span<const Flow>{}, o);
  for (int j = 0; j < R * R; ++j) EXPECT_EQ(one[sz(j) * 4], 0.f);
  const std::vector<float> moved = assimilate(m, std::span(fields).subspan(2), std::span(flows).subspan(2), o);
  for (int j = 0; j < R * R; ++j) EXPECT_FLOAT_EQ(moved[sz(j) * 4], std::clamp(flows[2].uv[sz(j) * 2], m.lo[0], m.hi[0]));
}

TEST(Footage, EstimatedStartsPlayInTheRuntime) {
  const rollout::Model m = small_model();
  const int S = 32;
  InverseSpec spec;
  spec.hidden = 8;
  spec.levels = 3;
  Inverse inv = init_inverse(spec, 6);
  inv.size = S;
  inv.scale = {0.8f, 0.6f};
  std::vector<Frame> frames;
  for (int i = 0; i < 4; ++i) {
    Frame fr(sz(S) * S * 4);
    render_fields(m, blobs(S, fl(i), 0.f), fr);
    frames.push_back(fr);
  }
  EstimateOptions o;
  o.context = 3;
  o.refine_opt.iterations = 5;
  o.assim.controls = {0.2f, 0.7f, 0.4f};
  o.assim.time = 1.5f;
  const Estimate e = estimate_start(m, inv, frames, S, o);
  ASSERT_EQ(e.fields.size(), 3u);
  EXPECT_EQ(e.start.controls, o.assim.controls);
  EXPECT_FLOAT_EQ(e.start.time, 1.5f);
  EXPECT_EQ(e.start.fine_t.size(), sz(S) * S);
  const std::vector<float> c = coarse_fields(e.fields.back(), m.h.res);
  for (int j = 0; j < m.h.res * m.h.res; ++j) EXPECT_FLOAT_EQ(e.start.coarse[sz(j) * 4 + 3], c[sz(j) * 2 + 1]);
  // A copy of the effect with the estimate: fine fields box-averaged to 16, stored ones kept or not.
  const rollout::Model only = with_starts(m, std::span(&e.start, 1), 16, false);
  ASSERT_EQ(only.starts.size(), 1u);
  EXPECT_EQ(only.h.start_fine, 16);
  EXPECT_EQ(only.starts[0].fine_t.size(), 256u);
  double sum_a = 0, sum_b = 0;
  for (const float v : e.start.fine_t) sum_a += v / static_cast<double>(e.start.fine_t.size());
  for (const float v : only.starts[0].fine_t) sum_b += v / 256.0;
  EXPECT_NEAR(sum_a, sum_b, 1e-5);
  const rollout::Model both = with_starts(m, std::span(&e.start, 1), 16, true);
  EXPECT_EQ(both.starts.size(), 2u);
  EXPECT_TRUE(both.starts[0].fine_t.empty());
  std::ostringstream os;
  ASSERT_TRUE(rollout::save_model(os, both));
  const std::string bytes = os.str();
  nvfx_effect* fx = nullptr;
  ASSERT_EQ(nvfx_effect_load_memory(bytes.data(), bytes.size(), &fx), NVFX_OK);
  nvfx_instance* in = nullptr;
  ASSERT_EQ(nvfx_instance_create(fx, 32, &in), NVFX_OK);
  ASSERT_EQ(nvfx_instance_set_variation(in, 1), NVFX_OK);
  nvfx_instance_set_drift(in, 0.f);
  std::vector<std::uint8_t> rgba(32 * 32 * 4);
  for (int f = 0; f < 5; ++f) ASSERT_EQ(nvfx_render(in, f / 30.0, rgba.data(), 32 * 4), NVFX_OK);
  nvfx_instance_free(in);
  nvfx_effect_free(fx);
  rollout::StartPoint wrong = e.start;
  wrong.coarse.resize(10);
  EXPECT_THROW(with_starts(m, std::span(&wrong, 1), 16, false), std::invalid_argument);
}

TEST(Footage, MotionNetworkRoundTripsAndReadsItsInputs) {
  Inverse inv = init_inverse(InverseSpec{8, 2, false}, 1);
  inv.size = 32;
  std::stringstream plain;
  ASSERT_TRUE(save_inverse(plain, inv));
  ASSERT_TRUE(load_inverse(plain)->motion.empty());  // version 1: no motion network
  inv.motion = init_motion(6, 2);
  inv.motion.in_scale = {0.5f, 0.25f};
  inv.motion.out_scale = 0.2f;
  std::stringstream ss;
  ASSERT_TRUE(save_inverse(ss, inv));
  auto back = load_inverse(ss);
  ASSERT_TRUE(back) << back.error();
  EXPECT_EQ(back->motion.w, inv.motion.w);
  EXPECT_EQ(back->motion.hidden, 6);
  EXPECT_FLOAT_EQ(back->motion.out_scale, 0.2f);
  // Inputs: coarse fields of three frames (5 x 5), the flow into the last frame (3 x 3, with its flag), the flow into
  // the middle one, the position.
  const int S = 32, R = 8;
  const std::vector<Fields> three = {blobs(S), blobs(S, 1.f), blobs(S, 2.f)};
  const Flow into_last = block_flow(three[1], three[2], R, {1.f, 1.f}, {}), into_middle = block_flow(three[0], three[1], R, {1.f, 1.f}, {});
  const std::vector<float> in = motion_inputs(inv.motion, three, into_middle, into_last, R);
  ASSERT_EQ(in.size(), sz(R) * R * Motion::kInputs);
  const std::vector<float> c = coarse_fields(three[2], R);
  const int x = 3, y = 4;
  const float* row = in.data() + sz(y * R + x) * Motion::kInputs;
  EXPECT_FLOAT_EQ(row[2 * 50 + 2 * 12], c[sz(y * R + x) * 2] / 0.5f);  // last frame, centre of the 5 x 5, heat
  EXPECT_FLOAT_EQ(row[2 * 50 + 2 * 12 + 1], c[sz(y * R + x) * 2 + 1] / 0.25f);
  if (into_last.known[sz(y * R + x)]) {
    EXPECT_FLOAT_EQ(row[150 + 4 * 3], 4.f * into_last.uv[sz(y * R + x) * 2]);
  }
  EXPECT_FLOAT_EQ(row[150 + 4 * 3 + 2], into_last.known[sz(y * R + x)] ? 1.f : 0.f);
  EXPECT_NEAR(row[Motion::kInputs - 1], (4.5f / 8.f) * 2.f - 1.f, 1e-6f);
  // Training on a known map (the velocity is the flow into the last frame) gets close to it.
  std::vector<MotionSample> samples(1);
  samples[0].inputs = in;
  for (int q = 0; q < R * R; ++q) {
    samples[0].target.push_back(0.3f * in[sz(q) * Motion::kInputs + 162] + 0.05f);
    samples[0].target.push_back(-0.1f);
  }
  Motion mo;
  mo.hidden = 16;
  mo.in_scale = {0.5f, 0.25f};
  InverseTrainOptions o;
  o.iterations = 800;
  o.batch = 64;
  o.lr = 5e-3f;
  const double loss = train_motion(mo, samples, o);
  EXPECT_LT(loss, 0.05);
  EXPECT_FLOAT_EQ(mo.in_scale[1], 0.25f);  // training keeps the input units the inputs were made with
  EXPECT_GT(mo.out_scale, 0.04f);
  const Flow v = apply_motion(mo, in, R);
  EXPECT_EQ(v.known, std::vector<std::uint8_t>(sz(R) * R, 1));
}
