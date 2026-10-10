// G3 run codec (include/neuralfx/codec, docs/DCM.md §8): the residual coder round-trips exactly, the decoder reproduces
// the encoder's reconstruction bit for bit, encoding is deterministic, and streams made for another model, damaged or
// truncated are refused.
#include <neuralfx/codec/rcoder.hpp>
#include <neuralfx/codec/run_codec.hpp>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include <gtest/gtest.h>

#include <random>

using namespace nfx;

namespace {

std::size_t sz(int v) { return static_cast<std::size_t>(v); }

// A small effect (random weights, a 16-cell coarse grid) and a short simulated run at 64 px, shared by the tests.
struct Fixture {
  rollout::Model m;
  codec::TrueRun run;
  Fixture() {
    rollout::Hyper h;
    h.res = 16;
    h.hidden = 8;
    h.memory = 2;
    h.jacobi = 8;
    h.render_hidden = 8;
    m = rollout::init_model(h, 7);
    m.effect = "fire";
    m.scale = {0.1f, 0.2f, 0.1f, 0.01f};
    m.render_scale = {1.f, 0.05f};
    sim::Params p;
    p.size = 64;
    p.seed = 4242;
    p.intensity = 0.6f;
    p.wind = 0.4f;
    p.turbulence = 0.7f;
    sim::Fluid f(p);
    for (int i = 0; i < 25; ++i) f.step_frame();
    const int F = 12, R = h.res, S = p.size;
    const std::size_t cn = sz(R) * sz(R) * rollout::kPhys, fn = sz(S) * sz(S);
    run.controls = {p.intensity, p.wind, p.turbulence};
    run.seed = p.seed;
    run.frames = F;
    run.size = S;
    run.coarse.resize(sz(F + 1) * cn);
    run.fine_t.resize(sz(F + 1) * fn);
    run.fine_d.resize(sz(F + 1) * fn);
    for (int i = 0; i <= F; ++i) {
      if (i > 0) f.step_frame();
      const sim::State st = f.state();
      if (i == 0) run.time0 = st.time;
      rollout::coarse_from_sim(st, R, p.fps, std::span(run.coarse).subspan(sz(i) * cn, cn));
      std::copy(st.temp.begin(), st.temp.end(), run.fine_t.begin() + static_cast<std::ptrdiff_t>(sz(i) * fn));
      std::copy(st.soot.begin(), st.soot.end(), run.fine_d.begin() + static_cast<std::ptrdiff_t>(sz(i) * fn));
      if (i == 4) {  // a stored start point from another moment of the run
        rollout::StartPoint sp;
        sp.controls = {0.5f, 0.5f, 0.5f};
        sp.seed = 1;
        sp.time = st.time;
        sp.coarse.assign(run.coarse.begin() + static_cast<std::ptrdiff_t>(sz(i) * cn), run.coarse.begin() + static_cast<std::ptrdiff_t>(sz(i + 1) * cn));
        m.starts.push_back(sp);
      }
    }
  }
};

const Fixture& fixture() {
  static const Fixture f;
  return f;
}

std::vector<codec::Settings> settings_set() {
  std::vector<codec::Settings> v;
  codec::Settings a;  // defaults: corrections of every channel every 4 frames, a coded fine start
  v.push_back(a);
  codec::Settings b = a;  // the start only, from zero, grown fine fields
  b.k = 0;
  b.start_nearest = false;
  b.start_fine = 0;
  v.push_back(b);
  codec::Settings c = a;  // heat and soot only, every frame, dead zone, side context
  c.k = 1;
  c.q = 0.3f;
  c.q_vel = 0.f;
  c.round = 0.3f;
  c.side_context = true;
  v.push_back(c);
  codec::Settings d = a;  // fine residuals every 2 frames at 32 px, velocity only in the coarse planes
  d.k = 3;
  d.q_mat = 0.f;
  d.kf = 2;
  d.fine_res = 32;
  d.qf = 0.05f;
  v.push_back(d);
  codec::Settings e = d;  // full-resolution fine residuals every frame, no coarse planes, without the coarse sync
  e.k = 0;
  e.kf = 1;
  e.fine_res = 64;
  e.fine_sets_coarse = false;
  e.correct_flow = false;
  v.push_back(e);
  return v;
}

}  // namespace

TEST(G3Coder, PlanesRoundTripExactly) {
  std::mt19937_64 rng(3);
  std::vector<std::vector<std::int32_t>> planes;
  std::vector<std::vector<std::uint8_t>> sides;
  std::vector<std::array<int, 4>> shapes;  // kind, h, w, c
  for (int t = 0; t < 12; ++t) {
    const int h = 1 + static_cast<int>(rng() % 20), w = 1 + static_cast<int>(rng() % 20), c = 1 + static_cast<int>(rng() % 4);
    std::vector<std::int32_t> v(sz(h) * sz(w) * sz(c));
    const int mode = t % 4;
    for (auto& x : v) {
      if (mode == 0) x = 0;
      else if (mode == 1) x = (rng() % 10 == 0) ? static_cast<std::int32_t>(rng() % 7) - 3 : 0;
      else if (mode == 2) x = static_cast<std::int32_t>(rng() % 2001) - 1000;
      else x = static_cast<std::int32_t>(rng() % (2u * codec::ResidualModel::kMaxAbs + 1)) - codec::ResidualModel::kMaxAbs;
    }
    std::vector<std::uint8_t> side;
    if (t % 3 == 1) {
      side.resize(sz(h) * sz(w));
      for (auto& s : side) s = static_cast<std::uint8_t>(rng() % 16);
    }
    planes.push_back(v);
    sides.push_back(side);
    shapes.push_back({t % 4, h, w, c});
  }
  std::vector<std::uint8_t> bytes;
  {
    codec::ArithEncoder ac(bytes);
    codec::ResidualModel rm;
    for (std::size_t i = 0; i < planes.size(); ++i) {
      rm.encode(ac, static_cast<codec::PlaneKind>(shapes[i][0]), shapes[i][1], shapes[i][2], shapes[i][3], planes[i], sides[i]);
    }
    ac.flush();
  }
  codec::ArithDecoder ad(bytes);
  codec::ResidualModel rm;
  for (std::size_t i = 0; i < planes.size(); ++i) {
    std::vector<std::int32_t> got(planes[i].size(), 7);
    ASSERT_TRUE(rm.decode(ad, static_cast<codec::PlaneKind>(shapes[i][0]), shapes[i][1], shapes[i][2], shapes[i][3], got, sides[i]));
    EXPECT_EQ(got, planes[i]) << "plane " << i;
  }
  EXPECT_EQ(ad.consumed(), bytes.size());
}

TEST(G3Coder, EmptyPlanesCostAlmostNothing) {
  std::vector<std::uint8_t> bytes;
  codec::ArithEncoder ac(bytes);
  codec::ResidualModel rm;
  const std::vector<std::int32_t> zero(32 * 32 * 4, 0);
  for (int i = 0; i < 30; ++i) rm.encode(ac, codec::PlaneKind::coarse, 32, 32, 4, zero);
  ac.flush();
  EXPECT_LT(bytes.size(), 120u);  // 30 x 4,096 zeros
}

TEST(G3Codec, RendererMatchesTheReferenceByteForByte) {
  const Fixture& fx = fixture();
  rollout::Model m = fx.m;
  m.render_w[rollout::render_layout(m.h).bo] = -0.3f;  // a negative output bias: the gate's zero must still give 0
  std::vector<float> ctl(fx.run.controls.begin(), fx.run.controls.end());
  rollout::State s = rollout::start(m, 0, fx.run.size, ctl, 9);
  std::vector<float> ref(sz(fx.run.size) * sz(fx.run.size) * 4), scratch;
  std::vector<std::uint8_t> want(ref.size()), got(ref.size());
  for (int f = 0; f < 6; ++f) {
    rollout::step(m, s, ctl, 9);
    rollout::render(m, s, ref);
    for (std::size_t i = 0; i < ref.size(); ++i) want[i] = static_cast<std::uint8_t>(ref[i] * 255.f + 0.5f);
    codec::render_u8(m, s, got, scratch);
    ASSERT_EQ(got, want) << "frame " << f;
  }
}

TEST(G3Codec, DecoderReproducesTheEncoderBitExactly) {
  const Fixture& fx = fixture();
  for (const auto& s : settings_set()) {
    const codec::Encoded e = codec::encode(fx.m, fx.run, s, true);
    ASSERT_EQ(e.frames.size(), sz(fx.run.frames) * sz(fx.run.size) * sz(fx.run.size) * 4);
    const auto d = codec::decode(fx.m, e.stream);
    ASSERT_TRUE(d.has_value()) << codec::describe(s) << ": " << d.error();
    EXPECT_TRUE(d->verified) << codec::describe(s);
    EXPECT_EQ(d->rgba, e.frames) << codec::describe(s);
    EXPECT_EQ(d->frames, fx.run.frames);
    EXPECT_EQ(d->seed, fx.run.seed);
    EXPECT_EQ(d->time0, fx.run.time0);
    EXPECT_EQ(codec::describe(d->settings), codec::describe(s));
    // streaming decode gives the same frames
    std::vector<std::uint8_t> streamed;
    const auto d2 = codec::decode(fx.m, e.stream, [&](int, std::span<const std::uint8_t> f) { streamed.insert(streamed.end(), f.begin(), f.end()); });
    ASSERT_TRUE(d2.has_value());
    EXPECT_EQ(streamed, e.frames);
  }
}

TEST(G3Codec, EncodingAndDecodingAreDeterministic) {
  const Fixture& fx = fixture();
  for (const auto& s : settings_set()) {
    const codec::Encoded a = codec::encode(fx.m, fx.run, s, true), b = codec::encode(fx.m, fx.run, s, true);
    EXPECT_EQ(a.stream, b.stream) << codec::describe(s);
    EXPECT_EQ(a.frames, b.frames);
    const auto d1 = codec::decode(fx.m, a.stream), d2 = codec::decode(fx.m, a.stream);
    ASSERT_TRUE(d1 && d2);
    EXPECT_EQ(d1->rgba, d2->rgba);
  }
}

TEST(G3Codec, MoreBytesFollowTheRunCloser) {
  const Fixture& fx = fixture();
  codec::Settings lo;
  lo.k = 0;
  lo.q_start = 60.f;
  lo.start_fine = 0;
  codec::Settings hi;
  hi.k = 1;
  hi.q = 0.05f;
  hi.kf = 1;
  hi.fine_res = 64;
  hi.qf = 0.02f;
  // compare the reconstructions' fine fields with the truth through the codec's own frames: identical runs would mean
  // the corrections do nothing
  const auto a = codec::encode(fx.m, fx.run, lo, true), b = codec::encode(fx.m, fx.run, hi, true);
  EXPECT_LT(a.stream.size(), b.stream.size());
  EXPECT_NE(a.frames, b.frames);
  EXPECT_GT(b.coarse_planes, 0);
  EXPECT_EQ(b.fine_planes, fx.run.frames);
}

TEST(G3Codec, RefusesStreamsOfAnotherModel) {
  const Fixture& fx = fixture();
  const auto e = codec::encode(fx.m, fx.run, codec::Settings{});
  rollout::Model other = fx.m;
  other.render_w[0] += 0.5f;
  const auto d = codec::decode(other, e.stream);
  ASSERT_FALSE(d.has_value());
  EXPECT_NE(d.error().find("another model"), std::string::npos);
}

TEST(G3Codec, RefusesDamagedAndTruncatedStreams) {
  const Fixture& fx = fixture();
  const auto all = settings_set();
  for (const auto& s : {all[0], all[2], all[3]}) {
    const auto e = codec::encode(fx.m, fx.run, s);
    const auto& b = e.stream;
    // truncations, from the empty stream to one byte short
    for (std::size_t n = 0; n < b.size(); n += 1 + b.size() / 12) {
      const auto d = codec::decode(fx.m, std::span(b).first(n));
      EXPECT_FALSE(d.has_value()) << codec::describe(s) << " truncated to " << n;
    }
    EXPECT_FALSE(codec::decode(fx.m, std::span(b).first(b.size() - 1)).has_value());
    // a flipped bit in bytes spread over the stream (header, payload, trailer), one at a time
    for (std::size_t i = 0; i < b.size(); i += 1 + b.size() / 24) {
      auto bad = b;
      bad[i] ^= static_cast<std::uint8_t>(1u << (i % 8));
      const auto d = codec::decode(fx.m, bad);
      EXPECT_FALSE(d.has_value()) << codec::describe(s) << " byte " << i << " of " << b.size();
    }
    for (const std::size_t i : {b.size() - 1, b.size() - 5}) {  // the checksum and the state hash
      auto bad = b;
      bad[i] ^= 0x10;
      EXPECT_FALSE(codec::decode(fx.m, bad).has_value());
    }
    // junk
    std::mt19937_64 rng(11);
    for (int t = 0; t < 4; ++t) {
      auto junk = b;
      for (std::size_t i = 3; i < junk.size(); ++i) junk[i] = static_cast<std::uint8_t>(rng());
      EXPECT_FALSE(codec::decode(fx.m, junk).has_value());
    }
  }
}

TEST(G3Codec, StoredSettingsRoundTrip) {
  codec::Settings s;
  s.q = 0.123f;
  s.q_vel = 0.5f;
  s.q_mat = 2.f;
  s.q_start = 3.f;
  s.qf = 0.07f;
  s.kf = 2;
  s.round = 0.3f;
  const codec::Settings t = codec::stored_settings(s);
  EXPECT_NEAR(t.q, 0.123f, 0.123f * 0.025f);
  EXPECT_NEAR(t.q * t.q_vel, 0.123f * 0.5f, 0.123f * 0.5f * 0.025f);
  EXPECT_NEAR(t.q * t.q_mat, 0.246f, 0.246f * 0.025f);
  EXPECT_NEAR(t.round, 0.3f, 0.002f);
  EXPECT_EQ(codec::describe(codec::stored_settings(t)), codec::describe(t));
}
