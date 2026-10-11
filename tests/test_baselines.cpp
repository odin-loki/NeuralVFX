// Phase 2: metrics, block compression and flipbook baselines.
#include <neuralfx/flipbook.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/sim.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <algorithm>
#include <ranges>
#include <thread>

using namespace nfx;

namespace {

Clip smooth_clip(int size, int frames, bool loop) {
  Clip c;
  c.allocate(size, frames);
  c.loop = loop;
  for (int f = 0; f < frames; ++f) {
    auto fr = c.frame(f);
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        const double t = 2 * 3.14159265 * f / frames;
        const std::size_t i = (static_cast<std::size_t>(y) * size + x) * 4;
        const double a = 0.5 + 0.5 * std::sin(0.2 * x + t) * std::cos(0.15 * y);
        fr[i + 3] = static_cast<std::uint8_t>(255 * a);
        fr[i] = static_cast<std::uint8_t>(255 * a * 0.9);
        fr[i + 1] = static_cast<std::uint8_t>(255 * a * 0.5);
        fr[i + 2] = static_cast<std::uint8_t>(255 * a * 0.2);
      }
    }
  }
  return c;
}

}  // namespace

TEST(Metrics, PsnrOfIdenticalIsCappedAndKnownErrorIsExact) {
  std::vector<std::uint8_t> a(64, 100), b(64, 100);
  EXPECT_EQ(metrics::psnr_from_mse(metrics::mse(a, b)), metrics::kPsnrCap);
  std::ranges::fill(b, 100 + 51);  // error 0.2 everywhere -> MSE 0.04 -> 13.98 dB
  EXPECT_NEAR(metrics::psnr_from_mse(metrics::mse(a, b)), 13.9794, 1e-3);
}

TEST(Metrics, SsimIsOneForIdenticalAndDropsWithNoise) {
  const Clip c = smooth_clip(32, 1, false);
  EXPECT_NEAR(metrics::ssim(c.frame(0), c.frame(0), 32), 1.0, 1e-9);
  Clip n = c;
  std::mt19937 rng(1);
  for (auto& v : n.rgba) v = static_cast<std::uint8_t>(std::clamp(int(v) + int(rng() % 41) - 20, 0, 255));
  const double s = metrics::ssim(c.frame(0), n.frame(0), 32);
  EXPECT_LT(s, 0.95);
  EXPECT_GT(s, 0.2);
}

TEST(Metrics, ScoreRestrictsFramesAndMeasuresFlicker) {
  const Clip ref = smooth_clip(16, 8, true);
  EXPECT_EQ(metrics::score(ref, ref).psnr, metrics::kPsnrCap);
  EXPECT_NEAR(metrics::score(ref, ref).flicker, 1.0, 1e-12);
  Clip jitter = ref;
  for (int f = 1; f < 8; f += 2) {
    for (auto& v : jitter.frame(f)) v = static_cast<std::uint8_t>(std::min(255, v + 12));
  }
  const auto s = metrics::score(ref, jitter);
  EXPECT_GT(s.flicker, 1.5);  // alternating frames jitter: far more second-difference energy
  const int odd[] = {1, 3, 5, 7}, even[] = {0, 2, 4, 6};
  EXPECT_LT(metrics::score(ref, jitter, odd).psnr, 40.0);
  EXPECT_EQ(metrics::score(ref, jitter, even).psnr, metrics::kPsnrCap);
}

TEST(Metrics, StatsDistanceIsZeroForTheSameClip) {
  const Clip c = smooth_clip(32, 4, true);
  const auto d = metrics::distance(metrics::stats(c), metrics::stats(c));
  EXPECT_EQ(d.coverage_l1, 0.0);
  EXPECT_EQ(d.spectrum_l1, 0.0);
  EXPECT_EQ(d.mean_frame_psnr, metrics::kPsnrCap);
  EXPECT_NEAR(d.motion_ratio, 1.0, 1e-12);
  EXPECT_EQ(metrics::stats(c).spectrum.size(), 16u);
}

TEST(Metrics, BootstrapIntervalsBehave) {
  std::vector<double> a(50), b(50);
  std::mt19937 rng(3);
  std::normal_distribution<double> n(0, 1);
  for (std::size_t i = 0; i < a.size(); ++i) {
    b[i] = n(rng);
    a[i] = b[i] + 2.0 + 0.1 * n(rng);
  }
  const auto r = metrics::paired_bootstrap(a, b);
  EXPECT_NEAR(r.mean, 2.0, 0.1);
  EXPECT_LT(r.lo, r.mean);
  EXPECT_GT(r.hi, r.mean);
  EXPECT_FALSE(r.covers_zero());
  const auto t = metrics::paired_bootstrap(b, b);
  EXPECT_TRUE(t.covers_zero());
}

TEST(Bc, FlatBlocksAreExactAndGradientsAreClose) {
  std::array<std::uint8_t, 64> px{};
  for (int i = 0; i < 16; ++i) {
    px[static_cast<std::size_t>(i * 4)] = 200;
    px[static_cast<std::size_t>(i * 4 + 1)] = 100;
    px[static_cast<std::size_t>(i * 4 + 2)] = 48;
  }
  std::array<std::uint8_t, 64> out{};
  flipbook::decode_bc1(flipbook::encode_bc1(px), out);
  for (int i = 0; i < 16; ++i) {
    EXPECT_NEAR(out[static_cast<std::size_t>(i * 4)], 200, 4);
    EXPECT_NEAR(out[static_cast<std::size_t>(i * 4 + 1)], 100, 2);
    EXPECT_NEAR(out[static_cast<std::size_t>(i * 4 + 2)], 48, 4);
  }
  std::array<std::uint8_t, 16> a{}, ad{};
  for (int i = 0; i < 16; ++i) a[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(10 + 15 * i);
  flipbook::decode_bc4(flipbook::encode_bc4(a), ad);
  for (int i = 0; i < 16; ++i) EXPECT_NEAR(ad[static_cast<std::size_t>(i)], a[static_cast<std::size_t>(i)], 17);
}

TEST(Bc, Bc3RoundTripOfASimulatedFrameIsGood) {
  sim::Params p;
  p.effect = sim::Effect::smoke;
  p.size = 64;
  p.frames = 1;
  p.warmup = 40;
  const Clip c = sim::simulate(p);
  const auto blocks = flipbook::compress_bc3(c.frame(0), 64, 64);
  EXPECT_EQ(blocks.size(), 64u * 64u);  // 8 bits per pixel
  const auto back = flipbook::decompress_bc3(blocks, 64, 64);
  EXPECT_GT(metrics::psnr_from_mse(metrics::mse(c.frame(0), back)), 34.0);
}

TEST(Flipbook, AllFramesRawIsExactAndMemoryIsAccounted) {
  const Clip c = smooth_clip(32, 8, true);
  const auto fb = flipbook::build(c, {8, 32, flipbook::Codec::raw, 0});
  EXPECT_EQ(flipbook::play(fb).rgba, c.rgba);
  EXPECT_EQ(fb.bytes, 8u * 32 * 32 * 4);
  EXPECT_EQ(flipbook::memory_bytes({16, 128, flipbook::Codec::bc3, 32}, 16), 16u * (128 * 128 + 32 * 32 * 2));
}

TEST(Flipbook, FewerFramesBlendAndLowerResolutionCostQuality) {
  const Clip c = smooth_clip(32, 16, true);
  const double full = metrics::score(c, flipbook::play(flipbook::build(c, {16, 32, flipbook::Codec::raw, 0}))).psnr;
  const double half_t = metrics::score(c, flipbook::play(flipbook::build(c, {8, 32, flipbook::Codec::raw, 0}))).psnr;
  const double half_s = metrics::score(c, flipbook::play(flipbook::build(c, {16, 16, flipbook::Codec::raw, 0}))).psnr;
  EXPECT_EQ(full, metrics::kPsnrCap);
  EXPECT_LT(half_t, 60.0);
  EXPECT_GT(half_t, 20.0);
  EXPECT_LT(half_s, 60.0);
  EXPECT_GT(half_s, 20.0);
}

TEST(Flipbook, ExplicitKeepListServesTheInterpolationTest) {
  const Clip c = smooth_clip(16, 8, true);
  const int even[] = {0, 2, 4, 6};
  const auto fb = flipbook::build(c, {0, 16, flipbook::Codec::raw, 0}, even);
  EXPECT_EQ(fb.kept, (std::vector<int>{0, 2, 4, 6}));
  const Clip out = flipbook::play(fb);
  for (const int f : even) EXPECT_TRUE(std::ranges::equal(out.frame(f), c.frame(f)));
}

TEST(Flipbook, MotionVectorsFollowATranslatingBlob) {
  // A soft blob moving 6 pixels per frame: blending ghosts it, motion vectors move it.
  Clip c;
  c.allocate(64, 3);
  for (int f = 0; f < 3; ++f) {
    auto fr = c.frame(f);
    for (int y = 0; y < 64; ++y) {
      for (int x = 0; x < 64; ++x) {
        const double dx = x - (20 + 6 * f), dy = y - 32;
        const auto v = static_cast<std::uint8_t>(255 * std::exp(-(dx * dx + dy * dy) / 40.0));
        const std::size_t i = (static_cast<std::size_t>(y) * 64 + x) * 4;
        fr[i] = fr[i + 1] = fr[i + 2] = fr[i + 3] = v;
      }
    }
  }
  const int ends[] = {0, 2};
  const double blend = metrics::score(c, flipbook::play(flipbook::build(c, {0, 64, flipbook::Codec::raw, 0}, ends)),
                                      std::array{1}).psnr;
  const double motion = metrics::score(c, flipbook::play(flipbook::build(c, {0, 64, flipbook::Codec::raw, 16}, ends)),
                                       std::array{1}).psnr;
  EXPECT_GT(motion, blend + 6.0);
}

TEST(Flipbook, LadderCoversRawBc3AndMotionVectors) {
  const auto l = flipbook::ladder(128, 64);
  EXPECT_TRUE(std::ranges::any_of(l, [](const auto& s) { return s.codec == flipbook::Codec::raw; }));
  EXPECT_TRUE(std::ranges::any_of(l, [](const auto& s) { return s.flow_res > 0; }));
  EXPECT_TRUE(std::ranges::any_of(l, [](const auto& s) { return s.frames == 64 && s.res == 128; }));
}

// --- production block formats (BC7, ASTC; encoders fetched at build time) ---------------------------------------------

namespace {

std::vector<std::uint8_t> smoke_frame(int size) {
  sim::Params p;
  p.effect = sim::Effect::smoke;
  p.size = size;
  p.frames = 1;
  p.warmup = 40;
  const Clip c = sim::simulate(p);
  return {c.frame(0).begin(), c.frame(0).end()};
}

double round_trip_psnr(flipbook::Codec codec, const std::vector<std::uint8_t>& img, int w, int h) {
  const auto back = flipbook::decode(codec, flipbook::encode(codec, img, w, h), w, h);
  return metrics::psnr_from_mse(metrics::mse(img, back));
}

}  // namespace

TEST(Formats, NamesSizesAndFamilies) {
  using flipbook::Codec;
  EXPECT_EQ(flipbook::frame_bytes(Codec::raw, 32), 32u * 32 * 4);
  EXPECT_EQ(flipbook::frame_bytes(Codec::bc3, 128), 128u * 128);
  EXPECT_EQ(flipbook::frame_bytes(Codec::bc7, 128), 128u * 128);      // 8 bits per pixel
  EXPECT_EQ(flipbook::frame_bytes(Codec::astc4x4, 64), 64u * 64);
  EXPECT_EQ(flipbook::frame_bytes(Codec::astc6x6, 128), 22u * 22 * 16);  // partial blocks are whole blocks in memory
  EXPECT_EQ(flipbook::frame_bytes(Codec::astc8x8, 128), 16u * 16 * 16);  // 2 bits per pixel
  EXPECT_EQ(flipbook::frame_bytes(Codec::astc12x12, 32), 3u * 3 * 16);
  EXPECT_EQ(flipbook::memory_bytes({16, 64, Codec::astc8x8, 16}, 16), 16u * (8 * 8 * 16 + 16 * 16 * 2));
  EXPECT_EQ((flipbook::Spec{64, 128, Codec::bc7, 0}.describe()), "bc7 64f 128px");
  EXPECT_EQ((flipbook::Spec{16, 64, Codec::astc10x10, 16}.describe()), "astc10x10 16f 64px +mv16");
  EXPECT_EQ((flipbook::Spec{16, 64, Codec::bc3, 16}.family()), "flipbook_mv");  // the original names stay
  EXPECT_EQ((flipbook::Spec{16, 64, Codec::raw, 0}.family()), "flipbook_raw");
  EXPECT_EQ((flipbook::Spec{16, 64, Codec::bc7, 16}.family()), "flipbook_bc7_mv");
  EXPECT_EQ((flipbook::Spec{16, 64, Codec::astc5x5, 0}.family()), "flipbook_astc");
  EXPECT_FALSE(flipbook::production(Codec::bc3));
  EXPECT_TRUE(flipbook::available(Codec::bc3));
  EXPECT_EQ(flipbook::ladder(128, 64).size(), 31u);  // the studies' original ladder is unchanged
}

TEST(Formats, LadderHasEveryAvailableFormatAndUnavailableOnesRefuse) {
  const auto l = flipbook::ladder_production(128, 64);
  std::size_t have = 0;
  for (const auto codec : flipbook::production_codecs()) {
    const auto n = std::ranges::count_if(l, [&](const auto& s) { return s.codec == codec; });
    if (flipbook::available(codec)) {
      ++have;
      EXPECT_EQ(n, 21);  // the BC3 rows' frame counts and resolutions, with motion vectors where they have them
    } else {
      EXPECT_EQ(n, 0);
      const std::vector<std::uint8_t> img(16 * 16 * 4, 0);
      EXPECT_THROW((void)flipbook::encode(codec, img, 16, 16), std::runtime_error);
    }
  }
  EXPECT_EQ(l.size(), 21 * have);
  std::vector<std::string> names;
  for (const auto& s : l) names.push_back(s.describe());
  std::ranges::sort(names);
  EXPECT_EQ(std::ranges::adjacent_find(names), names.end());  // every configuration has its own name
}

TEST(Formats, Bc7BeatsOurBc3AtTheSameSize) {
  if (!flipbook::available(flipbook::Codec::bc7)) GTEST_SKIP() << "no BC7 encoder in this build (NEURALFX_FETCH_ENCODERS)";
  const auto img = smoke_frame(64);
  const auto blocks = flipbook::encode(flipbook::Codec::bc7, img, 64, 64);
  EXPECT_EQ(blocks.size(), 64u * 64u);
  EXPECT_EQ(blocks, flipbook::encode(flipbook::Codec::bc7, img, 64, 64));  // deterministic
  const double bc7 = round_trip_psnr(flipbook::Codec::bc7, img, 64, 64), bc3 = round_trip_psnr(flipbook::Codec::bc3, img, 64, 64);
  EXPECT_GT(bc7, 40.0);
  EXPECT_GT(bc7, bc3 + 3.0);
}

TEST(Formats, AstcBlockSizesTradeBitsForQuality) {
  using flipbook::Codec;
  if (!flipbook::available(Codec::astc4x4)) GTEST_SKIP() << "no ASTC encoder in this build (NEURALFX_FETCH_ENCODERS)";
  const auto img = smoke_frame(64);
  double prev = 1e9;
  for (const Codec c : {Codec::astc4x4, Codec::astc5x5, Codec::astc6x6, Codec::astc8x8, Codec::astc10x10, Codec::astc12x12}) {
    const auto blocks = flipbook::encode(c, img, 64, 64);
    EXPECT_EQ(blocks.size(), flipbook::frame_bytes(c, 64));
    EXPECT_EQ(blocks, flipbook::encode(c, img, 64, 64));
    const double q = round_trip_psnr(c, img, 64, 64);
    EXPECT_LT(q, prev) << flipbook::codec_name(c);  // fewer bits, lower quality
    EXPECT_GT(q, 24.0) << flipbook::codec_name(c);
    prev = q;
  }
  EXPECT_GT(round_trip_psnr(Codec::astc4x4, img, 64, 64), round_trip_psnr(Codec::bc3, img, 64, 64));
  // An image that is not a whole number of blocks is padded to one.
  std::vector<std::uint8_t> odd(60 * 60 * 4);
  for (int y = 0; y < 60; ++y) std::copy_n(img.begin() + y * 64 * 4, 60 * 4, odd.begin() + y * 60 * 4);
  EXPECT_EQ(flipbook::encode(Codec::astc8x8, odd, 60, 60).size(), 8u * 8 * 16);
  EXPECT_GT(round_trip_psnr(Codec::astc8x8, odd, 60, 60), 24.0);
}

TEST(Formats, EncodersGiveTheSameBlocksOnSeveralThreads) {
  std::vector<flipbook::Codec> codecs;
  for (const auto c : flipbook::production_codecs()) {
    if (flipbook::available(c)) codecs.push_back(c);
  }
  if (codecs.empty()) GTEST_SKIP() << "no production encoder in this build";
  const auto img = smoke_frame(32);
  std::vector<std::vector<std::uint8_t>> serial, a(codecs.size()), b(codecs.size());
  for (const auto c : codecs) serial.push_back(flipbook::encode(c, img, 32, 32));
  {
    std::jthread t1([&] { for (std::size_t i = 0; i < codecs.size(); ++i) a[i] = flipbook::encode(codecs[i], img, 32, 32); });
    std::jthread t2([&] { for (std::size_t i = codecs.size(); i-- > 0;) b[i] = flipbook::encode(codecs[i], img, 32, 32); });
  }
  EXPECT_EQ(a, serial);
  EXPECT_EQ(b, serial);
}

TEST(Formats, ProductionFlipbooksPlayLikeTheOthers) {
  using flipbook::Codec;
  if (!flipbook::available(Codec::bc7) && !flipbook::available(Codec::astc8x8)) GTEST_SKIP() << "no production encoder in this build";
  sim::Params p;
  p.effect = sim::Effect::fire;
  p.size = 32;
  p.frames = 8;
  p.warmup = 20;
  const Clip c = sim::simulate(p);
  for (const Codec codec : {Codec::bc7, Codec::astc8x8}) {
    if (!flipbook::available(codec)) continue;
    const auto fb = flipbook::build(c, {8, 32, codec, 0});
    EXPECT_EQ(fb.bytes, 8 * flipbook::frame_bytes(codec, 32));
    const Clip out = flipbook::play(fb);
    for (int f = 0; f < 8; ++f) {  // every frame kept: playback is each frame's decode
      const auto img = std::vector<std::uint8_t>(c.frame(f).begin(), c.frame(f).end());
      const auto back = flipbook::decode(codec, flipbook::encode(codec, img, 32, 32), 32, 32);
      EXPECT_TRUE(std::ranges::equal(out.frame(f), back)) << flipbook::codec_name(codec) << " frame " << f;
    }
    const auto mv = flipbook::build(c, {4, 32, codec, 8});
    EXPECT_EQ(mv.bytes, 4 * (flipbook::frame_bytes(codec, 32) + 8 * 8 * 2));
    EXPECT_GT(metrics::score(c, flipbook::play(mv)).psnr, 20.0);
  }
}
