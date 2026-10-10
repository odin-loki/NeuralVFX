// Study F2's video-codec harness (include/neuralfx/video_codec.hpp): the container payload parsers on hand-made
// files, and (when ffmpeg and its encoders exist) a round trip through H.264 with stacked alpha and VP9 with alpha.
#include <neuralfx/metrics.hpp>
#include <neuralfx/video_codec.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <unistd.h>

using namespace nfx;

namespace {

void put_le(std::vector<std::uint8_t>& b, std::uint64_t v, int n) {
  for (int i = 0; i < n; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}

// An EBML element: id bytes, a one-byte size (body under 127 bytes) or an 8-byte size, then the body.
std::vector<std::uint8_t> el(std::vector<std::uint8_t> id, const std::vector<std::uint8_t>& body, bool long_size = false) {
  std::vector<std::uint8_t> e = std::move(id);
  if (long_size) {
    e.push_back(0x01);
    for (int i = 6; i >= 0; --i) e.push_back(static_cast<std::uint8_t>(body.size() >> (8 * i)));
  } else {
    e.push_back(static_cast<std::uint8_t>(0x80 | body.size()));
  }
  e.insert(e.end(), body.begin(), body.end());
  return e;
}

Clip test_clip(int size, int frames) {
  Clip c;
  c.allocate(size, frames);
  c.loop = true;
  for (int f = 0; f < frames; ++f) {
    auto fr = c.frame(f);
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        const double r = std::hypot(x - size / 2.0 - 3.0 * std::sin(f * 0.4), y - size / 2.0);
        const double a = std::clamp(1.0 - r / (0.4 * size), 0.0, 1.0);
        const std::size_t i = (static_cast<std::size_t>(y) * size + x) * 4;
        fr[i] = static_cast<std::uint8_t>(250 * a);
        fr[i + 1] = static_cast<std::uint8_t>(140 * a * a);
        fr[i + 2] = static_cast<std::uint8_t>(40 * a * a * a);
        fr[i + 3] = static_cast<std::uint8_t>(255 * a);
      }
    }
  }
  return c;
}

bool have_encoder(const std::string& name) {
  return std::system(("ffmpeg -hide_banner -encoders 2>/dev/null | grep -q ' " + name + " '").c_str()) == 0;
}

}  // namespace

TEST(VideoCodec, IvfPayloadCountsFramesOnly) {
  std::vector<std::uint8_t> f = {'D', 'K', 'I', 'F'};
  put_le(f, 0, 2);
  put_le(f, 32, 2);
  f.resize(32, 0);
  for (const int n : {10, 0, 300}) {
    put_le(f, static_cast<std::uint64_t>(n), 4);
    put_le(f, 0, 8);
    f.insert(f.end(), static_cast<std::size_t>(n), 0x55);
  }
  auto p = video::ivf_payload(f);
  ASSERT_TRUE(p.has_value()) << p.error();
  EXPECT_EQ(*p, 310u);
  f.pop_back();  // a truncated frame
  EXPECT_FALSE(video::ivf_payload(f).has_value());
  EXPECT_FALSE(video::ivf_payload(std::vector<std::uint8_t>(40, 0)).has_value());
}

TEST(VideoCodec, WebmPayloadCountsBlocksAndAdditions) {
  // SimpleBlock: track 1 (0x81), timecode 0, flags; then 20 bytes of frame. A BlockGroup with a Block of 7 bytes and a
  // BlockAddition of 5 bytes (the alpha frame). Headers and other elements are not payload.
  std::vector<std::uint8_t> simple = {0x81, 0x00, 0x00, 0x80};
  simple.insert(simple.end(), 20, 0x11);
  std::vector<std::uint8_t> block = {0x81, 0x00, 0x01, 0x00};
  block.insert(block.end(), 7, 0x22);
  const auto more = el({0xA6}, [] {
    auto id = el({0xEE}, {0x81});
    auto add = el({0xA5}, std::vector<std::uint8_t>(5, 0x33));
    id.insert(id.end(), add.begin(), add.end());
    return id;
  }());
  auto group_body = el({0xA1}, block);
  const auto additions = el({0x75, 0xA1}, more);
  group_body.insert(group_body.end(), additions.begin(), additions.end());
  auto cluster_body = el({0xE7}, {0x00});  // timecode
  const auto sb = el({0xA3}, simple), bg = el({0xA0}, group_body);
  cluster_body.insert(cluster_body.end(), sb.begin(), sb.end());
  cluster_body.insert(cluster_body.end(), bg.begin(), bg.end());
  const auto cluster = el({0x1F, 0x43, 0xB6, 0x75}, cluster_body, true);
  auto segment_body = el({0x15, 0x49, 0xA9, 0x66}, {0x2A, 0xD7, 0xB1, 0x83, 0x0F, 0x42, 0x40});  // Info
  segment_body.insert(segment_body.end(), cluster.begin(), cluster.end());
  auto file = el({0x1A, 0x45, 0xDF, 0xA3}, {0x42, 0x82, 0x84, 'w', 'e', 'b', 'm'});  // EBML header
  const auto segment = el({0x18, 0x53, 0x80, 0x67}, segment_body, true);
  file.insert(file.end(), segment.begin(), segment.end());
  auto p = video::webm_payload(file);
  ASSERT_TRUE(p.has_value()) << p.error();
  EXPECT_EQ(*p, 20u + 7u + 5u);
  file.resize(file.size() - 3);  // cut inside the last block
  EXPECT_FALSE(video::webm_payload(file).has_value());
}

TEST(VideoCodec, RoundTripsThroughFfmpegWithAlpha) {
  if (std::system("ffmpeg -version > /dev/null 2>&1") != 0) GTEST_SKIP() << "ffmpeg not installed";
  const Clip clip = test_clip(32, 6);
  const auto dir = std::filesystem::temp_directory_path() / std::format("nfx_video_{}", ::getpid());
  for (const auto& codec : video::codecs()) {
    if (codec.name != "x264" && codec.name != "vp9") continue;
    if (!have_encoder(codec.encoder)) continue;
    const auto r = video::code(clip, codec, codec.ladder[2], dir);
    ASSERT_TRUE(r.has_value()) << r.error();
    EXPECT_EQ(r->decoded.frames, clip.frames);
    EXPECT_GT(r->payload_bytes, 0u);
    EXPECT_LE(r->payload_bytes, r->file_bytes);
    const auto s = metrics::score(clip, r->decoded);
    EXPECT_GT(s.active_psnr, 30.0) << codec.name;  // colour and alpha both come back
    std::size_t visible = 0;
    for (int f = 0; f < clip.frames; ++f) {
      for (std::size_t i = 3; i < clip.frame_bytes(); i += 4) visible += r->decoded.frame(f)[i] > 128;
    }
    EXPECT_GT(visible, 100u) << codec.name;
  }
  std::filesystem::remove_all(dir);
}
