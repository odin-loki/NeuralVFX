// Phase 1b: licence checks, alpha modes, loops, the register, and decoding frames and (when ffmpeg exists) video.
#include <neuralfx/ingest.hpp>

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <unistd.h>

using namespace nfx;
namespace fs = std::filesystem;

namespace {

fs::path temp_dir(std::string_view name) {
  const auto d = fs::temp_directory_path() / std::format("nfx_ingest_{}_{}", ::getpid(), name);
  fs::remove_all(d);
  fs::create_directories(d);
  return d;
}

void write_pam(const fs::path& p, int size, std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) {
  std::ofstream o(p, std::ios::binary);
  o << std::format("P7\nWIDTH {}\nHEIGHT {}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", size, size);
  for (int i = 0; i < size * size; ++i) o << r << g << b << a;
}

// Pixel channel through the multidimensional subscript (kept out of the test macros, whose commas would split it).
int px(const Clip& c, int f, int y, int x, int ch) { return c[f, y, x, ch]; }

}  // namespace

TEST(Ingest, LicencesAreCheckedBeforeAnything) {
  using L = ingest::Licence;
  EXPECT_TRUE(ingest::check_licence(L{"own", "shot by the studio", "", ""}).has_value());
  EXPECT_TRUE(ingest::check_licence(L{"CC0-1.0", "https://example.org/a", "", ""}).has_value());
  EXPECT_TRUE(ingest::check_licence(L{"CC-BY-4.0", "https://example.org/b", "A. Artist", ""}).has_value());
  EXPECT_TRUE(ingest::check_licence(L{"commercial", "vendor pack", "", "EULA 2.1 allows ML training"}).has_value());
  EXPECT_FALSE(ingest::check_licence(L{"CC-BY-4.0", "https://example.org/b", "", ""}).has_value());  // no author
  EXPECT_FALSE(ingest::check_licence(L{"CC-BY-NC-4.0", "x", "y", ""}).has_value());
  EXPECT_FALSE(ingest::check_licence(L{"CC-BY-ND-4.0", "x", "y", ""}).has_value());
  EXPECT_FALSE(ingest::check_licence(L{"unknown", "x", "", ""}).has_value());
  EXPECT_FALSE(ingest::check_licence(L{"", "x", "", ""}).has_value());
  EXPECT_FALSE(ingest::check_licence(L{"commercial", "vendor pack", "", ""}).has_value());  // no agreement named
  EXPECT_FALSE(ingest::check_licence(L{"own", "", "", ""}).has_value());                      // no source
  EXPECT_FALSE(ingest::check_licence(L{"GPL-3.0", "x", "y", ""}).has_value());                // not on the list
}

TEST(Ingest, FramesWithEachAlphaModeAndALoop) {
  const auto dir = temp_dir("frames");
  for (int f = 0; f < 6; ++f) write_pam(dir / std::format("f{:03}.pam", f), 16, static_cast<std::uint8_t>(40 * f), 100, 20, 128);
  ingest::Options o;
  o.name = "test";
  o.alpha = ingest::AlphaMode::keep;
  auto c = ingest::from_frames(dir, o);
  ASSERT_TRUE(c.has_value()) << c.error();
  EXPECT_EQ(c->frames, 6);
  EXPECT_EQ(c->size, 16);
  EXPECT_EQ(px(*c, 0, 0, 0, 1), (100 * 128 + 127) / 255);  // premultiplied
  o.alpha = ingest::AlphaMode::luma;
  c = ingest::from_frames(dir, o);
  EXPECT_EQ(px(*c, 5, 3, 3, 3), 200);  // alpha = max(r, g, b)
  o.alpha = ingest::AlphaMode::additive;
  o.loop_blend = 2;
  c = ingest::from_frames(dir, o);
  EXPECT_EQ(c->frames, 4);
  EXPECT_TRUE(c->loop);
  EXPECT_EQ(px(*c, 0, 0, 0, 3), 0);
  fs::remove_all(dir);
}

TEST(Ingest, RegisterRecordsEveryClip) {
  const auto dir = temp_dir("register");
  Clip c;
  c.allocate(16, 3);
  c.effect = "smoke_a";
  const auto reg = dir / "licences.tsv";
  ASSERT_TRUE(ingest::register_clip(reg, {"CC-BY-SA-4.0", "https://example.org/s", "B. Author", ""}, "in.mov", "out.nfxclip", c).has_value());
  ASSERT_TRUE(ingest::register_clip(reg, {"own", "studio", "", ""}, "in2.mov", "out2.nfxclip", c).has_value());
  EXPECT_FALSE(ingest::register_clip(reg, {"CC-BY-NC-4.0", "x", "y", ""}, "in3.mov", "out3.nfxclip", c).has_value());
  std::ifstream in(reg);
  std::string line;
  int lines = 0;
  bool flagged = false;
  while (std::getline(in, line)) {
    ++lines;
    flagged = flagged || line.find("share-alike") != std::string::npos;
  }
  EXPECT_EQ(lines, 3);  // header + two accepted clips
  EXPECT_TRUE(flagged);
  fs::remove_all(dir);
}

TEST(Ingest, DecodesAVideoWhenFfmpegIsAvailable) {
  if (std::system("ffmpeg -version > /dev/null 2>&1") != 0) GTEST_SKIP() << "ffmpeg not installed";
  const auto dir = temp_dir("video");
  const auto video = dir / "test.mp4";
  ASSERT_EQ(std::system(std::format("ffmpeg -loglevel error -y -f lavfi -i testsrc=size=320x240:rate=25 -t 1 -pix_fmt yuv420p \"{}\"",
                                    video.string()).c_str()), 0);
  ingest::Options o;
  o.name = "testsrc";
  o.size = 64;
  o.fps = 30;
  auto c = ingest::from_video(video, o);
  ASSERT_TRUE(c.has_value()) << c.error();
  EXPECT_EQ(c->size, 64);
  EXPECT_GE(c->frames, 28);
  EXPECT_LE(c->frames, 31);
  EXPECT_FALSE(ingest::from_video(dir / "missing.mp4", o).has_value());
  fs::remove_all(dir);
}
