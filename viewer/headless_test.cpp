// Windowless viewer checks: library and volume playback without GLFW (neuralfx.ViewerHeadless.*).
#include "headless.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
namespace headless = nfx::viewer::headless;

namespace {

std::string source_root() {
#ifdef NEURALFX_SOURCE_DIR
  return NEURALFX_SOURCE_DIR;
#else
  return ".";
#endif
}

std::string data_root() {
  if (const char* d = std::getenv("NEURALVFX_DATA"); d && *d) return d;
  if (const char* h = std::getenv("HOME"); h && *h) return std::string(h) + "/nvfx-data";
  return {};
}

std::string library_models() {
  const auto data = data_root();
  if (!data.empty()) {
    const fs::path p = fs::path(data) / "library/models";
    std::error_code ec;
    if (fs::exists(p / "fire.nvfx", ec)) return p.string();
  }
  return {};
}

}  // namespace

TEST(ViewerHeadless, ProceduralSparksRenders) {
  std::vector<std::uint8_t> rgba;
  int w = 0, h = 0;
  std::string err;
  const fs::path cat = fs::path(source_root()) / "library";
  ASSERT_TRUE(headless::library_play(cat.string(), {}, "sparks", 90, rgba, w, h, 2, &err)) << err;
  EXPECT_EQ(rgba.size(), static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
  const auto s = headless::luma_stats(rgba, w, h, 0, 0, w, h);
  EXPECT_TRUE(headless::picture_ok(s, 2.0, 3.0)) << "mean " << s.mean << " spread " << s.spread;
}

TEST(ViewerHeadless, LibraryCampfireNeedsModels) {
  const std::string models = library_models();
  if (models.empty()) GTEST_SKIP() << "no $NEURALVFX_DATA/library/models (run models/restore.sh)";
  std::vector<std::uint8_t> rgba;
  int w = 0, h = 0;
  std::string err;
  const fs::path cat = fs::path(source_root()) / "library";
  ASSERT_TRUE(headless::library_play(cat.string(), models, "campfire", 120, rgba, w, h, 2, &err)) << err;
  const auto s = headless::luma_stats(rgba, w, h, 0, 0, w, h);
  EXPECT_TRUE(headless::picture_ok(s));
}

TEST(ViewerHeadless, LibrarySmokeColumnRenders) {
  const std::string models = library_models();
  if (models.empty()) GTEST_SKIP() << "no library models";
  std::vector<std::uint8_t> rgba;
  int w = 0, h = 0;
  std::string err;
  const fs::path cat = fs::path(source_root()) / "library";
  ASSERT_TRUE(headless::library_play(cat.string(), models, "smoke_column", 150, rgba, w, h, 2, &err)) << err;
  const auto s = headless::luma_stats(rgba, w, h, w / 5, h / 5, 4 * w / 5, 4 * h / 5);
  EXPECT_TRUE(headless::picture_ok(s));
}

TEST(ViewerHeadless, VolumeFireHresRenders) {
  const fs::path model = fs::path(source_root()) / "models/3d/fire_hres.nvfx3";
  if (!fs::is_regular_file(model)) GTEST_SKIP() << "no shipped fire_hres.nvfx3";
  std::vector<std::uint8_t> rgba;
  int w = 0, h = 0;
  std::string err;
  ASSERT_TRUE(headless::volume_play(model.string(), 45, rgba, w, h, 2, &err)) << err;
  EXPECT_EQ(rgba.size(), static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
  const auto s = headless::luma_stats(rgba, w, h, w / 4, h / 4, 3 * w / 4, 3 * h / 4);
  EXPECT_TRUE(headless::picture_ok(s, 3.0, 3.0));
}

TEST(ViewerHeadless, LibraryFrameBufferMatchesStageSize) {
  std::vector<std::uint8_t> rgba;
  int w = 0, h = 0;
  const fs::path cat = fs::path(source_root()) / "library";
  ASSERT_TRUE(headless::library_play(cat.string(), {}, "sparks", 30, rgba, w, h));
  EXPECT_NE(w, h);  // gallery stages are not square; catches width-only texture bugs
  EXPECT_EQ(rgba.size(), static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
}
