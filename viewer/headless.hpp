// Windowless playback checks for the viewer (library and volume APIs). Used by GTest and Xvfb screenshot tests.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace nfx::viewer::headless {

struct LumaStats {
  double mean = 0;
  double spread = 0;
};

// Mean and standard deviation of luma in a sub-rectangle of premultiplied RGBA8.
LumaStats luma_stats(std::span<const std::uint8_t> rgba, int width, int height, int x0, int y0, int x1, int y1);

// True when the picture has visible structure (not empty UI chrome).
bool picture_ok(const LumaStats& s, double min_mean = 2.0, double min_spread = 4.0);

// Play one library entry for `frames` steps at 60 Hz; writes the last scene frame to rgba (w*h*4).
bool library_play(const std::string& catalogue_dir, const std::string& models_dir, const char* entry_name, int frames,
                  std::vector<std::uint8_t>& rgba, int& width, int& height, int threads = 2, std::string* error = nullptr);

// Step a .nvfx3 volume and ray-march one picture (default 512x288).
bool volume_play(const std::string& model_path, int frames, std::vector<std::uint8_t>& rgba, int& width, int& height, int threads = 4,
                 std::string* error = nullptr);

}  // namespace nfx::viewer::headless
