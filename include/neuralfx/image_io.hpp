// Image and video output for previews, figures and side-by-side videos. Everything written here belongs outside
// git (data root or a scratch directory), except figures deliberately placed under docs/.
#pragma once

#include <neuralfx/clip.hpp>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace nfx {

// An RGBA8 image, rows top to bottom.
struct Image {
  int width = 0, height = 0;
  std::vector<std::uint8_t> rgba;
  void allocate(int w, int h) {
    width = w;
    height = h;
    rgba.assign(static_cast<std::size_t>(w) * h * 4, 0);
  }
  std::uint8_t* pixel(int x, int y) { return rgba.data() + (static_cast<std::size_t>(y) * width + x) * 4; }
};

// Composite a premultiplied RGBA frame over a solid or checker background, for viewing.
enum class Background { black, grey, checker };
bool parse_background(std::string_view text, Background& out);
void composite(std::span<const std::uint8_t> premul, int size, Background bg, std::span<std::uint8_t> rgba_out);

// Copy a size x size RGBA tile into an image at (x, y), with an integer upscale; clipped to the image.
void blit(Image& dst, std::span<const std::uint8_t> src, int size, int x, int y, int scale = 1);

std::expected<void, std::string> write_png(const std::filesystem::path& path, const Image& img);  // zlib
std::expected<void, std::string> write_pam(const std::filesystem::path& path, const Image& img);  // RGB_ALPHA

// A contact sheet: `columns` frames per row, every `step`-th frame, composited over `bg`.
Image contact_sheet(const Clip& clip, int columns, int step, Background bg, int scale = 1);

// Rows of clips side by side (each row a clip, each column a time), for comparisons.
Image comparison_sheet(std::span<const Clip* const> rows, int columns, Background bg, int scale = 1);

// Pipe frames to ffmpeg (H.264 mp4).
std::expected<void, std::string> write_video(const std::filesystem::path& path, std::span<const Image> frames, float fps);

// Side-by-side video of several clips (same size), composited over `bg`, played `loops` times.
std::expected<void, std::string> write_comparison_video(const std::filesystem::path& path,
                                                        std::span<const Clip* const> clips, Background bg, int scale,
                                                        int loops);

}  // namespace nfx
