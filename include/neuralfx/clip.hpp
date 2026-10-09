// Clips: square RGBA8 animations with premultiplied alpha, plus what made them (effect, controls, seed, source).
// Blend contract for every clip and every renderer in this project: dst = src.rgb + dst * (1 - src.a). Emission
// (fire) has rgb > 0 with small alpha; smoke has alpha > 0 with rgb = colour * alpha.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace nfx {

inline constexpr int kMaxControls = 8;

struct Clip {
  int size = 0;  // width and height in pixels
  int frames = 0;
  float fps = 30.f;
  bool loop = false;    // the last frame flows into the first
  std::string effect;   // "fire", "smoke", "explosion", or an ingested name
  std::string source;   // "sim" or the ingest id
  std::array<float, kMaxControls> controls{};  // normalised to [0, 1]; meaning set by the effect
  int n_controls = 0;
  std::uint64_t seed = 0;
  std::vector<std::uint8_t> rgba;  // frames * size * size * 4

  std::size_t frame_bytes() const { return static_cast<std::size_t>(size) * size * 4; }
  std::size_t pixels() const { return static_cast<std::size_t>(size) * size; }
  std::span<std::uint8_t> frame(int i) { return {rgba.data() + frame_bytes() * static_cast<std::size_t>(i), frame_bytes()}; }
  std::span<const std::uint8_t> frame(int i) const {
    return {rgba.data() + frame_bytes() * static_cast<std::size_t>(i), frame_bytes()};
  }
  // Channel c of pixel (x, y) in frame f.
  std::uint8_t operator[](int f, int y, int x, int c) const {
    return rgba[((static_cast<std::size_t>(f) * size + y) * size + x) * 4 + c];
  }
  std::span<const float, kMaxControls> control_span() const { return controls; }
  void allocate(int s, int n) {
    size = s;
    frames = n;
    rgba.clear();  // clear + resize (zero-filled): assign(n, 0) trips a GCC 14 -Wnonnull false positive when n may be 0
    rgba.resize(frame_bytes() * static_cast<std::size_t>(n));
  }
};

// Binary .nfxclip: a fixed little-endian header then the frames.
std::expected<void, std::string> write_clip(const std::filesystem::path& path, const Clip& clip);
std::expected<Clip, std::string> read_clip(const std::filesystem::path& path);

// Frames [first, first + count) of a clip as a new clip (metadata copied; loop kept only for the whole clip).
Clip slice_clip(const Clip& clip, int first, int count);

// The data root for clips, weights and videos: $NEURALVFX_DATA, else ~/nvfx-data. Never inside the repository.
std::filesystem::path data_root();

}  // namespace nfx
