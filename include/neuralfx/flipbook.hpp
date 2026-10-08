// The baselines a neural effect must beat (docs/PLAN.md §6, Phase 2): texture flipbooks at matched memory.
//
// A flipbook keeps some frames of a reference clip at some resolution, stored raw (RGBA8, 32 bits per pixel) or
// block-compressed (BC1 colour + BC4 alpha = the BC3 layout, 8 bits per pixel; our own encoder, so a lower bound
// on what a production BC7 encoder reaches at the same size). Playback does what a game does: bilinear texture
// filtering and linear blending between the two nearest kept frames, optionally with motion vectors (each kept
// frame stores a low-resolution flow to the next, and both frames are warped towards the in-between time).
#pragma once

#include <neuralfx/clip.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace nfx::flipbook {

enum class Codec { raw, bc3 };

struct Spec {
  int frames = 64;        // kept frames (ignored when an explicit keep list is given)
  int res = 128;          // stored resolution per side
  Codec codec = Codec::bc3;
  int flow_res = 0;       // motion-vector resolution per side; 0 = no motion vectors
  std::string describe() const;
};

// Bytes the flipbook occupies in memory: frames x pixels x bits per pixel, plus 2 bytes per flow vector.
std::size_t memory_bytes(const Spec& spec, int kept_frames);

struct Flipbook {
  Spec spec;
  int out_size = 0;      // playback resolution (the reference clip's)
  int source_frames = 0; // frames in the reference clip
  bool loop = false;
  std::vector<int> kept;                         // source frame index of each kept frame, ascending
  std::vector<std::vector<std::uint8_t>> frames; // decoded RGBA at spec.res, one per kept frame
  std::vector<std::vector<float>> flow;          // per kept frame: (dx, dy) to the next kept frame, output pixels
  std::size_t bytes = 0;
};

// Build a flipbook from `ref`. `keep` lists the source frames available to it (for example only the even frames
// in the frame-interpolation test); empty = `spec.frames` frames evenly spaced over the clip.
Flipbook build(const Clip& ref, const Spec& spec, std::span<const int> keep = {});

// Play back at every source frame time 0..source_frames-1, at out_size: what a game would show.
Clip play(const Flipbook& fb);

// The standard set of configurations for a clip of `size` pixels and `frames` frames.
std::vector<Spec> ladder(int size, int frames);

// --- block compression ------------------------------------------------------------------------------------------

// BC1: 16 RGB texels (row-major 4x4, 4 bytes each, alpha ignored) <-> 8 bytes (two RGB565 endpoints, 2-bit indices).
std::array<std::uint8_t, 8> encode_bc1(std::span<const std::uint8_t, 64> rgba);
void decode_bc1(std::span<const std::uint8_t, 8> block, std::span<std::uint8_t, 64> rgba);  // writes rgb only
// BC4: 16 single-channel values <-> 8 bytes (two 8-bit endpoints, 3-bit indices).
std::array<std::uint8_t, 8> encode_bc4(std::span<const std::uint8_t, 16> v);
void decode_bc4(std::span<const std::uint8_t, 8> block, std::span<std::uint8_t, 16> v);

// A whole RGBA image (width and height multiples of 4) through BC3-layout compression and back.
std::vector<std::uint8_t> compress_bc3(std::span<const std::uint8_t> rgba, int width, int height);
std::vector<std::uint8_t> decompress_bc3(std::span<const std::uint8_t> blocks, int width, int height);

// --- helpers shared with the evaluation ------------------------------------------------------------------------

// Box-filter downsample of a size x size RGBA image to res x res (size a multiple of res).
std::vector<std::uint8_t> downsample(std::span<const std::uint8_t> rgba, int size, int res);
// Bilinear sample of an RGBA image of side `res` at continuous pixel coordinates (clamped), into out[4] as floats.
void sample_bilinear(std::span<const std::uint8_t> rgba, int res, float x, float y, std::span<float, 4> out);

}  // namespace nfx::flipbook
