// The baselines a neural effect must beat (docs/PLAN.md §6, Phase 2): texture flipbooks at matched memory.
//
// A flipbook keeps some frames of a reference clip at some resolution, stored raw (RGBA8, 32 bits per pixel) or
// block-compressed. The studies' original baseline used our own encoder for the BC3 layout (BC1 colour + BC4 alpha,
// 8 bits per pixel). The production block formats come from open-source encoders fetched at build time
// (NEURALFX_FETCH_ENCODERS; cmake/encoders.cmake, docs/DATA.md): BC7 (8 bits per pixel) and ASTC at block sizes
// 4x4 to 12x12 (8 to 0.89 bits per pixel). Playback does what a game does: bilinear texture filtering and linear
// blending between the two nearest kept frames, optionally with motion vectors (each kept frame stores a
// low-resolution flow to the next, and both frames are warped towards the in-between time).
#pragma once

#include <neuralfx/clip.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nfx::flipbook {

// How a flipbook's frames are stored.
//   raw        RGBA8, 32 bits per pixel
//   bc3        the BC3 layout (BC1 + BC4) by our own encoder (src/core/flipbook.cpp), 8 bits per pixel
//   bc7        BC7 by bc7e (Binomial's encoder, all eight modes; its plain C++ port in Basis Universal), decoded by
//              bc7enc_rdo's reference decoder, 8 bits per pixel
//   astcNxN    ASTC, LDR profile, by Arm's astc-encoder at its "thorough" preset, decoded by the same library to
//              8 bits per channel: 128 bits per N x N block (4x4: 8, 5x5: 5.12, 6x6: 3.56, 8x8: 2, 10x10: 1.28,
//              12x12: 0.89 bits per pixel); a frame that is not a whole number of blocks is padded to one
// Both production encoders minimise plain RGBA error (no perceptual weights), the error the studies score.
enum class Codec { raw, bc3, bc7, astc4x4, astc5x5, astc6x6, astc8x8, astc10x10, astc12x12 };

std::string_view codec_name(Codec codec);  // "raw", "bc3", "bc7", "astc4x4", ... as in Spec::describe()
int block_dim(Codec codec);                // texels per block side (raw: 1)
// Bytes of one stored res x res frame: pixels x 4 (raw) or whole blocks x 16.
std::size_t frame_bytes(Codec codec, int res);
// bc7 and astc*: encoded by a fetched production encoder.
bool production(Codec codec);
// Whether this build can encode the format (the production ones need their encoder fetched at build time).
bool available(Codec codec);
// The production formats, in ladder order.
std::span<const Codec> production_codecs();

struct Spec {
  int frames = 64;        // kept frames (ignored when an explicit keep list is given)
  int res = 128;          // stored resolution per side
  Codec codec = Codec::bc3;
  int flow_res = 0;       // motion-vector resolution per side; 0 = no motion vectors
  std::string describe() const;  // "bc3 64f 128px", "astc8x8 16f 64px +mv16"
  // The family the study tools file it under: flipbook_raw, flipbook_bc3, flipbook_mv (BC3 with motion vectors, as
  // first named), and for production formats flipbook_bc7 / flipbook_astc with "_mv" added for motion vectors.
  std::string family() const;
};

// Bytes the flipbook occupies in memory: frames x stored frame bytes, plus 2 bytes per flow vector.
std::size_t memory_bytes(const Spec& spec, int kept_frames);

struct Flipbook {
  Spec spec;
  int out_size = 0;      // playback resolution (the reference clip's)
  int source_frames = 0; // frames in the reference clip
  bool loop = false;
  std::vector<int> kept;                         // source frame index of each kept frame, ascending
  std::vector<std::vector<std::uint8_t>> stored; // the stored bytes of each kept frame (encode()), what memory holds
  std::vector<std::vector<std::uint8_t>> frames; // decoded RGBA at spec.res, one per kept frame
  std::vector<std::vector<float>> flow;          // per kept frame: (dx, dy) to the next kept frame, output pixels
  std::size_t bytes = 0;
};

// The frames of one clip, downsampled, encoded and decoded once per (format, resolution, source frame), for all the
// flipbooks built from that clip (a ladder encodes each frame once instead of once per configuration). One cache per
// clip; safe to share between threads.
class FrameCache {
 public:
  struct Frame {
    std::vector<std::uint8_t> stored, decoded;
  };
  FrameCache();
  ~FrameCache();
  FrameCache(const FrameCache&) = delete;
  FrameCache& operator=(const FrameCache&) = delete;
  std::shared_ptr<const Frame> get(const Clip& ref, Codec codec, int res, int frame);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Build a flipbook from `ref`. `keep` lists the source frames available to it (for example only the even frames
// in the frame-interpolation test); empty = `spec.frames` frames evenly spaced over the clip. With a cache, frames
// already encoded for another flipbook of the same clip are reused (the result is the same).
Flipbook build(const Clip& ref, const Spec& spec, std::span<const int> keep = {}, FrameCache* cache = nullptr);

// Play back at every source frame time 0..source_frames-1, at out_size: what a game would show.
Clip play(const Flipbook& fb);

// The standard set of configurations for a clip of `size` pixels and `frames` frames: the studies' original 31, in our
// BC3 layout and raw RGBA8.
std::vector<Spec> ladder(int size, int frames);
// The same frame counts and resolutions (with motion vectors where the BC3 ladder has them) in each production format
// this build has: 21 configurations per format, none when the encoders were not fetched.
std::vector<Spec> ladder_production(int size, int frames);

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

// A whole RGBA image in any format: the stored bytes (raw: the pixels; block formats: 16-byte blocks, row by row) and
// back to RGBA. Block formats other than ASTC need width and height multiples of 4. Throws std::runtime_error for a
// production format this build cannot encode (available()). Deterministic, and safe to call from several threads.
std::vector<std::uint8_t> encode(Codec codec, std::span<const std::uint8_t> rgba, int width, int height);
std::vector<std::uint8_t> decode(Codec codec, std::span<const std::uint8_t> stored, int width, int height);

// --- helpers shared with the evaluation ------------------------------------------------------------------------

// Box-filter downsample of a size x size RGBA image to res x res (size a multiple of res).
std::vector<std::uint8_t> downsample(std::span<const std::uint8_t> rgba, int size, int res);
// Bilinear sample of an RGBA image of side `res` at continuous pixel coordinates (clamped), into out[4] as floats.
void sample_bilinear(std::span<const std::uint8_t> rgba, int res, float x, float y, std::span<float, 4> out);

}  // namespace nfx::flipbook
