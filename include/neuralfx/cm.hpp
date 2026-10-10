// Lossless context-mixing coder for stored tensors: the weights, feature volumes and start states of .nvfx files,
// and (for fair comparisons) the blocks, pixels and motion vectors of flipbooks. Results: results/compression.
//
// The technique is that of PAQ and lpaq. Each value is coded bit by bit, most significant bit first, with a binary
// arithmetic coder (32-bit range, carry-less). The probability of each bit comes from:
//   - context models: seventeen statistics, each keyed by something already decoded (the value to the left, the
//     one above, the one in the previous plane or time slice, the bytes of the pixel or block decoded so far, a
//     numeric prediction from the neighbours, the row and column, the local error level), each giving a probability;
//   - two mixers: small online logistic regressions over those probabilities, with weights chosen by a small context
//     (one by the bit position and the local error level, one by the bits of the value decoded so far);
//   - two adaptive probability maps (APM / SSE) that correct the mixed probability in context.
// Everything the decoder computes is integer arithmetic in a fixed order, so every compiler and instruction set
// decodes the same bits. The encoder and decoder run the same model code; only the arithmetic coder differs.
//
// Values are 8-bit (features, fine fields, pixels, block bytes) or fp16 bit patterns. An fp16 value is coded as the
// high byte (sign, exponent and two mantissa bits) and then the low byte with the high byte as context, after a
// bijective reordering that makes the 16-bit code monotone in the value (so "near the prediction" means near in
// value, also across zero). Numeric predictions are made in the value domain: fp16 values as exact fixed point,
// 8-bit values through their plane's affine map, so that a neighbour in another plane is first mapped into this
// plane's scale.
//
// The runtime does not change: a packed file is unpacked to the .nvfx bytes at load, so resident memory is what it
// was; only the bytes on disk and in a download shrink.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nfx::cm {

// What a tensor holds: each kind learns its own statistics (the kinds share the model, not their contexts).
enum class Kind : std::uint8_t {
  bytes = 0,  // headers and anything without structure
  features,   // feature or latent volumes of frame models
  ranges,     // per-plane (lo, hi) of 8-bit feature planes, fp16
  weights,    // weight matrices and convolution kernels
  biases,     // bias vectors
  codes,      // variation codes and their statistics
  coarse,     // coarse start states of rollout effects [start][y][x][channel], fp16
  fine,       // 8-bit fine fields of rollout start points
  scales,     // fp16 scales of the fine fields
  bc3,        // BC3-layout blocks [frame][block row][block column][16 bytes]
  rgba,       // raw RGBA8 pixels [frame][y][x][4]
  flow,       // flipbook motion vectors [frame][y][x][2], 8-bit (quarter pixels, offset by 128)
  count
};

// The shape of a tensor. The last axis varies fastest. Without `channels`, the last two axes are x and y and every
// axis before them is a plane axis (time slice, channel, basis...). With `channels`, the last axis holds the channels
// of one pixel or the bytes of one block, the two before it are x and y, and the rest are plane axes.
struct Shape {
  Kind kind = Kind::bytes;
  int width = 1;                      // bytes per value: 1 (8-bit) or 2 (fp16 bit patterns)
  std::vector<std::uint32_t> dims;    // outermost first
  bool channels = false;
  // Optional, 8-bit tensors without channels: the affine map of each plane, value = lo + q / 255 * (hi - lo), in
  // fixed point (the value times 2^24, as from fp16). Neighbours in other planes are mapped into this plane's scale.
  std::vector<std::int64_t> lo, hi;

  std::size_t size() const;    // number of values
  std::size_t planes() const;  // number of planes (product of the plane axes)
};

// A tensor with its values: 8-bit values in the low byte, fp16 values as their bit patterns.
struct Tensor {
  Shape shape;
  std::vector<std::uint16_t> values;
};

// Coded size by kind, as the ideal code length the model assigned while encoding (the arithmetic coder adds a few
// bytes in all). `bytes` is the size of those values in the original.
struct Part {
  Kind kind = Kind::bytes;
  std::size_t bytes = 0;
  std::size_t values = 0;
  double coded_bytes = 0;
};

struct Packed {
  std::vector<std::uint8_t> data;
  std::vector<Part> parts;  // one entry per kind present, in Kind order
};

// A .nvfx file (frame model NVFXMDL1 or rollout effect NVFXROL1) into .nvfz, parsed into its tensors; anything else
// is coded as plain bytes. Unpacking restores the exact bytes (checked against a stored checksum). Files up to 256 MB.
Packed pack_model(std::span<const std::uint8_t> file);
std::expected<std::vector<std::uint8_t>, std::string> unpack_model(std::span<const std::uint8_t> packed);

// Format 2 (study H, docs/DCM.md §9): the same container and predictions with options for speed.
//   - lz: LZ tokens (as LZP). Before a value is coded, the value that followed the last occurrence of the four values
//     before it is offered, and one adaptive flag says whether it is the value; a match is followed while it holds.
//     Repeats (empty fields, saturated features, repeated headers) then cost a flag per value and no modelling.
//   - light: a light model instead of the full one: seven predictors instead of fourteen, and per bit four directly
//     indexed statistics, one small mixer and one APM (no hashed contexts, no second mixer). Many times faster to
//     decode, a few percent larger.
//   - fast: in the manner of LOCO-I (JPEG-LS): one of three cheap predictions per value, the residual as a Golomb-Rice
//     code with an adaptive parameter, its unary part coded with one adaptive statistic per decision and its low bits
//     plainly. Several times faster again than the light model, and larger.
//   - seekable (light or fast): every tensor, large ones in slices of whole planes of about `segment` values, is its own
//     segment with its own model and arithmetic coder, so one slice decodes without the others (list_slices,
//     unpack_slice). Headers and the small tensors that parsing reads (feature ranges, field scales) share segment 0,
//     which is always decoded. Each segment carries a 32-bit checksum.
// unpack_model and unpack_tensors read either format.
enum class Literal : std::uint8_t {
  full = 0,   // the full model of format 1
  light = 1,  // the light model
  fast = 2,   // a Golomb-Rice code of the residual of a cheap prediction (as LOCO-I), few coder steps per value
};
struct Options {
  Literal literal = Literal::light;
  bool lz = true;
  bool seekable = false;
  std::size_t segment = std::size_t{1} << 16;
};
Packed pack_model(std::span<const std::uint8_t> file, const Options& options);

// One segment of a seekable file: which tensor (in coding order, counting every tensor the parser codes), which planes.
struct SliceInfo {
  Kind kind = Kind::bytes;
  std::size_t values = 0;
  std::size_t packed_bytes = 0;
  std::size_t tensor = 0;
  std::size_t first_plane = 0, planes = 0;
};
std::expected<std::vector<SliceInfo>, std::string> list_slices(std::span<const std::uint8_t> packed);

// Segment `index` (of list_slices) alone: its tensor's shape, its values and where each value's bytes are in the file
// (`at`, as the parser saw them). Decodes segment 0 and this segment only; both checksums are checked.
struct Slice {
  Tensor tensor;  // the whole tensor's shape; values of planes [first_plane, first_plane + planes) only
  std::size_t first_plane = 0;
  std::vector<std::size_t> at;
};
std::expected<Slice, std::string> unpack_slice(std::span<const std::uint8_t> packed, std::size_t index);

// The bytes of each kind of value in a .nvfx file, in the order they are coded (for reports: zlib on each part).
std::vector<std::pair<Kind, std::vector<std::uint8_t>>> split_model(std::span<const std::uint8_t> file);

// Tensors without a container (flipbooks): shapes and values in, shapes and values out.
Packed pack_tensors(std::span<const Tensor> tensors);
Packed pack_tensors(std::span<const Tensor> tensors, const Options& options);
std::expected<std::vector<Tensor>, std::string> unpack_tensors(std::span<const std::uint8_t> packed);

// The fp16 reordering used for coding: a bijection on 16-bit patterns that is monotone in the value (-0 just below
// +0, NaNs at the ends). Exposed for the tests.
std::uint16_t f16_order(std::uint16_t bits);
std::uint16_t f16_unorder(std::uint16_t code);

const char* kind_name(Kind k);

}  // namespace nfx::cm
