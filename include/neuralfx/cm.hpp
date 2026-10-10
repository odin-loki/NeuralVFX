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
  // Optional, 8-bit tensors without channels: the affine map of each plane, value = lo + q / qmax * (hi - lo), in
  // fixed point (the value times 2^24, as from fp16). Neighbours in other planes are mapped into this plane's scale.
  std::vector<std::int64_t> lo, hi;
  // Bits per value of an 8-bit-wide tensor (2 to 8): values in [0, 2^bits - 1], qmax = 2^bits - 1, and only the low
  // `bits` bits are coded. 8 for everything except bit-packed feature planes (model.hpp); not serialised.
  int bits = 8;

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

// The bytes of each kind of value in a .nvfx file, in the order they are coded (for reports: zlib on each part).
std::vector<std::pair<Kind, std::vector<std::uint8_t>>> split_model(std::span<const std::uint8_t> file);

// Tensors without a container (flipbooks): shapes and values in, shapes and values out.
Packed pack_tensors(std::span<const Tensor> tensors);
std::expected<std::vector<Tensor>, std::string> unpack_tensors(std::span<const std::uint8_t> packed);

// The fp16 reordering used for coding: a bijection on 16-bit patterns that is monotone in the value (-0 just below
// +0, NaNs at the ends). Exposed for the tests.
std::uint16_t f16_order(std::uint16_t bits);
std::uint16_t f16_unorder(std::uint16_t code);

const char* kind_name(Kind k);

}  // namespace nfx::cm
