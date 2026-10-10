// G3b (docs/DCM.md §8): a clip coded as a frame model's output plus a coded residual. The prediction of every frame is
// what the model renders for it (study A's frame models reproduce one clip); the stream holds the residual, the true
// frame minus the prediction in 8-bit RGBA, quantised with one step for all channels and coded by the residual coder
// (rcoder.hpp), frame by frame. The prediction does not depend on decoded data (open loop), and the coder's contexts
// use only coded integers, so a decoder whose model renders a pixel one level differently (another instruction set)
// still decodes every integer and is off by that level there.
#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace nfx::codec {

struct ClipResidual {
  std::vector<std::uint8_t> stream;
  std::vector<std::uint8_t> frames;  // the decoder's output, frames * size * size * 4
};

// pred and truth: frames * size * size * 4. step: the quantiser step in 8-bit levels (>= 1); round: the rounding offset
// (0.5 rounds to nearest; less is a dead zone).
ClipResidual encode_clip_residual(std::span<const std::uint8_t> pred, std::span<const std::uint8_t> truth, int size, int frames, float step,
                                  float round = 0.5f);
// Refuses a damaged or truncated stream, or one whose size and frame count do not match the prediction.
std::expected<std::vector<std::uint8_t>, std::string> decode_clip_residual(std::span<const std::uint8_t> pred, std::span<const std::uint8_t> stream);

}  // namespace nfx::codec
