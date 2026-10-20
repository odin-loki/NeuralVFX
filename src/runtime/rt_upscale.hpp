// Study P (docs/STUDY_P.md): a rollout effect drawn at half size and brought to full size, in the runtime.
//
// The upscaler of include/neuralfx/upscale.hpp (the float reference, src/core/upscale.cpp) for one frame size: the half-size
// frame to planar floats, the network's convolutions on rows of the half-size frame, then each pair of full-size rows from
// the bilinear 2x, the residuals and the seeded micro-detail, written as RGBA8 with the instance's colour matrix. The
// noise slices are cached (two at a time, recomputed only when the time crosses a slice). Every buffer is allocated when
// the Upscaler is made, so run() allocates nothing. With no network it is the bilinear 2x alone (what a GPU's bilinear
// sampler gives), which needs no file.
#pragma once

#include <neuralfx/upscale.hpp>

#include "rt_common.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace nfx::rt {

class Upscaler {
 public:
  virtual ~Upscaler() = default;
  // lo: the n x n half-size frame (premultiplied RGBA8, rows top to bottom, lo_stride bytes apart); hi: the 2n x 2n frame
  // (rows hi_stride bytes apart). The micro-detail's noise is that of `seconds` from `seed`; the colour matrix of `in`
  // is applied to the result (in.apply_colour). No allocation.
  virtual void run(const std::uint8_t* lo, std::size_t lo_stride, std::uint64_t seed, double seconds, const FrameInput& in, std::uint8_t* hi,
                   std::size_t hi_stride) = 0;
  [[nodiscard]] virtual std::size_t scratch_bytes() const = 0;
};

// One factory per ISA build (rt_upscale_base.cpp, rt_upscale_avx2.cpp, rt_upscale_avx512.cpp) for a half size n (a
// multiple of 16). net == nullptr: the bilinear 2x alone. The net must outlive the upscaler. Throws std::bad_alloc, and
// std::invalid_argument for a size it does not take.
namespace isa_base {
std::unique_ptr<Upscaler> make_upscaler(const upscale::Net* net, int n);
}  // namespace isa_base
namespace isa_avx2 {
std::unique_ptr<Upscaler> make_upscaler(const upscale::Net* net, int n);
}  // namespace isa_avx2
namespace isa_avx512 {
std::unique_ptr<Upscaler> make_upscaler(const upscale::Net* net, int n);
}  // namespace isa_avx512

}  // namespace nfx::rt
