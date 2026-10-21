// Inside the library only: the volume runtime's kernels (src/runtime/nvfx_volume.cpp; study N, docs/STUDY_3D.md).
#pragma once

#include <neuralfx/nvfx.h>
#include <neuralfx/rollout3d.hpp>

#include <memory>

namespace nfx::rt {

// The 3 x 3 x 3 convolution with AVX2 and FMA (rt_volume_avx2.cpp): the reference's sums in its order, fused.
void volume_conv3_avx2(int rx, int ry, int rz, const float* in, int ci, const float* W, const float* b, int co, float* out, int y0, int y1);

// The detail layer on vectors (rt_detail3.hpp): rollout3d::detail_step's values to the bit, for a model and a workspace
// of it. Allocates its buffers (about 20 MB for a 64 x 96 x 64 volume); its step() allocates nothing.
namespace d3_avx2 {
std::unique_ptr<rollout3d::DetailKernel> make_detail3(const rollout3d::Model& m, const rollout3d::Workspace& ws);
}  // namespace d3_avx2
namespace d3_avx512 {
std::unique_ptr<rollout3d::DetailKernel> make_detail3(const rollout3d::Model& m, const rollout3d::Workspace& ws);
}  // namespace d3_avx512

// The detail layer a volume takes on an ISA: the AVX-512 or AVX2 kernel, or none (the reference's detail_step) at the
// baseline. Every choice gives the same fields to the bit.
std::unique_ptr<rollout3d::DetailKernel> make_volume_detail(const rollout3d::Model& m, const rollout3d::Workspace& ws, nvfx_isa isa);

}  // namespace nfx::rt
