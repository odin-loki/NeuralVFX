// Optional CUDA backend for study N's 3D rollout trainer (rollout3d_train.hpp). The forward and backward pass match the
// CPU trainer; convolutions are launched in Y slabs (`TrainOptions::shards` CUDA streams). Built when NEURALFX_CUDA is
// on at configure time.
#pragma once

#include <neuralfx/rollout3d_train.hpp>

#include <functional>

namespace nfx::rollout3d {

bool cuda_train_available();

void cuda_train_stepper(Model& m, std::span<const Run> runs, const TrainOptions& o, TrainState& s,
                        const std::function<void(const Model&, const TrainState&)>& checkpoint = {});

// Same contract as window_loss; gradients must match the CPU path within float tolerance (tests).
double window_loss_cuda(const Model& m, const Run& r, int first, int unroll, int burn, float sigma, float profile, std::uint64_t noise_seed,
                        std::vector<float>* grad, float activity = 0.f, int shards = 4);

}  // namespace nfx::rollout3d
