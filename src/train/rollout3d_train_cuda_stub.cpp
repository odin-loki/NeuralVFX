#include <neuralfx/rollout3d_train_cuda.hpp>

#include <stdexcept>

namespace nfx::rollout3d {

bool cuda_train_available() { return false; }

void cuda_train_stepper(Model&, std::span<const Run>, const TrainOptions&, TrainState&,
                        const std::function<void(const Model&, const TrainState&)>&) {
  throw std::runtime_error("rollout3d: CUDA training was not built (cmake -DNEURALFX_CUDA=ON)");
}

double window_loss_cuda(const Model&, const Run&, int, int, int, float, float, std::uint64_t, std::vector<float>*, float, int) {
  throw std::runtime_error("rollout3d: CUDA training was not built (cmake -DNEURALFX_CUDA=ON)");
}

}  // namespace nfx::rollout3d
