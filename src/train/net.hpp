// Trainer internals (also used by the gradient-check tests): per-thread workspaces that run the forward and backward
// pass of one frame and accumulate gradients into a Model-shaped buffer.
#pragma once

#include <neuralfx/model.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace nfx::train::detail {

// Gradients have the shape of the model; feature gradients are tracked by touched time slice.
struct Grads {
  Model g;
  std::vector<std::uint8_t> touched;  // per time slice: 1 if any feature gradient was written
  explicit Grads(const Model& m);
  void zero();
  void add(const Grads& o);  // this += o (features: touched slices only)
};

class Net {
 public:
  explicit Net(const Model& m);
  // Forward and backward for one frame. `target` is the RGBA8 frame at `size`; `pixels` the sampled pixel indices
  // (grid family; ignored by the conv family, which uses every pixel). Each squared channel error is weighted by
  // `scale`. Adds parameter gradients to `g` and the gradient with respect to the condition vector to `dc`.
  // Returns the unweighted sum of squared channel errors.
  double step(const Model& m, Grads& g, float t, std::span<const float> c, std::span<const std::uint8_t> target,
              std::span<const int> pixels, int size, float scale, std::span<float> dc);
  // Forward only, RGBA floats [size][size][4].
  void render(const Model& m, float t, std::span<const float> c, int size, std::span<float> rgba);

 private:
  struct Frame;  // per-frame conditioning state
  void begin_frame(const Model& m, float t, std::span<const float> c);
  void end_frame(const Model& m, Grads& g, std::span<const float> c, std::span<float> dc);
  double grid_pixels(const Model& m, Grads* g, std::span<const int> pixels, int size, std::span<const std::uint8_t> target,
                     float scale, std::span<float> out);
  double conv_frame(const Model& m, Grads* g, std::span<const std::uint8_t> target, float scale, std::span<float> out);

  // per-frame state
  int i0_ = 0, i1_ = 0;
  float ft_ = 0;
  std::vector<float> w_, film_, slice_, dslice_, dfilm_;
  // grid workspace (chunk of pixels, SoA [features][chunk])
  std::vector<float> x0_, z1_, a1_, dx_, dz_;
  std::vector<std::vector<float>> hs_;  // hidden activations per layer
  std::vector<int> cx_, cy_;
  std::vector<float> cfx_, cfy_;
  // conv workspace
  std::vector<float> pad0_, y0_, h0_, pad1_, y1_, h1_, pad2_, out_, dpad_, dy_, dh_;
};

}  // namespace nfx::train::detail
