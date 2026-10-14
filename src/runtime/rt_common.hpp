// Runtime internals shared by the C API (nvfx.cpp) and the per-ISA renderers (rt_impl.hpp).
#pragma once

#include <neuralfx/model.hpp>
#include <neuralfx/rollout.hpp>

#include "rt_aligned.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace nfx::rt {

// A loaded effect: the model with its features kept only in their storage format (fp16 or 8-bit; the float copy
// is dropped), all other weights as floats.
struct Effect {
  Model m;
  std::size_t stored_bytes = 0;
  std::size_t resident_bytes = 0;
};

// What one frame needs, prepared by the instance without allocating.
struct FrameInput {
  float t = 0;                      // model time in [0, 1]
  std::span<const float> c;         // condition vector (controls, variation code)
  std::array<float, 9> colour{};    // RGB matrix (hue rotation times brightness), row-major
  bool apply_colour = false;
};

class Renderer {
 public:
  virtual ~Renderer() = default;
  virtual void render(const FrameInput& in, std::uint8_t* rgba, std::size_t stride) = 0;
  virtual std::size_t scratch_bytes() const = 0;
  virtual double macs_per_pixel() const = 0;
};

// A loaded rollout effect (rollout.hpp): stepper, renderer and start points as floats.
struct RolloutEffect {
  rollout::Model m;
  std::size_t stored_bytes = 0;
  std::size_t resident_bytes = 0;
};

struct RolloutScratch;

// Study K (docs/REPORT.md §13): where a rollout frame's time goes. While a profile is attached
// (RolloutRunner::set_profile), step() and render() add the thread CPU time of each of their stages to it, in ms;
// detached (the default) nothing is timed and the frames are the same.
struct RolloutProfile {
  enum Stage {
    noise,         // the step's noise inputs, condition and FiLM folding
    conv,          // the stepper's network: inputs, two 3x3 convolutions, the output layer
    project,       // the state update, divergence, pressure projection and the coarse advection
    detail,        // the detail layer: the fine fields carried by the flow and locked to the coarse state
    shade_cells,   // the renderer's per-cell part: directional soot sums, the first layer's coarse part
    shade_pixels,  // the renderer per pixel: the rest of the first layer, the hidden layer, the output, the gate
    kStages
  };
  std::array<double, kStages> ms{};
};

// A running rollout effect: the coarse state, pressure and fine fields of one instance. Every buffer is allocated at
// construction; start(), step() and render() allocate nothing. Same operations as the reference in
// src/core/rollout.cpp (tests/test_runtime.cpp holds them to it).
class RolloutRunner {
 public:
  virtual ~RolloutRunner() = default;
  // Begin at start point `index` with a noise seed: its coarse state and fine fields (or fine fields upsampled from
  // the coarse state, to be grown by h.warmup ordinary steps).
  virtual void begin(int index, std::uint64_t seed) = 0;
  // begin(), then the warm-up steps when the start point has no fine fields.
  virtual void start(int index, std::span<const float> controls, std::uint64_t seed) = 0;
  virtual void step(std::span<const float> controls, std::uint64_t seed) = 0;
  virtual void render(const FrameInput& in, std::uint8_t* rgba, std::size_t stride) = 0;
  virtual std::size_t scratch_bytes() const = 0;  // its own buffers (and its own RolloutScratch; a shared one is not counted)
  virtual double macs_per_pixel() const = 0;
  virtual float time() const = 0;  // seconds since the effect began (the start point's run time plus frames stepped)
  virtual std::span<const float> coarse() const = 0;  // res * res * channels
  virtual std::span<const float> fine_heat() const = 0;
  virtual std::span<const float> fine_soot() const = 0;

  // Composition (src/compose, docs/COMPOSE.md): between steps other effects may read and write the state, for example
  // to push it with their flow or to hand over a whole state. Nothing here allocates.
  virtual int size() const = 0;
  virtual std::span<const float> flow() const = 0;  // projected velocity of the last step: res * res * 2, cells per frame
  virtual std::span<float> coarse_mut() = 0;
  virtual std::span<float> fine_heat_mut() = 0;
  virtual std::span<float> fine_soot_mut() = 0;
  // The state was written from outside (a hand-over): continue from it at `seconds` since the effect began, with the
  // pressure and flow of the old state dropped and the sub-grid swirl at full strength.
  virtual void adopt(float seconds) = 0;
  // Study H (H2): the detail step and the renderer skip the parts of a row where the fine fields are zero and nothing
  // can reach them, which leaves every value as it was to the bit. On by default; off computes every pixel (tests).
  virtual void skip_empty(bool on) = 0;
  // Continue from `other`, a runner of the same effect built for the same ISA, of any size (the effect manager's level
  // of detail, docs/MANAGER.md): its coarse state, pressure, flow, time, swirl ramp and seed exactly, and its fine
  // fields resampled to this runner's size (block averages down, bilinear up). The coarse state does not depend on the
  // size, so from here it evolves exactly as it would have in `other`. No allocation. std::invalid_argument for a
  // runner of another effect or ISA.
  virtual void transfer_from(const RolloutRunner& other) = 0;

  // The working memory of step() (RolloutScratch): the runner's own by default. use_scratch(s) makes it step with s
  // from then on (s fit() for this runner, else std::invalid_argument) and frees its own; s must outlive the runner's
  // steps with it, and runners that share one must not step at the same time. use_scratch(nullptr) returns to an own
  // one (allocates). Switching between scratches allocates nothing once the own one is gone. A step reads nothing that
  // an earlier step left in its scratch, so the frames are the same to the bit whichever scratch, and whichever order.
  virtual void use_scratch(RolloutScratch* s) = 0;
  virtual std::array<std::size_t, 2> scratch_need() const = 0;  // floats and 32-bit integers that step() uses
  // Study K: time the stages of step() and render() into *p from now on (nullptr: stop). The runner does not own it.
  virtual void set_profile(RolloutProfile* p) = 0;
};

// What a rollout runner's step() uses and leaves nothing in for the next step: the detail layer's rings of padded rows,
// row records, stencils and column sums, and the coarse step's activations (about 8 MB for a 384-pixel tile, 1 MB at
// 128). Runners stepped one after another by one thread can share one (RolloutRunner::use_scratch), so a scene needs
// one per thread that steps rather than one per runner (docs/COMPOSE.md §7.3). Any ISA's runners can use it.
struct RolloutScratch {
  AlignedFloats floats;
  std::vector<std::int32_t> ints;
  // Room for runner r too (a set-up call: allocates when it grows).
  void fit(const RolloutRunner& r) {
    const auto n = r.scratch_need();
    if (floats.size() < n[0]) floats.assign(n[0], 0.f);
    if (ints.size() < n[1]) ints.assign(n[1], 0);
  }
  bool fits(const RolloutRunner& r) const {
    const auto n = r.scratch_need();
    return floats.size() >= n[0] && ints.size() >= n[1];
  }
  std::size_t bytes() const { return 4 * (floats.size() + ints.size()); }
};

// Precision of a network (nvfx_instance_set_precision): float throughout, or 8-bit integers where they pay. Frame
// models: the grid family's hidden layers with a projected first layer (rt_int8.hpp; the conv family is float either
// way). Rollout effects: the stepper's second convolution (rt_rollout.hpp, rt_int8_conv.hpp; docs/REPORT.md §13).
enum class Precision { float32, int8 };
// Study K's candidates for a rollout effect's int8 path: the stepper's second convolution in 8 bits (its inputs follow a
// ReLU: 7-bit codes), or both (the first one's inputs are the state, the noise and the position, signed: 8-bit codes
// with a zero point, and 7-bit weights). kInt8Convs is the one Precision::int8 means.
enum class Int8Convs { second, both };
inline constexpr Int8Convs kInt8Convs = Int8Convs::second;
// The precision a rollout effect runs at unless asked otherwise (nvfx_instance_set_precision's DEFAULT; scenes, the
// library's stages and the effect manager too). Study K's test passed its rule (docs/REPORT.md §13), whose cost
// condition holds where the runner takes the AVX-512 build with VNNI (int8 0.90 of float's time at 128 px); without
// VNNI int8 costs what float does (0.99, a tie), so the condition fails there. So: int8 where the CPU has AVX-512 VNNI
// and no other ISA is forced (nvfx_set_isa), float elsewhere. Defined in nvfx.cpp.
Precision default_rollout_precision();

// One factory per ISA build (rt_base.cpp, rt_avx2.cpp, rt_avx512.cpp). Allocates every buffer the renderer will use.
// A lean rollout runner keeps in itself only what lasts from one step to the next (state, noise caches, folded
// weights); the buffers a step or a picture fills and uses up come from its RolloutScratch, which render() then needs
// too (use_scratch before rendering when the scratch is shared). Its frames are those of a runner that is not lean, to
// the bit; it is about 190 KB smaller (the effect manager's simulations, docs/MANAGER.md).
namespace isa_base {
std::unique_ptr<Renderer> make_renderer(const Effect& e, int size, Precision p = Precision::float32);
std::unique_ptr<RolloutRunner> make_rollout(const RolloutEffect& e, int size, bool lean = false, Precision p = Precision::float32, Int8Convs convs = kInt8Convs);
}  // namespace isa_base
namespace isa_avx2 {
std::unique_ptr<Renderer> make_renderer(const Effect& e, int size, Precision p = Precision::float32);
std::unique_ptr<RolloutRunner> make_rollout(const RolloutEffect& e, int size, bool lean = false, Precision p = Precision::float32, Int8Convs convs = kInt8Convs);
}  // namespace isa_avx2
namespace isa_avx512 {
std::unique_ptr<Renderer> make_renderer(const Effect& e, int size, Precision p = Precision::float32);
std::unique_ptr<RolloutRunner> make_rollout(const RolloutEffect& e, int size, bool lean = false, Precision p = Precision::float32, Int8Convs convs = kInt8Convs);
}  // namespace isa_avx512

}  // namespace nfx::rt
