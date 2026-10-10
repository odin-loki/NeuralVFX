// Runtime internals shared by the C API (nvfx.cpp) and the per-ISA renderers (rt_impl.hpp).
#pragma once

#include <neuralfx/model.hpp>
#include <neuralfx/rollout.hpp>

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
  virtual std::size_t scratch_bytes() const = 0;
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
};

// One factory per ISA build (rt_base.cpp, rt_avx2.cpp, rt_avx512.cpp). Allocates every buffer the renderer will use.
namespace isa_base {
std::unique_ptr<Renderer> make_renderer(const Effect& e, int size);
std::unique_ptr<RolloutRunner> make_rollout(const RolloutEffect& e, int size);
}  // namespace isa_base
namespace isa_avx2 {
std::unique_ptr<Renderer> make_renderer(const Effect& e, int size);
std::unique_ptr<RolloutRunner> make_rollout(const RolloutEffect& e, int size);
}  // namespace isa_avx2
namespace isa_avx512 {
std::unique_ptr<Renderer> make_renderer(const Effect& e, int size);
std::unique_ptr<RolloutRunner> make_rollout(const RolloutEffect& e, int size);
}  // namespace isa_avx512

}  // namespace nfx::rt
