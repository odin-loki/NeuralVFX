// Runtime internals shared by the C API (nvfx.cpp) and the per-ISA renderers (rt_impl.hpp).
#pragma once

#include <neuralfx/model.hpp>

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

// One factory per ISA build (rt_base.cpp, rt_avx2.cpp, rt_avx512.cpp). Allocates every buffer the renderer will use.
namespace isa_base { std::unique_ptr<Renderer> make_renderer(const Effect& e, int size); }
namespace isa_avx2 { std::unique_ptr<Renderer> make_renderer(const Effect& e, int size); }
namespace isa_avx512 { std::unique_ptr<Renderer> make_renderer(const Effect& e, int size); }

}  // namespace nfx::rt
