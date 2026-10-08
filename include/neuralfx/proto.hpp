// Phase 0 prototypes: candidate tiny networks that render one RGBA sprite frame, used to measure the cost of each
// architecture family on one CPU core before any training exists (docs/PLAN.md, §4 and §8).
// The weights are random: cost does not depend on their values. Quality is Phase 3's question.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace nfx::proto {

enum class Isa { base, avx2, avx512 };

const char* isa_name(Isa isa);
bool isa_supported(Isa isa);  // the running CPU can execute this ISA's code
Isa best_isa();
bool parse_isa(std::string_view text, Isa& out);  // "base", "avx2", "avx512"

// One candidate. Fields that a kind does not use are ignored.
struct Spec {
  std::string kind = "mlp_sep";  // mlp_naive, mlp_sep, grid_mlp, hash_mlp, conv_dec
  int size = 128;                // sprite side in pixels, a multiple of 16
  int hidden = 32;               // MLP width
  int layers = 2;                // hidden layers (each H wide)
  int freqs = 6;                 // Fourier frequencies per coordinate (mlp_*)
  int grid = 32;                 // grid side (grid_mlp)
  int grid_t = 16;               // grid time slices (grid_mlp, conv_dec latent slices)
  int channels = 8;              // grid features (grid_mlp)
  int levels = 8;                // hash levels (hash_mlp)
  int log2_table = 14;           // hash table entries per level, log2 (hash_mlp)
  int features = 2;              // features per hash level (hash_mlp)
  int latent = 16;               // latent side (conv_dec); size / latent must be 8
  int c0 = 32, c1 = 16, c2 = 8;  // conv_dec channels: latent, after the first and second up-convolution
  std::uint64_t seed = 1;        // weight initialisation

  std::string describe() const;
};

// Per-frame inputs: animation time in [0, 1) and the artist controls.
struct Controls {
  float t = 0.f;
  float intensity = 1.f, wind = 0.f, speed = 1.f, hue = 0.f;
  float seed = 0.f;
};
inline constexpr int kControls = 5;  // intensity, wind, speed, hue, seed

class Model {
 public:
  virtual ~Model() = default;
  // Writes size * size RGBA8 pixels, row-major. No allocation after construction.
  virtual void render(const Controls& c, std::uint8_t* rgba) = 0;
  virtual std::size_t param_count() const = 0;   // learnable floats a shipped effect would store
  virtual double macs_per_pixel() const = 0;     // nominal multiply-adds per output pixel
  virtual int size() const = 0;
};

// Throws std::invalid_argument on a bad spec or an ISA the CPU cannot run.
std::unique_ptr<Model> make_model(const Spec& spec, Isa isa);

}  // namespace nfx::proto
