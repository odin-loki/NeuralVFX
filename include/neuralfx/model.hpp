// The neural effect models shared by the trainer, the runtime and the tools (docs/PLAN.md §5.1), the .nvfx file
// format, and a plain reference forward pass that the trainer and the runtime are both tested against.
//
// Two families, both conditioned on c = [controls..., variation code z...]:
//
//   grid  K learned feature volumes [grid_t][C][G][G]. Per frame they are blended with weights w(c) = Wb c + bb
//         and sliced at time t (linear in t, wrapping for loops). Per pixel: bilinear sample (C features), then
//         an MLP: h1 = relu((1 + gamma(c)) * (W1 f + b1) + beta(c)), more relu layers, a linear RGBA head.
//         Resolution-free: renders at any sprite size.
//   conv  K learned latent volumes [grid_t][c0][l][l], blended and sliced the same way, then three stages of
//         (2x nearest upsample, 3x3 conv) to c1, c2 and RGBA at 8 l pixels; FiLM (1 + gamma, beta) from c on the
//         first two stages, relu after them.
//
// Output: premultiplied RGBA in [0, 1] (clamped when converted to 8 bits). Time: t in [0, 1) spans the training
// clip; frame f of an F-frame clip is t = f / F for loops, f / (F - 1) for one-shot effects.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

namespace nfx {

enum class Arch : std::uint32_t { grid = 1, conv = 2 };

struct Hyper {
  Arch arch = Arch::grid;
  int size = 128;     // native output side (conv: must be 8 * latent)
  int frames = 64;    // frames of the clip the model spans
  bool loop = true;
  int n_controls = 0; // learned controls (sim: intensity, wind, turbulence)
  int n_latent = 0;   // variation code dimensions
  int bases = 1;      // K
  int grid_t = 16;    // time slices
  // grid family
  int grid = 32, channels = 8, hidden = 32, layers = 2;
  // conv family
  int latent = 16, c0 = 32, c1 = 16, c2 = 8;

  int dims() const { return n_controls + n_latent; }
  int feature_channels() const { return arch == Arch::grid ? channels : c0; }
  int feature_side() const { return arch == Arch::grid ? grid : latent; }
  std::string describe() const;
};

// y = W x + b with W row-major [out][in].
struct Dense {
  int in = 0, out = 0;
  std::vector<float> w, b;
  Dense() = default;
  Dense(int i, int o) : in(i), out(o), w(static_cast<std::size_t>(i) * o, 0.f), b(static_cast<std::size_t>(o), 0.f) {}
  std::size_t params() const { return w.size() + b.size(); }
};

struct Model {
  Hyper h;
  std::string effect;
  float fps = 30.f;
  std::vector<float> features;  // [K][grid_t][C][side][side]
  Dense basis;                  // D -> K (blend weights)
  std::vector<Dense> layers;    // grid: C -> H, (H -> H) * (layers - 1), H -> 4. conv: 3 convs as Dense(9 ci, co)
  std::vector<Dense> films;     // D -> 2 width: grid one (width H); conv two (widths c1, c2)
  std::vector<float> z_mean, z_std;          // statistics of the training variation codes
  std::vector<std::vector<float>> z_train;   // the codes themselves (replay a training variation)
  int feature_bits = 16;  // storage precision of `features` in the file: 16 (fp16) or 8 (per-plane affine)
  std::vector<std::string> control_names;    // n_controls names (stored, 15 characters each at most)

  // The features in their storage format, as the runtime keeps them resident: fp16 bit patterns, or bytes with a
  // (lo, hi) range per [side][side] plane. Filled by load_model() and by pack_features().
  std::vector<std::uint16_t> raw_f16;
  std::vector<std::uint8_t> raw_u8;
  std::vector<float> raw_ranges;
  void pack_features();

  std::size_t feature_count() const;
  std::size_t param_count() const;     // everything a shipped effect stores, codes included
  std::size_t storage_bytes() const;   // on disk and in memory as shipped (features at feature_bits, rest fp16)
  double macs_per_pixel(int out_size) const;  // per-pixel cost at a given output size (per-frame work amortised)
};

// Allocate and initialise a model (deterministic for a seed). Throws std::invalid_argument on a bad shape.
Model init_model(const Hyper& h, std::uint64_t seed);

// Map a frame index to model time.
float frame_time(const Hyper& h, int frame, int frames);

// The conditioning vector for given controls and a variation code (missing entries are zero).
std::vector<float> condition(const Model& m, std::span<const float> controls, std::span<const float> z);

// Plain reference forward pass: RGBA floats [size][size][4] at time t for condition c. `size` must be the native
// size for the conv family; any size for the grid family. Slow and simple on purpose.
void reference_render(const Model& m, float t, std::span<const float> c, int size, std::span<float> rgba);

// Quantise a float RGBA frame to premultiplied RGBA8.
void to_rgba8(std::span<const float> rgba, std::span<std::uint8_t> out);

// .nvfx files. Features are stored at m.feature_bits; all other tensors as fp16. Loading gives floats.
std::expected<void, std::string> save_model(const std::filesystem::path& path, const Model& m);
std::expected<void, std::string> save_model(std::ostream& out, const Model& m);
std::expected<Model, std::string> load_model(const std::filesystem::path& path);
std::expected<Model, std::string> load_model(std::istream& in);  // from any stream (memory: std::ispanstream)

// Round a model's weights through its storage precision (what a saved and reloaded model computes with).
void quantise_like_storage(Model& m);

}  // namespace nfx
