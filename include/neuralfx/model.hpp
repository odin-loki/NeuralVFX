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
//   multi (study F4) several feature volumes ("levels"), each with its own side, time slices and channels, each blended,
//         sliced and sampled as the grid family's; their features concatenated, optionally with Fourier features of
//         the pixel's position and of time, then the grid family's MLP with FiLM. Stored only as file version 4.
//
// Output: premultiplied RGBA in [0, 1] (clamped when converted to 8 bits). Time: t in [0, 1) spans the training
// clip; frame f of an F-frame clip is t = f / F for loops, f / (F - 1) for one-shot effects.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <iosfwd>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nfx {

enum class Arch : std::uint32_t { grid = 1, conv = 2, multi = 4 };  // (3 is the C API's rollout effects)

// One feature volume of the multi family: grid x grid points over the sprite, grid_t time slices (1: one still plane,
// the same at every time), `channels` features per point.
struct Level {
  int grid = 32, grid_t = 16, channels = 8;
  bool operator==(const Level&) const = default;
};

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
  // multi family (hidden and layers as the grid family's): the levels in storage order, and the Fourier features of the
  // pixel's position (pe_xy frequencies: sin and cos of 2^k pi u for u = x and y in [0, 1] across the sprite, k < pe_xy)
  // and of time (pe_t: sin and cos of 2^k 2 pi t, k < pe_t; whole periods, so loops stay seamless).
  std::vector<Level> levels{};
  int pe_xy = 0, pe_t = 0;

  int dims() const { return n_controls + n_latent; }
  // Feature planes have feature_side() x feature_side() values in the grid and conv families; the multi family's
  // differ by level (volumes(), below).
  int feature_channels() const;
  int feature_side() const { return arch == Arch::grid ? grid : arch == Arch::conv ? latent : levels.empty() ? 0 : levels[0].grid; }
  int mlp_in() const { return feature_channels() + 4 * pe_xy + 2 * pe_t; }  // inputs of the first layer (grid, multi)
  std::string describe() const;
};

// The feature volumes of a model in storage order: the grid and conv families' one volume, or the multi family's
// levels. Model::features holds each volume's [basis][slice][channel][side][side] values one volume after another;
// planes ([basis][slice][channel] within a volume), per-slice flags and masks ([slice][side][side]) follow the same
// order. `value0`, `plane0`, `slice0` and `mask0` are a volume's first value, plane, slice and mask point.
struct Volume {
  int side = 0, slices = 0, channels = 0;
  std::size_t value0 = 0, plane0 = 0, slice0 = 0, mask0 = 0;
  std::size_t plane_values() const { return static_cast<std::size_t>(side) * static_cast<std::size_t>(side); }
};
std::vector<Volume> volumes(const Hyper& h);

// Time slices bracketing t in a volume of `slices` slices, and the weight of the second (the grid family's rule; one
// slice: still).
inline void slice_lerp(int slices, bool loop, float t, int& i0, int& i1, float& w) {
  if (slices <= 1) {
    i0 = i1 = 0;
    w = 0.f;
  } else if (loop) {
    const float u = (t - std::floor(t)) * static_cast<float>(slices);
    i0 = std::min(static_cast<int>(u), slices - 1);
    i1 = (i0 + 1) % slices;
    w = u - static_cast<float>(i0);
  } else {
    const float u = std::clamp(t, 0.f, 1.f) * static_cast<float>(slices - 1);
    i0 = std::min(static_cast<int>(u), slices - 2);
    i1 = i0 + 1;
    w = u - static_cast<float>(i0);
  }
}

// The Fourier features of the multi family: position (4 pe_xy values: per frequency sin x, cos x, sin y, cos y) and
// time (2 pe_t values: per frequency sin t, cos t). `u`, `v` in [0, 1] across the sprite ((pixel + 0.5) / size).
inline void position_features(int pe_xy, float u, float v, float* out) {
  for (int k = 0; k < pe_xy; ++k) {
    const float f = std::ldexp(std::numbers::pi_v<float>, k);
    out[4 * k] = std::sin(f * u);
    out[4 * k + 1] = std::cos(f * u);
    out[4 * k + 2] = std::sin(f * v);
    out[4 * k + 3] = std::cos(f * v);
  }
}
// Loops use whole periods of t in [0, 1) (seamless); one-shot effects half periods of t in [0, 1] (so that the first and
// the last frame differ).
inline void time_features(int pe_t, bool loop, float t, float* out) {
  const float base = loop ? 2.f * std::numbers::pi_v<float> : std::numbers::pi_v<float>;
  const float u = loop ? t - std::floor(t) : std::clamp(t, 0.f, 1.f);
  for (int k = 0; k < pe_t; ++k) {
    const float a = std::ldexp(base, k) * u;
    out[2 * k] = std::sin(a);
    out[2 * k + 1] = std::cos(a);
  }
}

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
  int feature_bits = 16;  // storage precision of `features` in the file: 16 (fp16), or 2 to 8 (per-plane affine, below 8
                          // bit-packed; see packed_plane_bytes)
  bool feature_trim = false;  // affine planes: the range that quantises the plane best (feature_plane_range), clipping
                              // its tails, instead of its min and max. Chosen when saving; the file stores the range.
  // Vector-quantised features (study F2): with vq_bits in 2..8, the channels of every grid point are split into groups
  // of vq_dim, and each group stores one index of vq_bits bits into the group's codebook of 2^vq_bits vectors (fp16).
  // The features are the codewords nearest to `features` (vq_assign); feature_bits is then not used. File version 2.
  int vq_bits = 0, vq_dim = 0;
  std::vector<float> vq_codebook;  // [group][2^vq_bits][vq_dim]
  std::vector<float> raw_codebook; // resident: the codebook as floats (raw_u8 then holds the packed indices)
  // Per-plane storage (study F3, file version 3; not combined with vector quantisation). plane_bits, when not empty,
  // holds one width per feature plane [basis][slice][channel], 0 to 8 bits (affine planes as below; 0 bits: the plane's
  // stored values are one value, its lo, and it has no codes); feature_bits is then not used for the features.
  std::vector<std::uint8_t> plane_bits;
  // Optional with plane_bits, grid family only: the grid points each time slice stores, [grid_t][side][side], 1 =
  // stored (the same in every basis and channel). A plane stores codes for those points only (raster order) and one
  // fill value (fp16) that every other point of the plane takes: the mean of the plane's values there.
  std::vector<std::uint8_t> feature_mask;
  // Resident, per-plane storage: each plane's first byte in raw_u8, the mask bit-packed per slice
  // (packed_plane_bytes(side * side, 1) bytes each), the fill value of each plane.
  std::vector<std::uint32_t> raw_offsets;
  std::vector<std::uint8_t> raw_mask;
  std::vector<float> raw_fill;
  // The multi family (file version 4): plane_bits holds one width per plane (1 to 8, the same for every plane of a
  // level), feature_mask, when not empty, one mask per level ([slice][side][side] each, in volume order), stored as runs;
  // mlp_bits (16 or 8) the storage of the first, hidden and output layers' weights (8: per output unit an fp16 scale and
  // signed 8-bit weights, the unit's largest |w| at 127).
  int mlp_bits = 16;
  std::vector<std::uint32_t> raw_mask_at;  // resident, multi: each (level, slice)'s first byte of runs in raw_mask
  bool per_plane() const { return !plane_bits.empty(); }
  bool masked() const { return !feature_mask.empty(); }
  // Stored points of time slice t (per_plane storage): side * side, or the mask's count.
  std::size_t plane_points(int t) const;
  std::vector<std::string> control_names;    // n_controls names (stored, 15 characters each at most)

  // The features in their storage format, as the runtime keeps them resident: fp16 bit patterns, or N-bit codes with
  // a (lo, hi) range per [side][side] plane (one byte per code at 8 bits, bit-packed below; packed_plane_bytes()
  // bytes per plane). Filled by load_model() and by pack_features().
  std::vector<std::uint16_t> raw_f16;
  std::vector<std::uint8_t> raw_u8;
  std::vector<float> raw_ranges;
  void pack_features();
  int vq_groups() const { return vq_bits > 0 && vq_dim > 0 ? h.feature_channels() / vq_dim : 0; }

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

// Affine feature storage at N bits (2 to 8): each [side][side] plane keeps an fp16 (lo, hi) and codes
// q in [0, 2^N - 1], value = lo + q / (2^N - 1) * (hi - lo). Below 8 bits the codes of a plane are bit-packed: code j
// occupies bits [j N, j N + N) of the plane's bytes, least significant bit first, and the plane is padded with zero
// bits to a whole byte.
inline bool valid_feature_bits(int bits) { return bits == 16 || (bits >= 2 && bits <= 8); }

// The (lo, hi) range of an affine plane at `bits`, both fp16 values: the plane's min and max, or with `trim` the
// candidate range with the least squared quantisation error among the min and max and the ranges that clip the
// plane's lowest and highest 0.2%, 0.5%, 1%, 2%, 4% and 8% of values (values outside are stored as the end codes).
// At 0 bits (mixed precision only) both are the plane's mean: the plane is stored as that one value.
std::pair<float, float> feature_plane_range(std::span<const float> plane, int bits, bool trim);
// A plane's values as stored at `bits` (0 to 8; above 8: left as they are) with the range of feature_plane_range, in
// place. With `active` (one flag per value; empty: all), only the active values are coded (the range is theirs), and
// every other value becomes the plane's fill: the mean of those values, rounded to fp16.
void quantise_plane(std::span<float> plane, int bits, bool trim, std::span<const std::uint8_t> active = {});
inline std::size_t packed_plane_bytes(std::size_t values, int bits) {
  return bits >= 8 ? values : (values * static_cast<std::size_t>(bits) + 7) / 8;
}
inline unsigned packed_code(const std::uint8_t* plane, std::size_t j, int bits) {
  const std::size_t bit = j * static_cast<std::size_t>(bits);
  const unsigned shift = static_cast<unsigned>(bit & 7);
  unsigned v = static_cast<unsigned>(plane[bit >> 3]) >> shift;
  if (shift + static_cast<unsigned>(bits) > 8) v |= static_cast<unsigned>(plane[(bit >> 3) + 1]) << (8 - shift);
  return v & ((1u << bits) - 1u);
}
inline void put_packed_code(std::uint8_t* plane, std::size_t j, int bits, unsigned code) {  // plane zeroed first
  const std::size_t bit = j * static_cast<std::size_t>(bits);
  const unsigned shift = static_cast<unsigned>(bit & 7);
  plane[bit >> 3] = static_cast<std::uint8_t>(plane[bit >> 3] | ((code << shift) & 0xffu));
  if (shift + static_cast<unsigned>(bits) > 8) plane[(bit >> 3) + 1] = static_cast<std::uint8_t>(plane[(bit >> 3) + 1] | (code >> (8 - shift)));
}

// A mask (one byte per point, 0 or 1) as runs, as the multi family stores it: the lengths of alternating runs of 0s and
// 1s in raster order, starting with 0s (possibly an empty run), each an unsigned LEB128 varint (7 bits per byte, low
// first, the high bit set on every byte but the last); the runs end where they cover the mask.
std::vector<std::uint8_t> mask_runs(std::span<const std::uint8_t> mask);
// The mask back from its runs: the bytes they take, or 0 when they are malformed (too long, past the input, an empty
// run anywhere but first).
std::size_t read_mask_runs(std::span<const std::uint8_t> runs, std::span<std::uint8_t> mask);

// Vector quantisation: for every grid point of every slice and every channel group, the index of the nearest codeword
// (squared distance; the lowest index on ties). Indices are laid out as planes [basis][slice][group][side][side].
std::vector<std::uint8_t> vq_assign(const Model& m);

// Plain reference forward pass: RGBA floats [size][size][4] at time t for condition c. `size` must be the native
// size for the conv family; any size for the grid family. Slow and simple on purpose.
void reference_render(const Model& m, float t, std::span<const float> c, int size, std::span<float> rgba);

// Quantise a float RGBA frame to premultiplied RGBA8.
void to_rgba8(std::span<const float> rgba, std::span<std::uint8_t> out);

// .nvfx files. Features are stored at m.feature_bits (or per plane: m.plane_bits, m.feature_mask); all other tensors
// as fp16. Loading gives floats.
std::expected<void, std::string> save_model(const std::filesystem::path& path, const Model& m);
std::expected<void, std::string> save_model(std::ostream& out, const Model& m);
std::expected<Model, std::string> load_model(const std::filesystem::path& path);
std::expected<Model, std::string> load_model(std::istream& in);  // from any stream (memory: std::ispanstream)

// Round a model's weights through its storage precision (what a saved and reloaded model computes with).
void quantise_like_storage(Model& m);

// A layer's weights as the multi family stores them at Model::mlp_bits = 8, in place: per output unit an fp16 scale
// (its largest |w| over 127) times a signed 8-bit code.
void quantise_weights8(Dense& d);

}  // namespace nfx
