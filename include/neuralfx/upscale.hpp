// Study P (docs/STUDY_P.md): a rollout effect drawn at half size and brought to full size by a small learned 2x upscaler
// with seeded micro-detail.
//
// The upscaler sees only the half-size frame (premultiplied RGBA8, n x n) and draws the 2n x 2n frame:
//   out = up + r + gain * a * eta
// where
//   - up is the bilinear 2x of the frame (pixel centres, clamped at the edges: what a GPU's bilinear sampler gives);
//   - r is a learned residual: two small convolutions on the half-size frame (3 x 3, 4 -> c1, ReLU; k2 x k2, c1 -> c2,
//     ReLU) and a 1 x 1 output layer whose first 16 values per half-size pixel are the residuals of its 2 x 2 output
//     pixels (sub-pixel s = 2 dy + dx, channel c at 4 s + c);
//   - a >= 0 is the amplitude of the micro-detail per half-size pixel and channel (the output layer's last 4 values
//     through a ReLU), learned from the local energy of what the residual leaves out;
//   - eta is seeded noise at the full size with unit variance and no energy at the lowest frequencies (white noise minus
//     its 3 x 3 mean), changing smoothly in time (slices `rate` per second, blended by a smoothstep), a function of the
//     instance's seed, the pixel and the time only, so every frame is deterministic.
// Pixels where up is zero in every channel stay exactly transparent (a rollout effect's empty pixels are exactly zero,
// and so are the upscaler's). Everything here is plain float C++: the reference for the trainer (tools/nvfx_study_p.cpp)
// and for the runtime's vectorised copy (src/runtime/rt_upscale_impl.hpp, held to it by tests/test_upscale.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

namespace nfx::upscale {

inline constexpr int kIn = 4;          // input channels (RGBA / 255)
inline constexpr int kResid = 16;      // residual outputs per half-size pixel (2 x 2 sub-pixels x RGBA)
inline constexpr int kAmp = 4;         // amplitude outputs per half-size pixel (RGBA)
inline constexpr int kOut = kResid + kAmp;
inline constexpr float kNoiseNorm = 1.83711731f;  // 1 / sqrt(72 / 81 * 1 / 3): white noise in [-1, 1] minus its 3 x 3
                                                   // mean has variance 72 / 243; this scales it to 1

struct Net {
  int c1 = 8, c2 = 8, k2 = 1;  // widths of the two hidden layers, and the second one's kernel (1 or 3)
  float gain = 1.f;            // the micro-detail's gain (0: no micro-detail)
  float rate = 8.f;            // noise slices per second
  std::string tag;             // what it was trained for: "shared" or an effect's name
  std::vector<float> w;        // layout()

  struct Layout {
    std::size_t w1, b1, w2, b2, wo, bo, size;
  };
  // w1 [tap 9][in 4][c1], b1 [c1], w2 [tap k2 * k2][c1][c2], b2 [c2], wo [c2][20], bo [20]; tap = (dy + r) * k + (dx + r).
  Layout layout() const;
  double macs_per_output_pixel() const;  // the network's multiply-adds, per full-size pixel
};

// Fresh weights (He initialisation; the output layer small, so the upscaler starts as bilinear).
Net init_net(int c1, int c2, int k2, std::uint64_t seed);

// .nvfxup files: magic NVFXUPS1, version 1, little-endian (int32 c1, c2, k2; float gain, rate; a 32-byte tag; uint32
// weight count; the weights as float32).
inline constexpr char kMagic[8] = {'N', 'V', 'F', 'X', 'U', 'P', 'S', '1'};
std::expected<void, std::string> save_net(const std::filesystem::path& path, const Net& n);
std::expected<void, std::string> save_net(std::ostream& out, const Net& n);
std::expected<Net, std::string> load_net(const std::filesystem::path& path);
std::expected<Net, std::string> load_net(std::istream& in);
std::expected<Net, std::string> parse_net(std::span<const char> bytes);

// --- reference ---------------------------------------------------------------------------------------------------------

// Bilinear 2x of an n x n RGBA8 frame (rows top to bottom) as floats in [0, 1]: [2n][2n][4]. Output pixel x samples the
// frame at (x + 0.5) / 2 - 0.5: 0.75 of the nearer pixel and 0.25 of the next one on each axis, clamped at the edges.
void bilinear2x(std::span<const std::uint8_t> lo, int n, std::span<float> up);

// The network's outputs per half-size pixel: [n][n][kOut] (residuals, then amplitudes after their ReLU).
void network(const Net& net, std::span<const std::uint8_t> lo, int n, std::span<float> out);

// The seed of the noise for an instance's seed (kept apart from every other noise of the effect), and slice k's key.
std::uint64_t noise_seed(std::uint64_t seed);
std::uint32_t noise_key(std::uint64_t seed, std::int64_t k);
// White noise in [-1, 1] of a slice's key at any integer pixel (one hash per row, one per pixel).
float white(std::uint32_t key, std::int32_t x, std::int32_t y);
// The micro-detail noise of slice k at full-size pixel (x, y): unit variance (white noise minus its 3 x 3 mean, times
// kNoiseNorm; the 3 x 3 sum as three sums along x added along y), from the instance's seed.
float noise_slice(std::uint64_t seed, std::int32_t x, std::int32_t y, std::int64_t k);
// The noise of a frame at `seconds` (slices k and k + 1 blended, with variance kept at 1): [size][size].
void noise_frame(std::uint64_t seed, double seconds, float rate, int size, std::span<float> eta);
// The blend of a time: slice k, weight s of slice k + 1, and the factor that keeps the variance at 1.
struct NoiseBlend {
  std::int64_t k = 0;
  float s = 0, norm = 1;
};
NoiseBlend noise_blend(double seconds, float rate);

// The upscaled frame as floats before clamping, [2n][2n][4]: up + r (+ gain a eta when eta is not empty), zero where up
// is zero in every channel. With net == nullptr: bilinear alone.
void upscale_float(const Net* net, std::span<const std::uint8_t> lo, int n, std::span<const float> eta, std::span<float> out);

// The whole reference: lo (n x n) to hi (2n x 2n, rows `stride` bytes apart), learned (net) or bilinear (net == nullptr),
// with the noise of `seconds` from `seed`, and the colour matrix (row-major 3 x 3 on RGB) when colour is not null.
// Bytes as the runtime writes them: clamp to [0, 1], times 255, plus 0.5, truncated.
void upscale(const Net* net, std::span<const std::uint8_t> lo, int n, std::uint64_t seed, double seconds, std::uint8_t* hi,
             std::size_t stride, const float* colour = nullptr);

}  // namespace nfx::upscale
