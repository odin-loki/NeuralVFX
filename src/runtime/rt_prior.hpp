// The prior against drift in the runtime (docs/DCM.md G2.13): study G's coarse-state denoiser (G2b), inference only.
//
// A rollout effect played as one continuous rollout drifts after about 20 s; every N frames the prior moves its coarse
// state towards the denoiser's one-step (Tweedie) estimate of a clean state, so the run stays alive for a minute
// without shards. This is the trainer's dcm::ddpm::prior_step (src/dcm/ddpm.cpp) for the runtime: the .ddpm file is read
// here (no link to the training libraries), the forward pass is written again for inference only, and every buffer is
// allocated when the Prior is made, so a pass allocates nothing.
//
// Parity (tests/test_runtime_prior.cpp): the forward pass keeps the reference's operation order per value, so on the
// reference's own ISA (AVX2 + FMA, with contraction) and on AVX-512 a prior step gives the reference's floats bit for
// bit; the baseline (SSE2, no FMA) rounds each multiply-add twice and stays within a stated tolerance.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace nfx::rt {

// A denoiser read from a .ddpm file ("NVFXDDPM", version 1, little-endian: nine int32 sizes, scale, lo and hi per
// channel, a uint64 weight count, the weights as float32; written by dcm::ddpm::save). The weight offsets are those of
// dcm::ddpm::layout (the test holds them to it).
struct PriorNet {
  static constexpr int kBlocks = 6;  // e0, e1, m0, m1, d1, d0
  int res = 0, channels = 0, c0 = 0, c1 = 0, c2 = 0, cond = 0, freqs = 0, film_hidden = 0, timesteps = 0;
  std::vector<float> scale, lo, hi;  // per channel: physical units of one network unit; range of network values
  std::vector<float> w;              // the weights as stored
  struct Block {
    int width = 0, side = 0;
    std::size_t wa = 0, ba = 0, wb = 0, bb = 0, film = 0;
  };
  std::array<Block, kBlocks> blocks{};
  std::size_t stem_w = 0, stem_b = 0, down1_w = 0, down1_b = 0, down2_w = 0, down2_b = 0, up2_w = 0, up2_b = 0, up1_w = 0,
              up1_b = 0, out_w = 0, out_b = 0, mlp1_w = 0, mlp1_b = 0, mlp2_w = 0, mlp2_b = 0, film_size = 0, size = 0;
  std::size_t file_bytes = 0;

  [[nodiscard]] int inputs() const noexcept { return channels + 2; }  // the state, then x and y in [-1, 1]
  [[nodiscard]] int embed() const noexcept { return 2 * freqs + cond; }
  [[nodiscard]] double macs() const;                                   // multiply-adds of one pass
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return 4 * (w.size() + scale.size() + lo.size() + hi.size()); }
};

// Reads a .ddpm file's bytes. Errors: not a denoiser, another version, sizes out of range, a truncated file.
[[nodiscard]] std::expected<PriorNet, std::string> parse_prior(std::span<const char> bytes);

// alpha_bar(t) of the denoiser's cosine schedule with its beta clip (dcm::ddpm::cosine_alpha_bar, the same arithmetic),
// t in [0, T].
[[nodiscard]] double prior_alpha_bar(int timesteps, int t);

// One instance's prior: the network's work buffers, allocated here. Not shared between threads.
class Prior {
 public:
  virtual ~Prior() = default;
  // eps_hat(x_t, t) for one state in network units (res^2 x channels), t in [1, T].
  virtual void predict(std::span<const float> xt, int t, std::span<const float> cond, std::span<float> eps) = 0;
  // The prior step (dcm::ddpm::prior_step): x <- (1 - beta) x + beta (x - r eps_hat(sa x, t)), sa = sqrt(alpha_bar(t)),
  // r = sqrt((1 - alpha_bar(t)) / alpha_bar(t)), kept inside the denoiser's [lo, hi]. x in network units.
  virtual void step(std::span<float> x, int t, float beta, std::span<const float> cond) = 0;
  // The prior on a rollout's coarse state as study G applied it (tools/experiment_g.cpp, long_run): the first
  // `channels` values of each cell (physical units, `stride` values per cell) divided by the denoiser's scale, the
  // step, multiplied back, then each physical channel clamped to the stepper's range [phys_lo, phys_hi].
  virtual void apply(std::span<float> coarse, int stride, std::span<const float> phys_lo, std::span<const float> phys_hi, int t, float beta,
                     std::span<const float> cond) = 0;
  [[nodiscard]] virtual std::size_t scratch_bytes() const = 0;
};

// One factory per ISA build (rt_prior_base.cpp, rt_prior_avx2.cpp, rt_prior_avx512.cpp). The net must outlive the
// prior. Throws std::bad_alloc.
namespace isa_base {
std::unique_ptr<Prior> make_prior(const PriorNet& net);
}  // namespace isa_base
namespace isa_avx2 {
std::unique_ptr<Prior> make_prior(const PriorNet& net);
}  // namespace isa_avx2
namespace isa_avx512 {
std::unique_ptr<Prior> make_prior(const PriorNet& net);
}  // namespace isa_avx512

}  // namespace nfx::rt
