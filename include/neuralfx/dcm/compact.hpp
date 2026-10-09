// Released (frozen) mixers for inference only, ported from the owner's CameraDetector, cabinlab/src/diffusion
// (Phase 12, compact.hpp); CompactValueNet and the flat binary form are new here.
//
// A frozen MixerNet or ValueNet (mixer.hpp) is released as its canonical serialisation, whose SHA-256 is its version.
// The compact classes rebuild the network from that text with weights of type Real (float halves the bytes) and
// predict without any learning state (no remembered inputs, no use counts), so one instance can serve many threads.
// predict() follows the trained network's predict() operation for operation.
//
// The flat binary form (to_bytes / from_bytes) is what a .nvfx file can embed: little-endian, the weights stored as
// Real, and the version of the text it was built from carried along (the text cannot be rebuilt from float weights).
//   "NVFXDCM1"  8 bytes
//   u8 kind (1: PAQ mixer, 2: value net), u8 sizeof(Real), u16 0
//   64 bytes: the version (lowercase hex)
//   PAQ mixer:  u32 first-layer mixers m; per mixer (m first-layer ones, then the final one): u32 n, u32 k, n·k Real;
//               f64 APM weight; u32 APM contexts; 33 Real per APM context
//   value net:  f64 limit; u32 m; mixers as above; f64 AVM weight; u32 AVM contexts; f64 lo; f64 hi; 33 Real per AVM
//               context; scale net: u32 n, u32 k, f64 log_b_min, f64 log_b_max, n·k Real
#pragma once

#include <neuralfx/dcm/mixer.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nfx::dcm {

template <class Real>
class CompactMixer {
 public:
  // Parses MixerNet::serialise() output. Throws std::runtime_error when the text is not a v1 mixer serialisation.
  explicit CompactMixer(std::string_view serialised);
  // Rebuilds from to_bytes() output (of the same Real). Throws std::runtime_error on a malformed or truncated form.
  [[nodiscard]] static CompactMixer from_bytes(std::span<const std::uint8_t> bytes);
  [[nodiscard]] std::vector<std::uint8_t> to_bytes() const;
  // Probability of the positive class; contexts as MixerNet::predict (one per first-layer mixer, the final mixer's,
  // the APM's). Throws std::invalid_argument on wrong sizes and std::out_of_range on a context outside a mixer's range.
  [[nodiscard]] double predict(std::span<const double> x, std::span<const int> contexts) const;
  [[nodiscard]] int inputs() const noexcept { return n_; }
  [[nodiscard]] int first_layer() const noexcept { return static_cast<int>(layer1_.size()); }
  // Weights and APM table values held.
  [[nodiscard]] std::size_t values() const noexcept;
  // values() * sizeof(Real).
  [[nodiscard]] std::size_t bytes() const noexcept { return values() * sizeof(Real); }
  // SHA-256 of the serialisation it was built from (MixerNet::version of the released mixer).
  [[nodiscard]] const std::string& version() const noexcept { return version_; }

  struct Layer {
    int n = 0, k = 0;
    std::vector<Real> w;  // k x n
  };

 private:
  CompactMixer() = default;
  void check() const;

  int n_ = 0;
  std::vector<Layer> layer1_;
  Layer final_;
  double apm_weight_ = 0.0;
  int apm_k_ = 0;
  std::vector<Real> apm_;  // 33 per context
  std::string version_;
};

template <class Real>
class CompactValueNet {
 public:
  // Parses ValueNet::serialise() output. Throws std::runtime_error when the text is not a v1 value-net serialisation.
  explicit CompactValueNet(std::string_view serialised);
  [[nodiscard]] static CompactValueNet from_bytes(std::span<const std::uint8_t> bytes);
  [[nodiscard]] std::vector<std::uint8_t> to_bytes() const;
  // contexts as ValueNet::predict (one per first-layer mixer, then the final mixer's, the AVM's, the scale net's); z:
  // the scale features. Throws std::invalid_argument on wrong sizes and std::out_of_range on a context outside a
  // mixer's range.
  [[nodiscard]] ValuePrediction predict(std::span<const double> x, std::span<const int> contexts,
                                        std::span<const double> z) const;
  [[nodiscard]] int inputs() const noexcept { return n_; }
  [[nodiscard]] int scale_features() const noexcept { return scale_.n - 1; }
  [[nodiscard]] int first_layer() const noexcept { return static_cast<int>(layer1_.size()); }
  // Weights, AVM offsets and scale weights held.
  [[nodiscard]] std::size_t values() const noexcept;
  [[nodiscard]] std::size_t bytes() const noexcept { return values() * sizeof(Real); }
  // SHA-256 of the serialisation it was built from (ValueNet::version of the released net).
  [[nodiscard]] const std::string& version() const noexcept { return version_; }

  using Layer = typename CompactMixer<Real>::Layer;

 private:
  CompactValueNet() = default;
  void check() const;

  int n_ = 0;
  double limit_ = 0.0;
  std::vector<Layer> layer1_;
  Layer final_;
  double avm_weight_ = 0.0, avm_lo_ = 0.0, avm_hi_ = 1.0;
  int avm_k_ = 0;
  std::vector<Real> avm_;  // 33 offsets per context
  Layer scale_;
  double log_b_min_ = 0.0, log_b_max_ = 0.0;
  std::string version_;
};

extern template class CompactMixer<float>;
extern template class CompactMixer<double>;
extern template class CompactValueNet<float>;
extern template class CompactValueNet<double>;

}  // namespace nfx::dcm
