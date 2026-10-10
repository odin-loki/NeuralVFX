// A small context-mixing coder for planes of signed integers: the quantised corrections and residuals of the G3a run
// codec (docs/DCM.md §8). The technique is that of PAQ and lpaq, as in the model-file coder (cm.hpp), cut down to what
// residual planes need:
//   - each integer is binarised: zero or not, the sign, the magnitude class (unary over floor(log2 |v|)), then the
//     bits below the leading one;
//   - each binary decision gets a probability from seven adaptive context models (counters indexed by a hash of the
//     decision and a context built from integers already coded: neighbours in this plane, the same place in the
//     previous plane of the same kind, the other channels of this cell, the position), mixed by an online logistic
//     mixer whose weight set is chosen by the decision and the local activity, and refined by an APM;
//   - a 32-bit carry-less binary arithmetic coder writes the bits.
// Every operation is integer arithmetic in a fixed order, so the coded bits do not depend on the compiler, the
// instruction set or the floating-point environment. Contexts never use floating-point state, so a stream decodes on
// any machine even where a float reconstruction loop would round differently.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nfx::codec {

// Binary arithmetic coder, 32-bit range, carry-less (lpaq1). p1 = P(bit = 1) in 1/4096, clamped to [1, 4095].
class ArithEncoder {
 public:
  explicit ArithEncoder(std::vector<std::uint8_t>& out) : out_(out) {}
  void encode(int bit, int p1);
  void flush();  // writes the 4 bytes that settle the last interval

 private:
  std::vector<std::uint8_t>& out_;
  std::uint32_t x1_ = 0, x2_ = 0xffffffffu;
};

class ArithDecoder {
 public:
  explicit ArithDecoder(std::span<const std::uint8_t> in);
  int decode(int p1);
  // The decoder has asked for more bytes than the stream holds: the data is damaged or truncated.
  [[nodiscard]] bool overrun() const noexcept { return pos_ > in_.size(); }
  [[nodiscard]] std::size_t consumed() const noexcept { return pos_; }

 private:
  std::uint32_t next();
  std::span<const std::uint8_t> in_;
  std::size_t pos_ = 0;
  std::uint32_t x1_ = 0, x2_ = 0xffffffffu, x_ = 0;
};

// What a plane holds. Each kind keeps its own previous plane (a context) and its own hashed statistics.
enum class PlaneKind : std::uint8_t { coarse_start = 0, coarse, fine_start, fine, pixel, count };

// The model: adaptive state for a whole stream. The encoder and the decoder each make one and code the same planes in
// the same order.
class ResidualModel {
 public:
  static constexpr int kTableBits = 16;   // counters per context model: 2^16 (256 KB each)
  static constexpr int kModels = 9;
  static constexpr int kMaxExp = 20;      // |v| < 2^21
  static constexpr std::int32_t kMaxAbs = (1 << (kMaxExp + 1)) - 1;

  ResidualModel();
  // One plane of h x w cells with c channels each (channels fastest), values |v| <= kMaxAbs. side: optional, one small
  // value (0..15) per cell that the decoder knows before decoding the plane (a class of its own prediction there), used
  // as a context.
  void encode(ArithEncoder& ac, PlaneKind kind, int h, int w, int c, std::span<const std::int32_t> v, std::span<const std::uint8_t> side = {});
  // Fills v. Returns false if the decoder overran its input (damaged data); v then holds garbage.
  bool decode(ArithDecoder& ac, PlaneKind kind, int h, int w, int c, std::span<std::int32_t> v, std::span<const std::uint8_t> side = {});
  // Ideal code length (bits) the model assigned to the planes coded so far, by kind (reports only; not coded).
  [[nodiscard]] double bits(PlaneKind kind) const { return bits_[static_cast<std::size_t>(kind)]; }
  // Working memory of the model, bytes.
  [[nodiscard]] std::size_t memory_bytes() const;

 private:
  struct Ctx;
  template <class Bit>
  std::int32_t code_value(Bit& bit, const Ctx& ctx, std::int32_t v);
  template <class Bit>
  void code_plane(Bit& bit, PlaneKind kind, int h, int w, int c, std::int32_t* v, std::span<const std::uint8_t> side);
  int predict(const Ctx& ctx, int bin);
  void update(int y);

  std::vector<std::uint32_t> counters_;   // kModels tables of 2^kTableBits: probability (high 22 bits), count (low 10)
  std::vector<std::int32_t> weights_;     // mixer weight sets (16 fractional bits)
  std::vector<std::uint16_t> apm_;        // APM: 33 buckets per context, 16-bit probabilities
  std::array<std::vector<std::int32_t>, static_cast<std::size_t>(PlaneKind::count)> prev_;  // previous plane per kind
  std::array<int, static_cast<std::size_t>(PlaneKind::count) * 3> prev_dims_{};
  std::array<double, static_cast<std::size_t>(PlaneKind::count)> bits_{};
  // state of the decision in flight
  std::array<std::size_t, kModels> idx_{};
  std::array<int, kModels + 1> st_{};
  std::size_t wset_ = 0, apm_idx_ = 0;
  int apm_w_ = 0, p_mix_ = 2048, p_ = 2048;
  PlaneKind kind_ = PlaneKind::coarse;
};

}  // namespace nfx::codec
