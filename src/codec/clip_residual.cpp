// G3b: a frame model's output plus a coded residual (include/neuralfx/codec/clip_residual.hpp).
#include <neuralfx/codec/clip_residual.hpp>
#include <neuralfx/codec/rcoder.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace nfx::codec {

namespace {

constexpr std::uint8_t kVersion = 1;
constexpr std::uint32_t kFnvBasis = 2166136261u, kFnvPrime = 16777619u;
void fnv(std::uint32_t& h, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    h ^= (v >> (8 * i)) & 0xffu;
    h *= kFnvPrime;
  }
}
void put_le(std::vector<std::uint8_t>& o, std::uint64_t v, int bytes) {
  for (int i = 0; i < bytes; ++i) o.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}
std::uint64_t get_le(std::span<const std::uint8_t> b, std::size_t pos, int bytes) {
  std::uint64_t v = 0;
  for (int i = 0; i < bytes; ++i) v |= static_cast<std::uint64_t>(b[pos + static_cast<std::size_t>(i)]) << (8 * i);
  return v;
}
constexpr std::size_t kHeader = 12;  // 'G' '3' 'B' version, frames (2), size (2), step * 16 (2), round * 256 (1), spare (1)

std::uint8_t level(int pred, int r, int step) { return static_cast<std::uint8_t>(std::clamp(pred + r * step / 16, 0, 255)); }

}  // namespace

ClipResidual encode_clip_residual(std::span<const std::uint8_t> pred, std::span<const std::uint8_t> truth, int size, int frames, float step_in, float round) {
  const std::size_t fb = static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4;
  if (size <= 0 || frames <= 0 || frames > 65535 || size > 4096 || pred.size() != fb * static_cast<std::size_t>(frames) || truth.size() != pred.size()) {
    throw std::invalid_argument("clip residual: bad sizes");
  }
  // the step in sixteenths of a level, so encoder and decoder share it exactly
  const int step = std::clamp(static_cast<int>(std::lround(step_in * 16.f)), 16, 65535);
  const auto rq = static_cast<int>(std::clamp(std::lround(round * 256.f), 0L, 128L));
  ClipResidual out;
  auto& s = out.stream;
  s = {'G', '3', 'B', kVersion};
  put_le(s, static_cast<std::uint64_t>(frames), 2);
  put_le(s, static_cast<std::uint64_t>(size), 2);
  put_le(s, static_cast<std::uint64_t>(step), 2);
  s.push_back(static_cast<std::uint8_t>(rq));
  s.push_back(0);
  std::uint32_t sum = kFnvBasis;
  for (const std::uint8_t b : s) fnv(sum, b);
  ArithEncoder ac(s);
  ResidualModel rm;
  out.frames.resize(pred.size());
  std::vector<std::int32_t> r(static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4);
  for (int f = 0; f < frames; ++f) {
    const std::size_t o = static_cast<std::size_t>(f) * fb;
    for (std::size_t i = 0; i < fb; ++i) {
      const int d = static_cast<int>(truth[o + i]) - static_cast<int>(pred[o + i]);
      // sign(d) floor(|d| / step + round), with the step in sixteenths of a level; the decoder clamps to 8 bits
      const int q = (std::abs(d) * 16 * 256 + rq * step) / (step * 256);
      r[i] = d < 0 ? -q : q;
      out.frames[o + i] = level(pred[o + i], r[i], step);
    }
    rm.encode(ac, PlaneKind::pixel, size, size, 4, r);
    for (const std::int32_t v : r) fnv(sum, static_cast<std::uint32_t>(v));
  }
  ac.flush();
  put_le(s, sum, 4);
  return out;
}

std::expected<std::vector<std::uint8_t>, std::string> decode_clip_residual(std::span<const std::uint8_t> pred, std::span<const std::uint8_t> stream) {
  if (stream.size() < kHeader + 4 || stream[0] != 'G' || stream[1] != '3' || stream[2] != 'B') return std::unexpected("g3b: not a clip residual stream");
  if (stream[3] != kVersion) return std::unexpected("g3b: unsupported version");
  const int frames = static_cast<int>(get_le(stream, 4, 2)), size = static_cast<int>(get_le(stream, 6, 2)), step = static_cast<int>(get_le(stream, 8, 2));
  const std::size_t fb = static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4;
  if (size <= 0 || frames <= 0 || step < 16 || stream[10] > 128 || stream[11] != 0) return std::unexpected("g3b: corrupt header");
  if (pred.size() != fb * static_cast<std::size_t>(frames)) return std::unexpected("g3b: the prediction does not match the stream's size or length");
  std::uint32_t sum = kFnvBasis;
  for (std::size_t i = 0; i < kHeader; ++i) fnv(sum, stream[i]);
  ArithDecoder ad(stream.subspan(kHeader, stream.size() - kHeader - 4));
  ResidualModel rm;
  std::vector<std::uint8_t> out(pred.size());
  std::vector<std::int32_t> r(static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4);
  for (int f = 0; f < frames; ++f) {
    if (!rm.decode(ad, PlaneKind::pixel, size, size, 4, r)) return std::unexpected("g3b: damaged stream (the decoder ran past its end)");
    const std::size_t o = static_cast<std::size_t>(f) * fb;
    for (std::size_t i = 0; i < fb; ++i) {
      out[o + i] = level(pred[o + i], r[i], step);
      fnv(sum, static_cast<std::uint32_t>(r[i]));
    }
  }
  if (sum != static_cast<std::uint32_t>(get_le(stream, stream.size() - 4, 4))) return std::unexpected("g3b: damaged stream (checksum mismatch)");
  return out;
}

}  // namespace nfx::codec
