// SHA-256 (FIPS 180-4) for the effect pack's checksums (docs/INTEGRATION.md §5): streaming, no allocation.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace nfx::pack {

using Digest = std::array<std::uint8_t, 32>;

class Sha256 {
 public:
  Sha256();
  void update(std::span<const std::uint8_t> bytes);
  Digest finish();  // the digest; the object is then spent

 private:
  void block(const std::uint8_t* p);
  std::array<std::uint32_t, 8> h_;
  std::array<std::uint8_t, 64> buf_{};
  std::size_t fill_ = 0;
  std::uint64_t total_ = 0;
};

Digest sha256(std::span<const std::uint8_t> bytes);
std::string hex(const Digest& d);
// 64 hex digits (either case) into a digest; false if it is not one.
bool parse_hex(std::string_view s, Digest& out);

}  // namespace nfx::pack
