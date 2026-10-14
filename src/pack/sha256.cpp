// SHA-256 (FIPS 180-4), streaming: 64-byte blocks compressed as they fill, the message never copied whole.
#include "sha256.hpp"

#include <algorithm>
#include <cstring>

namespace nfx::pack {

namespace {

constexpr std::array<std::uint32_t, 64> kK = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

}  // namespace

Sha256::Sha256() : h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19} {}

void Sha256::block(const std::uint8_t* p) {
  std::array<std::uint32_t, 64> w;
  for (std::size_t i = 0; i < 16; ++i)
    w[i] = (std::uint32_t{p[4 * i]} << 24) | (std::uint32_t{p[4 * i + 1]} << 16) | (std::uint32_t{p[4 * i + 2]} << 8) | std::uint32_t{p[4 * i + 3]};
  for (std::size_t i = 16; i < 64; ++i) {
    const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
  for (std::size_t i = 0; i < 64; ++i) {
    const std::uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + kK[i] + w[i];
    const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h_[0] += a;
  h_[1] += b;
  h_[2] += c;
  h_[3] += d;
  h_[4] += e;
  h_[5] += f;
  h_[6] += g;
  h_[7] += h;
}

void Sha256::update(std::span<const std::uint8_t> bytes) {
  total_ += bytes.size();
  std::size_t i = 0;
  if (fill_ > 0) {
    const std::size_t n = std::min(bytes.size(), 64 - fill_);
    std::memcpy(buf_.data() + fill_, bytes.data(), n);
    fill_ += n;
    i = n;
    if (fill_ < 64) return;
    block(buf_.data());
    fill_ = 0;
  }
  for (; i + 64 <= bytes.size(); i += 64) block(bytes.data() + i);
  if (i < bytes.size()) {
    std::memcpy(buf_.data(), bytes.data() + i, bytes.size() - i);
    fill_ = bytes.size() - i;
  }
}

Digest Sha256::finish() {
  const std::uint64_t bits = total_ * 8;
  std::array<std::uint8_t, 72> pad{};
  pad[0] = 0x80;
  const std::size_t n = (fill_ < 56 ? 56 - fill_ : 120 - fill_);
  for (int k = 0; k < 8; ++k) pad[n + static_cast<std::size_t>(k)] = static_cast<std::uint8_t>(bits >> (56 - 8 * k));
  const std::uint64_t keep = total_;
  update(std::span(pad.data(), n + 8));
  total_ = keep;
  Digest out;
  for (std::size_t k = 0; k < 8; ++k) {
    out[4 * k] = static_cast<std::uint8_t>(h_[k] >> 24);
    out[4 * k + 1] = static_cast<std::uint8_t>(h_[k] >> 16);
    out[4 * k + 2] = static_cast<std::uint8_t>(h_[k] >> 8);
    out[4 * k + 3] = static_cast<std::uint8_t>(h_[k]);
  }
  return out;
}

Digest sha256(std::span<const std::uint8_t> bytes) {
  Sha256 s;
  s.update(bytes);
  return s.finish();
}

std::string hex(const Digest& d) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string s(64, '0');
  for (std::size_t i = 0; i < 32; ++i) {
    s[2 * i] = kHex[d[i] >> 4];
    s[2 * i + 1] = kHex[d[i] & 15];
  }
  return s;
}

bool parse_hex(std::string_view s, Digest& out) {
  if (s.size() != 64) return false;
  const auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  for (std::size_t i = 0; i < 32; ++i) {
    const int hi = nib(s[2 * i]), lo = nib(s[2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = static_cast<std::uint8_t>(hi * 16 + lo);
  }
  return true;
}

}  // namespace nfx::pack
