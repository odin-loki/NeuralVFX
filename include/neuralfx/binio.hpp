// Little-endian binary I/O for the project's file formats (.nfxclip, .nvfx), independent of the host's byte order.
#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <expected>
#include <istream>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

namespace nfx::bin {

template <class T>
  requires std::is_arithmetic_v<T>
void put(std::ostream& o, T v) {
  if constexpr (sizeof(T) > 1 && std::endian::native == std::endian::big) {
    using U = std::conditional_t<sizeof(T) == 2, std::uint16_t, std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>>;
    v = std::bit_cast<T>(std::byteswap(std::bit_cast<U>(v)));
  }
  o.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <class T>
  requires std::is_arithmetic_v<T>
std::expected<T, std::string> get(std::istream& i) {
  T v{};
  i.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (!i) return std::unexpected("truncated file");
  if constexpr (sizeof(T) > 1 && std::endian::native == std::endian::big) {
    using U = std::conditional_t<sizeof(T) == 2, std::uint16_t, std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>>;
    v = std::bit_cast<T>(std::byteswap(std::bit_cast<U>(v)));
  }
  return v;
}

// A fixed-width, NUL-padded string field.
inline void put_str(std::ostream& o, std::string_view s, std::size_t width) {
  std::string t(s.substr(0, width - 1));
  t.resize(width, '\0');
  o.write(t.data(), static_cast<std::streamsize>(width));
}

inline std::expected<std::string, std::string> get_str(std::istream& i, std::size_t width) {
  std::string s(width, '\0');
  i.read(s.data(), static_cast<std::streamsize>(width));
  if (!i) return std::unexpected("truncated file");
  s.resize(std::strlen(s.c_str()));
  return s;
}

// Arrays of arithmetic values, element by element in little-endian order.
template <class T>
void put_array(std::ostream& o, std::span<const T> v) {
  if constexpr (std::endian::native == std::endian::little) {
    o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size_bytes()));
  } else {
    for (const T x : v) put(o, x);
  }
}

template <class T>
std::expected<void, std::string> get_array(std::istream& i, std::span<T> v) {
  if constexpr (std::endian::native == std::endian::little) {
    i.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size_bytes()));
    if (!i) return std::unexpected("truncated file");
  } else {
    for (T& x : v) {
      auto r = get<T>(i);
      if (!r) return std::unexpected(r.error());
      x = *r;
    }
  }
  return {};
}

}  // namespace nfx::bin
