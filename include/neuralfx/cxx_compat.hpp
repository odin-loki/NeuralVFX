// Forced on Windows exe builds (-include). Supplies C++23 headers missing from llvm-mingw libc++.
#pragma once

#include <cstdint>
#include <istream>
#include <span>
#include <streambuf>

#if defined(NEURALFX_WIN_CROSS) || (defined(_WIN32) && !defined(__has_include))
#define NEURALFX_WIN_CROSS 1
#endif

#if defined(NEURALFX_WIN_CROSS) || !__has_include(<stdfloat>)
namespace std {
#if defined(__FLT16_MANT_DIG__)
using float16_t = _Float16;
#elif defined(__GNUC__)
using float16_t = __float16;
#else
using float16_t = _Float16;
#endif
}
#endif

#if defined(NEURALFX_WIN_CROSS) || !__has_include(<spanstream>)
namespace std {
class spanstreambuf : public std::streambuf {
 public:
  explicit spanstreambuf(std::span<const char> bytes) {
    char* p = const_cast<char*>(bytes.data());
    setg(p, p, p + static_cast<std::ptrdiff_t>(bytes.size()));
  }
};
class ispanstream : public std::istream {
 public:
  explicit ispanstream(std::span<const char> bytes) : std::istream(&buf_), buf_(bytes) {}
 private:
  spanstreambuf buf_;
};
}  // namespace std
#endif
