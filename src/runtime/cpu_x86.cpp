#include "cpu_x86.hpp"

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#elif defined(__GNUC__)
#include <cpuid.h>
#else
#include <immintrin.h>
#endif

namespace nfx::rt {
namespace {

void cpuid_leaf(int leaf, int sub, std::uint32_t* eax, std::uint32_t* ebx, std::uint32_t* ecx, std::uint32_t* edx) {
#if defined(_MSC_VER) || defined(__MINGW32__)
  int r[4];
  __cpuidex(r, leaf, sub);
  *eax = static_cast<std::uint32_t>(r[0]);
  *ebx = static_cast<std::uint32_t>(r[1]);
  *ecx = static_cast<std::uint32_t>(r[2]);
  *edx = static_cast<std::uint32_t>(r[3]);
#else
  __cpuid_count(leaf, sub, *eax, *ebx, *ecx, *edx);
#endif
}

std::uint64_t xcr0() {
#if defined(_MSC_VER) || defined(__MINGW32__)
  return _xgetbv(0);
#else
  unsigned int eax = 0, edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<std::uint64_t>(edx) << 32) | eax;
#endif
}

bool os_avx_state() {
  std::uint32_t a = 0, b = 0, c = 0, d = 0;
  cpuid_leaf(1, 0, &a, &b, &c, &d);
  if ((c & (1u << 27)) == 0) return false;  // OSXSAVE
  return (xcr0() & 0x6u) == 0x6u;
}

bool os_avx512_state() {
  if (!os_avx_state()) return false;
  return (xcr0() & 0xe6u) == 0xe6u;
}

}  // namespace

bool cpu_x86_has_avx2() {
  if (!os_avx_state()) return false;
  std::uint32_t a = 0, b = 0, c = 0, d = 0;
  cpuid_leaf(7, 0, &a, &b, &c, &d);
  return (b & (1u << 5)) != 0;
}

bool cpu_x86_has_avx512f() {
  if (!os_avx512_state()) return false;
  std::uint32_t a = 0, b = 0, c = 0, d = 0;
  cpuid_leaf(7, 0, &a, &b, &c, &d);
  return (b & (1u << 16)) != 0;
}

bool cpu_x86_has_avx512_vnni() {
  if (!cpu_x86_has_avx512f()) return false;
  std::uint32_t a = 0, b = 0, c = 0, d = 0;
  cpuid_leaf(7, 0, &a, &b, &c, &d);
  return (c & (1u << 11)) != 0;
}

bool cpu_x86_has_f16c() {
  if (!os_avx_state()) return false;
  std::uint32_t a = 0, b = 0, c = 0, d = 0;
  cpuid_leaf(1, 0, &a, &b, &c, &d);
  return (c & (1u << 29)) != 0;
}

}  // namespace nfx::rt

#else

namespace nfx::rt {
bool cpu_x86_has_avx2() { return false; }
bool cpu_x86_has_avx512f() { return false; }
bool cpu_x86_has_avx512_vnni() { return false; }
bool cpu_x86_has_f16c() { return false; }
}  // namespace nfx::rt

#endif
