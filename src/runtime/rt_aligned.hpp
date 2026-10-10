// Storage aligned to a cache line, for the runtime's kernels (rt_prior.hpp, rt_rollout.hpp).
#pragma once

#include <cstddef>
#include <new>
#include <vector>

namespace nfx::rt {

// The kernels read 16-, 32- and 64-byte vectors at offsets that are multiples of their width; with std::vector's 16-byte
// alignment every other 32-byte load straddled two cache lines (about 25% of the prior's pass, rt_prior.hpp).
template <class T>
struct CacheAligned {
  using value_type = T;
  static constexpr std::align_val_t kAlign{64};
  CacheAligned() noexcept = default;
  template <class U>
  CacheAligned(const CacheAligned<U>&) noexcept {}
  T* allocate(std::size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), kAlign)); }
  void deallocate(T* p, std::size_t n) noexcept { ::operator delete(p, n * sizeof(T), kAlign); }
  friend bool operator==(const CacheAligned&, const CacheAligned&) noexcept { return true; }
};
using AlignedFloats = std::vector<float, CacheAligned<float>>;

}  // namespace nfx::rt
