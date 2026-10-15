// The volume runtime's 3 x 3 x 3 convolution with AVX2 and FMA (study N; src/runtime/nvfx_volume.cpp picks it when the
// ISA is not forced to the baseline). The sums of rollout3d::conv3_reference in the same order, fused, so the results
// differ from the reference's in the last bits (the parity tests allow for it).
#include <neuralfx/rollout3d.hpp>

#include <cstddef>

#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#pragma GCC target("avx2,fma")
#endif

namespace nfx::rt {

namespace {
typedef float v8 __attribute__((vector_size(32)));
typedef float v8u __attribute__((vector_size(32), aligned(4)));
std::size_t zu(int v) { return static_cast<std::size_t>(v); }
}  // namespace

void volume_conv3_avx2(int rx, int ry, int rz, const float* in, int ci, const float* W, const float* b, int co, float* out, int y0, int y1) {
  if (co % 8 != 0 || co > 64) {
    rollout3d::conv3_reference(rx, ry, rz, in, ci, W, b, co, out, y0, y1);
    return;
  }
  const int nv = co / 8;
  for (int y = y0; y < y1; ++y) {
    for (int z = 0; z < rz; ++z) {
      for (int x = 0; x < rx; ++x) {
        v8 acc[8];
        for (int k = 0; k < nv; ++k) acc[k] = *reinterpret_cast<const v8u*>(b + 8 * k);
        for (int dy = -1; dy <= 1; ++dy) {
          const int yy = y + dy;
          if (yy < 0 || yy >= ry) continue;
          for (int dz = -1; dz <= 1; ++dz) {
            const int zz = z + dz;
            if (zz < 0 || zz >= rz) continue;
            for (int dx = -1; dx <= 1; ++dx) {
              const int xx = x + dx;
              if (xx < 0 || xx >= rx) continue;
              const float* a = in + ((zu(yy) * zu(rz) + zu(zz)) * zu(rx) + zu(xx)) * zu(ci);
              const float* w = W + zu(((dy + 1) * 3 + (dz + 1)) * 3 + (dx + 1)) * zu(ci) * zu(co);
              for (int c = 0; c < ci; ++c) {
                const float av = a[c];
                const v8 bv = {av, av, av, av, av, av, av, av};
                const float* wr = w + zu(c) * zu(co);
                for (int k = 0; k < nv; ++k) acc[k] += *reinterpret_cast<const v8u*>(wr + 8 * k) * bv;
              }
            }
          }
        }
        float* o = out + ((zu(y) * zu(rz) + zu(z)) * zu(rx) + zu(x)) * zu(co);
        for (int k = 0; k < nv; ++k) *reinterpret_cast<v8u*>(o + 8 * k) = acc[k];
      }
    }
  }
}

}  // namespace nfx::rt
