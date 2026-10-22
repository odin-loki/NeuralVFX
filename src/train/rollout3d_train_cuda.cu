// CUDA trainer for study N's 3D rollout (rollout3d_train.cpp). Matches the CPU forward and backward; conv3 uses Y-slab
// sharding across CUDA streams (TrainOptions::shards).
#include <neuralfx/rollout3d_train_cuda_internal.hpp>

using nfx::rollout3d::cuda_detail::kNoise;
using nfx::rollout3d::cuda_detail::kPhys;

#include <cuda_runtime.h>

#include <cstring>

namespace nfx::rollout3d::cuda_detail {
namespace {

using GpuHyper = cuda_detail::GpuHyper;
using GpuLayout = cuda_detail::GpuLayout;

__device__ int g_at(int x, int y, int z, int rx, int rz) { return (y * rz + z) * rx + x; }
__device__ int cl(int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); }

struct TriDev {
  int c[8];
  unsigned char in;
  float fx, fy, fz;
  bool inx, iny, inz;
};

__device__ TriDev tri_at_dev(const GpuHyper& h, int x, int y, int z, float rxp, float ryp, float rzp) {
  TriDev t{};
  const int rx = h.rx, ry = h.ry, rz = h.rz;
  if (h.edges == 0) {
    const float px = fminf(fmaxf(rxp, 0.f), float(rx - 1));
    const float py = fminf(fmaxf(ryp, 0.f), float(ry - 1));
    const float pz = fminf(fmaxf(rzp, 0.f), float(rz - 1));
    t.inx = rxp > 0.f && rxp < float(rx - 1);
    t.iny = ryp > 0.f && ryp < float(ry - 1);
    t.inz = rzp > 0.f && rzp < float(rz - 1);
    const int x0 = min((int)px, rx - 2), y0 = min((int)py, ry - 2), z0 = min((int)pz, rz - 2);
    t.fx = px - float(x0);
    t.fy = py - float(y0);
    t.fz = pz - float(z0);
    for (int k = 0; k < 8; ++k) t.c[k] = g_at(x0 + (k & 1), y0 + ((k >> 2) & 1), z0 + ((k >> 1) & 1), rx, rz);
    t.in = 0xff;
    return t;
  }
  const float px = fminf(fmaxf(rxp, -1.f), float(rx));
  const float py = fminf(fmaxf(ryp, 0.f), float(ry));
  const float pz = fminf(fmaxf(rzp, -1.f), float(rz));
  t.inx = rxp > -1.f && rxp < float(rx);
  t.iny = ryp > 0.f && ryp < float(ry);
  t.inz = rzp > -1.f && rzp < float(rz);
  const int x0 = min((int)floorf(px), rx - 1), y0 = min((int)py, ry - 1), z0 = min((int)floorf(pz), rz - 1);
  t.fx = px - float(x0);
  t.fy = py - float(y0);
  t.fz = pz - float(z0);
  t.in = 0;
  for (int k = 0; k < 8; ++k) {
    const int xx = x0 + (k & 1), yy = y0 + ((k >> 2) & 1), zz = z0 + ((k >> 1) & 1);
    if (xx < 0 || zz < 0 || xx >= rx || yy >= ry || zz >= rz) {
      t.c[k] = 0;
      continue;
    }
    t.c[k] = g_at(xx, yy, zz, rx, rz);
    t.in = (unsigned char)(t.in | (1u << k));
  }
  return t;
}

__device__ float tri_value_dev(const TriDev& t, const float* S, int C, int c) {
  const auto v = [&](int k) { return ((t.in >> k) & 1u) ? S[t.c[k] * C + c] : 0.f; };
  const float a00 = v(0) + t.fx * (v(1) - v(0)), a10 = v(2) + t.fx * (v(3) - v(2));
  const float a01 = v(4) + t.fx * (v(5) - v(4)), a11 = v(6) + t.fx * (v(7) - v(6));
  const float b0 = a00 + t.fz * (a10 - a00), b1 = a01 + t.fz * (a11 - a01);
  return b0 + t.fy * (b1 - b0);
}

__global__ void conv3_fwd_y(GpuHyper h, const float* in, int ci, const float* W, const float* b, int co, float* out, int y0, int y1) {
  const int rx = h.rx, rz = h.rz;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int cells = rx * (y1 - y0) * rz;
  if (tid >= cells) return;
  const int z = tid / (rx * (y1 - y0));
  const int rem = tid % (rx * (y1 - y0));
  const int y = y0 + rem / rx;
  const int x = rem % rx;
  float* o = out + g_at(x, y, z, rx, rz) * co;
  for (int k = 0; k < co; ++k) o[k] = b[k];
  for (int dy = -1; dy <= 1; ++dy) {
    const int yy = y + dy;
    if (yy < 0 || yy >= h.ry) continue;
    for (int dz = -1; dz <= 1; ++dz) {
      const int zz = z + dz;
      if (zz < 0 || zz >= rz) continue;
      for (int dx = -1; dx <= 1; ++dx) {
        const int xx = x + dx;
        if (xx < 0 || xx >= rx) continue;
        const float* a = in + g_at(xx, yy, zz, rx, rz) * ci;
        const float* w = W + (((dy + 1) * 3 + (dz + 1)) * 3 + (dx + 1)) * ci * co;
        for (int c = 0; c < ci; ++c) {
          const float av = a[c];
          const float* wr = w + c * co;
          for (int k = 0; k < co; ++k) o[k] += wr[k] * av;
        }
      }
    }
  }
}

__global__ void conv3_bwd_y(GpuHyper h, const float* in, int ci, const float* W, int co, const float* go_all, float* gin, float* gW,
                            float* gb, int y0, int y1) {
  const int rx = h.rx, rz = h.rz;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int cells = rx * (y1 - y0) * rz;
  if (tid >= cells) return;
  const int z = tid / (rx * (y1 - y0));
  const int rem = tid % (rx * (y1 - y0));
  const int y = y0 + rem / rx;
  const int x = rem % rx;
  const float* go = go_all + g_at(x, y, z, rx, rz) * co;
  for (int k = 0; k < co; ++k) atomicAdd(gb + k, go[k]);
  for (int dy = -1; dy <= 1; ++dy) {
    const int yy = y + dy;
    if (yy < 0 || yy >= h.ry) continue;
    for (int dz = -1; dz <= 1; ++dz) {
      const int zz = z + dz;
      if (zz < 0 || zz >= rz) continue;
      for (int dx = -1; dx <= 1; ++dx) {
        const int xx = x + dx;
        if (xx < 0 || xx >= rx) continue;
        const int nb = g_at(xx, yy, zz, rx, rz);
        const float* a = in + nb * ci;
        float* ga = gin ? gin + nb * ci : nullptr;
        const int t = (((dy + 1) * 3 + (dz + 1)) * 3 + (dx + 1)) * ci * co;
        for (int c = 0; c < ci; ++c) {
          const float av = a[c];
          float* gw = gW + t + c * co;
          const float* wr = W + t + c * co;
          float s = 0.f;
          for (int k = 0; k < co; ++k) {
            atomicAdd(gw + k, av * go[k]);
            s += wr[k] * go[k];
          }
          if (ga) atomicAdd(ga + c, s);
        }
      }
    }
  }
}

__global__ void conv1_fwd(GpuHyper h, const float* in, int ci, const float* W, const float* b, int co, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  float* o = out + i * co;
  for (int k = 0; k < co; ++k) o[k] = b[k];
  const float* a = in + i * ci;
  for (int c = 0; c < ci; ++c) {
    const float av = a[c];
    for (int k = 0; k < co; ++k) o[k] += W[c * co + k] * av;
  }
}

__global__ void conv1_bwd(GpuHyper h, const float* in, int ci, const float* W, int co, const float* g, float* gin, float* gW, float* gb) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  const float* go = g + i * co;
  const float* a = in + i * ci;
  float* ga = gin + i * ci;
  for (int k = 0; k < co; ++k) atomicAdd(gb + k, go[k]);
  for (int c = 0; c < ci; ++c) {
    float s = 0.f;
    for (int k = 0; k < co; ++k) {
      atomicAdd(gW + c * co + k, a[c] * go[k]);
      s += W[c * co + k] * go[k];
    }
    atomicAdd(ga + c, s);
  }
}

__global__ void build_x(GpuHyper h, const float* in, const float* noise, float* X) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  const int rx = h.rx, ry = h.ry, rz = h.rz;
  const int z = (i / rx) % rz;
  const int y = i / (rx * rz);
  const int x = i % rx;
  float* xi = X + i * h.I;
  for (int k = 0; k < h.C; ++k) xi[k] = in[i * h.C + k] / (k < kPhys ? h.scale[k] : 1.f);
  for (int k = 0; k < kNoise; ++k) xi[h.C + k] = noise[i * kNoise + k];
  xi[h.C + kNoise] = (float(x) + 0.5f) / float(rx) * 2.f - 1.f;
  xi[h.C + kNoise + 1] = (float(y) + 0.5f) / float(ry) * 2.f - 1.f;
  xi[h.C + kNoise + 2] = (float(z) + 0.5f) / float(rz) * 2.f - 1.f;
}

__global__ void film_relu(GpuHyper h, const float* y, const float* g, const float* e, float* a) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N * h.H) return;
  const int j = i % h.H;
  a[i] = fmaxf(0.f, g[j] * y[i] + e[j]);
}

__global__ void mid_fwd(const GpuHyper h, const float* in, const float* out, float* mid) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  for (int k = 0; k < kPhys; ++k) mid[i * h.C + k] = in[i * h.C + k] + h.scale[k] * out[i * h.O + k];
  for (int k = kPhys; k < h.C; ++k) mid[i * h.C + k] = tanhf(in[i * h.C + k] + out[i * h.O + k]);
}

__global__ void div_fwd(const GpuHyper h, const float* mid, const float* out, float* div) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  const int rx = h.rx, ry = h.ry, rz = h.rz;
  const int z = (i / rx) % rz;
  const int y = i / (rx * rz);
  const int x = i % rx;
  const auto U = [&](int xx, int yy, int zz, int c) { return mid[g_at(cl(xx, rx), cl(yy, ry), cl(zz, rz), rx, rz) * h.C + c]; };
  div[i] = -0.5f * (U(x + 1, y, z, 0) - U(x - 1, y, z, 0) + U(x, y + 1, z, 1) - U(x, y - 1, z, 1) + U(x, y, z + 1, 2) - U(x, y, z - 1, 2)) +
           h.qscale * out[i * h.O + (h.O - 1)];
}

__global__ void jacobi_iter(GpuHyper h, const float* div, float* p, float* tmp) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  const int rx = h.rx, ry = h.ry, rz = h.rz;
  const int z = (i / rx) % rz;
  const int y = i / (rx * rz);
  const int x = i % rx;
  float s = 0.f;
  if (x > 0) s += p[g_at(x - 1, y, z, rx, rz)];
  if (x < rx - 1) s += p[g_at(x + 1, y, z, rx, rz)];
  if (z > 0) s += p[g_at(x, y, z - 1, rx, rz)];
  if (z < rz - 1) s += p[g_at(x, y, z + 1, rx, rz)];
  s += p[g_at(x, max(y - 1, 0), z, rx, rz)];
  if (y < ry - 1) s += p[g_at(x, y + 1, z, rx, rz)];
  tmp[i] = s;
  p[i] = (1.f / 6.f) * (div[i] + tmp[i]);
}

__device__ float pval_dev(const GpuHyper& h, const float* p, int x, int y, int z) {
  if (x < 0 || z < 0 || x >= h.rx || z >= h.rz || y >= h.ry) return 0.f;
  return p[g_at(x, max(y, 0), z, h.rx, h.rz)];
}

__global__ void subtract_grad(GpuHyper h, const float* p, float* S) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  const int rx = h.rx, ry = h.ry, rz = h.rz;
  const int z = (i / rx) % rz;
  const int y = i / (rx * rz);
  const int x = i % rx;
  float* s = S + i * h.C;
  s[0] -= 0.5f * (pval_dev(h, p, x + 1, y, z) - pval_dev(h, p, x - 1, y, z));
  s[1] -= 0.5f * (pval_dev(h, p, x, y + 1, z) - pval_dev(h, p, x, y - 1, z));
  s[2] -= 0.5f * (pval_dev(h, p, x, y, z + 1) - pval_dev(h, p, x, y, z - 1));
}

__global__ void advect_fwd(GpuHyper h, const float* S, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  const int rx = h.rx, ry = h.ry, rz = h.rz;
  const int z = (i / rx) % rz;
  const int y = i / (rx * rz);
  const int x = i % rx;
  const float* s = S + i * h.C;
  const TriDev t = tri_at_dev(h, x, y, z, float(x) - s[0], float(y) - s[1], float(z) - s[2]);
  float* o = out + i * h.C;
  for (int c = 0; c < h.C; ++c) o[c] = tri_value_dev(t, S, h.C, c);
}

__global__ void clamp_phys(GpuHyper h, float* next, unsigned char* clamped) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  for (int k = 0; k < kPhys; ++k) {
    float& v = next[i * h.C + k];
    if (v < h.lo[k] || v > h.hi[k]) {
      clamped[i * kPhys + k] = 1;
      v = fminf(fmaxf(v, h.lo[k]), h.hi[k]);
    }
  }
}

__global__ void zero_clamped_g(GpuHyper h, float* gS, const unsigned char* clamped) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  for (int k = 0; k < kPhys; ++k) {
    if (clamped[i * kPhys + k]) gS[i * h.C + k] = 0.f;
  }
}

__global__ void advect_bwd(GpuHyper h, const float* S, int C, const float* gout, float* gS) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  const int rx = h.rx, ry = h.ry, rz = h.rz;
  const int z = (i / rx) % rz;
  const int y = i / (rx * rz);
  const int x = i % rx;
  const float* s = S + i * C;
  const TriDev t = tri_at_dev(h, x, y, z, float(x) - s[0], float(y) - s[1], float(z) - s[2]);
  const float fx = t.fx, fy = t.fy, fz = t.fz;
  const float w[8] = {(1 - fx) * (1 - fz) * (1 - fy), fx * (1 - fz) * (1 - fy), (1 - fx) * fz * (1 - fy), fx * fz * (1 - fy),
                        (1 - fx) * (1 - fz) * fy,       fx * (1 - fz) * fy,       (1 - fx) * fz * fy,       fx * fz * fy};
  const float* go = gout + i * C;
  float dpx = 0.f, dpy = 0.f, dpz = 0.f;
  for (int c = 0; c < C; ++c) {
    const float gc = go[c];
    if (gc == 0.f) continue;
    float v[8];
    for (int k = 0; k < 8; ++k) {
      if (!((t.in >> k) & 1u)) {
        v[k] = 0.f;
        continue;
      }
      v[k] = S[t.c[k] * C + c];
      atomicAdd(gS + t.c[k] * C + c, gc * w[k]);
    }
    dpx += gc * ((1 - fz) * (1 - fy) * (v[1] - v[0]) + fz * (1 - fy) * (v[3] - v[2]) + (1 - fz) * fy * (v[5] - v[4]) + fz * fy * (v[7] - v[6]));
    dpz += gc * ((1 - fx) * (1 - fy) * (v[2] - v[0]) + fx * (1 - fy) * (v[3] - v[1]) + (1 - fx) * fy * (v[6] - v[4]) + fx * fy * (v[7] - v[5]));
    dpy += gc * ((1 - fx) * (1 - fz) * (v[4] - v[0]) + fx * (1 - fz) * (v[5] - v[1]) + (1 - fx) * fz * (v[6] - v[2]) + fx * fz * (v[7] - v[3]));
  }
  if (t.inx) atomicAdd(gS + i * C, -dpx);
  if (t.iny) atomicAdd(gS + i * C + 1, -dpy);
  if (t.inz) atomicAdd(gS + i * C + 2, -dpz);
}

__device__ void add_gp(const GpuHyper& h, float* gp, int x, int y, int z, float v) {
  if (x < 0 || z < 0 || x >= h.rx || z >= h.rz || y >= h.ry) return;
  atomicAdd(gp + g_at(x, max(y, 0), z, h.rx, h.rz), v);
}

__global__ void subtract_grad_bwd(GpuHyper h, const float* gS, int C, float* gp) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  const int rx = h.rx, ry = h.ry, rz = h.rz;
  const int z = (i / rx) % rz;
  const int y = i / (rx * rz);
  const int x = i % rx;
  const float* s = gS + i * C;
  add_gp(h, gp, x + 1, y, z, -0.5f * s[0]);
  add_gp(h, gp, x - 1, y, z, 0.5f * s[0]);
  add_gp(h, gp, x, y + 1, z, -0.5f * s[1]);
  add_gp(h, gp, x, y - 1, z, 0.5f * s[1]);
  add_gp(h, gp, x, y, z + 1, -0.5f * s[2]);
  add_gp(h, gp, x, y, z - 1, 0.5f * s[2]);
}

__global__ void neighbour_sum(GpuHyper h, const float* p, float* o) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  const int rx = h.rx, ry = h.ry, rz = h.rz;
  const int z = (i / rx) % rz;
  const int y = i / (rx * rz);
  const int x = i % rx;
  float s = 0.f;
  if (x > 0) s += p[g_at(x - 1, y, z, rx, rz)];
  if (x < rx - 1) s += p[g_at(x + 1, y, z, rx, rz)];
  if (z > 0) s += p[g_at(x, y, z - 1, rx, rz)];
  if (z < rz - 1) s += p[g_at(x, y, z + 1, rx, rz)];
  s += p[g_at(x, max(y - 1, 0), z, rx, rz)];
  if (y < ry - 1) s += p[g_at(x, y + 1, z, rx, rz)];
  o[i] = s;
}

__global__ void jacobi_bwd_add(GpuHyper h, const float* w, float* gdiv) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  atomicAdd(gdiv + i, (1.f / 6.f) * w[i]);
}

__global__ void scale_w(GpuHyper h, float* w) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  w[i] *= (1.f / 6.f);
}

__global__ void divergence_bwd(GpuHyper h, int O, float qs, const float* gdiv, float* gS, float* gd) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  const int rx = h.rx, ry = h.ry, rz = h.rz;
  const int z = (i / rx) % rz;
  const int y = i / (rx * rz);
  const int x = i % rx;
  const float v = gdiv[i];
  const int C = h.C;
  const auto addS = [&](int xx, int yy, int zz, int c, float d) { atomicAdd(gS + g_at(cl(xx, rx), cl(yy, ry), cl(zz, rz), rx, rz) * C + c, d); };
  addS(x + 1, y, z, 0, -0.5f * v);
  addS(x - 1, y, z, 0, 0.5f * v);
  addS(x, y + 1, z, 1, -0.5f * v);
  addS(x, y - 1, z, 1, 0.5f * v);
  addS(x, y, z + 1, 2, -0.5f * v);
  addS(x, y, z - 1, 2, 0.5f * v);
  atomicAdd(gd + i * O + (O - 1), qs * v);
}

__global__ void tanh_mid_bwd(GpuHyper h, const float* mid, float* gmid) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  for (int k = kPhys; k < h.C; ++k) {
    const float t = mid[i * h.C + k];
    gmid[i * h.C + k] *= 1.f - t * t;
  }
}

__global__ void mid_to_gout(GpuHyper h, const float* gmid, float* gout) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h.N) return;
  for (int k = 0; k < h.C; ++k) atomicAdd(gout + i * h.O + k, (k < kPhys ? h.scale[k] : 1.f) * gmid[i * h.C + k]);
}

}  // namespace

void sync_streams(int shards, cudaStream_t* streams) {
  for (int s = 0; s < shards; ++s) cudaStreamSynchronize(streams[s]);
}

void launch_conv3_fwd(const GpuHyper& gh, cudaStream_t* streams, int shards, const float* in, int ci, const float* W, const float* b, int co,
                      float* out) {
  const int block = 256;
  for (int s = 0; s < shards; ++s) {
    const int y0 = s * gh.ry / shards;
    const int y1 = (s + 1) * gh.ry / shards;
    if (y1 <= y0) continue;
    const int cells = gh.rx * (y1 - y0) * gh.rz;
    conv3_fwd_y<<<(cells + block - 1) / block, block, 0, streams[s]>>>(gh, in, ci, W, b, co, out, y0, y1);
  }
}

void launch_conv3_bwd(const GpuHyper& gh, cudaStream_t* streams, int shards, const float* in, int ci, const float* W, int co, const float* go,
                      float* gin, float* gW, float* gb) {
  const int block = 256;
  for (int s = 0; s < shards; ++s) {
    const int y0 = s * gh.ry / shards;
    const int y1 = (s + 1) * gh.ry / shards;
    if (y1 <= y0) continue;
    const int cells = gh.rx * (y1 - y0) * gh.rz;
    conv3_bwd_y<<<(cells + block - 1) / block, block, 0, streams[s]>>>(gh, in, ci, W, co, go, gin, gW, gb, y0, y1);
  }
}

void cuda_forward_step(const GpuHyper& gh, int shards, cudaStream_t* streams, const CudaBuffers& buf, const FilmHost& film) {
  const cudaStream_t st = streams[0];
  const int block = 256;
  const int gridN = (gh.N + block - 1) / block;
  const float* w = buf.d_w;
  const GpuLayout& L = gh.L;
  build_x<<<gridN, block, 0, st>>>(gh, buf.d_S, buf.d_noise, buf.d_X);
  cudaStreamSynchronize(st);
  launch_conv3_fwd(gh, streams, shards, buf.d_X, gh.I, w + L.w1, w + L.b1, gh.H, buf.d_y1);
  sync_streams(shards, streams);
  float* d_g1 = buf.d_film_g;
  float* d_e1 = buf.d_film_e;
  cudaMemcpyAsync(d_g1, film.g1, gh.H * sizeof(float), cudaMemcpyHostToDevice, st);
  cudaMemcpyAsync(d_e1, film.e1, gh.H * sizeof(float), cudaMemcpyHostToDevice, st);
  film_relu<<<(gh.N * gh.H + block - 1) / block, block, 0, st>>>(gh, buf.d_y1, d_g1, d_e1, buf.d_a1);
  cudaStreamSynchronize(st);
  launch_conv3_fwd(gh, streams, shards, buf.d_a1, gh.H, w + L.w2, w + L.b2, gh.H, buf.d_y2);
  sync_streams(shards, streams);
  cudaMemcpyAsync(d_g1, film.g2, gh.H * sizeof(float), cudaMemcpyHostToDevice, st);
  cudaMemcpyAsync(d_e1, film.e2, gh.H * sizeof(float), cudaMemcpyHostToDevice, st);
  film_relu<<<(gh.N * gh.H + block - 1) / block, block, 0, st>>>(gh, buf.d_y2, d_g1, d_e1, buf.d_a2);
  cudaStreamSynchronize(st);
  conv1_fwd<<<gridN, block, 0, st>>>(gh, buf.d_a2, gh.H, w + L.wo, w + L.bo, gh.O, buf.d_out);
  mid_fwd<<<gridN, block, 0, st>>>(gh, buf.d_S, buf.d_out, buf.d_mid);
  div_fwd<<<gridN, block, 0, st>>>(gh, buf.d_mid, buf.d_out, buf.d_div);
  for (int k = 0; k < gh.jacobi; ++k) jacobi_iter<<<gridN, block, 0, st>>>(gh, buf.d_div, buf.d_pressure, buf.d_tmp);
  cudaMemcpyAsync(buf.d_proj, buf.d_mid, gh.N * gh.C * sizeof(float), cudaMemcpyDeviceToDevice, st);
  subtract_grad<<<gridN, block, 0, st>>>(gh, buf.d_pressure, buf.d_proj);
  advect_fwd<<<gridN, block, 0, st>>>(gh, buf.d_proj, buf.d_next);
  cudaMemsetAsync(buf.d_clamped, 0, gh.N * kPhys, st);
  clamp_phys<<<gridN, block, 0, st>>>(gh, buf.d_next, buf.d_clamped);
  cudaStreamSynchronize(st);
}

void cuda_backward_through_conv1(const GpuHyper& gh, int shards, cudaStream_t* streams, const CudaBuffers& buf) {
  const cudaStream_t st = streams[0];
  const int block = 256;
  const int gridN = (gh.N + block - 1) / block;
  const GpuLayout& L = gh.L;
  const float* w = buf.d_w;
  zero_clamped_g<<<gridN, block, 0, st>>>(gh, buf.d_gS, buf.d_clamped);
  cudaMemsetAsync(buf.d_gproj, 0, gh.N * gh.C * sizeof(float), st);
  advect_bwd<<<gridN, block, 0, st>>>(gh, buf.d_proj, gh.C, buf.d_gS, buf.d_gproj);
  subtract_grad_bwd<<<gridN, block, 0, st>>>(gh, buf.d_gproj, gh.C, buf.d_gp);
  cudaMemcpyAsync(buf.d_tmp, buf.d_gp, gh.N * sizeof(float), cudaMemcpyDeviceToDevice, st);
  cudaMemsetAsync(buf.d_gdiv, 0, gh.N * sizeof(float), st);
  for (int k = 0; k < gh.jacobi; ++k) {
    jacobi_bwd_add<<<gridN, block, 0, st>>>(gh, buf.d_tmp, buf.d_gdiv);
    neighbour_sum<<<gridN, block, 0, st>>>(gh, buf.d_tmp, buf.d_gp);
    scale_w<<<gridN, block, 0, st>>>(gh, buf.d_gp);
    cudaMemcpyAsync(buf.d_tmp, buf.d_gp, gh.N * sizeof(float), cudaMemcpyDeviceToDevice, st);
  }
  cudaMemcpyAsync(buf.d_gp_in, buf.d_tmp, gh.N * sizeof(float), cudaMemcpyDeviceToDevice, st);
  cudaMemsetAsync(buf.d_gout, 0, gh.N * gh.O * sizeof(float), st);
  divergence_bwd<<<gridN, block, 0, st>>>(gh, gh.O, gh.qscale, buf.d_gdiv, buf.d_gproj, buf.d_gout);
  tanh_mid_bwd<<<gridN, block, 0, st>>>(gh, buf.d_mid, buf.d_gproj);
  mid_to_gout<<<gridN, block, 0, st>>>(gh, buf.d_gproj, buf.d_gout);
  cudaMemsetAsync(buf.d_ga2, 0, gh.N * gh.H * sizeof(float), st);
  cudaMemsetAsync(buf.d_gwo, 0, gh.H * gh.O * sizeof(float), st);
  cudaMemsetAsync(buf.d_gbo, 0, gh.O * sizeof(float), st);
  conv1_bwd<<<gridN, block, 0, st>>>(gh, buf.d_a2, gh.H, w + L.wo, gh.O, buf.d_gout, buf.d_ga2, buf.d_gwo, buf.d_gbo);
  cudaStreamSynchronize(st);
}

void cuda_backward_conv3_layer2(const GpuHyper& gh, int shards, cudaStream_t* streams, const CudaBuffers& buf) {
  const GpuLayout& L = gh.L;
  const float* w = buf.d_w;
  cudaMemset(buf.d_gw2, 0, 27 * gh.H * gh.H * sizeof(float));
  cudaMemset(buf.d_gb2, 0, gh.H * sizeof(float));
  cudaMemset(buf.d_gin, 0, gh.N * gh.H * sizeof(float));
  launch_conv3_bwd(gh, streams, shards, buf.d_a1, gh.H, w + L.w2, gh.H, buf.d_ga2, buf.d_gin, buf.d_gw2, buf.d_gb2);
  sync_streams(shards, streams);
}

void cuda_backward_conv3_layer1(const GpuHyper& gh, int shards, cudaStream_t* streams, const CudaBuffers& buf) {
  const GpuLayout& L = gh.L;
  const float* w = buf.d_w;
  cudaMemset(buf.d_gw1, 0, 27 * gh.I * gh.H * sizeof(float));
  cudaMemset(buf.d_gb1, 0, gh.H * sizeof(float));
  cudaMemset(buf.d_gX, 0, gh.N * gh.I * sizeof(float));
  launch_conv3_bwd(gh, streams, shards, buf.d_X, gh.I, w + L.w1, gh.H, buf.d_gin, buf.d_gX, buf.d_gw1, buf.d_gb1);
  sync_streams(shards, streams);
}

}  // namespace nfx::rollout3d::cuda_detail
