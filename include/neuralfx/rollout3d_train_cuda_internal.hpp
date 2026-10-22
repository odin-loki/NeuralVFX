#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace nfx::rollout3d::cuda_detail {

inline constexpr int kPhys = 5;
inline constexpr int kNoise = 4;

struct GpuLayout {
  std::size_t w1, b1, w2, b2, wo, bo, g1, e1, g2, e2, size;
};

struct GpuHyper {
  int rx, ry, rz, N, C, I, H, O, jacobi, cond, edges;
  float scale[5], lo[5], hi[5], qscale;
  GpuLayout L;
};

void sync_streams(int shards, cudaStream_t* streams);
void launch_conv3_fwd(const GpuHyper& gh, cudaStream_t* streams, int shards, const float* in, int ci, const float* W, const float* b, int co, float* out);
void launch_conv3_bwd(const GpuHyper& gh, cudaStream_t* streams, int shards, const float* in, int ci, const float* W, int co, const float* go, float* gin,
                      float* gW, float* gb);

struct CudaBuffers {
  float* d_S = nullptr;
  float* d_noise = nullptr;
  float* d_pressure = nullptr;
  float* d_tmp = nullptr;
  float* d_w = nullptr;
  float* d_X = nullptr;
  float* d_y1 = nullptr;
  float* d_a1 = nullptr;
  float* d_y2 = nullptr;
  float* d_a2 = nullptr;
  float* d_out = nullptr;
  float* d_mid = nullptr;
  float* d_div = nullptr;
  float* d_proj = nullptr;
  float* d_next = nullptr;
  unsigned char* d_clamped = nullptr;
  float* d_gS = nullptr;
  float* d_gproj = nullptr;
  float* d_gp = nullptr;
  float* d_gout = nullptr;
  float* d_ga2 = nullptr;
  float* d_gin = nullptr;
  float* d_gX = nullptr;
  float* d_gw1 = nullptr;
  float* d_gw2 = nullptr;
  float* d_gb1 = nullptr;
  float* d_gb2 = nullptr;
  float* d_gwo = nullptr;
  float* d_gbo = nullptr;
  float* d_gdiv = nullptr;
  float* d_gp_in = nullptr;
  float* d_film_g = nullptr;  // H coeffs for film_relu (reused each step)
  float* d_film_e = nullptr;
};

struct FilmHost {
  float g1[128], e1[128], g2[128], e2[128];
};

void cuda_forward_step(const GpuHyper& gh, int shards, cudaStream_t* streams, const CudaBuffers& b, const FilmHost& film);
void cuda_backward_through_conv1(const GpuHyper& gh, int shards, cudaStream_t* streams, const CudaBuffers& b);
void cuda_backward_conv3_layer2(const GpuHyper& gh, int shards, cudaStream_t* streams, const CudaBuffers& b);
void cuda_backward_conv3_layer1(const GpuHyper& gh, int shards, cudaStream_t* streams, const CudaBuffers& b);

}  // namespace nfx::rollout3d::cuda_detail
