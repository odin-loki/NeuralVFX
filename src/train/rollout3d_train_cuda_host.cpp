#include <neuralfx/rollout3d_train.hpp>
#include <neuralfx/rollout3d_train_cuda.hpp>
#include <neuralfx/rollout3d_train_cuda_internal.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace nfx::rollout3d {
namespace {

#define CUDA_CHECK(x)                                                                                      \
  do {                                                                                                     \
    const cudaError_t e = (x);                                                                             \
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e));         \
  } while (0)

using cuda_detail::CudaBuffers;
using cuda_detail::FilmHost;
using cuda_detail::GpuHyper;
using cuda_detail::GpuLayout;

GpuHyper make_hyper(const Model& m) {
  const StepLayout L = step_layout(m.h);
  GpuHyper h{};
  h.rx = m.h.rx;
  h.ry = m.h.ry;
  h.rz = m.h.rz;
  h.N = m.h.cells();
  h.C = m.h.channels();
  h.I = m.h.inputs();
  h.H = m.h.hidden;
  h.O = m.h.outputs();
  h.jacobi = m.h.jacobi;
  h.cond = m.h.cond();
  h.edges = m.edges == Edges::ambient ? 1 : 0;
  h.qscale = m.qscale;
  for (int k = 0; k < kPhys; ++k) {
    h.scale[k] = m.scale[k];
    h.lo[k] = m.lo[k];
    h.hi[k] = m.hi[k];
  }
  h.L = {L.w1, L.b1, L.w2, L.b2, L.wo, L.bo, L.g1, L.e1, L.g2, L.e2, L.size};
  return h;
}

struct StepCacheGpu {
  FilmHost film{};
  std::vector<float> cond;
  std::vector<float> X, y1, a1, y2, a2, mid, proj;
  std::vector<unsigned char> clamped;
};

class CudaTrainer {
 public:
  explicit CudaTrainer(const Model& m, int shards) : gh_(make_hyper(m)), shards_(std::max(1, shards)) {
    CUDA_CHECK(cudaMalloc(&d_w_, m.step_w.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_w_, m.step_w.data(), m.step_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    streams_.resize(shards_);
    for (int s = 0; s < shards_; ++s) CUDA_CHECK(cudaStreamCreate(&streams_[s]));
    const int N = gh_.N, C = gh_.C, I = gh_.I, H = gh_.H, O = gh_.O;
    buf_.d_w = d_w_;
    alloc(&buf_.d_S, N * C);
    alloc(&buf_.d_pressure, N);
    alloc(&buf_.d_tmp, N);
    alloc(&buf_.d_noise, N * kNoise);
    alloc(&buf_.d_X, N * I);
    alloc(&buf_.d_y1, N * H);
    alloc(&buf_.d_a1, N * H);
    alloc(&buf_.d_y2, N * H);
    alloc(&buf_.d_a2, N * H);
    alloc(&buf_.d_out, N * O);
    alloc(&buf_.d_mid, N * C);
    alloc(&buf_.d_div, N);
    alloc(&buf_.d_proj, N * C);
    alloc(&buf_.d_next, N * C);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buf_.d_clamped), N * kPhys));
    alloc(&buf_.d_gS, N * C);
    alloc(&buf_.d_gproj, N * C);
    alloc(&buf_.d_gp, N);
    alloc(&buf_.d_gout, N * O);
    alloc(&buf_.d_ga2, N * H);
    alloc(&buf_.d_gin, N * H);
    alloc(&buf_.d_gX, N * I);
    alloc(&buf_.d_gw1, 27 * I * H);
    alloc(&buf_.d_gw2, 27 * H * H);
    alloc(&buf_.d_gb1, H);
    alloc(&buf_.d_gb2, H);
    alloc(&buf_.d_gwo, H * O);
    alloc(&buf_.d_gbo, O);
    alloc(&buf_.d_gdiv, N);
    alloc(&buf_.d_gp_in, N);
    alloc(&buf_.d_film_g, H);
    alloc(&buf_.d_film_e, H);
  }

  ~CudaTrainer() {
    cudaFree(d_w_);
    cudaFree(buf_.d_S);
    cudaFree(buf_.d_pressure);
    cudaFree(buf_.d_tmp);
    cudaFree(buf_.d_noise);
    cudaFree(buf_.d_X);
    cudaFree(buf_.d_y1);
    cudaFree(buf_.d_a1);
    cudaFree(buf_.d_y2);
    cudaFree(buf_.d_a2);
    cudaFree(buf_.d_out);
    cudaFree(buf_.d_mid);
    cudaFree(buf_.d_div);
    cudaFree(buf_.d_proj);
    cudaFree(buf_.d_next);
    cudaFree(buf_.d_clamped);
    cudaFree(buf_.d_gS);
    cudaFree(buf_.d_gproj);
    cudaFree(buf_.d_gp);
    cudaFree(buf_.d_gout);
    cudaFree(buf_.d_ga2);
    cudaFree(buf_.d_gin);
    cudaFree(buf_.d_gX);
    cudaFree(buf_.d_gw1);
    cudaFree(buf_.d_gw2);
    cudaFree(buf_.d_gb1);
    cudaFree(buf_.d_gb2);
    cudaFree(buf_.d_gwo);
    cudaFree(buf_.d_gbo);
    cudaFree(buf_.d_gdiv);
    cudaFree(buf_.d_gp_in);
    cudaFree(buf_.d_film_g);
    cudaFree(buf_.d_film_e);
    for (cudaStream_t st : streams_) cudaStreamDestroy(st);
  }

  void sync_weights(const Model& m) { CUDA_CHECK(cudaMemcpy(d_w_, m.step_w.data(), m.step_w.size() * sizeof(float), cudaMemcpyHostToDevice)); }

  void film_host(const Model& m, std::span<const float> cond, StepCacheGpu& c) {
    const int H = gh_.H;
    const GpuLayout& L = gh_.L;
    for (int j = 0; j < H; ++j) {
      c.film.g1[j] = 1.f;
      c.film.e1[j] = 0.f;
      c.film.g2[j] = 1.f;
      c.film.e2[j] = 0.f;
      for (int k = 0; k < gh_.cond; ++k) {
        const float v = cond[k];
        c.film.g1[j] += m.step_w[L.g1 + static_cast<std::size_t>(k) * H + j] * v;
        c.film.e1[j] += m.step_w[L.e1 + static_cast<std::size_t>(k) * H + j] * v;
        c.film.g2[j] += m.step_w[L.g2 + static_cast<std::size_t>(k) * H + j] * v;
        c.film.e2[j] += m.step_w[L.e2 + static_cast<std::size_t>(k) * H + j] * v;
      }
    }
    c.cond.assign(cond.begin(), cond.end());
  }

  void forward_step(const Model& m, std::vector<float>& S, const std::vector<float>& noise, std::span<const float> cond, std::vector<float>& pressure,
                    StepCacheGpu& c) {
    film_host(m, cond, c);
    CUDA_CHECK(cudaMemcpy(buf_.d_S, S.data(), gh_.N * gh_.C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf_.d_noise, noise.data(), gh_.N * kNoise * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf_.d_pressure, pressure.data(), gh_.N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_detail::cuda_forward_step(gh_, shards_, streams_.data(), buf_, c.film);
    const int N = gh_.N, C = gh_.C, I = gh_.I, H = gh_.H;
    c.X.resize(static_cast<std::size_t>(N) * I);
    c.y1.resize(static_cast<std::size_t>(N) * H);
    c.a1.resize(c.y1.size());
    c.y2.resize(c.y1.size());
    c.a2.resize(c.y1.size());
    c.mid.resize(static_cast<std::size_t>(N) * C);
    c.proj.resize(c.mid.size());
    c.clamped.resize(static_cast<std::size_t>(N) * kPhys);
    CUDA_CHECK(cudaMemcpy(c.X.data(), buf_.d_X, c.X.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c.y1.data(), buf_.d_y1, c.y1.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c.a1.data(), buf_.d_a1, c.a1.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c.y2.data(), buf_.d_y2, c.y2.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c.a2.data(), buf_.d_a2, c.a2.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c.mid.data(), buf_.d_mid, c.mid.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c.proj.data(), buf_.d_proj, c.proj.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c.clamped.data(), buf_.d_clamped, c.clamped.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pressure.data(), buf_.d_pressure, gh_.N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(S.data(), buf_.d_next, gh_.N * gh_.C * sizeof(float), cudaMemcpyDeviceToHost));
  }

  void film_back_host(const StepCacheGpu& c, const std::vector<float>& a, const std::vector<float>& y, const std::array<float, 128>& gam, std::vector<float>& ga,
                      std::size_t G, std::size_t E, std::vector<float>& grad_h) {
    const int N = gh_.N, H = gh_.H;
    std::vector<double> sg(H, 0.0), se(H, 0.0);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < H; ++j) {
        const std::size_t q = static_cast<std::size_t>(i) * H + j;
        const float gz = a[q] > 0.f ? ga[q] : 0.f;
        sg[j] += static_cast<double>(gz * y[q]);
        se[j] += static_cast<double>(gz);
        ga[q] = gz * gam[j];
      }
    }
    for (int k = 0; k < gh_.cond; ++k) {
      for (int j = 0; j < H; ++j) {
        grad_h[G + static_cast<std::size_t>(k) * H + j] += static_cast<float>(sg[j]) * c.cond[k];
        grad_h[E + static_cast<std::size_t>(k) * H + j] += static_cast<float>(se[j]) * c.cond[k];
      }
    }
  }

  void backward_step(const Model&, const StepCacheGpu& c, const std::vector<float>& gS_h, const std::vector<float>& gp_out_h, std::vector<float>& gS_in_h,
                     std::vector<float>& gp_in_h, std::vector<float>& grad_h) {
    CUDA_CHECK(cudaMemcpy(buf_.d_gS, gS_h.data(), gh_.N * gh_.C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf_.d_gp, gp_out_h.data(), gh_.N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf_.d_proj, c.proj.data(), c.proj.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf_.d_mid, c.mid.data(), c.mid.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf_.d_a2, c.a2.data(), c.a2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf_.d_clamped, c.clamped.data(), c.clamped.size(), cudaMemcpyHostToDevice));
    cuda_detail::cuda_backward_through_conv1(gh_, shards_, streams_.data(), buf_);
    gp_in_h.resize(gh_.N);
    CUDA_CHECK(cudaMemcpy(gp_in_h.data(), buf_.d_gp_in, gh_.N * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> ga2(gh_.N * gh_.H);
    CUDA_CHECK(cudaMemcpy(ga2.data(), buf_.d_ga2, ga2.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::array<float, 128> g2{};
    std::memcpy(g2.data(), c.film.g2, gh_.H * sizeof(float));
    film_back_host(c, c.a2, c.y2, g2, ga2, gh_.L.g2, gh_.L.e2, grad_h);
    CUDA_CHECK(cudaMemcpy(buf_.d_ga2, ga2.data(), ga2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf_.d_a1, c.a1.data(), c.a1.size() * sizeof(float), cudaMemcpyHostToDevice));
    cuda_detail::cuda_backward_conv3_layer2(gh_, shards_, streams_.data(), buf_);
    std::vector<float> ga1(gh_.N * gh_.H);
    CUDA_CHECK(cudaMemcpy(ga1.data(), buf_.d_gin, ga1.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::array<float, 128> g1{};
    std::memcpy(g1.data(), c.film.g1, gh_.H * sizeof(float));
    film_back_host(c, c.a1, c.y1, g1, ga1, gh_.L.g1, gh_.L.e1, grad_h);
    CUDA_CHECK(cudaMemcpy(buf_.d_gin, ga1.data(), ga1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf_.d_X, c.X.data(), c.X.size() * sizeof(float), cudaMemcpyHostToDevice));
    cuda_detail::cuda_backward_conv3_layer1(gh_, shards_, streams_.data(), buf_);
    gS_in_h.resize(gh_.N * gh_.C);
    CUDA_CHECK(cudaMemcpy(gS_in_h.data(), buf_.d_gproj, gS_in_h.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> gX(gh_.N * gh_.I);
    CUDA_CHECK(cudaMemcpy(gX.data(), buf_.d_gX, gX.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < gh_.N; ++i) {
      for (int k = 0; k < gh_.C; ++k) {
        gS_in_h[static_cast<std::size_t>(i) * gh_.C + k] += gX[static_cast<std::size_t>(i) * gh_.I + k] / (k < kPhys ? gh_.scale[k] : 1.f);
      }
    }
    const GpuLayout& L = gh_.L;
    auto add_block = [&](float* src, std::size_t off, std::size_t n) {
      std::vector<float> buf(n);
      CUDA_CHECK(cudaMemcpy(buf.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost));
      for (std::size_t j = 0; j < n; ++j) grad_h[off + j] += buf[j];
    };
    add_block(buf_.d_gw1, L.w1, 27 * gh_.I * gh_.H);
    add_block(buf_.d_gb1, L.b1, gh_.H);
    add_block(buf_.d_gw2, L.w2, 27 * gh_.H * gh_.H);
    add_block(buf_.d_gb2, L.b2, gh_.H);
    add_block(buf_.d_gwo, L.wo, gh_.H * gh_.O);
    add_block(buf_.d_gbo, L.bo, gh_.O);
  }

 private:
  static void alloc(float** p, std::size_t n) { CUDA_CHECK(cudaMalloc(p, n * sizeof(float))); }

  GpuHyper gh_;
  int shards_;
  float* d_w_ = nullptr;
  CudaBuffers buf_{};
  std::vector<cudaStream_t> streams_;
};

struct GridHost {
  int rx, ry, rz;
  std::size_t at(int x, int y, int z) const {
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(rz) + static_cast<std::size_t>(z)) * static_cast<std::size_t>(rx) + static_cast<std::size_t>(x);
  }
};

void state_of_host(const Model& m, const Run& r, int i, std::vector<float>& S, std::vector<float>& phys) {
  const int N = m.h.cells(), C = m.h.channels();
  phys.resize(static_cast<std::size_t>(N) * kPhys);
  r.state(i, static_cast<std::size_t>(N) * kPhys, phys);
  S.assign(static_cast<std::size_t>(N) * C, 0.f);
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < kPhys; ++k) S[static_cast<std::size_t>(j) * C + k] = phys[static_cast<std::size_t>(j) * kPhys + k];
  }
}

double window_loss_cuda_impl(CudaTrainer& tr, const Model& m, const Run& r, int first, int unroll, int burn, float sigma, float profile, std::uint64_t noise_seed,
                             std::vector<float>* grad, float activity, const ExtraLosses& extra) {
  const Hyper& h = m.h;
  const int N = h.cells(), C = h.channels();
  const std::size_t per = static_cast<std::size_t>(N) * kPhys;
  std::vector<float> S, phys, pressure(static_cast<std::size_t>(N), 0.f), noise(static_cast<std::size_t>(N) * kNoise);
  state_of_host(m, r, first, S, phys);
  Model noise_model = m;
  noise_model.h.fine = 1;
  Workspace ws(noise_model);
  const auto noise_at = [&](int state) { coarse_noise(noise_model, r.p.seed, r.time(state), ws, noise); };
  int i0 = first;
  if (burn > 0) {
    StepCacheGpu cb;
    for (int s = 0; s < burn; ++s) {
      noise_at(i0 + s);
      tr.forward_step(m, S, noise, r.controls_at(i0 + s + 1), pressure, cb);
    }
    i0 += burn;
  }
  std::vector<StepCacheGpu> cs(static_cast<std::size_t>(unroll));
  std::vector<std::vector<float>> gout(static_cast<std::size_t>(unroll)), gprev(static_cast<std::size_t>(unroll)), outs(static_cast<std::size_t>(unroll));
  for (int s = 0; s < unroll; ++s) gprev[static_cast<std::size_t>(s)].assign(activity > 0.f ? static_cast<std::size_t>(N) * C : 0, 0.f);
  std::array<float, kPhys> wch{};
  for (int k = 0; k < kPhys; ++k) wch[k] = 1.f / (m.scale[k] * m.scale[k]);
  const float inv = 1.f / (static_cast<float>(N) * kPhys * unroll);
  std::mt19937 rng(static_cast<unsigned>(noise_seed));
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> t(per), tp(per);
  double loss = 0;
  const GridHost g{h.rx, h.ry, h.rz};
  for (int s = 0; s < unroll; ++s) {
    noise_at(i0 + s);
    if (sigma > 0.f) {
      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < kPhys; ++k) {
          float& x = S[static_cast<std::size_t>(j) * C + k];
          x = std::clamp(x + sigma * m.scale[k] * nd(rng), m.lo[k], m.hi[k]);
        }
      }
    }
    tr.forward_step(m, S, noise, r.controls_at(i0 + s + 1), pressure, cs[static_cast<std::size_t>(s)]);
    if (activity > 0.f) outs[static_cast<std::size_t>(s)] = S;
    r.state(i0 + s + 1, per, t);
    auto& gs = gout[static_cast<std::size_t>(s)];
    gs.assign(static_cast<std::size_t>(N) * C, 0.f);
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < kPhys; ++k) {
        const float e = S[static_cast<std::size_t>(j) * C + k] - t[static_cast<std::size_t>(j) * kPhys + k];
        loss += static_cast<double>(wch[k] * e * e * inv);
        gs[static_cast<std::size_t>(j) * C + k] = 2.f * wch[k] * e * inv;
      }
    }
    if (profile > 0.f) {
      const float pinv = profile / static_cast<float>(2 * (g.rx + g.ry + g.rz) * unroll);
      for (int k = 3; k < kPhys; ++k) {
        for (int axis = 0; axis < 3; ++axis) {
          const int bins = axis == 0 ? g.rx : (axis == 1 ? g.ry : g.rz);
          const float per_slice = static_cast<float>(N / bins) * m.scale[k];
          std::vector<float> dm(static_cast<std::size_t>(bins), 0.f);
          for (int y = 0; y < g.ry; ++y) {
            for (int z = 0; z < g.rz; ++z) {
              for (int x = 0; x < g.rx; ++x) {
                const std::size_t j = g.at(x, y, z);
                dm[static_cast<std::size_t>(axis == 0 ? x : (axis == 1 ? y : z))] += (S[j * C + k] - t[j * kPhys + k]) / per_slice;
              }
            }
          }
          for (int b = 0; b < bins; ++b) loss += static_cast<double>(pinv * dm[b] * dm[b]);
          for (int y = 0; y < g.ry; ++y) {
            for (int z = 0; z < g.rz; ++z) {
              for (int x = 0; x < g.rx; ++x) {
                const std::size_t j = g.at(x, y, z);
                gs[j * C + k] += 2.f * pinv * dm[static_cast<std::size_t>(axis == 0 ? x : (axis == 1 ? y : z))] / per_slice;
              }
            }
          }
        }
      }
    }
    if (activity > 0.f && s > 0) {
      r.state(i0 + s, per, tp);
      const auto& prev = outs[static_cast<std::size_t>(s - 1)];
      for (int k = 0; k < kPhys; ++k) {
        const float is2 = wch[k];
        double am = 0, at = 0;
        for (int j = 0; j < N; ++j) {
          const float dmod = S[static_cast<std::size_t>(j) * C + k] - prev[static_cast<std::size_t>(j) * C + k];
          const float dtru = t[static_cast<std::size_t>(j) * kPhys + k] - tp[static_cast<std::size_t>(j) * kPhys + k];
          am += static_cast<double>(dmod * dmod * is2);
          at += static_cast<double>(dtru * dtru * is2);
        }
        am /= N;
        at /= N;
        const float e = static_cast<float>(am - at), wgt = activity / (static_cast<float>(kPhys) * static_cast<float>(unroll - 1));
        loss += static_cast<double>(wgt * e * e);
        for (int j = 0; j < N; ++j) {
          const float dmod = S[static_cast<std::size_t>(j) * C + k] - prev[static_cast<std::size_t>(j) * C + k];
          const float gj = wgt * 2.f * e * 2.f * dmod * is2 / static_cast<float>(N);
          gs[static_cast<std::size_t>(j) * C + k] += gj;
          gprev[static_cast<std::size_t>(s - 1)][static_cast<std::size_t>(j) * C + k] -= gj;
        }
      }
    }
    r2_extras_step(m, unroll, S, t, extra, loss, gs);
  }
  if (grad) {
    grad->assign(step_layout(h).size, 0.f);
    std::vector<float> gS(static_cast<std::size_t>(N) * C, 0.f), gp(static_cast<std::size_t>(N), 0.f), gS_in, gp_in;
    for (int s = unroll - 1; s >= 0; --s) {
      for (std::size_t j = 0; j < gS.size(); ++j) gS[j] += gout[static_cast<std::size_t>(s)][j];
      if (activity > 0.f) {
        for (std::size_t j = 0; j < gS.size(); ++j) gS[j] += gprev[static_cast<std::size_t>(s)][j];
      }
      tr.backward_step(m, cs[static_cast<std::size_t>(s)], gS, gp, gS_in, gp_in, *grad);
      gS.swap(gS_in);
      gp.swap(gp_in);
    }
  }
  return loss;
}

}  // namespace

bool cuda_train_available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

double window_loss_cuda(const Model& m, const Run& r, int first, int unroll, int burn, float sigma, float profile, std::uint64_t noise_seed,
                        std::vector<float>* grad, float activity, const ExtraLosses& extra, int shards) {
  CudaTrainer tr(m, shards);
  tr.sync_weights(m);
  return window_loss_cuda_impl(tr, m, r, first, unroll, burn, sigma, profile, noise_seed, grad, activity, extra);
}

void cuda_train_stepper(Model& m, std::span<const Run> runs, const TrainOptions& o, TrainState& s,
                        const std::function<void(const Model&, const TrainState&)>& checkpoint) {
  if (runs.empty()) throw std::invalid_argument("rollout3d: no runs");
  const StepLayout L = step_layout(m.h);
  if (m.step_w.size() != L.size) throw std::invalid_argument("rollout3d: weights do not match the hyperparameters");
  const std::size_t per = static_cast<std::size_t>(m.h.cells()) * kPhys;
  for (const Run& r : runs) {
    if (r.coarse.size() != static_cast<std::size_t>(r.frames) * per || r.frames < o.max_unroll + o.burn_max + 2)
      throw std::invalid_argument("rollout3d: a run has the wrong shape");
  }
  const auto table = [&](int min_frames) {
    std::vector<std::uint64_t> cum;
    std::uint64_t t = 0;
    for (const Run& r : runs) {
      t += r.frames >= min_frames ? static_cast<std::uint64_t>(r.frames) : 0;
      cum.push_back(t);
    }
    return cum;
  };
  const std::vector<std::uint64_t> cum_all = table(0), cum_long = table(o.burn_long_min + o.max_unroll + 2);
  const bool one_length = std::ranges::all_of(runs, [&](const Run& r) { return r.frames == runs[0].frames; });
  const auto pick = [&](const std::vector<std::uint64_t>& cum, std::uint64_t u) {
    if (one_length) return static_cast<std::size_t>(u % runs.size());
    const std::uint64_t k = u % cum.back();
    return static_cast<std::size_t>(std::ranges::upper_bound(cum, k) - cum.begin());
  };
  if (s.m1.size() != L.size) {
    s.m1.assign(L.size, 0.f);
    s.m2.assign(L.size, 0.f);
  }
  CudaTrainer tr(m, o.shards);
  tr.sync_weights(m);
  const int total = stage_end(o, 4);
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> gsum(L.size, 0.f);
  for (int it = s.iteration; it < total; ++it) {
    const int stage = train_stage(o, it);
    const bool fine = stage >= 2, act = stage >= 3, lng = stage == 4;
    const int unroll = fine ? o.max_unroll : std::min(o.max_unroll, 1 + it * o.max_unroll / std::max(1, o.iterations / 2));
    double loss = 0;
    std::ranges::fill(gsum, 0.f);
    for (int b = 0; b < o.batch; ++b) {
      std::mt19937_64 rng(o.seed * 1000003ULL + static_cast<std::uint64_t>(it) * 977ULL + static_cast<std::uint64_t>(b) * 7919ULL);
      const Run& r = runs[pick(lng ? cum_long : cum_all, rng())];
      int burn = 0;
      if (lng) {
        const int hi = std::min(o.burn_long_max, r.frames - unroll - 2);
        burn = o.burn_long_min + static_cast<int>(rng() % static_cast<std::uint64_t>(std::max(1, hi - o.burn_long_min + 1)));
      } else if (fine && o.burn_max > 0 && (b & 1)) {
        burn = static_cast<int>(rng() % static_cast<std::uint64_t>(o.burn_max + 1));
      }
      const int span = r.frames - unroll - burn - 1;
      const int first = static_cast<int>(rng() % static_cast<std::uint64_t>(std::max(1, span)));
      std::vector<float> grad(L.size, 0.f);
      const ExtraLosses ex{fine ? o.spread : 0.f, fine ? o.edge : 0.f};
      loss += window_loss_cuda_impl(tr, m, r, first, unroll, burn, o.sigma, fine ? o.profile : 0.f, rng(), &grad, act ? o.activity : 0.f, ex);
      for (std::size_t j = 0; j < L.size; ++j) gsum[j] += grad[j];
    }
    loss /= o.batch;
    double gn = 0;
    for (float& v : gsum) {
      v /= static_cast<float>(o.batch);
      gn += static_cast<double>(v) * v;
    }
    gn = std::sqrt(gn);
    if (gn > o.clip) {
      const float k = static_cast<float>(o.clip / gn);
      for (float& v : gsum) v *= k;
    }
    const int stage_first = stage == 1 ? 0 : stage_end(o, stage - 1);
    const int stage_it = it - stage_first, stage_n = stage_end(o, stage) - stage_first;
    const float base = fine ? o.lr_finetune : o.lr;
    const float lr = base * (0.05f + 0.95f * 0.5f * (1.f + std::cos(3.14159265f * static_cast<float>(stage_it) / static_cast<float>(std::max(1, stage_n)))));
    ++s.adam_t;
    constexpr float b1 = 0.9f, b2 = 0.999f;
    const float c1 = 1.f - std::pow(b1, static_cast<float>(s.adam_t)), c2 = 1.f - std::pow(b2, static_cast<float>(s.adam_t));
    for (std::size_t j = 0; j < L.size; ++j) {
      s.m1[j] = b1 * s.m1[j] + (1.f - b1) * gsum[j];
      s.m2[j] = b2 * s.m2[j] + (1.f - b2) * gsum[j] * gsum[j];
      m.step_w[j] -= lr * (s.m1[j] / c1) / (std::sqrt(s.m2[j] / c2) + 1e-8f);
    }
    tr.sync_weights(m);
    s.iteration = it + 1;
    if (it % std::max(1, o.log_every) == 0 || it == total - 1) {
      s.curve.emplace_back(it, loss);
      if (o.progress) o.progress(it, unroll, loss, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    bool at_stage_end = false;
    for (int k = 1; k <= 4; ++k) at_stage_end = at_stage_end || s.iteration == stage_end(o, k);
    const bool stop = o.stop_at > 0 && s.iteration >= o.stop_at;
    if (checkpoint && ((o.checkpoint_every > 0 && s.iteration % o.checkpoint_every == 0) || at_stage_end || stop)) checkpoint(m, s);
    if (stop) break;
  }
}

}  // namespace nfx::rollout3d
