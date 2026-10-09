// The hand-written denoiser on coarse states (include/neuralfx/dcm/ddpm.hpp). tests/test_dcm_ddpm.cpp checks the
// gradients against finite differences, the fast kernels against the plain patterns, that training lowers the loss,
// and that a short training gives one SHA-256 over repeats.
#include <neuralfx/binio.hpp>
#include <neuralfx/dcm/ddpm.hpp>
#include <neuralfx/dcm/mixer.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

// AVX2 + FMA for the kernels, switched on after the standard headers (the trainer's requirement, as in
// src/train/rollout_train.cpp; the tools check the CPU before training).
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#pragma GCC push_options
#pragma GCC target("arch=x86-64-v3")
#define NFX_DDPM_PUSHED 1
#endif

#include "ddpm_kernels.hpp"

namespace nfx::dcm::ddpm {

namespace {

using Vec = std::vector<float>;
using kernels::sz;
float fl(int v) { return static_cast<float>(v); }

// --- random numbers ----------------------------------------------------------------------------------------------

std::uint64_t splitmix(std::uint64_t& s) {
  s += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
std::uint64_t mix(std::uint64_t a, std::uint64_t b) {
  std::uint64_t s = a * 0xD1B54A32D192ED03ULL + b;
  return splitmix(s);
}
std::uint64_t mix(std::uint64_t a, std::uint64_t b, std::uint64_t c) { return mix(mix(a, b), c); }

// --- pooling and upsampling on [cell][channel] --------------------------------------------------------------------

// R x R -> R/2 x R/2, mean of each 2 x 2 block.
void avgpool(int R, const float* in, int C, float* out) {
  const int r = R / 2;
  for (int y = 0; y < r; ++y) {
    for (int x = 0; x < r; ++x) {
      float* o = out + (sz(y) * sz(r) + sz(x)) * sz(C);
      const float* a = in + (sz(2 * y) * sz(R) + sz(2 * x)) * sz(C);
      const float* b = a + C;
      const float* c = a + sz(R) * sz(C);
      const float* d = c + C;
      for (int k = 0; k < C; ++k) o[k] = 0.25f * (a[k] + b[k] + c[k] + d[k]);
    }
  }
}
// The transpose of avgpool, added: each cell of the R x R grid gets a quarter of its block's gradient.
void avgpool_back_add(int R, const float* g, int C, float* gin) {
  const int r = R / 2;
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const float* s = g + (sz(y / 2) * sz(r) + sz(x / 2)) * sz(C);
      float* o = gin + (sz(y) * sz(R) + sz(x)) * sz(C);
      for (int k = 0; k < C; ++k) o[k] += 0.25f * s[k];
    }
  }
}
// r x r -> 2r x 2r nearest, added to out (the skip connection is already there).
void upsample_add(int r, const float* in, int C, float* out) {
  const int R = 2 * r;
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const float* s = in + (sz(y / 2) * sz(r) + sz(x / 2)) * sz(C);
      float* o = out + (sz(y) * sz(R) + sz(x)) * sz(C);
      for (int k = 0; k < C; ++k) o[k] += s[k];
    }
  }
}
// The transpose of nearest upsampling: the sum over each 2 x 2 block.
void upsample_back(int R, const float* g, int C, float* out) {
  const int r = R / 2;
  std::fill_n(out, sz(r) * sz(r) * sz(C), 0.f);
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const float* s = g + (sz(y) * sz(R) + sz(x)) * sz(C);
      float* o = out + (sz(y / 2) * sz(r) + sz(x / 2)) * sz(C);
      for (int k = 0; k < C; ++k) o[k] += s[k];
    }
  }
}

// --- the network ----------------------------------------------------------------------------------------------------

struct BlockCache {
  Vec in, apad, h, bpad;  // input; padded SiLU(input); first convolution before FiLM; padded SiLU(FiLM(h))
};

// Everything one forward pass keeps for the backward pass, and the backward pass's scratch. One per thread.
struct Work {
  Config c;
  int R0 = 0, R1 = 0, R2 = 0;
  Vec emb, mh, ma, film;          // MLP input, hidden before and after SiLU, output (FiLM parameters)
  Vec xin, xpad;                  // input with position channels, and padded
  std::array<BlockCache, kBlocks> b;
  Vec s0, s1, pool0, pool1, d1in, d2in, mm, mout, u2c, u1in, dec1, u1c, u0in, dec0, act, opad, y;
  // backward scratch
  Vec gpad, gtmp, gtmp2, gfilm, gma, gmh;
  Vec g0a, g0b, g1a, g1b, g2a, g2b, gp0, gp1;

  void prepare(const Config& cfg) {
    if (R0 == cfg.res && c.c0 == cfg.c0 && c.c1 == cfg.c1 && c.c2 == cfg.c2 && c.channels == cfg.channels && c.cond == cfg.cond &&
        c.freqs == cfg.freqs && c.film_hidden == cfg.film_hidden && !y.empty()) {
      return;
    }
    c = cfg;
    R0 = cfg.res;
    R1 = R0 / 2;
    R2 = R0 / 4;
    const auto N = [](int R) { return sz(R) * sz(R); };
    const auto P = [](int R) { return sz(R + 2) * sz(R + 2); };
    const int widths[kBlocks] = {cfg.c0, cfg.c1, cfg.c2, cfg.c2, cfg.c1, cfg.c0};
    const int sides[kBlocks] = {R0, R1, R2, R2, R1, R0};
    for (int k = 0; k < kBlocks; ++k) {
      BlockCache& bc = b[sz(k)];
      bc.in.resize(N(sides[k]) * sz(widths[k]));
      bc.apad.resize(P(sides[k]) * sz(widths[k]));
      bc.h.resize(N(sides[k]) * sz(widths[k]));
      bc.bpad.resize(P(sides[k]) * sz(widths[k]));
    }
    xin.resize(N(R0) * sz(cfg.inputs()));
    xpad.resize(P(R0) * sz(cfg.inputs()));
    s0.resize(N(R0) * sz(cfg.c0));
    s1.resize(N(R1) * sz(cfg.c1));
    pool0.resize(N(R1) * sz(cfg.c0));
    pool1.resize(N(R2) * sz(cfg.c1));
    d1in.resize(N(R1) * sz(cfg.c1));
    d2in.resize(N(R2) * sz(cfg.c2));
    mm.resize(N(R2) * sz(cfg.c2));
    mout.resize(N(R2) * sz(cfg.c2));
    u2c.resize(N(R2) * sz(cfg.c1));
    u1in.resize(N(R1) * sz(cfg.c1));
    dec1.resize(N(R1) * sz(cfg.c1));
    u1c.resize(N(R1) * sz(cfg.c0));
    u0in.resize(N(R0) * sz(cfg.c0));
    dec0.resize(N(R0) * sz(cfg.c0));
    const int cmax = std::max({cfg.c0, cfg.c1, cfg.c2, cfg.inputs()});
    act.resize(N(R0) * sz(cmax));
    opad.resize(P(R0) * sz(cfg.c0));
    y.resize(N(R0) * sz(cfg.channels));
    gpad.resize(P(R0) * sz(cmax));
    gtmp.resize(N(R0) * sz(cmax));
    gtmp2.resize(N(R0) * sz(cmax));
    for (Vec* v : {&g0a, &g0b, &gp0}) v->resize(N(R0) * sz(cmax));
    for (Vec* v : {&g1a, &g1b, &gp1}) v->resize(N(R1) * sz(cmax));
    for (Vec* v : {&g2a, &g2b}) v->resize(N(R2) * sz(cmax));
  }
};

Work& thread_work(const Config& c) {
  thread_local Work w;
  w.prepare(c);
  return w;
}

// The 3 x 3 weights of the input gradients (kernels::flip_transpose of every convolution that passes a gradient on).
struct Transposed {
  std::array<Vec, kBlocks> wa, wb;
  Vec out;
};
Transposed transpose_weights(const Config& c, const Layout& L, const float* w) {
  Transposed T;
  for (int k = 0; k < kBlocks; ++k) {
    const BlockLayout& B = L.blocks[sz(k)];
    const int C = B.width;
    T.wa[sz(k)].resize(9 * sz(C) * sz(C));
    T.wb[sz(k)].resize(9 * sz(C) * sz(C));
    kernels::flip_transpose(w + B.wa, C, C, T.wa[sz(k)].data());
    kernels::flip_transpose(w + B.wb, C, C, T.wb[sz(k)].data());
  }
  T.out.resize(9 * sz(c.c0) * sz(c.channels));
  kernels::flip_transpose(w + L.out_w, c.c0, c.channels, T.out.data());
  return T;
}

void block_fwd(const float* w, const BlockLayout& B, const float* film, int R, BlockCache& bc, const float* in, float* out, Vec& act) {
  const int C = B.width;
  const std::size_t n = sz(R) * sz(R) * sz(C);
  std::copy_n(in, n, bc.in.data());
  kernels::silu_n(in, act.data(), n);
  kernels::pad(R, act.data(), C, bc.apad.data());
  kernels::conv3_fwd(R, bc.apad.data(), C, w + B.wa, w + B.ba, C, bc.h.data());
  const float* gamma = film + B.film;
  const float* beta = gamma + C;
  for (std::size_t i = 0; i < n; i += sz(C)) {
    for (int k = 0; k < C; ++k) act[i + sz(k)] = bc.h[i + sz(k)] * (1.f + gamma[k]) + beta[k];
  }
  kernels::silu_n(act.data(), act.data(), n);
  kernels::pad(R, act.data(), C, bc.bpad.data());
  kernels::conv3_fwd(R, bc.bpad.data(), C, w + B.wb, w + B.bb, C, out);
  for (std::size_t i = 0; i < n; ++i) out[i] += in[i];
}

// gin (written) = d loss / d input, from gout; weight gradients added to gw, FiLM gradients to gfilm.
void block_bwd(const float* w, const BlockLayout& B, const float* film, const float* wa_t, const float* wb_t, int R, const BlockCache& bc,
               const float* gout, float* gin, float* gw, float* gfilm, Work& k) {
  const int C = B.width;
  const std::size_t n = sz(R) * sz(R) * sz(C);
  float* gb = k.gtmp.data();
  float* gh = k.gtmp2.data();
  kernels::conv3_wgrad(R, bc.bpad.data(), C, gout, C, gw + B.wb, gw + B.bb);
  kernels::pad(R, gout, C, k.gpad.data());
  kernels::conv3_fwd(R, k.gpad.data(), C, wb_t, nullptr, C, gb);
  const float* gamma = film + B.film;
  const float* beta = gamma + C;
  float* ggamma = gfilm + B.film;
  float* gbeta = ggamma + C;
  float* hf = k.act.data();
  for (std::size_t i = 0; i < n; i += sz(C)) {
    for (int c = 0; c < C; ++c) hf[i + sz(c)] = bc.h[i + sz(c)] * (1.f + gamma[c]) + beta[c];
  }
  kernels::silu_grad_mul(hf, gb, n);
  for (std::size_t i = 0; i < n; i += sz(C)) {
    for (int c = 0; c < C; ++c) {
      const float g = gb[i + sz(c)];
      ggamma[c] += g * bc.h[i + sz(c)];
      gbeta[c] += g;
      gh[i + sz(c)] = g * (1.f + gamma[c]);
    }
  }
  kernels::conv3_wgrad(R, bc.apad.data(), C, gh, C, gw + B.wa, gw + B.ba);
  kernels::pad(R, gh, C, k.gpad.data());
  kernels::conv3_fwd(R, k.gpad.data(), C, wa_t, nullptr, C, gb);
  kernels::silu_grad_mul(bc.in.data(), gb, n);
  for (std::size_t i = 0; i < n; ++i) gin[i] = gout[i] + gb[i];
  (void)w;
}

// The forward pass: k.y = eps_hat(x, t). Every intermediate stays in k for the backward pass.
void forward(const Config& c, const Layout& L, const float* w, const float* x, int t, std::span<const float> cond, Work& k) {
  const int F = c.freqs, H = c.film_hidden, E = c.embed();
  // embedding of t and the condition, then the MLP to the FiLM parameters
  k.emb.resize(sz(E));
  for (int j = 0; j < F; ++j) {
    const double f = std::exp(-std::log(10000.0) * static_cast<double>(j) / static_cast<double>(F));
    const double a = static_cast<double>(t) * f;
    k.emb[sz(j)] = static_cast<float>(std::sin(a));
    k.emb[sz(F + j)] = static_cast<float>(std::cos(a));
  }
  for (int j = 0; j < c.cond; ++j) k.emb[sz(2 * F + j)] = sz(j) < cond.size() ? cond[sz(j)] : 0.f;
  k.mh.resize(sz(H));
  k.ma.resize(sz(H));
  kernels::conv1_fwd(1, k.emb.data(), E, w + L.mlp1_w, w + L.mlp1_b, H, k.mh.data());
  kernels::silu_n(k.mh.data(), k.ma.data(), sz(H));
  const int FS = static_cast<int>(L.film_size);
  k.film.resize(L.film_size);
  kernels::conv1_fwd(1, k.ma.data(), H, w + L.mlp2_w, w + L.mlp2_b, FS, k.film.data());
  const float* film = k.film.data();
  // input with position channels
  const int R0 = k.R0, R1 = k.R1, R2 = k.R2, I = c.inputs(), ch = c.channels;
  for (int yy = 0; yy < R0; ++yy) {
    for (int xx = 0; xx < R0; ++xx) {
      const std::size_t i = sz(yy) * sz(R0) + sz(xx);
      float* o = k.xin.data() + i * sz(I);
      for (int q = 0; q < ch; ++q) o[q] = x[i * sz(ch) + sz(q)];
      o[ch] = (fl(xx) + 0.5f) / fl(R0) * 2.f - 1.f;
      o[ch + 1] = (fl(yy) + 0.5f) / fl(R0) * 2.f - 1.f;
    }
  }
  kernels::pad(R0, k.xin.data(), I, k.xpad.data());
  // encoder
  kernels::conv3_fwd(R0, k.xpad.data(), I, w + L.stem_w, w + L.stem_b, c.c0, k.gtmp.data());  // the stem
  block_fwd(w, L.blocks[0], film, R0, k.b[0], k.gtmp.data(), k.s0.data(), k.act);
  avgpool(R0, k.s0.data(), c.c0, k.pool0.data());
  kernels::conv1_fwd(R1 * R1, k.pool0.data(), c.c0, w + L.down1_w, w + L.down1_b, c.c1, k.d1in.data());
  block_fwd(w, L.blocks[1], film, R1, k.b[1], k.d1in.data(), k.s1.data(), k.act);
  avgpool(R1, k.s1.data(), c.c1, k.pool1.data());
  kernels::conv1_fwd(R2 * R2, k.pool1.data(), c.c1, w + L.down2_w, w + L.down2_b, c.c2, k.d2in.data());
  block_fwd(w, L.blocks[2], film, R2, k.b[2], k.d2in.data(), k.mm.data(), k.act);
  block_fwd(w, L.blocks[3], film, R2, k.b[3], k.mm.data(), k.mout.data(), k.act);
  // decoder: 1 x 1 convolution at the low resolution (it commutes with nearest upsampling), upsample, add the skip
  kernels::conv1_fwd(R2 * R2, k.mout.data(), c.c2, w + L.up2_w, w + L.up2_b, c.c1, k.u2c.data());
  std::copy(k.s1.begin(), k.s1.end(), k.u1in.begin());
  upsample_add(R2, k.u2c.data(), c.c1, k.u1in.data());
  block_fwd(w, L.blocks[4], film, R1, k.b[4], k.u1in.data(), k.dec1.data(), k.act);
  kernels::conv1_fwd(R1 * R1, k.dec1.data(), c.c1, w + L.up1_w, w + L.up1_b, c.c0, k.u1c.data());
  std::copy(k.s0.begin(), k.s0.end(), k.u0in.begin());
  upsample_add(R1, k.u1c.data(), c.c0, k.u0in.data());
  block_fwd(w, L.blocks[5], film, R0, k.b[5], k.u0in.data(), k.dec0.data(), k.act);
  // output
  const std::size_t n0 = sz(R0) * sz(R0) * sz(c.c0);
  kernels::silu_n(k.dec0.data(), k.act.data(), n0);
  kernels::pad(R0, k.act.data(), c.c0, k.opad.data());
  kernels::conv3_fwd(R0, k.opad.data(), c.c0, w + L.out_w, w + L.out_b, ch, k.y.data());
}

// The backward pass from gy (d loss / d eps_hat, R0^2 x channels), adding the weight gradient to gw.
void backward(const Config& c, const Layout& L, const float* w, const Transposed& T, Work& k, const float* gy, float* gw) {
  const int R0 = k.R0, R1 = k.R1, R2 = k.R2, ch = c.channels;
  k.gfilm.assign(L.film_size, 0.f);
  float* gfilm = k.gfilm.data();
  const float* film = k.film.data();
  // output convolution, then SiLU
  kernels::conv3_wgrad(R0, k.opad.data(), c.c0, gy, ch, gw + L.out_w, gw + L.out_b);
  kernels::pad(R0, gy, ch, k.gpad.data());
  float* g_dec0 = k.g0a.data();
  kernels::conv3_fwd(R0, k.gpad.data(), ch, T.out.data(), nullptr, c.c0, g_dec0);
  const std::size_t n0 = sz(R0) * sz(R0) * sz(c.c0);
  kernels::silu_grad_mul(k.dec0.data(), g_dec0, n0);
  // block d0 -> gradient of u0in = up(u1c) + s0
  float* g_u0in = k.g0b.data();
  block_bwd(w, L.blocks[5], film, T.wa[5].data(), T.wb[5].data(), R0, k.b[5], g_dec0, g_u0in, gw, gfilm, k);
  float* g_s0 = k.gp0.data();
  std::copy_n(g_u0in, n0, g_s0);
  float* g_u1c = k.g1a.data();
  upsample_back(R0, g_u0in, c.c0, g_u1c);
  float* g_dec1 = k.g1b.data();
  const std::size_t n1 = sz(R1) * sz(R1) * sz(c.c1);
  std::fill_n(g_dec1, n1, 0.f);
  kernels::conv1_back(R1 * R1, k.dec1.data(), c.c1, w + L.up1_w, c.c0, g_u1c, g_dec1, gw + L.up1_w, gw + L.up1_b);
  // block d1 -> gradient of u1in = up(u2c) + s1
  float* g_u1in = k.g1a.data();
  block_bwd(w, L.blocks[4], film, T.wa[4].data(), T.wb[4].data(), R1, k.b[4], g_dec1, g_u1in, gw, gfilm, k);
  float* g_s1 = k.gp1.data();
  std::copy_n(g_u1in, n1, g_s1);
  float* g_u2c = k.g2a.data();
  upsample_back(R1, g_u1in, c.c1, g_u2c);
  float* g_mout = k.g2b.data();
  const std::size_t n2 = sz(R2) * sz(R2) * sz(c.c2);
  std::fill_n(g_mout, n2, 0.f);
  kernels::conv1_back(R2 * R2, k.mout.data(), c.c2, w + L.up2_w, c.c1, g_u2c, g_mout, gw + L.up2_w, gw + L.up2_b);
  // blocks m1, m0
  float* g_mm = k.g2a.data();
  block_bwd(w, L.blocks[3], film, T.wa[3].data(), T.wb[3].data(), R2, k.b[3], g_mout, g_mm, gw, gfilm, k);
  float* g_d2in = k.g2b.data();
  block_bwd(w, L.blocks[2], film, T.wa[2].data(), T.wb[2].data(), R2, k.b[2], g_mm, g_d2in, gw, gfilm, k);
  // down 1 -> 2
  float* g_pool1 = k.g2a.data();
  std::fill_n(g_pool1, sz(R2) * sz(R2) * sz(c.c1), 0.f);
  kernels::conv1_back(R2 * R2, k.pool1.data(), c.c1, w + L.down2_w, c.c2, g_d2in, g_pool1, gw + L.down2_w, gw + L.down2_b);
  avgpool_back_add(R1, g_pool1, c.c1, g_s1);
  // block e1
  float* g_d1in = k.g1a.data();
  block_bwd(w, L.blocks[1], film, T.wa[1].data(), T.wb[1].data(), R1, k.b[1], g_s1, g_d1in, gw, gfilm, k);
  // down 0 -> 1
  float* g_pool0 = k.g1b.data();
  std::fill_n(g_pool0, sz(R1) * sz(R1) * sz(c.c0), 0.f);
  kernels::conv1_back(R1 * R1, k.pool0.data(), c.c0, w + L.down1_w, c.c1, g_d1in, g_pool0, gw + L.down1_w, gw + L.down1_b);
  avgpool_back_add(R0, g_pool0, c.c0, g_s0);
  // block e0, then the stem's weights (the input needs no gradient)
  float* g_stem = k.g0a.data();
  block_bwd(w, L.blocks[0], film, T.wa[0].data(), T.wb[0].data(), R0, k.b[0], g_s0, g_stem, gw, gfilm, k);
  kernels::conv3_wgrad(R0, k.xpad.data(), c.inputs(), g_stem, c.c0, gw + L.stem_w, gw + L.stem_b);
  // the MLP
  const int H = c.film_hidden, E = c.embed(), FS = static_cast<int>(L.film_size);
  k.gma.assign(sz(H), 0.f);
  kernels::conv1_back(1, k.ma.data(), H, w + L.mlp2_w, FS, gfilm, k.gma.data(), gw + L.mlp2_w, gw + L.mlp2_b);
  k.gmh = k.gma;
  kernels::silu_grad_mul(k.mh.data(), k.gmh.data(), sz(H));
  kernels::conv1_back(1, k.emb.data(), E, w + L.mlp1_w, H, k.gmh.data(), nullptr, gw + L.mlp1_w, gw + L.mlp1_b);
}

// Loss of one example and (with gw) its gradient times grad_scale.
double loss_and_grad(const Config& c, const Layout& L, const float* w, const Transposed* T, std::span<const float> x0, int t,
                     std::span<const float> eps, std::span<const float> cond, const std::vector<double>& ab, float* gw, float grad_scale,
                     Work& k) {
  const std::size_t n = sz(c.res) * sz(c.res) * sz(c.channels);
  thread_local Vec xt;
  xt.resize(n);
  const float sa = static_cast<float>(std::sqrt(ab[sz(t)])), sn = static_cast<float>(std::sqrt(1.0 - ab[sz(t)]));
  for (std::size_t i = 0; i < n; ++i) xt[i] = sa * x0[i] + sn * eps[i];
  forward(c, L, w, xt.data(), t, cond, k);
  double loss = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(k.y[i]) - static_cast<double>(eps[i]);
    loss += d * d;
  }
  loss /= static_cast<double>(n);
  if (gw) {
    thread_local Vec gy;
    gy.resize(n);
    const float s = 2.f * grad_scale / static_cast<float>(n);
    for (std::size_t i = 0; i < n; ++i) gy[i] = s * (k.y[i] - eps[i]);
    if (T) {
      backward(c, L, w, *T, k, gy.data(), gw);
    } else {
      const Transposed own = transpose_weights(c, L, w);
      backward(c, L, w, own, k, gy.data(), gw);
    }
  }
  return loss;
}

const std::vector<double>& schedule(int T) {
  thread_local int cached = -1;
  thread_local std::vector<double> ab;
  if (cached != T) {
    ab = cosine_alpha_bar(T);
    cached = T;
  }
  return ab;
}

void check_state(const Denoiser& d, std::size_t n, const char* what) {
  if (n != sz(d.cfg.res) * sz(d.cfg.res) * sz(d.cfg.channels)) throw std::invalid_argument(std::string("ddpm: ") + what + ": wrong state size");
  if (d.w.size() != layout(d.cfg).size) throw std::invalid_argument("ddpm: weights do not match the configuration");
}

void clamp_range(const Denoiser& d, std::span<float> x) {
  const int ch = d.cfg.channels;
  if (d.lo.size() != sz(ch) || d.hi.size() != sz(ch)) return;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const std::size_t q = i % sz(ch);
    x[i] = std::clamp(x[i], d.lo[q], d.hi[q]);
  }
}

}  // namespace

// --- configuration and layout -----------------------------------------------------------------------------------------

void Config::validate() const {
  const auto bad = [](const char* what) { throw std::invalid_argument(std::string("ddpm: config: ") + what); };
  if (res < 4 || res % 4 != 0) bad("res is a positive multiple of 4");
  if (channels < 1 || c0 < 1 || c1 < 1 || c2 < 1) bad("channels and widths are positive");
  if (cond < 0 || freqs < 1 || film_hidden < 1) bad("cond >= 0, freqs >= 1, film_hidden >= 1");
  if (timesteps < 10) bad("timesteps >= 10");
}

Layout layout(const Config& c) {
  c.validate();
  Layout L;
  std::size_t o = 0;
  const auto take = [&](std::size_t n) {
    const std::size_t r = o;
    o += n;
    return r;
  };
  const std::size_t I = sz(c.inputs()), C0 = sz(c.c0), C1 = sz(c.c1), C2 = sz(c.c2);
  L.stem_w = take(9 * I * C0);
  L.stem_b = take(C0);
  const int widths[kBlocks] = {c.c0, c.c1, c.c2, c.c2, c.c1, c.c0};
  const int levels[kBlocks] = {0, 1, 2, 2, 1, 0};
  std::size_t film = 0;
  for (int k = 0; k < kBlocks; ++k) {
    BlockLayout& B = L.blocks[sz(k)];
    const std::size_t C = sz(widths[k]);
    B.level = levels[k];
    B.width = widths[k];
    B.wa = take(9 * C * C);
    B.ba = take(C);
    B.wb = take(9 * C * C);
    B.bb = take(C);
    B.film = film;
    film += 2 * C;
  }
  L.down1_w = take(C0 * C1);
  L.down1_b = take(C1);
  L.down2_w = take(C1 * C2);
  L.down2_b = take(C2);
  L.up2_w = take(C2 * C1);
  L.up2_b = take(C1);
  L.up1_w = take(C1 * C0);
  L.up1_b = take(C0);
  L.out_w = take(9 * C0 * sz(c.channels));
  L.out_b = take(sz(c.channels));
  L.film_size = film;
  L.mlp1_w = take(sz(c.embed()) * sz(c.film_hidden));
  L.mlp1_b = take(sz(c.film_hidden));
  L.mlp2_w = take(sz(c.film_hidden) * film);
  L.mlp2_b = take(film);
  L.size = o;
  return L;
}

double forward_macs(const Config& c) {
  const double R0 = c.res, R1 = c.res / 2, R2 = c.res / 4;
  const double n0 = R0 * R0, n1 = R1 * R1, n2 = R2 * R2;
  double m = n0 * 9.0 * c.inputs() * c.c0;                                 // stem
  m += 2.0 * 9.0 * (n0 * c.c0 * c.c0 * 2 + n1 * c.c1 * c.c1 * 2 + n2 * c.c2 * c.c2 * 2);  // six blocks of two convolutions
  m += n1 * c.c0 * c.c1 + n2 * c.c1 * c.c2 + n2 * c.c2 * c.c1 + n1 * c.c1 * c.c0;         // 1 x 1 convolutions
  m += n0 * 9.0 * c.c0 * c.channels;                                        // output
  m += static_cast<double>(c.embed()) * c.film_hidden + static_cast<double>(c.film_hidden) * (4.0 * (c.c0 + c.c1 + c.c2));  // MLP
  return m;
}

std::vector<double> cosine_alpha_bar(int timesteps) {
  constexpr double s = 0.008;
  const auto f = [&](double t) {
    const double v = std::cos(((t / timesteps) + s) / (1.0 + s) * std::numbers::pi / 2.0);
    return v * v;
  };
  std::vector<double> ab(sz(timesteps) + 1);
  ab[0] = 1.0;
  const double f0 = f(0.0);
  for (int t = 1; t <= timesteps; ++t) {
    double v = f(t) / f0;
    const double prev = ab[sz(t) - 1];
    if (1.0 - v / prev > 0.999) v = prev * 0.001;  // beta clip
    ab[sz(t)] = v;
  }
  return ab;
}

void gaussian(std::uint64_t seed, std::span<float> out) {
  std::uint64_t s = seed;
  const auto uni = [&] { return (static_cast<double>(splitmix(s) >> 11) + 1.0) * 0x1.0p-53; };  // (0, 1]
  for (std::size_t i = 0; i < out.size(); i += 2) {
    const double r = std::sqrt(-2.0 * std::log(uni())), a = 2.0 * std::numbers::pi * uni();
    out[i] = static_cast<float>(r * std::cos(a));
    if (i + 1 < out.size()) out[i + 1] = static_cast<float>(r * std::sin(a));
  }
}

Denoiser init_denoiser(const Config& c, std::uint64_t seed) {
  const Layout L = layout(c);
  Denoiser d;
  d.cfg = c;
  d.scale.assign(sz(c.channels), 1.f);
  d.lo.assign(sz(c.channels), -10.f);
  d.hi.assign(sz(c.channels), 10.f);
  d.w.assign(L.size, 0.f);
  std::uint64_t part = 0;
  const auto fill = [&](std::size_t at, std::size_t n, double sd) {
    std::vector<float> g(n);
    gaussian(mix(seed, ++part), g);
    for (std::size_t i = 0; i < n; ++i) d.w[at + i] = static_cast<float>(sd) * g[i];
  };
  const std::size_t I = sz(c.inputs()), C0 = sz(c.c0), C1 = sz(c.c1), C2 = sz(c.c2);
  fill(L.stem_w, 9 * I * C0, std::sqrt(2.0 / (9.0 * static_cast<double>(I))));
  for (const BlockLayout& B : L.blocks) {
    const std::size_t C = sz(B.width);
    fill(B.wa, 9 * C * C, std::sqrt(2.0 / (9.0 * static_cast<double>(C))));  // B.wb stays zero: the block starts as identity
  }
  fill(L.down1_w, C0 * C1, std::sqrt(1.0 / static_cast<double>(C0)));
  fill(L.down2_w, C1 * C2, std::sqrt(1.0 / static_cast<double>(C1)));
  fill(L.up2_w, C2 * C1, std::sqrt(1.0 / static_cast<double>(C2)));
  fill(L.up1_w, C1 * C0, std::sqrt(1.0 / static_cast<double>(C1)));
  fill(L.mlp1_w, sz(c.embed()) * sz(c.film_hidden), std::sqrt(2.0 / static_cast<double>(c.embed())));
  fill(L.mlp2_w, sz(c.film_hidden) * L.film_size, 0.1 / std::sqrt(static_cast<double>(c.film_hidden)));
  // the output layer stays zero: eps_hat = 0 at the start
  return d;
}

// --- serialisation --------------------------------------------------------------------------------------------------

namespace {
constexpr char kDdpmMagic[8] = {'N', 'V', 'F', 'X', 'D', 'D', 'P', 'M'};
constexpr std::uint32_t kDdpmVersion = 1;
}  // namespace

std::string serialise(const Denoiser& d) {
  std::ostringstream o(std::ios::binary);
  o.write(kDdpmMagic, 8);
  bin::put(o, kDdpmVersion);
  const Config& c = d.cfg;
  for (const int v : {c.res, c.channels, c.c0, c.c1, c.c2, c.cond, c.freqs, c.film_hidden, c.timesteps}) bin::put(o, static_cast<std::int32_t>(v));
  for (const std::vector<float>* v : {&d.scale, &d.lo, &d.hi}) {
    if (v->size() != sz(c.channels)) throw std::invalid_argument("ddpm: scale and range need one value per channel");
    for (const float x : *v) bin::put(o, x);
  }
  bin::put(o, static_cast<std::uint64_t>(d.w.size()));
  for (const float x : d.w) bin::put(o, x);
  return o.str();
}

std::expected<Denoiser, std::string> parse(std::string_view bytes) {
  std::istringstream i(std::string(bytes), std::ios::binary);
  char magic[8] = {};
  i.read(magic, 8);
  if (!i || std::memcmp(magic, kDdpmMagic, 8) != 0) return std::unexpected("not a denoiser file (NVFXDDPM)");
  const auto ver = bin::get<std::uint32_t>(i);
  if (!ver || *ver != kDdpmVersion) return std::unexpected("unsupported denoiser version");
  Denoiser d;
  std::array<int, 9> v{};
  for (int& x : v) {
    const auto r = bin::get<std::int32_t>(i);
    if (!r) return std::unexpected(r.error());
    x = *r;
  }
  d.cfg = Config{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]};
  try {
    d.cfg.validate();
  } catch (const std::exception& e) {
    return std::unexpected(e.what());
  }
  for (std::vector<float>* vec : {&d.scale, &d.lo, &d.hi}) {
    vec->resize(sz(d.cfg.channels));
    for (float& x : *vec) {
      const auto r = bin::get<float>(i);
      if (!r) return std::unexpected(r.error());
      x = *r;
    }
  }
  const auto n = bin::get<std::uint64_t>(i);
  if (!n || *n != layout(d.cfg).size) return std::unexpected("weight count does not match the configuration");
  d.w.resize(*n);
  for (float& x : d.w) {
    const auto r = bin::get<float>(i);
    if (!r) return std::unexpected(r.error());
    x = *r;
  }
  return d;
}

std::string version(const Denoiser& d) { return sha256_hex(serialise(d)); }

std::expected<void, std::string> save(const std::filesystem::path& path, const Denoiser& d) {
  const std::string bytes = serialise(d);
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  std::ofstream o(path, std::ios::binary);
  o.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!o) return std::unexpected("cannot write " + path.string());
  return {};
}

std::expected<Denoiser, std::string> load(const std::filesystem::path& path) {
  std::ifstream i(path, std::ios::binary);
  if (!i) return std::unexpected("cannot read " + path.string());
  std::ostringstream ss;
  ss << i.rdbuf();
  return parse(ss.str());
}

// --- inference ---------------------------------------------------------------------------------------------------

void predict_eps(const Denoiser& d, std::span<const float> xt, int t, std::span<const float> cond, std::span<float> eps) {
  check_state(d, xt.size(), "predict_eps");
  if (t < 1 || t > d.cfg.timesteps) throw std::invalid_argument("ddpm: t outside [1, T]");
  const Config& c = d.cfg;
  const Layout L = layout(c);
  Work& k = thread_work(c);
  forward(c, L, d.w.data(), xt.data(), t, cond, k);
  std::copy(k.y.begin(), k.y.end(), eps.begin());
}

void tweedie(const Denoiser& d, std::span<const float> xt, int t, std::span<const float> cond, std::span<float> x0_hat) {
  const auto& ab = schedule(d.cfg.timesteps);
  thread_local Vec eps;
  eps.resize(xt.size());
  predict_eps(d, xt, t, cond, eps);
  const double sa = std::sqrt(ab[sz(t)]), sn = std::sqrt(1.0 - ab[sz(t)]);
  for (std::size_t i = 0; i < xt.size(); ++i) x0_hat[i] = static_cast<float>((static_cast<double>(xt[i]) - sn * eps[i]) / sa);
}

void prior_step(const Denoiser& d, std::span<float> x, int t, float beta, std::span<const float> cond) {
  const auto& ab = schedule(d.cfg.timesteps);
  thread_local Vec xt, eps;
  xt.resize(x.size());
  eps.resize(x.size());
  const float sa = static_cast<float>(std::sqrt(ab[sz(t)]));
  const float r = static_cast<float>(std::sqrt((1.0 - ab[sz(t)]) / ab[sz(t)]));
  for (std::size_t i = 0; i < x.size(); ++i) xt[i] = sa * x[i];
  predict_eps(d, xt, t, cond, eps);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = (1.f - beta) * x[i] + beta * (x[i] - r * eps[i]);
  clamp_range(d, x);
}

int ddim_passes(const Config& c, int t_start, int steps) {
  const int stride = std::max(1, c.timesteps / std::max(1, steps));
  return (t_start + stride - 1) / stride;
}

void ddim(const Denoiser& d, std::span<float> x, int t_start, int steps, std::span<const float> cond) {
  check_state(d, x.size(), "ddim");
  const auto& ab = schedule(d.cfg.timesteps);
  const int stride = std::max(1, d.cfg.timesteps / std::max(1, steps));
  thread_local Vec eps, x0;
  eps.resize(x.size());
  x0.resize(x.size());
  for (int t = std::min(t_start, d.cfg.timesteps); t > 0;) {
    const int tp = std::max(0, t - stride);
    predict_eps(d, x, t, cond, eps);
    const double sa = std::sqrt(ab[sz(t)]), sn = std::sqrt(1.0 - ab[sz(t)]);
    for (std::size_t i = 0; i < x.size(); ++i) x0[i] = static_cast<float>((static_cast<double>(x[i]) - sn * eps[i]) / sa);
    clamp_range(d, x0);
    // the noise that the kept x0 implies, so the step stays on the DDIM path after the clamp
    for (std::size_t i = 0; i < x.size(); ++i) eps[i] = static_cast<float>((static_cast<double>(x[i]) - sa * x0[i]) / sn);
    const double pa = std::sqrt(ab[sz(tp)]), pn = std::sqrt(1.0 - ab[sz(tp)]);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(pa * x0[i] + pn * eps[i]);
    t = tp;
  }
}

void sample(const Denoiser& d, std::span<const float> cond, int steps, std::uint64_t seed, std::span<float> out) {
  gaussian(mix(seed, 0x5A4D), out);
  ddim(d, out, d.cfg.timesteps, steps, cond);
}

void sdedit(const Denoiser& d, std::span<const float> x0, int t0, int steps, std::span<const float> cond, std::uint64_t seed,
            std::span<float> out) {
  check_state(d, x0.size(), "sdedit");
  const int stride = std::max(1, d.cfg.timesteps / std::max(1, steps));
  const int t = std::clamp((t0 + stride / 2) / stride * stride, stride, d.cfg.timesteps);
  const auto& ab = schedule(d.cfg.timesteps);
  gaussian(mix(seed, 0x5DED), out);
  const float sa = static_cast<float>(std::sqrt(ab[sz(t)])), sn = static_cast<float>(std::sqrt(1.0 - ab[sz(t)]));
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = sa * x0[i] + sn * out[i];
  ddim(d, out, t, steps, cond);
}

void fixed_noise(const Config& c, int t, std::uint64_t seed, std::span<float> out) {
  if (out.size() != sz(c.res) * sz(c.res) * sz(c.channels)) throw std::invalid_argument("ddpm: fixed_noise: wrong size");
  gaussian(seed * 1000003ULL + static_cast<std::uint64_t>(t), out);
}

Features features_at(const Denoiser& d, std::span<const float> x0, int t, std::span<const float> cond, std::uint64_t noise_seed) {
  check_state(d, x0.size(), "features_at");
  if (t < 1 || t > d.cfg.timesteps) throw std::invalid_argument("ddpm: t outside [1, T]");
  const auto& ab = schedule(d.cfg.timesteps);
  thread_local Vec xt;
  xt.resize(x0.size());
  fixed_noise(d.cfg, t, noise_seed, xt);
  const float sa = static_cast<float>(std::sqrt(ab[sz(t)])), sn = static_cast<float>(std::sqrt(1.0 - ab[sz(t)]));
  for (std::size_t i = 0; i < xt.size(); ++i) xt[i] = sa * x0[i] + sn * xt[i];
  const Layout L = layout(d.cfg);
  Work& k = thread_work(d.cfg);
  forward(d.cfg, L, d.w.data(), xt.data(), t, cond, k);
  Features f;
  f.r1 = k.R1;
  f.r2 = k.R2;
  f.c1 = d.cfg.c1;
  f.c2 = d.cfg.c2;
  f.enc1 = k.s1;
  f.mid = k.mout;
  f.dec1 = k.dec1;
  return f;
}

double example_loss(const Config& c, std::span<const float> w, std::span<const float> x0, int t, std::span<const float> eps,
                    std::span<const float> cond, std::span<float> grad, float grad_scale) {
  const Layout L = layout(c);
  if (w.size() != L.size || (!grad.empty() && grad.size() != L.size)) throw std::invalid_argument("ddpm: weights or gradient of the wrong size");
  if (x0.size() != sz(c.res) * sz(c.res) * sz(c.channels) || eps.size() != x0.size()) throw std::invalid_argument("ddpm: wrong state size");
  if (t < 1 || t > c.timesteps) throw std::invalid_argument("ddpm: t outside [1, T]");
  Work& k = thread_work(c);
  return loss_and_grad(c, L, w.data(), nullptr, x0, t, eps, cond, schedule(c.timesteps), grad.empty() ? nullptr : grad.data(), grad_scale, k);
}

double kernel_max_error(int res, int ci, int co, std::uint64_t seed) {
  const std::size_t n = sz(res) * sz(res);
  Vec in(n * sz(ci)), W(9 * sz(ci) * sz(co)), b(sz(co)), g(n * sz(co));
  gaussian(mix(seed, 1), in);
  gaussian(mix(seed, 2), W);
  gaussian(mix(seed, 3), b);
  gaussian(mix(seed, 4), g);
  Vec ref(n * sz(co)), fast(n * sz(co)), pad(sz(res + 2) * sz(res + 2) * sz(std::max(ci, co)));
  kernels::conv3_ref(res, in.data(), ci, W.data(), b.data(), co, ref.data());
  kernels::pad(res, in.data(), ci, pad.data());
  kernels::conv3_fwd(res, pad.data(), ci, W.data(), b.data(), co, fast.data());
  double err = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) err = std::max(err, static_cast<double>(std::abs(ref[i] - fast[i])));
  Vec gin_ref(n * sz(ci), 0.f), gW_ref(W.size(), 0.f), gb_ref(b.size(), 0.f), gW(W.size(), 0.f), gb(b.size(), 0.f);
  kernels::conv3_back_ref(res, in.data(), ci, W.data(), co, g.data(), gin_ref.data(), gW_ref.data(), gb_ref.data());
  kernels::conv3_wgrad(res, pad.data(), ci, g.data(), co, gW.data(), gb.data());
  for (std::size_t i = 0; i < gW.size(); ++i) err = std::max(err, static_cast<double>(std::abs(gW[i] - gW_ref[i])));
  for (std::size_t i = 0; i < gb.size(); ++i) err = std::max(err, static_cast<double>(std::abs(gb[i] - gb_ref[i])));
  Vec Wt(W.size()), gin(n * sz(ci));
  kernels::flip_transpose(W.data(), ci, co, Wt.data());
  kernels::pad(res, g.data(), co, pad.data());
  kernels::conv3_fwd(res, pad.data(), co, Wt.data(), nullptr, ci, gin.data());
  for (std::size_t i = 0; i < gin.size(); ++i) err = std::max(err, static_cast<double>(std::abs(gin[i] - gin_ref[i])));
  // SiLU and its derivative against the standard library's exponential (relative error, scaled to count like the rest)
  Vec xs(1001), ys(1001), gs(1001, 1.f);
  for (std::size_t i = 0; i < xs.size(); ++i) xs[i] = -30.f + 0.06f * static_cast<float>(i);
  kernels::silu_n(xs.data(), ys.data(), xs.size());
  kernels::silu_grad_mul(xs.data(), gs.data(), xs.size());
  for (std::size_t i = 0; i < xs.size(); ++i) {
    const double x = xs[i], s = 1.0 / (1.0 + std::exp(-x));
    err = std::max(err, 1e3 * std::abs(ys[i] - x * s) / std::max(1e-3, std::abs(x * s)));
    err = std::max(err, 1e3 * std::abs(gs[i] - s * (1.0 + x * (1.0 - s))) / std::max(1e-3, std::abs(s * (1.0 + x * (1.0 - s)))));
  }
  return err;
}

// --- training -------------------------------------------------------------------------------------------------------

namespace {

struct Adam {
  Vec m, v;
  int t = 0;
  explicit Adam(std::size_t n) : m(n, 0.f), v(n, 0.f) {}
  void step(Vec& w, const Vec& g, float lr) {
    ++t;
    constexpr float b1 = 0.9f, b2 = 0.999f;
    const float c1 = 1.f - std::pow(b1, static_cast<float>(t)), c2 = 1.f - std::pow(b2, static_cast<float>(t));
    for (std::size_t j = 0; j < w.size(); ++j) {
      m[j] = b1 * m[j] + (1.f - b1) * g[j];
      v[j] = b2 * v[j] + (1.f - b2) * g[j] * g[j];
      w[j] -= lr * (m[j] / c1) / (std::sqrt(v[j] / c2) + 1e-8f);
    }
  }
};

// Runs fn(i) for i in [0, n) on `threads` threads, items handed out in order.
template <class Fn>
void parallel_for(int n, int threads, Fn&& fn) {
  if (threads <= 1 || n <= 1) {
    for (int i = 0; i < n; ++i) fn(i);
    return;
  }
  std::atomic<int> next{0};
  std::vector<std::jthread> pool;
  for (int t = 0; t < std::min(threads, n); ++t) {
    pool.emplace_back([&] {
      for (int i; (i = next++) < n;) fn(i);
    });
  }
}

}  // namespace

double eval_loss(const Denoiser& d, const Dataset& data, int t, std::uint64_t seed, int count, int threads) {
  const int n = count > 0 ? std::min(count, data.count) : data.count;
  if (n <= 0) return 0.0;
  std::vector<double> loss(sz(n), 0.0);
  const std::size_t vals = sz(data.values);
  const Layout L = layout(d.cfg);
  parallel_for(n, threads, [&](int i) {
    thread_local Vec eps;
    eps.resize(vals);
    gaussian(mix(seed, static_cast<std::uint64_t>(t), static_cast<std::uint64_t>(i)), eps);
    Work& k = thread_work(d.cfg);
    loss[sz(i)] = loss_and_grad(d.cfg, L, d.w.data(), nullptr, data.state(i), t, eps, data.condition(i), schedule(d.cfg.timesteps), nullptr, 1.f, k);
  });
  double s = 0;
  for (const double v : loss) s += v;
  return s / static_cast<double>(n);
}

TrainResult train(Denoiser& d, const Dataset& data, const TrainOptions& o, const Dataset* eval) {
  const Config& c = d.cfg;
  const Layout L = layout(c);
  const std::size_t vals = sz(c.res) * sz(c.res) * sz(c.channels);
  if (data.count < 1 || sz(data.values) != vals || data.conds != c.cond) throw std::invalid_argument("ddpm: data set does not match the configuration");
  if (d.w.size() != L.size) throw std::invalid_argument("ddpm: weights do not match the configuration");
  if (o.steps < 1 || o.batch < 1) throw std::invalid_argument("ddpm: steps and batch must be positive");
  Vec w = d.w, ema = d.w;
  Adam adam(L.size);
  // The batch is cut into fixed chunks; each chunk's gradient is summed in sample order and the chunks in chunk order,
  // so the result does not depend on the thread count.
  const int chunks = std::min(o.batch, 8);
  std::vector<Vec> grads(sz(chunks), Vec(L.size, 0.f));
  Vec g(L.size);
  std::vector<double> losses(sz(o.batch));
  TrainResult res;
  const auto t0 = std::chrono::steady_clock::now();
  double window = 0;
  int in_window = 0;
  const float inv_batch = 1.f / static_cast<float>(o.batch);
  for (int step = 1; step <= o.steps; ++step) {
    float lr = o.lr;
    if (step <= o.warmup) {
      lr = o.lr * static_cast<float>(step) / static_cast<float>(std::max(1, o.warmup));
    } else {
      const double p = static_cast<double>(step - o.warmup) / static_cast<double>(std::max(1, o.steps - o.warmup));
      lr = static_cast<float>(o.lr * (0.1 + 0.9 * 0.5 * (1.0 + std::cos(std::numbers::pi * p))));
    }
    const Transposed T = transpose_weights(c, L, w.data());
    parallel_for(chunks, o.threads, [&](int ch) {
      Vec& gc = grads[sz(ch)];
      std::fill(gc.begin(), gc.end(), 0.f);
      thread_local Vec eps;
      eps.resize(vals);
      Work& k = thread_work(c);
      for (int b = ch * o.batch / chunks; b < (ch + 1) * o.batch / chunks; ++b) {
        std::uint64_t s = mix(o.seed, static_cast<std::uint64_t>(step), static_cast<std::uint64_t>(b));
        const int idx = static_cast<int>(splitmix(s) % static_cast<std::uint64_t>(data.count));
        const int t = 1 + static_cast<int>(splitmix(s) % static_cast<std::uint64_t>(c.timesteps));
        gaussian(splitmix(s), eps);
        losses[sz(b)] = loss_and_grad(c, L, w.data(), &T, data.state(idx), t, eps, data.condition(idx), schedule(c.timesteps), gc.data(),
                                      inv_batch, k);
      }
    });
    std::copy(grads[0].begin(), grads[0].end(), g.begin());
    for (int ch = 1; ch < chunks; ++ch) {
      const Vec& gc = grads[sz(ch)];
      for (std::size_t j = 0; j < L.size; ++j) g[j] += gc[j];
    }
    double gn = 0;
    for (const float v : g) gn += static_cast<double>(v) * v;
    gn = std::sqrt(gn);
    if (o.clip > 0 && gn > o.clip) {
      const float s = static_cast<float>(o.clip / gn);
      for (float& v : g) v *= s;
    }
    adam.step(w, g, lr);
    const float decay = step < 500 ? std::min(o.ema, 0.99f) : o.ema;
    for (std::size_t j = 0; j < L.size; ++j) ema[j] = decay * ema[j] + (1.f - decay) * w[j];
    double loss = 0;
    for (const double v : losses) loss += v;
    window += loss / o.batch;
    ++in_window;
    if (step % std::max(1, o.log_every) == 0 || step == o.steps) {
      TrainLog log;
      log.step = step;
      log.loss = window / in_window;
      log.lr = lr;
      log.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      Denoiser e = d;
      e.w = ema;
      if (eval && eval->count > 0) {
        for (const int t : o.eval_t) log.eval.push_back(eval_loss(e, *eval, t, o.eval_seed, o.eval_count, o.threads));
      }
      res.curve.push_back(log);
      if (o.progress) o.progress(log, e);
      window = 0;
      in_window = 0;
    }
  }
  d.w = ema;
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return res;
}

// --- coarse states ----------------------------------------------------------------------------------------------------

void to_network(const Denoiser& d, std::span<const float> phys, int stride, std::span<float> x) {
  const int ch = d.cfg.channels, N = d.cfg.res * d.cfg.res;
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < ch; ++k) x[sz(i) * sz(ch) + sz(k)] = phys[sz(i) * sz(stride) + sz(k)] / d.scale[sz(k)];
  }
}

void to_physical(const Denoiser& d, std::span<const float> x, int stride, std::span<float> phys) {
  const int ch = d.cfg.channels, N = d.cfg.res * d.cfg.res;
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < ch; ++k) phys[sz(i) * sz(stride) + sz(k)] = x[sz(i) * sz(ch) + sz(k)] * d.scale[sz(k)];
  }
}

Dataset states_from_runs(const rollout::Model& m, std::span<const rollout::Run> runs, int first, int every) {
  Dataset ds;
  const int R = m.h.res;
  ds.values = R * R * rollout::kPhys;
  ds.conds = m.h.cond();
  const std::size_t per = sz(ds.values);
  std::vector<float> cnd(sz(ds.conds));
  for (const rollout::Run& r : runs) {
    if (r.coarse.size() != sz(r.frames) * per) throw std::invalid_argument("ddpm: run has the wrong shape");
    const std::vector<float> controls = r.controls();
    for (int i = first; i < r.frames; i += std::max(1, every)) {
      const float* s = r.coarse.data() + sz(i) * per;
      for (std::size_t j = 0; j < per; ++j) ds.x.push_back(s[j] / m.scale[j % rollout::kPhys]);
      rollout::condition(m, controls, static_cast<float>(i + 1) / r.p.fps, cnd);
      ds.cond.insert(ds.cond.end(), cnd.begin(), cnd.end());
      ++ds.count;
    }
  }
  return ds;
}

void set_range(Denoiser& d, const Dataset& data) {
  const int ch = d.cfg.channels;
  std::vector<float> lo(sz(ch), std::numeric_limits<float>::infinity()), hi(sz(ch), -std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < data.x.size(); ++i) {
    const std::size_t k = i % sz(ch);
    lo[k] = std::min(lo[k], data.x[i]);
    hi[k] = std::max(hi[k], data.x[i]);
  }
  for (int k = 0; k < ch; ++k) {
    const float span = hi[sz(k)] - lo[sz(k)];
    d.lo[sz(k)] = lo[sz(k)] - 0.1f * span;
    d.hi[sz(k)] = hi[sz(k)] + 0.1f * span;
  }
}

std::expected<void, std::string> save_dataset(const std::filesystem::path& path, const Dataset& data) {
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  std::ofstream o(path, std::ios::binary);
  o.write("NVFXGST1", 8);
  bin::put(o, static_cast<std::int32_t>(data.count));
  bin::put(o, static_cast<std::int32_t>(data.values));
  bin::put(o, static_cast<std::int32_t>(data.conds));
  bin::put_array<float>(o, data.x);
  bin::put_array<float>(o, data.cond);
  if (!o) return std::unexpected("cannot write " + path.string());
  return {};
}

std::expected<Dataset, std::string> load_dataset(const std::filesystem::path& path) {
  std::ifstream i(path, std::ios::binary);
  if (!i) return std::unexpected("cannot read " + path.string());
  char magic[8] = {};
  i.read(magic, 8);
  if (!i || std::memcmp(magic, "NVFXGST1", 8) != 0) return std::unexpected(path.string() + " is not a state set (NVFXGST1)");
  Dataset d;
  for (int* v : {&d.count, &d.values, &d.conds}) {
    const auto r = bin::get<std::int32_t>(i);
    if (!r || *r < 0) return std::unexpected("truncated or damaged state set");
    *v = *r;
  }
  d.x.resize(sz(d.count) * sz(d.values));
  d.cond.resize(sz(d.count) * sz(d.conds));
  i.read(reinterpret_cast<char*>(d.x.data()), static_cast<std::streamsize>(d.x.size() * sizeof(float)));
  i.read(reinterpret_cast<char*>(d.cond.data()), static_cast<std::streamsize>(d.cond.size() * sizeof(float)));
  if (!i) return std::unexpected("truncated state set");
  static_assert(std::endian::native == std::endian::little, "state sets are read as little-endian floats");
  return d;
}

// --- contexts -------------------------------------------------------------------------------------------------------

namespace {

// Mean of the cells of level map f (r x r x C) that fall in region (rx, ry) of kRegions x kRegions, appended to out.
void region_mean(const std::vector<float>& f, int r, int C, int rx, int ry, std::vector<double>& out) {
  const int span = std::max(1, r / kRegions);
  const int x0 = rx * r / kRegions, y0 = ry * r / kRegions;
  const std::size_t at = out.size();
  out.resize(at + sz(C), 0.0);
  for (int y = y0; y < y0 + span; ++y) {
    for (int x = x0; x < x0 + span; ++x) {
      for (int k = 0; k < C; ++k) out[at + sz(k)] += f[(sz(y) * sz(r) + sz(x)) * sz(C) + sz(k)];
    }
  }
  for (int k = 0; k < C; ++k) out[at + sz(k)] /= static_cast<double>(span * span);
}

}  // namespace

std::vector<double> region_features(const Denoiser* d, const ContextSpec& s, std::span<const float> x, std::span<const float> cond, int res,
                                    int channels) {
  constexpr int NR = kRegions * kRegions;
  std::vector<std::vector<double>> rows(NR);
  if (s.diffusion) {
    if (!d) throw std::invalid_argument("ddpm: diffusion contexts need a denoiser");
    for (const int t : s.ts) {
      const Features f = features_at(*d, x, t, cond, s.noise_seed);
      for (int ry = 0; ry < kRegions; ++ry) {
        for (int rx = 0; rx < kRegions; ++rx) {
          auto& row = rows[sz(ry * kRegions + rx)];
          region_mean(f.enc1, f.r1, f.c1, rx, ry, row);
          region_mean(f.mid, f.r2, f.c2, rx, ry, row);
          region_mean(f.dec1, f.r1, f.c1, rx, ry, row);
        }
      }
    }
  } else {  // plain coarse statistics: the region's cells, every channel
    const std::vector<float> v(x.begin(), x.end());
    const int span = std::max(1, res / kRegions);
    for (int ry = 0; ry < kRegions; ++ry) {
      for (int rx = 0; rx < kRegions; ++rx) {
        auto& row = rows[sz(ry * kRegions + rx)];
        const int x0 = rx * res / kRegions, y0 = ry * res / kRegions;
        for (int y = y0; y < y0 + span; ++y) {
          for (int xx = x0; xx < x0 + span; ++xx) {
            for (int k = 0; k < channels; ++k) row.push_back(v[(sz(y) * sz(res) + sz(xx)) * sz(channels) + sz(k)]);
          }
        }
      }
    }
  }
  std::vector<double> out;
  out.reserve(rows.size() * rows[0].size());
  for (const auto& r : rows) out.insert(out.end(), r.begin(), r.end());
  return out;
}

ContextModel fit_contexts(const Denoiser* d, const ContextSpec& s, const Dataset& data, std::span<const int> states, int res, int channels,
                          int threads) {
  constexpr int NR = kRegions * kRegions;
  const int n = static_cast<int>(states.size());
  if (n < 1) throw std::invalid_argument("ddpm: fit_contexts needs states");
  std::vector<std::vector<double>> per(sz(n));
  parallel_for(n, threads, [&](int i) {
    const int st = states[sz(i)];
    per[sz(i)] = region_features(d, s, data.state(st), data.condition(st), res, channels);
  });
  const std::size_t D = per[0].size() / NR;
  std::vector<double> X;
  X.reserve(sz(n) * NR * D);
  for (const auto& p : per) X.insert(X.end(), p.begin(), p.end());
  per.clear();
  ContextModel cm;
  cm.spec = s;
  cm.mean.assign(D, 0.0);
  cm.sd.assign(D, 0.0);
  const std::size_t rows = X.size() / D;
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t j = 0; j < D; ++j) cm.mean[j] += X[r * D + j];
  }
  for (double& m : cm.mean) m /= static_cast<double>(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t j = 0; j < D; ++j) {
      const double e = X[r * D + j] - cm.mean[j];
      cm.sd[j] += e * e;
    }
  }
  for (double& v : cm.sd) v = std::max(1e-6, std::sqrt(v / static_cast<double>(rows)));
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t j = 0; j < D; ++j) X[r * D + j] = (X[r * D + j] - cm.mean[j]) / cm.sd[j];
  }
  cm.pca = fit_pca(X, D, s.pca);
  const std::vector<double> Y = cm.pca.project(X, D);
  X.clear();
  for (const int k : s.ks) {
    KMeansOptions ko;
    ko.k = k;
    ko.restarts = 4;
    ko.max_iter = 60;
    ko.seed = s.seed;
    cm.km.push_back(fit_kmeans(Y, cm.pca.count(), ko));
  }
  return cm;
}

std::vector<std::vector<int>> context_planes(const Denoiser* d, const ContextModel& cm, std::span<const float> coarse, int stride,
                                             std::span<const float> scale, std::span<const float> cond, int res) {
  const int channels = static_cast<int>(scale.size());
  std::vector<float> x(sz(res) * sz(res) * sz(channels));
  for (int i = 0; i < res * res; ++i) {
    for (int k = 0; k < channels; ++k) x[sz(i) * sz(channels) + sz(k)] = coarse[sz(i) * sz(stride) + sz(k)] / scale[sz(k)];
  }
  std::vector<double> X = region_features(d, cm.spec, x, cond, res, channels);
  const std::size_t D = cm.mean.size();
  if (X.size() != D * kRegions * kRegions) throw std::invalid_argument("ddpm: context model does not match the features");
  for (std::size_t r = 0; r < X.size() / D; ++r) {
    for (std::size_t j = 0; j < D; ++j) X[r * D + j] = (X[r * D + j] - cm.mean[j]) / cm.sd[j];
  }
  const std::vector<double> Y = cm.pca.project(X, D);
  std::vector<std::vector<int>> out;
  for (const KMeans& km : cm.km) out.push_back(km.assign(Y, cm.pca.count()));
  return out;
}

HandMade fit_handmade(const Dataset& data, std::span<const int> states, int res, int channels) {
  std::vector<double> heat, speed;
  const int span = std::max(1, res / kRegions);
  for (const int st : states) {
    const auto x = data.state(st);
    for (int ry = 0; ry < kRegions; ++ry) {
      for (int rx = 0; rx < kRegions; ++rx) {
        double h = 0, sp = 0;
        for (int y = ry * span; y < (ry + 1) * span; ++y) {
          for (int xx = rx * span; xx < (rx + 1) * span; ++xx) {
            const float* v = x.data() + (sz(y) * sz(res) + sz(xx)) * sz(channels);
            h += v[2];
            sp += std::sqrt(static_cast<double>(v[0]) * v[0] + static_cast<double>(v[1]) * v[1]);
          }
        }
        heat.push_back(h / (span * span));
        speed.push_back(sp / (span * span));
      }
    }
  }
  HandMade hm;
  hm.heat_empty = 0.05;
  std::vector<double> full;
  for (const double h : heat) {
    if (h >= hm.heat_empty) full.push_back(h);
  }
  const auto quant = [](std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::ranges::sort(v);
    return v[std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size())))];
  };
  hm.heat_t1 = quant(full, 1.0 / 3.0);
  hm.heat_t2 = quant(full, 2.0 / 3.0);
  hm.flow_t1 = quant(speed, 1.0 / 3.0);
  hm.flow_t2 = quant(speed, 2.0 / 3.0);
  return hm;
}

std::vector<std::array<int, HandMade::kKinds>> handmade_contexts(const HandMade& h, std::span<const float> x, std::span<const float> controls,
                                                                 int res, int channels) {
  std::vector<std::array<int, HandMade::kKinds>> out(sz(kRegions) * kRegions);
  const int span = std::max(1, res / kRegions);
  int cbin = 0;
  for (std::size_t k = 0; k < 3 && k < controls.size(); ++k) cbin |= (controls[k] >= 0.5f ? 1 : 0) << k;
  for (int ry = 0; ry < kRegions; ++ry) {
    for (int rx = 0; rx < kRegions; ++rx) {
      double heat = 0, sp = 0;
      for (int y = ry * span; y < (ry + 1) * span; ++y) {
        for (int xx = rx * span; xx < (rx + 1) * span; ++xx) {
          const float* v = x.data() + (sz(y) * sz(res) + sz(xx)) * sz(channels);
          heat += v[2];
          sp += std::sqrt(static_cast<double>(v[0]) * v[0] + static_cast<double>(v[1]) * v[1]);
        }
      }
      heat /= span * span;
      sp /= span * span;
      auto& o = out[sz(ry * kRegions + rx)];
      o[0] = heat < h.heat_empty ? 0 : heat < h.heat_t1 ? 1 : heat < h.heat_t2 ? 2 : 3;
      o[1] = std::min(3, ry * 4 / kRegions);
      o[2] = sp < h.flow_t1 ? 0 : sp < h.flow_t2 ? 1 : 2;
      o[3] = cbin;
    }
  }
  return out;
}

}  // namespace nfx::dcm::ddpm

#ifdef NFX_DDPM_PUSHED
#pragma GCC pop_options
#endif
